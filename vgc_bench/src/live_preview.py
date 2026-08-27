"""Bounded exact Team Preview planning for the fixed Reg M-B team.

The move planner cannot rescue a structurally bad bring-four.  This module uses the
replay-trained preview model only as a broad candidate generator, then lets exact
Showdown simulation rank those candidates through the first move turn.  It is kept
separate from ``LiveExactSession`` because no selected four or live shadow exists yet
at Team Preview.

Two repairs over the original single-world planner:

1. **Multi-determinization.**  One sampled hidden-set world made every ranking
   hostage to that sample's items/spreads.  The planner now runs up to eight
   mass-weighted worlds sequentially and merges their rankings with the same
   60/30/10 risk blend the move planner uses.  Only worlds whose search completed
   cleanly count; a truncated world is dropped rather than letting its partial
   rankings vote (partial trees are how the 20260823 re-gate silently measured
   nothing).  Budget note: the VGC Timer grants 90 seconds at Team Preview against
   a 420-second bank, so preview can legitimately spend far more than the 8-second
   move-turn budget this planner inherited.
2. **Champion-pick injection.**  The candidate list was the preview predictor's
   top ``root_width`` plans; the champion policy's own pick was invisible to the
   search whenever the predictor ranked it lower.  The champion's plan is now
   guaranteed a candidate slot in every world, so the search can only override the
   champion after actually evaluating the champion's plan.
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from typing import Any

from poke_env.battle import DoubleBattle
from poke_env.data import to_id_str

from vgc_bench.src.exact_observation import (
    ExactPolicyAdapter,
    OpponentModelPrior,
    RankedChoice,
)
from vgc_bench.src.exact_planner import (
    ExactMultiTurnPlanner,
    ExactNode,
    PlannerConfig,
    PlanResult,
    Prior,
    aggregate_plans,
)
from vgc_bench.src.exact_sim import ExactShowdownBridge, ExactSimulatorError
from vgc_bench.src.live_exact import _particle_mass, _roster_from_battle
from vgc_bench.src.set_particles import (
    ParticleDatabase,
    TeamBelief,
    determination_team_text,
)

_ROOT_WIDTH = 12
_MAX_WORLD_SLICE_S = 60.0
_MIN_WORLD_SLICE_S = 0.75


@dataclass(frozen=True)
class PreviewDecision:
    showdown_choice: str
    command: str
    elapsed_s: float
    nodes: int
    truncated: bool
    score: float
    rankings: tuple[dict[str, Any], ...]
    open_sheet: bool
    worlds_requested: int = 1
    worlds_clean: int = 1
    worlds_failed: int = 0
    champion_choice: str | None = None
    champion_rank: int | None = None
    override_margin: float | None = None


def _command(choice: str) -> str:
    parts = choice.removeprefix("team ").replace(",", " ").split()
    if len(parts) != 4:
        raise ValueError(f"exact preview returned malformed choice: {choice!r}")
    indexes = [int(part) for part in parts]
    if len(set(indexes)) != 4 or any(index < 1 or index > 6 for index in indexes):
        raise ValueError(f"exact preview returned illegal team order: {choice!r}")
    return "/team " + "".join(str(index) for index in indexes)


def _team_choice_key(choice: str) -> str:
    """Normalize a Showdown team-order choice for equality comparison."""
    return "team " + ",".join(choice.removeprefix("team ").replace(",", " ").split())


class ChampionInjectedPrior:
    """Guarantee the champion policy's preview plan survives root truncation.

    Wraps the planner prior; only the controlled side's Team Preview ranking is
    touched.  If the champion's plan is missing from the top ``width`` rows it is
    moved (or inserted) to the front of the list, which is the slice
    ``_diverse_prefix`` keeps for ``team`` choices.  Candidate order never affects
    exact scores -- only which candidates get simulated at all.
    """

    def __init__(
        self,
        base: Prior,
        controlled_role: str,
        champion_order: tuple[int, int, int, int],
        width: int = _ROOT_WIDTH,
    ):
        self.base = base
        self.controlled_role = controlled_role
        self.champion_key = "team " + ",".join(str(index) for index in champion_order)
        self.width = int(width)
        self.injections = 0
        self.promotions = 0

    def rank(
        self,
        state: dict[str, Any],
        requests: list[dict[str, Any] | None],
        role: str,
        choices: list[str],
    ) -> list[RankedChoice]:
        ranked = self.base.rank(state, requests, role, choices)
        if (
            role != self.controlled_role
            or not choices
            or not choices[0].startswith("team ")
        ):
            return ranked
        position = next(
            (
                index
                for index, item in enumerate(ranked)
                if _team_choice_key(item.choice) == self.champion_key
            ),
            None,
        )
        if position is not None and position < self.width:
            return ranked
        if position is not None:
            item = ranked[position]
            self.promotions += 1
            return [item] + ranked[:position] + ranked[position + 1 :]
        legal = next(
            (
                choice
                for choice in choices
                if _team_choice_key(choice) == self.champion_key
            ),
            None,
        )
        if legal is None:
            return ranked
        floor = min((item.probability for item in ranked), default=1.0)
        self.injections += 1
        return [RankedChoice(legal, None, max(1e-4, floor))] + list(ranked)


class LivePreviewPlanner:
    """Rank matchup-aware bring/lead candidates inside the opening clock."""

    def __init__(
        self,
        *,
        policy,
        our_team_text: str,
        outcome_evaluator,
        preview_predictor,
        open_sheet: bool,
        budget_s: float = 8.0,
        determinizations: int = 1,
        champion_order: tuple[int, int, int, int] | None = None,
        seed: int = 20260822,
        policy_inference_lock=None,
    ):
        # The VGC Timer allows 90 seconds at Team Preview (420-second bank), so the
        # preview budget may exceed the 9-second per-move-turn ceiling. Each world
        # still gets at most 9 seconds so PlannerConfig's invariant holds.
        if not 0.1 <= budget_s <= 60.0:
            raise ValueError("preview search budget must be within [0.1, 60] seconds")
        if not 1 <= determinizations <= 8:
            raise ValueError("preview determinizations must be within [1, 8]")
        if champion_order is not None:
            order = tuple(int(index) for index in champion_order)
            if len(order) != 4 or len(set(order)) != 4 or not all(
                1 <= index <= 6 for index in order
            ):
                raise ValueError(
                    f"malformed champion preview order: {champion_order!r}"
                )
            champion_order = order
        self.policy = policy
        self.our_team_text = our_team_text
        self.outcome_evaluator = outcome_evaluator
        self.preview_predictor = preview_predictor
        self.open_sheet = bool(open_sheet)
        self.budget_s = float(budget_s)
        # A sheet reveals every set, so extra hidden-set worlds would be identical.
        self.determinizations = 1 if self.open_sheet else int(determinizations)
        self.champion_order = champion_order
        self.rng = random.Random(seed)
        self.policy_inference_lock = policy_inference_lock
        self.database = ParticleDatabase.load(max_particles=12)

    def _opponent_worlds(self, battle: DoubleBattle) -> list[tuple[str, float, str]]:
        roster = _roster_from_battle(battle, own=False)
        if len(roster) != 6:
            raise ValueError(f"preview opponent roster has {len(roster)} Pokemon")
        belief = TeamBelief.from_roster(
            self.database, (slot.species for slot in roster)
        )
        if self.open_sheet:
            by_species = {
                to_id_str(mon.base_species or mon.species): mon
                for mon in battle.opponent_team.values()
            }
            for species, mon in by_species.items():
                belief.condition(
                    species,
                    moves=mon.moves.keys(),
                    item=mon.item,
                    ability=mon.ability,
                )
        determinations = belief.sample_determinizations(
            self.determinizations, self.rng, open_sheet=self.open_sheet
        )
        return [
            (
                determination_team_text(roster, determination),
                _particle_mass(determination),
                f"preview-{index + 1}",
            )
            for index, determination in enumerate(determinations)
        ]

    def _world_config(self, slice_s: float) -> PlannerConfig:
        return PlannerConfig(
            # Preview itself does not consume a move depth, so depth one resolves the
            # selected leads and then searches their first simultaneous move turn.
            depth=1,
            root_width=_ROOT_WIDTH,
            opponent_width=6,
            continuation_width=1,
            replacement_width=1,
            chance_samples=1,
            expected_weight=0.60,
            cvar_weight=0.30,
            worst_weight=0.10,
            time_budget_s=slice_s,
            screen_budget_s=min(2.0, slice_s),
            max_nodes=2500,
            anytime=False,
        )

    def choose(self, battle: DoubleBattle) -> PreviewDecision:
        started = time.monotonic()
        adapter = ExactPolicyAdapter(
            self.policy,
            preview_predictor=self.preview_predictor,
            reveal_opponent_sets=self.open_sheet,
            inference_lock=self.policy_inference_lock,
        )
        prior: Prior = OpponentModelPrior(adapter, controlled_role="p1")
        champion_key = None
        if self.champion_order is not None:
            prior = ChampionInjectedPrior(prior, "p1", self.champion_order)
            champion_key = prior.champion_key
        worlds = self._opponent_worlds(battle)
        deadline = started + self.budget_s
        results: list[tuple[float, PlanResult]] = []
        failed = 0
        nodes = 0
        config = self._world_config(min(_MAX_WORLD_SLICE_S, self.budget_s))
        with ExactShowdownBridge() as bridge:
            for index, (team_text, mass, _label) in enumerate(worlds):
                remaining = deadline - time.monotonic()
                if remaining < _MIN_WORLD_SLICE_S and results:
                    break
                slice_s = min(
                    _MAX_WORLD_SLICE_S,
                    max(0.5, remaining / (len(worlds) - index)),
                )
                config = self._world_config(slice_s)
                try:
                    root = bridge.create(
                        formatid="gen9championsvgc2026regmb",
                        seed=[self.rng.randrange(1, 65536) for _ in range(4)],
                        p1_name="planner",
                        p2_name="opponent",
                        p1_team_text=self.our_team_text,
                        p2_team_text=team_text,
                    )
                    node = ExactNode.from_result(root)
                    result = ExactMultiTurnPlanner(
                        bridge, prior, self.outcome_evaluator, config
                    ).plan(node, "p1")
                except (ExactSimulatorError, ValueError):
                    failed += 1
                    continue
                nodes += result.nodes
                if result.truncated:
                    # A partial tree ranks whatever it happened to simulate first.
                    # Dropping it keeps every vote a complete comparison.
                    failed += 1
                    continue
                results.append((mass, result))
        return self._decide(
            results,
            worlds_requested=len(worlds),
            worlds_failed=failed,
            started=started,
            nodes=nodes,
            config=config,
            champion_key=champion_key,
        )

    def _decide(
        self,
        results: list[tuple[float, PlanResult]],
        *,
        worlds_requested: int,
        worlds_failed: int,
        started: float,
        nodes: int,
        config: PlannerConfig,
        champion_key: str | None,
    ) -> PreviewDecision:
        elapsed = time.monotonic() - started
        if not results:
            # No world completed. The caller treats a truncated decision as
            # "stand down": the champion's own preview path plays instead.
            return PreviewDecision(
                showdown_choice="",
                command="",
                elapsed_s=elapsed,
                nodes=nodes,
                truncated=True,
                score=-1.0,
                rankings=(),
                open_sheet=self.open_sheet,
                worlds_requested=worlds_requested,
                worlds_clean=0,
                worlds_failed=worlds_failed,
                champion_choice=champion_key,
            )
        aggregate = aggregate_plans(results, elapsed, config=config)
        champion_rank = None
        override_margin = None
        if champion_key is not None:
            champion_row = next(
                (
                    (position, row)
                    for position, row in enumerate(aggregate.rankings)
                    if _team_choice_key(row.choice) == champion_key
                ),
                None,
            )
            if champion_row is not None:
                champion_rank = champion_row[0] + 1
                override_margin = float(aggregate.score - champion_row[1].score)
        return PreviewDecision(
            showdown_choice=aggregate.choice,
            command=_command(aggregate.choice),
            elapsed_s=elapsed,
            nodes=nodes,
            truncated=False,
            score=aggregate.score,
            rankings=tuple(asdict(row) for row in aggregate.rankings[:12]),
            open_sheet=self.open_sheet,
            worlds_requested=worlds_requested,
            worlds_clean=len(results),
            worlds_failed=worlds_failed,
            champion_choice=champion_key,
            champion_rank=champion_rank,
            override_margin=override_margin,
        )
