"""Live exact-search session built from public Showdown state.

The remote server never exposes its serialized simulator state.  A session therefore
keeps a bounded set of concrete local worlds, advances them with the actions visible
in the protocol, and repairs RNG-dependent public state before every search.  Hidden
sets and unrevealed back Pokemon remain explicit uncertainty rather than being copied
from a private simulator state.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from poke_env.battle import DoubleBattle, Pokemon
from poke_env.data import to_id_str
from poke_env.environment import DoublesEnv

from vgc_bench.src.exact_observation import (
    ExactPolicyAdapter,
    OpponentModelPrior,
    choice_to_actions,
)
from vgc_bench.src.exact_planner import (
    ActionScore,
    BranchContinuation,
    BranchOutcome,
    ExactDeterminizationPlanner,
    ExactNode,
    PlannerConfig,
    PlanResult,
    WeightedExactNode,
)
from vgc_bench.src.exact_sim import ExactShowdownBridge
from vgc_bench.src.live_snapshot import public_snapshot
from vgc_bench.src.outcome_value import OutcomeValueEvaluator
from vgc_bench.src.ponder import (
    BackgroundPonder,
    PonderConfig,
    PonderOutcome,
    PonderRoot,
)
from vgc_bench.src.set_particles import (
    ParticleDatabase,
    SetParticle,
    TeamBelief,
    TeamSlot,
    determination_team_text,
    team_roster,
)


@dataclass(frozen=True)
class LiveRoot:
    node: ExactNode
    probability: float
    determination: dict[str, SetParticle]
    brought: tuple[str, ...]
    label: str


@dataclass(frozen=True)
class ObservedAction:
    kind: str
    identifier: str
    target: int | None = None
    mega: bool = False


def _species(mon: Pokemon | None) -> str:
    return to_id_str(mon.species) if mon is not None else ""


def _actor_slot(actor: str, role: str) -> int | None:
    prefix = f"{role}"
    if not actor.startswith(prefix):
        return None
    position = actor.split(":", 1)[0]
    if position.endswith("a"):
        return 0
    if position.endswith("b"):
        return 1
    return None


def _target_location(target: str, role: str, actor_slot: int) -> int | None:
    if not target or ":" not in target:
        return None
    position = target.split(":", 1)[0]
    target_role = position[:2]
    target_slot = 0 if position.endswith("a") else 1 if position.endswith("b") else -1
    if target_slot < 0:
        return None
    if target_role != role:
        # Showdown numbers opposing targets from the submitting player's camera.
        # Player 2's camera is mirrored: p1b is +1 and p1a is +2.
        if role == "p2":
            return 2 - target_slot
        return target_slot + 1
    if target_slot == actor_slot:
        return None
    return -(target_slot + 1)


def observed_opponent_actions(
    events: Sequence[Sequence[str]], role: str = "p2"
) -> dict[int, ObservedAction]:
    """Recover the opponent's submitted turn atoms from public protocol events.

    Voluntary switches happen before every move.  Switches after the first move are
    pivots or forced replacements and are deliberately excluded from the original
    simultaneous choice.
    """
    actions: dict[int, ObservedAction] = {}
    mega_slots: set[int] = set()
    move_started = False
    for event in events:
        if len(event) < 2:
            continue
        event_type = event[1]
        actor = event[2] if len(event) > 2 else ""
        slot = _actor_slot(actor, role)
        if event_type == "-mega" and slot is not None:
            mega_slots.add(slot)
            continue
        if event_type == "move":
            move_started = True
            if slot is None or slot in actions or len(event) < 4:
                continue
            actions[slot] = ObservedAction(
                "move",
                to_id_str(event[3]),
                _target_location(event[4] if len(event) > 4 else "", role, slot),
                slot in mega_slots,
            )
        elif event_type in {"switch", "drag"}:
            if move_started or slot is None or slot in actions or len(event) < 4:
                continue
            species = to_id_str(event[3].split(",", 1)[0])
            actions[slot] = ObservedAction("switch", species)
    return actions


def _choice_atoms(choice: str) -> list[str]:
    atoms = [atom.strip() for atom in choice.split(",")]
    return (atoms + ["pass", "pass"])[:2]


def _switch_species(node: ExactNode, role: str, atom: str) -> str:
    parts = atom.split()
    if len(parts) < 2:
        return ""
    try:
        pokemon = node.state["sides"][int(role[1]) - 1]["pokemon"][int(parts[1]) - 1]
    except (IndexError, KeyError, TypeError, ValueError):
        return ""
    pokemon_set = pokemon.get("set") or {}
    return to_id_str(
        pokemon_set.get("species")
        or pokemon.get("baseSpecies")
        or pokemon.get("species")
    )


def choice_matches_observation(
    node: ExactNode, role: str, choice: str, observed: dict[int, ObservedAction]
) -> bool:
    for slot, evidence in observed.items():
        atom = _choice_atoms(choice)[slot]
        parts = atom.split()
        if not parts or parts[0] != evidence.kind:
            return False
        if evidence.kind == "move":
            if len(parts) < 2 or to_id_str(parts[1]) != evidence.identifier:
                return False
            if evidence.mega and "mega" not in parts:
                return False
            targets = [int(part) for part in parts[2:] if part.lstrip("+-").isdigit()]
            # Showdown's public log names one affected Pokemon for spread moves such
            # as Rock Slide, while the submitted command correctly has no target.
            # Only compare targets when the command itself encoded one.
            if (
                targets
                and evidence.target is not None
                and evidence.target not in targets
            ):
                return False
        elif _switch_species(node, role, atom) != evidence.identifier:
            return False
    return True


def _state_public_error(state: dict[str, Any], snapshot: dict[str, Any]) -> float:
    """Soft damage/status likelihood distance for one concrete particle."""
    errors: list[float] = []
    for side_index in range(2):
        exact_by_name = {}
        for pokemon in state["sides"][side_index].get("pokemon", []):
            pokemon_set = pokemon.get("set") or {}
            name = to_id_str(
                pokemon_set.get("name")
                or pokemon_set.get("species")
                or pokemon.get("name")
            )
            exact_by_name[name] = pokemon
        for row in snapshot["sides"][side_index].get("pokemon", []):
            exact = exact_by_name.get(to_id_str(row.get("nickname")))
            if exact is None:
                continue
            maximum = float(exact.get("maxhp") or exact.get("baseMaxhp") or 1)
            exact_fraction = float(exact.get("hp") or 0) / maximum
            visible = row.get("hp_fraction")
            if visible is not None:
                errors.append(abs(exact_fraction - float(visible)))
            if bool(exact.get("fainted")) != bool(row.get("fainted")):
                errors.append(0.75)
            exact_status = to_id_str(exact.get("status") or "")
            if exact_status != to_id_str(row.get("status") or ""):
                errors.append(0.25)
    return sum(errors) / max(1, len(errors))


def _strategic_state_signature(state: dict[str, Any]) -> tuple[Any, ...]:
    """RNG-insensitive exact-state facts that can invalidate a saved line."""
    field = state.get("field") or {}
    field_signature = (
        to_id_str(field.get("weather") or ""),
        to_id_str(field.get("terrain") or ""),
        tuple(sorted(to_id_str(key) for key in (field.get("pseudoWeather") or {}))),
    )
    side_signatures = []
    for side in state.get("sides", []):
        conditions = tuple(
            sorted(to_id_str(key) for key in (side.get("sideConditions") or {}))
        )
        active = []
        for pokemon in side.get("pokemon", []):
            if not pokemon.get("isActive"):
                continue
            pokemon_set = pokemon.get("set") or {}
            species = to_id_str(
                pokemon.get("baseSpecies")
                or pokemon_set.get("species")
                or pokemon.get("species")
            )
            boosts = tuple(
                sorted(
                    (to_id_str(stat), int(value))
                    for stat, value in (pokemon.get("boosts") or {}).items()
                    if int(value)
                )
            )
            volatiles = tuple(
                sorted(to_id_str(key) for key in (pokemon.get("volatiles") or {}))
            )
            active.append(
                (
                    int(pokemon.get("position") or 0),
                    species,
                    bool(pokemon.get("fainted")),
                    to_id_str(pokemon.get("status") or ""),
                    boosts,
                    volatiles,
                )
            )
        side_signatures.append((conditions, tuple(sorted(active))))
    return field_signature, tuple(side_signatures)


def _move_order_from_events(events: Sequence[Sequence[str]]) -> list[str]:
    return [
        to_id_str(event[2].split(":", 1)[-1])
        for event in events
        if len(event) > 2 and event[1] == "move"
    ]


def _move_order_from_log(lines: Sequence[str]) -> list[str]:
    order = []
    for line in lines:
        parts = line.split("|")
        if len(parts) > 2 and parts[1] == "move":
            order.append(to_id_str(parts[2].split(":", 1)[-1]))
    return order


def _order_error(observed: Sequence[str], simulated: Sequence[str]) -> float:
    """Pairwise inversion rate among actors that moved in both outcomes."""
    observed_unique = list(dict.fromkeys(observed))
    simulated_unique = list(dict.fromkeys(simulated))
    common = [name for name in observed_unique if name in simulated_unique]
    if len(common) < 2:
        return 0.0
    simulated_index = {name: simulated_unique.index(name) for name in common}
    inversions = 0
    pairs = 0
    for left, right in itertools.combinations(common, 2):
        pairs += 1
        if simulated_index[left] > simulated_index[right]:
            inversions += 1
    return inversions / max(1, pairs)


def _roster_from_battle(battle: DoubleBattle, own: bool) -> tuple[TeamSlot, ...]:
    preview = battle.teampreview_team if own else battle.teampreview_opponent_team
    mons = (
        list(preview)
        if preview
        else list(battle.team.values() if own else battle.opponent_team.values())
    )
    return tuple(
        TeamSlot(_species(mon), str(mon.base_species or mon.species)) for mon in mons
    )


def _index_by_species(roster: Sequence[TeamSlot]) -> dict[str, int]:
    return {slot.species: index for index, slot in enumerate(roster, start=1)}


def _our_preview(
    battle: DoubleBattle,
    roster: Sequence[TeamSlot],
    nickname_species: dict[str, str] | None = None,
) -> str:
    indexes = _index_by_species(roster)
    nickname_species = nickname_species or {}
    identity_species: dict[int, str] = {}
    for ident, pokemon in battle.team.items():
        nickname = to_id_str(ident.split(":", 1)[-1])
        mapped = nickname_species.get(nickname)
        if mapped in indexes:
            identity_species[id(pokemon)] = mapped

    def roster_species(mon: Pokemon) -> str:
        mapped = identity_species.get(id(mon))
        if mapped:
            return mapped
        candidates = (
            to_id_str(getattr(mon, "base_species", None) or ""),
            _species(mon),
        )
        return next((candidate for candidate in candidates if candidate in indexes), "")

    active = [roster_species(mon) for mon in battle.active_pokemon if mon is not None]
    selected = [
        roster_species(mon)
        for mon in battle.team.values()
        if getattr(mon, "selected_in_teampreview", False)
        or getattr(mon, "_selected_in_teampreview", False)
    ]
    if len(selected) != 4 and len(battle.team) == 4:
        selected = [roster_species(mon) for mon in battle.team.values()]
    ordered = active + [species for species in selected if species not in active]
    if len(ordered) < 4:
        ordered.extend(slot.species for slot in roster if slot.species not in ordered)
    picked = [indexes[species] for species in ordered[:4]]
    if len(picked) != 4 or len(set(picked)) != 4:
        raise ValueError(f"could not recover our selected four: {ordered}")
    return "team " + ",".join(str(index) for index in picked)


def _opponent_previews(
    battle: DoubleBattle,
    roster: Sequence[TeamSlot],
    nickname_species: dict[str, str] | None = None,
) -> list[tuple[str, tuple[str, ...]]]:
    indexes = _index_by_species(roster)
    nickname_species = nickname_species or {}
    identity_species: dict[int, str] = {}
    for ident, pokemon in battle.opponent_team.items():
        nickname = to_id_str(ident.split(":", 1)[-1])
        mapped = nickname_species.get(nickname)
        if mapped in indexes:
            identity_species[id(pokemon)] = mapped

    def roster_species(mon: Pokemon) -> str:
        mapped = identity_species.get(id(mon))
        if mapped:
            return mapped
        candidates = (
            to_id_str(getattr(mon, "base_species", None) or ""),
            _species(mon),
        )
        return next((candidate for candidate in candidates if candidate in indexes), "")

    leads = [
        roster_species(mon) for mon in battle.opponent_active_pokemon if mon is not None
    ]
    if len(leads) != 2 or len(set(leads)) != 2:
        raise ValueError(f"opponent leads unavailable: {leads}")
    # On Turn 1 only the leads are revealed, so all six possible back pairs remain.
    # Later rebuilds must retain every Pokemon that has physically appeared. Without
    # this constraint a newly revealed reserve could be absent from every refreshed
    # root, causing the exact planner to fall back precisely when hidden information
    # becomes most useful.
    revealed = []
    for mon in battle.opponent_team.values():
        species = roster_species(mon)
        if getattr(mon, "revealed", False) and species and species not in revealed:
            revealed.append(species)
    required = leads + [species for species in revealed if species not in leads]
    if len(required) > 4:
        raise ValueError(
            f"opponent revealed more than four brought Pokemon: {required}"
        )
    remaining = [slot.species for slot in roster if slot.species not in required]
    needed = 4 - len(required)
    previews = []
    for back in itertools.combinations(remaining, needed):
        brought = tuple((*required, *back))
        previews.append(
            ("team " + ",".join(str(indexes[species]) for species in brought), brought)
        )
    if not previews:
        raise ValueError(f"could not build opponent brought-four worlds: {required}")
    return previews


def _particle_mass(determination: dict[str, SetParticle]) -> float:
    mass = 1.0
    for particle in determination.values():
        mass *= max(1e-8, particle.probability)
    return mass


class LiveExactSession:
    """One battle's exact roots, public evidence, and anytime planner."""

    def __init__(
        self,
        *,
        battle_tag: str,
        policy,
        our_team_text: str,
        formatid: str,
        open_sheet: bool,
        outcome_value_path: Path,
        outcome_evaluator=None,
        residual_ranker=None,
        preview_predictor=None,
        move_predictor=None,
        switch_predictor=None,
        device: str = "cpu",
        config: PlannerConfig | None = None,
        max_determinizations: int = 8,
        search_determinizations: int = 2,
        min_deep_coverage: float = 0.50,
        low_prior_override_ratio: float = 0.10,
        low_prior_override_min_coverage: float = 0.90,
        selective_search: bool = False,
        enable_ponder: bool = False,
        ponder_config: PonderConfig | None = None,
        policy_inference_lock=None,
        seed: int = 20260820,
    ):
        self.battle_tag = battle_tag
        self.policy = policy
        self.our_team_text = our_team_text
        self.formatid = formatid
        self.open_sheet = open_sheet
        self.max_determinizations = min(8, max(1, max_determinizations))
        self.search_determinizations = min(
            self.max_determinizations, max(1, search_determinizations)
        )
        if not 0.0 <= min_deep_coverage <= 1.0:
            raise ValueError("min_deep_coverage must be within [0, 1]")
        self.min_deep_coverage = float(min_deep_coverage)
        if not 0.0 <= low_prior_override_ratio <= 1.0:
            raise ValueError("low_prior_override_ratio must be within [0, 1]")
        if not 0.0 <= low_prior_override_min_coverage <= 1.0:
            raise ValueError(
                "low_prior_override_min_coverage must be within [0, 1]"
            )
        self.low_prior_override_ratio = float(low_prior_override_ratio)
        self.low_prior_override_min_coverage = float(
            low_prior_override_min_coverage
        )
        self.selective_search = bool(selective_search)
        self.enable_ponder = bool(enable_ponder and selective_search)
        self.ponder_config = ponder_config or PonderConfig()
        self.rng = random.Random(f"{seed}:{battle_tag}")
        self.bridge = ExactShowdownBridge()
        self.database = ParticleDatabase.load(max_particles=12)
        self.adapter = ExactPolicyAdapter(
            policy,
            preview_predictor=preview_predictor,
            reveal_opponent_sets=open_sheet,
            residual_ranker=residual_ranker,
            inference_lock=policy_inference_lock,
        )
        self.prior = OpponentModelPrior(
            self.adapter,
            move_predictor=move_predictor,
            switch_predictor=switch_predictor,
            controlled_role="p1",
        )
        self.evaluator = outcome_evaluator or OutcomeValueEvaluator.load(
            outcome_value_path, device=device, mechanics_weight=0.10
        )
        self.config = config or PlannerConfig(
            depth=2,
            root_width=6,
            opponent_width=6,
            continuation_width=3,
            replacement_width=2,
            chance_samples=1,
            anytime=True,
            screen_budget_s=2.0,
            time_budget_s=9.0,
            max_nodes=5000,
        )
        self.roots: list[LiveRoot] = []
        self.event_cursor = 0
        self.pending_our_choice: str | None = None
        self.last_submitted_move_choice: str | None = None
        self.last_executed_choice: str | None = None
        self.last_result: PlanResult | None = None
        self.planned_continuations: tuple[BranchContinuation, ...] = ()
        self.planned_outcomes: tuple[BranchOutcome, ...] = ()
        self.plan_parent_nodes: dict[str, ExactNode] = {}
        self.planned_root_margin = 0.0
        self.planned_reference_score = 0.0
        self.last_search_turn: int | None = None
        self.last_recent_events: list[Sequence[str]] = []
        self.last_observed_actions: dict[int, ObservedAction] = {}
        self.last_snapshot: dict[str, Any] | None = None
        self.last_schedule: dict[str, Any] = {"mode": "not_started", "reasons": []}
        self.last_live_guards: dict[str, Any] = {
            "changed_pick": False,
            "stages": [],
            "demotions": {},
            "strict_rejections": [],
        }
        self.last_reuse_rejection: dict[str, Any] | None = None
        self.last_outcome_rejection: dict[str, Any] | None = None
        self.last_ponder_rejection: dict[str, Any] | None = None
        self._ponder_job: BackgroundPonder | None = None
        self._pending_ponder_choice: str | None = None
        self._ponder_generation = 0
        self.pondered_outcomes: tuple[PonderOutcome, ...] = ()
        self.ponder_parent_nodes: dict[str, ExactNode] = {}
        self.ponder_reference_values: dict[str, float] = {}
        self.last_ponder: dict[str, Any] | None = None
        self.ponder_started = 0
        self.ponder_completed = 0
        self.ponder_partial = 0
        self.ponder_late = 0
        self.ponder_errors = 0
        self.ponder_matches = 0
        self.full_searches = 0
        self.reused_plans = 0
        self.skipped_searches = 0
        self.fallbacks = 0
        self.root_reconcile_failures = 0
        self.reconciliations = 0
        self.root_refreshes = 0
        self.last_root_refresh_turn: int | None = None
        self.eliminated_roots = 0
        self.last_reconcile_errors: list[str] = []
        self.our_species_by_nickname: dict[str, str] = {}
        self.opponent_species_by_nickname: dict[str, str] = {}
        self.opponent_roster_species: set[str] = set()
        self.current_battle: DoubleBattle | None = None

    def close(self) -> None:
        if self._ponder_job is not None:
            self._ponder_job.cancel()
            self._ponder_job.join(0.2)
            self._ponder_job = None
        self.bridge.close()

    def _belief(self, battle: DoubleBattle, roster: Sequence[TeamSlot]) -> TeamBelief:
        belief = TeamBelief.from_roster(
            self.database, (slot.species for slot in roster)
        )
        if self.open_sheet:
            by_species = {_species(mon): mon for mon in battle.opponent_team.values()}
            for species, mon in by_species.items():
                belief.condition(
                    species, moves=mon.moves.keys(), item=mon.item, ability=mon.ability
                )
        return belief

    def _create_roots(self, battle: DoubleBattle, snapshot: dict[str, Any]) -> None:
        our_roster = team_roster(self.our_team_text)
        opponent_roster = _roster_from_battle(battle, own=False)
        if len(our_roster) != 6 or len(opponent_roster) != 6:
            raise ValueError(
                f"incomplete preview rosters ours={len(our_roster)} "
                f"theirs={len(opponent_roster)}"
            )
        our_indexes = _index_by_species(our_roster)
        for ident, mon in battle.team.items():
            nickname = to_id_str(ident.split(":", 1)[-1])
            candidates = (
                to_id_str(getattr(mon, "base_species", None) or ""),
                _species(mon),
            )
            canonical = next(
                (candidate for candidate in candidates if candidate in our_indexes), ""
            )
            if nickname and canonical:
                self.our_species_by_nickname.setdefault(nickname, canonical)
        p1_preview = _our_preview(battle, our_roster, self.our_species_by_nickname)
        self.opponent_roster_species = {slot.species for slot in opponent_roster}
        for ident, mon in battle.opponent_team.items():
            nickname = to_id_str(ident.split(":", 1)[-1])
            self.opponent_species_by_nickname.setdefault(
                nickname, self._canonical_opponent_species(mon)
            )
        opponent_previews = _opponent_previews(
            battle, opponent_roster, self.opponent_species_by_nickname
        )
        belief = self._belief(battle, opponent_roster)
        determinations = belief.sample_determinizations(
            1 if self.open_sheet else self.max_determinizations,
            self.rng,
            open_sheet=self.open_sheet,
        )
        # Round-robin the two uncertainty axes. Sorting a Cartesian product by back
        # pair accidentally spent all eight roots on one pair with eight sets.
        # This ordering covers all six possible back pairs first while still using a
        # different hidden-set sample in each root.
        count = min(
            self.max_determinizations, max(len(determinations), len(opponent_previews))
        )
        candidates = [
            (
                determinations[index % len(determinations)],
                opponent_previews[index % len(opponent_previews)],
            )
            for index in range(count)
        ]
        roots: list[LiveRoot] = []
        for index, (determination, (p2_preview, brought)) in enumerate(candidates):
            result = self.bridge.create(
                formatid=self.formatid,
                seed=[self.rng.randrange(1, 65536) for _ in range(4)],
                p1_name="planner",
                p2_name="opponent",
                p1_team_text=self.our_team_text,
                p2_team_text=determination_team_text(opponent_roster, determination),
                p1_preview=p1_preview,
                p2_preview=p2_preview,
            )
            reconciled = self.bridge.reconcile(result["state"], snapshot)
            roots.append(
                LiveRoot(
                    ExactNode.from_result(reconciled),
                    _particle_mass(determination),
                    dict(determination),
                    brought,
                    f"live-{index + 1}:{','.join(brought[2:])}",
                )
            )
        self.roots = self._normalise(roots)
        self.event_cursor = len(getattr(battle, "_replay_data", []))

    def _canonical_opponent_species(self, mon: Pokemon) -> str:
        """Map a revealed battle forme back to its preview/determination species."""
        candidates = (
            to_id_str(getattr(mon, "base_species", None) or ""),
            _species(mon),
        )
        for candidate in candidates:
            if candidate in self.opponent_roster_species:
                return candidate
        return next((candidate for candidate in candidates if candidate), "")

    def _refresh_opponent_identities(self, battle: DoubleBattle) -> None:
        """Learn nickname identities as back Pokemon appear during the battle."""
        for ident, mon in battle.opponent_team.items():
            nickname = to_id_str(ident.split(":", 1)[-1])
            species = self._canonical_opponent_species(mon)
            if nickname and species:
                # Keep the first public identity. Ditto/Transform and Illusion can
                # later make poke-env report another species for the same nickname;
                # overwriting here corrupts brought-four reconstruction.
                self.opponent_species_by_nickname.setdefault(nickname, species)

    @staticmethod
    def _normalise(roots: Sequence[LiveRoot]) -> list[LiveRoot]:
        total = sum(max(0.0, root.probability) for root in roots)
        if total <= 0:
            raise ValueError("live exact roots have no probability mass")
        return [replace(root, probability=root.probability / total) for root in roots]

    def _revealed_evidence(self, battle: DoubleBattle) -> dict[str, dict[str, Any]]:
        self._refresh_opponent_identities(battle)
        nickname_species = self.opponent_species_by_nickname
        evidence: dict[str, dict[str, Any]] = {}
        if self.open_sheet:
            for ident, mon in battle.opponent_team.items():
                nickname = to_id_str(ident.split(":", 1)[-1])
                species = nickname_species.get(nickname, _species(mon))
                evidence[species] = {
                    "moves": set(mon.moves),
                    "item": to_id_str(mon.item) if mon.item else None,
                    "ability": to_id_str(mon.ability) if mon.ability else None,
                }
        for event in getattr(battle, "_replay_data", []):
            if len(event) < 3 or not event[2].startswith("p2"):
                continue
            nickname = to_id_str(event[2].split(":", 1)[-1])
            species = nickname_species.get(nickname)
            if not species:
                continue
            row = evidence.setdefault(
                species, {"moves": set(), "item": None, "ability": None}
            )
            if event[1] == "move" and len(event) > 3:
                row["moves"].add(to_id_str(event[3]))
            elif event[1] in {"-item", "-enditem"} and len(event) > 3:
                row["item"] = to_id_str(event[3])
            elif event[1] == "-ability" and len(event) > 3:
                row["ability"] = to_id_str(event[3])
            elif event[1] == "-mega" and len(event) > 4:
                row["item"] = to_id_str(event[4])
        return evidence

    def _condition_roots(self, battle: DoubleBattle) -> None:
        evidence = self._revealed_evidence(battle)
        revealed_species = set(evidence)
        conditioned: list[LiveRoot] = []
        for root in self.roots:
            # Open sheets reveal all six sets, but never which four were selected.
            # Only species that have physically appeared can collapse bring-pair
            # uncertainty.
            if not self.open_sheet and not revealed_species.issubset(root.brought):
                self.eliminated_roots += 1
                continue
            compatible = True
            for species, row in evidence.items():
                particle = root.determination.get(species)
                if particle is None:
                    compatible = False
                    break
                if not set(row["moves"]).issubset(particle.moves):
                    compatible = False
                    break
                if row["item"] and to_id_str(particle.item or "") != row["item"]:
                    compatible = False
                    break
                if (
                    row["ability"]
                    and to_id_str(particle.ability or "") != row["ability"]
                ):
                    compatible = False
                    break
            if compatible:
                conditioned.append(root)
            else:
                # Set evidence can eliminate a particle, but it must not also erase
                # the only world containing one unrevealed back-pair. Keep an
                # effectively-zero mass placeholder until bring evidence resolves
                # that independent uncertainty axis.
                self.eliminated_roots += 1
                conditioned.append(replace(root, probability=root.probability * 1e-3))
        self.roots = self._normalise(conditioned)

    def _opponent_choice(
        self, root: LiveRoot, observed: dict[int, ObservedAction]
    ) -> str:
        legal = self.bridge.choices(root.node.state, "p2")
        compatible = [
            choice
            for choice in legal
            if choice_matches_observation(root.node, "p2", choice, observed)
        ]
        if not compatible:
            compatible = legal
        ranked = self.prior.rank(root.node.state, root.node.requests, "p2", compatible)
        return ranked[0].choice if ranked else compatible[0]

    def _advance_and_reconcile(
        self, battle: DoubleBattle, snapshot: dict[str, Any]
    ) -> None:
        events = getattr(battle, "_replay_data", [])
        recent = events[self.event_cursor :]
        observed = observed_opponent_actions(recent)
        self.last_recent_events = list(recent)
        self.last_observed_actions = observed
        observed_order = _move_order_from_events(recent)
        shared_rng = [
            ",".join(str(self.rng.randrange(1, 65536)) for _ in range(4))
            for _ in range(4)
        ]
        self.last_reconcile_errors = []
        previous_roots = list(self.roots)
        advanced: list[LiveRoot] = []
        for root in previous_roots:
            state = root.node.state
            if self.pending_our_choice is not None:
                try:
                    p1_legal = self.bridge.choices(state, "p1")
                    if self.pending_our_choice in p1_legal:
                        p2_choice = self._opponent_choice(root, observed)
                        results = self.bridge.simulate_batch(
                            state,
                            [
                                {
                                    "p1_choice": self.pending_our_choice,
                                    "p2_choice": p2_choice,
                                    "rng_seed": seed,
                                }
                                for seed in shared_rng
                            ],
                        )
                        scored = [
                            (
                                _state_public_error(result["state"], snapshot)
                                + 0.20
                                * _order_error(
                                    observed_order,
                                    _move_order_from_log(result.get("log") or []),
                                ),
                                result,
                            )
                            for result in results
                        ]
                        numeric_error, result = min(scored, key=lambda item: item[0])
                        state = result["state"]
                        root = replace(
                            root,
                            probability=root.probability
                            * max(0.02, math.exp(-numeric_error / 0.10)),
                        )
                except Exception as exc:
                    self.root_reconcile_failures += 1
                    self.last_reconcile_errors.append(
                        f"advance {root.label}: {type(exc).__name__}: {exc}"
                    )
            try:
                reconciled = self.bridge.reconcile(state, snapshot)
            except Exception as exc:
                self.root_reconcile_failures += 1
                self.last_reconcile_errors.append(f"{type(exc).__name__}: {exc}")
                continue
            advanced.append(replace(root, node=ExactNode.from_result(reconciled)))
        if len(advanced) < len(previous_roots):
            # A newly revealed reserve legitimately invalidates brought-four worlds.
            # Rebuild around every Pokemon seen so far, restoring both back-pair and
            # hidden-set coverage instead of limping forward with one particle or
            # abandoning exact search when the final reserve appears.
            try:
                if self._ponder_job is not None:
                    self._ponder_job.cancel()
                    self._ponder_job.join(0.2)
                    self._ponder_job = None
                self.pending_our_choice = None
                self._pending_ponder_choice = None
                self.planned_continuations = ()
                self.planned_outcomes = ()
                self.plan_parent_nodes = {}
                self.pondered_outcomes = ()
                self.ponder_parent_nodes = {}
                self.ponder_reference_values = {}
                self.roots = []
                self._create_roots(battle, snapshot)
                self.root_refreshes += 1
                self.last_root_refresh_turn = int(battle.turn)
                self.reconciliations += len(self.roots)
                return
            except Exception as exc:
                self.last_reconcile_errors.append(
                    f"root refresh: {type(exc).__name__}: {exc}"
                )
                if not advanced:
                    detail = "; ".join(self.last_reconcile_errors[:4])
                    raise ValueError(
                        f"all live exact roots failed reconciliation: {detail}"
                    ) from exc
        if not advanced:
            detail = "; ".join(self.last_reconcile_errors[:3])
            raise ValueError(f"all live exact roots failed reconciliation: {detail}")
        self.roots = self._normalise(advanced)
        self.reconciliations += len(advanced)
        self.pending_our_choice = None
        self.event_cursor = len(events)

    def prepare(self, battle: DoubleBattle) -> None:
        self.current_battle = battle
        snapshot = public_snapshot(
            battle,
            battle.last_request,
            request_state=("switch" if any(battle.force_switch) else "move"),
            pending_our_choice=self.last_submitted_move_choice,
        )
        self.last_snapshot = snapshot
        if not self.roots:
            self.last_recent_events = []
            self.last_observed_actions = {}
            self._create_roots(battle, snapshot)
        else:
            self._advance_and_reconcile(battle, snapshot)
        self._condition_roots(battle)
        self._consume_ponder()

    def _consume_ponder(self) -> None:
        """Take a finished background result without ever waiting for it."""
        self.pondered_outcomes = ()
        self.ponder_parent_nodes = {}
        self.ponder_reference_values = {}
        job = self._ponder_job
        if job is None:
            return
        result = job.result_if_done()
        if result is None:
            job.cancel()
            result = job.result_now()
            if result is None:
                self.ponder_late += 1
                self.last_ponder = {
                    "generation": job.generation,
                    "status": "late_cancelled_no_completed_roots",
                }
                self._ponder_job = None
                return
            self.ponder_partial += 1
            status = "partial_cancelled"
        else:
            self.ponder_completed += 1
            status = "completed"
        self._ponder_job = None
        if result.error:
            self.ponder_errors += 1
        self.last_ponder = {
            "generation": result.generation,
            "status": status,
            "elapsed_s": result.elapsed_s,
            "roots_completed": result.roots_completed,
            "choices_simulated": result.choices_simulated,
            "outcomes": len(result.outcomes),
            "truncated": result.truncated,
            "cancelled": result.cancelled,
            "error": result.error,
        }
        if not result.outcomes:
            return
        self.pondered_outcomes = result.outcomes
        self.ponder_parent_nodes = {root.label: root.node for root in result.roots}
        self.ponder_reference_values = dict(result.reference_values)

    def _start_ponder(self, choice: str) -> None:
        """Launch isolated expansion after selecting an action; return immediately."""
        if not self.enable_ponder or not choice or not self.roots:
            return
        if self._ponder_job is not None:
            self._ponder_job.cancel()
        self._ponder_generation += 1
        roots = [
            PonderRoot(root.node, root.probability, root.label) for root in self.roots
        ]
        # The foreground search has already ranked a small set of plausible opponent
        # replies.  Put those replies at the front of the background queue, then let
        # the ponder worker spend any remaining budget on broad move-family coverage.
        # This is the chess-engine ordering rule: examine the likely principal
        # variations first so short opponent think times still produce useful work.
        reply_mass: dict[str, dict[str, float]] = {}
        for outcome in self.planned_outcomes:
            if outcome.root_choice != choice or not outcome.root_label:
                continue
            by_choice = reply_mass.setdefault(outcome.root_label, {})
            by_choice[outcome.opponent_choice] = by_choice.get(
                outcome.opponent_choice, 0.0
            ) + float(outcome.probability)
        if not reply_mass:
            for continuation in self.planned_continuations:
                if continuation.root_choice != choice or not continuation.root_label:
                    continue
                by_choice = reply_mass.setdefault(continuation.root_label, {})
                by_choice[continuation.opponent_choice] = by_choice.get(
                    continuation.opponent_choice, 0.0
                ) + float(continuation.probability)
        preferred_choices = {
            label: tuple(
                opponent_choice
                for opponent_choice, _mass in sorted(
                    masses.items(), key=lambda item: (-item[1], item[0])
                )
            )
            for label, masses in reply_mass.items()
        }
        job = BackgroundPonder(
            roots,
            choice,
            generation=self._ponder_generation,
            config=self.ponder_config,
            preferred_choices=preferred_choices,
        )
        self._ponder_job = job
        self.ponder_started += 1
        self.last_ponder = {
            "generation": job.generation,
            "status": "running",
            "root_choice": choice,
            "roots": len(roots),
            "preferred_replies": sum(
                len(replies) for replies in preferred_choices.values()
            ),
        }
        job.start()

    def start_pending_ponder(self) -> None:
        """Called only after the websocket submission has completed."""
        choice = self._pending_ponder_choice
        self._pending_ponder_choice = None
        if choice is not None:
            self._start_ponder(choice)

    def _root_margin(self, result: PlanResult) -> float:
        eligible = [
            row
            for row in result.rankings
            if row.depth_coverage + 1e-9 >= self.min_deep_coverage
        ]
        if not eligible:
            eligible = list(result.rankings)
        if len(eligible) < 2:
            return float("inf")
        selected = result.rankings[0]
        alternatives = [row.score for row in eligible if row.choice != selected.choice]
        if not alternatives:
            return float("inf")
        return float(selected.score - max(alternatives))

    def _reaches_required_depth(self, row: ActionScore) -> bool:
        """Whether a live-safe replacement retained enough searched-world depth."""
        return row.depth_coverage + 1e-9 >= self.min_deep_coverage

    @staticmethod
    def _priority_ko_justifies_low_prior_override(
        battle: DoubleBattle, winner: ActionScore
    ) -> bool:
        """Allow a low-prior priority move when normal move order loses the KO."""
        from poke_env.battle import Move, MoveCategory

        from vgc_bench.src import vgc_knowledge as K
        from vgc_bench.src.guards import (
            _decode,
            _effective_priority,
            _foe_moves_first,
            resolved_foe_targets,
        )

        if winner.actions is None:
            return False
        for pos, action in enumerate(winner.actions):
            attacker = battle.active_pokemon[pos]
            if attacker is None or attacker.fainted:
                continue
            order = _decode(battle, action, pos)
            move = getattr(order, "order", None)
            if (
                not isinstance(move, Move)
                or move.category == MoveCategory.STATUS
                or _effective_priority(attacker, move) <= 0
            ):
                continue
            for defender in resolved_foe_targets(battle, order, move):
                if _foe_moves_first(battle, attacker, defender) and K.guaranteed_ko(
                    battle, attacker, defender, move
                ):
                    return True
        return False

    def _low_prior_override_rejection(
        self,
        battle: DoubleBattle,
        winner: ActionScore,
        rankings: Sequence[ActionScore],
    ) -> dict[str, Any] | None:
        """Reject speculative shallow overrides of a strongly preferred policy line."""
        max_prior = max((max(0.0, float(row.prior)) for row in rankings), default=0.0)
        selected_prior = max(0.0, float(winner.prior))
        prior_ratio = selected_prior / max_prior if max_prior > 0 else 1.0
        coverage = float(winner.depth_coverage)
        if prior_ratio + 1e-12 >= self.low_prior_override_ratio:
            return None
        if bool(self.last_live_guards.get("changed_pick")):
            return None
        if self._priority_ko_justifies_low_prior_override(battle, winner):
            return None
        return {
            # A complete lookahead can still be confidently wrong when its leaf
            # evaluator is weak. Until the evaluator earns promotion, do not let a
            # line the champion considers extremely implausible replace it without
            # independent tactical support. This caught full-depth Last Respects
            # into Kommo-o and Double Protect while Wave Crash won immediately.
            "reason": (
                "shallow_low_prior_override"
                if coverage + 1e-9 < self.low_prior_override_min_coverage
                else "unsupported_low_prior_override"
            ),
            "choice": winner.choice,
            "selected_prior": selected_prior,
            "maximum_prior": max_prior,
            "prior_ratio": prior_ratio,
            "selected_depth_coverage": coverage,
            "required_depth_coverage": self.low_prior_override_min_coverage,
            "minimum_prior_ratio": self.low_prior_override_ratio,
        }

    @staticmethod
    def _critical_event_reasons(events: Sequence[Sequence[str]]) -> list[str]:
        """Return only events that can invalidate a strategic line.

        Ordinary damage and ordinary moves are deliberately absent.  The point is to
        search when the position changed qualitatively, not merely because another
        turn elapsed.
        """
        reasons: set[str] = set()
        direct = {
            "switch": "switch_or_new_pokemon",
            "drag": "forced_switch",
            "faint": "faint",
            "-weather": "weather_changed",
            "-fieldstart": "field_changed",
            "-fieldend": "field_changed",
            "-sidestart": "side_condition_changed",
            "-sideend": "side_condition_changed",
            "-status": "status_changed",
            "-curestatus": "status_changed",
            "-boost": "stat_stage_changed",
            "-unboost": "stat_stage_changed",
            "-setboost": "stat_stage_changed",
            "-clearboost": "stat_stage_changed",
            "-clearallboost": "stat_stage_changed",
            "-invertboost": "stat_stage_changed",
            "-mega": "major_mechanic",
            "-terastallize": "major_mechanic",
            "-ability": "hidden_information_revealed",
            "-item": "hidden_information_revealed",
            "-enditem": "hidden_information_revealed",
        }
        volatile = {
            "encore",
            "disable",
            "yawn",
            "drowsy",
            "taunt",
            "confusion",
            "perishsong",
            "substitute",
            "leechseed",
        }
        for event in events:
            if len(event) < 2:
                continue
            event_type = event[1]
            # Showdown repeats ``|-weather|RainDance|[upkeep]`` every turn.  That
            # is not a qualitative board change and previously forced exact search
            # on virtually every rain/sun/sand turn.
            if event_type == "-weather" and any(
                "[upkeep]" in str(value).lower() for value in event[2:]
            ):
                continue
            reason = direct.get(event_type)
            if reason:
                reasons.add(reason)
            if event_type in {"-start", "-end"} and len(event) > 3:
                effect = to_id_str(event[3].replace("move:", ""))
                if effect in volatile:
                    reasons.add("volatile_effect_changed")
        return sorted(reasons)

    def _importance_reasons(self, battle: DoubleBattle) -> list[str]:
        if not self.selective_search:
            return ["search_every_turn"]
        reasons = self._critical_event_reasons(self.last_recent_events)
        if self.last_search_turn is None:
            reasons.append("first_decision")
        request_state = self.roots[0].node.request_state if self.roots else ""
        if request_state != "move" or any(getattr(battle, "force_switch", [])):
            reasons.append("forced_replacement")
        # Low HP by itself is not a chess-style trigger. It persists across turns
        # and made one damaged Pokemon force full search forever. Faints, switches,
        # statuses and genuinely unanticipated actions are handled separately.
        if (
            self.last_search_turn is not None
            and int(battle.turn) - self.last_search_turn >= 3
        ):
            reasons.append("periodic_refresh")
        return sorted(set(reasons))

    def _planned_opponent_action_coverage(self) -> float:
        """Posterior mass where the observed opponent action was searched.

        Damage rolls frequently keep the exact public snapshot from matching a saved
        child bit-for-bit.  That is RNG drift, not evidence that the opponent left
        the searched line.  Re-search only when their *action* was absent from at
        least half of the surviving hidden-set worlds.
        """
        if (
            self.last_executed_choice is None
            or not self.last_observed_actions
            or not self.plan_parent_nodes
        ):
            return 0.0
        branches = (*self.planned_outcomes, *self.planned_continuations)
        if not branches:
            return 0.0
        matching_labels: set[str] = set()
        for branch in branches:
            if branch.root_choice != self.last_executed_choice:
                continue
            parent = self.plan_parent_nodes.get(branch.root_label)
            if parent is None:
                continue
            if choice_matches_observation(
                parent, "p2", branch.opponent_choice, self.last_observed_actions
            ):
                matching_labels.add(branch.root_label)
        # Foreground search intentionally deepens only a representative subset of
        # the belief roots. Coverage must be relative to that searched posterior,
        # not all eight retained particles; otherwise two-for-two agreement among
        # representatives is mislabeled as roughly 25% coverage and forces another
        # expensive search on the next turn.
        searched_labels = set(self.plan_parent_nodes)
        searched_mass = sum(
            root.probability
            for root in self.roots
            if root.label in searched_labels
        )
        if searched_mass <= 0:
            return 0.0
        matching_mass = sum(
            root.probability
            for root in self.roots
            if root.label in matching_labels
        )
        return min(1.0, matching_mass / searched_mass)

    def _planned_opponent_family_coverage(self) -> float:
        """Coverage where every observed slot action appeared in searched branches.

        The finite opponent width cannot enumerate every Cartesian pairing of two
        independently plausible moves. For deciding whether to think again *after*
        the result is public, require each slot's move/switch family to have been
        searched somewhere in that world, while keeping exact joint-pair matching
        for continuation reuse and position-value claims.
        """
        if (
            self.last_executed_choice is None
            or not self.last_observed_actions
            or not self.plan_parent_nodes
        ):
            return 0.0
        branches = (*self.planned_outcomes, *self.planned_continuations)
        if not branches:
            return 0.0
        by_label: dict[str, list[Any]] = {}
        for branch in branches:
            if branch.root_choice == self.last_executed_choice:
                by_label.setdefault(branch.root_label, []).append(branch)
        matching_labels: set[str] = set()
        for label, rows in by_label.items():
            parent = self.plan_parent_nodes.get(label)
            if parent is None:
                continue
            if all(
                any(
                    choice_matches_observation(
                        parent,
                        "p2",
                        branch.opponent_choice,
                        {slot: observed},
                    )
                    for branch in rows
                )
                for slot, observed in self.last_observed_actions.items()
            ):
                matching_labels.add(label)
        searched_labels = set(self.plan_parent_nodes)
        searched_mass = sum(
            root.probability
            for root in self.roots
            if root.label in searched_labels
        )
        if searched_mass <= 0:
            return 0.0
        matching_mass = sum(
            root.probability
            for root in self.roots
            if root.label in matching_labels
        )
        return min(1.0, matching_mass / searched_mass)

    def _planned_position_coverage(self) -> float:
        """Searched-world agreement that the strategic successor was predicted.

        This deliberately ignores HP-roll drift while retaining exact weather,
        field, side-condition, active identity, faint, status, boost, and volatile
        agreement. It never reuses an action; it only says that a fresh search can
        be skipped and the champion-plus-guards can handle this already-examined
        position.
        """
        if (
            self.last_executed_choice is None
            or not self.plan_parent_nodes
            or not self.planned_outcomes
            or not self.roots
        ):
            return 0.0
        current = {root.label: root for root in self.roots}
        matching_labels: set[str] = set()
        for outcome in self.planned_outcomes:
            if outcome.root_choice != self.last_executed_choice:
                continue
            parent = self.plan_parent_nodes.get(outcome.root_label)
            root = current.get(outcome.root_label)
            if parent is None or root is None:
                continue
            if self.last_observed_actions and not choice_matches_observation(
                parent, "p2", outcome.opponent_choice, self.last_observed_actions
            ):
                continue
            if outcome.predicted_node.request_state != root.node.request_state:
                continue
            if _strategic_state_signature(
                outcome.predicted_node.state
            ) != _strategic_state_signature(root.node.state):
                continue
            matching_labels.add(outcome.root_label)
        searched_labels = set(self.plan_parent_nodes)
        searched_mass = sum(
            root.probability
            for root in self.roots
            if root.label in searched_labels
        )
        if searched_mass <= 0:
            return 0.0
        matching_mass = sum(
            root.probability
            for root in self.roots
            if root.label in matching_labels
        )
        return min(1.0, matching_mass / searched_mass)

    @staticmethod
    def _expected_action_is_quiet(
        reasons: Sequence[str], planned_action_coverage: float
    ) -> bool:
        """Whether only anticipated, deterministic board updates occurred.

        Boosts, speed control, field/side conditions, and weather are exactly the
        kinds of consequences a searched move line already considered. Faints,
        switches, status/RNG effects, new hidden information, major mechanics, and
        refresh boundaries still demand a new calculation.
        """
        if planned_action_coverage < 0.75:
            return False
        expected_updates = {
            "field_changed",
            "side_condition_changed",
            "stat_stage_changed",
            "weather_changed",
        }
        return bool(reasons) and set(reasons).issubset(expected_updates)

    def _foreground_determinizations(self, turn: int) -> int:
        return (
            1
            if self.last_root_refresh_turn == int(turn)
            else self.search_determinizations
        )

    def _reuse_contingent_plan(self, battle: DoubleBattle) -> PlanResult | None:
        """Reuse one searched continuation only when the public line still matches."""
        self.last_reuse_rejection = None
        continuations = self.planned_continuations
        snapshot = self.last_snapshot
        prerequisites = {
            "selective": self.selective_search,
            "continuations": len(continuations),
            "has_snapshot": snapshot is not None,
            "has_executed_choice": self.last_executed_choice is not None,
            "root_margin": self.planned_root_margin,
        }
        if not self.selective_search or not continuations or snapshot is None:
            self.last_reuse_rejection = {
                "reason": "missing_plan_prerequisite",
                **prerequisites,
            }
            return None
        if self.last_executed_choice is None or self.planned_root_margin < 0.025:
            self.last_reuse_rejection = {
                "reason": "weak_or_unrecorded_root",
                **prerequisites,
            }
            return None

        current_roots = {root.label: root for root in self.roots}
        matching_mass = 0.0
        eligible_mass = 0.0
        grouped: dict[str, list[tuple[float, BranchContinuation, float]]] = {}
        rejected = {
            "root_choice": 0,
            "missing_world": 0,
            "opponent_action": 0,
            "illegal_next": 0,
            "request_state": 0,
            "strategic_state": 0,
            "damage_state": 0,
        }
        for continuation in continuations:
            if continuation.root_choice != self.last_executed_choice:
                rejected["root_choice"] += 1
                continue
            current = current_roots.get(continuation.root_label)
            parent = self.plan_parent_nodes.get(continuation.root_label)
            if current is None or parent is None:
                rejected["missing_world"] += 1
                continue
            if self.last_observed_actions and not choice_matches_observation(
                parent, "p2", continuation.opponent_choice, self.last_observed_actions
            ):
                rejected["opponent_action"] += 1
                continue
            matching_mass += continuation.probability
            if continuation.next_choice not in self.bridge.choices(
                current.node.state, "p1"
            ):
                rejected["illegal_next"] += 1
                continue
            if continuation.predicted_node.request_state != current.node.request_state:
                rejected["request_state"] += 1
                continue
            if _strategic_state_signature(
                continuation.predicted_node.state
            ) != _strategic_state_signature(current.node.state):
                rejected["strategic_state"] += 1
                continue
            error = _state_public_error(continuation.predicted_node.state, snapshot)
            if error > 0.16:
                rejected["damage_state"] += 1
                continue
            weight = continuation.probability * math.exp(-error / 0.12)
            eligible_mass += weight
            grouped.setdefault(continuation.next_choice, []).append(
                (weight, continuation, error)
            )
        if matching_mass <= 0 or eligible_mass <= 0 or not grouped:
            self.last_reuse_rejection = {
                "reason": "no_matching_successor",
                "matching_mass": matching_mass,
                "eligible_mass": eligible_mass,
                "saved_continuations": len(continuations),
                "rejected": rejected,
                "observed_opponent_actions": {
                    str(slot): asdict(action)
                    for slot, action in self.last_observed_actions.items()
                },
                "searched_opponent_choices": sorted(
                    {item.opponent_choice for item in continuations}
                )[:12],
            }
            return None

        next_choice, rows = max(
            grouped.items(), key=lambda item: sum(row[0] for row in item[1])
        )
        winning_mass = sum(row[0] for row in rows)
        consensus = winning_mass / eligible_mass
        coverage = min(1.0, eligible_mass / matching_mass)
        weighted_margin = (
            sum(
                weight * min(1.0, continuation.margin)
                for weight, continuation, _error in rows
            )
            / winning_mass
        )
        if consensus < 0.72 or coverage < 0.50 or weighted_margin < 0.02:
            self.last_reuse_rejection = {
                "reason": "insufficient_branch_confidence",
                "consensus": consensus,
                "coverage": coverage,
                "continuation_margin": weighted_margin,
            }
            return None
        # Own-side legality should not vary with an opponent set particle.  Requiring
        # it in every surviving world protects against a desynchronized clone.
        if any(
            next_choice not in self.bridge.choices(root.node.state, "p1")
            for root in self.roots
        ):
            self.last_reuse_rejection = {
                "reason": "choice_not_legal_in_every_world",
                "choice": next_choice,
            }
            return None

        actions = self._live_actions(next_choice, battle)
        strict_reason = self._strict_live_rejection_reason(battle, actions)
        if strict_reason is not None:
            self.last_reuse_rejection = {
                "reason": "continuation_failed_factual_guard",
                "choice": next_choice,
                "guard": strict_reason,
            }
            return None
        score = (
            sum(weight * continuation.value for weight, continuation, _error in rows)
            / winning_mass
        )
        row = ActionScore(
            choice=next_choice,
            actions=actions,
            score=score,
            expected=score,
            cvar=score,
            worst=min(continuation.value for _weight, continuation, _error in rows),
            standard_deviation=0.0,
            prior=consensus,
            opponent_branches=len(rows),
        )
        result = PlanResult(
            choice=next_choice,
            actions=actions,
            score=score,
            rankings=(row,),
            nodes=0,
            elapsed_s=0.0,
            completed_depth=1,
            truncated=False,
            screened_actions=0,
            deepened_actions=0,
            fallback_reason="reused_contingent_plan",
        )
        self.last_schedule = {
            "mode": "reuse",
            "reasons": ["observed_line_matched_deep_search"],
            "consensus": consensus,
            "coverage": coverage,
            "continuation_margin": weighted_margin,
            "matching_branches": len(rows),
        }
        self.reused_plans += 1
        self.last_reuse_rejection = None
        self.planned_continuations = ()
        self.planned_outcomes = ()
        self.plan_parent_nodes = {}
        self.pondered_outcomes = ()
        self.ponder_parent_nodes = {}
        self.ponder_reference_values = {}
        return result

    def _matching_planned_outcome(self) -> dict[str, Any] | None:
        """Recognize an acceptable position already evaluated by the prior search."""
        self.last_outcome_rejection = None
        outcomes = self.planned_outcomes
        snapshot = self.last_snapshot
        if (
            not self.selective_search
            or not outcomes
            or snapshot is None
            or self.last_executed_choice is None
            or self.planned_root_margin < 0.025
        ):
            self.last_outcome_rejection = {
                "reason": "missing_outcome_prerequisite",
                "outcomes": len(outcomes),
                "root_margin": self.planned_root_margin,
            }
            return None
        current_roots = {root.label: root for root in self.roots}
        best_by_world: dict[str, tuple[float, BranchOutcome]] = {}
        rejected = {
            "root_choice": 0,
            "missing_world": 0,
            "opponent_action": 0,
            "request_state": 0,
            "strategic_state": 0,
            "damage_state": 0,
        }
        for outcome in outcomes:
            if outcome.root_choice != self.last_executed_choice:
                rejected["root_choice"] += 1
                continue
            current = current_roots.get(outcome.root_label)
            parent = self.plan_parent_nodes.get(outcome.root_label)
            if current is None or parent is None:
                rejected["missing_world"] += 1
                continue
            if self.last_observed_actions and not choice_matches_observation(
                parent, "p2", outcome.opponent_choice, self.last_observed_actions
            ):
                rejected["opponent_action"] += 1
                continue
            if outcome.predicted_node.request_state != current.node.request_state:
                rejected["request_state"] += 1
                continue
            if _strategic_state_signature(
                outcome.predicted_node.state
            ) != _strategic_state_signature(current.node.state):
                rejected["strategic_state"] += 1
                continue
            error = _state_public_error(outcome.predicted_node.state, snapshot)
            if error > 0.16:
                rejected["damage_state"] += 1
                continue
            previous = best_by_world.get(outcome.root_label)
            # Prefer the closest public outcome, then the branch searched most deeply.
            quality = error - 0.001 * outcome.searched_depth
            if previous is None or quality < previous[0]:
                best_by_world[outcome.root_label] = (quality, outcome)
        if not best_by_world:
            self.last_outcome_rejection = {
                "reason": "no_evaluated_position_match",
                "outcomes": len(outcomes),
                "rejected": rejected,
            }
            return None

        covered = [root for root in self.roots if root.label in best_by_world]
        coverage = sum(root.probability for root in covered)
        if coverage < 0.50:
            self.last_outcome_rejection = {
                "reason": "insufficient_world_coverage",
                "coverage": coverage,
                "matched_worlds": len(covered),
                "rejected": rejected,
            }
            return None
        value = (
            sum(
                root.probability * best_by_world[root.label][1].value
                for root in covered
            )
            / coverage
        )
        # A branch can be expected yet bad (for example, the opponent found the one
        # response that swings the game). Search those positions again instead of
        # coasting merely because the simulator predicted them.
        if value < self.planned_reference_score - 0.18:
            self.last_outcome_rejection = {
                "reason": "matched_bad_branch",
                "predicted_value": value,
                "reference_score": self.planned_reference_score,
                "coverage": coverage,
            }
            return None
        self.last_outcome_rejection = None
        return {
            "coverage": coverage,
            "predicted_value": value,
            "reference_score": self.planned_reference_score,
            "matched_worlds": len(covered),
        }

    def _matching_pondered_outcome(self) -> dict[str, Any] | None:
        """Recognize a broad opponent response expanded after move submission."""
        self.last_ponder_rejection = None
        if (
            not self.enable_ponder
            or not self.pondered_outcomes
            or self.last_snapshot is None
            or self.last_executed_choice is None
        ):
            self.last_ponder_rejection = {
                "reason": "missing_ponder_prerequisite",
                "outcomes": len(self.pondered_outcomes),
                "last_ponder": self.last_ponder,
            }
            return None
        current_roots = {root.label: root for root in self.roots}
        best_by_world: dict[str, tuple[float, PonderOutcome]] = {}
        rejected = {
            "root_choice": 0,
            "missing_world": 0,
            "opponent_action": 0,
            "request_state": 0,
            "strategic_state": 0,
            "damage_state": 0,
        }
        for outcome in self.pondered_outcomes:
            if outcome.root_choice != self.last_executed_choice:
                rejected["root_choice"] += 1
                continue
            current = current_roots.get(outcome.root_label)
            parent = self.ponder_parent_nodes.get(outcome.root_label)
            if current is None or parent is None:
                rejected["missing_world"] += 1
                continue
            if self.last_observed_actions and not choice_matches_observation(
                parent, "p2", outcome.opponent_choice, self.last_observed_actions
            ):
                rejected["opponent_action"] += 1
                continue
            if outcome.predicted_node.request_state != current.node.request_state:
                rejected["request_state"] += 1
                continue
            if _strategic_state_signature(
                outcome.predicted_node.state
            ) != _strategic_state_signature(current.node.state):
                rejected["strategic_state"] += 1
                continue
            error = _state_public_error(
                outcome.predicted_node.state, self.last_snapshot
            )
            if error > 0.16:
                rejected["damage_state"] += 1
                continue
            previous = best_by_world.get(outcome.root_label)
            if previous is None or error < previous[0]:
                best_by_world[outcome.root_label] = (error, outcome)
        if not best_by_world:
            self.last_ponder_rejection = {
                "reason": "no_pondered_position_match",
                "outcomes": len(self.pondered_outcomes),
                "rejected": rejected,
                "observed_opponent_actions": {
                    str(slot): asdict(action)
                    for slot, action in self.last_observed_actions.items()
                },
                "pondered_opponent_choices": sorted(
                    {item.opponent_choice for item in self.pondered_outcomes}
                )[:16],
            }
            return None

        acceptable = []
        for root in self.roots:
            matched = best_by_world.get(root.label)
            reference = self.ponder_reference_values.get(root.label)
            if matched is None or reference is None:
                continue
            if matched[1].value >= reference - 0.06:
                acceptable.append(root)
        coverage = sum(root.probability for root in acceptable)
        if coverage < 0.50:
            self.last_ponder_rejection = {
                "reason": "pondered_branch_bad_or_low_coverage",
                "coverage": coverage,
                "matched_worlds": len(best_by_world),
                "acceptable_worlds": len(acceptable),
                "rejected": rejected,
            }
            return None
        value = (
            sum(
                root.probability * best_by_world[root.label][1].value
                for root in acceptable
            )
            / coverage
        )
        self.ponder_matches += 1
        self.last_ponder_rejection = None
        return {
            "coverage": coverage,
            "predicted_value": value,
            "matched_worlds": len(acceptable),
            "expanded_outcomes": len(self.pondered_outcomes),
        }

    def plan(self, battle: DoubleBattle) -> PlanResult | None:
        decision_started = time.monotonic()
        self.prepare(battle)
        preparation_elapsed_s = time.monotonic() - decision_started
        self.last_result = None
        reused = self._reuse_contingent_plan(battle)
        if reused is not None:
            self.last_result = reused
            return reused

        on_plan = self._matching_planned_outcome()
        if on_plan is not None:
            self.skipped_searches += 1
            self.last_schedule = {
                "mode": "skip_on_plan",
                "reasons": ["position_matches_acceptable_searched_branch"],
                **on_plan,
                "reuse_rejection": self.last_reuse_rejection,
            }
            self.planned_continuations = ()
            self.planned_outcomes = ()
            self.plan_parent_nodes = {}
            self.pondered_outcomes = ()
            self.ponder_parent_nodes = {}
            self.ponder_reference_values = {}
            return None

        on_ponder = self._matching_pondered_outcome()
        if on_ponder is not None:
            self.skipped_searches += 1
            self.last_schedule = {
                "mode": "skip_on_ponder",
                "reasons": ["position_matches_background_expansion"],
                **on_ponder,
                "ponder": self.last_ponder,
            }
            self.pondered_outcomes = ()
            self.ponder_parent_nodes = {}
            self.ponder_reference_values = {}
            self.planned_continuations = ()
            self.planned_outcomes = ()
            self.plan_parent_nodes = {}
            return None

        reasons = self._importance_reasons(battle)
        planned_action_coverage = self._planned_opponent_action_coverage()
        planned_family_coverage = self._planned_opponent_family_coverage()
        planned_position_coverage = self._planned_position_coverage()
        # If the observed move and every strategic consequence match a searched
        # successor in most representative worlds, this is the chess-style quiet
        # continuation the scheduler is meant to recognize. Do not suppress events
        # absent from the strategic signature (new set information or Mega/Tera), a
        # forced replacement, or the periodic refresh safety valve.
        always_research = {
            "first_decision",
            "forced_replacement",
            "hidden_information_revealed",
            "major_mechanic",
            "periodic_refresh",
        }
        if (
            self.selective_search
            and self._expected_action_is_quiet(reasons, planned_family_coverage)
        ):
            self.skipped_searches += 1
            self.last_schedule = {
                "mode": "skip_on_plan",
                "reasons": ["searched_action_produced_only_expected_soft_updates"],
                "planned_action_coverage": planned_action_coverage,
                "planned_family_coverage": planned_family_coverage,
                "planned_position_coverage": planned_position_coverage,
                "observed_soft_updates": reasons,
            }
            self.planned_continuations = ()
            self.planned_outcomes = ()
            self.plan_parent_nodes = {}
            self.pondered_outcomes = ()
            self.ponder_parent_nodes = {}
            self.ponder_reference_values = {}
            return None
        if (
            self.selective_search
            and planned_position_coverage >= 0.75
            and not always_research.intersection(reasons)
        ):
            self.skipped_searches += 1
            self.last_schedule = {
                "mode": "skip_on_plan",
                "reasons": ["strategic_position_matches_searched_successor"],
                "planned_action_coverage": planned_action_coverage,
                "planned_family_coverage": planned_family_coverage,
                "planned_position_coverage": planned_position_coverage,
            }
            self.planned_continuations = ()
            self.planned_outcomes = ()
            self.plan_parent_nodes = {}
            self.pondered_outcomes = ()
            self.ponder_parent_nodes = {}
            self.ponder_reference_values = {}
            return None
        if self.selective_search and (
            self.planned_continuations
            or self.planned_outcomes
            or self.pondered_outcomes
        ) and self.last_observed_actions and planned_family_coverage < 0.50:
            # The opponent actually chose a move/switch absent from most searched
            # hidden worlds. Ordinary damage-roll drift no longer triggers this.
            reasons.append("unplanned_opponent_action")
        if self.selective_search and not reasons:
            self.skipped_searches += 1
            self.last_schedule = {
                "mode": "skip",
                "reasons": ["quiet_position_use_champion_and_guards"],
                "planned_action_coverage": planned_action_coverage,
                "planned_family_coverage": planned_family_coverage,
                "planned_position_coverage": planned_position_coverage,
            }
            self.planned_continuations = ()
            self.planned_outcomes = ()
            self.plan_parent_nodes = {}
            self.pondered_outcomes = ()
            self.ponder_parent_nodes = {}
            self.ponder_reference_values = {}
            return None

        # Keep every particle alive for evidence updates and reconciliation, but put
        # the finite move clock into a small, spread-out subset. Searching all eight
        # roots sequentially produced eight shallow answers and almost never reached
        # a second move turn. Evenly spaced indexes preserve brought-four diversity;
        # selected masses are renormalized only for this foreground calculation.
        # A reserve reveal can invalidate most prior brought-four roots at once. The
        # rebuilt worlds are already evidence-conditioned, but splitting their first
        # turn across two short slices occasionally lets neither return a usable
        # answer. Concentrate that one turn on the highest-posterior rebuilt world;
        # retain all beliefs and return to the normal representative count next turn.
        search_determinizations = self._foreground_determinizations(battle.turn)
        if len(self.roots) <= search_determinizations:
            planning_roots = list(self.roots)
        elif search_determinizations == 1:
            planning_roots = [max(self.roots, key=lambda root: root.probability)]
        else:
            indexes = {
                round(index * (len(self.roots) - 1) / (search_determinizations - 1))
                for index in range(search_determinizations)
            }
            planning_roots = [self.roots[index] for index in sorted(indexes)]
        planning_mass = sum(root.probability for root in planning_roots)
        weighted = [
            WeightedExactNode(root.node, root.probability / planning_mass, root.label)
            for root in planning_roots
        ]
        planning_budget_s = self.config.time_budget_s - preparation_elapsed_s
        if planning_budget_s < 0.10:
            self.last_result = None
            self.skipped_searches += 1
            self.last_schedule = {
                "mode": "search_fallback",
                "reasons": ["snapshot_preparation_consumed_search_budget"],
                "preparation_elapsed_s": preparation_elapsed_s,
                "planning_budget_s": max(0.0, planning_budget_s),
                "planning_roots": len(planning_roots),
                "belief_roots": len(self.roots),
            }
            self.planned_continuations = ()
            self.planned_outcomes = ()
            self.plan_parent_nodes = {}
            return None
        effective_config = replace(
            self.config,
            time_budget_s=planning_budget_s,
            screen_budget_s=min(self.config.screen_budget_s, planning_budget_s),
        )
        planner = ExactDeterminizationPlanner(
            self.bridge,
            prior=self.prior,
            evaluator=self.evaluator,
            config=effective_config,
        )
        result = planner.plan(
            weighted,
            "p1",
            minimum_depth_coverage=self.min_deep_coverage,
        )
        if result.selected_depth_coverage + 1e-9 < self.min_deep_coverage:
            self.last_result = None
            self.skipped_searches += 1
            self.last_schedule = {
                "mode": "search_fallback",
                "reasons": ["selected_action_did_not_reach_required_depth"],
                "selected_depth_coverage": result.selected_depth_coverage,
                "required_depth_coverage": self.min_deep_coverage,
                "planning_roots": len(planning_roots),
                "belief_roots": len(self.roots),
                "fallback_reason": result.fallback_reason,
                "preparation_elapsed_s": preparation_elapsed_s,
                "planning_budget_s": planning_budget_s,
            }
            self.planned_continuations = ()
            self.planned_outcomes = ()
            self.plan_parent_nodes = {}
            return None
        # The exact clone and the live poke-env battle can store the same four
        # Pokemon in different dictionary/party orders after switching. Planner
        # choices are canonical Showdown strings; convert those strings against the
        # *live* battle immediately before submission instead of returning action IDs
        # encoded against a reconstructed clone.
        rankings, illegal_choices = self._live_legal_rankings(result, battle)
        winner = rankings[0]
        selected_changed = winner.choice != result.choice
        legality_changed = result.choice in illegal_choices
        if not self._reaches_required_depth(winner):
            # The aggregate winner reached depth before live legality and factual
            # guards ran. If those checks select a different shallow row, it has not
            # earned the right to replace the champion merely because the original
            # searched line was unsafe. Let champion-plus-guards choose among the
            # safe actions instead.
            self.last_result = None
            self.skipped_searches += 1
            self.last_schedule = {
                "mode": "search_fallback",
                "reasons": ["live_safe_action_did_not_reach_required_depth"],
                "selected_choice": winner.choice,
                "selected_depth_coverage": winner.depth_coverage,
                "required_depth_coverage": self.min_deep_coverage,
                "planning_roots": len(planning_roots),
                "belief_roots": len(self.roots),
                "fallback_reason": result.fallback_reason,
                "preparation_elapsed_s": preparation_elapsed_s,
                "planning_budget_s": planning_budget_s,
                "live_illegal_choices": illegal_choices,
                "live_legality_changed_pick": legality_changed,
                "live_safety_changed_pick": selected_changed,
                "live_guards": self.last_live_guards,
            }
            self.planned_continuations = ()
            self.planned_outcomes = ()
            self.plan_parent_nodes = {}
            return None
        low_prior_rejection = self._low_prior_override_rejection(
            battle, winner, rankings
        )
        if low_prior_rejection is not None:
            self.last_result = None
            self.skipped_searches += 1
            self.last_schedule = {
                "mode": "search_fallback",
                "reasons": [str(low_prior_rejection["reason"])],
                "low_prior_rejection": low_prior_rejection,
                "planning_roots": len(planning_roots),
                "belief_roots": len(self.roots),
                "fallback_reason": result.fallback_reason,
                "preparation_elapsed_s": preparation_elapsed_s,
                "planning_budget_s": planning_budget_s,
                "live_illegal_choices": illegal_choices,
                "live_legality_changed_pick": legality_changed,
                "live_safety_changed_pick": selected_changed,
                "live_guards": self.last_live_guards,
            }
            self.planned_continuations = ()
            self.planned_outcomes = ()
            self.plan_parent_nodes = {}
            return None
        self.last_result = replace(
            result,
            choice=winner.choice,
            actions=winner.actions,
            score=winner.score,
            rankings=rankings,
            selected_depth_coverage=winner.depth_coverage,
            fallback_reason=(
                "live_legality_filter"
                if legality_changed
                else "live_safety_filter"
                if selected_changed
                else result.fallback_reason
            ),
        )
        self.plan_parent_nodes = {root.label: root.node for root in self.roots}
        self.planned_continuations = planner.continuations
        self.planned_outcomes = planner.outcomes
        self.planned_root_margin = self._root_margin(self.last_result)
        self.planned_reference_score = self.last_result.score
        self.pondered_outcomes = ()
        self.ponder_parent_nodes = {}
        self.ponder_reference_values = {}
        self.last_search_turn = int(battle.turn)
        self.full_searches += 1
        self.last_schedule = {
            "mode": "search",
            "reasons": sorted(set(reasons)),
            "root_margin": self.planned_root_margin,
            "saved_continuations": len(self.planned_continuations),
            "saved_outcomes": len(self.planned_outcomes),
            "reuse_rejection": self.last_reuse_rejection,
            "outcome_rejection": self.last_outcome_rejection,
            "ponder_rejection": self.last_ponder_rejection,
            "live_illegal_choices": illegal_choices,
            "live_legality_changed_pick": legality_changed,
            "live_safety_changed_pick": selected_changed,
            "live_guards": self.last_live_guards,
            "selected_depth_coverage": self.last_result.selected_depth_coverage,
            "planning_roots": len(planning_roots),
            "belief_roots": len(self.roots),
            "planned_action_coverage": planned_action_coverage,
            "planned_family_coverage": planned_family_coverage,
            "planned_position_coverage": planned_position_coverage,
            "preparation_elapsed_s": preparation_elapsed_s,
            "planning_budget_s": planning_budget_s,
        }
        return self.last_result

    @staticmethod
    def _strict_live_rejection_reason(
        battle: DoubleBattle, actions: tuple[int, int]
    ) -> str | None:
        """Reject factual no-effect actions even if a learned value likes them.

        The ordinary guard stack demotes rather than deletes. A searched continuation
        contains only one row, however, so that contract would stand down and permit a
        second Tailwind while Tailwind is already active. This check is intentionally
        limited to side conditions that mechanically fail; Trick Room is excluded
        because using it again toggles the room off.
        """
        for pos, action in enumerate(actions):
            try:
                order = DoublesEnv._action_to_order_individual(
                    np.int64(action), battle, fake=True, pos=pos
                )
            except Exception:
                continue
            move = getattr(order, "order", None)
            condition = getattr(move, "side_condition", None)
            if condition is not None and condition in getattr(
                battle, "side_conditions", {}
            ):
                return "redundant_side_condition"
        # Reused continuations contain only one candidate, so the ordinary guard
        # contract would stand down. Reject these two narrow ladder regressions here
        # as well: first-use Protect into a revealed faster Encore, and no-weather
        # Weather Ball when legal Heat Wave is overwhelmingly stronger.
        try:
            from vgc_bench.src.guards import (
                Candidate,
                battle_can_contest_catastrophic_setup,
                candidate_exposes_protect_to_encore,
                candidate_gives_catastrophic_free_setup,
                candidate_repeats_solo_protect,
                candidate_uses_blocked_priority,
                candidate_uses_dominated_single_target_heat_wave,
                candidate_uses_dominated_weather_ball,
                solo_has_attack_progress,
            )

            candidate = Candidate(tuple(int(action) for action in actions), 1.0)
            if candidate_uses_blocked_priority(battle, candidate):
                return "blocked_priority"
            if candidate_gives_catastrophic_free_setup(
                battle, candidate
            ) and battle_can_contest_catastrophic_setup(battle):
                return "free_catastrophic_setup"
            if candidate_repeats_solo_protect(
                battle, candidate
            ) and solo_has_attack_progress(battle):
                return "solo_repeated_protect"
            if candidate_exposes_protect_to_encore(battle, candidate):
                return "encore_exposure"
            if candidate_uses_dominated_weather_ball(battle, candidate):
                return "dominated_weather_ball"
            if candidate_uses_dominated_single_target_heat_wave(battle, candidate):
                return "single_target_weather_ball"
        except Exception:
            # A tactical check must never jeopardize submitting a legal move. The
            # ordinary multi-candidate hard-guard pass still gets another chance.
            pass
        return None

    def _apply_live_hard_guards(
        self, battle: DoubleBattle, rankings: Sequence[ActionScore]
    ) -> tuple[ActionScore, ...]:
        """Put the production factual guard stack after exact outcome scoring."""
        from vgc_bench.src.guards import GUARDS, HARD_GUARDS, Candidate, apply_guards

        before = rankings[0].choice
        candidates = [
            Candidate(
                actions=tuple(int(action) for action in row.actions),
                prob=max(0.0, float(row.prior)),
            )
            for row in rankings
        ]
        enabled = {name: name in HARD_GUARDS for name in GUARDS}
        guarded, report = apply_guards(battle, candidates, enabled)
        by_actions = {
            tuple(int(action) for action in row.actions): row for row in rankings
        }
        ordered = tuple(by_actions[candidate.actions] for candidate in guarded)
        self.last_live_guards = {
            "changed_pick": ordered[0].choice != before,
            "stages": list(report.stages),
            "demotions": dict(report.demotions),
            "strict_rejections": self.last_live_guards.get("strict_rejections", []),
        }
        return ordered

    def _live_legal_rankings(
        self, result: PlanResult, battle: DoubleBattle
    ) -> tuple[tuple[ActionScore, ...], list[str]]:
        """Map ranked exact choices, reject factual failures, then run hard guards."""
        mapped: list[ActionScore] = []
        illegal: list[str] = []
        strict_rejections: list[str] = []
        for score in result.rankings:
            try:
                actions = self._live_actions(score.choice, battle)
            except Exception:
                illegal.append(score.choice)
                continue
            reason = self._strict_live_rejection_reason(battle, actions)
            if reason is not None:
                strict_rejections.append(f"{score.choice} [{reason}]")
                continue
            mapped.append(replace(score, actions=actions))
        if not mapped:
            raise ValueError(
                "exact search produced no live-safe ranked choice; rejected "
                + ", ".join((illegal + strict_rejections)[:6])
            )
        self.last_live_guards = {
            "changed_pick": False,
            "stages": [],
            "demotions": {},
            "strict_rejections": strict_rejections,
        }
        guarded = self._apply_live_hard_guards(battle, tuple(mapped))
        return guarded, illegal

    def _live_actions(self, choice: str, battle: DoubleBattle) -> tuple[int, int]:
        errors: list[str] = []
        for root in self.roots:
            if choice not in self.bridge.choices(root.node.state, "p1"):
                continue
            try:
                actions = self._root_live_actions(root, choice, battle)
                # A reconciled clone may still retain stale trapping or party-order
                # information. Never trust a simulator-legal switch until poke-env
                # confirms that the exact pair is legal in the live request.
                DoublesEnv.action_to_order(
                    np.asarray(actions, dtype=np.int64), battle, fake=False, strict=True
                )
                return actions
            except Exception as exc:
                errors.append(f"{root.label}: {type(exc).__name__}: {exc}")
        raise ValueError(
            f"exact choice cannot round-trip to live action: {choice}; "
            + "; ".join(errors[:3])
        )

    @staticmethod
    def _action_pair_is_live_legal(
        actions: Sequence[int], battle: DoubleBattle
    ) -> bool:
        try:
            DoublesEnv.action_to_order(
                np.asarray(actions, dtype=np.int64), battle, fake=False, strict=True
            )
            return True
        except (AssertionError, IndexError, TypeError, ValueError):
            return False

    @staticmethod
    def _root_live_actions(
        root: LiveRoot, choice: str, battle: DoubleBattle
    ) -> tuple[int, int]:
        request = root.node.requests[0]
        if request is None:
            raise ValueError("exact root has no controlled-side request")
        return choice_to_actions(
            choice, request, state=root.node.state, role="p1", battle=battle
        )

    def record_actions(self, actions: Sequence[int]) -> None:
        wanted = tuple(int(action) for action in actions)
        if self.last_result is not None and self.last_result.actions == wanted:
            self.pending_our_choice = self.last_result.choice
            self.last_executed_choice = self.last_result.choice
            self._pending_ponder_choice = self.last_result.choice
            if self.current_battle is not None and not any(
                self.current_battle.force_switch
            ):
                self.last_submitted_move_choice = self.last_result.choice
            return
        if self.current_battle is None:
            self.pending_our_choice = None
            self.fallbacks += 1
            return
        for root in self.roots:
            legal = self.bridge.choices(root.node.state, "p1")
            match = None
            for choice in legal:
                try:
                    mapped = self._root_live_actions(root, choice, self.current_battle)
                except Exception:
                    continue
                if mapped == wanted:
                    match = choice
                    break
            if match is not None:
                self.pending_our_choice = match
                self.last_executed_choice = match
                self._pending_ponder_choice = match
                if not any(self.current_battle.force_switch):
                    self.last_submitted_move_choice = match
                return
        self.pending_our_choice = None
        self.last_executed_choice = None
        self._pending_ponder_choice = None
        self.fallbacks += 1

    def audit(self) -> dict[str, Any]:
        result = self.last_result
        return {
            "backend": "live-exact-showdown",
            "open_sheet": self.open_sheet,
            "determinizations": len(self.roots),
            "search_determinizations": self.search_determinizations,
            "min_deep_coverage": self.min_deep_coverage,
            "low_prior_override_ratio": self.low_prior_override_ratio,
            "low_prior_override_min_coverage": (
                self.low_prior_override_min_coverage
            ),
            "configuration": asdict(self.config),
            "selective_search": self.selective_search,
            "ponder_enabled": self.enable_ponder,
            "ponder_configuration": asdict(self.ponder_config),
            "schedule": self.last_schedule,
            "full_searches": self.full_searches,
            "reused_plans": self.reused_plans,
            "skipped_searches": self.skipped_searches,
            "saved_continuations": len(self.planned_continuations),
            "saved_outcomes": len(self.planned_outcomes),
            "pondered_outcomes": len(self.pondered_outcomes),
            "ponder_started": self.ponder_started,
            "ponder_completed": self.ponder_completed,
            "ponder_partial": self.ponder_partial,
            "ponder_late": self.ponder_late,
            "ponder_errors": self.ponder_errors,
            "ponder_matches": self.ponder_matches,
            "last_ponder": self.last_ponder,
            "fallbacks": self.fallbacks,
            "root_reconcile_failures": self.root_reconcile_failures,
            "reconciliations": self.reconciliations,
            "root_refreshes": self.root_refreshes,
            "eliminated_roots": self.eliminated_roots,
            "last_reconcile_errors": self.last_reconcile_errors[-8:],
            "result": (
                {
                    "choice": result.choice,
                    "score": result.score,
                    "nodes": result.nodes,
                    "elapsed_s": result.elapsed_s,
                    "completed_depth": result.completed_depth,
                    "truncated": result.truncated,
                    "screened_actions": result.screened_actions,
                    "deepened_actions": result.deepened_actions,
                    "deepened_choices": list(result.deepened_choices),
                    "selected_depth_coverage": result.selected_depth_coverage,
                    "fallback_reason": result.fallback_reason,
                    "rankings": [
                        {
                            "choice": score.choice,
                            "actions": score.actions,
                            "score": score.score,
                            "expected": score.expected,
                            "cvar": score.cvar,
                            "worst": score.worst,
                            "prior": score.prior,
                            "depth_coverage": score.depth_coverage,
                        }
                        for score in result.rankings[:8]
                    ],
                }
                if result is not None
                else None
            ),
            "recorded_at": time.time(),
        }


def append_exact_audit(path: Path | None, battle: DoubleBattle, payload: dict) -> None:
    if path is None:
        return
    row = {
        "battle": battle.battle_tag,
        "turn": int(battle.turn),
        "exact_search": payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
