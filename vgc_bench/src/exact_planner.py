"""Risk-aware multi-turn planning over exact Pokemon Showdown states.

Every edge is resolved by Pokemon Showdown itself.  The learned policy supplies an
action prior and a leaf value, but it no longer gets to assume that a superficially
common move is good: the planner ranks the concrete positions produced several turns
later against plausible opponent responses.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, replace
from typing import Any, Protocol, Sequence

from poke_env.battle import (
    Effect,
    Field,
    Move,
    MoveCategory,
    SideCondition,
    Status,
    Weather,
)

from vgc_bench.src import vgc_knowledge as K
from vgc_bench.src.exact_observation import (
    ExactPolicyAdapter,
    RankedChoice,
    state_to_battle,
)
from vgc_bench.src.exact_sim import ExactShowdownBridge, ExactSimulatorError
from vgc_bench.src.tempo_reranker import speed_control_snapshot


@dataclass(frozen=True)
class ExactNode:
    state: dict[str, Any]
    requests: list[dict[str, Any] | None]
    turn: int
    request_state: str
    ended: bool = False
    winner: str | None = None

    @classmethod
    def from_result(cls, result: dict[str, Any]) -> "ExactNode":
        return cls(
            state=result["state"],
            requests=result.get("requests") or [None, None],
            turn=int(result.get("turn") or 0),
            request_state=str(result.get("request_state") or ""),
            ended=bool(result.get("ended")),
            winner=result.get("winner"),
        )


class Prior(Protocol):
    def rank(
        self,
        state: dict[str, Any],
        requests: list[dict[str, Any] | None],
        role: str,
        choices: list[str],
    ) -> list[RankedChoice]: ...


class Evaluator(Protocol):
    def __call__(self, node: ExactNode, role: str) -> float: ...


class DeterminizationBudgetExhausted(ValueError):
    """No determinization returned before the planner's shared deadline."""


@dataclass(frozen=True)
class PlannerConfig:
    """Search shape and risk appetite.

    ``depth`` counts completed move turns. Forced replacements do not consume depth.
    Width is deliberately larger at the root and narrows in continuation positions.
    """

    depth: int = 2
    root_width: int = 6
    opponent_width: int = 6
    continuation_width: int = 3
    replacement_width: int = 2
    chance_samples: int = 1
    volatile_chance_samples: int = 2
    volatile_accuracy_threshold: float = 0.90
    pre_move_ko_penalty: float = 0.22
    expected_weight: float = 0.60
    cvar_weight: float = 0.30
    worst_weight: float = 0.10
    cvar_alpha: float = 0.25
    discount: float = 0.995
    time_budget_s: float = 9.0
    max_nodes: int = 5000
    anytime: bool = False
    screen_budget_s: float = 2.0
    screen_opponent_width: int = 3
    deep_root_width: int = 2

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("planner depth must be at least one turn")
        if self.chance_samples < 1:
            raise ValueError("chance_samples must be positive")
        if self.volatile_chance_samples < 1:
            raise ValueError("volatile chance samples must be positive")
        if not 0 < self.volatile_accuracy_threshold <= 1:
            raise ValueError("volatile accuracy threshold must be in (0, 1]")
        if not 0 <= self.pre_move_ko_penalty <= 1:
            raise ValueError("pre-move KO penalty must be in [0, 1]")
        # Move turns must stay under the 10s ladder cap (entry points enforce
        # <= 9 for move search). Team Preview worlds may legitimately run longer:
        # the VGC Timer grants 90 seconds at the first request.
        if not 0 < self.time_budget_s <= 60.0:
            raise ValueError("exact search budget must be in (0, 60] seconds")
        if not 0 < self.screen_budget_s <= self.time_budget_s:
            raise ValueError("screen budget must be positive and within search budget")
        if self.screen_opponent_width < 1 or self.deep_root_width < 1:
            raise ValueError("anytime widths must be positive")
        if not math.isclose(
            self.expected_weight + self.cvar_weight + self.worst_weight,
            1.0,
            abs_tol=1e-6,
        ):
            raise ValueError("risk weights must sum to one")


@dataclass(frozen=True)
class ActionScore:
    choice: str
    actions: tuple[int, int] | None
    score: float
    expected: float
    cvar: float
    worst: float
    standard_deviation: float
    prior: float
    opponent_branches: int
    depth_coverage: float = 0.0


@dataclass(frozen=True)
class PlanResult:
    choice: str
    actions: tuple[int, int] | None
    score: float
    rankings: tuple[ActionScore, ...]
    nodes: int
    elapsed_s: float
    completed_depth: int
    truncated: bool
    screened_actions: int = 0
    deepened_actions: int = 0
    fallback_reason: str | None = None
    deepened_choices: tuple[str, ...] = ()
    selected_depth_coverage: float = 0.0


@dataclass(frozen=True)
class BranchContinuation:
    """One next-request action discovered while deepening a root branch.

    The live player uses these as a contingent plan.  A continuation is never
    submitted merely because it was searched: the next public position, observed
    opponent action, legality, and agreement across determinizations/RNG samples must
    still match first.
    """

    root_choice: str
    opponent_choice: str
    next_choice: str
    value: float
    margin: float
    probability: float
    predicted_node: ExactNode
    root_label: str = ""


@dataclass(frozen=True)
class BranchOutcome:
    """A successor position that the root search explicitly evaluated."""

    root_choice: str
    opponent_choice: str
    value: float
    probability: float
    predicted_node: ExactNode
    searched_depth: int
    root_label: str = ""


@dataclass(frozen=True)
class WeightedExactNode:
    """One concrete hidden-set/RNG-compatible state and its posterior mass."""

    node: ExactNode
    probability: float
    label: str = ""


def _hp_fraction(pokemon: dict[str, Any]) -> float:
    maximum = float(pokemon.get("maxhp") or pokemon.get("baseMaxhp") or 1)
    return max(0.0, min(1.0, float(pokemon.get("hp") or 0) / maximum))


def material_value(node: ExactNode, role: str) -> float:
    """Mechanics-independent fallback value in [-1, 1]."""
    ours_index = int(role[1]) - 1
    theirs_index = 1 - ours_index

    def side_features(index: int) -> tuple[float, float, float, float]:
        pokemon = node.state["sides"][index].get("pokemon", [])
        alive = [
            p for p in pokemon if not p.get("fainted") and float(p.get("hp") or 0) > 0
        ]
        hp = sum(_hp_fraction(p) for p in pokemon)
        status = sum(bool(p.get("status")) for p in alive)
        active_boosts = 0.0
        for p in alive:
            if not p.get("isActive"):
                continue
            boosts = p.get("boosts") or {}
            active_boosts += sum(
                float(boosts.get(stat) or 0)
                for stat in ("atk", "def", "spa", "spd", "spe")
            )
        return float(len(alive)), hp, float(status), active_boosts

    oa, oh, os, ob = side_features(ours_index)
    ta, th, ts, tb = side_features(theirs_index)
    team_size = max(
        1.0, float(max(len(side.get("pokemon", [])) for side in node.state["sides"]))
    )
    raw = (
        0.58 * (oa - ta) / team_size
        + 0.32 * (oh - th) / team_size
        - 0.04 * (os - ts) / team_size
        + 0.06 * math.tanh((ob - tb) / 4.0)
    )
    return float(math.tanh(1.8 * raw))


_SCREENS = {
    SideCondition.AURORA_VEIL,
    SideCondition.LIGHT_SCREEN,
    SideCondition.REFLECT,
}
_WEATHER_ABILITIES = {
    "rain": {"swiftswim", "raindish", "hydration", "dryskin"},
    "sun": {"chlorophyll", "solarpower", "harvest", "leafguard"},
    "sand": {"sandrush", "sandforce", "sandveil"},
    "snow": {"slushrush", "icebody", "snowcloak"},
}


def _norm(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def speed_position_value(battle) -> float:
    """Current equal-priority move-order advantage in [-1, 1].

    ``speed_control_snapshot`` reports which side is slower after Tailwind, weather,
    items, abilities, paralysis, and boosts. Slower is good only while Trick Room is
    active, so this explicit inversion prevents Tailwind+Trick Room from being treated
    as two independently beneficial flags.
    """
    snapshot = speed_control_snapshot(battle)
    if not snapshot.has_speed_evidence:
        return 0.0
    room = Field.TRICK_ROOM in battle.fields
    return float(
        snapshot.trick_room_advantage
        if room
        else -snapshot.trick_room_advantage
    )


def _status_burden(mon) -> float:
    status = mon.status
    burden = 0.0
    if status in {Status.SLP, Status.FRZ}:
        burden = 1.0
    elif status == Status.PAR:
        burden = 0.45
    elif status in {Status.BRN, Status.PSN, Status.TOX}:
        burden = 0.30
    if Effect.YAWN in getattr(mon, "effects", {}):
        burden = max(burden, 0.75)
    return burden


def _status_position_value(battle) -> float:
    def burden(team) -> float:
        alive = [mon for mon in team.values() if not mon.fainted]
        if not alive:
            return 0.0
        return sum(_status_burden(mon) for mon in alive) / len(alive)

    return max(-1.0, min(1.0, burden(battle.opponent_team) - burden(battle.team)))


def _screen_position_value(battle) -> float:
    ours = sum(condition in battle.side_conditions for condition in _SCREENS)
    theirs = sum(
        condition in battle.opponent_side_conditions for condition in _SCREENS
    )
    return (ours - theirs) / len(_SCREENS)


def _weather_name(battle) -> str | None:
    active = set(battle.weather)
    if active & {Weather.RAINDANCE, Weather.PRIMORDIALSEA}:
        return "rain"
    if active & {Weather.SUNNYDAY, Weather.DESOLATELAND}:
        return "sun"
    if Weather.SANDSTORM in active:
        return "sand"
    if active & {Weather.HAIL, Weather.SNOWSCAPE}:
        return "snow"
    return None


def _weather_position_value(battle) -> float:
    weather = _weather_name(battle)
    if weather is None:
        return 0.0
    useful = _WEATHER_ABILITIES[weather]

    def synergy(active) -> float:
        mons = [mon for mon in active if mon is not None and not mon.fainted]
        if not mons:
            return 0.0
        return sum(_norm(mon.ability) in useful for mon in mons) / len(mons)

    return synergy(battle.active_pokemon) - synergy(battle.opponent_active_pokemon)


def _side_pressure(battle, attackers, defenders) -> float:
    live_attackers = [mon for mon in attackers if mon is not None and not mon.fainted]
    live_defenders = [mon for mon in defenders if mon is not None and not mon.fainted]
    if not live_attackers or not live_defenders:
        return 0.0
    pressure = 0.0
    for defender in live_defenders:
        current_hp = max(0.01, float(defender.current_hp_fraction or 0.0))
        best = 0.0
        guaranteed = False
        for attacker in live_attackers:
            for move in attacker.moves.values():
                if move.category == MoveCategory.STATUS or move.base_power <= 0:
                    continue
                fraction = K.damage_fraction(battle, attacker, defender, move)
                if fraction is None:
                    continue
                accuracy = move.accuracy
                hit_probability = (
                    float(accuracy)
                    if isinstance(accuracy, (int, float))
                    and not isinstance(accuracy, bool)
                    else 1.0
                )
                expected = (fraction[0] + fraction[1]) * 0.5 * hit_probability
                best = max(best, expected / current_hp)
                guaranteed |= fraction[0] >= current_hp and hit_probability >= 1.0
        pressure += min(1.0, best) + (0.20 if guaranteed else 0.0)
    return pressure / (1.20 * len(live_defenders))


def threat_position_value(battle) -> float:
    """Difference in immediate damage/KO pressure for the current active board."""
    ours = _side_pressure(
        battle, battle.active_pokemon, battle.opponent_active_pokemon
    )
    theirs = _side_pressure(
        battle, battle.opponent_active_pokemon, battle.active_pokemon
    )
    return max(-1.0, min(1.0, ours - theirs))


def strategic_value(node: ExactNode, role: str) -> float:
    """Mechanics-aware leaf value independent of the learned critic."""
    fallback = material_value(node, role)
    if any(request is None for request in node.requests):
        return fallback
    battle = state_to_battle(
        node.state,
        node.requests,
        role,
        # A leaf is one sampled determinization. The student still receives a hidden
        # observation when sheets are closed, but the teacher must judge the concrete
        # threats that the exact simulator actually sampled.
        reveal_opponent_sets=True,
    )
    raw = (
        0.52 * fallback
        + 0.21 * threat_position_value(battle)
        + 0.13 * speed_position_value(battle)
        + 0.07 * _status_position_value(battle)
        + 0.04 * _screen_position_value(battle)
        + 0.03 * _weather_position_value(battle)
    )
    return float(math.tanh(1.35 * raw))


class HybridEvaluator:
    """Blend the learned critic with an independent mechanics-aware evaluator."""

    def __init__(self, policy: ExactPolicyAdapter | None, critic_weight: float = 0.45):
        if not 0 <= critic_weight <= 1:
            raise ValueError("critic_weight must be in [0, 1]")
        self.policy = policy
        self.critic_weight = critic_weight
        self._cache: dict[tuple[str, str], float] = {}

    @staticmethod
    def _key(node: ExactNode, role: str) -> tuple[str, str]:
        inputs = node.state.get("inputLog") or []
        digest = hashlib.blake2b(
            ("\n".join(inputs) + str(node.state.get("prng"))).encode(), digest_size=12
        ).hexdigest()
        return role, digest

    def __call__(self, node: ExactNode, role: str) -> float:
        key = self._key(node, role)
        if key in self._cache:
            return self._cache[key]
        fallback = material_value(node, role)
        try:
            mechanics = strategic_value(node, role)
        except (
            AssertionError,
            KeyError,
            NotImplementedError,
            RuntimeError,
            ValueError,
        ):
            mechanics = fallback
        if self.policy is None or any(request is None for request in node.requests):
            value = mechanics
        else:
            try:
                critic = max(
                    -1.0, min(1.0, self.policy.value(node.state, node.requests, role))
                )
                value = (
                    self.critic_weight * critic + (1 - self.critic_weight) * mechanics
                )
            except (AssertionError, KeyError, NotImplementedError, ValueError):
                value = mechanics
        self._cache[key] = float(value)
        return float(value)


class UniformPrior:
    def rank(self, state, requests, role, choices):
        probability = 1.0 / max(1, len(choices))
        return [RankedChoice(choice, None, probability) for choice in choices]


def _normalise(items: list[RankedChoice]) -> list[RankedChoice]:
    total = sum(max(0.0, item.probability) for item in items)
    if total <= 0:
        probability = 1.0 / max(1, len(items))
        return [RankedChoice(x.choice, x.actions, probability) for x in items]
    return [
        RankedChoice(x.choice, x.actions, max(0.0, x.probability) / total)
        for x in items
    ]


def _atom_signature(atom: str) -> str:
    parts = atom.strip().split()
    if not parts:
        return "pass"
    if parts[0] == "move" and len(parts) > 1:
        return f"move:{parts[1]}"
    if parts[0] == "switch" and len(parts) > 1:
        return f"switch:{parts[1]}"
    return parts[0]


def _choice_signatures(item: RankedChoice) -> set[tuple[int, str]]:
    atoms = [atom.strip() for atom in item.choice.split(",")]
    while len(atoms) < 2:
        atoms.append("pass")
    return {
        (slot, _atom_signature(atom)) for slot, atom in enumerate(atoms[:2])
    }


def _diverse_prefix(
    ranked: list[RankedChoice], width: int, *, guarantee_moves: bool = True
) -> list[RankedChoice]:
    """Guarantee every legal move family before filling with policy favorites.

    A pure top-k inherited the exact failure search is meant to repair: if Earthquake
    has low policy probability, the planner never simulates Earthquake and therefore
    cannot discover that it wins. Coverage is per active slot, so Dragon Claw from
    slot zero and Dragon Claw from slot one are distinct requirements. If the nominal
    width is too small, move coverage is allowed to exceed it rather than silently
    dropping a legal move. Targets and Mega variants share the same move family.
    """
    if len(ranked) <= width or ranked[0].choice.startswith("team "):
        return ranked[:width]

    signatures = {id(item): _choice_signatures(item) for item in ranked}
    all_tokens = set().union(*signatures.values())
    uncovered_moves = {
        token for token in all_tokens if token[1].startswith("move:")
    }
    selected: list[RankedChoice] = []
    selected_ids: set[int] = set()

    def add(item: RankedChoice) -> None:
        if id(item) not in selected_ids:
            selected.append(item)
            selected_ids.add(id(item))

    def add_best_covering(uncovered: set[tuple[int, str]]) -> bool:
        candidates = [item for item in ranked if id(item) not in selected_ids]
        if not candidates:
            return False
        best = max(
            candidates,
            key=lambda item: (
                len(signatures[id(item)] & uncovered),
                item.probability,
            ),
        )
        covered = signatures[id(best)] & uncovered
        if not covered:
            return False
        add(best)
        uncovered.difference_update(covered)
        return True

    # Start with compact set coverage so screening still leaves time for depth.
    while (
        uncovered_moves
        and (guarantee_moves or len(selected) < width)
        and add_best_covering(uncovered_moves)
    ):
        pass

    # Repair the pathological set-cover case: a low-probability Earthquake + bad
    # partner can win coverage because it also mentions another missing family. Add
    # the best joint representative when the covered pairing carries under half its
    # policy support. This normally adds zero rows and adds one in the ladder failure.
    if guarantee_moves:
        for token in sorted(
            token for token in all_tokens if token[1].startswith("move:")
        ):
            best = max(
                (item for item in ranked if token in signatures[id(item)]),
                key=lambda item: item.probability,
            )
            represented = max(
                (
                    item.probability
                    for item in selected
                    if token in signatures[id(item)]
                ),
                default=0.0,
            )
            if represented + 1e-12 < 0.50 * best.probability:
                add(best)

    covered_tokens = (
        set().union(*(signatures[item_id] for item_id in selected_ids))
        if selected_ids
        else set()
    )
    uncovered_actions = all_tokens - covered_tokens
    while len(selected) < width and uncovered_actions:
        candidates = [item for item in ranked if id(item) not in selected_ids]
        if not candidates:
            break
        best = max(
            candidates,
            key=lambda item: (
                len(signatures[id(item)] & uncovered_actions),
                item.probability,
            ),
        )
        covered = signatures[id(best)] & uncovered_actions
        if not covered:
            break
        add(best)
        uncovered_actions.difference_update(covered)
    if len(selected) < width:
        selected.extend(
            item for item in ranked if id(item) not in selected_ids
        )
    limit = max(width, len(selected_ids))
    return sorted(
        selected[:limit], key=lambda item: item.probability, reverse=True
    )


_MOVE_ACCURACY_CACHE: dict[str, float] = {}


def _move_accuracy(move_id: str) -> float:
    """Return Showdown-style hit probability for a move id, conservatively."""
    if move_id not in _MOVE_ACCURACY_CACHE:
        try:
            accuracy = Move(move_id, 9).accuracy
            _MOVE_ACCURACY_CACHE[move_id] = (
                float(accuracy)
                if isinstance(accuracy, (int, float))
                and not isinstance(accuracy, bool)
                else 1.0
            )
        except (KeyError, NotImplementedError, ValueError):
            _MOVE_ACCURACY_CACHE[move_id] = 1.0
    return _MOVE_ACCURACY_CACHE[move_id]


def _choice_has_volatile_accuracy(choice: str, threshold: float) -> bool:
    for atom in choice.split(","):
        parts = atom.strip().split()
        if len(parts) >= 2 and parts[0] == "move":
            accuracy = _move_accuracy(parts[1])
            if 0 < accuracy < threshold:
                return True
    return False


def _lost_unexecuted_move_slots(
    log: Sequence[str] | None, role: str, choice: str
) -> tuple[int, ...]:
    """Identify selected moves whose user fainted before it could act this turn."""
    if not log:
        return ()
    atoms = [atom.strip() for atom in choice.split(",")]
    selected_move_slots = {
        slot
        for slot, atom in enumerate(atoms[:2])
        if atom.startswith("move ")
    }
    if not selected_move_slots:
        return ()
    resolved: set[int] = set()
    lost: set[int] = set()
    positions = {f"{role}a:": 0, f"{role}b:": 1}
    for line in log:
        fields = line.split("|")
        if len(fields) < 3:
            continue
        event, actor = fields[1], fields[2]
        slot = next(
            (index for prefix, index in positions.items() if actor.startswith(prefix)),
            None,
        )
        if slot not in selected_move_slots:
            continue
        if event in {"move", "cant"}:
            resolved.add(slot)
        elif event == "faint" and slot not in resolved:
            lost.add(slot)
    return tuple(sorted(lost))


def _weighted_cvar(
    values: list[float], probabilities: list[float], alpha: float
) -> float:
    remaining = alpha
    total = 0.0
    for value, probability in sorted(zip(values, probabilities)):
        take = min(remaining, probability)
        total += value * take
        remaining -= take
        if remaining <= 1e-12:
            break
    if remaining > 0 and values:
        total += min(values) * remaining
    return total / alpha


def risk_blend(
    values: list[float], probabilities: list[float], config: PlannerConfig
) -> tuple[float, float, float, float, float]:
    """Blend outcome values into (score, expected, cvar, worst, deviation)."""
    total = sum(probabilities)
    probabilities = [p / total for p in probabilities]
    expected = float(sum(p * v for p, v in zip(probabilities, values)))
    cvar = float(_weighted_cvar(values, probabilities, config.cvar_alpha))
    worst = float(min(values))
    variance = sum(p * (v - expected) ** 2 for p, v in zip(probabilities, values))
    score = (
        config.expected_weight * expected
        + config.cvar_weight * cvar
        + config.worst_weight * worst
    )
    return score, expected, cvar, worst, math.sqrt(max(0.0, variance))


class ExactMultiTurnPlanner:
    """Expectimax/CVaR planner for simultaneous doubles turns."""

    def __init__(
        self,
        bridge: ExactShowdownBridge,
        prior: Prior | None = None,
        evaluator: Evaluator | None = None,
        config: PlannerConfig | None = None,
    ):
        self.bridge = bridge
        self.prior = prior or UniformPrior()
        self.evaluator = evaluator or material_value
        self.config = config or PlannerConfig()
        self._deadline = 0.0
        self._nodes = 0
        self._truncated = False
        self._principal_choices: dict[int, tuple[str, float]] = {}
        self.continuations: tuple[BranchContinuation, ...] = ()
        self._captured_continuations: list[BranchContinuation] = []
        self.outcomes: tuple[BranchOutcome, ...] = ()
        self._captured_outcomes: list[BranchOutcome] = []

    @staticmethod
    def _opponent(role: str) -> str:
        return "p2" if role == "p1" else "p1"

    def _terminal(self, node: ExactNode, role: str) -> float | None:
        if not node.ended:
            return None
        if not node.winner:
            return 0.0
        side = node.state["sides"][int(role[1]) - 1]
        return 1.0 if node.winner in {role, side.get("name")} else -1.0

    def _leaf(self, node: ExactNode, role: str) -> float:
        terminal = self._terminal(node, role)
        return terminal if terminal is not None else float(self.evaluator(node, role))

    def _rank(
        self,
        node: ExactNode,
        role: str,
        width: int,
        *,
        guarantee_moves: bool = False,
    ) -> list[RankedChoice]:
        choices = self.bridge.choices(node.state, role)
        if not choices:
            return []
        ranked = self.prior.rank(node.state, node.requests, role, choices)
        if not ranked:
            ranked = UniformPrior().rank(node.state, node.requests, role, choices)
        return _normalise(
            _diverse_prefix(ranked, width, guarantee_moves=guarantee_moves)
        )

    def _rng_seed(
        self,
        node: ExactNode,
        ours: str,
        theirs: str,
        sample: int,
        sample_count: int,
    ) -> str | None:
        if sample_count == 1:
            return None
        # Common random numbers: candidate actions should face the same sampled RNG
        # streams. Including the action strings here gave every candidate unrelated
        # rolls, so a lucky Heat Wave branch could outrank Earthquake merely because
        # their accuracy/critical-hit samples differed.
        del ours, theirs
        digest = hashlib.blake2b(
            f"{node.turn}|{node.state.get('prng')}|{sample}".encode(), digest_size=8
        ).digest()
        parts = [int.from_bytes(digest[i : i + 2], "big") for i in range(0, 8, 2)]
        return ",".join(str(part) for part in parts)

    def _risk(self, values: list[float], probabilities: list[float]):
        return risk_blend(values, probabilities, self.config)

    def _search(self, node: ExactNode, role: str, depth: int) -> float:
        terminal = self._terminal(node, role)
        if terminal is not None:
            return terminal
        if (
            (depth <= 0 and node.request_state == "move")
            or time.monotonic() >= self._deadline
            or self._nodes >= self.config.max_nodes
        ):
            self._truncated |= depth > 0
            return self._leaf(node, role)
        width = (
            self.config.continuation_width
            if node.request_state == "move"
            else self.config.replacement_width
        )
        ours = self._rank(node, role, width)
        theirs = self._rank(node, self._opponent(role), width)
        if not ours or not theirs:
            return self._leaf(node, role)
        scores = self._score_actions(node, role, depth, ours, theirs)
        if not scores:
            self._truncated = True
            return self._leaf(node, role)
        best = scores[0]
        margin = (
            best.score - scores[1].score if len(scores) > 1 else float("inf")
        )
        self._principal_choices[id(node)] = (best.choice, margin)
        return best.score

    def _score_actions(
        self,
        node: ExactNode,
        role: str,
        depth: int,
        ours: list[RankedChoice],
        theirs: list[RankedChoice],
        *,
        chance_samples: int | None = None,
        capture_continuations: bool = False,
        capture_outcomes: bool = False,
    ) -> list[ActionScore]:
        branches: list[dict[str, Any]] = []
        metadata: list[tuple[int, int, int, int]] = []
        # Team Preview and forced replacements contain no accuracy/crit rolls. Chance
        # sampling them multiplied identical subtrees (and then sampled every later
        # move again), wasting nearly the entire planning budget after a KO.
        configured_samples = (
            self.config.chance_samples
            if chance_samples is None
            else max(1, chance_samples)
        )
        for i, our_choice in enumerate(ours):
            for j, their_choice in enumerate(theirs):
                sample_count = configured_samples if node.request_state == "move" else 1
                if (
                    chance_samples is None
                    and node.request_state == "move"
                    and (
                        _choice_has_volatile_accuracy(
                            our_choice.choice,
                            self.config.volatile_accuracy_threshold,
                        )
                        or _choice_has_volatile_accuracy(
                            their_choice.choice,
                            self.config.volatile_accuracy_threshold,
                        )
                    )
                ):
                    sample_count = max(
                        sample_count, self.config.volatile_chance_samples
                    )
                for sample in range(sample_count):
                    choices = {
                        role: our_choice.choice,
                        self._opponent(role): their_choice.choice,
                    }
                    branch: dict[str, Any] = {
                        "p1_choice": choices["p1"],
                        "p2_choice": choices["p2"],
                    }
                    seed = self._rng_seed(
                        node,
                        our_choice.choice,
                        their_choice.choice,
                        sample,
                        sample_count,
                    )
                    if seed is not None:
                        branch["rng_seed"] = seed
                    branches.append(branch)
                    metadata.append((i, j, sample, sample_count))
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            self._truncated = True
            return []
        try:
            results = self.bridge.simulate_batch(
                node.state, branches, timeout_s=max(0.05, remaining)
            )
        except TypeError:
            # Small deterministic test bridges predate the timeout argument.
            results = self.bridge.simulate_batch(node.state, branches)
        self._nodes += len(results)
        by_action: list[list[tuple[float, float]]] = [[] for _ in ours]
        for result, (i, j, _sample, pair_sample_count) in zip(results, metadata):
            child = ExactNode.from_result(result)
            # Showdown requests forced replacements before incrementing ``turn``.
            # Requiring child.turn > node.turn therefore gave every KO line one extra
            # searched move turn. A parent move request has completed a move turn even
            # when its child is a same-numbered replacement request.
            completed_turn = node.request_state == "move"
            next_depth = depth - 1 if completed_turn else depth
            value = self._search(child, role, next_depth)
            if completed_turn and self.config.pre_move_ko_penalty:
                lost_slots = _lost_unexecuted_move_slots(
                    result.get("log"), role, ours[i].choice
                )
                value = max(
                    -1.0,
                    value
                    - self.config.pre_move_ko_penalty * len(lost_slots),
                )
            principal = self._principal_choices.pop(id(child), None)
            if capture_outcomes:
                self._captured_outcomes.append(
                    BranchOutcome(
                        root_choice=ours[i].choice,
                        opponent_choice=theirs[j].choice,
                        value=value,
                        probability=(
                            theirs[j].probability / float(pair_sample_count)
                        ),
                        predicted_node=child,
                        searched_depth=depth,
                    )
                )
            if capture_continuations:
                if principal is not None:
                    next_choice, margin = principal
                    self._captured_continuations.append(
                        BranchContinuation(
                            root_choice=ours[i].choice,
                            opponent_choice=theirs[j].choice,
                            next_choice=next_choice,
                            value=value,
                            margin=margin,
                            probability=(
                                theirs[j].probability / float(pair_sample_count)
                            ),
                            predicted_node=child,
                        )
                    )
            if completed_turn:
                value *= self.config.discount
            probability = theirs[j].probability / float(pair_sample_count)
            by_action[i].append((value, probability))

        scores: list[ActionScore] = []
        for candidate, outcomes in zip(ours, by_action):
            values = [value for value, _probability in outcomes]
            probabilities = [probability for _value, probability in outcomes]
            score, expected, cvar, worst, deviation = self._risk(values, probabilities)
            scores.append(
                ActionScore(
                    choice=candidate.choice,
                    actions=candidate.actions,
                    score=score,
                    expected=expected,
                    cvar=cvar,
                    worst=worst,
                    standard_deviation=deviation,
                    prior=candidate.probability,
                    opponent_branches=len(outcomes),
                )
            )
        return sorted(scores, key=lambda item: item.score, reverse=True)

    def _result(
        self,
        rankings: list[ActionScore],
        started: float,
        *,
        completed_depth: int,
        screened_actions: int,
        deepened_actions: int,
        fallback_reason: str | None = None,
        deepened_choices: Sequence[str] = (),
        selected_depth_coverage: float | None = None,
    ) -> PlanResult:
        if not rankings:
            raise ValueError("exact planner produced no ranked root actions")
        best = rankings[0]
        return PlanResult(
            choice=best.choice,
            actions=best.actions,
            score=best.score,
            rankings=tuple(rankings),
            nodes=self._nodes,
            elapsed_s=time.monotonic() - started,
            completed_depth=completed_depth,
            truncated=self._truncated,
            screened_actions=screened_actions,
            deepened_actions=deepened_actions,
            fallback_reason=fallback_reason,
            deepened_choices=tuple(deepened_choices),
            selected_depth_coverage=(
                float(bool(deepened_choices))
                if selected_depth_coverage is None
                else float(selected_depth_coverage)
            ),
        )

    def _plan_anytime(
        self,
        root: ExactNode,
        role: str,
        ours: list[RankedChoice],
        started: float,
    ) -> PlanResult:
        """Screen every root move family, then deepen the strongest candidates."""
        hard_deadline = started + self.config.time_budget_s
        self._deadline = min(
            hard_deadline,
            max(started + self.config.screen_budget_s, time.monotonic() + 0.05),
        )
        screen_theirs = self._rank(
            root,
            self._opponent(role),
            self.config.screen_opponent_width,
            guarantee_moves=False,
        )
        if not screen_theirs:
            raise ValueError("no legal opponent choices available at planner root")
        try:
            screened = self._score_actions(
                root,
                role,
                1,
                ours,
                screen_theirs,
                chance_samples=1,
                capture_outcomes=True,
            )
        except ExactSimulatorError as exc:
            raise ValueError(f"root screening failed: {exc}") from exc
        fallback_reason = None
        if not screened:
            # Policy ranking already covered every move family. Preserve that complete
            # prefix as the anytime fallback when embedding/ranking consumed the
            # screen slice; any completed exact deep pass below replaces its row.
            self._truncated = True
            fallback_reason = "screen_timeout"
            screened = [
                ActionScore(
                    choice=item.choice,
                    actions=item.actions,
                    score=2.0 * item.probability - 1.0,
                    expected=2.0 * item.probability - 1.0,
                    cvar=2.0 * item.probability - 1.0,
                    worst=2.0 * item.probability - 1.0,
                    standard_deviation=0.0,
                    prior=item.probability,
                    opponent_branches=0,
                )
                for item in ours
            ]

        # Screening scores remain valid fallbacks for every action. A candidate only
        # replaces its screen score after a complete deeper pass, so interruption can
        # never leave an unranked action or a half-computed winner.
        merged = {score.choice: score for score in screened}
        deepened = 0
        deepened_choices: list[str] = []
        self._deadline = hard_deadline
        full_theirs = self._rank(
            root,
            self._opponent(role),
            self.config.opponent_width,
            guarantee_moves=True,
        )
        for candidate in screened[: self.config.deep_root_width]:
            if (
                time.monotonic() >= hard_deadline
                or self._nodes >= self.config.max_nodes
            ):
                self._truncated = True
                fallback_reason = fallback_reason or "time_or_node_budget"
                break
            try:
                deeper = self._score_actions(
                    root,
                    role,
                    self.config.depth,
                    [
                        RankedChoice(
                            candidate.choice, candidate.actions, candidate.prior
                        )
                    ],
                    full_theirs,
                    capture_continuations=True,
                    capture_outcomes=True,
                )
            except ExactSimulatorError:
                self._truncated = True
                fallback_reason = fallback_reason or "simulator_timeout"
                break
            if deeper:
                merged[candidate.choice] = deeper[0]
                deepened += 1
                deepened_choices.append(candidate.choice)
            else:
                self._truncated = True
                fallback_reason = fallback_reason or "time_or_node_budget"
                break
        rankings = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        completed_depth = self.config.depth if deepened else 1
        return self._result(
            rankings,
            started,
            completed_depth=completed_depth,
            screened_actions=len(screened),
            deepened_actions=deepened,
            fallback_reason=fallback_reason,
            deepened_choices=deepened_choices,
        )

    def plan(self, root: ExactNode, role: str = "p1") -> PlanResult:
        """Rank root actions by exact multi-turn outcomes."""
        started = time.monotonic()
        self._deadline = started + self.config.time_budget_s
        self._nodes = 0
        self._truncated = False
        self._principal_choices = {}
        self._captured_continuations = []
        self.continuations = ()
        self._captured_outcomes = []
        self.outcomes = ()
        terminal = self._terminal(root, role)
        if terminal is not None:
            raise ValueError("cannot plan from an ended battle")
        ours = self._rank(
            root, role, self.config.root_width, guarantee_moves=True
        )
        if not ours:
            raise ValueError("no legal exact choices available at planner root")
        if self.config.anytime and root.request_state == "move":
            result = self._plan_anytime(root, role, ours, started)
            self.continuations = tuple(self._captured_continuations)
            self.outcomes = tuple(self._captured_outcomes)
            return result
        theirs = self._rank(
            root,
            self._opponent(role),
            self.config.opponent_width,
            guarantee_moves=True,
        )
        if not theirs:
            raise ValueError("no legal exact opponent choices at planner root")
        rankings = self._score_actions(
            root,
            role,
            self.config.depth,
            ours,
            theirs,
            capture_continuations=True,
            capture_outcomes=True,
        )
        result = self._result(
            rankings,
            started,
            completed_depth=self.config.depth,
            screened_actions=len(ours),
            deepened_actions=len(ours),
            deepened_choices=[item.choice for item in ours],
        )
        self.continuations = tuple(self._captured_continuations)
        self.outcomes = tuple(self._captured_outcomes)
        return result


def aggregate_plans(
    results: list[tuple[float, PlanResult]],
    elapsed_s: float,
    *,
    config: PlannerConfig,
    total_probability: float | None = None,
    failed_roots: int = 0,
    required_choices: set[str] | None = None,
    minimum_depth_coverage: float = 0.0,
) -> PlanResult:
    """Merge per-determinization plans into one risk-weighted ranking.

    Used both for hidden-set move planning (``ExactDeterminizationPlanner``) and for
    multi-world Team Preview planning, which runs its own root loop so it can apply
    a different acceptance rule to incomplete worlds.
    """
    completed_probability = sum(probability for probability, _result in results)
    total_probability = (
        completed_probability if total_probability is None else total_probability
    )
    if total_probability <= 0:
        raise ValueError("determinization probabilities must have positive mass")
    normalized = [
        (probability / total_probability, result)
        for probability, result in results
    ]
    by_choice: dict[str, list[tuple[float, ActionScore]]] = {}
    for probability, result in normalized:
        for score in result.rankings:
            by_choice.setdefault(score.choice, []).append((probability, score))

    rankings: list[ActionScore] = []
    for choice, outcomes in by_choice.items():
        if required_choices is not None and choice not in required_choices:
            continue
        represented = sum(probability for probability, _score in outcomes)
        # A controlled-side choice should exist in every determinization. Treat a
        # missing mapping conservatively rather than rewarding the omission.
        values = [score.score for _probability, score in outcomes]
        probabilities = [probability for probability, _score in outcomes]
        if represented < 1.0 - 1e-9:
            values.append(-1.0)
            probabilities.append(1.0 - represented)
        score, expected, cvar, worst, deviation = risk_blend(
            values, probabilities, config
        )
        first = outcomes[0][1]
        rankings.append(
            ActionScore(
                choice=choice,
                actions=first.actions,
                score=score,
                expected=expected,
                cvar=cvar,
                worst=worst,
                standard_deviation=deviation,
                prior=sum(
                    probability * outcome.prior
                    for probability, outcome in outcomes
                ),
                opponent_branches=sum(
                    outcome.opponent_branches for _probability, outcome in outcomes
                ),
            )
        )
    rankings.sort(key=lambda item: item.score, reverse=True)
    if not rankings:
        raise ValueError("no action survived determinization aggregation")
    depth_coverage = {
        choice: sum(
            probability
            for probability, result in normalized
            if choice in result.deepened_choices
        )
        for choice in by_choice
    }
    rankings = [
        replace(
            row,
            depth_coverage=float(depth_coverage.get(row.choice, 0.0)),
        )
        for row in rankings
    ]
    if minimum_depth_coverage > 0:
        sufficiently_searched = [
            row
            for row in rankings
            if depth_coverage.get(row.choice, 0.0) + 1e-9
            >= minimum_depth_coverage
        ]
        if sufficiently_searched:
            sufficiently_searched_ids = {
                row.choice for row in sufficiently_searched
            }
            # Never reject a useful exact search merely because its absolute top
            # row was only deepened in a low-mass hidden world. Prefer the best
            # action whose future was actually searched across enough posterior
            # mass, while retaining every other row for guards and diagnostics.
            rankings = sufficiently_searched + [
                row
                for row in rankings
                if row.choice not in sufficiently_searched_ids
            ]
    best = rankings[0]
    selected_depth_coverage = depth_coverage.get(best.choice, 0.0)
    return PlanResult(
        choice=best.choice,
        actions=best.actions,
        score=best.score,
        rankings=tuple(rankings),
        nodes=sum(result.nodes for _probability, result in results),
        elapsed_s=elapsed_s,
        completed_depth=(
            config.depth
            if selected_depth_coverage >= 1.0 - 1e-9
            else 1
        ),
        truncated=any(result.truncated for _probability, result in results),
        screened_actions=min(
            result.screened_actions for _probability, result in results
        ),
        deepened_actions=sum(
            result.deepened_actions for _probability, result in results
        ),
        fallback_reason=(
            "partial_root_failure"
            if failed_roots
            else "partial_determinization_search"
            if any(result.truncated for _probability, result in results)
            else None
        ),
        deepened_choices=tuple(
            sorted(
                {
                    choice
                    for _probability, result in normalized
                    for choice in result.deepened_choices
                }
            )
        ),
        selected_depth_coverage=selected_depth_coverage,
    )


class ExactDeterminizationPlanner:
    """Aggregate exact plans across up to eight hidden-set determinizations.

    The total wall-clock budget is divided across concrete states. Each state still
    screens every root move family before deepening, and incomplete states retain
    their complete screen rankings. Open sheets use one set determination, but may
    still carry several roots for the unknown pair of back Pokemon selected at Team
    Preview; a sheet reveals sets, not which four were brought.
    """

    def __init__(
        self,
        bridge: ExactShowdownBridge,
        prior: Prior | None = None,
        evaluator: Evaluator | None = None,
        config: PlannerConfig | None = None,
    ):
        self.bridge = bridge
        self.prior = prior or UniformPrior()
        self.evaluator = evaluator or material_value
        self.config = config or PlannerConfig(anytime=True)
        self.continuations: tuple[BranchContinuation, ...] = ()
        self.outcomes: tuple[BranchOutcome, ...] = ()

    def _aggregate(
        self,
        results: list[tuple[float, PlanResult]],
        elapsed_s: float,
        *,
        total_probability: float | None = None,
        failed_roots: int = 0,
        required_choices: set[str] | None = None,
        minimum_depth_coverage: float = 0.0,
    ) -> PlanResult:
        return aggregate_plans(
            results,
            elapsed_s,
            config=self.config,
            total_probability=total_probability,
            failed_roots=failed_roots,
            required_choices=required_choices,
            minimum_depth_coverage=minimum_depth_coverage,
        )

    def plan(
        self,
        roots: Sequence[WeightedExactNode],
        role: str = "p1",
        minimum_depth_coverage: float = 0.0,
    ) -> PlanResult:
        if not roots:
            raise ValueError("at least one determinization is required")
        if len(roots) > 8:
            raise ValueError("hidden search supports at most eight determinizations")
        if not 0 <= minimum_depth_coverage <= 1:
            raise ValueError("minimum depth coverage must be in [0, 1]")
        started = time.monotonic()
        # Leaf inference and Python aggregation can finish a few hundred ms after a
        # bridge deadline. Reserve that tail inside the configured search budget so
        # a nominal nine-second plan actually returns by nine seconds and leaves the
        # caller's final second for submission.
        scheduler_reserve = min(0.65, self.config.time_budget_s * 0.15)
        deadline = started + self.config.time_budget_s - scheduler_reserve
        results: list[tuple[float, PlanResult]] = []
        continuations: list[BranchContinuation] = []
        outcomes: list[BranchOutcome] = []
        failed_roots = 0
        root_errors: list[str] = []
        required_choices: set[str] | None = None
        total_probability = sum(max(0.0, root.probability) for root in roots)
        for index, weighted in enumerate(roots):
            remaining_states = len(roots) - index
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                break
            slice_s = min(9.0, max(0.05, remaining_s / remaining_states))
            screen_s = min(
                self.config.screen_budget_s / len(roots),
                max(0.01, slice_s * 0.45),
            )
            planner = ExactMultiTurnPlanner(
                self.bridge,
                self.prior,
                self.evaluator,
                replace(
                    self.config,
                    anytime=True,
                    time_budget_s=slice_s,
                    screen_budget_s=screen_s,
                ),
            )
            try:
                result = planner.plan(weighted.node, role)
            except (ExactSimulatorError, ValueError) as exc:
                failed_roots += 1
                root_errors.append(f"{weighted.label}: {type(exc).__name__}: {exc}")
                continue
            if index == 0:
                # Offline generation defines root zero as the sampled real world.
                # Live roots should agree after reconciliation, but this constraint
                # also prevents a divergent hidden particle from proposing an action
                # that cannot be submitted in the actual public position.
                required_choices = {score.choice for score in result.rankings}
            results.append((weighted.probability, result))
            continuations.extend(
                replace(
                    continuation,
                    probability=continuation.probability * weighted.probability,
                    root_label=weighted.label,
                )
                for continuation in planner.continuations
            )
            outcomes.extend(
                replace(
                    outcome,
                    probability=outcome.probability * weighted.probability,
                    root_label=weighted.label,
                )
                for outcome in planner.outcomes
            )
        if not results:
            raise DeterminizationBudgetExhausted(
                "determinization search exhausted its budget"
                + (f"; {'; '.join(root_errors[:3])}" if root_errors else "")
            )
        aggregate = self._aggregate(
            results,
            time.monotonic() - started,
            total_probability=total_probability,
            failed_roots=failed_roots,
            required_choices=required_choices,
            minimum_depth_coverage=minimum_depth_coverage,
        )
        if failed_roots:
            aggregate = replace(aggregate, truncated=True)
        self.continuations = tuple(
            continuation
            for continuation in continuations
            if continuation.root_choice == aggregate.choice
        )
        self.outcomes = tuple(
            outcome
            for outcome in outcomes
            if outcome.root_choice == aggregate.choice
        )
        return aggregate
