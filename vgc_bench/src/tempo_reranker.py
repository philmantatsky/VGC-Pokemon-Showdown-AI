"""Mechanics-aware scoring for speed control and coordinated doubles turns.

The PPO policy can observe that Trick Room and Tailwind exist, but it is not given
their remaining duration or the resulting move order.  More importantly, the
training reward is terminal win/loss, so a bad Protect can receive the same return as
the sequence that eventually wins the battle.  This module supplies a small amount
of exact, inspectable tactical evidence without pretending to solve the whole turn.

It is deliberately a *soft* score consumed by ``opponent_reranker``.  Factual guards
still run first, only near-tied policy candidates are eligible, and this layer cannot
resurrect a candidate that a factual guard demoted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from poke_env.battle import (
    DoubleBattle,
    Effect,
    Field,
    Move,
    MoveCategory,
    Pokemon,
    SideCondition,
    Status,
    Target,
    Weather,
)

from vgc_bench.src import guards
from vgc_bench.src.guards import FIRST_TURN_ONLY, PROTECT_MOVES, Candidate

if TYPE_CHECKING:
    from vgc_bench.src.opponent_tactics import MovePrediction, SwitchPrediction


_RAIN = {Weather.RAINDANCE, Weather.PRIMORDIALSEA}
_SUN = {Weather.SUNNYDAY, Weather.DESOLATELAND}
_SNOW = {Weather.HAIL, Weather.SNOWSCAPE}
_SPEED_ITEMS = {
    "choicescarf": 1.5,
    "ironball": 0.5,
    "machobrace": 0.5,
    "poweranklet": 0.5,
    "powerband": 0.5,
    "powerbelt": 0.5,
    "powerbracer": 0.5,
    "powerlens": 0.5,
    "powerweight": 0.5,
}
_WEATHER_SPEED_ABILITIES = {
    "swiftswim": (_RAIN, 2.0),
    "chlorophyll": (_SUN, 2.0),
    "sandrush": ({Weather.SANDSTORM}, 2.0),
    "slushrush": (_SNOW, 2.0),
}
_REDIRECTION_MOVES = {"followme", "ragepowder"}
_SPREAD_TARGETS = {Target.ALL, Target.ALL_ADJACENT, Target.ALL_ADJACENT_FOES}


def _norm(value: str | None) -> str:
    if not value:
        return ""
    return "".join(character for character in value.lower() if character.isalnum())


def _condition_age(turn: int, started: object) -> int:
    """Age of a timed condition, tolerating protocol sentinels in test/live states."""
    if isinstance(started, bool):
        return 0
    if isinstance(started, (int, float)):
        return max(0, turn - int(started))
    return 0


def trick_room_turns(battle: DoubleBattle) -> int:
    """Action turns left, including the turn currently awaiting choices."""
    if Field.TRICK_ROOM not in battle.fields:
        return 0
    age = _condition_age(battle.turn, battle.fields[Field.TRICK_ROOM])
    return max(0, 5 - age)


def tailwind_turns(battle: DoubleBattle, ours: bool) -> int:
    conditions = battle.side_conditions if ours else battle.opponent_side_conditions
    if SideCondition.TAILWIND not in conditions:
        return 0
    age = _condition_age(battle.turn, conditions[SideCondition.TAILWIND])
    return max(0, 4 - age)


def _abilities_suppressed(battle: DoubleBattle) -> bool:
    if Field.NEUTRALIZING_GAS in battle.fields:
        return True
    actives = (*battle.active_pokemon, *battle.opponent_active_pokemon)
    return any(
        mon is not None and _norm(mon.ability) == "neutralizinggas" for mon in actives
    )


def _weather_suppressed(battle: DoubleBattle) -> bool:
    actives = (*battle.active_pokemon, *battle.opponent_active_pokemon)
    return any(
        mon is not None and _norm(mon.ability) in {"airlock", "cloudnine"}
        for mon in actives
    )


def _has_effect(mon: Pokemon, effect: Effect) -> bool:
    try:
        return effect in mon.effects
    except (AttributeError, TypeError):
        return False


def effective_speed(battle: DoubleBattle, mon: Pokemon, ours: bool) -> float | None:
    """Known effective Speed before Trick Room reverses ordering.

    This covers the deterministic modifiers that matter most in VGC: stages,
    Tailwind, paralysis, known speed items, weather abilities, Surge Surfer, Quick
    Feet, Slow Start, and the explicit Protosynthesis/Quark Drive Speed effects.
    Unknown abilities/items simply contribute no modifier; this is a soft scorer, not
    a claim that a hidden set is certain.
    """
    base = (mon.stats or {}).get("spe")
    if not base:
        return None
    speed = float(base) * guards._BOOST_MULT.get(mon.boosts.get("spe", 0), 1.0)
    if tailwind_turns(battle, ours):
        speed *= 2.0

    ability = _norm(mon.ability)
    abilities_suppressed = _abilities_suppressed(battle)
    if mon.status == Status.PAR and (abilities_suppressed or ability != "quickfeet"):
        speed *= 0.5

    speed *= _SPEED_ITEMS.get(_norm(mon.item), 1.0)
    if not abilities_suppressed:
        if ability == "quickfeet" and mon.status is not None:
            speed *= 1.5
        weather_rule = _WEATHER_SPEED_ABILITIES.get(ability)
        if weather_rule is not None and not _weather_suppressed(battle):
            weathers, multiplier = weather_rule
            if any(weather in battle.weather for weather in weathers):
                speed *= multiplier
        if ability == "surgesurfer" and Field.ELECTRIC_TERRAIN in battle.fields:
            speed *= 2.0

    if _has_effect(mon, Effect.SLOW_START):
        speed *= 0.5
    if _has_effect(mon, Effect.PROTOSYNTHESISSPE) or _has_effect(
        mon, Effect.QUARKDRIVESPE
    ):
        speed *= 1.5
    return speed


def effective_speed_bounds(
    battle: DoubleBattle, mon: Pokemon, ours: bool
) -> tuple[float, float] | None:
    """Plausible Speed interval, widening hidden opponent information.

    poke-env supplies or synthesizes a single opponent stat even when Champions team
    sheets omit EVs/nature, and hidden-sheet damage setup can impute a complete stat
    line. Treating that point estimate as exact caused confident but false Trick Room
    conclusions. Our own team is exact. Opponents use the full Champions 0..32 Speed
    investment and 0.9..1.1 nature range, plus possible hidden speed items/abilities.
    """
    point = effective_speed(battle, mon, ours)
    if point is None:
        return None
    base_stats = getattr(mon, "base_stats", None)
    if ours or not base_stats:
        return point, point

    base_speed = base_stats.get("spe")
    if not base_speed:
        return point, point
    low = float(int((base_speed + 20) * 0.9))
    high = float(int((base_speed + 52) * 1.1))
    stage = guards._BOOST_MULT.get(mon.boosts.get("spe", 0), 1.0)
    low *= stage
    high *= stage
    if tailwind_turns(battle, ours):
        low *= 2.0
        high *= 2.0

    ability = _norm(mon.ability)
    abilities_suppressed = _abilities_suppressed(battle)
    if mon.status == Status.PAR and (abilities_suppressed or ability != "quickfeet"):
        low *= 0.5
        high *= 0.5

    item = _norm(mon.item)
    if item and item != "unknownitem":
        modifier = _SPEED_ITEMS.get(item, 1.0)
        low *= modifier
        high *= modifier
    else:
        # Hidden Choice Scarf and Iron Ball-style items are both legal. This broad
        # interval intentionally disables strategic reranking when either would flip
        # the conclusion.
        low *= 0.5
        high *= 1.5

    if not abilities_suppressed:
        if ability:
            if ability == "quickfeet" and mon.status is not None:
                low *= 1.5
                high *= 1.5
            weather_rule = _WEATHER_SPEED_ABILITIES.get(ability)
            if weather_rule is not None and not _weather_suppressed(battle):
                weathers, multiplier = weather_rule
                if any(weather in battle.weather for weather in weathers):
                    low *= multiplier
                    high *= multiplier
            if ability == "surgesurfer" and Field.ELECTRIC_TERRAIN in battle.fields:
                low *= 2.0
                high *= 2.0
        else:
            possible = {
                _norm(value) for value in getattr(mon, "possible_abilities", ())
            }
            possible_multiplier = 1.0
            for possible_ability in possible:
                weather_rule = _WEATHER_SPEED_ABILITIES.get(possible_ability)
                if weather_rule is not None and not _weather_suppressed(battle):
                    weathers, multiplier = weather_rule
                    if any(weather in battle.weather for weather in weathers):
                        possible_multiplier = max(possible_multiplier, multiplier)
                if (
                    possible_ability == "surgesurfer"
                    and Field.ELECTRIC_TERRAIN in battle.fields
                ):
                    possible_multiplier = max(possible_multiplier, 2.0)
                if possible_ability == "quickfeet" and mon.status is not None:
                    possible_multiplier = max(possible_multiplier, 1.5)
            high *= possible_multiplier

    if _has_effect(mon, Effect.SLOW_START):
        low *= 0.5
        high *= 0.5
    if _has_effect(mon, Effect.PROTOSYNTHESISSPE) or _has_effect(
        mon, Effect.QUARKDRIVESPE
    ):
        low *= 1.5
        high *= 1.5
    return low, high


def effective_priority(mon: Pokemon, move: Move) -> int:
    """Deterministic priority modifiers relevant to Encore timing."""
    priority = int(move.priority or 0)
    ability = _norm(mon.ability)
    if ability == "prankster" and move.category == MoveCategory.STATUS:
        priority += 1
    elif (
        ability == "galewings"
        and move.type is not None
        and move.type.name == "FLYING"
        and (mon.current_hp_fraction or 0.0) >= 1.0
    ):
        priority += 1
    elif ability == "triage" and (move.heal > 0 or move.drain > 0):
        priority += 3
    return priority


def acts_before(
    battle: DoubleBattle,
    first_mon: Pokemon,
    first_move: Move,
    first_ours: bool,
    second_mon: Pokemon,
    second_move: Move,
    second_ours: bool,
    *,
    under_trick_room: bool | None = None,
) -> bool | None:
    """Whether the first action precedes the second; None means a speed tie/unknown."""
    first_priority = effective_priority(first_mon, first_move)
    second_priority = effective_priority(second_mon, second_move)
    if first_priority != second_priority:
        return first_priority > second_priority

    # Stall/Lagging Tail/Full Incense act last within their priority bracket. Random
    # Quick Claw-style effects are intentionally not guessed.
    first_last = _norm(first_mon.ability) == "stall" or _norm(first_mon.item) in {
        "fullincense",
        "laggingtail",
    }
    second_last = _norm(second_mon.ability) == "stall" or _norm(second_mon.item) in {
        "fullincense",
        "laggingtail",
    }
    if first_last != second_last:
        return not first_last

    first_bounds = effective_speed_bounds(battle, first_mon, first_ours)
    second_bounds = effective_speed_bounds(battle, second_mon, second_ours)
    if first_bounds is None or second_bounds is None:
        return None
    room = (
        Field.TRICK_ROOM in battle.fields
        if under_trick_room is None
        else under_trick_room
    )
    first_low, first_high = first_bounds
    second_low, second_high = second_bounds
    if room:
        if first_high < second_low:
            return True
        if first_low > second_high:
            return False
    else:
        if first_low > second_high:
            return True
        if first_high < second_low:
            return False
    return None


@dataclass(frozen=True)
class SpeedControlSnapshot:
    trick_room_turns: int
    our_tailwind_turns: int
    their_tailwind_turns: int
    trick_room_advantage: float
    known_comparisons: int

    @property
    def has_speed_evidence(self) -> bool:
        return self.known_comparisons > 0


def speed_control_snapshot(battle: DoubleBattle) -> SpeedControlSnapshot:
    """Threat-weighted share of active matchups that our side wins under Trick Room.

    +1 means every known equal-priority comparison favors us; -1 means every one
    favors the opponent.  Opponent HP is the threat weight, which is important in the
    reported Turn 5: a 5% Tyranitar should not cancel the significance of a healthy
    Sand Rush Excadrill.
    """
    score = total_weight = 0.0
    known = 0
    for ours in battle.active_pokemon:
        if ours is None or ours.fainted:
            continue
        for foe in battle.opponent_active_pokemon:
            if foe is None or foe.fainted:
                continue
            our_bounds = effective_speed_bounds(battle, ours, True)
            foe_bounds = effective_speed_bounds(battle, foe, False)
            if our_bounds is None or foe_bounds is None:
                continue
            # This metric describes equal-priority speed order. Using a dummy status
            # move here would accidentally give Prankster users +1 priority and turn
            # a pure speed comparison into a move-specific one.
            our_low, our_high = our_bounds
            foe_low, foe_high = foe_bounds
            if our_high < foe_low:
                result = True
            elif our_low > foe_high:
                result = False
            else:
                continue
            weight = max(0.02, float(foe.current_hp_fraction or 0.0))
            weight *= max(0.20, float(ours.current_hp_fraction or 0.0))
            score += weight if result else -weight
            total_weight += weight
            known += 1
    advantage = score / total_weight if total_weight else 0.0
    return SpeedControlSnapshot(
        trick_room_turns=trick_room_turns(battle),
        our_tailwind_turns=tailwind_turns(battle, True),
        their_tailwind_turns=tailwind_turns(battle, False),
        trick_room_advantage=advantage,
        known_comparisons=known,
    )


def _orders(battle: DoubleBattle, candidate: Candidate) -> tuple[object | None, ...]:
    return tuple(
        cached if cached is not None else guards._decode(battle, action, position)
        for position, (action, cached) in enumerate(
            zip(candidate.actions, candidate.orders)
        )
    )


def _move(order: object | None) -> Move | None:
    selected = getattr(order, "order", None)
    return selected if isinstance(selected, Move) else None


def _is_protect(order: object | None) -> bool:
    move = _move(order)
    return move is not None and move.id in PROTECT_MOVES


def _makes_progress(order: object | None) -> bool:
    chosen = getattr(order, "order", None)
    if isinstance(chosen, Pokemon):
        return True
    return isinstance(chosen, Move) and (
        chosen.category != MoveCategory.STATUS or chosen.base_power > 0
    )


def _protected_spread_pair(orders: tuple[object | None, ...]) -> bool:
    for protected_position in range(2):
        attacker_position = 1 - protected_position
        move = _move(orders[attacker_position])
        if (
            _is_protect(orders[protected_position])
            and move is not None
            and move.category != MoveCategory.STATUS
            and move.base_power > 0
            and move.target == Target.ALL_ADJACENT
        ):
            return True
    return False


def _prediction_move(foe: Pokemon, move_id: str, gen: int) -> Move | None:
    known = (foe.moves or {}).get(move_id)
    if known is not None:
        return known
    try:
        return Move(move_id, gen=gen)
    except Exception:
        return None


def _lock_utility(
    battle: DoubleBattle,
    locked: Move,
    snapshot: SpeedControlSnapshot,
    partner_order: object | None,
) -> float:
    """Value of the move Encore would actually lock, not the stale previous move."""
    if locked.id == "trickroom":
        # Reusing Trick Room toggles it. If current room favors them (negative),
        # forcing the toggle is good for us; if it favors us, it is actively bad.
        direction = -snapshot.trick_room_advantage
        return 0.85 * direction
    if locked.id in PROTECT_MOVES:
        return 0.18
    if locked.id in FIRST_TURN_ONLY:
        return 0.40
    if locked.side_condition is not None:
        conditions = battle.opponent_side_conditions
        return 0.28 if locked.side_condition in conditions else 0.05
    if locked.id in _REDIRECTION_MOVES:
        partner_move = _move(partner_order)
        if partner_move is not None and partner_move.target in _SPREAD_TARGETS:
            return 0.14
        return 0.03
    if (
        locked.category == MoveCategory.STATUS
        and locked.target == Target.SELF
        and any(value > 0 for value in (locked.boosts or {}).values())
    ):
        return 0.08
    return 0.0


def _encore_utility(
    battle: DoubleBattle,
    position: int,
    order: object,
    partner_order: object | None,
    snapshot: SpeedControlSnapshot,
    opponent_moves: tuple[MovePrediction, MovePrediction] | None,
    opponent_switches: tuple[SwitchPrediction, SwitchPrediction] | None,
) -> float:
    encore = _move(order)
    if encore is None or encore.id != "encore":
        return 0.0
    target = int(getattr(order, "move_target", 0) or 0) - 1
    if target not in {0, 1}:
        return 0.0
    foe = battle.opponent_active_pokemon[target]
    user = battle.active_pokemon[position]
    if foe is None or user is None or foe.fainted:
        return 0.0
    previous = foe.last_move
    if previous is None or "failencore" in previous.flags:
        return -0.20

    prediction = opponent_moves[target] if opponent_moves is not None else None
    if prediction is None or prediction.reliability <= 0 or not prediction.moves:
        return 0.0
    expected = probability_sum = 0.0
    for move_id, probability in prediction.moves:
        predicted = _prediction_move(foe, move_id, battle.gen)
        if predicted is None or probability <= 0:
            continue
        foe_first = acts_before(battle, foe, predicted, False, user, encore, True)
        if foe_first is None:
            continue
        locked = predicted if foe_first else previous
        expected += probability * _lock_utility(battle, locked, snapshot, partner_order)
        probability_sum += probability
    if probability_sum <= 0:
        return 0.0
    reliability = min(1.0, max(0.0, prediction.reliability))
    switch_probability = 0.0
    if opponent_switches is not None:
        switch_probability = min(
            0.85, max(0.0, opponent_switches[target].switch_probability)
        )
    return expected / probability_sum * reliability * (1.0 - switch_probability)


@dataclass(frozen=True)
class TempoScores:
    utility: dict[tuple[int, int], float]
    factors: dict[tuple[int, int], dict[str, float]]
    snapshot: SpeedControlSnapshot
    factor_totals: dict[str, float]

    @property
    def has_evidence(self) -> bool:
        return any(abs(value) > 1e-9 for value in self.utility.values())


def score_candidates(
    battle: DoubleBattle,
    candidates: list[Candidate],
    opponent_moves: tuple[MovePrediction, MovePrediction] | None,
    opponent_switches: tuple[SwitchPrediction, SwitchPrediction] | None,
) -> TempoScores:
    """Return additive, bounded tactical utility for each joint candidate."""
    snapshot = speed_control_snapshot(battle)
    utilities: dict[tuple[int, int], float] = {}
    all_factors: dict[tuple[int, int], dict[str, float]] = {}
    for candidate in candidates:
        orders = _orders(battle, candidate)
        protects = sum(_is_protect(order) for order in orders)
        factors: dict[str, float] = {}
        room = snapshot.trick_room_advantage
        remaining = snapshot.trick_room_turns

        if remaining and snapshot.has_speed_evidence:
            urgency = 1.0 / math.sqrt(remaining)
            if protects == 2:
                if room > 0.10:
                    factors["double_protect_in_good_room"] = -1.10 * room * urgency
                elif room < -0.10:
                    # Stalling bad room is real, but a fully passive turn gives the
                    # opponent freedom to switch or set up, so this reward is modest.
                    factors["double_protect_stalls_bad_room"] = 0.18 * -room * urgency
            elif protects == 1 and any(
                _makes_progress(order) and not _is_protect(order) for order in orders
            ):
                if room < -0.10:
                    factors["protect_stalls_bad_room"] = 0.26 * -room * urgency

        if remaining and room < -0.10 and _protected_spread_pair(orders):
            # Earthquake/Surf/Discharge beside Protect is an intentional joint play,
            # not ally damage. It is not automatically *better* than two targeted
            # attacks, though; credit only the concrete use reported here--making
            # progress while preserving a partner through harmful Trick Room.
            factors["protected_spread_in_bad_room"] = (
                0.12 * -room / math.sqrt(remaining)
            )

        encore = 0.0
        for position, order in enumerate(orders):
            encore += _encore_utility(
                battle,
                position,
                order,
                orders[1 - position],
                snapshot,
                opponent_moves,
                opponent_switches,
            )
        if abs(encore) > 1e-9:
            factors["encore"] = encore

        utility = max(-1.25, min(1.25, sum(factors.values())))
        utilities[candidate.actions] = utility
        all_factors[candidate.actions] = factors
    factor_totals: dict[str, float] = {}
    for factors in all_factors.values():
        for name, value in factors.items():
            factor_totals[name] = factor_totals.get(name, 0.0) + value
    return TempoScores(utilities, all_factors, snapshot, factor_totals)
