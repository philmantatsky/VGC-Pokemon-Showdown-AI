"""Knowledge guards over the policy's action distribution.

Modelled on the Laplace bot's guard stack (engine_search.py:470-479). The policy
proposes; knowledge reorders a prefix. Every guard follows Laplace's contract, which
is what makes layering rules on a learned policy safe:

  * DEMOTE, never delete -- a guard moves candidates to the back, so a wrong guard
    costs ranking, not legality
  * STAND DOWN if the veto would empty the list (engine_search.py:796)
  * record which guard changed the front-runner, so we can measure whether each one
    earns its keep instead of guessing (engine_search.py:463 `_stage`)

Doubles-specific wrinkle: the two action heads are NOT independent. The policy samples
slot 0, re-masks, then samples slot 1 conditioned on it (policy.py:81-91). So guards
must score JOINT PAIRS -- a per-slot guard would, for instance, veto the Earthquake in
a legitimate Earthquake-under-ally-Protect play.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from poke_env.battle import (
    DoubleBattle,
    Effect,
    Field,
    Move,
    MoveCategory,
    Pokemon,
    PokemonType,
    SideCondition,
    Status,
    Target,
    Weather,
)
from poke_env.environment import DoublesEnv

from vgc_bench.src import vgc_knowledge as K

# Moves that auto-fail unless it is this Pokemon's first turn out.
FIRST_TURN_ONLY = {"fakeout", "firstimpression", "matblock"}

# Blocks damage aimed at the user for the turn, so an ally-damage veto stands down.
PROTECT_MOVES = {
    "protect",
    "detect",
    "spikyshield",
    "banefulbunker",
    "burningbulwark",
    "kingsshield",
    "obstruct",
    "silktrap",
    "maxguard",
}

# Abilities that pull single-target moves of a given type away from their target and
# absorb them. Only ever consulted when the ability is REVEALED (Reg M-B's Open Team
# Sheets usually reveal it at teampreview); an unrevealed ability is a probability,
# not a certainty, and Laplace's bar for a hard veto is certainty.
REDIRECT_ABILITIES = {
    "lightningrod": PokemonType.ELECTRIC,
    "stormdrain": PokemonType.WATER,
}

# These abilities make positive-priority moves fail against the entire opposing side.
# They are only hard facts once revealed; hidden-set priors remain probabilities.
PRIORITY_BLOCK_ABILITIES = {"armortail", "dazzling", "queenlymajesty"}
_JOINT_SET_CACHE: dict | None = None

# Setup that can turn one free turn into an immediate sweep. These are deliberately
# narrower than every boosting move: Double Protect into Calm Mind can still be a
# reasonable scout, while handing Shell Smash or Geomancy an uncontested turn changes
# both damage and move order at once.
CATASTROPHIC_SETUP_MOVES = {
    "bellydrum",
    "clangoroussoul",
    "filletaway",
    "geomancy",
    "noretreat",
    "quiverdance",
    "shellsmash",
    "shiftgear",
}

# Single-target shapes. A spread move cannot be redirected.
SINGLE_TARGET = {Target.NORMAL, Target.ANY, Target.ADJACENT_FOE}

# Status moves whose effect a defending TYPE is outright immune to. Small and explicit
# on purpose: this is a rule list, not a computation, so it is gated separately from
# the calculate_damage-driven guards (the calc returns 0 for every status move and so
# cannot distinguish "no effect" from "not a damaging move").
STATUS_TYPE_IMMUNITY: dict[str, set[PokemonType]] = {
    "thunderwave": {PokemonType.GROUND, PokemonType.ELECTRIC},
    "glare": {PokemonType.ELECTRIC},
    "stunspore": {PokemonType.ELECTRIC},
    "willowisp": {PokemonType.FIRE},
    "toxic": {PokemonType.POISON, PokemonType.STEEL},
    "poisonpowder": {PokemonType.POISON, PokemonType.STEEL},
    "poisongas": {PokemonType.POISON, PokemonType.STEEL},
    "leechseed": {PokemonType.GRASS},
}

# Powder/spore moves: Grass types, Overcoat and Safety Goggles are all immune.
POWDER_MOVES = {
    "sleeppowder",
    "stunspore",
    "poisonpowder",
    "spore",
    "cottonspore",
    "ragepowder",
    "magicpowder",
}

# Moves that inflict a major status; they fail on an already-statused target.
_MAJOR_STATUS = {Status.BRN, Status.PAR, Status.PSN, Status.TOX, Status.SLP, Status.FRZ}

# Speed stage multipliers.
_BOOST_MULT = {
    -6: 2 / 8,
    -5: 2 / 7,
    -4: 2 / 6,
    -3: 2 / 5,
    -2: 2 / 4,
    -1: 2 / 3,
    0: 1.0,
    1: 3 / 2,
    2: 4 / 2,
    3: 5 / 2,
    4: 6 / 2,
    5: 7 / 2,
    6: 8 / 2,
}


def _norm(name: str | None) -> str:
    """poke-env ability/item ids are lowercase and punctuation-free; normalise ours."""
    if not name:
        return ""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _effective_speed(battle: DoubleBattle, mon: Pokemon, ours: bool) -> float | None:
    """Effective speed, or None when it cannot be known for certain."""
    base = (mon.stats or {}).get("spe")
    if not base:
        return None
    spe = float(base) * _BOOST_MULT.get(mon.boosts.get("spe", 0), 1.0)
    conds = battle.side_conditions if ours else battle.opponent_side_conditions
    if SideCondition.TAILWIND in conds:
        spe *= 2
    if mon.status == Status.PAR:
        spe *= 0.5
    return spe


def _foe_moves_first(battle: DoubleBattle, me: Pokemon, foe: Pokemon) -> bool:
    """True only when the foe CERTAINLY acts first. Unknown resolves to False.

    Trick Room inverts the comparison, so it has to be read here -- treating a Trick
    Room turn as a normal one would invert the guard's conclusion rather than merely
    weaken it.
    """
    mine = _effective_speed(battle, me, ours=True)
    theirs = _effective_speed(battle, foe, ours=False)
    if mine is None or theirs is None or mine == theirs:
        return False  # a speed tie is a coin flip, not certainty
    if Field.TRICK_ROOM in battle.fields:
        return theirs < mine
    return theirs > mine


def _known_move(mon: Pokemon, move_id: str) -> Move | None:
    """Return a revealed move by id without assuming how the move dict is keyed."""
    for move in (getattr(mon, "moves", None) or {}).values():
        if isinstance(move, Move) and move.id == move_id:
            return move
    return None


def _available_move(
    battle: DoubleBattle, mon: Pokemon, pos: int, move_id: str
) -> Move | None:
    """Return a currently selectable move, falling back to the known moveset.

    Live requests expose ``available_moves`` and correctly remove disabled or
    choice-locked moves. Small tactical fixtures and some reconstructed states do
    not, so those use the Pokemon's known move list instead.
    """
    try:
        available = battle.available_moves[pos]
    except (AttributeError, IndexError, TypeError):
        available = None
    moves = available if available else (getattr(mon, "moves", None) or {}).values()
    return next(
        (move for move in moves if isinstance(move, Move) and move.id == move_id), None
    )


def _priority_is_blocked(
    battle: DoubleBattle, attacker: Pokemon, target: Pokemon, priority: int
) -> bool:
    """Whether a known field/type/ability fact blocks a priority status move."""
    if priority <= 0:
        return False
    if _norm(attacker.ability) == "prankster" and PokemonType.DARK in target.types:
        return True
    psychic_terrain = getattr(Field, "PSYCHIC_TERRAIN", None)
    if (
        psychic_terrain is not None
        and psychic_terrain in getattr(battle, "fields", {})
        and _is_grounded(battle, target)
    ):
        return True
    return any(
        mon is not None
        and not mon.fainted
        and _norm(mon.ability) in PRIORITY_BLOCK_ABILITIES
        for mon in getattr(battle, "active_pokemon", ())
    )


def _known_encore_threat(
    battle: DoubleBattle, target: Pokemon
) -> tuple[Pokemon, Move] | None:
    """A revealed Encore user that can lock ``target`` before its next normal move."""
    encore_effect = getattr(Effect, "ENCORE", None)
    if encore_effect is not None and encore_effect in (target.effects or {}):
        return None  # already locked; this guard cannot undo the current restriction
    for foe in getattr(battle, "opponent_active_pokemon", ()):
        if foe is None or foe.fainted:
            continue
        encore = _known_move(foe, "encore")
        if encore is None:
            continue
        priority = _effective_priority(foe, encore)
        if _priority_is_blocked(battle, foe, target, priority):
            continue
        if priority > 0 or _foe_moves_first(battle, target, foe):
            return foe, encore
    return None


def _candidate_guarantees_ko(
    battle: DoubleBattle, candidate: Candidate, defender: Pokemon, excluded_pos: int
) -> bool:
    """Whether the partner certainly removes a tactical threat this turn."""
    for pos, action in enumerate(candidate.actions):
        if pos == excluded_pos:
            continue
        attacker = battle.active_pokemon[pos]
        if attacker is None or attacker.fainted:
            continue
        order = _decode(battle, action, pos)
        move = getattr(order, "order", None)
        if not isinstance(move, Move) or defender not in resolved_foe_targets(
            battle, order, move
        ):
            continue
        if K.guaranteed_ko(battle, attacker, defender, move):
            return True
    return False


def candidate_exposes_protect_to_encore(
    battle: DoubleBattle, candidate: Candidate
) -> bool:
    """True when first-use Protect hands a revealed faster Encore user a lock.

    This is intentionally narrower than "do not Protect with Whimsicott present":
    Encore must be revealed, it must move before the protected Pokemon's next normal
    action, and the partner must not already guarantee the Encore user's removal.
    """
    for pos, action in enumerate(candidate.actions):
        me = battle.active_pokemon[pos]
        if me is None or me.fainted:
            continue
        order = _decode(battle, action, pos)
        move = getattr(order, "order", None)
        if not isinstance(move, Move) or move.id not in PROTECT_MOVES:
            continue
        threat = _known_encore_threat(battle, me)
        if threat is None:
            continue
        foe, _encore = threat
        if not _candidate_guarantees_ko(battle, candidate, foe, pos):
            return True
    return False


def _expected_damage_value(
    battle: DoubleBattle, attacker: Pokemon, defender: Pokemon, move: Move
) -> float | None:
    """Expected damage fraction, with a type/BP fallback when stats are absent."""
    fraction = K.damage_fraction(battle, attacker, defender, move)
    accuracy = float(move.accuracy or 1.0)
    if fraction is not None:
        return (fraction[0] + fraction[1]) / 2 * accuracy

    # poke-env's full calculator handles Weather Ball correctly, but this fallback is
    # used when a hidden-sheet state cannot be made damage-calculator complete. Do not
    # silently compare its printed 50 BP Normal profile with Heat Wave in that case.
    move_type = move.type
    base_power = float(move.base_power)
    weather = set(getattr(battle, "weather", {}))
    umbrella = _norm(attacker.item) == "utilityumbrella"
    if move.id == "weatherball":
        if not umbrella and weather & {Weather.SUNNYDAY, Weather.DESOLATELAND}:
            move_type, base_power = PokemonType.FIRE, 100.0
        elif not umbrella and weather & {Weather.RAINDANCE, Weather.PRIMORDIALSEA}:
            move_type, base_power = PokemonType.WATER, 100.0
        elif Weather.SANDSTORM in weather:
            move_type, base_power = PokemonType.ROCK, 100.0
        elif weather & {Weather.HAIL, Weather.SNOWSCAPE}:
            move_type, base_power = PokemonType.ICE, 100.0
    try:
        multiplier = defender.damage_multiplier(move_type)
    except Exception:
        return None
    stab = 1.5 if move_type in attacker.types else 1.0
    weather_modifier = 1.0
    if not umbrella:
        if weather & {Weather.SUNNYDAY, Weather.DESOLATELAND}:
            if move_type == PokemonType.FIRE:
                weather_modifier = 1.5
            elif move_type == PokemonType.WATER:
                weather_modifier = 0.5
        elif weather & {Weather.RAINDANCE, Weather.PRIMORDIALSEA}:
            if move_type == PokemonType.WATER:
                weather_modifier = 1.5
            elif move_type == PokemonType.FIRE:
                weather_modifier = 0.5
    spread = 1.0
    if move.target in {Target.ALL_ADJACENT_FOES, Target.ALL_ADJACENT, Target.ALL}:
        live_foes = sum(
            foe is not None and not foe.fainted
            for foe in battle.opponent_active_pokemon
        )
        if live_foes > 1:
            spread = 0.75
    return base_power * accuracy * multiplier * stab * weather_modifier * spread


def candidate_uses_dominated_weather_ball(
    battle: DoubleBattle, candidate: Candidate
) -> bool:
    """True for no-weather Weather Ball when legal Heat Wave is clearly stronger.

    Weather Ball remains valid into Fire/Water/Rock/Dragon targets, through Wide
    Guard, or whenever the damage comparison is close. This catches only the ladder
    failure mode: 50 BP Normal Weather Ball into a neutral target while STAB Heat
    Wave offers at least 50% more expected damage.
    """
    if getattr(battle, "weather", {}):
        return False
    if any(
        foe is not None and _known_move(foe, "wideguard") is not None
        for foe in getattr(battle, "opponent_active_pokemon", ())
    ):
        return False
    for pos, action in enumerate(candidate.actions):
        attacker = battle.active_pokemon[pos]
        if attacker is None or attacker.fainted:
            continue
        order = _decode(battle, action, pos)
        move = getattr(order, "order", None)
        if not isinstance(move, Move) or move.id != "weatherball":
            continue
        targets = resolved_foe_targets(battle, order, move)
        heat_wave = _available_move(battle, attacker, pos, "heatwave")
        if len(targets) != 1 or heat_wave is None:
            continue
        weather_ball_value = _expected_damage_value(battle, attacker, targets[0], move)
        heat_wave_value = _expected_damage_value(
            battle, attacker, targets[0], heat_wave
        )
        if weather_ball_value is None or heat_wave_value is None:
            continue
        if heat_wave_value > 0 and (
            weather_ball_value <= 0 or heat_wave_value >= weather_ball_value * 1.50
        ):
            return True
    return False


def candidate_uses_dominated_single_target_heat_wave(
    battle: DoubleBattle, candidate: Candidate
) -> bool:
    """True when active-weather Weather Ball clearly beats Heat Wave into one foe.

    This is a strict local comparison, not a general preference for Weather Ball.
    Heat Wave keeps its spread value with two foes. With one foe, a currently legal
    Weather Ball must offer at least 10% more expected damage, and Bulletproof makes
    the rule stand down. The margin leaves room for Heat Wave's burn chance instead
    of declaring a tiny damage edge universally decisive.
    """
    foes = [
        foe
        for foe in getattr(battle, "opponent_active_pokemon", ())
        if foe is not None and not foe.fainted
    ]
    if len(foes) != 1 or not getattr(battle, "weather", {}):
        return False
    defender = foes[0]
    if _norm(defender.ability) == "bulletproof":
        return False
    for pos, action in enumerate(candidate.actions):
        attacker = battle.active_pokemon[pos]
        if attacker is None or attacker.fainted:
            continue
        order = _decode(battle, action, pos)
        move = getattr(order, "order", None)
        if not isinstance(move, Move) or move.id != "heatwave":
            continue
        if defender not in resolved_foe_targets(battle, order, move):
            continue
        weather_ball = _available_move(battle, attacker, pos, "weatherball")
        if weather_ball is None:
            continue
        heat_wave_value = _expected_damage_value(battle, attacker, defender, move)
        weather_ball_value = _expected_damage_value(
            battle, attacker, defender, weather_ball
        )
        if (
            heat_wave_value is not None
            and weather_ball_value is not None
            and weather_ball_value >= heat_wave_value * 1.10
        ):
            return True
    return False


def _ally_hit(battle: DoubleBattle, order, pos: int) -> Pokemon | None:
    """Our own Pokemon that this order would damage, if any.

    Two ways to hit your own side: a spread move with ALL_ADJACENT shape (Earthquake,
    Surf, Discharge), or a single-target move aimed at your partner's slot, which
    poke-env encodes as a negative move_target.
    """
    move = getattr(order, "order", None)
    if not isinstance(move, Move):
        return None
    if move.category == MoveCategory.STATUS or move.base_power <= 0:
        return None
    ally = battle.active_pokemon[1 - pos]
    if ally is None or ally.fainted:
        return None
    if move.target == Target.ALL_ADJACENT:
        return ally
    if (getattr(order, "move_target", 0) or 0) < 0:
        return ally
    return None


@dataclass
class Candidate:
    """One joint (slot0, slot1) action pair and what we know about it."""

    actions: tuple[int, int]
    prob: float
    orders: tuple[object | None, object | None] = (None, None)
    demoted_by: str | None = None
    # True when this pair exists only because the tactical layer preserved a switch
    # or non-Mega action outside the policy's normal top-k. Generic learned
    # rerankers must not pull these low-probability actions forward; only the narrow
    # rule that requested the alternative may promote it.
    strategic_only: bool = False


@dataclass
class GuardReport:
    """Which guards fired, for attribution. Mirrors Laplace's `reorders` list.

    `stages` counts only the times a guard changed the FRONT-RUNNER, which is the
    thing that changes play. `demotions` counts every candidate a guard pushed back,
    including the ones that did not reach the top. The two are very different
    diagnoses: no demotions means the rule is inert and can be deleted, while many
    demotions with no stage means the rule is firing but only ever on options the
    policy was not going to take anyway.
    """

    stages: list[str] = field(default_factory=list)
    vetoed: set[tuple[int, int]] = field(default_factory=set)
    demotions: Counter[str] = field(default_factory=Counter)

    def note(self, name: str) -> None:
        if name not in self.stages:
            self.stages.append(name)


def _decode(battle: DoubleBattle, action: int, pos: int):
    """Action index -> BattleOrder, using poke-env's own decoder.

    Deliberately not re-deriving (band, move_slot, target) arithmetic here: doing that
    by hand is exactly how the [-4:] vs [:4] slot bug arose.
    """
    try:
        return DoublesEnv._action_to_order_individual(
            np.int64(action), battle, fake=True, pos=pos
        )
    except Exception:
        return None


def _move_and_targets(
    battle: DoubleBattle, order, pos: int
) -> tuple[Move | None, list[Pokemon]]:
    """Extract (move, defenders actually hit) from a decoded order."""
    move = getattr(order, "order", None)
    if not isinstance(move, Move):
        return None, []
    return move, resolved_foe_targets(battle, order, move)


def resolved_foe_targets(
    battle: DoubleBattle, order: object, move: Move | None = None
) -> list[Pokemon]:
    """Resolve the foes Showdown will actually hit, including auto-retargeting.

    In doubles, a command may still name a slot whose Pokemon fainted earlier in the
    turn. If only one foe remains, Showdown automatically redirects a single-target
    move to it. The old guards inspected the empty requested slot and concluded there
    was nothing to check, which let Last Respects auto-retarget into Normal-type
    Farigiraf for zero damage.
    """
    selected = move if move is not None else getattr(order, "order", None)
    if not isinstance(selected, Move):
        return []
    target = getattr(order, "move_target", 0) or 0
    foes = battle.opponent_active_pokemon
    live = [foe for foe in foes if foe is not None and not foe.fainted]
    name = selected.target.name if selected.target is not None else ""
    if name in {"ALL_ADJACENT_FOES", "ALL_ADJACENT", "ALL"}:
        return live
    if target <= 0:
        return []
    foe = foes[target - 1] if target - 1 < len(foes) else None
    if foe is not None and not foe.fainted:
        return [foe]
    # Showdown's smart retarget is deterministic only when exactly one foe remains.
    return live if len(live) == 1 else []


def _effective_priority(mon: Pokemon | None, move: Move) -> int:
    """Known priority after the common ability modifiers."""
    priority = int(move.priority or 0)
    ability = _norm(mon.ability) if mon is not None else ""
    if ability == "prankster" and move.category == MoveCategory.STATUS:
        priority += 1
    elif (
        mon is not None
        and ability == "galewings"
        and move.type == PokemonType.FLYING
        and (mon.current_hp_fraction or 0.0) >= 1.0
    ):
        priority += 1
    elif ability == "triage" and (move.heal > 0 or move.drain > 0):
        priority += 3
    return priority


def _is_grounded(battle: DoubleBattle, mon: Pokemon) -> bool:
    try:
        return bool(battle.is_grounded(mon))
    except (AttributeError, TypeError):
        return PokemonType.FLYING not in mon.types and _norm(mon.ability) != "levitate"


def _priority_block_prior_probability(battle: DoubleBattle) -> float:
    """Probability an active hidden foe has a side-wide priority blocker.

    This is intentionally separate from the hard revealed-ability check. It is only
    used above a very high threshold (99% by default), and a demoted action remains a
    legal fallback under the guard contract.
    """
    try:
        from vgc_bench.src.policy_player import PolicyPlayer

        if not PolicyPlayer.moveset_prior_enabled():
            return 0.0
    except ImportError:
        return 0.0

    global _JOINT_SET_CACHE
    if _JOINT_SET_CACHE is None:
        try:
            root = Path(__file__).resolve().parents[2]
            _JOINT_SET_CACHE = json.loads(
                (root / "data" / "joint_sets_regmb.json").read_text()
            )
        except (OSError, ValueError):
            _JOINT_SET_CACHE = {}
    joint_sets = _JOINT_SET_CACHE or {}

    no_block_probability = 1.0
    for foe in battle.opponent_active_pokemon:
        if foe is None or foe.fainted:
            continue
        if foe.ability is not None:
            probability = float(_norm(foe.ability) in PRIORITY_BLOCK_ABILITIES)
        else:
            entry = joint_sets.get(_norm(foe.base_species))
            if not entry:
                probability = 0.0
            else:
                seen_moves = set(foe.moves)
                seen_item = foe.item if foe.item not in (None, "unknown_item") else None
                sets = entry.get("sets", [])
                consistent = [
                    candidate
                    for candidate in sets
                    if seen_moves.issubset(set(candidate.get("moves", [])))
                    and (seen_item is None or candidate.get("item") == seen_item)
                ]
                candidates = consistent or sets
                total = sum(
                    max(0, int(candidate.get("count", 0))) for candidate in candidates
                )
                blocked = sum(
                    max(0, int(candidate.get("count", 0)))
                    for candidate in candidates
                    if _norm(candidate.get("ability")) in PRIORITY_BLOCK_ABILITIES
                )
                probability = blocked / total if total else 0.0
        no_block_probability *= 1.0 - probability
    return 1.0 - no_block_probability


def _hidden_move_probability(mon: Pokemon, move_ids: set[str]) -> float:
    """Posterior probability that a hidden set contains one of ``move_ids``.

    Revealed moves are facts. Otherwise condition the top-player joint-set table on
    every move/item/ability already observed. This estimates set *availability*, not
    whether the opponent presses the move this turn; callers therefore use a high
    threshold and only correct extreme no-progress lines.
    """
    revealed = {_norm(move.id) for move in (mon.moves or {}).values()}
    if revealed & move_ids:
        return 1.0

    global _JOINT_SET_CACHE
    if _JOINT_SET_CACHE is None:
        try:
            root = Path(__file__).resolve().parents[2]
            _JOINT_SET_CACHE = json.loads(
                (root / "data" / "joint_sets_regmb.json").read_text()
            )
        except (OSError, ValueError):
            _JOINT_SET_CACHE = {}
    entry = (_JOINT_SET_CACHE or {}).get(_norm(mon.base_species)) or {}
    sets = entry.get("sets") or []
    seen_item = mon.item if mon.item not in (None, "unknown_item") else None
    seen_ability = mon.ability if mon.ability not in (None, "unknown_ability") else None
    consistent = [
        candidate
        for candidate in sets
        if revealed.issubset({_norm(move) for move in candidate.get("moves", [])})
        and (seen_item is None or _norm(candidate.get("item")) == _norm(seen_item))
        and (
            seen_ability is None
            or _norm(candidate.get("ability")) == _norm(seen_ability)
        )
    ]
    candidates = consistent or sets
    if not candidates:
        return 0.0
    count_total = sum(
        max(0, int(candidate.get("count", 0))) for candidate in candidates
    )

    def weight(candidate) -> float:
        if count_total:
            return float(max(0, int(candidate.get("count", 0))))
        return max(0.0, float(candidate.get("prob", 0.0)))

    total = sum(weight(candidate) for candidate in candidates)
    setup = sum(
        weight(candidate)
        for candidate in candidates
        if {_norm(move) for move in candidate.get("moves", [])} & move_ids
    )
    return setup / total if total else 0.0


def _candidate_damage_to(
    battle: DoubleBattle, candidate: Candidate, defender: Pokemon
) -> float:
    """Expected joint damage directed at one specific opposing Pokemon."""
    damage = 0.0
    for pos, action in enumerate(candidate.actions):
        attacker = battle.active_pokemon[pos]
        if attacker is None or attacker.fainted:
            continue
        order = _decode(battle, action, pos)
        move = getattr(order, "order", None)
        if (
            not isinstance(move, Move)
            or move.category == MoveCategory.STATUS
            or move.base_power <= 0
            or defender not in resolved_foe_targets(battle, order, move)
        ):
            continue
        value = _expected_damage_value(battle, attacker, defender, move)
        if value is not None:
            damage += max(0.0, value)
    return damage


def candidate_gives_catastrophic_free_setup(
    battle: DoubleBattle, candidate: Candidate
) -> bool:
    """Whether Double Protect gives a likely snowball setup user a free turn."""
    ours = [mon for mon in battle.active_pokemon if mon is not None and not mon.fainted]
    if len(ours) < 2:
        return False
    orders = [
        _decode(battle, action, pos) for pos, action in enumerate(candidate.actions)
    ]
    if not all(
        isinstance(getattr(order, "order", None), Move)
        and getattr(order, "order").id in PROTECT_MOVES
        for order in orders
    ):
        return False
    # Protect can be the correct way to consume the last turn of opposing speed
    # control. Do not let a setup prior overrule that concrete board objective.
    if Field.TRICK_ROOM in getattr(battle, "fields", {}):
        return False
    if (SideCondition.TAILWIND in getattr(battle, "side_conditions", {})) != (
        SideCondition.TAILWIND in getattr(battle, "opponent_side_conditions", {})
    ):
        return False
    threshold = float(os.environ.get("VGC_FREE_SETUP_PROBABILITY", "0.70"))
    return any(
        foe is not None
        and not foe.fainted
        and _hidden_move_probability(foe, CATASTROPHIC_SETUP_MOVES) >= threshold
        for foe in battle.opponent_active_pokemon
    )


def battle_can_contest_catastrophic_setup(battle: DoubleBattle) -> bool:
    """Whether either active has a legal move that meaningfully damages the threat."""
    threshold = float(os.environ.get("VGC_FREE_SETUP_PROBABILITY", "0.70"))
    threats = [
        foe
        for foe in battle.opponent_active_pokemon
        if foe is not None
        and not foe.fainted
        and _hidden_move_probability(foe, CATASTROPHIC_SETUP_MOVES) >= threshold
    ]
    for pos, attacker in enumerate(battle.active_pokemon):
        if attacker is None or attacker.fainted:
            continue
        try:
            available = battle.available_moves[pos]
        except (AttributeError, IndexError, TypeError):
            available = list((attacker.moves or {}).values())
        for move in available:
            if (
                not isinstance(move, Move)
                or move.category == MoveCategory.STATUS
                or move.base_power <= 0
            ):
                continue
            for threat in threats:
                value = _expected_damage_value(battle, attacker, threat, move)
                if value is not None and value >= 0.12:
                    return True
    return False


def build_candidates(
    policy, obs_dict, mask: torch.Tensor, top_k: int = 6
) -> tuple[list[Candidate], torch.Tensor]:
    """Top-k x top-k joint pairs ranked by p(slot0) * p(slot1 | slot0).

    This is our analogue of Laplace's pooled MCTS visit shares: the policy's own
    ranking, which guards then permute. Costs no extra network passes -- only
    softmaxes over the already-computed logits.
    """
    with torch.no_grad():
        action_logits, value = policy.get_logits(obs_dict, actor_grad=False)
        dist0 = policy.get_dist_from_logits(action_logits, mask)
        probs0 = dist0.distribution[0].probs[0]
        top0 = _candidate_action_prefix(probs0, top_k)

        cands: list[Candidate] = []
        for p0, a0, strategic0 in top0:
            prev = torch.tensor([[a0]], device=action_logits.device)
            dist1 = policy.get_dist_from_logits(action_logits, mask, prev)
            probs1 = dist1.distribution[1].probs[0]
            top1 = _candidate_action_prefix(probs1, top_k)
            if not top1:
                continue
            for p1, a1, strategic1 in top1:
                cands.append(
                    Candidate(
                        actions=(a0, a1),
                        prob=p0 * p1,
                        strategic_only=strategic0 or strategic1,
                    )
                )

    cands.sort(key=lambda c: -c.prob)
    return cands, value


def _candidate_action_prefix(
    probabilities: torch.Tensor, top_k: int
) -> list[tuple[float, int, bool]]:
    """Top policy actions plus one switch and one ungimmicked move when legal.

    A pure top-k prefix repeatedly removed the only strategically relevant option:
    switching a Yawned or -2 Attack Pokemon, and declining Mega Floette to preserve
    Charizard Y's weather control. These two additions keep the policy ordering while
    guaranteeing that the tactical layer can at least inspect those alternatives.
    """
    legal = int((probabilities > 0).sum().item())
    if legal == 0:
        return []
    count = min(top_k, legal)
    policy_top = {
        int(index) for index in torch.topk(probabilities, count).indices.tolist()
    }
    selected = set(policy_top)
    if os.environ.get("VGC_STRATEGIC_CANDIDATES", "1") == "0":
        return sorted(
            ((float(probabilities[index].item()), index, False) for index in selected),
            reverse=True,
        )
    # poke-env doubles actions: 1..6 switches; 7..26 ordinary move/target pairs.
    # Mega/Z/Dmax/Tera occupy later copies of the same move bands.
    for start, stop in ((1, 7), (7, 27)):
        band = probabilities[start:stop]
        if band.numel() == 0 or not bool((band > 0).any().item()):
            continue
        selected.add(start + int(torch.argmax(band).item()))
    return sorted(
        (
            (float(probabilities[index].item()), index, index not in policy_top)
            for index in selected
        ),
        reverse=True,
    )


def _demote(cands: list[Candidate], dead: set[int], name: str, report: GuardReport):
    """Move newly flagged live candidates behind all prior-vetoed candidates.

    Guards compose over the surviving prefix.  A later guard must never pull a
    candidate that an earlier guard vetoed back into consideration, and it must
    stand down if it would veto every candidate still alive.
    """
    live = {i for i, candidate in enumerate(cands) if candidate.demoted_by is None}
    newly_dead = dead & live
    if not newly_dead or len(newly_dead) >= len(live):
        return cands
    before = cands[0].actions
    keep = [c for i, c in enumerate(cands) if i in live and i not in newly_dead]
    prior = [c for i, c in enumerate(cands) if i not in live]
    push = [c for i, c in enumerate(cands) if i in newly_dead]
    for c in push:
        report.demotions[name] += 1
        c.demoted_by = name
        report.vetoed.add(c.actions)
    out = keep + prior + push
    if out[0].actions != before:
        report.note(name)
    return out


def guard_zero_damage(battle, cands, report) -> list[Candidate]:
    """G1: demote pairs whose damaging move cannot damage anything it hits.

    Supersedes a type-chart lookup: calculate_damage already accounts for absorb
    abilities, Tera typing, Ring Target and Tera Blast. Protected defenders are
    excluded -- the calc returns 0 for them too, but that is temporary, not immunity.
    """
    dead = set()
    for i, c in enumerate(cands):
        useless = False
        for pos, action in enumerate(c.actions):
            order = _decode(battle, action, pos)
            move, hit = _move_and_targets(battle, order, pos)
            if move is None or not hit:
                continue
            if move.category == MoveCategory.STATUS or move.base_power <= 0:
                continue
            me = battle.active_pokemon[pos]
            if me is None:
                continue
            # A spread move nullified against one foe but not the other is fine.
            if all(K.deals_no_damage(battle, me, foe, move) for foe in hit):
                useless = True
        if useless:
            dead.add(i)
    return _demote(cands, dead, "zero_damage", report)


def guard_first_turn(battle, cands, report) -> list[Candidate]:
    """G3: demote Fake Out / First Impression / Mat Block after the first turn.

    Showdown keeps them selectable; they simply fail.
    """
    dead = set()
    for i, c in enumerate(cands):
        for pos, action in enumerate(c.actions):
            me = battle.active_pokemon[pos]
            if me is None or me.first_turn:
                continue
            order = _decode(battle, action, pos)
            move = getattr(order, "order", None)
            if isinstance(move, Move) and move.id in FIRST_TURN_ONLY:
                dead.add(i)
    return _demote(cands, dead, "first_turn", report)


def guard_priority_block(battle, cands, report) -> list[Candidate]:
    """Demote priority moves a known mechanic or near-certain usage prior blocks.

    Armor Tail, Dazzling and Queenly Majesty protect the whole opposing side, not
    only their user. Psychic Terrain is target-specific and therefore stands down for
    airborne foes. Once Showdown reveals an ability through a ``|cant|`` message,
    ``pokeenv_patches`` records it as a hard fact. Before that, a separately attributed
    soft demotion is allowed only when conditioned replay usage exceeds 99%.
    """
    foes = [
        foe
        for foe in (getattr(battle, "opponent_active_pokemon", ()) or ())
        if foe is not None and not foe.fainted
    ]
    side_blocked = any(_norm(foe.ability) in PRIORITY_BLOCK_ABILITIES for foe in foes)
    try:
        prior_probability = _priority_block_prior_probability(battle)
    except (AttributeError, TypeError):
        prior_probability = 0.0
    prior_blocked = not side_blocked and prior_probability >= float(
        os.environ.get("VGC_PRIORITY_BLOCK_PRIOR_THRESHOLD", "0.99")
    )
    psychic_terrain = Field.PSYCHIC_TERRAIN in getattr(battle, "fields", {})
    if not side_blocked and not prior_blocked and not psychic_terrain:
        return cands

    hard_dead = set()
    prior_dead = set()
    for i, candidate in enumerate(cands):
        for pos, action in enumerate(candidate.actions):
            order = _decode(battle, action, pos)
            move = getattr(order, "order", None)
            me = battle.active_pokemon[pos]
            if not isinstance(move, Move) or _effective_priority(me, move) <= 0:
                continue
            hit = resolved_foe_targets(battle, order, move)
            if not hit:
                continue
            terrain_blocked = psychic_terrain and all(
                _is_grounded(battle, foe) for foe in hit
            )
            if side_blocked or terrain_blocked:
                hard_dead.add(i)
                break
            if prior_blocked:
                prior_dead.add(i)
                break
    prior_actions = {cands[index].actions for index in prior_dead}
    cands = _demote(cands, hard_dead, "priority_block", report)
    prior_dead = {
        index
        for index, candidate in enumerate(cands)
        if candidate.actions in prior_actions
    }
    return _demote(cands, prior_dead, "priority_block_prior", report)


def candidate_uses_blocked_priority(battle: DoubleBattle, candidate: Candidate) -> bool:
    """Whether a single line is blocked by a known or near-certain side mechanic.

    Multi-candidate guards demote rather than delete and intentionally stand down if
    every candidate is bad. A reused contingent plan has no alternatives, so it
    needs this strict predicate before the planner trusts the cached line.
    """
    foes = [
        foe
        for foe in (getattr(battle, "opponent_active_pokemon", ()) or ())
        if foe is not None and not foe.fainted
    ]
    side_blocked = any(_norm(foe.ability) in PRIORITY_BLOCK_ABILITIES for foe in foes)
    try:
        prior_probability = _priority_block_prior_probability(battle)
    except (AttributeError, TypeError):
        prior_probability = 0.0
    prior_blocked = not side_blocked and prior_probability >= float(
        os.environ.get("VGC_PRIORITY_BLOCK_PRIOR_THRESHOLD", "0.99")
    )
    psychic_terrain = Field.PSYCHIC_TERRAIN in getattr(battle, "fields", {})
    if not side_blocked and not prior_blocked and not psychic_terrain:
        return False
    for pos, action in enumerate(candidate.actions):
        order = _decode(battle, action, pos)
        move = getattr(order, "order", None)
        actives = getattr(battle, "active_pokemon", ()) or ()
        me = actives[pos] if pos < len(actives) else None
        if not isinstance(move, Move) or _effective_priority(me, move) <= 0:
            continue
        hit = resolved_foe_targets(battle, order, move)
        if not hit:
            continue
        if side_blocked or prior_blocked:
            return True
        if psychic_terrain and all(_is_grounded(battle, foe) for foe in hit):
            return True
    return False


def _ko_promotion_mode() -> str:
    """VGC_KO_PROMOTION_MODE: off (historical), robust, or skip.

    Defaults to "off" until the realized-KO counting validation promotes a mode,
    per the project rule that rare-firing rules are validated by correctness
    counting before they change production behavior.
    """
    mode = os.environ.get("VGC_KO_PROMOTION_MODE", "off").lower()
    return mode if mode in ("off", "robust", "skip") else "off"


def _candidate_guaranteed_progress(
    battle: DoubleBattle, candidate: Candidate, promotion_bound: bool = False
) -> tuple[int, float, bool]:
    """Return guaranteed KOs, minimum damage, and whether the pair is ally-safe.

    Only 100%-accurate moves contribute. An ALL_ADJACENT move is considered safe
    when its partner Protects or is mechanically immune. This is intentionally much
    narrower than the old generic ally-damage heuristic that hurt evaluation.

    promotion_bound tightens the KO test for the PROMOTING guards: when a
    defender's stats were synthesized from absence (sheet denied), its min-roll
    damage is scaled to the worst plausible defensive spread ("robust" mode),
    or its KO claims are refused outright ("skip" mode), plus a hidden-item
    margin (VGC_KO_PROMOTION_MARGIN, default 0.10, covering Sitrus-class
    recovery). Mode "off" reproduces historical behavior exactly.
    """
    protected: set[int] = set()
    decoded: list[object | None] = []
    for pos, action in enumerate(candidate.actions):
        order = _decode(battle, action, pos)
        decoded.append(order)
        move = getattr(order, "order", None)
        if isinstance(move, Move) and move.id in PROTECT_MOVES:
            protected.add(pos)

    mode = _ko_promotion_mode() if promotion_bound else "off"
    margin = 0.0
    if mode != "off":
        try:
            margin = float(os.environ.get("VGC_KO_PROMOTION_MARGIN", "0.10"))
        except ValueError:
            margin = 0.10

    damage: dict[int, float] = {}
    foes_by_id: dict[int, Pokemon] = {}
    synthesized: set[int] = set()
    for pos, order in enumerate(decoded):
        move, hit = _move_and_targets(battle, order, pos)
        me = battle.active_pokemon[pos]
        if move is None or me is None or move.category == MoveCategory.STATUS:
            continue
        if move.target == Target.ALL_ADJACENT:
            ally = _ally_hit(battle, order, pos)
            if ally is not None:
                ally_safe = (1 - pos) in protected or K.deals_no_damage(
                    battle, me, ally, move
                )
                if not ally_safe:
                    return 0, 0.0, False
        if float(move.accuracy or 0.0) < 1.0:
            continue
        for foe in hit:
            fraction = K.damage_fraction(battle, me, foe, move)
            if fraction is None:
                continue
            contribution = max(0.0, fraction[0])
            if mode != "off" and K.stats_were_synthesized(foe):
                synthesized.add(id(foe))
                if mode == "skip":
                    contribution = 0.0
                else:
                    contribution *= K.robust_ko_scale(foe, move)
            key = id(foe)
            foes_by_id[key] = foe
            damage[key] = damage.get(key, 0.0) + contribution

    kos = sum(
        amount
        >= float(foes_by_id[key].current_hp_fraction or 0.0)
        + (margin if key in synthesized else 0.0)
        for key, amount in damage.items()
    )
    return int(kos), sum(damage.values()), True


def _uses_unplanned_tera(candidate: Candidate, argmax: tuple[int, int]) -> bool:
    return candidate.actions != argmax and any(
        86 < action <= 106 for action in candidate.actions
    )


def _has_ally_spread_move(battle: DoubleBattle, candidate: Candidate) -> bool:
    for pos, action in enumerate(candidate.actions):
        order = _decode(battle, action, pos)
        move = getattr(order, "order", None)
        if (
            isinstance(move, Move)
            and move.target == Target.ALL_ADJACENT
            and _ally_hit(battle, order, pos) is not None
        ):
            return True
    return False


def _has_resisted_attack(battle: DoubleBattle, candidate: Candidate) -> bool:
    """Whether this pair spends an action on known resisted direct damage."""
    for pos, action in enumerate(candidate.actions):
        order = _decode(battle, action, pos)
        move, targets = _move_and_targets(battle, order, pos)
        if move is None or move.category == MoveCategory.STATUS:
            continue
        for target in targets:
            multiplier = K.type_multiplier(move, target)
            if multiplier is not None and 0.0 < multiplier < 1.0:
                return True
    return False


def _promote_candidate(
    cands: list[Candidate], best: Candidate, name: str, report: GuardReport
) -> list[Candidate]:
    live = [candidate for candidate in cands if candidate.demoted_by is None]
    dead = [candidate for candidate in cands if candidate.demoted_by is not None]
    if not live or live[0] is best:
        return cands
    report.note(name)
    return [best] + [candidate for candidate in live if candidate is not best] + dead


def guard_guaranteed_ko(battle, cands, report) -> list[Candidate]:
    """Promote an ally-safe pair only when it adds a guaranteed opposing KO.

    Covers Earthquake beside Flying Charizard, solo Earthquake into Archaludon, and
    Last Respects over weak priority into Pelipper. Unlike the rejected broad damage
    tiebreak, this does nothing unless the minimum roll adds a KO. It cannot resurrect
    an earlier demotion or promote an unplanned Tera action.
    """
    live = [candidate for candidate in cands if candidate.demoted_by is None]
    if len(live) < 2:
        return cands
    top_kos, _, top_valid = _candidate_guaranteed_progress(
        battle, live[0], promotion_bound=True
    )
    if not top_valid:
        top_kos = 0
    argmax = live[0].actions

    scored = []
    scope = os.environ.get("VGC_GUARANTEED_KO_SCOPE", "resisted")
    safe_spread_only = scope == "safe_spread"
    broad = scope == "broad"
    min_ratio = float(os.environ.get("VGC_GUARANTEED_KO_MIN_RATIO", "0.48"))
    top_is_resisted: bool | None = None
    for candidate in live[1:]:
        kos, damage, valid = _candidate_guaranteed_progress(
            battle, candidate, promotion_bound=True
        )
        safe_spread = _has_ally_spread_move(battle, candidate)
        # The old safe-Earthquake rule is a hard mechanics correction and may cross
        # a large policy gap. Ordinary attacks need meaningful policy support; the
        # first broad version promoted them 184 times per 100 battles and made the
        # matched result worse.
        near_policy = candidate.prob >= live[0].prob * min_ratio
        if not safe_spread and not safe_spread_only and top_is_resisted is None:
            top_is_resisted = _has_resisted_attack(battle, live[0])
        in_scope = safe_spread or (
            not safe_spread_only and near_policy and (broad or bool(top_is_resisted))
        )
        if (
            valid
            and in_scope
            and kos > top_kos
            and not _uses_unplanned_tera(candidate, argmax)
        ):
            scored.append((kos, damage, candidate.prob, candidate))
    if not scored:
        return cands
    best = max(scored, key=lambda entry: entry[:3])[3]
    return _promote_candidate(cands, best, "guaranteed_ko", report)


# Compatibility for existing experiments/imports; the rule is now deliberately
# broader than safe spread but retains the same strict guaranteed-KO threshold.
guard_safe_spread_ko = guard_guaranteed_ko


def _order_signature(order: object | None) -> tuple[object, ...]:
    selected = getattr(order, "order", None)
    if isinstance(selected, Move):
        return ("move", selected.id, int(getattr(order, "move_target", 0) or 0))
    if isinstance(selected, Pokemon):
        return ("switch", _norm(selected.base_species))
    return ("pass",)


def guard_reserve_weather_mega(battle, cands, report) -> list[Candidate]:
    """Preserve Charizard Y's weather control against active rain strategies.

    This fixed team carries two Mega stones but can spend only one. Mega Floette is
    demoted only when rain is already active, a selected healthy Charizard Y remains,
    and an otherwise identical non-Mega line gives up no guaranteed KO.
    """
    live = [candidate for candidate in cands if candidate.demoted_by is None]
    rain_active = any(
        weather in battle.weather
        for weather in {Weather.RAINDANCE, Weather.PRIMORDIALSEA}
    )
    opponent_species = {
        _norm(mon.base_species)
        for mon in getattr(battle, "opponent_team", {}).values()
        if mon is not None and not mon.fainted
    }
    active_species = {
        _norm(mon.base_species)
        for mon in battle.opponent_active_pokemon
        if mon is not None and not mon.fainted
    }
    # The exact ladder failure happened one action before Pelipper switched in, so
    # checking only current weather was one turn too late. Keep this prediction
    # deliberately narrow: a visible Pelipper/Politoed plus an active Swampert is a
    # concrete rain-speed board, not a generic guess based on team aesthetics.
    weather_v2 = os.environ.get("VGC_WEATHER_MEGA_V2", "1") != "0"
    rain_abusers = (
        {
            "archaludon",
            "barraskewda",
            "basculegion",
            "kingdra",
            "ludicolo",
            "swampert",
            "swampertmega",
        }
        if weather_v2
        else {"swampert", "swampertmega"}
    )
    visible_rain_core = bool(
        opponent_species & {"pelipper", "politoed"} and active_species & rain_abusers
    )
    if len(live) < 2 or not (rain_active or visible_rain_core):
        return cands
    charizard_ready = any(
        _norm(mon.base_species) == "charizard"
        and not mon.fainted
        and _norm(mon.item) == "charizarditey"
        and bool(getattr(mon, "selected_in_teampreview", False))
        for mon in battle.team.values()
    )
    if not charizard_ready:
        return cands

    top_orders = tuple(
        _decode(battle, action, pos) for pos, action in enumerate(live[0].actions)
    )
    mega_positions = [
        pos
        for pos, order in enumerate(top_orders)
        if bool(getattr(order, "mega", False))
        and isinstance(getattr(order, "order", None), Move)
        and _norm(battle.active_pokemon[pos].base_species) != "charizard"
    ]
    if not mega_positions:
        return cands
    signatures = tuple(_order_signature(order) for order in top_orders)
    top_kos, _, _ = _candidate_guaranteed_progress(battle, live[0])
    top_attacks, top_damage = _candidate_attack_progress(battle, live[0])
    alternatives = []
    for candidate in live[1:]:
        orders = tuple(
            _decode(battle, action, pos) for pos, action in enumerate(candidate.actions)
        )
        if any(bool(getattr(order, "mega", False)) for order in orders):
            continue
        # Preserve the partner's action exactly. The non-Mega slot may choose a
        # different attack: requiring an identical move made the rule fire on Turn 1
        # and then disappear on Turn 2 whenever the policy's best ordinary Floette
        # move changed from Dazzling Gleam to Moonblast.
        if any(
            (not weather_v2 or pos not in mega_positions)
            and _order_signature(orders[pos]) != signatures[pos]
            for pos in range(2)
        ):
            continue
        kos, _, valid = _candidate_guaranteed_progress(battle, candidate)
        if not valid or kos < top_kos:
            continue
        # Saving Charizard Y is valuable, but not at the price of turning an
        # attacking slot into Protect and surrendering the whole turn.  This exact
        # failure appeared locally versus Pelipper + Basculegion: Protect +
        # Dazzling Gleam was replaced by double Protect, after which both actives
        # were removed.  Require the non-Mega line to retain the number of useful
        # attacks and at least half of the immediate damage floor.
        attacks, damage = _candidate_attack_progress(battle, candidate)
        if top_attacks and (attacks < top_attacks or damage + 1e-9 < top_damage * 0.50):
            continue
        alternatives.append(candidate)
    if not alternatives:
        return cands
    best = max(alternatives, key=lambda candidate: candidate.prob)
    return _promote_candidate(cands, best, "reserve_weather_mega", report)


def _has_remaining_reserve(battle: DoubleBattle, ours: bool) -> bool:
    """Whether a usable Pokemon can still replace one of the active pair.

    poke-env records our four preview selections, but cannot know which four the
    opponent brought until they are revealed. Treat that uncertainty as "a reserve
    may exist". Only declare the opposing bench empty after all ``max_team_size``
    brought Pokemon have been revealed and every survivor is already active.
    """
    team = battle.team if ours else battle.opponent_team
    active = battle.active_pokemon if ours else battle.opponent_active_pokemon
    active_ids = {id(mon) for mon in active if mon is not None}
    if ours:
        return any(
            id(mon) not in active_ids
            and not mon.fainted
            and bool(getattr(mon, "selected_in_teampreview", False))
            for mon in team.values()
        )

    revealed = [
        mon
        for mon in team.values()
        if bool(getattr(mon, "revealed", False)) or id(mon) in active_ids or mon.fainted
    ]
    max_team_size = getattr(battle, "max_team_size", None)
    if max_team_size is None or len(revealed) < int(max_team_size):
        return True
    return any(id(mon) not in active_ids and not mon.fainted for mon in revealed)


def _candidate_attack_progress(
    battle: DoubleBattle, candidate: Candidate
) -> tuple[int, float]:
    attacks = 0
    damage = 0.0
    for pos, action in enumerate(candidate.actions):
        order = _decode(battle, action, pos)
        move, targets = _move_and_targets(battle, order, pos)
        attacker = battle.active_pokemon[pos]
        if (
            move is None
            or attacker is None
            or move.category == MoveCategory.STATUS
            or move.base_power <= 0
        ):
            continue
        useful = False
        for target in targets:
            fraction = K.damage_fraction(battle, attacker, target, move)
            if fraction is None or fraction[1] <= 0:
                continue
            useful = True
            damage += max(0.0, fraction[0]) * float(move.accuracy or 0.0)
        attacks += int(useful)
    return attacks, damage


def candidate_repeats_solo_protect(battle: DoubleBattle, candidate: Candidate) -> bool:
    """Whether the final Pokemon repeats Protect without stalling speed control."""
    ours = [mon for mon in battle.active_pokemon if mon is not None and not mon.fainted]
    if (
        len(ours) != 1
        or _has_remaining_reserve(battle, True)
        or Field.TRICK_ROOM in battle.fields
        or (SideCondition.TAILWIND in battle.side_conditions)
        != (SideCondition.TAILWIND in battle.opponent_side_conditions)
    ):
        return False
    for pos, action in enumerate(candidate.actions):
        me = battle.active_pokemon[pos]
        if me is None or (me.protect_counter or 0) < 1:
            continue
        order = _decode(battle, action, pos)
        move = getattr(order, "order", None)
        if isinstance(move, Move) and move.id in PROTECT_MOVES:
            return True
    return False


def solo_has_attack_progress(battle: DoubleBattle) -> bool:
    """Whether the final Pokemon currently has a legal damaging move."""
    foes = [
        foe
        for foe in battle.opponent_active_pokemon
        if foe is not None and not foe.fainted
    ]
    for pos, attacker in enumerate(battle.active_pokemon):
        if attacker is None or attacker.fainted:
            continue
        try:
            available = battle.available_moves[pos]
        except (AttributeError, IndexError, TypeError):
            available = list((attacker.moves or {}).values())
        for move in available:
            if (
                not isinstance(move, Move)
                or move.category == MoveCategory.STATUS
                or move.base_power <= 0
            ):
                continue
            if any(
                (value := _expected_damage_value(battle, attacker, foe, move))
                is not None
                and value > 0
                for foe in foes
            ):
                return True
    return False


def guard_endgame_progress(battle, cands, report) -> list[Candidate]:
    """Do not spend a reserve-less endgame turn making zero progress.

    Double Protect can correctly stall asymmetric Trick Room or Tailwind. When both
    sides have exactly their two active Pokemon left and speed-control presence is
    symmetric, however, it only postpones the same board. A repeated one-slot
    Protect is also rejected when a policy-supported attack line exists, including
    the last-Pokemon 1v2 from the audited Blastoise game.
    """
    ours = [mon for mon in battle.active_pokemon if mon is not None and not mon.fainted]
    foes = [
        mon
        for mon in battle.opponent_active_pokemon
        if mon is not None and not mon.fainted
    ]
    live = [candidate for candidate in cands if candidate.demoted_by is None]
    speed_stall = Field.TRICK_ROOM in battle.fields or (
        SideCondition.TAILWIND in battle.side_conditions
    ) != (SideCondition.TAILWIND in battle.opponent_side_conditions)
    if (
        len(ours) == 1
        and len(foes) >= 1
        and len(live) >= 2
        and not _has_remaining_reserve(battle, True)
        and not speed_stall
    ):
        top_orders = [
            _decode(battle, action, pos) for pos, action in enumerate(live[0].actions)
        ]
        repeated = any(
            battle.active_pokemon[pos] is not None
            and (battle.active_pokemon[pos].protect_counter or 0) >= 1
            and isinstance(getattr(order, "order", None), Move)
            and getattr(order, "order").id in PROTECT_MOVES
            for pos, order in enumerate(top_orders)
        )
        if repeated:
            top_attacks, top_damage = _candidate_attack_progress(battle, live[0])
            alternatives = []
            for candidate in live[1:]:
                if candidate.prob < live[0].prob * 0.20:
                    continue
                attacks, damage = _candidate_attack_progress(battle, candidate)
                if attacks > top_attacks and damage > top_damage:
                    alternatives.append((attacks, damage, candidate.prob, candidate))
            if alternatives:
                best = max(alternatives, key=lambda entry: entry[:3])[3]
                return _promote_candidate(cands, best, "endgame_progress", report)

    if (
        len(ours) != 2
        or len(foes) != 2
        or len(live) < 2
        or _has_remaining_reserve(battle, True)
        or _has_remaining_reserve(battle, False)
        or speed_stall
    ):
        return cands

    top_orders = [
        _decode(battle, action, pos) for pos, action in enumerate(live[0].actions)
    ]
    protect_positions = {
        pos
        for pos, order in enumerate(top_orders)
        if isinstance(getattr(order, "order", None), Move)
        and getattr(order, "order").id in PROTECT_MOVES
    }
    double_protect = len(protect_positions) == 2
    repeated_positions = {
        pos
        for pos in protect_positions
        if (battle.active_pokemon[pos].protect_counter or 0) >= 1
    }
    if not double_protect and not repeated_positions:
        return cands

    top_attacks, top_damage = _candidate_attack_progress(battle, live[0])
    minimum_ratio = 0.002 if double_protect else 0.20
    alternatives = []
    for candidate in live[1:]:
        if candidate.prob < live[0].prob * minimum_ratio:
            continue
        attacks, damage = _candidate_attack_progress(battle, candidate)
        if attacks <= top_attacks or damage <= top_damage:
            continue
        alternatives.append((attacks, damage, candidate.prob, candidate))
    if not alternatives:
        return cands
    best = max(alternatives, key=lambda entry: entry[:3])[3]
    return _promote_candidate(cands, best, "endgame_progress", report)


def guard_yawn_switch(battle, cands, report) -> list[Candidate]:
    """Switch a valuable Yawned Pokemon before guaranteed end-of-turn sleep.

    Stands down if the current pair guarantees the end of the battle or no legal
    switch candidate survived. This is future-state reasoning: the attack this turn
    can look strong while the resulting sleeping board is losing.
    """
    live = [candidate for candidate in cands if candidate.demoted_by is None]
    if len(live) < 2:
        return cands
    yawning = {
        pos
        for pos, mon in enumerate(battle.active_pokemon)
        if mon is not None
        and not mon.fainted
        and mon.status != Status.SLP
        and Effect.YAWN in mon.effects
        and float(mon.current_hp_fraction or 0.0) > 0.25
    }
    if not yawning:
        return cands
    top_kos, _, _ = _candidate_guaranteed_progress(battle, live[0])
    remaining_foes = sum(
        not mon.fainted for mon in getattr(battle, "opponent_team", {}).values()
    )
    if remaining_foes and top_kos >= remaining_foes:
        return cands

    top_orders = tuple(
        _decode(battle, action, pos) for pos, action in enumerate(live[0].actions)
    )
    alternatives = []
    for candidate in live[1:]:
        orders = tuple(
            _decode(battle, action, pos) for pos, action in enumerate(candidate.actions)
        )
        if not all(
            isinstance(getattr(orders[pos], "order", None), Pokemon) for pos in yawning
        ):
            continue
        partner_matches = sum(
            _order_signature(orders[pos]) == _order_signature(top_orders[pos])
            for pos in range(2)
            if pos not in yawning
        )
        alternatives.append((partner_matches, candidate.prob, candidate))
    if not alternatives:
        return cands
    best = max(alternatives, key=lambda entry: entry[:2])[2]
    return _promote_candidate(cands, best, "yawn_switch", report)


def guard_severe_attack_drop_switch(battle, cands, report) -> list[Candidate]:
    """Offer a reset when a healthy physical attacker is crippled to -2 Attack."""
    live = [candidate for candidate in cands if candidate.demoted_by is None]
    if len(live) < 2:
        return cands
    top_orders = tuple(
        _decode(battle, action, pos) for pos, action in enumerate(live[0].actions)
    )
    top_kos, _, _ = _candidate_guaranteed_progress(battle, live[0])
    if top_kos:
        return cands
    affected = set()
    for pos, (mon, order) in enumerate(zip(battle.active_pokemon, top_orders)):
        move = getattr(order, "order", None)
        if (
            mon is not None
            and not mon.fainted
            and float(mon.current_hp_fraction or 0.0) > 0.25
            and mon.boosts.get("atk", 0) <= -2
            and _norm(mon.ability) not in {"contrary", "defiant"}
            and isinstance(move, Move)
            and move.category == MoveCategory.PHYSICAL
        ):
            affected.add(pos)
    if not affected:
        return cands
    alternatives = []
    for candidate in live[1:]:
        orders = tuple(
            _decode(battle, action, pos) for pos, action in enumerate(candidate.actions)
        )
        if not all(
            isinstance(getattr(orders[pos], "order", None), Pokemon) for pos in affected
        ):
            continue
        partner_matches = sum(
            _order_signature(orders[pos]) == _order_signature(top_orders[pos])
            for pos in range(2)
            if pos not in affected
        )
        alternatives.append((partner_matches, candidate.prob, candidate))
    if not alternatives:
        return cands
    best = max(alternatives, key=lambda entry: entry[:2])[2]
    return _promote_candidate(cands, best, "severe_attack_drop_switch", report)


def guard_two_on_one_focus(battle, cands, report) -> list[Candidate]:
    """Prefer two attacks over a near-tied Protect gamble in a clean 2v1."""
    our_live = [
        mon for mon in battle.active_pokemon if mon is not None and not mon.fainted
    ]
    foes = [
        mon
        for mon in battle.opponent_active_pokemon
        if mon is not None and not mon.fainted
    ]
    live = [candidate for candidate in cands if candidate.demoted_by is None]
    if len(our_live) != 2 or len(foes) != 1 or len(live) < 2:
        return cands

    def progress(candidate: Candidate) -> tuple[int, float, bool]:
        attacks = 0
        damage = 0.0
        for pos, action in enumerate(candidate.actions):
            order = _decode(battle, action, pos)
            move, targets = _move_and_targets(battle, order, pos)
            attacker = battle.active_pokemon[pos]
            if (
                move is None
                or attacker is None
                or move.category == MoveCategory.STATUS
                or foes[0] not in targets
            ):
                continue
            fraction = K.damage_fraction(battle, attacker, foes[0], move)
            if fraction is not None:
                attacks += 1
                damage += max(0.0, fraction[0]) * float(move.accuracy or 0.0)
        valid = _candidate_guaranteed_progress(battle, candidate)[2]
        return attacks, damage, valid

    top_orders = [
        _decode(battle, action, pos) for pos, action in enumerate(live[0].actions)
    ]
    if (
        sum(
            isinstance(getattr(order, "order", None), Move)
            and getattr(order, "order").id in PROTECT_MOVES
            for order in top_orders
        )
        != 1
    ):
        return cands
    top_attacks, top_damage, _ = progress(live[0])
    threshold = live[0].prob * float(os.environ.get("VGC_TWO_ON_ONE_MIN_RATIO", "0.5"))
    alternatives = []
    for candidate in live[1:]:
        attacks, damage, valid = progress(candidate)
        if (
            valid
            and attacks == 2
            and damage > top_damage
            and candidate.prob >= threshold
            and not _uses_unplanned_tera(candidate, live[0].actions)
        ):
            alternatives.append((damage, candidate.prob, candidate))
    if top_attacks >= 2 or not alternatives:
        return cands
    best = max(alternatives, key=lambda entry: entry[:2])[2]
    return _promote_candidate(cands, best, "two_on_one_focus", report)


def guard_ko_tiebreak(
    battle, cands, report, eps_frac: float | None = None
) -> list[Candidate]:
    """G8: among near-tied pairs, prefer the one that secures more guaranteed KOs.

    Removing a threat outright is the core currency of VGC. Tera discipline from
    Laplace (engine_search.py:1062): a tera action may only be played as the argmax,
    never promoted by a tiebreak, since the resource is once per game.
    """
    # How wide the "near-tied" band is. This is the one knob worth sweeping: at ~1
    # firing per battle, ko_tiebreak is the only guard frequent enough for a win rate
    # to resolve. Read at call time so a sweep needs no code edit.
    if eps_frac is None:
        eps_frac = float(os.environ.get("VGC_KO_EPS", "0.5"))

    # Rerank only among candidates no earlier guard vetoed. Two reasons: a tiebreak
    # must never resurrect a demotion (the stack runs vetoes first precisely so their
    # verdict stands), and after a demotion `cands` is keep+push, so a positional
    # prefix would mix surviving candidates with pushed-back ones.
    live = [c for c in cands if c.demoted_by is None]
    dead_tail = [c for c in cands if c.demoted_by is not None]
    if len(live) < 2:
        return cands
    top = live[0].prob
    tied_n = sum(1 for c in live if c.prob >= top * eps_frac)
    if tied_n < 2:
        return cands

    argmax_actions = live[0].actions

    def is_tera(actions):
        return any(86 < a <= 106 for a in actions)

    def score(c: Candidate):
        kos = dmg = 0.0
        for pos, action in enumerate(c.actions):
            order = _decode(battle, action, pos)
            move, hit = _move_and_targets(battle, order, pos)
            me = battle.active_pokemon[pos]
            if move is None or me is None:
                continue
            for foe in hit:
                if K.guaranteed_ko(battle, me, foe, move):
                    kos += 1
                frac = K.damage_fraction(battle, me, foe, move)
                if frac:
                    dmg += (frac[0] + frac[1]) / 2 * (move.accuracy or 1.0)
        # never let a tiebreak promote an unplanned tera
        tera_penalty = 1 if (is_tera(c.actions) and c.actions != argmax_actions) else 0
        return (-tera_penalty, kos, dmg, c.prob)

    before = cands[0].actions
    head = sorted(live[:tied_n], key=score, reverse=True)
    out = head + live[tied_n:] + dead_tail
    if out[0].actions != before:
        report.note("ko_tiebreak")
    return out


def guard_redirection(battle, cands, report) -> list[Candidate]:
    """G2: demote a single-target Electric/Water move aimed past a revealed absorber.

    Lightning Rod / Storm Drain pull the move away and absorb it, turning your attack
    into a free +1 SpA for them. Fires ONLY when the ability is already revealed:
    guessing at a hidden ability is a probability, and probabilities do not earn a
    veto. Move-based redirection (Rage Powder, Follow Me) is deliberately NOT handled
    here -- it is a choice the opponent makes this turn, never a certainty.
    """
    dead = set()
    for i, c in enumerate(cands):
        for pos, action in enumerate(c.actions):
            order = _decode(battle, action, pos)
            move = getattr(order, "order", None)
            if not isinstance(move, Move) or move.category == MoveCategory.STATUS:
                continue
            if move.target not in SINGLE_TARGET or move.base_power <= 0:
                continue
            if (getattr(order, "move_target", 0) or 0) <= 0:
                continue  # not aimed at a specific foe
            foes = battle.opponent_active_pokemon
            aimed = resolved_foe_targets(battle, order, move)
            for other in foes:
                if other is None or other.fainted or other in aimed:
                    continue
                pulled = REDIRECT_ABILITIES.get(_norm(other.ability))
                if pulled is not None and move.type == pulled:
                    dead.add(i)
    return _demote(cands, dead, "redirection", report)


def guard_ally_damage(battle, cands, report) -> list[Candidate]:
    """G5: demote pairs that hurt our own side more than the opponent's.

    This is the guard that forces the joint-pair design. Scored per slot, it would
    veto the Earthquake in a legitimate Earthquake-under-ally-Protect play; scored
    per pair, it sees the Protect and stands down.
    """
    dead = set()
    for i, c in enumerate(cands):
        # Which of our slots are shielded by their own action this turn?
        protected = set()
        for pos, action in enumerate(c.actions):
            order = _decode(battle, action, pos)
            mv = getattr(order, "order", None)
            if isinstance(mv, Move) and mv.id in PROTECT_MOVES:
                protected.add(pos)

        foe_dmg = ally_dmg = 0.0
        for pos, action in enumerate(c.actions):
            order = _decode(battle, action, pos)
            move, hit = _move_and_targets(battle, order, pos)
            me = battle.active_pokemon[pos]
            if move is None or me is None:
                continue
            for foe in hit:
                frac = K.damage_fraction(battle, me, foe, move)
                if frac:
                    foe_dmg += min(1.0, frac[0])
            ally = _ally_hit(battle, order, pos)
            # Stand down when the ally is shielded this turn or already protected.
            if ally is None or (1 - pos) in protected or K.is_protected(ally):
                continue
            frac = K.damage_fraction(battle, me, ally, move)
            if frac:
                ally_dmg += min(1.0, frac[0])
        if ally_dmg > 0 and ally_dmg > foe_dmg:
            dead.add(i)
    return _demote(cands, dead, "ally_damage", report)


def guard_status_immunity(battle, cands, report) -> list[Candidate]:
    """G4: demote status moves the target cannot possibly be affected by.

    calculate_damage returns 0 for every status move, so the damage guards cannot see
    these at all -- hence a small explicit rule list. Four rules: type immunity to the
    specific status, powder immunity (Grass / Overcoat / Safety Goggles), Prankster
    into a Dark type, and a major status onto an already-statused target.
    """
    dead = set()
    for i, c in enumerate(cands):
        for pos, action in enumerate(c.actions):
            order = _decode(battle, action, pos)
            move = getattr(order, "order", None)
            if not isinstance(move, Move) or move.category != MoveCategory.STATUS:
                continue
            if (getattr(order, "move_target", 0) or 0) <= 0:
                continue  # only foe-targeted status moves
            targets = resolved_foe_targets(battle, order, move)
            if not targets:
                continue
            me = battle.active_pokemon[pos]
            all_immune = True
            for foe in targets:
                types = {pokemon_type for pokemon_type in foe.types if pokemon_type}
                immune = bool(STATUS_TYPE_IMMUNITY.get(move.id, set()) & types)
                if move.id in POWDER_MOVES and (
                    PokemonType.GRASS in types
                    or _norm(foe.ability) == "overcoat"
                    or _norm(foe.item) == "safetygoggles"
                ):
                    immune = True
                # Prankster-boosted status moves simply fail against Dark types.
                if (
                    me is not None
                    and _norm(me.ability) == "prankster"
                    and PokemonType.DARK in types
                ):
                    immune = True
                # A second major status never lands on an already-statused target.
                if (
                    move.status in _MAJOR_STATUS
                    and foe.status is not None
                    and foe.status != Status.FNT
                ):
                    immune = True
                all_immune &= immune
            if all_immune:
                dead.add(i)
    return _demote(cands, dead, "status_immunity", report)


def guard_setup_into_ko(battle, cands, report) -> list[Candidate]:
    """G7: demote setup when a faster foe has a move that is a guaranteed KO on us.

    All three conditions must be certain: the move is self-setup, the foe moves first
    for sure, and its KO is guaranteed at the MINIMUM roll. Anything less and this
    would suppress the setup turns that actually win games.
    """
    dead = set()
    foes = [
        f for f in battle.opponent_active_pokemon if f is not None and not f.fainted
    ]
    if not foes:
        return cands
    for i, c in enumerate(cands):
        for pos, action in enumerate(c.actions):
            me = battle.active_pokemon[pos]
            if me is None or me.fainted:
                continue
            order = _decode(battle, action, pos)
            move = getattr(order, "order", None)
            if not isinstance(move, Move):
                continue
            is_setup = (
                move.category == MoveCategory.STATUS
                and move.target == Target.SELF
                and bool(move.boosts)
                and any(v > 0 for v in (move.boosts or {}).values())
            )
            if not is_setup:
                continue
            for foe in foes:
                if not _foe_moves_first(battle, me, foe):
                    continue
                if any(
                    K.guaranteed_ko(battle, foe, me, fm)
                    for fm in (foe.moves or {}).values()
                ):
                    dead.add(i)
                    break
    return _demote(cands, dead, "setup_into_ko", report)


def guard_redundant_side_condition(battle, cands, report) -> list[Candidate]:
    """G9: demote a move setting a side condition our side already has.

    Tailwind under Tailwind, Reflect under Reflect and so on simply fail -- a wasted
    turn in a format that lasts about a dozen of them. Deliberately limited to side
    conditions: Trick Room looks similar but is a TOGGLE, and re-using it to cancel an
    opponent's is a real play, so demoting it would suppress correct decisions.
    """
    dead = set()
    for i, c in enumerate(cands):
        for pos, action in enumerate(c.actions):
            order = _decode(battle, action, pos)
            move = getattr(order, "order", None)
            if not isinstance(move, Move) or move.category != MoveCategory.STATUS:
                continue
            cond = getattr(move, "side_condition", None)
            if cond is not None and cond in battle.side_conditions:
                dead.add(i)
    return _demote(cands, dead, "redundant_side_condition", report)


def guard_free_catastrophic_setup(battle, cands, report) -> list[Candidate]:
    """G10: do not Double Protect into likely Shell Smash-class setup.

    The veto only applies when another surviving candidate can immediately damage the
    setup user. This retains Double Protect as a legal fallback when neither active
    Pokemon can actually contest the setup.
    """
    threshold = float(os.environ.get("VGC_FREE_SETUP_PROBABILITY", "0.70"))
    threats = [
        foe
        for foe in battle.opponent_active_pokemon
        if foe is not None
        and not foe.fainted
        and _hidden_move_probability(foe, CATASTROPHIC_SETUP_MOVES) >= threshold
    ]
    if not threats:
        return cands
    live = [candidate for candidate in cands if candidate.demoted_by is None]
    contest_exists = any(
        _candidate_damage_to(battle, candidate, threat) >= 0.12
        for candidate in live
        for threat in threats
        if not candidate_gives_catastrophic_free_setup(battle, candidate)
    )
    if not contest_exists:
        return cands
    dead = {
        i
        for i, candidate in enumerate(cands)
        if candidate_gives_catastrophic_free_setup(battle, candidate)
    }
    return _demote(cands, dead, "free_catastrophic_setup", report)


def guard_encore_exposure(battle, cands, report) -> list[Candidate]:
    """G11: avoid handing a revealed faster Encore user a Protect lock."""
    dead = {
        i
        for i, candidate in enumerate(cands)
        if candidate_exposes_protect_to_encore(battle, candidate)
    }
    return _demote(cands, dead, "encore_exposure", report)


def guard_dominated_weather_ball(battle, cands, report) -> list[Candidate]:
    """G12: avoid 50 BP no-weather Weather Ball when Heat Wave dominates it."""
    dead = {
        i
        for i, candidate in enumerate(cands)
        if candidate_uses_dominated_weather_ball(battle, candidate)
    }
    return _demote(cands, dead, "dominated_weather_ball", report)


def guard_single_target_weather_ball(battle, cands, report) -> list[Candidate]:
    """Prefer the stronger active-weather Weather Ball when only one foe remains."""
    dead = {
        i
        for i, candidate in enumerate(cands)
        if candidate_uses_dominated_single_target_heat_wave(battle, candidate)
    }
    return _demote(cands, dead, "single_target_weather_ball", report)


def guard_protect_spam(battle, cands, report) -> list[Candidate]:
    """G13: demote Protect when this Pokemon already protected last turn.

    Consecutive Protect succeeds only 1/3 of the time, and falls off geometrically
    after that. poke-env tracks the streak in `protect_counter`, which resets as soon
    as the Pokemon does anything else. Only the repeat is demoted -- the first Protect
    is one of the strongest moves in doubles.
    """
    dead = set()
    for i, c in enumerate(cands):
        for pos, action in enumerate(c.actions):
            me = battle.active_pokemon[pos]
            if me is None or (me.protect_counter or 0) < 1:
                continue
            order = _decode(battle, action, pos)
            move = getattr(order, "order", None)
            if isinstance(move, Move) and move.id in PROTECT_MOVES:
                dead.add(i)
    return _demote(cands, dead, "protect_spam", report)


GUARDS = {
    "zero_damage": guard_zero_damage,
    "first_turn": guard_first_turn,
    "priority_block": guard_priority_block,
    "redirection": guard_redirection,
    "status_immunity": guard_status_immunity,
    "ally_damage": guard_ally_damage,
    "setup_into_ko": guard_setup_into_ko,
    "redundant_side_condition": guard_redundant_side_condition,
    "free_catastrophic_setup": guard_free_catastrophic_setup,
    "encore_exposure": guard_encore_exposure,
    "dominated_weather_ball": guard_dominated_weather_ball,
    "single_target_weather_ball": guard_single_target_weather_ball,
    "protect_spam": guard_protect_spam,
    "guaranteed_ko": guard_guaranteed_ko,
    "reserve_weather_mega": guard_reserve_weather_mega,
    "severe_attack_drop_switch": guard_severe_attack_drop_switch,
    "yawn_switch": guard_yawn_switch,
    "two_on_one_focus": guard_two_on_one_focus,
    "endgame_progress": guard_endgame_progress,
    "ko_tiebreak": guard_ko_tiebreak,
}

# Production profile: factual vetoes plus the narrow regression-tested planning rules
# that are allowed to cross a wide policy gap. The remaining broad experimental rules
# previously underperformed against a historical learned-policy population.
HARD_GUARDS = frozenset(
    {
        "zero_damage",
        "first_turn",
        "priority_block",
        "redirection",
        "status_immunity",
        "redundant_side_condition",
        "free_catastrophic_setup",
        "encore_exposure",
        "dominated_weather_ball",
        "single_target_weather_ball",
        # Search may rank a one-in-three consecutive Protect above every action
        # the policy guard stack originally supplied.  Keep this factual doubles
        # safety rule after exact scoring as well, provided a non-repeat survived.
        "protect_spam",
        "guaranteed_ko",
        "reserve_weather_mega",
        "severe_attack_drop_switch",
        "yawn_switch",
        "two_on_one_focus",
        "endgame_progress",
    }
)

# Hard vetoes first, soft reranks last: ko_tiebreak reorders whatever survives.
#
# Measured on our fixed team vs the 481-team held-out pool (300 battles), some of
# these are structurally inert and that is a property of the MATCHUP, not a bug:
#   * setup_into_ko  -- our team has no self-boosting moves at all
#   * redirection    -- we carry no Electric attacks, and Storm Drain appears 0 times
#                       in the pool (Lightning Rod appears, but only pulls Electric)
# They are kept because they are correct and nearly free, and they start mattering the
# moment the team changes. Guard value is team-specific -- check firing counts against
# YOUR team before concluding a rule is worthless.
GUARD_ORDER = (
    "zero_damage",
    "first_turn",
    "priority_block",
    "redirection",
    "status_immunity",
    "ally_damage",
    "setup_into_ko",
    "redundant_side_condition",
    "free_catastrophic_setup",
    "encore_exposure",
    "dominated_weather_ball",
    "single_target_weather_ball",
    "protect_spam",
    "guaranteed_ko",
    "reserve_weather_mega",
    "severe_attack_drop_switch",
    "yawn_switch",
    "two_on_one_focus",
    "endgame_progress",
    "ko_tiebreak",
)


def apply_guards(
    battle: DoubleBattle, cands: list[Candidate], enabled: dict[str, bool] | None = None
) -> tuple[list[Candidate], GuardReport]:
    """Run the stack in order: hard vetoes first, soft reranks last."""
    report = GuardReport()
    for name in GUARD_ORDER:
        if enabled is not None and not enabled.get(name, True):
            continue
        # One guard raising must not cost the turn, and must not silently take the
        # rest of the stack down with it.
        try:
            cands = GUARDS[name](battle, cands, report)
        except Exception:
            report.note(f"{name}_error")
    return cands, report
