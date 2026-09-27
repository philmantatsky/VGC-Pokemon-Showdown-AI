"""Tactical teacher: the move and damage facts the guards patch at play time,
as training targets for the network itself (2026-09-26, the user: "make your own
decision on how to train it further BASED ON MISTAKES").

Across the T6-era ladder games the network's own top pick was corrected by a
guard in 25-47% of move decisions -- a damage-dominated attack, a target that
resists the move, Heat Wave over a full-HP Eruption, Fake Out into Psychic
Terrain, Leaf Storm into a 4x resist -- in wins as often as in losses. The
guards fix the pick they see, but only among the network's top eight pairs and
only when a later stage does not undo them, so the lesson never reaches the
network. This module states two kinds of fact for one active Pokemon:

* ``useless``: actions that certainly fail or do nothing -- a damaging move
  every target is immune to, a priority move into Psychic Terrain or Armor Tail,
  Fake Out after the first turn, a status move every target is immune to,
  weather that is already up, Helping Hand with no partner;
* ``attack_values``: for plain damaging moves in the normal band (no Mega this
  turn: the calculator would use the pre-Mega form), the accuracy-weighted
  damage capped at the HP left after the partner's attack plus a knockout
  bonus -- the valuation dominated_attack / dominated_spread use, with Solar
  Beam counted as a plain attack in sun and spread moves into a shown Wide
  Guard counted at a fraction.

``target_distribution`` turns the policy's own distribution into a training
target: useless actions lose their mass (spread over the rest in proportion),
and the mass the policy puts on plain attacks is re-spread over them by value
(softmax with temperature ``tau``). Attack vs Protect vs switch vs support and
Mega timing keep the policy's own proportions: the teacher only says WHICH
attack, never WHETHER to attack -- with one exception (2026-09-27, the user
chose it: "2"):

* ``doomed_facts``: the chance our Pokemon is knocked out BEFORE it moves this
  turn -- a foe that certainly acts first (speed bounds with Trick Room /
  Tailwind / paralysis / weather abilities, or a priority move our side does not
  block) has a likely damaging move that knocks it out from its current HP.
  A move it would lose that way is wasted; when the Pokemon has a legal Protect
  it did not use last turn, ``doomed_target`` moves that share of the mass its
  wasted moves carry onto Protect (playbook rule 6: "when Trick Room is down and
  a slow Pokemon will be knocked out before it moves, Protect ... instead of
  clicking an attack"). In 165 T6-era ladder games one of our Pokemon attacked
  and was knocked out first 217 times, 191 of them in losses. Switching out is
  not taught: whether the Pokemon coming in survives is far less certain.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from poke_env.battle import (
    DoubleBattle,
    Field,
    Move,
    MoveCategory,
    PokemonType,
    Status,
    Target,
    Weather,
)

from vgc_bench.src import guards as G
from vgc_bench.src import vgc_knowledge as K
from vgc_bench.src.utils import act_len
from vgc_bench.src.wide_guard import spread_attack, wide_guard_users

NORMAL_BAND = range(7, 27)  # moves without a gimmick: 4 moves x 5 targets
MOVE_BANDS = range(7, 47)  # normal + Mega bands
WEATHER_MOVES = {
    "raindance": {Weather.RAINDANCE},
    "sunnyday": {Weather.SUNNYDAY},
    "sandstorm": {Weather.SANDSTORM},
    "snowscape": {Weather.SNOWSCAPE, Weather.HAIL},
    "hail": {Weather.SNOWSCAPE, Weather.HAIL},
}
SOLAR_MOVES = {"solarbeam", "solarblade"}
WIDE_GUARD_SHARE = 0.15  # a shown Wide Guard is used often, not always


def _in_sun(battle: DoubleBattle, attacker) -> bool:
    weather = set(getattr(battle, "weather", {}) or {})
    if G._norm(getattr(attacker, "item", None)) == "utilityumbrella":
        return False
    return bool(weather & {Weather.SUNNYDAY, Weather.DESOLATELAND})


def plain_attack(battle: DoubleBattle, attacker, move: Move) -> bool:
    """A damaging move whose job is its damage this turn (see guards._plain_attack);
    Solar Beam counts when sun makes it a one-turn move."""
    if move.id in SOLAR_MOVES:
        return (
            _in_sun(battle, attacker)
            and move.category != MoveCategory.STATUS
            and float(move.base_power or 0) > 0
        )
    return G._plain_attack(move)


def useless(battle: DoubleBattle, pos: int, action: int) -> str | None:
    """Why ``action`` for our Pokemon in ``pos`` certainly fails or does nothing."""
    if action < MOVE_BANDS.start or action >= MOVE_BANDS.stop:
        return None
    me = battle.active_pokemon[pos]
    if me is None or me.fainted:
        return None
    order = G._decode(battle, action, pos)
    move = getattr(order, "order", None)
    if not isinstance(move, Move):
        return None
    if move.id in G.FIRST_TURN_ONLY and not me.first_turn:
        return "first_turn_only"
    hit = G.resolved_foe_targets(battle, order, move)
    if G._effective_priority(me, move) > 0 and hit:
        foes = [
            f for f in battle.opponent_active_pokemon if f is not None and not f.fainted
        ]
        if any(G._norm(f.ability) in G.PRIORITY_BLOCK_ABILITIES for f in foes):
            return "priority_blocked"
        if Field.PSYCHIC_TERRAIN in getattr(battle, "fields", {}) and all(
            G._is_grounded(battle, f) for f in hit
        ):
            return "psychic_terrain"
    if move.category != MoveCategory.STATUS and float(move.base_power or 0) > 0:
        if hit and all(K.deals_no_damage(battle, me, foe, move) for foe in hit):
            return "zero_damage"
        return None
    if move.id in WEATHER_MOVES:
        if set(getattr(battle, "weather", {}) or {}) & WEATHER_MOVES[move.id]:
            return "weather_already_up"
        return None
    if move.id == "helpinghand":
        partner = battle.active_pokemon[1 - pos]
        if partner is None or partner.fainted:
            return "no_partner"
        return None
    if (getattr(order, "move_target", 0) or 0) > 0 and hit:
        all_immune = True
        for foe in hit:
            types = {t for t in foe.types if t}
            immune = bool(G.STATUS_TYPE_IMMUNITY.get(move.id, set()) & types)
            if move.id in G.POWDER_MOVES and (
                PokemonType.GRASS in types
                or G._norm(foe.ability) == "overcoat"
                or G._norm(foe.item) == "safetygoggles"
            ):
                immune = True
            if G._norm(me.ability) == "prankster" and PokemonType.DARK in types:
                immune = True
            if (
                move.status in G._MAJOR_STATUS
                and foe.status is not None
                and foe.status != Status.FNT
            ):
                immune = True
            if G._norm(foe.ability) == "goodasgold":
                immune = True
            all_immune &= immune
        if all_immune:
            return "status_immune"
    return None


def attack_values(
    battle: DoubleBattle, pos: int, mask_row: np.ndarray, partner_action: int | None
) -> dict[int, float]:
    """Value of every legal plain attack of the normal band for ``pos``.

    ``partner_action`` (slot 1's chosen action, for slot 2) lets the partner's
    expected damage cap what is left to take, like the attack guards do.
    Actions the calculator cannot evaluate are left out (no lesson there).
    """
    me = battle.active_pokemon[pos]
    if me is None or me.fainted:
        return {}
    already: dict[int, float] = {}
    if pos == 1 and partner_action is not None:
        actions = (int(partner_action), 0)
        try:
            already = G._partner_damage(battle, actions, pos)
        except Exception:
            already = {}
    wide = bool(wide_guard_users(battle))
    values: dict[int, float] = {}
    for action in NORMAL_BAND:
        if not mask_row[action]:
            continue
        order = G._decode(battle, action, pos)
        move = getattr(order, "order", None)
        if not isinstance(move, Move) or not plain_attack(battle, me, move):
            continue
        if useless(battle, pos, action) is not None:
            continue
        try:
            value = G._attack_value(battle, me, order, pos, already)
        except Exception:
            value = None
        if value is None:
            continue
        if wide and spread_attack(order):
            value *= WIDE_GUARD_SHARE
        values[action] = float(value)
    return values


def position_facts(
    battle: DoubleBattle, pos: int, mask_row: np.ndarray, partner_action: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """(useless flags, attack values with NaN for non-attacks) over act_len."""
    flags = np.zeros(act_len, dtype=bool)
    values = np.full(act_len, np.nan, dtype=np.float32)
    for action in MOVE_BANDS:
        if mask_row[action] and useless(battle, pos, action) is not None:
            flags[action] = True
    for action, value in attack_values(battle, pos, mask_row, partner_action).items():
        values[action] = value
    return flags, values


USE_SHARE = 0.8  # a foe attacks most turns (it may also Protect, set up or switch)
TARGET_SHARE = 0.6  # a foe that can knock our Pokemon out usually aims at it
DOOMED_MIN = 0.25  # below this the doomed lesson stays silent
SPREAD_TARGETS = (Target.ALL_ADJACENT_FOES, Target.ALL_ADJACENT)
WEATHER_SPEED = {
    "chlorophyll": (Weather.SUNNYDAY, Weather.DESOLATELAND),
    "swiftswim": (Weather.RAINDANCE, Weather.PRIMORDIALSEA),
    "sandrush": (Weather.SANDSTORM,),
    "slushrush": (Weather.SNOWSCAPE, Weather.HAIL),
}


def _speed_bounds(battle: DoubleBattle, mon, ours: bool) -> tuple[float, float] | None:
    """(slowest, fastest) effective Speed. Ours is known; a foe's spans every
    Champions build of its species (0-32 points, -/+ Speed nature)."""
    if ours:
        base = (mon.stats or {}).get("spe")
        if not base:
            return None
        lo = hi = float(base)
    else:
        stats = getattr(mon, "base_stats", None) or {}
        if not stats.get("spe"):
            return None
        lo = (stats["spe"] + 20) * 0.9
        hi = (stats["spe"] + 32 + 20) * 1.1
    mult = G._BOOST_MULT.get(mon.boosts.get("spe", 0), 1.0)
    conds = battle.side_conditions if ours else battle.opponent_side_conditions
    if G.SideCondition.TAILWIND in conds:
        mult *= 2
    if mon.status == Status.PAR:
        mult *= 0.5
    weather = set(getattr(battle, "weather", {}) or {})
    if weather & set(WEATHER_SPEED.get(G._norm(mon.ability), ())):
        mult *= 2
    return lo * mult, hi * mult


def _acts_first(battle: DoubleBattle, me, foe) -> bool:
    """True only when the foe CERTAINLY outspeeds us this turn (Trick Room read)."""
    mine = _speed_bounds(battle, me, ours=True)
    theirs = _speed_bounds(battle, foe, ours=False)
    if mine is None or theirs is None:
        return False
    if Field.TRICK_ROOM in getattr(battle, "fields", {}):
        return theirs[1] < mine[0]
    return theirs[0] > mine[1]


def _priority(battle: DoubleBattle, mon, move: Move) -> int:
    priority = G._effective_priority(mon, move)
    if move.id == "grassyglide" and Field.GRASSY_TERRAIN in getattr(
        battle, "fields", {}
    ):
        priority += 1
    return priority


def _ko_chance(fraction: tuple[float, float], hp: float) -> float:
    """Share of the damage roll range that knocks out ``hp`` (max-HP fractions)."""
    lo, hi = fraction
    if lo >= hp:
        return 1.0
    if hi < hp or hi <= lo:
        return 0.0
    return (hi - hp) / (hi - lo)


def doomed_probability(battle: DoubleBattle, pos: int) -> float:
    """Chance our Pokemon in ``pos`` is knocked out before it moves this turn.

    Each foe contributes its best likely damaging move (revealed, else the
    species' most-used set) that acts first: a priority move our side does not
    block, or any move when the foe certainly outspeeds us. The foe attacks with
    USE_SHARE; a spread move then hits us for sure, a single-target move is aimed
    at us with TARGET_SHARE; accuracy counts, critical hits do not. Unknowns
    contribute nothing.
    """
    from vgc_bench.src.playbook_opening import _likely_moves

    me = battle.active_pokemon[pos]
    if me is None or me.fainted:
        return 0.0
    hp = float(me.current_hp_fraction or 0.0)
    if hp <= 0.0:
        return 0.0
    survive = 1.0
    for foe in battle.opponent_active_pokemon:
        if foe is None or foe.fainted:
            continue
        K.ensure_stats(foe)
        best = 0.0
        for move in _likely_moves(battle, foe):
            if move.category == MoveCategory.STATUS or float(move.base_power or 0) <= 0:
                continue
            priority = _priority(battle, foe, move)
            if priority > 0:
                if G._priority_is_blocked(battle, foe, me, priority):
                    continue
            elif priority < 0 or not _acts_first(battle, me, foe):
                continue
            fraction = K.damage_fraction(battle, foe, me, move)
            if fraction is None:
                continue
            accuracy = move.accuracy if isinstance(move.accuracy, float) else 1.0
            aimed = USE_SHARE * (1.0 if move.target in SPREAD_TARGETS else TARGET_SHARE)
            best = max(best, _ko_chance(fraction, hp) * min(accuracy, 1.0) * aimed)
        survive *= 1.0 - best
    return 1.0 - survive


def doomed_facts(
    battle: DoubleBattle, pos: int, mask_row: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """(doomed chance, Protect actions, wasted actions) over act_len for ``pos``.

    Silent (0.0) unless the Pokemon has a legal Protect it did not use last turn
    and the chance reaches DOOMED_MIN. Wasted = its legal moves other than
    Protect that do not act before the foes (priority <= 0).
    """
    protect = np.zeros(act_len, dtype=bool)
    wasted = np.zeros(act_len, dtype=bool)
    me = battle.active_pokemon[pos]
    if me is None or me.fainted or int(getattr(me, "protect_counter", 0) or 0) > 0:
        return 0.0, protect, wasted
    for action in MOVE_BANDS:
        if not mask_row[action]:
            continue
        order = G._decode(battle, action, pos)
        move = getattr(order, "order", None)
        if not isinstance(move, Move):
            continue
        if move.id in G.PROTECT_MOVES:
            protect[action] = True
        elif _priority(battle, me, move) <= 0:
            wasted[action] = True
    if not protect.any() or not wasted.any():
        return 0.0, protect, wasted
    chance = doomed_probability(battle, pos)
    return (chance if chance >= DOOMED_MIN else 0.0), protect, wasted


def doomed_target(
    q: np.ndarray, doomed: float, protect: np.ndarray, wasted: np.ndarray
) -> np.ndarray | None:
    """Move the ``doomed`` share of the wasted moves' mass onto Protect (in
    Protect's own proportions, evenly if it has none). None: nothing to move."""
    if doomed <= 0.0 or not protect.any():
        return None
    q = np.asarray(q, dtype=np.float64).copy()
    moved = doomed * q[wasted].sum()
    if moved <= 0.0:
        return None
    q[wasted] *= 1.0 - doomed
    base = q[protect]
    share = base / base.sum() if base.sum() > 0 else np.full(len(base), 1 / len(base))
    q[protect] = base + moved * share
    return q / q.sum()


def target_distribution(
    probs: np.ndarray, flags: np.ndarray, values: np.ndarray, tau: float
) -> np.ndarray | None:
    """The training target for one position from the policy's own distribution.

    Returns None when the teacher has nothing to say (no useless action with
    mass and fewer than two valued attacks).
    """
    probs = np.asarray(probs, dtype=np.float64)
    q = probs.copy()
    changed = False
    dead = flags & (q > 0)
    if dead.any():
        alive = (~flags) & (q > 0)
        if alive.any():
            q[dead] = 0.0
            changed = True
    total = q.sum()
    if total <= 0:
        return None
    q /= total
    valued = ~np.isnan(values) & (q >= 0) & (probs > 0)
    idx = np.flatnonzero(valued)
    if len(idx) >= 2:
        mass = q[idx].sum()
        if mass > 0:
            logits = values[idx].astype(np.float64) / max(tau, 1e-6)
            logits -= logits.max()
            w = np.exp(logits)
            q[idx] = mass * w / w.sum()
            changed = True
    return q if changed else None
