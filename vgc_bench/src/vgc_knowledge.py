"""Explicit battle knowledge for VGC doubles: damage, KOs, immunities.

The policy's observation encodes a move's type and a defender's types as separate
one-hot vectors and never computes their interaction, so the network has to infer the
whole 18x18 type chart from self-play. Immunities are rare and therefore learned last
-- which is how the bot came to fire a Ghost move at a Normal type on ladder.

This module supplies the missing knowledge directly, using poke-env's gen-9 damage
calculator (which accepts DoubleBattle). It is the VGC analogue of the Laplace bot's
knowledge.py, and like that module it is pure functions over battle objects: no
policy imports, no mutation of anything the observation reads.

Three facts this module is built on, all verified against replayed battles:

1. calculate_damage takes IDENTIFIER STRINGS -- the dict keys of battle.team /
   battle.opponent_team, formatted "p1: Sinistcha". Resolve by object identity, never
   by constructing the string: AbstractBattle.get_pokemon fabricates a new Pokemon for
   an unknown key rather than failing.

2. Opponent stats arrive at ZERO EVs. In the Champions format the stat formula is
   `base + evs + 20` (`+75` for HP), which at level 50 with 31 IVs is algebraically
   identical to the mainline formula at 0 EVs -- so poke-env's numbers are right, just
   un-invested. Champions EVs then add 1:1 to the stat (not EV/4), capped at 32 per
   stat with a 66 total budget. That makes imputation a simple addition.

3. calculate_damage returns (0, 0) for a PROTECTED defender. That is not an immunity,
   and treating it as one would veto perfectly good attacks. Always check effects.
"""

from __future__ import annotations

from copy import copy
from typing import Iterable

from poke_env.battle import DoubleBattle, Move, MoveCategory, Pokemon
from poke_env.calc import calculate_damage
from poke_env.data import GenData

_TYPE_CHART: dict[str, dict[str, float]] | None = None


def type_multiplier(move: Move, defender: Pokemon) -> float | None:
    """Type effectiveness of `move` against `defender`, or None if undeterminable.

    Needs no stats, so it works when calculate_damage cannot run at all. That case is
    not rare: Reg M-B's Open Team Sheets are OPT-IN, and when an opponent denies them
    their Pokemon arrive with no stats, calculate_damage raises, and every calc-driven
    piece of knowledge silently evaluates to "not immune".
    """
    global _TYPE_CHART
    if not defender.types:
        return None
    if _TYPE_CHART is None:
        _TYPE_CHART = GenData.from_gen(9).type_chart
    try:
        return move.type.damage_multiplier(*defender.types, type_chart=_TYPE_CHART)
    except Exception:
        return None


# Champions EV scale, confirmed against every bundled team file (each sums to 66).
CHAMPIONS_EV_CAP = 32
CHAMPIONS_EV_BUDGET = 66
_STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")
# Smogon spread strings are "Nature:hp/atk/def/spa/spd/spe" in Champions EV units.
_SPREAD_ORDER = ("hp", "atk", "def", "spa", "spd", "spe")


def identifier(battle: DoubleBattle, mon: Pokemon) -> str | None:
    """Dict key for `mon`, resolved by identity. None if it isn't in either team."""
    for table in (battle.team, battle.opponent_team):
        for key, value in table.items():
            if value is mon:
                return key
    return None


def is_protected(mon: Pokemon) -> bool:
    """True if a protect-family effect is currently on this Pokemon."""
    for effect in mon.effects:
        name = effect.name.upper()
        if "PROTECT" in name or name in {
            "ENDURE",
            "SPIKY_SHIELD",
            "BANEFUL_BUNKER",
            "SILK_TRAP",
            "BURNING_BULWARK",
            "WIDE_GUARD",
            "QUICK_GUARD",
            "MAX_GUARD",
            "OBSTRUCT",
            "KINGS_SHIELD",
            "CRAFTY_SHIELD",
            "MAT_BLOCK",
        }:
            return True
    return False


def parse_spread(spread: str | None) -> dict[str, int] | None:
    """'Adamant:32/32/0/0/2/0' -> {'hp':32,...}. None if malformed or off-budget."""
    if not spread or ":" not in spread:
        return None
    _nature, _, numbers = spread.partition(":")
    parts = numbers.split("/")
    if len(parts) != 6:
        return None
    try:
        values = [int(p) for p in parts]
    except ValueError:
        return None
    if any(v < 0 or v > CHAMPIONS_EV_CAP for v in values):
        return None
    if sum(values) > CHAMPIONS_EV_BUDGET:
        return None
    return dict(zip(_SPREAD_ORDER, values))


_NATURE_MULTS = (0.9, 1.0, 1.1)


def _nature_mult(mon: Pokemon, key: str) -> float | None:
    """Recover the nature multiplier on one stat, or None if it looks invested.

    Champions computes `(base + evs + 20) * nature`. With evs=0 that leaves exactly
    three possible values per stat, so an exact match identifies both "un-invested"
    and which way the nature leans -- without needing the nature itself, which is
    absent from the battle object.
    """
    stats = mon.stats
    if not stats or stats.get(key) is None:
        return None
    uninvested = mon.base_stats[key] + 20
    for mult in _NATURE_MULTS:
        if stats[key] == int(uninvested * mult):
            return mult
    return None


def is_uninvested(mon: Pokemon) -> bool:
    """True if this Pokemon's stats are the 0-EV Champions baseline.

    Makes imputation idempotent without a marker attribute -- Pokemon uses __slots__,
    so arbitrary attributes cannot be attached to it.
    """
    stats = mon.stats
    if not stats or stats.get("hp") is None:
        return False
    if stats["hp"] != mon.base_stats["hp"] + 75:
        return False
    return all(
        _nature_mult(mon, key) is not None for key in mon.base_stats if key != "hp"
    )


def impute_stats(mon: Pokemon, spread: dict[str, int] | None = None) -> bool:
    """Add Champions EV investment to a Pokemon whose stats came through at 0 EVs.

    Safe to call on opponents: embed_pokemon hardcodes `stats = [-1]*6` for the
    opposing side, so this never perturbs the observation the policy sees.

    Returns True if stats were modified; a no-op on already-invested stats.
    """
    stats = mon.stats
    if not stats or any(v is None for v in stats.values()):
        return False
    if not is_uninvested(mon):
        return False

    if spread is None:
        # No usage data: invest the standard 32/32 into bulk and the better attack,
        # which is the shape of most VGC spreads.
        offense = "atk" if mon.base_stats["atk"] >= mon.base_stats["spa"] else "spa"
        spread = {"hp": CHAMPIONS_EV_CAP, offense: CHAMPIONS_EV_CAP, "spe": 2}

    # Recompute from base rather than adding to the current value, so the nature
    # multiplier applies to the invested stat as Showdown does it.
    for key, base in mon.base_stats.items():
        ev = min(int(spread.get(key, 0)), CHAMPIONS_EV_CAP)
        if key == "hp":
            stats[key] = base + ev + 75
        else:
            mult = _nature_mult(mon, key) or 1.0
            stats[key] = int((base + ev + 20) * mult)
    mon.stats = stats
    return True


def ensure_stats(mon: Pokemon) -> bool:
    """Give `mon` usable stats, whatever state it arrived in. True if modified.

    impute_stats covers only the "sheets shown, EV field blank" case -- it requires
    existing stats to top up. When an opponent DENIES Open Team Sheets the stats are
    absent entirely, impute_stats no-ops, and calculate_damage raises for every single
    pairing. That is not a rare corner: it disabled the zero-damage guard, flattened
    guaranteed_ko to always-False (which is what ko_tiebreak ranks on), and made every
    decision throw exceptions in a loop.

    Synthesising from base stats is approximate -- the nature is unknown, so it assumes
    neutral -- but an approximate defender is enormously better than no defender, and
    every guard threshold is a bound rather than a point estimate for this reason.
    """
    if impute_stats(mon):
        return True
    stats = mon.stats
    if stats and all(v is not None for v in stats.values()):
        return False  # already usable
    if not mon.base_stats:
        return False
    offense = "atk" if mon.base_stats["atk"] >= mon.base_stats["spa"] else "spa"
    spread = {"hp": CHAMPIONS_EV_CAP, offense: CHAMPIONS_EV_CAP, "spe": 2}
    built: dict[str, int | None] = {}
    for key, base in mon.base_stats.items():
        ev = min(int(spread.get(key, 0)), CHAMPIONS_EV_CAP)
        if key == "hp":
            built[key] = base + ev + 75
        else:
            built[key] = int((base + ev + 20) * (_nature_mult(mon, key) or 1.0))
    mon.stats = built
    return True


def stats_were_synthesized(mon: Pokemon) -> bool:
    """True when ``mon``'s stats match ensure_stats' from-absent synthetic spread.

    Stateless detection: recompute the synthetic formula and compare. A real set
    that happens to equal the default guess is treated as synthesized, which
    only makes the promotion bound slightly conservative for that mon.
    """
    stats = mon.stats
    if not stats or any(v is None for v in stats.values()) or not mon.base_stats:
        return False
    offense = "atk" if mon.base_stats["atk"] >= mon.base_stats["spa"] else "spa"
    spread = {"hp": CHAMPIONS_EV_CAP, offense: CHAMPIONS_EV_CAP, "spe": 2}
    for key, base in mon.base_stats.items():
        ev = min(int(spread.get(key, 0)), CHAMPIONS_EV_CAP)
        if key == "hp":
            expected = base + ev + 75
        else:
            expected = int((base + ev + 20) * (_nature_mult(mon, key) or 1.0))
        if stats.get(key) != expected:
            return False
    return True


def robust_ko_scale(defender: Pokemon, move: Move) -> float:
    """Analytic worst-case-defender scale for a min-roll damage fraction.

    The synthetic spread invests nothing defensively (Def = base + 20, neutral),
    so min-roll damage against it overstates damage against a real bulky set. A
    "guaranteed" KO used to PROMOTE over the policy pick must instead survive
    the worst plausible spread: max defensive EVs plus a boosting nature.
    Damage scales with 1/Def, so the correction is synthetic_def / worst_def.
    """
    key = "def" if move.category == MoveCategory.PHYSICAL else "spd"
    base = (defender.base_stats or {}).get(key)
    if not base:
        return 1.0
    synthetic = base + 20
    worst = (base + CHAMPIONS_EV_CAP + 20) * 1.1
    return synthetic / worst if worst > 0 else 1.0


def damage_range(
    battle: DoubleBattle, attacker: Pokemon, defender: Pokemon, move: Move
) -> tuple[float, float] | None:
    """Raw (min, max) damage. None if the calc cannot evaluate this pairing.

    Total by contract: every failure path returns None so callers can fall back to the
    type chart. Resolving the identifiers can itself raise, so that is inside the try.
    """
    try:
        att_id = identifier(battle, attacker)
        def_id = identifier(battle, defender)
        if att_id is None or def_id is None:
            return None
        # Without this the calc raises on every pairing whenever the opponent denied
        # Open Team Sheets, turning damage knowledge off exactly when it is needed most.
        ensure_stats(attacker)
        ensure_stats(defender)
        effective_move = move
        if move.id == "lastrespects":
            # poke-env's calculator currently leaves Last Respects at its printed
            # 50 BP. In battle it gains 50 BP for each fainted ally. That omission
            # made the planner compare a 150 BP Last Respects with rain-boosted Aqua
            # Jet as if they were the same base power. Copy rather than mutating the
            # shared Move object stored on the Pokemon.
            sides = (battle.team, battle.opponent_team)
            side = next(
                (
                    members
                    for members in sides
                    if any(mon is attacker for mon in members.values())
                ),
                {},
            )
            fainted_allies = sum(
                mon is not attacker and mon.fainted for mon in side.values()
            )
            effective_move = copy(move)
            effective_move._base_power_override = min(5050, 50 * (1 + fainted_allies))
        return calculate_damage(att_id, def_id, effective_move, battle)
    except Exception:
        # Missing opponent stats raise AssertionError; unusual moves can raise
        # KeyError. A failed calc must never be read as "does nothing".
        return None


def damage_fraction(
    battle: DoubleBattle, attacker: Pokemon, defender: Pokemon, move: Move
) -> tuple[float, float] | None:
    """(min, max) damage as a fraction of the defender's max HP.

    Denominator is stats['hp'], never max_hp: opponents' max_hp is 100 on the percent
    scale, so max_hp would silently produce percentages of a percentage.
    """
    raw = damage_range(battle, attacker, defender, move)
    if raw is None:
        return None
    hp = (defender.stats or {}).get("hp")
    if not hp:
        return None
    return raw[0] / hp, raw[1] / hp


def deals_no_damage(
    battle: DoubleBattle, attacker: Pokemon, defender: Pokemon, move: Move
) -> bool:
    """True only for a genuine immunity: a damaging move that cannot do anything.

    Excludes status moves (the calc returns 0 for all of them) and protected
    defenders (also 0, but temporary and not a property of the matchup).
    """
    if move.category == MoveCategory.STATUS or move.base_power <= 0:
        return False
    if is_protected(defender) or defender.fainted:
        return False
    raw = damage_range(battle, attacker, defender, move)
    if raw is not None:
        return raw[1] == 0
    # The calc could not evaluate this pairing -- almost always because the defender
    # has no stats, which is what happens whenever an opponent denies Open Team Sheets.
    # Returning "not immune" here is what let the bot fire Dragon Claw into a Fairy on
    # ladder: the type-chart answer was available the whole time, it just was not being
    # asked for. The calc is still preferred when it works, since it also accounts for
    # absorb abilities, Tera typing and Ring Target, which the chart misses.
    mult = type_multiplier(move, defender)
    return mult == 0


def guaranteed_ko(
    battle: DoubleBattle, attacker: Pokemon, defender: Pokemon, move: Move
) -> bool:
    """True if the MINIMUM roll already removes the defender's remaining HP.

    Named for what it is: a KO *if the move connects*. Accuracy is not in the calc,
    so callers must not read this as 'this wins the turn'.
    """
    frac = damage_fraction(battle, attacker, defender, move)
    if frac is None or defender.fainted:
        return False
    return frac[0] >= (defender.current_hp_fraction or 0.0)


KNOWLEDGE_LEN = 4 * 5 + 4


def pokemon_knowledge(battle: DoubleBattle, mon: Pokemon, is_ours: bool) -> list[float]:
    """The 24-float knowledge block for one Pokemon token.

    Layout (must stay stable -- it maps to trailing columns of pokemon_proj):
      per move slot m in 0..3, five floats:
        dmg vs foe0, dmg vs foe1, ko vs foe0, ko vs foe1, log2(type mult)/2 vs foe0
      then four per-mon floats:
        worst incoming fraction, faints-this-turn flag, speed rank, ally splash

    Only meaningful for active Pokemon; benched mons get zeros, which the network can
    tell apart via the existing active_a/active_b flags.
    """
    out = [0.0] * KNOWLEDGE_LEN
    foes = [
        f for f in battle.opponent_active_pokemon if f is not None and not f.fainted
    ]
    if not is_ours or not foes or mon.fainted:
        return out

    move_list = list(mon.moves.values())[:4]
    for m, move in enumerate(move_list):
        if move.category == MoveCategory.STATUS:
            continue
        base = m * 5
        for f, foe in enumerate(foes[:2]):
            frac = damage_fraction(battle, mon, foe, move)
            if frac is None:
                continue
            out[base + f] = min((frac[0] + frac[1]) / 2, 1.5)
            out[base + 2 + f] = float(guaranteed_ko(battle, mon, foe, move))
        # type multiplier against the primary target, log-scaled so 0.25x..4x maps
        # to roughly -1..1 and immunity is a distinct floor
        try:
            mult = foes[0].damage_multiplier(move)
            out[base + 4] = -1.0 if mult == 0 else max(-1.0, min(1.0, _log2(mult) / 2))
        except Exception:
            pass

    worst = 0.0
    for foe in foes[:2]:
        worst = max(worst, best_damage(battle, foe, mon))
    out[20] = min(worst, 1.5)
    out[21] = float(worst >= (mon.current_hp_fraction or 0.0))
    spe = (mon.stats or {}).get("spe") or 0
    foe_spes = [(f.stats or {}).get("spe") or 0 for f in foes[:2]]
    out[22] = 1.0 if foe_spes and spe > max(foe_spes) else 0.0
    return out


def _log2(x: float) -> float:
    from math import log2

    return log2(x)


def best_damage(
    battle: DoubleBattle,
    attacker: Pokemon,
    defender: Pokemon,
    moves: Iterable[Move] | None = None,
) -> float:
    """Highest expected damage fraction over `moves` (default: the attacker's own)."""
    best = 0.0
    for move in moves if moves is not None else list(attacker.moves.values())[:4]:
        if move.category == MoveCategory.STATUS:
            continue
        frac = damage_fraction(battle, attacker, defender, move)
        if frac is None:
            continue
        expected = (frac[0] + frac[1]) / 2 * (move.accuracy or 1.0)
        best = max(best, expected)
    return best
