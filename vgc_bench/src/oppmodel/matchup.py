"""Matchup estimates for the opponent predictor (layout version 2).

Pure functions over plain data: who moves first, and how hard a move of one
Pokemon hits another, under ONE assumed stat line per Pokemon (the project's
imputation: full HP investment, full investment in the better attack stat, a
neutral nature, no item, no ability). ``features.Featurizer`` turns them into
three small integer arrays per example (``mu_cand``, ``mu_slot``,
``mu_roster``); nothing here reads a battle object, a team file or a label,
and nothing here imports torch or poke-env.

The damage chain is the simulator's integer chain at the maximum roll (base
damage, spread, weather, always-crit, STAB, type steps, burn, minimum 1),
then one factor for the mean of the sixteen rolls. Left out on purpose:
items, abilities, screens, terrain, random critical hits, accuracy, Protect,
fixed-damage and one-hit-KO moves (no estimate), and the base-power
overrides of the Champions mod.

Game rules that need a type or a status are a small table below (type ids
and status codes); the weather, side-condition and field ids they attach to
are found in the dex by property (``rules``), never named.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

# Any change of a definition or of a width below, once a version-2 dataset
# exists, raises this number (new columns go at the END of their block).
MATCHUP_VERSION = 1

MU_CAND_COLUMNS: tuple[str, ...] = (
    "est",
    "immune",
    "eff",
    "dmg",
    "ko",
    "kill",
    "first",
    "kill_first",
)
MU_SLOT_COLUMNS: tuple[str, ...] = (
    "first",
    "first_sure",
    "spd",
    "in_dmg",
    "in_ko",
    "in_kill",
    "in_kill_first",
    "out_dmg",
    "out_ko",
    "out_kill",
)
MU_ROSTER_COLUMNS: tuple[str, ...] = ("in_dmg", "in_ko", "in_kill", "out_dmg")

SCALE = 50  # a damage share of 1.0 (the whole assumed HP) is stored as 50
# What the network multiplies each stored column by.
MU_CAND_SCALES: tuple[float, ...] = (1.0, 1.0, 0.5, 0.02, 0.02, 1.0, 1.0, 1.0)
MU_SLOT_SCALES: tuple[float, ...] = (
    1.0,
    1.0,
    0.02,
    0.02,
    0.02,
    1.0,
    1.0,
    0.02,
    0.02,
    1.0,
)
MU_ROSTER_SCALES: tuple[float, ...] = (0.02, 0.02, 1.0, 0.02)

# The assumed stat line (Champions stat points; vgc_knowledge.ensure_stats).
POINTS_HP = 32
POINTS_ATTACK = 32
POINTS_SPEED = 2
POINTS_CAP = 32
STAT_FLAT = 20  # every stat but HP: base + points + 20
HP_FLAT = 75  # HP: base + points + 75
LEVEL_FACTOR = 22  # tr(2 * level / 5 + 2) at level 50
PROXY_POWER = 80  # the stand-in move of a type whose real move is not public
ROLL = 0.925  # mean of the sixteen damage rolls (0.85 .. 1.00)
SPEED_DOUBLE = 2.0  # factor of a speed-doubling side condition
NATURE_LOW, NATURE_HIGH = 0.9, 1.1

STATS: tuple[str, ...] = ("hp", "atk", "def", "spa", "spd", "spe")
PHYSICAL, SPECIAL, STATUS = "Physical", "Special", "Status"
SPREAD_FOES, SPREAD_ALL = "allAdjacentFoes", "allAdjacent"

# --- game rules, by type id and status code ------------------------------------
# A weather is known here only by the TYPE of the move that sets it (see
# ``rules``): which move type it strengthens and which it weakens ...
WEATHER_MOVE: dict[str, tuple[str, str]] = {
    "fire": ("fire", "water"),
    "water": ("water", "fire"),
}
# ... and which defender type it guards, on which defence stat.
WEATHER_GUARD: dict[str, tuple[str, str]] = {
    "rock": ("rock", "spd"),
    "ice": ("ice", "def"),
}
BURN = "brn"  # halves the damage of a physical move
PARALYSIS = "par"  # halves speed


def ident(value: Any) -> str:
    """A dex id: lower case, letters and digits only."""
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


# --- move facts ----------------------------------------------------------------


@dataclass(frozen=True)
class MoveFacts:
    """What the estimate needs of one damaging move (dex properties only)."""

    type: str  # type id; '' for a move whose type is decided by a callback
    physical: bool
    power: int
    est: bool  # the chain applies: a power, no fixed damage, no one-hit KO
    priority: int
    spread: str  # SPREAD_FOES / SPREAD_ALL / '' (single target)
    attack_stat: str
    defence_stat: str
    from_target: bool  # the attack stat is read from the target
    ignore_defensive: bool  # the target's defence stages are ignored
    crit: bool  # always a critical hit
    hits: float  # expected number of hits


def _hits(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            low, high = int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return 1.0
        if (low, high) == (2, 5):
            return 3.1  # 7/20, 7/20, 3/20, 3/20
        return (low + high) / 2.0
    return 1.0


def move_facts(entry: Mapping[str, Any] | None) -> MoveFacts | None:
    """Facts of a dex move entry; None for a Status move or no entry."""
    if not entry:
        return None
    category = entry.get("category")
    if category not in (PHYSICAL, SPECIAL):
        return None
    physical = category == PHYSICAL
    try:
        power = int(entry.get("basePower") or 0)
    except (TypeError, ValueError):
        power = 0
    try:
        priority = int(entry.get("priority") or 0)
    except (TypeError, ValueError):
        priority = 0
    fixed = any(entry.get(key) for key in ("damage", "ohko", "damageCallback"))
    target = str(entry.get("target") or "")
    attack = entry.get("overrideOffensiveStat")
    defence = entry.get("overrideDefensiveStat")
    return MoveFacts(
        type="" if "onModifyType" in entry else ident(entry.get("type") or ""),
        physical=physical,
        power=power,
        est=power > 0 and not fixed,
        priority=priority,
        spread=target if target in (SPREAD_FOES, SPREAD_ALL) else "",
        attack_stat=attack if attack in STATS[1:5] else ("atk" if physical else "spa"),
        defence_stat=defence
        if defence in ("def", "spd")
        else ("def" if physical else "spd"),
        from_target=entry.get("overrideOffensivePokemon") == "target",
        ignore_defensive=bool(entry.get("ignoreDefensive")),
        crit=bool(entry.get("willCrit")),
        hits=_hits(entry.get("multihit")),
    )


def proxy(type_id: str, physical: bool) -> MoveFacts:
    """The stand-in for a same-type move that is not public: one target, no extras."""
    return MoveFacts(
        type=type_id,
        physical=physical,
        power=PROXY_POWER,
        est=True,
        priority=0,
        spread="",
        attack_stat="atk" if physical else "spa",
        defence_stat="def" if physical else "spd",
        from_target=False,
        ignore_defensive=False,
        crit=False,
        hits=1.0,
    )


# --- dex-derived rules ---------------------------------------------------------


@dataclass(frozen=True)
class Rules:
    """Ids the estimate reacts to, found in the dex by property."""

    weather_types: dict[str, str]  # weather id -> type id of the move that sets it
    speed_double: tuple[str, ...]  # side-condition ids that double speed
    speed_reverse: tuple[str, ...]  # field ids that reverse the speed order

    def to_dict(self) -> dict[str, Any]:
        return {
            "weather_types": dict(sorted(self.weather_types.items())),
            "speed_double": list(self.speed_double),
            "speed_reverse": list(self.speed_reverse),
        }


def rules(moves: Mapping[str, Any]) -> Rules:
    """The three rules from a move dex (``events.moves_dex()``). Never raises.

    * a weather's type: the type of the Status move(s) that set it; a weather
      two moves give different types is dropped;
    * a speed-doubling side condition: set by a Status move whose condition
      modifies speed;
    * an order-reversing field: set by a Status move of negative priority.
    """
    weather: dict[str, str] = {}
    clash: set[str] = set()
    double: set[str] = set()
    reverse: set[str] = set()
    try:
        for entry in moves.values():
            if not isinstance(entry, Mapping) or entry.get("category") != STATUS:
                continue
            if entry.get("weather"):
                name, kind = ident(entry["weather"]), ident(entry.get("type") or "")
                if name and kind:
                    if weather.setdefault(name, kind) != kind:
                        clash.add(name)
            condition = entry.get("condition")
            if (
                entry.get("sideCondition")
                and isinstance(condition, Mapping)
                and "onModifySpe" in condition
            ):
                double.add(ident(entry["sideCondition"]))
            if entry.get("pseudoWeather"):
                try:
                    negative = int(entry.get("priority") or 0) < 0
                except (TypeError, ValueError):
                    negative = False
                if negative:
                    reverse.add(ident(entry["pseudoWeather"]))
    except Exception:
        return Rules({}, (), ())
    for name in clash:
        weather.pop(name, None)
    return Rules(weather, tuple(sorted(double)), tuple(sorted(reverse)))


# --- stats ---------------------------------------------------------------------


def stat_line(base_stats: Mapping[str, Any] | None) -> dict[str, int] | None:
    """The assumed stats of a forme from its base stats; None when unusable.

    Also carries ``base_spe`` (the speed bounds are made from the base).
    """
    if not base_stats:
        return None
    try:
        base = {key: int(base_stats[key]) for key in STATS}
    except (KeyError, TypeError, ValueError):
        return None
    physical = base["atk"] >= base["spa"]
    return {
        "hp": base["hp"] + POINTS_HP + HP_FLAT,
        "atk": base["atk"] + STAT_FLAT + (POINTS_ATTACK if physical else 0),
        "def": base["def"] + STAT_FLAT,
        "spa": base["spa"] + STAT_FLAT + (0 if physical else POINTS_ATTACK),
        "spd": base["spd"] + STAT_FLAT,
        "spe": base["spe"] + STAT_FLAT + POINTS_SPEED,
        "base_spe": base["spe"],
        "physical": int(physical),
    }


def staged(value: int, stage: Any) -> int:
    """``value`` under a stat stage, truncated as the simulator does."""
    try:
        s = max(-6, min(6, int(stage)))
    except (TypeError, ValueError):
        s = 0
    return value * (2 + s) // 2 if s >= 0 else value * 2 // (2 - s)


def stage_factor(stage: Any) -> float:
    try:
        s = max(-6, min(6, int(stage)))
    except (TypeError, ValueError):
        s = 0
    return (2 + s) / 2.0 if s >= 0 else 2.0 / (2 - s)


def modify(value: int, factor: float) -> int:
    """The simulator's fixed-point modifier (4096ths, rounded half down)."""
    return (value * int(factor * 4096) + 2047) // 4096


# --- damage --------------------------------------------------------------------


def effectiveness(
    chart: Mapping[str, Mapping[str, float]],
    move_type: str,
    defender_types: Sequence[str],
) -> tuple[int, bool]:
    """(doubling steps, immune) of a move type on a defender's types.

    ``chart[attack type id][defender type id]`` is the multiplier; a pair it
    does not hold counts as 1.
    """
    row = chart.get(move_type)
    if not row:
        return 0, False
    steps = 0
    for kind in defender_types:
        value = row.get(kind, 1.0)
        if value == 0:
            return 0, True
        if value > 1:
            steps += 1
        elif value < 1:
            steps -= 1
    return steps, False


def multiplier(
    chart: Mapping[str, Mapping[str, float]],
    move_type: str,
    defender_types: Sequence[str],
) -> float:
    """The type multiplier as a number (0 when immune)."""
    steps, immune = effectiveness(chart, move_type, defender_types)
    return 0.0 if immune else 2.0**steps


def max_damage(
    power: int,
    attack: int,
    defence: int,
    *,
    spread: bool = False,
    weather: float = 1.0,
    crit: bool = False,
    stab: bool = False,
    steps: int = 0,
    burn: bool = False,
) -> int:
    """HP lost to one hit at the maximum roll: the simulator's integer chain."""
    x = (LEVEL_FACTOR * power * attack // max(1, defence)) // 50 + 2
    if spread:
        x = modify(x, 0.75)
    if weather != 1.0:
        x = modify(x, weather)
    if crit:
        x = x * 3 // 2
    if stab:
        x = modify(x, 1.5)
    if steps > 0:
        x <<= steps
    else:
        for _ in range(-steps):
            x //= 2
    if burn:
        x = modify(x, 0.5)
    return max(1, x)


@dataclass(frozen=True)
class Fighter:
    """One Pokemon as the estimate sees it: public state plus the assumed line."""

    line: Mapping[str, int] | None  # ``stat_line``; None = no estimate at all
    types: tuple[str, ...]
    boosts: Mapping[str, int]
    status: str | None
    hp: float  # public HP fraction
    attacks: tuple[MoveFacts, ...] = ()  # known damaging moves, then stand-ins
    speed_factor: float = 1.0  # side conditions on its side


@dataclass(frozen=True)
class Board:
    """What is shared by every pair of one example."""

    chart: Mapping[str, Mapping[str, float]]
    weather_type: str = ""  # type id of the weather-setting move, '' = none
    reverse: bool = False  # an order-reversing field is up


def hit(
    move: MoveFacts,
    user: Fighter,
    target: Fighter,
    board: Board,
    n_opposing: int = 1,
    n_allies: int = 0,
) -> tuple[int, int, int, float]:
    """(est, immune, type steps, expected share of the target's assumed HP).

    ``n_opposing`` / ``n_allies`` count the Pokemon on the field on the
    target's side and beside the user: they decide the spread modifier.
    """
    if user.line is None or target.line is None:
        return 0, 0, 0, 0.0
    steps, immune = 0, False
    if move.type:
        steps, immune = effectiveness(board.chart, move.type, target.types)
    if immune:
        return int(move.est), 1, 0, 0.0
    if not move.est:
        return 0, 0, steps, 0.0
    damage = _one_hit(move, user, target, board, steps, n_opposing, n_allies)
    return 1, 0, steps, ROLL * damage * move.hits / max(1, target.line["hp"])


def max_hit(
    move: MoveFacts,
    user: Fighter,
    target: Fighter,
    board: Board,
    n_opposing: int = 1,
    n_allies: int = 0,
) -> int | None:
    """HP one hit takes at the maximum roll; 0 when immune, None without an estimate."""
    if user.line is None or target.line is None or not move.est:
        return None
    steps, immune = 0, False
    if move.type:
        steps, immune = effectiveness(board.chart, move.type, target.types)
    if immune:
        return 0
    return _one_hit(move, user, target, board, steps, n_opposing, n_allies)


def _one_hit(
    move: MoveFacts,
    user: Fighter,
    target: Fighter,
    board: Board,
    steps: int,
    n_opposing: int,
    n_allies: int,
) -> int:
    source = target if move.from_target else user
    source_line = source.line or {}
    target_line = target.line or {}
    attack = staged(
        source_line[move.attack_stat], source.boosts.get(move.attack_stat, 0)
    )
    defence = target_line[move.defence_stat]
    if not move.ignore_defensive:
        defence = staged(defence, target.boosts.get(move.defence_stat, 0))
    guard = WEATHER_GUARD.get(board.weather_type)
    if guard is not None and guard[1] == move.defence_stat and guard[0] in target.types:
        defence = modify(defence, 1.5)
    weather = 1.0
    if move.type:
        strong_weak = WEATHER_MOVE.get(board.weather_type)
        if strong_weak is not None:
            if move.type == strong_weak[0]:
                weather = 1.5
            elif move.type == strong_weak[1]:
                weather = 0.5
    spread = (move.spread == SPREAD_FOES and n_opposing >= 2) or (
        move.spread == SPREAD_ALL and n_opposing + n_allies >= 2
    )
    return max_damage(
        move.power,
        attack,
        defence,
        spread=spread,
        weather=weather,
        crit=move.crit,
        stab=bool(move.type) and move.type in user.types,
        steps=steps,
        burn=move.physical and user.status == BURN,
    )


def best(
    user: Fighter, target: Fighter, board: Board, n_opposing: int = 1, n_allies: int = 0
) -> float:
    """The largest expected share over ``user.attacks``; 0 when nothing applies."""
    top = 0.0
    for move in user.attacks:
        share = hit(move, user, target, board, n_opposing, n_allies)[3]
        if share > top:
            top = share
    return top


# --- speed ---------------------------------------------------------------------


def speeds(fighter: Fighter) -> tuple[float, float, float] | None:
    """(slowest, assumed, fastest) speed of a Pokemon, field effects included."""
    line = fighter.line
    if line is None:
        return None
    mods = stage_factor(fighter.boosts.get("spe", 0)) * fighter.speed_factor
    if fighter.status == PARALYSIS:
        mods *= 0.5
    base = line["base_spe"]
    return (
        (base + STAT_FLAT) * NATURE_LOW * mods,
        (base + STAT_FLAT + POINTS_SPEED) * mods,
        (base + STAT_FLAT + POINTS_CAP) * NATURE_HIGH * mods,
    )


def order(
    mine: tuple[float, float, float] | None,
    theirs: tuple[float, float, float] | None,
    reverse: bool = False,
) -> tuple[int, int, float]:
    """(first, first_sure, log2 speed ratio clipped to [-1, 1]) of mine vs theirs.

    ``first`` compares the assumed speeds; ``first_sure`` is non-zero only when
    the whole range of one is beyond the whole range of the other. All three
    change sign under an order-reversing field.
    """
    if mine is None or theirs is None or mine[1] <= 0 or theirs[1] <= 0:
        return 0, 0, 0.0
    first = (mine[1] > theirs[1]) - (mine[1] < theirs[1])
    sure = 1 if mine[0] > theirs[2] else (-1 if mine[2] < theirs[0] else 0)
    spd = math.log2(min(2.0, max(0.5, mine[1] / theirs[1])))
    if reverse:
        return -first, -sure, -spd
    return first, sure, spd


# --- quantisation --------------------------------------------------------------


def q(value: float) -> int:
    """A share in [0, 2] as an integer 0 .. 100."""
    return int(min(2.0, max(0.0, value)) * SCALE + 0.5)


def qlog(value: float) -> int:
    """A value in [-1, 1] as an integer -50 .. 50, odd in its argument."""
    size = int(min(1.0, abs(value)) * SCALE + 0.5)
    return size if value >= 0 else -size


def ko_ratio(share: float, hp: float) -> float:
    return share / max(hp, 0.01)


def kills(share: float, hp: float) -> bool:
    """The unrounded expected damage covers the public HP that is left."""
    return share > 0.0 and share >= hp


def signature(found: Rules) -> dict[str, Any]:
    """Plain data that pins the definitions a version-2 dataset was built with."""
    return {
        "version": MATCHUP_VERSION,
        "columns": {
            "mu_cand": list(MU_CAND_COLUMNS),
            "mu_slot": list(MU_SLOT_COLUMNS),
            "mu_roster": list(MU_ROSTER_COLUMNS),
        },
        "assumptions": {
            "points_hp": POINTS_HP,
            "points_attack": POINTS_ATTACK,
            "points_speed": POINTS_SPEED,
            "proxy_power": PROXY_POWER,
            "roll": ROLL,
            "scale": SCALE,
        },
        "rules": found.to_dict(),
    }
