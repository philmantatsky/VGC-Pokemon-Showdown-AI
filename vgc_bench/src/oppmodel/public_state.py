"""Event-fed public battle state: one tracker for training logs and the live bot.

``PublicBattle`` consumes raw split events (``['', 'move', 'p1a: X', ...]``) and
holds everything a spectator could know, indexed by absolute side (``p1`` /
``p2``) and by team-preview order. It never looks at a poke-env battle object,
so nothing private (request data, exact HP) can reach a feature.

Offline::

    result = drive_log(log_text, battle_id)          # never raises
    if result.usable:
        for record in result.turns:
            record.snapshot    # PublicSnapshot at the start of record.turn
            record.actions     # {'p1a': SlotAction, ...} for that turn, both sides

Live (inside the bot, once per decision)::

    shadow = LiveShadow()                             # one per battle
    shadow.feed_sheet(side, sets)                     # open sheets, if any ...
    shadow.mark_sheets_known()                        # ... or "none were shown"
    snapshot = shadow.sync(battle)                    # PublicSnapshot or None

``LiveShadow.sync`` reads ``battle._replay_data`` from a cursor, rewrites the
bot's own exact HP to the public percent the opponent sees, and feeds the same
``PublicBattle``; a saved player-view replay pushed through it gives snapshots
identical to ``drive_log`` on the HP-rewritten log (checked by
``evaluation/oppmodel_parity.py``).

A ``PublicSnapshot`` is plain data (``to_dict()`` for comparisons). It is taken
right after the ``|turn|N`` event and contains nothing from turn N's own events.
Unknown values are ``None``, never a default, with two exceptions by design: the
public HP of a Pokemon that has not been seen is 1.0, and ``rated`` is False both
for a game that is not a ladder game and when no ``|rated|`` line was seen.

Team sheets need a word from the caller. A spectator log shows an open sheet as
a ``|showteam|`` line, so its absence means closed. A player-view stream (a
saved own replay page, ``battle._replay_data``) never carries that line:
poke-env consumes it before the stream. There ``sheet_open`` is None (unknown)
for a side until the caller hands the sheet over or says that none was shown
(``sheets_known``).

What the tracker keeps that poke-env does not, or gets wrong: moves revealed in
battle versus moves known only from the sheet, the Mega stone and consumed
items, a per-Pokemon Mega flag, effect starts as (turn, after upkeep),
"protected last turn" from the protecting move's own success line, kept apart
from the stall counter (``protect_streak``) that side guards raise as well,
turns on the field, the previous turn's action of every slot
(``events.read_turn_actions``), the bring state, volatiles only from ``-start``
/ ``-end`` lines (no stale single-turn effects), an ability copied or replaced
on the field kept apart from the Pokemon's own, and the ratings on the first
``|player|`` lines, which trailing empty ``|player|`` lines must not erase.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence

from vgc_bench.src.oppmodel.events import (
    CHATTER_KINDS,
    SIDES,
    SLOT_KEYS,
    Event,
    FieldTracker,
    SheetSet,
    SlotAction,
    TurnSegment,
    dex_available,
    effect_id,
    forme_id,
    is_mega_forme,
    is_protect_family,
    move_entry,
    moves_dex,
    parse_ident,
    parse_showteam,
    pokedex,
    raises_stall_counter,
    read_turn_actions,
    species_from_details,
    split_log,
    tag_value,
    to_id,
    user_id,
)

BOOST_STATS: tuple[str, ...] = (
    "atk",
    "def",
    "spa",
    "spd",
    "spe",
    "accuracy",
    "evasion",
)

ITEM_UNKNOWN = "unknown"
ITEM_KNOWN = "known"
ITEM_CONSUMED = "consumed"

SKIP_NO_TURN = "no_turn_1"
SKIP_ILLUSION = "illusion_species"
SKIP_NOT_DOUBLES = "not_doubles"
SKIP_TURN_ORDER = "turn_not_increasing"
SKIP_DEX_MISSING = "dex_missing"

# ``rated_kind`` of a ladder game: a ``|rated|`` line with nothing after it. Any
# other value is the id of the server's note (a tournament game has no Elo).
RATED_LADDER = "ladder"

# Protocol: the only ability whose holder's logged identity is false. The log
# shows it as |-end|<ident>|Illusion and |replace|; such battles are dropped.
_FALSE_IDENTITY_TAG = "illusion"

_CONDITION = re.compile(r"^\s*(\d+)\s*(?:/\s*(\d+))?([a-z])?(?:\s+([a-z]+))?\s*$")
_COUNTED_EFFECT = re.compile(r"^(.*?)(\d+)$")
_BEST_OF = re.compile(
    r"Game (\d+)</strong> of <a href=\"/?(game-bestof(\d+)-[a-z0-9-]+)"
)

# Message types that carry a condition string (HP and status), and where.
CONDITION_INDEX: dict[str, int] = {
    "switch": 4,
    "drag": 4,
    "replace": 4,
    "-damage": 3,
    "-heal": 3,
    "-sethp": 3,
}

# "[from] ability: X" with "[of] Y": the simulator puts the holder in either
# place, by message type and by ability. When the subject and Y are different
# Pokemon the dex decides (which of the two can have X); these are the defaults
# for when it cannot: the [of] Pokemon on these types, the subject on the rest.
_OF_HOLDS_ABILITY = frozenset(
    {"-block", "-clearboost", "-damage", "-item", "-start", "-status"}
)
# Types whose first field is not a Pokemon: the holder is the [of] Pokemon.
_NO_SUBJECT_KINDS = frozenset(
    {"-fieldend", "-fieldstart", "-sideend", "-sidestart", "-weather"}
)
# Types on which a "[from] ability:" tag says nothing about either Pokemon of
# the line: -ability has its own reading, and on -enditem the ability belongs
# to the Pokemon that took the item, which the line does not name as holder.
_ABILITY_TAG_UNREAD = frozenset({"-ability", "-enditem"})
_EFFECT_FIELD_KINDS = frozenset({"-activate", "-start", "-end", "-block", "cant"})
# Battle message types with no handler that are known to carry nothing a
# snapshot holds (their tags are still read). Any other unhandled type is
# counted under ``unhandled_kind:<type>``.
_STATELESS_KINDS = frozenset(
    {
        "-block",
        "-center",
        "-combine",
        "-crit",
        "-fail",
        "-fieldactivate",
        "-hitcount",
        "-immune",
        "-miss",
        "-notarget",
        "-nothing",
        "-ohko",
        "-resisted",
        "-singlemove",
        "-supereffective",
        "-waiting",
        "-zbroken",
        "-zpower",
        "done",
        "gen",
        "rule",
        "start",
        "teampreview",
    }
)
_KIND_NAME = re.compile(r"^[-a-z:]{1,24}$")


def parse_condition(condition: str) -> tuple[float, str | None, bool] | None:
    """(HP fraction, status id or None, fainted) from ``50/100y par`` / ``0 fnt``."""
    match = _CONDITION.match(condition)
    if match is None:
        return None
    current = int(match.group(1))
    status = match.group(4)
    if status == "fnt" or current == 0:
        return 0.0, None, True
    maximum = int(match.group(2)) if match.group(2) else 0
    if maximum <= 0:
        return None
    return min(1.0, current / maximum), status, False


def public_condition(condition: str) -> str:
    """Rewrite an exact condition (``93/186 par``) to the Champions public form.

    The server shows the other player ``max(1, floor(100 * hp / maxhp))/100``,
    with a colour letter at exactly 20 and 50 that says which side of the
    threshold the exact HP is on, and the status suffix unchanged. A fainted
    condition, an already-public one (colour letter present) and anything
    unparseable are returned as they are.
    """
    match = _CONDITION.match(condition)
    if match is None or match.group(2) is None or match.group(3) is not None:
        return condition
    current, maximum = int(match.group(1)), int(match.group(2))
    if current <= 0 or maximum <= 0:
        return condition
    percent = (100 * current) // maximum or 1
    colour = ""
    if percent == 20:
        colour = "y" if current * 5 > maximum else "r"
    elif percent == 50:
        colour = "g" if current * 2 > maximum else "y"
    status = f" {match.group(4)}" if match.group(4) else ""
    return f"{percent}/100{colour}{status}"


def rewrite_event(event: Sequence[str], role: str) -> Event:
    """``event`` with ``role``'s exact HP replaced by the public percent."""
    out = list(event)
    if len(out) < 4:
        return out
    index = CONDITION_INDEX.get(out[1])
    if index is None or len(out) <= index or not out[2].startswith(role):
        return out
    out[index] = public_condition(out[index])
    return out


@dataclass(frozen=True)
class EffectStart:
    """When a weather / field / side effect started.

    ``after_upkeep`` is True when it was set between turns (a lead's or a
    replacement's ability): such an effect has not yet lived through an
    end-of-turn tick of ``turn``.
    """

    turn: int
    after_upkeep: bool

    def ticks(self, turn: int) -> int:
        """End-of-turn ticks the effect has been through at the start of ``turn``."""
        return max(0, turn - self.turn - (1 if self.after_upkeep else 0))


@dataclass(frozen=True)
class MovePublic:
    """A known move: shown in battle, listed on the open sheet, or both."""

    id: str
    revealed: bool
    from_sheet: bool


@dataclass(frozen=True)
class MonPublic:
    """One roster Pokemon as the public sees it at the start of a turn."""

    side: str
    index: int  # position in team-preview order
    species: str  # base-species id
    forme: str  # current forme id (the Mega forme after Mega Evolution)
    types: tuple[str, ...]
    hp: float  # public fraction; 1.0 while never seen
    status: str | None
    boosts: dict[str, int]
    slot: str | None  # 'a' / 'b' while on the field and alive, else None
    revealed: bool
    fainted: bool
    brought: bool | None  # None = not known yet
    is_mega: bool
    mega_possible: bool
    first_turn: bool  # on the field, and this is its first turn there
    protected_last_turn: bool
    protect_streak: int  # turns in a row, up to last turn, the stall counter rose
    turns_on_field: int  # completed turns since it entered; 0 when off field
    item: str | None
    item_state: str  # ITEM_UNKNOWN / ITEM_KNOWN / ITEM_CONSUMED
    ability: str | None  # the one in effect: a copied one while it stays in
    ability_known: bool
    moves: tuple[MovePublic, ...]
    volatiles: dict[str, int]  # effect id -> turn it started
    last_action: SlotAction | None  # previous turn's action, if it was on field


@dataclass(frozen=True)
class SidePublic:
    side: str
    player: str | None  # display name; match accounts by DriveResult.player_ids
    rating: int | None
    sheet_open: bool | None  # None = the stream cannot tell and nobody has said
    mega_used: bool
    team_size: int | None  # how many were brought (|teamsize|)
    n_fainted: int
    n_revealed: int
    n_unknown_brought: int | None  # team_size - n_revealed
    conditions: dict[str, EffectStart]  # side-condition id -> start
    active: dict[str, int | None]  # slot letter -> roster index
    mons: tuple[MonPublic, ...]  # team-preview order


@dataclass(frozen=True)
class PublicSnapshot:
    """Everything public at the start of ``turn``. Plain data, no poke-env objects."""

    battle_id: str
    turn: int
    format_id: str | None
    best_of: int | None  # 3 for a best-of-three game, None for a single game
    rated: bool  # a ladder game; False also when no |rated| line was seen
    weather: str | None
    weather_start: EffectStart | None
    weather_side: str | None  # side that set it, when the log tells
    fields: dict[str, EffectStart]  # terrain / room ids -> start
    sides: dict[str, SidePublic]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def mon(self, side: str, slot: str) -> MonPublic | None:
        """The living Pokemon in ``side``'s slot ``a`` / ``b``, if any."""
        team = self.sides.get(side)
        index = None if team is None else team.active.get(slot)
        return None if team is None or index is None else team.mons[index]


@dataclass
class _Mon:
    side: str
    index: int
    species: str
    forme: str
    name: str = ""
    revealed: bool = False
    fainted: bool = False
    hp: float = 1.0
    status: str | None = None
    boosts: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(BOOST_STATS, 0)
    )
    active_turns: int = 0
    is_mega: bool = False
    item: str | None = None
    item_state: str = ITEM_UNKNOWN
    # Its own ability, and the one put on it while it stays in: None = not
    # replaced, "" = replaced by an ability nobody has seen.
    base_ability: str | None = None
    field_ability: str | None = None
    revealed_moves: list[str] = field(default_factory=list)
    volatiles: dict[str, int] = field(default_factory=dict)
    types_override: tuple[str, ...] | None = None
    stall_turn: int = -9
    protect_turn: int = -9
    stall_streak: int = 0
    last_action: SlotAction | None = None
    last_action_turn: int = -9

    def leave_field(self) -> None:
        self.boosts = dict.fromkeys(BOOST_STATS, 0)
        self.volatiles = {}
        self.types_override = None
        self.field_ability = None
        self.active_turns = 0
        self.stall_turn = self.protect_turn = -9
        self.stall_streak = 0


@lru_cache(maxsize=2048)
def _mega_stones(species: str) -> frozenset[str]:
    dex = pokedex()
    base = dex.get(species) or {}
    stones: set[str] = set()
    for name in base.get("otherFormes", ()):
        entry = dex.get(to_id(name)) or {}
        if "Mega" in str(entry.get("forme", "")).split("-") and entry.get(
            "requiredItem"
        ):
            stones.add(to_id(str(entry["requiredItem"])))
    return frozenset(stones)


@lru_cache(maxsize=2048)
def _forme_abilities(forme: str) -> frozenset[str]:
    abilities = (pokedex().get(forme) or {}).get("abilities") or {}
    return frozenset(to_id(str(name)) for name in abilities.values())


@lru_cache(maxsize=2048)
def _has_false_identity(forme: str) -> bool:
    return _FALSE_IDENTITY_TAG in _forme_abilities(forme)


class PublicBattle:
    """Public state of one battle, built by feeding raw events in order.

    ``feed(event)`` never raises: a failing event is counted in ``counters``
    under ``feed_error:<type>:<exception>`` and the tracker moves on.
    ``feed_sheet(side, sets)`` records an open team sheet (a ``|showteam|``
    event does this by itself; the live bot must call it, because poke-env keeps
    that line out of ``_replay_data``). ``sheets_known`` says whether a side
    without a sheet is known to be closed: True for a spectator log, False for
    a player-view stream until ``set_sheets_known()``. ``snapshot()`` returns
    the state now; ``turn_snapshot`` is the one taken right after the latest
    ``|turn|`` event (None while that turn could not be read), and
    ``at_turn_start`` tells whether anything but chatter came after it.
    ``segments`` holds every turn's events with the slot occupants at its start,
    and ``last_actions`` the previous turn's ``SlotAction`` per slot.
    ``players`` are display names and ``player_ids`` the account ids.
    """

    def __init__(self, battle_id: str = "", sheets_known: bool = True) -> None:
        self.battle_id = battle_id
        self.counters: Counter[str] = Counter()
        self.turn = 0
        self.after_upkeep = True
        self.format_id: str | None = None
        self.gametype: str | None = None
        self.rated = False
        self.rated_kind: str | None = None
        self.best_of: int | None = None
        self.series_id: str | None = None
        self.game_number: int | None = None
        self.start_time: int | None = None
        self.end_time: int | None = None
        self.sheets_known = sheets_known
        self.players: dict[str, str | None] = {side: None for side in SIDES}
        self.player_ids: dict[str, str | None] = {side: None for side in SIDES}
        self.ratings: dict[str, int | None] = {side: None for side in SIDES}
        self.team_size: dict[str, int | None] = {side: None for side in SIDES}
        self.mega_used: dict[str, bool] = {side: False for side in SIDES}
        self.sheets: dict[str, dict[str, SheetSet]] = {}
        self.weather: str | None = None
        self.weather_start: EffectStart | None = None
        self.weather_side: str | None = None
        self.fields: dict[str, EffectStart] = {}
        self.side_conditions: dict[str, dict[str, EffectStart]] = {
            side: {} for side in SIDES
        }
        self.tracker = FieldTracker()
        self.segments: list[TurnSegment] = []
        self.last_actions: dict[str, SlotAction] = {}
        self.turn_snapshot: PublicSnapshot | None = None
        self.at_turn_start = False
        self.winner: str | None = None
        self.tie = False
        self.finished = False
        self.false_identity = False
        self.events_fed = 0
        self._roster: dict[str, list[_Mon]] = {side: [] for side in SIDES}
        self._by_name: dict[str, dict[str, _Mon]] = {side: {} for side in SIDES}
        self._last_mover_side: str | None = None
        self._handlers = {
            "-ability": self._on_ability,
            "-activate": self._on_activate,
            "-anim": self._on_anim,
            "-boost": self._on_boost,
            "-clearallboost": self._on_clearallboost,
            "-clearboost": self._on_clearboost,
            "-clearnegativeboost": self._on_clearboost,
            "-clearpositiveboost": self._on_clearboost,
            "-copyboost": self._on_copyboost,
            "-curestatus": self._on_curestatus,
            "-cureteam": self._on_cureteam,
            "-damage": self._on_hp,
            "-end": self._on_end,
            "-enditem": self._on_enditem,
            "-fieldend": self._on_fieldend,
            "-fieldstart": self._on_fieldstart,
            "-formechange": self._on_formechange,
            "-heal": self._on_hp,
            "-invertboost": self._on_invertboost,
            "-item": self._on_item,
            "-mega": self._on_mega,
            "-mustrecharge": self._on_marker,
            "-prepare": self._on_marker,
            "-setboost": self._on_setboost,
            "-sethp": self._on_hp,
            "-sideend": self._on_sideend,
            "-sidestart": self._on_sidestart,
            "-singleturn": self._on_singleturn,
            "-start": self._on_start,
            "-status": self._on_status,
            "-swapboost": self._on_swapboost,
            "-swapsideconditions": self._on_swapsideconditions,
            "-unboost": self._on_boost,
            "-weather": self._on_weather,
            "cant": self._on_cant,
            "clearpoke": self._on_clearpoke,
            "detailschange": self._on_formechange,
            "drag": self._on_switch,
            "faint": self._on_faint,
            "gametype": self._on_gametype,
            "move": self._on_move,
            "player": self._on_player,
            "poke": self._on_poke,
            "rated": self._on_rated,
            "replace": self._on_replace,
            "showteam": self._on_showteam,
            "swap": self._on_swap,
            "switch": self._on_switch,
            "t:": self._on_time,
            "teamsize": self._on_teamsize,
            "tie": self._on_tie,
            "tier": self._on_tier,
            "uhtml": self._on_uhtml,
            "upkeep": self._on_upkeep,
            "win": self._on_win,
        }

    # --- feeding --------------------------------------------------------------

    def feed(self, event: Sequence[str]) -> None:
        kind = "?"
        try:
            if not isinstance(event, (list, tuple)) or len(event) < 2:
                self.counters["feed_error:?:not_an_event"] += 1
                return
            if not all(isinstance(part, str) for part in event):
                self.counters["feed_error:?:not_an_event"] += 1
                return
            kind = event[1]
            self._apply(event, kind)
        except Exception as exc:  # the caller is a decision path: count, go on
            self.counters[f"feed_error:{kind}:{type(exc).__name__}"] += 1

    def _apply(self, event: Sequence[str], kind: str) -> None:
        self.events_fed += 1
        if kind == "turn":
            self._on_turn(event)
            return
        if self.segments:
            self.segments[-1].events.append(list(event))
        handler = self._handlers.get(kind)
        if kind not in CHATTER_KINDS:
            known = handler is not None or kind in _STATELESS_KINDS
            if not known:
                name = kind if _KIND_NAME.match(kind) else "?"
                self.counters[f"unhandled_kind:{name}"] += 1
            # Battle actions end the turn start. An unknown type does so only
            # when it is shaped like one (minor actions start with "-"): a new
            # room message must not silently cancel a turn's forecast.
            if known or kind.startswith("-"):
                self.at_turn_start = False
        if handler is not None:
            handler(event)
        if len(event) > 3:
            self._reveal_from_tags(event, kind)

    def feed_sheet(self, side: str, sets: Iterable[SheetSet]) -> None:
        """Record ``side``'s open team sheet; its moves are flagged ``from_sheet``.

        Given at the start of a turn it is part of that turn's snapshot.
        """
        try:
            if side not in SIDES:
                self.counters["sheet_bad_side"] += 1
                return
            self.sheets[side] = {entry.species: entry for entry in sets}
            self._refresh_turn_snapshot()
        except Exception as exc:
            self.counters[f"feed_error:sheet:{type(exc).__name__}"] += 1

    def set_sheets_known(self, known: bool = True) -> None:
        """Say whether a side without a sheet is known to be closed."""
        try:
            self.sheets_known = bool(known)
            self._refresh_turn_snapshot()
        except Exception as exc:
            self.counters[f"feed_error:sheet:{type(exc).__name__}"] += 1

    def _refresh_turn_snapshot(self) -> None:
        # Only what the caller knew when the turn began may enter: nothing of
        # the turn's own events has been fed while at_turn_start holds.
        if self.at_turn_start and self.turn_snapshot is not None:
            self.turn_snapshot = self.snapshot()

    # --- lookup ---------------------------------------------------------------

    def _mon_for(self, side: str, species: str, forme: str) -> _Mon:
        for mon in self._roster[side]:
            if mon.species == species:
                return mon
        mon = _Mon(side, len(self._roster[side]), species, forme)
        self._roster[side].append(mon)
        self.counters["roster_added_in_battle"] += 1
        return mon

    def _in_slot(self, key: str) -> _Mon | None:
        occupant = self.tracker.active.get(key)
        if occupant is None:
            return None
        for mon in self._roster[key[:2]]:
            if mon.species == occupant.species:
                return mon
        return None

    def _resolve(self, text: str) -> _Mon | None:
        ident = parse_ident(text)
        if ident is None or ident.side not in self._roster:
            return None
        if ident.key is not None:
            mon = self._in_slot(ident.key)
            if mon is not None:
                return mon
        mon = self._by_name[ident.side].get(ident.name)
        if mon is not None:
            return mon
        species = species_from_details(ident.name)
        for candidate in self._roster[ident.side]:
            if candidate.species == species:
                return candidate
        return None

    def _now(self) -> EffectStart:
        return EffectStart(self.turn, self.after_upkeep)

    # --- preamble -------------------------------------------------------------

    def _on_player(self, event: Sequence[str]) -> None:
        side = event[2]
        name = event[3] if len(event) > 3 else ""
        # A trailing "|player|p1|" (the player left) must not erase name or rating.
        if side not in self.players or not name or self.players[side] is not None:
            return
        self.players[side] = name
        self.player_ids[side] = user_id(name)
        rating = event[5].strip() if len(event) > 5 else ""
        self.ratings[side] = int(rating) if rating.isdigit() else None

    def _on_gametype(self, event: Sequence[str]) -> None:
        self.gametype = event[2]

    def _on_tier(self, event: Sequence[str]) -> None:
        self.format_id = re.sub("[^a-z0-9]+", "", event[2].lower())

    def _on_rated(self, event: Sequence[str]) -> None:
        # "|rated|" is a ladder game; "|rated|<note>" (a tournament game) has no
        # Elo on its player lines.
        note = to_id("|".join(event[2:]))
        self.rated_kind = note or RATED_LADDER
        self.rated = self.rated_kind == RATED_LADDER

    def _on_time(self, event: Sequence[str]) -> None:
        stamp = event[2].strip() if len(event) > 2 else ""
        if stamp.isdigit():
            if self.start_time is None:
                self.start_time = int(stamp)
            self.end_time = int(stamp)

    def _on_uhtml(self, event: Sequence[str]) -> None:
        if len(event) < 4 or event[2] != "bestof":
            return
        match = _BEST_OF.search("|".join(event[3:]))
        if match is not None:
            self.game_number = int(match.group(1))
            self.series_id = match.group(2)
            self.best_of = int(match.group(3))

    def _on_clearpoke(self, event: Sequence[str]) -> None:
        if self.turn == 0 and not any(mon.revealed for mon in self._all_mons()):
            self._roster = {side: [] for side in SIDES}
            self._by_name = {side: {} for side in SIDES}

    def _on_poke(self, event: Sequence[str]) -> None:
        side, details = event[2], event[3]
        if side not in self._roster:
            return
        self._roster[side].append(
            _Mon(
                side,
                len(self._roster[side]),
                species_from_details(details),
                forme_id(details),
            )
        )

    def _on_teamsize(self, event: Sequence[str]) -> None:
        if event[2] in self.team_size:
            self.team_size[event[2]] = int(event[3])

    def _on_showteam(self, event: Sequence[str]) -> None:
        self.feed_sheet(event[2], parse_showteam("|".join(event[3:])))

    # --- field ----------------------------------------------------------------

    def _on_switch(self, event: Sequence[str]) -> None:
        ident = parse_ident(event[2])
        if ident is None or ident.key not in SLOT_KEYS:
            self.counters["switch_bad_ident"] += 1
            return
        details = event[3]
        # A fainted occupant has already left; it may even stand, revived, in
        # the other slot, so it must not be reset again here.
        occupant = self.tracker.active.get(ident.key)
        standing = occupant is not None and not occupant.fainted
        leaving = self._in_slot(ident.key) if standing else None
        passed: dict[str, int] | None = None
        if leaving is not None:
            origin = tag_value(event, "[from]", 5)
            entry = move_entry(effect_id(origin)) if origin else None
            if entry is not None and entry.get("selfSwitch") == "copyvolatile":
                passed = dict(leaving.boosts)
            leaving.leave_field()
        mon = self._mon_for(
            ident.side, species_from_details(details), forme_id(details)
        )
        self.tracker.apply(event)
        mon.leave_field()
        if passed is not None:
            mon.boosts = passed
        mon.name = ident.name
        self._by_name[ident.side][ident.name] = mon
        mon.revealed = True
        self._set_forme(mon, forme_id(details))
        if len(event) > 4:
            self._set_condition(mon, event[4])

    def _on_replace(self, event: Sequence[str]) -> None:
        ident = parse_ident(event[2])
        if ident is None or ident.key not in SLOT_KEYS:
            return
        self.false_identity = True
        shown = self._in_slot(ident.key)
        details = event[3]
        real = self._mon_for(
            ident.side, species_from_details(details), forme_id(details)
        )
        self.tracker.apply(event)
        if shown is not None and shown is not real:
            real.hp, real.status = shown.hp, shown.status
            real.boosts, real.volatiles = dict(shown.boosts), dict(shown.volatiles)
            real.active_turns = shown.active_turns
        real.name = ident.name
        real.revealed = True
        real.fainted = False
        self._by_name[ident.side][ident.name] = real

    def _on_swap(self, event: Sequence[str]) -> None:
        self.tracker.apply(event)

    def _on_faint(self, event: Sequence[str]) -> None:
        mon = self._resolve(event[2])
        self.tracker.apply(event)
        if mon is None:
            self.counters["faint_unresolved"] += 1
            return
        mon.leave_field()
        mon.fainted = True
        mon.hp = 0.0
        mon.status = None

    def _set_forme(self, mon: _Mon, forme: str) -> None:
        if forme == mon.forme:
            return
        mon.forme = forme
        if is_mega_forme(forme) and not mon.is_mega:
            mon.is_mega = True
            self.mega_used[mon.side] = True
            # The Mega forme has its own ability; the base one no longer applies.
            mon.base_ability = mon.field_ability = None

    def _on_formechange(self, event: Sequence[str]) -> None:
        mon = self._resolve(event[2])
        self.tracker.apply(event)
        if mon is not None:
            self._set_forme(mon, forme_id(event[3]))

    def _on_mega(self, event: Sequence[str]) -> None:
        mon = self._resolve(event[2])
        if mon is None:
            self.counters["mega_unresolved"] += 1
            return
        if not mon.is_mega:
            mon.is_mega = True
            mon.base_ability = mon.field_ability = None
        self.mega_used[mon.side] = True
        stone = to_id(event[4]) if len(event) > 4 else ""
        if stone:
            mon.item, mon.item_state = stone, ITEM_KNOWN

    # --- HP, status, boosts ---------------------------------------------------

    def _set_condition(self, mon: _Mon, condition: str) -> None:
        parsed = parse_condition(condition)
        if parsed is None:
            self.counters["condition_unparsed"] += 1
            return
        fraction, status, fainted = parsed
        mon.hp = fraction
        mon.status = status
        # A living condition also un-faints: a revived Pokemon is healed on the
        # bench before it re-enters.
        mon.fainted = fainted

    def _on_hp(self, event: Sequence[str]) -> None:
        mon = self._resolve(event[2])
        if mon is None:
            self.counters["hp_unresolved"] += 1
            return
        self._set_condition(mon, event[3])

    def _on_status(self, event: Sequence[str]) -> None:
        mon = self._resolve(event[2])
        if mon is not None:
            mon.status = to_id(event[3]) or None

    def _on_curestatus(self, event: Sequence[str]) -> None:
        mon = self._resolve(event[2])
        if mon is not None:
            mon.status = None

    def _on_cureteam(self, event: Sequence[str]) -> None:
        ident = parse_ident(event[2])
        if ident is not None:
            for mon in self._roster.get(ident.side, ()):
                mon.status = None

    def _on_boost(self, event: Sequence[str]) -> None:
        mon = self._resolve(event[2])
        stat = event[3]
        if mon is None or stat not in mon.boosts:
            return
        amount = int(event[4]) * (1 if event[1] == "-boost" else -1)
        mon.boosts[stat] = max(-6, min(6, mon.boosts[stat] + amount))

    def _on_setboost(self, event: Sequence[str]) -> None:
        mon = self._resolve(event[2])
        if mon is not None and event[3] in mon.boosts:
            mon.boosts[event[3]] = max(-6, min(6, int(event[4])))

    def _on_clearboost(self, event: Sequence[str]) -> None:
        mon = self._resolve(event[2])
        if mon is None:
            return
        for stat, value in mon.boosts.items():
            if (
                event[1] == "-clearboost"
                or (event[1] == "-clearnegativeboost" and value < 0)
                or (event[1] == "-clearpositiveboost" and value > 0)
            ):
                mon.boosts[stat] = 0

    def _on_clearallboost(self, event: Sequence[str]) -> None:
        for key in SLOT_KEYS:
            mon = self._in_slot(key)
            if mon is not None:
                mon.boosts = dict.fromkeys(BOOST_STATS, 0)

    def _on_invertboost(self, event: Sequence[str]) -> None:
        mon = self._resolve(event[2])
        if mon is not None:
            mon.boosts = {stat: -value for stat, value in mon.boosts.items()}

    def _on_swapboost(self, event: Sequence[str]) -> None:
        first, second = self._resolve(event[2]), self._resolve(event[3])
        if first is None or second is None:
            return
        listed = event[4] if len(event) > 4 and not event[4].startswith("[") else ""
        stats = [stat.strip() for stat in listed.split(",") if stat.strip()]
        for stat in stats or BOOST_STATS:
            if stat in first.boosts:
                first.boosts[stat], second.boosts[stat] = (
                    second.boosts[stat],
                    first.boosts[stat],
                )

    def _on_copyboost(self, event: Sequence[str]) -> None:
        # |-copyboost|USER|FROM: the user takes a copy of the other's stages.
        user, source = self._resolve(event[2]), self._resolve(event[3])
        if user is not None and source is not None:
            user.boosts = dict(source.boosts)

    # --- volatiles, moves -----------------------------------------------------

    def _on_start(self, event: Sequence[str]) -> None:
        mon = self._resolve(event[2])
        if mon is None:
            return
        effect = effect_id(event[3])
        # Protocol keyword: |-start|POKEMON|typechange|TYPE/TYPE.
        if effect == "typechange":
            names = event[4] if len(event) > 4 else ""
            if names.startswith("["):
                source = self._resolve(tag_value(event, "[of]", 4) or "")
                if source is not None:
                    mon.types_override = self._types(source)
                return
            mon.types_override = tuple(
                to_id(name) for name in names.split("/") if to_id(name)
            )
            return
        counted = _COUNTED_EFFECT.match(effect)
        if counted is not None:
            family = counted.group(1)
            for other in [key for key in mon.volatiles if key.startswith(family)]:
                tail = other[len(family) :]
                if tail.isdigit():
                    del mon.volatiles[other]
        mon.volatiles.setdefault(effect, self.turn)

    def _on_end(self, event: Sequence[str]) -> None:
        mon = self._resolve(event[2])
        if mon is None:
            return
        effect = effect_id(event[3])
        if effect == "typechange":
            mon.types_override = None
        mon.volatiles.pop(effect, None)

    def _on_marker(self, event: Sequence[str]) -> None:
        # -prepare (a charge turn) and -mustrecharge announce a forced next action.
        mon = self._resolve(event[2])
        if mon is not None:
            mon.volatiles.setdefault(event[1].lstrip("-"), self.turn)

    def _clear_markers(self, mon: _Mon) -> None:
        mon.volatiles.pop("prepare", None)
        mon.volatiles.pop("mustrecharge", None)

    def _on_anim(self, event: Sequence[str]) -> None:
        mon = self._resolve(event[2])
        if mon is not None:
            mon.volatiles.pop("prepare", None)

    def _reveal_move(self, mon: _Mon, move_id: str) -> None:
        entry = moves_dex().get(move_id)
        if entry is not None and (
            entry.get("struggleRecoil") or entry.get("isZ") or entry.get("isMax")
        ):
            return
        if move_id and move_id not in mon.revealed_moves:
            mon.revealed_moves.append(move_id)

    def _on_move(self, event: Sequence[str]) -> None:
        ident = parse_ident(event[2])
        if ident is not None:
            self._last_mover_side = ident.side
        mon = self._resolve(event[2])
        if mon is None:
            self.counters["move_unresolved"] += 1
            return
        move_id = to_id(event[3])
        origin = tag_value(event, "[from]", 5)
        self._clear_markers(mon)
        # A called move ([from] another effect) is not one of the user's own.
        if (
            origin is None
            or to_id(origin) == "lockedmove"
            or effect_id(origin) == move_id
        ):
            self._reveal_move(mon, move_id)

    def _on_cant(self, event: Sequence[str]) -> None:
        mon = self._resolve(event[2])
        if mon is None:
            return
        reason = event[3] if len(event) > 3 else ""
        if reason.lower().startswith("ability:") and tag_value(event, "[of]"):
            return  # the subject is the ability holder, not the Pokemon stopped
        self._clear_markers(mon)
        if len(event) > 4 and event[4] and not event[4].startswith("["):
            self._reveal_move(mon, to_id(event[4]))

    def _on_singleturn(self, event: Sequence[str]) -> None:
        mon = self._resolve(event[2])
        if mon is None:
            return
        effect = effect_id(event[3])
        # Side guards raise the same counter as the Protect family without
        # being one: the next Protect succeeds one time in three.
        if not raises_stall_counter(effect):
            return
        mon.stall_streak = (
            mon.stall_streak + 1 if mon.stall_turn == self.turn - 1 else 1
        )
        mon.stall_turn = self.turn
        if is_protect_family(effect):
            mon.protect_turn = self.turn

    # --- items, abilities -----------------------------------------------------

    def _reveal_item(self, mon: _Mon | None, item: str) -> None:
        if mon is None or not item:
            return
        if mon.item_state == ITEM_CONSUMED and mon.item == item:
            return
        mon.item, mon.item_state = item, ITEM_KNOWN

    def _reveal_ability(self, mon: _Mon | None, ability: str) -> None:
        """``ability`` is what ``mon`` has right now."""
        if mon is None or not ability:
            return
        if mon.field_ability is not None:
            mon.field_ability = ability
        else:
            mon.base_ability = ability

    def _ability_now(self, mon: _Mon) -> str | None:
        """The ability in effect while ``mon`` is on the field, if known."""
        if mon.field_ability is not None:
            return mon.field_ability or None
        if mon.base_ability is not None or mon.is_mega:
            return mon.base_ability
        sheet = self.sheets.get(mon.side, {}).get(mon.species)
        return None if sheet is None else sheet.ability

    def _may_hold(self, mon: _Mon, ability: str) -> bool | None:
        """Whether ``mon`` can have ``ability`` right now; None = cannot tell."""
        if mon.field_ability is not None:
            return mon.field_ability == ability if mon.field_ability else None
        if mon.base_ability == ability:
            return True
        listed = _forme_abilities(mon.forme)
        if mon.is_mega:
            # The dex is out of date for some Mega formes of this format: a
            # listed ability counts, a missing one rules nothing out.
            return True if ability in listed else None
        listed = listed | _forme_abilities(mon.species)
        return ability in listed if listed else None

    def _tag_holder(
        self, event: Sequence[str], kind: str, other: str | None, ability: str
    ) -> _Mon | None:
        """Who holds the ability a ``[from] ability:`` tag names."""
        subject = None if kind in _NO_SUBJECT_KINDS else self._resolve(event[2])
        source = self._resolve(other) if other else None
        if other and source is None and kind in _OF_HOLDS_ABILITY:
            return None  # the likely holder is named but cannot be found
        if subject is None or source is None or source is subject:
            return subject if subject is not None else source
        ranks = {True: 2, None: 1, False: 0}
        first = ranks[self._may_hold(subject, ability)]
        second = ranks[self._may_hold(source, ability)]
        if first != second:
            return subject if first > second else source
        if kind == "-heal":
            # A heal given to a partner names the giver in [of]; a heal a
            # Pokemon draws from a foe's move names the attacker there.
            return source if source.side == subject.side else subject
        return source if kind in _OF_HOLDS_ABILITY else subject

    def _on_ability(self, event: Sequence[str]) -> None:
        mon = self._resolve(event[2])
        ability = to_id(event[3])
        if mon is None or not ability:
            return
        before = event[4] if len(event) > 4 and not event[4].startswith("[") else ""
        if not before or tag_value(event, "[from]", 5) is None:
            self._reveal_ability(mon, ability)
            return
        # |-ability|POKEMON|NEW|OLD|[from] EFFECT|[of] SOURCE: the ability was
        # replaced (pokemon-showdown/sim/pokemon.ts setAbility). OLD is what it
        # had; NEW lasts while it stays in.
        if mon.field_ability is None:
            mon.base_ability = to_id(before) or mon.base_ability
        mon.field_ability = ability
        source = self._resolve(tag_value(event, "[of]", 5) or "")
        if source is not None and source is not mon:
            # Every replacement that names a source copies that Pokemon's ability.
            self._reveal_ability(source, ability)

    def _on_activate(self, event: Sequence[str]) -> None:
        # |-activate|USER|MOVE|ABILITY|ABILITY|[of] TARGET: two Pokemon exchanged
        # abilities; USER now has the first, TARGET the second, and the names are
        # blank between partners (pokemon-showdown/data/moves.ts skillswap). No
        # other move line has this shape.
        if (
            len(event) < 7
            or not event[6].startswith("[of]")
            or event[4].startswith("[")
            or event[5].startswith("[")
            or move_entry(effect_id(event[3])) is None
        ):
            return
        user = self._resolve(event[2])
        target = self._resolve(tag_value(event, "[of]", 6) or "")
        if user is None or target is None or user is target:
            return
        taken, given = to_id(event[4]), to_id(event[5])
        had_user, had_target = self._ability_now(user), self._ability_now(target)
        if given and user.field_ability is None:
            user.base_ability = given
        if taken and target.field_ability is None:
            target.base_ability = taken
        user.field_ability = taken or had_target or ""
        target.field_ability = given or had_user or ""

    def _on_item(self, event: Sequence[str]) -> None:
        self._reveal_item(self._resolve(event[2]), to_id(event[3]))

    def _on_enditem(self, event: Sequence[str]) -> None:
        mon = self._resolve(event[2])
        if mon is not None:
            mon.item, mon.item_state = to_id(event[3]) or mon.item, ITEM_CONSUMED

    def _reveal_from_tags(self, event: Sequence[str], kind: str) -> None:
        origin = tag_value(event, "[from]", 3)
        if origin is not None:
            lowered = origin.lower()
            other = tag_value(event, "[of]", 3)
            if lowered.startswith("item:"):
                holder = other if kind == "-damage" and other else event[2]
                self._reveal_item(self._resolve(holder), effect_id(origin))
            elif lowered.startswith("ability:") and kind not in _ABILITY_TAG_UNREAD:
                ability = effect_id(origin)
                self._reveal_ability(
                    self._tag_holder(event, kind, other, ability), ability
                )
        if kind in _EFFECT_FIELD_KINDS:
            lowered = event[3].lower()
            if lowered.startswith("ability:"):
                other = tag_value(event, "[of]", 4)
                holder = other if kind == "-block" and other else event[2]
                self._reveal_ability(self._resolve(holder), effect_id(event[3]))
            elif lowered.startswith("item:"):
                self._reveal_item(self._resolve(event[2]), effect_id(event[3]))

    # --- weather, fields, side conditions ------------------------------------

    def _on_weather(self, event: Sequence[str]) -> None:
        weather = to_id(event[2])
        if weather == "none" or not weather:
            self.weather = self.weather_start = self.weather_side = None
            return
        if "[upkeep]" in event[3:]:
            if self.weather is None:
                self.weather, self.weather_start = weather, self._now()
            return
        holder = parse_ident(tag_value(event, "[of]", 3) or "")
        self.weather = weather
        self.weather_start = self._now()
        self.weather_side = holder.side if holder else self._last_mover_side

    def _on_fieldstart(self, event: Sequence[str]) -> None:
        effect = effect_id(event[2])
        entry = move_entry(effect) or {}
        if entry.get("terrain"):
            for other in [key for key in self.fields if key != effect]:
                if (move_entry(other) or {}).get("terrain"):
                    del self.fields[other]
        self.fields[effect] = self._now()

    def _on_fieldend(self, event: Sequence[str]) -> None:
        self.fields.pop(effect_id(event[2]), None)

    def _on_sidestart(self, event: Sequence[str]) -> None:
        side = event[2][:2]
        if side in self.side_conditions:
            self.side_conditions[side].setdefault(effect_id(event[3]), self._now())

    def _on_sideend(self, event: Sequence[str]) -> None:
        side = event[2][:2]
        if side in self.side_conditions:
            self.side_conditions[side].pop(effect_id(event[3]), None)

    def _on_swapsideconditions(self, event: Sequence[str]) -> None:
        first, second = SIDES
        self.side_conditions[first], self.side_conditions[second] = (
            self.side_conditions[second],
            self.side_conditions[first],
        )

    # --- turns ----------------------------------------------------------------

    def _on_upkeep(self, event: Sequence[str]) -> None:
        self.after_upkeep = True

    def _on_win(self, event: Sequence[str]) -> None:
        self.finished = True
        name = event[2] if len(event) > 2 else ""
        self.winner = next(
            (side for side, player in self.players.items() if player == name), None
        )
        if self.winner is None:
            self.counters["winner_not_a_player"] += 1
        if self.segments:
            self.segments[-1].ended = True

    def _on_tie(self, event: Sequence[str]) -> None:
        self.finished = self.tie = True
        if self.segments:
            self.segments[-1].ended = True

    def _on_turn(self, event: Sequence[str]) -> None:
        # There is no turn start to serve until the new snapshot exists: a
        # failure below must read as "no forecast", never as the previous
        # turn's snapshot.
        self.turn_snapshot = None
        self.at_turn_start = False
        number = int(event[2])
        if number <= self.turn:
            self.counters["turn_not_increasing"] += 1
        if self.segments:
            self._record_last_actions(self.segments[-1])
        self.turn = number
        self.after_upkeep = False
        for key in SLOT_KEYS:
            mon = self._in_slot(key)
            occupant = self.tracker.active.get(key)
            if mon is not None and occupant is not None and not occupant.fainted:
                mon.active_turns += 1
        self.segments.append(TurnSegment(number, self.tracker.alive()))
        self.turn_snapshot = self.snapshot()
        self.at_turn_start = True

    def _record_last_actions(self, previous: TurnSegment) -> None:
        # The reader never raises; a failure shows only in its counter. Its
        # other counts stay out: drive_events reads every turn again for the
        # labels and counts the unusual lines there.
        reader: Counter[str] = Counter()
        self.last_actions = read_turn_actions(previous, None, reader)
        for name, count in reader.items():
            if name.startswith("reader_error:"):
                self.counters["last_action_error:" + name.split(":", 1)[1]] += count
        for action in self.last_actions.values():
            for mon in self._roster.get(action.side, ()):
                if mon.species == action.species:
                    mon.last_action, mon.last_action_turn = action, previous.turn
                    break

    # --- snapshot -------------------------------------------------------------

    def _all_mons(self) -> list[_Mon]:
        return [mon for side in SIDES for mon in self._roster[side]]

    def _types(self, mon: _Mon) -> tuple[str, ...]:
        if mon.types_override is not None:
            return mon.types_override
        entry = pokedex().get(mon.forme) or pokedex().get(mon.species) or {}
        return tuple(to_id(str(name)) for name in entry.get("types", ()))

    def _mon_public(self, mon: _Mon, slot: str | None, all_seen: bool) -> MonPublic:
        sheet = self.sheets.get(mon.side, {}).get(mon.species)
        item, item_state = mon.item, mon.item_state
        if item_state == ITEM_UNKNOWN and sheet is not None and sheet.item:
            item, item_state = sheet.item, ITEM_KNOWN
        on_field = slot is not None
        if on_field and mon.field_ability is not None:
            ability = mon.field_ability or None
        else:
            ability = mon.base_ability
            if ability is None and sheet is not None and not mon.is_mega:
                ability = sheet.ability
        moves: list[MovePublic] = []
        if sheet is not None:
            moves = [
                MovePublic(move, move in mon.revealed_moves, True)
                for move in sheet.moves
            ]
        listed = {move.id for move in moves}
        moves.extend(
            MovePublic(move, True, False)
            for move in mon.revealed_moves
            if move not in listed
        )
        stones = _mega_stones(mon.species)
        mega_possible = (
            bool(stones)
            and not self.mega_used[mon.side]
            and not mon.fainted
            and not mon.is_mega
            and (
                item_state == ITEM_UNKNOWN
                or (item_state == ITEM_KNOWN and item in stones)
            )
        )
        last_turn = self.turn - 1
        return MonPublic(
            side=mon.side,
            index=mon.index,
            species=mon.species,
            forme=mon.forme,
            types=self._types(mon),
            hp=mon.hp if mon.revealed else 1.0,
            status=mon.status,
            boosts=dict(mon.boosts),
            slot=slot,
            revealed=mon.revealed,
            fainted=mon.fainted,
            brought=True if mon.revealed else (False if all_seen else None),
            is_mega=mon.is_mega,
            mega_possible=mega_possible,
            first_turn=on_field and mon.active_turns == 1,
            protected_last_turn=on_field and mon.protect_turn == last_turn,
            protect_streak=(
                mon.stall_streak if on_field and mon.stall_turn == last_turn else 0
            ),
            turns_on_field=max(0, mon.active_turns - 1) if on_field else 0,
            item=item,
            item_state=item_state,
            ability=ability,
            ability_known=ability is not None,
            moves=tuple(moves),
            volatiles=dict(sorted(mon.volatiles.items())) if on_field else {},
            last_action=(
                mon.last_action if mon.last_action_turn == last_turn else None
            ),
        )

    def _side_public(self, side: str) -> SidePublic:
        roster = self._roster[side]
        slots: dict[int, str] = {}
        active: dict[str, int | None] = {"a": None, "b": None}
        for letter in active:
            occupant = self.tracker.active.get(side + letter)
            mon = self._in_slot(side + letter)
            if occupant is not None and mon is not None and not occupant.fainted:
                slots[mon.index] = letter
                active[letter] = mon.index
        n_revealed = sum(mon.revealed for mon in roster)
        size = self.team_size[side]
        all_seen = size is not None and n_revealed >= size
        return SidePublic(
            side=side,
            player=self.players[side],
            rating=self.ratings[side],
            sheet_open=True if side in self.sheets else self._closed(),
            mega_used=self.mega_used[side],
            team_size=size,
            n_fainted=sum(mon.fainted for mon in roster),
            n_revealed=n_revealed,
            n_unknown_brought=None if size is None else max(0, size - n_revealed),
            conditions=dict(sorted(self.side_conditions[side].items())),
            active=active,
            mons=tuple(
                self._mon_public(mon, slots.get(mon.index), all_seen) for mon in roster
            ),
        )

    def snapshot(self) -> PublicSnapshot:
        """The public state right now (see ``turn_snapshot`` for the turn start)."""
        return PublicSnapshot(
            battle_id=self.battle_id,
            turn=self.turn,
            format_id=self.format_id,
            best_of=self.best_of,
            rated=self.rated,
            weather=self.weather,
            weather_start=self.weather_start,
            weather_side=self.weather_side,
            fields=dict(sorted(self.fields.items())),
            sides={side: self._side_public(side) for side in SIDES},
        )

    def _closed(self) -> bool | None:
        """``sheet_open`` of a side with no sheet: False when that is known."""
        return False if self.sheets_known else None

    def skip_reason(self) -> str | None:
        """Why this battle should not become training data, or None."""
        if not dex_available():
            return SKIP_DEX_MISSING
        if self.gametype is not None and self.gametype != "doubles":
            return SKIP_NOT_DOUBLES
        if self.false_identity or any(
            _has_false_identity(mon.forme) or _has_false_identity(mon.species)
            for mon in self._all_mons()
        ):
            return SKIP_ILLUSION
        if not self.segments:
            return SKIP_NO_TURN
        # A repeated or backward turn line: the turn counts and the per-turn
        # records after it cannot be trusted.
        if self.counters.get("turn_not_increasing"):
            return SKIP_TURN_ORDER
        return None


@dataclass(frozen=True)
class TurnRecord:
    """One turn of a driven log: the state before it and what each slot then did."""

    turn: int
    snapshot: PublicSnapshot
    actions: dict[str, SlotAction]


@dataclass
class DriveResult:
    """Everything ``drive_log`` extracts from one battle log.

    ``skip_reason`` names why the battle must not be used (``no_turn_1``,
    ``illusion_species``, ``not_doubles``, ``turn_not_increasing``,
    ``dex_missing``) and is also counted in ``counters`` as ``skip:<reason>``;
    ``turns`` is still filled with whatever was parsed. ``winner`` is ``p1`` /
    ``p2`` / None (tie, unfinished, or unknown name).

    ``players`` holds DISPLAY names, which one account can write in several
    ways; ``player_ids`` holds the account ids, the only thing to hash, cap or
    match players by. ``sheets`` is True / False / None per side (None: a
    player-view stream whose sheet state nobody gave). ``rated_kind`` is
    ``ladder``, the id of the server's note for any other rated game, or None;
    ``best_of`` tells a series game from a single one whatever file or format
    id it came under. ``start_time`` / ``end_time`` are the first and last
    ``|t:|`` stamps (Unix seconds), None on player-view streams, which never
    carry them.
    """

    battle_id: str
    format_id: str | None
    players: dict[str, str | None]
    ratings: dict[str, int | None]
    sheets: dict[str, bool | None]
    winner: str | None
    tie: bool
    finished: bool
    best_of: int | None
    series_id: str | None
    game_number: int | None
    turns: list[TurnRecord]
    skip_reason: str | None
    counters: Counter[str]
    player_ids: dict[str, str | None] = field(default_factory=dict)
    rated_kind: str | None = None
    start_time: int | None = None
    end_time: int | None = None

    @property
    def usable(self) -> bool:
        return self.skip_reason is None


def drive_events(
    events: Iterable[Sequence[str]],
    battle_id: str = "",
    *,
    sheets: Mapping[str, Iterable[SheetSet]] | None = None,
    sheets_known: bool = True,
) -> DriveResult:
    """Feed a whole event list through a ``PublicBattle``; never raises.

    ``sheets`` are open team sheets recovered outside the stream, given before
    the first event as ``LiveShadow.feed_sheet`` does. ``sheets_known=False``
    is for a player-view stream (a saved own replay page) whose sheet state
    nobody recorded: sides without a sheet then read ``sheet_open=None``.
    """
    public = PublicBattle(battle_id, sheets_known)
    snapshots: list[PublicSnapshot | None] = []
    counters: Counter[str] = Counter()
    turns: list[TurnRecord] = []
    try:
        for side, sets in (sheets or {}).items():
            public.feed_sheet(side, sets)
        for event in events:
            before = len(public.segments)
            public.feed(event)
            if len(public.segments) > before:
                snapshots.append(public.turn_snapshot)
        segments = public.segments
        for index, segment in enumerate(segments):
            # Paired by position in the turn list, never by a turn number that
            # a broken log may repeat. A turn whose snapshot failed is dropped.
            snapshot = snapshots[index] if index < len(snapshots) else None
            if snapshot is None:
                counters["turn_without_snapshot"] += 1
                continue
            following = segments[index + 1] if index + 1 < len(segments) else None
            actions = read_turn_actions(segment, following, counters)
            turns.append(TurnRecord(segment.turn, snapshot, actions))
    except Exception as exc:
        counters[f"drive_error:{type(exc).__name__}"] += 1
    counters.update(public.counters)
    reason = public.skip_reason()
    if reason is not None:
        counters[f"skip:{reason}"] += 1
    return DriveResult(
        battle_id=battle_id,
        format_id=public.format_id,
        players=dict(public.players),
        ratings=dict(public.ratings),
        sheets={
            side: True if side in public.sheets else public._closed() for side in SIDES
        },
        winner=public.winner,
        tie=public.tie,
        finished=public.finished,
        best_of=public.best_of,
        series_id=public.series_id,
        game_number=public.game_number,
        turns=turns,
        skip_reason=reason,
        counters=counters,
        player_ids=dict(public.player_ids),
        rated_kind=public.rated_kind,
        start_time=public.start_time,
        end_time=public.end_time,
    )


def drive_log(
    log_text: str,
    battle_id: str = "",
    *,
    sheets: Mapping[str, Iterable[SheetSet]] | None = None,
    sheets_known: bool = True,
) -> DriveResult:
    """Drive one raw log (spectator view, or HP-rewritten player view).

    See ``drive_events`` for ``sheets`` and ``sheets_known``.
    """
    try:
        events = split_log(log_text)
    except Exception as exc:
        result = drive_events([], battle_id, sheets=sheets, sheets_known=sheets_known)
        result.counters[f"drive_error:{type(exc).__name__}"] += 1
        return result
    return drive_events(events, battle_id, sheets=sheets, sheets_known=sheets_known)


def _is_event(event: Any) -> bool:
    return (
        isinstance(event, (list, tuple))
        and len(event) >= 2
        and all(isinstance(part, str) for part in event)
    )


class LiveShadow:
    """Serve-time twin of ``drive_log``: one per live battle.

    ``sync(battle)`` duck-types the battle (``_replay_data``: list of split
    events, ``player_role``: ``p1`` / ``p2``, ``battle_tag``), feeds the events
    that arrived since the last call, with the bot's exact HP rewritten to the
    public percent, and returns the ``PublicSnapshot`` of the turn that is about
    to be played. It returns None, with a named count in ``counters``, when
    there is nothing to forecast (no turn yet, or the stream is past the turn
    line: a forced-switch request) and after any error. It never raises. An
    entry of the stream that is not a list of strings is skipped and counted
    (``sync_skip:not_an_event``); the cursor always moves past what was read.

    The stream never carries ``|showteam|``. Until the caller gives the sheets
    (``feed_sheet``) or says that every open sheet has been given
    (``mark_sheets_known``), a side's ``sheet_open`` is None. Both are kept
    across rebuilds of the same battle and dropped when the battle tag changes.

    A reconnect replays the log. When the stream is shorter than the cursor, the
    battle tag or the role changed, a new preamble starts, or a turn number goes
    backwards, the state is rebuilt from the start of the (replayed) stream.
    After a turn number went backwards WITHOUT a preamble the state has no
    roster and no ratings: the shadow stands down (``stand_down:degraded``)
    until a preamble arrives.
    """

    _PREAMBLE = ("init", "gametype")

    def __init__(self) -> None:
        self.counters: Counter[str] = Counter()
        self._tag: str | None = None
        self._role: str | None = None
        self._sheets: dict[str, list[SheetSet]] = {}
        self._sheets_known = False
        self._public = PublicBattle(sheets_known=False)
        self._cursor = 0
        self._base = 0
        self._degraded = False

    @property
    def public(self) -> PublicBattle:
        return self._public

    def feed_sheet(self, side: str, sets: Iterable[SheetSet]) -> None:
        """Give an open team sheet (kept across rebuilds). Never raises.

        Given before the turn's events it is part of that turn's snapshot.
        """
        try:
            self._sheets[side] = list(sets)
            self._public.feed_sheet(side, self._sheets[side])
        except Exception as exc:
            self.counters[f"sync_error:sheet:{type(exc).__name__}"] += 1

    def mark_sheets_known(self) -> None:
        """Say that every open sheet has been given: any other side is closed."""
        try:
            self._sheets_known = True
            self._public.set_sheets_known(True)
        except Exception as exc:
            self.counters[f"sync_error:sheet:{type(exc).__name__}"] += 1

    def _rebuild(self, base: int, why: str) -> None:
        self.counters[f"rebuild:{why}"] += 1
        self._public = PublicBattle(self._tag or "", self._sheets_known)
        for side, sets in self._sheets.items():
            self._public.feed_sheet(side, sets)
        self._base = self._cursor = base
        self._degraded = False

    def _restart_at(self, stream: Sequence[Any], index: int, kind: str) -> int | None:
        """Start of a replayed log if ``stream[index]`` shows one, else None."""
        public = self._public
        if kind == "init" and public.events_fed > 0:
            return index
        if kind == "gametype" and (public.gametype is not None or self._degraded):
            return index
        if kind == "turn" and public.turn > 0:
            try:
                number = int(stream[index][2])
            except (IndexError, TypeError, ValueError):
                return None
            if number <= public.turn:
                return index
        return None

    def sync(self, battle: Any) -> PublicSnapshot | None:
        try:
            return self._sync(battle)
        except Exception as exc:
            self.counters[f"sync_error:{type(exc).__name__}"] += 1
            return None

    def _sync(self, battle: Any) -> PublicSnapshot | None:
        stream = getattr(battle, "_replay_data", None)
        if not isinstance(stream, list):
            self.counters["stand_down:no_stream"] += 1
            return None
        if not dex_available():
            self.counters["stand_down:dex_missing"] += 1
            return None
        tag = getattr(battle, "battle_tag", None)
        role = getattr(battle, "player_role", None)
        role = role if role in SIDES else None
        if tag != self._tag:
            first = self._tag is None and self._cursor == 0
            self._tag = tag
            if first:
                self._public.battle_id = tag or ""
            else:
                # Another battle: nothing said about the previous one applies.
                self._sheets = {}
                self._sheets_known = False
                self._rebuild(0, "battle_tag_changed")
        if len(stream) < self._cursor:
            self._rebuild(0, "stream_shorter_than_cursor")
        if role is not None and self._role is not None and role != self._role:
            self._role = role
            self._rebuild(0, "role_changed")
        if role is not None:
            self._role = role
        index = self._cursor
        restarts = 0
        while index < len(stream):
            event = stream[index]
            if not _is_event(event):
                self.counters["sync_skip:not_an_event"] += 1
                index += 1
                self._cursor = index
                continue
            kind = event[1]
            start = self._restart_at(stream, index, kind)
            if start is not None:
                restarts += 1
                if restarts > 8:
                    self.counters["sync_error:restart_loop"] += 1
                    self._cursor = len(stream)
                    return None
                self._rebuild(start, "replayed_log")
                if kind == "turn":
                    self.counters["rebuild_without_preamble"] += 1
                    self._degraded = True
                    self._public.feed(event)
                    index = start + 1
                    self._cursor = index
                continue
            if kind in CONDITION_INDEX and self._role is None:
                self.counters["stand_down:role_unknown"] += 1
                self._cursor = index
                return None
            try:
                self._public.feed(
                    rewrite_event(event, self._role)
                    if self._role is not None
                    else event
                )
            except Exception as exc:
                self.counters[f"sync_error:event:{type(exc).__name__}"] += 1
            # Moved past every event read, so a failing one is never fed twice.
            index += 1
            self._cursor = index
        if self._degraded:
            self.counters["stand_down:degraded"] += 1
            return None
        if self._public.turn_snapshot is None:
            unread = self._public.turn > 0 or bool(self._public.segments)
            name = "turn_unreadable" if unread else "no_turn_yet"
            self.counters[f"stand_down:{name}"] += 1
            return None
        if not self._public.at_turn_start:
            self.counters["stand_down:mid_turn"] += 1
            return None
        return self._public.turn_snapshot
