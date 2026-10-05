"""Pure-string reading of Showdown doubles battle logs: turns and per-slot actions.

This module is the label reader of the opponent predictor (design:
OPPONENT_PREDICTOR.md, "Labels"). It imports nothing from poke-env; move and
species properties come from poke-env's static JSON files, read from disk.

Typical use::

    events = split_log(log_text)                 # [['', 'move', 'p1a: X', ...], ...]
    segments = segment_turns(events)             # one TurnSegment per |turn|N
    for seg, nxt in zip(segments, segments[1:] + [None]):
        actions = read_turn_actions(seg, nxt)    # {'p1a': SlotAction, ...}

``read_turn_actions`` returns one ``SlotAction`` for every slot whose Pokemon was
alive when the turn started, for BOTH sides, keyed by the slot it stood in at that
moment (``p1a`` / ``p1b`` / ``p2a`` / ``p2b``). Reading rules, each verified on the
Reg M-C corpus or in the simulator source:

* A voluntary switch is a ``switch`` line for such a slot before the first
  ``move`` / ``cant`` / ``-mega`` line of the turn (the simulator runs every switch
  before any Mega and any move). ``drag``, pivots (``[from]`` tag, or any switch
  after the first move), ejections and end-of-turn replacements are not choices.
* The move is the first ``move`` line of that Pokemon with no ``[from]`` tag. A
  ``[from]`` naming the move itself (Round's early partner) is a real choice;
  ``[from] lockedmove`` is a forced continuation (``locked``); every other
  ``[from]`` is a called move. Later untagged lines are repeats. A repeat that a
  partner inserts BEFORE the Pokemon's own action (a single-turn line naming a
  move that leaves no effect, with ``[of]``) is skipped the same way.
* An effect that replaces its target's queued action (dex ``onOverrideAction``)
  and starts before the target has acted hides the click: the move line that
  follows is the forced move (``hidden`` / ``overridden``, ``forced_move``).
* ``cant`` with ``[of]`` and an ``ability:`` reason names the ability HOLDER as
  subject; the blocked attacker is in ``[of]`` and already has its own move line.
* A confusion self-hit has no ``cant`` and no ``move`` line, only a
  ``[from] confusion`` damage line.
* The target is where the move LANDED. It is ``foe_a`` / ``foe_b`` / ``ally`` /
  ``self`` only for moves whose dex target type lets the player choose; every
  other move is ``auto``. A ``[spread]`` tag on an aimed move also gives
  ``auto``: the simulator turned the move into a spread hit and replaced the
  click by a random target. A spread move that hit one Pokemon carries no tag,
  so the tag is never needed to call a spread move ``auto``. A blank ``[still]``
  target is back-filled from the following ``-anim`` line or the next turn's
  locked-move line, else it stays ``None`` (unknown).
* ``target_trusted`` is False when the move landed on a Pokemon that had a
  redirection effect up, when an ability of the Pokemon it landed on activated
  directly after the move line (how the simulator logs a redirecting ability,
  on either side), when a Pokemon on the target's side had already fainted that
  turn (auto-retarget), or when only one foe stood at the start of the turn
  (``target_forced``: the simulator sends every foe-aimed move there).
* Actions belong to the Pokemon in the slot at turn start: ``swap`` (Ally Switch)
  is followed, ``replace`` (Illusion) keeps the slot, and everything is keyed by
  slot letter, never by nickname.

Censoring is described by ``kind`` / ``reason`` / ``switch_known`` /
``protect_known`` so a trainer can mask instead of guessing; see ``SlotAction``.
"""

from __future__ import annotations

import importlib.util
import json
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

Event = list[str]

SIDES: tuple[str, str] = ("p1", "p2")
SLOT_LETTERS: tuple[str, str] = ("a", "b")
SLOT_KEYS: tuple[str, ...] = ("p1a", "p1b", "p2a", "p2b")

KIND_MOVE = "move"
KIND_SWITCH = "switch"
KIND_HIDDEN = "hidden"
KIND_NONE = "none"

TARGET_FOE_A = "foe_a"
TARGET_FOE_B = "foe_b"
TARGET_ALLY = "ally"
TARGET_SELF = "self"
TARGET_AUTO = "auto"
TARGET_CLASSES: tuple[str, ...] = (
    TARGET_FOE_A,
    TARGET_FOE_B,
    TARGET_ALLY,
    TARGET_SELF,
    TARGET_AUTO,
)

INTENT_SWITCH = "switch"
INTENT_PROTECT = "protect"
INTENT_ATTACK_FOE_A = "attack_foe_a"
INTENT_ATTACK_FOE_B = "attack_foe_b"
INTENT_SPREAD = "spread_attack"
INTENT_STATUS_FOE = "status_foe"
INTENT_SUPPORT = "support"
INTENT_CLASSES: tuple[str, ...] = (
    INTENT_SWITCH,
    INTENT_PROTECT,
    INTENT_ATTACK_FOE_A,
    INTENT_ATTACK_FOE_B,
    INTENT_SPREAD,
    INTENT_STATUS_FOE,
    INTENT_SUPPORT,
)

REASON_FAINTED_FIRST = "fainted_first"
REASON_LEFT_FIELD = "left_field"
REASON_GAME_ENDED = "game_ended"
REASON_UNRESOLVED = "unresolved"
REASON_UNEXPLAINED = "unexplained"
REASON_FORCED = "forced"
REASON_CONFUSION = "confusion"
REASON_FLINCH = "flinch"
REASON_OVERRIDDEN = "overridden"

# Message types that carry no battle state: chat, timers, room bookkeeping. A
# stream that ends with a |turn| line followed only by these is still "at the
# start of the turn".
CHATTER_KINDS: frozenset[str] = frozenset(
    {
        "",
        "-hint",
        "-message",
        ":",
        "B",
        "J",
        "L",
        "N",
        "askreg",
        "b",
        "badge",
        "bigerror",
        "c",
        "c:",
        "chat",
        "debug",
        "deinit",
        "error",
        "expire",
        "hidelines",
        "html",
        "inactive",
        "inactiveoff",
        "init",
        "j",
        "join",
        "l",
        "leave",
        "message",
        "n",
        "name",
        "notify",
        "player",
        "pm",
        "popup",
        "raw",
        "request",
        "seed",
        "sentchoice",
        "split",
        "t:",
        "tempnotify",
        "tempnotifyoff",
        "title",
        "uhtml",
        "uhtmlchange",
        "unlink",
    }
)

# Dex target types the player aims; everything else resolves without a choice.
_CHOOSABLE_TARGETS = frozenset(
    {"normal", "any", "adjacentFoe", "adjacentAlly", "adjacentAllyOrSelf"}
)
_SPREAD_TARGETS = frozenset({"allAdjacentFoes", "allAdjacent"})
_FOE_FACING_TARGETS = frozenset({"allAdjacentFoes", "allAdjacent", "foeSide"})
_ALLY_ONLY_TARGETS = frozenset({"adjacentAlly", "adjacentAllyOrSelf"})
_FOE_TARGET_CLASSES = frozenset({TARGET_FOE_A, TARGET_FOE_B})

_LOG_BLOCK = re.compile(
    r'<script type="text/plain" class="battle-log-data">(.*?)</script>', re.S
)
_IDENT = re.compile(r"^(p[1-4])([a-c])?(?::\s?(.*))?$", re.S)
_EFFECT_PREFIX = re.compile(r"^(move|ability|item):\s*", re.I)
_SIM_MOVE_KEY = re.compile(r"^\t([a-z0-9]+): \{$")
_SIM_FIRST_TURN_GATE = re.compile(r"if \(\w+\.activeMoveActions(?: > 1)?\)")
_NOT_USER_ID = re.compile(r"[^a-z0-9]+")
_REPO_ROOT = Path(__file__).resolve().parents[3]


def to_id(text: str) -> str:
    """Showdown id: lower-case alphanumerics only (same as poke-env's to_id_str)."""
    return "".join(ch for ch in text if ch.isalnum()).lower()


def user_id(name: str) -> str:
    """Showdown account id of a display name: ASCII letters and digits, lower-cased.

    The server identifies an account by this id, so ``Some_Name`` and
    ``some name`` are one player. Use it, never the display name, to split,
    cap or match players.
    """
    return _NOT_USER_ID.sub("", name.lower())


def effect_id(text: str) -> str:
    """Id of an effect field such as ``move: Follow Me`` or ``ability: Armor Tail``."""
    return to_id(_EFFECT_PREFIX.sub("", text.strip()))


def split_log(text: str) -> list[Event]:
    """Split a raw log into events, keeping only lines that start with ``|``.

    The result has the shape of poke-env's ``battle._replay_data``: each event is
    the line split on ``|``, so ``event[1]`` is the message type.
    """
    return [
        line.rstrip("\r").split("|") for line in text.split("\n") if line[:1] == "|"
    ]


def extract_log_from_html(html_text: str) -> str | None:
    """The log embedded in one of the bot's saved replay pages, or None.

    poke-env writes ``battle._replay_data`` verbatim (no HTML escaping) into a
    ``battle-log-data`` script block, so the text is returned as it stands: it is
    a PLAYER-view log and carries the bot's exact HP.
    """
    match = _LOG_BLOCK.search(html_text)
    if match is None:
        return None
    return match.group(1).strip()


def extract_battle_tag(log_text: str) -> str | None:
    """The ``>battle-...`` room line of a log, without the ``>``; None if absent."""
    for line in log_text.split("\n", 8)[:8]:
        line = line.strip()
        if line.startswith(">"):
            return line[1:].strip() or None
    return None


@dataclass(frozen=True)
class Ident:
    """A parsed Pokemon reference: ``p1a: Nick`` or ``p1: Nick`` (no slot)."""

    side: str
    slot: str | None
    name: str

    @property
    def key(self) -> str | None:
        """Slot key such as ``p1a``; None for a reference without a slot letter."""
        return None if self.slot is None else self.side + self.slot


@lru_cache(maxsize=8192)
def parse_ident(text: str) -> Ident | None:
    match = _IDENT.match(text)
    if match is None:
        return None
    return Ident(match.group(1), match.group(2), match.group(3) or "")


def tag_value(event: Sequence[str], tag: str, start: int = 3) -> str | None:
    """Value of a bracket tag (``[from]``, ``[of]``) in ``event[start:]``."""
    for part in event[start:]:
        if part.startswith(tag):
            return part[len(tag) :].strip()
    return None


# --- dex data ---------------------------------------------------------------


def _static_file(*parts: str) -> Path | None:
    try:
        spec = importlib.util.find_spec("poke_env")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)), "data", "static", *parts)


def _load_json(*parts: str) -> dict[str, Any]:
    path = _static_file(*parts)
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def moves_dex() -> dict[str, Any]:
    """Gen 9 move data by move id (the file ``GenData.from_gen(9).moves`` reads)."""
    return _load_json("moves", "gen9moves.json")


@lru_cache(maxsize=1)
def pokedex() -> dict[str, Any]:
    """Gen 9 species data by forme id; cosmetic formes alias their base entry."""
    dex = _load_json("pokedex", "gen9pokedex.json")
    # As poke-env does: a cosmetic forme resolves to its base entry, replacing
    # the stub the file carries for some of them (a stub has no types).
    aliases: dict[str, Any] = {}
    for entry in dex.values():
        for cosmetic in entry.get("cosmeticFormes", ()):
            aliases[to_id(cosmetic)] = entry
    dex.update(aliases)
    return dex


def dex_available() -> bool:
    """Whether both dex files were read.

    Without them every move is unknown: targets, Protect and intent classes come
    out wrong while nothing raises. Callers must refuse to label or featurise.
    """
    return bool(moves_dex()) and bool(pokedex())


def forme_id(details: str) -> str:
    """Forme id of a details field: ``Gengar-Mega, L50, M`` -> ``gengarmega``."""
    return to_id(details.split(",", 1)[0])


@lru_cache(maxsize=4096)
def base_species_id(name: str) -> str:
    """Canonical base-species id of a species or forme name.

    Matches ``vgc_bench.src.opponent_preview.species_id`` (tested on the dex).
    """
    forme = forme_id(name)
    entry = pokedex().get(forme)
    if entry is None:
        return forme
    return to_id(str(entry.get("baseSpecies") or entry.get("name") or forme))


def species_from_details(details: str) -> str:
    """Base-species id of a ``switch`` / ``poke`` details field."""
    return base_species_id(details.split(",", 1)[0])


@lru_cache(maxsize=4096)
def is_mega_forme(forme: str) -> bool:
    """Whether a forme id is a Mega forme, by the dex ``forme`` field."""
    entry = pokedex().get(forme)
    if entry is None:
        return False
    return "Mega" in str(entry.get("forme", "")).split("-")


def move_entry(move_id: str) -> dict[str, Any] | None:
    return moves_dex().get(move_id)


def move_target_type(move_id: str) -> str | None:
    entry = moves_dex().get(move_id)
    return None if entry is None else entry.get("target")


def is_choosable_target(move_id: str) -> bool | None:
    """Whether the player aims this move; None when the move is not in the dex."""
    target = move_target_type(move_id)
    return None if target is None else target in _CHOOSABLE_TARGETS


def is_damaging(move_id: str) -> bool:
    entry = moves_dex().get(move_id)
    return entry is not None and entry.get("category") != "Status"


def is_spread(move_id: str) -> bool:
    """A damaging move that hits every adjacent foe (or everyone adjacent)."""
    return is_damaging(move_id) and move_target_type(move_id) in _SPREAD_TARGETS


def is_stalling(move_id: str) -> bool:
    """Counts toward the consecutive-Protect penalty (dex ``stallingMove``)."""
    entry = moves_dex().get(move_id)
    return bool(entry and entry.get("stallingMove"))


def is_protect_family(move_id: str) -> bool:
    """A stalling move the user casts on itself that blocks incoming moves.

    Protect, Detect and the shield variants qualify; Endure (does not block) and
    the side-wide guards (not a stalling move, or not self-targeted) do not.
    """
    entry = moves_dex().get(move_id)
    if not entry or not entry.get("stallingMove") or entry.get("target") != "self":
        return False
    volatile = entry.get("volatileStatus")
    owner = moves_dex().get(to_id(str(volatile))) if volatile else None
    condition = (owner or entry).get("condition") or entry.get("condition") or {}
    return "onTryHit" in condition


def raises_stall_counter(move_id: str) -> bool:
    """Whether a successful use raises the user's consecutive-Protect counter.

    True for dex ``stallingMove`` moves and for the side guards whose
    ``onHitSide`` adds the same counter (pokemon-showdown/data/moves.ts: the
    wide and quick guards call ``source.addVolatile('stall')``; the guard that
    only blocks status moves has no ``onHitSide`` and does not).
    """
    entry = moves_dex().get(move_id)
    if not entry:
        return False
    return bool(
        entry.get("stallingMove")
        or (entry.get("sideCondition") and "onHitSide" in entry)
    )


def is_redirection(move_id: str) -> bool:
    """A move whose effect pulls the foes' single-target moves onto one Pokemon."""
    entry = moves_dex().get(move_id)
    return bool(entry and "onFoeRedirectTarget" in (entry.get("condition") or {}))


def overrides_action(effect: str) -> bool:
    """An effect that replaces its target's queued move (dex ``onOverrideAction``).

    When it starts before the target has acted, the move the target then uses
    is the one forced on it, with a random target, and the click is not shown
    (pokemon-showdown/sim/battle-actions.ts runMove; the Champions mod changes
    the queued action at once, data/mods/champions/moves.ts).
    """
    entry = moves_dex().get(effect)
    return bool(entry and "onOverrideAction" in (entry.get("condition") or {}))


def _inserts_repeat(effect: str) -> bool:
    """Whether ``-singleturn|TARGET|move: <effect>|[of] USER`` announces a repeat.

    The simulator prints such a line for a move that leaves a single-turn effect
    on the target (the move has a ``volatileStatus``), and for a move that leaves
    nothing and makes the target use its previous move at once
    (pokemon-showdown/data/moves.ts: ``queue.prioritizeAction``). Only the
    second kind has no effect entry of its own.
    """
    entry = moves_dex().get(effect)
    if not entry or entry.get("category") != "Status":
        return False
    return not any(
        entry.get(key)
        for key in ("volatileStatus", "condition", "sideCondition", "pseudoWeather")
    )


def _first_turn_only_from_sim(path: Path) -> frozenset[str] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    found: set[str] = set()
    current: str | None = None
    for line in text.split("\n"):
        key = _SIM_MOVE_KEY.match(line)
        if key is not None:
            current = key.group(1)
        elif line.startswith("\t}"):
            current = None
        elif current is not None and _SIM_FIRST_TURN_GATE.search(line):
            found.add(current)
    return frozenset(found) if found else None


@lru_cache(maxsize=1)
def _first_turn_only() -> tuple[frozenset[str], str]:
    """(move ids usable only on the first turn on the field, where that came from).

    The move JSON does not carry this property, so it is read from the simulator
    source in the repo's Showdown clone (moves gated on ``activeMoveActions``).
    Without the clone the fallback is a dex heuristic (a damaging move with
    priority >= 2 and a try-gate, or a stalling side guard with a try-gate), which
    also admits one move gated on the target's action instead.
    """
    scanned = _first_turn_only_from_sim(
        _REPO_ROOT / "pokemon-showdown" / "data" / "moves.ts"
    )
    if scanned is not None:
        return scanned, "sim_source"
    heuristic = {
        move_id
        for move_id, entry in moves_dex().items()
        if "onTry" in entry
        and (
            (entry.get("category") != "Status" and entry.get("priority", 0) >= 2)
            or (entry.get("stallingMove") and entry.get("sideCondition"))
        )
    }
    return frozenset(heuristic), "dex_heuristic"


def first_turn_only_moves() -> frozenset[str]:
    return _first_turn_only()[0]


def first_turn_only_source() -> str:
    """``sim_source`` (exact) or ``dex_heuristic`` (clone absent, slightly wide)."""
    return _first_turn_only()[1]


def is_first_turn_only(move_id: str) -> bool:
    """Whether the move works only on the user's first turn on the field."""
    return move_id in _first_turn_only()[0]


def is_fake_out_like(move_id: str) -> bool:
    """A damaging first-turn-only move: the 'first turn only' candidate flag."""
    return is_first_turn_only(move_id) and is_damaging(move_id)


def dex_signature() -> dict[str, Any]:
    """What the dex-derived rules were built from, for manifests and artifacts.

    The first-turn-only set depends on whether the repo's simulator clone is
    present, so a dataset and the runtime that serves its model must compare
    this and refuse or warn on a difference.
    """
    moves, source = _first_turn_only()
    return {
        "dex_available": dex_available(),
        "n_moves": len(moves_dex()),
        "n_species": len(pokedex()),
        "first_turn_only_source": source,
        "first_turn_only_moves": sorted(moves),
    }


def move_intent(move_id: str, target: str | None) -> str | None:
    """One of ``INTENT_CLASSES`` for a move and its target class, or None.

    None means the class cannot be decided: an unknown move, or a single-target
    attack whose target was not logged. An attack the player does not aim is
    ``spread_attack``: a spread move, a random or scripted target, or an aimed
    move the simulator turned into a spread hit (target ``auto``).
    """
    entry = moves_dex().get(move_id)
    if entry is None:
        return None
    if is_protect_family(move_id):
        return INTENT_PROTECT
    target_type = entry.get("target")
    if entry.get("category") != "Status":
        if target_type not in _CHOOSABLE_TARGETS or target == TARGET_AUTO:
            return INTENT_SPREAD
        if target == TARGET_FOE_A:
            return INTENT_ATTACK_FOE_A
        if target == TARGET_FOE_B:
            return INTENT_ATTACK_FOE_B
        if target in (TARGET_ALLY, TARGET_SELF):
            return INTENT_SUPPORT
        return None
    if target_type in _FOE_FACING_TARGETS:
        return INTENT_STATUS_FOE
    if target_type not in _CHOOSABLE_TARGETS or target_type in _ALLY_ONLY_TARGETS:
        return INTENT_SUPPORT
    if target in (TARGET_FOE_A, TARGET_FOE_B):
        return INTENT_STATUS_FOE
    if target in (TARGET_ALLY, TARGET_SELF):
        return INTENT_SUPPORT
    flags = entry.get("flags") or {}
    boosts = entry.get("boosts") or {}
    helps = bool(flags.get("heal")) or any(value > 0 for value in boosts.values())
    return INTENT_SUPPORT if helps else INTENT_STATUS_FOE


# --- team sheets --------------------------------------------------------------


@dataclass(frozen=True)
class SheetSet:
    """One Pokemon of an open team sheet. Ids throughout; moves in sheet order."""

    species: str
    forme: str
    item: str | None
    ability: str | None
    moves: tuple[str, ...]
    nickname: str = ""


def parse_showteam(payload: str) -> list[SheetSet]:
    """Parse the packed team of a ``|showteam|side|<payload>`` line.

    ``payload`` is everything after the side field (re-join the split event with
    ``|``). Packed entries are ``Nick|Species|Item|Ability|Move,Move,..|Nature|..``
    separated by ``]``; the species field is empty when it equals the nickname.
    """
    sets: list[SheetSet] = []
    for entry in payload.split("]"):
        fields = entry.split("|")
        if len(fields) < 5 or not fields[0].strip():
            continue
        species = fields[1].strip() or fields[0].strip()
        moves = tuple(to_id(move) for move in fields[4].split(",") if move.strip())
        sets.append(
            SheetSet(
                species=base_species_id(species),
                forme=to_id(species),
                item=to_id(fields[2]) or None,
                ability=to_id(fields[3]) or None,
                moves=moves,
                nickname=fields[0].strip(),
            )
        )
    return sets


# --- labels -------------------------------------------------------------------


@dataclass(frozen=True)
class SlotAction:
    """What one slot did in one turn, as far as the log shows.

    ``kind``
        ``move`` (a move line, or a ``cant`` line that names the move),
        ``switch`` (voluntary), ``hidden`` (a ``cant`` line, a confusion
        self-hit or an overridden click: some move was chosen, which one is not
        shown), ``none`` (no action line at all).
    ``move`` / ``target``
        Move id and where it landed: ``foe_a`` / ``foe_b`` (the foe's slot letter,
        never a seat-mirrored number), ``ally``, ``self``, or ``auto`` for a move
        the player does not aim. ``None`` target = aimed but not logged.
    ``switch_to``
        Base species brought in by a voluntary switch.
    ``mega``
        A Mega Evolution line for this slot this turn. Megas resolve before every
        move, so it is visible even when the move is not, and proves the slot
        chose a move rather than a switch.
    ``reason``
        Why the action is hidden, missing or not a free choice: a ``cant`` reason
        id (``flinch``, ``slp``, ``par``, ``frz``, ...), ``confusion``,
        ``fainted_first``, ``left_field`` (dragged or ejected before acting),
        ``game_ended`` (the game ended while the turn was resolving),
        ``unresolved`` (the turn never resolved: forfeit, timeout), ``forced``
        (recharge, a locked-move continuation, or the move of a Pokemon with
        no usable move left), ``overridden`` (another Pokemon's effect replaced
        the click before the slot acted), ``unexplained``.
    ``switch_known`` / ``protect_known``
        Whether the log settles "did this slot switch" / "did it use a
        Protect-family move". Both hold for every visible action; for censored
        slots they say which exclusions are valid (voluntary switches and
        Protect resolve before anything that could have censored the slot).
    ``target_trusted``
        False when the logged target may not be the clicked one.
    ``locked``
        A forced action, not a choice: exclude from any loss.
    ``side`` / ``slot`` / ``species``
        The slot at turn start and the base species that stood there.
    ``forced_move``
        With ``overridden``: the move the slot was made to use. It is what the
        Pokemon did, not what was clicked, so it is not in ``move``.
    ``target_forced``
        The move was aimed at a foe and only one foe stood at the start of the
        turn, so its slot was not a choice (``target_trusted`` is False too).
        Where the move landed is still certain: mask it from a target loss, keep
        it for "which slot was attacked".
    """

    kind: str
    move: str | None = None
    target: str | None = None
    switch_to: str | None = None
    mega: bool = False
    reason: str | None = None
    switch_known: bool = True
    protect_known: bool = True
    target_trusted: bool = True
    locked: bool = False
    side: str = ""
    slot: str = ""
    species: str = ""
    forced_move: str | None = None
    target_forced: bool = False

    @property
    def key(self) -> str:
        return self.side + self.slot

    @property
    def is_protect(self) -> bool:
        return self.kind == KIND_MOVE and is_protect_family(self.move or "")

    @property
    def intent(self) -> str | None:
        return intent_class(self)


def intent_class(action: SlotAction) -> str | None:
    """The design's 7-way intent of an action; None when it is not decidable."""
    if action.kind == KIND_SWITCH:
        return INTENT_SWITCH
    if action.kind != KIND_MOVE or action.move is None:
        return None
    return move_intent(action.move, action.target)


@dataclass
class Occupant:
    """The Pokemon standing in a slot. ``species`` is the base-species id."""

    species: str
    forme: str
    name: str = ""
    fainted: bool = False


class FieldTracker:
    """Who stands in each of the four slots, updated from raw events.

    Keyed by slot letter. ``apply`` understands ``switch`` / ``drag`` /
    ``replace`` (Illusion ends: same slot, true identity) / ``swap`` (Ally Switch)
    / ``faint`` / ``detailschange`` / ``-formechange`` and ignores everything
    else. A Pokemon revived on the bench re-enters through ``switch`` as a fresh,
    living occupant.
    """

    def __init__(self) -> None:
        self.active: dict[str, Occupant | None] = {key: None for key in SLOT_KEYS}

    def apply(self, event: Sequence[str]) -> None:
        if len(event) < 3:
            return
        kind = event[1]
        if kind not in _TRACKED_KINDS:
            return
        ident = parse_ident(event[2])
        if ident is None or ident.key is None or ident.key not in self.active:
            return
        key = ident.key
        if kind in ("switch", "drag"):
            details = event[3] if len(event) > 3 else ident.name
            self.active[key] = Occupant(
                species_from_details(details), forme_id(details), ident.name
            )
        elif kind == "replace":
            details = event[3] if len(event) > 3 else ident.name
            occupant = self.active[key]
            if occupant is None:
                self.active[key] = Occupant(
                    species_from_details(details), forme_id(details), ident.name
                )
            else:
                occupant.species = species_from_details(details)
                occupant.forme = forme_id(details)
                occupant.name = ident.name
        elif kind == "swap":
            if not swap_is_noop(ident, event):
                first, second = ident.side + "a", ident.side + "b"
                self.active[first], self.active[second] = (
                    self.active[second],
                    self.active[first],
                )
        elif kind == "faint":
            occupant = self.active[key]
            if occupant is not None:
                occupant.fainted = True
        elif len(event) > 3:  # detailschange / -formechange
            occupant = self.active[key]
            if occupant is not None:
                occupant.forme = forme_id(event[3])

    def alive(self) -> dict[str, Occupant]:
        """Copies of the living occupants, keyed by slot."""
        return {
            key: replace(occupant)
            for key, occupant in self.active.items()
            if occupant is not None and not occupant.fainted
        }


_TRACKED_KINDS = frozenset(
    {"switch", "drag", "replace", "swap", "faint", "detailschange", "-formechange"}
)


def swap_is_noop(ident: Ident, event: Sequence[str]) -> bool:
    """``|swap|p2a: X|1`` moves the Pokemon in slot a to position 1 (slot b)."""
    position = event[3].strip() if len(event) > 3 else ""
    return (ident.slot == "a" and position == "0") or (
        ident.slot == "b" and position == "1"
    )


@dataclass
class TurnSegment:
    """One turn of a log: the events after ``|turn|N`` up to the next ``|turn|``.

    ``start`` holds the living occupant of each slot when the turn began (absent
    key = empty or fainted slot). ``ended`` is set when a ``win`` / ``tie`` line
    fell inside the segment.
    """

    turn: int
    start: dict[str, Occupant]
    events: list[Event] = field(default_factory=list)
    ended: bool = False


def segment_turns(events: Iterable[Sequence[str]]) -> list[TurnSegment]:
    """Cut a whole log into turns, tracking slot occupancy across it."""
    tracker = FieldTracker()
    segments: list[TurnSegment] = []
    current: TurnSegment | None = None
    for event in events:
        if len(event) < 2:
            continue
        kind = event[1]
        if kind == "turn":
            try:
                number = int(event[2])
            except (IndexError, ValueError):
                continue
            current = TurnSegment(number, tracker.alive())
            segments.append(current)
            continue
        if current is not None:
            current.events.append(list(event))
            if kind in ("win", "tie"):
                current.ended = True
        tracker.apply(event)
    return segments


@dataclass
class _Record:
    side: str
    slot: str
    species: str
    kind: str | None = None
    move: str | None = None
    target: str | None = None
    switch_to: str | None = None
    mega: bool = False
    reason: str | None = None
    locked: bool = False
    trusted: bool = True
    awaiting_anim: bool = False
    fainted_phase: str | None = None
    left_phase: str | None = None
    forced: bool = False
    forced_move: str | None = None
    # Effect that will replace this slot's click (it started before the slot acted).
    override: str | None = None
    # The slot's next untagged move line is a repeat a partner inserted.
    repeat_pending: bool = False
    # Slot the move line named, and the slot whose ability activated right after
    # a blank-target move line (compared with the -anim target when it arrives).
    target_key: str | None = None
    redirect_key: str | None = None


@dataclass
class _Scan:
    records: dict[str, _Record]
    end_slot: dict[str, str]
    phase: str
    resolved: bool
    foes: dict[str, int]


def _target_class(mover_key: str, target: Ident) -> str | None:
    if target.key is None:
        return None
    if target.side != mover_key[:2]:
        return TARGET_FOE_A if target.slot == "a" else TARGET_FOE_B
    return TARGET_SELF if target.key == mover_key else TARGET_ALLY


def _scan_turn(segment: TurnSegment, counters: Counter[str] | None) -> _Scan:
    records = {
        key: _Record(key[:2], key[2:], occupant.species)
        for key, occupant in segment.start.items()
        if key in SLOT_KEYS
    }
    # position[current slot] = slot the Pokemon standing there held at turn start.
    position: dict[str, str | None] = {
        key: (key if key in records else None) for key in SLOT_KEYS
    }
    redirectors: set[str] = set()
    bare_item_loss: set[str] = set()
    faints = {side: 0 for side in SIDES}
    # Living foes of each side when the turn started (the decision point).
    foes = {side: sum(1 for key in records if key[:2] != side) for side in SIDES}
    phase = "pre"
    resolved = False
    just_moved: _Record | None = None

    def bump(name: str) -> None:
        if counters is not None:
            counters[name] += 1

    def distrust(mover_key: str, target: Ident) -> bool:
        if target.side == mover_key[:2] or target.key is None:
            return False
        return target.key in redirectors or faints.get(target.side, 0) > 0

    def aim(record: _Record, mover_key: str, target: Ident) -> None:
        record.target = _target_class(mover_key, target)
        record.target_key = target.key
        if distrust(mover_key, target):
            record.trusted = False
        # With one foe standing the simulator sends every foe-aimed move to it,
        # whichever slot was clicked (pokemon-showdown/sim/battle.ts getTarget).
        if target.side != mover_key[:2] and foes.get(mover_key[:2], 2) <= 1:
            record.forced = True

    for event in segment.events:
        if len(event) < 2:
            continue
        kind = event[1]
        if kind in CHATTER_KINDS or kind in ("win", "tie"):
            continue
        resolved = True
        moved_before, just_moved = just_moved, None
        if kind == "upkeep":
            phase = "post"
            continue
        ident = parse_ident(event[2]) if len(event) > 2 else None
        if ident is None:
            continue
        current = ident.key
        if current is None or current not in position:
            continue
        start_slot = position[current]
        record = records.get(start_slot) if start_slot is not None else None
        lost_item_bare = current in bare_item_loss
        bare_item_loss.discard(current)

        if kind in ("switch", "drag"):
            if record is not None and record.kind is None and not record.fainted_phase:
                tagged = tag_value(event, "[from]", 5) is not None
                # A consumed item with nothing after it but the holder leaving is
                # an ejection, not a click.
                if (
                    kind == "switch"
                    and phase == "pre"
                    and not tagged
                    and not lost_item_bare
                ):
                    record.kind = KIND_SWITCH
                    if len(event) > 3:
                        record.switch_to = species_from_details(event[3])
                else:
                    record.left_phase = phase
                    bump("left_field_before_acting")
            position[current] = None
            redirectors.discard(current)
        elif kind == "swap":
            if not swap_is_noop(ident, event):
                first, second = ident.side + "a", ident.side + "b"
                position[first], position[second] = position[second], position[first]
                for group in (redirectors, bare_item_loss):
                    swapped = {
                        (second if key == first else first)
                        for key in group
                        if key in (first, second)
                    }
                    group.difference_update((first, second))
                    group.update(swapped)
        elif kind == "faint":
            faints[ident.side] = faints.get(ident.side, 0) + 1
            if record is not None and record.fainted_phase is None:
                record.fainted_phase = phase
        elif kind == "-mega":
            if phase == "pre":
                phase = "mid"
            if record is not None:
                record.mega = True
        elif kind == "detailschange":
            if len(event) > 3 and is_mega_forme(forme_id(event[3])):
                if phase == "pre":
                    phase = "mid"
                if record is not None:
                    record.mega = True
        elif kind == "-enditem":
            if not any(part.startswith("[") for part in event[4:]):
                bare_item_loss.add(current)
        elif kind == "-singleturn":
            if len(event) > 3:
                effect = effect_id(event[3])
                if is_redirection(effect):
                    redirectors.add(current)
                giver = parse_ident(tag_value(event, "[of]", 4) or "")
                if (
                    record is not None
                    and giver is not None
                    and giver.key != current
                    and _inserts_repeat(effect)
                ):
                    record.repeat_pending = True
        elif kind == "-start":
            if (
                record is not None
                and record.kind is None
                and len(event) > 3
                and overrides_action(effect_id(event[3]))
            ):
                record.override = effect_id(event[3])
        elif kind == "-end":
            # Cured before the slot acted (an item): the click stands again.
            if (
                record is not None
                and record.override is not None
                and len(event) > 3
                and effect_id(event[3]) == record.override
            ):
                record.override = None
        elif kind == "-activate":
            # The simulator prints a redirecting ability directly after the move
            # line it redirected and rewrites that line's target to the ability's
            # holder, which may stand on either side
            # (pokemon-showdown/sim/pokemon.ts getMoveTargets, retargetLastMove).
            if (
                moved_before is not None
                and record is not moved_before
                and len(event) > 3
                and event[3].lower().startswith("ability:")
            ):
                if moved_before.awaiting_anim:
                    moved_before.redirect_key = current
                elif moved_before.target_key == current and moved_before.trusted:
                    moved_before.trusted = False
                    bump("target_untrusted_ability")
        elif kind == "-damage":
            # Protocol tag: a confusion self-hit is the only trace of that turn.
            if tag_value(event, "[from]", 4) == "confusion":
                if phase == "pre":
                    phase = "mid"
                if record is not None and record.kind is None:
                    record.kind = KIND_HIDDEN
                    record.reason = REASON_CONFUSION
        elif kind == "cant":
            if phase == "pre":
                phase = "mid"
            reason_text = event[3] if len(event) > 3 else ""
            if reason_text.lower().startswith("ability:") and tag_value(event, "[of]"):
                bump("cant_blocked_by_ability")
                continue
            if record is None or record.kind is not None:
                continue
            if record.repeat_pending:
                # The inserted repeat was stopped; the slot's own action follows.
                record.repeat_pending = False
                bump("repeat_move_ignored")
                bump("repeat_before_own_move")
                continue
            reason = effect_id(reason_text)
            named = event[4] if len(event) > 4 and not event[4].startswith("[") else ""
            # Protocol keyword: |cant|POKEMON|recharge is the forced second turn.
            if reason == "recharge":
                record.kind = KIND_NONE
                record.reason = REASON_FORCED
                record.locked = True
            elif named and record.override is not None:
                # The move named is the one forced on the slot, not its click.
                record.kind = KIND_HIDDEN
                record.reason = REASON_OVERRIDDEN
                record.forced_move = to_id(named)
                bump("overridden_move")
            elif named:
                record.kind = KIND_MOVE
                record.move = to_id(named)
                record.reason = reason
                if is_choosable_target(record.move) is False:
                    record.target = TARGET_AUTO
            else:
                record.kind = KIND_HIDDEN
                record.reason = reason
        elif kind == "move":
            if phase == "pre":
                phase = "mid"
            if record is None:
                bump("move_by_later_entrant")
                continue
            move_id = to_id(event[3]) if len(event) > 3 else ""
            origin = tag_value(event, "[from]", 5)
            # Protocol keyword: [from] lockedmove marks a forced continuation.
            locked = origin is not None and to_id(origin) == "lockedmove"
            own = origin is not None and effect_id(origin) == move_id
            if origin is not None and not locked and not own:
                bump("called_move_ignored")
                continue
            if record.repeat_pending and origin is None:
                # A repeat a partner inserted: not this slot's choice, whether
                # its own action came before or is still to come.
                record.repeat_pending = False
                bump("repeat_move_ignored")
                if record.kind is None:
                    bump("repeat_before_own_move")
                continue
            if record.kind is not None:
                bump("repeat_move_ignored")
                continue
            if record.override is not None and not locked:
                record.kind = KIND_HIDDEN
                record.reason = REASON_OVERRIDDEN
                record.forced_move = move_id
                bump("overridden_move")
                continue
            record.kind = KIND_MOVE
            record.move = move_id
            # A Pokemon with no usable move left has its move chosen for it.
            if locked or (moves_dex().get(move_id) or {}).get("struggleRecoil"):
                record.locked = True
                record.reason = REASON_FORCED
            choosable = is_choosable_target(move_id)
            if choosable is None:
                bump("unknown_move")
            # Protocol tag: "[spread] <slots>" marks a move that went for more
            # than one target. On an aimed move it means the simulator changed
            # the target type in flight and replaced the click by a random pick
            # (pokemon-showdown/sim/battle-actions.ts useMoveInner).
            spread = tag_value(event, "[spread]", 5) is not None
            if choosable is False or spread:
                record.target = TARGET_AUTO
                if choosable and spread:
                    bump("aimed_move_became_spread")
            else:
                target = parse_ident(event[4]) if len(event) > 4 else None
                if target is not None and target.key is not None:
                    aim(record, current, target)
                elif target is not None and target.side == current[:2]:
                    # A reference without a slot letter is a Pokemon that is not
                    # on the field: the mover's fainted partner, which the
                    # simulator keeps as the target (the line has "[notarget]").
                    record.target = TARGET_ALLY
                else:
                    record.awaiting_anim = True
                just_moved = record
        elif kind == "-anim":
            if (
                record is not None
                and record.awaiting_anim
                and len(event) > 4
                and to_id(event[3]) == record.move
            ):
                target = parse_ident(event[4])
                if target is not None and target.key is not None:
                    aim(record, current, target)
                    if record.redirect_key == target.key and record.trusted:
                        record.trusted = False
                        bump("target_untrusted_ability")
                    record.awaiting_anim = False
                    bump("target_from_anim")

    end_slot = {start: now for now, start in position.items() if start is not None}
    return _Scan(records, end_slot, phase, resolved, foes)


def _finish(record: _Record, scan: _Scan, upkeep_seen: bool) -> SlotAction:
    kind = record.kind
    reason = record.reason
    switch_known = True
    protect_known = True
    if kind == KIND_HIDDEN:
        # A flinch proves the slot was hit before acting, an override that a
        # move of lower priority resolved before it: both exclude a Protect
        # click, which would have resolved first. Nothing else does.
        protect_known = reason in (REASON_FLINCH, REASON_OVERRIDDEN)
    elif kind is None:
        kind = KIND_NONE
        if record.fainted_phase is not None:
            reason = REASON_FAINTED_FIRST
            switch_known = protect_known = record.fainted_phase != "pre"
        elif record.left_phase is not None:
            reason = REASON_LEFT_FIELD
            switch_known = protect_known = record.left_phase != "pre"
        elif not scan.resolved:
            reason = REASON_UNRESOLVED
            switch_known = protect_known = False
        elif upkeep_seen:
            reason = REASON_UNEXPLAINED
        else:
            reason = REASON_GAME_ENDED
            switch_known = protect_known = scan.phase != "pre"
        if record.mega:
            switch_known = True
    return SlotAction(
        kind=kind,
        move=record.move,
        target=record.target,
        switch_to=record.switch_to,
        mega=record.mega,
        reason=reason,
        switch_known=switch_known,
        protect_known=protect_known,
        target_trusted=record.trusted and not record.forced,
        locked=record.locked,
        side=record.side,
        slot=record.slot,
        species=record.species,
        forced_move=record.forced_move,
        target_forced=record.forced,
    )


def read_turn_actions(
    segment: TurnSegment,
    next_segment: TurnSegment | None = None,
    counters: Counter[str] | None = None,
) -> dict[str, SlotAction]:
    """One ``SlotAction`` per slot alive at the start of ``segment``, both sides.

    Keys are turn-start slot keys (``p1a`` ...). ``next_segment`` lets a blank
    charge-move target be read from the next turn's locked-move line; without it
    (the serve-time "previous action" feature) such a target stays ``None``.
    ``counters`` receives named counts of the unusual lines met. Never raises:
    a segment that cannot be read yields ``none`` / ``unresolved`` actions (no
    information) and, when ``counters`` is given, a ``reader_error:<exception>``
    count. A caller that must notice such a failure has to pass ``counters``.
    """
    try:
        return _read_turn(segment, next_segment, counters)
    except Exception as exc:
        try:
            if counters is not None:
                counters[f"reader_error:{type(exc).__name__}"] += 1
        except Exception:
            pass
        return _no_information(segment)


def _no_information(segment: TurnSegment) -> dict[str, SlotAction]:
    """``none`` / ``unresolved`` for every readable turn-start slot; never raises."""
    out: dict[str, SlotAction] = {}
    try:
        for key, occupant in segment.start.items():
            if key not in SLOT_KEYS:
                continue
            species = getattr(occupant, "species", "")
            out[key] = SlotAction(
                kind=KIND_NONE,
                reason=REASON_UNRESOLVED,
                switch_known=False,
                protect_known=False,
                side=key[:2],
                slot=key[2:],
                species=species if isinstance(species, str) else "",
            )
    except Exception:
        return out
    return out


def _read_turn(
    segment: TurnSegment,
    next_segment: TurnSegment | None,
    counters: Counter[str] | None,
) -> dict[str, SlotAction]:
    scan = _scan_turn(segment, counters)
    if next_segment is not None:
        pending = [
            (start, record)
            for start, record in scan.records.items()
            if record.kind == KIND_MOVE
            and record.target is None
            and record.awaiting_anim
        ]
        if pending:
            following = _scan_turn(next_segment, None).records
            for start, record in pending:
                later = following.get(scan.end_slot.get(start, ""))
                if (
                    later is not None
                    and later.locked
                    and later.kind == KIND_MOVE
                    and later.move == record.move
                    and later.species == record.species
                    and later.target is not None
                ):
                    record.target = later.target
                    # The click was made this turn; it landed next turn, when a
                    # lone foe would have taken it whatever was clicked.
                    record.trusted = later.trusted and not later.forced
                    record.forced = (
                        later.target in _FOE_TARGET_CLASSES
                        and scan.foes.get(record.side, 2) <= 1
                    )
                    if counters is not None:
                        counters["target_from_next_turn"] += 1
    upkeep_seen = scan.phase == "post"
    return {
        key: _finish(record, scan, upkeep_seen) for key, record in scan.records.items()
    }


def read_log_actions(
    events: Iterable[Sequence[str]], counters: Counter[str] | None = None
) -> list[tuple[TurnSegment, dict[str, SlotAction]]]:
    """Segment a whole log and read every turn, with next-turn target back-fill."""
    segments = segment_turns(events)
    following: list[TurnSegment | None] = [*segments[1:], None]
    return [
        (segment, read_turn_actions(segment, nxt, counters))
        for segment, nxt in zip(segments, following)
    ]
