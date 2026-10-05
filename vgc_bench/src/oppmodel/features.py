"""Features, labels and scoring of the opponent predictor (OPPONENT_PREDICTOR.md).

One ``Featurizer`` turns a ``PublicSnapshot`` (the public state at the start of a
turn) into one EXAMPLE for one ACTOR side: a dict of small numpy arrays with the
layout returned by ``layout()``. The same code encodes training logs and the
live bot's shadow state, so there is nothing to keep in step.

    featurizer = Featurizer.build(repertoire)          # or Featurizer.load(dir)
    example = featurizer.encode(snapshot, "p2")        # None when it stands down
    labels = featurizer.encode_labels(example, snapshot, "p2", record.actions)
    batch = collate([example | labels, ...])
    scores = slot_nll(predictor.predict(batch), batch)

Token order is actor-relative: sides are (actor, other) and the twelve Pokemon
are the actor's roster in team-preview order, then the other roster. Targets
stay absolute: ``foe_a`` / ``foe_b`` are the OTHER side's slot letters.

The action space of each of the actor's two slots (``a``, ``b``) has
``n_cand + 1 + 6`` entries: ``n_cand`` candidate moves, the ``OTHER`` bucket (a
move outside the candidates) and a pointer over the actor's six roster Pokemon
(voluntary switch). Candidates are the Pokemon's sheet moves when its sheet is
known; otherwise the moves it has shown, then its set key's most used moves in
the training split (``Repertoire``), then the most used moves overall. A
Pokemon that has shown four moves has no further candidate.

Ids are vocabulary indices with 0 = unknown. Species and moves that the current
dex knows but the stored vocabulary does not get EXTENSION rows after the
vocabulary in the numeric tables, so they keep their dex-derived numerics;
``Featurizer.densify`` gathers those numerics and then clamps such ids to 0 for
embedding layers. With a vocabulary built from the same dex there are none.

Labels come from ``events.SlotAction``. ``y_set`` is the set of actions the log
is consistent with: one entry for a visible free choice, the censored set for a
slot whose action is hidden, all zero when the log says nothing or the action
was not a choice. The censored set follows the label's own flags: a slot known
not to have switched keeps its moves and OTHER, and loses its Protect-family
candidates when the log also rules a Protect out (after a faint or a flinch;
not after sleep / paralysis / freeze / confusion). A turn that never resolved,
a forced action (recharge, a locked move) and a slot with no action line and no
explanation carry no set. Every model is scored by ``slot_nll``: ``-log`` of
the probability mass on ``y_set``.

Nothing here raises into a caller on the runtime path: ``Featurizer.encode`` /
``encode_labels`` / ``encode_turn`` / ``collate`` / ``densify`` count failures
in ``Featurizer.counters`` and return None or their input; ``collate``,
``apply_elo_mode``, ``sheet_unknown_as_closed``, ``slot_view``,
``uniform_prediction``, ``event_probs`` and ``intent_probs`` return an empty
result (the last one None) and count in ``COUNTERS``. The score helpers
(``normalize_prediction``, ``slot_nll``, ``fine_probs``, ``fine_label``,
``topk_hits``) and the loaders are offline tools: they raise ``ValueError`` /
``OSError`` on malformed input, so a broken prediction is never scored quietly.
"""

from __future__ import annotations

import json
import zlib
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from poke_env.data import GenData

from vgc_bench.src.move_semantics import MOVE_SEM_LEN, move_semantics
from vgc_bench.src.oppmodel.events import (
    INTENT_CLASSES,
    INTENT_PROTECT,
    INTENT_SWITCH,
    KIND_HIDDEN,
    KIND_MOVE,
    KIND_NONE,
    KIND_SWITCH,
    REASON_CONFUSION,
    REASON_FAINTED_FIRST,
    REASON_FLINCH,
    REASON_FORCED,
    REASON_GAME_ENDED,
    REASON_LEFT_FIELD,
    REASON_OVERRIDDEN,
    REASON_UNEXPLAINED,
    REASON_UNRESOLVED,
    SIDES,
    SLOT_LETTERS,
    TARGET_ALLY,
    TARGET_AUTO,
    TARGET_CLASSES,
    TARGET_FOE_A,
    TARGET_FOE_B,
    TARGET_SELF,
    SlotAction,
    dex_available,
    dex_signature,
    is_choosable_target,
    is_damaging,
    is_fake_out_like,
    is_first_turn_only,
    is_mega_forme,
    is_protect_family,
    is_redirection,
    is_spread,
    is_stalling,
    move_intent,
    moves_dex,
    raises_stall_counter,
    to_id,
)
from vgc_bench.src.oppmodel.public_state import (
    BOOST_STATS,
    ITEM_CONSUMED,
    ITEM_KNOWN,
    ITEM_UNKNOWN,
    EffectStart,
    MonPublic,
    PublicSnapshot,
    SidePublic,
)

Example = dict[str, np.ndarray]
Batch = dict[str, np.ndarray]

LAYOUT_VERSION = 1
N_CAND_DEFAULT = 12
N_ROSTER = 6
N_MON = 2 * N_ROSTER
N_SLOT = 2
N_TARGET = len(TARGET_CLASSES)
N_INTENT = len(INTENT_CLASSES)
MAX_MOVES = 4
MAX_TYPES = 3
MAX_VOLATILES = 4
N_VOLATILE_HASH = 8
MAX_AGE = 15

# ``mon_id`` columns.
ID_SPECIES, ID_FORME, ID_KEY, ID_ITEM, ID_ABILITY, ID_LAST_MOVE = range(6)
N_MON_ID = 6
# ``mon_cat`` columns (small integer codes).
(
    CAT_STATUS,
    CAT_SLOT,
    CAT_ITEM_STATE,
    CAT_BROUGHT,
    CAT_LAST_KIND,
    CAT_LAST_TARGET,
    CAT_PROTECT_STREAK,
    CAT_TURNS_ON_FIELD,
    CAT_N_MOVES,
) = range(9)
N_MON_CAT = 9
# ``mon_flag`` columns (0 / 1).
(
    FLAG_PRESENT,
    FLAG_REVEALED,
    FLAG_FAINTED,
    FLAG_IS_MEGA,
    FLAG_MEGA_POSSIBLE,
    FLAG_FIRST_TURN,
    FLAG_PROTECTED_LAST,
    FLAG_ABILITY_KNOWN,
    FLAG_LAST_MEGA,
    FLAG_LAST_LOCKED,
    FLAG_LAST_PROTECT,
) = range(11)
N_MON_FLAG = 11
MOVE_REVEALED = 1  # ``mon_move_flag`` bits
MOVE_FROM_SHEET = 2

SLOT_BENCH, SLOT_A, SLOT_B = 0, 1, 2
BROUGHT_NO, BROUGHT_YES, BROUGHT_UNKNOWN = 0, 1, 2
ITEM_STATE_CODES = {ITEM_UNKNOWN: 0, ITEM_KNOWN: 1, ITEM_CONSUMED: 2}

# ``game_flag`` columns and the sheet codes used in two of them.
G_BO3, G_RATED, G_ACTOR_SHEET, G_OTHER_SHEET = range(4)
SHEET_CLOSED, SHEET_OPEN, SHEET_UNKNOWN = 0, 1, 2
# ``ctx_cat`` columns; ``field_age`` starts with weather and terrain.
C_WEATHER, C_TERRAIN, C_WEATHER_SETTER = range(3)
SETTER_UNKNOWN, SETTER_ACTOR, SETTER_OTHER = 0, 1, 2
# ``side_cnt`` columns.
S_MEGA_USED, S_FAINTED, S_REVEALED, S_UNKNOWN_BROUGHT, S_TEAM_SIZE = range(5)

# ``cand_flag`` bits. The last five are properties of the move, by the dex.
CAND_VALID = 1
CAND_REVEALED = 2
CAND_SHEET = 4
CAND_REPERTOIRE = 8
CAND_GLOBAL = 16
CAND_PROTECT = 32
CAND_DAMAGING = 64
CAND_AIMED = 128
CAND_SPREAD = 256
CAND_FIRST_TURN = 512
_CLASS_BITS = CAND_PROTECT | CAND_DAMAGING | CAND_AIMED | CAND_SPREAD | CAND_FIRST_TURN

# Target classes as bits of ``cand_tmask`` (bit k = TARGET_CLASSES[k] is legal).
T_FOE_A = TARGET_CLASSES.index(TARGET_FOE_A)
T_FOE_B = TARGET_CLASSES.index(TARGET_FOE_B)
T_ALLY = TARGET_CLASSES.index(TARGET_ALLY)
T_SELF = TARGET_CLASSES.index(TARGET_SELF)
T_AUTO = TARGET_CLASSES.index(TARGET_AUTO)
_TARGET_INDEX = {name: index for index, name in enumerate(TARGET_CLASSES)}
_INTENT_INDEX = {name: index for index, name in enumerate(INTENT_CLASSES)}

# Label codes.
Y_KIND_ABSENT, Y_KIND_MOVE, Y_KIND_SWITCH, Y_KIND_HIDDEN, Y_KIND_NONE = -1, 0, 1, 2, 3
KIND_CODES = {
    KIND_MOVE: Y_KIND_MOVE,
    KIND_SWITCH: Y_KIND_SWITCH,
    KIND_HIDDEN: Y_KIND_HIDDEN,
    KIND_NONE: Y_KIND_NONE,
}
(Y_SWITCHED, Y_SWITCH_KNOWN, Y_PROTECTED, Y_PROTECT_KNOWN, Y_LOCKED, Y_OTHER) = range(6)
N_Y_FLAG = 6
# ``y_reason`` codes: 0 = none, then these, then a status condition, then other.
REASON_NAMES: tuple[str, ...] = (
    "",
    REASON_FAINTED_FIRST,
    REASON_LEFT_FIELD,
    REASON_GAME_ENDED,
    REASON_UNRESOLVED,
    REASON_UNEXPLAINED,
    REASON_FORCED,
    REASON_CONFUSION,
    REASON_FLINCH,
    REASON_OVERRIDDEN,
    "status",
    "other",
)
_REASON_INDEX = {name: index for index, name in enumerate(REASON_NAMES)}

ELO_KEEP = "keep"
ELO_BLANK = "blank"
ELO_SHUFFLE = "shuffle"
ELO_SWAP = "swap"
ELO_MODES: tuple[str, ...] = (ELO_KEEP, ELO_BLANK, ELO_SHUFFLE, ELO_SWAP)

PROBABILITY_FLOOR = 1e-6

# Sentinels of the repo's item / ability vocabularies (data/items.json and
# data/abilities.json as vgc_bench.src.utils loads them): index 0 is the
# out-of-vocabulary entry, '' is "none", and items have an "unknown" entry.
_OOV = "null"
_NO_ITEM = ""
_UNKNOWN_ITEM = "unknown_item"
# Markers public_state makes from the protocol's -prepare / -mustrecharge
# message types (a forced next action); they are not effect names of the dex.
_PROTOCOL_MARKERS = ("prepare", "mustrecharge")
# Dex target types the player aims, and what each can be aimed at.
_AIM_FOES = frozenset({"normal", "any", "adjacentFoe"})
_AIM_ALLY = frozenset({"normal", "any", "adjacentAlly", "adjacentAllyOrSelf"})
_AIM_SELF = frozenset({"adjacentAllyOrSelf"})
# Dex target types that hit every adjacent foe.
_HITS_ALL_FOES = frozenset({"allAdjacentFoes", "allAdjacent"})
_BASE_STATS = ("hp", "atk", "def", "spa", "spd", "spe")
_CATEGORIES = ("Physical", "Special", "Status")
_REPO_ROOT = Path(__file__).resolve().parents[3]


# --- dex helpers --------------------------------------------------------------


@lru_cache(maxsize=1)
def _species_dex() -> dict[str, Any]:
    """Species data by forme id, as ``GenData.from_gen(9).pokedex`` holds it."""
    try:
        return dict(GenData.from_gen(9).pokedex)
    except Exception:
        return {}


@lru_cache(maxsize=4096)
def set_key(forme: str, species: str = "") -> str:
    """The forme whose moveset a Pokemon carries: the key of the repertoire.

    A regional or appliance forme has its own moves, so it keys itself; a Mega
    or another battle-only forme carries the moves of the forme it came from; a
    cosmetic forme is its base. ``species`` is the fallback for a forme the dex
    does not know.
    """
    entry = _species_dex().get(forme)
    if entry is None:
        return species or forme
    battle_only = entry.get("battleOnly")
    if isinstance(battle_only, str) and battle_only:
        return to_id(battle_only)
    base = to_id(str(entry.get("baseSpecies") or ""))
    if battle_only or is_mega_forme(forme):
        return base or species or forme
    return to_id(str(entry.get("name") or "")) or forme


def _dex_ids(key: str) -> tuple[str, ...]:
    """Sorted ids of the values a move field takes across the dex."""
    found: set[str] = set()
    for entry in moves_dex().values():
        if entry.get("isZ") or entry.get("isMax"):
            continue
        value = entry.get(key)
        if isinstance(value, str) and value:
            found.add(to_id(value))
    return tuple(sorted(found))


def _dex_statuses() -> tuple[str, ...]:
    found: set[str] = set()
    for entry in moves_dex().values():
        effects = [
            entry,
            entry.get("secondary") or {},
            *(entry.get("secondaries") or []),
        ]
        for effect in effects:
            if isinstance(effect, dict) and isinstance(effect.get("status"), str):
                found.add(to_id(effect["status"]))
    return tuple(sorted(found))


def _dex_volatiles() -> tuple[str, ...]:
    """Effect ids a Pokemon can carry, as far as the move dex names them."""
    found: set[str] = set(_PROTOCOL_MARKERS)
    placed = ("sideCondition", "pseudoWeather", "weather", "terrain", "slotCondition")
    for move_id, entry in moves_dex().items():
        if entry.get("isZ") or entry.get("isMax"):
            continue
        secondary = entry.get("secondary") or {}
        sources = [entry, secondary, entry.get("self") or {}]
        if isinstance(secondary, dict):
            sources.append(secondary.get("self") or {})
        for source in sources:
            if isinstance(source, dict) and isinstance(
                source.get("volatileStatus"), str
            ):
                found.add(to_id(source["volatileStatus"]))
        if entry.get("condition") and not any(entry.get(key) for key in placed):
            found.add(move_id)
    return tuple(sorted(found))


def _dex_types() -> tuple[str, ...]:
    found: set[str] = set()
    for entry in _species_dex().values():
        for name in entry.get("types", ()):
            found.add(to_id(str(name)))
    for entry in moves_dex().values():
        if entry.get("type"):
            found.add(to_id(str(entry["type"])))
    return tuple(sorted(found))


def _dex_target_types() -> tuple[str, ...]:
    return tuple(
        sorted({str(e["target"]) for e in moves_dex().values() if e.get("target")})
    )


def _load_list(name: str) -> tuple[str, ...]:
    try:
        data = json.loads((_REPO_ROOT / "data" / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return (_OOV,)
    if not isinstance(data, list) or not data:
        return (_OOV,)
    return tuple(str(item) for item in data)


def _has_terrain(fields: Mapping[str, EffectStart]) -> bool:
    dex = moves_dex()
    return any((dex.get(effect) or {}).get("terrain") for effect in fields)


def has_terrain(snapshot: PublicSnapshot) -> bool:
    """Whether a terrain is up: the condition ``Repertoire`` keys aim statistics on."""
    try:
        return _has_terrain(snapshot.fields)
    except Exception:
        return False


# --- vocabulary ---------------------------------------------------------------


@dataclass
class Vocab:
    """Every id list an example indexes. Index 0 is "unknown" in the first four.

    ``species`` is ``<unknown>`` followed by the sorted forme ids of
    ``GenData.from_gen(9).pokedex``; ``moves`` / ``items`` / ``abilities`` are the
    repo's ``data/*.json`` lists unchanged (``moves[0]`` is "no move",
    ``items[0]`` and ``abilities[0]`` the out-of-vocabulary entry). The other
    lists are small categorical vocabularies read off the dex.
    """

    species: tuple[str, ...]
    moves: tuple[str, ...]
    items: tuple[str, ...]
    abilities: tuple[str, ...]
    types: tuple[str, ...]
    statuses: tuple[str, ...]
    weathers: tuple[str, ...]
    terrains: tuple[str, ...]
    pseudo: tuple[str, ...]
    side_conditions: tuple[str, ...]
    volatiles: tuple[str, ...]
    target_types: tuple[str, ...]

    _FIELDS = (
        "species",
        "moves",
        "items",
        "abilities",
        "types",
        "statuses",
        "weathers",
        "terrains",
        "pseudo",
        "side_conditions",
        "volatiles",
        "target_types",
    )

    @classmethod
    def build(cls) -> "Vocab":
        return cls(
            species=("<unknown>", *sorted(_species_dex())),
            moves=_load_list("moves.json"),
            items=_load_list("items.json"),
            abilities=_load_list("abilities.json"),
            types=_dex_types(),
            statuses=_dex_statuses(),
            weathers=_dex_ids("weather"),
            terrains=_dex_ids("terrain"),
            pseudo=_dex_ids("pseudoWeather"),
            side_conditions=_dex_ids("sideCondition"),
            volatiles=_dex_volatiles(),
            target_types=_dex_target_types(),
        )

    def to_dict(self) -> dict[str, list[str]]:
        return {name: list(getattr(self, name)) for name in self._FIELDS}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Vocab":
        return cls(**{name: tuple(data[name]) for name in cls._FIELDS})

    def sizes(self) -> dict[str, int]:
        """Embedding sizes: number of ids of each kind, index 0 included."""
        return {
            "species": len(self.species),
            "moves": len(self.moves),
            "items": len(self.items),
            "abilities": len(self.abilities),
            "types": len(self.types) + 2,  # 0 = none, last = other
            "statuses": len(self.statuses) + 2,
            "weathers": len(self.weathers) + 2,
            "terrains": len(self.terrains) + 2,
            "volatiles": len(self.volatiles) + N_VOLATILE_HASH + 1,
        }


def _index(names: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for position, name in enumerate(names):
        out.setdefault(name, position)
    return out


# --- numeric tables -----------------------------------------------------------


def species_columns(vocab: Vocab) -> list[str]:
    return [
        *(f"type:{name}" for name in vocab.types),
        *(f"base:{stat}" for stat in _BASE_STATS),
        "is_mega_forme",
        "has_mega_forme",
        "known",
    ]


def move_columns(vocab: Vocab) -> list[str]:
    return [
        *(f"sem:{index}" for index in range(MOVE_SEM_LEN)),
        *(f"category:{name.lower()}" for name in _CATEGORIES),
        *(f"target:{name}" for name in vocab.target_types),
        "priority",
        "priority_positive",
        "priority_negative",
        "base_power",
        "accuracy",
        "never_misses",
        *(f"type:{name}" for name in vocab.types),
        "damaging",
        "aimed",
        "spread",
        "protect_family",
        "stalling",
        "raises_stall_counter",
        "first_turn_only",
        "fake_out_like",
        "redirection",
        "known",
    ]


def _species_row(forme: str, vocab: Vocab, width: int) -> np.ndarray:
    row = np.zeros(width, dtype=np.float32)
    dex = _species_dex()
    entry = dex.get(forme)
    if entry is None:
        return row
    types = _index(vocab.types)
    for name in entry.get("types", ()):
        position = types.get(to_id(str(name)))
        if position is not None:
            row[position] = 1.0
    base = len(vocab.types)
    stats = entry.get("baseStats") or {}
    for offset, stat in enumerate(_BASE_STATS):
        row[base + offset] = float(stats.get(stat, 0)) / 255.0
    row[base + 6] = float(is_mega_forme(forme))
    row[base + 7] = float(
        any(is_mega_forme(to_id(str(name))) for name in entry.get("otherFormes", ()))
    )
    row[base + 8] = 1.0
    return row


def _move_class_bits(move_id: str) -> int:
    bits = 0
    if is_protect_family(move_id):
        bits |= CAND_PROTECT
    if is_damaging(move_id):
        bits |= CAND_DAMAGING
    if is_choosable_target(move_id):
        bits |= CAND_AIMED
    if is_spread(move_id):
        bits |= CAND_SPREAD
    if is_first_turn_only(move_id):
        bits |= CAND_FIRST_TURN
    return bits


def _move_row(move_id: str, vocab: Vocab, width: int) -> np.ndarray:
    row = np.zeros(width, dtype=np.float32)
    entry = moves_dex().get(move_id)
    if entry is None:
        return row
    row[:MOVE_SEM_LEN] = move_semantics(move_id)
    at = MOVE_SEM_LEN
    category = str(entry.get("category"))
    for offset, name in enumerate(_CATEGORIES):
        row[at + offset] = float(category == name)
    at += len(_CATEGORIES)
    target = str(entry.get("target"))
    for offset, name in enumerate(vocab.target_types):
        row[at + offset] = float(target == name)
    at += len(vocab.target_types)
    priority = float(entry.get("priority", 0) or 0)
    row[at] = priority / 5.0
    row[at + 1] = float(priority > 0)
    row[at + 2] = float(priority < 0)
    row[at + 3] = min(2.0, float(entry.get("basePower", 0) or 0) / 150.0)
    accuracy = entry.get("accuracy", True)
    row[at + 4] = 1.0 if accuracy is True else float(accuracy or 0) / 100.0
    row[at + 5] = float(accuracy is True)
    at += 6
    types = _index(vocab.types)
    position = types.get(to_id(str(entry.get("type") or "")))
    if position is not None:
        row[at + position] = 1.0
    at += len(vocab.types)
    row[at : at + 10] = (
        float(is_damaging(move_id)),
        float(bool(is_choosable_target(move_id))),
        float(is_spread(move_id)),
        float(is_protect_family(move_id)),
        float(is_stalling(move_id)),
        float(raises_stall_counter(move_id)),
        float(is_first_turn_only(move_id)),
        float(is_fake_out_like(move_id)),
        float(is_redirection(move_id)),
        1.0,
    )
    return row


def _move_intents(move_id: str) -> list[int]:
    out: list[int] = []
    for target in TARGET_CLASSES:
        intent = move_intent(move_id, target)
        out.append(-1 if intent is None else _INTENT_INDEX[intent])
    return out


@dataclass
class Tables:
    """Fixed per-id numerics, derived from the dex and stored with the artifact.

    Row i belongs to id i of the vocabulary; rows after the vocabulary are
    extension rows (see the module docstring). ``species_num`` is per FORME, so a
    Mega reads its own types and base stats. ``move_num`` holds the 84
    ``move_semantics`` floats and the columns named by ``move_columns``.
    ``move_intent[row, t]`` is the ``INTENT_CLASSES`` index of the move aimed at
    ``TARGET_CLASSES[t]`` (-1 = not decidable), and ``move_class`` the
    ``CAND_PROTECT`` .. ``CAND_FIRST_TURN`` bits.
    """

    species_num: np.ndarray
    move_num: np.ndarray
    move_intent: np.ndarray
    move_class: np.ndarray
    species_ids: tuple[str, ...]
    move_ids: tuple[str, ...]

    @classmethod
    def build(
        cls, vocab: Vocab, species_ids: Sequence[str], move_ids: Sequence[str]
    ) -> "Tables":
        species_width = len(species_columns(vocab))
        move_width = len(move_columns(vocab))
        species_num = np.zeros((len(species_ids), species_width), dtype=np.float32)
        for row, forme in enumerate(species_ids):
            species_num[row] = _species_row(forme, vocab, species_width)
        move_num = np.zeros((len(move_ids), move_width), dtype=np.float32)
        intents = np.full((len(move_ids), N_TARGET), -1, dtype=np.int8)
        classes = np.zeros(len(move_ids), dtype=np.uint16)
        for row, move_id in enumerate(move_ids):
            move_num[row] = _move_row(move_id, vocab, move_width)
            if move_id in moves_dex():
                intents[row] = _move_intents(move_id)
                classes[row] = _move_class_bits(move_id)
        return cls(
            species_num, move_num, intents, classes, tuple(species_ids), tuple(move_ids)
        )

    def extended(self, vocab: Vocab) -> "Tables":
        """These tables plus rows for ids the current dex has and they lack."""
        known_species = set(self.species_ids)
        known_moves = set(self.move_ids)
        new_species = sorted(k for k in _species_dex() if k not in known_species)
        new_moves = sorted(k for k in moves_dex() if k not in known_moves)
        if not new_species and not new_moves:
            return self
        extra = Tables.build(vocab, new_species, new_moves)
        return Tables(
            np.concatenate([self.species_num, extra.species_num]),
            np.concatenate([self.move_num, extra.move_num]),
            np.concatenate([self.move_intent, extra.move_intent]),
            np.concatenate([self.move_class, extra.move_class]),
            self.species_ids + extra.species_ids,
            self.move_ids + extra.move_ids,
        )

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "species_num": self.species_num,
            "move_num": self.move_num,
            "move_intent": self.move_intent,
            "move_class": self.move_class,
        }


# --- repertoire ---------------------------------------------------------------


@dataclass
class Repertoire:
    """How often each set key used each move, from TRAINING-split actions only.

    ``counts[key][move]`` is a weighted number of visible, freely chosen uses.
    ``aim[move]`` is ``[uses, auto]`` without a terrain and ``[uses, auto]`` with
    one, for moves the dex says are aimed: ``auto`` counts the uses the
    simulator turned into a spread hit, which is how a terrain-dependent spread
    move shows up without naming it.
    """

    counts: dict[str, dict[str, float]] = field(default_factory=dict)
    aim: dict[str, list[float]] = field(default_factory=dict)
    _ranked: dict[str, tuple[tuple[str, ...], dict[str, tuple[float, int]], float]] = (
        field(default_factory=dict, repr=False, compare=False)
    )
    _global: tuple[str, ...] | None = field(default=None, repr=False, compare=False)

    def add(
        self,
        key: str,
        move: str,
        weight: float = 1.0,
        *,
        auto: bool | None = None,
        terrain: bool = False,
    ) -> None:
        """Count one use. ``auto`` is given for an aimed move with a known target."""
        if not key or not move or weight <= 0:
            return
        moves = self.counts.setdefault(key, {})
        moves[move] = moves.get(move, 0.0) + weight
        if auto is not None:
            cell = self.aim.setdefault(move, [0.0, 0.0, 0.0, 0.0])
            base = 2 if terrain else 0
            cell[base] += weight
            if auto:
                cell[base + 1] += weight
        self._ranked.clear()
        self._global = None

    def lookup(
        self, key: str
    ) -> tuple[tuple[str, ...], dict[str, tuple[float, int]], float]:
        """(moves by falling frequency, move -> (frequency, rank from 1), support)."""
        cached = self._ranked.get(key)
        if cached is not None:
            return cached
        moves = self.counts.get(key) or {}
        total = float(sum(moves.values()))
        order = sorted(moves, key=lambda move: (-moves[move], move))
        table = {
            move: (moves[move] / total if total > 0 else 0.0, rank)
            for rank, move in enumerate(order, start=1)
        }
        result = (tuple(order), table, total)
        self._ranked[key] = result
        return result

    def global_ranked(self) -> tuple[str, ...]:
        """Every move by falling total use, for species the training split lacks."""
        if self._global is None:
            totals: dict[str, float] = {}
            for moves in self.counts.values():
                for move, count in moves.items():
                    totals[move] = totals.get(move, 0.0) + count
            self._global = tuple(sorted(totals, key=lambda m: (-totals[m], m)))
        return self._global

    def auto_seen(self, move: str) -> bool:
        cell = self.aim.get(move)
        return cell is not None and (cell[1] > 0 or cell[3] > 0)

    def auto_share(self, move: str, terrain: bool) -> float:
        """Share of an aimed move's uses that became a spread hit, by terrain."""
        cell = self.aim.get(move)
        if cell is None:
            return 0.0
        base = 2 if terrain else 0
        if cell[base] > 0:
            return cell[base + 1] / cell[base]
        total = cell[0] + cell[2]
        return (cell[1] + cell[3]) / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": {
                key: {move: round(count, 6) for move, count in sorted(moves.items())}
                for key, moves in sorted(self.counts.items())
            },
            "aim": {
                move: [round(v, 6) for v in cell]
                for move, cell in sorted(self.aim.items())
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Repertoire":
        counts = {
            str(key): {str(move): float(count) for move, count in moves.items()}
            for key, moves in (data.get("counts") or {}).items()
        }
        aim = {
            str(move): [float(value) for value in cell][:4]
            for move, cell in (data.get("aim") or {}).items()
            if len(cell) >= 4
        }
        return cls(counts=counts, aim=aim)


# --- layout -------------------------------------------------------------------


@dataclass(frozen=True)
class ArraySpec:
    """Shape (per example), dtype and meaning of one array of the layout."""

    shape: tuple[int, ...]
    dtype: str
    group: str  # 'feature' or 'label'
    doc: str


def action_size(n_cand: int = N_CAND_DEFAULT) -> int:
    """Entries of a slot's action space: candidates, OTHER, six switch pointers."""
    return n_cand + 1 + N_ROSTER


def other_index(n_cand: int = N_CAND_DEFAULT) -> int:
    return n_cand


def switch_index(roster_index: int, n_cand: int = N_CAND_DEFAULT) -> int:
    return n_cand + 1 + roster_index


def layout(
    n_cand: int = N_CAND_DEFAULT, n_pseudo: int = 0, n_side_conditions: int = 0
) -> dict[str, ArraySpec]:
    """The documented arrays of one example: name -> shape, dtype, meaning.

    ``Featurizer.layout()`` fills in the two vocabulary-dependent widths. A
    batch has the same arrays with a leading axis. Mon rows 0-5 are the actor's
    roster, 6-11 the other side's; slot rows are the actor's slots ``a``, ``b``.
    """
    a = action_size(n_cand)
    f, lab = "feature", "label"
    return {
        "turn": ArraySpec((), "uint8", f, "turn number, capped at 255"),
        "ctx_cat": ArraySpec(
            (3,),
            "uint8",
            f,
            "weather id, terrain id (0 none, last other), "
            "weather setter (0 unknown, 1 actor, 2 other)",
        ),
        "field_age": ArraySpec(
            (2 + n_pseudo,),
            "uint8",
            f,
            "weather, terrain, each pseudo-weather: 0 absent, else 1 + "
            "end-of-turn ticks lived (capped)",
        ),
        "game_flag": ArraySpec(
            (4,),
            "uint8",
            f,
            "bo3, ladder-rated, actor sheet, other sheet (0 closed, 1 open, 2 unknown)",
        ),
        "elo": ArraySpec((2,), "int16", f, "rating of actor, other; 0 when unknown"),
        "elo_known": ArraySpec((2,), "uint8", f, "1 when that rating is known"),
        "side_cnt": ArraySpec(
            (2, 5),
            "uint8",
            f,
            "mega used, fainted, revealed, 1 + unknown brought (0 unknown), "
            "team size (0 unknown)",
        ),
        "side_age": ArraySpec(
            (2, n_side_conditions),
            "uint8",
            f,
            "each side condition: 0 absent, else 1 + ticks lived (capped)",
        ),
        "mon_id": ArraySpec(
            (N_MON, N_MON_ID),
            "uint16",
            f,
            "species, forme, set key (species ids); item; ability; last move",
        ),
        "mon_move": ArraySpec((N_MON, MAX_MOVES), "uint16", f, "known move ids"),
        "mon_move_flag": ArraySpec(
            (N_MON, MAX_MOVES), "uint8", f, "bit 1 revealed in battle, 2 from sheet"
        ),
        "mon_hp": ArraySpec((N_MON,), "float16", f, "public HP fraction"),
        "mon_boost": ArraySpec((N_MON, len(BOOST_STATS)), "int8", f, "stat stages"),
        "mon_cat": ArraySpec(
            (N_MON, N_MON_CAT),
            "uint8",
            f,
            "status, slot (0 bench, 1 a, 2 b), item state, brought (0 no, 1 yes, "
            "2 unknown), last action kind (0 none, 1 move, 2 switch, 3 hidden, "
            "4 no action), last target (0 n/a, 1 + class), protect streak, "
            "turns on field, known moves",
        ),
        "mon_flag": ArraySpec(
            (N_MON, N_MON_FLAG),
            "uint8",
            f,
            "present, revealed, fainted, is mega, mega possible, first turn, "
            "protected last turn, ability known, last: mega, locked, protect",
        ),
        "mon_type": ArraySpec(
            (N_MON, MAX_TYPES), "uint8", f, "current type ids (0 none, last other)"
        ),
        "mon_vol": ArraySpec(
            (N_MON, MAX_VOLATILES), "uint8", f, "volatile ids (0 none)"
        ),
        "act_mon": ArraySpec((2,), "int8", f, "mon row of the actor's slot, -1 empty"),
        "foe_mon": ArraySpec((2,), "int8", f, "mon row of the foe's slot, -1 empty"),
        "slot_support": ArraySpec(
            (2,), "float16", f, "log1p of the set key's weighted training uses"
        ),
        "cand_move": ArraySpec((2, n_cand), "uint16", f, "candidate move ids"),
        "cand_flag": ArraySpec((2, n_cand), "uint16", f, "CAND_* bits"),
        "cand_prior": ArraySpec(
            (2, n_cand), "float16", f, "use frequency of the move for the set key"
        ),
        "cand_rank": ArraySpec(
            (2, n_cand), "uint8", f, "rank in the set key's repertoire, 0 unseen"
        ),
        "cand_auto": ArraySpec(
            (2, n_cand),
            "float16",
            f,
            "share of uses that hit as a spread move in this terrain state",
        ),
        "cand_tmask": ArraySpec(
            (2, n_cand + 1),
            "uint8",
            f,
            "legal target classes of each candidate and of OTHER, as bits",
        ),
        "other_prior": ArraySpec(
            (2,), "float16", f, "repertoire mass outside the candidates"
        ),
        "switch_mask": ArraySpec((2, N_ROSTER), "uint8", f, "legal switch targets"),
        "action_mask": ArraySpec((2, a), "uint8", f, "legal actions of the slot"),
        "y_kind": ArraySpec(
            (2,), "int8", lab, "-1 no record, 0 move, 1 switch, 2 hidden, 3 none"
        ),
        "y_action": ArraySpec(
            (2,), "int8", lab, "action index of a visible free choice, else -1"
        ),
        "y_set": ArraySpec(
            (2, a), "uint8", lab, "actions the log is consistent with; all 0 = none"
        ),
        "y_target": ArraySpec(
            (2,), "int8", lab, "target class of a visible move when certain, else -1"
        ),
        "y_mega": ArraySpec((2,), "int8", lab, "1 / 0, -1 when the log cannot tell"),
        "y_intent": ArraySpec((2,), "int8", lab, "INTENT_CLASSES index, else -1"),
        "y_flag": ArraySpec(
            (2, N_Y_FLAG),
            "uint8",
            lab,
            "switched, switch known, protected, protect known, locked, "
            "true move outside the candidates",
        ),
        "y_attack": ArraySpec(
            (2, 2),
            "int8",
            lab,
            "per foe slot a, b: 1 a damaging move was aimed there or spread "
            "over it, 0 not, -1 unknown",
        ),
        "y_reason": ArraySpec((2,), "uint8", lab, "REASON_NAMES index"),
    }


# --- pure batch helpers -------------------------------------------------------


# Failures of the module-level helpers a runtime may call, by helper and exception.
COUNTERS: Counter[str] = Counter()


def _failed(name: str, exc: Exception) -> None:
    try:
        COUNTERS[f"{name}:{type(exc).__name__}"] += 1
    except Exception:
        pass


def collate(examples: Sequence[Mapping[str, np.ndarray]]) -> Batch:
    """Stack examples into a batch (leading axis N). Never raises.

    Only arrays present in every example are kept; an empty or unstackable
    input gives an empty dict.
    """
    try:
        rows = [example for example in examples if example]
        if not rows:
            return {}
        names = set(rows[0])
        for row in rows[1:]:
            names &= set(row)
        return {
            name: np.stack([np.asarray(row[name]) for row in rows])
            for name in sorted(names)
        }
    except Exception:
        return {}


def concat_batches(batches: Sequence[Mapping[str, np.ndarray]]) -> Batch:
    """Join batches along the example axis, keeping arrays all of them have."""
    rows = [batch for batch in batches if batch]
    if not rows:
        return {}
    names = set(rows[0])
    for row in rows[1:]:
        names &= set(row)
    return {name: np.concatenate([row[name] for row in rows]) for name in sorted(names)}


def take(batch: Mapping[str, np.ndarray], index: Any) -> Batch:
    """Rows ``index`` (indices or a boolean mask) of every array of a batch."""
    return {name: array[index] for name, array in batch.items()}


def save_batch(path: Path | str, batch: Mapping[str, np.ndarray]) -> None:
    """Write one shard (compressed npz), through a temporary file."""
    target = Path(path)
    scratch = target.with_name(target.name + ".tmp.npz")
    arrays: dict[str, Any] = {name: np.asarray(a) for name, a in batch.items()}
    np.savez_compressed(scratch, **arrays)
    scratch.replace(target)


def load_batch(path: Path | str) -> Batch:
    with np.load(Path(path), allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def load_dataset(
    directory: Path | str, splits: Sequence[str] | None = None
) -> tuple[Batch, dict[str, Any]]:
    """Every shard of a built dataset as one batch, and its manifest.

    ``splits`` keeps only the examples of those split names (the manifest's
    ``splits`` list gives the codes of ``m_split``). An offline helper: raises
    ``OSError`` / ``ValueError`` when the directory is not a finished dataset.
    """
    source = Path(directory)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    names = list(manifest.get("splits") or [])
    codes = None if splits is None else [names.index(name) for name in splits]
    parts: list[Batch] = []
    for shard in manifest.get("shards") or []:
        batch = load_batch(source / str(shard["file"]))
        if codes is not None:
            batch = take(batch, np.isin(batch["m_split"], codes))
        parts.append(batch)
    return concat_batches(parts), manifest


def expand_target_mask(cand_tmask: np.ndarray) -> np.ndarray:
    """``cand_tmask`` bits as a boolean array with a trailing target axis."""
    bits = np.arange(N_TARGET, dtype=np.uint8)
    return ((np.asarray(cand_tmask)[..., None] >> bits) & 1).astype(bool)


def apply_elo_mode(
    batch: Mapping[str, np.ndarray], mode: str, rng: np.random.Generator | None = None
) -> Batch:
    """A batch with only its Elo fields changed; every other array is shared.

    ``keep`` returns the fields as they are, ``blank`` sets both ratings to
    unknown, ``shuffle`` gives every example the (actor, other) ratings of
    another example of the batch, ``swap`` exchanges actor and other. Never
    raises: an unknown mode or a batch without Elo fields comes back unchanged.
    """
    out = dict(batch)
    try:
        if mode == ELO_KEEP or "elo" not in batch or "elo_known" not in batch:
            return out
        elo, known = np.asarray(batch["elo"]), np.asarray(batch["elo_known"])
        if mode == ELO_BLANK:
            out["elo"], out["elo_known"] = np.zeros_like(elo), np.zeros_like(known)
        elif mode == ELO_SWAP:
            out["elo"], out["elo_known"] = (
                elo[..., ::-1].copy(),
                known[..., ::-1].copy(),
            )
        elif mode == ELO_SHUFFLE and elo.ndim == 2:
            generator = rng if rng is not None else np.random.default_rng(0)
            order = generator.permutation(elo.shape[0])
            out["elo"], out["elo_known"] = elo[order], known[order]
    except Exception:
        return dict(batch)
    return out


def sheet_unknown_as_closed(batch: Mapping[str, np.ndarray]) -> Batch:
    """A batch whose "unknown" sheet codes read "closed"; nothing else changes.

    A spectator log always tells whether a sheet was shown, so no training
    example carries ``SHEET_UNKNOWN``; a player-view stream whose sheet state
    nobody recorded (most saved own games, and the live bot until it hands the
    sheets over) carries nothing else. A model that was not trained with that
    code masked in must be fed through this, at evaluation and at serve time
    alike. Never raises.
    """
    out = dict(batch)
    try:
        flags = np.asarray(batch["game_flag"]).copy()
        for column in (G_ACTOR_SHEET, G_OTHER_SHEET):
            flags[..., column] = np.where(
                flags[..., column] == SHEET_UNKNOWN, SHEET_CLOSED, flags[..., column]
            )
        out["game_flag"] = flags
    except Exception:
        return dict(batch)
    return out


def _slot_take(batch: Mapping[str, np.ndarray], name: str, column: int) -> np.ndarray:
    rows = np.maximum(np.asarray(batch["act_mon"]), 0).astype(np.int64)
    values = np.asarray(batch[name])[..., column]  # [N, 12]
    return np.take_along_axis(values, rows, axis=1)


def slot_view(batch: Mapping[str, np.ndarray]) -> Batch:
    """Per-slot columns of the acting Pokemon, each ``[N, 2]``, for count tables.

    ``active`` says whether the slot holds a Pokemon; the other arrays are 0
    where it does not. Keys: ``species``, ``forme``, ``key`` (set-key id),
    ``turn1``, ``first_turn``, ``protected_last``, ``mega_possible``,
    ``status``, ``elo`` (the actor's rating) and ``elo_known``. Never raises:
    a batch without the needed arrays gives an empty dict (see ``COUNTERS``).
    """
    try:
        return _slot_view(batch)
    except Exception as exc:
        _failed("slot_view", exc)
        return {}


def _slot_view(batch: Mapping[str, np.ndarray]) -> Batch:
    active = np.asarray(batch["act_mon"]) >= 0
    turn = np.asarray(batch["turn"])

    def column(name: str, index: int) -> np.ndarray:
        return np.where(active, _slot_take(batch, name, index), 0)

    return {
        "active": active,
        "species": column("mon_id", ID_SPECIES),
        "forme": column("mon_id", ID_FORME),
        "key": column("mon_id", ID_KEY),
        "turn1": np.broadcast_to((turn == 1)[:, None], active.shape) & active,
        "first_turn": column("mon_flag", FLAG_FIRST_TURN).astype(bool),
        "protected_last": column("mon_flag", FLAG_PROTECTED_LAST).astype(bool),
        "mega_possible": column("mon_flag", FLAG_MEGA_POSSIBLE).astype(bool),
        "status": column("mon_cat", CAT_STATUS),
        "elo": np.broadcast_to(np.asarray(batch["elo"])[:, :1], active.shape),
        "elo_known": np.broadcast_to(
            np.asarray(batch["elo_known"])[:, :1].astype(bool), active.shape
        ),
    }


# --- predictor contract and scoring -------------------------------------------


@runtime_checkable
class Predictor(Protocol):
    """What count tables and the neural model both implement.

    ``predict(batch)`` takes a batch of feature arrays (labels, if present, must
    not be read) and returns probabilities:

    * ``action`` ``[N, 2, A]``: per slot, over candidates, OTHER and the six
      switch pointers (``A = n_cand + 7``); mass belongs on ``action_mask``.
    * ``target`` ``[N, 2, n_cand + 1, 5]``: per move action, over
      ``TARGET_CLASSES``, given that move; mass belongs on ``cand_tmask``.
    * ``mega`` ``[N, 2]``: the slot Mega-evolves this turn.
    """

    def predict(self, batch: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]: ...


def uniform_prediction(batch: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The floor every model must beat: uniform over what is legal.

    Never raises: a batch without masks gives an empty dict (see ``COUNTERS``).
    """
    try:
        return _uniform_prediction(batch)
    except Exception as exc:
        _failed("uniform_prediction", exc)
        return {}


def _uniform_prediction(batch: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    mask = np.asarray(batch["action_mask"], dtype=np.float64)
    total = mask.sum(-1, keepdims=True)
    action = np.divide(mask, total, out=np.zeros_like(mask), where=total > 0)
    legal = expand_target_mask(batch["cand_tmask"]).astype(np.float64)
    count = legal.sum(-1, keepdims=True)
    target = np.divide(legal, count, out=np.zeros_like(legal), where=count > 0)
    return {
        "action": action,
        "target": target,
        "mega": np.full(mask.shape[:2], 0.5, dtype=np.float64),
    }


class UniformPredictor:
    """``Predictor`` wrapper of ``uniform_prediction``."""

    def predict(self, batch: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        return uniform_prediction(batch)


def _checked(
    pred: Mapping[str, np.ndarray], batch: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.asarray(batch["action_mask"]).astype(bool)
    legal = expand_target_mask(batch["cand_tmask"])
    action = np.asarray(pred["action"], dtype=np.float64)
    target = np.asarray(pred["target"], dtype=np.float64)
    mega = np.asarray(pred["mega"], dtype=np.float64)
    if action.shape != mask.shape:
        raise ValueError(f"action {action.shape} does not match mask {mask.shape}")
    if target.shape != legal.shape:
        raise ValueError(f"target {target.shape} does not match mask {legal.shape}")
    if mega.shape != mask.shape[:2]:
        raise ValueError(f"mega {mega.shape} does not match slots {mask.shape[:2]}")
    if not (np.isfinite(action).all() and np.isfinite(target).all()):
        raise ValueError("prediction holds a non-finite probability")
    return action, target, mega


def normalize_prediction(
    pred: Mapping[str, np.ndarray], batch: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """A prediction with illegal mass removed and every row renormalised.

    A slot or move with no legal entry gets zeros. Scoring always goes through
    this, so two models are compared on the same support.
    """
    action, target, mega = _checked(pred, batch)
    mask = np.asarray(batch["action_mask"]).astype(bool)
    legal = expand_target_mask(batch["cand_tmask"])
    action = np.where(mask, np.maximum(action, 0.0), 0.0)
    total = action.sum(-1, keepdims=True)
    action = np.divide(action, total, out=np.zeros_like(action), where=total > 0)
    target = np.where(legal, np.maximum(target, 0.0), 0.0)
    total = target.sum(-1, keepdims=True)
    target = np.divide(target, total, out=np.zeros_like(target), where=total > 0)
    return {"action": action, "target": target, "mega": np.clip(mega, 0.0, 1.0)}


def slot_nll(
    pred: Mapping[str, np.ndarray],
    batch: Mapping[str, np.ndarray],
    floor: float = PROBABILITY_FLOOR,
) -> dict[str, np.ndarray]:
    """Per-slot negative log-likelihoods, each ``[N, 2]`` with a ``*_scored`` mask.

    * ``action``: ``-log`` of the mass on ``y_set`` (one action for a visible
      choice, the censored set otherwise). ``censored`` marks the set-scored
      slots and ``informative`` those whose set leaves a legal action out.
    * ``target``: ``-log p(target | move)`` for a visible move whose target is
      certain; 0 for a move that is not aimed.
    * ``fine``: ``action`` plus ``target`` where the target is scored: the
      design's fine action (move x target, or switch destination).
    * ``mega``: binary cross-entropy where the log shows whether a Pokemon that
      could Mega-evolve did.

    Probabilities are floored at ``floor`` so a label a model gave zero mass
    costs ``-log(floor)``, never infinity. Unscored entries are 0.
    """
    norm = normalize_prediction(pred, batch)
    mask = np.asarray(batch["action_mask"]).astype(bool)
    y_set = np.asarray(batch["y_set"]).astype(bool) & mask
    y_action = np.asarray(batch["y_action"]).astype(np.int64)
    y_target = np.asarray(batch["y_target"]).astype(np.int64)
    y_mega = np.asarray(batch["y_mega"]).astype(np.int64)
    n_move = norm["target"].shape[2]

    scored = y_set.any(-1)
    mass = (norm["action"] * y_set).sum(-1)
    action_nll = np.where(scored, -np.log(np.maximum(mass, floor)), 0.0)

    target_scored = scored & (y_action >= 0) & (y_action < n_move) & (y_target >= 0)
    row = np.clip(y_action, 0, n_move - 1)[..., None, None]
    chosen = np.take_along_axis(norm["target"], row, axis=2)[:, :, 0, :]
    picked = np.take_along_axis(chosen, np.clip(y_target, 0, None)[..., None], axis=2)
    target_nll = np.where(
        target_scored, -np.log(np.maximum(picked[..., 0], floor)), 0.0
    )

    possible = np.asarray(_slot_view(batch)["mega_possible"]).astype(bool)
    mega_scored = (y_mega >= 0) & possible
    p_mega = np.clip(norm["mega"], floor, 1.0 - floor)
    mega_nll = np.where(
        mega_scored, -np.where(y_mega == 1, np.log(p_mega), np.log(1.0 - p_mega)), 0.0
    )
    return {
        "action": action_nll,
        "action_scored": scored,
        "censored": scored & (y_action < 0),
        "informative": scored & (y_set != mask).any(-1),
        "target": target_nll,
        "target_scored": target_scored,
        "fine": action_nll + target_nll,
        "fine_scored": scored,
        "mega": mega_nll,
        "mega_scored": mega_scored,
    }


def fine_probs(
    pred: Mapping[str, np.ndarray], batch: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Joint distribution over fine actions, ``[N, 2, (n_cand + 1) * 5 + 6]``.

    Entry ``m * 5 + t`` is move action ``m`` aimed at ``TARGET_CLASSES[t]``; the
    last six entries are the switch pointers.
    """
    norm = normalize_prediction(pred, batch)
    n_move = norm["target"].shape[2]
    moves = norm["action"][:, :, :n_move, None] * norm["target"]
    shape = moves.shape[:2]
    return np.concatenate(
        [moves.reshape(*shape, n_move * N_TARGET), norm["action"][:, :, n_move:]], -1
    )


def fine_label(batch: Mapping[str, np.ndarray]) -> np.ndarray:
    """Index into ``fine_probs`` of each visible, fully known action; else -1."""
    y_action = np.asarray(batch["y_action"]).astype(np.int64)
    y_target = np.asarray(batch["y_target"]).astype(np.int64)
    n_move = np.asarray(batch["cand_tmask"]).shape[-1]
    is_move = (y_action >= 0) & (y_action < n_move)
    out = np.full(y_action.shape, -1, dtype=np.int64)
    known = is_move & (y_target >= 0)
    out[known] = y_action[known] * N_TARGET + y_target[known]
    switch = y_action >= n_move
    out[switch] = n_move * N_TARGET + (y_action[switch] - n_move)
    return out


def topk_hits(
    probs: np.ndarray, labels: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """(label is among the k most probable entries, label is scored), per slot."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64)
    scored = labels >= 0
    order = np.argsort(-probs, axis=-1, kind="stable")[..., : max(1, k)]
    hits = (order == np.clip(labels, 0, None)[..., None]).any(-1) & scored
    return hits, scored


def event_probs(
    pred: Mapping[str, np.ndarray], batch: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Probabilities of the events a guard reads, from one prediction.

    ``switch`` and ``protect`` (keyed by the intent names) are ``[N, 2]``.
    ``attack`` is ``[N, 2, 2]``: the actor's slot aims a damaging move at foe
    slot a / b, or spreads one over it. The OTHER bucket counts as neither a
    Protect nor an attack. Never raises: a prediction that does not fit the
    batch gives an empty dict (see ``COUNTERS``).
    """
    try:
        return _event_probs(pred, batch)
    except Exception as exc:
        _failed("event_probs", exc)
        return {}


def _event_probs(
    pred: Mapping[str, np.ndarray], batch: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    norm = normalize_prediction(pred, batch)
    flags = np.asarray(batch["cand_flag"]).astype(np.int64)
    n_cand = flags.shape[-1]
    p_move = norm["action"][:, :, :n_cand]
    valid = (flags & CAND_VALID) > 0
    damaging = valid & ((flags & CAND_DAMAGING) > 0)
    aimed = (flags & CAND_AIMED) > 0
    spread = (flags & CAND_SPREAD) > 0
    target = norm["target"][:, :, :n_cand]
    foe_up = np.asarray(batch["foe_mon"]) >= 0  # [N, 2]
    attack = np.zeros((*p_move.shape[:2], 2), dtype=np.float64)
    for foe, column in ((0, T_FOE_A), (1, T_FOE_B)):
        hits = np.where(aimed, target[..., column] + target[..., T_AUTO], spread)
        mass = (p_move * damaging * hits).sum(-1)
        attack[:, :, foe] = np.where(foe_up[:, foe : foe + 1], mass, 0.0)
    protect = (p_move * (valid & ((flags & CAND_PROTECT) > 0))).sum(-1)
    return {
        INTENT_SWITCH: norm["action"][:, :, n_cand + 1 :].sum(-1),
        INTENT_PROTECT: protect,
        "attack": attack,
        "mega": norm["mega"],
    }


def intent_probs(
    pred: Mapping[str, np.ndarray], batch: Mapping[str, np.ndarray], tables: Tables
) -> np.ndarray | None:
    """``[N, 2, 8]``: the seven ``INTENT_CLASSES`` and the unassigned mass.

    Unassigned is the OTHER bucket plus any move whose intent the dex cannot
    decide for the target class. Call it on a batch that has NOT been through
    ``densify`` (extension rows keep their intent only before the clamp).
    Never raises: None when the prediction does not fit (see ``COUNTERS``).
    """
    try:
        return _intent_probs(pred, batch, tables)
    except Exception as exc:
        _failed("intent_probs", exc)
        return None


def _intent_probs(
    pred: Mapping[str, np.ndarray], batch: Mapping[str, np.ndarray], tables: Tables
) -> np.ndarray:
    norm = normalize_prediction(pred, batch)
    moves = np.asarray(batch["cand_move"]).astype(np.int64)
    n_cand = moves.shape[-1]
    if moves.size and int(moves.max()) >= tables.move_intent.shape[0]:
        raise ValueError("candidate move id outside the tables")
    intent = tables.move_intent[moves]  # [N, 2, n_cand, 5]
    joint = norm["action"][:, :, :n_cand, None] * norm["target"][:, :, :n_cand]
    out = np.zeros((*joint.shape[:2], N_INTENT + 1), dtype=np.float64)
    for index in range(N_INTENT):
        out[:, :, index] = (joint * (intent == index)).sum((-1, -2))
    out[:, :, _INTENT_INDEX[INTENT_SWITCH]] += norm["action"][:, :, n_cand + 1 :].sum(
        -1
    )
    out[:, :, N_INTENT] = np.maximum(0.0, norm["action"].sum(-1) - out.sum(-1))
    return out


def signature_diff(saved: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """Names of the ``events.dex_signature()`` entries that differ."""
    return sorted(
        key for key in set(saved) | set(current) if saved.get(key) != current.get(key)
    )


# --- featurizer ---------------------------------------------------------------


@dataclass
class _SideBlock:
    """One side's encoded rows, shared by the examples of both actors."""

    ids: list[list[int]]
    moves: list[list[int]]
    move_flags: list[list[int]]
    hp: list[float]
    boosts: list[list[int]]
    cats: list[list[int]]
    flags: list[list[int]]
    types: list[list[int]]
    vols: list[list[int]]
    cnt: list[int]
    age: list[int]


@dataclass(frozen=True)
class _MoveInfo:
    row: int
    bits: int
    target_type: str
    self_too: bool  # an aimed move some users cast on themselves instead


def _age(start: EffectStart | None, turn: int) -> int:
    if start is None:
        return 0
    return 1 + min(MAX_AGE - 1, max(0, start.ticks(turn)))


class Featurizer:
    """Encode snapshots and labels; own the vocabulary, tables and repertoire.

    Build one with ``Featurizer.build(repertoire)`` (vocabulary and tables from
    the installed dex) or ``Featurizer.load(directory)`` /
    ``Featurizer.from_payload`` (what a dataset or artifact stored).
    ``signature_diff`` then lists how the stored ``events.dex_signature()``
    differs from the current one; a caller that serves a model must refuse or
    warn when it is not empty. ``counters`` names everything unusual that was
    met. ``elo_mode`` is ``keep`` or ``blank`` (an Elo-blind model's runtime).
    """

    def __init__(
        self,
        vocab: Vocab,
        tables: Tables,
        repertoire: Repertoire | None = None,
        n_cand: int = N_CAND_DEFAULT,
        elo_mode: str = ELO_KEEP,
        signature: Mapping[str, Any] | None = None,
    ) -> None:
        self.vocab = vocab
        self.repertoire = repertoire if repertoire is not None else Repertoire()
        self.n_cand = max(1, int(n_cand))
        self.elo_mode = elo_mode if elo_mode in (ELO_KEEP, ELO_BLANK) else ELO_KEEP
        self.counters: Counter[str] = Counter()
        self.signature: dict[str, Any] = dict(signature or dex_signature())
        self.signature_diff: list[str] = signature_diff(self.signature, dex_signature())
        self.tables = tables.extended(vocab)
        self.n_species = len(vocab.species)
        self.n_moves = len(vocab.moves)
        self._species_ix = _index(self.tables.species_ids)
        self._move_ix = _index(self.tables.move_ids)
        self._item_ix = _index(vocab.items)
        self._ability_ix = _index(vocab.abilities)
        self._type_ix = _index(vocab.types)
        self._status_ix = _index(vocab.statuses)
        self._weather_ix = _index(vocab.weathers)
        self._terrain_ix = _index(vocab.terrains)
        self._pseudo_ix = _index(vocab.pseudo)
        self._side_ix = _index(vocab.side_conditions)
        self._vol_ix = _index(vocab.volatiles)
        self._item_unknown = self._item_ix.get(_UNKNOWN_ITEM, 0)
        self._item_none = self._item_ix.get(_NO_ITEM, 0)
        self._moves: dict[str, _MoveInfo] = {}
        dex = moves_dex()
        for move_id, row in self._move_ix.items():
            entry = dex.get(move_id)
            self._moves[move_id] = _MoveInfo(
                row,
                int(self.tables.move_class[row]),
                str(entry.get("target")) if entry else "",
                # Dex field: the target type for a user that is not a Ghost.
                bool(entry) and (entry or {}).get("nonGhostTarget") == "self",
            )
        self._blocks: tuple[PublicSnapshot, dict[str, _SideBlock]] | None = None

    # --- construction and storage ------------------------------------------

    @classmethod
    def build(
        cls,
        repertoire: Repertoire | None = None,
        n_cand: int = N_CAND_DEFAULT,
        elo_mode: str = ELO_KEEP,
    ) -> "Featurizer":
        vocab = Vocab.build()
        tables = Tables.build(vocab, vocab.species, vocab.moves)
        return cls(vocab, tables, repertoire, n_cand, elo_mode)

    def to_payload(self) -> dict[str, Any]:
        """Everything needed to rebuild this featurizer, as plain data + arrays.

        Only the vocabulary's own table rows are stored: extension rows are
        recomputed from whatever dex is installed at load time.
        """
        return {
            "layout_version": LAYOUT_VERSION,
            "n_cand": self.n_cand,
            "vocab": self.vocab.to_dict(),
            "dex_signature": dict(self.signature),
            "species_columns": species_columns(self.vocab),
            "move_columns": move_columns(self.vocab),
            "repertoire": self.repertoire.to_dict(),
            "tables": {
                "species_num": self.tables.species_num[: self.n_species].copy(),
                "move_num": self.tables.move_num[: self.n_moves].copy(),
                "move_intent": self.tables.move_intent[: self.n_moves].copy(),
                "move_class": self.tables.move_class[: self.n_moves].copy(),
            },
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], elo_mode: str = ELO_KEEP, strict: bool = False
    ) -> "Featurizer":
        """Rebuild from ``to_payload()``. Raises ``ValueError`` on a bad payload.

        ``strict`` also refuses a payload whose dex signature differs from the
        installed dex; without it the difference is left in ``signature_diff``.
        """
        made = cls._from_payload(payload, elo_mode)
        if strict and made.signature_diff:
            raise ValueError(f"dex signature differs: {made.signature_diff}")
        return made

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any], elo_mode: str) -> "Featurizer":
        try:
            version = int(payload["layout_version"])
            if version != LAYOUT_VERSION:
                raise ValueError(f"layout version {version}, code is {LAYOUT_VERSION}")
            vocab = Vocab.from_dict(payload["vocab"])
            stored = payload["tables"]
            tables = Tables(
                np.asarray(stored["species_num"], dtype=np.float32),
                np.asarray(stored["move_num"], dtype=np.float32),
                np.asarray(stored["move_intent"], dtype=np.int8),
                np.asarray(stored["move_class"], dtype=np.uint16),
                vocab.species,
                vocab.moves,
            )
            if tables.species_num.shape != (
                len(vocab.species),
                len(species_columns(vocab)),
            ) or tables.move_num.shape != (len(vocab.moves), len(move_columns(vocab))):
                raise ValueError("stored tables do not match the stored vocabulary")
            return cls(
                vocab,
                tables,
                Repertoire.from_dict(payload.get("repertoire") or {}),
                int(payload["n_cand"]),
                elo_mode,
                payload.get("dex_signature"),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"not a featurizer payload: {exc!r}") from exc

    def save(self, directory: Path | str) -> None:
        """Write ``vocab.json``, ``tables.npz`` and ``repertoire.json``."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        payload = self.to_payload()
        tables = payload.pop("tables")
        repertoire = payload.pop("repertoire")
        (out / "vocab.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
        (out / "repertoire.json").write_text(
            json.dumps(repertoire, indent=1, sort_keys=True), encoding="utf-8"
        )
        save_batch(out / "tables.npz", tables)

    @classmethod
    def load(
        cls, directory: Path | str, elo_mode: str = ELO_KEEP, strict: bool = False
    ) -> "Featurizer":
        """Read what ``save`` wrote. Raises ``ValueError`` / ``OSError``."""
        source = Path(directory)
        payload = json.loads((source / "vocab.json").read_text(encoding="utf-8"))
        payload["repertoire"] = json.loads(
            (source / "repertoire.json").read_text(encoding="utf-8")
        )
        payload["tables"] = load_batch(source / "tables.npz")
        return cls.from_payload(payload, elo_mode, strict)

    def layout(self) -> dict[str, ArraySpec]:
        return layout(
            self.n_cand, len(self.vocab.pseudo), len(self.vocab.side_conditions)
        )

    @property
    def n_actions(self) -> int:
        return action_size(self.n_cand)

    # --- encoding ----------------------------------------------------------

    def encode(self, snapshot: PublicSnapshot, actor_side: str) -> Example | None:
        """Feature arrays for ``actor_side`` at this turn start, or None.

        None (with a named count) when there is nothing to predict: no dex, an
        unknown side, or no Pokemon of the actor on the field. Never raises.
        """
        try:
            return self._encode(snapshot, actor_side)
        except Exception as exc:
            self._count(f"encode_error:{type(exc).__name__}")
            return None

    def encode_labels(
        self,
        example: Mapping[str, np.ndarray],
        snapshot: PublicSnapshot,
        actor_side: str,
        actions: Mapping[str, SlotAction],
    ) -> Example | None:
        """Label arrays for the same turn, from ``TurnRecord.actions``. Never raises."""
        try:
            return self._encode_labels(example, snapshot, actor_side, actions)
        except Exception as exc:
            self._count(f"label_error:{type(exc).__name__}")
            return None

    def encode_turn(
        self,
        snapshot: PublicSnapshot,
        actions: Mapping[str, SlotAction],
        actor_side: str,
    ) -> Example | None:
        """Features and labels of one actor's turn in one dict, or None."""
        example = self.encode(snapshot, actor_side)
        if example is None:
            return None
        labels = self.encode_labels(example, snapshot, actor_side, actions)
        if labels is None:
            return None
        example.update(labels)
        return example

    def collate(self, examples: Sequence[Mapping[str, np.ndarray]]) -> Batch:
        batch = collate(examples)
        if not batch and len(examples) > 0:
            self._count("collate_failed")
        return batch

    def densify(self, batch: Mapping[str, np.ndarray], moves: bool = False) -> Batch:
        """A batch with dex numerics gathered and ids made safe for embeddings.

        Adds ``mon_dex`` ``[N, 12, species columns]`` (by current forme) and
        ``cand_num`` ``[N, 2, n_cand, move columns]``, and ``mon_move_num``
        ``[N, 12, 4, move columns]`` when ``moves`` is set; then clamps species
        and move ids beyond the vocabulary (extension rows) to 0. Never raises:
        on failure the input comes back unchanged, with a count.
        """
        try:
            out = dict(batch)
            mon_id = np.asarray(batch["mon_id"]).astype(np.int64)
            cand = np.asarray(batch["cand_move"]).astype(np.int64)
            known = np.asarray(batch["mon_move"]).astype(np.int64)
            out["mon_dex"] = self.tables.species_num[mon_id[..., ID_FORME]]
            out["cand_num"] = self.tables.move_num[cand]
            if moves:
                out["mon_move_num"] = self.tables.move_num[known]
            safe = mon_id.copy()
            for column in (ID_SPECIES, ID_FORME, ID_KEY):
                safe[..., column] = np.where(
                    safe[..., column] < self.n_species, safe[..., column], 0
                )
            safe[..., ID_LAST_MOVE] = np.where(
                safe[..., ID_LAST_MOVE] < self.n_moves, safe[..., ID_LAST_MOVE], 0
            )
            out["mon_id"] = safe.astype(np.uint16)
            out["cand_move"] = np.where(cand < self.n_moves, cand, 0).astype(np.uint16)
            out["mon_move"] = np.where(known < self.n_moves, known, 0).astype(np.uint16)
            return out
        except Exception as exc:
            self._count(f"densify_error:{type(exc).__name__}")
            return dict(batch)

    # --- internals ---------------------------------------------------------

    def _count(self, name: str) -> None:
        try:
            self.counters[name] += 1
        except Exception:
            pass

    def _species_id(self, name: str) -> int:
        row = self._species_ix.get(name)
        if row is None:
            self._count("unknown_species")
            return 0
        return row

    def _move_id(self, name: str) -> int:
        info = self._moves.get(name)
        if info is None:
            self._count("unknown_move")
            return 0
        return info.row

    def _volatile_id(self, effect: str) -> int:
        position = self._vol_ix.get(effect)
        if position is None:
            position = self._vol_ix.get(effect.rstrip("0123456789"))
        if position is not None:
            return 1 + position
        bucket = zlib.crc32(effect.encode("utf-8")) % N_VOLATILE_HASH
        return 1 + len(self.vocab.volatiles) + bucket

    def _mon_rows(self, mon: MonPublic, block: _SideBlock) -> None:
        if mon.item_state == ITEM_UNKNOWN:
            item = self._item_unknown
        elif mon.item is None:
            item = self._item_none
        else:
            item = self._item_ix.get(mon.item, 0)
        ability = self._ability_ix.get(mon.ability, 0) if mon.ability else 0
        last = mon.last_action
        last_move = last_kind = last_target = 0
        last_mega = last_locked = last_protect = 0
        if last is not None:
            shown = last.move or last.forced_move
            if shown:
                last_move = self._move_id(shown)
            last_kind = 1 + KIND_CODES.get(last.kind, Y_KIND_NONE)
            if last.target is not None:
                last_target = 1 + _TARGET_INDEX.get(last.target, T_AUTO)
            last_mega = int(last.mega)
            last_locked = int(last.locked)
            last_protect = int(last.is_protect)
        block.ids.append(
            [
                self._species_id(mon.species),
                self._species_id(mon.forme),
                self._species_id(set_key(mon.forme, mon.species)),
                item,
                ability,
                last_move,
            ]
        )
        known = mon.moves[:MAX_MOVES]
        if len(mon.moves) > MAX_MOVES:
            self._count("mon_moves_truncated")
        pad = MAX_MOVES - len(known)
        block.moves.append([self._move_id(move.id) for move in known] + [0] * pad)
        block.move_flags.append(
            [
                (MOVE_REVEALED if move.revealed else 0)
                | (MOVE_FROM_SHEET if move.from_sheet else 0)
                for move in known
            ]
            + [0] * pad
        )
        block.hp.append(float(mon.hp))
        boosts = mon.boosts
        block.boosts.append([int(boosts.get(stat, 0)) for stat in BOOST_STATS])
        if mon.status is None:
            status = 0
        else:
            status = 1 + self._status_ix.get(mon.status, len(self.vocab.statuses))
        slot = (
            SLOT_BENCH if mon.slot is None else (SLOT_A if mon.slot == "a" else SLOT_B)
        )
        if mon.brought is None:
            brought = BROUGHT_UNKNOWN
        else:
            brought = BROUGHT_YES if mon.brought else BROUGHT_NO
        block.cats.append(
            [
                status,
                slot,
                ITEM_STATE_CODES.get(mon.item_state, 0),
                brought,
                last_kind,
                last_target,
                min(3, max(0, mon.protect_streak)),
                min(MAX_AGE, max(0, mon.turns_on_field)),
                min(MAX_AGE, len(mon.moves)),
            ]
        )
        block.flags.append(
            [
                1,
                int(mon.revealed),
                int(mon.fainted),
                int(mon.is_mega),
                int(mon.mega_possible),
                int(mon.first_turn),
                int(mon.protected_last_turn),
                int(mon.ability_known),
                last_mega,
                last_locked,
                last_protect,
            ]
        )
        other_type = len(self.vocab.types) + 1
        types = [
            1 + self._type_ix[name] if name in self._type_ix else other_type
            for name in mon.types[:MAX_TYPES]
        ]
        block.types.append(types + [0] * (MAX_TYPES - len(types)))
        vols = [self._volatile_id(effect) for effect in mon.volatiles][:MAX_VOLATILES]
        if len(mon.volatiles) > MAX_VOLATILES:
            self._count("volatiles_truncated")
        block.vols.append(vols + [0] * (MAX_VOLATILES - len(vols)))

    def _side_block(self, side: SidePublic, turn: int) -> _SideBlock:
        unknown = side.n_unknown_brought
        block = _SideBlock(
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [
                int(side.mega_used),
                min(255, side.n_fainted),
                min(255, side.n_revealed),
                0 if unknown is None else 1 + min(254, unknown),
                min(255, side.team_size or 0),
            ],
            [0] * len(self.vocab.side_conditions),
        )
        for effect, start in side.conditions.items():
            position = self._side_ix.get(effect)
            if position is None:
                self._count("side_condition_unknown")
            else:
                block.age[position] = _age(start, turn)
        if len(side.mons) > N_ROSTER:
            self._count("roster_truncated")
        for mon in side.mons[:N_ROSTER]:
            self._mon_rows(mon, block)
        for _ in range(N_ROSTER - len(block.ids)):
            block.ids.append([0] * N_MON_ID)
            block.moves.append([0] * MAX_MOVES)
            block.move_flags.append([0] * MAX_MOVES)
            block.hp.append(0.0)
            block.boosts.append([0] * len(BOOST_STATS))
            block.cats.append([0] * N_MON_CAT)
            block.flags.append([0] * N_MON_FLAG)
            block.types.append([0] * MAX_TYPES)
            block.vols.append([0] * MAX_VOLATILES)
        return block

    def _side_blocks(self, snapshot: PublicSnapshot) -> dict[str, _SideBlock]:
        cached = self._blocks
        if cached is not None and cached[0] is snapshot:
            return cached[1]
        blocks = {
            side: self._side_block(snapshot.sides[side], snapshot.turn)
            for side in SIDES
        }
        self._blocks = (snapshot, blocks)
        return blocks

    def _candidates(
        self, mon: MonPublic, terrain: bool
    ) -> tuple[list[str], list[int], list[float], list[int], list[float], float, float]:
        """(move ids, flags, prior, rank, auto share, OTHER prior, support)."""
        key = set_key(mon.forme, mon.species)
        ranked, table, support = self.repertoire.lookup(key)
        names: list[str] = []
        flags: list[int] = []
        sheet_known = False
        revealed = 0
        for move in mon.moves:
            if move.id in names:
                continue
            names.append(move.id)
            flags.append(
                CAND_VALID
                | (CAND_REVEALED if move.revealed else 0)
                | (CAND_SHEET if move.from_sheet else 0)
            )
            sheet_known = sheet_known or move.from_sheet
            revealed += int(move.revealed)
        complete = sheet_known or revealed >= MAX_MOVES
        if not complete:
            seen = set(names)
            for move_id in ranked:
                if len(names) >= self.n_cand:
                    break
                if move_id not in seen:
                    seen.add(move_id)
                    names.append(move_id)
                    flags.append(CAND_VALID | CAND_REPERTOIRE)
            if len(names) < self.n_cand:
                for move_id in self.repertoire.global_ranked():
                    if len(names) >= self.n_cand:
                        break
                    if move_id not in seen:
                        seen.add(move_id)
                        names.append(move_id)
                        flags.append(CAND_VALID | CAND_GLOBAL)
        if len(names) > self.n_cand:
            self._count("candidates_truncated")
            names, flags = names[: self.n_cand], flags[: self.n_cand]
        prior: list[float] = []
        rank: list[int] = []
        auto: list[float] = []
        for position, move_id in enumerate(names):
            frequency, place = table.get(move_id, (0.0, 0))
            prior.append(frequency)
            rank.append(min(255, place))
            info = self._moves.get(move_id)
            bits = info.bits if info is not None else 0
            flags[position] |= bits
            if bits & CAND_AIMED:
                auto.append(self.repertoire.auto_share(move_id, terrain))
            else:
                auto.append(1.0 if info is not None and info.target_type else 0.0)
        outside = 0.0 if complete else max(0.0, 1.0 - sum(prior))
        if not complete and support <= 0:
            outside = 1.0
        return names, flags, prior, rank, auto, outside, support

    def _target_bits(
        self, move_id: str, foe_bits: int, ally: bool, damaging_auto: bool
    ) -> int:
        info = self._moves.get(move_id)
        if info is None or not info.target_type:
            # A move the dex does not know: nothing can be ruled out.
            return foe_bits | (1 << T_ALLY if ally else 0) | 1 << T_SELF | 1 << T_AUTO
        kind = info.target_type
        if not info.bits & CAND_AIMED:
            return 1 << T_AUTO
        bits = 0
        if kind in _AIM_FOES:
            bits |= foe_bits
        if kind in _AIM_ALLY and (ally or kind not in _AIM_FOES):
            bits |= 1 << T_ALLY
        if kind in _AIM_SELF or info.self_too:
            bits |= 1 << T_SELF
        if damaging_auto:
            bits |= 1 << T_AUTO
        return bits or (1 << T_FOE_A | 1 << T_FOE_B | 1 << T_ALLY)

    def _encode(self, snapshot: PublicSnapshot, actor_side: str) -> Example | None:
        if actor_side not in SIDES:
            self._count("encode_skip:bad_side")
            return None
        if not dex_available():
            self._count("encode_skip:dex_missing")
            return None
        other_side = SIDES[1 - SIDES.index(actor_side)]
        actor = snapshot.sides.get(actor_side)
        other = snapshot.sides.get(other_side)
        if actor is None or other is None:
            self._count("encode_skip:side_missing")
            return None
        blocks = self._side_blocks(snapshot)
        mine, theirs = blocks[actor_side], blocks[other_side]
        turn = snapshot.turn
        n_cand = self.n_cand
        n_action = action_size(n_cand)

        act_mon = [-1, -1]
        foe_mon = [-1, -1]
        for position, letter in enumerate(SLOT_LETTERS):
            index = actor.active.get(letter)
            if index is not None and 0 <= index < min(N_ROSTER, len(actor.mons)):
                act_mon[position] = index
            elif index is not None:
                self._count("active_outside_roster")
            index = other.active.get(letter)
            if index is not None and 0 <= index < min(N_ROSTER, len(other.mons)):
                foe_mon[position] = N_ROSTER + index
        if act_mon[0] < 0 and act_mon[1] < 0:
            self._count("encode_skip:no_active")
            return None

        weather = 0
        if snapshot.weather:
            weather = 1 + self._weather_ix.get(
                snapshot.weather, len(self.vocab.weathers)
            )
        setter = SETTER_UNKNOWN
        if snapshot.weather and snapshot.weather_side == actor_side:
            setter = SETTER_ACTOR
        elif snapshot.weather and snapshot.weather_side == other_side:
            setter = SETTER_OTHER
        field_age = [0] * (2 + len(self.vocab.pseudo))
        if snapshot.weather:
            field_age[0] = _age(snapshot.weather_start, turn) or 1
        terrain = 0
        for effect, start in snapshot.fields.items():
            position = self._terrain_ix.get(effect)
            if position is not None:
                terrain = 1 + position
                field_age[1] = _age(start, turn)
                continue
            position = self._pseudo_ix.get(effect)
            if position is not None:
                field_age[2 + position] = _age(start, turn)
            elif (moves_dex().get(effect) or {}).get("terrain"):
                terrain = 1 + len(self.vocab.terrains)
                field_age[1] = _age(start, turn)
            else:
                self._count("field_unknown")
        terrain_up = terrain > 0

        elo = [0, 0]
        elo_known = [0, 0]
        if self.elo_mode == ELO_KEEP:
            for position, side in enumerate((actor, other)):
                if side.rating is not None:
                    elo[position] = max(0, min(32767, int(side.rating)))
                    elo_known[position] = 1
        bo3 = snapshot.best_of == 3 or (snapshot.format_id or "").endswith("bo3")
        sheets = [
            SHEET_UNKNOWN
            if side.sheet_open is None
            else (SHEET_OPEN if side.sheet_open else SHEET_CLOSED)
            for side in (actor, other)
        ]

        switch_row = [
            int(mon.slot is None and not mon.fainted and mon.brought is not False)
            for mon in actor.mons[:N_ROSTER]
        ]
        switch_row += [0] * (N_ROSTER - len(switch_row))
        foe_bits = (1 << T_FOE_A if foe_mon[0] >= 0 else 0) | (
            1 << T_FOE_B if foe_mon[1] >= 0 else 0
        )
        cand_move = [[0] * n_cand, [0] * n_cand]
        cand_flag = [[0] * n_cand, [0] * n_cand]
        cand_prior = [[0.0] * n_cand, [0.0] * n_cand]
        cand_rank = [[0] * n_cand, [0] * n_cand]
        cand_auto = [[0.0] * n_cand, [0.0] * n_cand]
        cand_tmask = [[0] * (n_cand + 1), [0] * (n_cand + 1)]
        other_prior = [0.0, 0.0]
        support = [0.0, 0.0]
        switch_mask = [[0] * N_ROSTER, [0] * N_ROSTER]
        action_mask = [[0] * n_action, [0] * n_action]
        for position in range(N_SLOT):
            if act_mon[position] < 0:
                continue
            mon = actor.mons[act_mon[position]]
            ally = act_mon[1 - position] >= 0
            names, flags, prior, rank, auto, outside, uses = self._candidates(
                mon, terrain_up
            )
            for column, move_id in enumerate(names):
                cand_move[position][column] = self._move_id(move_id)
                cand_flag[position][column] = flags[column]
                cand_prior[position][column] = prior[column]
                cand_rank[position][column] = rank[column]
                cand_auto[position][column] = auto[column]
                became_spread = bool(
                    flags[column] & CAND_DAMAGING
                ) and self.repertoire.auto_seen(move_id)
                cand_tmask[position][column] = self._target_bits(
                    move_id, foe_bits, ally, became_spread
                )
                action_mask[position][column] = 1
            cand_tmask[position][n_cand] = (
                foe_bits | (1 << T_ALLY if ally else 0) | 1 << T_SELF | 1 << T_AUTO
            )
            other_prior[position] = outside
            support[position] = float(np.log1p(uses))
            switch_mask[position] = list(switch_row)
            action_mask[position][n_cand] = 1
            action_mask[position][n_cand + 1 :] = switch_row

        return {
            "turn": np.array(min(255, max(0, turn)), dtype=np.uint8),
            "ctx_cat": np.array([weather, terrain, setter], dtype=np.uint8),
            "field_age": np.array(field_age, dtype=np.uint8),
            "game_flag": np.array(
                [int(bo3), int(snapshot.rated), sheets[0], sheets[1]], dtype=np.uint8
            ),
            "elo": np.array(elo, dtype=np.int16),
            "elo_known": np.array(elo_known, dtype=np.uint8),
            "side_cnt": np.array([mine.cnt, theirs.cnt], dtype=np.uint8),
            "side_age": np.array([mine.age, theirs.age], dtype=np.uint8),
            "mon_id": np.array(mine.ids + theirs.ids, dtype=np.uint16),
            "mon_move": np.array(mine.moves + theirs.moves, dtype=np.uint16),
            "mon_move_flag": np.array(
                mine.move_flags + theirs.move_flags, dtype=np.uint8
            ),
            "mon_hp": np.array(mine.hp + theirs.hp, dtype=np.float16),
            "mon_boost": np.array(mine.boosts + theirs.boosts, dtype=np.int8),
            "mon_cat": np.array(mine.cats + theirs.cats, dtype=np.uint8),
            "mon_flag": np.array(mine.flags + theirs.flags, dtype=np.uint8),
            "mon_type": np.array(mine.types + theirs.types, dtype=np.uint8),
            "mon_vol": np.array(mine.vols + theirs.vols, dtype=np.uint8),
            "act_mon": np.array(act_mon, dtype=np.int8),
            "foe_mon": np.array(foe_mon, dtype=np.int8),
            "slot_support": np.array(support, dtype=np.float16),
            "cand_move": np.array(cand_move, dtype=np.uint16),
            "cand_flag": np.array(cand_flag, dtype=np.uint16),
            "cand_prior": np.array(cand_prior, dtype=np.float16),
            "cand_rank": np.array(cand_rank, dtype=np.uint8),
            "cand_auto": np.array(cand_auto, dtype=np.float16),
            "cand_tmask": np.array(cand_tmask, dtype=np.uint8),
            "other_prior": np.array(other_prior, dtype=np.float16),
            "switch_mask": np.array(switch_mask, dtype=np.uint8),
            "action_mask": np.array(action_mask, dtype=np.uint8),
        }

    def _move_action(
        self, example: Mapping[str, np.ndarray], slot: int, move: str
    ) -> int:
        """Action index of ``move`` for the slot: its candidate, else OTHER."""
        info = self._moves.get(move)
        if info is not None and info.row > 0:
            row = example["cand_move"][slot]
            flags = example["cand_flag"][slot]
            for column in range(self.n_cand):
                if int(flags[column]) & CAND_VALID and int(row[column]) == info.row:
                    return column
        return other_index(self.n_cand)

    def _encode_labels(
        self,
        example: Mapping[str, np.ndarray],
        snapshot: PublicSnapshot,
        actor_side: str,
        actions: Mapping[str, SlotAction],
    ) -> Example | None:
        n_cand = self.n_cand
        n_action = action_size(n_cand)
        other = other_index(n_cand)
        actor = snapshot.sides.get(actor_side)
        if actor is None or actor_side not in SIDES:
            self._count("label_skip:side_missing")
            return None
        y_kind = [Y_KIND_ABSENT, Y_KIND_ABSENT]
        y_action = [-1, -1]
        y_set = [[0] * n_action, [0] * n_action]
        y_target = [-1, -1]
        y_mega = [-1, -1]
        y_intent = [-1, -1]
        y_flag = [[0] * N_Y_FLAG, [0] * N_Y_FLAG]
        y_attack = [[-1, -1], [-1, -1]]
        y_reason = [0, 0]
        act_mon = example["act_mon"]
        foe_up = [int(example["foe_mon"][0]) >= 0, int(example["foe_mon"][1]) >= 0]
        statuses = self._status_ix
        for slot, letter in enumerate(SLOT_LETTERS):
            action = actions.get(actor_side + letter)
            row = int(act_mon[slot])
            if action is None or row < 0:
                if action is not None:
                    self._count("label_without_slot")
                elif row >= 0:
                    self._count("slot_without_label")
                continue
            mon = actor.mons[row]
            if action.species != mon.species:
                self._count("label_species_mismatch")
                continue
            kind = KIND_CODES.get(action.kind)
            if kind is None:
                self._count("label_unknown_kind")
                continue
            y_kind[slot] = kind
            mask = example["action_mask"][slot]
            flags = example["cand_flag"][slot]
            reason = action.reason or ""
            if reason in _REASON_INDEX:
                y_reason[slot] = _REASON_INDEX[reason]
            elif reason in statuses:
                y_reason[slot] = _REASON_INDEX["status"]
            else:
                y_reason[slot] = _REASON_INDEX["other"]
            flag = y_flag[slot]
            flag[Y_SWITCH_KNOWN] = int(action.switch_known)
            flag[Y_PROTECT_KNOWN] = int(action.protect_known)
            flag[Y_LOCKED] = int(action.locked)
            visible = kind in (Y_KIND_MOVE, Y_KIND_SWITCH, Y_KIND_HIDDEN)
            if visible or action.switch_known:
                y_mega[slot] = int(action.mega)

            if kind == Y_KIND_SWITCH:
                flag[Y_SWITCHED] = flag[Y_SWITCH_KNOWN] = flag[Y_PROTECT_KNOWN] = 1
                y_intent[slot] = _INTENT_INDEX[INTENT_SWITCH]
                y_attack[slot] = [0 if foe_up[0] else -1, 0 if foe_up[1] else -1]
                target = next(
                    (
                        index
                        for index, bench in enumerate(actor.mons[:N_ROSTER])
                        if bench.species == action.switch_to
                    ),
                    None,
                )
                if target is not None and int(mask[switch_index(target, n_cand)]):
                    y_action[slot] = switch_index(target, n_cand)
                    y_set[slot][y_action[slot]] = 1
                else:
                    # Known to be a switch, destination not representable.
                    self._count(
                        "switch_target_unresolved"
                        if target is None
                        else "switch_outside_mask"
                    )
                    for index in range(n_cand + 1, n_action):
                        y_set[slot][index] = int(mask[index])
                continue

            if kind == Y_KIND_MOVE and action.move:
                move = action.move
                flag[Y_SWITCH_KNOWN] = flag[Y_PROTECT_KNOWN] = 1
                flag[Y_PROTECTED] = int(action.is_protect)
                y_attack[slot] = self._attack_label(action, foe_up)
                if action.locked:
                    continue  # a forced continuation is not a choice
                index = self._move_action(example, slot, move)
                y_action[slot] = index
                y_set[slot][index] = 1
                flag[Y_OTHER] = int(index == other)
                intent = action.intent
                if intent is not None:
                    y_intent[slot] = _INTENT_INDEX[intent]
                if action.target is not None and (
                    action.target_trusted or action.target_forced
                ):
                    target = _TARGET_INDEX.get(action.target)
                    legal = int(example["cand_tmask"][slot][index])
                    if target is not None and legal >> target & 1:
                        y_target[slot] = target
                    else:
                        self._count("target_outside_mask")
                continue

            # Hidden, or no action line: what the log still rules out.
            if kind == Y_KIND_MOVE:
                self._count("move_label_without_move")
                continue
            some_move = kind == Y_KIND_HIDDEN or (
                action.switch_known
                and not action.locked
                and reason not in (REASON_UNEXPLAINED, REASON_FORCED)
            )
            if not some_move:
                continue
            for column in range(n_cand):
                bits = int(flags[column])
                if not bits & CAND_VALID:
                    continue
                if action.protect_known and bits & CAND_PROTECT:
                    continue
                y_set[slot][column] = 1
            y_set[slot][other] = 1

        return {
            "y_kind": np.array(y_kind, dtype=np.int8),
            "y_action": np.array(y_action, dtype=np.int8),
            "y_set": np.array(y_set, dtype=np.uint8),
            "y_target": np.array(y_target, dtype=np.int8),
            "y_mega": np.array(y_mega, dtype=np.int8),
            "y_intent": np.array(y_intent, dtype=np.int8),
            "y_flag": np.array(y_flag, dtype=np.uint8),
            "y_attack": np.array(y_attack, dtype=np.int8),
            "y_reason": np.array(y_reason, dtype=np.uint8),
        }

    def _attack_label(self, action: SlotAction, foe_up: Sequence[bool]) -> list[int]:
        """Per foe slot: 1 attacked, 0 not, -1 unknown (see ``y_attack``)."""
        move = action.move or ""
        info = self._moves.get(move)
        if info is None or not info.target_type:
            return [-1, -1]
        absent = [0 if up else -1 for up in foe_up]
        if not info.bits & CAND_DAMAGING:
            return absent
        if action.target == TARGET_AUTO:
            # Every adjacent foe, by the dex or because the simulator spread an
            # aimed move; a random or scripted target is not known from here.
            if info.target_type in _HITS_ALL_FOES or info.bits & CAND_AIMED:
                return [1 if up else -1 for up in foe_up]
            return [-1, -1]
        if action.target == TARGET_FOE_A:
            return [1 if foe_up[0] else -1, absent[1]]
        if action.target == TARGET_FOE_B:
            return [absent[0], 1 if foe_up[1] else -1]
        if action.target in (TARGET_ALLY, TARGET_SELF):
            return absent
        return [-1, -1]


def encode_many(
    featurizer: Featurizer,
    turns: Iterable[tuple[PublicSnapshot, Mapping[str, SlotAction]]],
    sides: Sequence[str] = SIDES,
) -> list[Example]:
    """Examples with labels for every turn and every listed actor side."""
    out: list[Example] = []
    for snapshot, actions in turns:
        for side in sides:
            example = featurizer.encode_turn(snapshot, actions, side)
            if example is not None:
                out.append(example)
    return out
