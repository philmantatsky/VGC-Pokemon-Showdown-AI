"""Count-table predictors of the opponent's action (OPPONENT_PREDICTOR.md, "Models").

Three predictors over the plain collated batch, numpy only, fitted from weighted
counts (``m_weight``) of whatever rows ``fit`` is given (the caller passes the
TRAIN mask):

* ``SpeciesTable`` - set key -> action frequencies. The floor.
* ``FlagsTable``   - set key x (turn 1, first turn on field, protected last
  turn), shrunk towards the set key's row and from there to a global row. The
  bar of reading R1, and the day-one model behind the runtime interface.
* ``EloTable``     - ``FlagsTable`` times an Elo-band offset on coarse action
  classes. Exists to test the Elo premise cheaply.

    table = FlagsTable.fit(batch, mask=train_rows, featurizer=featurizer)
    pred = table.predict(batch)            # {'action', 'target', 'mega'}
    again = from_payload(table.to_payload(), featurizer)

How a row becomes a prediction
------------------------------
The action space differs per example (candidate moves, ``OTHER``, six switch
pointers), so a table stores counts per MOVE ID and a separate switch rate, and
lays them onto each slot's candidates at predict time:

1. Move part, ``usage='rate'`` (the default). Whether a move is in the set
   matters more than how popular it is: a rare move that a Pokemon has shown
   is a likely click, and its share of all the set key's clicks says the
   opposite. So a move has two usage rates per row, each a count over an
   exposure: clicks while the move was KNOWN to be in the set (shown, or on an
   open sheet) over the turns it stood there known, and clicks while it was a
   not yet shown candidate over those turns. ``OTHER`` has its own rate over
   the turns the moveset was still open. Each candidate reads the rate of its
   state; the entries are renormalised over the slot.
   ``usage='share'`` is the plain table: one share of the row's clicks per
   move whatever its state, a small smoothed share (``unseen_count``) for a
   candidate the table never counted, and ``OTHER`` = the share of the moves
   outside the candidates. It is kept to measure what the rates buy.
2. Reveal offsets (``reveal_offsets``). One ratio per (sheet known, number of
   shown moves, candidate status: shown / not shown / global fill / OTHER)
   multiplies the entries before the renormalisation, because how the click
   splits between shown and not shown moves depends on how many are shown.
   Class offsets (``class_offsets``) do the same per (flags, sheet known,
   coarse class), which keeps P(Protect) right in every flags cell after the
   first ratio has moved it. Both are fitted on the fit rows by iterative
   proportional fitting: observed over expected weight. ``other_boost`` then
   scales ``OTHER`` (the fit rows built the repertoire, so their own OTHER
   rate is optimistic; the caller picks the boost on validation). A complete
   moveset (a known sheet, or four shown moves) leaves ``OTHER`` the floor.
3. Switch part. ``P(switch | row)`` is counted over slots that had a legal
   switch target, censored slots in the denominator. Its mass goes to the
   legal pointers: shown bench Pokemon against not yet shown ones in the
   counted proportion for that (number shown, number not shown) pattern,
   uniform inside each group (``switch_dest='uniform'`` ignores the counts).
   A not yet shown pointer is one of up to four roster Pokemon of which two
   were brought, which is why the counts favour a shown one about 3.7 to 1.
4. A floor on every legal action, then a last renormalisation. Illegal entries
   and empty slots are exactly zero.

Shrinkage (``_shrink``): ``(count + strength * prior) / (total + strength)``.
Strength 0 gives the raw frequencies (the prior where the row is empty),
infinity gives the prior. A rate goes: all rows -> the move over all set keys
(``move_strength``) -> the set key (``key_strength``) -> (set key, flags)
(``cell_strength``); a share goes: global row -> set key -> (set key, flags).
The prior of a (set key, flags) row is the set key's row times a flags x
coarse-class ratio (observed over expected class weight in that flags cell,
``flag_effect_strength``; infinity switches the ratio off), so a rarely seen
set key still learns that a first-turn-only move is gone after the first
turn. The switch rate and the Mega rate use the same steps with their own
strengths. An unseen set key reads the row of all set keys (with the flags
ratio, for ``FlagsTable``); a move the table never counted reads the overall
rate of its state.

Censored slots (``censored``). A voluntary switch and a Protect are always
visible, an attack is not (the slot may faint or flinch first), so counting
visible labels alone over-predicts both. ``'fractional'`` (the default) keeps a
censored slot in the switch denominator and in the exposures, and adds its
weight to the moves of its ``y_set``: the split between the Protect family
and the other moves follows the table's own probabilities (an EM pass,
``iterations`` rounds), and inside each of the two groups the row keeps the
mix of its visible labels. ``'fractional_slot'`` spreads the weight per slot
over the slot's own candidates instead, which hands priority moves weight
they did not earn (a move that goes first is censored less often).
``'drop'`` counts visible labels only. The last two exist to measure the
difference.

Targets: per (set key, move) target-class counts from the labels that carry a
target (``y_target``: trusted, or forced by a lone foe), shrunk to the move
across all set keys, then to the move's dex class (the ``CAND_*`` class bits
the batch stores: aimed / spread / damaging / Protect / first-turn-only), then
to uniform; restricted to ``cand_tmask`` and renormalised. For an aimed move
that the simulator can turn into a spread hit, ``P(auto)`` is the batch's
``cand_auto`` (the training share in this terrain state) when
``use_cand_auto`` is set. Targets are not conditioned on the flags.

Which of two foes (``target_context``, on by default). Set key and move alone
cannot tell which of the two opposing Pokemon is the better target, so the
split between ``foe_a`` and ``foe_b`` would be a slot-letter habit. Where both
are legal targets of a candidate, each foe gets one of 30 ratios: for a
damaging move, keyed by the move's type effectiveness against that foe (below
1 / 1 / above 1), against the other foe, and whether that foe has less, about
as much or more HP than the other; for any other move by the HP relation
alone. The ratios multiply the two foes' probabilities, whose sum stays what
it was (the ally / self / auto classes are not touched). They are fitted on
the fit rows by proportional fitting, observed over expected weight, on top
of the rows above (``iterations`` rounds, pseudo-count
``target_context_strength``). The effectiveness comes from the dex's type
chart and the move's type as the featurizer's numeric tables hold it, both
stored with the table (``type_chart``, ``move_type``); the foes' types and HP
are read from the batch (``mon_type``, ``mon_hp``, ``foe_mon``). No ability,
item or move-specific exception is modelled. A table fitted without a
featurizer, and a batch without those three arrays, read every ratio as 1.

Mega: P(the slot moves) times the Mega rate of a slot that moves (a Mega never
comes with a switch), the rate per (set key, turn 1) for ``SpeciesTable`` and
per (set key, flags) for the other two, over slots where a Mega was possible
and the log tells.

Nothing here reads a label (``y_*``) or a meta array (``m_*``) at predict time:
``predict`` reads the twelve arrays of ``_FEATURES`` and, when the batch has
them, the three of ``_CONTEXT_FEATURES``.
``predict`` never raises: on any failure it counts ``predict_error:<type>`` in
``counters`` and returns the uniform prediction (an empty dict when the batch
has no masks). ``fit`` and ``from_payload`` are offline tools and raise
``ValueError`` on malformed input.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, fields, replace
from typing import Any, Mapping, Sequence, TypeVar

import numpy as np

from vgc_bench.src.oppmodel.events import INTENT_SWITCH
from vgc_bench.src.oppmodel.features import (
    CAND_AIMED,
    CAND_DAMAGING,
    CAND_FIRST_TURN,
    CAND_GLOBAL,
    CAND_PROTECT,
    CAND_REVEALED,
    CAND_SHEET,
    CAND_SPREAD,
    FLAG_REVEALED,
    MAX_MOVES,
    N_ROSTER,
    N_TARGET,
    T_AUTO,
    T_FOE_A,
    T_FOE_B,
    expand_target_mask,
    move_columns,
    slot_view,
    uniform_prediction,
)

KIND = "table"
PAYLOAD_VERSION = 1
TABLE_SPECIES = "species_table"
TABLE_FLAGS = "flags_table"
TABLE_ELO = "elo_table"

CENSORED_FRACTIONAL = "fractional"
CENSORED_SLOT = "fractional_slot"
CENSORED_DROP = "drop"
CENSORED_MODES = (CENSORED_FRACTIONAL, CENSORED_SLOT, CENSORED_DROP)
DEST_REVEALED = "revealed"
DEST_UNIFORM = "uniform"
USAGE_RATE = "rate"
USAGE_SHARE = "share"
KNOWN, NOT_KNOWN = 0, 1  # first axis of the move counts: is the move in the set?

# Coarse action classes, from the candidate's dex class bits. They key the
# flags x class ratio and the Elo-band offset.
(
    CLASS_PROTECT,
    CLASS_FIRST_TURN,
    CLASS_SPREAD_ATTACK,
    CLASS_AIMED_ATTACK,
    CLASS_OTHER_ATTACK,
    CLASS_AIMED_STATUS,
    CLASS_OTHER_STATUS,
    CLASS_OUTSIDE,
    CLASS_SWITCH,
) = range(9)
N_MOVE_CLASS = 8  # the first eight: every class a move action can have
N_CLASS = 9
CLASS_NAMES: tuple[str, ...] = (
    "protect_family",
    "first_turn_only",
    "spread_attack",
    "aimed_attack",
    "other_attack",
    "aimed_status",
    "other_status",
    "outside_candidates",
    INTENT_SWITCH,
)

# Candidate status for the reveal offsets, and the offsets' row index.
STATUS_SHOWN, STATUS_NOT_SHOWN, STATUS_GLOBAL, STATUS_OTHER = range(4)
N_STATUS = 4
N_REVEAL_ROW = 2 * (MAX_MOVES + 1)  # (sheet known) x (0 .. 4 shown moves)

_CLASS_BITS = (CAND_PROTECT, CAND_DAMAGING, CAND_AIMED, CAND_SPREAD, CAND_FIRST_TURN)
N_GROUP = (1 << len(_CLASS_BITS)) + 1  # every bit pattern, then OTHER
GROUP_OTHER = N_GROUP - 1
N_FLAG_CELL = 8  # turn 1 x first turn on field x protected last turn
# Target context cells of one (candidate, foe): for a damaging move
# effectiveness against this foe x against the other foe x HP relation, then
# three cells (the HP relation alone) for every other move.
N_EFFECT = 3  # below 1, exactly 1, above 1
N_HP_RELATION = 3  # this foe has less, about as much, more HP than the other
N_TARGET_CELL = N_EFFECT * N_EFFECT * N_HP_RELATION + N_HP_RELATION
_STATUS_CELL = N_EFFECT * N_EFFECT * N_HP_RELATION
_HP_STEPS = 5.0  # HP fractions closer than half of 1 / 5 count as "about as much"
_TYPE_COLUMN = "type:"  # prefix of the type columns of features.move_columns
MAX_PATTERN = N_ROSTER + 1  # shown / not shown legal pointers: 0 .. 6 each

_FEATURES = (
    "turn",
    "mon_id",
    "mon_flag",
    "mon_cat",
    "act_mon",
    "elo",
    "elo_known",
    "cand_move",
    "cand_flag",
    "cand_tmask",
    "cand_auto",
    "action_mask",
)
# Read when the batch has them: the two foes' types and HP (target context).
_CONTEXT_FEATURES = ("mon_type", "mon_hp", "foe_mon")
_LABELS = ("y_action", "y_set", "y_target", "y_mega")
_WEIGHT = "m_weight"
_GROUP_PRIOR_STRENGTH = 1.0  # class-bits row -> uniform over the five classes
_DEST_STRENGTH = 2.0  # (shown, not shown) pattern -> pooled preference
_AUTO_CLIP = 0.01
_RATE_PRIOR = (0.5, 1.0)  # half an event in one trial behind every global rate
_ALL_CLASS_BITS = sum(_CLASS_BITS)

TableT = TypeVar("TableT", bound="_CountTable")


# --- configuration ------------------------------------------------------------


@dataclass(frozen=True)
class TableConfig:
    """Strengths and switches of a count table. Every strength is a pseudo-count.

    ``math.inf`` is a legal strength (the row equals its prior). The fit script
    chooses the strengths, ``other_boost`` and ``floor`` on the validation
    split and stores the chosen configuration in the artifact. The defaults
    are near what it chose for the flags table on ``v1_ondisk`` (2026-10-04).
    """

    usage: str = USAGE_RATE
    move_strength: float = 100.0  # rate: the overall rate behind a move's rate
    key_strength: float = 100.0  # the move's (or global) row behind a set key's row
    cell_strength: float = 100.0  # set-key row behind a (set key, flags) row
    switch_key_strength: float = 300.0  # the same two steps for the switch rate
    switch_cell_strength: float = 100.0
    flag_effect_strength: float = 5.0  # flags x class ratio; inf = no ratio
    target_key_strength: float = 300.0  # move row behind a (set key, move) row
    target_move_strength: float = 5.0  # dex-class row behind a move row
    mega_strength: float = 10.0
    elo_strength: float = 500.0  # Elo band x class ratio; inf = no offset
    elo_edges: tuple[int, ...] = (1100, 1200, 1300, 1400, 1500)
    other_boost: float = 1.5
    unseen_count: float = 0.5  # share: pseudo-count of a move in the global row
    floor: float = 1e-5  # least probability of a legal action
    censored: str = CENSORED_FRACTIONAL
    iterations: int = 4  # EM rounds (censored slots) and IPF rounds (offsets)
    reveal_offsets: bool = True
    class_offsets: bool = True
    offset_strength: float = 5.0  # pseudo-count of a reveal or class ratio
    switch_dest: str = DEST_REVEALED
    use_cand_auto: bool = True
    target_context: bool = True  # which of two foes: effectiveness and HP
    target_context_strength: float = 5.0  # pseudo-count of a context ratio

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            out[item.name] = list(value) if isinstance(value, tuple) else value
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TableConfig":
        """Rebuild from ``to_dict()``: unknown names ignored, missing ones default."""
        default = cls()
        values: dict[str, Any] = {}
        for item in fields(cls):
            if item.name not in data:
                continue
            value = data[item.name]
            current = getattr(default, item.name)
            if isinstance(current, bool):
                values[item.name] = bool(value)
            elif isinstance(current, int):
                values[item.name] = int(value)
            elif isinstance(current, float):
                values[item.name] = float(value)
            elif isinstance(current, tuple):
                values[item.name] = tuple(int(edge) for edge in value)
            else:
                values[item.name] = str(value)
        return replace(default, **values)


# --- small array helpers ------------------------------------------------------


def _shrink(
    counts: np.ndarray, total: np.ndarray, prior: np.ndarray, strength: float
) -> np.ndarray:
    """``(counts + strength * prior) / (total + strength)``, limits included.

    Infinite strength returns the prior; where ``total + strength`` is zero
    (an empty row with strength 0) the prior stands in.
    """
    prior = np.broadcast_to(np.asarray(prior, dtype=np.float64), counts.shape)
    if math.isinf(strength):
        return prior.copy()
    denominator = np.broadcast_to(total + strength, counts.shape)
    safe = np.where(denominator > 0, denominator, 1.0)
    return np.where(denominator > 0, (counts + strength * prior) / safe, prior)


def _ratio(observed: np.ndarray, expected: np.ndarray, strength: float) -> np.ndarray:
    """``(observed + strength) / (expected + strength)``; 1 at infinite strength."""
    if math.isinf(strength):
        return np.ones_like(observed, dtype=np.float64)
    denominator = expected + strength
    safe = np.where(denominator > 0, denominator, 1.0)
    return np.where(denominator > 0, (observed + strength) / safe, 1.0)


def _normalized(values: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Rows of ``values`` summing to 1; ``fallback`` (normalised) where a row is 0."""
    total = values.sum(-1, keepdims=True)
    spare = fallback.sum(-1, keepdims=True)
    even = np.divide(
        fallback, spare, out=np.zeros_like(values, dtype=np.float64), where=spare > 0
    )
    return np.where(total > 0, values / np.where(total > 0, total, 1.0), even)


def _floored(values: np.ndarray, legal: np.ndarray, floor: float) -> np.ndarray:
    """At least ``floor`` on legal entries, zero elsewhere, rows renormalised."""
    lifted = np.where(legal, np.maximum(values, floor), 0.0)
    total = lifted.sum(-1, keepdims=True)
    return np.divide(lifted, total, out=np.zeros_like(lifted), where=total > 0)


def _bincount(index: np.ndarray, weights: np.ndarray, size: int) -> np.ndarray:
    """Weighted counts per index as float64 (numpy returns ints for no input)."""
    counted = np.bincount(
        index.ravel().astype(np.int64),
        weights=np.asarray(weights, dtype=np.float64).ravel(),
        minlength=size,
    )
    return counted[:size].astype(np.float64)


def _pack(array: np.ndarray) -> dict[str, Any]:
    """A dense count array as shape + flat indices + values of its non-zeros."""
    dense = np.asarray(array, dtype=np.float64)
    flat = dense.ravel()
    index = np.flatnonzero(flat)
    return {
        "shape": [int(size) for size in dense.shape],
        "index": index.astype(np.int64),
        "value": flat[index].copy(),
    }


def _unpack(packed: Mapping[str, Any]) -> np.ndarray:
    shape = tuple(int(size) for size in packed["shape"])
    flat = np.zeros(int(np.prod(shape, dtype=np.int64)), dtype=np.float64)
    index = np.asarray(packed["index"]).astype(np.int64).ravel()
    value = np.asarray(packed["value"], dtype=np.float64).ravel()
    if index.shape != value.shape or (index.size and int(index.max()) >= flat.size):
        raise ValueError("packed array does not fit its shape")
    flat[index] = value
    return flat.reshape(shape)


def coarse_class(bits: np.ndarray) -> np.ndarray:
    """``CLASS_*`` of candidate moves from their ``cand_flag`` class bits."""
    bits = np.asarray(bits).astype(np.int64)
    damaging = (bits & CAND_DAMAGING) > 0
    aimed = (bits & CAND_AIMED) > 0
    attack = np.where(
        (bits & CAND_SPREAD) > 0,
        CLASS_SPREAD_ATTACK,
        np.where(aimed, CLASS_AIMED_ATTACK, CLASS_OTHER_ATTACK),
    )
    out = np.where(
        damaging, attack, np.where(aimed, CLASS_AIMED_STATUS, CLASS_OTHER_STATUS)
    )
    out = np.where((bits & CAND_FIRST_TURN) > 0, CLASS_FIRST_TURN, out)
    return np.where((bits & CAND_PROTECT) > 0, CLASS_PROTECT, out)


def _class_group(bits: np.ndarray) -> np.ndarray:
    """Index of a candidate's class-bit pattern (the target back-off row)."""
    bits = np.asarray(bits).astype(np.int64)
    out = np.zeros(bits.shape, dtype=np.int64)
    for position, bit in enumerate(_CLASS_BITS):
        out |= ((bits & bit) > 0).astype(np.int64) << position
    return out


def _lookup(ids: np.ndarray, missing: int) -> np.ndarray:
    """id -> position in ``ids``; one spare entry at the end maps to ``missing``."""
    size = int(ids.max()) + 2 if ids.size else 1
    table = np.full(size, missing, dtype=np.int64)
    if ids.size:
        table[ids] = np.arange(ids.size, dtype=np.int64)
    return table


def _gather(table: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """``table[ids]`` with ids outside the table reading its spare last entry."""
    ids = np.asarray(ids).astype(np.int64)
    return table[np.where((ids >= 0) & (ids < table.size), ids, table.size - 1)]


def type_context(
    featurizer: Any, move_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(type id of each move, effectiveness chart, type names) for a table.

    Type ids are those of the batch's ``mon_type``: 0 none, ``1 + i`` for the
    featurizer's i-th type, then one id for any other type. ``chart[a, d]`` is
    the damage multiplier of an attack of type ``a`` against a defending type
    ``d``, by the installed dex's type chart; rows and columns of "none" and
    "other" are 1. A move's type is read from the type columns of the
    featurizer's ``move_num``. Without a featurizer, or when anything is
    missing, every move is typeless and the chart is all ones: the context
    layer then reads only the HP relation.
    """
    move_ids = np.asarray(move_ids).astype(np.int64)
    blank = (np.zeros(move_ids.size, dtype=np.int64), np.ones((2, 2)), [])
    try:
        names = [str(name) for name in featurizer.vocab.types]
        columns = move_columns(featurizer.vocab)
        at = [columns.index(_TYPE_COLUMN + name) for name in names]
        block = np.asarray(featurizer.tables.move_num)[:, at]
        typed = np.where(block.sum(1) > 0, block.argmax(1) + 1, 0).astype(np.int64)
        inside = (move_ids >= 0) & (move_ids < typed.size)
        move_type = np.where(inside, typed[np.where(inside, move_ids, 0)], 0)
        chart = _dex_chart(names)
    except Exception:
        return blank
    return move_type, chart, names


def _dex_chart(names: Sequence[str]) -> np.ndarray:
    """``chart[attack id, defender id]`` over the ids of ``names`` (see above)."""
    from poke_env.data import GenData

    table = GenData.from_gen(9).type_chart
    chart = np.ones((len(names) + 2, len(names) + 2), dtype=np.float64)
    for attack, attack_name in enumerate(names):
        for defend, defend_name in enumerate(names):
            try:
                value = table[defend_name.upper()][attack_name.upper()]
            except (KeyError, TypeError):
                continue
            chart[attack + 1, defend + 1] = float(value)
    return chart


def _retyped(
    move_type: np.ndarray, chart: np.ndarray, names: Sequence[str], featurizer: Any
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """A stored type context under the type ids of another vocabulary.

    The ids of ``mon_type`` follow the featurizer's type list. When that list
    is the stored one nothing changes; otherwise types are matched by name,
    and a type either side lacks reads as "other" (multiplier 1).
    """
    names = [str(name) for name in names]
    try:
        current = [str(name) for name in featurizer.vocab.types]
    except Exception:
        return move_type, chart, names
    if current == names or not names:
        return move_type, chart, names
    old = {name: index + 1 for index, name in enumerate(names)}
    new = {name: index + 1 for index, name in enumerate(current)}
    made = np.ones((len(current) + 2, len(current) + 2), dtype=np.float64)
    for attack, a in new.items():
        for defend, d in new.items():
            if attack in old and defend in old:
                made[a, d] = chart[old[attack], old[defend]]
    remap = np.zeros(len(names) + 2, dtype=np.int64)
    for name, index in old.items():
        remap[index] = new.get(name, 0)
    safe = np.clip(np.asarray(move_type).astype(np.int64), 0, len(names) + 1)
    return remap[safe], made, current


# --- per-slot view of a batch -------------------------------------------------


@dataclass
class _Rows:
    """Everything ``predict`` reads, one row per slot (``n = 2 * N``)."""

    shape: tuple[int, int]  # (N, 2)
    n_cand: int
    active: np.ndarray  # [n]
    key: np.ndarray  # [n] row of the set key; n_key = unseen
    cell: np.ndarray  # [n] action cell
    mega_cell: np.ndarray  # [n]
    mega_possible: np.ndarray  # [n]
    col: np.ndarray  # [n, C] column of the candidate's move; n_move + 1 = unseen
    bits: np.ndarray  # [n, C] cand_flag
    legal: np.ndarray  # [n, A]
    status: np.ndarray  # [n, C] STATUS_*
    reveal_row: np.ndarray  # [n]
    class_row: np.ndarray  # [n] action cell, and whether the sheet is known
    complete: np.ndarray  # [n] the moveset is fully known
    known: np.ndarray  # [n, C] KNOWN / NOT_KNOWN: the candidate is in the set
    shown_bench: np.ndarray  # [n, 6] legal pointer to a shown Pokemon
    hidden_bench: np.ndarray  # [n, 6] legal pointer to one not shown yet
    elo: np.ndarray  # [n]
    elo_known: np.ndarray  # [n]
    action_class: np.ndarray  # [n, A] CLASS_*


def _feature_arrays(batch: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {name: np.asarray(batch[name]) for name in _FEATURES}
    if all(name in batch for name in _CONTEXT_FEATURES):
        for name in _CONTEXT_FEATURES:
            out[name] = np.asarray(batch[name])
    return out


def _select(
    batch: Mapping[str, np.ndarray], mask: np.ndarray | None
) -> dict[str, np.ndarray]:
    """The arrays ``fit`` uses, cut down to the masked rows before anything else."""
    names = [*_FEATURES, *_LABELS]
    if _WEIGHT in batch:
        names.append(_WEIGHT)
    if all(name in batch for name in _CONTEXT_FEATURES):
        names.extend(_CONTEXT_FEATURES)
    missing = [name for name in names if name not in batch]
    if missing:
        raise ValueError(f"batch lacks {missing}")
    if mask is None:
        return {name: np.asarray(batch[name]) for name in names}
    rows = np.asarray(mask)
    if rows.dtype != bool or rows.shape != (len(np.asarray(batch["turn"])),):
        raise ValueError("mask must be one boolean per example")
    return {name: np.asarray(batch[name])[rows] for name in names}


# --- the tables ---------------------------------------------------------------


class _CountTable:
    """Shared machinery of the three tables; see the module docstring."""

    kind = KIND
    table = ""
    n_cell = 1
    n_mega_cell = 2

    def __init__(
        self,
        *,
        config: TableConfig,
        key_ids: np.ndarray,
        move_ids: np.ndarray,
        move_bits: np.ndarray,
        move_counts: np.ndarray,
        move_exposure: np.ndarray,
        switch_counts: np.ndarray,
        target_counts: np.ndarray,
        group_target: np.ndarray,
        mega_counts: np.ndarray,
        dest_counts: np.ndarray,
        reveal_ratio: np.ndarray,
        class_ratio: np.ndarray,
        key_names: Sequence[str] | None = None,
        move_names: Sequence[str] | None = None,
        fit_info: Mapping[str, Any] | None = None,
        name: str | None = None,
        target_ratio: np.ndarray | None = None,
        move_type: np.ndarray | None = None,
        type_chart: np.ndarray | None = None,
        type_names: Sequence[str] | None = None,
    ) -> None:
        self.config = config
        self.name = name or self.table
        self.counters: Counter[str] = Counter()
        self.key_ids = np.asarray(key_ids).astype(np.int64)
        self.move_ids = np.asarray(move_ids).astype(np.int64)
        self.move_bits = np.asarray(move_bits).astype(np.int64)
        self.key_names = None if key_names is None else [str(n) for n in key_names]
        self.move_names = None if move_names is None else [str(n) for n in move_names]
        self.fit_info: dict[str, Any] = dict(fit_info or {})
        n_key, n_move = self.key_ids.size, self.move_ids.size
        self.move_counts = np.asarray(move_counts, dtype=np.float64)
        self.move_exposure = np.asarray(move_exposure, dtype=np.float64)
        self.switch_counts = np.asarray(switch_counts, dtype=np.float64)
        self.target_counts = np.asarray(target_counts, dtype=np.float64)
        self.group_target = np.asarray(group_target, dtype=np.float64)
        self.mega_counts = np.asarray(mega_counts, dtype=np.float64)
        self.dest_counts = np.asarray(dest_counts, dtype=np.float64)
        self.reveal_ratio = np.asarray(reveal_ratio, dtype=np.float64)
        self.class_ratio = np.asarray(class_ratio, dtype=np.float64)
        # The target context: absent in a payload written before it existed,
        # which then reads every ratio as 1 (the table it always was).
        self.target_ratio = (
            np.ones(N_TARGET_CELL)
            if target_ratio is None
            else np.asarray(target_ratio, dtype=np.float64)
        )
        self.move_type = (
            np.zeros(n_move, dtype=np.int64)
            if move_type is None
            else np.asarray(move_type).astype(np.int64)
        )
        self.type_chart = (
            np.ones((2, 2))
            if type_chart is None
            else np.asarray(type_chart, dtype=np.float64)
        )
        self.type_names = [str(n) for n in (type_names or ())]
        if self.type_chart.ndim != 2 or self.type_chart.shape[0] < 2:
            raise ValueError(f"type_chart has shape {self.type_chart.shape}")
        if self.type_chart.shape[0] != self.type_chart.shape[1]:
            raise ValueError(f"type_chart has shape {self.type_chart.shape}")
        expected = {
            "move_counts": (2, self.n_cell, n_key, n_move + 1),
            "move_exposure": (2, self.n_cell, n_key, n_move + 1),
            "switch_counts": (self.n_cell, n_key, 2),
            "target_counts": (n_key + 1, n_move + 2, N_TARGET),
            "group_target": (N_GROUP, N_TARGET),
            "mega_counts": (self.n_mega_cell, n_key, 2),
            "dest_counts": (MAX_PATTERN, MAX_PATTERN, 2),
            "reveal_ratio": (N_REVEAL_ROW, N_STATUS),
            "class_ratio": (2 * self.n_cell, N_MOVE_CLASS),
            "move_bits": (n_move,),
            "target_ratio": (N_TARGET_CELL,),
            "move_type": (n_move,),
        }
        for label, shape in expected.items():
            found = getattr(self, label).shape
            if found != shape:
                raise ValueError(f"{label} has shape {found}, expected {shape}")
        self._key_lookup = _lookup(self.key_ids, n_key)
        self._move_lookup = _lookup(self.move_ids, n_move + 1)
        # Type of each move column; OTHER and an unseen move are typeless.
        last = self.type_chart.shape[0] - 1
        self._col_type = np.zeros(n_move + 2, dtype=np.int64)
        self._col_type[:n_move] = np.clip(self.move_type, 0, last)
        self._derive()

    # --- derived probabilities ----------------------------------------------

    def _derive(self) -> None:
        """Shrunk rows from the counts and the configuration (no lazy state)."""
        cfg = self.config
        n_key, n_move = self.key_ids.size, self.move_ids.size
        width = n_move + 2  # moves, the OTHER bucket, one unseen move
        col_class = np.full(width, CLASS_OUTSIDE, dtype=np.int64)
        col_class[:n_move] = coarse_class(self.move_bits)
        one_hot = np.eye(N_MOVE_CLASS, dtype=np.float64)[col_class]  # [width, class]
        uses = np.zeros((2, self.n_cell, n_key + 1, width), dtype=np.float64)
        uses[:, :, :n_key, : n_move + 1] = self.move_counts
        if cfg.usage == USAGE_SHARE:
            counts = uses.sum(0)  # [cell, key, width]
            prior = counts.sum((0, 1)) + cfg.unseen_count
            if prior.sum() <= 0:
                prior = np.ones(width, dtype=np.float64)
            prior = prior / prior.sum()
            by_key = counts.sum(0)
            total = by_key.sum(-1, keepdims=True)
            q_key = _shrink(by_key, total, prior, cfg.key_strength)
            if self.n_cell == 1:
                share = q_key[None]
            else:
                total = counts.sum(-1, keepdims=True)  # [cell, key, 1]
                observed = counts.sum(1) @ one_hot  # [cell, class]
                expected = np.einsum("ck,kj->cj", total[..., 0], q_key @ one_hot)
                effect = _ratio(observed, expected, cfg.flag_effect_strength)
                tilted = q_key[None] * effect[:, col_class][:, None, :]
                tilted = _normalized(tilted, np.broadcast_to(q_key[None], tilted.shape))
                share = _shrink(counts, total, tilted, cfg.cell_strength)
            self._q = np.stack([share, share])
        else:
            seen = np.zeros_like(uses)
            seen[:, :, :n_key, : n_move + 1] = self.move_exposure
            overall = (uses.sum((1, 2, 3)) + _RATE_PRIOR[0]) / (
                seen.sum((1, 2, 3)) + _RATE_PRIOR[1]
            )
            by_move = _shrink(
                uses.sum((1, 2)), seen.sum((1, 2)), overall[:, None], cfg.move_strength
            )  # [2, width]
            by_key = _shrink(
                uses.sum(1), seen.sum(1), by_move[:, None, :], cfg.key_strength
            )  # [2, key, width]
            if self.n_cell == 1:
                self._q = by_key[:, None]
            else:
                observed = np.einsum("sckm,mj->cj", uses, one_hot)
                expected = np.einsum("sckm,skm,mj->cj", seen, by_key, one_hot)
                effect = _ratio(observed, expected, cfg.flag_effect_strength)
                tilted = by_key[:, None] * effect[:, col_class][None, :, None, :]
                tilted = np.clip(tilted, 0.0, 1.0)
                self._q = _shrink(uses, seen, tilted, cfg.cell_strength)
        self._col_class = col_class

        self._switch = self._rates(
            self.switch_counts, cfg.switch_key_strength, cfg.switch_cell_strength
        )
        self._mega = self._rates(self.mega_counts, cfg.mega_strength, cfg.mega_strength)

        group_total = self.group_target.sum(-1, keepdims=True)
        self._group_dist = _shrink(
            self.group_target,
            group_total,
            np.full(N_TARGET, 1.0 / N_TARGET),
            _GROUP_PRIOR_STRENGTH,
        )
        self._target_move = self.target_counts.sum(0)  # [width, 5]

        dest = self.dest_counts
        shown = np.arange(MAX_PATTERN, dtype=np.float64)[:, None]
        hidden = np.arange(MAX_PATTERN, dtype=np.float64)[None, :]
        pointers = np.maximum(shown + hidden, 1.0)
        mixed = (shown > 0) & (hidden > 0)
        to_shown, to_hidden = dest[..., 0], dest[..., 1] - dest[..., 0]
        above = float((to_shown * hidden / pointers)[mixed].sum())
        below = float((to_hidden * shown / pointers)[mixed].sum())
        # Mantel-Haenszel pooled odds of a shown pointer against a hidden one.
        odds = above / below if above > 0 and below > 0 else 1.0
        pooled = np.divide(
            odds * shown,
            odds * shown + hidden,
            out=np.zeros_like(pointers),
            where=(shown + hidden) > 0,
        )
        self._dest_odds = odds
        self._dest_shown = _shrink(to_shown, dest[..., 1], pooled, _DEST_STRENGTH)
        self._dest_shown = np.where(hidden > 0, self._dest_shown, 1.0)
        self._dest_shown = np.where(shown > 0, self._dest_shown, 0.0)

    def _rates(
        self, counts: np.ndarray, key_strength: float, cell_strength: float
    ) -> np.ndarray:
        """``[cell, key + 1]`` event rates from ``[cell, key, (events, trials)]``."""
        n_cell, n_key = counts.shape[0], counts.shape[1]
        events = np.zeros((n_cell, n_key + 1), dtype=np.float64)
        trials = np.zeros((n_cell, n_key + 1), dtype=np.float64)
        events[:, :n_key], trials[:, :n_key] = counts[..., 0], counts[..., 1]
        overall = (events.sum() + _RATE_PRIOR[0]) / (trials.sum() + _RATE_PRIOR[1])
        by_key = _shrink(events.sum(0), trials.sum(0), np.array(overall), key_strength)
        if n_cell == 1:
            return by_key[None]
        expected = trials @ by_key  # [cell]
        effect = _ratio(events.sum(1), expected, self.config.flag_effect_strength)
        tilted = np.clip(by_key[None] * effect[:, None], 0.0, 1.0)
        return _shrink(events, trials, tilted, cell_strength)

    # --- batch -> rows ------------------------------------------------------

    def _cells(self, view: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """(action cell, Mega cell) per slot, ``[N, 2]`` each."""
        turn1 = np.asarray(view["turn1"]).astype(np.int64)
        return np.zeros_like(turn1), turn1

    def _rows(self, arrays: Mapping[str, np.ndarray]) -> _Rows:
        view = slot_view(arrays)
        if not view:
            raise ValueError("batch lacks the per-slot arrays")
        active = np.asarray(view["active"]).astype(bool)
        n_batch = active.shape[0]
        n = n_batch * 2
        moves = np.asarray(arrays["cand_move"]).astype(np.int64)
        bits = np.asarray(arrays["cand_flag"]).astype(np.int64)
        legal = np.asarray(arrays["action_mask"]).astype(bool) & active[..., None]
        n_cand = moves.shape[-1]
        if legal.shape[-1] != n_cand + 1 + N_ROSTER or bits.shape != moves.shape:
            raise ValueError("action arrays do not agree on the candidate count")
        moves, bits = moves.reshape(n, n_cand), bits.reshape(n, n_cand)
        legal = legal.reshape(n, legal.shape[-1])  # explicit: n may be 0
        valid = legal[:, :n_cand]
        shown = valid & ((bits & CAND_REVEALED) > 0)
        sheet = (valid & ((bits & CAND_SHEET) > 0)).any(-1)
        n_shown = np.minimum(shown.sum(-1), MAX_MOVES)
        status = np.where(
            shown,
            STATUS_SHOWN,
            np.where((bits & CAND_GLOBAL) > 0, STATUS_GLOBAL, STATUS_NOT_SHOWN),
        )
        bench = np.asarray(arrays["mon_flag"])[:, :N_ROSTER, FLAG_REVEALED] > 0
        bench = np.repeat(bench, 2, axis=0)  # one row per slot
        pointers = legal[:, n_cand + 1 :]
        cell, mega_cell = self._cells(view)
        classes = np.full(legal.shape, CLASS_SWITCH, dtype=np.int64)
        classes[:, :n_cand] = coarse_class(bits)
        classes[:, n_cand] = CLASS_OUTSIDE
        return _Rows(
            shape=(n_batch, 2),
            n_cand=n_cand,
            active=active.reshape(n),
            key=_gather(self._key_lookup, np.asarray(view["key"]).reshape(n)),
            cell=np.asarray(cell).reshape(n),
            mega_cell=np.asarray(mega_cell).reshape(n),
            mega_possible=np.asarray(view["mega_possible"]).astype(bool).reshape(n),
            col=_gather(self._move_lookup, moves),
            bits=bits,
            legal=legal,
            status=status,
            reveal_row=sheet.astype(np.int64) * (MAX_MOVES + 1) + n_shown,
            known=np.where(shown | ((bits & CAND_SHEET) > 0), KNOWN, NOT_KNOWN),
            class_row=np.asarray(cell).reshape(n) + self.n_cell * sheet,
            complete=sheet | (n_shown >= MAX_MOVES),
            shown_bench=pointers & bench,
            hidden_bench=pointers & ~bench,
            elo=np.asarray(view["elo"]).astype(np.int64).reshape(n),
            elo_known=np.asarray(view["elo_known"]).astype(bool).reshape(n),
            action_class=classes,
        )

    # --- probabilities ------------------------------------------------------

    def _move_part(self, rows: _Rows, fitting: bool) -> np.ndarray:
        """``[n, C + 1]``: candidates and OTHER, given that the slot moves."""
        cfg = self.config
        n_cand = rows.n_cand
        valid = rows.legal[:, :n_cand]
        mass = self._q[rows.known, rows.cell[:, None], rows.key[:, None], rows.col]
        mass = np.where(valid, mass, 0.0)
        if cfg.usage == USAGE_SHARE:
            outside = np.clip(1.0 - mass.sum(-1), 0.0, 1.0)
        else:
            outside = self._q[NOT_KNOWN, rows.cell, rows.key, self.move_ids.size]
        ratio = self.reveal_ratio[rows.reveal_row]  # [n, status]
        by_class = self.class_ratio[rows.class_row]  # [n, class]
        mass = mass * np.take_along_axis(ratio, rows.status, axis=1)
        mass = mass * np.take_along_axis(
            by_class, rows.action_class[:, :n_cand], axis=1
        )
        boost = 1.0 if fitting else cfg.other_boost
        other = outside * ratio[:, STATUS_OTHER] * by_class[:, CLASS_OUTSIDE] * boost
        other = np.where(rows.complete | ~rows.legal[:, n_cand], 0.0, other)
        part = np.concatenate([mass, other[:, None]], axis=1)
        return _normalized(part, rows.legal[:, : n_cand + 1].astype(np.float64))

    def _switch_part(self, rows: _Rows) -> np.ndarray:
        """``[n, 6]``: where a switch goes, given that the slot switches."""
        n_shown = rows.shown_bench.sum(-1)
        n_hidden = rows.hidden_bench.sum(-1)
        if self.config.switch_dest == DEST_UNIFORM:
            total = np.maximum(n_shown + n_hidden, 1)
            return (rows.shown_bench | rows.hidden_bench) / total[:, None]
        to_shown = self._dest_shown[
            np.minimum(n_shown, MAX_PATTERN - 1), np.minimum(n_hidden, MAX_PATTERN - 1)
        ]
        per_shown = to_shown / np.maximum(n_shown, 1)
        per_hidden = (1.0 - to_shown) / np.maximum(n_hidden, 1)
        return (
            rows.shown_bench * per_shown[:, None]
            + rows.hidden_bench * per_hidden[:, None]
        )

    def _adjust(self, action: np.ndarray, rows: _Rows) -> np.ndarray:
        """Hook of ``EloTable``; the other tables leave the action as it is."""
        return action

    def _action(self, rows: _Rows, fitting: bool = False) -> np.ndarray:
        """``[n, A]`` action probabilities: zero off the mask, floored on it."""
        n_cand = rows.n_cand
        moves = self._move_part(rows, fitting)
        can_move = rows.legal[:, : n_cand + 1].any(-1)
        can_switch = rows.legal[:, n_cand + 1 :].any(-1)
        rate = self._switch[rows.cell, rows.key]
        rate = np.where(can_switch, np.where(can_move, rate, 1.0), 0.0)
        action = np.concatenate(
            [moves * (1.0 - rate)[:, None], self._switch_part(rows) * rate[:, None]],
            axis=1,
        )
        if not fitting:
            action = self._adjust(action, rows)
        return _floored(action, rows.legal, self.config.floor)

    def _target(self, rows: _Rows, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
        """``[n, C + 1, 5]`` target-class probabilities given each move action."""
        dist, legal = self._target_rows(rows, arrays)
        if self.config.target_context:
            cells = self._target_cells(rows, arrays)
            if cells is None:
                self._count("target_context_unread")
            else:
                dist = self._with_context(dist, legal, cells, self.target_ratio)
        return _floored(dist, legal, self.config.floor)

    def _target_cells(
        self, rows: _Rows, arrays: Mapping[str, np.ndarray]
    ) -> np.ndarray | None:
        """``[n, C, 2]``: the context cell of each candidate against foe a, b.

        None when the batch does not carry the foes' types and HP.
        """
        if any(name not in arrays for name in _CONTEXT_FEATURES):
            return None
        n, n_cand = rows.active.size, rows.n_cand
        n_batch = rows.shape[0]
        foe = np.asarray(arrays["foe_mon"]).astype(np.int64)  # [N, 2]
        if foe.shape != (n_batch, 2):
            return None
        types = np.asarray(arrays["mon_type"]).astype(np.int64)  # [N, 12, T]
        hp = np.asarray(arrays["mon_hp"]).astype(np.float64)  # [N, 12]
        at = np.clip(foe, 0, types.shape[1] - 1)
        example = np.arange(n_batch)[:, None]
        last = self.type_chart.shape[0] - 1
        foe_type = np.clip(types[example, at], 0, last)  # [N, 2, T]
        foe_type = np.where((foe >= 0)[..., None], foe_type, 0)
        foe_hp = np.where(foe >= 0, hp[example, at], 0.0)  # [N, 2]
        # One row per slot: both slots of an example face the same two foes.
        foe_type = np.repeat(foe_type, 2, axis=0)  # [n, 2, T]
        step = np.sign(np.round((foe_hp[:, 0] - foe_hp[:, 1]) * _HP_STEPS))
        step = np.repeat(step.astype(np.int64), 2)  # [n]: -1 a lower, 0, 1 a higher
        relation = np.stack([step + 1, 1 - step], axis=-1)  # per foe: 0 / 1 / 2
        move_type = self._col_type[rows.col]  # [n, C]
        effect = np.ones((n, n_cand, 2), dtype=np.float64)
        for position in range(foe_type.shape[-1]):
            effect *= self.type_chart[
                move_type[:, :, None], foe_type[:, None, :, position]
            ]
        bucket = np.where(effect < 1.0, 0, np.where(effect > 1.0, 2, 1))
        cell = (
            bucket * (N_EFFECT * N_HP_RELATION)
            + bucket[..., ::-1] * N_HP_RELATION
            + relation[:, None, :]
        )
        damaging = (rows.bits & CAND_DAMAGING) > 0
        return np.where(damaging[..., None], cell, _STATUS_CELL + relation[:, None, :])

    @staticmethod
    def _with_context(
        dist: np.ndarray, legal: np.ndarray, cells: np.ndarray, ratio: np.ndarray
    ) -> np.ndarray:
        """``dist`` with the two foes' mass re-split by the context ratios.

        Only where both foes are legal targets of a candidate; the sum of the
        two stays what it was, so no other target class moves. OTHER (the last
        row) is left alone: its move is not known.
        """
        n_cand = cells.shape[1]
        out = dist.copy()
        first, second = out[:, :n_cand, T_FOE_A], out[:, :n_cand, T_FOE_B]
        both = legal[:, :n_cand, T_FOE_A] & legal[:, :n_cand, T_FOE_B]
        tilted_a = first * ratio[cells[..., 0]]
        tilted_b = second * ratio[cells[..., 1]]
        total = tilted_a + tilted_b
        use = both & (total > 0)
        scale = np.divide(first + second, total, out=np.zeros_like(total), where=use)
        out[:, :n_cand, T_FOE_A] = np.where(use, tilted_a * scale, first)
        out[:, :n_cand, T_FOE_B] = np.where(use, tilted_b * scale, second)
        return out

    def _target_rows(
        self, rows: _Rows, arrays: Mapping[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """(target probabilities before the context and the floor, legal mask)."""
        cfg = self.config
        n, n_cand = rows.active.size, rows.n_cand
        legal = expand_target_mask(arrays["cand_tmask"]).reshape(
            n, n_cand + 1, N_TARGET
        )
        legal = legal & rows.legal[:, : n_cand + 1, None]
        n_move = self.move_ids.size
        col = np.concatenate([rows.col, np.full((n, 1), n_move)], axis=1)
        group = np.concatenate(
            [_class_group(rows.bits), np.full((n, 1), GROUP_OTHER)], axis=1
        )
        by_move = self._target_move[col]
        dist = _shrink(
            by_move,
            by_move.sum(-1, keepdims=True),
            self._group_dist[group],
            cfg.target_move_strength,
        )
        by_key = self.target_counts[rows.key[:, None], col]
        dist = _shrink(
            by_key, by_key.sum(-1, keepdims=True), dist, cfg.target_key_strength
        )
        dist = _normalized(np.where(legal, dist, 0.0), legal.astype(np.float64))
        if cfg.use_cand_auto:
            share = np.asarray(arrays["cand_auto"], dtype=np.float64).reshape(n, n_cand)
            share = np.clip(share, _AUTO_CLIP, 1.0 - _AUTO_CLIP)
            aimed = (rows.bits & CAND_AIMED) > 0
            rest = legal[:, :n_cand].copy()
            rest[..., T_AUTO] = False
            split = aimed & legal[:, :n_cand, T_AUTO] & rest.any(-1)
            aimed_part = _normalized(
                np.where(rest, dist[:, :n_cand], 0.0), rest.astype(np.float64)
            ) * (1.0 - share[..., None])
            aimed_part[..., T_AUTO] = share
            dist[:, :n_cand] = np.where(split[..., None], aimed_part, dist[:, :n_cand])
        return dist, legal

    def _mega_rate(self, rows: _Rows, action: np.ndarray) -> np.ndarray:
        """``[n]``: P(the slot moves) times the Mega rate of a slot that moves."""
        floor = self.config.floor
        moving = action[:, : rows.n_cand + 1].sum(-1)
        rate = np.clip(self._mega[rows.mega_cell, rows.key] * moving, floor, 1 - floor)
        return np.where(rows.active & rows.mega_possible, rate, floor)

    def predict(self, batch: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Probabilities for every slot of a batch. Never raises; reads no label."""
        try:
            arrays = _feature_arrays(batch)
            rows = self._rows(arrays)
            shape = rows.shape
            action = self._action(rows)
            target = self._target(rows, arrays)
            mega = self._mega_rate(rows, action)
            if not (np.isfinite(action).all() and np.isfinite(target).all()):
                raise FloatingPointError("non-finite probability")
            return {
                "action": action.reshape(*shape, action.shape[-1]),
                "target": target.reshape(*shape, rows.n_cand + 1, N_TARGET),
                "mega": mega.reshape(shape),
            }
        except Exception as exc:
            self._count(f"predict_error:{type(exc).__name__}")
            return uniform_prediction(batch)

    def _count(self, name: str) -> None:
        try:
            self.counters[name] += 1
        except Exception:
            pass

    # --- fitting ------------------------------------------------------------

    @classmethod
    def default_config(cls) -> TableConfig:
        return TableConfig()

    @classmethod
    def fit(
        cls: type[TableT],
        batch: Mapping[str, np.ndarray],
        *,
        mask: np.ndarray | None = None,
        config: TableConfig | None = None,
        featurizer: Any = None,
        name: str | None = None,
    ) -> TableT:
        """Count the rows of ``batch`` that ``mask`` keeps (all, when None).

        Reads ``m_weight`` as the example weight when the batch has it. Rows
        outside the mask are dropped before anything else is looked at.
        ``featurizer`` only supplies the names stored next to the ids. Raises
        ``ValueError`` on a batch without the needed arrays.
        """
        config = config or cls.default_config()
        if config.censored not in CENSORED_MODES:
            raise ValueError(f"unknown censored mode {config.censored!r}")
        if config.switch_dest not in (DEST_REVEALED, DEST_UNIFORM):
            raise ValueError(f"unknown switch_dest {config.switch_dest!r}")
        if config.usage not in (USAGE_RATE, USAGE_SHARE):
            raise ValueError(f"unknown usage {config.usage!r}")
        arrays = _select(batch, mask)
        view = slot_view(arrays)
        if not view:
            raise ValueError("batch lacks the per-slot arrays")
        active = np.asarray(view["active"]).astype(bool)
        cand = arrays["cand_move"].astype(np.int64)
        flags = arrays["cand_flag"].astype(np.int64)
        n_cand = cand.shape[-1]
        valid = arrays["action_mask"].astype(bool)[..., :n_cand] & active[..., None]
        key_ids = np.unique(np.asarray(view["key"]).astype(np.int64)[active])
        move_ids, first = np.unique(cand[valid], return_index=True)
        move_bits = flags[valid][first] & _ALL_CLASS_BITS
        n_key, n_move = key_ids.size, move_ids.size
        move_type, type_chart, type_names = type_context(featurizer, move_ids)
        table = cls(
            config=config,
            key_ids=key_ids,
            move_ids=move_ids,
            move_bits=move_bits,
            move_counts=np.zeros((2, cls.n_cell, n_key, n_move + 1)),
            move_exposure=np.zeros((2, cls.n_cell, n_key, n_move + 1)),
            switch_counts=np.zeros((cls.n_cell, n_key, 2)),
            target_counts=np.zeros((n_key + 1, n_move + 2, N_TARGET)),
            group_target=np.zeros((N_GROUP, N_TARGET)),
            mega_counts=np.zeros((cls.n_mega_cell, n_key, 2)),
            dest_counts=np.zeros((MAX_PATTERN, MAX_PATTERN, 2)),
            reveal_ratio=np.ones((N_REVEAL_ROW, N_STATUS)),
            class_ratio=np.ones((2 * cls.n_cell, N_MOVE_CLASS)),
            key_names=_names(featurizer, "species_ids", key_ids),
            move_names=_names(featurizer, "move_ids", move_ids),
            name=name,
            move_type=move_type,
            type_chart=type_chart,
            type_names=type_names,
        )
        table._fit(arrays)
        return table

    def _labels(
        self, rows: _Rows, arrays: Mapping[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(y_set on the mask, y_action, slot weight, visible) per slot."""
        n = rows.active.size
        y_set = arrays["y_set"].astype(bool).reshape(rows.legal.shape) & rows.legal
        y_action = arrays["y_action"].astype(np.int64).reshape(n)
        if _WEIGHT in arrays:
            weight = np.repeat(arrays[_WEIGHT].astype(np.float64), 2)
        else:
            weight = np.ones(n, dtype=np.float64)
        visible = (y_action >= 0) & (y_action < y_set.shape[1])
        visible &= np.take_along_axis(
            y_set, np.clip(y_action, 0, y_set.shape[1] - 1)[:, None], axis=1
        )[:, 0]
        return y_set, y_action, weight, visible

    def _responsibility(
        self,
        rows: _Rows,
        y_set: np.ndarray,
        y_action: np.ndarray,
        visible: np.ndarray,
        action: np.ndarray | None,
    ) -> np.ndarray:
        """``[n, A]``: each slot's weight share per action (rows sum to 1 or 0)."""
        out = np.zeros(y_set.shape, dtype=np.float64)
        where = np.flatnonzero(visible)
        out[where, y_action[where]] = 1.0
        if action is not None and self.config.censored != CENSORED_DROP:
            hidden = ~visible & y_set.any(-1)
            share = _normalized(np.where(y_set, action, 0.0), y_set.astype(np.float64))
            out = np.where(hidden[:, None], share, out)
        return out

    def _accumulate(
        self, rows: _Rows, share: np.ndarray, weight: np.ndarray, visible: np.ndarray
    ) -> None:
        """Move and switch counts from the slots' action shares."""
        n_cand = rows.n_cand
        n_key, n_move = self.key_ids.size, self.move_ids.size
        width = n_move + 1
        used = share.sum(-1) > 0
        key = np.minimum(rows.key, max(n_key - 1, 0))
        row = rows.cell * n_key + key
        col = np.minimum(rows.col, n_move)  # an unseen column cannot occur in fit
        index = np.concatenate(
            [
                ((rows.known * self.n_cell * n_key + row[:, None]) * width) + col,
                ((NOT_KNOWN * self.n_cell * n_key + row) * width + n_move)[:, None],
            ],
            axis=1,
        )
        weights = share[:, : n_cand + 1] * weight[:, None]
        shape = (2, self.n_cell, n_key, width)
        size = int(np.prod(shape))
        seen = _bincount(index[visible], weights[visible], size).reshape(shape)
        unseen = _bincount(index[~visible], weights[~visible], size).reshape(shape)
        if self.config.censored == CENSORED_FRACTIONAL:
            # Keep each row's visible mix inside the Protect family and outside
            # it; the censored weight only changes the two groups' totals.
            guard = np.zeros(width, dtype=bool)
            guard[:n_move] = coarse_class(self.move_bits) == CLASS_PROTECT
            counts = np.zeros(shape, dtype=np.float64)
            for group in (guard, ~guard):
                have = seen[..., group].sum((0, -1), keepdims=True)
                more = unseen[..., group].sum((0, -1), keepdims=True)
                scale = 1.0 + np.divide(
                    more, have, out=np.zeros_like(more), where=have > 0
                )
                counts[..., group] = np.where(
                    have > 0, seen[..., group] * scale, unseen[..., group]
                )
            self.move_counts = counts
        else:
            self.move_counts = seen + unseen
        # Exposure: the slot chose a move while this candidate (or OTHER, with
        # the moveset still open) was on offer.
        moved = share[:, : n_cand + 1].sum(-1) * weight
        offered = np.concatenate(
            [rows.legal[:, :n_cand], (rows.legal[:, n_cand] & ~rows.complete)[:, None]],
            axis=1,
        )
        self.move_exposure = _bincount(index, offered * moved[:, None], size).reshape(
            shape
        )
        can_switch = rows.legal[:, n_cand + 1 :].any(-1) & used
        slot = rows.cell * n_key + key
        switched = share[:, n_cand + 1 :].sum(-1) * weight
        size = self.n_cell * n_key
        self.switch_counts = np.stack(
            [
                _bincount(slot[can_switch], switched[can_switch], size),
                _bincount(slot[can_switch], weight[can_switch], size),
            ],
            axis=-1,
        ).reshape(self.n_cell, n_key, 2)

    def _fit_offsets(
        self,
        rows: _Rows,
        action: np.ndarray,
        share: np.ndarray,
        weight: np.ndarray,
        reveal: bool,
    ) -> None:
        """One proportional-fitting step of the reveal or the class ratios.

        Observed: the slots' weight shares per candidate status (or coarse
        class). A censored slot enters with its share over its set, so the
        step does not inherit the visible labels' surplus of Protect.
        Expected: the table's own move probabilities on the same slots.
        """
        n_cand = rows.n_cand
        moved = share[:, : n_cand + 1].sum(-1)
        where = np.flatnonzero(moved > 0)
        moves = action[where, : n_cand + 1]
        moves = _normalized(moves, np.ones_like(moves)) * moved[where, None]
        if reveal:
            column = np.concatenate(
                [rows.status, np.full((rows.active.size, 1), STATUS_OTHER)], axis=1
            )[where]
            row, width, current = rows.reveal_row[where], N_STATUS, self.reveal_ratio
        else:
            column = rows.action_class[where, : n_cand + 1]
            row, width, current = rows.class_row[where], N_MOVE_CLASS, self.class_ratio
        index = row[:, None] * width + column
        slot_weight = weight[where, None]
        observed = _bincount(
            index, share[where, : n_cand + 1] * slot_weight, current.size
        )
        expected = _bincount(index, moves * slot_weight, current.size)
        step = _ratio(observed, expected, self.config.offset_strength)
        if reveal:
            self.reveal_ratio = current * step.reshape(current.shape)
        else:
            self.class_ratio = current * step.reshape(current.shape)

    def _fit(self, arrays: Mapping[str, np.ndarray]) -> None:
        cfg = self.config
        rows = self._rows(arrays)
        n, n_cand = rows.active.size, rows.n_cand
        y_set, y_action, weight, visible = self._labels(rows, arrays)
        share = self._responsibility(rows, y_set, y_action, visible, None)
        self._accumulate(rows, share, weight, visible)
        self._derive()
        for _ in range(max(0, int(cfg.iterations))):
            action = self._action(rows, fitting=True)
            share = self._responsibility(rows, y_set, y_action, visible, action)
            if cfg.reveal_offsets:
                self._fit_offsets(rows, action, share, weight, reveal=True)
                action = self._action(rows, fitting=True)
            if cfg.class_offsets:
                self._fit_offsets(rows, action, share, weight, reveal=False)
            self._accumulate(rows, share, weight, visible)
            self._derive()

        n_key, n_move = self.key_ids.size, self.move_ids.size
        key = np.minimum(rows.key, max(n_key - 1, 0))

        y_target = arrays["y_target"].astype(np.int64).reshape(n)
        aimed = visible & (y_action <= n_cand) & (y_target >= 0)
        aimed &= y_target < N_TARGET
        where = np.flatnonzero(aimed)
        chosen = np.minimum(y_action[where], n_cand - 1)
        is_other = y_action[where] == n_cand
        col = np.where(is_other, n_move, rows.col[where, chosen])
        group = np.where(is_other, GROUP_OTHER, _class_group(rows.bits[where, chosen]))
        width = n_move + 2
        self.target_counts = _bincount(
            (key[where] * width + col) * N_TARGET + y_target[where],
            weight[where],
            (n_key + 1) * width * N_TARGET,
        ).reshape(n_key + 1, width, N_TARGET)
        self.group_target = _bincount(
            group * N_TARGET + y_target[where], weight[where], N_GROUP * N_TARGET
        ).reshape(N_GROUP, N_TARGET)
        self._derive()
        context_labels = self._fit_context(
            rows, arrays, where[~is_other], chosen[~is_other], y_target, weight
        )

        y_mega = arrays["y_mega"].astype(np.int64).reshape(n)
        told = (
            rows.active
            & rows.mega_possible
            & (y_mega >= 0)
            & ~(visible & (y_action > n_cand))
        )
        slot = (rows.mega_cell * n_key + key)[told]
        size = self.n_mega_cell * n_key
        self.mega_counts = np.stack(
            [
                _bincount(slot, weight[told] * (y_mega[told] == 1), size),
                _bincount(slot, weight[told], size),
            ],
            axis=-1,
        ).reshape(self.n_mega_cell, n_key, 2)

        switched = visible & (y_action > n_cand)
        where = np.flatnonzero(switched)
        pointer = y_action[where] - n_cand - 1
        to_shown = rows.shown_bench[where, pointer]
        pattern = np.minimum(
            rows.shown_bench[where].sum(-1), MAX_PATTERN - 1
        ) * MAX_PATTERN + np.minimum(rows.hidden_bench[where].sum(-1), MAX_PATTERN - 1)
        size = MAX_PATTERN * MAX_PATTERN
        self.dest_counts = np.stack(
            [
                _bincount(pattern, weight[where] * to_shown, size),
                _bincount(pattern, weight[where], size),
            ],
            axis=-1,
        ).reshape(MAX_PATTERN, MAX_PATTERN, 2)
        self._derive()

        scored = y_set.any(-1)
        self.fit_info = {
            "examples": int(rows.shape[0]),
            "slots_active": int(rows.active.sum()),
            "slots_visible": int(visible.sum()),
            "slots_censored_with_set": int((scored & ~visible).sum()),
            "slots_counted": int((share.sum(-1) > 0).sum()),
            "weight_counted": float(weight[share.sum(-1) > 0].sum()),
            "set_keys": int(n_key),
            "moves": int(n_move),
            "target_labels": int(aimed.sum()),
            "target_context_labels": int(context_labels),
            "mega_labels": int(told.sum()),
            "switch_labels": int(switched.sum()),
            "switch_dest_odds_shown": float(self._dest_odds),
        }
        self._fit_extra(rows, arrays, share, weight)

    def _fit_extra(
        self,
        rows: _Rows,
        arrays: Mapping[str, np.ndarray],
        share: np.ndarray,
        weight: np.ndarray,
    ) -> None:
        """Hook of ``EloTable``."""

    def _fit_context(
        self,
        rows: _Rows,
        arrays: Mapping[str, np.ndarray],
        slots: np.ndarray,
        columns: np.ndarray,
        y_target: np.ndarray,
        weight: np.ndarray,
    ) -> int:
        """Fit ``target_ratio`` by proportional fitting; returns the labels used.

        ``slots`` / ``columns`` are the labelled candidate moves (slot row and
        candidate column). Used: those aimed at one of two legal foes.
        Observed is the label weight per cell of the foe that was aimed at,
        expected the table's own split between the two foes under the ratios
        so far, both summed over the two foes of every label.
        """
        cfg = self.config
        self.target_ratio = np.ones(N_TARGET_CELL)
        cells = self._target_cells(rows, arrays) if cfg.target_context else None
        if cells is None or slots.size == 0:
            return 0
        dist, legal = self._target_rows(rows, arrays)
        aimed = y_target[slots]
        keep = (aimed == T_FOE_A) | (aimed == T_FOE_B)
        keep &= legal[slots, columns, T_FOE_A] & legal[slots, columns, T_FOE_B]
        slots, columns = slots[keep], columns[keep]
        if slots.size == 0:
            return 0
        base = dist[slots, columns][:, [T_FOE_A, T_FOE_B]]
        base = _normalized(base, np.ones_like(base))
        cell = cells[slots, columns]  # [labels, 2]
        at_b = y_target[slots] == T_FOE_B
        load = weight[slots]
        observed = _bincount(cell[:, 0], load * ~at_b, N_TARGET_CELL)
        observed += _bincount(cell[:, 1], load * at_b, N_TARGET_CELL)
        ratio = np.ones(N_TARGET_CELL)
        for _ in range(max(1, int(cfg.iterations))):
            split = base * ratio[cell]
            split = _normalized(split, np.ones_like(split))
            expected = _bincount(cell[:, 0], load * split[:, 0], N_TARGET_CELL)
            expected += _bincount(cell[:, 1], load * split[:, 1], N_TARGET_CELL)
            ratio = ratio * _ratio(observed, expected, cfg.target_context_strength)
        self.target_ratio = ratio
        return int(slots.size)

    # --- copies and storage -------------------------------------------------

    def _state(self) -> dict[str, Any]:
        return {
            "key_ids": self.key_ids,
            "move_ids": self.move_ids,
            "move_bits": self.move_bits,
            "move_counts": self.move_counts,
            "move_exposure": self.move_exposure,
            "switch_counts": self.switch_counts,
            "target_counts": self.target_counts,
            "group_target": self.group_target,
            "mega_counts": self.mega_counts,
            "dest_counts": self.dest_counts,
            "reveal_ratio": self.reveal_ratio,
            "class_ratio": self.class_ratio,
            "key_names": self.key_names,
            "move_names": self.move_names,
            "fit_info": self.fit_info,
            "name": self.name,
            "target_ratio": self.target_ratio,
            "move_type": self.move_type,
            "type_chart": self.type_chart,
            "type_names": self.type_names,
        }

    def with_config(self: TableT, **changes: Any) -> TableT:
        """The same counts under a changed configuration (nothing is refitted).

        Strengths, ``other_boost``, ``floor``, ``switch_dest``,
        ``use_cand_auto`` and ``target_context`` (the fitted ratios stay with
        the table; False stops reading them) take effect at once.
        ``censored``, ``iterations``, ``reveal_offsets``, ``elo_edges`` and
        ``target_context_strength`` only describe how the counts and ratios
        were made, so changing them here changes nothing: refit instead.
        """
        return type(self)(config=replace(self.config, **changes), **self._state())

    def to_payload(self) -> dict[str, Any]:
        """Plain data and numpy arrays: what ``from_payload`` rebuilds from."""
        state = self._state()
        payload: dict[str, Any] = {
            "payload_version": PAYLOAD_VERSION,
            "kind": KIND,
            "table": self.table,
            "name": self.name,
            "config": self.config.to_dict(),
            "class_names": list(CLASS_NAMES),
            "key_ids": self.key_ids.copy(),
            "move_ids": self.move_ids.copy(),
            "move_bits": self.move_bits.copy(),
            "key_names": None if self.key_names is None else list(self.key_names),
            "move_names": None if self.move_names is None else list(self.move_names),
            "reveal_ratio": self.reveal_ratio.copy(),
            "class_ratio": self.class_ratio.copy(),
            "target_ratio": self.target_ratio.copy(),
            "move_type": self.move_type.copy(),
            "type_chart": self.type_chart.copy(),
            "type_names": list(self.type_names),
            "fit_info": dict(self.fit_info),
        }
        for label in (
            "move_counts",
            "move_exposure",
            "switch_counts",
            "target_counts",
            "group_target",
            "mega_counts",
            "dest_counts",
        ):
            payload[label] = _pack(state[label])
        return payload

    @classmethod
    def _payload_state(
        cls, payload: Mapping[str, Any], featurizer: Any
    ) -> dict[str, Any]:
        if str(payload.get("table")) != cls.table:
            raise ValueError(f"payload holds {payload.get('table')!r}, not {cls.table}")
        if int(payload["payload_version"]) != PAYLOAD_VERSION:
            raise ValueError(f"payload version {payload['payload_version']}")
        key_names, move_names = payload.get("key_names"), payload.get("move_names")
        context: dict[str, Any] = {}
        if payload.get("target_ratio") is not None:
            # Absent in a payload written before the target context existed.
            move_type, chart, names = _retyped(
                np.asarray(payload["move_type"]),
                np.asarray(payload["type_chart"], dtype=np.float64),
                payload.get("type_names") or (),
                featurizer,
            )
            context = {
                "target_ratio": np.asarray(payload["target_ratio"], dtype=np.float64),
                "move_type": move_type,
                "type_chart": chart,
                "type_names": names,
            }
        return {
            **context,
            "config": TableConfig.from_dict(payload["config"]),
            "key_ids": _remap(payload["key_ids"], key_names, featurizer, "species_ids"),
            "move_ids": _remap(payload["move_ids"], move_names, featurizer, "move_ids"),
            "move_bits": np.asarray(payload["move_bits"]),
            "move_counts": _unpack(payload["move_counts"]),
            "move_exposure": _unpack(payload["move_exposure"]),
            "switch_counts": _unpack(payload["switch_counts"]),
            "target_counts": _unpack(payload["target_counts"]),
            "group_target": _unpack(payload["group_target"]),
            "mega_counts": _unpack(payload["mega_counts"]),
            "dest_counts": _unpack(payload["dest_counts"]),
            "reveal_ratio": np.asarray(payload["reveal_ratio"], dtype=np.float64),
            "class_ratio": np.asarray(payload["class_ratio"], dtype=np.float64),
            "key_names": key_names,
            "move_names": move_names,
            "fit_info": payload.get("fit_info"),
            "name": payload.get("name"),
        }

    @classmethod
    def from_payload(
        cls: type[TableT], payload: Mapping[str, Any], featurizer: Any = None
    ) -> TableT:
        """Rebuild a table from ``to_payload()``. Raises ``ValueError`` when malformed.

        With a featurizer and stored names, ids are looked up again by name, so
        a payload survives a vocabulary whose order changed; a name the
        vocabulary lacks keeps its row but can no longer be reached.
        """
        try:
            return cls(**cls._payload_state(payload, featurizer))
        except (KeyError, TypeError, IndexError, AttributeError) as exc:
            raise ValueError(f"not a {cls.table} payload: {exc!r}") from exc


def _names(featurizer: Any, attribute: str, ids: np.ndarray) -> list[str] | None:
    """Names of vocabulary ids, when a featurizer is at hand."""
    try:
        universe = getattr(featurizer.tables, attribute)
        return [str(universe[int(index)]) for index in ids]
    except Exception:
        return None


def _remap(
    ids: Any, names: Sequence[str] | None, featurizer: Any, attribute: str
) -> np.ndarray:
    """Stored ids, re-read by name from the featurizer's vocabulary when possible.

    A name the vocabulary lacks gets an id nothing in a batch can equal, so its
    row stays in the table but is never read.
    """
    stored = np.asarray(ids).astype(np.int64).ravel()
    if featurizer is None or names is None or len(names) != stored.size:
        return stored
    try:
        universe = [str(item) for item in getattr(featurizer.tables, attribute)]
    except Exception:
        return stored
    index: dict[str, int] = {}
    for position, item in enumerate(universe):
        index.setdefault(item, position)
    spare = max(len(universe), int(stored.max()) + 1 if stored.size else 0)
    out = stored.copy()
    for position, item in enumerate(names):
        found = index.get(str(item))
        if found is None:
            out[position] = spare
            spare += 1
        else:
            out[position] = found
    if np.unique(out).size != out.size:
        raise ValueError(f"{attribute}: two stored names share one vocabulary id")
    return out


class SpeciesTable(_CountTable):
    """Set key -> action frequencies; Mega rate per (set key, turn 1). The floor."""

    table = TABLE_SPECIES
    n_cell = 1
    n_mega_cell = 2


class FlagsTable(_CountTable):
    """Set key x (turn 1, first turn on field, protected last turn). The bar."""

    table = TABLE_FLAGS
    n_cell = N_FLAG_CELL
    n_mega_cell = N_FLAG_CELL

    def _cells(self, view: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        cell = (
            np.asarray(view["turn1"]).astype(np.int64) * 4
            + np.asarray(view["first_turn"]).astype(np.int64) * 2
            + np.asarray(view["protected_last"]).astype(np.int64)
        )
        return cell, cell


class EloTable(FlagsTable):
    """``FlagsTable`` times an Elo-band x coarse-class ratio.

    ``P'(action) ~ P(action) * ratio[band of the actor's rating, class]`` with
    ``ratio = (observed + elo_strength) / (expected + elo_strength)`` class
    weight per band over the fit rows: one proportional-fitting step, the
    lowest-variance way to let a rating move the action mix. A slot whose
    rating is unknown gets exactly the ``FlagsTable`` prediction, and so does
    every slot at infinite ``elo_strength``.
    """

    table = TABLE_ELO

    def __init__(
        self,
        *,
        elo_observed: np.ndarray | None = None,
        elo_expected: np.ndarray | None = None,
        **state: Any,
    ) -> None:
        config: TableConfig = state["config"]
        self.elo_edges = np.asarray(sorted(config.elo_edges), dtype=np.int64)
        shape = (self.elo_edges.size + 1, N_CLASS)
        self.elo_observed = (
            np.zeros(shape) if elo_observed is None else np.asarray(elo_observed)
        ).astype(np.float64)
        self.elo_expected = (
            np.zeros(shape) if elo_expected is None else np.asarray(elo_expected)
        ).astype(np.float64)
        if self.elo_observed.shape != shape or self.elo_expected.shape != shape:
            raise ValueError("Elo statistics do not match the band edges")
        super().__init__(**state)

    def _derive(self) -> None:
        super()._derive()
        self._elo_ratio = _ratio(
            self.elo_observed, self.elo_expected, self.config.elo_strength
        )

    def _band(self, rows: _Rows) -> np.ndarray:
        return np.searchsorted(self.elo_edges, rows.elo, side="right")

    def _adjust(self, action: np.ndarray, rows: _Rows) -> np.ndarray:
        factor = self._elo_ratio[self._band(rows)[:, None], rows.action_class]
        factor = np.where(rows.elo_known[:, None], factor, 1.0)
        return _normalized(action * factor, action)

    def _fit_extra(
        self,
        rows: _Rows,
        arrays: Mapping[str, np.ndarray],
        share: np.ndarray,
        weight: np.ndarray,
    ) -> None:
        """Observed and expected class weight per band, under the flags model.

        Expected is the table as it was fitted (no ``other_boost``, no Elo
        ratio), so a band's ratio says how its players differ from all
        players and not how the fit rows differ from new ones.
        """
        used = (share.sum(-1) > 0) & rows.elo_known
        where = np.flatnonzero(used)
        expected = self._action(rows, fitting=True)[where]
        band = self._band(rows)[where]
        index = band[:, None] * N_CLASS + rows.action_class[where]
        size = (self.elo_edges.size + 1) * N_CLASS
        slot_weight = weight[where, None]
        self.elo_observed = _bincount(index, share[where] * slot_weight, size).reshape(
            -1, N_CLASS
        )
        self.elo_expected = _bincount(index, expected * slot_weight, size).reshape(
            -1, N_CLASS
        )
        self.fit_info["elo_slots"] = int(used.sum())
        self._derive()

    def _state(self) -> dict[str, Any]:
        state = super()._state()
        state["elo_observed"] = self.elo_observed
        state["elo_expected"] = self.elo_expected
        return state

    def with_config(self: TableT, **changes: Any) -> TableT:
        if "elo_edges" in changes and tuple(changes["elo_edges"]) != tuple(
            self.config.elo_edges
        ):
            raise ValueError("elo_edges cannot change without a refit")
        return super().with_config(**changes)

    def to_payload(self) -> dict[str, Any]:
        payload = super().to_payload()
        payload["elo_edges"] = self.elo_edges.copy()
        payload["elo_observed"] = self.elo_observed.copy()
        payload["elo_expected"] = self.elo_expected.copy()
        return payload

    @classmethod
    def _payload_state(
        cls, payload: Mapping[str, Any], featurizer: Any
    ) -> dict[str, Any]:
        state = super()._payload_state(payload, featurizer)
        state["elo_observed"] = np.asarray(payload["elo_observed"], dtype=np.float64)
        state["elo_expected"] = np.asarray(payload["elo_expected"], dtype=np.float64)
        return state


TABLES: dict[str, type[_CountTable]] = {
    TABLE_SPECIES: SpeciesTable,
    TABLE_FLAGS: FlagsTable,
    TABLE_ELO: EloTable,
}


def from_payload(payload: Mapping[str, Any], featurizer: Any = None) -> _CountTable:
    """The table a payload describes (``artifact.load_predictor`` calls this).

    Returns a ``SpeciesTable`` / ``FlagsTable`` / ``EloTable``: an object with
    ``predict(batch)``, ``name`` and ``kind``. Raises ``ValueError`` on a
    payload that is not one of theirs.
    """
    try:
        made = TABLES.get(str(payload["table"]))
    except (KeyError, TypeError) as exc:
        raise ValueError(f"not a table payload: {exc!r}") from exc
    if made is None:
        raise ValueError(f"unknown table {payload['table']!r}")
    return made.from_payload(payload, featurizer)
