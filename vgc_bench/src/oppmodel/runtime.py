"""Runtime of the opponent predictor: what the bot calls once per turn.

    predictor = OpponentPredictor.load("results_oppmodel/<run>/artifact.pt")
    predictor.note_sheets(battle, known_open=False)     # once, at team preview
    forecast = predictor.predict(battle)                # None when it stands down
    if forecast is not None and forecast.a is not None:
        forecast.a.p_protect, forecast.a.p_attacks[0], forecast.a.actions[0]
    predictor.forget(battle.battle_tag)                 # when the battle ends

``predict`` reads three attributes of the battle and nothing else:
``_replay_data`` (the raw split events poke-env keeps), ``player_role`` and
``battle_tag``. It never looks at the live battle's own fields, so no exact HP
and no request data can reach a feature. The state path is the one the
training data went through: a ``LiveShadow`` per (battle tag, player role) is
synced from the stream, its turn-start ``PublicSnapshot`` is encoded with the
artifact's ``Featurizer`` for the OPPONENT's side as the actor, "unknown" sheet
codes are mapped to "closed" (``features.sheet_unknown_as_closed``, as at
evaluation) and the artifact's predictor is called on that one example.

Orientation. A ``Forecast`` is about the opposing side. ``forecast.slots[0]`` /
``.a`` is the opponent's slot a, ``[1]`` / ``.b`` its slot b; an empty slot is
None. Targets keep the names of the labels: ``foe_a`` / ``foe_b`` are the slot
letters of the side the mover faces, which is OUR side, so ``foe_a`` is our
slot a (``battle.active_pokemon[0]``) whichever seat we have. ``p_attacks`` is
indexed the same way: ``p_attacks[0]`` is the probability that this opposing
slot attacks our slot a.

Probabilities. ``SlotForecast.action_probs`` / ``target_probs`` are the
predictor's output with illegal mass removed and every row renormalised
(``features.normalize_prediction``, what every score is computed on);
``Forecast.raw`` holds the predictor's output untouched. The scalars come from
those arrays through ``features.event_probs`` and ``features.intent_probs``,
the helpers the scorecard calibrates, so a threshold read off a reliability
table applies to the same number here. The two opposing slots are predicted
separately: no per-slot field of a forecast says how their choices go together.

The pair. EVERY per-slot field (lists, scalars, ``action_probs`` /
``target_probs``, ``raw``, ``p_attacked``, ``to_dict`` apart from the entry
named below) is the uncoupled marginal, whatever the artifact carries. An
artifact with a pair coupling (``coupling.PairCoupling``, the artifact's
``coupling`` key) adds, on a turn where both opposing slots act:
``Forecast.pair`` (the coupling's name, this turn's bucket, the 8 x 8 class
weights), ``Forecast.pair_class_weight(class_a, class_b)`` and
``Forecast.pair_weight(action_a, action_b)``: the number to multiply the
PRODUCT of two per-slot probabilities by; 1.0 without a coupling, with an
empty slot, or for anything that cannot be classed. ``Forecast.joint_top(k)``
is the ranked joint list itself (``joint.joint_replies`` on what the forecast
kept), computed only when asked: with the artifact's coupling when it has one,
the plain product otherwise. Never multiply an entry of ``joint_top`` by
``pair_weight``: that applies the coupling twice. ``to_dict`` gains one small
entry, ``pair`` (coupling name and bucket), only when ``Forecast.pair`` is
set; without a coupling it is the dict of before, key for key.

A pair method that cannot answer says so in a counter, not in its value: 1.0
is also the honest answer without a coupling, and ``()`` the answer of a
forecast without inputs. On a forecast that HAS a pair (a coupling, two acting
slots), an action or reply that cannot be classed counts
``pair_weight:unclassed`` / ``reply_weight:unread``, and a list that cannot be
built ``joint_top:not_built``, in this module's ``COUNTERS``; the joint's and
the coupling's own failures are in ``joint.COUNTERS`` / ``coupling.COUNTERS``.
One case of "cannot be classed" has its own name, ``pair_weight:other_mixed``:
a slot's ``other`` entry is the OTHER bucket (class ``unassigned``) plus every
candidate the runtime could not name, and ``joint_top`` classes such a
candidate by its own row of the intent table. With the tables a featurizer
builds that row is unassigned too and the two agree; when it is not, the
entry holds two classes, no one number fits it, and ``pair_weight`` /
``reply_weight`` answer 1.0 and count instead of answering for the wrong class.
A forecast has no runtime to count in, so these three are process-wide (every
runtime of the process adds to them): ``OpponentPredictor.diagnostics()``
reports them under ``process``, with the module they come from as a prefix
(``runtime:`` also holds the adapters' and the sheet helper's failures).

Standing down. ``predict`` never raises. It returns None and counts a name in
``counters``: ``stand_down:<why>`` for the expected cases (no turn yet, the
stream is past the turn line as at a forced-switch request, no role, the
opponent has nobody on the field, nothing loaded, and a battle of a kind the
dataset drops or whose stream lost something: see ``_unfit``), and a name
containing ``error`` only for a real failure
(``opp_predictor_error:<where>:<type>``, the shadow's ``sync_error:*``,
``state:feed_error:*``, ``load_error:<type>``). A predictor that failed inside
and fell back to its uniform prediction is noticed through its own
``predict_error:*`` counters and treated as a failure, never served. So is a
prediction that is not a probability table for an occupied slot: a negative
entry, a slot whose legal actions carry no mass, a move with mass whose legal
targets carry none (``opp_predictor_error:prediction_malformed``). A forecast
that is served always says something.

Version-2 inputs. A layout-version-2 featurizer never loses an example to its
matchup or set-prior block: when one fails it writes ZEROS and counts a name
(``matchup_error:*``, ``setprior_*``), which is right for building a dataset.
A network fitted on real values would read those zeros as facts (a speed tie,
no damage either way, a move no set holds), so the runtime does not serve
that call: when the featurizer's counters show, across the encode of THIS
call, that a block failed whose arrays the predictor reads
(``extras_read``: the network's ``extra_keys``), ``predict`` returns None with
``opp_predictor_error:extras_degraded`` and counts ``extras_degraded:<family>``
(``matchup`` / ``set_prior``). A predictor that reads none of a block's arrays
(a count table, a network fitted with ``--extras 0``, the live version-1
model) is served as before whatever that block did. The comparison is made
under the runtime's lock, like the one that notices a degraded predictor; a
featurizer shared with another runtime can only make it stand down once too
often, never serve a degraded call.

Cache, lock, bound. A second ``predict`` at the same turn start returns the
same ``Forecast`` object without recomputing (a failure is remembered for the
turn as well). One lock covers every public method, so battles in several
threads cannot disturb each other. At most ``max_battles`` shadows are kept,
least recently used first out; ``forget(battle_tag)`` drops one at once. What
``note_sheets`` was told is kept beside the shadows and given again to a shadow
that is rebuilt after an eviction, so a battle never changes from open to
closed sheets in the middle.

Team sheets. The stream never carries a ``|showteam|`` line, so the caller
says what it learned from poke-env through ``note_sheets``; until then every
sheet is "unknown" and is treated as closed. ``sheet_sets_from_team`` builds
the sets from a poke-env team dict. It is the only function here that touches
poke-env objects, and it is optional.

Existing consumers. ``to_move_prediction`` / ``to_switch_prediction`` turn a
slot's forecast into the ``MovePrediction`` / ``SwitchPrediction`` shapes that
``opponent_reranker``, ``tempo_reranker`` and the exact planner's opponent
prior read, following ``evaluation/oppmodel_oracle.certain_predictions``.
``MovePrediction.reliability`` keeps the meaning those consumers give it: how
much of the moveset is KNOWN (shown or on an open sheet, over four), not how
well the probabilities are calibrated; see ``to_move_prediction``. What those
shapes cannot carry, and is therefore lost on that route:

* a Protect: it is one more status move, read by no damage term, so the only
  effect is less predicted incoming damage; nothing says "our attack into this
  slot is wasted";
* an aimed move the simulator turns into a spread hit: its target class is
  ``field``, which the reranker reads for no slot;
* the OTHER bucket (a move outside the candidates) and the Mega probability;
* a switch into a Pokemon the bot has not seen: the species is named, but the
  reranker's bench lookup holds only seen Pokemon;
* any dependence between the two opposing slots.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from vgc_bench.src.oppmodel import coupling as _coupling
from vgc_bench.src.oppmodel import joint as _joint
from vgc_bench.src.oppmodel.artifact import load_predictor
from vgc_bench.src.oppmodel.coupling import (
    BUCKETS,
    C_SWITCH,
    C_UNASSIGNED,
    CLASSES,
    KEY_BUCKET,
    KEY_CLASS,
    class_index,
    reply_buckets,
    reply_classes,
)
from vgc_bench.src.oppmodel.events import (
    INTENT_CLASSES,
    INTENT_PROTECT,
    INTENT_SWITCH,
    KIND_MOVE,
    KIND_SWITCH,
    REASON_FLINCH,
    SIDES,
    SLOT_LETTERS,
    TARGET_ALLY,
    TARGET_AUTO,
    TARGET_CLASSES,
    TARGET_FOE_A,
    TARGET_FOE_B,
    TARGET_SELF,
    SheetSet,
    base_species_id,
    is_fake_out_like,
    move_entry,
    to_id,
)
from vgc_bench.src.oppmodel.features import (
    CAND_REVEALED,
    CAND_SHEET,
    CAND_VALID,
    ELO_BLANK,
    ELO_KEEP,
    EXTRA_ARRAYS,
    FAMILY_MATCHUP,
    FAMILY_SETPRIOR,
    FLAG_MEGA_POSSIBLE,
    G_ACTOR_SHEET,
    G_OTHER_SHEET,
    MAX_MOVES,
    N_INTENT,
    N_ROSTER,
    N_SLOT,
    N_TARGET,
    SHEET_CLOSED,
    SHEET_OPEN,
    SHEET_UNKNOWN,
    T_AUTO,
    T_FOE_A,
    T_FOE_B,
    Featurizer,
    collate,
    event_probs,
    expand_target_mask,
    intent_probs,
    normalize_prediction,
    sheet_unknown_as_closed,
)
from vgc_bench.src.oppmodel.joint import MEGA_NONE, UNKNOWN, joint_replies
from vgc_bench.src.oppmodel.public_state import LiveShadow, PublicSnapshot
from vgc_bench.src.opponent_tactics import MovePrediction, SwitchPrediction

ACTION_MOVE = KIND_MOVE
ACTION_SWITCH = KIND_SWITCH
ACTION_OTHER = "other"  # a move the candidates do not name

SHEET_STATE_CLOSED = "closed"
SHEET_STATE_OPEN = "open"
SHEET_STATE_UNKNOWN = "unknown"
_SHEET_NAMES = {
    SHEET_CLOSED: SHEET_STATE_CLOSED,
    SHEET_OPEN: SHEET_STATE_OPEN,
    SHEET_UNKNOWN: SHEET_STATE_UNKNOWN,
}

# The label reader's target classes under the names MovePredictor.TARGET_NAMES
# uses (the mapping of evaluation/oppmodel_oracle.py). ``foe_a`` means the same
# in both: slot a of the side the mover faces.
LEGACY_TARGET_NAMES: dict[str, str] = {
    TARGET_FOE_A: "foe_a",
    TARGET_FOE_B: "foe_b",
    TARGET_ALLY: "ally",
    TARGET_SELF: "self",
    TARGET_AUTO: "field",
}

# A plain reply's target name (the label reader's, the legacy consumers', or
# None for a move nobody aims) as an index into ``TARGET_CLASSES``.
_PLAIN_TARGETS: dict[str | None, int] = {
    **{name: index for index, name in enumerate(TARGET_CLASSES)},
    **{
        legacy: TARGET_CLASSES.index(name)
        for name, legacy in LEGACY_TARGET_NAMES.items()
    },
    None: T_AUTO,
}

DEFAULT_TOP_K = 8
DEFAULT_MAX_BATTLES = 64
# Sheet notes are kept for this many times ``max_battles`` battles (they are
# small, and outlive an evicted shadow on purpose).
_NOTED_PER_BATTLE = 8

# Counter names. A stand-down is an expected "no forecast"; none of these names
# contains the word a harness scans for to find failures.
STAND_DOWN = "stand_down:"
FAILURE = "opp_predictor_error:"
_STATE_PREFIX = "state:"
_PREDICTOR_FAILURE = "predict_error"
_ENCODE_SKIP = "encode_skip:"
_ENCODE_FAILURE = "encode_error:"
# A version-2 block of the featurizer that failed for an example: it wrote
# zeros and counted a name starting with one of these (``Featurizer._matchup``
# / ``_set_prior``), by the family of ``features.EXTRA_ARRAYS`` it writes.
# ``matchup_no_stats`` is not among them: a Pokemon without a stat line is
# encoded the same way in every dataset, so the network has seen it.
_EXTRAS_FAILED: dict[str, tuple[str, ...]] = {
    FAMILY_MATCHUP: ("matchup_error:",),
    FAMILY_SETPRIOR: ("setprior_",),
}
# The stand-down of a call whose version-2 inputs were zeroed, and the prefix
# of the counter that names the family.
EXTRAS_DEGRADED = "extras_degraded"
_FEED_FAILURE = "feed_error"
_STREAM_SKIP = "sync_skip:"  # the shadow passed over an entry of the stream
# A served slot's legal actions, and the legal targets of each move with mass,
# must sum to 1 within this; a raw entry may be negative by no more than it.
_MASS_TOLERANCE = 1e-6
# The key of the dataset's format ids in an artifact's ``extra``.
_FORMATS_KEY = "formats"
# The dex's id of the effect a flinching move leaves; the label reader uses the
# same id as the reason of a slot that could not act.
_FLINCH = REASON_FLINCH
# poke-env's placeholder for an item nobody has seen, as an id.
_UNSEEN_ITEM = "unknownitem"

# Failures of the module-level helpers (adapters, sheet helper) and of a
# forecast's pair methods, by name. Process-wide: a forecast has no runtime.
COUNTERS: Counter[str] = Counter()
# A pair method asked something it could not answer on a forecast that has a
# pair (see the module text, "The pair").
PAIR_UNCLASSED = "pair_weight:unclassed"
REPLY_UNREAD = "reply_weight:unread"
JOINT_NOT_BUILT = "joint_top:not_built"
# A slot's ``other`` entry that holds two classes (see the module text): it is
# counted here and, by the method that asked, as unclassed / unread.
PAIR_OTHER_MIXED = "pair_weight:other_mixed"

_LOG = logging.getLogger(__name__)


def _failed(name: str, exc: Exception) -> None:
    try:
        COUNTERS[f"{name}:{type(exc).__name__}"] += 1
    except Exception:
        pass


def _count(name: str) -> None:
    try:
        COUNTERS[name] += 1
    except Exception:
        pass


def _warn(message: str, *values: Any) -> None:
    """One WARNING on this module's logger. Never raises."""
    try:
        _LOG.warning(message, *values)
    except Exception:
        pass


def process_counters() -> dict[str, int]:
    """The process-wide counters no runtime owns: this module's ``COUNTERS``
    (``runtime:``: the adapters, the sheet helper, a forecast's pair
    methods), ``joint.COUNTERS`` (``joint:``) and ``coupling.COUNTERS``
    (``coupling:``). Never raises."""
    out: dict[str, int] = {}
    for prefix, counters in (
        ("runtime:", COUNTERS),
        ("joint:", _joint.COUNTERS),
        ("coupling:", _coupling.COUNTERS),
    ):
        try:
            for name, count in dict(counters).items():
                out[f"{prefix}{name}"] = int(count)
        except Exception:
            continue
    return out


def is_first_turn_flinch(move_id: Any) -> bool:
    """A damaging move that works only on the user's first turn and always
    makes its target flinch, by the dex (never by name). Never raises; False
    for anything that is not a move id."""
    if not isinstance(move_id, str):
        return False
    try:
        return _first_turn_flinch(move_id)
    except Exception:
        return False


@lru_cache(maxsize=4096)
def _first_turn_flinch(move_id: str) -> bool:
    try:
        if not is_fake_out_like(move_id):
            return False
        entry = move_entry(move_id) or {}
        effects = [entry.get("secondary"), *(entry.get("secondaries") or ())]
        for effect in effects:
            if not isinstance(effect, dict):
                continue
            if to_id(str(effect.get("volatileStatus") or "")) != _FLINCH:
                continue
            chance = effect.get("chance")
            if chance is None or float(chance) >= 100.0:
                return True
        return False
    except Exception:
        return False


# --- the forecast ---------------------------------------------------------------


@dataclass(frozen=True)
class ActionForecast:
    """One entry of a slot's ranked list.

    ``kind`` is ``move`` (``move`` id and ``target`` class: ``foe_a`` /
    ``foe_b`` = OUR slot a / b, ``ally``, ``self``, ``auto`` for a move the
    player does not aim), ``switch`` (``switch_to`` base species and its
    ``roster_index`` in the opponent's team-preview order) or ``other`` (a move
    outside the candidates: no id, no target).
    """

    kind: str
    probability: float
    move: str | None = None
    target: str | None = None
    switch_to: str | None = None
    roster_index: int | None = None

    @property
    def our_slot(self) -> int | None:
        """0 / 1 when a move is aimed at our slot a / b, else None."""
        if self.target == TARGET_FOE_A:
            return 0
        if self.target == TARGET_FOE_B:
            return 1
        return None

    def to_dict(self, digits: int = 4) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind, "p": round(self.probability, digits)}
        if self.move is not None:
            out["move"] = self.move
        if self.target is not None:
            out["target"] = self.target
        if self.switch_to is not None:
            out["switch_to"] = self.switch_to
        if self.roster_index is not None:
            out["roster_index"] = self.roster_index
        return out


@dataclass(frozen=True, eq=False)
class SlotForecast:
    """What one opposing slot is predicted to do this turn.

    ``actions`` is the ranked list of fine actions (move x target, switch
    destination, OTHER), most probable first, cut to the runtime's ``top_k``.
    Scalars, all probabilities of this slot's choice:

    * ``p_switch`` a voluntary switch; ``p_protect`` a Protect-family move;
    * ``p_fake_out`` a first-turn-only flinching move (``is_first_turn_flinch``)
      and ``p_fake_out_at[i]`` that move aimed at our slot ``i``;
    * ``p_attacks[i]`` a damaging move aimed at our slot ``i`` (0 = a, 1 = b)
      or spread over it; 0 where we have nobody. How it is split between our
      two slots is only as good as the model's target head. Measured on the
      bot's own games (2026-10-04, aimed attacks with both our slots filled):
      the larger of the two is the slot really attacked 50% of the time for a
      count table whose targets are keyed by set key and move alone (a coin),
      59-61% for the flags table with its target context, 65-66% for the
      network. A guard that needs "which of my slots" must check what model
      is behind the forecast (``Forecast.model_kind`` / ``model_name``) and
      take its threshold from that model's two-target rows in the scorecard;
    * ``intents`` the seven ``INTENT_CLASSES``; ``p_unassigned`` what none of
      them takes (OTHER, and moves whose intent the dex cannot decide);
    * ``p_other`` the OTHER bucket alone;
    * ``p_mega`` Mega Evolution this turn, 0 where the public state rules it
      out (the predictor's own value stays in ``Forecast.raw``).

    None of these scalars is calibrated beyond what the scorecard's
    reliability tables show for the model behind them: read a threshold off
    those tables, not off the number's face value.

    ``support`` is the weighted number of training uses behind the acting
    Pokemon's set key (``support_log`` is the feature the model saw, its
    log1p); near zero means the candidates are a guess from the overall
    ranking. ``known_moves`` counts the candidates that are known to be in the
    set (shown in battle or on an open sheet, at most four): the rest are
    guesses from the repertoire. ``moves`` names the candidate columns ('' =
    empty or not nameable) and ``roster`` the six switch pointers; with them
    ``action_probs`` ``[n_cand + 7]`` and ``target_probs`` ``[n_cand + 1, 5]``
    (read-only) are the whole distribution.
    """

    slot: str
    species: str
    forme: str
    roster_index: int
    actions: tuple[ActionForecast, ...]
    p_switch: float
    p_protect: float
    p_fake_out: float
    p_fake_out_at: tuple[float, float]
    p_attacks: tuple[float, float]
    intents: dict[str, float]
    p_unassigned: float
    p_other: float
    p_mega: float
    support: float
    support_log: float
    moves: tuple[str, ...]
    roster: tuple[str, ...]
    action_probs: np.ndarray
    target_probs: np.ndarray
    known_moves: int = 0

    @property
    def top(self) -> ActionForecast | None:
        return self.actions[0] if self.actions else None

    def move_probabilities(self) -> tuple[tuple[str, float], ...]:
        """(move id, probability) of every named candidate, most probable first."""
        pairs = [
            (name, float(self.action_probs[column]))
            for column, name in enumerate(self.moves)
            if name and float(self.action_probs[column]) > 0.0
        ]
        pairs.sort(key=lambda item: -item[1])
        return tuple(pairs)

    def to_dict(self, digits: int = 4) -> dict[str, Any]:
        """Plain data for a decision audit: scalars and the ranked list."""
        return {
            "slot": self.slot,
            "species": self.species,
            "p_switch": round(self.p_switch, digits),
            "p_protect": round(self.p_protect, digits),
            "p_fake_out": round(self.p_fake_out, digits),
            "p_attacks": [round(value, digits) for value in self.p_attacks],
            "p_mega": round(self.p_mega, digits),
            "intents": {
                name: round(value, digits) for name, value in self.intents.items()
            },
            "p_unassigned": round(self.p_unassigned, digits),
            "support": round(self.support, 2),
            "known_moves": int(self.known_moves),
            "actions": [action.to_dict(digits) for action in self.actions],
        }


@dataclass(frozen=True, eq=False)
class PairWeights:
    """How the two opposing slots' choices go together this turn.

    ``coupling`` is the name of the artifact's pair coupling, ``bucket`` this
    turn's context bucket (``coupling.BUCKETS``), ``classes`` the eight reply
    classes and ``weight`` ``[8, 8]`` (read-only) the number for (class of
    slot a's action, class of slot b's action): multiply the product of the
    two per-slot probabilities by it and renormalise over the pairs kept.
    """

    coupling: str
    bucket: str
    classes: tuple[str, ...]
    weight: np.ndarray


@dataclass(frozen=True)
class JointEntry:
    """One entry of ``Forecast.joint_top``: what both opposing slots do.

    ``probability`` is the joint reply's probability after renormalising;
    ``slots`` the two slots' actions (None for an empty slot; each carries its
    own per-slot marginal as ``probability``; a candidate without a name
    reads ``other``, as in the per-slot lists); ``replies`` the two reply
    indices of ``joint.py``; ``mega`` the slot that Mega-evolves (0 / 1) or
    None; ``coupled`` whether the list was built with the pair coupling.
    """

    probability: float
    slots: tuple[ActionForecast | None, ActionForecast | None]
    replies: tuple[int, int]
    mega: int | None = None
    coupled: bool = False


@dataclass(frozen=True, eq=False)
class Forecast:
    """The opponent's predicted actions at the start of ``turn``.

    ``slots`` are the opponent's slots (a, b); an empty slot is None.
    ``player_role`` is our seat, ``opponent_role`` the side the forecast is
    about. ``elo`` / ``own_elo`` are the ratings the model was given (the
    opponent's, ours), None when unknown or when the model is Elo-blind.
    ``sheets`` is the sheet state the model was given, (the opponent's, ours),
    ``open`` or ``closed``; ``sheets_reported`` is what the shadow knew before
    "unknown" was mapped to closed. ``latency_ms`` is the time the computing
    call took, sync and encoding included. ``raw`` is the predictor's own
    output for this example (``action`` ``[2, A]``, ``target``
    ``[2, n_cand + 1, 5]``, ``mega`` ``[2]``); ``features`` is the encoded
    batch of one example when the runtime was told to keep it, else None.

    ``pair`` is set only when the artifact carries a pair coupling and both
    opposing slots act (see the module text, "The pair"); ``joint_inputs``
    holds the few small arrays ``joint_top`` needs (masks, and with a coupling
    the reply classes and the bucket); ``coupling`` is the artifact's coupling
    object or None. None of the three changes a per-slot number.
    """

    battle_tag: str
    turn: int
    player_role: str
    opponent_role: str
    slots: tuple[SlotForecast | None, SlotForecast | None]
    model_name: str
    model_kind: str
    elo: int | None
    own_elo: int | None
    sheets: tuple[str, str]
    sheets_reported: tuple[str, str]
    latency_ms: float
    raw: dict[str, np.ndarray] = field(default_factory=dict)
    features: dict[str, np.ndarray] | None = None
    pair: PairWeights | None = None
    joint_inputs: dict[str, np.ndarray] | None = None
    coupling: Any = None

    @property
    def a(self) -> SlotForecast | None:
        return self.slots[0]

    @property
    def b(self) -> SlotForecast | None:
        return self.slots[1]

    def slot(self, which: int | str) -> SlotForecast | None:
        """The opponent's slot by index (0 / 1) or letter; None when empty."""
        if which in SLOT_LETTERS:
            return self.slots[SLOT_LETTERS.index(str(which))]
        if which in (0, 1):
            return self.slots[int(which)]
        return None

    @property
    def p_attacked(self) -> tuple[float, float]:
        """P(our slot a / b is attacked by at least one opposing slot).

        The two opposing slots are combined as if independent: the model
        predicts each on its own.
        """
        spared = [1.0, 1.0]
        for made in self.slots:
            if made is None:
                continue
            for index in range(N_SLOT):
                spared[index] *= 1.0 - min(1.0, max(0.0, made.p_attacks[index]))
        return (1.0 - spared[0], 1.0 - spared[1])

    def to_dict(self, digits: int = 4) -> dict[str, Any]:
        """Plain, JSON-ready data for a decision audit (no arrays).

        Without a pair coupling this is the dict of before, key for key; with
        ``pair`` set it gains ``pair`` (the coupling's name and the bucket).
        """
        out = {
            "turn": self.turn,
            "model": self.model_name,
            "kind": self.model_kind,
            "elo": self.elo,
            "own_elo": self.own_elo,
            "sheets": list(self.sheets),
            "sheets_reported": list(self.sheets_reported),
            "latency_ms": round(self.latency_ms, 3),
            "slots": [
                None if made is None else made.to_dict(digits) for made in self.slots
            ],
        }
        if self.pair is not None:
            out["pair"] = {"coupling": self.pair.coupling, "bucket": self.pair.bucket}
        return out

    # --- the pair -------------------------------------------------------------

    def pair_class_weight(self, class_a: Any, class_b: Any) -> float:
        """The coupling's number for (class of slot a's action, class of slot
        b's action), each a name of ``coupling.CLASSES`` or its index.

        1.0 without a coupling, on a turn with an empty opposing slot, or for
        a class that does not exist. Never raises.
        """
        try:
            pair = self.pair
            first, second = class_index(class_a), class_index(class_b)
            if pair is None or first is None or second is None:
                return 1.0
            value = float(pair.weight[first, second])
            return value if math.isfinite(value) and value > 0.0 else 1.0
        except Exception as exc:
            _failed("pair_class_weight", exc)
            return 1.0

    def _other_class(self, which: int) -> int | None:
        """The class of slot ``which``'s ``other`` entry; None when it holds
        two (then ``pair_weight:other_mixed`` is counted).

        The entry is the OTHER bucket (``unassigned``) plus every candidate
        without a name. ``joint_top`` classes such a candidate by its own row
        of the intent table. Where that row is ``unassigned`` as well (every
        table a featurizer builds: the row of "no move"), or the candidate
        carries no mass, the entry is one class and ``pair_weight`` agrees
        with ``joint_top``. Otherwise no one class fits it.
        """
        inputs, made = self.joint_inputs, self.slots[which]
        if made is None or inputs is None or KEY_CLASS not in inputs:
            return None
        classes = np.asarray(inputs[KEY_CLASS])[0, which]
        for column, name in enumerate(made.moves):
            if name or float(made.action_probs[column]) <= 0.0:
                continue
            kinds = classes[column * N_TARGET : (column + 1) * N_TARGET]
            aimed = np.asarray(made.target_probs[column]) > 0.0
            if aimed.any():
                kinds = kinds[aimed]
            if (kinds != C_UNASSIGNED).any():
                _count(PAIR_OTHER_MIXED)
                return None
        return C_UNASSIGNED

    def action_class(self, which: int, action: Any) -> int | None:
        """The reply class (index into ``coupling.CLASSES``) of an entry of
        slot ``which``'s list; None without a coupling or when the entry
        cannot be classed (an ``other`` entry that holds two classes
        included: see ``_other_class``). Never raises."""
        try:
            inputs = self.joint_inputs
            if which not in (0, 1) or inputs is None or KEY_CLASS not in inputs:
                return None
            made = self.slots[which]
            if made is None or not isinstance(action, ActionForecast):
                return None
            if action.kind == ACTION_SWITCH:
                return C_SWITCH
            if action.kind == ACTION_OTHER:
                return self._other_class(which)
            if action.kind != ACTION_MOVE or not action.move:
                return None
            if action.move not in made.moves or action.target not in TARGET_CLASSES:
                return None
            column = made.moves.index(action.move)
            place = column * N_TARGET + TARGET_CLASSES.index(action.target)
            return int(np.asarray(inputs[KEY_CLASS])[0, which, place])
        except Exception as exc:
            _failed("action_class", exc)
            return None

    def pair_weight(self, action_a: Any, action_b: Any) -> float:
        """The coupling's number for slot a doing ``action_a`` together with
        slot b doing ``action_b`` (two ``ActionForecast`` of this forecast).

        MULTIPLY A PRODUCT OF TWO PER-SLOT PROBABILITIES BY IT, then
        renormalise over the pairs kept; never multiply an entry of
        ``joint_top``, which already holds it. 1.0 without a coupling, with
        an empty slot, or for an entry that cannot be classed. Never raises.

        The last case is a failure, not an answer: on a forecast that has a
        pair it counts ``pair_weight:unclassed`` in this module's
        ``COUNTERS`` (``OpponentPredictor.diagnostics()['process']``), so a
        1.0 that hides a broken forecast can be told from the 1.0 of no
        coupling. An ``other`` entry that holds two classes is such a case
        and also counts ``pair_weight:other_mixed``. For every other entry
        of ``joint_top`` (without the Mega bit) the product of its two
        slots' probabilities times this number is the entry's probability up
        to the list's one normalising constant.
        """
        if self.pair is None:
            return 1.0  # no coupling, or an empty slot: nothing to weigh
        first, second = self.action_class(0, action_a), self.action_class(1, action_b)
        if first is None or second is None:
            _count(PAIR_UNCLASSED)
            return 1.0
        return self.pair_class_weight(first, second)

    def reply_weight(self, first: Any, second: Any) -> float:
        """``pair_weight`` for a consumer that holds plain replies, not entries
        of this forecast (a search ranking legal choice strings).

        ``first`` / ``second`` describe what the opponent's slot a / slot b
        does: None (no action, a pass), ``("switch",)``, or ``("move",
        move_id, target)`` with ``target`` one of ``foe_a`` / ``foe_b`` /
        ``ally`` / ``self`` (``foe_a`` = OUR slot a) or None / ``auto`` /
        ``field`` for a move the player does not aim. A move this forecast
        does not name is the OTHER bucket's move (class ``unassigned``).
        Returns the coupling's number for the pair; 1.0 without a coupling,
        with an empty slot, for a pass, or for anything that cannot be read.
        Multiply a product of per-slot likelihoods by it. Never raises.

        A reply that cannot be read (not a pass) on a forecast that has a
        pair counts ``reply_weight:unread`` in this module's ``COUNTERS``;
        ``("other",)`` is this forecast's ``other`` entry and is unread when
        that entry holds two classes (``pair_weight:other_mixed``).
        """
        if self.pair is None or first is None or second is None:
            return 1.0  # no coupling, an empty slot, or a pass
        try:
            one, two = self._plain_class(0, first), self._plain_class(1, second)
            if one is None or two is None:
                _count(REPLY_UNREAD)
                return 1.0
            return self.pair_class_weight(one, two)
        except Exception as exc:
            _failed("reply_weight", exc)
            return 1.0

    def _plain_class(self, which: int, reply: Any) -> int | None:
        inputs, made = self.joint_inputs, self.slots[which]
        if made is None or inputs is None or KEY_CLASS not in inputs:
            return None
        if reply is None:
            return None
        parts = (reply,) if isinstance(reply, str) else tuple(reply)
        if not parts:
            return None
        if parts[0] == ACTION_SWITCH:
            return C_SWITCH
        if parts[0] == ACTION_OTHER:
            # This forecast's ``other`` entry, as ``action_class`` reads it.
            return self._other_class(which)
        if parts[0] != ACTION_MOVE or len(parts) < 2:
            return None
        move = str(parts[1] or "")
        if not move or move not in made.moves:
            # A move outside the candidates: the OTHER bucket's own move.
            return C_UNASSIGNED
        aim = _PLAIN_TARGETS.get(parts[2] if len(parts) > 2 else None)
        if aim is None:
            return None
        place = made.moves.index(move) * N_TARGET + aim
        return int(np.asarray(inputs[KEY_CLASS])[0, which, place])

    def _reply_action(self, which: int, reply: int) -> ActionForecast | None:
        made = self.slots[which]
        if made is None or reply < 0:
            return None
        n_cand = len(made.moves)
        other = n_cand * N_TARGET
        if reply < other:
            column, aim = divmod(reply, N_TARGET)
            mass = float(made.action_probs[column]) * float(
                made.target_probs[column, aim]
            )
            name = made.moves[column]
            if not name:
                return ActionForecast(ACTION_OTHER, mass)
            return ActionForecast(ACTION_MOVE, mass, name, TARGET_CLASSES[aim])
        if reply == other:
            return ActionForecast(ACTION_OTHER, float(made.action_probs[n_cand]))
        pointer = reply - other - 1
        if pointer >= N_ROSTER:
            return None
        return ActionForecast(
            ACTION_SWITCH,
            float(made.action_probs[n_cand + 1 + pointer]),
            switch_to=made.roster[pointer] or None,
            roster_index=pointer,
        )

    def joint_top(
        self, k: int = DEFAULT_TOP_K, *, mega: bool = False, coupled: bool = True
    ) -> tuple[JointEntry, ...]:
        """The ``k`` most probable joint replies of the two opposing slots.

        Computed when asked (``joint.joint_replies`` on ``raw`` and
        ``joint_inputs``), most probable first. ``coupled=True`` uses the
        artifact's pair coupling when it has one and the plain product
        otherwise; ``coupled=False`` is always the plain product. Each entry
        says which (``JointEntry.coupled``). ``mega=True`` also names who
        Mega-evolves. Never raises: an empty tuple when the list cannot be
        built (a coupling that cannot be applied gives an empty tuple, never
        the plain list). An empty tuple from a forecast that holds its inputs
        counts ``joint_top:not_built`` in this module's ``COUNTERS``; the
        reason is in ``joint.COUNTERS`` (both under
        ``OpponentPredictor.diagnostics()['process']``).
        """
        try:
            inputs = self.joint_inputs
            if inputs is None or not self.raw:
                return ()
            use = self.coupling if coupled else None
            made = joint_replies(self.raw, inputs, k, mega=mega, coupling=use)
            if made.n != 1:
                _count(JOINT_NOT_BUILT)
                return ()
            out: list[JointEntry] = []
            for position in range(made.k):
                first, second = (int(value) for value in made.reply[0, position])
                if UNKNOWN in (first, second):
                    break
                state = int(made.mega[0, position])
                out.append(
                    JointEntry(
                        probability=float(made.prob[0, position]),
                        slots=(
                            self._reply_action(0, first),
                            self._reply_action(1, second),
                        ),
                        replies=(first, second),
                        mega=None if state == MEGA_NONE else state - 1,
                        coupled=bool(made.coupled),
                    )
                )
            return tuple(out)
        except Exception as exc:
            _failed("joint_top", exc)
            _count(JOINT_NOT_BUILT)
            return ()


# --- sheets from poke-env (optional, the only poke-env contact) -----------------


def sheet_sets_from_team(team: Any, sort_moves: bool = False) -> list[SheetSet]:
    """An open team sheet from a poke-env team (``battle.opponent_team``,
    ``battle.team``): a mapping or an iterable of Pokemon objects.

    Reads ``species``, ``item``, ``ability`` and the keys of ``moves`` of each
    Pokemon, duck-typed. Moves keep their order, which right after team preview
    is the sheet's own (what the training logs show); ``sort_moves`` gives the
    alphabetical order the decision audit stores. Call it at team preview: a
    Pokemon that has not been given a sheet yet carries no moves and is left
    out. Never raises; an unreadable team gives an empty list.
    """
    try:
        members = team.values() if isinstance(team, Mapping) else team
        sets: list[SheetSet] = []
        for mon in members or ():
            species = str(getattr(mon, "species", "") or "")
            names = getattr(mon, "moves", None) or ()
            moves = [to_id(str(name)) for name in names]
            moves = [name for name in moves if name]
            if not species or not moves:
                continue
            item = to_id(str(getattr(mon, "item", "") or ""))
            ability = to_id(str(getattr(mon, "ability", "") or ""))
            sets.append(
                SheetSet(
                    species=base_species_id(species),
                    forme=to_id(species),
                    item=item if item and item != _UNSEEN_ITEM else None,
                    ability=ability or None,
                    moves=tuple(sorted(moves) if sort_moves else moves),
                )
            )
        return sets
    except Exception as exc:
        _failed("sheet_sets_from_team", exc)
        return []


def _sheet_list(sets: Any) -> list[SheetSet]:
    if sets is None:
        return []
    return [entry for entry in sets if isinstance(entry, SheetSet)]


# --- the runtime ----------------------------------------------------------------


@dataclass
class _Entry:
    """One battle seen from one seat: its shadow and this turn's result."""

    shadow: LiveShadow
    snapshot: PublicSnapshot | None = None
    forecast: Forecast | None = None
    reason: str = ""
    seen: Counter[str] = field(default_factory=Counter)
    public: Any = None
    public_seen: Counter[str] = field(default_factory=Counter)
    # Turn-line bookkeeping of the current ``public``: segments already looked
    # at, the last turn number among them, and whether a number was skipped.
    segments_checked: int = 0
    last_turn: int = 0
    turn_gap: bool = False


def _whole(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _frozen(values: Any, dtype: Any = None) -> np.ndarray:
    out = np.array(values, dtype=dtype)
    out.setflags(write=False)
    return out


class OpponentPredictor:
    """Forecast the opponent's actions from a live battle's event stream.

    Build one with ``OpponentPredictor.load(path)`` (an artifact written by
    ``oppmodel.artifact``) or directly from a ``features.Predictor`` and its
    ``Featurizer``. See the module docstring for what ``predict`` reads, when
    it stands down and how ``counters`` are named.

    ``top_k`` cuts each slot's ranked list (None keeps every entry),
    ``max_battles`` bounds the kept shadows, ``keep_features`` attaches the
    encoded example to every forecast (for audits and the parity check).
    ``allow_dex_difference``: by default a model whose artifact was written
    with a different dex (``meta['dex_signature_diff']``) is not served, since
    its candidate flags would differ from training; True serves it and counts
    ``dex_signature_differs``. ``meta['extra']['formats']`` (the format ids of
    the dataset the model was fitted on, as the fit scripts store them) makes
    the runtime stand down in a battle of any other format; an artifact that
    names none is served in every doubles format. ``coupling`` is the
    artifact's pair coupling (``load`` passes it on); it adds
    ``Forecast.pair`` and changes no per-slot number. A predictor that reads
    version-2 arrays (``extras_read``) is not served on a call whose
    featurizer block failed (``opp_predictor_error:extras_degraded``; the
    module text, "Version-2 inputs").
    """

    def __init__(
        self,
        predictor: Any,
        featurizer: Featurizer | None,
        *,
        name: str = "",
        kind: str = "",
        top_k: int | None = DEFAULT_TOP_K,
        max_battles: int = DEFAULT_MAX_BATTLES,
        keep_features: bool = False,
        allow_dex_difference: bool = False,
        meta: Mapping[str, Any] | None = None,
        coupling: Any = None,
    ) -> None:
        self.counters: Counter[str] = Counter()
        self._lock = threading.RLock()
        self._predictor = predictor
        self._featurizer = featurizer
        # The artifact's pair coupling (None for an artifact without one):
        # read by Forecast.pair / joint_top only, never by a per-slot number.
        self._coupling = coupling
        self.name = str(name or getattr(predictor, "name", "") or "")
        self.kind = str(kind or getattr(predictor, "kind", "") or "")
        self.top_k = None if top_k is None else _whole(top_k, DEFAULT_TOP_K)
        self.max_battles = max(1, _whole(max_battles, DEFAULT_MAX_BATTLES))
        self.keep_features = bool(keep_features)
        self.meta: dict[str, Any] = dict(meta or {})
        self.load_failure: str | None = None
        self._refusal: str | None = None
        self._entries: OrderedDict[tuple[str, str], _Entry] = OrderedDict()
        # What note_sheets was told, per (battle tag, role): given again to a
        # shadow that is rebuilt after its entry was evicted.
        self._noted: OrderedDict[tuple[str, str], tuple[bool, Any, Any]] = OrderedDict()
        self._formats = _format_ids(self.meta)
        # The version-2 arrays the predictor reads, and the families of
        # featurizer blocks they come from: a call whose block failed is not
        # served (see the module text, "Version-2 inputs").
        self._extras_read = extras_read(predictor)
        self._extra_families = _families_of(self._extras_read)
        blind = featurizer is not None and featurizer.elo_mode == ELO_BLANK
        blind = blind or getattr(predictor, "elo_mode", ELO_KEEP) == ELO_BLANK
        self._elo_blind = bool(blind)
        difference = list(getattr(featurizer, "signature_diff", None) or ())
        difference = difference or list(self.meta.get("dex_signature_diff") or ())
        if difference:
            self._bump("dex_signature_differs")
            if not allow_dex_difference:
                self._refusal = STAND_DOWN + "dex_signature"
                self.load_failure = f"dex signature differs: {sorted(difference)}"

    # --- construction ---------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: Path | str,
        elo_mode: str = ELO_KEEP,
        *,
        top_k: int | None = DEFAULT_TOP_K,
        max_battles: int = DEFAULT_MAX_BATTLES,
        keep_features: bool = False,
        allow_dex_difference: bool = False,
        strict: bool = False,
    ) -> "OpponentPredictor":
        """Read an artifact. By default this never raises.

        When the artifact cannot be read the result is a runtime that stands
        down on every call (``loaded`` is False, ``load_failure`` says why,
        ``load_error:<type>`` is counted once, and one WARNING goes to this
        module's logger: an artifact newer than the code, for instance, is
        otherwise visible only in counters and the audit). ``strict=True`` is for a
        launcher that would rather refuse to start: the loader's ``OSError`` /
        ``ValueError`` come through, also for a differing dex signature.
        ``elo_mode='blank'`` encodes every rating as unknown.
        """
        try:
            loaded = load_predictor(path, elo_mode, strict)
        except Exception as exc:
            if strict:
                raise
            made = cls(None, None, top_k=top_k, max_battles=max_battles)
            made.load_failure = f"{type(exc).__name__}: {exc}"
            made._bump(f"load_error:{type(exc).__name__}")
            _warn(
                "opponent predictor NOT loaded from %s: %s (every forecast "
                "stands down; pass strict=True to refuse to start instead)",
                path,
                made.load_failure,
            )
            return made
        return cls(
            loaded.predictor,
            loaded.featurizer,
            name=loaded.name,
            kind=loaded.kind,
            top_k=top_k,
            max_battles=max_battles,
            keep_features=keep_features,
            allow_dex_difference=allow_dex_difference,
            meta=loaded.meta,
            coupling=loaded.coupling,
        )

    @property
    def loaded(self) -> bool:
        """Whether there is a model to serve."""
        return self._predictor is not None and self._featurizer is not None

    @property
    def serving(self) -> bool:
        """Loaded, and not refused for a differing dex."""
        return self.loaded and self._refusal is None

    @property
    def predictor(self) -> Any:
        return self._predictor

    @property
    def featurizer(self) -> Featurizer | None:
        return self._featurizer

    @property
    def coupling(self) -> Any:
        """The artifact's pair coupling, or None."""
        return self._coupling

    @property
    def extras_read(self) -> tuple[str, ...]:
        """The version-2 arrays the predictor reads (``extras_read``); a call
        whose featurizer block failed for one of them is not served."""
        return self._extras_read

    @property
    def elo_mode(self) -> str:
        """``blank`` when the model is given no rating, else ``keep``."""
        return ELO_BLANK if self._elo_blind else ELO_KEEP

    @property
    def n_battles(self) -> int:
        with self._lock:
            return len(self._entries)

    def diagnostics(self) -> dict[str, dict[str, int]]:
        """Every counter in play: the runtime's, the featurizer's, the
        predictor's, and under ``process`` what no runtime owns
        (``process_counters``): a forecast's pair methods (``pair_weight``,
        ``reply_weight``, ``joint_top``), the joint and coupling code under
        them, the adapters and the sheet helper.

        The first three belong to this runtime. ``process`` is PROCESS-WIDE (a
        forecast has no runtime to count in): with two runtimes in one
        process each reports the same numbers there. Empty when nothing
        failed.
        """
        try:
            with self._lock:
                own = dict(self.counters)
                encoder = dict(getattr(self._featurizer, "counters", None) or {})
                model = dict(getattr(self._predictor, "counters", None) or {})
            return {
                "runtime": own,
                "featurizer": encoder,
                "predictor": model,
                "process": process_counters(),
            }
        except Exception:
            return {"runtime": {}, "featurizer": {}, "predictor": {}, "process": {}}

    # --- bookkeeping ----------------------------------------------------------

    def _bump(self, name: str, count: int = 1) -> str:
        if not name:
            return name
        try:
            self.counters[name] += count
        except Exception:
            pass
        return name

    def _entry(self, tag: str, role: str) -> _Entry:
        key = (tag, role)
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
            return entry
        entry = _Entry(LiveShadow())
        self._entries[key] = entry
        self._bump("battles_tracked")
        noted = self._noted.get(key)
        if noted is not None:
            # The entry was evicted and the battle is back: the fresh shadow
            # reads the whole stream again, under the sheets it was told.
            self._apply_sheets(entry, role, *noted)
            self._bump("sheets_restored")
        while len(self._entries) > self.max_battles:
            _, old = self._entries.popitem(last=False)
            self._fold(old)
            self._bump("evicted")
        return entry

    def _apply_sheets(
        self, entry: _Entry, role: str, known_open: bool, theirs: Any, ours: Any
    ) -> str:
        """Give a shadow its sheet knowledge; returns the counter name."""
        if not known_open:
            entry.shadow.mark_sheets_known()
            return "sheets:closed"
        other = SIDES[1 - SIDES.index(role)]
        if theirs:
            entry.shadow.feed_sheet(other, theirs)
        if ours:
            entry.shadow.feed_sheet(role, ours)
        if theirs and ours:
            entry.shadow.mark_sheets_known()
            return "sheets:open"
        return "sheets:open_partial"

    def _fold(self, entry: _Entry) -> str:
        """Move the shadow's new counts into ``counters``; name why it stood down."""
        reason = ""
        shadow = entry.shadow
        fresh = shadow.counters - entry.seen
        if fresh:
            entry.seen = Counter(shadow.counters)
            for name, count in fresh.items():
                self._bump(name, count)
            stands = [name for name in fresh if name.startswith(STAND_DOWN)]
            faults = [name for name in fresh if name.startswith("sync_error")]
            reason = (stands or faults or [""])[0]
        public = shadow.public
        if public is not entry.public:
            entry.public, entry.public_seen = public, Counter()
            entry.segments_checked, entry.last_turn, entry.turn_gap = 0, 0, False
        fresh = public.counters - entry.public_seen
        if fresh:
            entry.public_seen = Counter(public.counters)
            for name, count in fresh.items():
                self._bump(_STATE_PREFIX + name, count)
        return reason

    @staticmethod
    def _key(battle: Any) -> tuple[str | None, str | None]:
        tag = getattr(battle, "battle_tag", None)
        role = getattr(battle, "player_role", None)
        return (
            tag if isinstance(tag, str) and tag else None,
            role if isinstance(role, str) and role in SIDES else None,
        )

    # --- sheets ---------------------------------------------------------------

    def note_sheets(
        self,
        battle: Any,
        known_open: bool,
        opponent_sets: Iterable[SheetSet] | None = None,
        own_sets: Iterable[SheetSet] | None = None,
    ) -> bool:
        """Tell the battle's shadow what is known about team sheets.

        ``known_open=False``: the game is known to be closed-sheet; both sides
        then read "closed" as a fact instead of "unknown". ``known_open=True``:
        sheets were exchanged; ``opponent_sets`` / ``own_sets`` are the sheets
        (``sheet_sets_from_team`` builds them). A side whose sets are not
        given stays "unknown", and "every sheet is known" is recorded only
        when both were given. Call it before the first ``predict`` of the
        battle (at team preview): knowledge handed over later changes the
        turns from then on, which is not how any training game looked.
        Returns whether it was recorded. Never raises.
        """
        try:
            with self._lock:
                return self._note_sheets(battle, known_open, opponent_sets, own_sets)
        except Exception as exc:
            self._bump(f"{FAILURE}sheets:{type(exc).__name__}")
            return False

    def _note_sheets(
        self, battle: Any, known_open: bool, opponent_sets: Any, own_sets: Any
    ) -> bool:
        tag, role = self._key(battle)
        if tag is None:
            self._bump("sheets_ignored:no_battle_tag")
            return False
        if role is None:
            self._bump("sheets_ignored:role_unknown")
            return False
        theirs, ours = _sheet_list(opponent_sets), _sheet_list(own_sets)
        known_open = bool(known_open)
        if not known_open and (theirs or ours):
            self._bump("sheets_ignored:closed_with_sets")
            theirs, ours = [], []
        key = (tag, role)
        self._noted.pop(key, None)
        entry = self._entry(tag, role)
        self._noted[key] = (known_open, theirs, ours)
        while len(self._noted) > _NOTED_PER_BATTLE * self.max_battles:
            self._noted.popitem(last=False)
        self._bump(self._apply_sheets(entry, role, known_open, theirs, ours))
        self._fold(entry)
        return True

    def forget(self, battle_tag: Any) -> int:
        """Drop everything kept for a battle (both seats). Returns how many
        shadows went. Accepts the tag or the battle object. Never raises."""
        try:
            tag = battle_tag
            if not isinstance(tag, str):
                tag = getattr(battle_tag, "battle_tag", None)
            with self._lock:
                keys = [key for key in self._entries if key[0] == tag]
                for key in keys:
                    self._fold(self._entries.pop(key))
                for key in [key for key in self._noted if key[0] == tag]:
                    del self._noted[key]
                if keys:
                    self._bump("forgotten", len(keys))
                return len(keys)
        except Exception as exc:
            self._bump(f"{FAILURE}forget:{type(exc).__name__}")
            return 0

    # --- prediction -----------------------------------------------------------

    def predict(self, battle: Any) -> Forecast | None:
        """The forecast for the turn about to be played, or None. Never raises."""
        return self.predict_with_reason(battle)[0]

    def predict_with_reason(self, battle: Any) -> tuple[Forecast | None, str]:
        """``predict`` plus the counter name of a stand-down or failure ('' on
        success). Never raises."""
        try:
            with self._lock:
                return self._predict(battle)
        except Exception as exc:
            return None, self._bump(f"{FAILURE}predict:{type(exc).__name__}")

    def _predict(self, battle: Any) -> tuple[Forecast | None, str]:
        started = time.perf_counter()
        self._bump("predict_calls")
        if not self.loaded:
            return None, self._bump(STAND_DOWN + "not_loaded")
        if self._refusal is not None:
            return None, self._bump(self._refusal)
        tag, role = self._key(battle)
        if tag is None:
            return None, self._bump(STAND_DOWN + "no_battle_tag")
        if role is None:
            return None, self._bump(STAND_DOWN + "role_unknown")
        entry = self._entry(tag, role)
        snapshot = entry.shadow.sync(battle)
        reason = self._fold(entry)
        if snapshot is None:
            return None, reason or self._bump(STAND_DOWN + "no_snapshot")
        if entry.snapshot is snapshot:
            self._bump("cache_hit")
            return entry.forecast, entry.reason
        forecast = None
        reason = self._bump(_unfit(entry.shadow.public, self._formats))
        if not reason:
            reason = self._bump(_stream_damage(entry))
        if not reason:
            try:
                forecast, reason = self._compute(snapshot, tag, role, started)
            except Exception as exc:
                forecast = None
                reason = self._bump(f"{FAILURE}compute:{type(exc).__name__}")
        entry.snapshot, entry.forecast, entry.reason = snapshot, forecast, reason
        if forecast is not None:
            self._bump("forecasts")
        return forecast, reason

    def _call_predictor(
        self, batch: Mapping[str, np.ndarray]
    ) -> tuple[Mapping[str, Any] | None, str]:
        predictor = self._predictor
        before = _predictor_failures(predictor)
        try:
            made = predictor.predict(batch)
        except Exception as exc:
            return None, self._bump(f"{FAILURE}predictor:{type(exc).__name__}")
        if _predictor_failures(predictor) > before:
            # It caught its own failure and answered with the uniform fallback.
            return None, self._bump(FAILURE + "predictor_degraded")
        if not isinstance(made, Mapping) or any(
            name not in made for name in ("action", "target", "mega")
        ):
            return None, self._bump(FAILURE + "prediction_malformed")
        try:
            negative = any(
                bool(
                    (np.asarray(made[name], dtype=np.float64) < -_MASS_TOLERANCE).any()
                )
                for name in ("action", "target", "mega")
            )
        except (TypeError, ValueError):
            negative = True
        if negative:
            # A probability below zero: normalising would hide it as a zero.
            return None, self._bump(FAILURE + "prediction_malformed")
        return made, ""

    def _compute(
        self, snapshot: PublicSnapshot, tag: str, role: str, started: float
    ) -> tuple[Forecast | None, str]:
        featurizer = self._featurizer
        if featurizer is None:
            return None, self._bump(STAND_DOWN + "not_loaded")
        other = SIDES[1 - SIDES.index(role)]
        before = Counter(featurizer.counters)
        example = featurizer.encode(snapshot, other)
        if example is None:
            return None, self._bump(_encode_reason(featurizer.counters - before))
        if self._extra_families:
            # A version-2 block the predictor reads wrote zeros for this
            # example: no forecast from made-up inputs.
            failed = _failed_extras(featurizer.counters - before, self._extra_families)
            if failed:
                for family in failed:
                    self._bump(f"{EXTRAS_DEGRADED}:{family}")
                return None, self._bump(FAILURE + EXTRAS_DEGRADED)
        reported = collate([example])
        if not reported:
            return None, self._bump(FAILURE + "collate")
        batch = sheet_unknown_as_closed(reported)
        made, reason = self._call_predictor(batch)
        if made is None:
            return None, reason
        try:
            norm = normalize_prediction(made, batch)
        except (ValueError, TypeError, KeyError, IndexError):
            return None, self._bump(FAILURE + "prediction_malformed")
        if not np.isfinite(norm["mega"]).all():
            return None, self._bump(FAILURE + "prediction_malformed")
        if not _says_something(norm, example):
            return None, self._bump(FAILURE + "prediction_malformed")
        events = event_probs(norm, batch)
        intents = intent_probs(norm, batch, featurizer.tables)
        if not events or intents is None:
            return None, self._bump(FAILURE + "scalars")

        side = snapshot.sides[other]
        roster = [mon.species for mon in side.mons[:N_ROSTER]]
        roster += [""] * (N_ROSTER - len(roster))
        slots: list[SlotForecast | None] = [None, None]
        for index in range(N_SLOT):
            row = int(example["act_mon"][index])
            if row < 0 or row >= len(side.mons):
                continue
            slots[index] = self._slot_forecast(
                index, side.mons[row], roster, example, norm, events, intents
            )

        elo, known = np.asarray(batch["elo"])[0], np.asarray(batch["elo_known"])[0]
        ratings: list[int | None] = [None, None]
        if not self._elo_blind:
            for index in range(2):
                if int(known[index]):
                    ratings[index] = int(elo[index])
        told, used = (
            np.asarray(reported["game_flag"])[0],
            np.asarray(batch["game_flag"])[0],
        )
        columns = (G_ACTOR_SHEET, G_OTHER_SHEET)
        raw = {
            name: _frozen(np.asarray(made[name])[0])
            for name in ("action", "target", "mega")
        }
        kept: dict[str, np.ndarray] | None = None
        if self.keep_features:
            kept = {name: _frozen(value) for name, value in batch.items()}
        inputs, pair = self._pair_inputs(batch, example, slots)
        forecast = Forecast(
            battle_tag=tag,
            turn=int(snapshot.turn),
            player_role=role,
            opponent_role=other,
            slots=(slots[0], slots[1]),
            model_name=self.name,
            model_kind=self.kind,
            elo=ratings[0],
            own_elo=ratings[1],
            sheets=(_sheet_name(used[columns[0]]), _sheet_name(used[columns[1]])),
            sheets_reported=(
                _sheet_name(told[columns[0]]),
                _sheet_name(told[columns[1]]),
            ),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            raw=raw,
            features=kept,
            pair=pair,
            joint_inputs=inputs,
            coupling=self._coupling,
        )
        return forecast, ""

    def _pair_inputs(
        self,
        batch: Mapping[str, np.ndarray],
        example: Mapping[str, np.ndarray],
        slots: Sequence[SlotForecast | None],
    ) -> tuple[dict[str, np.ndarray] | None, PairWeights | None]:
        """What ``Forecast.joint_top`` and the pair weights read: a few small
        arrays, and with a coupling the turn's class weights. Never raises: a
        failure is counted and leaves the pair out, the per-slot forecast is
        served as it is."""
        try:
            rows = np.asarray(example["act_mon"]).astype(np.int64)
            flags = np.asarray(example["mon_flag"])[..., FLAG_MEGA_POSSIBLE]
            can = [
                bool(row >= 0 and int(flags[min(int(row), len(flags) - 1)]) > 0)
                for row in rows
            ]
            inputs = {
                "action_mask": _frozen(batch["action_mask"]),
                "cand_tmask": _frozen(batch["cand_tmask"]),
                "mega_possible": _frozen([can], bool),
            }
        except Exception as exc:
            self._bump(f"{FAILURE}joint_inputs:{type(exc).__name__}")
            return None, None
        coupling, featurizer = self._coupling, self._featurizer
        if coupling is None or featurizer is None:
            return inputs, None
        try:
            classes = reply_classes(batch, featurizer.tables.move_intent)
            bucket = reply_buckets(batch)
            inputs[KEY_CLASS] = _frozen(classes)
            inputs[KEY_BUCKET] = _frozen(bucket)
            if slots[0] is None or slots[1] is None:
                return inputs, None
            weight = coupling.weights(int(bucket[0]))
            if weight is None:
                self._bump(FAILURE + "pair:bucket")
                return inputs, None
            pair = PairWeights(
                coupling=str(getattr(coupling, "name", "") or ""),
                bucket=BUCKETS[int(bucket[0])],
                classes=CLASSES,
                weight=_frozen(weight, np.float64),
            )
            return inputs, pair
        except Exception as exc:
            self._bump(f"{FAILURE}pair:{type(exc).__name__}")
            inputs.pop(KEY_CLASS, None)
            inputs.pop(KEY_BUCKET, None)
            return inputs, None

    def _slot_forecast(
        self,
        index: int,
        mon: Any,
        roster: Sequence[str],
        example: Mapping[str, np.ndarray],
        norm: Mapping[str, np.ndarray],
        events: Mapping[str, np.ndarray],
        intents: np.ndarray,
    ) -> SlotForecast:
        featurizer = self._featurizer
        names_by_row = featurizer.tables.move_ids if featurizer is not None else ()
        action = np.asarray(norm["action"], dtype=np.float64)[0, index]
        target = np.asarray(norm["target"], dtype=np.float64)[0, index]
        n_cand = target.shape[0] - 1
        flags = np.asarray(example["cand_flag"])[index]
        rows = np.asarray(example["cand_move"])[index]
        moves: list[str] = []
        for column in range(n_cand):
            row = int(rows[column])
            valid = bool(int(flags[column]) & CAND_VALID)
            named = valid and 0 < row < len(names_by_row)
            moves.append(str(names_by_row[row]) if named else "")

        foe_up = np.asarray(example["foe_mon"]) >= 0
        ranked: list[ActionForecast] = []
        unnamed = float(action[n_cand])
        fake_out = 0.0
        fake_out_at = [0.0, 0.0]
        for column, name in enumerate(moves):
            mass = float(action[column])
            if mass <= 0.0:
                continue
            if not name:
                unnamed += mass
                continue
            split = 0.0
            for position, klass in enumerate(TARGET_CLASSES):
                share = mass * float(target[column, position])
                if share > 0.0:
                    split += share
                    ranked.append(ActionForecast(ACTION_MOVE, share, name, klass))
            if split <= 0.0:  # a candidate with no legal target class
                ranked.append(ActionForecast(ACTION_MOVE, mass, name, None))
            if is_first_turn_flinch(name):
                fake_out += mass
                spread = float(target[column, T_AUTO])
                for ours, position in enumerate((T_FOE_A, T_FOE_B)):
                    if bool(foe_up[ours]):
                        aimed = float(target[column, position]) + spread
                        fake_out_at[ours] += mass * aimed
        if unnamed > 0.0:
            ranked.append(ActionForecast(ACTION_OTHER, unnamed))
        for pointer in range(N_ROSTER):
            mass = float(action[n_cand + 1 + pointer])
            if mass > 0.0:
                ranked.append(
                    ActionForecast(
                        ACTION_SWITCH,
                        mass,
                        switch_to=roster[pointer] or None,
                        roster_index=pointer,
                    )
                )
        ranked.sort(key=lambda entry: -entry.probability)
        top_k = self.top_k
        if top_k is not None and top_k > 0:
            ranked = ranked[:top_k]

        known = sum(
            1
            for column in range(n_cand)
            if int(flags[column]) & CAND_VALID
            and int(flags[column]) & (CAND_REVEALED | CAND_SHEET)
        )
        row = int(example["act_mon"][index])
        can_mega = bool(int(np.asarray(example["mon_flag"])[row, FLAG_MEGA_POSSIBLE]))
        attack = np.asarray(events["attack"], dtype=np.float64)[0, index]
        per_intent = np.asarray(intents, dtype=np.float64)[0, index]
        support_log = float(np.asarray(example["slot_support"])[index])
        return SlotForecast(
            slot=SLOT_LETTERS[index],
            species=str(mon.species),
            forme=str(mon.forme),
            roster_index=row,
            actions=tuple(ranked),
            p_switch=float(np.asarray(events[INTENT_SWITCH])[0, index]),
            p_protect=float(np.asarray(events[INTENT_PROTECT])[0, index]),
            p_fake_out=fake_out,
            p_fake_out_at=(fake_out_at[0], fake_out_at[1]),
            p_attacks=(float(attack[0]), float(attack[1])),
            intents={
                name: float(per_intent[position])
                for position, name in enumerate(INTENT_CLASSES)
            },
            p_unassigned=float(per_intent[N_INTENT]),
            p_other=float(action[n_cand]),
            p_mega=float(np.asarray(norm["mega"])[0, index]) if can_mega else 0.0,
            support=max(0.0, math.expm1(support_log)),
            support_log=support_log,
            moves=tuple(moves),
            roster=tuple(roster),
            action_probs=_frozen(action, np.float64),
            target_probs=_frozen(target, np.float64),
            known_moves=min(MAX_MOVES, known),
        )


def _format_ids(meta: Mapping[str, Any]) -> frozenset[str]:
    """The format ids an artifact says its dataset was built from (may be none)."""
    try:
        extra = meta.get("extra") or {}
        found = extra.get(_FORMATS_KEY)
        if found is None and isinstance(extra.get("dataset"), Mapping):
            found = extra["dataset"].get(_FORMATS_KEY)
        if isinstance(found, str):
            found = [found]
        return frozenset(to_id(str(name)) for name in (found or ()) if name)
    except Exception:
        return frozenset()


def _unfit(public: Any, formats: Iterable[str] = ()) -> str:
    """Why this battle is one no training example was made from, or ''.

    The dataset drops a battle the tracker cannot vouch for: a Pokemon whose
    logged identity can be false, a game that is not doubles, a turn order
    that cannot be trusted (``PublicBattle.skip_reason``), any event the
    tracker failed on, and a battle of a format the dataset was not built
    from (``formats``: the artifact's own list; empty = any format, and a
    stream that never named its format is served). A forecast there would
    rest on a state that may be wrong, or on a model fitted to another game,
    so the runtime stands down for it as well.
    """
    try:
        why = public.skip_reason()
    except Exception as exc:
        return f"{FAILURE}skip_reason:{type(exc).__name__}"
    if why is not None:
        return STAND_DOWN + str(why)
    for name in list(public.counters):
        if str(name).startswith(_FEED_FAILURE):
            return STAND_DOWN + "state_unreliable"
    try:
        wanted = frozenset(formats)
        played = getattr(public, "format_id", None)
        if wanted and isinstance(played, str) and played and played not in wanted:
            return STAND_DOWN + "format"
    except Exception as exc:
        return f"{FAILURE}format:{type(exc).__name__}"
    return ""


def _stream_damage(entry: _Entry) -> str:
    """Why the battle's stream cannot be trusted any more, or ''.

    Two losses the tracker itself cannot see: an entry of the stream that was
    not an event (the shadow skipped it, so whatever it was is missing from
    the state), and a turn number that jumps by more than one (a lost turn
    line: two turns were read as one). Either stays true for the rest of the
    battle, like ``state_unreliable``.
    """
    try:
        shadow = entry.shadow
        for name, count in list(shadow.counters.items()):
            if str(name).startswith(_STREAM_SKIP) and count:
                return STAND_DOWN + "stream_damaged"
        segments = shadow.public.segments
        for index in range(entry.segments_checked, len(segments)):
            number = int(segments[index].turn)
            if entry.last_turn and number > entry.last_turn + 1:
                entry.turn_gap = True
            entry.last_turn = number
        entry.segments_checked = len(segments)
        return STAND_DOWN + "turn_gap" if entry.turn_gap else ""
    except Exception as exc:
        return f"{FAILURE}stream:{type(exc).__name__}"


def _says_something(
    norm: Mapping[str, np.ndarray], example: Mapping[str, np.ndarray]
) -> bool:
    """Whether a normalised prediction is a probability table for every
    occupied slot of the one example it was made for.

    ``normalize_prediction`` turns a row with no legal mass into zeros. Served,
    that reads "this slot does nothing" (no action, ``p_attacks`` 0) with no
    counter raised. So: the legal actions of an occupied slot must sum to 1,
    and every move with mass whose target classes are not all illegal must
    have its legal targets sum to 1.
    """
    action = np.asarray(norm["action"], dtype=np.float64)[0]
    target = np.asarray(norm["target"], dtype=np.float64)[0]
    legal = expand_target_mask(np.asarray(example["cand_tmask"]))
    occupied = np.asarray(example["act_mon"]) >= 0
    for index in range(N_SLOT):
        if not bool(occupied[index]):
            continue
        if abs(float(action[index].sum()) - 1.0) > _MASS_TOLERANCE:
            return False
        n_move = target.shape[1]
        for column in range(n_move):
            if float(action[index, column]) <= 0.0 or not legal[index, column].any():
                continue
            if abs(float(target[index, column].sum()) - 1.0) > _MASS_TOLERANCE:
                return False
    return True


def _predictor_failures(predictor: Any) -> int:
    """How often the predictor has fallen back to its uniform answer so far."""
    counters = getattr(predictor, "counters", None)
    if not isinstance(counters, Mapping):
        return 0
    total = 0
    for name, count in list(counters.items()):
        if str(name).startswith(_PREDICTOR_FAILURE):
            total += _whole(count, 0)
    return total


def extras_read(predictor: Any) -> tuple[str, ...]:
    """The version-2 arrays (``features.EXTRA_ARRAYS`` names) a predictor
    reads, in table order: its own ``extra_keys``, or its network's
    (``model.OppNet.extra_keys``). ``()`` for a predictor that names none: a
    count table, a network fitted with ``--extras 0``, every model from
    before layout version 2. Never raises."""
    try:
        for holder in (predictor, getattr(predictor, "net", None)):
            keys = getattr(holder, "extra_keys", None)
            if isinstance(keys, (tuple, list)):
                asked = {str(key) for key in keys}
                return tuple(spec.name for spec in EXTRA_ARRAYS if spec.name in asked)
    except Exception as exc:
        _failed("extras_read", exc)
    return ()


def _families_of(names: Iterable[str]) -> frozenset[str]:
    """The block families (``features.FAMILY_*``) that write ``names``."""
    asked = set(names)
    return frozenset(spec.family for spec in EXTRA_ARRAYS if spec.name in asked)


def _failed_extras(fresh: Mapping[str, int], families: Iterable[str]) -> list[str]:
    """The families among ``families`` whose block failed, by the names the
    featurizer counted during one encode (``fresh``), sorted."""
    out: list[str] = []
    for family in sorted(families):
        prefixes = _EXTRAS_FAILED.get(family, ())
        if any(
            count > 0 and str(name).startswith(prefixes)
            for name, count in fresh.items()
        ):
            out.append(family)
    return out


def _encode_reason(fresh: Mapping[str, int]) -> str:
    """The counter name for an example the featurizer did not make."""
    for name in fresh:
        if name.startswith(_ENCODE_SKIP):
            return STAND_DOWN + name[len(_ENCODE_SKIP) :]
    for name in fresh:
        if name.startswith(_ENCODE_FAILURE):
            return f"{FAILURE}encode:{name[len(_ENCODE_FAILURE) :]}"
    return STAND_DOWN + "no_example"


def _sheet_name(code: Any) -> str:
    return _SHEET_NAMES.get(_whole(code, SHEET_UNKNOWN), SHEET_STATE_UNKNOWN)


# --- adapters to the existing prediction shapes ----------------------------------

# "Says nothing": reliability 0 closes every gate of the existing consumers.
EMPTY_MOVES = MovePrediction((), (), (), reliability=0.0)
NO_SWITCH = SwitchPrediction(0.0, ())


def known_set_reliability(slot: SlotForecast | None) -> float:
    """``MovePrediction.reliability`` as the existing consumers mean it: the
    share of the slot's moveset that is known (shown in battle or on an open
    sheet), ``known_moves / 4``. 0.0 without a slot. Never raises."""
    try:
        if slot is None:
            return 0.0
        return min(1.0, max(0.0, float(slot.known_moves) / float(MAX_MOVES)))
    except Exception as exc:
        _failed("known_set_reliability", exc)
        return 0.0


def to_move_prediction(
    slot: SlotForecast | None,
    reliability: float | None = None,
    *,
    conditional: bool = False,
) -> MovePrediction:
    """A slot's forecast as the ``MovePrediction`` the existing consumers read.

    ``moves`` are (move id, probability) of the named candidates, ``actions``
    (move id, target name, joint probability) with the legacy target names
    (``LEGACY_TARGET_NAMES``; ``foe_a`` is our slot a for both seats) and
    ``targets`` the target distribution of the most probable move, as
    ``MovePredictor`` gives it. Probabilities are ABSOLUTE: they sum to
    1 - P(switch) - P(OTHER), so a slot that will probably switch threatens
    less. The old ``MovePredictor`` gave shares among moves only;
    ``conditional=True`` renormalises to that.

    ``reliability`` means to the consumers "how much of the set is known":
    ``PolicyPlayer`` sets it to shown moves / 4, ``opponent_reranker`` opens
    its switch-evidence gate only when both slots are at 0.999 or above (in a
    hidden-sheet game, guessed moves taken as facts made it worse), and
    ``exact_observation`` blends with a uniform prior by it. The default
    (None) therefore gives ``known_set_reliability(slot)``: 1.0 only for a
    complete known moveset. Pass a number to override it, knowingly: 1.0 on a
    closed-sheet turn opens that gate on up to twelve guessed moves. A
    consumer that wants the probabilities taken at face value with the gate
    closed should scale the probabilities itself and keep reliability below
    0.999. The moves are every named candidate with mass (the old hook gave
    at most four). A one-hot forecast with ``reliability=1.0`` gives exactly
    what ``oppmodel_oracle.certain_predictions`` gives for that action. See
    the module docstring for what this shape cannot carry. Never raises; None
    or a broken slot gives ``EMPTY_MOVES``.
    """
    if slot is None:
        return EMPTY_MOVES
    try:
        if reliability is None:
            reliability = float(slot.known_moves) / float(MAX_MOVES)
        reliability = min(1.0, max(0.0, float(reliability)))
        action, target = slot.action_probs, slot.target_probs
        named = [
            (column, name, float(action[column]))
            for column, name in enumerate(slot.moves)
            if name and float(action[column]) > 0.0
        ]
        scale = 1.0
        if conditional:
            total = sum(mass for _, _, mass in named)
            if total <= 0.0:
                return MovePrediction((), (), (), reliability=float(reliability))
            scale = 1.0 / total
        named.sort(key=lambda item: -item[2])
        moves = tuple((name, mass * scale) for _, name, mass in named)
        actions: list[tuple[str, str, float]] = []
        for column, name, mass in named:
            for position, klass in enumerate(TARGET_CLASSES):
                joint = mass * scale * float(target[column, position])
                if joint > 0.0:
                    actions.append((name, LEGACY_TARGET_NAMES[klass], joint))
        actions.sort(key=lambda item: -item[2])
        targets: list[tuple[str, float]] = []
        if named:
            top = named[0][0]
            targets = [
                (LEGACY_TARGET_NAMES[klass], float(target[top, position]))
                for position, klass in enumerate(TARGET_CLASSES)
                if float(target[top, position]) > 0.0
            ]
            targets.sort(key=lambda item: -item[1])
        return MovePrediction(
            moves, tuple(targets), tuple(actions), reliability=float(reliability)
        )
    except Exception as exc:
        _failed("to_move_prediction", exc)
        return EMPTY_MOVES


def to_switch_prediction(slot: SlotForecast | None) -> SwitchPrediction:
    """A slot's forecast as a ``SwitchPrediction``: the probability of a
    voluntary switch and, given one, (base species, probability) of each
    destination. Never raises; None or a broken slot gives ``NO_SWITCH``."""
    if slot is None:
        return NO_SWITCH
    try:
        first = len(slot.moves) + 1  # after the candidates and OTHER
        pointers = np.asarray(slot.action_probs, dtype=np.float64)[first:]
        total = float(pointers.sum())
        if total <= 0.0:
            return NO_SWITCH
        targets = [
            (slot.roster[index], float(pointers[index]) / total)
            for index in range(min(len(slot.roster), pointers.shape[0]))
            if slot.roster[index] and float(pointers[index]) > 0.0
        ]
        targets.sort(key=lambda item: -item[1])
        return SwitchPrediction(total, tuple(targets))
    except Exception as exc:
        _failed("to_switch_prediction", exc)
        return NO_SWITCH


def to_move_predictions(
    forecast: Forecast | None,
    reliability: float | None = None,
    *,
    conditional: bool = False,
    require_both: bool = True,
) -> tuple[MovePrediction, MovePrediction] | None:
    """Both opposing slots, as ``PolicyPlayer.opponent_move_predictions`` gives them.

    None without a forecast and, like that hook, when an opposing slot is
    empty; ``require_both=False`` fills an empty slot with ``EMPTY_MOVES``.
    ``reliability`` as in ``to_move_prediction`` (None: per slot, from its
    known moves). Never raises: anything that is not a ``Forecast`` gives None.
    """
    if forecast is None:
        return None
    try:
        first, second = forecast.a, forecast.b
        if require_both and (first is None or second is None):
            return None
        return (
            to_move_prediction(first, reliability, conditional=conditional),
            to_move_prediction(second, reliability, conditional=conditional),
        )
    except Exception as exc:
        _failed("to_move_predictions", exc)
        return None


def to_switch_predictions(
    forecast: Forecast | None, *, require_both: bool = True
) -> tuple[SwitchPrediction, SwitchPrediction] | None:
    """Both opposing slots, as ``PolicyPlayer.opponent_switch_predictions`` gives
    them; None under the same conditions as ``to_move_predictions``. Never
    raises."""
    if forecast is None:
        return None
    try:
        first, second = forecast.a, forecast.b
        if require_both and (first is None or second is None):
            return None
        return to_switch_prediction(first), to_switch_prediction(second)
    except Exception as exc:
        _failed("to_switch_predictions", exc)
        return None
