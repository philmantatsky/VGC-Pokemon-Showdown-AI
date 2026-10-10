"""A class-pair coupling of a player's two slots: one number per pair of classes.

``joint.py`` builds the joint reply of a turn as the PRODUCT of the two slots'
marginals. Players do not choose their two actions independently (both protect,
or one protects while the other sets up; they split targets more than they
double), so this module holds a small correction that ``joint.joint_replies``
multiplies in when it is given one:

    joint(i, j) = p_a(i) * p_b(j) * allowed(i, j) * R_g[class(i), class(j)] / Z

``allowed`` is the legality rule of ``joint.py`` (unchanged), ``g`` the turn's
context bucket and ``class`` one of eight classes of a slot's reply: the seven
``events.INTENT_CLASSES`` and ``UNASSIGNED`` (the OTHER bucket, and a move
whose intent the dex cannot decide for its target class). R multiplies TODAY's
joint, so it is a pure interaction on top of the legality-filtered product:
inside one class pair the order and the probability ratios of the fine replies
are exactly today's. With one acting slot there is no pair and nothing changes.

Classes of a slot's reply index (``reply_classes``, the same function at fit
time and at apply time):

    candidate c at target class t   tables.move_intent[cand_move[c], t]; -1 = UNASSIGNED
    the OTHER bucket                UNASSIGNED
    a switch pointer                the switch class
    "no action" (an empty slot)     C_NONE: a ninth class whose factor is 1

Context buckets (``reply_buckets``), read from feature arrays only (``turn``
and ``foe_mon``; never a label):

    turn1      turn == 1
    later_two  a later turn, both opposing slots occupied
    later_one  a later turn, one opposing slot occupied (one target: "both
               attack the same slot" is forced there, not a choice)

The table is symmetric (renaming the actor's slots) and unchanged when the two
attack classes are exchanged in both arguments (renaming the faced side's
slots): 29 free numbers per bucket (``N_ORBIT``). Where a row faces ONE
opposing slot it holds one of the two attack classes and never the other
(which one is only the name of the occupied slot): the two are then one class,
"attack the one target", and the stored table says so (its numbers for the two
are equal, the cell that pairs them included).

THE FIT (``fit_tilt``, ``normal_form``, ``fit_coupling``). Per row two 8 x 8
matrices come from the predictor's output alone: ``M`` the mass of today's
joint on each class pair (sums to 1) and ``O`` the part of it consistent with
what the log shows (``consistent_replies``: one reply for a fully seen action,
the censored set otherwise, so a hidden slot still teaches). The likelihood of
a row under a tilt T is ``<O, T> / <M, T>``; the estimate maximises

    sum_n w_n log(<O_n, T> / <M_n, T>)  +  kappa * sum over cells (log T - T)

by a minorise-maximise step that never decreases it,

    T <- T * (Nbar + kappa * |orbit|) / (Ebar + kappa * |orbit| * T)

with ``Nbar`` / ``Ebar`` the expected observed / model counts pooled over each
symmetry orbit. The prior has mean 1 and is worth ``kappa`` rows per cell; a
cell with nothing expected and nothing observed stays exactly 1. T is then
written ``D[c] * D[d] * R[c, d]`` with R having unit weighted margins under the
bucket's mean class mix (``normal_form``): D holds whatever is per class (the
model's marginal bias on the fit rows, selection of what was seen) and is
DISCARDED, R is stored. What is stored is an interaction, not a second
calibration.

THE MARGINS OF A ONE-TARGET FIT. The mean class mix gives each attack class
half of the attacks. That is the mix of a row only where a row holds both. In
a fit whose rows never hold both (``attack_classes_merge``: the later-one
bucket) a row's own mix has all of it on one class, so the margins are taken
with the two merged (``normal_form(merged=True)``): R is then margin-neutral
under the mix every row really has. Without that the cell that pairs the two
attack classes, which holds no mass there and keeps its prior value, was
weighed as if it did, and part of D stayed in the stored table (found
2026-10-10: the attack margin of the later-one table read 1.07 under the
bucket's real mix; tables fitted before that date carry it).

WHAT DISCARDING D GIVES UP. The estimator T reproduces every pair event on
its fit rows. R, the part that is applied, does not: it is T without the
per-class factors, so a pair event whose two classes D had scaled down comes
out too high under R, and the reverse (on the first real fit, 2026-10-10:
"both use a Protect-family move" went from 23% under-predicted by the product
to 10% over-predicted by R, where T was exact; R kept 0.0108 of T's 0.0123
nats per row in sample). That is the price of storing an interaction only:
the per-slot marginals are the predictor's business. ``fit_coupling`` reports
the full tilt's in-sample gain beside R's (``in_sample_gain_full_tilt``) so
the gap can be read off every fit.

Nothing a runtime calls here raises (``PairCoupling.reply_maps``,
``PairCoupling.weights``, ``class_index``): a batch that cannot be classed
gives None and a name in ``COUNTERS``. ``PairCoupling.from_payload`` is a
loader and raises ``ValueError``; the fit functions are offline helpers.

This module imports no torch (``artifact.py`` imports it to rebuild the object).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from vgc_bench.src.oppmodel.events import (
    INTENT_ATTACK_FOE_A,
    INTENT_ATTACK_FOE_B,
    INTENT_CLASSES,
    INTENT_PROTECT,
    INTENT_SWITCH,
)
from vgc_bench.src.oppmodel.features import (
    N_ROSTER,
    N_SLOT,
    N_TARGET,
    expand_target_mask,
)

FORMAT = "oppmodel-pair-coupling"
VERSION = 1
CLASS_SCHEME = "intent8"
BUCKET_SCHEME = "turn_targets"
MARGINALS_FREE = "free"  # the coupled joint's marginals are not forced back
UNASSIGNED = "unassigned"
CLASSES: tuple[str, ...] = (*INTENT_CLASSES, UNASSIGNED)
N_CLASS = len(CLASSES)
C_SWITCH = INTENT_CLASSES.index(INTENT_SWITCH)
C_PROTECT = INTENT_CLASSES.index(INTENT_PROTECT)
C_FOE_A = INTENT_CLASSES.index(INTENT_ATTACK_FOE_A)
C_FOE_B = INTENT_CLASSES.index(INTENT_ATTACK_FOE_B)
C_UNASSIGNED = N_CLASS - 1
C_NONE = N_CLASS  # "no action": row and column of ones in the padded table
BUCKET_TURN1, BUCKET_TWO, BUCKET_ONE = 0, 1, 2
BUCKETS: tuple[str, ...] = ("turn1", "later_two", "later_one")
N_BUCKET = len(BUCKETS)
# Batch keys under which a caller may hand over the two maps already made
# (the runtime keeps them with a forecast instead of the whole feature batch).
KEY_CLASS = "pair_class"
KEY_BUCKET = "pair_bucket"

DEFAULT_KAPPAS: tuple[float, ...] = (5.0, 20.0, 50.0, 200.0)
DEFAULT_FOLDS = 10
DEFAULT_SEED = 20261010
DEFAULT_MARGIN = 0.002  # nats per two-slot row
DEFAULT_FOLD_SHARE = 0.8
VERDICT_TAKEN = "taken"
VERDICT_NOT_TAKEN = "not_taken"
# How a fit's normal form read the two attack classes (``parts[...]``).
ATTACKS_SEPARATE = "separate"  # rows hold both: each has half of the mix
ATTACKS_MERGED = "merged"  # no row holds both: one class, "the one target"
_TEMPERATURE_TOLERANCE = 1e-9
# A predictor may name its whole state in ``describe()`` under this key: a
# hash of everything that decides its answers (the ensemble does: its
# members' own payloads and their weights, ``ensemble.STATE_KEY``).
# ``fitted_after`` copies it into the record and ``PairCoupling.fits`` then
# refuses every other state. A predictor that names none (a single network, a
# count table) has the record of before: kind, temperatures, calibration flag.
KEY_STATE = "state_sha256"

# Failures of the functions a runtime calls, by function and exception type.
COUNTERS: Counter[str] = Counter()


def _failed(name: str, exc: Exception) -> None:
    try:
        COUNTERS[f"{name}:{type(exc).__name__}"] += 1
    except Exception:
        pass


# --- the symmetry ---------------------------------------------------------------


def _exchange() -> np.ndarray:
    """The class permutation that renames the faced side's two slots."""
    order = np.arange(N_CLASS)
    order[C_FOE_A], order[C_FOE_B] = C_FOE_B, C_FOE_A
    return order


EXCHANGE = _exchange()


def _orbits() -> tuple[np.ndarray, np.ndarray]:
    """(orbit id of every cell ``[8, 8]``, cells per orbit)."""
    index = np.full((N_CLASS, N_CLASS), -1, dtype=np.int64)
    count = 0
    for c in range(N_CLASS):
        for d in range(N_CLASS):
            if index[c, d] >= 0:
                continue
            e, f = int(EXCHANGE[c]), int(EXCHANGE[d])
            for a, b in ((c, d), (d, c), (e, f), (f, e)):
                index[a, b] = count
            count += 1
    return index, np.bincount(index.reshape(-1), minlength=count)


ORBIT, ORBIT_SIZE = _orbits()
N_ORBIT = int(ORBIT_SIZE.size)


def symmetrise(table: np.ndarray) -> np.ndarray:
    """``table`` ``[..., 8, 8]`` with every symmetry orbit set to its mean.

    The result is exactly symmetric and exactly unchanged by the exchange of
    the two attack classes (every cell of an orbit holds the same float).
    """
    values = np.asarray(table, dtype=np.float64)
    flat = values.reshape(-1, N_CLASS * N_CLASS)
    cells = ORBIT.reshape(-1)
    out = np.empty_like(flat)
    for row in range(flat.shape[0]):
        mean = np.bincount(cells, weights=flat[row], minlength=N_ORBIT) / ORBIT_SIZE
        out[row] = mean[cells]
    return out.reshape(values.shape)


def is_symmetric(table: np.ndarray) -> bool:
    """Whether ``table`` ``[..., 8, 8]`` has both symmetries exactly."""
    values = np.asarray(table)
    if values.shape[-2:] != (N_CLASS, N_CLASS):
        return False
    swapped = np.swapaxes(values, -1, -2)
    renamed = values[..., EXCHANGE, :][..., :, EXCHANGE]
    return bool(np.array_equal(values, swapped) and np.array_equal(values, renamed))


def class_index(value: Any) -> int | None:
    """A class name or number as an index into ``CLASSES``; None for anything
    else. Never raises."""
    try:
        if isinstance(value, str):
            return CLASSES.index(value) if value in CLASSES else None
        if isinstance(value, (bool, np.bool_)):
            return None
        number = int(value)
        return number if 0 <= number < N_CLASS and number == value else None
    except Exception:
        return None


# --- classes and buckets of a batch ---------------------------------------------


def reply_classes(batch: Mapping[str, np.ndarray], move_intent: Any) -> np.ndarray:
    """``[N, 2, R]`` int8: the class of every reply index of each slot.

    ``R = n_cand * 5 + 1 + 6`` (``joint.reply_size``). Reads ``cand_move``
    only; ``move_intent`` is the featurizer's ``tables.move_intent``. Call it
    on a batch that has not been through ``densify`` (as ``intent_probs``).
    Raises ``ValueError`` for a candidate id outside the table.
    """
    moves = np.asarray(batch["cand_move"]).astype(np.int64)
    if moves.ndim == 2:
        moves = moves[None]
    if moves.ndim != 3 or moves.shape[1] != N_SLOT:
        raise ValueError(f"cand_move {moves.shape} is not [N, 2, n_cand]")
    table = np.asarray(move_intent).astype(np.int64)
    if table.ndim != 2 or table.shape[1] != N_TARGET:
        raise ValueError(f"move_intent {table.shape} is not [moves, {N_TARGET}]")
    if moves.size and (int(moves.max()) >= table.shape[0] or int(moves.min()) < 0):
        raise ValueError("candidate move id outside the tables")
    n, _, n_cand = moves.shape
    intent = table[moves]  # [N, 2, n_cand, 5]
    intent = np.where((intent >= 0) & (intent < N_CLASS - 1), intent, C_UNASSIGNED)
    out = np.empty((n, N_SLOT, n_cand * N_TARGET + 1 + N_ROSTER), dtype=np.int8)
    out[:, :, : n_cand * N_TARGET] = intent.reshape(n, N_SLOT, n_cand * N_TARGET)
    out[:, :, n_cand * N_TARGET] = C_UNASSIGNED
    out[:, :, n_cand * N_TARGET + 1 :] = C_SWITCH
    return out


def reply_buckets(batch: Mapping[str, np.ndarray]) -> np.ndarray:
    """``[N]`` int8: the context bucket of every example (``BUCKETS``).

    Reads ``turn`` and ``foe_mon`` only.
    """
    turn = np.asarray(batch["turn"]).astype(np.int64).reshape(-1)
    foes = np.asarray(batch["foe_mon"]).astype(np.int64).reshape(-1, N_SLOT)
    if turn.shape[0] != foes.shape[0]:
        raise ValueError(f"turn {turn.shape} and foe_mon {foes.shape} disagree")
    two = (foes >= 0).all(-1)
    return np.where(
        turn == 1, BUCKET_TURN1, np.where(two, BUCKET_TWO, BUCKET_ONE)
    ).astype(np.int8)


# --- the stored object ----------------------------------------------------------


def _checked_table(value: Any) -> np.ndarray:
    table = np.array(value, dtype=np.float64)
    if table.shape != (N_BUCKET, N_CLASS, N_CLASS):
        raise ValueError(
            f"coupling table {table.shape} is not [{N_BUCKET}, {N_CLASS}, {N_CLASS}]"
        )
    if not np.isfinite(table).all() or not (table > 0).all():
        raise ValueError("coupling table holds a non-finite or non-positive entry")
    if not is_symmetric(table):
        raise ValueError(
            "coupling table is not symmetric, or changes when the two attack "
            "classes are exchanged"
        )
    table.setflags(write=False)
    return table


@dataclass(frozen=True, eq=False)
class PairCoupling:
    """The stored table and what it was fitted after.

    ``table`` ``[3, 8, 8]`` (bucket, class of slot a, class of slot b),
    positive, symmetric, unchanged by the exchange of the two attack classes,
    read-only. ``fitted_after`` names the predictor state the fit saw (kind,
    action and target temperature, whether an event calibration was in
    force); ``info`` is the fit record, plain data, read by reports only.
    """

    table: np.ndarray
    name: str = ""
    fitted_after: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    marginals: str = MARGINALS_FREE
    class_scheme: str = CLASS_SCHEME
    bucket_scheme: str = BUCKET_SCHEME

    @classmethod
    def build(
        cls,
        table: Any,
        *,
        name: str = "",
        fitted_after: Mapping[str, Any] | None = None,
        info: Mapping[str, Any] | None = None,
    ) -> "PairCoupling":
        """A coupling from a ``[3, 8, 8]`` table. Raises ``ValueError`` when
        the table is not positive, finite and symmetric both ways."""
        return cls(
            _checked_table(table), str(name), dict(fitted_after or {}), dict(info or {})
        )

    @classmethod
    def identity(cls, name: str = "identity") -> "PairCoupling":
        """All ones: the joint of today."""
        return cls.build(np.ones((N_BUCKET, N_CLASS, N_CLASS)), name=name)

    @property
    def is_identity(self) -> bool:
        return bool((self.table == 1.0).all())

    def padded(self) -> np.ndarray:
        """``[3, 9, 9]``: the table with a row and column of ones for
        ``C_NONE`` (an empty slot)."""
        out = np.ones((N_BUCKET, N_CLASS + 1, N_CLASS + 1), dtype=np.float64)
        out[:, :N_CLASS, :N_CLASS] = self.table
        return out

    def weights(self, bucket: Any) -> np.ndarray | None:
        """The ``[8, 8]`` table of one bucket (a name or an index), read-only.
        Never raises: None for a bucket that does not exist."""
        try:
            index = BUCKETS.index(bucket) if isinstance(bucket, str) else int(bucket)
            if not 0 <= index < N_BUCKET:
                return None
            return self.table[index]
        except Exception as exc:
            _failed("weights", exc)
            return None

    def reply_maps(
        self, batch: Mapping[str, np.ndarray], move_intent: Any = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """What ``joint.joint_replies`` needs to apply this coupling.

        (bucket ``[N]``, class of every reply index ``[N, 2, R + 1]`` with the
        "no action" column last in class ``C_NONE``, the padded table
        ``[3, 9, 9]``). A batch that already carries ``pair_class`` /
        ``pair_bucket`` (the runtime's kept inputs) is read as it is;
        otherwise ``cand_move``, ``turn`` and ``foe_mon`` are read and
        ``move_intent`` (the featurizer's ``tables.move_intent``) is needed.
        Never raises: None, and a name in ``COUNTERS``, when the batch cannot
        be classed or the coupling is of another scheme.
        """
        try:
            if self.class_scheme != CLASS_SCHEME or self.bucket_scheme != BUCKET_SCHEME:
                raise ValueError(
                    f"coupling scheme {self.class_scheme!r} / "
                    f"{self.bucket_scheme!r} is not this code's"
                )
            if KEY_CLASS in batch and KEY_BUCKET in batch:
                classes = np.asarray(batch[KEY_CLASS]).astype(np.int64)
                if classes.ndim == 2:
                    classes = classes[None]
                bucket = np.asarray(batch[KEY_BUCKET]).astype(np.int64).reshape(-1)
            else:
                if move_intent is None:
                    raise ValueError("a coupling needs the featurizer's move_intent")
                classes = reply_classes(batch, move_intent).astype(np.int64)
                bucket = reply_buckets(batch).astype(np.int64)
            if classes.ndim != 3 or classes.shape[1] != N_SLOT:
                raise ValueError(f"reply classes {classes.shape} are not [N, 2, R]")
            if bucket.shape[0] != classes.shape[0]:
                raise ValueError("bucket and reply classes disagree")
            if classes.size and (classes.min() < 0 or classes.max() >= N_CLASS):
                raise ValueError("a reply class outside the scheme")
            if bucket.size and (bucket.min() < 0 or bucket.max() >= N_BUCKET):
                raise ValueError("a bucket outside the scheme")
            n, _, size = classes.shape
            wide = np.full((n, N_SLOT, size + 1), C_NONE, dtype=np.int64)
            wide[:, :, :size] = classes
            return bucket, wide, self.padded()
        except Exception as exc:
            _failed("reply_maps", exc)
            return None

    def fits(self, kind: str, described: Mapping[str, Any] | None) -> str:
        """'' when this coupling was fitted after the predictor state
        ``described`` (a predictor's ``describe()``; None when it has none),
        else the reason it was not. Never raises."""
        try:
            after = self.fitted_after
            if not after:
                return ""
            if after.get("kind") not in (None, kind):
                return f"fitted after kind {after.get('kind')!r}, not {kind!r}"
            state = str(after.get(KEY_STATE) or "")
            if described is None:
                if state:
                    return (
                        f"fitted after predictor state {state[:16]}; this "
                        "predictor names no state"
                    )
                return ""
            want = after.get("temperatures") or {}
            have = described.get("temperatures") or {}
            for head in ("action", "target"):
                if head not in want:
                    continue
                if abs(float(want[head]) - float(have.get(head, 1.0))) > (
                    _TEMPERATURE_TOLERANCE
                ):
                    return (
                        f"fitted after {head} temperature {want[head]!r}; "
                        f"this predictor uses {have.get(head, 1.0)!r}"
                    )
            if "event_calibrated" in after and bool(after["event_calibrated"]) != bool(
                described.get("event_calibrated", False)
            ):
                return (
                    "fitted with event calibration "
                    f"{bool(after['event_calibrated'])}; this predictor has "
                    f"{bool(described.get('event_calibrated', False))}"
                )
            # A record that names a state is bound to it (``KEY_STATE``).
            have_state = str(described.get(KEY_STATE) or "")
            if state and state != have_state:
                return (
                    f"fitted after predictor state {state[:16]}; this predictor's "
                    f"state is {have_state[:16] or 'not named'} (its members, "
                    "their weights or their payloads differ)"
                )
            return ""
        except Exception as exc:
            return f"fitted_after cannot be read: {exc!r}"

    def describe(self) -> dict[str, Any]:
        """Plain metadata for a report (no fit record)."""
        return {
            "format": FORMAT,
            "version": VERSION,
            "name": self.name,
            "class_scheme": self.class_scheme,
            "bucket_scheme": self.bucket_scheme,
            "marginals": self.marginals,
            "classes": list(CLASSES),
            "buckets": list(BUCKETS),
            "is_identity": self.is_identity,
            "fitted_after": dict(self.fitted_after),
            "table": self.table.tolist(),
        }

    def to_payload(self) -> dict[str, Any]:
        """Plain data in the artifact's storable types."""
        return {
            "format": FORMAT,
            "version": VERSION,
            "name": self.name,
            "class_scheme": self.class_scheme,
            "bucket_scheme": self.bucket_scheme,
            "marginals": self.marginals,
            "classes": list(CLASSES),
            "buckets": list(BUCKETS),
            "table": np.array(self.table, dtype=np.float64),
            "fitted_after": dict(self.fitted_after),
            "info": dict(self.info),
        }

    @classmethod
    def from_payload(cls, payload: Any) -> "PairCoupling":
        """Rebuild a coupling. Raises ``ValueError`` for anything that is not
        one of this format, version and scheme, or whose table is damaged."""
        try:
            if not isinstance(payload, Mapping):
                raise ValueError(f"not a coupling payload: {type(payload).__name__}")
            if payload["format"] != FORMAT:
                raise ValueError(f"coupling format {payload['format']!r}")
            if payload["version"] != VERSION:
                raise ValueError(f"coupling version {payload['version']!r}")
            if payload["class_scheme"] != CLASS_SCHEME:
                raise ValueError(f"class scheme {payload['class_scheme']!r}")
            if payload["bucket_scheme"] != BUCKET_SCHEME:
                raise ValueError(f"bucket scheme {payload['bucket_scheme']!r}")
            if list(payload["classes"]) != list(CLASSES):
                raise ValueError("the stored class names are not this code's")
            if list(payload["buckets"]) != list(BUCKETS):
                raise ValueError("the stored bucket names are not this code's")
            marginals = str(payload.get("marginals") or MARGINALS_FREE)
            if marginals != MARGINALS_FREE:
                raise ValueError(f"marginals mode {marginals!r}")
            after = payload.get("fitted_after") or {}
            info = payload.get("info") or {}
            if not isinstance(after, Mapping) or not isinstance(info, Mapping):
                raise ValueError("fitted_after / info are not dicts")
            return cls(
                _checked_table(payload["table"]),
                str(payload.get("name") or ""),
                dict(after),
                dict(info),
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"not a coupling payload: {exc!r}") from exc

    def same_as(self, other: Any) -> bool:
        """Equal table, name, schemes and ``fitted_after`` (not ``info``)."""
        return bool(
            isinstance(other, PairCoupling)
            and np.array_equal(self.table, other.table)
            and self.name == other.name
            and self.fitted_after == other.fitted_after
            and self.marginals == other.marginals
            and self.class_scheme == other.class_scheme
            and self.bucket_scheme == other.bucket_scheme
        )


def fitted_after(kind: str, described: Mapping[str, Any] | None) -> dict[str, Any]:
    """The ``fitted_after`` record of a predictor (its ``describe()`` or None).

    Kind, temperatures and calibration flag, and ``KEY_STATE`` when the
    predictor names its state (the key is absent otherwise: the record of a
    single network is what it always was)."""
    out: dict[str, Any] = {"kind": str(kind)}
    if described is not None:
        heat = described.get("temperatures") or {}
        out["temperatures"] = {
            head: float(heat.get(head, 1.0)) for head in ("action", "target")
        }
        out["event_calibrated"] = bool(described.get("event_calibrated", False))
        if described.get(KEY_STATE):
            out[KEY_STATE] = str(described[KEY_STATE])
    return out


# --- the masses of a row --------------------------------------------------------


def consistent_replies(batch: Mapping[str, np.ndarray]) -> np.ndarray:
    """``[N, 2, R]`` bool: the replies of each slot the log is consistent with.

    ``y_set`` (inside ``action_mask``) expanded over the legal targets of each
    move, and cut to the one target where ``y_target`` is certain. A slot
    whose action is fully seen has one reply; a hidden one the censored set; a
    slot with no free choice (forced, locked, unresolved, empty) none. Reads
    labels: never give its result to a predictor.
    """
    mask = np.asarray(batch["action_mask"]).astype(bool)
    chosen = np.asarray(batch["y_set"]).astype(bool) & mask
    legal = expand_target_mask(batch["cand_tmask"])
    y_action = np.asarray(batch["y_action"]).astype(np.int64)
    y_target = np.asarray(batch["y_target"]).astype(np.int64)
    n = mask.shape[0]
    n_cand = legal.shape[-2] - 1
    moves = chosen[:, :, :n_cand, None] & legal[:, :, :n_cand]
    certain = (y_action >= 0) & (y_action < n_cand) & (y_target >= 0)
    certain &= y_target < N_TARGET
    rows, slots = np.nonzero(certain)
    column, aim = y_action[rows, slots], y_target[rows, slots]
    held = moves[rows, slots, column, aim].copy()
    moves[rows, slots, column, :] = False
    moves[rows, slots, column, aim] = held
    return np.concatenate(
        [moves.reshape(n, N_SLOT, n_cand * N_TARGET), chosen[:, :, n_cand:]], axis=-1
    )


def class_mass(
    prob: np.ndarray, classes: np.ndarray, keep: np.ndarray | None = None
) -> np.ndarray:
    """``[N, 2, 8]``: each slot's reply probability summed by class.

    ``prob`` / ``classes`` ``[N, 2, R]`` (``joint.slot_replies`` and
    ``reply_classes``); ``keep`` restricts the replies that count.
    """
    held = np.asarray(prob, dtype=np.float64)
    if keep is not None:
        held = np.where(np.asarray(keep, dtype=bool), held, 0.0)
    kinds = np.asarray(classes)
    out = np.zeros((*held.shape[:2], N_CLASS), dtype=np.float64)
    for index in range(N_CLASS):
        out[:, :, index] = np.where(kinds == index, held, 0.0).sum(-1)
    return out


def pair_masses(
    prob: np.ndarray, classes: np.ndarray, consistent: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Class-pair masses of today's joint for rows with two acting slots.

    ``model`` ``[N, 8, 8]``: the mass of today's joint (the product, the
    same-bench double switch removed, renormalised) on each class pair; sums
    to 1 where ``kept`` is positive. ``seen`` (with ``consistent``): the part
    of it on pairs of replies the log is consistent with, same denominator.
    ``kept`` ``[N]``: the mass the product kept before renormalising.
    """
    held = np.asarray(prob, dtype=np.float64)
    first_switch = held.shape[-1] - N_ROSTER

    def table(mask: np.ndarray | None) -> np.ndarray:
        use = held if mask is None else np.where(mask, held, 0.0)
        by_class = class_mass(use, classes)
        made = by_class[:, 0, :, None] * by_class[:, 1, None, :]
        overlap = (use[:, 0, first_switch:] * use[:, 1, first_switch:]).sum(-1)
        made[:, C_SWITCH, C_SWITCH] = np.maximum(
            0.0, made[:, C_SWITCH, C_SWITCH] - overlap
        )
        return made

    model = table(None)
    kept = model.sum((1, 2))
    scale = np.where(kept > 0, kept, 1.0)[:, None, None]
    out = {"model": model / scale, "kept": kept}
    if consistent is not None:
        out["seen"] = table(np.asarray(consistent, dtype=bool)) / scale
    return out


def row_gain(table: np.ndarray, model: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """``[N]``: log-probability of the consistent pair set under ``table``
    ``[8, 8]`` minus under today's joint. ``seen`` must have positive mass."""
    flat = np.asarray(table, dtype=np.float64).reshape(-1)
    cells = N_CLASS * N_CLASS
    seen_flat = np.asarray(seen, dtype=np.float64).reshape(len(seen), cells)
    model_flat = np.asarray(model, dtype=np.float64).reshape(len(model), cells)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (
            np.log(seen_flat @ flat)
            - np.log(model_flat @ flat)
            - np.log(seen_flat.sum(-1))
            + np.log(model_flat.sum(-1))
        )


# --- the estimate ---------------------------------------------------------------


def fit_tilt(
    model: np.ndarray,
    seen: np.ndarray,
    weight: np.ndarray | None = None,
    kappa: float = 20.0,
    *,
    iterations: int = 20000,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, dict[str, Any]]:
    """The raw tilt T ``[8, 8]`` (see the module text) and a record.

    The record holds ``observed`` / ``expected`` (the one-step counts at
    T = 1: summed consistent share and summed model mass per cell), the
    number of ``iterations`` used, ``converged`` and the largest last change
    of log T. A cell whose orbit has nothing expected and nothing observed is
    exactly 1. The step is slow where the prior is weak (thousands of
    iterations at ``kappa`` 1 on tens of thousands of rows): read
    ``converged`` before trusting a table.
    """
    kappa = max(0.0, float(kappa))
    n = int(np.asarray(model).shape[0])
    m_flat = np.asarray(model, dtype=np.float64).reshape(n, N_CLASS * N_CLASS)
    o_flat = np.asarray(seen, dtype=np.float64).reshape(n, N_CLASS * N_CLASS)
    w = np.ones(n) if weight is None else np.asarray(weight, dtype=np.float64)
    cells = ORBIT.reshape(-1)
    prior = kappa * ORBIT_SIZE
    tilt = np.ones(N_ORBIT, dtype=np.float64)
    first: dict[str, np.ndarray] = {}
    change, used = 0.0, 0
    for used in range(1, max(1, int(iterations)) + 1):
        flat = tilt[cells]
        over_o = o_flat @ flat
        over_m = m_flat @ flat
        share_o = np.divide(w, over_o, out=np.zeros(n), where=over_o > 0)
        share_m = np.divide(w, over_m, out=np.zeros(n), where=over_m > 0)
        n_cell = flat * (share_o @ o_flat)
        e_cell = flat * (share_m @ m_flat)
        if not first:
            first = {
                "observed": n_cell.reshape(N_CLASS, N_CLASS).copy(),
                "expected": e_cell.reshape(N_CLASS, N_CLASS).copy(),
            }
        n_orbit = np.bincount(cells, weights=n_cell, minlength=N_ORBIT)
        e_orbit = np.bincount(cells, weights=e_cell, minlength=N_ORBIT)
        bottom = e_orbit + prior * tilt
        ratio = np.divide(
            n_orbit + prior, bottom, out=np.ones(N_ORBIT), where=bottom > 0
        )
        ratio = np.where((n_orbit + prior) > 0, ratio, 1.0)
        change = float(np.abs(np.log(np.maximum(ratio, 1e-300))).max())
        tilt = tilt * ratio
        if change < tolerance:
            break
    if not first:
        first = {
            "observed": np.zeros((N_CLASS, N_CLASS)),
            "expected": np.zeros((N_CLASS, N_CLASS)),
        }
    record = {
        "rows": n,
        "weight": float(w.sum()),
        "kappa": kappa,
        "iterations": used,
        "converged": bool(change < tolerance),
        "last_change": change,
        "observed": first["observed"],
        "expected": first["expected"],
    }
    return tilt[cells].reshape(N_CLASS, N_CLASS), record


def class_mix(model: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    """``[8]``: the mean class mix of today's joint over the rows, both slots
    pooled and the two attack classes averaged (the symmetries of the table)."""
    held = np.asarray(model, dtype=np.float64)
    n = held.shape[0]
    if n == 0:
        return np.zeros(N_CLASS)
    w = np.ones(n) if weight is None else np.asarray(weight, dtype=np.float64)
    total = float(w.sum())
    if total <= 0:
        return np.zeros(N_CLASS)
    mix = 0.5 * (w @ held.sum(2) + w @ held.sum(1)) / total
    return 0.5 * (mix + mix[EXCHANGE])


def attack_classes_merge(model: np.ndarray, weight: np.ndarray | None = None) -> bool:
    """Whether the two attack classes are ONE class on these rows.

    True when some row (of positive weight) holds mass on an attack class and
    no row holds mass on both: every row then faces one target, and which of
    the two classes it holds is only the name of the occupied slot (the
    later-one bucket). False when a row holds both, and for rows without an
    attack at all.
    """
    held = np.asarray(model, dtype=np.float64)
    if held.shape[0] == 0:
        return False
    first = held[:, C_FOE_A, :].sum(-1) + held[:, :, C_FOE_A].sum(-1)
    second = held[:, C_FOE_B, :].sum(-1) + held[:, :, C_FOE_B].sum(-1)
    counted = np.ones(held.shape[0], dtype=bool)
    if weight is not None:
        counted = np.asarray(weight, dtype=np.float64) > 0
    one = counted & ((first > 0) | (second > 0))
    both = counted & (first > 0) & (second > 0)
    return bool(one.any() and not both.any())


def normal_form(
    tilt: np.ndarray,
    mix: np.ndarray,
    *,
    merged: bool = False,
    iterations: int = 5000,
    tolerance=1e-13,
) -> tuple[np.ndarray, np.ndarray]:
    """(R, D) with ``tilt[c, d] = D[c] * D[d] * R[c, d]`` and unit margins.

    ``sum_d mix[d] * R[c, d] = 1`` for every class with mass in ``mix``. A
    class without mass has no margin to fix: its D is 1 and its row and
    column of R are 1 (nothing was fitted there). R comes back exactly
    symmetric both ways (``symmetrise``).

    ``merged`` (rows that never hold both attack classes,
    ``attack_classes_merge``): the two attack classes are one class. The
    margins are taken with the whole attack mass of ``mix`` on that one class,
    which is the mix each such row has, and both classes get its numbers:
    ``D[b] = D[a]``, row and column b of R are row and column a, and the cell
    that pairs the two is the class's own cell (``R[a, b] = R[a, a]``; the
    tilt's value there is not read: no row has mass on it). R's margins are 1
    under ``mix`` and under either single-class version of it.
    """
    t = np.asarray(tilt, dtype=np.float64)
    m = np.array(mix, dtype=np.float64)
    if merged:
        m[C_FOE_A] += m[C_FOE_B]
        m[C_FOE_B] = 0.0
    live = m > 0
    scale = np.ones(N_CLASS, dtype=np.float64)
    for _ in range(max(1, int(iterations))):
        margin = (t / scale[None, :]) @ m  # what D[c] must equal
        new = np.where(live & (margin > 0), np.sqrt(scale * margin), 1.0)
        moved = float(np.abs(np.log(new) - np.log(scale)).max())
        scale = new
        if moved < tolerance:
            break
    rest = t / (scale[:, None] * scale[None, :])
    rest[~live, :] = 1.0
    rest[:, ~live] = 1.0
    if merged:
        scale[C_FOE_B] = scale[C_FOE_A]
        rest[C_FOE_B, :] = rest[C_FOE_A, :]
        rest[:, C_FOE_B] = rest[:, C_FOE_A]
    return symmetrise(rest), scale


def fold_index(groups: np.ndarray, folds: int, seed: int) -> tuple[np.ndarray, int]:
    """(fold of every row, number of folds): whole groups stay together.

    The convention of ``calibration._fold_index`` (same arithmetic, same use
    of the seed), repeated here so this module needs nothing but numpy.
    """
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    inverse = np.asarray(inverse, dtype=np.int64).reshape(-1)
    n_groups = int(inverse.max()) + 1 if inverse.size else 0
    used = int(max(0, min(int(folds), n_groups)))
    if used < 2:
        return np.zeros(inverse.shape, dtype=np.int64), 0
    fold_of_group = np.empty(n_groups, dtype=np.int64)
    fold_of_group[np.random.default_rng(seed).permutation(n_groups)] = (
        np.arange(n_groups) % used
    )
    return fold_of_group[inverse], used


def fit_tables(
    model: np.ndarray,
    seen: np.ndarray,
    weight: np.ndarray | None,
    bucket: np.ndarray,
    kappa: float,
    bucketed: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """The stored table ``[3, 8, 8]`` from the rows given, and its parts.

    Pooled (``bucketed`` False): one fit over the turn-1 and later-two rows,
    carried by both of those buckets. Bucketed: each of the two has its own.
    The later-one bucket always has its own fit and never inherits the pooled
    one. ``parts[name]`` holds, per fit, the raw tilt ``T``, ``D``, ``R``, the
    class ``mix``, the one-step ``observed`` / ``expected`` counts, the
    ``buckets`` that carry it and ``attack_classes``: ``ATTACKS_MERGED`` when
    no row of the fit holds both attack classes (the normal form then reads
    them as one class, and ``T``'s cell pairing the two is given the class's
    own value so that ``T = D D R`` holds in every cell), else
    ``ATTACKS_SEPARATE``.
    """
    group = np.asarray(bucket).astype(np.int64)
    w = None if weight is None else np.asarray(weight, dtype=np.float64)
    table = np.ones((N_BUCKET, N_CLASS, N_CLASS), dtype=np.float64)
    parts: dict[str, Any] = {}

    def one(name: str, rows: np.ndarray, targets: Sequence[int]) -> None:
        part_w = None if w is None else w[rows]
        tilt, record = fit_tilt(model[rows], seen[rows], part_w, kappa)
        mix = class_mix(model[rows], part_w)
        merged = attack_classes_merge(model[rows], part_w)
        rest, scale = normal_form(tilt, mix, merged=merged)
        if merged:
            tilt = tilt.copy()
            own = tilt[C_FOE_A, C_FOE_A]
            tilt[C_FOE_A, C_FOE_B] = tilt[C_FOE_B, C_FOE_A] = own
        for index in targets:
            table[index] = rest
        parts[name] = {
            **record,
            "T": tilt,
            "D": scale,
            "R": rest,
            "mix": mix,
            "buckets": [int(index) for index in targets],
            "attack_classes": ATTACKS_MERGED if merged else ATTACKS_SEPARATE,
        }

    two = (group == BUCKET_TURN1) | (group == BUCKET_TWO)
    if bucketed:
        one(BUCKETS[BUCKET_TURN1], group == BUCKET_TURN1, (BUCKET_TURN1,))
        one(BUCKETS[BUCKET_TWO], group == BUCKET_TWO, (BUCKET_TWO,))
    else:
        one("pooled", two, (BUCKET_TURN1, BUCKET_TWO))
    one(BUCKETS[BUCKET_ONE], group == BUCKET_ONE, (BUCKET_ONE,))
    return table, parts


def full_tilt(parts: Mapping[str, Any]) -> np.ndarray:
    """``[3, 8, 8]``: the raw tilt T of every bucket, from ``fit_tables``'
    ``parts`` (ones for a bucket no part carries). What the fit estimated
    before its per-class factors were taken out; never stored, never applied:
    for reports that say what the stored R gives up."""
    out = np.ones((N_BUCKET, N_CLASS, N_CLASS), dtype=np.float64)
    for part in parts.values():
        for index in part.get("buckets") or ():
            if 0 <= int(index) < N_BUCKET:
                out[int(index)] = np.asarray(part["T"], dtype=np.float64)
    return out


def table_gain(
    table: np.ndarray, model: np.ndarray, seen: np.ndarray, bucket: np.ndarray
) -> np.ndarray:
    """``[N]``: ``row_gain`` of every row under its own bucket's table."""
    group = np.asarray(bucket).astype(np.int64)
    out = np.zeros(group.shape[0], dtype=np.float64)
    for index in range(N_BUCKET):
        rows = group == index
        if rows.any():
            out[rows] = row_gain(table[index], model[rows], seen[rows])
    return out


def _mean(values: np.ndarray, weight: np.ndarray | None) -> float:
    if values.size == 0:
        return float("nan")
    if weight is None:
        return float(values.mean())
    total = float(weight.sum())
    return float((values * weight).sum() / total) if total > 0 else float("nan")


def fit_coupling(
    model: np.ndarray,
    seen: np.ndarray,
    weight: np.ndarray | None,
    bucket: np.ndarray,
    battle: np.ndarray,
    *,
    kappas: Sequence[float] = DEFAULT_KAPPAS,
    folds: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
    margin: float = DEFAULT_MARGIN,
    fold_share: float = DEFAULT_FOLD_SHARE,
    buckets: bool = True,
) -> dict[str, Any]:
    """Choose the shrinkage and the bucket rule by grouped cross-validation,
    decide whether the coupling is taken, and fit the table on every row.

    Rows are two-slot rows whose consistent set has positive mass (the caller
    filters). Folds hold whole battles. The score of a fold is the weighted
    mean of ``row_gain`` on its held-out rows under the table fitted on the
    other folds, in nats per row. ``kappa`` is the grid value with the best
    mean score (ties to the larger); bucket tables replace the pooled one only
    when they beat it by ``margin`` in at least ``fold_share`` of the folds.

    ACCEPTANCE: ``verdict`` is ``taken`` only when the chosen setting's
    held-out gain is above ``margin`` in at least ``fold_share`` of the folds
    (8 of 10 at the defaults). The grid choice is made on the same folds, so
    the per-fold gains of the chosen setting are slightly favourable; every
    grid value's folds are in the record (``cv``) for that reason.
    ``design_reading`` is the rule as the design first worded it (the mean
    held-out gain above ``margin`` AND a positive gain in at least
    ``fold_share`` of the folds), reported beside the verdict and never the
    verdict: neither rule implies the other in every case.

    Returns ``verdict``, ``table`` (``[3, 8, 8]``, fitted on all rows at the
    chosen setting, whatever the verdict), ``parts``, ``kappa``, ``bucketed``,
    ``fold_gain`` (weighted) and ``fold_gain_unweighted`` per fold, ``cv``,
    ``unweighted`` (the same fit with unit weights: ``table``, ``parts``),
    ``converged`` (the final fits) and ``cv_fits_not_converged``.
    ``in_sample_gain_full_tilt`` is the in-sample gain the raw tilt T would
    have (D kept): the gap to ``in_sample_gain`` is what storing R alone
    gives up.
    """
    n = int(np.asarray(model).shape[0])
    w = None if weight is None else np.asarray(weight, dtype=np.float64)
    fold, used = fold_index(np.asarray(battle), folds, seed)
    grid = sorted({float(value) for value in kappas})
    schemes = (False, True) if buckets else (False,)
    scores = np.full((len(schemes), len(grid), max(used, 1)), np.nan)
    plain = np.full_like(scores, np.nan)
    sizes: list[int] = []
    unfinished = 0
    for index in range(used):
        held = fold == index
        rest = ~held
        sizes.append(int(held.sum()))
        held_w = None if w is None else w[held]
        rest_w = None if w is None else w[rest]
        for s, bucketed in enumerate(schemes):
            for g, kappa in enumerate(grid):
                table, made = fit_tables(
                    model[rest], seen[rest], rest_w, bucket[rest], kappa, bucketed
                )
                unfinished += sum(1 for part in made.values() if not part["converged"])
                gain = table_gain(table, model[held], seen[held], bucket[held])
                scores[s, g, index] = _mean(gain, held_w)
                plain[s, g, index] = _mean(gain, None)

    def best(s: int) -> int:
        means = np.nanmean(scores[s], axis=1) if used else np.zeros(len(grid))
        top = float(np.nanmax(means)) if np.isfinite(means).any() else 0.0
        return max(g for g in range(len(grid)) if not means[g] < top)

    need = int(np.ceil(float(fold_share) * used - 1e-9)) if used else 1
    pooled_g = best(0)
    chosen_s, chosen_g = 0, pooled_g
    bucket_folds = 0
    if buckets and used:
        bucket_g = best(1)
        better = scores[1, bucket_g] - scores[0, pooled_g]
        bucket_folds = int((better > margin).sum())
        if bucket_folds >= need:
            chosen_s, chosen_g = 1, bucket_g
    fold_gain = scores[chosen_s, chosen_g] if used else np.zeros(0)
    passed = int((fold_gain > margin).sum()) if used else 0
    taken = bool(used and passed >= need)
    positive = int((fold_gain > 0).sum()) if used else 0
    mean_gain = float(np.nanmean(fold_gain)) if used else float("nan")
    kappa = grid[chosen_g]
    bucketed = bool(schemes[chosen_s])
    table, parts = fit_tables(model, seen, w, bucket, kappa, bucketed)
    flat_table, flat_parts = fit_tables(model, seen, None, bucket, kappa, bucketed)
    in_sample = table_gain(table, model, seen, bucket)
    in_sample_full = table_gain(full_tilt(parts), model, seen, bucket)
    return {
        "verdict": VERDICT_TAKEN if taken else VERDICT_NOT_TAKEN,
        "rows": n,
        "folds": used,
        "fold_rows": sizes,
        "seed": int(seed),
        "margin": float(margin),
        "fold_share": float(fold_share),
        "folds_needed": need,
        "folds_passed": passed,
        "folds_positive": positive,
        "design_reading": bool(used and mean_gain > margin and positive >= need),
        "kappa": kappa,
        "bucketed": bucketed,
        "bucket_folds_better": bucket_folds,
        "fold_gain": fold_gain,
        "fold_gain_unweighted": plain[chosen_s, chosen_g] if used else np.zeros(0),
        "mean_gain": mean_gain,
        "in_sample_gain": _mean(in_sample, w),
        "in_sample_gain_unweighted": _mean(in_sample, None),
        "in_sample_gain_full_tilt": _mean(in_sample_full, w),
        "converged": all(part["converged"] for part in parts.values()),
        "cv_fits_not_converged": unfinished,
        "cv": {
            "kappas": grid,
            "schemes": ["pooled", "bucketed"][: len(schemes)],
            "gain": scores,
            "gain_unweighted": plain,
        },
        "table": table,
        "parts": parts,
        "unweighted": {"table": flat_table, "parts": flat_parts},
    }
