"""An ensemble of opponent predictors: the mean of the members' fine distributions.

    built = from_artifacts([("base", base_path), ("wide", wide_path)])
    save_artifact(out, kind=KIND, name=built.predictor.name,
                  featurizer=built.featurizer,
                  predictor_payload=built.predictor.to_payload(), extra=built.extra)
    loaded = load_predictor(out)          # loaded.predictor is an EnsemblePredictor
    pred = loaded.predictor.predict(batch)

WHAT ``predict`` RETURNS. The three arrays of ``features.Predictor``. A
member's ``target`` array is conditional on its ``action`` array, so the mean
of two models is taken on their JOINT fine distributions (move x target, and
the six switch pointers), each member first cleaned by
``features.normalize_prediction`` (the step every score and the runtime apply
to a prediction anyway), and written back in the contract's two factors. With
weights ``w`` that sum to 1:

    action_e(m)     = sum_i w_i a_i(m)
    target_e(t | m) = sum_i w_i a_i(m) t_i(t | m)  /  sum_i w_i a_i(m)
    mega_e          = sum_i w_i q_i

so ``features.fine_probs`` of the result IS the weighted mean of the members'
``features.fine_probs`` (to rounding, 1e-15), the action marginal is the mean
of the action marginals (what a censored slot-turn is scored on), and the
result is a proper distribution on the legal support. A move no member gives
any mass keeps the weighted mean of the members' target rows (nothing reads
it; it stays a distribution). The Mega probability is a marginal of its own in
the models and its mean is a plain mean. Output arrays are float64. An
ensemble of ONE member returns that member's own prediction, untouched.

A MEMBER THAT FAILS FAILS THE CALL. A member that raises, that counts one of
its own ``predict_error`` (it fell back to its uniform answer), that returns
something that cannot be normalised, or whose fine distribution is not proper
(a move with mass and no mass on any legal target) makes the WHOLE ensemble
answer with the failure value of every predictor, ``features.uniform_prediction``,
and count ``predict_error:member_<why>:<label>`` in ``counters``. The mean of
the members that are left is never served. The runtime
(``runtime._predictor_failures``) and the scorecard (``predict_errors``) read
that prefix: the runtime returns no forecast for the call, the scorecard
refuses to score it. What each member counted during a call is copied to
``counters`` as ``member:<label>:<name>``, so ``OpponentPredictor.diagnostics``
shows it. ``predict`` never raises.

ONE FEATURIZER. The members are fed ONE batch, made by one featurizer, so the
members' own featurizers must be that one. Allowed (``shared_featurizer``):

* every member's featurizer payload is identical; or
* the payloads are identical once the keys that describe the version-2 extra
  arrays are removed (``EXTRA_PAYLOAD_KEYS`` and the ``set_prior*`` keys of
  ``Featurizer.to_payload``), ONE member's featurizer writes every extra array
  any member's does (the richest: it becomes the ensemble's), and each other
  member's payload is exactly the richest narrowed to its own arrays
  (``Featurizer.narrowed``: the same matchup definitions, the same set table).
  A version-2 featurizer writes every version-1 array bit for bit and each
  network reads the arrays it names (``model.to_tensors`` with its own
  ``extra_keys``), so a member fed the richer batch gives its stand-alone
  output exactly.

Anything else is refused with ``ValueError``. ``extra_keys`` is the union of
the arrays the members read: a runtime stands down on a call whose featurizer
block failed for any of them (``opp_predictor_error:extras_degraded``).

THE PAYLOAD (artifact kind ``'ensemble'``):

    format    'oppmodel-ensemble-payload'
    version   1
    name      the ensemble's name
    combine   'mean_of_fine_distributions'
    weights   one positive number per member, summing to 1
    elo_mode  'blank' when every member is Elo-blind, else 'keep'
    members   a list, in order, of
        label    the member's name inside the ensemble (unique)
        kind     the member's own artifact kind ('oppnet' or 'table')
        name     the member artifact's name
        tag      the tag of the member's training run (its name when none)
        sha256   of the member's artifact file ('' when not built from a file)
        path     where that file was ('' likewise)
        extras   the version-2 arrays the member's own featurizer wrote
        payload  the member's own predictor payload, UNCHANGED: its weights,
                 its temperatures and any event calibration travel with it

A member's payload is handed to its kind's own ``from_payload`` with the
ensemble's featurizer, exactly as ``artifact.load_predictor`` hands over a
stand-alone payload. An ensemble is not a member kind (no nesting).

A PAIR COUPLING belongs to one predictor state, so a member that carries one
is refused (``from_artifacts`` / ``from_loaded``). A coupling is fitted on the
ensemble afterwards (``training/fit_oppmodel_coupling.py`` on the ensemble's
artifact). It binds to ``describe()``:

* kind ``'ensemble'``;
* ``state_sha256`` (``EnsemblePredictor.state_sha256``): a hash of what
  decides ``predict``, namely the way to combine, the weights (to 12
  significant digits) and, member by member in order, the member's kind and
  the sha256 of its own predictor payload (``payload_hash``: every number of
  its network or count tables, its temperatures, its calibrations). Other
  weights, another member, a member more or less, a re-tempered or fine-tuned
  member: each is another state, and ``artifact.load_predictor`` refuses the
  coupling on it. The hash is taken from the payloads the ensemble holds,
  never from the recorded ``sha256`` of a member's file (a record is not the
  weights). Labels and names are not part of it, and neither is the
  featurizer (as for a single network);
* as for any predictor, per head the weighted mean of the members'
  temperatures and whether any member is event calibrated. That mean alone
  does NOT bind the weights: with members of equal temperatures (two
  uncalibrated networks, any count tables) it is the same number for every
  weighting.

A coupling whose record names no state (one fitted on an ensemble before the
state was recorded, or a record written by hand) is read as before and is
bound by the kind, the temperatures and the calibration flag alone.

FINE-TUNED MEMBERS. A member fine-tuned on the bot's own ladder games
(``training/finetune_oppmodel.py``) says so in its artifact's ``extra``:
``finetune`` with the list ``seen_games``, and a marked dataset manifest. The
ensemble has seen what any member has seen, so its own ``extra``
(``from_artifacts`` / ``from_loaded``) carries, in the same keys a reader of a
fine-tuned artifact looks at: ``finetune['seen_games']`` (the union of the
members' games; ``finetune['members']`` holds each fine-tuned member's whole
record), ``dataset['manifest_sha256']`` marked
``finetuned-on-ladder-holdout:<games>-games:<build>`` (so a reader comparing
it with a build's manifest finds another training set, also when only some
members were fine-tuned) with the build under
``dataset['source_manifest_sha256']``, and ``dataset_tag``. A member that was
fine-tuned and does not list its games is refused: the ensemble could not say
what it has seen. Without a fine-tuned member none of these keys is written.

OLD READERS. Code from before this kind refuses the file in
``artifact.read_artifact`` ("unknown kind 'ensemble'"), whatever its version.

Importing this module loads no torch code: a member's module is imported when
a payload is rebuilt.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np

from vgc_bench.src.oppmodel.features import (
    ELO_BLANK,
    ELO_KEEP,
    EXTRA_ARRAYS,
    Featurizer,
    normalize_prediction,
    take,
    uniform_prediction,
)

KIND = "ensemble"
PAYLOAD_FORMAT = "oppmodel-ensemble-payload"
PAYLOAD_VERSION = 1
COMBINE = "mean_of_fine_distributions"
# The kinds a member may have, and the module that rebuilds each (the same
# modules ``artifact.load_predictor`` uses for a stand-alone artifact).
MEMBER_MODULES: dict[str, str] = {
    "table": "vgc_bench.src.oppmodel.tables",
    "oppnet": "vgc_bench.src.oppmodel.model",
}
MEMBER_KINDS: tuple[str, ...] = tuple(MEMBER_MODULES)
# Keys of ``Featurizer.to_payload()`` that describe the version-2 extra arrays
# (``Featurizer.narrowed`` drops or rewrites exactly these): two featurizers
# whose payloads agree without them differ only by those arrays. Every key
# that starts with ``SET_PRIOR_PREFIX`` belongs to them as well.
EXTRA_PAYLOAD_KEYS: tuple[str, ...] = ("layout_version", "extras", "matchup")
SET_PRIOR_PREFIX = "set_prior"
# Counter names. ``FAILURE`` is the prefix every reader of a predictor's
# counters takes for "it fell back" (the runtime, the scorecard).
FAILURE = "predict_error"
MEMBER_RAISED = FAILURE + ":member_raised:"
MEMBER_DEGRADED = FAILURE + ":member_degraded:"
MEMBER_MALFORMED = FAILURE + ":member_malformed:"
MEMBER_IMPROPER = FAILURE + ":member_improper:"
MEMBER_COUNTED = "member:"
# Rows per chunk of one ``predict`` call. A multiple of the networks' own
# batch size (1024), so a member sees the same batches as when called alone.
CHUNK = 8192
_HEADS = ("action", "target", "mega")
# The key of ``describe()`` under which the ensemble names its whole state: a
# pair coupling records it and is refused on any other
# (``coupling.fitted_after`` / ``PairCoupling.fits`` read the same key).
STATE_KEY = "state_sha256"
# Significant digits of a weight inside the state hash: weights that agree
# this far give predictions that agree to rounding.
WEIGHT_DIGITS = 12
# Counted when a member's payload cannot be hashed (the state is then None).
STATE_UNREAD = "state_unread"
# What a fine-tune on the bot's own ladder games leaves in an artifact's
# ``extra`` (``training/finetune_oppmodel.py``: ``KEY_FINETUNE``, ``KEY_SEEN``,
# ``MANIFEST_MARK``; its ``seen_games`` / ``check_unseen`` read these keys).
KEY_FINETUNE = "finetune"
KEY_SEEN = "seen_games"
MANIFEST_MARK = "finetuned-on-ladder-holdout"
SEVERAL_BUILDS = "several-builds"
_PLAIN_TYPES = (bool, int, float, str)


# --- the mean -------------------------------------------------------------------


def checked_weights(weights: Sequence[float] | None, count: int) -> tuple[float, ...]:
    """``count`` weights that sum to 1: equal for None. Raises ``ValueError``
    for a wrong count or a weight that is not a positive finite number."""
    if count <= 0:
        raise ValueError("an ensemble needs at least one member")
    if weights is None:
        return (1.0 / count,) * count
    try:
        values = [float(value) for value in weights]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"weights are not numbers: {exc!r}") from exc
    if len(values) != count:
        raise ValueError(f"{len(values)} weights for {count} members")
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError(f"weights must be positive and finite: {values}")
    total = math.fsum(values)
    return tuple(value / total for value in values)


def improper_rows(norm: Mapping[str, np.ndarray]) -> int:
    """Moves of a normalised prediction that carry action mass and no mass on
    any legal target: there the two factors are not a fine distribution."""
    target = np.asarray(norm["target"])
    mass = np.asarray(norm["action"])[:, :, : target.shape[2]]
    return int(((mass > 0.0) & (target.sum(-1) <= 0.0)).sum())


def mix_predictions(
    preds: Sequence[Mapping[str, np.ndarray]],
    batch: Mapping[str, np.ndarray],
    weights: Sequence[float] | None = None,
) -> dict[str, np.ndarray]:
    """The weighted mean of predictions, as one ``action`` / ``target`` /
    ``mega`` (float64). See the module text for the formula.

    Each prediction is first normalised on the batch's masks. Raises
    ``ValueError`` when one does not fit the batch, holds a non-finite number,
    or is not a proper fine distribution (``improper_rows``).
    """
    share = checked_weights(weights, len(preds))
    norms = [normalize_prediction(pred, batch) for pred in preds]
    for position, norm in enumerate(norms):
        if not np.isfinite(norm["mega"]).all():
            raise ValueError(f"prediction {position} holds a non-finite Mega number")
        bad = improper_rows(norm)
        if bad:
            raise ValueError(
                f"prediction {position}: {bad} moves with mass and no legal target mass"
            )
    return _mean(norms, share)


def _mean(
    norms: Sequence[Mapping[str, np.ndarray]], weights: Sequence[float]
) -> dict[str, np.ndarray]:
    """``mix_predictions`` on predictions that are already normalised."""
    n_move = norms[0]["target"].shape[2]
    action = np.zeros_like(norms[0]["action"])
    mega = np.zeros_like(norms[0]["mega"])
    plain = np.zeros_like(norms[0]["target"])
    top = np.zeros_like(norms[0]["target"])
    for weight, norm in zip(weights, norms):
        action += weight * norm["action"]
        mega += weight * norm["mega"]
        plain += weight * norm["target"]
        top += weight * norm["action"][:, :, :n_move, None] * norm["target"]
    bottom = top.sum(-1, keepdims=True)
    target = np.divide(top, bottom, out=plain, where=bottom > 0.0)
    return {"action": action, "target": target, "mega": mega}


# --- one featurizer ---------------------------------------------------------------


def _canonical(value: Any, digest: Any) -> None:
    if isinstance(value, Mapping):
        digest.update(b"{")
        for key in sorted(value, key=str):
            digest.update(str(key).encode("utf-8") + b":")
            _canonical(value[key], digest)
        digest.update(b"}")
    elif isinstance(value, (list, tuple)):
        digest.update(b"[")
        for item in value:
            _canonical(item, digest)
            digest.update(b",")
        digest.update(b"]")
    elif isinstance(value, (set, frozenset)):
        try:
            ordered = sorted(value)
        except TypeError:
            ordered = sorted(value, key=repr)
        _canonical(ordered, digest)
    elif isinstance(value, np.ndarray):
        digest.update(f"nd{value.dtype.str}{tuple(value.shape)}".encode("utf-8"))
        digest.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, np.generic):
        _canonical(value.item(), digest)
    elif callable(getattr(value, "detach", None)) and callable(
        getattr(value, "numpy", None)
    ):
        # A torch tensor, told by its methods (this module imports no torch):
        # every number of it, as an array of its dtype and shape. Its repr
        # would show the corners of a large tensor only.
        _canonical(np.asarray(value.detach().cpu().numpy()), digest)
    elif isinstance(value, Path):
        _canonical(str(value), digest)
    elif type(value) not in _PLAIN_TYPES and isinstance(value, (str, float, int)):
        # An instance of a subclass reads as the built-in value it is stored as.
        if isinstance(value, str):
            _canonical(str.__str__(value), digest)
        elif isinstance(value, float):
            _canonical(float.__float__(value), digest)
        else:
            _canonical(int.__int__(value), digest)
    else:
        digest.update(f"{type(value).__name__}:{value!r}".encode("utf-8"))


def payload_hash(value: Any) -> str:
    """sha256 of nested plain data, numpy arrays and tensors.

    Dtype, shape and every byte of an array or a tensor (a tensor and an
    array of the same dtype, shape and numbers hash alike). What
    ``artifact.save_artifact`` stores in another type hashes as the stored
    value, so a payload in memory and the same payload read back from its
    file have one hash: a tuple reads as a list, a set as its sorted list, a
    numpy scalar or an instance of a subclass of str / float / int as the
    built-in value, a Path as its text.
    """
    digest = hashlib.sha256()
    _canonical(value, digest)
    return digest.hexdigest()


def is_extra_key(key: str) -> bool:
    """Whether a key of a featurizer payload describes the version-2 arrays."""
    return key in EXTRA_PAYLOAD_KEYS or str(key).startswith(SET_PRIOR_PREFIX)


def core_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """A featurizer payload without the keys of its version-2 arrays."""
    return {key: value for key, value in payload.items() if not is_extra_key(key)}


def payload_extras(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """The version-2 arrays a featurizer payload says it writes, in table order."""
    listed = {str(name) for name in (payload.get("extras") or ())}
    return tuple(spec.name for spec in EXTRA_ARRAYS if spec.name in listed)


def shared_featurizer(
    payloads: Mapping[str, Mapping[str, Any]],
) -> tuple[str, dict[str, tuple[str, ...]]]:
    """The member whose featurizer serves every member, and each member's arrays.

    ``payloads`` maps a member's label to its featurizer payload
    (``Featurizer.to_payload()``, or what an artifact stores). Returns the
    label of the richest member (the first, when all are identical) and, per
    label, the version-2 arrays that member's featurizer writes. Raises
    ``ValueError`` (the refusal of the module text) when the payloads differ
    by more than their version-2 arrays, when no one member writes every
    array, or when a member is not the richest narrowed to its own arrays.
    """
    if not payloads:
        raise ValueError("an ensemble needs at least one member")
    labels = list(payloads)
    first = labels[0]
    core = {label: core_payload(payloads[label]) for label in labels}
    for label in labels[1:]:
        differing = sorted(
            key
            for key in set(core[first]) | set(core[label])
            if key not in core[first]
            or key not in core[label]
            or payload_hash(core[first][key]) != payload_hash(core[label][key])
        )
        if differing:
            raise ValueError(
                f"members {first!r} and {label!r} have different featurizers "
                f"(payload keys that differ: {differing}): an ensemble feeds its "
                "members one batch, so their featurizers may differ only by the "
                "version-2 extra arrays"
            )
    extras = {label: payload_extras(payloads[label]) for label in labels}
    richest = max(labels, key=lambda label: len(extras[label]))
    for label in labels:
        missing = [name for name in extras[label] if name not in extras[richest]]
        if missing:
            raise ValueError(
                f"member {label!r} reads featurizer arrays {missing} that member "
                f"{richest!r} (the one with the most) does not write: no one "
                "featurizer of the members serves them all"
            )
    whole = payload_hash(payloads[richest])
    wide: Featurizer | None = None
    for label in labels:
        if label == richest or payload_hash(payloads[label]) == whole:
            continue
        if wide is None:
            wide = Featurizer.from_payload(payloads[richest])
        narrow = wide.narrowed(extras[label]).to_payload()
        if payload_hash(narrow) != payload_hash(payloads[label]):
            differing = sorted(
                key
                for key in set(narrow) | set(payloads[label])
                if key not in narrow
                or key not in payloads[label]
                or payload_hash(narrow[key]) != payload_hash(payloads[label][key])
            )
            raise ValueError(
                f"member {label!r}'s featurizer is not member {richest!r}'s "
                f"narrowed to {list(extras[label])} (payload keys that differ: "
                f"{differing}): their version-2 definitions or set tables differ"
            )
    return richest, extras


# --- members ----------------------------------------------------------------------


def predictor_failures(predictor: Any) -> int:
    """How often a predictor has fallen back to its uniform answer so far."""
    counters = getattr(predictor, "counters", None)
    if not isinstance(counters, Mapping):
        return 0
    total = 0
    for name, count in list(counters.items()):
        if str(name).startswith(FAILURE):
            try:
                total += int(count)
            except (TypeError, ValueError):
                total += 1
    return total


def extras_read(predictor: Any) -> tuple[str, ...]:
    """The version-2 arrays a predictor reads (its ``extra_keys`` or its
    network's), in table order; ``()`` when it names none."""
    for holder in (predictor, getattr(predictor, "net", None)):
        keys = getattr(holder, "extra_keys", None)
        if isinstance(keys, (tuple, list)):
            asked = {str(key) for key in keys}
            return tuple(spec.name for spec in EXTRA_ARRAYS if spec.name in asked)
    return ()


@dataclass(frozen=True)
class Member:
    """One predictor of an ensemble.

    ``payload`` is the member's own predictor payload as its artifact stored
    it (None: made from ``predictor.to_payload()`` when the ensemble is
    stored). ``extras`` are the version-2 arrays the member's own featurizer
    wrote; ``info`` holds ``name``, ``tag``, ``sha256`` and ``path``.
    """

    label: str
    kind: str
    predictor: Any
    payload: Mapping[str, Any] | None = None
    extras: tuple[str, ...] = ()
    info: Mapping[str, Any] = field(default_factory=dict)


def _label(value: Any) -> str:
    label = str(value or "").strip()
    if not label or any(mark in label for mark in ":= \t\n"):
        raise ValueError(
            f"member label {value!r}: a label is a non-empty word without "
            "':', '=' or spaces"
        )
    return label


class EnsemblePredictor:
    """``features.Predictor`` over several predictors: the weighted mean of
    their fine distributions (the module text has the formula and the rules).

    ``predict`` is safe to call from several threads (one lock around a
    call) and never raises: a failure is counted under ``counters`` and the
    answer is ``features.uniform_prediction``. The constructor raises
    ``ValueError`` for no member, a repeated label, a member of a kind that
    cannot be one, or bad weights.
    """

    kind = KIND

    def __init__(
        self,
        members: Sequence[Member],
        featurizer: Featurizer | None = None,
        *,
        weights: Sequence[float] | None = None,
        name: str = KIND,
        chunk: int = CHUNK,
    ) -> None:
        held = tuple(members)
        self.weights = checked_weights(weights, len(held))
        labels = [_label(member.label) for member in held]
        if len(set(labels)) != len(labels):
            raise ValueError(f"member labels repeat: {labels}")
        for member in held:
            if member.kind not in MEMBER_KINDS:
                raise ValueError(
                    f"member {member.label!r} has kind {member.kind!r}; a member "
                    f"is one of {MEMBER_KINDS} (an ensemble is not nested)"
                )
            if not callable(getattr(member.predictor, "predict", None)):
                raise ValueError(f"member {member.label!r} has no predict")
        self.members = held
        self.featurizer = featurizer
        self.name = str(name or KIND)
        self.chunk = max(1, int(chunk))
        self.counters: Counter[str] = Counter()
        self._lock = threading.Lock()
        # position -> sha256 of that member's stored payload (``member_states``).
        self._digests: dict[int, str] = {}
        asked = {key for member in held for key in extras_read(member.predictor)}
        # The version-2 arrays any member reads (what a runtime checks).
        self.extra_keys: tuple[str, ...] = tuple(
            spec.name for spec in EXTRA_ARRAYS if spec.name in asked
        )
        blind = all(
            getattr(member.predictor, "elo_mode", ELO_KEEP) == ELO_BLANK
            for member in held
        )
        blind = blind or (featurizer is not None and featurizer.elo_mode == ELO_BLANK)
        self.elo_mode = ELO_BLANK if blind else ELO_KEEP

    # --- what it is -----------------------------------------------------------

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(member.label for member in self.members)

    @property
    def n_parameters(self) -> int | None:
        """Parameters of all member networks; None when a member names none."""
        total = 0
        for member in self.members:
            net: Any = getattr(member.predictor, "net", None)
            count: Any = getattr(net, "n_parameters", None)
            if not callable(count):
                return None
            found: Any = count()
            total += int(found)
        return total

    def _member_described(self, member: Member) -> dict[str, Any] | None:
        describe = getattr(member.predictor, "describe", None)
        found = describe() if callable(describe) else None
        return dict(found) if isinstance(found, Mapping) else None

    def member_states(self) -> tuple[str, ...] | None:
        """Per member, in order, the sha256 of its own predictor payload
        (``payload_hash``): of the payload its artifact stored when the member
        holds one (hashed once), else of what the member would store now.
        Never raises: None, and a ``state_unread`` counter, when a member's
        payload cannot be made or hashed."""
        try:
            found: list[str] = []
            for position, member in enumerate(self.members):
                if member.payload is None:
                    found.append(payload_hash(member.predictor.to_payload()))
                    continue
                digest = self._digests.get(position)
                if digest is None:
                    digest = self._digests[position] = payload_hash(member.payload)
                found.append(digest)
            return tuple(found)
        except Exception as exc:
            self._count(f"{STATE_UNREAD}:{type(exc).__name__}")
            return None

    @property
    def state_sha256(self) -> str | None:
        """A hash of what decides ``predict``: the way to combine, the weights
        (``WEIGHT_DIGITS`` significant digits) and, member by member in order,
        its kind and the hash of its own payload (``member_states``). What a
        pair coupling binds to (the module text). None when a member's
        payload cannot be hashed."""
        return self._state(self.member_states())

    def _state(self, states: Sequence[str] | None) -> str | None:
        """``state_sha256`` from the members' payload hashes (None for None)."""
        if states is None:
            return None
        return payload_hash(
            {
                "format": PAYLOAD_FORMAT,
                "combine": COMBINE,
                "weights": [
                    format(float(weight), f".{WEIGHT_DIGITS}g")
                    for weight in self.weights
                ],
                "members": [
                    {"kind": member.kind, "payload_sha256": state}
                    for member, state in zip(self.members, states)
                ],
            }
        )

    @property
    def temperatures(self) -> dict[str, float]:
        """Per head, the weighted mean of the members' temperatures (1.0 for a
        member that has none). A pair coupling records it, as it does a single
        network's; it does not bind the weights (``state_sha256`` does)."""
        out = {"action": 0.0, "target": 0.0}
        for weight, member in zip(self.weights, self.members):
            described = self._member_described(member) or {}
            heat = described.get("temperatures") or {}
            for head in out:
                out[head] += weight * float(heat.get(head, 1.0))
        return out

    @property
    def event_calibrated(self) -> bool:
        """Whether any member rescales its switch / Protect-family events."""
        return any(
            bool((self._member_described(member) or {}).get("event_calibrated", False))
            for member in self.members
        )

    def describe(self) -> dict[str, Any]:
        """Plain metadata: what a coupling's ``fitted_after`` reads
        (``state_sha256``, ``temperatures``, ``event_calibrated``) and each
        member's own description under its label (``payload_sha256``: the
        hash of the payload it holds; ``sha256``: the record of its file)."""
        states = self.member_states()
        hashes: Sequence[str | None] = states or (None,) * len(self.members)
        return {
            "name": self.name,
            "kind": self.kind,
            "elo_mode": self.elo_mode,
            "combine": COMBINE,
            STATE_KEY: self._state(states),
            "temperatures": self.temperatures,
            "event_calibrated": self.event_calibrated,
            "weights": dict(zip(self.labels, self.weights)),
            "members": {
                member.label: {
                    "kind": member.kind,
                    "weight": weight,
                    "extras": list(member.extras),
                    **{key: member.info.get(key) for key in ("name", "tag", "sha256")},
                    "payload_sha256": state,
                    "describe": self._member_described(member),
                }
                for weight, member, state in zip(self.weights, self.members, hashes)
            },
        }

    def member_counters(self) -> dict[str, dict[str, int]]:
        """Each member's own counters, by label."""
        return {
            member.label: dict(getattr(member.predictor, "counters", None) or {})
            for member in self.members
        }

    # --- predict --------------------------------------------------------------

    def _count(self, name: str, count: int = 1) -> None:
        try:
            self.counters[name] += count
        except Exception:
            pass

    def predict(self, batch: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Probabilities ``action`` / ``target`` / ``mega``. Never raises."""
        try:
            with self._lock:
                made = self._predict(batch)
        except Exception as exc:
            self._count(f"{FAILURE}:{type(exc).__name__}")
            made = None
        if made is None:
            return uniform_prediction(batch)
        return made

    def _ask(
        self, member: Member, batch: Mapping[str, np.ndarray]
    ) -> Mapping[str, np.ndarray] | None:
        """One member's prediction, or None after counting why it failed."""
        counters = getattr(member.predictor, "counters", None)
        seen = Counter(counters) if isinstance(counters, Mapping) else Counter()
        before = predictor_failures(member.predictor)
        try:
            made = member.predictor.predict(batch)
        except Exception:
            self._count(MEMBER_RAISED + member.label)
            return None
        if isinstance(counters, Mapping):
            for name, count in (Counter(counters) - seen).items():
                self._count(f"{MEMBER_COUNTED}{member.label}:{name}", int(count))
        if predictor_failures(member.predictor) > before:
            # It caught its own failure and answered with its uniform fallback.
            self._count(MEMBER_DEGRADED + member.label)
            return None
        if not isinstance(made, Mapping) or any(head not in made for head in _HEADS):
            self._count(MEMBER_MALFORMED + member.label)
            return None
        return made

    def _predict(self, batch: Mapping[str, np.ndarray]) -> dict[str, np.ndarray] | None:
        n = int(np.asarray(batch["action_mask"]).shape[0])
        if len(self.members) == 1:
            # One member: its own answer, untouched.
            made = self._ask(self.members[0], batch)
            return None if made is None else {head: made[head] for head in _HEADS}
        parts: list[dict[str, np.ndarray]] = []
        for start in range(0, max(n, 1), self.chunk):
            rows = batch
            if n > self.chunk:
                rows = take(batch, slice(start, start + self.chunk))
            norms: list[dict[str, np.ndarray]] = []
            for member in self.members:
                made = self._ask(member, rows)
                if made is None:
                    return None
                try:
                    norm = normalize_prediction(made, rows)
                except Exception:
                    self._count(MEMBER_MALFORMED + member.label)
                    return None
                if not np.isfinite(norm["mega"]).all():
                    self._count(MEMBER_MALFORMED + member.label)
                    return None
                if improper_rows(norm):
                    self._count(MEMBER_IMPROPER + member.label)
                    return None
                norms.append(norm)
            parts.append(_mean(norms, self.weights))
        if len(parts) == 1:
            return parts[0]
        return {head: np.concatenate([part[head] for part in parts]) for head in _HEADS}

    # --- storage --------------------------------------------------------------

    def to_payload(self) -> dict[str, Any]:
        """Plain data and tensors from which ``from_payload`` rebuilds this:
        each member's own payload unchanged, the weights, the members' records."""
        members: list[dict[str, Any]] = []
        for member in self.members:
            payload = member.payload
            if payload is None:
                payload = member.predictor.to_payload()
            members.append(
                {
                    "label": member.label,
                    "kind": member.kind,
                    "name": str(member.info.get("name") or ""),
                    "tag": str(member.info.get("tag") or member.info.get("name") or ""),
                    "sha256": str(member.info.get("sha256") or ""),
                    "path": str(member.info.get("path") or ""),
                    "extras": list(member.extras),
                    "payload": payload,
                }
            )
        return {
            "format": PAYLOAD_FORMAT,
            "version": PAYLOAD_VERSION,
            "name": self.name,
            "combine": COMBINE,
            "weights": [float(weight) for weight in self.weights],
            "elo_mode": self.elo_mode,
            "n_parameters": self.n_parameters,
            "members": members,
        }


# --- rebuilding -------------------------------------------------------------------


def _member_module(kind: str) -> Any:
    if kind not in MEMBER_MODULES:
        raise ValueError(
            f"member kind {kind!r}; a member is one of {MEMBER_KINDS} "
            "(an ensemble is not nested)"
        )
    try:
        return importlib.import_module(MEMBER_MODULES[kind])
    except ImportError as exc:
        raise ValueError(
            f"kind {kind!r} needs {MEMBER_MODULES[kind]}: {exc!r}"
        ) from exc


def from_payload(
    payload: Mapping[str, Any], featurizer: Featurizer
) -> EnsemblePredictor:
    """Rebuild an ensemble from ``EnsemblePredictor.to_payload()``.

    ``featurizer`` is the ensemble's (the artifact's): every member is rebuilt
    on it by its own kind's ``from_payload``. Raises ``ValueError`` for a
    payload that is not one, a member that does not load on this featurizer,
    or a member whose arrays the featurizer does not write.
    """
    try:
        if payload["format"] != PAYLOAD_FORMAT:
            raise ValueError(f"payload format {payload['format']!r}")
        if int(payload["version"]) != PAYLOAD_VERSION:
            raise ValueError(f"ensemble payload version {payload['version']!r}")
        if payload.get("combine") != COMBINE:
            raise ValueError(f"unknown way to combine: {payload.get('combine')!r}")
        stored = list(payload["members"])
        written = set(getattr(featurizer, "extras", ()) or ())
        members: list[Member] = []
        for entry in stored:
            label = _label(entry["label"])
            kind = str(entry["kind"])
            extras = tuple(str(name) for name in (entry.get("extras") or ()))
            missing = sorted(set(extras) - written)
            if missing:
                raise ValueError(
                    f"member {label!r} was built on featurizer arrays {missing} "
                    "the ensemble's featurizer does not write"
                )
            try:
                predictor = _member_module(kind).from_payload(
                    entry["payload"], featurizer
                )
            except ValueError as exc:
                raise ValueError(f"member {label!r} did not load: {exc}") from exc
            unfed = sorted(set(extras_read(predictor)) - set(extras))
            if unfed:
                raise ValueError(
                    f"member {label!r} reads {unfed}, which its own featurizer "
                    "did not write"
                )
            members.append(
                Member(
                    label,
                    kind,
                    predictor,
                    entry["payload"],
                    extras,
                    {
                        key: str(entry.get(key) or "")
                        for key in ("name", "tag", "sha256", "path")
                    },
                )
            )
        return EnsemblePredictor(
            members,
            featurizer,
            weights=list(payload["weights"]),
            name=str(payload.get("name") or KIND),
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"not an ensemble payload: {exc!r}") from exc


# --- building from artifacts --------------------------------------------------------


class Built(NamedTuple):
    """What ``from_artifacts`` / ``from_loaded`` return: the three things
    ``artifact.save_artifact`` needs (``kind`` is ``KIND``)."""

    predictor: EnsemblePredictor
    featurizer: Featurizer
    extra: dict[str, Any]


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_record(extra: Mapping[str, Any]) -> dict[str, Any]:
    """What a member's ``extra`` says of the dataset it was fitted on."""
    found = extra.get("dataset")
    record: dict[str, Any] = dict(found) if isinstance(found, Mapping) else {}
    for key in ("manifest_sha256", "formats"):
        if key not in record and extra.get(key) is not None:
            record[key] = extra[key]
    formats = record.get("formats")
    if isinstance(formats, str):
        record["formats"] = [formats]
    return record


def _finetune_record(label: str, extra: Mapping[str, Any]) -> dict[str, Any] | None:
    """A member's own fine-tune record (its ``extra['finetune']``, the games
    as strs); None for a member that was never fine-tuned.

    Raises ``ValueError`` for a member that was fine-tuned (it has the
    record, or its dataset manifest carries ``MANIFEST_MARK``) and does not
    list the games it has seen.
    """
    found = extra.get(KEY_FINETUNE)
    manifests = (
        extra.get("manifest_sha256"),
        _dataset_record(extra).get("manifest_sha256"),
    )
    marked = any(str(value or "").startswith(MANIFEST_MARK) for value in manifests)
    if found is None and not marked:
        return None
    games = found.get(KEY_SEEN) if isinstance(found, Mapping) else None
    if (
        not isinstance(found, Mapping)
        or not isinstance(games, (list, tuple))
        or not games
    ):
        raise ValueError(
            f"member {label!r} was fine-tuned on own ladder games and its record "
            f"(extra[{KEY_FINETUNE!r}][{KEY_SEEN!r}]) does not list them: an "
            "ensemble of it could not say which games it has seen"
        )
    record = dict(found)
    record[KEY_SEEN] = [str(game) for game in games]
    return record


def _build_of(record: Mapping[str, Any]) -> str:
    """The sha256 of the dataset build behind a member's dataset record: its
    manifest, or for a fine-tuned member the build its marked manifest
    stands on. '' when the record names none."""
    manifest = str(record.get("manifest_sha256") or "")
    if not manifest.startswith(MANIFEST_MARK):
        return manifest
    source = str(record.get("source_manifest_sha256") or "")
    if not source and manifest.count(":") >= 2:
        source = manifest.split(":", 2)[2]
    return "" if source == str(None) else source


def _tag(extra: Mapping[str, Any], name: str) -> str:
    args = extra.get("args")
    found = args.get("tag") if isinstance(args, Mapping) else None
    return str(found or extra.get("tag") or name)


def _assemble(
    sources: Sequence[tuple[str, str, Any, Mapping[str, Any]]],
    featurizer_payloads: Mapping[str, Mapping[str, Any]],
    extras_of: Mapping[str, Mapping[str, Any]],
    weights: Sequence[float] | None,
    name: str,
    elo_mode: str,
) -> Built:
    """``sources``: (label, kind, predictor payload, info) per member."""
    richest, extras = shared_featurizer(featurizer_payloads)
    featurizer = Featurizer.from_payload(featurizer_payloads[richest], elo_mode)
    members: list[Member] = []
    for label, kind, payload, info in sources:
        try:
            predictor = _member_module(kind).from_payload(payload, featurizer)
        except ValueError as exc:
            raise ValueError(f"member {label!r} did not load: {exc}") from exc
        members.append(Member(label, kind, predictor, payload, extras[label], info))
    predictor = EnsemblePredictor(members, featurizer, weights=weights, name=name)
    records = {label: _dataset_record(extras_of[label]) for label in extras_of}
    tuned: dict[str, dict[str, Any]] = {}
    for label in predictor.labels:
        found = _finetune_record(label, extras_of[label])
        if found is not None:
            tuned[label] = found
    formats = {
        label: sorted(str(item) for item in (record.get("formats") or ()))
        for label, record in records.items()
    }
    if len({tuple(value) for value in formats.values()}) > 1:
        raise ValueError(
            f"the members were fitted on different formats: {formats}; a runtime "
            "serves an artifact in the formats of ONE dataset"
        )
    manifests = {
        str(record.get("manifest_sha256") or "") for record in records.values()
    }
    extra: dict[str, Any] = {
        "name": predictor.name,
        "kind": KIND,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "combine": COMBINE,
        "weights": dict(zip(predictor.labels, predictor.weights)),
        "elo_mode": predictor.elo_mode,
        "n_parameters": predictor.n_parameters,
        "featurizer_of": richest,
        "members": {
            member.label: {
                "kind": member.kind,
                "weight": weight,
                "extras": list(member.extras),
                **{
                    key: member.info.get(key)
                    for key in ("name", "tag", "sha256", "path")
                },
                "featurizer_sha256": payload_hash(featurizer_payloads[member.label]),
                "dataset": records[member.label],
                # own ladder games the member was fine-tuned on (0: never)
                "finetuned_on_own_games": len(
                    tuned.get(member.label, {}).get(KEY_SEEN) or ()
                ),
                "n_parameters": (extras_of[member.label] or {}).get("n_parameters"),
                "temperatures": (predictor._member_described(member) or {}).get(
                    "temperatures"
                ),
            }
            for weight, member in zip(predictor.weights, predictor.members)
        },
    }
    first = records[predictor.labels[0]]
    if tuned:
        # The ensemble has seen what any member has seen: the record of a
        # fine-tuned artifact, in the keys its readers look at (module text).
        seen = sorted({game for record in tuned.values() for game in record[KEY_SEEN]})
        fine = list(tuned)
        builds = {_build_of(records[label]) for label in predictor.labels}
        one_build = len(builds) == 1 and "" not in builds
        build = next(iter(builds)) if one_build else SEVERAL_BUILDS
        kept = ("path", "tag", "formats") if one_build else ("formats",)
        dataset: dict[str, Any] = {key: first[key] for key in kept if key in first}
        dataset["manifest_sha256"] = f"{MANIFEST_MARK}:{len(seen)}-games:{build}"
        dataset["source_manifest_sha256"] = build if one_build else None
        dataset["fitted_on"] = [
            "train",
            f"ladder_holdout: {len(seen)} own games (members {', '.join(fine)})",
        ]
        extra["dataset"] = dataset
        extra["dataset_tag"] = (
            f"{first.get('tag') if one_build else SEVERAL_BUILDS} + {len(seen)} own "
            f"ladder games (members {', '.join(fine)} were fine-tuned on them: any "
            "reading on them is in-sample)"
        )
        extra[KEY_FINETUNE] = {
            "of": KIND,
            "in_sample": f"Members {', '.join(fine)} of this ensemble were "
            "fine-tuned on own ladder games (each member's own record is under "
            f"members). {KEY_SEEN} is every game any of them was fitted on: a "
            "reading of the ensemble on one of them is in-sample.",
            KEY_SEEN: seen,
            "finetuned_members": fine,
            "members": tuned,
        }
    elif len(manifests) == 1 and "" not in manifests:
        # One dataset: the record a fit script reads (its manifest, its formats).
        extra["dataset"] = {
            key: first[key]
            for key in ("path", "tag", "manifest_sha256", "formats")
            if key in first
        }
    elif first.get("formats"):
        extra["formats"] = list(first["formats"])
    return Built(predictor, featurizer, extra)


def from_artifacts(
    sources: Sequence[tuple[str, Path | str]],
    *,
    weights: Sequence[float] | None = None,
    name: str = KIND,
    elo_mode: str = ELO_KEEP,
    strict: bool = True,
) -> Built:
    """An ensemble of stored artifacts: (label, path) per member, in order.

    Each member's predictor payload is taken from its file unchanged. Raises
    ``ValueError`` (``OSError`` for a file) when a member cannot be read, is
    itself an ensemble, carries a pair coupling, is given twice, or when the
    members' featurizers are not one (``shared_featurizer``). ``strict``
    (the default) also refuses a member written under another dex.
    """
    from vgc_bench.src.oppmodel.artifact import KEY_COUPLING, read_artifact

    rows: list[tuple[str, str, Any, Mapping[str, Any]]] = []
    payloads: dict[str, Mapping[str, Any]] = {}
    extras_of: dict[str, Mapping[str, Any]] = {}
    seen: dict[str, str] = {}
    for raw, path in sources:
        label = _label(raw)
        if label in payloads:
            raise ValueError(f"member label {label!r} is given twice")
        document = read_artifact(path, strict=strict)
        kind = str(document["kind"])
        if kind not in MEMBER_KINDS:
            raise ValueError(
                f"member {label!r} ({path}) has kind {kind!r}; a member is one "
                f"of {MEMBER_KINDS} (an ensemble is not nested)"
            )
        if document.get(KEY_COUPLING) is not None:
            raise ValueError(
                f"member {label!r} ({path}) carries a pair coupling: a coupling "
                "belongs to one predictor state and is not carried into an "
                "ensemble. Use the member's artifact from before its coupling, "
                "and fit a coupling on the ensemble afterwards "
                "(training/fit_oppmodel_coupling.py)"
            )
        digest = sha256_file(path)
        if digest in seen:
            raise ValueError(
                f"members {seen[digest]!r} and {label!r} are the same file ({path})"
            )
        seen[digest] = label
        extra = document.get("extra")
        extra = extra if isinstance(extra, Mapping) else {}
        info = {
            "name": str(document["name"]),
            "tag": _tag(extra, str(document["name"])),
            "sha256": digest,
            "path": str(path),
        }
        rows.append((label, kind, document["predictor"], info))
        payloads[label] = document["featurizer"]
        extras_of[label] = extra
    return _assemble(rows, payloads, extras_of, weights, name, elo_mode)


def from_loaded(
    loaded: Mapping[str, Any],
    *,
    weights: Sequence[float] | None = None,
    name: str = KIND,
    elo_mode: str = ELO_KEEP,
) -> Built:
    """An ensemble of predictors already in memory: label ->
    ``artifact.LoadedPredictor`` (or anything with ``predictor``,
    ``featurizer``, ``kind``, ``name``, ``meta`` and ``coupling``).

    The members are REBUILT on the shared featurizer from their own payloads
    (``predictor.to_payload()``); the same refusals as ``from_artifacts``.
    """
    rows: list[tuple[str, str, Any, Mapping[str, Any]]] = []
    payloads: dict[str, Mapping[str, Any]] = {}
    extras_of: dict[str, Mapping[str, Any]] = {}
    for raw, held in loaded.items():
        label = _label(raw)
        if getattr(held, "coupling", None) is not None:
            raise ValueError(
                f"member {label!r} carries a pair coupling: a coupling belongs "
                "to one predictor state and is not carried into an ensemble"
            )
        kind = str(held.kind)
        if kind not in MEMBER_KINDS:
            raise ValueError(
                f"member {label!r} has kind {kind!r}; a member is one of "
                f"{MEMBER_KINDS} (an ensemble is not nested)"
            )
        meta = getattr(held, "meta", None) or {}
        extra = meta.get("extra") if isinstance(meta, Mapping) else None
        extra = extra if isinstance(extra, Mapping) else {}
        info = {
            "name": str(held.name),
            "tag": _tag(extra, str(held.name)),
            "sha256": "",
            "path": str(meta.get("path") or "") if isinstance(meta, Mapping) else "",
        }
        rows.append((label, kind, held.predictor.to_payload(), info))
        payloads[label] = held.featurizer.to_payload()
        extras_of[label] = extra
    return _assemble(rows, payloads, extras_of, weights, name, elo_mode)
