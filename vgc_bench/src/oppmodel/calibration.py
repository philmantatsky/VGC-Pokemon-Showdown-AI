"""Event calibration of the opponent predictor (OPPONENT_PREDICTOR.md, R3).

The neural predictor's P(the slot switches out) and P(the slot uses a
Protect-family move) are over-confident where they are high, and one action
temperature cannot repair that. This module calibrates the two EVENTS inside
the action distribution, so that the ranked actions, the scalars a guard reads
(``features.event_probs``) and every score stay one coherent distribution.

A slot's action distribution is ``[candidates | OTHER | six switch pointers]``.

1. ``s`` = share of the legal mass on the switch pointers. It becomes
   ``s' = g_switch(s)``; the switch pointers are scaled by ``s' / s`` and every
   other action by ``(1 - s') / (1 - s)``.
2. ``r`` = share of the NON-switch mass on the Protect-family candidates (the
   valid candidates carrying ``CAND_PROTECT``). It becomes ``r' = g_protect(r)``;
   those candidates are scaled by ``r' / r`` and the other moves (OTHER
   included) by ``(1 - r') / (1 - r)``.

A group that cannot happen is left alone: step 1 is skipped for a slot with no
legal switch pointer, step 2 for a slot with no Protect-family candidate (a
slot with neither comes back bit for bit). The row total is unchanged, illegal
entries stay exactly as they were, and the destination of a switch, the choice
among the other moves, the targets and the Mega head are untouched.

``g`` is an ``EventMap``: monotone, ``g(0) = 0``, ``g(1) = 1``. Three kinds:

* ``identity``;
* ``logistic``: ``g(p) = sigmoid(slope * logit(p) + bias)``, two parameters,
  plus an optional additive logit TERM per public context flag of the slot
  (``CONTEXT``: turn 1, the Pokemon's first turn on the field, it protected
  last turn; the three flags the count table conditions on);
* ``isotonic``: a monotone piecewise-linear map through fitted knots.

Why the terms exist. One map over all slot-turns is right on average and
wrong for a part of them: after a two-parameter fit P(switch) still read 8-10
points too high on turn 1 on held-out players (review of 2026-10-05). The
flags are features the predictor is given anyway (``event_context`` reads
``turn``, ``act_mon`` and ``mon_flag``, never a label), so a map with terms is
still a function of the public state alone. With a term the map stays
monotone in the share and keeps 0 and 1 for every context. A map with terms
needs the context to be applied: without it ``EventMap.__call__`` and
``apply_checked`` raise, they never fall back to the map without its terms.

Fitting (offline, numpy only). ``fit_event_calibration`` takes the predictor's
action probabilities and the batch WITH its labels, and reads only the rows it
is given. The two maps are the maximum-likelihood fit of the three-way split
"switch / Protect-family / another move", which separates into two event log
losses that do not interact:

* ``g_switch`` on the slot-turns where the log tells whether the slot switched
  (``y_flag`` switch known, hidden actions included) and a switch is legal:
  ``-log s'`` for a switch, ``-log(1 - s')`` otherwise;
* ``g_protect`` on the slot-turns where the log tells whether the slot used a
  Protect-family move, the slot has such a candidate, and the slot is known not
  to have switched: ``-log r'`` for a Protect, ``-log(1 - r')`` otherwise. (A
  slot that switched says nothing about the split of the non-switch mass.)

Both sets include the slot-turns whose action is hidden (a Pokemon knocked out
before it moved, a flinch): the log proves they were no switch and, where it
says so, no Protect. So the calibrated numbers are right over ALL slot-turns
at the start of a turn, which is where a guard reads them. They are not the
rate among the slots that get to act: a hidden action is never the event, so
among visible actions alone both events happen more often than predicted.

``select_map`` prefers the simpler map: the logistic one unless it is not
better than no map at all; a context term only when adding it is better out
of sample by more than ``TERM_MARGIN`` and in most folds, one term at a time
(a flag that adds nothing stays out, whatever another one gains); and the
isotonic one only when it is clearly better than that choice (grouped
cross-validation inside the rows given).

    calibration = fit_event_calibration(pred["action"], batch, rows=mask)
    context = event_context(batch) if calibration.needs_context else None
    action = calibration.apply(pred["action"], batch["action_mask"],
                               batch["cand_flag"], context)
    stored = calibration.to_payload()
    again = EventCalibration.from_payload(stored)

Applying a calibration is NOT idempotent (``g(g(p)) != g(p)``), so it has one
home: ``model.OppNetPredictor`` holds at most one and applies it once, inside
``predict``. Nothing else should call ``apply`` on a prediction it got from a
predictor.

Runtime contract: ``EventCalibration.apply`` never raises (it returns its
input and counts under ``COUNTERS`` when the arrays do not fit), and neither
does ``EventMap.__call__`` of a map without context terms. ``apply_checked``,
``event_context``, a map with terms called without a fitting context, the
constructors, ``from_payload`` and the fitting helpers raise ``ValueError`` /
``KeyError`` on malformed input: the predictor counts that as a failed
prediction.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, NamedTuple

import numpy as np

from vgc_bench.src.oppmodel.events import INTENT_PROTECT, INTENT_SWITCH
from vgc_bench.src.oppmodel.features import (
    CAND_PROTECT,
    CAND_VALID,
    FLAG_FIRST_TURN,
    FLAG_PROTECTED_LAST,
    N_ROSTER,
    N_SLOT,
    PROBABILITY_FLOOR,
    Y_PROTECT_KNOWN,
    Y_PROTECTED,
    Y_SWITCH_KNOWN,
    Y_SWITCHED,
)

FORMAT = "oppmodel-event-calibration"
VERSION = 1
# A calibration whose maps carry context terms is written as version 2, so a
# reader that knows only version 1 refuses it instead of dropping the terms.
VERSION_CONTEXT = 2
VERSIONS: tuple[int, ...] = (VERSION, VERSION_CONTEXT)
MAP_IDENTITY = "identity"
MAP_LOGISTIC = "logistic"
MAP_ISOTONIC = "isotonic"
MAP_KINDS: tuple[str, ...] = (MAP_IDENTITY, MAP_LOGISTIC, MAP_ISOTONIC)
# Not a kind: the name, in the selection records, of the logistic map WITH
# context terms (its ``kind`` is ``logistic``).
MAP_LOGISTIC_CONTEXT = "logistic_context"
EVENT_SWITCH = INTENT_SWITCH
EVENT_PROTECT = INTENT_PROTECT
EVENTS: tuple[str, ...] = (EVENT_SWITCH, EVENT_PROTECT)

# Public context flags of a slot at the start of a turn: the three the count
# table conditions on (OPPONENT_PREDICTOR.md, reconnaissance point 2). A
# logistic map may carry one additive logit term for each.
CONTEXT_TURN_ONE = "turn_one"
CONTEXT_FIRST_TURN = "first_turn_on_field"
CONTEXT_PROTECTED_LAST = "protected_last_turn"
CONTEXT: tuple[str, ...] = (
    CONTEXT_TURN_ONE,
    CONTEXT_FIRST_TURN,
    CONTEXT_PROTECTED_LAST,
)
# The feature arrays ``event_context`` reads.
CONTEXT_ARRAYS: tuple[str, ...] = ("turn", "act_mon", "mon_flag")

LOGIT_EPS = 1.0e-12  # shares are clipped to [eps, 1 - eps] before their logit
TEMPERATURE_TOLERANCE = 1.0e-9
MIN_EACH = 25  # fitted rows needed of each outcome; fewer gives the identity
FOLDS = 10
SEED = 20261005
# The isotonic map replaces the simpler one only when its cross-validated event
# NLL is lower by more than this (nats per fitted slot-turn) AND it is the
# better one in at least this share of the folds.
ISOTONIC_MARGIN = 5.0e-4
ISOTONIC_FOLD_SHARE = 0.8
ISOTONIC_MAX_BINS = 40
ISOTONIC_MIN_BIN = 200
ISOTONIC_PRIOR = 1.0  # pseudo-rows per bin pulling its rate to the identity
# A context term is taken only when adding it lowers the cross-validated
# event NLL by more than this (nats per fitted slot-turn) AND the map with it
# is the better one in at least ``ISOTONIC_FOLD_SHARE`` of the folds: the rule
# of the isotonic map, with the same two numbers, applied to one term at a
# time.
TERM_MARGIN = ISOTONIC_MARGIN
NEWTON_STEPS = 100

# Failures of ``EventCalibration.apply``, by exception type.
COUNTERS: Counter[str] = Counter()


def _failed(name: str, exc: Exception) -> None:
    try:
        COUNTERS[f"{name}:{type(exc).__name__}"] += 1
    except Exception:
        pass


def _sigmoid(value: np.ndarray) -> np.ndarray:
    """The logistic function without overflow at either end."""
    with np.errstate(over="ignore", under="ignore"):
        grown = np.exp(-np.abs(value))
    return np.where(value >= 0, 1.0 / (1.0 + grown), grown / (1.0 + grown))


def _logit(share: np.ndarray) -> np.ndarray:
    clipped = np.clip(share, LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(clipped) - np.log1p(-clipped)


def _plain(value: Any, where: str = "info") -> Any:
    """A deep copy as plain data: dicts with string keys, lists, scalars.

    Numpy scalars and arrays become Python numbers and lists, a tuple a list,
    a non-finite number None. Raises ``ValueError`` for anything else, so what
    is stored reads back equal.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.generic):
        return _plain(value.item(), where)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _plain(value.tolist(), where)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{where}: the key {key!r} is not a string")
            out[key] = _plain(item, f"{where}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [_plain(item, where) for item in value]
    raise ValueError(f"{where}: cannot store a {type(value).__name__}")


# --- the maps -----------------------------------------------------------------


@dataclass(frozen=True)
class EventMap:
    """A monotone map of a probability share with ``g(0) = 0`` and ``g(1) = 1``.

    ``kind`` is ``identity``, ``logistic`` (``slope`` > 0 and ``bias``) or
    ``isotonic`` (``knots_x`` strictly increasing from 0 to 1, ``knots_y``
    non-decreasing from 0 to 1; linear in between). A logistic map may carry
    ``terms``: pairs (a name of ``CONTEXT``, a finite number), given as pairs
    or as a mapping; each is added to the logit of the slots whose context
    flag is set, so the map is ``sigmoid(slope * logit(p) + bias + sum of the
    terms of the slot's flags)``. Fields the kind does not use are reset, a
    term of exactly 0 is dropped and the rest is kept in the order of
    ``CONTEXT``, so two equal maps compare equal. Raises ``ValueError`` when
    the parameters do not describe such a map.
    """

    kind: str = MAP_IDENTITY
    slope: float = 1.0
    bias: float = 0.0
    knots_x: tuple[float, ...] = ()
    knots_y: tuple[float, ...] = ()
    terms: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        kind = self.kind
        if kind not in MAP_KINDS:
            raise ValueError(f"unknown map {kind!r}; expected one of {MAP_KINDS}")
        try:
            slope, bias = float(self.slope), float(self.bias)
            xs = tuple(float(value) for value in self.knots_x)
            ys = tuple(float(value) for value in self.knots_y)
            given = self.terms
            pairs = list(given.items() if isinstance(given, Mapping) else given)
            found = {str(name): float(value) for name, value in pairs}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"map parameters are not numbers: {exc!r}") from exc
        terms: tuple[tuple[str, float], ...] = ()
        if kind == MAP_LOGISTIC:
            if not (math.isfinite(slope) and math.isfinite(bias) and slope > 0.0):
                raise ValueError("a logistic map needs a finite slope > 0 and bias")
            if len(found) != len(pairs) or any(name not in CONTEXT for name in found):
                raise ValueError(
                    f"context terms {sorted(found)} must be distinct names of {CONTEXT}"
                )
            if not all(math.isfinite(value) for value in found.values()):
                raise ValueError("a context term is not finite")
            terms = tuple(
                (name, found[name]) for name in CONTEXT if found.get(name, 0.0) != 0.0
            )
            xs, ys = (), ()
        elif kind == MAP_ISOTONIC:
            if len(xs) < 2 or len(xs) != len(ys):
                raise ValueError("an isotonic map needs matching knots, two or more")
            if not all(math.isfinite(value) for value in xs + ys):
                raise ValueError("an isotonic knot is not finite")
            if xs[0] != 0.0 or xs[-1] != 1.0 or ys[0] != 0.0 or ys[-1] != 1.0:
                raise ValueError("isotonic knots must run from (0, 0) to (1, 1)")
            if any(right <= left for left, right in zip(xs, xs[1:])):
                raise ValueError("isotonic knots_x must be strictly increasing")
            if any(right < left for left, right in zip(ys, ys[1:])):
                raise ValueError("isotonic knots_y must not decrease")
            slope, bias = 1.0, 0.0
        else:
            slope, bias, xs, ys = 1.0, 0.0, (), ()
        object.__setattr__(self, "slope", slope)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "knots_x", xs)
        object.__setattr__(self, "knots_y", ys)
        object.__setattr__(self, "terms", terms)

    @property
    def is_identity(self) -> bool:
        return self.kind == MAP_IDENTITY

    @property
    def needs_context(self) -> bool:
        """Whether the map carries context terms (it then needs the flags)."""
        return bool(self.terms)

    def offset(self, context: Any, shape: tuple[int, ...]) -> np.ndarray | float:
        """The sum of this map's terms for slots with ``context`` flags.

        ``context`` is ``[*shape, len(CONTEXT)]`` (``event_context``). 0.0 for
        a map without terms, whatever is given. Raises ``ValueError`` when the
        map has terms and the context is missing or of another shape.
        """
        if not self.terms:
            return 0.0
        if context is None:
            raise ValueError(
                f"the map has context terms {[name for name, _ in self.terms]}: "
                "it cannot be applied without the context flags"
            )
        flags = np.asarray(context)
        if flags.shape != (*shape, len(CONTEXT)):
            raise ValueError(
                f"context {flags.shape} does not fit shares {shape}: "
                f"{(*shape, len(CONTEXT))} is needed"
            )
        total = np.zeros(shape, dtype=np.float64)
        for name, value in self.terms:
            total = total + value * (flags[..., CONTEXT.index(name)] != 0)
        return total

    def __call__(self, share: Any, context: Any = None) -> np.ndarray:
        """The mapped share(s), float64. 0 stays 0, 1 stays 1.

        ``context`` (``event_context``: ``[*share.shape, len(CONTEXT)]``) is
        read only by a map with terms. A map without terms never raises for
        an array of numbers (a non-finite entry comes back as it is); a map
        with terms raises ``ValueError`` when the context is missing or does
        not fit, rather than answer without its terms.
        """
        p = np.asarray(share, dtype=np.float64)
        if self.kind == MAP_IDENTITY:
            return p.copy()
        with np.errstate(invalid="ignore"):
            if self.kind == MAP_LOGISTIC:
                shift = self.offset(context, p.shape)
                out = _sigmoid(self.slope * _logit(p) + self.bias + shift)
            else:
                out = np.interp(p, np.asarray(self.knots_x), np.asarray(self.knots_y))
            out = np.where(p <= 0.0, 0.0, np.where(p >= 1.0, 1.0, out))
        return np.where(np.isfinite(p), out, p)

    def to_payload(self) -> dict[str, Any]:
        return {
            "map": self.kind,
            "slope": self.slope,
            "bias": self.bias,
            "x": list(self.knots_x),
            "y": list(self.knots_y),
            "terms": dict(self.terms),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EventMap":
        """Rebuild from ``to_payload()``. Raises ``ValueError`` on a bad one."""
        try:
            return cls(
                str(payload["map"]),
                payload.get("slope", 1.0),
                payload.get("bias", 0.0),
                tuple(payload.get("x") or ()),
                tuple(payload.get("y") or ()),
                # Absent in a payload written before context terms existed.
                tuple(dict(payload.get("terms") or {}).items()),
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"not an event map: {exc!r}") from exc


class EventShares(NamedTuple):
    """The two event shares of a batch of action distributions, each ``[N, 2]``.

    ``switch``: share of the legal mass on the switch pointers (the event
    probability ``features.event_probs`` reports). ``protect_share``: share of
    the non-switch mass on the Protect-family candidates. ``protect``: their
    product with the non-switch share, the Protect event probability.
    ``*_possible``: the slot has a legal switch pointer / a valid legal
    Protect-family candidate (where False the share is 0 by the mask).
    """

    switch: np.ndarray
    protect_share: np.ndarray
    protect: np.ndarray
    switch_possible: np.ndarray
    protect_possible: np.ndarray


class _Groups(NamedTuple):
    weight: np.ndarray  # legal, non-negative mass, float64
    switch: np.ndarray  # legal switch pointers
    guard: np.ndarray  # legal valid Protect-family candidates
    rest: np.ndarray  # every other legal move, OTHER included


def _groups(action: Any, action_mask: Any, cand_flag: Any) -> _Groups:
    mass = np.asarray(action, dtype=np.float64)
    legal = np.asarray(action_mask).astype(bool)
    flags = np.asarray(cand_flag).astype(np.int64)
    if mass.ndim < 1 or mass.shape != legal.shape:
        raise ValueError(f"action {mass.shape} does not match mask {legal.shape}")
    n_cand = flags.shape[-1] if flags.ndim else -1
    if flags.shape[:-1] != mass.shape[:-1] or mass.shape[-1] != n_cand + 1 + N_ROSTER:
        raise ValueError(
            f"candidate flags {flags.shape} do not match action {mass.shape}"
        )
    if not np.isfinite(mass).all():
        raise ValueError("action holds a non-finite probability")
    switch = legal.copy()
    switch[..., : n_cand + 1] = False
    guard = np.zeros_like(legal)
    guard[..., :n_cand] = (
        legal[..., :n_cand] & ((flags & CAND_VALID) > 0) & ((flags & CAND_PROTECT) > 0)
    )
    rest = legal & ~switch & ~guard
    return _Groups(np.where(legal, np.maximum(mass, 0.0), 0.0), switch, guard, rest)


def _share(part: np.ndarray, whole: np.ndarray) -> np.ndarray:
    return np.divide(part, whole, out=np.zeros_like(part), where=whole > 0)


def event_shares(action: Any, action_mask: Any, cand_flag: Any) -> EventShares:
    """The switch share and the Protect share of the non-switch mass.

    ``action`` is ``[..., A]`` over candidates, OTHER and six switch pointers;
    mass outside ``action_mask`` is ignored. Raises ``ValueError`` when the
    arrays do not fit each other.
    """
    groups = _groups(action, action_mask, cand_flag)
    switch = (groups.weight * groups.switch).sum(-1)
    guard = (groups.weight * groups.guard).sum(-1)
    rest = (groups.weight * groups.rest).sum(-1)
    total = switch + guard + rest
    share = _share(guard, guard + rest)
    return EventShares(
        _share(switch, total),
        share,
        _share(guard, total),
        groups.switch.any(-1),
        groups.guard.any(-1),
    )


def event_context(batch: Mapping[str, np.ndarray]) -> np.ndarray:
    """The public context flags of every slot: ``[..., 2, len(CONTEXT)]`` bool.

    In the order of ``CONTEXT``: the turn is turn 1; the Pokemon in the slot
    is on its first turn on the field; it used a Protect-family move last
    turn. All False for a slot with no Pokemon. Reads the FEATURE arrays
    ``turn`` ``[...]``, ``act_mon`` ``[..., 2]`` and ``mon_flag``
    ``[..., mons, flags]`` only (``CONTEXT_ARRAYS``): what a predictor is
    given, never a label. A batch or one example. Raises ``KeyError`` /
    ``ValueError`` when an array is missing or does not fit.
    """
    rows = np.asarray(batch["act_mon"]).astype(np.int64)
    turn = np.asarray(batch["turn"]).astype(np.int64)
    flags = np.asarray(batch["mon_flag"])
    lead = rows.shape[:-1]
    if rows.ndim < 1 or rows.shape[-1] != N_SLOT:
        raise ValueError(f"act_mon {rows.shape} is not [..., {N_SLOT}]")
    if turn.shape != lead or flags.ndim != len(lead) + 2 or flags.shape[:-2] != lead:
        raise ValueError(
            f"turn {turn.shape} / mon_flag {flags.shape} do not fit act_mon "
            f"{rows.shape}"
        )
    n_mon, width = flags.shape[-2:]
    if n_mon < 1 or width <= max(FLAG_FIRST_TURN, FLAG_PROTECTED_LAST):
        raise ValueError(f"mon_flag {flags.shape} holds no first-turn / protect flag")
    present = (rows >= 0) & (rows < n_mon)
    own = np.take_along_axis(flags, np.clip(rows, 0, n_mon - 1)[..., None], axis=-2)
    out = np.zeros((*rows.shape, len(CONTEXT)), dtype=bool)
    out[..., CONTEXT.index(CONTEXT_TURN_ONE)] = present & (turn[..., None] == 1)
    out[..., CONTEXT.index(CONTEXT_FIRST_TURN)] = present & (
        own[..., FLAG_FIRST_TURN] > 0
    )
    out[..., CONTEXT.index(CONTEXT_PROTECTED_LAST)] = present & (
        own[..., FLAG_PROTECTED_LAST] > 0
    )
    return out


def rescale(
    action: Any,
    action_mask: Any,
    cand_flag: Any,
    switch_map: EventMap,
    protect_map: EventMap,
    context: Any = None,
) -> np.ndarray:
    """The action distribution with both events mapped (see the module text).

    Returns float64 of the shape of ``action``. A group that holds no legal
    entry, or no mass, or all of the mass, is left as it is (``g`` keeps 0 and
    1 anyway). ``context`` is ``event_context`` of the same slots; it is read
    only by a map with context terms. Raises ``ValueError`` when the arrays do
    not fit each other, or a map with terms gets no fitting context.
    """
    groups = _groups(action, action_mask, cand_flag)
    mass = np.asarray(action, dtype=np.float64)
    switch = (groups.weight * groups.switch).sum(-1)
    guard = (groups.weight * groups.guard).sum(-1)
    rest = (groups.weight * groups.rest).sum(-1)
    moves = guard + rest
    total = switch + moves
    ones = np.ones_like(total)
    # A step runs only where both of its sides hold mass. Every map keeps 0
    # and 1, so the arithmetic would leave the other slots alone as well; the
    # conditions make that hold by construction (their factors are exactly 1).

    can_switch = (switch > 0) & (moves > 0)
    old = _share(switch, total)
    new = np.where(can_switch, switch_map(old, context), old)
    switch_factor = np.where(can_switch, _share(new, old), ones)
    move_factor = np.where(can_switch, _share(1.0 - new, _share(moves, total)), ones)

    can_guard = (guard > 0) & (rest > 0)
    old = _share(guard, moves)
    new = np.where(can_guard, protect_map(old, context), old)
    guard_factor = np.where(can_guard, _share(new, old), ones)
    rest_factor = np.where(can_guard, _share(1.0 - new, _share(rest, moves)), ones)

    factor = np.ones_like(mass)
    factor = np.where(groups.switch, switch_factor[..., None], factor)
    factor = np.where(groups.guard, (move_factor * guard_factor)[..., None], factor)
    factor = np.where(groups.rest, (move_factor * rest_factor)[..., None], factor)
    out = mass * factor
    if not np.isfinite(out).all():
        raise ValueError("the rescaled action holds a non-finite probability")
    return out


@dataclass(frozen=True)
class EventCalibration:
    """The two event maps of one predictor, and how they were fitted.

    ``switch`` maps the switch share, ``protect`` the Protect-family share of
    the non-switch mass. ``action_temperature`` is the action temperature the
    maps were fitted after (None = not recorded): a predictor with another
    temperature must not carry them. ``info`` is plain data for the record
    (dataset tag, validation event rates, NLL before and after); it does not
    change what ``apply`` does. Raises ``ValueError`` on malformed fields.
    """

    switch: EventMap = field(default_factory=EventMap)
    protect: EventMap = field(default_factory=EventMap)
    action_temperature: float | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.switch, EventMap) or not isinstance(
            self.protect, EventMap
        ):
            raise ValueError("an event calibration needs two EventMap objects")
        temperature = self.action_temperature
        if temperature is not None:
            try:
                temperature = float(temperature)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"action temperature {temperature!r}") from exc
            if not (math.isfinite(temperature) and temperature > 0.0):
                raise ValueError(f"action temperature {temperature!r}")
        if not isinstance(self.info, Mapping):
            raise ValueError("info must be a mapping of plain data")
        object.__setattr__(self, "action_temperature", temperature)
        object.__setattr__(self, "info", _plain(self.info))

    @property
    def maps(self) -> dict[str, EventMap]:
        """The map of each event, keyed by the names in ``EVENTS``."""
        return {EVENT_SWITCH: self.switch, EVENT_PROTECT: self.protect}

    @property
    def is_identity(self) -> bool:
        """Whether applying it changes nothing."""
        return self.switch.is_identity and self.protect.is_identity

    @property
    def needs_context(self) -> bool:
        """Whether a map carries context terms: ``apply`` then needs
        ``event_context`` of the batch."""
        return self.switch.needs_context or self.protect.needs_context

    def fits_temperature(self, action_temperature: float) -> bool:
        """Whether the maps were fitted after this action temperature."""
        mine = self.action_temperature
        if mine is None:
            return True
        try:
            return math.isclose(
                mine,
                float(action_temperature),
                rel_tol=TEMPERATURE_TOLERANCE,
                abs_tol=TEMPERATURE_TOLERANCE,
            )
        except (TypeError, ValueError):
            return False

    def apply_checked(
        self, action: Any, action_mask: Any, cand_flag: Any, context: Any = None
    ) -> np.ndarray:
        """``rescale`` with this calibration's maps. Raises ``ValueError``.

        ``context`` is ``event_context`` of the same batch; it is needed when
        ``needs_context`` (a missing one is an error, the terms are never
        dropped) and ignored otherwise.
        """
        if self.is_identity:
            return np.array(action, copy=True)
        return rescale(
            action, action_mask, cand_flag, self.switch, self.protect, context
        )

    def apply(
        self, action: Any, action_mask: Any, cand_flag: Any, context: Any = None
    ) -> np.ndarray:
        """The calibrated action distribution. Never raises.

        Arrays that do not fit (a missing context included) come back
        unchanged, counted in ``COUNTERS``; a caller that must know uses
        ``apply_checked``.
        """
        try:
            return self.apply_checked(action, action_mask, cand_flag, context)
        except Exception as exc:
            _failed("apply", exc)
            return action

    def to_payload(self) -> dict[str, Any]:
        """Plain data from which ``from_payload`` rebuilds an equal object."""
        return {
            "format": FORMAT,
            "version": VERSION_CONTEXT if self.needs_context else VERSION,
            "action_temperature": self.action_temperature,
            "maps": {name: found.to_payload() for name, found in self.maps.items()},
            "info": _plain(self.info),
        }

    def describe(self) -> dict[str, Any]:
        """The payload plus ``applied`` and the names of the mapped events."""
        out = self.to_payload()
        out["applied"] = not self.is_identity
        out["events"] = [name for name, m in self.maps.items() if not m.is_identity]
        return out

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EventCalibration":
        """Rebuild from ``to_payload()``. Raises ``ValueError`` on a bad one."""
        try:
            if payload["format"] != FORMAT:
                raise ValueError(f"calibration format {payload['format']!r}")
            version = int(payload["version"])
            if version not in VERSIONS:
                raise ValueError(f"calibration version {payload['version']!r}")
            maps = payload["maps"]
            made = cls(
                EventMap.from_payload(maps[EVENT_SWITCH]),
                EventMap.from_payload(maps[EVENT_PROTECT]),
                payload.get("action_temperature"),
                dict(payload.get("info") or {}),
            )
            if made.needs_context != (version == VERSION_CONTEXT):
                raise ValueError(
                    f"calibration version {version} does not go with maps that "
                    f"{'carry' if made.needs_context else 'carry no'} context terms"
                )
            return made
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"not an event calibration: {exc!r}") from exc


# --- labels and scores (offline) ----------------------------------------------


def event_labels(
    batch: Mapping[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """For each event of ``EVENTS``: (the log tells, it happened), ``[N, 2]``.

    Known wherever the slot holds a Pokemon and ``y_flag`` says so, hidden
    actions included: the rows the scorecard counts. Reads ``act_mon`` and
    ``y_flag`` only. Raises ``KeyError`` on a batch without labels.
    """
    active = np.asarray(batch["act_mon"]) >= 0
    flags = np.asarray(batch["y_flag"])
    return {
        EVENT_SWITCH: (
            active & (flags[..., Y_SWITCH_KNOWN] == 1),
            flags[..., Y_SWITCHED] == 1,
        ),
        EVENT_PROTECT: (
            active & (flags[..., Y_PROTECT_KNOWN] == 1),
            flags[..., Y_PROTECTED] == 1,
        ),
    }


def _rows(
    share: Any, outcome: Any, weight: Any = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(share, dtype=np.float64).reshape(-1)
    y = np.asarray(outcome).reshape(-1).astype(np.float64)
    if p.shape != y.shape:
        raise ValueError(f"{p.size} shares for {y.size} outcomes")
    w = (
        np.ones_like(p)
        if weight is None
        else np.asarray(weight, dtype=np.float64).reshape(-1)
    )
    if w.shape != p.shape:
        raise ValueError(f"{w.size} weights for {p.size} shares")
    if not (np.isfinite(p).all() and np.isfinite(w).all() and (w >= 0).all()):
        raise ValueError("shares and weights must be finite, weights not negative")
    if ((y != 0) & (y != 1)).any():
        raise ValueError("outcomes must be 0 or 1")
    return p, y, w


def binary_nll(
    share: Any, outcome: Any, weight: Any = None, floor: float = PROBABILITY_FLOOR
) -> float:
    """Mean event log loss, the probability clipped to ``[floor, 1 - floor]``.

    NaN when there is nothing to score.
    """
    p, y, w = _rows(share, outcome, weight)
    total = float(w.sum())
    if not total > 0:
        return float("nan")
    q = np.clip(p, floor, 1.0 - floor)
    loss = -(y * np.log(q) + (1.0 - y) * np.log1p(-q))
    return float((w * loss).sum() / total)


def _enough(y: np.ndarray, w: np.ndarray, min_each: float) -> bool:
    return bool(min(float((w * y).sum()), float((w * (1.0 - y)).sum())) >= min_each)


def _context_rows(context: Any, n: int) -> np.ndarray:
    """``context`` as ``[n, len(CONTEXT)]`` bool. Raises ``ValueError``."""
    flags = np.asarray(context)
    if (
        flags.ndim < 1
        or flags.shape[-1] != len(CONTEXT)
        or flags.size != n * len(CONTEXT)
    ):
        raise ValueError(
            f"context {flags.shape} for {n} rows: one row of {len(CONTEXT)} flags "
            "each is needed"
        )
    return flags.reshape(n, len(CONTEXT)) != 0


def fit_logistic(
    share: Any,
    outcome: Any,
    weight: Any = None,
    *,
    min_each: float = MIN_EACH,
    context: Any = None,
) -> EventMap:
    """The logistic map minimising the event log loss of ``g(share)``.

    Newton's method on the two parameters (the loss is convex in them), each
    step halved until it does not increase the loss. Returns the identity
    when either outcome has fewer than ``min_each`` (weighted) rows, or the
    fitted slope is not positive (the share is not informative), or the fit
    is not finite.

    With ``context`` (one row of ``len(CONTEXT)`` flags per share) the map
    also gets one additive logit term per usable flag, all parameters fitted
    together from the two-parameter fit as the start. A flag is usable when
    the rows that carry it and the rows that do not each hold ``min_each``
    rows of both outcomes; the others get no term. When no flag is usable, or
    the joint fit is not a valid map, the two-parameter map is returned.
    """
    plain = _fit_logistic(share, outcome, weight, min_each)
    if context is None:
        return plain
    p, y, w = _rows(share, outcome, weight)
    flags = _context_rows(context, p.size)
    if not _enough(y, w, min_each):
        return plain
    usable = [
        column
        for column in range(len(CONTEXT))
        if _enough(y[flags[:, column]], w[flags[:, column]], min_each)
        and _enough(y[~flags[:, column]], w[~flags[:, column]], min_each)
    ]
    if not usable:
        return plain
    design = np.column_stack(
        [_logit(p), np.ones_like(p)] + [flags[:, c].astype(np.float64) for c in usable]
    )
    start = np.zeros(design.shape[1], dtype=np.float64)
    start[0], start[1] = plain.slope, plain.bias  # (1, 0) for the identity
    theta = _newton(design, y, w, start)
    if theta is None or not (np.isfinite(theta).all() and theta[0] > 0.0):
        return plain
    terms = tuple(
        (CONTEXT[column], float(theta[2 + i])) for i, column in enumerate(usable)
    )
    try:
        return EventMap(MAP_LOGISTIC, float(theta[0]), float(theta[1]), terms=terms)
    except ValueError:
        return plain


def _newton(
    design: np.ndarray, y: np.ndarray, w: np.ndarray, start: np.ndarray
) -> np.ndarray | None:
    """Minimise the weighted logistic loss of ``design @ theta`` from ``start``.

    Newton steps, each halved until the loss does not go up; None when the
    Hessian cannot be solved at the start. The loss is convex, so the result
    is never worse than ``start``.
    """
    total = float(w.sum())
    theta = np.asarray(start, dtype=np.float64).copy()

    def loss(at: np.ndarray) -> float:
        eta = design @ at
        return float((w * (np.logaddexp(0.0, eta) - y * eta)).sum() / total)

    best = loss(theta)
    if not math.isfinite(best):
        return None
    for index in range(NEWTON_STEPS):
        q = _sigmoid(design @ theta)
        grad = design.T @ (w * (q - y))
        hess = (design * (w * q * (1.0 - q))[:, None]).T @ design
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return None if index == 0 else theta
        if not np.isfinite(step).all():
            return None if index == 0 else theta
        scale = 1.0
        found: tuple[np.ndarray, float] | None = None
        while scale > 1e-8 and found is None:
            trial = theta - scale * step
            value = loss(trial)
            if math.isfinite(value) and value <= best:
                found = (trial, value)
            scale /= 2.0
        if found is None:
            break
        gain = best - found[1]
        theta, best = found
        if gain < 1e-13:
            break
    return theta


def _fit_logistic(share: Any, outcome: Any, weight: Any, min_each: float) -> EventMap:
    """The two-parameter logistic fit (see ``fit_logistic``)."""
    p, y, w = _rows(share, outcome, weight)
    if not _enough(y, w, min_each):
        return EventMap()
    z = _logit(p)
    total = float(w.sum())

    def loss(slope: float, bias: float) -> float:
        eta = slope * z + bias
        return float((w * (np.logaddexp(0.0, eta) - y * eta)).sum() / total)

    slope, bias = 1.0, 0.0
    best = loss(slope, bias)
    for _ in range(NEWTON_STEPS):
        q = _sigmoid(slope * z + bias)
        error = w * (q - y)
        grad_slope, grad_bias = float((error * z).sum()), float(error.sum())
        curve = w * q * (1.0 - q)
        h_ss = float((curve * z * z).sum())
        h_sb = float((curve * z).sum())
        h_bb = float(curve.sum())
        det = h_ss * h_bb - h_sb * h_sb
        if not (math.isfinite(det) and det > 1e-12 * max(1.0, h_ss * h_bb)):
            break
        step_slope = (h_bb * grad_slope - h_sb * grad_bias) / det
        step_bias = (h_ss * grad_bias - h_sb * grad_slope) / det
        scale = 1.0
        found: tuple[float, float, float] | None = None
        while scale > 1e-8 and found is None:
            trial = (slope - scale * step_slope, bias - scale * step_bias)
            value = loss(*trial)
            if math.isfinite(value) and value <= best:
                found = (trial[0], trial[1], value)
            scale /= 2.0
        if found is None:
            break
        gain = best - found[2]
        slope, bias, best = found
        if gain < 1e-13:
            break
    if not (math.isfinite(slope) and math.isfinite(bias) and slope > 0.0):
        return EventMap()
    if slope == 1.0 and bias == 0.0:
        return EventMap()
    return EventMap(MAP_LOGISTIC, slope, bias)


def _pool(levels: np.ndarray, weights: np.ndarray) -> tuple[list[float], list[int]]:
    """Pool adjacent violators: (block levels, bins per block), non-decreasing."""
    level: list[float] = []
    mass: list[float] = []
    count: list[int] = []
    for value, weight in zip(levels.tolist(), weights.tolist()):
        level.append(value)
        mass.append(weight)
        count.append(1)
        while len(level) > 1 and level[-2] >= level[-1]:
            joined = mass[-2] + mass[-1]
            merged = (level[-2] * mass[-2] + level[-1] * mass[-1]) / joined
            level[-2:] = [merged]
            mass[-2:] = [joined]
            count[-2:] = [count[-2] + count[-1]]
    return level, count


def fit_isotonic(
    share: Any,
    outcome: Any,
    weight: Any = None,
    *,
    max_bins: int = ISOTONIC_MAX_BINS,
    min_bin: int = ISOTONIC_MIN_BIN,
    prior: float = ISOTONIC_PRIOR,
    min_each: float = MIN_EACH,
) -> EventMap:
    """A monotone piecewise-linear map from binned event rates.

    The rows are sorted by share and cut into at most ``max_bins`` bins of
    equal count (``min_bin`` rows or more). Each bin's event rate is pulled
    to its mean share by ``prior`` pseudo-rows (so a bin without an event does
    not map a positive share to exactly 0), the rates are made non-decreasing
    by pooling adjacent violators, and the map runs linearly through
    ``(0, 0)``, one knot per pooled block (its mean share, its rate) and
    ``(1, 1)``. Returns the identity when either outcome has fewer than
    ``min_each`` rows or no knot is left.
    """
    p, y, w = _rows(share, outcome, weight)
    keep = w > 0
    p, y, w = p[keep], y[keep], w[keep]
    if not _enough(y, w, min_each):
        return EventMap()
    order = np.argsort(p, kind="stable")
    p, y, w = p[order], y[order], w[order]
    n_bins = int(min(max(1, int(max_bins)), max(1, p.size // max(1, int(min_bin)))))
    edges = np.unique(np.linspace(0, p.size, n_bins + 1).round().astype(np.int64))
    starts = edges[:-1]
    mass = np.add.reduceat(w, starts)
    mean = np.add.reduceat(w * p, starts) / mass
    rate = (np.add.reduceat(w * y, starts) + prior * mean) / (mass + prior)
    levels, counts = _pool(rate, mass + prior)
    xs: list[float] = [0.0]
    ys: list[float] = [0.0]
    start = 0
    for level, count in zip(levels, counts):
        block = slice(start, start + count)
        start += count
        centre = float((mean[block] * mass[block]).sum() / mass[block].sum())
        height = float(min(1.0, max(0.0, level)))
        if not (LOGIT_EPS < centre < 1.0 - LOGIT_EPS):
            continue
        if centre <= xs[-1] + LOGIT_EPS or height < ys[-1]:
            continue
        xs.append(centre)
        ys.append(height)
    if len(xs) == 1:
        return EventMap()
    xs.append(1.0)
    ys.append(1.0)
    return EventMap(MAP_ISOTONIC, knots_x=tuple(xs), knots_y=tuple(ys))


def _fold_index(groups: np.ndarray, folds: int, seed: int) -> tuple[np.ndarray, int]:
    """(fold of every row, number of folds): whole groups stay together."""
    _, inverse = np.unique(groups, return_inverse=True)
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


def _only(flags: np.ndarray, columns: list[int]) -> np.ndarray:
    """``flags`` with every column but ``columns`` cleared (a cleared flag is
    not usable, so ``fit_logistic`` gives it no term)."""
    out = np.zeros_like(flags)
    out[:, columns] = flags[:, columns]
    return out


def select_map(
    share: Any,
    outcome: Any,
    groups: Any = None,
    *,
    folds: int = FOLDS,
    seed: int = SEED,
    margin: float = ISOTONIC_MARGIN,
    fold_share: float = ISOTONIC_FOLD_SHARE,
    context: Any = None,
    term_margin: float = TERM_MARGIN,
) -> tuple[EventMap, dict[str, Any]]:
    """(the map to use, the record of how it was chosen) for one event.

    All three kinds are fitted on every row given. Their out-of-sample event
    NLL comes from a cross-validation inside those rows that keeps the rows
    of one group (a battle) in one fold. The choice prefers the simpler map:

    * ``logistic`` when its cross-validated NLL is below the identity's, else
      ``identity``;
    * with ``context`` (one row of ``len(CONTEXT)`` flags per share): context
      terms are added ONE AT A TIME. Each step tries every flag not yet taken
      (all parameters refitted together) and takes the one that lowers the
      cross-validated NLL most, if it beats the map so far by more than
      ``term_margin`` nats per row AND is the better one in at least
      ``fold_share`` of the folds; the first step that takes nothing ends
      it. So a term stays out when it does not earn its place, however much
      another one gains;
    * ``isotonic`` only when it beats the choice so far by more than
      ``margin`` nats per row AND is the better one in at least
      ``fold_share`` of the folds.

    With fewer than two groups there is no cross-validation: the logistic map
    without terms is taken when it is fitted at all. The record holds
    ``chosen`` (a kind), ``chosen_terms``, ``reason``, the in-sample and the
    cross-validated NLL of each kind, the folds and how many of them the
    isotonic map won, and under ``context`` (None without one): how many rows
    and events carry each flag; ``steps`` (per step what every flag tried
    gave, the best one and whether it was taken); ``chosen`` (the flags taken,
    in order); and for the map with the terms taken, or with the best single
    term when none was, ``fitted``, its NLL, ``gain_over_simpler`` and
    ``better_folds`` against the map without terms, ``accepted`` and why.
    """
    p, y, _ = _rows(share, outcome)
    flags = None if context is None else _context_rows(context, p.size)
    fitted = {
        MAP_IDENTITY: EventMap(),
        MAP_LOGISTIC: fit_logistic(p, y),
        MAP_ISOTONIC: fit_isotonic(p, y),
    }
    record: dict[str, Any] = {
        "rows": int(p.size),
        "events": int(y.sum()),
        "in_sample_nll": {kind: binary_nll(m(p), y) for kind, m in fitted.items()},
        "fitted": {kind: found.to_payload() for kind, found in fitted.items()},
        "margin": float(margin),
        "fold_share": float(fold_share),
        "chosen_terms": {},
    }
    about: dict[str, Any] | None = None
    if flags is not None:
        about = {
            "names": list(CONTEXT),
            "rows": {name: int(flags[:, i].sum()) for i, name in enumerate(CONTEXT)},
            "events": {
                name: int((flags[:, i] & (y == 1)).sum())
                for i, name in enumerate(CONTEXT)
            },
            "steps": [],
            "chosen": [],
            "fitted": None,
            "in_sample_nll": None,
            "cross_validated_nll": None,
            "gain_over_simpler": None,
            "better_folds": None,
            "margin": float(term_margin),
            "fold_share": float(fold_share),
            "accepted": False,
            "used": False,
            "reason": "no flag has enough rows of both outcomes: no term fitted",
        }
    record["context"] = about
    ids = np.arange(p.size) if groups is None else np.asarray(groups).reshape(-1)
    if ids.shape != p.shape:
        raise ValueError(f"{ids.size} groups for {p.size} rows")
    fold, used = _fold_index(ids, folds, seed)
    record["folds"] = used
    if fitted[MAP_LOGISTIC].is_identity and fitted[MAP_ISOTONIC].is_identity:
        record.update(
            chosen=MAP_IDENTITY,
            reason="too few rows of one outcome, or an uninformative share: no map",
            cross_validated_nll=None,
            isotonic_better_folds=None,
        )
        if about is not None:
            about["reason"] = "no map is fitted for this event: no terms either"
        return fitted[MAP_IDENTITY], record
    if used < 2:
        chosen = MAP_IDENTITY if fitted[MAP_LOGISTIC].is_identity else MAP_LOGISTIC
        record.update(
            chosen=chosen,
            reason="fewer than two groups: no cross-validation, the simpler map",
            cross_validated_nll=None,
            isotonic_better_folds=None,
        )
        if about is not None:
            about["reason"] = "no cross-validation: the terms are not taken"
        return fitted[chosen], record
    helds = [fold == index for index in range(used)]
    helds = [held for held in helds if held.any()]

    def held_out(kind: str, around: np.ndarray | None = None) -> list[float]:
        """Per fold: the held-out NLL sum of the map fitted on the other folds."""
        out: list[float] = []
        for held in helds:
            train_p, train_y = p[~held], y[~held]
            if kind == MAP_IDENTITY:
                found = EventMap()
            elif kind == MAP_ISOTONIC:
                found = fit_isotonic(train_p, train_y)
            elif around is None:
                found = fit_logistic(train_p, train_y)
            else:
                found = fit_logistic(train_p, train_y, context=around[~held])
            given = None if around is None else around[held]
            out.append(binary_nll(found(p[held], given), y[held]) * int(held.sum()))
        return out

    parts = {kind: held_out(kind) for kind in MAP_KINDS}
    cross = {kind: sum(parts[kind]) / p.size for kind in MAP_KINDS}
    record["cross_validated_nll"] = cross
    simple = MAP_IDENTITY
    if not fitted[MAP_LOGISTIC].is_identity and cross[MAP_LOGISTIC] < cross[simple]:
        simple = MAP_LOGISTIC
    needed = math.ceil(fold_share * used)
    base_map, base_parts, base_nll, base_text = (
        fitted[simple],
        parts[simple],
        cross[simple],
        simple,
    )
    if flags is not None and about is not None:
        taken: list[int] = []
        shown: tuple[EventMap, list[float]] | None = None  # the best map with terms
        while len(taken) < len(CONTEXT):
            tried: dict[str, Any] = {}
            best: tuple[float, int, EventMap, list[float], int] | None = None
            for column in range(len(CONTEXT)):
                if column in taken:
                    continue
                around = _only(flags, [*taken, column])
                found = fit_logistic(p, y, context=around)
                if CONTEXT[column] not in dict(found.terms):
                    continue  # not usable: too few rows of one outcome
                own = held_out(MAP_LOGISTIC, around)
                gain = base_nll - sum(own) / p.size
                wins = sum(int(a < b) for a, b in zip(own, base_parts))
                tried[CONTEXT[column]] = {
                    "terms": dict(found.terms),
                    "cross_validated_nll": sum(own) / p.size,
                    "gain": gain,
                    "better_folds": wins,
                }
                if best is None or gain > best[0]:
                    best = (gain, column, found, own, wins)
            step: dict[str, Any] = {"tried": tried, "best": None, "accepted": False}
            about["steps"].append(step)
            if best is None:
                break
            gain, column, found, own, wins = best
            step["best"] = CONTEXT[column]
            if shown is None:
                shown = (found, own)
            if not (gain > term_margin and wins >= needed):
                break
            step["accepted"] = True
            taken.append(column)
            shown = (found, own)
            base_map, base_parts, base_nll = found, own, sum(own) / p.size
            base_text = "logistic with context terms"
        about["chosen"] = [CONTEXT[column] for column in taken]
        about["accepted"] = bool(taken)
        if shown is not None:
            found, own = shown
            total = cross[simple] - sum(own) / p.size
            wins = sum(int(a < b) for a, b in zip(own, parts[simple]))
            about.update(
                fitted=found.to_payload(),
                in_sample_nll=binary_nll(found(p, flags), y),
                cross_validated_nll=sum(own) / p.size,
                gain_over_simpler=total,
                better_folds=wins,
            )
            last = about["steps"][-1]
            left = (
                ""
                if last["accepted"] or last["best"] is None
                else f"; the next best, {last['best']}, adds "
                f"{last['tried'][last['best']]['gain']:+.5f} nats in "
                f"{last['tried'][last['best']]['better_folds']} of {len(helds)} "
                "folds and is left out"
            )
            if taken:
                about["reason"] = (
                    f"taken ({', '.join(about['chosen'])}): {total:.5f} nats better "
                    f"than {simple} without terms out of sample and better in "
                    f"{wins} of {len(helds)} folds; each term had to add more than "
                    f"{term_margin:g} nats in at least {needed} folds{left}"
                )
            else:
                about["reason"] = (
                    f"not taken: the best single term ({last['best']}) is "
                    f"{total:+.5f} nats against {simple} without terms out of "
                    f"sample (needs more than {term_margin:g}) and better in "
                    f"{wins} of {len(helds)} folds (needs {needed})"
                )
    iso_wins = sum(int(a < b) for a, b in zip(parts[MAP_ISOTONIC], base_parts))
    record["isotonic_better_folds"] = iso_wins
    gain = base_nll - cross[MAP_ISOTONIC]
    clearly = (
        not fitted[MAP_ISOTONIC].is_identity and gain > margin and iso_wins >= needed
    )
    chosen_map = fitted[MAP_ISOTONIC] if clearly else base_map
    record["chosen"] = chosen_map.kind
    record["chosen_terms"] = dict(chosen_map.terms)
    record["isotonic_gain_over_simpler"] = gain
    if about is not None:
        about["used"] = chosen_map.needs_context
    if clearly:
        record["reason"] = (
            f"isotonic is {gain:.5f} nats better than {base_text} out of sample "
            f"(margin {margin:g}) and wins {iso_wins} of {used} folds"
        )
    elif chosen_map.is_identity:
        record["reason"] = "the logistic map is not better than no map out of sample"
    else:
        record["reason"] = (
            f"{base_text}; isotonic is {gain:+.5f} nats against it out of sample "
            f"(needs more than {margin:g}) and wins {iso_wins} of {used} folds"
        )
    return chosen_map, record


def fit_event_calibration(
    action: Any,
    batch: Mapping[str, np.ndarray],
    *,
    rows: Any = None,
    groups: Any = None,
    action_temperature: float | None = None,
    info: Mapping[str, Any] | None = None,
    folds: int = FOLDS,
    seed: int = SEED,
    margin: float = ISOTONIC_MARGIN,
    context: bool = True,
    term_margin: float = TERM_MARGIN,
) -> EventCalibration:
    """Fit both event maps on the rows given; nothing else is read.

    ``action`` is the predictor's action probabilities ``[N, 2, A]`` (after its
    temperatures, made from the features alone) and ``batch`` the same
    examples with their labels. ``rows`` (a boolean mask or an index array
    over the examples) selects the examples to fit on BEFORE anything is
    computed: a row outside it cannot influence the result. ``groups`` is one
    id per example (the battle) for the cross-validation folds; without it
    every example is its own group.

    With ``context`` (the default) and a batch that holds the feature arrays
    of ``CONTEXT_ARRAYS``, each map may get context terms (``select_map``
    decides, by the same cross-validation); a batch without those arrays, or
    ``context=False``, gives maps without terms. Of the batch only
    ``action_mask``, ``cand_flag``, ``act_mon``, ``y_flag`` and, for the
    terms, ``turn`` and ``mon_flag`` are read.

    ``info`` is copied into the result's ``info`` (dataset tag and the like),
    next to what the fit records per event: the rows, the observed rate and
    the mean prediction and event NLL before and after over every slot-turn
    where the event is known (the scorecard's rows), the same for the fitted
    objective, and ``select_map``'s record. Raises ``ValueError`` /
    ``KeyError`` on arrays that do not fit.
    """
    made = np.asarray(action, dtype=np.float64)
    names = ["action_mask", "cand_flag", "act_mon", "y_flag"]
    with_context = bool(context) and all(name in batch for name in CONTEXT_ARRAYS)
    if with_context:
        names += [name for name in CONTEXT_ARRAYS if name not in names]
    data = {name: np.asarray(batch[name]) for name in names}
    n = int(made.shape[0]) if made.ndim == 3 else -1
    if n < 0 or any(array.shape[:1] != (n,) for array in data.values()):
        raise ValueError("action must be [N, 2, A], one row per example of the batch")
    ids = np.arange(n) if groups is None else np.asarray(groups).reshape(-1)
    if ids.shape != (n,):
        raise ValueError(f"{ids.size} groups for {n} examples")
    if rows is not None:
        index = np.asarray(rows)
        if index.dtype == bool and index.shape != (n,):
            raise ValueError(f"a row mask of {index.shape} for {n} examples")
        made, ids = made[index], ids[index]
        data = {name: array[index] for name, array in data.items()}
    mask, flags = data["action_mask"], data["cand_flag"]
    before = event_shares(made, mask, flags)
    labels = event_labels(data)
    around = event_context(data) if with_context else None
    slot_ids = np.broadcast_to(ids[:, None], before.switch.shape)

    known_s, switched = labels[EVENT_SWITCH]
    known_p, protected = labels[EVENT_PROTECT]
    fit_s = known_s & before.switch_possible
    fit_p = known_p & before.protect_possible & known_s & ~switched
    switch_map, switch_record = select_map(
        before.switch[fit_s],
        switched[fit_s],
        slot_ids[fit_s],
        folds=folds,
        seed=seed,
        margin=margin,
        context=None if around is None else around[fit_s],
        term_margin=term_margin,
    )
    protect_map, protect_record = select_map(
        before.protect_share[fit_p],
        protected[fit_p],
        slot_ids[fit_p],
        folds=folds,
        seed=seed,
        margin=margin,
        context=None if around is None else around[fit_p],
        term_margin=term_margin,
    )
    after = event_shares(
        rescale(made, mask, flags, switch_map, protect_map, around), mask, flags
    )

    def block(
        known: np.ndarray,
        happened: np.ndarray,
        fitted: np.ndarray,
        event: tuple[np.ndarray, np.ndarray],
        objective: tuple[np.ndarray, np.ndarray],
        record: dict[str, Any],
    ) -> dict[str, Any]:
        def mean(values: np.ndarray, where: np.ndarray) -> float | None:
            return float(values[where].mean()) if where.any() else None

        return {
            "rows_known": int(known.sum()),
            "rows_fitted": int(fitted.sum()),
            "observed": mean(happened.astype(np.float64), known),
            "predicted_before": mean(event[0], known),
            "predicted_after": mean(event[1], known),
            "nll_before": binary_nll(event[0][known], happened[known]),
            "nll_after": binary_nll(event[1][known], happened[known]),
            "fit": {
                "observed": mean(happened.astype(np.float64), fitted),
                "predicted_before": mean(objective[0], fitted),
                "predicted_after": mean(objective[1], fitted),
                "nll_before": binary_nll(objective[0][fitted], happened[fitted]),
                "nll_after": binary_nll(objective[1][fitted], happened[fitted]),
            },
            "selection": record,
        }

    record: dict[str, Any] = dict(info or {})
    record.update(
        examples=int(made.shape[0]),
        groups=int(np.unique(ids).size),
        seed=int(seed),
        context=list(CONTEXT) if with_context else None,
        events={
            EVENT_SWITCH: block(
                known_s,
                switched,
                fit_s,
                (before.switch, after.switch),
                (before.switch, after.switch),
                switch_record,
            ),
            EVENT_PROTECT: block(
                known_p,
                protected,
                fit_p,
                (before.protect, after.protect),
                (before.protect_share, after.protect_share),
                protect_record,
            ),
        },
    )
    return EventCalibration(switch_map, protect_map, action_temperature, record)
