"""Joint replies of a player's two slots, built from one per-slot prediction.

A ``features.Predictor`` answers for each of a player's two active slots on its
own: ``action`` ``[N, 2, A]``, ``target`` ``[N, 2, n_cand + 1, 5]`` given the
move, ``mega`` ``[N, 2]``. A search wants "probability per joint reply": what
the player does with BOTH slots this turn. This module builds that list.

    made = joint_replies(pred, batch, k=8)            # a batch, or one example
    made.prob[i]                                      # [K], most probable first
    made.listing(i)                                   # the same as plain data
    made = joint_replies(pred, batch, truth=true_replies(batch))
    made.rank[i]                                      # 0 = the first of the list

THE FACTORISATION IS THE MODEL'S OWN, NOT A FINDING. The probability of a joint
reply is the PRODUCT of the two slots' marginals (and, with ``mega=True``, of
each slot's Mega marginal): the predictors say nothing about how the two
choices go together, so the joint holds no such dependence either. Two slots
that usually act together (one protects while the other sets up) are
under-rated, and combinations nobody plays are over-rated, by exactly as much
as the truth departs from independence. The only dependence added here is the
removal of what the rules forbid, after which the rest is renormalised:

* both slots switching to the same bench Pokemon;
* with the Mega bit: both slots Mega-evolving (a side has one Mega Evolution a
  game), a switch together with a Mega Evolution of the same slot, and a Mega
  Evolution of a Pokemon the public state says cannot (``mon_flag`` "mega
  possible").

A slot with no Pokemon on the field contributes the single reply "no action".

A COUPLING IS OPTIONAL. ``joint_replies(..., coupling=<coupling.PairCoupling>,
move_intent=<the featurizer's tables.move_intent>)`` multiplies every pair of
replies by one number per pair of reply classes and context bucket before the
renormalisation (see ``coupling.py``): the two slots are then no longer
independent. Without one (the default) the code path and every array are
those of the plain product; an all-ones coupling is read as none. ``kept``
stays the mass under the plain product; ``tilt`` is the coupled total over it
and ``coupled`` says which joint the list holds. A coupling that cannot be
applied to the batch gives the failure value (``n == 0``) and a counter, never
the plain list under the coupled name.

THE COUPLING AND THE MEGA BIT. The table is fitted on the joint WITHOUT the
Mega bit (``coupling.pair_masses`` knows no Mega state) and, with
``mega=True``, multiplied unchanged into each of the three Mega states. The
plain mass of a class pair is then not the fit's: a switch cannot carry a Mega
Evolution, so a pair that holds a switch has less of the Mega states' mass
than a pair of moves. Without the Mega bit the change of the true reply's
log-probability is exactly the fit's own formula (``coupling.row_gain``; 4e-15
on the fully seen two-slot validation rows of the first fit, 2026-10-10); with
it the two differ by up to 0.04 nats on a row and 0.0001 in the mean (+0.0139
against +0.0140). The coupled Mega joint is "the plain Mega joint times R,
renormalised", not a joint whose pair marginal is the coupled pair joint.

The Mega head is a marginal too: "this slot Mega-evolves this turn", whatever
it does. It is multiplied in as it is, so once a switch with a Mega Evolution
is removed the joint holds a little less Mega probability than the head said.
Reading the head as P(moves) x P(Mega | moves) instead moves the top-8
coverage with the Mega bit by 0.3 points at most (2026-10-05, v2 artifacts).

One slot's reply is an index (``n_cand`` candidates, ``N_TARGET`` = 5 classes):

    c * 5 + t           candidate move ``c`` aimed at ``TARGET_CLASSES[t]``
    n_cand * 5          OTHER: some move outside the candidates, as ONE bucket
                        (no target: a move nobody named has no target split)
    n_cand * 5 + 1 + r  a switch to roster Pokemon ``r`` (team-preview order)
    NO_ACTION (-1)      the slot is empty

With the Mega bit a joint reply also says who Mega-evolves: ``MEGA_NONE``,
``MEGA_A`` or ``MEGA_B`` (never both, so three states, not four).

Order and ties. The list is sorted by probability, highest first; replies with
exactly the same probability keep the order (Mega state, slot a's index, slot
b's index). ``rank`` is the position of the true reply in that same order, so
"the truth is among the first K" and ``rank < K`` are one statement. ``ties``
counts the other possible replies with exactly the truth's probability: where
it is large (a uniform predictor) the rank is decided by that order, not by
the predictor.

OTHER is a place in the list, not a move: a consumer that needs a concrete
move cannot play it, and a true reply that used a move outside the candidates
can only ever match the bucket. ``true_replies`` marks those (``other``) so a
caller can count them as not covered.

Nothing a runtime calls here raises into its caller (``joint_replies``,
``slot_replies``, ``true_replies``, ``describe_reply`` and the methods of
``JointReplies``): a prediction that does not fit the batch or holds a
non-finite number, a ``k`` that is not a positive whole number, or a ``mega``
that is not a plain True / False (a caller's slip such as handing over the
Mega array) gives an empty ``JointReplies`` (``n == 0``) or an empty dict, and
a name in ``COUNTERS``. The four index helpers (``reply_size``, ``move_reply``,
``other_reply``, ``switch_reply``) are plain arithmetic on whole numbers and
have no such guard. An example whose possible replies carry no probability at
all comes back with ``ok`` False, zero probabilities and rank -1.

The work is a full enumeration of the joint table (4,624 entries for 12
candidates, three times that for the rows where a Mega Evolution is
possible), in chunks: about 0.05 ms an example on one core. The list is never
longer than the table: a ``k`` above the number of entries is cut to it, so a
huge ``k`` cannot ask for memory the answer does not need.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from vgc_bench.src.oppmodel.events import (
    INTENT_PROTECT,
    KIND_MOVE,
    KIND_SWITCH,
    TARGET_CLASSES,
)
from vgc_bench.src.oppmodel.features import (
    CAND_PROTECT,
    CAND_REVEALED,
    CAND_SHEET,
    FLAG_MEGA_POSSIBLE,
    N_ROSTER,
    N_SLOT,
    N_TARGET,
    expand_target_mask,
)

NO_ACTION = -1  # reply index of a slot with no Pokemon on the field
UNKNOWN = -2  # a true reply the log does not show whole; padding of a short list
MEGA_NONE, MEGA_A, MEGA_B = 0, 1, 2
N_MEGA_STATE = 3
KIND_OTHER = "other"
DEFAULT_K = 8
MAX_K = 1 << 20  # a list length above this is cut to it before any table is known
CHUNK_BYTES = 32 << 20  # of the float64 joint table held at once

# Failures of the public functions, by function and exception type.
COUNTERS: Counter[str] = Counter()


def _failed(name: str, exc: Exception) -> None:
    try:
        COUNTERS[f"{name}:{type(exc).__name__}"] += 1
    except Exception:
        pass


# --- the reply index of one slot ------------------------------------------------


def reply_size(n_cand: int) -> int:
    """Number of replies of one slot on the field: moves x targets, OTHER, switches."""
    return n_cand * N_TARGET + 1 + N_ROSTER


def move_reply(candidate: int, target: int) -> int:
    """Reply index of candidate move ``candidate`` aimed at target class ``target``."""
    return candidate * N_TARGET + target


def other_reply(n_cand: int) -> int:
    """Reply index of the OTHER bucket."""
    return n_cand * N_TARGET


def switch_reply(roster_index: int, n_cand: int) -> int:
    """Reply index of a switch to roster Pokemon ``roster_index``."""
    return n_cand * N_TARGET + 1 + roster_index


def describe_reply(index: int, n_cand: int) -> dict[str, Any] | None:
    """One slot's reply as plain data; None for an empty slot or a bad index.

    ``action`` is the index on the predictor's own action axis (candidate
    column, OTHER, switch pointer), so a caller can name the move or the
    Pokemon from what it already holds. Never raises.
    """
    try:
        index, n_cand = int(index), int(n_cand)
        other = other_reply(n_cand)
        if index < 0 or index >= reply_size(n_cand):
            return None
        if index < other:
            candidate, target = divmod(index, N_TARGET)
            return {
                "kind": KIND_MOVE,
                "action": candidate,
                "target": TARGET_CLASSES[target],
            }
        if index == other:
            return {"kind": KIND_OTHER, "action": n_cand}
        roster = index - other - 1
        return {"kind": KIND_SWITCH, "action": n_cand + 1 + roster, "roster": roster}
    except Exception as exc:
        _failed("describe_reply", exc)
        return None


# --- per-slot replies from a prediction -----------------------------------------


def _lift(
    pred: Mapping[str, np.ndarray], batch: Mapping[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """A batch of one from one example's arrays; a batch is passed through.

    Each of the two is judged on its own (``action`` ``[2, A]``, ``action_mask``
    ``[2, A]``), so one example's prediction goes with a batch of one.
    """
    made, masks = dict(pred), dict(batch)
    if np.asarray(pred["action"]).ndim == 2:
        made = {name: np.asarray(value)[None] for name, value in pred.items()}
    if np.asarray(batch["action_mask"]).ndim == 2:
        masks = {name: np.asarray(value)[None] for name, value in batch.items()}
    return made, masks


def _slot_replies(
    pred: Mapping[str, np.ndarray], batch: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(batch["action_mask"]).astype(bool)
    legal_target = expand_target_mask(batch["cand_tmask"])
    action = np.asarray(pred["action"], dtype=np.float64)
    target = np.asarray(pred["target"], dtype=np.float64)
    if mask.ndim != 3 or mask.shape[1] != N_SLOT:
        raise ValueError(f"action_mask {mask.shape} is not [N, 2, A]")
    n_cand = legal_target.shape[-2] - 1
    if n_cand < 0 or mask.shape[-1] != n_cand + 1 + N_ROSTER:
        raise ValueError(f"masks disagree: {mask.shape} and {legal_target.shape}")
    if action.shape != mask.shape or target.shape != legal_target.shape:
        raise ValueError(f"prediction {action.shape} / {target.shape} does not fit")
    if not (np.isfinite(action).all() and np.isfinite(target).all()):
        raise ValueError("prediction holds a non-finite probability")
    action = np.where(mask, np.maximum(action, 0.0), 0.0)
    total = action.sum(-1, keepdims=True)
    action = np.divide(action, total, out=np.zeros_like(action), where=total > 0)
    target = np.where(legal_target, np.maximum(target, 0.0), 0.0)
    total = target.sum(-1, keepdims=True)
    target = np.divide(target, total, out=np.zeros_like(target), where=total > 0)
    lead = mask.shape[:2]
    moves = action[:, :, :n_cand, None] * target[:, :, :n_cand]
    prob = np.concatenate(
        [moves.reshape(*lead, n_cand * N_TARGET), action[:, :, n_cand:]], axis=-1
    )
    aimed = mask[:, :, :n_cand, None] & legal_target[:, :, :n_cand]
    legal = np.concatenate(
        [aimed.reshape(*lead, n_cand * N_TARGET), mask[:, :, n_cand:]], axis=-1
    )
    return prob, legal


def slot_replies(
    pred: Mapping[str, np.ndarray], batch: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Each slot's replies: (probability, possible), both ``[N, 2, R]``.

    ``R = reply_size(n_cand)``. Illegal mass is removed and each head
    renormalised first, as ``features.normalize_prediction`` does; a move's
    probability is then split over its legal target classes, and OTHER keeps
    its own as one bucket. A slot with no Pokemon has no possible reply. Never
    raises: two empty arrays when the prediction does not fit the batch.
    """
    try:
        return _slot_replies(*_lift(pred, batch))
    except Exception as exc:
        _failed("slot_replies", exc)
        return np.zeros((0, N_SLOT, 0)), np.zeros((0, N_SLOT, 0), dtype=bool)


def _mega_possible(batch: Mapping[str, np.ndarray], active: np.ndarray) -> np.ndarray:
    """``[N, 2]``: the slot holds a Pokemon the public state lets Mega-evolve."""
    if "mega_possible" in batch:
        return np.asarray(batch["mega_possible"]).astype(bool) & active
    rows = np.asarray(batch["act_mon"]).astype(np.int64)
    flag = np.asarray(batch["mon_flag"])[..., FLAG_MEGA_POSSIBLE]
    taken = np.take_along_axis(flag, np.maximum(rows, 0), axis=1)
    return (taken > 0) & (rows >= 0) & active


# --- what the labels say --------------------------------------------------------


def _true_replies(batch: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    active = np.asarray(batch["act_mon"]) >= 0
    y_action = np.asarray(batch["y_action"]).astype(np.int64)
    y_target = np.asarray(batch["y_target"]).astype(np.int64)
    y_mega = np.asarray(batch["y_mega"]).astype(np.int64)
    flags = np.asarray(batch["cand_flag"]).astype(np.int64)
    n_cand = flags.shape[-1]
    candidate = active & (y_action >= 0) & (y_action < n_cand)
    other = active & (y_action == n_cand)
    switch = active & (y_action > n_cand)
    aimed = y_target >= 0
    reply = np.full(y_action.shape, UNKNOWN, dtype=np.int64)
    known = candidate & aimed
    reply[known] = y_action[known] * N_TARGET + y_target[known]
    reply[other & aimed] = other_reply(n_cand)
    reply[switch] = y_action[switch] - (n_cand + 1) + other_reply(n_cand) + 1
    reply[~active] = NO_ACTION
    if n_cand:
        column = np.clip(y_action, 0, n_cand - 1)[..., None]
        own = np.take_along_axis(flags, column, axis=-1)[..., 0]
    else:
        own = np.zeros_like(y_action)
    shown = candidate & ((own & (CAND_REVEALED | CAND_SHEET)) > 0)
    guarded = candidate & ((own & CAND_PROTECT) > 0)
    took = active & (y_mega == 1)
    untold = active & (y_mega < 0) & _mega_possible(batch, active)
    mega = np.where(took[:, 0], MEGA_A, np.where(took[:, 1], MEGA_B, MEGA_NONE))
    mega = np.where(took.all(-1) | (~took.any(-1) & untold.any(-1)), UNKNOWN, mega)
    return {
        "reply": reply,
        "mega": mega.astype(np.int64),
        "visible": (reply != UNKNOWN).all(-1) & active.any(-1),
        "active": active,
        "other": other,
        "switch": switch,
        "shown": shown,
        "unshown": (candidate | other) & ~shown,
        INTENT_PROTECT: guarded,
    }


def true_replies(batch: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """What the labels of a batch say the player did, in this module's indices.

    * ``reply`` ``[N, 2]``: the slot's reply index; ``NO_ACTION`` for an empty
      slot; ``UNKNOWN`` where the log does not show the whole action (a hidden
      or forced action, or a move whose target is not certain: exactly the
      slots where ``features.fine_label`` is -1).
    * ``visible`` ``[N]``: every slot on the field has a known reply.
    * ``mega`` ``[N]``: ``MEGA_NONE`` / ``MEGA_A`` / ``MEGA_B``; ``UNKNOWN``
      when the log cannot tell for a slot that could, or names both slots.
    * per slot ``[N, 2]``: ``active``; ``other`` (a move outside the
      candidates); ``switch``; ``shown`` (a candidate move the Pokemon had
      shown in this battle or that is on its open sheet); ``unshown`` (any
      other move: a guessed candidate, or OTHER); and, under the key
      ``events.INTENT_PROTECT``, a candidate of the Protect family.

    Reads label arrays: never give its result to a predictor. Never raises: an
    empty dict when the batch lacks an array.
    """
    try:
        return _true_replies(batch)
    except Exception as exc:
        _failed("true_replies", exc)
        return {}


# --- the joint ------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class JointReplies:
    """The first ``k`` joint replies of ``n`` examples, and the truth's place.

    * ``reply`` ``[n, k, 2]``: slot a's and slot b's reply index (``NO_ACTION``
      for an empty slot); ``UNKNOWN`` pads an example with fewer than ``k``
      possible joint replies.
    * ``mega`` ``[n, k]``: ``MEGA_NONE`` / ``MEGA_A`` / ``MEGA_B`` (always
      ``MEGA_NONE`` when the list was built without the Mega bit).
    * ``prob`` ``[n, k]``: probability after the impossible combinations were
      removed and the rest renormalised; 0 for padding.
    * ``size`` ``[n]``: how many joint replies are possible; ``kept`` ``[n]``:
      the mass they held under the plain product, before renormalising.
    * ``ok`` ``[n]``: the example has a joint reply with positive probability.
    * ``rank`` ``[n]``: 0-based place of the true reply in the full order, -1
      when no truth was given, it is not a possible joint reply, or not ``ok``;
      ``prob_true`` ``[n]``: its probability (0 without a rank); ``above``
      ``[n]``: possible replies with a strictly higher probability; ``ties``
      ``[n]``: other possible replies with exactly the same one. Whatever the
      order of equal replies, the rank lies in ``[above, above + ties]``.
    * ``coupled``: the list was built with a pair coupling; ``tilt`` ``[n]``:
      the coupled total over ``kept`` (1 without a coupling, and for an
      example with one acting slot).
    """

    n_cand: int
    k: int
    with_mega: bool
    reply: np.ndarray
    mega: np.ndarray
    prob: np.ndarray
    size: np.ndarray
    kept: np.ndarray
    ok: np.ndarray
    rank: np.ndarray
    prob_true: np.ndarray
    above: np.ndarray
    ties: np.ndarray
    tilt: np.ndarray
    coupled: bool = False

    @property
    def n(self) -> int:
        return int(self.prob.shape[0])

    def mass(self, k: int | None = None) -> np.ndarray:
        """``[n]``: probability held by the first ``k`` replies (default: all kept).

        Never raises: zeros, and a count in ``COUNTERS``, for a ``k`` that is
        not a number.
        """
        try:
            width = self.k if k is None else max(0, min(int(k), self.k))
            return self.prob[:, :width].sum(axis=1)
        except Exception as exc:
            _failed("mass", exc)
            return np.zeros(self.n, dtype=np.float64)

    def listing(self, row: int = 0) -> list[dict[str, Any]]:
        """One example's joint replies as plain data, most probable first.

        Each entry: ``p``, ``slots`` (two ``describe_reply`` dicts, None for an
        empty slot) and ``mega`` (0 / 1 = the slot that Mega-evolves, else
        None). Never raises: an empty list for a row that does not exist.
        """
        out: list[dict[str, Any]] = []
        try:
            if not 0 <= int(row) < self.n:
                return []
            for position in range(self.k):
                pair = [int(value) for value in self.reply[row, position]]
                if UNKNOWN in pair:
                    break
                state = int(self.mega[row, position])
                out.append(
                    {
                        "p": float(self.prob[row, position]),
                        "slots": [describe_reply(v, self.n_cand) for v in pair],
                        "mega": None if state == MEGA_NONE else state - 1,
                    }
                )
        except Exception as exc:
            _failed("listing", exc)
            return []
        return out


def _empty(
    k: int, with_mega: bool, n: int = 0, n_cand: int = 0, coupled: bool = False
) -> JointReplies:
    """``n`` examples with nothing in them (``n == 0``: the failure value)."""
    width = max(1, int(k))
    return JointReplies(
        n_cand=n_cand,
        k=width,
        with_mega=with_mega,
        reply=np.full((n, width, N_SLOT), UNKNOWN, dtype=np.int16),
        mega=np.zeros((n, width), dtype=np.int8),
        prob=np.zeros((n, width), dtype=np.float64),
        size=np.zeros(n, dtype=np.int32),
        kept=np.zeros(n, dtype=np.float64),
        ok=np.zeros(n, dtype=bool),
        rank=np.full(n, -1, dtype=np.int32),
        prob_true=np.zeros(n, dtype=np.float64),
        above=np.zeros(n, dtype=np.int32),
        ties=np.zeros(n, dtype=np.int32),
        tilt=np.ones(n, dtype=np.float64),
        coupled=coupled,
    )


def _truth_index(
    truth: Mapping[str, np.ndarray] | None,
    n: int,
    width: int,
    with_mega: bool,
    wide: np.ndarray,
) -> np.ndarray:
    """``[n]``: the true joint reply's place in the flat table, -1 = none.

    ``width`` is ``R + 1`` (the last column is "no action"); ``wide`` marks the
    rows whose table holds the three Mega states.
    """
    if truth is None:
        return np.full(n, -1, dtype=np.int64)
    reply = np.asarray(truth["reply"]).astype(np.int64).reshape(n, N_SLOT)
    column = np.where(reply == NO_ACTION, width - 1, reply)
    valid = ((reply == NO_ACTION) | ((reply >= 0) & (reply < width - 1))).all(-1)
    state = np.zeros(n, dtype=np.int64)
    if with_mega:
        if "mega" not in truth:
            return np.full(n, -1, dtype=np.int64)
        state = np.asarray(truth["mega"]).astype(np.int64).reshape(n)
        valid &= (state >= 0) & (state < N_MEGA_STATE)
        valid &= (state == MEGA_NONE) | wide  # a Mega only where one is possible
    flat = state * width * width + column[:, 0] * width + column[:, 1]
    return np.where(valid, flat, -1)


def _chunk(
    pa: np.ndarray,
    pb: np.ndarray,
    la: np.ndarray,
    lb: np.ndarray,
    q: np.ndarray | None,
    can: np.ndarray | None,
    k: int,
    truth: np.ndarray,
    factor: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Enumerate the joint replies of ``b`` rows; ``q`` None = one Mega state.

    ``pa`` / ``pb`` ``[b, R + 1]`` are the two slots' reply probabilities with
    the "no action" column last, ``la`` / ``lb`` which replies are possible,
    ``q`` / ``can`` ``[b, 2]`` the Mega probability and whether a Mega
    Evolution is possible. An impossible joint reply holds -1 in the table, so
    it sorts after every possible one and equals no probability. ``factor``
    ``[b, R + 1, R + 1]`` (a coupling's number for every pair of replies,
    positive) multiplies every possible entry, the same in each Mega state;
    None is the plain product and runs no arithmetic of its own.
    """
    b, width = pa.shape
    first_switch = width - 1 - N_ROSTER
    table = pa[:, :, None] * pb[:, None, :]
    allowed = la[:, :, None] & lb[:, None, :]
    for column in range(first_switch, width - 1):
        allowed[:, column, column] = False  # both to the same bench Pokemon
    if q is None or can is None:
        flat = table.reshape(b, -1)
        flat[~allowed.reshape(b, -1)] = -1.0
    else:
        qa, qb = q[:, 0], q[:, 1]
        scale = np.stack([(1 - qa) * (1 - qb), qa * (1 - qb), (1 - qa) * qb], axis=1)
        moves = np.arange(width) < first_switch  # a move or OTHER: can carry a Mega
        states = np.empty((b, N_MEGA_STATE, width, width), dtype=bool)
        states[:, MEGA_NONE] = allowed
        states[:, MEGA_A] = allowed & (can[:, :1] & moves[None, :])[:, :, None]
        states[:, MEGA_B] = allowed & (can[:, 1:] & moves[None, :])[:, None, :]
        flat = (table[:, None] * scale[:, :, None, None]).reshape(b, -1)
        flat[~states.reshape(b, -1)] = -1.0
    total = flat.shape[1]
    size = (flat >= 0).sum(axis=1)
    kept = np.maximum(flat, 0.0).sum(axis=1)
    ok = kept > 0
    scale_back = np.where(ok, kept, 1.0)
    tilt = np.ones(b, dtype=np.float64)
    if factor is not None:
        layers = total // (width * width)  # 1, or the three Mega states
        wide = np.broadcast_to(factor[:, None], (b, layers, width, width))
        flat = np.where(flat >= 0, flat * wide.reshape(b, -1), -1.0)
        coupled = np.maximum(flat, 0.0).sum(axis=1)
        ok = ok & (coupled > 0)
        scale_back = np.where(ok, coupled, 1.0)
        tilt = np.where(ok, coupled / np.where(kept > 0, kept, 1.0), 1.0)

    # The first ``k`` in the stable order: everything above the k-th value, then
    # the lowest-indexed entries that equal it.
    take = min(k, total)
    kth = np.partition(flat, total - take, axis=1)[:, total - take]
    above = flat > kth[:, None]
    equal = flat == kth[:, None]
    need = take - above.sum(axis=1)
    chosen = above | (
        equal & (np.cumsum(equal, axis=1, dtype=np.int32) <= need[:, None])
    )
    index = np.nonzero(chosen)[1].reshape(b, take)
    value = np.take_along_axis(flat, index, axis=1)
    order = np.argsort(-value, axis=1, kind="stable")
    index = np.take_along_axis(index, order, axis=1)
    value = np.take_along_axis(value, order, axis=1)
    real = (value >= 0) & ok[:, None]
    slot_a = (index // width) % width
    slot_b = index % width
    reply = np.stack([slot_a, slot_b], axis=-1)
    reply = np.where(reply == width - 1, NO_ACTION, reply)
    reply = np.where(real[..., None], reply, UNKNOWN)
    out = {
        "reply": np.full((b, k, N_SLOT), UNKNOWN, dtype=np.int16),
        "mega": np.zeros((b, k), dtype=np.int8),
        "prob": np.zeros((b, k), dtype=np.float64),
        "size": size.astype(np.int32),
        "kept": kept,
        "ok": ok,
        "tilt": tilt,
    }
    out["reply"][:, :take] = reply
    out["mega"][:, :take] = np.where(real, index // (width * width), MEGA_NONE)
    out["prob"][:, :take] = np.where(real, value / scale_back[:, None], 0.0)

    # The truth's place: possible replies above it, and equal ones before it.
    place = np.clip(truth, 0, total - 1)
    own = flat[np.arange(b), place]
    found = (truth >= 0) & ok & (own >= 0)
    bar = np.where(found, own, np.inf)
    same = flat == bar[:, None]
    before = (same & (np.arange(total)[None, :] < place[:, None])).sum(axis=1)
    higher = (flat > bar[:, None]).sum(axis=1)
    out["rank"] = np.where(found, higher + before, -1).astype(np.int32)
    out["prob_true"] = np.where(found, own / scale_back, 0.0)
    out["above"] = np.where(found, higher, 0).astype(np.int32)
    out["ties"] = np.where(found, same.sum(axis=1) - 1, 0).astype(np.int32)
    return out


def _joint(
    pred: Mapping[str, np.ndarray],
    batch: Mapping[str, np.ndarray],
    k: int,
    with_mega: bool,
    truth: Mapping[str, np.ndarray] | None,
    chunk_bytes: int,
    coupling: Any = None,
    move_intent: Any = None,
) -> JointReplies:
    pred, batch = _lift(pred, batch)
    if truth is not None:
        truth = {name: np.asarray(value) for name, value in truth.items()}
    prob, legal = _slot_replies(pred, batch)
    n, _, size = prob.shape
    n_cand = (size - 1 - N_ROSTER) // N_TARGET
    width = size + 1
    active = legal.any(-1)
    wide_p = np.zeros((n, N_SLOT, width), dtype=np.float64)
    wide_l = np.zeros((n, N_SLOT, width), dtype=bool)
    wide_p[..., :size], wide_p[..., size] = prob, ~active
    wide_l[..., :size], wide_l[..., size] = legal, ~active

    can = np.zeros((n, N_SLOT), dtype=bool)
    q = np.zeros((n, N_SLOT), dtype=np.float64)
    if with_mega:
        can = _mega_possible(batch, active)
        mega = np.asarray(pred["mega"], dtype=np.float64)
        if mega.shape != (n, N_SLOT) or not np.isfinite(mega).all():
            raise ValueError(f"mega {mega.shape} does not fit or is not finite")
        q = np.where(can, np.clip(mega, 0.0, 1.0), 0.0)
    wide = can.any(-1)
    place = _truth_index(truth, n, width, with_mega, wide)

    # An all-ones coupling is the plain product: it takes the plain path.
    maps = None
    if coupling is not None and not bool(coupling.is_identity):
        maps = coupling.reply_maps(batch, move_intent)
        if maps is None:
            raise ValueError("the coupling cannot be applied to this batch")
        if maps[1].shape != (n, N_SLOT, width) or maps[0].shape != (n,):
            raise ValueError("the coupling's reply classes do not fit the batch")

    # No list is longer than the table it is cut from.
    k = min(k, (N_MEGA_STATE if with_mega else 1) * width * width)
    out = _empty(k, with_mega, n, n_cand, coupled=maps is not None)
    for states, rows in (
        (1, np.nonzero(~wide)[0]),
        (N_MEGA_STATE, np.nonzero(wide)[0]),
    ):
        step = max(1, int(chunk_bytes) // (states * width * width * 8))
        if maps is not None:
            step = max(1, step // 3)  # the factor and the coupled copy
        for start in range(0, rows.size, step):
            pick = rows[start : start + step]
            factor = None
            if maps is not None:
                group, kinds, table = maps
                factor = table[
                    group[pick][:, None, None],
                    kinds[pick, 0][:, :, None],
                    kinds[pick, 1][:, None, :],
                ]
            made = _chunk(
                wide_p[pick, 0],
                wide_p[pick, 1],
                wide_l[pick, 0],
                wide_l[pick, 1],
                q[pick] if states > 1 else None,
                can[pick] if states > 1 else None,
                out.k,
                place[pick],
                factor,
            )
            for name, values in made.items():
                getattr(out, name)[pick] = values
    return out


def joint_replies(
    pred: Mapping[str, np.ndarray],
    batch: Mapping[str, np.ndarray],
    k: int = DEFAULT_K,
    *,
    mega: bool = False,
    truth: Mapping[str, np.ndarray] | None = None,
    chunk_bytes: int = CHUNK_BYTES,
    coupling: Any = None,
    move_intent: Any = None,
) -> JointReplies:
    """The ``k`` most probable joint replies of every example, and the truth's rank.

    ``pred`` is a predictor's output (``action`` / ``target`` / ``mega``) for a
    batch, or for one example without the leading axis (the runtime's
    ``Forecast.raw``); ``batch`` holds the masks it was predicted on, again
    for a batch or for one example: ``action_mask`` and ``cand_tmask``, and
    for ``mega=True`` either ``mega_possible`` ``[N, 2]`` or ``act_mon`` with
    ``mon_flag``. With ``mega=True`` a joint reply also says which slot, if
    any, Mega-evolves. ``truth`` is ``true_replies(batch)`` or any mapping with
    ``reply`` ``[N, 2]`` (and ``mega`` ``[N]`` when the Mega bit is on); only
    those two keys are read. See the module text for how the joint is built,
    which is the model's own factorisation, and for the order of ties.

    ``k`` is a whole number (below 1 is read as 1); the list is cut to the
    size of the joint table when ``k`` is larger (``JointReplies.k`` says how
    long it is). ``mega`` is a plain True or False.

    ``coupling`` (a ``coupling.PairCoupling``; None = the plain product, the
    default) multiplies every pair of replies by its number for the pair's
    two reply classes and the example's bucket; ``batch`` must then also hold
    ``cand_move``, ``turn`` and ``foe_mon`` and ``move_intent`` be the
    featurizer's ``tables.move_intent`` (or ``batch`` hold ``pair_class`` and
    ``pair_bucket`` already made). An all-ones coupling is the plain product.

    Never raises: an input that does not fit (a ``k`` or a ``mega`` of another
    kind included) gives ``JointReplies`` with ``n == 0`` and a count in
    ``COUNTERS``.
    """
    # Read before the guarded part, so the failure value never depends on what
    # failed: an array handed over as ``mega`` has no truth value at all.
    with_mega = _flag(mega)
    width = _width(k)
    if with_mega is None or width is None:
        _failed("joint_replies", TypeError("mega is not a flag or k not a number"))
        return _empty(DEFAULT_K if width is None else width, bool(with_mega))
    try:
        if coupling is None:
            return _joint(pred, batch, width, with_mega, truth, chunk_bytes)
        return _joint(
            pred, batch, width, with_mega, truth, chunk_bytes, coupling, move_intent
        )
    except Exception as exc:
        _failed("joint_replies", exc)
        return _empty(width, with_mega)


def _flag(value: Any) -> bool | None:
    """``value`` as a plain flag; None for anything that is not one."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return None


def _width(value: Any) -> int | None:
    """``value`` as a list length: a whole number, at least 1; None when it is
    not a number.

    A length above the largest table any batch can have is as good as that
    table's size, so it is capped here: the failure value stays small too.
    """
    try:
        return max(1, min(int(value), MAX_K))
    except Exception:
        return None
