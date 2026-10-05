"""Event calibration: the maps, the rescaling, the fit, the payload, the script.

Library tests run on synthetic action distributions whose true event rates are
known, and on the hand-written game of ``test_oppmodel_model`` driven through
the real featurizer and a small network. The script tests build a small
dataset whose labels are drawn from a recalibrated version of the model's own
predictions, so the fit has something real to find.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from evaluation import oppmodel_scorecard as S
from training import calibrate_oppmodel as K
from training import train_oppmodel as T
from unit_tests import test_oppmodel_model as TM
from unit_tests import test_oppmodel_scorecard as TS
from vgc_bench.src.oppmodel import artifact as A
from vgc_bench.src.oppmodel import calibration as C
from vgc_bench.src.oppmodel import events as E
from vgc_bench.src.oppmodel import features as F
from vgc_bench.src.oppmodel import joint as J
from vgc_bench.src.oppmodel import model as M
from vgc_bench.src.oppmodel import runtime as R

ROOT = Path(__file__).resolve().parents[1]
REAL_ARTIFACT = ROOT / "results_oppmodel" / "oppnet_v2_blind" / "artifact.pt"
REAL_DATASET = ROOT / "results_oppmodel" / "v2_feed"
N_CAND = F.N_CAND_DEFAULT
N_ACTION = F.action_size(N_CAND)
SPLITS = ["train", "val", "test", "ladder_holdout"]
# What the synthetic model does to the true shares, and so what a fit must undo.
OVER_SWITCH = (1.6, 0.4)
OVER_PROTECT = (1.4, 0.3)
# The truth of the script's dataset, as maps of the small network's shares: the
# network says too much about switches and too little about Protect.
TRUE_SWITCH = C.EventMap(C.MAP_LOGISTIC, 0.6, -0.3)
TRUE_PROTECT = C.EventMap(C.MAP_LOGISTIC, 1.5, 2.0)


def logit(p: Any) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return np.log(q) - np.log1p(-q)


def sigmoid(x: Any) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def sample_actions(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One action index per slot, drawn from ``probs`` ``[N, 2, A]``."""
    cum = np.cumsum(np.asarray(probs, dtype=np.float64), -1)
    cum = cum / np.maximum(cum[..., -1:], 1e-300)
    draw = rng.random((*probs.shape[:2], 1))
    return np.minimum((draw > cum).sum(-1), probs.shape[-1] - 1)


def synthetic(
    n: int, seed: int, unknown: float = 0.08, shift: np.ndarray | None = None
) -> tuple[F.Batch, np.ndarray, np.ndarray]:
    """(batch with labels, the over-confident model's action, the true action).

    Every slot draws its switch share and its Protect share of the non-switch
    mass; the model reports both pushed outwards on the logit scale
    (``OVER_SWITCH`` / ``OVER_PROTECT``); the label is drawn from the truth.
    Some slots are empty, some have no legal switch, some no Protect-family
    candidate, and for a share of them the log tells nothing. ``shift``
    ``[n, 2]`` is added to the logit of the model's switch share (it draws no
    random number: without it the batch is what it always was).
    """
    rng = np.random.default_rng(seed)
    active = rng.random((n, 2)) > 0.05
    valid = (np.arange(N_CAND) < rng.integers(2, 7, (n, 2, 1))) & active[..., None]
    has_guard = (rng.random((n, 2)) < 0.8) & active
    flags = np.where(valid, F.CAND_VALID, 0).astype(np.uint16)
    flags[..., 0] |= np.where(has_guard, F.CAND_PROTECT, 0).astype(np.uint16)
    pointers = (rng.random((n, 2, F.N_ROSTER)) < 0.35) & active[..., None]
    pointers &= (rng.random((n, 2)) > 0.2)[..., None]
    can_switch = pointers.any(-1)
    mask = np.zeros((n, 2, N_ACTION), dtype=bool)
    mask[..., :N_CAND] = valid
    mask[..., N_CAND] = active
    mask[..., N_CAND + 1 :] = pointers

    def spread(legal: np.ndarray) -> np.ndarray:
        weight = rng.random(legal.shape) * legal
        total = weight.sum(-1, keepdims=True)
        return np.divide(weight, total, out=np.zeros_like(weight), where=total > 0)

    switch_part = np.zeros((n, 2, N_ACTION))
    switch_part[..., N_CAND + 1 :] = spread(pointers)
    guard_part = np.zeros((n, 2, N_ACTION))
    guard_part[..., 0] = has_guard
    rest = mask.copy()
    rest[..., N_CAND + 1 :] = False
    rest[..., 0] &= ~has_guard
    rest_part = spread(rest)
    s_true = np.where(can_switch, rng.beta(1.0, 5.0, (n, 2)), 0.0)
    r_true = np.where(has_guard, rng.beta(1.2, 4.0, (n, 2)), 0.0)
    lean = 0.0 if shift is None else np.asarray(shift, dtype=np.float64)
    s_model = np.where(
        can_switch, sigmoid(OVER_SWITCH[0] * logit(s_true) + OVER_SWITCH[1] + lean), 0.0
    )
    r_model = np.where(
        has_guard, sigmoid(OVER_PROTECT[0] * logit(r_true) + OVER_PROTECT[1]), 0.0
    )

    def mix(s: np.ndarray, r: np.ndarray) -> np.ndarray:
        s, r = s[..., None], r[..., None]
        return s * switch_part + (1 - s) * (r * guard_part + (1 - r) * rest_part)

    truth, model = mix(s_true, r_true), mix(s_model, r_model)
    picked = sample_actions(truth, rng)
    known = active & (rng.random((n, 2)) >= unknown)
    y_flag = np.zeros((n, 2, F.N_Y_FLAG), dtype=np.uint8)
    y_flag[..., F.Y_SWITCHED] = known & (picked > N_CAND)
    y_flag[..., F.Y_SWITCH_KNOWN] = known
    y_flag[..., F.Y_PROTECTED] = known & (picked == 0) & has_guard
    y_flag[..., F.Y_PROTECT_KNOWN] = known
    batch = {
        "action_mask": mask.astype(np.uint8),
        "cand_flag": flags,
        "act_mon": np.where(active, 0, -1).astype(np.int8),
        "y_flag": y_flag,
    }
    return batch, model, truth


# What the synthetic model adds to the logit of its switch share on turn 1, on
# a Pokemon's first turn on the field and after a Protect: nothing after a
# Protect. A fit must find the first two and undo them.
CONTEXT_SHIFT = (0.8, 0.6, 0.0)


def synthetic_context(
    n: int, seed: int, offsets: tuple[float, float, float] = CONTEXT_SHIFT
) -> tuple[F.Batch, np.ndarray, np.ndarray]:
    """``synthetic`` with the public context arrays, and a model whose switch
    share is further off by ``offsets`` (logit) where the three flags are set."""
    rng = np.random.default_rng(seed + 1000)
    turn = np.where(rng.random(n) < 0.15, 1, rng.integers(2, 12, n)).astype(np.uint8)
    lead = np.broadcast_to(turn[:, None] == 1, (n, 2))
    first = lead | (rng.random((n, 2)) < 0.25)
    guarded = ~lead & (rng.random((n, 2)) < 0.12)
    shift = offsets[0] * lead + offsets[1] * first + offsets[2] * guarded
    data, model, truth = synthetic(n, seed, shift=shift)
    active = data["act_mon"] >= 0
    data["act_mon"] = np.where(active, np.arange(2), -1).astype(np.int8)
    mon_flag = np.zeros((n, F.N_MON, F.N_MON_FLAG), dtype=np.uint8)
    mon_flag[:, :2, F.FLAG_FIRST_TURN] = first & active
    mon_flag[:, :2, F.FLAG_PROTECTED_LAST] = guarded & active
    data["turn"], data["mon_flag"] = turn, mon_flag
    return data, model, truth


@pytest.fixture(scope="module")
def fz() -> F.Featurizer:
    return F.Featurizer.build(TM.repertoire())


@pytest.fixture(scope="module")
def batch(fz: F.Featurizer) -> F.Batch:
    return TM.game_batch(fz)


@pytest.fixture(scope="module")
def predictor(fz: F.Featurizer) -> M.OppNetPredictor:
    return M.OppNetPredictor(
        TM.jolt(TM.small_net(fz)), fz, name="unit", action_temperature=1.3
    )


def some_calibration(temperature: float | None = 1.3) -> C.EventCalibration:
    return C.EventCalibration(
        C.EventMap(C.MAP_LOGISTIC, 0.7, -0.4),
        C.EventMap(
            C.MAP_ISOTONIC, knots_x=(0.0, 0.2, 0.7, 1.0), knots_y=(0.0, 0.1, 0.5, 1.0)
        ),
        temperature,
        {"dataset_tag": "unit", "events": {C.EVENT_SWITCH: {"observed": 0.1}}},
    )


MAPS = [
    C.EventMap(),
    C.EventMap(C.MAP_LOGISTIC, 0.8, -0.3),
    C.EventMap(C.MAP_LOGISTIC, 2.5, 1.0),
    C.EventMap(
        C.MAP_ISOTONIC, knots_x=(0.0, 0.1, 0.5, 1.0), knots_y=(0.0, 0.05, 0.3, 1.0)
    ),
    C.EventMap(
        C.MAP_ISOTONIC, knots_x=(0.0, 0.4, 0.6, 1.0), knots_y=(0.0, 0.5, 0.5, 1.0)
    ),
]


# --- the maps -------------------------------------------------------------------


@pytest.mark.parametrize("found", MAPS, ids=lambda m: f"{m.kind}-{m.slope}")
def test_a_map_is_monotone_and_keeps_zero_and_one(found: C.EventMap):
    grid = np.concatenate(
        [
            [0.0, 1e-300, 1e-30, 1e-12],
            np.linspace(1e-6, 1 - 1e-6, 2001),
            [1 - 1e-12, 1.0],
        ]
    )
    out = found(grid)
    assert out.dtype == np.float64 and out.shape == grid.shape
    assert out[0] == 0.0 and out[-1] == 1.0
    assert ((out >= 0.0) & (out <= 1.0)).all()
    assert (np.diff(out[1:]) >= 0.0).all()
    # Out of range clips to the ends (the identity returns what it was given);
    # a non-finite entry comes back as it is.
    odd = found(np.array([-0.5, 1.5, np.nan]))
    assert np.isnan(odd[2])
    assert (odd[0], odd[1]) == ((-0.5, 1.5) if found.is_identity else (0.0, 1.0))
    assert float(found(0.25)) == pytest.approx(float(found(np.array([0.25]))[0]))


def test_the_identity_map_returns_the_share_itself():
    grid = np.random.default_rng(0).random(50)
    assert np.array_equal(C.EventMap()(grid), grid)
    assert C.EventMap().is_identity and not MAPS[1].is_identity
    # Fields a kind does not use are reset, so equal maps compare equal.
    assert C.EventMap(C.MAP_IDENTITY, 3.0, 2.0, (0.0, 1.0), (0.0, 1.0)) == C.EventMap()
    assert C.EventMap(C.MAP_LOGISTIC, 0.8, 0.1, (0.0, 1.0), (0.0, 1.0)) == C.EventMap(
        C.MAP_LOGISTIC, 0.8, 0.1
    )


def test_a_logistic_map_is_the_sigmoid_of_the_scaled_logit():
    found = C.EventMap(C.MAP_LOGISTIC, 0.8, -0.3)
    grid = np.linspace(0.01, 0.99, 99)
    assert np.allclose(found(grid), sigmoid(0.8 * logit(grid) - 0.3), atol=1e-12)
    knots = C.EventMap(C.MAP_ISOTONIC, knots_x=(0.0, 0.5, 1.0), knots_y=(0.0, 0.2, 1.0))
    assert np.allclose(knots([0.25, 0.5, 0.75]), [0.1, 0.2, 0.6])


@pytest.mark.parametrize(
    "arguments",
    [
        ("something",),
        (C.MAP_LOGISTIC, 0.0, 0.0),
        (C.MAP_LOGISTIC, -1.0, 0.0),
        (C.MAP_LOGISTIC, float("nan"), 0.0),
        (C.MAP_LOGISTIC, 1.0, float("inf")),
        (C.MAP_LOGISTIC, "x", 0.0),
        (C.MAP_ISOTONIC,),
        (C.MAP_ISOTONIC, 1.0, 0.0, (0.0, 1.0), (0.0,)),
        (C.MAP_ISOTONIC, 1.0, 0.0, (0.1, 1.0), (0.0, 1.0)),
        (C.MAP_ISOTONIC, 1.0, 0.0, (0.0, 0.9), (0.0, 1.0)),
        (C.MAP_ISOTONIC, 1.0, 0.0, (0.0, 1.0), (0.1, 1.0)),
        (C.MAP_ISOTONIC, 1.0, 0.0, (0.0, 0.5, 0.5, 1.0), (0.0, 0.2, 0.3, 1.0)),
        (C.MAP_ISOTONIC, 1.0, 0.0, (0.0, 0.3, 0.6, 1.0), (0.0, 0.5, 0.4, 1.0)),
        (C.MAP_ISOTONIC, 1.0, 0.0, (0.0, 0.5, 1.0), (0.0, float("nan"), 1.0)),
    ],
)
def test_parameters_that_are_not_a_monotone_map_raise(arguments: tuple[Any, ...]):
    with pytest.raises(ValueError):
        C.EventMap(*arguments)


TERMS = {C.CONTEXT_TURN_ONE: -0.5, C.CONTEXT_PROTECTED_LAST: 0.25}


def test_a_logistic_map_adds_a_term_to_the_logit_of_flagged_slots():
    found = C.EventMap(C.MAP_LOGISTIC, 0.8, -0.3, terms=tuple(TERMS.items()))
    assert found.needs_context and not MAPS[1].needs_context
    assert dict(found.terms) == TERMS
    grid = np.linspace(0.01, 0.99, 99)
    flags = np.zeros((99, len(C.CONTEXT)), dtype=bool)
    turn_one, last = (C.CONTEXT.index(name) for name in TERMS)
    assert np.allclose(found(grid, flags), MAPS[1](grid), atol=1e-12)  # no flag set
    flags[:, turn_one] = True
    assert np.allclose(
        found(grid, flags), sigmoid(0.8 * logit(grid) - 0.3 - 0.5), atol=1e-12
    )
    flags[:, last] = True  # both terms add up
    assert np.allclose(
        found(grid, flags), sigmoid(0.8 * logit(grid) - 0.3 - 0.25), atol=1e-12
    )
    flags[:, C.CONTEXT.index(C.CONTEXT_FIRST_TURN)] = True  # no term: no change
    assert np.allclose(
        found(grid, flags), sigmoid(0.8 * logit(grid) - 0.3 - 0.25), atol=1e-12
    )
    # In every context: monotone in the share, 0 stays 0 and 1 stays 1.
    ends = np.concatenate([[0.0, 1e-300], np.linspace(1e-6, 1 - 1e-6, 501), [1.0]])
    for row in ((0, 0, 0), (1, 0, 0), (1, 1, 1), (0, 0, 1)):
        context = np.tile(np.array(row, dtype=np.uint8), (ends.size, 1))
        out = found(ends, context)
        assert out[0] == 0.0 and out[-1] == 1.0 and (np.diff(out[1:]) >= 0).all()
    # Any leading shape, as rescale hands it over.
    shares = np.random.default_rng(0).random((7, 2))
    context = np.random.default_rng(1).random((7, 2, len(C.CONTEXT))) < 0.5
    want = sigmoid(
        0.8 * logit(shares)
        - 0.3
        - 0.5 * context[..., turn_one]
        + 0.25 * context[..., last]
    )
    assert np.allclose(found(shares, context), want, atol=1e-12)
    assert float(found(0.3, np.zeros(3))) == pytest.approx(float(MAPS[1](0.3)))


def test_a_map_with_terms_is_never_applied_without_its_context():
    """Dropping the terms silently would be an uncalibrated answer under a
    calibrated name: a missing or misshapen context is an error."""
    found = C.EventMap(C.MAP_LOGISTIC, 0.8, -0.3, terms=tuple(TERMS.items()))
    shares = np.full((4, 2), 0.3)
    for context in (None, np.zeros((4, 2)), np.zeros((4, 2, 2)), np.zeros((3, 2, 3))):
        with pytest.raises(ValueError, match="context"):
            found(shares, context)
    # A map without terms ignores whatever it is given.
    assert np.array_equal(MAPS[1](shares, "anything"), MAPS[1](shares))
    assert np.array_equal(MAPS[0](shares, None), shares)
    assert np.array_equal(MAPS[3](shares, np.zeros(5)), MAPS[3](shares))


def test_terms_are_stored_in_one_form_and_only_for_a_logistic_map():
    a = C.EventMap(C.MAP_LOGISTIC, 0.8, -0.3, terms=tuple(TERMS.items()))
    # a mapping, another order, and a term of exactly 0: the same map
    b = C.EventMap(
        C.MAP_LOGISTIC,
        0.8,
        -0.3,
        terms={  # type: ignore[arg-type]
            C.CONTEXT_PROTECTED_LAST: 0.25,
            C.CONTEXT_FIRST_TURN: 0.0,
            C.CONTEXT_TURN_ONE: -0.5,
        },
    )
    assert a == b and a.terms == b.terms == tuple(TERMS.items())
    assert a != MAPS[1] and C.EventMap(C.MAP_LOGISTIC, 0.8, -0.3, terms=()) == MAPS[1]
    payload = a.to_payload()
    assert payload["terms"] == TERMS and MAPS[1].to_payload()["terms"] == {}
    assert C.EventMap.from_payload(payload) == a
    json.dumps(payload)
    # a payload written before terms existed has no such key
    old = {key: value for key, value in MAPS[1].to_payload().items() if key != "terms"}
    assert C.EventMap.from_payload(old) == MAPS[1]
    # the other kinds carry none
    assert C.EventMap(C.MAP_IDENTITY, terms=tuple(TERMS.items())) == C.EventMap()
    knots = C.EventMap(
        C.MAP_ISOTONIC,
        knots_x=(0.0, 0.5, 1.0),
        knots_y=(0.0, 0.2, 1.0),
        terms=tuple(TERMS.items()),
    )
    assert knots.terms == () and not knots.needs_context
    for bad in (
        (("no_such_flag", 0.1),),
        ((C.CONTEXT_TURN_ONE, float("nan")),),
        ((C.CONTEXT_TURN_ONE, float("inf")),),
        ((C.CONTEXT_TURN_ONE, "x"),),
        ((C.CONTEXT_TURN_ONE, 0.1), (C.CONTEXT_TURN_ONE, 0.2)),
        "turn_one",
        7,
    ):
        with pytest.raises(ValueError):
            C.EventMap(C.MAP_LOGISTIC, 0.8, -0.3, terms=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        C.EventMap.from_payload({**payload, "terms": {"no_such_flag": 1.0}})


def test_event_context_reads_three_public_flags_of_the_slot():
    turn = np.array([1, 2, 7, 1], dtype=np.uint8)
    act_mon = np.array([[0, 1], [3, -1], [1, 0], [-1, 2]], dtype=np.int8)
    mon_flag = np.zeros((4, F.N_MON, F.N_MON_FLAG), dtype=np.uint8)
    mon_flag[0, 0, F.FLAG_FIRST_TURN] = 1  # turn 1, slot a
    mon_flag[0, 1, F.FLAG_FIRST_TURN] = 1
    mon_flag[1, 3, F.FLAG_PROTECTED_LAST] = 1  # the Pokemon in slot a (row 3)
    mon_flag[1, 0, F.FLAG_FIRST_TURN] = 1  # a benched Pokemon: nobody's flag
    mon_flag[2, 0, F.FLAG_FIRST_TURN] = 1  # the Pokemon in slot B (row 0)
    mon_flag[2, 1, F.FLAG_PROTECTED_LAST] = 1
    mon_flag[3, 0, F.FLAG_FIRST_TURN] = 1  # row 0 is in no slot of example 3
    batch = Recording({"turn": turn, "act_mon": act_mon, "mon_flag": mon_flag})
    batch["y_flag"] = np.ones((4, 2, F.N_Y_FLAG))  # a label array it must not read
    made = C.event_context(batch)
    assert made.shape == (4, 2, len(C.CONTEXT)) and made.dtype == bool
    assert batch.read == set(C.CONTEXT_ARRAYS) == {"turn", "act_mon", "mon_flag"}
    one, first, last = (
        C.CONTEXT.index(name)
        for name in (C.CONTEXT_TURN_ONE, C.CONTEXT_FIRST_TURN, C.CONTEXT_PROTECTED_LAST)
    )
    assert made[..., one].tolist() == [[True, True], [False, False], [False, False]] + [
        [False, True]  # the empty slot carries no flag, turn 1 or not
    ]
    assert made[..., first].tolist() == [
        [True, True],
        [False, False],
        [False, True],
        [False, False],
    ]
    assert made[..., last].tolist() == [
        [False, False],
        [True, False],
        [True, False],
        [False, False],
    ]
    # one example without the leading axis: the same flags
    single = {name: batch[name][2] for name in C.CONTEXT_ARRAYS}
    assert np.array_equal(C.event_context(single), made[2])
    for name in C.CONTEXT_ARRAYS:
        with pytest.raises(KeyError):
            C.event_context({k: v for k, v in batch.items() if k != name})
    for bad in (
        {**batch, "turn": turn[:3]},
        {**batch, "act_mon": act_mon[:, :1]},
        {**batch, "mon_flag": mon_flag[:, :, : F.FLAG_FIRST_TURN]},
        {**batch, "mon_flag": mon_flag[0]},
    ):
        with pytest.raises(ValueError):
            C.event_context(bad)


def test_bad_calibrations_raise_value_error():
    for arguments in (
        ("switch", C.EventMap()),
        (C.EventMap(), None),
        (C.EventMap(), C.EventMap(), 0.0),
        (C.EventMap(), C.EventMap(), float("nan")),
        (C.EventMap(), C.EventMap(), "warm"),
        (C.EventMap(), C.EventMap(), None, ["not", "a", "mapping"]),
        (C.EventMap(), C.EventMap(), None, {"an object": object()}),
        (C.EventMap(), C.EventMap(), None, {1: "a key that is not a string"}),
    ):
        with pytest.raises(ValueError):
            C.EventCalibration(*arguments)  # type: ignore[arg-type]


# --- the rescaling --------------------------------------------------------------


def group_masks(data: F.Batch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(switch pointers, Protect-family candidates, the other moves), legal ones."""
    mask = np.asarray(data["action_mask"]).astype(bool)
    flags = np.asarray(data["cand_flag"]).astype(np.int64)
    switch = mask.copy()
    switch[..., : N_CAND + 1] = False
    guard = np.zeros_like(mask)
    guard[..., :N_CAND] = (
        mask[..., :N_CAND]
        & ((flags & F.CAND_VALID) > 0)
        & ((flags & F.CAND_PROTECT) > 0)
    )
    return switch, guard, mask & ~switch & ~guard


def check_rescaling(data: F.Batch, action: np.ndarray, found: C.EventCalibration):
    mask = np.asarray(data["action_mask"]).astype(bool)
    before = np.asarray(action, dtype=np.float64)
    after = found.apply(action, data["action_mask"], data["cand_flag"])
    assert after.shape == before.shape and after.dtype == np.float64
    # Still a distribution over the legal actions.
    assert np.allclose(after.sum(-1), before.sum(-1), atol=1e-12)
    assert (after >= 0.0).all() and not after[~mask].any()
    switch, guard, rest = group_masks(data)
    old = C.event_shares(before, data["action_mask"], data["cand_flag"])
    new = C.event_shares(after, data["action_mask"], data["cand_flag"])
    s0, s1 = (before * switch).sum(-1), (after * switch).sum(-1)
    g0, g1 = (before * guard).sum(-1), (after * guard).sum(-1)
    r0, r1 = (before * rest).sum(-1), (after * rest).sum(-1)
    assert np.allclose(old.switch, s0) and np.allclose(new.switch, s1)
    # Each event share is the map of the old one, where the group holds mass.
    moved_s = (s0 > 0) & (g0 + r0 > 0)
    assert np.allclose(new.switch[moved_s], found.switch(old.switch)[moved_s])
    assert np.array_equal(s1[~moved_s], s0[~moved_s])
    moved_g = (g0 > 0) & (r0 > 0)
    assert np.allclose(
        new.protect_share[moved_g], found.protect(old.protect_share)[moved_g]
    )
    assert np.allclose(new.protect_share[~moved_g], old.protect_share[~moved_g])
    assert np.allclose(new.protect, (1.0 - new.switch) * new.protect_share)
    # Inside a group nothing moves: destinations, other moves, Protect variants.
    for group, was, now in ((switch, s0, s1), (guard, g0, g1), (rest, r0, r1)):
        held = (was > 0) & (now > 0)
        want = (before * group)[held] / was[held][:, None]
        got = (after * group)[held] / now[held][:, None]
        assert np.allclose(got, want, atol=1e-12)
    return before, after, old, new


def test_rescaling_keeps_normalisation_and_legality():
    data, model, _ = synthetic(4000, seed=1)
    for found in (
        some_calibration(None),
        C.EventCalibration(MAPS[2], MAPS[1]),
        C.EventCalibration(MAPS[3], MAPS[4]),
    ):
        before, after, old, new = check_rescaling(data, model, found)
        active = np.asarray(data["act_mon"]) >= 0
        assert np.allclose(after.sum(-1)[active], 1.0, atol=1e-9)
        assert not after[~active].any()
        assert not np.allclose(after, before, atol=1e-3)
    # The same on float32 input, as a predictor hands it over.
    check_rescaling(data, model.astype(np.float32), some_calibration(None))


def test_rescaling_on_real_features(batch: F.Batch, predictor: M.OppNetPredictor):
    pred = predictor.predict(batch)
    before, after, old, new = check_rescaling(batch, pred["action"], some_calibration())
    assert (batch["act_mon"] < 0).any()  # an empty slot is in the batch
    assert not after[batch["act_mon"] < 0].any()
    assert old.switch_possible.any() and not old.switch_possible.all()
    assert old.protect_possible.any()
    scored = dict(pred)
    scored["action"] = after.astype(np.float32)
    norm = F.normalize_prediction(scored, batch)
    assert np.abs(norm["action"] - scored["action"]).max() < 1e-6
    # The events a guard reads are the calibrated shares.
    events = F.event_probs(scored, batch)
    assert np.allclose(events[E.INTENT_SWITCH], new.switch, atol=1e-6)
    assert np.allclose(events[E.INTENT_PROTECT], new.protect, atol=1e-6)


def test_impossible_groups_are_left_alone():
    data, model, _ = synthetic(4000, seed=2)
    found = C.EventCalibration(MAPS[2], MAPS[1])
    after = found.apply(model, data["action_mask"], data["cand_flag"])
    switch, guard, rest = group_masks(data)
    can_switch, has_guard = switch.any(-1), guard.any(-1)
    active = np.asarray(data["act_mon"]) >= 0
    neither = active & ~can_switch & ~has_guard
    only_guard = active & ~can_switch & has_guard
    only_switch = active & can_switch & ~has_guard
    assert neither.sum() > 20 and only_guard.sum() > 20 and only_switch.sum() > 20
    # No legal switch and no Protect-family candidate: bit for bit the input.
    assert np.array_equal(after[neither], model[neither])
    assert np.array_equal(after[~active], model[~active])
    # No legal switch: the switch pointers stay at zero and the non-switch
    # mass stays 1; only the Protect split inside it moves.
    assert not after[only_guard][:, N_CAND + 1 :].any()
    assert np.allclose(after[only_guard][:, : N_CAND + 1].sum(-1), 1.0)
    assert not np.allclose(after[only_guard], model[only_guard], atol=1e-3)
    # No Protect-family candidate: every move is scaled by one common factor.
    ratio = np.divide(
        after[only_switch],
        model[only_switch],
        out=np.full_like(after[only_switch], np.nan),
        where=model[only_switch] > 0,
    )[:, : N_CAND + 1]
    assert np.allclose(np.nanmax(ratio, -1), np.nanmin(ratio, -1))
    # A slot whose mass sits wholly on one group is left alone too.
    one = np.zeros((1, 2, N_ACTION))
    mask = np.zeros((1, 2, N_ACTION), dtype=np.uint8)
    flags = np.zeros((1, 2, N_CAND), dtype=np.uint16)
    flags[0, :, 0] = F.CAND_VALID | F.CAND_PROTECT
    flags[0, :, 1] = F.CAND_VALID
    mask[0, :, [0, 1, N_CAND, N_CAND + 2]] = 1
    one[0, 0, N_CAND + 2] = 1.0  # certain to switch
    one[0, 1, 0] = 1.0  # certain to protect
    assert np.array_equal(found.apply(one, mask, flags), one)


def test_the_identity_calibration_changes_nothing(
    batch: F.Batch, predictor: M.OppNetPredictor
):
    data, model, _ = synthetic(500, seed=3)
    nothing = C.EventCalibration()
    assert nothing.is_identity and not some_calibration().is_identity
    for action in (model, model.astype(np.float32)):
        out = nothing.apply(action, data["action_mask"], data["cand_flag"])
        assert out is not action and out.dtype == action.dtype
        assert np.array_equal(out, action)
    # A logistic map with slope 1 and bias 0 is the identity up to rounding.
    unit = C.EventCalibration(
        C.EventMap(C.MAP_LOGISTIC, 1.0, 0.0), C.EventMap(C.MAP_LOGISTIC, 1.0, 0.0)
    )
    out = unit.apply(model, data["action_mask"], data["cand_flag"])
    assert np.allclose(out, model, atol=1e-12)
    # In a predictor: the same arrays, and the metadata says nothing is applied.
    plain = predictor.predict(batch)
    twin = predictor.with_event_calibration(C.EventCalibration(action_temperature=1.3))
    same = twin.predict(batch)
    assert all(np.array_equal(plain[name], same[name]) for name in plain)
    assert not twin.event_calibrated and not predictor.event_calibrated
    assert twin.describe()["event_calibration"]["applied"] is False
    assert predictor.describe()["event_calibration"] is None


def test_apply_never_raises_and_apply_checked_does():
    data, model, _ = synthetic(50, seed=4)
    found = some_calibration(None)
    mask, flags = data["action_mask"], data["cand_flag"]
    bad_inputs = [
        (model[:, :, :5], mask, flags),
        (model, mask[:10], flags),
        (model, mask, flags[:, :, :4]),
        (np.full_like(model, np.nan), mask, flags),
        (None, mask, flags),
        ("text", mask, flags),
        (model, None, None),
    ]
    for action, action_mask, cand_flag in bad_inputs:
        before = sum(C.COUNTERS.values())
        out = found.apply(action, action_mask, cand_flag)
        assert out is action
        assert sum(C.COUNTERS.values()) == before + 1
        with pytest.raises((ValueError, TypeError)):
            found.apply_checked(action, action_mask, cand_flag)
    before = sum(C.COUNTERS.values())
    found.apply(model, mask, flags)
    assert sum(C.COUNTERS.values()) == before


def test_event_shares_and_labels_are_the_scorecards(
    batch: F.Batch, predictor: M.OppNetPredictor
):
    """The fit counts the rows the scorecard counts and reads the event
    probabilities ``features.event_probs`` reports."""
    pred = predictor.predict(batch)
    shares = C.event_shares(pred["action"], batch["action_mask"], batch["cand_flag"])
    theirs = S.event_predictions(pred, batch)
    assert np.allclose(shares.switch, theirs[S.EVENT_SWITCH], atol=1e-7)
    assert np.allclose(shares.protect, theirs[S.EVENT_PROTECT], atol=1e-7)
    possible = S.event_possible(batch)
    assert np.array_equal(shares.switch_possible, possible[S.EVENT_SWITCH])
    assert np.array_equal(shares.protect_possible, possible[S.EVENT_PROTECT])
    mine, card = C.event_labels(batch), S.event_labels(batch)
    assert C.EVENTS == (S.EVENT_SWITCH, S.EVENT_PROTECT)
    for event in C.EVENTS:
        assert np.array_equal(mine[event][0], card[event][0])
        assert np.array_equal(mine[event][1], card[event][1])
        assert mine[event][0].any()
    with pytest.raises(KeyError):
        C.event_labels(M.strip_labels(batch))


# --- fitting --------------------------------------------------------------------


def test_logistic_fit_recovers_a_known_map():
    rng = np.random.default_rng(5)
    share = rng.beta(1.0, 3.0, 60_000)
    truth = C.EventMap(C.MAP_LOGISTIC, 0.7, -0.4)
    outcome = rng.random(share.size) < truth(share)
    found = C.fit_logistic(share, outcome)
    assert found.kind == C.MAP_LOGISTIC
    assert found.slope == pytest.approx(0.7, abs=0.03)
    assert found.bias == pytest.approx(-0.4, abs=0.04)
    assert C.binary_nll(found(share), outcome) < C.binary_nll(share, outcome) - 0.005
    # It is the minimum of the event log loss: no nearby map is better.
    best = C.binary_nll(found(share), outcome)
    for d_slope, d_bias in ((0.02, 0.0), (-0.02, 0.0), (0.0, 0.02), (0.0, -0.02)):
        near = C.EventMap(C.MAP_LOGISTIC, found.slope + d_slope, found.bias + d_bias)
        assert C.binary_nll(near(share), outcome) > best
    # An intercept is fitted, so the mean prediction is the observed rate.
    assert float(found(share).mean()) == pytest.approx(float(outcome.mean()), abs=1e-6)
    optimize = pytest.importorskip("scipy.optimize")
    z = logit(share)

    def loss(theta: np.ndarray) -> float:
        return C.binary_nll(sigmoid(theta[0] * z + theta[1]), outcome, floor=1e-12)

    reference = optimize.minimize(loss, [1.0, 0.0], method="BFGS").x
    assert found.slope == pytest.approx(reference[0], abs=1e-3)
    assert found.bias == pytest.approx(reference[1], abs=1e-3)


def test_weights_count_as_repeated_rows():
    rng = np.random.default_rng(6)
    share = rng.beta(1.0, 3.0, 4000)
    outcome = rng.random(share.size) < share**1.3
    weight = rng.integers(1, 4, share.size)
    repeated = (np.repeat(share, weight), np.repeat(outcome, weight))
    first, second = C.fit_logistic(share, outcome, weight), C.fit_logistic(*repeated)
    assert first.slope == pytest.approx(second.slope, abs=1e-9)
    assert first.bias == pytest.approx(second.bias, abs=1e-9)
    assert C.binary_nll(share, outcome, weight) == pytest.approx(
        C.binary_nll(*repeated), abs=1e-12
    )


def test_isotonic_fit_follows_a_shape_no_logistic_map_has():
    rng = np.random.default_rng(7)
    share = rng.random(60_000)

    def truth(p: np.ndarray) -> np.ndarray:
        return np.where(p < 0.4, 0.05 + 0.1 * p, np.where(p < 0.7, 0.45, 0.9))

    outcome = rng.random(share.size) < truth(share)
    groups = np.arange(share.size) // 4
    iso = C.fit_isotonic(share, outcome)
    assert (
        iso.kind == C.MAP_ISOTONIC and 4 <= len(iso.knots_x) <= C.ISOTONIC_MAX_BINS + 2
    )
    grid = np.linspace(0.02, 0.98, 49)
    # Away from the two jumps, and from the last stretch that runs up to (1, 1).
    inner = (np.abs(grid - 0.4) > 0.04) & (np.abs(grid - 0.7) > 0.04) & (grid < 0.93)
    assert np.abs(iso(grid) - truth(grid))[inner].max() < 0.04
    chosen, record = C.select_map(share, outcome, groups)
    assert chosen == iso and record["chosen"] == C.MAP_ISOTONIC
    cross = record["cross_validated_nll"]
    assert cross[C.MAP_ISOTONIC] < cross[C.MAP_LOGISTIC] - C.ISOTONIC_MARGIN
    assert record["isotonic_better_folds"] >= 8 and record["folds"] == C.FOLDS
    sklearn = pytest.importorskip("sklearn.isotonic")
    reference = sklearn.IsotonicRegression(y_min=0.0, y_max=1.0).fit(share, outcome)
    assert np.abs(iso(grid) - reference.predict(grid))[inner].max() < 0.04


def test_the_simpler_map_is_preferred():
    rng = np.random.default_rng(8)
    share = rng.beta(1.0, 3.0, 40_000)
    groups = np.arange(share.size) // 4
    # The truth is a logistic map: isotonic may fit as well, it is not chosen.
    outcome = rng.random(share.size) < C.EventMap(C.MAP_LOGISTIC, 0.7, -0.4)(share)
    chosen, record = C.select_map(share, outcome, groups)
    assert chosen.kind == C.MAP_LOGISTIC and record["chosen"] == C.MAP_LOGISTIC
    assert (
        record["in_sample_nll"][C.MAP_LOGISTIC]
        < record["in_sample_nll"][C.MAP_IDENTITY]
    )
    assert set(record["fitted"]) == set(C.MAP_KINDS)
    # Already calibrated: whatever is chosen, it changes next to nothing.
    outcome = rng.random(share.size) < share
    chosen, record = C.select_map(share, outcome, groups)
    assert chosen.kind in (C.MAP_IDENTITY, C.MAP_LOGISTIC)
    grid = np.linspace(0.01, 0.99, 99)
    assert np.abs(chosen(grid) - grid).max() < 0.03


def test_isotonic_needs_both_the_margin_and_the_folds():
    rng = np.random.default_rng(14)
    share = rng.random(60_000)
    groups = np.arange(share.size) // 4
    steps = np.where(share < 0.4, 0.05 + 0.1 * share, np.where(share < 0.7, 0.45, 0.9))
    outcome = rng.random(share.size) < steps
    # Clearly better here: chosen by the rule as it stands ...
    chosen, record = C.select_map(share, outcome, groups)
    assert chosen.kind == C.MAP_ISOTONIC
    assert record["isotonic_gain_over_simpler"] > 10 * C.ISOTONIC_MARGIN
    assert record["isotonic_better_folds"] == record["folds"] == C.FOLDS
    # ... and by neither a margin nobody can reach nor more folds than there are.
    assert C.select_map(share, outcome, groups, margin=10.0)[0].kind == C.MAP_LOGISTIC
    assert (
        C.select_map(share, outcome, groups, fold_share=1.1)[0].kind == C.MAP_LOGISTIC
    )
    # Where the truth is a logistic map, only dropping both conditions lets it in.
    share = rng.beta(1.0, 3.0, 40_000)
    groups = np.arange(share.size) // 4
    outcome = rng.random(share.size) < C.EventMap(C.MAP_LOGISTIC, 0.7, -0.4)(share)
    assert C.select_map(share, outcome, groups)[0].kind == C.MAP_LOGISTIC
    assert (
        C.select_map(share, outcome, groups, margin=-10.0, fold_share=0.0)[0].kind
        == C.MAP_ISOTONIC
    )


def test_pooling_makes_the_binned_rates_monotone():
    levels, counts = C._pool(
        np.array([0.1, 0.3, 0.2, 0.5, 0.4, 0.4]),
        np.array([1.0, 1.0, 3.0, 2.0, 1.0, 1.0]),
    )
    assert counts == [1, 2, 3]
    assert levels == pytest.approx(
        [0.1, (0.3 + 3 * 0.2) / 4, (2 * 0.5 + 0.4 + 0.4) / 4]
    )
    levels, counts = C._pool(np.array([0.1, 0.2, 0.3]), np.ones(3))
    assert levels == [0.1, 0.2, 0.3] and counts == [1, 1, 1]
    levels, counts = C._pool(np.array([0.3, 0.2, 0.1]), np.ones(3))
    assert levels == pytest.approx([0.2]) and counts == [3]
    # A noisy sample whose bins are out of order: the knots are the pooled
    # blocks, each at its mean share, and nothing else.
    rng = np.random.default_rng(15)
    share = np.sort(rng.random(2000))
    outcome = rng.random(2000) < 0.2 + 0.3 * share
    found = C.fit_isotonic(share, outcome, min_bin=50, max_bins=40)
    edges = np.linspace(0, 2000, 41).round().astype(int)
    mean = np.array([share[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
    hits = np.array([outcome[a:b].sum() for a, b in zip(edges[:-1], edges[1:])])
    rate = (hits + C.ISOTONIC_PRIOR * mean) / (50 + C.ISOTONIC_PRIOR)
    assert (np.diff(rate) < 0).sum() > 5  # the bins do violate the order
    levels, counts = C._pool(rate, np.full(40, 50 + C.ISOTONIC_PRIOR))
    assert 2 < len(levels) < 40
    assert found.knots_y[1:-1] == pytest.approx(levels)
    starts = np.cumsum([0, *counts[:-1]])
    centres = [mean[a : a + n].mean() for a, n in zip(starts, counts)]
    assert found.knots_x[1:-1] == pytest.approx(centres)


def test_too_little_evidence_gives_the_identity():
    rng = np.random.default_rng(9)
    share = rng.random(400)
    few = np.zeros(400, dtype=bool)
    few[:10] = True  # ten events: fewer than MIN_EACH
    for outcome in (few, ~few, np.zeros(400, dtype=bool)):
        assert C.fit_logistic(share, outcome).is_identity
        assert C.fit_isotonic(share, outcome).is_identity
        chosen, record = C.select_map(share, outcome)
        assert chosen.is_identity and record["chosen"] == C.MAP_IDENTITY
    empty, record = C.select_map(np.zeros(0), np.zeros(0, dtype=bool))
    assert empty.is_identity and record["rows"] == 0
    # A share that points the wrong way is not turned around: no map.
    outcome = rng.random(400) < 1.0 - share
    assert C.fit_logistic(share, outcome).is_identity
    # One group only: no cross-validation, the logistic map when it fits.
    outcome = rng.random(400) < share**2
    chosen, record = C.select_map(share, outcome, np.zeros(400, dtype=np.int64))
    assert chosen.kind == C.MAP_LOGISTIC and record["folds"] == 0
    with pytest.raises(ValueError):
        C.fit_logistic(share, outcome[:10])
    with pytest.raises(ValueError):
        C.fit_logistic(share, np.full(400, 2))
    with pytest.raises(ValueError):
        C.select_map(share, outcome, np.zeros(3))


def test_folds_keep_a_battle_together():
    groups = np.repeat(np.arange(37), 5)
    fold, used = C._fold_index(groups, 10, seed=1)
    assert used == 10 and set(fold.tolist()) == set(range(10))
    for group in range(37):
        assert len(set(fold[groups == group].tolist())) == 1
    again, _ = C._fold_index(groups, 10, seed=1)
    other, _ = C._fold_index(groups, 10, seed=2)
    assert np.array_equal(fold, again) and not np.array_equal(fold, other)
    assert C._fold_index(np.zeros(8), 10, seed=1)[1] == 0
    assert C._fold_index(np.arange(3), 10, seed=1)[1] == 3


def test_an_overconfident_predictor_is_pulled_back_to_the_true_rate():
    data, model, truth = synthetic(20_000, seed=10)
    groups = np.arange(20_000) // 6
    found = C.fit_event_calibration(model, data, groups=groups, info={"note": "unit"})
    # The fit undoes what the synthetic model did to the true shares.
    assert found.switch.kind == C.MAP_LOGISTIC and found.protect.kind == C.MAP_LOGISTIC
    assert found.switch.slope == pytest.approx(1 / OVER_SWITCH[0], abs=0.04)
    assert found.switch.bias == pytest.approx(
        -OVER_SWITCH[1] / OVER_SWITCH[0], abs=0.06
    )
    assert found.protect.slope == pytest.approx(1 / OVER_PROTECT[0], abs=0.04)
    assert found.protect.bias == pytest.approx(
        -OVER_PROTECT[1] / OVER_PROTECT[0], abs=0.06
    )
    after = found.apply(model, data["action_mask"], data["cand_flag"])
    active = np.asarray(data["act_mon"]) >= 0
    assert (
        np.abs(model - truth)[active].mean() > 5 * np.abs(after - truth)[active].mean()
    )

    labels = C.event_labels(data)
    old = C.event_shares(model, data["action_mask"], data["cand_flag"])
    new = C.event_shares(after, data["action_mask"], data["cand_flag"])
    for event, threshold, before_p, after_p in (
        (C.EVENT_SWITCH, 0.35, old.switch, new.switch),
        (C.EVENT_SWITCH, 0.6, old.switch, new.switch),
        (C.EVENT_PROTECT, 0.5, old.protect, new.protect),
        (C.EVENT_PROTECT, 0.6, old.protect, new.protect),
    ):
        known, happened = labels[event]
        fired = known & (before_p >= threshold)
        assert fired.sum() > 300
        # Before: said far more than happened. After: the two agree, on the
        # slot-turns that fired before and on those that fire now.
        assert before_p[fired].mean() - happened[fired].mean() > 0.07
        assert abs(after_p[fired].mean() - happened[fired].mean()) < 0.03
        now = known & (after_p >= threshold)
        assert 50 < now.sum() < fired.sum()
        assert abs(after_p[now].mean() - happened[now].mean()) < 0.05
        # ... and the record stored with the calibration says the same.
        record = found.info["events"][event]
        assert record["rows_known"] == int(known.sum())
        assert record["observed"] == pytest.approx(happened[known].mean())
        assert record["predicted_before"] == pytest.approx(before_p[known].mean())
        assert record["predicted_after"] == pytest.approx(after_p[known].mean())
        assert record["nll_before"] == pytest.approx(
            C.binary_nll(before_p[known], happened[known])
        )
        assert record["nll_after"] == pytest.approx(
            C.binary_nll(after_p[known], happened[known])
        )
        assert record["nll_after"] < record["nll_before"] - 0.005
        assert abs(record["predicted_after"] - record["observed"]) < 0.004
        assert record["selection"]["chosen"] == C.MAP_LOGISTIC
        assert record["fit"]["nll_after"] < record["fit"]["nll_before"]
    assert found.info["note"] == "unit" and found.info["examples"] == 20_000
    assert found.action_temperature is None


def test_a_miscalibration_on_turn_one_is_found_and_removed():
    """One map over all slot-turns is right on average and wrong on the lead
    turn (review of 2026-10-05: 8-10 points too high on turn 1 after a
    two-parameter fit). With context terms the lead turn is right as well."""
    n = 30_000
    data, model, truth = synthetic_context(n, seed=20)
    groups = np.arange(n) // 6
    found = C.fit_event_calibration(model, data, groups=groups)
    plain = C.fit_event_calibration(model, data, groups=groups, context=False)
    assert not plain.needs_context and plain.switch.terms == ()
    assert plain.info["context"] is None and found.info["context"] == list(C.CONTEXT)
    assert plain.info["events"][C.EVENT_SWITCH]["selection"]["context"] is None
    # The terms undo what the synthetic model added: (0.8, 0.6, 0) / 1.6.
    assert found.switch.kind == C.MAP_LOGISTIC and found.needs_context
    terms = dict(found.switch.terms)
    assert terms[C.CONTEXT_TURN_ONE] == pytest.approx(-0.5, abs=0.1)
    assert terms[C.CONTEXT_FIRST_TURN] == pytest.approx(-0.375, abs=0.08)
    # The third flag changes nothing in this model: one term at a time, it
    # never earns its place and is not carried along by the other two.
    assert C.CONTEXT_PROTECTED_LAST not in terms
    assert found.switch.slope == pytest.approx(1 / OVER_SWITCH[0], abs=0.04)
    assert found.switch.bias == pytest.approx(
        -OVER_SWITCH[1] / OVER_SWITCH[0], abs=0.07
    )
    about = found.info["events"][C.EVENT_SWITCH]["selection"]["context"]
    assert about["accepted"] is True and about["used"] is True
    assert about["gain_over_simpler"] > C.TERM_MARGIN and about["better_folds"] >= 8
    assert (
        about["cross_validated_nll"]
        < (
            found.info["events"][C.EVENT_SWITCH]["selection"]["cross_validated_nll"][
                C.MAP_LOGISTIC
            ]
        )
    )
    assert about["fitted"] == found.switch.to_payload()
    assert set(about["chosen"]) == {C.CONTEXT_TURN_ONE, C.CONTEXT_FIRST_TURN}
    steps = about["steps"]
    assert [step["accepted"] for step in steps] == [True, True, False]
    assert [step["best"] for step in steps[:2]] == about["chosen"]
    assert steps[2]["best"] == C.CONTEXT_PROTECTED_LAST
    assert list(steps[2]["tried"]) == [C.CONTEXT_PROTECTED_LAST]
    assert steps[2]["tried"][C.CONTEXT_PROTECTED_LAST]["gain"] < C.TERM_MARGIN
    assert set(steps[0]["tried"]) == set(C.CONTEXT)
    for step in steps[:2]:
        took = step["tried"][step["best"]]
        assert took["gain"] > C.TERM_MARGIN and took["better_folds"] >= 8
        assert took["gain"] == max(found["gain"] for found in step["tried"].values())
    assert "is left out" in about["reason"] and about["reason"].startswith("taken (")
    assert about["rows"][C.CONTEXT_TURN_ONE] > 3000
    assert found.info["events"][C.EVENT_SWITCH]["selection"]["chosen_terms"] == terms
    # Protect was not shifted by any flag: its map stays the two-parameter one.
    guard = found.info["events"][C.EVENT_PROTECT]["selection"]["context"]
    assert guard["accepted"] is False and found.protect.terms == ()
    assert found.protect == plain.protect

    mask, flags, context = data["action_mask"], data["cand_flag"], C.event_context(data)
    after = found.apply_checked(model, mask, flags, context)
    two = plain.apply_checked(model, mask, flags)
    known, happened = C.event_labels(data)[C.EVENT_SWITCH]
    lead = context[..., C.CONTEXT.index(C.CONTEXT_TURN_ONE)]

    def gap(action: np.ndarray, rows: np.ndarray, threshold: float = 0.35) -> float:
        share = C.event_shares(action, mask, flags).switch
        fired = known & rows & (share >= threshold)
        assert fired.sum() > 150
        return float(share[fired].mean() - happened[fired].mean())

    # Over all slot-turns both are calibrated at the threshold ...
    everywhere = np.ones_like(lead)
    assert abs(gap(two, everywhere)) < 0.03 and abs(gap(after, everywhere)) < 0.03
    # ... but on turn 1 the two-parameter map still says far too much, and the
    # map with terms does not; later turns are right too.
    assert gap(two, lead) > 0.06
    assert abs(gap(after, lead)) < 0.035
    assert abs(gap(after, ~lead)) < 0.03
    assert gap(two, ~lead) < -0.01  # the one map splits the difference
    active = np.asarray(data["act_mon"]) >= 0
    assert (
        np.abs(after - truth)[active].mean() < 0.8 * np.abs(two - truth)[active].mean()
    )
    # The stored record is of the calibrated shares, terms included.
    record = found.info["events"][C.EVENT_SWITCH]
    share = C.event_shares(after, mask, flags).switch
    assert record["predicted_after"] == pytest.approx(share[known].mean())
    assert record["nll_after"] < plain.info["events"][C.EVENT_SWITCH]["nll_after"]

    # Without the context the terms are never dropped: an error, or the input.
    with pytest.raises(ValueError, match="context"):
        found.apply_checked(model, mask, flags)
    before = sum(C.COUNTERS.values())
    assert found.apply(model, mask, flags) is model
    assert found.apply(model, mask, flags, context[:5]) is model
    assert sum(C.COUNTERS.values()) == before + 2
    assert np.array_equal(found.apply(model, mask, flags, context), after)
    # rescale is the same function of the maps and the context
    assert np.array_equal(
        C.rescale(model, mask, flags, found.switch, found.protect, context), after
    )


def test_context_terms_are_not_taken_where_no_flag_matters():
    """The control: a model that is off by the same map on every turn."""
    n = 12_000
    data, model, _ = synthetic_context(n, seed=21, offsets=(0.0, 0.0, 0.0))
    groups = np.arange(n) // 6
    found = C.fit_event_calibration(model, data, groups=groups)
    plain = C.fit_event_calibration(model, data, groups=groups, context=False)
    assert not found.needs_context and found.maps == plain.maps
    for event in C.EVENTS:
        about = found.info["events"][event]["selection"]["context"]
        assert about["accepted"] is False and about["used"] is False
        assert about["reason"].startswith("not taken:")
        assert about["fitted"]["terms"]  # they were fitted, and found wanting
        assert abs(about["gain_over_simpler"]) < C.TERM_MARGIN
    # The same batch without its context arrays: fitted as before they existed.
    bare = {k: v for k, v in data.items() if k not in ("turn", "mon_flag")}
    assert C.fit_event_calibration(model, bare, groups=groups).maps == plain.maps
    # a margin nobody can reach, and a margin of nothing, decide as they say
    never = C.fit_event_calibration(model, data, groups=groups, term_margin=1.0)
    assert not never.needs_context


def test_select_map_takes_terms_only_with_the_margin_and_the_folds():
    rng = np.random.default_rng(30)
    n = 16_000
    p = rng.beta(1.0, 5.0, n)
    flags = np.zeros((n, len(C.CONTEXT)), dtype=bool)
    flags[:, 0] = rng.random(n) < 0.2
    flags[:, 1] = flags[:, 0] | (rng.random(n) < 0.2)
    true_p = sigmoid(0.7 * logit(p) - 0.2 - 0.9 * flags[:, 0])
    y = rng.random(n) < true_p
    groups = np.arange(n) // 4
    without, record = C.select_map(p, y, groups)
    assert not without.needs_context and record["context"] is None
    assert record["chosen_terms"] == {}
    chosen, record = C.select_map(p, y, groups, context=flags)
    about = record["context"]
    assert chosen.kind == C.MAP_LOGISTIC and chosen.needs_context
    assert dict(chosen.terms)[C.CONTEXT_TURN_ONE] == pytest.approx(-0.9, abs=0.15)
    # the third flag is never set: it has no rows of its own, so it gets no term
    assert C.CONTEXT_PROTECTED_LAST not in dict(chosen.terms)
    assert about["rows"][C.CONTEXT_PROTECTED_LAST] == 0
    assert C.CONTEXT_PROTECTED_LAST not in about["steps"][0]["tried"]
    # One term at a time: the second flag holds every row of the first and so
    # shares its gain when tried alone, but once the first is in it adds
    # nothing, and a flag that adds nothing stays out.
    assert about["chosen"] == [C.CONTEXT_TURN_ONE]
    assert [step["accepted"] for step in about["steps"]] == [True, False]
    first, second = about["steps"]
    assert first["best"] == C.CONTEXT_TURN_ONE
    assert first["tried"][C.CONTEXT_FIRST_TURN]["gain"] > C.TERM_MARGIN
    assert second["best"] == C.CONTEXT_FIRST_TURN
    assert second["tried"][C.CONTEXT_FIRST_TURN]["gain"] < C.TERM_MARGIN
    assert C.CONTEXT_FIRST_TURN not in dict(chosen.terms)
    assert "is left out" in about["reason"]
    assert about["accepted"] and about["better_folds"] >= 8
    assert record["chosen"] == C.MAP_LOGISTIC and record["chosen_terms"] == dict(
        chosen.terms
    )
    assert "logistic with context terms" in record["reason"]
    assert set(record["cross_validated_nll"]) == set(C.MAP_KINDS)
    # in sample the larger model is never worse; out of sample it is clearly better
    assert about["in_sample_nll"] <= record["in_sample_nll"][C.MAP_LOGISTIC]
    assert about["gain_over_simpler"] == pytest.approx(
        record["cross_validated_nll"][C.MAP_LOGISTIC] - about["cross_validated_nll"]
    )
    # The same gain under a margin it does not reach: not taken, and said.
    kept, record = C.select_map(p, y, groups, context=flags, term_margin=0.5)
    assert kept == without and record["context"]["accepted"] is False
    assert "needs more than 0.5" in record["context"]["reason"]
    # ... and under a share of the folds no fit can win.
    kept, record = C.select_map(p, y, groups, context=flags, fold_share=1.01)
    assert kept == without and "(needs 11)" in record["context"]["reason"]
    # No cross-validation, no terms.
    alone, record = C.select_map(p, y, np.zeros(n), context=flags)
    assert not alone.needs_context
    assert record["context"]["reason"].startswith("no cross-validation")
    # Too few rows for any map: none for the terms either.
    none, record = C.select_map(p[:30], y[:30], context=flags[:30])
    assert none.is_identity and record["context"]["accepted"] is False
    with pytest.raises(ValueError, match="context"):
        C.select_map(p, y, groups, context=flags[:5])
    # fit_logistic with a context: the two-parameter fit when no flag is usable
    assert C.fit_logistic(p, y, context=np.zeros_like(flags)) == C.fit_logistic(p, y)
    with_terms = C.fit_logistic(p, y, context=flags)
    assert C.binary_nll(with_terms(p, flags), y) < C.binary_nll(
        C.fit_logistic(p, y)(p), y
    )


def test_the_protect_map_is_fitted_on_slots_that_did_not_switch():
    data, model, _ = synthetic(8000, seed=11)
    found = C.fit_event_calibration(model, data)
    shares = C.event_shares(model, data["action_mask"], data["cand_flag"])
    labels = C.event_labels(data)
    known_s, switched = labels[C.EVENT_SWITCH]
    known_p, protected = labels[C.EVENT_PROTECT]
    rows_s = known_s & shares.switch_possible
    rows_p = known_p & shares.protect_possible & known_s & ~switched
    assert found.info["events"][C.EVENT_SWITCH]["rows_fitted"] == int(rows_s.sum())
    assert found.info["events"][C.EVENT_PROTECT]["rows_fitted"] == int(rows_p.sum())
    assert rows_p.sum() < (known_p & shares.protect_possible).sum()
    # Each map is the fit of its own rows and objective, nothing else.
    assert found.switch == C.select_map(shares.switch[rows_s], switched[rows_s])[0]
    assert (
        found.protect
        == C.select_map(shares.protect_share[rows_p], protected[rows_p])[0]
    )


class Recording(dict):
    """A batch that remembers which arrays were read."""

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.read: set[str] = set()

    def __getitem__(self, key: str) -> Any:
        self.read.add(key)
        return super().__getitem__(key)


def test_the_fit_uses_only_the_rows_it_is_given():
    data, model, _ = synthetic(6000, seed=12)
    n = 6000
    split = np.arange(n) % 3  # 0 "train", 1 "validation", 2 "test"
    rows = split == 1
    groups = np.arange(n) // 5
    want = C.fit_event_calibration(model, data, rows=rows, groups=groups)
    # The same as fitting the rows taken out by hand, by mask or by index.
    taken = C.fit_event_calibration(
        model[rows], F.take(data, rows), groups=groups[rows]
    )
    by_index = C.fit_event_calibration(
        model, data, rows=np.flatnonzero(rows), groups=groups
    )
    assert taken == want and by_index == want
    assert want.info["examples"] == int(rows.sum())
    # Whatever the other rows hold - labels, predictions, masks - changes nothing.
    rng = np.random.default_rng(0)
    spoiled = {name: array.copy() for name, array in data.items()}
    other = ~rows
    spoiled["y_flag"][other] = rng.integers(0, 2, spoiled["y_flag"][other].shape)
    spoiled["action_mask"][other] = 1
    spoiled["cand_flag"][other] = F.CAND_VALID | F.CAND_PROTECT
    spoiled["act_mon"][other] = 0
    noisy = model.copy()
    noisy[other] = rng.random(noisy[other].shape)
    assert C.fit_event_calibration(noisy, spoiled, rows=rows, groups=groups) == want
    assert C.fit_event_calibration(noisy, spoiled, groups=groups) != want
    # A row inside the mask does count.
    inside = {name: array.copy() for name, array in data.items()}
    inside["y_flag"][rows, :, F.Y_SWITCHED] = 0
    assert C.fit_event_calibration(model, inside, rows=rows, groups=groups) != want
    # Only four arrays are read; split and bookkeeping arrays are not among them.
    watched = Recording(
        {**data, "m_split": split, "m_flag": np.zeros(n), "y_set": np.zeros(n)}
    )
    assert C.fit_event_calibration(model, watched, rows=rows, groups=groups) == want
    assert watched.read == {"action_mask", "cand_flag", "act_mon", "y_flag"}
    # With the public context in the batch the fit reads those two feature
    # arrays as well (for the context terms), and still no other label.
    around, shifted, _ = synthetic_context(3000, seed=14)
    watched = Recording({**around, "m_split": np.zeros(3000), "y_set": np.zeros(3000)})
    C.fit_event_calibration(shifted, watched)
    assert watched.read == {
        "action_mask",
        "cand_flag",
        "act_mon",
        "y_flag",
        "turn",
        "mon_flag",
    }
    watched.read.clear()
    C.fit_event_calibration(shifted, watched, context=False)
    assert watched.read == {"action_mask", "cand_flag", "act_mon", "y_flag"}
    for bad in (np.ones(5, dtype=bool), np.ones((n, 2), dtype=bool)):
        with pytest.raises((ValueError, IndexError)):
            C.fit_event_calibration(model, data, rows=bad)
    with pytest.raises(ValueError):
        C.fit_event_calibration(model[:10], data)
    with pytest.raises(ValueError):
        C.fit_event_calibration(model, data, groups=np.arange(7))
    with pytest.raises(KeyError):
        C.fit_event_calibration(model, {k: v for k, v in data.items() if k != "y_flag"})


# --- payload --------------------------------------------------------------------


def plain_data(value: Any) -> bool:
    if isinstance(value, dict):
        return all(isinstance(k, str) and plain_data(v) for k, v in value.items())
    if isinstance(value, list):
        return all(plain_data(item) for item in value)
    return value is None or type(value) in (str, int, float, bool)


def test_calibration_payload_round_trip(tmp_path: Path):
    data, model, _ = synthetic(6000, seed=13)
    fitted = C.fit_event_calibration(
        model, data, action_temperature=1.25, info={"dataset_tag": "unit"}
    )
    for found in (some_calibration(), fitted, C.EventCalibration()):
        payload = found.to_payload()
        assert plain_data(payload) and payload["format"] == C.FORMAT
        json.dumps(payload)  # also fit for the report
        again = C.EventCalibration.from_payload(payload)
        assert again == found and again.to_payload() == payload
        path = tmp_path / "calibration.pt"
        torch.save(payload, path)
        stored = torch.load(path, map_location="cpu", weights_only=True)
        assert C.EventCalibration.from_payload(stored) == found
        out = again.apply(model, data["action_mask"], data["cand_flag"])
        want = found.apply(model, data["action_mask"], data["cand_flag"])
        assert np.array_equal(out, want)
        described = found.describe()
        assert described["applied"] is (not found.is_identity)
        assert set(described["events"]) <= set(C.EVENTS)
    stored = fitted.to_payload()
    assert stored["action_temperature"] == 1.25
    assert stored["maps"][C.EVENT_SWITCH]["map"] == C.MAP_LOGISTIC
    record = stored["info"]["events"][C.EVENT_PROTECT]
    assert stored["info"]["dataset_tag"] == "unit"
    for key in ("observed", "predicted_before", "predicted_after", "nll_before"):
        assert isinstance(record[key], float)
    # The info is a copy: changing the source afterwards does not reach it.
    source = {"dataset_tag": "first", "numbers": np.arange(3), "pair": (1, 2.5)}
    made = C.EventCalibration(info=source)
    source["dataset_tag"] = "second"
    assert made.info == {"dataset_tag": "first", "numbers": [0, 1, 2], "pair": [1, 2.5]}


def test_damaged_calibration_payloads_raise_value_error():
    payload = some_calibration().to_payload()
    for change in (
        {"format": "something-else"},
        {"version": 99},
        {"maps": {}},
        {"maps": {C.EVENT_SWITCH: payload["maps"][C.EVENT_SWITCH]}},
        {
            "maps": {
                **payload["maps"],
                C.EVENT_SWITCH: {"map": C.MAP_LOGISTIC, "slope": -1},
            }
        },
        {"maps": {**payload["maps"], C.EVENT_PROTECT: {"map": "unknown"}}},
        {"maps": {**payload["maps"], C.EVENT_PROTECT: {"slope": 1.0}}},
        {"maps": "nonsense"},
        {"action_temperature": -2.0},
        {"info": {"x": object()}},
    ):
        with pytest.raises(ValueError):
            C.EventCalibration.from_payload({**payload, **change})
    with pytest.raises(ValueError):
        C.EventCalibration.from_payload({})
    with pytest.raises(ValueError):
        C.EventCalibration.from_payload(None)  # type: ignore[arg-type]


# --- in the predictor -----------------------------------------------------------


def test_a_predictor_applies_its_calibration_once(
    fz: F.Featurizer, batch: F.Batch, predictor: M.OppNetPredictor
):
    found = some_calibration()
    plain = predictor.predict(batch)
    twin = predictor.with_event_calibration(found)
    assert twin.net is predictor.net and twin.temperatures == predictor.temperatures
    assert twin.event_calibration == found and predictor.event_calibration is None
    assert twin.event_calibrated and twin.name == predictor.name
    made = twin.predict(batch)
    assert not twin.counters and made["action"].dtype == np.float32
    mask, flags = batch["action_mask"], batch["cand_flag"]
    once = found.apply(plain["action"], mask, flags)
    twice = found.apply(once, mask, flags)
    assert np.abs(made["action"] - once).max() < 1e-6
    assert np.abs(made["action"] - twice).max() > 1e-2  # the maps are not idempotent
    # Targets and the Mega head are untouched; the result is still normalised.
    assert np.array_equal(made["target"], plain["target"])
    assert np.array_equal(made["mega"], plain["mega"])
    norm = F.normalize_prediction(made, batch)
    assert np.abs(norm["action"] - made["action"]).max() < 1e-6
    assert not made["action"][~mask.astype(bool)].any()
    # Giving the calibration again REPLACES it: still one application.
    again = twin.with_event_calibration(found).predict(batch)
    assert np.array_equal(again["action"], made["action"])
    other = twin.with_event_calibration(C.EventCalibration(MAPS[2], MAPS[1]))
    want = other.event_calibration.apply(plain["action"], mask, flags)  # type: ignore[union-attr]
    assert np.abs(other.predict(batch)["action"] - want).max() < 1e-6
    # None takes it off again.
    off = twin.with_event_calibration(None).predict(batch)
    assert np.array_equal(off["action"], plain["action"])
    renamed = predictor.with_event_calibration(found, name="unit_cal")
    assert renamed.name == "unit_cal" and predictor.name == "unit"
    # Chunked and whole predictions agree, as without a calibration.
    small = M.OppNetPredictor(
        predictor.net, fz, action_temperature=1.3, batch_size=3, event_calibration=found
    )
    assert np.allclose(small.predict(batch)["action"], made["action"], atol=1e-6)
    described = twin.describe()
    assert described["event_calibrated"] is True
    assert described["event_calibration"]["applied"] is True
    assert (
        described["event_calibration"]["maps"][C.EVENT_SWITCH]["map"] == C.MAP_LOGISTIC
    )
    assert described["event_calibration"]["info"]["dataset_tag"] == "unit"
    assert described["temperatures"] == predictor.temperatures
    blind = M.OppNetPredictor(
        predictor.net, fz, elo_mode=F.ELO_BLANK, action_temperature=1.3
    ).with_event_calibration(found)
    assert blind.elo_mode == F.ELO_BLANK and blind.event_calibrated
    # Every twin shares the network AND the lock around its forward pass: a
    # predictor and its calibrated twin toggle the same network's mode.
    twins = (
        twin,
        renamed,
        twin.with_event_calibration(None),
        twin.with_event_calibration(found),
        predictor.with_temperatures(action=1.0),
        twin.with_temperatures(action=1.3),
    )
    for made_twin in twins:
        assert made_twin.net is predictor.net
        assert made_twin._lock is predictor._lock
    assert M.OppNetPredictor(predictor.net, fz)._lock is not predictor._lock


def with_terms(temperature: float | None = 1.3) -> C.EventCalibration:
    """A calibration whose switch map carries context terms."""
    return C.EventCalibration(
        C.EventMap(
            C.MAP_LOGISTIC,
            0.7,
            -0.4,
            terms=(
                (C.CONTEXT_TURN_ONE, -0.6),
                (C.CONTEXT_FIRST_TURN, -0.3),
                (C.CONTEXT_PROTECTED_LAST, 0.5),
            ),
        ),
        C.EventMap(C.MAP_LOGISTIC, 0.9, -0.2),
        temperature,
        {"dataset_tag": "unit"},
    )


def test_a_predictor_hands_the_context_to_a_calibration_with_terms(
    fz: F.Featurizer, batch: F.Batch, predictor: M.OppNetPredictor
):
    found = with_terms()
    bare = C.EventCalibration(
        C.EventMap(C.MAP_LOGISTIC, 0.7, -0.4), found.protect, 1.3, found.info
    )
    assert found.needs_context and not bare.needs_context
    plain = predictor.predict(batch)
    twin = predictor.with_event_calibration(found)
    made = twin.predict(batch)
    assert not twin.counters
    mask, flags = batch["action_mask"], batch["cand_flag"]
    context = C.event_context(batch)
    # The hand-written game has a turn 1, first turns on the field and a Protect.
    assert context.any(axis=(0, 1)).all()
    want = found.apply_checked(plain["action"], mask, flags, context)
    assert np.abs(made["action"] - want).max() < 1e-6
    without = predictor.with_event_calibration(bare).predict(batch)
    assert np.abs(made["action"] - without["action"]).max() > 1e-2  # the terms act
    flagged = context.any(-1)
    untouched = ~flagged & (batch["act_mon"] >= 0)
    assert untouched.any()
    assert np.allclose(made["action"][untouched], without["action"][untouched])
    assert np.array_equal(made["target"], plain["target"])
    assert np.array_equal(made["mega"], plain["mega"])
    norm = F.normalize_prediction(made, batch)
    assert np.abs(norm["action"] - made["action"]).max() < 1e-6
    # It reads features only: labels and bookkeeping change nothing.
    assert all(
        np.array_equal(made[name], twin.predict(M.strip_labels(batch))[name])
        for name in made
    )
    # The context arrays missing: a failed prediction, never the map without
    # its terms.
    lacking = {k: v for k, v in M.strip_labels(batch).items() if k != "mon_flag"}
    answer = twin.predict(lacking)
    assert sum(twin.counters.values()) == 1
    assert next(iter(twin.counters)).startswith("predict_error:")
    assert S.predict_errors(twin) == 1
    # Through the payload and an artifact file: the same numbers, terms and all.
    payload = twin.to_payload()
    assert payload["version"] == M.PAYLOAD_VERSION_CALIBRATED
    assert payload["event_calibration"]["version"] == C.VERSION_CONTEXT == 2
    assert payload["event_calibration"]["maps"][C.EVENT_SWITCH]["terms"] == dict(
        found.switch.terms
    )
    again = M.from_payload(payload, fz)
    assert again.event_calibration == found
    assert all(np.array_equal(made[n], again.predict(batch)[n]) for n in made)
    assert again.describe()["event_calibration"]["maps"][C.EVENT_SWITCH]["terms"]
    del answer


def test_a_calibration_with_terms_is_its_own_payload_version():
    """A reader of version 1 knows no terms and would drop them: a calibration
    that carries terms says version 2, and the two must agree."""
    found, plain = with_terms(), some_calibration()
    assert found.to_payload()["version"] == C.VERSION_CONTEXT == 2
    assert plain.to_payload()["version"] == C.VERSION == 1
    assert C.EventCalibration().to_payload()["version"] == 1
    for made in (found, plain):
        payload = made.to_payload()
        json.dumps(payload)
        assert C.EventCalibration.from_payload(payload) == made
    with pytest.raises(ValueError, match="version 1"):
        C.EventCalibration.from_payload({**found.to_payload(), "version": 1})
    with pytest.raises(ValueError, match="version 2"):
        C.EventCalibration.from_payload({**plain.to_payload(), "version": 2})
    with pytest.raises(ValueError):
        C.EventCalibration.from_payload({**found.to_payload(), "version": 3})
    described = found.describe()
    assert described["applied"] is True and described["version"] == 2


def test_predictor_payload_round_trip_with_a_calibration(
    tmp_path: Path, fz: F.Featurizer, batch: F.Batch, predictor: M.OppNetPredictor
):
    found = some_calibration()
    twin = predictor.with_event_calibration(found, name="unit_cal")
    want = twin.predict(batch)
    payload = twin.to_payload()
    # A payload that carries a calibration says so in its version (2); one
    # without is version 1, as every payload written before the field existed.
    assert TM.plain_data(payload)
    assert payload["version"] == M.PAYLOAD_VERSION_CALIBRATED == 2
    assert payload["event_calibration"] == found.to_payload()
    assert predictor.to_payload()["event_calibration"] is None
    assert predictor.to_payload()["version"] == M.PAYLOAD_VERSION == 1
    again = M.from_payload(payload, fz)
    assert again.event_calibration == found and again.name == "unit_cal"
    # Stored once, applied once: a second trip through the payload changes nothing.
    third = M.from_payload(again.to_payload(), fz)
    for made in (again.predict(batch), third.predict(batch)):
        assert all(np.array_equal(want[name], made[name]) for name in want)
    path = tmp_path / "artifact.pt"
    A.save_artifact(
        path,
        kind=M.KIND,
        name="unit_cal",
        featurizer=fz,
        predictor_payload=payload,
        extra={"note": "unit"},
    )
    loaded = A.load_predictor(path)
    assert isinstance(loaded.predictor, M.OppNetPredictor)
    assert loaded.predictor.event_calibration == found
    assert loaded.predictor.describe()["event_calibrated"] is True
    made = loaded.predictor.predict(batch)
    assert all(np.array_equal(want[name], made[name]) for name in want)
    assert A.try_load_predictor(path) is not None


def test_an_old_payload_without_calibration_predicts_as_before(
    fz: F.Featurizer, batch: F.Batch, predictor: M.OppNetPredictor
):
    payload = predictor.to_payload()
    old = {key: value for key, value in payload.items() if key != "event_calibration"}
    assert set(old) == {
        "format",
        "version",
        "name",
        "config",
        "state_dict",
        "temperatures",
        "elo_mode",
        "n_parameters",
    }  # the keys of every payload written before event calibration existed
    again = M.from_payload(old, fz)
    assert again.event_calibration is None and not again.event_calibrated
    assert again.describe()["event_calibration"] is None
    made = again.predict(batch)
    # ... which is the tempered softmax of the network, nothing more.
    out = M.collect_outputs(predictor.net, batch)
    want = M.probabilities(out, action_temperature=1.3)
    for name in ("action", "target", "mega"):
        assert np.array_equal(made[name], want[name].numpy().astype(np.float32))
    assert all(np.array_equal(made[n], predictor.predict(batch)[n]) for n in made)


def test_a_calibrated_payload_is_version_two_and_an_old_reader_refuses_it(
    tmp_path: Path,
    fz: F.Featurizer,
    batch: F.Batch,
    predictor: M.OppNetPredictor,
    monkeypatch: pytest.MonkeyPatch,
):
    """The model code from before event calibration reads ``version == 1``
    and ignores keys it does not know: a calibrated payload written as
    version 1 loaded there under the calibrated NAME and predicted
    uncalibrated. Version 2 makes that reader refuse the file."""
    twin = predictor.with_event_calibration(some_calibration(), name="unit_cal")
    payload = twin.to_payload()
    assert payload["version"] == M.PAYLOAD_VERSION_CALIBRATED == 2
    assert predictor.to_payload()["version"] == M.PAYLOAD_VERSION == 1
    # a calibration that is carried counts, even one that changes nothing
    idle = predictor.with_event_calibration(C.EventCalibration(action_temperature=1.3))
    assert idle.to_payload()["version"] == 2
    assert twin.with_event_calibration(None).to_payload()["version"] == 1
    # Version and content must agree, both ways.
    with pytest.raises(ValueError, match="must not carry an event calibration"):
        M.from_payload(dict(payload, version=1), fz)
    with pytest.raises(ValueError, match="must carry an event calibration"):
        M.from_payload(dict(payload, event_calibration=None), fz)
    stripped = {k: v for k, v in payload.items() if k != "event_calibration"}
    with pytest.raises(ValueError, match="must carry an event calibration"):
        M.from_payload(stripped, fz)
    with pytest.raises(ValueError, match="must carry an event calibration"):
        M.from_payload(dict(predictor.to_payload(), version=2), fz)
    for unknown in (0, 3, "two"):
        with pytest.raises(ValueError):
            M.from_payload(dict(payload, version=unknown), fz)
    path = tmp_path / "artifact.pt"
    A.save_artifact(
        path, kind=M.KIND, name="unit_cal", featurizer=fz, predictor_payload=payload
    )
    assert A.read_artifact(path)["predictor"]["version"] == 2
    loaded = A.load_predictor(path).predictor
    assert isinstance(loaded, M.OppNetPredictor) and loaded.event_calibrated
    # The reader as it was before this field: version 1 and nothing else.
    monkeypatch.setattr(M, "PAYLOAD_VERSIONS", (M.PAYLOAD_VERSION,))
    with pytest.raises(ValueError, match="payload version 2"):
        M.from_payload(payload, fz)
    with pytest.raises(ValueError, match="payload version 2"):
        A.load_predictor(path)
    assert A.try_load_predictor(path) is None  # the runtime: not serving
    rt = R.OpponentPredictor.load(path)
    assert not rt.loaded and not rt.serving
    # ... while an uncalibrated payload is, for that reader, what it always was.
    old = M.from_payload(predictor.to_payload(), fz)
    assert all(
        np.array_equal(old.predict(batch)[n], predictor.predict(batch)[n])
        for n in ("action", "target", "mega")
    )


def test_the_artifact_on_disk_loads_without_calibration():
    """The shipped artifact was written before this field existed."""
    if not REAL_ARTIFACT.is_file() or not (REAL_DATASET / "manifest.json").is_file():
        pytest.skip("results_oppmodel/oppnet_v2_blind or v2_feed is not here")
    stored = A.read_artifact(REAL_ARTIFACT)
    assert "event_calibration" not in stored["predictor"]
    loaded = A.load_predictor(REAL_ARTIFACT).predictor
    assert isinstance(loaded, M.OppNetPredictor)
    assert loaded.event_calibration is None and not loaded.event_calibrated
    data, _ = F.load_dataset(REAL_DATASET, splits=["val"])
    rows = F.sheet_unknown_as_closed(M.strip_labels(F.take(data, slice(0, 256))))
    made = loaded.predict(rows)
    assert not loaded.counters
    blank = F.apply_elo_mode(rows, loaded.elo_mode)
    out = M.collect_outputs(loaded.net, blank)
    want = M.probabilities(
        out,
        loaded.action_temperature,
        loaded.target_temperature,
        loaded.mega_temperature,
        loaded.mega_bias,
    )
    for name in ("action", "target", "mega"):
        assert np.array_equal(made[name], want[name].numpy().astype(np.float32))


def test_a_calibration_belongs_to_one_action_temperature(
    fz: F.Featurizer, batch: F.Batch, predictor: M.OppNetPredictor
):
    found = some_calibration(1.3)
    assert found.fits_temperature(1.3) and not found.fits_temperature(1.31)
    assert some_calibration(None).fits_temperature(7.0)
    twin = predictor.with_event_calibration(found)
    # Other temperatures: the calibration says nothing about them and is dropped.
    cooled = twin.with_temperatures(action=1.0, target=1.0)
    assert cooled.event_calibration is None and twin.event_calibration == found
    raw = M.OppNetPredictor(predictor.net, fz).predict(batch)
    assert np.array_equal(cooled.predict(batch)["action"], raw["action"])
    # It cannot be put on a predictor with another action temperature ...
    with pytest.raises(ValueError):
        M.OppNetPredictor(predictor.net, fz, event_calibration=found)
    with pytest.raises(ValueError):
        cooled.with_event_calibration(found)
    with pytest.raises(ValueError):
        M.OppNetPredictor(predictor.net, fz, event_calibration="maps")  # type: ignore[arg-type]
    # ... nor read from a payload whose temperature was changed afterwards.
    payload = twin.to_payload()
    moved = dict(payload, temperatures={**payload["temperatures"], "action": 1.0})
    with pytest.raises(ValueError):
        M.from_payload(moved, fz)
    # A damaged stored calibration is an error, never silently dropped.
    for broken in ({"format": "x"}, "text", {**found.to_payload(), "maps": {}}):
        with pytest.raises(ValueError):
            M.from_payload(dict(payload, event_calibration=broken), fz)


def test_a_failing_calibration_is_a_failed_prediction(
    batch: F.Batch, predictor: M.OppNetPredictor, monkeypatch: pytest.MonkeyPatch
):
    twin = predictor.with_event_calibration(some_calibration())
    good = twin.predict(batch)
    assert not twin.counters

    def broken(self: Any, *args: Any) -> Any:
        raise ValueError("no")

    monkeypatch.setattr(C.EventCalibration, "apply_checked", broken)
    made = twin.predict(batch)  # does not raise
    assert twin.counters["predict_error:ValueError"] == 1
    uniform = F.uniform_prediction(batch)
    assert np.array_equal(made["action"], uniform["action"])
    assert S.predict_errors(twin) == 1  # the harnesses see it and stop
    monkeypatch.undo()
    again = twin.predict(batch)
    assert all(np.array_equal(good[name], again[name]) for name in good)
    # A calibrated predictor still never raises on a malformed batch.
    assert twin.predict({}) == {}
    truncated = dict(batch)
    truncated["cand_flag"] = batch["cand_flag"][:, :, :5]
    before = sum(twin.counters.values())
    assert set(twin.predict(truncated)) == {"action", "target", "mega"}
    assert sum(twin.counters.values()) == before + 1


def test_a_calibrated_predictor_reads_no_label(
    batch: F.Batch, predictor: M.OppNetPredictor
):
    twin = predictor.with_event_calibration(some_calibration())
    data = dict(batch)
    data["m_split"] = np.zeros(batch["act_mon"].shape[0], dtype=np.uint8)
    want = twin.predict(M.strip_labels(data))
    scrambled = dict(data)
    for name in data:
        if name.startswith(("y_", "m_")):
            scrambled[name] = np.ones_like(data[name])
    made = twin.predict(scrambled)
    assert all(np.array_equal(want[name], made[name]) for name in want)


def runtime_forecasts(rt: R.OpponentPredictor, role: str) -> dict[int, R.Forecast]:
    """The hand-written game fed to a runtime event by event, as the bot would."""
    fake = SimpleNamespace(_replay_data=[], player_role=role, battle_tag="battle-cal-1")
    out: dict[int, R.Forecast] = {}
    for event in E.split_log(TM.HEADER + TM.GAME):
        fake._replay_data.append(list(event))
        if event[1] == "turn":
            made = rt.predict(fake)
            if made is not None:
                out[int(event[2])] = made
    return out


@pytest.mark.parametrize("role", ["p1", "p2"])
def test_the_runtime_forecast_reads_the_calibrated_events(
    fz: F.Featurizer, predictor: M.OppNetPredictor, role: str
):
    """``Forecast.p_switch`` / ``p_protect`` are what a guard reads: with a
    calibrated predictor they are the calibrated events, nothing beside them."""
    found = some_calibration()
    options: dict[str, Any] = {"keep_features": True, "top_k": None}
    plain = runtime_forecasts(R.OpponentPredictor(predictor, fz, **options), role)
    twin = predictor.with_event_calibration(found)
    after = runtime_forecasts(R.OpponentPredictor(twin, fz, **options), role)
    assert len(plain) >= 6 and set(plain) == set(after)
    compared = moved = 0
    for turn, before in plain.items():
        now = after[turn]
        features = before.features
        assert features is not None
        mask, flags = features["action_mask"], features["cand_flag"]
        want = C.event_shares(
            found.apply(before.raw["action"][None], mask, flags), mask, flags
        )
        assert np.array_equal(now.raw["target"], before.raw["target"])
        assert np.array_equal(now.raw["mega"], before.raw["mega"])
        for index, (old, new) in enumerate(zip(before.slots, now.slots)):
            assert (old is None) == (new is None)
            if old is None or new is None:
                continue
            compared += 1
            assert new.p_switch == pytest.approx(want.switch[0, index], abs=1e-6)
            assert new.p_protect == pytest.approx(want.protect[0, index], abs=1e-6)
            assert new.p_mega == old.p_mega
            if 0.0 < old.p_switch < 1.0:
                assert new.p_switch == pytest.approx(
                    float(found.switch(old.p_switch)), abs=1e-5
                )
                moved += abs(new.p_switch - old.p_switch) > 1e-3
            # The ranked list is the same distribution: it still sums to one,
            # and its switch entries add up to the scalar.
            total = sum(action.probability for action in new.actions)
            assert total == pytest.approx(1.0, abs=1e-5)
            switches = sum(a.probability for a in new.actions if a.kind == "switch")
            assert switches == pytest.approx(new.p_switch, abs=1e-6)
    assert compared >= 12 and moved >= 4


# --- the script -----------------------------------------------------------------


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def labelled(
    features: F.Batch, probs: np.ndarray, rng: np.random.Generator, hide: float = 0.0
) -> F.Batch:
    """``features`` with one choice per slot drawn from ``probs``.

    The choice is visible, except that a share ``hide`` of the slots that
    chose neither a switch nor a Protect-family move never get to act (as a
    Pokemon knocked out before it moved): their action is hidden, the log
    proves "no switch, no Protect", and the label is the set of the other
    moves. Hidden actions are therefore never a switch or a Protect, as in
    the real corpus.
    """
    out = dict(features)
    n = int(features["act_mon"].shape[0])
    active = features["act_mon"] >= 0
    mask = features["action_mask"].astype(bool)
    picked = sample_actions(probs, rng)
    flags = features["cand_flag"].astype(np.int64)
    guard = ((flags & F.CAND_VALID) > 0) & ((flags & F.CAND_PROTECT) > 0)
    on_guard = np.take_along_axis(
        guard, np.minimum(picked, N_CAND - 1)[..., None], axis=-1
    )[..., 0]
    switched = active & (picked > N_CAND)
    protected = active & (picked < N_CAND) & on_guard
    hidden = active & ~switched & ~protected & (rng.random((n, 2)) < hide)
    shown = active & ~hidden
    y_flag = np.zeros((n, 2, F.N_Y_FLAG), dtype=np.uint8)
    y_flag[..., F.Y_SWITCHED] = switched
    y_flag[..., F.Y_SWITCH_KNOWN] = active
    y_flag[..., F.Y_PROTECTED] = protected
    y_flag[..., F.Y_PROTECT_KNOWN] = active
    y_flag[..., F.Y_OTHER] = shown & (picked == N_CAND)
    one = np.zeros((n, 2, N_ACTION), dtype=bool)
    np.put_along_axis(one, picked[..., None], True, axis=-1)
    others = mask.copy()  # every legal move that is neither a switch nor a Protect
    others[..., N_CAND + 1 :] = False
    others[..., :N_CAND] &= ~guard
    y_set = np.where(hidden[..., None], others, one & shown[..., None])
    foes = features["foe_mon"] >= 0
    y_attack = np.where(shown[..., None] & foes[:, None, :], 0, -1)
    out.update(
        y_kind=np.where(
            shown,
            np.where(switched, F.Y_KIND_SWITCH, F.Y_KIND_MOVE),
            np.where(hidden, F.Y_KIND_NONE, F.Y_KIND_ABSENT),
        ).astype(np.int8),
        y_action=np.where(shown, picked, -1).astype(np.int8),
        y_set=y_set.astype(np.uint8),
        y_flag=y_flag,
        y_target=np.full((n, 2), -1, dtype=np.int8),
        y_mega=np.full((n, 2), -1, dtype=np.int8),
        y_intent=np.full((n, 2), -1, dtype=np.int8),
        y_attack=y_attack.astype(np.int8),  # known: nobody was attacked
        y_reason=np.where(
            hidden, F.REASON_NAMES.index(E.REASON_FAINTED_FIRST), 0
        ).astype(np.uint8),
    )
    return out


class World:
    """A dataset folder, an artifact trained "on" it, and what was put in."""

    def __init__(self, root: Path, fz: F.Featurizer, tiles: int = 40) -> None:
        self.root = root
        self.dataset = root / "data"
        self.source = root / "model" / "artifact.pt"
        self.fz = fz
        self.predictor = M.OppNetPredictor(
            TM.jolt(TM.small_net(fz)), fz, name="tiny", action_temperature=1.3
        )
        game = TM.game_batch(fz)
        base = F.concat_batches(
            [
                M.strip_labels(M.swap_slots(game, actor, other))
                for actor in (False, True)
                for other in (False, True)
            ]
        )
        self.features = base
        self.base = int(base["act_mon"].shape[0])
        features = {
            name: np.concatenate([array] * tiles) for name, array in base.items()
        }
        n = self.base * tiles
        model = self.predictor.predict(features)["action"]
        truth = C.rescale(
            model,
            features["action_mask"],
            features["cand_flag"],
            TRUE_SWITCH,
            TRUE_PROTECT,
        )
        rng = np.random.default_rng(21)
        parts = []
        self.sizes: dict[str, int] = {}
        for code, name in enumerate(SPLITS):
            part = labelled(features, truth, rng, hide=0.2)
            part["m_split"] = np.full(n, code, dtype=np.uint8)
            part["m_battle"] = (np.arange(n) // self.base + 1000 * code).astype(
                np.int32
            )
            part["m_flag"] = np.zeros(n, dtype=np.uint8)
            part["m_weight"] = np.ones(n, dtype=np.float32)
            if name == "val":  # one battle with a ladder-holdout opponent
                part["m_flag"][: self.base] = T.FLAG_HOLDOUT_BATTLE
            if name == "ladder_holdout":  # a sheet state nobody recorded
                flags = part["game_flag"].copy()
                flags[:, [F.G_ACTOR_SHEET, F.G_OTHER_SHEET]] = F.SHEET_UNKNOWN
                part["game_flag"] = flags
            self.sizes[name] = n
            parts.append(part)
        self.dataset.mkdir(parents=True)
        data = F.concat_batches(parts)
        F.save_batch(self.dataset / "shard-00000.npz", data)
        manifest = {
            "tag": "tiny",
            "splits": SPLITS,
            "shards": [
                {"file": "shard-00000.npz", "examples": int(data["turn"].shape[0])}
            ],
        }
        (self.dataset / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        self.manifest_sha = T.manifest_sha256(self.dataset)
        self.source.parent.mkdir(parents=True)
        self.save(self.source, self.predictor)

    def save(self, path: Path, predictor: M.OppNetPredictor, **extra: Any) -> None:
        A.save_artifact(
            path,
            kind=M.KIND,
            name=predictor.name,
            featurizer=self.fz,
            predictor_payload=predictor.to_payload(),
            extra={
                "created": "then",
                "dataset": {"tag": "tiny", "manifest_sha256": self.manifest_sha},
                **extra,
            },
        )

    def run(self, out: Path, *extra: str, source: Path | None = None) -> int:
        argv = [
            "--artifact",
            str(source or self.source),
            "--dataset",
            str(self.dataset),
        ]
        return K.main([*argv, "--out", str(out), "--resamples", "200", *extra])


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory, fz: F.Featurizer) -> World:
    return World(tmp_path_factory.mktemp("calibration"), fz)


@pytest.fixture(scope="module")
def finished(world: World) -> tuple[Path, dict[str, Any], str]:
    """One whole run of the script: (its directory, its report, its log)."""
    out = world.root / "run"
    before = sha256(world.source)
    with pytest.MonkeyPatch.context() as patch:
        lines: list[str] = []
        patch.setattr(K, "say", lines.append)
        assert world.run(out) == 0, lines[-1]
    assert sha256(world.source) == before  # the input artifact is not touched
    report = json.loads((out / K.REPORT_JSON).read_text(encoding="utf-8"))
    return out, report, "\n".join(lines)


def test_the_script_writes_a_new_calibrated_artifact(
    world: World, finished: tuple[Path, dict[str, Any], str], batch: F.Batch
):
    out, report, log = finished
    lines = log.splitlines()
    assert lines[0].startswith("CAL_START name tiny_cal source tiny dataset tiny")
    assert lines[-1].startswith(f"{K.DONE} name tiny_cal maps ")
    assert any(line.startswith("FIT ") for line in lines)
    assert sum("PREDICT val before" in line for line in lines) >= 1
    assert sorted(path.name for path in out.iterdir()) == sorted(
        [K.ARTIFACT_NAME, K.REPORT_JSON, K.REPORT_MD]
    )
    assert report["status"] == K.DONE and report["name"] == "tiny_cal"
    assert report["artifact"] == str(out / K.ARTIFACT_NAME)
    assert report["source"]["sha256"] == sha256(world.source)
    assert report["dataset"]["manifest_sha256"] == world.manifest_sha
    assert report["dataset"]["same_build_as_training"] is True

    loaded = A.load_predictor(out / K.ARTIFACT_NAME)
    assert (loaded.name, loaded.kind) == ("tiny_cal", M.KIND)
    predictor = loaded.predictor
    assert isinstance(predictor, M.OppNetPredictor) and predictor.event_calibrated
    assert predictor.name == "tiny_cal"
    found = predictor.event_calibration
    assert found is not None and found.to_payload() == {
        key: value
        for key, value in report["calibration"].items()
        if key not in ("applied", "events")
    }
    assert found.fits_temperature(1.3) and predictor.action_temperature == 1.3
    # The fit found the truth the labels were drawn from: on the shares the
    # network produces, the fitted maps are the true ones.
    assert found.switch.kind == C.MAP_LOGISTIC and found.protect.kind == C.MAP_LOGISTIC
    base = world.features
    shares = C.event_shares(
        world.predictor.predict(base)["action"], base["action_mask"], base["cand_flag"]
    )
    switch = shares.switch[shares.switch_possible]
    guard = shares.protect_share[shares.protect_possible]
    assert np.abs(TRUE_SWITCH(switch) - switch).max() > 0.1
    assert np.abs(TRUE_PROTECT(guard) - guard).max() > 0.1
    assert np.abs(found.switch(switch) - TRUE_SWITCH(switch)).max() < 0.06
    assert np.abs(found.protect(guard) - TRUE_PROTECT(guard)).max() < 0.06
    # What the artifact stores about its calibration.
    info = found.info
    assert info["dataset_tag"] == "tiny" and info["split"] == "val"
    assert info["manifest_sha256"] == world.manifest_sha
    for event in C.EVENTS:
        record = info["events"][event]
        assert record["nll_after"] < record["nll_before"]
        assert abs(record["predicted_after"] - record["observed"]) < 0.01
    extra = loaded.meta["extra"]
    assert extra["dataset"]["manifest_sha256"] == world.manifest_sha  # carried over
    assert extra["created"] == "then"
    assert extra["event_calibration"]["source"]["name"] == "tiny"
    assert extra["event_calibration"]["calibration"]["applied"] is True
    # The network is the input's: without the calibration, the input's output.
    plain = world.predictor.predict(batch)
    stripped = predictor.with_event_calibration(None).predict(batch)
    assert all(np.array_equal(plain[name], stripped[name]) for name in plain)
    made = predictor.predict(batch)
    want = found.apply(plain["action"], batch["action_mask"], batch["cand_flag"])
    assert np.abs(made["action"] - want).max() < 1e-6


def test_the_script_checks_and_reports(
    world: World, finished: tuple[Path, dict[str, Any], str]
):
    out, report, log = finished
    checks = report["checks"]
    assert checks["passed"] is True and checks["failures"] == []
    assert checks["reloaded_calibration_equals_fitted"] is True
    assert checks["network_and_temperatures_unchanged"] is True
    assert checks["featurizer_unchanged"] is True
    assert checks["targets_and_mega_untouched"] is True
    assert checks["normalize_prediction_unchanged"] is True
    once = checks["applied_once"]
    assert once["max_abs_difference_to_one_application"] <= K.ONCE_TOLERANCE
    assert once["max_abs_difference_to_two_applications"] > 1e-3
    fine = checks["fine_nll"]
    assert fine["after"] < fine["before"] and fine["change"] == pytest.approx(
        fine["after"] - fine["before"]
    )
    assert fine["tolerance"] == K.FINE_TOLERANCE == 0.002
    assert "CHECK passed" in log
    # The stored payload says "carries a calibration"; the input's did not.
    assert checks["payload_version"] == M.PAYLOAD_VERSION_CALIBRATED == 2
    assert report["source"]["payload_version"] == M.PAYLOAD_VERSION == 1
    assert A.read_artifact(out / K.ARTIFACT_NAME)["predictor"]["version"] == 2
    assert A.read_artifact(world.source)["predictor"]["version"] == 1
    # The labels of this dataset do not depend on the turn: no term is taken.
    assert report["context_terms"] == {
        "offered": True,
        "names": list(C.CONTEXT),
        "margin": C.TERM_MARGIN,
    }
    for event in C.EVENTS:
        about = report["calibration"]["info"]["events"][event]["selection"]["context"]
        assert about["accepted"] is False and about["fitted"] is not None
        assert report["calibration"]["maps"][event]["terms"] == {}
    # Validation, then test and the ladder holdout: the same tables.
    assert set(report["informational"]) == set(K.INFORMATIONAL_SPLITS)
    assert report["thresholds"] == {
        C.EVENT_SWITCH: [0.35, 0.6],
        C.EVENT_PROTECT: [0.5, 0.6, 0.8],
    }
    held_out = world.sizes["val"] - world.base
    assert report["dataset"]["fit_examples"] == held_out
    assert report["dataset"]["validation_rows_left_out_holdout_battles"] == world.base
    assert report["validation"]["examples"] == held_out
    for name, split in [
        ("val", report["validation"]),
        *report["informational"].items(),
    ]:
        assert split["examples"] == (held_out if name == "val" else world.sizes[name])
        assert split["heads_untouched"] == {"target": True, "mega": True}
        assert split["sanity"]["after"]["unchanged"] is True
        assert split["fine_nll"]["change"] < 0
        fine = split["fine_nll"]
        paired = fine["paired_change"]
        assert paired["diff"] == pytest.approx(fine["change"])
        assert paired["low"] <= paired["diff"] <= paired["high"] < 0
        assert paired["slots"] == fine["before"]["slots"]
        # The change, split by slot-turns with a visible and a hidden action.
        shown, hidden = fine["paired_change_visible"], fine["paired_change_hidden"]
        assert shown["slots"] == fine["before"]["visible_slots"] > 0
        assert hidden["slots"] == fine["before"]["hidden_slots"] > 0
        assert shown["slots"] + hidden["slots"] == paired["slots"]
        assert shown["slots"] * shown["diff"] + hidden["slots"] * hidden["diff"] == (
            pytest.approx(paired["slots"] * paired["diff"])
        )
        assert fine["after"]["hidden"] == pytest.approx(
            fine["before"]["hidden"] + hidden["diff"]
        )
        for event in C.EVENTS:
            both = split["events"][event]
            seen, unseen = (split["populations"][event][kind] for kind in K.POPULATIONS)
            assert seen["slots"] + unseen["slots"] == both["slots"]
            assert seen["share_of_known"] + unseen["share_of_known"] == pytest.approx(1)
            # A hidden action is never the event, so the visible ones alone
            # show it more often than all slot-turns together ...
            assert unseen["slots"] > 0 and unseen["observed"] == 0.0
            assert seen["observed"] > both["observed"]
            # ... and after a fit over both kinds they read low on their own.
            assert seen["predicted_after"] < seen["observed"]
            assert unseen["predicted_after"] > 0.0
            assert [row["threshold"] for row in seen["thresholds"]] == list(
                K.THRESHOLDS[event]
            )
            total = seen["slots"] * seen["predicted_after"]
            total += unseen["slots"] * unseen["predicted_after"]
            assert total / both["slots"] == pytest.approx(both["after"]["predicted"])
        # By public context: each flag set and not set split the known rows.
        for event in C.EVENTS:
            groups = split["context"][event]
            assert list(groups) == [w for pair in K.CONTEXT_TEXT.values() for w in pair]
            both = split["events"][event]
            for yes, no in K.CONTEXT_TEXT.values():
                assert groups[yes]["slots"] + groups[no]["slots"] == both["slots"]
                assert groups[yes]["slots"] > 0 and groups[no]["slots"] > 0
                total = sum(
                    groups[w]["slots"] * groups[w]["after"]["predicted"]
                    for w in (yes, no)
                )
                assert total / both["slots"] == pytest.approx(
                    both["after"]["predicted"]
                )
                for word in (yes, no):
                    part = groups[word]
                    assert [r["threshold"] for r in part["thresholds"]] == list(
                        K.THRESHOLDS[event]
                    )
                    assert part["before"]["bias"]["value"] == pytest.approx(
                        part["before"]["predicted"] - part["observed"]
                    )
            fired = sum(
                groups[w]["thresholds"][0]["after"]["slots"]
                for w in K.CONTEXT_TEXT[C.CONTEXT_TURN_ONE]
            )
            assert fired == both["thresholds"][0]["after"]["slots"]
        # Joint reply coverage, before and after, on the counted examples.
        coverage = split["coverage"]
        assert coverage["k"] == S.JOINT_K and coverage["examples"] == split["examples"]
        assert 0 <= coverage["counted"] <= coverage["examples"]
        if coverage["counted"]:
            assert list(coverage["slices"]) == list(K.COVERAGE_SLICES)
            whole = coverage["slices"][S.SLICE_ALL]
            assert whole["examples"] == coverage["counted"]
            assert whole["top_change"]["diff"] == pytest.approx(
                whole["after"]["top"] - whole["before"]["top"]
            )
        assert set(split["attack"]) == set(K.ATTACK_EVENTS)
        for block in split["attack"].values():
            assert block["slots"] > 0 and block["observed"] == 0.0
            assert block["before"]["predicted"] != block["after"]["predicted"]
            assert block["before"]["brier"] > 0 and block["after"]["brier"] > 0
            assert [row["threshold"] for row in block["thresholds"]] == list(
                K.ATTACK_THRESHOLDS
            )
        for event in C.EVENTS:
            block = split["events"][event]
            assert block["slots"] > 0 and 0 < block["observed"] < 1
            assert [row["threshold"] for row in block["thresholds"]] == list(
                K.THRESHOLDS[event]
            )
            for side in K.SIDES:
                part = block[side]
                assert len(part["reliability"]) == S.BINS
                assert (
                    sum(row["slots"] for row in part["reliability"]) == block["slots"]
                )
                assert part["brier"] > 0 and part["ece"] >= 0 and part["nll"] > 0
                assert (
                    part["ece_if_calibrated"]["p95"]
                    >= part["ece_if_calibrated"]["mean"]
                )
                assert part["bias"]["value"] == pytest.approx(
                    part["predicted"] - block["observed"]
                )
            # The model was miscalibrated by construction; the fit removes it.
            assert abs(block["after"]["bias"]["value"]) < abs(
                block["before"]["bias"]["value"]
            )
            assert block["after"]["nll"] < block["before"]["nll"]
            assert block["after"]["brier"] < block["before"]["brier"]
            first = block["thresholds"][0]
            same = first["rows_fired_before"]
            assert same["slots"] == first["before"]["slots"]
            if event == C.EVENT_SWITCH:
                # It said too much: fewer slot-turns reach the threshold after,
                # and those that did before now read lower and nearer the truth.
                last = block["thresholds"][-1]
                assert first["before"]["slots"] >= first["after"]["slots"] > 0
                assert last["before"]["slots"] > last["after"]["slots"]
                assert same["predicted_before"] == pytest.approx(
                    first["before"]["predicted"]
                )
                assert same["predicted_after"] < same["predicted_before"]
                assert abs(same["predicted_after"] - same["happened"]) < abs(
                    same["predicted_before"] - same["happened"]
                )
                assert first["before"]["gap"]["value"] > 0.05
            else:
                # It said too little: nothing reached the threshold before.
                assert first["before"]["slots"] == 0 < first["after"]["slots"]
                assert first["before"]["happened"]["value"] is None
                assert first["before"]["predicted"] is None
                assert same["predicted_before"] is None and same["happened"] is None
            assert abs(first["after"]["gap"]["value"]) < 0.08
            for side in K.SIDES:
                row = first[side]
                if not row["slots"]:
                    continue
                assert row["happened"]["low"] <= row["happened"]["value"]
                assert row["happened"]["value"] <= row["happened"]["high"]
                assert row["gap"]["value"] == pytest.approx(
                    row["predicted"] - row["happened"]["value"]
                )
    for event in C.EVENTS:
        entry = report["shift"][event]
        assert set(entry["splits"]) == {"val", *K.INFORMATIONAL_SPLITS}
        left = entry["ladder_remaining"]
        ladder = report["informational"]["ladder_holdout"]["events"][event]
        assert left["bias_before"] == ladder["before"]["bias"]["value"]
        assert left["bias_after"] == ladder["after"]["bias"]["value"]
        assert left["bias_share_left"] == pytest.approx(
            abs(left["bias_after"]) / abs(left["bias_before"])
        )
        assert left["ece_share_left"] == pytest.approx(
            left["ece_after"] / left["ece_before"]
        )
    assert (
        sum(
            line.startswith("INFORMATIONAL (nothing fitted")
            for line in log.splitlines()
        )
        == 4
    )


def test_the_report_is_rendered_from_its_json(
    finished: tuple[Path, dict[str, Any], str], tmp_path: Path
):
    out, report, _ = finished
    text = (out / K.REPORT_MD).read_text(encoding="utf-8")
    assert text == K.render_report(report)
    assert text.startswith("# Event calibration: tiny_cal")
    for needle in (
        "Status: CAL_DONE",
        "## The maps",
        "## Checks on the reloaded artifact",
        "## At the pre-registered thresholds",
        "## Visible and hidden actions: what the calibrated numbers mean",
        "## By public context: turn 1, first turn on the field, protected last turn",
        "| validation | the slot switches out by choice | turn 1 |",
        "logistic with context terms",
        "context terms not taken:",
        "payload version 2 (the input's: 1)",
        "After the fit, among the rows above with at least 100 slot-turns",
        "## Side effect: the attack events",
        "## Ladder holdout: what a map fitted on human validation cannot fix",
        "## Reliability: validation",
        "INFORMATIONAL: nothing fitted here",
        "sigmoid(",
    ):
        assert needle in text, needle
    # --render-only rewrites the Markdown from the JSON and touches nothing else.
    copy = tmp_path / "copy"
    copy.mkdir()
    (copy / K.REPORT_JSON).write_text(json.dumps(report), encoding="utf-8")
    assert K.main(["--out", str(copy), "--render-only"]) == 0
    assert (copy / K.REPORT_MD).read_text(encoding="utf-8") == text
    assert sorted(path.name for path in copy.iterdir()) == [K.REPORT_JSON, K.REPORT_MD]
    assert K.main(["--out", str(tmp_path / "nowhere"), "--render-only"]) == 1
    # A report of a run that failed early still renders.
    assert "CAL_FAILED" in K.render_report({"status": K.FAILED, "reason": "why"})


def test_the_script_fits_on_validation_and_shows_the_predictor_features_only(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fitted: list[dict[str, Any]] = []
    seen: list[dict[str, Any]] = []
    loads: list[tuple[tuple[str, ...], bool]] = []
    out = tmp_path / "run"
    real_fit, real_predict, real_load = (
        C.fit_event_calibration,
        M.OppNetPredictor.predict,
        T.load_splits,
    )

    def fit(action: Any, data: Any, **options: Any) -> Any:
        fitted.append(
            {
                "splits": np.unique(data["m_split"]).tolist(),
                "flags": np.unique(data["m_flag"]).tolist(),
                "rows": int(np.asarray(action).shape[0]),
                "groups": np.asarray(options["groups"]).copy(),
                "battles": np.asarray(data["m_battle"]).copy(),
                "temperature": options["action_temperature"],
            }
        )
        return real_fit(action, data, **options)

    def predict(self: Any, data: Any) -> Any:
        seen.append(
            {
                "names": set(data),
                "sheets": np.unique(
                    np.asarray(data["game_flag"])[:, [F.G_ACTOR_SHEET, F.G_OTHER_SHEET]]
                ).tolist(),
            }
        )
        return real_predict(self, data)

    def load(directory: Any, names: Any) -> Any:
        loads.append((tuple(names), (out / K.ARTIFACT_NAME).is_file()))
        return real_load(directory, names)

    monkeypatch.setattr(C, "fit_event_calibration", fit)
    monkeypatch.setattr(M.OppNetPredictor, "predict", predict)
    monkeypatch.setattr(T, "load_splits", load)
    monkeypatch.setattr(K, "say", lambda text: None)
    assert world.run(out) == 0
    # One fit, on validation rows only, without the battle of a holdout opponent.
    assert len(fitted) == 1
    assert fitted[0]["splits"] == [SPLITS.index("val")] and fitted[0]["flags"] == [0]
    assert fitted[0]["rows"] == world.sizes["val"] - world.base
    assert np.array_equal(fitted[0]["groups"], fitted[0]["battles"])
    assert fitted[0]["temperature"] == 1.3
    # Test and the ladder holdout are read only once the artifact is written.
    assert loads == [(("val",), False), (K.INFORMATIONAL_SPLITS, True)]
    # Every predict call got features only, unknown sheets read as closed.
    assert len(seen) >= 6
    for call in seen:
        assert not [name for name in call["names"] if name.startswith(("y_", "m_"))]
        assert set(M.FEATURE_KEYS) <= call["names"]
        assert F.SHEET_UNKNOWN not in call["sheets"]


def test_an_output_directory_is_not_written_over(
    world: World,
    finished: tuple[Path, dict[str, Any], str],
    tmp_path: Path,
    capsys: Any,
):
    out, _, _ = finished
    digests = {path.name: sha256(path) for path in out.iterdir()}
    assert world.run(out) == 1
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.startswith(f"{K.FAILED} CalibrationError") and "--overwrite" in last
    assert {path.name: sha256(path) for path in out.iterdir()} == digests
    # Never the input artifact's own directory, with or without --overwrite.
    source_dir = world.source.parent
    before = sha256(world.source)
    for extra in ((), ("--overwrite",)):
        assert world.run(source_dir, *extra) == 1
        assert "own directory" in capsys.readouterr().out
    assert sorted(path.name for path in source_dir.iterdir()) == [K.ARTIFACT_NAME]
    assert sha256(world.source) == before
    # A directory that is not a calibration run is not replaced even with
    # --overwrite: another model's artifact is safe there.
    other = tmp_path / "other_model"
    other.mkdir()
    (other / K.ARTIFACT_NAME).write_bytes(b"another model")
    assert world.run(other, "--overwrite") == 1
    assert "not a calibration run" in capsys.readouterr().out
    assert (other / K.ARTIFACT_NAME).read_bytes() == b"another model"
    # A run directory of this script is replaced with --overwrite; a file the
    # script does not write is left where it is.
    again = tmp_path / "again"
    assert world.run(again) == 0
    (again / "notes.txt").write_text("keep", encoding="utf-8")
    first = json.loads((again / K.REPORT_JSON).read_text(encoding="utf-8"))
    assert world.run(again, "--overwrite") == 0
    capsys.readouterr()
    second = json.loads((again / K.REPORT_JSON).read_text(encoding="utf-8"))
    assert second["calibration"]["maps"] == first["calibration"]["maps"]
    assert (again / "notes.txt").read_text(encoding="utf-8") == "keep"
    # Bad arguments and missing inputs end with the status line too.
    assert K.main(["--artifact", "x"]) == 2
    assert capsys.readouterr().out.strip().endswith(f"{K.FAILED} bad arguments")
    assert world.run(tmp_path / "none", source=tmp_path / "missing.pt") == 1
    assert not (tmp_path / "none").exists()
    assert K.main(["--out", str(tmp_path / "none")]) == 1
    capsys.readouterr()


def test_a_refused_overwrite_leaves_the_earlier_run_as_it_was(
    world: World, tmp_path: Path, capsys: Any, monkeypatch: pytest.MonkeyPatch
):
    """--overwrite replaces an earlier run. A run that is refused for its
    arguments is no run: it must not take the earlier artifact with it (it
    used to remove artifact.pt first and refuse afterwards)."""
    out = tmp_path / "run"
    assert world.run(out) == 0
    capsys.readouterr()
    digests = {path.name: sha256(path) for path in out.iterdir()}
    assert set(digests) == {K.ARTIFACT_NAME, K.REPORT_JSON, K.REPORT_MD}
    other = tmp_path / "other_model" / "artifact.pt"
    other.parent.mkdir()
    A.save_artifact(
        other,
        kind=M.KIND,
        name="tiny",
        featurizer=world.fz,
        predictor_payload=world.predictor.to_payload(),
        extra={"dataset": {"tag": "older", "manifest_sha256": "0" * 64}},
    )
    stacked = tmp_path / "calibrated_copy" / "artifact.pt"
    stacked.parent.mkdir()
    stacked.write_bytes((out / K.ARTIFACT_NAME).read_bytes())
    refusals: list[tuple[tuple[str, ...], Path | None, str]] = [
        (("--overwrite",), other, "--allow-other-dataset"),  # another dataset build
        (("--overwrite",), stacked, "already carries"),  # calibrated, no --refit
        (("--overwrite",), out / K.ARTIFACT_NAME, "own directory"),
        (("--overwrite", "--suffix", ""), None, "needs a name suffix"),
        (("--overwrite", "--limit", "-3"), None, "is negative"),
        (("--overwrite",), tmp_path / "missing.pt", "no artifact at"),
    ]
    for extra, source, needle in refusals:
        assert world.run(out, *extra, source=source) == 1, (extra, source)
        last = capsys.readouterr().out.strip().splitlines()[-1]
        assert last.startswith(f"{K.FAILED} CalibrationError") and needle in last
        assert {path.name: sha256(path) for path in out.iterdir()} == digests, needle
    # an artifact of a kind that carries no calibration
    real = A.load_predictor
    with monkeypatch.context() as patch:
        patch.setattr(
            A,
            "load_predictor",
            lambda path, *a, **k: real(path, *a, **k)._replace(kind=A.KIND_TABLE),
        )
        assert world.run(out, "--overwrite") == 1
    assert "only 'oppnet' carries" in capsys.readouterr().out
    assert {path.name: sha256(path) for path in out.iterdir()} == digests
    # A refused run into a NEW directory leaves no directory behind either.
    fresh = tmp_path / "fresh"
    assert world.run(fresh, source=other) == 1
    capsys.readouterr()
    assert not fresh.exists()
    # ... and a run that is accepted does replace the earlier one.
    assert world.run(out, "--overwrite") == 0
    capsys.readouterr()
    assert {path.name for path in out.iterdir()} == set(digests)
    assert sha256(out / K.REPORT_JSON) != digests[K.REPORT_JSON]  # a new report


def test_a_limited_run_never_takes_the_name_of_the_real_calibration(
    world: World, tmp_path: Path, capsys: Any
):
    """--limit fits the maps on the first N validation examples: a smoke. Its
    artifact used to be named like the real one, and the name is what a
    forecast and the decision audit carry."""
    assert K.calibrated_name("tiny", "_cal", None) == "tiny_cal"
    assert K.calibrated_name("tiny", "_cal", 600) == "tiny_cal_limit600"
    assert K.calibrated_name("tiny", "_cal", 0) == "tiny_cal_limit0"
    # a refit keeps one suffix, whichever kind of run made the source
    for source in ("tiny_cal", "tiny_cal_limit600"):
        assert K.calibrated_name(source, "_cal", None, refit=True) == "tiny_cal"
        assert K.calibrated_name(source, "_cal", 50, refit=True) == "tiny_cal_limit50"
    assert K.calibrated_name("tiny_cal", "_cal", None) == "tiny_cal_cal"  # not a refit
    assert K.calibrated_name("_cal", "_cal", None, refit=True) == "_cal_cal"
    for limit in (0, 1, 600, 10**9):
        for source in ("tiny", "tiny_cal"):
            for refit in (False, True):
                name = K.calibrated_name(source, "_cal", limit, refit=refit)
                assert name.endswith(f"{K.LIMIT_MARK}{limit}")
                assert name != K.calibrated_name(source, "_cal", None, refit=refit)
    out = tmp_path / "smoke"
    assert world.run(out, "--limit", "600") == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("CAL_START name tiny_cal_limit600 source tiny")
    assert "LIMITED to 600 examples per split: a smoke" in lines[0]
    assert lines[-1].startswith(f"{K.DONE} name tiny_cal_limit600 maps ")
    report = json.loads((out / K.REPORT_JSON).read_text(encoding="utf-8"))
    assert report["name"] == "tiny_cal_limit600" and report["dataset"]["limit"] == 600
    loaded = A.load_predictor(out / K.ARTIFACT_NAME)
    assert loaded.name == loaded.predictor.name == "tiny_cal_limit600"
    whole = tmp_path / "whole"
    assert world.run(whole) == 0
    capsys.readouterr()
    assert A.load_predictor(whole / K.ARTIFACT_NAME).name == "tiny_cal"
    # a refit of the smoke under no limit is the real name again
    again = tmp_path / "again"
    assert world.run(again, "--refit", source=out / K.ARTIFACT_NAME) == 0
    capsys.readouterr()
    assert A.load_predictor(again / K.ARTIFACT_NAME).name == "tiny_cal"


def test_context_terms_go_through_the_script_and_its_checks(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
):
    """A calibration whose map carries terms: the script must hand the context
    to every application it checks, store the terms, and print them."""
    real_fit = C.fit_event_calibration
    terms = ((C.CONTEXT_TURN_ONE, -0.05), (C.CONTEXT_PROTECTED_LAST, 0.04))

    def fitted_with_terms(action: Any, data: Any, **options: Any) -> Any:
        fitted = real_fit(action, data, **options)
        switch = C.EventMap(
            C.MAP_LOGISTIC, fitted.switch.slope, fitted.switch.bias, terms=terms
        )
        return C.EventCalibration(
            switch, fitted.protect, fitted.action_temperature, fitted.info
        )

    monkeypatch.setattr(C, "fit_event_calibration", fitted_with_terms)
    out = tmp_path / "run"
    assert world.run(out) == 0, capsys.readouterr().out[-400:]
    log = capsys.readouterr().out
    assert "terms turn 1 -0.0500, protected last turn +0.0400" in log
    report = json.loads((out / K.REPORT_JSON).read_text(encoding="utf-8"))
    assert report["checks"]["passed"] is True
    once = report["checks"]["applied_once"]
    assert once["max_abs_difference_to_one_application"] <= K.ONCE_TOLERANCE
    assert once["max_abs_difference_to_two_applications"] > 1e-3
    assert report["calibration"]["version"] == C.VERSION_CONTEXT
    assert report["calibration"]["maps"][C.EVENT_SWITCH]["terms"] == dict(terms)
    predictor = A.load_predictor(out / K.ARTIFACT_NAME).predictor
    assert isinstance(predictor, M.OppNetPredictor)
    found = predictor.event_calibration
    assert found is not None and found.needs_context
    assert found.switch.terms == terms
    # The artifact's output is one application WITH the context ...
    base = world.features
    plain = world.predictor.predict(base)["action"]
    mask, flags = base["action_mask"], base["cand_flag"]
    want = found.apply_checked(plain, mask, flags, C.event_context(base))
    made = predictor.predict(base)["action"]
    assert np.abs(made - want).max() < 1e-6
    # ... which is not the map without its terms.
    bare = C.EventCalibration(
        C.EventMap(C.MAP_LOGISTIC, found.switch.slope, found.switch.bias),
        found.protect,
        found.action_temperature,
    )
    assert np.abs(made - bare.apply_checked(plain, mask, flags)).max() > 1e-4
    text = (out / K.REPORT_MD).read_text(encoding="utf-8")
    assert "- 0.0500 [turn 1] + 0.0400 [protected last turn])" in text


def test_a_calibrated_payload_stored_as_version_one_fails_the_run(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
):
    """The file the script writes must be one an old reader refuses."""
    real = A.save_artifact

    def as_version_one(path: Any, **options: Any) -> None:
        payload = dict(options["predictor_payload"], version=M.PAYLOAD_VERSION)
        real(path, **{**options, "predictor_payload": payload})

    monkeypatch.setattr(A, "save_artifact", as_version_one)
    out = tmp_path / "run"
    assert world.run(out) == 1
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.startswith(K.FAILED) and "must not carry an event calibration" in last
    assert not (out / K.ARTIFACT_NAME).exists()
    report = json.loads((out / K.REPORT_JSON).read_text(encoding="utf-8"))
    assert report["status"] == K.FAILED and report["artifact"] is None


def test_the_script_itself_refuses_to_keep_a_calibrated_version_one_file(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
):
    """The model code as it was: a calibrated payload written as version 1,
    and a reader that takes it. The script's own check does not lean on the
    model's refusal: such a file is never kept as artifact.pt."""
    assert K.OLD_READER_VERSION == M.PAYLOAD_VERSION == 1
    real_save, real_read = A.save_artifact, M.from_payload

    def as_version_one(path: Any, **options: Any) -> None:
        payload = dict(options["predictor_payload"], version=M.PAYLOAD_VERSION)
        real_save(path, **{**options, "predictor_payload": payload})

    def lenient(payload: Any, featurizer: Any, device: Any = "cpu") -> Any:
        fixed = dict(payload)
        if fixed.get("event_calibration") is not None:
            fixed["version"] = M.PAYLOAD_VERSION_CALIBRATED
        return real_read(fixed, featurizer, device)

    monkeypatch.setattr(A, "save_artifact", as_version_one)
    monkeypatch.setattr(M, "from_payload", lenient)
    out = tmp_path / "run"
    assert world.run(out) == 1
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1].startswith(f"{K.FAILED} CalibrationError")
    assert "a calibrated payload must not be version 1" in lines[-1]
    assert not (out / K.ARTIFACT_NAME).exists()
    assert (out / (K.ARTIFACT_NAME + K.UNVERIFIED_SUFFIX)).is_file()
    report = json.loads((out / K.REPORT_JSON).read_text(encoding="utf-8"))
    assert report["status"] == K.FAILED and report["checks"]["payload_version"] == 1
    assert len(report["checks"]["failures"]) == 1  # nothing else is wrong with it


def test_the_context_table_is_the_events_counted_by_flag(
    batch: F.Batch, predictor: M.OppNetPredictor
):
    """``context_report`` against a count written out by hand."""
    before = predictor.predict(batch)
    after = predictor.with_event_calibration(with_terms()).predict(batch)
    found = K.context_report(before, after, batch, resamples=50, seed=3)
    labels = S.event_labels(batch)
    context = C.event_context(batch)
    groups = K.context_groups(batch)
    assert list(groups) == [word for pair in K.CONTEXT_TEXT.values() for word in pair]
    for index, name in enumerate(C.CONTEXT):
        yes, no = K.CONTEXT_TEXT[name]
        assert np.array_equal(groups[yes], context[..., index])
        assert np.array_equal(groups[no], ~context[..., index])
    for event in C.EVENTS:
        known, happened = labels[event]
        for side, pred in (("before", before), ("after", after)):
            p = S.event_predictions(pred, batch)[event]
            for word, rows in groups.items():
                part = found[event][word]
                where = known & rows
                assert part["slots"] == int(where.sum())
                if not where.any():
                    assert part["observed"] is None
                    continue
                assert part["observed"] == pytest.approx(happened[where].mean())
                assert part[side]["predicted"] == pytest.approx(p[where].mean())
                assert part[side]["nll"] == pytest.approx(
                    C.binary_nll(p[where], happened[where])
                )
                for row in part["thresholds"]:
                    fired = where & (p >= row["threshold"])
                    assert row[side]["slots"] == int(fired.sum())
                    if fired.any():
                        assert row[side]["happened"]["value"] == pytest.approx(
                            happened[fired].mean()
                        )
                        assert row[side]["gap"]["value"] == pytest.approx(
                            p[fired].mean() - happened[fired].mean()
                        )
    # A batch without the arrays the flags are read from: no table, no failure.
    blind = {k: v for k, v in batch.items() if k != "mon_flag"}
    assert K.context_groups(blind) == {}
    assert K.context_report(before, after, blind) == {}


def test_the_coverage_side_effect_is_the_scorecards_coverage():
    """``coverage_report`` is ``scorecard.joint_coverage_set`` read twice, plus
    the paired change."""
    data = TS.coverage_batch()
    before = TS.dyadic(data)
    after = {name: value.copy() for name, value in before.items()}
    after["action"][:, :, 1] += 0.25  # more Protect, as a calibration might not
    after["action"] /= after["action"].sum(-1, keepdims=True)
    found = K.coverage_report(before, after, data, resamples=100, seed=2)
    card = S.joint_coverage_set(
        data, {"before": before, "after": after}, resamples=100, seed=2
    )
    assert found["examples"] == 7 and found["counted"] == card["counted"]["examples"]
    assert found["k"] == S.JOINT_K and list(found["slices"]) == list(K.COVERAGE_SLICES)
    for label in K.COVERAGE_SLICES:
        part = found["slices"][label]
        assert part["examples"] == card["slices"][label]["examples"]
        for side in K.SIDES:
            row = card["predictors"][side][S.JOINT_PLAIN]["slices"][label]
            assert part[side]["top"] == pytest.approx(row["top"][str(S.JOINT_K)])
            assert part[side]["top1"] == pytest.approx(row["top"]["1"])
            assert part[side]["log_prob"] == pytest.approx(row["log_prob"])
        change = part["top_change"]
        assert change["diff"] == pytest.approx(
            part["after"]["top"] - part["before"]["top"]
        )
        assert change["slots"] == part["examples"]
        assert part["log_prob_change"]["diff"] == pytest.approx(
            part["after"]["log_prob"] - part["before"]["log_prob"]
        )
    # identical predictions: no change at all, an interval of zero width
    same = K.coverage_report(before, before, data, resamples=50)
    whole = same["slices"][S.SLICE_ALL]
    assert whole["top_change"]["diff"] == 0.0 == whole["log_prob_change"]["diff"]
    assert whole["top_change"]["low"] == whole["top_change"]["high"] == 0.0
    # labels that cannot be read as replies, or nothing counted: said, not raised
    assert K.coverage_report(before, after, {"act_mon": data["act_mon"]}) == {}
    hidden = dict(data, y_action=np.full_like(data["y_action"], -1))
    none = K.coverage_report(before, after, hidden)
    assert none["counted"] == 0 and none["slices"] == {}
    with pytest.raises(K.CalibrationError, match="joint replies"):
        K.coverage_report({**before, "action": before["action"][:, :, :3]}, after, data)
    J.COUNTERS.clear()


def test_a_calibration_is_never_stacked_on_another(
    world: World,
    finished: tuple[Path, dict[str, Any], str],
    tmp_path: Path,
    capsys: Any,
):
    out, report, _ = finished
    calibrated = out / K.ARTIFACT_NAME
    # The script's own output is refused as an input ...
    assert world.run(tmp_path / "stacked", source=calibrated) == 1
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.startswith(K.FAILED) and "already carries an event calibration" in last
    assert not (tmp_path / "stacked" / K.ARTIFACT_NAME).exists()
    # ... unless told to drop it and fit anew: the same maps as the first fit,
    # not maps of maps, and the name keeps one suffix.
    refit = tmp_path / "refit"
    assert world.run(refit, "--refit", source=calibrated) == 0
    capsys.readouterr()
    second = json.loads((refit / K.REPORT_JSON).read_text(encoding="utf-8"))
    assert second["source"]["had_event_calibration"] is True
    assert second["name"] == "tiny_cal"
    assert second["calibration"]["maps"] == report["calibration"]["maps"]
    loaded = A.load_predictor(refit / K.ARTIFACT_NAME).predictor
    first = A.load_predictor(calibrated).predictor
    assert loaded.event_calibration.maps == first.event_calibration.maps


def test_a_calibration_that_hurts_fails_the_run(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
):
    """The fine NLL on validation may not get worse by more than 0.002 nats."""

    real_fit = C.fit_event_calibration

    def harmful(action: Any, data: Any, **options: Any) -> C.EventCalibration:
        fitted = real_fit(action, data, **options)
        sharp = C.EventMap(C.MAP_LOGISTIC, 3.0, 1.0)
        return C.EventCalibration(sharp, sharp, fitted.action_temperature, fitted.info)

    monkeypatch.setattr(C, "fit_event_calibration", harmful)
    out = tmp_path / "run"
    assert world.run(out) == 1
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1].startswith(f"{K.FAILED} CalibrationError")
    assert "validation fine NLL got worse" in lines[-1]
    assert any(line.startswith("CHECK FAILED") for line in lines)
    # No artifact.pt; the rejected file keeps its other name; the numbers stay.
    assert not (out / K.ARTIFACT_NAME).exists()
    assert (out / (K.ARTIFACT_NAME + K.UNVERIFIED_SUFFIX)).is_file()
    report = json.loads((out / K.REPORT_JSON).read_text(encoding="utf-8"))
    assert report["status"] == K.FAILED and report["artifact"] is None
    assert report["rejected_artifact"].endswith(K.UNVERIFIED_SUFFIX)
    fine = report["checks"]["fine_nll"]
    assert fine["change"] > K.FINE_TOLERANCE and report["checks"]["passed"] is False
    assert "informational" not in report  # test and ladder were never read
    assert "CAL_FAILED" in (out / K.REPORT_MD).read_text(encoding="utf-8")


def test_a_changed_target_head_fails_the_run(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
):
    """Calibration is of the action distribution only."""
    real = M.OppNetPredictor.predict

    def tampered(self: Any, data: Any) -> Any:
        made = real(self, data)
        if self.event_calibrated:
            uniform = F.uniform_prediction(data)["target"]
            made = dict(made, target=(0.5 * made["target"] + 0.5 * uniform))
            made["target"] = made["target"].astype(np.float32)
        return made

    monkeypatch.setattr(M.OppNetPredictor, "predict", tampered)
    out = tmp_path / "run"
    assert world.run(out) == 1
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.startswith(f"{K.FAILED} CalibrationError")
    assert "the target or Mega probabilities changed" in last
    report = json.loads((out / K.REPORT_JSON).read_text(encoding="utf-8"))
    assert report["checks"]["failures"] == ["the target or Mega probabilities changed"]
    assert report["validation"]["heads_untouched"] == {"target": False, "mega": True}
    assert not (out / K.ARTIFACT_NAME).exists()


@pytest.mark.parametrize("how", ["twice", "nudged"])
def test_output_that_is_not_one_application_fails_the_run(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any, how: str
):
    """The reloaded predictor must give exactly one application of the maps to
    the input's output: not two, and not something close to one."""
    real = M.OppNetPredictor.predict

    def tampered(self: Any, data: Any) -> Any:
        made = real(self, data)
        if not self.event_calibrated:
            return made
        made = dict(made)
        if how == "twice":
            made["action"] = self.event_calibration.apply(
                made["action"], data["action_mask"], data["cand_flag"]
            ).astype(np.float32)
        else:  # still normalised, fine NLL within the tolerance: only this check
            uniform = F.uniform_prediction(data)["action"]
            made["action"] = (0.999 * made["action"] + 0.001 * uniform).astype(
                np.float32
            )
        return made

    monkeypatch.setattr(M.OppNetPredictor, "predict", tampered)
    out = tmp_path / "run"
    assert world.run(out) == 1
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.startswith(f"{K.FAILED} CalibrationError")
    assert "from one application" in last
    assert ("look calibrated twice" in last) is (how == "twice")
    assert not (out / K.ARTIFACT_NAME).exists()
    report = json.loads((out / K.REPORT_JSON).read_text(encoding="utf-8"))
    once = report["checks"]["applied_once"]
    assert once["max_abs_difference_to_one_application"] > K.ONCE_TOLERANCE
    if how == "twice":
        assert once["max_abs_difference_to_two_applications"] <= K.ONCE_TOLERANCE
    else:
        assert len(report["checks"]["failures"]) == 1
        assert report["checks"]["fine_nll"]["change"] < K.FINE_TOLERANCE


@pytest.mark.parametrize("what", ["weights", "temperature", "featurizer"])
def test_a_stored_model_that_is_not_the_inputs_fails_the_run(
    world: World,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
    what: str,
):
    """Calibration changes no weight, no temperature and no vocabulary."""
    real = A.save_artifact

    def tampered(path: Any, **options: Any) -> None:
        payload = dict(options["predictor_payload"])
        if what == "weights":
            state = {
                name: value.clone() for name, value in payload["state_dict"].items()
            }
            state["switch_bias"] += 1e-3
            payload["state_dict"] = state
        elif what == "temperature":
            payload["temperatures"] = {**payload["temperatures"], "target": 2.0}
        else:
            other = TM.repertoire()
            other.add("garchomp", "ironhead", 40)
            options["featurizer"] = F.Featurizer.build(other)
        real(path, **{**options, "predictor_payload": payload})

    monkeypatch.setattr(A, "save_artifact", tampered)
    out = tmp_path / "run"
    assert world.run(out) == 1
    last = capsys.readouterr().out.strip().splitlines()[-1]
    wanted = "featurizer differs" if what == "featurizer" else "network or temperatures"
    assert last.startswith(f"{K.FAILED} CalibrationError") and wanted in last
    assert not (out / K.ARTIFACT_NAME).exists()


def test_another_dataset_build_is_refused(world: World, tmp_path: Path, capsys: Any):
    other = tmp_path / "model" / "artifact.pt"
    other.parent.mkdir()
    A.save_artifact(
        other,
        kind=M.KIND,
        name="tiny",
        featurizer=world.fz,
        predictor_payload=world.predictor.to_payload(),
        extra={"dataset": {"tag": "older", "manifest_sha256": "0" * 64}},
    )
    out = tmp_path / "run"
    assert world.run(out, source=other) == 1
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.startswith(K.FAILED) and "--allow-other-dataset" in last
    assert not (out / K.ARTIFACT_NAME).exists()
    assert world.run(out, "--overwrite", "--allow-other-dataset", source=other) == 0
    capsys.readouterr()
    report = json.loads((out / K.REPORT_JSON).read_text(encoding="utf-8"))
    assert report["dataset"]["same_build_as_training"] is False
    assert report["source"]["trained_on_manifest_sha256"] == "0" * 64


def test_a_count_table_artifact_is_refused(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
):
    real = A.load_predictor

    def as_table(path: Any, *args: Any, **options: Any) -> Any:
        return real(path, *args, **options)._replace(kind=A.KIND_TABLE)

    monkeypatch.setattr(A, "load_predictor", as_table)
    assert world.run(tmp_path / "run") == 1
    assert "only 'oppnet' carries" in capsys.readouterr().out


def test_the_limit_and_the_count_table_quotation(
    world: World, tmp_path: Path, capsys: Any
):
    card = tmp_path / "scorecard.json"
    events = {
        event: {"slots": 10, "predicted": 0.2, "observed": 0.1} for event in C.EVENTS
    }
    card.write_text(
        json.dumps(
            {
                "sets": {
                    "ladder_holdout": {
                        "predictors": {"flags_table": {"events": events}}
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "run"
    assert world.run(out, "--limit", "600", "--scorecard", str(card)) == 0
    capsys.readouterr()
    report = json.loads((out / K.REPORT_JSON).read_text(encoding="utf-8"))
    assert report["dataset"]["fit_examples"] == 600
    assert report["validation"]["examples"] == 600
    assert all(s["examples"] == 600 for s in report["informational"].values())
    assert report["count_table"]["predictor"] == "flags_table"
    assert report["count_table"]["events"][C.EVENT_SWITCH]["predicted"] == 0.2
    assert "The count table `flags_table`" in (out / K.REPORT_MD).read_text("utf-8")
    # A card without that predictor costs the quotation, not the run; a card
    # that is not there stops the run before anything is written.
    other = tmp_path / "other"
    assert world.run(other, "--scorecard", str(card), "--scorecard-predictor", "x") == 0
    assert "WARNING" in capsys.readouterr().out
    report = json.loads((other / K.REPORT_JSON).read_text(encoding="utf-8"))
    assert report["count_table"] is None and report["warnings"]
    assert world.run(tmp_path / "third", "--scorecard", str(tmp_path / "no.json")) == 1
    assert not (tmp_path / "third").exists()
    capsys.readouterr()


def test_rows_still_off_after_the_fit_are_called_out():
    """The report names the calls whose gap after the fit excludes zero, so a
    residual (turn 1, the highest threshold) is not left to be found in a
    table."""

    def side(slots: int, gap: tuple[Any, Any, Any] | None) -> dict[str, Any]:
        found = (
            None if gap is None else {"value": gap[0], "low": gap[1], "high": gap[2]}
        )
        return {"slots": slots, "predicted": 0.44, "gap": found}

    rows = [
        ("validation, turn 1", side(204, (0.067, 0.001, 0.128))),  # high
        ("test, later turns", side(3035, (-0.004, -0.022, 0.014))),  # fine
        ("test, at 0.8", side(80, (-0.078, -0.125, -0.023))),  # too few to call
        ("ladder, protect", side(208, (-0.072, -0.135, -0.006))),  # low
        ("one game", side(500, (0.2, None, None))),  # no interval: not called
        ("nothing fired", side(0, None)),
    ]
    lines = K._still_off(rows)
    listed = [line for line in lines if line.startswith("- ")]
    assert listed == [
        "- validation, turn 1: 204 said 44.0%, gap +6.7 [+0.1, +12.8]",
        "- ladder, protect: 208 said 44.0%, gap -7.2 [-13.5, -0.6]",
    ]
    assert f"at least {K.STILL_OFF_SLOTS} slot-turns" in lines[1]
    none = K._still_off(rows[1:3])
    assert none[-1].endswith("excludes zero in none.") and len(none) == 2


def test_report_helpers():
    rng = np.random.default_rng(3)
    p = rng.random(5000)
    y = rng.random(5000) < p
    table = S.reliability_table(p, y)
    by_hand = (
        sum(row["slots"] * abs(row["predicted"] - row["observed"]) for row in table)
        / 5000
    )
    assert K.ece(p, y) == pytest.approx(by_hand)
    assert K.ece(p, p > 2) == pytest.approx(p.mean())  # nothing ever happens
    chance = K.ece_if_calibrated(p, np.random.default_rng(0), draws=100)
    assert chance["mean"] < chance["p95"] < 0.03 and K.ece(p, y) < 0.03
    assert K.ece_if_calibrated(np.zeros(0), rng) == {"mean": None, "p95": None}
    assert np.isnan(K.ece(np.zeros(0), np.zeros(0)))
    assert K.same({"a": [1, 2.0, np.arange(3)]}, {"a": [1, 2.0, np.arange(3)]})
    assert not K.same({"a": np.arange(3)}, {"a": np.arange(3).astype(np.int32)})
    assert not K.same({"a": 1}, {"a": 1.0}) and not K.same({"a": 1}, {"b": 1})
    assert K.same(torch.ones(2), torch.ones(2)) and not K.same(
        torch.ones(2), np.ones(2)
    )
    assert K.same(float("nan"), float("nan")) and not K.same([1], [1, 2])
    assert K.trained_on_manifest({"dataset": {"manifest_sha256": "abc"}}) == "abc"
    assert K.trained_on_manifest({"manifest_sha256": "top"}) == "top"
    assert K.trained_on_manifest({}) is None


# --- library rules --------------------------------------------------------------


def library_paths() -> list[Path]:
    return [
        ROOT / "vgc_bench/src/oppmodel/calibration.py",
        ROOT / "training/calibrate_oppmodel.py",
    ]


def test_library_holds_no_species_move_item_or_ability_name(fz: F.Featurizer):
    """String literals of the library and the script name nothing from the dex."""
    known = set(E.moves_dex()) | set(E.pokedex())
    known |= set(fz.vocab.items[3:]) | set(fz.vocab.abilities[3:])
    for path in library_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            )
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        }
        found = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and node.value in known
        }
        assert not found, (path.name, sorted(found))


def test_library_has_no_bare_assert():
    for path in library_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        lines = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert)]
        assert not lines, (path.name, lines)


def test_the_library_reads_no_split_and_fits_nothing_by_itself():
    """``calibration.py`` has no file access and names no split: which rows a
    map is fitted on is decided by its caller alone."""
    tree = ast.parse(library_paths()[0].read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (
            [ast.alias(node.module or "")]
            if isinstance(node, ast.ImportFrom)
            else node.names
        )
    }
    assert imported <= {
        "__future__",
        "math",
        "collections",
        "dataclasses",
        "typing",
        "numpy",
        "vgc_bench",
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not names & {"open", "load_dataset", "load_batch"}
    strings = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert not strings & {"m_split", "test", "ladder_holdout", "val", "train"}
