"""The calibrated critic leaf (2026-10-04): evaluation/leaf_calibration.py fits a
monotone raw-value -> win-probability map per phase of the game, and
vgc_bench/src/critic_leaf.py applies it."""

from __future__ import annotations

import json
import random
import threading
from types import SimpleNamespace as NS

import pytest
import torch

from evaluation.leaf_calibration import auc, fit, fit_knots, isotonic, phase_of
from vgc_bench.src.critic_leaf import CriticLeafEvaluator, LeafCalibration


def test_isotonic_pools_violators_and_keeps_the_weighted_mean():
    blocks = isotonic([(0.0, 0.2, 1), (1.0, 0.6, 1), (2.0, 0.4, 3), (3.0, 0.9, 1)])
    assert [round(y, 3) for _x, y, _w in blocks] == [0.2, 0.45, 0.9]
    assert blocks[1][2] == 4 and blocks[1][0] == pytest.approx((1.0 + 2.0 * 3) / 4)


def test_knots_are_monotone_and_follow_the_data():
    rng = random.Random(1)
    samples = []
    for _ in range(4000):
        raw = rng.uniform(-1.2, 1.2)
        # flat and uninformative below +0.5, steep above: the shape seen on 10-04
        truth = 0.35 if raw < 0.5 else 0.35 + 0.8 * (raw - 0.5)
        samples.append((raw, float(rng.random() < truth)))
    knots = fit_knots(samples)
    assert knots == sorted(knots) and all(0 <= p <= 1 for _raw, p in knots)
    assert all(b[1] >= a[1] for a, b in zip(knots, knots[1:]))
    low = [p for raw, p in knots if raw < 0.3]
    assert max(low) - min(low) < 0.12  # the flat stretch stays flat
    assert knots[-1][1] > 0.75
    with pytest.raises(ValueError, match="no samples"):
        fit_knots([])


def test_auc_counts_ties_as_half_and_needs_both_outcomes():
    assert auc([(0.1, 0.0), (0.9, 1.0)]) == 1.0
    assert auc([(0.9, 0.0), (0.1, 1.0)]) == 0.0
    assert auc([(0.5, 0.0), (0.5, 1.0)]) == 0.5
    assert auc([(0.5, 1.0), (0.7, 1.0)]) is None


def test_phases_split_by_turn():
    assert [phase_of(turn, [3, 6]) for turn in (1, 3, 4, 6, 7, 30)] == [
        0,
        0,
        1,
        1,
        2,
        2,
    ]


def test_fit_reports_each_phase_and_beats_the_clipped_raw_value_held_out():
    rng = random.Random(2)
    rows = []
    for game in range(600):
        won = float(rng.random() < 0.5)
        for turn in range(1, 10):
            # early: an optimistic constant; late: informative but far off scale
            raw = 0.7 + rng.gauss(0, 0.1) if turn <= 3 else rng.gauss(won - 0.2, 0.5)
            rows.append({"battle": f"g{game}", "turn": turn, "raw": raw, "won": won})
    summary = fit(rows, [3, 6])
    early, middle, late = summary["phases"]
    assert [p["through_turn"] for p in summary["phases"]] == [3, 6, None]
    assert summary["battles"] == 600 and early["samples"] == 1800
    assert early["auc"] == pytest.approx(0.5, abs=0.05)
    assert late["auc"] > 0.8
    # an uninformative phase calibrates to (about) the base rate everywhere
    assert (
        max(p for _raw, p in early["knots"]) - min(p for _r, p in early["knots"]) < 0.15
    )
    for phase in (early, late):
        assert phase["brier_calibrated_held_out"] < phase["brier_raw_clipped"]
    calibration = LeafCalibration.from_payload(json.loads(json.dumps(summary)))
    assert calibration.win_probability(0.7, 2) == pytest.approx(0.5, abs=0.1)
    assert calibration.win_probability(1.0, 8) > calibration.win_probability(-1.0, 8)


def test_calibration_interpolates_clamps_and_picks_the_phase():
    calibration = LeafCalibration.from_payload(
        {
            "phases": [
                {"through_turn": 3, "knots": [[0.0, 0.5]]},
                {"through_turn": None, "knots": [[-1.0, 0.1], [0.0, 0.3], [1.0, 0.9]]},
            ]
        }
    )
    assert calibration.win_probability(0.9, 2) == 0.5
    assert calibration.win_probability(-3.0, 5) == 0.1
    assert calibration.win_probability(3.0, 5) == 0.9
    assert calibration.win_probability(0.5, 5) == pytest.approx(0.6)
    assert calibration.win_probability(-0.5, 5) == pytest.approx(0.2)


@pytest.mark.parametrize(
    "phases, message",
    [
        ([{"through_turn": 3, "knots": [[0.0, 0.5]]}], "open-ended"),
        ([{"through_turn": None, "knots": []}], "at least one knot"),
        ([{"through_turn": None, "knots": [[0.0, 0.6], [1.0, 0.4]]}], "ascend"),
        ([{"through_turn": None, "knots": [[0.0, 1.4]]}], r"\[0, 1\]"),
    ],
)
def test_calibration_refuses_a_malformed_file(phases, message):
    with pytest.raises(ValueError, match=message):
        LeafCalibration.from_payload({"phases": phases})


def test_leaf_returns_twice_the_calibrated_win_probability_minus_one():
    battle = NS(team={}, opponent_team={}, turn=5)  # potential 0: six unseen each
    calibration = LeafCalibration(((None, ((-1.0, 0.2), (1.0, 0.8))),))
    for raw, expected in ((0.0, 0.0), (1.0, 0.6), (5.0, 0.6), (-1.0, -0.6)):
        adapter = NS(
            _inputs=lambda *_args: (battle, None, None, {}),
            inference_lock=threading.Lock(),
            policy=NS(
                get_logits=lambda _obs, actor_grad=False, raw=raw: (
                    torch.zeros(1),
                    torch.tensor([[raw]]),
                )
            ),
        )
        node = NS(state={}, requests=[{}, {}])
        leaf = CriticLeafEvaluator(adapter, calibration=calibration)
        assert leaf(node, "p1") == pytest.approx(expected)
