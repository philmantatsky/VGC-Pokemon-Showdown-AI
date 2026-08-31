from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pytest

from datagen.generate_outcome_dataset import _sample_rows
from training.train_outcome_value import _split
from vgc_bench.src.exact_planner import ExactNode
from vgc_bench.src.outcome_value import (
    OutcomeValueEvaluator,
    calibration_error,
    probability_to_value,
)


def test_even_state_sampling_keeps_opening_and_endgame():
    rows = [{"index": index} for index in range(20)]

    sampled = _sample_rows(rows, 8)

    assert len(sampled) == 8
    assert sampled[0]["index"] == 0
    assert sampled[-1]["index"] == 19


def test_probability_value_conversion_and_calibration():
    assert probability_to_value(0.0) == -1.0
    assert probability_to_value(0.5) == 0.0
    assert probability_to_value(1.0) == 1.0
    error = calibration_error(np.array([0.1, 0.9]), np.array([0.0, 1.0]))
    assert error == pytest.approx(0.1)


def test_terminal_outcome_bypasses_the_network():
    policy = NS(eval=lambda: policy)
    evaluator = OutcomeValueEvaluator(policy, mechanics_weight=0)
    won = ExactNode(
        state={"sides": [{"name": "planner"}, {"name": "opponent"}]},
        requests=[None, None],
        turn=7,
        request_state=None,
        ended=True,
        winner="planner",
    )
    lost = ExactNode(
        state=won.state,
        requests=won.requests,
        turn=won.turn,
        request_state=None,
        ended=True,
        winner="opponent",
    )

    assert evaluator.probability(won, "p1") == 1.0
    assert evaluator.probability(lost, "p1") == 0.0


def test_outcome_split_is_disjoint_by_opponent_team():
    dataset = NS(opponents=[f"team-{index % 20}" for index in range(200)])

    indices, groups = _split(dataset, seed=17)

    assert all(indices.values())
    assert not (set(groups["train"]) & set(groups["validation"]))
    assert not (set(groups["train"]) & set(groups["test"]))
    assert not (set(groups["validation"]) & set(groups["test"]))
