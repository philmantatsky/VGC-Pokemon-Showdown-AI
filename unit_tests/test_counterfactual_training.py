from types import SimpleNamespace as NS

import pytest
import torch

from training.train_counterfactual import joint_log_prob, planner_confidence
from training.train_residual_ranker import _select_gate_threshold


def test_planner_confidence_ignores_ties_and_weights_clear_preferences():
    scores = torch.tensor(
        [
            [0.50, 0.498, 0.10],
            [0.50, 0.40, 0.39],
            [0.20, float("nan"), 0.10],
        ]
    )
    valid = torch.isfinite(scores)
    safe, best, margin, confidence = planner_confidence(
        scores, valid, minimum_margin=0.005, margin_scale=0.10
    )

    assert best.tolist() == [0, 0, 0]
    assert confidence[0] == 0
    assert float(confidence[1]) >= 0.999
    assert 0.99 <= float(confidence[2]) <= 1.0
    assert torch.isneginf(safe[2, 1])
    assert float(margin[1]) > float(margin[0])


class _RecordingPolicy:
    def __init__(self):
        self.conditioning_actions = None

    def get_dist_from_logits(self, logits, _mask, first_actions=None):
        if first_actions is not None:
            self.conditioning_actions = first_actions.clone()
        batch = logits.shape[0]
        dist = NS(logits=torch.zeros((batch, 5), device=logits.device))
        return NS(distribution=[dist, dist])


def test_joint_log_prob_never_conditions_on_padding():
    policy = _RecordingPolicy()
    actions = torch.tensor([[[2, 3], [-1, -1], [-1, -1]]])

    result = joint_log_prob(
        policy,
        torch.zeros((1, 10)),
        torch.ones((1, 10)),
        actions,
    )

    assert result.shape == (1, 3)
    assert policy.conditioning_actions.flatten().tolist() == [2, 2, 2]


def test_joint_log_prob_rejects_rows_without_candidates():
    with pytest.raises(ValueError, match="at least one candidate"):
        joint_log_prob(
            _RecordingPolicy(),
            torch.zeros((1, 10)),
            torch.ones((1, 10)),
            torch.full((1, 2, 2), -1),
        )


def test_gate_calibration_uses_only_precise_conservative_overrides():
    # High-confidence changes repair two champion mistakes; low-confidence changes
    # are noisy and should remain gated off.
    result = _select_gate_threshold(
        confidence=[0.9, 0.8, 0.2, 0.1, 0.05],
        champion_pick=[0, 0, 1, 1, 0],
        adjusted_pick=[1, 1, 0, 0, 1],
        desired=[1, 1, 1, 1, 0],
        max_changed_fraction=0.40,
        minimum_gain=0.01,
    )

    assert result["changed_count"] == 2
    assert result["threshold"] == pytest.approx(0.8)
    assert result["deployed_rank_accuracy"] == pytest.approx(1.0)


def test_gate_calibration_disables_ranker_without_held_out_gain():
    result = _select_gate_threshold(
        confidence=[0.9, 0.8],
        champion_pick=[0, 1],
        adjusted_pick=[1, 0],
        desired=[0, 1],
        max_changed_fraction=1.0,
        minimum_gain=0.01,
    )

    assert result["changed_count"] == 0
    assert result["deployed_rank_accuracy"] == 1.0
    assert result["threshold"] > 0.9


def test_dataset_normalizes_candidate_width_across_chunks(tmp_path):
    """Chunks pad to their own widest beam; the loader must align them globally.

    v5h chunks came out width 10 and 11 (wider determinization unions), and
    torch's default collate crashed stacking ragged rows mid-pipeline.
    """
    import numpy as np
    from torch.utils.data import DataLoader

    from datagen.generate_counterfactuals import _save_chunk
    from training.train_counterfactual import CounterfactualDataset

    def example(candidates: int) -> dict:
        return {
            "observation": np.zeros(8, dtype=np.float16),
            "action_mask": np.zeros(4, dtype=np.uint8),
            "actions": np.asarray(
                [(1, 2)] * candidates, dtype=np.int16
            ).reshape(candidates, 2),
            "scores": np.linspace(0.1, 0.9, candidates).astype(np.float32),
            "expected": np.zeros(candidates, dtype=np.float32),
            "priors": np.zeros(candidates, dtype=np.float32),
            "metadata": {"game": 0},
        }

    _save_chunk(tmp_path, 0, [example(3)])
    _save_chunk(tmp_path, 1, [example(5)])
    dataset = CounterfactualDataset(tmp_path)
    assert {row[2].shape for row in dataset.rows} == {(5, 2)}
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch[2].shape == (2, 5, 2)
    # padding follows the established convention
    assert (batch[2][0, 3:] == -1).all()
