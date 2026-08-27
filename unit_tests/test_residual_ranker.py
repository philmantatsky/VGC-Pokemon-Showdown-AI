from __future__ import annotations

from types import SimpleNamespace as NS

import torch

from vgc_bench.src.residual_ranker import (
    ACTION_STRUCTURE_DIM,
    ResidualConfig,
    ResidualJointRanker,
    candidate_semantic_features,
)
from vgc_bench.src.utils import (
    chunk_obs_len,
    glob_obs_len,
    knowledge_obs_len,
    pokemon_obs_len,
    semantics_obs_len,
    side_obs_len,
)


def test_residual_ranker_initializes_as_low_confidence_noop():
    model = ResidualJointRanker(ResidualConfig(feature_dim=8, action_count=12))
    features = torch.randn(3, 8)
    actions = torch.randint(0, 12, (3, 5, 2))
    champion = torch.randn(3, 5)
    adjusted, confidence, residual = model(features, actions, champion)
    assert torch.equal(adjusted, champion)
    assert torch.count_nonzero(residual) == 0
    assert torch.all(confidence < model.config.confidence_threshold)


def test_residual_checkpoint_round_trip(tmp_path):
    model = ResidualJointRanker(ResidualConfig(feature_dim=4, action_count=7))
    path = tmp_path / "ranker.pt"
    model.save(path, {"source": "champion.zip"})
    loaded = ResidualJointRanker.load(path)
    assert loaded.config == model.config
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            model.state_dict().values(), loaded.state_dict().values()
        )
    )


def test_candidate_shape_matches_joint_actions():
    candidates = [NS(actions=(1, 2), prob=0.7), NS(actions=(3, 4), prob=0.3)]
    actions = torch.as_tensor([[candidate.actions for candidate in candidates]])
    assert actions.shape == (1, 2, 2)


class _SemanticExtractor:
    features_dim = 2
    embed_len = 3

    def pokemon_tokens(self, obs_dict):
        batch = obs_dict["observation"].shape[0]
        base = torch.arange(12, dtype=torch.float32).view(1, 12, 1)
        return torch.cat([base, base + 0.5], dim=2).expand(batch, -1, -1)

    def move_embed(self, move_ids):
        values = move_ids.to(torch.float32)
        return torch.stack([values, values + 1, values + 2], dim=-1)


def test_candidate_semantics_resolve_move_and_switch_identity():
    observation = torch.zeros((1, 12, chunk_obs_len))
    pokemon_start = glob_obs_len + side_obs_len
    active_a = pokemon_start + pokemon_obs_len - (
        2 + 6 + 1 + knowledge_obs_len + semantics_obs_len
    )
    observation[0, 2, active_a] = 1
    observation[0, 4, active_a + 1] = 1
    observation[0, 2, pokemon_start + 2 : pokemon_start + 6] = torch.tensor(
        [10, 11, 12, 13]
    )
    # Slot 0: move slot 2 aimed at foe +1. Slot 1: switch to party index 4.
    actions = torch.tensor([[[20, 4]]])
    policy = NS(features_extractor=_SemanticExtractor())

    result = candidate_semantic_features(
        policy,
        {"observation": observation.reshape(1, -1)},
        actions,
    )

    assert result.shape == (1, 1, 2, 2 + 3 + ACTION_STRUCTURE_DIM)
    assert result[0, 0, 0, :2].tolist() == [2.0, 2.5]
    assert result[0, 0, 0, 2:5].tolist() == [12.0, 13.0, 14.0]
    assert result[0, 0, 1, :2].tolist() == [3.0, 3.5]
    assert result[0, 0, 1, 2:5].tolist() == [0.0, 0.0, 0.0]


def test_contextual_ranker_starts_as_noop_with_semantic_candidates():
    config = ResidualConfig(
        feature_dim=8,
        action_count=12,
        candidate_feature_dim=5,
        contextual_confidence=True,
    )
    model = ResidualJointRanker(config)
    champion = torch.randn(3, 4)
    adjusted, confidence, residual = model(
        torch.randn(3, 8),
        torch.randint(0, 12, (3, 4, 2)),
        champion,
        torch.randn(3, 4, 2, 5),
        torch.ones(3, 4, dtype=torch.bool),
    )

    assert torch.equal(adjusted, champion)
    assert torch.count_nonzero(residual) == 0
    assert torch.all(confidence < config.confidence_threshold)
