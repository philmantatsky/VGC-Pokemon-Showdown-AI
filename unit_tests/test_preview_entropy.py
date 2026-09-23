"""training/preview_entropy.py: the preview flag index matches embed_global, the
mask reads it from flattened or token-shaped observations, and the patch scales
only preview samples' entropy, once."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from training import preview_entropy
from training.preview_entropy import PREVIEW_INDEX, install, preview_mask
from vgc_bench.src.policy import MaskedActorCriticPolicy
from vgc_bench.src.policy_player import PolicyPlayer
from vgc_bench.src.utils import chunk_obs_len, glob_obs_len


def _battle(teampreview: bool) -> SimpleNamespace:
    return SimpleNamespace(
        weather={},
        fields={},
        turn=0,
        format="gen9championsvgc2026regmc",
        teampreview=teampreview,
        reviving=False,
        commanding=False,
    )


def test_index_is_the_teampreview_feature_of_embed_global() -> None:
    on = PolicyPlayer.embed_global(_battle(True))  # type: ignore[arg-type]
    off = PolicyPlayer.embed_global(_battle(False))  # type: ignore[arg-type]
    assert len(on) == glob_obs_len
    assert on[PREVIEW_INDEX] == 1.0 and off[PREVIEW_INDEX] == 0.0
    assert (on != off).sum() == 1  # the only difference is the preview flag


def test_mask_reads_the_first_token_of_flat_or_shaped_observations() -> None:
    obs = torch.zeros(3, 12 * chunk_obs_len)
    obs[0, PREVIEW_INDEX] = 1.0
    obs[2, PREVIEW_INDEX] = 1.0
    assert preview_mask(obs).tolist() == [1.0, 0.0, 1.0]
    assert preview_mask(obs.reshape(3, 12, chunk_obs_len)).tolist() == [1.0, 0.0, 1.0]


def test_patch_scales_only_preview_entropy(monkeypatch: pytest.MonkeyPatch) -> None:
    def stub(self, obs, actions):
        n = obs["observation"].shape[0]
        return torch.zeros(n, 1), torch.full((n,), -1.0), torch.ones(n)

    monkeypatch.setattr(MaskedActorCriticPolicy, "evaluate_actions", stub)
    install(4.0)
    obs = torch.zeros(2, 12 * chunk_obs_len)
    obs[1, PREVIEW_INDEX] = 1.0
    patched: Any = MaskedActorCriticPolicy.evaluate_actions
    values, log_prob, entropy = patched(
        SimpleNamespace(), {"observation": obs, "action_mask": None}, None
    )
    assert entropy.tolist() == [1.0, 5.0]
    assert log_prob.tolist() == [-1.0, -1.0] and values.shape == (2, 1)
    with pytest.raises(RuntimeError, match="already installed"):
        install(4.0)


def test_negative_boost_refused() -> None:
    with pytest.raises(ValueError):
        preview_entropy.install(-1.0)
