"""The KL anchor for practice (vgc_bench/src/anchored_ppo.py, 2026-10-04): plain
practice from a fine-tuned brain washed the fine-tune out, so the anchored PPO pays
beta * KL(start || policy) per minibatch, with beta adapted toward a target."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from vgc_bench.src.anchored_ppo import (
    BETA_MAX,
    BETA_MIN,
    AnchoredPPO,
    kl_from_log_probs,
    next_beta,
    slot_log_probs,
)

ROOT = Path(__file__).resolve().parents[1]
BRAIN = ROOT / "results_deployed/champion_mc_T6tac.zip"
DATA = ROOT / "results_tactical_t6e_ep1/data"


def test_kl_matches_the_definition_and_ignores_masked_actions():
    q = torch.log(torch.tensor([[0.5, 0.5, 0.0]]))
    p = torch.log(torch.tensor([[0.25, 0.75, 0.0]]))
    expected = 0.5 * np.log(0.5 / 0.25) + 0.5 * np.log(0.5 / 0.75)
    assert kl_from_log_probs(q, p).item() == pytest.approx(expected, rel=1e-5)
    assert kl_from_log_probs(q, q).item() == pytest.approx(0.0, abs=1e-7)
    assert torch.isfinite(kl_from_log_probs(q, p)).all()  # -inf masks add nothing


def test_beta_follows_the_target():
    assert next_beta(1.0, 0.30, 0.10) == pytest.approx(1.5)  # too far: pull harder
    assert next_beta(1.0, 0.01, 0.10) == pytest.approx(1 / 1.5)  # slack: loosen
    assert next_beta(1.0, 0.10, 0.10) == 1.0  # within the band
    assert next_beta(BETA_MAX, 9.0, 0.10) == BETA_MAX
    assert next_beta(BETA_MIN, 0.0, 0.10) == BETA_MIN


def test_the_anchor_never_enters_a_save():
    model = object.__new__(AnchoredPPO)
    assert "anchor_policy" in model._excluded_save_params()


@pytest.mark.skipif(
    not BRAIN.exists() or not any(DATA.glob("*.npz")),
    reason="needs the deployed brain and recorded T6e positions",
)
def test_anchor_kl_is_zero_at_the_start_and_pulls_back_after_a_step():
    from stable_baselines3 import PPO

    from training.tactical_sft import load_data

    data = load_data(DATA)
    rows = np.arange(16)
    obs = {
        "observation": torch.as_tensor(data["obs"][rows]),
        "action_mask": torch.as_tensor(data["mask"][rows].astype(np.int64)),
    }
    played0 = torch.as_tensor(data["played"][rows, :1].astype(np.int64))
    policy = PPO.load(BRAIN, device="cpu").policy
    anchor = copy.deepcopy(policy).eval()
    lp0, lp1, _, _ = slot_log_probs(policy, obs, played0, True)
    with torch.no_grad():
        q0, q1, _, _ = slot_log_probs(anchor, obs, played0, False)
    start = kl_from_log_probs(q0, lp0) + kl_from_log_probs(q1, lp1)
    assert start.abs().max().item() < 1e-5
    with torch.no_grad():
        for p in policy.action_net.parameters():
            p.add_(0.05 * torch.randn_like(p))
    lp0, lp1, _, _ = slot_log_probs(policy, obs, played0, True)
    moved = (kl_from_log_probs(q0, lp0) + kl_from_log_probs(q1, lp1)).mean()
    assert moved.item() > 1e-4
    moved.backward()
    grads = [p.grad for p in policy.action_net.parameters()]
    assert all(g is not None and g.abs().sum() > 0 for g in grads)
