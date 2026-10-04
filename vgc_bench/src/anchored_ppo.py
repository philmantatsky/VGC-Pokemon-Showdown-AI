"""PPO with a KL anchor to the brain practice starts from (2026-10-04).

Practice from a fine-tuned brain washed the fine-tune out: plain PPO from T6tac
moved the policy about as far from its start (KL 0.52 nats on slot 1 over
983,040 steps) as the practice that had improved T6hp (0.48), and that distance
overwrote the tactical fine-tune's smaller, precise lessons (0.24 nats; the
practised brain lost 35.2% head-to-head to its start). This adds

    beta * KL(anchor || policy)

to the PPO loss on every minibatch: slot 1's distribution, plus slot 2's
conditioned on the slot-1 action actually played (as the joint head samples),
from a frozen copy of the start brain. beta adapts after each update toward a
target KL (the PPO paper's adaptive-penalty rule, applied to the anchor instead
of the previous policy): above 1.5x the target it grows by 1.5x, below
target / 1.5 it shrinks by 1.5x, within [BETA_MIN, BETA_MAX]. The anchor never
enters a save.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F

BETA_MIN = 0.05
BETA_MAX = 100.0
BETA_STEP = 1.5


def slot_log_probs(
    policy: Any, obs: dict[str, th.Tensor], played0: th.Tensor, actor_grad: bool
) -> tuple[th.Tensor, th.Tensor, Any, th.Tensor]:
    """(slot-1 log-probs, slot-2 log-probs | played slot-1 action, the joint
    distribution, the value logits) from one forward pass of ``policy``."""
    logits, value_logits, latent = policy.logits_with_latent(obs, actor_grad)
    dist = policy.get_dist_from_logits(logits, obs["action_mask"])
    dist2 = policy.get_dist_from_logits(logits, obs["action_mask"], played0, latent)
    dist.distribution[1] = dist2.distribution[1]
    lp0 = th.log_softmax(dist.distribution[0].logits, dim=-1)
    lp1 = th.log_softmax(dist.distribution[1].logits, dim=-1)
    return lp0, lp1, dist, value_logits


def kl_from_log_probs(q: th.Tensor, lp: th.Tensor) -> th.Tensor:
    """Per-row KL(q || p) from log-probabilities; masked actions (-inf) add 0."""
    pq = q.exp()
    keep = pq > 0
    terms = th.where(keep, pq * (q - lp), th.zeros_like(lp))
    return terms.sum(-1)


def next_beta(beta: float, kl: float, target: float) -> float:
    """The adaptive-penalty rule: keep the measured anchor KL near ``target``."""
    if kl > 1.5 * target:
        beta *= BETA_STEP
    elif kl < target / 1.5:
        beta /= BETA_STEP
    return float(min(max(beta, BETA_MIN), BETA_MAX))


class AnchoredPPO(PPO):
    """PPO whose loss also pays beta * KL(anchor || policy) per minibatch."""

    def __init__(
        self,
        *args: Any,
        anchor_kl_target: float = 0.1,
        anchor_coef: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not anchor_kl_target > 0:
            raise ValueError("anchor_kl_target must be positive")
        self.anchor_policy: Any = None
        self.anchor_kl_target = float(anchor_kl_target)
        self.anchor_coef = float(anchor_coef)

    def set_anchor(self, policy: Any) -> None:
        """The frozen start brain (same architecture as ``self.policy``)."""
        self.anchor_policy = policy.eval()
        for p in self.anchor_policy.parameters():
            p.requires_grad_(False)

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "anchor_policy"]

    def anchor_terms(
        self, obs: dict[str, th.Tensor], actions: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """(values, log_prob, entropy, per-row anchor KL) for a minibatch."""
        if self.anchor_policy is None:
            raise RuntimeError("AnchoredPPO trains only with an anchor: set_anchor()")
        played0 = actions[:, :1].long()
        actor_grad = bool(getattr(self.policy, "actor_grad", True))
        lp0, lp1, dist, values = slot_log_probs(self.policy, obs, played0, actor_grad)
        with th.no_grad():
            q0, q1, _, _ = slot_log_probs(self.anchor_policy, obs, played0, False)
        kl = kl_from_log_probs(q0, lp0) + kl_from_log_probs(q1, lp1)
        return values, dist.log_prob(actions), dist.entropy(), kl

    def train(self) -> None:
        """stable_baselines3 2.8 PPO.train with the anchor term (and its logs)."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses, pg_losses, value_losses = [], [], []
        clip_fractions, anchor_kls = [], []
        approx_kl_divs: list[float] = []
        loss = th.zeros(())
        continue_training = True
        for _ in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()
                values, log_prob, entropy, kl = self.anchor_terms(
                    rollout_data.observations,  # type: ignore[arg-type]
                    actions,
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(
                    th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                )
                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())
                anchor_kl = kl.mean()
                anchor_kls.append(anchor_kl.item())
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.anchor_coef * anchor_kl
                )
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(float(approx_kl_div))
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()
            self._n_updates += 1
            if not continue_training:
                break

        measured = float(np.mean(anchor_kls)) if anchor_kls else 0.0
        self.logger.record("train/anchor_kl", measured)
        self.logger.record("train/anchor_coef", self.anchor_coef)
        self.anchor_coef = next_beta(self.anchor_coef, measured, self.anchor_kl_target)
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
