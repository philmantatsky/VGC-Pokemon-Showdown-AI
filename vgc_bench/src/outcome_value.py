"""Calibrated terminal-outcome value model used by exact future search."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from vgc_bench.src.exact_observation import state_to_battle
from vgc_bench.src.exact_planner import ExactNode, strategic_value
from vgc_bench.src.policy_player import PolicyPlayer


def critic_logits(policy, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run only the policy's value path and interpret its scalar as a logit."""
    if policy.share_features_extractor:
        features = policy.features_extractor(obs_dict)
        _latent_pi, latent_vf = policy.mlp_extractor(features)
    else:
        vf_features = policy.vf_features_extractor(obs_dict)
        latent_vf = policy.mlp_extractor.forward_critic(vf_features)
    return policy.value_net(latent_vf).squeeze(-1)


def probability_to_value(probability: float) -> float:
    """Convert calibrated P(win) to the planner's [-1, 1] value convention."""
    return max(-1.0, min(1.0, 2.0 * probability - 1.0))


class OutcomeValueEvaluator:
    """Evaluate concrete exact states with a calibrated outcome critic.

    Hidden-information search calls this once per sampled determinization.  It always
    sees that concrete particle; uncertainty is aggregated by the planner rather than
    leaked into one impossible-to-interpret observation.
    """

    def __init__(
        self,
        policy,
        *,
        temperature: float = 1.0,
        mechanics_weight: float = 0.10,
    ):
        if temperature <= 0:
            raise ValueError("outcome temperature must be positive")
        if not 0 <= mechanics_weight <= 1:
            raise ValueError("mechanics_weight must be in [0, 1]")
        self.policy = policy.eval()
        self.temperature = float(temperature)
        self.mechanics_weight = float(mechanics_weight)
        self._cache: dict[tuple[str, int, str], float] = {}

    @classmethod
    def load(
        cls,
        checkpoint: Path,
        *,
        device: str | torch.device = "cpu",
        mechanics_weight: float = 0.10,
    ) -> "OutcomeValueEvaluator":
        model = PPO.load(checkpoint, device=device)
        metrics_path = checkpoint.with_suffix(".metrics.json")
        payload = json.loads(metrics_path.read_text())
        return cls(
            model.policy,
            temperature=float(payload["calibration"]["temperature"]),
            mechanics_weight=mechanics_weight,
        )

    def probability(self, node: ExactNode, role: str = "p1") -> float:
        terminal = node.ended
        if terminal:
            if not node.winner:
                return 0.5
            side = node.state["sides"][int(role[1]) - 1]
            return 1.0 if node.winner in {role, side.get("name")} else 0.0
        digest = hashlib.blake2b(
            (
                "\n".join(node.state.get("inputLog") or node.state.get("log") or [])
                + str(node.state.get("prng"))
            ).encode(),
            digest_size=12,
        ).hexdigest()
        key = (role, node.turn, digest)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        battle = state_to_battle(
            node.state, node.requests, role, reveal_opponent_sets=True
        )
        obs = PolicyPlayer.embed_battle(battle, fake_rating=2000)
        obs_dict = {
            "observation": torch.as_tensor(
                obs, dtype=torch.float32, device=self.policy.device
            ).unsqueeze(0),
            # The value extractor ignores this mask, but retaining the policy's
            # normal dictionary interface prevents architecture-specific call paths.
            "action_mask": torch.ones(
                (1, 214), dtype=torch.float32, device=self.policy.device
            ),
        }
        with torch.no_grad():
            logit = float(critic_logits(self.policy, obs_dict).item())
        probability = 1.0 / (1.0 + math.exp(-logit / self.temperature))
        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[key] = probability
        return probability

    def __call__(self, node: ExactNode, role: str = "p1") -> float:
        learned = probability_to_value(self.probability(node, role))
        if self.mechanics_weight == 0 or node.ended:
            return learned
        try:
            mechanics = strategic_value(node, role)
        except (
            AssertionError,
            KeyError,
            NotImplementedError,
            RuntimeError,
            ValueError,
        ):
            mechanics = learned
        return float(
            (1.0 - self.mechanics_weight) * learned
            + self.mechanics_weight * mechanics
        )


def calibration_error(
    probabilities: np.ndarray, targets: np.ndarray, bins: int = 10
) -> float:
    """Expected calibration error for reporting and promotion gates."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if not len(probabilities):
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (probabilities >= edges[index]) & (
                probabilities <= edges[index + 1]
            )
        else:
            selected = (probabilities >= edges[index]) & (
                probabilities < edges[index + 1]
            )
        if not selected.any():
            continue
        total += float(selected.mean()) * abs(
            float(probabilities[selected].mean()) - float(targets[selected].mean())
        )
    return total
