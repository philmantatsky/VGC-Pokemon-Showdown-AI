"""Conservative joint-action residual layered over the frozen champion.

The champion remains the deployed policy and supplies both state features and joint
action log-probabilities.  This small network may only adjust a candidate prefix when
its learned confidence clears a threshold; it cannot mutate champion parameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from vgc_bench.src.utils import (
    act_len,
    chunk_obs_len,
    glob_obs_len,
    knowledge_obs_len,
    pokemon_obs_len,
    semantics_obs_len,
    side_obs_len,
)

ACTION_STRUCTURE_DIM = 3 + 5 + 5 + 4


@dataclass(frozen=True)
class ResidualConfig:
    feature_dim: int
    action_count: int = act_len
    action_embedding_dim: int = 24
    candidate_feature_dim: int = 0
    contextual_confidence: bool = False
    hidden_dim: int = 128
    # Champion joint log-probability gaps are large: when its preferred pair is
    # tactically wrong, the planner's winner is typically 6.8 log units behind and
    # the 75th percentile is 10.9. A 1.25 cap could mathematically never change most
    # wrong picks. Eight permits correction while confidence gating and residual L2
    # keep uncertain states at the untouched champion ranking.
    max_adjustment: float = 12.0
    confidence_threshold: float = 0.62


@dataclass(frozen=True)
class ResidualReport:
    confidence: float
    applied: bool
    changed: bool
    before: tuple[tuple[int, int], ...]
    after: tuple[tuple[int, int], ...]


class ResidualJointRanker(nn.Module):
    def __init__(self, config: ResidualConfig):
        super().__init__()
        self.config = config
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(config.feature_dim),
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.action_embedding = nn.Embedding(
            config.action_count, config.action_embedding_dim
        )
        candidate_descriptor_dim = (
            2 * config.action_embedding_dim
            + 2 * config.candidate_feature_dim
            + 1
        )
        candidate_dim = config.hidden_dim + candidate_descriptor_dim
        self.residual_head = nn.Sequential(
            nn.Linear(candidate_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        confidence_input_dim = config.hidden_dim + (
            candidate_descriptor_dim if config.contextual_confidence else 0
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(confidence_input_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        # Start as an exact no-op even if a checkpoint is accidentally enabled before
        # training. Confidence also starts below the deployment threshold.
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        nn.init.zeros_(self.confidence_head[-1].weight)
        nn.init.constant_(self.confidence_head[-1].bias, -2.0)

    def forward(
        self,
        state_features: torch.Tensor,
        actions: torch.Tensor,
        champion_log_prob: torch.Tensor,
        candidate_features: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return adjusted scores, confidence, and bounded raw residuals."""
        encoded = self.state_encoder(state_features)
        embedded = self.action_embedding(actions.clamp(0, self.config.action_count - 1))
        expanded = encoded[:, None, :].expand(-1, actions.shape[1], -1)
        descriptors = [
            embedded[:, :, 0],
            embedded[:, :, 1],
        ]
        if self.config.candidate_feature_dim:
            expected = (
                actions.shape[0],
                actions.shape[1],
                2,
                self.config.candidate_feature_dim,
            )
            if candidate_features is None or candidate_features.shape != expected:
                shape = None if candidate_features is None else candidate_features.shape
                raise ValueError(
                    f"candidate features must have shape {expected}, got {shape}"
                )
            descriptors.extend(
                [candidate_features[:, :, 0], candidate_features[:, :, 1]]
            )
        descriptor = torch.cat(
            [*descriptors, champion_log_prob.unsqueeze(-1)], dim=-1
        )
        inputs = torch.cat([expanded, descriptor], dim=-1)
        residual = torch.tanh(self.residual_head(inputs).squeeze(-1))
        confidence_inputs = encoded
        if self.config.contextual_confidence:
            if candidate_mask is None:
                candidate_mask = torch.ones_like(champion_log_prob, dtype=torch.bool)
            weights = candidate_mask.to(descriptor.dtype).unsqueeze(-1)
            pooled = (descriptor * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(
                1.0
            )
            confidence_inputs = torch.cat([encoded, pooled], dim=-1)
        confidence = torch.sigmoid(
            self.confidence_head(confidence_inputs).squeeze(-1)
        )
        # Confidence decides whether a correction is deployed; it must not also
        # shrink the correction during training. The old double use limited a
        # 6.8-logit tactical correction to less than one logit in early epochs.
        adjusted = champion_log_prob + self.config.max_adjustment * residual
        return adjusted, confidence, residual

    def save(self, path: Path, metadata: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": 2,
                "config": asdict(self.config),
                "state_dict": self.state_dict(),
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: torch.device | str = "cpu"):
        payload = torch.load(path, map_location=device, weights_only=True)
        if int(payload.get("schema", 0)) not in {1, 2}:
            raise ValueError(f"unsupported residual schema in {path}")
        model = cls(ResidualConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        return model.to(device).eval()


def champion_features(policy, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """Frozen actor-side state representation from the preserved champion."""
    with torch.no_grad():
        features = policy.extract_features(obs_dict)
        if isinstance(features, tuple):
            features = features[0]
    return features.detach()


def candidate_semantic_dim(policy) -> int:
    """Per-slot feature size for a concrete switch or selected move."""
    extractor = policy.features_extractor
    if not hasattr(extractor, "pokemon_tokens"):
        raise TypeError("policy extractor does not expose Pokemon token features")
    return int(extractor.features_dim + extractor.embed_len + ACTION_STRUCTURE_DIM)


@torch.no_grad()
def candidate_semantic_features(
    policy,
    obs_dict: dict[str, torch.Tensor],
    actions: torch.Tensor,
) -> torch.Tensor:
    """Describe the actual Pokemon, move, target, and mechanic for each action.

    Raw action IDs only encode slots. For example, action 9 can mean Earthquake in
    one state and Protect in another. These features resolve the slot against the
    observation and reuse the champion's frozen Pokemon and move embeddings.
    """
    if actions.ndim != 3 or actions.shape[-1] != 2:
        raise ValueError("joint actions must have shape (batch, candidates, 2)")
    extractor = policy.features_extractor
    observations = obs_dict["observation"]
    batch, width, _slots = actions.shape
    chunks = observations.view(batch, 12, chunk_obs_len)
    own_chunks = chunks[:, :6]
    own_tokens = extractor.pokemon_tokens(obs_dict)[:, :6]
    pokemon_start = glob_obs_len + side_obs_len
    active_a = pokemon_start + pokemon_obs_len - (
        2 + 6 + 1 + knowledge_obs_len + semantics_obs_len
    )
    active_flags = torch.stack(
        [own_chunks[:, :, active_a], own_chunks[:, :, active_a + 1]], dim=1
    )
    move_ids = own_chunks[:, :, pokemon_start + 2 : pokemon_start + 6].long()
    slot_features = []
    safe_actions = actions.clamp(0, act_len - 1)
    for slot in range(2):
        action = safe_actions[:, :, slot]
        is_switch = (action >= 1) & (action <= 6)
        is_move = action >= 7
        is_pass = ~(is_switch | is_move)

        active_index = active_flags[:, slot].argmax(dim=1)
        has_active = active_flags[:, slot].amax(dim=1) > 0
        selected_index = torch.where(
            is_switch,
            (action - 1).clamp(0, 5),
            active_index[:, None].expand(-1, width),
        )
        gather_tokens = selected_index[:, :, None].expand(
            -1, -1, own_tokens.shape[-1]
        )
        selected_tokens = own_tokens.gather(1, gather_tokens)
        token_present = is_switch | (is_move & has_active[:, None])
        selected_tokens = selected_tokens * token_present[:, :, None]

        gather_moves = selected_index[:, :, None].expand(-1, -1, 4)
        selected_move_ids = move_ids.gather(1, gather_moves)
        within_band = (action - 7).clamp_min(0).remainder(20)
        move_slot = torch.div(within_band, 5, rounding_mode="floor").clamp(0, 3)
        move_id = selected_move_ids.gather(2, move_slot[:, :, None]).squeeze(-1)
        move_embedding = extractor.move_embed(move_id)
        move_embedding = move_embedding * is_move[:, :, None]

        kind = torch.stack([is_pass, is_switch, is_move], dim=-1).to(
            observations.dtype
        )
        mechanic_index = torch.div(
            (action - 7).clamp_min(0), 20, rounding_mode="floor"
        ).clamp(0, 4)
        mechanic = F.one_hot(mechanic_index, 5).to(observations.dtype)
        mechanic = mechanic * is_move[:, :, None]
        target_index = within_band.remainder(5)
        target = F.one_hot(target_index, 5).to(observations.dtype)
        target = target * is_move[:, :, None]
        move_slot_one_hot = F.one_hot(move_slot, 4).to(observations.dtype)
        move_slot_one_hot = move_slot_one_hot * is_move[:, :, None]
        structure = torch.cat(
            [kind, mechanic, target, move_slot_one_hot], dim=-1
        )
        slot_features.append(
            torch.cat([selected_tokens, move_embedding, structure], dim=-1)
        )
    result = torch.stack(slot_features, dim=2)
    expected_dim = candidate_semantic_dim(policy)
    if result.shape != (batch, width, 2, expected_dim):
        raise RuntimeError(
            f"candidate semantic shape mismatch: {result.shape} != "
            f"{(batch, width, 2, expected_dim)}"
        )
    return result.detach()


def rerank_candidates(
    ranker: ResidualJointRanker,
    policy,
    obs_dict: dict[str, torch.Tensor],
    candidates: Sequence,
) -> tuple[list, ResidualReport]:
    before = tuple(candidate.actions for candidate in candidates)
    if len(candidates) < 2:
        return list(candidates), ResidualReport(0.0, False, False, before, before)
    device = next(ranker.parameters()).device
    features = champion_features(policy, obs_dict).to(device)
    actions = torch.as_tensor(
        [[candidate.actions for candidate in candidates]],
        dtype=torch.long,
        device=device,
    )
    probabilities = torch.as_tensor(
        [[max(1e-12, float(candidate.prob)) for candidate in candidates]],
        dtype=torch.float32,
        device=device,
    )
    champion_log_prob = probabilities.log()
    with torch.no_grad():
        adjusted, confidence, _residual = ranker(
            features,
            actions,
            champion_log_prob,
            (
                candidate_semantic_features(policy, obs_dict, actions)
                if ranker.config.candidate_feature_dim
                else None
            ),
            torch.ones_like(champion_log_prob, dtype=torch.bool),
        )
    confidence_value = float(confidence[0])
    if confidence_value < ranker.config.confidence_threshold:
        return list(candidates), ResidualReport(
            confidence_value, False, False, before, before
        )

    scores = adjusted[0].clone()
    # Strategic-only candidates were injected as guard fallbacks outside the policy's
    # normal beam. A generic learned head must not promote them.
    for index, candidate in enumerate(candidates):
        if getattr(candidate, "strategic_only", False):
            scores[index] = champion_log_prob[0, index]
    probabilities = scores.softmax(dim=0).cpu().tolist()
    rescored = []
    for candidate, probability in zip(candidates, probabilities):
        candidate.prob = float(probability)
        rescored.append(candidate)
    rescored.sort(key=lambda candidate: candidate.prob, reverse=True)
    after = tuple(candidate.actions for candidate in rescored)
    return rescored, ResidualReport(
        confidence_value, True, after != before, before, after
    )
