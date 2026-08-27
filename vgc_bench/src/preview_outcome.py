"""Terminal-outcome Team Preview model specialized to the fixed six-Pokemon team."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Sequence

import torch
from poke_env.data import to_id_str
from torch import nn

from vgc_bench.src.opponent_preview import PreviewPlan


class PreviewOutcomeNet(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 48, hidden_dim: int = 160):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.species = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.head = nn.Sequential(
            nn.Linear(4 * embed_dim + 12, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        our_roster: torch.Tensor,
        opponent_roster: torch.Tensor,
        lead_mask: torch.Tensor,
        bring_mask: torch.Tensor,
    ) -> torch.Tensor:
        ours = self.species(our_roster)
        opponents = self.species(opponent_roster)
        lead = (ours * lead_mask.unsqueeze(-1)).sum(1) / 2.0
        bring = (ours * bring_mask.unsqueeze(-1)).sum(1) / 4.0
        opponent_mean = opponents.mean(1)
        opponent_max = opponents.max(1).values
        features = torch.cat(
            [lead, bring, opponent_mean, opponent_max, lead_mask, bring_mask], dim=-1
        )
        return self.head(features).squeeze(-1)


@dataclass(frozen=True)
class ScoredPreviewPlan:
    plan: PreviewPlan
    win_probability: float


class PreviewOutcomePredictor:
    def __init__(self, model: PreviewOutcomeNet, vocab: dict[str, int], our_roster):
        self.model = model.eval()
        self.vocab = dict(vocab)
        self.our_roster = tuple(to_id_str(species) for species in our_roster)

    @classmethod
    def load(cls, path: Path, device: str | torch.device = "cpu"):
        payload = torch.load(path, map_location=device, weights_only=False)
        config = payload["config"]
        model = PreviewOutcomeNet(
            len(payload["vocab"]),
            int(config["embed_dim"]),
            int(config["hidden_dim"]),
        )
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        return cls(model, payload["vocab"], payload["our_roster"])

    def _encode(self, roster: Sequence[str]) -> torch.Tensor:
        return torch.tensor(
            [self.vocab.get(to_id_str(species), 0) for species in roster],
            dtype=torch.long,
            device=next(self.model.parameters()).device,
        )

    @torch.no_grad()
    def rank(
        self,
        our_roster: Sequence[str],
        opponent_roster: Sequence[str],
        candidates: Sequence[PreviewPlan] | None = None,
    ) -> list[ScoredPreviewPlan]:
        ours = tuple(to_id_str(species) for species in our_roster)
        if ours != self.our_roster:
            raise ValueError("preview outcome model is specialized to another team")
        plans = list(candidates or ())
        if not plans:
            for bring in combinations(range(6), 4):
                for lead in combinations(bring, 2):
                    plans.append(PreviewPlan(lead, bring, 1.0))
        count = len(plans)
        our_tensor = self._encode(ours).repeat(count, 1)
        opponent_tensor = self._encode(opponent_roster).repeat(count, 1)
        lead_mask = torch.zeros((count, 6), device=our_tensor.device)
        bring_mask = torch.zeros((count, 6), device=our_tensor.device)
        for row, plan in enumerate(plans):
            lead_mask[row, list(plan.lead_indices)] = 1.0
            bring_mask[row, list(plan.bring_indices)] = 1.0
        probabilities = torch.sigmoid(
            self.model(our_tensor, opponent_tensor, lead_mask, bring_mask)
        ).cpu()
        ranked = [
            ScoredPreviewPlan(plan, float(probability))
            for plan, probability in zip(plans, probabilities)
        ]
        return sorted(ranked, key=lambda item: item.win_probability, reverse=True)
