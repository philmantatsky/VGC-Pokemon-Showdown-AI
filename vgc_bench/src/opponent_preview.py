"""Learned bring/lead prediction and coherent team-preview plans.

The turn policy currently drafts in two independent calls: first two leads, then two
backline Pokemon.  This module instead scores all valid (bring four, lead two) plans
jointly.  The same model is run in both directions:

* our roster conditioned on theirs -> a human-like plan for us;
* their roster conditioned on ours -> a belief over their likely plans.

The model is deliberately small.  It learns species and matchup interactions from
top-player replays while keeping the runtime cheap enough to use at every preview.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from poke_env.battle import Pokemon
from poke_env.data import to_id_str
from torch import nn

PAIR_INDICES: tuple[tuple[int, int], ...] = tuple(combinations(range(6), 2))
BRING_INDICES: tuple[tuple[int, int, int, int], ...] = tuple(combinations(range(6), 4))

_POKE_RE = re.compile(r"^\|poke\|(p[12])\|([^|]+)", re.MULTILINE)
_SWITCH_RE = re.compile(
    r"^\|(?:switch|drag|replace)\|(p[12])[ab]: [^|]+\|([^|]+)", re.MULTILINE
)
_PLAYER_RE = re.compile(r"^\|player\|(p[12])\|[^|]*\|[^|]*\|(\d+)$", re.MULTILINE)

# Floors printed by the top-500 scrape that produced battle_logs_top. They are kept
# with the dataset logic so "top-player training" cannot silently include both the
# ranked player and a 1000-Elo opponent from the same replay again.
TOP_500_ELO_FLOORS = {"regmb": 1655, "regmbbo3": 1432}


def species_id(details: str) -> str:
    """Canonical base-species id from a Showdown details field."""
    try:
        return to_id_str(Pokemon(9, details=details).base_species)
    except Exception:
        return to_id_str(details.split(",", 1)[0])


@dataclass(frozen=True)
class PreviewExample:
    battle_id: str
    roster: tuple[str, ...]
    opponent_roster: tuple[str, ...]
    lead: tuple[int, int]
    bring: tuple[int, int, int, int] | None
    rating: int | None = None


def parse_preview_examples(battle_id: str, log: str) -> list[PreviewExample]:
    """Extract one example per side from a complete Showdown replay.

    Leads are always observable.  The complete bring-four label is retained only when
    all four selected Pokemon appeared during the battle; unrevealed backline slots
    are unknown, not negative labels.
    """
    rosters: dict[str, list[str]] = {"p1": [], "p2": []}
    ratings = {side: int(value) for side, value in _PLAYER_RE.findall(log)}
    for side, details in _POKE_RE.findall(log):
        rosters[side].append(species_id(details))
    if any(len(roster) != 6 or len(set(roster)) != 6 for roster in rosters.values()):
        return []

    start = log.find("|start")
    turn_one = log.find("|turn|1", start)
    if start < 0 or turn_one < 0:
        return []
    opening = log[start:turn_one]
    leads: dict[str, list[str]] = {"p1": [], "p2": []}
    for side, details in _SWITCH_RE.findall(opening):
        mon = species_id(details)
        if mon not in leads[side]:
            leads[side].append(mon)

    revealed: dict[str, list[str]] = {"p1": [], "p2": []}
    for side, details in _SWITCH_RE.findall(log[start:]):
        mon = species_id(details)
        if mon in rosters[side] and mon not in revealed[side]:
            revealed[side].append(mon)

    out = []
    for side, opponent in (("p1", "p2"), ("p2", "p1")):
        if len(leads[side]) != 2:
            continue
        try:
            lead = tuple(sorted(rosters[side].index(mon) for mon in leads[side]))
        except ValueError:
            continue
        bring = None
        if len(revealed[side]) == 4:
            try:
                bring = tuple(
                    sorted(rosters[side].index(mon) for mon in revealed[side])
                )
            except ValueError:
                bring = None
        out.append(
            PreviewExample(
                battle_id=battle_id,
                roster=tuple(rosters[side]),
                opponent_roster=tuple(rosters[opponent]),
                lead=lead,  # type: ignore[arg-type]
                bring=bring,  # type: ignore[arg-type]
                rating=ratings.get(side),
            )
        )
    return out


def top_500_rating_floor(battle_id: str) -> int:
    """Elo floor from the source ladder snapshot for this replay's format."""
    return (
        TOP_500_ELO_FLOORS["regmbbo3"]
        if "regmbbo3" in battle_id
        else TOP_500_ELO_FLOORS["regmb"]
    )


def load_replay_examples(
    paths: Iterable[Path], top_500_only: bool = False
) -> list[PreviewExample]:
    examples = []
    for path in paths:
        payload = json.loads(path.read_text())
        for battle_id, value in payload.items():
            log = value[1] if isinstance(value, (list, tuple)) else value
            parsed = parse_preview_examples(str(battle_id), str(log))
            if top_500_only:
                floor = top_500_rating_floor(str(battle_id))
                parsed = [
                    example
                    for example in parsed
                    if example.rating is not None and example.rating >= floor
                ]
            examples.extend(parsed)
    return examples


class PreviewNet(nn.Module):
    """DeepSets-style matchup encoder with lead-pair and bring-four heads."""

    def __init__(self, vocab_size: int, embed_dim: int = 48, hidden_dim: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.species_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        pair_in = 4 * embed_dim
        bring_in = 3 * embed_dim
        self.lead_head = nn.Sequential(
            nn.Linear(pair_in, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.bring_head = nn.Sequential(
            nn.Linear(bring_in, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(
        self, roster: torch.Tensor, opponent_roster: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ours = self.species_embed(roster)
        theirs = self.species_embed(opponent_roster)
        our_context = ours.mean(dim=1)
        their_context = theirs.mean(dim=1)

        lead_features = []
        for i, j in PAIR_INDICES:
            pair_sum = ours[:, i] + ours[:, j]
            pair_diff = torch.abs(ours[:, i] - ours[:, j])
            lead_features.append(
                torch.cat([pair_sum, pair_diff, our_context, their_context], dim=-1)
            )
        lead_tensor = torch.stack(lead_features, dim=1)
        lead_logits = self.lead_head(lead_tensor).squeeze(-1)

        bring_features = []
        for indices in BRING_INDICES:
            selected = ours[:, list(indices)].mean(dim=1)
            bring_features.append(
                torch.cat([selected, our_context, their_context], dim=-1)
            )
        bring_tensor = torch.stack(bring_features, dim=1)
        bring_logits = self.bring_head(bring_tensor).squeeze(-1)
        return lead_logits, bring_logits


@dataclass(frozen=True)
class PreviewPlan:
    lead_indices: tuple[int, int]
    bring_indices: tuple[int, int, int, int]
    probability: float

    @property
    def back_indices(self) -> tuple[int, int]:
        lead = set(self.lead_indices)
        return tuple(i for i in self.bring_indices if i not in lead)  # type: ignore[return-value]


@dataclass
class OpponentBelief:
    roster: tuple[str, ...]
    plans: list[PreviewPlan]

    def observe(
        self, revealed_species: Sequence[str], opening_leads: Sequence[str] = ()
    ) -> None:
        """Bayesian-style hard conditioning on observed bring/lead information."""
        revealed = {to_id_str(name) for name in revealed_species}
        leads = {to_id_str(name) for name in opening_leads}
        kept = []
        for plan in self.plans:
            brought = {self.roster[i] for i in plan.bring_indices}
            planned_leads = {self.roster[i] for i in plan.lead_indices}
            if not revealed.issubset(brought):
                continue
            if leads and leads != planned_leads:
                continue
            kept.append(plan)
        if not kept:
            return
        total = sum(plan.probability for plan in kept)
        self.plans = [
            PreviewPlan(plan.lead_indices, plan.bring_indices, plan.probability / total)
            for plan in kept
        ]

    def bring_marginals(self) -> dict[str, float]:
        """Probability that each roster member is among the opponent's four."""
        return {
            species: sum(
                plan.probability for plan in self.plans if index in plan.bring_indices
            )
            for index, species in enumerate(self.roster)
        }

    def lead_marginals(self) -> dict[str, float]:
        """Probability that each roster member is among the opponent's leads."""
        return {
            species: sum(
                plan.probability for plan in self.plans if index in plan.lead_indices
            )
            for index, species in enumerate(self.roster)
        }


@dataclass
class BattlePlanState:
    """Preview plan plus opponent-plan belief updated from actual switch-ins."""

    own_plan: PreviewPlan | None
    opponent_belief: OpponentBelief
    opening_leads: tuple[str, ...] = ()
    seen_in_battle: set[str] = field(default_factory=set)

    def observe_active(self, active_species: Sequence[str]) -> None:
        active = tuple(dict.fromkeys(to_id_str(name) for name in active_species))
        if len(active) == 2 and not self.opening_leads:
            self.opening_leads = active
        self.seen_in_battle.update(active)
        self.opponent_belief.observe(tuple(self.seen_in_battle), self.opening_leads)

    def likely_bench(self, active_species: Sequence[str]) -> list[tuple[str, float]]:
        """Rank possible switch-ins from the remaining bring-four distribution."""
        unavailable = self.seen_in_battle | {to_id_str(name) for name in active_species}
        marginals = self.opponent_belief.bring_marginals()
        return sorted(
            (
                (species, probability)
                for species, probability in marginals.items()
                if species not in unavailable
            ),
            key=lambda item: item[1],
            reverse=True,
        )


class PreviewPredictor:
    def __init__(self, model: PreviewNet, vocab: dict[str, int]):
        self.model = model.eval()
        self.vocab = vocab

    @classmethod
    def load(cls, path: Path, device: str | torch.device = "cpu"):
        payload = torch.load(path, map_location=device, weights_only=False)
        config = payload["config"]
        model = PreviewNet(
            vocab_size=len(payload["vocab"]),
            embed_dim=int(config["embed_dim"]),
            hidden_dim=int(config["hidden_dim"]),
        )
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        return cls(model, dict(payload["vocab"]))

    def encode(self, roster: Sequence[str]) -> torch.Tensor:
        if len(roster) != 6:
            raise ValueError(f"expected six Pokemon, got {len(roster)}")
        return torch.tensor(
            [self.vocab.get(to_id_str(name), 0) for name in roster],
            dtype=torch.long,
            device=next(self.model.parameters()).device,
        )

    @torch.no_grad()
    def predict_plans(
        self, roster: Sequence[str], opponent_roster: Sequence[str], top_k: int = 30
    ) -> list[PreviewPlan]:
        own = self.encode(roster).unsqueeze(0)
        opp = self.encode(opponent_roster).unsqueeze(0)
        lead_logits, bring_logits = self.model(own, opp)
        lead_logp = lead_logits.log_softmax(dim=-1)[0]
        bring_logp = bring_logits.log_softmax(dim=-1)[0]
        candidates = []
        for li, lead in enumerate(PAIR_INDICES):
            lead_set = set(lead)
            for bi, bring in enumerate(BRING_INDICES):
                if not lead_set.issubset(bring):
                    continue
                score = float(lead_logp[li] + bring_logp[bi])
                candidates.append((score, lead, bring))
        candidates.sort(reverse=True, key=lambda item: item[0])
        candidates = candidates[: max(1, top_k)]
        peak = candidates[0][0]
        weights = [math.exp(score - peak) for score, _, _ in candidates]
        total = sum(weights)
        return [
            PreviewPlan(lead, bring, weight / total)
            for weight, (_score, lead, bring) in zip(weights, candidates)
        ]


def plan_to_showdown_order(plan: PreviewPlan) -> tuple[int, int, int, int]:
    """One-indexed Showdown order: two leads followed by two backline slots."""
    return tuple(i + 1 for i in (*plan.lead_indices, *plan.back_indices))  # type: ignore[return-value]


def model_metadata(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return dict(payload.get("metrics", {}))
