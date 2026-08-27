"""Learned voluntary-switch prediction from high-rated VGC replay decisions."""

from __future__ import annotations

import json
import re
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from poke_env.data import to_id_str
from torch import nn

from vgc_bench.src.move_semantics import MOVE_SEM_LEN, move_semantics
from vgc_bench.src.opponent_preview import species_id, top_500_rating_floor

_POKE_RE = re.compile(r"^\|poke\|(p[12])\|([^|]+)", re.MULTILINE)
_PLAYER_RE = re.compile(r"^\|player\|(p[12])\|[^|]*\|[^|]*\|(\d+)$", re.MULTILINE)
_SLOT_RE = re.compile(r"^(p[12])([ab]): ")


def _slot(identifier: str) -> tuple[str, int] | None:
    match = _SLOT_RE.match(identifier)
    if match is None:
        return None
    return match.group(1), 0 if match.group(2) == "a" else 1


def _hp_fraction(condition: str) -> float:
    token = condition.split(" ", 1)[0]
    if token in {"0", "0/0"}:
        return 0.0
    if "/" not in token:
        return 1.0
    current, maximum = token.split("/", 1)
    try:
        return max(0.0, min(1.0, float(current) / float(maximum)))
    except (ValueError, ZeroDivisionError):
        return 1.0


@dataclass(frozen=True)
class SwitchExample:
    battle_id: str
    roster: tuple[str, ...]
    opponent_roster: tuple[str, ...]
    active: tuple[str, str]
    opponent_active: tuple[str, str]
    hp: tuple[float, float, float, float]
    actor_slot: int
    turn: int
    switch_to: str | None
    rating: int | None


@dataclass(frozen=True)
class MoveExample:
    battle_id: str
    roster: tuple[str, ...]
    opponent_roster: tuple[str, ...]
    active: tuple[str, str]
    opponent_active: tuple[str, str]
    hp: tuple[float, float, float, float]
    actor_slot: int
    turn: int
    move_id: str
    target_class: int
    rating: int | None


def _target_class(actor_side: str, actor_pos: int, target: str) -> int:
    """self, ally, foe-a, foe-b, or field/unknown -> 0..4."""
    target_slot = _slot(target)
    if target_slot is None:
        return 4
    target_side, target_pos = target_slot
    if target_side == actor_side:
        return 0 if target_pos == actor_pos else 1
    return 2 + target_pos


def _parse_tactical_examples(
    battle_id: str, log: str
) -> tuple[list[SwitchExample], list[MoveExample]]:
    """Extract pre-action states with voluntary-switch and observed-move labels."""
    rosters: dict[str, list[str]] = {"p1": [], "p2": []}
    for side, details in _POKE_RE.findall(log):
        rosters[side].append(species_id(details))
    if any(len(team) != 6 or len(set(team)) != 6 for team in rosters.values()):
        return [], []
    ratings = {side: int(value) for side, value in _PLAYER_RE.findall(log)}

    active: dict[str, list[str | None]] = {"p1": [None, None], "p2": [None, None]}
    hp: dict[str, list[float]] = {"p1": [1.0, 1.0], "p2": [1.0, 1.0]}
    lines = log.splitlines()
    turn_positions = [i for i, line in enumerate(lines) if line.startswith("|turn|")]
    if not turn_positions:
        return [], []

    def update_state(line: str) -> None:
        parts = line.split("|")
        if len(parts) < 3:
            return
        event = parts[1]
        slot = _slot(parts[2])
        if slot is None:
            return
        side, pos = slot
        if event in {"switch", "drag", "replace"} and len(parts) >= 5:
            active[side][pos] = species_id(parts[3])
            hp[side][pos] = _hp_fraction(parts[4])
        elif event == "faint":
            hp[side][pos] = 0.0
        elif event in {"-damage", "-heal", "-sethp"} and len(parts) >= 4:
            hp[side][pos] = _hp_fraction(parts[3])

    for line in lines[: turn_positions[0]]:
        update_state(line)

    examples: list[SwitchExample] = []
    move_examples: list[MoveExample] = []
    for turn_i, start in enumerate(turn_positions):
        end = (
            turn_positions[turn_i + 1]
            if turn_i + 1 < len(turn_positions)
            else len(lines)
        )
        try:
            turn = int(lines[start].split("|")[2])
        except (IndexError, ValueError):
            continue
        start_active = {side: list(values) for side, values in active.items()}
        start_hp = {side: list(values) for side, values in hp.items()}
        moved: set[tuple[str, int]] = set()
        fainted: set[tuple[str, int]] = set()
        acted_sides: set[str] = set()
        chosen_switch: dict[tuple[str, int], str] = {}
        chosen_moves: list[tuple[str, int, str, int]] = []

        for line in lines[start + 1 : end]:
            parts = line.split("|")
            if len(parts) < 3:
                continue
            event = parts[1]
            slot = _slot(parts[2])
            if slot is not None:
                side, pos = slot
                if event == "move":
                    moved.add(slot)
                    acted_sides.add(side)
                    if len(parts) >= 4 and "[from]" not in line:
                        target = parts[4] if len(parts) >= 5 else ""
                        chosen_moves.append(
                            (
                                side,
                                pos,
                                to_id_str(parts[3]),
                                _target_class(side, pos, target),
                            )
                        )
                elif event == "faint":
                    fainted.add(slot)
                elif event == "switch" and len(parts) >= 5:
                    # Chosen switches happen before that slot moves and while its
                    # turn-start Pokemon is alive. A replacement after a KO and an
                    # U-turn-style pivot fail one of those checks.
                    if (
                        start_active[side][pos] is not None
                        and start_hp[side][pos] > 0
                        and slot not in moved
                        and slot not in fainted
                        and active[side][pos] == start_active[side][pos]
                    ):
                        chosen_switch[slot] = species_id(parts[3])
                        acted_sides.add(side)
            update_state(line)

        for side, opponent in (("p1", "p2"), ("p2", "p1")):
            if side not in acted_sides:
                continue
            if any(mon is None for mon in start_active[side] + start_active[opponent]):
                continue
            active_pair = tuple(start_active[side])
            opponent_pair = tuple(start_active[opponent])
            assert len(active_pair) == 2 and len(opponent_pair) == 2
            common = {
                "battle_id": battle_id,
                "roster": tuple(rosters[side]),
                "opponent_roster": tuple(rosters[opponent]),
                "active": active_pair,
                "opponent_active": opponent_pair,
                "hp": tuple(start_hp[side] + start_hp[opponent]),
                "turn": turn,
                "rating": ratings.get(side),
            }
            for move_side, pos, move_id, target_class in chosen_moves:
                if move_side != side:
                    continue
                move_examples.append(
                    MoveExample(
                        **common,  # type: ignore[arg-type]
                        actor_slot=pos,
                        move_id=move_id,
                        target_class=target_class,
                    )
                )
            for pos in range(2):
                if start_hp[side][pos] <= 0:
                    continue
                switch_to = chosen_switch.get((side, pos))
                # A handful of Illusion/replace protocol sequences appear to switch
                # into a species already active at turn start. That is not a legal
                # voluntary-switch label, and feeding it to the masked target head
                # creates a 1e9 loss. Unknown is safer than inventing a stay label.
                if switch_to in active_pair:
                    continue
                examples.append(
                    SwitchExample(
                        **common,  # type: ignore[arg-type]
                        actor_slot=pos,
                        switch_to=switch_to,
                    )
                )
    return examples, move_examples


def parse_switch_examples(battle_id: str, log: str) -> list[SwitchExample]:
    """Extract decisions made at turn start; forced and pivot switches are excluded."""
    return _parse_tactical_examples(battle_id, log)[0]


def parse_move_examples(battle_id: str, log: str) -> list[MoveExample]:
    """Extract each observed move with its state before the turn began."""
    return _parse_tactical_examples(battle_id, log)[1]


def load_switch_examples(
    paths: Iterable[Path], top_500_only: bool = True
) -> list[SwitchExample]:
    examples = []
    for path in paths:
        payload = json.loads(path.read_text())
        for battle_id, value in payload.items():
            log = value[1] if isinstance(value, (list, tuple)) else value
            parsed = parse_switch_examples(str(battle_id), str(log))
            if top_500_only:
                floor = top_500_rating_floor(str(battle_id))
                parsed = [
                    example
                    for example in parsed
                    if example.rating is not None and example.rating >= floor
                ]
            examples.extend(parsed)
    return examples


def load_move_examples(
    paths: Iterable[Path], top_500_only: bool = True
) -> list[MoveExample]:
    examples = []
    for path in paths:
        payload = json.loads(path.read_text())
        for battle_id, value in payload.items():
            log = value[1] if isinstance(value, (list, tuple)) else value
            parsed = parse_move_examples(str(battle_id), str(log))
            if top_500_only:
                floor = top_500_rating_floor(str(battle_id))
                parsed = [
                    example
                    for example in parsed
                    if example.rating is not None and example.rating >= floor
                ]
            examples.extend(parsed)
    return examples


class SwitchNet(nn.Module):
    """Two-stage switch/no-switch and switch-target model."""

    def __init__(self, vocab_size: int, embed_dim: int = 48, hidden_dim: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.species_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.slot_embed = nn.Embedding(2, 8)
        state_dim = 8 * embed_dim + 13
        self.state = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.switch_head = nn.Linear(hidden_dim, 1)
        self.target_head = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        roster: torch.Tensor,
        opponent_roster: torch.Tensor,
        active: torch.Tensor,
        opponent_active: torch.Tensor,
        hp: torch.Tensor,
        actor_slot: torch.Tensor,
        turn: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        roster_e = self.species_embed(roster)
        opponent_roster_e = self.species_embed(opponent_roster)
        active_e = self.species_embed(active)
        opponent_active_e = self.species_embed(opponent_active)
        batch = torch.arange(roster.shape[0], device=roster.device)
        actor = active_e[batch, actor_slot]
        ally = active_e[batch, 1 - actor_slot]
        scalars = torch.cat(
            [hp, (turn.float() / 20.0).unsqueeze(1), self.slot_embed(actor_slot)], dim=1
        )
        features = torch.cat(
            [
                actor,
                ally,
                opponent_active_e[:, 0],
                opponent_active_e[:, 1],
                roster_e.mean(1),
                opponent_roster_e.mean(1),
                torch.abs(opponent_active_e[:, 0] - opponent_active_e[:, 1]),
                roster_e.mean(1) - opponent_roster_e.mean(1),
                scalars,
            ],
            dim=1,
        )
        state = self.state(features)
        switch_logit = self.switch_head(state).squeeze(1)
        repeated = state.unsqueeze(1).expand(-1, 6, -1)
        target_logits = self.target_head(
            torch.cat([repeated, roster_e], dim=2)
        ).squeeze(2)
        # An active Pokemon cannot be its own switch target.
        active_mask = (roster[:, :, None] == active[:, None, :]).any(dim=2)
        target_logits = target_logits.masked_fill(active_mask, -1e9)
        return switch_logit, target_logits


@dataclass(frozen=True)
class SwitchPrediction:
    switch_probability: float
    targets: tuple[tuple[str, float], ...]


class SwitchPredictor:
    def __init__(
        self,
        model: SwitchNet,
        vocab: dict[str, int],
        calibration: dict[str, list[float]] | None = None,
    ):
        self.model = model.eval()
        self.vocab = vocab
        self.calibration = calibration

    @classmethod
    def load(cls, path: Path, device: str | torch.device = "cpu"):
        payload = torch.load(path, map_location=device, weights_only=False)
        config = payload["config"]
        model = SwitchNet(
            len(payload["vocab"]), int(config["embed_dim"]), int(config["hidden_dim"])
        )
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        return cls(model, dict(payload["vocab"]), payload.get("calibration"))

    def calibrate(self, raw_probability: float) -> float:
        if not self.calibration:
            return raw_probability
        bounds = self.calibration.get("upper_bounds", [])
        rates = self.calibration.get("rates", [])
        if not bounds or len(bounds) != len(rates):
            return raw_probability
        index = min(bisect_left(bounds, raw_probability), len(rates) - 1)
        return float(rates[index])

    def _encode(self, names: Sequence[str]) -> torch.Tensor:
        return torch.tensor(
            [self.vocab.get(to_id_str(name), 0) for name in names],
            dtype=torch.long,
            device=next(self.model.parameters()).device,
        )

    @torch.no_grad()
    def predict(
        self,
        roster: Sequence[str],
        opponent_roster: Sequence[str],
        active: Sequence[str],
        opponent_active: Sequence[str],
        hp: Sequence[float],
        actor_slot: int,
        turn: int,
        bring_marginals: dict[str, float] | None = None,
    ) -> SwitchPrediction:
        tensors = [
            self._encode(roster).unsqueeze(0),
            self._encode(opponent_roster).unsqueeze(0),
            self._encode(active).unsqueeze(0),
            self._encode(opponent_active).unsqueeze(0),
            torch.tensor(
                hp, dtype=torch.float32, device=next(self.model.parameters()).device
            ).unsqueeze(0),
            torch.tensor([actor_slot], device=next(self.model.parameters()).device),
            torch.tensor([turn], device=next(self.model.parameters()).device),
        ]
        switch_logit, target_logits = self.model(*tensors)
        probability = self.calibrate(float(torch.sigmoid(switch_logit[0])))
        target_probability = target_logits[0].softmax(0).cpu()
        weighted = []
        for index, species in enumerate(roster):
            weight = float(target_probability[index])
            if bring_marginals is not None:
                weight *= bring_marginals.get(to_id_str(species), 0.0)
            weighted.append(weight)
        total = sum(weighted)
        if total > 0:
            weighted = [value / total for value in weighted]
        targets = tuple(
            sorted(
                zip((to_id_str(name) for name in roster), weighted),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        return SwitchPrediction(probability, targets)


class MoveNet(nn.Module):
    """Candidate-scoring model for a high-rated player's move and target choice."""

    def __init__(
        self,
        species_vocab_size: int,
        move_ids: Sequence[str],
        embed_dim: int = 48,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.species_vocab_size = species_vocab_size
        self.move_ids = tuple(move_ids)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.species_embed = nn.Embedding(species_vocab_size, embed_dim, padding_idx=0)
        self.move_embed = nn.Embedding(len(move_ids), embed_dim, padding_idx=0)
        self.slot_embed = nn.Embedding(2, 8)
        state_dim = 8 * embed_dim + 13
        self.state = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.candidate_hidden = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim + MOVE_SEM_LEN, hidden_dim), nn.ReLU()
        )
        self.move_head = nn.Linear(hidden_dim, 1)
        self.target_head = nn.Linear(hidden_dim, 5)
        semantics = np.stack(
            [
                np.zeros(MOVE_SEM_LEN, dtype=np.float32)
                if index == 0
                else move_semantics(move_id)
                for index, move_id in enumerate(move_ids)
            ]
        )
        self.register_buffer("move_semantics", torch.from_numpy(semantics))

    def forward(
        self,
        roster: torch.Tensor,
        opponent_roster: torch.Tensor,
        active: torch.Tensor,
        opponent_active: torch.Tensor,
        hp: torch.Tensor,
        actor_slot: torch.Tensor,
        turn: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        roster_e = self.species_embed(roster)
        opponent_roster_e = self.species_embed(opponent_roster)
        active_e = self.species_embed(active)
        opponent_active_e = self.species_embed(opponent_active)
        batch = torch.arange(roster.shape[0], device=roster.device)
        actor = active_e[batch, actor_slot]
        ally = active_e[batch, 1 - actor_slot]
        scalars = torch.cat(
            [hp, (turn.float() / 20.0).unsqueeze(1), self.slot_embed(actor_slot)], dim=1
        )
        features = torch.cat(
            [
                actor,
                ally,
                opponent_active_e[:, 0],
                opponent_active_e[:, 1],
                roster_e.mean(1),
                opponent_roster_e.mean(1),
                torch.abs(opponent_active_e[:, 0] - opponent_active_e[:, 1]),
                roster_e.mean(1) - opponent_roster_e.mean(1),
                scalars,
            ],
            dim=1,
        )
        state = self.state(features)
        move_ids = torch.arange(len(self.move_ids), device=roster.device)
        move_e = self.move_embed(move_ids)
        move_semantics_buffer = self.get_buffer("move_semantics")
        batch_size = roster.shape[0]
        candidate_features = torch.cat(
            [
                state[:, None, :].expand(-1, len(self.move_ids), -1),
                move_e[None, :, :].expand(batch_size, -1, -1),
                move_semantics_buffer[None, :, :].expand(batch_size, -1, -1),
            ],
            dim=2,
        )
        candidate_hidden = self.candidate_hidden(candidate_features)
        move_logits = self.move_head(candidate_hidden).squeeze(2)
        move_logits[:, 0] = -1e9
        return move_logits, self.target_head(candidate_hidden)


@dataclass(frozen=True)
class MovePrediction:
    moves: tuple[tuple[str, float], ...]
    targets: tuple[tuple[str, float], ...]
    actions: tuple[tuple[str, str, float], ...] = ()
    reliability: float = 1.0


class MovePredictor:
    TARGET_NAMES = ("self", "ally", "foe_a", "foe_b", "field")

    def __init__(
        self, model: MoveNet, species_vocab: dict[str, int], move_vocab: dict[str, int]
    ):
        self.model = model.eval()
        self.species_vocab = species_vocab
        self.move_vocab = move_vocab

    @classmethod
    def load(cls, path: Path, device: str | torch.device = "cpu"):
        payload = torch.load(path, map_location=device, weights_only=False)
        config = payload["config"]
        move_vocab = dict(payload["move_vocab"])
        move_ids = [""] * len(move_vocab)
        for move_id, index in move_vocab.items():
            move_ids[index] = move_id
        model = MoveNet(
            len(payload["species_vocab"]),
            move_ids,
            int(config["embed_dim"]),
            int(config["hidden_dim"]),
        )
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        return cls(model, dict(payload["species_vocab"]), move_vocab)

    def _encode(self, names: Sequence[str]) -> torch.Tensor:
        return torch.tensor(
            [self.species_vocab.get(to_id_str(name), 0) for name in names],
            dtype=torch.long,
            device=next(self.model.parameters()).device,
        )

    @torch.no_grad()
    def predict(
        self,
        roster: Sequence[str],
        opponent_roster: Sequence[str],
        active: Sequence[str],
        opponent_active: Sequence[str],
        hp: Sequence[float],
        actor_slot: int,
        turn: int,
        available_moves: Sequence[str],
    ) -> MovePrediction:
        device = next(self.model.parameters()).device
        inputs = [
            self._encode(roster).unsqueeze(0),
            self._encode(opponent_roster).unsqueeze(0),
            self._encode(active).unsqueeze(0),
            self._encode(opponent_active).unsqueeze(0),
            torch.tensor(hp, dtype=torch.float32, device=device).unsqueeze(0),
            torch.tensor([actor_slot], device=device),
            torch.tensor([turn], device=device),
        ]
        move_logits, target_logits = self.model(*inputs)
        allowed = [
            (to_id_str(move_id), self.move_vocab.get(to_id_str(move_id), 0))
            for move_id in available_moves
        ]
        allowed = [(move_id, index) for move_id, index in allowed if index > 0]
        if not allowed:
            return MovePrediction((), (), ())
        scores = torch.stack([move_logits[0, index] for _, index in allowed])
        probabilities = scores.softmax(0).cpu().tolist()
        moves = tuple(
            sorted(
                (
                    (move_id, float(probability))
                    for (move_id, _), probability in zip(allowed, probabilities)
                ),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        top_move_index = allowed[probabilities.index(max(probabilities))][1]
        target_probabilities = (
            target_logits[0, top_move_index].softmax(0).cpu().tolist()
        )
        targets = tuple(
            sorted(
                zip(self.TARGET_NAMES, map(float, target_probabilities)),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        actions = []
        for (move_id, index), move_probability in zip(allowed, probabilities):
            conditional_targets = target_logits[0, index].softmax(0).cpu().tolist()
            actions.extend(
                (move_id, target_name, float(move_probability * target_probability))
                for target_name, target_probability in zip(
                    self.TARGET_NAMES, conditional_targets
                )
            )
        actions.sort(key=lambda item: item[2], reverse=True)
        return MovePrediction(moves, targets, tuple(actions))
