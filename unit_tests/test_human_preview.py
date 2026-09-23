"""training/human_preview.py: temperature weights, the sampled '/team' order and
its selection flags, the incomplete-roster fallback, and idempotent install."""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from training import human_preview
from vgc_bench.src.opponent_preview import PreviewPlan


def test_plan_weights_follow_the_temperature() -> None:
    assert human_preview.plan_weights([0.75, 0.25], 1.0) == pytest.approx([0.75, 0.25])
    sharp = human_preview.plan_weights([0.75, 0.25], 0.5)  # squares, renormalised
    assert sharp == pytest.approx([0.9, 0.1])
    assert human_preview.plan_weights([0.0, 0.0], 1.0) == [0.5, 0.5]
    with pytest.raises(ValueError):
        human_preview.plan_weights([1.0], 0.0)


class _Mon:
    def __init__(self, species: str) -> None:
        self.base_species = species
        self._selected_in_teampreview = False


def _battle(ours: list[str], theirs: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        team={f"p1: {s}": _Mon(s) for s in ours},
        opponent_team={f"p2: {s}": _Mon(s) for s in theirs},
    )


class _Predictor:
    def __init__(self, plans: list[PreviewPlan]) -> None:
        self.plans = plans
        self.calls: list[tuple[list[str], list[str]]] = []

    def predict_plans(self, ours, theirs, top_k=90):
        self.calls.append((list(ours), list(theirs)))
        return self.plans


SIX = ["Blastoise", "Farigiraf", "Charizard", "Venusaur", "Torkoal", "Incineroar"]
THEM = ["Tyranitar", "Excadrill", "Rillaboom", "Sneasler", "Salamence", "Milotic"]


def test_human_order_returns_leads_then_back_and_flags_the_four() -> None:
    plan = PreviewPlan(lead_indices=(0, 1), bring_indices=(0, 1, 4, 5), probability=1.0)
    predictor = _Predictor([plan])
    battle = _battle(SIX, THEM)
    order = human_preview.human_order(predictor, battle, 1.0, random.Random(0))
    assert order == "/team 1256"  # Blastoise, Farigiraf lead; Torkoal, Incineroar back
    assert (
        predictor.calls[0][0][0] == "blastoise"
        and predictor.calls[0][1][0] == "tyranitar"
    )
    flagged = [
        m.base_species for m in battle.team.values() if m._selected_in_teampreview
    ]
    assert flagged == ["Blastoise", "Farigiraf", "Torkoal", "Incineroar"]


def test_human_order_samples_by_probability_and_refuses_incomplete_rosters() -> None:
    a = PreviewPlan(lead_indices=(0, 1), bring_indices=(0, 1, 2, 3), probability=0.9)
    b = PreviewPlan(lead_indices=(2, 3), bring_indices=(2, 3, 4, 5), probability=0.1)
    predictor = _Predictor([a, b])
    rng = random.Random(1)
    picks = []
    for _ in range(400):
        order = human_preview.human_order(predictor, _battle(SIX, THEM), 1.0, rng)
        assert order is not None
        picks.append(order[6:8])
    assert 0.8 < picks.count("12") / 400 < 0.97
    assert (
        human_preview.human_order(predictor, _battle(SIX[:5], THEM), 1.0, rng) is None
    )


def test_install_is_idempotent_and_refuses_missing_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from poke_env.player.player import Player

    monkeypatch.setattr(Player, "random_teampreview", Player.random_teampreview)
    monkeypatch.setattr(
        "vgc_bench.src.opponent_preview.PreviewPredictor.load",
        classmethod(lambda cls, path, device="cpu": _Predictor([])),
    )
    with pytest.raises(FileNotFoundError):
        human_preview.install(tmp_path / "missing.pt")
    model = tmp_path / "preview.pt"
    model.write_bytes(b"x")
    assert human_preview.install(model) is True
    assert human_preview.install(model) is False  # already installed
    fallback: Any = Player.random_teampreview  # empty plans -> random order
    battle = _battle(SIX, THEM)
    battle.format = "gen9championsvgc2026regmc"
    order = fallback(SimpleNamespace(), battle)
    assert order.startswith("/team ") and len(order) == len("/team ") + 4
