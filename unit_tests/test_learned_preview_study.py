"""evaluation/learned_preview_study.py: only the study's own player gets the
learned preview, and its opponent-plan state is dropped after preview."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evaluation import learned_preview_study, opening_study
from vgc_bench.src.policy_player import PolicyPlayer


def test_install_patches_only_the_study_player(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "preview.pt"
    model.write_bytes(b"x")
    monkeypatch.setattr(
        opening_study.StudyPlayer, "__init__", lambda self, *a, **k: None
    )
    calls = []

    def fake_learned(self, battle):
        calls.append(battle.battle_tag)
        self._battle_plans[battle.battle_tag] = "belief"
        return "/team 1234"

    monkeypatch.setattr(PolicyPlayer, "_learned_teampreview", fake_learned)
    monkeypatch.setattr(
        opening_study.StudyPlayer, "_learned_preview_model", None, raising=False
    )
    learned_preview_study.install(model)

    ours = opening_study.StudyPlayer()
    assert ours.preview_model_path == model and ours.use_learned_teampreview
    ours._battle_plans = {}
    learned: Any = PolicyPlayer._learned_teampreview
    battle = SimpleNamespace(battle_tag="b1")
    assert learned(ours, battle) == "/team 1234"
    assert ours._battle_plans == {}  # opponent belief discarded for our player

    other = SimpleNamespace(_battle_plans={})  # e.g. the opponent's player
    assert learned(other, SimpleNamespace(battle_tag="b2"))
    assert other._battle_plans == {"b2": "belief"}  # untouched
    assert calls == ["b1", "b2"]

    with pytest.raises(RuntimeError, match="already installed"):
        learned_preview_study.install(model)


def test_install_refuses_a_missing_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        learned_preview_study.install(tmp_path / "missing.pt")
