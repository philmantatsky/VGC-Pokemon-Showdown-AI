"""training/t6tac_practice_gate.py (2026-10-03): the same team on both sides, so the
candidate must WIN its head-to-head (Wilson lower bound above 50%), not just tie."""

from __future__ import annotations

import json
from pathlib import Path

from training.t6tac_practice_gate import head_to_head_wins


def _result(tmp_path: Path, low: float, high: float, complete: bool = True) -> Path:
    run = tmp_path / "mirror"
    run.mkdir()
    (run / "result.json").write_text(
        json.dumps(
            {
                "complete": complete,
                "wilson_95": [low, high],
                "a_win_rate": (low + high) / 2,
                "games": 2000,
            }
        )
    )
    return run


def test_a_won_head_to_head_passes(tmp_path: Path) -> None:
    assert head_to_head_wins(_result(tmp_path, 0.512, 0.556)) == []


def test_a_tie_holds(tmp_path: Path) -> None:
    assert head_to_head_wins(_result(tmp_path, 0.478, 0.522))


def test_an_unfinished_head_to_head_holds(tmp_path: Path) -> None:
    assert head_to_head_wins(_result(tmp_path, 0.6, 0.7, complete=False))
