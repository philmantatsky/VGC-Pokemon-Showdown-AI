"""tools/pick_team_from_grid.py: the pre-registered team pick -- score = mean of
the human-clone read and the tournament read, top three confirmed, the highest
confirmation read wins unless the top two are within the tie window."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.pick_team_from_grid import (
    confirmation_reads,
    final_pick,
    grid_reads,
    scores,
    top,
    tournament_reads,
)


def _cell(path: Path, candidate: float, deployed: float) -> None:
    arms = {
        "distilled_policy": {"win_rate": candidate},
        "champion_policy": {"win_rate": deployed},
    }
    path.write_text(json.dumps({"arms": arms}))


def test_reads_from_the_grid_and_tournament_files(tmp_path: Path) -> None:
    _cell(tmp_path / "T0_human.json", 0.733, 0.790)
    _cell(tmp_path / "T2_human.json", 0.767, 0.547)
    _cell(tmp_path / "T2_heuristic.json", 0.657, 0.493)
    (tmp_path / "confirm_T2.json").write_text(
        json.dumps({"arms": {"champion_policy": {"win_rate": 0.75}}})
    )
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "teams": [
                    {"team": "T2", "win_rate": 0.84},
                    {"team": "T0", "win_rate": 0.83},
                ]
            }
        )
    )
    assert grid_reads(tmp_path, "human") == {"T0": 0.733, "T2": 0.767}
    assert grid_reads(tmp_path, "heuristic") == {"T2": 0.657}
    assert confirmation_reads(tmp_path) == {"T2": 0.75}
    assert tournament_reads(tmp_path) == {"T2": 0.84, "T0": 0.83}


def test_score_is_the_mean_and_top_three_is_ordered() -> None:
    human = {"T0": 0.733, "T1": 0.747, "T2": 0.767, "T3": 0.60}
    tournament = {"T0": 0.83, "T1": 0.72, "T2": 0.84, "T3": 0.60, "T9": 0.99}
    score = scores(human, tournament)
    assert set(score) == {"T0", "T1", "T2", "T3"}
    assert score["T2"] == pytest.approx(0.8035)
    assert top(score) == ["T2", "T0", "T1"]


def test_clear_confirmation_lead_wins_even_against_the_score() -> None:
    score = {"T0": 0.80, "T2": 0.78}
    team, reason = final_pick({"T0": 0.74, "T2": 0.77}, score)
    assert team == "T2" and "lead" in reason


def test_confirmation_tie_falls_back_to_the_score() -> None:
    score = {"T0": 0.782, "T2": 0.804, "T4": 0.79}
    team, reason = final_pick({"T0": 0.765, "T2": 0.755, "T4": 0.70}, score)
    assert team == "T2" and "tie" in reason
    # the tie window only looks at the top two confirmation reads
    team, _ = final_pick({"T0": 0.765, "T2": 0.70, "T4": 0.755}, score)
    assert team == "T4"


def test_final_pick_needs_a_confirmation_read() -> None:
    with pytest.raises(ValueError):
        final_pick({}, {"T0": 0.8})
    assert final_pick({"T1": 0.7}, {"T1": 0.7})[0] == "T1"
