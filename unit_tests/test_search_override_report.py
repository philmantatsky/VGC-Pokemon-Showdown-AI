"""evaluation/search_override_report.py: overrides of the bot's own pair, their payoff
edges and kinds, and side A's results by how many a game held."""

from __future__ import annotations

import json

import pytest

from evaluation.search_override_report import change_kind, report


def test_change_kinds():
    assert change_kind("move icebeam +1", "move waterpulse +1") == "move"
    assert change_kind("move icebeam +1", "move icebeam +2") == "target"
    assert change_kind("move icebeam +1", "move protect") == "protect"
    assert change_kind("move protect", "move eruption") == "protect"
    assert change_kind("move icebeam +1", "switch 3") == "switch"
    assert change_kind("switch 3", "switch 4") == "switch"


def _decision(battle, turn, played, champion, outcome, mode="search", open_sheet=True):
    rows = [
        {"choice": "move icebeam +1, move eruption", "actions": [25, 9]},
        {"choice": "move waterpulse +1, move protect", "actions": [20, 24]},
    ]
    rows[0].update(prior=0.6, expected=0.10)
    rows[1].update(prior=0.2, expected=0.45)
    return {
        "battle": battle,
        "turn": turn,
        "exact_search": {
            "open_sheet": open_sheet,
            "champion_actions": champion,
            "actions": played,
            "schedule": {"mode": mode, "live_guards": {"champion": outcome}},
            "result": {"rankings": rows},
        },
    }


def test_report_counts_overrides_and_lines_them_up_with_results(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    battles = [
        {"battle": "b1", "a_won": True, "turns": 8},
        {"battle": "b2", "a_won": False, "turns": 9},
        {"battle": "b3", "a_won": None, "turns": 7},
    ]
    (run / "result.json").write_text(json.dumps({"blocks": [{"battles": battles}]}))
    rows = [
        _decision("b1", 1, [25, 9], [25, 9], "kept"),
        _decision("b1", 2, [20, 24], [25, 9], "overridden"),
        _decision("b1", 3, [20, 24], [25, 9], "overridden"),
        _decision("b2", 1, [25, 9], [25, 9], "override_vetoed", open_sheet=False),
        _decision(
            "b2", 2, None, [25, 9], None, mode="search_fallback", open_sheet=False
        ),
        _decision("b3", 1, [25, 9], [25, 9], "kept"),
        {"battle": "b3", "turn": 2},  # a champion-path row in the same log
    ]
    (run / "a_decisions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    summary = report([run])
    assert summary["decisions"] == 6
    assert summary["outcomes"] == {
        "kept": 2,
        "overridden": 2,
        "override_vetoed": 1,
        "fallback": 1,
    }
    assert summary["outcomes_by_sheets"]["hidden"] == {
        "override_vetoed": 1,
        "fallback": 1,
    }
    assert summary["overrides"] == 2 and summary["overrides_per_game"] == 2 / 3
    edge = summary["override_payoff_edge"]
    assert edge["median"] == pytest.approx(0.35) and edge["non_positive"] == 0
    assert summary["override_prior_ratio_median"] == pytest.approx(3.0)
    assert summary["changed_slots_by_kind"] == {"move": 2, "protect": 2}
    assert summary["top_move_swaps"] == {"icebeam -> waterpulse": 2}
    by = summary["side_a_by_overrides_in_the_game"]
    assert by["2"]["games"] == 1 and by["2"]["a_wins"] == 1.0
    assert by["0"]["games"] == 2 and by["0"]["a_wins"] == 0.5  # a loss and a tie
    assert by["1"]["games"] == 0 and by["3+"]["games"] == 0
