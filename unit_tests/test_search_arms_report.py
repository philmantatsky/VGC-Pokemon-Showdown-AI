"""evaluation/search_arms_report.py: which shards count, the pooled records, the
difference between two groups of arms, and the search's own numbers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.search_arms_report import (
    complete_shards,
    difference,
    parse_factor,
    record,
    render,
    report,
    verdict,
)


def _decision(seconds, roots, samples, played, champion, outcome=None, hidden=False):
    schedule = {"mode": "search", "planning_roots": roots, "chance_samples": samples}
    if outcome:
        schedule["live_guards"] = {"champion": outcome}
    return {
        "battle": "b",
        "exact_search": {
            "open_sheet": not hidden,
            "schedule": schedule,
            "result": {"elapsed_s": seconds, "rankings": []},
            "actions": played,
            "champion_actions": champion,
        },
    }


def _shard(root: Path, name: str, wins, complete=True, decisions=()):
    """A finished (or not) run directory: ``wins`` = (open wins, open games, hidden
    wins, hidden games)."""
    run = root / f"results_mirror_{name}"
    run.mkdir()
    open_wins, open_games, hidden_wins, hidden_games = wins
    blocks = [
        {
            "hidden_sheets": False,
            "a_challenges": True,
            "a_wins": open_wins,
            "games": open_games,
        },
        {
            "hidden_sheets": True,
            "a_challenges": True,
            "a_wins": hidden_wins,
            "games": hidden_games,
        },
    ]
    (run / "result.json").write_text(
        json.dumps({"blocks": blocks, "complete": complete})
    )
    (run / "a_decisions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in decisions)
    )
    return run


def test_only_an_arms_own_finished_shards_count(tmp_path):
    _shard(tmp_path, "nash5_r2_s1", (30, 50, 25, 50))
    _shard(tmp_path, "nash5_r1_s2", (20, 50, 30, 50))
    _shard(tmp_path, "nash5_r10_s1", (1, 2, 1, 2))
    _shard(tmp_path, "nash5_r3_s1", (9, 50, 9, 50), complete=False)  # still playing
    _shard(tmp_path, "nash5_r1_s1_unfinished_1791", (9, 50, 9, 50))  # set aside
    _shard(tmp_path, "nash5cal_r1_s1", (9, 50, 9, 50))  # another arm, same prefix
    (tmp_path / "results_mirror_nash5_r1_s2.log").write_text("a shard's log\n")
    (tmp_path / "results_mirror_nash5_rounds.log").write_text("the arm's log\n")
    names = [run.name for run in complete_shards(tmp_path, "nash5")]
    assert names == [
        "results_mirror_nash5_r1_s2",
        "results_mirror_nash5_r2_s1",
        "results_mirror_nash5_r10_s1",  # rounds in numeric, not text, order
    ]
    assert [run.name for run in complete_shards(tmp_path, "nash5cal")] == [
        "results_mirror_nash5cal_r1_s1"
    ]
    assert complete_shards(tmp_path, "missing") == []


def test_the_verdict_is_the_pre_registered_reading():
    assert verdict(record(560, 1000)) == "wins"  # [52.9, 59.0]
    assert verdict(record(440, 1000)) == "loses"
    assert verdict(record(520, 1000)) == "no detectable difference"  # [48.9, 55.1]
    assert verdict(record(0, 0)) == "no games"


def test_difference_of_two_rates_has_newcombes_interval():
    first, second = record(560, 1000), record(500, 1000)
    row = difference(first, second)
    assert row["difference"] == pytest.approx(0.06)
    low, high = row["interval_95"]
    # Newcombe (1998) method 10 from the two Wilson intervals
    assert low == pytest.approx(0.0164, abs=5e-4)
    assert high == pytest.approx(0.1032, abs=5e-4)
    same = difference(first, first)
    assert (
        same["difference"] == 0 and same["interval_95"][0] < 0 < same["interval_95"][1]
    )
    assert difference(record(0, 0), second) == {"difference": None, "interval_95": None}


def test_factor_text_names_two_groups_of_arms():
    assert parse_factor("streams 4 vs 1=s4, s4cal:raw,cal") == (
        "streams 4 vs 1",
        ["s4", "s4cal"],
        ["raw", "cal"],
    )
    for bad in ("no groups", "label=a", "label=:b", "=a:b"):
        with pytest.raises(ValueError, match="--factor"):
            parse_factor(bad)


def test_report_pools_arms_factors_and_search_numbers(tmp_path):
    kept, changed = [1, 2], [3, 4]
    _shard(
        tmp_path,
        "raw_r1_s1",
        (30, 50, 20, 50),
        decisions=[
            _decision(0.5, 4, 1, kept, kept, "kept"),
            _decision(7.2, 1, 1, changed, kept, "overridden"),  # one world, at budget
            _decision(1.0, 1, 1, kept, kept, "override_vetoed", hidden=True),
            {"battle": "b", "exact_search": {"schedule": {"mode": "error_fallback"}}},
        ],
    )
    _shard(tmp_path, "raw_r2_s1", (25, 50, 25, 50))
    _shard(
        tmp_path,
        "s4_r1_s1",
        (35, 50, 30, 50),
        decisions=[
            _decision(2.0, 1, 4, kept, kept, "kept"),  # one world, four streams
            _decision(3.0, 3, 2, kept, kept, "kept"),
        ],
    )
    summary = report(tmp_path, {"raw": "raw", "s4": "s4"}, ["streams 4 vs 1=s4:raw"])
    raw, s4 = summary["arms"]["raw"], summary["arms"]["s4"]
    assert (raw["shards"], raw["a_wins"], raw["games"]) == (2, 100, 200)
    assert raw["verdict"] == "no detectable difference"
    assert raw["open_sheets"]["a_wins"] == 55 and raw["open_sheets"]["games"] == 100
    assert raw["hidden_sheets"]["a_win_rate"] == pytest.approx(0.45)
    assert raw["by_round"] == {"1": [50, 100], "2": [50, 100]}
    search = raw["search"]
    assert search["decisions"] == 4 and search["searched"] == 3
    assert search["not_searched"] == {"error_fallback": 1}
    assert search["at_budget_share"] == pytest.approx(1 / 3)
    assert search["one_stream_share"] == pytest.approx(2 / 3)
    assert search["override_share"] == pytest.approx(1 / 3)
    assert search["sent_back_by_guards_share"] == pytest.approx(1 / 3)
    assert search["seconds_median"] == 1.0
    assert s4["search"]["one_stream_share"] == 0  # 1 x 4 and 3 x 2 streams
    assert s4["a_win_rate"] == pytest.approx(0.65) and s4["verdict"] == "wins"
    factor = summary["factors"]["streams 4 vs 1"]
    assert factor["first"]["arms"] == ["s4"] and factor["second"]["games"] == 200
    assert factor["difference"] == pytest.approx(0.15)
    assert factor["interval_95"][0] > 0
    assert summary["all_arms"]["games"] == 300
    text = render(summary)
    assert "raw" in text and "+15.0 points" in text and "one stream" in text


def test_report_refuses_a_factor_over_an_unknown_arm(tmp_path):
    _shard(tmp_path, "raw_r1_s1", (30, 50, 20, 50))
    with pytest.raises(ValueError, match="unknown arms"):
        report(tmp_path, {"raw": "raw"}, ["x=raw:missing"])


def test_an_arm_without_finished_shards_reads_as_no_games(tmp_path):
    summary = report(tmp_path, {"raw": "raw"}, [])
    assert summary["arms"]["raw"]["games"] == 0
    assert summary["arms"]["raw"]["verdict"] == "no games"
    assert "no games" in render(summary)
