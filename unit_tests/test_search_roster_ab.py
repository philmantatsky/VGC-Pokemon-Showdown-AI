"""evaluation/search_roster_ab.py: the deployed bot with and without the exact search
against a held-out opponent on held-out rosters -- which cells count, the roster-paired
difference, and that its default selection is the battery's own."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evaluation.search_roster_ab import (
    complete_cells,
    paired_delta,
    pool,
    record,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]


def _cell(roster, hidden, arm, results):
    return [
        {"opponent": roster, "hidden_sheets": hidden, "arm": arm, "target": result}
        for result in results
    ]


def test_record_counts_ties_as_halves():
    row = record(_cell("MC1.txt", True, "plain", [1.0, 0.0, 0.5, 1.0]))
    assert (row["games"], row["wins"], row["win_rate"]) == (4, 2.5, 0.625)
    assert row["wilson_95"][0] < 0.625 < row["wilson_95"][1]
    assert record([])["win_rate"] is None


def test_only_whole_cells_are_resumed():
    rows = _cell("MC1.txt", True, "plain", [1, 0]) + _cell(
        "MC1.txt", True, "search", [1, 1]
    )
    assert complete_cells(rows, 2) == {
        ("MC1.txt", True, "plain"),
        ("MC1.txt", True, "search"),
    }
    assert complete_cells([], 2) == set()
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        complete_cells(rows + _cell("MC2.txt", False, "plain", [1]), 2)
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        complete_cells(rows + _cell("MC1.txt", True, "plain", [1, 1]), 2)


def test_the_difference_is_paired_by_roster_and_resamples_rosters():
    rows = []
    for index in range(10):
        roster = f"MC{index}.txt"
        for hidden in (True, False):
            rows += _cell(roster, hidden, "plain", [1, 0, 0, 0])  # 25%
            rows += _cell(roster, hidden, "search", [1, 1, 0, 0])  # 50%
    out = paired_delta(rows)
    for mode in ("overall", "hidden", "open"):
        assert out[mode]["rosters"] == 10
        assert out[mode]["delta"] == pytest.approx(0.25)
        # every roster shows the same difference: the interval is a point
        assert out[mode]["bootstrap_95"] == pytest.approx([0.25, 0.25])
    assert out["overall"]["games_per_arm"] == 80
    assert out["hidden"]["games_per_arm"] == 40
    assert out["overall"]["plain_win_rate"] == pytest.approx(0.25)
    assert out["overall"]["search_win_rate"] == pytest.approx(0.5)


def test_a_roster_played_in_one_arm_only_is_left_out_of_the_difference():
    rows = (
        _cell("MC1.txt", True, "plain", [0, 0])
        + _cell("MC1.txt", True, "search", [1, 1])
        + _cell("MC2.txt", True, "plain", [1, 1])  # its search cell is still to come
    )
    out = paired_delta(rows)
    assert out["hidden"]["rosters"] == 1 and out["hidden"]["delta"] == 1.0
    assert out["open"] == {"rosters": 0, "delta": None, "bootstrap_95": None}


def test_the_interval_widens_when_rosters_disagree():
    rows = []
    for index in range(12):
        roster = f"MC{index}.txt"
        better = index % 2 == 0
        rows += _cell(roster, True, "plain", [1, 0])
        rows += _cell(roster, True, "search", [1, 1] if better else [0, 0])
    out = paired_delta(rows)["hidden"]
    assert out["delta"] == pytest.approx(0.0)
    low, high = out["bootstrap_95"]
    assert low < -0.1 and high > 0.1


def test_summary_carries_both_arms_and_the_searchs_own_counts():
    rows = _cell("MC1.txt", True, "plain", [1, 0]) + _cell(
        "MC1.txt", True, "search", [1, 1]
    )
    telemetry = [
        {"arm": "plain", "search_counts": {}},
        {
            "arm": "search",
            "search_counts": {"exact_search": 20, "exact_search_error": 1},
        },
        {"arm": "search", "search_counts": {"exact_search": 22}},
    ]
    out = summarize(rows, telemetry)
    assert out["arms"]["plain"]["overall"]["win_rate"] == 0.5
    assert out["arms"]["search"]["hidden"]["win_rate"] == 1.0
    assert out["arms"]["search"]["open"]["games"] == 0
    assert out["search_minus_plain"]["overall"]["delta"] == 0.5
    assert out["search_counts"] == {"exact_search": 42, "exact_search_error": 1}


def test_the_default_selection_is_the_held_out_batterys(tmp_path):
    """47 held-out rosters x 2 sheets x 11 games: what evaluation/run_guard_ab.py
    plays, so the search is read on the same rosters as every guard."""
    from evaluation.run_t6_confirmation import EXCLUDE
    from tools.deployed_config import resolve

    needed = [ROOT / resolve()["CKPT"], ROOT / EXCLUDE]
    if not all(path.is_file() for path in needed):
        pytest.skip("the deployed checkpoint or the battery manifest is not here")
    out = tmp_path / "run"
    done = subprocess.run(
        [
            sys.executable,
            "evaluation/search_roster_ab.py",
            "--opponent",
            "heuristic",
            "--a-search-streams",
            "4",
            "--prepare-only",
            "--output",
            str(out),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(done.stdout.splitlines()[-1]) == {
        "rosters": 47,
        "cells": 188,
        "done": 0,
    }
    manifest = json.loads((out / "manifest.json").read_text())
    assert (manifest["repeats"], manifest["seed"], manifest["device"]) == (
        11,
        20923,
        "cpu",
    )
    assert manifest["search"]["anchor"] == 0.07 and manifest["search"]["streams"] == 4
    assert manifest["search"]["replies"] == 8 and manifest["search"]["sample"] is False
    assert len(manifest["our_guards"]) >= 14
    # another study in the same directory is refused, not mixed in
    refused = subprocess.run(
        [
            sys.executable,
            "evaluation/search_roster_ab.py",
            "--opponent",
            "heuristic",
            "--prepare-only",
            "--output",
            str(out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0 and "another study" in refused.stderr


def _run(root: Path, name: str, manifest: dict, rows: list[dict]) -> Path:
    run = root / name
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (run / "telemetry.jsonl").write_text(
        json.dumps({"arm": "search", "search_counts": {"exact_search": 5}}) + "\n"
    )
    return run


def test_pooling_keeps_one_roster_apart_for_each_opponent(tmp_path):
    study = {"repeats": 2, "seed": 1, "search": {"anchor": 0.07}}
    won = _cell("MC1.txt", True, "plain", [0, 0]) + _cell(
        "MC1.txt", True, "search", [1, 1]
    )
    lost = _cell("MC1.txt", True, "plain", [1, 1]) + _cell(
        "MC1.txt", True, "search", [0, 0]
    )
    first = _run(tmp_path, "frozen", {**study, "opponent": "a.zip"}, won)
    second = _run(tmp_path, "heuristic", {**study, "opponent": "heuristic"}, lost)
    out = pool([first, second])
    # the same roster file against two opponents is two rosters, not one
    assert out["search_minus_plain"]["overall"]["rosters"] == 2
    assert out["search_minus_plain"]["overall"]["delta"] == 0.0
    assert out["by_opponent"]["frozen"]["delta"] == 1.0
    assert out["by_opponent"]["heuristic"]["delta"] == -1.0
    assert out["arms"]["search"]["overall"]["games"] == 4
    assert out["search_counts"] == {"exact_search": 10}
    assert out["opponents"] == ["a.zip", "heuristic"]
    other = _run(tmp_path, "other", {**study, "repeats": 3, "opponent": "b.zip"}, won)
    with pytest.raises(ValueError, match="not the same study"):
        pool([first, other])
    twin = _run(tmp_path, "twin", {**study, "opponent": "a.zip"}, won)
    with pytest.raises(ValueError, match="same opponent twice"):
        pool([first, twin])


# --- the forecast arm (2026-10-09): the same search, replies by the predictor ---


def test_without_a_forecast_arm_a_summary_is_what_it_was():
    rows = _cell("MC1.txt", True, "plain", [1, 0]) + _cell(
        "MC1.txt", True, "search", [1, 1]
    )
    out = summarize(rows, [])
    assert set(out) == {"arms", "search_minus_plain", "search_counts"}
    assert set(out["arms"]) == {"plain", "search"}
    assert set(out["search_minus_plain"]["overall"]) >= {
        "plain_win_rate",
        "search_win_rate",
        "delta",
        "bootstrap_95",
    }


def test_the_forecast_arm_is_compared_with_the_search_roster_by_roster():
    rows = (
        _cell("MC1.txt", True, "plain", [1, 0])
        + _cell("MC1.txt", True, "search", [1, 0])
        + _cell("MC1.txt", True, "forecast", [1, 1])
        + _cell("MC2.txt", False, "plain", [0, 0])
        + _cell("MC2.txt", False, "search", [1, 1])
        + _cell("MC2.txt", False, "forecast", [1, 0])
        # a roster the forecast arm has not played yet is left out of its differences
        + _cell("MC3.txt", True, "plain", [1, 1])
        + _cell("MC3.txt", True, "search", [1, 1])
    )
    telemetry = [
        {"arm": "search", "search_counts": {"exact_search": 9}},
        {"arm": "forecast", "search_counts": {"exact_search": 8, "reply_forecast": 7}},
    ]
    out = summarize(rows, telemetry)
    against_search = out["forecast_minus_search"]["overall"]
    assert against_search["rosters"] == 2 and against_search["delta"] == 0.0
    assert (against_search["search_win_rate"], against_search["forecast_win_rate"]) == (
        0.75,
        0.75,
    )
    assert out["forecast_minus_search"]["hidden"]["delta"] == 0.5
    assert out["forecast_minus_search"]["open"]["delta"] == -0.5
    assert out["forecast_minus_plain"]["overall"]["delta"] == 0.5
    assert out["arms"]["forecast"]["overall"]["games"] == 4
    assert out["search_counts"] == {"exact_search": 9}
    assert out["forecast_counts"] == {"exact_search": 8, "reply_forecast": 7}
    # the search's own difference still counts all three of its rosters
    assert out["search_minus_plain"]["overall"]["rosters"] == 3
    assert (
        paired_delta(rows, arm="forecast", base="search")
        == (out["forecast_minus_search"])
    )


def test_pooling_reports_the_forecast_arm_by_opponent(tmp_path):
    study = {"repeats": 2, "seed": 1, "search": {"anchor": 0.07}}
    better = (
        _cell("MC1.txt", True, "plain", [1, 1])
        + _cell("MC1.txt", True, "search", [0, 0])
        + _cell("MC1.txt", True, "forecast", [1, 1])
    )
    worse = (
        _cell("MC1.txt", True, "plain", [1, 1])
        + _cell("MC1.txt", True, "search", [1, 1])
        + _cell("MC1.txt", True, "forecast", [1, 0])
    )
    first = _run(tmp_path, "human_new", {**study, "opponent": "a.zip"}, better)
    second = _run(tmp_path, "frozen", {**study, "opponent": "b.zip"}, worse)
    out = pool([first, second])
    assert out["forecast_minus_search_by_opponent"]["human_new"]["delta"] == 1.0
    assert out["forecast_minus_search_by_opponent"]["frozen"]["delta"] == -0.5
    assert out["forecast_minus_search"]["overall"]["delta"] == 0.25
