"""tools/cross_team_read.py: the specialist's rate comes from the candidate arm
of its own battery, the deployed rate from the baseline arm of the reference
battery; deltas are unpaired with a two-proportion standard error."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.cross_team_read import cross_read, weighted_delta


def _battery(
    path: Path, arm: str, candidate: tuple[int, int], baseline: tuple[int, int]
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    arms = {
        "distilled_policy": {"wins": candidate[0], "battles": candidate[1]},
        "champion_policy": {"wins": baseline[0], "battles": baseline[1]},
    }
    (path / f"battery_{arm}.json").write_text(json.dumps({"arms": arms}))


def test_cross_read_takes_each_side_from_its_own_battery(tmp_path: Path) -> None:
    spec, ref = tmp_path / "spec", tmp_path / "ref"
    _battery(spec, "human_bc", (820, 1000), (760, 1000))  # baseline here = save 7
    _battery(ref, "human_bc", (785, 1000), (799, 1000))  # baseline here = deployed
    _battery(spec, "heuristic", (700, 1000), (650, 1000))
    _battery(ref, "heuristic", (798, 1000), (864, 1000))
    _battery(spec, "frozen", (900, 1000), (880, 1000))  # no reference file: skipped
    rows = {r["arm"]: r for r in cross_read(spec, ref)}
    assert set(rows) == {"heuristic", "human_bc"}
    assert rows["human_bc"]["specialist"] == pytest.approx(0.82)
    assert rows["human_bc"]["deployed"] == pytest.approx(0.799)
    assert rows["human_bc"]["delta_pp"] == pytest.approx(2.1)
    expected_se = 100 * (0.82 * 0.18 / 1000 + 0.799 * 0.201 / 1000) ** 0.5
    assert rows["human_bc"]["se_pp"] == pytest.approx(expected_se)
    assert rows["heuristic"]["delta_pp"] == pytest.approx(-16.4)


def test_weighted_delta_counts_the_human_arm_twice() -> None:
    rows = [
        {"arm": "heuristic", "delta_pp": -6.0},
        {"arm": "human_bc", "delta_pp": 3.0},
    ]
    assert weighted_delta(rows) == pytest.approx(0.0)
    assert weighted_delta(rows, 1.0) == pytest.approx(-1.5)
