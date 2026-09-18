"""tools/compare_team_tournaments.py: summary loading, the unpaired delta and
its standard error, and the side-by-side table with a missing team."""

from __future__ import annotations

import json
from pathlib import Path

from tools.compare_team_tournaments import delta_pp, load_summary, table


def _summary(path: Path, rows: dict[str, tuple[int, int]]) -> Path:
    path.mkdir(parents=True)
    data = {
        "pilot": "x.zip",
        "teams": [
            {"team": t, "wins": w, "battles": n, "win_rate": w / n}
            for t, (w, n) in rows.items()
        ],
    }
    (path / "summary.json").write_text(json.dumps(data))
    return path


def test_load_summary_from_dir_or_file(tmp_path: Path) -> None:
    d = _summary(tmp_path / "a", {"T0": (247, 300), "T1": (107, 300)})
    assert load_summary(d) == {"T0": (247, 300), "T1": (107, 300)}
    assert load_summary(d / "summary.json") == load_summary(d)


def test_delta_pp_and_standard_error() -> None:
    d, se = delta_pp((150, 300), (150, 300))
    assert d == 0.0
    assert abs(se - 100 * (2 * 0.25 / 300) ** 0.5) < 1e-9
    d, _ = delta_pp((240, 300), (150, 300))
    assert abs(d - 30.0) < 1e-9


def test_table_marks_missing_team_and_means(tmp_path: Path) -> None:
    ref = load_summary(_summary(tmp_path / "ref", {"T0": (240, 300), "T1": (120, 300)}))
    cand = load_summary(_summary(tmp_path / "cand", {"T0": (210, 300)}))
    lines = table([("deployed", ref), ("v1", cand)])
    assert lines[0].split()[1:] == ["deployed", "v1"]
    assert "T0" in lines[1] and "-10.0" in lines[1]
    assert lines[2].startswith("T1") and lines[2].rstrip().endswith("-")
    assert lines[3].startswith("mean") and "60.0%" in lines[3] and "70.0%" in lines[3]
