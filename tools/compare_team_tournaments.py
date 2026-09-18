"""Side-by-side team tournaments: per-team win rates of several pilots.

Each tournament directory holds the summary.json written by
evaluation/run_team_tournament.sh (one pilot piloting every candidate team
against the weighted Reg M-C pool). The first argument is the reference pilot;
every other pilot's per-team delta is printed against it, with the paired-free
standard error of the difference of two proportions. Usage:
  .venv/bin/python tools/compare_team_tournaments.py \\
      deployed=results_team_tournament/deployed \\
      v1_s7=results_team_tournament/brainv1_19660800
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_summary(path: Path) -> dict[str, tuple[int, int]]:
    """team -> (wins, battles) from a tournament summary.json (or its dir)."""
    if path.is_dir():
        path = path / "summary.json"
    data = json.loads(path.read_text())
    return {
        row["team"]: (int(row["wins"]), int(row["battles"])) for row in data["teams"]
    }


def delta_pp(a: tuple[int, int], b: tuple[int, int]) -> tuple[float, float]:
    """(delta in pp, standard error in pp) of rate(a) - rate(b), unpaired."""
    ra, rb = a[0] / a[1], b[0] / b[1]
    se = math.sqrt(ra * (1 - ra) / a[1] + rb * (1 - rb) / b[1])
    return 100 * (ra - rb), 100 * se


def table(pilots: list[tuple[str, dict[str, tuple[int, int]]]]) -> list[str]:
    ref_name, ref = pilots[0]
    teams = sorted(ref)
    lines = ["team  " + "  ".join(f"{name:>16}" for name, _ in pilots)]
    for team in teams:
        cells = []
        for i, (_, rows) in enumerate(pilots):
            if team not in rows:
                cells.append(f"{'-':>16}")
                continue
            wins, n = rows[team]
            cell = f"{100 * wins / n:5.1f}% ({wins}/{n})"
            if i > 0:
                d, se = delta_pp(rows[team], ref[team])
                cell = f"{100 * wins / n:5.1f}% {d:+5.1f}±{se:3.1f}"
            cells.append(f"{cell:>16}")
        lines.append(f"{team:<4}  " + "  ".join(cells))
    means = []
    for _, rows in pilots:
        rates = [w / n for w, n in rows.values()]
        means.append(sum(rates) / len(rates) if rates else float("nan"))
    lines.append(
        "mean  "
        + "  ".join(
            f"{100 * m:5.1f}%{'' if i == 0 else f' {100 * (m - means[0]):+5.1f}':>10}"
            for i, m in enumerate(means)
        )
    )
    lines.append(f"(deltas vs {ref_name}; ± = one standard error, unpaired)")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "pilots", nargs="+", help="label=tournament_dir (first = reference)"
    )
    args = ap.parse_args()
    pilots = []
    for spec in args.pilots:
        label, _, path = spec.partition("=")
        pilots.append((label, load_summary(Path(path or label))))
    print("\n".join(table(pilots)))


if __name__ == "__main__":
    main()
