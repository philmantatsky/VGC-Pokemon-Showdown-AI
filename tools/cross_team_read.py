"""Cross-team read: a specialist on ITS team vs the deployed brain on MB430.

A paired battery needs both pilots on the same team, which is meaningless when
the question is "new brain + new team against old brain + old team". The arms'
opponents are the same populations either way, so the honest comparison is the
unpaired difference of absolute win rates, arm by arm:

  specialist rate  = arm ``distilled_policy`` of the specialist's battery
  deployed rate    = arm ``champion_policy`` of a battery whose baseline was the
                     deployed brain on MB430 (any brain-v1 / round-6 battery)

Usage (from the repo root):
  .venv/bin/python tools/cross_team_read.py \\
      --specialist results_gate_battery_brainv1_spec/<stem>/screening \\
      --reference results_gate_battery_brainv1/19660800/screening
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ARMS = ("heuristic", "frozen", "rotation1", "rotation2", "human_bc")


def arm_record(battery_dir: Path, arm: str, policy: str) -> tuple[int, int] | None:
    path = battery_dir / f"battery_{arm}.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())["arms"].get(policy)
    if not record:
        return None
    return int(record["wins"]), int(record["battles"])


def cross_read(specialist: Path, reference: Path) -> list[dict]:
    rows = []
    for arm in ARMS:
        spec = arm_record(specialist, arm, "distilled_policy")
        ref = arm_record(reference, arm, "champion_policy")
        if spec is None or ref is None:
            continue
        rs, rr = spec[0] / spec[1], ref[0] / ref[1]
        se = math.sqrt(rs * (1 - rs) / spec[1] + rr * (1 - rr) / ref[1])
        rows.append(
            {
                "arm": arm,
                "specialist": rs,
                "deployed": rr,
                "delta_pp": 100 * (rs - rr),
                "se_pp": 100 * se,
            }
        )
    return rows


def weighted_delta(rows: list[dict], human_weight: float = 2.0) -> float:
    """Mean delta with the human arm counted ``human_weight`` times."""
    total = weight = 0.0
    for row in rows:
        w = human_weight if row["arm"] == "human_bc" else 1.0
        total += w * row["delta_pp"]
        weight += w
    return total / weight if weight else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--specialist", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    args = ap.parse_args()
    rows = cross_read(args.specialist, args.reference)
    for row in rows:
        print(
            f"CROSS {row['arm']:<10} specialist {100 * row['specialist']:5.1f}%  "
            f"deployed-on-MB430 {100 * row['deployed']:5.1f}%  "
            f"delta {row['delta_pp']:+5.1f} ± {row['se_pp']:.1f}pp (unpaired)"
        )
    print(
        f"CROSS weighted (human x2) {weighted_delta(rows):+.2f}pp   "
        f"equal {weighted_delta(rows, 1.0):+.2f}pp"
    )


if __name__ == "__main__":
    main()
