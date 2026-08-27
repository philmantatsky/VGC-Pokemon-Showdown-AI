"""Refuse expensive training unless ladder-representative evaluations agree.

This is deliberately conservative. Previous self-play checkpoints improved against a
static local heuristic while remaining flat or worse on ladder, so a single noisy win
rate is not permission to spend another night training.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

POLICY = "policy preview + policy"
RANDOM_PREVIEW = "random preview + policy"
HEURISTIC = "random preview + heuristic"


def load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("n_battles", 0) < 100:
        raise SystemExit(f"FAIL: {path} has fewer than 100 battles per arm")
    return payload


def rate(payload: dict, arm: str) -> float:
    return float(payload["arms"][arm]["win_rate"])


def lower_bound(payload: dict, arm: str) -> float:
    """Lower endpoint of the 95% Wilson interval."""
    result = payload["arms"][arm]
    wins = int(result["wins"])
    total = int(result["battles"])
    z = 1.96
    p = wins / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return center - radius


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--open", type=Path, required=True)
    parser.add_argument("--hidden", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--write_pass", type=Path, default=None)
    args = parser.parse_args()

    open_eval = load(args.open)
    hidden_eval = load(args.hidden)
    population_eval = load(args.population)
    if open_eval.get("hidden_sheets") or not hidden_eval.get("hidden_sheets"):
        raise SystemExit("FAIL: --open and --hidden files have the wrong sheet modes")
    for label, payload in (
        ("open", open_eval),
        ("hidden", hidden_eval),
        ("population", population_eval),
    ):
        if payload.get("guard_profile") != "hard":
            raise SystemExit(f"FAIL: {label} evaluation did not use hard guards")

    checks = {
        "open policy 95% lower bound >= 50%": (lower_bound(open_eval, POLICY) >= 0.50),
        "hidden policy 95% lower bound >= 50%": (
            lower_bound(hidden_eval, POLICY) >= 0.50
        ),
        "open policy beats heuristic by >= 5pp": (
            rate(open_eval, POLICY) - rate(open_eval, HEURISTIC) >= 0.05
        ),
        "hidden policy beats heuristic by >= 5pp": (
            rate(hidden_eval, POLICY) - rate(hidden_eval, HEURISTIC) >= 0.05
        ),
        "preview is not >5pp worse than random (open)": (
            rate(open_eval, POLICY) >= rate(open_eval, RANDOM_PREVIEW) - 0.05
        ),
        "preview is not >5pp worse than random (hidden)": (
            rate(hidden_eval, POLICY) >= rate(hidden_eval, RANDOM_PREVIEW) - 0.05
        ),
        # This is the correlation check that the static heuristic lacked. The
        # historical learned policy produced ~40% in the smoke test, matching the
        # real ladder result, while the simple heuristic claimed 74%.
        "learned-population 95% lower bound >= 50%": (
            lower_bound(population_eval, POLICY) >= 0.50
        ),
        "preview is not >5pp worse than random (population)": (
            rate(population_eval, POLICY)
            >= rate(population_eval, RANDOM_PREVIEW) - 0.05
        ),
    }
    for label, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}  {label}")

    if not all(checks.values()):
        raise SystemExit("TRAINING BLOCKED: representative evaluation gate failed")

    print("TRAINING GATE PASSED")
    if args.write_pass:
        args.write_pass.parent.mkdir(parents=True, exist_ok=True)
        args.write_pass.write_text("passed\n")


if __name__ == "__main__":
    main()
