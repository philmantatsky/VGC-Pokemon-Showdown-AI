"""Run the standard multi-opponent gate battery for a candidate checkpoint.

One command, four opponent configurations, one scorecard. This is the powered
replacement for single-opponent 500-battle gates: the 2026-08 replan found every
learned artifact improved against the opponent it was fit on and regressed
against every other population, so no candidate may be judged on fewer arms.

Arms (each runs eval_counterfactual.py baseline-vs-candidate):

    heuristic     poke-env SimpleHeuristicsPlayer
    frozen        results_repaired/opponents/64opp_3932160_v4.zip (deterministic)
    rotation      the other two frozen PPOs (8opp / tuned), deterministic
    human_bc      results_bc/eval_B/... behavior-cloned human imitation,
                  STOCHASTIC, quarantined eval-only (never trained against)

Tiers: --tier screening (1,000 battles/arm, MDE ~4-6pp, kills clearly-bad
candidates) or --tier promotion (5,000/arm on the arms that matter). Statistical
notes: at p~0.8 a 500-battle arm resolves only ~6pp; do not draw conclusions
below the tier the effect requires.

    .venv/bin/python run_gate_battery.py --candidate <ckpt.zip> --tier screening
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

FROZEN = "results_repaired/opponents/64opp_3932160_v4.zip"
ROTATION = (
    "results_repaired/opponents/8opp_4915200_v4.zip",
    "results_repaired/opponents/tuned_983040_v4.zip",
)
HUMAN_BC_DEFAULT = "results_bc/eval_B/saves_bc/seed2/30.zip"

TIER_BATTLES = {"screening": 1000, "promotion": 5000}


def run_arm(
    name: str, args: argparse.Namespace, out_dir: Path, extra: list[str], n_battles: int
) -> dict | None:
    output = out_dir / f"battery_{name}.json"
    cmd = [
        sys.executable,
        str(ROOT / "eval_counterfactual.py"),
        "--baseline",
        args.baseline,
        "--candidate",
        args.candidate,
        "--n-battles",
        str(n_battles),
        "--port",
        str(args.port),
        "--seed",
        str(args.seed),
        "--workers",
        str(args.workers),
        "--output",
        str(output),
        *(["--hidden-sheets"] if args.hidden_sheets else []),
        *extra,
    ]
    print(f"\n=== arm: {name} ({n_battles} battles) ===", flush=True)
    print(" ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"arm {name} FAILED (exit {result.returncode})", flush=True)
        return None
    if not output.exists():
        return None
    return json.loads(output.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", default="results_repaired/champion.zip")
    ap.add_argument("--tier", choices=tuple(TIER_BATTLES), default="screening")
    ap.add_argument("--human-bc", default=HUMAN_BC_DEFAULT)
    ap.add_argument("--port", type=int, default=7600)
    ap.add_argument("--seed", type=int, default=83)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hidden-sheets", action="store_true", default=True)
    ap.add_argument(
        "--open-sheets",
        dest="hidden_sheets",
        action="store_false",
        help="run the battery with open team sheets instead of hidden",
    )
    ap.add_argument("--out-dir", default="results_gate_battery")
    ap.add_argument(
        "--arms",
        default="heuristic,frozen,rotation,human_bc",
        help="comma-separated subset of arms to run",
    )
    args = ap.parse_args()

    n_battles = TIER_BATTLES[args.tier]
    out_dir = ROOT / args.out_dir / args.tier
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(args.arms.split(","))

    arm_specs: list[tuple[str, list[str]]] = []
    if "heuristic" in wanted:
        arm_specs.append(("heuristic", []))
    if "frozen" in wanted:
        arm_specs.append(("frozen", ["--opponent-checkpoint", FROZEN]))
    if "rotation" in wanted:
        for index, ckpt in enumerate(ROTATION, start=1):
            arm_specs.append((f"rotation{index}", ["--opponent-checkpoint", ckpt]))
    if "human_bc" in wanted:
        human_bc = Path(args.human_bc)
        if not human_bc.exists():
            print(f"human_bc arm skipped: {human_bc} missing", flush=True)
        else:
            arm_specs.append(
                (
                    "human_bc",
                    ["--opponent-checkpoint", str(human_bc), "--opponent-stochastic"],
                )
            )

    scorecard: dict[str, dict] = {}
    for name, extra in arm_specs:
        payload = run_arm(name, args, out_dir, extra, n_battles)
        if payload is None:
            scorecard[name] = {"status": "failed"}
            continue
        arms = payload.get("arms", {})
        summary = {}
        for arm_name, arm in arms.items():
            loss_shape = arm.get("loss_shape") or {}
            summary[arm_name] = {
                "wins": arm.get("wins"),
                "battles": arm.get("battles"),
                "win_rate": arm.get("win_rate"),
                "wilson_95": arm.get("wilson_95"),
                "first_faint_ours": loss_shape.get("first_faint_ours"),
                "tr_wins": loss_shape.get("tr_wins"),
                "tr_battles": loss_shape.get("tr_battles"),
            }
        scorecard[name] = summary

    scorecard_path = out_dir / "scorecard.json"
    scorecard_path.write_text(
        json.dumps(
            {
                "candidate": args.candidate,
                "baseline": args.baseline,
                "tier": args.tier,
                "n_battles_per_arm": n_battles,
                "hidden_sheets": args.hidden_sheets,
                "arms": scorecard,
            },
            indent=1,
        )
    )
    print(f"\nscorecard -> {scorecard_path}")
    for name, summary in scorecard.items():
        if "status" in summary:
            print(f"{name:12} FAILED")
            continue
        parts = [
            f"{arm_name}: {arm['wins']}/{arm['battles']}"
            for arm_name, arm in summary.items()
        ]
        print(f"{name:12} {'  '.join(parts)}")


if __name__ == "__main__":
    main()
