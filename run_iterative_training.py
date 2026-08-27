"""Sequential conservative aggregation rounds for the fixed Reg M-B team.

Four simulator workers run inside each round, followed by one MPS trainer. Rounds are
strictly sequential so two trainers never contend for the same Apple GPU. Every prior
round remains in the training set, and exactly half of later trajectories use the
latest residual while half continue from the untouched champion.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--root", default="results_iterative_v1")
    parser.add_argument("--baseline", default="results_repaired/champion.zip")
    parser.add_argument(
        "--outcome-value", default="results_outcome_v1/outcome_value.zip"
    )
    parser.add_argument("--games-per-round", type=int, default=10000)
    parser.add_argument("--generation-seconds", type=float, default=5400.0)
    parser.add_argument("--evaluation-battles", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    if args.rounds < 1:
        raise SystemExit("--rounds must be positive")

    root = (ROOT / args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    historical: list[Path] = []
    rollout_residual: Path | None = None
    for round_index in range(1, args.rounds + 1):
        name = f"round_{round_index:02d}"
        data = root / name / "data"
        output = root / name / "results"
        command = [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "run_counterfactual_pipeline.py"),
            "--baseline",
            str((ROOT / args.baseline).resolve()),
            "--outcome-value",
            str((ROOT / args.outcome_value).resolve()),
            "--data",
            str(data),
            "--output",
            str(output),
            "--games",
            str(args.games_per_round),
            "--generation-seconds",
            str(args.generation_seconds),
            "--evaluation-battles",
            str(args.evaluation_battles),
            "--seed",
            str(args.seed + round_index * 1000),
        ]
        for previous in historical:
            command.extend(["--historical-data", str(previous)])
        if rollout_residual is not None:
            command.extend(["--rollout-residual", str(rollout_residual)])
        print(f"starting {name}", flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            raise SystemExit(f"{name} failed with status {result.returncode}")
        historical.append(data)
        metrics_path = output / "candidate_residual.metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            top = metrics.get("top_three") or []
            if top:
                rollout_residual = Path(top[0]["path"]).resolve()
        deployment = output / "deployment.json"
        if deployment.exists() and json.loads(deployment.read_text()).get("passed"):
            print(f"promotion gate passed in {name}; stopping rounds", flush=True)
            break


if __name__ == "__main__":
    main()
