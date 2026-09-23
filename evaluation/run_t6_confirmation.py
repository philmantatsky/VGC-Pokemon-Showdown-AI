"""Bounded multi-population confirmation of T6 checkpoint 20152320.

Same pilot stack as the opening study; no search/reranker/preview-model additions.
Matched opponent rosters and previews, independent battle RNG. Holds deployment
fixed, performs no training, and stops for human review even if scores improve.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time

import numpy as np

from evaluation.opening_study import select_matchups
from evaluation.review_t6_audits import review

BASELINE = "results_brainv1/saves_fp_hs_wt/reg_mc/seed1/19660800.zip"
CANDIDATE = "results_brainv1_t6_repair1/saves_fp_hs_wt/reg_mc/seed1/20152320.zip"
EXCLUDE = "results_brainv1_t6_repair1/comparison_baseline.manifest.json"
OPPONENTS = {
    "human_new": ("results_bc/eval_mcB_20260920/saves_bc/seed2/2.zip", False),
    "frozen": ("results_repaired/opponents/64opp_3932160_v4.zip", True),
    "rotation1": ("results_repaired/opponents/8opp_4915200_v4.zip", True),
    "rotation2": ("results_repaired/opponents/tuned_983040_v4.zip", True),
    "human_previous": ("results_bc/eval_mcB_20260913/saves_bc/seed2/2.zip", False),
    "heuristic": ("heuristic", True),
}


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def paired_roster_summary(control, candidate):
    """Resample whole opponent rosters, not correlated games on the same roster."""

    def by_team(rows):
        result = {}
        for r in rows:
            result.setdefault(r["opponent"], []).append(r["target"])
        return result

    summaries = {}
    for mode in ("overall", "hidden", "open"):
        groups = []
        for rows in (control, candidate):
            groups.append(
                [
                    r
                    for r in rows
                    if mode == "overall" or r["hidden_sheets"] == (mode == "hidden")
                ]
            )
        a, b = map(by_team, groups)
        if set(a) != set(b) or any(len(a[k]) != len(b[k]) for k in a):
            raise ValueError("unmatched confirmation cells")
        keys = sorted(a)
        if not keys:
            raise ValueError("empty confirmation")
        delta = np.array([np.mean(b[k]) - np.mean(a[k]) for k in keys])
        boot = (
            np.random.default_rng(20923).choice(delta, (4000, len(delta))).mean(axis=1)
        )
        summaries[mode] = {
            "games_per_model": len(groups[0]),
            "opponent_rosters": len(keys),
            "baseline_win_rate": float(np.mean([r["target"] for r in groups[0]])),
            "candidate_win_rate": float(np.mean([r["target"] for r in groups[1]])),
            "roster_mean_delta": float(delta.mean()),
            "roster_bootstrap_95_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
        }
    return summaries


def run_job(command, log_path, timeout):
    with log_path.open("a") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
            if code:
                raise RuntimeError(f"{log_path.name}: child exited {code}")
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("results_t6_confirmation_v2"))
    ap.add_argument("--per-category", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=11)
    ap.add_argument("--port", type=int, default=7610)
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    if os.environ.get("VGC_SET_PRIOR_REG"):
        raise RuntimeError("clear prior override before confirmation")
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [
        BASELINE,
        CANDIDATE,
        "teams/candidates_mc/T6.txt",
        "data/joint_sets_regmc.json",
        EXCLUDE,
    ]
    paths += [p for p, _ in OPPONENTS.values() if p != "heuristic"]
    paths += [str(p) for p in Path("vgc_bench").rglob("*.py")]
    paths += [
        "evaluation/opening_study.py",
        "evaluation/run_t6_confirmation.py",
        "evaluation/review_t6_audits.py",
    ]
    matchups = select_matchups(20923, args.per_category, "heldout", EXCLUDE)
    paths += [m["path"] for m in matchups]
    config = {
        "baseline": BASELINE,
        "candidate": CANDIDATE,
        "opponents": OPPONENTS,
        "per_category": args.per_category,
        "repeats": args.repeats,
        "seed": 20923,
        "matchups": matchups,
        "expected_games_per_model_per_opponent": len(matchups) * 2 * args.repeats,
        "sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        "battle_rng_paired": False,
        "holdout": (
            "Excluded from this fine-tune and original checkpoint selection; "
            "not generalist history"
        ),
        "deployment_comparison": False,
        "decision": "No promotion; review populations, modes and tactical sample first",
    }
    # JSON round-trip normalizes tuples for exact resume checks.
    config = json.loads(json.dumps(config))
    manifest = args.output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise RuntimeError("confirmation code/data/config changed; choose a new output")
    atomic_json(manifest, config)
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "matchups": len(matchups),
                    "games_per_model_per_opponent": len(matchups) * 2 * args.repeats,
                }
            )
        )
        return
    with (args.output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = {}
        try:
            for population, (opponent, deterministic) in OPPONENTS.items():
                for label, checkpoint in (
                    ("baseline", BASELINE),
                    ("candidate", CANDIDATE),
                ):
                    output = args.output / f"{population}_{label}.jsonl"
                    atomic_json(
                        args.output / "status.json",
                        {"phase": population, "model": label, "started": time.time()},
                    )
                    command = [
                        sys.executable,
                        "evaluation/opening_study.py",
                        "--checkpoint",
                        checkpoint,
                        "--opponent",
                        opponent,
                        "--arms",
                        "policy",
                        "--split",
                        "heldout",
                        "--per-category",
                        str(args.per_category),
                        "--repeats",
                        str(args.repeats),
                        "--seed",
                        "20923",
                        "--port",
                        str(args.port),
                        "--exclude-matchups",
                        EXCLUDE,
                        "--output",
                        str(output),
                    ]
                    if deterministic:
                        command.append("--deterministic-opponent")
                    if label == "candidate":
                        command += [
                            "--pair-from",
                            str(args.output / f"{population}_baseline.jsonl"),
                        ]
                    if population in ("human_new", "frozen"):
                        command += [
                            "--audit-dir",
                            str(args.output / "audits" / f"{population}_{label}"),
                        ]
                    run_job(
                        command, args.output / f"{population}_{label}.log", timeout=3600
                    )
                    if (
                        len(read_rows(output))
                        != config["expected_games_per_model_per_opponent"]
                    ):
                        raise RuntimeError("incomplete population arm")
                report[population] = paired_roster_summary(
                    read_rows(args.output / f"{population}_baseline.jsonl"),
                    read_rows(args.output / f"{population}_candidate.jsonl"),
                )
                atomic_json(args.output / "scorecard.json", report)
                atomic_json(
                    args.output / "tactical_review_queue.json",
                    review(args.output / "audits"),
                )
            atomic_json(
                args.output / "status.json",
                {
                    "phase": "complete_review_required",
                    "promoted": False,
                    "ladder_games": 0,
                },
            )
        except BaseException as exc:
            atomic_json(
                args.output / "status.json",
                {"phase": "stopped_needs_review", "reason": str(exc)},
            )
            raise


if __name__ == "__main__":
    main()
