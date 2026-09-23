"""One bounded local comparison: deployed T4 configuration versus repaired T6.

Match opponent rosters, NOT opponent preview choices across different own teams.
Opponents must be allowed to react to the team they actually face. Battle RNG is
independent. No training, promotion, ladder play or automatic next experiment.
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
import time
from collections import Counter

from evaluation.opening_study import select_matchups
from evaluation.review_t6_audits import review
from evaluation.run_t6_confirmation import (
    CANDIDATE,
    EXCLUDE,
    OPPONENTS,
    atomic_json,
    paired_roster_summary,
    read_rows,
    run_job,
)
from vgc_bench.src.set_particles import team_roster


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def deployed_config(path="results_deployed/DEPLOYED.json"):
    config = json.loads(Path(path).read_text())["deployed"]
    for file_key, hash_key in (("checkpoint", "sha256"), ("team", "team_sha256")):
        if sha256(config[file_key]) != config[hash_key]:
            raise ValueError(f"deployed {file_key} hash mismatch")
    if config["reg"] != "mc" or config["search"] or not config["knowledge_obs"]:
        raise ValueError("deployment configuration is not supported by this study")
    if set(config["guards_extra"].split(",")) != {
        "resisted_target",
        "overkill_split",
        "dominated_weather_ball_weather",
    }:
        raise ValueError("deployment guard profile differs from study")
    return config


def validate_arm(rows, telemetry, matchups, repeats, roster):
    expected = Counter(
        {(Path(m["path"]).name, h): repeats for m in matchups for h in (True, False)}
    )
    actual = Counter((r["opponent"], r["hidden_sheets"]) for r in rows)
    if actual != expected or len({r["battle"] for r in rows}) != len(rows):
        raise ValueError("incomplete, duplicated or unexpected battle cells")
    cells = Counter((r["opponent"], r["hidden_sheets"]) for r in telemetry)
    if cells != Counter({key: 1 for key in expected}):
        raise ValueError("incomplete or duplicate telemetry")
    for row in rows:
        if row["arm"] != "policy" or row["target"] not in (0.0, 1.0):
            raise ValueError("invalid battle outcome or arm")
        if len(row["bring_species"]) != 4 or not set(row["bring_species"]) <= roster:
            raise ValueError("wrong own team in result")
    for cell in telemetry:
        if cell["games"] != repeats or cell["preview_pairing_mismatches"]:
            raise ValueError("invalid cell telemetry")
        if any(
            v and ("error:" in k or "fallback_replaced" in k)
            for k, v in cell["guard_counts"].items()
        ):
            raise ValueError("guard error in evaluated cell")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("results_t6_vs_deployed_v1"))
    ap.add_argument("--per-category", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=11)
    ap.add_argument("--port", type=int, default=7610)
    ap.add_argument("--populations", default=",".join(OPPONENTS))
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    if os.environ.get("VGC_SET_PRIOR_REG"):
        raise ValueError("clear prior override before evaluation")
    if min(args.repeats, args.per_category) < 1:
        raise ValueError("positive study sizes required")
    populations = args.populations.split(",")
    if (
        len(set(populations)) != len(populations)
        or not set(populations) <= OPPONENTS.keys()
    ):
        raise ValueError("unknown or repeated populations")
    deployment = deployed_config()
    plans = {
        "deployed_T4": "data/opening_plans_t4_control.json",
        "candidate_T6": "data/opening_plans_t6.json",
    }
    checkpoints = {"deployed_T4": deployment["checkpoint"], "candidate_T6": CANDIDATE}
    if json.loads(Path(plans["deployed_T4"]).read_text())["team"] != deployment["team"]:
        raise ValueError("T4 team does not match deployment")
    original = json.loads(Path("results_t6_confirmation_v2/manifest.json").read_text())
    if sha256(CANDIDATE) != original["sha256"][CANDIDATE]:
        raise ValueError("candidate changed since confirmation")
    matchups = select_matchups(20923, args.per_category, "heldout", EXCLUDE)
    if not matchups:
        raise ValueError("empty matchup set")
    sources = (
        list(checkpoints.values())
        + list(plans.values())
        + [
            deployment["team"],
            "teams/candidates_mc/T6.txt",
            "data/joint_sets_regmc.json",
            "results_deployed/DEPLOYED.json",
            "results_t6_confirmation_v2/manifest.json",
            EXCLUDE,
            "evaluation/opening_study.py",
            "evaluation/run_t6_confirmation.py",
            "evaluation/run_t6_vs_deployed.py",
            "evaluation/review_t6_audits.py",
        ]
    )
    sources += [
        p for pop in populations for p, _ in [OPPONENTS[pop]] if p != "heuristic"
    ]
    sources += [str(p) for p in Path("vgc_bench").rglob("*.py")]
    sources += [m["path"] for m in matchups]
    config = {
        "checkpoints": checkpoints,
        "plans": plans,
        "populations": populations,
        "matchups": matchups,
        "repeats": args.repeats,
        "seed": 20923,
        "port": args.port,
        "games_per_configuration_per_population": len(matchups) * 2 * args.repeats,
        "sha256": {p: sha256(p) for p in sources},
        "battle_rng_paired": False,
        "opponent_previews_paired": False,
        "comparison": (
            "Different own teams and weights; opponents choose preview naturally."
        ),
        "holdout": original["holdout"],
        "limits": "8 concurrent local battles, one MPS process, 3h per invocation",
        "decision": "Stop for review. Never promote or start ladder automatically.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("comparison changed: use a new output directory")
    atomic_json(manifest, config)
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "populations": populations,
                    "matchups": len(matchups),
                    "games_per_arm": len(matchups) * 2 * args.repeats,
                }
            )
        )
        return
    deadline = time.monotonic() + 3 * 3600
    with (args.output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = {"comparison": config["comparison"], "populations": {}}
        try:
            for pop in populations:
                opponent, deterministic = OPPONENTS[pop]
                results = {}
                for label, checkpoint in checkpoints.items():
                    output = args.output / f"{pop}_{label}.jsonl"
                    atomic_json(
                        args.output / "status.json",
                        {"phase": pop, "configuration": label, "started": time.time()},
                    )
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("bounded comparison time budget exhausted")
                    if any(sha256(p) != h for p, h in config["sha256"].items()):
                        raise ValueError("pinned source changed during comparison")
                    command = [
                        sys.executable,
                        "evaluation/opening_study.py",
                        "--checkpoint",
                        checkpoint,
                        "--opponent",
                        opponent,
                        "--plans",
                        plans[label],
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
                    if pop in ("human_new", "frozen"):
                        command += [
                            "--audit-dir",
                            str(args.output / "audits" / f"{pop}_{label}"),
                        ]
                    run_job(
                        command,
                        args.output / f"{pop}_{label}.log",
                        min(3600, remaining),
                    )
                    rows = read_rows(output)
                    team = json.loads(Path(plans[label]).read_text())["team"]
                    roster = {p.species for p in team_roster(Path(team).read_text())}
                    validate_arm(
                        rows,
                        read_rows(output.with_suffix(".telemetry.jsonl")),
                        matchups,
                        args.repeats,
                        roster,
                    )
                    results[label] = rows
                summary = paired_roster_summary(
                    results["deployed_T4"], results["candidate_T6"]
                )
                for mode in summary.values():
                    mode["deployed_T4_win_rate"] = mode.pop("baseline_win_rate")
                    mode["candidate_T6_win_rate"] = mode.pop("candidate_win_rate")
                report["populations"][pop] = summary
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
