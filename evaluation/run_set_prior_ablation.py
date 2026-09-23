"""Side-by-side: the T4 brain with Reg M-C set data versus the Reg M-B set data.

Question (2026-09-23): Codex's format-aware loader (vgc_bench/src/set_priors.py)
changed what the T4 brain reads about unrevealed opponent sets -- current Reg M-C
data instead of the Reg M-B data it was trained with and went 27-23 on ladder
with. Does that change help or hurt it?

Design: reuse the T4 arm of Codex's completed study (results_t6_vs_deployed_v1,
Reg M-C data) as the NEW-data arm, and play only the OLD-data arm here with
VGC_SET_PRIOR_REG=mb. Every code and data file that study pinned must still hash
the same (checked before and during the run), so the two arms differ only in the
set data: same checkpoint, team, opening-plan file, 47 held-out opponent rosters,
11 repeats, both sheet modes, seed and six opponent populations. Battle RNG is
independent, exactly as in that study. Uncertainty resamples whole opponent
rosters. Deltas are NEW minus OLD. No training, promotion or ladder play.

Usage (from the repo root; a Showdown server must listen on --port):
  .venv/bin/python evaluation/run_set_prior_ablation.py
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

import numpy as np

from evaluation.opening_study import select_matchups
from evaluation.run_t6_confirmation import (
    EXCLUDE,
    OPPONENTS,
    atomic_json,
    paired_roster_summary,
    read_rows,
    run_job,
)
from evaluation.run_t6_vs_deployed import validate_arm
from vgc_bench.src.set_particles import team_roster

REFERENCE = Path("results_t6_vs_deployed_v1")
REFERENCE_LABEL = "deployed_T4"
# DEPLOYED.json may change (a promotion) without touching anything the arms run.
UNPINNED = {"results_deployed/DEPLOYED.json"}
OLD_DATA = ("data/joint_sets_regmb.json", "data/movesets_regmb.json")
PER_CATEGORY = 10


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def changed_pins(pins: dict[str, str], skip: set[str] = UNPINNED) -> list[str]:
    """Pinned files that no longer exist or no longer hash the same."""
    changed = []
    for path, digest in pins.items():
        if path in skip:
            continue
        if not Path(path).exists() or sha256(path) != digest:
            changed.append(path)
    return changed


def check_arm_manifest(path: Path, reg: str, joint_sha: str) -> None:
    """The arm really ran with the intended set data (recorded by opening_study)."""
    record = json.loads(path.read_text())
    if record.get("set_prior_reg") != reg:
        raise ValueError(
            f"{path.name}: set data {record.get('set_prior_reg')} != {reg}"
        )
    if record.get("set_prior_sha256") != joint_sha:
        raise ValueError(f"{path.name}: set data hash differs from the pinned file")


def roster_deltas(control: list[dict], candidate: list[dict]) -> dict[str, float]:
    """Per opponent roster: candidate win rate minus control win rate."""

    def by_roster(rows: list[dict]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for row in rows:
            out.setdefault(row["opponent"], []).append(float(row["target"]))
        return out

    a, b = by_roster(control), by_roster(candidate)
    if set(a) != set(b) or any(len(a[k]) != len(b[k]) for k in a):
        raise ValueError("unmatched roster cells between the arms")
    return {k: float(np.mean(b[k]) - np.mean(a[k])) for k in sorted(a)}


def pooled_roster_bootstrap(
    deltas_by_population: dict[str, dict[str, float]],
    resamples: int = 10_000,
    seed: int = 20923,
) -> dict[str, float | list[float]]:
    """Equal-population mean delta; rosters resampled jointly across populations."""
    populations = sorted(deltas_by_population)
    if not populations:
        raise ValueError("no populations")
    rosters = sorted(deltas_by_population[populations[0]])
    if any(sorted(deltas_by_population[p]) != rosters for p in populations):
        raise ValueError("populations cover different rosters")
    table = np.array(
        [[deltas_by_population[p][r] for r in rosters] for p in populations]
    )
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(rosters), size=(resamples, len(rosters)))
    boot = table[:, picks].mean(axis=2).mean(axis=0)
    return {
        "populations": len(populations),
        "rosters": len(rosters),
        "mean_delta": float(table.mean()),
        "bootstrap_95_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
        "resamples": resamples,
        "seed": seed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output", type=Path, default=Path("results_set_prior_ablation_T4")
    )
    ap.add_argument("--port", type=int, default=7610)
    ap.add_argument(
        "--prepare-only",
        action="store_true",
        help="run every check and write the manifest, play no games",
    )
    args = ap.parse_args()
    os.chdir(ROOT)

    reference = json.loads((REFERENCE / "manifest.json").read_text())
    stale = changed_pins(reference["sha256"])
    if stale:
        raise ValueError(f"reference arm no longer matches current files: {stale}")
    checkpoint = reference["checkpoints"][REFERENCE_LABEL]
    plans = reference["plans"][REFERENCE_LABEL]
    populations = reference["populations"]
    repeats, seed = reference["repeats"], reference["seed"]
    matchups = select_matchups(seed, PER_CATEGORY, "heldout", EXCLUDE)
    if matchups != reference["matchups"]:
        raise ValueError("held-out roster selection differs from the reference study")
    team = json.loads(Path(plans).read_text())["team"]
    roster = {p.species for p in team_roster(Path(team).read_text())}
    old_joint_sha = sha256(OLD_DATA[0])
    new_joint_sha = reference["sha256"]["data/joint_sets_regmc.json"]

    config = {
        "question": "T4 brain: Reg M-C set data (new, Codex) vs Reg M-B set data (old)",
        "reference_study": str(REFERENCE),
        "reference_manifest_sha256": sha256(REFERENCE / "manifest.json"),
        "new_data_arm": f"{REFERENCE}/<population>_{REFERENCE_LABEL}.jsonl",
        "old_data_arm": f"{args.output}/<population>_T4_mbdata.jsonl",
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256(checkpoint),
        "plans": plans,
        "team": team,
        "populations": populations,
        "matchups": len(matchups),
        "repeats": repeats,
        "seed": seed,
        "port": args.port,
        "old_data_sha256": {p: sha256(p) for p in OLD_DATA},
        "new_data_sha256": {"data/joint_sets_regmc.json": new_joint_sha},
        "delta": "new minus old",
        "battle_rng_paired": False,
        "uncertainty": "whole-roster bootstrap; pooled = equal-population mean",
        "limits": "8 concurrent local battles, one MPS process, 3h per invocation",
        "decision": "Stop for review. Never promote or start ladder automatically.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("study changed: use a new output directory")
    atomic_json(manifest, config)
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "populations": populations,
                    "matchups": len(matchups),
                    "games_to_play": len(populations) * len(matchups) * 2 * repeats,
                }
            )
        )
        return

    os.environ["VGC_SET_PRIOR_REG"] = "mb"  # inherited by every game subprocess
    deadline = time.monotonic() + 3 * 3600
    with (args.output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report: dict = {"delta": config["delta"], "populations": {}}
        deltas: dict[str, dict[str, float]] = {}
        try:
            for pop in populations:
                opponent, deterministic = OPPONENTS[pop]
                output = args.output / f"{pop}_T4_mbdata.jsonl"
                atomic_json(
                    args.output / "status.json",
                    {"phase": pop, "arm": "old_mb_data", "started": time.time()},
                )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("bounded comparison time budget exhausted")
                stale = changed_pins(reference["sha256"])
                if stale:
                    raise ValueError(f"pinned source changed during the run: {stale}")
                command = [
                    sys.executable,
                    "evaluation/opening_study.py",
                    "--checkpoint",
                    checkpoint,
                    "--opponent",
                    opponent,
                    "--plans",
                    plans,
                    "--arms",
                    "policy",
                    "--split",
                    "heldout",
                    "--per-category",
                    str(PER_CATEGORY),
                    "--repeats",
                    str(repeats),
                    "--seed",
                    str(seed),
                    "--port",
                    str(args.port),
                    "--exclude-matchups",
                    EXCLUDE,
                    "--output",
                    str(output),
                ]
                if deterministic:
                    command.append("--deterministic-opponent")
                run_job(
                    command, args.output / f"{pop}_T4_mbdata.log", min(3600, remaining)
                )
                old_rows = read_rows(output)
                validate_arm(
                    old_rows,
                    read_rows(output.with_suffix(".telemetry.jsonl")),
                    matchups,
                    repeats,
                    roster,
                )
                check_arm_manifest(
                    output.with_suffix(".manifest.json"), "mb", old_joint_sha
                )
                ref_output = REFERENCE / f"{pop}_{REFERENCE_LABEL}.jsonl"
                new_rows = read_rows(ref_output)
                check_arm_manifest(
                    ref_output.with_suffix(".manifest.json"), "mc", new_joint_sha
                )
                summary = paired_roster_summary(old_rows, new_rows)
                for mode in summary.values():
                    mode["old_mb_data_win_rate"] = mode.pop("baseline_win_rate")
                    mode["new_mc_data_win_rate"] = mode.pop("candidate_win_rate")
                report["populations"][pop] = summary
                deltas[pop] = roster_deltas(old_rows, new_rows)
                atomic_json(args.output / "scorecard.json", report)
            report["pooled"] = pooled_roster_bootstrap(deltas)
            atomic_json(args.output / "scorecard.json", report)
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
