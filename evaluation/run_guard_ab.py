"""The deployed configuration with extra opt-in guards versus without them.

The deployed configuration (T6 brain trained on human openings, save
21,135,360, its preview chosen by the human-trained model) was already played
for its promotion: results_candidate_vs_t6_humanpreview_21135360, six
populations x 47 held-out rosters x 22 games. That arm is reused as the
"without" side, which is valid only while nothing it depended on changed except
the guard module, where the new guards are opt-in and were off:

* every pin of results_t6_vs_deployed_v1 still hashes the same, except
  vgc_bench/src/guards.py;
* the reference arm played exactly the deployed checkpoint and preview model;
* each new population's opening_study manifest equals the reference's except
  the output path (the extra guards are a per-player override on our side only,
  evaluation/learned_preview_study.py --extra-guards, so the opponents and the
  recorded class-level flags are unchanged).

Only the "with" side is played, with identical arguments; battle RNG is
independent, uncertainty resamples whole opponent rosters. Delta = with minus
without. Also reports how often each extra guard changed the played action. No
promotion, no ladder.

Usage (from the repo root; a Showdown server must listen on --port):
  .venv/bin/python evaluation/run_guard_ab.py --guards dominated_attack
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
from evaluation.run_set_prior_ablation import (
    PER_CATEGORY,
    changed_pins,
    check_arm_manifest,
    pooled_roster_bootstrap,
    roster_deltas,
)
from evaluation.run_t6_confirmation import (
    EXCLUDE,
    OPPONENTS,
    atomic_json,
    paired_roster_summary,
    read_rows,
    run_job,
)
from evaluation.run_t6_vs_deployed import validate_arm
from vgc_bench.src.guards import GUARDS, HARD_GUARDS
from vgc_bench.src.set_particles import team_roster

REFERENCE = Path("results_candidate_vs_t6_humanpreview_21135360")
REFERENCE_LABEL = "humanpreview_21135360"
PINS = Path("results_t6_vs_deployed_v1/manifest.json")
ALLOWED_CHANGED_PINS = {"vgc_bench/src/guards.py"}
DEPLOYED = Path("results_deployed/DEPLOYED.json")


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_study(reference: dict, candidate: dict) -> list[str]:
    """Manifest fields that differ between two arms, ignoring the output path."""
    keys = (set(reference) | set(candidate)) - {"output"}
    return sorted(k for k in keys if reference.get(k) != candidate.get(k))


def guard_firing(telemetry: list[dict], guards: list[str]) -> dict[str, int]:
    """Summed fire counts for the extra guards across every cell."""
    counts: Counter[str] = Counter()
    for cell in telemetry:
        for key, value in (cell.get("guard_counts") or {}).items():
            if any(key == g or key.startswith(g + ":") for g in guards):
                counts[key] += int(value)
    return dict(counts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--guards", required=True, help="comma-separated opt-in guards")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--port", type=int, default=7610)
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    guards = [name for name in args.guards.split(",") if name]
    invalid = [g for g in guards if g not in GUARDS or g in HARD_GUARDS]
    if not guards or invalid:
        raise ValueError(f"need opt-in guards, got {guards} (invalid: {invalid})")
    output = args.output or Path(f"results_guard_ab_{'_'.join(guards)}")

    reference = json.loads((REFERENCE / "manifest.json").read_text())
    if json.loads((REFERENCE / "status.json").read_text())["phase"] != (
        "complete_review_required"
    ):
        raise ValueError("the reference arm is not complete")
    deployed = json.loads(DEPLOYED.read_text())["deployed"]
    if reference["candidate_sha256"] != deployed["sha256"]:
        raise ValueError("the reference arm is not the deployed brain")
    preview = reference["our_preview"]
    if not deployed.get("learned_preview") or (
        preview["model_sha256"] != deployed["preview_model_sha256"]
    ):
        raise ValueError("the reference arm is not the deployed preview model")
    already = set(deployed["guards_extra"].split(","))
    if already & set(guards):
        raise ValueError(f"already deployed: {sorted(already & set(guards))}")
    pins = json.loads(PINS.read_text())["sha256"]
    stale = set(changed_pins(pins))
    if not stale <= ALLOWED_CHANGED_PINS:
        raise ValueError(f"pinned files changed beyond the guards: {stale}")
    if sha256(reference["candidate"]) != reference["candidate_sha256"]:
        raise ValueError("the reference checkpoint changed")
    if sha256(preview["model"]) != preview["model_sha256"]:
        raise ValueError("the preview model changed")

    populations = reference["populations"]
    repeats, seed = reference["repeats"], reference["seed"]
    matchups = select_matchups(seed, PER_CATEGORY, "heldout", EXCLUDE)
    if len(matchups) != reference["matchups"]:
        raise ValueError("held-out roster selection differs from the reference")
    roster = {p.species for p in team_roster(Path(reference["team"]).read_text())}
    joint_sha = pins["data/joint_sets_regmc.json"]

    config = {
        "question": f"deployed configuration + {', '.join(guards)} vs without",
        "extra_guards": guards,
        "extra_guards_scope": "our player only (per-player override)",
        "reference_arm": f"{REFERENCE}/<population>_{REFERENCE_LABEL}.jsonl",
        "reference_manifest_sha256": sha256(REFERENCE / "manifest.json"),
        "changed_pins_accepted": sorted(stale),
        "guards_module_sha256": sha256("vgc_bench/src/guards.py"),
        "wrapper_sha256": sha256("evaluation/learned_preview_study.py"),
        "checkpoint": reference["candidate"],
        "checkpoint_sha256": reference["candidate_sha256"],
        "preview_model": preview["model"],
        "preview_model_sha256": preview["model_sha256"],
        "plans": reference["plans"],
        "populations": populations,
        "matchups": len(matchups),
        "repeats": repeats,
        "seed": seed,
        "port": args.port,
        "delta": "with the extra guards minus without",
        "battle_rng_paired": False,
        "uncertainty": "whole-roster bootstrap; pooled = equal-population mean",
        "decision": "Stop for review. Never promote or start ladder automatically.",
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("study changed: use a new output directory")
    atomic_json(manifest, config)
    if args.prepare_only:
        games = len(populations) * len(matchups) * 2 * repeats
        print(json.dumps({"populations": populations, "games_to_play": games}))
        return

    os.environ.pop("VGC_SET_PRIOR_REG", None)  # both arms read the Reg M-C data
    label = "with_" + "_".join(guards)
    deadline = time.monotonic() + 3 * 3600
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report: dict = {"delta": config["delta"], "populations": {}, "firing": {}}
        deltas: dict[str, dict[str, float]] = {}
        firing: Counter[str] = Counter()
        try:
            for pop in populations:
                opponent, deterministic = OPPONENTS[pop]
                arm = output / f"{pop}_{label}.jsonl"
                atomic_json(
                    output / "status.json",
                    {"phase": pop, "arm": label, "started": time.time()},
                )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("bounded comparison time budget exhausted")
                if not set(changed_pins(pins)) <= ALLOWED_CHANGED_PINS:
                    raise ValueError("pinned source changed during the run")
                command = [
                    sys.executable,
                    "evaluation/learned_preview_study.py",
                    "--preview-model",
                    preview["model"],
                    "--extra-guards",
                    ",".join(guards),
                    "--",
                    "--checkpoint",
                    reference["candidate"],
                    "--opponent",
                    opponent,
                    "--plans",
                    reference["plans"],
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
                    str(arm),
                ]
                if deterministic:
                    command.append("--deterministic-opponent")
                run_job(command, output / f"{pop}_{label}.log", min(3600, remaining))
                rows = read_rows(arm)
                telemetry = read_rows(arm.with_suffix(".telemetry.jsonl"))
                validate_arm(rows, telemetry, matchups, repeats, roster)
                check_arm_manifest(arm.with_suffix(".manifest.json"), "mc", joint_sha)
                ref_arm = REFERENCE / f"{pop}_{REFERENCE_LABEL}.jsonl"
                differs = same_study(
                    json.loads(ref_arm.with_suffix(".manifest.json").read_text()),
                    json.loads(arm.with_suffix(".manifest.json").read_text()),
                )
                if differs:
                    raise ValueError(f"{pop}: arms differ beyond output: {differs}")
                ref_rows = read_rows(ref_arm)
                summary = paired_roster_summary(ref_rows, rows)
                for mode in summary.values():
                    mode["without_win_rate"] = mode.pop("baseline_win_rate")
                    mode["with_win_rate"] = mode.pop("candidate_win_rate")
                report["populations"][pop] = summary
                deltas[pop] = roster_deltas(ref_rows, rows)
                pop_firing = guard_firing(telemetry, guards)
                firing.update(pop_firing)
                report["firing"][pop] = pop_firing | {"games": len(rows)}
                atomic_json(output / "scorecard.json", report)
            report["pooled"] = pooled_roster_bootstrap(deltas)
            report["firing"]["total"] = dict(firing)
            atomic_json(output / "scorecard.json", report)
            atomic_json(
                output / "status.json",
                {
                    "phase": "complete_review_required",
                    "promoted": False,
                    "ladder_games": 0,
                },
            )
        except BaseException as exc:
            atomic_json(
                output / "status.json",
                {"phase": "stopped_needs_review", "reason": str(exc)},
            )
            raise


if __name__ == "__main__":
    main()
