"""A candidate T6 brain versus the deployed T6 brain, reusing the deployed arm.

The deployed T6 brain (results_deployed/champion_mc_T6.zip) was already played
as the candidate_T6 arm of Codex's results_t6_vs_deployed_v1: 47 held-out
rosters x 11 repeats x both sheet modes x six populations, seed 20923. Every
code and data file that study pinned must still hash the same (checked before
and between populations), so that arm is the reference and only the candidate
is played here, with identical arguments. Battle RNG is independent, as in that
study; uncertainty resamples whole opponent rosters. Delta = candidate minus
deployed. Also reports how varied each brain's team preview is. No promotion,
no ladder.

With --preview-model, OUR preview comes from a learned, human-trained preview
model through evaluation/learned_preview_study.py (opponent belief discarded,
so only the opening differs from the reference arm).

Usage (from the repo root; a Showdown server must listen on --port):
  .venv/bin/python evaluation/run_candidate_vs_t6.py --candidate <ckpt.zip> \\
      --label <name> [--output results_candidate_vs_t6_<name>]
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
from vgc_bench.src.set_particles import team_roster

REFERENCE = Path("results_t6_vs_deployed_v1")
REFERENCE_LABEL = "candidate_T6"
DEPLOYED = Path("results_deployed/DEPLOYED.json")


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preview_profile(rows: list[dict]) -> dict:
    """How varied the team preview is: distinct lead pairs and brings."""
    leads = Counter(tuple(sorted(r["lead_species"])) for r in rows)
    brings = Counter(tuple(sorted(r["bring_species"])) for r in rows)
    per_roster = {}
    for r in rows:
        per_roster.setdefault(r["opponent"], set()).add(
            tuple(sorted(r["lead_species"]))
        )
    top_lead, top_count = leads.most_common(1)[0] if leads else ((), 0)
    return {
        "games": len(rows),
        "distinct_lead_pairs": len(leads),
        "distinct_brings": len(brings),
        "top_lead": "+".join(top_lead),
        "top_lead_share": top_count / len(rows) if rows else 0.0,
        "rosters_with_more_than_one_lead": sum(len(v) > 1 for v in per_roster.values()),
        "leads": {"+".join(k): v for k, v in leads.most_common(6)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--port", type=int, default=7610)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument(
        "--preview-model",
        type=Path,
        default=None,
        help="human-trained preview model that chooses OUR four and leads",
    )
    args = ap.parse_args()
    os.chdir(ROOT)
    output = args.output or Path(f"results_candidate_vs_t6_{args.label}")

    reference = json.loads((REFERENCE / "manifest.json").read_text())
    stale = changed_pins(reference["sha256"])
    if stale:
        raise ValueError(f"reference arm no longer matches current files: {stale}")
    ref_checkpoint = reference["checkpoints"][REFERENCE_LABEL]
    deployed = json.loads(DEPLOYED.read_text())["deployed"]
    if reference["sha256"][ref_checkpoint] != deployed["sha256"]:
        raise ValueError("the reference arm is not the deployed brain")
    plans = reference["plans"][REFERENCE_LABEL]
    populations = reference["populations"]
    repeats, seed = reference["repeats"], reference["seed"]
    matchups = select_matchups(seed, PER_CATEGORY, "heldout", EXCLUDE)
    if matchups != reference["matchups"]:
        raise ValueError("held-out roster selection differs from the reference study")
    team = json.loads(Path(plans).read_text())["team"]
    roster = {p.species for p in team_roster(Path(team).read_text())}
    joint_sha = reference["sha256"]["data/joint_sets_regmc.json"]

    config = {
        "question": f"{args.label} vs the deployed T6 brain on T6",
        "reference_study": str(REFERENCE),
        "reference_manifest_sha256": sha256(REFERENCE / "manifest.json"),
        "reference_arm": f"{REFERENCE}/<population>_{REFERENCE_LABEL}.jsonl",
        "deployed_sha256": deployed["sha256"],
        "candidate": args.candidate,
        "candidate_sha256": sha256(args.candidate),
        "plans": plans,
        "team": team,
        "populations": populations,
        "matchups": len(matchups),
        "repeats": repeats,
        "seed": seed,
        "port": args.port,
        "set_data": "Reg M-C (data/joint_sets_regmc.json) for both arms",
        "our_preview": (
            {
                "model": str(args.preview_model),
                "model_sha256": sha256(args.preview_model),
                "wrapper": "evaluation/learned_preview_study.py",
                "wrapper_sha256": sha256("evaluation/learned_preview_study.py"),
                "opponent_belief": "discarded after preview",
            }
            if args.preview_model
            else "the candidate policy's own preview"
        ),
        "delta": "candidate minus deployed",
        "battle_rng_paired": False,
        "uncertainty": "whole-roster bootstrap; pooled = equal-population mean",
        "limits": "8 concurrent local battles, one MPS process, 3h per invocation",
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
    deadline = time.monotonic() + 3 * 3600
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report: dict = {"delta": config["delta"], "populations": {}, "preview": {}}
        deltas: dict[str, dict[str, float]] = {}
        all_candidate: list[dict] = []
        all_reference: list[dict] = []
        try:
            for pop in populations:
                opponent, deterministic = OPPONENTS[pop]
                arm = output / f"{pop}_{args.label}.jsonl"
                atomic_json(
                    output / "status.json",
                    {"phase": pop, "arm": args.label, "started": time.time()},
                )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("bounded comparison time budget exhausted")
                stale = changed_pins(reference["sha256"])
                if stale:
                    raise ValueError(f"pinned source changed during the run: {stale}")
                runner = (
                    [
                        "evaluation/learned_preview_study.py",
                        "--preview-model",
                        str(args.preview_model),
                        "--",
                    ]
                    if args.preview_model
                    else ["evaluation/opening_study.py"]
                )
                command = [
                    sys.executable,
                    *runner,
                    "--checkpoint",
                    args.candidate,
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
                    str(arm),
                ]
                if deterministic:
                    command.append("--deterministic-opponent")
                run_job(
                    command, output / f"{pop}_{args.label}.log", min(3600, remaining)
                )
                cand_rows = read_rows(arm)
                validate_arm(
                    cand_rows,
                    read_rows(arm.with_suffix(".telemetry.jsonl")),
                    matchups,
                    repeats,
                    roster,
                )
                check_arm_manifest(arm.with_suffix(".manifest.json"), "mc", joint_sha)
                ref_arm = REFERENCE / f"{pop}_{REFERENCE_LABEL}.jsonl"
                ref_rows = read_rows(ref_arm)
                check_arm_manifest(
                    ref_arm.with_suffix(".manifest.json"), "mc", joint_sha
                )
                summary = paired_roster_summary(ref_rows, cand_rows)
                for mode in summary.values():
                    mode["deployed_win_rate"] = mode.pop("baseline_win_rate")
                    mode["candidate_win_rate"] = mode.pop("candidate_win_rate")
                report["populations"][pop] = summary
                deltas[pop] = roster_deltas(ref_rows, cand_rows)
                all_candidate += cand_rows
                all_reference += ref_rows
                report["preview"] = {
                    "candidate": preview_profile(all_candidate),
                    "deployed": preview_profile(all_reference),
                }
                atomic_json(output / "scorecard.json", report)
            report["pooled"] = pooled_roster_bootstrap(deltas)
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
