"""The deployed configuration with extra opt-in guards versus without them.

The deployed configuration (T6 brain trained on human openings, save
21,135,360, its preview chosen by the human-trained model) was already played
for its promotion: results_candidate_vs_t6_humanpreview_21135360, six
populations x 47 held-out rosters x 22 games. That arm is reused as the
"without" side, which is valid only while nothing it depended on changed except
the guard module, where the new guards are opt-in and were off:

* every pin of results_t6_vs_deployed_v1 still hashes the same, except the
  guard code: vgc_bench/src/guards.py, vgc_bench/src/pokeenv_patches.py
  (2026-09-25: a weather-setter tracker read only by the opt-in guard
  keep_our_weather) and vgc_bench/src/policy_player.py (2026-09-26: near-tie
  mixing options, default off -- a player without mixing runs the same code);
* the reference arm played exactly the deployed checkpoint and preview model;
* each new population's opening_study manifest equals the reference's except
  the output path (the extra guards are a per-player override on our side only,
  evaluation/learned_preview_study.py --extra-guards, so the opponents and the
  recorded class-level flags are unchanged).

Only the "with" side is played, with identical arguments; battle RNG is
independent, uncertainty resamples whole opponent rosters. Delta = with minus
without. Also reports how often each extra guard changed the played action. No
promotion, no ladder.

--without-arm DIR reuses a completed run of this script as the "without" side
instead (2026-09-25: results_guard_ab_dominated_attack, once dominated_attack
was deployed): its extra guards -- which must all be deployed -- stay on for
both sides, and only the new guards differ.

--candidate CKPT (with --without-arm and --label) plays another BRAIN on the
"with" side (2026-09-26: a newly trained save against the deployed brain):
the arms may then differ in the checkpoint as well; --guards may be empty.
--candidate-plans PLANS (2026-10-01: a brain practised on T6m, a set variant of
T6) lets that brain play its own team file -- the same six species, so the
rosters, the preview model and the opponents are unchanged; the arms then differ
in the checkpoint and our team's sets, nothing else.

--baseline (2026-09-26, once T6ctx replaced the brain this study was built on)
plays the DEPLOYED configuration -- deployed brain, every deployed guard -- as
an arm of its own, with no comparison: the "without" side for later guard and
brain A/Bs on that brain (--without-arm results_brain_ab_deployed_<replay tag>).
A reused without side must have played the deployed brain.

--playbook PLAYBOOK plays our four and leads from our own plan cards (with the
opt-in guard playbook_opening for the card's turn-1 script); with
--playbook-script-only (2026-10-03) our preview stays the deployed one and the
card supplies only its turn-1 script, played when the card's own leads are out.

Usage (from the repo root; a Showdown server must listen on --port):
  .venv/bin/python evaluation/run_guard_ab.py --guards dominated_attack
  .venv/bin/python evaluation/run_guard_ab.py --guards focus_boosted \
      --without-arm results_guard_ab_dominated_attack
  .venv/bin/python evaluation/run_guard_ab.py --baseline
  .venv/bin/python evaluation/run_guard_ab.py --candidate <save.zip> --label <name> \
      --without-arm results_brain_ab_deployed_T6ctx
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
ALLOWED_CHANGED_PINS = {
    "vgc_bench/src/guards.py",
    "vgc_bench/src/pokeenv_patches.py",
    "vgc_bench/src/policy_player.py",
}
# Everything the extra guards run; hashed into the manifest, re-checked between
# populations so the code cannot change during a comparison.
GUARD_SOURCES = (
    "vgc_bench/src/guards.py",
    "vgc_bench/src/trick_room_guard.py",
    "vgc_bench/src/pokeenv_patches.py",
)
DEPLOYED = Path("results_deployed/DEPLOYED.json")


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _flags_match(reference: dict | None, candidate: dict | None) -> bool:
    """Same class-level guard flags. A guard registered after the reference was
    played appears only in the new flags, and must be off there."""
    reference, candidate = reference or {}, candidate or {}
    shared = reference.keys() & candidate.keys()
    only = [reference[k] for k in reference.keys() - shared] + [
        candidate[k] for k in candidate.keys() - shared
    ]
    return all(reference[k] == candidate[k] for k in shared) and not any(only)


def same_study(reference: dict, candidate: dict) -> list[str]:
    """Manifest fields that differ between two arms, ignoring the output path."""
    keys = (set(reference) | set(candidate)) - {"output"}
    return sorted(
        k
        for k in keys
        if (
            not _flags_match(reference.get(k), candidate.get(k))
            if k == "guard_flags"
            else reference.get(k) != candidate.get(k)
        )
    )


def unexplained(differs: list[str], candidate: bool, variant: bool) -> list[str]:
    """The manifest differences that are not the change under test: a candidate arm
    differs in its brain, and a candidate on a set variant of the team
    (--candidate-plans) also in our team's sets."""
    allowed: set[str] = set()
    if candidate:
        allowed |= {"checkpoint", "checkpoint_sha256"}
        if variant:
            allowed |= {"plans", "team_sha256"}
    return [k for k in differs if k not in allowed]


def arm_label(path: Path, prior: dict) -> str:
    """The label a completed run's arm files carry (<population>_<label>.jsonl)."""
    if prior.get("arm_label"):
        return str(prior["arm_label"])
    if prior.get("extra_guards"):
        return "with_" + "_".join(prior["extra_guards"])
    return "candidate_" + path.name.removeprefix("results_brain_ab_")


def without_arm(
    path: Path, reference: dict, deployed: set[str], brain_sha256: str | None = None
) -> tuple[list[str], str]:
    """(base guards, arm label) of a completed run reused as the "without" side.

    It must be this study (same brain, preview model, seed, repeats and
    populations) and its guards must all be deployed: they stay on for both
    sides, so the comparison isolates what differs. The brain defaults to the
    reference study's; ``brain_sha256`` names another (the deployed brain once
    it is no longer the reference's)."""
    prior = json.loads((path / "manifest.json").read_text())
    if json.loads((path / "status.json").read_text())["phase"] != (
        "complete_review_required"
    ):
        raise ValueError(f"{path} is not complete")
    brain = brain_sha256 or reference["candidate_sha256"]
    same = (
        prior["checkpoint_sha256"] == brain
        and prior["preview_model_sha256"] == reference["our_preview"]["model_sha256"]
        and prior["seed"] == reference["seed"]
        and prior["repeats"] == reference["repeats"]
        and prior["populations"] == reference["populations"]
    )
    if not same:
        raise ValueError(f"{path} is not the same study as the reference")
    base = list(prior.get("base_guards_both_sides") or []) + [
        g
        for g in prior["extra_guards"]
        if g not in (prior.get("base_guards_both_sides") or [])
    ]
    if not base or not set(base) <= deployed:
        raise ValueError(f"{path}'s extra guards are not all deployed: {base}")
    return base, arm_label(path, prior)


def without_brain(path: Path, brain_sha256: str) -> str:
    """The brain file a guard or playbook arm plays against a without side: that
    side's own file (the deployed brain), so the two arms' manifests differ only
    in the change tested. Since T6ctx the reference study's brain is no longer
    the deployed one (2026-09-27: a playbook arm would have played T6hp)."""
    checkpoint = json.loads((path / "manifest.json").read_text())["checkpoint"]
    if sha256(checkpoint) != brain_sha256:
        raise ValueError(f"{path}'s brain file {checkpoint} is not the deployed brain")
    return str(checkpoint)


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
    ap.add_argument("--guards", default="", help="comma-separated opt-in guards")
    ap.add_argument("--candidate", type=Path, default=None, help="another brain")
    ap.add_argument("--label", default=None, help="names the candidate's arm")
    ap.add_argument(
        "--candidate-plans",
        type=Path,
        default=None,
        help="the candidate's plans file: a set variant of the reference team "
        "(same six species) that the candidate brain was practised on",
    )
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument(
        "--without-arm",
        type=Path,
        default=None,
        help="a completed run of this script reused as the without side",
    )
    ap.add_argument(
        "--playbook",
        type=Path,
        default=None,
        help="our preview from this playbook (evaluation/learned_preview_study.py)",
    )
    ap.add_argument(
        "--playbook-script-only",
        action="store_true",
        help="with --playbook: our usual preview, the card's turn-1 script only",
    )
    ap.add_argument(
        "--sheet-preview",
        action="store_true",
        help="our preview reads the opponent's open team sheet (sheet_preview.py)",
    )
    ap.add_argument(
        "--baseline",
        action="store_true",
        help="play the deployed configuration as a reference arm (no comparison)",
    )
    ap.add_argument("--port", type=int, default=7610)
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    guards = [name for name in args.guards.split(",") if name]
    invalid = [g for g in guards if g not in GUARDS or g in HARD_GUARDS]
    if args.baseline and (guards or args.candidate is not None or args.without_arm):
        raise ValueError("--baseline plays the deployed configuration alone")
    if invalid or (
        not guards
        and args.candidate is None
        and not args.baseline
        and not args.sheet_preview
    ):
        raise ValueError(f"need opt-in guards, got {guards} (invalid: {invalid})")
    if args.candidate is not None and (args.without_arm is None or not args.label):
        raise ValueError("--candidate needs --without-arm and --label")
    if args.candidate_plans is not None and args.candidate is None:
        raise ValueError("--candidate-plans needs --candidate")
    if args.playbook_script_only and args.playbook is None:
        raise ValueError("--playbook-script-only needs --playbook")
    playbook_kind = "playbook_script_" if args.playbook_script_only else "playbook_"
    deployed = json.loads(DEPLOYED.read_text())["deployed"]
    if args.baseline:
        output = args.output or Path(
            f"results_brain_ab_deployed_{deployed['replay_tag']}"
        )
    elif args.candidate is not None:
        output = args.output or Path(f"results_brain_ab_{args.label}")
    elif args.playbook is not None:
        output = args.output or Path(
            f"results_guard_ab_{playbook_kind}{args.playbook.stem}_{'_'.join(guards)}"
        )
    elif args.sheet_preview:
        output = args.output or Path(
            "results_guard_ab_sheet_preview" + "".join(f"_{g}" for g in guards)
        )
    else:
        output = args.output or Path(f"results_guard_ab_{'_'.join(guards)}")

    reference = json.loads((REFERENCE / "manifest.json").read_text())
    if json.loads((REFERENCE / "status.json").read_text())["phase"] != (
        "complete_review_required"
    ):
        raise ValueError("the reference arm is not complete")
    if (
        not args.baseline
        and args.without_arm is None
        and reference["candidate_sha256"] != deployed["sha256"]
    ):
        raise ValueError(
            "the reference arm is not the deployed brain: give --without-arm "
            "(a --baseline run of the deployed configuration)"
        )
    preview = reference["our_preview"]
    if not deployed.get("learned_preview") or (
        preview["model_sha256"] != deployed["preview_model_sha256"]
    ):
        raise ValueError("the reference arm is not the deployed preview model")
    already = set(deployed["guards_extra"].split(","))
    if already & set(guards):
        raise ValueError(f"already deployed: {sorted(already & set(guards))}")
    base, without_dir, without_label = [], REFERENCE, REFERENCE_LABEL
    if args.baseline:
        base = sorted(already)
        without_dir, without_label = None, None
    elif args.without_arm is not None:
        base, without_label = without_arm(
            args.without_arm, reference, already, deployed["sha256"]
        )
        without_dir = args.without_arm
    pins = json.loads(PINS.read_text())["sha256"]
    stale = set(changed_pins(pins))
    if not stale <= ALLOWED_CHANGED_PINS:
        raise ValueError(f"pinned files changed beyond the guards: {stale}")
    if sha256(reference["candidate"]) != reference["candidate_sha256"]:
        raise ValueError("the reference checkpoint changed")
    if sha256(preview["model"]) != preview["model_sha256"]:
        raise ValueError("the preview model changed")
    checkpoint = reference["candidate"]
    if args.baseline:
        checkpoint = deployed["checkpoint"]
        if sha256(checkpoint) != deployed["sha256"]:
            raise ValueError("the deployed checkpoint changed")
    elif args.candidate is not None:
        checkpoint = str(args.candidate)
        if not args.candidate.is_file():
            raise ValueError(f"no candidate checkpoint {args.candidate}")
    elif args.without_arm is not None:
        checkpoint = without_brain(args.without_arm, deployed["sha256"])

    populations = reference["populations"]
    repeats, seed = reference["repeats"], reference["seed"]
    matchups = select_matchups(seed, PER_CATEGORY, "heldout", EXCLUDE)
    if len(matchups) != reference["matchups"]:
        raise ValueError("held-out roster selection differs from the reference")
    roster = {p.species for p in team_roster(Path(reference["team"]).read_text())}
    joint_sha = pins["data/joint_sets_regmc.json"]
    plans = reference["plans"]
    variant: dict = {}
    if args.candidate_plans is not None:
        team = Path(json.loads(args.candidate_plans.read_text())["team"])
        if {p.species for p in team_roster(team.read_text())} != roster:
            raise ValueError(f"{team} is not a set variant of {reference['team']}")
        plans = str(args.candidate_plans)
        variant = {
            "reference_plans": reference["plans"],
            "candidate_team": str(team),
            "candidate_team_sha256": sha256(team),
            "candidate_plans_sha256": sha256(args.candidate_plans),
        }

    label = (
        f"deployed_{deployed['replay_tag']}"
        if args.baseline
        else (
            playbook_kind
            if args.playbook is not None
            else "sheet_preview_"
            if args.sheet_preview
            else "with_"
        )
        + "_".join(guards)
        if args.candidate is None
        else f"candidate_{args.label}"
    )
    config = {
        "question": (
            f"the deployed configuration ({checkpoint} + {', '.join(base)}) as a "
            "reference arm"
            if args.baseline
            else f"deployed configuration + {', '.join(guards)} vs without"
            if args.candidate is None
            else f"brain {checkpoint}"
            + (f" on {variant['candidate_team']}" if variant else "")
            + f" vs the deployed brain, both with {', '.join(base + guards)}"
        ),
        "arm_label": label,
        "playbook": str(args.playbook) if args.playbook else None,
        "playbook_sha256": sha256(args.playbook) if args.playbook else None,
        **({"playbook_script_only": True} if args.playbook_script_only else {}),
        **({"sheet_preview": True} if args.sheet_preview else {}),
        "extra_guards": guards,
        "extra_guards_scope": "our player only (per-player override)",
        "base_guards_both_sides": base,
        "reference_arm": (
            None
            if without_dir is None
            else f"{without_dir}/<population>_{without_label}.jsonl"
        ),
        "reference_manifest_sha256": (
            None if without_dir is None else sha256(without_dir / "manifest.json")
        ),
        "changed_pins_accepted": sorted(stale),
        "guards_module_sha256": sha256("vgc_bench/src/guards.py"),
        "guard_sources_sha256": {path: sha256(path) for path in GUARD_SOURCES},
        "wrapper_sha256": sha256("evaluation/learned_preview_study.py"),
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256(checkpoint),
        "preview_model": preview["model"],
        "preview_model_sha256": preview["model_sha256"],
        "plans": plans,
        **variant,
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
                if (
                    not set(changed_pins(pins)) <= ALLOWED_CHANGED_PINS
                    or sha256("evaluation/learned_preview_study.py")
                    != config["wrapper_sha256"]
                    or any(
                        sha256(path) != digest
                        for path, digest in config["guard_sources_sha256"].items()
                    )
                ):
                    raise ValueError(
                        "pinned source or guard code changed during the run"
                    )
                command = [
                    sys.executable,
                    "evaluation/learned_preview_study.py",
                    "--preview-model",
                    preview["model"],
                    *(["--playbook", str(args.playbook)] if args.playbook else []),
                    *(["--playbook-script-only"] if args.playbook_script_only else []),
                    *(["--sheet-preview"] if args.sheet_preview else []),
                    "--extra-guards",
                    ",".join(base + guards),
                    "--",
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
                    str(arm),
                ]
                if deterministic:
                    command.append("--deterministic-opponent")
                run_job(command, output / f"{pop}_{label}.log", min(3600, remaining))
                rows = read_rows(arm)
                telemetry = read_rows(arm.with_suffix(".telemetry.jsonl"))
                validate_arm(rows, telemetry, matchups, repeats, roster)
                check_arm_manifest(arm.with_suffix(".manifest.json"), "mc", joint_sha)
                if without_dir is None:  # --baseline: the arm alone
                    wins = sum(r["target"] for r in rows)
                    report["populations"][pop] = {
                        "games": len(rows),
                        "win_rate": wins / len(rows),
                    }
                    report["firing"][pop] = guard_firing(telemetry, base) | {
                        "games": len(rows)
                    }
                    atomic_json(output / "scorecard.json", report)
                    continue
                ref_arm = without_dir / f"{pop}_{without_label}.jsonl"
                differs = same_study(
                    json.loads(ref_arm.with_suffix(".manifest.json").read_text()),
                    json.loads(arm.with_suffix(".manifest.json").read_text()),
                )
                differs = unexplained(
                    differs, args.candidate is not None, bool(variant)
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
            if deltas:
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
