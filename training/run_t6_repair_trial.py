"""One bounded T6 repair experiment; local comparison only, never promotion/ladder.

Run from the repo root under caffeinate. Full-game PPO is retained. Tiny opening
cells do not become forced-preview labels or matchup reweighting. The roster
holdout is NEW to this round, not unseen by the pretrained generalist.
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
import math
import os
import signal
import subprocess
import time

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from evaluation.opening_study import roster_split
from training.build_league import build_league
from training.make_brainv1_config import build_config
from vgc_bench.src.utils import chunk_obs_len, verify_league_dir

START = 19_660_800
END = START + 983_040
NAME = "brainv1_t6_repair1"
OUT = ROOT / f"results_{NAME}"
SAVES = OUT / "saves_fp_hs_wt/reg_mc/seed1"
INIT = "results_brainv1/saves_fp_hs_wt/reg_mc/seed1/19660800.zip"
CLONE = "results_bc/mc_A_20260920/saves_bc/seed1/2.zip"
EVAL = "results_bc/eval_mcB_20260920/saves_bc/seed2/2.zip"
TEAM = "teams/candidates_mc/T6.txt"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def prepare():
    config = build_config(
        INIT,
        CLONE,
        resume_stem=START,
        dest=str(SAVES.relative_to(ROOT)),
        weights_dest=str((OUT / "team_weights.json").relative_to(ROOT)),
        swap_deployed="results_brainv1/saves_fp_hs_wt/reg_mc/seed1/20643840.zip",
        eval_only_clones=[
            "results_bc/eval_mcB",
            "results_bc/eval_mcB_20260913",
            "results_bc/eval_mcB_20260920",
        ],
    )
    if not SAVES.exists():
        build_league(
            SAVES,
            {int(k): v for k, v in config["sources"].items()},
            START,
            [Path(p) for p in config["eval_only_roots"]],
            Path(config["weights_source"]),
            Path(config["weights_dest"]),
        )
        write_json(OUT / "league_config.json", config)
        weights = json.loads((OUT / "team_weights.json").read_text())
        holdout = []
        for path in sorted((ROOT / "teams/reg_mc").glob("MC*.txt")):
            if roster_split(path.read_text())[1] == "heldout":
                weights[path.name] = 0.0
                holdout.append(path.name)
        weights["our_team.txt"] = 0.0
        write_json(OUT / "team_weights.json", weights)
        write_json(OUT / "heldout_teams.json", holdout)
        write_json(
            OUT / "experiment.json",
            {
                "init": INIT,
                "init_sha256": digest(INIT),
                "start": START,
                "end": END,
                "team": TEAM,
                "team_sha256": digest(TEAM),
                "chunk_obs_len": chunk_obs_len,
                "prior": "data/joint_sets_regmc.json",
                "prior_sha256": digest("data/joint_sets_regmc.json"),
                "weights_sha256": digest(OUT / "team_weights.json"),
                "human_opponent_fraction": 0.2,
                "hidden_sheet_fraction": 0.5,
                "save_interval": 491_520,
                "training_uses_production_guards": False,
                "evaluation_uses_production_guards": True,
                "holdout_caveat": (
                    "Held out of this round, not pretrained generalist history"
                ),
                "evaluation_opponent": EVAL,
                "evaluation_opponent_sha256": digest(EVAL),
                "evaluation_battle_rng_paired": False,
                "no_forced_previews_or_pilot_based_reweighting": True,
                "source_sha256": {
                    str(p.relative_to(ROOT)): digest(p)
                    for p in sorted((ROOT / "vgc_bench").rglob("*.py"))
                },
            },
        )
    verify_league_dir(SAVES)
    manifest = json.loads((OUT / "experiment.json").read_text())
    for field, path in [
        ("init", INIT),
        ("team", TEAM),
        ("prior", "data/joint_sets_regmc.json"),
        ("weights", OUT / "team_weights.json"),
        ("evaluation_opponent", EVAL),
    ]:
        if digest(path) != manifest[field + "_sha256"]:
            raise RuntimeError(f"{field} changed since trial preparation")
    for path, expected in manifest["source_sha256"].items():
        if digest(path) != expected:
            raise RuntimeError(f"source changed during this trial: {path}")


def training_health():
    """Check the terminal-return invariant and freshest completed rollout."""
    latest = None
    for event in OUT.glob("logs_*/**/events.out.tfevents.*"):
        acc = EventAccumulator(str(event), size_guidance={"scalars": 0})
        acc.Reload()
        if "rollout/ep_rew_mean" in acc.Tags()["scalars"]:
            values = acc.Scalars("rollout/ep_rew_mean")
            if values and (latest is None or values[-1].wall_time > latest.wall_time):
                latest = values[-1]
            if len(values) >= 10 and any(
                not math.isfinite(v.value) or abs(v.value) > 1.05 for v in values[-3:]
            ):
                raise RuntimeError("terminal-return invariant failed")
    return latest


def stop_owned_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_logged(command, name, timeout, monitor=False):
    started = time.time()
    env = dict(os.environ)
    # Make inherited shell experiment overrides impossible to hide in this trial.
    env.pop("VGC_SET_PRIOR_REG", None)
    write_json(
        OUT / "status.json", {"phase": name, "started": started, "command": command}
    )
    with (OUT / f"{name}.log").open("a") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                if time.time() - started > timeout:
                    raise TimeoutError(f"{name} exceeded its wall-clock limit")
                if monitor:
                    progress = training_health()
                    last = progress.wall_time if progress else started
                    if time.time() - last > 900:
                        raise RuntimeError(
                            "training made no logged progress for 15 minutes"
                        )
                time.sleep(5)
            if process.returncode:
                raise RuntimeError(
                    f"{name} exited {process.returncode}; inspect its log"
                )
        except BaseException:
            stop_owned_group(process)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-training-minutes", type=int, default=150)
    args = parser.parse_args()
    os.chdir(ROOT)
    OUT.mkdir(exist_ok=True)
    with (OUT / "trial.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare()
        if args.prepare_only:
            print(f"Verified trial at {OUT}", flush=True)
            return
        # Prevent accidental fresh runs/resumes after a verdict or interrupted trial.
        if (OUT / "status.json").exists():
            raise RuntimeError(
                "trial already started; inspect status before any continuation"
            )
        py = str(ROOT / ".venv/bin/python")
        command = [
            py,
            "-u",
            "-m",
            "vgc_bench.train",
            "--fictitious_play",
            "--reg",
            "mc",
            "--run_id",
            "1",
            "--our_team",
            TEAM,
            "--joint_head",
            "--knowledge_obs",
            "--shaping_faint",
            "0.10",
            "--shaping_hp",
            "0.05",
            "--hidden_sheet_prob",
            "0.50",
            "--team_weights",
            str(OUT / "team_weights.json"),
            "--results_suffix",
            NAME,
            "--num_envs",
            "8",
            "--num_eval_workers",
            "8",
            "--port",
            "7700",
            "--device",
            "mps",
            "--learning_rate",
            "0.00003",
            "--n_epochs",
            "3",
            "--target_kl",
            "0.02",
            "--total_steps",
            str(END),
            "--save_interval",
            "491520",
            "--fixed_opponent_stems",
            "100,200",
            "--fixed_opponent_fraction",
            "0.20",
        ]
        try:
            run_logged(
                command, "training", args.max_training_minutes * 60, monitor=True
            )
            candidates = sorted(
                (p for p in SAVES.glob("*.zip") if int(p.stem) > START),
                key=lambda p: int(p.stem),
            )
            if not candidates:
                raise RuntimeError("no newly trained checkpoints; nothing to compare")
            prepare()  # Refuse comparisons under silently changed code/data.
            for label, checkpoint in [("baseline", INIT)] + [
                (p.stem, str(p)) for p in candidates
            ]:
                run_logged(
                    [
                        py,
                        "evaluation/opening_study.py",
                        "--checkpoint",
                        checkpoint,
                        "--opponent",
                        EVAL,
                        "--arms",
                        "policy",
                        "--split",
                        "heldout",
                        "--per-category",
                        "5",
                        "--repeats",
                        "6",
                        "--port",
                        "7610",
                        "--output",
                        str(OUT / f"comparison_{label}.jsonl"),
                    ],
                    f"evaluation_{label}",
                    3600,
                )
            write_json(
                OUT / "status.json",
                {
                    "phase": "comparison_complete_review_required",
                    "candidates": [str(p) for p in candidates],
                    "promoted": False,
                    "ladder_games": 0,
                },
            )
        except BaseException as exc:
            write_json(
                OUT / "status.json",
                {"phase": "stopped_needs_review", "reason": str(exc)},
            )
            raise


if __name__ == "__main__":
    main()
