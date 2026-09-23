"""One bounded T6 training cycle: human-style openings (user-directed, 2026-09-23).

The user: "it needs to understand to bring other pokemon in and not the same
everytime because of opponent matchups ... find a player with a lot of replays
and show the bot how many different combinations they make based off their
opponents". Pilots of our kind of team vary their four and their leads by
opponent (gankyburner, 19 games with five of our six: 7 lead pairs, 15-4,
Farigiraf + Incineroar 11 times; dksnnfud: Blastoise + Farigiraf), where the T6
brain leads Farigiraf + Torkoal in every game. The preview model
data/preview_t6_focus_20260923.pt learned those choices from replays (pilots of
teams sharing >= 4 of our six weighted x10). This cycle changes ONE thing
against Codex's T6 recipe: the preview is taken out of the policy
(--no_teampreview) and sampled from that model for both sides
(training/human_preview.py), so the battle policy learns to play human-chosen
openings; at play time the same model chooses the preview.

Start = the deployed T6 brain (step 20,152,320; copied into the new league,
never modified) -> +983,040 steps, saves every 491,520; same pool recipe, fixed
20% human-clone share, 50% hidden sheets, unchanged reward/shaping, the same
roster-hash holdout zeroed. Local comparison only: never promotion or ladder.

Usage (from the repo root, under caffeinate):
  .venv/bin/python training/run_t6_human_preview_trial.py [--prepare-only]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import fcntl
import json
import math
import os
import subprocess
import time

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from evaluation.opening_study import roster_split
from training.build_league import build_league
from training.make_brainv1_config import build_config
from training.run_t6_repair_trial import digest, stop_owned_group, write_json
from vgc_bench.src.utils import chunk_obs_len, verify_league_dir

START = 20_152_320
END = START + 983_040
SAVE_INTERVAL = 491_520
NAME = "brainv1_t6_humanpreview1"
OUT = ROOT / f"results_{NAME}"
# --no_teampreview adds "xt" to the method tag
SAVES = OUT / "saves_fp_xt_hs_wt/reg_mc/seed1"
INIT = "results_deployed/champion_mc_T6.zip"
INIT_SHA = "bed6840ee8a1d995c67aeb7d9dc69c59c27547633ba8c596bdfc6fe1fb1d83b3"
SWAP = "results_brainv1/saves_fp_hs_wt/reg_mc/seed1/20643840.zip"
CLONE = "results_bc/mc_A_20260920/saves_bc/seed1/2.zip"
TEAM = "teams/candidates_mc/T6.txt"
PREVIEW_MODEL = "data/preview_t6_focus_20260923.pt"
TEMPERATURE = 1.0
EVAL_ONLY = [
    "results_bc/eval_mcB",
    "results_bc/eval_mcB_20260913",
    "results_bc/eval_mcB_20260920",
]


def prepare() -> dict:
    if digest(INIT) != INIT_SHA:
        raise RuntimeError(
            "the deployed T6 checkpoint changed; refuse to train from it"
        )
    config = build_config(
        INIT,
        CLONE,
        resume_stem=START,
        dest=str(SAVES.relative_to(ROOT)),
        weights_dest=str((OUT / "team_weights.json").relative_to(ROOT)),
        swap_deployed=SWAP,
        eval_only_clones=EVAL_ONLY,
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
                "question": "do human-chosen openings fix the fixed script",
                "single_variable": "human-model previews vs Codex's T6 repair recipe",
                "preview_model": PREVIEW_MODEL,
                "preview_model_sha256": digest(PREVIEW_MODEL),
                "preview_temperature": TEMPERATURE,
                "preview_in_policy_trajectory": False,
                "init": INIT,
                "init_sha256": INIT_SHA,
                "start": START,
                "end": END,
                "save_interval": SAVE_INTERVAL,
                "team": TEAM,
                "team_sha256": digest(TEAM),
                "chunk_obs_len": chunk_obs_len,
                "prior": "data/joint_sets_regmc.json",
                "prior_sha256": digest("data/joint_sets_regmc.json"),
                "weights_sha256": digest(OUT / "team_weights.json"),
                "human_opponent_fraction": 0.2,
                "hidden_sheet_fraction": 0.5,
                "holdout": "roster-hash heldout split zeroed (the evaluation rosters)",
                "evaluation": "evaluation/run_candidate_vs_t6.py per save",
                "source_sha256": {
                    str(p.relative_to(ROOT)): digest(p)
                    for p in sorted((ROOT / "vgc_bench").rglob("*.py"))
                }
                | {
                    "training/human_preview.py": digest(
                        ROOT / "training/human_preview.py"
                    )
                },
            },
        )
    verify_league_dir(SAVES)
    manifest = json.loads((OUT / "experiment.json").read_text())
    for field, path in [
        ("team", TEAM),
        ("prior", "data/joint_sets_regmc.json"),
        ("weights", OUT / "team_weights.json"),
        ("preview_model", PREVIEW_MODEL),
    ]:
        if digest(path) != manifest[field + "_sha256"]:
            raise RuntimeError(f"{field} changed since trial preparation")
    for path, expected in manifest["source_sha256"].items():
        if digest(ROOT / path) != expected:
            raise RuntimeError(f"source changed during this trial: {path}")
    return manifest


def training_health():
    """Latest ep_rew_mean event; raises when the return invariant breaks."""
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


def run_logged(command: list[str], name: str, timeout: float, monitor: bool) -> None:
    """Run one child under a wall-clock ceiling (and progress checks when training)."""
    started = time.time()
    env = dict(os.environ)
    env.pop("VGC_SET_PRIOR_REG", None)  # T6 reads the Reg M-C data it trained with
    env["VGC_HUMAN_PREVIEW_MODEL"] = str(ROOT / PREVIEW_MODEL)
    env["VGC_HUMAN_PREVIEW_TEMPERATURE"] = str(TEMPERATURE)
    write_json(OUT / "status.json", {"phase": name, "started": started})
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
                        raise RuntimeError("no logged training progress for 15 minutes")
                time.sleep(10)
            if process.returncode:
                raise RuntimeError(f"{name} exited {process.returncode}; see its log")
        except BaseException:
            stop_owned_group(process)
            raise


def main() -> None:
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
        if (OUT / "status.json").exists():
            raise RuntimeError("trial already started; inspect status first")
        py = str(ROOT / ".venv/bin/python")
        command = [
            py,
            "-u",
            "training/human_preview.py",
            "--",
            "--no_teampreview",
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
            str(SAVE_INTERVAL),
            "--fixed_opponent_stems",
            "100,200",
            "--fixed_opponent_fraction",
            "0.20",
        ]
        try:
            run_logged(command, "training", args.max_training_minutes * 60, True)
            saves = sorted(
                (p for p in SAVES.glob("*.zip") if int(p.stem) > START),
                key=lambda p: int(p.stem),
            )
            if not saves:
                raise RuntimeError("no newly trained checkpoints")
            prepare()  # refuse silently changed code or data
            write_json(
                OUT / "status.json",
                {
                    "phase": "training_complete_evaluation_next",
                    "saves": [str(p.relative_to(ROOT)) for p in saves],
                    "finished": time.time(),
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
