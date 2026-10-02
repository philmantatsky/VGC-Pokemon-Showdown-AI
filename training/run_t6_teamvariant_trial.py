"""Practise a set variant of the team (the user, 2026-10-01: "add the guards to
the bot and start the practice run").

T6m is T6 with two moves changed -- Torkoal Earth Power for Heat Wave, Charizard
Weather Ball for Solar Beam (TEAM_REVIEW_T6.md 4b). In the clone team tournament
(2026-09-28) it beat T6 against sand teams, 36.0% vs 24.8% (+11.2pp [+7.2,
+15.2]), and was even on the full pool (48.3 vs 46.7%). That was a human-like
pilot; our brain has only ever played T6, and moves it never practised have hurt
it before. This cycle is the human-opponent recipe that made T6ctx (35%
human-clone games, human-model previews for both sides, 50% hidden sheets, reward
unchanged, no roster weighting) with ONE change: OUR team file is T6m. The
held-out evaluation rosters stay zeroed.

Start = --init (the deployed brain T6tac; copied into the new league, never
modified) at its own step count -> +983,040 steps, saves every 491,520. Local
comparison only: never promotion or ladder.

Usage (from the repo root, under caffeinate):
  .venv/bin/python training/run_t6_teamvariant_trial.py --init <start.zip> \
      [--prepare-only]
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
import zipfile

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from evaluation.opening_study import roster_split
from training.build_league import build_league
from training.make_brainv1_config import build_config
from training.run_t6_repair_trial import digest, stop_owned_group, write_json
from vgc_bench.src.utils import chunk_obs_len, verify_league_dir

SAVE_INTERVAL = 491_520
STEPS = 983_040
NAME = "brainv1_t6m_practice1"
OUT = ROOT / f"results_{NAME}"
# --no_teampreview adds "xt" to the method tag
SAVES = OUT / "saves_fp_xt_hs_wt/reg_mc/seed1"
SWAP = "results_brainv1/saves_fp_hs_wt/reg_mc/seed1/20643840.zip"
CLONE = "results_bc/mc_A_20260920/saves_bc/seed1/2.zip"
SECOND_CLONE = "results_bc/mc_A_20260913/saves_bc/seed1/3.zip"
HUMAN_FRACTION = 0.35
TEAM = "teams/candidates_mc/T6m.txt"
REFERENCE_TEAM = "teams/candidates_mc/T6.txt"
PREVIEW_MODEL = "data/preview_t6_focus_20260923.pt"
TEMPERATURE = 1.0
EVAL_ONLY = [
    "results_bc/eval_mcB",
    "results_bc/eval_mcB_20260913",
    "results_bc/eval_mcB_20260920",
]


def start_step(init: str) -> int:
    """The step count stored in the start checkpoint (training resumes there)."""
    with zipfile.ZipFile(ROOT / init) as zf:
        return int(json.loads(zf.read("data"))["num_timesteps"])


def prepare(init: str) -> dict:
    START = start_step(init)
    INIT, INIT_SHA = init, digest(init)
    manifest_path = OUT / "experiment.json"
    if manifest_path.exists():
        recorded = json.loads(manifest_path.read_text())
        if recorded["init"] != INIT or recorded["init_sha256"] != INIT_SHA:
            raise RuntimeError("this trial was prepared from another start checkpoint")
    config = build_config(
        INIT,
        CLONE,
        resume_stem=START,
        dest=str(SAVES.relative_to(ROOT)),
        weights_dest=str((OUT / "team_weights.json").relative_to(ROOT)),
        swap_deployed=SWAP,
        eval_only_clones=EVAL_ONLY,
    )
    # as in the T6ctx cycle: a second human clone joins the fixed human stems
    config["sources"]["150"] = SECOND_CLONE
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
                "question": "after practice, does our brain play T6m (Torkoal Earth "
                "Power, Charizard Weather Ball) at least as well as T6, and better "
                "against sand",
                "single_variable": "our team file is T6m instead of T6, vs the "
                "35%-human-opponent recipe that made T6ctx",
                "reference_team": REFERENCE_TEAM,
                "reference_team_sha256": digest(REFERENCE_TEAM),
                "preview_model": PREVIEW_MODEL,
                "preview_model_sha256": digest(PREVIEW_MODEL),
                "preview_temperature": TEMPERATURE,
                "preview_in_policy_trajectory": False,
                "init": INIT,
                "init_sha256": INIT_SHA,
                "start": START,
                "end": START + STEPS,
                "save_interval": SAVE_INTERVAL,
                "team": TEAM,
                "team_sha256": digest(TEAM),
                "chunk_obs_len": chunk_obs_len,
                "prior": "data/joint_sets_regmc.json",
                "prior_sha256": digest("data/joint_sets_regmc.json"),
                "weights_sha256": digest(OUT / "team_weights.json"),
                "human_opponent_fraction": HUMAN_FRACTION,
                "human_clones": {"100": CLONE, "150": SECOND_CLONE, "200": CLONE},
                "hidden_sheet_fraction": 0.5,
                "holdout": "roster-hash heldout split zeroed (the evaluation rosters)",
                "evaluation": "evaluation/mirror_guard_ab.py --a-checkpoint <save> "
                "--a-team <this team> (head-to-head vs the deployed brain on T6) and "
                "evaluation/run_guard_ab.py --candidate <save> --candidate-plans "
                "data/opening_plans_t6m.json --without-arm <an arm of the deployed "
                "brain with the deployed guards>",
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
    parser.add_argument("--init", required=True, help="start checkpoint (read only)")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-training-minutes", type=int, default=150)
    args = parser.parse_args()
    os.chdir(ROOT)
    OUT.mkdir(exist_ok=True)
    with (OUT / "trial.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = prepare(args.init)
        START, END = manifest["start"], manifest["end"]
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
            "100,150,200",
            "--fixed_opponent_fraction",
            str(HUMAN_FRACTION),
        ]
        try:
            run_logged(command, "training", args.max_training_minutes * 60, True)
            saves = sorted(
                (p for p in SAVES.glob("*.zip") if int(p.stem) > START),
                key=lambda p: int(p.stem),
            )
            if not saves:
                raise RuntimeError("no newly trained checkpoints")
            prepare(args.init)  # refuse silently changed code or data
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
