"""Positions for the tactical teacher (training/tactical_teacher.py).

The deployed bot (DEPLOYED.json: brain, team, learned preview, guards) plays
local games against TRAINING-role opponents only -- the two Reg M-C human
clones of the last training cycle, the previous brain T6hp, and the deployed
brain itself -- on train-split rosters (the held-out evaluation rosters and the
eval-only clones never appear, so the held-out battery stays clean). Every move
decision records exactly what the network saw (observation, action mask), the
actions played, and the teacher's facts for both positions (slot 2's conditioned
on slot 1's played action) -- since 2026-09-27 also the doomed facts (the chance
of being knocked out before moving, its Protect and wasted actions), and since
2026-09-28 the Fake Out pairing facts for slot 2 (drain / receive). The
training target is built later from the network's own distribution, so the
temperature can change without replaying.

Usage (from the repo root; a Showdown server must listen on --port):
  .venv/bin/python training/gen_tactical_data.py --games-per-cell 150
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import asyncio
import hashlib
import json
import os
import random
import time

import numpy as np
import torch
from poke_env.ps_client import ServerConfiguration

from evaluation.mirror_guard_ab import _NoCache
from evaluation.opening_study import (
    StudyOpponent,
    StudyPlayer,
    fresh_local_account,
    roster_split,
)
from tools.deployed_config import resolve
from training.tactical_teacher import doomed_facts, pair_facts, position_facts
from vgc_bench.src import pokeenv_patches
from vgc_bench.src.guards import GUARDS, HARD_GUARDS
from vgc_bench.src.policy import MaskedActorCriticPolicy
from vgc_bench.src.policy_player import PolicyPlayer
from vgc_bench.src.teams import RandomTeamBuilder
from vgc_bench.src.utils import (
    act_len,
    chunk_obs_len,
    format_map,
    refuse_eval_only_checkpoint,
)

OPPONENTS = {
    "clone_0920": ("results_bc/mc_A_20260920/saves_bc/seed1/2.zip", False),
    "clone_0913": ("results_bc/mc_A_20260913/saves_bc/seed1/3.zip", False),
    "t6hp": ("results_deployed/champion_mc_T6hp.zip", True),
    "self": (None, True),  # the deployed brain
}
MOVE_ACTIONS = slice(7, 47)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RecordingPlayer(StudyPlayer):
    """The deployed player; every move decision is recorded with the teacher's facts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records: list[dict] = []
        self.teacher_errors = 0

    def _guarded_action(self, battle, obs_dict, mask):
        action = super()._guarded_action(battle, obs_dict, mask)
        if battle.teampreview:
            return action
        try:
            m = mask if isinstance(mask, torch.Tensor) else torch.as_tensor(mask)
            m = m.reshape(1, -1).to("cpu").long()
            row = m[0].numpy().astype(np.int8)
            if (
                not row[:act_len][MOVE_ACTIONS].any()
                and not row[act_len:][MOVE_ACTIONS].any()
            ):
                return action  # a forced switch: nothing for the teacher
            a0, a1 = int(action[0]), int(action[1])
            f0, v0 = position_facts(battle, 0, row[:act_len], None)
            m1 = MaskedActorCriticPolicy._update_mask(m, torch.tensor([[a0]]))
            f1, v1 = position_facts(battle, 1, m1[0, act_len:].numpy(), a0)
            d0, p0, w0 = doomed_facts(battle, 0, row[:act_len])
            d1, p1, w1 = doomed_facts(battle, 1, m1[0, act_len:].numpy(), a0)
            dr1, rc1 = pair_facts(battle, m1[0, act_len:].numpy(), a0)
            none = np.zeros(act_len, dtype=bool)
            self.records.append(
                {
                    "obs": obs_dict["observation"][0]
                    .detach()
                    .to("cpu")
                    .numpy()
                    .astype(np.float32),
                    "mask": row,
                    "played": np.array([a0, a1], dtype=np.int16),
                    "useless": np.stack([f0, f1]),
                    "values": np.stack([v0, v1]),
                    "doomed": np.array([d0, d1], dtype=np.float32),
                    "protect": np.stack([p0, p1]),
                    "wasted": np.stack([w0, w1]),
                    "drain": np.stack([none, dr1]),
                    "receive": np.stack([none, rc1]),
                    "turn": int(battle.turn),
                    "battle": battle.battle_tag,
                }
            )
        except Exception:
            self.teacher_errors += 1
        return action


def save_shard(path: Path, records: list[dict]) -> int:
    if not records:
        return 0
    np.savez_compressed(
        path,
        obs=np.stack([r["obs"] for r in records]),
        mask=np.stack([r["mask"] for r in records]),
        played=np.stack([r["played"] for r in records]),
        useless=np.stack([r["useless"] for r in records]),
        values=np.stack([r["values"] for r in records]),
        doomed=np.stack([r["doomed"] for r in records]),
        protect=np.stack([r["protect"] for r in records]),
        wasted=np.stack([r["wasted"] for r in records]),
        drain=np.stack([r["drain"] for r in records]),
        receive=np.stack([r["receive"] for r in records]),
        turn=np.array([r["turn"] for r in records], dtype=np.int16),
        battle=np.array([r["battle"] for r in records]),
    )
    return len(records)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games-per-cell", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20926)
    ap.add_argument("--port", type=int, default=7630)
    ap.add_argument("--output", type=Path, default=Path("results_tactical1/data"))
    ap.add_argument("--cell-timeout", type=float, default=5400)
    args = ap.parse_args()
    os.chdir(ROOT)
    config = resolve()
    guards = [g for g in config["GUARDS"].split(",") if g]
    rosters = sorted(
        p
        for p in (ROOT / "teams/reg_mc").glob("MC*.txt")
        if roster_split(p.read_text())[1] == "train"
    )
    for name, (path, _) in OPPONENTS.items():
        if path is not None:
            refuse_eval_only_checkpoint(path)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "positions for the tactical teacher (training/tactical_teacher.py)",
        "checkpoint": config["CKPT"],
        "checkpoint_sha256": sha256(ROOT / config["CKPT"]),
        "team": config["TEAM"],
        "team_sha256": sha256(ROOT / config["TEAM"]),
        "preview_model": config["PREVIEW_MODEL"],
        "guards": guards,
        "opponents": {k: v[0] or config["CKPT"] for k, v in OPPONENTS.items()},
        "rosters": "teams/reg_mc MC*.txt, train split only",
        "roster_count": len(rosters),
        "games_per_cell": args.games_per_cell,
        "cells": [
            f"{k}_{'hidden' if h else 'open'}" for k in OPPONENTS for h in (False, True)
        ],
        "seed": args.seed,
        "chunk_obs_len": chunk_obs_len,
        "teacher_sha256": sha256(ROOT / "training/tactical_teacher.py"),
        "guards_sha256": sha256(ROOT / "vgc_bench/src/guards.py"),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    os.environ.pop("VGC_SET_PRIOR_REG", None)
    if config["SET_PRIOR"] != "mc":
        os.environ["VGC_SET_PRIOR_REG"] = config["SET_PRIOR"]
    pokeenv_patches.install()
    torch.set_num_threads(1)
    PolicyPlayer.use_knowledge_obs = PolicyPlayer.use_moveset_prior = True
    PolicyPlayer.use_knowledge_guards = PolicyPlayer.mask_immunities = True
    PolicyPlayer.guard_flags = {g: g in HARD_GUARDS or g in guards for g in GUARDS}
    PolicyPlayer._knowledge_cache = _NoCache()  # type: ignore[assignment]
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    server = ServerConfiguration(
        f"ws://localhost:{args.port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )
    summary = []
    for cell_index, (name, (path, deterministic)) in enumerate(OPPONENTS.items()):
        for hidden in (False, True):
            shard = args.output / f"{name}_{'hidden' if hidden else 'open'}.npz"
            if shard.exists():
                print(f"skip {shard.name} (done)", flush=True)
                continue
            common = dict(
                server_configuration=server,
                battle_format=format_map["mc"],
                log_level=40,
                max_concurrent_battles=8,
                accept_open_team_sheet=not hidden,
                open_timeout=None,
            )
            ours = RecordingPlayer(
                account_configuration=fresh_local_account(),
                deterministic=True,
                team=RandomTeamBuilder(args.seed, 1, "mc", [ROOT / config["TEAM"]]),
                **common,
            )
            ours.set_policy(ROOT / config["CKPT"], torch.device("mps"))
            if config["PREVIEW_MODEL"]:
                ours.preview_model_path = ROOT / config["PREVIEW_MODEL"]
                ours.use_learned_teampreview = True
            foe = StudyOpponent(
                deterministic=deterministic,
                account_configuration=fresh_local_account(),
                team=RandomTeamBuilder(
                    args.seed + cell_index,
                    None,
                    "mc",
                    rosters,
                    sampling_seed=args.seed + 17 * cell_index + int(hidden),
                ),
                **common,
            )
            foe.set_policy(ROOT / (path or config["CKPT"]), torch.device("cpu"))
            started = time.monotonic()
            asyncio.run(
                asyncio.wait_for(
                    ours.battle_against(foe, n_battles=args.games_per_cell),
                    args.cell_timeout,
                )
            )
            errors = {
                k: v for k, v in PolicyPlayer.guard_fire_counts.items() if "error" in k
            }
            wins = sum(1 for b in ours.battles.values() if b.won)
            n = save_shard(shard, ours.records)
            row = {
                "cell": shard.stem,
                "games": len(ours.battles),
                "wins": wins,
                "decisions": n,
                "teacher_errors": ours.teacher_errors,
                "guard_errors": errors,
                "minutes": round((time.monotonic() - started) / 60, 1),
            }
            summary.append(row)
            print(json.dumps(row), flush=True)
            PolicyPlayer.guard_fire_counts.clear()
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
