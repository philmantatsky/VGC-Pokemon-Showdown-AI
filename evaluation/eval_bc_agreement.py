"""Held-out human-action agreement for a behavior-cloned policy.

Fidelity check for the human-imitation arms: score a policy's agreement with
real human decisions on trajectories it never trained on (the opposite crc32
bucket corpus). The shallow move predictor's 37% top-1 / 77% top-3 is the
baseline to beat; if a BC policy lands far below ~50% top-3 it is a weak style
probe and the gate documentation must say so (it still functions as an
out-of-distribution opponent, which is the property the gates actually need).

    .venv/bin/python eval_bc_agreement.py \
        --policy results_bc/mix_A/saves_bc/seed1/30.zip \
        --trajs_dir trajs_regmb_human_B --sample 800
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import torch

from vgc_bench.src.utils import act_len


def load_policy(path: str, device: str):
    from poke_env.environment import SingleAgentWrapper
    from poke_env.player import RandomPlayer
    from stable_baselines3 import PPO

    from vgc_bench.src.env import ShowdownEnv
    from vgc_bench.src.policy import MaskedActorCriticPolicy
    from vgc_bench.src.teams import get_available_regs
    from vgc_bench.src.utils import format_map

    env = ShowdownEnv(
        battle_format=format_map[get_available_regs()[0]],
        log_level=40,
        accept_open_team_sheet=True,
        start_listening=False,
        choose_on_teampreview=True,
    )
    saw = SingleAgentWrapper(env, RandomPlayer(start_listening=False))
    ppo = PPO(
        MaskedActorCriticPolicy,
        saw,
        policy_kwargs={"d_model": 256, "choose_on_teampreview": True},
        device=device,
    )
    ppo.set_parameters(path, device=ppo.device)
    ppo.policy.eval()
    return ppo.policy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--trajs_dir", required=True)
    ap.add_argument("--sample", type=int, default=800, help="trajectories to score")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    policy = load_policy(args.policy, args.device)
    files = sorted(Path(args.trajs_dir).glob("*.pkl"))
    random.Random(args.seed).shuffle(files)
    files = files[: args.sample]

    top1_hits = top3_hits = slot_hits = total = slots = 0
    opening_top1 = opening_total = 0
    with torch.no_grad():
        for file in files:
            traj = pickle.load(file.open("rb"))
            obs = np.asarray(traj.obs, dtype=np.float32)
            acts = np.asarray(traj.acts)
            steps = acts.shape[0]
            if steps == 0:
                continue
            mask = np.ones((steps, 2 * act_len), dtype=np.float32)
            mask[:, 47:87] = 0
            mask[:, act_len + 47 : act_len + 87] = 0
            obs_dict = {
                "observation": torch.as_tensor(obs[:steps], device=policy.device),
                "action_mask": torch.as_tensor(mask, device=policy.device),
            }
            dist = policy.get_distribution(obs_dict)
            # MultiDiscrete: one categorical per slot
            probs = [d.probs for d in dist.distribution]
            for step in range(steps):
                human = acts[step]
                per_slot_ok = []
                joint_rank_ok_top3 = True
                for slot in (0, 1):
                    p = probs[slot][step]
                    human_action = int(human[slot])
                    argmax = int(torch.argmax(p).item())
                    per_slot_ok.append(argmax == human_action)
                    top3 = torch.topk(p, k=min(3, p.shape[-1])).indices.tolist()
                    if human_action not in top3:
                        joint_rank_ok_top3 = False
                total += 1
                slots += 2
                slot_hits += sum(per_slot_ok)
                if all(per_slot_ok):
                    top1_hits += 1
                if joint_rank_ok_top3:
                    top3_hits += 1
                if step < 2:
                    opening_total += 1
                    if all(per_slot_ok):
                        opening_top1 += 1

    print(f"policy   : {args.policy}")
    print(f"held-out : {args.trajs_dir} ({len(files)} trajectories, {total} decisions)")
    print(f"top-1 joint agreement : {top1_hits / total * 100:5.1f}%")
    print(f"top-3 per-slot both   : {top3_hits / total * 100:5.1f}%")
    print(f"top-1 per-slot        : {slot_hits / slots * 100:5.1f}%")
    if opening_total:
        print(
            f"top-1 joint, preview steps (first 2): "
            f"{opening_top1 / opening_total * 100:5.1f}%"
        )


if __name__ == "__main__":
    main()
