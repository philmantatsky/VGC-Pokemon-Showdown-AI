"""Run the guard stack over real replayed positions with a real trained policy.

Answers the question the Phase 0 audit exists for: on positions from actual games,
how often does the policy's top choice do literally nothing, and do the guards catch it?
"""

import asyncio
import json
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from poke_env import AccountConfiguration
from poke_env.environment import DoublesEnv, SingleAgentWrapper
from poke_env.player import RandomPlayer
from stable_baselines3 import PPO

from vgc_bench.logs2trajs import LogReader
from vgc_bench.src import guards as G
from vgc_bench.src.env import ShowdownEnv
from vgc_bench.src.policy import MaskedActorCriticPolicy
from vgc_bench.src.policy_player import PolicyPlayer
from vgc_bench.src.teams import get_available_regs
from vgc_bench.src.utils import format_map

CKPT = "results_mb_ourteam_64opp/saves_bc_sp/reg_mb/64_teams/seed1/3932160.zip"
LOOP = asyncio.new_event_loop()
Thread(target=LOOP.run_forever, daemon=True).start()


def replay(tag, log, role="p1"):
    i = log.index(f"|player|{role}|")
    _, _, _, username, _, _ = log[i : log.index("\n", i)].split("|")
    r = LogReader(
        account_configuration=AccountConfiguration(username, None),
        battle_format=tag.split("-")[0],
        log_level=51,
        accept_open_team_sheet=True,
        loop=LOOP,
    )
    asyncio.run_coroutine_threadsafe(r.follow_log(tag, log), LOOP).result()
    return r


print("loading policy...")
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
    device="cpu",
)
ppo.set_parameters(CKPT, device=torch.device("cpu"))
policy = ppo.policy
policy.eval()

logs = json.load(Path("battle_logs/logs_gen9championsvgc2026regmb.json").open())

positions = blunder_top = blunder_any = fired_zero = fired_ft = fired_ko = 0
examples = []

for tag, (_ts, log) in list(logs.items())[:20]:
    try:
        r = replay(tag, log)
    except Exception:
        continue
    for b in r.states:
        if not any(b.active_pokemon) or not any(b.opponent_active_pokemon):
            continue
        if b.teampreview:
            continue
        try:
            obs = PolicyPlayer.embed_battle(b, fake_rating=2000)
            mask = np.array(DoublesEnv.get_action_mask(b))
        except Exception:
            continue
        if mask.sum() == 0:
            continue
        obs_dict = {
            "observation": torch.as_tensor(obs).unsqueeze(0),
            "action_mask": torch.as_tensor(mask).unsqueeze(0),
        }
        try:
            cands, _v = G.build_candidates(policy, obs_dict, obs_dict["action_mask"])
        except Exception as e:
            print("build_candidates failed:", type(e).__name__, e)
            break
        if not cands:
            continue
        positions += 1
        before = cands[0].actions
        out, rep = G.apply_guards(b, cands)
        if before in rep.vetoed:
            blunder_top += 1
            if len(examples) < 6:
                me = next((m for m in b.active_pokemon if m), None)
                examples.append(
                    f"turn {b.turn}: {me.species if me else '?'} "
                    f"actions {before} demoted by {rep.stages}"
                )
        if rep.vetoed:
            blunder_any += 1
        fired_zero += "zero_damage" in rep.stages
        fired_ft += "first_turn" in rep.stages
        fired_ko += "ko_tiebreak" in rep.stages

print(f"\npositions evaluated        : {positions}")
top_rate = blunder_top / max(positions, 1) * 100
any_rate = blunder_any / max(positions, 1) * 100
print(f"TOP CHOICE was a blunder   : {blunder_top}  ({top_rate:.1f}%)")
print(f"some candidate was a blunder: {blunder_any}  ({any_rate:.1f}%)")
print("\nguard changed the top pick:")
print(f"  zero_damage : {fired_zero}")
print(f"  first_turn  : {fired_ft}")
print(f"  ko_tiebreak : {fired_ko}")
print("\nexamples of blunders caught:")
for e in examples:
    print("  ", e)
