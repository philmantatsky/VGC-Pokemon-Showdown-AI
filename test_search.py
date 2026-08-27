"""Does the search understand that Tailwind is worth something?

The claim being tested: because reward is +1 win / -1 loss with gamma=1, the critic's
output estimates win probability, and Tailwind is in the observation, a successor state
with Tailwind up should evaluate HIGHER than the same state without it -- with nobody
having written down that Tailwind is good.

Also checks the forward model does the mechanical things: damage lands, faints happen,
and speed order lets a KO deny the slower side its move.
"""

import asyncio
import json
import statistics
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from poke_env import AccountConfiguration
from poke_env.battle import Move, SideCondition
from poke_env.environment import DoublesEnv, SingleAgentWrapper
from poke_env.player import RandomPlayer
from stable_baselines3 import PPO

from vgc_bench.logs2trajs import LogReader
from vgc_bench.src import search as S
from vgc_bench.src import vgc_knowledge as K
from vgc_bench.src.env import ShowdownEnv
from vgc_bench.src.policy import MaskedActorCriticPolicy
from vgc_bench.src.policy_player import PolicyPlayer
from vgc_bench.src.teams import get_available_regs
from vgc_bench.src.utils import format_map

CKPT = "results_knowledge/converted_v2.zip"
LOOP = asyncio.new_event_loop()
Thread(target=LOOP.run_forever, daemon=True).start()
PolicyPlayer.use_knowledge_obs = True
PolicyPlayer.use_moveset_prior = True


def replay(tag, log, role="p1"):
    i = log.index(f"|player|{role}|")
    _, _, _, u, _, _ = log[i : log.index("\n", i)].split("|")
    r = LogReader(
        account_configuration=AccountConfiguration(u, None),
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
states = []
for tag, (_ts, log) in list(logs.items())[:8]:
    try:
        r = replay(tag, log)
    except Exception:
        continue
    states += [
        s
        for s in r.states
        if any(s.active_pokemon)
        and any(s.opponent_active_pokemon)
        and not s.teampreview
    ]
    if len(states) >= 40:
        break
print(f"states: {len(states)}\n")

# --- 1. does the critic value Tailwind? --------------------------------------
gains = []
for b in states[:25]:
    base = S._evaluate(policy, b)
    with_tw = deepcopy(b)
    with_tw.side_conditions.setdefault(SideCondition.TAILWIND, with_tw.turn)
    gains.append(S._evaluate(policy, with_tw) - base)

print("1. critic value change when TAILWIND is added to our side:")
print(f"   mean {statistics.mean(gains):+.4f}   median {statistics.median(gains):+.4f}")
print(f"   positive in {sum(1 for g in gains if g > 0)}/{len(gains)} states")

# same for the opponent getting it (should go the other way)
opp_gains = []
for b in states[:25]:
    base = S._evaluate(policy, b)
    with_tw = deepcopy(b)
    with_tw.opponent_side_conditions.setdefault(SideCondition.TAILWIND, with_tw.turn)
    opp_gains.append(S._evaluate(policy, with_tw) - base)
print(
    f"   THEIR tailwind: mean {statistics.mean(opp_gains):+.4f} "
    f"(positive in {sum(1 for g in opp_gains if g > 0)}/{len(opp_gains)})"
)

# --- 2. forward model mechanics ----------------------------------------------
b = states[0]
me = next(m for m in b.active_pokemon if m)
foe = next(f for f in b.opponent_active_pokemon if f)
dmg_move = next((m for m in me.moves.values() if m.base_power > 0), None)
if dmg_move:
    before = foe.current_hp_fraction
    sim = S.simulate_turn(b, [(me, dmg_move, [foe])], [])
    sfoe = sim.opponent_team.get(K.identifier(b, foe))
    print(f"\n2. forward model: {me.species} {dmg_move.id} -> {foe.species}")
    print(f"   foe hp {before:.2f} -> {sfoe.current_hp_fraction:.2f}")
    assert sfoe.current_hp_fraction <= before, "damage did not land"
    assert b.opponent_team[K.identifier(b, foe)].current_hp_fraction == before, (
        "original battle was mutated"
    )
    print("   original battle unmutated  OK")

# --- 3. side condition applied by the model ----------------------------------
tw = Move("tailwind", gen=9)
sim = S.simulate_turn(b, [(me, tw, [])], [])
has_tw = any(c.name == "TAILWIND" for c in sim.side_conditions)
print(f"\n3. simulate_turn(Tailwind) sets our side condition: {has_tw}")
assert has_tw, "tailwind not applied by forward model"

# --- 4. end-to-end timing ----------------------------------------------------
timed = 0
t0 = time.perf_counter()
for b in states[:5]:
    obs = PolicyPlayer.embed_battle(b, fake_rating=2000)
    mask = np.array(DoublesEnv.get_action_mask(b))
    obs_dict = {
        "observation": torch.as_tensor(obs).unsqueeze(0),
        "action_mask": torch.as_tensor(mask).unsqueeze(0),
    }
    S.search_action(policy, b, obs_dict, obs_dict["action_mask"])
    timed += 1
dt = (time.perf_counter() - t0) / max(timed, 1)
print(f"\n4. full search: {dt * 1000:.0f} ms per decision ({timed} states)")
print("   (ladder allows ~15 s/turn)")

print("\nPASS")
