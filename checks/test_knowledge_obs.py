"""Verify the knowledge observation: populates, memoises, and is off by default."""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import asyncio
import json
import time
from pathlib import Path
from threading import Thread

import numpy as np
from poke_env import AccountConfiguration

from vgc_bench.logs2trajs import LogReader
from vgc_bench.src.policy_player import PolicyPlayer
from vgc_bench.src.utils import (
    chunk_obs_len,
    correct_accuracy_obs_len,
    global_presence_obs_len,
    knowledge_obs_len,
    presence_obs_len,
    semantics_obs_len,
)

LOOP = asyncio.new_event_loop()
Thread(target=LOOP.run_forever, daemon=True).start()


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


logs = json.load(Path("battle_logs/logs_gen9championsvgc2026regmb.json").open())
states = []
for tag, (_ts, log) in list(logs.items())[:10]:
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
    if len(states) >= 60:
        break
states = states[:60]
print(f"states: {len(states)}")

# Token layout tail: ... | knowledge (gated) | semantics (always on)
KSTART = (
    chunk_obs_len
    - knowledge_obs_len
    - semantics_obs_len
    - presence_obs_len
    - global_presence_obs_len
    - correct_accuracy_obs_len
)
KEND = KSTART + knowledge_obs_len
SEND = KEND + semantics_obs_len
OLD = KSTART

# --- 1. off by default: knowledge block must be all zeros ---------------------
PolicyPlayer.use_knowledge_obs = False
PolicyPlayer._knowledge_cache.clear()
off = PolicyPlayer.embed_battle(states[0], fake_rating=2000)
tok_off = off.reshape(12, chunk_obs_len)
assert off.shape == (12 * chunk_obs_len,), off.shape
assert np.count_nonzero(tok_off[:, KSTART:KEND]) == 0, (
    "knowledge should be zero when off"
)
sem_nz = np.count_nonzero(tok_off[:, KEND:SEND])
print(
    f"1. flag OFF: obs {off.shape[0]}, knowledge zero, "
    f"semantics populated ({sem_nz} nonzero)  OK"
)
assert sem_nz > 0, "semantics should always be populated"

# --- 2. on: our actives get populated -----------------------------------------
PolicyPlayer.use_knowledge_obs = True
PolicyPlayer._knowledge_cache.clear()
populated = 0
for s in states:
    o = PolicyPlayer.embed_battle(s, fake_rating=2000).reshape(12, chunk_obs_len)
    populated += int(np.count_nonzero(o[:6, KSTART:KEND]) > 0)
print(f"2. flag ON : {populated}/{len(states)} states have populated knowledge")
assert populated > len(states) * 0.5, "knowledge rarely populating - check wiring"

o = PolicyPlayer.embed_battle(states[0], fake_rating=2000).reshape(12, chunk_obs_len)
nz = [i for i in range(6) if np.count_nonzero(o[i, KSTART:KEND])]
if nz:
    blk = o[nz[0], KSTART:KEND]
    print(f"   sample active token knowledge: {np.round(blk, 3).tolist()}")

# --- 3. opponent tokens stay zero (we only compute our actives) ---------------
assert np.count_nonzero(o[6:, KSTART:KEND]) == 0, (
    "opponent knowledge should not be populated"
)
print("3. opponent tokens: zero as intended  OK")

# --- 4. memoisation actually saves work ---------------------------------------
PolicyPlayer._knowledge_cache.clear()
t0 = time.perf_counter()
for s in states:
    PolicyPlayer.embed_battle(s, fake_rating=2000)
cold = time.perf_counter() - t0
t0 = time.perf_counter()
for s in states:
    PolicyPlayer.embed_battle(s, fake_rating=2000)  # same states -> cache hits
warm = time.perf_counter() - t0
print(
    f"4. cold {cold / len(states) * 1e6:.0f} us/state, "
    f"warm {warm / len(states) * 1e6:.0f} us/state "
    f"-> {(1 - warm / cold) * 100:.0f}% saved on repeat"
)
assert warm < cold, "memoisation not helping"

PolicyPlayer.use_knowledge_obs = False
print("\nPASS")
