"""Measure what computing knowledge features costs per observation.

Self-play runs at ~230 steps/sec with embed_battle called twice per turn per env.
If knowledge computation dominates that, it has to be scoped down (actives only) or
memoised before it goes anywhere near a training run.
"""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import asyncio
import json
import time
from pathlib import Path
from threading import Thread

from poke_env import AccountConfiguration
from poke_env.calc import calculate_damage

from vgc_bench.logs2trajs import LogReader
from vgc_bench.src import vgc_knowledge as K
from vgc_bench.src.policy_player import PolicyPlayer

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
for tag, (_ts, log) in list(logs.items())[:12]:
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
    if len(states) >= 120:
        break
states = states[:120]
print(f"benchmarking on {len(states)} real battle states\n")


def timeit(fn, n=3):
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best / len(states)


# 1. a single damage calc
b0 = states[0]
me = next(m for m in b0.active_pokemon if m)
foe = next(m for m in b0.opponent_active_pokemon if m)
mv = next(iter(me.moves.values()))
mk, fk = K.identifier(b0, me), K.identifier(b0, foe)
t0 = time.perf_counter()
for _ in range(2000):
    calculate_damage(mk, fk, mv, b0)
one_calc = (time.perf_counter() - t0) / 2000
print(f"1. single calculate_damage      : {one_calc * 1e6:8.1f} us")

# 2. embed_battle as it stands (knowledge = zeros)
base = timeit(lambda: [PolicyPlayer.embed_battle(s, fake_rating=2000) for s in states])
print(f"2. embed_battle (zeros)         : {base * 1e6:8.1f} us")


# 3. knowledge for the 4 actives only
def knowledge_actives():
    for s in states:
        for m in s.active_pokemon:
            if m is not None:
                K.pokemon_knowledge(s, m, True)


act = timeit(knowledge_actives)
print(f"3. knowledge, 4 actives         : {act * 1e6:8.1f} us")


# 4. knowledge for all 12 tokens (the naive version)
def knowledge_all():
    for s in states:
        for m in list(s.team.values())[:6]:
            K.pokemon_knowledge(s, m, True)
        for m in list(s.opponent_team.values())[:6]:
            K.pokemon_knowledge(s, m, False)


allt = timeit(knowledge_all)
print(f"4. knowledge, all 12 tokens     : {allt * 1e6:8.1f} us")

print()
print(f"overhead if actives-only : {act / base * 100:6.0f}% of current embed cost")
print(f"overhead if all 12 tokens: {allt / base * 100:6.0f}% of current embed cost")
fps = 230
print(f"\nat {fps} steps/sec, ~2 embeds/step:")
for label, cost in (("actives-only", act), ("all 12", allt)):
    frac = cost * 2 * fps
    print(
        f"  {label:13}: {frac * 100:5.1f}% of wall time -> "
        f"~{fps / (1 + frac):.0f} steps/sec ({(1 - 1 / (1 + frac)) * 100:.0f}% slower)"
    )
