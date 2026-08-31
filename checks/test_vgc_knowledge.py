"""Validate vgc_knowledge against real replayed battle positions."""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import asyncio
import json
from pathlib import Path
from threading import Thread

from poke_env import AccountConfiguration

from vgc_bench.logs2trajs import LogReader
from vgc_bench.src import vgc_knowledge as K

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


logs = json.load(Path("battle_logs/logs_gen9championsvgc2026regmb.json").open())

# --- 1. spread parsing on real Smogon data -----------------------------------
movesets = json.load(Path("data/movesets_regmb.json").open())
ok = sum(1 for v in movesets.values() if K.parse_spread(v.get("spread")))
print(f"1. spreads parsed: {ok}/{len(movesets)}")
assert ok > len(movesets) * 0.9, "spread parser rejecting too much"

# --- 2. identifier + damage on live positions --------------------------------
checked = zero = ko = protected = 0
examples = []
for tag, (_ts, log) in list(logs.items())[:25]:
    try:
        r = replay(tag, log)
    except Exception:
        continue
    for b in r.states:
        actives = [m for m in b.active_pokemon if m]
        foes = [m for m in b.opponent_active_pokemon if m]
        if not actives or not foes:
            continue
        for me in actives:
            assert K.identifier(b, me) is not None, "own mon must resolve"
            for foe in foes:
                fid = K.identifier(b, foe)
                if fid is None:
                    continue
                if K.is_protected(foe):
                    protected += 1
                for mv in list(me.moves.values())[:4]:
                    frac = K.damage_fraction(b, me, foe, mv)
                    if frac is None:
                        continue
                    checked += 1
                    if K.deals_no_damage(b, me, foe, mv):
                        zero += 1
                        if len(examples) < 5:
                            examples.append(
                                f"{me.species} {mv.id} ({mv.type.name}) -> "
                                f"{foe.species} {[t.name for t in foe.types]}"
                            )
                    if K.guaranteed_ko(b, me, foe, mv):
                        ko += 1

print(f"2. damage evaluated on {checked} move/target pairs")
print(f"   genuine immunities found : {zero}")
print(f"   guaranteed KOs found     : {ko}")
print(f"   protected defenders seen : {protected}")
print("   immunity examples:")
for e in examples:
    print(f"     {e}")

# --- 3. stat imputation -------------------------------------------------------
tag, (_ts, log) = next(iter(logs.items()))
r = replay(tag, log)
b = next(s for s in r.states if any(s.opponent_active_pokemon))
foe = next(m for m in b.opponent_active_pokemon if m)
before = dict(foe.stats)
spread = K.parse_spread((movesets.get(foe.base_species) or {}).get("spread"))
changed = K.impute_stats(foe, spread)
print(f"\n3. imputation on {foe.species} (spread={spread})")
print(f"   before {before}")
print(f"   after  {dict(foe.stats)}  changed={changed}")
assert changed, "expected imputation to apply"
assert not K.impute_stats(foe, spread), "must be idempotent"
assert all(foe.stats[k] >= before[k] for k in before), "EVs must not reduce stats"

print("\nPASS")
