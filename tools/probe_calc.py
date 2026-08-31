"""Probe calculate_damage on a real replayed DoubleBattle before building on it.

Checks the three things the knowledge layer depends on: identifier format, whether
opponent stats are usable, and how far off they are (Champions EVs cap at 32/stat,
66 total -- a much smaller correction than the standard 252/508 scale).

Uses the same LogReader construction as logs2trajs.process_log.
"""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import asyncio
import json
from pathlib import Path
from threading import Thread

from poke_env import AccountConfiguration
from poke_env.calc import calculate_damage

from vgc_bench.logs2trajs import LogReader

LOOP = asyncio.new_event_loop()
Thread(target=LOOP.run_forever, daemon=True).start()


def replay(tag, log, role="p1"):
    i = log.index(f"|player|{role}|")
    _, _, _, username, _, _rating = log[i : log.index("\n", i)].split("|")
    reader = LogReader(
        account_configuration=AccountConfiguration(username, None),
        battle_format=tag.split("-")[0],
        log_level=51,
        accept_open_team_sheet=True,
        loop=LOOP,
    )
    asyncio.run_coroutine_threadsafe(reader.follow_log(tag, log), LOOP).result()
    return reader


def key_of(mon, d):
    return next((k for k, v in d.items() if v is mon), None)


def main():
    logs = json.load(Path("battle_logs/logs_gen9championsvgc2026regmb.json").open())
    for tag, (_ts, log) in list(logs.items())[:6]:
        try:
            reader = replay(tag, log)
        except Exception as e:
            print(f"{tag}: replay failed ({type(e).__name__})")
            continue

        usable = [
            s
            for s in reader.states
            if any(s.active_pokemon) and any(s.opponent_active_pokemon)
        ]
        if not usable:
            continue
        b = usable[len(usable) // 2]
        me = next(m for m in b.active_pokemon if m)
        foe = next(m for m in b.opponent_active_pokemon if m)
        mk, fk = key_of(me, b.team), key_of(foe, b.opponent_team)

        print(f"\n=== {tag} (turn {b.turn}) ===")
        print(f"attacker key {mk!r}  defender key {fk!r}")
        print(f"foe stats {foe.stats}")
        print(f"foe base  {foe.base_stats}")
        hp = (foe.stats or {}).get("hp")
        for mv in list(me.moves.values())[:4]:
            try:
                lo, hi = calculate_damage(mk, fk, mv, b)
                frac = f"{lo / hp:.0%}-{hi / hp:.0%}" if hp else "?"
                print(f"  {mv.id:16} {str(mv.type):24} {lo:>4}-{hi:<4} ({frac})")
            except Exception as e:
                print(f"  {mv.id:16} FAILED {type(e).__name__}: {e}")
        return  # one good battle is enough for the probe


if __name__ == "__main__":
    main()


def debug():
    """Why did an 80-BP Grass move return 0-0?"""
    logs = json.load(Path("battle_logs/logs_gen9championsvgc2026regmb.json").open())
    tag, (_ts, log) = next(iter(logs.items()))
    reader = replay(tag, log)
    usable = [
        s
        for s in reader.states
        if any(s.active_pokemon) and any(s.opponent_active_pokemon)
    ]
    b = usable[len(usable) // 2]
    me = next(m for m in b.active_pokemon if m)
    foe = next(m for m in b.opponent_active_pokemon if m)
    mv = me.moves.get("matchagotcha")
    print("attacker :", me.species, "stats:", me.stats)
    print("  fainted:", me.fainted, " hp_frac:", me.current_hp_fraction)
    print("defender :", foe.species, "types:", foe.types)
    print("  effects:", list(foe.effects))
    print("  hp_frac:", foe.current_hp_fraction, "fainted:", foe.fainted)
    print(
        "move     :",
        mv.id if mv else None,
        "bp:",
        mv.base_power if mv else None,
        "cat:",
        mv.category if mv else None,
        "target:",
        mv.target if mv else None,
    )
    print("battle player_role:", b.player_role, " opp_role:", b.opponent_role)
