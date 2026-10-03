"""Scratch (2026-10-03): for each of our turn 1-3 first faints on ladder, would a
switch have saved the Pokemon? At the start of that turn, the move that knocked it
out (attacker, move) is scored by the damage calculator against each of our benched
Pokemon: a switch-in that survives it (and is not knocked out by it from its
current HP) is a saving switch -- if the attacker would still have used that move
into that slot. Read-only.
Run from the repo root:
    .venv/bin/python results_analysis/early_faints_20261003/switch_counterfactual.py
"""

from __future__ import annotations

import collections
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results_analysis/openings_20260927"))
logging.disable(logging.CRITICAL)

from poke_env.battle import Move  # noqa: E402
from poke_env.data import to_id_str  # noqa: E402
from setswap_audit import battle_lines  # noqa: E402

from tools.ladder_loss_profile import extract_log  # noqa: E402
from unit_tests.ladder_position import position  # noqa: E402
from vgc_bench.src import vgc_knowledge as K  # noqa: E402

OUR = "antonius1"
DIRS = [
    "deployed_T6_humanpreview1_attackcheck",
    "deployed_T6_humanpreview1_throatchop",
    "deployed_T6_humanpreview1_wideguard",
    "deployed_T6ctx",
    "deployed_T6tac",
    "deployed_T6tac_playbook_t6",
    "practice1_water_t6",
    "tactical2_e4",
    "T6tac_review_guards",
    "t6m1",
]
rows, saved_by = [], collections.Counter()
for d in DIRS:
    for f in sorted((ROOT / f"ladder_replays_mc_{d}").glob("*.html")):
        log = extract_log(f)
        if not log:
            continue
        lines = battle_lines(log, f"|player|p2|{OUR}" in log)
        turn, start, last = 0, {}, {}
        first = None
        for i, line in enumerate(lines):
            p = line.split("|")
            if p[1] == "turn":
                turn = int(p[2])
                start[turn] = i
            elif p[1] == "move":
                last["move"] = (p[2][:3], p[3])
            elif p[1] == "-damage" and p[2].startswith("p1") and "[from]" not in line:
                last[p[2][:3]] = last.get("move")
            elif p[1] == "faint":
                if p[2].startswith("p1") and turn <= 3:
                    first = (turn, p[2][:3], last.get(p[2][:3]))
                break
        if not first or not first[2] or not first[2][0].startswith("p2"):
            continue
        turn, slot, (foe_slot, move_name) = first
        try:
            battle = position(lines[: start[turn] + 1])
        except Exception:
            continue
        idx = 0 if slot.endswith("a") else 1
        victim = battle.active_pokemon[idx]
        foe = battle.opponent_active_pokemon[0 if foe_slot.endswith("a") else 1]
        if victim is None or foe is None:
            continue
        K.ensure_stats(foe)
        move = Move(to_id_str(move_name), gen=9)
        # only our brought four can switch in: those the replay ever shows on our side
        brought = {
            to_id_str(x.split("|")[3].split(",")[0].split("-Mega")[0])
            for x in lines
            if x.startswith(("|switch|p1", "|drag|p1"))
        }
        bench = [
            m
            for m in battle.team.values()
            if not m.fainted
            and m not in battle.active_pokemon
            and to_id_str(m.species.split("mega")[0]) in brought
        ]
        survivors = []
        for mon in bench:
            fr = K.damage_fraction(battle, foe, mon, move)
            # a benched Pokemon not yet seen in the replay has no HP on record: full
            hp = mon.current_hp_fraction or 1.0
            if fr is not None and fr[1] < hp:
                survivors.append(f"{mon.species} ({100 * fr[1]:.0f}%)")
        rows.append((d, turn, victim.species, foe.species, move_name, survivors))
        saved_by[bool(survivors)] += 1

print(f"turn 1-3 first faints with a known killing move: {len(rows)}")
print(f"   a benched Pokemon would have survived that hit: {saved_by[True]}")
for d, t, v, foe, mv, s in rows:
    print(f"   T{t} {v:10s} <- {foe}:{mv:16s} survivors: {', '.join(s) or '-'}")
