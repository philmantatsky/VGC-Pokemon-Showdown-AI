"""Scratch (2026-10-03): for our turn-1 first faints whose killer threat_first2 flags,
could one of our leads have Faked the killer out (fake_out_threat.flinches), and was
that Fake Out pair among the brain's ranked candidates (the guard promotes only ranked
pairs)? Live settings (usage prior, Reg M-C). Read-only.
Run from the repo root:
    .venv/bin/python results_analysis/threat_first3_20261003/fake_out_reach.py
"""

from __future__ import annotations

import collections
import json
import logging
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results_analysis/openings_20260927"))
logging.disable(logging.CRITICAL)
os.environ["VGC_MOVESET_PRIOR"] = "1"

from poke_env.data import to_id_str  # noqa: E402
from setswap_audit import battle_lines  # noqa: E402

from tools.ladder_loss_profile import extract_log  # noqa: E402
from unit_tests.ladder_position import position  # noqa: E402
from vgc_bench.src import threat_first2 as T2  # noqa: E402
from vgc_bench.src.fake_out_threat import flinches  # noqa: E402

OUR = "antonius1"
FMT = "gen9championsvgc2026regmc"
ERA = ("T6", "t6m1", "tactical2", "practice1")


def main() -> None:
    counts = collections.Counter()
    for folder in sorted(ROOT.glob("ladder_replays_mc_*")):
        decisions = folder / "decisions.jsonl"
        if not any(k in folder.name for k in ERA) or not decisions.exists():
            continue
        turn1 = {}
        for line in decisions.read_text().splitlines():
            row = json.loads(line) if line.strip() else {}
            m = re.search(r"regmc-(\d+)", row.get("battle", ""))
            if m and row.get("turn") == 1 and row.get("candidates"):
                turn1.setdefault(m.group(1), row)
        for path in sorted(folder.glob("*.html")):
            m = re.search(r"regmc-(\d+)", path.name)
            log = extract_log(path) if m else None
            if not log or m is None or m.group(1) not in turn1:
                continue
            lines = battle_lines(log, f"|player|p2|{OUR}" in log)
            start = next(i for i, ln in enumerate(lines) if ln.startswith("|turn|1"))
            killer = victim = None
            last = None
            for ln in lines[start + 1 :]:
                p = ln.split("|")
                if len(p) > 1 and p[1] == "turn":
                    break
                if len(p) > 4 and p[1] == "move":
                    last = (p[2], p[3])
                if len(p) > 2 and p[1] == "faint":
                    if p[2].startswith("p1") and last and last[0].startswith("p2"):
                        killer, victim = last, p[2]
                    break
            if killer is None:
                continue
            counts["turn-1 first faints (ours)"] += 1
            try:
                battle = position(lines[: start + 1])
                battle._format = FMT
                found = T2.threats(battle)
            except Exception as exc:
                counts[f"error {type(exc).__name__}"] += 1
                continue
            vslot = 0 if victim.startswith("p1a") else 1
            name = to_id_str(killer[0].split(": ", 1)[1])
            hit = [
                t
                for t in found
                if t[2] == vslot and to_id_str(t[1].species).startswith(name[:5])
            ]
            if not hit:
                continue
            counts["  killer flagged"] += 1
            foe = hit[0][1]
            fslot = next(
                i for i, f in enumerate(battle.opponent_active_pokemon, 1) if f is foe
            )
            users = [
                pos
                for pos, mon in enumerate(battle.active_pokemon)
                if mon is not None
                and "fakeout" in (mon.moves or {})
                and flinches(battle, mon, foe)
            ]
            if not users:
                counts["    no lead can Fake Out the killer"] += 1
                continue
            counts["    a lead can Fake Out the killer"] += 1
            ranked = False
            for c in turn1[m.group(1)]["candidates"]:
                for pos in users:
                    order = c["orders"][pos]
                    if order.get("id") == "fakeout" and order.get("target") == fslot:
                        ranked = True
            chosen = turn1[m.group(1)]["chosen"]["orders"]
            played = any(
                chosen[p].get("id") == "fakeout" and chosen[p].get("target") == fslot
                for p in users
            )
            counts["      that Fake Out was played"] += played
            counts["      ranked but not played"] += ranked and not played
            counts["      not even ranked"] += not ranked
    for k, v in counts.items():
        print(f"{k:45s} {v}")


if __name__ == "__main__":
    main()
