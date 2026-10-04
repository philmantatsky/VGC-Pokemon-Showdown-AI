"""Scratch (2026-10-03): where wasted_fake_out and throat_chop_main_threat would
have changed the bot's choice on ladder (every logged T6-era decision rebuilt from
its replay with the logged candidates; usage prior on, Reg M-C), with the pick
before and after. Rare guards are judged by correctness, not win rate.
Run from the repo root:
    .venv/bin/python results_analysis/review1003/rare_guard_audit.py
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
sys.path.insert(0, str(ROOT / "results_analysis/threat_first_20261003"))
logging.disable(logging.CRITICAL)
os.environ["VGC_MOVESET_PRIOR"] = "1"

from setswap_audit import battle_lines  # noqa: E402

from tools.ladder_loss_profile import extract_log  # noqa: E402
from unit_tests.ladder_position import position  # noqa: E402
from vgc_bench.src import game_review_1003 as R  # noqa: E402
from vgc_bench.src import guards as G  # noqa: E402

OUR = "antonius1"
ERA = ("T6", "t6m1", "tactical2", "practice1", "challenge")
GUARDS = {
    "wasted_fake_out": R.guard_wasted_fake_out,
    "throat_chop_main_threat": R.guard_throat_chop_main_threat,
}


def label(battle, action, pos):
    order = G._decode(battle, action, pos)
    move = getattr(order, "order", None)
    if move is None:
        return "?"
    target = getattr(order, "move_target", 0)
    name = getattr(move, "id", None) or getattr(move, "species", "?")
    foes = battle.opponent_active_pokemon
    aim = ""
    if target in (1, 2) and foes[target - 1] is not None:
        aim = ">" + foes[target - 1].species
    return f"{name}{aim}"


def main() -> None:
    decisions, fired = 0, collections.defaultdict(list)
    folders = sorted(ROOT.glob("ladder_replays_mc_*")) + sorted(
        ROOT.glob("challenge_replays_mc_*")
    )
    for folder in folders:
        log_path = folder / "decisions.jsonl"
        if not any(k in folder.name for k in ERA) or not log_path.exists():
            continue
        rows = collections.defaultdict(dict)
        for line in log_path.read_text().splitlines():
            row = json.loads(line) if line.strip() else {}
            m = re.search(r"regmc-(\d+)", row.get("battle", ""))
            if m and row.get("candidates") and row.get("turn"):
                rows[m.group(1)].setdefault(int(row["turn"]), row)
        for path in sorted(folder.glob("*.html")):
            m = re.search(r"regmc-(\d+)", path.name)
            log = extract_log(path) if m else None
            if not log or m is None or m.group(1) not in rows:
                continue
            lines = battle_lines(log, f"|player|p2|{OUR}" in log)
            turn_at = {
                int(ln.split("|")[2]): i
                for i, ln in enumerate(lines)
                if ln.startswith("|turn|")
            }
            for turn, row in sorted(rows[m.group(1)].items()):
                if turn not in turn_at:
                    continue
                try:
                    battle = position(lines[: turn_at[turn] + 1])
                    battle._format = "gen9championsvgc2026regmc"
                    cands = [
                        G.Candidate(
                            actions=tuple(c["actions"]), prob=c["policy_probability"]
                        )
                        for c in row["candidates"]
                        if not c.get("demoted_by")
                    ]
                    if len(cands) < 2:
                        continue
                    decisions += 1
                    for name, guard in GUARDS.items():
                        out = guard(battle, list(cands), G.GuardReport())
                        if out[0] is not cands[0]:
                            fired[name].append(
                                (
                                    folder.name[-28:],
                                    m.group(1),
                                    turn,
                                    [
                                        label(battle, a, p)
                                        for p, a in enumerate(cands[0].actions)
                                    ],
                                    [
                                        label(battle, a, p)
                                        for p, a in enumerate(out[0].actions)
                                    ],
                                )
                            )
                except Exception:
                    continue
    print(f"decisions replayed: {decisions}")
    for name, rows_ in fired.items():
        print(f"== {name}: changed {len(rows_)}")
        for row in rows_:
            print("   ", row)


if __name__ == "__main__":
    main()
