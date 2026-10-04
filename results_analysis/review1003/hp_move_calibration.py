"""Scratch (2026-10-03): is hp_move_after_hits' expected HP loss calibrated? Every
logged T6-era ladder decision whose top pair uses Eruption / Water Spout is rebuilt
(unit_tests/ladder_position.py, usage prior on, Reg M-C) and the guard's
expected_loss is compared with what happened in the real turn: the HP our Pokemon
lost before its move resolved (from the log's -damage lines; fainting before moving
counts as all of it). Also how often the guard would have changed the pick.
Run from the repo root:
    .venv/bin/python results_analysis/review1003/hp_move_calibration.py
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

from setswap_audit import battle_lines  # noqa: E402

from tools.ladder_loss_profile import extract_log  # noqa: E402
from unit_tests.ladder_position import position  # noqa: E402
from vgc_bench.src import game_review_1003 as R  # noqa: E402
from vgc_bench.src import guards as G  # noqa: E402

OUR = "antonius1"
ERA = ("T6", "t6m1", "tactical2", "practice1")
NAMES = {"eruption": "Eruption", "waterspout": "Water Spout"}
NARROW = "--narrow" in sys.argv  # the hp_move_after_spread variant


def actual_loss(
    turn_lines: list[str], slot: str, move_name: str, start_hp: float, max_hp: int
):
    """Share of max HP our Pokemon in ``slot`` lost before its HP move resolved:
    all of its HP if it fainted first; None when it did not try the move."""
    hp = None
    for line in turn_lines:
        p = line.split("|")
        if len(p) < 3 or not p[2].startswith(slot):
            continue
        if p[1] == "faint":
            return start_hp
        if p[1] in ("-damage", "-heal") and len(p) > 3:
            value = p[3].split()[0]
            hp = 0 if value == "0" else int(value.split("/")[0])
        if p[1] == "move":
            if p[3] != move_name:
                return None
            return 0.0 if hp is None else max(0.0, start_hp - hp / max_hp)
    return None


def main() -> None:
    rows = []
    for folder in sorted(ROOT.glob("ladder_replays_mc_*")):
        decisions = folder / "decisions.jsonl"
        if not any(k in folder.name for k in ERA) or not decisions.exists():
            continue
        by_game = collections.defaultdict(dict)
        for line in decisions.read_text().splitlines():
            row = json.loads(line) if line.strip() else {}
            m = re.search(r"regmc-(\d+)", row.get("battle", ""))
            if m and row.get("candidates") and row.get("turn"):
                by_game[m.group(1)].setdefault(int(row["turn"]), row)
        for path in sorted(folder.glob("*.html")):
            m = re.search(r"regmc-(\d+)", path.name)
            log = extract_log(path) if m else None
            if not log or m is None or m.group(1) not in by_game:
                continue
            lines = battle_lines(log, f"|player|p2|{OUR}" in log)
            turn_at = {
                int(ln.split("|")[2]): i
                for i, ln in enumerate(lines)
                if ln.startswith("|turn|")
            }
            for turn, row in sorted(by_game[m.group(1)].items()):
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
                    top = cands[0]
                    for pos in (0, 1):
                        order = G._decode(battle, top.actions[pos], pos)
                        move = getattr(order, "order", None)
                        if getattr(move, "id", None) not in R.HP_MOVES:
                            continue
                        me = battle.active_pokemon[pos]
                        predicted = R.expected_loss(battle, me, move, NARROW)
                        guard = (
                            R.guard_hp_move_after_spread
                            if NARROW
                            else R.guard_hp_move_after_hits
                        )
                        out = guard(battle, cands, G.GuardReport())
                        changed = out[0] is not top
                        end = turn_at.get(turn + 1, len(lines))
                        slot = f"p1{'ab'[pos]}"
                        actual = actual_loss(
                            lines[turn_at[turn] + 1 : end],
                            slot,
                            NAMES.get(move.id, move.id),
                            float(me.current_hp_fraction or 0),
                            me.max_hp,
                        )
                        hindsight = None
                        if changed and actual is not None:
                            already = G._partner_damage(battle, top.actions, pos)
                            alt = G._decode(battle, out[0].actions[pos], pos)
                            left = float(me.current_hp_fraction or 0) - actual
                            if left > 0:
                                with R.at_hp(me, left):
                                    kept = G._attack_value(
                                        battle, me, order, pos, already
                                    )
                                    swapped = G._attack_value(
                                        battle, me, alt, pos, already
                                    )
                                if kept is not None and swapped is not None:
                                    hindsight = swapped >= kept
                            else:
                                hindsight = True  # it fainted first: nothing was lost
                        rows.append(
                            dict(
                                game=m.group(1),
                                turn=turn,
                                mon=me.species,
                                predicted=predicted,
                                actual=actual,
                                changed=changed,
                                hindsight=hindsight,
                            )
                        )
                except Exception:
                    continue
    known = [r for r in rows if r["actual"] is not None]
    print(
        f"HP-move decisions: {len(rows)}, the move happened/was attempted: {len(known)}"
    )
    print(
        f"guard would change the pick: {sum(r['changed'] for r in rows)} of {len(rows)}"
    )
    for label, group in (
        ("all", known),
        ("guard would change", [r for r in known if r["changed"]]),
        ("guard keeps", [r for r in known if not r["changed"]]),
    ):
        if not group:
            continue
        p = sum(r["predicted"] for r in group) / len(group)
        a = sum(r["actual"] for r in group) / len(group)
        print(
            f"  {label:20s} n={len(group):3d}  "
            f"predicted loss {p:.2f}  actual loss {a:.2f}"
        )
    judged = [r for r in rows if r.get("hindsight") is not None]
    right = sum(r["hindsight"] for r in judged)
    print(
        "swaps right in hindsight (at the real HP when the move landed): "
        f"{right} of {len(judged)}"
    )
    print("by predicted loss:")
    for lo, hi in ((0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 2)):
        group = [r for r in known if lo <= r["predicted"] < hi]
        if group:
            a = sum(r["actual"] for r in group) / len(group)
            print(f"  predicted {lo:.2f}-{hi:.2f}: n={len(group):3d} actual {a:.2f}")


if __name__ == "__main__":
    main()
