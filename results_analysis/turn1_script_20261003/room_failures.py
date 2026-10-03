"""Scratch (2026-10-03): in our Blastoise + Farigiraf leads, why our turn-1 Trick
Room did not go up, by Blastoise's turn-1 move. Reasons read from the turn-1 log:
Farigiraf flinched / was taunted / fainted before moving / did not click Trick Room /
Imprison blocked it / their own Trick Room reversed ours. Free wins excluded.
Run from the repo root:
    .venv/bin/python results_analysis/turn1_script_20261003/room_failures.py
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results_analysis/openings_20260927"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ab_read import quit_turn  # noqa: E402
from openings_parse import base, parse, sid  # noqa: E402
from turn1_history import ERA, OUR, WATER  # noqa: E402

from tools.ladder_loss_profile import extract_log  # noqa: E402


def turn1_lines(log: str) -> list[list[str]]:
    out, turn = [], 0
    for line in log.split("\n"):
        parts = line.split("|")
        if len(parts) > 2 and parts[1] == "turn":
            turn = int(parts[2])
            if turn > 1:
                break
            continue
        if turn == 1 and len(parts) > 1:
            out.append(parts)
    return out


def reason(log: str, side: str) -> str:
    """Why our turn-1 Trick Room did not stay up (or 'up')."""
    lines = turn1_lines(log)
    fari = None  # our Farigiraf's slot, from its lead switch-in
    for line in log.split("\n"):
        if line.startswith("|turn|"):
            break
        p = line.split("|")
        if len(p) > 3 and p[1] == "switch" and p[2].startswith(side):
            if p[3].startswith("Farigiraf"):
                fari = p[2][:3]
    ours_up = any(
        p[1] == "-fieldstart"
        and "Trick Room" in "|".join(p)
        and f"[of] {side}" in "|".join(p)
        for p in lines
    )
    ended = any(p[1] == "-fieldend" and "Trick Room" in "|".join(p) for p in lines)
    if ours_up and not ended:
        return "up"
    if ours_up and ended:
        return "theirs reversed ours"
    for p in lines:
        joined = "|".join(p)
        if fari and len(p) > 2 and p[2].startswith(fari):
            if p[1] == "cant":
                return f"Farigiraf could not move: {p[3]}"
            if p[1] == "faint":
                return "Farigiraf fainted before moving"
            if p[1] == "move":
                if p[3] != "Trick Room":
                    return f"Farigiraf used {p[3]}"
                if "[still]" in joined or "[notarget]" in joined:
                    return "Trick Room failed"
        if p[1] == "-fail" and fari and p[2].startswith(fari):
            return "Trick Room failed"
    if any(p[1] == "-fieldstart" and "Trick Room" in "|".join(p) for p in lines):
        return "their Trick Room (ours not set)"
    return "other"


def main() -> None:
    table: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for folder in sorted(ROOT.glob("ladder_replays_mc_*")):
        if not folder.is_dir() or not any(k in folder.name for k in ERA):
            continue
        for path in folder.glob("*.html"):
            log = extract_log(path) or ""
            rec = parse(log) if log else None
            if rec is None:
                continue
            side = next(
                (s for s, n in rec["players"].items() if sid(n) == sid(OUR)), None
            )
            if side is None or {base(s) for s in rec["leads"][side]} != WATER:
                continue
            other = "p2" if side == "p1" else "p1"
            q = quit_turn(log, rec["players"].get(other, ""))
            if q is not None and q <= 1:
                continue
            moves = {base(a["user"]): a["move"] for a in rec["actions"]["1"][side]}
            move = moves.get("Blastoise", "(none)")
            group = move if move in ("Fake Out", "Water Spout") else "other"
            table[group][reason(log, side)] += 1
    for group, counts in table.items():
        n = sum(counts.values())
        print(f"== Blastoise turn 1: {group} ({n} games)")
        for why, c in counts.most_common():
            print(f"   {why:40s} {c:3d} ({100 * c / n:.0f}%)")


if __name__ == "__main__":
    main()
