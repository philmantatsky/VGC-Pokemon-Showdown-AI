"""Scratch (2026-10-03): our own ladder history of the Water Room turn 1. In every
T6-era ladder game where our preview led Blastoise + Farigiraf: what Blastoise did
on turn 1, whether our Trick Room went up that turn, whether Blastoise or Farigiraf
fainted on turns 1-2, and the result (free wins excluded). Correlational -- the
brain chose Fake Out in particular positions -- so a prior for the A/B, not a test.
Run from the repo root:
    .venv/bin/python results_analysis/turn1_script_20261003/turn1_history.py
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results_analysis/openings_20260927"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ab_read import QUIT, quit_turn  # noqa: E402
from openings_parse import base, parse, sid  # noqa: E402

from tools.ladder_loss_profile import extract_log  # noqa: E402

OUR = "antonius1"
ERA = ("T6", "t6m1", "tactical2", "practice1")
WATER = {"Blastoise", "Farigiraf"}


def rows() -> list[dict]:
    out = []
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
            if side is None:
                continue
            if {base(s) for s in rec["leads"][side]} != WATER:
                continue
            other = "p2" if side == "p1" else "p1"
            quit_at = quit_turn(log, rec["players"].get(other, ""))
            if quit_at is not None and quit_at <= 1:
                continue  # a free win
            moves = {
                base(a["user"]): a for a in rec["actions"]["1"][side] if not a["still"]
            }
            blastoise = moves.get("Blastoise")
            flinched = [c for c in rec["cant"]["1"][other] if c["reason"] == "flinch"]
            early = {base(s) for turn, s in rec["faints"][side] if turn <= 2}
            out.append(
                {
                    "folder": folder.name.removeprefix("ladder_replays_mc_"),
                    "win": rec["winner_side"] == side,
                    "blastoise_t1": blastoise["move"] if blastoise else "(none)",
                    "room_t1": any(
                        t == 1 and s == side and kind == "start"
                        for t, s, kind in rec["tr"]
                    ),
                    "flinch": bool(flinched),
                    "farigiraf_t12": "Farigiraf" in early,
                    "blastoise_t12": "Blastoise" in early,
                    "quit": bool(QUIT.search(log)),
                }
            )
    return out


def line(label: str, group: list[dict]) -> str:
    n = len(group)
    if not n:
        return f"{label:28s} 0"

    def share(key: str) -> str:
        return f"{100 * sum(r[key] for r in group) / n:3.0f}%"

    wins = sum(r["win"] for r in group)
    return (
        f"{label:28s} {n:3d} games  won {wins:3d}-{n - wins:<3d} ({share('win')})  "
        f"room on t1 {share('room_t1')}  Farigiraf down t1-2 {share('farigiraf_t12')}  "
        f"Blastoise down t1-2 {share('blastoise_t12')}"
    )


def main() -> None:
    data = rows()
    print(line("all Blastoise + Farigiraf", data))
    by_move = collections.defaultdict(list)
    for r in data:
        by_move[r["blastoise_t1"]].append(r)
    for move, group in sorted(by_move.items(), key=lambda kv: -len(kv[1])):
        print(line(f"  Blastoise t1: {move}", group))
    fake_out = by_move.get("Fake Out", [])
    print(line("  Fake Out that flinched", [r for r in fake_out if r["flinch"]]))
    print(line("  room up on turn 1", [r for r in data if r["room_t1"]]))
    print(line("  room not up on turn 1", [r for r in data if not r["room_t1"]]))
    print("\nper ladder read (Fake Out share of Blastoise's turn 1, record):")
    by_folder = collections.defaultdict(list)
    for r in data:
        by_folder[r["folder"]].append(r)
    for folder, group in sorted(by_folder.items()):
        fo = sum(r["blastoise_t1"] == "Fake Out" for r in group)
        wins = sum(r["win"] for r in group)
        print(
            f"  {folder:44s} Fake Out {fo:2d}/{len(group):<2d}  "
            f"won {wins}-{len(group) - wins}"
        )


if __name__ == "__main__":
    main()
