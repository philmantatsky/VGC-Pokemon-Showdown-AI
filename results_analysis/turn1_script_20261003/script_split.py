"""Scratch (2026-10-03): our past Blastoise + Farigiraf ladder games split by whether
the turn-1 script would have changed turn 1 (script_audit.py's replay) and by what
Blastoise actually did. Separates the script's reach from the confounding: Water
Spout games include the leads Fake Out cannot touch (Psychic Terrain, Armor Tail,
Ghosts), where the script does nothing. Free wins excluded.
Run from the repo root:
    .venv/bin/python results_analysis/turn1_script_20261003/script_split.py
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import script_audit as SA  # noqa: E402  (sets the usage prior, as on ladder)
from ab_read import quit_turn  # noqa: E402
from openings_parse import base, parse, sid  # noqa: E402
from room_failures import reason  # noqa: E402
from setswap_audit import battle_lines  # noqa: E402
from turn1_history import ERA, OUR, WATER  # noqa: E402

from tools.ladder_loss_profile import extract_log  # noqa: E402
from unit_tests.ladder_position import position  # noqa: E402
from vgc_bench.src import guards as G  # noqa: E402
from vgc_bench.src.playbook_opening import guard_playbook_opening  # noqa: E402

KNOWN = ("Fake Out", "Water Spout", "Ice Beam")


def main() -> None:
    groups: dict[str, list[tuple[bool, bool]]] = collections.defaultdict(list)
    for folder in sorted(SA.ROOT.glob("ladder_replays_mc_*")):
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
            rec = parse(log) if log else None
            if rec is None or m is None or m.group(1) not in turn1:
                continue
            side = next(
                (s for s, n in rec["players"].items() if sid(n) == sid(OUR)), None
            )
            if side is None or {base(s) for s in rec["leads"][side]} != WATER:
                continue
            other = "p2" if side == "p1" else "p1"
            quit_at = quit_turn(log, rec["players"].get(other, ""))
            if quit_at is not None and quit_at <= 1:
                continue
            lines = battle_lines(log, f"|player|p2|{OUR}" in log)
            cut = next(i for i, ln in enumerate(lines) if ln.startswith("|turn|1"))
            try:
                battle = position(lines[: cut + 1])
                battle._format = "gen9championsvgc2026regmc"
                setattr(battle, "_vgc_playbook", SA.PLAN)
                cands = [
                    G.Candidate(
                        actions=tuple(c["actions"]), prob=c["policy_probability"]
                    )
                    for c in turn1[m.group(1)]["candidates"]
                    if not c.get("demoted_by")
                ]
                out = guard_playbook_opening(battle, cands, G.GuardReport())
            except Exception:
                continue
            changed = bool(cands) and out[0] is not cands[0]
            moves = {base(a["user"]): a["move"] for a in rec["actions"]["1"][side]}
            move = moves.get("Blastoise", "(none)")
            key = (
                f"{'changed' if changed else 'kept':8s}"
                f"Blastoise {move if move in KNOWN else 'other'}"
            )
            groups[key].append((rec["winner_side"] == side, reason(log, side) == "up"))
    print("script would change turn 1? / Blastoise's real turn 1:")
    for key in sorted(groups):
        g = groups[key]
        n, wins, up = len(g), sum(w for w, _ in g), sum(u for _, u in g)
        print(
            f"  {key:32s} {n:3d} games  won {wins}-{n - wins} ({100 * wins / n:.0f}%)"
            f"  room up {100 * up / n:.0f}%"
        )


if __name__ == "__main__":
    main()
