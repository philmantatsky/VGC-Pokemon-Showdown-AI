"""Scratch (2026-10-03): the turn-1 script (Water Room card, script only) replayed on
our real ladder turn-1 positions with Blastoise + Farigiraf leads. Each turn-1
decision is rebuilt from its replay (unit_tests/ladder_position.py) with the logged
candidates; playbook_opening runs on them. Prints how often the script would have
changed turn 1, its Fake Out target, and -- for the games where our room did not go
up -- whether a Fake Out could reach the Pokemon that stopped it.
Run from the repo root:
    .venv/bin/python results_analysis/turn1_script_20261003/script_audit.py
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.disable(logging.CRITICAL)
os.environ["VGC_MOVESET_PRIOR"] = "1"  # hidden sets from usage, as on ladder

from openings_parse import base, parse, sid  # noqa: E402
from room_failures import reason  # noqa: E402
from setswap_audit import battle_lines  # noqa: E402
from turn1_history import ERA, OUR, WATER  # noqa: E402

from tools.ladder_loss_profile import extract_log  # noqa: E402
from unit_tests.ladder_position import position  # noqa: E402
from vgc_bench.src import guards as G  # noqa: E402
from vgc_bench.src.playbook_opening import (  # noqa: E402
    guard_playbook_opening,
    scripted_steps,
)

CARD = next(
    c
    for c in json.loads((ROOT / "data/playbook_t6_trial.json").read_text())["cards"]
    if c["name"] == "water_room"
)
PLAN = {
    "card": "water_room",
    "lead": CARD["lead"],
    "turn1": CARD["turn1"],
    "script_only": True,
}


def main() -> None:
    stats = collections.Counter()
    failed = collections.defaultdict(collections.Counter)
    examples = []
    for folder in sorted(ROOT.glob("ladder_replays_mc_*")):
        if not folder.is_dir() or not any(k in folder.name for k in ERA):
            continue
        log_path = folder / "decisions.jsonl"
        if not log_path.exists():
            continue
        turn1 = {}
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
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
            lines = battle_lines(log, f"|player|p2|{OUR}" in log)
            cut = next(i for i, ln in enumerate(lines) if ln.startswith("|turn|1"))
            try:
                battle = position(lines[: cut + 1])
                battle._format = "gen9championsvgc2026regmc"  # as on ladder
                setattr(battle, "_vgc_playbook", PLAN)
                steps = scripted_steps(battle, PLAN)
                cands = [
                    G.Candidate(
                        actions=tuple(c["actions"]), prob=c["policy_probability"]
                    )
                    for c in turn1[m.group(1)]["candidates"]
                    if not c.get("demoted_by")
                ]
                report = G.GuardReport()
                out = guard_playbook_opening(battle, cands, report) if cands else cands
            except Exception as exc:
                stats[f"error {type(exc).__name__}"] += 1
                continue
            stats["games"] += 1
            fo = next((t for s, (mv, t) in steps.items() if mv == "fakeout"), None)
            foes = battle.opponent_active_pokemon
            target = foes[fo - 1].species if fo and foes[fo - 1] is not None else None
            changed = bool(cands) and out[0] is not cands[0]
            why = reason(log, side)
            stats["script has a Fake Out step"] += fo is not None
            stats["script changes turn 1"] += changed
            stats["room up"] += why == "up"
            if why != "up":
                failed[why]["games"] += 1
                failed[why]["Fake Out step"] += fo is not None
                failed[why]["changed"] += changed
                examples.append(
                    (
                        folder.name.removeprefix("ladder_replays_mc_"),
                        m.group(1),
                        why,
                        " + ".join(f.species for f in foes if f),
                        target,
                        changed,
                    )
                )
    print(dict(stats))
    print(
        "\nroom not up, by reason: games / the script has a Fake Out step / "
        "it changes turn 1"
    )
    for why, c in sorted(failed.items(), key=lambda kv: -kv[1]["games"]):
        print(f"  {why:42s} {c['games']:3d} {c['Fake Out step']:3d} {c['changed']:3d}")
    print(
        "\nexamples (read, battle, reason, their leads, script Fake Out target, "
        "changed):"
    )
    for e in examples:
        print("  ", e)


if __name__ == "__main__":
    main()
