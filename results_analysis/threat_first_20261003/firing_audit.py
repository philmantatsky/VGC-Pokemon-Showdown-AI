"""Scratch (2026-10-03): where threat_first would have changed the bot's choice on
ladder. Every logged decision of the T6-era ladder games is rebuilt from its
replay (unit_tests/ladder_position.py) with the logged candidate pairs, and the
guard runs on them. Prints how often it fires and every change, with what happened
next in the real game (did the threatened Pokemon faint that turn).
Run from the repo root (guard: threat_first, the default, fake_out_threat, threat_first2 or doomed_switch):
    .venv/bin/python results_analysis/threat_first_20261003/firing_audit.py [guard]
"""

from __future__ import annotations

import collections
import json
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results_analysis/openings_20260927"))
logging.disable(logging.CRITICAL)

from setswap_audit import battle_lines  # noqa: E402

from tools.ladder_loss_profile import extract_log  # noqa: E402
from unit_tests.ladder_position import position  # noqa: E402
from vgc_bench.src import doomed_switch  # noqa: E402
from vgc_bench.src import fake_out_threat  # noqa: E402
from vgc_bench.src import threat_first2  # noqa: E402
from vgc_bench.src import guards as G  # noqa: E402

GUARD = {
    "threat_first": G.guard_threat_first,
    "fake_out_threat": fake_out_threat.guard_fake_out_threat,
    "threat_first2": threat_first2.guard_threat_first2,
    "doomed_switch": doomed_switch.guard_doomed_switch,
}[sys.argv[1] if len(sys.argv) > 1 else "threat_first"]

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
]


def label(battle, action, pos):
    order = G._decode(battle, action, pos)
    move = getattr(order, "order", None)
    if move is None:
        return "?"
    target = getattr(order, "move_target", 0)
    name = getattr(move, "id", None) or getattr(move, "species", "?")
    foes = battle.opponent_active_pokemon
    aim = ""
    if target in (1, 2) and target - 1 < len(foes) and foes[target - 1] is not None:
        aim = ">" + foes[target - 1].species
    return f"{name}{aim}"


fired, decisions, errors = [], 0, collections.Counter()
for d in DIRS:
    folder = ROOT / f"ladder_replays_mc_{d}"
    rows = collections.defaultdict(dict)
    for line in (folder / "decisions.jsonl").read_text().splitlines():
        row = json.loads(line)
        m = re.search(r"regmc-(\d+)", row["battle"])
        if m and row.get("candidates"):
            rows[m.group(1)].setdefault(int(row["turn"]), row)
    for f in sorted(folder.glob("*.html")):
        m = re.search(r"regmc-(\d+)", f.name)
        log = extract_log(f) if m else None
        if not log or m.group(1) not in rows:
            continue
        lines = battle_lines(log, f"|player|p2|{OUR}" in log)
        turn_at = {}
        for i, line in enumerate(lines):
            if line.startswith("|turn|"):
                turn_at[int(line.split("|")[2])] = i
        for turn, row in sorted(rows[m.group(1)].items()):
            if turn not in turn_at:
                continue
            try:
                battle = position(lines[: turn_at[turn] + 1])
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
                report = G.GuardReport()
                out = GUARD(battle, cands, report)
            except Exception as exc:
                errors[type(exc).__name__] += 1
                continue
            if out[0] is cands[0]:
                continue
            # the threat the promoted pair answers: the one whose foe it now attacks
            changed = [
                pp for pp in (0, 1) if out[0].actions[pp] != cands[0].actions[pp]
            ]
            pp = changed[0] if changed else 0
            _, hit = G._move_and_targets(
                battle, G._decode(battle, out[0].actions[pp], pp), pp
            )
            found = threat_first2.threats(battle)
            threat = next((t for t in found if any(f is t[1] for f in hit)), found[0])
            nxt = "\n".join(
                lines[turn_at[turn] + 1 : turn_at.get(turn + 1, len(lines))]
            )
            victim = battle.active_pokemon[threat[2]]
            fainted = bool(
                re.search(
                    rf"\|faint\|p1[ab]: {re.escape(victim.species.split('mega')[0].capitalize())}",
                    nxt,
                    re.I,
                )
            )
            fired.append(
                dict(
                    dir=d,
                    game=m.group(1),
                    turn=turn,
                    won=f"|win|{OUR}" in log,
                    threat=f"{threat[1].species}:{threat[3].id}->{victim.species} ({threat[0]:.2f})",
                    before=[
                        label(battle, a, p) for p, a in enumerate(cands[0].actions)
                    ],
                    after=[label(battle, a, p) for p, a in enumerate(out[0].actions)],
                    victim_fainted_that_turn=fainted,
                )
            )

print(
    f"decisions {decisions}, fired {len(fired)} ({100 * len(fired) / max(1, decisions):.1f}%), errors {dict(errors)}"
)
for row in fired:
    print(
        f"  {row['dir'][:22]:22s} {row['game']} T{row['turn']:<2d} {'W' if row['won'] else 'L'} "
        f"{row['threat']}: {row['before']} -> {row['after']}  victim fainted: {row['victim_fainted_that_turn']}"
    )
