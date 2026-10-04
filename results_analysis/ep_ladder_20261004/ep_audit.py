"""Earth Power on ladder (2026-10-04, pre-registered in PROJECT_STATUS): every turn our
Torkoal was on the field with Earth Power as a valued attack, rebuilt from the replay
(unit_tests/ladder_position.py with the T6e sets) and scored by the tactical teacher
(training/tactical_teacher.py position_facts: the calculator value of each attack).
Was Earth Power the best attack, and what did the bot click?
Run from the repo root:
    .venv/bin/python results_analysis/ep_ladder_20261004/ep_audit.py <replay dir> [...]
"""

from __future__ import annotations

import collections
import logging
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results_analysis/openings_20260927"))
logging.disable(logging.CRITICAL)

import numpy as np  # noqa: E402
from setswap_audit import battle_lines  # noqa: E402

from tools.ladder_loss_profile import extract_log  # noqa: E402
from training.tactical_teacher import focus_actions, position_facts  # noqa: E402
from unit_tests.ladder_position import position  # noqa: E402
from vgc_bench.src import guards as G  # noqa: E402
from vgc_bench.src.utils import act_len  # noqa: E402

OUR = "antonius1"
TEAM = "teams/candidates_mc/T6e.txt"
EP = frozenset({"earthpower"})
LEGAL = np.zeros(act_len, dtype=np.int8)
LEGAL[7:27] = 1  # the normal move band: what the teacher values


def label(battle, pos, action):
    order = G._decode(battle, int(action), pos)
    move = getattr(order, "order", None)
    target = getattr(order, "move_target", 0)
    return f"{getattr(move, 'id', '?')}->{target}"


def clicked(segment, slot):
    """Torkoal's move this turn (from |move| lines), 'switched', or None."""
    for line in segment:
        p = line.split("|")
        if len(p) > 3 and p[1] == "move" and p[2].startswith(slot):
            return re.sub(r"[^a-z]", "", p[3].lower())
        if len(p) > 2 and p[1] in ("switch", "drag") and p[2].startswith(slot):
            return "switched"
        if len(p) > 2 and p[1] == "cant" and p[2].startswith(slot):
            return "cant"
    return None


def main() -> None:
    rows = []
    for folder in sys.argv[1:]:
        for path in sorted(Path(folder).glob("*.html")):
            log = extract_log(path)
            if not log:
                continue
            lines = battle_lines(log, f"|player|p2|{OUR}" in log)
            won = f"|win|{OUR}" in log
            turn_at = {
                int(ln.split("|")[2]): i
                for i, ln in enumerate(lines)
                if ln.startswith("|turn|")
            }
            for turn in sorted(turn_at):
                try:
                    battle = position(lines[: turn_at[turn] + 1], team=TEAM)
                except Exception:
                    continue
                end = turn_at.get(turn + 1, len(lines))
                segment = lines[turn_at[turn] + 1 : end]
                for pos in (0, 1):
                    me = battle.active_pokemon[pos]
                    if me is None or me.species != "torkoal" or me.fainted:
                        continue
                    focus = focus_actions(battle, pos, LEGAL, EP)
                    try:
                        _, values = position_facts(battle, pos, LEGAL, None)
                    except Exception:
                        continue
                    valued = ~np.isnan(values)
                    if not (valued & focus).any() or valued.sum() < 2:
                        continue
                    cand = np.flatnonzero(valued)
                    best = int(cand[np.argmax(values[cand])])
                    ep_best = max(values[a] for a in cand if focus[a])
                    other_best = max(
                        (values[a] for a in cand if not focus[a]), default=-1.0
                    )
                    rows.append(
                        dict(
                            game=path.name.split("battle-")[-1][:40],
                            turn=turn,
                            won=won,
                            best=label(battle, pos, best),
                            ep_is_best=bool(focus[best]),
                            margin=float(ep_best - other_best),
                            clicked=clicked(segment, f"p1{'ab'[pos]}"),
                        )
                    )
    print(f"Torkoal turns with Earth Power valued: {len(rows)}")
    groups = {
        "EP best by > 0.05": [
            r for r in rows if r["ep_is_best"] and r["margin"] > 0.05
        ],
        "EP best by <= 0.05": [
            r for r in rows if r["ep_is_best"] and r["margin"] <= 0.05
        ],
        "EP worse": [r for r in rows if not r["ep_is_best"]],
    }
    attacks = {"earthpower", "eruption", "heatwave"}
    for name, group in groups.items():
        counts = collections.Counter(r["clicked"] for r in group)
        hits = [r for r in group if r["clicked"] in attacks]
        ep = sum(r["clicked"] == "earthpower" for r in hits)
        share = f"{ep}/{len(hits)} attacks were Earth Power" if hits else "no attacks"
        every = dict(counts.most_common())
        print(f"  {name:20s} n={len(group):3d}  {share}; all: {every}")
    for r in rows:
        if r["ep_is_best"] and r["clicked"] in attacks - {"earthpower"}:
            print(
                f"    missed? {r['game']} T{r['turn']} {'W' if r['won'] else 'L'} "
                f"best {r['best']} (+{r['margin']:.2f}) clicked {r['clicked']}"
            )


if __name__ == "__main__":
    main()
