# ruff: noqa: E501, E741, F841
"""What the likely-move worlds would have held on the logged ladder decisions (offline, read-only).
For replies made of moves only: is every move of the reply in the set of ONE of the four worlds?"""

import collections
import glob
import html
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")
import os

from poke_env.data import to_id_str

from vgc_bench.src.set_particles import ParticleDatabase

TURNING = int(os.environ.get("TURNING", "1"))
WORLDS = int(os.environ.get("WORLDS", "4"))


def likely_moves(shown, ranked, index):
    known = list(dict.fromkeys(m for m in shown if m))
    free = 4 - len(known)
    rest = [m for m in dict.fromkeys(ranked) if m and m not in known]
    if free <= 0 or not rest:
        return None
    turning_slots = min(TURNING, free)
    fixed, pool = rest[: free - turning_slots], rest[free - turning_slots :]
    groups = max(1, min(WORLDS, -(-len(pool) // turning_slots))) if pool else 1
    start = (index % groups) * turning_slots
    return known + fixed + pool[start : start + turning_slots]


DB = ParticleDatabase.load(formatid="gen9championsvgc2026regmc")
LOG = re.compile(
    r'<script type="text/plain" class="battle-log-data">(.*?)</script>', re.S
)


def timeline(path):
    text = Path(path).read_text(errors="replace")
    m = LOG.search(text)
    log = html.unescape(m.group(1)) if m else text
    ours = re.search(r"\|player\|(p[12])\|antonius1", log)
    if not ours:
        return None
    opp = "p2" if ours.group(1) == "p1" else "p1"
    active = {}
    shown = collections.defaultdict(list)
    nick = {}
    turns = {}
    for line in log.split("\n"):
        p = line.split("|")
        if len(p) < 2:
            continue
        if p[1] == "turn":
            turns[int(p[2])] = {
                "active": dict(active),
                "shown": {k: list(v) for k, v in shown.items()},
            }
        elif p[1] in ("switch", "drag", "replace") and p[2].startswith(opp):
            species = to_id_str(p[3].split(",")[0])
            active["ab".index(p[2][2])] = species
            nick[p[2].split(": ", 1)[1]] = species
        elif p[1] == "move" and p[2].startswith(opp):
            species = nick.get(p[2].split(": ", 1)[1])
            move = to_id_str(p[3])
            if (
                species
                and move not in shown[species]
                and move not in ("struggle", "recharge")
                and "[from]" not in line
            ):
                shown[species].append(move)
    return turns


def forecast_moves(record, slot):
    slots = (record or {}).get("slots") or []
    if slot >= len(slots) or not slots[slot]:
        return []
    by = collections.Counter()
    for a in slots[slot].get("actions") or []:
        if a.get("kind") == "move" and a.get("move"):
            by[a["move"]] += float(a.get("p") or 0)
    return [m for m, _ in by.most_common()]


t = collections.Counter()
for directory in sys.argv[1:]:
    tl = {
        re.search(r"regmc-(\d+)", p).group(1): timeline(p)
        for p in glob.glob(f"{directory}/*.html")
    }
    own = {}
    for line in open(f"{directory}/decisions_champion.jsonl"):
        r = json.loads(line)
        if r.get("turn") and "slots" in (r.get("opponent_forecast") or {}):
            own[(r["battle"], r["turn"])] = r["opponent_forecast"]
    by = collections.defaultdict(list)
    for line in open(f"{directory}/decisions.jsonl"):
        r = json.loads(line)
        if r.get("exact_search"):
            by[r["battle"]].append(r)
    for tag, rs in by.items():
        turns = tl.get(re.search(r"regmc-(\d+)", tag).group(1))
        moves = []
        for r in rs:
            e = r["exact_search"]
            cov = e.get("reply_coverage") or {}
            if (
                cov.get("observed")
                and moves
                and cov.get("priors")
                and "best_rank" in cov["priors"]
                and turns
                and not e.get("open_sheet")
            ):
                turn = moves[-1].get("turn")
                start = turns.get(turn)
                obs = cov["observed"]
                if start and all(v[0] == "move" for v in obs.values()):
                    best = cov["priors"]["best_rank"]
                    was = any(best.get(k) for k in best)
                    fc = own.get((tag, turn))
                    worlds = [True] * WORLDS
                    known = True
                    for slot, (_k, move, _t, _m) in obs.items():
                        slot = int(slot)
                        species = start["active"].get(slot)
                        if species is None:
                            known = False
                            break
                        shown = start["shown"].get(species, [])
                        ranked = forecast_moves(fc, slot) + [
                            m for m, _ in DB.move_usage(species)
                        ]
                        for index in range(WORLDS):
                            have = (
                                likely_moves(shown, ranked, index)
                                if len(shown) < 4
                                else list(shown)
                            )
                            if have is None:
                                have = list(shown)
                            if move not in have:
                                worlds[index] = False
                    if known:
                        now = any(worlds)
                        t["move-only replies, hidden sheets"] += 1
                        t[f"legal before: {was} -> in a likely world: {now}"] += 1
                        t["worlds holding it (of 4), sum"] += sum(worlds)
            champion = e.get("champion_actions") or [9, 9]
            if not (
                0 in [int(a) for a in champion] and all(int(a) < 7 for a in champion)
            ):
                moves.append(r)
n = t["move-only replies, hidden sheets"]
for k, v in sorted(t.items()):
    print(f"{v:5d}  {k}")
before = (
    t["legal before: True -> in a likely world: True"]
    + t["legal before: True -> in a likely world: False"]
)
after = (
    t["legal before: True -> in a likely world: True"]
    + t["legal before: False -> in a likely world: True"]
)
print(
    f"legal in some world: before {before} of {n} ({100 * before / n:.0f}%), with likely-move worlds {after} of {n} ({100 * after / n:.0f}%); mean worlds holding it {t['worlds holding it (of 4), sum'] / n:.2f} of 4"
)
