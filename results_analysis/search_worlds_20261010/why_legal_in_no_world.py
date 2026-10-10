# ruff: noqa: E501, E741, F841
"""Why is a real reply legal in no world? From the ladder logs, the replays and the set data.
Read-only. Usage (repo root): .venv/bin/python <this> <replay dir> [<replay dir> ...]"""

import collections
import glob
import html
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")
from poke_env.data import to_id_str

from vgc_bench.src.set_particles import ParticleDatabase

DB = ParticleDatabase.load(formatid="gen9championsvgc2026regmc")
LOG = re.compile(
    r'<script type="text/plain" class="battle-log-data">(.*?)</script>', re.S
)


def particles(species):
    try:
        return DB.particles(species)
    except Exception:
        return ()


def game_timeline(path):
    """turn -> opponent's actives at the start of the turn, species seen before it, moves each
    opposing Pokemon had shown before it."""
    text = Path(path).read_text(errors="replace")
    m = LOG.search(text)
    log = html.unescape(m.group(1)) if m else text
    ours = re.search(r"\|player\|(p[12])\|antonius1", log)
    if not ours:
        return None
    opp = "p2" if ours.group(1) == "p1" else "p1"
    active = {}
    seen = set()
    shown = collections.defaultdict(set)
    megas = set()
    turns = {}
    turn = 0
    nick = {}
    for line in log.split("\n"):
        p = line.split("|")
        if len(p) < 2:
            continue
        if p[1] == "turn":
            turn = int(p[2])
            turns[turn] = {
                "active": dict(active),
                "seen": set(seen),
                "shown": {k: set(v) for k, v in shown.items()},
            }
        elif p[1] in ("switch", "drag", "replace") and p[2].startswith(opp):
            slot = "ab".index(p[2][2])
            species = to_id_str(p[3].split(",")[0])
            active[slot] = species
            seen.add(species)
            nick[p[2].split(": ", 1)[1]] = species
        elif p[1] == "move" and p[2].startswith(opp):
            species = nick.get(p[2].split(": ", 1)[1])
            if species:
                shown[species].add(to_id_str(p[3]))
        elif p[1] == "faint" and p[2].startswith(opp):
            pass
    return turns


def classify(observed, start):
    """One reply legal in no world -> the reasons the set data and the game give for it."""
    reasons = []
    for slot, (kind, ident, _target, mega) in sorted(observed.items()):
        slot = int(slot)
        if kind == "switch":
            reasons.append(
                "switch to a Pokemon not seen before"
                if ident not in start["seen"]
                else "switch to a Pokemon seen before"
            )
            continue
        species = start["active"].get(slot)
        if species is None:
            reasons.append("mover unknown")
            continue
        sets = particles(species)
        if not sets:
            reasons.append("species without sets")
            continue
        known = start["shown"].get(species, set())
        if ident in known:
            reasons.append("a move it had shown before" + (" + Mega" if mega else ""))
        elif not any(ident in s.moves for s in sets):
            reasons.append("a move in none of the species' sets")
        elif not any(ident in s.moves and known <= set(s.moves) for s in sets):
            reasons.append("a move in no set that also holds what it had shown")
        else:
            share = sum(
                s.probability
                for s in sets
                if ident in s.moves and known <= set(s.moves)
            ) / max(1e-9, sum(s.probability for s in sets if known <= set(s.moves)))
            reasons.append(
                "a new move some set holds (%s of the fitting sets' weight)"
                % (
                    "under 25%"
                    if share < 0.25
                    else "25-50%"
                    if share < 0.5
                    else "over 50%"
                )
            )
        if mega and not any(
            s.mega()
            if callable(getattr(s, "mega", None))
            else getattr(s, "mega", False)
            for s in sets
        ):
            reasons.append("Mega Evolution no set allows")
    return reasons


total = collections.Counter()
causes = collections.Counter()
examples = collections.defaultdict(list)
for directory in sys.argv[1:]:
    timelines = {}
    for path in glob.glob(f"{directory}/*.html"):
        number = re.search(r"regmc-(\d+)", path).group(1)
        timelines[number] = game_timeline(path)
    rows = [json.loads(l) for l in open(f"{directory}/decisions.jsonl")]
    by = collections.defaultdict(list)
    for r in rows:
        if r.get("exact_search"):
            by[r["battle"]].append(r)
    for tag, rs in by.items():
        number = re.search(r"regmc-(\d+)", tag).group(1)
        tl = timelines.get(number)
        if not tl:
            continue
        moves = []
        for r in rs:
            e = r["exact_search"]
            cov = e.get("reply_coverage") or {}
            if (
                cov.get("observed")
                and moves
                and cov.get("priors")
                and "best_rank" in cov["priors"]
            ):
                total["replies with a logged ranking"] += 1
                best = cov["priors"]["best_rank"]
                if not any(best.get(k) for k in best):
                    total["legal in no world"] += 1
                    start = tl.get(moves[-1].get("turn"))
                    if start is None:
                        causes["(turn not found in the replay)"] += 1
                        continue
                    reasons = classify(cov["observed"], start)
                    key = " + ".join(sorted(set(reasons)))
                    causes[key] += 1
                    if len(examples[key]) < 3:
                        examples[key].append(
                            (
                                number[-6:],
                                moves[-1].get("turn"),
                                cov["observed"],
                                {s: start["active"].get(s) for s in (0, 1)},
                            )
                        )
            champion = e.get("champion_actions") or [9, 9]
            forced = 0 in [int(a) for a in champion] and all(
                int(a) < 7 for a in champion
            )
            if not forced:
                moves.append(r)
print(dict(total))
ORDER = [
    "a move in none of the species' sets",
    "a move in no set that also holds what it had shown",
    "species without sets",
    "Mega Evolution no set allows",
    "switch to a Pokemon not seen before",
    "switch to a Pokemon seen before",
    "a new move some set holds (under 25%",
    "a new move some set holds (25-50%",
    "a new move some set holds (over 50%",
    "a move it had shown before",
    "mover unknown",
]
binding = collections.Counter()
anywhere = collections.Counter()
for key, n in causes.items():
    hit = next((o for o in ORDER if o in key), key)
    binding[hit] += n
    for o in ORDER:
        if o in key:
            anywhere[o] += n
n_all = total["legal in no world"]
print("BY THE HARDEST PART OF THE REPLY (each reply once):")
for o in ORDER:
    if binding[o]:
        print(
            f"  {binding[o]:4d} ({100 * binding[o] / n_all:.0f}%)  {o}   [present in {anywhere[o]} replies]"
        )
