"""Team review 3: our six sets vs what the meta runs, and what each option does.

(1) Set shares on open team sheets (human games) vs ours; (2) the record of human
sides by move / item on the sheet (vsElo; confounded by team and pilot);
(3) usage-weighted type coverage of candidate moves against the top-40 meta.
"""

from __future__ import annotations

import collections
import json
import re

from openings_common import HERE, base, dex, elo_expect, load, other, wl
from poke_env.data import GenData, to_id_str

HIGH_VOLUME = {"sccontrol", "scsme", "scorecardpokemon", "kevdan42"}
sid = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())  # noqa: E731

rows = [r for r in load() if r["source"] in ("top", "t6like")]
rows += [json.loads(line) for line in (HERE / "games_web.jsonl").open()]
seen, human = set(), []
for r in rows:
    if r["id"] not in seen:
        seen.add(r["id"])
        human.append(r)

OURS = {
    "Blastoise": ["Fake Out", "Water Spout", "Water Pulse", "Ice Beam"],
    "Farigiraf": ["Trick Room", "Psychic", "Helping Hand", "Rain Dance"],
    "Charizard": ["Heat Wave", "Solar Beam", "Ancient Power", "Protect"],
    "Venusaur": ["Sludge Bomb", "Leaf Storm", "Sleep Powder", "Protect"],
    "Torkoal": ["Eruption", "Heat Wave", "Weather Ball", "Protect"],
    "Incineroar": ["Fake Out", "Flare Blitz", "Parting Shot", "Throat Chop"],
}
ITEMS = {
    "Blastoise": "Blastoisinite",
    "Farigiraf": "Sitrus Berry",
    "Charizard": "Charizardite Y",
    "Venusaur": "Focus Sash",
    "Torkoal": "Charcoal",
    "Incineroar": "Passho Berry",
}


def norm(m: str) -> str:
    return to_id_str(m)


def perf(lst):
    n = len(lst)
    if n == 0:
        return "-"
    w = sum(r["winner_side"] == s for r, s in lst)
    exp = sum(elo_expect(r["rating"].get(s), r["rating"].get(other(s))) for r, s in lst)
    return f"{wl(w, n):22s} vsElo {100 * (w - exp) / n:+5.1f}"


for sp, our_moves in OURS.items():
    sheets = []
    for r in human:
        for s in ("p1", "p2"):
            if sid(r["players"].get(s, "")) in HIGH_VOLUME:
                continue
            for name, st in r["sheet"][s].items():
                if base(name) == sp or name.split("-")[0] == sp:
                    sheets.append((r, s, st))
    n = len(sheets)
    print(f"\n######## {sp}: {n} open sheets")
    if not n:
        continue
    mv = collections.Counter(norm(m) for _, _, st in sheets for m in set(st["moves"]))
    it = collections.Counter(norm(st["item"]) for _, _, st in sheets)
    ours_ids = {norm(m) for m in our_moves}
    print(
        "  moves:",
        ", ".join(
            f"{m}{'*' if m in ours_ids else ''} {100 * c / n:.0f}%"
            for m, c in mv.most_common(12)
        ),
    )
    print(
        "  items:",
        ", ".join(
            f"{i}{'*' if i == norm(ITEMS[sp]) else ''} {100 * c / n:.0f}%"
            for i, c in it.most_common(6)
        ),
    )
    exact = [
        (r, s) for r, s, st in sheets if {norm(m) for m in st["moves"]} == ours_ids
    ]
    print(f"  sheets with exactly our four moves: {len(exact)}  {perf(exact)}")
    print("  record by move on the sheet (with / without):")
    for m, c in mv.most_common(10):
        with_ = [(r, s) for r, s, st in sheets if m in {norm(x) for x in st["moves"]}]
        without = [
            (r, s) for r, s, st in sheets if m not in {norm(x) for x in st["moves"]}
        ]
        if len(with_) >= 25 and len(without) >= 25:
            tag = m + ("*" if m in ours_ids else "")
            print(f"    {tag:18s} with {perf(with_)} | without {perf(without)}")
    print("  record by item:")
    for i, c in it.most_common(4):
        lst = [(r, s) for r, s, st in sheets if norm(st["item"]) == i]
        if len(lst) >= 25:
            print(f"    {i + ('*' if i == norm(ITEMS[sp]) else ''):18s} {perf(lst)}")

# ---- (3) usage-weighted coverage vs the meta
print("\n######## coverage: usage-weighted effectiveness vs the top-40 meta")
use = collections.Counter()
nsides = 0
for r in human:
    for s in ("p1", "p2"):
        nsides += 1
        for x in {base(y) for y in r["preview"][s]}:
            use[x] += 1
top = use.most_common(40)
chart = GenData.from_gen(9).type_chart
MEGA_TYPES = {
    "Salamence": ["Dragon", "Flying"],
    "Golisopod": ["Bug", "Steel"],
    "Garchomp": ["Dragon"],
    "Absol": ["Dark", "Ghost"],
}  # Champions Megas that change typing (as reported)


def eff(mtype: str, species: str) -> float:
    types = MEGA_TYPES.get(species) or dex(species).get("types", [])
    m = 1.0
    for t in types:
        m *= float(chart.get(t.upper(), {}).get(mtype.upper(), 1.0))
    return m


CANDIDATES = {
    "Blastoise": [
        ("Water Spout (spread, Water)", "Water"),
        ("Water Pulse (Water)", "Water"),
        ("Ice Beam", "Ice"),
        ("Dark Pulse", "Dark"),
        ("Aura Sphere", "Fighting"),
        ("Dragon Pulse", "Dragon"),
    ],
    "Charizard": [
        ("Heat Wave / Weather Ball (Fire)", "Fire"),
        ("Solar Beam", "Grass"),
        ("Ancient Power", "Rock"),
        ("Hurricane / Air Slash", "Flying"),
        ("Dragon Pulse", "Dragon"),
    ],
    "Venusaur": [
        ("Sludge Bomb", "Poison"),
        ("Leaf Storm", "Grass"),
        ("Earth Power", "Ground"),
    ],
    "Torkoal": [
        ("Eruption / Heat Wave", "Fire"),
        ("Earth Power", "Ground"),
        ("Solar Beam", "Grass"),
        ("Ancient Power", "Rock"),
    ],
    "Farigiraf": [
        ("Psychic", "Psychic"),
        ("Thunderbolt", "Electric"),
        ("Hyper Voice / Twin Beam", "Normal"),
    ],
    "Incineroar": [
        ("Flare Blitz", "Fire"),
        ("Throat Chop / Darkest Lariat", "Dark"),
        ("Close Combat", "Fighting"),
    ],
}
for sp, cands in CANDIDATES.items():
    print(f"  {sp}:")
    for label, t in cands:
        se = sum(c for s, c in top if eff(t, s) >= 2) / sum(c for _, c in top)
        res = sum(c for s, c in top if eff(t, s) < 1) / sum(c for _, c in top)
        hits = [s for s, c in top[:20] if eff(t, s) >= 2]
        print(f"    {label:32s} super-effective on {100 * se:4.1f}% of top-40 usage,")
        print(f"      resisted by {100 * res:4.1f}%  | SE top-20: {', '.join(hits)}")
