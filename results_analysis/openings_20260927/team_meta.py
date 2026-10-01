"""Team review 2: which teams win for humans at our rating (corpus + web replays)."""

from __future__ import annotations

import collections
import itertools
import json
import re

from openings_common import HERE, archetype, base, elo_expect, features, load, other, wl

HIGH_VOLUME = {"sccontrol", "scsme", "scorecardpokemon", "kevdan42"}
sid = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())  # noqa: E731

rows = [r for r in load() if r["source"] in ("top", "t6like")]
rows += [json.loads(line) for line in (HERE / "games_web.jsonl").open()]
seen = set()
human = []
for r in rows:
    if r["id"] in seen:
        continue
    seen.add(r["id"])
    human.append(r)
print("human games:", len(human))

TEAMS = {
    "T6 (ours)": {
        "Blastoise",
        "Farigiraf",
        "Charizard",
        "Venusaur",
        "Torkoal",
        "Incineroar",
    },
    "T4 sun": {"Charizard", "Sylveon", "Garchomp", "Venusaur", "Incineroar", "Toxapex"},
    "T0 MB430": {
        "Floette-Eternal",
        "Charizard",
        "Whimsicott",
        "Garchomp",
        "Basculegion",
        "Kingambit",
    },
    "T1 consensus": {
        "Rillaboom",
        "Incineroar",
        "Floette-Eternal",
        "Sneasler",
        "Salamence",
        "Kingambit",
    },
    "T2 TR/PT": {
        "Indeedee-F",
        "Hatterene",
        "Golisopod",
        "Torkoal",
        "Farigiraf",
        "Kingambit",
    },
    "T3 PT offense": {
        "Gardevoir",
        "Salamence",
        "Indeedee-F",
        "Sneasler",
        "Kingambit",
        "Rillaboom",
    },
    "T5 rain": {
        "Golisopod",
        "Swampert",
        "Pelipper",
        "Archaludon",
        "Sinistcha",
        "Grimmsnarl",
    },
    "Baltimore #1 sand": {
        "Salamence",
        "Tyranitar",
        "Excadrill",
        "Indeedee",
        "Corviknight",
        "Sneasler",
    },
    "Baltimore #2 (Scorecard)": {
        "Charizard",
        "Golisopod",
        "Politoed",
        "Archaludon",
        "Grimmsnarl",
        "Farigiraf",
    },
    "Raichu/Gholdengo balance": {
        "Raichu",
        "Gholdengo",
        "Rillaboom",
        "Arcanine-Hisui",
        "Staraptor",
        "Milotic",
    },
}


def sides_all(exclude_hv=True, band=(0, 9999)):
    for r in human:
        for s in ("p1", "p2"):
            name = sid(r["players"].get(s, ""))
            if exclude_hv and name in HIGH_VOLUME:
                continue
            ra = r["rating"].get(s)
            if ra and not (band[0] <= ra <= band[1]):
                continue
            yield r, s


def perf(lst):
    n = len(lst)
    if n == 0:
        return "-"
    w = sum(r["winner_side"] == s for r, s in lst)
    exp = sum(elo_expect(r["rating"].get(s), r["rating"].get(other(s))) for r, s in lst)
    return f"{wl(w, n):24s} vsElo {100 * (w - exp) / n:+5.1f}pp"


allsides = list(sides_all())
print("sides (high-volume accounts excluded):", len(allsides))

print("\n== human teams built like each candidate (>=4 of its six on the roster)")
for name, team in TEAMS.items():
    lst = [
        (r, s)
        for r, s in allsides
        if len({base(x) for x in r["preview"][s]} & team) >= 4
    ]
    lst5 = [
        (r, s) for r, s in lst if len({base(x) for x in r["preview"][s]} & team) >= 5
    ]
    print(f"  {name:26s} >=4: {perf(lst)}   >=5: {perf(lst5)}")

print("\n== the high-volume accounts (their team, their record)")
hv = [
    (r, s)
    for r in human
    for s in ("p1", "p2")
    if sid(r["players"].get(s, "")) in HIGH_VOLUME
]
print(
    "  ",
    perf(hv),
    collections.Counter(tuple(sorted(r["preview"][s])) for r, s in hv).most_common(2),
)
ratings = sorted(r["rating"].get(s) or 0 for r, s in hv)
if ratings:
    print(
        "   their ratings: min",
        ratings[0],
        "median",
        ratings[len(ratings) // 2],
        "max",
        ratings[-1],
    )

print("\n== archetype performance (primary label)")
arch = collections.defaultdict(list)
for r, s in allsides:
    arch[archetype(features(r, s))].append((r, s))
for a, lst in sorted(arch.items(), key=lambda kv: -len(kv[1])):
    print(f"  {a:16s} {perf(lst)}  share {100 * len(lst) / len(allsides):.1f}%")

print("\n== hybrid structures")
for label, pred in (
    ("TR setter + sun setter", lambda f: f["tr_setter"] and f["sun_setter"]),
    ("TR setter + rain setter", lambda f: f["tr_setter"] and f["rain_setter"]),
    ("TR setter + Tailwind", lambda f: f["tr_setter"] and f["tailwind"]),
    ("sun, no TR", lambda f: f["sun_setter"] and not f["tr_setter"]),
    ("sand", lambda f: f["sand_setter"]),
    ("rain, no TR", lambda f: f["rain_setter"] and not f["tr_setter"]),
):
    lst = [(r, s) for r, s in allsides if pred(features(r, s))]
    print(f"  {label:26s} {perf(lst)}")

print("\n== species on roster, vsElo (top 45 by usage)")
spc = collections.defaultdict(list)
for r, s in allsides:
    for x in {base(y) for y in r["preview"][s]}:
        spc[x].append((r, s))
for sp, lst in sorted(spc.items(), key=lambda kv: -len(kv[1]))[:45]:
    print(f"  {sp:16s} {100 * len(lst) / len(allsides):4.1f}%  {perf(lst)}")

print("\n== best-performing pairs on roster (n>=250), by vsElo")
pairs = collections.defaultdict(list)
for r, s in allsides:
    ro = sorted({base(y) for y in r["preview"][s]})
    for a, b in itertools.combinations(ro, 2):
        pairs[(a, b)].append((r, s))
scored = []
for k, lst in pairs.items():
    if len(lst) >= 250:
        w = sum(r["winner_side"] == s for r, s in lst)
        exp = sum(
            elo_expect(r["rating"].get(s), r["rating"].get(other(s))) for r, s in lst
        )
        scored.append(((w - exp) / len(lst), k, lst))
for v, k, lst in sorted(scored, reverse=True)[:15]:
    print(f"  {' + '.join(k):32s} {perf(lst)}")
print("  ... worst:")
for v, k, lst in sorted(scored)[:8]:
    print(f"  {' + '.join(k):32s} {perf(lst)}")
