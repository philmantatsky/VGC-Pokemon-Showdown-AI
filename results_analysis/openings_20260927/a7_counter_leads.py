"""Scratch analysis 7: what beats the leads that beat us (human corpus).

For each problem opposing lead pair, the responding side's record split by
properties of ITS lead / turn-1 play (species-agnostic): Fake Out used,
Trick Room set on turn 1, a weather move / weather setter lead, a Water / Fire /
Dark / Fighting / Ground attacker led, Intimidate, Protect on turn 1, and
the turn-1 target focus (double-target on one foe vs split).
"""

from __future__ import annotations

import collections

from openings_common import dex, elo_expect, features, load, other, wl

rows = [r for r in load() if r["source"] in ("top", "t6like")]

PROBLEM = {
    "Excadrill + Tyranitar": {"Excadrill", "Tyranitar"},
    "Gardevoir + Indeedee-F": {"Gardevoir", "Indeedee-F"},
    "Armarouge + Indeedee-F": {"Armarouge", "Indeedee-F"},
    "Hatterene + Indeedee-F": {"Hatterene", "Indeedee-F"},
    "Archaludon + Pelipper": {"Archaludon", "Pelipper"},
    "Archaludon + Grimmsnarl": {"Archaludon", "Grimmsnarl"},
    "Salamence + Sneasler": {"Salamence", "Sneasler"},
    "Gholdengo + Raichu": {"Gholdengo", "Raichu"},
    "Kingambit + Sneasler": {"Kingambit", "Sneasler"},
    "Rillaboom + Sneasler": {"Rillaboom", "Sneasler"},
    "Floette-Eternal + Incineroar": {"Floette-Eternal", "Incineroar"},
    "Farigiraf + Incineroar": {"Farigiraf", "Incineroar"},
    "Gengar + Incineroar": {"Gengar", "Incineroar"},
}


def perf(sides):
    n = len(sides)
    if not n:
        return "  -"
    w = sum(r["winner_side"] == s for r, s in sides)
    exp = sum(
        elo_expect(r["rating"].get(s), r["rating"].get(other(s))) for r, s in sides
    )
    return f"{wl(w, n):22s} vsElo {100 * (w - exp) / n:+5.1f}"


def tags(r, s):
    o = other(s)
    t = set()
    acts = r["actions"]["1"][s]
    moves = [a["move"] for a in acts]
    f = features(r, s)
    if "Fake Out" in moves:
        t.add("used Fake Out")
    if "Trick Room" in moves:
        t.add("set Trick Room T1")
    if "Tailwind" in moves:
        t.add("set Tailwind T1")
    if any(
        m in ("Protect", "Detect", "Spiky Shield", "King's Shield", "Baneful Bunker")
        for m in moves
    ):
        t.add("a Protect T1")
    if any(m in ("Rain Dance", "Sunny Day", "Sandstorm", "Snowscape") for m in moves):
        t.add("weather move T1")
    for sp in r["leads"][s]:
        if (
            sp in f.get("rain_setter_by", [])
            or sp in f.get("sun_setter_by", [])
            or sp in f.get("sand_setter_by", [])
            or sp in f.get("snow_setter_by", [])
        ):
            t.add("led a weather setter")
        if sp in f.get("tr_setter_by", []):
            t.add("led a TR setter")
        if sp in f.get("follow_me_by", []) or sp in f.get("rage_powder_by", []):
            t.add("led a redirector")
        types = dex(sp).get("types", [])
        if "Water" in types:
            t.add("led a Water type")
        if "Fire" in types:
            t.add("led a Fire type")
        if "Dark" in types:
            t.add("led a Dark type")
        if "Steel" in types:
            t.add("led a Steel type")
        if "Fighting" in types:
            t.add("led a Fighting type")
        if "Ground" in types:
            t.add("led a Ground type")
        if (
            dex(sp).get("abilities", {}).get("0") == "Intimidate"
            or "Intimidate" in dex(sp).get("abilities", {}).values()
        ):
            if (
                r["sheet"][s].get(sp, {}).get("ability", "Intimidate") or "Intimidate"
            ) == "Intimidate":
                t.add("led Intimidate")
    tg = [a["target"] for a in acts if a["target_side"] == o and not a["spread"]]
    if len(tg) == 2 and tg[0] == tg[1]:
        t.add("double-targeted one foe")
    if any(a["spread"] for a in acts):
        t.add("used a spread move")
    return t


for name, pair in PROBLEM.items():
    sides = [
        (r, other(s)) for r in rows for s in ("p1", "p2") if set(r["leads"][s]) == pair
    ]
    if len(sides) < 30:
        continue
    print(
        f"\n##### vs {name}: n={len(sides)}; the responding side overall {perf(sides)}"
    )
    c = collections.defaultdict(list)
    for r, s in sides:
        for t in tags(r, s):
            c[t].append((r, s))
    for t, lst in sorted(c.items(), key=lambda kv: -len(kv[1])):
        if len(lst) >= 12:
            rest = [x for x in sides if x not in lst]
            without = perf(rest).split("  ")[0].strip()
            print(f"   {t:26s} {perf(lst)}   (without: {without})")
