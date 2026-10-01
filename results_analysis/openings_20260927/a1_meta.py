"""Scratch analysis 1: the meta as seen at our rating band.

Archetype frequencies (human corpus vs our ladder opponents), most-used
species, most common lead pairs, and turn-1 behaviour of the common leads.
"""

from __future__ import annotations

import collections
import json

from openings_common import HERE, T6_ERA, archetype, base, features, load, other

rows = load()
top = [r for r in rows if r["source"] == "top"]
ours = [r for r in rows if r["source"] in T6_ERA and r.get("our_side")]

# --- archetypes: corpus (both sides) vs our ladder opponents
arch_c = collections.Counter()
feat_c = collections.Counter()
n_c = 0
cache = {}
for r in top:
    for side in ("p1", "p2"):
        f = features(r, side)
        cache[(r["id"], side)] = f
        arch_c[archetype(f)] += 1
        n_c += 1
        for k, v in f.items():
            if k.endswith("_by"):
                continue
            if isinstance(v, bool) and v:
                feat_c[k] += 1
            elif k == "fake_out_users" and v >= 2:
                feat_c["fake_out_users>=2"] += 1
            elif k == "physical_threats" and v >= 4:
                feat_c["physical_threats>=4"] += 1
arch_o = collections.Counter()
feat_o = collections.Counter()
for r in ours:
    side = other(r["our_side"])
    f = features(r, side)
    cache[(r["id"], side)] = f
    arch_o[archetype(f)] += 1
    for k, v in f.items():
        if k.endswith("_by"):
            continue
        if isinstance(v, bool) and v:
            feat_o[k] += 1
        elif k == "fake_out_users" and v >= 2:
            feat_o["fake_out_users>=2"] += 1
        elif k == "physical_threats" and v >= 4:
            feat_o["physical_threats>=4"] += 1
print(
    f"== primary archetype: corpus rosters n={n_c} | our T6-era ladder opponents "
    f"n={len(ours)}"
)
for a, c in arch_c.most_common():
    print(
        f"  {a:16s} {100 * c / n_c:5.1f}%   ours {100 * arch_o[a] / len(ours):5.1f}% "
        f"({arch_o[a]})"
    )
print("== features (multi-label)")
for k, c in feat_c.most_common():
    print(f"  {k:22s} {100 * c / n_c:5.1f}%   ours {100 * feat_o[k] / len(ours):5.1f}%")

# --- species usage
sp_c = collections.Counter()
lead_c = collections.Counter()
pair_c = collections.Counter()
for r in top:
    for side in ("p1", "p2"):
        for s in set(base(x) for x in r["preview"][side]):
            sp_c[s] += 1
        for s in r["leads"][side]:
            lead_c[base(s)] += 1
        pair_c[tuple(sorted(base(s) for s in r["leads"][side]))] += 1
print("== species on rosters (top 40) | led% of games where on roster")
for s, c in sp_c.most_common(40):
    print(f"  {s:20s} {100 * c / n_c:5.1f}%   led {100 * lead_c[s] / c:4.0f}%")
print("== lead pairs (top 30)")
for p, c in pair_c.most_common(30):
    print(f"  {' + '.join(p):40s} {c:4d} ({100 * c / n_c:.1f}%)")

# --- turn-1 behaviour of common leads
t1 = collections.defaultdict(collections.Counter)
t1n = collections.Counter()
for r in top:
    for side in ("p1", "p2"):
        acted = collections.Counter()
        for a in r["actions"]["1"][side]:
            u = base(a["user"])
            if u in acted:
                continue  # first action only (Dancer etc. excluded)
            acted[u] += 1
            t1[u][a["move"]] += 1
        for c in r["cant"]["1"][side]:
            u = base(c["user"])
            if u not in acted:
                t1[u]["<cant:" + c["reason"].replace("ability: ", "") + ">"] += 1
                acted[u] += 1
        for s in r["leads"][side]:
            s = base(s)
            t1n[s] += 1
            if s not in acted:
                t1[s]["<switched/no action>"] += 1
print("== turn-1 action of the 30 most-led species")
for s, n in t1n.most_common(30):
    moves = ", ".join(f"{m} {100 * c / n:.0f}%" for m, c in t1[s].most_common(6))
    print(f"  {s:18s} n={n:4d}: {moves}")

json.dump(
    {f"{k[0]}|{k[1]}": {kk: vv for kk, vv in v.items()} for k, v in cache.items()},
    (HERE / "features_cache.json").open("w"),
)
