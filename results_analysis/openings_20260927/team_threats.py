"""Team review 1: what beats T6 -- ladder (134 games) and the local battery (6,204)."""

from __future__ import annotations

import collections
import glob
import json

from openings_common import REPO, T6_ERA, T6_LATER, base, elo_expect, load, other, wl

rows = load()
ours = [r for r in rows if r["source"] in T6_ERA | T6_LATER and r.get("our_side")]
t4 = [
    r
    for r in rows
    if r["source"] == "ours:brainv1spec_T4_20260920" and r.get("our_side")
]
mb = [r for r in rows if r["source"] == "ours:guards3_20260909" and r.get("our_side")]


def rec(lst):
    w = sum(r["winner_side"] == r["our_side"] for r in lst)
    exp = sum(
        elo_expect(
            r["rating"].get(r["our_side"]), r["rating"].get(other(r["our_side"]))
        )
        for r in lst
    )
    return w, len(lst), (w - exp) / max(1, len(lst))


tot = rec(ours)
print(f"T6-era ladder: {wl(tot[0], tot[1])}  vsElo {100 * tot[2]:+.1f}pp")
for name, lst in (("T4 (09-20)", t4), ("MB430 (09-09)", mb)):
    t = rec(lst)
    print(f"{name}: {wl(t[0], t[1])} vsElo {100 * t[2]:+.1f}pp")

print("\n== T6-era ladder record by opponent species ON ROSTER / BROUGHT (n>=8)")
spc = collections.Counter(
    s for r in ours for s in set(base(x) for x in r["preview"][other(r["our_side"])])
)
lines = []
for sp, n in spc.most_common():
    if n < 8:
        continue
    on = [
        r for r in ours if sp in {base(x) for x in r["preview"][other(r["our_side"])]}
    ]
    br = [
        r for r in ours if sp in {base(x) for x in r["brought"][other(r["our_side"])]}
    ]
    w, k, v = rec(on)
    wb, kb, vb = rec(br)
    t4on = [
        r for r in t4 if sp in {base(x) for x in r["preview"][other(r["our_side"])]}
    ]
    w4, k4, _ = rec(t4on)
    lines.append(
        (
            w / k,
            sp,
            f"  {sp:16s} on roster {w:2d}-{k - w:<2d} ({100 * w / k:3.0f}%)  "
            f"brought {wb:2d}-{kb - wb:<2d} ({100 * wb / max(1, kb):3.0f}%)   "
            f"| T4 era on roster {w4}-{k4 - w4}",
        )
    )
for _, _, line in sorted(lines):
    print(line)

# ---- local battery: T6tac (tactical_e4) vs 47 held-out rosters, 6 populations
print("\n== local battery (T6tac + 8 guards): by category and worst rosters")
games = []
for f in glob.glob(
    str(REPO / "results_brain_ab_tactical1_e4/*_candidate_tactical1_e4.jsonl")
):
    pop = f.split("/")[-1].split("_candidate")[0]
    for line in open(f):
        g = json.loads(line)
        g["pop"] = pop
        games.append(g)
print("games:", len(games))
cat = collections.defaultdict(list)
for g in games:
    cat[g["category"]].append(g["target"] > 0.5)
for c, v in sorted(cat.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
    print(f"  {c:16s} {100 * sum(v) / len(v):5.1f}%  n={len(v)}")
ros = collections.defaultdict(list)
for g in games:
    ros[(g["opponent"], tuple(g["opponent_roster"]), g["category"])].append(
        g["target"] > 0.5
    )
print("  worst 12 rosters (all populations pooled):")
for (o, roster, c), v in sorted(ros.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))[
    :12
]:
    wr = 100 * sum(v) / len(v)
    print(f"    {wr:5.1f}% n={len(v):3d} {c:14s} {' '.join(roster)}")
# species presence effect in the battery (human-clone populations: closest to people)
print(
    "  species on the opponent roster -> our local win rate (human-clone populations)"
)
hum = [g for g in games if g["pop"].startswith("human")]
base_wr = sum(g["target"] > 0.5 for g in hum) / len(hum)
sp_l = collections.defaultdict(list)
for g in hum:
    for s in set(g["opponent_roster"]):
        sp_l[s].append(g["target"] > 0.5)
out = []
for s, v in sp_l.items():
    if len(v) >= 60:
        out.append((sum(v) / len(v), s, len(v)))
for wr, s, n in sorted(out)[:15]:
    print(f"    {s:16s} {100 * wr:5.1f}% (n={n})  vs overall {100 * base_wr:.1f}%")
