"""Scratch analysis 3: scouting report per opponent archetype (human corpus).

For each archetype: its top lead pairs, what those leads click on turn 1,
the turn-1 Protect / Fake Out / speed-control rates, where their Fake Outs
go, and which of OUR six they tend to target when facing it (if present).
"""

from __future__ import annotations

import collections
import json

from openings_common import HERE, OURS6, archetype, features, load, other

rows = [r for r in load() if r["source"] in ("top", "t6like")]
feats = json.load((HERE / "features_cache.json").open())


def feat(r, side):
    k = f"{r['id']}|{side}"
    if k not in feats:
        feats[k] = features(r, side)
    return feats[k]


by_arch = collections.defaultdict(list)
for r in rows:
    for s in ("p1", "p2"):
        by_arch[archetype(feat(r, s))].append((r, s))

for arch, sides in sorted(by_arch.items(), key=lambda kv: -len(kv[1])):
    n = len(sides)
    print(f"\n######## {arch}  (n={n} rosters)")
    pairs = collections.Counter(" + ".join(sorted(r["leads"][s])) for r, s in sides)
    print(
        "  top leads:",
        "; ".join(f"{p} {100 * c / n:.1f}%" for p, c in pairs.most_common(8)),
    )
    lead_sp = collections.Counter(sp for r, s in sides for sp in r["leads"][s])
    print(
        "  lead species:",
        ", ".join(f"{p} {100 * c / n:.0f}%" for p, c in lead_sp.most_common(10)),
    )
    per = collections.defaultdict(collections.Counter)
    for r, s in sides:
        seen = set()
        for a in r["actions"]["1"][s]:
            if a["user"] in seen:
                continue
            seen.add(a["user"])
            per[a["user"]][a["move"]] += 1
        for sp in r["leads"][s]:
            if sp not in seen:
                per[sp]["<none/switch/cant>"] += 1
    for sp, _ in lead_sp.most_common(6):
        tot = sum(per[sp].values())
        print(
            f"    {sp:16s} "
            + ", ".join(f"{m} {100 * c / tot:.0f}%" for m, c in per[sp].most_common(5))
        )
    # rates
    prot = sum(
        any(
            a["move"]
            in (
                "Protect",
                "Detect",
                "Spiky Shield",
                "Baneful Bunker",
                "King's Shield",
                "Silk Trap",
                "Burning Bulwark",
            )
            for a in r["actions"]["1"][s]
        )
        for r, s in sides
    )
    fo = sum(
        any(a["move"] == "Fake Out" for a in r["actions"]["1"][s]) for r, s in sides
    )
    tw = sum(
        any(a["move"] == "Tailwind" for a in r["actions"]["1"][s]) for r, s in sides
    )
    tr = sum(
        any(a["move"] == "Trick Room" for a in r["actions"]["1"][s]) for r, s in sides
    )
    wg = sum(
        any(a["move"] == "Wide Guard" for a in r["actions"]["1"][s]) for r, s in sides
    )
    setup = sum(
        any(
            a["move"]
            in (
                "Swords Dance",
                "Nasty Plot",
                "Calm Mind",
                "Dragon Dance",
                "Quiver Dance",
                "Shell Smash",
                "Coil",
                "Bulk Up",
                "Belly Drum",
                "Shift Gear",
                "Tidy Up",
                "Victory Dance",
            )
            for a in r["actions"]["1"][s]
        )
        for r, s in sides
    )
    spread = sum(any(a["spread"] for a in r["actions"]["1"][s]) for r, s in sides)
    print(
        f"  turn 1: Protect {100 * prot / n:.0f}% | Fake Out {100 * fo / n:.0f}% "
        f"| Tailwind {100 * tw / n:.0f}% | Trick Room {100 * tr / n:.0f}% | Wide "
        f"Guard {100 * wg / n:.0f}% | setup {100 * setup / n:.0f}% | a spread "
        f"move {100 * spread / n:.0f}%"
    )
    # vs our pieces: what they do to them on turn 1 when our species leads against them
    tgt = collections.Counter()
    tgt_n = collections.Counter()
    for r, s in sides:
        o = other(s)
        ours_on = [sp for sp in r["leads"][o] if sp in OURS6]
        if not ours_on:
            continue
        for sp in ours_on:
            tgt_n[sp] += 1
        for a in r["actions"]["1"][s]:
            if a["target"] in ours_on:
                tgt[(a["target"], a["move"])] += 1
    if tgt_n:
        lines = []
        for sp, c in tgt_n.most_common():
            top = [f"{m} {k}" for (t, m), k in tgt.most_common() if t == sp][:4]
            lines.append(f"{sp} (led vs them {c}x): " + ", ".join(top))
        print("  what they throw at OUR leads on turn 1:")
        for line in lines:
            print("    " + line)
