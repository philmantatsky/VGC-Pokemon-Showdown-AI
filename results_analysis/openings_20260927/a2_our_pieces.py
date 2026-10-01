"""Scratch analysis 2: how humans open with OUR six Pokemon.

For every human side that led one of our species: partner, turn-1 action, win
rate and score above the Elo expectation. Then every lead pair drawn from our
six, and T6-like rosters (>= 3 of our six) by opponent archetype.
"""

from __future__ import annotations

import collections
import itertools
import json

from openings_common import HERE, OURS6, archetype, elo_expect, load, other, wl

rows = load()
human = [r for r in rows if r["source"] in ("top", "t6like")]
feats = json.load((HERE / "features_cache.json").open())


def feat(r: dict, side: str) -> dict:
    k = f"{r['id']}|{side}"
    if k not in feats:
        from openings_common import features

        feats[k] = features(r, side)
    return feats[k]


def first_actions(r: dict, side: str, turn: str = "1") -> dict[str, str]:
    out: dict[str, str] = {}
    for a in r["actions"][turn][side]:
        if a["user"] not in out:
            tgt = (
                a["target"]
                if a["target_side"] and a["target_side"] != side
                else (
                    "ally"
                    if a["target_side"] == side and a["target"] != a["user"]
                    else ""
                )
            )
            out[a["user"]] = a["move"] + (f" -> {tgt}" if tgt and tgt != "ally" else "")
    for c in r["cant"][turn][side]:
        if c["user"] not in out:
            out[c["user"]] = "<cant:" + c["reason"].replace("ability: ", "") + ">"
    return out


def perf(sides: list[tuple[dict, str]]) -> str:
    n = len(sides)
    if n == 0:
        return "-"
    w = sum(r["winner_side"] == s for r, s in sides)
    exp = sum(
        elo_expect(r["rating"].get(s), r["rating"].get(other(s))) for r, s in sides
    )
    return f"{wl(w, n)}  vsElo {100 * (w - exp) / n:+.1f}pp"


# ---- 1. each of our species as a lead
print("=== humans leading each of our species")
for sp in OURS6:
    sides = [(r, s) for r in human for s in ("p1", "p2") if sp in r["leads"][s]]
    on_roster = sum(sp in r["preview"][s] for r in human for s in ("p1", "p2"))
    brought_not_led = [
        (r, s)
        for r in human
        for s in ("p1", "p2")
        if sp in r["brought"][s] and sp not in r["leads"][s]
    ]
    print(
        f"\n## {sp}: on roster {on_roster}, led {len(sides)}: {perf(sides)} | "
        f"brought-not-led {perf(brought_not_led)}"
    )
    partners = collections.Counter(
        next((x for x in r["leads"][s] if x != sp), "?") for r, s in sides
    )
    print("   partners:", ", ".join(f"{p} {c}" for p, c in partners.most_common(10)))
    acts = collections.Counter()
    for r, s in sides:
        a = first_actions(r, s).get(sp, "<switched/none>")
        acts[a.split(" -> ")[0]] += 1
    print(
        "   turn-1:",
        ", ".join(f"{m} {100 * c / len(sides):.0f}%" for m, c in acts.most_common(8)),
    )
    # turn-1 action -> result
    by_act = collections.defaultdict(list)
    for r, s in sides:
        by_act[first_actions(r, s).get(sp, "<switched/none>").split(" -> ")[0]].append(
            (r, s)
        )
    for m, lst in sorted(by_act.items(), key=lambda kv: -len(kv[1]))[:5]:
        print(f"      {m:28s} {perf(lst)}")
    # vs opponent archetype
    by_arch = collections.defaultdict(list)
    for r, s in sides:
        by_arch[archetype(feat(r, other(s)))].append((r, s))
    print(
        "   vs archetype:",
        " | ".join(
            f"{a} {perf(lst).split('  ')[0]}"
            for a, lst in sorted(by_arch.items(), key=lambda kv: -len(kv[1]))
        ),
    )

# ---- 2. lead pairs from our six
print("\n=== human lead pairs drawn from our six")
for a, b in itertools.combinations(OURS6, 2):
    sides = [
        (r, s) for r in human for s in ("p1", "p2") if set(r["leads"][s]) == {a, b}
    ]
    if len(sides) < 5:
        continue
    print(f"  {a} + {b}: {perf(sides)}")

# ---- 3. T6-like rosters
print("\n=== human rosters with >= 3 of our six")
t6like = [
    (r, s)
    for r in human
    for s in ("p1", "p2")
    if len(set(r["preview"][s]) & set(OURS6)) >= 3
]
print("n sides:", len(t6like), perf(t6like))
combo = collections.Counter(
    tuple(sorted(set(r["preview"][s]) & set(OURS6))) for r, s in t6like
)
for c, n in combo.most_common(12):
    print(f"  {' + '.join(c)}: {n}")
players = collections.Counter(r["players"][s] for r, s in t6like)
print("  players:", ", ".join(f"{p} {c}" for p, c in players.most_common(12)))
print("\n  leads of T6-like rosters (lead -> record):")
lead_c = collections.defaultdict(list)
for r, s in t6like:
    lead_c[" + ".join(sorted(r["leads"][s]))].append((r, s))
for k, lst in sorted(lead_c.items(), key=lambda kv: -len(kv[1]))[:15]:
    print(f"    {k:40s} {perf(lst)}")
print("\n  T6-like by opponent archetype -> best leads (n>=3)")
by_arch = collections.defaultdict(lambda: collections.defaultdict(list))
for r, s in t6like:
    by_arch[archetype(feat(r, other(s)))][" + ".join(sorted(r["leads"][s]))].append(
        (r, s)
    )
for a, d in sorted(
    by_arch.items(), key=lambda kv: -sum(len(v) for v in kv[1].values())
):
    tot = [x for v in d.values() for x in v]
    print(f"   [{a}] total {perf(tot)}")
    for k, lst in sorted(d.items(), key=lambda kv: -len(kv[1]))[:6]:
        if len(lst) >= 3:
            print(f"       {k:40s} {perf(lst)}")
