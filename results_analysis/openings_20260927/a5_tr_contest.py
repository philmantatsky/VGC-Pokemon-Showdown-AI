"""Scratch analysis 5: the turn-1 Trick Room contest in the human corpus.

Sides that lead a Trick Room setter: does Trick Room go up on turn 1, what
stops it, and what is it worth, by opponent archetype. Then the double-setter
lead (both sides lead a setter): who clicks, who gets the room, who wins.
Also: the value of Fake Out targets on turn 1 (setter vs attacker).
"""

from __future__ import annotations

import collections
import json

from openings_common import HERE, archetype, elo_expect, features, load, other, wl

rows = [r for r in load() if r["source"] in ("top", "t6like")]
feats = json.load((HERE / "features_cache.json").open())


def feat(r, side):
    k = f"{r['id']}|{side}"
    if k not in feats:
        feats[k] = features(r, side)
    return feats[k]


def perf(sides):
    n = len(sides)
    if not n:
        return "-"
    w = sum(r["winner_side"] == s for r, s in sides)
    exp = sum(
        elo_expect(r["rating"].get(s), r["rating"].get(other(s))) for r, s in sides
    )
    return f"{wl(w, n)} vsElo {100 * (w - exp) / n:+.1f}pp"


def tr_clicked(r, side, sp, turn="1"):
    return any(
        a["user"] == sp and a["move"] == "Trick Room" for a in r["actions"][turn][side]
    )


def first_act(r, side, sp, turn="1"):
    for a in r["actions"][turn][side]:
        if a["user"] == sp:
            return a
    for c in r["cant"][turn][side]:
        if c["user"] == sp:
            return {
                "move": "<cant:"
                + c["reason"].replace("ability: ", "").replace("move: ", "")
                + ">",
                "target": "",
            }
    return None


def tr_after_t1(r):
    """Is Trick Room up at the end of turn 1, and whose?"""
    up, owner = False, None
    for t, side, kind in r["tr"]:
        if t != 1:
            continue
        if kind == "start":
            up, owner = True, side
        else:
            up, owner = False, None
    return up, owner


def setters_led(r, side):
    f = feat(r, side)
    return [sp for sp in r["leads"][side] if sp in f.get("tr_setter_by", [])]


# ---- 1. single-setter leads
print("=== sides that LEAD a Trick Room setter (opponent does not)")
clicked, not_clicked = [], []
outcome = collections.Counter()
stoppers = collections.Counter()
for r in rows:
    for s in ("p1", "p2"):
        mine = setters_led(r, s)
        if not mine or setters_led(r, other(s)):
            continue
        if any(tr_clicked(r, s, sp) for sp in mine):
            clicked.append((r, s))
            up, owner = tr_after_t1(r)
            if up and owner == s:
                outcome["up"] += 1
            else:
                sp = (
                    next(sp for sp in mine if tr_clicked(r, s, sp))
                    if any(tr_clicked(r, s, sp) for sp in mine)
                    else mine[0]
                )
                a = first_act(r, s, sp)
                fainted_t1 = any(t == 1 and x == sp for t, x in r["faints"][s])
                stoppers[
                    "setter KO'd turn 1"
                    if fainted_t1
                    else (
                        "clicked but not up (?)"
                        if a and a["move"] == "Trick Room"
                        else str(a["move"] if a else None)
                    )
                ] += 1
                outcome["failed"] += 1
        else:
            not_clicked.append((r, s))
            sp = mine[0]
            a = first_act(r, s, sp)
            fainted_t1 = any(t == 1 and x == sp for t, x in r["faints"][s])
            if fainted_t1 and (a is None):
                stoppers["(no click) setter KO'd before moving"] += 1
            elif a and a["move"].startswith("<cant"):
                stoppers["(tried?) " + a["move"]] += 1
print(
    f"  TR attempted turn 1: {len(clicked)}  up after T1: {outcome['up']}",
    f"({100 * outcome['up'] / max(1, len(clicked)):.0f}%)",
)
print("  what stopped it / other:", dict(stoppers.most_common(12)))
up_s = [(r, s) for r, s in clicked if tr_after_t1(r) == (True, s)]
fail_s = [(r, s) for r, s in clicked if tr_after_t1(r) != (True, s)]
print(f"  TR up T1 -> {perf(up_s)}")
print(f"  TR tried but not up T1 -> {perf(fail_s)}")
print(f"  setter led, TR not clicked T1 -> {perf(not_clicked)}")
print("  by opponent archetype: [TR up T1] | [tried, failed] | [not clicked]")
arch_u = collections.defaultdict(list)
arch_f = collections.defaultdict(list)
arch_n = collections.defaultdict(list)
for r, s in up_s:
    arch_u[archetype(feat(r, other(s)))].append((r, s))
for r, s in fail_s:
    arch_f[archetype(feat(r, other(s)))].append((r, s))
for r, s in not_clicked:
    arch_n[archetype(feat(r, other(s)))].append((r, s))
for a in sorted(set(arch_u) | set(arch_f)):
    tot = len(arch_u[a]) + len(arch_f[a])
    rate = 100 * len(arch_u[a]) / max(1, tot)
    print(f"    vs {a:16s} success {len(arch_u[a])}/{tot} ({rate:.0f}%)")
    print(f"      up: {perf(arch_u[a])} | failed: {perf(arch_f[a])}")
    print(f"      no-click: {perf(arch_n[a])}")

# what did the opponent do on turn 1 against a TR attempt, and did it work?
print("\n  opponent turn-1 answers to a TR attempt (their lead actions), by success")
ans = collections.defaultdict(lambda: [0, 0])
for r, s in clicked:
    o = other(s)
    up = tr_after_t1(r) == (True, s)
    tags = set()
    for a in r["actions"]["1"][o]:
        tgt_setter = a["target"] in setters_led(r, s)
        if a["move"] in ("Fake Out",):
            tags.add("Fake Out -> setter" if tgt_setter else "Fake Out -> partner")
        elif a["move"] in (
            "Taunt",
            "Imprison",
            "Encore",
            "Trick Room",
            "Tailwind",
            "Protect",
            "Detect",
            "Follow Me",
            "Rage Powder",
            "Spore",
            "Sleep Powder",
            "Yawn",
            "Quash",
            "Roar",
            "Whirlwind",
            "Haze",
        ):
            tags.add(a["move"])
        elif tgt_setter:
            tags.add("attack -> setter")
    for c in r["cant"]["1"][o]:
        if (
            "Armor Tail" in c["reason"]
            or "Dazzling" in c["reason"]
            or "Queenly" in c["reason"]
            or "Psychic" in c["reason"]
        ):
            tags.add(
                "their priority blocked (" + c["reason"].replace("ability: ", "") + ")"
            )
    for t in tags or {"none of these"}:
        ans[t][0 if up else 1] += 1
for t, (u, f) in sorted(ans.items(), key=lambda kv: -sum(kv[1])):
    print(
        f"    {t:40s} TR up {u:4d} / failed {f:4d}  ({100 * u / max(1, u + f):.0f}% up)"
    )

# ---- 2. Farigiraf specifically
print("\n=== Farigiraf leads (opponent without a setter lead): TR on turn 1")
fs = [
    (r, s)
    for r, s in clicked
    if "Farigiraf" in r["leads"][s] and tr_clicked(r, s, "Farigiraf")
]
fu = [(r, s) for r, s in fs if tr_after_t1(r) == (True, s)]
print(
    f"  attempts {len(fs)}, up {len(fu)} ({100 * len(fu) / max(1, len(fs)):.0f}%) "
    f" up: {perf(fu)}  failed: {perf([x for x in fs if x not in fu])}"
)
blk = collections.Counter()
for r, s in fs:
    o = other(s)
    for c in r["cant"]["1"][o]:
        blk[c["reason"] + " " + c["move"]] += 1
print("  their moves blocked on turn 1:", dict(blk.most_common(8)))

# ---- 3. double-setter leads
print("\n=== BOTH sides lead a Trick Room setter")
dbl = [r for r in rows if setters_led(r, "p1") and setters_led(r, "p2")]
print("  games:", len(dbl))
cat = collections.defaultdict(list)
for r in dbl:
    for s in ("p1", "p2"):
        me = any(tr_clicked(r, s, sp) for sp in setters_led(r, s))
        them = any(tr_clicked(r, other(s), sp) for sp in setters_led(r, other(s)))
        up, owner = tr_after_t1(r)
        key = (
            ("I click" if me else "I hold")
            + " / "
            + ("they click" if them else "they hold")
            + " -> "
            + (
                "room up (" + ("mine" if owner == s else "theirs") + ")"
                if up
                else "no room"
            )
        )
        cat[key].append((r, s))
for k, lst in sorted(cat.items(), key=lambda kv: -len(kv[1])):
    print(f"    {k:55s} {perf(lst)}")

# ---- 4. Fake Out targeting on turn 1 vs a setter lead
print("\n=== Fake Out on turn 1 when the opponent leads a Trick Room / Tailwind setter")
for kind in ("tr_setter_by", "tailwind_by"):
    grp = collections.defaultdict(list)
    for r in rows:
        for s in ("p1", "p2"):
            o = other(s)
            fo = [
                a
                for a in r["actions"]["1"][s]
                if a["move"] == "Fake Out" and a["target_side"] == o
            ]
            if not fo:
                continue
            setters = [sp for sp in r["leads"][o] if sp in feat(r, o).get(kind, [])]
            if not setters:
                continue
            tgt = fo[0]["target"]
            grp[
                "Fake Out the setter" if tgt in setters else "Fake Out the other"
            ].append((r, s))
    print(f"  opponent leads a {kind.replace('_by', '')}:")
    for k, lst in grp.items():
        print(f"    {k:24s} {perf(lst)}")

# ---- 5. Tailwind vs Trick Room turn 1
print("\n=== one side sets Tailwind turn 1, the other Trick Room turn 1")
tw_tr = collections.defaultdict(list)
for r in rows:
    for s in ("p1", "p2"):
        o = other(s)
        my_tr = any(a["move"] == "Trick Room" for a in r["actions"]["1"][s])
        their_tw = any(a["move"] == "Tailwind" for a in r["actions"]["1"][o])
        if my_tr and their_tw:
            up, owner = tr_after_t1(r)
            tw_tr[
                "TR side, room up" if up and owner == s else "TR side, room failed"
            ].append((r, s))
for k, lst in tw_tr.items():
    print(f"    {k:28s} {perf(lst)}")
