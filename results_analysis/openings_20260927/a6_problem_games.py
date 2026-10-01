"""Scratch analysis 6: compact turn-1/2 summaries of our problem matchups."""

from __future__ import annotations

import sys

from openings_common import T6_ERA, features, load, other

rows = [r for r in load() if r["source"] in T6_ERA and r.get("our_side")]
which = sys.argv[1] if len(sys.argv) > 1 else "sand_lead"


def lead_has(r, key):
    s = other(r["our_side"])
    f = features(r, s)
    return any(sp in f.get(key + "_by", []) for sp in r["leads"][s])


def roster_has(r, key):
    return bool(features(r, other(r["our_side"])).get(key))


sel = {
    "sand_lead": lambda r: lead_has(r, "sand_setter"),
    "pt_lead": lambda r: lead_has(r, "psychic_terrain"),
    "rain": lambda r: roster_has(r, "rain_setter"),
    "wide_guard": lambda r: roster_has(r, "wide_guard"),
    "tr_lead": lambda r: lead_has(r, "tr_setter"),
    "tw_lead": lambda r: lead_has(r, "tailwind"),
}[which]


def acts(r, side, turn):
    out = []
    for a in r["actions"][turn][side]:
        t = a["target"] if a["target_side"] and a["target_side"] != side else ""
        tgt = ">" + t.split("-")[0][:6] if t else ""
        miss = "(miss)" if a["miss"] else ""
        out.append(f"{a['user'].split('-')[0][:6]}:{a['move']}{tgt}{miss}")
    for c in r["cant"][turn][side]:
        why = c["reason"].replace("ability: ", "").replace("move: ", "")
        out.append(f"{c['user'].split('-')[0][:6]}:<{why}>")
    sw = r["switches"][turn][side]
    if sw:
        out.append("in:" + "/".join(x[:6] for x in sw))
    return ", ".join(out)


n = w = 0
for r in rows:
    if not sel(r):
        continue
    n += 1
    me, op = r["our_side"], other(r["our_side"])
    win = r["winner_side"] == me
    w += win
    f = features(r, op)
    faint = [
        f"T{t}:{'US ' if s == me else 'them '}{sp[:8]}"
        for s in (me, op)
        for t, sp in r["faints"][s]
        if t <= 3
    ]
    tr = ",".join(
        f"T{t}{'ours' if s == me else ('theirs' if s else '')}{k[0]}"
        for t, s, k in r["tr"]
        if t <= 5
    )
    wx = ",".join(
        f"T{t}{x[:5]}{'(us)' if s == me else '(them)' if s else ''}"
        for t, x, s, _ in r["weather"]
        if t <= 4
    )
    tag = r["source"].replace("ours:deployed_", "")[:14]
    us_lead = "+".join(x[:5] for x in r["leads"][me])
    us_back = "/".join(x[:5] for x in r["brought"][me][2:])
    them6 = " ".join(x[:9] for x in r["preview"][op])
    them_lead = "+".join(x[:9] for x in r["leads"][op])
    print(
        f"{'WIN ' if win else 'LOSS'} {tag:14s} {r['id'][-10:]} | us {us_lead} "
        f"back {us_back} | them {them6} lead {them_lead}"
    )
    print(f"      T1 us: {acts(r, me, '1')} || them: {acts(r, op, '1')}")
    print(f"      T2 us: {acts(r, me, '2')} || them: {acts(r, op, '2')}")
    print(f"      faints<=T3 {faint}  TR {tr}  weather {wx}  turns {r['turns']}")
print(f"{which}: {w}-{n - w}")
