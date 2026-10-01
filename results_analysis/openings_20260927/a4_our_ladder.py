"""Scratch analysis 4: our own T6-era ladder games (134) by lead and opponent."""

from __future__ import annotations

import collections
import json

from openings_common import HERE, T6_ERA, archetype, features, load, other, wl

rows = [r for r in load() if r["source"] in T6_ERA and r.get("our_side")]
print("games:", len(rows))


def rec_str(lst):
    w = sum(r["winner_side"] == r["our_side"] for r in lst)
    return wl(w, len(lst))


def acts(r, side, turn="1"):
    out = {}
    for a in r["actions"][turn][side]:
        if a["user"] not in out:
            t = a["target"] if a["target_side"] != side else ""
            out[a["user"]] = a["move"] + (f">{t}" if t else "")
    for c in r["cant"][turn][side]:
        if c["user"] not in out:
            out[c["user"]] = "<cant:" + c["reason"].replace("ability: ", "") + ">"
    return out


def lead_key(r):
    return " + ".join(sorted(r["leads"][r["our_side"]]))


def cell(k, lst):
    a, b = k.split(" + ")
    rec = rec_str([r for r in lst if lead_key(r) == k]).split(" (")[0]
    return f"{a[:5]}+{b[:5]} {rec}"


F = {}
for r in rows:
    F[r["id"]] = features(r, other(r["our_side"]))

by_lead = collections.defaultdict(list)
for r in rows:
    by_lead[" + ".join(sorted(r["leads"][r["our_side"]]))].append(r)
print("\n== our lead pair (all T6-era brains)")
for k, lst in sorted(by_lead.items(), key=lambda kv: -len(kv[1])):
    eras = collections.Counter(r["source"].replace("ours:deployed_", "") for r in lst)
    print(f"  {k:32s} {rec_str(lst)}   {dict(eras)}")

print("\n== our lead x opponent primary archetype")
arches = collections.Counter(archetype(F[r["id"]]) for r in rows)
for k, lst in sorted(by_lead.items(), key=lambda kv: -len(kv[1])):
    if len(lst) < 5:
        continue
    cells = collections.defaultdict(list)
    for r in lst:
        cells[archetype(F[r["id"]])].append(r)
    print(f"  {k}")
    for a, _ in arches.most_common():
        if cells[a]:
            print(f"      {a:16s} {rec_str(cells[a])}")

print("\n== opponent feature -> our record (all leads) and by top leads")
top_leads = [k for k, v in sorted(by_lead.items(), key=lambda kv: -len(kv[1]))[:4]]
feat_names = [
    "tailwind",
    "tr_setter",
    "rain_setter",
    "sand_setter",
    "psychic_terrain",
    "grassy_terrain",
    "wide_guard",
    "imprison",
    "rock_slide",
    "follow_me",
    "rage_powder",
    "encore",
    "taunt",
    "perish",
    "snow_setter",
    "sun_setter",
]
for fn in feat_names:
    lst = [r for r in rows if F[r["id"]].get(fn)]
    if not lst:
        continue
    cells = " | ".join(cell(k, lst) for k in top_leads)
    print(f"  {fn:16s} {rec_str(lst):26s}  {cells}")
for fn, thr in (("fake_out_users", 2), ("physical_threats", 4), ("fire_weak", 2)):
    lst = [r for r in rows if F[r["id"]].get(fn, 0) >= thr]
    print(f"  {fn}>={thr:<5} {rec_str(lst)}")

print(
    "\n== opponent LEAD features (what was actually on the field turn 1) -> our "
    "record by our lead"
)


def lead_tags(r):
    s = other(r["our_side"])
    tags = set()
    f = F[r["id"]]
    for sp in r["leads"][s]:
        if sp in f.get("tr_setter_by", []):
            tags.add("TR setter")
        if sp in f.get("tailwind_by", []):
            tags.add("Tailwind user")
        if sp in f.get("rain_setter_by", []):
            tags.add("rain setter")
        if sp in f.get("sand_setter_by", []):
            tags.add("sand setter")
        if sp in f.get("psychic_terrain_by", []):
            tags.add("Psychic Terrain")
        if sp in f.get("fake_out_users_by", []):
            tags.add("Fake Out user")
        if sp in f.get("follow_me_by", []) or sp in f.get("rage_powder_by", []):
            tags.add("redirector")
    return tags or {"none of these"}


tag_rows = collections.defaultdict(list)
for r in rows:
    for t in lead_tags(r):
        tag_rows[t].append(r)
for t, lst in sorted(tag_rows.items(), key=lambda kv: -len(kv[1])):
    cells = " | ".join(cell(k, lst) for k in top_leads)
    print(f"  opp leads {t:16s} {rec_str(lst):26s}  {cells}")

print("\n== our turn-1 actions by lead (T6hp onward: human-preview era)")
hp = [r for r in rows if r["source"] != "ours:deployed_T6"]
for k in top_leads:
    lst = [r for r in hp if " + ".join(sorted(r["leads"][r["our_side"]])) == k]
    if not lst:
        continue
    print(f"  {k}: {rec_str(lst)}")
    per = collections.defaultdict(collections.Counter)
    for r in lst:
        a = acts(r, r["our_side"])
        for sp in r["leads"][r["our_side"]]:
            per[sp][a.get(sp, "<switch/none>").split(">")[0]] += 1
    for sp, c in per.items():
        print(f"      {sp:12s} " + ", ".join(f"{m} {n}" for m, n in c.most_common(6)))

print(
    "\n== Trick Room on our side: went up turn 1? and result (games with Farigiraf "
    "lead)"
)
fl = [r for r in rows if "Farigiraf" in r["leads"][r["our_side"]]]


def tr_state(r):
    ours = r["our_side"]
    ev = r["tr"]
    first = next((e for e in ev if e[2] == "start"), None)
    if not first:
        return "never"
    who = "ours" if first[1] == ours else ("theirs" if first[1] else "?")
    return f"{who} T{first[0]}"


c = collections.defaultdict(list)
for r in fl:
    c[tr_state(r)].append(r)
for k, lst in sorted(c.items(), key=lambda kv: -len(kv[1])):
    print(f"   {k:12s} {rec_str(lst)}")

print("\n== first faint on turns 1-2 (ours vs theirs) by our lead")
for k in top_leads:
    lst = [r for r in rows if " + ".join(sorted(r["leads"][r["our_side"]])) == k]
    ours_first = [
        r
        for r in lst
        if r["faints"][r["our_side"]]
        and (
            not r["faints"][other(r["our_side"])]
            or r["faints"][r["our_side"]][0][0]
            <= r["faints"][other(r["our_side"])][0][0]
        )
        and r["faints"][r["our_side"]][0][0] <= 2
    ]
    lost_what = collections.Counter(
        r["faints"][r["our_side"]][0][1] for r in ours_first
    )
    print(
        f"   {k:32s} our mon fainted first by T2: {len(ours_first)}/{len(lst)} "
        f"{dict(lost_what)}  -> {rec_str(ours_first)}"
    )

json.dump(
    {
        r["id"]: {
            "lead": sorted(r["leads"][r["our_side"]]),
            "arch": archetype(F[r["id"]]),
            "win": r["winner_side"] == r["our_side"],
            "src": r["source"],
        }
        for r in rows
    },
    (HERE / "our_games_index.json").open("w"),
)
