"""Scratch (2026-09-28): OPENINGS_RESEARCH_T6.md checked against the ladder games
played after it -- the two 09-27 day reads and the two 09-28 overnight reads -- plus
how opponent forfeits should count, and Wide Guard blocks of our spread moves.

Reuses the research's parser and features (results_analysis/openings_20260927/,
the other session's folder). Run from the repo root:
    .venv/bin/python results_analysis/openings_oos_20260928/oos_check.py > \
        results_analysis/openings_oos_20260928/oos_check.txt
"""

from __future__ import annotations

import collections
import json
import os
import re
import sys
import tempfile
from pathlib import Path

REPO = Path("/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench")
os.environ.setdefault(
    "OPENINGS_DATA", str(Path(tempfile.gettempdir()) / "openings_20260927")
)
sys.path.insert(0, str(REPO / "results_analysis/openings_20260927"))
sys.path.insert(0, str(REPO))

from openings_common import T6_ERA, archetype, features, other, wl  # noqa: E402
from openings_parse import parse, sid  # noqa: E402

from tools.ladder_loss_profile import extract_log  # noqa: E402

OUR = "antonius1"
DAY = {"ours:deployed_T6tac_playbook_t6", "ours:practice1_water_t6"}
NIGHT = {"ours:tactical2_e4", "ours:T6tac_review_guards"}
CLONE_ACCTS = {"sccontrol", "scsme", "scorecardpokemon", "kevdan42"}
RUNNER_UP = {
    "Charizard",
    "Golisopod",
    "Politoed",
    "Archaludon",
    "Grimmsnarl",
    "Farigiraf",
}
WATER = "Blastoise + Farigiraf"
QUIT = re.compile(r"\|-message\|(.+?) (forfeited|lost due to inactivity)")


def era_of(src: str) -> str | None:
    if src in T6_ERA:
        return "research"
    if src in DAY:
        return "day0927"
    if src in NIGHT:
        return "night0928"
    return None


def scan(log: str, ours_side: str) -> dict:
    """Forfeit timing / material, and Wide Guard blocks of our moves."""
    turn, faints, last, blocks, quit_at = 0, {"p1": 0, "p2": 0}, None, [], None
    for line in log.split("\n"):
        p = line.split("|")
        if len(p) < 3:
            continue
        if p[1] == "turn":
            turn = int(p[2])
        elif p[1] == "faint":
            faints[p[2][:2]] += 1
        elif p[1] == "move":
            last = (p[2][:2], p[3])
        elif p[1] == "-activate" and "Wide Guard" in line:
            if last and last[0] == ours_side and p[2][:2] != ours_side:
                blocks.append((turn, last[1]))
        elif quit_at is None and QUIT.match(line):
            quit_at = (turn, dict(faints))
    return {"blocks": blocks, "quit_at": quit_at}


rows = []
for d in sorted(REPO.glob("ladder_replays_mc_*")):
    src = "ours:" + d.name.replace("ladder_replays_mc_", "")
    era = era_of(src)
    if not d.is_dir() or era is None:
        continue
    for f in sorted(d.glob("*.html")):
        log = extract_log(f)
        rec = parse(log) if log else None
        if rec is None:
            continue
        side = next((s for s, n in rec["players"].items() if sid(n) == sid(OUR)), None)
        if side is None:
            continue
        opp = rec["players"].get(other(side), "")
        m = QUIT.search(log or "")
        extra = scan(log or "", side)
        quitter = sid(m.group(1)) if m else None
        quit_turn = extra["quit_at"][0] if extra["quit_at"] else None
        state = None
        if quitter == sid(opp) and extra["quit_at"]:
            left = {s: 4 - n for s, n in extra["quit_at"][1].items()}
            mine, theirs = left[side], left[other(side)]
            state = "ahead" if mine > theirs else "even" if mine == theirs else "behind"
        rec.update(
            source=src,
            era=era,
            id=f.stem,
            our_side=side,
            opp_name=opp,
            win=rec["winner_side"] == side,
            # a free win: the opponent left by turn 1, before any real play; later
            # forfeits are mostly concessions from behind and count as wins
            free=quitter == sid(opp) and quit_turn is not None and quit_turn <= 1,
            quit_state=state,
            wg_blocks=extra["blocks"],
        )
        rows.append(rec)

F = {r["id"]: features(r, other(r["our_side"])) for r in rows}


def rs(lst: list[dict], all_games: bool = False) -> str:
    """Record excluding free wins (the default) or of every game."""
    if not all_games:
        lst = [r for r in lst if not r["free"]]
    return wl(sum(r["win"] for r in lst), len(lst))


def lead(r: dict) -> str:
    return " + ".join(sorted(r["leads"][r["our_side"]]))


def opp_leads(r: dict) -> set[str]:
    return set(r["leads"][other(r["our_side"])])


def roster(r: dict) -> set[str]:
    return set(r["preview"][other(r["our_side"])])


def act(r: dict, turn: int, sp: str) -> str | None:
    side = r["our_side"]
    for a in r["actions"][str(turn)][side]:
        if a["user"] == sp:
            return a["move"]
    for c in r["cant"].get(str(turn), {}).get(side, []):
        if c["user"] == sp:
            return "<cant:" + c["reason"].replace("ability: ", "") + ">"
    return None


ERAS = ["research", "day0927", "night0928"]
new = [r for r in rows if r["era"] != "research"]
print({e: sum(r["era"] == e for r in rows) for e in ERAS})
print("new games, all:", rs(new, True), "  excluding free wins:", rs(new))

print("\n== opponent forfeits / inactivity: when they quit (all T6-era + new)")
q = collections.Counter()
for r in rows:
    if r["quit_state"]:
        q["by turn 1 (free win)" if r["free"] else r["quit_state"]] += 1
print("  ", dict(q))
print("  per read (parsed games -> excluding free wins; a quit at team preview has")
print("  no leads, so the parser drops it: 3 more free wins, 7 in all)")
for src in sorted({r["source"] for r in rows}):
    lst = [r for r in rows if r["source"] == src]
    print(f"      {src[5:]:40s} {rs(lst, True):24s} -> {rs(lst)}")

print("\n== the research's holes (excluding free wins): research 134 / new / both")
tests = {
    "Psychic Terrain roster": lambda r: F[r["id"]]["psychic_terrain"],
    "Indeedee led": lambda r: any(s.startswith("Indeedee") for s in opp_leads(r)),
    "sand setter roster": lambda r: F[r["id"]]["sand_setter"],
    "Tyranitar led": lambda r: "Tyranitar" in opp_leads(r),
    "rain setter roster": lambda r: F[r["id"]]["rain_setter"],
    "rain + Archaludon": lambda r: F[r["id"]]["rain_setter"]
    and "Archaludon" in roster(r),
    "Wide Guard roster": lambda r: F[r["id"]].get("wide_guard"),
    "Volcarona roster": lambda r: "Volcarona" in roster(r),
    "Sinistcha roster": lambda r: "Sinistcha" in roster(r),
    "redirector roster": lambda r: F[r["id"]].get("follow_me")
    or F[r["id"]].get("rage_powder"),
    "Tailwind roster": lambda r: F[r["id"]]["tailwind"],
    "TR setter roster": lambda r: F[r["id"]]["tr_setter"],
    "none of PT/sand/rain": lambda r: not (
        F[r["id"]]["psychic_terrain"]
        or F[r["id"]]["sand_setter"]
        or F[r["id"]]["rain_setter"]
    ),
}
for name, t in tests.items():
    old = [r for r in rows if r["era"] == "research" and t(r)]
    nw = [r for r in new if t(r)]
    print(f"  {name:22s} {rs(old):24s} {rs(nw):24s} {rs(old + nw)}")

print("\n== Water Room lead by opponent archetype: research / new")
ARCHES = ["trick_room", "tailwind", "rain", "sand", "psychic_terrain", "sun", "balance"]
for a in ARCHES:
    old = [
        r
        for r in rows
        if r["era"] == "research" and lead(r) == WATER and archetype(F[r["id"]]) == a
    ]
    nw = [r for r in new if lead(r) == WATER and archetype(F[r["id"]]) == a]
    print(f"  {a:16s} {rs(old):24s} {rs(nw)}")

print("\n== our lead pair, by era")
for e in ERAS:
    c = collections.defaultdict(list)
    for r in rows:
        if r["era"] == e:
            c[lead(r)].append(r)
    print(f"  {e}")
    for k, lst in sorted(c.items(), key=lambda kv: -len(kv[1]))[:6]:
        print(f"      {k:30s} {rs(lst)}")

print("\n== Water Room turns 1-2 (human-preview brains; correlational)")
wr = [r for r in rows if lead(r) == WATER and r["source"] != "ours:deployed_T6"]
t1 = collections.Counter((act(r, 1, "Blastoise"), act(r, 1, "Farigiraf")) for r in wr)
print("  turn 1 (Blastoise, Farigiraf):")
for k, n in t1.most_common(8):
    print(f"      {n:3d}  {k}")
t2 = collections.defaultdict(list)
for r in wr:
    first = next((e for e in r["tr"] if e[2] == "start"), None)
    ours_t1 = bool(first and first[0] == 1 and first[1] == r["our_side"])
    t2[
        (
            ours_t1,
            act(r, 2, "Farigiraf") == "Rain Dance",
            act(r, 2, "Blastoise") == "Water Spout",
        )
    ].append(r)
print("  turn 2 by (our room up T1, Farigiraf Rain Dance, Blastoise Water Spout):")
for k, lst in sorted(t2.items(), key=lambda kv: -len(kv[1])):
    print(f"      {str(k):22s} {rs(lst)}")

print("\n== sand rosters: Rain Dance on turns 1-2? (human-preview brains)")
sand = [
    r for r in rows if F[r["id"]]["sand_setter"] and r["source"] != "ours:deployed_T6"
]


def rain_dance(r: dict) -> bool:
    return any(
        a["move"] == "Rain Dance"
        for t in ("1", "2")
        for a in r["actions"][t][r["our_side"]]
    )


print(
    f"  yes {rs([r for r in sand if rain_dance(r)])}   "
    f"no {rs([r for r in sand if not rain_dance(r)])}"
)

print("\n== Wide Guard blocks of our moves (the wide_guard guard is deployed 09-26)")
hit = [r for r in rows if r["wg_blocks"]]
print(f"  games {len(hit)}: {rs(hit)}")
print("  blocked:", collections.Counter(m for r in hit for _, m in r["wg_blocks"]))
for r in hit:
    turns = sorted({t for t, _ in r["wg_blocks"]})
    result = "W" if r["win"] else "L"
    print(f"      {r['source'][5:]:40s} {result} blocked on turns {turns}")

print("\n== the Baltimore runner-up team (research finding 9)")
for r in rows:
    if sid(r["opp_name"]) in CLONE_ACCTS or len(roster(r) & RUNNER_UP) >= 5:
        print(f"  {r['era']:10s} {r['opp_name']:20s} win={r['win']} free={r['free']}")

json.dump(
    [
        {k: r[k] for k in ("id", "era", "source", "win", "free", "quit_state")}
        | {
            "lead": lead(r),
            "arch": archetype(F[r["id"]]),
            "roster": sorted(roster(r)),
            "opp_leads": sorted(opp_leads(r)),
            "wg_blocks": r["wg_blocks"],
        }
        for r in rows
    ],
    (Path(__file__).parent / "oos_rows.json").open("w"),
    indent=0,
)
