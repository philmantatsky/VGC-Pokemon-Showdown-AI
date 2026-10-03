"""Scratch (2026-10-03): how and when we lose our first Pokemon, over every T6-era
ladder game (the user: "what do u think for the losing a pokemon problem").

Per game: who loses a Pokemon first, and on which turn. For each game where OURS
fainted first: which Pokemon, whether it had acted that turn, how many opposing
attacks hit it that turn (and how many were spread moves), its HP at the start of
the turn, whether our Trick Room was up, whether its set has Protect, and what the
bot had chosen for it that turn (decision logs).
Run from the repo root:
    .venv/bin/python results_analysis/early_faints_20261003/early_faints.py
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.ladder_loss_profile import extract_log  # noqa: E402

OUR = "antonius1"
DIRS = [
    "deployed_T6",
    "deployed_T6_humanpreview1",
    "deployed_T6_humanpreview1_attackcheck",
    "deployed_T6_humanpreview1_throatchop",
    "deployed_T6_humanpreview1_wideguard",
    "deployed_T6ctx",
    "deployed_T6tac",
    "deployed_T6tac_playbook_t6",
    "practice1_water_t6",
    "tactical2_e4",
    "T6tac_review_guards",
    "t6m1",
]
HAS_PROTECT = {"Torkoal", "Venusaur", "Charizard"}  # T6 / T6m sets


def sid(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def hp_frac(field: str) -> float | None:
    field = field.strip()
    if field.startswith("0"):
        return 0.0
    m = re.match(r"(\d+)/(\d+)", field)
    return int(m.group(1)) / int(m.group(2)) if m else None


def decisions(folder: Path) -> dict[tuple[str, int], list]:
    out = {}
    path = folder / "decisions.jsonl"
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        row = json.loads(line)
        m = re.search(r"regmc-(\d+)", row["battle"])
        orders = (row.get("chosen") or {}).get("orders")
        # the turn's move decision comes first; a forced switch after a faint is
        # logged under the same turn and must not overwrite it
        if m and orders:
            out.setdefault((m.group(1), int(row["turn"])), orders)
    return out


games = []
for d in DIRS:
    folder = ROOT / f"ladder_replays_mc_{d}"
    dec = decisions(folder)
    for f in sorted(folder.glob("*.html")):
        log = extract_log(f)
        if not log:
            continue
        players = dict(re.findall(r"\|player\|(p[12])\|([^|]+)\|", log))
        ours = next((s for s, n in players.items() if sid(n) == sid(OUR)), None)
        if ours is None or "|win|" not in log:
            continue
        num = (re.search(r"regmc-(\d+)", f.name) or [None, ""])[1]
        lines = log.split("\n")
        quit_ = re.search(r"\|-message\|(.+?) (forfeited|lost due to inactivity)", log)
        cur, hp, turn, room = {}, {}, 0, False
        start_hp: dict[str, float | None] = {}
        acted: set[str] = set()
        hits: collections.Counter = collections.Counter()
        first = None
        for ln in lines:
            p = ln.split("|")
            if len(p) < 3:
                continue
            tag = p[1]
            if tag == "turn":
                turn = int(p[2])
                acted, hits = set(), collections.Counter()
                start_hp = dict(hp)
            elif tag in ("switch", "drag", "replace") and len(p) >= 5:
                cur[p[2][:3]] = p[3].split(",")[0].split("-Mega")[0]
                hp[p[2][:3]] = hp_frac(p[4])
            elif tag in ("move", "cant"):
                acted.add(p[2][:3])
            elif tag == "-damage" and len(p) >= 4:
                if "[from]" not in ln and p[2].startswith(ours):
                    hits[p[2][:3]] += 1
                v = hp_frac(p[3])
                if v is not None:
                    hp[p[2][:3]] = v
            elif tag == "-heal" and len(p) >= 4:
                v = hp_frac(p[3])
                if v is not None:
                    hp[p[2][:3]] = v
            elif tag == "-fieldstart" and "Trick Room" in ln:
                room = bool(re.search(rf"\[of\] {ours}", ln))
            elif tag == "-fieldend" and "Trick Room" in ln:
                room = False
            elif tag == "faint" and first is None:
                slot = p[2][:3]
                side = "ours" if slot.startswith(ours) else "theirs"
                first = {"side": side, "turn": turn, "species": cur.get(slot, "?")}
                if side == "ours":
                    pos = 0 if slot.endswith("a") else 1
                    orders = dec.get((num, turn))
                    order = orders[pos] if orders and pos < len(orders) else None
                    first.update(
                        acted=slot in acted,
                        hits=hits[slot],
                        start_hp=start_hp.get(slot),
                        our_room=room,
                        has_protect=first["species"] in HAS_PROTECT,
                        chosen=(order or {}).get("id") if order else None,
                    )
        win = f"|win|{OUR}" in log
        games.append(
            {
                "dir": d,
                "win": win,
                "first": first,
                "opp_quit": bool(quit_ and sid(quit_.group(1)) != sid(OUR)),
            }
        )


def rec(lst):
    w = sum(g["win"] for g in lst)
    return f"{w}-{len(lst) - w} ({100 * w / max(1, len(lst)):.0f}%)"


print(f"games {len(games)}: {rec(games)}")
ours_first = [g for g in games if g["first"] and g["first"]["side"] == "ours"]
theirs_first = [g for g in games if g["first"] and g["first"]["side"] == "theirs"]
print(f"we lose a Pokemon first: {len(ours_first)} games, {rec(ours_first)}")
print(f"they lose one first:     {len(theirs_first)} games, {rec(theirs_first)}")
print("\nour first faint, by turn:")
by_turn = collections.defaultdict(list)
for g in ours_first:
    by_turn[min(g["first"]["turn"], 5)].append(g)
for t in sorted(by_turn):
    print(
        f"   turn {t if t < 5 else '5+'}: {len(by_turn[t]):3d} games, {rec(by_turn[t])}"
    )
early = [g for g in ours_first if g["first"]["turn"] <= 3]
print(f"\nour first faint on turns 1-3: {len(early)} games ({rec(early)})")
sp = collections.Counter(g["first"]["species"] for g in early)
print("   which Pokemon:", dict(sp.most_common()))
f = [g["first"] for g in early]
n = len(f)
print(f"   fainted before it acted that turn: {sum(not x['acted'] for x in f)}/{n}")
print(f"   hit by 2+ opposing attacks that turn: {sum(x['hits'] >= 2 for x in f)}/{n}")
print(
    f"   already damaged at the start of the turn (<75% HP): {sum((x['start_hp'] or 1) < 0.75 for x in f)}/{n}"
)
print(f"   under our own Trick Room: {sum(x['our_room'] for x in f)}/{n}")
print(f"   its set has Protect: {sum(x['has_protect'] for x in f)}/{n}")
print(
    "   what the bot had chosen for it:",
    dict(collections.Counter(x["chosen"] for x in f).most_common(8)),
)
print("\n   by Pokemon: before-acting / 2+ hits / under our TR / turn")
for s, _ in sp.most_common():
    xs = [x for x in f if x["species"] == s]
    turns = collections.Counter(x["turn"] for x in xs)
    print(
        f"      {s:11s} {len(xs):3d}: {sum(not x['acted'] for x in xs)} / "
        f"{sum(x['hits'] >= 2 for x in xs)} / {sum(x['our_room'] for x in xs)} / {dict(sorted(turns.items()))}"
    )
