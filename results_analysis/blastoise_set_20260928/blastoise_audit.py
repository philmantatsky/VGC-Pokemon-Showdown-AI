"""Scratch (2026-09-28): our Mega Blastoise's moves on ladder, for the user's
questions -- "how often do we use ice beam on blastoise ... and how often does it
actually do good damage" and "protect instead of fake out and then dark pulse
instead of ice beam ... is dark better for blastoise in this meta?".

Over the T6-era ladder games (every brain; same team and sets):
1. What Blastoise used, and what each Ice Beam did (KO / share of the target's HP /
   Protect), incl. how often the opponent switched a different Pokemon into it.
2. Move variants vs the current set at every turn Blastoise attacked: the best move
   of each set on the rebuilt position, scored by the real damage calculator
   (Mega Launcher included), with the set-swap audit's scoring
   (results_analysis/openings_20260927/setswap_audit.py) but KOs strictly first.
   Hindsight, one turn at a time: it judges the moveset, not the bot's choices.
3. Fake Out: what each of Blastoise's Fake Outs did; turns Blastoise fainted
   before it moved (where a Protect could have bought a turn).
Run from the repo root:
    .venv/bin/python results_analysis/blastoise_set_20260928/blastoise_audit.py > \
        results_analysis/blastoise_set_20260928/blastoise_audit.txt
"""

from __future__ import annotations

import collections
import logging
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "results_analysis/openings_20260927"))
logging.disable(logging.CRITICAL)

from poke_env.data import GenData, to_id_str  # noqa: E402
from setswap_audit import (  # noqa: E402
    MARGIN,
    active,
    after_move,
    battle_lines,
    best_of,
)

from tools.ladder_loss_profile import extract_log  # noqa: E402
from unit_tests.ladder_position import position  # noqa: E402

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
]
CURRENT = ["waterspout", "waterpulse", "icebeam"]
VARIANTS = {
    "Dark Pulse for Ice Beam": ["waterspout", "waterpulse", "darkpulse"],
    "Dark Pulse for Water Pulse": ["waterspout", "darkpulse", "icebeam"],
    "Aura Sphere for Ice Beam": ["waterspout", "waterpulse", "aurasphere"],
    "Aura Sphere for Water Pulse": ["waterspout", "aurasphere", "icebeam"],
    "Dark Pulse + Aura Sphere": ["waterspout", "darkpulse", "aurasphere"],
}
BLOCKERS = "Armor Tail|Dazzling|Queenly Majesty"
GEN = GenData.from_gen(9)


def base(species: str) -> str:
    return re.sub(r"-Mega(-[XYZ])?$", "", species)


def ice_mult(species: str) -> float:
    dex = GEN.pokedex
    entry = dex.get(to_id_str(species)) or dex.get(to_id_str(base(species))) or {}
    m = 1.0
    for t in entry.get("types", []):
        m *= GEN.type_chart.get(t.upper(), {}).get("ICE", 1.0)
    return m


def hp_of(field: str) -> float | None:
    field = field.strip()
    if field.startswith("0"):
        return 0.0
    m = re.match(r"(\d+)/(\d+)", field)
    return 100.0 * int(m.group(1)) / int(m.group(2)) if m else None


def verdict(old: dict, new: dict) -> str:
    """Guaranteed KOs first; damage (+MARGIN of a foe's HP) only when KOs tie."""
    if new["ko"] != old["ko"]:
        return "new" if new["ko"] > old["ko"] else "old"
    if new["dmg"] >= old["dmg"] + MARGIN:
        return "new"
    if old["dmg"] >= new["dmg"] + MARGIN:
        return "old"
    return "same"


uses = collections.Counter()
ice = []
fake_out = collections.Counter()
faint_before = collections.Counter()
positions = []
games = with_blastoise = 0
for d in DIRS:
    for f in sorted((REPO / f"ladder_replays_mc_{d}").glob("*.html")):
        log = extract_log(f)
        if not log:
            continue
        games += 1
        lines = battle_lines(log, f"|player|p2|{OUR}" in log)
        occ: dict[str, str] = {}
        start: dict[str, str] = {}
        hp: dict[str, float | None] = {}
        turn, acted, seen = 0, set(), False
        for i, line in enumerate(lines):
            p = line.split("|")
            tag = p[1]
            if tag == "turn":
                turn, acted, start = int(p[2]), set(), dict(occ)
            elif tag in ("switch", "drag", "replace") and len(p) >= 5:
                occ[p[2][:3]] = p[3].split(",")[0]
                hp[p[2][:3]] = hp_of(p[4])
                seen |= p[2].startswith("p1") and p[3].startswith("Blastoise")
            elif tag == "detailschange" and len(p) >= 4:
                occ[p[2][:3]] = p[3].split(",")[0]
            elif tag in ("move", "cant") and p[2].startswith("p1"):
                acted.add(p[2][:3])
            if tag == "faint" and p[2].startswith("p1") and "Blastoise" in p[2]:
                if turn > 0 and p[2][:3] not in acted:
                    faint_before["turn 1" if turn == 1 else "turn 2+"] += 1
            if not (
                tag == "move"
                and p[2].startswith("p1")
                and "Blastoise" in p[2]
                and "[from]" not in line
            ):
                if tag in ("-damage", "-heal", "-sethp") and len(p) >= 4:
                    v = hp_of(p[3])
                    if v is not None:
                        hp[p[2][:3]] = v
                continue
            move_id = to_id_str(p[3])
            uses[p[3]] += 1
            effects = after_move(lines, i)
            text = "\n".join("|".join(e) for e in effects)
            if move_id == "fakeout":
                tgt = p[4][:3] if len(p) > 4 else "?"
                turn_rest = "\n".join(lines[i + 1 : i + 60]).split("|turn|")[0]
                if re.search(
                    rf"\|cant\|p1[ab]: Blastoise[^|]*\|ability: ({BLOCKERS})", text
                ):
                    how = "blocked by Armor Tail / Dazzling"
                elif "Psychic Terrain" in text:
                    how = "blocked by Psychic Terrain"
                elif "|-immune|" in text:
                    how = "immune (Ghost)"
                elif re.search(r"\|-activate\|p2[ab][^|]*\|move: Protect", text):
                    how = "Protected"
                elif re.search(rf"\|cant\|{tgt}[^|]*\|flinch", turn_rest):
                    how = "flinched its target"
                else:
                    how = "hit, no flinch"
                fake_out[how] += 1
                continue
            if move_id == "icebeam" and len(p) > 4:
                slot = p[4][:3]
                row = {
                    "aimed": start.get(slot, occ.get(slot, "?")),
                    "hit": occ.get(slot, "?"),
                    "out": "no damage",
                }
                if re.search(r"\|-activate\|p2[ab][^|]*\|move: Protect", text):
                    row["out"] = "blocked by Protect"
                for e in effects:
                    if e[1] == "-damage" and e[2][:3] == slot and len(e) < 5:
                        new, old = hp_of(e[3]), hp.get(slot)
                        dmg = (
                            (old - new) if (old is not None and new is not None) else 0
                        )
                        row["out"] = (
                            "KO"
                            if new == 0
                            else ">=50%"
                            if dmg >= 50
                            else "25-49%"
                            if dmg >= 25
                            else "<25%"
                        )
                        break
                ice.append(row)
            if move_id in CURRENT:
                try:
                    b = position(lines[:i])
                    me = active(b, p[2][:3])
                    foes = [
                        x
                        for x in b.opponent_active_pokemon
                        if x is not None and not x.fainted
                    ]
                    positions.append((b, me, foes))
                except Exception:  # counted below
                    uses["<position failed>"] += 1
        with_blastoise += seen

print(f"games {games}; Blastoise on the field in {with_blastoise}")
print("1. Blastoise's moves:", dict(uses.most_common()))
out = collections.Counter(r["out"] for r in ice)
print(f"   Ice Beam {len(ice)}:", dict(out.most_common()))
switched = [r for r in ice if base(r["aimed"]) != base(r["hit"])]
print(f"   opponent switched a different Pokemon into it: {len(switched)}")
aimed4 = [r for r in ice if ice_mult(r["aimed"]) >= 4]
print(
    f"   aimed at a 4x-weak target: {len(aimed4)}:",
    dict(
        collections.Counter(
            "switch-in" if base(r["aimed"]) != base(r["hit"]) else r["out"]
            for r in aimed4
        )
    ),
)

print(f"\n2. Variants vs the current set over {len(positions)} attacking turns")
for name, moves in VARIANTS.items():
    v = collections.Counter()
    gained = collections.Counter()
    lost = collections.Counter()
    for b, me, foes in positions:
        old, new = best_of(b, me, foes, CURRENT), best_of(b, me, foes, moves)
        if old is None or new is None:
            v["no calc"] += 1
            continue
        v[verdict(old, new)] += 1
        if new["ko"] > old["ko"]:
            gained[new["target"]] += 1
        if old["ko"] > new["ko"]:
            lost[old["target"]] += 1
    print(f"   {name}: better {v['new']} / worse {v['old']} / same {v['same']}")
    print(f"      KOs gained {sum(gained.values())}: {gained.most_common(6)}")
    print(f"      KOs lost   {sum(lost.values())}: {lost.most_common(6)}")

print(f"\n3. Blastoise's Fake Outs {sum(fake_out.values())}:", dict(fake_out))
print("   Blastoise fainted before it moved:", dict(faint_before))
