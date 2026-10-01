"""Scratch helpers shared by the opening analyses."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

REPO = Path("/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench")
sys.path.insert(0, str(REPO))
HERE = Path(
    os.environ.get("OPENINGS_DATA", Path(tempfile.gettempdir()) / "openings_20260927")
)
HERE.mkdir(parents=True, exist_ok=True)

from poke_env.data import GenData, to_id_str  # noqa: E402

from vgc_bench.src.playbook import OpenSet, opponent_features, set_share  # noqa: E402

FMT = "gen9championsvgc2026regmc"
OURS6 = ["Blastoise", "Farigiraf", "Charizard", "Venusaur", "Torkoal", "Incineroar"]
T6_ERA = {
    "ours:deployed_T6",
    "ours:deployed_T6_humanpreview1",
    "ours:deployed_T6_humanpreview1_attackcheck",
    "ours:deployed_T6_humanpreview1_throatchop",
    "ours:deployed_T6_humanpreview1_wideguard",
    "ours:deployed_T6ctx",
    "ours:deployed_T6tac",
}
# 2026-09-27 daytime: the playbook ladder trial (T6tac + Water Room) and the
# practised brain + Water Room -- same team, used by the team review only
T6_LATER = {"ours:deployed_T6tac_playbook_t6", "ours:practice1_water_t6"}


def load() -> list[dict]:
    return [json.loads(line) for line in (HERE / "games.jsonl").open()]


def other(side: str) -> str:
    return "p2" if side == "p1" else "p1"


def base(species: str) -> str:
    import re

    return re.sub(r"-Mega(-[XYZ])?$", "", species)


def wilson(w: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def wl(w: int, n: int) -> str:
    if n == 0:
        return "  -  "
    lo, hi = wilson(w, n)
    return f"{w}-{n - w} ({100 * w / n:.0f}% [{100 * lo:.0f},{100 * hi:.0f}])"


def elo_expect(ra: int | None, rb: int | None) -> float:
    if not ra or not rb:
        return 0.5
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def sheets_of(rec: dict, side: str) -> dict[str, OpenSet]:
    out = {}
    for sp, s in rec["sheet"][side].items():
        out[sp] = OpenSet(ability=s["ability"], item=s["item"], moves=tuple(s["moves"]))
    return out


@lru_cache(maxsize=None)
def _share(species: str, ability: str | None, move: str | None) -> float:
    return set_share(species, ability=ability, move=move, formatid=FMT)


def has(
    rec: dict,
    side: str,
    species: str,
    *,
    ability: str | None = None,
    move: str | None = None,
    item: str | None = None,
) -> float:
    sheet = rec["sheet"][side].get(species)
    if sheet is not None:
        if ability is not None:
            return float(to_id_str(sheet["ability"]) == ability)
        if move is not None:
            return float(move in {to_id_str(m) for m in sheet["moves"]})
        if item is not None:
            return float(to_id_str(sheet["item"]) == item)
    if item is not None:
        return _item_share(species, item)
    return _share(species, ability, move)


@lru_cache(maxsize=None)
def _item_share(species: str, item: str) -> float:
    from vgc_bench.src.playbook import _base_species, _sets

    sets = _sets(FMT)
    entry = sets.get(to_id_str(species)) or sets.get(_base_species(species)) or {}
    return sum(
        float(s.get("prob", 0.0))
        for s in entry.get("sets", [])
        if to_id_str(s.get("item") or "") == item
    )


def features(rec: dict, side: str) -> dict:
    roster = rec["preview"][side]
    feats = opponent_features(roster, sheets_of(rec, side), formatid=FMT)

    def flag(name: str, thr: float, **what) -> None:
        who = [s for s in roster if has(rec, side, s, **what) >= thr]
        feats[name] = bool(who)
        feats[name + "_by"] = who

    flag("sun_setter", 0.5, ability="drought")
    mega_sun = [
        x
        for x in roster
        if has(rec, side, x, item="charizarditey") >= 0.5
        and x not in feats["sun_setter_by"]
    ]
    if mega_sun:
        feats["sun_setter"] = True
        feats["sun_setter_by"] = feats["sun_setter_by"] + mega_sun
    flag("snow_setter", 0.5, ability="snowwarning")
    flag("grassy_terrain", 0.5, ability="grassysurge")
    flag("electric_terrain", 0.5, ability="electricsurge")
    flag("misty_terrain", 0.5, ability="mistysurge")
    flag("follow_me", 0.5, move="followme")
    flag("rage_powder", 0.5, move="ragepowder")
    flag("taunt", 0.3, move="taunt")
    flag("encore", 0.3, move="encore")
    flag("sleep", 0.3, move="spore")
    flag("perish", 0.3, move="perishsong")
    return feats


def archetype(f: dict) -> str:
    """One primary label (for tables); features stay multi-label."""
    if f["tr_setter"] and f["slow_count"] >= 2:
        return "trick_room"
    if f["rain_setter"]:
        return "rain"
    if f["sand_setter"]:
        return "sand"
    if f["sun_setter"]:
        return "sun"
    if f["snow_setter"]:
        return "snow"
    if f["psychic_terrain"]:
        return "psychic_terrain"
    if f["tailwind"]:
        return "tailwind"
    if f["tr_setter"]:
        return "tr_splash"  # a setter without a slow core: a mode/anti-mode
    return "balance"


@lru_cache(maxsize=None)
def dex(species: str) -> dict:
    return GenData.from_gen(9).pokedex.get(to_id_str(species)) or {}
