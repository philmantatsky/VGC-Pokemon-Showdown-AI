"""Scratch (2026-10-03): does the threat guard see the hits that actually knocked our
Pokemon out early? For every first faint of ours on turns 1-3 in the T6-era ladder
games, rebuild the position at the start of that turn (unit_tests/ladder_position.py)
and ask threat_first2.threats() whether the killer was flagged as a knockout threat
to the victim -- (a) as deployed: an unrevealed ability / item count as none; (b) with
the foe's most-likely ability and item from usage (>= 50% of its recorded sets
consistent with its revealed moves). Live settings: usage prior on, Reg M-C format.
Run from the repo root:
    .venv/bin/python results_analysis/threat_first3_20261003/likely_set_audit.py
"""

from __future__ import annotations

import collections
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results_analysis/openings_20260927"))
logging.disable(logging.CRITICAL)
os.environ["VGC_MOVESET_PRIOR"] = "1"

from poke_env.data import to_id_str  # noqa: E402
from setswap_audit import battle_lines  # noqa: E402

from tools.ladder_loss_profile import extract_log  # noqa: E402
from unit_tests.ladder_position import position  # noqa: E402
from vgc_bench.src import threat_first2 as T2  # noqa: E402
from vgc_bench.src.playbook import _sets  # noqa: E402

OUR = "antonius1"
FMT = "gen9championsvgc2026regmc"
ERA = ("T6", "t6m1", "tactical2", "practice1")
SHARE = 0.5


def likely(foe) -> tuple[str | None, str | None]:
    """The foe's most-likely (ability, item) from usage, given its revealed moves."""
    entry = _sets(FMT).get(to_id_str(foe.species)) or _sets(FMT).get(
        to_id_str(getattr(foe, "base_species", "") or "")
    )
    if not entry:
        return None, None
    seen = {m for m in (foe.moves or {})}
    sets = [
        s
        for s in entry.get("sets", [])
        if seen <= {to_id_str(m) for m in s.get("moves") or []}
    ]
    total = sum(s.get("prob", 0.0) for s in sets) or 1.0
    abilities, items = collections.Counter(), collections.Counter()
    for s in sets:
        abilities[to_id_str(s.get("ability") or "")] += s.get("prob", 0.0) / total
        items[to_id_str(s.get("item") or "")] += s.get("prob", 0.0) / total
    ability = max(abilities.items(), key=lambda kv: kv[1], default=(None, 0))
    item = max(items.items(), key=lambda kv: kv[1], default=(None, 0))
    return (
        ability[0] if ability[0] and ability[1] >= SHARE else None,
        item[0] if item[0] and item[1] >= SHARE else None,
    )


@contextmanager
def likely_sets(battle):
    saved = []
    for foe in battle.opponent_active_pokemon:
        if foe is None:
            continue
        ability, item = likely(foe)
        saved.append((foe, foe._ability, foe._item))
        if foe.ability is None and ability:
            foe._ability = ability
        if foe.item in (None, "unknown_item") and item:
            foe._item = item
    try:
        yield
    finally:
        for foe, ability, item in saved:
            foe._ability, foe._item = ability, item


def main() -> None:
    counts = collections.Counter()
    gained = []
    for folder in sorted(ROOT.glob("ladder_replays_mc_*")):
        if not folder.is_dir() or not any(k in folder.name for k in ERA):
            continue
        for path in sorted(folder.glob("*.html")):
            log = extract_log(path) or ""
            ours_p2 = f"|player|p2|{OUR}" in log
            lines = battle_lines(log, ours_p2)  # our side is p1 in these lines
            turn, start, killer, victim = 0, {}, None, None
            last_move = {}
            for i, line in enumerate(lines):
                p = line.split("|")
                if len(p) < 3:
                    continue
                if p[1] == "turn":
                    turn = int(p[2])
                    start[turn] = i
                    if turn > 3:
                        break
                elif p[1] == "move" and len(p) > 4:
                    last_move = {"user": p[2], "move": p[3], "target": p[4]}
                elif p[1] == "faint":
                    if p[2].startswith("p1") and turn in (1, 2, 3) and last_move:
                        if last_move["user"].startswith("p2"):
                            killer, victim = (turn, last_move), p[2]
                    break
            if killer is None:
                continue
            t, move = killer
            try:
                battle = position(lines[: start[t] + 1])
                battle._format = FMT
                plain = T2.threats(battle)
                with likely_sets(battle):
                    imputed = T2.threats(battle)
            except Exception as exc:
                counts[f"error {type(exc).__name__}"] += 1
                continue
            vslot = 0 if victim.startswith("p1a") else 1
            attacker = to_id_str(move["user"].split(": ", 1)[1])
            mid = to_id_str(move["move"])

            def flagged(found):
                return any(
                    pos == vslot and to_id_str(f.species).startswith(attacker[:5])
                    for _, f, pos, _ in found
                )

            counts["early first faints"] += 1
            a, b = flagged(plain), flagged(imputed)
            counts["flagged as deployed"] += a
            counts["flagged with likely ability/item"] += b
            if b and not a:
                gained.append(
                    (
                        folder.name.removeprefix("ladder_replays_mc_"),
                        path.name[-15:-5],
                        t,
                        move["user"],
                        mid,
                    )
                )
    print(dict(counts))
    print("newly flagged killers (read, battle, turn, attacker, move):")
    for g in gained:
        print("  ", g)


if __name__ == "__main__":
    main()
