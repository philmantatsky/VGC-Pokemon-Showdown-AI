"""Scratch (2026-10-03): how strong human pilots of our exact six avoid losing a
Pokemon early, vs our bot. Human side: games in battle_logs_players_t6like and
battle_logs_player_dksnnfud where one side brought >= 5 of our six species.
Read-only.
Run from the repo root:
    .venv/bin/python results_analysis/early_faints_20261003/experts_vs_us.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results_analysis/openings_20260927"))
from openings_parse import parse, sid  # noqa: E402

from tools.ladder_loss_profile import extract_log  # noqa: E402

OURS6 = {"Blastoise", "Farigiraf", "Charizard", "Venusaur", "Torkoal", "Incineroar"}
OUR = "antonius1"


def stats(records: list[tuple[dict, str]], label: str) -> None:
    n = len(records)
    first = collections.Counter()
    early = collections.Counter()
    early_species = collections.Counter()
    switch_t2 = 0
    wins = 0
    t1 = collections.Counter()
    for rec, side in records:
        other = "p2" if side == "p1" else "p1"
        win = rec["winner_side"] == side
        wins += win
        f_ours = rec["faints"][side]
        f_theirs = rec["faints"][other]
        ours_first = bool(f_ours) and (not f_theirs or f_ours[0][0] <= f_theirs[0][0])
        first[(ours_first, win)] += 1
        if ours_first and f_ours[0][0] <= 3:
            early[win] += 1
            early_species[f_ours[0][1]] += 1
        if rec["switches"]["1"][side] or rec["switches"]["2"][side]:
            switch_t2 += 1
        for a in rec["actions"]["1"][side]:
            if a["user"] in ("Blastoise", "Farigiraf"):
                tgt = (
                    a["target"] if a["target_side"] and a["target_side"] != side else ""
                )
                t1[f"{a['user']}:{a['move']}"] += 1
    lose_first = sum(v for (o, _), v in first.items() if o)
    lf_w = first[(True, True)]
    print(f"== {label}: {n} games, won {wins} ({100 * wins / max(1, n):.0f}%)")
    print(
        f"   lost a Pokemon first: {lose_first} ({100 * lose_first / max(1, n):.0f}%), "
        f"won {lf_w} of those"
    )
    print(
        f"   lost one first on turns 1-3: {sum(early.values())} "
        f"({100 * sum(early.values()) / max(1, n):.0f}%): {dict(early_species.most_common(5))}"
    )
    print(
        f"   switched on turn 1 or 2: {switch_t2} ({100 * switch_t2 / max(1, n):.0f}%)"
    )
    print(f"   turn-1 moves of Blastoise / Farigiraf: {dict(t1.most_common(8))}")


humans = []
for folder in ("battle_logs_players_t6like", "battle_logs_player_dksnnfud"):
    for path in sorted((ROOT / folder).glob("logs_gen9championsvgc2026regmc*.json")):
        data = json.loads(path.read_text())
        for value in data.values():
            log = value[1] if isinstance(value, list) else value
            rec = parse(log)
            if rec is None:
                continue
            for side in ("p1", "p2"):
                roster = {s.split("-Mega")[0] for s in rec["preview"][side]}
                if len(roster & OURS6) >= 5:
                    humans.append((rec, side))
ours = []
for d in sorted(ROOT.glob("ladder_replays_mc_*")):
    if not d.is_dir() or not any(
        k in d.name for k in ("T6", "t6m1", "tactical2", "practice1")
    ):
        continue
    for f in d.glob("*.html"):
        log = extract_log(f)
        rec = parse(log) if log else None
        if rec is None:
            continue
        side = next((s for s, n in rec["players"].items() if sid(n) == sid(OUR)), None)
        if side:
            ours.append((rec, side))
stats(humans, "human pilots of our six (T6-like)")
stats(ours, "our bot on T6 / T6m (ladder)")
