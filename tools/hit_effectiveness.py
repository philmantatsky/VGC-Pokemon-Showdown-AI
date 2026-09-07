"""Type-effectiveness scoreboard for a ladder replay directory.

Counts, for our side and the opponents', how many damaging attacks were
super-effective / resisted / immune, and how many of OUR single-target attacks
went into a foe that resists them while a second foe was on the field -- the
firing budget of the `resisted_target` guard (2026-09-06: 15% of our
single-target attacks, both in today's 23 games and the August 125).

Usage (from the repo root):
  .venv/bin/python tools/hit_effectiveness.py ladder_replays_league_20260906 [more dirs...]
"""

from __future__ import annotations

import collections
import glob
import html
import re
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

OUR_NAME = "antonius1"


def analyze(replay_dir: str, ours: str = OUR_NAME) -> dict:
    total = collections.Counter()
    opp = collections.Counter()
    resisted_with_alt = 0
    single_attacks = 0
    games = 0
    for path in glob.glob(f"{replay_dir}/*.html"):
        text = html.unescape(open(path, encoding="utf-8", errors="ignore").read())
        lines = [line for line in text.split("\n") if line.startswith("|")]
        p_us = "p1" if re.search(r"\|player\|p1\|" + re.escape(ours), text) else "p2"
        games += 1
        active: dict[str, dict[str, str]] = {"p1": {}, "p2": {}}
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith(("|switch|", "|drag|")):
                m = re.match(r"\|(?:switch|drag)\|(p[12][ab]): ([^|]+)\|", line)
                if m:
                    active[m.group(1)[:2]][m.group(1)[2]] = m.group(2).split(",")[0]
            if line.startswith("|faint|"):
                m = re.match(r"\|faint\|(p[12][ab]):", line)
                if m:
                    active[m.group(1)[:2]].pop(m.group(1)[2], None)
            if line.startswith("|move|"):
                m = re.match(r"\|move\|(p[12][ab]): [^|]+\|([^|]+)\|(p[12][ab])?", line)
                if m:
                    side = m.group(1)[:2]
                    spread = "[spread]" in line
                    j = i + 1
                    eff = "neutral"
                    while j < len(lines) and not lines[j].startswith(
                        ("|move|", "|turn|", "|switch|")
                    ):
                        if lines[j].startswith("|-supereffective|"):
                            eff = "super"
                            break
                        if lines[j].startswith("|-resisted|"):
                            eff = "resisted"
                            break
                        if lines[j].startswith("|-immune|"):
                            eff = "immune"
                            break
                        j += 1
                    damaged = any(
                        lines[k].startswith("|-damage|")
                        for k in range(i + 1, min(j + 1, len(lines)))
                    )
                    if damaged or eff != "neutral":
                        counter = total if side == p_us else opp
                        counter[eff] += 1
                        counter["attacks"] += 1
                        if side == p_us and not spread and m.group(3):
                            single_attacks += 1
                            foe_side = "p2" if p_us == "p1" else "p1"
                            if eff == "resisted" and len(active[foe_side]) == 2:
                                resisted_with_alt += 1
            i += 1
    return {
        "dir": replay_dir,
        "games": games,
        "ours": dict(total),
        "opponents": dict(opp),
        "single_target_attacks": single_attacks,
        "resisted_with_second_foe": resisted_with_alt,
    }


def render(result: dict) -> str:
    def rate(counter: dict, key: str) -> str:
        return f"{100 * counter.get(key, 0) / max(counter.get('attacks', 0), 1):.0f}%"

    ours, opp = result["ours"], result["opponents"]
    return (
        f"{result['dir']} ({result['games']} games)\n"
        f"  our attacks {ours.get('attacks', 0)}: super {rate(ours, 'super')} "
        f"resisted {rate(ours, 'resisted')} immune {rate(ours, 'immune')}\n"
        f"  opp attacks {opp.get('attacks', 0)}: super {rate(opp, 'super')} "
        f"resisted {rate(opp, 'resisted')}\n"
        f"  our single-target resisted hits with a second foe on the field: "
        f"{result['resisted_with_second_foe']}/{result['single_target_attacks']} "
        f"({100 * result['resisted_with_second_foe'] / max(result['single_target_attacks'], 1):.1f}%)"
    )


if __name__ == "__main__":
    for directory in _sys.argv[1:] or ["ladder_replays_league_20260906"]:
        print(render(analyze(directory)))
