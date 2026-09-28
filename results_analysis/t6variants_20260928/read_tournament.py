"""Reads the T6 set-variant clone tournament exactly as pre-registered
(PROJECT_STATUS, 2026-09-28 11:02, commit a4d098c): per variant and pool,
d = win rate - T6's with a 95% interval for a difference of two proportions
(unpaired): better = lower > 0, worse = upper < 0, else no detectable difference.
A variant qualifies for a practice cycle if better on the full pool, or better on
the sand pool with the full pool's point estimate >= -2pp; both -> the higher
full-pool estimate (tie -> T6m). Descriptive splits below the verdict are not part
of the decision.
Run from the repo root:
    .venv/bin/python results_analysis/t6variants_20260928/read_tournament.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
POOLS = {
    "full": REPO / "results_team_tournament/clone_t6variants_20260928",
    "sand": REPO / "results_team_tournament/clone_t6variants_sand_20260928",
}
VARIANTS = ("T6m", "T6mAS")


def arm(pool: Path, team: str) -> dict:
    return json.loads((pool / f"{team}.json").read_text())["arms"]["champion_policy"]


def diff(a: dict, b: dict) -> tuple[float, float, float]:
    pa, pb = a["wins"] / a["battles"], b["wins"] / b["battles"]
    se = math.sqrt(pa * (1 - pa) / a["battles"] + pb * (1 - pb) / b["battles"])
    d = pa - pb
    return d, d - 1.96 * se, d + 1.96 * se


def verdict(lo: float, hi: float) -> str:
    return "better" if lo > 0 else "worse" if hi < 0 else "no detectable difference"


results: dict[str, dict[str, tuple[float, float, float]]] = {}
for name, pool in POOLS.items():
    base = arm(pool, "T6")
    print(
        f"== {name} pool: T6 {base['wins']}/{base['battles']} = {base['win_rate']:.1%}"
    )
    results[name] = {}
    for v in VARIANTS:
        a = arm(pool, v)
        d, lo, hi = diff(a, base)
        results[name][v] = (d, lo, hi)
        print(
            f"   {v:6s} {a['wins']}/{a['battles']} = {a['win_rate']:.1%}   "
            f"d {100 * d:+.1f}pp [{100 * lo:+.1f}, {100 * hi:+.1f}]  {verdict(lo, hi)}"
        )

qualified = []
for v in VARIANTS:
    d_full, lo_full, _ = results["full"][v]
    _, lo_sand, _ = results["sand"][v]
    if lo_full > 0 or (lo_sand > 0 and d_full >= -0.02):
        qualified.append((d_full, v == "T6m", v))
if qualified:
    pick = max(qualified)[2]
    names = [q[2] for q in qualified]
    print(f"\nDECISION: qualifies {names} -> practice cycle candidate {pick}")
else:
    print("\nDECISION: no variant qualifies -> T6 stays, no practice cycle")


def rate(lst: list[dict]) -> str:
    if not lst:
        return "-"
    wins = sum(r["won"] for r in lst)
    return f"{wins}/{len(lst)} = {wins / len(lst):.1%}"


print("\n-- descriptive (not part of the decision) --")
for name, pool in POOLS.items():
    for team in ("T6", *VARIANTS):
        rows = arm(pool, team)["battle_results"]
        ttar = [r for r in rows if "tyranitar" in r.get("opponent_team", [])]
        rest = [r for r in rows if "tyranitar" not in r.get("opponent_team", [])]
        print(
            f"   {name:4s} {team:6s} Tyranitar brought {rate(ttar):18s} "
            f"not brought {rate(rest)}"
        )
