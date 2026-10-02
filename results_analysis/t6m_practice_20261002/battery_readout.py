"""Descriptive readout of the T6m candidate's held-out battery (not part of the
gate; PROJECT_STATUS 2026-10-01 23:50): candidate on T6m vs the deployed brain on
T6, by population, by roster category, on the sand rosters (a Sand Stream setter
on the opponent's roster) and by our lead.
Run from the repo root:
    .venv/bin/python results_analysis/t6m_practice_20261002/battery_readout.py \
        <candidate battery dir> [<reference dir>]
"""

from __future__ import annotations

import collections
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAND = re.compile(r"ability:\s*sand\s*stream|@\s*tyranitarite", re.IGNORECASE)


def rows_of(run: Path) -> dict[str, list[dict]]:
    label = json.loads((run / "manifest.json").read_text())["arm_label"]
    out = {}
    for file in sorted(run.glob(f"*_{label}.jsonl")):
        pop = file.name.removesuffix(f"_{label}.jsonl")
        out[pop] = [json.loads(x) for x in file.read_text().splitlines() if x.strip()]
    return out


def rate(rows: list[dict]) -> float:
    return sum(r["target"] for r in rows) / len(rows) if rows else float("nan")


def roster_bootstrap(ref: list[dict], cand: list[dict], seed: int = 20923):
    """Delta of win rates with whole opponent rosters resampled (95%)."""
    by = collections.defaultdict(lambda: [[], []])
    for r in ref:
        by[r["opponent"]][0].append(r["target"])
    for r in cand:
        by[r["opponent"]][1].append(r["target"])
    rosters = sorted(by)
    if not rosters:
        return float("nan"), float("nan"), float("nan")

    def delta(sample: list[str]) -> float:
        a = [x for k in sample for x in by[k][0]]
        b = [x for k in sample for x in by[k][1]]
        return sum(b) / len(b) - sum(a) / len(a)

    rng = random.Random(seed)
    draws = sorted(delta([rng.choice(rosters) for _ in rosters]) for _ in range(4000))
    return delta(rosters), draws[100], draws[3899]


def main() -> None:
    cand_dir = ROOT / sys.argv[1]
    manifest = json.loads((cand_dir / "manifest.json").read_text())
    ref_dir = ROOT / (
        sys.argv[2] if len(sys.argv) > 2 else manifest["reference_arm"].split("/")[0]
    )
    cand, ref = rows_of(cand_dir), rows_of(ref_dir)
    print(f"candidate {cand_dir.name} ({manifest['checkpoint']})")
    print(f"reference {ref_dir.name}\n")
    print("== by population (win rate: reference -> candidate)")
    for pop in cand:
        d, lo, hi = roster_bootstrap(ref[pop], cand[pop])
        print(
            f"   {pop:15s} {rate(ref[pop]):.1%} -> {rate(cand[pop]):.1%}  "
            f"{100 * d:+.1f}pp [{100 * lo:+.1f}, {100 * hi:+.1f}]"
        )
    all_ref = [r for pop in cand for r in ref[pop]]
    all_cand = [r for pop in cand for r in cand[pop]]

    sand = {
        p.name
        for p in (ROOT / "teams/reg_mc").glob("MC*.txt")
        if SAND.search(p.read_text())
    }
    print("\n== sand rosters (a Sand Stream setter on their team) vs the rest")
    for name, keep in (("sand", True), ("rest", False)):
        a = [r for r in all_ref if (r["opponent"] in sand) == keep]
        b = [r for r in all_cand if (r["opponent"] in sand) == keep]
        d, lo, hi = roster_bootstrap(a, b)
        n = len({r["opponent"] for r in a})
        print(
            f"   {name:5s} {n:2d} rosters, {len(b):4d} games: {rate(a):.1%} -> "
            f"{rate(b):.1%}  {100 * d:+.1f}pp [{100 * lo:+.1f}, {100 * hi:+.1f}]"
        )

    print("\n== by roster category")
    for cat in sorted({r["category"] for r in all_cand}):
        a = [r for r in all_ref if r["category"] == cat]
        b = [r for r in all_cand if r["category"] == cat]
        d, lo, hi = roster_bootstrap(a, b)
        print(
            f"   {cat:18s} {rate(a):.1%} -> {rate(b):.1%}  "
            f"{100 * d:+.1f}pp [{100 * lo:+.1f}, {100 * hi:+.1f}]"
        )

    print("\n== our lead (share of games, win rate): reference | candidate")
    leads = collections.Counter()
    for rows, side in ((all_ref, 0), (all_cand, 1)):
        for r in rows:
            leads[(" + ".join(sorted(r.get("lead_species") or [])), side)] += 1
    names = sorted({k for k, _ in leads}, key=lambda k: -leads[(k, 1)])
    for k in names[:6]:
        a = [r for r in all_ref if " + ".join(sorted(r.get("lead_species") or [])) == k]
        b = [
            r for r in all_cand if " + ".join(sorted(r.get("lead_species") or [])) == k
        ]
        print(
            f"   {k:26s} {len(a) / len(all_ref):5.1%} {rate(a):6.1%} | "
            f"{len(b) / len(all_cand):5.1%} {rate(b):6.1%}"
        )

    print("\n== worst and best rosters for the candidate (delta, games per arm)")
    per = {}
    for name in {r["opponent"] for r in all_cand}:
        a = [r for r in all_ref if r["opponent"] == name]
        b = [r for r in all_cand if r["opponent"] == name]
        per[name] = (rate(b) - rate(a), len(b), b[0]["category"], name in sand)
    ranked = sorted(per.items(), key=lambda kv: kv[1][0])
    for name, (d, n, cat, is_sand) in ranked[:5] + ranked[-5:]:
        tag = " sand" if is_sand else ""
        print(f"   {name:12s} {cat:18s}{tag:5s} {100 * d:+.1f}pp ({n})")


if __name__ == "__main__":
    main()
