"""Round-4 league config from the meta-game equilibrium.

Reads a finished meta-game (its ``config.json`` for column -> checkpoint and
``meta_game.json`` for the payoff matrix and the column equilibrium y*) and
writes a ``build_league.py`` config: the deployed brain at the resume stem
plus seeded copies (default 12) at stems 100, 200, ... below the save
interval.

Mixtures. ``nash`` uses y* as is. The 2026-09-07 matrix showed why that is
not enough: one column (the exploiter) beats every row harder than anything
else, so the pure equilibrium is 100% exploiter -- the sparring-partner
overfit rounds 3 and 3b already measured. ``hardness`` (default) weights
each column by how much it beats the DEPLOYED brain (1 - win rate), keeps
every member, then applies the documented overrides with mass
redistributed: the exploiter is capped (a light adversary dose) and each
September human clone is floored (the lever that transferred to ladder).

Usage (from the repo root):
  .venv/bin/python training/make_league4_config.py --meta results_meta_game/round4 \\
      --out training/league4_config.json [--copies 12] [--tr-boost 1.5]
"""

from __future__ import annotations

import argparse
import json
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from training.meta_game import multiplicities  # noqa: E402

DEPLOYED = "results_league/league_champion.zip"
RESUME_STEM = 12779520
HUMAN_FLOOR = {"bc_mix_C": 2, "bc_mix_AC": 2}
EXPLOITER_CAP = {"exploiter_final": 2}


def mixture_weights(report: dict, mode: str, rows_key: str = "deployed") -> list[float]:
    """Column weights before overrides: y* (nash) or hardness vs the deployed row."""
    cols = list(report["cols"])
    if mode == "nash":
        return [float(report["col_equilibrium"][c]) for c in cols]
    if mode != "hardness":
        raise ValueError(f"unknown mixture {mode!r}")
    rows = list(report["rows"])
    matrix = report["payoff_row_win_rate"]
    row = rows.index(rows_key) if rows_key in rows else 0
    losses = [max(1.0 - float(matrix[row][j]), 0.0) for j in range(len(cols))]
    total = sum(losses)
    return [x / total for x in losses] if total > 0 else [1.0 / len(cols)] * len(cols)


def apply_overrides(
    cols: list[str], weights: list[float], copies: int
) -> tuple[dict[str, int], dict[str, str]]:
    """Cap and floor as fractions of the pool, renormalising the other members so
    the copies still sum to ``copies`` (mass is redistributed, not deleted)."""
    w = dict(zip(cols, weights, strict=True))
    notes: dict[str, str] = {}
    fixed: dict[str, float] = {}
    for name, cap in EXPLOITER_CAP.items():
        if name in w and w[name] > cap / copies:
            notes[name] = f"cap {cap} copies (weight was {100 * w[name]:.0f}%)"
            fixed[name] = cap / copies
    for name, floor in HUMAN_FLOOR.items():
        if name in w and w[name] < floor / copies:
            notes[name] = f"floor {floor} copies (weight was {100 * w[name]:.0f}%)"
            fixed[name] = floor / copies
    free = [c for c in cols if c not in fixed]
    free_mass = max(1.0 - sum(fixed.values()), 0.0)
    free_total = sum(w[c] for c in free)
    for c in free:
        w[c] = free_mass * (w[c] / free_total if free_total > 0 else 1.0 / len(free))
    w.update(fixed)
    counts = dict(zip(cols, multiplicities([w[c] for c in cols], copies), strict=True))
    return counts, notes


def build_config(
    meta_dir: _Path,
    copies: int = 12,
    tr_boost: float = 1.5,
    dest: str = "results_league4/saves_fp_hs_wt/reg_mb/seed1",
    weights_dest: str = "data/team_weights_regmb_league4.json",
    mixture: str = "hardness",
) -> dict:
    meta_config = json.loads((meta_dir / "config.json").read_text())
    report = json.loads((meta_dir / "meta_game.json").read_text())
    cols = list(report["cols"])
    weights = mixture_weights(report, mixture)
    counts, overrides = apply_overrides(cols, weights, copies)
    sources: dict[str, str] = {}
    stem = 0
    for name in cols:
        path = meta_config["cols"][name]
        n = counts[name]
        if path == DEPLOYED:
            n -= 1  # the resume checkpoint is the deployed brain's first copy
        for _ in range(max(n, 0)):
            stem += 100
            sources[str(stem)] = path
    sources[str(RESUME_STEM)] = DEPLOYED
    return {
        "comment": (
            f"Round 4 (2026-09-07): pool copies from the meta-game ({meta_dir}), "
            f"mixture={mixture}, {copies} seeded copies; overrides: "
            f"{overrides or 'none'}; TR rosters x{tr_boost} in the opponent team pool; "
            "eval_B and eval_D banned by content."
        ),
        "dest": dest,
        "resume_stem": RESUME_STEM,
        "sources": sources,
        "copies": counts,
        "mixture": mixture,
        "mixture_weights": dict(zip(cols, weights, strict=True)),
        "equilibrium": report["col_equilibrium"],
        "overrides": overrides,
        "weights_source": "data/team_weights_regmb.json",
        "weights_dest": weights_dest,
        "tr_boost": tr_boost,
        "eval_only_roots": ["results_bc/eval_B", "results_bc/eval_D"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", type=_Path, default=_Path("results_meta_game/round4"))
    ap.add_argument("--out", type=_Path, default=_Path("training/league4_config.json"))
    ap.add_argument("--copies", type=int, default=12)
    ap.add_argument("--tr-boost", type=float, default=1.5)
    ap.add_argument("--mixture", choices=("hardness", "nash"), default="hardness")
    args = ap.parse_args()
    config = build_config(args.meta, args.copies, args.tr_boost, mixture=args.mixture)
    args.out.write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {args.out}")
    for name, n in config["copies"].items():
        print(
            f"  {name:18s} {n:2d} copies  (y* {100 * config['equilibrium'][name]:.1f}%)"
        )
    seeded = len(config["sources"]) - 1
    print(f"  seeded stems: {seeded} + resume {RESUME_STEM}")
    print(f"  overrides: {config['overrides'] or 'none'}")


if __name__ == "__main__":
    main()
