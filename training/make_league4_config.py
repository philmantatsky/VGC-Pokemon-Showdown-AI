"""Round-4 league config from the meta-game equilibrium.

Reads a finished meta-game (its ``config.json`` for column -> checkpoint and
``meta_game.json`` for the column equilibrium y*), converts y* into integer
copies over the pool (default 12, largest-remainder rounding), applies two
documented overrides -- every September human clone keeps at least one copy
(the lever that transferred to ladder) and the exploiter is capped (rounds
3/3b: a heavy adversary dose overfits) -- and writes a ``build_league.py``
config: the deployed brain at the resume stem plus the seeded copies at
stems 100, 200, ... below the save interval.

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
HUMAN_FLOOR = {"bc_mix_C": 1, "bc_mix_AC": 1}
EXPLOITER_CAP = {"exploiter_final": 2}


def build_config(
    meta_dir: _Path,
    copies: int = 12,
    tr_boost: float = 1.5,
    dest: str = "results_league4/saves_fp_hs_wt/reg_mb/seed1",
    weights_dest: str = "data/team_weights_regmb_league4.json",
) -> dict:
    meta_config = json.loads((meta_dir / "config.json").read_text())
    report = json.loads((meta_dir / "meta_game.json").read_text())
    cols = list(report["cols"])
    weights = [float(report["col_equilibrium"][c]) for c in cols]
    counts = dict(zip(cols, multiplicities(weights, copies), strict=True))
    overrides = {}
    for name, floor in HUMAN_FLOOR.items():
        if name in counts and counts[name] < floor:
            overrides[name] = f"floor {floor} (was {counts[name]})"
            counts[name] = floor
    for name, cap in EXPLOITER_CAP.items():
        if name in counts and counts[name] > cap:
            overrides[name] = f"cap {cap} (was {counts[name]})"
            counts[name] = cap
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
            "Round 4 (2026-09-07): pool copies from the meta-game column equilibrium "
            f"({meta_dir}/meta_game.json), {copies} seeded copies; overrides: "
            f"{overrides or 'none'}; TR rosters x{tr_boost} in the opponent team pool; "
            "eval_B and eval_D banned by content."
        ),
        "dest": dest,
        "resume_stem": RESUME_STEM,
        "sources": sources,
        "copies": counts,
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
    args = ap.parse_args()
    config = build_config(args.meta, args.copies, args.tr_boost)
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
