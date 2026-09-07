"""Pick league finalists from a training log's per-interval eval probes.

Parses the SB3 logger tables for ``eval/heuristic`` and ``eval/bc`` in order,
maps each pair to its checkpoint stem (resume stem + i * save interval) and
prints the finalists: the final checkpoint plus the save with the best
probe sum (if different). Usage:
  .venv/bin/python training/triage_league_log.py league4_043806.log \\
      --save-dir results_league4/saves_fp_hs_wt/reg_mb/seed1 --resume 12779520
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SAVE_INTERVAL = 983040


def parse_probes(text: str) -> list[tuple[float, float]]:
    heur = [float(x) for x in re.findall(r"eval/heuristic\s*\|\s*([0-9.]+)", text)]
    bc = [float(x) for x in re.findall(r"eval/bc\s*\|\s*([0-9.]+)", text)]
    return list(zip(heur, bc))


def finalists(
    probes: list[tuple[float, float]], resume: int, save_dir: Path
) -> list[Path]:
    stems = [resume + (i + 1) * SAVE_INTERVAL for i in range(len(probes))]
    existing = [
        (s, p) for s, p in zip(stems, probes) if (save_dir / f"{s}.zip").exists()
    ]
    if not existing:
        return []
    final = existing[-1][0]
    best = max(existing, key=lambda sp: sp[1][0] + sp[1][1])[0]
    picks = [best, final] if best != final else [final]
    return [save_dir / f"{s}.zip" for s in picks]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path)
    ap.add_argument("--save-dir", type=Path, required=True)
    ap.add_argument("--resume", type=int, default=12779520)
    args = ap.parse_args()
    probes = parse_probes(args.log.read_text(errors="ignore"))
    for i, (h, b) in enumerate(probes):
        print(
            f"save {i + 1} ({args.resume + (i + 1) * SAVE_INTERVAL}): heuristic {h:.2f} bc {b:.2f}"
        )
    for p in finalists(probes, args.resume, args.save_dir):
        print(p)


if __name__ == "__main__":
    main()
