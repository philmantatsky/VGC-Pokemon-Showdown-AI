"""Our Reg M-C ladder rating over every replay we saved (the rating Showdown shows
for us at the start of each game), in blocks of 20 games, with the peaks. Read-only.
Run from the repo root:
    .venv/bin/python results_analysis/threat_first3_20261003/rating_trace.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.ladder_loss_profile import extract_log  # noqa: E402

OUR = "antonius1"


def main() -> None:
    rows = []
    for folder in sorted(ROOT.glob("ladder_replays_mc_*")):
        if not folder.is_dir():
            continue
        for path in folder.glob("*.html"):
            log = extract_log(path) or ""
            ours = re.search(r"\|player\|(p[12])\|" + OUR + r"\|[^|]*\|(\d+)", log)
            battle = re.search(r"regmc-(\d+)", path.name)
            if ours and battle:
                rows.append(
                    (
                        int(battle.group(1)),
                        int(ours.group(2)),
                        f"|win|{OUR}" in log,
                        folder.name.removeprefix("ladder_replays_mc_"),
                    )
                )
    rows.sort()
    print(f"{len(rows)} Reg M-C ladder games with our rating shown")
    for i in range(0, len(rows), 20):
        block = rows[i : i + 20]
        wins = sum(r[2] for r in block)
        reads = ", ".join(sorted({r[3] for r in block}))
        print(
            f"games {i + 1:4d}-{i + len(block):4d}: "
            f"{block[0][1]:4d} -> {block[-1][1]:4d}"
            f"  won {wins}-{len(block) - wins}  {reads}"
        )
    for game, rating, _, read in sorted(rows, key=lambda r: -r[1])[:3]:
        print(f"peak {rating} at game {[r[0] for r in rows].index(game) + 1} ({read})")


if __name__ == "__main__":
    main()
