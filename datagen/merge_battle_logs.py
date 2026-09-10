"""Merge dated scrape directories into one corpus, de-duplicated by battle id.

The scraper writes ``logs_<format>.json`` (battle id -> [upload time, log])
into a fresh directory per run; consumers that take several files
(``logs2trajs``, the prior trainers, ``build_team_weights``) would otherwise
count a replay once per scrape it appears in. Usage:

  .venv/bin/python datagen/merge_battle_logs.py --out battle_logs_top_mc_merged \
      battle_logs_top_mc_20260909 battle_logs_top_mc_20260910
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+", type=Path, help="scrape dirs, oldest first")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    merged: dict[str, dict] = defaultdict(dict)
    seen_per_file: dict[str, int] = {}
    for directory in args.dirs:
        for path in sorted(directory.glob("logs_*.json")):
            payload = json.loads(path.read_text())
            seen_per_file[str(path)] = len(payload)
            # later scrapes win on a collision (same id, possibly a re-upload)
            merged[path.name].update(payload)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, payload in merged.items():
        (args.out / name).write_text(json.dumps(payload))
        print(f"{args.out / name}: {len(payload)} unique replays")
    for path, count in seen_per_file.items():
        print(f"  {path}: {count}")
    total = sum(len(v) for v in merged.values())
    print(f"TOTAL unique replays: {total}")


if __name__ == "__main__":
    main()
