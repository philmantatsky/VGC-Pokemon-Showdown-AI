"""Pool several head-to-head runs of evaluation/mirror_guard_ab.py (2026-10-04: the
matrix search plays one game at a time per process, so its head-to-head runs as
parallel shards with different seeds).

Sums side A's wins (ties as halves) and games per block and overall, gives the Wilson
95% interval, and summarizes side A's search audits (evaluation/search_audit.py).
Refuses shards whose manifests differ in anything but seed and games.
Usage (from the repo root):
    .venv/bin/python evaluation/pool_mirrors.py <run dir> [<run dir> ...] [--json out]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.mirror_guard_ab import wilson  # noqa: E402
from evaluation.search_audit import summarize  # noqa: E402

VARYING = {"seed", "games", "blocks"}


def pool(runs: list[Path]) -> dict:
    manifests = [json.loads((run / "manifest.json").read_text()) for run in runs]
    first = {k: v for k, v in manifests[0].items() if k not in VARYING}
    for run, manifest in zip(runs, manifests):
        if {k: v for k, v in manifest.items() if k not in VARYING} != first:
            raise ValueError(f"{run} is not the same study as {runs[0]}")
    blocks: dict[str, dict[str, float]] = {}
    wins = games = 0.0
    complete = True
    for run in runs:
        result = json.loads((run / "result.json").read_text())
        complete &= bool(result.get("complete"))
        for block in result["blocks"]:
            key = (
                f"{'hidden' if block['hidden_sheets'] else 'open'} sheets, "
                f"{'A' if block['a_challenges'] else 'B'} challenges"
            )
            row = blocks.setdefault(key, {"a_wins": 0.0, "games": 0})
            row["a_wins"] += block["a_wins"]
            row["games"] += block["games"]
            wins += block["a_wins"]
            games += block["games"]
    for row in blocks.values():
        row["a_win_rate"] = row["a_wins"] / row["games"] if row["games"] else 0.0
    low, high = wilson(wins, int(games))
    audits = [run / "a_decisions.jsonl" for run in runs]
    audits = [path for path in audits if path.exists()]
    return {
        "runs": [str(run) for run in runs],
        "complete": complete,
        "games": int(games),
        "a_wins": wins,
        "a_win_rate": wins / games if games else 0.0,
        "wilson_95": [low, high],
        "blocks": blocks,
        "search_audit": summarize(audits) if audits else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    pooled = pool(args.runs)
    text = json.dumps(pooled, indent=2)
    if args.json is not None:
        args.json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
