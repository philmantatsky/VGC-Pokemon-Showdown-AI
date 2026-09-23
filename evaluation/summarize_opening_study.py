"""Summarize local opening trials without treating tiny cells as promotion evidence."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def summarize(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    cells = defaultdict(list)
    for row in rows:
        cells[(row["category"], row["arm"], row["hidden_sheets"])].append(row)
    result = []
    for (category, arm, hidden), group in sorted(cells.items()):
        result.append(
            {
                "category": category,
                "arm": arm,
                "hidden_sheets": hidden,
                "games": len(group),
                "wins": sum(r["target"] for r in group),
                "opponent_rosters": len({r["opponent"] for r in group}),
                "first_faint": dict(Counter(r["first_faint"] for r in group)),
            }
        )
    return {
        "source": str(path),
        "games": len(rows),
        "cells": result,
        "policy_previews": dict(
            Counter(",".join(r["bring_species"]) for r in rows if r["arm"] == "policy")
        ),
        "caveats": [
            "Independent battle RNG, not paired counterfactual outcomes.",
            "Training-split trials are development evidence, not held-out gates.",
            "Four-game cells cannot establish strength or justify opening rules.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.path)
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
