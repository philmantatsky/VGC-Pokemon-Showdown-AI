"""Summarize live exact-search audits (2026-10-04, the matrix search): how many move
decisions were really searched, how often search fell back to the champion and why,
search latency, and how often the searched choice differed from the policy's own
favourite among the searched candidates.

Reads the ``exact_search`` rows that PolicyPlayer appends to its decision log
(evaluation/mirror_guard_ab.py --a-search writes <output>/a_decisions.jsonl).
Usage (from the repo root):
    .venv/bin/python evaluation/search_audit.py <a_decisions.jsonl> [...]
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def summarize(paths: list[Path]) -> dict:
    modes: collections.Counter[str] = collections.Counter()
    reasons: collections.Counter[str] = collections.Counter()
    elapsed: list[float] = []
    changed = searched = 0
    weights: list[float] = []
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line) if line.strip() else {}
            audit = row.get("exact_search")
            if not audit:
                continue
            schedule = audit.get("schedule") or {}
            mode = schedule.get("mode", "?")
            modes[mode] += 1
            if mode != "search":
                for reason in schedule.get("reasons") or ["?"]:
                    reasons[f"{mode}:{reason}"] += 1
                continue
            result = audit.get("result") or {}
            searched += 1
            if result.get("elapsed_s") is not None:
                elapsed.append(float(result["elapsed_s"]))
            rankings = result.get("rankings") or []
            if rankings:
                weights.append(float(rankings[0].get("score", 0.0)))
                favourite = max(rankings, key=lambda r: float(r.get("prior", 0.0)))
                changed += int(favourite.get("choice") != result.get("choice"))
    elapsed.sort()

    def pct(q: float) -> float | None:
        return (
            elapsed[min(len(elapsed) - 1, int(q * len(elapsed)))] if elapsed else None
        )

    return {
        "decisions": sum(modes.values()),
        "modes": dict(modes),
        "fallback_reasons": dict(reasons.most_common(12)),
        "searched": searched,
        "search_changed_policy_favourite": changed,
        "elapsed_s": {
            "p50": pct(0.5),
            "p90": pct(0.9),
            "max": elapsed[-1] if elapsed else None,
        },
        "mean_weight_of_chosen": sum(weights) / len(weights) if weights else None,
    }


if __name__ == "__main__":
    print(json.dumps(summarize([Path(p) for p in sys.argv[1:]]), indent=2))
