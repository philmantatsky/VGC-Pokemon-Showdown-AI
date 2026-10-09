"""Is the search really running? From a searching run's decision log.

One HEALTH line: battles, decisions by mode (search / error_fallback / skips), the
overrides, the share of the most recent decisions that were NOT searched, and the
search time per decision; then the error texts and the last battles. Exit status 1
when more than a fifth of the last ``--window`` decisions (default 120, never judged
on fewer than 40) were not searched -- the stop rule of the ladder trial of
2026-10-09, where a search that silently stopped (the roster run's dead bridge, a
species without sets, Imprison) was the thing to catch.

Usage (repo root):
  .venv/bin/python evaluation/search_health.py <replay dir> [--window 120] [--quiet]
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def load(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a row being written
    return rows


def main() -> int:
    directory = Path(sys.argv[1])
    window = (
        int(sys.argv[sys.argv.index("--window") + 1]) if "--window" in sys.argv else 120
    )
    rows = [r for r in load(directory / "decisions.jsonl") if "exact_search" in r]
    battles = list(dict.fromkeys(r["battle"] for r in rows))
    modes = collections.Counter()
    errors = collections.Counter()
    elapsed = []
    overrides = 0
    per_battle = collections.defaultdict(collections.Counter)
    for r in rows:
        audit = r["exact_search"]
        schedule = audit.get("schedule") or {}
        mode = schedule.get("mode", "?")
        modes[mode] += 1
        per_battle[r["battle"]][mode] += 1
        fallback = audit.get("decision_fallback")
        if fallback:
            errors[
                f"{fallback.get('error_type')}: {str(fallback.get('error'))[:90]}"
            ] += 1
        if mode != "search":
            for reason in schedule.get("reasons") or []:
                errors[f"{mode}:{reason}"] += 1
        result = audit.get("result") or {}
        if mode == "search" and result.get("elapsed_s") is not None:
            elapsed.append(float(result["elapsed_s"]))
        champion, played = audit.get("champion_actions"), audit.get("actions")
        if (
            mode == "search"
            and champion is not None
            and played is not None
            and played != champion
        ):
            overrides += 1
            per_battle[r["battle"]]["override"] += 1
    recent = rows[-window:]
    recent_modes = collections.Counter(
        (r["exact_search"].get("schedule") or {}).get("mode", "?") for r in recent
    )
    recent_bad = sum(
        n for m, n in recent_modes.items() if m not in ("search", "skip_replacement")
    )
    share = recent_bad / len(recent) if recent else 0.0
    elapsed.sort()

    def q(share_below: float) -> float:
        if not elapsed:
            return float("nan")
        return elapsed[min(len(elapsed) - 1, int(share_below * len(elapsed)))]

    print(
        f"HEALTH battles={len(battles)} decisions={len(rows)} modes={dict(modes)} "
        f"overrides={overrides} recent{len(recent)}_not_searched={share:.1%} "
        f"elapsed p50={q(0.5):.1f}s p90={q(0.9):.1f}s max={q(1.0):.1f}s"
    )
    if "--quiet" not in sys.argv:
        for text, n in errors.most_common(8):
            print(f"  {n:4d}  {text}")
        for battle in battles[-6:]:
            print(f"  {battle.rsplit('-', 1)[-1]}: {dict(per_battle[battle])}")
    return 1 if (len(recent) >= 40 and share > 0.20) else 0


if __name__ == "__main__":
    sys.exit(main())
