"""Pre-registered ladder gate for an opt-in guard (2026-10-03, threat_first; the
rule the review guards used on 2026-09-28).

go <battery dir> <mirror dir> <guard>
    Ladder only if the held-out battery is deploy-eligible (pooled upper bound >= 0,
    no population below -3pp), the mirror is not lost (Wilson upper bound >= 50%),
    and the guard's errors stay within 1% of the battery's games.
continue <ladder replay dir> <games so far> <stop wins>
    After the first ladder block: continue unless it won <= <stop wins> of them.

Exit 0 = go / continue, 3 = hold / stop. Read-only.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.t6m_practice_gate import (  # noqa: E402
    battery_holds,
    ladder_record,
    mirror_holds,
)


def error_holds(run: Path, guard: str) -> list[str]:
    errors, games = Counter(), 0
    for tel in run.glob("*.telemetry.jsonl"):
        for line in tel.read_text().splitlines():
            if not line.strip():
                continue
            cell = json.loads(line)
            games += int(cell.get("games", 0))
            for key, value in (cell.get("guard_counts") or {}).items():
                if key.startswith(guard) and "error" in key:
                    errors[key] += int(value)
    total = sum(errors.values())
    print(f"{guard} errors: {total} in {games} games {dict(errors)}")
    return (
        [f"guard errors {total}/{games} > 1%"] if total > 0.01 * max(1, games) else []
    )


def main() -> None:
    mode = sys.argv[1]
    if mode == "go":
        battery, mirror, guard = Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
        hold = (
            battery_holds(battery) + mirror_holds(mirror) + error_holds(battery, guard)
        )
    elif mode == "continue":
        replays, games_needed, stop_wins = (
            Path(sys.argv[2]),
            int(sys.argv[3]),
            int(sys.argv[4]),
        )
        wins, games = ladder_record(replays)
        print(f"ladder read: {wins}-{games - wins}")
        hold = []
        if games < games_needed:
            hold.append(f"only {games} of {games_needed} games were played")
        elif wins <= stop_wins:
            hold.append(f"stop rule: {wins} wins in the first {games} games")
    else:
        raise SystemExit("mode: go | continue")
    if hold:
        print("HOLD:", "; ".join(hold))
        sys.exit(3)
    print("GO")


if __name__ == "__main__":
    main()
