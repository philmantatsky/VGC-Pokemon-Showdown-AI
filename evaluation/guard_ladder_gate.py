"""Pre-registered ladder gate for an opt-in guard (2026-10-03, threat_first; the
rule the review guards used on 2026-09-28).

go <battery dir> <mirror dir> <guard>
    Ladder only if the held-out battery is deploy-eligible (pooled upper bound >= 0,
    no population below -3pp), the mirror is not lost (Wilson upper bound >= 50%),
    and the guard's errors stay within 1% of the battery's games.
continue <ladder replay dir> <games so far> <stop wins>
    After the first ladder block: continue unless it won <= <stop wins> of them.
script <battery dir> <min share>
    The turn-1 script alone (2026-10-03, playbook_script_only): the battery as in
    go, playbook / learned-preview errors within 1% of its games, and the script
    changed turn 1 in at least <min share> of them (else nothing would be tested).
    No mirror: its opponent is our own team, whose Armor Tail blocks every Fake Out.

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


def acted_holds(run: Path, guard: str, min_share: float) -> list[str]:
    acted = games = 0
    for tel in run.glob("*.telemetry.jsonl"):
        for line in tel.read_text().splitlines():
            if not line.strip():
                continue
            cell = json.loads(line)
            games += int(cell.get("games", 0))
            acted += int((cell.get("guard_counts") or {}).get(guard, 0))
    share = acted / max(1, games)
    print(f"{guard} changed turn 1 in {acted} of {games} games ({100 * share:.1f}%)")
    return (
        []
        if share >= min_share
        else [f"{guard} acted in {100 * share:.1f}% < {100 * min_share:.0f}%"]
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
    elif mode == "script":
        battery, min_share = Path(sys.argv[2]), float(sys.argv[3])
        hold = (
            battery_holds(battery)
            + error_holds(battery, "playbook")
            + error_holds(battery, "learned_preview")
            + acted_holds(battery, "playbook_opening", min_share)
        )
    else:
        raise SystemExit("mode: go | continue | script")
    if hold:
        print("HOLD:", "; ".join(hold))
        sys.exit(3)
    print("GO")


if __name__ == "__main__":
    main()
