"""The pre-registered rules of the T6tac practice cycle (PROJECT_STATUS 2026-10-03;
the user: "add threat_first2 to official bot and do that next training").

go <battery dir> <head-to-head dir>
    "Better" = the candidate wins its head-to-head against the deployed bot (both
    with the deployed guards: Wilson 95% lower bound above 50%) AND its held-out
    battery against the deployed bot is deploy-eligible (pooled upper bound >= 0,
    no population below -3pp). The same team on both sides, so a tie is not enough:
    T6ctx and T6tac cleared their predecessors at 59.3% and 58.0%.
continue <ladder replay dir>
    After a 15-game ladder read (run only on the user's word): continue to 40 games
    unless it won 4 or fewer.

Exit 0 = go / continue, 3 = hold / stop. Read-only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.t6m_practice_gate import (  # noqa: E402
    STOP_WINS,
    battery_holds,
    ladder_record,
)


def head_to_head_wins(run: Path) -> list[str]:
    result = json.loads((run / "result.json").read_text())
    if not result.get("complete"):
        return [f"{run.name} is not complete"]
    low, high = result["wilson_95"]
    print(
        f"{run.name}: {100 * result['a_win_rate']:.1f}% "
        f"[{100 * low:.1f}, {100 * high:.1f}] over {result['games']} games"
    )
    if low <= 0.5:
        return [f"head-to-head not won: lower bound {100 * low:.1f}% <= 50%"]
    return []


def main() -> None:
    mode = sys.argv[1]
    if mode == "go":
        hold = battery_holds(Path(sys.argv[2])) + head_to_head_wins(Path(sys.argv[3]))
    elif mode == "continue":
        wins, games = ladder_record(Path(sys.argv[2]))
        print(f"ladder read: {wins}-{games - wins}")
        hold = []
        if games < 15:
            hold.append(f"only {games} of 15 games were played")
        elif wins <= STOP_WINS:
            hold.append(f"stop rule: {wins} wins in the first {games} games")
    else:
        raise SystemExit("mode: go | continue")
    if hold:
        print("HOLD:", "; ".join(hold))
        sys.exit(3)
    print("GO")


if __name__ == "__main__":
    main()
