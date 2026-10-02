"""The pre-registered rules of the T6m practice cycle (PROJECT_STATUS 2026-10-02;
the user, 2026-10-01: "if it does well start 15 ladder games and go from there on
your own").

go <battery dir> <mirror dir>
    "Does well" = the candidate's held-out battery against the deployed bot is
    deploy-eligible (pooled upper bound >= 0, no population below -3pp) AND its
    head-to-head against the deployed bot is not lost (Wilson upper bound >= 50%).
continue <ladder replay dir>
    After the 15-game ladder read: continue to 40 games unless it won 4 or fewer.

Exit 0 = go / continue, 3 = hold / stop. Read-only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEPLOY_FLOOR = -0.03  # no population below -3pp
STOP_WINS = 4  # a 15-game read at 4-11 or worse ends ladder play


def battery_holds(run: Path) -> list[str]:
    hold = []
    status = json.loads((run / "status.json").read_text())
    if status.get("phase") != "complete_review_required":
        return [f"{run.name} is not complete: {status}"]
    card = json.loads((run / "scorecard.json").read_text())
    low, high = card["pooled"]["bootstrap_95_ci"]
    worst = min(
        pop["overall"]["roster_mean_delta"] for pop in card["populations"].values()
    )
    print(
        f"{run.name}: pooled {100 * card['pooled']['mean_delta']:+.2f}pp "
        f"[{100 * low:+.2f}, {100 * high:+.2f}], worst population {100 * worst:+.1f}pp"
    )
    if high < 0:
        hold.append("battery: pooled upper bound < 0 (worse than the deployed bot)")
    if worst < DEPLOY_FLOOR:
        hold.append(f"battery: a population at {100 * worst:+.1f}pp (< -3pp)")
    return hold


def mirror_holds(run: Path) -> list[str]:
    result = json.loads((run / "result.json").read_text())
    if not result.get("complete"):
        return [f"{run.name} is not complete"]
    low, high = result["wilson_95"]
    print(
        f"{run.name}: {100 * result['a_win_rate']:.1f}% "
        f"[{100 * low:.1f}, {100 * high:.1f}] over {result['games']} games"
    )
    if high < 0.5:
        return [f"head-to-head lost: upper bound {100 * high:.1f}% < 50%"]
    return []


def ladder_record(replays: Path) -> tuple[int, int]:
    from tools.ladder_loss_profile import extract_log

    wins = games = 0
    for file in sorted(replays.glob("*.html")):
        log = extract_log(file) or ""
        me = file.name.split(" - battle-")[0]
        games += 1
        wins += f"|win|{me}" in log
    return wins, games


def main() -> None:
    mode = sys.argv[1]
    if mode == "go":
        hold = battery_holds(Path(sys.argv[2])) + mirror_holds(Path(sys.argv[3]))
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
