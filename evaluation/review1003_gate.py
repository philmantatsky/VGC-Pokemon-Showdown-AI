"""Pre-registered readings for the user's 2026-10-03 requests (PROJECT_STATUS
2026-10-03, written before any of these runs).

guards <battery dir> <mirror dir> <guard,guard,...>
    The three review guards together: the held-out battery is deploy-eligible
    (pooled upper bound >= 0, no population below -3pp), the mirror is not lost
    (Wilson upper bound >= 50%), and each guard's errors stay within 1% of the
    battery's games.
team <battery dir> <mirror dir>
    T6e (Torkoal Earth Power) on the deployed brain without practice: the battery
    is deploy-eligible and the head-to-head against the brain on T6 is not lost
    (upper bound >= 50%); else the set needs a practice round first.
sheet <battery dir>
    The open-sheet preview: the battery is deploy-eligible and its preview errors
    stay within 1% of the games.

Exit 0 = pass, 3 = hold. Read-only. Deployment is the user's call either way.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.guard_ladder_gate import error_holds  # noqa: E402
from training.t6m_practice_gate import battery_holds, mirror_holds  # noqa: E402


def main() -> None:
    mode = sys.argv[1]
    if mode == "guards":
        battery, mirror = Path(sys.argv[2]), Path(sys.argv[3])
        hold = battery_holds(battery) + mirror_holds(mirror)
        for guard in sys.argv[4].split(","):
            hold += error_holds(battery, guard)
    elif mode == "team":
        hold = battery_holds(Path(sys.argv[2])) + mirror_holds(Path(sys.argv[3]))
    elif mode == "sheet":
        battery = Path(sys.argv[2])
        hold = battery_holds(battery) + error_holds(battery, "sheet_preview")
    else:
        raise SystemExit("mode: guards | team | sheet")
    if hold:
        print("HOLD:", "; ".join(hold))
        sys.exit(3)
    print("PASS")


if __name__ == "__main__":
    main()
