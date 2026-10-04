"""The pre-registered readings of the Earth Power lesson on T6e (PROJECT_STATUS
2026-10-04, written before the run; the user: "start the earth power practice
training but like its pretty obvious just use ep when its super effective on a
pokemon and it does better damage than the rest of the moves and theres no better
switch in").

<sft dir> <battery dir> <head-to-head dir>
    lesson: on the validation positions where Earth Power is a valued attack (at
        least 100 of them), the brain's disagreements with the teacher (its
        favourite attack is not the teacher's best, nor within 0.02) at least halve,
        at the epoch the chain picked (lowest validation cross-entropy);
    better: the head-to-head against the deployed bot (both on T6e with the deployed
        guards) is won -- Wilson 95% lower bound above 50%;
    safe: the held-out battery against the deployed brain on T6e is deploy-eligible
        (pooled upper bound >= 0, no population below -3pp).
    GO = better and safe (the practice-cycle rule: a tie is not enough); NEUTRAL =
    lesson and safe and the head-to-head not lost (upper bound >= 50%) -- the
    lesson took at no measured cost; HOLD otherwise.

Exit 0 = GO, 4 = NEUTRAL, 3 = HOLD. Read-only; deployment is the user's call.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.t6m_practice_gate import battery_holds, mirror_holds  # noqa: E402
from training.t6tac_practice_gate import head_to_head_wins  # noqa: E402

MIN_ROWS = 100
KEEP = 0.5  # the share of the disagreements that may remain


def picked_epoch(log: dict) -> dict:
    return min(log["epochs"], key=lambda e: e["cross_entropy"])


def lesson_holds(sft: Path) -> list[str]:
    log = json.loads((sft / "log.json").read_text())
    before, after = log["val_before"], picked_epoch(log)
    rows = after.get("focus_rows", 0)
    need = 1.0 - KEEP * (1.0 - before["focus_agreement"])

    def change(key: str) -> str:
        return f"{100 * before[key]:.1f}% -> {100 * after[key]:.1f}%"

    print(
        f"lesson (epoch {after['epoch']}): Earth Power agreement "
        f"{change('focus_agreement')} (needs {100 * need:.1f}%) over {rows} "
        f"validation positions; attack share on Earth Power when best "
        f"{change('focus_best_share')}, when worse {change('focus_worse_share')}"
    )
    if rows < MIN_ROWS:
        return [f"lesson unmeasured: {rows} validation positions (< {MIN_ROWS})"]
    if after["focus_agreement"] < need:
        return ["lesson did not take"]
    return []


def main() -> None:
    sft, battery, h2h = (Path(a) for a in sys.argv[1:4])
    lesson = lesson_holds(sft)
    safe = battery_holds(battery)
    better = head_to_head_wins(h2h)
    not_lost = mirror_holds(h2h)
    if not better and not safe:
        print("GO")
        sys.exit(0)
    if not lesson and not safe and not not_lost:
        print("NEUTRAL: the lesson took at no measured cost;", "; ".join(better))
        sys.exit(4)
    print("HOLD:", "; ".join(lesson + safe + better))
    sys.exit(3)


if __name__ == "__main__":
    main()
