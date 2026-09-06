"""Tier-aware verdict for a gate-battery directory.

Why (2026-09-06): at n=1,000 per arm the paired standard error of a
candidate-minus-deployed win-rate delta is 1.2-1.6pp, so a hard "no arm below
the deployed brain by more than 2pp" rule fails a perfectly NEUTRAL candidate
about one time in three across five arms. The same rule at the promotion
tier (n=5,000) fails a neutral candidate ~0.5% of the time. So:

  screening : ADVISORY. An arm inside the confirmation window of the bar is
              flagged CONFIRM (a fresh-seed n=1,500 re-run of that arm
              decides); an arm below the window is a breach; an arm above
              the window is clear. A --confirm reading replaces the screening
              reading for its arm and is judged with the hard rule.
  promotion : HARD. Any arm below the bar fails.

Reads battery_<arm>.json files (per-battle records give the paired SE; falls
back to the unpaired binomial SE when they are absent).

Usage (from the repo root):
  .venv/bin/python evaluation/scorecard_verdict.py <battery_dir> \\
      [--confirm ARM=<fresh-seed battery json>] [--json <out.json>]
"""

from __future__ import annotations

import argparse
import json
import math
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

ARM_ORDER = ("heuristic", "frozen", "rotation1", "rotation2", "human_bc")
HUMAN_ARM = "human_bc"


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def arm_stats(path: _Path) -> dict:
    """Candidate-minus-deployed delta (pp) and its standard error for one arm."""
    data = json.loads(path.read_text())
    cand = data["arms"]["distilled_policy"]
    base = data["arms"]["champion_policy"]
    c_rows = cand.get("battle_results") or []
    b_rows = base.get("battle_results") or []
    n_c, n_b = int(cand["battles"]), int(base["battles"])
    p_c, p_b = cand["wins"] / max(n_c, 1), base["wins"] / max(n_b, 1)
    delta = 100.0 * (p_c - p_b)
    if c_rows and b_rows and len(c_rows) == len(b_rows):
        diffs = [
            (1 if c["won"] else 0) - (1 if b["won"] else 0)
            for c, b in zip(c_rows, b_rows)
        ]
        mean = sum(diffs) / len(diffs)
        var = sum((d - mean) ** 2 for d in diffs) / len(diffs)
        se = 100.0 * math.sqrt(var / len(diffs))
        paired = True
    else:
        se = 100.0 * math.sqrt(
            p_c * (1 - p_c) / max(n_c, 1) + p_b * (1 - p_b) / max(n_b, 1)
        )
        paired = False
    return {
        "delta_pp": delta,
        "se_pp": se,
        "paired": paired,
        "n": n_c,
        "candidate_rate": p_c,
        "deployed_rate": p_b,
    }


def status_for(delta: float, tier: str, bar: float, window: float) -> str:
    delta = round(delta, 6)  # 100*(0.895-0.905) is -1.0000000000000009
    if tier == "promotion":
        return "clear" if delta >= bar else "breach"
    if delta < bar - window:
        return "breach"
    if delta < bar + window:
        return "confirm"
    return "clear"


def weighted_deltas(deltas: dict[str, float]) -> tuple[float, float]:
    """(human-weighted x2, equal-weight) mean delta over the arms present."""
    values = list(deltas.values())
    equal = sum(values) / len(values) if values else 0.0
    if HUMAN_ARM in deltas:
        weighted = (sum(values) + deltas[HUMAN_ARM]) / (len(values) + 1)
    else:
        weighted = equal
    return weighted, equal


def neutral_false_fail(ses: list[float], bar: float) -> float:
    """P(some arm reads below the bar | the candidate is truly neutral)."""
    keep = 1.0
    for se in ses:
        if se > 0:
            keep *= 1.0 - _phi(bar / se)
    return 1.0 - keep


def verdict(
    battery_dir: _Path,
    confirmations: dict[str, _Path] | None = None,
    tier: str | None = None,
    bar: float = -2.0,
    window: float = 1.0,
) -> dict:
    confirmations = confirmations or {}
    scorecard_path = battery_dir / "scorecard.json"
    scorecard = (
        json.loads(scorecard_path.read_text()) if scorecard_path.exists() else {}
    )
    tier = tier or scorecard.get("tier") or "screening"
    arms: dict[str, dict] = {}
    for arm in ARM_ORDER:
        path = battery_dir / f"battery_{arm}.json"
        if not path.exists():
            continue
        stats = arm_stats(path)
        stats["status"] = status_for(stats["delta_pp"], tier, bar, window)
        stats["source"] = "screening"
        if arm in confirmations:
            fresh = arm_stats(confirmations[arm])
            fresh["status"] = status_for(fresh["delta_pp"], "promotion", bar, window)
            fresh["source"] = "confirmation"
            fresh["screening_delta_pp"] = stats["delta_pp"]
            stats = fresh
        arms[arm] = stats
    deltas = {arm: s["delta_pp"] for arm, s in arms.items()}
    weighted, equal = weighted_deltas(deltas)
    breaches = [a for a, s in arms.items() if s["status"] == "breach"]
    confirms = [a for a, s in arms.items() if s["status"] == "confirm"]
    # Advisory tier: a pending confirmation can still move the weighted delta,
    # so it outranks the weighted rule; a breach is final at either tier.
    if breaches:
        outcome = "FAIL"
    elif confirms:
        outcome = "CONFIRM"
    elif weighted < 0:
        outcome = "FAIL"
    else:
        outcome = "PASS"
    return {
        "battery_dir": str(battery_dir),
        "tier": tier,
        "bar_pp": bar,
        "confirm_window_pp": window if tier != "promotion" else 0.0,
        "arms": arms,
        "weighted_delta_pp": weighted,
        "equal_delta_pp": equal,
        "breaches": breaches,
        "confirm_needed": confirms,
        "neutral_false_fail": neutral_false_fail(
            [s["se_pp"] for s in arms.values()], bar
        ),
        "outcome": outcome,
    }


def render(result: dict) -> str:
    lines = [
        f"{result['battery_dir']}  tier={result['tier']}  bar={result['bar_pp']:+.1f}pp"
        + (
            f"  confirm-window ±{result['confirm_window_pp']:.1f}pp"
            if result["confirm_window_pp"]
            else ""
        )
    ]
    for arm, s in result["arms"].items():
        extra = (
            f"  (screening read {s['screening_delta_pp']:+.1f})"
            if s.get("source") == "confirmation"
            else ""
        )
        lines.append(
            f"  {arm:10s} {s['delta_pp']:+5.1f} ± {s['se_pp']:.1f}pp "
            f"({100 * s['candidate_rate']:.1f} v {100 * s['deployed_rate']:.1f}, "
            f"n={s['n']}, {'paired' if s['paired'] else 'unpaired'})  "
            f"{s['status'].upper()}{extra}"
        )
    lines.append(
        f"  weighted (human x2) {result['weighted_delta_pp']:+.2f}pp   "
        f"equal {result['equal_delta_pp']:+.2f}pp"
    )
    lines.append(
        f"  a truly neutral candidate would breach some arm with p = "
        f"{100 * result['neutral_false_fail']:.0f}% at these standard errors"
    )
    if result["confirm_needed"]:
        lines.append(
            f"  confirmation needed (fresh seed, n=1,500): {result['confirm_needed']}"
        )
    lines.append(f"  VERDICT: {result['outcome']}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("battery_dir", type=_Path)
    ap.add_argument("--tier", choices=("screening", "promotion"), default=None)
    ap.add_argument("--bar", type=float, default=-2.0, help="regression bar in pp")
    ap.add_argument(
        "--window", type=float, default=1.0, help="screening confirmation window in pp"
    )
    ap.add_argument(
        "--confirm",
        action="append",
        default=[],
        metavar="ARM=PATH",
        help="fresh-seed confirmation JSON that replaces the screening reading for ARM",
    )
    ap.add_argument(
        "--json", type=_Path, default=None, help="also write the verdict here"
    )
    args = ap.parse_args()
    confirmations = {}
    for item in args.confirm:
        arm, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--confirm expects ARM=PATH, got {item!r}")
        confirmations[arm] = _Path(path)
    result = verdict(args.battery_dir, confirmations, args.tier, args.bar, args.window)
    print(render(result))
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
