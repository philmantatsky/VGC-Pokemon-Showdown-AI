"""Ladder-candidate decision as a file-driven state machine.

The rule was pre-registered in PROJECT_STATUS.md (2026-09-20) before any
verdict arm existed: a specialist finalist qualifies if, paired against the
generalist on the chosen team (n=1,000 per arm), its human-clone delta is at
least ``human_floor`` and no PPO arm (frozen, rotation1, rotation2) is below
``ppo_floor``; among qualifiers the highest absolute human-clone rate goes; if
none qualifies the generalist goes. Whoever goes must read at least
``cross_team_bar`` against the human clone (the deployed brain's rate on MB430).

Evaluated lazily so the ladder is reached sooner without changing the outcome:
the human arm of every finalist first, then the PPO arms of the finalists in
descending human rate until one qualifies. ``next_action`` inspects which arm
files exist and says what to run next:

  RUN_HUMAN <stem> | RUN_PPO <stem> | PICK <checkpoint> <reason> | PICK_NONE <reason>

Usage (from the repo root):
  .venv/bin/python tools/ladder_pick.py --root results_gate_battery_brainv1_spec \\
      --save-dir results_brainv1_spec/saves_fp_hs_wt/reg_mc/seed1 \\
      --generalist results_brainv1/saves_fp_hs_wt/reg_mc/seed1/19660800.zip \\
      --finalists 26542080 27525120
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PPO_ARMS = ("frozen", "rotation1", "rotation2")


def arm(root: Path, stem: str, name: str) -> tuple[float, float] | None:
    """(candidate rate, baseline rate) of one paired arm, or None if not run yet."""
    path = root / stem / "screening" / f"battery_{name}.json"
    if not path.exists():
        return None
    arms = json.loads(path.read_text())["arms"]
    return (
        float(arms["distilled_policy"]["win_rate"]),
        float(arms["champion_policy"]["win_rate"]),
    )


def next_action(
    root: Path,
    save_dir: Path,
    generalist: Path,
    finalists: list[str],
    human_floor: float = -1.0,
    ppo_floor: float = -2.0,
    cross_team_bar: float = 0.799,
) -> str:
    human: dict[str, tuple[float, float]] = {}
    for stem in finalists:
        read = arm(root, stem, "human_bc")
        if read is None:
            return f"RUN_HUMAN {stem}"
        human[stem] = read
    notes = []
    for stem in sorted(finalists, key=lambda s: (-human[s][0], s)):
        cand, base = human[stem]
        delta = 100 * (cand - base)
        if delta < human_floor:
            notes.append(f"{stem} human {delta:+.1f}pp")
            continue
        if cand < cross_team_bar:
            notes.append(f"{stem} human rate {100 * cand:.1f} below the cross-team bar")
            continue
        ppo = {name: arm(root, stem, name) for name in PPO_ARMS}
        if any(read is None for read in ppo.values()):
            return f"RUN_PPO {stem}"
        deltas = {name: 100 * (read[0] - read[1]) for name, read in ppo.items() if read}
        worst = min(deltas, key=lambda name: deltas[name])
        if deltas[worst] < ppo_floor:
            notes.append(f"{stem} {worst} {deltas[worst]:+.1f}pp")
            continue
        ppo_text = " ".join(f"{name} {deltas[name]:+.1f}" for name in PPO_ARMS)
        reason = (
            f"specialist {stem}: human {100 * cand:.1f}% "
            f"({delta:+.1f}pp vs generalist); {ppo_text}"
        )
        return f"PICK {save_dir / (stem + '.zip')} {reason}"
    base_rate = max(base for _, base in human.values()) if human else 0.0
    why = "; ".join(notes) if notes else "no finalists"
    if base_rate >= cross_team_bar:
        return (
            f"PICK {generalist} generalist ({100 * base_rate:.1f}% human): "
            f"no specialist qualified ({why})"
        )
    bar = f"{100 * cross_team_bar:.1f}%"
    return f"PICK_NONE nothing clears the cross-team bar {bar} ({why})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--save-dir", type=Path, required=True)
    ap.add_argument("--generalist", type=Path, required=True)
    ap.add_argument("--finalists", nargs="+", required=True, help="checkpoint stems")
    ap.add_argument("--human-floor", type=float, default=-1.0)
    ap.add_argument("--ppo-floor", type=float, default=-2.0)
    ap.add_argument("--cross-team-bar", type=float, default=0.799)
    args = ap.parse_args()
    print(
        next_action(
            args.root,
            args.save_dir,
            args.generalist,
            args.finalists,
            args.human_floor,
            args.ppo_floor,
            args.cross_team_bar,
        )
    )


if __name__ == "__main__":
    main()
