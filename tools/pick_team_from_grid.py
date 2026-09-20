"""Team pick from the cross-pilot grid and the pilot's own tournament.

The rule was pre-registered in PROJECT_STATUS.md (2026-09-19) before the last
grid cells were read:

1. score(team) = mean of the pilot's win rate against the eval-only human
   clone in the grid and the pilot's tournament win rate (both opponents are
   competent pilots of Reg M-C teams; the scripted-opponent read is reported
   but never used);
2. the top three by score get a confirmation read (pilot alone vs the human
   clone, n=1,000, fresh seed); the highest confirmation read wins, and when
   the top two confirmation reads are within ``tie_pp`` the higher score wins.

Usage (from the repo root):
  .venv/bin/python tools/pick_team_from_grid.py \\
      --grid results_team_grid/brainv1_19660800 \\
      --tournament results_team_tournament/brainv1_19660800 [--confirm]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def grid_reads(grid_dir: Path, arm: str) -> dict[str, float]:
    """team -> the candidate's win rate on ``arm`` from the grid's cell files."""
    reads: dict[str, float] = {}
    for path in sorted(grid_dir.glob(f"T*_{arm}.json")):
        arms = json.loads(path.read_text())["arms"]
        reads[path.stem.split("_")[0]] = float(arms["distilled_policy"]["win_rate"])
    return reads


def tournament_reads(tournament_dir: Path) -> dict[str, float]:
    data = json.loads((tournament_dir / "summary.json").read_text())
    return {row["team"]: float(row["win_rate"]) for row in data["teams"]}


def confirmation_reads(grid_dir: Path) -> dict[str, float]:
    """team -> the pilot's confirmation win rate (confirm_<team>.json)."""
    reads: dict[str, float] = {}
    for path in sorted(grid_dir.glob("confirm_T*.json")):
        arms = json.loads(path.read_text())["arms"]
        reads[path.stem.split("_")[1]] = float(arms["champion_policy"]["win_rate"])
    return reads


def scores(human: dict[str, float], tournament: dict[str, float]) -> dict[str, float]:
    return {t: (human[t] + tournament[t]) / 2 for t in human if t in tournament}


def top(score: dict[str, float], k: int = 3) -> list[str]:
    return sorted(score, key=lambda t: (-score[t], t))[:k]


def final_pick(
    confirm: dict[str, float], score: dict[str, float], tie_pp: float = 2.0
) -> tuple[str, str]:
    """(team, reason) under the pre-registered rule."""
    ranked = sorted(confirm, key=lambda t: (-confirm[t], t))
    if not ranked:
        raise ValueError("no confirmation reads")
    if len(ranked) == 1:
        return ranked[0], "only confirmed team"
    first, second = ranked[0], ranked[1]
    gap_pp = 100 * (confirm[first] - confirm[second])
    if gap_pp >= tie_pp:
        return first, f"confirmation lead {gap_pp:+.1f}pp over {second}"
    winner = max((first, second), key=lambda t: (score[t], t == first))
    return winner, (
        f"confirmation tie ({first} {100 * confirm[first]:.1f} vs {second} "
        f"{100 * confirm[second]:.1f}, gap {gap_pp:.1f}pp < {tie_pp:.1f}): higher score"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--tournament", type=Path, required=True)
    ap.add_argument(
        "--confirm", action="store_true", help="apply step 2 and write team_pick.json"
    )
    ap.add_argument("--tie-pp", type=float, default=2.0)
    args = ap.parse_args()

    human = grid_reads(args.grid, "human")
    scripted = grid_reads(args.grid, "heuristic")
    tournament = tournament_reads(args.tournament)
    score = scores(human, tournament)
    for team in sorted(score, key=lambda t: -score[t]):
        print(
            f"SCORE {team}: {100 * score[team]:5.1f}  (human {100 * human[team]:5.1f}, "
            f"tournament {100 * tournament[team]:5.1f}; scripted "
            f"{100 * scripted.get(team, float('nan')):5.1f} not used)"
        )
    shortlist = top(score)
    print("TOP3 " + " ".join(shortlist))
    if not args.confirm:
        return
    confirm = {t: r for t, r in confirmation_reads(args.grid).items() if t in shortlist}
    for team in sorted(confirm, key=lambda t: -confirm[t]):
        print(f"CONFIRM {team}: {100 * confirm[team]:5.1f}")
    team, reason = final_pick(confirm, score, args.tie_pp)
    print(f"TEAM_PICK {team} ({reason})")
    (args.grid / "team_pick.json").write_text(
        json.dumps(
            {
                "team": team,
                "reason": reason,
                "scores": score,
                "confirmation": confirm,
                "scripted_read": scripted.get(team),
            },
            indent=1,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
