#!/usr/bin/env python
"""Export the best teams from a training run (by Wilson lower bound of their
win rate in training games) for laddering, like mikumiku37 rotating among the
ten teams that did best in training.

  python scripts/top_teams.py --run-dir runs/base1 --k 10 --out runs/base1/top_teams.json
  python scripts/play.py --checkpoint runs/base1/checkpoints/latest.pt --team-file runs/base1/top_teams.json --team-top 10 ...
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from vgczero import data as D  # noqa: E402
from vgczero.live.sets import pool  # noqa: E402
from vgczero.teambuilder import TeamBuilder  # noqa: E402
from vgczero.teams import TeamStats  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--min-games", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    D.load()
    stats = json.loads((Path(args.run_dir) / "team_stats.json").read_text())
    ts = TeamStats(stats["pool"])
    ts.load_state(stats)
    p, tb = pool(), TeamBuilder()
    # Evolved teams were registered after the base pool, in this order.
    evolved_path = Path(args.run_dir) / "evolved_teams.json"
    evolved = json.loads(evolved_path.read_text()) if evolved_path.exists() else []
    n_base = len(p.teams)
    rows = []
    for t, lb, g in ts.top(args.k, args.min_games):
        team = p.team(t) if t < n_base else evolved[t - n_base]
        i = ts.pos[t]
        rows.append({"team": tb.with_text({"name": team["name"], "mons": team["mons"]}), "games": g,
                     "wins": float(ts.wins[i]), "win_rate": float(ts.wins[i] / max(1, ts.games[i])), "lb": lb})
        species = ", ".join(m["species"] for m in team["mons"])
        print(f"{team['name']:>8}  lb {lb:.3f}  games {g:6d}  {species}")
    out = Path(args.out or Path(args.run_dir) / "top_teams.json")
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {len(rows)} teams to {out}")


if __name__ == "__main__":
    main()
