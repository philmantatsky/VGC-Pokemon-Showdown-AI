#!/usr/bin/env python
"""Evolve teams with a trained policy (see python/vgczero/evolve.py).

  python scripts/evolve.py --checkpoint runs/base1/checkpoints/latest.pt --run-dir runs/base1 \
      --population 8 --children 2 --games 400 --generations 20
Seeds the population with the best teams from the run's team_stats.json
(or --teams NAME,NAME,...), uses the top --field teams as the opponent field.
Validate survivors with: node showdown/validate_team.js runs/base1/evolve/population_latest.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from vgczero import data as D  # noqa: E402
from vgczero.evolve import Evolver  # noqa: E402
from vgczero.model import load_model  # noqa: E402
from vgczero.teams import TeamStats  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--run-dir", default=None, help="training run (for team_stats.json)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--teams", default=None)
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--children", type=int, default=2)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--field", type=int, default=64, help="opponent field: top-N rated teams (or random supported)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    D.load()
    model = load_model(args.checkpoint, device=args.device)
    names = D.team_names()
    supported = D.supported_teams()
    ranked = supported
    if args.run_dir and (Path(args.run_dir) / "team_stats.json").exists():
        ts = TeamStats(supported)
        ts.load_state(json.loads((Path(args.run_dir) / "team_stats.json").read_text()))
        top = [t for t, _, _ in ts.top(max(args.population, args.field), min_games=20)]
        ranked = top + [t for t in supported if t not in top]
    seeds = [names.index(t.strip()) for t in args.teams.split(",")] if args.teams else ranked[: args.population]
    field = ranked[: args.field]
    out = args.out or str(Path(args.run_dir or "runs") / "evolve")
    ev = Evolver(model, field, out, args.population, args.children, args.games, args.device, args.seed)
    ev.seed_population(seeds)
    for _ in range(args.generations):
        ev.step()
    print(f"population written to {out}/population_latest.json")


if __name__ == "__main__":
    main()
