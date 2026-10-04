#!/usr/bin/env python
"""Play on Pokemon Showdown with a vgczero checkpoint.

  # accept challenges on the main server (search on, 16 worlds)
  python scripts/play.py --checkpoint runs/base1/checkpoints/latest.pt --username NAME --password PASS --mode accept
  # local server test: one bot accepts, the other challenges it
  python scripts/play.py --checkpoint ... --username botA --server ws://localhost:8000/showdown/websocket --mode accept
  python scripts/play.py --checkpoint ... --username botB --server ws://localhost:8000/showdown/websocket --mode challenge --opponent botA --games 5
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from vgczero import data as D  # noqa: E402
from vgczero.live.agent import Agent  # noqa: E402
from vgczero.live.client import FORMAT, OFFICIAL_WS, Client  # noqa: E402
from vgczero.live.sets import pool  # noqa: E402
from vgczero.model import load_model  # noqa: E402
from vgczero.search import SearchConfig  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--username", required=True)
    ap.add_argument("--password", default=None)
    ap.add_argument("--server", default=OFFICIAL_WS)
    ap.add_argument("--mode", choices=["accept", "ladder", "challenge"], default="accept")
    ap.add_argument("--opponent", default=None)
    ap.add_argument("--games", type=int, default=0, help="stop after this many battles (0 = forever)")
    ap.add_argument("--teams", default=None, help="comma-separated pool team names (default: a random supported team)")
    ap.add_argument("--team-file", default=None, help="evolve.py population JSON; plays its top --team-top teams")
    ap.add_argument("--team-top", type=int, default=1)
    ap.add_argument("--format", default=FORMAT)
    ap.add_argument("--no-search", action="store_true")
    ap.add_argument("--worlds", type=int, default=16)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--ots", choices=["reject", "accept"], default="reject")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--log-dir", default="runs/live")
    ap.add_argument("--timer", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s %(name)s %(message)s")
    D.load()
    model = load_model(args.checkpoint, device=args.device)
    cfg = None if args.no_search else SearchConfig(worlds=args.worlds, k_self=args.k, k_opp=args.k)
    agent = Agent(model, cfg, device=args.device, seed=args.seed)
    p = pool()
    if args.team_file:
        import json
        rows = json.loads(Path(args.team_file).read_text())
        teams = [r["team"] for r in rows[: args.team_top]]
    elif args.teams:
        teams = [p.team_index(t.strip()) for t in args.teams.split(",")]
    else:
        teams = [D.supported_teams()[args.seed % len(D.supported_teams())]]
    client = Client(agent, args.username, args.password, args.server, args.mode, teams, args.format, args.opponent,
                    args.games, args.ots, args.log_dir, args.timer)
    asyncio.run(client.run())
    wins = sum(r["won"] for r in client.results)
    print(f"{wins}/{len(client.results)} won; {len(agent.disagreements)} legality disagreements logged")
    for d in agent.disagreements[:20]:
        print("  ", d)


if __name__ == "__main__":
    main()
