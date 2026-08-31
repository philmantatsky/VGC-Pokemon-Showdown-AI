"""Run real battles with the guard stack on and report what it caught.

Replayed states cannot be used for this: LogReader suppresses |request| messages, so
available_moves is empty and `pass` is the only legal action. Only live battles have a
real action space.
"""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from poke_env.player import SimpleHeuristicsPlayer
from poke_env.ps_client import ServerConfiguration
from torch import device

from vgc_bench.src.callback import Callback
from vgc_bench.src.policy_player import BatchPolicyPlayer, PolicyPlayer
from vgc_bench.src.teams import RandomTeamBuilder
from vgc_bench.src.utils import format_map

ap = argparse.ArgumentParser()
ap.add_argument(
    "--checkpoint",
    default="results_mb_ourteam_64opp/saves_bc_sp/reg_mb/64_teams/seed1/3932160.zip",
)
ap.add_argument("--n_battles", type=int, default=30)
ap.add_argument("--port", type=int, default=7400)
ap.add_argument("--no_guards", action="store_true")
ap.add_argument(
    "--search", action="store_true", help="use the one-ply matrix-game search"
)
args = ap.parse_args()

PolicyPlayer.use_knowledge_guards = not args.no_guards
PolicyPlayer.use_search = args.search
PolicyPlayer.use_knowledge_obs = True
PolicyPlayer.use_moveset_prior = True
PolicyPlayer.guard_fire_counts.clear()

srv = ServerConfiguration(
    f"ws://localhost:{args.port}/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)
fmt = format_map["mb"]

ours = BatchPolicyPlayer(
    server_configuration=srv,
    battle_format=fmt,
    log_level=40,
    max_concurrent_battles=4,
    accept_open_team_sheet=True,
    open_timeout=None,
    team=RandomTeamBuilder(1, 1, "mb", [Path("teams/reg_mb/our_team.txt")]),
)
ours.set_policy(Path(args.checkpoint), device("mps"))

opp = SimpleHeuristicsPlayer(
    server_configuration=srv,
    battle_format=fmt,
    log_level=40,
    max_concurrent_battles=4,
    accept_open_team_sheet=True,
    open_timeout=None,
    team=RandomTeamBuilder(1, 20, "mb", None, None, take_from_end=True),
)

print(
    f"guards: {'OFF' if args.no_guards else 'ON'}  "
    f"search: {'ON' if args.search else 'OFF'}  battles: {args.n_battles}"
)
wr = Callback.compare(ours, opp, args.n_battles, per_reg=False)
print(f"win rate: {list(wr.values())[0] * 100:.0f}%")
print("guard firings (times a guard changed the top pick):")
counts = dict(PolicyPlayer.guard_fire_counts)
if counts:
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:14} {v}")
else:
    print("  (none)")
