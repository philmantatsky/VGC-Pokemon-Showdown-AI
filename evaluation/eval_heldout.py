"""Evaluate a checkpoint against UNSEEN opponent teams.

The built-in eval (src/callback.py:142) builds its opponent with
`RandomTeamBuilder(run_id, num_teams, reg, ...)` -- the same seed and therefore the
same team slice used for training. That measures play against teams the agent has
already trained on, which for a ladder bot is the wrong question: on ladder every
opponent team is unseen.

`_select_paths` shuffles deterministically by run_id and slices either the front or
the back (`take_from_end`, src/teams.py:162-166). Training takes the front; this takes
the back, so the two sets are disjoint whenever 2*num_teams <= pool size.

    python eval_heldout.py --checkpoint <path.zip> --num_teams 8 --n_battles 100
"""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import os
from pathlib import Path

from poke_env.player import SimpleHeuristicsPlayer
from poke_env.ps_client import ServerConfiguration
from torch import device

from vgc_bench.src.callback import Callback
from vgc_bench.src.policy_player import BatchPolicyPlayer
from vgc_bench.src.teams import RandomTeamBuilder
from vgc_bench.src.utils import format_map


def main():
    # Callback.compare is sync and calls asyncio.run() internally, so this must not
    # run inside an event loop.
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to a .zip policy")
    ap.add_argument("--reg", default="mb")
    ap.add_argument("--run_id", type=int, default=1)
    ap.add_argument(
        "--num_teams",
        type=int,
        default=8,
        help="size of the TRAINING slice; held-out set is the same size "
        "taken from the opposite end",
    )
    ap.add_argument("--our_team", default="teams/reg_mb/our_team.txt")
    ap.add_argument("--n_battles", type=int, default=100)
    ap.add_argument("--port", type=int, default=7400)
    ap.add_argument("--device", default="mps")
    ap.add_argument(
        "--train_num_teams",
        type=int,
        default=None,
        help="team count TRAINING used, if different from --num_teams; "
        "only affects the overlap sanity check",
    )
    ap.add_argument(
        "--knowledge_obs",
        action="store_true",
        help="populate knowledge features; REQUIRED for checkpoints "
        "fine-tuned with --knowledge_obs, and wrong for older ones",
    )
    ap.add_argument(
        "--search",
        action="store_true",
        help="one-ply matrix-game search over joint actions",
    )
    ap.add_argument(
        "--guards",
        action="store_true",
        help="enable the knowledge guard stack during evaluation",
    )
    ap.add_argument(
        "--guard_profile",
        choices=("hard", "all"),
        default="hard",
        help="factual hard guards (default) or the full experimental stack",
    )
    ap.add_argument(
        "--seen",
        action="store_true",
        help="evaluate on the TRAINED teams instead (for comparison)",
    )
    args = ap.parse_args()

    if args.search:
        from vgc_bench.src.search import backend_status

        ready, reason = backend_status()
        if not ready:
            raise SystemExit(f"Search refused: {reason}")
        print(f"search backend: {reason}", flush=True)

    # The env var is for any subprocess that imports fresh; policy_player is already
    # imported here (top of file), so the explicit assignment below is what takes
    # effect in this process.
    if args.knowledge_obs:
        os.environ["VGC_KNOWLEDGE_OBS"] = "1"
    from vgc_bench.src import pokeenv_patches as _patches
    from vgc_bench.src.policy_player import PolicyPlayer as _PP

    _patches.install()

    _PP.use_knowledge_obs = args.knowledge_obs
    _PP.use_knowledge_guards = args.guards or args.search
    if _PP.use_knowledge_guards and args.guard_profile == "hard":
        from vgc_bench.src.guards import GUARDS, HARD_GUARDS

        _PP.guard_flags = {name: name in HARD_GUARDS for name in GUARDS}
    else:
        _PP.guard_flags = None
    _PP.use_search = args.search
    _PP.use_moveset_prior = True
    print(
        f"knowledge_obs={args.knowledge_obs}  "
        f"guards={args.guards or args.search}  search={args.search}"
    )

    server = ServerConfiguration(
        f"ws://localhost:{args.port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )
    fmt = format_map[args.reg]

    # Our side: always the ladder team.
    ours = BatchPolicyPlayer(
        server_configuration=server,
        battle_format=fmt,
        log_level=40,
        max_concurrent_battles=8,
        accept_open_team_sheet=True,
        open_timeout=None,
        team=RandomTeamBuilder(args.run_id, 1, args.reg, [Path(args.our_team)]),
    )
    ours.set_policy(Path(args.checkpoint), device(args.device))

    # Opponent: held-out teams by default (take_from_end flips the slice).
    opp_builder = RandomTeamBuilder(
        args.run_id, args.num_teams, args.reg, None, None, take_from_end=not args.seen
    )
    opponent = SimpleHeuristicsPlayer(
        server_configuration=server,
        battle_format=fmt,
        log_level=40,
        max_concurrent_battles=8,
        accept_open_team_sheet=True,
        open_timeout=None,
        team=opp_builder,
    )

    # Overlap must be checked against the slice TRAINING used, which is not
    # necessarily the eval pool size (e.g. train on 8, evaluate against 500).
    train_n = args.train_num_teams or args.num_teams
    train_set = RandomTeamBuilder(args.run_id, train_n, args.reg, None, None)
    train_names = {p.name for p in (train_set._team_paths or [])}
    eval_names = {p.name for p in (opp_builder._team_paths or [])}
    overlap = train_names & eval_names
    label = "SEEN (trained)" if args.seen else "HELD-OUT (unseen)"
    print(f"our team : {Path(args.our_team).name}")
    print(f"opponents: {label}, {len(eval_names)} teams")
    print(
        f"overlap with training slice: {len(overlap)} teams"
        f"{' <-- NOT held out!' if overlap and not args.seen else ''}"
    )

    wr = Callback.compare(ours, opponent, args.n_battles, per_reg=False)
    for k, v in wr.items():
        print(f"win rate{k}: {v * 100:.0f}%  ({args.n_battles} battles)")

    # Which guard actually moved the needle. A win rate alone cannot tell whether the
    # gain came from the zero-damage veto or the KO tiebreak, and a silent counter is
    # how the "guards never ran" bug hid: zero firings AND zero errors looked like
    # "nothing to fix" when the code was simply not being reached. `error` and
    # `search_error` are failures falling back to plain sampling, not successes.
    counts = dict(_PP.guard_fire_counts)
    if counts:
        print(
            f"\nguard firings over {args.n_battles} battles "
            f"(times a guard changed the top pick):"
        )
        for k, v in sorted(counts.items(), key=lambda x: -x[1]):
            flag = "  <-- FAILURE, fell back to sampling" if "error" in k else ""
            print(f"  {k:26} {v}{flag}")
    elif args.guards or args.search:
        print(
            "\nWARNING: guards/search enabled but NOTHING fired -- "
            "the code path is probably not being reached."
        )

    if args.search:
        from vgc_bench.src.search import latency_summary

        print(latency_summary())
    print(_patches.report())


if __name__ == "__main__":
    main()
