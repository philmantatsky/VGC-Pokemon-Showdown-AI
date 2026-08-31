"""Matched local A/B for prediction-conditioned move reranking."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import asyncio
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.ps_client import ServerConfiguration
from torch import device

from evaluation.eval_openings import ShapeAwareBatchPolicyPlayer
from vgc_bench.src.policy import MaskedActorCriticPolicy
from vgc_bench.src.policy_player import BatchPolicyPlayer, PolicyPlayer
from vgc_bench.src.teams import RandomTeamBuilder
from vgc_bench.src.utils import format_map


def interval(wins: int, total: int) -> tuple[float, float]:
    z = 1.96
    p = wins / max(total, 1)
    denominator = 1 + z * z / max(total, 1)
    center = (p + z * z / (2 * max(total, 1))) / denominator
    radius = (
        z
        * math.sqrt(p * (1 - p) / max(total, 1) + z * z / (4 * max(total, 1) ** 2))
        / denominator
    )
    return center - radius, center + radius


def player(args, server, opponent_aware: bool, tempo_aware: bool) -> BatchPolicyPlayer:
    kwargs = {}
    if opponent_aware:
        kwargs = {
            "preview_model_path": Path(args.preview_model),
            "switch_model_path": Path(args.switch_model),
            "move_model_path": Path(args.move_model),
            "use_opponent_reranker": True,
            "use_tempo_reranker": tempo_aware,
        }
    agent = BatchPolicyPlayer(
        server_configuration=server,
        battle_format=format_map[args.reg],
        log_level=40,
        max_concurrent_battles=args.workers,
        accept_open_team_sheet=not args.hidden_sheets,
        open_timeout=None,
        team=RandomTeamBuilder(args.seed, 1, args.reg, [Path(args.our_team)]),
        **kwargs,
    )
    agent.set_policy(Path(args.checkpoint), device(args.device))
    assert isinstance(agent.policy, MaskedActorCriticPolicy)
    agent.policy.choose_on_teampreview = True
    return agent


def opponent(args, server):
    common = dict(
        server_configuration=server,
        battle_format=format_map[args.reg],
        log_level=40,
        max_concurrent_battles=args.workers,
        accept_open_team_sheet=not args.hidden_sheets,
        open_timeout=None,
        team=RandomTeamBuilder(
            args.seed, None, args.reg, weights_path=Path(args.team_weights)
        ),
    )
    if not args.opponent_checkpoint:
        return SimpleHeuristicsPlayer(**common)
    # The goal of this script is a low-variance A/B of our decision layer. Sampling
    # the opponent's policy made repeated 300-game baselines wander by roughly five
    # points and obscured the sign of small changes.
    foe = ShapeAwareBatchPolicyPlayer(deterministic=True, **common)
    foe.set_policy(Path(args.opponent_checkpoint), device(args.device))
    return foe


def run_arm(args, server, name: str, aware: bool, tempo_aware: bool):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    PolicyPlayer.guard_fire_counts.clear()
    PolicyPlayer._decisions_seen = 0
    ours = player(args, server, aware, tempo_aware)
    foe = opponent(args, server)
    started = time.perf_counter()
    asyncio.run(ours.battle_against(foe, n_battles=args.n_battles))
    elapsed = time.perf_counter() - started
    wins = ours.n_won_battles
    total = ours.n_finished_battles
    low, high = interval(wins, total)
    telemetry = dict(PolicyPlayer.guard_fire_counts)
    print(
        f"{name:22} {wins:3d}/{total:3d} = {wins / total * 100:5.1f}% "
        f"95% CI [{low * 100:4.1f}, {high * 100:4.1f}] "
        f"({elapsed:.1f}s)",
        flush=True,
    )
    print(f"  telemetry: {telemetry}", flush=True)
    return {
        "wins": wins,
        "battles": total,
        "win_rate": wins / total,
        "elapsed_seconds": elapsed,
        "telemetry": telemetry,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--our_team", default="teams/reg_mb/our_team.txt")
    parser.add_argument("--team_weights", default="data/team_weights_regmb.json")
    parser.add_argument("--reg", default="mb")
    parser.add_argument("--port", type=int, default=7400)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_battles", type=int, default=300)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--hidden_sheets", action="store_true")
    parser.add_argument("--opponent_checkpoint", default="")
    parser.add_argument(
        "--preview_model", default="data/opponent_preview_top500_regmb.pt"
    )
    parser.add_argument(
        "--switch_model", default="data/opponent_switch_top500_regmb.pt"
    )
    parser.add_argument("--move_model", default="data/opponent_move_top500_regmb.pt")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--tempo_only",
        action="store_true",
        help="run only the tempo arm when a matched baseline is already saved",
    )
    args = parser.parse_args()

    from vgc_bench.src import pokeenv_patches
    from vgc_bench.src.guards import GUARDS, HARD_GUARDS

    pokeenv_patches.install()
    PolicyPlayer.use_knowledge_obs = True
    PolicyPlayer.use_moveset_prior = True
    PolicyPlayer.mask_immunities = True
    PolicyPlayer.use_knowledge_guards = True
    PolicyPlayer.guard_flags = {name: name in HARD_GUARDS for name in GUARDS}
    PolicyPlayer.use_search = False
    server = ServerConfiguration(
        f"ws://localhost:{args.port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )
    results = {}
    if not args.tempo_only:
        results["opponent_aware"] = run_arm(args, server, "opponent-aware", True, False)
    results["opponent_plus_tempo"] = run_arm(
        args, server, "opponent + tempo", True, True
    )
    print(pokeenv_patches.report())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
