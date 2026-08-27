"""Separate team-preview quality, battle policy quality, and team quality.

The original evaluation reported only one win rate, so a weak bring/lead decision was
indistinguishable from bad turn play or a bad fixed team. This script runs three arms
against the same full opponent pool:

* policy preview + policy turns
* random preview + the same policy turns
* random preview + poke-env's heuristic turns (team-only sanity baseline)
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from poke_env.player import SimpleHeuristicsPlayer
from poke_env.ps_client import ServerConfiguration
from torch import device

from vgc_bench.src.policy import MaskedActorCriticPolicy
from vgc_bench.src.policy_player import BatchPolicyPlayer, PolicyPlayer
from vgc_bench.src.teams import RandomTeamBuilder
from vgc_bench.src.utils import chunk_obs_len, format_map


class ShapeAwareBatchPolicyPlayer(BatchPolicyPlayer):
    """Feed historical checkpoints the observation prefix they were trained on."""

    def embed_battle(self, battle, fake_rating=None):
        full = PolicyPlayer.embed_battle(battle, fake_rating)
        assert self.policy is not None
        space = self.policy.observation_space["observation"]
        expected = space.shape[0]
        if expected == full.shape[0]:
            return full
        if expected > full.shape[0] or expected % 12:
            raise ValueError(
                f"unsupported historical observation {expected}; "
                f"current={full.shape[0]}"
            )
        old_chunk = expected // 12
        assert old_chunk < chunk_obs_len
        return full.reshape(12, chunk_obs_len)[:, :old_chunk].reshape(-1)


def interval(wins: int, total: int) -> tuple[float, float]:
    """95% Wilson interval for a binomial win rate."""
    if not total:
        return 0.0, 0.0
    z = 1.96
    p = wins / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return center - radius, center + radius


def policy_player(
    args, server, choose_preview: bool, learned_preview: bool = False
) -> BatchPolicyPlayer:
    player = BatchPolicyPlayer(
        server_configuration=server,
        battle_format=format_map[args.reg],
        log_level=40,
        max_concurrent_battles=args.workers,
        accept_open_team_sheet=not args.hidden_sheets,
        open_timeout=None,
        team=RandomTeamBuilder(args.seed, 1, args.reg, [Path(args.our_team)]),
        preview_model_path=(
            Path(args.preview_model) if learned_preview and args.preview_model else None
        ),
        use_learned_teampreview=learned_preview,
    )
    player.set_policy(Path(args.checkpoint), device(args.device))
    assert isinstance(player.policy, MaskedActorCriticPolicy)
    player.policy.choose_on_teampreview = choose_preview
    return player


def opponent(args, server):
    weights = Path(args.team_weights) if args.team_weights else None
    common = dict(
        server_configuration=server,
        battle_format=format_map[args.reg],
        log_level=40,
        max_concurrent_battles=args.workers,
        # One denial hides sheets from both players.
        accept_open_team_sheet=not args.hidden_sheets,
        open_timeout=None,
        team=RandomTeamBuilder(args.seed, None, args.reg, weights_path=weights),
    )
    if not args.opponent_checkpoint:
        return SimpleHeuristicsPlayer(**common)
    foe = ShapeAwareBatchPolicyPlayer(deterministic=False, **common)
    foe.set_policy(Path(args.opponent_checkpoint), device(args.device))
    return foe


def battle_arm(args, name: str, ours, foe) -> tuple[str, int, int]:
    import asyncio

    random.seed(args.seed)
    asyncio.run(ours.battle_against(foe, n_battles=args.n_battles))
    wins = ours.n_won_battles
    total = ours.n_finished_battles
    low, high = interval(wins, total)
    print(
        f"{name:28} {wins:3d}/{total:3d} = {wins / total * 100:5.1f}%  "
        f"95% CI [{low * 100:4.1f}, {high * 100:4.1f}]",
        flush=True,
    )
    ours.reset_battles()
    foe.reset_battles()
    return name, wins, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--our_team", default="teams/reg_mb/our_team.txt")
    parser.add_argument("--team_weights", default="data/team_weights_regmb.json")
    parser.add_argument("--reg", default="mb")
    parser.add_argument("--port", type=int, default=7400)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--preview_model",
        default="",
        help="optional replay-trained bring/lead model to add a fourth preview arm",
    )
    parser.add_argument("--n_battles", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--hidden_sheets", action="store_true")
    parser.add_argument(
        "--opponent_checkpoint",
        default="",
        help="use a historical learned policy instead of the simple heuristic",
    )
    parser.add_argument("--no_guards", action="store_true")
    parser.add_argument(
        "--guard_profile",
        choices=("all", "hard", "none"),
        default="all",
        help="all guards, factual hard guards only, or no guard stack",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from vgc_bench.src import pokeenv_patches

    pokeenv_patches.install()
    PolicyPlayer.use_knowledge_obs = True
    PolicyPlayer.use_moveset_prior = True
    PolicyPlayer.mask_immunities = True
    guard_profile = "none" if args.no_guards else args.guard_profile
    PolicyPlayer.use_knowledge_guards = guard_profile != "none"
    if guard_profile == "hard":
        from vgc_bench.src.guards import GUARDS, HARD_GUARDS

        PolicyPlayer.guard_flags = {name: name in HARD_GUARDS for name in GUARDS}
    else:
        PolicyPlayer.guard_flags = None
    PolicyPlayer.use_search = False
    PolicyPlayer.guard_fire_counts.clear()

    server = ServerConfiguration(
        f"ws://localhost:{args.port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )
    sheet_label = "hidden" if args.hidden_sheets else "open"
    opponent_label = args.opponent_checkpoint or "simple heuristic"
    print(
        f"full weighted pool; {sheet_label} sheets; opponent={opponent_label}; "
        f"{args.n_battles} battles/arm",
        flush=True,
    )

    results = []
    if args.preview_model:
        learned = policy_player(args, server, choose_preview=True, learned_preview=True)
        results.append(
            battle_arm(
                args, "learned preview + policy", learned, opponent(args, server)
            )
        )

    controlled = policy_player(args, server, choose_preview=True)
    results.append(
        battle_arm(args, "policy preview + policy", controlled, opponent(args, server))
    )

    random_preview = policy_player(args, server, choose_preview=False)
    results.append(
        battle_arm(
            args, "random preview + policy", random_preview, opponent(args, server)
        )
    )

    heuristic = SimpleHeuristicsPlayer(
        server_configuration=server,
        battle_format=format_map[args.reg],
        log_level=40,
        max_concurrent_battles=args.workers,
        accept_open_team_sheet=not args.hidden_sheets,
        open_timeout=None,
        team=RandomTeamBuilder(args.seed, 1, args.reg, [Path(args.our_team)]),
    )
    results.append(
        battle_arm(
            args, "random preview + heuristic", heuristic, opponent(args, server)
        )
    )
    print(f"guard telemetry: {dict(PolicyPlayer.guard_fire_counts)}")
    print(pokeenv_patches.report())
    if args.output:
        payload = {
            "checkpoint": args.checkpoint,
            "hidden_sheets": args.hidden_sheets,
            "guard_profile": guard_profile,
            "n_battles": args.n_battles,
            "arms": {
                name: {"wins": wins, "battles": total, "win_rate": wins / total}
                for name, wins, total in results
            },
            "guard_telemetry": dict(PolicyPlayer.guard_fire_counts),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
