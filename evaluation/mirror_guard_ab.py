"""Mirror match: the deployed bot against itself, one side with an extra guard.

The user, 2026-09-24: "at higher elo our opponents will win because of it esp
cuz its 2v2 and only 4 mons each move matters so much". Against the local
held-out opponents the bot already wins ~90% whatever it does, so a better
attack rarely decides a game there (dominated_attack A/B: +0.05pp). In a mirror
-- the deployed brain, team, preview model and guards on BOTH sides -- the
baseline is exactly 50%, so the only systematic difference is the guard under
test: side A plays with it, side B without it. Every game is as close as the
bot can make it, which is where a better move should decide results.

Both sides are the study's player class with the deployed setup read from
DEPLOYED.json through tools/deployed_config.py (like the evaluation arms: no
opponent-aware reranker). Games are split evenly into four blocks: open / hidden
team sheets x A challenging / B challenging (so neither side keeps the p1 seat).
The shared knowledge-observation cache is switched off: its key (battle, turn,
species, HP) cannot tell the two sides of a mirror apart.

Reports A's win rate (ties count half) with a Wilson 95% interval and how often
the guard changed A's action. Pre-registered reading (PROJECT_STATUS): the
guard wins close games if the lower bound is above 50%, loses them if the upper
bound is below 50%, else inconclusive. No promotion, no ladder.

Usage (from the repo root; a Showdown server must listen on --port):
  .venv/bin/python evaluation/mirror_guard_ab.py --guard dominated_attack --games 2000
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import time

import torch
from poke_env.ps_client import ServerConfiguration

from evaluation.opening_study import StudyPlayer, fresh_local_account
from tools.deployed_config import resolve
from vgc_bench.src import pokeenv_patches
from vgc_bench.src.guards import GUARDS, HARD_GUARDS
from vgc_bench.src.policy_player import PolicyPlayer
from vgc_bench.src.teams import RandomTeamBuilder
from vgc_bench.src.utils import format_map

BLOCKS = [(hidden, a_first) for hidden in (False, True) for a_first in (True, False)]


def wilson(wins: float, games: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a win rate (ties already counted as halves)."""
    if games <= 0:
        return 0.0, 1.0
    p = wins / games
    denom = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denom
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def block_sizes(games: int) -> list[int]:
    """Split the games as evenly as possible over the four blocks."""
    base, extra = divmod(games, len(BLOCKS))
    return [base + (i < extra) for i in range(len(BLOCKS))]


class _NoCache(dict):
    """A dict that never stores: the mirror's two sides must not share entries."""

    def get(self, key, default=None):  # noqa: D401
        return default

    def __setitem__(self, key, value) -> None:
        return None


def _player(config: dict, hidden: bool, seed: int, port: int, overrides: dict):
    player = StudyPlayer(
        account_configuration=fresh_local_account(),
        deterministic=True,
        team=RandomTeamBuilder(seed, 1, "mc", [ROOT / config["TEAM"]]),
        server_configuration=ServerConfiguration(
            f"ws://localhost:{port}/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?",
        ),
        battle_format=format_map["mc"],
        log_level=40,
        max_concurrent_battles=8,
        accept_open_team_sheet=not hidden,
        open_timeout=None,
        guard_overrides=overrides,
    )
    player.set_policy(ROOT / config["CKPT"], torch.device("mps"))
    if config["PREVIEW_MODEL"]:
        player.preview_model_path = ROOT / config["PREVIEW_MODEL"]
        player.use_learned_teampreview = True
    return player


def _outcomes(player) -> tuple[float, int, int]:
    """(wins with ties as halves, games, ties) from this player's side."""
    wins, games, ties = 0.0, 0, 0
    for battle in player.battles.values():
        if not battle.finished:
            raise RuntimeError(f"unfinished battle {battle.battle_tag}")
        games += 1
        if battle.won is None:
            ties += 1
            wins += 0.5
        elif battle.won:
            wins += 1
    return wins, games, ties


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--guard", required=True)
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20924)
    ap.add_argument("--port", type=int, default=7610)
    ap.add_argument("--block-timeout", type=float, default=5400)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    if args.guard not in GUARDS or args.guard in HARD_GUARDS:
        raise ValueError(f"{args.guard} is not an opt-in guard")
    config = resolve()
    deployed = [g for g in config["GUARDS"].split(",") if g]
    base = [g for g in deployed if g != args.guard]
    output = args.output or Path(f"results_mirror_{args.guard}")
    manifest = {
        "question": f"deployed bot with {args.guard} vs itself without it (mirror)",
        "guard": args.guard,
        "side_a": f"deployed setup + {args.guard}",
        "side_b": f"deployed setup without {args.guard}",
        "shared_guards": base,
        "checkpoint": config["CKPT"],
        "checkpoint_sha256": sha256(ROOT / config["CKPT"]),
        "team": config["TEAM"],
        "team_sha256": sha256(ROOT / config["TEAM"]),
        "preview_model": config["PREVIEW_MODEL"],
        "guards_module_sha256": sha256(ROOT / "vgc_bench/src/guards.py"),
        "games": args.games,
        "blocks": [
            {"hidden_sheets": h, "a_challenges": a, "games": n}
            for (h, a), n in zip(BLOCKS, block_sizes(args.games))
        ],
        "seed": args.seed,
        "reading": "lower bound > 50%: wins close games; upper < 50%: loses them",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.prepare_only:
        print(json.dumps(manifest["blocks"]))
        return

    os.environ.pop("VGC_SET_PRIOR_REG", None)
    if config["SET_PRIOR"] != "mc":
        os.environ["VGC_SET_PRIOR_REG"] = config["SET_PRIOR"]
    pokeenv_patches.install()
    torch.set_num_threads(1)
    PolicyPlayer.use_knowledge_obs = PolicyPlayer.use_moveset_prior = True
    PolicyPlayer.use_knowledge_guards = PolicyPlayer.mask_immunities = True
    PolicyPlayer.guard_flags = {g: g in HARD_GUARDS or g in base for g in GUARDS}
    PolicyPlayer._knowledge_cache = _NoCache()  # type: ignore[assignment]
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    results = []
    totals = {"wins": 0.0, "games": 0, "ties": 0, "fired": 0}
    for (hidden, a_first), n in zip(BLOCKS, block_sizes(args.games)):
        if n == 0:
            continue
        PolicyPlayer.guard_fire_counts.clear()
        a = _player(config, hidden, args.seed, args.port, {args.guard: True})
        b = _player(config, hidden, args.seed + 1, args.port, {args.guard: False})
        first, second = (a, b) if a_first else (b, a)
        started = time.monotonic()
        asyncio.run(
            asyncio.wait_for(
                first.battle_against(second, n_battles=n), args.block_timeout
            )
        )
        wins, games, ties = _outcomes(a)
        if games != n:
            raise RuntimeError(f"block expected {n} games, got {games}")
        fired = int(PolicyPlayer.guard_fire_counts.get(args.guard, 0))
        errors = {
            k: v for k, v in PolicyPlayer.guard_fire_counts.items() if "error" in k
        }
        if errors:
            raise RuntimeError(f"guard errors in the mirror: {errors}")
        results.append(
            {
                "hidden_sheets": hidden,
                "a_challenges": a_first,
                "games": games,
                "a_wins": wins,
                "ties": ties,
                "a_win_rate": wins / games,
                "guard_changed_actions": fired,
                "elapsed_s": round(time.monotonic() - started, 1),
            }
        )
        totals["wins"] += wins
        totals["games"] += games
        totals["ties"] += ties
        totals["fired"] += fired
        low, high = wilson(totals["wins"], totals["games"])
        print(
            f"block hidden={hidden} a_first={a_first}: A {wins:.1f}/{games}; "
            f"running A {100 * totals['wins'] / totals['games']:.1f}% "
            f"[{100 * low:.1f}, {100 * high:.1f}]",
            flush=True,
        )
        (output / "result.json").write_text(
            json.dumps({"blocks": results, "complete": False}, indent=2) + "\n"
        )
    low, high = wilson(totals["wins"], totals["games"])
    verdict = (
        "guard wins close games"
        if low > 0.5
        else "guard loses close games"
        if high < 0.5
        else "inconclusive"
    )
    summary = {
        "blocks": results,
        "games": totals["games"],
        "a_wins": totals["wins"],
        "ties": totals["ties"],
        "a_win_rate": totals["wins"] / totals["games"],
        "wilson_95": [low, high],
        "guard_changed_actions_per_game": totals["fired"] / totals["games"],
        "verdict": verdict,
        "complete": True,
    }
    (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "blocks"}, indent=2))


if __name__ == "__main__":
    main()
