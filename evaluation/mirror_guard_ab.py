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
bound is below 50%, else inconclusive. No promotion, no ladder. --guard takes a
comma-separated list to measure several guards together (side A has all).
--a-mixing gives side A mixed-strategy play instead of (or as well as) guards
(2026-09-26, the user's near-tie "wheel"): against a deterministic copy of
itself the mirror measures what the sampling COSTS; what unpredictability buys
against adaptive players only the ladder can show. --a-checkpoint puts another
BRAIN on side A (2026-09-26: a trained save head-to-head against the deployed
brain, both with the deployed guards and preview model). --rerankers gives BOTH
sides the ladder's opponent/tempo reranker (the opponent move and switch
models), which the evaluation arms otherwise leave out; --a-sticky gives side A
sticky guard corrections (the reranker may not restore a pair a guard corrected
away; ladder 2026-09-26 game 6). --a-playbook gives side A our own playbook at
team preview (vgc_bench/src/playbook.py: the plan card for this opponent, its
four, its leads and the turn-1 script for the opt-in guard playbook_opening --
pass --guard playbook_opening to play the script too). --a-team (with
--a-checkpoint) gives side A a set variant of the deployed team (2026-10-01: a
brain practised on T6m against the deployed brain on T6); it must have the same
six species, since both sides share the deployed, team-focused preview model.
--a-search risk|nash (2026-10-04, the user: "start the matrix search") gives side A
live exact search on every move turn (vgc_bench/src/live_exact.py; nash = the one-turn
matrix game per world, RESEARCH_TOP_BOTS.md) with --a-search-worlds hidden worlds and
--a-search-budget seconds; its decisions are audited to <output>/a_decisions.jsonl.
Search failures fall back to the champion plus guards by design, so they are counted
and reported, not treated as guard errors. Searched games are slow: run several
processes with different --seed / --output (--concurrency 1 each) and pool them.

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
from vgc_bench.src.set_particles import team_roster
from vgc_bench.src.teams import RandomTeamBuilder
from vgc_bench.src.utils import format_map, prior_path

BLOCKS = [(hidden, a_first) for hidden in (False, True) for a_first in (True, False)]
GUARD_SOURCES = (
    "vgc_bench/src/guards.py",
    "vgc_bench/src/trick_room_guard.py",
    "vgc_bench/src/pokeenv_patches.py",
)


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


def _player(
    config: dict,
    hidden: bool,
    seed: int,
    port: int,
    overrides: dict,
    mixing: dict | None = None,
    checkpoint: str | None = None,
    extra: dict | None = None,
    team: str | None = None,
    concurrency: int = 8,
):
    player = StudyPlayer(
        account_configuration=fresh_local_account(),
        deterministic=True,
        team=RandomTeamBuilder(seed, 1, "mc", [ROOT / (team or config["TEAM"])]),
        server_configuration=ServerConfiguration(
            f"ws://localhost:{port}/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?",
        ),
        battle_format=format_map["mc"],
        log_level=40,
        max_concurrent_battles=concurrency,
        accept_open_team_sheet=not hidden,
        open_timeout=None,
        guard_overrides=overrides,
        **(mixing or {}),
        **(extra or {}),
    )
    player.set_policy(ROOT / (checkpoint or config["CKPT"]), torch.device("mps"))
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


def same_species(team_a: Path, team_b: Path) -> bool:
    """Two team files with the same six species (a set variant of one team)."""
    return {p.species for p in team_roster(team_a.read_text())} == {
        p.species for p in team_roster(team_b.read_text())
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--guard", default="", help="opt-in guard(s), comma-separated")
    ap.add_argument("--a-mixing", choices=("off", "opening", "always"), default="off")
    ap.add_argument("--a-mixing-top-k", type=int, default=3)
    ap.add_argument("--a-mixing-temperature", type=float, default=1.0)
    ap.add_argument("--a-mixing-min-ratio", type=float, default=0.0)
    ap.add_argument("--a-mixing-keep-corrections", action="store_true")
    ap.add_argument("--a-checkpoint", type=Path, default=None, help="side A's brain")
    ap.add_argument(
        "--a-team",
        type=Path,
        default=None,
        help="side A's team: a set variant of the deployed team (same six species)",
    )
    ap.add_argument(
        "--rerankers",
        action="store_true",
        help="both sides run the ladder's opponent/tempo reranker",
    )
    ap.add_argument(
        "--a-playbook",
        type=Path,
        default=None,
        help="side A picks its preview from this playbook (data/playbook_<team>.json)",
    )
    ap.add_argument(
        "--a-sticky",
        action="store_true",
        help="side A keeps guard corrections the reranker would undo",
    )
    ap.add_argument(
        "--a-search",
        choices=("off", "risk", "nash"),
        default="off",
        help="side A searches every move turn with this solution",
    )
    ap.add_argument("--a-search-worlds", type=int, default=4)
    ap.add_argument(
        "--a-search-leaf",
        choices=("critic", "outcome"),
        default="critic",
        help="leaf value: the brain's critic + shaping potential, or the August "
        "outcome net (calibrated on the Reg M-B champion)",
    )
    ap.add_argument("--a-search-budget", type=float, default=8.0)
    ap.add_argument(
        "--a-search-outcome",
        type=Path,
        default=Path("results_outcome_v2h/outcome_value.zip"),
        help="the outcome-net checkpoint for --a-search-leaf outcome",
    )
    ap.add_argument(
        "--a-search-argmax",
        action="store_true",
        help="nash: play the averaged equilibrium's top action instead of sampling",
    )
    ap.add_argument(
        "--a-search-prior-mix",
        type=float,
        default=0.0,
        help="nash: play (1 - m) * equilibrium + m * policy prior",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="concurrent battles (searching sides share one inference lock: use 1)",
    )
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20924)
    ap.add_argument("--port", type=int, default=7610)
    ap.add_argument("--block-timeout", type=float, default=5400)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    guards = [g for g in args.guard.split(",") if g]
    if any(g not in GUARDS or g in HARD_GUARDS for g in guards):
        raise ValueError(f"{args.guard} is not a list of opt-in guards")
    mixing = None
    if args.a_mixing != "off":
        mixing = {
            "mixing_mode": args.a_mixing,
            "mixing_top_k": args.a_mixing_top_k,
            "mixing_temperature": args.a_mixing_temperature,
            "mixing_min_ratio": args.a_mixing_min_ratio,
            "mixing_keep_corrections": args.a_mixing_keep_corrections,
            "mixing_seed": args.seed,
        }
    if (
        not guards
        and mixing is None
        and args.a_checkpoint is None
        and not args.a_sticky
        and args.a_playbook is None
        and args.a_search == "off"
    ):
        raise ValueError(
            "nothing to compare: give --guard, --a-mixing, --a-checkpoint or --a-sticky"
        )
    if args.a_sticky and not args.rerankers:
        raise ValueError("--a-sticky only matters with --rerankers")
    shared_extra: dict = {}
    if args.rerankers:
        shared_extra = {
            "move_model_path": ROOT / prior_path("move", "mc"),
            "switch_model_path": ROOT / prior_path("switch", "mc"),
            "use_opponent_reranker": True,
            "use_tempo_reranker": True,
        }
    a_checkpoint = None
    if args.a_checkpoint is not None:
        if not (ROOT / args.a_checkpoint).is_file():
            raise ValueError(f"no checkpoint {args.a_checkpoint}")
        a_checkpoint = str(args.a_checkpoint)
    config = resolve()
    a_team = None
    if args.a_team is not None:
        if a_checkpoint is None:
            raise ValueError("--a-team needs --a-checkpoint (a brain practised on it)")
        if not (ROOT / args.a_team).is_file():
            raise ValueError(f"no team {args.a_team}")
        if not same_species(ROOT / args.a_team, ROOT / config["TEAM"]):
            raise ValueError("--a-team must have the deployed team's six species")
        a_team = str(args.a_team)
    deployed = [g for g in config["GUARDS"].split(",") if g]
    base = [g for g in deployed if g not in guards]
    parts = list(guards)
    if mixing is not None:
        parts.append(
            f"{args.a_mixing} mixing (top {args.a_mixing_top_k}, "
            f"T={args.a_mixing_temperature:g}, min ratio {args.a_mixing_min_ratio:g}"
            + (", corrections kept)" if args.a_mixing_keep_corrections else ")")
        )
    if a_checkpoint is not None:
        parts.insert(0, f"brain {a_checkpoint}")
    if a_team is not None:
        parts.insert(1, f"team {a_team}")
    if args.a_sticky:
        parts.append("sticky guard corrections")
    if args.a_playbook is not None:
        if not (ROOT / args.a_playbook).is_file():
            raise ValueError(f"no playbook {args.a_playbook}")
        parts.insert(0, f"playbook {args.a_playbook}")
    named = ", ".join(parts)
    tag = "_".join(guards) if guards else f"mixing_{args.a_mixing}"
    if a_checkpoint is not None:
        tag = f"brain_{Path(a_checkpoint).stem}"
    if a_team is not None:
        tag += f"_team_{Path(a_team).stem}"
    if mixing is not None and guards:
        tag += f"_mixing_{args.a_mixing}"
    if args.a_sticky and not guards and mixing is None and a_checkpoint is None:
        tag = "sticky_corrections"
    if args.a_playbook is not None:
        tag = "playbook_" + Path(args.a_playbook).stem
        tag += ("_" + "_".join(guards)) if guards else ""
    if args.rerankers:
        tag += "_rerankers"
    search = None
    if args.a_search != "off":
        if not 1 <= args.a_search_worlds <= 8 or not 0 < args.a_search_budget <= 9:
            raise ValueError("--a-search-worlds must be 1-8, the budget in (0, 9]")
        search = {
            "solution": args.a_search,
            "worlds": args.a_search_worlds,
            "budget_s": args.a_search_budget,
            "every_turn": True,
            "leaf": args.a_search_leaf,
            "sample": not args.a_search_argmax,
            "prior_mix": args.a_search_prior_mix,
            "outcome_value": str(args.a_search_outcome),
            "outcome_value_sha256": sha256(ROOT / args.a_search_outcome),
        }
        parts.append(
            f"{args.a_search} exact search ({args.a_search_worlds} worlds, "
            f"{args.a_search_budget:g}s)"
        )
        named = ", ".join(parts)
        if not guards and mixing is None and a_checkpoint is None:
            tag = f"search_{args.a_search}"
    output = args.output or Path(f"results_mirror_{tag}")
    manifest = {
        "question": (
            f"brain {a_checkpoint}"
            + (f" on {a_team}" if a_team else "")
            + " vs the deployed brain, both with the deployed guards and preview "
            "(head-to-head mirror)"
            if a_checkpoint is not None and not guards and mixing is None
            else f"deployed bot with {named} vs itself without (mirror)"
        ),
        "guard": args.guard,
        "side_a": f"deployed setup + {named}",
        "side_b": f"deployed setup without {named}",
        "shared_guards": base,
        "a_mixing": mixing,
        "a_checkpoint": a_checkpoint,
        "a_search": search,
        "concurrency": args.concurrency,
        "rerankers_both_sides": args.rerankers,
        "reranker_models": {
            k: str(v) for k, v in shared_extra.items() if k.endswith("_path")
        },
        "a_sticky_guard_corrections": args.a_sticky,
        "a_playbook": str(args.a_playbook) if args.a_playbook else None,
        "a_playbook_sha256": (
            sha256(ROOT / args.a_playbook) if args.a_playbook else None
        ),
        "a_checkpoint_sha256": sha256(ROOT / a_checkpoint) if a_checkpoint else None,
        "a_team": a_team,
        "a_team_sha256": sha256(ROOT / a_team) if a_team else None,
        "checkpoint": config["CKPT"],
        "checkpoint_sha256": sha256(ROOT / config["CKPT"]),
        "team": config["TEAM"],
        "team_sha256": sha256(ROOT / config["TEAM"]),
        "preview_model": config["PREVIEW_MODEL"],
        "guards_module_sha256": sha256(ROOT / "vgc_bench/src/guards.py"),
        "guard_sources_sha256": {path: sha256(ROOT / path) for path in GUARD_SOURCES},
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

    search_extra: dict = {}
    if search is not None:
        from vgc_bench.src.exact_planner import PlannerConfig

        search_extra = {
            "enable_search": True,
            "exact_search_config": PlannerConfig(
                depth=2,
                root_width=6,
                opponent_width=6,
                continuation_width=3,
                replacement_width=2,
                chance_samples=1,
                deep_root_width=4,
                anytime=True,
                screen_budget_s=min(2.0, args.a_search_budget),
                time_budget_s=args.a_search_budget,
                max_nodes=5000,
                solution=args.a_search,
                nash_sample=not args.a_search_argmax,
                nash_prior_mix=args.a_search_prior_mix,
            ),
            "outcome_value_path": ROOT / search["outcome_value"],
            "exact_team_path": ROOT / (a_team or config["TEAM"]),
            "exact_selective_search": False,
            "exact_max_determinizations": 8,
            "exact_search_determinizations": args.a_search_worlds,
            "exact_min_deep_coverage": 0.5,
            "exact_leaf": args.a_search_leaf,
            "decision_log_path": output / "a_decisions.jsonl",
        }
    results = []
    totals = {"wins": 0.0, "games": 0, "ties": 0, "fired": 0}
    for (hidden, a_first), n in zip(BLOCKS, block_sizes(args.games)):
        if n == 0:
            continue
        PolicyPlayer.guard_fire_counts.clear()
        a = _player(
            config,
            hidden,
            args.seed,
            args.port,
            dict.fromkeys(guards, True),
            mixing,
            a_checkpoint,
            shared_extra
            | ({"sticky_guard_corrections": True} if args.a_sticky else {})
            | (
                {"playbook_path": ROOT / args.a_playbook}
                if args.a_playbook is not None
                else {}
            )
            | search_extra,
            a_team,
            args.concurrency,
        )
        b = _player(
            config,
            hidden,
            args.seed + 1,
            args.port,
            dict.fromkeys(guards, False),
            extra=shared_extra,
            concurrency=args.concurrency,
        )
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
        counts = PolicyPlayer.guard_fire_counts
        per_guard = {g: int(counts.get(g, 0)) for g in guards}
        if mixing is not None:  # only side A mixes, so the counter is A's
            per_guard["mixing"] = int(counts.get("mixing_changed_pick", 0))
        if args.a_sticky:  # only side A is sticky
            per_guard["sticky"] = int(counts.get("sticky_correction_kept", 0))
        fired = sum(per_guard.values())
        # exact-search failures fall back to the champion plus guards by design:
        # counted and reported below, never a reason to stop
        errors = {
            k: v
            for k, v in PolicyPlayer.guard_fire_counts.items()
            if "error" in k and not k.startswith("exact_")
        }
        if errors:
            raise RuntimeError(f"guard errors in the mirror: {errors}")
        search_counts = {
            k: int(v)
            for k, v in PolicyPlayer.guard_fire_counts.items()
            if k.startswith("exact_")
        }
        results.append(
            {
                "hidden_sheets": hidden,
                "a_challenges": a_first,
                "games": games,
                "a_wins": wins,
                "ties": ties,
                "a_win_rate": wins / games,
                "guard_changed_actions": fired,
                "changed_by_guard": per_guard,
                "a_search_counts": search_counts,
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
