"""Does the exact search help against other teams? (2026-10-05)

evaluation/mirror_guard_ab.py measures the search in one matchup: our own team on
both sides, piloted by our own brain, so the search's model of the opponent is the
opponent. This plays the deployed bot against ONE opponent of the held-out battery (a
clone of human play, a frozen league opponent, the heuristic) on the held-out rosters,
twice per cell: as deployed ("plain") and with the search ("search"). Both arms get
the same rosters, sheets and game counts and are compared roster by roster (a
bootstrap over rosters, as the battery does), and every cell records how often the
search really ran, fell back or failed -- the held-out rosters carry species and
mechanics our own team never shows it.

One opponent per process: the searching side plays one battle at a time
(tools/search_roster_arms.sh runs the battery's six side by side). Cells are appended
whole, so a run can be resumed. Nothing is promoted by this script; search stays off
in the deployed bot.
Usage (from the repo root; a Showdown server on --port):
    .venv/bin/python evaluation/search_roster_ab.py --opponent heuristic \\
        --output results_search_rosters/heuristic \\
        --a-search-anchor 0.07 --a-search-replies 8 --a-search-streams 4
    .venv/bin/python evaluation/search_roster_ab.py --pool <run dir> [...] --json out
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402
import asyncio  # noqa: E402
import collections  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from evaluation.mirror_guard_ab import wilson  # noqa: E402

ARMS = ("plain", "search")
# what every player in the battery runs besides the hard guards (opening_study.py)
BATTERY_EXTRAS = ("resisted_target", "overkill_split", "dominated_weather_ball_weather")


def record(rows: list[dict]) -> dict:
    wins = float(sum(row["target"] for row in rows))
    low, high = wilson(wins, len(rows)) if rows else (0.0, 1.0)
    return {
        "games": len(rows),
        "wins": wins,
        "win_rate": wins / len(rows) if rows else None,
        "wilson_95": [low, high],
    }


def complete_cells(rows: list[dict], repeats: int) -> set[tuple[str, bool, str]]:
    """(roster, hidden sheets, arm) cells that hold all their games; a cell with
    another count means the file was edited by hand and nothing can be trusted."""
    counts = collections.Counter(
        (row["opponent"], bool(row["hidden_sheets"]), row["arm"]) for row in rows
    )
    partial = {cell: n for cell, n in counts.items() if n != repeats}
    if partial:
        raise ValueError(f"incomplete or duplicated cells: {partial}")
    return set(counts)


def paired_delta(rows: list[dict], draws: int = 4000, seed: int = 20925) -> dict:
    """Search minus plain, roster by roster, with a bootstrap over rosters.

    Only rosters whose cells exist in both arms count. Games on one roster are
    correlated, so whole rosters are resampled, never single games."""
    out = {}
    for mode in ("overall", "hidden", "open"):
        scope = [
            row
            for row in rows
            if mode == "overall" or bool(row["hidden_sheets"]) == (mode == "hidden")
        ]
        by_arm: dict[str, dict[str, list[float]]] = {arm: {} for arm in ARMS}
        for row in scope:
            by_arm[row["arm"]].setdefault(row["opponent"], []).append(row["target"])
        rosters = sorted(
            roster
            for roster in by_arm["plain"]
            if len(by_arm["search"].get(roster, [])) == len(by_arm["plain"][roster])
        )
        if not rosters:
            out[mode] = {"rosters": 0, "delta": None, "bootstrap_95": None}
            continue
        delta = np.array(
            [
                np.mean(by_arm["search"][roster]) - np.mean(by_arm["plain"][roster])
                for roster in rosters
            ]
        )
        boot = (
            np.random.default_rng(seed).choice(delta, (draws, len(delta))).mean(axis=1)
        )
        out[mode] = {
            "rosters": len(rosters),
            "games_per_arm": int(sum(len(by_arm["plain"][r]) for r in rosters)),
            "plain_win_rate": float(
                np.mean([t for r in rosters for t in by_arm["plain"][r]])
            ),
            "search_win_rate": float(
                np.mean([t for r in rosters for t in by_arm["search"][r]])
            ),
            "delta": float(delta.mean()),
            "bootstrap_95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        }
    return out


def summarize(rows: list[dict], telemetry: list[dict] | None = None) -> dict:
    counts: collections.Counter[str] = collections.Counter()
    for cell in telemetry or []:
        if cell["arm"] == "search":
            counts.update(cell.get("search_counts") or {})
    return {
        "arms": {
            arm: {
                "overall": record([r for r in rows if r["arm"] == arm]),
                "hidden": record(
                    [r for r in rows if r["arm"] == arm and r["hidden_sheets"]]
                ),
                "open": record(
                    [r for r in rows if r["arm"] == arm and not r["hidden_sheets"]]
                ),
            }
            for arm in ARMS
        },
        "search_minus_plain": paired_delta(rows),
        "search_counts": dict(counts),
    }


def pool(runs: list[Path]) -> dict:
    """Several opponents' runs as one: a roster against another opponent is another
    roster. Refuses runs that differ in anything but the opponent."""
    rows: list[dict] = []
    telemetry: list[dict] = []
    first: dict | None = None
    opponents = []
    for run in runs:
        manifest = json.loads((run / "manifest.json").read_text())
        opponents.append(manifest["opponent"])
        shared = {
            key: value
            for key, value in manifest.items()
            if key not in {"opponent", "opponent_sha256", "deterministic_opponent"}
        }
        if first is None:
            first = shared
        elif shared != first:
            raise ValueError(f"{run} is not the same study as {runs[0]}")
        for row in _read(run / "rows.jsonl"):
            rows.append({**row, "opponent": f"{run.name}/{row['opponent']}"})
        telemetry += _read(run / "telemetry.jsonl")
    if len(set(opponents)) != len(opponents):
        raise ValueError("the same opponent twice")
    return {
        "runs": [str(run) for run in runs],
        "opponents": opponents,
        **summarize(rows, telemetry),
        "by_opponent": {
            run.name: summarize(_read(run / "rows.jsonl"))["search_minus_plain"][
                "overall"
            ]
            for run in runs
        },
    }


def _append(path: Path, rows: list[dict]) -> None:
    """Whole-file replacement: a crash cannot leave half a cell."""
    old = path.read_text() if path.exists() else ""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(old)
        stream.write("".join(json.dumps(row) + "\n" for row in rows))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--opponent", help="checkpoint path or heuristic")
    ap.add_argument("--deterministic-opponent", action="store_true")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--pool", nargs="+", type=Path, help="summarize finished runs")
    ap.add_argument("--json", type=Path, default=None, help="with --pool")
    ap.add_argument("--arms", default="plain,search")
    ap.add_argument(
        "--per-category",
        type=int,
        default=10,
        help="rosters a category (10 with the default seed and exclusions: the "
        "held-out battery's own 47 rosters)",
    )
    ap.add_argument(
        "--exclude-matchups",
        default=None,
        help="a study manifest whose rosters are left out (default: the battery's)",
    )
    ap.add_argument("--max-rosters", type=int, default=0, help="0: all of them")
    ap.add_argument(
        "--repeats",
        type=int,
        default=11,
        help="games a cell (roster x sheets) and arm; the battery plays 11",
    )
    ap.add_argument("--seed", type=int, default=20923, help="the battery's")
    ap.add_argument("--port", type=int, default=7630)
    ap.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    ap.add_argument("--cell-timeout", type=float, default=1800)
    ap.add_argument("--a-search-worlds", type=int, default=4)
    ap.add_argument("--a-search-budget", type=float, default=8.0)
    ap.add_argument("--a-search-anchor", type=float, default=0.07)
    ap.add_argument("--a-search-replies", type=int, default=8)
    ap.add_argument("--a-search-streams", type=int, default=0)
    ap.add_argument("--a-search-leaf-calibration", type=Path, default=None)
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    if args.pool:
        pooled = pool(args.pool)
        text = json.dumps(pooled, indent=2)
        if args.json is not None:
            args.json.write_text(text + "\n")
        print(text)
        return
    if not args.opponent or args.output is None:
        ap.error("--opponent and --output are required")
    os.chdir(ROOT)
    arms = [arm for arm in args.arms.split(",") if arm]
    if not arms or not set(arms) <= set(ARMS) or len(set(arms)) != len(arms):
        raise ValueError(f"--arms names {ARMS}")
    if args.repeats < 1 or args.cell_timeout <= 0 or args.max_rosters < 0:
        raise ValueError("sizes and the timeout must be positive")
    if not 1 <= args.a_search_worlds <= 8 or not 0 < args.a_search_budget <= 9:
        raise ValueError("--a-search-worlds must be 1-8, the budget in (0, 9]")
    if not 2 <= args.a_search_replies <= 16 or not 0 <= args.a_search_streams <= 16:
        raise ValueError("--a-search-replies must be 2-16, the streams 0-16")
    if args.a_search_anchor <= 0:
        raise ValueError("the search is the anchored one: --a-search-anchor > 0")
    if args.opponent != "heuristic" and not Path(args.opponent).is_file():
        raise ValueError(f"no opponent checkpoint {args.opponent}")
    calibration = args.a_search_leaf_calibration
    if calibration is not None and not (ROOT / calibration).is_file():
        raise ValueError(f"no calibration file {calibration}")

    from evaluation.opening_study import select_matchups
    from evaluation.run_t6_confirmation import EXCLUDE
    from tools.deployed_config import resolve

    config = resolve()
    exclude = EXCLUDE if args.exclude_matchups is None else args.exclude_matchups
    matchups = select_matchups(args.seed, args.per_category, "heldout", exclude)
    if args.max_rosters:
        matchups = matchups[: args.max_rosters]
    deployed = [g for g in config["GUARDS"].split(",") if g]

    def sha256(path: Path | str) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    manifest = {
        "question": "the deployed bot against a held-out opponent on held-out "
        "rosters, as deployed and with the exact search",
        "opponent": args.opponent,
        "opponent_sha256": (
            None if args.opponent == "heuristic" else sha256(args.opponent)
        ),
        "deterministic_opponent": args.deterministic_opponent,
        "checkpoint": config["CKPT"],
        "checkpoint_sha256": sha256(ROOT / config["CKPT"]),
        "team": config["TEAM"],
        "team_sha256": sha256(ROOT / config["TEAM"]),
        "preview_model": config["PREVIEW_MODEL"],
        "our_guards": deployed,
        "everyone_else_guards": list(BATTERY_EXTRAS),
        "device": args.device,
        "search": {
            "solution": "nash",
            "leaf": "critic",
            "worlds": args.a_search_worlds,
            "budget_s": args.a_search_budget,
            "anchor": args.a_search_anchor,
            "replies": args.a_search_replies,
            "streams": args.a_search_streams,
            "sample": False,
            "leaf_calibration": str(calibration) if calibration else None,
            "leaf_calibration_sha256": (
                sha256(ROOT / calibration) if calibration else None
            ),
        },
        "matchups": matchups,
        "excluded_matchups": exclude,
        "repeats": args.repeats,
        "seed": args.seed,
    }
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(f"{output} belongs to another study")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    rows_path, telemetry_path = output / "rows.jsonl", output / "telemetry.jsonl"
    done = complete_cells(_read(rows_path), args.repeats)
    if args.prepare_only:
        cells = len(matchups) * 2 * len(arms)
        print(json.dumps({"rosters": len(matchups), "cells": cells, "done": len(done)}))
        return

    import torch
    from poke_env.ps_client import ServerConfiguration

    from evaluation.eval_counterfactual import PreviewLedger, _first_faint_side
    from evaluation.mirror_guard_ab import _player
    from evaluation.opening_study import (
        StudyHeuristic,
        StudyOpponent,
        fresh_local_account,
    )
    from vgc_bench.src import pokeenv_patches
    from vgc_bench.src.exact_planner import PlannerConfig
    from vgc_bench.src.guards import GUARDS, HARD_GUARDS
    from vgc_bench.src.policy_player import PolicyPlayer
    from vgc_bench.src.teams import RandomTeamBuilder
    from vgc_bench.src.utils import format_map

    os.environ.pop("VGC_SET_PRIOR_REG", None)
    if config["SET_PRIOR"] != "mc":
        os.environ["VGC_SET_PRIOR_REG"] = config["SET_PRIOR"]
    pokeenv_patches.install()
    torch.set_num_threads(1)
    PolicyPlayer.use_knowledge_obs = PolicyPlayer.use_moveset_prior = True
    PolicyPlayer.use_knowledge_guards = PolicyPlayer.mask_immunities = True
    PolicyPlayer.guard_flags = {
        g: g in HARD_GUARDS or g in BATTERY_EXTRAS for g in GUARDS
    }
    ours_guards = dict.fromkeys(deployed, True)
    search_extra = {
        "enable_search": True,
        "exact_search_config": PlannerConfig(
            depth=2,
            root_width=6,
            opponent_width=args.a_search_replies,
            continuation_width=3,
            replacement_width=2,
            chance_samples=1,
            deep_root_width=4,
            anytime=True,
            screen_budget_s=min(2.0, args.a_search_budget),
            time_budget_s=args.a_search_budget,
            max_nodes=5000,
            solution="nash",
            nash_sample=False,
            nash_anchor=args.a_search_anchor,
        ),
        "outcome_value_path": ROOT / "results_outcome_v2h/outcome_value.zip",
        "exact_team_path": ROOT / config["TEAM"],
        "exact_selective_search": False,
        "exact_max_determinizations": 8,
        "exact_search_determinizations": args.a_search_worlds,
        "exact_min_deep_coverage": 0.5,
        "exact_leaf": "critic",
        "exact_leaf_calibration": ROOT / calibration if calibration else None,
        "exact_min_streams": args.a_search_streams,
    }
    server = ServerConfiguration(
        f"ws://localhost:{args.port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )
    players: dict[bool, dict] = {}
    for hidden in (True, False):
        common: dict[str, Any] = dict(
            server_configuration=server,
            battle_format=format_map["mc"],
            log_level=40,
            max_concurrent_battles=8,
            accept_open_team_sheet=not hidden,
            open_timeout=None,
        )
        foe: Any
        if args.opponent == "heuristic":
            foe = StudyHeuristic(account_configuration=fresh_local_account(), **common)
        else:
            foe = StudyOpponent(
                deterministic=args.deterministic_opponent,
                account_configuration=fresh_local_account(),
                **common,
            )
            foe.set_policy(ROOT / args.opponent, torch.device("cpu"))
        foe.preview_ledger = PreviewLedger()
        foe.replay_previews = False
        sheets = "hidden" if hidden else "open"
        ours = {}
        if "plain" in arms:
            ours["plain"] = _player(
                config, hidden, args.seed, args.port, ours_guards, device=args.device
            )
        if "search" in arms:
            ours["search"] = _player(
                config,
                hidden,
                args.seed,
                args.port,
                ours_guards,
                extra=search_extra
                | {"decision_log_path": output / f"search_{sheets}_decisions.jsonl"},
                concurrency=1,
                device=args.device,
            )
        players[hidden] = {"foe": foe, **ours}

    def stopped() -> bool:
        """A STOP file beside or above the output ends the run at the next cell."""
        return (output / "STOP").exists() or (output.parent / "STOP").exists()

    for index, matchup in enumerate(matchups, 1):
        path = Path(matchup["path"])
        for hidden in (True, False):
            foe = players[hidden]["foe"]
            for arm in arms:
                if (path.name, hidden, arm) in done:
                    continue
                if stopped():
                    summary = summarize(_read(rows_path), _read(telemetry_path))
                    summary["complete"] = False
                    (output / "result.json").write_text(
                        json.dumps(summary, indent=2) + "\n"
                    )
                    print(f"STOP file found before {path.name}; resumable", flush=True)
                    return
                ours = players[hidden][arm]
                foe._team = RandomTeamBuilder(args.seed, 1, "mc", [path])
                PolicyPlayer.guard_fire_counts.clear()
                PolicyPlayer._threat_cache.clear()
                PolicyPlayer._knowledge_cache.clear()
                random.seed(args.seed)
                torch.manual_seed(args.seed)
                started = time.monotonic()
                asyncio.run(
                    asyncio.wait_for(
                        ours.battle_against(foe, n_battles=args.repeats),
                        args.cell_timeout,
                    )
                )
                rows = []
                for battle in ours.battles.values():
                    if not battle.finished:
                        raise RuntimeError(f"unfinished battle {battle.battle_tag}")
                    rows.append(
                        {
                            "opponent": path.name,
                            "category": matchup["category"],
                            "hidden_sheets": hidden,
                            "arm": arm,
                            "battle": battle.battle_tag,
                            # a tie is half a win, as in the head-to-head
                            "target": 0.5 if battle.won is None else float(battle.won),
                            "turns": battle.turn,
                            "first_faint": _first_faint_side(battle),
                        }
                    )
                if len(rows) != args.repeats:
                    raise RuntimeError(
                        f"expected {args.repeats} games, got {len(rows)}"
                    )
                counts = PolicyPlayer.guard_fire_counts
                _append(rows_path, rows)
                _append(
                    telemetry_path,
                    [
                        {
                            "opponent": path.name,
                            "hidden_sheets": hidden,
                            "arm": arm,
                            "elapsed_s": time.monotonic() - started,
                            "search_counts": {
                                key: int(value)
                                for key, value in counts.items()
                                if key.startswith("exact_search")
                            },
                            "errors": {
                                key: int(value)
                                for key, value in counts.items()
                                if "error" in key and not key.startswith("exact_search")
                            },
                        }
                    ],
                )
                ours.reset_battles()
                foe.reset_battles()
                print(
                    f"[{index}/{len(matchups)}] {path.name} {matchup['category']} "
                    f"hidden={hidden} {arm}: "
                    f"{sum(r['target'] for r in rows):g}/{len(rows)}",
                    flush=True,
                )
        summary = summarize(_read(rows_path), _read(telemetry_path))
        summary["complete"] = index == len(matchups)
        (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary = summarize(_read(rows_path), _read(telemetry_path))
    summary["complete"] = True
    (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["search_minus_plain"], indent=2))


if __name__ == "__main__":
    main()
