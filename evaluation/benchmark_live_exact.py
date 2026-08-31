"""Measure end-to-end live exact-search latency on local Showdown battles."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from vgc_bench.src.exact_observation import choice_to_actions, state_to_battle
from vgc_bench.src.exact_planner import PlannerConfig
from vgc_bench.src.exact_sim import ExactShowdownBridge
from vgc_bench.src.live_exact import LiveExactSession
from vgc_bench.src.opponent_tactics import MovePredictor, SwitchPredictor
from vgc_bench.src.outcome_value import OutcomeValueEvaluator
from vgc_bench.src.ponder import PonderConfig

ROOT = Path(__file__).resolve().parent
FORMAT = "gen9championsvgc2026regmb"


def _seed(rng: random.Random) -> list[int]:
    return [rng.randrange(1, 65536) for _ in range(4)]


def _preview(rng: random.Random) -> str:
    return "team " + ",".join(str(index) for index in rng.sample(range(1, 7), 4))


def _advance_non_move(bridge, result, rng):
    while not result["ended"] and result["request_state"] != "move":
        p1 = bridge.choices(result["state"], "p1")[0]
        p2 = bridge.choices(result["state"], "p2")[0]
        result = bridge.simulate(
            result["state"], p1, p2, ",".join(map(str, _seed(rng)))
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=int, default=12)
    parser.add_argument("--budget", type=float, default=9.0)
    parser.add_argument("--screen-budget", type=float, default=2.0)
    parser.add_argument("--chance-samples", type=int, default=4)
    parser.add_argument("--deep-root-width", type=int, default=4)
    parser.add_argument("--determinizations", type=int, default=8)
    parser.add_argument("--search-determinizations", type=int, default=2)
    parser.add_argument("--min-deep-coverage", type=float, default=0.50)
    parser.add_argument(
        "--selective",
        action="store_true",
        help="benchmark contingent-plan reuse and quiet-turn search skipping",
    )
    parser.add_argument(
        "--follow-plan-branch",
        action="store_true",
        help="diagnostic: make the local opponent choose a saved searched branch",
    )
    parser.add_argument("--ponder", action="store_true")
    parser.add_argument("--ponder-budget", type=float, default=6.0)
    parser.add_argument("--ponder-choices", type=int, default=96)
    parser.add_argument("--ponder-chance-samples", type=int, default=1)
    parser.add_argument(
        "--opponent-think-s",
        type=float,
        default=0.0,
        help="local delay after submission so the background worker can ponder",
    )
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument(
        "--move-model",
        type=Path,
        default=ROOT / "data/opponent_move_top500_regmb.pt",
    )
    parser.add_argument(
        "--switch-model",
        type=Path,
        default=ROOT / "data/opponent_switch_top500_regmb.pt",
    )
    parser.add_argument(
        "--sheet-mode",
        choices=("mixed", "open", "hidden"),
        default="mixed",
    )
    parser.add_argument(
        "--max-decisions-per-game",
        type=int,
        default=0,
        help="cycle matchups after this many decisions (0 keeps each full game)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results_parity/live_search_latency.json",
    )
    args = parser.parse_args()
    rng = random.Random(args.seed)
    our_team = (ROOT / "teams/reg_mb/our_team.txt").read_text()
    opponents = sorted((ROOT / "teams/reg_mb").glob("MB*.txt"))
    policy = PPO.load(ROOT / "results_repaired/champion.zip", device="cpu").policy
    evaluator = OutcomeValueEvaluator.load(
        ROOT / "results_outcome_v1/outcome_value.zip", device="cpu"
    )
    move_predictor = MovePredictor.load(args.move_model, device="cpu")
    switch_predictor = SwitchPredictor.load(args.switch_model, device="cpu")
    config = PlannerConfig(
        depth=2,
        root_width=6,
        opponent_width=6,
        continuation_width=3,
        replacement_width=2,
        chance_samples=args.chance_samples,
        deep_root_width=args.deep_root_width,
        anytime=True,
        screen_budget_s=args.screen_budget,
        time_budget_s=args.budget,
        max_nodes=5000,
    )
    rows = []
    source_bridge = ExactShowdownBridge()
    try:
        game = 0
        while len(rows) < args.decisions:
            opponent = opponents[game % len(opponents)]
            open_sheet = (
                args.sheet_mode == "open"
                or (args.sheet_mode == "mixed" and game % 2 == 0)
            )
            source = source_bridge.create(
                formatid=FORMAT,
                seed=_seed(rng),
                p1_team_text=our_team,
                p2_team_text=opponent.read_text(),
                p1_preview=_preview(rng),
                p2_preview=_preview(rng),
            )
            setup_started = time.monotonic()
            session = LiveExactSession(
                battle_tag=f"latency-{game}",
                policy=policy,
                our_team_text=our_team,
                formatid=FORMAT,
                open_sheet=open_sheet,
                outcome_value_path=ROOT / "results_outcome_v1/outcome_value.zip",
                outcome_evaluator=evaluator,
                move_predictor=move_predictor,
                switch_predictor=switch_predictor,
                device="cpu",
                config=config,
                max_determinizations=args.determinizations,
                search_determinizations=args.search_determinizations,
                min_deep_coverage=args.min_deep_coverage,
                selective_search=args.selective,
                enable_ponder=args.ponder,
                ponder_config=PonderConfig(
                    budget_s=args.ponder_budget,
                    max_opponent_choices=args.ponder_choices,
                    chance_samples=args.ponder_chance_samples,
                    max_roots=8,
                ),
            )
            setup_s = time.monotonic() - setup_started
            try:
                game_decisions = 0
                while (
                    not source["ended"]
                    and len(rows) < args.decisions
                    and (
                        args.max_decisions_per_game <= 0
                        or game_decisions < args.max_decisions_per_game
                    )
                ):
                    battle = state_to_battle(
                        source["state"],
                        source["requests"],
                        "p1",
                        reveal_opponent_sets=open_sheet,
                    )
                    started = time.monotonic()
                    planning_error = None
                    try:
                        plan = session.plan(battle)
                    except Exception as exc:
                        plan = None
                        planning_error = f"{type(exc).__name__}: {exc}"
                        session.last_schedule = {
                            "mode": "search_fallback",
                            "reasons": ["planner_error_use_champion"],
                            "error": planning_error,
                        }
                    elapsed = time.monotonic() - started
                    mode = session.last_schedule["mode"]
                    if plan is None:
                        p1_choices = source_bridge.choices(source["state"], "p1")
                        ranked = session.adapter.rank(
                            source["state"], source["requests"], "p1", p1_choices
                        )
                        selected = None
                        for candidate in ranked:
                            try:
                                live_actions = choice_to_actions(
                                    candidate.choice,
                                    source["requests"][0],
                                    state=source["state"],
                                    role="p1",
                                    battle=battle,
                                )
                            except Exception:
                                continue
                            if session._action_pair_is_live_legal(
                                live_actions, battle
                            ):
                                selected = (candidate, live_actions)
                                break
                        if selected is None:
                            raise RuntimeError(
                                "champion fallback produced no live-legal action"
                            )
                        selected_choice = selected[0].choice
                        selected_actions = selected[1]
                    else:
                        selected_actions = plan.actions or (0, 0)
                        # A refreshed live root can have a different internal party
                        # order from the authoritative source battle. Production
                        # submits poke-env action IDs, not the root's Showdown switch
                        # indexes, so make the benchmark perform the same identity-
                        # based round trip before advancing its source state.
                        selected_choice = None
                        for candidate_choice in source_bridge.choices(
                            source["state"], "p1"
                        ):
                            try:
                                candidate_actions = choice_to_actions(
                                    candidate_choice,
                                    source["requests"][0],
                                    state=source["state"],
                                    role="p1",
                                    battle=battle,
                                )
                            except Exception:
                                continue
                            if tuple(candidate_actions) == tuple(selected_actions):
                                selected_choice = candidate_choice
                                break
                        if selected_choice is None:
                            raise RuntimeError(
                                "exact result did not round-trip to the source battle"
                            )
                    first_decision = session.reconciliations == 0
                    total = elapsed + (setup_s if first_decision else 0)
                    rows.append(
                        {
                            "game": game,
                            "turn": source["turn"],
                            "open_sheet": open_sheet,
                            "elapsed_s": elapsed,
                            "end_to_end_s": total,
                            "mode": mode,
                            "schedule": dict(session.last_schedule),
                            "ponder": (
                                dict(session.last_ponder)
                                if session.last_ponder is not None
                                else None
                            ),
                            "nodes": plan.nodes if plan is not None else 0,
                            "completed_depth": (
                                plan.completed_depth if plan is not None else 0
                            ),
                            "selected_depth_coverage": float(
                                session.last_schedule.get(
                                    "selected_depth_coverage",
                                    plan.selected_depth_coverage
                                    if plan is not None
                                    else 0.0,
                                )
                            ),
                            "roots": len(session.roots),
                            "truncated": plan.truncated if plan is not None else False,
                            "fallback_reason": (
                                plan.fallback_reason if plan is not None else None
                            ),
                            "saved_continuations": len(
                                session.planned_continuations
                            ),
                            "fallbacks": session.fallbacks,
                            "root_reconcile_failures": (
                                session.root_reconcile_failures
                            ),
                            "root_refreshes": session.root_refreshes,
                            "last_reconcile_errors": list(
                                session.last_reconcile_errors
                            ),
                            "planning_error": planning_error,
                        }
                    )
                    game_decisions += 1
                    print(
                        f"decision {len(rows)}/{args.decisions}: "
                        f"{'open' if open_sheet else 'hidden'} "
                        f"{mode} {total:.3f}s "
                        f"nodes={plan.nodes if plan is not None else 0} "
                        f"roots={len(session.roots)}",
                        flush=True,
                    )
                    record_started = time.monotonic()
                    session.record_actions(selected_actions)
                    rows[-1]["submission_overhead_s"] = (
                        time.monotonic() - record_started
                    )
                    ponder_started = time.monotonic()
                    session.start_pending_ponder()
                    rows[-1]["post_submit_ponder_start_s"] = (
                        time.monotonic() - ponder_started
                    )
                    if args.opponent_think_s > 0:
                        time.sleep(args.opponent_think_s)
                    p2_choices = source_bridge.choices(source["state"], "p2")
                    followed = next(
                        (
                            item.opponent_choice
                            for item in session.planned_continuations
                            if item.opponent_choice in p2_choices
                        ),
                        None,
                    )
                    if args.follow_plan_branch and followed is not None:
                        p2 = followed
                    else:
                        p2_ranked = session.prior.rank(
                            source["state"], source["requests"], "p2", p2_choices
                        )
                        p2 = p2_ranked[0].choice if p2_ranked else p2_choices[0]
                    rows[-1]["p1_choice"] = selected_choice
                    rows[-1]["p2_choice"] = p2
                    source = source_bridge.simulate(
                        source["state"],
                        selected_choice,
                        p2,
                        ",".join(map(str, _seed(rng))),
                    )
                    source = _advance_non_move(source_bridge, source, rng)
            finally:
                session.close()
            game += 1
    finally:
        source_bridge.close()

    latencies = np.asarray([row["end_to_end_s"] for row in rows])
    search_attempts = [
        row for row in rows if row["mode"] in {"search", "search_fallback"}
    ]
    completed_future = [
        row
        for row in search_attempts
        if row["mode"] == "search"
        and row["selected_depth_coverage"] + 1e-9 >= args.min_deep_coverage
    ]
    illegal_choices = sum(
        len(row["schedule"].get("live_illegal_choices") or []) for row in rows
    )
    legality_changed_picks = sum(
        bool(row["schedule"].get("live_legality_changed_pick")) for row in rows
    )
    safety_changed_picks = sum(
        bool(row["schedule"].get("live_safety_changed_pick")) for row in rows
    )
    future_completion_rate = len(completed_future) / max(1, len(search_attempts))
    payload = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "configuration": {
            "budget_s": args.budget,
            "screen_budget_s": args.screen_budget,
            "chance_samples": args.chance_samples,
            "deep_root_width": args.deep_root_width,
            "determinizations": args.determinizations,
            "search_determinizations": args.search_determinizations,
            "min_deep_coverage": args.min_deep_coverage,
            "decisions": args.decisions,
            "sheet_mode": args.sheet_mode,
            "selective": args.selective,
            "follow_plan_branch": args.follow_plan_branch,
            "max_decisions_per_game": args.max_decisions_per_game,
            "ponder": args.ponder,
            "ponder_budget_s": args.ponder_budget,
            "ponder_choices": args.ponder_choices,
            "ponder_chance_samples": args.ponder_chance_samples,
            "opponent_think_s": args.opponent_think_s,
            "move_model": str(args.move_model),
            "switch_model": str(args.switch_model),
        },
        "modes": {
            mode: sum(row["mode"] == mode for row in rows)
            for mode in (
                "search",
                "search_fallback",
                "reuse",
                "skip_on_plan",
                "skip_on_ponder",
                "skip",
            )
        },
        "p50_s": float(np.quantile(latencies, 0.50)),
        "p90_s": float(np.quantile(latencies, 0.90)),
        "max_s": float(latencies.max()),
        "missed_submissions": 0,
        "future_completion_rate": future_completion_rate,
        "live_illegal_choices": illegal_choices,
        "live_legality_changed_picks": legality_changed_picks,
        "live_safety_changed_picks": safety_changed_picks,
        "accepted": bool(
            len(rows) == args.decisions
            and np.quantile(latencies, 0.90) <= 9.0
            and latencies.max() <= 10.0
            and future_completion_rate >= 0.90
            and illegal_choices == 0
            and legality_changed_picks == 0
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"latency p50={payload['p50_s']:.3f}s p90={payload['p90_s']:.3f}s "
        f"max={payload['max_s']:.3f}s accepted={payload['accepted']}"
    )
    if not payload["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
