"""Local promotion A/B for a counterfactual-distilled policy.

Each arm uses the production knowledge guards, opponent models, tempo reranker, fixed
ladder team, opponent-team weights, and deterministic policy inference. Re-seeding
before every arm keeps the team sequence matched as closely as poke-env permits.
Arm names describe what actually ran: preview control and exact move search are
independent and are never implied by a checkpoint filename.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.ps_client import ServerConfiguration
from torch import device

from eval_openings import ShapeAwareBatchPolicyPlayer
from vgc_bench.src.policy_player import BatchPolicyPlayer, PolicyPlayer
from vgc_bench.src.teams import RandomTeamBuilder
from vgc_bench.src.utils import format_map


class DelayedSimpleHeuristicsPlayer(SimpleHeuristicsPlayer):
    """Yield realistic post-submission think time to a pondering opponent."""

    decision_delay_s = 0.0

    async def choose_move(self, battle):
        if self.decision_delay_s > 0:
            await asyncio.sleep(self.decision_delay_s)
        return super().choose_move(battle)

    async def teampreview(self, battle):
        scripted = _scripted_preview(self, battle)
        if scripted is not None:
            return scripted
        choice = super().teampreview(battle)
        choice = await choice if inspect.isawaitable(choice) else choice
        _record_preview(self, battle, choice)
        return choice


class DelayedShapeAwareBatchPolicyPlayer(ShapeAwareBatchPolicyPlayer):
    decision_delay_s = 0.0

    async def choose_move(self, battle):
        if self.decision_delay_s > 0:
            await asyncio.sleep(self.decision_delay_s)
        choice = super().choose_move(battle)
        return await choice if inspect.isawaitable(choice) else choice

    async def teampreview(self, battle):
        scripted = _scripted_preview(self, battle)
        if scripted is not None:
            return scripted
        choice = super().teampreview(battle)
        choice = await choice if inspect.isawaitable(choice) else choice
        _record_preview(self, battle, choice)
        return choice


class PairedBatchPolicyPlayer(BatchPolicyPlayer):
    """Batch policy player whose unchanged Team Preview can be paired in A/Bs."""

    async def _teampreview(self, battle):
        scripted = _scripted_preview(self, battle)
        if scripted is not None:
            return scripted
        choice = await super()._teampreview(battle)
        _record_preview(self, battle, choice)
        return choice


class PreviewLedger:
    """Record the control opponent's draft and replay it in every A/B arm."""

    def __init__(self):
        self._choices: dict[tuple[str, ...], list[str]] = defaultdict(list)
        self._cursors: dict[tuple[str, ...], int] = {}
        self.recorded = 0
        self.replayed = 0

    def record(self, fingerprint: tuple[str, ...], choice: str) -> None:
        self._choices[fingerprint].append(choice)
        self.recorded += 1

    def begin_replay(self) -> None:
        self._cursors.clear()

    def next(self, fingerprint: tuple[str, ...]) -> str:
        cursor = self._cursors.get(fingerprint, 0)
        choices = self._choices.get(fingerprint, [])
        if cursor >= len(choices):
            raise RuntimeError("preview pairing mismatch for " + ",".join(fingerprint))
        self._cursors[fingerprint] = cursor + 1
        self.replayed += 1
        return choices[cursor]


def _first_faint_side(battle) -> str | None:
    """Which side lost the first Pokemon: 'ours', 'theirs', or None (no faint).

    The first KO exchange is the strongest single outcome predictor measured on
    ladder (win rate 25.6% after losing the first mon vs 63.4% after taking it),
    so every gate reports it alongside win rate.
    """
    role = getattr(battle, "player_role", None)
    if role not in ("p1", "p2"):
        return None
    for split_message in getattr(battle, "_replay_data", None) or []:
        if len(split_message) > 2 and split_message[1] == "faint":
            return "ours" if split_message[2][:2] == role else "theirs"
    return None


def _preview_fingerprint(battle) -> tuple[str, ...]:
    def roster(team) -> tuple[str, ...]:
        return tuple(
            "".join(
                character
                for character in str(mon.base_species or mon.species).lower()
                if character.isalnum()
            )
            for mon in (team or {}).values()
        )

    ours = roster(getattr(battle, "team", {}))
    theirs = roster(getattr(battle, "opponent_team", {}))
    return ours + (("__vs__",) + theirs if theirs else ())


def _apply_preview_choice(player, battle, choice: str) -> None:
    digits = [int(value) for value in choice.removeprefix("/team ") if value.isdigit()]
    if len(digits) != 4 or len(set(digits)) != 4:
        raise RuntimeError(f"invalid paired opponent preview: {choice}")
    team = list(battle.team.values())
    for pokemon in team:
        pokemon._selected_in_teampreview = False
    for index in digits:
        team[index - 1]._selected_in_teampreview = True
    recorder = getattr(player, "_record_own_preview", None)
    if recorder is not None:
        recorder(battle, (digits[0], digits[1]), (digits[2], digits[3]))


def _scripted_preview(player, battle) -> str | None:
    ledger = getattr(player, "preview_ledger", None)
    if ledger is None or not getattr(player, "replay_previews", False):
        return None
    choice = ledger.next(_preview_fingerprint(battle))
    _apply_preview_choice(player, battle, choice)
    return choice


def _record_preview(player, battle, choice: str) -> None:
    ledger = getattr(player, "preview_ledger", None)
    if ledger is not None and not getattr(player, "replay_previews", False):
        ledger.record(_preview_fingerprint(battle), choice)


def _interval(wins: int, total: int) -> tuple[float, float]:
    if not total:
        return 0.0, 0.0
    z = 1.96
    p = wins / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1 - p) / max(total, 1) + z * z / (4 * total * total))
        / denominator
    )
    return center - radius, center + radius


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _player(
    args,
    server,
    checkpoint: Path,
    preview: Path,
    learned: bool,
    residual: Path | None = None,
    move_search: bool = False,
    search_audit: Path | None = None,
    replay_dir: Path | None = None,
    selective_search: bool | None = None,
    enable_ponder: bool | None = None,
    planned_preview: bool = False,
    outcome_preview: bool = False,
):
    if replay_dir is not None:
        replay_dir.mkdir(parents=True, exist_ok=True)
    exact_config = None
    ponder_config = None
    if move_search:
        from vgc_bench.src.exact_planner import PlannerConfig
        from vgc_bench.src.ponder import PonderConfig

        exact_config = PlannerConfig(
            depth=2,
            root_width=6,
            opponent_width=6,
            continuation_width=3,
            replacement_width=2,
            chance_samples=args.chance_samples,
            deep_root_width=args.deep_root_width,
            anytime=True,
            screen_budget_s=args.screen_budget,
            time_budget_s=args.search_budget,
            max_nodes=5000,
        )
        ponder_config = PonderConfig(
            budget_s=args.ponder_budget,
            max_opponent_choices=args.ponder_choices,
            chance_samples=args.ponder_chance_samples,
            max_roots=args.determinizations,
        )
    # Wall-clock-budgeted searches are only valid serially: 8 concurrent preview
    # searches sharing one inference lock all blew their deadlines in the 20260823
    # re-gate (300/300 truncated -> every decision silently fell back to champion,
    # making the "exact preview" arm measure champion-vs-champion).
    effective_workers = 1 if (move_search or planned_preview) else args.workers
    agent = PairedBatchPolicyPlayer(
        deterministic=True,
        server_configuration=server,
        battle_format=format_map[args.reg],
        log_level=40,
        max_concurrent_battles=effective_workers,
        accept_open_team_sheet=not args.hidden_sheets,
        open_timeout=None,
        team=RandomTeamBuilder(args.seed, 1, args.reg, [Path(args.our_team)]),
        preview_model_path=preview,
        preview_outcome_model_path=(
            Path(args.preview_outcome_model) if outcome_preview else None
        ),
        switch_model_path=Path(args.switch_model),
        move_model_path=Path(args.move_model),
        use_learned_teampreview=learned,
        use_outcome_teampreview=outcome_preview,
        use_opponent_reranker=True,
        use_tempo_reranker=True,
        residual_ranker_path=residual,
        decision_log_path=search_audit,
        exact_team_path=Path(args.our_team),
        outcome_value_path=(
            Path(args.outcome_value) if move_search or planned_preview else None
        ),
        exact_search_config=exact_config,
        exact_max_determinizations=args.determinizations,
        exact_search_determinizations=2,
        exact_min_deep_coverage=0.50,
        exact_preview_search=planned_preview,
        exact_preview_budget=args.preview_search_budget,
        exact_preview_determinizations=args.preview_determinizations,
        exact_selective_search=move_search
        and (args.selective_search if selective_search is None else selective_search),
        exact_enable_ponder=move_search
        and (args.ponder if enable_ponder is None else enable_ponder),
        exact_ponder_config=ponder_config,
        enable_search=move_search,
        save_replays=str(replay_dir) if replay_dir is not None else False,
    )
    agent.set_policy(checkpoint, device(args.device))
    return agent


def _opponent(
    args,
    server,
    preview_ledger: PreviewLedger | None = None,
    replay_previews: bool = False,
):
    team = RandomTeamBuilder(
        args.seed,
        None,
        args.reg,
        weights_path=Path(args.team_weights),
        sampling_seed=args.seed,
    )
    if not args.opponent_checkpoint:
        foe = DelayedSimpleHeuristicsPlayer(
            server_configuration=server,
            battle_format=format_map[args.reg],
            log_level=40,
            max_concurrent_battles=args.workers,
            accept_open_team_sheet=not args.hidden_sheets,
            open_timeout=None,
            team=team,
        )
        foe.decision_delay_s = args.opponent_think_s
        foe.preview_ledger = preview_ledger
        foe.replay_previews = replay_previews
        return foe
    # A behavior-cloned human imitation must SAMPLE: a deterministic BC collapses
    # to one line per matchup, which is not human-style opposition. The default
    # stays deterministic so paired PPO-opponent arms remain exactly reproducible.
    opponent_metadata = Path(str(args.opponent_checkpoint) + ".metadata.json")
    if opponent_metadata.exists():
        meta = json.loads(opponent_metadata.read_text())
        expected_sha = meta.get("sha256")
        if expected_sha:
            import hashlib

            actual_sha = hashlib.sha256(
                Path(args.opponent_checkpoint).read_bytes()
            ).hexdigest()
            if actual_sha != expected_sha:
                raise SystemExit(
                    f"{opponent_metadata} does not match {args.opponent_checkpoint}; "
                    "restamp with stamp_checkpoint_metadata.py"
                )
    foe = DelayedShapeAwareBatchPolicyPlayer(
        deterministic=not getattr(args, "opponent_stochastic", False),
        server_configuration=server,
        battle_format=format_map[args.reg],
        log_level=40,
        max_concurrent_battles=args.workers,
        accept_open_team_sheet=not args.hidden_sheets,
        open_timeout=None,
        team=team,
    )
    foe.decision_delay_s = args.opponent_think_s
    # CPU inference is slower by only milliseconds for this small policy and avoids
    # MPS kernel variation changing a supposedly paired Team Preview decision.
    foe.set_policy(Path(args.opponent_checkpoint), device(args.opponent_device))
    foe.preview_ledger = preview_ledger
    foe.replay_previews = replay_previews
    return foe


def _run_arm(
    args,
    server,
    name: str,
    checkpoint: Path,
    preview: Path,
    learned_preview: bool,
    move_search: bool = False,
    residual: Path | None = None,
    selective_search: bool | None = None,
    enable_ponder: bool | None = None,
    planned_preview: bool = False,
    outcome_preview: bool = False,
    opponent_preview_ledger: PreviewLedger | None = None,
    replay_opponent_previews: bool = False,
    own_preview_ledger: PreviewLedger | None = None,
    replay_own_previews: bool = False,
    reliability_floor: float | None = None,
    bench_species: tuple[str, ...] | None = None,
) -> dict:
    _seed_everything(args.seed)
    PolicyPlayer.guard_fire_counts.clear()
    PolicyPlayer._decisions_seen = 0
    # Search is an attribute of our player, not a class-wide mode. Otherwise a
    # learned population opponent also enters the exact path without its required
    # team/outcome configuration.
    PolicyPlayer.use_search = False
    search_backend = None
    if move_search:
        from vgc_bench.src import search

        search.SEARCH_LATENCIES_MS.clear()
        ready, search_backend = search.backend_status()
        if not ready:
            raise RuntimeError(
                f"live exact search requested but unavailable: {search_backend}"
            )
    arm_slug = name.replace(" ", "_")
    replay_dir = (
        Path(args.replay_dir) / args.output.stem / arm_slug if args.replay_dir else None
    )
    search_audit = (
        args.output.parent / "search_audits" / f"{args.output.stem}_{arm_slug}.jsonl"
        if move_search
        # When replays are kept, keep the decision audit beside them: pairing
        # decisions with what actually happened next is how rare-firing rules
        # (guard promotions) get validated by counting instead of win rate.
        else (replay_dir / "decisions.jsonl" if replay_dir is not None else None)
    )
    if search_audit is not None:
        search_audit.parent.mkdir(parents=True, exist_ok=True)
        # A resumed evaluation reruns the entire arm with matched seeds. Do not mix
        # stale partial-audit rows from an interrupted attempt into the review.
        search_audit.unlink(missing_ok=True)
    ours = _player(
        args,
        server,
        checkpoint,
        preview,
        learned_preview,
        residual=residual,
        move_search=move_search,
        search_audit=search_audit,
        replay_dir=replay_dir,
        selective_search=selective_search,
        enable_ponder=enable_ponder,
        planned_preview=planned_preview,
        outcome_preview=outcome_preview,
    )
    if reliability_floor is not None:
        # per-arm turn-1/2 reliability floor (Stage C.1 A/B); production uses
        # the VGC_PRIOR_RELIABILITY_FLOOR env var instead
        ours.reliability_floor = reliability_floor
    if bench_species:
        from poke_env.data import to_id_str

        ours.forced_bench_species = tuple(to_id_str(name) for name in bench_species)
    if replay_own_previews and own_preview_ledger is not None:
        own_preview_ledger.begin_replay()
    own_recorded_before = (
        own_preview_ledger.recorded if own_preview_ledger is not None else 0
    )
    own_replayed_before = (
        own_preview_ledger.replayed if own_preview_ledger is not None else 0
    )
    ours.preview_ledger = own_preview_ledger
    ours.replay_previews = replay_own_previews
    if replay_opponent_previews and opponent_preview_ledger is not None:
        opponent_preview_ledger.begin_replay()
    recorded_before = (
        opponent_preview_ledger.recorded if opponent_preview_ledger is not None else 0
    )
    replayed_before = (
        opponent_preview_ledger.replayed if opponent_preview_ledger is not None else 0
    )
    foe = _opponent(
        args,
        server,
        preview_ledger=opponent_preview_ledger,
        replay_previews=replay_opponent_previews,
    )
    started = time.perf_counter()
    asyncio.run(ours.battle_against(foe, n_battles=args.n_battles))
    elapsed = time.perf_counter() - started
    wins = ours.n_won_battles
    total = ours.n_finished_battles
    low, high = _interval(wins, total)
    telemetry = dict(PolicyPlayer.guard_fire_counts)
    search_payload = {
        "enabled": move_search,
        "backend": search_backend,
        "configuration": {
            "normal_turn_budget_s": args.search_budget if move_search else None,
            "screen_budget_s": args.screen_budget if move_search else None,
            "submission_reserve_s": 1.0 if move_search else None,
            "risk_weights": [0.60, 0.30, 0.10] if move_search else None,
            "chance_samples": args.chance_samples if move_search else None,
            "maximum_determinizations": args.determinizations if move_search else None,
            "selective_search": (
                args.selective_search if selective_search is None else selective_search
            )
            if move_search
            else None,
            "ponder": (args.ponder if enable_ponder is None else enable_ponder)
            if move_search
            else None,
            "opponent_think_s": args.opponent_think_s,
            "requested_workers": args.workers,
            "effective_workers": (
                1 if (move_search or planned_preview) else args.workers
            ),
        },
        "decisions": int(telemetry.get("exact_search", 0)),
        "fallbacks": sum(
            int(count)
            for key, count in telemetry.items()
            if key.startswith("exact_search_error")
        ),
        "truncations": int(telemetry.get("exact_search_truncated", 0)),
        "audit": str(search_audit) if search_audit is not None else None,
    }
    if move_search:
        from vgc_bench.src import search

        latencies = sorted(search.SEARCH_LATENCIES_MS)
        search_payload["latency_ms"] = {
            "count": len(latencies),
            "p50": latencies[min(len(latencies) - 1, len(latencies) // 2)]
            if latencies
            else None,
            "p90": latencies[min(len(latencies) - 1, int(len(latencies) * 0.9))]
            if latencies
            else None,
            "max": latencies[-1] if latencies else None,
        }
        audit_rows = []
        if search_audit is not None and search_audit.exists():
            for line in search_audit.read_text().splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                exact = record.get("exact_search")
                if isinstance(exact, dict):
                    audit_rows.append(exact)
        schedules = [row.get("schedule") or {} for row in audit_rows]
        schedule_modes = {
            mode: sum(schedule.get("mode") == mode for schedule in schedules)
            for mode in (
                "search",
                "search_fallback",
                "error_fallback",
                "reuse",
                "skip_on_plan",
                "skip_on_ponder",
                "skip",
            )
        }
        error_fallbacks = schedule_modes["error_fallback"]
        # Older audit files predate the explicit error schedule. Retain telemetry as
        # a compatibility fallback without double-counting new rows.
        if not error_fallbacks:
            error_fallbacks = sum(
                int(count)
                for key, count in telemetry.items()
                if key.startswith("exact_search_error")
            )
        attempted = (
            schedule_modes["search"]
            + schedule_modes["search_fallback"]
            + error_fallbacks
        )
        shallow_fallbacks = schedule_modes["search_fallback"]
        search_payload.update(
            {
                "audit_decisions": len(audit_rows),
                "schedule_modes": schedule_modes,
                "fallbacks": error_fallbacks + shallow_fallbacks,
                "error_fallbacks": error_fallbacks,
                "shallow_search_fallbacks": shallow_fallbacks,
                "future_depth_completion_rate": (
                    schedule_modes["search"] / attempted if attempted else None
                ),
                "live_illegal_choices": sum(
                    len(schedule.get("live_illegal_choices") or [])
                    for schedule in schedules
                ),
                "live_legality_changed_picks": sum(
                    bool(schedule.get("live_legality_changed_pick"))
                    for schedule in schedules
                ),
                "live_safety_changed_picks": sum(
                    bool(schedule.get("live_safety_changed_pick"))
                    for schedule in schedules
                ),
            }
        )
    # First-class preview-search accounting: an exact-preview arm whose searches all
    # truncate plays champion preview in every battle. The 20260823 re-gate reported
    # such an arm as a valid measurement; this block makes that failure loud.
    preview_payload = {
        "enabled": planned_preview,
        "budget_s": args.preview_search_budget if planned_preview else None,
        "determinizations": (
            getattr(args, "preview_determinizations", 1) if planned_preview else None
        ),
        "decisions": int(telemetry.get("exact_preview", 0)),
        "truncated_fallbacks": int(telemetry.get("exact_preview_truncated", 0)),
        "missing_dependency": int(telemetry.get("exact_preview_missing_dependency", 0)),
        "errors": sum(
            int(count)
            for key, count in telemetry.items()
            if key.startswith("exact_preview_error")
        ),
    }
    if planned_preview and preview_payload["decisions"] == 0:
        print(
            f"WARNING: {name}: exact preview made ZERO decisions "
            f"(truncated={preview_payload['truncated_fallbacks']}, "
            f"errors={preview_payload['errors']}); every Team Preview fell back to "
            "the champion path, so this arm does NOT measure exact preview",
            flush=True,
        )
    print(
        f"{name:28} {wins:3d}/{total:3d} = {wins / total * 100:5.1f}% "
        f"95% CI [{low * 100:4.1f}, {high * 100:4.1f}] ({elapsed:.1f}s)",
        flush=True,
    )
    from vgc_bench.src.preview_rules import trick_room_probability

    battle_results = []
    first_faint_ours = 0
    first_faint_theirs = 0
    for tag, battle in sorted(ours.battles.items()):
        opponent_roster = [
            str(mon.base_species or mon.species)
            for mon in battle.opponent_team.values()
        ]
        faint_side = _first_faint_side(battle)
        if faint_side == "ours":
            first_faint_ours += 1
        elif faint_side == "theirs":
            first_faint_theirs += 1
        opponent_tr = trick_room_probability(opponent_roster)
        battle_results.append(
            {
                "battle": tag,
                "won": bool(battle.won),
                "lost": bool(battle.lost),
                "turns": int(battle.turn),
                "opponent": getattr(battle, "opponent_username", None),
                "opponent_team": [
                    str(mon.species) for mon in battle.opponent_team.values()
                ],
                "first_faint_side": faint_side,
                "opponent_trick_room_probability": round(opponent_tr, 4),
                "opponent_is_tr": opponent_tr >= 0.6,
            }
        )
    return {
        "checkpoint": str(checkpoint),
        "preview_model": str(preview),
        "preview_control": (
            "live_exact_first_turn"
            if planned_preview
            else (
                "terminal_outcome_hidden"
                if args.hidden_sheets
                else "champion_policy_open_sheet_fallback"
            )
            if outcome_preview
            else "learned"
            if learned_preview
            else "champion_policy"
        ),
        "move_search": search_payload,
        "preview_search": preview_payload,
        "residual_ranker": str(residual) if residual is not None else None,
        "wins": wins,
        "battles": total,
        "win_rate": wins / max(total, 1),
        "wilson_95": [low, high],
        "elapsed_seconds": elapsed,
        "telemetry": telemetry,
        "loss_shape": {
            "first_faint_ours": first_faint_ours,
            "first_faint_theirs": first_faint_theirs,
            "win_rate_after_first_faint_ours": (
                sum(
                    1
                    for row in battle_results
                    if row["won"] and row["first_faint_side"] == "ours"
                )
                / first_faint_ours
                if first_faint_ours
                else None
            ),
            "tr_battles": sum(1 for row in battle_results if row["opponent_is_tr"]),
            "tr_wins": sum(
                1 for row in battle_results if row["opponent_is_tr"] and row["won"]
            ),
        },
        "resolved_flags": {
            "use_knowledge_obs": PolicyPlayer.knowledge_obs_enabled(),
            "knowledge_obs_env": os.environ.get("VGC_KNOWLEDGE_OBS"),
            "mask_immunities": bool(PolicyPlayer.mask_immunities),
            "use_knowledge_guards": bool(PolicyPlayer.use_knowledge_guards),
            "use_moveset_prior": bool(PolicyPlayer.use_moveset_prior),
        },
        "battle_results": battle_results,
        "opponent_preview_pairing": {
            "mode": "replay" if replay_opponent_previews else "record",
            "choices": (
                opponent_preview_ledger.replayed - replayed_before
                if replay_opponent_previews and opponent_preview_ledger is not None
                else opponent_preview_ledger.recorded - recorded_before
                if opponent_preview_ledger is not None
                else 0
            ),
            "opponent_device": args.opponent_device,
        },
        "own_preview_pairing": {
            "mode": "replay" if replay_own_previews else "record",
            "choices": (
                own_preview_ledger.replayed - own_replayed_before
                if replay_own_previews and own_preview_ledger is not None
                else own_preview_ledger.recorded - own_recorded_before
                if own_preview_ledger is not None
                else 0
            ),
        },
        "replay_dir": str(replay_dir) if replay_dir is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="results_repaired/champion.zip")
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--candidate-residual",
        default="",
        help="confidence-gated residual that defines the distilled-policy arm",
    )
    parser.add_argument(
        "--baseline-preview", default="data/opponent_preview_top500_regmb.pt"
    )
    parser.add_argument("--candidate-preview", default="")
    parser.add_argument(
        "--candidate-preview-trained",
        action="store_true",
        help="evaluate candidate preview control only when this run trained it",
    )
    parser.add_argument(
        "--include-live-search",
        action="store_true",
        help="add a separately named live exact-search arm; refuses unsafe backends",
    )
    parser.add_argument(
        "--include-exact-preview",
        action="store_true",
        help="add the bounded exact Team Preview arm",
    )
    parser.add_argument(
        "--include-outcome-preview",
        action="store_true",
        help="add the terminal-outcome Team Preview arm",
    )
    parser.add_argument(
        "--preview-outcome-model", default="data/preview_outcome_regmb.pt"
    )
    parser.add_argument(
        "--switch-model", default="data/opponent_switch_top500_regmb.pt"
    )
    parser.add_argument("--move-model", default="data/opponent_move_top500_regmb.pt")
    parser.add_argument("--opponent-checkpoint", default="")
    parser.add_argument(
        "--candidate-bench-species",
        default="",
        help=(
            "give ONLY the candidate arm this comma-separated forced bench "
            "(bring-selection A/B; baseline drafts freely)"
        ),
    )
    parser.add_argument(
        "--candidate-reliability-floor",
        type=float,
        default=None,
        help=(
            "give ONLY the candidate arm this turn-1/2 prior reliability floor "
            "(Stage C.1 A/B; baseline keeps revealed-moves-only reliability)"
        ),
    )
    parser.add_argument(
        "--opponent-stochastic",
        action="store_true",
        help=(
            "sample the learned opponent instead of playing it deterministically; "
            "REQUIRED for behavior-cloned human-imitation opponents, which "
            "collapse to one line per matchup when deterministic"
        ),
    )
    parser.add_argument("--our-team", default="teams/reg_mb/our_team.txt")
    parser.add_argument(
        "--outcome-value", default="results_outcome_v2h/outcome_value.zip"
    )
    parser.add_argument("--team-weights", default="data/team_weights_regmb.json")
    parser.add_argument("--reg", default="mb")
    parser.add_argument("--port", type=int, default=7600)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--opponent-device",
        default="cpu",
        help="deterministic device for the paired local opponent",
    )
    parser.add_argument("--n-battles", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--search-budget", type=float, default=9.0)
    parser.add_argument("--preview-search-budget", type=float, default=8.0)
    parser.add_argument(
        "--preview-determinizations",
        type=int,
        default=1,
        help=(
            "hidden-set worlds the exact preview planner samples and merges "
            "(VGC Timer allows 90s at Team Preview, so several full worlds fit)"
        ),
    )
    parser.add_argument("--screen-budget", type=float, default=2.0)
    parser.add_argument(
        "--chance-samples",
        type=int,
        default=1,
        help="shared exact-simulator RNG samples per branch (production default: 1)",
    )
    parser.add_argument(
        "--deep-root-width",
        type=int,
        default=4,
        help="number of screened root choices that receive full future expansion",
    )
    parser.add_argument("--determinizations", type=int, default=8)
    parser.add_argument(
        "--selective-search",
        action="store_true",
        help="enable contingent-plan and searched-position reuse in the live arm",
    )
    parser.add_argument(
        "--ponder",
        action="store_true",
        help="enable post-submission background expansion in the live arm",
    )
    parser.add_argument("--ponder-budget", type=float, default=6.0)
    parser.add_argument("--ponder-choices", type=int, default=96)
    parser.add_argument("--ponder-chance-samples", type=int, default=1)
    parser.add_argument(
        "--compare-search-every-turn",
        action="store_true",
        help="add a live exact arm that recalculates every turn",
    )
    parser.add_argument(
        "--search-comparison-only",
        action="store_true",
        help="skip policy-only arms when comparing exact-search schedules",
    )
    parser.add_argument(
        "--preview-comparison-only",
        action="store_true",
        help="run only champion Team Preview versus bounded exact Team Preview",
    )
    parser.add_argument(
        "--opponent-think-s",
        type=float,
        default=0.0,
        help="local opponent delay available to post-submission pondering",
    )
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--hidden-sheets", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--replay-dir",
        default="",
        help="optional root directory for per-arm local replay files",
    )
    args = parser.parse_args()

    from vgc_bench.src import pokeenv_patches
    from vgc_bench.src.guards import GUARDS, HARD_GUARDS

    pokeenv_patches.install()
    os.environ["VGC_KNOWLEDGE_OBS"] = "1"
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

    baseline = Path(args.baseline)
    candidate = Path(args.candidate)
    candidate_residual = (
        Path(args.candidate_residual) if args.candidate_residual else None
    )
    baseline_preview = Path(args.baseline_preview)
    candidate_preview = Path(args.candidate_preview) if args.candidate_preview else None
    required = [baseline, candidate, baseline_preview]
    if candidate_residual is not None:
        required.append(candidate_residual)
    if args.candidate_preview_trained:
        if candidate_preview is None:
            raise SystemExit("--candidate-preview-trained requires --candidate-preview")
        required.append(candidate_preview)
    if args.include_live_search:
        required.append(Path(args.outcome_value))
    if args.include_exact_preview:
        required.append(Path(args.outcome_value))
    if args.include_outcome_preview:
        required.append(Path(args.preview_outcome_model))
    for path in required:
        if not path.exists():
            raise SystemExit(f"required model not found: {path}")

    if args.search_comparison_only and not args.include_live_search:
        raise SystemExit("--search-comparison-only requires --include-live-search")
    if args.search_comparison_only and candidate.resolve() != baseline.resolve():
        raise SystemExit(
            "--search-comparison-only requires --candidate to equal --baseline so "
            "the only changed variable is live move search"
        )
    if args.search_comparison_only and args.candidate_preview_trained:
        raise SystemExit(
            "--search-comparison-only cannot enable a newly trained preview model"
        )
    if args.preview_comparison_only and not (
        args.include_exact_preview or args.include_outcome_preview
    ):
        raise SystemExit("--preview-comparison-only requires a preview candidate arm")
    if not 0 < args.search_budget <= 9:
        # PlannerConfig now admits Team Preview budgets up to 60s; the 10s ladder
        # move cap is enforced here for move search instead.
        raise SystemExit("--search-budget must be in (0, 9]")
    if args.preview_comparison_only and args.search_comparison_only:
        raise SystemExit("choose only one comparison-only mode")
    arms = {}
    opponent_preview_ledger = PreviewLedger()
    own_preview_ledger = PreviewLedger()
    if args.preview_comparison_only:
        arms["champion_policy"] = _run_arm(
            args,
            server,
            "champion policy",
            baseline,
            baseline_preview,
            False,
            opponent_preview_ledger=opponent_preview_ledger,
        )
    elif args.search_comparison_only:
        arms["champion_policy"] = _run_arm(
            args,
            server,
            "champion policy",
            baseline,
            baseline_preview,
            False,
            opponent_preview_ledger=opponent_preview_ledger,
            own_preview_ledger=own_preview_ledger,
        )
    elif not args.search_comparison_only:
        arms.update(
            {
                "champion_policy": _run_arm(
                    args,
                    server,
                    "champion policy",
                    baseline,
                    baseline_preview,
                    False,
                    opponent_preview_ledger=opponent_preview_ledger,
                ),
                "distilled_policy": _run_arm(
                    args,
                    server,
                    "distilled policy",
                    candidate,
                    baseline_preview,
                    False,
                    residual=candidate_residual,
                    opponent_preview_ledger=opponent_preview_ledger,
                    replay_opponent_previews=True,
                    reliability_floor=args.candidate_reliability_floor,
                    bench_species=(
                        tuple(
                            name.strip()
                            for name in args.candidate_bench_species.split(",")
                            if name.strip()
                        )
                        or None
                    ),
                ),
            }
        )
    if args.candidate_preview_trained:
        assert candidate_preview is not None
        arms["preview_policy"] = _run_arm(
            args,
            server,
            "preview policy",
            candidate,
            candidate_preview,
            True,
            residual=candidate_residual,
            opponent_preview_ledger=opponent_preview_ledger,
            replay_opponent_previews=True,
        )
    if args.include_exact_preview:
        arms["live_exact_preview"] = _run_arm(
            args,
            server,
            "live exact preview",
            candidate,
            baseline_preview,
            False,
            residual=candidate_residual,
            planned_preview=True,
            opponent_preview_ledger=opponent_preview_ledger,
            replay_opponent_previews=True,
        )
    if args.include_outcome_preview:
        outcome_arm = (
            "terminal_outcome_preview"
            if args.hidden_sheets
            else "champion_open_sheet_fallback"
        )
        arms[outcome_arm] = _run_arm(
            args,
            server,
            (
                "terminal outcome preview"
                if args.hidden_sheets
                else "champion open-sheet fallback"
            ),
            candidate,
            baseline_preview,
            False,
            residual=candidate_residual,
            outcome_preview=True,
            opponent_preview_ledger=opponent_preview_ledger,
            replay_opponent_previews=True,
        )
    if args.include_live_search:
        preview = (
            candidate_preview if args.candidate_preview_trained else baseline_preview
        )
        assert preview is not None
        arms["live_exact_search"] = _run_arm(
            args,
            server,
            "live exact search",
            candidate,
            preview,
            args.candidate_preview_trained,
            move_search=True,
            residual=candidate_residual,
            opponent_preview_ledger=opponent_preview_ledger,
            replay_opponent_previews=True,
            own_preview_ledger=(
                own_preview_ledger if args.search_comparison_only else None
            ),
            replay_own_previews=args.search_comparison_only,
        )
        if args.compare_search_every_turn:
            arms["live_exact_search_every_turn"] = _run_arm(
                args,
                server,
                "live exact search every turn",
                candidate,
                preview,
                args.candidate_preview_trained,
                move_search=True,
                residual=candidate_residual,
                selective_search=False,
                enable_ponder=False,
                opponent_preview_ledger=opponent_preview_ledger,
                replay_opponent_previews=True,
                own_preview_ledger=(
                    own_preview_ledger if args.search_comparison_only else None
                ),
                replay_own_previews=args.search_comparison_only,
            )
    payload = {
        "evaluation_schema": 3,
        "hidden_sheets": args.hidden_sheets,
        "opponent_checkpoint": args.opponent_checkpoint or None,
        "seed": args.seed,
        "n_battles": args.n_battles,
        "arms": arms,
        "candidate_preview_trained": args.candidate_preview_trained,
        "live_search_requested": args.include_live_search,
        "exact_preview_requested": args.include_exact_preview,
        "outcome_preview_requested": args.include_outcome_preview,
        "search_every_turn_comparison": args.compare_search_every_turn,
        "search_comparison_only": args.search_comparison_only,
        "preview_comparison_only": args.preview_comparison_only,
        "opponent_preview_pairing": {
            "recorded": opponent_preview_ledger.recorded,
            "replayed": opponent_preview_ledger.replayed,
            "opponent_device": args.opponent_device,
        },
        "own_preview_pairing": {
            "enabled": args.search_comparison_only,
            "recorded": own_preview_ledger.recorded,
            "replayed": own_preview_ledger.replayed,
        },
        "patch_report": pokeenv_patches.report(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
