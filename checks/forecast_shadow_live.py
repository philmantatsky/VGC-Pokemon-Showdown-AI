"""Live check of shadow mode (2026-10-05, PolicyPlayer opponent_forecast_path): the
deployed configuration as tools/deployed_config.py resolves it, on a local server,
against a heuristic opponent -- one game with open team sheets and one without.

Passes when every move decision in the decision log carries an opponent forecast for
its own turn (or a named stand-down), the open-sheet game's forecasts say both sheets
were known and the closed game's say closed, and no shadow-mode failure was counted.
The forecast changes no decision; this only checks that it is logged on the real
game path (connection, requests, _guarded_action, the audit).

Usage (repo root, a local Showdown server on <port>):
  .venv/bin/python checks/forecast_shadow_live.py <port> <decision log> <opponent team>
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from poke_env.ps_client import ServerConfiguration

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.opening_study import StudyHeuristic, fresh_local_account  # noqa: E402
from tools.deployed_config import resolve  # noqa: E402
from vgc_bench.src.policy_player import PolicyPlayer  # noqa: E402
from vgc_bench.src.teams import RandomTeamBuilder  # noqa: E402
from vgc_bench.src.utils import format_map  # noqa: E402


def main() -> int:
    port, log, opponent_team = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    if log.exists():
        print(f"SHADOW_LIVE_FAILED {log} already exists; give a fresh decision log")
        return 1
    deployed = resolve()
    if not deployed["FORECAST"]:
        print("SHADOW_LIVE_FAILED the deployed configuration has no opponent_forecast")
        return 1
    server = ServerConfiguration(
        f"ws://localhost:{port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )
    common: dict[str, Any] = dict(
        server_configuration=server,
        battle_format=format_map["mc"],
        log_level=40,
        max_concurrent_battles=1,
        open_timeout=None,
    )
    # the class-wide switches ladder_ourteam.py sets for the deployed launch: without
    # the guard stack a decision never reaches _guarded_action and nothing is audited
    from vgc_bench.src.guards import GUARDS, HARD_GUARDS

    PolicyPlayer.use_knowledge_obs = PolicyPlayer.use_moveset_prior = True
    PolicyPlayer.mask_immunities = True
    PolicyPlayer.use_knowledge_guards = True
    PolicyPlayer.guard_flags = {name: name in HARD_GUARDS for name in GUARDS}
    PolicyPlayer.use_search = False
    ours = PolicyPlayer(
        account_configuration=fresh_local_account(),
        deterministic=True,
        accept_open_team_sheet=True,
        team=RandomTeamBuilder(1, 1, "mc", [Path(deployed["TEAM"])]),
        preview_model_path=(
            Path(deployed["PREVIEW_MODEL"]) if deployed["PREVIEW_MODEL"] else None
        ),
        use_learned_teampreview=bool(deployed["PREVIEW_MODEL"]),
        switch_model_path=Path("data/opponent_switch_top500_regmc.pt"),
        move_model_path=Path("data/opponent_move_top500_regmc.pt"),
        use_opponent_reranker=True,
        use_tempo_reranker=True,
        sticky_guard_corrections=bool(deployed["STICKY"]),
        sheet_preview=bool(deployed["SHEET_PREVIEW"]),
        opponent_forecast_path=deployed["FORECAST"],
        guard_overrides={name: True for name in deployed["GUARDS"].split(",") if name},
        team_sheet_wait_timeout=3.0,
        decision_log_path=log,
        **common,
    )
    ours.set_policy(Path(deployed["CKPT"]), torch.device("cpu"))
    runtime = ours._opponent_forecaster
    if runtime is None or not runtime.serving:
        why = getattr(runtime, "load_failure", None)
        print(f"SHADOW_LIVE_FAILED the forecaster is not serving ({why})")
        return 1
    sheet_state: dict[str, str] = {}
    for wanted in ("open", "closed"):
        foe = StudyHeuristic(
            account_configuration=fresh_local_account(),
            team=RandomTeamBuilder(1, 1, "mc", [opponent_team]),
            accept_open_team_sheet=wanted == "open",
            **common,
        )
        before = set(ours.battles)
        asyncio.run(ours.battle_against(foe, n_battles=1))
        for tag in set(ours.battles) - before:
            sheet_state[tag] = wanted
    print("finished", ours.n_finished_battles, "won", ours.n_won_battles)

    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    problems: list[str] = []
    seen: Counter[str] = Counter()
    for row in rows:
        tag, turn = row.get("battle", ""), row.get("turn")
        if turn == 0:  # team preview
            if "opponent_forecast" in row:
                problems.append(f"{tag} preview record carries a forecast")
            continue
        orders = [o for c in row.get("candidates", []) for o in c.get("orders", [])]
        # a forced replacement after a faint: every order is a switch or a pass
        forced = bool(orders) and all(
            isinstance(o, dict) and o.get("kind") in ("switch", "pass") for o in orders
        )
        record = row.get("opponent_forecast")
        if record is None:
            problems.append(f"{tag} turn {turn}: no opponent_forecast")
            continue
        if "stand_down" in record:
            seen[
                f"stand_down:{'forced' if forced else 'move'}:{record['stand_down']}"
            ] += 1
            if not forced:
                problems.append(
                    f"{tag} turn {turn}: {record['stand_down']} on a move decision"
                )
            continue
        seen[f"forecast:{sheet_state.get(tag, '?')}"] += 1
        if record.get("turn") != turn:
            problems.append(
                f"{tag} turn {turn}: forecast is for turn {record.get('turn')}"
            )
        want = {"open": ["open", "open"], "closed": ["closed", "closed"]}.get(
            sheet_state.get(tag, "")
        )
        if want is not None and list(record.get("sheets", [])) != want:
            problems.append(
                f"{tag} turn {turn}: sheets {record.get('sheets')}, want {want}"
            )
        if not any(slot for slot in record.get("slots", [])):
            problems.append(f"{tag} turn {turn}: forecast has no slot")
    counts = {
        name: count
        for name, count in PolicyPlayer.guard_fire_counts.items()
        if name.startswith("opponent_forecast")
    }
    print("decision records:", len(rows), dict(seen))
    print("counters:", counts)
    print("runtime:", runtime.diagnostics())
    if any(
        name.startswith(("opponent_forecast_failed", "opponent_forecast_unloaded"))
        for name in counts
    ):
        problems.append(f"shadow-mode failures were counted: {counts}")
    for wanted in ("open", "closed"):
        if not seen[f"forecast:{wanted}"]:
            problems.append(f"no forecast was logged in the {wanted}-sheet game")
    if ours.n_finished_battles != 2:
        problems.append(f"{ours.n_finished_battles} of 2 games finished")
    if runtime.n_battles:
        problems.append(f"{runtime.n_battles} battle(s) not forgotten at the end")
    for problem in problems[:20]:
        print("PROBLEM", problem)
    print("SHADOW_LIVE_FAILED" if problems else "SHADOW_LIVE_PASSED")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
