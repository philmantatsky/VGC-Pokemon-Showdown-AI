"""Ground local confidence and value estimates in our real ladder outcomes.

The standing sim-to-real instrument of the 2026-08 replan: nothing that predicts
winning may ship unless its predictions on OUR OWN ladder games beat the
incumbent's. Two independent passes:

    confidence   Join every decisions.jsonl record to its game's real outcome
                 (via tools/analyze_ladder_previews.py output) and measure how
                 the policy's chosen-action probability separates wins from
                 losses. The Aug-23 audit's qualitative finding -- "the policy
                 is exactly as confident when losing as when winning" -- becomes
                 a tracked number (AUC) here.

    value        Replay stored ladder games through the deployed terminal-
                 outcome value net and report Brier/ECE against real results,
                 for comparison with its local test Brier of 0.1035.
                 (Implemented in a later stage; this file starts with the
                 zero-machinery confidence pass so the instrument exists now.)

Run `tools/analyze_ladder_previews.py ladder` first; it writes the per-game
outcome table this script joins against. Rerun both after every ladder batch.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results_analysis" / "ladder_preview_analysis.json"


def auc(positives: list[float], negatives: list[float]) -> float | None:
    """Mann-Whitney AUC: P(random win-confidence > random loss-confidence)."""
    if not positives or not negatives:
        return None
    wins = 0.0
    for p in positives:
        for n in negatives:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def summarize(name: str, values: list[float]) -> str:
    if not values:
        return f"{name}: none"
    ordered = sorted(values)
    mid = ordered[len(ordered) // 2]
    mean = sum(ordered) / len(ordered)
    return f"{name}: n={len(ordered)} mean={mean:.3f} median={mid:.3f}"


def confidence_pass(_args: argparse.Namespace) -> None:
    assert ANALYSIS.exists(), (
        f"{ANALYSIS} missing -- run tools/analyze_ladder_previews.py ladder first"
    )
    games = json.loads(ANALYSIS.read_text())["games"]
    outcome = {row["battle"]: row["won"] for row in games}
    batch_of = {row["battle"]: row["batch"] for row in games}

    # every decisions.jsonl beside the replays
    preview_conf: dict[str, list[float]] = defaultdict(list)
    move_conf: dict[str, list[float]] = defaultdict(list)
    matched_battles: set[str] = set()
    decision_paths = sorted(ROOT.glob("ladder_replays*/**/decisions.jsonl")) + sorted(
        ROOT.glob("ladder_replays*/decisions.jsonl")
    )
    for decision_path in decision_paths:
        with decision_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                tag = row.get("battle", "")
                if tag not in outcome:
                    continue
                chosen = row.get("chosen") or {}
                probability = chosen.get("policy_probability")
                if probability is None:
                    continue
                matched_battles.add(tag)
                bucket = "win" if outcome[tag] else "loss"
                if row.get("turn") == 0:
                    preview_conf[bucket].append(float(probability))
                else:
                    move_conf[bucket].append(float(probability))

    print(f"joined decision records from {len(matched_battles)} battles\n")
    print("--- Team Preview (turn-0) chosen-action policy probability ---")
    print(summarize("wins  ", preview_conf["win"]))
    print(summarize("losses", preview_conf["loss"]))
    preview_auc = auc(preview_conf["win"], preview_conf["loss"])
    print(
        f"AUC(preview confidence -> game outcome): {preview_auc:.3f}"
        if preview_auc is not None
        else "AUC: insufficient data"
    )

    print("\n--- in-battle chosen-action policy probability ---")
    print(summarize("wins  ", move_conf["win"]))
    print(summarize("losses", move_conf["loss"]))
    move_auc = auc(move_conf["win"], move_conf["loss"])
    print(
        f"AUC(move confidence -> game outcome): {move_auc:.3f}"
        if move_auc is not None
        else "AUC: insufficient data"
    )

    # per-battle mean preview confidence AUC (one number per game, cleaner unit)
    per_battle: dict[str, list[float]] = defaultdict(list)
    decision_paths = sorted(ROOT.glob("ladder_replays*/**/decisions.jsonl")) + sorted(
        ROOT.glob("ladder_replays*/decisions.jsonl")
    )
    for decision_path in decision_paths:
        with decision_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                tag = row.get("battle", "")
                chosen = row.get("chosen") or {}
                if (
                    tag in outcome
                    and row.get("turn") == 0
                    and chosen.get("policy_probability") is not None
                ):
                    per_battle[tag].append(float(chosen["policy_probability"]))
    game_wins = [
        sum(vals) / len(vals) for tag, vals in per_battle.items() if outcome[tag]
    ]
    game_losses = [
        sum(vals) / len(vals) for tag, vals in per_battle.items() if not outcome[tag]
    ]
    game_auc = auc(game_wins, game_losses)
    if game_auc is not None:
        print(
            f"\nAUC(mean preview confidence per game -> outcome): {game_auc:.3f} "
            f"({len(game_wins)}W/{len(game_losses)}L games)"
        )
    out_dir = ROOT / "results_analysis"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "confidence_calibration.json"
    out_path.write_text(
        json.dumps(
            {
                "battles_joined": len(matched_battles),
                "preview_auc": preview_auc,
                "move_auc": move_auc,
                "per_game_preview_auc": game_auc,
                "batches": sorted({batch_of[tag] for tag in matched_battles}),
            },
            indent=1,
        )
    )
    print(f"wrote {out_path}")


def value_pass(args: argparse.Namespace) -> None:
    """Replay stored ladder games through the deployed value net; Brier vs truth.

    Stored replays lack |showteam| and |request| lines, so our own movesets are
    unknown to a plain replay. We inject our fixed team as a synthetic showteam
    line (we know exactly what we played), then follow the log with the same
    LogReader used for trajectory conversion and score every recorded decision
    state with the deployed evaluator: champion-critic value head + calibrated
    temperature, fake_rating=2000, exactly as OutcomeValueEvaluator does at
    exact-search leaves.
    """
    import asyncio
    import math
    import re
    from threading import Thread

    import numpy as np
    import torch
    from poke_env.ps_client import AccountConfiguration
    from poke_env.teambuilder import ConstantTeambuilder

    from vgc_bench.logs2trajs import LogReader
    from vgc_bench.src.outcome_value import (
        OutcomeValueEvaluator,
        calibration_error,
        critic_logits,
    )
    from vgc_bench.src.policy_player import PolicyPlayer

    assert ANALYSIS.exists(), (
        f"{ANALYSIS} missing -- run tools/analyze_ladder_previews.py ladder first"
    )
    games = {row["battle"]: row for row in json.loads(ANALYSIS.read_text())["games"]}
    PolicyPlayer.use_moveset_prior = True  # deployed configuration
    evaluator = OutcomeValueEvaluator.load(
        Path(args.outcome_value), device=args.device
    )
    packed = ConstantTeambuilder(Path(args.team).read_text()).yield_team()

    loop = asyncio.new_event_loop()
    Thread(target=loop.run_forever, daemon=True).start()
    embed_log = re.compile(r'class="battle-log-data"[^>]*>(.*?)</script>', re.S)

    rows = []
    failures: dict[str, int] = defaultdict(int)
    replay_files = sorted(ROOT.glob("ladder_replays*/**/*.html")) + sorted(
        ROOT.glob("ladder_replays*/*.html")
    )
    if args.limit:
        replay_files = replay_files[: args.limit]
    for html in replay_files:
        username = html.name.split(" - ")[0]
        raw = html.read_text(encoding="utf-8", errors="replace")
        match = embed_log.search(raw)
        if match is None:
            failures["no_embedded_log"] += 1
            continue
        log = "\n".join(
            line for line in match.group(1).split("\n") if not line.startswith(">")
        ).strip()
        tag = re.sub(
            r"^(battle-[a-z0-9]+-\d+).*$", r"\1", html.stem.split(" - ")[-1]
        )
        info = games.get(tag)
        if info is None:
            failures["no_outcome"] += 1
            continue
        role_match = re.search(
            rf"\|player\|(p[12])\|{re.escape(username)}\|", log
        )
        if role_match is None:
            failures["no_role"] += 1
            continue
        role = role_match.group(1)
        # Inject our known team AFTER the |poke| preview lines: poke-env's
        # showteam handler matches against battle.teampreview_team, which those
        # lines populate; injected earlier it silently no-ops.
        if "|poke|" not in log:
            failures["no_poke_lines"] += 1
            continue
        poke_lines_end = log.index("\n", log.rindex("|poke|"))
        log = (
            log[:poke_lines_end]
            + f"\n|showteam|{role}|{packed}"
            + log[poke_lines_end:]
        )
        reader = LogReader(
            account_configuration=AccountConfiguration(username, None),
            battle_format=tag.split("-")[1],
            log_level=51,
            accept_open_team_sheet=True,
            loop=loop,
        )
        try:
            asyncio.run_coroutine_threadsafe(
                reader.follow_log(tag.removeprefix("battle-"), log), loop
            ).result(timeout=60)
        except Exception as exc:
            failures[f"replay:{type(exc).__name__}"] += 1
            continue
        states = reader.states
        if not states:
            failures["no_states"] += 1
            continue
        our_species = {
            row.split("|")[0].lower().replace("-", "").replace(" ", "")
            for row in packed.split("]")
        }
        got = {
            mon.base_species.lower().replace("-", "").replace(" ", "")
            for mon in states[-1].team.values()
        }
        if not got or len(got & {s[:12] for s in our_species}) == 0:
            failures["team_mismatch"] += 1
            continue
        revealed = [
            i for i, mon in enumerate(states[-1].team.values(), start=1) if mon.revealed
        ]
        with torch.no_grad():
            for state_index, state in enumerate(states):
                draft = revealed if state_index else []
                for j, mon in enumerate(state.team.values(), start=1):
                    mon._selected_in_teampreview = j in draft
                try:
                    obs = PolicyPlayer.embed_battle(state, fake_rating=2000)
                except Exception as exc:
                    failures[f"embed:{type(exc).__name__}"] += 1
                    continue
                obs_dict = {
                    "observation": torch.as_tensor(
                        obs, dtype=torch.float32, device=evaluator.policy.device
                    ).unsqueeze(0),
                    "action_mask": torch.ones(
                        (1, 214),
                        dtype=torch.float32,
                        device=evaluator.policy.device,
                    ),
                }
                logit = float(critic_logits(evaluator.policy, obs_dict).item())
                probability = 1.0 / (
                    1.0 + math.exp(-logit / evaluator.temperature)
                )
                rows.append(
                    {
                        "battle": tag,
                        "turn": int(state.turn),
                        "p_win": probability,
                        "won": bool(info["won"]),
                        "opp_rating": info.get("opp_rating"),
                        "no_faints": info.get("first_faint") is None,
                    }
                )

    probabilities = np.array([row["p_win"] for row in rows])
    targets = np.array([1.0 if row["won"] else 0.0 for row in rows])
    battles_scored = len({row["battle"] for row in rows})

    def brier(p, t):
        return float(np.mean((p - t) ** 2)) if len(p) else None

    overall = brier(probabilities, targets)
    ece = float(calibration_error(probabilities, targets)) if len(rows) else None
    print(
        f"scored {len(rows)} states from {battles_scored} battles "
        f"(failures: {dict(failures)})"
    )
    print(f"mean predicted P(win): {float(probabilities.mean()):.3f}   "
          f"actual win rate: {float(targets.mean()):.3f}")
    print(f"ladder Brier: {overall:.4f}   (local test Brier was 0.1035)")
    print(f"ladder ECE  : {ece:.4f}   (local test ECE was 0.0222)")
    print("\n--- by turn bucket ---")
    for label, low, high in (
        ("preview/turn0", -1, 0),
        ("turns 1-3", 1, 3),
        ("turns 4-6", 4, 6),
        ("turns 7+", 7, 99),
    ):
        subset = [
            i for i, row in enumerate(rows) if low <= row["turn"] <= high
        ]
        if subset:
            print(
                f"{label:14} n={len(subset):5d} "
                f"Brier={brier(probabilities[subset], targets[subset]):.4f} "
                f"mean p={float(probabilities[subset].mean()):.3f} "
                f"win rate={float(targets[subset].mean()):.3f}"
            )
    out_dir = ROOT / "results_analysis"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "value_ladder_calibration.json"
    out_path.write_text(
        json.dumps(
            {
                "outcome_value": str(args.outcome_value),
                "states": len(rows),
                "battles": battles_scored,
                "brier": overall,
                "ece": ece,
                "mean_p": float(probabilities.mean()) if len(rows) else None,
                "base_rate": float(targets.mean()) if len(rows) else None,
                "failures": dict(failures),
            },
            indent=1,
        )
    )
    print(f"\nwrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("confidence", help="policy-confidence vs real outcomes")
    value = sub.add_parser("value", help="value-net Brier on real ladder outcomes")
    value.add_argument(
        "--outcome-value", default="results_outcome_v1/outcome_value.zip"
    )
    value.add_argument("--team", default="teams/reg_mb/our_team.txt")
    value.add_argument("--device", default="cpu")
    value.add_argument("--limit", type=int, default=0, help="debug: first N replays")
    args = ap.parse_args()
    if args.command == "confidence":
        confidence_pass(args)
    elif args.command == "value":
        value_pass(args)


if __name__ == "__main__":
    main()
