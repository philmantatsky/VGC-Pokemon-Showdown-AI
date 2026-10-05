"""Calibrate the exact search's critic leaf against real results (2026-10-04).

The search scores a position with the brain's critic plus its shaping potential
(vgc_bench/src/critic_leaf.py). Against an equal opponent that number is optimistic
and unevenly scaled, and early in a game it says nothing about who wins. This turns it
into a win probability:

    collect   the deployed bot plays itself, no search; at each of side A's move
              decisions the raw leaf value is logged with the battle and the turn,
              then joined to who won (<output>/samples.jsonl).
    fit       one monotone (isotonic) map per phase of the game from raw value to
              side A's win rate, written as the JSON ``LeafCalibration`` loads
              (<output>/calibration.json), with held-out Brier scores and AUC.

Usage (from the repo root; collect needs a local server on --port):
    .venv/bin/python evaluation/leaf_calibration.py collect --games 2000 \\
        --port 7614 --output results_leaf_calibration_T6ep
    .venv/bin/python evaluation/leaf_calibration.py fit results_leaf_calibration_T6ep
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from typing import Any, Sequence  # noqa: E402

Sample = tuple[float, float]  # (raw leaf value, 1 won / 0 lost / 0.5 tied)


def isotonic(points: Sequence[tuple[float, float, float]]) -> list[list[float]]:
    """Pool-adjacent-violators over ``(x, y, weight)`` sorted by x: blocks
    ``[mean x, mean y, weight]`` whose y strictly increases (equal neighbours are
    pooled too, so a flat stretch is one knot, not twenty)."""
    blocks: list[list[float]] = []
    for x, y, weight in points:
        blocks.append([x, y, weight])
        while len(blocks) > 1 and blocks[-2][1] >= blocks[-1][1]:
            x1, y1, w1 = blocks.pop()
            x0, y0, w0 = blocks.pop()
            total = w0 + w1
            blocks.append(
                [(x0 * w0 + x1 * w1) / total, (y0 * w0 + y1 * w1) / total, total]
            )
    return blocks


def fit_knots(samples: Sequence[Sample], bins: int = 20) -> list[list[float]]:
    """Monotone ``[raw, win probability]`` knots: equal-count bins, then isotonic.

    Binning first keeps a handful of lucky games at one raw value from carving a
    step into the map.
    """
    ordered = sorted(samples)
    if not ordered:
        raise ValueError("no samples to calibrate on")
    size = max(1, len(ordered) // max(1, bins))
    points = []
    for start in range(0, len(ordered), size):
        chunk = ordered[start : start + size]
        points.append(
            (
                sum(raw for raw, _won in chunk) / len(chunk),
                sum(won for _raw, won in chunk) / len(chunk),
                float(len(chunk)),
            )
        )
    knots: list[list[float]] = []
    for x, y, _weight in isotonic(points):
        if knots and x <= knots[-1][0]:
            continue
        # never certainty: a won or lost game is the planner's own +1 / -1, and a
        # bin of twenty lucky positions is not one
        knots.append([round(x, 4), round(min(0.98, max(0.02, y)), 4)])
    return knots


def phase_of(turn: int, bounds: Sequence[int]) -> int:
    return next((i for i, bound in enumerate(bounds) if turn <= bound), len(bounds))


def auc(samples: Sequence[Sample]) -> float | None:
    """Probability that a won position has the higher raw value (ties excluded)."""
    ranked = sorted(samples)
    wins = sum(1 for _raw, won in ranked if won == 1.0)
    losses = sum(1 for _raw, won in ranked if won == 0.0)
    if not wins or not losses:
        return None
    below = 0.0  # losses seen so far, counting equal raw values as half
    total = 0.0
    index = 0
    while index < len(ranked):
        end = index
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        tied_losses = sum(1 for _raw, won in ranked[index:end] if won == 0.0)
        tied_wins = sum(1 for _raw, won in ranked[index:end] if won == 1.0)
        total += tied_wins * (below + 0.5 * tied_losses)
        below += tied_losses
        index = end
    return total / (wins * losses)


def _interpolate(knots: Sequence[Sequence[float]], raw: float) -> float:
    from vgc_bench.src.critic_leaf import LeafCalibration

    return LeafCalibration(
        ((None, tuple((k[0], k[1]) for k in knots)),)
    ).win_probability(raw, 0)


def fit(rows: Sequence[dict[str, Any]], bounds: Sequence[int], folds: int = 5) -> dict:
    """Per-phase knots from every row, and held-out scores by battle."""
    battles = sorted({row["battle"] for row in rows})
    rng = random.Random(0)
    rng.shuffle(battles)
    fold_of = {battle: i % folds for i, battle in enumerate(battles)}
    phases = []
    for index in range(len(bounds) + 1):
        mine = [row for row in rows if phase_of(int(row["turn"]), bounds) == index]
        samples = [(float(row["raw"]), float(row["won"])) for row in mine]
        knots = fit_knots(samples) if samples else [[0.0, 0.5]]
        raw_brier = held_out = 0.0
        for fold in range(folds):
            train = [
                (float(r["raw"]), float(r["won"]))
                for r in mine
                if fold_of[r["battle"]] != fold
            ]
            test = [r for r in mine if fold_of[r["battle"]] == fold]
            fold_knots = fit_knots(train) if train else [[0.0, 0.5]]
            for row in test:
                won = float(row["won"])
                clipped = (max(-1.0, min(1.0, float(row["raw"]))) + 1.0) / 2.0
                raw_brier += (clipped - won) ** 2
                held_out += (_interpolate(fold_knots, float(row["raw"])) - won) ** 2
        count = max(1, len(mine))
        phases.append(
            {
                "through_turn": bounds[index] if index < len(bounds) else None,
                "knots": knots,
                "samples": len(mine),
                "a_win_rate": sum(won for _raw, won in samples) / count,
                "auc": auc(samples),
                "brier_raw_clipped": raw_brier / count,
                "brier_calibrated_held_out": held_out / count,
            }
        )
    return {"phases": phases, "samples": len(rows), "battles": len(battles)}


def _collect(args: argparse.Namespace) -> None:
    import os

    import torch

    from evaluation.mirror_guard_ab import BLOCKS, _NoCache, _player, block_sizes
    from tools.deployed_config import resolve
    from vgc_bench.src import pokeenv_patches
    from vgc_bench.src.critic_leaf import raw_leaf_value
    from vgc_bench.src.guards import GUARDS, HARD_GUARDS
    from vgc_bench.src.policy_player import PolicyPlayer

    config = resolve()
    deployed = [g for g in config["GUARDS"].split(",") if g]
    os.environ.pop("VGC_SET_PRIOR_REG", None)
    if config["SET_PRIOR"] != "mc":
        os.environ["VGC_SET_PRIOR_REG"] = config["SET_PRIOR"]
    pokeenv_patches.install()
    torch.set_num_threads(1)
    PolicyPlayer.use_knowledge_obs = PolicyPlayer.use_moveset_prior = True
    PolicyPlayer.use_knowledge_guards = PolicyPlayer.mask_immunities = True
    PolicyPlayer.guard_flags = {g: g in HARD_GUARDS or g in deployed for g in GUARDS}
    PolicyPlayer._knowledge_cache = _NoCache()  # type: ignore[assignment]
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for (hidden, a_first), games in zip(BLOCKS, block_sizes(args.games)):
        if games == 0:
            continue
        PolicyPlayer.guard_fire_counts.clear()
        a = _player(
            config, hidden, args.seed, args.port, {}, concurrency=args.concurrency
        )
        b = _player(
            config, hidden, args.seed + 1, args.port, {}, concurrency=args.concurrency
        )
        seen: list[tuple[str, int, float]] = []
        inner = a._guarded_action

        def tapped(battle, obs_dict, mask, inner=inner, a=a, seen=seen):
            try:
                if not battle.teampreview and not any(battle.force_switch):
                    with a._exact_policy_lock:
                        raw = raw_leaf_value(a.policy, obs_dict, battle)
                    seen.append((battle.battle_tag, int(battle.turn), raw))
            except Exception:
                PolicyPlayer.guard_fire_counts["leaf_tap_error"] += 1
            return inner(battle, obs_dict, mask)

        a._guarded_action = tapped  # type: ignore[method-assign]
        first, second = (a, b) if a_first else (b, a)
        asyncio.run(
            asyncio.wait_for(
                first.battle_against(second, n_battles=games), args.timeout
            )
        )
        results = {
            tag: (0.5 if battle.won is None else float(bool(battle.won)), battle.turn)
            for tag, battle in a.battles.items()
            if battle.finished
        }
        for tag, turn, raw in seen:
            if tag in results:
                won, turns = results[tag]
                rows.append(
                    {
                        "battle": tag,
                        "turn": turn,
                        "raw": raw,
                        "won": won,
                        "turns": int(turns),
                        "hidden": hidden,
                        "a_first": a_first,
                    }
                )
        wins = sum(won for won, _turns in results.values())
        print(
            f"block hidden={hidden} a_first={a_first}: {len(results)} games, "
            f"A {wins:.1f}; {len(rows)} samples so far; "
            f"tap errors {PolicyPlayer.guard_fire_counts.get('leaf_tap_error', 0)}",
            flush=True,
        )
        (output / "samples.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    manifest = {
        "games": args.games,
        "seed": args.seed,
        "concurrency": args.concurrency,
        "checkpoint": config["CKPT"],
        "team": config["TEAM"],
        "guards": deployed,
        "samples": len(rows),
        "elapsed_s": round(time.monotonic() - started, 1),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


def _fit(args: argparse.Namespace) -> None:
    run: Path = args.run
    rows = [
        json.loads(line)
        for line in (run / "samples.jsonl").read_text().splitlines()
        if line.strip()
    ]
    bounds = [int(part) for part in args.phases.split(",") if part.strip()]
    summary = fit(rows, bounds)
    manifest_path = run / "manifest.json"
    summary["source"] = str(run)
    summary["phase_bounds"] = bounds
    if manifest_path.exists():
        summary["collected_from"] = json.loads(manifest_path.read_text())
    target = args.output or run / "calibration.json"
    target.write_text(json.dumps(summary, indent=2) + "\n")
    for phase in summary["phases"]:
        through = phase["through_turn"]
        label = f"turns <= {through}" if through is not None else "later turns"
        print(
            f"{label}: n={phase['samples']} A wins {phase['a_win_rate']:.3f} "
            f"AUC {phase['auc'] if phase['auc'] is None else round(phase['auc'], 3)} "
            f"Brier raw {phase['brier_raw_clipped']:.4f} -> calibrated "
            f"{phase['brier_calibrated_held_out']:.4f} (held out); "
            f"knots {phase['knots']}"
        )
    print(f"wrote {target}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--games", type=int, default=2000)
    collect.add_argument("--seed", type=int, default=20941)
    collect.add_argument("--port", type=int, default=7614)
    collect.add_argument("--concurrency", type=int, default=8)
    collect.add_argument("--timeout", type=float, default=7200.0)
    collect.add_argument("--output", type=Path, required=True)
    fitter = sub.add_parser("fit")
    fitter.add_argument("run", type=Path)
    fitter.add_argument("--phases", default="3,6", help="last turn of each early phase")
    fitter.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    if args.command == "collect":
        _collect(args)
    else:
        _fit(args)


if __name__ == "__main__":
    main()
