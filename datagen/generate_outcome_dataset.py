"""Generate terminal-outcome states for a calibrated exact-search value model.

Four workers run independent Showdown simulators.  Every output file is one completed
game, making generation resumable without sharing mutable simulator or model state.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import fcntl
import json
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from vgc_bench.src.exact_observation import ExactPolicyAdapter, OpponentModelPrior
from vgc_bench.src.exact_planner import ExactNode, HybridEvaluator
from vgc_bench.src.exact_sim import ExactShowdownBridge
from vgc_bench.src.opponent_preview import PreviewPredictor
from vgc_bench.src.opponent_tactics import MovePredictor, SwitchPredictor
from vgc_bench.src.policy_player import PolicyPlayer

FORMAT = "gen9championsvgc2026regmb"


def _seed(rng: random.Random) -> list[int]:
    return [rng.randrange(1, 65536) for _ in range(4)]


def _game_rng(seed: int, game: int) -> random.Random:
    return random.Random(f"outcome:{seed}:{game}")


def _opponent_for_game(opponents: list[Path], seed: int, game: int) -> Path:
    cycle, position = divmod(game, len(opponents))
    ordered = list(opponents)
    random.Random(f"outcome-opponents:{seed}:{cycle}").shuffle(ordered)
    return ordered[position]


@contextmanager
def _manifest_lock(output: Path):
    with (output / ".outcome.lock").open("a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _initialise_manifest(output: Path, config: dict) -> None:
    path = output / "outcome_manifest.json"
    with _manifest_lock(output):
        if path.exists():
            existing = json.loads(path.read_text())
            if existing.get("config") != config:
                raise SystemExit(
                    f"{path} belongs to another configuration; choose a new output"
                )
        else:
            _write_json_atomic(path, {"config": config, "completed": {}, "failed": {}})


def _record(output: Path, game: int, summary: dict, *, failed: bool = False) -> None:
    path = output / "outcome_manifest.json"
    with _manifest_lock(output):
        manifest = json.loads(path.read_text())
        target = manifest["failed" if failed else "completed"]
        other = manifest["completed" if failed else "failed"]
        target[str(game)] = summary
        other.pop(str(game), None)
        _write_json_atomic(path, manifest)


def raise_if_generation_unusable(
    completed: int,
    failures: int,
    games: int,
    minimum_games: int,
    max_failed_games: int | None = None,
) -> None:
    """Exit nonzero only when the dataset is short or failures exceed the budget.

    Isolated exact-sim failures (a few per thousand games) are noise; exiting
    on any of them silently killed a chained pipeline after a complete run.
    The default budget is 1% of the requested games.
    """
    budget = max_failed_games if max_failed_games is not None else max(1, games // 100)
    if failures > budget:
        raise SystemExit(
            f"outcome generation had {failures} failed games; the budget is "
            f"{budget} (--max-failed-games)"
        )
    if completed < min(minimum_games, games):
        raise SystemExit(
            f"only {completed} completed games; minimum is {minimum_games}"
        )
    if failures:
        print(
            f"WARNING: {failures} isolated game failures tolerated (budget {budget})",
            flush=True,
        )


def _run_workers(count: int) -> None:
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--worker-index",
                str(worker),
            ]
        )
        for worker in range(count)
    ]
    try:
        codes = [process.wait() for process in processes]
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
    failed = [index for index, code in enumerate(codes) if code]
    if failed:
        raise SystemExit(f"outcome workers failed: {failed}")


def _sample_ranked(
    adapter,
    bridge: ExactShowdownBridge,
    node: ExactNode,
    role: str,
    rng: random.Random,
    *,
    top_k: int = 4,
    exploration: float = 0.0,
) -> str:
    legal = bridge.choices(node.state, role)
    if not legal:
        raise RuntimeError(f"no legal {role} choices in {node.request_state}")
    ranked = adapter.rank(node.state, node.requests, role, legal)
    if not ranked:
        raise RuntimeError(f"no encodable {role} choices in {node.request_state}")
    prefix = ranked[: min(top_k, len(ranked))]
    if exploration > 0 and rng.random() < exploration:
        return rng.choice(prefix).choice
    return rng.choices(
        prefix, weights=[max(1e-8, item.probability) for item in prefix], k=1
    )[0].choice


def _uniform_choice(
    bridge: ExactShowdownBridge, node: ExactNode, role: str, rng: random.Random
) -> str:
    legal = bridge.choices(node.state, role)
    if not legal:
        raise RuntimeError(f"no legal {role} choices in {node.request_state}")
    return rng.choice(legal)


def _advance(bridge, node: ExactNode, p1: str, p2: str) -> ExactNode:
    return ExactNode.from_result(
        bridge.simulate(node.state, p1_choice=p1, p2_choice=p2)
    )


def _sample_rows(rows: list[dict], limit: int) -> list[dict]:
    if len(rows) <= limit:
        return rows
    indices = np.linspace(0, len(rows) - 1, num=limit, dtype=np.int64)
    return [rows[int(index)] for index in dict.fromkeys(indices.tolist())]


def _save_game(output: Path, game: int, rows: list[dict], target: float) -> Path:
    path = output / f"outcome_{game:05d}.npz"
    temporary = output / f".outcome_{game:05d}.tmp.npz"
    np.savez_compressed(
        temporary,
        observations=np.stack([row["observation"] for row in rows]),
        action_masks=np.stack([row["action_mask"] for row in rows]),
        targets=np.full(len(rows), target, dtype=np.float32),
        critic_values=np.asarray(
            [row["critic_value"] for row in rows], dtype=np.float32
        ),
        hybrid_values=np.asarray(
            [row["hybrid_value"] for row in rows], dtype=np.float32
        ),
        metadata=np.asarray(
            [json.dumps(row["metadata"], separators=(",", ":")) for row in rows]
        ),
    )
    temporary.replace(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results_repaired/champion.zip")
    parser.add_argument(
        "--opponent-checkpoint",
        nargs="+",
        default=["results_repaired/opponents/64opp_3932160_v4.zip"],
        help="frozen PPO opponents; several rotate per game (style 'historical')",
    )
    parser.add_argument(
        "--human-checkpoint",
        default="",
        help=(
            "behavior-cloned human policy for the 'human_bc' and 'human_prior' "
            "styles (bc_mix_A; the eval_only arm is refused)"
        ),
    )
    parser.add_argument(
        "--styles",
        default="model,model,historical,uniform",
        help=(
            "comma-separated per-game opponent style cycle; repetition = weight. "
            "Styles: model (champion-based prior), historical (frozen PPO), "
            "uniform (random legal), human_bc (BC policy sampled), human_prior "
            "(replay predictors over the BC base -- champion-free)"
        ),
    )
    parser.add_argument(
        "--our-team-pool-prob",
        type=float,
        default=0.0,
        help=(
            "per-game probability that OUR side also draws from the team pool, "
            "so the value net learns positions rather than one fixed team"
        ),
    )
    parser.add_argument("--our-team", default="teams/reg_mb/our_team.txt")
    parser.add_argument("--opponent-dir", default="teams/reg_mb")
    parser.add_argument(
        "--preview-model", default="data/opponent_preview_top500_regmb.pt"
    )
    parser.add_argument("--move-model", default="data/opponent_move_top500_regmb.pt")
    parser.add_argument(
        "--switch-model", default="data/opponent_switch_top500_regmb.pt"
    )
    parser.add_argument("--output", default="outcome_data_v1")
    parser.add_argument("--games", type=int, default=10_000)
    parser.add_argument("--minimum-games", type=int, default=5_000)
    parser.add_argument(
        "--max-failed-games",
        type=int,
        default=None,
        help=(
            "tolerate this many isolated game failures (default: 1%% of --games);"
            " beyond it the run exits nonzero. Previously ANY failure exited"
            " nonzero, which silently killed chained pipelines after a complete"
            " dataset (98 failures in 19,902 games, 2026-09-03)"
        ),
    )
    parser.add_argument("--max-seconds", type=float, default=5_400.0)
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--states-per-game", type=int, default=8)
    parser.add_argument("--hidden-sheet-prob", type=float, default=0.50)
    parser.add_argument("--exploration", type=float, default=0.10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="retry only failed/missing deterministic game IDs after a bug fix",
    )
    parser.add_argument(
        "--worker-index", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    if args.workers < 1 or args.games < 1 or args.states_per_game < 1:
        raise SystemExit("workers, games and states-per-game must be positive")
    if not 0 <= args.hidden_sheet_prob <= 1 or not 0 <= args.exploration <= 1:
        raise SystemExit("probabilities must be in [0, 1]")
    if args.worker_index is not None and not 0 <= args.worker_index < args.workers:
        raise SystemExit("--worker-index must be within --workers")
    styles = [style.strip() for style in args.styles.split(",") if style.strip()]
    known_styles = {"model", "historical", "uniform", "human_bc", "human_prior"}
    unknown = sorted(set(styles) - known_styles)
    if not styles or unknown:
        raise SystemExit(
            f"unknown styles {unknown}; choose from {sorted(known_styles)}"
        )
    needs_human = bool({"human_bc", "human_prior"} & set(styles))
    if needs_human and not args.human_checkpoint:
        raise SystemExit("human styles require --human-checkpoint")
    if not 0 <= args.our_team_pool_prob <= 1:
        raise SystemExit("--our-team-pool-prob must be in [0, 1]")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "opponent_checkpoints": [
            str(Path(path).resolve()) for path in args.opponent_checkpoint
        ],
        "human_checkpoint": (
            str(Path(args.human_checkpoint).resolve()) if args.human_checkpoint else ""
        ),
        "styles": styles,
        "our_team_pool_prob": args.our_team_pool_prob,
        "our_team": str(Path(args.our_team).resolve()),
        "opponent_dir": str(Path(args.opponent_dir).resolve()),
        "preview_model": str(Path(args.preview_model).resolve()),
        "move_model": str(Path(args.move_model).resolve()),
        "switch_model": str(Path(args.switch_model).resolve()),
        "games": args.games,
        "max_turns": args.max_turns,
        "states_per_game": args.states_per_game,
        "hidden_sheet_prob": args.hidden_sheet_prob,
        "exploration": args.exploration,
        "seed": args.seed,
    }
    _initialise_manifest(output, config)
    if args.workers > 1 and args.worker_index is None:
        _run_workers(args.workers)
        manifest = json.loads((output / "outcome_manifest.json").read_text())
        completed = len(manifest["completed"])
        states = sum(int(row["states"]) for row in manifest["completed"].values())
        failures = len(manifest["failed"])
        print(
            f"outcome generation complete: games={completed}/{args.games}; "
            f"states={states}; failures={failures}",
            flush=True,
        )
        raise_if_generation_unusable(
            completed, failures, args.games, args.minimum_games, args.max_failed_games
        )
        return

    manifest = json.loads((output / "outcome_manifest.json").read_text())
    completed = {int(game) for game in manifest["completed"]}
    failed = {int(game) for game in manifest["failed"]}
    opponents = sorted(Path(args.opponent_dir).glob("*.txt"))
    our_team = Path(args.our_team)
    opponents = [path for path in opponents if path.resolve() != our_team.resolve()]
    if not opponents:
        raise SystemExit("no opponent teams found")

    PolicyPlayer.use_knowledge_obs = True
    PolicyPlayer.use_moveset_prior = False
    champion = PPO.load(args.checkpoint, device=args.device).policy
    from vgc_bench.src.utils import refuse_eval_only_checkpoint

    for path in args.opponent_checkpoint:
        refuse_eval_only_checkpoint(path)
    preview = PreviewPredictor.load(Path(args.preview_model), device=args.device)
    move_predictor = MovePredictor.load(Path(args.move_model), device=args.device)
    switch_predictor = SwitchPredictor.load(Path(args.switch_model), device=args.device)
    acting = ExactPolicyAdapter(champion, preview, reveal_opponent_sets=True)
    concrete = ExactPolicyAdapter(champion, preview, reveal_opponent_sets=True)
    historical_adapters = [
        ExactPolicyAdapter(
            PPO.load(path, device=args.device).policy,
            preview,
            reveal_opponent_sets=True,
        )
        for path in args.opponent_checkpoint
    ]
    opponent_prior = OpponentModelPrior(
        acting, move_predictor=move_predictor, switch_predictor=switch_predictor
    )
    human_adapter = None
    human_prior = None
    if needs_human:
        refuse_eval_only_checkpoint(args.human_checkpoint)
        human_policy = PPO.load(args.human_checkpoint, device=args.device).policy
        human_adapter = ExactPolicyAdapter(
            human_policy, preview, reveal_opponent_sets=True
        )
        # champion-free human prior: replay-trained predictors reweight the BC
        # base instead of the champion's own ranking (the contamination the
        # replan diagnosed in the v1 dataset)
        human_prior = OpponentModelPrior(
            human_adapter,
            move_predictor=move_predictor,
            switch_predictor=switch_predictor,
        )
    hybrid = HybridEvaluator(concrete)
    started = time.monotonic()
    first = args.worker_index or 0
    stride = args.workers if args.worker_index is not None else 1

    with ExactShowdownBridge() as bridge:
        for game in range(first, args.games, stride):
            if game in completed or (game in failed and not args.retry_failed):
                continue
            if args.max_seconds > 0 and time.monotonic() - started >= args.max_seconds:
                break
            rng = _game_rng(args.seed, game)
            opponent = _opponent_for_game(opponents, args.seed, game)
            game_our_team = our_team
            if args.our_team_pool_prob > 0 and rng.random() < args.our_team_pool_prob:
                pooled = _opponent_for_game(opponents, args.seed + 777, game)
                if pooled.resolve() != opponent.resolve():
                    game_our_team = pooled
            hidden = rng.random() < args.hidden_sheet_prob
            style = styles[game % len(styles)]
            historical_adapter = historical_adapters[game % len(historical_adapters)]
            for adapter in (acting, historical_adapter, human_adapter):
                if adapter is not None:
                    adapter.reveal_opponent_sets = not hidden
            node = None
            try:
                node = ExactNode.from_result(
                    bridge.create(
                        formatid=FORMAT,
                        seed=_seed(rng),
                        p1_name="planner",
                        p2_name="opponent",
                        p1_team_text=game_our_team.read_text(),
                        p2_team_text=opponent.read_text(),
                    )
                )
                rows: list[dict] = []
                preview_failures = 0
                while not node.ended and node.turn <= args.max_turns:
                    # Team Preview states are labeled too: the v1 dataset skipped
                    # them, which left the value net predicting ~5% win at every
                    # real preview (ladder Brier 0.354 in that bucket) -- blind
                    # exactly where games are decided.
                    if node.request_state in ("move", "teampreview"):
                        try:
                            obs, mask = concrete.observation(
                                node.state, node.requests, "p1"
                            )
                            rows.append(
                                {
                                    "observation": obs.astype(np.float16),
                                    "action_mask": mask.astype(np.uint8),
                                    "critic_value": concrete.value(
                                        node.state, node.requests, "p1"
                                    ),
                                    "hybrid_value": hybrid(node, "p1"),
                                    "metadata": {
                                        "game": game,
                                        "turn": node.turn,
                                        "opponent": opponent.name,
                                        "our_team": game_our_team.name,
                                        "hidden_sheets": hidden,
                                        "opponent_style": style,
                                        "preview": node.request_state == "teampreview",
                                    },
                                }
                            )
                        except Exception:
                            if node.request_state != "teampreview":
                                raise
                            preview_failures += 1
                    p1 = _sample_ranked(
                        acting,
                        bridge,
                        node,
                        "p1",
                        rng,
                        exploration=(
                            args.exploration if node.request_state == "move" else 0
                        ),
                    )
                    if style == "uniform" and node.request_state == "move":
                        p2 = _uniform_choice(bridge, node, "p2", rng)
                    else:
                        p2_adapter = {
                            "historical": historical_adapter,
                            "human_bc": human_adapter,
                            "human_prior": human_prior,
                        }.get(style, opponent_prior)
                        assert p2_adapter is not None
                        p2 = _sample_ranked(p2_adapter, bridge, node, "p2", rng)
                    node = _advance(bridge, node, p1, p2)
                if not node.ended or not rows:
                    raise RuntimeError(
                        "battle did not finish with labeled states by turn "
                        f"{args.max_turns}"
                    )
                side = node.state["sides"][0]
                target = (
                    0.5
                    if not node.winner
                    else float(node.winner in {"p1", side.get("name")})
                )
                selected = _sample_rows(rows, args.states_per_game)
                path = _save_game(output, game, selected, target)
                summary = {
                    "opponent": opponent.name,
                    "our_team": game_our_team.name,
                    "hidden_sheets": hidden,
                    "opponent_style": style,
                    "states": len(selected),
                    "preview_states": sum(
                        1 for row in selected if row["metadata"].get("preview")
                    ),
                    "preview_failures": preview_failures,
                    "turn": node.turn,
                    "target": target,
                    "path": path.name,
                }
                _record(output, game, summary)
                print(
                    f"game {game + 1}/{args.games}: {opponent.name}; style={style}; "
                    f"hidden={hidden}; states={len(selected)}; winner={node.winner}",
                    flush=True,
                )
            except Exception as exc:
                summary = {
                    "opponent": opponent.name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "turn": node.turn if node else 0,
                }
                if node is not None:
                    _write_json_atomic(
                        output / f"failure_{game:05d}.json",
                        {
                            **summary,
                            "state": node.state,
                            "requests": node.requests,
                            "request_state": node.request_state,
                        },
                    )
                _record(output, game, summary, failed=True)
                print(f"game {game + 1} failed: {summary['error']}", flush=True)


if __name__ == "__main__":
    main()
