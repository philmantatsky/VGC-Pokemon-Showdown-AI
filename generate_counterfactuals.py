"""Generate exact multi-turn counterfactual labels for policy distillation.

This is deliberately offline. Each candidate future is resolved by the bundled
Pokemon Showdown engine, then the resulting action rankings are stored beside the
policy observation. No ladder credentials or live battles are involved.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import random
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from vgc_bench.src.exact_observation import ExactPolicyAdapter, OpponentModelPrior
from vgc_bench.src.exact_planner import (
    DeterminizationBudgetExhausted,
    ExactDeterminizationPlanner,
    ExactNode,
    HybridEvaluator,
    PlannerConfig,
    WeightedExactNode,
)
from vgc_bench.src.exact_sim import ExactShowdownBridge
from vgc_bench.src.opponent_preview import PreviewPredictor
from vgc_bench.src.opponent_tactics import MovePredictor, SwitchPredictor
from vgc_bench.src.policy_player import PolicyPlayer
from vgc_bench.src.set_particles import (
    ParticleDatabase,
    TeamBelief,
    determination_team_text,
    team_roster,
)

FORMAT = "gen9championsvgc2026regmb"


class SplitBranchPrior:
    """Champion ranks our candidates; a separate prior ranks the opponent's.

    The first four counterfactual rounds ranked BOTH sides from the champion
    adapter (p2 blended 60/40 with the replay-trained predictors), so every
    label was computed against the champion's own idea of the opponent --
    self-confirming, and one of the two autopsied causes of their rejections.
    This wrapper lets the opponent's branches and rollout choices come from a
    human-grounded policy instead while our side's candidate generation stays
    on the champion being improved.
    """

    def __init__(self, ours, theirs):
        self.ours = ours
        self.theirs = theirs

    def rank(self, state, requests, role, choices):
        prior = self.ours if role == "p1" else self.theirs
        return prior.rank(state, requests, role, choices)


def _seed(rng: random.Random) -> list[int]:
    return [rng.randrange(1, 65536) for _ in range(4)]


def _ranked_choice(adapter, bridge, node: ExactNode, role: str) -> str:
    legal = bridge.choices(node.state, role)
    ranked = adapter.rank(node.state, node.requests, role, legal)
    if ranked:
        return ranked[0].choice
    if not legal:
        raise RuntimeError(f"no legal choice for {role} in {node.request_state}")
    return legal[0]


def _sampled_ranked_choice(
    prior,
    bridge,
    node: ExactNode,
    role: str,
    rng: random.Random,
    top_k: int = 4,
) -> str:
    """Sample a plausible response instead of replaying one deterministic opponent."""
    legal = bridge.choices(node.state, role)
    ranked = prior.rank(node.state, node.requests, role, legal)
    if not ranked:
        if not legal:
            raise RuntimeError(f"no legal choice for {role} in {node.request_state}")
        return legal[0]
    prefix = ranked[:top_k]
    weights = [max(1e-8, item.probability) for item in prefix]
    return rng.choices(prefix, weights=weights, k=1)[0].choice


def _trajectory_choice(result, rng: random.Random, exploration: float) -> str:
    """Occasionally follow a near-best line to collect recovery/off-policy states."""
    if exploration <= 0 or len(result.rankings) < 2 or rng.random() >= exploration:
        return result.choice
    prefix = result.rankings[: min(3, len(result.rankings))]
    best = prefix[0].score
    weights = [math.exp(max(-20.0, (item.score - best) / 0.12)) for item in prefix]
    return rng.choices(prefix, weights=weights, k=1)[0].choice


def _preview_plan_or_fallback(planner, roots, adapter, bridge, node):
    """Use the exact preview teacher when it finishes, otherwise the champion.

    Preview has hundreds of legal team orders. Exhausting its deliberately small
    offline budget is not an action-compatibility failure and must not discard an
    otherwise valid game. Other simulator and mapping errors still propagate to the
    generator's fail-fast handler.
    """
    try:
        return planner.plan(roots, "p1"), None
    except DeterminizationBudgetExhausted:
        return None, _ranked_choice(adapter, bridge, node, "p1")


def _advance(bridge, node: ExactNode, p1_choice: str, p2_choice: str) -> ExactNode:
    return ExactNode.from_result(
        bridge.simulate(node.state, p1_choice=p1_choice, p2_choice=p2_choice)
    )


def _move_example(
    adapter,
    node: ExactNode,
    result,
    game: int,
    opponent: Path,
    config: PlannerConfig,
):
    obs, mask = adapter.observation(node.state, node.requests, "p1")
    rankings = [ranking for ranking in result.rankings if ranking.actions is not None]
    return {
        "observation": obs.astype(np.float16),
        "action_mask": mask.astype(np.uint8),
        "actions": np.asarray(
            [ranking.actions for ranking in rankings], dtype=np.int16
        ),
        "scores": np.asarray([ranking.score for ranking in rankings], dtype=np.float32),
        "expected": np.asarray(
            [ranking.expected for ranking in rankings], dtype=np.float32
        ),
        "priors": np.asarray([ranking.prior for ranking in rankings], dtype=np.float32),
        "metadata": {
            "game": game,
            "turn": node.turn,
            "opponent": opponent.name,
            "best_choice": result.choice,
            "choices": [ranking.choice for ranking in rankings],
            "nodes": result.nodes,
            "elapsed_s": result.elapsed_s,
            "truncated": result.truncated,
            "complete_screen": result.screened_actions == len(result.rankings),
            "completed_depth": result.completed_depth,
            "screened_actions": result.screened_actions,
            "deepened_actions": result.deepened_actions,
            "fallback_reason": result.fallback_reason,
            "search_config": asdict(config),
            "hidden_sheets": not adapter.reveal_opponent_sets,
        },
    }


def _normalise_roots(roots: list[WeightedExactNode]) -> list[WeightedExactNode]:
    total = sum(max(0.0, root.probability) for root in roots)
    if total <= 0:
        raise ValueError("determinization roots have no probability mass")
    return [
        WeightedExactNode(root.node, root.probability / total, root.label)
        for root in roots
    ]


def _create_roots(
    bridge: ExactShowdownBridge,
    *,
    seed: list[int],
    our_team_text: str,
    opponent_team_text: str,
    reveal_opponent_sets: bool,
    particle_database: ParticleDatabase,
    rng: random.Random,
) -> list[WeightedExactNode]:
    """Create one open-sheet world or eight hidden-set concrete worlds."""
    if reveal_opponent_sets:
        opponent_teams = [(opponent_team_text, "open-sheet")]
    else:
        roster = team_roster(opponent_team_text)
        belief = TeamBelief.from_roster(
            particle_database, (slot.species for slot in roster)
        )
        sampled = belief.sample_determinizations(8, rng, open_sheet=False)
        opponent_teams = [
            (
                determination_team_text(roster, determination),
                "hidden-" + str(index + 1),
            )
            for index, determination in enumerate(sampled)
        ]
    mass = 1.0 / len(opponent_teams)
    roots = []
    for team_text, label in opponent_teams:
        created = bridge.create(
            formatid=FORMAT,
            seed=seed,
            p1_name="planner",
            p2_name="opponent",
            p1_team_text=our_team_text,
            p2_team_text=team_text,
        )
        roots.append(WeightedExactNode(ExactNode.from_result(created), mass, label))
    return roots


def _advance_roots(
    bridge: ExactShowdownBridge,
    roots: list[WeightedExactNode],
    p1_choice: str,
    p2_choice: str,
) -> list[WeightedExactNode]:
    """Advance all compatible particles; the first root is the sampled real world."""
    advanced: list[WeightedExactNode] = []
    for index, weighted in enumerate(roots):
        p1_legal = bridge.choices(weighted.node.state, "p1")
        p2_legal = bridge.choices(weighted.node.state, "p2")
        if p1_choice not in p1_legal or p2_choice not in p2_legal:
            if index == 0:
                raise RuntimeError("sampled real world rejected its own legal choice")
            continue
        child = _advance(bridge, weighted.node, p1_choice, p2_choice)
        advanced.append(WeightedExactNode(child, weighted.probability, weighted.label))
    if not advanced:
        raise RuntimeError("all hidden-set particles were eliminated")
    return _normalise_roots(advanced)


def _resolve_non_move_roots(
    bridge,
    adapter,
    opponent_prior,
    roots: list[WeightedExactNode],
    rng: random.Random,
) -> list[WeightedExactNode]:
    while (
        not roots[0].node.ended
        and roots[0].node.request_state
        and roots[0].node.request_state != "move"
    ):
        actual = roots[0].node
        p1 = _ranked_choice(adapter, bridge, actual, "p1")
        p2 = _sampled_ranked_choice(opponent_prior, bridge, actual, "p2", rng)
        roots = _advance_roots(bridge, roots, p1, p2)
    return roots


def _save_chunk(output: Path, index: int, examples: list[dict]) -> Path:
    width = max(len(example["scores"]) for example in examples)
    observations = np.stack([example["observation"] for example in examples])
    masks = np.stack([example["action_mask"] for example in examples])
    actions = np.full((len(examples), width, 2), -1, dtype=np.int16)
    scores = np.full((len(examples), width), np.nan, dtype=np.float32)
    expected = np.full_like(scores, np.nan)
    priors = np.zeros_like(scores)
    for row, example in enumerate(examples):
        count = len(example["scores"])
        if not count:
            # An empty candidate list is np shape (0,), which cannot broadcast
            # into the (0, 2) actions slice; the row's -1/NaN fill already says
            # "no candidates".
            continue
        actions[row, :count] = example["actions"]
        scores[row, :count] = example["scores"]
        expected[row, :count] = example["expected"]
        priors[row, :count] = example["priors"]
    path = output / f"moves_{index:05d}.npz"
    np.savez_compressed(
        path,
        observations=observations,
        action_masks=masks,
        candidate_actions=actions,
        planner_scores=scores,
        planner_expected=expected,
        policy_priors=priors,
        metadata=np.asarray(
            [
                json.dumps(example["metadata"], separators=(",", ":"))
                for example in examples
            ]
        ),
    )
    return path


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@contextmanager
def _generation_lock(output: Path):
    """Serialize the two shared files while independent simulators run in parallel."""
    lock_path = output / ".generation.lock"
    with lock_path.open("a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _initialise_manifest(
    output: Path, manifest_path: Path, generation_config: dict
) -> dict:
    # A resumed process may intentionally receive only the *remaining* wall-clock
    # allowance. That changes how long workers run, not the meaning of any label.
    # All search, model, team, RNG, and schema settings remain strict.
    semantic_config = {
        key: value
        for key, value in generation_config.items()
        if key != "max_seconds"
    }
    with _generation_lock(output):
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            existing_semantic_config = {
                key: value
                for key, value in manifest.get("config", {}).items()
                if key != "max_seconds"
            }
            if existing_semantic_config != semantic_config:
                raise SystemExit(
                    f"{manifest_path} belongs to a different generation config; "
                    "use another --output directory"
                )
        else:
            manifest = {"config": generation_config, "completed": {}}
            _write_json_atomic(manifest_path, manifest)
    return manifest


def _record_game(
    output: Path,
    manifest_path: Path,
    preview_path: Path,
    game: int,
    summary: dict,
    preview_record: dict | None,
) -> tuple[int, int]:
    """Atomically merge one worker's result into the shared progress checkpoint."""
    with _generation_lock(output):
        manifest = json.loads(manifest_path.read_text())
        key = str(game)
        previous = manifest["completed"].get(key)
        if previous is None or (
            "error" in previous and "error" not in summary
        ):
            if preview_record is not None:
                with preview_path.open("a") as preview_file:
                    preview_file.write(
                        json.dumps(preview_record, separators=(",", ":")) + "\n"
                    )
            manifest["completed"][key] = summary
            _write_json_atomic(manifest_path, manifest)
        positions = sum(
            int(record.get("positions", 0))
            for record in manifest["completed"].values()
        )
        return len(manifest["completed"]), positions


def _run_parallel_workers(worker_count: int) -> None:
    """Launch isolated simulators; children synchronize only checkpoints."""
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
        for worker in range(worker_count)
    ]
    try:
        remaining = set(range(worker_count))
        while remaining:
            for index in tuple(remaining):
                code = processes[index].poll()
                if code is None:
                    continue
                remaining.remove(index)
                if code:
                    raise RuntimeError(
                        f"counterfactual worker {index} failed with status {code}"
                    )
            if remaining:
                time.sleep(0.2)
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


def _game_rng(seed: int, game: int) -> random.Random:
    """Return a stable per-game RNG so skipped games do not change later games."""
    return random.Random(f"counterfactual:{seed}:{game}")


def _opponent_for_game(opponents: list[Path], seed: int, game: int) -> Path:
    """Cover every team once per cycle before repeating any matchup."""
    cycle, position = divmod(game, len(opponents))
    ordered = list(opponents)
    random.Random(f"counterfactual-opponents:{seed}:{cycle}").shuffle(ordered)
    return ordered[position]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results_repaired/champion.zip")
    parser.add_argument(
        "--rollout-residual",
        default="",
        help=(
            "latest residual used for exactly half of trajectory rollouts; the "
            "other half always uses the untouched champion"
        ),
    )
    parser.add_argument(
        "--outcome-value",
        default="",
        help="optional calibrated terminal-outcome checkpoint for planner leaves",
    )
    parser.add_argument(
        "--opponent-base-checkpoint",
        default="",
        help=(
            "optional policy (e.g. the human-imitation bc_mix_A) whose ranking "
            "drives the OPPONENT's search branches and rollout choices; without "
            "it the champion ranks both sides, which self-confirms its own play"
        ),
    )
    parser.add_argument("--our-team", default="teams/reg_mb/our_team.txt")
    parser.add_argument("--opponent-dir", default="teams/reg_mb")
    parser.add_argument(
        "--preview-model", default="data/opponent_preview_top500_regmb.pt"
    )
    parser.add_argument(
        "--move-model", default="data/opponent_move_top500_regmb.pt"
    )
    parser.add_argument(
        "--switch-model", default="data/opponent_switch_top500_regmb.pt"
    )
    parser.add_argument("--opponent-model-weight", type=float, default=0.60)
    parser.add_argument("--open-sheet-model-weight", type=float, default=0.40)
    parser.add_argument(
        "--trajectory-exploration",
        type=float,
        default=0.20,
        help="chance to follow a top-three alternative after labeling a state",
    )
    parser.add_argument("--output", default="counterfactual_data")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="stop starting games after this many seconds; zero means no time cap",
    )
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--root-width", type=int, default=8)
    parser.add_argument("--opponent-width", type=int, default=6)
    parser.add_argument("--continuation-width", type=int, default=3)
    parser.add_argument("--replacement-width", type=int, default=2)
    parser.add_argument("--chance-samples", type=int, default=4)
    parser.add_argument("--budget", type=float, default=9.0)
    parser.add_argument(
        "--anytime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="screen every root move family before deepening within the hard budget",
    )
    parser.add_argument(
        "--hidden-sheet-prob",
        type=float,
        default=0.50,
        help="fraction of games whose policy view hides unrevealed opponent sets",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument(
        "--max-failed-games",
        type=int,
        default=0,
        help=(
            "tolerate up to this many isolated failed games (recorded in the "
            "manifest with full tracebacks, retried on resume) before aborting; "
            "0 preserves the original abort-on-first-failure behavior"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="independent exact simulator processes",
    )
    parser.add_argument(
        "--worker-index", type=int, default=None, help=argparse.SUPPRESS
    )
    args = parser.parse_args()

    if not 0 <= args.hidden_sheet_prob <= 1:
        raise SystemExit("--hidden-sheet-prob must be between 0 and 1")
    if not 0 <= args.trajectory_exploration <= 1:
        raise SystemExit("--trajectory-exploration must be between 0 and 1")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.worker_index is not None and not 0 <= args.worker_index < args.workers:
        raise SystemExit("--worker-index must be within --workers")
    if args.workers > 1 and args.worker_index is None:
        started = time.monotonic()
        _run_parallel_workers(args.workers)
        manifest = json.loads(
            (Path(args.output) / "generation_manifest.json").read_text()
        )
        completed = manifest.get("completed", {})
        positions = sum(
            int(summary.get("positions", 0)) for summary in completed.values()
        )
        failed = sum("error" in summary for summary in completed.values())
        successful = len(completed) - failed
        remaining = args.games - successful
        print(
            f"parallel complete: {positions} move positions; "
            f"games={successful}/{args.games}; failed={failed}; "
            f"remaining={remaining}; elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )
        if failed > args.max_failed_games:
            raise SystemExit(
                f"generation produced {failed} failed games; exact legal-action "
                f"coverage is mandatory beyond --max-failed-games="
                f"{args.max_failed_games}"
            )
        if failed:
            print(
                f"WARNING: {failed} isolated game failures tolerated; see "
                "failure_*.json for tracebacks (failed games retry on resume)",
                flush=True,
            )
        if remaining and args.max_seconds <= 0:
            raise SystemExit(1)
        return

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    our_team = Path(args.our_team)
    opponents = sorted(Path(args.opponent_dir).glob("*.txt"))
    opponents = [path for path in opponents if path.resolve() != our_team.resolve()]
    if not opponents:
        raise SystemExit("no opponent teams found")

    PolicyPlayer.use_knowledge_obs = True
    PolicyPlayer.use_moveset_prior = False
    policy = PPO.load(args.checkpoint, device=args.device).policy
    preview = PreviewPredictor.load(Path(args.preview_model), device=args.device)
    champion_adapter = ExactPolicyAdapter(policy, preview)
    candidate_adapter = None
    if args.rollout_residual:
        from vgc_bench.src.residual_ranker import ResidualJointRanker

        residual = ResidualJointRanker.load(
            Path(args.rollout_residual), device=args.device
        )
        candidate_adapter = ExactPolicyAdapter(
            policy, preview, residual_ranker=residual
        )
    opponent_base_adapter = None
    if args.opponent_base_checkpoint:
        from vgc_bench.src.utils import refuse_eval_only_checkpoint

        opponent_base_path = Path(args.opponent_base_checkpoint)
        # The eval-only human holdout (bc_eval_B) must never leak into training
        # data; only train-eligible checkpoints may drive opponent branches.
        refuse_eval_only_checkpoint(opponent_base_path)
        opponent_base_policy = PPO.load(opponent_base_path, device=args.device).policy
        opponent_base_adapter = ExactPolicyAdapter(opponent_base_policy, preview)
    move_predictor = MovePredictor.load(Path(args.move_model), device=args.device)
    switch_predictor = SwitchPredictor.load(
        Path(args.switch_model), device=args.device
    )
    particle_database = ParticleDatabase.load(max_particles=12)
    config = PlannerConfig(
        depth=args.depth,
        root_width=args.root_width,
        opponent_width=args.opponent_width,
        continuation_width=args.continuation_width,
        replacement_width=args.replacement_width,
        chance_samples=args.chance_samples,
        time_budget_s=args.budget,
        anytime=args.anytime,
    )
    # Preview has 360 legal orders per side and occurs before any public move
    # evidence. Spending four RNG samples and two complete move turns on each order
    # can consume the whole game budget before one preview candidate finishes. A
    # dedicated teacher still looks through the selected leads into the first full
    # move turn, but uses common deterministic chance and depth one so every saved
    # preview label is an actual completed exact result.
    preview_config = replace(
        config,
        depth=1,
        root_width=min(8, config.root_width),
        opponent_width=min(6, config.opponent_width),
        continuation_width=min(3, config.continuation_width),
        chance_samples=1,
        anytime=False,
    )
    if args.outcome_value:
        from vgc_bench.src.outcome_value import OutcomeValueEvaluator

        evaluator = OutcomeValueEvaluator.load(
            Path(args.outcome_value), device=args.device, mechanics_weight=0.10
        )
    else:
        evaluator = HybridEvaluator(champion_adapter)
    manifest_path = output / "generation_manifest.json"
    generation_config = {
        "schema": 2,
        "preview_search": "depth1-shared-rng",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "rollout_residual": (
            str(Path(args.rollout_residual).resolve())
            if args.rollout_residual
            else None
        ),
        "outcome_value": (
            str(Path(args.outcome_value).resolve()) if args.outcome_value else None
        ),
        "our_team": str(our_team.resolve()),
        "opponent_dir": str(Path(args.opponent_dir).resolve()),
        "preview_model": str(Path(args.preview_model).resolve()),
        "move_model": str(Path(args.move_model).resolve()),
        "switch_model": str(Path(args.switch_model).resolve()),
        "opponent_model_weight": args.opponent_model_weight,
        "open_sheet_model_weight": args.open_sheet_model_weight,
        "trajectory_exploration": args.trajectory_exploration,
        "max_turns": args.max_turns,
        "max_seconds": args.max_seconds,
        "depth": args.depth,
        "root_width": args.root_width,
        "opponent_width": args.opponent_width,
        "continuation_width": args.continuation_width,
        "replacement_width": args.replacement_width,
        "chance_samples": args.chance_samples,
        "budget": args.budget,
        "anytime": args.anytime,
        "hidden_sheet_prob": args.hidden_sheet_prob,
        "seed": args.seed,
    }
    manifest = _initialise_manifest(output, manifest_path, generation_config)
    completed = {
        int(game)
        for game, summary in manifest.get("completed", {}).items()
        if "error" not in summary
    }
    preview_path = output / "preview_rankings.jsonl"
    started = time.monotonic()

    with ExactShowdownBridge() as bridge:
        first_game = args.worker_index or 0
        stride = args.workers if args.worker_index is not None else 1
        for game in range(first_game, args.games, stride):
            if args.max_seconds > 0 and time.monotonic() - started >= args.max_seconds:
                print(
                    f"worker {args.worker_index}: generation time cap reached",
                    flush=True,
                )
                break
            if game in completed:
                continue
            rng = _game_rng(args.seed, game)
            opponent = _opponent_for_game(opponents, args.seed, game)
            use_candidate = candidate_adapter is not None and game % 2 == 1
            adapter = candidate_adapter if use_candidate else champion_adapter
            assert adapter is not None
            adapter.reveal_opponent_sets = rng.random() >= args.hidden_sheet_prob
            branch_base = opponent_base_adapter or adapter
            branch_base.reveal_opponent_sets = adapter.reveal_opponent_sets
            opponent_prior = OpponentModelPrior(
                branch_base,
                move_predictor=move_predictor,
                switch_predictor=switch_predictor,
                model_weight=args.opponent_model_weight,
                open_sheet_model_weight=args.open_sheet_model_weight,
            )
            if opponent_base_adapter is not None:
                # p1 candidates stay champion-ranked; only p2 goes through the
                # human-grounded blend.
                opponent_prior = SplitBranchPrior(adapter, opponent_prior)
            game_examples: list[dict] = []
            preview_record = None
            preview_fallback_reason = None
            node = None
            try:
                roots = _create_roots(
                    bridge,
                    seed=_seed(rng),
                    our_team_text=our_team.read_text(),
                    opponent_team_text=opponent.read_text(),
                    reveal_opponent_sets=adapter.reveal_opponent_sets,
                    particle_database=particle_database,
                    rng=rng,
                )
                node = roots[0].node
                determinization_planner = ExactDeterminizationPlanner(
                    bridge, opponent_prior, evaluator, config
                )
                preview_planner = ExactDeterminizationPlanner(
                    bridge, opponent_prior, evaluator, preview_config
                )
                preview_result, preview_fallback = _preview_plan_or_fallback(
                    preview_planner, roots, adapter, bridge, node
                )
                if preview_result is not None:
                    preview_record = {
                        "game": game,
                        "opponent": opponent.name,
                        "hidden_sheets": not adapter.reveal_opponent_sets,
                        "rollout_policy": (
                            "latest_candidate" if use_candidate else "champion"
                        ),
                        "our_roster": adapter._roster(node.state, "p1"),
                        "opponent_roster": adapter._roster(node.state, "p2"),
                        "best_choice": preview_result.choice,
                        "rankings": [
                            asdict(item) for item in preview_result.rankings
                        ],
                        "search_config": asdict(preview_config),
                        "nodes": preview_result.nodes,
                        "elapsed_s": preview_result.elapsed_s,
                        "truncated": preview_result.truncated,
                    }
                    p1_preview = _trajectory_choice(
                        preview_result, rng, args.trajectory_exploration
                    )
                else:
                    p1_preview = preview_fallback
                    preview_fallback_reason = "preview_budget_exhausted"
                    print(
                        f"game {game + 1}/{args.games}: preview budget exhausted; "
                        "using champion preview",
                        flush=True,
                    )
                p2_preview = _sampled_ranked_choice(
                    opponent_prior, bridge, node, "p2", rng, top_k=8
                )
                roots = _advance_roots(bridge, roots, p1_preview, p2_preview)
                node = roots[0].node

                while not node.ended and node.turn <= args.max_turns:
                    roots = _resolve_non_move_roots(
                        bridge, adapter, opponent_prior, roots, rng
                    )
                    node = roots[0].node
                    if (
                        node.ended
                        or node.request_state != "move"
                        or node.turn > args.max_turns
                    ):
                        break
                    result = determinization_planner.plan(roots, "p1")
                    example = _move_example(
                        adapter, node, result, game, opponent, config
                    )
                    if len(example["scores"]):
                        game_examples.append(example)
                    else:
                        # Every ranked choice failed policy action-encoding
                        # (rare forced positions). A zero-candidate example
                        # cannot train a ranker and crashes the NPZ writer.
                        print(
                            f"game {game + 1}/{args.games}: skipped a position "
                            f"with no encodable candidates (turn {node.turn})",
                            flush=True,
                        )
                    p2_choice = _sampled_ranked_choice(
                        opponent_prior, bridge, node, "p2", rng
                    )
                    p1_choice = _trajectory_choice(
                        result, rng, args.trajectory_exploration
                    )
                    roots = _advance_roots(
                        bridge, roots, p1_choice, p2_choice
                    )
                    node = roots[0].node
                if game_examples:
                    path = _save_chunk(output, game, game_examples)
                    print(f"saved {len(game_examples)} positions -> {path}", flush=True)
                summary = {
                    "opponent": opponent.name,
                    "hidden_sheets": not adapter.reveal_opponent_sets,
                    "rollout_policy": (
                        "latest_candidate" if use_candidate else "champion"
                    ),
                    "positions": len(game_examples),
                    "ended": node.ended,
                    "turn": node.turn,
                    "preview_fallback_reason": preview_fallback_reason,
                }
                completed_count, move_count = _record_game(
                    output,
                    manifest_path,
                    preview_path,
                    game,
                    summary,
                    preview_record,
                )
                sheet_label = "open" if adapter.reveal_opponent_sets else "hidden"
                worker_label = (
                    f"worker={args.worker_index}; "
                    if args.worker_index is not None
                    else ""
                )
                print(
                    f"game {game + 1}/{args.games}: {opponent.name}; "
                    f"{worker_label}sheets={sheet_label}; "
                    f"turn={node.turn} ended={node.ended}; positions={move_count}; "
                    f"completed={completed_count}; "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                failure_traceback = traceback.format_exc()
                if node is not None:
                    failure_path = output / f"failure_{game:05d}.json"
                    _write_json_atomic(
                        failure_path,
                        {
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": failure_traceback,
                            "state": node.state,
                            "requests": node.requests,
                            "turn": node.turn,
                            "request_state": node.request_state,
                        },
                    )
                if game_examples:
                    path = _save_chunk(output, game, game_examples)
                    print(
                        f"saved {len(game_examples)} partial positions -> {path}",
                        flush=True,
                    )
                summary = {
                    "opponent": opponent.name,
                    "hidden_sheets": not adapter.reveal_opponent_sets,
                    "positions": len(game_examples),
                    "ended": bool(node and node.ended),
                    "turn": int(node.turn if node else 0),
                    "error": f"{type(exc).__name__}: {exc}",
                    "request_state": node.request_state if node else None,
                    "request_keys": [
                        sorted(request) if request else []
                        for request in (node.requests if node else [])
                    ],
                }
                _completed_count, _move_count = _record_game(
                    output,
                    manifest_path,
                    preview_path,
                    game,
                    summary,
                    None,
                )
                print(
                    f"game {game + 1}/{args.games}: {opponent.name} failed: "
                    f"{type(exc).__name__}: {exc}; kept={len(game_examples)}",
                    flush=True,
                )
                print(failure_traceback, flush=True)
                with _generation_lock(output):
                    current = json.loads(manifest_path.read_text())
                failed_so_far = sum(
                    "error" in record
                    for record in current.get("completed", {}).values()
                )
                if failed_so_far > args.max_failed_games:
                    raise RuntimeError(
                        f"exact generation aborted: {failed_so_far} failed games "
                        f"exceeds --max-failed-games={args.max_failed_games}; "
                        "the manifest remains resumable"
                    ) from exc
                print(
                    f"tolerating isolated failure {failed_so_far}/"
                    f"{args.max_failed_games}; continuing",
                    flush=True,
                )
                continue
    with _generation_lock(output):
        manifest = json.loads(manifest_path.read_text())
    completed_records = manifest["completed"]
    move_count = sum(
        int(summary.get("positions", 0)) for summary in completed_records.values()
    )
    successful = sum("error" not in summary for summary in completed_records.values())
    remaining = args.games - successful
    failed = sum("error" in summary for summary in completed_records.values())
    print(
        f"complete: {move_count} move positions; "
        f"games={len(completed_records)}/{args.games}; "
        f"failed={failed}; remaining={remaining}; "
        f"elapsed={time.monotonic() - started:.1f}s"
    )
    if args.worker_index is None:
        if failed > args.max_failed_games:
            raise SystemExit(
                f"generation produced {failed} failed games; exact legal-action "
                f"coverage is mandatory beyond --max-failed-games="
                f"{args.max_failed_games}"
            )
        if remaining and args.max_seconds <= 0:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
