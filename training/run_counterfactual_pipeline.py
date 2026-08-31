"""Resumable planner -> distillation -> local promotion pipeline.

This script never accesses Showdown credentials and never starts ladder games. A
candidate is copied to the deployable paths only after it beats the current champion
in the local open-sheet, hidden-sheet, and learned-population gate.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class Pipeline:
    def __init__(self, output: Path):
        self.output = output
        self.status_path = output / "pipeline_status.json"
        self.status = {
            "started_at": _now(),
            "updated_at": _now(),
            "state": "running",
            "stage": "initializing",
            "message": "",
        }
        _write_json_atomic(self.status_path, self.status)

    def update(self, stage: str, message: str, **extra) -> None:
        self.status.update(
            {"updated_at": _now(), "stage": stage, "message": message, **extra}
        )
        _write_json_atomic(self.status_path, self.status)
        print(f"[{_now()}] {stage}: {message}", flush=True)

    def finish(self, state: str, message: str, **extra) -> None:
        self.status.update(
            {
                "updated_at": _now(),
                "finished_at": _now(),
                "state": state,
                "message": message,
                **extra,
            }
        )
        _write_json_atomic(self.status_path, self.status)
        print(f"[{_now()}] {state.upper()}: {message}", flush=True)

    def run(self, stage: str, command: list[str]) -> None:
        log_path = self.output / "logs" / f"{stage}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.update(stage, "started", command=command, log=str(log_path))
        with log_path.open("a") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                returncode = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise
        if returncode:
            raise RuntimeError(
                f"{stage} exited with status {returncode}; see {log_path}"
            )
        self.update(stage, "completed")


def _port_ready(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def _ensure_server(port: int, log_path: Path) -> subprocess.Popen | None:
    if _port_ready(port):
        print(f"using existing local Showdown server on port {port}", flush=True)
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a")
    process = subprocess.Popen(
        [
            str(ROOT / "pokemon-showdown" / "pokemon-showdown"),
            "start",
            str(port),
            "--no-security",
        ],
        cwd=ROOT / "pokemon-showdown",
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    for _ in range(60):
        if _port_ready(port):
            print(f"started local Showdown server on port {port}", flush=True)
            return process
        if process.poll() is not None:
            raise RuntimeError("local Showdown server exited during startup")
        time.sleep(1)
    raise TimeoutError(f"local Showdown server did not open port {port}")


def _stop_server(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass


def _dataset_summary(directory: Path) -> dict:
    total = 0
    usable = 0
    hidden = 0
    games = set()
    for path in sorted(directory.glob("moves_*.npz")):
        payload = np.load(path)
        metadata = [json.loads(str(value)) for value in payload["metadata"]]
        total += len(metadata)
        usable += sum(
            not record.get("truncated", False)
            or bool(record.get("complete_screen", False))
            for record in metadata
        )
        hidden += sum(bool(record.get("hidden_sheets")) for record in metadata)
        games.update(int(record.get("game", -1)) for record in metadata)
    return {
        "positions": total,
        "usable_positions": usable,
        "games_with_positions": len(games),
        "hidden_positions": hidden,
    }


def _preview_rows(directories: list[Path], destination: Path) -> int:
    """Merge genuine preview labels from all retained aggregation rounds."""
    lines: list[str] = []
    for directory in directories:
        path = directory / "preview_rankings.jsonl"
        if not path.exists():
            continue
        lines.extend(line for line in path.read_text().splitlines() if line.strip())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def _valid_evaluation(
    path: Path, candidate: Path, residual: Path, battles: int
) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
        if int(payload.get("evaluation_schema", 0)) != 3:
            return False
        arm = payload["arms"]["distilled_policy"]
        return (
            Path(arm["checkpoint"]).resolve() == candidate.resolve()
            and Path(arm["residual_ranker"]).resolve() == residual.resolve()
            and int(arm["battles"]) >= battles
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _score_evaluations(paths: dict[str, Path]) -> dict:
    weights = {"open": 1.0, "hidden": 1.0, "population": 2.0}
    payloads = {mode: json.loads(path.read_text()) for mode, path in paths.items()}
    common_arms = set.intersection(
        *(set(payload["arms"]) for payload in payloads.values())
    )
    required = {"champion_policy", "distilled_policy"}
    if not required.issubset(common_arms):
        raise ValueError(
            f"evaluation arms missing {sorted(required - common_arms)}; "
            "stale or mislabeled evaluation output cannot be promoted"
        )
    optional = (
        "preview_policy",
        "live_exact_search",
    )
    arms = ("champion_policy", "distilled_policy") + tuple(
        arm for arm in optional if arm in common_arms
    )
    rates = {arm: {} for arm in arms}
    for mode, payload in payloads.items():
        for arm in arms:
            rates[arm][mode] = float(payload["arms"][arm]["win_rate"])
    scores = {
        arm: sum(weights[mode] * rate for mode, rate in modes.items())
        / sum(weights.values())
        for arm, modes in rates.items()
    }
    baseline = rates["champion_policy"]
    candidates = {}
    for arm in arms[1:]:
        deltas = {mode: rates[arm][mode] - baseline[mode] for mode in weights}
        candidates[arm] = {
            "score": scores[arm],
            "score_delta": scores[arm] - scores["champion_policy"],
            "rates": rates[arm],
            "deltas": deltas,
            "passes_regression_gate": min(deltas.values()) >= -0.02,
        }
    return {
        "weights": weights,
        "champion": {"score": scores["champion_policy"], "rates": baseline},
        "candidates": candidates,
    }


def _promotion(
    output: Path,
    baseline: Path,
    baseline_preview: Path,
    candidate: Path,
    candidate_residual: Path,
    candidate_preview: Path,
    evaluation: dict,
    preview_trained: bool,
) -> dict:
    eligible = [
        (name, result)
        for name, result in evaluation["candidates"].items()
        if result["passes_regression_gate"] and result["score_delta"] >= 0.02
    ]
    selected = max(eligible, key=lambda item: item[1]["score"]) if eligible else None
    if selected is None:
        return {
            "passed": False,
            "reason": (
                "no candidate improved the weighted score by 2pp without a mode "
                "regressing more than 2pp"
            ),
            "evaluation": evaluation,
        }

    arm, result = selected
    learned_preview = arm == "preview_policy" or (
        arm == "live_exact_search" and preview_trained
    )
    use_search = arm == "live_exact_search"
    use_residual = arm in {
        "distilled_policy",
        "preview_policy",
        "live_exact_search",
    }
    champion = output / "champion.zip"
    preview = output / "champion_preview.pt"
    residual = output / "champion_residual.pt"
    shutil.copy2(candidate, champion)
    shutil.copy2(candidate_preview if learned_preview else baseline_preview, preview)
    if use_residual:
        shutil.copy2(candidate_residual, residual)
    return {
        "passed": True,
        "selected_arm": arm,
        "learned_preview": learned_preview,
        "use_search": use_search,
        "residual_ranker": str(residual) if use_residual else None,
        "checkpoint": str(champion),
        "preview_model": str(preview),
        "source_checkpoint": str(candidate),
        "baseline_checkpoint": str(baseline),
        "reason": (
            f"weighted local score improved by {result['score_delta'] * 100:.1f}pp "
            "and passed all regression gates"
        ),
        "evaluation": evaluation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="results_repaired/champion.zip")
    parser.add_argument(
        "--outcome-value",
        default="",
        help="calibrated terminal-outcome checkpoint used by exact label search",
    )
    parser.add_argument(
        "--baseline-preview", default="data/opponent_preview_top500_regmb.pt"
    )
    parser.add_argument("--data", default="counterfactual_data_v3")
    parser.add_argument(
        "--historical-data",
        action="append",
        default=[],
        help="earlier aggregation directory retained during residual training",
    )
    parser.add_argument("--output", default="results_counterfactual_v3")
    parser.add_argument(
        "--rollout-residual",
        default="",
        help="latest candidate used for half of generated trajectories",
    )
    parser.add_argument(
        "--opponent-base-checkpoint",
        default="",
        help=(
            "policy whose ranking drives the OPPONENT's search branches and "
            "rollout choices during generation (e.g. the human-imitation "
            "bc_mix_A); empty keeps the champion ranking both sides"
        ),
    )
    parser.add_argument(
        "--minimum-validation-positions",
        type=int,
        default=0,
        help="hard-fail residual training when the held-out split is smaller",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--max-failed-games",
        type=int,
        default=0,
        help="isolated generation failures tolerated before aborting",
    )
    parser.add_argument(
        "--move-model", default="data/opponent_move_top500_regmb.pt"
    )
    parser.add_argument(
        "--switch-model", default="data/opponent_switch_top500_regmb.pt"
    )
    parser.add_argument("--opponent-model-weight", type=float, default=0.60)
    parser.add_argument("--open-sheet-model-weight", type=float, default=0.40)
    parser.add_argument("--trajectory-exploration", type=float, default=0.20)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="reuse an existing failure-free generation manifest and NPZ dataset",
    )
    parser.add_argument(
        "--generation-seconds",
        type=float,
        default=5400.0,
        help="maximum search-label generation time (default: 90 minutes)",
    )
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--root-width", type=int, default=8)
    parser.add_argument("--opponent-width", type=int, default=6)
    parser.add_argument("--continuation-width", type=int, default=3)
    parser.add_argument("--replacement-width", type=int, default=2)
    parser.add_argument("--chance-samples", type=int, default=4)
    parser.add_argument("--budget", type=float, default=9.0)
    parser.add_argument("--hidden-sheet-prob", type=float, default=0.50)
    parser.add_argument("--minimum-positions", type=int, default=400)
    parser.add_argument("--minimum-preview-positions", type=int, default=1500)
    parser.add_argument("--generation-workers", type=int, default=4)
    parser.add_argument("--training-device", default="mps")
    parser.add_argument("--evaluation-device", default="mps")
    parser.add_argument("--evaluation-battles", type=int, default=500)
    parser.add_argument("--evaluation-workers", type=int, default=8)
    parser.add_argument(
        "--evaluate-live-search",
        action="store_true",
        help="include a separately labeled live exact-search arm once parity is ready",
    )
    parser.add_argument("--port", type=int, default=7600)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    if args.evaluate_live_search and not args.outcome_value:
        raise SystemExit("--evaluate-live-search requires --outcome-value")

    output = (ROOT / args.output).resolve()
    data = (ROOT / args.data).resolve()
    historical_data = [(ROOT / path).resolve() for path in args.historical_data]
    training_data = [*historical_data, data]
    baseline = (ROOT / args.baseline).resolve()
    baseline_preview = (ROOT / args.baseline_preview).resolve()
    candidate = output / "candidate.zip"
    candidate_residual = output / "candidate_residual.pt"
    candidate_preview = output / "candidate_preview.pt"
    output.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline(output)
    server_process = None

    try:
        generation_command = (
            [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "datagen" / "generate_counterfactuals.py"),
                "--checkpoint",
                str(baseline),
                "--preview-model",
                str(baseline_preview),
                "--move-model",
                str((ROOT / args.move_model).resolve()),
                "--switch-model",
                str((ROOT / args.switch_model).resolve()),
                "--opponent-model-weight",
                str(args.opponent_model_weight),
                "--open-sheet-model-weight",
                str(args.open_sheet_model_weight),
                "--trajectory-exploration",
                str(args.trajectory_exploration),
                "--output",
                str(data),
                "--games",
                str(args.games),
                "--max-seconds",
                str(args.generation_seconds),
                "--max-turns",
                str(args.max_turns),
                "--depth",
                str(args.depth),
                "--root-width",
                str(args.root_width),
                "--opponent-width",
                str(args.opponent_width),
                "--continuation-width",
                str(args.continuation_width),
                "--replacement-width",
                str(args.replacement_width),
                "--chance-samples",
                str(args.chance_samples),
                "--budget",
                str(args.budget),
                "--hidden-sheet-prob",
                str(args.hidden_sheet_prob),
                "--device",
                "cpu",
                "--seed",
                str(args.seed),
                "--workers",
                str(args.generation_workers),
                "--max-failed-games",
                str(args.max_failed_games),
            ]
            + (
                ["--rollout-residual", str((ROOT / args.rollout_residual).resolve())]
                if args.rollout_residual
                else []
            )
            + (
                ["--outcome-value", str((ROOT / args.outcome_value).resolve())]
                if args.outcome_value
                else []
            )
            + (
                [
                    "--opponent-base-checkpoint",
                    str((ROOT / args.opponent_base_checkpoint).resolve()),
                ]
                if args.opponent_base_checkpoint
                else []
            )
        )
        if args.skip_generation:
            manifest_path = data / "generation_manifest.json"
            if not manifest_path.exists():
                raise RuntimeError(
                    "--skip-generation requires an existing generation manifest"
                )
            manifest = json.loads(manifest_path.read_text())
            records = manifest.get("completed", {})
            failures = {
                game: record.get("error")
                for game, record in records.items()
                if record.get("error")
            }
            if failures:
                raise RuntimeError(
                    f"cannot reuse generation with {len(failures)} failed games"
                )
            pipeline.update(
                "generate",
                "reusing failure-free generated dataset",
                completed_games=len(records),
                reused=True,
            )
        else:
            pipeline.run("generate", generation_command)
        summary = _dataset_summary(data)
        pipeline.update("validate_data", "dataset inspected", dataset=summary)
        if summary["usable_positions"] < args.minimum_positions:
            raise RuntimeError(
                f"only {summary['usable_positions']} complete planner labels; "
                f"minimum is {args.minimum_positions}"
            )

        # The champion actor stays byte-for-byte frozen. The candidate policy path is
        # only a local copy used to keep evaluation arm schemas uniform.
        shutil.copy2(baseline, candidate)
        pipeline.run(
            "train_residual",
            [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "training" / "train_residual_ranker.py"),
                "--data",
                *[str(directory) for directory in training_data],
                "--checkpoint",
                str(baseline),
                "--output",
                str(candidate_residual),
                "--device",
                args.training_device,
                "--validation-fraction",
                str(args.validation_fraction),
                "--minimum-validation-positions",
                str(args.minimum_validation_positions),
                "--seed",
                str(args.seed),
            ],
        )
        training_metrics = json.loads(
            candidate_residual.with_suffix(".metrics.json").read_text()
        )
        selected_epoch = int(training_metrics.get("selected_epoch", 0))
        selected_metrics = (
            training_metrics.get("history", [])[selected_epoch - 1]
            if selected_epoch > 0
            else {}
        )
        baseline_rank = float(training_metrics.get("baseline_rank_accuracy", 0.0))
        rank_gain = float(
            selected_metrics.get("validation_rank_accuracy", 0.0)
        ) - baseline_rank
        if selected_epoch == 0 or rank_gain < 0.005:
            rejection = {
                "passed": False,
                "created_at": _now(),
                "reason": (
                    "residual did not improve held-out action ranking by 0.5pp; "
                    "local battle evaluation was skipped"
                ),
                "dataset": summary,
                "training": training_metrics,
                "rank_gain": rank_gain,
            }
            _write_json_atomic(output / "deployment.json", rejection)
            pipeline.finish("rejected", rejection["reason"], promotion=rejection)
            return

        pipeline.run(
            "tactical_gate",
            [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "evaluation" / "evaluate_tactical_gate.py"),
                "--output",
                str(output / "tactical_gate.json"),
                "--minimum",
                "0.90",
            ],
        )

        merged_preview = output / "preview_rankings_all.jsonl"
        preview_rows = _preview_rows(training_data, merged_preview)
        preview_trained = preview_rows >= args.minimum_preview_positions
        if preview_trained:
            pipeline.run(
                "train_preview",
                [
                    str(ROOT / ".venv/bin/python"),
                    str(ROOT / "training" / "train_counterfactual_preview.py"),
                    "--data",
                    str(merged_preview),
                    "--checkpoint",
                    str(baseline_preview),
                    "--output",
                    str(candidate_preview),
                    "--device",
                    args.training_device,
                    "--seed",
                    str(args.seed),
                ],
            )
        else:
            pipeline.update(
                "train_preview",
                f"preview candidate disabled; only {preview_rows} labels",
                preview_rows=preview_rows,
                minimum_preview_positions=args.minimum_preview_positions,
                trained=False,
            )

        server_process = _ensure_server(
            args.port, output / "logs" / "showdown_server.log"
        )
        population = ROOT / "results_repaired/opponents/64opp_3932160_v4.zip"
        from vgc_bench.src.utils import refuse_eval_only_checkpoint

        refuse_eval_only_checkpoint(population)
        top_residuals = [
            Path(record["path"]).resolve()
            for record in training_metrics.get("top_three", [])[:3]
        ]
        if not top_residuals:
            top_residuals = [candidate_residual]
        evaluated: list[tuple[Path, dict]] = []
        for residual_index, residual_path in enumerate(top_residuals, start=1):
            evaluations = {
                mode: output / f"eval_residual{residual_index}_{mode}.json"
                for mode in ("open", "hidden", "population")
            }
            for mode, result_path in evaluations.items():
                stage = f"evaluate_residual{residual_index}_{mode}"
                if _valid_evaluation(
                    result_path,
                    candidate,
                    residual_path,
                    args.evaluation_battles,
                ):
                    pipeline.update(stage, "reusing completed evaluation")
                    continue
                command = [
                    str(ROOT / ".venv/bin/python"),
                    str(ROOT / "evaluation" / "eval_counterfactual.py"),
                    "--baseline",
                    str(baseline),
                    "--candidate",
                    str(candidate),
                    "--candidate-residual",
                    str(residual_path),
                    "--baseline-preview",
                    str(baseline_preview),
                    "--port",
                    str(args.port),
                    "--device",
                    args.evaluation_device,
                    "--n-battles",
                    str(args.evaluation_battles),
                    "--workers",
                    str(args.evaluation_workers),
                    "--seed",
                    str(
                        args.seed
                        + {"open": 11, "hidden": 23, "population": 37}[mode]
                    ),
                    "--output",
                    str(result_path),
                ]
                if preview_trained:
                    command.extend(
                        [
                            "--candidate-preview",
                            str(candidate_preview),
                            "--candidate-preview-trained",
                        ]
                    )
                if args.evaluate_live_search:
                    command.extend(
                        [
                            "--include-live-search",
                            "--outcome-value",
                            str((ROOT / args.outcome_value).resolve()),
                        ]
                    )
                if mode == "hidden":
                    command.append("--hidden-sheets")
                if mode == "population":
                    if not population.exists():
                        raise FileNotFoundError(population)
                    command.extend(["--opponent-checkpoint", str(population)])
                pipeline.run(stage, command)
            evaluated.append((residual_path, _score_evaluations(evaluations)))

        def evaluation_key(item: tuple[Path, dict]) -> tuple[bool, float]:
            residual_result = item[1]["candidates"]["distilled_policy"]
            passes = bool(
                residual_result["passes_regression_gate"]
                and residual_result["score_delta"] >= 0.02
            )
            return passes, float(residual_result["score"])

        selected_residual, evaluation = max(evaluated, key=evaluation_key)
        promotion = _promotion(
            output,
            baseline,
            baseline_preview,
            candidate,
            selected_residual,
            candidate_preview,
            evaluation,
            preview_trained,
        )
        promotion["created_at"] = _now()
        promotion["dataset"] = summary
        promotion["training_data"] = [str(path) for path in training_data]
        promotion["evaluated_residuals"] = [str(path) for path, _ in evaluated]
        _write_json_atomic(output / "deployment.json", promotion)
        if promotion["passed"]:
            pipeline.finish("passed", promotion["reason"], promotion=promotion)
        else:
            pipeline.finish("rejected", promotion["reason"], promotion=promotion)
    except KeyboardInterrupt:
        pipeline.finish(
            "interrupted", "pipeline stopped; completed games are resumable"
        )
        raise
    except Exception as error:
        pipeline.finish("failed", f"{type(error).__name__}: {error}")
        raise
    finally:
        _stop_server(server_process)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        sys.exit(1)
