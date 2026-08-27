"""Final paired local rollout gate for the future-looking fixed-team bot.

This is deliberately separate from aggregation training. It evaluates only the
selected candidate, records every local loss and exact-search fallback, and writes a
new deployment manifest without modifying ``results_repaired/champion.zip``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from run_counterfactual_pipeline import (
    ROOT,
    _ensure_server,
    _score_evaluations,
    _stop_server,
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _valid_result(path: Path, residual: Path, battles: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
        live = payload["arms"]["live_exact_search"]
        distilled = payload["arms"]["distilled_policy"]
        return (
            int(payload.get("evaluation_schema", 0)) == 3
            and int(live["battles"]) >= battles
            and Path(live["residual_ranker"]).resolve() == residual.resolve()
            and Path(distilled["residual_ranker"]).resolve() == residual.resolve()
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _audit_fallbacks(payload: dict) -> list[dict]:
    audit = payload["arms"]["live_exact_search"]["move_search"].get("audit")
    if not audit:
        return []
    path = Path(audit)
    if not path.exists():
        return [{"error": f"missing exact audit: {path}"}]
    fallbacks = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        exact = row.get("exact_search") or {}
        if exact.get("decision_fallback"):
            fallbacks.append(
                {
                    "battle": row.get("battle"),
                    "turn": row.get("turn"),
                    **exact["decision_fallback"],
                }
            )
    return fallbacks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="results_repaired/champion.zip")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--residual", required=True)
    parser.add_argument(
        "--outcome-value", default="results_outcome_v1/outcome_value.zip"
    )
    parser.add_argument(
        "--baseline-preview", default="data/opponent_preview_top500_regmb.pt"
    )
    parser.add_argument("--candidate-preview", default="")
    parser.add_argument("--candidate-preview-trained", action="store_true")
    parser.add_argument(
        "--population",
        default="results_repaired/opponents/64opp_3932160_v4.zip",
    )
    parser.add_argument("--output", default="results_rollout_v1")
    parser.add_argument("--battles", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--port", type=int, default=7700)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    baseline = (ROOT / args.baseline).resolve()
    candidate = (ROOT / args.candidate).resolve()
    residual = (ROOT / args.residual).resolve()
    outcome = (ROOT / args.outcome_value).resolve()
    baseline_preview = (ROOT / args.baseline_preview).resolve()
    candidate_preview = (
        (ROOT / args.candidate_preview).resolve()
        if args.candidate_preview
        else None
    )
    population = (ROOT / args.population).resolve()
    output = (ROOT / args.output).resolve()
    required = [baseline, candidate, residual, outcome, baseline_preview, population]
    if args.candidate_preview_trained:
        if candidate_preview is None:
            raise SystemExit(
                "--candidate-preview-trained requires --candidate-preview"
            )
        required.append(candidate_preview)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing rollout inputs: " + ", ".join(missing))
    if args.battles < 1 or not 1 <= args.workers <= 8:
        raise SystemExit("--battles must be positive and --workers must be in [1, 8]")

    output.mkdir(parents=True, exist_ok=True)
    server = _ensure_server(args.port, output / "showdown_server.log")
    paths = {
        mode: output / f"eval_{mode}.json"
        for mode in ("open", "hidden", "population")
    }
    try:
        for mode, path in paths.items():
            if _valid_result(path, residual, args.battles):
                print(f"reusing completed {mode} rollout", flush=True)
                continue
            command = [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "eval_counterfactual.py"),
                "--baseline",
                str(baseline),
                "--candidate",
                str(candidate),
                "--candidate-residual",
                str(residual),
                "--baseline-preview",
                str(baseline_preview),
                "--include-live-search",
                "--outcome-value",
                str(outcome),
                "--port",
                str(args.port),
                "--device",
                args.device,
                "--n-battles",
                str(args.battles),
                "--workers",
                str(args.workers),
                "--seed",
                str(args.seed + {"open": 11, "hidden": 23, "population": 37}[mode]),
                "--replay-dir",
                str(output / "replays"),
                "--output",
                str(path),
            ]
            if args.candidate_preview_trained:
                assert candidate_preview is not None
                command.extend(
                    [
                        "--candidate-preview",
                        str(candidate_preview),
                        "--candidate-preview-trained",
                    ]
                )
            if mode == "hidden":
                command.append("--hidden-sheets")
            elif mode == "population":
                command.extend(["--opponent-checkpoint", str(population)])
            print(f"starting {mode} rollout", flush=True)
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode:
                raise SystemExit(f"{mode} rollout failed: {completed.returncode}")
    finally:
        _stop_server(server)

    payloads = {mode: json.loads(path.read_text()) for mode, path in paths.items()}
    evaluation = _score_evaluations(paths)
    live = evaluation["candidates"]["live_exact_search"]
    fallback_rows = [
        {"mode": mode, **row}
        for mode, payload in payloads.items()
        for row in _audit_fallbacks(payload)
    ]
    losses = [
        {"mode": mode, **battle}
        for mode, payload in payloads.items()
        for battle in payload["arms"]["live_exact_search"].get(
            "battle_results", []
        )
        if battle.get("lost")
    ]
    timeout_counts = {
        mode: sum(
            int(value)
            for key, value in payload["arms"]["live_exact_search"][
                "telemetry"
            ].items()
            if "timeout" in key
        )
        for mode, payload in payloads.items()
    }
    latencies = {
        mode: payload["arms"]["live_exact_search"]["move_search"].get(
            "latency_ms", {}
        )
        for mode, payload in payloads.items()
    }

    tactical_path = output / "tactical_gate.json"
    tactical = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "evaluate_tactical_gate.py"),
            "--output",
            str(tactical_path),
            "--minimum",
            "0.90",
        ],
        cwd=ROOT,
        check=False,
    )
    tactical_payload = json.loads(tactical_path.read_text())
    completed_battles = all(
        int(payload["arms"]["live_exact_search"]["battles"]) >= args.battles
        for payload in payloads.values()
    )
    latency_ok = all(
        values.get("max") is not None and float(values["max"]) <= 10000.0
        for values in latencies.values()
    )
    accepted = bool(
        live["score_delta"] >= 0.02
        and live["passes_regression_gate"]
        and tactical.returncode == 0
        and tactical_payload.get("accepted")
        and completed_battles
        and latency_ok
    )
    review = {
        "created_at": _now(),
        "evaluation": evaluation,
        "live_search": live,
        "loss_count": len(losses),
        "losses": losses,
        "fallback_count": len(fallback_rows),
        "fallbacks": fallback_rows,
        "timeouts": timeout_counts,
        "latency_ms": latencies,
        "tactical_gate": tactical_payload,
        "completed_battles": completed_battles,
        "latency_ok": latency_ok,
        "accepted": accepted,
    }
    _write_json(output / "rollout_review.json", review)
    deployment = {
        "schema": 1,
        "created_at": _now(),
        "passed": accepted,
        "checkpoint": str(candidate),
        "residual_ranker": str(residual),
        "preview_model": str(
            candidate_preview
            if args.candidate_preview_trained
            else baseline_preview
        ),
        "learned_preview": bool(args.candidate_preview_trained),
        "use_search": True,
        "outcome_value": str(outcome),
        "fixed_team": str((ROOT / "teams/reg_mb/our_team.txt").resolve()),
        "source_champion_untouched": str(baseline),
        "review": str(output / "rollout_review.json"),
        "reason": (
            "full local rollout gate passed"
            if accepted
            else "full local rollout gate rejected; inspect rollout_review.json"
        ),
    }
    _write_json(output / "deployment.json", deployment)
    print(
        f"rollout {'PASSED' if accepted else 'REJECTED'}: "
        f"live delta={live['score_delta'] * 100:+.1f}pp, "
        f"losses={len(losses)}, fallbacks={len(fallback_rows)}",
        flush=True,
    )
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
