"""Monitor repaired training, evaluate its checkpoints, and select a champion.

This intentionally does not ladder. Ladder games are a final distribution-shift
test and require credentials; unattended checkpoint selection should stay local and
reproducible.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results_repaired"
SAVE_DIR = RESULTS / "saves_fp_hs_wt" / "reg_mb" / "seed1"
EVAL_DIR = RESULTS / "overnight_eval"
BASELINE = RESULTS / "converted_v4.zip"
BASELINE_RESULTS = {
    "open": RESULTS / "openings_open_hard.json",
    "hidden": RESULTS / "openings_hidden_hard.json",
    "population": RESULTS / "openings_population_hard.json",
}
POPULATION_OPPONENT = RESULTS / "opponents" / "64opp_3932160_v4.zip"
FINAL_STEP = 7_864_320
POLICY_ARM = "policy preview + policy"
RANDOM_PREVIEW_ARM = "random preview + policy"


def announce(message: str) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{stamp}] {message}", flush=True)


def training_alive(pid: int) -> bool:
    try:
        output = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "command="], text=True
        )
    except subprocess.CalledProcessError:
        return False
    return "vgc_bench.train" in output


def checkpoints() -> list[Path]:
    found = []
    for path in SAVE_DIR.glob("*.zip"):
        try:
            int(path.stem)
        except ValueError:
            continue
        found.append(path)
    return sorted(found, key=lambda path: int(path.stem))


def port_ready(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def ensure_server(port: int) -> subprocess.Popen | None:
    if port_ready(port):
        announce(f"using existing local Showdown server on port {port}")
        return None
    server_log = (EVAL_DIR / "showdown_server.log").open("a")
    process = subprocess.Popen(
        [
            str(ROOT / "pokemon-showdown" / "pokemon-showdown"),
            "start",
            str(port),
            "--no-security",
        ],
        cwd=ROOT / "pokemon-showdown",
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    for _ in range(60):
        if port_ready(port):
            announce(f"started local Showdown server on port {port}")
            return process
        if process.poll() is not None:
            raise RuntimeError("local Showdown server exited during startup")
        time.sleep(1)
    raise TimeoutError(f"local Showdown server did not open port {port}")


def valid_result(path: Path, checkpoint: Path, hidden: bool) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        Path(payload.get("checkpoint", "")).resolve() == checkpoint.resolve()
        and payload.get("hidden_sheets") is hidden
        and payload.get("guard_profile") == "hard"
        and payload.get("n_battles", 0) >= 100
        and payload.get("arms", {}).get(POLICY_ARM, {}).get("battles", 0) >= 100
    )


def evaluate(checkpoint: Path, mode: str, port: int) -> Path:
    output = EVAL_DIR / f"{checkpoint.stem}_{mode}.json"
    hidden = mode == "hidden"
    if valid_result(output, checkpoint, hidden):
        announce(f"reusing {checkpoint.stem} {mode} evaluation")
        return output

    command = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "evaluation" / "eval_openings.py"),
        "--checkpoint",
        str(checkpoint),
        "--port",
        str(port),
        "--device",
        "mps",
        "--n_battles",
        "100",
        "--workers",
        "8",
        "--seed",
        "17",
        "--guard_profile",
        "hard",
        "--output",
        str(output),
    ]
    if hidden:
        command.append("--hidden_sheets")
    if mode == "population":
        command.extend(["--opponent_checkpoint", str(POPULATION_OPPONENT)])

    log_path = EVAL_DIR / f"{checkpoint.stem}_{mode}.log"
    announce(f"evaluating checkpoint {checkpoint.stem}: {mode}")
    env = os.environ.copy()
    env["VGC_KNOWLEDGE_OBS"] = "1"
    with log_path.open("w") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=2 * 60 * 60,
            check=False,
        )
    if completed.returncode != 0 or not valid_result(output, checkpoint, hidden):
        raise RuntimeError(
            f"evaluation failed for {checkpoint.stem} {mode}; see {log_path}"
        )
    return output


def arm_rate(path: Path, arm: str = POLICY_ARM) -> float:
    payload = json.loads(path.read_text())
    return float(payload["arms"][arm]["win_rate"])


def wilson_lower(path: Path, arm: str = POLICY_ARM) -> float:
    result = json.loads(path.read_text())["arms"][arm]
    wins = int(result["wins"])
    total = int(result["battles"])
    z = 1.96
    p = wins / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return center - radius


def result_bundle(checkpoint: Path, paths: dict[str, Path]) -> dict:
    rates = {mode: arm_rate(path) for mode, path in paths.items()}
    preview_deltas = {
        mode: arm_rate(path, POLICY_ARM) - arm_rate(path, RANDOM_PREVIEW_ARM)
        for mode, path in paths.items()
    }
    score = (rates["open"] + rates["hidden"] + 2 * rates["population"]) / 4
    gate = all(wilson_lower(path) >= 0.50 for path in paths.values()) and all(
        delta >= -0.05 for delta in preview_deltas.values()
    )
    return {
        "checkpoint": str(checkpoint),
        "rates": rates,
        "preview_deltas": preview_deltas,
        "score": score,
        "gate_passed": gate,
    }


def choose_champion(candidates: list[dict], baseline: dict) -> tuple[dict, str]:
    eligible = []
    for candidate in candidates:
        if not candidate["gate_passed"]:
            continue
        if any(
            candidate["rates"][mode] < baseline["rates"][mode] - 0.05
            for mode in ("open", "hidden", "population")
        ):
            continue
        eligible.append(candidate)
    if not eligible:
        return baseline, "no new checkpoint passed the safety and regression gates"

    best = max(eligible, key=lambda item: item["score"])
    improvement = best["score"] - baseline["score"]
    if improvement < 0.02:
        return baseline, (
            f"best new checkpoint improved the weighted score by only "
            f"{improvement * 100:.1f}pp; retained baseline to avoid promoting noise"
        )
    return best, (
        f"promoted after a {improvement * 100:.1f}pp weighted improvement with no "
        "evaluation arm regressing by more than 5pp"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-pid", type=int, required=True)
    parser.add_argument("--port", type=int, default=7500)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    known = {path.name for path in checkpoints()}
    announce(
        f"monitoring training PID {args.training_pid}; existing checkpoints: "
        f"{', '.join(sorted(known))}"
    )
    last_heartbeat = 0.0
    while training_alive(args.training_pid):
        current = {path.name for path in checkpoints()}
        for name in sorted(current - known):
            announce(f"new checkpoint saved: {name}")
        known = current
        now = time.monotonic()
        if now - last_heartbeat >= 600:
            latest = max((int(Path(name).stem) for name in known), default=0)
            announce(f"training alive; latest saved step={latest:,}")
            last_heartbeat = now
        time.sleep(max(5, min(args.poll_seconds, 60)))

    saved = checkpoints()
    latest = int(saved[-1].stem) if saved else 0
    if latest >= FINAL_STEP:
        announce(f"training completed at step {latest:,}; beginning checkpoint audit")
    else:
        announce(
            f"training stopped before target; latest checkpoint is {latest:,}. "
            "Auditing recoverable checkpoints without automatically restarting."
        )

    new_checkpoints = [path for path in saved if int(path.stem) > 3_932_160]
    if not new_checkpoints:
        announce("no new checkpoint exists; stopping without changing the champion")
        return
    if not all(path.exists() for path in BASELINE_RESULTS.values()):
        raise FileNotFoundError("baseline evaluation bundle is incomplete")
    if not POPULATION_OPPONENT.exists():
        raise FileNotFoundError(POPULATION_OPPONENT)

    ensure_server(args.port)
    baseline = result_bundle(BASELINE, BASELINE_RESULTS)
    evaluated = []
    for checkpoint in new_checkpoints:
        try:
            paths = {
                mode: evaluate(checkpoint, mode, args.port)
                for mode in ("open", "hidden", "population")
            }
            bundle = result_bundle(checkpoint, paths)
            evaluated.append(bundle)
            announce(
                f"checkpoint {checkpoint.stem}: score={bundle['score']:.3f}, "
                f"rates={bundle['rates']}, gate={bundle['gate_passed']}"
            )
        except Exception as error:
            announce(f"checkpoint {checkpoint.stem} audit failed: {error}")

    champion, reason = choose_champion(evaluated, baseline)
    champion_path = Path(champion["checkpoint"])
    selection = {
        "selected_at": datetime.now().astimezone().isoformat(),
        "training_target_reached": latest >= FINAL_STEP,
        "latest_training_checkpoint": latest,
        "baseline": baseline,
        "candidates": evaluated,
        "champion": champion,
        "reason": reason,
        "next_step": "controlled ladder pilot; not started unattended",
    }
    (RESULTS / "champion_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )
    shutil.copy2(champion_path, RESULTS / "champion.zip")
    announce(f"champion={champion_path}; {reason}")
    announce("overnight work complete; controlled ladder pilot is ready for review")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        announce(f"FATAL: {type(error).__name__}: {error}")
        sys.exit(1)
