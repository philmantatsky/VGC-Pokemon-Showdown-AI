"""Persistent client for Pokemon Showdown's exact serialized-state simulator.

The bridge operates only on Showdown ``Battle.toJSON()`` states. A poke-env battle is
not accepted because reconstructing one incompletely would repeat the central bug in
the old search: calling a hand-mutated state "simulation" despite missing switches,
Protect, Mega Evolution, weather, abilities, items, and targeting effects.

Live search stays disabled until a snapshot synchronizer demonstrates parity against
the local server. The bridge is still useful now for exact forward-model tests and for
building that synchronizer without paying process startup on every matrix cell.
"""

from __future__ import annotations

import atexit
import json
import select
import subprocess
import threading
from pathlib import Path
from typing import Any


class ExactSimulatorError(RuntimeError):
    """The exact simulator rejected a request or terminated unexpectedly."""


class ExactShowdownBridge:
    """JSON-lines client backed by one persistent Node process."""

    def __init__(self, node: str = "node", bridge_path: Path | None = None):
        root = Path(__file__).resolve().parents[2]
        self.node = node
        self.bridge_path = bridge_path or root / "tools" / "exact_showdown_bridge.js"
        self._proc = self._spawn()
        self._lock = threading.Lock()
        self._next_id = 1
        atexit.register(self.close)

    def _spawn(self) -> subprocess.Popen:
        return subprocess.Popen(
            [self.node, str(self.bridge_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def _restart(self) -> None:
        """Replace a timed-out worker; serialized states remain reusable."""
        proc = self._proc
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=1)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()
        self._proc = self._spawn()

    def request(
        self, op: str, *, timeout_s: float | None = None, **payload: Any
    ) -> dict[str, Any]:
        with self._lock:
            if self._proc.poll() is not None:
                detail = self._proc.stderr.read().strip() if self._proc.stderr else ""
                raise ExactSimulatorError(
                    f"exact simulator exited with {self._proc.returncode}: {detail}"
                )
            request_id = self._next_id
            self._next_id += 1
            message = {"id": request_id, "op": op, **payload}
            assert self._proc.stdin is not None
            assert self._proc.stdout is not None
            self._proc.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
            self._proc.stdin.flush()
            if timeout_s is not None:
                ready, _, _ = select.select(
                    [self._proc.stdout], [], [], max(0.0, timeout_s)
                )
                if not ready:
                    self._restart()
                    raise ExactSimulatorError(
                        f"exact simulator request {op!r} exceeded "
                        f"{timeout_s:.3f}s"
                    )
            line = self._proc.stdout.readline()
            if not line:
                detail = self._proc.stderr.read().strip() if self._proc.stderr else ""
                raise ExactSimulatorError(
                    f"exact simulator closed its output: {detail}"
                )
            response = json.loads(line)
            if response.get("id") != request_id:
                raise ExactSimulatorError(
                    f"bridge response id {response.get('id')} != {request_id}"
                )
            if not response.get("ok"):
                raise ExactSimulatorError(
                    response.get("error", "unknown simulator error")
                )
            return response["result"]

    def ping(self) -> dict[str, Any]:
        return self.request("ping")

    def create(self, **kwargs: Any) -> dict[str, Any]:
        """Create an exact Showdown battle, optionally through team preview."""
        return self.request("create", **kwargs)

    def simulate(
        self,
        state: dict[str, Any],
        p1_choice: str,
        p2_choice: str,
        rng_seed: str | None = None,
    ) -> dict[str, Any]:
        """Clone ``state`` and resolve one exact pair of Showdown choices."""
        payload: dict[str, Any] = {
            "state": state,
            "p1_choice": p1_choice,
            "p2_choice": p2_choice,
        }
        if rng_seed is not None:
            payload["rng_seed"] = rng_seed
        return self.request("simulate", **payload)

    def simulate_batch(
        self,
        state: dict[str, Any],
        branches: list[dict[str, Any]],
        timeout_s: float | None = None,
    ) -> list[dict[str, Any]]:
        """Resolve several choice pairs from one state in one bridge round-trip."""
        result = self.request(
            "simulate_batch", state=state, branches=branches, timeout_s=timeout_s
        )
        if not isinstance(result, list):
            raise ExactSimulatorError("simulate_batch returned a non-list result")
        return result

    def choices(self, state: dict[str, Any], side: str) -> list[str]:
        """Return every legal joint choice for one side of an exact state."""
        result = self.request("choices", state=state, side=side)
        if not isinstance(result, list) or not all(isinstance(x, str) for x in result):
            raise ExactSimulatorError("choices returned malformed data")
        return result

    def reconcile(
        self, state: dict[str, Any], snapshot: dict[str, Any]
    ) -> dict[str, Any]:
        """Repair a concrete shadow from client-visible live battle state."""
        return self.request("reconcile", state=state, snapshot=snapshot)

    def close(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is None or proc.poll() is not None:
            return
        if proc.stdin:
            proc.stdin.close()
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1)

    def __enter__(self) -> "ExactShowdownBridge":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def live_snapshot_supported() -> bool:
    """Whether poke-env -> exact Showdown state parity has been established."""
    report = Path(__file__).resolve().parents[2] / "results_parity" / (
        "live_exact_parity.json"
    )
    try:
        payload = json.loads(report.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return bool(
        payload.get("accepted")
        and int(payload.get("snapshot_schema") or 0) >= 2
        and int(payload.get("checked_states") or 0) >= 1000
        and int(payload.get("mismatch_count") or 0) == 0
    )
