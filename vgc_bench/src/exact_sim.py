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
import weakref
from pathlib import Path
from typing import Any


class ExactSimulatorError(RuntimeError):
    """The exact simulator rejected a request or terminated unexpectedly."""


# Every bridge still open, held weakly. Until 2026-10-06 each bridge registered its
# own close() with atexit, which kept the object -- and the two output pipes close()
# never released -- alive for the whole process: two file descriptors a battle. A
# process that searched its 495th battle got pipes numbered past 1023, select()
# refused them ("filedescriptor out of range"), the answer to that request stayed
# in the pipe, and every later request read the one before it ("bridge response id
# 23 != 24"): the search was dead for the rest of the process and nothing said so.
_OPEN_BRIDGES: weakref.WeakSet[ExactShowdownBridge] = weakref.WeakSet()


def _close_open_bridges() -> None:
    for bridge in list(_OPEN_BRIDGES):
        bridge.close()


atexit.register(_close_open_bridges)


def _release(proc: subprocess.Popen, *, wait_s: float = 1.0) -> None:
    """End a worker and close all three of its pipes."""
    if proc.poll() is None:
        try:
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()  # the worker leaves at end of input
            proc.wait(timeout=wait_s)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            proc.kill()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        try:
            if stream is not None and not stream.closed:
                stream.close()
        except (OSError, ValueError):
            pass


class ExactShowdownBridge:
    """JSON-lines client backed by one persistent Node process."""

    def __init__(self, node: str = "node", bridge_path: Path | None = None):
        root = Path(__file__).resolve().parents[2]
        self.node = node
        self.bridge_path = bridge_path or root / "tools" / "exact_showdown_bridge.js"
        self._proc = self._spawn()
        self._lock = threading.Lock()
        self._next_id = 1
        _OPEN_BRIDGES.add(self)

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
        """Replace a worker whose stream can no longer be trusted; serialized states
        remain reusable."""
        proc = self._proc
        if proc.poll() is None:
            proc.kill()
        _release(proc)
        self._proc = self._spawn()

    @staticmethod
    def _readable(stream: Any, timeout_s: float) -> bool:
        """Whether ``stream`` has something to read within ``timeout_s``. poll(), not
        select(): select() cannot watch a descriptor numbered 1024 or more."""
        poller = select.poll()
        poller.register(stream, select.POLLIN)
        return bool(poller.poll(max(0.0, timeout_s) * 1000.0))

    @staticmethod
    def _last_words(proc: subprocess.Popen) -> str:
        try:
            return proc.stderr.read().strip() if proc.stderr else ""
        except (OSError, ValueError):
            return ""

    def request(
        self, op: str, *, timeout_s: float | None = None, **payload: Any
    ) -> dict[str, Any]:
        with self._lock:
            if self._proc.poll() is not None:
                # a worker that died between requests: say so once, start another
                code, detail = self._proc.returncode, self._last_words(self._proc)
                self._restart()
                raise ExactSimulatorError(
                    f"exact simulator exited with {code}: {detail}"
                )
            proc = self._proc
            request_id = self._next_id
            self._next_id += 1
            message = {"id": request_id, "op": op, **payload}
            try:
                if proc.stdin is None or proc.stdout is None:
                    raise ExactSimulatorError("exact simulator has no pipes")
                proc.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
                proc.stdin.flush()
                if timeout_s is not None and not self._readable(proc.stdout, timeout_s):
                    raise ExactSimulatorError(
                        f"exact simulator request {op!r} exceeded {timeout_s:.3f}s"
                    )
                line = proc.stdout.readline()
                if not line:
                    raise ExactSimulatorError(
                        f"exact simulator closed its output: {self._last_words(proc)}"
                    )
                response = json.loads(line)
                if not isinstance(response, dict) or response.get("id") != request_id:
                    got = response.get("id") if isinstance(response, dict) else response
                    raise ExactSimulatorError(
                        f"bridge response id {got} != {request_id}"
                    )
            except Exception as error:
                # Whatever happened between sending the request and holding its own
                # answer, an answer may still be on its way down this pipe, and the
                # next request would read it as its own. Only a fresh worker has a
                # stream that is known to be in step.
                try:
                    self._restart()
                except OSError as failure:
                    raise ExactSimulatorError(
                        f"exact simulator could not be restarted after {error}: "
                        f"{failure}"
                    ) from error
                if isinstance(error, ExactSimulatorError):
                    raise
                raise ExactSimulatorError(
                    f"exact simulator request {op!r} failed: "
                    f"{type(error).__name__}: {error}"
                ) from error
            if not response.get("ok"):
                # the simulator's own verdict on this request: the stream is in step
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
        """End the worker and release its pipes; safe to call again."""
        proc = getattr(self, "_proc", None)
        if proc is not None:
            _release(proc)
        _OPEN_BRIDGES.discard(self)

    def __del__(self) -> None:
        # a bridge dropped without close(): no worker, no pipe is left behind
        try:
            self.close()
        except Exception:  # noqa: BLE001 - never raise from a finalizer
            pass

    def __enter__(self) -> "ExactShowdownBridge":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def live_snapshot_supported() -> bool:
    """Whether poke-env -> exact Showdown state parity has been established."""
    report = (
        Path(__file__).resolve().parents[2]
        / "results_parity"
        / ("live_exact_parity.json")
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
