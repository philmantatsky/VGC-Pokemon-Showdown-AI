"""The client side of the exact simulator bridge must not leak and must not lose step.

2026-10-06, the search against the battery's rosters: every bridge kept two pipes
open for the life of the process (close() released only stdin, and an atexit hook kept
the object alive). A process that searched its 495th battle got pipes numbered past
1023, select() refused them, the answer to that request stayed in the pipe, and every
later request read the answer to the one before it -- the search fell back to the
bot's own move in 75-90% of decisions for the rest of the process, silently.

A stand-in worker (a few lines of Python speaking the same JSON-lines protocol) makes
each failure on demand; no Showdown checkout is needed.
"""

from __future__ import annotations

import gc
import os
import resource
import sys
import time
import weakref
from pathlib import Path

import pytest

from vgc_bench.src.exact_sim import ExactShowdownBridge, ExactSimulatorError

WORKER = r"""
import json, os, sys, time


def say(**answer):
    print(json.dumps(answer), flush=True)


for line in sys.stdin:
    message = json.loads(line)
    op, number = message["op"], message["id"]
    if op == "stray":  # something else on stdout before the answer
        print("a line nobody asked for", flush=True)
    if op == "stale":  # the answer to the request before this one
        say(id=number - 1, ok=True, result={})
    if op == "sleep":
        time.sleep(message["seconds"])
    if op == "die":
        sys.exit(3)
    if op == "refuse":  # the simulator's own verdict on a request
        say(id=number, ok=False, error="no such move")
        continue
    say(id=number, ok=True, result={"pid": os.getpid()})
"""


@pytest.fixture
def worker(tmp_path: Path) -> Path:
    path = tmp_path / "worker.py"
    path.write_text(WORKER)
    return path


def _bridge(worker: Path) -> ExactShowdownBridge:
    return ExactShowdownBridge(node=sys.executable, bridge_path=worker)


def _descriptors() -> int:
    return len(os.listdir("/dev/fd"))


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _gone(pid: int, within_s: float = 3.0) -> bool:
    deadline = time.monotonic() + within_s
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.02)
    return not _alive(pid)


@pytest.mark.parametrize("fault", ["stray", "stale"])
def test_an_answer_out_of_step_costs_one_request_not_the_bridge(worker, fault):
    """Before: the request after a stale or stray line read the wrong answer, and so
    did every request after that."""
    with _bridge(worker) as bridge:
        first = bridge.ping()["pid"]
        with pytest.raises(ExactSimulatorError):
            bridge.request(fault)
        # a fresh worker, in step again -- and it stays in step
        second = bridge.ping()["pid"]
        assert second != first
        assert bridge.ping()["pid"] == second


def test_a_request_the_simulator_refuses_keeps_the_worker(worker):
    with _bridge(worker) as bridge:
        pid = bridge.ping()["pid"]
        with pytest.raises(ExactSimulatorError, match="no such move"):
            bridge.request("refuse")
        assert bridge.ping()["pid"] == pid  # its answer was read: nothing to repair


def test_a_request_past_its_time_gets_a_fresh_worker(worker):
    with _bridge(worker) as bridge:
        pid = bridge.ping()["pid"]
        with pytest.raises(ExactSimulatorError, match=r"'sleep' exceeded 0\.200s"):
            bridge.request("sleep", seconds=30, timeout_s=0.2)
        assert _gone(pid)
        assert bridge.ping()["pid"] != pid


def test_a_worker_that_died_is_replaced(worker):
    with _bridge(worker) as bridge:
        pid = bridge.ping()["pid"]
        with pytest.raises(ExactSimulatorError, match="closed its output"):
            bridge.request("die")
        assert bridge.ping()["pid"] != pid
        # ... also when it died between two requests
        bridge._proc.kill()
        bridge._proc.wait(timeout=2)
        with pytest.raises(ExactSimulatorError, match="exited with"):
            bridge.ping()
        assert "pid" in bridge.ping()


def test_closing_a_bridge_releases_every_pipe(worker):
    """Two descriptors a battle was the leak; one searched battle is one bridge."""
    before = _descriptors()
    for _ in range(40):
        bridge = _bridge(worker)
        bridge.ping()
        bridge.close()
        bridge.close()  # again is fine
    gc.collect()
    assert _descriptors() <= before + 2


def test_a_restart_releases_the_old_workers_pipes(worker):
    with _bridge(worker) as bridge:
        bridge.ping()
        before = _descriptors()
        for _ in range(20):
            with pytest.raises(ExactSimulatorError):
                bridge.request("stray")
        assert _descriptors() <= before + 2


def test_a_bridge_dropped_without_close_leaves_nothing_behind(worker):
    before = _descriptors()
    bridge = _bridge(worker)
    pid = bridge.ping()["pid"]
    watcher = weakref.ref(bridge)
    del bridge
    gc.collect()
    # nothing pins a bridge for the life of the process (an atexit hook used to)
    assert watcher() is None
    assert _gone(pid)
    assert _descriptors() <= before + 2


def test_requests_work_when_pipes_are_numbered_past_1023(worker):
    """select() cannot watch a descriptor of 1024 or more ("filedescriptor out of
    range in select()"); a process that has opened enough files hands such numbers
    to the next bridge."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft != resource.RLIM_INFINITY and soft < 1300:
        if hard != resource.RLIM_INFINITY and hard < 1300:
            pytest.skip("this process may not open 1,300 files")
        resource.setrlimit(resource.RLIMIT_NOFILE, (1300, hard))
    held = []
    try:
        while not held or held[-1] < 1100:
            held.append(os.open(os.devnull, os.O_RDONLY))
        with _bridge(worker) as bridge:
            assert bridge._proc.stdout is not None
            assert bridge._proc.stdout.fileno() >= 1024
            assert "pid" in bridge.request("ping", timeout_s=5.0)
            with pytest.raises(ExactSimulatorError, match="exceeded"):
                bridge.request("sleep", seconds=30, timeout_s=0.2)
            assert "pid" in bridge.request("ping", timeout_s=5.0)
    finally:
        for descriptor in held:
            os.close(descriptor)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
