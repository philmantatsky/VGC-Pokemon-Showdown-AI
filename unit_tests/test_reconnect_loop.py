"""A lost server connection restarts the challenge listener (2026-10-03: Showdown
dropped its connection at 17:52 and the listener sat deaf for hours): the bot exits
75 and records any unfinished battle, and tools/reconnect_loop.sh reruns it with
--rejoin-battle for that battle."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ladder_ourteam import DISCONNECTED_EXIT, play_until_disconnect, record_disconnect

ROOT = Path(__file__).resolve().parents[1]
LOOP = ROOT / "tools/reconnect_loop.sh"
ROOM = "battle-gen9championsvgc2026regmc-123-abcdef"


def _agent(battles: dict | None = None) -> tuple[Any, concurrent.futures.Future]:
    listening: concurrent.futures.Future = concurrent.futures.Future()
    agent = SimpleNamespace(
        ps_client=SimpleNamespace(_listening_coroutine=listening), battles=battles or {}
    )
    return agent, listening


def test_play_that_finishes_returns_none():
    async def scenario():
        agent, _ = _agent()

        async def play():
            await asyncio.sleep(0.01)

        return await play_until_disconnect(agent, play(), poll=0.01)

    assert asyncio.run(scenario()) is None


def test_lost_connection_cancels_play_and_names_the_unfinished_battle():
    cancelled = []

    async def scenario():
        agent, listening = _agent(
            {
                "battle-gen9championsvgc2026regmc-1-done": SimpleNamespace(
                    finished=True
                ),
                ROOM: SimpleNamespace(finished=False),
            }
        )

        async def play():  # accept_challenges waiting for a challenge forever
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        listening.set_result(None)  # poke-env's listener returned: socket closed
        return await play_until_disconnect(agent, play(), poll=0.01)

    assert asyncio.run(scenario()) == ROOM
    assert cancelled == [True]


def test_lost_connection_without_a_battle_returns_empty():
    async def scenario():
        agent, listening = _agent()
        listening.set_result(None)
        return await play_until_disconnect(agent, asyncio.sleep(3600), poll=0.01)

    assert asyncio.run(scenario()) == ""


def test_errors_in_play_still_raise():
    async def scenario():
        agent, _ = _agent()

        async def play():
            raise RuntimeError("boom")

        await play_until_disconnect(agent, play(), poll=0.01)

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(scenario())


def test_record_disconnect_appends(tmp_path):
    record_disconnect(tmp_path, ROOM)
    path = record_disconnect(tmp_path, "")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["unfinished"] for r in rows] == [ROOM, ""]
    assert DISCONNECTED_EXIT == 75


def _fake(tmp_path: Path, exits: list[int], rooms: list[str]) -> Path:
    """A listener stand-in: call k records its arguments, writes rooms[k] to
    disconnects.jsonl when it exits 75, and exits exits[k] (the last repeats)."""
    script = tmp_path / "fake.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        f'D="{tmp_path}"\n'
        'echo "$*" >> "$D/calls.txt"\n'
        'k=$(($(wc -l < "$D/calls.txt") - 1))\n'
        f"exits=({' '.join(map(str, exits))})\n"
        f"rooms=({' '.join(repr(r) for r in rooms)})\n"
        "n=${#exits[@]}; [ $k -lt $n ] || k=$((n - 1))\n"
        "rc=${exits[$k]}\n"
        'if [ "$rc" = 75 ]; then\n'
        '  printf \'{"at": "x", "unfinished": "%s"}\\n\' "${rooms[$k]}" '
        '>> "$D/disconnects.jsonl"\n'
        "fi\n"
        'exit "$rc"\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def _loop(tmp_path: Path, room: str, fake: Path, **env: str):
    run_env = {**os.environ, "REJOIN_PAUSE": "0", "RECONNECT_PAUSE": "0", **env}
    out = subprocess.run(
        ["bash", str(LOOP), str(tmp_path), room, str(fake), "--challenges"],
        capture_output=True,
        text=True,
        env=run_env,
        timeout=60,
        check=False,
    )
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    return out, calls


def test_loop_reconnects_and_rejoins_the_unfinished_battle(tmp_path):
    out, calls = _loop(tmp_path, "", _fake(tmp_path, [75, 0], [ROOM, ""]))
    assert out.returncode == 0
    assert calls == ["--challenges", f"--challenges --rejoin-battle {ROOM}"]
    assert f"then rejoining {ROOM}" in out.stdout


def test_loop_passes_the_first_room_once(tmp_path):
    first = "battle-gen9championsvgc2026regmc-9-first"
    out, calls = _loop(tmp_path, first, _fake(tmp_path, [75, 0], ["", ""]))
    assert out.returncode == 0
    assert calls == [f"--challenges --rejoin-battle {first}", "--challenges"]


def test_loop_ends_on_any_other_exit(tmp_path):
    out, calls = _loop(tmp_path, "", _fake(tmp_path, [3], [""]))
    assert out.returncode == 3
    assert calls == ["--challenges"]


def test_loop_gives_up_after_short_runs_in_a_row(tmp_path):
    out, calls = _loop(tmp_path, "", _fake(tmp_path, [75], [""]), MAX_QUICK="3")
    assert out.returncode == 75
    assert len(calls) == 3
    assert "RECONNECT_GAVE_UP" in out.stdout


def test_loop_rejoins_only_reg_mc_rooms(tmp_path):
    other = "battle-gen9randombattle-1"
    out, calls = _loop(tmp_path, "", _fake(tmp_path, [75, 0], [other, ""]))
    assert out.returncode == 0
    assert calls == ["--challenges", "--challenges"]


def test_challenge_launcher_runs_under_the_loop():
    text = (ROOT / "tools/challenges_deployed.sh").read_text()
    assert 'exec tools/reconnect_loop.sh "$DIR" "$REJOIN"' in text
    assert "CHALLENGE_REFUSED rejoin room is not Reg M-C" in text
    assert os.access(LOOP, os.X_OK)
    parse = subprocess.run(["bash", "-n", str(LOOP)], check=False)
    assert parse.returncode == 0
