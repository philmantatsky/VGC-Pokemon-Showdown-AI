"""Live check (2026-10-03): a dropped server connection ends the listener's wait.

Showdown dropped the challenge listener's connection at 17:52 ("no close frame
received or sent") and the listener sat deaf for hours: poke-env stopped reading
while accept_challenges kept waiting. This starts a throwaway local Showdown
server, has real poke-env players connect, kills the server with SIGKILL (the same
close-frame-less drop) and checks that ladder_ourteam.play_until_disconnect
returns within seconds -- once while waiting for a challenge (no battle: "") and
once in the middle of a battle (that battle's room, for the restart to rejoin).

Run from the repo root (no other server needed; uses port 7612):
    .venv/bin/python checks/disconnect_live.py
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poke_env.player import RandomPlayer  # noqa: E402
from poke_env.ps_client import ServerConfiguration  # noqa: E402

from evaluation.opening_study import fresh_local_account  # noqa: E402
from ladder_ourteam import play_until_disconnect  # noqa: E402

PORT = 7612
SERVER = ServerConfiguration(
    f"ws://localhost:{PORT}/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)
FORMAT = "gen9randombattle"


class Stalling(RandomPlayer):
    """Never answers a move request, so the battle stays open until the kill."""

    async def choose_move(self, battle):  # type: ignore[override]
        await asyncio.sleep(3600)
        return self.choose_random_move(battle)


def start_server() -> subprocess.Popen:
    proc = subprocess.Popen(
        ["node", "pokemon-showdown", "start", str(PORT), "--no-security"],
        cwd=ROOT / "pokemon-showdown",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(120):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", PORT)) == 0:
                return proc
        time.sleep(0.5)
    os.killpg(proc.pid, signal.SIGKILL)
    raise SystemExit(f"no Showdown server on port {PORT}")


def kill_server(proc: subprocess.Popen) -> None:
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()


def player(cls=RandomPlayer):
    return cls(
        account_configuration=fresh_local_account(),
        server_configuration=SERVER,
        battle_format=FORMAT,
        log_level=50,  # the dropped socket's traceback is the expected part
        max_concurrent_battles=1,
    )


async def idle_case() -> tuple[str | None, float, bool, bool]:
    """Also runs the old code path (a bare accept_challenges) as a control: it
    should still be waiting after the kill."""
    proc = start_server()
    agent, control = player(), player()
    await agent.ps_client.wait_for_login(wait_for=30)
    await control.ps_client.wait_for_login(wait_for=30)
    waiting = asyncio.ensure_future(
        play_until_disconnect(
            agent, agent.accept_challenges(opponent=None, n_challenges=1), poll=0.5
        )
    )
    bare = asyncio.ensure_future(
        control.accept_challenges(opponent=None, n_challenges=1)
    )
    await asyncio.sleep(2)
    was_waiting = not waiting.done()
    killed = time.monotonic()
    kill_server(proc)
    lost = await asyncio.wait_for(waiting, timeout=30)
    seconds = time.monotonic() - killed
    await asyncio.sleep(3)
    old_still_waiting = not bare.done()
    bare.cancel()
    return lost, seconds, was_waiting, old_still_waiting


async def battle_case() -> tuple[str | None, float, str, bool]:
    proc = start_server()
    agent, foe = player(), player(Stalling)
    await agent.ps_client.wait_for_login(wait_for=30)
    await foe.ps_client.wait_for_login(wait_for=30)
    waiting = asyncio.ensure_future(
        play_until_disconnect(
            agent, agent.accept_challenges(opponent=None, n_challenges=1), poll=0.5
        )
    )
    challenge = asyncio.ensure_future(foe.send_challenges(agent.username, 1))
    for _ in range(120):
        if any(b.turn >= 1 for b in agent.battles.values()):
            break
        await asyncio.sleep(0.25)
    room = next(iter(agent.battles), "")
    was_waiting = not waiting.done()
    killed = time.monotonic()
    kill_server(proc)
    lost = await asyncio.wait_for(waiting, timeout=30)
    challenge.cancel()
    return lost, time.monotonic() - killed, room, was_waiting


async def main() -> None:
    lost, seconds, was_waiting, old_waiting = await idle_case()
    ok_idle = lost == "" and seconds < 10 and was_waiting and old_waiting
    print(
        f"waiting for a challenge: still waiting before the kill: {was_waiting}; "
        f"returned {lost!r} {seconds:.1f}s after it; the old bare wait still "
        f"waiting 3s later: {old_waiting}"
    )
    lost, seconds, room, was_waiting = await battle_case()
    ok_battle = bool(room) and lost == room and seconds < 10 and was_waiting
    print(
        f"mid-battle ({room or 'no battle started'}): still waiting before the "
        f"kill: {was_waiting}; returned {lost!r} {seconds:.1f}s after it"
    )
    print("PASS" if ok_idle and ok_battle else "FAIL")
    sys.exit(0 if ok_idle and ok_battle else 1)


if __name__ == "__main__":
    asyncio.run(main())
