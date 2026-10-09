"""Dress rehearsal of a ladder session: ladder_ourteam.main() itself -- the script
tools/ladder_read_loop.sh launches, with the argument list it would pass -- against a
LOCAL Showdown server in challenge mode.

Why (2026-10-09): the ladder launcher's search flags had been built and unit-tested
but never played. The first rehearsal, with the null search, changed 1 decision of 21
(a search world created while a slot held a fainted Pokemon) -- found here instead of
in rated games. The ladder script's path differs from the local harnesses' in ways no
unit test pins: its own argument parsing, the reranker models, the open-sheet preview,
the shadow forecast, the timer.

Nothing here touches the real server or the real account: the name is a throwaway
local guest (PF_NAME to choose it) and no password is set or read.

Usage (repo root; tools/ladder_rehearse.sh builds the argument list):
  .venv/bin/python checks/ladder_rehearsal.py <port> <games> -- <ladder_ourteam.py args>
<games>: "closed:MC2610,open:MC2566" -- a heuristic opponent on those rosters of
teams/reg_mc, one game each, with closed or open team sheets; or "external:<n>" --
wait for <n> challenges from another process (checks/ladder_rehearsal_challenger.py:
the deployed bot itself, for longer games).
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

port = int(sys.argv[1])
external = sys.argv[2].startswith("external")
games = [] if external else [g.split(":") for g in sys.argv[2].split(",")]
split = sys.argv.index("--")
ladder_args = sys.argv[split + 1 :]

# a throwaway local guest; the real credentials are never loaded into this process
os.environ["SHOWDOWN_USERNAME"] = os.environ.get("PF_NAME") or "pf" + uuid4().hex[:10]
os.environ.pop("SHOWDOWN_PASSWORD", None)

from poke_env.ps_client import ServerConfiguration  # noqa: E402

import ladder_ourteam  # noqa: E402
from evaluation.opening_study import StudyHeuristic, fresh_local_account  # noqa: E402
from vgc_bench.src.teams import RandomTeamBuilder  # noqa: E402
from vgc_bench.src.utils import format_map  # noqa: E402

SERVER = ServerConfiguration(
    f"ws://localhost:{port}/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)
ladder_ourteam.ShowdownServerConfiguration = SERVER  # the one substitution


async def drive() -> None:
    ready = asyncio.Event()
    holder: dict = {}
    original = ladder_ourteam.play_until_disconnect

    async def announced(agent, awaitable, *args, **kwargs):
        holder["agent"] = agent
        ready.set()
        return await original(agent, awaitable, *args, **kwargs)

    ladder_ourteam.play_until_disconnect = announced
    sys.argv = ["ladder_ourteam.py", *ladder_args, "--challenges"]
    bot = asyncio.ensure_future(ladder_ourteam.main())
    waiter = asyncio.ensure_future(ready.wait())
    await asyncio.wait({bot, waiter}, return_when=asyncio.FIRST_COMPLETED)
    if bot.done():
        bot.result()
        raise SystemExit("the ladder script ended before it waited for a challenge")
    agent = holder["agent"]
    await agent.ps_client.wait_for_login(wait_for=60)
    print(f"REHEARSAL_READY {agent.username}", flush=True)
    for sheets, roster in games:
        foe = StudyHeuristic(
            account_configuration=fresh_local_account(),
            team=RandomTeamBuilder(
                1, 1, "mc", [ROOT / "teams" / "reg_mc" / f"{roster}.txt"]
            ),
            accept_open_team_sheet=sheets == "open",
            server_configuration=SERVER,
            battle_format=format_map["mc"],
            log_level=40,
            max_concurrent_battles=1,
            open_timeout=None,
        )
        started = time.monotonic()
        await foe.send_challenges(agent.username, 1)
        print(
            f"REHEARSAL_GAME sheets={sheets} roster={roster} "
            f"seconds={time.monotonic() - started:.0f} "
            f"bot_record={agent.n_won_battles}-{agent.n_lost_battles}",
            flush=True,
        )
    await bot
    print("REHEARSAL_DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(drive())
