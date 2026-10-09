"""The deployed bot (no search) as the opponent of a ladder rehearsal: it challenges
the rehearsing ladder script (checks/ladder_rehearsal.py ... "external:<n>") by name
on a local server, once per entry of the spec, with closed or open team sheets. A
separate process, so the two bots share no class-wide switch.

Usage (repo root, VGC_SET_PRIOR_REG exported as the launchers do):
  .venv/bin/python checks/ladder_rehearsal_challenger.py <port> <bot name> closed,open
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poke_env.ps_client import ServerConfiguration  # noqa: E402

from evaluation.opening_study import fresh_local_account  # noqa: E402
from tools.deployed_config import resolve  # noqa: E402
from vgc_bench.src.guards import GUARDS, HARD_GUARDS  # noqa: E402
from vgc_bench.src.policy_player import PolicyPlayer  # noqa: E402
from vgc_bench.src.teams import RandomTeamBuilder  # noqa: E402
from vgc_bench.src.utils import format_map  # noqa: E402

port, name, spec = int(sys.argv[1]), sys.argv[2], sys.argv[3].split(",")
deployed = resolve()
SERVER = ServerConfiguration(
    f"ws://localhost:{port}/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)
PolicyPlayer.use_knowledge_obs = PolicyPlayer.use_moveset_prior = True
PolicyPlayer.mask_immunities = True
PolicyPlayer.use_knowledge_guards = True
PolicyPlayer.guard_flags = {g: g in HARD_GUARDS for g in GUARDS}
PolicyPlayer.use_search = False


async def main() -> None:
    for sheets in spec:
        foe = PolicyPlayer(
            account_configuration=fresh_local_account(),
            deterministic=True,
            accept_open_team_sheet=sheets == "open",
            team=RandomTeamBuilder(1, 1, "mc", [Path(deployed["TEAM"])]),
            preview_model_path=Path(deployed["PREVIEW_MODEL"]),
            use_learned_teampreview=True,
            sticky_guard_corrections=bool(deployed["STICKY"]),
            guard_overrides={g: True for g in deployed["GUARDS"].split(",") if g},
            team_sheet_wait_timeout=3.0,
            server_configuration=SERVER,
            battle_format=format_map["mc"],
            log_level=40,
            max_concurrent_battles=1,
            open_timeout=None,
        )
        foe.set_policy(Path(deployed["CKPT"]), torch.device("cpu"))
        started = time.monotonic()
        await foe.send_challenges(name, 1)
        print(
            f"CHALLENGER_GAME sheets={sheets} seconds={time.monotonic() - started:.0f} "
            f"challenger_won={foe.n_won_battles}",
            flush=True,
        )
    print("CHALLENGER_DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
