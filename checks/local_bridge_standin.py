"""A stand-in for the external client of ``evaluation/local_bridge.py``: it logs into
the local server under a name of its own and either challenges the bridge's bot or
accepts the bot's challenges -- all the bridge asks of a client. For rehearsing the
harness before the real client exists, and as the pattern for one.

Two players: ``deployed`` (default) the deployed bot without search, on the deployed
team unless ``--team`` names another file; ``heuristic`` poke-env's simple heuristics
on ``--team`` (default a pool roster). A separate process, so the two bots share no
class-wide switch.

Usage (repo root, VGC_SET_PRIOR_REG exported as the launchers do):
  .venv/bin/python checks/local_bridge_standin.py <port> <own name> <bot name> \
      <challenge|accept> <games> <open|closed> [--team FILE] \
      [--player deployed|heuristic]
As the bridge's ``--client`` template:
  ".venv/bin/python checks/local_bridge_standin.py {port} {opponent} {bot} {mode}
   {games} {sheets} --team '{team}'"
(an empty ``--team`` is the player's own default).
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_ROSTER = ROOT / "teams" / "reg_mc" / "MC2566.txt"


def parse(args: list[str]) -> dict:
    """The command line as a dict; raises ValueError with what is wrong."""
    args = list(args)
    options = {"--team": "", "--player": "deployed"}
    for flag in options:
        if flag in args:
            at = args.index(flag)
            options[flag] = args[at + 1]
            del args[at : at + 2]
    if len(args) != 6:
        raise ValueError("six arguments: port, own name, bot name, mode, games, sheets")
    port, own, bot, mode, games, sheets = args
    if mode not in ("challenge", "accept"):
        raise ValueError(f"mode {mode!r} is not challenge or accept")
    if sheets not in ("open", "closed"):
        raise ValueError(f"sheets {sheets!r} is not open or closed")
    if options["--player"] not in ("deployed", "heuristic"):
        raise ValueError(f"player {options['--player']!r} is not deployed or heuristic")
    return {
        "port": int(port),
        "own": own,
        "bot": bot,
        "mode": mode,
        "games": int(games),
        "sheets": sheets,
        "team": options["--team"],
        "player": options["--player"],
    }


async def play(spec: dict) -> None:
    import torch
    from poke_env import AccountConfiguration
    from poke_env.ps_client import ServerConfiguration

    from tools.deployed_config import resolve
    from vgc_bench.src.teams import RandomTeamBuilder
    from vgc_bench.src.utils import format_map

    server = ServerConfiguration(
        f"ws://localhost:{spec['port']}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )
    common: dict[str, Any] = dict(
        account_configuration=AccountConfiguration(spec["own"], None),
        accept_open_team_sheet=spec["sheets"] == "open",
        server_configuration=server,
        battle_format=format_map["mc"],
        log_level=40,
        max_concurrent_battles=1,
        open_timeout=None,
    )
    if spec["player"] == "heuristic":
        from evaluation.opening_study import StudyHeuristic

        team = Path(spec["team"]) if spec["team"] else DEFAULT_ROSTER
        foe = StudyHeuristic(team=RandomTeamBuilder(1, 1, "mc", [team]), **common)
    else:
        from vgc_bench.src.guards import GUARDS, HARD_GUARDS
        from vgc_bench.src.policy_player import PolicyPlayer

        deployed = resolve()
        PolicyPlayer.use_knowledge_obs = PolicyPlayer.use_moveset_prior = True
        PolicyPlayer.mask_immunities = True
        PolicyPlayer.use_knowledge_guards = True
        PolicyPlayer.guard_flags = {g: g in HARD_GUARDS for g in GUARDS}
        PolicyPlayer.use_search = False
        team = Path(spec["team"]) if spec["team"] else Path(deployed["TEAM"])
        foe = PolicyPlayer(
            deterministic=True,
            team=RandomTeamBuilder(1, 1, "mc", [team]),
            preview_model_path=Path(deployed["PREVIEW_MODEL"]),
            use_learned_teampreview=True,
            sticky_guard_corrections=bool(deployed["STICKY"]),
            guard_overrides={g: True for g in deployed["GUARDS"].split(",") if g},
            team_sheet_wait_timeout=3.0,
            **common,
        )
        foe.set_policy(Path(deployed["CKPT"]), torch.device("cpu"))
    await foe.ps_client.wait_for_login(wait_for=60)
    print(
        f"STANDIN_READY {foe.username} {spec['player']} on {team.name} "
        f"{spec['mode']} {spec['games']} sheets {spec['sheets']}",
        flush=True,
    )
    started = time.monotonic()
    if spec["mode"] == "challenge":
        await foe.send_challenges(spec["bot"], spec["games"])
    else:
        await foe.accept_challenges(spec["bot"], spec["games"])
    print(
        f"STANDIN_DONE won {foe.n_won_battles} of {foe.n_finished_battles} "
        f"in {time.monotonic() - started:.0f} s",
        flush=True,
    )


def main() -> int:
    try:
        spec = parse(sys.argv[1:])
    except (ValueError, IndexError) as error:
        print(f"STANDIN_REFUSED {error}\n{__doc__}")
        return 2
    asyncio.run(play(spec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
