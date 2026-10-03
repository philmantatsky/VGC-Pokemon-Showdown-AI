"""Live protocol check of the late-sheet re-plan (2026-10-03, PolicyPlayer
sheet_preview): our player (deployed brain, learned preview, sheet_preview on, a 3 s
sheet wait) against a heuristic opponent that accepts open team sheets 10 s late and,
like a person reading our sheet, picks its team 5 s after the sheets appear. The
re-plan is forced to a plan with different leads, so the first two Pokemon we send
out show which /team the server used. Result 2026-10-03: two games, both led with the
re-planned pair (Farigiraf + Incineroar, not Blastoise + Farigiraf). Against an
opponent that picks instantly the re-plan arrives too late and the first plan stands.

Usage (repo root, a server on port 7611):
  .venv/bin/python checks/late_sheet_live.py <decision log> teams/reg_mc/MC2610.txt
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from poke_env.ps_client import ServerConfiguration

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.opening_study import (  # noqa: E402
    StudyHeuristic,
    StudyPlayer,
    fresh_local_account,
)
from vgc_bench.src import sheet_preview  # noqa: E402
from vgc_bench.src.policy_player import PolicyPlayer  # noqa: E402
from vgc_bench.src.teams import RandomTeamBuilder  # noqa: E402
from vgc_bench.src.utils import format_map  # noqa: E402

LOG = Path(sys.argv[1])
OPPONENT_TEAM = Path(sys.argv[2])
sheet_preview.WEIGHT = 50.0  # make the late sheet decide the plan
import vgc_bench.src.policy_player as _pp  # noqa: E402

_real = _pp.sheet_rerank


def _forced(battle, our_plans, their_plans):
    """Protocol test only: the re-plan always takes the model's second plan."""
    plan, audit = _real(battle, our_plans, their_plans)
    other = [
        q for q in our_plans if set(q.lead_indices) != set(our_plans[0].lead_indices)
    ]
    if other:
        plan, audit["chosen_rank"] = other[0], our_plans.index(other[0]) + 1
    return plan, audit


_pp.sheet_rerank = _forced
server = ServerConfiguration(
    "ws://localhost:7611/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)
common = dict(
    server_configuration=server,
    battle_format=format_map["mc"],
    log_level=40,
    max_concurrent_battles=1,
    accept_open_team_sheet=True,
    open_timeout=None,
)
PolicyPlayer.use_knowledge_obs = PolicyPlayer.use_moveset_prior = True
ours = StudyPlayer(
    account_configuration=fresh_local_account(),
    deterministic=True,
    team=RandomTeamBuilder(1, 1, "mc", [Path("teams/candidates_mc/T6.txt")]),
    team_sheet_wait_timeout=3.0,
    **common,
)
ours.set_policy(Path("results_deployed/champion_mc_T6tac.zip"), torch.device("cpu"))
ours.preview_model_path = Path("data/preview_t6_focus_20260923.pt")
ours.use_learned_teampreview = True
ours.sheet_preview = True
ours.decision_log_path = LOG
foe = StudyHeuristic(
    account_configuration=fresh_local_account(),
    team=RandomTeamBuilder(1, 1, "mc", [OPPONENT_TEAM]),
    **common,
)
send = foe.ps_client.send_message


async def late_accept(message, room="", message_2=None):
    if message.startswith("/team"):  # a human reads our sheet before choosing

        async def choose_later():
            await asyncio.sleep(5)
            await send(message, room, message_2)

        asyncio.get_running_loop().create_task(choose_later())
        return
    if message == "/acceptopenteamsheets":

        async def later():
            await asyncio.sleep(10)
            await send(message, room)

        asyncio.get_running_loop().create_task(later())
        return
    await send(message, room, message_2)


foe.ps_client.send_message = late_accept  # type: ignore[method-assign]
sent_orders = []
_our_send = ours.ps_client.send_message


async def log_send(message, room="", message_2=None):
    if message.startswith("/team"):
        sent_orders.append((room[-12:], message))
    await _our_send(message, room, message_2)


ours.ps_client.send_message = log_send  # type: ignore[method-assign]
switched: dict[str, list] = {}
from poke_env.battle import DoubleBattle  # noqa: E402

_parse = DoubleBattle.parse_message


def log_parse(self, split_message):
    if (
        self.player_username == ours.username
        and len(split_message) > 3
        and split_message[1] in ("switch", "drag")
        and split_message[2].startswith(self.player_role or "?")
    ):
        seen = switched.setdefault(self.battle_tag, [])
        name = split_message[3].split(",")[0]
        if name not in seen:
            seen.append(name)
    return _parse(self, split_message)


DoubleBattle.parse_message = log_parse  # type: ignore[method-assign]
asyncio.run(ours.battle_against(foe, n_battles=2))
print("finished", ours.n_finished_battles, "won", ours.n_won_battles)
print("orders sent:", sent_orders)
for tag, seen in switched.items():
    print("battle", tag[-12:], "leads (first two in):", seen[:2], "then", seen[2:])
counts = Counter(
    {
        k: v
        for k, v in PolicyPlayer.guard_fire_counts.items()
        if "sheet" in k or "preview" in k
    }
)
print(dict(counts))
rows = (
    [json.loads(x) for x in LOG.read_text().splitlines() if x.strip()]
    if LOG.exists()
    else []
)
print("sheet audits:", sum("sheet_preview" in r for r in rows))
