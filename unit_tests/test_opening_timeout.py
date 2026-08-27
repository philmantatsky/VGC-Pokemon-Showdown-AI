import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from poke_env.player import Player

from vgc_bench.src.policy_player import PolicyPlayer


def _player() -> PolicyPlayer:
    player = PolicyPlayer.__new__(PolicyPlayer)
    player.team_sheet_wait_timeout = 0
    player._preview_requests_submitted = set()
    return player


def _battle():
    return SimpleNamespace(
        battle_tag="battle-test-1", last_request={"rqid": 7}, teampreview=True
    )


def test_team_sheet_timeout_submits_preview_once():
    player = _player()
    battle = _battle()
    key = player._preview_request_key(battle)
    submit = AsyncMock()
    player._handle_battle_request = submit

    asyncio.run(player._team_sheet_wait_fallback(battle, key))

    submit.assert_awaited_once_with(battle)


def test_team_sheet_timeout_stands_down_after_ots_reply():
    player = _player()
    battle = _battle()
    key = player._preview_request_key(battle)
    player._preview_requests_submitted.add(key)
    submit = AsyncMock()
    player._handle_battle_request = submit

    asyncio.run(player._team_sheet_wait_fallback(battle, key))

    submit.assert_not_awaited()


def test_ponder_starts_only_after_parent_submission_returns(monkeypatch):
    events = []

    async def submitted(_self, _battle, _maybe_default_order=False):
        events.append("submitted")

    class Session:
        def start_pending_ponder(self):
            events.append("ponder_started")

    monkeypatch.setattr(Player, "_handle_battle_request", submitted)
    player = _player()
    player._exact_sessions = {"battle-test-1": Session()}
    battle = SimpleNamespace(
        battle_tag="battle-test-1",
        last_request={"rqid": 8},
        teampreview=False,
    )

    asyncio.run(PolicyPlayer._handle_battle_request(player, battle))

    assert events == ["submitted", "ponder_started"]
