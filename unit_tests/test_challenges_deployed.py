import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ladder_ourteam import _rejoin_active_battle_in_loop

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "challenges_deployed.sh"


def test_challenge_launcher_is_reg_mc_only_and_unranked():
    text = SCRIPT.read_text()
    assert "gen9championsvgc2026regmc" in text
    assert "--reg mc" in text
    assert "--challenges" in text
    assert "--n_games" in text
    assert "ladder_deployed.sh" not in text


def test_challenge_launcher_pins_both_deployed_artifacts():
    text = SCRIPT.read_text()
    assert "results_deployed/DEPLOYED.json" in text
    # checkpoint, team (and a deployed preview model) are sha256-verified by the
    # shared helper (unit_tests/test_deployed_config.py); failure refuses the launch
    assert 'tools/deployed_config.py --manifest "$MANIFEST"' in text
    assert "CHALLENGE_REFUSED the deployed configuration failed verification" in text
    assert "SHOWDOWN_PASSWORD" not in text


class FakeClient:
    def __init__(self):
        self.logged_in = asyncio.Event()
        self.logged_in.set()
        self.messages = []

    async def send_message(self, message):
        self.messages.append(message)


def test_rejoin_reconstructs_existing_mc_room():
    async def scenario():
        agent: Any = SimpleNamespace(
            ps_client=FakeClient(),
            _battle_semaphore=asyncio.Semaphore(1),
            _battle_start_condition=asyncio.Condition(),
            _battles={},
        )
        room = "battle-gen9championsvgc2026regmc-123-abcdef"
        task = asyncio.create_task(
            _rejoin_active_battle_in_loop(agent, room, timeout=1)
        )
        await asyncio.sleep(0)
        async with agent._battle_start_condition:
            agent._battles[room] = object()
            agent._battle_semaphore.release()
            agent._battle_start_condition.notify_all()
        assert await task is True
        assert agent.ps_client.messages == [f"/join {room}"]
        assert not agent._battle_semaphore.locked()

    asyncio.run(scenario())


def test_rejoin_rejects_other_formats():
    agent: Any = SimpleNamespace()
    with pytest.raises(ValueError, match="Reg M-C"):
        asyncio.run(_rejoin_active_battle_in_loop(agent, "battle-gen9randombattle-1"))
