"""PolicyPlayer sheet_preview (2026-10-03): with their open sheet the learned preview
chooses among the model's top plans by the sheet (sheet_preview.py), and a sheet
that arrives after our preview went out re-plans and resends a changed order."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from unit_tests.test_sheet_preview import PLANS, THEIR_LEAD, USUAL, _preview
from vgc_bench.src.policy_player import PolicyPlayer

NO_FAKE_OUT = USUAL.replace(
    "- Fake Out\n- Flare Blitz", "- Will-O-Wisp\n- Flare Blitz"
).replace("- Fake Out\n- Grassy Glide", "- Protect\n- Grassy Glide")
TAG = "battle-gen9championsvgc2026regmc-fixture"


class _Model:
    """Our plans: Blastoise + Farigiraf (0.40) over Charizard + Venusaur (0.35);
    theirs: Incineroar + Rillaboom lead."""

    def predict_plans(self, roster, other, top_k):
        return list(THEIR_LEAD if roster[0] == "incineroar" else PLANS)[:top_k]


class _Client:
    def __init__(self):
        self.sent: list[tuple[str, str]] = []

    async def send_message(self, message, room=""):
        self.sent.append((message, room))


def _player(tmp_path: Path) -> Any:
    player: Any = PolicyPlayer.__new__(PolicyPlayer)
    player.playbook_path = None
    player.playbook_script_only = False
    player._playbook = None
    player.preview_model_path = Path("model.pt")
    player._preview_predictor = _Model()
    player.use_learned_teampreview = True
    player.sheet_preview = True
    player._battle_plans = {}
    player._preview_orders = {}
    player._preview_sheet_seen = {}
    player._preview_requests_submitted = set()
    player.decision_log_path = tmp_path / "decisions.jsonl"
    player.teampreview = player._learned_teampreview  # the deployed chain's step
    return player


@pytest.fixture(autouse=True)
def _counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PolicyPlayer, "guard_fire_counts", Counter())


def test_their_open_sheet_moves_the_plan_and_is_logged(tmp_path: Path) -> None:
    player = _player(tmp_path)
    battle = _preview(NO_FAKE_OUT)
    order = player._learned_teampreview(battle)
    assert order == "/team 3425"  # Charizard + Venusaur, back Farigiraf + Torkoal
    assert player._preview_sheet_seen[battle.battle_tag] is True
    row = json.loads((tmp_path / "decisions.jsonl").read_text().splitlines()[-1])
    assert row["sheet_preview"]["chosen_rank"] == 2
    assert "fakeout" not in row["their_sheet"]["incineroar"]["moves"]
    assert PolicyPlayer.guard_fire_counts["sheet_preview:changed"] == 1


def test_without_a_sheet_the_model_plan_stands(tmp_path: Path) -> None:
    player = _player(tmp_path)
    battle = _preview(NO_FAKE_OUT, shown=False)
    assert player._learned_teampreview(battle) == "/team 1265"
    assert player._preview_sheet_seen[battle.battle_tag] is False
    assert PolicyPlayer.guard_fire_counts["sheet_preview"] == 0


def _show_sheet(battle, sheet: str) -> None:
    from poke_env.teambuilder import Teambuilder

    for tb in Teambuilder.parse_showdown_team(sheet):
        species = tb.species or tb.nickname
        battle.opponent_team[f"p2: {species}"]._update_from_teambuilder(tb)


def _late_case(tmp_path: Path, sheet: str):
    player = _player(tmp_path)
    battle = _preview(sheet, shown=False)
    battle._teampreview = True
    battle._last_request = {"rqid": 3, "teamPreview": True}
    player._battles = {TAG: battle}
    player.ps_client = _Client()
    first = player._learned_teampreview(battle)
    player._preview_requests_submitted.add((TAG, 3))
    _show_sheet(battle, sheet)
    asyncio.run(player._replan_after_late_sheet(TAG))
    return player, first


def test_a_late_sheet_that_changes_the_plan_resends_it(tmp_path: Path) -> None:
    player, first = _late_case(tmp_path, NO_FAKE_OUT)
    assert first == "/team 1265"
    assert player.ps_client.sent == [("/team 3425", TAG)]
    assert PolicyPlayer.guard_fire_counts["late_sheet_replan:changed"] == 1


def test_a_late_sheet_of_usual_sets_sends_nothing(tmp_path: Path) -> None:
    player, _ = _late_case(tmp_path, USUAL)
    assert player.ps_client.sent == []
    assert PolicyPlayer.guard_fire_counts["late_sheet_replan:same"] == 1


def test_no_replan_when_the_preview_already_saw_the_sheet(tmp_path: Path) -> None:
    player = _player(tmp_path)
    battle = _preview(NO_FAKE_OUT)
    battle._teampreview = True
    battle._last_request = {"rqid": 3, "teamPreview": True}
    player._battles = {TAG: battle}
    player.ps_client = _Client()
    player._learned_teampreview(battle)
    player._preview_requests_submitted.add((TAG, 3))
    asyncio.run(player._replan_after_late_sheet(TAG))
    assert player.ps_client.sent == []
