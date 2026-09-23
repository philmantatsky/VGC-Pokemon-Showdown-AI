import asyncio
from types import SimpleNamespace

import pytest

from evaluation.eval_counterfactual import DelayedSimpleHeuristicsPlayer
from evaluation.opening_study import StudyHeuristic, fresh_local_account
from evaluation.review_t6_audits import scan_battle
from evaluation.run_t6_confirmation import paired_roster_summary


def rows(value):
    return [
        {"opponent": team, "hidden_sheets": hidden, "target": value}
        for team in ("a", "b", "c")
        for hidden in (True, False)
    ]


def test_identical_scores_are_zero_not_a_gain():
    report = paired_roster_summary(rows(1), rows(1))
    assert report["overall"]["roster_mean_delta"] == 0
    assert report["hidden"]["roster_bootstrap_95_ci"] == [0, 0]
    assert report["open"]["games_per_model"] == 3


def test_rejects_missing_matchups():
    with pytest.raises(ValueError, match="unmatched"):
        paired_roster_summary(rows(1), rows(1)[:-1])


def test_roster_delta_sign():
    assert paired_roster_summary(rows(1), rows(0))["overall"]["roster_mean_delta"] == -1


def test_fingerprint_captured_before_battle_reveals_shrink_roster(monkeypatch):
    mon = SimpleNamespace(base_species="charizard")
    battle = SimpleNamespace(
        battle_tag="test", team={"a": mon}, opponent_team={"b": mon}
    )

    async def choose(self, battle):
        battle.opponent_team.clear()
        return "/team 1234"

    monkeypatch.setattr(DelayedSimpleHeuristicsPlayer, "teampreview", choose)
    player = object.__new__(StudyHeuristic)
    asyncio.run(player.teampreview(battle))
    assert player.preview_fingerprints["test"] == ("charizard", "__vs__", "charizard")


def test_review_signals_preserve_context():
    record = {
        "our_role": "p1",
        "protocol": [
            ["", "turn", "1"],
            ["", "move", "p1a: Charizard", "Protect", "p1a: Charizard"],
            ["", "move", "p1b: Torkoal", "Protect", "p1b: Torkoal"],
            ["", "-boost", "p2a: Blastoise", "spa", "2"],
        ],
    }
    flags = scan_battle(record)
    assert flags[0]["signal"] == "double_protect_with_enemy_setup"
    assert flags[0]["turn"] == 1


def test_review_accepts_public_hp_color_suffix():
    record = {
        "our_role": "p1",
        "protocol": [
            ["", "turn", "1"],
            ["", "-heal", "p2b: Sneasler", "50/100g", "[from] Grassy Terrain"],
            ["", "-damage", "p1a: Torkoal", "30/100y"],
            ["", "move", "p1a: Torkoal", "Eruption", "p2b: Sneasler"],
        ],
    }
    assert scan_battle(record)[0]["hp"] == 0.3


def test_local_accounts_do_not_rejoin_interrupted_runs():
    a, b = fresh_local_account(), fresh_local_account()
    assert a.username != b.username
    assert len(a.username) <= 18
    assert a.password is None
