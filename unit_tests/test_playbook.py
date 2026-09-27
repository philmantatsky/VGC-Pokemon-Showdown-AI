"""The playbook planner (vgc_bench/src/playbook.py): OUR plan at team preview,
picked by rules over opponent features, with its reasons (the user, 2026-09-27:
"our team picker needs to come up with a plan of action as to why its choosing
its specific 4 and why it wants its first two")."""

from __future__ import annotations

import json

import pytest

from vgc_bench.src.opponent_preview import plan_to_showdown_order
from vgc_bench.src.playbook import OpenSet, Playbook, opponent_features, rule_matches

OURS = ["blastoise", "farigiraf", "charizard", "venusaur", "torkoal", "incineroar"]


@pytest.fixture(scope="module")
def playbook():
    return Playbook("data/playbook_t6.json")


def test_sand_and_wide_guard_are_read_from_the_set_data():
    feats = opponent_features(
        ["tyranitar", "pelipper", "garchomp", "incineroar", "amoonguss", "gholdengo"]
    )
    assert feats["sand_setter"] and feats["sand_setter_by"] == ["tyranitar"]
    assert feats["rain_setter"] and "pelipper" in feats["rain_setter_by"]
    assert feats["wide_guard"]
    assert feats["fake_out_users"] >= 1


def test_a_clean_fire_weak_team_gets_sun_room(playbook):
    choice = playbook.choose(
        OURS, ["kingambit", "amoonguss", "scizor", "sinistcha", "garchomp", "dragonite"]
    )
    assert choice.card == "sun_room"
    assert choice.lead == ("farigiraf", "torkoal")
    assert any("fire weak" in r for r in choice.reasons)


def test_sand_keeps_torkoal_out_of_the_lead(playbook):
    """Ladder: Farigiraf + Torkoal went 1-8 against sand."""
    choice = playbook.choose(
        OURS,
        [
            "tyranitar",
            "excadrill",
            "indeedeef",
            "corviknight",
            "gholdengo",
            "sinistcha",
        ],
    )
    assert "torkoal" not in choice.lead
    assert "venusaur" in choice.back  # the sand back line
    assert any("sand" in r for r in choice.reasons)


def test_the_default_card_and_its_order(playbook):
    choice = playbook.choose(
        OURS,
        [
            "pelipper",
            "archaludon",
            "basculegion",
            "whimsicott",
            "gholdengo",
            "maushold",
        ],
    )
    assert choice.card == "water_room"
    assert choice.lead == ("blastoise", "farigiraf")
    order = plan_to_showdown_order(choice.plan)
    assert [OURS[i - 1] for i in order[:2]] == ["blastoise", "farigiraf"]
    assert sorted(order) == sorted(i + 1 for i in choice.plan.bring_indices)


def test_heavy_physical_teams_get_support_room(playbook):
    choice = playbook.choose(
        OURS,
        ["rillaboom", "incineroar", "sneasler", "kingambit", "tyranitar", "pelipper"],
    )
    assert choice.card == "support_room"
    assert choice.lead == ("incineroar", "farigiraf")


def test_experimental_cards_wait_for_their_own_test(playbook):
    their_tr = [
        "hatterene",
        "indeedeef",
        "ursaluna",
        "farigiraf",
        "amoonguss",
        "torkoal",
    ]
    assert playbook.choose(OURS, their_tr).card != "fast_sun"
    trial = Playbook("data/playbook_t6.json", allow_experimental=True)
    assert trial.choose(OURS, their_tr).card == "fast_sun"


def test_an_open_sheet_beats_the_usage_guess():
    guess = opponent_features(["pelipper"])
    shown = opponent_features(
        ["pelipper"],
        {"pelipper": OpenSet("drizzle", "sitrusberry", ("hurricane", "protect"))},
    )
    assert guess["wide_guard"] and not shown["wide_guard"]
    assert shown["rain_setter"]


def test_rules_and_the_default_card(tmp_path):
    feats = {"sand_setter": True, "physical_threats": 3}
    assert rule_matches({"all": ["sand_setter", "physical_threats>=3"]}, feats)
    assert not rule_matches({"none": ["sand_setter"]}, feats)
    assert rule_matches({"any": ["rain_setter", "physical_threats>=2"]}, feats)
    bad = tmp_path / "playbook.json"
    bad.write_text(
        json.dumps(
            {
                "cards": [
                    {
                        "name": "x",
                        "lead": ["a", "b"],
                        "back": ["c", "d"],
                        "when": {"all": ["sand_setter"]},
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="unconditional default"):
        Playbook(bad)
