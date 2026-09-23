from types import SimpleNamespace

import pytest
from poke_env.battle import Move, Pokemon

from datagen.build_joint_sets import parse_showteam
from vgc_bench.src.policy_player import PolicyPlayer
from vgc_bench.src.set_priors import load_set_priors, prior_reg

MC = "gen9championsvgc2026regmc"


@pytest.fixture(autouse=True)
def priors(monkeypatch):
    monkeypatch.delenv("VGC_SET_PRIOR_REG", raising=False)
    monkeypatch.setattr(PolicyPlayer, "use_moveset_prior", True)
    monkeypatch.setattr(PolicyPlayer, "use_knowledge_obs", True)
    monkeypatch.setattr(PolicyPlayer, "_prior_cache", None)


def test_current_format_fills_rillaboom_and_cache_does_not_cross_formats():
    mon = Pokemon(gen=9, species="rillaboom")
    assert PolicyPlayer._resolved_moves(mon, True, "gen9championsvgc2026regmb") == []
    predicted = {m.id for m in PolicyPlayer._resolved_moves(mon, True, MC)}
    assert "fakeout" in predicted
    assert predicted & {"grassyglide", "woodhammer", "drumbeating"}
    assert not mon.moves  # estimates never become observed facts
    assert prior_reg(MC + "bo3") == "mc"


def test_override_permits_reproducible_old_information_control(monkeypatch):
    monkeypatch.setenv("VGC_SET_PRIOR_REG", "mb")
    assert "rillaboom" not in load_set_priors(MC)[0]


def test_unknown_and_guessed_threats_are_distinguishable():
    ours = Pokemon(gen=9, species="blastoise")
    foe = Pokemon(gen=9, species="rillaboom")
    battle = SimpleNamespace(
        format=MC, active_pokemon=[ours, None], opponent_active_pokemon=[foe, None]
    )
    assert PolicyPlayer._threat_evidence(battle, ours, True) == [0, 1, 0, 0]
    battle.format = "gen9championsvgc2026regmb"
    assert PolicyPlayer._threat_evidence(battle, ours, True) == [0, 0, 0, 0]
    foe._moves["fakeout"] = Move("fakeout", gen=9)
    assert PolicyPlayer._threat_evidence(battle, ours, True) == [0.25, 0, 0, 0]


def test_disproved_set_cannot_reappear_with_high_confidence(monkeypatch):
    monkeypatch.setattr(
        PolicyPlayer,
        "_prior_cache",
        {
            "joint": {
                "rillaboom": {
                    "sets": [
                        {
                            "moves": ["fakeout"],
                            "item": "assaultvest",
                            "ability": "grassysurge",
                            "prob": 1.0,
                        }
                    ]
                }
            },
            "marginal": {},
        },
    )
    mon = Pokemon(gen=9, species="rillaboom")
    mon._moves["protect"] = Move("protect", gen=9)
    assert PolicyPlayer._moveset_prior(mon, MC) is None
    assert [m.id for m in PolicyPlayer._resolved_moves(mon, True, MC)] == ["protect"]


def test_packed_sheet_uses_species_not_nickname():
    rows = list(
        parse_showteam(
            "|showteam|p1|Drummer|rillaboom|assaultvest|grassysurge|fakeout,woodhammer|Adamant"
        )
    )
    assert rows[0][0] == "rillaboom"
