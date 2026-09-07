"""dominated_weather_ball_weather (2026-09-06): Ice Weather Ball into Abomasnow
under snow while Heat Wave was 4x. G12 covers only the no-weather case; this
opt-in guard covers active non-sun weather through the calculator."""

from types import SimpleNamespace as NS

import pytest
from poke_env.battle import Move, Pokemon, Weather

from vgc_bench.src import guards as G

MOVES = {
    0: ["earthquake", "dragonclaw", "rocktomb", "protect"],
    1: ["heatwave", "weatherball", "solarbeam", "protect"],
}


def _decode(_battle, action, pos):
    action = int(action)
    if action < 7:
        return NS(order=None, move_target=0)
    band, offset = (action - 7) // 5, (action - 7) % 5
    return NS(order=Move(MOVES[pos][band % 4], gen=9), move_target=offset - 2)


def _action(pos, move_id, target):
    return 7 + 5 * MOVES[pos].index(move_id) + (target + 2)


@pytest.fixture
def snow(monkeypatch):
    abomasnow, talonflame = (
        Pokemon(gen=9, species="abomasnow"),
        Pokemon(gen=9, species="talonflame"),
    )
    battle = NS(
        active_pokemon=[
            Pokemon(gen=9, species="garchomp"),
            Pokemon(gen=9, species="charizardmegay"),
        ],
        opponent_active_pokemon=[talonflame, abomasnow],
        weather={Weather.SNOWSCAPE: 1},
        fields={},
    )
    monkeypatch.setattr(G, "_decode", _decode)
    monkeypatch.setattr(G, "_available_move", lambda b, a, p, mid: Move(mid, gen=9))
    monkeypatch.setattr(G, "_known_move", lambda foe, mid: None)
    values = {"weatherball": 0.38, "heatwave": 0.62}
    monkeypatch.setattr(G, "_expected_damage_value", lambda b, a, d, m: values[m.id])
    return battle


def test_ice_weather_ball_into_abomasnow_is_demoted_under_snow(snow):
    top = G.Candidate((_action(0, "dragonclaw", 2), _action(1, "weatherball", 2)), 0.55)
    heat = G.Candidate((_action(0, "dragonclaw", 2), _action(1, "heatwave", 0)), 0.09)
    report = G.GuardReport()
    out = G.guard_dominated_weather_ball_weather(snow, [top, heat], report)
    assert out[0] is heat and top.demoted_by == "dominated_weather_ball_weather"
    assert report.demotions["dominated_weather_ball_weather"] == 1


def test_sun_is_left_to_the_existing_rules(snow):
    snow.weather = {Weather.SUNNYDAY: 1}
    top = G.Candidate((_action(0, "dragonclaw", 2), _action(1, "weatherball", 2)), 0.55)
    heat = G.Candidate((_action(0, "dragonclaw", 2), _action(1, "heatwave", 0)), 0.09)
    out = G.guard_dominated_weather_ball_weather(snow, [top, heat], G.GuardReport())
    assert out[0] is top and top.demoted_by is None


def test_close_comparison_stands_down(snow, monkeypatch):
    monkeypatch.setattr(
        G,
        "_expected_damage_value",
        lambda b, a, d, m: {"weatherball": 0.5, "heatwave": 0.6}[m.id],
    )
    top = G.Candidate((_action(0, "dragonclaw", 2), _action(1, "weatherball", 2)), 0.55)
    heat = G.Candidate((_action(0, "dragonclaw", 2), _action(1, "heatwave", 0)), 0.09)
    out = G.guard_dominated_weather_ball_weather(snow, [top, heat], G.GuardReport())
    assert out[0] is top


def test_registered_but_opt_in():
    assert "dominated_weather_ball_weather" in G.GUARDS
    assert "dominated_weather_ball_weather" not in G.HARD_GUARDS
