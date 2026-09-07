"""resisted_target guard (2026-09-06): the ladder blunder class the user watched
live -- Weather Ball (Fire in sun) into Rotom-Wash beside a Meganium; Wave Crash
into Altaria beside Sneasler. Facts only, twin verified by decoding, opt-in."""

from types import SimpleNamespace as NS

import pytest
from poke_env.battle import Move, Pokemon, Weather

from vgc_bench.src import guards as G

# our slot 0 = Garchomp [earthquake, dragonclaw, rocktomb, protect]
# our slot 1 = Charizard [heatwave, weatherball, solarbeam, protect]
MOVES = {
    0: ["earthquake", "dragonclaw", "rocktomb", "protect"],
    1: ["heatwave", "weatherball", "solarbeam", "protect"],
}


def _decode(_battle, action, pos):
    action = int(action)
    if action < 7:
        return NS(order=None, move_target=0)
    band, offset = (action - 7) // 5, (action - 7) % 5
    move_id = MOVES[pos][band % 4]
    return NS(order=Move(move_id, gen=9), move_target=offset - 2)


def _action(pos: int, move_id: str, target: int) -> int:
    return 7 + 5 * MOVES[pos].index(move_id) + (target + 2)


def _battle(foes, weather=None):
    garchomp = Pokemon(gen=9, species="garchomp")
    charizard = Pokemon(gen=9, species="charizardmegay")
    return NS(
        active_pokemon=[garchomp, charizard],
        opponent_active_pokemon=list(foes),
        weather=weather or {},
        fields={},
    )


@pytest.fixture(autouse=True)
def _stub_calc(monkeypatch):
    monkeypatch.setattr(G, "_decode", _decode)
    monkeypatch.setattr(G.K, "damage_fraction", lambda *a, **k: None)
    monkeypatch.setattr(G.K, "deals_no_damage", lambda *a, **k: False)


def _run(battle, cands):
    report = G.GuardReport()
    out = G.guard_resisted_target(battle, cands, report)
    return out, report


def test_weather_ball_in_sun_retargets_from_rotom_wash_to_meganium_injected():
    rotom, meganium = (
        Pokemon(gen=9, species="rotomwash"),
        Pokemon(gen=9, species="meganium"),
    )
    battle = _battle([meganium, rotom], weather={Weather.SUNNYDAY: 1})
    top = G.Candidate((_action(0, "rocktomb", 2), _action(1, "weatherball", 2)), 0.28)
    other = G.Candidate((_action(0, "rocktomb", 1), _action(1, "weatherball", 2)), 0.27)
    out, report = _run(battle, [top, other])
    assert out[0].actions == (_action(0, "rocktomb", 2), _action(1, "weatherball", 1))
    assert out[0].prob == top.prob
    assert report.demotions["resisted_target:injected"] == 1
    assert "resisted_target" in report.stages
    assert len(out) == 3 and out[1] is top


def test_existing_twin_is_promoted_not_duplicated():
    rotom, meganium = (
        Pokemon(gen=9, species="rotomwash"),
        Pokemon(gen=9, species="meganium"),
    )
    battle = _battle([meganium, rotom], weather={Weather.SUNNYDAY: 1})
    top = G.Candidate((_action(0, "rocktomb", 2), _action(1, "weatherball", 2)), 0.28)
    twin = G.Candidate((_action(0, "rocktomb", 2), _action(1, "weatherball", 1)), 0.03)
    out, report = _run(battle, [top, twin])
    assert out[0] is twin and len(out) == 2
    assert report.demotions["resisted_target:promoted"] == 1


def test_no_weather_means_normal_weather_ball_and_no_retarget():
    rotom, meganium = (
        Pokemon(gen=9, species="rotomwash"),
        Pokemon(gen=9, species="meganium"),
    )
    battle = _battle([meganium, rotom])
    top = G.Candidate((_action(0, "rocktomb", 2), _action(1, "weatherball", 2)), 0.3)
    out, report = _run(battle, [top])
    assert out[0] is top and not report.stages


def test_wave_crash_retargets_from_altaria_to_sneasler(monkeypatch):
    MOVES[0] = ["wavecrash", "lastrespects", "aquajet", "protect"]
    try:
        altaria, sneasler = (
            Pokemon(gen=9, species="altaria"),
            Pokemon(gen=9, species="sneasler"),
        )
        battle = _battle([altaria, sneasler])
        top = G.Candidate((_action(0, "wavecrash", 1), _action(1, "protect", 0)), 0.4)
        out, report = _run(battle, [top])
        assert out[0].actions == (_action(0, "wavecrash", 2), _action(1, "protect", 0))
        assert report.demotions["resisted_target:injected"] == 1
    finally:
        MOVES[0] = ["earthquake", "dragonclaw", "rocktomb", "protect"]


def test_both_foes_resisting_leaves_the_pick_alone():
    wash, heat = (
        Pokemon(gen=9, species="rotomwash"),
        Pokemon(gen=9, species="rotomheat"),
    )
    battle = _battle([heat, wash], weather={Weather.SUNNYDAY: 1})
    top = G.Candidate((_action(0, "rocktomb", 2), _action(1, "weatherball", 2)), 0.3)
    out, report = _run(battle, [top])
    assert out[0] is top and not report.stages


def test_spread_and_status_moves_are_ignored():
    rotom, meganium = (
        Pokemon(gen=9, species="rotomwash"),
        Pokemon(gen=9, species="meganium"),
    )
    battle = _battle([meganium, rotom], weather={Weather.SUNNYDAY: 1})
    top = G.Candidate((_action(0, "earthquake", 0), _action(1, "heatwave", 0)), 0.3)
    out, report = _run(battle, [top])
    assert out[0] is top and not report.stages


def test_known_immunity_on_the_better_target_blocks_the_retarget(monkeypatch):
    rotom, meganium = (
        Pokemon(gen=9, species="rotomwash"),
        Pokemon(gen=9, species="meganium"),
    )
    battle = _battle([meganium, rotom], weather={Weather.SUNNYDAY: 1})
    monkeypatch.setattr(G.K, "deals_no_damage", lambda b, a, d, m: d is meganium)
    top = G.Candidate((_action(0, "rocktomb", 2), _action(1, "weatherball", 2)), 0.3)
    out, report = _run(battle, [top])
    assert out[0] is top
    assert report.demotions["resisted_target:other_immune"] == 1


def test_calculator_disagreement_blocks_the_retarget(monkeypatch):
    rotom, meganium = (
        Pokemon(gen=9, species="rotomwash"),
        Pokemon(gen=9, species="meganium"),
    )
    battle = _battle([meganium, rotom], weather={Weather.SUNNYDAY: 1})
    rotom._current_hp, rotom._max_hp = 100, 100  # healthy: a 0.6 roll is no KO
    monkeypatch.setattr(
        G.K,
        "damage_fraction",
        lambda b, a, d, m: (0.6, 0.7) if d is rotom else (0.3, 0.4),
    )
    top = G.Candidate((_action(0, "rocktomb", 2), _action(1, "weatherball", 2)), 0.3)
    out, report = _run(battle, [top])
    assert out[0] is top
    assert report.demotions["resisted_target:calc_disagrees"] == 1


def test_twin_action_arithmetic_matches_poke_env_bands():
    # move 2 (band 12..16) aimed at foe slot 2 -> same move at foe slot 1
    assert G.twin_target_action(16, 1) == 15
    assert G.twin_target_action(15, 2) == 16
    # every band start keeps its band; switches and bad targets are refused
    for start in (7, 12, 17, 22, 27, 42, 87, 102):
        for target in (-2, -1, 0, 1, 2):
            assert G.twin_target_action(start + 3, target) == start + target + 2
    assert G.twin_target_action(3, 1) is None
    assert G.twin_target_action(16, 3) is None


def test_guard_is_registered_but_opt_in():
    assert "resisted_target" in G.GUARDS and "resisted_target" in G.GUARD_ORDER
    assert "resisted_target" not in G.HARD_GUARDS


def test_a_resisted_hit_that_finishes_the_foe_is_left_alone(monkeypatch):
    rotom, meganium = (
        Pokemon(gen=9, species="rotomwash"),
        Pokemon(gen=9, species="meganium"),
    )
    battle = _battle([meganium, rotom], weather={Weather.SUNNYDAY: 1})
    monkeypatch.setattr(G.K, "guaranteed_ko", lambda b, a, d, m: d is rotom)
    top = G.Candidate((_action(0, "rocktomb", 2), _action(1, "weatherball", 2)), 0.3)
    out, report = _run(battle, [top])
    assert out[0] is top and not report.stages
    assert report.demotions["resisted_target:current_ko"] == 1


def test_promoted_twin_inherits_the_corrected_pairs_probability():
    """The reranker scores log(prob / top prob): a promoted twin with its own
    4% would lose that term to the 28% original and be put back (ladder
    2026-09-06 turn 2). It must carry the confidence of the pair it corrects."""
    rotom, meganium = (
        Pokemon(gen=9, species="rotomwash"),
        Pokemon(gen=9, species="meganium"),
    )
    battle = _battle([meganium, rotom], weather={Weather.SUNNYDAY: 1})
    top = G.Candidate((_action(0, "rocktomb", 2), _action(1, "weatherball", 2)), 0.28)
    twin = G.Candidate((_action(0, "rocktomb", 2), _action(1, "weatherball", 1)), 0.04)
    out, _ = _run(battle, [top, twin])
    assert out[0] is twin and twin.prob == 0.28
