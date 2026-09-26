"""dominated_spread (ladder 2026-09-26, game 3): a weak spread attack should give
way to a clearly stronger single-target attack -- "water spout and eruption are
HP based moves ... sometimes its better to just use the plain water move" -- but
not when the spread hit is the smart one: "we could be low hp but the spread
attack is better bc one of the opposing mons is really low on hp and we can get
damage on the other" (the user). Positions rebuilt from the ladder protocol;
the real damage calculator scores them."""

from __future__ import annotations

import pytest

from unit_tests.ladder_position import move_action, position, run
from vgc_bench.src import guards as G


def _game3(rotom_hp: int = 100, garchomp_hp: int = 91):
    """Game 3 (2688076692), turn 2, sides swapped: Mega-evolving Blastoise at
    47/186 after Thunderbolt and Rough Skin, Farigiraf beside it; Rotom-Wash and
    Garchomp; our Trick Room up."""
    return position(
        [
            "|switch|p1a: Blastoise|Blastoise, L50, M|47/186",
            "|switch|p1b: Farigiraf|Farigiraf, L50, F|227/227",
            f"|switch|p2a: Rotom|Rotom-Wash, L50|{rotom_hp}/100",
            f"|switch|p2b: Garchomp|Garchomp, L50, M|{garchomp_hp}/100",
            "|-fieldstart|move: Trick Room",
        ]
    )


def _mega(battle, move_id: str, target: int) -> int:
    """The same move with Mega Evolution this turn (the next band of 20)."""
    return move_action(battle, 0, move_id, target) + 20


def test_ice_beam_into_garchomp_replaces_a_quarter_hp_water_spout():
    battle = _game3()
    rain = move_action(battle, 1, "raindance", 0)
    psychic_1 = move_action(battle, 1, "psychic", 1)
    helping = move_action(battle, 1, "helpinghand", -1)
    spout = _mega(battle, "waterspout", 0)
    ice_1 = _mega(battle, "icebeam", 1)
    ice_2 = _mega(battle, "icebeam", 2)
    # The bot's logged top 8 (probabilities as logged); Ice Beam into Garchomp
    # ranked below them -- the audit keeps only eight -- so it enters small.
    out, report = run(
        G.guard_dominated_spread,
        battle,
        [
            ((spout, rain), 0.5398),
            ((spout, psychic_1), 0.1869),
            ((spout, helping), 0.1014),
            ((ice_1, rain), 0.0824),
            ((ice_1, psychic_1), 0.0329),
            ((ice_1, helping), 0.0161),
            ((ice_2, rain), 0.004),
        ],
    )
    assert out[0].actions == (ice_2, rain)  # not the resisted Ice Beam into Rotom
    assert out[0].prob == pytest.approx(0.5398)
    assert report.demotions["dominated_spread:promoted"] == 1


def test_a_low_hp_spread_hit_that_finishes_one_foe_and_chips_the_other_stays():
    """The user's case: Water Spout at 25% HP still finishes a 3% Rotom and puts
    chip on Garchomp; one single-target hit into Garchomp is worth less."""
    battle = _game3(rotom_hp=3, garchomp_hp=100)
    rain = move_action(battle, 1, "raindance", 0)
    spout = _mega(battle, "waterspout", 0)
    pulse_2 = _mega(battle, "waterpulse", 2)
    out, report = run(
        G.guard_dominated_spread, battle, [((spout, rain), 0.6), ((pulse_2, rain), 0.2)]
    )
    assert out[0].actions == (spout, rain)
    assert not report.stages


def test_a_full_hp_water_spout_keeps_its_place():
    battle = position(
        [
            "|switch|p1a: Blastoise|Blastoise, L50, M|186/186",
            "|detailschange|p1a: Blastoise|Blastoise-Mega, L50, M",
            "|switch|p1b: Farigiraf|Farigiraf, L50, F|227/227",
            "|switch|p2a: Rotom|Rotom-Wash, L50|100/100",
            "|switch|p2b: Garchomp|Garchomp, L50, M|100/100",
            "|-weather|RainDance",
        ]
    )
    spout = move_action(battle, 0, "waterspout", 0)
    pulse_2 = move_action(battle, 0, "waterpulse", 2)
    psychic_1 = move_action(battle, 1, "psychic", 1)
    out, report = run(
        G.guard_dominated_spread,
        battle,
        [((spout, psychic_1), 0.6), ((pulse_2, psychic_1), 0.2)],
    )
    assert out[0].actions == (spout, psychic_1)
    assert not report.stages


def test_one_foe_left_is_the_attack_check_s_job():
    battle = position(
        [
            "|switch|p1a: Blastoise|Blastoise, L50, M|47/186",
            "|switch|p1b: Farigiraf|Farigiraf, L50, F|227/227",
            "|switch|p2a: Rotom|Rotom-Wash, L50|100/100",
            "|switch|p2b: Garchomp|Garchomp, L50, M|91/100",
            "|faint|p2a: Rotom",
        ]
    )
    rain = move_action(battle, 1, "raindance", 0)
    spout = move_action(battle, 0, "waterspout", 0)
    ice_2 = move_action(battle, 0, "icebeam", 2)
    out, report = run(
        G.guard_dominated_spread, battle, [((spout, rain), 0.6), ((ice_2, rain), 0.2)]
    )
    assert out[0].actions == (spout, rain)
    assert not report.stages
    out, _ = run(
        G.guard_dominated_attack, battle, [((spout, rain), 0.6), ((ice_2, rain), 0.2)]
    )
    assert out[0].actions == (ice_2, rain)


def test_registered_but_opt_in_between_the_attack_guards():
    assert "dominated_spread" in G.GUARDS
    assert "dominated_spread" not in G.HARD_GUARDS
    order = G.GUARD_ORDER
    assert order.index("dominated_attack") < order.index("dominated_spread")
    assert order.index("dominated_spread") < order.index("resisted_target")
