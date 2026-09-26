"""The tactical teacher (training/tactical_teacher.py): the move and damage facts
the guards patch at play time, as training targets. Positions rebuilt from the
T6ctx ladder read of 2026-09-26 (the bot's own mistakes, wins and losses)."""

from __future__ import annotations

import numpy as np
import pytest

from training.tactical_teacher import position_facts, target_distribution, useless
from unit_tests.ladder_position import move_action, position

LEGAL = np.zeros(107, dtype=np.int8)
LEGAL[7:47] = 1


def _sun():
    """Game 10 turn 2: full-HP Torkoal in sun; the network's pick was Heat Wave."""
    return position(
        [
            "|switch|p1a: Torkoal|Torkoal, L50, M|177/177",
            "|switch|p1b: Farigiraf|Farigiraf, L50, F|227/227",
            "|switch|p2a: Pincurchin|Pincurchin, L50|21/100",
            "|switch|p2b: Oranguru|Oranguru, L50|48/100",
            "|-weather|SunnyDay|[from] ability: Drought|[of] p1a: Torkoal",
        ]
    )


def _rain():
    """Game 6 turn 4: Leaf Storm went into Archaludon (4x resist) three times."""
    return position(
        [
            "|switch|p1a: Farigiraf|Farigiraf, L50, F|136/227",
            "|switch|p1b: Venusaur|Venusaur, L50, M|155/155",
            "|switch|p2a: Archaludon|Archaludon, L50, M|94/100",
            "|switch|p2b: Pelipper|Pelipper, L50, M|62/100",
            "|-weather|RainDance|[from] ability: Drizzle|[of] p2b: Pelipper",
        ]
    )


def _terrain():
    """Game 8 turn 1: the network put ~95% on Fake Out into Psychic Terrain."""
    return position(
        [
            "|switch|p1a: Blastoise|Blastoise, L50, M|186/186",
            "|switch|p1b: Farigiraf|Farigiraf, L50, F|227/227",
            "|switch|p2a: Tyranitar|Tyranitar, L50, M|100/100",
            "|switch|p2b: Indeedee|Indeedee-F, L50, F|100/100",
            "|-fieldstart|move: Psychic Terrain|[from] ability: Psychic Surge"
            "|[of] p2b: Indeedee",
            "|turn|1",
        ]
    )


def test_full_hp_eruption_is_worth_more_than_heat_wave():
    battle = _sun()
    _, values = position_facts(battle, 0, LEGAL, None)
    eruption = move_action(battle, 0, "eruption", 0)
    heat_wave = move_action(battle, 0, "heatwave", 0)
    assert values[eruption] > values[heat_wave] + 0.2


def test_leaf_storm_goes_where_it_is_not_resisted():
    battle = _rain()
    flags, values = position_facts(battle, 1, LEGAL, None)
    into_pelipper = move_action(battle, 1, "leafstorm", 2)
    into_archaludon = move_action(battle, 1, "leafstorm", 1)
    assert values[into_pelipper] > 2 * values[into_archaludon]
    assert flags[move_action(battle, 1, "sludgebomb", 1)]  # Steel: immune


def test_useless_actions():
    rain = _rain()
    assert (
        useless(rain, 0, move_action(rain, 0, "raindance", 0)) == "weather_already_up"
    )
    terrain = _terrain()
    assert (
        useless(terrain, 0, move_action(terrain, 0, "fakeout", 1)) == "psychic_terrain"
    )
    assert useless(terrain, 1, move_action(terrain, 1, "psychic", 1)) == "zero_damage"
    assert useless(terrain, 0, move_action(terrain, 0, "waterspout", 0)) is None


def test_fake_out_after_the_first_turn_fails():
    battle = position(
        [
            "|switch|p1a: Blastoise|Blastoise, L50, M|186/186",
            "|switch|p1b: Farigiraf|Farigiraf, L50, F|227/227",
            "|switch|p2a: Garchomp|Garchomp, L50, M|100/100",
            "|switch|p2b: Rotom|Rotom-Wash, L50|100/100",
            "|turn|1",
            "|move|p1a: Blastoise|Fake Out|p2a: Garchomp",
            "|turn|2",
        ]
    )
    assert useless(battle, 0, move_action(battle, 0, "fakeout", 1)) == "first_turn_only"


def test_target_keeps_the_policy_s_attack_mass_and_drops_useless_mass():
    probs = np.zeros(107)
    probs[[9, 14, 19, 24]] = [0.5, 0.2, 0.2, 0.1]  # three attacks and a status move
    flags = np.zeros(107, dtype=bool)
    flags[24] = True  # the status move certainly fails
    values = np.full(107, np.nan)
    values[[9, 14, 19]] = [0.2, 0.9, 0.1]
    q = target_distribution(probs, flags, values, tau=0.1)
    assert q is not None
    assert q.sum() == pytest.approx(1.0)
    assert q[24] == 0.0
    assert q[[9, 14, 19]].sum() == pytest.approx(1.0)  # all the live mass is attacks
    assert q[14] > 0.99 * q[[9, 14, 19]].sum() - 0.01
    # a support action the teacher knows nothing about keeps its share
    probs[29] = 0.2
    probs /= probs.sum()
    q = target_distribution(probs, flags, values, tau=0.1)
    assert q is not None
    live = probs[[9, 14, 19, 29]].sum()
    assert q[29] == pytest.approx(probs[29] / live)


def test_nothing_to_teach():
    probs = np.zeros(107)
    probs[[9, 24]] = [0.6, 0.4]
    values = np.full(107, np.nan)
    values[9] = 0.5  # one attack: no ranking to teach
    assert target_distribution(probs, np.zeros(107, dtype=bool), values, 0.1) is None
