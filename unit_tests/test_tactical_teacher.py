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


# -- doomed: knocked out before it moves (the user chose this lesson, 2026-09-27) --


def _torkoal(extra: list[str] | None = None, hp: str = "60/177"):
    """Torkoal at a third of its HP beside Farigiraf; Garchomp outspeeds it and
    its Rock Slide / Earthquake knocks it out (T6-era ladder: 53 times a Torkoal
    attacked and fainted first)."""
    return position(
        [
            f"|switch|p1a: Torkoal|Torkoal, L50, M|{hp}",
            "|switch|p1b: Farigiraf|Farigiraf, L50, F|227/227",
            "|switch|p2a: Garchomp|Garchomp, L50, M|100/100",
            "|switch|p2b: Kingambit|Kingambit, L50, M|100/100",
            *(extra or []),
        ]
    )


def test_a_doomed_torkoal_is_taught_to_protect():
    from training.tactical_teacher import doomed_facts

    battle = _torkoal()
    chance, protect, wasted = doomed_facts(battle, 0, LEGAL[:107])
    assert chance >= 0.5
    assert protect[move_action(battle, 0, "protect", 0)]
    assert wasted[move_action(battle, 0, "eruption", 0)]
    assert not wasted[move_action(battle, 0, "protect", 0)]


def test_trick_room_reverses_it():
    from training.tactical_teacher import doomed_facts

    room = _torkoal(["|-fieldstart|move: Trick Room|[of] p1b: Farigiraf"])
    assert doomed_facts(room, 0, LEGAL[:107])[0] == 0.0


def test_no_lesson_after_a_protect_or_without_one():
    from training.tactical_teacher import doomed_facts

    battle = _torkoal()
    torkoal = battle.active_pokemon[0]
    assert torkoal is not None
    torkoal._protect_counter = 1  # it Protected last turn
    assert doomed_facts(battle, 0, LEGAL[:107])[0] == 0.0
    weak_farigiraf = position(
        [
            "|switch|p1a: Torkoal|Torkoal, L50, M|177/177",
            "|switch|p1b: Farigiraf|Farigiraf, L50, F|20/227",
            "|switch|p2a: Garchomp|Garchomp, L50, M|100/100",
            "|switch|p2b: Kingambit|Kingambit, L50, M|100/100",
        ]
    )
    chance, protect, _ = doomed_facts(weak_farigiraf, 1, LEGAL[:107])
    assert chance == 0.0 and not protect.any()  # Farigiraf has no Protect


def test_grassy_glide_is_priority_unless_our_armor_tail_blocks_it():
    from training.tactical_teacher import doomed_probability

    lines = [
        "|switch|p2a: Rillaboom|Rillaboom, L50, M|100/100",
        "|switch|p2b: Kingambit|Kingambit, L50, M|100/100",
        "|-fieldstart|move: Grassy Terrain|[from] ability: Grassy Surge"
        "|[of] p2a: Rillaboom",
        "|-fieldstart|move: Trick Room|[of] p1b: Farigiraf",
    ]
    exposed = position(
        [
            "|switch|p1a: Blastoise|Blastoise, L50, M|40/186",
            "|switch|p1b: Torkoal|Torkoal, L50, M|177/177",
            *lines,
        ]
    )
    shielded = position(
        [
            "|switch|p1a: Blastoise|Blastoise, L50, M|40/186",
            "|switch|p1b: Farigiraf|Farigiraf, L50, F|227/227",
            *lines,
        ]
    )
    # under our own Trick Room only the priority Grassy Glide can move first
    assert doomed_probability(exposed, 0) > 0.0
    assert doomed_probability(shielded, 0) == 0.0


def test_the_doomed_share_moves_from_wasted_moves_to_protect():
    from training.tactical_teacher import doomed_target

    q = np.array([0.0] * 7 + [0.6, 0.2, 0.1, 0.1] + [0.0] * 96)
    protect = np.zeros(107, dtype=bool)
    wasted = np.zeros(107, dtype=bool)
    protect[10] = True
    wasted[[7, 8]] = True
    t = doomed_target(q, 0.5, protect, wasted)
    assert t is not None
    assert t[7] == pytest.approx(0.3) and t[8] == pytest.approx(0.1)
    assert t[10] == pytest.approx(0.5)  # 0.1 + half of 0.8
    assert t[9] == pytest.approx(0.1) and t.sum() == pytest.approx(1.0)
    assert doomed_target(q, 0.0, protect, wasted) is None


def test_the_fine_tune_applies_the_doomed_lesson_on_top_of_the_others():
    from training.tactical_sft import build_targets

    p = np.zeros((1, 107))
    p[0, [7, 8, 10]] = [0.6, 0.3, 0.1]
    useless = np.zeros((1, 2, 107), dtype=bool)
    values = np.full((1, 2, 107), np.nan)
    doomed = np.array([[0.5, 0.0]])
    protect = np.zeros((1, 2, 107), dtype=bool)
    wasted = np.zeros((1, 2, 107), dtype=bool)
    protect[0, 0, 10] = True
    wasted[0, 0, [7, 8]] = True
    q0, q1, lesson = build_targets(p, p, useless, values, 0.1, doomed, protect, wasted)
    assert lesson[0].tolist() == [True, False]  # slot 2 is not doomed
    assert q0[0, 10] == pytest.approx(0.55) and q0[0, 7] == pytest.approx(0.3)
    old = build_targets(p, p, useless, values, 0.1)  # data without doomed facts
    assert not old[2].any()


# -- tactical fine-tune 3: the 2026-09-28 ladder review (vs s9mmow) --


def test_leaf_storm_loses_the_tie_to_a_drop_free_knockout():
    """Turn 7: Leaf Storm and Sludge Bomb both knock out the 9% Raichu."""
    from training.tactical_teacher import attack_values

    battle = position(
        [
            "|switch|p1a: Venusaur|Venusaur, L50, M|2/155",
            "|switch|p1b: Torkoal|Torkoal, L50, M|177/177",
            "|switch|p2a: Raichu|Raichu, L50, F|9/100",
            "|switch|p2b: Garchomp|Garchomp, L50, M|100/100",
            "|-weather|SunnyDay|[from] ability: Drought|[of] p1b: Torkoal",
        ]
    )
    values = attack_values(battle, 0, LEGAL[:107], None)
    leaf_storm = values[move_action(battle, 0, "leafstorm", 1)]
    sludge_bomb = values[move_action(battle, 0, "sludgebomb", 1)]
    assert sludge_bomb > leaf_storm


def _fake_out_beside_torkoal(foe_b: str = "Amoonguss"):
    return position(
        [
            "|switch|p1a: Incineroar|Incineroar, L50, F|201/201",
            "|switch|p1b: Torkoal|Torkoal, L50, M|60/177",
            "|switch|p2a: Garchomp|Garchomp, L50, M|100/100",
            f"|switch|p2b: {foe_b}|{foe_b}, L50, M|100/100",
            "|turn|1",  # Fake Out is a first-turn move
        ]
    )


def test_our_fake_out_takes_the_threat_off_the_partner():
    from training.tactical_teacher import doomed_facts

    battle = _fake_out_beside_torkoal()
    flare_blitz = move_action(battle, 0, "flareblitz", 1)
    fake_out_garchomp = move_action(battle, 0, "fakeout", 1)
    assert doomed_facts(battle, 1, LEGAL[:107], flare_blitz)[0] >= 0.5
    assert doomed_facts(battle, 1, LEGAL[:107], fake_out_garchomp)[0] == 0.0


def test_no_protect_lesson_when_both_are_doomed():
    """Turn 8: Venusaur (1%) and Torkoal (7%) both Protected, buying nothing."""
    from training.tactical_teacher import doomed_facts

    battle = position(
        [
            "|switch|p1a: Venusaur|Venusaur, L50, M|2/155",
            "|switch|p1b: Torkoal|Torkoal, L50, M|13/177",
            "|switch|p2a: Garchomp|Garchomp, L50, M|96/100",
            "|switch|p2b: Kingambit|Kingambit, L50, M|100/100",
        ]
    )
    assert doomed_facts(battle, 0, LEGAL[:107])[0] == 0.0
    assert doomed_facts(battle, 1, LEGAL[:107])[0] == 0.0


def test_a_fake_out_and_a_protect_do_not_pair():
    """Turn 10: Torkoal Protected beside Incineroar's Fake Out (here the Fake Out
    is slot 1, so the lesson lands on slot 2's Protect)."""
    from training.tactical_teacher import pair_facts, pair_target

    battle = _fake_out_beside_torkoal()
    fake_out = move_action(battle, 0, "fakeout", 1)
    drain, receive = pair_facts(battle, LEGAL[:107], fake_out)
    protect = move_action(battle, 1, "protect", 0)
    heat_wave = move_action(battle, 1, "heatwave", 0)
    assert drain[protect] and receive[heat_wave] and not receive[protect]
    q = np.zeros(107)
    q[[protect, heat_wave]] = [0.7, 0.3]
    t = pair_target(q, drain, receive)
    assert t is not None and t[protect] == 0.0 and t[heat_wave] == pytest.approx(1.0)
    # beside a Protect it is the Fake Out that gives way
    protect0 = move_action(battle, 0, "flareblitz", 1)  # an attack: no lesson
    assert not pair_facts(battle, LEGAL[:107], protect0)[0].any()


def test_beside_a_protect_the_fake_out_gives_way():
    """The turn-10 pair the other way round: Torkoal (slot 1) Protects, so
    Incineroar's Fake Out (slot 2) buys nothing; its attacks take the mass."""
    from training.tactical_teacher import pair_facts

    battle = position(
        [
            "|switch|p1a: Torkoal|Torkoal, L50, M|13/177",
            "|switch|p1b: Incineroar|Incineroar, L50, F|120/201",
            "|switch|p2a: Volcarona|Volcarona, L50, F|100/100",
            "|switch|p2b: Garchomp|Garchomp, L50, M|64/100",
            "|turn|10",
        ]
    )
    protect = move_action(battle, 0, "protect", 0)
    drain, receive = pair_facts(battle, LEGAL[:107], protect)
    assert drain[move_action(battle, 1, "fakeout", 1)]
    assert receive[move_action(battle, 1, "flareblitz", 2)]
    assert not receive[move_action(battle, 1, "fakeout", 2)]
