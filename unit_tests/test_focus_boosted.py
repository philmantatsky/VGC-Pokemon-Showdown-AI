"""focus_boosted (ladder 2026-09-25): attacks went into the other foe while one
had set up. Positions rebuilt from the ladder protocol with the pairs and
probabilities the bot logged; the real damage calculator scores them."""

from __future__ import annotations

import pytest
from poke_env.battle import Pokemon

from unit_tests.ladder_position import move_action, position, run
from vgc_bench.src import guards as G


def _baxcalibur(boost: int = 2):
    """T6 read, game 2686259122, turn 5: Mega Blastoise + Farigiraf in our rain
    and Trick Room against Milotic (70%) and a +2 Attack Mega Baxcalibur (71%),
    which KO'd Blastoise that turn."""
    lines = [
        "|switch|p1a: Blastoise|Blastoise, L50, M|186/186",
        "|detailschange|p1a: Blastoise|Blastoise-Mega, L50, M",
        "|switch|p1b: Farigiraf|Farigiraf, L50, F|227/227",
        "|switch|p2a: Milotic|Milotic, L50, F|70/100",
        "|switch|p2b: Baxcalibur|Baxcalibur, L50, M|71/100",
        "|detailschange|p2b: Baxcalibur|Baxcalibur-Mega, L50, M",
        "|-weather|RainDance",
        "|-fieldstart|move: Trick Room",
    ]
    if boost:
        lines.append(f"|-boost|p2b: Baxcalibur|atk|{boost}")
    battle = position(lines)
    spout = move_action(battle, 0, "waterspout", 0)
    pulse_1 = move_action(battle, 0, "waterpulse", 1)
    pulse_2 = move_action(battle, 0, "waterpulse", 2)
    psychic_1 = move_action(battle, 1, "psychic", 1)
    psychic_2 = move_action(battle, 1, "psychic", 2)
    helping = move_action(battle, 1, "helpinghand", -1)
    rain = move_action(battle, 1, "raindance", 0)
    pairs = [
        ((spout, psychic_1), 0.7655),
        ((spout, psychic_2), 0.118),
        ((spout, helping), 0.0577),
        ((spout, rain), 0.0313),
        ((pulse_2, psychic_1), 0.01),
        ((pulse_1, psychic_1), 0.0088),
        ((pulse_1, psychic_2), 0.0016),
        ((pulse_2, psychic_2), 0.0015),
    ]
    return battle, pairs, (spout, psychic_2)


def test_psychic_goes_into_the_plus_two_baxcalibur():
    battle, pairs, focused = _baxcalibur()
    out, report = run(G.guard_focus_boosted, battle, pairs)
    assert out[0].actions == focused
    assert out[0].prob == pytest.approx(0.7655)  # survives the opponent reranker
    assert report.demotions["focus_boosted:promoted"] == 1
    assert "focus_boosted" in report.stages


def test_no_set_up_foe_no_change():
    battle, pairs, _ = _baxcalibur(boost=0)
    out, report = run(G.guard_focus_boosted, battle, pairs)
    assert out[0].actions == pairs[0][0]
    assert not report.stages


def test_a_boosted_foe_the_partner_already_knocks_out_draws_nothing():
    """Game 21 (2687981911), turn 5: Serperior at 3% with +2 Sp. Atk; Torkoal's
    sun Eruption already knocks it out, so Throat Chop stays on Golisopod."""
    battle = position(
        [
            "|switch|p1a: Torkoal|Torkoal, L50, M|177/177",
            "|switch|p1b: Incineroar|Incineroar, L50, F|202/202",
            "|switch|p2a: Serperior|Serperior, L50, M|3/100",
            "|switch|p2b: Golisopod|Golisopod, L50, F|53/100",
            "|detailschange|p2b: Golisopod|Golisopod-Mega, L50, F",
            "|-weather|SunnyDay|[from] ability: Drought|[of] p1a: Torkoal",
            "|-boost|p2a: Serperior|spa|2",
            "|-boost|p2a: Serperior|atk|1",
            "|-boost|p2a: Serperior|spd|1",
            "|-unboost|p2b: Golisopod|atk|1",
        ]
    )
    eruption = move_action(battle, 0, "eruption", 0)
    chop_1 = move_action(battle, 1, "throatchop", 1)
    chop_2 = move_action(battle, 1, "throatchop", 2)
    blitz_1 = move_action(battle, 1, "flareblitz", 1)
    blitz_2 = move_action(battle, 1, "flareblitz", 2)
    out, report = run(
        G.guard_focus_boosted,
        battle,
        [
            ((eruption, chop_2), 0.4184),
            ((eruption, blitz_1), 0.4132),
            ((eruption, blitz_2), 0.123),
            ((eruption, chop_1), 0.0298),
        ],
    )
    assert out[0].actions == (eruption, chop_2)
    assert not report.stages


def test_a_resisted_hit_into_the_boosted_foe_is_not_forced():
    """T6 read, game 2686267948, turn 4: Archaludon at +3 Sp. Atk resists Psychic
    (Steel); Psychic stays on the Garchomp it hits neutrally."""
    battle = position(
        [
            "|switch|p1a: Charizard|Charizard, L50, M|155/155",
            "|switch|p1b: Farigiraf|Farigiraf, L50, F|227/227",
            "|switch|p2a: Archaludon|Archaludon, L50, M|57/100",
            "|switch|p2b: Garchomp|Garchomp, L50, M|100/100",
            "|-boost|p2a: Archaludon|def|2",
            "|-boost|p2a: Archaludon|spa|3",
            "|-weather|RainDance",
        ]
    )
    heat_wave = move_action(battle, 0, "heatwave", 0)
    protect = move_action(battle, 0, "protect", 0)
    psychic_1 = move_action(battle, 1, "psychic", 1)
    psychic_2 = move_action(battle, 1, "psychic", 2)
    helping = move_action(battle, 1, "helpinghand", -1)
    out, report = run(
        G.guard_focus_boosted,
        battle,
        [
            ((heat_wave, psychic_2), 0.5735),
            ((heat_wave, helping), 0.1983),
            ((heat_wave, psychic_1), 0.1916),
            ((protect, psychic_2), 0.0093),
        ],
    )
    assert out[0].actions == (heat_wave, psychic_2)
    assert not report.stages


def _mon(species: str, **boosts: int) -> Pokemon:
    mon = Pokemon(gen=9, species=species)
    for stat, stage in boosts.items():
        mon.boost(stat, stage)
    return mon


def test_set_up_stages_count_offence_and_speed():
    assert G._setup_stages(_mon("blastoise", atk=2, spa=2, spe=2)) == 6
    assert G._setup_stages(_mon("dragonite", atk=1, spe=1)) == 2
    assert G._setup_stages(_mon("corviknight", atk=1)) == 1
    assert G._setup_stages(_mon("snorlax", atk=6, spe=6)) == 6  # capped
    assert G._setup_stages(_mon("clefable", spd=2)) == 0


def test_defense_counts_for_a_shown_body_press():
    aggron = _mon("aggron", **{"def": 4})
    assert G._setup_stages(aggron) == 0
    aggron._add_move("bodypress")
    assert G._setup_stages(aggron) == 4


def test_registered_but_opt_in_after_the_target_guards():
    assert "focus_boosted" in G.GUARDS
    assert "focus_boosted" not in G.HARD_GUARDS
    order = G.GUARD_ORDER
    assert order.index("resisted_target") < order.index("focus_boosted")
    assert order.index("overkill_split") < order.index("focus_boosted")
