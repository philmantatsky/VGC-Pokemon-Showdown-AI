"""dominated_throat_chop (ladder 2026-09-25): dominated_attack never swapped
Throat Chop (its 100% "no sound moves" secondary made it an effect attack), so
Incineroar kept using it where Flare Blitz hit harder. The positions are rebuilt
from the ladder protocol; the candidate pairs and probabilities are the ones the
bot logged on those turns; the real damage calculator scores them."""

from __future__ import annotations

import pytest

from unit_tests.ladder_position import move_action, position, run
from vgc_bench.src import guards as G


def _annihilape(extra: tuple[str, ...] = ()):
    """Game 21 (2687981911), turn 8: a full-HP Annihilape alone against our Torkoal
    (13/177) and Incineroar, in our sun."""
    battle = position(
        [
            "|switch|p1a: Torkoal|Torkoal, L50, M|13/177",
            "|switch|p1b: Incineroar|Incineroar, L50, F|153/202",
            "|switch|p2a: Pawmot|Pawmot, L50, M|100/100",
            "|switch|p2b: Annihilape|Annihilape, L50, M|100/100",
            "|faint|p2a: Pawmot",
            "|-weather|SunnyDay|[from] ability: Drought|[of] p1a: Torkoal",
            *extra,
        ]
    )
    heat_wave = move_action(battle, 0, "heatwave", 0)
    eruption = move_action(battle, 0, "eruption", 0)
    protect = move_action(battle, 0, "protect", 0)
    chop = move_action(battle, 1, "throatchop", 2)
    blitz_1 = move_action(battle, 1, "flareblitz", 1)
    blitz_2 = move_action(battle, 1, "flareblitz", 2)
    pairs = [
        ((heat_wave, chop), 0.3752),
        ((eruption, chop), 0.3752),
        ((protect, chop), 0.2786),
        ((eruption, blitz_1), 0.0432),
        ((eruption, blitz_2), 0.0431),
        ((protect, blitz_2), 0.0344),
        ((protect, blitz_1), 0.0333),
        ((heat_wave, blitz_1), 0.0177),
    ]
    return battle, pairs, (heat_wave, blitz_1)


def test_flare_blitz_replaces_throat_chop_into_annihilape():
    battle, pairs, fixed = _annihilape()
    out, report = run(G.guard_dominated_throat_chop, battle, pairs)
    assert out[0].actions == fixed  # Heat Wave kept, Flare Blitz auto-retargets
    assert out[0].prob == pytest.approx(0.3752)  # survives the opponent reranker
    assert report.demotions["dominated_throat_chop:promoted"] == 1
    assert "dominated_throat_chop" in report.stages


def test_dominated_attack_itself_still_leaves_throat_chop_alone():
    battle, pairs, _ = _annihilape()
    out, report = run(G.guard_dominated_attack, battle, pairs)
    assert out[0].actions == pairs[0][0]
    assert not report.stages


def test_a_shown_sound_move_keeps_throat_chop():
    """The block is the point against a foe with a sound move."""
    battle, pairs, _ = _annihilape(("|move|p2b: Annihilape|Screech|p1b: Incineroar",))
    out, report = run(G.guard_dominated_throat_chop, battle, pairs)
    assert out[0].actions == pairs[0][0]
    assert not report.stages


def test_flare_blitz_replaces_throat_chop_into_baxcalibur_beside_parting_shot():
    """Game 14 (2687979475), turn 5: Throat Chop into Mega Baxcalibur (neutral)
    beside Torkoal's sun Eruption, Flare Blitz neutral too and half as strong
    again; the foe Incineroar's Parting Shot keeps any Throat Chop into IT an
    effect attack."""
    battle = position(
        [
            "|switch|p1a: Torkoal|Torkoal, L50, M|177/177",
            "|switch|p1b: Incineroar|Incineroar, L50, F|184/202",
            "|switch|p2a: Baxcalibur|Baxcalibur, L50, M|83/100",
            "|detailschange|p2a: Baxcalibur|Baxcalibur-Mega, L50, M",
            "|switch|p2b: Incineroar|Incineroar, L50, M|65/100",
            "|-weather|SunnyDay|[from] ability: Drought|[of] p1a: Torkoal",
            "|move|p2b: Incineroar|Parting Shot|p1b: Incineroar",
            "|-unboost|p1b: Incineroar|atk|2",
        ]
    )
    eruption = move_action(battle, 0, "eruption", 0)
    heat_wave = move_action(battle, 0, "heatwave", 0)
    chop_1 = move_action(battle, 1, "throatchop", 1)
    chop_2 = move_action(battle, 1, "throatchop", 2)
    blitz_1 = move_action(battle, 1, "flareblitz", 1)
    blitz_2 = move_action(battle, 1, "flareblitz", 2)
    out, _ = run(
        G.guard_dominated_throat_chop,
        battle,
        [
            ((eruption, chop_1), 0.7051),
            ((eruption, chop_2), 0.7051),
            ((eruption, blitz_1), 0.1274),
            ((heat_wave, chop_2), 0.0718),
            ((eruption, blitz_2), 0.0191),
            ((heat_wave, blitz_1), 0.0138),
        ],
    )
    assert out[0].actions == (eruption, blitz_1)


def test_throat_chop_replaces_flare_blitz_into_chandelure():
    """Game 11 (2687975114), turn 7: Flare Blitz into Fire-type Chandelure
    (resisted) while Throat Chop hits the Ghost type twice as hard."""
    battle = position(
        [
            "|switch|p1a: Incineroar|Incineroar, L50, F|24/202",
            "|switch|p2a: Chandelure|Chandelure, L50, F|53/100",
            "|switch|p2b: Milotic|Milotic, L50, F|67/100",
            "|-weather|SunnyDay",
            "|-boost|p2b: Milotic|spa|2",
        ]
    )
    blitz_1 = move_action(battle, 0, "flareblitz", 1)
    blitz_2 = move_action(battle, 0, "flareblitz", 2)
    chop_1 = move_action(battle, 0, "throatchop", 1)
    chop_2 = move_action(battle, 0, "throatchop", 2)
    out, _ = run(
        G.guard_dominated_throat_chop,
        battle,
        [
            ((blitz_1, 0), 0.6122),
            ((chop_1, 0), 0.1656),
            ((blitz_2, 0), 0.1234),
            ((chop_2, 0), 0.0661),
        ],
    )
    assert out[0].actions == (chop_1, 0)


def test_throat_chop_into_milotic_is_already_the_better_hit():
    """Game 11, turn 5: Flare Blitz is resisted by Milotic; nothing to swap."""
    battle = position(
        [
            "|switch|p1a: Incineroar|Incineroar, L50, F|202/202",
            "|switch|p1b: Torkoal|Torkoal, L50, M|62/177",
            "|switch|p2a: Chandelure|Chandelure, L50, F|53/100",
            "|switch|p2b: Milotic|Milotic, L50, F|95/100",
            "|-weather|SunnyDay|[from] ability: Drought|[of] p1b: Torkoal",
            "|-boost|p2b: Milotic|spa|2",
        ]
    )
    heat_wave = move_action(battle, 1, "heatwave", 0)
    eruption = move_action(battle, 1, "eruption", 0)
    chop_1 = move_action(battle, 0, "throatchop", 1)
    chop_2 = move_action(battle, 0, "throatchop", 2)
    blitz_1 = move_action(battle, 0, "flareblitz", 1)
    blitz_2 = move_action(battle, 0, "flareblitz", 2)
    out, report = run(
        G.guard_dominated_throat_chop,
        battle,
        [
            ((chop_2, heat_wave), 0.635),
            ((chop_1, heat_wave), 0.1676),
            ((blitz_2, heat_wave), 0.044),
            ((chop_2, eruption), 0.033),
            ((blitz_1, heat_wave), 0.0257),
        ],
    )
    assert out[0].actions == (chop_2, heat_wave)
    assert not report.stages


def test_small_gains_into_a_resisting_wall_are_left_alone():
    """Game 8 (2687971839), turn 9: Flare Blitz scores 1.5x Throat Chop into the
    Mega Tyranitar that resists both, but only 4.8% of its HP more -- under the
    attack check's 0.05 floor, so the guard stands down, as dominated_attack
    would for any other pair of moves."""
    battle = position(
        [
            "|switch|p1a: Torkoal|Torkoal, L50, M|177/177",
            "|switch|p1b: Incineroar|Incineroar, L50, F|98/202",
            "|switch|p2a: Krookodile|Krookodile, L50, M|100/100",
            "|switch|p2b: Tyranitar|Tyranitar, L50, F|59/100",
            "|detailschange|p2b: Tyranitar|Tyranitar-Mega, L50, F",
            "|-weather|Sandstorm|[from] ability: Sand Stream|[of] p2b: Tyranitar",
            "|-unboost|p2b: Tyranitar|atk|1",
            "|faint|p2a: Krookodile",
            "|faint|p1a: Torkoal",
        ]
    )
    chop_2 = move_action(battle, 1, "throatchop", 2)
    blitz_1 = move_action(battle, 1, "flareblitz", 1)
    chop_1 = move_action(battle, 1, "throatchop", 1)
    blitz_2 = move_action(battle, 1, "flareblitz", 2)
    out, report = run(
        G.guard_dominated_throat_chop,
        battle,
        [
            ((0, chop_2), 0.633),
            ((0, blitz_1), 0.164),
            ((0, chop_1), 0.120),
            ((0, blitz_2), 0.072),
        ],
    )
    assert out[0].actions == (0, chop_2)
    assert not report.stages


def test_registered_but_opt_in_after_the_attack_check():
    assert "dominated_throat_chop" in G.GUARDS
    assert "dominated_throat_chop" not in G.HARD_GUARDS
    order = G.GUARD_ORDER
    assert order.index("dominated_attack") < order.index("dominated_throat_chop")
    assert order.index("dominated_throat_chop") < order.index("resisted_target")
    assert "partingshot" in G.SOUND_MOVES and "snarl" in G.SOUND_MOVES
