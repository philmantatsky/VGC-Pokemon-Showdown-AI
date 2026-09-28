"""Three opt-in guards from the user's review of a ladder loss (2026-09-28, the
doomed-lesson read, vs s9mmow): Leaf Storm to finish a 9% Raichu (turn 7), Torkoal
switched out instead of the -2 Sp. Atk Venusaur (turn 9), Fake Out beside a
Protect (turn 10). Positions rebuilt from that game; the candidate pairs and
their probabilities are the logged ones."""

from __future__ import annotations

from unit_tests.ladder_position import move_action, position, run, switch_action
from vgc_bench.src import guards as G


def _turn7():
    return position(
        [
            "|switch|p1a: Venusaur|Venusaur, L50, M|2/155",
            "|switch|p1b: Torkoal|Torkoal, L50, M|177/177",
            "|switch|p2a: Raichu|Raichu, L50, F|9/100",
            "|switch|p2b: Garchomp|Garchomp, L50, M|100/100",
            "|-weather|SunnyDay|[from] ability: Drought|[of] p1b: Torkoal",
            "|turn|7",
        ]
    )


def test_a_drop_free_move_finishes_the_low_raichu():
    battle = _turn7()
    eruption = move_action(battle, 1, "eruption", 0)
    leaf_storm_raichu = move_action(battle, 0, "leafstorm", 1)
    sludge_bomb_raichu = move_action(battle, 0, "sludgebomb", 1)
    out, report = run(
        G.guard_drop_free_finish,
        battle,
        [
            ((leaf_storm_raichu, eruption), 0.312),
            ((move_action(battle, 0, "sludgebomb", 2), eruption), 0.255),
            ((move_action(battle, 0, "leafstorm", 2), eruption), 0.187),
            ((sludge_bomb_raichu, eruption), 0.060),
        ],
    )
    assert out[0].actions == (sludge_bomb_raichu, eruption)
    assert out[0].prob == 0.312
    assert report.demotions["drop_free_finish:promoted"] == 1


def test_leaf_storm_stays_when_nothing_else_finishes():
    """Into a full-HP Garchomp neither move is a sure knockout: no change."""
    battle = _turn7()
    eruption = move_action(battle, 1, "eruption", 0)
    pairs = [
        ((move_action(battle, 0, "leafstorm", 2), eruption), 0.3),
        ((move_action(battle, 0, "sludgebomb", 2), eruption), 0.2),
    ]
    out, report = run(G.guard_drop_free_finish, battle, pairs)
    assert out[0].actions == pairs[0][0] and not report.stages


def _turn10():
    return position(
        [
            "|switch|p1a: Torkoal|Torkoal, L50, M|13/177",
            "|switch|p1b: Incineroar|Incineroar, L50, F|120/201",
            "|switch|p2a: Volcarona|Volcarona, L50, F|100/100",
            "|switch|p2b: Garchomp|Garchomp, L50, M|64/100",
            "|-weather|SunnyDay|[from] ability: Drought|[of] p1a: Torkoal",
            "|turn|10",
        ]
    )


def test_a_fake_out_is_for_the_partner_to_act():
    battle = _turn10()
    fake_out = move_action(battle, 1, "fakeout", 1)
    protect = move_action(battle, 0, "protect", 0)
    heat_wave = move_action(battle, 0, "heatwave", 0)
    out, report = run(
        G.guard_fake_out_partner_acts,
        battle,
        [
            ((protect, fake_out), 0.410),
            ((heat_wave, fake_out), 0.116),
            ((protect, move_action(battle, 1, "throatchop", 2)), 0.064),
        ],
    )
    assert out[0].actions == (heat_wave, fake_out)
    assert report.demotions["fake_out_partner_acts:promoted"] == 1


def test_protect_beside_a_real_attack_is_left_alone():
    battle = _turn10()
    protect = move_action(battle, 0, "protect", 0)
    pairs = [
        ((protect, move_action(battle, 1, "flareblitz", 1)), 0.5),
        (
            (
                move_action(battle, 0, "heatwave", 0),
                move_action(battle, 1, "fakeout", 1),
            ),
            0.2,
        ),
    ]
    out, report = run(G.guard_fake_out_partner_acts, battle, pairs)
    assert out[0].actions == pairs[0][0] and not report.stages


def _turn9(spa_drop: bool = True):
    return position(
        [
            "|switch|p1a: Venusaur|Venusaur, L50, M|2/155",
            "|switch|p1b: Torkoal|Torkoal, L50, M|13/177",
            "|switch|p2a: Volcarona|Volcarona, L50, F|100/100",
            "|switch|p2b: Garchomp|Garchomp, L50, M|96/100",
            "|-weather|SunnyDay|[from] ability: Drought|[of] p1b: Torkoal",
            *(["|-unboost|p1a: Venusaur|spa|2"] if spa_drop else []),
            "|-unboost|p1b: Torkoal|spa|1",
            "|turn|9",
        ]
    )


def test_the_crippled_venusaur_takes_the_switch():
    battle = _turn9()
    incineroar = switch_action(battle, "incineroar")
    weather_ball = move_action(battle, 1, "weatherball", 1)
    out, report = run(
        G.guard_switch_the_crippled,
        battle,
        [
            ((move_action(battle, 0, "leafstorm", 2), incineroar), 0.113),
            ((move_action(battle, 0, "sludgebomb", 2), incineroar), 0.113),
            ((move_action(battle, 0, "sludgebomb", 2), weather_ball), 0.056),
        ],
    )
    assert out[0].actions == (incineroar, weather_ball)  # Venusaur out, Torkoal acts
    assert report.demotions["switch_the_crippled:injected"] == 1


def test_no_switch_swap_without_a_real_drop():
    battle = _turn9(spa_drop=False)
    incineroar = switch_action(battle, "incineroar")
    pairs = [
        ((move_action(battle, 0, "leafstorm", 2), incineroar), 0.2),
        (
            (
                move_action(battle, 0, "sludgebomb", 2),
                move_action(battle, 1, "weatherball", 1),
            ),
            0.1,
        ),
    ]
    out, report = run(G.guard_switch_the_crippled, battle, pairs)
    assert out[0].actions == pairs[0][0] and not report.stages
