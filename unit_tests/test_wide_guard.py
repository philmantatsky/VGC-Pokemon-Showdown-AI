"""wide_guard (ladder 2026-09-26, game 2): Aerodactyl showed Wide Guard on turn
2 and the bot kept throwing spread moves into it -- Water Spout with Helping Hand
on turn 3, Eruption with Helping Hand on turn 5. "it needs to be taught wide
guard if its gonna use water spout and eruption too" (the user). Positions
rebuilt from the ladder protocol with the pairs and probabilities the bot
logged."""

from __future__ import annotations

from unit_tests.ladder_position import move_action, position, run
from vgc_bench.src import guards as G

TURN3 = [
    "|switch|p1a: Blastoise|Blastoise, L50, M|102/186",
    "|detailschange|p1a: Blastoise|Blastoise-Mega, L50, M",
    "|switch|p1b: Farigiraf|Farigiraf, L50, F|160/227",
    "|switch|p2a: Indeedee|Indeedee, L50, M|100/100",
    "|switch|p2b: Aerodactyl|Aerodactyl, L50, M|93/100",
    "|-fieldstart|move: Psychic Terrain|[from] ability: Psychic Surge",
    "|-fieldstart|move: Trick Room",
    "|-weather|RainDance",
    "|move|p2b: Aerodactyl|Wide Guard|p2b: Aerodactyl",
]


def _turn3(extra: tuple[str, ...] = ()):
    battle = position([*TURN3, *extra])
    spout = move_action(battle, 0, "waterspout", 0)
    pulse_1 = move_action(battle, 0, "waterpulse", 1)
    pulse_2 = move_action(battle, 0, "waterpulse", 2)
    helping = move_action(battle, 1, "helpinghand", -1)
    psychic_1 = move_action(battle, 1, "psychic", 1)
    psychic_2 = move_action(battle, 1, "psychic", 2)
    pairs = [
        ((spout, helping), 0.2494),
        ((spout, psychic_1), 0.2082),
        ((spout, psychic_2), 0.1826),
        ((pulse_2, helping), 0.0589),
        ((pulse_2, psychic_1), 0.0542),
        ((pulse_2, psychic_2), 0.0482),
        ((pulse_1, helping), 0.0454),
        ((pulse_1, psychic_2), 0.0391),
    ]
    return battle, pairs


def test_no_water_spout_into_the_wide_guard_it_just_saw():
    battle, pairs = _turn3()
    out, report = run(G.guard_wide_guard, battle, pairs)
    assert out[0].actions == pairs[3][0]  # Water Pulse into Aerodactyl + Helping Hand
    assert report.demotions["wide_guard"] == 3
    assert "wide_guard" in report.stages


def test_a_taunted_wide_guard_user_cannot_block():
    battle, pairs = _turn3(("|-start|p2b: Aerodactyl|move: Taunt",))
    out, report = run(G.guard_wide_guard, battle, pairs)
    assert out[0].actions == pairs[0][0]
    assert not report.stages


def test_no_eruption_into_it_either():
    """Turn 5: Torkoal fresh in beside Farigiraf (54/227) against Basculegion and
    the 32% Aerodactyl. The logged top 8 all use Eruption or Heat Wave; Weather
    Ball pairs ranked below them (the audit keeps eight) and enter small."""
    battle = position(
        [
            "|switch|p1a: Torkoal|Torkoal, L50, M|177/177",
            "|switch|p1b: Farigiraf|Farigiraf, L50, F|54/227",
            "|switch|p2a: Basculegion|Basculegion, L50, M|100/100",
            "|switch|p2b: Aerodactyl|Aerodactyl, L50, M|32/100",
            "|move|p2b: Aerodactyl|Wide Guard|p2b: Aerodactyl",
            "|move|p2b: Aerodactyl|Rock Slide|p1a: Torkoal",
            "|-weather|SunnyDay|[from] ability: Drought|[of] p1a: Torkoal",
            "|-fieldstart|move: Trick Room",
        ]
    )
    eruption = move_action(battle, 0, "eruption", 0)
    heat_wave = move_action(battle, 0, "heatwave", 0)
    ball_2 = move_action(battle, 0, "weatherball", 2)
    helping = move_action(battle, 1, "helpinghand", -1)
    psychic_1 = move_action(battle, 1, "psychic", 1)
    psychic_2 = move_action(battle, 1, "psychic", 2)
    rain = move_action(battle, 1, "raindance", 0)
    out, report = run(
        G.guard_wide_guard,
        battle,
        [
            ((eruption, helping), 0.6926),
            ((eruption, psychic_1), 0.2599),
            ((eruption, psychic_2), 0.0393),
            ((heat_wave, helping), 0.0046),
            ((heat_wave, psychic_1), 0.0017),
            ((eruption, rain), 0.0005),
            ((heat_wave, psychic_2), 0.0003),
            ((ball_2, helping), 0.0002),
        ],
    )
    assert out[0].actions == (ball_2, helping)
    assert report.demotions["wide_guard"] == 7


def _fake_out(trick_room: bool):
    """Wide Guard on Aerodactyl's sheet; Incineroar fresh in beside Blastoise."""
    lines = [
        "|switch|p1a: Blastoise|Blastoise, L50, M|186/186",
        "|switch|p1b: Incineroar|Incineroar, L50, F|202/202",
        "|switch|p2a: Garchomp|Garchomp, L50, M|100/100",
        "|switch|p2b: Aerodactyl|Aerodactyl, L50, M|100/100",
        "|move|p2b: Aerodactyl|Wide Guard|p2b: Aerodactyl",
    ]
    if trick_room:
        lines.append("|-fieldstart|move: Trick Room")
    battle = position([*lines, "|turn|1"])
    aerodactyl = battle.opponent_active_pokemon[1]
    assert aerodactyl is not None
    aerodactyl._item = "focussash"  # its sheet shows it
    spout = move_action(battle, 0, "waterspout", 0)
    pulse_2 = move_action(battle, 0, "waterpulse", 2)
    fake_out = move_action(battle, 1, "fakeout", 2)
    return battle, [((spout, fake_out), 0.6), ((pulse_2, fake_out), 0.4)]


def test_fake_out_that_surely_moves_first_keeps_the_spread_move():
    """Both are +3: under Trick Room our slow Incineroar surely acts before
    Aerodactyl, so the flinch stops Wide Guard."""
    battle, pairs = _fake_out(trick_room=True)
    out, report = run(G.guard_wide_guard, battle, pairs)
    assert out[0].actions == pairs[0][0]
    assert not report.stages


def test_fake_out_that_may_move_second_does_not():
    battle, pairs = _fake_out(trick_room=False)
    out, _ = run(G.guard_wide_guard, battle, pairs)
    assert out[0].actions == pairs[1][0]


def test_registered_but_opt_in_before_the_attack_guards():
    assert "wide_guard" in G.GUARDS
    assert "wide_guard" not in G.HARD_GUARDS
    order = G.GUARD_ORDER
    assert order.index("single_target_weather_ball") < order.index("wide_guard")
    assert order.index("wide_guard") < order.index("dominated_attack")
