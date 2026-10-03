"""Guards from the user's review of the 2026-10-03 challenge game (battle 2692336128,
open team sheets; vgc_bench/src/game_review_1003.py). Positions rebuilt from the
replay with the sides swapped (the bot was p2)."""

from __future__ import annotations

from unit_tests.ladder_position import move_action, position, run, switch_action
from vgc_bench.src import game_review_1003 as R


def _mon(mons, index: int):
    mon = mons[index]
    assert mon is not None
    return mon


def _turn2(raichu_attempts: int = 0):
    """Venusaur + Incineroar (just in) vs a Mega Raichu Y put to sleep last turn
    and a Sylveon that must recharge after Hyper Beam."""
    lines = [
        "|switch|p1a: Venusaur|Venusaur, L50, M|157/157",
        "|switch|p1b: Incineroar|Incineroar, L50, F|202/202",
        "|switch|p2a: Raichu|Raichu, L50, F|100/100",
        "|switch|p2b: Sylveon|Sylveon, L50, F|100/100",
        "|detailschange|p2a: Raichu|Raichu-Mega-Y, L50, F",
        "|-mega|p2a: Raichu|Raichu|Raichunite Y",
        "|-status|p2a: Raichu|slp|[from] move: Sleep Powder",
        "|move|p2b: Sylveon|Hyper Beam|p1b: Incineroar",
        "|-mustrecharge|p2b: Sylveon",
        "|-unboost|p2a: Raichu|atk|1",
        "|-unboost|p2b: Sylveon|atk|1",
    ]
    lines += ["|cant|p2a: Raichu|slp"] * raichu_attempts
    return position([*lines, "|turn|2"])


def _turn2_pairs(battle):
    sludge_bomb = move_action(battle, 0, "sludgebomb", 2)
    return [
        ((sludge_bomb, move_action(battle, 1, "fakeout", 1)), 0.191),
        ((sludge_bomb, move_action(battle, 1, "flareblitz", 1)), 0.158),
        ((sludge_bomb, move_action(battle, 1, "flareblitz", 2)), 0.124),
    ]


def test_no_fake_out_into_a_foe_that_cannot_act():
    battle = _turn2()
    raichu, sylveon = battle.opponent_active_pokemon
    assert R.cannot_act(battle, raichu) and R.cannot_act(battle, sylveon)
    pairs = _turn2_pairs(battle)
    out, report = run(R.guard_wasted_fake_out, battle, pairs)
    # Flare Blitz, not Fake Out; the calculator prefers finishing the recharging
    # Sylveon (Sludge Bomb leaves it about 37%) to chip on the sleeping Raichu
    assert out[0].actions == pairs[2][0]
    assert report.demotions["wasted_fake_out:promoted"] == 1


def test_a_sleeper_that_may_wake_is_still_faked_out():
    """After one failed attempt it may wake (Champions: 1 in 3), so it may act."""
    battle = _turn2(raichu_attempts=1)
    assert not R.cannot_act(battle, battle.opponent_active_pokemon[0])
    pairs = _turn2_pairs(battle)
    out, report = run(R.guard_wasted_fake_out, battle, pairs)
    assert out[0].actions == pairs[0][0] and not report.stages


def _turn6():
    """Venusaur + Incineroar vs a +1 Volcarona (sheet: Heat Wave, Bug Buzz, Quiver
    Dance, Protect) and Rillaboom in Grassy Terrain, no weather."""
    return position(
        [
            "|switch|p1a: Venusaur|Venusaur, L50, M|136/157",
            "|switch|p1b: Incineroar|Incineroar, L50, F|137/202",
            "|switch|p2a: Volcarona|Volcarona, L50, F|50/100",
            "|switch|p2b: Rillaboom|Rillaboom, L50, F|100/100",
            "|-fieldstart|move: Grassy Terrain|[from] ability: Grassy Surge"
            "|[of] p2b: Rillaboom",
            "|move|p2a: Volcarona|Protect|p2a: Volcarona",
            "|move|p2a: Volcarona|Quiver Dance|p2a: Volcarona",
            "|move|p2a: Volcarona|Bug Buzz|p1a: Venusaur",
            "|move|p2a: Volcarona|Heat Wave|p1a: Venusaur|[spread] p1a,p1b",
            "|-boost|p2a: Volcarona|def|1|[from] item: Grassy Seed",
            "|-boost|p2a: Volcarona|spa|1",
            "|-boost|p2a: Volcarona|spd|1",
            "|-boost|p2a: Volcarona|spe|1",
            "|turn|6",
        ]
    )


def test_throat_chop_gives_way_when_bug_buzz_is_not_the_threat():
    battle = _turn6()
    volcarona = _mon(battle.opponent_active_pokemon, 0)
    assert not R.sound_main_threat(battle, volcarona)  # Heat Wave is the threat
    sludge_bomb = move_action(battle, 0, "sludgebomb", 2)
    flare_blitz = move_action(battle, 1, "flareblitz", 1)
    out, report = run(
        R.guard_throat_chop_main_threat,
        battle,
        [
            ((sludge_bomb, move_action(battle, 1, "throatchop", 1)), 0.228),
            ((sludge_bomb, move_action(battle, 1, "flareblitz", 2)), 0.120),
            ((sludge_bomb, switch_action(battle, "torkoal")), 0.088),
            ((sludge_bomb, flare_blitz), 0.060),
        ],
    )
    assert out[0].actions == (sludge_bomb, flare_blitz)
    assert report.demotions["throat_chop_main_threat:promoted"] == 1


def _sound_attacker(fourth_move: str):
    """A Sylveon whose sheet has no Hyper Beam: Pixilate Hyper Voice is its best."""
    return position(
        [
            "|switch|p1a: Venusaur|Venusaur, L50, M|157/157",
            "|switch|p1b: Incineroar|Incineroar, L50, F|202/202",
            "|switch|p2a: Sylveon|Sylveon, L50, F|100/100",
            "|switch|p2b: Rillaboom|Rillaboom, L50, F|100/100",
            "|move|p2a: Sylveon|Hyper Voice|p1a: Venusaur|[spread] p1a,p1b",
            "|-ability|p2a: Sylveon|Pixilate",
            "|move|p2a: Sylveon|Quick Attack|p1a: Venusaur",
            "|move|p2a: Sylveon|Detect|p2a: Sylveon",
            f"|move|p2a: Sylveon|{fourth_move}|p2a: Sylveon",
            "|turn|3",
        ]
    )


def test_throat_chop_stays_against_a_sound_attacker():
    battle = _sound_attacker("Calm Mind")
    assert R.sound_main_threat(battle, _mon(battle.opponent_active_pokemon, 0))
    sludge_bomb = move_action(battle, 0, "sludgebomb", 1)
    pairs = [
        ((sludge_bomb, move_action(battle, 1, "throatchop", 1)), 0.3),
        ((sludge_bomb, move_action(battle, 1, "flareblitz", 1)), 0.2),
    ]
    out, report = run(R.guard_throat_chop_main_threat, battle, pairs)
    assert out[0].actions == pairs[0][0] and not report.stages


def _turn7(trick_room: bool = False):
    """Torkoal (just in, full HP, sun) + Incineroar vs a +1 Volcarona at 32% and a
    full-HP Rillaboom in Grassy Terrain."""
    return position(
        [
            "|switch|p1a: Torkoal|Torkoal, L50, M|177/177",
            "|switch|p1b: Incineroar|Incineroar, L50, F|56/202",
            "|switch|p2a: Volcarona|Volcarona, L50, F|32/100",
            "|switch|p2b: Rillaboom|Rillaboom, L50, F|100/100",
            "|-weather|SunnyDay|[from] ability: Drought|[of] p1a: Torkoal",
            "|-fieldstart|move: Grassy Terrain|[from] ability: Grassy Surge"
            "|[of] p2b: Rillaboom",
            *(
                ["|-fieldstart|move: Trick Room|[of] p1a: Torkoal"]
                if trick_room
                else []
            ),
            "|move|p2a: Volcarona|Protect|p2a: Volcarona",
            "|move|p2a: Volcarona|Quiver Dance|p2a: Volcarona",
            "|move|p2a: Volcarona|Bug Buzz|p1b: Incineroar",
            "|move|p2a: Volcarona|Heat Wave|p1b: Incineroar|[spread] p1a,p1b",
            "|-boost|p2a: Volcarona|def|1|[from] item: Grassy Seed",
            "|-boost|p2a: Volcarona|spa|1",
            "|-boost|p2a: Volcarona|spd|1",
            "|-boost|p2a: Volcarona|spe|1",
            "|-unboost|p1b: Incineroar|spe|1",
            "|turn|7",
        ]
    )


def _turn7_pairs(battle):
    flare_blitz = move_action(battle, 1, "flareblitz", 1)
    return [
        ((move_action(battle, 0, "eruption", 0), flare_blitz), 0.361),
        (
            (
                move_action(battle, 0, "eruption", 0),
                move_action(battle, 1, "flareblitz", 2),
            ),
            0.130,
        ),
        ((move_action(battle, 0, "heatwave", 0), flare_blitz), 0.121),
        ((move_action(battle, 0, "weatherball", 1), flare_blitz), 0.087),
    ]


def test_heat_wave_over_an_eruption_that_lands_after_the_hits():
    battle = _turn7()
    torkoal = _mon(battle.active_pokemon, 0)
    loss = R.expected_loss(battle, torkoal, torkoal.moves["eruption"])
    assert 0.35 < loss < 0.8  # the real Heat Wave took 49.7%
    pairs = _turn7_pairs(battle)
    out, report = run(R.guard_hp_move_after_hits, battle, pairs)
    assert out[0].actions == pairs[2][0]
    assert report.demotions["hp_move_after_hits:promoted"] == 1
    assert torkoal.current_hp == 177  # the HP override was undone


def test_under_our_trick_room_torkoal_erupts_first():
    battle = _turn7(trick_room=True)
    pairs = _turn7_pairs(battle)
    out, report = run(R.guard_hp_move_after_hits, battle, pairs)
    assert out[0].actions == pairs[0][0] and not report.stages


def test_with_hyper_beam_on_the_sheet_hyper_voice_is_not_its_main_threat():
    """The game's Sylveon: Hyper Beam (Pixilate, 150 base power) hits harder."""
    battle = _sound_attacker("Hyper Beam")
    assert not R.sound_main_threat(battle, _mon(battle.opponent_active_pokemon, 0))


def test_parting_shot_keeps_throat_chop():
    battle = position(
        [
            "|switch|p1a: Venusaur|Venusaur, L50, M|157/157",
            "|switch|p1b: Incineroar|Incineroar, L50, F|202/202",
            "|switch|p2a: Incineroar|Incineroar, L50, M|100/100",
            "|switch|p2b: Rillaboom|Rillaboom, L50, F|100/100",
            "|move|p2a: Incineroar|Parting Shot|p1a: Venusaur",
            "|turn|3",
        ]
    )
    assert R.sound_main_threat(battle, _mon(battle.opponent_active_pokemon, 0))
