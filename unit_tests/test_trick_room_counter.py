"""trick_room_counter (ladder 2026-09-25): Trick Room pressed into a counter the
opponent had already shown. Real positions are rebuilt from the ladder protocol
with the pairs and probabilities the bot logged; the rest are the boundaries
(the user's fast-mode counter, Fake Out support, Armor Tail)."""

from __future__ import annotations

from unit_tests.ladder_position import move_action, position, run, switch_action
from vgc_bench.src import guards as G
from vgc_bench.src import trick_room_guard as T


def _game5(extra: tuple[str, ...] = ()):
    """Game 5 (2687969131), turn 2, sides swapped: on turn 1 our room and
    Cofagrigus's cancelled; the bot pressed Trick Room again, Cofagrigus
    reversed it on turn 3 and Farigiraf fell on turn 4."""
    battle = position(
        [
            "|switch|p1a: Blastoise|Blastoise, L50, M|131/186",
            "|detailschange|p1a: Blastoise|Blastoise-Mega, L50, M",
            "|switch|p1b: Farigiraf|Farigiraf, L50, F|117/227",
            "|switch|p2b: Cofagrigus|Cofagrigus, L50, M|63/100",
            "|turn|1",
            "|move|p1b: Farigiraf|Trick Room|p1b: Farigiraf",
            "|-fieldstart|move: Trick Room|[of] p1b: Farigiraf",
            "|move|p2b: Cofagrigus|Trick Room|p2b: Cofagrigus",
            "|-fieldend|move: Trick Room",
            "|switch|p2a: Gengar|Gengar, L50, M|100/100",
            "|turn|2",
            *extra,
        ]
    )
    room = move_action(battle, 1, "trickroom", 0)
    psychic_1 = move_action(battle, 1, "psychic", 1)
    spout = move_action(battle, 0, "waterspout", 0)
    pulse_1 = move_action(battle, 0, "waterpulse", 1)
    pulse_2 = move_action(battle, 0, "waterpulse", 2)
    ice_1 = move_action(battle, 0, "icebeam", 1)
    ice_2 = move_action(battle, 0, "icebeam", 2)
    pairs = [
        ((spout, room), 0.4196),
        ((pulse_1, room), 0.4196),
        ((ice_2, room), 0.1286),
        ((pulse_1, psychic_1), 0.1084),
        ((ice_1, room), 0.0919),
        ((pulse_2, room), 0.0904),
        ((ice_2, psychic_1), 0.0362),
        ((pulse_2, psychic_1), 0.026),
    ]
    return battle, pairs, (pulse_1, psychic_1)


def test_no_second_room_into_cofagrigus_that_cancelled_the_first():
    battle, pairs, attack = _game5()
    out, report = run(G.guard_trick_room_counter, battle, pairs)
    assert out[0].actions == attack  # both attacks into Gengar instead
    assert report.demotions["trick_room_counter:reverser"] == 1
    assert "trick_room_counter" in report.stages


def test_a_room_already_up_is_left_to_the_policy():
    battle, pairs, _ = _game5(("|-fieldstart|move: Trick Room",))
    out, report = run(G.guard_trick_room_counter, battle, pairs)
    assert out[0].actions == pairs[0][0]
    assert not report.stages


def test_the_partner_knocking_the_setter_out_keeps_the_room():
    """T6 read, game 2686291003, turn 3: Indeedee-F (33%) had shown Trick Room;
    Torkoal's sun Heat Wave knocks it out before -7 priority, so Heat Wave +
    Trick Room replaces Protect + Trick Room."""
    battle = position(
        [
            "|switch|p1a: Torkoal|Torkoal, L50, M|177/177",
            "|-weather|SunnyDay|[from] ability: Drought|[of] p1a: Torkoal",
            "|switch|p1b: Farigiraf|Farigiraf, L50, F|227/227",
            "|switch|p2a: Sneasler|Sneasler, L50, F|100/100",
            "|switch|p2b: Indeedee|Indeedee-F, L50, F|33/100",
            "|move|p2b: Indeedee|Trick Room|p2b: Indeedee",
        ]
    )
    room = move_action(battle, 1, "trickroom", 0)
    psychic_2 = move_action(battle, 1, "psychic", 2)
    protect = move_action(battle, 0, "protect", 0)
    heat_wave = move_action(battle, 0, "heatwave", 0)
    eruption = move_action(battle, 0, "eruption", 0)
    out, report = run(
        G.guard_trick_room_counter,
        battle,
        [
            ((protect, room), 0.7284),
            ((heat_wave, room), 0.2074),
            ((eruption, room), 0.0372),
            ((protect, psychic_2), 0.0139),
            ((heat_wave, psychic_2), 0.0041),
        ],
    )
    assert out[0].actions == (heat_wave, room)
    assert report.demotions["trick_room_counter:reverser"] == 1


def test_imprison_blocks_the_room_even_when_its_user_will_faint():
    """Game 11 (2687975114), turn 2: Indeedee's Imprison was up (it knows Trick
    Room at 58% in the set prior); the policy's top six pairs all pressed it.
    Showdown refuses the choice outright, so Water Spout knocking the 11%
    Indeedee out first does not rescue it."""
    battle = position(
        [
            "|switch|p1a: Blastoise|Blastoise, L50, M|186/186",
            "|detailschange|p1a: Blastoise|Blastoise-Mega, L50, M",
            "|switch|p1b: Farigiraf|Farigiraf, L50, F|128/227 brn",
            "|switch|p2a: Indeedee|Indeedee, L50, M|11/100",
            "|switch|p2b: Milotic|Milotic, L50, F|81/100",
            "|move|p2a: Indeedee|Imprison|p2a: Indeedee",
            "|-start|p2a: Indeedee|move: Imprison",
        ]
    )
    room = move_action(battle, 1, "trickroom", 0)
    psychic_1 = move_action(battle, 1, "psychic", 1)
    spout = move_action(battle, 0, "waterspout", 0)
    pulse_1 = move_action(battle, 0, "waterpulse", 1)
    pulse_2 = move_action(battle, 0, "waterpulse", 2)
    ice_1 = move_action(battle, 0, "icebeam", 1)
    ice_2 = move_action(battle, 0, "icebeam", 2)
    incineroar = switch_action(battle, "incineroar")
    out, report = run(
        G.guard_trick_room_counter,
        battle,
        [
            ((spout, room), 0.3788),
            ((pulse_2, room), 0.3788),
            ((pulse_1, room), 0.3242),
            ((ice_2, room), 0.1679),
            ((ice_1, room), 0.0306),
            ((incineroar, room), 0.0044),
            ((pulse_2, psychic_1), 0.001),
            ((pulse_1, psychic_1), 0.0007),
        ],
    )
    assert out[0].actions == (pulse_2, psychic_1)
    assert report.demotions["trick_room_counter:imprison"] == 1


def _taunt(blastoise_fresh: bool):
    """Sneasler showed Taunt on turn 1; Blastoise is either fresh in (Fake Out
    works) or has been out since the start (it does not)."""
    lead = "Blastoise" if not blastoise_fresh else "Incineroar"
    lines = [
        f"|switch|p1a: {lead}|{lead}, L50, M|186/186",
        "|switch|p1b: Farigiraf|Farigiraf, L50, F|227/227",
        "|switch|p2a: Sneasler|Sneasler, L50, F|100/100",
        "|switch|p2b: Kingambit|Kingambit, L50, M|100/100",
        "|turn|1",
        "|move|p2a: Sneasler|Taunt|p1b: Farigiraf",
        "|-start|p1b: Farigiraf|move: Taunt",
        "|turn|2",
        "|-end|p1b: Farigiraf|move: Taunt",
    ]
    if blastoise_fresh:
        lines.append("|switch|p1a: Blastoise|Blastoise, L50, M|186/186")
    battle = position([*lines, "|turn|3"])
    room = move_action(battle, 1, "trickroom", 0)
    psychic_1 = move_action(battle, 1, "psychic", 1)
    pulse_1 = move_action(battle, 0, "waterpulse", 1)
    fake_out = move_action(battle, 0, "fakeout", 1)
    pairs = [
        ((pulse_1, room), 0.5),
        ((fake_out, room), 0.3),
        ((pulse_1, psychic_1), 0.2),
    ]
    return battle, pairs


def test_a_shown_taunt_demotes_the_room_unless_fake_out_stops_it():
    battle, pairs = _taunt(blastoise_fresh=True)
    out, report = run(G.guard_trick_room_counter, battle, pairs)
    assert out[0].actions == pairs[1][0]  # Fake Out on Sneasler + Trick Room
    assert report.demotions["trick_room_counter:taunt"] == 1


def test_a_shown_taunt_with_no_fake_out_left():
    battle, pairs = _taunt(blastoise_fresh=False)
    out, report = run(G.guard_trick_room_counter, battle, pairs)
    assert out[0].actions == pairs[2][0]
    assert report.demotions["trick_room_counter:taunt"] == 1


def test_prankster_taunt_cannot_pass_armor_tail():
    battle = position(
        [
            "|switch|p1a: Blastoise|Blastoise, L50, M|186/186",
            "|switch|p1b: Farigiraf|Farigiraf, L50, F|227/227",
            "|switch|p2a: Whimsicott|Whimsicott, L50, M|100/100",
            "|switch|p2b: Kingambit|Kingambit, L50, M|100/100",
            "|-ability|p2a: Whimsicott|Prankster",
            "|move|p2a: Whimsicott|Taunt|p1a: Blastoise",
        ]
    )
    room = move_action(battle, 1, "trickroom", 0)
    psychic_1 = move_action(battle, 1, "psychic", 1)
    pulse_1 = move_action(battle, 0, "waterpulse", 1)
    out, report = run(
        G.guard_trick_room_counter,
        battle,
        [((pulse_1, room), 0.6), ((pulse_1, psychic_1), 0.4)],
    )
    assert out[0].actions == (pulse_1, room)
    assert not report.stages


def test_fast_mode_keeps_the_counter_room():
    """The user's lesson (2026-09-24): with our fast side out against a Trick
    Room team, pressing Trick Room as they press theirs cancels it. Charizard
    (Speed 152) outruns Indeedee-F's fastest spread (150) and Hatterene."""
    battle = position(
        [
            "|switch|p1a: Charizard|Charizard, L50, M|155/155",
            "|switch|p1b: Farigiraf|Farigiraf, L50, F|227/227",
            "|switch|p2a: Indeedee|Indeedee-F, L50, F|100/100",
            "|switch|p2b: Hatterene|Hatterene, L50, F|100/100",
            "|move|p2a: Indeedee|Trick Room|p2a: Indeedee",
        ]
    )
    assert T._fast_mode(battle, 1)
    room = move_action(battle, 1, "trickroom", 0)
    psychic_1 = move_action(battle, 1, "psychic", 1)
    heat_wave = move_action(battle, 0, "heatwave", 0)
    out, report = run(
        G.guard_trick_room_counter,
        battle,
        [((heat_wave, room), 0.6), ((heat_wave, psychic_1), 0.4)],
    )
    assert out[0].actions == (heat_wave, room)
    assert not report.stages


def test_slow_blastoise_is_not_fast_mode():
    battle, _, _ = _game5()
    assert not T._fast_mode(battle, 1)


def test_registered_but_opt_in():
    assert "trick_room_counter" in G.GUARDS
    assert "trick_room_counter" not in G.HARD_GUARDS
    order = G.GUARD_ORDER
    assert order.index("trick_room_counter") < order.index("trick_room_direction")
    assert order.index("trick_room_counter") < order.index("protect_spam")
