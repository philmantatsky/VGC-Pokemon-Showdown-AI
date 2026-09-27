"""playbook_opening (vgc_bench/src/playbook_opening.py): the chosen plan card's
turn-1 script goes on top -- Fake Out their Trick Room / Tailwind setter (else the
lead that threatens our Trick Room setter most) while Farigiraf sets Trick Room.
Positions from the T6ctx / T6tac ladder reads of 2026-09-26."""

from __future__ import annotations

from unit_tests.ladder_position import move_action, position, run
from vgc_bench.src.playbook_opening import (
    fake_out_target,
    guard_playbook_opening,
    scripted_steps,
)

WATER_ROOM = {
    "card": "water_room",
    "turn1": {
        "mega": "blastoise",
        "fake_out": {"user": "blastoise", "target": "setter_or_threat"},
        "setter": {"user": "farigiraf", "move": "trickroom"},
    },
}


def _whimsicott_lead():
    """T6ctx game 4: Whimsicott (Prankster Trick Room) + Staraptor. Our Fake Out
    went into Staraptor and our Trick Room cancelled theirs."""
    battle = position(
        [
            "|switch|p1a: Blastoise|Blastoise, L50, M|186/186",
            "|switch|p1b: Farigiraf|Farigiraf, L50, F|227/227",
            "|switch|p2a: Whimsicott|Whimsicott, L50, F|100/100",
            "|switch|p2b: Staraptor|Staraptor, L50, F|100/100",
            "|turn|1",
        ]
    )
    setattr(battle, "_vgc_playbook", WATER_ROOM)
    return battle


def test_fake_out_goes_to_their_setter():
    battle = _whimsicott_lead()
    assert fake_out_target(battle, battle.active_pokemon[1]) == 1
    steps = scripted_steps(battle, WATER_ROOM)
    assert steps == {0: ("fakeout", 1), 1: ("trickroom", None)}


def test_the_script_is_promoted_over_the_policy_pick():
    battle = _whimsicott_lead()
    fake_out_staraptor = move_action(battle, 0, "fakeout", 2) + 20  # Mega band
    fake_out_whimsicott = move_action(battle, 0, "fakeout", 1) + 20
    trick_room = move_action(battle, 1, "trickroom", 0)
    out, report = run(
        guard_playbook_opening,
        battle,
        [
            ((fake_out_staraptor, trick_room), 0.48),
            ((fake_out_whimsicott, trick_room), 0.10),
        ],
    )
    assert out[0].actions == (fake_out_whimsicott, trick_room)
    assert out[0].prob == 0.48
    assert "playbook_opening" in report.stages


def test_the_script_is_built_when_the_policy_never_ranked_it():
    battle = _whimsicott_lead()
    water_spout = move_action(battle, 0, "waterspout", 0) + 20
    psychic = move_action(battle, 1, "psychic", 1)
    out, report = run(guard_playbook_opening, battle, [((water_spout, psychic), 0.7)])
    assert out[0].actions == (
        move_action(battle, 0, "fakeout", 1) + 20,
        move_action(battle, 1, "trickroom", 0),
    )
    assert report.demotions["playbook_opening:injected"] == 1


def test_no_setter_means_the_biggest_threat_to_farigiraf():
    """Kingambit's Dark moves hit Farigiraf super effectively; Sneasler's do not."""
    battle = position(
        [
            "|switch|p1a: Blastoise|Blastoise, L50, M|186/186",
            "|switch|p1b: Farigiraf|Farigiraf, L50, F|227/227",
            "|switch|p2a: Sneasler|Sneasler, L50, F|100/100",
            "|switch|p2b: Kingambit|Kingambit, L50, M|100/100",
            "|turn|1",
        ]
    )
    assert fake_out_target(battle, battle.active_pokemon[1]) == 2


def test_only_on_turn_one_and_never_into_our_own_room():
    battle = _whimsicott_lead()
    setattr(battle, "_vgc_playbook", None)
    pairs = [
        (
            (
                move_action(battle, 0, "waterspout", 0),
                move_action(battle, 1, "psychic", 1),
            ),
            0.7,
        )
    ]
    out, report = run(guard_playbook_opening, battle, pairs)
    assert out[0].actions == pairs[0][0] and not report.stages
    room = position(
        [
            "|switch|p1a: Blastoise|Blastoise, L50, M|186/186",
            "|switch|p1b: Farigiraf|Farigiraf, L50, F|227/227",
            "|switch|p2a: Whimsicott|Whimsicott, L50, F|100/100",
            "|switch|p2b: Staraptor|Staraptor, L50, F|100/100",
            "|-fieldstart|move: Trick Room",
            "|turn|1",
        ]
    )
    setattr(room, "_vgc_playbook", WATER_ROOM)
    out, report = run(guard_playbook_opening, room, pairs)
    assert not report.stages


def test_a_live_request_decides_what_can_be_scripted(monkeypatch):
    """The scripted pair is built, not drawn from the masked policy: a Mega the
    request does not allow is dropped, and the rest stays legal."""
    battle = _whimsicott_lead()
    moves = [list(mon.moves.values()) for mon in battle.active_pokemon if mon]
    cls = type(battle)
    monkeypatch.setattr(cls, "available_moves", property(lambda self: moves))
    monkeypatch.setattr(cls, "can_mega_evolve", property(lambda self: [False, False]))
    water_spout = move_action(battle, 0, "waterspout", 0)
    psychic = move_action(battle, 1, "psychic", 1)
    out, _ = run(guard_playbook_opening, battle, [((water_spout, psychic), 0.7)])
    assert out[0].actions == (
        move_action(battle, 0, "fakeout", 1),  # the plain band: no Mega allowed
        move_action(battle, 1, "trickroom", 0),
    )
    monkeypatch.setattr(cls, "can_mega_evolve", property(lambda self: [True, False]))
    out, _ = run(guard_playbook_opening, battle, [((water_spout, psychic), 0.7)])
    assert out[0].actions[0] == move_action(battle, 0, "fakeout", 1) + 20  # Mega


def _lead(*lines: str):
    battle = position(
        [
            "|switch|p1a: Blastoise|Blastoise, L50, M|186/186",
            "|switch|p1b: Farigiraf|Farigiraf, L50, F|227/227",
            *lines,
            "|turn|1",
        ]
    )
    setattr(battle, "_vgc_playbook", WATER_ROOM)
    return battle


def test_fake_out_skips_a_setter_it_cannot_touch():
    """Sinistcha sets Trick Room but is a Ghost: Fake Out goes to its partner
    instead of into an immunity (2026-09-27 openings research)."""
    battle = _lead(
        "|switch|p2a: Sinistcha|Sinistcha, L50|100/100",
        "|switch|p2b: Kingambit|Kingambit, L50, M|100/100",
    )
    assert fake_out_target(battle, battle.active_pokemon[1]) == 2


def test_no_fake_out_step_when_neither_foe_can_be_faked_out():
    """Psychic Terrain shields grounded foes and Armor Tail their whole side: the
    script keeps only Trick Room; Blastoise keeps the policy's own move."""
    terrain = _lead(
        "|switch|p2a: Indeedee|Indeedee-F, L50, F|100/100",
        "|-fieldstart|move: Psychic Terrain|[from] ability: Psychic Surge"
        "|[of] p2a: Indeedee",
        "|switch|p2b: Hatterene|Hatterene, L50, F|100/100",
    )
    assert scripted_steps(terrain, WATER_ROOM) == {1: ("trickroom", None)}
    armor_tail = _lead(
        "|switch|p2a: Farigiraf|Farigiraf, L50, M|100/100",
        "|switch|p2b: Kingambit|Kingambit, L50, M|100/100",
        "|-ability|p2a: Farigiraf|Armor Tail",
    )
    assert fake_out_target(armor_tail, armor_tail.active_pokemon[1]) is None
    water_spout = move_action(armor_tail, 0, "waterspout", 0) + 20
    psychic = move_action(armor_tail, 1, "psychic", 1)
    out, _ = run(guard_playbook_opening, armor_tail, [((water_spout, psychic), 0.7)])
    assert out[0].actions == (water_spout, move_action(armor_tail, 1, "trickroom", 0))


def test_double_intimidate_does_not_switch_out_a_fake_out():
    """Playbook trial game 6 turn 1 (2026-09-27): Salamence + Incineroar
    Intimidated Blastoise to -2; severe_attack_drop_switch took the physical Fake
    Out for a crippled attack and switched Mega Blastoise out for Torkoal (p=0.00).
    The flinch does not depend on Attack."""
    from vgc_bench.src import guards as G

    battle = position(
        [
            "|switch|p1a: Blastoise|Blastoise, L50, M|186/186",
            "|switch|p1b: Farigiraf|Farigiraf, L50, F|227/227",
            "|switch|p2a: Salamence|Salamence, L50, M|100/100",
            "|switch|p2b: Incineroar|Incineroar, L50, M|100/100",
            "|-ability|p2a: Salamence|Intimidate|boost",
            "|-unboost|p1a: Blastoise|atk|1",
            "|-unboost|p1b: Farigiraf|atk|1",
            "|-ability|p2b: Incineroar|Intimidate|boost",
            "|-unboost|p1a: Blastoise|atk|1",
            "|-unboost|p1b: Farigiraf|atk|1",
            "|turn|1",
        ]
    )
    fake_out = move_action(battle, 0, "fakeout", 1) + 20  # Mega band
    trick_room = move_action(battle, 1, "trickroom", 0)
    out, report = run(
        G.guard_severe_attack_drop_switch,
        battle,
        [((fake_out, trick_room), 0.72), ((5, trick_room), 0.0)],  # 5 = Torkoal
    )
    assert out[0].actions == (fake_out, trick_room)
    assert not report.stages


def test_a_ranked_pair_without_the_scripted_mega_is_not_the_script():
    """Playbook trial game 12 (2026-09-27): the policy had ranked Fake Out + Trick
    Room without the Mega, and promoting it cost Water Room its turn-1 Mega."""
    battle = _whimsicott_lead()
    plain_fake_out = move_action(battle, 0, "fakeout", 1)
    ice_beam = move_action(battle, 0, "icebeam", 2) + 20
    trick_room = move_action(battle, 1, "trickroom", 0)
    out, report = run(
        guard_playbook_opening,
        battle,
        [((ice_beam, trick_room), 0.63), ((plain_fake_out, trick_room), 0.30)],
    )
    assert out[0].actions == (plain_fake_out + 20, trick_room)
    assert report.demotions["playbook_opening:injected"] == 1
