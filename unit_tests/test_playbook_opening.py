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
