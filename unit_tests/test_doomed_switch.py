"""doomed_switch (2026-10-03): Rillaboom's Wood Hammer knocks our Blastoise out on
turn 1 before it moves (outside Trick Room Rillaboom is faster); Torkoal, which
resists Grass, takes it. A Pokemon that moves first, or whose action is a status
move such as Trick Room, keeps its action."""

from __future__ import annotations

from unit_tests.ladder_position import move_action, position, run, switch_action
from vgc_bench.src import doomed_switch as D


def _turn(trick_room: bool = False, farigiraf_hp: str = "227/227"):
    return position(
        [
            "|switch|p1a: Blastoise|Blastoise, L50, M|186/186",
            f"|switch|p1b: Farigiraf|Farigiraf, L50, F|{farigiraf_hp}",
            "|switch|p2a: Rillaboom|Rillaboom, L50, M|100/100",
            "|switch|p2b: Kingambit|Kingambit, L50, M|100/100",
            "|-fieldstart|move: Grassy Terrain|[from] ability: Grassy Surge"
            "|[of] p2a: Rillaboom",
            *(
                ["|-fieldstart|move: Trick Room|[of] p1b: Farigiraf"]
                if trick_room
                else []
            ),
            "|turn|2",
        ]
    )


def _partner(battle):
    return move_action(battle, 1, "psychic", 1)


def test_blastoise_switches_to_torkoal_before_the_wood_hammer():
    battle = _turn()
    partner = _partner(battle)
    torkoal = switch_action(battle, "torkoal")
    out, report = run(
        D.guard_doomed_switch,
        battle,
        [
            ((move_action(battle, 0, "waterspout", 0), partner), 0.50),
            ((move_action(battle, 0, "icebeam", 1), partner), 0.20),
            ((torkoal, partner), 0.05),
        ],
    )
    assert out[0].actions == (torkoal, partner)
    assert report.demotions["doomed_switch:promoted"] == 1


def test_under_our_trick_room_blastoise_moves_first_and_stays():
    battle = _turn(trick_room=True)
    partner = _partner(battle)
    pairs = [
        ((move_action(battle, 0, "waterspout", 0), partner), 0.50),
        ((switch_action(battle, "torkoal"), partner), 0.05),
    ]
    out, report = run(D.guard_doomed_switch, battle, pairs)
    assert out[0].actions == pairs[0][0] and not report.stages


def test_a_trick_room_click_is_never_switched_away():
    """A low Farigiraf facing Kowtow Cleave still sets the room if the bot chose it."""
    battle = _turn(farigiraf_hp="40/227")
    blastoise_move = move_action(battle, 0, "waterspout", 0)
    trick_room = None
    for target in (0, -1, -2):
        try:
            trick_room = move_action(battle, 1, "trickroom", target)
            break
        except AssertionError:
            continue
    assert trick_room is not None
    pairs = [
        ((blastoise_move, trick_room), 0.50),
        ((blastoise_move, switch_action(battle, "incineroar")), 0.05),
    ]
    out, _ = run(D.guard_doomed_switch, battle, pairs)
    assert out[0].actions[1] == trick_room
