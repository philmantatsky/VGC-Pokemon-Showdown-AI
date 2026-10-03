"""threat_first (2026-10-03, the user: "what do u think for the losing a pokemon
problem"): Rillaboom's Wood Hammer is the top killer of our leads on ladder, and
under our Trick Room our Mega Blastoise moves before it and knocks it out with Ice
Beam. Positions follow the ladder pattern: Blastoise at half HP beside Farigiraf,
our room up, a damaged Rillaboom in Grassy Terrain."""

from __future__ import annotations

from unit_tests.ladder_position import move_action, position, run, switch_action
from vgc_bench.src import guards as G


def _turn(rillaboom_hp: str = "55/100", trick_room: bool = True):
    return position(
        [
            "|switch|p1a: Blastoise|Blastoise, L50, M|90/186",
            "|switch|p1b: Farigiraf|Farigiraf, L50, F|200/227",
            f"|switch|p2a: Rillaboom|Rillaboom, L50, M|{rillaboom_hp}",
            "|switch|p2b: Incineroar|Incineroar, L50, M|100/100",
            "|-fieldstart|move: Grassy Terrain|[from] ability: Grassy Surge"
            "|[of] p2a: Rillaboom",
            "|detailschange|p1a: Blastoise|Blastoise-Mega, L50, M",
            "|-mega|p1a: Blastoise|Blastoise|Blastoisinite",
            *(
                ["|-fieldstart|move: Trick Room|[of] p1b: Farigiraf"]
                if trick_room
                else []
            ),
            "|turn|3",
        ]
    )


def _helping_hand(battle):
    for target in (-1, -2, 0):
        try:
            return move_action(battle, 1, "helpinghand", target)
        except AssertionError:
            continue
    raise AssertionError("no Helping Hand action")


def test_ice_beam_removes_the_rillaboom_before_its_wood_hammer():
    battle = _turn()
    helping_hand = _helping_hand(battle)
    ice_beam = move_action(battle, 0, "icebeam", 1)
    out, report = run(
        G.guard_threat_first,
        battle,
        [
            ((move_action(battle, 0, "waterspout", 0), helping_hand), 0.40),
            ((move_action(battle, 0, "waterpulse", 2), helping_hand), 0.20),
            ((ice_beam, helping_hand), 0.10),
        ],
    )
    assert out[0].actions == (ice_beam, helping_hand)
    assert out[0].prob == 0.40
    assert report.demotions["threat_first:promoted"] == 1


def test_nothing_changes_when_the_top_pair_already_knocks_it_out():
    battle = _turn()
    helping_hand = _helping_hand(battle)
    pairs = [
        ((move_action(battle, 0, "icebeam", 1), helping_hand), 0.40),
        ((move_action(battle, 0, "waterspout", 0), helping_hand), 0.30),
    ]
    out, report = run(G.guard_threat_first, battle, pairs)
    assert out[0].actions == pairs[0][0] and not report.stages


def test_nothing_changes_without_trick_room():
    """Outside the room Rillaboom may outspeed Blastoise: no sure pre-emption."""
    battle = _turn(trick_room=False)
    helping_hand = _helping_hand(battle)
    pairs = [
        ((move_action(battle, 0, "waterspout", 0), helping_hand), 0.40),
        ((move_action(battle, 0, "icebeam", 1), helping_hand), 0.10),
    ]
    out, report = run(G.guard_threat_first, battle, pairs)
    assert out[0].actions == pairs[0][0] and not report.stages


def test_nothing_changes_when_ice_beam_would_not_knock_it_out():
    battle = _turn(rillaboom_hp="100/100")
    helping_hand = _helping_hand(battle)
    pairs = [
        ((move_action(battle, 0, "waterspout", 0), helping_hand), 0.40),
        ((move_action(battle, 0, "icebeam", 1), helping_hand), 0.10),
    ]
    out, report = run(G.guard_threat_first, battle, pairs)
    assert out[0].actions == pairs[0][0] and not report.stages


def test_the_partner_action_is_never_traded_away():
    """Only a pair that keeps Farigiraf's action may be promoted."""
    battle = _turn()
    helping_hand = _helping_hand(battle)
    pairs = [
        ((move_action(battle, 0, "waterspout", 0), helping_hand), 0.40),
        (
            (
                move_action(battle, 0, "icebeam", 1),
                move_action(battle, 1, "psychic", 2),
            ),
            0.10,
        ),
    ]
    out, report = run(G.guard_threat_first, battle, pairs)
    assert out[0].actions == pairs[0][0] and not report.stages


def test_a_threatened_pokemon_that_switches_out_is_left_alone():
    battle = _turn()
    helping_hand = _helping_hand(battle)
    pairs = [
        ((switch_action(battle, "incineroar"), helping_hand), 0.40),
        ((move_action(battle, 0, "icebeam", 1), helping_hand), 0.10),
    ]
    out, report = run(G.guard_threat_first, battle, pairs)
    assert out[0].actions == pairs[0][0] and not report.stages
