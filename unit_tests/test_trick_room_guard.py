"""trick_room_direction (the user, 2026-09-24, ladder game 2): set, keep or
reverse Trick Room by whom it helps -- counter a heavy Trick Room team's room
with our own when our side is faster, never hand them the room, reverse theirs,
keep ours. Positions are rebuilt with poke-env's parser (team preview included)
and our sets from teams/candidates_mc/T6.txt."""

from __future__ import annotations

import logging
from pathlib import Path

from poke_env.battle import DoubleBattle, Move
from poke_env.data import to_id_str
from poke_env.teambuilder import Teambuilder

from vgc_bench.src import guards as G
from vgc_bench.src import trick_room_guard as T

ROOT = Path(__file__).resolve().parents[1]
SETS = {
    to_id_str(mon.species or mon.nickname): mon
    for mon in Teambuilder.parse_showdown_team(
        (ROOT / "teams/candidates_mc/T6.txt").read_text()
    )
}
OURS = ["Blastoise", "Farigiraf", "Charizard", "Venusaur", "Torkoal", "Incineroar"]
TR_TEAM = ["Indeedee-F", "Hatterene", "Farigiraf", "Camerupt", "Incineroar", "Gallade"]
FAST_TEAM = ["Garchomp", "Talonflame", "Rillaboom", "Sneasler", "Kingambit", "Milotic"]


def _battle(theirs: list[str], lines: list[str]) -> DoubleBattle:
    battle = DoubleBattle(
        "battle-gen9championsvgc2026regmc-fixture",
        "antonius1",
        logging.getLogger("trick_room_guard"),
        gen=9,
    )
    battle._player_role = "p1"
    header = ["|player|p1|antonius1|1|1100", "|player|p2|rival|1|1100", "|gen|9"]
    header += [f"|poke|p1|{name}, L50|" for name in OURS]
    header += [f"|poke|p2|{name}, L50|" for name in theirs]
    header += ["|teampreview|4", "|start"]
    for line in [*header, *lines]:
        battle.parse_message(line.split("|"))
    for mon in battle.team.values():
        mon._update_from_teambuilder(SETS[to_id_str(mon.base_species)])
    return battle


def _action(battle: DoubleBattle, pos: int, move_id: str, target: int) -> int:
    for action in range(7, 27):
        order = G._decode(battle, action, pos)
        move = getattr(order, "order", None)
        if (
            isinstance(move, Move)
            and move.id == move_id
            and getattr(order, "move_target", None) == target
        ):
            return action
    raise AssertionError(f"no action for {move_id} -> {target}")


def _run(battle, pairs):
    cands = [G.Candidate(actions, prob) for actions, prob in pairs]
    report = G.GuardReport()
    return T.guard_trick_room_direction(battle, cands, report), report


GAME2_TURN1 = [
    "|switch|p1a: Blastoise|Blastoise, L50, M|100/100",
    "|switch|p1b: Farigiraf|Farigiraf, L50, M|100/100",
    "|switch|p2a: Farigiraf|Farigiraf, L50, M|100/100",
    "|switch|p2b: Incineroar|Incineroar, L50, M|100/100",
    "|turn|1",
]


def test_heavy_trick_room_rosters_are_recognised():
    assert T.heavy_trick_room(_battle(TR_TEAM, GAME2_TURN1))
    assert not T.heavy_trick_room(_battle(FAST_TEAM, GAME2_TURN1))


def test_counter_their_room_when_we_are_faster_and_their_setter_is_in():
    battle = _battle(TR_TEAM, GAME2_TURN1)
    edge, known = T.trick_room_edge(battle)
    assert known == 4 and edge == -1.0  # both of ours outspeed both of theirs
    spout = _action(battle, 0, "waterspout", 0)
    psychic = _action(battle, 1, "psychic", 1)
    trick_room = _action(battle, 1, "trickroom", 0)
    out, report = _run(battle, [((spout, psychic), 0.40), ((spout, trick_room), 0.20)])
    assert out[0].actions == (spout, trick_room)
    assert report.demotions["trick_room_direction:counter"] == 1


def test_never_hand_them_the_room_without_a_setter_in():
    battle = _battle(
        TR_TEAM,
        [
            "|switch|p1a: Blastoise|Blastoise, L50, M|100/100",
            "|switch|p1b: Farigiraf|Farigiraf, L50, M|100/100",
            "|switch|p2a: Camerupt|Camerupt, L50, M|100/100",
            "|switch|p2b: Incineroar|Incineroar, L50, M|100/100",
            "|turn|3",
        ],
    )
    spout = _action(battle, 0, "waterspout", 0)
    trick_room = _action(battle, 1, "trickroom", 0)
    psychic = _action(battle, 1, "psychic", 2)
    top = G.Candidate((spout, trick_room), 0.5)
    attack = G.Candidate((spout, psychic), 0.3)
    report = G.GuardReport()
    out = T.guard_trick_room_direction(battle, [top, attack], report)
    assert out[0] is attack and top.demoted_by == "trick_room_direction"
    assert report.demotions["trick_room_direction:no_gift"] == 1


def test_reverse_their_room_when_it_favours_them():
    battle = _battle(
        TR_TEAM,
        [*GAME2_TURN1, "|-fieldstart|move: Trick Room|[of] p2a: Farigiraf", "|turn|2"],
    )
    spout = _action(battle, 0, "waterspout", 0)
    psychic = _action(battle, 1, "psychic", 1)
    trick_room = _action(battle, 1, "trickroom", 0)
    out, report = _run(battle, [((spout, psychic), 0.40), ((spout, trick_room), 0.10)])
    assert out[0].actions == (spout, trick_room)
    assert report.demotions["trick_room_direction:reverse"] == 1


def test_keep_our_room_when_it_favours_us():
    battle = _battle(
        FAST_TEAM,
        [
            "|switch|p1a: Torkoal|Torkoal, L50, M|100/100",
            "|switch|p1b: Farigiraf|Farigiraf, L50, M|100/100",
            "|switch|p2a: Garchomp|Garchomp, L50, M|100/100",
            "|switch|p2b: Talonflame|Talonflame, L50, M|100/100",
            "|turn|1",
            "|-fieldstart|move: Trick Room|[of] p1b: Farigiraf",
            "|turn|2",
        ],
    )
    edge, _ = T.trick_room_edge(battle)
    assert edge == 1.0
    eruption = _action(battle, 0, "eruption", 0)
    trick_room = _action(battle, 1, "trickroom", 0)
    psychic = _action(battle, 1, "psychic", 1)
    undo = G.Candidate((eruption, trick_room), 0.5)
    keep = G.Candidate((eruption, psychic), 0.3)
    out, report = _run_cands(battle, [undo, keep])
    assert out[0] is keep and undo.demoted_by == "trick_room_direction"
    assert report.demotions["trick_room_direction:keep_ours"] == 1


def test_no_action_when_speeds_cannot_be_ordered():
    # Not a Trick Room roster: their base-60 Incineroar may run 72-123 Speed,
    # overlapping our Farigiraf (80) and Mega Blastoise (88).
    battle = _battle(
        FAST_TEAM + ["Incineroar"],
        [
            "|switch|p1a: Blastoise|Blastoise, L50, M|100/100",
            "|switch|p1b: Farigiraf|Farigiraf, L50, M|100/100",
            "|switch|p2a: Incineroar|Incineroar, L50, M|100/100",
            "|turn|1",
        ],
    )
    spout = _action(battle, 0, "waterspout", 0)
    trick_room = _action(battle, 1, "trickroom", 0)
    psychic = _action(battle, 1, "psychic", 1)
    top = G.Candidate((spout, trick_room), 0.5)
    out, report = _run_cands(battle, [top, G.Candidate((spout, psychic), 0.3)])
    assert out[0] is top and not report.stages


def test_pairs_without_trick_room_are_untouched():
    battle = _battle(TR_TEAM, GAME2_TURN1)
    spout = _action(battle, 0, "waterspout", 0)
    psychic = _action(battle, 1, "psychic", 1)
    top = G.Candidate((spout, psychic), 0.5)
    out, report = _run_cands(battle, [top])
    assert out == [top] and not report.stages


def _run_cands(battle, cands):
    report = G.GuardReport()
    return T.guard_trick_room_direction(battle, cands, report), report
