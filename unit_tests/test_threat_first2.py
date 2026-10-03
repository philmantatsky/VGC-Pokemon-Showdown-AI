"""threat_first2 (2026-10-03): the foe's Mega forme counts, and a first-turn Fake
Out answers a faster threat. Mega Raichu Y's Zap Cannon knocked out our Mega
Blastoise on turn 1 three times on ladder while threat_first judged the foe as its
base forme (no threat)."""

from __future__ import annotations

from unit_tests.ladder_position import move_action, position, run
from vgc_bench.src import threat_first2 as T2


def _turn1(extra: list[str] | None = None):
    return position(
        [
            "|switch|p1a: Blastoise|Blastoise, L50, M|186/186",
            "|switch|p1b: Farigiraf|Farigiraf, L50, F|227/227",
            "|switch|p2a: Raichu|Raichu, L50, F|100/100",
            "|switch|p2b: Gengar|Gengar, L50, M|100/100",
            *(extra or []),
            "|turn|1",
        ]
    )


def _trick_room(battle):
    for target in (0, -1, -2):
        try:
            return move_action(battle, 1, "trickroom", target)
        except AssertionError:
            continue
    raise AssertionError("no Trick Room action")


def test_the_unevolved_raichu_counts_as_mega_raichu_y():
    battle = _turn1()
    raichu = battle.opponent_active_pokemon[0]
    assert raichu is not None
    assert T2.likely_mega(battle, raichu) == "raichumegay"
    found = T2.threats(battle)
    assert found and found[0][1] is raichu and found[0][2] == 0  # it KOs Blastoise
    assert battle.opponent_team["p2: Raichu"] is raichu  # the swap was undone


def test_fake_out_the_raichu_instead_of_water_spout():
    battle = _turn1()
    trick_room = _trick_room(battle)
    fake_out = move_action(battle, 0, "fakeout", 1)
    out, report = run(
        T2.guard_threat_first2,
        battle,
        [
            ((move_action(battle, 0, "waterspout", 0), trick_room), 0.45),
            ((move_action(battle, 0, "fakeout", 2), trick_room), 0.20),
            ((fake_out, trick_room), 0.15),
        ],
    )
    assert out[0].actions == (fake_out, trick_room)
    assert report.demotions["threat_first2:promoted"] == 1


def test_a_fake_out_into_the_harmless_foe_moves_to_the_threat():
    battle = _turn1()
    trick_room = _trick_room(battle)
    fake_out = move_action(battle, 0, "fakeout", 1)
    out, _ = run(
        T2.guard_threat_first2,
        battle,
        [
            ((move_action(battle, 0, "fakeout", 2), trick_room), 0.45),
            ((fake_out, trick_room), 0.15),
        ],
    )
    assert out[0].actions == (fake_out, trick_room)


def test_no_fake_out_into_inner_focus():
    battle = _turn1(["|-ability|p2a: Raichu|Inner Focus"])
    trick_room = _trick_room(battle)
    pairs = [
        ((move_action(battle, 0, "waterspout", 0), trick_room), 0.45),
        ((move_action(battle, 0, "fakeout", 1), trick_room), 0.15),
    ]
    out, report = run(T2.guard_threat_first2, battle, pairs)
    assert out[0].actions == pairs[0][0] and not report.stages


def test_the_attack_answer_still_works():
    """threat_first's own case: Ice Beam on the Rillaboom under our room."""
    battle = position(
        [
            "|switch|p1a: Blastoise|Blastoise, L50, M|90/186",
            "|switch|p1b: Farigiraf|Farigiraf, L50, F|200/227",
            "|switch|p2a: Rillaboom|Rillaboom, L50, M|55/100",
            "|switch|p2b: Incineroar|Incineroar, L50, M|100/100",
            "|-fieldstart|move: Grassy Terrain|[from] ability: Grassy Surge"
            "|[of] p2a: Rillaboom",
            "|detailschange|p1a: Blastoise|Blastoise-Mega, L50, M",
            "|-mega|p1a: Blastoise|Blastoise|Blastoisinite",
            "|-fieldstart|move: Trick Room|[of] p1b: Farigiraf",
            "|turn|3",
        ]
    )
    partner = move_action(battle, 1, "psychic", 2)
    ice_beam = move_action(battle, 0, "icebeam", 1)
    out, _ = run(
        T2.guard_threat_first2,
        battle,
        [
            ((move_action(battle, 0, "waterspout", 0), partner), 0.40),
            ((ice_beam, partner), 0.10),
        ],
    )
    assert out[0].actions == (ice_beam, partner)
