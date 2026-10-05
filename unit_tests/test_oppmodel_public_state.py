"""Event-fed public state: snapshot timing, HP rewrite, live/offline parity."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from vgc_bench.src.oppmodel import events as E
from vgc_bench.src.oppmodel import public_state as P

ROOT = Path(__file__).resolve().parents[1]

HEADER = """|j|☆Alice
|j|☆Bob
|gametype|doubles
|player|p1|Alice|1|1300
|player|p2|Bob|2|1250
|gen|9
|tier|[Gen 9 Champions] VGC 2026 Reg M-C
|rated|
|clearpoke
|poke|p1|Incineroar, L50, M|
|poke|p1|Rillaboom, L50, F|
|poke|p1|Charizard, L50, M|
|poke|p1|Farigiraf, L50, F|
|poke|p1|Torkoal, L50, M|
|poke|p1|Venusaur, L50, M|
|poke|p2|Garchomp, L50, M|
|poke|p2|Sneasler, L50, M|
|poke|p2|Politoed, L50, M|
|poke|p2|Archaludon, L50, M|
|poke|p2|Indeedee-F, L50, F|
|poke|p2|Pawmot, L50, M|
|teampreview|4
|teamsize|p1|4
|teamsize|p2|4
|start
|switch|p1a: Incineroar|Incineroar, L50, M|100/100
|switch|p1b: Rillaboom|Rillaboom, L50, F|100/100
|switch|p2a: Garchomp|Garchomp, L50, M|100/100
|switch|p2b: Sneasler|Sneasler, L50, M|100/100
"""


def drive(body: str, header: str = HEADER) -> P.DriveResult:
    return P.drive_log(header + body, "battle-test-1")


def snapshots(body: str, header: str = HEADER) -> dict[int, P.PublicSnapshot]:
    return {record.turn: record.snapshot for record in drive(body, header).turns}


def mon(snapshot: P.PublicSnapshot, side: str, species: str) -> P.MonPublic:
    return next(m for m in snapshot.sides[side].mons if m.species == species)


# --- HP rewrite ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("exact", "public"),
    [
        ("186/186", "100/100"),
        ("1/186", "1/100"),  # never rounds down to 0
        ("1/400", "1/100"),
        ("185/186", "99/100"),  # floor, not round
        ("40/200", "20/100r"),  # exactly a fifth: red
        ("41/200", "20/100y"),  # just above a fifth, same percent: yellow
        ("100/200", "50/100y"),  # exactly half: yellow
        ("101/200", "50/100g"),  # just above half, same percent: green
        ("93/186 par", "50/100y par"),
        ("150/186 brn", "80/100 brn"),
        ("0 fnt", "0 fnt"),
        ("50/100g", "50/100g"),  # already public: untouched
        ("garbage", "garbage"),
    ],
)
def test_public_condition(exact: str, public: str):
    assert P.public_condition(exact) == public


def _simulator_public(hp: int, maxhp: int) -> str:
    """Port of ``getHealth`` (pokemon-showdown/sim/pokemon.ts), Champions branch."""
    if not hp:
        return "0 fnt"
    percentage = (100 * hp) // maxhp or 1
    shared = f"{percentage}/100"
    if percentage == 20:
        shared += "y" if hp * 5 > maxhp else "r"
    elif percentage == 50:
        shared += "g" if hp * 2 > maxhp else "y"
    return shared


def test_public_condition_matches_the_simulator_for_every_hp_value():
    checked = 0
    for maxhp in range(1, 521):
        for hp in range(1, maxhp + 1):
            assert P.public_condition(f"{hp}/{maxhp}") == _simulator_public(hp, maxhp)
            checked += 1
    assert checked == 520 * 521 // 2
    for maxhp in (131, 186, 207, 362):
        for hp in range(1, maxhp + 1):
            expected = _simulator_public(hp, maxhp) + " tox"
            assert P.public_condition(f"{hp}/{maxhp} tox") == expected
    # The rule of every other format (round up, 99 below full) must not pass.
    assert P.public_condition("1/186") == "1/100"
    assert P.public_condition("185/186") == "99/100"
    assert P.public_condition("92/186") == "49/100"


def test_parse_condition_handles_colour_and_status_suffixes():
    assert P.parse_condition("50/100g") == (0.5, None, False)
    assert P.parse_condition("20/100y par") == (0.2, "par", False)
    assert P.parse_condition("0 fnt") == (0.0, None, True)
    assert P.parse_condition("93/186 slp") == (0.5, "slp", False)
    assert P.parse_condition("what") is None


def test_rewrite_event_touches_every_condition_field_of_one_side_only():
    cases = [
        (["", "switch", "p2a: X", "Garchomp, L50, M", "183/183"], 4, "100/100"),
        (["", "drag", "p2b: X", "Garchomp, L50, M", "91/183 brn"], 4, "49/100 brn"),
        (["", "replace", "p2a: X", "Zoroark, L50, M", "50/183"], 4, "27/100"),
        (["", "-damage", "p2a: X", "37/183", "[from] item: Life Orb"], 3, "20/100y"),
        (
            ["", "-heal", "p2: X", "92/183", "[from] move: Revival Blessing"],
            3,
            "50/100g",
        ),
        (["", "-sethp", "p2a: X", "1/183", "[from] move: Pain Split"], 3, "1/100"),
    ]
    for event, index, expected in cases:
        assert P.rewrite_event(event, "p2")[index] == expected
        assert P.rewrite_event(event, "p1") == event
    untouched = ["", "move", "p2a: X", "Protect", "p2a: X"]
    assert P.rewrite_event(untouched, "p2") == untouched
    assert set(P.CONDITION_INDEX) == {
        "switch",
        "drag",
        "replace",
        "-damage",
        "-heal",
        "-sethp",
    }


# --- snapshot content and timing ----------------------------------------------

TWO_TURNS = """|turn|1
|detailschange|p2a: Garchomp|Garchomp-Mega, L50, M
|-mega|p2a: Garchomp|Garchomp|Garchompite
|move|p1a: Incineroar|Fake Out|p2b: Sneasler
|-damage|p2b: Sneasler|88/100
|cant|p2b: Sneasler|flinch
|move|p1b: Rillaboom|Protect|p1b: Rillaboom
|-singleturn|p1b: Rillaboom|Protect
|move|p2a: Garchomp|Earthquake|p1a: Incineroar|[spread] p1a,p1b
|-activate|p1b: Rillaboom|move: Protect
|-damage|p1a: Incineroar|41/100
|-enditem|p1a: Incineroar|Sitrus Berry|[eat]
|-heal|p1a: Incineroar|66/100|[from] item: Sitrus Berry
|upkeep
|turn|2
|move|p1a: Incineroar|Parting Shot|p2a: Garchomp
|-unboost|p2a: Garchomp|atk|1
|-unboost|p2a: Garchomp|spa|1
|switch|p1a: Charizard|Charizard, L50, M|100/100|[from] Parting Shot
|move|p1b: Rillaboom|Wood Hammer|p2b: Sneasler
|-damage|p2b: Sneasler|0 fnt
|faint|p2b: Sneasler
|move|p2a: Garchomp|Trick Room|p2a: Garchomp
|-fieldstart|move: Trick Room|[of] p2a: Garchomp
|upkeep
|switch|p2b: Politoed|Politoed, L50, M|100/100
|-weather|RainDance|[from] ability: Drizzle|[of] p2b: Politoed
|turn|3
|win|Alice
"""


def test_snapshot_at_turn_n_holds_nothing_from_turn_n():
    turn = snapshots(TWO_TURNS)
    first = turn[1]
    assert first.turn == 1
    assert mon(first, "p2", "sneasler").hp == 1.0
    assert not mon(first, "p2", "garchomp").is_mega
    assert not first.sides["p2"].mega_used
    assert mon(first, "p1", "incineroar").item_state == P.ITEM_UNKNOWN
    assert mon(first, "p1", "incineroar").moves == ()
    assert mon(first, "p1", "incineroar").last_action is None
    second = turn[2]
    assert mon(second, "p2", "sneasler").hp == 0.88
    assert mon(second, "p1", "incineroar").hp == 0.66
    assert second.fields == {} and second.weather is None
    assert mon(second, "p2", "garchomp").boosts["atk"] == 0
    third = turn[3]
    assert mon(third, "p2", "garchomp").boosts["atk"] == -1
    assert third.fields == {"trickroom": P.EffectStart(2, False)}


def test_snapshot_taken_at_turn_event_is_frozen():
    public = P.PublicBattle("x")
    for event in E.split_log(HEADER + "|turn|1\n"):
        public.feed(event)
    taken = public.turn_snapshot
    assert taken is not None and public.at_turn_start
    before = taken.to_dict()
    for event in E.split_log("|inactive|Time left: 55 sec\n|\n"):
        public.feed(event)
    assert public.at_turn_start  # chatter does not end the turn start
    for event in E.split_log("|move|p1a: Incineroar|Fake Out|p2b: Sneasler\n"):
        public.feed(event)
    assert not public.at_turn_start
    for event in E.split_log("|-damage|p2b: Sneasler|50/100\n"):
        public.feed(event)
    assert taken.to_dict() == before
    assert public.snapshot().sides["p2"].mons[1].hp == 0.5


def test_mega_item_consumed_item_and_moves_revealed():
    turn = snapshots(TWO_TURNS)
    garchomp = mon(turn[2], "p2", "garchomp")
    assert garchomp.is_mega and garchomp.forme == "garchompmega"
    assert (garchomp.item, garchomp.item_state) == ("garchompite", P.ITEM_KNOWN)
    assert turn[2].sides["p2"].mega_used
    assert [m.id for m in garchomp.moves] == ["earthquake"]
    assert garchomp.moves[0].revealed and not garchomp.moves[0].from_sheet
    incineroar = mon(turn[2], "p1", "incineroar")
    assert (incineroar.item, incineroar.item_state) == ("sitrusberry", P.ITEM_CONSUMED)
    # Mega possible: needs a Mega forme, an unused side Mega and no other item.
    assert mon(turn[1], "p2", "garchomp").mega_possible
    assert not mon(turn[1], "p2", "sneasler").mega_possible
    assert mon(turn[1], "p1", "charizard").mega_possible
    assert not mon(turn[2], "p2", "garchomp").mega_possible
    assert mon(turn[2], "p1", "charizard").mega_possible


def test_protected_last_turn_by_move_and_stall_counter_also_by_side_guards():
    turn = snapshots(
        TWO_TURNS.split("|turn|2")[0]
        + """|turn|2
|move|p1b: Rillaboom|Protect||[still]
|-fail|p1b: Rillaboom
|move|p1a: Incineroar|Wide Guard|p1a: Incineroar
|-singleturn|p1a: Incineroar|Wide Guard
|move|p2a: Garchomp|Endure|p2a: Garchomp
|-singleturn|p2a: Garchomp|move: Endure
|move|p2b: Sneasler|Detect|p2b: Sneasler
|-singleturn|p2b: Sneasler|Protect
|upkeep
|turn|3
|move|p2b: Sneasler|Spiky Shield|p2b: Sneasler
|-singleturn|p2b: Sneasler|move: Protect
|move|p1a: Incineroar|Quick Guard|p1a: Incineroar
|-singleturn|p1a: Incineroar|Quick Guard
|move|p1b: Rillaboom|Crafty Shield|p1b: Rillaboom
|-singleturn|p1b: Rillaboom|Crafty Shield
|upkeep
|turn|4
|move|p1a: Incineroar|Protect|p1a: Incineroar
|-singleturn|p1a: Incineroar|Protect
|upkeep
|turn|5
"""
    )
    rillaboom = mon(turn[2], "p1", "rillaboom")
    assert rillaboom.protected_last_turn and rillaboom.protect_streak == 1
    assert not mon(turn[2], "p1", "incineroar").protected_last_turn
    at_three = turn[3]
    # The repeat failed: no protection, and the streak is over.
    assert not mon(at_three, "p1", "rillaboom").protected_last_turn
    assert mon(at_three, "p1", "rillaboom").protect_streak == 0
    # A side guard is not a Protect, but it raises the simulator's stall
    # counter: a Protect right after it succeeds one time in three.
    assert not mon(at_three, "p1", "incineroar").protected_last_turn
    assert mon(at_three, "p1", "incineroar").protect_streak == 1
    # Endure stalls (streak) but does not protect.
    assert not mon(at_three, "p2", "garchomp").protected_last_turn
    assert mon(at_three, "p2", "garchomp").protect_streak == 1
    assert mon(at_three, "p2", "sneasler").protected_last_turn
    assert mon(turn[4], "p2", "sneasler").protect_streak == 2
    assert mon(turn[4], "p2", "garchomp").protect_streak == 0
    # Two side guards in a row count twice; the guard against status moves
    # does not touch the counter.
    assert mon(turn[4], "p1", "incineroar").protect_streak == 2
    assert not mon(turn[4], "p1", "incineroar").protected_last_turn
    assert mon(turn[4], "p1", "rillaboom").protect_streak == 0
    assert mon(turn[5], "p1", "incineroar").protect_streak == 3
    assert mon(turn[5], "p1", "incineroar").protected_last_turn


def test_turns_on_field_first_turn_and_bring_state():
    turn = snapshots(TWO_TURNS)
    lead = mon(turn[1], "p1", "incineroar")
    assert lead.first_turn and lead.turns_on_field == 0 and lead.slot == "a"
    assert not mon(turn[2], "p1", "rillaboom").first_turn
    assert mon(turn[2], "p1", "rillaboom").turns_on_field == 1
    third = turn[3]
    assert mon(third, "p1", "charizard").first_turn  # came in by a pivot
    assert mon(third, "p2", "politoed").first_turn  # end-of-turn replacement
    benched = mon(third, "p1", "incineroar")
    assert benched.slot is None and not benched.first_turn
    assert benched.turns_on_field == 0 and benched.revealed
    side = third.sides["p1"]
    assert (side.n_revealed, side.n_unknown_brought, side.team_size) == (3, 1, 4)
    assert side.active == {"a": 2, "b": 1}
    assert third.sides["p2"].n_fainted == 1
    assert (
        mon(third, "p2", "sneasler").fainted and mon(third, "p2", "sneasler").hp == 0.0
    )
    assert mon(third, "p2", "sneasler").slot is None
    assert mon(third, "p1", "torkoal").brought is None  # not known yet
    assert (
        mon(third, "p1", "torkoal").hp == 1.0
        and not mon(third, "p1", "torkoal").revealed
    )
    assert [m.species for m in side.mons][:3] == [
        "incineroar",
        "rillaboom",
        "charizard",
    ]


def test_all_four_seen_settles_who_was_brought():
    turn = snapshots(
        """|turn|1
|switch|p1a: Charizard|Charizard, L50, M|100/100
|switch|p1b: Farigiraf|Farigiraf, L50, F|100/100
|upkeep
|turn|2
"""
    )
    side = turn[2].sides["p1"]
    assert side.n_revealed == 4 and side.n_unknown_brought == 0
    assert mon(turn[2], "p1", "torkoal").brought is False
    assert mon(turn[2], "p1", "incineroar").brought is True


def test_effect_starts_record_turn_and_upkeep():
    turn = snapshots(
        """|-weather|SunnyDay|[from] ability: Drought|[of] p1a: Incineroar
|turn|1
|move|p2a: Garchomp|Tailwind|p2a: Garchomp
|-sidestart|p2: Bob|move: Tailwind
|move|p1a: Incineroar|Rain Dance|p1a: Incineroar
|-weather|RainDance
|-weather|RainDance|[upkeep]
|upkeep
|turn|2
|-weather|RainDance|[upkeep]
|upkeep
|-fieldstart|move: Psychic Terrain|[from] ability: Psychic Surge|[of] p2b: Sneasler
|turn|3
|-fieldstart|move: Grassy Terrain|[from] ability: Grassy Surge|[of] p1b: Rillaboom
|-sideend|p2: Bob|move: Tailwind
|-weather|none
|upkeep
|turn|4
"""
    )
    first = turn[1]
    assert first.weather == "sunnyday" and first.weather_side == "p1"
    assert first.weather_start == P.EffectStart(0, True)
    second = turn[2]
    assert second.weather == "raindance" and second.weather_side == "p1"
    assert second.weather_start == P.EffectStart(1, False)
    assert second.sides["p2"].conditions == {"tailwind": P.EffectStart(1, False)}
    assert second.sides["p1"].conditions == {}
    third = turn[3]
    assert third.weather_start == P.EffectStart(1, False)  # upkeep lines do not restart
    assert third.fields == {"psychicterrain": P.EffectStart(2, True)}
    # Set in turn 1 it has been through two end-of-turn ticks; set after turn 2's
    # upkeep the terrain has been through none.
    assert third.weather_start is not None and third.weather_start.ticks(3) == 2
    assert third.fields["psychicterrain"].ticks(3) == 0
    fourth = turn[4]
    assert fourth.fields == {"grassyterrain": P.EffectStart(3, False)}
    assert fourth.weather is None and fourth.sides["p2"].conditions == {}


def test_volatiles_come_from_start_and_end_lines_only():
    turn = snapshots(
        """|turn|1
|move|p2b: Sneasler|Helping Hand|p2a: Garchomp
|-singleturn|p2a: Garchomp|Helping Hand|[of] p2b: Sneasler
|move|p1a: Incineroar|Taunt|p2a: Garchomp
|-start|p2a: Garchomp|move: Taunt
|move|p1b: Rillaboom|Perish Song|p1b: Rillaboom
|-start|p1b: Rillaboom|perish3|[silent]
|-start|p2b: Sneasler|perish3|[silent]
|-start|p2b: Sneasler|perish3
|upkeep
|turn|2
|-start|p2b: Sneasler|perish2
|-end|p2a: Garchomp|move: Taunt
|move|p1a: Incineroar|Double Shock|p2a: Garchomp
|-start|p1a: Incineroar|typechange|???/Fire|[from] move: Double Shock
|upkeep
|turn|3
"""
    )
    second = turn[2]
    # A single-turn effect (Helping Hand) must not linger into the next turn.
    assert mon(second, "p2", "garchomp").volatiles == {"taunt": 1}
    assert mon(second, "p2", "sneasler").volatiles == {"perish3": 1}
    third = turn[3]
    assert mon(third, "p2", "garchomp").volatiles == {}
    assert mon(third, "p2", "sneasler").volatiles == {"perish2": 2}
    assert mon(third, "p1", "incineroar").types == ("fire",)
    assert mon(second, "p1", "incineroar").types == ("fire", "dark")


def test_last_action_is_the_previous_turn_only():
    turn = snapshots(TWO_TURNS)
    assert mon(turn[2], "p1", "incineroar").last_action is not None
    fake_out = mon(turn[2], "p1", "incineroar").last_action
    assert fake_out is not None
    assert (fake_out.move, fake_out.target) == ("fakeout", "foe_b")
    flinched = mon(turn[2], "p2", "sneasler").last_action
    assert flinched is not None and flinched.kind == "hidden"
    third = turn[3]
    pivoted = mon(third, "p1", "incineroar").last_action
    assert pivoted is not None and pivoted.move == "partingshot"
    assert mon(third, "p1", "charizard").last_action is None
    assert mon(third, "p2", "politoed").last_action is None


def test_boost_passing_and_copyboost_direction():
    turn = snapshots(
        """|turn|1
|move|p1a: Incineroar|Swords Dance|p1a: Incineroar
|-boost|p1a: Incineroar|atk|2
|move|p2a: Garchomp|Psych Up|p1a: Incineroar
|-copyboost|p2a: Garchomp|p1a: Incineroar|[from] move: Psych Up
|upkeep
|turn|2
|move|p1a: Incineroar|Baton Pass|p1a: Incineroar
|switch|p1a: Charizard|Charizard, L50, M|100/100|[from] Baton Pass
|move|p2a: Garchomp|U-turn|p1b: Rillaboom
|switch|p2a: Politoed|Politoed, L50, M|100/100|[from] U-turn
|upkeep
|turn|3
"""
    )
    # The Psych Up user takes the other Pokemon's stages, not the reverse.
    assert mon(turn[2], "p2", "garchomp").boosts["atk"] == 2
    assert mon(turn[2], "p1", "incineroar").boosts["atk"] == 2
    third = turn[3]
    assert mon(third, "p1", "charizard").boosts["atk"] == 2  # passed on
    assert mon(third, "p1", "incineroar").boosts["atk"] == 0
    assert (
        mon(third, "p2", "politoed").boosts["atk"] == 0
    )  # a plain pivot passes nothing
    assert mon(third, "p2", "garchomp").boosts["atk"] == 0


def test_revived_pokemon_is_no_longer_fainted():
    turn = snapshots(
        """|turn|1
|move|p2a: Garchomp|Earthquake|p1a: Incineroar|[spread] p1a,p1b
|-damage|p1a: Incineroar|0 fnt
|faint|p1a: Incineroar
|upkeep
|switch|p1a: Charizard|Charizard, L50, M|100/100
|turn|2
|move|p1a: Charizard|Revival Blessing|p1a: Charizard
|-heal|p1: Incineroar|50/100y|[from] move: Revival Blessing
|upkeep
|turn|3
"""
    )
    assert mon(turn[2], "p1", "incineroar").fainted
    assert turn[2].sides["p1"].n_fainted == 1
    revived = mon(turn[3], "p1", "incineroar")
    assert not revived.fainted and revived.hp == 0.5 and revived.slot is None
    assert turn[3].sides["p1"].n_fainted == 0


def test_abilities_and_items_revealed_by_holder():
    turn = snapshots(
        """|-ability|p1a: Incineroar|Intimidate|boost
|-unboost|p2a: Garchomp|atk|1
|turn|1
|move|p1b: Rillaboom|Fake Out||[still]
|cant|p2a: Garchomp|ability: Armor Tail|Fake Out|[of] p1b: Rillaboom
|move|p2b: Sneasler|Close Combat|p1a: Incineroar
|-damage|p1a: Incineroar|40/100
|-damage|p2b: Sneasler|84/100|[from] item: Rocky Helmet|[of] p1a: Incineroar
|-heal|p2a: Garchomp|100/100|[from] item: Leftovers
|upkeep
|turn|2
"""
    )
    second = turn[2]
    assert mon(second, "p1", "incineroar").ability == "intimidate"
    assert mon(second, "p1", "incineroar").ability_known
    # |cant| with [of]: the subject holds the ability, the [of] Pokemon was blocked.
    assert mon(second, "p2", "garchomp").ability == "armortail"
    assert mon(second, "p1", "rillaboom").ability is None
    assert not mon(second, "p1", "rillaboom").ability_known
    assert mon(second, "p1", "incineroar").item == "rockyhelmet"
    assert mon(second, "p2", "sneasler").item is None
    assert mon(second, "p2", "garchomp").item == "leftovers"
    assert [m.id for m in mon(second, "p1", "rillaboom").moves] == ["fakeout"]


# --- sheets -------------------------------------------------------------------

SHEET_P2 = (
    "|showteam|p2|Garchomp||Garchompite|RoughSkin|Earthquake,DragonClaw,Protect,"
    "RockSlide|Jolly||M|||50|]Sneasler||FocusSash|Unburden|CloseCombat,DireClaw,"
    "FakeOut,Protect|Jolly||M|||50|]Politoed||Leftovers|Drizzle|WeatherBall,"
    "Protect,IcyWind,Encore|Calm||M|||50|"
)


def test_open_sheet_marks_moves_from_sheet_and_revealed():
    with_sheet = HEADER.replace("|teamsize|p1|4", SHEET_P2 + "\n|teamsize|p1|4")
    result = drive(TWO_TURNS, with_sheet)
    assert result.sheets == {"p1": False, "p2": True}
    second = result.turns[1].snapshot
    assert second.sides["p2"].sheet_open and not second.sides["p1"].sheet_open
    garchomp = mon(second, "p2", "garchomp")
    assert [(m.id, m.revealed, m.from_sheet) for m in garchomp.moves] == [
        ("earthquake", True, True),
        ("dragonclaw", False, True),
        ("protect", False, True),
        ("rockslide", False, True),
    ]
    politoed = mon(second, "p2", "politoed")
    assert not politoed.revealed and len(politoed.moves) == 4
    assert (politoed.item, politoed.item_state) == ("leftovers", P.ITEM_KNOWN)
    assert politoed.ability == "drizzle" and politoed.ability_known
    first = result.turns[0].snapshot
    assert mon(first, "p2", "garchomp").mega_possible  # the sheet shows its stone
    assert not mon(first, "p2", "politoed").mega_possible
    sneasler = mon(first, "p2", "sneasler")
    assert (sneasler.item, sneasler.item_state) == ("focussash", P.ITEM_KNOWN)
    # A Pokemon with no sheet entry stays unknown.
    assert mon(first, "p2", "archaludon").moves == ()
    assert mon(first, "p2", "archaludon").item_state == P.ITEM_UNKNOWN


def test_feed_sheet_equals_a_showteam_line():
    with_sheet = HEADER.replace("|teamsize|p1|4", SHEET_P2 + "\n|teamsize|p1|4")
    offline = drive(TWO_TURNS, with_sheet)
    fake = SimpleNamespace(
        _replay_data=E.split_log(HEADER + TWO_TURNS),
        player_role="p1",
        battle_tag="battle-test-1",
    )
    shadow = P.LiveShadow()
    shadow.feed_sheet("p2", E.parse_showteam(SHEET_P2.split("|", 3)[3]))
    shadow.mark_sheets_known()  # the other side showed none
    stream = fake._replay_data
    for end in range(1, len(stream) + 1):
        fake._replay_data = stream[:end]
        if stream[end - 1][1] == "turn":
            snapshot = shadow.sync(fake)
            assert snapshot is not None
            expected = offline.turns[snapshot.turn - 1].snapshot
            assert snapshot.to_dict() == expected.to_dict()


# --- drive_log ----------------------------------------------------------------


def test_drive_log_result_fields():
    result = drive(TWO_TURNS)
    assert result.usable and result.skip_reason is None
    assert result.players == {"p1": "Alice", "p2": "Bob"}
    assert result.ratings == {"p1": 1300, "p2": 1250}
    assert result.format_id == "gen9championsvgc2026regmc"
    assert result.winner == "p1" and result.finished and not result.tie
    assert [record.turn for record in result.turns] == [1, 2, 3]
    actions = result.turns[0].actions
    assert set(actions) == {"p1a", "p1b", "p2a", "p2b"}
    assert actions["p2a"].mega and actions["p2a"].move == "earthquake"
    assert actions["p1b"].is_protect
    assert result.turns[2].actions["p2b"].reason == E.REASON_UNRESOLVED
    assert result.best_of is None
    assert isinstance(result.turns[0].snapshot.to_dict(), dict)


def test_ratings_survive_trailing_empty_player_lines():
    result = drive("|turn|1\n|player|p1|\n|player|p2|\n|win|Bob\n")
    assert result.ratings == {"p1": 1300, "p2": 1250}
    assert result.players == {"p1": "Alice", "p2": "Bob"}
    assert result.winner == "p2"
    snapshot = result.turns[0].snapshot
    assert snapshot.sides["p1"].rating == 1300 and snapshot.sides["p2"].rating == 1250
    unrated = P.drive_log(
        HEADER.replace("|1300", "|").replace("|1250", "") + "|turn|1\n"
    )
    assert unrated.ratings == {"p1": None, "p2": None}


def test_skip_reasons_are_counted_not_raised():
    no_turn = drive("|win|Alice\n")
    assert no_turn.skip_reason == P.SKIP_NO_TURN and not no_turn.usable
    assert no_turn.counters["skip:no_turn_1"] == 1
    illusion = drive(
        "|turn|1\n", HEADER.replace("Pawmot, L50, M", "Zoroark-Hisui, L50, M")
    )
    assert illusion.skip_reason == P.SKIP_ILLUSION
    assert illusion.counters["skip:illusion_species"] == 1
    singles = drive(
        "|turn|1\n", HEADER.replace("|gametype|doubles", "|gametype|singles")
    )
    assert singles.skip_reason == P.SKIP_NOT_DOUBLES
    empty = P.drive_log("", "nothing")
    assert empty.skip_reason == P.SKIP_NO_TURN and empty.turns == []


def test_best_of_three_marker():
    line = (
        '|uhtml|bestof|<h2><strong>Game 2</strong> of <a href="/game-bestof3-'
        'gen9championsvgc2026regmcbo3-2679348520">a best-of-3</a></h2>\n'
    )
    result = drive("|turn|1\n", line + HEADER)
    assert (result.best_of, result.game_number) == (3, 2)
    assert result.series_id == "game-bestof3-gen9championsvgc2026regmcbo3-2679348520"
    assert result.turns[0].snapshot.best_of == 3


def test_feed_never_raises_on_garbage():
    public = P.PublicBattle("x")
    garbage = [
        None,
        42,
        "a string",
        [],
        [""],
        ["", None],
        ["", "switch"],
        ["", "switch", "nonsense", "???"],
        ["", "switch", "p1a: X", "Garchomp, L50", "not/a/number"],
        ["", "-damage", "p1a: Nobody", "x/y"],
        ["", "-boost", "p1a: X", "atk", "many"],
        ["", "turn", "soon"],
        ["", "turn"],
        ["", "teamsize", "p1", "four"],
        ["", "-mega"],
        ["", "move", "p3z: ?"],
        ["", "showteam", "p9", "]]]|||"],
        ["", "-weather"],
        ["", "win"],
        ["", "faint", "p2b: Ghost"],
        ["", "move", None, 3],
        ("", "-damage", "p1a: X", b"50/100"),
    ]
    for event in garbage:
        public.feed(event)  # type: ignore[arg-type]
    public.feed(["", "turn", "1"])
    assert public.turn == 1 and public.turn_snapshot is not None
    assert sum(v for k, v in public.counters.items() if k.startswith("feed_error")) >= 8
    assert P.drive_log("|turn|x\n|move\n|switch|p1a\n|\x00|\n").turns == []
    shadow = P.LiveShadow()
    assert shadow.sync(None) is None
    assert (
        shadow.sync(SimpleNamespace(_replay_data="no", player_role=7, battle_tag=3))
        is None
    )
    assert shadow.counters["stand_down:no_stream"] == 2


# --- LiveShadow ---------------------------------------------------------------


def as_player_view(text: str, role: str, max_hp: int = 183) -> str:
    """A spectator log as ``role`` would receive it: its own HP exact.

    The exact value is found with this file's port of the simulator's formula,
    not with the library's rule, so the seat tests check that rule as well.
    """
    lines = []
    for event in E.split_log(text):
        index = P.CONDITION_INDEX.get(event[1])
        if index is not None and len(event) > index and event[2].startswith(role):
            parsed = P.parse_condition(event[index])
            if parsed is not None and not parsed[2]:
                shown = event[index].split()[0].rstrip("gyr")
                current = next(
                    hp
                    for hp in range(1, max_hp + 1)
                    if _simulator_public(hp, max_hp).rstrip("gyr") == shown
                )
                status = f" {parsed[1]}" if parsed[1] else ""
                event[index] = f"{current}/{max_hp}{status}"
        lines.append("|".join(event))
    return "\n".join(lines)


def live(
    text: str, role: str, tag: str = "battle-test-1", sheets_known: bool = True
) -> dict[int, P.PublicSnapshot]:
    """Snapshots a bot seated ``role`` gets, syncing once per decision.

    ``sheets_known``: the runtime has said that no sheet was shown, which is
    what a spectator log without a showteam line says by itself.
    """
    stream = E.split_log(text)
    fake = SimpleNamespace(_replay_data=[], player_role=role, battle_tag=tag)
    shadow = P.LiveShadow()
    if sheets_known:
        shadow.mark_sheets_known()
    out: dict[int, P.PublicSnapshot] = {}
    for event in stream:
        fake._replay_data.append(event)
        if event[1] == "turn":
            snapshot = shadow.sync(fake)
            assert snapshot is not None
            out[snapshot.turn] = snapshot
    assert not any(key.startswith("sync_error") for key in shadow.counters)
    return out


@pytest.mark.parametrize("role", ["p1", "p2"])
def test_live_shadow_equals_offline_for_either_seat(role: str):
    spectator = HEADER + TWO_TURNS
    player_view = as_player_view(spectator, role)
    assert "/183" in player_view and "/183" not in spectator
    offline = {
        r.turn: r.snapshot for r in P.drive_log(spectator, "battle-test-1").turns
    }
    got = live(player_view, role)
    assert set(got) == set(offline) == {1, 2, 3}
    for number, snapshot in got.items():
        assert snapshot.to_dict() == offline[number].to_dict()
    hp = {m.hp for s in got.values() for side in s.sides.values() for m in side.mons}
    assert all(abs(value * 100 - round(value * 100)) < 1e-9 for value in hp)


def test_live_shadow_stands_down_mid_turn_and_before_turn_one():
    stream = E.split_log(HEADER + TWO_TURNS)
    fake = SimpleNamespace(
        _replay_data=[], player_role="p1", battle_tag="battle-test-1"
    )
    shadow = P.LiveShadow()
    first_turn = next(i for i, event in enumerate(stream) if event[1] == "turn")
    fake._replay_data = stream[:first_turn]
    assert shadow.sync(fake) is None
    assert shadow.counters["stand_down:no_turn_yet"] == 1
    fake._replay_data = stream[: first_turn + 1] + [
        ["", "inactive", "Time left"],
        ["", ""],
    ]
    at_start = shadow.sync(fake)
    assert at_start is not None and at_start.turn == 1
    assert shadow.sync(fake) is at_start  # nothing new: same answer
    fake._replay_data = fake._replay_data + stream[first_turn + 1 : first_turn + 4]
    assert shadow.sync(fake) is None  # a forced-switch request, not a turn start
    assert shadow.counters["stand_down:mid_turn"] == 1


def test_live_shadow_waits_for_the_role_before_any_hp_line():
    stream = E.split_log(as_player_view(HEADER + TWO_TURNS, "p2"))
    fake = SimpleNamespace(
        _replay_data=stream, player_role=None, battle_tag="battle-test-1"
    )
    shadow = P.LiveShadow()
    assert shadow.sync(fake) is None
    assert shadow.counters["stand_down:role_unknown"] == 1
    fake.player_role = "p2"
    snapshot = shadow.sync(fake)
    # The whole stream is past the last turn line (the game is over): stand down,
    # but the state behind it is complete and public.
    assert snapshot is None
    final = shadow.public.snapshot()
    assert all(
        abs(m.hp * 100 - round(m.hp * 100)) < 1e-9 for m in final.sides["p2"].mons
    )


def test_reconnect_shorter_stream_rebuilds():
    text = as_player_view(HEADER + TWO_TURNS, "p2")
    expected = live(text, "p2", sheets_known=False)
    stream = E.split_log(text)
    turns = [i for i, event in enumerate(stream) if event[1] == "turn"]
    fake = SimpleNamespace(
        _replay_data=stream[: turns[1] + 1],
        player_role="p2",
        battle_tag="battle-test-1",
    )
    shadow = P.LiveShadow()
    second = shadow.sync(fake)
    assert second is not None and second.turn == 2
    # A new battle object after a rejoin: the stream starts over, shorter.
    fake._replay_data = stream[: turns[0] + 1]
    again = shadow.sync(fake)
    assert again is not None and again.turn == 1
    assert again.to_dict() == expected[1].to_dict()
    assert shadow.counters["rebuild:stream_shorter_than_cursor"] == 1
    fake._replay_data = stream[: turns[2] + 1]
    third = shadow.sync(fake)
    assert third is not None and third.to_dict() == expected[3].to_dict()


@pytest.mark.parametrize("with_init", [True, False])
def test_reconnect_replayed_log_in_the_same_stream_rebuilds(with_init: bool):
    preamble = "|init|battle\n|title|Alice vs. Bob\n" if with_init else ""
    text = as_player_view(preamble + HEADER + TWO_TURNS, "p1")
    expected = live(text, "p1", sheets_known=False)
    stream = E.split_log(text)
    turns = [i for i, event in enumerate(stream) if event[1] == "turn"]
    fake = SimpleNamespace(
        _replay_data=[], player_role="p1", battle_tag="battle-test-1"
    )
    shadow = P.LiveShadow()
    fake._replay_data = stream[: turns[1] + 1]
    assert shadow.sync(fake) is not None
    # The server replays the whole log into the same list, then play goes on.
    fake._replay_data = stream[: turns[1] + 1] + stream[: turns[1] + 1]
    replayed = shadow.sync(fake)
    assert replayed is not None and replayed.to_dict() == expected[2].to_dict()
    assert shadow.counters["rebuild:replayed_log"] == 1
    assert len(replayed.sides["p1"].mons) == 6  # the roster was not doubled
    fake._replay_data = fake._replay_data + stream[turns[1] + 1 : turns[2] + 1]
    third = shadow.sync(fake)
    assert third is not None and third.to_dict() == expected[3].to_dict()


def test_turn_going_backwards_without_a_preamble_stands_down_until_a_preamble():
    stream = E.split_log(HEADER + TWO_TURNS)
    turns = [i for i, event in enumerate(stream) if event[1] == "turn"]
    fake = SimpleNamespace(
        _replay_data=stream[: turns[2] + 1] + [["", "turn", "2"]],
        player_role="p1",
        battle_tag="battle-test-1",
    )
    shadow = P.LiveShadow()
    # The rebuilt state has no roster, no ratings and no team size: no forecast.
    assert shadow.sync(fake) is None
    assert shadow.counters["rebuild_without_preamble"] == 1
    assert shadow.counters["stand_down:degraded"] == 1
    assert len(shadow.public.snapshot().sides["p1"].mons) < 6
    fake._replay_data = fake._replay_data + [["", "upkeep"], ["", "turn", "3"]]
    assert shadow.sync(fake) is None
    assert shadow.counters["stand_down:degraded"] == 2
    # A replay WITH its preamble repairs it.
    fake._replay_data = fake._replay_data + stream[: turns[1] + 1]
    repaired = shadow.sync(fake)
    expected = P.drive_log(HEADER + TWO_TURNS, "battle-test-1", sheets_known=False)
    assert repaired is not None
    assert repaired.to_dict() == expected.turns[1].snapshot.to_dict()
    assert shadow.counters["stand_down:degraded"] == 2


def test_repeated_turn_line_makes_a_log_unusable_by_name():
    clean = drive(TWO_TURNS)
    assert clean.usable and "turn_not_increasing" not in clean.counters
    doubled = drive(TWO_TURNS.replace("|turn|3\n", "|turn|3\n|turn|3\n"))
    assert doubled.skip_reason == P.SKIP_TURN_ORDER and not doubled.usable
    assert doubled.counters["skip:turn_not_increasing"] == 1
    backwards = drive(TWO_TURNS.replace("|turn|3\n", "|turn|1\n"))
    assert backwards.skip_reason == P.SKIP_TURN_ORDER


def test_sheets_survive_a_rebuild():
    stream = E.split_log(HEADER + TWO_TURNS)
    turns = [i for i, event in enumerate(stream) if event[1] == "turn"]
    fake = SimpleNamespace(
        _replay_data=stream[: turns[1] + 1],
        player_role="p1",
        battle_tag="battle-test-1",
    )
    shadow = P.LiveShadow()
    shadow.feed_sheet("p2", E.parse_showteam(SHEET_P2.split("|", 3)[3]))
    shadow.mark_sheets_known()
    assert shadow.sync(fake) is not None
    fake._replay_data = stream[: turns[0] + 1]
    rebuilt = shadow.sync(fake)
    assert shadow.counters["rebuild:stream_shorter_than_cursor"] == 1
    assert rebuilt is not None and rebuilt.sides["p2"].sheet_open
    assert len(mon(rebuilt, "p2", "politoed").moves) == 4
    # What was said about the other side is kept as well.
    assert rebuilt.sides["p1"].sheet_open is False


# --- rules behind the main state features ---------------------------------------

IN_AND_OUT = """|turn|1
|switch|p1a: Charizard|Charizard, L50, M|100/100
|move|p1b: Rillaboom|Wood Hammer|p2a: Garchomp
|upkeep
|turn|2
|switch|p1a: Incineroar|Incineroar, L50, M|100/100
|move|p1b: Rillaboom|Wood Hammer|p2a: Garchomp
|upkeep
|turn|3
|move|p1a: Incineroar|Fake Out|p2a: Garchomp
|upkeep
|turn|4
"""


def test_first_turn_and_turns_on_field_reset_on_re_entry():
    turn = snapshots(IN_AND_OUT)
    assert mon(turn[1], "p1", "incineroar").first_turn
    back = mon(turn[3], "p1", "incineroar")
    assert back.slot == "a" and back.first_turn and back.turns_on_field == 0
    assert not mon(turn[4], "p1", "incineroar").first_turn
    assert mon(turn[4], "p1", "incineroar").turns_on_field == 1
    assert mon(turn[3], "p1", "rillaboom").turns_on_field == 2


def test_last_action_is_dropped_after_one_turn():
    turn = snapshots(IN_AND_OUT)
    left = mon(turn[2], "p1", "incineroar").last_action
    assert left is not None and left.kind == "switch"
    # It re-entered during turn 2 and did nothing that turn: no previous action.
    assert mon(turn[3], "p1", "incineroar").last_action is None
    assert mon(turn[4], "p1", "charizard").last_action is None


def test_protect_streak_restarts_after_a_gap():
    turn = snapshots(
        """|turn|1
|move|p1b: Rillaboom|Protect|p1b: Rillaboom
|-singleturn|p1b: Rillaboom|Protect
|upkeep
|turn|2
|move|p1b: Rillaboom|Wood Hammer|p2a: Garchomp
|upkeep
|turn|3
|move|p1b: Rillaboom|Protect|p1b: Rillaboom
|-singleturn|p1b: Rillaboom|Protect
|upkeep
|turn|4
"""
    )
    assert mon(turn[4], "p1", "rillaboom").protect_streak == 1
    assert mon(turn[4], "p1", "rillaboom").protected_last_turn


def test_mega_possible_ends_for_the_whole_side_once_one_has_mega_evolved():
    turn = snapshots(
        """|turn|1
|switch|p1a: Charizard|Charizard, L50, M|100/100
|upkeep
|turn|2
|detailschange|p1a: Charizard|Charizard-Mega-Y, L50, M
|-mega|p1a: Charizard|Charizard|Charizardite Y
|upkeep
|turn|3
"""
    )
    assert mon(turn[2], "p1", "venusaur").mega_possible
    assert not mon(turn[3], "p1", "venusaur").mega_possible
    assert not mon(turn[3], "p1", "charizard").mega_possible
    assert mon(turn[3], "p2", "garchomp").mega_possible  # the other side is untouched


def test_mega_possible_needs_no_other_item():
    turn = snapshots(
        """|turn|1
|switch|p1a: Charizard|Charizard, L50, M|100/100
|-heal|p1a: Charizard|100/100|[from] item: Leftovers
|-enditem|p2a: Garchomp|Sitrus Berry|[eat]
|upkeep
|turn|2
"""
    )
    assert mon(turn[2], "p1", "charizard").item == "leftovers"
    assert not mon(turn[2], "p1", "charizard").mega_possible
    assert not mon(turn[2], "p2", "garchomp").mega_possible  # item consumed


def test_mega_evolution_clears_the_base_ability():
    body = """|-ability|p2a: Garchomp|Rough Skin
|turn|1
|detailschange|p2a: Garchomp|Garchomp-Mega, L50, M
|-mega|p2a: Garchomp|Garchomp|Garchompite
|upkeep
|turn|2
"""
    turn = snapshots(body)
    assert mon(turn[1], "p2", "garchomp").ability == "roughskin"
    assert mon(turn[2], "p2", "garchomp").ability is None
    assert not mon(turn[2], "p2", "garchomp").ability_known
    with_sheet = HEADER.replace("|teamsize|p1|4", SHEET_P2 + "\n|teamsize|p1|4")
    sheet = snapshots(body.split("\n", 1)[1], with_sheet)
    assert mon(sheet[1], "p2", "garchomp").ability == "roughskin"
    assert mon(sheet[2], "p2", "garchomp").ability is None
    # A Mega line with no forme line before it is enough.
    bare = snapshots(
        body.replace("|detailschange|p2a: Garchomp|Garchomp-Mega, L50, M\n", "")
    )
    assert mon(bare[2], "p2", "garchomp").is_mega
    assert mon(bare[2], "p2", "garchomp").ability is None
    assert bare[2].sides["p2"].mega_used


def test_field_end_status_and_cure_lines():
    turn = snapshots(
        """|turn|1
|move|p2a: Garchomp|Trick Room|p2a: Garchomp
|-fieldstart|move: Trick Room|[of] p2a: Garchomp
|move|p1a: Incineroar|Will-O-Wisp|p2a: Garchomp
|-status|p2a: Garchomp|brn
|upkeep
|turn|2
|-curestatus|p2a: Garchomp|brn|[msg]
|-fieldend|move: Trick Room
|upkeep
|turn|3
"""
    )
    assert turn[2].fields == {"trickroom": P.EffectStart(1, False)}
    assert mon(turn[2], "p2", "garchomp").status == "brn"
    assert turn[3].fields == {}
    assert mon(turn[3], "p2", "garchomp").status is None


def test_cureteam_clears_the_status_of_the_whole_side():
    turn = snapshots(
        """|turn|1
|-status|p1a: Incineroar|brn
|-status|p1b: Rillaboom|par
|-status|p2a: Garchomp|slp
|upkeep
|turn|2
|switch|p1b: Charizard|Charizard, L50, M|100/100
|-cureteam|p1a: Incineroar|[from] move: Heal Bell
|upkeep
|turn|3
"""
    )
    assert mon(turn[2], "p1", "rillaboom").status == "par"
    third = turn[3]
    assert mon(third, "p1", "incineroar").status is None
    assert mon(third, "p1", "rillaboom").status is None  # on the bench
    assert mon(third, "p2", "garchomp").status == "slp"


def test_boost_line_family():
    turn = snapshots(
        """|turn|1
|-setboost|p1a: Incineroar|atk|6|[from] move: Belly Drum
|-boost|p2a: Garchomp|atk|2
|-unboost|p2a: Garchomp|def|1
|-boost|p2b: Sneasler|spe|1
|-unboost|p2b: Sneasler|def|1
|-boost|p1b: Rillaboom|spa|1
|-unboost|p1b: Rillaboom|spd|2
|upkeep
|turn|2
|-swapboost|p1a: Incineroar|p2a: Garchomp|atk, spa|[from] move: Power Swap
|-clearnegativeboost|p2b: Sneasler|[silent]
|-invertboost|p1b: Rillaboom|[from] move: Topsy-Turvy
|upkeep
|turn|3
|-clearboost|p1a: Incineroar
|-clearpositiveboost|p2b: Sneasler|p1a: Incineroar|move: Spectral Thief
|upkeep
|turn|4
|-clearallboost
|upkeep
|turn|5
"""
    )
    assert mon(turn[2], "p1", "incineroar").boosts["atk"] == 6
    third = turn[3]
    assert mon(third, "p1", "incineroar").boosts["atk"] == 2
    assert mon(third, "p2", "garchomp").boosts["atk"] == 6
    assert mon(third, "p2", "garchomp").boosts["def"] == -1  # not in the swap list
    assert mon(third, "p2", "sneasler").boosts["def"] == 0
    assert mon(third, "p2", "sneasler").boosts["spe"] == 1
    assert mon(third, "p1", "rillaboom").boosts["spa"] == -1
    assert mon(third, "p1", "rillaboom").boosts["spd"] == 2
    fourth = turn[4]
    assert mon(fourth, "p1", "incineroar").boosts["atk"] == 0
    assert mon(fourth, "p2", "sneasler").boosts["spe"] == 0
    assert mon(fourth, "p2", "garchomp").boosts["atk"] == 6
    assert set(mon(turn[5], "p2", "garchomp").boosts.values()) == {0}
    assert set(mon(turn[5], "p1", "rillaboom").boosts.values()) == {0}


def test_court_change_swaps_side_conditions():
    turn = snapshots(
        """|turn|1
|move|p2a: Garchomp|Tailwind|p2a: Garchomp
|-sidestart|p2: Bob|move: Tailwind
|-swapsideconditions
|upkeep
|turn|2
"""
    )
    assert turn[2].sides["p1"].conditions == {"tailwind": P.EffectStart(1, False)}
    assert turn[2].sides["p2"].conditions == {}


def test_volatiles_do_not_survive_leaving_the_field():
    turn = snapshots(
        """|turn|1
|move|p1a: Incineroar|Taunt|p2a: Garchomp
|-start|p2a: Garchomp|move: Taunt
|upkeep
|turn|2
|switch|p2a: Politoed|Politoed, L50, M|100/100
|upkeep
|turn|3
|switch|p2a: Garchomp|Garchomp, L50, M|100/100
|upkeep
|turn|4
"""
    )
    assert mon(turn[2], "p2", "garchomp").volatiles == {"taunt": 1}
    assert mon(turn[4], "p2", "garchomp").slot == "a"
    assert mon(turn[4], "p2", "garchomp").volatiles == {}


def test_charge_and_recharge_markers_last_one_turn():
    turn = snapshots(
        """|turn|1
|move|p2a: Garchomp|Solar Beam||[still]
|-prepare|p2a: Garchomp|Solar Beam
|move|p1a: Incineroar|Hyper Beam|p2b: Sneasler
|-mustrecharge|p1a: Incineroar
|upkeep
|turn|2
|move|p2a: Garchomp|Solar Beam|p1a: Incineroar|[from] lockedmove
|cant|p1a: Incineroar|recharge
|upkeep
|turn|3
"""
    )
    assert mon(turn[2], "p2", "garchomp").volatiles == {"prepare": 1}
    assert mon(turn[2], "p1", "incineroar").volatiles == {"mustrecharge": 1}
    assert mon(turn[3], "p2", "garchomp").volatiles == {}
    assert mon(turn[3], "p1", "incineroar").volatiles == {}


def test_revealed_moves_exclude_called_moves_and_struggle():
    turn = snapshots(
        """|turn|1
|move|p2b: Sneasler|Copycat|p2b: Sneasler
|move|p2b: Sneasler|Wood Hammer|p1a: Incineroar|[from] move: Copycat
|cant|p1b: Rillaboom|Disable|Wood Hammer
|move|p1a: Incineroar|Struggle|p2a: Garchomp
|upkeep
|turn|2
"""
    )
    assert [m.id for m in mon(turn[2], "p2", "sneasler").moves] == ["copycat"]
    assert [m.id for m in mon(turn[2], "p1", "rillaboom").moves] == ["woodhammer"]
    assert mon(turn[2], "p1", "incineroar").moves == ()


def test_drive_log_labels_carry_the_next_turn_backfill():
    result = drive(
        """|turn|1
|move|p2a: Garchomp|Solar Beam||[still]
|-prepare|p2a: Garchomp|Solar Beam
|upkeep
|turn|2
|move|p2a: Garchomp|Solar Beam|p1a: Incineroar|[from] lockedmove
|upkeep
|turn|3
"""
    )
    assert result.turns[0].actions["p2a"].target == "foe_a"
    # The serve-time previous-action feature cannot see the next turn.
    previous = mon(result.turns[1].snapshot, "p2", "garchomp").last_action
    assert previous is not None and previous.target is None


def test_typechange_copied_from_another_pokemon():
    turn = snapshots(
        """|turn|1
|move|p1a: Incineroar|Reflect Type|p2a: Garchomp
|-start|p1a: Incineroar|typechange|[from] move: Reflect Type|[of] p2a: Garchomp
|upkeep
|turn|2
"""
    )
    copied = mon(turn[2], "p1", "incineroar").types
    assert copied == mon(turn[2], "p2", "garchomp").types == ("dragon", "ground")


def test_repeated_sidestart_keeps_the_first_start():
    turn = snapshots(
        """|turn|1
|-sidestart|p2: Bob|Spikes
|upkeep
|turn|2
|-sidestart|p2: Bob|Spikes
|upkeep
|turn|3
"""
    )
    assert turn[3].sides["p2"].conditions == {"spikes": P.EffectStart(1, False)}


# --- abilities: who holds what ---------------------------------------------------

ABILITY_HEADER = (
    HEADER.replace("Incineroar", "Sinistcha")
    .replace("Rillaboom", "Blastoise")
    .replace("Charizard", "Delphox")
    .replace("Farigiraf", "Gardevoir")
    .replace("Sneasler", "Froslass")
    .replace("Politoed", "Weavile")
    .replace("Archaludon", "Vaporeon")
)


CURSED_BODY = (
    "|-start|p1a: Sinistcha|Disable|Rage Powder"
    "|[from] ability: Cursed Body|[of] p2b: Froslass\n"
)
PICKPOCKET = (
    "|-enditem|p1b: Blastoise|Sitrus Berry|[silent]"
    "|[from] ability: Pickpocket|[of] p1b: Blastoise\n"
    "|-item|p2b: Weavile|Sitrus Berry|[from] ability: Pickpocket|[of] p1b: Blastoise\n"
)


def test_ability_tag_with_of_goes_to_the_pokemon_that_can_have_it():
    body = (
        """|turn|1
|-heal|p1b: Blastoise|84/100|[from] ability: Hospitality|[of] p1a: Sinistcha
|-damage|p1b: Blastoise|70/100|[from] ability: Rough Skin|[of] p2a: Garchomp
"""
        + CURSED_BODY
        + """|upkeep
|turn|2
|switch|p1a: Delphox|Delphox, L50, M|100/100
|switch|p2b: Weavile|Weavile, L50, M|100/100
|-item|p1a: Delphox|Rocky Helmet|[from] ability: Magician|[of] p2a: Garchomp
"""
        + PICKPOCKET
        + """|upkeep
|turn|3
|switch|p2b: Vaporeon|Vaporeon, L50, M|100/100
|move|p2a: Garchomp|Surf|p1a: Delphox|[spread] p1a,p1b,p2b
|-heal|p2b: Vaporeon|100/100|[from] ability: Water Absorb|[of] p2a: Garchomp
|upkeep
|turn|4
"""
    )
    turn = snapshots(body, ABILITY_HEADER)
    second = turn[2]
    # A heal given to a partner: the giver is in [of].
    assert mon(second, "p1", "sinistcha").ability == "hospitality"
    assert mon(second, "p1", "blastoise").ability is None
    assert not mon(second, "p1", "blastoise").ability_known
    # The ability of the Pokemon that was hit disabled the attacker's move.
    assert mon(second, "p2", "froslass").ability == "cursedbody"
    assert mon(second, "p1", "sinistcha").ability == "hospitality"  # not overwritten
    assert mon(second, "p2", "garchomp").ability == "roughskin"
    third = turn[3]
    # The thief is the subject for one stealing ability and so is its victim
    # for a revealing one: the dex tells them apart.
    assert mon(third, "p1", "delphox").ability == "magician"
    assert mon(third, "p1", "delphox").item == "rockyhelmet"
    assert mon(third, "p2", "garchomp").ability == "roughskin"
    assert mon(third, "p2", "weavile").ability == "pickpocket"
    assert mon(third, "p2", "weavile").item == "sitrusberry"
    assert mon(third, "p1", "blastoise").ability is None
    assert mon(third, "p1", "blastoise").item_state == P.ITEM_CONSUMED
    # A heal a Pokemon draws from a partner's move: the healed one holds it.
    assert mon(turn[4], "p2", "vaporeon").ability == "waterabsorb"
    assert mon(turn[4], "p2", "garchomp").ability == "roughskin"


def test_ability_tag_defaults_when_the_dex_cannot_tell():
    turn = snapshots(
        """|-item|p2a: Garchomp|Sitrus Berry|[from] ability: Frisk|[of] p1a: Incineroar
|turn|1
|-heal|p1b: Rillaboom|90/100|[from] ability: Hospitality|[of] p1a: Incineroar
|-heal|p2b: Sneasler|90/100|[from] ability: Water Absorb|[of] p1a: Incineroar
|-start|p1b: Rillaboom|Disable|Protect|[from] ability: Cursed Body|[of] p2b: Sneasler
|detailschange|p2a: Garchomp|Garchomp-Mega, L50, M
|-mega|p2a: Garchomp|Garchomp|Garchompite
|-heal|p2a: Garchomp|100/100|[from] ability: Water Absorb|[of] p2b: Sneasler
|upkeep
|turn|2
"""
    )
    # Neither Pokemon lists the ability (a copied one, or a dex gap).
    garchomp = mon(turn[1], "p2", "garchomp")
    assert (garchomp.item, garchomp.item_state) == ("sitrusberry", P.ITEM_KNOWN)
    assert mon(turn[1], "p1", "incineroar").ability == "frisk"
    assert garchomp.ability is None
    second = turn[2]
    # Same side: the giver in [of]. Other side: the healed Pokemon itself.
    assert mon(second, "p1", "incineroar").ability == "hospitality"
    assert mon(second, "p1", "rillaboom").ability is None
    # (the later line on the same Pokemon: the [of] Pokemon by default)
    assert mon(second, "p2", "sneasler").ability == "cursedbody"
    assert mon(second, "p1", "rillaboom").ability is None
    # A Mega forme the dex may be out of date for is not ruled out by it: the
    # partner, whose entry is complete, cannot have the ability, so the Mega
    # does (the same-side default alone would have picked the partner).
    assert mon(second, "p2", "garchomp").ability == "waterabsorb"


def test_ability_tag_naming_a_pokemon_that_cannot_be_found_reveals_nothing():
    turn = snapshots(
        """|turn|1
|-damage|p1a: Incineroar|80/100|[from] ability: Rough Skin|[of] p9z: Nobody
|-status|p1b: Rillaboom|par|[from] ability: Static|[of] ???
|-weather|RainDance|[from] ability: Drizzle
|-heal|p2b: Sneasler|90/100|[from] ability: Water Absorb|[of] p9z: Nobody
|-immune|p2a: Garchomp|[from] ability: Levitate
|upkeep
|turn|2
"""
    )
    second = turn[2]
    # The attacker must not be credited with the ability that hurt it.
    assert mon(second, "p1", "incineroar").ability is None
    assert mon(second, "p1", "rillaboom").ability is None
    assert mon(second, "p1", "rillaboom").status == "par"
    assert second.weather == "raindance"
    # Types on which the subject is the usual holder still reveal it.
    assert mon(second, "p2", "sneasler").ability == "waterabsorb"
    assert mon(second, "p2", "garchomp").ability == "levitate"


def test_copied_ability_lasts_only_while_on_the_field():
    body = """|switch|p1b: Gardevoir|Gardevoir, L50, F|100/100
|-ability|p1b: Gardevoir|Rough Skin|Trace|[from] ability: Trace|[of] p2a: Garchomp
|turn|1
|move|p1a: Sinistcha|Simple Beam|p2b: Froslass
|-ability|p2b: Froslass|Simple|Cursed Body|[from] move: Simple Beam
|upkeep
|turn|2
|switch|p1b: Blastoise|Blastoise, L50, F|100/100
|switch|p2b: Weavile|Weavile, L50, M|100/100
|upkeep
|turn|3
|switch|p1b: Gardevoir|Gardevoir, L50, F|100/100
|upkeep
|turn|4
"""
    turn = snapshots(body, ABILITY_HEADER)
    first = turn[1]
    assert mon(first, "p1", "gardevoir").ability == "roughskin"  # the copy
    assert mon(first, "p2", "garchomp").ability == "roughskin"  # where it came from
    assert mon(turn[2], "p2", "froslass").ability == "simple"
    third = turn[3]
    # On the bench each shows its own ability, which the change line named.
    assert mon(third, "p1", "gardevoir").ability == "trace"
    assert mon(third, "p2", "froslass").ability == "cursedbody"
    assert mon(third, "p2", "garchomp").ability == "roughskin"
    # Back in with nothing copied this time: its own ability, not a stale copy.
    fourth = turn[4]
    assert mon(fourth, "p1", "gardevoir").slot == "b"
    assert mon(fourth, "p1", "gardevoir").ability == "trace"


def test_exchanged_abilities_follow_the_pokemon_while_on_the_field():
    turn = snapshots(
        """|-ability|p1a: Incineroar|Intimidate|boost
|turn|1
|-activate|p1a: Incineroar|Skill Swap|Rough Skin|Intimidate|[of] p2a: Garchomp
|upkeep
|turn|2
|-activate|p1a: Incineroar|Skill Swap|||[of] p1b: Rillaboom
|upkeep
|turn|3
|switch|p1a: Charizard|Charizard, L50, M|100/100
|switch|p1b: Farigiraf|Farigiraf, L50, F|100/100
|-activate|p2b: Sneasler|move: Poltergeist|Sitrus Berry
|upkeep
|turn|4
"""
    )
    second = turn[2]
    assert mon(second, "p1", "incineroar").ability == "roughskin"
    assert mon(second, "p2", "garchomp").ability == "intimidate"
    third = turn[3]
    # Between partners the names are not shown: what each had moves across.
    assert mon(third, "p1", "rillaboom").ability == "roughskin"
    assert mon(third, "p1", "incineroar").ability is None
    assert not mon(third, "p1", "incineroar").ability_known
    fourth = turn[4]
    # Off the field each has its own again; the foe still stands with the swap.
    assert mon(fourth, "p1", "incineroar").ability == "intimidate"
    assert mon(fourth, "p1", "rillaboom").ability is None
    assert mon(fourth, "p2", "garchomp").ability == "intimidate"
    assert mon(fourth, "p2", "sneasler").ability is None


# --- sheets: open, closed, unknown ------------------------------------------------


def _sheet_p2() -> list[E.SheetSet]:
    return E.parse_showteam(SHEET_P2.split("|", 3)[3])


def test_sheet_state_is_unknown_on_a_player_view_until_the_caller_says():
    stream = E.split_log(HEADER + TWO_TURNS)
    first_turn = next(i for i, event in enumerate(stream) if event[1] == "turn")
    fake = SimpleNamespace(
        _replay_data=stream[: first_turn + 1],
        player_role="p1",
        battle_tag="battle-test-1",
    )
    shadow = P.LiveShadow()
    blind = shadow.sync(fake)
    assert blind is not None
    assert blind.sides["p1"].sheet_open is None and blind.sides["p2"].sheet_open is None
    shadow.mark_sheets_known()
    told = shadow.sync(fake)
    assert told is not None
    assert told.sides["p1"].sheet_open is False and told.sides["p2"].sheet_open is False
    # Offline the same three states, per side.
    spectator = P.drive_log(HEADER + TWO_TURNS, "x")
    assert spectator.sheets == {"p1": False, "p2": False}
    page = P.drive_log(HEADER + TWO_TURNS, "x", sheets_known=False)
    assert page.sheets == {"p1": None, "p2": None}
    assert page.turns[0].snapshot.sides["p2"].sheet_open is None
    recovered = P.drive_log(
        HEADER + TWO_TURNS, "x", sheets={"p2": _sheet_p2()}, sheets_known=False
    )
    assert recovered.sheets == {"p1": None, "p2": True}
    assert len(mon(recovered.turns[0].snapshot, "p2", "politoed").moves) == 4


def test_sheets_given_to_drive_log_equal_a_showteam_line():
    with_line = drive(
        TWO_TURNS, HEADER.replace("|teamsize|p1|4", SHEET_P2 + "\n|teamsize|p1|4")
    )
    given = P.drive_log(HEADER + TWO_TURNS, "battle-test-1", sheets={"p2": _sheet_p2()})
    assert given.sheets == with_line.sheets == {"p1": False, "p2": True}
    assert len(given.turns) == len(with_line.turns) == 3
    for ours, theirs in zip(given.turns, with_line.turns):
        assert ours.snapshot.to_dict() == theirs.snapshot.to_dict()


def test_sheet_given_at_the_turn_start_is_in_that_turns_snapshot():
    stream = E.split_log(HEADER + TWO_TURNS)
    first_turn = next(i for i, event in enumerate(stream) if event[1] == "turn")
    fake = SimpleNamespace(
        _replay_data=stream[: first_turn + 1],
        player_role="p1",
        battle_tag="battle-test-1",
    )
    shadow = P.LiveShadow()
    before = shadow.sync(fake)
    assert before is not None and before.sides["p2"].sheet_open is None
    # Handed over lazily, on the first forecast of the game.
    shadow.feed_sheet("p2", _sheet_p2())
    after = shadow.sync(fake)
    assert after is not None and after.turn == 1
    assert after.sides["p2"].sheet_open is True
    assert len(mon(after, "p2", "politoed").moves) == 4
    assert before.sides["p2"].sheet_open is None  # the old object is not edited
    # Mid-turn the turn-start snapshot is not rebuilt from mid-turn state.
    public = P.PublicBattle("x")
    for event in stream[: first_turn + 3]:
        public.feed(event)
    taken = public.turn_snapshot
    assert taken is not None and not public.at_turn_start
    public.feed_sheet("p2", _sheet_p2())
    assert public.turn_snapshot is taken


def test_sheet_does_not_survive_a_change_of_battle():
    stream = E.split_log(HEADER + TWO_TURNS)
    turns = [i for i, event in enumerate(stream) if event[1] == "turn"]
    shadow = P.LiveShadow()
    shadow.feed_sheet("p2", _sheet_p2())
    shadow.mark_sheets_known()
    first = shadow.sync(_fake(stream[: turns[0] + 1], "p1", "battle-a"))
    assert first is not None and first.sides["p2"].sheet_open is True
    assert first.sides["p1"].sheet_open is False
    other = shadow.sync(_fake(stream[: turns[1] + 1], "p1", "battle-b"))
    assert other is not None and shadow.counters["rebuild:battle_tag_changed"] == 1
    # Nothing said about battle A applies to battle B.
    assert other.sides["p2"].sheet_open is None
    assert other.sides["p1"].sheet_open is None
    assert mon(other, "p2", "politoed").moves == ()


# --- metadata -------------------------------------------------------------------


def test_player_ids_are_account_ids_and_players_are_display_names():
    result = drive("|turn|1\n")
    assert result.players == {"p1": "Alice", "p2": "Bob"}
    assert result.player_ids == {"p1": "alice", "p2": "bob"}
    spelled = drive("|turn|1\n", HEADER.replace("|Alice|", "|A_li ce|"))
    assert spelled.players["p1"] == "A_li ce"
    assert spelled.player_ids["p1"] == result.player_ids["p1"]
    assert P.drive_log("", "nothing").player_ids == {"p1": None, "p2": None}


def test_rated_means_a_ladder_game_and_the_note_is_kept():
    ladder = drive("|turn|1\n")
    assert ladder.rated_kind == P.RATED_LADDER and ladder.turns[0].snapshot.rated
    tournament = drive(
        "|turn|1\n", HEADER.replace("|rated|\n", "|rated|Tournament battle\n")
    )
    assert tournament.rated_kind == "tournamentbattle"
    assert not tournament.turns[0].snapshot.rated
    unrated = drive("|turn|1\n", HEADER.replace("|rated|\n", ""))
    assert unrated.rated_kind is None and not unrated.turns[0].snapshot.rated


def test_battle_times_come_from_the_time_lines():
    result = drive(
        "|t:|1759000100\n|turn|1\n|t:|1759000160\n|upkeep\n|t:|1759000200\n"
        "|turn|2\n|t:|not-a-number\n",
        "|t:|1759000000\n" + HEADER,
    )
    assert (result.start_time, result.end_time) == (1759000000, 1759000200)
    assert drive("|turn|1\n").start_time is None and drive("|turn|1\n").end_time is None
    # A time line is chatter: it does not end the turn start.
    public = P.PublicBattle("x")
    for event in E.split_log(HEADER + "|turn|1\n|t:|1759000100\n"):
        public.feed(event)
    assert public.at_turn_start and public.start_time == 1759000100


# --- robustness of the turn and live paths ----------------------------------------


def test_unknown_message_types_are_counted_and_do_not_cancel_a_turn_start():
    public = P.PublicBattle("x")
    for event in E.split_log(HEADER + TWO_TURNS.split("|turn|3")[0] + "|turn|3\n"):
        public.feed(event)
    # Every type of an ordinary log is known.
    assert not any(key.startswith("unhandled_kind") for key in public.counters)
    assert public.at_turn_start
    public.feed(["", "somenewservermessage", "hello"])
    assert public.at_turn_start
    assert public.counters["unhandled_kind:somenewservermessage"] == 1
    public.feed(["", "Weird Kind 0123456789 0123456789 0123456789", ""])
    assert public.at_turn_start and public.counters["unhandled_kind:?"] == 1
    # Shaped like a battle action: the turn has started, and it is still counted.
    public.feed(["", "-brandnewaction", "p1a: Incineroar"])
    assert not public.at_turn_start
    assert public.counters["unhandled_kind:-brandnewaction"] == 1
    # A known battle line that carries no state ends the turn start uncounted.
    fresh = P.PublicBattle("y")
    for event in E.split_log(HEADER + "|turn|1\n|-supereffective|p2a: Garchomp\n"):
        fresh.feed(event)
    assert not fresh.at_turn_start
    assert not any(key.startswith("unhandled_kind") for key in fresh.counters)
    # Live: the forecast of the turn survives the unknown room message.
    stream = E.split_log(HEADER + "|turn|1\n") + [["", "somenewservermessage", "x"]]
    shadow = P.LiveShadow()
    assert shadow.sync(_fake(stream, "p1", "battle-test-1")) is not None
    assert "stand_down:mid_turn" not in shadow.counters


def test_live_shadow_skips_an_entry_that_is_not_an_event_and_moves_on():
    stream = E.split_log(HEADER + TWO_TURNS)
    turns = [i for i, event in enumerate(stream) if event[1] == "turn"]
    expected = live(HEADER + TWO_TURNS, "p1")
    bad = [
        ["", "-damage", None, None],
        ["", ["turn"], "4"],
        None,
        7,
        ["", "switch", ["p1a: X"], "Foo", "1/1"],
        [""],
    ]
    cut = turns[0] + 3
    fake = _fake(stream[:cut] + bad + stream[cut : turns[1] + 1], "p1", "battle-test-1")
    shadow = P.LiveShadow()
    shadow.mark_sheets_known()
    second = shadow.sync(fake)
    assert second is not None and second.to_dict() == expected[2].to_dict()
    assert shadow.counters["sync_skip:not_an_event"] == len(bad)
    assert not any(key.startswith("sync_error") for key in shadow.counters)
    # The shadow carries on, and nothing was fed twice.
    fake._replay_data = fake._replay_data + stream[turns[1] + 1 : turns[2] + 1]
    third = shadow.sync(fake)
    assert third is not None and third.to_dict() == expected[3].to_dict()
    assert shadow.public.events_fed == turns[2] + 1


def test_failed_turn_snapshot_stands_down_instead_of_serving_the_old_turn(monkeypatch):
    stream = E.split_log(HEADER + TWO_TURNS)
    turns = [i for i, event in enumerate(stream) if event[1] == "turn"]
    original = P.PublicBattle.snapshot

    def fails_at_turn_two(self: P.PublicBattle) -> P.PublicSnapshot:
        if self.turn == 2:
            raise RuntimeError("no snapshot")
        return original(self)

    monkeypatch.setattr(P.PublicBattle, "snapshot", fails_at_turn_two)
    fake = _fake(stream[: turns[0] + 1], "p1", "battle-test-1")
    shadow = P.LiveShadow()
    first = shadow.sync(fake)
    assert first is not None and first.turn == 1
    fake._replay_data = stream[: turns[1] + 1]
    # The stream is at turn 2; the turn-1 snapshot must not be served for it.
    assert shadow.sync(fake) is None
    assert shadow.counters["stand_down:turn_unreadable"] == 1
    assert shadow.public.counters["feed_error:turn:RuntimeError"] == 1
    assert shadow.public.turn_snapshot is None and not shadow.public.at_turn_start
    fake._replay_data = stream[: turns[2] + 1]
    third = shadow.sync(fake)
    assert third is not None and third.turn == 3
    # Offline the turn is dropped by name and the others keep their own pairs.
    result = drive(TWO_TURNS)
    assert [record.turn for record in result.turns] == [1, 3]
    assert [record.snapshot.turn for record in result.turns] == [1, 3]
    assert result.counters["turn_without_snapshot"] == 1
    assert result.turns[1].actions["p2b"].species == "politoed"
    # A turn line that cannot be read leaves no turn start either.
    public = P.PublicBattle("x")
    for event in stream[: turns[0] + 1]:
        public.feed(event)
    assert public.at_turn_start
    public.feed(["", "turn", "soon"])
    assert public.turn_snapshot is None and not public.at_turn_start


def test_reader_failure_behind_the_previous_action_feature_is_counted(monkeypatch):
    def boom(segment: E.TurnSegment, counters: Counter[str] | None) -> None:
        raise ValueError("unreadable turn")

    monkeypatch.setattr(E, "_scan_turn", boom)
    result = drive(TWO_TURNS)
    # Two previous-turn reads by the tracker (at turns 2 and 3), three label reads.
    assert result.counters["last_action_error:ValueError"] == 2
    assert result.counters["reader_error:ValueError"] == 3
    previous = mon(result.turns[1].snapshot, "p1", "incineroar").last_action
    assert previous is not None and previous.reason == E.REASON_UNRESOLVED
    public = P.PublicBattle("x")
    for event in E.split_log(HEADER + TWO_TURNS):
        public.feed(event)
    assert public.counters["last_action_error:ValueError"] == 2


def test_missing_dex_is_a_named_skip_and_a_stand_down(monkeypatch):
    caches = (
        E.moves_dex,
        E.pokedex,
        E.base_species_id,
        E.is_mega_forme,
        P._mega_stones,
        P._forme_abilities,
        P._has_false_identity,
    )

    def clear() -> None:
        for cached in caches:
            cached.cache_clear()

    monkeypatch.setattr(E, "_static_file", lambda *parts: None)
    clear()
    try:
        assert not E.dex_available() and not E.dex_signature()["dex_available"]
        result = drive(TWO_TURNS)
        assert result.skip_reason == P.SKIP_DEX_MISSING and not result.usable
        assert result.counters["skip:dex_missing"] == 1
        shadow = P.LiveShadow()
        fake = _fake(E.split_log(HEADER + "|turn|1\n"), "p1", "battle-test-1")
        assert shadow.sync(fake) is None
        assert shadow.counters["stand_down:dex_missing"] == 1
    finally:
        monkeypatch.undo()
        clear()
    assert E.dex_available() and drive(TWO_TURNS).usable
    assert (
        E.is_protect_family("protect")
        and E.species_from_details("Indeedee-F") == "indeedee"
    )


def _fake(stream: list, role: str | None, tag: str) -> SimpleNamespace:
    return SimpleNamespace(_replay_data=stream, player_role=role, battle_tag=tag)


def test_live_shadow_rebuilds_for_another_battle_tag():
    first = E.split_log(HEADER + IN_AND_OUT)
    other_text = HEADER.replace("Garchomp", "Archaludon", 1).replace(
        "p2a: Garchomp|Garchomp", "p2a: Archaludon|Archaludon"
    ) + IN_AND_OUT.replace("Garchomp", "Archaludon")
    other = E.split_log(other_text)
    turns_a = [i for i, e in enumerate(first) if e[1] == "turn"]
    turns_b = [i for i, e in enumerate(other) if e[1] == "turn"]
    shadow = P.LiveShadow()
    assert shadow.sync(_fake(first[: turns_a[1] + 1], "p1", "battle-a")) is not None
    got = shadow.sync(_fake(other[: turns_b[2] + 1], "p1", "battle-b"))
    expected = P.drive_log(other_text, "battle-b", sheets_known=False).turns[2].snapshot
    assert got is not None and got.to_dict() == expected.to_dict()
    assert shadow.counters["rebuild:battle_tag_changed"] == 1


def test_live_shadow_rebuilds_when_the_role_changes():
    body = """|turn|1
|move|p1a: Incineroar|Flare Blitz|p2a: Garchomp
|-damage|p2a: Garchomp|161/183
|upkeep
|turn|2
"""
    player_view = E.split_log(HEADER + body)
    spectator = P.drive_log(
        HEADER + body.replace("161/183", "87/100"), "battle-a", sheets_known=False
    )
    shadow = P.LiveShadow()
    wrong = shadow.sync(_fake(player_view, "p1", "battle-a"))
    assert wrong is not None
    right = shadow.sync(_fake(player_view, "p2", "battle-a"))
    assert right is not None and shadow.counters["rebuild:role_changed"] == 1
    assert right.to_dict() == spectator.turns[1].snapshot.to_dict()


# --- the acceptance script's own checks --------------------------------------------


def test_parity_script_fails_an_empty_part_and_checks_the_hp_rule(monkeypatch):
    parity = pytest.importorskip("evaluation.oppmodel_parity")
    # A part that was asked for and compared nothing is not a pass.
    empty = {"argv": [], "c_view": {"compared": 0, "notes": {"no_cache": 1}}}
    assert parity.failures(empty) == ["c_view: compared nothing"]
    assert parity.failures({"d_pokeenv": {"compared": 0, "stats": {}}}) == [
        "d_pokeenv: compared nothing"
    ]
    clean = {"a_live": {"compared": 9, "stats": {"turns[per_turn]": 9}, "counters": {}}}
    assert parity.failures(clean) == []
    seat = {"b_corpus": {"compared": 3, "seat": {"seat_turns": 9, "seat_mismatch": 2}}}
    assert parity.failures(seat) == ["b_corpus: seat_mismatch = 2"]
    skipped = {
        "a_live": {"compared": 9, "counters": {"live[x]:sync_skip:not_an_event": 1}}
    }
    assert len(parity.failures(skipped)) == 1
    # Its port of the simulator's formula agrees with the library's rule.
    for hp, maxhp in (
        (1, 186),
        (37, 183),
        (40, 200),
        (41, 200),
        (100, 200),
        (101, 200),
    ):
        assert parity.simulator_public_hp(hp, maxhp) == P.public_condition(
            f"{hp}/{maxhp}"
        )
    # The synthetic player view carries exact HP that the real rewrite maps back
    # to the public string of the log ...
    events = E.split_log(HEADER + TWO_TURNS)
    always_140 = SimpleNamespace(choice=lambda options: 140)
    stats: Counter[str] = Counter()
    view = parity.synthetic_player_view(events, "p1", always_140, stats)
    assert stats["seat_tokens"] == 5 and not stats["rewrite_roundtrip_mismatch"]
    assert not stats["seat_token_not_reproducible"]
    assert ["", "-damage", "p1a: Incineroar", "58/140"] in view
    assert all(event[1] != "showteam" for event in view)
    live_view, shadow = parity.live_snapshots(
        view, "battle-test-1", "p1", sheets_known=True
    )
    offline = P.drive_log(HEADER + TWO_TURNS, "battle-test-1")
    assert [live_view[r.turn].to_dict() for r in offline.turns] == [
        r.snapshot.to_dict() for r in offline.turns
    ]
    # ... and a wrong rule (rounding up, as other formats do) is caught there.

    def round_up(event: list[str], role: str) -> list[str]:
        out = list(event)
        index = P.CONDITION_INDEX.get(out[1])
        if index is not None and "/140" in out[index]:
            hp = int(out[index].split("/")[0])
            out[index] = f"{-(-100 * hp // 140)}/100"
        return out

    monkeypatch.setattr(parity, "rewrite_event", round_up)
    wrong: Counter[str] = Counter()
    parity.synthetic_player_view(events, "p1", always_140, wrong)
    assert wrong["rewrite_roundtrip_mismatch"] >= 2


# --- real data (skipped when the git-ignored folders are absent) ---------------


def _own_replays(limit: int) -> list[Path]:
    pages: list[Path] = []
    for folder in sorted(ROOT.glob("ladder_replays_mc*")):
        if folder.is_dir():
            pages.extend(sorted(folder.glob("*regmc-*.html"))[:3])
    return pages[:limit]


def test_saved_own_replays_live_equals_offline():
    pages = _own_replays(24)
    if not pages:
        pytest.skip("no ladder_replays_mc* folders on this machine")
    compared = 0
    for page in pages:
        log = E.extract_log_from_html(
            page.read_text(encoding="utf-8", errors="replace")
        )
        assert log is not None
        account = page.name.split(" - battle-")[0]
        stream = E.split_log(log)
        role = next(e[2] for e in stream if e[1] == "player" and e[3] == account)
        tag = E.extract_battle_tag(log) or page.stem
        rewritten = "\n".join("|".join(P.rewrite_event(e, role)) for e in stream)
        offline = P.drive_log(rewritten, tag, sheets_known=False)
        assert not any(key.startswith("feed_error") for key in offline.counters)
        got = live(log, role, tag, sheets_known=False)
        assert len(got) == len(offline.turns)
        for record in offline.turns:
            assert got[record.turn].to_dict() == record.snapshot.to_dict()
            compared += 1
            bot_side = record.snapshot.sides[role]
            assert bot_side.sheet_open is None  # a saved page cannot tell
            for member in bot_side.mons:
                assert abs(member.hp * 100 - round(member.hp * 100)) < 1e-9
    assert compared > 50


def test_human_logs_drive_without_errors():
    folder = ROOT / "battle_logs_web_mc_20260927" / "low"
    if not folder.is_dir():
        pytest.skip("battle_logs_web_mc_20260927 is not on this machine")
    usable = 0
    for path in sorted(folder.glob("*.log"))[:60]:
        result = P.drive_log(
            path.read_text(encoding="utf-8", errors="replace"), path.stem
        )
        assert not any(
            key.startswith(
                ("feed_error", "drive_error", "last_action_error", "reader_error")
            )
            for key in result.counters
        ), (path.name, result.counters)
        if not result.usable:
            continue
        usable += 1
        assert result.ratings["p1"] is not None and result.ratings["p2"] is not None
        for record in result.turns:
            assert record.snapshot.turn == record.turn
            for side in record.snapshot.sides.values():
                assert len(side.mons) == 6
                assert 0 <= side.n_revealed <= 4
    assert usable > 40
