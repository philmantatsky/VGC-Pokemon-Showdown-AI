"""evaluation/forme_stats_audit.py: the reader that picks a turn's clean hits out of
a saved log (the accuracy reading compares each with the calculator's range at the
start of the turn, so anything the calculator could not know then must be dropped)."""

from __future__ import annotations

import pytest

from evaluation.forme_stats_audit import Hit, _hp, accuracy, faints, read_hits, team_key

HP = {
    "p1: Blastoise": (186, 186),
    "p1: Farigiraf": (227, 227),
    "p2: Raichu": (100, 100),
    "p2: Incineroar": (100, 100),
}


def _events(*lines: str) -> list[list[str]]:
    return [line.split("|") for line in ("|turn|3", *lines, "|upkeep", "|turn|4")]


def _hits(*lines: str, hp=None) -> list[Hit]:
    return read_hits(_events(*lines), 0, HP if hp is None else hp)[0]


def test_idents_and_hp_strings():
    assert team_key("p2a: Raichu") == "p2: Raichu"
    assert team_key("p1b: Mr. Mime") == "p1: Mr. Mime"
    assert team_key("p2") is None
    assert _hp("120/186 par") == (120, 186)
    assert _hp("0 fnt") == (0, 0)
    assert _hp("oops") is None


def test_a_plain_exchange_gives_both_hits_as_shares_of_maximum_hp():
    hits = _hits(
        "|move|p2a: Raichu|Thunderbolt|p1a: Blastoise",
        "|-supereffective|p1a: Blastoise",
        "|-damage|p1a: Blastoise|86/186",
        "|move|p1a: Blastoise|Water Pulse|p2a: Raichu",
        "|-damage|p2a: Raichu|64/100",
    )
    assert hits == [
        Hit("p2: Raichu", "p1: Blastoise", "thunderbolt", 100 / 186),
        Hit("p1: Blastoise", "p2: Raichu", "waterpulse", 0.36),
    ]


def test_a_spread_move_gives_one_hit_per_target_and_a_second_hit_starts_lower():
    hits = _hits(
        "|move|p2b: Incineroar|Snarl|p1a: Blastoise|[spread] p1a,p1b",
        "|-damage|p1a: Blastoise|160/186",
        "|-damage|p1b: Farigiraf|200/227",
        "|move|p2a: Raichu|Thunderbolt|p1a: Blastoise",
        "|-damage|p1a: Blastoise|60/186",
    )
    assert [(h.defender, h.move) for h in hits] == [
        ("p1: Blastoise", "snarl"),
        ("p1: Farigiraf", "snarl"),
        ("p1: Blastoise", "thunderbolt"),
    ]
    assert hits[2].realized == pytest.approx(100 / 186)


@pytest.mark.parametrize(
    "spoiler",
    [
        "|-crit|p1a: Blastoise",
        "|-hitcount|p1a: Blastoise|2",
        "|-enditem|p1a: Blastoise|Wacan Berry|[weaken]",
    ],
    ids=["critical hit", "multi-hit", "resist berry"],
)
def test_hits_the_calculator_could_not_predict_are_dropped(spoiler):
    lines = [
        "|move|p2a: Raichu|Thunderbolt|p1a: Blastoise",
        "|-damage|p1a: Blastoise|86/186",
    ]
    # the berry is announced before the damage, the others after it
    lines.insert(1 if "enditem" in spoiler else 2, spoiler)
    assert _hits(*lines) == []


def test_a_knock_out_is_dropped_because_the_damage_is_cut_off():
    hits = _hits(
        "|move|p2a: Raichu|Thunderbolt|p1a: Blastoise",
        "|-damage|p1a: Blastoise|0 fnt",
        "|faint|p1a: Blastoise",
    )
    assert hits == []


def test_indirect_damage_is_not_a_hit_but_moves_the_hp():
    hits = _hits(
        "|move|p2a: Raichu|Thunderbolt|p1a: Blastoise",
        "|-damage|p1a: Blastoise|86/186",
        "|-damage|p2a: Raichu|90/100|[from] item: Life Orb",
        "|move|p1a: Blastoise|Water Pulse|p2a: Raichu",
        "|-damage|p2a: Raichu|54/100",
    )
    assert [h.move for h in hits] == ["thunderbolt", "waterpulse"]
    assert hits[1].realized == pytest.approx(0.36)  # from 90, not from 100


def test_a_stat_change_earlier_in_the_turn_spoils_that_pokemons_hits():
    boosted_attacker = _hits(
        "|-boost|p2a: Raichu|spa|2",
        "|move|p2a: Raichu|Thunderbolt|p1a: Blastoise",
        "|-damage|p1a: Blastoise|6/186",
    )
    weakened_defender = _hits(
        "|-unboost|p1a: Blastoise|spd|1",
        "|move|p2a: Raichu|Thunderbolt|p1a: Blastoise",
        "|-damage|p1a: Blastoise|46/186",
    )
    helped = _hits(
        "|move|p2b: Incineroar|Helping Hand|p2a: Raichu",
        "|-singleturn|p2a: Raichu|Helping Hand|[of] p2b: Incineroar",
        "|move|p2a: Raichu|Thunderbolt|p1a: Blastoise",
        "|-damage|p1a: Blastoise|36/186",
    )
    evolved_this_turn = _hits(
        "|detailschange|p2a: Raichu|Raichu-Mega-Y, L50, F",
        "|-mega|p2a: Raichu|Raichu|Raichunite Y",
        "|move|p2a: Raichu|Thunderbolt|p1a: Blastoise",
        "|-damage|p1a: Blastoise|6/186",
    )
    assert boosted_attacker == weakened_defender == helped == evolved_this_turn == []


def test_a_changed_board_spoils_what_follows_but_trick_room_does_not():
    after_a_switch = _hits(
        "|switch|p2b: Gengar|Gengar, L50, M|100/100",
        "|move|p2a: Raichu|Thunderbolt|p1a: Blastoise",
        "|-damage|p1a: Blastoise|86/186",
    )
    after_new_weather = _hits(
        "|-weather|RainDance|[from] ability: Drizzle|[of] p2b: Incineroar",
        "|move|p1a: Blastoise|Water Pulse|p2a: Raichu",
        "|-damage|p2a: Raichu|40/100",
    )
    after_a_faint = _hits(
        "|move|p1b: Farigiraf|Psychic|p2b: Incineroar",
        "|-damage|p2b: Incineroar|0 fnt",
        "|faint|p2b: Incineroar",
        "|move|p2a: Raichu|Thunderbolt|p1a: Blastoise",
        "|-damage|p1a: Blastoise|86/186",
    )
    assert after_a_switch == after_new_weather == after_a_faint == []
    under_a_new_room = _hits(
        "|move|p1b: Farigiraf|Trick Room|p1b: Farigiraf",
        "|-fieldstart|move: Trick Room|[of] p1b: Farigiraf",
        "|-weather|SunnyDay|[upkeep]",
        "|move|p2a: Raichu|Thunderbolt|p1a: Blastoise",
        "|-damage|p1a: Blastoise|86/186",
    )
    assert [h.move for h in under_a_new_room] == ["thunderbolt"]


def test_water_spout_counts_only_while_its_user_is_untouched():
    first = _hits(
        "|move|p1a: Blastoise|Water Spout|p2a: Raichu|[spread] p2a,p2b",
        "|-damage|p2a: Raichu|40/100",
        "|-damage|p2b: Incineroar|70/100",
    )
    assert len(first) == 2
    after_being_hit = _hits(
        "|move|p2a: Raichu|Thunderbolt|p1a: Blastoise",
        "|-damage|p1a: Blastoise|86/186",
        "|move|p1a: Blastoise|Water Spout|p2a: Raichu|[spread] p2a,p2b",
        "|-damage|p2a: Raichu|70/100",
        "|-damage|p2b: Incineroar|85/100",
    )
    assert [h.move for h in after_being_hit] == ["thunderbolt"]


def test_only_pokemon_that_stood_there_at_the_turn_start_count():
    hits = _hits(
        "|move|p2a: Raichu|Thunderbolt|p1a: Blastoise",
        "|-damage|p1a: Blastoise|86/186",
        hp={"p2: Raichu": (100, 100)},  # our Blastoise came in during the turn
    )
    assert hits == []


def test_the_reader_stops_at_the_next_turn():
    events = [
        *_events(
            "|move|p2a: Raichu|Thunderbolt|p1a: Blastoise",
            "|-damage|p1a: Blastoise|86/186",
            "|faint|p2b: Incineroar",
        ),
        "|move|p1a: Blastoise|Water Pulse|p2a: Raichu".split("|"),
        "|-damage|p2a: Raichu|64/100".split("|"),
        "|faint|p2a: Raichu".split("|"),
    ]
    hits, _ = read_hits(events, 0, HP)
    assert [h.move for h in hits] == ["thunderbolt"]
    assert faints(events, 0) == {"p2: Incineroar"}


def test_accuracy_compares_the_real_damage_with_the_range():
    rows = [
        # the real hit is above the stored range and inside the right one
        {
            "direction": "by_foe",
            "realized": 0.50,
            "stored": [0.30, 0.36],
            "fresh": [0.44, 0.52],
        },
        {
            "direction": "by_foe",
            "realized": 0.24,
            "stored": [0.20, 0.24],
            "fresh": [0.26, 0.31],
        },
    ]
    stored, fresh = accuracy(rows, "stored"), accuracy(rows, "fresh")
    assert stored is not None and fresh is not None
    assert stored["hits"] == 2
    assert stored["real_above_range"] == 0.5 and stored["inside_range"] == 0.5
    assert fresh["inside_range"] == 0.5 and fresh["real_below_range"] == 0.5
    assert stored["median_real_over_predicted"] == pytest.approx(
        (0.50 / 0.33 + 0.24 / 0.22) / 2
    )
    assert accuracy([], "stored") is None
