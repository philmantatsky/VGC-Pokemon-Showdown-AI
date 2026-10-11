"""The length of real games in decision steps (evaluation/feed_game_length.py): a step
is a stop of the simulator for somebody's choice, and Showdown marks the end of each
with a ``|t:|`` line."""

from __future__ import annotations

import gzip
import json

import pytest

from evaluation.feed_game_length import Remaining, game_length, read, render

START = """|j|☆Ann
|j|☆Bob
|t:|100
|gametype|doubles
|player|p1|Ann|1|1510
|player|p2|Bob|2|1490
|clearpoke
|poke|p1|Incineroar, L50|
|poke|p2|Garchomp, L50|
|teampreview|4
|inactive|Battle timer is ON: inactive players will automatically lose.
|
|t:|110
|teamsize|p1|2
|teamsize|p2|2
|start
|switch|p1a: Incineroar|Incineroar, L50|100/100
|switch|p2a: Garchomp|Garchomp, L50|100/100
|turn|1
|c|☆Ann|gl
|
|t:|120
|move|p1a: Incineroar|U-turn|p2a: Garchomp
|-damage|p2a: Garchomp|80/100
|inactive|Ann has 30 seconds left.
|
|t:|125
|switch|p1a: Torkoal|Torkoal, L50|100/100|[from] U-turn
|move|p2a: Garchomp|Earthquake|p1a: Torkoal
|-damage|p1a: Torkoal|0 fnt
|faint|p1a: Torkoal
|
|upkeep
|l|☆Bob
|
|t:|130
|switch|p1a: Incineroar|Incineroar, L50|100/100
|turn|2
"""

PLAYED_OUT = (
    START
    + """|
|t:|140
|move|p2a: Garchomp|Earthquake|p1a: Incineroar
|-damage|p1a: Incineroar|0 fnt
|faint|p1a: Incineroar
|
|win|Bob
"""
)

FORFEITED = START + "|-message|Ann forfeited.\n|\n|win|Bob\n"
TIMED_OUT = START + "|-message|Ann lost due to inactivity.\n|\n|win|Bob\n"
QUIT_IN_PREVIEW = (
    START.split("|inactive|Battle timer")[0]
    + "|showteam|p1|Incineroar||SitrusBerry|Intimidate|FakeOut|||||50|\n"
    + "|-message|Bob forfeited.\n|\n|win|Ann\n"
)


def test_a_step_is_every_stop_somebody_answered():
    game = game_length(PLAYED_OUT)
    # preview, the two move requests, the switch U-turn asks for, the replacement
    assert game["steps"] == 5 and game["turns"] == 2
    assert [game[k] for k in ("preview", "move", "inside_turn", "after_turn")] == [
        1,
        2,
        1,
        1,
    ]
    assert game["ended"] == "natural" and not game["pending"]
    assert game["loser"] == "p1"
    # both sides choose at preview and at a move request, one side at each switch here
    assert game["choices"] == 2 + 2 * 2 + 1 + 1
    # what each side had left when each request was asked (teams of two here; the
    # preview is asked before the log says how many were brought)
    assert game["left_at_requests"] == [(4, 4), (2, 2), (2, 2), (1, 2), (1, 2)]


def test_a_request_nobody_answered_is_not_a_step():
    for log, ended in ((FORFEITED, "forfeit"), (TIMED_OUT, "timer")):
        game = game_length(log)
        assert game["steps"] == 4 and game["turns"] == 2
        assert game["ended"] == ended and game["pending"]
        assert game["loser"] == "p1" and game["left_at_end"] == (1, 2)
    early = game_length(QUIT_IN_PREVIEW)
    assert early["steps"] == 0 and early["turns"] == 0
    assert early["ended"] == "forfeit" and early["loser"] == "p2"


def test_open_sheets_after_the_preview_line_do_not_hide_it():
    log = PLAYED_OUT.replace(
        "|teampreview|4\n", "|teampreview|4\n|showteam|p1|Incineroar||]Torkoal||\n"
    )
    assert game_length(log)["preview"] == 1
    assert game_length(log)["inside_turn"] == 1


def test_a_step_is_named_by_where_the_battle_stood():
    """A Quick Claw is announced after ``|turn|N`` while the choices are read: the
    stop was still a move request (the first version read the last line before the
    stop and called 9 of 13,229 move requests a switch). A replacement that faints
    on the way in is replaced before the next turn too."""
    quick = PLAYED_OUT.replace(
        "|turn|2\n", "|turn|2\n|-activate|p2a: Garchomp|item: Quick Claw\n"
    )
    assert game_length(quick)["move"] == 2 and game_length(quick)["inside_turn"] == 1
    rocks = START.replace("|teamsize|p1|2", "|teamsize|p1|3").replace(
        "|t:|130\n|switch|p1a: Incineroar|Incineroar, L50|100/100\n",
        "|t:|130\n|switch|p1a: Incineroar|Incineroar, L50|100/100\n"
        "|-damage|p1a: Incineroar|0 fnt|[from] Stealth Rock\n|faint|p1a: Incineroar\n"
        "|\n|t:|135\n|switch|p1a: Venusaur|Venusaur, L50|100/100\n",
    )
    game = game_length(rocks)
    assert (game["move"], game["inside_turn"], game["after_turn"]) == (1, 1, 2)
    assert game["steps"] == 5 and game["left_at_end"] == (1, 2)
    # both sides replace at once: one step, two choices
    both = PLAYED_OUT.replace(
        "|t:|130\n|switch|p1a: Incineroar|Incineroar, L50|100/100\n",
        "|t:|130\n|switch|p1a: Incineroar|Incineroar, L50|100/100\n"
        "|switch|p2a: Kingambit|Kingambit, L50|100/100\n",
    )
    assert game_length(both)["choices"] == game_length(PLAYED_OUT)["choices"] + 1
    # a Pokemon brought back by Revival Blessing is no longer among the fainted
    back = rocks.replace(
        "|\n|t:|135\n",
        "|\n|t:|135\n|-heal|p1: Torkoal|50/100|[from] move: Revival Blessing\n",
    )
    assert game_length(back)["left_at_end"] == (2, 2)


def test_a_game_cut_short_is_completed_from_games_played_out():
    """The simulator has no forfeit: a game that ended early is given the steps games
    played to the end still took from the same Pokemon left."""
    remaining = Remaining()
    remaining.add(game_length(PLAYED_OUT))
    # the loser at one left against two: the requests at steps 4 and 5 of 5
    assert remaining.expected(game_length(FORFEITED)) == pytest.approx(1.5)
    # in preview the whole game is still to come, and the forfeiter is the loser
    assert remaining.expected(game_length(QUIT_IN_PREVIEW)) == pytest.approx(5.0)
    # a game nobody won teaches nothing about losing positions
    drawn = game_length(PLAYED_OUT.replace("|win|Bob", "|tie"))
    assert drawn["ended"] == "natural" and drawn["loser"] == "p1"
    unfinished = game_length(START)
    assert unfinished["ended"] == "unfinished"
    assert remaining.expected(unfinished) is None


def test_the_corpus_is_read_by_format_and_rating_band(tmp_path):
    rows = {
        "gen9championsvgc2026regmc": [
            {"log": PLAYED_OUT, "rating": 1520},
            {"log": FORFEITED, "rating": 1310},
            {"log": QUIT_IN_PREVIEW, "rating": None},
        ],
        "gen9championsvgc2026regmcbo3": [{"log": PLAYED_OUT, "rating": 1250}],
    }
    for name, games in rows.items():
        directory = tmp_path / "logs" / name
        directory.mkdir(parents=True)
        with gzip.open(directory / "part-00000.jsonl.gz", "wt") as out:
            for game in games:
                out.write(json.dumps(game) + "\n")
    reading = read(tmp_path)
    table = reading["table"]
    assert reading["games"] == 4 and reading["rated"] == 3
    everything = table["all formats | all"]
    assert everything["games"] == 4
    assert everything["steps"]["mean"] == pytest.approx((5 + 4 + 0 + 5) / 4)
    assert everything["played_to_the_end"]["games"] == 2
    assert everything["ended"] == {"forfeit": 2, "natural": 2}
    assert everything["ended_in_preview"] == 1
    # 5, 4 + 1.5, 0 + 5, 5
    assert everything["completed_steps"]["mean"] == pytest.approx(5.125)
    assert table["all formats | 1300+"]["games"] == 2
    assert table["all formats | 1500+"]["games"] == 1
    assert table["gen9championsvgc2026regmcbo3 | all"]["games"] == 1
    assert table["gen9championsvgc2026regmc | 1300+"]["steps"]["mean"] == 4.5
    text = "\n".join(render(reading))
    assert "all formats | all: 4 games" in text
    assert "ended before the end: 2 (50.0%) -- forfeit 2 (50.0%)" in text
