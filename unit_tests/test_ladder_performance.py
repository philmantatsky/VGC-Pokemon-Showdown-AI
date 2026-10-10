"""evaluation/ladder_performance.py: the level a set of ladder games shows -- the
rating at which the results against the opponents met are the likeliest."""

from __future__ import annotations

import math
import os

import pytest

import evaluation.ladder_performance as ladder_performance
from evaluation.ladder_performance import (
    difference,
    expected,
    performance,
    read_games,
    render,
    summary,
)


def _games(wins: int, losses: int, opponent: int = 1300, ours: int = 1300):
    rows = [{"won": True, "opponent": opponent, "ours": ours, "forfeit": False}]
    rows = rows * wins + [
        {"won": False, "opponent": opponent, "ours": ours, "forfeit": False}
    ] * (losses)
    return [dict(row) for row in rows]


def test_the_rating_that_makes_the_expected_score_the_score():
    assert expected(1300, 1300) == 0.5
    assert expected(1500, 1300) == pytest.approx(0.7597, abs=1e-4)
    # six of ten against 1300: 400 * log10(0.6 / 0.4) above them
    won_six = performance(_games(6, 4))
    assert won_six == pytest.approx(1300 + 400 * math.log10(1.5), abs=0.01)
    assert performance(_games(5, 5, opponent=1450)) == pytest.approx(1450, abs=0.01)
    # the same record against stronger opponents is a higher level
    mixed = _games(3, 2, opponent=1200) + _games(2, 3, opponent=1500)
    level = performance(mixed)
    assert level is not None and 1200 < level < 1500
    assert sum(expected(level, row["opponent"]) for row in mixed) == pytest.approx(5.0)
    # no finite rating explains a sweep either way, and nothing explains no games
    assert performance(_games(4, 0)) is None and performance(_games(0, 4)) is None
    assert performance([]) is None


@pytest.fixture(autouse=True)
def _fewer_resamples(monkeypatch):
    monkeypatch.setattr(ladder_performance, "RESAMPLES", 300)


def test_a_set_is_summarised_with_its_interval_and_what_its_own_rating_expected():
    rows = _games(60, 40, opponent=1300, ours=1350)
    block = summary(rows)
    assert (block["games"], block["wins"]) == (100, 60)
    low, high = block["interval"]
    assert low < block["performance"] < high
    assert 30 < high - low < 160  # a hundred games: tens of points either way
    assert block["expected_wins_at_own_rating"] == pytest.approx(
        100 * expected(1350, 1300)
    )
    assert block["own_rating"]["mean"] == 1350
    text = render("run", block)
    assert "60 of 100 (60%)" in text and "performance rating 1370" in text
    assert render("none", summary([])) == "none: no counted game"


def test_two_sets_are_compared_by_resampling_both():
    same = difference(_games(55, 45), _games(55, 45))
    assert same["difference"] == pytest.approx(0.0)
    assert same["interval"][0] < 0 < same["interval"][1]
    apart = difference(_games(160, 40), _games(100, 100))
    assert apart["difference"] == pytest.approx(400 * math.log10(4), abs=0.01)
    assert apart["interval"][0] > 0 and apart["share_above_zero"] == 1.0


PAGE = """<script type="text/plain" class="battle-log-data">
|player|p1|antonius1|1|{ours}
|player|p2|someone|2|{theirs}
|poke|p1|Charizard, L50|
|poke|p2|Garchomp, L50|
|switch|p1a: Charizard|Charizard, L50|100/100
|switch|p2a: Garchomp|Garchomp, L50|100/100
|turn|1
{middle}|win|{winner}
</script>"""
PLAYED = "|turn|2\n|faint|p2a: Garchomp\n|turn|3\n"


def test_games_are_read_from_replays_and_a_turn_one_quit_is_not_a_game(tmp_path):
    pages = [
        ("first", dict(ours=1300, theirs=1350, middle=PLAYED, winner="antonius1")),
        ("second", dict(ours=1320, theirs=1280, middle=PLAYED, winner="someone")),
        # the opponent left before anything happened
        ("third", dict(ours=1310, theirs=1400, middle="", winner="antonius1")),
        # an unrated game carries no rating on the player line
        ("fourth", dict(ours="", theirs="", middle=PLAYED, winner="antonius1")),
    ]
    for age, (name, fields) in enumerate(pages):
        page = tmp_path / f"antonius1 - battle-x-{name}.html"
        page.write_text(PAGE.format(**fields))
        os.utime(page, (1_000_000 + age, 1_000_000 + age))
    rows = read_games([tmp_path])
    assert [(row["won"], row["opponent"], row["ours"]) for row in rows] == [
        (True, 1350, 1300),
        (False, 1280, 1320),
    ]
    assert performance(rows) is not None
