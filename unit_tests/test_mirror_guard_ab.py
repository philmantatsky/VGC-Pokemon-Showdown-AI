"""evaluation/mirror_guard_ab.py: the win-rate interval, the four-block split,
the cache the mirror's two sides must not share, and outcome counting."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from evaluation.mirror_guard_ab import _NoCache, _outcomes, block_sizes, wilson


def test_wilson_interval_brackets_the_rate_and_narrows_with_games():
    low, high = wilson(1000, 2000)
    assert low < 0.5 < high and high - low == pytest.approx(0.0438, abs=0.002)
    small = wilson(10, 20)
    assert small[1] - small[0] > high - low
    assert wilson(0, 0) == (0.0, 1.0)
    assert wilson(20, 20)[1] == 1.0


def test_block_sizes_cover_every_game_evenly():
    assert block_sizes(2000) == [500, 500, 500, 500]
    assert block_sizes(10) == [3, 3, 2, 2] and sum(block_sizes(10)) == 10


def test_the_knowledge_cache_stores_nothing():
    cache = _NoCache()
    cache["key"] = 1
    assert cache.get("key") is None and len(cache) == 0


def test_outcomes_count_ties_as_halves_and_refuse_unfinished_battles():
    battles = {
        "a": NS(finished=True, won=True, battle_tag="a"),
        "b": NS(finished=True, won=False, battle_tag="b"),
        "c": NS(finished=True, won=None, battle_tag="c"),
    }
    assert _outcomes(NS(battles=battles)) == (1.5, 3, 1)
    battles["d"] = NS(finished=False, won=None, battle_tag="d")
    with pytest.raises(RuntimeError, match="unfinished"):
        _outcomes(NS(battles=battles))
