"""evaluation/mirror_guard_ab.py: the win-rate interval, the four-block split,
the cache the mirror's two sides must not share, and outcome counting."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from evaluation.mirror_guard_ab import (
    _battle_rows,
    _NoCache,
    _outcomes,
    block_sizes,
    wilson,
)


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


def test_battle_rows_carry_each_battles_tag_result_and_length():
    battles = {
        "a": NS(finished=True, won=True, battle_tag="battle-a", turn=7),
        "b": NS(finished=True, won=None, battle_tag="battle-b", turn=12),
    }
    assert _battle_rows(NS(battles=battles)) == [
        {"battle": "battle-a", "a_won": True, "turns": 7},
        {"battle": "battle-b", "a_won": None, "turns": 12},
    ]


def test_side_a_may_only_play_a_set_variant_of_the_deployed_team():
    """--a-team (2026-10-01): T6m is T6 with two moves changed; T4 is another team."""
    from evaluation.mirror_guard_ab import ROOT, same_species

    teams = ROOT / "teams/candidates_mc"
    assert same_species(teams / "T6m.txt", teams / "T6.txt")
    assert same_species(teams / "T6mAS.txt", teams / "T6.txt")
    assert not same_species(teams / "T4.txt", teams / "T6.txt")


def test_search_manifest_records_the_device_and_the_chance_samples(tmp_path):
    """--device / --a-search-chance-samples (2026-10-04): the search's one-position
    network calls are ~15x faster on cpu than on mps, so a run has to say where its
    networks ran; the default stays mps, the device of every earlier run."""
    import json
    import subprocess
    import sys

    from evaluation.mirror_guard_ab import ROOT
    from tools.deployed_config import resolve

    needed = [ROOT / resolve()["CKPT"], ROOT / "results_outcome_v2h/outcome_value.zip"]
    if not all(path.is_file() for path in needed):
        pytest.skip("the deployed checkpoint is not on this machine")

    def manifest(name: str, *flags: str) -> dict:
        out = tmp_path / name
        subprocess.run(
            [
                sys.executable,
                "evaluation/mirror_guard_ab.py",
                "--a-search",
                "nash",
                *flags,
                "--prepare-only",
                "--output",
                str(out),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        return json.loads((out / "manifest.json").read_text())

    chosen = manifest("cpu", "--device", "cpu", "--a-search-chance-samples", "2")
    assert chosen["device"] == "cpu"
    assert chosen["a_search"]["chance_samples"] == 2
    default = manifest("default")
    assert default["device"] == "mps"
    assert default["a_search"]["chance_samples"] == 1
