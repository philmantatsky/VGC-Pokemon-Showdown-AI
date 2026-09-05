from types import SimpleNamespace

import pytest

from evaluation.eval_counterfactual import (
    PreviewLedger,
    _apply_preview_choice,
    _preview_fingerprint,
)


def _mon(species: str):
    return SimpleNamespace(
        base_species=species,
        species=species,
        _selected_in_teampreview=False,
    )


def test_preview_ledger_replays_repeated_team_occurrences_in_order():
    ledger = PreviewLedger()
    key = ("pelipper", "archaludon", "swampert")
    ledger.record(key, "/team 1234")
    ledger.record(key, "/team 2143")

    ledger.begin_replay()

    assert ledger.next(key) == "/team 1234"
    assert ledger.next(key) == "/team 2143"
    # Exhausted pairing no longer raises inside a battle handler (which stalled
    # the serial search arm): it returns None and counts the drift.
    assert ledger.next(key) is None
    assert ledger.mismatched == 1


def test_paired_preview_marks_the_scripted_four_and_records_order():
    team = [_mon(name) for name in ("one", "two", "three", "four", "five", "six")]
    battle = SimpleNamespace(team={str(i): mon for i, mon in enumerate(team)})
    recorded = []
    player = SimpleNamespace(
        _record_own_preview=lambda _battle, leads, backline: recorded.append(
            (leads, backline)
        )
    )

    _apply_preview_choice(player, battle, "/team 3512")

    assert [mon._selected_in_teampreview for mon in team] == [True, True, True, False, True, False]
    assert recorded == [((3, 5), (1, 2))]
    assert _preview_fingerprint(battle) == (
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
    )


def test_preview_fingerprint_includes_both_rosters_for_concurrent_pairing():
    ours = [_mon(name) for name in ("one", "two", "three", "four", "five", "six")]
    rain = [_mon(name) for name in ("pelipper", "archaludon")]
    sand = [_mon(name) for name in ("tyranitar", "excadrill")]
    rain_battle = SimpleNamespace(
        team={str(i): mon for i, mon in enumerate(ours)},
        opponent_team={str(i): mon for i, mon in enumerate(rain)},
    )
    sand_battle = SimpleNamespace(
        team={str(i): mon for i, mon in enumerate(ours)},
        opponent_team={str(i): mon for i, mon in enumerate(sand)},
    )

    assert _preview_fingerprint(rain_battle) != _preview_fingerprint(sand_battle)
    assert "__vs__" in _preview_fingerprint(rain_battle)
