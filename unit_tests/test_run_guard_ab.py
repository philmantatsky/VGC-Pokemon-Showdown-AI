"""evaluation/run_guard_ab.py: the reused arm must match the new arm in every
recorded input but the output path, and only the extra guards' firing counts."""

from __future__ import annotations

from evaluation.run_guard_ab import guard_firing, same_study


def test_same_study_ignores_only_the_output_path():
    ref = {
        "checkpoint": "a.zip",
        "seed": 1,
        "output": "ref.jsonl",
        "guard_flags": {"x": True},
    }
    new = dict(ref, output="new.jsonl")
    assert same_study(ref, new) == []
    assert same_study(ref, dict(new, seed=2)) == ["seed"]
    assert same_study(ref, dict(new, guard_flags={"x": True, "y": True})) == [
        "guard_flags"
    ]
    assert same_study(ref, {k: v for k, v in new.items() if k != "seed"}) == ["seed"]


def test_guard_firing_counts_only_the_extra_guards():
    telemetry = [
        {
            "guard_counts": {
                "dominated_attack": 2,
                "dominated_attack:promoted:demoted": 3,
            }
        },
        {"guard_counts": {"dominated_attack": 1, "resisted_target": 5}},
        {"guard_counts": {"dominated_attack_other": 9}},
        {},
    ]
    assert guard_firing(telemetry, ["dominated_attack"]) == {
        "dominated_attack": 3,
        "dominated_attack:promoted:demoted": 3,
    }


def test_a_newly_registered_guard_that_is_off_is_not_a_difference():
    ref = {"seed": 1, "guard_flags": {"resisted_target": True, "ko_tiebreak": False}}
    later = {
        "seed": 1,
        "guard_flags": {**ref["guard_flags"], "dominated_attack": False},
    }
    assert same_study(ref, later) == []
    turned_on = {
        "seed": 1,
        "guard_flags": {**ref["guard_flags"], "dominated_attack": True},
    }
    assert same_study(ref, turned_on) == ["guard_flags"]
    flipped = {
        "seed": 1,
        "guard_flags": {"resisted_target": False, "ko_tiebreak": False},
    }
    assert same_study(ref, flipped) == ["guard_flags"]
