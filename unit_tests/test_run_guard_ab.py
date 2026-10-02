"""evaluation/run_guard_ab.py: the reused arm must match the new arm in every
recorded input but the output path, and only the extra guards' firing counts."""

from __future__ import annotations

import json

from evaluation.run_guard_ab import guard_firing, same_study, unexplained


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


def _prior_run(tmp_path, phase="complete_review_required", **changes):
    from pathlib import Path

    reference = {
        "candidate_sha256": "c" * 64,
        "our_preview": {"model_sha256": "p" * 64},
        "seed": 20923,
        "repeats": 11,
        "populations": ["human_new", "frozen"],
    }
    prior = {
        "extra_guards": ["dominated_attack"],
        "checkpoint_sha256": "c" * 64,
        "preview_model_sha256": "p" * 64,
        "seed": 20923,
        "repeats": 11,
        "populations": ["human_new", "frozen"],
    } | changes
    run = Path(tmp_path) / "results_guard_ab_dominated_attack"
    run.mkdir(exist_ok=True)
    (run / "manifest.json").write_text(json.dumps(prior))
    (run / "status.json").write_text(json.dumps({"phase": phase}))
    return run, reference


def test_a_completed_deployed_run_becomes_the_without_side(tmp_path):
    from evaluation.run_guard_ab import without_arm

    run, reference = _prior_run(tmp_path)
    deployed = {"resisted_target", "dominated_attack"}
    assert without_arm(run, reference, deployed) == (
        ["dominated_attack"],
        "with_dominated_attack",
    )


def test_the_without_side_must_be_complete_the_same_study_and_deployed(tmp_path):
    import pytest

    from evaluation.run_guard_ab import without_arm

    deployed = {"resisted_target", "dominated_attack"}
    run, reference = _prior_run(tmp_path, phase="rotation1")
    with pytest.raises(ValueError, match="not complete"):
        without_arm(run, reference, deployed)
    run, reference = _prior_run(tmp_path, seed=1)
    with pytest.raises(ValueError, match="not the same study"):
        without_arm(run, reference, deployed)
    run, reference = _prior_run(tmp_path, extra_guards=["trick_room_direction"])
    with pytest.raises(ValueError, match="not all deployed"):
        without_arm(run, reference, deployed)


def _named_run(tmp_path, name, **prior):
    from pathlib import Path

    run = Path(tmp_path) / name
    run.mkdir(exist_ok=True)
    (run / "manifest.json").write_text(json.dumps(prior))
    (run / "status.json").write_text(json.dumps({"phase": "complete_review_required"}))
    return run


def _study(**changes):
    return {
        "checkpoint_sha256": "d" * 64,
        "preview_model_sha256": "p" * 64,
        "seed": 20923,
        "repeats": 11,
        "populations": ["human_new", "frozen"],
    } | changes


def test_a_baseline_run_of_the_deployed_brain_becomes_the_without_side(tmp_path):
    """2026-09-26: T6ctx replaced the brain the reference study played, so the
    deployed configuration is played once (--baseline) and reused."""
    from evaluation.run_guard_ab import without_arm

    run = _named_run(
        tmp_path,
        "results_brain_ab_deployed_T6ctx",
        **_study(
            arm_label="deployed_T6ctx",
            extra_guards=[],
            base_guards_both_sides=["dominated_attack", "wide_guard"],
        ),
    )
    _, reference = _prior_run(tmp_path)  # the study design; its brain is older
    deployed = {"resisted_target", "dominated_attack", "wide_guard"}
    assert without_arm(run, reference, deployed, "d" * 64) == (
        ["dominated_attack", "wide_guard"],
        "deployed_T6ctx",
    )


def test_the_without_side_must_have_played_the_deployed_brain(tmp_path):
    import pytest

    from evaluation.run_guard_ab import without_arm

    run = _named_run(
        tmp_path,
        "results_brain_ab_deployed_T6ctx",
        **_study(
            arm_label="deployed_T6ctx",
            extra_guards=[],
            base_guards_both_sides=["dominated_attack"],
        ),
    )
    _, reference = _prior_run(tmp_path)
    with pytest.raises(ValueError, match="not the same study"):
        without_arm(run, reference, {"dominated_attack"}, "e" * 64)


def test_a_candidate_run_keeps_its_arm_label(tmp_path):
    """Runs from before arm_label was recorded: a brain arm is
    <population>_candidate_<label>.jsonl, its guards the base it ran with."""
    from evaluation.run_guard_ab import without_arm

    run = _named_run(
        tmp_path,
        "results_brain_ab_contexts1_22118400",
        **_study(extra_guards=[], base_guards_both_sides=["dominated_attack"]),
    )
    _, reference = _prior_run(tmp_path)
    assert without_arm(run, reference, {"dominated_attack"}, "d" * 64) == (
        ["dominated_attack"],
        "candidate_contexts1_22118400",
    )


def test_a_guard_arm_plays_the_without_sides_own_brain_file(tmp_path):
    """Since T6ctx the reference study's brain is not the deployed one: a guard or
    playbook arm plays the without side's own file (its sha is the deployed
    brain's), else the arm manifests differ in the checkpoint."""
    import hashlib

    import pytest

    from evaluation.run_guard_ab import without_brain

    brain = tmp_path / "brain.zip"
    brain.write_bytes(b"weights")
    run = tmp_path / "results_brain_ab_x"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"checkpoint": str(brain)}))
    assert without_brain(run, hashlib.sha256(b"weights").hexdigest()) == str(brain)
    with pytest.raises(ValueError, match="not the deployed brain"):
        without_brain(run, "0" * 64)


def test_a_candidate_on_a_set_variant_may_differ_in_brain_and_sets_only():
    """--candidate-plans (2026-10-01, a brain practised on T6m): the arms differ in
    the checkpoint and our team's sets; anything else still stops the comparison."""
    differs = ["checkpoint", "checkpoint_sha256", "plans", "team_sha256", "seed"]
    assert unexplained(differs, candidate=False, variant=False) == differs
    assert unexplained(differs, candidate=True, variant=False) == [
        "plans",
        "team_sha256",
        "seed",
    ]
    assert unexplained(differs, candidate=True, variant=True) == ["seed"]
    # a guard arm never plays another team
    assert unexplained(["plans"], candidate=False, variant=True) == ["plans"]
