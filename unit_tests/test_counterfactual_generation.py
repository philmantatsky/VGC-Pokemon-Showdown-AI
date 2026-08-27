from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from generate_counterfactuals import _initialise_manifest, _preview_plan_or_fallback
from vgc_bench.src.exact_planner import DeterminizationBudgetExhausted


class _BudgetPlanner:
    def plan(self, _roots, _role):
        raise DeterminizationBudgetExhausted(
            "determinization search exhausted its budget"
        )


class _BrokenPlanner:
    def plan(self, _roots, _role):
        raise ValueError("incompatible generated action")


class _Bridge:
    def choices(self, _state, _role):
        return ["team 1234"]


class _Adapter:
    def rank(self, _state, _requests, _role, _legal):
        return []


def test_preview_budget_exhaustion_falls_back_to_champion_choice():
    node = NS(state={}, requests=[{}, {}], request_state="teampreview")

    result, choice = _preview_plan_or_fallback(
        _BudgetPlanner(), [object()], _Adapter(), _Bridge(), node
    )

    assert result is None
    assert choice == "team 1234"


def test_preview_mapping_errors_still_fail_fast():
    node = NS(state={}, requests=[{}, {}], request_state="teampreview")

    with pytest.raises(ValueError, match="incompatible generated action"):
        _preview_plan_or_fallback(
            _BrokenPlanner(), [object()], _Adapter(), _Bridge(), node
        )


def test_resume_may_change_only_wall_clock_cap(tmp_path):
    manifest = tmp_path / "generation_manifest.json"
    original = {"schema": 2, "depth": 2, "max_seconds": 5400.0}
    resumed = {"schema": 2, "depth": 2, "max_seconds": 3300.0}

    _initialise_manifest(tmp_path, manifest, original)
    loaded = _initialise_manifest(tmp_path, manifest, resumed)

    assert loaded["config"] == original


def test_resume_still_rejects_search_changes(tmp_path):
    manifest = tmp_path / "generation_manifest.json"
    _initialise_manifest(
        tmp_path, manifest, {"schema": 2, "depth": 2, "max_seconds": 5400.0}
    )

    with pytest.raises(SystemExit, match="different generation config"):
        _initialise_manifest(
            tmp_path,
            manifest,
            {"schema": 2, "depth": 3, "max_seconds": 3300.0},
        )


def test_save_chunk_survives_a_zero_candidate_example(tmp_path):
    """A position where every choice fails action-encoding must not crash saves.

    Game 283 of the v5h run aborted on broadcasting shape (0,) into (0, 2);
    generation now skips such examples at the source, and the writer tolerates
    them defensively for partial-save paths.
    """
    import numpy as np

    from generate_counterfactuals import _save_chunk

    def example(candidates: int) -> dict:
        return {
            "observation": np.zeros(8, dtype=np.float16),
            "action_mask": np.zeros(4, dtype=np.uint8),
            "actions": np.asarray(
                [(1, 2)] * candidates, dtype=np.int16
            ).reshape(candidates, 2)
            if candidates
            else np.asarray([], dtype=np.int16),
            "scores": np.asarray([0.5] * candidates, dtype=np.float32),
            "expected": np.asarray([0.5] * candidates, dtype=np.float32),
            "priors": np.asarray([1.0] * candidates, dtype=np.float32),
            "metadata": {"game": 0},
        }

    path = _save_chunk(tmp_path, 0, [example(2), example(0)])
    payload = np.load(path)
    assert payload["candidate_actions"].shape == (2, 2, 2)
    assert (payload["candidate_actions"][1] == -1).all()
