from __future__ import annotations

import json

import pytest

from training.run_counterfactual_pipeline import _score_evaluations


def _write_mode(tmp_path, mode, champion, distilled, preview=None):
    arms = {
        "champion_policy": {"win_rate": champion},
        "distilled_policy": {"win_rate": distilled},
    }
    if preview is not None:
        arms["preview_policy"] = {"win_rate": preview}
    path = tmp_path / f"{mode}.json"
    path.write_text(json.dumps({"evaluation_schema": 3, "arms": arms}))
    return path


def test_evaluation_scores_only_arms_that_really_ran(tmp_path):
    paths = {
        "open": _write_mode(tmp_path, "open", 0.80, 0.84),
        "hidden": _write_mode(tmp_path, "hidden", 0.70, 0.72),
        "population": _write_mode(tmp_path, "population", 0.60, 0.64),
    }

    result = _score_evaluations(paths)

    assert set(result["candidates"]) == {"distilled_policy"}
    assert result["champion"]["score"] == pytest.approx(0.675)
    assert result["candidates"]["distilled_policy"]["score"] == pytest.approx(
        0.71
    )


def test_preview_arm_must_be_present_in_every_mode(tmp_path):
    paths = {
        "open": _write_mode(tmp_path, "open", 0.8, 0.8, preview=0.9),
        "hidden": _write_mode(tmp_path, "hidden", 0.8, 0.8),
        "population": _write_mode(tmp_path, "population", 0.8, 0.8),
    }

    result = _score_evaluations(paths)

    assert "preview_policy" not in result["candidates"]


def test_stale_mislabeled_evaluation_is_rejected(tmp_path):
    paths = {}
    for mode in ("open", "hidden", "population"):
        path = tmp_path / f"{mode}.json"
        path.write_text(
            json.dumps(
                {
                    "arms": {
                        "baseline": {"win_rate": 0.8},
                        "candidate_planner_preview": {"win_rate": 0.8},
                    }
                }
            )
        )
        paths[mode] = path

    with pytest.raises(ValueError, match="stale or mislabeled"):
        _score_evaluations(paths)
