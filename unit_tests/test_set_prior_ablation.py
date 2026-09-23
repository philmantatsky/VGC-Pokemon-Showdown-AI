"""evaluation/run_set_prior_ablation.py: pin checking (with the promotion-safe
exemption), the per-arm set-data check, per-roster deltas, and the pooled
whole-roster bootstrap."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evaluation.run_set_prior_ablation import (
    changed_pins,
    check_arm_manifest,
    pooled_roster_bootstrap,
    roster_deltas,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_changed_pins_reports_edits_and_missing_files(tmp_path: Path) -> None:
    kept, edited, gone = tmp_path / "a.py", tmp_path / "b.py", tmp_path / "c.py"
    for path in (kept, edited, gone):
        path.write_text(path.name)
    pins = {str(p): _sha(p) for p in (kept, edited, gone)}
    edited.write_text("changed")
    gone.unlink()
    assert changed_pins(pins, skip=set()) == [str(edited), str(gone)]
    assert changed_pins(pins, skip={str(edited), str(gone)}) == []


def test_check_arm_manifest_requires_the_intended_set_data(tmp_path: Path) -> None:
    record = tmp_path / "arm.manifest.json"
    record.write_text(json.dumps({"set_prior_reg": "mb", "set_prior_sha256": "abc"}))
    check_arm_manifest(record, "mb", "abc")
    with pytest.raises(ValueError, match="set data mb != mc"):
        check_arm_manifest(record, "mc", "abc")
    with pytest.raises(ValueError, match="hash"):
        check_arm_manifest(record, "mb", "other")


def _rows(results: dict[str, list[float]]) -> list[dict]:
    return [{"opponent": k, "target": t} for k, ts in results.items() for t in ts]


def test_roster_deltas_are_new_minus_old_per_roster() -> None:
    old = _rows({"A": [1, 0], "B": [0, 0]})
    new = _rows({"A": [1, 1], "B": [1, 0]})
    assert roster_deltas(old, new) == {"A": 0.5, "B": 0.5}
    with pytest.raises(ValueError, match="unmatched"):
        roster_deltas(old, _rows({"A": [1, 1]}))


def test_pooled_bootstrap_weights_populations_equally() -> None:
    flat = {"p1": {"A": 0.0, "B": 0.0}, "p2": {"A": 0.0, "B": 0.0}}
    result = pooled_roster_bootstrap(flat, resamples=500)
    assert result["mean_delta"] == 0.0 and result["bootstrap_95_ci"] == [0.0, 0.0]
    shifted = {"p1": {"A": 0.1, "B": 0.1}, "p2": {"A": -0.3, "B": -0.3}}
    result = pooled_roster_bootstrap(shifted, resamples=500)
    assert result["mean_delta"] == pytest.approx(-0.1)
    assert result["bootstrap_95_ci"] == pytest.approx([-0.1, -0.1])
    assert result["populations"] == 2 and result["rosters"] == 2


def test_pooled_bootstrap_refuses_mismatched_rosters() -> None:
    with pytest.raises(ValueError, match="different rosters"):
        pooled_roster_bootstrap({"p1": {"A": 0.0}, "p2": {"B": 0.0}})
    with pytest.raises(ValueError, match="no populations"):
        pooled_roster_bootstrap({})
