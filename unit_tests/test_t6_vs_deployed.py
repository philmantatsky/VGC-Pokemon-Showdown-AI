import copy
import json

import pytest

from evaluation.run_t6_vs_deployed import deployed_config, sha256, validate_arm


def sample():
    rows = [
        {
            "opponent": "a.txt",
            "hidden_sheets": h,
            "battle": str(h),
            "arm": "policy",
            "target": 1.0,
            "bring_species": list("abcd"),
        }
        for h in (True, False)
    ]
    telemetry = [
        {
            "opponent": "a.txt",
            "hidden_sheets": h,
            "games": 1,
            "preview_pairing_mismatches": 0,
            "guard_counts": {},
        }
        for h in (True, False)
    ]
    return rows, telemetry, [{"path": "teams/a.txt"}], 1, set("abcdef")


def test_complete_two_mode_comparison():
    validate_arm(*sample())


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "team", "guard", "telemetry"]
)
def test_bad_comparison_fails_closed(fault):
    args = copy.deepcopy(sample())
    rows, telemetry = args[:2]
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[1]["battle"] = rows[0]["battle"]
    elif fault == "team":
        rows[0]["bring_species"] = list("wxyz")
    elif fault == "guard":
        telemetry[0]["guard_counts"] = {"error:test": 1}
    else:
        telemetry.pop()
    with pytest.raises(ValueError):
        validate_arm(*args)


def test_deployed_control_is_content_pinned(tmp_path):
    model, team = tmp_path / "model", tmp_path / "team"
    model.write_text("frozen")
    team.write_text("team")
    spec = {
        "checkpoint": str(model),
        "sha256": sha256(model),
        "team": str(team),
        "team_sha256": sha256(team),
        "reg": "mc",
        "search": False,
        "knowledge_obs": True,
        "guards_extra": "resisted_target,overkill_split,dominated_weather_ball_weather",
    }
    manifest = tmp_path / "deployed.json"
    manifest.write_text(json.dumps({"deployed": spec}))
    assert deployed_config(manifest) == spec
    model.write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        deployed_config(manifest)
