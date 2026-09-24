"""tools/deployed_config.py: the launchers' verified view of DEPLOYED.json --
every file sha-checked, the learned preview only when the manifest says so, and
shell-safe output."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from tools.deployed_config import MANIFEST, ROOT, resolve


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _deployment(root: Path, **extra) -> Path:
    files = {
        "checkpoint": "results_deployed/brain.zip",
        "team": "teams/candidates_mc/T9.txt",
        "preview_model": "data/preview model.pt",
    }
    for name, relative in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    deployed = {
        "checkpoint": files["checkpoint"],
        "sha256": _digest(root / files["checkpoint"]),
        "team": files["team"],
        "team_sha256": _digest(root / files["team"]),
        "guards_extra": "resisted_target,overkill_split",
        "reg": "mc",
        "format": "gen9championsvgc2026regmc",
        "set_prior_reg": "mc",
    } | extra
    manifest = root / "results_deployed/DEPLOYED.json"
    manifest.write_text(json.dumps({"deployed": deployed}))
    return manifest.relative_to(root)


def test_policy_preview_by_default(tmp_path):
    config = resolve(_deployment(tmp_path), tmp_path)
    assert config["PREVIEW_MODEL"] == ""
    assert config["CKPT"] == "results_deployed/brain.zip"
    assert config["REPLAY_TAG"] == "T9"
    assert config["SET_PRIOR"] == "mc"


def test_learned_preview_is_verified_and_exported(tmp_path):
    model = tmp_path / "data/preview model.pt"
    manifest = _deployment(
        tmp_path,
        learned_preview=True,
        preview_model="data/preview model.pt",
        replay_tag="T9_humanpreview",
    )
    deployed = json.loads((tmp_path / manifest).read_text())["deployed"]
    deployed["preview_model_sha256"] = _digest(model)
    (tmp_path / manifest).write_text(json.dumps({"deployed": deployed}))
    config = resolve(manifest, tmp_path)
    assert config["PREVIEW_MODEL"] == "data/preview model.pt"
    assert config["REPLAY_TAG"] == "T9_humanpreview"


def test_mismatched_files_are_refused(tmp_path):
    manifest = _deployment(tmp_path)
    (tmp_path / "teams/candidates_mc/T9.txt").write_text("edited team")
    with pytest.raises(ValueError, match="team does not match"):
        resolve(manifest, tmp_path)
    manifest = _deployment(
        tmp_path,
        learned_preview=True,
        preview_model="data/preview model.pt",
        preview_model_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="preview_model does not match"):
        resolve(manifest, tmp_path)


def test_ambiguous_preview_fields_are_refused(tmp_path):
    with pytest.raises(ValueError, match="learned_preview is not true"):
        resolve(_deployment(tmp_path, preview_model="data/preview model.pt"), tmp_path)
    with pytest.raises(ValueError, match="true or false"):
        resolve(_deployment(tmp_path, learned_preview="yes"), tmp_path)
    with pytest.raises(KeyError):
        resolve(_deployment(tmp_path, learned_preview=True), tmp_path)


def test_current_deployment_resolves_unchanged():
    """The live manifest keeps today's launch: same brain, team and replay dir."""
    deployed = json.loads((ROOT / MANIFEST).read_text())["deployed"]
    config = resolve()
    assert config["CKPT"] == deployed["checkpoint"]
    assert config["TEAM"] == deployed["team"]
    assert config["GUARDS"] == deployed["guards_extra"]
    assert config["PREVIEW_MODEL"] == (
        "" if not deployed.get("learned_preview") else deployed["preview_model"]
    )
    assert config["REPLAY_TAG"] == deployed.get(
        "replay_tag", Path(deployed["team"]).stem
    )


def test_cli_output_round_trips_through_the_shell(tmp_path):
    model = tmp_path / "data/preview model.pt"
    manifest = _deployment(
        tmp_path, learned_preview=True, preview_model="data/preview model.pt"
    )
    deployed = json.loads((tmp_path / manifest).read_text())["deployed"]
    deployed["preview_model_sha256"] = _digest(model)
    (tmp_path / manifest).write_text(json.dumps({"deployed": deployed}))
    printed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/deployed_config.py"),
            "--root",
            str(tmp_path),
            "--manifest",
            str(manifest),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert printed.returncode == 0, printed.stderr
    echoed = subprocess.run(
        ["bash", "-c", printed.stdout + 'printf "%s|%s" "$PREVIEW_MODEL" "$GUARDS"'],
        capture_output=True,
        text=True,
        check=True,
    )
    assert echoed.stdout == "data/preview model.pt|resisted_target,overkill_split"
    assert all(
        "=" in line and shlex.split(line) for line in printed.stdout.splitlines()
    )


def test_cli_refuses_with_status_2(tmp_path):
    manifest = _deployment(tmp_path)
    (tmp_path / "results_deployed/brain.zip").write_bytes(b"retrained in place")
    printed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/deployed_config.py"),
            "--root",
            str(tmp_path),
            "--manifest",
            str(manifest),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert printed.returncode == 2
    assert printed.stdout == ""
    assert "checkpoint does not match" in printed.stderr


def test_cli_prefix_keeps_launcher_overrides(tmp_path):
    manifest = _deployment(tmp_path)
    printed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/deployed_config.py"),
            "--root",
            str(tmp_path),
            "--manifest",
            str(manifest),
            "--prefix",
            "DEP_",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert all(line.startswith("DEP_") for line in printed.stdout.splitlines())
    echoed = subprocess.run(
        [
            "bash",
            "-c",
            "TEAM=mine.txt; " + printed.stdout + 'printf "%s|%s" "$TEAM" "$DEP_TEAM"',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert echoed.stdout == "mine.txt|teams/candidates_mc/T9.txt"


def test_misspelled_guards_are_refused(tmp_path):
    manifest = _deployment(tmp_path, guards_extra="resisted_target,dominated_atack")
    with pytest.raises(ValueError, match="unknown guards"):
        resolve(manifest, tmp_path)
