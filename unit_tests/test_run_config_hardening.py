"""Stage-A hardening: knowledge_obs resolution, run-config echo, preview rules.

The knowledge_obs flag must never default silently (a silent False zeroed the
champion's 24 knowledge features for a whole ladder era), every ladder batch
must be auditable from <replay_dir>/run_config.json, and the Trick Room lookup
must stay team-agnostic.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from ladder_ourteam import record_run_config, resolve_knowledge_obs
from vgc_bench.src.preview_rules import species_trick_room_rate, trick_room_probability


def _fake_ckpt(tmp_path: Path, payload: bytes = b"weights") -> tuple[Path, str]:
    ckpt = tmp_path / "candidate.zip"
    ckpt.write_bytes(payload)
    return ckpt, hashlib.sha256(payload).hexdigest()


def _stamp(ckpt: Path, sha: str, requires: bool) -> Path:
    sidecar = Path(str(ckpt) + ".metadata.json")
    sidecar.write_text(json.dumps({"sha256": sha, "requires_knowledge_obs": requires}))
    return sidecar


class TestResolveKnowledgeObs:
    def test_unresolved_without_sidecar_hard_fails(self, tmp_path):
        ckpt, sha = _fake_ckpt(tmp_path)
        with pytest.raises(SystemExit, match="unresolved"):
            resolve_knowledge_obs(None, ckpt, sha)

    def test_sidecar_resolves_when_flag_omitted(self, tmp_path):
        ckpt, sha = _fake_ckpt(tmp_path)
        _stamp(ckpt, sha, requires=True)
        assert resolve_knowledge_obs(None, ckpt, sha) is True
        _stamp(ckpt, sha, requires=False)
        assert resolve_knowledge_obs(None, ckpt, sha) is False

    def test_explicit_flag_wins_without_sidecar(self, tmp_path):
        ckpt, sha = _fake_ckpt(tmp_path)
        assert resolve_knowledge_obs(True, ckpt, sha) is True
        assert resolve_knowledge_obs(False, ckpt, sha) is False

    def test_contradicting_flag_hard_fails(self, tmp_path):
        ckpt, sha = _fake_ckpt(tmp_path)
        _stamp(ckpt, sha, requires=True)
        with pytest.raises(SystemExit, match="contradicts"):
            resolve_knowledge_obs(False, ckpt, sha)

    def test_stale_sidecar_sha_hard_fails(self, tmp_path):
        ckpt, sha = _fake_ckpt(tmp_path)
        _stamp(ckpt, "0" * 64, requires=True)
        with pytest.raises(SystemExit, match="different file"):
            resolve_knowledge_obs(None, ckpt, sha)
        # even an explicit flag cannot override a stale stamp
        with pytest.raises(SystemExit, match="different file"):
            resolve_knowledge_obs(True, ckpt, sha)


def _args(**overrides) -> argparse.Namespace:
    base = {
        "checkpoint": "candidate.zip",
        "our_team": "teams/reg_mb/our_team.txt",
        "reg": "mb",
        "n_games": 10,
        "challenges": False,
        "debug": False,
        "replay_dir": "ladder_replays_test",
        "decision_log": "",
        "deployment": "",
        "knowledge_obs": True,
        "search": False,
        "no_immunity_mask": False,
        "no_moveset_prior": False,
        "guard_profile": "hard",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class TestRecordRunConfig:
    def test_writes_material_config(self, tmp_path):
        path = record_run_config(tmp_path, _args(), "abc123", "hard")
        recorded = json.loads(path.read_text())
        assert len(recorded["runs"]) == 1
        material = recorded["runs"][0]["material"]
        assert material["checkpoint_sha256"] == "abc123"
        assert material["knowledge_obs"] is True
        assert material["guard_profile_resolved"] == "hard"
        assert material["mask_immunities"] is True
        # volatile keys must not be material
        assert "n_games" not in material
        assert "replay_dir" not in material

    def test_same_config_appends(self, tmp_path):
        record_run_config(tmp_path, _args(n_games=5), "abc123", "hard")
        path = record_run_config(tmp_path, _args(n_games=25), "abc123", "hard")
        assert len(json.loads(path.read_text())["runs"]) == 2

    def test_changed_config_refused_and_names_the_change(self, tmp_path):
        record_run_config(tmp_path, _args(), "abc123", "hard")
        with pytest.raises(SystemExit, match="knowledge_obs"):
            record_run_config(tmp_path, _args(knowledge_obs=False), "abc123", "hard")
        with pytest.raises(SystemExit, match="checkpoint_sha256"):
            record_run_config(tmp_path, _args(), "different", "hard")


class TestPreviewRules:
    def test_known_setters_rank_high(self):
        assert species_trick_room_rate("farigiraf") > 0.9
        assert species_trick_room_rate("hatterene") > 0.9
        assert species_trick_room_rate("garchomp") < 0.05

    def test_unknown_species_is_zero_not_error(self):
        assert species_trick_room_rate("notarealpokemon") == 0.0

    def test_roster_aggregation_monotone(self):
        low = trick_room_probability(("garchomp", "kingambit", "whimsicott"))
        high = trick_room_probability(("farigiraf", "garchomp", "kingambit"))
        assert 0.0 <= low < 0.1
        assert high > 0.9
        assert trick_room_probability(()) == 0.0

    def test_team_agnostic_over_arbitrary_rosters(self):
        # Any roster must produce a probability without error -- no species may
        # be assumed. Three structurally different rosters:
        rosters = (
            ("farigiraf", "torkoal", "hatterene", "indeedee", "ursaluna", "incineroar"),
            (
                "charizard",
                "garchomp",
                "whimsicott",
                "kingambit",
                "basculegion",
                "floetteeternal",
            ),
            ("pikachu", "notarealpokemon", "mew"),
        )
        for roster in rosters:
            value = trick_room_probability(roster)
            assert 0.0 <= value <= 1.0
