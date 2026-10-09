"""Stage-A hardening: knowledge_obs resolution, run-config echo, preview rules.

The knowledge_obs flag must never default silently (a silent False zeroed the
champion's 24 knowledge features for a whole ladder era), every ladder batch
must be auditable from <replay_dir>/run_config.json, and the Trick Room lookup
must stay team-agnostic.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ladder_ourteam import (
    build_parser,
    check_playbook,
    material_config,
    record_run_config,
    resolve_knowledge_obs,
    search_planner_config,
)
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

    def test_set_prior_data_is_material(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VGC_SET_PRIOR_REG", raising=False)
        path = record_run_config(tmp_path, _args(reg="mc"), "abc123", "hard")
        material = json.loads(path.read_text())["runs"][0]["material"]
        assert material["set_prior_reg"] == "mc"
        monkeypatch.setenv("VGC_SET_PRIOR_REG", "mb")
        with pytest.raises(SystemExit, match="set_prior_reg"):
            record_run_config(tmp_path, _args(reg="mc"), "abc123", "hard")

    def test_sticky_corrections_are_material_only_when_on(self, tmp_path):
        """A directory recorded before the flag existed keeps accepting runs with
        it off; turning it on is a different configuration (2026-09-26)."""
        record_run_config(tmp_path, _args(), "abc123", "hard")
        path = record_run_config(
            tmp_path, _args(sticky_corrections=False), "abc123", "hard"
        )
        material = json.loads(path.read_text())["runs"][-1]["material"]
        assert "sticky_corrections" not in material
        with pytest.raises(SystemExit, match="sticky_corrections"):
            record_run_config(
                tmp_path, _args(sticky_corrections=True), "abc123", "hard"
            )

    def test_search_flags_are_material_only_when_not_default(self, tmp_path):
        """2026-10-04: --search-solution / --search-leaf entered the material with
        their defaults, so a directory recorded before them (the running challenge
        listener's) would have refused its next reconnect restart."""
        record_run_config(tmp_path, _args(), "abc123", "hard")
        path = record_run_config(
            tmp_path,
            _args(search_solution="risk", search_leaf="outcome"),
            "abc123",
            "hard",
        )
        material = json.loads(path.read_text())["runs"][-1]["material"]
        assert "search_solution" not in material and "search_leaf" not in material
        with pytest.raises(SystemExit, match="search_solution"):
            record_run_config(tmp_path, _args(search_solution="nash"), "abc123", "hard")
        with pytest.raises(SystemExit, match="search_leaf"):
            record_run_config(tmp_path, _args(search_leaf="critic"), "abc123", "hard")

    def test_the_matrix_search_flags_are_material_only_when_not_default(self, tmp_path):
        """2026-10-04 (evening): --search-anchor / --search-argmax / --search-replies /
        --search-leaf-calibration / --search-streams; 2026-10-09:
        --search-reply-prior. A directory recorded before
        them accepts a run that leaves them alone; setting one is another
        configuration, and a calibration is recorded with its hash."""
        record_run_config(tmp_path, _args(), "abc123", "hard")
        defaults = dict(
            search_anchor=0.0,
            search_argmax=False,
            search_replies=6,
            search_leaf_calibration="",
            search_streams=0,
            search_reply_prior="models",
        )
        path = record_run_config(tmp_path, _args(**defaults), "abc123", "hard")
        material = json.loads(path.read_text())["runs"][-1]["material"]
        assert not set(defaults) & set(material)
        calibration = tmp_path / "calibration.json"
        calibration.write_text("{}")
        changed = dict(
            search_anchor=0.07,
            search_argmax=True,
            search_replies=8,
            search_streams=4,
            search_leaf_calibration=str(calibration),
            search_reply_prior="brain",
        )
        for name, value in changed.items():
            with pytest.raises(SystemExit, match=name):
                record_run_config(
                    tmp_path, _args(**{**defaults, name: value}), "abc123", "hard"
                )
        trial = record_run_config(
            tmp_path / "trial", _args(**changed), "abc123", "hard"
        )
        material = json.loads(trial.read_text())["runs"][-1]["material"]
        assert {name: material[name] for name in changed} == changed
        assert (
            material["search_leaf_calibration_sha256"]
            == hashlib.sha256(b"{}").hexdigest()
        )

    def test_a_playbook_is_material_with_its_sha_only_when_on(self, tmp_path):
        """2026-09-27: the plan cards themselves are the configuration -- an edited
        playbook may not share a replay dir with the one before it."""
        record_run_config(tmp_path, _args(), "abc123", "hard")
        path = record_run_config(tmp_path, _args(playbook=""), "abc123", "hard")
        material = json.loads(path.read_text())["runs"][-1]["material"]
        assert "playbook" not in material and "playbook_sha256" not in material
        book = tmp_path / "playbook.json"
        book.write_text('{"cards": [1]}')
        with pytest.raises(SystemExit, match="playbook"):
            record_run_config(tmp_path, _args(playbook=str(book)), "abc123", "hard")
        fresh = tmp_path / "fresh"
        path = record_run_config(fresh, _args(playbook=str(book)), "abc123", "hard")
        material = json.loads(path.read_text())["runs"][-1]["material"]
        digest = hashlib.sha256(book.read_bytes()).hexdigest()
        assert material["playbook_sha256"] == digest
        book.write_text('{"cards": [2]}')
        with pytest.raises(SystemExit, match="playbook_sha256"):
            record_run_config(fresh, _args(playbook=str(book)), "abc123", "hard")

    def test_a_playbook_for_another_team_is_refused(self, tmp_path):
        """Its cards name our Pokemon: on another team every preview would fall
        back silently, so the ladder refuses to start instead."""
        team, other = tmp_path / "T6.txt", tmp_path / "T7.txt"
        team.write_text("ours")
        other.write_text("another")
        book = tmp_path / "playbook.json"
        book.write_text(json.dumps({"team": str(team), "cards": []}))
        check_playbook(book, team)
        with pytest.raises(SystemExit, match="written for"):
            check_playbook(book, other)
        with pytest.raises(SystemExit, match="does not exist"):
            check_playbook(tmp_path / "missing.json", team)
        check_playbook(
            Path("data/playbook_t6.json"), Path("teams/candidates_mc/T6.txt")
        )


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


def test_learned_preview_needs_an_explicit_model():
    """--learned_preview alone would let the opponent-model default (the general
    top-500 predictor) choose our preview; it must refuse before any connection."""
    root = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith("SHOWDOWN_")}
    env["SHOWDOWN_USERNAME"] = "unit-test-no-login"  # never connects: refused first
    result = subprocess.run(
        [
            sys.executable,
            "ladder_ourteam.py",
            "--checkpoint",
            "results_deployed/champion_mc_T6.zip",
            "--reg",
            "mc",
            "--our_team",
            "teams/candidates_mc/T6.txt",
            "--learned_preview",
            "--n_games",
            "1",
            "--replay_dir",
            "/nonexistent/never_created",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode != 0
    assert "--learned_preview needs an explicit --preview_model" in result.stderr
    assert not Path("/nonexistent/never_created").exists()


# What the deployed challenge listener's replay directory records
# (challenge_replays_mc_deployed_T6ep_guards14/run_config.json, 2026-10-04): the keys
# of its material configuration, and the arguments the launcher gives it.
DEPLOYED_LISTENER_ARGV = [
    "--checkpoint",
    "results_deployed/champion_mc_T6ep.zip",
    "--reg",
    "mc",
    "--our_team",
    "teams/candidates_mc/T6e.txt",
    "--guards-extra",
    "resisted_target,overkill_split,dominated_weather_ball_weather,dominated_attack,"
    "dominated_throat_chop,dominated_spread,focus_boosted,wide_guard,"
    "drop_free_finish,fake_out_partner_acts,switch_the_crippled,threat_first2,"
    "wasted_fake_out,throat_chop_main_threat",
    "--challenges",
    "--n_games",
    "1000",
    "--replay_dir",
    "challenge_replays_mc_deployed_T6ep_guards14",
    "--learned_preview",
    "--preview_model",
    "data/preview_t6_focus_20260923.pt",
    "--sticky-corrections",
    "--sheet-preview",
]
DEPLOYED_LISTENER_MATERIAL_KEYS = {
    "bench_species",
    "chance_samples",
    "checkpoint",
    "checkpoint_sha256",
    "deep_root_width",
    "determinizations",
    "device",
    "guard_profile",
    "guard_profile_resolved",
    "guards_extra",
    "knowledge_obs",
    "learned_preview",
    "mask_immunities",
    "min_deep_coverage",
    "mixing",
    "mixing_keep_corrections",
    "mixing_last_turn",
    "mixing_min_ratio",
    "mixing_temperature",
    "mixing_top_k",
    "move_model",
    "moveset_prior",
    "no_guards",
    "no_immunity_mask",
    "no_moveset_prior",
    "opening_wait",
    "opponent_aware",
    "our_team",
    "outcome_preview",
    "outcome_value",
    "planned_preview",
    "ponder",
    "ponder_budget",
    "ponder_chance_samples",
    "ponder_choices",
    "preview_determinizations",
    "preview_model",
    "preview_outcome_model",
    "preview_search_budget",
    "reg",
    "residual_ranker",
    "screen_budget",
    "search",
    "search_budget",
    "search_determinizations",
    "search_every_turn",
    "set_prior_reg",
    "sheet_preview",
    "stable_lead",
    "sticky_corrections",
    "switch_model",
    "tempo_aware",
}


def test_the_launcher_records_what_the_deployed_listener_already_recorded():
    """A launcher flag that enters the material at its default makes every replay
    directory from before refuse its next run -- the running challenge listener's on
    its next reconnect restart (2026-10-04, twice). Parsing the listener's own
    arguments must give exactly the keys its directory holds."""
    args = build_parser().parse_args(DEPLOYED_LISTENER_ARGV)
    material = material_config(args, "sha", "hard")
    assert set(material) == DEPLOYED_LISTENER_MATERIAL_KEYS
    assert material["search"] is False and material["device"] == "mps"


def test_search_flags_build_the_planner_the_head_to_head_measured():
    """The matrix search as evaluation/mirror_guard_ab.py ran it on 2026-10-04 (V6):
    a ladder trial has to play that configuration, not one beside it."""
    args = build_parser().parse_args(
        [
            "--search",
            "--search-every-turn",
            "--search-solution",
            "nash",
            "--search-leaf",
            "critic",
            "--search-anchor",
            "0.07",
            "--search-argmax",
            "--search-replies",
            "8",
            "--search-determinizations",
            "4",
            "--search-streams",
            "4",
            "--device",
            "cpu",
        ]
    )
    config = search_planner_config(args)
    assert (config.solution, config.nash_anchor, config.nash_sample) == (
        "nash",
        0.07,
        False,
    )
    assert (config.root_width, config.opponent_width) == (6, 8)
    assert (config.depth, config.continuation_width, config.replacement_width) == (
        2,
        3,
        2,
    )
    assert (config.chance_samples, config.deep_root_width, config.max_nodes) == (
        1,
        4,
        5000,
    )
    assert (config.anytime, config.screen_budget_s, config.time_budget_s) == (
        True,
        2.0,
        8.0,
    )
    assert config.nash_likeliest and config.nash_prior_mix == 0
    assert config.nash_champion_boost == 2.0
    assert (args.search_streams, args.search_determinizations, args.device) == (
        4,
        4,
        "cpu",
    )
    # without the new flags the planner is the one every earlier search run used
    old = search_planner_config(build_parser().parse_args(["--search"]))
    assert (old.solution, old.nash_anchor, old.nash_sample, old.opponent_width) == (
        "risk",
        0.0,
        True,
        6,
    )


def test_the_launchers_help_can_be_printed():
    """argparse reads a bare % in a help string as a format: one "53.4% [..." made
    --help raise (found 2026-10-04; a normal run never formats the help)."""
    text = " ".join(build_parser().format_help().split())
    assert "--search-anchor" in text and "53.4% [51.2, 55.5]" in text
