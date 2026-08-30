"""League fine-tune guards: the FP save dir doubles as the opponent pool, so
role quarantine must survive the integer-stem naming that strips sidecars.

These tests pin the two layers: build_league.py refuses bad sources up front
(role allow-list, stale sidecars, content ban list), and verify_league_dir()
-- run by vgc_bench.train on every launch -- re-checks the pool by content so
a hand-copied eval-only holdout or a stray non-checkpoint file can never
reach a training rollout.
"""

import hashlib
import json
from pathlib import Path

import pytest

from build_league import (
    build_league,
    check_source,
    collect_banned_shas,
    write_league_team_weights,
)
from vgc_bench.src.utils import find_league_manifest, verify_league_dir


def _ckpt(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _stamp(path: Path, sha: str, role: str) -> None:
    sidecar = Path(str(path) + ".metadata.json")
    sidecar.write_text(json.dumps({"sha256": sha, "role": role}))


def _seed_sources(tmp_path: Path) -> tuple[dict[int, str], Path]:
    """A miniature league: one stamped BC source, one bare lineage, a champion."""
    bc = tmp_path / "bc" / "30.zip"
    _stamp(bc, _ckpt(bc, b"bc-weights"), "training_opponent")
    lineage = tmp_path / "lineage" / "3932160.zip"
    _ckpt(lineage, b"lineage-weights")
    champion = tmp_path / "champion.zip"
    _stamp(champion, _ckpt(champion, b"champion-weights"), "production")
    eval_root = tmp_path / "eval_B"
    _ckpt(eval_root / "seed2" / "30.zip", b"holdout-weights")
    sources = {
        100: str(bc),
        3932160: str(lineage),
        7864320: str(champion),
    }
    return sources, eval_root


def _build(tmp_path: Path) -> Path:
    sources, eval_root = _seed_sources(tmp_path)
    weights_src = tmp_path / "weights.json"
    weights_src.write_text(json.dumps({"our_team.txt": 35.0, "MB430.txt": 35.0}))
    dest = tmp_path / "results_league" / "saves_fp_hs_wt" / "reg_mb" / "seed1"
    build_league(
        dest=dest,
        sources=sources,
        resume_stem=7864320,
        eval_only_root=eval_root,
        weights_source=weights_src,
        weights_dest=tmp_path / "weights_league.json",
    )
    return dest


def test_check_source_refuses_disallowed_roles(tmp_path: Path) -> None:
    ckpt = tmp_path / "opp.zip"
    _stamp(ckpt, _ckpt(ckpt, b"weights"), "eval_opponent")
    with pytest.raises(SystemExit, match="role='eval_opponent'"):
        check_source(ckpt)


def test_check_source_refuses_eval_only_by_sidecar(tmp_path: Path) -> None:
    ckpt = tmp_path / "holdout.zip"
    _stamp(ckpt, _ckpt(ckpt, b"holdout"), "eval_only")
    with pytest.raises(SystemExit, match="eval_only"):
        check_source(ckpt)


def test_check_source_refuses_stale_sidecar(tmp_path: Path) -> None:
    ckpt = tmp_path / "opp.zip"
    _ckpt(ckpt, b"new-weights")
    _stamp(ckpt, hashlib.sha256(b"old-weights").hexdigest(), "training_opponent")
    with pytest.raises(SystemExit, match="sidecar sha256"):
        check_source(ckpt)


def test_build_refuses_banned_content_under_innocent_name(tmp_path: Path) -> None:
    sources, eval_root = _seed_sources(tmp_path)
    smuggled = tmp_path / "smuggled.zip"
    _stamp(smuggled, _ckpt(smuggled, b"holdout-weights"), "training_opponent")
    sources[200] = str(smuggled)
    weights_src = tmp_path / "weights.json"
    weights_src.write_text(json.dumps({"our_team.txt": 35.0}))
    with pytest.raises(SystemExit, match="banned checkpoint by content"):
        build_league(
            dest=tmp_path / "league",
            sources=sources,
            resume_stem=7864320,
            eval_only_root=eval_root,
            weights_source=weights_src,
            weights_dest=tmp_path / "weights_league.json",
        )


def test_collect_banned_shas_covers_every_epoch(tmp_path: Path) -> None:
    root = tmp_path / "eval_B"
    shas = {_ckpt(root / "seed2" / f"{i}.zip", b"epoch-%d" % i) for i in range(3)}
    assert set(collect_banned_shas(root)) == shas


def test_built_league_passes_verification(tmp_path: Path) -> None:
    dest = _build(tmp_path)
    manifest = find_league_manifest(dest)
    assert manifest is not None and manifest.parent.name == "results_league"
    verify_league_dir(dest)  # must not raise
    league_weights = json.loads((tmp_path / "weights_league.json").read_text())
    assert league_weights["our_team.txt"] == 0.0
    assert league_weights["MB430.txt"] == 35.0


def test_verify_rejects_smuggled_holdout_by_content(tmp_path: Path) -> None:
    dest = _build(tmp_path)
    (dest / "500.zip").write_bytes(b"holdout-weights")
    with pytest.raises(SystemExit, match="banned checkpoint by content"):
        verify_league_dir(dest)


def test_verify_rejects_unlisted_seed_stem(tmp_path: Path) -> None:
    dest = _build(tmp_path)
    (dest / "600.zip").write_bytes(b"hand-copied-anything")
    with pytest.raises(SystemExit, match="not\\s+listed"):
        verify_league_dir(dest)


def test_verify_rejects_modified_seed_checkpoint(tmp_path: Path) -> None:
    dest = _build(tmp_path)
    (dest / "100.zip").write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="does not match the sha256"):
        verify_league_dir(dest)


def test_verify_rejects_stray_files(tmp_path: Path) -> None:
    dest = _build(tmp_path)
    (dest / ".DS_Store").write_bytes(b"finder junk")
    with pytest.raises(SystemExit, match="integer-stem"):
        verify_league_dir(dest)


def test_verify_exempts_self_history_above_resume_stem(tmp_path: Path) -> None:
    dest = _build(tmp_path)
    (dest / "8847360.zip").write_bytes(b"fresh self-save, no manifest entry")
    verify_league_dir(dest)  # must not raise


def test_verify_noop_without_manifest(tmp_path: Path) -> None:
    legacy = tmp_path / "results_repaired" / "saves_fp_hs_wt" / "reg_mb" / "seed1"
    legacy.mkdir(parents=True)
    (legacy / ".DS_Store").write_bytes(b"junk tolerated in legacy dirs")
    verify_league_dir(legacy)  # must not raise


def test_write_league_team_weights_requires_our_team_entry(tmp_path: Path) -> None:
    src = tmp_path / "weights.json"
    src.write_text(json.dumps({"MB1.txt": 1.0}))
    with pytest.raises(SystemExit, match="no our_team.txt"):
        write_league_team_weights(src, tmp_path / "out.json")


def test_tr_boost_multiplies_only_tr_likely_teams(tmp_path: Path) -> None:
    teams = tmp_path / "teams"
    teams.mkdir()
    # Property-derived from the real joint sets: Farigiraf is a dedicated
    # setter (per-species TR rate ~0.97); Raichu is not.
    (teams / "tr.txt").write_text("Farigiraf @ Leftovers\nAbility: Armor Tail\n")
    (teams / "fast.txt").write_text("Raichu @ Focus Sash\nAbility: Static\n")
    src = tmp_path / "weights.json"
    src.write_text(
        json.dumps({"our_team.txt": 35.0, "tr.txt": 10.0, "fast.txt": 10.0})
    )
    out = tmp_path / "out.json"
    boosted = write_league_team_weights(src, out, tr_boost=3.0, teams_dir=teams)
    weights = json.loads(out.read_text())
    assert boosted == 1
    assert weights["tr.txt"] == 30.0
    assert weights["fast.txt"] == 10.0
    assert weights["our_team.txt"] == 0.0
