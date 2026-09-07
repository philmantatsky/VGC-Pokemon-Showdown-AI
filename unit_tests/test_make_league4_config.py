"""Round-4 config from the meta-game: copies sum, floors/caps with redistributed
mass, unique stems, the deployed brain at the resume stem, both holdouts
banned, and the degenerate pure-Nash case replaced by the hardness mixture."""

import json
from pathlib import Path

from training.make_league4_config import DEPLOYED, RESUME_STEM, build_config


def _meta(
    tmp_path: Path, eq: dict[str, float], deployed_row: dict[str, float] | None = None
) -> Path:
    cols = {
        name: (DEPLOYED if name == "deployed" else f"ckpt/{name}.zip") for name in eq
    }
    (tmp_path / "config.json").write_text(json.dumps({"cols": cols}))
    row = deployed_row or {name: 0.85 for name in eq}
    (tmp_path / "meta_game.json").write_text(
        json.dumps(
            {
                "cols": list(eq),
                "rows": ["deployed"],
                "payoff_row_win_rate": [[row[name] for name in eq]],
                "col_equilibrium": eq,
            }
        )
    )
    return tmp_path


def test_nash_mixture_applies_floors_and_cap_and_keeps_the_copy_count(
    tmp_path: Path,
) -> None:
    eq = {
        "bc_mix_A": 0.0,
        "bc_mix_C": 0.0,
        "bc_mix_AC": 0.05,
        "exploiter_final": 0.6,
        "deployed": 0.25,
        "old_champion": 0.1,
    }
    cfg = build_config(_meta(tmp_path, eq), copies=12, tr_boost=1.5, mixture="nash")
    counts = cfg["copies"]
    assert (
        counts["exploiter_final"] == 2 and "cap" in cfg["overrides"]["exploiter_final"]
    )
    assert counts["bc_mix_C"] == 2 and "floor" in cfg["overrides"]["bc_mix_C"]
    assert counts["bc_mix_AC"] == 2
    assert sum(counts.values()) == 12  # capped mass is redistributed, not lost
    assert cfg["tr_boost"] == 1.5
    assert cfg["eval_only_roots"] == ["results_bc/eval_B", "results_bc/eval_D"]


def test_stems_are_unique_below_the_resume_and_deployed_is_the_resume(
    tmp_path: Path,
) -> None:
    eq = {"deployed": 0.5, "exploiter_final": 0.1, "bc_mix_C": 0.4}
    cfg = build_config(_meta(tmp_path, eq), copies=10, mixture="nash")
    stems = sorted(int(s) for s in cfg["sources"])
    assert stems[-1] == RESUME_STEM and cfg["sources"][str(RESUME_STEM)] == DEPLOYED
    assert len(stems) == len(set(stems)) and all(s < RESUME_STEM for s in stems[:-1])
    deployed_copies = sum(1 for p in cfg["sources"].values() if p == DEPLOYED)
    assert deployed_copies == cfg["copies"]["deployed"]  # the resume file is one copy


def test_degenerate_equilibrium_is_replaced_by_the_hardness_mixture(
    tmp_path: Path,
) -> None:
    """The 2026-09-07 matrix: pure Nash = 100% exploiter. The hardness mixture keeps
    every member, weights the exploiter most, caps it, and floors the clones."""
    eq = {
        "bc_mix_A": 0.0,
        "bc_mix_C": 0.0,
        "bc_mix_AC": 0.0,
        "exploiter_final": 1.0,
        "deployed": 0.0,
        "r3b_final": 0.0,
        "old_champion": 0.0,
    }
    row = {
        "bc_mix_A": 0.85,
        "bc_mix_C": 0.853,
        "bc_mix_AC": 0.81,
        "exploiter_final": 0.40,
        "deployed": 0.877,
        "r3b_final": 0.833,
        "old_champion": 0.853,
    }
    cfg = build_config(_meta(tmp_path, eq, row), copies=12)
    counts = cfg["copies"]
    assert cfg["mixture"] == "hardness"
    assert counts["exploiter_final"] == 2
    assert counts["bc_mix_C"] >= 2 and counts["bc_mix_AC"] >= 2
    assert all(n >= 1 for n in counts.values())  # nobody is dropped
    assert sum(counts.values()) == 12
