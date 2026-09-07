"""Round-4 config from the equilibrium: copies sum, floors/caps, unique stems,
the deployed brain at the resume stem, both holdouts banned."""

import json
from pathlib import Path

from training.make_league4_config import DEPLOYED, RESUME_STEM, build_config


def _meta(tmp_path: Path, eq: dict[str, float]) -> Path:
    cols = {
        name: (DEPLOYED if name == "deployed" else f"ckpt/{name}.zip") for name in eq
    }
    (tmp_path / "config.json").write_text(json.dumps({"cols": cols}))
    (tmp_path / "meta_game.json").write_text(
        json.dumps({"cols": list(eq), "col_equilibrium": eq})
    )
    return tmp_path


def test_copies_follow_the_equilibrium_with_floors_and_caps(tmp_path: Path) -> None:
    meta = _meta(
        tmp_path,
        {
            "bc_mix_A": 0.0,
            "bc_mix_C": 0.0,
            "bc_mix_AC": 0.05,
            "exploiter_final": 0.6,
            "deployed": 0.25,
            "old_champion": 0.1,
        },
    )
    cfg = build_config(meta, copies=12, tr_boost=1.5)
    counts = cfg["copies"]
    assert (
        counts["exploiter_final"] == 2 and "cap" in cfg["overrides"]["exploiter_final"]
    )
    assert counts["bc_mix_C"] == 1 and "floor" in cfg["overrides"]["bc_mix_C"]
    assert counts["bc_mix_AC"] >= 1
    assert cfg["tr_boost"] == 1.5
    assert cfg["eval_only_roots"] == ["results_bc/eval_B", "results_bc/eval_D"]


def test_stems_are_unique_below_the_resume_and_deployed_is_the_resume(
    tmp_path: Path,
) -> None:
    meta = _meta(tmp_path, {"deployed": 0.5, "exploiter_final": 0.1, "bc_mix_C": 0.4})
    cfg = build_config(meta, copies=10)
    stems = sorted(int(s) for s in cfg["sources"])
    assert stems[-1] == RESUME_STEM and cfg["sources"][str(RESUME_STEM)] == DEPLOYED
    assert len(stems) == len(set(stems)) and all(s < RESUME_STEM for s in stems[:-1])
    deployed_copies = sum(1 for p in cfg["sources"].values() if p == DEPLOYED)
    assert deployed_copies == cfg["copies"]["deployed"]  # resume file counts as one
