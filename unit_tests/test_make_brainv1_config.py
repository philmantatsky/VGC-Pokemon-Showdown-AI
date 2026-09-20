"""training/make_brainv1_config.py: the deployed-heavy pool shape, the resume
stem as the highest seeded stem, separate artifacts for the specialist round,
the save swap at stem 700, and both Reg M-C holdout clones banned."""

from __future__ import annotations

from training.make_brainv1_config import (
    DEFAULT_DEST,
    DEPLOYED,
    LEAGUE1_HISTORY,
    OLD_CHAMPION,
    build_config,
)


def test_default_pool_is_deployed_heavy() -> None:
    config = build_config(init="init.zip", clone="clone.zip")
    sources = config["sources"]
    assert [sources[s] for s in ("100", "200")] == ["clone.zip", "clone.zip"]
    assert sources["300"] == OLD_CHAMPION and sources["400"] == LEAGUE1_HISTORY
    assert [sources[s] for s in ("500", "600", "700")] == [DEPLOYED] * 3
    assert [sources[s] for s in ("800", "900", "12779520")] == ["init.zip"] * 3
    assert config["dest"] == DEFAULT_DEST
    assert max(int(s) for s in sources) == config["resume_stem"] == 12779520


def test_specialist_config_keeps_artifacts_separate() -> None:
    config = build_config(
        init="save7.zip",
        clone="clone.zip",
        resume_stem=19660800,
        dest="results_brainv1_spec/saves_fp_hs_wt/reg_mc/seed1",
        weights_dest="data/team_weights_regmc_brainv1_spec.json",
        swap_deployed="save8.zip",
    )
    sources = config["sources"]
    assert sources["700"] == "save8.zip"
    assert [sources[s] for s in ("500", "600")] == [DEPLOYED] * 2
    assert sources["19660800"] == "save7.zip"
    assert max(int(s) for s in sources) == 19660800
    assert config["dest"].startswith("results_brainv1_spec/")
    assert config["weights_dest"].endswith("_spec.json")


def test_holdout_clones_are_banned() -> None:
    roots = build_config(init="i.zip", clone="c.zip")["eval_only_roots"]
    for root in ("results_bc/eval_B", "results_bc/eval_D", "results_bc/eval_mcB"):
        assert root in roots
    assert "results_bc/eval_mcB_20260913" in roots
    custom = build_config(init="i.zip", clone="c.zip", eval_only_clones=["x/holdout"])
    assert custom["eval_only_roots"][-1] == "x/holdout"
