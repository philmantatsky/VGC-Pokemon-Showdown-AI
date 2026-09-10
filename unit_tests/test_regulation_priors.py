"""Opponent priors follow the regulation, and the top-500 Elo floor knows Reg M-C.

Before 2026-09-10 the floor lookup treated every non-"regmbbo3" battle id as
Reg M-B (floor 1655), which would have dropped the entire Reg M-C corpus
(median Elo ~1180) from every prior trainer without a single warning.
"""

from __future__ import annotations

import os

from vgc_bench.src.opponent_preview import TOP_500_ELO_FLOORS, top_500_rating_floor
from vgc_bench.src.utils import prior_path


def test_floor_matches_longest_format_token() -> None:
    assert top_500_rating_floor("gen9championsvgc2026regmb-1") == 1655
    assert top_500_rating_floor("gen9championsvgc2026regmbbo3-1") == 1432
    assert top_500_rating_floor("gen9championsvgc2026regmc-2677944137") == 1140
    assert top_500_rating_floor("gen9championsvgc2026regmcbo3-9") == 1045


def test_unknown_format_uses_loosest_floor() -> None:
    assert top_500_rating_floor("gen9championsvgc2027regmd-1") == min(
        TOP_500_ELO_FLOORS.values()
    )


def test_prior_path_prefers_regulation_file_and_falls_back(monkeypatch) -> None:
    present = {"data/opponent_move_top500_regmc.pt"}
    monkeypatch.setattr(os.path, "exists", lambda p: p in present)
    assert prior_path("move", "mc") == "data/opponent_move_top500_regmc.pt"
    assert prior_path("switch", "mc") == "data/opponent_switch_top500_regmb.pt"
    assert prior_path("preview", "mb") == "data/opponent_preview_top500_regmb.pt"
