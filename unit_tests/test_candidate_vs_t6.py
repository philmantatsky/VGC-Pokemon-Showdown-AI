"""evaluation/run_candidate_vs_t6.py: the team-preview variety summary."""

from __future__ import annotations

from evaluation.run_candidate_vs_t6 import preview_profile


def _row(opponent: str, lead: list[str], bring: list[str]) -> dict:
    return {"opponent": opponent, "lead_species": lead, "bring_species": bring}


def test_preview_profile_counts_distinct_plans_and_adaptive_rosters() -> None:
    fixed = [
        _row("A", ["torkoal", "farigiraf"], ["torkoal", "farigiraf", "x", "y"])
    ] * 3
    assert preview_profile(fixed) == {
        "games": 3,
        "distinct_lead_pairs": 1,
        "distinct_brings": 1,
        "top_lead": "farigiraf+torkoal",
        "top_lead_share": 1.0,
        "rosters_with_more_than_one_lead": 0,
        "leads": {"farigiraf+torkoal": 3},
    }
    varied = fixed + [
        _row("A", ["charizard", "venusaur"], ["charizard", "venusaur", "x", "y"])
    ]
    profile = preview_profile(varied)
    assert profile["distinct_lead_pairs"] == 2 and profile["distinct_brings"] == 2
    assert profile["rosters_with_more_than_one_lead"] == 1
    assert profile["top_lead_share"] == 0.75
    assert preview_profile([])["games"] == 0
