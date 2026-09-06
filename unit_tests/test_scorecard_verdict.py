"""The screening tier's non-regression rule is advisory (a fresh-seed
confirmation decides borderline arms); the promotion tier's is hard. A
neutral candidate must not be failed one time in three by five n=1,000 arms."""

import json
from pathlib import Path

from evaluation.scorecard_verdict import (
    neutral_false_fail,
    status_for,
    verdict,
    weighted_deltas,
)


def _battery(
    path: Path, cand_rate: float, base_rate: float, n: int = 1000, seed: int = 0
) -> None:
    # exact win counts, wins interleaved so the paired differences are spread
    c_wins, b_wins = round(cand_rate * n), round(base_rate * n)
    c_rows = [{"won": (i * c_wins) % n < c_wins} for i in range(n)]
    b_rows = [{"won": ((i + seed) * b_wins) % n < b_wins} for i in range(n)]
    payload = {
        "arms": {
            "distilled_policy": {
                "wins": sum(r["won"] for r in c_rows),
                "battles": n,
                "battle_results": c_rows,
            },
            "champion_policy": {
                "wins": sum(r["won"] for r in b_rows),
                "battles": n,
                "battle_results": b_rows,
            },
        }
    }
    path.write_text(json.dumps(payload))


def _dir(
    tmp_path: Path, rates: dict[str, tuple[float, float]], tier: str = "screening"
) -> Path:
    d = tmp_path / tier
    d.mkdir()
    for i, (arm, (c, b)) in enumerate(rates.items()):
        _battery(d / f"battery_{arm}.json", c, b, seed=i)
    (d / "scorecard.json").write_text(json.dumps({"tier": tier}))
    return d


def test_status_bands() -> None:
    assert status_for(-3.5, "screening", -2.0, 1.0) == "breach"
    assert status_for(-2.7, "screening", -2.0, 1.0) == "confirm"
    assert status_for(-1.2, "screening", -2.0, 1.0) == "confirm"
    assert status_for(-0.9, "screening", -2.0, 1.0) == "clear"
    assert status_for(-2.1, "promotion", -2.0, 1.0) == "breach"
    assert status_for(-1.9, "promotion", -2.0, 1.0) == "clear"


def test_weighting_counts_the_human_arm_twice() -> None:
    weighted, equal = weighted_deltas(
        {
            "heuristic": -2.7,
            "frozen": -0.1,
            "rotation1": 2.2,
            "rotation2": 3.4,
            "human_bc": 0.9,
        }
    )
    assert abs(weighted - 0.7667) < 1e-3 and abs(equal - 0.74) < 1e-3


def test_neutral_false_fail_rate_is_large_at_screening_small_at_promotion() -> None:
    screening = neutral_false_fail([1.4] * 5, -2.0)
    promotion = neutral_false_fail([1.4 / 5**0.5] * 5, -2.0)
    assert 0.25 < screening < 0.45
    assert promotion < 0.01


def test_screening_flags_a_borderline_arm_for_confirmation(tmp_path: Path) -> None:
    d = _dir(
        tmp_path,
        {"heuristic": (0.882, 0.909), "frozen": (0.87, 0.87), "human_bc": (0.86, 0.85)},
    )
    result = verdict(d)
    assert result["tier"] == "screening"
    assert result["arms"]["heuristic"]["paired"] is True
    assert result["arms"]["heuristic"]["status"] == "confirm"
    assert result["outcome"] == "CONFIRM" and result["confirm_needed"] == ["heuristic"]


def test_confirmation_reading_replaces_screening_and_is_judged_hard(
    tmp_path: Path,
) -> None:
    d = _dir(
        tmp_path,
        {"heuristic": (0.882, 0.909), "frozen": (0.87, 0.87), "human_bc": (0.86, 0.85)},
    )
    bad = tmp_path / "confirm_bad.json"
    _battery(bad, 0.878, 0.919, n=1500, seed=7)
    assert verdict(d, {"heuristic": bad})["outcome"] == "FAIL"
    good = tmp_path / "confirm_good.json"
    _battery(good, 0.90, 0.905, n=1500, seed=8)
    result = verdict(d, {"heuristic": good})
    assert result["arms"]["heuristic"]["source"] == "confirmation"
    assert result["outcome"] == "PASS"


def test_promotion_tier_is_hard(tmp_path: Path) -> None:
    d = _dir(
        tmp_path,
        {"heuristic": (0.882, 0.909), "frozen": (0.87, 0.87)},
        tier="promotion",
    )
    result = verdict(d)
    assert result["arms"]["heuristic"]["status"] == "breach"
    assert result["outcome"] == "FAIL"


def test_negative_weighted_fails_even_without_a_breach(tmp_path: Path) -> None:
    d = _dir(
        tmp_path,
        {
            "heuristic": (0.897, 0.905),
            "frozen": (0.862, 0.87),
            "human_bc": (0.842, 0.85),
        },
    )
    assert verdict(d)["weighted_delta_pp"] < 0
    assert verdict(d)["outcome"] == "FAIL"
