"""The outcome-dataset generator must tolerate isolated game failures.

It used to exit nonzero whenever ANY game failed, even with a complete dataset
(98 failures in 19,902 games), which killed a chained evaluator pipeline
overnight before training ever started. Failures below a budget (default 1% of
the requested games) are now tolerated; a short dataset or a failure flood
still exits nonzero.
"""

import pytest

from datagen.generate_outcome_dataset import raise_if_generation_unusable


def test_isolated_failures_within_default_budget_pass() -> None:
    raise_if_generation_unusable(
        completed=19_902, failures=98, games=20_000, minimum_games=15_000
    )  # must not raise: 98 < 1% of 20,000


def test_failure_flood_exits_nonzero() -> None:
    with pytest.raises(SystemExit, match="failed games"):
        raise_if_generation_unusable(
            completed=19_000, failures=1_000, games=20_000, minimum_games=15_000
        )


def test_explicit_budget_overrides_default() -> None:
    with pytest.raises(SystemExit, match="failed games"):
        raise_if_generation_unusable(
            completed=19_902,
            failures=98,
            games=20_000,
            minimum_games=15_000,
            max_failed_games=50,
        )


def test_short_dataset_exits_nonzero() -> None:
    with pytest.raises(SystemExit, match="minimum"):
        raise_if_generation_unusable(
            completed=9_000, failures=0, games=20_000, minimum_games=15_000
        )
