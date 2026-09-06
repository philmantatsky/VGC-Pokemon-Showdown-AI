"""Meta-game solve for the Nash-weighted league: known equilibria, copy
rounding, and cell bookkeeping."""

import json
from pathlib import Path

import numpy as np

from training.meta_game import (
    assemble,
    cell_path,
    multiplicities,
    solve_and_write,
    solve_zero_sum,
)


def test_rock_paper_scissors_is_uniform() -> None:
    m = np.array([[0.5, 0.0, 1.0], [1.0, 0.5, 0.0], [0.0, 1.0, 0.5]])
    eq = solve_zero_sum(m)
    assert np.allclose(eq["row"], [1 / 3] * 3, atol=1e-6)
    assert np.allclose(eq["col"], [1 / 3] * 3, atol=1e-6)
    assert abs(eq["value"] - 0.5) < 1e-6


def test_dominant_column_takes_all_the_weight() -> None:
    # column 1 beats every row harder than column 0 does
    m = np.array([[0.7, 0.4], [0.8, 0.3]])
    eq = solve_zero_sum(m)
    assert eq["col"][1] > 0.999
    assert abs(eq["value"] - 0.4) < 1e-6  # row 0 is the better row vs column 1


def test_equilibrium_spreads_over_columns_that_beat_different_rows() -> None:
    # column 0 beats row 0 only; column 1 beats row 1 only: the mixture must
    # use both, and it does not collapse onto the newest adversary.
    m = np.array([[0.3, 0.7], [0.7, 0.3]])
    eq = solve_zero_sum(m)
    assert np.allclose(eq["col"], [0.5, 0.5], atol=1e-6)
    assert abs(eq["value"] - 0.5) < 1e-6


def test_constant_matrix_gives_a_valid_distribution() -> None:
    eq = solve_zero_sum(np.full((3, 4), 0.55))
    assert abs(sum(eq["col"]) - 1) < 1e-9 and min(eq["col"]) >= 0
    assert abs(eq["value"] - 0.55) < 1e-6


def test_multiplicities_sum_to_copies() -> None:
    assert multiplicities([0.5, 0.3, 0.2], 12) == [6, 4, 2]
    assert sum(multiplicities([0.34, 0.33, 0.33], 10)) == 10
    assert multiplicities([0.0, 1.0], 5) == [0, 5]
    assert sum(multiplicities([0.0, 0.0], 4)) == 4  # degenerate -> uniform


def test_cells_assemble_and_solve_from_disk(tmp_path: Path) -> None:
    config = {
        "rows": {"a": "a.zip", "b": "b.zip"},
        "cols": {"x": "x.zip", "y": "y.zip"},
        "n": 10,
        "copies": 4,
    }
    rates = {("a", "x"): 0.3, ("a", "y"): 0.7, ("b", "x"): 0.7, ("b", "y"): 0.3}
    for (r, c), rate in rates.items():
        p = cell_path(tmp_path, r, c)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {"arms": {"champion_policy": {"wins": int(rate * 10), "battles": 10}}}
            )
        )
    m, rows, cols = assemble(config, tmp_path)
    assert rows == ["a", "b"] and cols == ["x", "y"]
    assert np.allclose(m, [[0.3, 0.7], [0.7, 0.3]])
    report = solve_and_write(config, tmp_path)
    assert report["suggested_multiplicity"] == {"x": 2, "y": 2}
    assert (tmp_path / "meta_game.json").exists()
