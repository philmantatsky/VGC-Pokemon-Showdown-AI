"""The zero-sum matrix-game solver (vgc_bench/src/matrix_game.py): the equilibrium
step of the one-turn search the two Reg M-C #1 bots use (RESEARCH_TOP_BOTS.md)."""

from __future__ import annotations

import numpy as np
import pytest

from vgc_bench.src.matrix_game import exploitability, solve


def test_matching_pennies_is_an_even_mix():
    x, y, value = solve(np.array([[1.0, -1.0], [-1.0, 1.0]]))
    np.testing.assert_allclose(x, [0.5, 0.5], atol=0.02)
    np.testing.assert_allclose(y, [0.5, 0.5], atol=0.02)
    assert value == pytest.approx(0.0, abs=0.02)


def test_a_dominant_action_takes_all_the_mass():
    # row 0 beats row 1 against every reply: Protect-into-nothing vs a real attack
    x, _, value = solve(np.array([[0.6, 0.4], [0.1, -0.2]]))
    assert x[0] > 0.99
    assert value == pytest.approx(0.4, abs=0.02)


def test_rock_paper_scissors_and_exploitability():
    rps = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    x, y, _ = solve(rps, iters=2000)
    np.testing.assert_allclose(x, [1 / 3] * 3, atol=0.02)
    assert exploitability(rps, x, y)[0] < 0.02
    assert exploitability(rps, np.array([1.0, 0, 0]), y)[0] > 0.5  # pure is exploitable


def test_batched_tables_and_padded_actions():
    tables = np.stack(
        [
            np.array([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
            np.array([[0.6, 0.4, 0.0], [0.1, -0.2, 0.0], [0.0, 0.0, 0.0]]),
        ]
    )
    rows = np.array([[True, True, False], [True, True, False]])
    cols = np.array([[True, True, False], [True, True, False]])
    x, y, value = solve(tables, row_mask=rows, col_mask=cols)
    assert np.all(x[:, 2] == 0) and np.all(y[:, 2] == 0)
    np.testing.assert_allclose(x[0, :2], [0.5, 0.5], atol=0.02)
    assert x[1, 0] > 0.99
    assert value.shape == (2,)
