"""The zero-sum matrix-game solver (vgc_bench/src/matrix_game.py): the equilibrium
step of the one-turn search the two Reg M-C #1 bots use (RESEARCH_TOP_BOTS.md)."""

from __future__ import annotations

import numpy as np
import pytest

from vgc_bench.src.matrix_game import exploitability, solve, solve_anchored


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


def test_anchored_solution_satisfies_both_sides_conditions():
    rng = np.random.default_rng(3)
    for temperature in (0.05, 0.2, 1.0):
        table = rng.uniform(-1, 1, (5, 4))
        ours, theirs = rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(4))
        x, y, value = solve_anchored(table, ours, theirs, temperature)
        want_x = ours * np.exp((table @ y) / temperature)
        want_y = theirs * np.exp(-(x @ table) / temperature)
        np.testing.assert_allclose(x, want_x / want_x.sum(), atol=1e-5)
        np.testing.assert_allclose(y, want_y / want_y.sum(), atol=1e-5)
        assert value == pytest.approx(float(x @ table @ y))


def test_anchor_ignores_a_noise_sized_edge_and_yields_to_a_real_one():
    # two of our moves against two replies that do not matter; the policy prefers
    # the first 0.512 to 0.117 (a decision from the 2026-10-04 audit, where the plain
    # equilibrium switched to the second for a payoff edge of 0.004)
    prior = [0.512, 0.117]
    noise = np.array([[0.373, 0.373], [0.377, 0.377]])
    plain, _, _ = solve(noise)
    assert plain[1] > 0.95
    anchored, _, _ = solve_anchored(noise, prior, [0.5, 0.5], 0.2)
    assert anchored[0] > 0.8
    found_win = np.array([[-0.845, -0.845], [1.0, 1.0]])
    anchored, _, _ = solve_anchored(found_win, prior, [0.5, 0.5], 0.2)
    assert anchored[1] > 0.99


def test_anchor_runs_from_the_plain_equilibrium_to_the_priors():
    pennies = np.array([[1.0, -1.0], [-1.0, 1.0]])
    cold, _, _ = solve_anchored(pennies, [0.8, 0.2], [0.5, 0.5], 0.02)
    np.testing.assert_allclose(cold, [0.5, 0.5], atol=0.02)
    warm, reply, _ = solve_anchored(pennies, [0.8, 0.2], [0.5, 0.5], 50.0)
    np.testing.assert_allclose(warm, [0.8, 0.2], atol=0.02)
    np.testing.assert_allclose(reply, [0.5, 0.5], atol=0.02)
    # the opponent is its prior leaning against us, not a perfect adversary
    _, leaning, _ = solve_anchored(pennies, [0.8, 0.2], [0.5, 0.5], 0.5)
    assert 0.5 < leaning[1] < 0.95


def test_anchored_solver_refuses_bad_input():
    table = np.zeros((2, 2))
    with pytest.raises(ValueError, match="temperature"):
        solve_anchored(table, [0.5, 0.5], [0.5, 0.5], 0.0)
    with pytest.raises(ValueError, match="prior"):
        solve_anchored(table, [0.5, 0.5, 0.0], [0.5, 0.5], 0.2)
    with pytest.raises(ValueError, match="prior"):
        solve_anchored(table, [0.0, 0.0], [0.5, 0.5], 0.2)
    with pytest.raises(ValueError, match="payoff"):
        solve_anchored(np.zeros((1, 2, 2)), [0.5, 0.5], [0.5, 0.5], 0.2)
