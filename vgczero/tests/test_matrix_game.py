import numpy as np

from vgczero.matrix_game import exploitability, solve


def test_rock_paper_scissors():
    rps = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], float)
    x, y, v = solve(rps, 3000)
    assert np.allclose(x, 1 / 3, atol=0.01) and np.allclose(y, 1 / 3, atol=0.01)
    assert abs(v) < 0.01


def test_pure_saddle_point():
    p = np.array([[3, 1], [2, 0]], float)  # row 0 dominates, column 1 dominates
    x, y, v = solve(p, 500)
    assert x[0] > 0.99 and y[1] > 0.99 and abs(v - 1) < 0.02


def test_batch_low_exploitability():
    b = np.random.default_rng(0).normal(size=(32, 8, 8))
    x, y, _ = solve(b, 2000)
    assert exploitability(b, x, y).max() < 0.05


def test_masks_respected():
    p = np.random.default_rng(1).normal(size=(4, 5, 5))
    cm = np.ones((4, 5), bool)
    cm[:, 3:] = False
    x, y, _ = solve(p, 500, col_mask=cm)
    assert np.all(y[:, 3:] == 0)
