"""Zero-sum matrix games, solved by regret matching+ (vectorized over a batch).

payoff[b, i, j] is the row player's payoff when the row player plays i and the
column player plays j. Returns approximate Nash strategies (x for rows, y for
columns) and the game value. Regret matching+ with linear averaging converges
fast on the small games search produces (8x8).
"""

from __future__ import annotations

import numpy as np


def solve(payoff: np.ndarray, iters: int = 1000, row_mask: np.ndarray | None = None, col_mask: np.ndarray | None = None):
    """payoff [B, R, C] -> (x [B, R], y [B, C], value [B]).

    Masks (bool [B, R] / [B, C]) exclude padded actions.
    """
    P = np.asarray(payoff, dtype=np.float64)
    if P.ndim == 2:
        x, y, v = solve(P[None], iters, None if row_mask is None else row_mask[None], None if col_mask is None else col_mask[None])
        return x[0], y[0], v[0]
    B, R, C = P.shape
    rm = np.ones((B, R), bool) if row_mask is None else row_mask.astype(bool)
    cm = np.ones((B, C), bool) if col_mask is None else col_mask.astype(bool)
    reg_x = np.zeros((B, R))
    reg_y = np.zeros((B, C))
    avg_x = np.zeros((B, R))
    avg_y = np.zeros((B, C))

    def strat(reg, mask):
        pos = np.where(mask, np.maximum(reg, 0.0), 0.0)
        s = pos.sum(-1, keepdims=True)
        uni = mask / np.maximum(mask.sum(-1, keepdims=True), 1)
        return np.where(s > 0, pos / np.maximum(s, 1e-300), uni)

    for t in range(1, iters + 1):
        x = strat(reg_x, rm)
        y = strat(reg_y, cm)
        ux = np.einsum("brc,bc->br", P, y)  # row payoff of each row action
        uy = -np.einsum("brc,br->bc", P, x)  # column payoff of each column action
        vx = (x * ux).sum(-1, keepdims=True)
        vy = (y * uy).sum(-1, keepdims=True)
        reg_x = np.maximum(reg_x + (ux - vx), 0.0) * rm
        reg_y = np.maximum(reg_y + (uy - vy), 0.0) * cm
        avg_x += t * x
        avg_y += t * y
    x = avg_x / avg_x.sum(-1, keepdims=True)
    y = avg_y / avg_y.sum(-1, keepdims=True)
    value = np.einsum("br,brc,bc->b", x, P, y)
    return x, y, value


def exploitability(payoff: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """How much each player could gain by deviating (0 at an exact equilibrium)."""
    P = np.asarray(payoff, dtype=np.float64)
    best_row = np.einsum("brc,bc->br", P, y).max(-1)
    best_col = (-np.einsum("brc,br->bc", P, x)).max(-1)
    v = np.einsum("br,brc,bc->b", x, P, y)
    return (best_row - v) + (best_col + v)
