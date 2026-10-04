"""Zero-sum matrix games solved by regret matching+ (2026-10-04).

The two bots that topped Reg M-C (mikumiku37, Nessie123; RESEARCH_TOP_BOTS.md) both
treat a doubles turn as the simultaneous-move game it is: a payoff table of our
candidate joint actions against the opponent's likely replies, solved for an
equilibrium mixed strategy that is then sampled. This is that solver, vectorized
over a batch of tables (one per hidden-information world), after the 60-line
version on the vgczero branch (origin/claude/nice-bohr-mvs9zh,
vgczero/python/vgczero/matrix_game.py). Nothing in the bot uses it yet: the
exact planner still scores actions by a risk blend over predicted replies.

payoff[b, i, j] is the row player's (our) payoff when we play i and the opponent
plays j. Regret matching+ with linear averaging converges quickly on the small
tables search produces (6x6 to 8x8).
"""

from __future__ import annotations

import numpy as np


def _strategy(regret: np.ndarray, mask: np.ndarray) -> np.ndarray:
    positive = np.where(mask, np.maximum(regret, 0.0), 0.0)
    total = positive.sum(-1, keepdims=True)
    uniform = mask / np.maximum(mask.sum(-1, keepdims=True), 1)
    return np.where(total > 0, positive / np.maximum(total, 1e-300), uniform)


def solve(
    payoff: np.ndarray,
    iters: int = 1000,
    row_mask: np.ndarray | None = None,
    col_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """payoff [B, R, C] (or [R, C]) -> (x [B, R], y [B, C], value [B]).

    ``row_mask`` / ``col_mask`` (bool [B, R] / [B, C]) exclude padded actions.
    """
    p = np.asarray(payoff, dtype=np.float64)
    if p.ndim == 2:
        x, y, v = solve(
            p[None],
            iters,
            None if row_mask is None else np.asarray(row_mask)[None],
            None if col_mask is None else np.asarray(col_mask)[None],
        )
        return x[0], y[0], v[0]
    batch, rows, cols = p.shape
    rm = np.ones((batch, rows), bool) if row_mask is None else row_mask.astype(bool)
    cm = np.ones((batch, cols), bool) if col_mask is None else col_mask.astype(bool)
    regret_x = np.zeros((batch, rows))
    regret_y = np.zeros((batch, cols))
    avg_x = np.zeros((batch, rows))
    avg_y = np.zeros((batch, cols))
    for t in range(1, iters + 1):
        x = _strategy(regret_x, rm)
        y = _strategy(regret_y, cm)
        ux = np.einsum("brc,bc->br", p, y)  # our payoff of each row
        uy = -np.einsum("brc,br->bc", p, x)  # their payoff of each column
        vx = (x * ux).sum(-1, keepdims=True)
        vy = (y * uy).sum(-1, keepdims=True)
        regret_x = np.maximum(regret_x + (ux - vx), 0.0) * rm
        regret_y = np.maximum(regret_y + (uy - vy), 0.0) * cm
        avg_x += t * x
        avg_y += t * y
    x = avg_x / avg_x.sum(-1, keepdims=True)
    y = avg_y / avg_y.sum(-1, keepdims=True)
    value = np.einsum("br,brc,bc->b", x, p, y)
    return x, y, value


def exploitability(payoff: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """How much each player could gain by deviating (0 at an exact equilibrium)."""
    p = np.asarray(payoff, dtype=np.float64)
    if p.ndim == 2:
        return exploitability(p[None], np.asarray(x)[None], np.asarray(y)[None])
    best_row = np.einsum("brc,bc->br", p, y).max(-1)
    best_col = (-np.einsum("brc,br->bc", p, x)).max(-1)
    v = np.einsum("br,brc,bc->b", x, p, y)
    return (best_row - v) + (best_col + v)
