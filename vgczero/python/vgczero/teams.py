"""Team statistics and selection.

Every finished training game records a result for the team the learner
piloted (and, in self-play games, for both teams). From those results the
trainer keeps a running rating per team; the best teams can be favoured when
sampling the learner's team (`team_focus`) and are exported for laddering,
like mikumiku37 rotating among the ten teams that did best in training.
`evolve.py` builds new teams from the strongest ones.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import data as D


class TeamStats:
    def __init__(self, pool: list[int], ema: float = 0.01):
        self.pool = list(pool)
        self.pos = {t: i for i, t in enumerate(self.pool)}
        n = len(self.pool)
        self.games = np.zeros(n)
        self.wins = np.zeros(n)
        self.ema = np.full(n, 0.5)
        self.ema_rate = ema

    def add(self, team: int) -> None:
        """Add a new team (e.g. an evolved one) to the pool."""
        if int(team) in self.pos:
            return
        self.pos[int(team)] = len(self.pool)
        self.pool.append(int(team))
        self.games = np.append(self.games, 0.0)
        self.wins = np.append(self.wins, 0.0)
        self.ema = np.append(self.ema, 0.5)

    def record(self, team: int, score: float) -> None:
        i = self.pos.get(int(team))
        if i is None:
            return
        self.games[i] += 1
        self.wins[i] += score
        self.ema[i] += self.ema_rate * (score - self.ema[i])

    def posterior_mean(self) -> np.ndarray:
        return (self.wins + 1.0) / (self.games + 2.0)

    def lower_bound(self, z: float = 1.64) -> np.ndarray:
        """Wilson lower bound of each team's win rate."""
        n = np.maximum(self.games, 1e-9)
        p = self.wins / n
        den = 1 + z * z / n
        centre = p + z * z / (2 * n)
        rad = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        lb = (centre - rad) / den
        return np.where(self.games > 0, lb, 0.0)

    def sampling_weights(self, focus: float, floor: float = 0.2) -> np.ndarray:
        """Weights over the pool: uniform at focus=0; softmax on recent win rate
        otherwise, mixed with a uniform floor so every team keeps being tried."""
        n = len(self.pool)
        if focus <= 0:
            return np.ones(n)
        logits = focus * (self.ema - self.ema.mean()) * 10.0
        w = np.exp(logits - logits.max())
        w = w / w.sum()
        return (1 - floor) * w + floor / n

    def top(self, k: int = 10, min_games: int = 50) -> list[tuple[int, float, int]]:
        lb = self.lower_bound()
        ok = np.nonzero(self.games >= min_games)[0]
        order = ok[np.argsort(-lb[ok])][:k]
        return [(self.pool[i], float(lb[i]), int(self.games[i])) for i in order]

    def state(self) -> dict:
        return {"pool": self.pool, "games": self.games.tolist(), "wins": self.wins.tolist(), "ema": self.ema.tolist()}

    def load_state(self, d: dict) -> None:
        if d.get("pool") != self.pool:
            # Pool changed: carry over the overlap.
            old = {t: i for i, t in enumerate(d["pool"])}
            for t, i in self.pos.items():
                j = old.get(t)
                if j is not None:
                    self.games[i] = d["games"][j]
                    self.wins[i] = d["wins"][j]
                    self.ema[i] = d["ema"][j]
            return
        self.games = np.array(d["games"])
        self.wins = np.array(d["wins"])
        self.ema = np.array(d["ema"])

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.state()))

    def report(self, k: int = 10) -> str:
        names = D.team_names()
        rows = []
        for t, lb, g in self.top(k):
            sp = ", ".join(D.E.team_species(t))
            rows.append(f"  {names[t]:>8}  lb {lb:.3f}  games {g:6d}  {sp}")
        return "\n".join(rows)


def wilson(wins: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = wins / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - rad) / den, (centre + rad) / den
