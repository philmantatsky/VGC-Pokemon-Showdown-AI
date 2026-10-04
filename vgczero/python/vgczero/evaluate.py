"""Head-to-head evaluation between policies (checkpoints or baselines).

A player is a VGCNet, the string "random" or "greedy" (engine baselines), or a
callable(obs dict, env, side) -> actions [n, 3]. Sides alternate across envs so
neither player always has side 0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from . import data as D
from .model import VGCNet, to_torch
from .teams import wilson


@dataclass
class MatchResult:
    games: int
    a_wins: float
    draws: int
    turns: float

    @property
    def win_rate(self) -> float:
        return self.a_wins / max(1, self.games)

    def __str__(self) -> str:
        lo, hi = wilson(self.a_wins, self.games)
        return f"{self.win_rate:.3f} [{lo:.3f}, {hi:.3f}] over {self.games} games ({self.draws} draws, {self.turns:.1f} turns)"


@torch.no_grad()
def _actions(player, obs_np, env, side: int, rows: np.ndarray, device, deterministic: bool, seed: int) -> np.ndarray:
    n = env.n_envs
    out = np.zeros((n, 3), np.int64)
    if len(rows) == 0:
        return out
    if isinstance(player, str):
        a = np.asarray(env.baseline_actions(side, player, seed))
        out[rows] = a[rows]
        return out
    if isinstance(player, VGCNet):
        ints, floats, field, mask, req = (x[:, side] for x in obs_np)
        o = to_torch(ints[rows], floats[rows], field[rows], mask[rows], req[rows], device)
        a, _, _ = player.act(o, deterministic=deterministic)
        out[rows] = a.cpu().numpy()
        return out
    out[rows] = player(obs_np, env, side)[rows]
    return out


def play_match(a, b, n_games: int = 1000, n_envs: int = 256, team_indices: list[int] | None = None, seed: int = 0,
               open_sheet_prob: float = 0.5, device="cpu", deterministic: bool = False, team_weights=None) -> MatchResult:
    D.load()
    n_envs = min(n_envs, n_games)
    env = D.E.VecEnv(n_envs, seed=seed, team_indices=team_indices or D.supported_teams(), open_sheet_prob=open_sheet_prob)
    if team_weights is not None:
        for s in range(2):
            env.set_team_weights(s, np.asarray(team_weights, dtype=np.float64))
    a_side = (np.arange(n_envs) % 2).astype(np.int64)  # side A plays in each env
    games, a_wins, draws, turns = 0, 0.0, 0, 0
    step = 0
    while games < n_games:
        obs = env.observe()
        acts = np.zeros((n_envs, 2, 3), np.int64)
        for s in range(2):
            rows_a = np.nonzero(a_side == s)[0]
            rows_b = np.nonzero(a_side != s)[0]
            acts[:, s] += _actions(a, obs, env, s, rows_a, device, deterministic, seed * 7919 + step)
            acts[:, s] += _actions(b, obs, env, s, rows_b, device, deterministic, seed * 104729 + step)
        _, dones, _ = env.step(acts.astype(np.uint8))
        step += 1
        if dones.any():
            _, winner, tr, _ = env.last_infos()
            for e in np.nonzero(dones)[0]:
                if games >= n_games:
                    break
                w = int(winner[e])
                games += 1
                turns += int(tr[e])
                if w == 2:
                    draws += 1
                    a_wins += 0.5
                elif w == a_side[e]:
                    a_wins += 1
    return MatchResult(games, a_wins, draws, turns / max(1, games))


def main() -> None:
    import argparse

    from .model import load_model

    ap = argparse.ArgumentParser(description="Evaluate two players head to head")
    ap.add_argument("a", help="checkpoint path, 'random' or 'greedy'")
    ap.add_argument("b", help="checkpoint path, 'random' or 'greedy'")
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()
    D.load()
    load = lambda x: x if x in ("random", "greedy") else load_model(x, device=args.device)
    t = time.time()
    r = play_match(load(args.a), load(args.b), args.games, args.envs, seed=args.seed, device=args.device, deterministic=args.deterministic)
    print(f"{args.a} vs {args.b}: {r}  ({time.time() - t:.1f}s)")


if __name__ == "__main__":
    main()
