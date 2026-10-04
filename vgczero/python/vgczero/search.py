"""One-turn simultaneous-move search (the mikumiku37 recipe).

For a decision of player `viewer`:
  1. sample W worlds: the opponent's hidden information (sets, unseen back
     row) drawn consistently with what the viewer has seen, each world with
     its own RNG stream (damage rolls, crits, accuracy...);
  2. take the viewer's top-K joint actions from the policy (the same in every
     world, since the viewer's information is identical) and, per world, the
     opponent's top-K joint replies from the policy run on the opponent's view
     of that world;
  3. play every (viewer action, opponent action) pair one turn ahead in the
     engine and score the resulting position with the value network (or the
     real result if the game ended) -- no rollouts;
  4. solve each world's K x K zero-sum payoff table for a mixed strategy and
     average the viewer's strategies over worlds.
The action is sampled from (or the argmax of) that averaged strategy.

Team preview is searched the same way: the top bring/lead options of both
players form the payoff table, scored after the leads are on the field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import data as D
from .matrix_game import solve
from .model import VGCNet, to_torch


@dataclass
class SearchConfig:
    worlds: int = 16
    k_self: int = 8
    k_opp: int = 8
    iters: int = 1000
    sample: bool = True
    # Mix the solved strategy with the policy prior (0 = pure search).
    prior_mix: float = 0.0
    search_preview: bool = True


@dataclass
class SearchResult:
    choice: tuple[int, int, int]
    strategy: np.ndarray
    candidates: np.ndarray
    prior: np.ndarray
    value: float
    world_values: np.ndarray


def _obs(batch_or_battle, side, device):
    o = batch_or_battle.observe(side)
    return to_torch(*o, device=device)


@torch.no_grad()
def search(model: VGCNet, battle, viewer: int, cfg: SearchConfig | None = None, seed: int = 0, device="cpu") -> SearchResult:
    cfg = cfg or SearchConfig()
    kind, *_ = battle.request(viewer)
    root_obs = _obs(battle, viewer, device)
    my_choices, my_probs = model.top_joint(root_obs, cfg.k_self)
    my_choices = my_choices[0].cpu().numpy()
    my_probs = my_probs[0].cpu().numpy()
    # Drop padded duplicates (probability ~0).
    keep = my_probs > 1e-6
    keep[0] = True
    my_choices, my_probs = my_choices[keep], my_probs[keep]
    K = len(my_choices)
    if K == 1 or kind == D.REQ_WAIT or (kind == D.REQ_PREVIEW and not cfg.search_preview):
        c = tuple(int(x) for x in my_choices[0])
        return SearchResult(c, np.ones(1), my_choices, my_probs, 0.0, np.zeros(0))

    worlds = D.E.BattleBatch.worlds(battle, viewer, cfg.worlds, seed)
    W = len(worlds)
    opp = 1 - viewer
    opp_obs = _obs(worlds, opp, device)
    opp_kind = opp_obs["req"][:, 0]
    oc, op = model.top_joint(opp_obs, cfg.k_opp)
    oc = oc.cpu().numpy()
    op = op.cpu().numpy()
    J = oc.shape[1]
    col_mask = op > 1e-6
    col_mask[:, 0] = True
    # Opponent with nothing to choose: a single "pass" column.
    waiting = (opp_kind == D.REQ_WAIT).cpu().numpy()
    if waiting.any():
        oc[waiting] = 0
        col_mask[waiting] = False
        col_mask[waiting, 0] = True

    children = worlds.expand(viewer, my_choices.astype(np.uint8), oc.astype(np.uint8))
    out = np.asarray(children.outcomes(viewer))  # +1/-1/0 or NaN
    need = np.isnan(out)
    vals = out.copy()
    if need.any():
        cobs = _obs(children, viewer, device)
        H, _ = model.encode(cobs["ints"], cobs["floats"], cobs["field"])
        v = model.value(H).float().cpu().numpy()
        vals = np.where(need, v, out)
    valid = np.asarray(children.valid_mask(), dtype=bool)
    vals = np.where(valid, vals, np.nan)
    P = vals.reshape(W, K, J)
    # Invalid cells (should not happen): pessimistic for the viewer's row.
    P = np.where(np.isnan(P), -1.0, P)
    row_mask = np.ones((W, K), bool)
    x, y, value = solve(P, cfg.iters, row_mask, col_mask)
    strat = x.mean(0)
    if cfg.prior_mix > 0:
        prior = my_probs / my_probs.sum()
        strat = (1 - cfg.prior_mix) * strat + cfg.prior_mix * prior
    strat = strat / strat.sum()
    if cfg.sample:
        rng = np.random.default_rng(seed)
        i = int(rng.choice(K, p=strat))
    else:
        i = int(np.argmax(strat))
    c = tuple(int(v) for v in my_choices[i])
    return SearchResult(c, strat, my_choices, my_probs, float(value.mean()), value)


class SearchPlayer:
    """Player callable for evaluate.play_match: runs search for every env
    where it has a decision (one battle at a time)."""

    def __init__(self, model: VGCNet, cfg: SearchConfig | None = None, device="cpu", seed: int = 0):
        self.model = model
        self.cfg = cfg or SearchConfig()
        self.device = device
        self.seed = seed
        self.calls = 0

    def __call__(self, obs_np, env, side: int, rows) -> np.ndarray:
        n = env.n_envs
        out = np.zeros((n, 3), np.int64)
        req = obs_np[4][:, side, 0]
        for e in rows:
            if req[e] == D.REQ_WAIT:
                continue
            b = env.battle(e)
            self.calls += 1
            r = search(self.model, b, side, self.cfg, seed=self.seed * 1_000_003 + self.calls, device=self.device)
            out[e] = r.choice
        return out
