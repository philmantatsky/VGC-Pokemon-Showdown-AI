"""Deeper search: open-loop, decoupled pUCT for simultaneous moves
(in the spirit of Jaxcalibur's pUCT over sampled worlds).

* Open loop: tree nodes are indexed by the sequence of joint actions, not by
  game state. Every simulation starts from a freshly sampled world (hidden
  sets and RNG), so chance outcomes and hidden information are averaged
  across visits instead of being branched on.
* Decoupled: at each node both players select their own joint action
  independently with pUCT on their own statistics (Q from their side,
  priors from the policy's top-k joint actions), which is the standard way
  to run UCT on simultaneous-move games.
* Leaves (new nodes, depth limit, or game end) are scored by the value
  network, batched across simulations with virtual loss.
The root decision is the viewer's visit distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from . import data as D
from .model import VGCNet, to_torch


@dataclass
class TreeConfig:
    simulations: int = 2048
    worlds: int = 32
    depth: int = 4  # decision points below the root (~2 turns)
    k: int = 8
    c_puct: float = 1.5
    batch: int = 64
    virtual_loss: float = 1.0
    sample: bool = True
    temperature: float = 1.0


@dataclass
class Node:
    actions: list  # per player: array [k, 3] of choices
    priors: list  # per player: array [k]
    n: int = 0
    N: list = field(default_factory=lambda: [None, None])
    W: list = field(default_factory=lambda: [None, None])
    children: dict = field(default_factory=dict)
    value: float = 0.0

    def init_stats(self) -> None:
        for p in range(2):
            k = len(self.actions[p])
            self.N[p] = np.zeros(k)
            self.W[p] = np.zeros(k)

    def select(self, p: int, c: float) -> int:
        n, w, pri = self.N[p], self.W[p], self.priors[p]
        if len(n) == 1:
            return 0
        q = np.where(n > 0, w / np.maximum(n, 1), 0.0)
        u = c * pri * math.sqrt(max(1, self.n)) / (1 + n)
        return int(np.argmax(q + u))


def _legal_default(b, side: int) -> tuple[int, int, int]:
    kind, s0, s1 = b.request(side)
    if kind == D.REQ_WAIT:
        return (0, 0, 0)
    if kind == D.REQ_PREVIEW:
        return (0, 0, 0)
    masks = b.legal_masks(side)
    out = [0, 0, 0]
    used = set()
    for k in range(2):
        for a in range(31):
            if masks[k][a] and not (1 <= a <= 6 and a in used) and not (a >= 7 and (a - 7) % 2 == 1):
                out[1 + k] = a
                if 1 <= a <= 6:
                    used.add(a)
                break
    return tuple(out)


class TreeSearch:
    def __init__(self, model: VGCNet, cfg: TreeConfig | None = None, device="cpu"):
        self.model = model
        self.cfg = cfg or TreeConfig()
        self.device = device

    @torch.no_grad()
    def _expand(self, battles: list, viewer: int) -> tuple[list[Node], np.ndarray]:
        """Priors for both players and the viewer's value for each battle."""
        n = len(battles)
        obs_v = [b.observe(viewer) for b in battles]
        obs_o = [b.observe(1 - viewer) for b in battles]
        cat = lambda obs: [np.concatenate([np.asarray(o[i]) for o in obs]) for i in range(5)]
        ov, oo = cat(obs_v), cat(obs_o)
        tv = to_torch(*ov, device=self.device)
        to = to_torch(*oo, device=self.device)
        H, _ = self.model.encode(tv["ints"], tv["floats"], tv["field"])
        values = self.model.value(H).float().cpu().numpy()
        cv, pv = self.model.top_joint(tv, self.cfg.k)
        co, po = self.model.top_joint(to, self.cfg.k)
        nodes = []
        for i in range(n):
            acts, pris = [], []
            for c, p, req in ((cv[i], pv[i], tv["req"][i, 0]), (co[i], po[i], to["req"][i, 0])):
                c = c.cpu().numpy()
                p = p.cpu().numpy()
                if int(req) == D.REQ_WAIT:
                    c, p = np.zeros((1, 3), np.int64), np.ones(1)
                keep = p > 1e-6
                keep[0] = True
                c, p = c[keep], p[keep]
                acts.append(c)
                pris.append(p / p.sum())
            # Player 0 = viewer, player 1 = opponent.
            nd = Node([acts[0], acts[1]], [pris[0], pris[1]])
            nd.init_stats()
            nd.value = float(values[i])
            nodes.append(nd)
        return nodes, values

    @staticmethod
    def _outcome(b, viewer: int) -> float | None:
        if not b.ended:
            return None
        w = b.winner
        return 0.0 if w == 2 or w is None else (1.0 if w == viewer else -1.0)

    def run(self, root_battle, viewer: int, seed: int = 0):
        cfg = self.cfg
        rng = np.random.default_rng(seed)
        worlds = D.E.BattleBatch.worlds(root_battle, viewer, cfg.worlds, seed)
        roots, _ = self._expand([worlds.get(0)], viewer)
        root = roots[0]
        sims = 0
        while sims < cfg.simulations:
            paths, leaves, terminal = [], [], []
            for _ in range(min(cfg.batch, cfg.simulations - sims)):
                b = worlds.get(int(rng.integers(cfg.worlds)))
                b.reseed(int(rng.integers(1 << 62)))
                node, path, depth = root, [], 0
                outcome = None
                while True:
                    iv = node.select(0, cfg.c_puct)
                    io = node.select(1, cfg.c_puct)
                    # Virtual loss so parallel simulations spread out.
                    node.N[0][iv] += cfg.virtual_loss
                    node.W[0][iv] -= cfg.virtual_loss
                    node.N[1][io] += cfg.virtual_loss
                    node.W[1][io] -= cfg.virtual_loss
                    node.n += 1
                    path.append((node, iv, io))
                    cv = tuple(int(x) for x in node.actions[0][iv])
                    co = tuple(int(x) for x in node.actions[1][io])
                    try:
                        b.step(cv, co) if viewer == 0 else b.step(co, cv)
                    except ValueError:
                        # Stored action illegal in this world: fall back to a legal default.
                        dv = cv if self._legal(b, viewer, cv) else _legal_default(b, viewer)
                        do = co if self._legal(b, 1 - viewer, co) else _legal_default(b, 1 - viewer)
                        try:
                            b.step(dv, do) if viewer == 0 else b.step(do, dv)
                        except ValueError:
                            outcome = node.value
                            break
                    depth += 1
                    outcome = self._outcome(b, viewer)
                    if outcome is not None:
                        break
                    key = (iv, io)
                    child = node.children.get(key)
                    if child is None or depth >= cfg.depth:
                        break
                    node = child
                paths.append((path, key if outcome is None else None))
                leaves.append(b if outcome is None else None)
                terminal.append(outcome)
            # Evaluate non-terminal leaves in one batch (and expand them).
            idx = [i for i, x in enumerate(leaves) if x is not None]
            values = {}
            if idx:
                nodes, vals = self._expand([leaves[i] for i in idx], viewer)
                for j, i in enumerate(idx):
                    values[i] = float(vals[j])
                    path, key = paths[i]
                    parent = path[-1][0]
                    if key is not None and key not in parent.children and len(path) < cfg.depth:
                        parent.children[key] = nodes[j]
            for i, (path, _) in enumerate(paths):
                v = terminal[i] if terminal[i] is not None else values[i]
                for node, iv, io in path:
                    node.N[0][iv] += 1 - cfg.virtual_loss
                    node.W[0][iv] += v + cfg.virtual_loss
                    node.N[1][io] += 1 - cfg.virtual_loss
                    node.W[1][io] += -v + cfg.virtual_loss
            sims += len(paths)
        counts = root.N[0].copy()
        if cfg.sample and cfg.temperature > 0:
            p = counts ** (1.0 / cfg.temperature)
            p = p / p.sum()
            i = int(rng.choice(len(p), p=p))
        else:
            i = int(np.argmax(counts))
            p = counts / counts.sum()
        q = root.W[0] / np.maximum(root.N[0], 1)
        return tuple(int(x) for x in root.actions[0][i]), p, root.actions[0], q

    @staticmethod
    def _legal(b, side: int, c: tuple[int, int, int]) -> bool:
        kind, _, _ = b.request(side)
        if kind in (D.REQ_WAIT, D.REQ_PREVIEW):
            return True
        m = b.legal_masks(side)
        return bool(m[0][c[1]] and m[1][c[2]])
