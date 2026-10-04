"""Team evolution: let the bot change its team when it finds something better.

A (mu + lambda) loop on top of the trained policy:
  1. start from the best teams by training rating (or given names);
  2. each generation, mutate every survivor (legal-by-construction
     recombinations of tournament sets, see teambuilder.py);
  3. score each team by self-play with the current policy piloting both sides,
     against a fixed field of opponent teams (the "meta"), with the team on
     alternating sides;
  4. keep the best by Wilson lower bound; parents keep accumulating games,
     so a lucky mutant has to keep winning to stay.
Survivors are written with Showdown export and packed text, ready for
`scripts/play.py --team-file`.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .model import VGCNet, to_torch
from .teambuilder import TeamBuilder, same_team, team_json
from .teams import wilson


@dataclass
class Entry:
    team: dict
    index: int  # engine registry index
    games: float = 0.0
    wins: float = 0.0
    born: int = 0

    def lb(self) -> float:
        return wilson(self.wins, self.games, z=1.28)[0] if self.games else 0.0

    def rate(self) -> float:
        return self.wins / self.games if self.games else 0.0


@torch.no_grad()
def score_team(model: VGCNet, team_index: int, field: list[int], games: int, seed: int, device="cpu", n_envs: int = 128) -> tuple[float, int]:
    """Win rate of `team_index` (policy on both sides) against teams drawn
    uniformly from `field`. Returns (wins, games)."""
    n_envs = min(n_envs, games)
    teams = [team_index] + [f for f in field if f != team_index]
    w_ours = np.zeros(len(teams))
    w_ours[0] = 1.0
    w_field = np.ones(len(teams))
    w_field[0] = 0.0
    # Half the games with our team on each side.
    wins, played = 0.0, 0
    for our_side in (0, 1):
        env = D.E.VecEnv(n_envs, seed=seed + our_side, team_indices=teams)
        env.set_team_weights(our_side, w_ours)
        env.set_team_weights(1 - our_side, w_field)
        target = games // 2
        got = 0
        # Battles created before the weights were set use uniform teams: skip
        # each env's first episode.
        first = np.ones(n_envs, bool)
        while got < target:
            obs = env.observe()
            o = to_torch(*obs, device=device)
            a, _, _ = model.act(o)
            _, dones, _ = env.step(a.reshape(n_envs, 2, 3).cpu().numpy().astype(np.uint8))
            if dones.any():
                _, winner, _, team = env.last_infos()
                for e in np.nonzero(dones)[0]:
                    if first[e]:
                        first[e] = False
                        continue
                    if got >= target:
                        break
                    w = int(winner[e])
                    wins += 1.0 if w == our_side else (0.5 if w == 2 else 0.0)
                    got += 1
        played += got
    return wins, played


class Evolver:
    def __init__(self, model: VGCNet, field: list[int], out_dir: str, population: int = 8, children: int = 2,
                 games: int = 400, device="cpu", seed: int = 0):
        D.load()
        self.model = model
        self.field = field
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.mu = population
        self.lam = children
        self.games = games
        self.device = device
        self.rng = random.Random(seed)
        self.builder = TeamBuilder(seed=seed)
        self.pop: list[Entry] = []
        self.gen = 0

    def seed_population(self, team_indices: list[int]) -> None:
        for i in team_indices:
            t = self.builder.pool.team(i)
            self.pop.append(Entry({"name": t["name"], "mons": t["mons"]}, i))

    def evaluate(self, e: Entry) -> None:
        w, n = score_team(self.model, e.index, self.field, self.games, self.rng.randrange(1 << 30), self.device)
        e.wins += w
        e.games += n

    def step(self) -> None:
        self.gen += 1
        t0 = time.time()
        kids: list[Entry] = []
        for parent in list(self.pop):
            for _ in range(self.lam):
                child = self.builder.mutate(parent.team, n_ops=self.rng.choice((1, 1, 2)))
                if child is None or any(same_team(child, x.team) for x in self.pop + kids):
                    continue
                try:
                    idx = D.E.register_team(team_json(child))
                except ValueError:
                    continue  # uses something the engine does not support yet
                kids.append(Entry(child, idx, born=self.gen))
        for e in self.pop + kids:
            self.evaluate(e)
        everyone = self.pop + kids
        everyone.sort(key=lambda e: e.lb(), reverse=True)
        self.pop = everyone[: self.mu]
        self.save()
        best = self.pop[0]
        print(f"gen {self.gen}: {len(kids)} children, best {best.team['name']} {best.rate():.3f} "
              f"(lb {best.lb():.3f}, {int(best.games)} games), {time.time() - t0:.0f}s")

    def save(self) -> None:
        rows = []
        for e in self.pop:
            t = self.builder.with_text(e.team)
            rows.append({"team": t, "games": e.games, "wins": e.wins, "win_rate": e.rate(), "lb": e.lb(), "born": e.born})
        (self.out / f"population_gen{self.gen:03d}.json").write_text(json.dumps(rows, indent=1))
        D.write_text_atomic(self.out / "population_latest.json", json.dumps(rows, indent=1))
