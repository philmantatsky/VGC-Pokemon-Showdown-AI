"""League of past policy snapshots with prioritized fictitious self-play.

The learner plays the current policy (pure self-play) part of the time and a
past snapshot otherwise. Snapshots are sampled with PFSP weights
(1 - learner win rate)^p, so opponents the learner still struggles with come
up more often, while beaten ones fade out.
"""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from . import data as D
from .model import VGCNet, load_model, save

SELF = -1  # opponent id for "the current policy"


@dataclass
class Member:
    id: int
    path: str
    update: int
    games: float = 0.0
    learner_wins: float = 0.0
    # Exponential moving win rate of the learner against this member.
    ema: float = 0.5
    tags: list[str] = field(default_factory=list)

    def win_rate(self) -> float:
        return self.ema


class League:
    def __init__(self, root: Path, max_size: int = 64, pfsp_power: float = 2.0, ema: float = 0.02, cache: int = 8):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.pfsp_power = pfsp_power
        self.ema_rate = ema
        self.members: dict[int, Member] = {}
        self.next_id = 0
        self._cache: OrderedDict[int, VGCNet] = OrderedDict()
        self._cache_size = cache
        self._load_index()

    # ---- persistence ---------------------------------------------------------------

    def _index_path(self) -> Path:
        return self.root / "league.json"

    def _load_index(self) -> None:
        p = self._index_path()
        if p.exists():
            d = json.loads(p.read_text())
            self.next_id = d["next_id"]
            self.members = {m["id"]: Member(**m) for m in d["members"]}

    def save_index(self) -> None:
        d = {"next_id": self.next_id, "members": [asdict(m) for m in self.members.values()]}
        D.write_text_atomic(self._index_path(), json.dumps(d, indent=1))

    # ---- membership ------------------------------------------------------------------

    def snapshot(self, model: VGCNet, update: int, tags: list[str] | None = None) -> int:
        mid = self.next_id
        self.next_id += 1
        path = self.root / f"snap_{mid:05d}_u{update}.pt"
        save(model, str(path), {"update": update})
        self.members[mid] = Member(id=mid, path=str(path), update=update, tags=tags or [])
        self._evict()
        self.save_index()
        return mid

    def _evict(self) -> None:
        if len(self.members) <= self.max_size:
            return
        # Keep the first snapshot and anything tagged "anchor"; drop the
        # member the learner beats most reliably, ties to the oldest.
        ids = sorted(self.members)
        protected = {ids[0]} | {i for i, m in self.members.items() if "anchor" in m.tags}
        cands = [i for i in ids if i not in protected]
        if not cands:
            return
        worst = max(cands, key=lambda i: (self.members[i].ema, -i))
        m = self.members.pop(worst)
        self._cache.pop(worst, None)
        try:
            Path(m.path).unlink()
        except OSError:
            pass

    # ---- sampling ---------------------------------------------------------------------

    def pfsp_weights(self) -> dict[int, float]:
        return {i: max(1e-3, (1.0 - m.ema)) ** self.pfsp_power for i, m in self.members.items()}

    def sample(self, rng: random.Random, k: int) -> list[int]:
        if not self.members:
            return []
        w = self.pfsp_weights()
        ids = list(w)
        return rng.choices(ids, weights=[w[i] for i in ids], k=k)

    def record(self, mid: int, learner_score: float, n: float = 1.0) -> None:
        m = self.members.get(mid)
        if m is None:
            return
        m.games += n
        m.learner_wins += learner_score * n
        for _ in range(int(max(1, n))):
            m.ema += self.ema_rate * (learner_score - m.ema)

    def model(self, mid: int, device) -> VGCNet:
        if mid in self._cache:
            self._cache.move_to_end(mid)
            return self._cache[mid]
        m = load_model(self.members[mid].path, device=device)
        for p in m.parameters():
            p.requires_grad_(False)
        self._cache[mid] = m
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return m

    def summary(self, top: int = 8) -> str:
        ms = sorted(self.members.values(), key=lambda m: m.ema)[:top]
        return " ".join(f"{m.id}@u{m.update}:{m.ema:.2f}" for m in ms)


@torch.no_grad()
def clone_frozen(model: VGCNet) -> VGCNet:
    import copy

    m = copy.deepcopy(model)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m
