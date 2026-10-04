"""Static dex/team data on the Python side (names, base species, set pool)."""

from __future__ import annotations

import copy
import functools
import gzip
import json
import re
from collections import Counter, defaultdict

from .. import data as D


def to_id(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


class Pool:
    def __init__(self) -> None:
        self.dex = json.load(gzip.open(D.DEX))
        self.teams = json.load(gzip.open(D.TEAMS))["teams"]
        self.species = {s["id"]: s for s in self.dex["species"]}
        self.moves = {m["id"]: m for m in self.dex["moves"]}
        self.items = {i["id"]: i for i in self.dex["items"]}
        self.abilities = {a["id"]: a for a in self.dex["abilities"]}
        # Sets by base species, with frequencies.
        self.sets: dict[str, list[dict]] = defaultdict(list)
        self.set_counts: dict[str, Counter] = defaultdict(Counter)
        for t in self.teams:
            for m in t["mons"]:
                base = self.base_id(m["species"])
                key = json.dumps(m, sort_keys=True)
                if self.set_counts[base][key] == 0:
                    self.sets[base].append(m)
                self.set_counts[base][key] += 1

    def base_id(self, species: str) -> str:
        sp = self.species.get(to_id(species))
        if sp is None:
            return to_id(species)
        return to_id(sp["baseSpecies"])

    def species_id(self, name: str) -> str | None:
        sid = to_id(name)
        if sid in self.species:
            return sid
        return None

    def placeholder(self, species: str, mega_species: str | None = None) -> dict | None:
        """The most common tournament set for a species (optionally one that
        Mega Evolves into `mega_species`)."""
        base = self.base_id(species)
        cands = self.sets.get(base, [])
        if mega_species:
            cands = [s for s in cands if s.get("mega") and s["mega"]["species"] == to_id(mega_species)] or cands
        exact = [s for s in cands if s["species"] == to_id(species)] or cands
        if not exact:
            return None
        counts = self.set_counts[base]
        best = max(exact, key=lambda s: counts[json.dumps(s, sort_keys=True)])
        return copy.deepcopy(best)

    def team(self, i: int) -> dict:
        return self.teams[i]

    def team_index(self, name: str) -> int:
        for i, t in enumerate(self.teams):
            if t["name"] == name:
                return i
        raise KeyError(name)

    def move_target(self, move_id: str) -> str:
        return self.moves.get(move_id, {}).get("target", "normal")


@functools.lru_cache(maxsize=1)
def pool() -> Pool:
    return Pool()
