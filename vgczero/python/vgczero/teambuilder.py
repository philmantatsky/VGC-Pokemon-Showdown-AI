"""Team mutations that stay legal by construction, plus Showdown text formats.

A team is a compiled-team dict: {"name": str, "mons": [6 compiled sets]}
(the same format as data/teams_regmc.json.gz). Mutations only recombine
pieces of real tournament sets, so no re-validation or stat recomputation is
needed:

  member   replace one Pokemon with another pool set of a species not on the team
  moves    swap one move for a move from another pool set of the same species
  item     swap a non-Mega item for one a pool set of that species used
  spread   take EVs/nature/stats (and Mega stats) from another pool set of the
           same species with the same item

Species clause and item clause (no duplicate items) are enforced.
"""

from __future__ import annotations

import copy
import json
import random
import re

from .live.sets import Pool, pool, to_id

STAT_LABELS = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]


def _pack_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", s)


class TeamBuilder:
    def __init__(self, pool_: Pool | None = None, seed: int = 0):
        self.pool = pool_ or pool()
        self.rng = random.Random(seed)

    # ---- validity -----------------------------------------------------------------------

    def legal(self, team: dict) -> bool:
        mons = team["mons"]
        bases = [self.pool.base_id(m["species"]) for m in mons]
        items = [m["item"] for m in mons if m["item"]]
        return len(set(bases)) == 6 and len(set(items)) == len(items)

    # ---- mutations ----------------------------------------------------------------------

    def mutate(self, team: dict, n_ops: int = 1, ops: tuple[str, ...] = ("member", "moves", "item", "spread")) -> dict | None:
        child = copy.deepcopy(team)
        done = []
        for _ in range(n_ops):
            for _attempt in range(20):
                op = self.rng.choice(ops)
                trial = getattr(self, "_" + op)(child)
                if trial is not None and self.legal(trial):
                    child = trial
                    done.append(op)
                    break
        if not done:
            return None
        child["name"] = f"{team.get('name', 'team')}~{'+'.join(done)}{self.rng.randrange(10**6):06d}"
        child["parent"] = team.get("name")
        return child

    def _member(self, team: dict) -> dict | None:
        i = self.rng.randrange(6)
        have = {self.pool.base_id(m["species"]) for k, m in enumerate(team["mons"]) if k != i}
        items = {m["item"] for k, m in enumerate(team["mons"]) if k != i and m["item"]}
        species = [s for s in self.pool.sets if s not in have]
        for _ in range(20):
            sp = self.rng.choice(species)
            cands = [s for s in self.pool.sets[sp] if not s["item"] or s["item"] not in items]
            if cands:
                t = copy.deepcopy(team)
                t["mons"][i] = copy.deepcopy(self.rng.choice(cands))
                return t
        return None

    def _moves(self, team: dict) -> dict | None:
        i = self.rng.randrange(6)
        m = team["mons"][i]
        base = self.pool.base_id(m["species"])
        pool_moves = sorted({mv for s in self.pool.sets.get(base, []) for mv in s["moves"]} - set(m["moves"]))
        if not pool_moves or not m["moves"]:
            return None
        t = copy.deepcopy(team)
        j = self.rng.randrange(len(m["moves"]))
        t["mons"][i]["moves"][j] = self.rng.choice(pool_moves)
        return t

    def _item(self, team: dict) -> dict | None:
        i = self.rng.randrange(6)
        m = team["mons"][i]
        if m.get("mega"):
            return None  # keep Mega stones (their stats come with the set)
        base = self.pool.base_id(m["species"])
        used = {x["item"] for k, x in enumerate(team["mons"]) if k != i and x["item"]}
        cands = sorted({s["item"] for s in self.pool.sets.get(base, []) if not s.get("mega")} - used - {m["item"]})
        if not cands:
            return None
        t = copy.deepcopy(team)
        t["mons"][i]["item"] = self.rng.choice(cands)
        return t

    def _spread(self, team: dict) -> dict | None:
        i = self.rng.randrange(6)
        m = team["mons"][i]
        base = self.pool.base_id(m["species"])
        donors = [s for s in self.pool.sets.get(base, []) if s["species"] == m["species"] and s["item"] == m["item"]
                  and (s["evs"] != m["evs"] or s["nature"] != m["nature"])]
        if not donors:
            return None
        d = self.rng.choice(donors)
        t = copy.deepcopy(team)
        for k in ("evs", "nature", "stats", "mega"):
            t["mons"][i][k] = copy.deepcopy(d[k])
        return t

    # ---- text formats ---------------------------------------------------------------------

    def export(self, team: dict) -> str:
        """Showdown export (paste) format."""
        out = []
        for m in team["mons"]:
            sp = self.pool.species.get(m["species"], {}).get("name", m["species"])
            item = self.pool.items.get(m["item"], {}).get("name", "") if m["item"] else ""
            ab = self.pool.abilities.get(m["ability"], {}).get("name", m["ability"])
            gender = f" ({m['gender']})" if m.get("gender") in ("M", "F") else ""
            lines = [f"{sp}{gender}" + (f" @ {item}" if item else ""), f"Ability: {ab}", f"Level: {m.get('level', 50)}"]
            evs = " / ".join(f"{v} {STAT_LABELS[k]}" for k, v in enumerate(m["evs"]) if v)
            if evs:
                lines.append(f"EVs: {evs}")
            lines.append(f"{m['nature'].capitalize()} Nature")
            for mv in m["moves"]:
                lines.append("- " + self.pool.moves.get(mv, {}).get("name", mv))
            out.append("\n".join(lines))
        return "\n\n".join(out) + "\n"

    def packed(self, team: dict) -> str:
        """Showdown packed format (for /utm)."""
        rows = []
        for m in team["mons"]:
            spd = self.pool.species.get(m["species"], {})
            sp = spd.get("name", m["species"])
            base = spd.get("baseSpecies", sp)
            # Showdown: nickname defaults to the base species; the species field
            # is blank unless it differs (alternate formes).
            species_field = "" if sp == base else _pack_name(sp)
            item = self.pool.items.get(m["item"], {}).get("name", "") if m["item"] else ""
            ab = self.pool.abilities.get(m["ability"], {}).get("name", m["ability"])
            moves = ",".join(_pack_name(self.pool.moves.get(mv, {}).get("name", mv)) for mv in m["moves"])
            evs = ",".join(str(v) if v else "" for v in m["evs"])
            rows.append("|".join([base, species_field, _pack_name(item), _pack_name(ab), moves, m["nature"].capitalize(), evs,
                                  m.get("gender", "") or "", "", "", str(m.get("level", 50)), ""]))
        return "]".join(rows)

    def with_text(self, team: dict) -> dict:
        t = dict(team)
        t["export"] = self.export(team)
        t["packed"] = self.packed(team)
        return t


def team_json(team: dict) -> str:
    return json.dumps({"name": team.get("name", ""), "mons": team["mons"]})


def same_team(a: dict, b: dict) -> bool:
    key = lambda t: sorted(json.dumps(m, sort_keys=True) for m in t["mons"])
    return key(a) == key(b)


def species_line(team: dict) -> str:
    return ", ".join(to_id(m["species"]) for m in team["mons"])
