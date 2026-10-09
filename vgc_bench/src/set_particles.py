"""Weighted hidden-set beliefs for Reg M-B exact planning.

The policy must not learn the opponent's unrevealed item, ability, or moves from the
simulator's private state.  This module turns the replay-derived joint-set table and
Smogon marginals into a small, explicit particle belief.  Beliefs are conditioned on
public evidence and full-team samples enforce the format's item and Mega constraints.

The objects here are simulator-independent on purpose.  They are used by offline
exact battles now and by the live snapshot synchronizer once its parity gate passes.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from poke_env.data import GenData, to_id_str

# Showdown UI placeholders that replay scrapers can record as "moves". A Ditto
# waiting to Transform shows "[nothing]", and the scraped joint-set table carried
# it into Ditto's dominant particle -- every determinized team containing that
# particle then crashed poke-env's battle view ("Unknown move: nothing"), which
# aborted counterfactual generation and poisoned live hidden-world roots.
_PLACEHOLDER_MOVES = frozenset({"nothing", "recharge", "struggle", "fight"})


def _id(value: object | None) -> str:
    return to_id_str(str(value)) if value else ""


def _normalise(weights: Sequence[float]) -> tuple[float, ...]:
    total = sum(max(0.0, float(weight)) for weight in weights)
    if total <= 0:
        if not weights:
            return ()
        return (1.0 / len(weights),) * len(weights)
    return tuple(max(0.0, float(weight)) / total for weight in weights)


def _is_mega_item(item: str | None) -> bool:
    item_id = _id(item)
    return bool(
        item_id in {"blueorb", "redorb"}
        or (item_id.endswith("ite") and item_id != "eviolite")
    )


@dataclass(frozen=True)
class SetParticle:
    species: str
    ability: str | None
    item: str | None
    moves: tuple[str, ...]
    spread: str | None
    probability: float
    source: str
    count: int = 0
    novel: bool = False

    @property
    def signature(self) -> tuple[str | None, str | None, tuple[str, ...]]:
        return self.ability, self.item, tuple(sorted(self.moves))

    @property
    def mega(self) -> bool:
        return _is_mega_item(self.item)


@dataclass(frozen=True)
class TeamSlot:
    species: str
    display: str


def team_roster(team_text: str) -> tuple[TeamSlot, ...]:
    """Read species identities from a Showdown export without retaining its sets."""
    roster: list[TeamSlot] = []
    for block in re.split(r"\n\s*\n", team_text.strip()):
        if not block.strip():
            continue
        heading = block.splitlines()[0].split("@", 1)[0].strip()
        heading = re.sub(r"\s+\([MFN]\)\s*$", "", heading)
        match = re.search(r"\(([^()]*)\)\s*$", heading)
        display = match.group(1).strip() if match else heading
        roster.append(TeamSlot(to_id_str(display), display))
    return tuple(roster)


def default_spread(species: str, moves: Sequence[str]) -> str:
    """A plausible Champions spread ("Nature:HP/Atk/Def/SpA/SpD/Spe") for a set whose
    spread is unknown.

    2026-10-04: the Reg M-C replay data carries no spreads, so every hidden-world
    opponent was simulated with a neutral nature and ZERO stat points -- frailer,
    weaker and differently ordered than any real set. The matrix search lost ~70% of
    its hidden-sheet games against our own team (open sheets: parity). Like
    vgc_knowledge.ensure_stats, the live damage features' own assumption: max HP,
    max in its attacking stat, 2 Speed, neutral nature. The attacking stat follows
    the set's damaging moves, then the higher base stat (ties: Attack).
    """
    data = GenData.from_gen(9)
    physical = special = 0
    for move in moves:
        category = str(data.moves.get(to_id_str(move), {}).get("category", ""))
        physical += category == "Physical"
        special += category == "Special"
    if physical == special:
        base = data.pokedex.get(to_id_str(species), {}).get("baseStats", {})
        special = int(base.get("spa", 0) > base.get("atk", 0))
        physical = 1 - special
    atk, spa = (32, 0) if physical > special else (0, 32)
    return f"Serious:32/{atk}/0/{spa}/0/2"


def determination_team_text(
    roster: Sequence[TeamSlot], determination: Mapping[str, SetParticle]
) -> str:
    """Render one sampled full team for Pokemon Showdown's exact simulator."""
    stat_names = ("HP", "Atk", "Def", "SpA", "SpD", "Spe")
    blocks: list[str] = []
    for slot in roster:
        particle = determination.get(slot.species)
        if particle is None:
            raise ValueError(f"determination has no set for {slot.display}")
        heading = slot.display
        if particle.item:
            heading += f" @ {particle.item}"
        lines = [heading]
        if particle.ability:
            lines.append(f"Ability: {particle.ability}")
        lines.append("Level: 50")
        nature = "Serious"
        spread = particle.spread or default_spread(slot.species, particle.moves)
        if spread and ":" in spread:
            candidate_nature, _, values_text = spread.partition(":")
            try:
                values = [int(value) for value in values_text.split("/")]
            except ValueError:
                values = []
            if len(values) == 6 and all(0 <= value <= 32 for value in values):
                evs = [
                    f"{value} {stat}"
                    for stat, value in zip(stat_names, values)
                    if value
                ]
                if evs:
                    lines.append("EVs: " + " / ".join(evs))
                nature = candidate_nature.strip() or nature
        lines.append(f"{nature} Nature")
        lines.extend(f"- {move}" for move in particle.moves)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


@dataclass(frozen=True)
class RevealEvidence:
    moves: frozenset[str] = frozenset()
    item: str | None = None
    ability: str | None = None

    @classmethod
    def build(
        cls,
        moves: Iterable[str] = (),
        item: str | None = None,
        ability: str | None = None,
    ) -> "RevealEvidence":
        return cls(
            frozenset(to_id_str(move) for move in moves if move),
            _id(item) or None,
            _id(ability) or None,
        )


class ParticleDatabase:
    """Build bounded species priors from joint top-player and marginal data."""

    def __init__(
        self,
        joint_sets: Mapping[str, object],
        marginals: Mapping[str, object],
        max_particles: int = 12,
    ):
        if max_particles < 1:
            raise ValueError("max_particles must be positive")
        self.joint_sets = joint_sets
        self.marginals = marginals
        self.max_particles = max_particles

    @staticmethod
    def _same_family(candidate: SetParticle, representative: SetParticle) -> bool:
        """Three shared moves plus item/ability represents one tactical set family."""
        return bool(
            candidate.ability == representative.ability
            and candidate.item == representative.item
            and len(set(candidate.moves) & set(representative.moves)) >= 3
        )

    @classmethod
    def _cluster_representatives(
        cls, candidates: Sequence[SetParticle], width: int
    ) -> list[SetParticle]:
        """Greedy weighted medoids cover more held-out sets than raw top-k rows."""
        uncovered = set(range(len(candidates)))
        selected: list[SetParticle] = []
        while uncovered and len(selected) < width:
            best_index = max(
                range(len(candidates)),
                key=lambda representative: sum(
                    candidates[index].probability
                    for index in uncovered
                    if cls._same_family(candidates[index], candidates[representative])
                ),
            )
            covered = {
                index
                for index in uncovered
                if cls._same_family(candidates[index], candidates[best_index])
            }
            mass = sum(candidates[index].probability for index in covered)
            count = sum(candidates[index].count for index in covered)
            selected.append(
                replace(candidates[best_index], probability=mass, count=count)
            )
            uncovered.difference_update(covered)
        return selected

    @classmethod
    def load(
        cls,
        root: Path | None = None,
        max_particles: int = 12,
        formatid: str | None = None,
    ) -> "ParticleDatabase":
        data = root or Path(__file__).resolve().parents[2] / "data"
        from vgc_bench.src.set_priors import prior_reg

        reg = prior_reg(formatid)
        joint_path = data / f"joint_sets_reg{reg}.json"
        joint_sets = json.loads(joint_path.read_text()) if joint_path.exists() else {}
        rare_path = data / f"rare_sets_reg{reg}.json"
        if rare_path.exists():
            joint_sets.update(json.loads(rare_path.read_text()))
        return cls(
            joint_sets,
            json.loads((data / f"movesets_reg{reg}.json").read_text())
            if (data / f"movesets_reg{reg}.json").exists()
            else {},
            max_particles=max_particles,
        )

    def _lookup_species(self, species: str) -> str:
        """Map preview-only Mega forms to the replay table's base set identity."""
        species_id = to_id_str(species)
        if species_id in self.joint_sets:
            return species_id
        entry = GenData.from_gen(9).pokedex.get(species_id) or {}
        candidates = [
            to_id_str(entry.get("battleOnly") or ""),
            to_id_str(entry.get("baseSpecies") or ""),
            to_id_str(entry.get("species") or ""),
        ]
        joint = next(
            (candidate for candidate in candidates if candidate in self.joint_sets),
            None,
        )
        if joint is not None:
            return joint
        if species_id in self.marginals:
            return species_id
        return next(
            (candidate for candidate in candidates if candidate in self.marginals),
            species_id,
        )

    @lru_cache(maxsize=None)
    def particles(self, species: str) -> tuple[SetParticle, ...]:
        species_id = to_id_str(species)
        lookup_id = self._lookup_species(species_id)
        marginal = self.marginals.get(lookup_id)
        marginal = marginal if isinstance(marginal, dict) else {}
        spread = marginal.get("spread")
        entry = self.joint_sets.get(lookup_id)
        entry = entry if isinstance(entry, dict) else {}
        raw_sets = entry.get("sets")
        raw_sets = raw_sets if isinstance(raw_sets, list) else []

        joint_candidates: list[SetParticle] = []
        for raw in raw_sets:
            if not isinstance(raw, dict):
                continue
            moves = tuple(
                move_id
                for move in raw.get("moves", [])
                if move and (move_id := to_id_str(move)) not in _PLACEHOLDER_MOVES
            )
            if not moves:
                continue
            joint_candidates.append(
                SetParticle(
                    species=species_id,
                    ability=_id(raw.get("ability")) or None,
                    item=_id(raw.get("item")) or None,
                    moves=moves[:4],
                    spread=str(spread) if spread else None,
                    probability=max(0.0, float(raw.get("prob") or 0.0)),
                    source="top-player-joint",
                    count=max(0, int(raw.get("count") or 0)),
                )
            )

        candidates = self._cluster_representatives(joint_candidates, self.max_particles)
        marginal_moves = tuple(
            move_id
            for move in marginal.get("moves", [])
            if move and (move_id := to_id_str(move)) not in _PLACEHOLDER_MOVES
        )[:4]
        if marginal_moves:
            marginal_particle = SetParticle(
                species=species_id,
                ability=_id(marginal.get("ability")) or None,
                item=_id(marginal.get("item")) or None,
                moves=marginal_moves,
                spread=str(spread) if spread else None,
                probability=0.02 if joint_candidates else 1.0,
                source="smogon-marginal",
            )
            if marginal_particle.signature not in {x.signature for x in candidates}:
                candidates.append(marginal_particle)

        if not candidates:
            return ()
        candidates.sort(
            key=lambda particle: (particle.probability, particle.count), reverse=True
        )
        selected = candidates[: self.max_particles]
        probabilities = _normalise([particle.probability for particle in selected])
        return tuple(
            replace(particle, probability=probability)
            for particle, probability in zip(selected, probabilities)
        )

    @lru_cache(maxsize=None)
    def top_coverage(self, species: str, width: int = 8) -> float:
        """Original joint-set probability represented by the top ``width`` sets."""
        species_id = to_id_str(species)
        entry = self.joint_sets.get(self._lookup_species(species_id))
        if not isinstance(entry, dict) or not isinstance(entry.get("sets"), list):
            return 0.0
        raw: list[SetParticle] = []
        for candidate in entry["sets"]:
            if not isinstance(candidate, dict):
                continue
            raw.append(
                SetParticle(
                    species=species_id,
                    ability=_id(candidate.get("ability")) or None,
                    item=_id(candidate.get("item")) or None,
                    moves=tuple(
                        to_id_str(move) for move in candidate.get("moves", []) if move
                    )[:4],
                    spread=None,
                    probability=max(0.0, float(candidate.get("prob") or 0.0)),
                    source="coverage",
                )
            )
        representatives = self._cluster_representatives(raw, width)
        return min(1.0, sum(particle.probability for particle in representatives))


class SpeciesBelief:
    """Mutable posterior over one species' bounded particles."""

    def __init__(self, species: str, particles: Sequence[SetParticle]):
        self.species = to_id_str(species)
        self.particles = tuple(particles)
        self.weights = _normalise([particle.probability for particle in particles])
        self.evidence = RevealEvidence()

    @classmethod
    def from_database(cls, database: ParticleDatabase, species: str):
        return cls(species, database.particles(species))

    def posterior(self) -> tuple[SetParticle, ...]:
        return tuple(
            replace(particle, probability=weight)
            for particle, weight in zip(self.particles, self.weights)
            if weight > 0
        )

    def condition(self, evidence: RevealEvidence) -> None:
        """Hard-remove particles contradicted by public move/item/ability reveals."""
        self.evidence = RevealEvidence(
            self.evidence.moves | evidence.moves,
            evidence.item or self.evidence.item,
            evidence.ability or self.evidence.ability,
        )
        compatible = []
        for particle in self.particles:
            compatible.append(
                self.evidence.moves.issubset(particle.moves)
                and (
                    self.evidence.item is None
                    or self.evidence.item == to_id_str(particle.item or "")
                )
                and (
                    self.evidence.ability is None
                    or self.evidence.ability == to_id_str(particle.ability or "")
                )
            )
        if any(compatible):
            self.weights = _normalise(
                [
                    weight if keep else 0.0
                    for weight, keep in zip(self.weights, compatible)
                ]
            )
            return

        # A real ladder set can be novel. Preserve the observed facts in one explicit
        # low-confidence particle instead of resurrecting sets known to be impossible.
        base = max(
            self.particles,
            key=lambda particle: particle.probability,
            default=SetParticle(self.species, None, None, (), None, 1.0, "novel"),
        )
        remaining = [move for move in base.moves if move not in self.evidence.moves]
        moves = tuple(self.evidence.moves) + tuple(remaining)
        novel = SetParticle(
            species=self.species,
            ability=self.evidence.ability or base.ability,
            item=self.evidence.item or base.item,
            moves=moves[:4],
            spread=base.spread,
            probability=1.0,
            source="observed-novel",
            novel=True,
        )
        self.particles = (novel,)
        self.weights = (1.0,)

    def update_likelihood(
        self, likelihood: Callable[[SetParticle], float], floor: float = 1e-6
    ) -> None:
        """Bayesian update for speed/damage evidence computed by an exact observer."""
        weighted = [
            weight * max(floor, float(likelihood(particle)))
            for particle, weight in zip(self.particles, self.weights)
        ]
        self.weights = _normalise(weighted)

    def update_numeric_interval(
        self,
        observed: float,
        predicted: Mapping[
            tuple[str | None, str | None, tuple[str, ...]], tuple[float, float]
        ],
        tolerance: float = 0.02,
    ) -> None:
        """Update from speed or damage evidence represented as predicted intervals."""
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")

        def likelihood(particle: SetParticle) -> float:
            interval = predicted.get(particle.signature)
            if interval is None:
                return 0.25
            low, high = sorted((float(interval[0]), float(interval[1])))
            if low - tolerance <= observed <= high + tolerance:
                return 1.0
            distance = min(abs(observed - low), abs(observed - high))
            return math.exp(-0.5 * (distance / tolerance) ** 2)

        self.update_likelihood(likelihood)

    def sample(self, rng: random.Random) -> SetParticle:
        if not self.particles:
            raise ValueError(f"no set particles available for {self.species}")
        return rng.choices(self.particles, weights=self.weights, k=1)[0]


class TeamBelief:
    """Opponent full-team belief with Item Clause-constrained sampling.

    Champions permits several Pokemon to hold their Mega stones; Showdown enforces
    the one-Mega-per-battle rule when choices are submitted. Removing extra stones
    here would incorrectly erase legitimate preview options such as Floette plus
    Charizard or Swampert plus Metagross.
    """

    def __init__(self, beliefs: Mapping[str, SpeciesBelief]):
        self.beliefs = dict(beliefs)

    @classmethod
    def from_roster(
        cls, database: ParticleDatabase, roster: Iterable[str]
    ) -> "TeamBelief":
        return cls(
            {
                to_id_str(species): SpeciesBelief.from_database(database, species)
                for species in roster
            }
        )

    def condition(
        self,
        species: str,
        *,
        moves: Iterable[str] = (),
        item: str | None = None,
        ability: str | None = None,
    ) -> None:
        species_id = to_id_str(species)
        if species_id not in self.beliefs:
            return
        self.beliefs[species_id].condition(
            RevealEvidence.build(moves=moves, item=item, ability=ability)
        )

    def sample_determinizations(
        self, count: int, rng: random.Random, *, open_sheet: bool = False
    ) -> tuple[dict[str, SetParticle], ...]:
        if count < 1:
            raise ValueError("count must be positive")
        target = 1 if open_sheet else min(8, count)
        species_order = tuple(self.beliefs)
        results: list[dict[str, SetParticle]] = []
        signatures: set[tuple[tuple[str, tuple], ...]] = set()
        attempts = max(100, target * 50)
        # First as ever: every Pokemon one of its weighted sets, no item twice. Only
        # if no such team exists at all, a second pass in which a Pokemon whose
        # weighted sets all hold a taken item keeps one of them with its item left
        # open. What the opponent has SHOWN (the weights) is fact, a stored item is
        # a guess. Until 2026-10-09 a set with weight zero was offered to
        # random.choices in that corner, which refuses it ("Total of weights must be
        # greater than zero"), or nothing was legal ("could not sample a legal
        # opponent determinization") -- no world, no search, for the first turns of
        # two ladder games in fifteen.
        for item_may_stay_open in (False, True):
            for _ in range(attempts):
                selected: dict[str, SetParticle] = {}
                used_items: set[str] = set()
                valid = True
                for species in species_order:
                    belief = self.beliefs[species]
                    options = list(belief.particles)
                    weights = list(belief.weights)
                    legal = [
                        index
                        for index, particle in enumerate(options)
                        if weights[index] > 0
                        and not (
                            particle.item and to_id_str(particle.item) in used_items
                        )
                    ]
                    if legal:
                        chosen_index = rng.choices(
                            legal, weights=[weights[index] for index in legal], k=1
                        )[0]
                        particle = options[chosen_index]
                    elif item_may_stay_open and options:
                        pool = [i for i, weight in enumerate(weights) if weight > 0]
                        pool = pool or list(range(len(options)))
                        chosen_index = rng.choices(
                            pool, weights=[weights[i] or 1.0 for i in pool], k=1
                        )[0]
                        particle = replace(options[chosen_index], item=None)
                    else:
                        valid = False
                        break
                    selected[species] = particle
                    if particle.item:
                        used_items.add(to_id_str(particle.item))
                if not valid:
                    continue
                signature = tuple(
                    (species, selected[species].signature) for species in species_order
                )
                if signature in signatures and len(signatures) < target:
                    continue
                signatures.add(signature)
                results.append(selected)
                if len(results) >= target:
                    break
            if results:
                break
        if not results:
            raise ValueError("could not sample a legal opponent determinization")
        return tuple(results)
