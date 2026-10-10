"""Set prior: which moves a Pokemon carries, given the moves it has shown.

``Repertoire`` (features.py) says how often a set key USES each move, whatever
the Pokemon has shown so far. This module answers the other question a human
asks with the sheets closed: "it has shown these two moves; what are the other
two?". It is a table of FULL movesets (with item and ability) read off open
team sheets, and a posterior over the moves of the set given what is shown.

Pure Python + numpy: no torch, no poke-env, no battle objects.

Keys
    ``species_id`` is the REPERTOIRE key, ``features.set_key(forme, species)``:
    a regional or appliance forme keys itself, a Mega or another battle-only
    forme keys the forme it came from, a cosmetic forme its base. For a sheet
    entry that is ``set_key(sheet.forme, sheet.species)``; for a stored example
    it is the name of ``mon_id[..., ID_KEY]``. Moves, items and abilities are
    Showdown ids. Every id is compared as a string (``str(x)`` of anything that
    is not one), so a caller may also key a table by integers as long as it
    asks with the same integers.

Records
    ``SetTable.build(records)`` takes ``(species_id, moves, item, ability,
    weight)`` tuples: one Pokemon of one open sheet. The caller decides what a
    record is (one per battle with the account-cap weight, or one per player
    and set) and MUST give records of TRAINING-split players only: the table
    is fitted knowledge, like the repertoire.

Posterior
    ``table.posterior(species_id, shown_moves, candidates, ...)`` returns
    ``[len(candidates), SET_COLS]`` float32, columns ``COLUMN_NAMES``:

    0 ``has_data``   1.0 when at least ``min_sets`` recorded sets of the species
                     hold every shown move; else the whole result is zeros.
    1 ``p_member``   P(candidate is in the set | shown moves, and the known
                     item / ability), smoothed toward ``p_prior``:
                     ``(S + smoothing * p_prior) / (W + smoothing)`` with ``W``
                     the weight of the consistent sets and ``S`` the part of it
                     that holds the candidate. Exactly 1.0 for a shown move.
    2 ``p_prior``    the candidate's unconditional membership rate.
    3 ``slots_left`` (4 - number of shown moves) / 4, clipped to [0, 1].

    A row whose candidate is empty is zeros. A known item or ability narrows
    the consistent sets only while at least ``min_sets`` of them remain; else
    it is dropped, ability first (the ability in effect on the field can be a
    Mega forme's or a copied one, which no sheet lists). ``strict=True`` turns
    that back-off off: then too few sets under the full condition mean zeros.

    Never raises: anything unexpected is counted in ``table.counters`` and the
    answer is zeros. Answers are cached per (species, shown, item, ability).
"""

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SET_COLS = 4
COL_HAS_DATA, COL_P_MEMBER, COL_P_PRIOR, COL_SLOTS_LEFT = range(SET_COLS)
COLUMN_NAMES: tuple[str, ...] = ("has_data", "p_member", "p_prior", "slots_left")

MAX_SET_MOVES = 4
DEFAULT_MIN_SETS = 5
DEFAULT_MAX_SETS_PER_SPECIES = 2000
DEFAULT_SMOOTHING = 2.0
PAYLOAD_FORMAT = "oppmodel-setprior"
PAYLOAD_VERSION = 1
SIGNATURE_LENGTH = 16
_ROUND = 6
_CACHE_LIMIT = 50_000

SetRecord = tuple[Any, Sequence[Any], Any, Any, float]


def _text(value: Any) -> str:
    """An id as the table compares it: a string, '' for nothing."""
    if isinstance(value, str):
        return value
    return "" if value is None else str(value)


def _move_tuple(moves: Any) -> tuple[str, ...]:
    """Distinct non-empty ids of ``moves``, sorted. A lone string is one move."""
    if moves is None:
        return ()
    if isinstance(moves, str):
        return (moves,) if moves else ()
    return tuple(sorted({text for text in map(_text, moves) if text}))


@dataclass
class _Species:
    """One species' recorded sets: the stored lists and what is derived from them."""

    key: str
    moves: tuple[str, ...]
    items: tuple[str, ...]
    abilities: tuple[str, ...]
    prior: tuple[float, ...]  # membership rate per move, over ALL records
    records: int  # records seen, before the cut to the most common sets
    weight: float
    set_moves: tuple[tuple[int, ...], ...]
    set_item: tuple[int, ...]
    set_ability: tuple[int, ...]
    set_weight: tuple[float, ...]
    set_count: tuple[int, ...]
    move_ix: dict[str, int] = field(default_factory=dict, repr=False)
    item_ix: dict[str, int] = field(default_factory=dict, repr=False)
    ability_ix: dict[str, int] = field(default_factory=dict, repr=False)
    member: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)), repr=False)
    weighted: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)), repr=False)
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    counts: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    item_of: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    ability_of: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    prior_array: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)

    def derive(self) -> None:
        """Fill the lookup dicts and arrays from the stored lists."""
        self.move_ix = {move: index for index, move in enumerate(self.moves)}
        self.item_ix = {item: index for index, item in enumerate(self.items)}
        self.ability_ix = {name: index for index, name in enumerate(self.abilities)}
        member = np.zeros((len(self.set_moves), len(self.moves)), dtype=bool)
        for row, held in enumerate(self.set_moves):
            member[row, list(held)] = True
        self.member = member
        self.weights = np.asarray(self.set_weight, dtype=np.float64)
        self.counts = np.asarray(self.set_count, dtype=np.float64)
        self.weighted = member.astype(np.float64) * self.weights[:, None]
        self.item_of = np.asarray(self.set_item, dtype=np.int64)
        self.ability_of = np.asarray(self.set_ability, dtype=np.int64)
        self.prior_array = np.asarray(self.prior, dtype=np.float64)

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.key,
            "moves": list(self.moves),
            "items": list(self.items),
            "abilities": list(self.abilities),
            "prior": list(self.prior),
            "records": self.records,
            "weight": self.weight,
            "set_moves": [list(held) for held in self.set_moves],
            "set_item": list(self.set_item),
            "set_ability": list(self.set_ability),
            "set_weight": list(self.set_weight),
            "set_count": list(self.set_count),
        }

    @classmethod
    def from_payload(cls, data: Mapping[str, Any]) -> "_Species":
        moves = tuple(str(move) for move in data["moves"])
        items = tuple(str(item) for item in data["items"])
        abilities = tuple(str(name) for name in data["abilities"])
        set_moves = tuple(
            tuple(int(index) for index in held) for held in data["set_moves"]
        )
        set_item = tuple(int(index) for index in data["set_item"])
        set_ability = tuple(int(index) for index in data["set_ability"])
        set_weight = tuple(float(value) for value in data["set_weight"])
        set_count = tuple(int(value) for value in data["set_count"])
        prior = tuple(float(value) for value in data["prior"])
        n_sets = len(set_moves)
        lengths = (len(set_item), len(set_ability), len(set_weight), len(set_count))
        if any(length != n_sets for length in lengths) or len(prior) != len(moves):
            raise ValueError("set prior payload: list lengths disagree")
        if any(not 0 <= index < len(moves) for held in set_moves for index in held):
            raise ValueError("set prior payload: move index out of range")
        if any(not 0 <= index < len(items) for index in set_item):
            raise ValueError("set prior payload: item index out of range")
        if any(not 0 <= index < len(abilities) for index in set_ability):
            raise ValueError("set prior payload: ability index out of range")
        species = cls(
            key=str(data["id"]),
            moves=moves,
            items=items,
            abilities=abilities,
            prior=prior,
            records=int(data["records"]),
            weight=float(data["weight"]),
            set_moves=set_moves,
            set_item=set_item,
            set_ability=set_ability,
            set_weight=set_weight,
            set_count=set_count,
        )
        species.derive()
        return species


def _build_species(
    key: str,
    sets: Mapping[tuple[tuple[str, ...], str, str], list[float]],
    max_sets: int,
) -> _Species:
    """Aggregate one species' distinct sets; keep the ``max_sets`` heaviest."""
    total_weight = sum(cell[0] for cell in sets.values())
    total_count = int(sum(cell[1] for cell in sets.values()))
    member_weight: dict[str, float] = {}
    for (held, _, _), cell in sets.items():
        for move in held:
            member_weight[move] = member_weight.get(move, 0.0) + cell[0]
    moves = tuple(sorted(member_weight))
    move_ix = {move: index for index, move in enumerate(moves)}
    prior = tuple(
        round(min(1.0, member_weight[move] / total_weight), _ROUND)
        if total_weight > 0
        else 0.0
        for move in moves
    )
    # Heaviest first; ties by count, then by the set itself, so the order (and
    # the payload) never depends on the order the records came in.
    order = sorted(sets, key=lambda cell: (-sets[cell][0], -sets[cell][1], cell))
    kept = order[: max(1, max_sets)]
    items = tuple(sorted({cell[1] for cell in kept}))
    abilities = tuple(sorted({cell[2] for cell in kept}))
    item_ix = {item: index for index, item in enumerate(items)}
    ability_ix = {name: index for index, name in enumerate(abilities)}
    species = _Species(
        key=key,
        moves=moves,
        items=items,
        abilities=abilities,
        prior=prior,
        records=total_count,
        weight=round(total_weight, _ROUND),
        set_moves=tuple(tuple(move_ix[move] for move in cell[0]) for cell in kept),
        set_item=tuple(item_ix[cell[1]] for cell in kept),
        set_ability=tuple(ability_ix[cell[2]] for cell in kept),
        set_weight=tuple(round(sets[cell][0], _ROUND) for cell in kept),
        set_count=tuple(int(sets[cell][1]) for cell in kept),
    )
    species.derive()
    return species


class SetTable:
    """Full movesets per set key, and the posterior over a set's moves.

    Build with ``SetTable.build(records)``; store with ``to_payload()`` and
    restore with ``SetTable.from_payload(payload)`` (exact round trip);
    ``signature()`` is a short stable hash of the payload. ``counters`` names
    everything unusual that was met while building or answering.
    """

    def __init__(
        self,
        species: Mapping[str, _Species] | None = None,
        *,
        min_sets: int = DEFAULT_MIN_SETS,
        max_sets_per_species: int = DEFAULT_MAX_SETS_PER_SPECIES,
        smoothing: float = DEFAULT_SMOOTHING,
        build_counts: Mapping[str, int] | None = None,
    ) -> None:
        self.min_sets = max(1, int(min_sets))
        self.max_sets_per_species = max(1, int(max_sets_per_species))
        value = float(smoothing)
        self.smoothing = value if math.isfinite(value) and value >= 0 else 0.0
        self._species: dict[str, _Species] = dict(species or {})
        self.build_counts: dict[str, int] = dict(build_counts or {})
        self.counters: Counter[str] = Counter()
        # How many answers were zeros because of an error (``posterior_error:*``
        # in ``counters``): a caller compares it before and after a call.
        self.errors = 0
        self._cache: dict[
            tuple[str, tuple[str, ...], str | None, str | None, bool],
            tuple[bool, np.ndarray],
        ] = {}

    # --- construction and storage ------------------------------------------

    @classmethod
    def build(
        cls,
        records: Iterable[SetRecord],
        *,
        min_sets: int = DEFAULT_MIN_SETS,
        max_sets_per_species: int = DEFAULT_MAX_SETS_PER_SPECIES,
        smoothing: float = DEFAULT_SMOOTHING,
    ) -> "SetTable":
        """A table from ``(species_id, moves, item, ability, weight)`` records.

        A record without a species or a move, with more than four moves or with
        a weight that is not a positive finite number is skipped and counted. A
        species with fewer than ``min_sets`` records is left out (it could never
        have data). Past ``max_sets_per_species`` distinct sets the lightest are
        dropped from the conditional; ``p_prior`` still counts every record.
        """
        min_sets = max(1, int(min_sets))
        max_sets = max(1, int(max_sets_per_species))
        counts: Counter[str] = Counter()
        grouped: dict[str, dict[tuple[tuple[str, ...], str, str], list[float]]] = {}
        for record in records:
            try:
                key, moves, item, ability, weight = record
                species = _text(key)
                held = _move_tuple(moves)
                amount = float(weight)
            except (TypeError, ValueError):
                counts["record_malformed"] += 1
                continue
            if not species or not held:
                counts["record_without_species_or_move"] += 1
                continue
            if len(held) > MAX_SET_MOVES:
                counts["record_with_too_many_moves"] += 1
                continue
            if not math.isfinite(amount) or amount <= 0:
                counts["record_without_weight"] += 1
                continue
            cell = grouped.setdefault(species, {}).setdefault(
                (held, _text(item), _text(ability)), [0.0, 0.0]
            )
            cell[0] += amount
            cell[1] += 1.0
            counts["records"] += 1
        table: dict[str, _Species] = {}
        for species in sorted(grouped):
            sets = grouped[species]
            n_records = int(sum(cell[1] for cell in sets.values()))
            if n_records < min_sets:
                counts["species_below_min_sets"] += 1
                counts["records_of_species_below_min_sets"] += n_records
                continue
            if len(sets) > max_sets:
                counts["species_cut_to_max_sets"] += 1
                counts["distinct_sets_dropped"] += len(sets) - max_sets
            table[species] = _build_species(species, sets, max_sets)
            counts["species"] += 1
            counts["distinct_sets"] += len(table[species].set_moves)
        return cls(
            table,
            min_sets=min_sets,
            max_sets_per_species=max_sets,
            smoothing=smoothing,
            build_counts=dict(sorted(counts.items())),
        )

    def to_payload(self) -> dict[str, Any]:
        """Plain lists, numbers and strings; the same table gives the same dict."""
        return {
            "format": PAYLOAD_FORMAT,
            "version": PAYLOAD_VERSION,
            "columns": list(COLUMN_NAMES),
            "min_sets": self.min_sets,
            "max_sets_per_species": self.max_sets_per_species,
            "smoothing": self.smoothing,
            "build_counts": dict(sorted(self.build_counts.items())),
            "species": [
                self._species[key].to_payload() for key in sorted(self._species)
            ],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SetTable":
        """Rebuild a table. Raises ``ValueError`` on a payload it cannot read."""
        try:
            if payload.get("format") != PAYLOAD_FORMAT:
                raise ValueError(f"not a set prior payload: {payload.get('format')!r}")
            if payload.get("version") != PAYLOAD_VERSION:
                raise ValueError(
                    f"set prior payload version {payload.get('version')!r}, "
                    f"this reader knows {PAYLOAD_VERSION}"
                )
            if list(payload.get("columns") or []) != list(COLUMN_NAMES):
                raise ValueError("set prior payload: other columns than this reader's")
            species = {
                str(entry["id"]): _Species.from_payload(entry)
                for entry in payload["species"]
            }
            return cls(
                species,
                min_sets=int(payload["min_sets"]),
                max_sets_per_species=int(payload["max_sets_per_species"]),
                smoothing=float(payload["smoothing"]),
                build_counts={
                    str(name): int(count)
                    for name, count in (payload.get("build_counts") or {}).items()
                },
            )
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"set prior payload unreadable: {exc!r}") from exc

    def signature(self) -> str:
        """Short stable hash of the payload."""
        text = json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:SIGNATURE_LENGTH]

    # --- reading -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._species)

    def __contains__(self, species_id: Any) -> bool:
        try:
            return _text(species_id) in self._species
        except Exception:
            return False

    def species_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._species))

    def n_records(self, species_id: Any) -> int:
        """Records seen for a species (0 when it is not in the table)."""
        entry = self._species.get(_text(species_id))
        return 0 if entry is None else entry.records

    def n_distinct_sets(self, species_id: Any) -> int:
        entry = self._species.get(_text(species_id))
        return 0 if entry is None else len(entry.set_moves)

    def sets(
        self, species_id: Any
    ) -> list[tuple[tuple[str, ...], str, str, float, int]]:
        """(moves, item, ability, weight, records) per kept set, heaviest first."""
        entry = self._species.get(_text(species_id))
        if entry is None:
            return []
        return [
            (
                tuple(entry.moves[index] for index in entry.set_moves[row]),
                entry.items[entry.set_item[row]],
                entry.abilities[entry.set_ability[row]],
                entry.set_weight[row],
                entry.set_count[row],
            )
            for row in range(len(entry.set_moves))
        ]

    def stats(self) -> dict[str, Any]:
        return {
            "species": len(self._species),
            "distinct_sets": sum(len(e.set_moves) for e in self._species.values()),
            "records": sum(e.records for e in self._species.values()),
            "min_sets": self.min_sets,
            "max_sets_per_species": self.max_sets_per_species,
            "smoothing": self.smoothing,
            "build_counts": dict(sorted(self.build_counts.items())),
        }

    # --- the posterior ---------------------------------------------------------

    def posterior(
        self,
        species_id: Any,
        shown_moves: Any,
        candidates: Any,
        *,
        item: Any = "",
        ability: Any = "",
        item_known: bool = False,
        ability_known: bool = False,
        strict: bool = False,
    ) -> np.ndarray:
        """``[len(candidates), SET_COLS]`` float32 (module text). Never raises."""
        try:
            names = [_text(move) for move in candidates]
        except Exception:
            self._count("posterior_error:candidates")
            return np.zeros((0, SET_COLS), dtype=np.float32)
        out = np.zeros((len(names), SET_COLS), dtype=np.float32)
        try:
            entry = self._species.get(_text(species_id))
            if entry is None:
                return out
            shown = _move_tuple(shown_moves)
            known_item = _text(item) if item_known else None
            known_ability = _text(ability) if ability_known else None
            key = (entry.key, shown, known_item, known_ability, bool(strict))
            cached = self._cache.get(key)
            if cached is None:
                cached = self._conditional(
                    entry, shown, known_item, known_ability, bool(strict)
                )
                if len(self._cache) >= _CACHE_LIMIT:
                    self._cache.clear()
                self._cache[key] = cached
            has_data, member = cached
            if not has_data:
                return out
            left = min(1.0, max(0.0, (MAX_SET_MOVES - len(shown)) / MAX_SET_MOVES))
            seen = set(shown)
            for row, name in enumerate(names):
                if not name:
                    continue
                index = entry.move_ix.get(name)
                prior = 0.0 if index is None else entry.prior_array[index]
                if name in seen:
                    belief = 1.0
                else:
                    belief = 0.0 if index is None else member[index]
                out[row, COL_HAS_DATA] = 1.0
                out[row, COL_P_MEMBER] = belief
                out[row, COL_P_PRIOR] = prior
                out[row, COL_SLOTS_LEFT] = left
            return out
        except Exception as exc:
            self._count(f"posterior_error:{type(exc).__name__}")
            return np.zeros((len(names), SET_COLS), dtype=np.float32)

    def _conditional(
        self,
        entry: _Species,
        shown: tuple[str, ...],
        item: str | None,
        ability: str | None,
        strict: bool,
    ) -> tuple[bool, np.ndarray]:
        """(has data, P(move in set | condition) per move of the species)."""
        empty = np.zeros(0, dtype=np.float64)
        mask = np.ones(len(entry.set_moves), dtype=bool)
        for move in shown:
            index = entry.move_ix.get(move)
            if index is None:
                return False, empty
            mask &= entry.member[:, index]
        item_mask = ability_mask = None
        if item is not None:
            index = entry.item_ix.get(item)
            item_mask = np.zeros_like(mask) if index is None else entry.item_of == index
        if ability is not None:
            index = entry.ability_ix.get(ability)
            ability_mask = (
                np.zeros_like(mask) if index is None else entry.ability_of == index
            )
        full = mask
        if item_mask is not None:
            full = full & item_mask
        if ability_mask is not None:
            full = full & ability_mask
        # The full condition first, then what is left of it when too few
        # recorded sets agree: without the ability, then without the item too.
        levels: list[np.ndarray] = [full]
        if not strict:
            if item_mask is not None and ability_mask is not None:
                levels.append(mask & item_mask)
            if item_mask is not None or ability_mask is not None:
                levels.append(mask)
        for depth, level in enumerate(levels):
            if float(entry.counts[level].sum()) < self.min_sets:
                continue
            if depth:
                self._count("posterior_backed_off")
            weight = float(entry.weights[level].sum())
            held = level.astype(np.float64) @ entry.weighted
            total = weight + self.smoothing
            if total <= 0:
                return False, empty
            member = (held + self.smoothing * entry.prior_array) / total
            return True, np.clip(member, 0.0, 1.0)
        return False, empty

    def _count(self, name: str) -> None:
        try:
            if name.startswith("posterior_error"):
                self.errors += 1
            self.counters[name] += 1
        except Exception:
            pass
