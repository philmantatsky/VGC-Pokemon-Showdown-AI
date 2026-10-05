"""The bot's shipped opponent models behind the predictor contract (a baseline).

``LegacyPredictor`` wraps ``opponent_tactics.MovePredictor`` and
``opponent_tactics.SwitchPredictor`` (the two files the ladder bot loads for its
opponent-aware reranker) as a ``features.Predictor``, so the scorecard can put
them through exactly the harness that scores the count tables and the neural
model (OPPONENT_PREDICTOR.md, "Models": legacy baseline).

    legacy = LegacyPredictor.load(move_path, switch_path, featurizer)
    legacy = legacy.with_config(**training_rates(train_batch, legacy))
    pred = legacy.predict(sheet_unknown_as_closed(features_only))

What the two networks read (the same seven inputs): the actor's six base
species, the other side's six, the actor's two active species, the other side's
two, four HP fractions (actor a, actor b, other a, other b), the acting slot and
the turn number. All of it is in a batch: ``mon_id[..., ID_SPECIES]`` rows 0-5
and 6-11, ``act_mon`` / ``foe_mon``, ``mon_hp`` and ``turn``. Species and move
ids are translated by NAME into the legacy vocabularies (base-species ids, as
``events.base_species_id`` gives them); a name they lack is their index 0.

Every mapping choice (``MAPPING_NOTES`` holds the same list for reports):

1. Empty slot. A slot with no Pokemon is fed as species index 0 with HP 0. The
   networks' training data kept the fainted Pokemon's species there; a batch no
   longer says which Pokemon that was.
2. P(switch) is ``SwitchPredictor.calibrate(sigmoid(logit))``, the number the
   bot reads; it is 0 for a slot with no legal switch pointer.
3. Switch destination is the switch network's softmax over the six roster
   entries (it masks the two active species itself), restricted to the batch's
   legal pointers and renormalised, so P(switch) keeps the calibrated value;
   uniform over the legal pointers when that leaves no mass. The bot also
   multiplies by its preview model's bring marginals; a batch has none.
4. Moves, scope ``'candidates'`` (the default; it is how the bot calls the
   network: ``MovePredictor.predict(available_moves=...)`` takes a softmax over
   the listed moves only). The slot's candidates that the move network knows
   share ``1 - OTHER - unknown`` by a softmax of their logits.
   Scope ``'vocabulary'`` instead takes the network's own softmax over its
   whole vocabulary (the distribution it was trained with): each known
   candidate keeps that probability and OTHER gets what is left, the mass the
   network puts on moves that are not candidates.
   ``move_temperature`` divides the move logits before either softmax and
   ``destination_temperature`` the switch network's roster logits. Both are
   1.0, the models as shipped; any other value is a refit and must be named so.
5. Candidates outside the move network's vocabulary share a floor mass:
   ``unknown_rate`` each (capped in total at ``unknown_cap``). The scorecard
   sets ``unknown_rate`` to the training split's rate at which such a
   candidate was the move made (``training_rates``).
6. OTHER (a move outside the candidates), scope ``'candidates'``: the
   featurizer's own ``other_prior`` for the slot, clipped to
   ``[other_floor, other_cap]``. The network has no such output.
7. Targets. The move network's 5-way head per move (self, ally, foe a, foe b,
   field) is read by name into ``TARGET_CLASSES`` with field -> ``auto``, then
   restricted to the candidate's legal targets (``cand_tmask``), floored at
   ``target_floor`` and renormalised. A move the network cannot score, and
   OTHER, get the uniform distribution over their legal targets.
8. Mega is one constant, ``mega_rate``, on every slot whose Pokemon can still
   Mega-evolve, and 0 elsewhere. The scorecard sets it to the training
   split's rate (``training_rates``). Neither network predicts Mega.

Where the baseline is disadvantaged, and where it is not, is ``DISADVANTAGES``.

``LegacyPredictor.predict`` never raises: it counts ``predict_error:<type>``
and returns ``features.uniform_prediction``. ``LegacyPredictor.load`` is a
loader and raises ``OSError`` / ``ValueError``; ``try_load`` returns None and
counts in ``COUNTERS``. ``training_rates`` is the only function here that reads
label arrays, and only of the batch it is handed.
"""

from __future__ import annotations

import threading
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from vgc_bench.src.oppmodel.events import TARGET_AUTO, TARGET_CLASSES
from vgc_bench.src.oppmodel.features import (
    CAND_VALID,
    FLAG_MEGA_POSSIBLE,
    ID_SPECIES,
    N_ROSTER,
    N_TARGET,
    Featurizer,
    expand_target_mask,
    uniform_prediction,
)
from vgc_bench.src.opponent_tactics import MovePredictor, SwitchPredictor

KIND = "legacy"
SCOPE_CANDIDATES = "candidates"
SCOPE_VOCABULARY = "vocabulary"
SCOPES: tuple[str, ...] = (SCOPE_CANDIDATES, SCOPE_VOCABULARY)

MAPPING_NOTES: tuple[str, ...] = (
    "Inputs are rebuilt from the batch: both rosters' base species, the four "
    "active species, their public HP, the acting slot and the turn. Names are "
    "translated into the old models' own vocabularies; a name they lack is "
    "their 'unknown' entry.",
    "An empty slot is fed as 'unknown' species with HP 0.",
    "P(switch) is the switch model's calibrated probability, 0 when no switch "
    "is legal.",
    "The switch destination is the switch model's distribution over the roster, "
    "restricted to the legal destinations and renormalised (uniform when "
    "nothing is left).",
    "Moves, scope 'candidates': a softmax of the move model's scores over the "
    "slot's candidates it knows, as the bot calls it. Scope 'vocabulary': the "
    "move model's own distribution over its whole vocabulary; each known "
    "candidate keeps its probability and OTHER gets the rest.",
    "A candidate the move model does not know gets a fixed floor mass: the "
    "training split's rate at which such a candidate was the move made.",
    "OTHER (a move outside the candidates), scope 'candidates': the "
    "featurizer's own prior for it, clipped to a floor and a cap.",
    "Targets: the move model's five classes (self, ally, foe a, foe b, field) "
    "read by name, field as 'auto', restricted to the legal targets and "
    "renormalised; uniform for a move it cannot score and for OTHER.",
    "Mega: one constant, the training split's rate among Pokemon that can still "
    "Mega-evolve; 0 for the others.",
)
DISADVANTAGES: tuple[str, ...] = (
    "It was trained on an older and smaller corpus, on higher-rated sides only, "
    "and its file holds the last training epoch, not the best one.",
    "It sees species, HP and the turn only: no revealed moves, status, field, "
    "last action or rating. That is the model, not the harness.",
    "It has no output for OTHER, for a candidate outside its vocabulary, for "
    "Mega or for a hidden action; the harness fills those with base rates.",
    "On turns with an empty slot it is fed a blank where its training data had "
    "the fainted Pokemon. The bot does not call it on such turns at all.",
    "Its switch destinations lack the bot's bring-probability weighting; the "
    "harness's legal mask (not fainted, not left at home) does part of that job.",
    "Its move softmax runs over up to twelve candidates; the bot gives it four.",
    "It labelled a spread move with one of the two foes and a terrain-spread "
    "move like an aimed one; the legal-target mask removes most of that.",
    "In its favour: its training split was by game, so it has seen other games "
    "of this scorecard's test players (not the ladder-holdout games).",
)

# Failures of ``try_load`` and ``training_rates``, by exception type.
COUNTERS: Counter[str] = Counter()
_TINY = 1e-12


@dataclass(frozen=True)
class LegacyConfig:
    """The constants of the mapping (module docstring, points 4-8)."""

    scope: str = SCOPE_CANDIDATES
    move_temperature: float = 1.0
    destination_temperature: float = 1.0
    unknown_rate: float = 0.02
    unknown_cap: float = 0.5
    other_floor: float = 1e-3
    other_cap: float = 0.5
    target_floor: float = 1e-4
    mega_rate: float = 0.5
    batch_size: int = 256

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def checked(self) -> "LegacyConfig":
        """These values with every one forced into its legal range."""

        def unit(value: float, low: float = 0.0, high: float = 1.0) -> float:
            number = float(value)
            if not np.isfinite(number):
                return low
            return min(high, max(low, number))

        return LegacyConfig(
            scope=self.scope if self.scope in SCOPES else SCOPE_CANDIDATES,
            move_temperature=unit(self.move_temperature, 0.05, 20.0),
            destination_temperature=unit(self.destination_temperature, 0.05, 20.0),
            unknown_rate=unit(self.unknown_rate),
            unknown_cap=unit(self.unknown_cap, 0.0, 0.9),
            other_floor=unit(self.other_floor, _TINY, 0.5),
            other_cap=unit(self.other_cap, _TINY, 0.9),
            target_floor=unit(self.target_floor),
            mega_rate=unit(self.mega_rate),
            batch_size=max(1, int(self.batch_size)),
        )


def _name_map(names: Any, vocab: Mapping[str, int], size: int) -> np.ndarray:
    """Our id -> legacy index (0 = the legacy model does not know the name)."""
    out = np.zeros(max(1, len(names)), dtype=np.int64)
    for position, name in enumerate(names):
        index = vocab.get(str(name), 0)
        if isinstance(index, int) and 0 < index < size:
            out[position] = index
    out[0] = 0
    return out


def _lookup(table: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """``table[ids]`` with ids outside the table read as 0."""
    ids = np.asarray(ids).astype(np.int64)
    safe = np.where((ids >= 0) & (ids < table.shape[0]), ids, 0)
    return table[safe]


def _target_permutation(names: Any) -> list[int]:
    """For each of our target classes, the legacy head's column."""
    legacy = [str(name) for name in names]
    spare = [i for i, name in enumerate(legacy) if name not in TARGET_CLASSES]
    out: list[int] = []
    for ours in TARGET_CLASSES:
        if ours in legacy:
            out.append(legacy.index(ours))
        elif ours == TARGET_AUTO and spare:
            out.append(spare[0])  # their 'field' class
        else:
            out.append(-1)
    return out


def _softmax(logits: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Softmax over the allowed entries of each row; all zero for an empty row."""
    masked = np.where(allowed, logits, -np.inf)
    top = masked.max(-1, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    weight = np.where(allowed, np.exp(np.minimum(masked - top, 0.0)), 0.0)
    total = weight.sum(-1, keepdims=True)
    return np.divide(weight, total, out=np.zeros_like(weight), where=total > 0)


def _renormalized(values: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """Rows restricted to ``legal`` and summing to 1; uniform over legal if empty."""
    kept = np.where(legal, np.maximum(values, 0.0), 0.0)
    total = kept.sum(-1, keepdims=True)
    count = legal.sum(-1, keepdims=True)
    uniform = np.divide(
        legal.astype(np.float64),
        count,
        out=np.zeros(legal.shape, dtype=np.float64),
        where=count > 0,
    )
    return np.where(total > _TINY, kept / np.maximum(total, _TINY), uniform)


class LegacyPredictor:
    """``features.Predictor`` over the shipped move and switch models.

    ``kind`` is ``'legacy'``. ``counters`` names what was met: slots fed,
    species and candidates the old vocabularies lack, examples with an empty
    slot, and ``predict_error:<type>`` when a call fell back to the uniform
    prediction. Safe to call from several threads (one lock around the two
    forward passes).
    """

    kind = KIND

    def __init__(
        self,
        move: MovePredictor,
        switch: SwitchPredictor,
        featurizer: Featurizer,
        config: LegacyConfig | None = None,
        *,
        name: str = KIND,
        files: Mapping[str, str] | None = None,
    ) -> None:
        self.move = move
        self.switch = switch
        self.featurizer = featurizer
        self.config = (config or LegacyConfig()).checked()
        self.name = str(name)
        self.files = dict(files or {})
        self.counters: Counter[str] = Counter()
        self._lock = threading.Lock()
        species = featurizer.tables.species_ids
        moves = featurizer.tables.move_ids
        self._move_species = _name_map(
            species, move.species_vocab, int(move.model.species_vocab_size)
        )
        self._switch_species = _name_map(
            species, switch.vocab, int(switch.model.vocab_size)
        )
        self._moves = _name_map(moves, move.move_vocab, len(move.model.move_ids))
        self._targets = _target_permutation(MovePredictor.TARGET_NAMES)

    # --- construction --------------------------------------------------------

    @classmethod
    def load(
        cls,
        move_path: Path | str,
        switch_path: Path | str,
        featurizer: Featurizer,
        config: LegacyConfig | None = None,
        *,
        name: str = KIND,
    ) -> "LegacyPredictor":
        """Read both files as the bot does. Raises ``OSError`` / ``ValueError``."""
        paths = {"move": Path(move_path), "switch": Path(switch_path)}
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"no legacy model at {path}")
        try:
            move = MovePredictor.load(paths["move"], "cpu")
            switch = SwitchPredictor.load(paths["switch"], "cpu")
            return cls(
                move,
                switch,
                featurizer,
                config,
                name=name,
                files={key: str(path) for key, path in paths.items()},
            )
        except OSError:
            raise
        except Exception as exc:
            raise ValueError(f"the legacy models did not load: {exc!r}") from exc

    @classmethod
    def try_load(
        cls,
        move_path: Path | str,
        switch_path: Path | str,
        featurizer: Featurizer,
        config: LegacyConfig | None = None,
        *,
        name: str = KIND,
    ) -> "LegacyPredictor | None":
        """``load`` for a caller that must not raise: None on any failure."""
        try:
            return cls.load(move_path, switch_path, featurizer, config, name=name)
        except Exception as exc:
            _failed("load_error", exc)
            return None

    def with_config(self, name: str | None = None, **changes: Any) -> "LegacyPredictor":
        """The same two networks under changed constants (unknown keys ignored)."""
        known = {key: value for key, value in changes.items() if key in _CONFIG_KEYS}
        twin = LegacyPredictor(
            self.move,
            self.switch,
            self.featurizer,
            replace(self.config, **known),
            name=self.name if name is None else name,
            files=self.files,
        )
        twin._lock = self._lock
        return twin

    def describe(self) -> dict[str, Any]:
        """Plain data for a report: files, constants, vocabulary sizes, notes."""
        return {
            "kind": self.kind,
            "name": self.name,
            "files": dict(self.files),
            "config": self.config.to_dict(),
            "species_known_to_move_model": int((self._move_species > 0).sum()),
            "species_known_to_switch_model": int((self._switch_species > 0).sum()),
            "moves_known_to_move_model": int((self._moves > 0).sum()),
            "target_columns": {
                ours: (MovePredictor.TARGET_NAMES[column] if column >= 0 else None)
                for ours, column in zip(TARGET_CLASSES, self._targets)
            },
            "mapping": list(MAPPING_NOTES),
            "disadvantages": list(DISADVANTAGES),
            "counters": dict(self.counters),
        }

    # --- prediction ----------------------------------------------------------

    def _count(self, name: str, amount: int = 1) -> None:
        try:
            if amount:
                self.counters[name] += int(amount)
        except Exception:
            pass

    def predict(self, batch: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Probabilities ``action`` / ``target`` / ``mega``. Never raises."""
        try:
            return self._predict(batch)
        except Exception as exc:
            self._count(f"predict_error:{type(exc).__name__}")
            return uniform_prediction(batch)

    def candidate_moves(self, batch: Mapping[str, np.ndarray]) -> np.ndarray:
        """``[N, 2, n_cand]``: each candidate's index in the move model (0 = none).

        Never raises: an empty array when the batch has no candidates.
        """
        try:
            return self._candidate_moves(batch)
        except Exception as exc:
            self._count(f"candidate_moves_error:{type(exc).__name__}")
            return np.zeros((0, 2, 0), dtype=np.int64)

    def _candidate_moves(self, batch: Mapping[str, np.ndarray]) -> np.ndarray:
        flags = np.asarray(batch["cand_flag"]).astype(np.int64)
        legal = np.asarray(batch["action_mask"])[..., : flags.shape[-1]] > 0
        valid = ((flags & CAND_VALID) > 0) & legal
        return np.where(valid, _lookup(self._moves, batch["cand_move"]), 0)

    def _predict(self, batch: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        config = self.config
        mask = np.asarray(batch["action_mask"]) > 0  # [N, 2, A]
        legal = expand_target_mask(batch["cand_tmask"])  # [N, 2, C + 1, 5]
        n, n_slot = mask.shape[:2]
        n_cand = legal.shape[2] - 1
        act_mon = np.asarray(batch["act_mon"]).astype(np.int64)
        foe_mon = np.asarray(batch["foe_mon"]).astype(np.int64)
        active = act_mon >= 0
        action = np.zeros(mask.shape, dtype=np.float64)
        count = legal.sum(-1, keepdims=True)
        target = np.divide(
            legal.astype(np.float64),
            count,
            out=np.zeros(legal.shape, dtype=np.float64),
            where=count > 0,
        )
        mega = np.zeros((n, n_slot), dtype=np.float64)
        example, slot = np.nonzero(active)
        if example.size == 0:
            return {"action": action, "target": target, "mega": mega}

        species = np.asarray(batch["mon_id"])[..., ID_SPECIES].astype(np.int64)
        hp_all = np.asarray(batch["mon_hp"], dtype=np.float64)
        foe_up = foe_mon >= 0
        actor_rows = np.maximum(act_mon, 0)
        foe_rows = np.maximum(foe_mon, 0)
        actor_species = np.where(active, np.take_along_axis(species, actor_rows, 1), 0)
        foe_species = np.where(foe_up, np.take_along_axis(species, foe_rows, 1), 0)
        hp = np.concatenate(
            [
                np.where(active, np.take_along_axis(hp_all, actor_rows, 1), 0.0),
                np.where(foe_up, np.take_along_axis(hp_all, foe_rows, 1), 0.0),
            ],
            axis=1,
        )
        hp = np.clip(np.nan_to_num(hp, nan=0.0), 0.0, 1.0)
        turn = np.asarray(batch["turn"]).astype(np.int64).reshape(n)
        legacy_moves = self._candidate_moves(batch)  # [N, 2, C]
        raw = self._forward(
            roster=species[:, :N_ROSTER][example],
            other_roster=species[:, N_ROSTER : 2 * N_ROSTER][example],
            active=actor_species[example],
            other_active=foe_species[example],
            hp=hp[example],
            slot=slot,
            turn=turn[example],
            moves=legacy_moves[example, slot],
            temperature=config.move_temperature,
            destination_temperature=config.destination_temperature,
        )

        flags = np.asarray(batch["cand_flag"]).astype(np.int64)[example, slot]
        slot_mask = mask[example, slot]  # [B, A]
        valid = ((flags & CAND_VALID) > 0) & slot_mask[:, :n_cand]
        scored = valid & (legacy_moves[example, slot] > 0)
        unknown = valid & ~scored
        n_unknown = unknown.sum(-1)
        other_legal = slot_mask[:, n_cand]
        unknown_mass = np.minimum(config.unknown_rate * n_unknown, config.unknown_cap)
        moves = np.zeros((example.size, n_cand + 1), dtype=np.float64)
        if config.scope == SCOPE_VOCABULARY:
            kept = np.where(scored, raw["vocabulary"], 0.0)
            rest = np.maximum(1.0 - kept.sum(-1), config.other_floor)
            moves[:, :n_cand] = kept * (1.0 - unknown_mass)[:, None]
            moves[:, n_cand] = np.where(other_legal, rest * (1.0 - unknown_mass), 0.0)
        else:
            prior = np.asarray(batch["other_prior"], dtype=np.float64)[example, slot]
            other = np.clip(
                np.nan_to_num(prior, nan=0.0), config.other_floor, config.other_cap
            )
            other = np.where(other_legal, other, 0.0)
            share = np.maximum(1.0 - other - unknown_mass, 0.0)
            moves[:, :n_cand] = _softmax(raw["logit"], scored) * share[:, None]
            moves[:, n_cand] = other
        moves[:, :n_cand] += np.where(
            unknown, (unknown_mass / np.maximum(n_unknown, 1))[:, None], 0.0
        )
        legal_move = np.concatenate([valid, other_legal[:, None]], axis=1)
        moves = _renormalized(moves, legal_move)

        pointers = slot_mask[:, n_cand + 1 :]
        can_switch = pointers.any(-1)
        p_switch = np.where(can_switch, self._calibrated(raw["switch"]), 0.0)
        p_switch = np.clip(np.nan_to_num(p_switch, nan=0.0), 0.0, 1.0)
        p_switch = np.where(legal_move.any(-1), p_switch, can_switch.astype(float))
        width = min(pointers.shape[1], raw["destination"].shape[1])
        wanted = np.zeros(pointers.shape, dtype=np.float64)
        wanted[:, :width] = raw["destination"][:, :width]
        destination = _renormalized(wanted, pointers)
        row = np.concatenate(
            [moves * (1.0 - p_switch)[:, None], destination * p_switch[:, None]], axis=1
        )
        action[example, slot] = np.where(slot_mask, row, 0.0)

        slot_legal = legal[example, slot]  # [B, C + 1, 5]
        head = raw["target"]  # [B, C, legacy classes]
        ours = np.zeros((example.size, n_cand, N_TARGET), dtype=np.float64)
        for position, column in enumerate(self._targets):
            if 0 <= column < head.shape[-1]:
                ours[:, :, position] = head[:, :, column]
        ours = _renormalized(ours + config.target_floor, slot_legal[:, :n_cand])
        slot_target = target[example, slot]
        slot_target[:, :n_cand] = np.where(
            scored[:, :, None], ours, slot_target[:, :n_cand]
        )
        target[example, slot] = slot_target

        possible = (
            np.take_along_axis(
                np.asarray(batch["mon_flag"])[..., FLAG_MEGA_POSSIBLE], actor_rows, 1
            )
            > 0
        )
        mega[example, slot] = np.where(possible[example, slot], config.mega_rate, 0.0)
        if not (
            np.isfinite(action).all()
            and np.isfinite(target).all()
            and np.isfinite(mega).all()
        ):
            raise FloatingPointError("non-finite probability")

        self._count("slots", int(example.size))
        self._count(
            "examples_with_an_empty_slot", int((~(active & foe_up)).any(1).sum())
        )
        acting = _lookup(self._move_species, actor_species[example, slot])
        self._count("acting_species_unknown_to_move_model", int((acting == 0).sum()))
        self._count("candidates", int(valid.sum()))
        self._count("candidates_unknown_to_move_model", int(unknown.sum()))
        self._count("slots_without_a_known_candidate", int((~scored.any(-1)).sum()))
        return {"action": action, "target": target, "mega": mega}

    def _calibrated(self, probability: np.ndarray) -> np.ndarray:
        """``SwitchPredictor.calibrate`` for an array."""
        calibration = self.switch.calibration
        if not calibration:
            return probability
        bounds = np.asarray(calibration.get("upper_bounds", []), dtype=np.float64)
        rates = np.asarray(calibration.get("rates", []), dtype=np.float64)
        if bounds.size == 0 or bounds.size != rates.size:
            return probability
        index = np.searchsorted(bounds, probability, side="left")
        return rates[np.minimum(index, rates.size - 1)]

    def _forward(
        self,
        *,
        roster: np.ndarray,
        other_roster: np.ndarray,
        active: np.ndarray,
        other_active: np.ndarray,
        hp: np.ndarray,
        slot: np.ndarray,
        turn: np.ndarray,
        moves: np.ndarray,
        temperature: float = 1.0,
        destination_temperature: float = 1.0,
    ) -> dict[str, np.ndarray]:
        """Both networks on ``B`` slots; candidate columns gathered per slot.

        ``logit`` ``[B, C]`` and ``vocabulary`` ``[B, C]`` are each candidate's
        move logit (divided by ``temperature``) and its probability under the
        softmax of those over the whole move vocabulary; ``target``
        ``[B, C, 5]`` is its target distribution in the legacy order;
        ``switch`` ``[B]`` the uncalibrated switch probability and
        ``destination`` ``[B, 6]`` the roster distribution (its logits divided
        by ``destination_temperature``).
        """
        parts: dict[str, list[np.ndarray]] = {
            "logit": [],
            "vocabulary": [],
            "target": [],
            "switch": [],
            "destination": [],
        }
        size = self.config.batch_size
        sides = (roster, other_roster, active, other_active)
        for_move = [_lookup(self._move_species, ids) for ids in sides]
        for_switch = [_lookup(self._switch_species, ids) for ids in sides]
        with self._lock, torch.no_grad():
            for start in range(0, int(slot.shape[0]), size):
                index = slice(start, start + size)
                shared = (
                    torch.as_tensor(hp[index], dtype=torch.float32),
                    torch.as_tensor(slot[index], dtype=torch.long),
                    torch.as_tensor(turn[index], dtype=torch.long),
                )
                move_logits, target_logits = self.move.model(
                    *(
                        torch.as_tensor(ids[index], dtype=torch.long)
                        for ids in for_move
                    ),
                    *shared,
                )
                move_logits = move_logits / temperature
                picked = torch.as_tensor(moves[index], dtype=torch.long)
                parts["logit"].append(move_logits.gather(1, picked).double().numpy())
                parts["vocabulary"].append(
                    move_logits.softmax(-1).gather(1, picked).double().numpy()
                )
                wide = picked[:, :, None].expand(-1, -1, target_logits.shape[-1])
                parts["target"].append(
                    target_logits.gather(1, wide).softmax(-1).double().numpy()
                )
                switch_logit, destination = self.switch.model(
                    *(
                        torch.as_tensor(ids[index], dtype=torch.long)
                        for ids in for_switch
                    ),
                    *shared,
                )
                parts["switch"].append(torch.sigmoid(switch_logit).double().numpy())
                parts["destination"].append(
                    (destination / destination_temperature).softmax(-1).double().numpy()
                )
        return {name: np.concatenate(chunks) for name, chunks in parts.items()}


_CONFIG_KEYS = frozenset(LegacyConfig.__dataclass_fields__)


def _failed(name: str, exc: Exception) -> None:
    try:
        COUNTERS[f"{name}:{type(exc).__name__}"] += 1
    except Exception:
        pass


def training_rates(
    batch: Mapping[str, np.ndarray], predictor: LegacyPredictor
) -> dict[str, float]:
    """Base rates of a TRAINING batch for what the old models cannot express.

    ``mega_rate``: weighted share of Mega evolutions among slots whose Pokemon
    could still Mega-evolve and whose label tells. ``unknown_rate``: among
    visible moves of slots that have a candidate outside the move model's
    vocabulary, weighted moves made with such a candidate divided by the
    weighted number of such candidates. Reads ``y_mega``, ``y_action`` and
    ``m_weight`` (ones when absent). Never raises: a rate that cannot be
    measured is left out, so ``with_config(**rates)`` keeps its default.
    """
    try:
        return _training_rates(batch, predictor)
    except Exception as exc:
        _failed("training_rates", exc)
        return {}


def _training_rates(
    batch: Mapping[str, np.ndarray], predictor: LegacyPredictor
) -> dict[str, float]:
    act_mon = np.asarray(batch["act_mon"]).astype(np.int64)
    active = act_mon >= 0
    shape = active.shape
    weight = np.broadcast_to(
        np.asarray(batch.get("m_weight", np.ones(shape[0])), dtype=np.float64)[:, None],
        shape,
    )
    out: dict[str, float] = {}
    possible = (
        np.take_along_axis(
            np.asarray(batch["mon_flag"])[..., FLAG_MEGA_POSSIBLE],
            np.maximum(act_mon, 0),
            1,
        )
        > 0
    )
    y_mega = np.asarray(batch["y_mega"]).astype(np.int64)
    told = active & possible & (y_mega >= 0)
    if weight[told].sum() > 0:
        out["mega_rate"] = float(
            (weight * (y_mega == 1))[told].sum() / weight[told].sum()
        )
    flags = np.asarray(batch["cand_flag"]).astype(np.int64)
    n_cand = flags.shape[-1]
    valid = (flags & CAND_VALID) > 0
    unknown = valid & (predictor.candidate_moves(batch) == 0)
    y_action = np.asarray(batch["y_action"]).astype(np.int64)
    visible = active & (y_action >= 0) & unknown.any(-1)
    chosen = np.take_along_axis(
        unknown, np.clip(y_action, 0, n_cand - 1)[..., None], -1
    )[..., 0] & (y_action < n_cand)
    offered = (weight * unknown.sum(-1))[visible].sum()
    if offered > 0:
        out["unknown_rate"] = float((weight * chosen)[visible].sum() / offered)
    return out
