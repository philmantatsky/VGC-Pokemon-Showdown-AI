"""OppNet: the neural opponent predictor (OPPONENT_PREDICTOR.md, "Models").

A small transformer over 15 tokens (1 context, 2 sides, 12 Pokemon with the
actor's roster first) reads one collated batch of ``features.Featurizer``
examples and scores, for each of the actor's two slots:

* ``action``  ``[N, 2, n_cand + 7]``: candidates, OTHER, six switch pointers;
* ``target``  ``[N, 2, n_cand + 1, 5]``: target class given each move action;
* ``mega``    ``[N, 2]``: the slot Mega-evolves this turn; exactly 0 where the
  public state says the Pokemon cannot (its ``mega possible`` flag is the
  legality mask of this head, as ``action_mask`` is for the actions).

The species prior is a logit OFFSET: ``log(cand_prior + eps)`` for a candidate,
``log(other_prior + eps)`` for OTHER and one learned scalar for the switch
pointers. Every scorer added to those offsets ends in a zero-initialised layer,
so an untrained network returns exactly the normalised prior and training
learns the residual.

    net = OppNet.for_featurizer(featurizer)            # sizes from the vocabulary
    out = net(to_tensors(batch))                       # raw logits and masks
    terms = nll_terms(out, to_labels(batch))           # per-slot losses
    predictor = OppNetPredictor(net, featurizer)       # features.Predictor
    probabilities = predictor.predict(batch)           # numpy, never raises
    predictor = from_payload(predictor.to_payload(), featurizer)

The batch is the plain collated one (NOT ``Featurizer.densify``-ed): the dex
numerics are gathered here, in torch, from the featurizer's tables, which the
network holds as non-trainable buffers. Ids beyond an embedding table read row
0; ids of extension rows keep their numerics.

``nll_terms`` is the training loss and equals ``features.slot_nll`` term by term
(action = ``-log`` of the probability mass on ``y_set``, target = cross-entropy
on the row of the true move, mega = binary cross-entropy where the Pokemon can
Mega-evolve), except that it does not floor a probability, so a badly wrong
prediction still has a gradient.

``swap_slots`` mirrors slot ``a`` <-> ``b`` of either side across every feature
and label array. Doubles is symmetric under that renaming, so it is a valid
augmentation; it is checked against a mirrored snapshot in the unit tests.

Event calibration. An ``OppNetPredictor`` may carry one
``calibration.EventCalibration`` (fitted on validation after the temperatures
by ``training/calibrate_oppmodel.py``). ``predict`` then rescales, inside each
slot's action distribution, the switch pointers and the Protect-family
candidates so that the two event probabilities are calibrated; targets and the
Mega head are untouched. It is applied exactly once, in ``predict``, and
stored in the payload; a payload without one loads as "no calibration". A
payload WITH one is written as version 2 (without: version 1, unchanged), so
the model code from before event calibration, which reads version 1 only,
refuses a calibrated artifact instead of serving it uncalibrated under the
calibrated name. A map with context terms (turn 1, first turn on the field,
protected last turn) reads those flags from the feature arrays of the batch.

Runtime contract: ``OppNetPredictor.predict`` and ``swap_slots`` never raise
(they count under ``counters`` / ``COUNTERS`` and degrade). The constructors,
``from_payload`` and the training helpers are offline tools and raise
``ValueError`` on malformed input.
"""

from __future__ import annotations

import math
import threading
from collections import Counter
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as fn

from vgc_bench.src.oppmodel.calibration import EventCalibration, event_context
from vgc_bench.src.oppmodel.events import (
    INTENT_ATTACK_FOE_A,
    INTENT_ATTACK_FOE_B,
    INTENT_CLASSES,
)
from vgc_bench.src.oppmodel.features import (
    BOOST_STATS,
    CAT_LAST_TARGET,
    CAT_SLOT,
    ELO_BLANK,
    ELO_KEEP,
    FLAG_MEGA_POSSIBLE,
    FLAG_PRESENT,
    G_ACTOR_SHEET,
    G_BO3,
    G_OTHER_SHEET,
    G_RATED,
    ID_ABILITY,
    ID_FORME,
    ID_ITEM,
    ID_KEY,
    ID_LAST_MOVE,
    ID_SPECIES,
    MAX_AGE,
    MAX_MOVES,
    N_MON,
    N_MON_CAT,
    N_MON_FLAG,
    N_ROSTER,
    N_SLOT,
    N_TARGET,
    SHEET_UNKNOWN,
    SLOT_A,
    SLOT_B,
    T_ALLY,
    T_AUTO,
    T_FOE_A,
    T_FOE_B,
    T_SELF,
    Batch,
    Featurizer,
    apply_elo_mode,
    expand_target_mask,
    sheet_unknown_as_closed,
    uniform_prediction,
)

KIND = "oppnet"
PAYLOAD_FORMAT = "oppnet-payload"
PAYLOAD_VERSION = 1
# A payload that carries an event calibration is written as version 2: the
# model code from before event calibration reads version 1 only, so it
# refuses such a file instead of serving it uncalibrated under its name.
PAYLOAD_VERSION_CALIBRATED = 2
PAYLOAD_VERSIONS: tuple[int, ...] = (PAYLOAD_VERSION, PAYLOAD_VERSION_CALIBRATED)
NEG = -1.0e9  # a masked logit: finite, so an all-masked row stays free of NaN
N_TOKENS = 3 + N_MON  # context, actor side, other side, twelve Pokemon
N_CAND_BITS = 10  # the CAND_* bits of ``cand_flag``
N_COUNT = 8  # embedding rows of a small count (0 .. 7)

# Feature arrays the network reads, by the tensor type it wants.
LONG_KEYS: tuple[str, ...] = (
    "turn",
    "ctx_cat",
    "game_flag",
    "side_cnt",
    "mon_id",
    "mon_move",
    "mon_move_flag",
    "mon_cat",
    "mon_type",
    "mon_vol",
    "act_mon",
    "foe_mon",
    "cand_move",
    "cand_flag",
    "cand_rank",
    "cand_tmask",
)
FLOAT_KEYS: tuple[str, ...] = (
    "field_age",
    "elo",
    "elo_known",
    "side_age",
    "mon_hp",
    "mon_boost",
    "mon_flag",
    "slot_support",
    "cand_prior",
    "cand_auto",
    "other_prior",
)
BOOL_KEYS: tuple[str, ...] = ("action_mask",)
FEATURE_KEYS: tuple[str, ...] = LONG_KEYS + FLOAT_KEYS + BOOL_KEYS
LABEL_KEYS: tuple[str, ...] = ("y_set", "y_action", "y_target", "y_mega")

# Arrays with one row per ACTOR slot: mirrored by reversing axis 1.
_ACTOR_SLOT_ARRAYS: tuple[str, ...] = (
    "act_mon",
    "slot_support",
    "cand_move",
    "cand_flag",
    "cand_prior",
    "cand_rank",
    "cand_auto",
    "cand_tmask",
    "cand_num",
    "other_prior",
    "switch_mask",
    "action_mask",
    "y_kind",
    "y_action",
    "y_set",
    "y_target",
    "y_mega",
    "y_intent",
    "y_flag",
    "y_attack",
    "y_reason",
)
_INTENT_FOE_A = INTENT_CLASSES.index(INTENT_ATTACK_FOE_A)
_INTENT_FOE_B = INTENT_CLASSES.index(INTENT_ATTACK_FOE_B)

# Failures of the module-level helpers a runtime may call, by helper and exception.
COUNTERS: Counter[str] = Counter()


def _failed(name: str, exc: Exception) -> None:
    try:
        COUNTERS[f"{name}:{type(exc).__name__}"] += 1
    except Exception:
        pass


# --- slot mirror --------------------------------------------------------------


def _pick(mask: Any, swapped: np.ndarray, original: np.ndarray) -> np.ndarray:
    """``swapped`` where ``mask`` says so (True, or one flag per example)."""
    if mask is True:
        return np.ascontiguousarray(swapped)
    rows = np.asarray(mask, dtype=bool).reshape((-1,) + (1,) * (original.ndim - 1))
    return np.where(rows, swapped, original).astype(original.dtype, copy=False)


def _exchange(values: np.ndarray, first: int, second: int) -> np.ndarray:
    """``values`` with the codes ``first`` and ``second`` exchanged."""
    return np.where(
        values == first, second, np.where(values == second, first, values)
    ).astype(values.dtype, copy=False)


def _wanted(flag: Any) -> bool:
    return bool(np.any(flag)) if not isinstance(flag, bool) else flag


def swap_slots(
    batch: Mapping[str, np.ndarray], actor: Any = True, other: Any = True
) -> Batch:
    """The batch with slot ``a`` <-> ``b`` renamed on the chosen side(s).

    ``actor`` / ``other`` are a bool, or one bool per example (a per-example
    augmentation). Every array that depends on a slot letter follows:

    * actor side: the per-slot arrays (candidates, masks, priors, every ``y_*``
      row) and ``act_mon`` are reversed, the slot code of the actor's Pokemon
      is exchanged, and so is the foe letter in the OTHER side's last targets;
    * other side: ``foe_mon`` is reversed, the slot code of the other side's
      Pokemon is exchanged, and ``foe_a`` <-> ``foe_b`` is exchanged wherever a
      target of the actor names it: the legal-target bits, ``y_target``, the
      ``y_attack`` columns, the two attack intents and the actor's last targets.

    Arrays without a slot (meta arrays included) are shared, not copied. Never
    raises: a batch it cannot mirror comes back unchanged (see ``COUNTERS``).
    """
    try:
        return _swap_slots(batch, actor, other)
    except Exception as exc:
        _failed("swap_slots", exc)
        return dict(batch)


def _swap_slots(batch: Mapping[str, np.ndarray], actor: Any, other: Any) -> Batch:
    out: Batch = dict(batch)
    do_actor, do_other = _wanted(actor), _wanted(other)
    if not do_actor and not do_other:
        return out
    cats = np.asarray(batch["mon_cat"]) if "mon_cat" in batch else None
    new_cats = None if cats is None else cats.copy()
    foe_a, foe_b = 1 + T_FOE_A, 1 + T_FOE_B  # ``CAT_LAST_TARGET`` codes
    if do_actor:
        for name in _ACTOR_SLOT_ARRAYS:
            if name in out:
                array = np.asarray(out[name])
                out[name] = _pick(actor, array[:, ::-1], array)
        if cats is not None and new_cats is not None:
            mine = cats[:, :N_ROSTER, CAT_SLOT]
            new_cats[:, :N_ROSTER, CAT_SLOT] = _pick(
                actor, _exchange(mine, SLOT_A, SLOT_B), mine
            )
            theirs = cats[:, N_ROSTER:, CAT_LAST_TARGET]
            new_cats[:, N_ROSTER:, CAT_LAST_TARGET] = _pick(
                actor, _exchange(theirs, foe_a, foe_b), theirs
            )
    if do_other:
        if "foe_mon" in out:
            array = np.asarray(out["foe_mon"])
            out["foe_mon"] = _pick(other, array[:, ::-1], array)
        if cats is not None and new_cats is not None:
            theirs = cats[:, N_ROSTER:, CAT_SLOT]
            new_cats[:, N_ROSTER:, CAT_SLOT] = _pick(
                other, _exchange(theirs, SLOT_A, SLOT_B), theirs
            )
            mine = cats[:, :N_ROSTER, CAT_LAST_TARGET]
            new_cats[:, :N_ROSTER, CAT_LAST_TARGET] = _pick(
                other, _exchange(mine, foe_a, foe_b), mine
            )
        if "cand_tmask" in out:
            bits = np.asarray(out["cand_tmask"])
            wide = bits.astype(np.int64)
            keep = wide & ~((1 << T_FOE_A) | (1 << T_FOE_B))
            moved = (((wide >> T_FOE_A) & 1) << T_FOE_B) | (
                ((wide >> T_FOE_B) & 1) << T_FOE_A
            )
            out["cand_tmask"] = _pick(other, (keep | moved).astype(bits.dtype), bits)
        if "y_target" in out:
            array = np.asarray(out["y_target"])
            out["y_target"] = _pick(other, _exchange(array, T_FOE_A, T_FOE_B), array)
        if "y_attack" in out:
            array = np.asarray(out["y_attack"])
            out["y_attack"] = _pick(other, array[..., ::-1], array)
        if "y_intent" in out:
            array = np.asarray(out["y_intent"])
            out["y_intent"] = _pick(
                other, _exchange(array, _INTENT_FOE_A, _INTENT_FOE_B), array
            )
    if new_cats is not None:
        out["mon_cat"] = new_cats
    return out


def strip_labels(batch: Mapping[str, np.ndarray]) -> Batch:
    """The batch without its label (``y_*``) and meta (``m_*``) arrays."""
    return {
        name: array
        for name, array in batch.items()
        if not name.startswith(("y_", "m_"))
    }


# --- tensors ------------------------------------------------------------------


def _array(batch: Mapping[str, np.ndarray], name: str, index: Any) -> np.ndarray:
    array = np.asarray(batch[name])
    return array if index is None else array[index]


def to_tensors(
    batch: Mapping[str, np.ndarray],
    device: torch.device | str | None = None,
    index: Any = None,
) -> dict[str, Tensor]:
    """The feature arrays the network reads, as tensors (labels are not read).

    ``index`` (a slice or an index array) selects examples first. Raises
    ``KeyError`` when a feature array is missing.
    """
    out: dict[str, Tensor] = {}
    for name in LONG_KEYS:
        values = np.ascontiguousarray(_array(batch, name, index), dtype=np.int64)
        out[name] = torch.from_numpy(values)
    for name in FLOAT_KEYS:
        values = np.ascontiguousarray(_array(batch, name, index), dtype=np.float32)
        out[name] = torch.from_numpy(values)
    for name in BOOL_KEYS:
        values = np.ascontiguousarray(_array(batch, name, index) > 0)
        out[name] = torch.from_numpy(values)
    if device is not None:
        out = {name: value.to(device) for name, value in out.items()}
    return out


def to_labels(
    batch: Mapping[str, np.ndarray],
    device: torch.device | str | None = None,
    index: Any = None,
) -> dict[str, Tensor]:
    """Label tensors of ``nll_terms`` plus the example weight (``m_weight`` or 1)."""
    out: dict[str, Tensor] = {
        "y_set": torch.from_numpy(
            np.ascontiguousarray(_array(batch, "y_set", index) > 0)
        )
    }
    for name in ("y_action", "y_target", "y_mega"):
        values = np.ascontiguousarray(_array(batch, name, index), dtype=np.int64)
        out[name] = torch.from_numpy(values)
    if "m_weight" in batch:
        weight = np.ascontiguousarray(
            _array(batch, "m_weight", index), dtype=np.float32
        )
    else:
        weight = np.ones(out["y_action"].shape[0], dtype=np.float32)
    out["weight"] = torch.from_numpy(weight)
    if device is not None:
        out = {name: value.to(device) for name, value in out.items()}
    return out


def target_mask(cand_tmask: Tensor) -> Tensor:
    """``features.expand_target_mask`` in torch: bits to a trailing target axis."""
    bits = torch.arange(N_TARGET, device=cand_tmask.device)
    return ((cand_tmask[..., None] >> bits) & 1) > 0


# --- configuration ------------------------------------------------------------


@dataclass(frozen=True)
class OppNetConfig:
    """Shape of the network. The ``n_*`` / ``*_width`` sizes come from a Featurizer."""

    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    pre_norm: bool = True
    d_embed: int = 48  # species / move / item / ability embeddings
    d_cat: int = 8  # small categorical embeddings
    prior_eps: float = 1.0e-3
    switch_logit_init: float = -3.2  # start value of the learned switch offset
    elo_bucket: int = 50  # rating points per bucket
    elo_buckets: int = 48  # known-rating buckets; the last holds everything above
    elo_center: float = 1200.0
    elo_scale: float = 300.0
    max_turn: int = 31
    n_cand: int = 12
    n_species: int = 1
    n_moves: int = 1
    n_items: int = 1
    n_abilities: int = 1
    n_types: int = 1
    n_statuses: int = 1
    n_weathers: int = 1
    n_terrains: int = 1
    n_volatiles: int = 1
    n_field: int = 2  # width of ``field_age``
    n_side_conditions: int = 0
    species_width: int = 0
    move_width: int = 0

    @classmethod
    def for_featurizer(cls, featurizer: Featurizer, **overrides: Any) -> "OppNetConfig":
        sizes = featurizer.vocab.sizes()
        made = cls(
            n_cand=featurizer.n_cand,
            n_species=featurizer.n_species,
            n_moves=featurizer.n_moves,
            n_items=sizes["items"],
            n_abilities=sizes["abilities"],
            n_types=sizes["types"],
            n_statuses=sizes["statuses"],
            n_weathers=sizes["weathers"],
            n_terrains=sizes["terrains"],
            n_volatiles=sizes["volatiles"],
            n_field=2 + len(featurizer.vocab.pseudo),
            n_side_conditions=len(featurizer.vocab.side_conditions),
            species_width=int(featurizer.tables.species_num.shape[1]),
            move_width=int(featurizer.tables.move_num.shape[1]),
        )
        return replace(made, **overrides) if overrides else made

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OppNetConfig":
        """Rebuild from ``to_dict()``. Raises ``ValueError`` on a bad mapping."""
        try:
            values: dict[str, Any] = {}
            for item in fields(cls):
                if item.name in data:
                    # Each field has a plain default; its type is the field's type.
                    values[item.name] = type(item.default)(data[item.name])
            return cls(**values)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"not an OppNet config: {exc!r}") from exc

    def check(self) -> None:
        if self.d_model <= 0 or self.n_heads <= 0 or self.d_model % self.n_heads:
            raise ValueError("d_model must be a positive multiple of n_heads")
        if self.n_layers < 0 or self.d_ff <= 0 or self.n_cand <= 0:
            raise ValueError("n_layers, d_ff and n_cand must be positive")
        if not 0.0 <= self.dropout < 1.0 or self.prior_eps <= 0.0:
            raise ValueError("dropout must be in [0, 1) and prior_eps positive")
        if self.elo_bucket <= 0 or self.elo_buckets <= 0 or self.elo_scale <= 0:
            raise ValueError("Elo bucket settings must be positive")
        if self.species_width <= 0 or self.move_width <= 0:
            raise ValueError("table widths are missing: build from a Featurizer")


# --- network ------------------------------------------------------------------


def _ids(values: Tensor, size: int) -> Tensor:
    """Ids safe for a table of ``size`` rows: anything outside reads row 0."""
    return torch.where((values >= 0) & (values < size), values, 0)


class _Block(nn.Module):
    """One encoder layer: self-attention and a feed-forward, pre- or post-norm."""

    def __init__(self, config: OppNetConfig) -> None:
        super().__init__()
        self.pre_norm = config.pre_norm
        self.norm_attention = nn.LayerNorm(config.d_model)
        self.norm_feed = nn.LayerNorm(config.d_model)
        self.attention = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.feed_in = nn.Linear(config.d_model, config.d_ff)
        self.feed_out = nn.Linear(config.d_ff, config.d_model)
        self.drop = nn.Dropout(config.dropout)

    def _attend(self, hidden: Tensor, pad: Tensor) -> Tensor:
        return self.attention(
            hidden, hidden, hidden, key_padding_mask=pad, need_weights=False
        )[0]

    def _feed(self, hidden: Tensor) -> Tensor:
        return self.feed_out(fn.gelu(self.feed_in(hidden)))

    def forward(self, tokens: Tensor, pad: Tensor) -> Tensor:
        if self.pre_norm:
            tokens = tokens + self.drop(self._attend(self.norm_attention(tokens), pad))
            return tokens + self.drop(self._feed(self.norm_feed(tokens)))
        tokens = self.norm_attention(tokens + self.drop(self._attend(tokens, pad)))
        return self.norm_feed(tokens + self.drop(self._feed(tokens)))


class _Scorer(nn.Module):
    """Score a tuple of vectors: summed projections, ReLU, one output layer.

    Summing one projection per input equals a linear layer over their
    concatenation, but each projection runs before the inputs are broadcast
    against each other. With ``zero`` the output layer starts at zero, so the
    scorer adds nothing until it is trained. No dropout in here: the hidden
    tensor is the largest of the network (slots x candidates x foes), and its
    inputs are already dropped where they are made.
    """

    def __init__(self, n_inputs: int, width: int, out: int, zero: bool = True) -> None:
        super().__init__()
        self.inputs = nn.ModuleList(
            nn.Linear(width, width, bias=position == 0) for position in range(n_inputs)
        )
        self.out = nn.Linear(width, out)
        if zero:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, *vectors: Tensor) -> Tensor:
        hidden = self.inputs[0](vectors[0])
        for layer, vector in zip(list(self.inputs)[1:], vectors[1:]):
            hidden = hidden + layer(vector)
        return self.out(fn.relu(hidden))


class OppNet(nn.Module):
    """The transformer and its three heads. ``forward`` returns raw logits.

    ``species_num`` / ``move_num`` are the featurizer's numeric tables
    (extension rows included); they are buffers outside the state dict, so a
    stored network reloads against whatever dex the featurizer was rebuilt on.
    """

    species_num: Tensor
    move_num: Tensor
    move_rows: Tensor
    bit_range: Tensor

    def __init__(
        self, config: OppNetConfig, species_num: np.ndarray, move_num: np.ndarray
    ) -> None:
        super().__init__()
        config.check()
        species = np.asarray(species_num, dtype=np.float32)
        moves = np.asarray(move_num, dtype=np.float32)
        if species.ndim != 2 or species.shape[1] != config.species_width:
            raise ValueError("species table does not match the config")
        if moves.ndim != 2 or moves.shape[1] != config.move_width:
            raise ValueError("move table does not match the config")
        if species.shape[0] < 1 or moves.shape[0] < 1:
            raise ValueError("empty numeric table")
        self.config = config
        c = config
        wide, small, d = c.d_embed, c.d_cat, c.d_model
        self.register_buffer("species_num", torch.from_numpy(species.copy()), False)
        self.register_buffer("move_num", torch.from_numpy(moves.copy()), False)
        rows = torch.arange(moves.shape[0])
        self.register_buffer("move_rows", torch.where(rows < c.n_moves, rows, 0), False)
        self.register_buffer("bit_range", torch.arange(N_CAND_BITS), False)

        self.emb_species = nn.Embedding(c.n_species, wide)
        self.emb_item = nn.Embedding(c.n_items, wide)
        self.emb_ability = nn.Embedding(c.n_abilities, wide)
        self.emb_move = nn.Embedding(c.n_moves, wide)
        self.move_proj = nn.Linear(c.move_width, wide)
        self.emb_move_flag = nn.Embedding(4, wide)
        cat_sizes = [0] * N_MON_CAT
        cat_sizes[0] = c.n_statuses  # CAT_STATUS
        cat_sizes[1] = cat_sizes[2] = cat_sizes[3] = 3  # slot, item state, brought
        cat_sizes[4] = 5  # last action kind
        cat_sizes[5] = N_TARGET + 1  # last target
        cat_sizes[6] = 4  # protect streak
        cat_sizes[7] = cat_sizes[8] = MAX_AGE + 1  # turns on field, known moves
        self.cat_sizes = tuple(cat_sizes)
        self.emb_cat = nn.ModuleList(nn.Embedding(size, small) for size in cat_sizes)
        self.emb_type = nn.Embedding(c.n_types, small, padding_idx=0)
        self.emb_vol = nn.Embedding(c.n_volatiles, 2 * small, padding_idx=0)
        mon_width = (
            7 * wide  # species, forme, set key, item, ability, last move, moves
            + N_MON_CAT * small
            + N_MON_FLAG
            + small  # types
            + 2 * small  # volatiles
            + c.species_width
            + 3  # HP
            + len(BOOST_STATS)
        )
        self.mon_proj = nn.Linear(mon_width, d)
        self.emb_role = nn.Embedding(2, d)  # actor / other
        self.emb_position = nn.Embedding(N_ROSTER, d)

        self.emb_turn = nn.Embedding(c.max_turn + 1, 2 * small)
        self.emb_weather = nn.Embedding(c.n_weathers, small)
        self.emb_terrain = nn.Embedding(c.n_terrains, small)
        self.emb_setter = nn.Embedding(3, small)
        self.emb_sheet = nn.Embedding(3, small)
        self.emb_elo = nn.Embedding(c.elo_buckets + 1, 2 * small)  # row 0 = unknown
        elo_width = 2 * small + 2
        ctx_width = 2 * small + 1 + 3 * small + 2 * c.n_field + 2 + 2 * small
        self.ctx_proj = nn.Linear(ctx_width + 2 * elo_width, d)
        self.emb_count = nn.ModuleList(nn.Embedding(N_COUNT, small) for _ in range(5))
        side_width = 5 * small + 5 + 2 * c.n_side_conditions + elo_width
        self.side_proj = nn.Linear(side_width, d)
        self.token_norm = nn.LayerNorm(d)
        self.token_drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(_Block(c) for _ in range(c.n_layers))
        self.final_norm = nn.LayerNorm(d)
        self.no_mon = nn.Parameter(torch.zeros(d))  # stands in for an empty slot

        self.cand_in = nn.Linear(wide + N_CAND_BITS + 6, d)
        self.cand_drop = nn.Dropout(c.dropout)
        self.other_cand = nn.Parameter(torch.zeros(d))  # OTHER's row in the targets
        self.other_feat = nn.Linear(2, d)
        self.head_drop = nn.Dropout(c.dropout)  # on the encoder output
        self.act_cand = _Scorer(2, d, 1)
        self.act_other = _Scorer(1, d, 1)
        self.act_switch = _Scorer(2, d, 1)
        self.switch_bias = nn.Parameter(torch.tensor(float(c.switch_logit_init)))
        self.tgt_foe = _Scorer(3, d, 1)
        self.tgt_ally = _Scorer(3, d, 1)
        self.tgt_own = _Scorer(2, d, 2)  # self, auto
        self.mega = _Scorer(1, d, 1, zero=False)
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.3)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()
        nn.init.normal_(self.other_cand, std=0.3)

    @classmethod
    def for_featurizer(cls, featurizer: Featurizer, **overrides: Any) -> "OppNet":
        """A fresh network sized for ``featurizer``; overrides are config fields."""
        config = OppNetConfig.for_featurizer(featurizer, **overrides)
        return cls(config, featurizer.tables.species_num, featurizer.tables.move_num)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # --- token inputs -------------------------------------------------------

    def _move_table(self) -> Tensor:
        """One vector per move row: its embedding plus its projected numerics."""
        return self.emb_move(self.move_rows) + self.move_proj(self.move_num)

    def _elo(self, elo: Tensor, known: Tensor) -> Tensor:
        """``[N, 2, 2 * d_cat + 2]``; reads the rating only where it is known."""
        c = self.config
        is_known = known > 0.5
        bucket = (elo.clamp(min=0.0) / c.elo_bucket).floor().long()
        bucket = torch.where(is_known, 1 + bucket.clamp(max=c.elo_buckets - 1), 0)
        scaled = torch.where(
            is_known, (elo - c.elo_center) / c.elo_scale, torch.zeros_like(elo)
        )
        return torch.cat(
            [self.emb_elo(bucket), scaled[..., None], is_known.float()[..., None]], -1
        )

    def _context(self, x: Mapping[str, Tensor], elo: Tensor) -> Tensor:
        c = self.config
        turn = x["turn"]
        cats = x["ctx_cat"]
        flags = x["game_flag"]
        age = x["field_age"]
        parts = [
            self.emb_turn(turn.clamp(0, c.max_turn)),
            (turn.float() / 20.0).clamp(max=3.0)[:, None],
            self.emb_weather(_ids(cats[:, 0], c.n_weathers)),
            self.emb_terrain(_ids(cats[:, 1], c.n_terrains)),
            self.emb_setter(_ids(cats[:, 2], 3)),
            age / MAX_AGE,
            (age > 0).float(),
            flags[:, G_BO3 : G_BO3 + 1].float(),
            flags[:, G_RATED : G_RATED + 1].float(),
            self.emb_sheet(_ids(flags[:, G_ACTOR_SHEET], 3)),
            self.emb_sheet(_ids(flags[:, G_OTHER_SHEET], 3)),
            elo.flatten(1),
        ]
        return self.ctx_proj(torch.cat(parts, -1))

    def _sides(self, x: Mapping[str, Tensor], elo: Tensor) -> Tensor:
        counts = x["side_cnt"]
        age = x["side_age"]
        parts = [
            table(counts[..., column].clamp(0, N_COUNT - 1))
            for column, table in enumerate(self.emb_count)
        ]
        parts += [
            counts.float().clamp(max=float(N_COUNT)) / N_ROSTER,
            age / MAX_AGE,
            (age > 0).float(),
            elo,
        ]
        return self.side_proj(torch.cat(parts, -1)) + self.emb_role.weight[None]

    def _mons(self, x: Mapping[str, Tensor], move_table: Tensor) -> Tensor:
        c = self.config
        ids = x["mon_id"]
        n_rows = move_table.shape[0]
        known = x["mon_move"]
        known_vec = move_table[_ids(known, n_rows)] + self.emb_move_flag(
            _ids(x["mon_move_flag"], 4)
        )
        pooled = (known_vec * (known > 0).float()[..., None]).sum(2) / MAX_MOVES
        cats = x["mon_cat"]
        hp = x["mon_hp"]
        parts = [
            self.emb_species(_ids(ids[..., ID_SPECIES], c.n_species)),
            self.emb_species(_ids(ids[..., ID_FORME], c.n_species)),
            self.emb_species(_ids(ids[..., ID_KEY], c.n_species)),
            self.emb_item(_ids(ids[..., ID_ITEM], c.n_items)),
            self.emb_ability(_ids(ids[..., ID_ABILITY], c.n_abilities)),
            move_table[_ids(ids[..., ID_LAST_MOVE], n_rows)],
            pooled,
        ]
        parts += [
            table(_ids(cats[..., column], self.cat_sizes[column]))
            for column, table in enumerate(self.emb_cat)
        ]
        parts += [
            x["mon_flag"],
            self.emb_type(_ids(x["mon_type"], c.n_types)).sum(2),
            self.emb_vol(_ids(x["mon_vol"], c.n_volatiles)).sum(2),
            self.species_num[_ids(ids[..., ID_FORME], self.species_num.shape[0])],
            torch.stack([hp, (hp <= 0.5).float(), (hp <= 0.25).float()], -1),
            x["mon_boost"] / 6.0,
        ]
        tokens = self.mon_proj(torch.cat(parts, -1))
        role = self.emb_role.weight.repeat_interleave(N_ROSTER, 0)
        position = self.emb_position.weight.repeat(2, 1)
        return tokens + (role + position)[None]

    def encode(self, x: Mapping[str, Tensor]) -> Tensor:
        """Encoder output, ``[N, 15, d_model]``."""
        elo = self._elo(x["elo"], x["elo_known"])
        move_table = self._move_table()
        tokens = torch.cat(
            [
                self._context(x, elo)[:, None],
                self._sides(x, elo),
                self._mons(x, move_table),
            ],
            1,
        )
        tokens = self.token_drop(self.token_norm(tokens))
        absent = x["mon_flag"][..., FLAG_PRESENT] < 0.5
        pad = torch.cat([torch.zeros_like(absent[:, :3]), absent], 1)
        for block in self.blocks:
            tokens = block(tokens, pad)
        return self.final_norm(tokens)

    def _token_at(self, mons: Tensor, index: Tensor) -> Tensor:
        """Encoder output of the Pokemon rows ``index`` ``[N, K]``; -1 = nobody."""
        safe = _ids(index, mons.shape[1])
        picked = mons.gather(1, safe[..., None].expand(-1, -1, mons.shape[-1]))
        return torch.where((index >= 0)[..., None], picked, self.no_mon)

    @staticmethod
    def _can_mega(mon_flag: Tensor, index: Tensor) -> Tensor:
        """``[N, 2]``: the slot holds a Pokemon whose ``mega possible`` flag is set."""
        flag = mon_flag[..., FLAG_MEGA_POSSIBLE]
        picked = flag.gather(1, _ids(index, flag.shape[1]))
        return (picked > 0.5) & (index >= 0)

    def _candidates(self, x: Mapping[str, Tensor], move_table: Tensor) -> Tensor:
        """Candidate encodings, ``[N, 2, n_cand, d_model]``."""
        eps = self.config.prior_eps
        prior = x["cand_prior"]
        rank = x["cand_rank"].float()
        support = x["slot_support"][..., None].expand_as(prior)
        parts = [
            move_table[_ids(x["cand_move"], move_table.shape[0])],
            ((x["cand_flag"][..., None] >> self.bit_range) & 1).float(),
            prior[..., None],
            (1.0 - torch.log(prior + eps) / math.log(eps))[..., None],
            (rank.clamp(max=16.0) / 16.0)[..., None],
            (rank == 0).float()[..., None],
            x["cand_auto"][..., None],
            (support / 10.0)[..., None],
        ]
        return self.cand_drop(fn.relu(self.cand_in(torch.cat(parts, -1))))

    # --- heads --------------------------------------------------------------

    def forward(self, x: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Raw logits ``action`` / ``target`` / ``mega`` and the masks they need.

        The logits are NOT masked: ``probabilities`` and ``nll_terms`` apply
        ``action_mask``, ``target_mask`` and ``mega_mask`` (the slot's Pokemon
        can Mega-evolve). ``active`` marks the slots that hold a Pokemon; an
        empty slot has nothing legal and is predicted as all zeros.
        """
        c = self.config
        eps = c.prior_eps
        hidden = self.head_drop(self.encode(x))
        mons = hidden[:, 3:]
        act = x["act_mon"]
        query = self._token_at(mons, act)  # [N, 2, d]
        foes = self._token_at(mons, x["foe_mon"])  # [N, 2, d]
        partner = query.flip(1)
        roster = mons[:, :N_ROSTER]
        cand = self._candidates(x, self._move_table())  # [N, 2, C, d]

        cand_logit = torch.log(x["cand_prior"] + eps) + self.act_cand(
            query[:, :, None], cand
        ).squeeze(-1)
        other_prior = x["other_prior"]
        other_in = query + self.other_feat(
            torch.stack(
                [
                    x["slot_support"] / 10.0,
                    1.0 - torch.log(other_prior + eps) / math.log(eps),
                ],
                -1,
            )
        )
        other_logit = torch.log(other_prior + eps) + self.act_other(other_in).squeeze(
            -1
        )
        switch_logit = self.switch_bias + self.act_switch(
            query[:, :, None], roster[:, None]
        ).squeeze(-1)
        action = torch.cat([cand_logit, other_logit[..., None], switch_logit], -1)

        rows = torch.cat(
            [cand, self.other_cand.expand(cand.shape[0], N_SLOT, 1, -1)], 2
        )  # [N, 2, C + 1, d]
        to_foe = self.tgt_foe(
            query[:, :, None, None], rows[:, :, :, None], foes[:, None, None]
        ).squeeze(-1)  # [N, 2, C + 1, 2]: foe slot a, b
        to_ally = self.tgt_ally(query[:, :, None], rows, partner[:, :, None])
        own = self.tgt_own(query[:, :, None], rows)  # self, auto
        columns: list[Tensor] = [to_ally] * N_TARGET
        columns[T_FOE_A] = to_foe[..., 0:1]
        columns[T_FOE_B] = to_foe[..., 1:2]
        columns[T_ALLY] = to_ally
        columns[T_SELF] = own[..., 0:1]
        columns[T_AUTO] = own[..., 1:2]
        return {
            "action": action,
            "target": torch.cat(columns, -1),
            "mega": self.mega(query).squeeze(-1),
            "action_mask": x["action_mask"],
            "target_mask": target_mask(x["cand_tmask"]),
            "mega_mask": self._can_mega(x["mon_flag"], act),
            "active": act >= 0,
        }


# --- probabilities and loss ---------------------------------------------------


def probabilities(
    out: Mapping[str, Tensor],
    action_temperature: float = 1.0,
    target_temperature: float = 1.0,
    mega_temperature: float = 1.0,
    mega_bias: float = 0.0,
) -> dict[str, Tensor]:
    """Masked, tempered probabilities from ``OppNet.forward`` output.

    Illegal actions and targets get exactly zero; a slot or a move with nothing
    legal is all zeros, and so is the Mega probability of a slot whose Pokemon
    cannot Mega-evolve (``mega_mask``). The Mega logit is calibrated as
    ``logit / mega_temperature + mega_bias`` before its sigmoid.
    """
    action_mask = out["action_mask"]
    legal = out["target_mask"]
    action = (out["action"] / action_temperature).masked_fill(~action_mask, NEG)
    target = (out["target"] / target_temperature).masked_fill(~legal, NEG)
    mega = out["mega"] / mega_temperature + mega_bias
    return {
        "action": torch.softmax(action, -1) * action_mask,
        "target": torch.softmax(target, -1) * legal,
        "mega": torch.sigmoid(mega) * out["mega_mask"],
    }


def nll_terms(
    out: Mapping[str, Tensor],
    labels: Mapping[str, Tensor],
    action_temperature: float = 1.0,
    target_temperature: float = 1.0,
    mega_temperature: float = 1.0,
    mega_bias: float = 0.0,
) -> dict[str, Tensor]:
    """Per-slot loss terms ``[N, 2]``, each with a ``*_scored`` mask.

    ``out`` holds logits ``action`` / ``target`` / ``mega`` and the masks
    ``action_mask`` / ``target_mask`` / ``mega_mask``; ``labels`` holds
    ``y_set``, ``y_action``,
    ``y_target`` and ``y_mega``. The terms are those of ``features.slot_nll``:

    * ``action``: ``-log`` of the mass on ``y_set`` (a visible choice is a set
      of one); a slot with an empty set is not scored;
    * ``target``: cross-entropy over the legal targets of the true move's row,
      where the move is visible and its target certain;
    * ``mega``: binary cross-entropy where ``y_mega`` is known and the Pokemon
      can Mega-evolve. Elsewhere the probability is exactly 0 by the mask (and
      no Mega label occurs there), so there is nothing to learn.

    Unscored entries are 0. No probability is floored. One difference to
    ``features.slot_nll``: a target label that the legal-target mask excludes
    is NOT scored here (``slot_nll`` charges it the floor's worth). No such
    label exists in a dataset built by the featurizer; ``count_masked_targets``
    counts them so that a trainer can say so.
    """
    action_mask = out["action_mask"]
    legal = out["target_mask"]
    logits = (out["action"] / action_temperature).masked_fill(~action_mask, NEG)
    log_p = torch.log_softmax(logits, -1)
    y_set = labels["y_set"] & action_mask
    scored = y_set.any(-1)
    mass = torch.logsumexp(log_p.masked_fill(~y_set, NEG), -1)
    action = torch.where(scored, -mass, torch.zeros_like(mass))

    n_move = legal.shape[2]
    y_action, y_target = labels["y_action"], labels["y_target"]
    target_logits = (out["target"] / target_temperature).masked_fill(~legal, NEG)
    target_log_p = torch.log_softmax(target_logits, -1)
    row = y_action.clamp(0, n_move - 1)[..., None, None].expand(-1, -1, 1, N_TARGET)
    column = y_target.clamp(0, N_TARGET - 1)[..., None]
    picked = target_log_p.gather(2, row)[:, :, 0].gather(2, column)[..., 0]
    allowed = legal.gather(2, row)[:, :, 0].gather(2, column)[..., 0]
    target_scored = (
        scored & (y_action >= 0) & (y_action < n_move) & (y_target >= 0) & allowed
    )
    target = torch.where(target_scored, -picked, torch.zeros_like(picked))

    y_mega = labels["y_mega"]
    mega_scored = (y_mega >= 0) & out["mega_mask"]
    mega_loss = fn.binary_cross_entropy_with_logits(
        out["mega"] / mega_temperature + mega_bias,
        (y_mega == 1).to(out["mega"].dtype),
        reduction="none",
    )
    mega = torch.where(mega_scored, mega_loss, torch.zeros_like(mega_loss))
    return {
        "action": action,
        "action_scored": scored,
        "target": target,
        "target_scored": target_scored,
        "mega": mega,
        "mega_scored": mega_scored,
    }


def total_loss(
    terms: Mapping[str, Tensor], weight: Tensor, mega_weight: float = 0.2
) -> tuple[Tensor, dict[str, float]]:
    """The scalar to minimise, and its parts as floats.

    Fine loss = weighted mean, over scored slots, of ``action + target`` (the
    design's fine NLL); the Mega term is its own weighted mean, added with
    ``mega_weight``. ``weight`` is one value per example (``m_weight``).
    """
    per_slot = weight[:, None]
    slots = (per_slot * terms["action_scored"]).sum().clamp(min=1e-8)
    action = (per_slot * terms["action"]).sum() / slots
    target = (per_slot * terms["target"]).sum() / slots
    megas = (per_slot * terms["mega_scored"]).sum().clamp(min=1e-8)
    mega = (per_slot * terms["mega"]).sum() / megas
    loss = action + target + mega_weight * mega
    parts = {
        "loss": float(loss.detach()),
        "fine": float((action + target).detach()),
        "action": float(action.detach()),
        "target": float(target.detach()),
        "mega": float(mega.detach()),
    }
    return loss, parts


def clone_state(module: nn.Module) -> dict[str, Tensor]:
    """A snapshot of a module's state that later training cannot change.

    ``state_dict()`` returns the live tensors; ``.detach().cpu()`` of a CPU
    tensor still shares their memory. Every tensor is cloned here.
    """
    return {
        name: value.detach().to("cpu").clone()
        for name, value in module.state_dict().items()
    }


def collect_outputs(
    net: OppNet,
    batch: Mapping[str, np.ndarray],
    device: torch.device | str = "cpu",
    batch_size: int = 1024,
) -> dict[str, Tensor]:
    """``OppNet.forward`` over a whole batch, in eval mode, as CPU tensors."""
    n = int(np.asarray(batch["action_mask"]).shape[0])
    was_training = net.training
    net.eval()
    parts: list[dict[str, Tensor]] = []
    try:
        with torch.no_grad():
            for start in range(0, n, max(1, batch_size)):
                index = slice(start, start + max(1, batch_size))
                out = net(to_tensors(batch, device, index))
                parts.append({name: value.to("cpu") for name, value in out.items()})
    finally:
        net.train(was_training)
    if not parts:
        raise ValueError("empty batch")
    return {name: torch.cat([part[name] for part in parts]) for name in parts[0]}


def fit_temperature(
    objective: Callable[[float], float],
    low: float = 0.2,
    high: float = 5.0,
    iterations: int = 40,
) -> float:
    """The temperature in ``[low, high]`` that minimises ``objective``.

    Golden-section search on ``log T``. Returns 1.0 unless the found value is
    strictly better than no temperature at all.
    """
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left, right = math.log(low), math.log(high)
    first = right - ratio * (right - left)
    second = left + ratio * (right - left)
    value_first, value_second = objective(math.exp(first)), objective(math.exp(second))
    for _ in range(max(1, iterations)):
        if value_first <= value_second:
            right, second, value_second = second, first, value_first
            first = right - ratio * (right - left)
            value_first = objective(math.exp(first))
        else:
            left, first, value_first = first, second, value_second
            second = left + ratio * (right - left)
            value_second = objective(math.exp(second))
    best = math.exp((left + right) / 2.0)
    found, plain = objective(best), objective(1.0)
    if not (math.isfinite(found) and found < plain):
        return 1.0
    return float(best)


def fit_offset(
    objective: Callable[[float], float],
    low: float = -4.0,
    high: float = 4.0,
    iterations: int = 40,
) -> float:
    """The offset in ``[low, high]`` that minimises ``objective``.

    Golden-section search. Returns 0.0 unless the found value is strictly
    better than no offset at all.
    """
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left, right = float(low), float(high)
    first = right - ratio * (right - left)
    second = left + ratio * (right - left)
    value_first, value_second = objective(first), objective(second)
    for _ in range(max(1, iterations)):
        if value_first <= value_second:
            right, second, value_second = second, first, value_first
            first = right - ratio * (right - left)
            value_first = objective(first)
        else:
            left, first, value_first = first, second, value_second
            second = left + ratio * (right - left)
            value_second = objective(second)
    best = (left + right) / 2.0
    found, plain = objective(best), objective(0.0)
    if not (math.isfinite(found) and found < plain):
        return 0.0
    return float(best)


def count_masked_targets(batch: Mapping[str, np.ndarray]) -> int:
    """Target labels that ``features.slot_nll`` scores and the legal-target
    mask excludes: the labels on which ``nll_terms`` and ``slot_nll`` differ.

    Zero in a dataset the featurizer built (it drops such a target and counts
    ``target_outside_mask``). Raises ``KeyError`` on a batch without labels.
    """
    mask = np.asarray(batch["action_mask"]).astype(bool)
    y_set = np.asarray(batch["y_set"]).astype(bool) & mask
    y_action = np.asarray(batch["y_action"]).astype(np.int64)
    y_target = np.asarray(batch["y_target"]).astype(np.int64)
    legal = expand_target_mask(batch["cand_tmask"])
    n_move = legal.shape[2]
    scored = y_set.any(-1) & (y_action >= 0) & (y_action < n_move) & (y_target >= 0)
    row = np.clip(y_action, 0, n_move - 1)[..., None, None]
    chosen = np.take_along_axis(legal, row, axis=2)[:, :, 0, :]
    column = np.clip(y_target, 0, N_TARGET - 1)[..., None]
    allowed = np.take_along_axis(chosen, column, axis=2)[..., 0]
    return int((scored & ~allowed).sum())


# --- predictor ----------------------------------------------------------------


class OppNetPredictor:
    """``features.Predictor`` over an ``OppNet``: numpy in, numpy out.

    Eval mode, ``torch.no_grad``, CPU unless told otherwise, batched, with the
    stored temperatures applied (one per head; the Mega head also has a bias:
    ``logit / mega_temperature + mega_bias``). ``predict`` is safe to call from several
    threads (one lock around the forward pass) and never raises: on any failure
    it counts under ``counters`` and returns ``features.uniform_prediction``.

    The predictor maps the "unknown" sheet code to "closed" itself (no training
    example carries it) and, with ``elo_mode='blank'``, blanks both ratings
    before every prediction: that is the Elo-blind model's runtime.

    ``event_calibration`` (a ``calibration.EventCalibration`` or None) is
    applied to the action probabilities once, after the temperatures, inside
    ``predict``; ``describe()`` says whether one is in force. The constructor
    raises ``ValueError`` for anything else, and for a calibration that was
    fitted after another action temperature than this predictor's.
    """

    kind = KIND

    def __init__(
        self,
        net: OppNet,
        featurizer: Featurizer | None = None,
        *,
        name: str = KIND,
        action_temperature: float = 1.0,
        target_temperature: float = 1.0,
        elo_mode: str = ELO_KEEP,
        device: torch.device | str = "cpu",
        batch_size: int = 1024,
        mega_temperature: float = 1.0,
        mega_bias: float = 0.0,
        event_calibration: EventCalibration | None = None,
    ) -> None:
        self.net = net
        self.featurizer = featurizer
        self.name = str(name)
        self.action_temperature = _temperature(action_temperature)
        self.target_temperature = _temperature(target_temperature)
        self.mega_temperature = _temperature(mega_temperature)
        self.mega_bias = _offset(mega_bias)
        self.event_calibration = _event_calibration(
            event_calibration, self.action_temperature
        )
        blind = elo_mode == ELO_BLANK or (
            featurizer is not None and featurizer.elo_mode == ELO_BLANK
        )
        self.elo_mode = ELO_BLANK if blind else ELO_KEEP
        self.device = torch.device(device)
        self.batch_size = max(1, int(batch_size))
        self.counters: Counter[str] = Counter()
        self._lock = threading.Lock()

    @property
    def temperatures(self) -> dict[str, float]:
        return {"action": self.action_temperature, "target": self.target_temperature}

    @property
    def mega_calibration(self) -> dict[str, float]:
        """The Mega head's own calibration: ``temperature`` and ``bias``."""
        return {"temperature": self.mega_temperature, "bias": self.mega_bias}

    @property
    def event_calibrated(self) -> bool:
        """Whether ``predict`` rescales the switch / Protect-family events."""
        found = self.event_calibration
        return found is not None and not found.is_identity

    def describe(self) -> dict[str, Any]:
        """Plain metadata: what this predictor applies to the network's logits.

        ``event_calibration`` is None without one, else its
        ``EventCalibration.describe()`` (the maps, ``applied``, the fit record).
        """
        found = self.event_calibration
        return {
            "name": self.name,
            "kind": self.kind,
            "elo_mode": self.elo_mode,
            "temperatures": self.temperatures,
            "mega_calibration": self.mega_calibration,
            "event_calibrated": self.event_calibrated,
            "event_calibration": None if found is None else found.describe(),
        }

    def _twin(
        self,
        action: float,
        target: float,
        mega: float,
        mega_bias: float,
        event_calibration: EventCalibration | None,
        name: str | None = None,
    ) -> "OppNetPredictor":
        twin = OppNetPredictor(
            self.net,
            self.featurizer,
            name=self.name if name is None else name,
            action_temperature=action,
            target_temperature=target,
            elo_mode=self.elo_mode,
            device=self.device,
            batch_size=self.batch_size,
            mega_temperature=mega,
            mega_bias=mega_bias,
            event_calibration=event_calibration,
        )
        twin._lock = self._lock
        return twin

    def with_temperatures(
        self,
        action: float = 1.0,
        target: float = 1.0,
        mega: float = 1.0,
        mega_bias: float = 0.0,
    ) -> "OppNetPredictor":
        """A predictor over the same network and lock, with other temperatures.

        The twin carries NO event calibration: one is fitted after a given
        action temperature and says nothing about another.
        """
        return self._twin(action, target, mega, mega_bias, None)

    def with_event_calibration(
        self, event_calibration: EventCalibration | None, name: str | None = None
    ) -> "OppNetPredictor":
        """A predictor over the same network, lock and temperatures that
        applies ``event_calibration`` (None: none), optionally renamed.

        The calibration REPLACES the one this predictor carries; the two are
        never composed, so a prediction is never calibrated twice. Raises
        ``ValueError`` like the constructor.
        """
        return self._twin(
            self.action_temperature,
            self.target_temperature,
            self.mega_temperature,
            self.mega_bias,
            event_calibration,
            name,
        )

    def _count(self, name: str) -> None:
        try:
            self.counters[name] += 1
        except Exception:
            pass

    def predict(self, batch: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Probabilities ``action`` / ``target`` / ``mega`` (float32). Never raises."""
        try:
            return self._predict(batch)
        except Exception as exc:
            self._count(f"predict_error:{type(exc).__name__}")
            return uniform_prediction(batch)

    def _predict(self, batch: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        flags = np.asarray(batch["game_flag"])
        if (flags[..., [G_ACTOR_SHEET, G_OTHER_SHEET]] == SHEET_UNKNOWN).any():
            self._count("sheet_unknown_as_closed")
            batch = sheet_unknown_as_closed(batch)
        if self.elo_mode == ELO_BLANK:
            batch = apply_elo_mode(batch, ELO_BLANK)
        n = int(np.asarray(batch["action_mask"]).shape[0])
        if n == 0:  # nothing to predict: empty arrays of the right shape
            return {
                name: np.asarray(value, dtype=np.float32)
                for name, value in uniform_prediction(batch).items()
            }
        parts: list[dict[str, np.ndarray]] = []
        with self._lock:
            was_training = self.net.training
            self.net.eval()
            try:
                with torch.no_grad():
                    for start in range(0, n, self.batch_size):
                        index = slice(start, start + self.batch_size)
                        out = self.net(to_tensors(batch, self.device, index))
                        made = probabilities(
                            out,
                            self.action_temperature,
                            self.target_temperature,
                            self.mega_temperature,
                            self.mega_bias,
                        )
                        parts.append(
                            {
                                name: value.to("cpu").numpy().astype(np.float32)
                                for name, value in made.items()
                            }
                        )
            finally:
                self.net.train(was_training)
        if not parts:
            raise ValueError("empty batch")
        result = {
            name: np.concatenate([part[name] for part in parts]) for name in parts[0]
        }
        if not all(np.isfinite(value).all() for value in result.values()):
            raise ValueError("non-finite probability")
        calibration = self.event_calibration
        if calibration is not None and not calibration.is_identity:
            # The one place the event calibration is applied. A failure here
            # (a batch without the context arrays a map's terms need included)
            # is a failed prediction (counted by ``predict``), never a silent
            # fall back to the uncalibrated numbers.
            context = event_context(batch) if calibration.needs_context else None
            result["action"] = calibration.apply_checked(
                result["action"], batch["action_mask"], batch["cand_flag"], context
            ).astype(np.float32)
        return result

    def to_payload(self) -> dict[str, Any]:
        """Plain data and tensors from which ``from_payload`` rebuilds this.

        ``version`` is ``PAYLOAD_VERSION`` (1) without an event calibration,
        so such a payload is what it always was, and
        ``PAYLOAD_VERSION_CALIBRATED`` (2) with one: a reader from before
        event calibration refuses version 2 rather than drop the maps.
        """
        return {
            "format": PAYLOAD_FORMAT,
            "version": PAYLOAD_VERSION
            if self.event_calibration is None
            else PAYLOAD_VERSION_CALIBRATED,
            "name": self.name,
            "config": self.net.config.to_dict(),
            "state_dict": clone_state(self.net),
            "temperatures": {
                **self.temperatures,
                "mega": self.mega_temperature,
                "mega_bias": self.mega_bias,
            },
            "elo_mode": self.elo_mode,
            "n_parameters": self.net.n_parameters(),
            # None (version 1), or ``EventCalibration.to_payload()`` (version
            # 2). Absent in a payload written before event calibration
            # existed: read as None.
            "event_calibration": None
            if self.event_calibration is None
            else self.event_calibration.to_payload(),
        }


def _event_calibration(
    value: Any, action_temperature: float
) -> EventCalibration | None:
    """``value`` as the calibration a predictor may carry. Raises ``ValueError``."""
    if value is None:
        return None
    if not isinstance(value, EventCalibration):
        raise ValueError(f"not an event calibration: {type(value).__name__}")
    if not value.fits_temperature(action_temperature):
        raise ValueError(
            f"the event calibration was fitted after action temperature "
            f"{value.action_temperature!r}; this predictor uses {action_temperature!r}"
        )
    return value


def _temperature(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 1.0
    return number if math.isfinite(number) and number > 0.0 else 1.0


def _offset(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def from_payload(
    payload: Mapping[str, Any],
    featurizer: Featurizer,
    device: torch.device | str = "cpu",
) -> OppNetPredictor:
    """Rebuild a predictor from ``OppNetPredictor.to_payload()``.

    The numeric tables come from ``featurizer`` (rebuilt from the same
    artifact), the weights from the payload. The predictor is Elo-blind when
    the payload or the featurizer says so. A payload without
    ``event_calibration`` (every one written before it existed) gives a
    predictor without one; a stored calibration that is damaged, or that was
    fitted after another action temperature, is an error, never dropped.
    The version says whether a calibration is carried (1: none, 2: one); a
    payload whose version and content disagree is refused.
    Raises ``ValueError`` on a payload that is not one or does not fit the
    featurizer.
    """
    try:
        if payload["format"] != PAYLOAD_FORMAT:
            raise ValueError(f"payload format {payload['format']!r}")
        version = int(payload["version"])
        if version not in PAYLOAD_VERSIONS:
            raise ValueError(f"payload version {payload['version']!r}")
        carried = payload.get("event_calibration") is not None
        if carried != (version == PAYLOAD_VERSION_CALIBRATED):
            raise ValueError(
                f"payload version {version} "
                + (
                    "must not carry an event calibration (a reader of version 1 "
                    "would drop it)"
                    if carried
                    else "must carry an event calibration and holds none"
                )
            )
        config = OppNetConfig.from_dict(payload["config"])
        if config.n_cand != featurizer.n_cand:
            raise ValueError("candidate count differs from the featurizer")
        net = OppNet(config, featurizer.tables.species_num, featurizer.tables.move_num)
        net.load_state_dict(dict(payload["state_dict"]))
        net.to(device)
        net.eval()
        temperatures = payload.get("temperatures") or {}
        stored = payload.get("event_calibration")
        calibration = None if stored is None else EventCalibration.from_payload(stored)
        return OppNetPredictor(
            net,
            featurizer,
            name=str(payload.get("name") or KIND),
            action_temperature=temperatures.get("action", 1.0),
            target_temperature=temperatures.get("target", 1.0),
            elo_mode=str(payload.get("elo_mode") or ELO_KEEP),
            device=device,
            # Absent in a payload written before the Mega head was calibrated.
            mega_temperature=temperatures.get("mega", 1.0),
            mega_bias=temperatures.get("mega_bias", 0.0),
            event_calibration=calibration,
        )
    except (KeyError, TypeError, AttributeError, RuntimeError) as exc:
        raise ValueError(f"not an OppNet payload: {exc!r}") from exc
