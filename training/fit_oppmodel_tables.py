"""Fit the opponent predictor's count tables and write their artifacts.

Design: OPPONENT_PREDICTOR.md ("Models"); the tables are documented in
vgc_bench/src/oppmodel/tables.py. Reads a built dataset, fits SpeciesTable,
FlagsTable and EloTable on the TRAIN split only (weights: m_weight), chooses
their strengths on the VALIDATION split, and scores them on validation, test
(and the test players' games inside the time slice) and the ladder holdout
with the shared scoring code.

What it does, in order:

1. Floors through the same scoring code: uniform over legal actions, and the
   set-key prior with a constant switch rate and uniform targets.
2. SpeciesTable, then FlagsTable: fit with the default strengths, pick each
   strength from a small grid by validation NLL (coordinate descent; fine NLL
   for the action and target strengths, Mega NLL for the Mega strength), refit
   with the chosen values, and repeat once so the censored-slot shares and the
   offset ratios are fitted under the strengths that are kept.
3. EloTable: the chosen FlagsTable configuration plus an Elo offset, for each
   set of band edges and each offset strength; the pair with the lowest
   validation NLL is kept. An infinite strength means "no offset" and is in
   the grid, so "Elo adds nothing" can be the outcome.
4. Ablations of the FlagsTable, each a refit with one thing changed: censored
   slots dropped or spread per slot, uniform switch destinations, no reveal
   offsets, no class offsets, no flags ratio in the prior, no terrain-aware
   spread share, no target context (targets by set key and move alone). Each
   with its NLL and the calibration of P(switch) and P(Protect) over every
   slot where the event is known (censored included).
5. Four artifacts, read back and compared: species_table.pt, flags_table.pt
   (the bar of reading R1), elo_table.pt, and flags_table_plain_targets.pt,
   the same flags table with its target context switched off (the bar as it
   was before that layer, kept so both margins can be stated); then
   fit_report.json and fit_report.md (rendered from the JSON).

Every predict call gets the features only: label (y_*) and meta (m_*) arrays
are stripped first, and unknown sheet codes are mapped to closed.

Validation is the ``val`` split WITHOUT the battles that have a ladder-holdout
opponent (``m_flag`` bit 4): the dataset keeps those players out of training
only, and everything chosen here is chosen on validation, so their other
games must not choose it. The report counts the rows left out.

Differences between two models are paired per slot and their standard error is
clustered by game (m_battle).

WHICH SPLITS ARE SCORED (``evaluation_plan``, decided before any example is
loaded). Nothing is ever fitted or chosen on them; they are scored for the
report. On a dataset of before the fourth build the default is all four, as
it always was: ``val``, ``test``, ``test_time_slice``, ``ladder_holdout``.
Two things are confirmation data of the fourth build (OPPONENT_PREDICTOR.md)
and are NOT read unless asked for by name:

* the bot's own games from ``features.OWN_SEALED_FROM`` on. A dataset that
  holds rows of them (told from the rows' own times, ``features.
  own_time_range``, never from the day it was built) is loaded WITHOUT those
  rows, so its ladder holdout is the old one; ``--allow-sealed-holdout`` reads
  them;
* the ``test`` split (and its time slice) of a dataset of the fourth build:
  one that holds such rows, or one that was built with ``--own-before`` (its
  manifest's ``own_cutoff``; that option exists only since the fourth build).
  There the two are not loaded and not scored; ``--score-test`` scores them.

``--eval-splits`` chooses among the four (``val`` is always needed: the
strengths are chosen on it). Naming ``test`` for a dataset of the fourth build
without ``--score-test`` is refused, never quietly dropped. The report and
every artifact's ``extra`` hold only the splits that were scored, and the
report says what was left out and why. The tables themselves do not depend on
any of this: they are fitted on ``train`` and tuned on ``val``.

A directory that already holds a fit is not written over without
``--overwrite``, and ``--limit`` (a smoke) needs its own ``--out``. The last
line printed is ``FIT_DONE`` (exit code 0) or ``FIT_FAILED <reason>`` (exit
code 1).

    nice -n 19 .venv/bin/python training/fit_oppmodel_tables.py \\
        --dataset results_oppmodel/v1_ondisk --out results_oppmodel/tables_v1
    nice -n 19 .venv/bin/python training/fit_oppmodel_tables.py \\
        --out results_oppmodel/tables_v1 --render-only
"""

import os as _os
import sys as _sys
from pathlib import Path as _Path

_os.environ.setdefault("OMP_NUM_THREADS", "1")
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from vgc_bench.src.oppmodel import tables as T
from vgc_bench.src.oppmodel.artifact import KIND_TABLE, load_predictor, save_artifact
from vgc_bench.src.oppmodel.events import INTENT_PROTECT, INTENT_SWITCH
from vgc_bench.src.oppmodel.features import (
    CAND_VALID,
    ELO_SHUFFLE,
    N_INTENT,
    OWN_SEALED_FROM,
    Y_PROTECT_KNOWN,
    Y_PROTECTED,
    Y_SWITCH_KNOWN,
    Y_SWITCHED,
    Batch,
    Featurizer,
    apply_elo_mode,
    event_probs,
    fine_label,
    fine_probs,
    intent_probs,
    load_dataset,
    local_time,
    normalize_prediction,
    own_time_range,
    sheet_unknown_as_closed,
    slot_nll,
    take,
    topk_hits,
    uniform_prediction,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN = "train"
VALIDATION = "val"
TEST = "test"
TIME_SLICE = "test_time_slice"  # the test players' games in the latest 10%
FLAG_TIME_SLICE = 1  # m_flag bits, as the dataset builder documents them
FLAG_HOLDOUT_BATTLE = 4  # the battle has a ladder-holdout opponent
LADDER = "ladder_holdout"
EVAL_SPLITS: tuple[str, ...] = ("val", TEST, TIME_SLICE, LADDER)
# The two evaluation splits that are rows of the dataset's ``test`` split.
TEST_SPLITS: tuple[str, ...] = (TEST, TIME_SLICE)
REPORT_JSON = "fit_report.json"
REPORT_MD = "fit_report.md"
DEFAULT_OUT = "results_oppmodel/tables_v1"
DONE = "FIT_DONE"
FAILED = "FIT_FAILED"
# The flags table with its target context switched off: the same counts and
# strengths, targets keyed by set key and move alone (the bar before the layer).
TABLE_FLAGS_PLAIN = "flags_table_plain_targets"
ARTIFACTS = {
    T.TABLE_SPECIES: "species_table.pt",
    T.TABLE_FLAGS: "flags_table.pt",
    T.TABLE_ELO: "elo_table.pt",
    TABLE_FLAGS_PLAIN: "flags_table_plain_targets.pt",
}
INF = math.inf

# Strength grids. Every value is a pseudo-count unless its name says otherwise.
KEY_GRID: dict[str, tuple[float, ...]] = {
    "key_strength": (3.0, 10.0, 30.0, 100.0, 300.0, 1000.0),
    "switch_key_strength": (10.0, 30.0, 100.0, 300.0, 1000.0),
    "other_boost": (1.0, 1.5, 2.0, 3.0, 4.0),
    "floor": (1e-6, 1e-5, 1e-4, 1e-3, 3e-3),
    "target_key_strength": (3.0, 10.0, 30.0, 100.0, 300.0, INF),
    "target_move_strength": (1.0, 5.0, 20.0, 100.0),
}
CELL_GRID: dict[str, tuple[float, ...]] = {
    "cell_strength": (3.0, 10.0, 30.0, 100.0, 300.0, 1000.0),
    "switch_cell_strength": (10.0, 30.0, 100.0, 300.0, 1000.0),
    "flag_effect_strength": (1.0, 5.0, 20.0, 100.0, INF),
}
# One more strength per way of counting moves (tables.USAGE_*).
USAGE_GRID: dict[str, dict[str, tuple[float, ...]]] = {
    T.USAGE_RATE: {"move_strength": (1.0, 10.0, 100.0, 1000.0)},
    T.USAGE_SHARE: {"unseen_count": (0.1, 0.5, 2.0, 10.0)},
}
MEGA_GRID: dict[str, tuple[float, ...]] = {
    "mega_strength": (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
}
ELO_EDGE_SETS: tuple[tuple[int, ...], ...] = (
    (1300,),
    (1200, 1400),
    (1100, 1200, 1300, 1400, 1500),
    (1000, 1100, 1200, 1300, 1400, 1500, 1600),
)
# Finite on purpose: the EloTable that is written always uses the rating, so
# its difference to the FlagsTable is a measurement. "No offset" is the
# FlagsTable itself and is reported next to the grid.
ELO_STRENGTHS: tuple[float, ...] = (5.0, 50.0, 500.0, 5000.0)
ABLATIONS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("censored slots dropped", {"censored": T.CENSORED_DROP}),
    ("censored slots spread per slot", {"censored": T.CENSORED_SLOT}),
    ("uniform switch destinations", {"switch_dest": T.DEST_UNIFORM}),
    ("no reveal offsets", {"reveal_offsets": False}),
    ("no class offsets", {"class_offsets": False}),
    ("no offsets at all", {"reveal_offsets": False, "class_offsets": False}),
    ("no flags ratio in the prior", {"flag_effect_strength": INF}),
    ("no terrain-aware spread share", {"use_cand_auto": False}),
    ("no target context: targets by set key and move alone", {"target_context": False}),
)
# Refitted AND retuned, so the comparison does not lean on strengths that were
# chosen for another model: move shares in place of usage rates, and the plain
# table (move shares, no reveal offsets, no class offsets).
RETUNED: tuple[tuple[str, dict[str, Any]], ...] = (
    ("move shares, not rates (strengths retuned)", {"usage": T.USAGE_SHARE}),
    (
        "the plain table: shares, no offsets (strengths retuned)",
        {"usage": T.USAGE_SHARE, "reveal_offsets": False, "class_offsets": False},
    ),
)
CALIBRATION_EDGES: tuple[float, ...] = (0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0)
EVENTS = (
    (INTENT_SWITCH, Y_SWITCH_KNOWN, Y_SWITCHED),
    (INTENT_PROTECT, Y_PROTECT_KNOWN, Y_PROTECTED),
)

Log = Callable[[str], None]


class FitError(RuntimeError):
    """The run cannot start: an output it would write over, or a bad input."""


def say(text: str) -> None:
    """Print a progress line at once (stdout may be a file)."""
    print(text, flush=True)


# --- scoring ------------------------------------------------------------------


def features_only(batch: Mapping[str, np.ndarray]) -> Batch:
    """A batch without its label (y_*) and meta (m_*) arrays."""
    return {
        name: array
        for name, array in batch.items()
        if not name.startswith(("y_", "m_"))
    }


def predict(model: Any, batch: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """``model.predict`` on the features alone, unknown sheets read as closed."""
    return model.predict(sheet_unknown_as_closed(features_only(batch)))


def _mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    return float(values[mask].mean()) if mask.any() else None


def score(
    pred: Mapping[str, np.ndarray], batch: Mapping[str, np.ndarray], tables: Any = None
) -> dict[str, Any]:
    """NLLs, top-k and intent accuracy of one prediction on one split."""
    nll = slot_nll(pred, batch)
    scored = nll["fine_scored"]
    censored = nll["censored"]
    out: dict[str, Any] = {
        "slots_scored": int(scored.sum()),
        "slots_visible": int((scored & ~censored).sum()),
        "slots_censored": int(censored.sum()),
        "fine_nll": _mean(nll["fine"], scored),
        "action_nll": _mean(nll["action"], scored),
        "action_nll_visible": _mean(nll["action"], scored & ~censored),
        "action_nll_censored": _mean(nll["action"], censored),
        "target_nll": _mean(nll["target"], nll["target_scored"]),
        "target_labels": int(nll["target_scored"].sum()),
        "mega_nll": _mean(nll["mega"], nll["mega_scored"]),
        "mega_labels": int(nll["mega_scored"].sum()),
    }
    norm = normalize_prediction(pred, batch)
    y_action = np.asarray(batch["y_action"])
    joint, joint_label = fine_probs(pred, batch), fine_label(batch)
    for k in (1, 3):
        hits, counted = topk_hits(norm["action"], y_action, k)
        out[f"action_top{k}"] = _mean(hits.astype(np.float64), counted)
        hits, counted = topk_hits(joint, joint_label, k)
        out[f"fine_top{k}"] = _mean(hits.astype(np.float64), counted)
    if tables is not None:
        intents = intent_probs(pred, batch, tables)
        y_intent = np.asarray(batch["y_intent"]).astype(np.int64)
        if intents is not None:
            known = y_intent >= 0
            guess = intents[..., :N_INTENT].argmax(-1)
            out["intent_top1"] = _mean((guess == y_intent).astype(np.float64), known)
            out["intent_labels"] = int(known.sum())
    return out


def calibration(
    pred: Mapping[str, np.ndarray], batch: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """P(switch) and P(Protect) against what happened, censored slots included.

    The denominator is every active slot where the log tells whether the event
    happened (``y_flag`` switch known / Protect known), whether or not the
    slot's action was visible.
    """
    events = event_probs(pred, batch)
    flags = np.asarray(batch["y_flag"])
    active = np.asarray(batch["act_mon"]) >= 0
    out: dict[str, Any] = {}
    if not events:
        return out
    for name, known_column, did_column in EVENTS:
        known = active & (flags[..., known_column] == 1)
        p = np.asarray(events[name], dtype=np.float64)[known]
        y = (flags[..., did_column] == 1)[known].astype(np.float64)
        bins: list[dict[str, Any]] = []
        for low, high in zip(CALIBRATION_EDGES[:-1], CALIBRATION_EDGES[1:]):
            inside = (p >= low) & ((p < high) | (high >= 1.0))
            if inside.any():
                bins.append(
                    {
                        "from": low,
                        "to": high,
                        "slots": int(inside.sum()),
                        "predicted": float(p[inside].mean()),
                        "observed": float(y[inside].mean()),
                    }
                )
        out[name] = {
            "slots": int(known.sum()),
            "predicted": float(p.mean()) if p.size else None,
            "observed": float(y.mean()) if y.size else None,
            "bins": bins,
        }
    return out


def paired_difference(
    first: np.ndarray, second: np.ndarray, scored: np.ndarray, battle: np.ndarray
) -> dict[str, Any]:
    """Mean of ``first - second`` over scored slots, standard error by game.

    The ratio estimator's linearised variance with games as clusters; the
    interval is the mean -/+ 1.96 standard errors.
    """
    games = np.broadcast_to(np.asarray(battle)[:, None], scored.shape)[scored]
    delta = (np.asarray(first) - np.asarray(second))[scored]
    n = int(delta.size)
    if n == 0:
        return {"slots": 0, "games": 0, "diff": None, "se": None}
    _, index = np.unique(games, return_inverse=True)
    n_games = int(index.max()) + 1
    total = np.bincount(index, weights=delta, minlength=n_games)
    size = np.bincount(index, minlength=n_games).astype(np.float64)
    mean = float(delta.mean())
    if n_games < 2:
        return {"slots": n, "games": n_games, "diff": mean, "se": None}
    variance = float(((total - mean * size) ** 2).sum()) * n_games / (n_games - 1)
    se = math.sqrt(variance) / n
    return {
        "slots": n,
        "games": n_games,
        "diff": mean,
        "se": se,
        "low": mean - 1.96 * se,
        "high": mean + 1.96 * se,
    }


def compare(
    first: Mapping[str, np.ndarray],
    second: Mapping[str, np.ndarray],
    batch: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Paired fine-NLL difference ``first - second`` on one split (below 0: better)."""
    a, b = slot_nll(first, batch), slot_nll(second, batch)
    return paired_difference(
        a["fine"], b["fine"], a["fine_scored"], np.asarray(batch["m_battle"])
    )


# --- floors -------------------------------------------------------------------


def train_switch_rate(train: Mapping[str, np.ndarray]) -> float:
    """Weighted share of switches among train slots whose switch state is known."""
    flags = np.asarray(train["y_flag"])
    weight = np.asarray(train["m_weight"], dtype=np.float64)[:, None]
    known = flags[..., Y_SWITCH_KNOWN] == 1
    switched = flags[..., Y_SWITCHED] == 1
    return float((weight * (known & switched)).sum() / (weight * known).sum())


def prior_floor_prediction(
    batch: Mapping[str, np.ndarray], switch_rate: float
) -> dict[str, np.ndarray]:
    """The dataset builder's floor: set-key prior, one switch rate, uniform targets.

    Uses only what the featurizer already put in the batch (``cand_prior`` and
    ``other_prior`` from the training repertoire).
    """
    mask = np.asarray(batch["action_mask"], dtype=np.float64)
    valid = (np.asarray(batch["cand_flag"]).astype(np.int64) & CAND_VALID) > 0
    prior = np.asarray(batch["cand_prior"], dtype=np.float64) + 1e-3 * valid
    other = np.asarray(batch["other_prior"], dtype=np.float64)[..., None] + 1e-3
    moves = np.concatenate([prior, other], -1)
    moves = moves / moves.sum(-1, keepdims=True)
    pointers = np.asarray(batch["switch_mask"], dtype=np.float64)
    count = pointers.sum(-1, keepdims=True)
    rate = np.where(count > 0, switch_rate, 0.0)
    spread = np.where(count > 0, pointers * rate / np.maximum(count, 1.0), 0.0)
    out = uniform_prediction(batch)
    out["action"] = np.concatenate([moves * (1.0 - rate), spread], -1) * mask
    out["mega"] = np.full(mask.shape[:2], 0.2)
    return out


# --- tuning -------------------------------------------------------------------


def objective(
    table: Any, batch: Mapping[str, np.ndarray], metric: str = "fine"
) -> float:
    """Mean validation NLL of one head: ``fine`` (action + target) or ``mega``."""
    nll = slot_nll(predict(table, batch), batch)
    scored = nll[f"{metric}_scored"]
    return float(nll[metric][scored].mean()) if scored.any() else 0.0


def tune(
    table: Any,
    batch: Mapping[str, np.ndarray],
    grid: Mapping[str, Sequence[float]],
    metric: str = "fine",
    sweeps: int = 2,
) -> tuple[Any, list[dict[str, Any]]]:
    """Coordinate descent over ``grid``; counts stay as they are.

    Returns the table under the best configuration found and one record per
    strength per sweep (every value tried with its NLL).
    """
    best = objective(table, batch, metric)
    history: list[dict[str, Any]] = []
    for sweep in range(sweeps):
        changed = False
        for name, values in grid.items():
            tried: dict[str, float] = {}
            for value in values:
                if value == getattr(table.config, name):
                    tried[_text(value)] = best
                    continue
                candidate = table.with_config(**{name: value})
                found = objective(candidate, batch, metric)
                tried[_text(value)] = found
                if found < best - 1e-7:
                    best, table, changed = found, candidate, True
            history.append(
                {
                    "sweep": sweep,
                    "metric": metric,
                    "name": name,
                    "chosen": _plain(getattr(table.config, name)),
                    "nll": tried,
                }
            )
        if not changed:
            break
    return table, history


def grid_for(config: T.TableConfig, cells: bool) -> dict[str, tuple[float, ...]]:
    """The strengths worth tuning for a configuration (flags cells or not)."""
    grid = {**KEY_GRID, **USAGE_GRID[config.usage]}
    return {**grid, **CELL_GRID} if cells else grid


def fit_tuned(
    cls: Any,
    data: Mapping[str, np.ndarray],
    train_mask: np.ndarray,
    validation: Mapping[str, np.ndarray],
    featurizer: Featurizer,
    config: T.TableConfig,
    grid: Mapping[str, Sequence[float]],
    rounds: int,
    log: Log,
) -> tuple[Any, list[dict[str, Any]]]:
    """Fit, tune on validation, refit with the tuned strengths; ``rounds`` times."""
    history: list[dict[str, Any]] = []
    for round_index in range(rounds):
        table = cls.fit(data, mask=train_mask, config=config, featurizer=featurizer)
        before = objective(table, validation)
        table, steps = tune(table, validation, grid, "fine")
        table, mega_steps = tune(table, validation, MEGA_GRID, "mega", sweeps=1)
        for step in steps + mega_steps:
            history.append({"refit": round_index, **step})
        config = table.config
        log(
            f"  {cls.table} round {round_index}: validation fine NLL "
            f"{before:.4f} -> {objective(table, validation):.4f}"
        )
    final = cls.fit(data, mask=train_mask, config=config, featurizer=featurizer)
    return final, history


# --- the run ------------------------------------------------------------------


def _plain(value: Any) -> Any:
    """JSON-safe copy: infinities as text, numpy scalars and tuples unpacked."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else ("-inf" if value < 0 else "nan")
    return value


def _text(value: Any) -> str:
    return str(_plain(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(
    model: Any, parts: Mapping[str, Mapping[str, np.ndarray]], tables: Any
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """(scores, calibration, predictions) of one model on every evaluation split."""
    scores: dict[str, Any] = {}
    reliability: dict[str, Any] = {}
    preds: dict[str, dict[str, np.ndarray]] = {}
    for name, batch in parts.items():
        preds[name] = predict(model, batch)
        scores[name] = score(preds[name], batch, tables)
        reliability[name] = calibration(preds[name], batch)
    return scores, reliability, preds


def _brief(calibrated: Mapping[str, Any]) -> dict[str, Any]:
    """Mean predicted and observed rate of each event, without the bins."""
    return {
        name: {key: entry.get(key) for key in ("slots", "predicted", "observed")}
        for name, entry in calibrated.items()
    }


def existing_outputs(out: Path) -> list[Path]:
    """The files of an earlier fit that a run into ``out`` would replace."""
    names = [*ARTIFACTS.values(), REPORT_JSON]
    return [out / name for name in names if (out / name).exists()]


def split_masks(
    data: Mapping[str, np.ndarray], names: Sequence[str]
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    """(train rows, rows of every evaluation split, validation rows left out).

    Train is the ``train`` split and nothing else. Validation is ``val``
    without the battles that have a ladder-holdout opponent; the time slice is
    the flagged part of ``test``.
    """
    split = np.asarray(data["m_split"])
    flags = np.asarray(data["m_flag"]).astype(np.int64)
    late = (flags & FLAG_TIME_SLICE) > 0
    holdout = (flags & FLAG_HOLDOUT_BATTLE) > 0
    val = split == names.index(VALIDATION)
    masks = {
        VALIDATION: val & ~holdout,
        TEST: split == names.index(TEST),
        TIME_SLICE: (split == names.index(TEST)) & late,
    }
    for name in EVAL_SPLITS:
        if name not in masks:
            masks[name] = split == names.index(name)
    return split == names.index(TRAIN), masks, int((val & holdout).sum())


def parse_splits(text: str | Sequence[str] | None) -> list[str] | None:
    """``--eval-splits`` as a list in ``EVAL_SPLITS`` order; None = not given.

    Raises ``FitError`` for a name that is not an evaluation split, an empty
    list, or a list without ``val`` (the strengths are chosen on it).
    """
    if text is None:
        return None
    parts = text.split(",") if isinstance(text, str) else list(text)
    asked = [str(part).strip() for part in parts if str(part).strip()]
    unknown = [name for name in asked if name not in EVAL_SPLITS]
    if unknown or not asked:
        raise FitError(f"--eval-splits {text!r}: choose from {EVAL_SPLITS}")
    if VALIDATION not in asked:
        raise FitError(
            f"--eval-splits {text!r}: {VALIDATION} is always needed (the "
            "strengths are chosen on it)"
        )
    return [name for name in EVAL_SPLITS if name in asked]


def evaluation_plan(
    dataset: Path,
    eval_splits: str | Sequence[str] | None = None,
    *,
    score_test: bool = False,
    allow_sealed_holdout: bool = False,
) -> dict[str, Any]:
    """Which splits a run scores and which rows it never loads, decided from
    the manifest and the meta arrays ``m_split`` / ``m_time`` alone.

    ``splits``: the evaluation splits scored, in ``EVAL_SPLITS`` order.
    ``not_scored``: the others, each with its reason. ``own_before``: the
    Unix time from which the bot's own games are left out of everything that
    is loaded (None: nothing is left out). ``load_test``: whether the
    dataset's ``test`` split is loaded at all. ``default``: nothing departs
    from the script's behaviour of before (all four splits, every row).

    A dataset is ``fourth_build`` when it holds own games dated from
    ``features.OWN_SEALED_FROM`` on, or was built with ``--own-before`` (its
    manifest's ``own_cutoff``): there the test split is scored only with
    ``score_test`` and the sealed rows are read only with
    ``allow_sealed_holdout`` (see the module text). Raises ``FitError`` for a
    dataset that cannot be read, a bad list, or ``test`` named for such a
    dataset without ``score_test``.
    """
    asked = parse_splits(eval_splits)
    try:
        text = (dataset / "manifest.json").read_text(encoding="utf-8")
        manifest = json.loads(text)
        sealed_from = local_time(OWN_SEALED_FROM)
        held = own_time_range(dataset, sealed_from)
    except (OSError, ValueError, KeyError) as exc:
        raise FitError(
            f"dataset {dataset}: its splits and the times of its own games "
            f"cannot be read ({exc!r})"
        ) from exc
    sealed = int(held.get("late_rows") or 0)
    cutoff = manifest.get("own_cutoff") if isinstance(manifest, Mapping) else None
    built_with_cutoff = isinstance(cutoff, Mapping)
    fourth_build = bool(sealed) or built_with_cutoff
    chosen = list(EVAL_SPLITS) if asked is None else list(asked)
    not_scored: dict[str, str] = {
        name: "not in --eval-splits" for name in EVAL_SPLITS if name not in chosen
    }
    if fourth_build and not score_test:
        why = (
            f"the dataset holds own games from {OWN_SEALED_FROM} on"
            if sealed
            else "the dataset was built with --own-before"
        ) + ": its test split is confirmation data (give --score-test to score it)"
        named = [name for name in TEST_SPLITS if name in chosen]
        if asked is not None and named:
            raise FitError(f"--eval-splits names {named}, and {why}")
        for name in named:
            not_scored[name] = why
        chosen = [name for name in chosen if name not in TEST_SPLITS]
    leave_out = bool(sealed) and not allow_sealed_holdout
    load_test = any(name in chosen for name in TEST_SPLITS)
    return {
        "splits": chosen,
        "not_scored": not_scored,
        "asked": asked,
        "score_test": bool(score_test),
        "allow_sealed_holdout": bool(allow_sealed_holdout),
        "fourth_build": fourth_build,
        "built_with_own_cutoff": built_with_cutoff,
        "sealed_from": OWN_SEALED_FROM,
        "sealed_rows_in_dataset": sealed,
        "sealed_rows_read": bool(sealed) and not leave_out,
        "own_before": sealed_from if leave_out else None,
        "own_time_known": bool(held.get("has_time", True)),
        "own_time": held.get("splits") or {},
        "load_test": load_test,
        "default": len(chosen) == len(EVAL_SPLITS) and not leave_out,
    }


def load_planned(
    dataset: Path, plan: Mapping[str, Any]
) -> tuple[Batch, dict[str, Any], list[str]]:
    """(the examples a plan allows, the manifest, the splits not loaded).

    The default plan loads the whole dataset, as this script always did. Any
    other plan loads without the ``test`` split when neither of its two
    evaluation splits is scored, and without the bot's own games from
    ``plan['own_before']`` on: what is not to be scored is not read.
    """
    if plan["default"]:
        data, manifest = load_dataset(dataset)
        return data, manifest, []
    text = (dataset / "manifest.json").read_text(encoding="utf-8")
    names = list(json.loads(text)["splits"])
    dropped = [] if plan["load_test"] else [name for name in names if name == TEST]
    wanted = [name for name in names if name not in dropped]
    if plan["own_before"] is None:
        data, manifest = load_dataset(dataset, splits=wanted)
    else:
        data, manifest = load_dataset(
            dataset, splits=wanted, own_before=int(plan["own_before"])
        )
    return data, manifest, dropped


def scored_splits(report: Mapping[str, Any]) -> list[str]:
    """The evaluation splits a report holds (a report of before the choice
    existed holds all four)."""
    found = report.get("evaluation_splits")
    return list(found) if found else list(EVAL_SPLITS)


def run(
    dataset: Path,
    out: Path,
    *,
    limit: int | None = None,
    overwrite: bool = False,
    log: Log = say,
    eval_splits: str | Sequence[str] | None = None,
    score_test: bool = False,
    allow_sealed_holdout: bool = False,
) -> dict[str, Any]:
    """Fit, tune, score, write artifacts; returns the report (also written).

    ``eval_splits`` / ``score_test`` / ``allow_sealed_holdout``: which splits
    are scored and whether the confirmation data is read (``evaluation_plan``;
    the defaults score everything an old dataset holds and neither the test
    split nor the sealed own games of a dataset of the fourth build).

    Raises ``FitError`` when ``out`` already holds a fit and ``overwrite`` is
    not set; nothing is read or written then.
    """
    found = existing_outputs(out)
    if found and not overwrite:
        raise FitError(
            f"{found[0]} exists; give --overwrite to replace the fit in {out}"
        )
    started = time.time()
    plan = evaluation_plan(
        dataset,
        eval_splits,
        score_test=score_test,
        allow_sealed_holdout=allow_sealed_holdout,
    )
    chosen: list[str] = list(plan["splits"])
    for name, why in plan["not_scored"].items():
        log(f"evaluation split {name} is NOT scored: {why}")
    if plan["own_before"] is not None:
        log(
            f"{plan['sealed_rows_in_dataset']} rows of own games from "
            f"{OWN_SEALED_FROM} on (the sealed set) are left out (give "
            "--allow-sealed-holdout to read them)"
        )
    data, manifest, not_loaded = load_planned(dataset, plan)
    data = sheet_unknown_as_closed(data)
    featurizer = Featurizer.load(dataset)
    names = list(manifest["splits"])
    split = np.asarray(data["m_split"])
    if limit is not None:
        keep = np.zeros(split.shape, dtype=bool)
        for code in np.unique(split):
            keep[np.flatnonzero(split == code)[:limit]] = True
        data = take(data, keep)
        split = np.asarray(data["m_split"])
    train_mask, masks, left_out = split_masks(data, names)
    parts = {name: take(data, masks[name]) for name in chosen}
    validation = parts[VALIDATION]
    tables = featurizer.tables
    log(
        f"dataset {dataset}: {len(split)} examples, train {int(train_mask.sum())}, "
        + ", ".join(f"{name} {len(part['turn'])}" for name, part in parts.items())
        + f" (validation leaves out {left_out} rows of battles with a "
        "ladder-holdout opponent)"
    )
    if not train_mask.any() or not len(validation["turn"]):
        raise FitError(
            f"dataset {dataset}: train {int(train_mask.sum())} examples, "
            f"validation {len(validation['turn'])}; both are needed"
        )

    report: dict[str, Any] = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": {
            "directory": str(dataset),
            "tag": manifest.get("tag"),
            "manifest_sha256": sha256_file(dataset / "manifest.json"),
            "examples": {name: int((split == i).sum()) for i, name in enumerate(names)},
            "limit_per_split": limit,
            "dex_signature_diff": list(featurizer.signature_diff),
            "formats": list(manifest.get("formats") or []),
            "fit_examples": int(train_mask.sum()),
            "evaluation_examples": {
                name: int(len(part["turn"])) for name, part in parts.items()
            },
            "validation_rows_left_out_holdout_battles": left_out,
        },
        "metric": "fine NLL = features.slot_nll 'fine' over 'fine_scored', nats per "
        "labelled slot-turn, after sheet_unknown_as_closed; lower is better",
    }
    if not plan["default"] or plan["fourth_build"]:
        # A run on an old dataset with the default plan writes the report of
        # before, key for key; anything else says what was scored and why.
        report["evaluation_splits"] = chosen
        report["evaluation_plan"] = _plain(plan)
        report["dataset"]["splits_not_loaded"] = not_loaded

    # 1. floors
    rate = train_switch_rate(take(data, train_mask))
    floors: dict[str, Any] = {"train_switch_rate": rate, "uniform": {}, "prior": {}}
    for name, batch in parts.items():
        clean = sheet_unknown_as_closed(features_only(batch))
        floors["uniform"][name] = score(uniform_prediction(clean), batch, tables)
        floors["prior"][name] = score(
            prior_floor_prediction(clean, rate), batch, tables
        )
    report["floors"] = floors
    log(
        "floors (fine NLL): uniform "
        + _fine(floors["uniform"], 3)
        + "; set-key prior "
        + _fine(floors["prior"], 3)
    )

    # 2. species and flags tables
    fitted: dict[str, Any] = {}
    tuning: dict[str, Any] = {}
    fitted[T.TABLE_SPECIES], tuning[T.TABLE_SPECIES] = fit_tuned(
        T.SpeciesTable,
        data,
        train_mask,
        validation,
        featurizer,
        T.SpeciesTable.default_config(),
        grid_for(T.SpeciesTable.default_config(), cells=False),
        2,
        log,
    )
    fitted[T.TABLE_FLAGS], tuning[T.TABLE_FLAGS] = fit_tuned(
        T.FlagsTable,
        data,
        train_mask,
        validation,
        featurizer,
        fitted[T.TABLE_SPECIES].config,
        grid_for(fitted[T.TABLE_SPECIES].config, cells=True),
        2,
        log,
    )
    flags = fitted[T.TABLE_FLAGS]

    # 3. Elo offset
    elo_grid: list[dict[str, Any]] = []
    best_elo: tuple[float, Any] | None = None
    for edges in ELO_EDGE_SETS:
        base = T.EloTable.fit(
            data,
            mask=train_mask,
            config=replace(flags.config, elo_edges=edges, elo_strength=INF),
            featurizer=featurizer,
        )
        for strength in ELO_STRENGTHS:
            candidate = base.with_config(elo_strength=strength)
            found = objective(candidate, validation)
            elo_grid.append(
                {
                    "edges": list(edges),
                    "elo_strength": _plain(strength),
                    "validation_fine_nll": found,
                }
            )
            if best_elo is None or found < best_elo[0] - 1e-9:
                best_elo = (found, candidate)
    if best_elo is None:
        raise ValueError("the Elo grid is empty")
    fitted[T.TABLE_ELO] = best_elo[1]
    without = objective(flags, validation)
    tuning[T.TABLE_ELO] = {
        "grid": elo_grid,
        "validation_fine_nll_without_offset": without,
        "validation_prefers_no_offset": bool(best_elo[0] >= without),
    }
    log(
        f"  elo_table: chose edges {list(best_elo[1].config.elo_edges)} strength "
        f"{_text(best_elo[1].config.elo_strength)} (validation {best_elo[0]:.5f}; "
        f"without the offset {without:.5f})"
    )

    # scores of the three tables
    preds: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    report["tables"] = {}
    for label, table in fitted.items():
        scores, reliability, preds[label] = evaluate(table, parts, tables)
        report["tables"][label] = {
            "config": _plain(table.config.to_dict()),
            "fit_info": _plain(table.fit_info),
            "scores": scores,
            "calibration": reliability,
            "tuning": _plain(tuning[label]),
        }
        log(f"{label}: fine NLL " + _fine(scores))
    report["tables"][T.TABLE_FLAGS]["reveal_ratio"] = flags.reveal_ratio.tolist()
    report["tables"][T.TABLE_FLAGS]["class_ratio"] = flags.class_ratio.tolist()
    report["tables"][T.TABLE_FLAGS]["target_ratio"] = {
        "cells": "damaging move: effectiveness against this foe (below 1, 1, "
        "above 1) x against the other foe x this foe's HP (lower, about the "
        "same, higher); then the HP relation alone for any other move",
        "damaging": flags.target_ratio[: T._STATUS_CELL]
        .reshape(T.N_EFFECT, T.N_EFFECT, T.N_HP_RELATION)
        .tolist(),
        "other_moves": flags.target_ratio[T._STATUS_CELL :].tolist(),
    }
    # The same table with the target context switched off: no refit, no tuning.
    fitted[TABLE_FLAGS_PLAIN] = type(flags)(
        **{
            **flags.with_config(target_context=False)._state(),
            "name": TABLE_FLAGS_PLAIN,
        },
        config=replace(flags.config, target_context=False),
    )
    scores, reliability, preds[TABLE_FLAGS_PLAIN] = evaluate(
        fitted[TABLE_FLAGS_PLAIN], parts, tables
    )
    report["tables"][TABLE_FLAGS_PLAIN] = {
        "config": _plain(fitted[TABLE_FLAGS_PLAIN].config.to_dict()),
        "fit_info": _plain(fitted[TABLE_FLAGS_PLAIN].fit_info),
        "scores": scores,
        "calibration": reliability,
        "tuning": {"note": "the flags table's counts and strengths; no tuning"},
    }
    log(f"{TABLE_FLAGS_PLAIN}: fine NLL " + _fine(scores))
    report["tables"][T.TABLE_ELO]["elo_ratio"] = {
        "edges": [int(edge) for edge in fitted[T.TABLE_ELO].elo_edges],
        "classes": list(T.CLASS_NAMES),
        "observed": fitted[T.TABLE_ELO].elo_observed.tolist(),
        "expected": fitted[T.TABLE_ELO].elo_expected.tolist(),
    }

    comparisons: dict[str, Any] = {
        "flags_minus_species": {},
        "flags_minus_flags_plain_targets": {},
        "elo_minus_flags": {},
        "elo_shuffled_minus_flags": {},
        "elo_fixed_minus_flags": {},
    }
    fixed_elo = T.EloTable.fit(
        data,
        mask=train_mask,
        config=replace(flags.config, elo_edges=ELO_EDGE_SETS[2], elo_strength=50.0),
        featurizer=featurizer,
    )
    for name, batch in parts.items():
        comparisons["flags_minus_species"][name] = compare(
            preds[T.TABLE_FLAGS][name], preds[T.TABLE_SPECIES][name], batch
        )
        comparisons["flags_minus_flags_plain_targets"][name] = compare(
            preds[T.TABLE_FLAGS][name], preds[TABLE_FLAGS_PLAIN][name], batch
        )
        comparisons["elo_minus_flags"][name] = compare(
            preds[T.TABLE_ELO][name], preds[T.TABLE_FLAGS][name], batch
        )
        shuffled = apply_elo_mode(batch, ELO_SHUFFLE, np.random.default_rng(0))
        comparisons["elo_shuffled_minus_flags"][name] = compare(
            predict(fitted[T.TABLE_ELO], shuffled), preds[T.TABLE_FLAGS][name], batch
        )
        comparisons["elo_fixed_minus_flags"][name] = compare(
            predict(fixed_elo, batch), preds[T.TABLE_FLAGS][name], batch
        )
    comparisons["elo_fixed_setting"] = {
        "edges": list(ELO_EDGE_SETS[2]),
        "elo_strength": 50.0,
        "why": "a fixed, not validation-selected, Elo offset for reference",
    }
    report["comparisons"] = comparisons

    # 4. ablations of the flags table
    ablations: list[dict[str, Any]] = []
    for label, changes in (*ABLATIONS, *RETUNED):
        config = replace(flags.config, **changes)
        if (label, changes) in RETUNED:
            variant, _ = fit_tuned(
                T.FlagsTable,
                data,
                train_mask,
                validation,
                featurizer,
                config,
                grid_for(config, cells=True),
                2,
                log,
            )
        else:
            variant = T.FlagsTable.fit(
                data, mask=train_mask, config=config, featurizer=featurizer
            )
        row: dict[str, Any] = {
            "name": label,
            "changes": _plain(changes),
            "config": _plain(variant.config.to_dict()),
            "splits": {},
        }
        for name, batch in parts.items():
            pred = predict(variant, batch)
            got = score(pred, batch)
            row["splits"][name] = {
                "fine_nll": got["fine_nll"],
                "action_nll_visible": got["action_nll_visible"],
                "action_nll_censored": got["action_nll_censored"],
                "target_nll": got["target_nll"],
                "calibration": _brief(calibration(pred, batch)),
                "minus_flags_table": compare(pred, preds[T.TABLE_FLAGS][name], batch),
            }
        ablations.append(row)
        log(
            f"  ablation {label}: validation fine NLL "
            f"{_number(row['splits'][VALIDATION]['fine_nll'])}"
        )
    report["ablations"] = ablations

    # 5. artifacts
    out.mkdir(parents=True, exist_ok=True)
    report["artifacts"] = {}
    for label, table in fitted.items():
        path = out / ARTIFACTS[label]
        entry = report["tables"][label]
        save_artifact(
            path,
            kind=KIND_TABLE,
            name=table.name,
            featurizer=featurizer,
            predictor_payload=table.to_payload(),
            extra={
                "dataset_tag": manifest.get("tag"),
                "dataset_directory": str(dataset),
                "manifest_sha256": report["dataset"]["manifest_sha256"],
                "formats": report["dataset"]["formats"],
                "created": report["created"],
                "config": entry["config"],
                "fit_info": entry["fit_info"],
                "calibration": {
                    name: _brief(entry["calibration"][name]) for name in chosen
                },
                "metrics": {
                    name: {
                        key: entry["scores"][name][key]
                        for key in ("fine_nll", "action_nll", "target_nll", "mega_nll")
                    }
                    for name in chosen
                },
                "note": "predict on features only, after sheet_unknown_as_closed",
            },
        )
        loaded = load_predictor(path)
        again = predict(loaded.predictor, validation)
        first = preds[label][VALIDATION]
        gap = max(float(np.abs(again[key] - first[key]).max()) for key in first)
        report["artifacts"][label] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "reloaded_max_abs_difference_on_validation": gap,
            "dex_signature_diff": loaded.meta["dex_signature_diff"],
        }
        log(f"wrote {path} ({path.stat().st_size} bytes), reload gap {gap:.1e}")

    report["seconds"] = round(time.time() - started, 1)
    (out / REPORT_JSON).write_text(
        json.dumps(_plain(report), indent=1, allow_nan=False), encoding="utf-8"
    )
    (out / REPORT_MD).write_text(render_markdown(_plain(report)), encoding="utf-8")
    log(f"wrote {out / REPORT_JSON} and {out / REPORT_MD} in {report['seconds']} s")
    return report


# --- markdown -----------------------------------------------------------------


def _number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return f"{int(value):,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fine(scores: Mapping[str, Mapping[str, Any]], digits: int = 4) -> str:
    """Fine NLL on every evaluation split that was scored, in ``EVAL_SPLITS``
    order; '-' for a split with no label."""
    return " / ".join(
        _number((scores.get(name) or {}).get("fine_nll"), digits)
        for name in EVAL_SPLITS
        if name in scores
    )


def _row(cells: Sequence[Any]) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _table(header: Sequence[Any], rows: Sequence[Sequence[Any]]) -> list[str]:
    return [_row(header), _row(["---"] * len(header)), *(_row(row) for row in rows), ""]


def _interval(entry: Mapping[str, Any]) -> str:
    if entry.get("diff") is None:
        return "-"
    if entry.get("se") is None:
        return f"{entry['diff']:+.4f}"
    return f"{entry['diff']:+.4f} [{entry['low']:+.4f}, {entry['high']:+.4f}]"


def render_markdown(report: Mapping[str, Any]) -> str:
    """The fit report as markdown; every number comes from the report dict."""
    lines: list[str] = ["# Opponent predictor: count tables", ""]
    data = report["dataset"]
    splits = scored_splits(report)
    lines += [
        f"Rendered from `{REPORT_JSON}` by `training/fit_oppmodel_tables.py`. "
        f"Created {report['created']}; dataset `{data['directory']}` (tag "
        f"{data['tag']}, manifest sha256 `{str(data['manifest_sha256'])[:12]}`); "
        f"{report.get('seconds', '-')} s on one core.",
        "",
        f"Metric: {report['metric']}.",
        "",
    ]
    left_out = data.get("validation_rows_left_out_holdout_battles")
    if left_out is not None:
        lines += [
            f"Fitted on {_number(data.get('fit_examples'))} train examples. "
            f"Validation leaves out {_number(left_out)} rows of battles with a "
            "ladder-holdout opponent: every strength below was chosen without "
            "them.",
            "",
        ]
    plan = report.get("evaluation_plan") or {}
    if plan:
        left = [
            f"`{name}` ({why})" for name, why in (plan.get("not_scored") or {}).items()
        ]
        sealed_text = "The dataset holds no own game of the sealed set."
        if plan.get("sealed_rows_in_dataset"):
            sealed_text = (
                f"{_number(plan.get('sealed_rows_in_dataset'))} rows of the bot's "
                f"own games from {plan.get('sealed_from')} on (the sealed "
                "confirmation set) were "
                + (
                    "READ (--allow-sealed-holdout)."
                    if plan.get("sealed_rows_read")
                    else "left out: never loaded."
                )
            )
        lines += [
            "Scored: "
            + ", ".join(f"`{name}`" for name in splits)
            + ". "
            + ("Not scored: " + "; ".join(left) + ". " if left else "")
            + sealed_text,
            "",
        ]
    lines += ["## Fine NLL", ""]
    floors, tables = report["floors"], report["tables"]
    rows: list[list[Any]] = [
        ["uniform over legal actions"]
        + [_number(floors["uniform"][n]["fine_nll"]) for n in splits],
        ["set-key prior + constant switch rate + uniform targets"]
        + [_number(floors["prior"][n]["fine_nll"]) for n in splits],
    ]
    for label, entry in tables.items():
        rows.append([label] + [_number(entry["scores"][n]["fine_nll"]) for n in splits])
    lines += _table(["model", *splits], rows)

    lines += ["## Per head", ""]
    keys = (
        "action_nll_visible",
        "action_nll_censored",
        "target_nll",
        "mega_nll",
        "action_top1",
        "action_top3",
        "fine_top1",
        "intent_top1",
    )
    rows = []
    for label, entry in tables.items():
        for name in splits:
            got = entry["scores"][name]
            rows.append(
                [label, name, _number(got["slots_scored"])]
                + [_number(got.get(key)) for key in keys]
            )
    lines += _table(["model", "split", "slots", *keys], rows)

    lines += ["## Paired differences in fine NLL (95% interval, games as clusters)", ""]
    rows = []
    for label, entry in report["comparisons"].items():
        if label.endswith("_setting"):
            continue
        rows.append([label] + [_interval(entry[n]) for n in splits])
    lines += _table(["difference", *splits], rows)
    fixed = report["comparisons"].get("elo_fixed_setting")
    if fixed:
        lines += [
            f"`elo_fixed`: edges {fixed['edges']}, strength {fixed['elo_strength']} "
            f"({fixed['why']}).",
            "",
        ]

    lines += ["## Chosen configuration", ""]
    names = sorted(next(iter(tables.values()))["config"])
    rows = [
        [name] + [tables[label]["config"][name] for label in tables] for name in names
    ]
    lines += _table(["setting", *tables], rows)

    lines += ["## Event calibration (every slot where the event is known)", ""]
    rows = []
    for label, entry in tables.items():
        for name in splits:
            for event, got in entry["calibration"][name].items():
                rows.append(
                    [
                        label,
                        name,
                        event,
                        _number(got["slots"]),
                        _number(got["predicted"]),
                        _number(got["observed"]),
                    ]
                )
    lines += _table(["model", "split", "event", "slots", "predicted", "observed"], rows)

    flags = tables.get(T.TABLE_FLAGS)
    if flags:
        lines += ["### Reliability of the flags table on validation", ""]
        for event, got in flags["calibration"][VALIDATION].items():
            rows = [
                [
                    f"{bin_['from']:.2f}-{bin_['to']:.2f}",
                    _number(bin_["slots"]),
                    _number(bin_["predicted"]),
                    _number(bin_["observed"]),
                ]
                for bin_ in got["bins"]
            ]
            lines += [f"P({event}):", ""]
            lines += _table(
                ["predicted in", "slots", "mean predicted", "observed"], rows
            )

    context = (flags or {}).get("target_ratio")
    if context:
        lines += [
            "### Target context of the flags table: which of two foes",
            "",
            "Ratio on a foe's target probability where both foes are legal "
            "targets of a damaging move, by the move's type effectiveness "
            "against that foe (rows) and against the other foe (columns), and "
            "by that foe's HP against the other's (lower / about the same / "
            "higher). Fitted on the train split.",
            "",
        ]
        labels = ("below 1", "1", "above 1")
        rows = [
            [labels[this]]
            + [
                " / ".join(f"{value:.2f}" for value in context["damaging"][this][other])
                for other in range(len(labels))
            ]
            for this in range(len(labels))
        ]
        lines += _table(
            ["against this foe", *(f"other foe {label}" for label in labels)], rows
        )
        lines += [
            "Any other move, by the HP relation alone: "
            + " / ".join(f"{value:.2f}" for value in context["other_moves"])
            + ".",
            "",
        ]

    lines += ["## Ablations of the flags table (each a refit with one change)", ""]
    rows = []
    for item in report["ablations"]:
        for name in splits:
            got = item["splits"][name]
            cal = got["calibration"]
            rows.append(
                [
                    item["name"],
                    name,
                    _number(got["fine_nll"]),
                    _interval(got["minus_flags_table"]),
                    _number(got["action_nll_visible"]),
                    _number(got["action_nll_censored"]),
                    f"{_number(cal[INTENT_SWITCH]['predicted'])} / "
                    f"{_number(cal[INTENT_SWITCH]['observed'])}",
                    f"{_number(cal[INTENT_PROTECT]['predicted'])} / "
                    f"{_number(cal[INTENT_PROTECT]['observed'])}",
                ]
            )
    lines += _table(
        [
            "variant",
            "split",
            "fine NLL",
            "minus flags table",
            "visible",
            "censored",
            "P(switch) pred / obs",
            "P(Protect) pred / obs",
        ],
        rows,
    )

    lines += ["## Elo offset grid (validation fine NLL)", ""]
    elo = tables.get(T.TABLE_ELO, {}).get("tuning") or {}
    grid = elo.get("grid") or []
    strengths = sorted({str(item["elo_strength"]) for item in grid}, key=float)
    by_edges: dict[str, dict[str, float]] = {}
    for item in grid:
        by_edges.setdefault(str(item["edges"]), {})[str(item["elo_strength"])] = item[
            "validation_fine_nll"
        ]
    rows = [
        [edges] + [_number(found.get(strength), 5) for strength in strengths]
        for edges, found in by_edges.items()
    ]
    lines += _table(["band edges / offset strength", *strengths], rows)
    if elo:
        prefers = "yes" if elo.get("validation_prefers_no_offset") else "no"
        lines += [
            "Without the offset (the flags table): "
            f"{_number(elo.get('validation_fine_nll_without_offset'), 5)}. "
            f"Validation prefers no offset: {prefers}. The elo_table that is "
            "written uses the best finite setting, so its difference to the flags "
            "table above is a measurement, not zero by construction.",
            "",
        ]

    lines += ["## Artifacts", ""]
    rows = [
        [
            label,
            f"`{entry['path']}`",
            _number(entry["bytes"]),
            f"{entry['reloaded_max_abs_difference_on_validation']:.1e}",
            entry["dex_signature_diff"] or "none",
        ]
        for label, entry in report.get("artifacts", {}).items()
    ]
    lines += _table(
        ["model", "file", "bytes", "reload max abs difference", "dex signature diff"],
        rows,
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--dataset", default="results_oppmodel/v1_ondisk")
    parser.add_argument(
        "--out",
        default=None,
        help=f"output directory (default {DEFAULT_OUT}; required with --limit)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="keep at most this many examples of each split (a smoke run)",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help=f"rewrite {REPORT_MD} from an existing {REPORT_JSON}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the fit that --out already holds",
    )
    parser.add_argument(
        "--eval-splits",
        default=None,
        help="comma list of the splits to score, from "
        + ", ".join(EVAL_SPLITS)
        + f" ({VALIDATION} is always needed). Default: all four for a dataset "
        "of before the fourth build; without the test split and its time "
        "slice for a dataset that holds own games from "
        f"{OWN_SEALED_FROM} on or was built with --own-before",
    )
    parser.add_argument(
        "--score-test",
        action="store_true",
        help="score the test split and its time slice of a dataset of the "
        "fourth build (confirmation data: only for the final reading)",
    )
    parser.add_argument(
        "--allow-sealed-holdout",
        action="store_true",
        help=f"read the bot's own games from {OWN_SEALED_FROM} on (the sealed "
        "set); without it their rows are never loaded",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Command line. The last line printed is ``FIT_DONE`` or ``FIT_FAILED ...``."""
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        if exc.code in (0, None):
            raise
        say(f"{FAILED} bad arguments")
        return 2
    try:
        if args.limit is not None and args.out is None:
            # A smoke fit must never land on the real tables (the R1 bar).
            raise FitError(
                f"--limit is a smoke run: give it its own --out, not {DEFAULT_OUT}"
            )
        out = Path(args.out or DEFAULT_OUT)
        if args.render_only:
            report = json.loads((out / REPORT_JSON).read_text(encoding="utf-8"))
            (out / REPORT_MD).write_text(render_markdown(report), encoding="utf-8")
            say(f"wrote {out / REPORT_MD}")
            return 0
        run(
            Path(args.dataset),
            out,
            limit=args.limit,
            overwrite=args.overwrite,
            eval_splits=args.eval_splits,
            score_test=args.score_test,
            allow_sealed_holdout=args.allow_sealed_holdout,
        )
    except Exception as exc:
        reason = " ".join(f"{type(exc).__name__}: {exc}".split())
        say(f"{FAILED} {reason}")
        return 1
    say(DONE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
