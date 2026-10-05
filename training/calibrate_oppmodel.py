"""Fit the event calibration of a neural opponent predictor and store it.

    nice -n 19 .venv/bin/python training/calibrate_oppmodel.py \\
        --artifact results_oppmodel/oppnet_v2_blind/artifact.pt \\
        --dataset results_oppmodel/v2_feed \\
        --out results_oppmodel/c_oppnet_v2_blind_cal

What it fixes. The predictor's P(the slot switches out) and P(the slot uses a
Protect-family move) read too high where they are high (scorecard R3). This
script fits one monotone map per event (``oppmodel.calibration``) and writes a
NEW artifact whose predictor applies them inside each slot's action
distribution, so ``Forecast.p_switch`` / ``p_protect`` and the ranked actions
stay one distribution. Targets and the Mega head are not touched. No network
weight is trained: only predict passes run.

Hygiene.

* The maps are fitted on the VALIDATION split only, without the battles that
  have a ladder-holdout opponent (``m_flag`` bit 4), exactly as the trainer
  chooses its epoch and temperatures. ``test`` and ``ladder_holdout`` are
  loaded only after the artifact has passed its checks, and only to print the
  same before / after tables: nothing is fitted or chosen on them.
* The predictor sees features only: label (``y_*``) and bookkeeping (``m_*``)
  arrays are removed and the "unknown" sheet code is read as "closed"
  (``features.sheet_unknown_as_closed``) before every ``predict``.
* The dataset must be the build the artifact was trained on (same manifest
  sha256), or its validation rows could be the model's training rows;
  ``--allow-other-dataset`` overrides and the report says so.
* An artifact that already carries an event calibration is refused: a second
  map fitted on calibrated output must never be stacked on the first.
  ``--refit`` drops the stored one and fits anew from the uncalibrated output.

Context terms. One map over all slot-turns left P(switch) 8-10 points too
high on turn 1 on held-out players (review of 2026-10-05). Each map may
therefore carry an additive logit term per public flag of the slot (turn 1,
the Pokemon's first turn on the field, it protected last turn: the count
table's three flags, feature arrays the predictor is given anyway). The same
grouped cross-validation inside validation decides per event which terms are
taken, one at a time: a term must lower the out-of-sample event NLL of the
map so far by more than ``--term-margin`` and in most folds, so a flag that
adds nothing stays out. ``--no-context-terms`` fits the two-parameter maps
only. The report shows every split by those flags, before and after.

Steps: load the artifact (strict: a differing dex is refused) and the
validation split; predict; fit (``calibration.fit_event_calibration``); write
``artifact.pt.unverified`` through ``oppmodel.artifact.save_artifact`` with
the name suffix; RELOAD it and check

* the reloaded predictor carries the fitted calibration, its payload is
  version 2 (the model code from before event calibration refuses it rather
  than serve it uncalibrated), and its network, temperatures and featurizer
  are the input's, bit for bit;
* its action probabilities equal ONE application of the maps to the input's
  (and not two); targets and Mega are identical;
* its predictions pass ``features.normalize_prediction`` unchanged;
* fine NLL on validation is not worse by more than 0.002 nats.

Only then is the file renamed to ``artifact.pt``. A run that fails a check
ends ``CAL_FAILED``, leaves no ``artifact.pt`` and keeps the numbers in its
report.

Report: ``calibration_report.json`` and ``calibration_report.md`` (rendered
from the JSON only; ``--render-only`` rewrites it). Per split and event: the
observed rate, mean prediction, event NLL, Brier score, expected calibration
error (10 equal-width bins, next to what a calibrated predictor of the same
size would show by chance), the reliability table, and at the pre-registered
thresholds (switch 0.35 and 0.6; Protect 0.5, 0.6, 0.8) the mean prediction
against what happened, with 95% intervals from a bootstrap over games.

What the calibrated numbers mean, and two things the report shows because of
it. The maps are fitted over every slot-turn where the event is known,
including those whose action is HIDDEN (knocked out before moving, flinch,
the game ended): there the log proves "no switch" and mostly "no Protect". A
guard reads a probability at the start of the turn, before anyone knows
whether the slot will get to act, so that is the population to be right on.
But a hidden action is never a switch or a Protect, so among the VISIBLE
actions alone both events are more frequent, and after the fit the visible
actions alone happen more often than predicted. The report therefore splits
every split's events and fine NLL by visible / hidden action, and shows the
side effect on the attack events (labelled only where the click is visible):
the mass taken from switch and Protect goes to the other moves. For the same
reason the report gives the joint reply coverage (``oppmodel.joint``, the
scorecard's measure) before and after: it is counted on turns where both
actions were seen, where switches and Protect moves are over-represented, so
the calibrated distribution covers slightly fewer of them. A guard reads the
calibrated scalars; a reply list for a search is not better for the
calibration.

A ``--limit`` run is a smoke: its maps are fitted on the first N validation
examples, so its artifact is named ``<source><suffix>_limit<N>``, never the
name of the real calibration.

Log lines, all flushed: one ``PREDICT`` line per chunk of every pass, ``FIT``,
``SAVED``, ``RELOAD``, ``CHECK``, ``INFORMATIONAL`` and a last line
``CAL_DONE`` (exit 0) or ``CAL_FAILED <reason>`` (exit 1; 2 for bad
arguments). An output directory that holds anything is refused without
``--overwrite``; with it, only a directory of an earlier calibration run is
replaced, never the input artifact's own directory, and only after every
refusal that needs no computation (the artifact's kind, a calibration it
already carries, another dataset build, the name) has passed: a refused run
leaves the earlier run's files as they were.
"""

from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path as _Path

_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # numpy's BLAS on macOS
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation import oppmodel_scorecard as S
from training import train_oppmodel as T
from vgc_bench.src.oppmodel import artifact as A
from vgc_bench.src.oppmodel import calibration as C
from vgc_bench.src.oppmodel import features as F
from vgc_bench.src.oppmodel import joint as J
from vgc_bench.src.oppmodel import model as M

ROOT = Path(__file__).resolve().parents[1]
SPLIT_VAL = "val"
SPLIT_LADDER = "ladder_holdout"
INFORMATIONAL_SPLITS: tuple[str, ...] = ("test", SPLIT_LADDER)
SPLIT_SHORT = {
    SPLIT_VAL: "validation",
    "test": "test players",
    SPLIT_LADDER: "ladder holdout",
}
SPLIT_NOTE = {
    SPLIT_VAL: "the split the maps were fitted on",
    "test": "INFORMATIONAL: nothing fitted here",
    SPLIT_LADDER: "INFORMATIONAL: nothing fitted here",
}
ARTIFACT_NAME = "artifact.pt"
UNVERIFIED_SUFFIX = ".unverified"
REPORT_JSON = "calibration_report.json"
REPORT_MD = "calibration_report.md"
DONE = "CAL_DONE"
FAILED = "CAL_FAILED"
RUNNING = "CAL_RUNNING"
NAME_SUFFIX = "_cal"
FINE_TOLERANCE = 0.002  # nats: validation fine NLL may not get worse by more
ONCE_TOLERANCE = 2.0e-6  # float32 action probabilities against one application
# The thresholds a guard would read, pre-registered for R3 and its second pass.
THRESHOLDS: dict[str, tuple[float, ...]] = {
    C.EVENT_SWITCH: (0.35, 0.6),
    C.EVENT_PROTECT: (0.5, 0.6, 0.8),
}
EVENT_TEXT = {
    C.EVENT_SWITCH: "the slot switches out by choice",
    C.EVENT_PROTECT: "the slot uses a Protect-family move",
}
SIDES: tuple[str, ...] = ("before", "after")
# A slot-turn's action is VISIBLE when the log shows the move or the switch,
# HIDDEN when the Pokemon never acted (knocked out first, flinch, sleep, the
# game ended): the log then still proves "no switch" and mostly "no Protect".
POPULATIONS: tuple[str, ...] = ("visible", "hidden")
# The attack events a guard reads (two opposing Pokemon on the field), and the
# pre-registered threshold with one above it.
ATTACK_EVENTS: tuple[str, ...] = (S.EVENT_ATTACK_TWO, S.EVENT_EITHER_TWO)
ATTACK_THRESHOLDS: tuple[float, ...] = (S.PRE_REGISTERED[S.EVENT_ATTACK_TWO], 0.9)
RESAMPLES = 2000
ECE_DRAWS = 200
CHUNK = 8192  # examples per predict call: one heartbeat line each
HEADS: tuple[str, ...] = ("action", "target", "mega")
LIMIT_MARK = "_limit"  # a --limit run's name ends with this and the limit
# The one payload version the model code from before event calibration reads.
# A calibrated artifact stored under it would load there and predict
# uncalibrated, so the run refuses to keep such a file, whatever the model
# code of the day says a calibrated payload's version is.
OLD_READER_VERSION = 1
# The public context of a slot-turn, as the report words it: each flag of
# ``calibration.CONTEXT`` set, and not set.
CONTEXT_TEXT: dict[str, tuple[str, str]] = {
    C.CONTEXT_TURN_ONE: ("turn 1", "turn 2 and later"),
    C.CONTEXT_FIRST_TURN: (
        "first turn on the field",
        "not the first turn on the field",
    ),
    C.CONTEXT_PROTECTED_LAST: ("protected last turn", "did not protect last turn"),
}
# The slices of the joint reply coverage the report follows through the fit.
COVERAGE_SLICES: tuple[str, ...] = (S.SLICE_ALL, *S.REPLY_CLASSES)


class CalibrationError(RuntimeError):
    """The run cannot give a valid artifact: a failed check or a bad input."""


def say(text: str) -> None:
    print(text, flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--artifact", default=None, help="the oppnet artifact to read")
    parser.add_argument("--dataset", default=None, help="the dataset it was fitted on")
    parser.add_argument("--out", required=True, help="a NEW run directory")
    parser.add_argument("--suffix", default=NAME_SUFFIX, help="added to the name")
    parser.add_argument("--folds", type=int, default=C.FOLDS)
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--isotonic-margin", type=float, default=C.ISOTONIC_MARGIN)
    parser.add_argument(
        "--term-margin",
        type=float,
        default=C.TERM_MARGIN,
        help="nats per fitted slot-turn the map with context terms must gain "
        "out of sample to be taken",
    )
    parser.add_argument(
        "--no-context-terms",
        action="store_true",
        help="fit the two-parameter maps only (no term for turn 1, first turn "
        "on the field, protected last turn)",
    )
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="smoke: examples")
    parser.add_argument(
        "--scorecard",
        default=None,
        help="a scorecard.json: its count table's event rates on the ladder "
        "holdout are quoted next to this run's",
    )
    parser.add_argument("--scorecard-predictor", default=S.DEFAULT_REFERENCE)
    parser.add_argument(
        "--refit",
        action="store_true",
        help="the artifact already has an event calibration: drop it, fit anew",
    )
    parser.add_argument(
        "--allow-other-dataset",
        action="store_true",
        help="fit on a dataset build other than the one the artifact names",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an earlier calibration run that --out already holds",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help=f"rewrite {REPORT_MD} from {REPORT_JSON} in --out and stop",
    )
    return parser.parse_args(argv)


# --- small helpers --------------------------------------------------------------


def _resolve(path: str | Path) -> Path:
    found = Path(path)
    return found if found.is_absolute() else ROOT / found


def _n(batch: Mapping[str, np.ndarray]) -> int:
    return int(np.asarray(batch["action_mask"]).shape[0])


def _mean(values: np.ndarray) -> float | None:
    data = np.asarray(values, dtype=np.float64)
    return float(data.mean()) if data.size else None


def _plain(value: Any) -> Any:
    """JSON-safe copy: numpy unpacked, tuples as lists, non-finite as None."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, str):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def same(first: Any, second: Any) -> bool:
    """Deep equality of artifact payloads: tensors and arrays bit for bit."""
    if isinstance(first, torch.Tensor) or isinstance(second, torch.Tensor):
        return (
            isinstance(first, torch.Tensor)
            and isinstance(second, torch.Tensor)
            and first.dtype == second.dtype
            and first.shape == second.shape
            and bool(torch.equal(first, second))
        )
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return (
            isinstance(first, np.ndarray)
            and isinstance(second, np.ndarray)
            and first.dtype == second.dtype
            and first.shape == second.shape
            and bool(np.array_equal(first, second, equal_nan=first.dtype.kind == "f"))
        )
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        return set(first) == set(second) and all(
            same(first[key], second[key]) for key in first
        )
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        return len(first) == len(second) and all(
            same(a, b) for a, b in zip(first, second)
        )
    if isinstance(first, float) and isinstance(second, float):
        return first == second or (math.isnan(first) and math.isnan(second))
    return type(first) is type(second) and first == second


def trained_on_manifest(extra: Mapping[str, Any]) -> str | None:
    """The dataset manifest's sha256 an artifact says it was fitted on."""
    found = extra.get("manifest_sha256")
    inner = extra.get("dataset")
    if found is None and isinstance(inner, Mapping):
        found = inner.get("manifest_sha256")
    return str(found) if found else None


def run_files(out_dir: Path) -> list[Path]:
    """Every file a run writes into its directory."""
    artifact = out_dir / ARTIFACT_NAME
    return [
        artifact,
        artifact.with_name(artifact.name + UNVERIFIED_SUFFIX),
        out_dir / REPORT_JSON,
        out_dir / REPORT_MD,
    ]


def check_out(out_dir: Path, source: Path, overwrite: bool) -> bool:
    """Whether ``out_dir`` holds an earlier run that this one may replace.

    Raises ``CalibrationError`` when the directory cannot be this run's:
    never the directory of the input artifact; a directory that holds
    anything needs ``overwrite``, and then it must be a calibration run's (it
    holds a calibration report): another model's directory is never emptied.
    Nothing is written or removed here.
    """
    if out_dir.resolve() == source.resolve().parent:
        raise CalibrationError(
            f"--out is the input artifact's own directory ({out_dir}); "
            "the calibrated artifact goes to a new one"
        )
    if out_dir.exists() and not out_dir.is_dir():
        raise CalibrationError(f"{out_dir} is not a directory")
    held = sorted(out_dir.iterdir()) if out_dir.exists() else []
    if not held:
        return False
    if not overwrite:
        raise CalibrationError(
            f"{out_dir} is not empty ({held[0].name}); give --overwrite to "
            "replace an earlier calibration run"
        )
    if not (out_dir / REPORT_JSON).is_file():
        raise CalibrationError(
            f"{out_dir} holds no {REPORT_JSON}: not a calibration run, not replaced"
        )
    return True


def prepare_out(out_dir: Path, source: Path, overwrite: bool) -> None:
    """Make ``out_dir`` this run's, or raise ``CalibrationError``.

    ``check_out`` first; then only the files a calibration run writes are
    removed. ``calibrate`` calls this after every refusal that needs no
    computation, so a run refused for its arguments removes nothing.
    """
    if check_out(out_dir, source, overwrite):
        for path in run_files(out_dir):
            if path.exists():
                path.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)


def calibrated_name(
    source_name: str, suffix: str, limit: int | None, refit: bool = False
) -> str:
    """The name of the calibrated artifact made from ``source_name``.

    ``<source><suffix>``, and ``<source><suffix>_limit<N>`` for a ``--limit``
    run, whose maps are a smoke's: it can never take the name of the real
    calibration. With ``refit`` (the source is itself an output of this
    script) the source's own ``<suffix>`` or ``<suffix>_limit<N>`` ending is
    dropped first, so a refit keeps one suffix.
    """
    stem = source_name
    if refit:
        ending = rf"{re.escape(suffix)}(?:{re.escape(LIMIT_MARK)}\d+)?$"
        stem = re.sub(ending, "", source_name) or source_name
    tail = suffix if limit is None else f"{suffix}{LIMIT_MARK}{int(limit)}"
    return f"{stem}{tail}"


# --- predicting and scoring -----------------------------------------------------


def predict_features(
    predictor: Any, batch: Mapping[str, np.ndarray], label: str, chunk: int = CHUNK
) -> dict[str, np.ndarray]:
    """``predictor.predict`` over a whole batch, on the features alone.

    Labels and bookkeeping arrays are removed and unknown sheets read as
    closed first. One log line per chunk. Raises ``CalibrationError`` when the
    predictor fell back to its uniform answer or returned something else than
    finite ``action`` / ``target`` / ``mega``.
    """
    features = F.sheet_unknown_as_closed(M.strip_labels(batch))
    n = _n(batch)
    if n == 0:
        raise CalibrationError(f"{label}: no examples")
    errors = S.predict_errors(predictor)
    started = time.perf_counter()
    parts: list[dict[str, np.ndarray]] = []
    for start in range(0, n, max(1, chunk)):
        made = predictor.predict(F.take(features, slice(start, start + max(1, chunk))))
        if S.predict_errors(predictor) != errors:
            raise CalibrationError(
                f"{label}: the predictor failed and fell back to the uniform "
                f"answer ({dict(getattr(predictor, 'counters', {}))})"
            )
        if not isinstance(made, Mapping) or any(name not in made for name in HEADS):
            raise CalibrationError(f"{label}: predict did not return {HEADS}")
        parts.append({name: np.asarray(made[name]) for name in HEADS})
        say(
            f"  PREDICT {label} {min(n, start + max(1, chunk))}/{n} examples "
            f"{time.perf_counter() - started:.1f} s"
        )
    pred = {name: np.concatenate([part[name] for part in parts]) for name in HEADS}
    if not all(np.isfinite(value).all() for value in pred.values()):
        raise CalibrationError(f"{label}: a prediction is not finite")
    return pred


def fine_scores(
    pred: Mapping[str, np.ndarray], batch: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Fine NLL per labelled slot-turn (``features.slot_nll``) and its parts.

    ``visible`` / ``hidden``: the same mean over the slot-turns whose action
    the log shows / hides (a hidden one is scored by its set loss).
    """
    nll = F.slot_nll(pred, batch)
    scored = nll["fine_scored"]
    hidden = nll["censored"]
    slots = int(scored.sum())
    return {
        "fine": _mean(nll["fine"][scored]),
        "action": _mean(nll["action"][nll["action_scored"]]),
        "target": float(nll["target"].sum() / slots) if slots else None,
        "mega": _mean(nll["mega"][nll["mega_scored"]]),
        "slots": slots,
        POPULATIONS[0]: _mean(nll["fine"][scored & ~hidden]),
        POPULATIONS[1]: _mean(nll["fine"][hidden]),
        f"{POPULATIONS[0]}_slots": int((scored & ~hidden).sum()),
        f"{POPULATIONS[1]}_slots": int(hidden.sum()),
    }


def ece(probability: np.ndarray, outcome: np.ndarray, bins: int = S.BINS) -> float:
    """Expected calibration error over equal-width bins (the reliability
    table's): the mean absolute gap between predicted and observed rate."""
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    y = np.asarray(outcome, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return float("nan")
    index = np.clip(np.floor(p * bins).astype(np.int64), 0, bins - 1)
    gap = np.bincount(index, weights=p - y, minlength=bins)
    return float(np.abs(gap).sum() / p.size)


def ece_if_calibrated(
    probability: np.ndarray, rng: np.random.Generator, draws: int = ECE_DRAWS
) -> dict[str, Any]:
    """The ECE a perfectly calibrated predictor with these probabilities
    would show on as many slot-turns: mean and 95th percentile over outcomes
    drawn from the probabilities themselves. Nothing is fitted."""
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    if p.size == 0 or draws <= 0:
        return {"mean": None, "p95": None}
    values = np.array([ece(p, rng.random(p.size) < p) for _ in range(draws)])
    return {"mean": float(values.mean()), "p95": float(np.quantile(values, 0.95))}


def _gap(total: Any, hits: Any, slots: Any) -> Any:
    return (total - hits) / slots


def event_report(
    before: Mapping[str, np.ndarray],
    after: Mapping[str, np.ndarray],
    batch: Mapping[str, np.ndarray],
    *,
    resamples: int = RESAMPLES,
    seed: int = C.SEED,
    ece_draws: int = ECE_DRAWS,
) -> dict[str, Any]:
    """Both events of one split, before and after, as the scorecard counts them.

    Rows: every slot-turn where the log tells whether the event happened
    (``scorecard.event_labels``); probabilities from ``features.event_probs``.
    Intervals are 95% percentile intervals of a bootstrap that resamples games.
    """
    labels = S.event_labels(batch)
    possible = S.event_possible(batch)
    probabilities = {
        SIDES[0]: S.event_predictions(before, batch),
        SIDES[1]: S.event_predictions(after, batch),
    }
    battle = (
        np.asarray(batch["m_battle"]) if "m_battle" in batch else np.arange(_n(batch))
    )
    sums = S.GameSums(battle)
    for event in C.EVENTS:
        known, happened = labels[event]
        sums.count(("n", event), known)
        sums.add(("y", event), happened, known)
        for side in SIDES:
            p = probabilities[side][event]
            sums.add(("p", side, event), p, known)
            for threshold in THRESHOLDS[event]:
                fired = known & (p >= threshold)
                sums.count(("fired", side, event, threshold), fired)
                sums.add(("hit", side, event, threshold), happened, fired)
                sums.add(("sum", side, event, threshold), p, fired)
    draws = sums.draws(resamples, seed)
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    for event in C.EVENTS:
        known, happened = labels[event]
        y = happened[known]
        block: dict[str, Any] = {
            "slots": int(known.sum()),
            "games": int((sums.column(("n", event)) > 0).sum()),
            "observed": _mean(y),
            "impossible": int((known & ~possible[event]).sum()),
        }
        for side in SIDES:
            p = probabilities[side][event][known]
            summary = S.brier_summary(p, y)
            block[side] = {
                "predicted": summary["predicted"],
                "bias": draws.apply(
                    _gap, ("p", side, event), ("y", event), ("n", event)
                ),
                "nll": C.binary_nll(p, y) if p.size else None,
                "brier": summary["brier"],
                "skill": summary["skill"],
                "ece": ece(p, y) if p.size else None,
                "ece_if_calibrated": ece_if_calibrated(p, rng, ece_draws),
                "reliability": S.reliability_table(p, y),
            }
        rows: list[dict[str, Any]] = []
        for threshold in THRESHOLDS[event]:
            row: dict[str, Any] = {"threshold": float(threshold)}
            for side in SIDES:
                decision = S.decision_row(
                    probabilities[side][event][known], y, threshold
                )
                key = (side, event, threshold)
                row[side] = {
                    "slots": decision["slots"],
                    "share": decision["share"],
                    "predicted": decision["predicted"],
                    "happened": draws.ratio(("hit", *key), ("fired", *key)),
                    "gap": draws.apply(
                        _gap, ("sum", *key), ("hit", *key), ("fired", *key)
                    ),
                    "recall": decision["recall"],
                }
            fired = known & (probabilities[SIDES[0]][event] >= threshold)
            row["rows_fired_before"] = {
                "slots": int(fired.sum()),
                "predicted_before": _mean(probabilities[SIDES[0]][event][fired]),
                "predicted_after": _mean(probabilities[SIDES[1]][event][fired]),
                "happened": _mean(happened[fired]),
            }
            rows.append(row)
        block["thresholds"] = rows
        out[event] = block
    return out


def _decision(probability: np.ndarray, outcome: np.ndarray, threshold: float) -> Any:
    """Slot-turns at or above ``threshold``: count, mean prediction, how often."""
    fired = probability >= threshold
    return {
        "slots": int(fired.sum()),
        "predicted": _mean(probability[fired]),
        "happened": _mean(outcome[fired]),
    }


def population_report(
    before: Mapping[str, np.ndarray],
    after: Mapping[str, np.ndarray],
    batch: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Both events split by whether the slot's action is visible in the log.

    The two kinds cannot be told apart at the start of a turn, and a hidden
    action is never a switch (and, where the log tells, never a Protect), so
    each kind alone is a population selected on the outcome. The split shows
    where a miscalibration over all slot-turns comes from, and what the
    calibrated numbers do NOT mean: the rate among the slots that got to act.
    """
    labels = S.event_labels(batch)
    probabilities = (
        S.event_predictions(before, batch),
        S.event_predictions(after, batch),
    )
    kind = np.asarray(batch["y_kind"]).astype(np.int64)
    visible = (kind == F.Y_KIND_MOVE) | (kind == F.Y_KIND_SWITCH)
    out: dict[str, Any] = {}
    for event in C.EVENTS:
        known, happened = labels[event]
        block: dict[str, Any] = {}
        for name, rows in zip(POPULATIONS, (known & visible, known & ~visible)):
            y = happened[rows]
            first, second = (p[event][rows] for p in probabilities)
            block[name] = {
                "slots": int(rows.sum()),
                "share_of_known": float(rows.sum() / known.sum())
                if known.any()
                else None,
                "observed": _mean(y),
                "predicted_before": _mean(first),
                "predicted_after": _mean(second),
                "nll_before": C.binary_nll(first, y) if y.size else None,
                "nll_after": C.binary_nll(second, y) if y.size else None,
                "thresholds": [
                    {
                        "threshold": float(threshold),
                        SIDES[0]: _decision(first, y, threshold),
                        SIDES[1]: _decision(second, y, threshold),
                    }
                    for threshold in THRESHOLDS[event]
                ],
            }
        out[event] = block
    return out


def attack_report(
    before: Mapping[str, np.ndarray],
    after: Mapping[str, np.ndarray],
    batch: Mapping[str, np.ndarray],
    *,
    resamples: int = RESAMPLES,
    seed: int = C.SEED,
) -> dict[str, Any]:
    """The side effect on the attack events (``ATTACK_EVENTS``), as the
    scorecard counts them: the mass taken from switch and Protect goes to the
    other moves, so these probabilities move too."""
    labels = S.event_labels(batch)
    probabilities = {
        SIDES[0]: S.event_predictions(before, batch),
        SIDES[1]: S.event_predictions(after, batch),
    }
    battle = (
        np.asarray(batch["m_battle"]) if "m_battle" in batch else np.arange(_n(batch))
    )
    sums = S.GameSums(battle)
    for event in ATTACK_EVENTS:
        known, happened = labels[event]
        for side in SIDES:
            for threshold in ATTACK_THRESHOLDS:
                fired = known & (probabilities[side][event] >= threshold)
                sums.count(("fired", side, event, threshold), fired)
                sums.add(("hit", side, event, threshold), happened, fired)
    draws = sums.draws(resamples, seed)
    out: dict[str, Any] = {}
    for event in ATTACK_EVENTS:
        known, happened = labels[event]
        y = happened[known]
        block: dict[str, Any] = {"slots": int(known.sum()), "observed": _mean(y)}
        for side in SIDES:
            p = probabilities[side][event][known]
            summary = S.brier_summary(p, y)
            block[side] = {
                "predicted": summary["predicted"],
                "brier": summary["brier"],
                "nll": C.binary_nll(p, y) if p.size else None,
            }
        block["thresholds"] = []
        for threshold in ATTACK_THRESHOLDS:
            row: dict[str, Any] = {"threshold": float(threshold)}
            for side in SIDES:
                found = _decision(probabilities[side][event][known], y, threshold)
                key = (side, event, threshold)
                found["happened"] = draws.ratio(("hit", *key), ("fired", *key))
                row[side] = found
            block["thresholds"].append(row)
        out[event] = block
    return out


def context_groups(batch: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-slot masks ``[N, 2]`` by public context: every flag of
    ``calibration.CONTEXT`` set, and not set (``CONTEXT_TEXT``). Empty when
    the batch lacks the feature arrays the flags are read from."""
    try:
        around = C.event_context(batch)
    except (KeyError, ValueError):
        return {}
    out: dict[str, np.ndarray] = {}
    for index, name in enumerate(C.CONTEXT):
        yes, no = CONTEXT_TEXT[name]
        out[yes] = around[..., index]
        out[no] = ~around[..., index]
    return out


def context_report(
    before: Mapping[str, np.ndarray],
    after: Mapping[str, np.ndarray],
    batch: Mapping[str, np.ndarray],
    *,
    resamples: int = RESAMPLES,
    seed: int = C.SEED,
) -> dict[str, Any]:
    """Both events by the public context of the slot, before and after.

    A map that is right over all slot-turns can be wrong for a part of them.
    Per event and per group of ``context_groups`` (turn 1 / later, first turn
    on the field / not, protected last turn / not): the slot-turns where the
    event is known, the observed rate, the mean prediction with its bias, the
    event NLL, and at every pre-registered threshold the mean prediction
    against what happened, with 95% intervals from a bootstrap over games.
    Nothing is fitted here.
    """
    groups = context_groups(batch)
    if not groups:
        return {}
    labels = S.event_labels(batch)
    probabilities = {
        SIDES[0]: S.event_predictions(before, batch),
        SIDES[1]: S.event_predictions(after, batch),
    }
    battle = (
        np.asarray(batch["m_battle"]) if "m_battle" in batch else np.arange(_n(batch))
    )
    sums = S.GameSums(battle)
    for event in C.EVENTS:
        known, happened = labels[event]
        for label, rows in groups.items():
            where = known & rows
            sums.count(("n", event, label), where)
            sums.add(("y", event, label), happened, where)
            for side in SIDES:
                p = probabilities[side][event]
                sums.add(("p", side, event, label), p, where)
                for threshold in THRESHOLDS[event]:
                    fired = where & (p >= threshold)
                    key = (side, event, label, threshold)
                    sums.count(("fired", *key), fired)
                    sums.add(("hit", *key), happened, fired)
                    sums.add(("sum", *key), p, fired)
    draws = sums.draws(resamples, seed)
    out: dict[str, Any] = {}
    for event in C.EVENTS:
        known, happened = labels[event]
        block: dict[str, Any] = {}
        for label, rows in groups.items():
            where = known & rows
            y = happened[where]
            entry: dict[str, Any] = {
                "slots": int(where.sum()),
                "share_of_known": float(where.sum() / known.sum())
                if known.any()
                else None,
                "observed": _mean(y),
            }
            for side in SIDES:
                p = probabilities[side][event][where]
                entry[side] = {
                    "predicted": _mean(p),
                    "bias": draws.apply(
                        _gap,
                        ("p", side, event, label),
                        ("y", event, label),
                        ("n", event, label),
                    ),
                    "nll": C.binary_nll(p, y) if p.size else None,
                }
            entry["thresholds"] = []
            for threshold in THRESHOLDS[event]:
                row: dict[str, Any] = {"threshold": float(threshold)}
                for side in SIDES:
                    p = probabilities[side][event][where]
                    key = (side, event, label, threshold)
                    row[side] = {
                        "slots": int((p >= threshold).sum()),
                        "predicted": _mean(p[p >= threshold]),
                        "happened": draws.ratio(("hit", *key), ("fired", *key)),
                        "gap": draws.apply(
                            _gap, ("sum", *key), ("hit", *key), ("fired", *key)
                        ),
                    }
                entry["thresholds"].append(row)
            block[label] = entry
        out[event] = block
    return out


def coverage_report(
    before: Mapping[str, np.ndarray],
    after: Mapping[str, np.ndarray],
    batch: Mapping[str, np.ndarray],
    *,
    resamples: int = RESAMPLES,
    seed: int = C.SEED,
) -> dict[str, Any]:
    """Joint reply coverage before and after, the scorecard's measure.

    Counted like ``scorecard.joint_coverage_set`` (every slot on the field
    shows its whole action; a true reply through the OTHER bucket is a miss;
    the joint without the Mega bit; the first ``scorecard.JOINT_K`` replies),
    per slice of ``COVERAGE_SLICES``: top-K and top-1 coverage and the mean
    log-probability of the true joint reply on each side, and the paired
    change (after minus before, on the same examples) with a 95% interval
    from resampling games. The labels decide what is counted; a prediction is
    only ranked. Empty when the labels cannot be read as joint replies.
    """
    truth = J.true_replies(batch)
    if not truth:
        return {}
    visible = np.asarray(truth["visible"], dtype=bool)
    count = int(visible.sum())
    out: dict[str, Any] = {
        "examples": _n(batch),
        "counted": count,
        "k": S.JOINT_K,
        "slices": {},
    }
    if count == 0:
        return out
    part = F.take(batch, visible)
    masks = S.features_only(part)
    told = {name: np.asarray(values)[visible] for name, values in truth.items()}
    other = told["other"].any(-1)
    slices = {
        name: np.asarray(mask, dtype=bool)[visible]
        for name, mask in S.joint_slices(batch, truth).items()
        if name in COVERAGE_SLICES
    }
    battle = np.asarray(part["m_battle"]) if "m_battle" in part else np.arange(count)
    hits: dict[str, np.ndarray] = {}
    firsts: dict[str, np.ndarray] = {}
    logs: dict[str, np.ndarray] = {}
    for side, pred in zip(SIDES, (before, after)):
        own = {name: np.asarray(pred[name])[visible] for name in HEADS}
        made = J.joint_replies(own, masks, k=S.JOINT_K, truth=told)
        if made.n != count:
            raise CalibrationError(
                f"the {side} prediction cannot be read as joint replies "
                f"({dict(J.COUNTERS)})"
            )
        found = made.rank >= 0
        hits[side] = (found & (made.rank < S.JOINT_K) & ~other).astype(np.float64)
        firsts[side] = (found & (made.rank < 1) & ~other).astype(np.float64)
        logs[side] = np.log(np.maximum(made.prob_true, F.PROBABILITY_FLOOR))
    for label, mask in slices.items():
        entry: dict[str, Any] = {"examples": int(mask.sum())}
        for side in SIDES:
            entry[side] = {
                "top": _mean(hits[side][mask]),
                "top1": _mean(firsts[side][mask]),
                "log_prob": _mean(logs[side][mask]),
            }
        for name, values in (("top_change", hits), ("log_prob_change", logs)):
            entry[name] = (
                S.paired_difference(
                    values[SIDES[1]],
                    values[SIDES[0]],
                    mask,
                    battle,
                    resamples=resamples,
                    seed=seed,
                )
                if mask.any()
                else None
            )
        out["slices"][label] = entry
    return out


def split_report(
    before: Mapping[str, np.ndarray],
    after: Mapping[str, np.ndarray],
    batch: Mapping[str, np.ndarray],
    *,
    resamples: int = RESAMPLES,
    seed: int = C.SEED,
) -> dict[str, Any]:
    """Everything reported about one split: fine NLL, sanity, both events,
    their split by visible / hidden actions and by public context, the attack
    events and the joint reply coverage."""
    n = _n(batch)
    fine = {SIDES[0]: fine_scores(before, batch), SIDES[1]: fine_scores(after, batch)}
    sanity = {}
    for side, pred in zip(SIDES, (before, after)):
        found = S.sanity_block(pred, batch)
        sanity[side] = {
            "max_abs_change": found["max_abs_change"],
            "unchanged": found["unchanged"],
            "floor_hits": found["floor_hits"],
        }
    change = None
    if fine[SIDES[0]]["fine"] is not None and fine[SIDES[1]]["fine"] is not None:
        change = fine[SIDES[1]]["fine"] - fine[SIDES[0]]["fine"]
    # The same change as a paired difference over the same slot-turns, with a
    # 95% interval from resampling games (the scorecard's estimator).
    first, second = F.slot_nll(before, batch), F.slot_nll(after, batch)
    battle = np.asarray(batch["m_battle"]) if "m_battle" in batch else np.arange(n)
    scored, hidden = first["fine_scored"], first["censored"]
    paired = {
        name: S.paired_difference(
            second["fine"], first["fine"], rows, battle, resamples=resamples, seed=seed
        )
        for name, rows in (
            ("paired_change", scored),
            (f"paired_change_{POPULATIONS[0]}", scored & ~hidden),
            (f"paired_change_{POPULATIONS[1]}", hidden),
        )
    }
    return {
        "examples": n,
        "fine_nll": {**fine, "change": change, **paired},
        "sanity": sanity,
        "heads_untouched": {
            name: bool(np.array_equal(before[name], after[name]))
            for name in HEADS
            if name != "action"
        },
        "events": event_report(before, after, batch, resamples=resamples, seed=seed),
        "populations": population_report(before, after, batch),
        "context": context_report(before, after, batch, resamples=resamples, seed=seed),
        "attack": attack_report(before, after, batch, resamples=resamples, seed=seed),
        "coverage": coverage_report(
            before, after, batch, resamples=resamples, seed=seed
        ),
    }


def shift_summary(splits: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Per event: the observed rate, the bias and the ECE of every split read,
    and how much of the ladder holdout's miscalibration is left after the fit."""
    out: dict[str, Any] = {}
    for event in C.EVENTS:
        rows: dict[str, Any] = {}
        for name, report in splits.items():
            block = report["events"][event]
            rows[name] = {
                "slots": block["slots"],
                "observed": block["observed"],
                **{
                    f"{key}_{side}": block[side][key]
                    for side in SIDES
                    for key in ("predicted", "bias", "ece", "ece_if_calibrated", "nll")
                },
            }
        entry: dict[str, Any] = {"splits": rows}
        ladder = rows.get(SPLIT_LADDER)
        if ladder is not None:
            bias = [(ladder[f"bias_{side}"] or {}).get("value") for side in SIDES]
            eces = [ladder[f"ece_{side}"] for side in SIDES]
            floors = [
                (ladder[f"ece_if_calibrated_{side}"] or {}).get("mean")
                for side in SIDES
            ]
            entry["ladder_remaining"] = {
                "bias_before": bias[0],
                "bias_after": bias[1],
                "bias_share_left": abs(bias[1]) / abs(bias[0])
                if bias[0] and bias[1] is not None
                else None,
                "ece_before": eces[0],
                "ece_after": eces[1],
                "ece_share_left": eces[1] / eces[0]
                if eces[0] and eces[1] is not None
                else None,
                "ece_chance_before": floors[0],
                "ece_chance_after": floors[1],
            }
        out[event] = entry
    return out


def count_table_reference(path: Path, predictor: str) -> dict[str, Any]:
    """A count table's predicted and observed event rates on the ladder
    holdout, read from a scorecard's JSON. Raises ``CalibrationError``."""
    try:
        card = json.loads(path.read_text(encoding="utf-8"))
        events = card["sets"][SPLIT_LADDER]["predictors"][predictor]["events"]
        return {
            "scorecard": str(path),
            "predictor": predictor,
            "events": {
                event: {
                    "slots": events[event]["slots"],
                    "predicted": events[event]["predicted"],
                    "observed": events[event]["observed"],
                }
                for event in C.EVENTS
            },
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CalibrationError(f"--scorecard {path}: {exc!r}") from exc


# --- the run --------------------------------------------------------------------


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    """Run one calibration and return its report. Raises on any failure.

    From the moment the output directory is this run's, its report says
    ``CAL_RUNNING`` until the run ends and ``CAL_FAILED`` with the reason when
    it raises.
    """
    if not args.artifact or not args.dataset:
        raise CalibrationError("--artifact and --dataset are needed")
    source, dataset, out_dir = (
        _resolve(args.artifact),
        _resolve(args.dataset),
        _resolve(args.out),
    )
    if not source.is_file():
        raise CalibrationError(f"no artifact at {source}")
    if not (dataset / "manifest.json").is_file():
        raise CalibrationError(f"no dataset manifest in {dataset}")
    if args.scorecard and not _resolve(args.scorecard).is_file():
        raise CalibrationError(f"no scorecard at {_resolve(args.scorecard)}")
    # Every refusal that needs no computation comes before the directory is
    # touched: with --overwrite a refused run must leave the earlier run's
    # artifact and report exactly as they were.
    check_out(out_dir, source, bool(args.overwrite))
    torch.set_num_threads(max(1, int(args.threads)))
    inputs = read_inputs(args, source, dataset)
    prepare_out(out_dir, source, bool(args.overwrite))
    opening = {
        "status": RUNNING,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "args": {key: value for key, value in sorted(vars(args).items())},
    }
    write_report(out_dir, opening)
    report: dict[str, Any] = dict(opening)
    try:
        return _calibrate(args, source, dataset, out_dir, report, inputs)
    except BaseException as exc:
        reason = " ".join(f"{type(exc).__name__}: {exc}".split())
        # ``artifact`` stays set when the failure came after the checks: the
        # file is then a verified artifact, and the reason says what is missing.
        report.update({"status": FAILED, "reason": reason})
        report.setdefault("artifact", None)
        try:
            write_report(out_dir, report)
        except (OSError, TypeError, ValueError):
            pass
        raise


@dataclass
class Inputs:
    """The input artifact as a run needs it, after the refusals of
    ``read_inputs``: nothing of it depends on the output directory."""

    document: dict[str, Any]
    loaded: Any
    base: M.OppNetPredictor
    had: bool
    extra: dict[str, Any]
    trained_on: str | None
    manifest_sha: str
    same_build: bool
    name: str


def read_inputs(args: argparse.Namespace, source: Path, dataset: Path) -> Inputs:
    """Load the artifact and apply every refusal that needs no computation.

    Raises ``CalibrationError`` for: a file that is not a readable artifact; a
    kind that carries no event calibration; an artifact that already has one
    without ``--refit``; a dataset build other than the one it was trained on
    without ``--allow-other-dataset``; an empty name suffix. Writes nothing.
    """
    try:
        document = A.read_artifact(source, strict=True)
        loaded = A.load_predictor(source, strict=True)
    except (OSError, ValueError) as exc:
        raise CalibrationError(f"artifact {source}: {exc}") from exc
    base = loaded.predictor
    if loaded.kind != M.KIND or not isinstance(base, M.OppNetPredictor):
        raise CalibrationError(
            f"{source} is a {loaded.kind!r} artifact; only {M.KIND!r} carries "
            "an event calibration"
        )
    had = base.event_calibration is not None
    if had:
        if not args.refit:
            raise CalibrationError(
                f"{source} already carries an event calibration; a second one "
                "is never stacked on it (give --refit to drop it and fit anew)"
            )
        base = base.with_event_calibration(None)
    extra = dict(loaded.meta.get("extra") or {})
    trained_on = trained_on_manifest(extra)
    try:
        manifest_sha = T.manifest_sha256(dataset)
    except OSError as exc:
        raise CalibrationError(f"dataset {dataset}: {exc}") from exc
    same_build = trained_on is not None and trained_on == manifest_sha
    if not same_build and not args.allow_other_dataset:
        raise CalibrationError(
            f"the artifact was fitted on dataset manifest {trained_on}, this "
            f"dataset is {manifest_sha}: its validation rows may be the model's "
            "training rows (give --allow-other-dataset to fit anyway)"
        )
    suffix = str(args.suffix or "")
    if not suffix:
        raise CalibrationError("the new artifact needs a name suffix")
    if args.limit is not None and int(args.limit) < 0:
        raise CalibrationError(f"--limit {args.limit} is negative")
    return Inputs(
        document=document,
        loaded=loaded,
        base=base,
        had=had,
        extra=extra,
        trained_on=trained_on,
        manifest_sha=manifest_sha,
        same_build=same_build,
        # A refit of an artifact this script wrote keeps one suffix; a --limit
        # run never gets the name of the real calibration.
        name=calibrated_name(loaded.name, suffix, args.limit, refit=had),
    )


def _calibrate(
    args: argparse.Namespace,
    source: Path,
    dataset: Path,
    out_dir: Path,
    report: dict[str, Any],
    inputs: Inputs,
) -> dict[str, Any]:
    started = time.time()
    document, loaded, base, extra = (
        inputs.document,
        inputs.loaded,
        inputs.base,
        dict(inputs.extra),
    )
    had, trained_on, manifest_sha, same_build, name = (
        inputs.had,
        inputs.trained_on,
        inputs.manifest_sha,
        inputs.same_build,
        inputs.name,
    )

    splits, manifest = T.load_splits(dataset, (SPLIT_VAL,))
    validation, left_out = T.without_holdout_battles(splits[SPLIT_VAL])
    if args.limit is not None:
        validation = F.take(validation, slice(0, max(0, int(args.limit))))
    if _n(validation) == 0:
        raise CalibrationError("the validation split is empty")
    report.update(
        name=name,
        kind=M.KIND,
        source={
            "artifact": str(source),
            "sha256": S.sha256_file(source),
            "name": loaded.name,
            "elo_mode": base.elo_mode,
            "temperatures": base.temperatures,
            "mega_calibration": base.mega_calibration,
            "had_event_calibration": had,
            "payload_version": document["predictor"].get("version"),
            "trained_on_manifest_sha256": trained_on,
        },
        dataset={
            "path": str(dataset),
            "tag": manifest.get("tag"),
            "manifest_sha256": manifest_sha,
            "same_build_as_training": same_build,
            "fit_split": SPLIT_VAL,
            "fit_examples": _n(validation),
            "validation_rows_left_out_holdout_battles": left_out,
            "limit": args.limit,
        },
        thresholds={event: list(values) for event, values in THRESHOLDS.items()},
        fine_tolerance=FINE_TOLERANCE,
        resamples=int(args.resamples),
        context_terms={
            "offered": not args.no_context_terms,
            "names": list(C.CONTEXT),
            "margin": float(args.term_margin),
        },
    )
    say(
        f"CAL_START name {name} source {loaded.name} dataset {manifest.get('tag')} "
        f"fit split {SPLIT_VAL} examples {_n(validation)} (left out: {left_out} rows "
        f"of holdout battles) elo_mode {base.elo_mode} "
        f"action_T {base.action_temperature:.4f} threads {torch.get_num_threads()}"
        + (
            ""
            if args.limit is None
            else f" LIMITED to {int(args.limit)} examples per split: a smoke, "
            "not the real calibration"
        )
    )

    before = predict_features(base, validation, f"{SPLIT_VAL} before")
    groups = np.asarray(validation["m_battle"]) if "m_battle" in validation else None
    calibration = C.fit_event_calibration(
        before["action"],
        validation,
        groups=groups,
        action_temperature=base.action_temperature,
        info={
            "dataset_tag": manifest.get("tag"),
            "manifest_sha256": manifest_sha,
            "split": SPLIT_VAL,
            "rows_left_out_holdout_battles": left_out,
            "source_artifact": loaded.name,
            "created": report["created"],
        },
        folds=int(args.folds),
        seed=int(args.seed),
        margin=float(args.isotonic_margin),
        context=not args.no_context_terms,
        term_margin=float(args.term_margin),
    )
    for event, found in calibration.maps.items():
        fit = calibration.info["events"][event]
        about = fit["selection"].get("context") or {}
        say(
            f"FIT {event} map {found.kind} slope {found.slope:.4f} "
            f"bias {found.bias:+.4f} knots {len(found.knots_x)} "
            f"terms {_terms_text(dict(found.terms))}; context terms "
            f"{_reason_words(about.get('reason', 'not offered'))}; "
            f"rows {fit['rows_fitted']} of {fit['rows_known']} known; "
            f"observed {_number(fit['observed'])} predicted "
            f"{_number(fit['predicted_before'])} -> {_number(fit['predicted_after'])} "
            f"event NLL {_number(fit['nll_before'], 5)} -> "
            f"{_number(fit['nll_after'], 5)}; {fit['selection']['reason']}"
        )
    report["calibration"] = calibration.describe()

    calibrated = base.with_event_calibration(calibration, name=name)
    path = out_dir / ARTIFACT_NAME
    unverified = path.with_name(path.name + UNVERIFIED_SUFFIX)
    extra["event_calibration"] = {
        "created": report["created"],
        "source": report["source"],
        "dataset": report["dataset"],
        "calibration": calibration.describe(),
    }
    A.save_artifact(
        unverified,
        kind=M.KIND,
        name=name,
        featurizer=loaded.featurizer,
        predictor_payload=calibrated.to_payload(),
        extra=_plain(extra),
    )
    say(f"SAVED {unverified}")

    stored = A.read_artifact(unverified, strict=True)
    again = A.load_predictor(unverified, strict=True).predictor
    if not isinstance(again, M.OppNetPredictor):
        raise CalibrationError(f"the artifact reloaded as {type(again).__name__}")
    after = predict_features(again, validation, f"{SPLIT_VAL} after")
    mask, flags = validation["action_mask"], validation["cand_flag"]
    # The flags a map's context terms read: feature arrays of the batch.
    around = C.event_context(validation) if calibration.needs_context else None
    once = calibration.apply_checked(before["action"], mask, flags, around)
    twice = calibration.apply_checked(once, mask, flags, around)
    made = after["action"].astype(np.float64)
    old, new = document["predictor"], stored["predictor"]
    # The version says "carries a calibration" (see ``model.to_payload``): it
    # differs from the input's by design and is checked on its own.
    skipped = ("name", "event_calibration", "version")
    val_report = split_report(
        before, after, validation, resamples=int(args.resamples), seed=int(args.seed)
    )
    change = val_report["fine_nll"]["change"]
    checks: dict[str, Any] = {
        "reloaded_calibration_equals_fitted": again.event_calibration == calibration,
        "reloaded_name": again.name,
        "payload_version": new.get("version"),
        "network_and_temperatures_unchanged": same(
            {key: value for key, value in old.items() if key not in skipped},
            {key: value for key, value in new.items() if key not in skipped},
        ),
        "featurizer_unchanged": same(document["featurizer"], stored["featurizer"]),
        "applied_once": {
            "max_abs_difference_to_one_application": float(np.abs(made - once).max()),
            "max_abs_difference_to_two_applications": float(np.abs(made - twice).max()),
            "tolerance": ONCE_TOLERANCE,
            "identity": calibration.is_identity,
        },
        "targets_and_mega_untouched": all(val_report["heads_untouched"].values()),
        "normalize_prediction_unchanged": val_report["sanity"][SIDES[1]]["unchanged"],
        "fine_nll": {
            "before": val_report["fine_nll"][SIDES[0]]["fine"],
            "after": val_report["fine_nll"][SIDES[1]]["fine"],
            "change": change,
            "tolerance": FINE_TOLERANCE,
        },
    }
    report["checks"] = checks
    report["validation"] = val_report
    say(
        f"RELOAD fine NLL {_number(checks['fine_nll']['before'], 6)} -> "
        f"{_number(checks['fine_nll']['after'], 6)} (change {_number(change, 6)}, "
        f"tolerance +{FINE_TOLERANCE}); one application off by "
        f"{checks['applied_once']['max_abs_difference_to_one_application']:.2e}, "
        f"two by {checks['applied_once']['max_abs_difference_to_two_applications']:.2e}"
    )
    failures: list[str] = []
    if not checks["reloaded_calibration_equals_fitted"]:
        failures.append("the reloaded calibration is not the fitted one")
    if checks["reloaded_name"] != name:
        failures.append(f"the reloaded predictor is named {again.name!r}")
    stored_version = checks["payload_version"]
    if (
        stored_version == OLD_READER_VERSION
        or stored_version != M.PAYLOAD_VERSION_CALIBRATED
    ):
        failures.append(
            f"the stored payload is version {stored_version!r}: a calibrated "
            f"payload must not be version {OLD_READER_VERSION}, which a reader "
            "from before event calibration would serve uncalibrated"
        )
    if not checks["network_and_temperatures_unchanged"]:
        failures.append("the stored network or temperatures differ from the input's")
    if not checks["featurizer_unchanged"]:
        failures.append("the stored featurizer differs from the input's")
    once_off = checks["applied_once"]["max_abs_difference_to_one_application"]
    twice_off = checks["applied_once"]["max_abs_difference_to_two_applications"]
    if not once_off <= ONCE_TOLERANCE:
        failures.append(
            f"reloaded action probabilities are {once_off:.2e} from one application"
        )
    if not calibration.is_identity and not twice_off > once_off:
        failures.append("reloaded action probabilities look calibrated twice")
    if not checks["targets_and_mega_untouched"]:
        failures.append("the target or Mega probabilities changed")
    if not checks["normalize_prediction_unchanged"]:
        failures.append(
            "normalize_prediction changes the calibrated prediction by "
            f"{val_report['sanity'][SIDES[1]]['max_abs_change']}"
        )
    if change is None or not change <= FINE_TOLERANCE:
        failures.append(
            f"validation fine NLL got worse by {_number(change, 6)} nats "
            f"(tolerance {FINE_TOLERANCE})"
        )
    checks["passed"] = not failures
    checks["failures"] = failures
    for line in failures:
        say(f"CHECK FAILED {line}")
    if failures:
        report["rejected_artifact"] = str(unverified)
        raise CalibrationError("; ".join(failures))
    say("CHECK passed: reload, one application, normalisation, fine NLL")
    unverified.replace(path)
    report["artifact"] = str(path)
    report["artifact_sha256"] = S.sha256_file(path)

    # Nothing below is fitted or chosen: the artifact is already written.
    report["informational"] = {}
    present = tuple(
        split
        for split in INFORMATIONAL_SPLITS
        if split in (manifest.get("splits") or [])
    )
    extra_splits = T.load_splits(dataset, present)[0] if present else {}
    for split in present:
        rows = extra_splits[split]
        if args.limit is not None:
            rows = F.take(rows, slice(0, max(0, int(args.limit))))
        if _n(rows) == 0:
            say(f"INFORMATIONAL {split}: no examples")
            continue
        first = predict_features(base, rows, f"{split} before")
        second = predict_features(again, rows, f"{split} after")
        found = split_report(
            first, second, rows, resamples=int(args.resamples), seed=int(args.seed)
        )
        report["informational"][split] = found
        for event in C.EVENTS:
            say(
                f"INFORMATIONAL (nothing fitted here) {split} "
                f"{_event_line(found, event)}"
            )
    report["shift"] = shift_summary({SPLIT_VAL: val_report, **report["informational"]})
    if args.scorecard:
        # A quotation for the report, read after everything else is settled;
        # a card without that predictor costs the quotation, not the run.
        try:
            report["count_table"] = count_table_reference(
                _resolve(args.scorecard), str(args.scorecard_predictor)
            )
        except CalibrationError as exc:
            say(f"WARNING {exc}")
            report["count_table"] = None
            report["warnings"] = [str(exc)]
    report["status"] = DONE
    report["seconds"] = time.time() - started
    write_report(out_dir, report)
    return report


# --- report ---------------------------------------------------------------------


def _number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}" if math.isfinite(float(value)) else "-"
    return str(value)


def _percent(value: Any, digits: int = 1) -> str:
    return "-" if value is None else f"{100.0 * float(value):.{digits}f}%"


def _points(value: Any, digits: int = 1) -> str:
    """A difference of two rates, in percentage points with its sign."""
    return "-" if value is None else f"{100.0 * float(value):+.{digits}f}"


def _bracket(found: Mapping[str, Any] | None, cell: Any = _percent) -> str:
    """``value [low, high]`` of a bootstrap result."""
    if not found or found.get("value") is None:
        return "-"
    text = cell(found["value"])
    if found.get("low") is None or found.get("high") is None:
        return text
    return f"{text} [{cell(found['low'])}, {cell(found['high'])}]"


def _as_value(found: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """A ``paired_difference`` result in the shape ``_bracket`` reads."""
    if not found:
        return None
    return {
        "value": found.get("diff"),
        "low": found.get("low"),
        "high": found.get("high"),
    }


def _change(found: Mapping[str, Any] | None) -> str:
    """A paired fine-NLL change with its interval, signed."""
    return _bracket(_as_value(found), lambda v: f"{float(v):+.5f}")


def _event_line(report: Mapping[str, Any], event: str) -> str:
    """One log line of an event of a split: rates, NLL and the first threshold."""
    block = report["events"][event]
    row = block["thresholds"][0]
    return (
        f"{event}: observed {_percent(block['observed'])} predicted "
        f"{_percent(block[SIDES[0]]['predicted'])} -> "
        f"{_percent(block[SIDES[1]]['predicted'])}; event NLL "
        f"{_number(block[SIDES[0]]['nll'], 5)} -> "
        f"{_number(block[SIDES[1]]['nll'], 5)}; "
        f"at P >= {row['threshold']:g}: predicted "
        f"{_percent(row[SIDES[0]]['predicted'])} happened "
        f"{_percent(row[SIDES[0]]['happened']['value'])} "
        f"({row[SIDES[0]]['slots']} slot-turns) -> predicted "
        f"{_percent(row[SIDES[1]]['predicted'])} happened "
        f"{_percent(row[SIDES[1]]['happened']['value'])} "
        f"({row[SIDES[1]]['slots']})"
    )


def _short(split: str) -> str:
    return SPLIT_SHORT.get(split, split)


def _long(split: str) -> str:
    note = SPLIT_NOTE.get(split)
    return f"{_short(split)} ({note})" if note else _short(split)


def _said(found: Mapping[str, Any], happened: str) -> str:
    """``N said X%, happened Y``: one side of a threshold row."""
    return (
        f"{found.get('slots')} said {_percent(found.get('predicted'))}, "
        f"happened {happened}"
    )


def data_text(texts: Mapping[str, str], key: str) -> str:
    return str(texts.get(key, key))


def _line(cells: Sequence[Any]) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _table(header: Sequence[Any], rows: Sequence[Sequence[Any]]) -> list[str]:
    return [_line(header), _line(["---"] * len(header)), *(_line(row) for row in rows)]


STILL_OFF_SLOTS = 100  # fewer calls at a threshold are not called out


def _still_off(rows: Sequence[tuple[str, Mapping[str, Any]]]) -> list[str]:
    """The calls that are still miscalibrated AFTER the fit, said in words.

    ``rows`` are (where, the after side of a threshold row). Listed: those
    with at least ``STILL_OFF_SLOTS`` slot-turns whose gap has a 95% interval
    that excludes zero. Among many rows a few do so by chance; the list says
    where to look, the fitted split first, not what is proven.
    """
    found = []
    for where, side in rows:
        gap = side.get("gap") or {}
        low, high = gap.get("low"), gap.get("high")
        if (side.get("slots") or 0) < STILL_OFF_SLOTS or low is None or high is None:
            continue
        if low > 0 or high < 0:
            found.append(
                f"- {where}: {side.get('slots')} said "
                f"{_percent(side.get('predicted'))}, gap {_bracket(gap, _points)}"
            )
    head = (
        f"After the fit, among the rows above with at least {STILL_OFF_SLOTS} "
        "slot-turns at the threshold, the gap's interval excludes zero "
    )
    if not found:
        return ["", head + "in none."]
    return [
        "",
        head + "in these (a few of many rows do so by chance; a guard that "
        "fires there should set its threshold from the row, not from the map):",
        "",
        *found,
    ]


def _context_words(name: str) -> str:
    """A context flag as the report words it (the flag set)."""
    return CONTEXT_TEXT.get(name, (name, name))[0]


def _reason_words(reason: Any) -> str:
    """A selection record's reason with the flags as the report words them."""
    text = str(reason)
    for name in C.CONTEXT:
        text = text.replace(name, _context_words(name))
    return text


def _terms_text(terms: Mapping[str, Any] | None) -> str:
    """``turn 1 -0.2772, first turn on the field -0.2202``, or ``none``."""
    if not terms:
        return "none"
    return ", ".join(
        f"{_context_words(name)} {float(value):+.4f}" for name, value in terms.items()
    )


def _map_text(found: Mapping[str, Any]) -> str:
    kind = found.get("map")
    if kind == C.MAP_LOGISTIC:
        bias = float(found.get("bias") or 0.0)
        extra = "".join(
            f" {'-' if float(value) < 0 else '+'} {abs(float(value)):.4f} "
            f"[{_context_words(name)}]"
            for name, value in (found.get("terms") or {}).items()
        )
        return (
            f"sigmoid({_number(found.get('slope'))} * logit(p) "
            f"{'-' if bias < 0 else '+'} {abs(bias):.4f}{extra})"
        )
    if kind == C.MAP_ISOTONIC:
        return f"piecewise linear through {len(found.get('x') or [])} knots"
    return "p (no change)"


def _splits_of(data: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    if data.get("validation"):
        out[SPLIT_VAL] = data["validation"]
    out.update(data.get("informational") or {})
    return out


def render_report(report: Mapping[str, Any]) -> str:
    """The Markdown view of ``calibration_report.json``."""
    data = _plain(report)
    source = data.get("source") or {}
    dataset = data.get("dataset") or {}
    calibration = data.get("calibration") or {}
    checks = data.get("checks") or {}
    maps = calibration.get("maps") or {}
    fit = (calibration.get("info") or {}).get("events") or {}
    splits = _splits_of(data)
    lines = [
        f"# Event calibration: {data.get('name')}",
        "",
        f"Status: {data.get('status')}"
        + (f" ({data.get('reason')})" if data.get("reason") else "")
        + f". Rendered from {REPORT_JSON}.",
        "",
        f"- input artifact: `{source.get('artifact')}` (name {source.get('name')}, "
        f"sha256 {str(source.get('sha256'))[:12]}, Elo mode {source.get('elo_mode')}, "
        f"action temperature "
        f"{_number((source.get('temperatures') or {}).get('action'))})",
        f"- dataset: {dataset.get('tag')} (manifest sha256 "
        f"{str(dataset.get('manifest_sha256'))[:12]}); the build the model was "
        f"trained on: {dataset.get('same_build_as_training')}",
        f"- fitted on: the `{dataset.get('fit_split')}` split only, "
        f"{dataset.get('fit_examples')} examples "
        f"({dataset.get('validation_rows_left_out_holdout_battles')} rows of battles "
        "with a ladder-holdout opponent left out). Nothing is fitted or chosen on "
        "test or the ladder holdout.",
        f"- new artifact: `{data.get('artifact')}`"
        + (
            f" (sha256 {str(data.get('artifact_sha256'))[:12]})"
            if data.get("artifact_sha256")
            else ""
        ),
        "",
        "## The maps",
        "",
        "Inside each slot's action distribution the switch pointers are rescaled "
        "so that their total s becomes g(s), and the Protect-family candidates so "
        "that their share r of the non-switch mass becomes g(r); everything else "
        "is rescaled to keep the total. Targets and the Mega head are untouched. "
        "The switch map is fitted where the log tells whether the slot switched "
        "and a switch is legal; the Protect map where it tells whether the slot "
        "protected, a Protect-family candidate exists and the slot did not switch. "
        "A term in square brackets is added to the logit of the slots with that "
        "public flag (turn 1; the Pokemon's first turn on the field; it "
        "protected last turn). Cross-validated NLL: grouped by battle inside the "
        "validation split, nats per fitted slot-turn.",
        "",
    ]
    rows = []
    notes: list[str] = []
    for event in C.EVENTS:
        found, record = maps.get(event) or {}, fit.get(event) or {}
        selection = record.get("selection") or {}
        cross = selection.get("cross_validated_nll") or {}
        about = selection.get("context") or {}
        rows.append(
            [
                EVENT_TEXT[event],
                found.get("map"),
                f"`{_map_text(found)}`",
                f"{record.get('rows_fitted')} of {record.get('rows_known')}",
                _number(cross.get(C.MAP_IDENTITY), 5),
                _number(cross.get(C.MAP_LOGISTIC), 5),
                _number(about.get("cross_validated_nll"), 5),
                f"{about.get('better_folds')} of {selection.get('folds')}"
                if about.get("better_folds") is not None
                else "-",
                _number(cross.get(C.MAP_ISOTONIC), 5),
                f"{selection.get('isotonic_better_folds')} of {selection.get('folds')}",
                selection.get("reason"),
            ]
        )
        if about:
            flags = ", ".join(
                f"{_context_words(name)} {(about.get('rows') or {}).get(name)} "
                f"slot-turns / {(about.get('events') or {}).get(name)} events"
                for name in about.get("names") or []
            )
            steps = []
            for number, step in enumerate(about.get("steps") or [], 1):
                tried = ", ".join(
                    f"{_context_words(name)} {float(found.get('gain') or 0.0):+.5f} "
                    f"nats in {found.get('better_folds')} folds"
                    for name, found in (step.get("tried") or {}).items()
                )
                took = (
                    f"took '{_context_words(str(step.get('best')))}'"
                    if step.get("accepted")
                    else "took none"
                )
                steps.append(f"step {number} ({tried or 'no usable flag'}) {took}")
            notes.append(
                f"- {EVENT_TEXT[event]}: context terms "
                f"{_reason_words(about.get('reason'))}. "
                + (
                    f"Tried, against the map so far: {'; '.join(steps)}. "
                    if steps
                    else ""
                )
                + f"Fitted rows that carry each flag: {flags}."
            )
    lines += _table(
        [
            "event",
            "map",
            "g(p)",
            "fitted slot-turns of known",
            "CV NLL no map",
            "logistic",
            "logistic with context terms",
            "folds the terms win",
            "isotonic",
            "folds isotonic wins",
            "choice",
        ],
        rows,
    )
    offered = data.get("context_terms") or {}
    if notes:
        lines += [
            "",
            "Context terms are added one at a time, the one that gains most "
            "first. A term is taken only when the map with it is better out of "
            f"sample than the map so far by more than {offered.get('margin')} "
            "nats per fitted slot-turn and in at least eight of ten folds; a "
            "flag that adds nothing stays out:",
            "",
            *notes,
        ]
    elif offered and not offered.get("offered"):
        lines += ["", "Context terms were not offered (`--no-context-terms`)."]
    rows = []
    for event in C.EVENTS:
        record = fit.get(event) or {}
        inner = record.get("fit") or {}
        rows.append(
            [
                EVENT_TEXT[event],
                record.get("rows_known"),
                _percent(record.get("observed"), 2),
                _percent(record.get("predicted_before"), 2),
                _percent(record.get("predicted_after"), 2),
                _number(record.get("nll_before"), 5),
                _number(record.get("nll_after"), 5),
                _number(inner.get("nll_before"), 5),
                _number(inner.get("nll_after"), 5),
            ]
        )
    lines += [
        "",
        "Stored with the artifact (validation, every slot-turn where the "
        "event is known; the last two columns are the fitted objective):",
        "",
    ]
    lines += _table(
        [
            "event",
            "slot-turns",
            "happened",
            "mean predicted before",
            "after",
            "event NLL before",
            "after",
            "objective before",
            "after",
        ],
        rows,
    )

    if checks:
        fine = checks.get("fine_nll") or {}
        once = checks.get("applied_once") or {}
        lines += [
            "",
            "## Checks on the reloaded artifact",
            "",
            f"- passed: {checks.get('passed')}"
            + (
                f" - FAILED: {'; '.join(checks.get('failures') or [])}"
                if checks.get("failures")
                else ""
            ),
            f"- the reloaded predictor carries the fitted calibration: "
            f"{checks.get('reloaded_calibration_equals_fitted')}; its network, "
            f"temperatures and Elo mode are the input's bit for bit: "
            f"{checks.get('network_and_temperatures_unchanged')}; featurizer: "
            f"{checks.get('featurizer_unchanged')}",
            f"- applied once: the reloaded action probabilities differ from one "
            f"application of the maps to the input's by at most "
            f"{float(once.get('max_abs_difference_to_one_application') or 0.0):.2e} "
            f"(tolerance {once.get('tolerance')}), from two applications by "
            f"{float(once.get('max_abs_difference_to_two_applications') or 0.0):.2e}",
            f"- targets and Mega identical: "
            f"{checks.get('targets_and_mega_untouched')}; "
            f"`features.normalize_prediction` leaves the prediction unchanged: "
            f"{checks.get('normalize_prediction_unchanged')}",
            f"- payload version {checks.get('payload_version')} (the input's: "
            f"{source.get('payload_version')}): the model code from before event "
            "calibration reads version 1 only and refuses this file instead of "
            "serving it uncalibrated",
            f"- fine NLL on validation: {_number(fine.get('before'), 5)} -> "
            f"{_number(fine.get('after'), 5)} "
            f"(change {_number(fine.get('change'), 5)}; "
            f"allowed +{fine.get('tolerance')})",
        ]

    if splits:
        lines += [
            "",
            "## Fine NLL by split",
            "",
            "Mean per labelled slot-turn (`features.slot_nll`). Test and ladder "
            "holdout are informational. The change is after minus before on the "
            "same slot-turns, with a 95% interval from resampling games. The last "
            "two columns split it: slot-turns whose action the log shows, and "
            "slot-turns whose action is hidden (scored by the probability of the "
            "actions the log still allows).",
            "",
        ]
        lines += _table(
            [
                "split",
                "examples",
                "slot-turns",
                "fine before",
                "after",
                "change",
                "visible actions: slot-turns, change",
                "hidden actions: slot-turns, change",
            ],
            [
                [
                    _long(name),
                    found.get("examples"),
                    found["fine_nll"][SIDES[0]].get("slots"),
                    _number(found["fine_nll"][SIDES[0]].get("fine"), 5),
                    _number(found["fine_nll"][SIDES[1]].get("fine"), 5),
                    _change(found["fine_nll"].get("paired_change")),
                    *(
                        f"{found['fine_nll'][SIDES[0]].get(f'{kind}_slots')}: "
                        + _change(found["fine_nll"].get(f"paired_change_{kind}"))
                        for kind in POPULATIONS
                    ),
                ]
                for name, found in splits.items()
            ],
        )
        lines += [
            "",
            "## At the pre-registered thresholds",
            "",
            "Slot-turns where the predicted probability is at least the threshold: "
            "their mean prediction against how often the event happened (95% "
            "interval, games resampled). Gap = mean predicted minus happened, in "
            "percentage points: 0 is calibrated. Each side has its own slot-turns "
            "(a calibrated model fires less often); the last column follows the "
            "slot-turns that fired BEFORE and gives their mean prediction after.",
            "",
        ]
        rows = []
        for name, found in splits.items():
            for event in C.EVENTS:
                for row in found["events"][event]["thresholds"]:
                    first, second = row[SIDES[0]], row[SIDES[1]]
                    same_rows = row["rows_fired_before"]
                    rows.append(
                        [
                            _short(name),
                            EVENT_TEXT[event],
                            f"{row['threshold']:.2f}",
                            first["slots"],
                            _percent(first["predicted"]),
                            _bracket(first["happened"]),
                            _bracket(first["gap"], _points),
                            second["slots"],
                            _percent(second["predicted"]),
                            _bracket(second["happened"]),
                            _bracket(second["gap"], _points),
                            f"{_percent(same_rows['predicted_before'])} -> "
                            f"{_percent(same_rows['predicted_after'])} "
                            f"(happened {_percent(same_rows['happened'])})",
                        ]
                    )
        lines += _table(
            [
                "split",
                "event",
                "P at least",
                "slot-turns before",
                "mean predicted",
                "happened",
                "gap",
                "slot-turns after",
                "mean predicted",
                "happened",
                "gap",
                "the slot-turns that fired before",
            ],
            rows,
        )
        lines += _still_off(
            [
                (
                    f"{_short(name)}, {EVENT_TEXT[event]}, P >= {row['threshold']:g}",
                    row[SIDES[1]],
                )
                for name, found in splits.items()
                for event in C.EVENTS
                for row in found["events"][event]["thresholds"]
            ]
        )
        lines += [
            "",
            "## Rates, Brier score and calibration error by split",
            "",
            "Bias = mean predicted minus observed over every slot-turn where the "
            "event is known, in percentage points (95% interval, games resampled). "
            "ECE = mean absolute gap between predicted and observed rate over the "
            "ten bins of the reliability table; 'by chance' is the ECE a perfectly "
            "calibrated predictor with the same predictions would show on as many "
            "slot-turns (mean, and 95th percentile in brackets).",
            "",
        ]
        rows = []
        for name, found in splits.items():
            for event in C.EVENTS:
                block = found["events"][event]
                for side in SIDES:
                    part = block[side]
                    chance = part.get("ece_if_calibrated") or {}
                    rows.append(
                        [
                            _short(name),
                            EVENT_TEXT[event],
                            side,
                            block["slots"],
                            _percent(block["observed"], 2),
                            _percent(part["predicted"], 2),
                            _bracket(part["bias"], lambda v: _points(v, 2)),
                            _number(part["nll"], 5),
                            _number(part["brier"], 5),
                            _number(part["ece"], 4),
                            f"{_number(chance.get('mean'), 4)} "
                            f"({_number(chance.get('p95'), 4)})",
                        ]
                    )
        lines += _table(
            [
                "split",
                "event",
                "",
                "slot-turns",
                "happened",
                "mean predicted",
                "bias",
                "event NLL",
                "Brier",
                "ECE",
                "ECE by chance",
            ],
            rows,
        )

    if any(found.get("populations") for found in splits.values()):
        lines += [
            "",
            "## Visible and hidden actions: what the calibrated numbers mean",
            "",
            "A slot-turn's action is hidden when the Pokemon never acted: knocked "
            "out before it moved, flinched, asleep, or the game ended. The log "
            "still proves that it did not switch (a switch comes before every "
            "move) and, where it says so, that it did not use a Protect move, so "
            "these slot-turns count as 'did not happen', as in the scorecard. At "
            "the start of a turn nobody knows which kind a slot-turn will be, so a "
            "probability read there has to be right over both kinds together: "
            "that is what the maps are fitted for. Each kind alone is selected on "
            "the outcome (a hidden action is never a switch). The table shows how "
            "much of the miscalibration over all slot-turns comes from the hidden "
            "ones, and that AFTER the fit the visible actions alone happen more "
            "often than predicted. The calibrated number is not the rate among "
            "the slots that get to act.",
            "",
        ]
        rows = []
        for name, found in splits.items():
            for event in C.EVENTS:
                for kind in POPULATIONS:
                    part = ((found.get("populations") or {}).get(event) or {}).get(kind)
                    if not part:
                        continue
                    first = (part.get("thresholds") or [{}])[0]
                    before, after = first.get(SIDES[0]) or {}, first.get(SIDES[1]) or {}
                    rows.append(
                        [
                            _short(name),
                            EVENT_TEXT[event],
                            kind,
                            part.get("slots"),
                            _percent(part.get("share_of_known")),
                            _percent(part.get("observed"), 2),
                            _percent(part.get("predicted_before"), 2),
                            _percent(part.get("predicted_after"), 2),
                            f"{_number(first.get('threshold'), 2)}: "
                            + _said(before, _percent(before.get("happened"))),
                            _said(after, _percent(after.get("happened"))),
                        ]
                    )
        lines += _table(
            [
                "split",
                "event",
                "action",
                "slot-turns",
                "share of known",
                "happened",
                "mean predicted before",
                "after",
                "at P at least, before",
                "after",
            ],
            rows,
        )

    if any(found.get("context") for found in splits.values()):
        lines += [
            "",
            "## By public context: turn 1, first turn on the field, "
            "protected last turn",
            "",
            "One map over all slot-turns can be right on average and wrong for a "
            "part of them, so every split is read again by three public flags of "
            "the slot (the count table's three). A guard that fires on the lead "
            "turn reads the 'turn 1' rows. Slot-turns where the event is known; "
            "at the first pre-registered threshold the mean prediction against "
            "what happened, with the gap (predicted minus happened, percentage "
            "points, 95% interval over resampled games). The other thresholds "
            f"are in `{REPORT_JSON}`. Thin rows are noise.",
            "",
        ]
        rows = []
        for name, found in splits.items():
            for event in C.EVENTS:
                groups = (found.get("context") or {}).get(event) or {}
                for label, part in groups.items():
                    first = (part.get("thresholds") or [{}])[0]
                    before, after = first.get(SIDES[0]) or {}, first.get(SIDES[1]) or {}
                    means = [
                        _percent((part.get(side) or {}).get("predicted"), 2)
                        for side in SIDES
                    ]
                    rows.append(
                        [
                            _short(name),
                            EVENT_TEXT[event],
                            label,
                            part.get("slots"),
                            _percent(part.get("observed"), 2),
                            " -> ".join(means),
                            f"{_number(first.get('threshold'), 2)}: "
                            + _said(before, _bracket(before.get("happened"))),
                            _bracket(before.get("gap"), _points),
                            _said(after, _bracket(after.get("happened"))),
                            _bracket(after.get("gap"), _points),
                        ]
                    )
        lines += _table(
            [
                "split",
                "event",
                "context",
                "slot-turns",
                "happened",
                "mean predicted before -> after",
                "at P at least, before",
                "gap before",
                "after",
                "gap after",
            ],
            rows,
        )
        lines += _still_off(
            [
                (
                    f"{_short(name)}, {EVENT_TEXT[event]}, {label}, "
                    f"P >= {row['threshold']:g}",
                    row[SIDES[1]],
                )
                for name, found in splits.items()
                for event in C.EVENTS
                for label, part in (
                    (found.get("context") or {}).get(event) or {}
                ).items()
                for row in part.get("thresholds") or []
            ]
        )

    if any((found.get("coverage") or {}).get("slices") for found in splits.values()):
        lines += [
            "",
            "## Side effect: joint reply coverage",
            "",
            "How often the reply a player made with both slots is among the "
            "joint replies ranked highest (`oppmodel.joint`, counted as in the "
            "scorecard: turns where every slot on the field shows its whole "
            "action, a reply through the OTHER bucket a miss, without the Mega "
            "bit). Those turns hold more switches and Protect moves than turn "
            "starts do (a Pokemon knocked out before it moved is not among "
            "them), and the calibration takes mass from exactly those two, so "
            "the calibrated distribution covers slightly fewer of them. Read "
            "the calibrated scalars in a guard; a reply list for a search is "
            "not better for the calibration. Change = after minus before on the "
            "same examples, 95% interval over resampled games.",
            "",
        ]
        rows = []
        for name, found in splits.items():
            coverage = found.get("coverage") or {}
            k = coverage.get("k")
            for label, part in (coverage.get("slices") or {}).items():
                first, second = part.get(SIDES[0]) or {}, part.get(SIDES[1]) or {}
                rows.append(
                    [
                        _short(name),
                        label,
                        part.get("examples"),
                        f"{_percent(first.get('top'), 2)} -> "
                        f"{_percent(second.get('top'), 2)}",
                        _bracket(
                            _as_value(part.get("top_change")), lambda v: _points(v, 2)
                        ),
                        f"{_percent(first.get('top1'), 2)} -> "
                        f"{_percent(second.get('top1'), 2)}",
                        f"{_number(first.get('log_prob'), 4)} -> "
                        f"{_number(second.get('log_prob'), 4)}",
                        _change(part.get("log_prob_change")),
                        k,
                    ]
                )
        lines += _table(
            [
                "split",
                "slice",
                "examples",
                "top-K before -> after",
                "change, points",
                "top-1 before -> after",
                "mean log p of the true reply",
                "change",
                "K",
            ],
            rows,
        )

    if any(found.get("attack") for found in splits.values()):
        lines += [
            "",
            "## Side effect: the attack events",
            "",
            "The mass taken from switch and Protect goes to the other moves, so "
            "the probability that a slot attacks a given opposing slot rises with "
            "it. The attack label exists only where the click is visible, and "
            "among visible actions switches and Protect moves are over-represented "
            "(they are always visible, a Pokemon knocked out before it moved is "
            "not). Against that label the attack probabilities read higher after "
            "the fit. Whether they are nearer the click rate over all turn starts "
            "cannot be measured: hidden actions have no attack label. Rows: two "
            "opposing Pokemon on the field, as the scorecard counts them; 95% "
            "intervals from resampling games.",
            "",
        ]
        rows = []
        for name, found in splits.items():
            for event, block in (found.get("attack") or {}).items():
                first, second = block[SIDES[0]], block[SIDES[1]]
                cells = [
                    _short(name),
                    data_text(S.EVENT_TEXT, event),
                    block["slots"],
                    _percent(block["observed"], 2),
                    f"{_percent(first['predicted'], 2)} -> "
                    f"{_percent(second['predicted'], 2)}",
                    f"{_number(first['brier'], 5)} -> {_number(second['brier'], 5)}",
                ]
                for row in block["thresholds"]:
                    cells.append(
                        " -> ".join(
                            _said(row[side], _bracket(row[side]["happened"]))
                            for side in SIDES
                        )
                    )
                rows.append(cells)
        lines += _table(
            [
                "split",
                "event",
                "slot-turns",
                "happened",
                "mean predicted",
                "Brier",
                *(
                    f"at P >= {threshold:g}: before -> after"
                    for threshold in ATTACK_THRESHOLDS
                ),
            ],
            rows,
        )

    shift = data.get("shift") or {}
    if any((shift.get(event) or {}).get("ladder_remaining") for event in C.EVENTS):
        lines += [
            "",
            "## Ladder holdout: what a map fitted on human validation cannot fix",
            "",
            "The maps are fitted on human validation games, so they remove the "
            "miscalibration as it shows there. Our ladder opponents are another "
            "population: over all slot-turns they switch less and protect more "
            "than the human corpus (observed rates below), and a map fitted on "
            "validation cannot see that. What is left on the ladder holdout after "
            "the fit is the part such a map does not remove:",
            "",
        ]
        for event in C.EVENTS:
            entry = shift.get(event) or {}
            left = entry.get("ladder_remaining")
            if not left:
                continue
            rates = ", ".join(
                f"{_short(name)} {_percent(row.get('observed'), 2)}"
                for name, row in (entry.get("splits") or {}).items()
            )
            lines += [
                f"- {EVENT_TEXT[event]}: happened {rates}. On the ladder holdout "
                f"the mean prediction is off by {_points(left.get('bias_before'), 2)} "
                f"points before and {_points(left.get('bias_after'), 2)} after "
                f"({_percent(left.get('bias_share_left'), 0)} of the bias is left); "
                f"ECE {_number(left.get('ece_before'), 4)} -> "
                f"{_number(left.get('ece_after'), 4)} "
                f"({_percent(left.get('ece_share_left'), 0)} left), where a calibrated "
                f"predictor would show {_number(left.get('ece_chance_after'), 4)} by "
                "chance on this many slot-turns."
            ]
        table = data.get("count_table")
        if table:
            quoted = "; ".join(
                f"{EVENT_TEXT[event]}: predicted "
                f"{_percent(found.get('predicted'))}, observed "
                f"{_percent(found.get('observed'))}"
                for event, found in (table.get("events") or {}).items()
            )
            lines += [
                "",
                f"The count table `{table.get('predictor')}` of "
                f"`{table.get('scorecard')}` carries the human corpus's rates to "
                f"the same ladder slot-turns and shows the shift without any "
                f"neural model: {quoted}.",
            ]
            for event, found in (table.get("events") or {}).items():
                left = (shift.get(event) or {}).get("ladder_remaining") or {}
                mine = left.get("bias_after")
                if mine is None or found.get("predicted") is None:
                    continue
                theirs = float(found["predicted"]) - float(found["observed"])
                if mine * theirs < 0:
                    lines += [
                        "",
                        f"For '{EVENT_TEXT[event]}' the calibrated model is off by "
                        f"{_points(mine, 2)} points where the count table is off by "
                        f"{_points(theirs, 2)}: the other sign, so what is left there "
                        "is not the difference in base rates.",
                    ]

    for name, found in splits.items():
        lines += ["", f"## Reliability: {_long(name)}"]
        for event in C.EVENTS:
            block = found["events"][event]
            lines += [
                "",
                f"{EVENT_TEXT[event]} ({block['slots']} slot-turns, "
                f"{block['impossible']} of them cannot happen by the mask):",
                "",
            ]
            rows = []
            for first, second in zip(
                block[SIDES[0]]["reliability"], block[SIDES[1]]["reliability"]
            ):
                rows.append(
                    [
                        f"{first['from']:.1f}-{first['to']:.1f}",
                        first["slots"],
                        _percent(first["predicted"]),
                        _percent(first["observed"]),
                        second["slots"],
                        _percent(second["predicted"]),
                        _percent(second["observed"]),
                    ]
                )
            lines += _table(
                [
                    "predicted",
                    "slot-turns before",
                    "mean predicted",
                    "happened",
                    "slot-turns after",
                    "mean predicted",
                    "happened",
                ],
                rows,
            )
    return "\n".join(lines).rstrip("\n") + "\n"


def write_report(out_dir: Path, report: Mapping[str, Any]) -> None:
    data = _plain(report)
    (out_dir / REPORT_JSON).write_text(json.dumps(data, indent=1), encoding="utf-8")
    (out_dir / REPORT_MD).write_text(render_report(data), encoding="utf-8")


def render_only(out_dir: Path) -> None:
    """Rewrite the Markdown report from the JSON one. Raises ``CalibrationError``."""
    try:
        data = json.loads((out_dir / REPORT_JSON).read_text(encoding="utf-8"))
        # A report whose text is already on disk is left alone, time included.
        S.write_if_changed(out_dir / REPORT_MD, render_report(data))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CalibrationError(
            f"cannot render {out_dir / REPORT_JSON}: {exc!r}"
        ) from exc


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        if exc.code in (0, None):
            raise
        say(f"{FAILED} bad arguments")
        return 2
    try:
        if args.render_only:
            render_only(_resolve(args.out))
            say(f"{DONE} rendered {_resolve(args.out) / REPORT_MD}")
            return 0
        report = calibrate(args)
    except Exception as exc:
        reason = " ".join(f"{type(exc).__name__}: {exc}".split())
        say(f"{FAILED} {reason}")
        return 1
    fine = report["checks"]["fine_nll"]
    say(
        f"{DONE} name {report['name']} maps "
        + " ".join(
            f"{event}={found['map']}"
            for event, found in report["calibration"]["maps"].items()
        )
        + f" val_fine {_number(fine['before'])} -> {_number(fine['after'])} "
        f"artifact {report['artifact']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
