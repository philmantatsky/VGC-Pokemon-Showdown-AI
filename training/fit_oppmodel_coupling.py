"""Fit the pair coupling of an opponent-predictor artifact and store it.

    nice -n 19 .venv/bin/python training/fit_oppmodel_coupling.py \\
        --artifact results_oppmodel/<run>/artifact.pt \\
        --dataset results_oppmodel/<the build it was trained on> \\
        --out results_oppmodel/<a new directory>

The coupling (``vgc_bench/src/oppmodel/coupling.py``) is one number per pair of
reply classes and context bucket that ``joint.joint_replies`` multiplies into
the product of the two slots' marginals. It changes no per-slot number:
``predict`` of the new artifact is the source's, array for array.

WHAT IS FITTED ON WHAT. Validation rows of the dataset the artifact was
trained on, without the battles of a ladder-holdout opponent (the rule of the
epoch choice and of the event calibration), rows with two acting slots whose
log shows a free choice for both (a hidden slot counts through its censored
set). Weighted by ``m_weight`` (the account cap the trainer's loss uses); the
unweighted table is reported beside it. Nothing else is read before the
verdict.

THE ACCEPTANCE RULE. 10 folds of whole battles; the score of a fold is the
mean gain in log-probability of the consistent pair set over today's joint on
its held-out rows, in nats per two-slot row. THE COUPLING IS TAKEN ONLY IF
THAT GAIN IS ABOVE 0.002 IN AT LEAST 8 OF THE 10 FOLDS. Otherwise the script
writes its report, prints ``CPL_NOT_TAKEN`` and writes no artifact. The
shrinkage ``kappa`` is the grid value with the best mean fold score (ties to
the larger) and bucket tables replace the pooled table only when they beat it
by the same margin in 8 of 10 folds; both choices are made on the same folds,
so every grid value's folds are in the report.

WHEN THE RULE WAS FIXED, exactly. In
``results_oppmodel/analysis_20261010/design_joint.md`` at 01:04 on 2026-10-10,
before any fit, worded "kept only if the cross-validated gain exceeds 0.002
nats per row and is positive in at least 8 of 10 folds". This script was
written with the per-fold reading above (each of 8 folds above 0.002), and
that reading is what OPPONENT_PREDICTOR.md states; its line was written at
02:23, AFTER this script's first fits (01:46 to 01:54), so it is the design
file that precedes the result, not that line. The two readings are not the
same rule (neither implies the other in every case); the report gives both
(``acceptance.design_reading``) and the verdict is the per-fold one.

A run whose rule or fit settings are not those fixed there (``--margin``,
``--fold-share``, ``--folds``, ``--kappa-grid``, ``--seed``, ``--buckets``)
says so in its report (``acceptance.as_fixed`` false, the departures named)
and writes NO artifact unless ``--allow-other-rule`` is given: a verdict
under a looser rule never becomes a file by accident.

Reported, not gated: top-1/4/8/16 coverage of the true pair and its mean
log-probability on the fully seen turns, plain against coupled, with the
paired change and a 95% interval from resampling whole games, overall and for
replies that hold a switch / a Protect-family move / moves only; the drift of
the class marginals; the pair events before and after. Informational, only
when asked (``--report-splits ladder_holdout``): the same on the ladder
holdout, plus the pair-set gain on its two-slot rows. The split ``test`` is
never read by this script. The bot's own games from
``features.OWN_SEALED_FROM`` on are the sealed confirmation set: what a
dataset HOLDS is read from its rows' own times (``features.own_time_range``,
never from the day it was built, which a manifest may not even give), and
the rows of those games (and of games of unknown time) are left out of the
ladder holdout, counted in the report, unless ``--allow-fresh-holdout`` is
given.

With a verdict of taken the artifact is written as ``artifact.pt.unverified``,
reloaded, checked (see ``verify``) and only then renamed. Outputs:
``artifact.pt`` (taken only), ``coupling_report.json``, ``coupling_report.md``.
The last line is the result a caller reads:

    CPL_DONE               taken, the artifact written and verified
    CPL_TAKEN_NO_ARTIFACT  taken, and no artifact written (``--no-artifact``,
                           or a rule other than the fixed one without
                           ``--allow-other-rule``)
    CPL_NOT_TAKEN          not taken, no artifact
    CPL_FAILED <reason>    a refusal or a failed check

``report['status']`` holds the same word and never contradicts
``report['verdict']``.

Two counts of "fully seen". The fit calls a two-slot row fully seen when each
slot's consistent set is ONE reply; the coverage tables count a turn when
every acting slot shows its whole action (``joint.true_replies``' ``visible``:
a move needs a logged target). A move outside the candidates has no target
split, so it is one reply for the fit even without a logged target, and it is
not a counted turn: the report gives both numbers under their own names.

One thread, CPU. It is still a process: not while ladder games are played.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from collections.abc import Mapping, Sequence  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from evaluation import oppmodel_scorecard as S  # noqa: E402
from vgc_bench.src.oppmodel import artifact as A  # noqa: E402
from vgc_bench.src.oppmodel import coupling as C  # noqa: E402
from vgc_bench.src.oppmodel import features as F  # noqa: E402
from vgc_bench.src.oppmodel import joint as J  # noqa: E402
from vgc_bench.src.oppmodel.events import INTENT_PROTECT  # noqa: E402

FORMAT = "oppmodel-coupling-report"
VERSION = 1
ARTIFACT_NAME = "artifact.pt"
UNVERIFIED_SUFFIX = ".unverified"
REPORT_JSON = "coupling_report.json"
REPORT_MD = "coupling_report.md"
SPLIT_VAL = "val"
SPLIT_LADDER = "ladder_holdout"
REPORT_SPLITS: tuple[str, ...] = (SPLIT_LADDER,)  # ``test`` is never offered
FLAG_HOLDOUT_BATTLE = 4  # m_flag bit: the battle has a ladder-holdout opponent
KS: tuple[int, ...] = (1, 4, 8, 16)
K_MAIN = 8
DONE, NOT_TAKEN, FAILED = "CPL_DONE", "CPL_NOT_TAKEN", "CPL_FAILED"
TAKEN_NO_ARTIFACT = "CPL_TAKEN_NO_ARTIFACT"  # taken, and no file was written
# Where and when the acceptance rule was fixed (see the module text).
RULE_FIXED = (
    "fixed before any fit in results_oppmodel/analysis_20261010/design_joint.md "
    "(2026-10-10 01:04); OPPONENT_PREDICTOR.md states it since 02:23, after the "
    "first fits"
)
DESIGN_WORDING = (
    "the cross-validated gain exceeds the margin and is positive in at least "
    "fold_share of the folds"
)

SLICE_ALL = "all counted turns"
SLICE_TWO = "two acting slots"
SLICE_SWITCH = "reply holds a switch"
SLICE_PROTECT = "reply holds a Protect-family move"
SLICE_MOVES = "moves only (no switch, no Protect-family move)"
TWO_PREFIX = "two acting slots, "


class CouplingError(RuntimeError):
    """A refusal or a failed check: the run ends ``CPL_FAILED``."""


def say(text: str) -> None:
    print(text, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def plain(value: Any) -> Any:
    """JSON-ready data: arrays as lists, numpy scalars as numbers, NaN as None."""
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _tensor(value: Any) -> Any:
    return value


def same(first: Any, second: Any) -> bool:
    """Deep equality of two decoded artifact values (dicts, lists, arrays,
    tensors, scalars), dtype and shape included."""
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        return set(first) == set(second) and all(
            same(first[key], second[key]) for key in first
        )
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        return len(first) == len(second) and all(
            same(a, b) for a, b in zip(first, second)
        )
    if hasattr(first, "detach") and hasattr(second, "detach"):
        one, two = _tensor(first), _tensor(second)
        a, b = one.detach().cpu().numpy(), two.detach().cpu().numpy()
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
        return (
            first.dtype == second.dtype
            and first.shape == second.shape
            and first.tobytes() == second.tobytes()
        )
    return type(first) is type(second) and first == second


# --- data -----------------------------------------------------------------------


def read_manifest(dataset: Path) -> tuple[dict[str, Any], str]:
    path = dataset / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CouplingError(f"dataset {dataset}: {exc}") from exc
    return manifest, sha256_file(path)


def load_split(dataset: Path, name: str, own_before: int | None = None) -> F.Batch:
    """One split of a built dataset. ``test`` is refused. ``own_before``
    (Unix seconds) leaves out the rows of the bot's own games played at or
    after it, and of unknown time (``features.load_dataset``)."""
    if name not in (SPLIT_VAL, "train", *REPORT_SPLITS):
        raise CouplingError(f"split {name!r} is not read by this script")
    try:
        if own_before is None:
            batch, _ = F.load_dataset(dataset, splits=[name])
        else:
            batch, _ = F.load_dataset(dataset, splits=[name], own_before=own_before)
    except (OSError, ValueError, KeyError) as exc:
        raise CouplingError(f"dataset {dataset}, split {name}: {exc!r}") from exc
    return batch


def sealed_own_games(dataset: Path) -> dict[str, Any]:
    """What the dataset holds of the bot's own games from the sealed day on.

    ``features.own_time_range`` at ``features.OWN_SEALED_FROM``: read from the
    rows' own times (meta arrays only), never from the day the dataset was
    built. ``late_rows`` 0 means its ladder holdout is the old one;
    ``time_known`` False that its rows carry no times at all, so nothing can
    be told (the caller refuses). ``cut`` is the day as Unix seconds.
    """
    try:
        cut = F.local_time(F.OWN_SEALED_FROM)
        found = dict(F.own_time_range(dataset, cut))
    except (OSError, ValueError, KeyError) as exc:
        raise CouplingError(
            f"dataset {dataset}: the times of its own games cannot be read "
            f"({exc!r}); its ladder holdout may hold the sealed games"
        ) from exc
    return {
        "sealed_from": F.OWN_SEALED_FROM,
        "cut": int(cut),
        "time_known": bool(found.get("has_time", True)),
        "late_rows": int(found.get("late_rows") or 0),
        "splits": found.get("splits") or {},
    }


def rule_departures(args: argparse.Namespace) -> list[str]:
    """The arguments that differ from the rule and fit settings fixed before
    the first fit (the module text): empty for the fixed rule."""
    fixed: dict[str, Any] = {
        "margin": C.DEFAULT_MARGIN,
        "fold_share": C.DEFAULT_FOLD_SHARE,
        "folds": C.DEFAULT_FOLDS,
        "seed": C.DEFAULT_SEED,
        "buckets": C.BUCKET_SCHEME,
        "kappa_grid": sorted(C.DEFAULT_KAPPAS),
    }
    given = {**vars(args), "kappa_grid": sorted(float(k) for k in args.kappa_grid)}
    return [
        f"{name} {given[name]!r} (fixed: {value!r})"
        for name, value in fixed.items()
        if given[name] != value
    ]


def without_holdout_battles(batch: F.Batch) -> tuple[F.Batch, int]:
    """(the batch without battles of a ladder-holdout opponent, rows left out)."""
    if "m_flag" not in batch:
        return batch, 0
    marked = (np.asarray(batch["m_flag"]).astype(np.int64) & FLAG_HOLDOUT_BATTLE) > 0
    if not marked.any():
        return batch, 0
    return F.take(batch, ~marked), int(marked.sum())


def predict_features(predictor: Any, batch: F.Batch, name: str) -> S.Prediction:
    """The predictor on the features alone (labels and bookkeeping stripped,
    unknown sheets read as closed); a fallback to uniform is an error."""
    try:
        pred, _ = S.guarded_predict(predictor, batch, name=name)
    except S.ScorecardError as exc:
        raise CouplingError(str(exc)) from exc
    return pred


def fit_rows(
    batch: F.Batch, pred: Mapping[str, np.ndarray], move_intent: np.ndarray
) -> dict[str, Any]:
    """The rows the fit reads and their masses.

    ``rows`` marks the examples with two acting slots, a non-empty consistent
    set for each and positive model mass on it; ``model`` / ``seen`` are the
    class-pair masses of those rows (``coupling.pair_masses``), ``weight``
    their ``m_weight``, ``bucket`` and ``battle`` theirs. ``skipped`` counts
    the rest by reason. ``fully_seen``: each slot's consistent set is one
    reply (the fit's meaning); ``counted``: the coverage tables count the turn
    (``joint.true_replies``' ``visible``). The second is the narrower one: a
    move outside the candidates without a logged target is one reply for the
    fit and not a counted turn.
    """
    features = S.features_only(batch)
    prob, legal = J.slot_replies(pred, features)
    if prob.shape[0] != np.asarray(batch["turn"]).shape[0]:
        raise CouplingError(f"slot replies cannot be read ({dict(J.COUNTERS)})")
    classes = C.reply_classes(features, move_intent)
    consistent = C.consistent_replies(batch) & legal
    masses = C.pair_masses(prob, classes, consistent)
    both = (np.asarray(batch["act_mon"]) >= 0).all(-1)
    has_set = consistent.any(-1).all(-1)
    positive = (masses["seen"].sum((1, 2)) > 0) & (masses["kept"] > 0)
    rows = both & has_set & positive
    single = consistent.sum(-1) == 1
    truth = J.true_replies(batch)
    visible = (
        np.asarray(truth["visible"]).astype(bool)
        if truth
        else np.zeros(rows.shape[0], dtype=bool)
    )
    weight = (
        np.asarray(batch["m_weight"], dtype=np.float64)
        if "m_weight" in batch
        else np.ones(rows.shape[0])
    )
    return {
        "rows": rows,
        "model": masses["model"][rows],
        "seen": masses["seen"][rows],
        "weight": weight[rows],
        "bucket": C.reply_buckets(features)[rows],
        "battle": np.asarray(batch["m_battle"])[rows],
        "fully_seen": (single.all(-1))[rows],
        "counted": visible[rows],
        "skipped": {
            "one_acting_slot": int((~both).sum()),
            "no_free_choice_shown": int((both & ~has_set).sum()),
            "no_model_mass_on_the_set": int((both & has_set & ~positive).sum()),
        },
    }


# --- reports --------------------------------------------------------------------


def coverage_slices(truth: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Masks over the counted examples (``truth`` already cut to them)."""
    two = np.asarray(truth["active"]).astype(bool).all(-1)
    switch = np.asarray(truth["switch"]).astype(bool).any(-1)
    protect = np.asarray(truth[INTENT_PROTECT]).astype(bool).any(-1)
    moves = ~switch & ~protect
    out = {
        SLICE_ALL: np.ones(two.shape, dtype=bool),
        SLICE_TWO: two,
        SLICE_SWITCH: switch,
        SLICE_PROTECT: protect,
        SLICE_MOVES: moves,
    }
    for name, mask in ((SLICE_SWITCH, switch), (SLICE_PROTECT, protect)):
        out[TWO_PREFIX + name] = two & mask
    out[TWO_PREFIX + SLICE_MOVES] = two & moves
    return out


def coverage_report(
    batch: F.Batch,
    pred: Mapping[str, np.ndarray],
    coupling: C.PairCoupling,
    move_intent: np.ndarray,
    *,
    resamples: int,
    seed: int,
    floor: float = F.PROBABILITY_FLOOR,
) -> dict[str, Any]:
    """Top-K coverage and log-probability of the true pair on the fully seen
    turns, plain against coupled, with the paired change per slice.

    A true reply through OTHER is a miss at every K (the scorecard's rule).
    ``change`` is coupled minus plain with a 95% interval from resampling
    whole games.
    """
    truth = J.true_replies(batch)
    if not truth:
        raise CouplingError(f"the labels cannot be read ({dict(J.COUNTERS)})")
    visible = np.asarray(truth["visible"]).astype(bool)
    part = F.take(batch, visible)
    masks = S.features_only(part)
    told = {name: np.asarray(values)[visible] for name, values in truth.items()}
    own = {key: np.asarray(value)[visible] for key, value in pred.items()}
    count = int(visible.sum())
    other = told["other"].any(-1)
    battle = np.asarray(part["m_battle"])
    width = max(KS)
    made = {
        "plain": J.joint_replies(own, masks, k=width, truth=told),
        "coupled": J.joint_replies(
            own, masks, k=width, truth=told, coupling=coupling, move_intent=move_intent
        ),
    }
    for name, found in made.items():
        if found.n != count:
            raise CouplingError(
                f"the {name} joint cannot be built ({dict(J.COUNTERS)}, "
                f"{dict(C.COUNTERS)})"
            )
    hits = {
        name: {k: (found.rank >= 0) & (found.rank < k) & ~other for k in KS}
        for name, found in made.items()
    }
    logs = {
        name: np.log(np.maximum(found.prob_true, floor)) for name, found in made.items()
    }
    out: dict[str, Any] = {
        "examples": int(visible.shape[0]),
        "counted": count,
        "games": int(np.unique(battle).size),
        "involves_other": int(other.sum()),
        "tilt_mean_two_slots": None,
        "slices": {},
    }
    two = told["active"].all(-1)
    if two.any():
        out["tilt_mean_two_slots"] = float(made["coupled"].tilt[two].mean())
    for label, mask in coverage_slices(told).items():
        size = int(mask.sum())
        row: dict[str, Any] = {
            "examples": size,
            "games": int(np.unique(battle[mask]).size),
        }
        for name in made:
            row[name] = {
                "top": {
                    str(k): float(hits[name][k][mask].mean()) if size else None
                    for k in KS
                },
                "log_prob": float(logs[name][mask].mean()) if size else None,
            }
        row["change"] = {
            f"top{k}": S.paired_difference(
                hits["coupled"][k],
                hits["plain"][k],
                mask,
                battle,
                resamples=resamples,
                seed=seed,
            )
            for k in KS
        }
        row["change"]["log_prob"] = S.paired_difference(
            logs["coupled"], logs["plain"], mask, battle, resamples=resamples, seed=seed
        )
        out["slices"][label] = row
    return out


def pair_set_gain(
    rows: Mapping[str, Any], table: np.ndarray, *, resamples: int, seed: int
) -> dict[str, Any]:
    """The fit's own measure on a set of rows: mean gain in log-probability of
    the consistent pair set, nats per two-slot row, games resampled."""
    gain = C.table_gain(table, rows["model"], rows["seen"], rows["bucket"])
    everything = np.ones(gain.shape[0], dtype=bool)
    out = S.paired_difference(
        gain,
        np.zeros_like(gain),
        everything,
        rows["battle"],
        resamples=resamples,
        seed=seed,
    )
    weight = np.asarray(rows["weight"], dtype=np.float64)
    out["weighted"] = (
        float((gain * weight).sum() / weight.sum()) if weight.sum() > 0 else None
    )
    seen = np.asarray(rows["fully_seen"], dtype=bool)
    out["fully_seen_rows"] = int(seen.sum())
    out["fully_seen"] = float(gain[seen].mean()) if seen.any() else None
    out["partly_hidden"] = float(gain[~seen].mean()) if (~seen).any() else None
    return out


def marginal_drift(rows: Mapping[str, Any], table: np.ndarray) -> dict[str, Any]:
    """How far the coupled joint's class marginals move from the model's own.

    Per row and class ``| sum_d coupled(c, d) - sum_d model(c, d) |`` for slot
    a (and the same over columns for slot b): mean and 99th percentile.
    """
    model = np.asarray(rows["model"], dtype=np.float64)
    if model.shape[0] == 0:
        return {"rows": 0, "mean": None, "p99": None, "max": None}
    tilt = table[np.asarray(rows["bucket"]).astype(np.int64)]
    coupled = model * tilt
    coupled /= coupled.sum((1, 2), keepdims=True)
    moved = np.concatenate(
        [np.abs(coupled.sum(2) - model.sum(2)), np.abs(coupled.sum(1) - model.sum(1))],
        axis=1,
    )
    by_class = 0.5 * (
        np.abs(coupled.sum(2) - model.sum(2)).mean(0)
        + np.abs(coupled.sum(1) - model.sum(1)).mean(0)
    )
    return {
        "rows": int(model.shape[0]),
        "mean": float(moved.mean()),
        "p99": float(np.quantile(moved, 0.99)),
        "max": float(moved.max()),
        "mean_by_class": dict(zip(C.CLASSES, by_class.tolist())),
    }


def pair_events(
    rows: Mapping[str, Any], table: np.ndarray, full: np.ndarray | None = None
) -> dict[str, Any]:
    """The scorecard's pair events on the fit rows, before and after.

    ``observed`` is the weighted share of the consistent mass on the event's
    cells (each row's ``seen`` renormalised: exact for a fully seen row, the
    model's own split inside the set for a hidden one), ``model`` today's
    joint's mass on them, ``coupled`` the coupled joint's. With ``full`` (the
    raw tilt T of every bucket, ``coupling.full_tilt``: what the fit
    estimated before its per-class part D was discarded; never stored) each
    event also holds ``full_tilt`` and ``ratio_full_tilt``: T reproduces the
    events on its fit rows, the stored table need not, and the difference is
    what discarding D costs.
    """
    model = np.asarray(rows["model"], dtype=np.float64)
    seen = np.asarray(rows["seen"], dtype=np.float64)
    weight = np.asarray(rows["weight"], dtype=np.float64)
    total = float(weight.sum())
    if model.shape[0] == 0 or total <= 0:
        return {}
    tilt = table[np.asarray(rows["bucket"]).astype(np.int64)]
    coupled = model * tilt
    coupled /= coupled.sum((1, 2), keepdims=True)
    shown = seen / seen.sum((1, 2), keepdims=True)
    shown_after = seen * tilt
    shown_after /= shown_after.sum((1, 2), keepdims=True)
    whole = shown_whole = None
    if full is not None:
        raw = np.asarray(full)[np.asarray(rows["bucket"]).astype(np.int64)]
        whole = model * raw
        whole /= whole.sum((1, 2), keepdims=True)
        shown_whole = seen * raw
        shown_whole /= shown_whole.sum((1, 2), keepdims=True)
    out: dict[str, Any] = {}
    for label, cells in S.PAIR_EVENTS.items():
        pick = np.zeros((C.N_CLASS, C.N_CLASS), dtype=bool)
        for a, b in cells:
            pick[a, b] = True
        observed = float((weight @ shown[:, pick].sum(-1)) / total)
        after = float((weight @ shown_after[:, pick].sum(-1)) / total)
        base = float((weight @ model[:, pick].sum(-1)) / total)
        tilted = float((weight @ coupled[:, pick].sum(-1)) / total)
        out[label] = {
            "observed": observed,
            "model": base,
            "ratio_before": observed / base if base > 0 else None,
            "observed_under_coupling": after,
            "coupled": tilted,
            "ratio_after": after / tilted if tilted > 0 else None,
        }
        if whole is not None and shown_whole is not None:
            kept = float((weight @ whole[:, pick].sum(-1)) / total)
            seen_whole = float((weight @ shown_whole[:, pick].sum(-1)) / total)
            out[label]["full_tilt"] = kept
            out[label]["ratio_full_tilt"] = seen_whole / kept if kept > 0 else None
    return out


# --- the run --------------------------------------------------------------------


def verify(
    source_document: Mapping[str, Any],
    source: Any,
    path: Path,
    made: C.PairCoupling,
    validation: F.Batch,
    before: Mapping[str, np.ndarray],
    in_sample: float,
    sample: int = 2000,
) -> dict[str, Any]:
    """Reload the written file and check it. Returns the checks and ``ok``."""
    stored = A.read_artifact(path, strict=True)
    again = A.load_predictor(path, strict=True)
    after = predict_features(again.predictor, validation, "reloaded")
    move_intent = again.featurizer.tables.move_intent
    n = int(np.asarray(validation["turn"]).shape[0])
    pick = np.arange(min(n, max(1, int(sample))))
    part = S.features_only(F.take(validation, pick))
    old = {key: np.asarray(value)[pick] for key, value in before.items()}
    new = {key: np.asarray(value)[pick] for key, value in after.items()}
    names = ("reply", "mega", "prob", "size", "kept", "ok", "tilt")
    plain_old = J.joint_replies(old, part, k=K_MAIN)
    plain_new = J.joint_replies(new, part, k=K_MAIN)
    direct = J.joint_replies(
        old,
        part,
        k=K_MAIN,
        coupling=made,
        move_intent=source.featurizer.tables.move_intent,
    )
    loaded = J.joint_replies(
        new, part, k=K_MAIN, coupling=again.coupling, move_intent=move_intent
    )
    full = J.joint_replies(
        new, part, k=J.MAX_K, coupling=again.coupling, move_intent=move_intent
    )
    first_switch = J.other_reply(full.n_cand) + 1
    a, b = full.reply[..., 0], full.reply[..., 1]
    same_bench = (a >= first_switch) & (a == b)
    sums = full.prob.sum(1)
    checks: dict[str, Any] = {
        "artifact_version": stored.get("version"),
        "version_is_coupled": stored.get("version") == A.VERSION_COUPLED,
        "reloaded_coupling_equals_fitted": bool(
            again.coupling is not None and again.coupling.same_as(made)
        ),
        "predictor_payload_unchanged": same(
            source_document["predictor"], stored["predictor"]
        ),
        "featurizer_unchanged": same(
            source_document["featurizer"], stored["featurizer"]
        ),
        "dex_signature_unchanged": same(
            source_document["dex_signature"], stored["dex_signature"]
        ),
        "predict_unchanged": all(
            np.array_equal(np.asarray(before[key]), np.asarray(after[key]))
            for key in ("action", "target", "mega")
        ),
        "plain_joint_unchanged": plain_old.n == pick.size
        and all(
            np.array_equal(getattr(plain_old, name), getattr(plain_new, name))
            for name in names
        ),
        "coupled_joint_is_one_application": direct.n == pick.size
        and all(
            np.array_equal(getattr(direct, name), getattr(loaded, name))
            for name in names
        ),
        "coupled_lists_sum_to_one": bool(
            full.n == pick.size and np.allclose(sums[full.ok], 1.0, atol=1e-9)
        ),
        "no_same_bench_double_switch": bool(
            full.n == pick.size and not same_bench.any()
        ),
        "in_sample_gain_not_negative": bool(in_sample >= -1e-12),
        "rows_checked_for_the_joint": int(pick.size),
    }
    checks["ok"] = all(
        bool(value)
        for key, value in checks.items()
        if key not in ("artifact_version", "rows_checked_for_the_joint")
    )
    return checks


def run(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    started = time.time()
    source = Path(args.artifact)
    dataset = Path(args.dataset)
    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise CouplingError(f"{out_dir} exists and is not empty; give a new directory")
    for name in args.report_splits:
        if name not in REPORT_SPLITS:
            raise CouplingError(f"--report-splits {name!r}: not one of {REPORT_SPLITS}")
    manifest, manifest_sha = read_manifest(dataset)
    created = str(manifest.get("created") or "")
    # The sealed own games: decided from the rows' times before anything of
    # the ladder holdout is loaded, and only when that split is asked for.
    own: dict[str, Any] | None = None
    own_before: int | None = None
    if SPLIT_LADDER in args.report_splits:
        own = sealed_own_games(dataset)
        own["read_whole"] = bool(args.allow_fresh_holdout)
        if not own["time_known"] and not args.allow_fresh_holdout:
            raise CouplingError(
                f"dataset {dataset}: its rows carry no times (m_time), so the "
                "own games of its ladder holdout cannot be told from the sealed "
                "ones (give --allow-fresh-holdout to read it whole)"
            )
        if own["late_rows"] and not args.allow_fresh_holdout:
            own_before = int(own["cut"])
            say(
                f"ladder holdout: {own['late_rows']} rows of own games from "
                f"{F.OWN_SEALED_FROM} on (the sealed set) are left out"
            )
    departures = rule_departures(args)
    try:
        document = A.read_artifact(source, strict=True)
        loaded = A.load_predictor(source, strict=True)
    except (OSError, ValueError) as exc:
        raise CouplingError(f"artifact {source}: {exc}") from exc
    if loaded.coupling is not None and not args.refit:
        raise CouplingError(
            f"{source} already carries a coupling; a second one is never "
            "stacked on it (give --refit to drop it and fit anew)"
        )
    extra = dict(loaded.meta.get("extra") or {})
    trained_on = S._trained_on(extra)
    same_build = trained_on is not None and trained_on == manifest_sha
    if not same_build and not args.allow_other_dataset:
        raise CouplingError(
            f"the artifact was fitted on dataset manifest {trained_on}, this "
            f"dataset is {manifest_sha}: its validation rows may be the model's "
            "training rows (give --allow-other-dataset to fit anyway)"
        )
    suffix = str(args.suffix or "")
    if not suffix:
        raise CouplingError("the new artifact needs a name suffix")
    name = loaded.name + suffix + (f"_limit{args.limit}" if args.limit else "")
    describe = getattr(loaded.predictor, "describe", None)
    described: Any = describe() if callable(describe) else None
    move_intent = loaded.featurizer.tables.move_intent

    validation, left_out = without_holdout_battles(load_split(dataset, SPLIT_VAL))
    if args.limit:
        keep = np.arange(min(int(args.limit), len(validation["turn"])))
        validation = F.take(validation, keep)
    say(
        f"validation: {len(validation['turn'])} examples "
        f"({left_out} of ladder-holdout battles left out)"
    )
    before = predict_features(loaded.predictor, validation, SPLIT_VAL)
    rows = fit_rows(validation, before, move_intent)
    count = int(rows["rows"].sum())
    say(f"fit rows: {count} two-slot rows, skipped {rows['skipped']}")
    if count == 0:
        raise CouplingError("no row to fit on")
    fit = C.fit_coupling(
        rows["model"],
        rows["seen"],
        rows["weight"],
        rows["bucket"],
        rows["battle"],
        kappas=tuple(args.kappa_grid),
        folds=int(args.folds),
        seed=int(args.seed),
        margin=float(args.margin),
        fold_share=float(args.fold_share),
        buckets=args.buckets == C.BUCKET_SCHEME,
    )
    taken = fit["verdict"] == C.VERDICT_TAKEN
    say(
        f"cross-validation: gain per fold {np.round(fit['fold_gain'], 5).tolist()} "
        f"(kappa {fit['kappa']}, {'bucketed' if fit['bucketed'] else 'pooled'}); "
        f"above {fit['margin']} in {fit['folds_passed']} of {fit['folds']} folds, "
        f"{fit['folds_needed']} needed: {fit['verdict']}"
    )
    made = C.PairCoupling.build(
        fit["table"],
        name=name,
        fitted_after=C.fitted_after(loaded.kind, described),
        info={
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source_artifact": str(source),
            "source_sha256": sha256_file(source),
            "source_name": loaded.name,
            "dataset": str(dataset),
            "dataset_tag": manifest.get("tag"),
            "manifest_sha256": manifest_sha,
            "split": SPLIT_VAL,
            "rows": count,
            "weighted": True,
            "kappa": fit["kappa"],
            "bucketed": fit["bucketed"],
            "folds": fit["folds"],
            "seed": fit["seed"],
            "margin": fit["margin"],
            "fold_gain": [float(value) for value in fit["fold_gain"]],
            "verdict": fit["verdict"],
            "rule_as_fixed": not departures,
        },
    )
    bucket_rows = {
        C.BUCKETS[index]: int((rows["bucket"] == index).sum())
        for index in range(C.N_BUCKET)
    }
    report: dict[str, Any] = {
        "format": FORMAT,
        "version": VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "verdict": fit["verdict"],
        "acceptance": {
            "rule": "held-out gain in pair-set log-probability above the margin "
            "in at least fold_share of the folds",
            "fixed": RULE_FIXED,
            "as_fixed": not departures,
            "departures": departures,
            "margin": fit["margin"],
            "fold_share": fit["fold_share"],
            "folds": fit["folds"],
            "folds_needed": fit["folds_needed"],
            "folds_passed": fit["folds_passed"],
            "weighted_by": "m_weight",
            "design_reading": {
                "wording": DESIGN_WORDING,
                "mean_gain": fit["mean_gain"],
                "folds_positive": fit["folds_positive"],
                "holds": fit["design_reading"],
            },
        },
        "source": {
            "artifact": str(source),
            "sha256": sha256_file(source),
            "name": loaded.name,
            "kind": loaded.kind,
            "fitted_after": made.fitted_after,
        },
        "dataset": {
            "path": str(dataset),
            "tag": manifest.get("tag"),
            "created": created,
            "manifest_sha256": manifest_sha,
            "same_build_as_training": same_build,
        },
        "name": name,
        "arguments": {key: plain(value) for key, value in vars(args).items()},
        "rows": {
            "validation_examples": int(len(validation["turn"])),
            "ladder_holdout_battle_rows_left_out": left_out,
            "fit_rows": count,
            "fit_weight": float(rows["weight"].sum()),
            "fully_seen": int(rows["fully_seen"].sum()),
            "fully_seen_and_counted": int((rows["fully_seen"] & rows["counted"]).sum()),
            "by_bucket": bucket_rows,
            "skipped": rows["skipped"],
        },
        "classes": list(C.CLASSES),
        "buckets": list(C.BUCKETS),
        "fit": fit,
        "validation": {
            "note": "in sample for the table (fitted on these rows); the "
            "cross-validated gain above is the honest number",
            "coverage": coverage_report(
                validation,
                before,
                made,
                move_intent,
                resamples=int(args.resamples),
                seed=int(args.seed),
            ),
            "pair_set_gain": pair_set_gain(
                rows, made.table, resamples=int(args.resamples), seed=int(args.seed)
            ),
            "marginal_drift": marginal_drift(rows, made.table),
            "pair_events": pair_events(rows, made.table, C.full_tilt(fit["parts"])),
        },
        "informational": {},
        "artifact": None,
        "artifact_withheld": None,
    }

    # The status is the verdict's own word: a taken coupling without a file
    # is never reported as not taken.
    status = TAKEN_NO_ARTIFACT if taken else NOT_TAKEN
    withheld = ""
    if taken and args.no_artifact:
        withheld = "--no-artifact was given"
    elif taken and departures and not args.allow_other_rule:
        withheld = (
            "the rule or the fit settings are not the fixed ones ("
            + "; ".join(departures)
            + "): give --allow-other-rule to write the artifact"
        )
    if withheld:
        report["artifact_withheld"] = withheld
        say(f"no artifact: {withheld}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if taken and args.fit_split == SPLIT_VAL and not withheld:
        path = out_dir / ARTIFACT_NAME
        unverified = path.with_name(path.name + UNVERIFIED_SUFFIX)
        extra[A.EXTRA_COUPLING] = {
            "created": report["created"],
            "source": report["source"],
            "dataset": report["dataset"],
            "name": name,
            "kappa": fit["kappa"],
            "bucketed": fit["bucketed"],
            "fold_gain": [float(value) for value in fit["fold_gain"]],
            "rule_as_fixed": not departures,
        }
        A.save_artifact(
            unverified,
            kind=loaded.kind,
            name=name,
            featurizer=loaded.featurizer,
            predictor_payload=document["predictor"],
            extra=plain(extra),
            coupling=made,
        )
        say(f"SAVED {unverified}")
        checks = verify(
            document,
            loaded,
            unverified,
            made,
            validation,
            before,
            float(fit["in_sample_gain"]),
        )
        report["checks"] = checks
        if not checks["ok"]:
            report["status"] = FAILED
            write_report(out_dir, report)
            failed = [key for key, value in checks.items() if value is False]
            raise CouplingError(f"the written artifact failed its checks: {failed}")
        unverified.replace(path)
        report["artifact"] = {"path": str(path), "sha256": sha256_file(path)}
        say(f"VERIFIED {path}")
        status = DONE

    # Informational, after the verdict and the checks: never part of a choice.
    for split in args.report_splits:
        batch = load_split(dataset, split, own_before)
        pred = predict_features(loaded.predictor, batch, split)
        split_rows = fit_rows(batch, pred, move_intent)
        report["informational"][split] = {
            "note": "read after the verdict; the table never saw these rows",
            "own_games": own,
            "sealed_rows_left_out": int(own["late_rows"]) if own_before and own else 0,
            "examples": int(len(batch["turn"])),
            "games": int(np.unique(np.asarray(batch["m_battle"])).size),
            "two_slot_rows": int(split_rows["rows"].sum()),
            "coverage": coverage_report(
                batch,
                pred,
                made,
                move_intent,
                resamples=int(args.resamples),
                seed=int(args.seed),
            ),
            "pair_set_gain": pair_set_gain(
                split_rows,
                made.table,
                resamples=int(args.resamples),
                seed=int(args.seed),
            ),
            "marginal_drift": marginal_drift(split_rows, made.table),
            "pair_events": pair_events(
                split_rows, made.table, C.full_tilt(fit["parts"])
            ),
        }
        say(f"informational: {split} read ({len(batch['turn'])} examples)")
    report["seconds"] = round(time.time() - started, 1)
    report["status"] = status
    write_report(out_dir, report)
    return status, report


# --- rendering ------------------------------------------------------------------


def _num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def _change(
    found: Mapping[str, Any] | None, scale: float = 1.0, digits: int = 4
) -> str:
    if not found or found.get("diff") is None:
        return "n/a"
    text = f"{scale * float(found['diff']):+.{digits}f}"
    if found.get("low") is not None and found.get("high") is not None:
        text += (
            f" [{scale * float(found['low']):+.{digits}f}, "
            f"{scale * float(found['high']):+.{digits}f}]"
        )
    return text


def _matrix(table: Sequence[Sequence[float]], classes: Sequence[str]) -> list[str]:
    lines = ["| slot a \\ slot b | " + " | ".join(classes) + " |"]
    lines.append("|---|" + "---|" * len(classes))
    for name, row in zip(classes, table):
        lines.append(f"| {name} | " + " | ".join(_num(v, 3) for v in row) + " |")
    return lines


def _coverage_lines(found: Mapping[str, Any]) -> list[str]:
    lines = [
        f"{found['counted']} of {found['examples']} examples counted (every acting "
        f"slot shows its whole action), {found['games']} games; "
        f"{found['involves_other']} true replies go through OTHER (a miss at every K). "
        f"Mean coupled total over the plain mass on two-slot turns: "
        f"{_num(found.get('tilt_mean_two_slots'), 4)}.",
        "",
        "| slice | turns | games | variant | top-1 | top-4 | top-8 | top-16 "
        "| mean log p |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, row in found["slices"].items():
        for name in ("plain", "coupled"):
            cells = [_pct(row[name]["top"][str(k)]) for k in KS]
            lines.append(
                f"| {label} | {row['examples']} | {row['games']} | {name} | "
                + " | ".join(cells)
                + f" | {_num(row[name]['log_prob'], 4)} |"
            )
        change = row["change"]
        cells = [_change(change[f"top{k}"], 100.0, 2) for k in KS]
        lines.append(
            f"| {label} | | | change (points; nats) [95%] | "
            + " | ".join(cells)
            + f" | {_change(change['log_prob'], 1.0, 4)} |"
        )
    return lines


def _own_games_text(found: Mapping[str, Any]) -> str:
    """What an informational ladder-holdout reading did about the sealed
    own games ('' for a report written before that was recorded)."""
    own = found.get("own_games")
    if not own:
        return ""
    if own.get("read_whole") and own.get("late_rows"):
        return (
            f" THE SEALED OWN GAMES WERE READ: {own['late_rows']} rows of games "
            f"from {own['sealed_from']} on (--allow-fresh-holdout)."
        )
    left = found.get("sealed_rows_left_out") or 0
    return (
        f" Own games from {own['sealed_from']} on (the sealed set): "
        f"{left} rows left out; none is read."
    )


def _events_lines(found: Mapping[str, Any]) -> list[str]:
    """The pair events as a table. With the full tilt's columns (a report
    written since they exist) a line under it says what they are."""
    whole = any("full_tilt" in row for row in found.values())
    head = "| pair event | observed | today's joint | ratio | coupled joint (R) "
    head += "| ratio after |"
    rule = "|---|---|---|---|---|---|"
    if whole:
        head += " full tilt T (not stored) | ratio under T |"
        rule += "---|---|"
    lines = [head, rule]
    for label, row in found.items():
        text = (
            f"| {label} | {_pct(row['observed'])} | {_pct(row['model'])} | "
            f"{_num(row['ratio_before'], 2)} | {_pct(row['coupled'])} | "
            f"{_num(row['ratio_after'], 2)} |"
        )
        if whole:
            text += (
                f" {_pct(row.get('full_tilt'))} | "
                f"{_num(row.get('ratio_full_tilt'), 2)} |"
            )
        lines.append(text)
    if whole:
        lines += [
            "",
            "Ratios are observed / predicted (1.00 = the event is predicted at "
            "its rate). T is the estimate before its per-class part D was "
            "discarded: it reproduces the events on its own fit rows. R, which "
            "is what is stored and applied, is T without D, so an event of two "
            "classes that D scaled down comes out too often under R (a ratio "
            "below 1), and the reverse. That gap is the price of storing an "
            "interaction only.",
        ]
    return lines


def render(report: Mapping[str, Any]) -> str:
    """The report as Markdown, from the JSON alone."""
    fit = report["fit"]
    classes = report["classes"]
    accept = report["acceptance"]
    scheme_text = (
        "bucket tables"
        if fit["bucketed"]
        else "the pooled table for both two-target buckets"
    )
    # Reports written before these fields existed render as they did.
    if accept.get("as_fixed") is False:
        rule_text = (
            "Rule (NOT the fixed one: "
            + "; ".join(accept.get("departures") or [])
            + ")"
        )
    elif accept.get("fixed"):
        rule_text = f"Rule ({accept['fixed']})"
    else:
        rule_text = "Rule"
    design = accept.get("design_reading") or {}
    design_lines: list[str] = []
    if design:
        design_lines = [
            "",
            f"The design's own wording ({design['wording']}): mean held-out gain "
            f"{_num(design['mean_gain'], 5)}, positive in "
            f"{design['folds_positive']} of {accept['folds']} folds: "
            f"{'holds' if design['holds'] else 'does not hold'}. Reported beside "
            "the verdict; the verdict is the per-fold rule above.",
        ]
    lines = [
        f"# Pair coupling of `{report['source']['name']}`: {report['verdict']}",
        "",
        f"Written {report['created']} by `training/fit_oppmodel_coupling.py`. Source "
        f"artifact `{report['source']['artifact']}` (sha256 "
        f"{report['source']['sha256'][:12]}), dataset `{report['dataset']['path']}` "
        f"(built {report['dataset']['created']}), fitted on validation.",
        "",
        "## Verdict",
        "",
        f"{rule_text}: the held-out gain in pair-set log-probability must "
        f"be above {accept['margin']} nats per two-slot row in at least "
        f"{accept['folds_needed']} of {accept['folds']} folds. It is in "
        f"**{accept['folds_passed']} of {accept['folds']}**: **{report['verdict']}**.",
        *design_lines,
        "",
        f"Chosen: kappa {fit['kappa']}, {scheme_text}"
        f" (bucket tables beat the pooled one by the margin in "
        f"{fit['bucket_folds_better']} folds). Mean held-out gain "
        f"{_num(fit['mean_gain'], 5)} nats per row; in sample "
        f"{_num(fit['in_sample_gain'], 5)} (weighted), "
        f"{_num(fit['in_sample_gain_unweighted'], 5)} (unweighted)."
        + (
            f" The full tilt T, with the per-class part that is discarded, would "
            f"gain {_num(fit['in_sample_gain_full_tilt'], 5)} in sample (weighted): "
            "the difference is what storing an interaction only gives up."
            if fit.get("in_sample_gain_full_tilt") is not None
            else ""
        ),
        "",
        "| fold | rows | gain (weighted) | gain (unweighted) |",
        "|---|---|---|---|",
    ]
    for index, (size, gain, flat) in enumerate(
        zip(fit["fold_rows"], fit["fold_gain"], fit["fold_gain_unweighted"])
    ):
        lines.append(f"| {index} | {size} | {_num(gain, 5)} | {_num(flat, 5)} |")
    lines += [
        "",
        "Every grid value (mean held-out gain over the folds, weighted; folds above "
        "the margin). The grid choice is made on these same folds:",
        "",
        "| scheme | kappa | mean gain | folds above the margin | lowest fold |",
        "|---|---|---|---|---|",
    ]
    cv = fit["cv"]
    for s, scheme in enumerate(cv["schemes"]):
        for g, kappa in enumerate(cv["kappas"]):
            values = [v for v in cv["gain"][s][g] if v is not None]
            if not values:
                continue
            above = sum(1 for v in values if v > accept["margin"])
            lines.append(
                f"| {scheme} | {kappa} | {_num(sum(values) / len(values), 5)} | "
                f"{above} of {len(values)} | {_num(min(values), 5)} |"
            )
    rows = report["rows"]
    lines += [
        "",
        "## Rows",
        "",
        f"{rows['validation_examples']} validation examples "
        f"({rows['ladder_holdout_battle_rows_left_out']} rows of ladder-holdout "
        f"battles left out); {rows['fit_rows']} two-slot rows in the fit "
        f"({rows['fully_seen']} with one consistent reply for each slot"
        + (
            f"; {rows['fully_seen_and_counted']} of those are counted turns of "
            "the coverage tables below, which also need a logged target: a move "
            "outside the candidates has none to split"
            if rows.get("fully_seen_and_counted") is not None
            else ""
        )
        + f"), by bucket {rows['by_bucket']}; skipped {rows['skipped']}.",
        "",
        "## The fitted tables (R: what is stored)",
        "",
        "R multiplies the product of the two slots' probabilities; 1 is "
        "independence. Unit margins under the class mix of the fit's rows, so R "
        "holds no per-class correction (that part, D, is reported below and "
        "discarded). Where no row of a fit holds both attack classes (one "
        "opposing slot: the later-one bucket) the two are one class, and the "
        "margins are taken that way.",
    ]
    for index, bucket in enumerate(report["buckets"]):
        lines += ["", f"### {bucket} ({rows['by_bucket'].get(bucket, 0)} rows)", ""]
        lines += _matrix(fit["table"][index], classes)
    lines += ["", "### Parts of each fit", ""]
    for name, part in fit["parts"].items():
        lines += [
            f"**{name}**: {part['rows']} rows, weight {_num(part['weight'], 1)}, "
            f"{part['iterations']} iterations, converged {part['converged']}"
            + (
                f", attack classes {part['attack_classes']}"
                if part.get("attack_classes")
                else ""
            )
            + ".",
            "",
            "| class | mix | D (discarded) |",
            "|---|---|---|",
        ]
        for label, mix, scale in zip(classes, part["mix"], part["D"]):
            lines.append(f"| {label} | {_pct(mix)} | {_num(scale, 3)} |")
        lines.append("")
    lines += [
        "### The unweighted table beside it",
        "",
        "The same fit with every row at weight 1 (same kappa and bucket rule).",
    ]
    for index, bucket in enumerate(report["buckets"]):
        lines += ["", f"#### {bucket}, unweighted", ""]
        lines += _matrix(fit["unweighted"]["table"][index], classes)
    val = report["validation"]
    lines += [
        "",
        "## Validation (in sample for the table)",
        "",
        "### Coverage of the true pair on fully seen turns",
        "",
        *_coverage_lines(val["coverage"]),
        "",
        "### Pair-set log-probability, turn starts (the fit's own measure)",
        "",
        f"Gain per two-slot row, in sample: {_change(val['pair_set_gain'], 1.0, 5)} "
        f"(unweighted, games resampled); weighted "
        f"{_num(val['pair_set_gain'].get('weighted'), 5)}; fully seen rows "
        f"{_num(val['pair_set_gain'].get('fully_seen'), 5)}, partly hidden rows "
        f"{_num(val['pair_set_gain'].get('partly_hidden'), 5)}.",
        "",
        "### Drift of the class marginals",
        "",
        f"Mean |coupled - model| per class and slot "
        f"{_num(val['marginal_drift']['mean'], 5)}, 99th percentile "
        f"{_num(val['marginal_drift']['p99'], 5)}, largest "
        f"{_num(val['marginal_drift']['max'], 5)}.",
        "",
        "### Pair events on the fit rows",
        "",
        *_events_lines(val["pair_events"]),
    ]
    for split, found in report["informational"].items():
        gain = found["pair_set_gain"]
        lines += [
            "",
            f"## Informational: {split} (never read by the fit)",
            "",
            f"{found['examples']} examples, {found['games']} games, "
            f"{found['two_slot_rows']} two-slot rows with a free choice shown."
            + _own_games_text(found),
            "",
            f"Pair-set log-probability gain per two-slot row: "
            f"**{_change(gain, 1.0, 5)}** (unweighted, 95% from resampling games); "
            f"fully seen rows {_num(gain.get('fully_seen'), 5)}, partly hidden rows "
            f"{_num(gain.get('partly_hidden'), 5)}.",
            "",
            *_coverage_lines(found["coverage"]),
            "",
            f"Marginal drift: mean {_num(found['marginal_drift']['mean'], 5)}, 99th "
            f"percentile {_num(found['marginal_drift']['p99'], 5)}.",
            "",
            *_events_lines(found["pair_events"]),
        ]
    checks = report.get("checks")
    if checks:
        lines += ["", "## Checks of the written artifact", ""]
        lines += [f"- {key}: {value}" for key, value in checks.items()]
    artifact = report.get("artifact")
    withheld = report.get("artifact_withheld")
    lines += [
        "",
        "## Artifact",
        "",
        f"`{artifact['path']}` (sha256 {artifact['sha256']})"
        if artifact
        else "None written" + (f": {withheld}." if withheld else "."),
        "",
        f"Status: {report.get('status')}. {report.get('seconds')} s.",
        "",
    ]
    return "\n".join(lines)


def write_report(out_dir: Path, report: Mapping[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = plain(report)
    (out_dir / REPORT_JSON).write_text(
        json.dumps(data, indent=1, sort_keys=False) + "\n", encoding="utf-8"
    )
    (out_dir / REPORT_MD).write_text(render(data), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True, help="a new directory")
    parser.add_argument("--fit-split", default=SPLIT_VAL, choices=(SPLIT_VAL,))
    parser.add_argument(
        "--buckets", default=C.BUCKET_SCHEME, choices=(C.BUCKET_SCHEME, "none")
    )
    parser.add_argument(
        "--kappa-grid", type=float, nargs="+", default=list(C.DEFAULT_KAPPAS)
    )
    parser.add_argument("--folds", type=int, default=C.DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=C.DEFAULT_SEED)
    parser.add_argument("--margin", type=float, default=C.DEFAULT_MARGIN)
    parser.add_argument("--fold-share", type=float, default=C.DEFAULT_FOLD_SHARE)
    parser.add_argument(
        "--report-splits",
        nargs="*",
        default=[],
        help=f"informational splits read after the verdict: {REPORT_SPLITS}",
    )
    parser.add_argument(
        "--allow-fresh-holdout",
        "--allow-sealed-holdout",  # the trainer's name for the same thing
        dest="allow_fresh_holdout",
        action="store_true",
        help="read the whole ladder holdout, the sealed own games included",
    )
    parser.add_argument(
        "--allow-other-rule",
        action="store_true",
        help="write the artifact although the rule or the fit settings are "
        "not the fixed ones (the report and the artifact say so)",
    )
    parser.add_argument("--allow-other-dataset", action="store_true")
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--no-artifact", action="store_true")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--suffix", default="_pair")
    parser.add_argument("--render-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.render_only:
        path = Path(args.out) / REPORT_JSON
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            say(f"{FAILED} {exc}")
            return 1
        (Path(args.out) / REPORT_MD).write_text(render(report), encoding="utf-8")
        say(str(report.get("status") or DONE))
        return 0
    try:
        import torch

        torch.set_num_threads(max(1, int(args.threads)))
    except Exception:
        pass
    try:
        status, _ = run(args)
    except CouplingError as exc:
        say(f"{FAILED} {exc}")
        return 1
    say(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
