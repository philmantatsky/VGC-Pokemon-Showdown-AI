"""Learning curve of the opponent predictor: does more replay data still help?

OPPONENT_PREDICTOR.md, second pass: going from 15,155 to 26,847 human battles
took fine NLL on the same 401 ladder-holdout games from 1.672 to 1.583, but the
added games were also the recent, rating-balanced part of the feed. This driver
measures volume alone. Every point is ONE corpus subsampled at random by battle
(``datagen/oppmodel_build_dataset.py --human-fraction``: a best-of-three series
stays whole, the subsamples of one seed are nested, the bot's own saved games
are always kept), then built, fitted and scored in the same way.

One process, one stage after the other, every stage skipped when its output is
already there (so a stopped run is simply started again):

  1. build   results_oppmodel/<base>_f<percent>/         for every fraction
  2. check   that all points read the same human corpus (the manifests' inputs)
  3. fit     results_oppmodel/<base>_f<percent>_oppnet/  training/train_oppmodel.py
     tables  results_oppmodel/<base>_f<percent>_tables/  only with --with-tables
  4. score   every point on (a) the ladder holdout and (b) that build's own
             test split, with the scorecard's functions
             (``evaluation.oppmodel_scorecard``: guarded_predict, score_set,
             paired_difference): fine NLL with its game-clustered 95% interval,
             its action and target parts, top-1 / top-3, intent accuracy; with
             --with-tables the same for the FlagsTable fitted on that build
  5. pair    consecutive points on (a): the same games are in every build, so
             the per-slot fine NLL of the larger point minus the smaller one is
             taken on the same slot-turns and its interval resamples games.
             Also on (b), over the test games both builds hold (the smaller
             build's test set is a subset of the larger one's).

The 100% point is ``--full-artifact`` on ``--full-dataset`` (the model already
trained on everything) unless 1.0 is among ``--fractions``, in which case the
driver builds and fits that point like the others.

Pairing. Examples of two builds are matched by replay id, turn and side through
each build's ``battles.jsonl`` and its ``m_battle`` / ``m_turn`` / ``m_side``
arrays, never by row number. Features differ by build (each has its own
repertoire, so its own candidate moves) but what the player did does not, so
the labels are compared on every matched slot in a form that does not depend on
the candidate list: kind, reason, Mega, intent, attack, the flags, the switch
destination, the NAME of the move when both builds name it and the target when
both give one. Any disagreement stops the pairing of that step and fails the
run (``CURVE_FAILED``, with the field, the count and examples in the JSON); it
is never averaged over. Two things are counted and are not errors: a move that
is a candidate in one build and falls in OTHER in the other, and a target label
that only one build could give.

Label space. Each build scores its own candidate list: a move that falls in
OTHER in the smaller build and is a named candidate in the larger one is scored
in two label spaces. Every step therefore also carries a CONTROL, the same
paired difference for a predictor that knows nothing (uniform over the legal
actions, ``label_space_only``), the model's step net of it
(``net_of_label_space``, paired, with its own interval) and the model's step
on the slots whose label is named alike in both builds (``same_label_space``).
The sentence says "more data still helps" only when the model's step AND its
step net of the control are below zero.

Not a reading. The sentence is a statement about more data only when the
points were made alike and in full. It starts with ``NO READING`` and names
the reason instead when the run is a smoke, a build / fit / table fit was
limited, the two points of the last step were trained with other arguments
(or their arguments are not recorded), or the points read corpora that differ
by more than the default drift. ``curve.json`` holds the reasons under
``reading``; the status stays ``CURVE_DONE`` (the pipeline ran).

Account cap. A build down-weights an account's examples so that no account
counts for more than the cap in THAT build, so a smaller build is trained with
relatively more weight on the heaviest accounts and the "doubling" of battles
is not a doubling of what the loss sees. Every point reports its cap-weighted
number of training examples and, for ONE set of accounts at every point (those
over the cap in the 100% point), their share of the point's training examples
before and after weighting; "per doubling" is computed from the ratio of the
weighted numbers.

One corpus. A fraction is a fraction of the 100% point's corpus only if both
builds read the same human logs. The manifests' input hashes are compared
after the builds and before any fit; a difference of more than
``--max-corpus-drift`` (share of human logs offered, default 0.02) stops the
run. The public feed may still be downloading: builds made minutes apart then
differ by a few hundred logs, which is tolerated and written down, while a
100% point built hours earlier is not.

Outputs: ``<out>/curve.json`` and ``<out>/curve.md`` (rendered from the JSON
only; ``--render-only`` rewrites it), default ``results_oppmodel/<base>_curve``.
One row per point (human battles kept, training examples, the metrics), the
paired steps, and a plain sentence on whether the last doubling still helps and
by how much. Every directory written starts with ``<base>_`` and sits directly
in the results folder; anything else is refused. A build or fit directory that
holds something else than what this run would make is refused too, never
replaced. Heartbeat lines carry the elapsed time; the last line printed is
``CURVE_DONE <path>`` (exit code 0) or ``CURVE_FAILED <reason>`` (exit code 1).

    nice -n 19 .venv/bin/python training/oppmodel_learning_curve.py \\
        --base-tag v2curve --fractions 0.125,0.25,0.5 --seed 0 --with-tables
    nice -n 19 .venv/bin/python training/oppmodel_learning_curve.py \\
        --base-tag c_curve_smoke --smoke          # tiny builds, 1-epoch fits
"""

from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path as _Path

_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # numpy's BLAS on macOS
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import importlib
import json
import math
import re
import shlex
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from datagen import oppmodel_build_dataset as B
from evaluation import oppmodel_scorecard as SC
from vgc_bench.src.oppmodel import features as F

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "oppmodel-learning-curve"
VERSION = 1
CURVE_JSON = "curve.json"
CURVE_MD = "curve.md"
DONE = "CURVE_DONE"
FAILED = "CURVE_FAILED"

MODEL = "oppnet"
BAR = "flags_table"
SET_LADDER = SC.SET_LADDER
SET_TEST = SC.SET_TEST
SETS: tuple[str, ...] = (SET_LADDER, SET_TEST)
SET_TEXT = {
    SET_LADDER: "(a) ladder holdout: the opponents in the bot's own ladder games, "
    "the same games in every build",
    SET_TEST: "(b) test players of that build: held-out human players (a smaller "
    "build has a smaller test set, a subset of the larger one's)",
}
STEP_TEXT = {
    SET_LADDER: "(a) ladder holdout, the same slot-turns under both points",
    SET_TEST: "(b) the test slot-turns both builds hold",
}

DEFAULT_FRACTIONS = "0.125,0.25,0.5"
DEFAULT_FULL_DATASET = "results_oppmodel/v2_feed"
DEFAULT_FULL_ARTIFACT = "results_oppmodel/oppnet_v2_blind/artifact.pt"
DEFAULT_FULL_TABLES = "results_oppmodel/tables_v2/flags_table.pt"
DEFAULT_DRIFT = 0.02
SMOKE_BUILD_LIMIT = 300
SMOKE_TRAIN_LIMIT = 3000
SMOKE_EPOCHS = 1
SMOKE_RESAMPLES = 200
DATASET_FILES: tuple[str, ...] = (
    "battles.jsonl",
    "vocab.json",
    "tables.npz",
    "repertoire.json",
)
ARTIFACT_NAME = "artifact.pt"
TRAIN_REPORT = "train_report.json"
TABLES_ARTIFACT = "flags_table.pt"
TABLES_REPORT = "fit_report.json"
# Measured on the v2 fit (one thread, batch 512, 228,512 examples): the wall
# time printed before the fits is an estimate from these two numbers.
ETA_STEP_SECONDS = 0.20
ETA_EPOCHS = 23
# Training arguments that must agree between points for the curve to compare
# like with like; a difference is listed among the warnings.
HYPER_KEYS: tuple[str, ...] = (
    "batch",
    "clip",
    "d_model",
    "dropout",
    "elo_drop",
    "elo_mode",
    "epochs",
    "extras",
    "ff",
    "heads",
    "layers",
    "limit",
    "lr",
    "mega_weight",
    "patience",
    "seed",
    "slot_swap",
    "warmup",
    "weight_decay",
)
# What a train report written before an argument existed says of it: the
# value the trainer used then. ``extras`` (the bit mask of the version-2
# arrays the network reads) is 0 for every fit from before the flag: the
# version-1 network. Without this an old fit could never be reused, and an
# old point would read as "trained with other arguments" beside a new one.
HYPER_UNRECORDED: dict[str, Any] = {"extras": 0}
MISMATCH_EXAMPLES = 5
DOUBLING = (1.8, 2.2)  # a step whose size ratio is in here is called a doubling
NO_READING = "NO READING"
SPLIT_TRAIN = "train"
# What the driver itself tells the trainer: ``--train-extra`` may add to these
# and may not change any of them (nor send the fit to another device, nor ask
# it to overwrite anything).
PINNED_TRAIN_ARGS: tuple[str, ...] = (
    "dataset",
    "out",
    "tag",
    "elo_mode",
    "threads",
    "epochs",
    "patience",
    "seed",
    "limit",
    "device",
    "overwrite",
)

_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# The label arrays that cannot depend on a build's candidate list, and the
# columns of y_flag that cannot (every one but "true move outside candidates").
PLAIN_LABELS: tuple[str, ...] = ("y_kind", "y_reason", "y_mega", "y_intent")
FLAG_COLUMNS: tuple[int, ...] = tuple(
    column for column in range(F.N_Y_FLAG) if column != F.Y_OTHER
)

Log = Callable[[str], None]
Key = tuple[str, int, int]


class CurveError(RuntimeError):
    """The run cannot go on, or cannot give a valid reading."""


def say(text: str) -> None:
    """Print a progress line at once (stdout may be a file)."""
    print(text, flush=True)


# --- names and the directories a run may write --------------------------------


def parse_fractions(text: str) -> list[float]:
    """``"0.125,0.25,0.5"`` as ascending, distinct fractions in (0, 1]."""
    values: set[float] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = float(part)
        except ValueError as exc:
            raise CurveError(f"--fractions: {part!r} is not a number") from exc
        if not 0.0 < value <= 1.0:
            raise CurveError(f"--fractions: {part} is not in (0, 1]")
        values.add(value)
    if not values:
        raise CurveError("--fractions names no fraction")
    return sorted(values)


def percent_tag(fraction: float) -> str:
    """0.125 -> ``12p5``, 0.25 -> ``25``, 1.0 -> ``100`` (for a directory name)."""
    text = f"{fraction * 100.0:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def percent_text(fraction: float) -> str:
    return f"{fraction * 100.0:.4f}".rstrip("0").rstrip(".") + "%"


def check_base_tag(base: str) -> str:
    """The base tag, or ``CurveError``: it is the prefix of every name written."""
    if not _TAG.fullmatch(base or ""):
        raise CurveError(
            f"--base-tag {base!r} must be letters, digits, '.', '_' or '-' and "
            "start with a letter or digit"
        )
    return base


def owned_dir(path: Path | str, base: str, results_root: Path | str) -> Path:
    """``path`` resolved, if this run may write there; ``CurveError`` otherwise.

    A run owns the directories that sit directly in the results folder and
    whose name starts with ``<base>_``. Everything a run creates, fills or
    reuses goes through here first, so no option can point it at another
    experiment's dataset, model or scorecard.
    """
    check_base_tag(base)
    target = Path(path).resolve()
    root = Path(results_root).resolve()
    if target.parent != root or not target.name.startswith(base + "_"):
        raise CurveError(
            f"refusing to touch {path}: not a directory of {root} whose name "
            f"starts with {base + '_'!r}"
        )
    return target


@dataclass
class Point:
    """One point of the curve.

    ``built`` points are made by this run (dataset, fit and tables directories
    it owns); the other kind is the 100% point given on the command line, which
    is only read.
    """

    fraction: float
    name: str
    built: bool
    dataset: Path
    artifact: Path
    fit_dir: Path | None = None
    tables: Path | None = None
    tables_dir: Path | None = None
    timings: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return percent_text(self.fraction)


def _absolute(path: str, root: Path) -> Path:
    found = Path(path)
    return found if found.is_absolute() else root / found


def plan_points(args: argparse.Namespace) -> list[Point]:
    """The points of a run, ascending by fraction, the 100% point last."""
    base = check_base_tag(args.base_tag)
    results = Path(args.results_root)
    points: list[Point] = []
    for fraction in args.fraction_list:
        stem = f"{base}_f{percent_tag(fraction)}"
        fit_dir = owned_dir(results / f"{stem}_oppnet", base, results)
        tables_dir = owned_dir(results / f"{stem}_tables", base, results)
        points.append(
            Point(
                fraction=fraction,
                name=stem,
                built=True,
                dataset=owned_dir(results / stem, base, results),
                artifact=fit_dir / ARTIFACT_NAME,
                fit_dir=fit_dir,
                tables=tables_dir / TABLES_ARTIFACT if args.with_tables else None,
                tables_dir=tables_dir if args.with_tables else None,
            )
        )
    if not any(point.fraction == 1.0 for point in points):
        repo = Path(args.repo_root)
        dataset = _absolute(args.full_dataset, repo)
        points.append(
            Point(
                fraction=1.0,
                name=dataset.name,
                built=False,
                dataset=dataset,
                artifact=_absolute(args.full_artifact, repo),
                tables=_absolute(args.full_tables, repo) if args.with_tables else None,
            )
        )
    return points


# --- small readers --------------------------------------------------------------


def read_json(path: Path) -> dict[str, Any] | None:
    """A JSON object from disk, or None when it is absent or not one."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_state(directory: Path) -> tuple[str, dict[str, Any] | None]:
    """``absent`` / ``partial`` / ``damaged`` / ``complete``, and the manifest.

    The builder writes the manifest last, so a directory without one is a
    build that was stopped (``partial``: it is built again). A manifest whose
    shards or featurizer files are missing is ``damaged`` and is not touched.
    """
    if not directory.exists():
        return "absent", None
    manifest = read_json(directory / "manifest.json")
    if manifest is None:
        return "partial", None
    names = [str(shard.get("file")) for shard in manifest.get("shards") or []]
    missing = [
        name for name in (*names, *DATASET_FILES) if not (directory / name).is_file()
    ]
    return ("damaged" if missing else "complete"), manifest


def build_differences(
    manifest: Mapping[str, Any], point: Point, args: argparse.Namespace
) -> list[str]:
    """How an existing build differs from the one this run would make."""
    found: list[str] = []
    chosen = manifest.get("human_fraction")
    if manifest.get("tag") != point.name:
        found.append(f"tag {manifest.get('tag')!r}")
    if point.fraction >= 1.0:
        if chosen is not None:
            found.append("a subsample")
    elif not isinstance(chosen, Mapping):
        found.append("not a subsample")
    else:
        if chosen.get("fraction") != point.fraction:
            found.append(f"fraction {chosen.get('fraction')}")
        if chosen.get("seed") != args.seed:
            found.append(f"seed {chosen.get('seed')}")
        if chosen.get("hash") != B.FRACTION_HASH:
            found.append(f"subsample hash {chosen.get('hash')!r}")
    if manifest.get("limit") != args.build_limit:
        found.append(f"limit {manifest.get('limit')}")
    if list(manifest.get("sources_requested") or []) != list(args.source_list):
        found.append(f"sources {manifest.get('sources_requested')}")
    return found


def battle_ids(dataset: Path) -> list[str]:
    """Replay id of every row of a build's ``battles.jsonl``, by ``m_battle``."""
    ids: list[str] = []
    try:
        text = (dataset / "battles.jsonl").read_text(encoding="utf-8")
    except OSError as exc:
        raise CurveError(f"{dataset}: battles.jsonl cannot be read: {exc}") from exc
    for position, line in enumerate(text.splitlines()):
        try:
            row = json.loads(line)
            index, battle = int(row["index"]), str(row["id"])
        except (ValueError, KeyError, TypeError) as exc:
            raise CurveError(f"{dataset}: battles.jsonl line {position + 1}") from exc
        if index != position:
            raise CurveError(
                f"{dataset}: battles.jsonl row {position} says index {index}"
            )
        ids.append(battle)
    return ids


def train_weights(
    dataset: Path, manifest: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray] | None:
    """(account-cap weight, account code) of every training example of a build.

    Read from the shards' ``m_split``, ``m_weight`` and ``m_actor`` alone (no
    features, no labels); None when the build has no training split. Raises
    ``CurveError`` on a shard that cannot be read.
    """
    names = list(manifest.get("splits") or [])
    if SPLIT_TRAIN not in names:
        return None
    code = names.index(SPLIT_TRAIN)
    weights: list[np.ndarray] = []
    actors: list[np.ndarray] = []
    for shard in manifest.get("shards") or []:
        try:
            with np.load(dataset / str(shard.get("file")), allow_pickle=False) as data:
                rows = np.asarray(data["m_split"]) == code
                weights.append(np.asarray(data["m_weight"], dtype=np.float64)[rows])
                actors.append(np.asarray(data["m_actor"]).astype(np.int64)[rows])
        except (OSError, ValueError, KeyError) as exc:
            raise CurveError(f"{dataset}: shard {shard.get('file')}: {exc!r}") from exc
    if not weights:
        return np.zeros(0), np.zeros(0, dtype=np.int64)
    return np.concatenate(weights), np.concatenate(actors)


def over_cap_accounts(dataset: Path, manifest: Mapping[str, Any]) -> np.ndarray:
    """Account codes whose training examples a build down-weights (weight
    below 1: the account is over the cap in that build), sorted."""
    found = train_weights(dataset, manifest)
    if found is None:
        return np.zeros(0, dtype=np.int64)
    weight, actor = found
    return np.unique(actor[weight < 1.0])


def cap_weighting(
    dataset: Path,
    manifest: Mapping[str, Any],
    heavy: np.ndarray | None = None,
    heavy_of: str | None = None,
) -> dict[str, Any]:
    """What the account cap makes of a build's training split.

    The training examples, their cap-weighted number (the sum of the weights,
    what the loss sees), and the examples of the accounts over the cap IN THIS
    BUILD (weight below 1) with their share before and after weighting.

    A smaller build has fewer accounts over the cap, so that share cannot be
    compared along a curve. ``heavy`` is ONE set of account codes for every
    point (``over_cap_accounts`` of the 100% point, named by ``heavy_of``):
    ``heaviest_accounts`` then gives the share of this build's training
    examples that belong to those accounts, before and after weighting. In a
    smaller build the same accounts are capped less, so their weighted share
    is larger: the mix changes along the curve.

    Empty when the build has no training split.
    """
    found = train_weights(dataset, manifest)
    if found is None:
        return {}
    weight, actor = found
    examples, total = int(weight.size), float(weight.sum())
    over = weight < 1.0
    out: dict[str, Any] = {
        "account_cap": manifest.get("account_cap"),
        "train_examples": examples,
        "train_examples_weighted": total,
        "examples_of_accounts_over_cap": int(over.sum()),
        "accounts_over_cap": int(np.unique(actor[over]).size),
        "share_over_cap": float(over.sum()) / examples if examples else None,
        "share_over_cap_weighted": float(weight[over].sum()) / total
        if total > 0
        else None,
    }
    if heavy is not None:
        theirs = np.isin(actor, np.asarray(heavy).astype(np.int64))
        out["heaviest_accounts"] = {
            "over_the_cap_in": heavy_of,
            "accounts": int(np.asarray(heavy).size),
            "share": float(theirs.sum()) / examples if examples else None,
            "share_weighted": float(weight[theirs].sum()) / total
            if total > 0
            else None,
        }
    return out


# --- stages: build, fit, tables -------------------------------------------------


def ensure_build(point: Point, args: argparse.Namespace, log: Log) -> dict[str, Any]:
    """The point's dataset: reused when complete, built otherwise; its manifest."""
    directory = owned_dir(point.dataset, args.base_tag, args.results_root)
    state, manifest = dataset_state(directory)
    if state == "damaged":
        raise CurveError(
            f"{directory} has a manifest but not all of its files; it is not "
            "rebuilt over. Remove it or use another --base-tag"
        )
    if state == "complete" and manifest is not None:
        differences = build_differences(manifest, point, args)
        if differences:
            raise CurveError(
                f"{directory} holds another build ({', '.join(differences)}); it is "
                "not replaced. Use another --base-tag"
            )
        seconds = (manifest.get("seconds") or {}).get("total")
        point.timings.update({"build_seconds": seconds, "build_reused": True})
        log(f"build {point.name}: complete, reused")
        return manifest
    log(f"build {point.name}: human fraction {point.fraction:g}, seed {args.seed}")
    started = time.time()
    try:
        manifest = B.build(
            point.name,
            root=Path(args.repo_root),
            out_root=Path(args.results_root),
            sources=list(args.source_list),
            limit=args.build_limit,
            human_fraction=point.fraction,
            fraction_seed=args.seed,
            log=log,
        )
    except SystemExit as exc:  # the builder's way of refusing
        raise CurveError(f"build {point.name}: {exc}") from exc
    took = round(time.time() - started, 1)
    point.timings.update({"build_seconds": took, "build_reused": False})
    return manifest


def human_inputs(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The manifest's inputs that hold human logs: path -> its recorded entry."""
    out: dict[str, dict[str, Any]] = {}
    for path, entry in (manifest.get("inputs") or {}).items():
        if not isinstance(entry, Mapping):
            continue
        kind = entry.get("kind")
        web = kind == "folder" and str(path).startswith(B.WEB_DIR)
        if kind in ("json", "feed_shard") or web:
            out[str(path)] = dict(entry)
    return out


def logs_offered(manifest: Mapping[str, Any]) -> int:
    """Human logs the build read from its sources, before any subsampling."""
    chosen = manifest.get("human_fraction")
    if isinstance(chosen, Mapping) and chosen.get("logs_offered") is not None:
        return int(chosen["logs_offered"])
    read = (manifest.get("battles") or {}).get("read") or {}
    return int(sum(count for source, count in read.items() if source != "own"))


def corpus_report(
    manifests: Mapping[str, Mapping[str, Any]], reference: str
) -> dict[str, Any]:
    """Whether every point read the reference point's human corpus.

    ``drift`` of a point is the difference between the human logs it was
    offered and the reference's, over the reference's.
    """
    base_inputs = human_inputs(manifests[reference])
    base_logs = logs_offered(manifests[reference])
    rows: dict[str, Any] = {}
    for name, manifest in manifests.items():
        inputs = human_inputs(manifest)
        differing = sorted(
            path
            for path in set(inputs) | set(base_inputs)
            if (inputs.get(path) or {}).get("sha256")
            != (base_inputs.get(path) or {}).get("sha256")
        )
        offered = logs_offered(manifest)
        rows[name] = {
            "human_logs_offered": offered,
            "inputs_differing": differing,
            "drift": abs(offered - base_logs) / base_logs if base_logs else None,
        }
    drifts = [row["drift"] for row in rows.values() if row["drift"] is not None]
    return {
        "reference": reference,
        "reference_human_logs_offered": base_logs,
        "identical_inputs": not any(row["inputs_differing"] for row in rows.values()),
        "largest_drift": max(drifts) if drifts else None,
        "points": rows,
    }


def check_corpus(report: Mapping[str, Any], limit: float) -> list[str]:
    """Warnings for a tolerated difference; ``CurveError`` for a larger one."""
    warnings: list[str] = []
    reference = report["reference"]
    for name, row in report["points"].items():
        if name == reference or not row["inputs_differing"]:
            continue
        drift = row["drift"]
        text = (
            f"{name} read {row['human_logs_offered']:,} human logs, the 100% point "
            f"{reference} {report['reference_human_logs_offered']:,} "
            f"({len(row['inputs_differing'])} input files differ"
            + ("" if drift is None else f", {100.0 * drift:.2f}% of the logs")
            + ")"
        )
        if drift is None or drift > limit:
            raise CurveError(
                f"not one corpus: {text}. The fractions would not be fractions of "
                "the 100% point's corpus. Add 1.0 to --fractions to build and fit "
                "the 100% point from the corpus as it is now, or raise "
                "--max-corpus-drift to accept the difference"
            )
        warnings.append(
            f"Not exactly one corpus: {text}; within --max-corpus-drift "
            f"({100.0 * limit:g}%)."
        )
    return warnings


def trainer_argv(
    dataset: Path | str, directory: Path | str, args: argparse.Namespace
) -> list[str]:
    """The trainer's command line as the driver itself writes it: the dataset,
    the fit directory (also the tag) and the run's own settings."""
    argv = [
        "--dataset",
        str(dataset),
        "--out",
        str(directory),
        "--tag",
        Path(directory).name,
        "--elo-mode",
        str(args.elo_mode),
        "--threads",
        str(args.threads),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--seed",
        str(args.train_seed),
    ]
    if args.train_limit is not None:
        argv += ["--limit", str(args.train_limit)]
    return argv


def hyper_value(args: Mapping[str, Any], key: str) -> Any:
    """A recorded training argument; one that is absent (or None) reads as
    the value of ``HYPER_UNRECORDED`` when the argument has one."""
    value = args.get(key)
    if value is None and key in HYPER_UNRECORDED:
        return HYPER_UNRECORDED[key]
    return value


def checked_trainer_args(
    dataset: Path | str, directory: Path | str, args: argparse.Namespace
) -> tuple[list[str], dict[str, Any]]:
    """(the trainer's full command line, its parsed form), or ``CurveError``.

    ``--train-extra`` is appended to the driver's own arguments, and argparse
    lets a later option win, so it is checked here: it may add options (the
    network's size, the batch, the learning rate) and may not change one the
    driver sets (``PINNED_TRAIN_ARGS``: above all ``--out`` / ``--dataset``,
    which would send the fit into a directory the run does not own), nor ask
    for another device or for ``--overwrite``.
    """
    trainer = importlib.import_module("training.train_oppmodel")
    own = trainer_argv(dataset, directory, args)
    try:
        extra = shlex.split(args.train_extra or "")
    except ValueError as exc:
        raise CurveError(f"--train-extra cannot be read: {exc}") from exc
    try:
        asked = vars(trainer.parse_args(own))
        wanted = vars(trainer.parse_args(own + extra))
    except SystemExit as exc:
        raise CurveError(f"bad trainer arguments ({exc}): {own + extra}") from exc
    changed = [key for key in PINNED_TRAIN_ARGS if wanted.get(key) != asked.get(key)]
    if changed:
        raise CurveError(
            "--train-extra may not change what the driver sets itself: "
            + ", ".join(
                f"{key} {wanted.get(key)!r} (the run's: {asked.get(key)!r})"
                for key in changed
            )
        )
    return own + extra, wanted


def ensure_fit(point: Point, args: argparse.Namespace, log: Log) -> None:
    """The point's Elo-blind (or ``--elo-mode``) OppNet: reused, or trained."""
    if point.fit_dir is None:
        return
    directory = owned_dir(point.fit_dir, args.base_tag, args.results_root)
    manifest_sha = sha256_file(point.dataset / "manifest.json")
    report = read_json(directory / TRAIN_REPORT) or {}
    trainer = importlib.import_module("training.train_oppmodel")
    argv, wanted = checked_trainer_args(point.dataset, directory, args)
    if point.artifact.is_file():
        # An artifact is never replaced: it is reused when it is the fit this
        # run would make, and the run stops when it is anything else.
        refusal = "it is not replaced. Remove the directory or use another --base-tag"
        if report.get("status") != trainer.DONE:
            raise CurveError(
                f"{point.artifact} has no finished {TRAIN_REPORT} next to it; {refusal}"
            )
        if (report.get("dataset") or {}).get("manifest_sha256") != manifest_sha:
            raise CurveError(
                f"{point.artifact} was trained on another build of {point.dataset}; "
                f"{refusal}"
            )
        had = report.get("args") or {}
        other = [
            f"{key} {hyper_value(had, key)!r}, not {hyper_value(wanted, key)!r}"
            for key in HYPER_KEYS
            if hyper_value(had, key) != hyper_value(wanted, key)
        ]
        if other:
            raise CurveError(
                f"{point.artifact} was trained with other settings "
                f"({'; '.join(other)}); {refusal}"
            )
        timing = report.get("timing") or {}
        point.timings.update(
            {
                "fit_seconds": timing.get("seconds_total"),
                "fit_seconds_per_step": timing.get("seconds_per_step"),
                "fit_steps": timing.get("steps"),
                "fit_reused": True,
            }
        )
        log(f"fit {point.name}: artifact present, reused")
        return
    # Leftovers of a fit that was stopped (a report, an unverified artifact):
    # the directory is this run's and holds no artifact.pt, so it is redone.
    if any(path.exists() for path in trainer.run_files(directory)):
        argv.append("--overwrite")
    log(f"fit {point.name}: train_oppmodel {' '.join(argv)}")
    started = time.time()
    try:
        done = trainer.train(trainer.parse_args(argv))
    except SystemExit as exc:
        raise CurveError(f"fit {point.name}: bad trainer arguments ({exc})") from exc
    except Exception as exc:
        raise CurveError(f"fit {point.name}: {type(exc).__name__}: {exc}") from exc
    timing = done.get("timing") or {}
    point.timings.update(
        {
            "fit_seconds": round(time.time() - started, 1),
            "fit_seconds_per_step": timing.get("seconds_per_step"),
            "fit_steps": timing.get("steps"),
            "fit_reused": False,
        }
    )
    if not point.artifact.is_file():
        raise CurveError(f"fit {point.name}: no {ARTIFACT_NAME} was written")


def ensure_tables(point: Point, args: argparse.Namespace, log: Log) -> None:
    """The point's count tables (``--with-tables``): reused, or fitted."""
    if point.tables_dir is None or point.tables is None:
        return
    directory = owned_dir(point.tables_dir, args.base_tag, args.results_root)
    report = read_json(directory / TABLES_REPORT)
    if point.tables.is_file() and report is not None:
        # A finished fit: reused when it is the one this run would make.
        fitted = report.get("dataset") or {}
        manifest_sha = sha256_file(point.dataset / "manifest.json")
        other = []
        if fitted.get("manifest_sha256") != manifest_sha:
            other.append("another build of the dataset")
        if fitted.get("limit_per_split") != args.tables_limit:
            other.append(f"limit {fitted.get('limit_per_split')}")
        if other:
            raise CurveError(
                f"{directory} holds another table fit ({', '.join(other)}); it is "
                "not replaced. Remove the directory or use another --base-tag"
            )
        point.timings.update(
            {"tables_seconds": report.get("seconds"), "tables_reused": True}
        )
        log(f"tables {point.name}: fit present, reused")
        return
    fitter = importlib.import_module("training.fit_oppmodel_tables")
    log(f"tables {point.name}: fit_oppmodel_tables on {point.dataset}")
    started = time.time()
    try:
        fitter.run(
            point.dataset,
            directory,
            limit=args.tables_limit,
            overwrite=bool(fitter.existing_outputs(directory)),
            log=log,
        )
    except Exception as exc:
        raise CurveError(f"tables {point.name}: {type(exc).__name__}: {exc}") from exc
    point.timings.update(
        {"tables_seconds": round(time.time() - started, 1), "tables_reused": False}
    )
    if not point.tables.is_file():
        raise CurveError(f"tables {point.name}: no {TABLES_ARTIFACT} was written")


# --- pairing two builds ---------------------------------------------------------


@dataclass
class EvalView:
    """One point's examples of one evaluation set, as the pairing needs them.

    ``keys`` are (replay id, turn, side) per example. ``labels`` holds the
    build's ``y_*`` arrays and ``cand_move``; ``moves`` is the build's move
    vocabulary (``cand_move`` indexes it). ``fine`` / ``scored`` are the
    model's per-slot fine NLL and its mask, ``[N, 2]``, from
    ``features.slot_nll``.
    """

    keys: list[Key]
    labels: dict[str, np.ndarray]
    moves: tuple[str, ...]
    n_cand: int
    fine: np.ndarray
    scored: np.ndarray


def eval_view(
    batch: Mapping[str, np.ndarray],
    ids: Sequence[str],
    featurizer: F.Featurizer,
    pred: Mapping[str, np.ndarray],
) -> EvalView:
    """The view of one evaluation set of a build under one prediction.

    ``batch`` is the set's examples with labels and ``m_*`` arrays, ``ids``
    the build's replay ids by ``m_battle`` (``battle_ids``), ``pred`` the
    model's prediction on the batch. The per-slot fine NLL is
    ``features.slot_nll``'s, the one the scorecard averages.
    """
    battles = np.asarray(batch["m_battle"]).astype(np.int64)
    if battles.size and (battles.min() < 0 or battles.max() >= len(ids)):
        raise CurveError("m_battle points outside battles.jsonl")
    turns = np.asarray(batch["m_turn"]).astype(np.int64)
    sides = np.asarray(batch["m_side"]).astype(np.int64)
    nll = F.slot_nll(pred, batch)
    return EvalView(
        keys=[
            (str(ids[int(battles[row])]), int(turns[row]), int(sides[row]))
            for row in range(int(battles.size))
        ],
        labels={
            name: np.asarray(batch[name])
            for name in batch
            if name.startswith("y_") or name == "cand_move"
        },
        moves=tuple(featurizer.vocab.moves),
        n_cand=int(featurizer.n_cand),
        fine=np.asarray(nll["fine"], dtype=np.float64),
        scored=np.asarray(nll["fine_scored"], dtype=bool),
    )


def match_keys(first: Sequence[Key], second: Sequence[Key]) -> dict[str, Any]:
    """Rows of two builds that are the same example, by key.

    Every example of a side is exactly one of: matched (its key occurs once
    on each side), alone (its key is not on the other side), or repeated (its
    key is on both sides and more than once on one of them: a repeated key
    cannot say which row is which, so it is never matched). Returns the
    aligned row numbers ``first`` / ``second`` and the other two counts per
    side.
    """
    count_first, count_second = Counter(first), Counter(second)
    where = {key: row for row, key in enumerate(second) if count_second[key] == 1}
    rows_first: list[int] = []
    rows_second: list[int] = []
    only_first = repeated_first = 0
    for row, key in enumerate(first):
        if key not in count_second:
            only_first += 1
        elif count_first[key] > 1 or key not in where:
            repeated_first += 1
        else:
            rows_first.append(row)
            rows_second.append(where[key])
    only_second = sum(n for key, n in count_second.items() if key not in count_first)
    return {
        "first": np.asarray(rows_first, dtype=np.int64),
        "second": np.asarray(rows_second, dtype=np.int64),
        "only_first": only_first,
        "only_second": only_second,
        "repeated_first": repeated_first,
        "repeated_second": len(second) - only_second - len(rows_second),
    }


def label_view(view: EvalView, rows: np.ndarray) -> dict[str, np.ndarray]:
    """The labels of ``rows`` in a form free of the build's candidate list.

    ``move`` is the NAME of the move for a label that points at a candidate
    and "" otherwise; ``named`` says which; ``switch_to`` is the roster index
    of a switch label, -1 otherwise; ``acted`` is "a visible free choice".
    """
    labels = view.labels
    n_cand = view.n_cand
    y_action = np.asarray(labels["y_action"]).astype(np.int64)[rows]
    named = (y_action >= 0) & (y_action < n_cand)
    switch = y_action > n_cand
    cand_move = np.asarray(labels["cand_move"]).astype(np.int64)[rows]
    chosen = np.take_along_axis(
        cand_move, np.clip(y_action, 0, max(0, n_cand - 1))[..., None], axis=-1
    )[..., 0]
    names = np.asarray([*view.moves, ""], dtype=object)  # last: not in the vocabulary
    known = len(names) - 1
    inside = named & (chosen >= 0) & (chosen < known)
    move = names[np.where(inside, chosen, known)]
    out: dict[str, np.ndarray] = {
        name: np.asarray(labels[name]).astype(np.int64)[rows] for name in PLAIN_LABELS
    }
    out["y_attack"] = np.asarray(labels["y_attack"]).astype(np.int64)[rows]
    out["y_flag"] = np.asarray(labels["y_flag"]).astype(np.int64)[rows][
        ..., list(FLAG_COLUMNS)
    ]
    out["acted"] = y_action >= 0
    out["switch_to"] = np.where(switch, y_action - n_cand - 1, -1)
    out["named"] = inside
    out["move"] = move
    out["y_target"] = np.asarray(labels["y_target"]).astype(np.int64)[rows]
    return out


def compare_labels(
    first: EvalView, second: EvalView, match: Mapping[str, Any]
) -> dict[str, Any]:
    """Whether two builds say the same thing happened on their common examples.

    ``mismatches`` counts, per field, the slots where the builds contradict
    each other (it must be empty); ``examples`` shows the first few with their
    key. ``move_named_in_one_build`` / ``target_in_one_build`` /
    ``scored_in_one_build`` count the slots where one build could express a
    label and the other could not: expected, and not a contradiction.
    """
    rows_first, rows_second = match["first"], match["second"]
    a, b = label_view(first, rows_first), label_view(second, rows_second)
    checks: dict[str, np.ndarray] = {
        name: a[name] != b[name] for name in (*PLAIN_LABELS, "acted", "switch_to")
    }
    checks["y_attack"] = (a["y_attack"] != b["y_attack"]).any(-1)
    checks["y_flag"] = (a["y_flag"] != b["y_flag"]).any(-1)
    both_named = a["named"] & b["named"]
    checks["move"] = both_named & (a["move"] != b["move"])
    both_target = (a["y_target"] >= 0) & (b["y_target"] >= 0)
    checks["y_target"] = both_target & (a["y_target"] != b["y_target"])
    mismatches = {name: int(bad.sum()) for name, bad in checks.items() if bad.any()}
    examples: list[dict[str, Any]] = []
    for name, bad in checks.items():
        for row, slot in np.argwhere(bad)[:MISMATCH_EXAMPLES]:
            if len(examples) >= MISMATCH_EXAMPLES:
                break
            battle, turn, side = first.keys[int(rows_first[int(row)])]
            examples.append(
                {
                    "field": name,
                    "battle": battle,
                    "turn": turn,
                    "side": side,
                    "slot": int(slot),
                    "first": np.asarray(a[name][int(row), int(slot)]).tolist(),
                    "second": np.asarray(b[name][int(row), int(slot)]).tolist(),
                }
            )
    any_bad = np.zeros(a["acted"].shape, dtype=bool)
    for bad in checks.values():
        any_bad |= bad
    scored_first = np.asarray(first.scored, dtype=bool)[rows_first]
    scored_second = np.asarray(second.scored, dtype=bool)[rows_second]
    return {
        "slots_compared": int(any_bad.size),
        "slots_mismatched": int(any_bad.sum()),
        "mismatches": mismatches,
        "examples": examples,
        "move_named_in_one_build": int((a["named"] != b["named"]).sum()),
        "target_in_one_build": int(
            ((a["y_target"] >= 0) != (b["y_target"] >= 0)).sum()
        ),
        "scored_in_one_build": int((scored_first != scored_second).sum()),
    }


def same_label_space(
    first: EvalView, second: EvalView, match: Mapping[str, Any]
) -> np.ndarray:
    """``[matched, 2]``: the slot's label is expressed alike in both builds.

    False where the move is a named candidate in one build and OTHER in the
    other, or only one build gives a target: there the two builds score the
    same action in two label spaces.
    """
    a = label_view(first, match["first"])
    b = label_view(second, match["second"])
    return (a["named"] == b["named"]) & ((a["y_target"] >= 0) == (b["y_target"] >= 0))


def paired_step(
    smaller: EvalView,
    larger: EvalView,
    *,
    resamples: int = SC.DEFAULT_RESAMPLES,
    seed: int = SC.DEFAULT_SEED,
    control: tuple[EvalView, EvalView] | None = None,
) -> dict[str, Any]:
    """Fine NLL of the larger point minus the smaller one on their common slots.

    Examples are matched by key and the labels compared first. With a label
    mismatch, or nothing in common, ``difference`` is None and ``error`` says
    why. Otherwise ``difference`` is ``scorecard.paired_difference`` over the
    slots both points score, games (replay ids) resampled: negative means the
    larger point predicts better.

    ``same_label_space`` is the same difference over the slots whose label is
    expressed alike in both builds. ``control`` is the two builds' views under
    a predictor that knows nothing (the same examples, uniform over the legal
    actions): ``label_space_only`` is then its paired difference, which is
    what the two candidate lists alone do to the score, and
    ``net_of_label_space`` the model's difference minus the control's, slot
    by slot, with its own interval. Both are None without a control.
    """
    match = match_keys(smaller.keys, larger.keys)
    out: dict[str, Any] = {
        "examples_matched": int(match["first"].size),
        "examples_only_smaller": int(match["only_first"]),
        "examples_only_larger": int(match["only_second"]),
        "examples_repeated_key": int(
            match["repeated_first"] + match["repeated_second"]
        ),
        "labels": None,
        "difference": None,
        "same_label_space": None,
        "label_space_only": None,
        "net_of_label_space": None,
        "error": None,
    }
    if match["first"].size == 0:
        out["error"] = "no_common_examples"
        return out
    labels = compare_labels(smaller, larger, match)
    out["labels"] = labels
    if labels["slots_mismatched"]:
        out["error"] = "label_mismatch"
        return out
    rows_small, rows_large = match["first"], match["second"]
    scored = (
        np.asarray(smaller.scored, dtype=bool)[rows_small]
        & np.asarray(larger.scored, dtype=bool)[rows_large]
    )
    games = np.unique(
        np.asarray([smaller.keys[int(row)][0] for row in rows_small], dtype=object),
        return_inverse=True,
    )[1]
    games = np.asarray(games, dtype=np.int64).reshape(-1)
    fine_large = np.asarray(larger.fine, dtype=np.float64)[rows_large]
    fine_small = np.asarray(smaller.fine, dtype=np.float64)[rows_small]
    options: dict[str, Any] = {"resamples": resamples, "seed": seed}
    out["difference"] = SC.paired_difference(
        fine_large, fine_small, scored, games, **options
    )
    alike = scored & same_label_space(smaller, larger, match)
    out["same_label_space"] = SC.paired_difference(
        fine_large, fine_small, alike, games, **options
    )
    if control is not None:
        plain_small, plain_large = control
        if plain_small.keys != smaller.keys or plain_large.keys != larger.keys:
            raise CurveError("the label-space control is not on the same examples")
        base_large = np.asarray(plain_large.fine, dtype=np.float64)[rows_large]
        base_small = np.asarray(plain_small.fine, dtype=np.float64)[rows_small]
        both = (
            scored
            & np.asarray(plain_small.scored, dtype=bool)[rows_small]
            & np.asarray(plain_large.scored, dtype=bool)[rows_large]
        )
        out["label_space_only"] = SC.paired_difference(
            base_large, base_small, both, games, **options
        )
        out["net_of_label_space"] = SC.paired_difference(
            fine_large - base_large, fine_small - base_small, both, games, **options
        )
    return out


# --- scoring one point ----------------------------------------------------------


def _metrics(found: Mapping[str, Any], name: str) -> dict[str, Any]:
    """The reported numbers of one predictor from a ``score_set`` result."""
    row = found["predictors"][name]
    difference = row.get("difference") or {}
    return {
        "fine_nll": row["fine_nll"],
        "action_nll": row["parts"]["action"],
        "target_nll": row["parts"]["target"],
        "visible_nll": row["parts"]["visible"],
        "censored_nll": row["parts"]["censored"],
        "top1": row["fine_top1"],
        "top3": row["fine_top3"],
        "action_top1": row["action_top1"],
        "intent_accuracy": row["intent_accuracy"],
        "minus_bar": difference.get("fine"),
    }


def other_rate(batch: Mapping[str, np.ndarray], n_cand: int) -> dict[str, Any]:
    """Share of visible move labels whose move is outside the candidates.

    Such a label is scored as the OTHER bucket instead of as the move. Part of
    what a larger corpus buys is a repertoire that names more of the moves
    played; this rate shows how much of the labels that concerns.
    """
    y_action = np.asarray(batch["y_action"]).astype(np.int64)
    moves = (y_action >= 0) & (y_action <= n_cand)
    outside = moves & (np.asarray(batch["y_flag"])[..., F.Y_OTHER] == 1)
    count = int(moves.sum())
    return {
        "move_labels": count,
        "outside_candidates": int(outside.sum()),
        "rate": float(outside.sum() / count) if count else None,
    }


def score_point(
    point: Point, args: argparse.Namespace, log: Log
) -> tuple[dict[str, Any], dict[str, EvalView], dict[str, EvalView]]:
    """One row of the curve, the point's views for the pairing, and the views
    of the same examples under the predictor that knows nothing (uniform over
    the legal actions): the label-space control of a paired step."""
    started = time.time()
    try:
        data, manifest = F.load_dataset(point.dataset, list(SETS))
        featurizer = F.Featurizer.load(point.dataset)
        manifest_sha = sha256_file(point.dataset / "manifest.json")
    except (OSError, ValueError, KeyError) as exc:
        raise CurveError(f"dataset {point.dataset}: {exc!r}") from exc
    names = list(manifest.get("splits") or [])
    if not data or any(name not in names for name in SETS):
        raise CurveError(f"dataset {point.dataset} lacks the splits {list(SETS)}")
    ids = battle_ids(point.dataset)
    entries: dict[str, SC.Entry] = {}
    try:
        entries[MODEL] = SC.load_artifact_entry(str(point.artifact), [], manifest_sha)
        if point.tables is not None:
            entries[BAR] = SC.load_artifact_entry(str(point.tables), [], manifest_sha)
    except SC.ScorecardError as exc:
        raise CurveError(f"{point.name}: {exc}") from exc
    warnings: list[str] = []
    for name, entry in entries.items():
        if entry.info.get("fitted_on_this_dataset") is False:
            text = f"{point.name}: {name} was fitted on another build of its dataset"
            if point.built:
                raise CurveError(text + "; it is not replaced. Use another --base-tag")
            warnings.append(text[0].upper() + text[1:] + ".")
        if entry.info.get("dex_signature_diff"):
            warnings.append(
                f"{point.name}: {name} was written with another dex "
                f"({entry.info['dex_signature_diff']})."
            )
    if featurizer.signature_diff:
        warnings.append(
            f"{point.name}: the dataset was built with another dex "
            f"({list(featurizer.signature_diff)})."
        )
    headline = manifest.get("headline") or {}
    report = read_json(point.artifact.parent / TRAIN_REPORT) or {}
    history = report.get("history") or []
    if not point.built:
        timing = report.get("timing") or {}
        point.timings.update(
            {
                "fit_seconds": timing.get("seconds_total"),
                "fit_seconds_per_step": timing.get("seconds_per_step"),
                "fit_steps": timing.get("steps"),
                "fit_reused": True,
                "build_seconds": (manifest.get("seconds") or {}).get("total"),
                "build_reused": True,
            }
        )
    row: dict[str, Any] = {
        "name": point.name,
        "fraction": point.fraction,
        "label": point.label,
        "built_by_this_run": point.built,
        "dataset": str(point.dataset),
        "dataset_manifest_sha256": manifest_sha,
        "artifact": str(point.artifact),
        "artifact_sha256": entries[MODEL].info.get("sha256"),
        "tables": None if point.tables is None else str(point.tables),
        "elo_mode": entries[MODEL].info.get("elo_mode"),
        "human_battles": headline.get("human_battles_kept"),
        "human_logs_offered": logs_offered(manifest),
        "ladder_holdout_games": headline.get("ladder_holdout_games"),
        "examples": dict((manifest.get("examples") or {}).get("by_split") or {}),
        "train_examples_in_dataset": (
            (manifest.get("examples") or {}).get("by_split") or {}
        ).get("train"),
        "train_examples_used": (report.get("examples") or {}).get("train"),
        "cap": cap_weighting(point.dataset, manifest),
        "validation_examples_used": (report.get("examples") or {}).get("val"),
        "best_epoch": (report.get("best") or {}).get("epoch"),
        "epochs_run": max(0, len(history) - 1) if history else None,
        "stopped_early": report.get("stopped_early"),
        "validation_fine_nll": (report.get("reload") or {}).get("val_fine_tempered"),
        "train_args": {
            key: (report.get("args") or {}).get(key)
            for key in HYPER_KEYS
            if key in (report.get("args") or {})
        },
        "sets": {},
        "warnings": warnings,
    }
    views: dict[str, EvalView] = {}
    controls: dict[str, EvalView] = {}
    for set_name in SETS:
        batch = F.take(data, np.asarray(data["m_split"]) == names.index(set_name))
        n = int(np.asarray(batch["turn"]).shape[0])
        if n == 0:
            row["sets"][set_name] = {"examples": 0, "games": 0, MODEL: None, BAR: None}
            continue
        preds: dict[str, SC.Prediction] = {}
        try:
            for name, entry in entries.items():
                preds[name], _ = SC.guarded_predict(
                    entry.predictor, batch, name=f"{point.name}:{name}"
                )
            found = SC.score_set(
                batch,
                preds,
                BAR if BAR in preds else MODEL,
                tables=featurizer.tables,
                resamples=args.resamples,
                seed=args.score_seed,
            )
        except SC.ScorecardError as exc:
            raise CurveError(f"{point.name} on {set_name}: {exc}") from exc
        row["sets"][set_name] = {
            "examples": found["examples"],
            "games": found["games"],
            "slots_scored": found["slots"]["scored"],
            "resamples": found["resamples"],
            "other": other_rate(batch, featurizer.n_cand),
            MODEL: _metrics(found, MODEL),
            BAR: _metrics(found, BAR) if BAR in preds else None,
        }
        try:
            views[set_name] = eval_view(batch, ids, featurizer, preds[MODEL])
            plain, _ = SC.guarded_predict(
                F.UniformPredictor(), batch, name=f"{point.name}:uniform"
            )
            controls[set_name] = eval_view(batch, ids, featurizer, plain)
        except (CurveError, SC.ScorecardError) as exc:
            raise CurveError(f"{point.dataset}: {exc}") from exc
        fine = row["sets"][set_name][MODEL]["fine_nll"]
        log(
            f"score {point.name} on {set_name}: {found['examples']:,} examples, "
            f"{found['games']:,} games, fine NLL {_value(fine.get('value'))}"
        )
    point.timings["score_seconds"] = round(time.time() - started, 1)
    row["timings"] = dict(point.timings)
    return row, views, controls


# --- the report -----------------------------------------------------------------


def _plain(value: Any) -> Any:
    """JSON-safe copy: numpy unpacked, paths as text, non-finite as None."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _value(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{float(value):.{digits}f}"


def _heaviest(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """A point's ``cap.heaviest_accounts`` block, or an empty one."""
    return (row.get("cap") or {}).get("heaviest_accounts") or {}


def _count(value: Any) -> str:
    """A number that need not be whole, shown as a count: ``208,568``."""
    return "-" if value is None else f"{float(value):,.0f}"


def _percent(value: Any, digits: int = 1) -> str:
    return "-" if value is None else f"{100.0 * float(value):.{digits}f}%"


def _bracket(found: Mapping[str, Any] | None, digits: int = 4) -> str:
    """``value [low, high]`` of a scorecard interval."""
    if not found or found.get("value") is None:
        return "-"
    if found.get("low") is None or found.get("high") is None:
        return _value(found["value"], digits)
    return (
        f"{found['value']:.{digits}f} [{found['low']:.{digits}f}, "
        f"{found['high']:.{digits}f}]"
    )


def _signed(found: Mapping[str, Any] | None, digits: int = 4) -> str:
    """``diff [low, high]`` of a paired difference, with signs."""
    if not found or found.get("diff") is None:
        return "-"
    text = f"{found['diff']:+.{digits}f}"
    if found.get("low") is None or found.get("high") is None:
        return text
    return f"{text} [{found['low']:+.{digits}f}, {found['high']:+.{digits}f}]"


def make_steps(
    rows: Sequence[Mapping[str, Any]],
    views: Sequence[Mapping[str, EvalView]],
    *,
    resamples: int,
    seed: int,
    controls: Sequence[Mapping[str, EvalView]] | None = None,
) -> list[dict[str, Any]]:
    """The paired difference between each pair of consecutive points.

    ``controls`` are the points' views under the predictor that knows nothing
    (``score_point``): with them every step also carries the label-space
    control (``paired_step``). ``weighted_ratio`` is the ratio of the two
    builds' cap-weighted training examples, the size the loss sees.
    """
    steps: list[dict[str, Any]] = []
    for index in range(1, len(rows)):
        small, large = rows[index - 1], rows[index]
        before, after = small.get("human_battles"), large.get("human_battles")
        weights = [
            (row.get("cap") or {}).get("train_examples_weighted")
            for row in (small, large)
        ]
        step: dict[str, Any] = {
            "from": small["name"],
            "to": large["name"],
            "from_label": small["label"],
            "to_label": large["label"],
            "human_battles": [before, after],
            "battle_ratio": after / before if before and after else None,
            "train_examples_weighted": weights,
            "weighted_ratio": weights[1] / weights[0]
            if weights[0] and weights[1]
            else None,
            "sets": {},
        }
        for set_name in SETS:
            first, second = views[index - 1].get(set_name), views[index].get(set_name)
            if first is None or second is None:
                step["sets"][set_name] = {"difference": None, "error": "set_missing"}
                continue
            control = None
            if controls is not None:
                plain = (
                    controls[index - 1].get(set_name),
                    controls[index].get(set_name),
                )
                if plain[0] is not None and plain[1] is not None:
                    control = (plain[0], plain[1])
            step["sets"][set_name] = paired_step(
                first, second, resamples=resamples, seed=seed, control=control
            )
        steps.append(step)
    return steps


def step_problems(steps: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    """(reasons to fail the run, warnings) read off the paired steps."""
    failures: list[str] = []
    warnings: list[str] = []
    for step in steps:
        for set_name, found in step["sets"].items():
            where = f"{step['from']} -> {step['to']} on {set_name}"
            labels = found.get("labels") or {}
            if found.get("error") == "label_mismatch":
                failures.append(
                    f"label mismatch, {where}: {labels.get('slots_mismatched')} of "
                    f"{labels.get('slots_compared')} slots "
                    f"({labels.get('mismatches')})"
                )
            elif found.get("error") and set_name == SET_LADDER:
                warnings.append(f"No paired difference, {where}: {found['error']}.")
            if set_name != SET_LADDER:
                continue
            alone = found.get("examples_only_smaller", 0) + found.get(
                "examples_only_larger", 0
            )
            if alone or found.get("examples_repeated_key"):
                warnings.append(
                    f"The ladder holdout is not the same examples, {where}: "
                    f"{found.get('examples_only_smaller')} only in the smaller "
                    f"build, {found.get('examples_only_larger')} only in the larger "
                    f"one, {found.get('examples_repeated_key')} with a repeated key; "
                    f"paired on the {found.get('examples_matched'):,} they share."
                )
    return failures, warnings


def hyper_warnings(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Training arguments that differ between a point and the 100% point."""
    if not rows:
        return []
    reference = rows[-1].get("train_args") or {}
    out: list[str] = []
    for row in rows[:-1]:
        args = row.get("train_args") or {}
        if not reference or not args:
            out.append(
                f"{row['name']} or {rows[-1]['name']} has no train_report.json next "
                "to its artifact: their training arguments were not compared."
            )
            continue
        changed = [
            f"{key} {hyper_value(args, key)!r} vs {hyper_value(reference, key)!r}"
            for key in HYPER_KEYS
            if hyper_value(args, key) != hyper_value(reference, key)
        ]
        if changed:
            out.append(
                f"{row['name']} was trained with other arguments than "
                f"{rows[-1]['name']}: {', '.join(changed)}."
            )
    return out


def reading_problems(curve: Mapping[str, Any]) -> list[str]:
    """Why the curve's last step is NOT a reading of "more data"; empty when it is.

    A paired step is a statement about more data only when its two points were
    made alike and in full. Reasons, each read from the curve itself: a smoke
    run; a limited build, fit or table fit; the two points of the last step
    trained with other arguments (``HYPER_KEYS``), or without recorded
    arguments; points that read corpora differing by more than the default
    drift (accepted only because ``--max-corpus-drift`` was raised).
    """
    settings = curve.get("settings") or {}
    problems: list[str] = []
    if settings.get("smoke"):
        problems.append("a smoke run (tiny builds and one-epoch fits)")
    else:
        for key, text in (
            ("build_limit", "the builds were limited to {:,} logs per source"),
            ("train_limit", "the fits were limited to {:,} training examples"),
            ("tables_limit", "the table fits were limited to {:,} examples per split"),
        ):
            if settings.get(key) is not None:
                problems.append(text.format(int(settings[key])))
    steps = list(curve.get("steps") or [])
    rows = {row.get("name"): row for row in curve.get("points") or []}
    if steps:
        small, large = rows.get(steps[-1].get("from")), rows.get(steps[-1].get("to"))
        if small is not None and large is not None:
            first, second = small.get("train_args") or {}, large.get("train_args") or {}
            unrecorded = [
                str(row.get("name"))
                for row, found in ((small, first), (large, second))
                if not found
            ]
            changed = [
                f"{key} {hyper_value(first, key)!r} vs {hyper_value(second, key)!r}"
                for key in HYPER_KEYS
                if hyper_value(first, key) != hyper_value(second, key)
            ]
            if unrecorded:
                problems.append(
                    "the training arguments of "
                    + " and ".join(unrecorded)
                    + " are not recorded, so the two fits cannot be shown to be alike"
                )
            elif changed:
                problems.append(
                    f"{small.get('name')} and {large.get('name')} were not trained "
                    f"alike ({', '.join(changed)})"
                )
    drift = (curve.get("corpus") or {}).get("largest_drift")
    if drift is not None and float(drift) > DEFAULT_DRIFT:
        problems.append(
            f"the points did not read one corpus (they differ by "
            f"{100.0 * float(drift):.2f}% of the human logs; the default limit is "
            f"{100.0 * DEFAULT_DRIFT:g}%)"
        )
    return problems


def verdict(curve: Mapping[str, Any]) -> str:
    """The plain sentence: does the last doubling still help, and by how much.

    A function of the curve alone. When ``reading_problems`` names a reason
    the sentence starts with ``NO READING`` (``SMOKE, NO READING`` for a
    smoke) and draws no conclusion. Otherwise "more data still helps" needs
    the model's paired step below zero AND, where the step carries the
    label-space control, its step net of the control below zero too.
    """
    steps = curve.get("steps") or []
    if not steps:
        return "Fewer than two points were scored: there is no step to judge."
    last = steps[-1]
    found = (last.get("sets") or {}).get(SET_LADDER) or {}
    difference = found.get("difference") or {}
    span = f"{last['from_label']} -> {last['to_label']} of the corpus"
    before, after = last.get("human_battles") or [None, None]
    if before and after:
        span += f" ({before:,} -> {after:,} human battles)"
    if difference.get("diff") is None:
        return (
            f"The last step, {span}, could not be paired on the ladder holdout "
            f"({found.get('error') or 'no common slot-turns'}): no reading."
        )
    ratio = last.get("battle_ratio")
    weighted = last.get("weighted_ratio")
    size = weighted if weighted else ratio  # what the loss sees, when recorded
    doubling = size is not None and DOUBLING[0] <= size <= DOUBLING[1]
    sizes = [] if ratio is None else [f"{ratio:.2f} times the battles"]
    if weighted:
        sizes.append(f"{weighted:.2f} times the cap-weighted training examples")
    what = (
        "The last doubling"
        if doubling and not weighted
        else ("The last doubling" if doubling else "The last step")
        + (f" ({', '.join(sizes)})" if sizes else "")
    )
    text = (
        f"{what}, {span}, changes fine NLL on the ladder holdout by "
        f"{_signed(difference)} nats (paired on {difference.get('slots'):,} "
        f"slot-turns of {difference.get('games'):,} games; 95% interval over "
        "resampled games)"
    )
    problems = reading_problems(curve)
    if problems:
        smoke = bool((curve.get("settings") or {}).get("smoke"))
        head = f"SMOKE, {NO_READING}" if smoke else NO_READING
        return (
            f"{head}: {'; '.join(problems)}. {text}; that number compares points "
            "that were not made alike or not made in full, and says nothing about "
            "whether more data helps."
        )
    control = found.get("label_space_only") or {}
    net = found.get("net_of_label_space") or {}
    controlled = control.get("diff") is not None and net.get("diff") is not None
    if controlled:
        text += (
            f"; a predictor that knows nothing moves by {_signed(control)} on the "
            f"same slot-turns (each build scores its own candidate list), and net "
            f"of that the step is {_signed(net)}"
        )
    low, high = difference.get("low"), difference.get("high")
    gain = -float(difference["diff"])
    per = ""
    if size and size > 1.0 and not doubling:
        of = " of the cap-weighted training examples" if weighted else ""
        per = f", {gain / math.log2(size):.3f} nats per doubling{of} at this rate"
    if high is not None and high < 0:
        beyond = not controlled or (net.get("high") is not None and net["high"] < 0)
        if beyond:
            return f"{text}: more data still helps, by {gain:.3f} nats{per}."
        return (
            f"{text}: the step is not beyond what the two candidate lists alone "
            "do to the score, so it does not show that more data still helps."
        )
    if low is not None and low > 0:
        return (
            f"{text}: the larger corpus scores WORSE by {-gain:.3f} nats, which a "
            "learning curve should not show; look at the two fits before reading it."
        )
    return (
        f"{text}: the interval includes zero, so a gain of this size cannot be "
        "told from none on these games."
    )


def _line(cells: Sequence[Any]) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _table(header: Sequence[Any], rows: Sequence[Sequence[Any]]) -> list[str]:
    return [
        _line(header),
        _line(["---"] * len(header)),
        *(_line(row) for row in rows),
        "",
    ]


def render(curve: Mapping[str, Any]) -> str:
    """``curve.md``: every number comes from ``curve.json``."""
    settings = curve.get("settings") or {}
    rows = list(curve.get("points") or [])
    lines = ["# Opponent predictor: learning curve", ""]
    status = str(curve.get("status"))
    if status != DONE:
        lines += [f"**{status}** {curve.get('reason') or ''}".rstrip(), ""]
    if settings.get("smoke"):
        lines += [
            "**SMOKE RUN: tiny builds and one-epoch fits on a few thousand "
            "examples. The numbers show that the pipeline runs, not the curve.**",
            "",
        ]
    problems = reading_problems(curve)
    if problems:
        lines += [
            f"**{NO_READING} OF THE LEARNING CURVE: "
            + "; ".join(problems)
            + ". The tables below show what was scored; no step in them says "
            "whether more data helps.**",
            "",
        ]
    lines += [
        f"Rendered from `{CURVE_JSON}` by `training/oppmodel_learning_curve.py`. "
        f"Created {curve.get('created')}; base tag `{curve.get('base_tag')}`, "
        f"subsample seed {settings.get('seed')}, Elo mode "
        f"{settings.get('elo_mode')}, {settings.get('resamples'):,} resamples; "
        f"{_value(curve.get('seconds'), 0)} s for this run.",
        "",
        f"**{verdict(curve)}**",
        "",
        "One fit per point: the intervals resample games, they do not hold the "
        "run-to-run difference between two trainings of the same data.",
        "",
    ]
    found_warnings = list(curve.get("warnings") or [])
    if found_warnings:
        lines += ["## Warnings", ""] + [f"- {text}" for text in found_warnings] + [""]

    lines += ["## Points", ""]
    table: list[list[Any]] = []
    for row in rows:
        table.append(
            [
                row.get("label"),
                f"`{row.get('name')}`",
                _value(row.get("human_battles")),
                _value(row.get("train_examples_in_dataset")),
                _count((row.get("cap") or {}).get("train_examples_weighted")),
                f"{_percent(_heaviest(row).get('share'))} / "
                f"{_percent(_heaviest(row).get('share_weighted'))}",
                _value(row.get("train_examples_used")),
                _value(row.get("best_epoch"))
                + ("" if not row.get("epochs_run") else f" of {row['epochs_run']}"),
                _value(row.get("validation_fine_nll")),
            ]
        )
    lines += _table(
        [
            "corpus",
            "build",
            "human battles",
            "training examples",
            "cap-weighted",
            "the heaviest accounts: share / weighted share",
            "used by the fit",
            "best epoch",
            "validation fine NLL",
        ],
        table,
    )
    heaviest = _heaviest(rows[-1]) if rows else {}
    lines += [
        "Cap-weighted: the sum of the account-cap weights of the training "
        "examples, the size the loss sees. The heaviest accounts are ONE set for "
        f"every row: the {_value(heaviest.get('accounts'))} accounts over the cap "
        f"in `{heaviest.get('over_the_cap_in')}`; the two numbers are their "
        "share of that row's training examples before and after weighting. An "
        "account's weight is set inside each build, so a smaller build caps the "
        "same accounts less and trains with relatively more weight on them: "
        "along the curve the mix changes with the volume.",
        "",
    ]
    for set_name in SETS:
        lines += [f"## {SET_TEXT[set_name]}", ""]
        for who, title in ((MODEL, "OppNet"), (BAR, "FlagsTable fitted on that build")):
            table = []
            for row in rows:
                found = (row.get("sets") or {}).get(set_name) or {}
                scores = found.get(who)
                if not scores:
                    continue
                cells = [
                    row.get("label"),
                    _value(row.get("human_battles")),
                    _value(found.get("games")),
                    _value(found.get("slots_scored")),
                    _bracket(scores.get("fine_nll")),
                    _value(scores.get("action_nll")),
                    _value(scores.get("target_nll")),
                    _percent(scores.get("top1")),
                    _percent(scores.get("top3")),
                    _percent(scores.get("intent_accuracy")),
                ]
                if who == MODEL:
                    cells.append(_signed(scores.get("minus_bar")))
                    cells.append(_percent((found.get("other") or {}).get("rate"), 2))
                table.append(cells)
            if not table:
                continue
            header = [
                "corpus",
                "human battles",
                "games",
                "slot-turns",
                "fine NLL [95%]",
                "action part",
                "target part",
                "top-1",
                "top-3",
                "intent accuracy",
            ]
            if who == MODEL:
                header += ["minus its table [95%]", "move outside candidates"]
            lines += [f"{title}:", ""] + _table(header, table)
    lines += ["## Paired steps", ""]
    lines += [
        "Fine NLL of the larger point minus the smaller one on the same "
        "slot-turns; negative means the larger corpus predicts better. Examples "
        "are matched by replay id, turn and side, and the labels compared "
        "before anything is averaged. 'Knows nothing' is the same difference "
        "for a predictor that is uniform over the legal actions: it has learnt "
        "nothing from either corpus, so what it shows is what the two builds' "
        "candidate lists alone do to the score (a move that is OTHER in one "
        "build and named in the other is scored in two label spaces). 'Net' is "
        "the model's step minus that control, slot by slot. 'Same label space' "
        "is the model's step over the slots whose label both builds express "
        "alike.",
        "",
    ]
    for set_name in SETS:
        table = []
        for step in curve.get("steps") or []:
            found = (step.get("sets") or {}).get(set_name) or {}
            difference = found.get("difference") or {}
            labels = found.get("labels") or {}
            before, after = step.get("human_battles") or [None, None]
            table.append(
                [
                    f"{step.get('from_label')} -> {step.get('to_label')}",
                    f"{_value(before)} -> {_value(after)}",
                    _value(step.get("weighted_ratio"), 2),
                    _signed(difference) if difference else f"- ({found.get('error')})",
                    _signed(found.get("label_space_only")),
                    _signed(found.get("net_of_label_space")),
                    _signed(found.get("same_label_space")),
                    _value(difference.get("slots")),
                    _value(difference.get("games")),
                    _value(found.get("examples_matched")),
                    f"{_value(found.get('examples_only_smaller'))} / "
                    f"{_value(found.get('examples_only_larger'))}",
                    _value(labels.get("slots_mismatched")),
                    _value(labels.get("move_named_in_one_build")),
                ]
            )
        if table:
            lines += [f"{STEP_TEXT[set_name]}:", ""] + _table(
                [
                    "step",
                    "human battles",
                    "cap-weighted training examples, ratio",
                    "larger minus smaller [95%]",
                    "knows nothing [95%]",
                    "net [95%]",
                    "same label space [95%]",
                    "slot-turns",
                    "games",
                    "examples matched",
                    "only smaller / only larger",
                    "label mismatches",
                    "move named in one build only",
                ],
                table,
            )
    corpus = curve.get("corpus") or {}
    if corpus:
        lines += ["## One corpus", ""]
        lines += [
            f"Reference: `{corpus.get('reference')}` "
            f"({_value(corpus.get('reference_human_logs_offered'))} human logs "
            f"offered). Identical inputs at every point: "
            f"{_value(bool(corpus.get('identical_inputs')))}; largest drift "
            f"{_percent(corpus.get('largest_drift'), 2)} (limit "
            f"{_percent(settings.get('max_corpus_drift'), 2)}).",
            "",
        ]
        lines += _table(
            ["build", "human logs offered", "input files that differ", "drift"],
            [
                [
                    f"`{name}`",
                    _value(row.get("human_logs_offered")),
                    _value(len(row.get("inputs_differing") or [])),
                    _percent(row.get("drift"), 2),
                ]
                for name, row in (corpus.get("points") or {}).items()
            ],
        )
    lines += ["## Timings (seconds)", ""]
    table = []
    for row in rows:
        timings = row.get("timings") or {}

        def cell(name: str, timings: Mapping[str, Any] = timings) -> str:
            text = _value(timings.get(f"{name}_seconds"), 1)
            return text + (" (reused)" if timings.get(f"{name}_reused") else "")

        table.append(
            [
                row.get("label"),
                cell("build"),
                cell("fit"),
                _value(timings.get("fit_steps")),
                _value(timings.get("fit_seconds_per_step"), 3),
                cell("tables"),
                _value(timings.get("score_seconds"), 1),
            ]
        )
    lines += _table(
        ["corpus", "build", "fit", "steps", "s / step", "tables", "score"], table
    )
    return "\n".join(lines).rstrip("\n") + "\n"


def write_curve(out: Path, curve: Mapping[str, Any]) -> dict[str, Any]:
    plain = _plain(curve)
    (out / CURVE_JSON).write_text(
        json.dumps(plain, indent=1, allow_nan=False), encoding="utf-8"
    )
    (out / CURVE_MD).write_text(render(plain), encoding="utf-8")
    return plain


# --- the run --------------------------------------------------------------------


def prepare_out(args: argparse.Namespace) -> tuple[Path, bool]:
    """(the output directory, whether this call made it).

    Refused when it is not this run's to write: outside the results folder,
    not named after the base tag, or holding anything but an earlier curve.
    """
    out = owned_dir(args.out, args.base_tag, args.results_root)
    made = not out.exists()
    if not made:
        if not out.is_dir():
            raise CurveError(f"{out} is not a directory")
        # A dot file (the Finder's .DS_Store in an iCloud folder) is nobody's
        # result: it neither blocks the run nor is it touched.
        foreign = sorted(
            path.name
            for path in out.iterdir()
            if path.name not in (CURVE_JSON, CURVE_MD) and not path.name.startswith(".")
        )
        if foreign:
            raise CurveError(
                f"refusing to write a curve into {out}: it holds {foreign[:3]}"
            )
        if (out / CURVE_JSON).exists() and not args.overwrite:
            raise CurveError(
                f"{out / CURVE_JSON} exists; give --overwrite to score again "
                "(builds and fits are reused either way)"
            )
    out.mkdir(parents=True, exist_ok=True)
    return out, made


def eta_line(
    point: Point, manifest: Mapping[str, Any], args: argparse.Namespace
) -> str:
    """A rough wall time of a fit, from the v2 fit's step time."""
    examples = ((manifest.get("examples") or {}).get("by_split") or {}).get("train", 0)
    if args.train_limit is not None:
        examples = min(int(examples), int(args.train_limit))
    steps = math.ceil(int(examples) / 512)
    epochs = min(int(args.epochs), ETA_EPOCHS)
    minutes = steps * epochs * ETA_STEP_SECONDS / 60.0
    return (
        f"estimate {point.name}: {int(examples):,} training examples, {steps} "
        f"steps per epoch at batch 512; about {minutes:.1f} min for {epochs} epochs "
        f"at {ETA_STEP_SECONDS:.2f} s per step (one thread)"
    )


def run(args: argparse.Namespace, log: Log = say) -> dict[str, Any]:
    """Build, fit, score and write; returns the curve (also written).

    Raises ``CurveError`` when the run cannot go on. A label mismatch between
    two builds does not raise: the curve is written with status
    ``CURVE_FAILED`` and the mismatch in it, and the caller reads the status.
    A run that raises keeps its builds and fits (the next run reuses them) and
    removes the output directory again if it made it and left it empty.
    """
    points = plan_points(args)
    for point in points:
        if point.fit_dir is not None:  # refused before anything is touched
            checked_trainer_args(point.dataset, point.fit_dir, args)
    out, made = prepare_out(args)
    try:
        return _run(args, points, out, log)
    except BaseException:
        if made:
            try:
                out.rmdir()
            except OSError:
                pass
        raise


def _run(
    args: argparse.Namespace, points: Sequence[Point], out: Path, log: Log
) -> dict[str, Any]:
    started = time.time()

    def beat(text: str) -> None:
        elapsed = int(time.time() - started)
        log(f"[curve {elapsed // 60:02d}:{elapsed % 60:02d}] {text}")

    # The fits set their own thread count; a run that only scores must too.
    importlib.import_module("torch").set_num_threads(max(1, int(args.threads)))
    beat(
        f"{len(points)} points: "
        + ", ".join(f"{p.label} {'build' if p.built else 'given'}" for p in points)
        + f"; output {out}; {max(1, int(args.threads))} thread(s)"
    )

    manifests: dict[str, dict[str, Any]] = {}
    for point in points:
        if point.built:
            manifests[point.name] = ensure_build(point, args, beat)
        else:
            state, manifest = dataset_state(point.dataset)
            if state != "complete" or manifest is None:
                raise CurveError(f"--full-dataset {point.dataset} is {state}")
            if not point.artifact.is_file():
                raise CurveError(f"--full-artifact {point.artifact} does not exist")
            if point.tables is not None and not point.tables.is_file():
                raise CurveError(f"--full-tables {point.tables} does not exist")
            manifests[point.name] = manifest
    corpus = corpus_report(manifests, points[-1].name)
    warnings = check_corpus(corpus, float(args.max_corpus_drift))
    beat(
        f"corpus: identical inputs {corpus['identical_inputs']}, largest drift "
        f"{_percent(corpus['largest_drift'], 2)}"
    )

    for point in points:
        if point.built and not point.artifact.is_file():
            beat(eta_line(point, manifests[point.name], args))
    for point in points:
        if point.built:
            ensure_fit(point, args, beat)
            ensure_tables(point, args, beat)

    rows: list[dict[str, Any]] = []
    views: list[dict[str, EvalView]] = []
    controls: list[dict[str, EvalView]] = []
    for point in points:
        row, view, control = score_point(point, args, beat)
        rows.append(row)
        views.append(view)
        controls.append(control)
    # One set of accounts for every point: those the 100% point caps. Their
    # weighted share of each build's training data shows how the mix moves.
    heavy = over_cap_accounts(points[-1].dataset, manifests[points[-1].name])
    for row, point in zip(rows, points):
        row["cap"] = cap_weighting(
            point.dataset, manifests[point.name], heavy, points[-1].name
        )
    steps = make_steps(
        rows, views, resamples=args.resamples, seed=args.score_seed, controls=controls
    )
    failures, step_warnings = step_problems(steps)
    for row in rows:
        warnings += row.pop("warnings")
    warnings += step_warnings + hyper_warnings(rows)

    curve: dict[str, Any] = {
        "format": FORMAT,
        "version": VERSION,
        "status": FAILED if failures else DONE,
        "reason": "; ".join(failures) or None,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_tag": args.base_tag,
        "settings": {
            "fractions": list(args.fraction_list),
            "seed": int(args.seed),
            "subsample_hash": B.FRACTION_HASH,
            "elo_mode": args.elo_mode,
            "threads": int(args.threads),
            "with_tables": bool(args.with_tables),
            "smoke": bool(args.smoke),
            "sources": list(args.source_list),
            "build_limit": args.build_limit,
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "train_limit": args.train_limit,
            "train_seed": int(args.train_seed),
            "train_extra": args.train_extra or "",
            "tables_limit": args.tables_limit,
            "resamples": int(args.resamples),
            "score_seed": int(args.score_seed),
            "max_corpus_drift": float(args.max_corpus_drift),
            "full_dataset": str(args.full_dataset),
            "full_artifact": str(args.full_artifact),
            "full_tables": str(args.full_tables) if args.with_tables else None,
        },
        "definitions": {
            "fine_nll": SC.DEFINITIONS.get("fine_nll"),
            "interval": SC.DEFINITIONS.get("interval"),
            "top1 / top3": SC.DEFINITIONS.get("fine_top1 / fine_top3"),
            "intent_accuracy": SC.DEFINITIONS.get("intent_accuracy / intent_nll"),
            "minus_bar": "the model's fine NLL minus the FlagsTable's fitted on the "
            "same build, paired, with its game-clustered interval",
            "other.rate": "share of visible move labels whose move is outside the "
            "build's candidate list (scored as the OTHER bucket)",
            "steps": "fine NLL of the larger point minus the smaller one over the "
            "slot-turns both score, examples matched by replay id, turn and side; "
            "interval over resampled games",
            "label_space_only": "the same paired difference for a predictor that "
            "is uniform over the legal actions: it learnt nothing from either "
            "corpus, so this is what the two builds' candidate lists alone do to "
            "the score",
            "net_of_label_space": "the model's paired difference minus the "
            "control's, slot by slot, with its own interval",
            "same_label_space": "the model's paired difference over the slots "
            "whose label both builds express alike (the move named in both or in "
            "neither, a target given by both or by neither)",
            "cap": "the account cap inside each build: train_examples_weighted is "
            "the sum of the training examples' weights (what the loss sees); "
            "share_over_cap the share of training examples that belong to "
            "accounts over the cap in THAT build, before and after weighting "
            "(not comparable between builds); heaviest_accounts the same two "
            "shares for one set of accounts at every point, those over the cap "
            "in the 100% point",
            "reading": "valid is False when the sentence may not be read as a "
            "statement about more data; problems names why",
        },
        "corpus": corpus,
        "points": rows,
        "steps": steps,
        "warnings": warnings,
    }
    problems = reading_problems(curve)
    curve["reading"] = {"valid": not problems, "problems": problems}
    curve["sentence"] = verdict(curve)
    curve["seconds"] = round(time.time() - started, 1)
    plain = write_curve(out, curve)
    beat(f"wrote {out / CURVE_JSON} and {out / CURVE_MD}")
    return plain


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--base-tag",
        required=True,
        help="prefix of every directory the run writes under results_oppmodel/",
    )
    parser.add_argument(
        "--fractions",
        default=DEFAULT_FRACTIONS,
        help="comma list of human fractions in (0, 1]; with 1.0 in it the run "
        "builds and fits the 100%% point itself",
    )
    parser.add_argument("--seed", type=int, default=0, help="subsample seed")
    parser.add_argument("--full-dataset", default=DEFAULT_FULL_DATASET)
    parser.add_argument("--full-artifact", default=DEFAULT_FULL_ARTIFACT)
    parser.add_argument(
        "--full-tables",
        default=DEFAULT_FULL_TABLES,
        help="the 100%% point's flags table, read with --with-tables",
    )
    parser.add_argument(
        "--elo-mode", choices=(F.ELO_KEEP, F.ELO_BLANK), default="blank"
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--with-tables",
        action="store_true",
        help="also fit and score the count tables of every build",
    )
    parser.add_argument("--out", default=None, help="results_oppmodel/<base>_curve")
    parser.add_argument("--overwrite", action="store_true", help="score again")
    parser.add_argument("--sources", default="all", help="as the dataset builder's")
    parser.add_argument(
        "--build-limit", type=int, default=None, help="logs per source (a smoke)"
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument(
        "--train-limit", type=int, default=None, help="training examples (a smoke)"
    )
    parser.add_argument("--train-seed", type=int, default=20261004)
    parser.add_argument(
        "--train-extra",
        default="",
        help="further train_oppmodel.py arguments, as one quoted string",
    )
    parser.add_argument(
        "--tables-limit", type=int, default=None, help="examples per split (a smoke)"
    )
    parser.add_argument("--resamples", type=int, default=None)
    parser.add_argument("--score-seed", type=int, default=SC.DEFAULT_SEED)
    parser.add_argument(
        "--max-corpus-drift",
        type=float,
        default=DEFAULT_DRIFT,
        help="largest tolerated difference in human logs read between a point "
        "and the 100%% point, as a share",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"tiny builds (--build-limit {SMOKE_BUILD_LIMIT}), fits of "
        f"--train-limit {SMOKE_TRAIN_LIMIT} --epochs {SMOKE_EPOCHS}, "
        f"{SMOKE_RESAMPLES} resamples, and the 100%% point built by the run",
    )
    parser.add_argument(
        "--render-only", action="store_true", help=f"rewrite {CURVE_MD} from the JSON"
    )
    parser.add_argument(
        "--results-root", default=None, help="default: results_oppmodel/ in the repo"
    )
    parser.add_argument(
        "--repo-root", default=None, help="where the corpus folders are (tests)"
    )
    return parser.parse_args(argv)


def resolve(args: argparse.Namespace) -> argparse.Namespace:
    """Fill the derived settings of a parsed command line (smoke, defaults)."""
    args.repo_root = (
        str(Path(args.repo_root).resolve()) if args.repo_root else str(ROOT)
    )
    args.results_root = str(
        Path(args.results_root).resolve()
        if args.results_root
        else Path(args.repo_root) / "results_oppmodel"
    )
    check_base_tag(args.base_tag)
    fractions = parse_fractions(args.fractions)
    if args.smoke:
        # The given 100% point was built on the whole corpus: a smoke builds its
        # own, on the same few logs as its fractions.
        if 1.0 not in fractions:
            fractions.append(1.0)
        if args.build_limit is None:
            args.build_limit = SMOKE_BUILD_LIMIT
        if args.train_limit is None:
            args.train_limit = SMOKE_TRAIN_LIMIT
        if args.tables_limit is None:
            args.tables_limit = SMOKE_TRAIN_LIMIT
        args.epochs = min(int(args.epochs), SMOKE_EPOCHS)
        if args.resamples is None:
            args.resamples = SMOKE_RESAMPLES
    if args.resamples is None:
        args.resamples = SC.DEFAULT_RESAMPLES
    args.fraction_list = fractions
    try:
        args.source_list = B.parse_sources(args.sources)
    except SystemExit as exc:
        raise CurveError(f"--sources: {exc}") from exc
    if args.out is None:
        args.out = str(Path(args.results_root) / f"{args.base_tag}_curve")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Command line. The last line printed is ``CURVE_DONE`` or ``CURVE_FAILED``."""
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        if exc.code in (0, None):
            raise
        say(f"{FAILED} bad arguments")
        return 2
    try:
        args = resolve(args)
        if args.render_only:
            out = owned_dir(args.out, args.base_tag, args.results_root)
            curve = read_json(out / CURVE_JSON)
            if curve is None:
                raise CurveError(f"{out / CURVE_JSON} cannot be read")
            # A report whose text is already on disk is left alone, time included.
            SC.write_if_changed(out / CURVE_MD, render(curve))
            say(f"{DONE} {out / CURVE_MD}")
            return 0
        curve = run(args)
    except Exception as exc:
        reason = " ".join(f"{type(exc).__name__}: {exc}".split())
        say(f"{FAILED} {reason}")
        return 1
    if curve.get("status") != DONE:
        say(f"{FAILED} {curve.get('reason')} (see {Path(args.out) / CURVE_JSON})")
        return 1
    say(str(curve.get("sentence")))
    reading = curve.get("reading") or {}
    say(
        f"{DONE} {Path(args.out) / CURVE_JSON}"
        + ("" if reading.get("valid", True) else f" ({NO_READING}: see the sentence)")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
