"""Fine-tune a trained OppNet on the opponent side of the bot's own ladder games.

    nice -n 10 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \\
        .venv/bin/python training/finetune_oppmodel.py \\
        --artifact results_oppmodel/<run>/artifact.pt \\
        --dataset results_oppmodel/<the build that artifact was trained on> \\
        --out results_oppmodel/night_ft_<a new name>

THE IDEA BEING TESTED. On the ladder the opposing player always faces the same
bot and the same six Pokemon; the human corpus holds few games against exactly
that. The bot has saved about 400 of its own earlier ladder games. Does a
careful fine-tune of the trained predictor on the OPPONENT side of those games
predict the opponents of other such games better? The data is small (about
4,800 labelled slot-turns), so over-fitting is the main risk and
cross-validation by game decides. "No gain" is an allowed outcome.

DATA. Three parts of the dataset the artifact was trained on, and nothing else:

* domain rows: the ``ladder_holdout`` rows whose battle id is in the ALLOWED
  list. The list is the set of ladder-holdout battle ids of ``--old-games`` (a
  ``battles.jsonl``; default: the build of 2026-10-05, whose ladder holdout is
  the 401 own games played before 2026-10-05). A newer build also holds the
  SEALED games played from 2026-10-09 on in that split: they are not in the
  list, so their rows are dropped when a shard is read and never reach a
  batch. The run refuses an empty list, and counts the rows and games dropped.
* corpus rows: a seeded random sample of the ``train`` split
  (``--corpus-rows``), used only to keep the network from forgetting.
* validation rows: the ``val`` split without the battles of a ladder-holdout
  opponent (the trainer's rule), to show what a fine-tune gives up on the
  human corpus and, with ``--refit-temperatures``, to refit the temperatures.

The split ``test`` is never taken out of a shard.

THE SEAL IS ALSO A DATE (``SEALED_SINCE``, the ladder read's ``FRESH_SINCE``;
there is no switch that moves it). The list alone is not trusted with it:

* a list file that names a ladder-holdout game dated on or after that day is
  refused before any shard is opened (so ``--old-games <a newer build's
  battles.jsonl>`` fails instead of allowing its sealed games);
* a game is used only when the DATASET's own ``battles.jsonl`` dates it before
  that day. An allowed game the dataset dates on or after it, or does not
  date at all, stops the run: it is never dropped quietly.

The report counts the ladder-holdout games of the dataset that are sealed by
date (all of them must be among the dropped ones).

THE FINE-TUNE continues the training of the artifact's network with the
trainer's own functions (``model.nll_terms`` and ``model.total_loss``: the
censored set losses and the account-cap weights; ``train_oppmodel.augment``:
the slot mirror and, for a model that reads ratings, the rating drop;
``train_oppmodel.parameter_groups``): AdamW at ONE constant learning rate, no
warm-up and no decay, dropout on as in training (the network's own rate),
the gradient norm capped as in training. Every batch holds 1 part domain rows
to 4 parts corpus rows (``--batch`` 160 = 32 + 128), and the domain rows are
up-weighted inside each batch so that the two parts carry the same loss
weight: the fine loss of a step is half the domain rows' weighted mean and
half the corpus rows'. One pass = every domain row once. An Elo-blind
artifact has every rating blanked, as in its training. The temperatures
stored in the artifact are kept; with ``--refit-temperatures`` each
fine-tuned network gets its own, fitted on validation as the trainer fits
them (``train_oppmodel.calibrate``).

The pass counts of one learning rate are checkpoints of ONE run (after 1, 2
and 4 passes). Because the rate is constant and every random draw is made
pass by pass, the checkpoint after ``p`` passes is the network a separate run
of ``p`` passes with the same seed ends on: bit for bit with one thread (a
unit test holds this). With more threads torch's own summation order is not
fixed, and two runs of the same fit differ in about the fourth decimal of a
weight and the sixth of a mean NLL (measured 2026-10-10: 1.6e-6 nats). A fold
uses the same seed for every learning rate, so two settings of a fold see the
same batches and the same dropout masks.

CROSS-VALIDATION. The allowed GAMES found in the dataset are split into
``--folds`` folds by game (seeded; a game is never on both sides, which the
run checks before every fit). For each fold the network is fine-tuned on the
other folds and predicts the held-out fold; the out-of-fold predictions are
pooled, so every domain row is predicted exactly once by a network that never
saw its game. Five folds by game are the PRE-REGISTERED folds. An opponent
with games in two folds is then on both sides (6 of the 395 accounts have
two games among the 401; the two seeds run on 2026-10-10 each split 5 of
them): ``--fold-unit account`` keeps every account's games in one fold. It
is a sensitivity reading, said so in its report, and not the pre-registered
one.

PRE-REGISTERED GRID (2026-10-10, fixed before any result and not to be
extended afterwards): learning rate in {1e-5, 3e-5, 1e-4} x passes over the
domain rows in {1, 2, 4}: nine settings. For each, on the pooled out-of-fold
rows and paired with the unfine-tuned artifact on the same rows, with 95%
intervals from resampling whole games (``--resamples`` 2,000):

* the change in fine NLL, and the same for visible switches / visible
  Protect-family moves / other visible moves / censored slots;
* the change in fine top-1 and top-3;
* the change in JOINT top-8 coverage (``joint.joint_replies``, the turns whose
  acting slots are all visible; a reply that needs the OTHER bucket is a miss,
  as on the scorecard), overall and for the replies that hold a switch and
  those that hold a Protect-family move.

Beside it, for the network fine-tuned on ALL allowed games with each setting:
the SAME readings on validation (held-out players of the human corpus: fine
NLL, the four kinds of slot, top-1 / top-3 and the three joint rows), and the
change on the domain rows it was fitted on (in-sample: the distance to the
out-of-fold number is the over-fit).

PRE-REGISTERED RULE. The setting with the best pooled out-of-fold fine NLL is
TAKEN only if

1. the interval of its change in fine NLL lies below zero, AND
2. its change in joint top-8 coverage is not below zero by more than its own
   interval's half-width, AND
3. the coverage of replies that hold a switch does not fall by more than 2
   points;

otherwise the verdict is ``FT_NOT_TAKEN`` and no artifact is written.

NOT GATED, AND REPORTED NEXT TO THE VERDICT (``decision['not_gated']``). The
rule gates switch replies and says nothing about Protect. A fine-tune that
moves probability off Protect can pass it and still cost a consumer that
needs Protect replies among the first eight. So the verdict is followed by
the best setting's Protect readings on BOTH sets: joint top-8 of replies that
hold a Protect-family move and the fine NLL of the slots where one was
clicked, out of fold on the old games and on validation; a reading whose
interval lies wholly on the worse side is named as a cost. Also there: the
size of the gain when the setting is chosen WITHOUT the fold it is read on
(each fold is read with the setting that is best on the other folds), which
is what the choice among nine settings on the same rows is worth.

THE ARTIFACT. When taken (and not ``--no-artifact``), the network fine-tuned
on all allowed games with that setting is written as ``<out>/artifact.pt``
through ``oppmodel.artifact.save_artifact`` (first as
``artifact.pt.unverified``, reloaded, and renamed only when the reloaded
predictor predicts what the in-memory one predicts). IT HAS SEEN EVERY
ALLOWED GAME: any reading of it on those games is in-sample (run of
2026-10-10: -0.119 nats against -0.022 out of fold). Its record says so in
the fields a reader looks at (``finetuned_extra``):

* its name is the source's plus ``_ft_own<games>``, and ``extra['name']`` /
  ``extra['created']`` are the fine-tune's, not the source's;
* ``extra['dataset']['manifest_sha256']`` (what the scorecard's
  ``_trained_on`` reads) is NOT the build's sha256 but
  ``finetuned-on-ladder-holdout:<games>-games:<the build's sha256>``: the
  scorecard then warns that the artifact's training data does not match the
  dataset's evaluation sets, and a calibration or coupling fit asks for its
  ``--allow-other-dataset`` (its validation rows are still unseen). The
  build's own sha256 is ``extra['dataset']['source_manifest_sha256']``; the
  formats the runtime stands down by are kept;
* ``extra['dataset_tag']`` (what the ladder read prints) names the games;
* ``extra['finetune']['seen_games']`` lists the battle ids it was fitted on.
  ``seen_games(extra)`` returns them and ``check_unseen(extra, ids)`` raises
  when a reading would be in-sample: a reader of own ladder games calls it;
* the source's whole training record (best epoch, history, calibration,
  examples) is under ``extra['source_training']``: none of it describes this
  network.

A source that is itself fine-tuned is refused: its cross-validation would be
in-sample. An event calibration or a pair coupling of the source was fitted
after another network: neither is carried over, and the report says so.

Outputs under ``--out`` (a NEW directory: one that already holds a report or
an artifact is refused, and there is no overwrite switch):
``finetune_report.json``, ``finetune_report.md`` and, when taken,
``artifact.pt``. The report says ``FT_RUNNING`` while the run lasts. The last
printed line is ``FT_TAKEN``, ``FT_NOT_TAKEN`` or ``FT_FAILED <reason>``.

``--learning-rates`` / ``--passes`` exist for the unit tests and smoke runs;
a run with another grid or other folds says so in its report and is not the
pre-registered reading. ``--no-artifact`` is for robustness runs (another
seed, other folds): the verdict is reported and nothing is written beside the
report. CPU only.
"""

from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path as _Path

_os.environ.setdefault("OMP_NUM_THREADS", "2")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")  # numpy's BLAS on macOS
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import copy
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation import oppmodel_scorecard as SC
from training import train_oppmodel as T
from vgc_bench.src.oppmodel import artifact as A
from vgc_bench.src.oppmodel import features as F
from vgc_bench.src.oppmodel import joint as J
from vgc_bench.src.oppmodel import model as M
from vgc_bench.src.oppmodel.events import INTENT_PROTECT

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "oppmodel-finetune-report"
VERSION = 2  # 2: the date seal, validation joint rows, the artifact's own record
DEFAULT_OLD_GAMES = "results_oppmodel/lc20261005_f100/battles.jsonl"
# The bot's own ladder games dated on or after this LOCAL day are the sealed
# confirmation set (``evaluation/oppmodel_ladder_read.FRESH_SINCE``; a unit
# test holds the two together). No argument moves it.
SEALED_SINCE = "2026-10-09"
SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_DOMAIN = "ladder_holdout"
ARTIFACT_NAME = "artifact.pt"
UNVERIFIED_SUFFIX = ".unverified"
REPORT_JSON = "finetune_report.json"
REPORT_MD = "finetune_report.md"
NAME_SUFFIX = "_ft_own"  # followed by the number of own games fitted on
# The artifact's record of what it was fitted on (see the module text).
KEY_FINETUNE = "finetune"
KEY_SEEN = "seen_games"
KEY_SOURCE_TRAINING = "source_training"
MANIFEST_MARK = "finetuned-on-ladder-holdout"
# What of the source's ``extra`` still describes the fine-tuned network (its
# shape, how it reads ratings, the formats the runtime serves it in).
KEPT_EXTRA: tuple[str, ...] = (
    "kind",
    "elo_mode",
    "config",
    "n_parameters",
    "args",
    "formats",
    "torch",
)
FOLD_GAME = "game"
FOLD_ACCOUNT = "account"
FOLD_UNITS: tuple[str, ...] = (FOLD_GAME, FOLD_ACCOUNT)
RUNNING = "FT_RUNNING"
DONE = "FT_DONE"
FAILED = "FT_FAILED"
TAKEN = "FT_TAKEN"
NOT_TAKEN = "FT_NOT_TAKEN"

# The pre-registered grid and rule (2026-10-10): see the module text.
GRID_LEARNING_RATES: tuple[float, ...] = (1e-5, 3e-5, 1e-4)
GRID_PASSES: tuple[int, ...] = (1, 2, 4)
CORPUS_PER_DOMAIN = 4  # corpus rows of a batch per domain row
DEFAULT_BATCH = 160  # 32 domain rows + 128 corpus rows
DEFAULT_CORPUS_ROWS = 50_000
DEFAULT_FOLDS = 5
DEFAULT_SEED = 20261010
DEFAULT_RESAMPLES = 2000
JOINT_K = SC.JOINT_K
SWITCH_DROP_LIMIT = 0.02  # joint coverage of replies holding a switch: 2 points
RELOAD_TOLERANCE = 1.0e-6
RELOAD_ROWS = 2000  # validation rows of the reload check, beside the domain rows

# What a settings row keeps of ``compare`` on validation (the all-games fit)
# and on the rows that fit was made on (in-sample).
VAL_ROWS: tuple[str, ...] = ("fine", "fine_by_class", "top1", "top3", "joint_top8")
IN_SAMPLE_ROWS: tuple[str, ...] = ("fine", "top1", "joint_top8")

CLASS_SWITCH = "visible switch"
CLASS_PROTECT = "visible Protect-family move"
CLASS_MOVE = "other visible move"
CLASS_CENSORED = "censored"
SLOT_CLASSES: tuple[str, ...] = (
    CLASS_SWITCH,
    CLASS_PROTECT,
    CLASS_MOVE,
    CLASS_CENSORED,
)
JOINT_ALL = "all"
JOINT_SWITCH = "reply holds a switch"
JOINT_PROTECT = "reply holds a Protect-family move"
JOINT_SLICES: tuple[str, ...] = (JOINT_ALL, JOINT_SWITCH, JOINT_PROTECT)

RULE_TEXT = (
    "The setting with the best pooled out-of-fold fine NLL is taken only if "
    "(1) the 95% interval of its change in fine NLL lies below zero, and "
    "(2) its change in joint top-8 coverage is not below zero by more than "
    "its own interval's half-width, and (3) the joint top-8 coverage of "
    "replies that hold a switch does not fall by more than 2 points. "
    "Otherwise FT_NOT_TAKEN."
)

Prediction = dict[str, np.ndarray]
Log = Callable[[str], None]


class FinetuneError(RuntimeError):
    """The run cannot go on (bad input, a failed check)."""


def say(text: str) -> None:
    print(text, flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--artifact", required=True, help="a trained OppNet artifact")
    parser.add_argument("--dataset", required=True, help="the build it was trained on")
    parser.add_argument(
        "--old-games",
        default=DEFAULT_OLD_GAMES,
        help="battles.jsonl whose ladder_holdout battle ids are the allowed games",
    )
    parser.add_argument("--out", required=True, help="a NEW directory")
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help=f"rows of a batch: 1 part domain to {CORPUS_PER_DOMAIN} parts corpus",
    )
    parser.add_argument(
        "--corpus-rows",
        type=int,
        default=DEFAULT_CORPUS_ROWS,
        help="size of the random sample of the train split",
    )
    parser.add_argument(
        "--val-limit", type=int, default=None, help="at most this many validation rows"
    )
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--refit-temperatures",
        action="store_true",
        help="refit the temperatures of every fine-tuned network on validation",
    )
    parser.add_argument("--tag", default=None, help="the fine-tuned artifact's name")
    parser.add_argument(
        "--fold-unit",
        choices=FOLD_UNITS,
        default=FOLD_GAME,
        help="what a fold never splits: a game (the pre-registered folds) or an "
        "opponent account (a sensitivity reading)",
    )
    parser.add_argument(
        "--no-artifact",
        action="store_true",
        help="a robustness run: report the verdict, write no artifact",
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=list(GRID_LEARNING_RATES),
        help="tests and smoke runs only: another grid is not the pre-registered one",
    )
    parser.add_argument(
        "--passes",
        type=int,
        nargs="+",
        default=list(GRID_PASSES),
        help="tests and smoke runs only: another grid is not the pre-registered one",
    )
    return parser.parse_args(argv)


# --- files ----------------------------------------------------------------------


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: Path | str) -> Path:
    found = Path(path)
    return found if found.is_absolute() else ROOT / found


def holdout_battles(path: Path | str) -> dict[int, dict[str, Any]]:
    """Row index -> ``{'id', 'time'}`` of the ladder-holdout battles of a
    ``battles.jsonl`` (the battles with a side in the ``ladder_holdout`` split).

    The index is the row's ``index`` field (what ``m_battle`` holds), or its
    line number in a file without one. Raises ``OSError`` / ``ValueError``.
    """
    out: dict[int, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            sides = row.get("split")
            if not isinstance(sides, Mapping) or SPLIT_DOMAIN not in sides.values():
                continue
            index = row.get("index")
            out[int(number if index is None else index)] = {
                "id": str(row["id"]),
                "time": row.get("time"),
            }
    return out


def sealed_start() -> float:
    """Unix seconds of the first moment of ``SEALED_SINCE`` (a local day, as
    the ladder read counts it): a game dated at or after it is sealed."""
    return time.mktime(time.strptime(SEALED_SINCE, "%Y-%m-%d"))


def _dated(value: Any) -> float | None:
    """A battle's ``time`` as Unix seconds; None when the row has no usable one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0.0 else None


def allowed_games(path: Path | str) -> set[str]:
    """The battle ids a fine-tune may use: the ladder-holdout ids of ``path``.

    Raises ``FinetuneError`` when the file names none, and when it names a
    game dated on or after ``SEALED_SINCE``: such a file is not a list of old
    games (a newer build's ``battles.jsonl`` given by mistake), and no part of
    it is used. A game the file does not date is allowed by the list alone;
    ``load_parts`` then needs the dataset's own date for it.
    """
    rows = list(holdout_battles(path).values())
    found = {str(row["id"]) for row in rows}
    if not found:
        raise FinetuneError(
            f"{path} names no {SPLIT_DOMAIN} game: nothing is allowed, nothing is run"
        )
    seal = sealed_start()
    late = sorted(
        (when, str(row["id"]))
        for row in rows
        if (when := _dated(row["time"])) is not None and when >= seal
    )
    if late:
        raise FinetuneError(
            f"{path} names {len(late)} {SPLIT_DOMAIN} games dated on or after "
            f"{SEALED_SINCE}, the sealed confirmation set (first: {late[0][1]}, "
            f"{_stamp(late[0][0])}): it is not a list of old games, nothing is run"
        )
    return found


def seen_games(extra: Mapping[str, Any] | None) -> frozenset[str]:
    """The battle ids an artifact was FINE-TUNED on, from its ``extra``.

    Empty for an artifact this script did not write. Any reading of the
    artifact on one of these games is in-sample. A record without the list
    (the first runs of 2026-10-10 stored only its hash) is read from the
    allowed-games file it names, when that file still gives that hash.
    Raises ``FinetuneError`` when the games cannot be named: such an artifact
    has seen own ladder games and cannot say which, so every own game before
    the seal counts as seen by it.
    """
    record = (extra or {}).get(KEY_FINETUNE)
    if not isinstance(record, Mapping):
        return frozenset()
    listed = record.get(KEY_SEEN)
    if isinstance(listed, (list, tuple)) and listed:
        return frozenset(str(name) for name in listed)
    try:
        named = sorted(
            str(row["id"])
            for row in holdout_battles(str(record["allowed_games_file"])).values()
        )
    except (KeyError, OSError, ValueError):
        named = []
    digest = hashlib.sha256("\n".join(named).encode()).hexdigest()
    if named and digest == record.get("domain_used_ids_sha256"):
        return frozenset(named)
    raise FinetuneError(
        f"the artifact was fine-tuned on {record.get('domain_games', 'some')} "
        "own ladder games and its record does not list them: every own "
        f"ladder game before {SEALED_SINCE} counts as seen by it"
    )


def check_unseen(
    extra: Mapping[str, Any] | None, battle_ids: Any, what: str = "this reading"
) -> None:
    """Raise ``FinetuneError`` when the artifact with this ``extra`` was
    fine-tuned on any of ``battle_ids``: ``what`` would be in-sample there.

    For a reader of own ladder games (a scorecard's ladder holdout, a ladder
    read of old replay directories). An artifact that was never fine-tuned
    passes; one whose record lists no games fails (``seen_games``).
    """
    seen = seen_games(extra)
    if not seen:
        return
    shared = sorted(seen & {str(name) for name in battle_ids})
    if shared:
        raise FinetuneError(
            f"{what} holds {len(shared)} games the artifact was fine-tuned on "
            f"(first: {shared[0]}): it would be in-sample. Its honest numbers "
            "on those games are the out-of-fold ones of its fine-tune report"
        )


# --- data -----------------------------------------------------------------------


@dataclass
class Parts:
    """What a run reads of a dataset. ``domain_ids`` is the battle id of every
    domain row, ``used_ids`` the sorted ids of the games they come from;
    ``counts`` is the report's account of what was kept and dropped."""

    domain: F.Batch
    corpus: F.Batch
    val: F.Batch
    manifest: dict[str, Any]
    domain_ids: np.ndarray
    counts: dict[str, Any]
    used_ids: list[str]


def _rows(batch: Mapping[str, np.ndarray]) -> int:
    return int(np.asarray(batch["act_mon"]).shape[0]) if "act_mon" in batch else 0


def load_parts(
    dataset: Path | str, allowed: set[str], corpus_rows: int, seed: int
) -> Parts:
    """The domain rows, a sample of the train split and the validation rows.

    Shard by shard: a ``ladder_holdout`` row is kept only when its battle id
    (``m_battle`` -> the dataset's ``battles.jsonl``) is in ``allowed``; every
    other row of that split is dropped here and counted. The train sample is
    drawn without replacement over the whole split (the split is counted
    first, from ``m_split`` alone). No other split leaves a shard.

    The seal is checked on the DATASET's own dates before a shard is opened:
    an allowed game that ``battles.jsonl`` dates on or after ``SEALED_SINCE``,
    or does not date, raises ``FinetuneError`` (the list is wrong for this
    build; nothing is dropped quietly). So does a dataset with no allowed
    game.
    """
    source = Path(dataset)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    names = list(manifest.get("splits") or [])
    for wanted in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_DOMAIN):
        if wanted not in names:
            raise FinetuneError(f"{source} has no split {wanted!r}")
    code = {name: names.index(name) for name in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_DOMAIN)}
    battles = holdout_battles(source / "battles.jsonl")
    seal = sealed_start()
    dates = {index: _dated(row["time"]) for index, row in battles.items()}
    sealed_here = {
        index for index, when in dates.items() if when is not None and when >= seal
    }
    listed = sorted(index for index, row in battles.items() if row["id"] in allowed)
    late = [index for index in listed if index in sealed_here]
    if late:
        raise FinetuneError(
            f"{source}: {len(late)} allowed games are dated on or after "
            f"{SEALED_SINCE} here, the sealed confirmation set (first: "
            f"{battles[late[0]]['id']}, {_stamp(dates[late[0]] or 0.0)}): the "
            "list is wrong for this build, nothing is run"
        )
    undated = [index for index in listed if dates[index] is None]
    if undated:
        raise FinetuneError(
            f"{source}: {len(undated)} allowed games have no date here (first: "
            f"{battles[undated[0]]['id']}): they cannot be shown to be older "
            f"than {SEALED_SINCE}, nothing is run"
        )
    usable = np.asarray(listed, dtype=np.int64)
    shards = [source / str(shard["file"]) for shard in manifest.get("shards") or []]

    train_counts: list[int] = []
    for shard in shards:
        with np.load(shard, allow_pickle=False) as data:
            train_counts.append(int((data["m_split"] == code[SPLIT_TRAIN]).sum()))
    train_total = int(sum(train_counts))
    want = max(0, min(int(corpus_rows), train_total))
    rng = np.random.default_rng(int(seed))
    picked = np.sort(rng.choice(train_total, size=want, replace=False))

    domain: list[F.Batch] = []
    corpus: list[F.Batch] = []
    val: list[F.Batch] = []
    dropped_rows = 0
    dropped_games: set[int] = set()
    split_rows = 0
    offset = 0
    for shard, count in zip(shards, train_counts):
        batch = F.load_batch(shard)
        split = np.asarray(batch["m_split"])
        game = np.asarray(batch["m_battle"]).astype(np.int64)
        held = split == code[SPLIT_DOMAIN]
        keep = held & np.isin(game, usable)
        split_rows += int(held.sum())
        dropped_rows += int((held & ~keep).sum())
        dropped_games.update(int(value) for value in np.unique(game[held & ~keep]))
        if keep.any():
            domain.append(F.take(batch, keep))
        if (split == code[SPLIT_VAL]).any():
            val.append(F.take(batch, split == code[SPLIT_VAL]))
        local = picked[(picked >= offset) & (picked < offset + count)] - offset
        if local.size:
            corpus.append(
                F.take(batch, np.flatnonzero(split == code[SPLIT_TRAIN])[local])
            )
        offset += count

    domain_rows = F.concat_batches(domain)
    if _rows(domain_rows) == 0:
        raise FinetuneError(
            f"{source}: none of the {len(allowed)} allowed games has a "
            f"{SPLIT_DOMAIN} row here"
        )
    games = np.asarray(domain_rows["m_battle"]).astype(np.int64)
    ids = np.asarray([battles[int(index)]["id"] for index in games])
    used = sorted({str(battles[int(index)]["id"]) for index in np.unique(games)})
    outside = [name for name in used if name not in allowed]
    if outside:  # cannot happen: ``keep`` is the filter; checked all the same
        raise FinetuneError(f"domain rows of games outside the list: {outside[:3]}")
    used_index = {int(index) for index in np.unique(games)}
    used_sealed = sorted(
        index
        for index in used_index
        if (dates.get(index) is None) or index in sealed_here
    )
    if used_sealed:  # cannot happen either: checked before a shard was opened
        raise FinetuneError(
            f"{len(used_sealed)} used games are not dated before {SEALED_SINCE}"
        )
    times = [dates[index] or 0.0 for index in sorted(used_index)]
    corpus_rows_found = F.concat_batches(corpus)
    val_rows = F.concat_batches(val)
    for name, part, split_name in (
        ("corpus", corpus_rows_found, SPLIT_TRAIN),
        ("validation", val_rows, SPLIT_VAL),
    ):
        if _rows(part) == 0:
            raise FinetuneError(f"{source}: no {name} row ({split_name} split)")
        if not (np.asarray(part["m_split"]) == code[split_name]).all():
            raise FinetuneError(f"{name} rows outside the {split_name} split")
    listed_here = {row["id"] for row in battles.values()}
    counts = {
        "allowed_games_listed": len(allowed),
        "domain": {
            "split": SPLIT_DOMAIN,
            "rows_in_split": split_rows,
            "games_in_split": len(battles),
            "rows_used": _rows(domain_rows),
            "games_used": len(used),
            "rows_dropped_game_not_allowed": dropped_rows,
            "games_dropped_not_allowed": len(dropped_games),
            "allowed_games_without_rows_here": len(allowed - set(used)),
            "allowed_games_not_in_this_build": len(allowed - listed_here),
            "used_games_outside_allowed": len(outside),
            "used_ids_sha256": hashlib.sha256("\n".join(used).encode()).hexdigest(),
            "time_first": _stamp(min(times)) if times else None,
            "time_last": _stamp(max(times)) if times else None,
            "sealed_since": SEALED_SINCE,
            "games_in_split_dated_sealed": len(sealed_here),
            "games_dropped_dated_sealed": len(dropped_games & sealed_here),
            "games_used_dated_sealed": len(used_sealed),
            "games_used_undated": 0,
        },
        "corpus": {
            "split": SPLIT_TRAIN,
            "rows_in_split": train_total,
            "rows_sampled": _rows(corpus_rows_found),
        },
        "val": {"split": SPLIT_VAL, "rows_in_split": _rows(val_rows)},
    }
    return Parts(domain_rows, corpus_rows_found, val_rows, manifest, ids, counts, used)


def _stamp(seconds: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(float(seconds)))


def game_folds(games: Any, folds: int, seed: int) -> dict[int, int]:
    """Game -> fold (0 .. folds - 1): a seeded split of the distinct games.

    Every game is in exactly one fold and the fold sizes differ by at most
    one game. Raises ``FinetuneError`` for fewer than two folds or fewer
    games than folds.
    """
    unique = np.unique(np.asarray(games).astype(np.int64))
    count = int(folds)
    if count < 2:
        raise FinetuneError("cross-validation needs at least two folds")
    if unique.size < count:
        raise FinetuneError(f"{unique.size} games cannot fill {count} folds")
    order = np.random.default_rng(int(seed)).permutation(unique.size)
    return {int(unique[row]): place % count for place, row in enumerate(order)}


def account_folds(games: Any, accounts: Any, folds: int, seed: int) -> dict[int, int]:
    """Game -> fold with every ACCOUNT's games in one fold (``--fold-unit
    account``): a seeded split of the distinct accounts, each game taking the
    fold of the account of its rows.

    ``games`` and ``accounts`` hold one entry per row. The accounts are dealt
    out one by one, so the folds' numbers of accounts differ by at most one
    (their numbers of games can differ by more). Raises ``FinetuneError`` for
    fewer than two folds, fewer accounts than folds, or a game with rows of
    two accounts.
    """
    game = np.asarray(games).astype(np.int64).reshape(-1)
    actor = np.asarray(accounts).astype(np.int64).reshape(-1)
    if game.shape != actor.shape:
        raise FinetuneError("one account per row is needed to fold by account")
    owner: dict[int, int] = {}
    for one, who in zip(game.tolist(), actor.tolist()):
        if owner.setdefault(int(one), int(who)) != int(who):
            raise FinetuneError(
                f"game {one} has rows of two accounts: it cannot be folded by account"
            )
    unique = np.unique(actor)
    count = int(folds)
    if count < 2:
        raise FinetuneError("cross-validation needs at least two folds")
    if unique.size < count:
        raise FinetuneError(f"{unique.size} accounts cannot fill {count} folds")
    order = np.random.default_rng(int(seed)).permutation(unique.size)
    fold_of = {int(unique[row]): place % count for place, row in enumerate(order)}
    return {one: fold_of[who] for one, who in owner.items()}


def fold_of_rows(games: Any, assignment: Mapping[int, int]) -> np.ndarray:
    """The fold of every row, from the fold of its game."""
    return np.asarray(
        [assignment[int(game)] for game in np.asarray(games).reshape(-1)],
        dtype=np.int64,
    )


def check_disjoint(train_games: Any, test_games: Any) -> None:
    """Raise ``FinetuneError`` when a game is on both sides of a fold."""
    shared = np.intersect1d(np.asarray(train_games), np.asarray(test_games))
    if shared.size:
        raise FinetuneError(
            f"{shared.size} games are on both sides of a fold (first: {shared[:3]})"
        )


# --- the fine-tune --------------------------------------------------------------


@dataclass(frozen=True)
class Recipe:
    """What a fine-tune step keeps from the source's training, and the batch mix."""

    domain_batch: int
    corpus_per_domain: int
    slot_swap: float
    elo_drop: float
    clip: float
    mega_weight: float
    weight_decay: float

    @classmethod
    def from_training(
        cls, stored: Mapping[str, Any] | None, *, batch: int, elo_blind: bool
    ) -> "Recipe":
        """From the arguments an artifact says it was trained with (``extra['args']``);
        a missing one reads as the trainer's default. An Elo-blind model has no
        rating drop: its ratings are blanked everywhere."""
        known = vars(T.parse_args([]))
        known.update({key: value for key, value in dict(stored or {}).items()})
        domain_batch = max(1, int(batch) // (1 + CORPUS_PER_DOMAIN))
        return cls(
            domain_batch=domain_batch,
            corpus_per_domain=CORPUS_PER_DOMAIN,
            slot_swap=float(known["slot_swap"]),
            elo_drop=0.0 if elo_blind else float(known["elo_drop"]),
            clip=float(known["clip"]),
            mega_weight=float(known["mega_weight"]),
            weight_decay=float(known["weight_decay"]),
        )


def balanced_weight(
    weight: torch.Tensor, scored: torch.Tensor, n_domain: int
) -> torch.Tensor:
    """Example weights of a mixed batch whose first ``n_domain`` rows are domain
    rows: those are scaled so that the two parts hold the same weighted number
    of scored slots, which makes ``model.total_loss``'s fine term half the
    domain rows' weighted mean and half the corpus rows'. ``weight`` is the
    account-cap weight, ``scored`` the ``action_scored`` mask ``[N, 2]``. A
    batch with nothing scored in one part comes back unchanged.
    """
    slots = scored.sum(-1).to(weight.dtype)
    mass = weight * slots
    domain, corpus = mass[:n_domain].sum(), mass[n_domain:].sum()
    out = weight.clone()
    if float(domain) > 0.0 and float(corpus) > 0.0:
        out[:n_domain] = out[:n_domain] * (corpus / domain)
    return out


class _Cursor:
    """Corpus rows without repetition until the sample is used up, then reshuffled."""

    def __init__(self, size: int, rng: np.random.Generator) -> None:
        self.size, self.rng = int(size), rng
        self.order = rng.permutation(self.size)
        self.at = 0
        self.reshuffles = 0

    def take(self, count: int) -> np.ndarray:
        parts: list[np.ndarray] = []
        left = int(count)
        while left > 0:
            if self.at >= self.size:
                self.order = self.rng.permutation(self.size)
                self.at = 0
                self.reshuffles += 1
            part = self.order[self.at : self.at + left]
            parts.append(part)
            self.at += int(part.size)
            left -= int(part.size)
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


def finetune(
    net: M.OppNet,
    domain: Mapping[str, np.ndarray],
    corpus: Mapping[str, np.ndarray],
    *,
    lr: float,
    checkpoints: Sequence[int],
    recipe: Recipe,
    seed: int,
    device: torch.device | str = "cpu",
) -> tuple[dict[int, dict[str, torch.Tensor]], list[dict[str, Any]]]:
    """Continue training ``net`` IN PLACE; (weights after each checkpoint, log).

    ``checkpoints`` are pass counts (one pass = every domain row once); the
    run lasts the largest and a cloned state is kept after each. ``domain``
    and ``corpus`` are prepared batches (sheet codes mapped, ratings blanked
    for an Elo-blind model) with labels and ``m_weight``. The loss is the
    trainer's (``model.nll_terms`` -> ``model.total_loss``) with
    ``balanced_weight``; AdamW at the constant rate ``lr`` over the trainer's
    parameter groups; dropout on; the gradient norm capped at
    ``recipe.clip``. The log has one row per pass. Raises ``FinetuneError``
    for empty data, a bad checkpoint or a non-finite loss.
    """
    wanted = sorted({int(value) for value in checkpoints})
    if not wanted or wanted[0] < 1:
        raise FinetuneError(f"pass counts must be positive: {list(checkpoints)}")
    n_domain, n_corpus = _rows(domain), _rows(corpus)
    if n_domain == 0 or n_corpus == 0:
        raise FinetuneError(f"empty data: domain {n_domain}, corpus {n_corpus}")
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    net.to(device)
    net.train()
    optimizer = torch.optim.AdamW(
        T.parameter_groups(net, recipe.weight_decay), lr=float(lr)
    )
    size, ratio = recipe.domain_batch, recipe.corpus_per_domain
    steps_per_pass = math.ceil(n_domain / size)
    cursor = _Cursor(n_corpus, rng)
    states: dict[int, dict[str, torch.Tensor]] = {}
    log: list[dict[str, Any]] = []
    step = 0
    try:
        for done in range(1, wanted[-1] + 1):
            started = time.time()
            view = T.augment(domain, rng, recipe.slot_swap, recipe.elo_drop)
            order = rng.permutation(n_domain)
            drawn = F.take(corpus, cursor.take(ratio * n_domain))
            others = T.augment(drawn, rng, recipe.slot_swap, recipe.elo_drop)
            sums = {"loss": 0.0, "domain_fine": 0.0, "corpus_fine": 0.0}
            for position in range(steps_per_pass):
                rows = np.sort(order[position * size : (position + 1) * size])
                first = position * size * ratio
                mixed = F.concat_batches(
                    [
                        F.take(view, rows),
                        F.take(others, slice(first, first + rows.size * ratio)),
                    ]
                )
                labels = M.to_labels(mixed, device)
                terms = M.nll_terms(
                    net(M.to_tensors(mixed, device, None, net.extra_keys)), labels
                )
                weight = balanced_weight(
                    labels["weight"], terms["action_scored"], int(rows.size)
                )
                loss, parts = M.total_loss(terms, weight, recipe.mega_weight)
                if not math.isfinite(parts["loss"]):
                    raise FinetuneError(f"non-finite loss at step {step}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), recipe.clip)
                optimizer.step()
                step += 1
                sums["loss"] += parts["loss"]
                for name, part in _part_means(terms, labels["weight"], int(rows.size)):
                    sums[name] += part
            log.append(
                {
                    "pass": done,
                    "steps": step,
                    "loss": sums["loss"] / steps_per_pass,
                    "domain_fine": sums["domain_fine"] / steps_per_pass,
                    "corpus_fine": sums["corpus_fine"] / steps_per_pass,
                    "corpus_reshuffles": cursor.reshuffles,
                    "seconds": time.time() - started,
                }
            )
            if done in wanted:
                states[done] = M.clone_state(net)
    finally:
        net.eval()
    return states, log


def _part_means(
    terms: Mapping[str, torch.Tensor], weight: torch.Tensor, n_domain: int
) -> list[tuple[str, float]]:
    """The weighted mean fine loss of each part of a mixed batch (for the log)."""
    out: list[tuple[str, float]] = []
    fine = (terms["action"] + terms["target"]).detach()
    scored = terms["action_scored"].detach().to(fine.dtype)
    per_slot = weight[:, None]
    for name, part in (
        ("domain_fine", slice(0, n_domain)),
        ("corpus_fine", slice(n_domain, None)),
    ):
        slots = float((per_slot[part] * scored[part]).sum())
        total = float((per_slot[part] * fine[part]).sum())
        out.append((name, total / slots if slots > 0 else 0.0))
    return out


# --- predictions and their comparison -------------------------------------------


def predictor_of(
    net: M.OppNet,
    source: M.OppNetPredictor,
    temperatures: Mapping[str, float] | None = None,
    name: str | None = None,
) -> M.OppNetPredictor:
    """A predictor over ``net`` with the source's Elo mode and (unless others
    are given) the source's temperatures; no event calibration."""
    held = (
        dict(temperatures) if temperatures is not None else source_temperatures(source)
    )
    return M.OppNetPredictor(
        net,
        source.featurizer,
        name=source.name if name is None else name,
        action_temperature=held["action"],
        target_temperature=held["target"],
        elo_mode=source.elo_mode,
        mega_temperature=held["mega"],
        mega_bias=held["mega_bias"],
    )


def source_temperatures(source: M.OppNetPredictor) -> dict[str, float]:
    return {
        "action": float(source.action_temperature),
        "target": float(source.target_temperature),
        "mega": float(source.mega_temperature),
        "mega_bias": float(source.mega_bias),
    }


def predict_rows(
    predictor: M.OppNetPredictor, batch: Mapping[str, np.ndarray]
) -> Prediction:
    """``predict`` on the features alone (no label, no bookkeeping array), as
    float64. Raises ``FinetuneError`` when the predictor fell back to uniform."""
    before = SC.predict_errors(predictor)
    made = predictor.predict(F.sheet_unknown_as_closed(M.strip_labels(batch)))
    if SC.predict_errors(predictor) != before:
        raise FinetuneError(f"predict failed: {dict(predictor.counters)}")
    return {
        name: np.asarray(made[name], dtype=np.float64)
        for name in ("action", "target", "mega")
    }


def slot_classes(batch: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Four ``[N, 2]`` masks that partition the scored slots (``SLOT_CLASSES``):
    a visible switch, a visible Protect-family candidate, any other visible
    move (the OTHER bucket included), and a censored slot (scored by its set).
    """
    mask = np.asarray(batch["action_mask"]).astype(bool)
    scored = (np.asarray(batch["y_set"]).astype(bool) & mask).any(-1)
    y_action = np.asarray(batch["y_action"]).astype(np.int64)
    flags = np.asarray(batch["cand_flag"]).astype(np.int64)
    n_cand = flags.shape[-1]
    visible = scored & (y_action >= 0)
    switch = visible & (y_action > n_cand)
    column = np.clip(y_action, 0, max(0, n_cand - 1))[..., None]
    own = np.take_along_axis(flags, column, axis=-1)[..., 0]
    protect = visible & (y_action < n_cand) & ((own & F.CAND_PROTECT) > 0)
    return {
        CLASS_SWITCH: switch,
        CLASS_PROTECT: protect,
        CLASS_MOVE: visible & ~switch & ~protect,
        CLASS_CENSORED: scored & (y_action < 0),
    }


def joint_hits(
    pred: Mapping[str, np.ndarray], batch: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(the true joint reply is among the predictor's first ``JOINT_K``, the
    slices), both ``[N]`` over all rows; a slice is False on a row that is not
    counted. Counted are the rows whose acting slots are all visible
    (``joint.true_replies``); a true reply that needs the OTHER bucket is a
    miss: the scorecard's definition. Raises ``FinetuneError`` when the
    prediction cannot be read as joint replies.
    """
    truth = J.true_replies(batch)
    if not truth:
        raise FinetuneError(f"labels are not joint replies ({dict(J.COUNTERS)})")
    visible = np.asarray(truth["visible"]).astype(bool)
    told = {name: np.asarray(values)[visible] for name, values in truth.items()}
    part = M.strip_labels(F.take(batch, visible))
    own = {name: np.asarray(values)[visible] for name, values in pred.items()}
    made = J.joint_replies(own, part, k=JOINT_K, truth=told)
    if made.n != int(visible.sum()):
        raise FinetuneError(f"no joint replies ({dict(J.COUNTERS)})")
    other = told["other"].any(-1)
    hit = np.zeros(visible.shape, dtype=bool)
    hit[visible] = (made.rank >= 0) & (made.rank < JOINT_K) & ~other
    slices = {JOINT_ALL: visible.copy()}
    for name, key in ((JOINT_SWITCH, "switch"), (JOINT_PROTECT, INTENT_PROTECT)):
        held = np.zeros(visible.shape, dtype=bool)
        held[visible] = told[key].any(-1)
        slices[name] = held
    return hit, slices


def _row(draws: SC.Draws, sums: SC.GameSums, key: Any) -> dict[str, Any]:
    """One paired row of ``compare``: base, new, their difference and its interval."""
    count = (key, "count")
    change = draws.ratio((key, "change"), count)
    games = int((sums.column(count) > 0).sum())
    if games < 2:  # one game resampled is that game again
        change["low"] = change["high"] = None
    return {
        "base": draws.ratio((key, "base"), count)["value"],
        "new": draws.ratio((key, "new"), count)["value"],
        "diff": change["value"],
        "low": change["low"],
        "high": change["high"],
        "n": int(round(draws.total(count))),
        "games": games,
    }


def compare(
    batch: Mapping[str, np.ndarray],
    base: Mapping[str, np.ndarray],
    new: Mapping[str, np.ndarray],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    joint: bool = True,
) -> dict[str, Any]:
    """``new`` against ``base`` on the same rows: every number of one setting.

    Each row holds ``base``, ``new``, ``diff`` (new - base), the 95% interval
    of the difference from ``resamples`` resamples of whole games
    (``m_battle``; one set of resampled games for every row), ``n`` (slots,
    or turns for the joint rows) and ``games``. Keys: ``fine`` (NLL), ``fine_by_class``
    (``SLOT_CLASSES``), ``action`` and ``target`` (the two terms of the fine
    NLL, each over its own scored slots), ``top1`` / ``top3`` (fine) and, with
    ``joint``, ``joint_top8`` by ``JOINT_SLICES``.
    """
    scores = {"base": F.slot_nll(base, batch), "new": F.slot_nll(new, batch)}
    sums = SC.GameSums(np.asarray(batch["m_battle"]))

    def pair(key: Any, first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> None:
        one = np.asarray(first, dtype=np.float64)
        two = np.asarray(second, dtype=np.float64)
        sums.add((key, "base"), one, mask)
        sums.add((key, "new"), two, mask)
        sums.add((key, "change"), two - one, mask)
        sums.count((key, "count"), mask)

    scored = scores["base"]["fine_scored"]
    pair("fine", scores["base"]["fine"], scores["new"]["fine"], scored)
    for term in ("action", "target"):
        pair(
            term,
            scores["base"][term],
            scores["new"][term],
            scores["base"][f"{term}_scored"],
        )
    classes = slot_classes(batch)
    for name, mask in classes.items():
        pair(("fine", name), scores["base"]["fine"], scores["new"]["fine"], mask)
    label = F.fine_label(batch)
    ranked = {"base": F.fine_probs(base, batch), "new": F.fine_probs(new, batch)}
    for k in (1, 3):
        first, counted = F.topk_hits(ranked["base"], label, k)
        second, _ = F.topk_hits(ranked["new"], label, k)
        pair(f"top{k}", first, second, counted)
    slices: dict[str, np.ndarray] = {}
    if joint:
        first_hit, slices = joint_hits(base, batch)
        second_hit, _ = joint_hits(new, batch)
        for name, mask in slices.items():
            pair(("joint", name), first_hit, second_hit, mask)
    draws = sums.draws(int(resamples), int(seed))
    out: dict[str, Any] = {
        "fine": _row(draws, sums, "fine"),
        "action": _row(draws, sums, "action"),
        "target": _row(draws, sums, "target"),
        "fine_by_class": {name: _row(draws, sums, ("fine", name)) for name in classes},
        "top1": _row(draws, sums, "top1"),
        "top3": _row(draws, sums, "top3"),
        "resamples": draws.resamples,
        "games": sums.n_games,
        "rows": _rows(batch),
    }
    if joint:
        out["joint_top8"] = {
            name: _row(draws, sums, ("joint", name)) for name in slices
        }
    return out


# --- the rule -------------------------------------------------------------------


def setting_key(lr: float, passes: int) -> str:
    return f"lr {float(lr):g} x {int(passes)} passes"


def decide(settings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The pre-registered rule over the settings' out-of-fold rows.

    ``settings`` rows hold ``key`` and ``oof`` (a ``compare`` result with the
    joint rows). The best setting is the one with the lowest pooled
    out-of-fold fine NLL (the first of the grid among equals). Returns the
    best key, the three conditions with the numbers they read, and ``verdict``
    (``FT_TAKEN`` only when all three hold; a missing interval fails).
    """
    if not settings:
        raise FinetuneError("no setting to decide on")
    best = min(settings, key=lambda row: float(row["oof"]["fine"]["new"]))
    fine = best["oof"]["fine"]
    joint = best["oof"]["joint_top8"][JOINT_ALL]
    switch = best["oof"]["joint_top8"][JOINT_SWITCH]
    half = (
        None
        if joint["low"] is None or joint["high"] is None
        else (float(joint["high"]) - float(joint["low"])) / 2.0
    )
    conditions = {
        "fine_nll_interval_below_zero": {
            "holds": fine["high"] is not None and float(fine["high"]) < 0.0,
            "diff": fine["diff"],
            "low": fine["low"],
            "high": fine["high"],
        },
        "joint_top8_not_below_zero_by_more_than_its_half_width": {
            "holds": half is not None
            and joint["diff"] is not None
            and float(joint["diff"]) >= -half,
            "diff": joint["diff"],
            "low": joint["low"],
            "high": joint["high"],
            "half_width": half,
        },
        "switch_reply_coverage_not_down_more_than_2_points": {
            "holds": switch["diff"] is not None
            and float(switch["diff"]) >= -SWITCH_DROP_LIMIT,
            "diff": switch["diff"],
            "low": switch["low"],
            "high": switch["high"],
            "limit": -SWITCH_DROP_LIMIT,
            "n": switch["n"],
        },
    }
    taken = all(bool(found["holds"]) for found in conditions.values())
    return {
        "rule": RULE_TEXT,
        "best": best["key"],
        "best_oof_fine_nll": fine["new"],
        "conditions": conditions,
        "verdict": TAKEN if taken else NOT_TAKEN,
    }


# --- beside the rule: what it does not gate ---------------------------------------

PLACE_OOF = "oof"  # the old games, every row predicted out of fold
PLACE_VAL = "val"  # validation, by the network fitted on all allowed games
PLACES: dict[str, str] = {
    PLACE_OOF: "old games, out of fold",
    PLACE_VAL: "validation, all-games fit",
}
READ_JOINT = "joint top-8, all turns"
READ_JOINT_SWITCH = "joint top-8, reply holds a switch"
READ_JOINT_PROTECT = "joint top-8, reply holds a Protect-family move"
READ_NLL_PROTECT = "fine NLL, slots where a Protect-family move was clicked"
READ_NLL_SWITCH = "fine NLL, slots where a switch was made"
# (reading, True when a LARGER number is the worse one)
NOT_GATED_READINGS: tuple[tuple[str, bool], ...] = (
    (READ_JOINT, False),
    (READ_JOINT_SWITCH, False),
    (READ_JOINT_PROTECT, False),
    (READ_NLL_PROTECT, True),
    (READ_NLL_SWITCH, True),
)
NOT_GATED_NOTE = (
    "Reported, not gated: the pre-registered rule gates the fine NLL, joint "
    "top-8 and switch replies of the old games only. A cost is a reading whose "
    "95% interval lies wholly on the worse side."
)


def _worse(row: Mapping[str, Any], larger_is_worse: bool) -> bool:
    """A ``compare`` row whose interval lies wholly on the worse side of zero."""
    low, high = row.get("low"), row.get("high")
    if low is None or high is None:
        return False
    return float(low) > 0.0 if larger_is_worse else float(high) < 0.0


def not_gated(
    best: Mapping[str, Any], corrected: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The best setting's readings that the rule does not gate.

    ``best`` is a settings row: ``oof`` and ``val`` are ``compare`` results
    (``val`` may lack rows, which are then left out). Returns ``readings``
    (place -> reading -> the ``compare`` row), ``costs`` (the ``place: reading``
    names whose interval lies wholly on the worse side) and
    ``selection_corrected`` as given. Nothing here changes a verdict.
    """
    readings: dict[str, dict[str, Any]] = {}
    costs: list[str] = []
    for place in PLACES:
        found = best.get(place) or {}
        joint = found.get("joint_top8") or {}
        classes = found.get("fine_by_class") or {}
        rows = {
            READ_JOINT: joint.get(JOINT_ALL),
            READ_JOINT_SWITCH: joint.get(JOINT_SWITCH),
            READ_JOINT_PROTECT: joint.get(JOINT_PROTECT),
            READ_NLL_PROTECT: classes.get(CLASS_PROTECT),
            READ_NLL_SWITCH: classes.get(CLASS_SWITCH),
        }
        kept = {name: row for name, row in rows.items() if row}
        readings[place] = kept
        costs += [
            f"{PLACES[place]}: {name}"
            for name, larger in NOT_GATED_READINGS
            if name in kept and _worse(kept[name], larger)
        ]
    return {
        "note": NOT_GATED_NOTE,
        "setting": best.get("key"),
        "readings": readings,
        "costs": costs,
        "selection_corrected": None if corrected is None else dict(corrected),
    }


def choose_without_fold(
    fine: Mapping[str, np.ndarray], scored: Any, row_fold: Any, folds: int
) -> list[str]:
    """For every fold, the setting with the lowest mean fine NLL over the
    scored slots of the OTHER folds (the first of the grid among equals).

    ``fine`` maps a setting to its out-of-fold fine NLL per slot ``[N, 2]``,
    ``scored`` is the mask of scored slots, ``row_fold`` the fold of every
    row. Raises ``FinetuneError`` without a setting, or when the other folds
    of a fold hold no scored slot.
    """
    if not fine:
        raise FinetuneError("no setting to choose from")
    mask = np.asarray(scored).astype(bool)
    place = np.asarray(row_fold).reshape(-1)
    chosen: list[str] = []
    for fold in range(int(folds)):
        keep = mask & (place != fold)[:, None]
        if not keep.any():
            raise FinetuneError(f"no scored slot outside fold {fold}")
        means = {
            key: float(np.asarray(values)[keep].mean()) for key, values in fine.items()
        }
        chosen.append(min(means, key=lambda key: means[key]))
    return chosen


def selection_corrected(
    domain: Mapping[str, np.ndarray],
    base: Mapping[str, np.ndarray],
    oof: Mapping[str, Prediction],
    row_fold: Any,
    folds: int,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """The gain when no fold is read with a setting that was chosen on it.

    The rule's best setting is the best of the grid on the very rows its gain
    is read on. Here every fold is read with the setting that
    ``choose_without_fold`` picks from the other folds, and the pooled rows
    are compared with ``base`` as every setting is. The distance to the best
    setting's own number is what that choice was worth. (The other folds'
    predictions come from networks that saw this fold's games; the fold's own
    labels are never read by the choice.) Descriptive: no rule reads it.
    """
    scores = {key: F.slot_nll(pred, domain) for key, pred in oof.items()}
    if not scores:
        raise FinetuneError("no setting to choose from")
    scored = next(iter(scores.values()))["fine_scored"]
    place = np.asarray(row_fold).reshape(-1)
    chosen = choose_without_fold(
        {key: found["fine"] for key, found in scores.items()}, scored, place, folds
    )
    mixed = {name: np.array(values) for name, values in oof[chosen[0]].items()}
    for fold, key in enumerate(chosen):
        held = place == fold
        for name in mixed:
            mixed[name][held] = oof[key][name][held]
    found = compare(domain, base, mixed, resamples=int(resamples), seed=int(seed))
    return {
        "chosen_by_fold": chosen,
        "fine": found["fine"],
        "top1": found["top1"],
        "joint_top8": found["joint_top8"],
    }


# --- the artifact's own record ----------------------------------------------------


def finetuned_extra(
    source_extra: Mapping[str, Any], *, name: str, record: Mapping[str, Any]
) -> dict[str, Any]:
    """The ``extra`` of a fine-tuned artifact: its own record, not the source's.

    ``record`` is the ``finetune`` field; it must hold ``created`` and the
    list ``seen_games`` (the battle ids fitted on). Kept at the top level:
    ``KEPT_EXTRA`` of the source (the network's shape, its Elo mode, the
    training arguments a later fine-tune would reuse) and ``dataset`` with its
    formats (the runtime stands down by them). Changed: ``name`` and
    ``created`` are the fine-tune's; ``dataset['manifest_sha256']`` and the
    top-level ``manifest_sha256`` are ``MANIFEST_MARK:<games>-games:<the
    build's sha256>``, so that a reader comparing it with a build's manifest
    (the scorecard's ``_trained_on``) finds another training set, which it is;
    the build's own sha256 is ``dataset['source_manifest_sha256']``.
    Everything else of the source (best epoch, history, calibration,
    examples, a coupling's name) moves under ``source_training``. Raises
    ``FinetuneError`` for a record without games.
    """
    source = dict(source_extra)
    games = [str(one) for one in record.get(KEY_SEEN) or []]
    if not games:
        raise FinetuneError("a fine-tuned artifact must list the games it has seen")
    out: dict[str, Any] = {
        key: copy.deepcopy(source[key]) for key in KEPT_EXTRA if key in source
    }
    before = source.get("dataset")
    dataset = copy.deepcopy(dict(before)) if isinstance(before, Mapping) else {}
    build = SC._trained_on(source)
    marked = f"{MANIFEST_MARK}:{len(games)}-games:{build}"
    tag = dataset.get("tag") or source.get("dataset_tag") or source.get("tag")
    dataset["source_manifest_sha256"] = build
    dataset["manifest_sha256"] = marked
    dataset["fitted_on"] = [SPLIT_TRAIN, f"{SPLIT_DOMAIN}: {len(games)} own games"]
    out.update(
        {
            "name": name,
            "created": record.get("created"),
            "dataset": dataset,
            "manifest_sha256": marked,
            "dataset_tag": f"{tag} + {len(games)} own ladder games "
            "(fine-tuned on them: any reading on them is in-sample)",
            KEY_FINETUNE: copy.deepcopy(dict(record)),
            KEY_SOURCE_TRAINING: copy.deepcopy(source),
        }
    )
    return out


# --- the run --------------------------------------------------------------------


def run_files(out_dir: Path) -> list[Path]:
    artifact = out_dir / ARTIFACT_NAME
    return [
        artifact,
        artifact.with_name(artifact.name + UNVERIFIED_SUFFIX),
        out_dir / REPORT_JSON,
        out_dir / REPORT_MD,
    ]


def run(args: argparse.Namespace, log: Log = say) -> dict[str, Any]:
    """One fine-tune study; returns its report. Raises on any failure.

    The output directory must not hold a report or an artifact. From then on
    it is this run's: the report says ``FT_RUNNING`` until the run ends and
    ``FT_FAILED`` with the reason if it raises.
    """
    out_dir = _resolve(args.out)
    found = [path for path in run_files(out_dir) if path.exists()]
    if found:
        raise FinetuneError(f"{found[0]} exists: give a new --out directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    opening = {
        "format": FORMAT,
        "version": VERSION,
        "status": RUNNING,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "args": {key: value for key, value in sorted(vars(args).items())},
    }
    write_report(out_dir, opening)
    try:
        return _run(args, out_dir, log)
    except BaseException as exc:
        reason = " ".join(f"{type(exc).__name__}: {exc}".split())
        try:
            failed = json.loads((out_dir / REPORT_JSON).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            failed = dict(opening)
        failed.update({"status": FAILED, "reason": reason})
        try:
            write_report(out_dir, failed)
        except OSError:
            pass
        raise


def _fit_seed(seed: int, fold: int) -> int:
    """The seed of a fit: one per fold (``fold`` = the fold count for the fit
    on all games), the same for every learning rate."""
    return int(seed) * 1000 + int(fold)


def _run(args: argparse.Namespace, out_dir: Path, log: Log) -> dict[str, Any]:
    started = time.time()
    torch.set_num_threads(max(1, int(args.threads)))
    cpu = torch.device("cpu")
    artifact_path = _resolve(args.artifact)
    dataset = _resolve(args.dataset)
    old_games = _resolve(args.old_games)
    seed = int(args.seed)
    rates = [float(value) for value in args.learning_rates]
    passes = sorted({int(value) for value in args.passes})
    if not rates or not passes or passes[0] < 1 or min(rates) < 0.0:
        raise FinetuneError(f"bad grid: rates {rates}, passes {passes}")
    pre_registered = (
        tuple(rates) == GRID_LEARNING_RATES and tuple(passes) == GRID_PASSES
    )

    loaded = A.load_predictor(artifact_path)
    source = loaded.predictor
    if not isinstance(source, M.OppNetPredictor):
        raise FinetuneError(
            f"{artifact_path} is a {loaded.kind} artifact, not an OppNet"
        )
    extra = dict(loaded.meta.get("extra") or {})
    earlier = extra.get(KEY_FINETUNE)
    if isinstance(earlier, Mapping):
        raise FinetuneError(
            f"{artifact_path} is itself fine-tuned on "
            f"{earlier.get('domain_games', 'some')} own ladder games: a "
            "cross-validation over it would be in-sample. Give the artifact it "
            "was made from"
        )
    manifest_sha = T.manifest_sha256(dataset)
    trained_on = SC._trained_on(extra)
    if trained_on is not None and trained_on != manifest_sha:
        raise FinetuneError(
            f"{artifact_path} was trained on a dataset with manifest sha256 "
            f"{trained_on}; {dataset} has {manifest_sha}"
        )
    allowed = allowed_games(old_games)
    parts = load_parts(dataset, allowed, int(args.corpus_rows), seed)
    blind = source.elo_mode == F.ELO_BLANK

    def prepare(batch: F.Batch) -> F.Batch:
        made = F.sheet_unknown_as_closed(batch)
        return F.apply_elo_mode(made, F.ELO_BLANK) if blind else made

    domain = prepare(parts.domain)
    corpus = prepare(parts.corpus)
    validation, left_out = T.without_holdout_battles(parts.val)
    val = prepare(T.subset(validation, args.val_limit, np.random.default_rng(seed)))
    masked = {
        name: M.count_masked_targets(batch)
        for name, batch in (("domain", domain), ("corpus", corpus), ("val", val))
    }
    recipe = Recipe.from_training(
        extra.get("args") if isinstance(extra.get("args"), Mapping) else None,
        batch=int(args.batch),
        elo_blind=blind,
    )
    stored = source_temperatures(source)
    base = predictor_of(source.net, source)  # the source without event calibration
    name = str(
        args.tag or f"{loaded.name}{NAME_SUFFIX}{parts.counts['domain']['games_used']}"
    )
    games = np.asarray(domain["m_battle"]).astype(np.int64)
    folds = int(args.folds)
    unit = str(getattr(args, "fold_unit", FOLD_GAME))
    actors = np.asarray(domain["m_actor"]) if "m_actor" in domain else None
    if unit == FOLD_ACCOUNT:
        if actors is None:
            raise FinetuneError("the dataset has no m_actor: no folds by account")
        assignment = account_folds(games, actors, folds, seed)
    elif unit == FOLD_GAME:
        assignment = game_folds(games, folds, seed)
    else:
        raise FinetuneError(f"unknown fold unit {unit!r}")
    row_fold = fold_of_rows(games, assignment)
    straddling = 0
    if actors is not None:
        for actor in np.unique(actors):
            straddling += int(np.unique(row_fold[actors == actor]).size > 1)
    if unit == FOLD_ACCOUNT and straddling:  # cannot happen: checked all the same
        raise FinetuneError(f"{straddling} accounts are in more than one fold")
    folds_pre_registered = unit == FOLD_GAME and folds == DEFAULT_FOLDS
    write_artifact_wanted = not bool(getattr(args, "no_artifact", False))
    counts = parts.counts
    counts["val"].update(
        {"rows_left_out_holdout_battles": left_out, "rows_used": _rows(val)}
    )
    report: dict[str, Any] = {
        "format": FORMAT,
        "version": VERSION,
        "status": RUNNING,
        "name": name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "args": {key: value for key, value in sorted(vars(args).items())},
        "source": {
            "artifact": str(artifact_path),
            "sha256": sha256_file(artifact_path),
            "name": loaded.name,
            "kind": loaded.kind,
            "elo_mode": source.elo_mode,
            "temperatures": stored,
            "dropout": float(source.net.config.dropout),
            "n_parameters": source.net.n_parameters(),
            "event_calibration_not_carried_over": source.event_calibration is not None,
            "coupling_not_carried_over": loaded.coupling is not None,
            "trained_on_manifest_sha256": trained_on,
        },
        "dataset": {
            "path": str(dataset),
            "tag": parts.manifest.get("tag"),
            "manifest_sha256": manifest_sha,
        },
        "allowed_games": {
            "file": str(old_games),
            "sha256": sha256_file(old_games),
            "games_listed": len(allowed),
        },
        "data": counts,
        "target_labels_outside_mask": masked,
        "recipe": {
            **asdict(recipe),
            "batch_rows": recipe.domain_batch * (1 + recipe.corpus_per_domain),
            "optimizer": "AdamW, constant learning rate, trainer's parameter groups",
            "loss": "model.nll_terms + model.total_loss with balanced_weight",
            "refit_temperatures": bool(args.refit_temperatures),
        },
        "grid": {
            "learning_rates": rates,
            "passes": passes,
            "pre_registered": pre_registered,
            "pre_registered_grid": {
                "learning_rates": list(GRID_LEARNING_RATES),
                "passes": list(GRID_PASSES),
            },
        },
        "folds": {
            "k": folds,
            "seed": seed,
            "unit": unit,
            "pre_registered": folds_pre_registered,
            "pre_registered_folds": {"k": DEFAULT_FOLDS, "unit": FOLD_GAME},
            "games": [
                int(sum(1 for fold in assignment.values() if fold == index))
                for index in range(folds)
            ],
            "rows": [int((row_fold == index).sum()) for index in range(folds)],
            "accounts": None if actors is None else int(np.unique(actors).size),
            "accounts_in_more_than_one_fold": straddling,
        },
        "pre_registered_reading": bool(pre_registered and folds_pre_registered),
        "write_artifact": write_artifact_wanted,
        "rule": RULE_TEXT,
    }
    write_report(out_dir, report)
    data = counts["domain"]
    log(
        f"FT_START {name} source {loaded.name} dataset {parts.manifest.get('tag')} "
        f"allowed games listed {len(allowed)} found {data['games_used']} "
        f"domain rows used {data['rows_used']} rows of other games ignored "
        f"{data['rows_dropped_game_not_allowed']} "
        f"({data['games_dropped_not_allowed']} games, "
        f"{data['games_dropped_dated_sealed']} of them dated on or after "
        f"{SEALED_SINCE}; used games dated so: {data['games_used_dated_sealed']}; "
        f"last used game {data['time_last']}) corpus rows {_rows(corpus)} "
        f"val rows {_rows(val)} folds {folds} by {unit} elo_mode {source.elo_mode} "
        f"threads {torch.get_num_threads()}"
    )

    def temperatures_of(net: M.OppNet) -> dict[str, float]:
        """The temperatures a fine-tuned network is read with."""
        if not args.refit_temperatures:
            return stored
        return T.calibrate(net, val, cpu)

    grid = [(rate, count) for rate in rates for count in passes]
    scratch = copy.deepcopy(source.net)
    base_domain = predict_rows(base, domain)
    base_val = predict_rows(base, val)

    # Out-of-fold predictions of every setting: each domain row once, by a
    # network fitted without its game.
    oof: dict[str, Prediction] = {
        setting_key(rate, count): {
            key: np.full_like(value, np.nan) for key, value in base_domain.items()
        }
        for rate, count in grid
    }
    cv_log: dict[str, list[dict[str, Any]]] = {setting_key(*pair): [] for pair in grid}
    for fold in range(folds):
        held = row_fold == fold
        check_disjoint(games[~held], games[held])
        train_rows, test_rows = F.take(domain, ~held), F.take(domain, held)
        for rate in rates:
            net = copy.deepcopy(source.net)
            fit_started = time.time()
            states, passes_log = finetune(
                net,
                train_rows,
                corpus,
                lr=rate,
                checkpoints=passes,
                recipe=recipe,
                seed=_fit_seed(seed, fold),
            )
            seconds = time.time() - fit_started
            for count in passes:
                key = setting_key(rate, count)
                scratch.load_state_dict(states[count])
                held_temperatures = temperatures_of(scratch)
                made = predict_rows(
                    predictor_of(scratch, source, held_temperatures), test_rows
                )
                for array, values in made.items():
                    oof[key][array][held] = values
                cv_log[key].append(
                    {
                        "fold": fold,
                        "train_rows": _rows(train_rows),
                        "test_rows": _rows(test_rows),
                        "steps": passes_log[count - 1]["steps"],
                        "loss_last_pass": passes_log[count - 1]["loss"],
                        "domain_fine_last_pass": passes_log[count - 1]["domain_fine"],
                        "corpus_fine_last_pass": passes_log[count - 1]["corpus_fine"],
                        "temperatures": held_temperatures,
                    }
                )
            log(
                f"  fold {fold + 1}/{folds} lr {rate:g}: {passes_log[-1]['steps']} "
                f"steps in {seconds:.1f} s, last-pass domain fine "
                f"{passes_log[-1]['domain_fine']:.4f} corpus fine "
                f"{passes_log[-1]['corpus_fine']:.4f}"
            )
    for key, pred in oof.items():
        if not all(np.isfinite(values).all() for values in pred.values()):
            raise FinetuneError(f"{key}: a domain row has no out-of-fold prediction")

    # The fit on ALL allowed games, per learning rate: validation and in-sample.
    full_states: dict[str, dict[str, torch.Tensor]] = {}
    full_log: dict[str, dict[str, Any]] = {}
    for rate in rates:
        net = copy.deepcopy(source.net)
        fit_started = time.time()
        states, passes_log = finetune(
            net,
            domain,
            corpus,
            lr=rate,
            checkpoints=passes,
            recipe=recipe,
            seed=_fit_seed(seed, folds),
        )
        seconds = time.time() - fit_started
        for count in passes:
            key = setting_key(rate, count)
            full_states[key] = states[count]
            full_log[key] = {
                "steps": passes_log[count - 1]["steps"],
                "seconds_whole_run": seconds,
                "passes": passes_log[:count],
            }
        log(
            f"  all games lr {rate:g}: {passes_log[-1]['steps']} steps in "
            f"{seconds:.1f} s"
        )

    settings: list[dict[str, Any]] = []
    full_temperatures: dict[str, dict[str, float]] = {}
    for rate, count in grid:
        key = setting_key(rate, count)
        found = compare(
            domain, base_domain, oof[key], resamples=int(args.resamples), seed=seed
        )
        nll = {
            "base": F.slot_nll(base_domain, domain),
            "new": F.slot_nll(oof[key], domain),
        }
        per_fold = []
        for fold in range(folds):
            keep = nll["base"]["fine_scored"] & (row_fold == fold)[:, None]
            per_fold.append(
                float((nll["new"]["fine"][keep] - nll["base"]["fine"][keep]).mean())
                if keep.any()
                else None
            )
        scratch.load_state_dict(full_states[key])
        full_temperatures[key] = temperatures_of(scratch)
        fitted = predictor_of(scratch, source, full_temperatures[key])
        # Validation gets every reading the old games get, the joint rows
        # included: a cost on Protect replies shows there as well.
        on_val = compare(
            val,
            base_val,
            predict_rows(fitted, val),
            resamples=int(args.resamples),
            seed=seed,
        )
        in_sample = compare(
            domain,
            base_domain,
            predict_rows(fitted, domain),
            resamples=int(args.resamples),
            seed=seed,
        )
        settings.append(
            {
                "key": key,
                "lr": rate,
                "passes": count,
                "oof": found,
                "oof_fine_diff_by_fold": per_fold,
                "cv_fits": cv_log[key],
                "val": {name: on_val[name] for name in VAL_ROWS},
                "in_sample": {name: in_sample[name] for name in IN_SAMPLE_ROWS},
                "full_fit": {**full_log[key], "temperatures": full_temperatures[key]},
            }
        )
        fine, top8 = found["fine"], found["joint_top8"][JOINT_ALL]
        guard, val_guard = (
            found["joint_top8"][JOINT_PROTECT],
            on_val["joint_top8"][JOINT_PROTECT],
        )
        log(
            f"SETTING {key}: oof fine {fine['base']:.4f} -> {fine['new']:.4f} "
            f"({fine['diff']:+.4f} [{fine['low']:+.4f}, {fine['high']:+.4f}]) "
            f"joint top-{JOINT_K} {top8['diff']:+.4f} "
            f"[{top8['low']:+.4f}, {top8['high']:+.4f}] "
            f"Protect replies {guard['diff']:+.4f} "
            f"val fine {on_val['fine']['diff']:+.4f} "
            f"val Protect replies {val_guard['diff']:+.4f} "
            f"in-sample {in_sample['fine']['diff']:+.4f}"
        )
    decision = decide(settings)
    best = next(row for row in settings if row["key"] == decision["best"])
    corrected = selection_corrected(
        domain,
        base_domain,
        oof,
        row_fold,
        folds,
        resamples=int(args.resamples),
        seed=seed,
    )
    decision["not_gated"] = not_gated(best, corrected)
    report["base"] = {
        "oof_rows": {
            "fine_nll": settings[0]["oof"]["fine"]["base"],
            "scored_slots": settings[0]["oof"]["fine"]["n"],
            "slots_by_class": {
                label: settings[0]["oof"]["fine_by_class"][label]["n"]
                for label in SLOT_CLASSES
            },
            "top1": settings[0]["oof"]["top1"]["base"],
            "top3": settings[0]["oof"]["top3"]["base"],
            "joint_top8": {
                label: {
                    "coverage": settings[0]["oof"]["joint_top8"][label]["base"],
                    "turns": settings[0]["oof"]["joint_top8"][label]["n"],
                }
                for label in JOINT_SLICES
            },
            "games": settings[0]["oof"]["games"],
        },
        "val_fine_nll": settings[0]["val"]["fine"]["base"],
        "val_scored_slots": settings[0]["val"]["fine"]["n"],
        "val_games": settings[0]["val"]["fine"]["games"],
        "val_slots_by_class": {
            label: settings[0]["val"]["fine_by_class"][label]["n"]
            for label in SLOT_CLASSES
        },
        "val_joint_top8": {
            label: {
                "coverage": settings[0]["val"]["joint_top8"][label]["base"],
                "turns": settings[0]["val"]["joint_top8"][label]["n"],
            }
            for label in JOINT_SLICES
        },
    }
    report["settings"] = settings
    report["decision"] = decision
    report["verdict"] = decision["verdict"]
    report["artifact"] = None
    log(
        f"DECISION best {decision['best']} -> {decision['verdict']} "
        + " ".join(
            f"{label}={'yes' if found['holds'] else 'NO'}"
            for label, found in decision["conditions"].items()
        )
        + f" | not gated, costs: {decision['not_gated']['costs'] or 'none'}"
        + f" | setting chosen without the fold read: "
        f"{corrected['fine']['diff']:+.4f} "
        f"[{corrected['fine']['low']:+.4f}, {corrected['fine']['high']:+.4f}]"
    )

    if decision["verdict"] == TAKEN and write_artifact_wanted:
        scratch.load_state_dict(full_states[best["key"]])
        final = predictor_of(scratch, source, full_temperatures[best["key"]], name)
        record = {
            "created": report["created"],
            "name": name,
            "in_sample": "This network was fitted on every game of seen_games: "
            "a reading of it on them is in-sample. Its honest numbers on them "
            "are the out-of-fold ones below.",
            KEY_SEEN: list(parts.used_ids),
            "sealed_since": SEALED_SINCE,
            "source_artifact": str(artifact_path),
            "source_artifact_sha256": report["source"]["sha256"],
            "source_name": loaded.name,
            "dataset_manifest_sha256": manifest_sha,
            "allowed_games_file": str(old_games),
            "allowed_games_sha256": report["allowed_games"]["sha256"],
            "setting": {"lr": best["lr"], "passes": best["passes"]},
            "domain_rows": data["rows_used"],
            "domain_games": data["games_used"],
            "domain_used_ids_sha256": data["used_ids_sha256"],
            "domain_time_first": data["time_first"],
            "domain_time_last": data["time_last"],
            "corpus_rows_sampled": _rows(corpus),
            "steps": best["full_fit"]["steps"],
            "recipe": report["recipe"],
            "seed": seed,
            "temperatures": full_temperatures[best["key"]],
            "grid_pre_registered": pre_registered,
            "folds": {"k": folds, "unit": unit, "pre_registered": folds_pre_registered},
            "pre_registered_reading": report["pre_registered_reading"],
            "oof": {
                "fine": best["oof"]["fine"],
                "fine_by_class": best["oof"]["fine_by_class"],
                "joint_top8": best["oof"]["joint_top8"],
            },
            "val": best["val"],
            "in_sample_fine": best["in_sample"]["fine"],
            "not_gated_costs": decision["not_gated"]["costs"],
            "selection_corrected_fine": corrected["fine"],
            "event_calibration_not_carried_over": source.event_calibration is not None,
            "coupling_not_carried_over": loaded.coupling is not None,
        }
        report["reload"] = write_artifact(
            out_dir,
            name=name,
            featurizer=loaded.featurizer,
            predictor=final,
            extra=finetuned_extra(extra, name=name, record=record),
            check_rows=F.concat_batches([domain, F.take(val, slice(0, RELOAD_ROWS))]),
            source=base,
        )
        report["artifact"] = str(out_dir / ARTIFACT_NAME)
        report["artifact_record"] = {
            "name": name,
            "seen_games": len(parts.used_ids),
            "manifest_sha256_as_read_by_the_scorecard": report["reload"]["trained_on"],
            "in_sample_on": f"the {len(parts.used_ids)} games of "
            f"extra['{KEY_FINETUNE}']['{KEY_SEEN}']",
        }
    report["status"] = DONE
    report["timing"] = {
        "seconds_total": time.time() - started,
        "threads": torch.get_num_threads(),
        "device": str(cpu),
    }
    report["torch"] = str(torch.__version__)
    write_report(out_dir, report)
    return report


def write_artifact(
    out_dir: Path,
    *,
    name: str,
    featurizer: F.Featurizer,
    predictor: M.OppNetPredictor,
    extra: Mapping[str, Any],
    check_rows: Mapping[str, np.ndarray],
    source: M.OppNetPredictor | None = None,
) -> dict[str, Any]:
    """Write ``predictor`` as ``artifact.pt`` and return the reload check.

    The file is written through ``oppmodel.artifact.save_artifact`` as
    ``artifact.pt.unverified``, reloaded through ``load_predictor`` and
    renamed only when the reloaded predictor's predictions on ``check_rows``
    are the in-memory predictor's (largest difference at most
    ``RELOAD_TOLERANCE``; ``identical`` says whether it is exactly zero).
    ``source`` adds how far the predictions moved from the unfine-tuned
    artifact's. The check also reads the stored record back: ``trained_on``
    (what the scorecard's ``_trained_on`` returns for the file) and the number
    of ``seen_games``, which must be the games of ``extra``. Raises
    ``FinetuneError`` when a check fails; the rejected file keeps its
    ``.unverified`` name.
    """
    path = out_dir / ARTIFACT_NAME
    unverified = path.with_name(path.name + UNVERIFIED_SUFFIX)
    A.save_artifact(
        unverified,
        kind=M.KIND,
        name=name,
        featurizer=featurizer,
        predictor_payload=predictor.to_payload(),
        extra=T._plain(dict(extra)),
    )
    again = A.load_predictor(unverified)
    if not isinstance(again.predictor, M.OppNetPredictor):
        raise FinetuneError(
            f"the artifact reloaded as {type(again.predictor).__name__}"
        )
    want = predict_rows(predictor, check_rows)
    got = predict_rows(again.predictor, check_rows)
    difference = {
        key: float(np.abs(want[key] - got[key]).max()) if want[key].size else 0.0
        for key in want
    }
    found: dict[str, Any] = {
        "rows": _rows(check_rows),
        "max_abs_difference": difference,
        "identical": all(np.array_equal(want[key], got[key]) for key in want),
        "tolerance": RELOAD_TOLERANCE,
        "name": again.name,
        "sha256": sha256_file(unverified),
    }
    if source is not None:
        before = predict_rows(source, check_rows)
        found["max_abs_change_from_source"] = {
            key: float(np.abs(want[key] - before[key]).max()) if want[key].size else 0.0
            for key in want
        }
    if again.name != name or max(difference.values()) > RELOAD_TOLERANCE:
        raise FinetuneError(
            f"the reloaded artifact differs from the in-memory model: {found}"
        )
    # The record a reader will see, read back from the file itself.
    stored = again.meta.get("extra") or {}
    found["trained_on"] = SC._trained_on(stored)
    found["seen_games"] = len(seen_games(stored))
    if seen_games(stored) != seen_games(extra):
        raise FinetuneError(
            "the reloaded artifact does not list the games it was fitted on"
        )
    unverified.replace(path)
    return found


# --- report ---------------------------------------------------------------------


def _signed(value: Any, digits: int = 4) -> str:
    return "-" if value is None else f"{float(value):+.{digits}f}"


def _plain_number(value: Any, digits: int = 4) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def _bracket(row: Mapping[str, Any] | None, digits: int = 4, scale: float = 1.0) -> str:
    """``diff [low, high]`` of a ``compare`` row (``scale`` 100 = points)."""
    if not row or row.get("diff") is None:
        return "-"
    text = _signed(float(row["diff"]) * scale, digits)
    if row.get("low") is None or row.get("high") is None:
        return text
    return (
        f"{text} [{_signed(float(row['low']) * scale, digits)}, "
        f"{_signed(float(row['high']) * scale, digits)}]"
    )


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def render_report(report: Mapping[str, Any]) -> str:
    """The Markdown view of ``finetune_report.json``."""
    data = T._plain(report)
    lines = [f"# OppNet fine-tune on own ladder games: {data.get('name', '-')}", ""]
    status = data.get("status")
    lines.append(
        f"Status: {status}"
        + (f" ({data.get('reason')})" if data.get("reason") else "")
        + (f"; verdict: **{data.get('verdict')}**" if data.get("verdict") else "")
        + f". Rendered from {REPORT_JSON}."
    )
    source, counts = data.get("source") or {}, data.get("data") or {}
    if not source or not counts:
        return "\n".join(lines) + "\n"
    domain, grid = counts.get("domain") or {}, data.get("grid") or {}
    recipe, folds = data.get("recipe") or {}, data.get("folds") or {}
    allowed = data.get("allowed_games") or {}
    lines += [
        "",
        f"- source artifact: {source.get('name')} (sha256 {source.get('sha256')}), "
        f"Elo mode {source.get('elo_mode')}, temperatures {source.get('temperatures')}"
        f", dropout {source.get('dropout')}",
        f"- dataset: {(data.get('dataset') or {}).get('tag')} (manifest sha256 "
        f"{(data.get('dataset') or {}).get('manifest_sha256')})",
        f"- allowed games: {allowed.get('games_listed')} listed in "
        f"{allowed.get('file')} (sha256 {allowed.get('sha256')})",
        f"- domain rows ({domain.get('split')} split): {domain.get('rows_used')} rows "
        f"of {domain.get('games_used')} allowed games used; "
        f"{domain.get('rows_dropped_game_not_allowed')} rows of "
        f"{domain.get('games_dropped_not_allowed')} games NOT in the list ignored "
        f"(split holds {domain.get('rows_in_split')} rows, "
        f"{domain.get('games_in_split')} games); used games outside the list: "
        f"{domain.get('used_games_outside_allowed')}; allowed games without rows "
        f"here: {domain.get('allowed_games_without_rows_here')}; games used were "
        f"played {domain.get('time_first')} .. {domain.get('time_last')}",
        f"- the seal ({domain.get('sealed_since')}, by the dataset's own dates): "
        f"{domain.get('games_in_split_dated_sealed')} games of the split are "
        f"dated on or after it, {domain.get('games_dropped_dated_sealed')} of "
        f"them had rows and were dropped; used games dated on or after it: "
        f"{domain.get('games_used_dated_sealed')}; used games without a date: "
        f"{domain.get('games_used_undated')}",
        f"- corpus rows: {(counts.get('corpus') or {}).get('rows_sampled')} sampled "
        f"of {(counts.get('corpus') or {}).get('rows_in_split')} train rows; "
        f"validation rows: {(counts.get('val') or {}).get('rows_used')} "
        f"({(counts.get('val') or {}).get('rows_left_out_holdout_battles')} left "
        "out: battles with a ladder-holdout opponent)",
        f"- recipe: batch {recipe.get('batch_rows')} = {recipe.get('domain_batch')} "
        f"domain + {recipe.get('corpus_per_domain')}x corpus rows, equal loss "
        f"weight for the two parts; slot swap {recipe.get('slot_swap')}, Elo drop "
        f"{recipe.get('elo_drop')}, gradient clip {recipe.get('clip')}, Mega weight "
        f"{recipe.get('mega_weight')}, weight decay {recipe.get('weight_decay')}; "
        f"temperatures refitted: {recipe.get('refit_temperatures')}",
        f"- grid: learning rates {grid.get('learning_rates')} x passes "
        f"{grid.get('passes')}"
        + (
            " (the pre-registered grid)"
            if grid.get("pre_registered")
            else " (NOT the pre-registered grid: not the pre-registered reading)"
        ),
        f"- folds: {folds.get('k')} by {folds.get('unit', FOLD_GAME)} (seed "
        f"{folds.get('seed')})"
        + (
            " (the pre-registered folds)"
            if folds.get("pre_registered", True)
            else " (NOT the pre-registered folds, which are "
            f"{DEFAULT_FOLDS} by {FOLD_GAME}: a sensitivity reading)"
        )
        + f", games per fold {folds.get('games')}, rows per fold "
        f"{folds.get('rows')}; opponent accounts: {folds.get('accounts')}, with "
        f"games in more than one fold: {folds.get('accounts_in_more_than_one_fold')}",
    ]
    if data.get("write_artifact") is False:
        lines.append("- a robustness run (--no-artifact): no artifact is written")
    if source.get("event_calibration_not_carried_over") or source.get(
        "coupling_not_carried_over"
    ):
        lines.append(
            "- the source's event calibration / pair coupling was fitted after "
            "another network: it is read without it here and not carried over"
        )
    settings = data.get("settings") or []
    whole = data.get("base") or {}
    base = whole.get("oof_rows") or {}
    if not settings:
        return "\n".join(lines) + "\n"
    joint_base = base.get("joint_top8") or {}
    joint_val = whole.get("val_joint_top8") or {}
    lines += [
        "",
        "## Unfine-tuned artifact on the same rows",
        "",
        f"Pooled out-of-fold rows: {base.get('scored_slots')} scored slots of "
        f"{base.get('games')} games ({base.get('slots_by_class')}); fine NLL "
        f"{_plain_number(base.get('fine_nll'))}, top-1 "
        f"{_plain_number(base.get('top1'))}, top-3 {_plain_number(base.get('top3'))}; "
        f"joint top-{JOINT_K} "
        + ", ".join(
            f"{label}: {_plain_number((joint_base.get(label) or {}).get('coverage'))} "
            f"of {(joint_base.get(label) or {}).get('turns')} turns"
            for label in JOINT_SLICES
        )
        + f". Validation: fine NLL {_plain_number(whole.get('val_fine_nll'))}"
        f" over {whole.get('val_scored_slots')} slots of {whole.get('val_games')} "
        f"games ({whole.get('val_slots_by_class')}); joint top-{JOINT_K} "
        + ", ".join(
            f"{label}: {_plain_number((joint_val.get(label) or {}).get('coverage'))} "
            f"of {(joint_val.get(label) or {}).get('turns')} turns"
            for label in JOINT_SLICES
        )
        + ".",
        "",
        "## Change against the unfine-tuned artifact (fine-tuned minus unfine-tuned)",
        "",
        "Out-of-fold: every domain row predicted by a network fitted without its "
        "game. Brackets: 95% interval, whole games resampled "
        f"({settings[0]['oof'].get('resamples')} resamples). NLL in nats per scored "
        "slot (lower is better); top-k and coverage in points.",
        "",
    ]
    rows = []
    for row in settings:
        oof = row["oof"]
        rows.append(
            [
                row["key"],
                _bracket(oof["fine"]),
                _bracket(oof["top1"], 2, 100.0),
                _bracket(oof["top3"], 2, 100.0),
                _bracket(oof["joint_top8"][JOINT_ALL], 2, 100.0),
                _bracket(oof["joint_top8"][JOINT_SWITCH], 2, 100.0),
                _bracket(oof["joint_top8"][JOINT_PROTECT], 2, 100.0),
                _bracket(row["val"]["fine"]),
                _signed(row["in_sample"]["fine"]["diff"]),
            ]
        )
    lines += _table(
        [
            "setting",
            "oof fine NLL",
            "oof top-1",
            "oof top-3",
            f"oof joint top-{JOINT_K}",
            "joint: holds a switch",
            "joint: holds a Protect",
            "val fine NLL (all-games fit)",
            "in-sample fine NLL",
        ],
        rows,
    )
    lines += ["", "Out-of-fold fine NLL by kind of slot:", ""]
    lines += _table(
        ["setting", *SLOT_CLASSES],
        [
            [
                row["key"],
                *(_bracket(row["oof"]["fine_by_class"][c]) for c in SLOT_CLASSES),
            ]
            for row in settings
        ],
    )
    lines += [
        "",
        "Validation (held-out players of the human corpus), the network fitted "
        "on ALL allowed games against the unfine-tuned artifact, and the same "
        "network on the old games it was fitted on (in-sample, not a gain):",
        "",
    ]
    lines += _table(
        [
            "setting",
            "val fine NLL",
            "val: Protect clicked",
            "val: switch made",
            f"val joint top-{JOINT_K}",
            "val joint: holds a switch",
            "val joint: holds a Protect",
            "in-sample fine NLL",
            f"in-sample joint top-{JOINT_K}",
        ],
        [
            [
                row["key"],
                _bracket(row["val"]["fine"]),
                _bracket((row["val"].get("fine_by_class") or {}).get(CLASS_PROTECT)),
                _bracket((row["val"].get("fine_by_class") or {}).get(CLASS_SWITCH)),
                _bracket((row["val"].get("joint_top8") or {}).get(JOINT_ALL), 2, 100.0),
                _bracket(
                    (row["val"].get("joint_top8") or {}).get(JOINT_SWITCH), 2, 100.0
                ),
                _bracket(
                    (row["val"].get("joint_top8") or {}).get(JOINT_PROTECT), 2, 100.0
                ),
                _signed(row["in_sample"]["fine"]["diff"]),
                _bracket(
                    (row["in_sample"].get("joint_top8") or {}).get(JOINT_ALL), 2, 100.0
                ),
            ]
            for row in settings
        ],
    )
    lines += ["", "Out-of-fold fine NLL change by fold, and steps of a fold's fit:", ""]
    lines += _table(
        ["setting", "by fold", "steps (fold fit)", "steps (all-games fit)"],
        [
            [
                row["key"],
                ", ".join(_signed(value) for value in row["oof_fine_diff_by_fold"]),
                ", ".join(str(fit["steps"]) for fit in row["cv_fits"]),
                row["full_fit"]["steps"],
            ]
            for row in settings
        ],
    )
    decision = data.get("decision") or {}
    lines += ["", "## Verdict", "", f"Rule: {decision.get('rule')}", ""]
    lines.append(
        f"Best pooled out-of-fold fine NLL: **{decision.get('best')}** "
        f"({_plain_number(decision.get('best_oof_fine_nll'))})."
    )
    for label, found in (decision.get("conditions") or {}).items():
        numbers = ", ".join(
            f"{key} {_signed(value)}"
            for key, value in found.items()
            if key not in ("holds", "n") and value is not None
        )
        lines.append(
            f"- {label}: {'holds' if found.get('holds') else 'FAILS'} ({numbers})"
        )
    lines += ["", f"**{decision.get('verdict')}**"]
    if data.get("pre_registered_reading") is False:
        lines.append(
            "(by the rule, on a grid or folds that are NOT the pre-registered "
            "ones: a sensitivity reading, not the pre-registered verdict)"
        )
    beside = decision.get("not_gated") or {}
    if beside:
        lines += [
            "",
            f"## Not gated by the rule ({beside.get('setting')})",
            "",
            str(beside.get("note")),
            "",
        ]
        found = beside.get("readings") or {}
        rows = []
        for name, larger in NOT_GATED_READINGS:
            cells: list[Any] = [name]
            for place in PLACES:
                row = (found.get(place) or {}).get(name)
                scale, digits = (1.0, 4) if larger else (100.0, 2)
                cells.append(_bracket(row, digits, scale))
                cells.append("-" if not row else row.get("n"))
            rows.append(cells)
        header = ["reading (NLL in nats, coverage in points)"]
        for place in PLACES:
            header += [PLACES[place], "n"]
        lines += _table(header, rows)
        costs = beside.get("costs") or []
        lines += [
            "",
            "Costs (interval wholly on the worse side): "
            + ("; ".join(costs) if costs else "none")
            + ".",
        ]
        corrected = beside.get("selection_corrected") or {}
        if corrected:
            lines += [
                "",
                "Setting chosen without the fold it is read on ("
                + ", ".join(str(key) for key in corrected.get("chosen_by_fold") or [])
                + f"): fine NLL {_bracket(corrected.get('fine'))}, joint "
                f"top-{JOINT_K} "
                + _bracket((corrected.get("joint_top8") or {}).get(JOINT_ALL), 2, 100.0)
                + " points. The best setting's own number is read on the rows "
                "it was chosen on.",
            ]
    reload = data.get("reload")
    if data.get("artifact") and reload:
        record = data.get("artifact_record") or {}
        lines += [
            "",
            f"Artifact: {data.get('artifact')} (sha256 {reload.get('sha256')}, name "
            f"{reload.get('name')}). Reload check on {reload.get('rows')} rows: "
            f"largest difference {reload.get('max_abs_difference')}, identical "
            f"{reload.get('identical')}; largest change from the source "
            f"{reload.get('max_abs_change_from_source')}.",
            "",
            f"**The artifact has seen {record.get('seen_games')} own ladder games "
            f"({record.get('in_sample_on')}): any reading of it on them is "
            "in-sample.** Its honest numbers there are the out-of-fold ones "
            "above. Its record names another training set than the build "
            f"(`{reload.get('trained_on')}`), so the scorecard warns; "
            "`finetune_oppmodel.check_unseen` refuses such a reading.",
        ]
    elif data.get("verdict") == TAKEN and data.get("write_artifact") is False:
        lines += ["", "No artifact: a robustness run (--no-artifact)."]
    timing = data.get("timing") or {}
    if timing:
        lines += [
            "",
            f"Timing: {_plain_number(timing.get('seconds_total'), 1)} s, "
            f"{timing.get('threads')} threads, {timing.get('device')}.",
        ]
    return "\n".join(lines) + "\n"


def write_report(out_dir: Path, report: Mapping[str, Any]) -> None:
    data = T._plain(report)
    (out_dir / REPORT_JSON).write_text(json.dumps(data, indent=1), encoding="utf-8")
    (out_dir / REPORT_MD).write_text(render_report(data), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        if exc.code in (0, None):
            raise
        say(f"{FAILED} bad arguments")
        return 2
    try:
        report = run(args)
    except Exception as exc:
        reason = " ".join(f"{type(exc).__name__}: {exc}".split())
        say(f"{FAILED} {reason}")
        return 1
    decision = report["decision"]
    costs = (decision.get("not_gated") or {}).get("costs") or []
    say(
        f"not gated, costs of the best setting: {'; '.join(costs) if costs else 'none'}"
    )
    say(
        f"{report['verdict']} best {decision['best']} "
        f"oof_fine {decision['best_oof_fine_nll']:.4f} artifact {report['artifact']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
