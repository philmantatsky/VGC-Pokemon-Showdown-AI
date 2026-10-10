"""Make an ensemble artifact from trained opponent-predictor artifacts.

    nice -n 10 .venv/bin/python training/make_oppmodel_ensemble.py \\
        --member base=results_oppmodel/<run>/artifact.pt \\
        --member wide=results_oppmodel/<other run>/artifact.pt \\
        --dataset results_oppmodel/<the build they were trained on> \\
        --out results_oppmodel/<a new directory>

Nothing is trained. At least two members (one member alone is that member:
its own artifact is the file to use). The ensemble
(``vgc_bench/src/oppmodel/ensemble.py``) is
the weighted mean of the members' fine distributions (equal weights unless
``--weights`` gives one number per member, in member order); each member's
own predictor payload is stored unchanged, with its temperatures and any
calibration. The members must share one featurizer (identical payloads, or
payloads that differ only by the version-2 extra arrays); a member that
carries a pair coupling is refused (fit a coupling on the ensemble afterwards
with ``training/fit_oppmodel_coupling.py``).

What is written, in this order: ``artifact.pt.unverified``, which is reloaded
through ``artifact.load_predictor`` and checked, and only then renamed to
``artifact.pt``. The checks (``verify``), on ``--check-rows`` validation rows
(a seeded sample) and with a bar of ``--tolerance``:

* the reloaded ensemble's fine distribution equals the weighted mean of the
  fine distributions of the members loaded on their own (each from its own
  file, with its own featurizer), and so do the action marginal and the Mega
  probability;
* the reloaded ensemble's arrays are a proper prediction
  (``features.normalize_prediction`` leaves them unchanged, every acting
  slot's fine distribution sums to 1);
* the stored member payloads are the members' own, array for array;
* the reloaded ensemble names its state (``state_sha256``, what a pair
  coupling binds to) and it is the state of the ensemble as built;
* the own ladder games the stored artifact says it has seen are exactly the
  games the members' own records list (none without a fine-tuned member).

Reported, not gated: the fine NLL per scored slot-turn, top-1 and top-3 of
each member and of the ensemble on the validation split and on the OLD ladder
rows, with the paired difference to the ensemble and a 95% interval from
resampling whole games (``oppmodel_scorecard.score_set``). The split ``test``
is never read. The bot's own games from ``features.OWN_SEALED_FROM`` on are
the sealed confirmation set: they are left out by their own times
(``features.load_dataset(..., own_before=...)``), counted in the report, and
a dataset whose rows carry no times has its ladder holdout left unread.

A MEMBER FINE-TUNED ON OWN LADDER GAMES (``training/finetune_oppmodel.py``)
is not fitted on ``--dataset`` by its record, so it needs
``--allow-other-dataset``. The ensemble then has seen those games: the
artifact's ``extra`` says so (``finetune['seen_games']`` and a marked
manifest, see ``ensemble.py``), the old ladder rows are NOT read or scored (a
reading of them would be in-sample), and the report names the members and the
number of games. The validation rows stay unseen by a fine-tune and are read.

Outputs: ``artifact.pt``, ``ensemble_report.json``, ``ensemble_report.md``.
The last line is the result a caller reads: ``ENS_DONE`` or
``ENS_FAILED <reason>``. One thread, CPU.
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
from vgc_bench.src.oppmodel import ensemble as E  # noqa: E402
from vgc_bench.src.oppmodel import features as F  # noqa: E402

FORMAT = "oppmodel-ensemble-report"
VERSION = 1
ARTIFACT_NAME = "artifact.pt"
UNVERIFIED_SUFFIX = ".unverified"
REPORT_JSON = "ensemble_report.json"
REPORT_MD = "ensemble_report.md"
SPLIT_VAL = "val"
SPLIT_LADDER = "ladder_holdout"
SET_VAL = "validation"
SET_LADDER = "old ladder rows"
ENSEMBLE = "ensemble"
DONE, FAILED = "ENS_DONE", "ENS_FAILED"
DEFAULT_CHECK_ROWS = 2000
DEFAULT_TOLERANCE = 1e-9
DEFAULT_RESAMPLES = 2000


class EnsembleError(RuntimeError):
    """The ensemble cannot be made, or the written file failed a check."""


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


def member_specs(values: Sequence[str]) -> list[tuple[str, Path]]:
    """``label=path`` arguments as (label, path), in order."""
    out: list[tuple[str, Path]] = []
    for value in values:
        label, mark, tail = str(value).partition("=")
        if not mark or not label.strip() or not tail.strip():
            raise EnsembleError(f"--member {value!r}: give label=path")
        out.append((label.strip(), Path(tail.strip())))
    if not out:
        raise EnsembleError("give at least one --member label=path")
    return out


def read_manifest(dataset: Path) -> tuple[dict[str, Any], str]:
    path = dataset / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EnsembleError(f"dataset {dataset}: {exc!r}") from exc
    return manifest, sha256_file(path)


def sealed_own_games(dataset: Path) -> dict[str, Any]:
    """What the dataset holds of the bot's own games from the sealed day on
    (``features.own_time_range``: the meta arrays only)."""
    try:
        cut = F.local_time(F.OWN_SEALED_FROM)
        found = dict(F.own_time_range(dataset, cut))
    except (OSError, ValueError, KeyError) as exc:
        raise EnsembleError(
            f"dataset {dataset}: the times of its own games cannot be read ({exc!r})"
        ) from exc
    return {
        "sealed_from": F.OWN_SEALED_FROM,
        "sealed_cut": int(cut),
        "time_known": bool(found.get("has_time", True)),
        "late_rows": int(found.get("late_rows") or 0),
        "splits": found.get("splits") or {},
    }


def finetune_record(extra: Any) -> dict[str, Any] | None:
    """What an ensemble artifact's ``extra`` says of fine-tuned members: their
    labels, the number of own ladder games the ensemble has seen (the union)
    and each member's own number. None when no member was fine-tuned."""
    found = extra.get(E.KEY_FINETUNE) if isinstance(extra, Mapping) else None
    if not isinstance(found, Mapping):
        return None
    members = found.get("members")
    members = members if isinstance(members, Mapping) else {}
    return {
        "members": [str(label) for label in members],
        "seen_own_games": len(found.get(E.KEY_SEEN) or ()),
        "by_member": {
            str(label): len((record or {}).get(E.KEY_SEEN) or ())
            for label, record in members.items()
        },
    }


def load_sets(
    dataset: Path, own: Mapping[str, Any], read_ladder: bool = True
) -> dict[str, F.Batch]:
    """The validation split and the OLD rows of the ladder holdout.

    One pass over the shards; only the two splits are kept (``test`` is never
    selected), and the rows of own games at or after the sealed day, or of
    unknown time, are dropped inside ``features.load_dataset``. The ladder
    holdout is not read at all when the dataset's rows carry no times, or
    with ``read_ladder=False`` (a member was fine-tuned on own ladder games).
    """
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    names = list(manifest.get("splits") or [])
    if SPLIT_VAL not in names:
        raise EnsembleError(f"dataset {dataset} has no {SPLIT_VAL} split")
    wanted = [SPLIT_VAL]
    if read_ladder and SPLIT_LADDER in names and own["time_known"]:
        wanted.append(SPLIT_LADDER)
    # Rows without times cannot be filtered by time (``load_dataset`` needs
    # ``m_time`` for it): then only the validation split is asked for, which
    # holds no own game (``features.OWN_SPLITS``).
    cut = int(own["sealed_cut"]) if own["time_known"] else None
    if cut is None and any(name in F.OWN_SPLITS for name in wanted):
        raise EnsembleError("own games would be read without their times")
    try:
        data, _ = F.load_dataset(dataset, splits=wanted, own_before=cut)
    except (OSError, ValueError, KeyError) as exc:
        raise EnsembleError(f"dataset {dataset}: {exc!r}") from exc
    split = np.asarray(data["m_split"])
    found = sorted(int(code) for code in np.unique(split))
    allowed = sorted(names.index(name) for name in wanted)
    if any(code not in allowed for code in found):
        raise EnsembleError(f"unexpected split codes loaded: {found}")
    out = {SET_VAL: F.take(data, split == names.index(SPLIT_VAL))}
    if SPLIT_LADDER in wanted:
        ladder = F.take(data, split == names.index(SPLIT_LADDER))
        when = np.asarray(ladder["m_time"]).astype(np.int64)
        if ladder["turn"].shape[0] and int(when.max()) >= int(own["sealed_cut"]):
            raise EnsembleError("a sealed own game came through the filter")
        out[SET_LADDER] = ladder
    return out


def predict(predictor: Any, batch: F.Batch, name: str) -> tuple[S.Prediction, float]:
    try:
        return S.guarded_predict(predictor, batch, name=name)
    except S.ScorecardError as exc:
        raise EnsembleError(str(exc)) from exc


def mean_identity(
    made: Mapping[str, np.ndarray],
    alone: Sequence[Mapping[str, np.ndarray]],
    weights: Sequence[float],
    batch: F.Batch,
) -> dict[str, Any]:
    """How far the ensemble's prediction is from the weighted mean of the
    members' own, entry by entry, and whether it is a proper prediction.

    The mean is taken here with ``features.fine_probs`` /
    ``normalize_prediction`` of each member alone, not with the ensemble's
    code.
    """
    fine = np.zeros_like(F.fine_probs(alone[0], batch))
    action = np.zeros_like(F.normalize_prediction(alone[0], batch)["action"])
    mega = np.zeros(action.shape[:2], dtype=np.float64)
    for weight, pred in zip(weights, alone):
        norm = F.normalize_prediction(pred, batch)
        fine += weight * F.fine_probs(pred, batch)
        action += weight * norm["action"]
        mega += weight * norm["mega"]
    norm = F.normalize_prediction(made, batch)
    ours = F.fine_probs(made, batch)
    acting = np.asarray(batch["action_mask"]).astype(bool).any(-1)
    sums = ours.sum(-1)
    change = max(
        float(np.abs(norm[key] - np.asarray(made[key], dtype=np.float64)).max())
        if norm[key].size
        else 0.0
        for key in ("action", "target", "mega")
    )

    def gap(first: np.ndarray, second: np.ndarray) -> float:
        return float(np.abs(first - second).max()) if first.size else 0.0

    return {
        "rows": int(action.shape[0]),
        "max_abs_difference": {
            "fine": gap(ours, fine),
            "action": gap(norm["action"], action),
            "mega": gap(norm["mega"], mega),
        },
        "max_change_by_normalising": change,
        "max_gap_of_a_slot_sum_to_one": gap(sums[acting], np.ones(int(acting.sum()))),
        "mass_on_slots_without_an_action": gap(
            sums[~acting], np.zeros(int((~acting).sum()))
        ),
    }


def _tensor(value: Any) -> Any:
    return value


def same(first: Any, second: Any) -> bool:
    """Deep equality of two decoded artifact values, dtype and shape included."""
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


def flipped(found: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """A paired difference with its sign turned (member - ensemble becomes
    ensemble - member)."""
    if not found or found.get("diff") is None:
        return None
    low, high = found.get("low"), found.get("high")
    return {
        "diff": -float(found["diff"]),
        "low": None if high is None else -float(high),
        "high": None if low is None else -float(low),
    }


def score(
    batch: F.Batch,
    preds: Mapping[str, S.Prediction],
    seconds: Mapping[str, float],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Each predictor's reading on one set, and the ensemble against each."""
    try:
        card = S.score_set(batch, preds, ENSEMBLE, resamples=resamples, seed=seed)
    except S.ScorecardError as exc:
        raise EnsembleError(str(exc)) from exc
    n = int(np.asarray(batch["turn"]).shape[0])
    rows: dict[str, Any] = {}
    for name, found in card["predictors"].items():
        difference = found.get("difference") or {}
        rows[name] = {
            "fine_nll": found["fine_nll"],
            "fine_top1": found.get("fine_top1"),
            "fine_top3": found.get("fine_top3"),
            "action_top1": found.get("action_top1"),
            "mega_nll": found.get("mega_nll"),
            "parts": found.get("parts"),
            "ensemble_minus_this": None
            if name == ENSEMBLE
            else {
                key: flipped(difference.get(key))
                for key in ("fine", "visible", "censored", "action", "target")
            },
            "ensemble_minus_this_by_account": None
            if name == ENSEMBLE
            else flipped(found.get("difference_by_account")),
            "predict_seconds": float(seconds[name]),
            "microseconds_per_example": 1e6 * float(seconds[name]) / n if n else None,
        }
    return {
        "examples": card["examples"],
        "games": card["games"],
        "slots": card["slots"],
        "clusters": card["clusters"],
        "resamples": card["resamples"],
        "predictors": rows,
    }


def run(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    started = time.time()
    dataset = Path(args.dataset)
    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise EnsembleError(f"{out_dir} exists and is not empty; give a new directory")
    specs = member_specs(args.member)
    if len(specs) < 2:
        # One member alone is that member, arrays untouched: there is no mean
        # to check, and its own artifact is the file to use.
        raise EnsembleError(
            "give at least two --member: an ensemble of one member is that "
            "member (use its artifact)"
        )
    name = str(args.name or out_dir.name)
    manifest, manifest_sha = read_manifest(dataset)
    own = sealed_own_games(dataset)
    try:
        built = E.from_artifacts(specs, weights=args.weights, name=name)
        alone = {label: A.load_predictor(path, strict=True) for label, path in specs}
        documents = {label: A.read_artifact(path, strict=True) for label, path in specs}
        home = F.Featurizer.load(dataset)
    except (OSError, ValueError) as exc:
        raise EnsembleError(str(exc)) from exc
    weights = list(built.predictor.weights)
    labels = list(built.predictor.labels)
    say(
        f"ensemble {name}: {len(labels)} members "
        + ", ".join(f"{label} x {weight:.4f}" for label, weight in zip(labels, weights))
    )
    # The rows of the dataset were encoded by the dataset's featurizer: the
    # readings below are the ensemble's only if that is the members' own.
    core = E.payload_hash(E.core_payload(home.to_payload()))
    ours = E.payload_hash(E.core_payload(built.featurizer.to_payload()))
    fed = set(home.extras)
    unfed = sorted(set(built.predictor.extra_keys) - fed)
    if core != ours or unfed:
        raise EnsembleError(
            f"dataset {dataset} was not encoded by the members' featurizer "
            f"(same without the version-2 arrays: {core == ours}; arrays the "
            f"members read that it lacks: {unfed})"
        )
    trained = {
        label: S._trained_on(alone[label].meta.get("extra") or {}) for label in labels
    }
    other = sorted(label for label, found in trained.items() if found != manifest_sha)
    if other and not args.allow_other_dataset:
        raise EnsembleError(
            f"members {other} were not fitted on this dataset (manifest "
            f"{manifest_sha}): its validation rows may be their training rows "
            "(give --allow-other-dataset to go on)"
        )

    extra = dict(built.extra)
    extra["args"] = {key: plain(value) for key, value in vars(args).items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ARTIFACT_NAME
    unverified = path.with_name(path.name + UNVERIFIED_SUFFIX)
    payload = built.predictor.to_payload()
    try:
        A.save_artifact(
            unverified,
            kind=E.KIND,
            name=name,
            featurizer=built.featurizer,
            predictor_payload=payload,
            extra=plain(extra),
        )
        stored = A.read_artifact(unverified, strict=True)
        again = A.load_predictor(unverified, strict=True)
    except (OSError, ValueError, TypeError) as exc:
        raise EnsembleError(
            f"the artifact could not be written and read: {exc}"
        ) from exc
    say(f"SAVED {unverified}")

    # A member fine-tuned on own ladder games has seen them, so the ensemble
    # has: its record is in the artifact's extra, and the old ladder rows are
    # not read (a reading of them would be in-sample).
    tuned = finetune_record(stored.get("extra"))
    if tuned is not None:
        say(
            f"members {', '.join(tuned['members'])} were fine-tuned on "
            f"{tuned['seen_own_games']} own ladder games: the ensemble has seen "
            "them, and the old ladder rows are not read"
        )
    sets = load_sets(dataset, own, read_ladder=tuned is None)
    left_out = int(own["late_rows"])
    unread = ""
    if SET_LADDER in sets:
        ladder_note = (
            f"{SET_LADDER}: {len(sets[SET_LADDER]['turn'])} examples "
            f"({left_out} rows of own games from {F.OWN_SEALED_FROM} on, the "
            "sealed set, left out)"
        )
    else:
        unread = (
            "a fine-tuned member has seen its old games"
            if tuned is not None
            else "the dataset's rows carry no times"
        )
        ladder_note = f"ladder holdout not read ({unread})"
    say(f"{SET_VAL}: {len(sets[SET_VAL]['turn'])} examples; {ladder_note}")

    # The checks, on a seeded sample of validation rows.
    validation = sets[SET_VAL]
    total = int(validation["turn"].shape[0])
    rng = np.random.default_rng(int(args.seed))
    count = min(total, max(1, int(args.check_rows)))
    pick = np.sort(rng.choice(total, size=count, replace=False))
    sample = F.take(validation, pick)
    made, _ = predict(again.predictor, sample, ENSEMBLE)
    lone = [predict(alone[label].predictor, sample, label)[0] for label in labels]
    identity = mean_identity(made, lone, weights, sample)
    members_stored = stored["predictor"]["members"]
    bar = float(args.tolerance)

    def seen_by(extra: Any) -> list[str]:
        """The own ladder games an artifact's ``extra`` lists as fitted on."""
        found = extra.get(E.KEY_FINETUNE) if isinstance(extra, Mapping) else None
        games = found.get(E.KEY_SEEN) if isinstance(found, Mapping) else None
        return sorted(str(game) for game in (games or ()))

    seen_alone = sorted(
        {game for label in labels for game in seen_by(documents[label].get("extra"))}
    )
    checks: dict[str, Any] = {
        "kind_is_ensemble": stored.get("kind") == E.KIND and again.kind == E.KIND,
        "artifact_version": stored.get("version"),
        "no_coupling": again.coupling is None,
        "tolerance": bar,
        "rows_checked": identity["rows"],
        "mean_identity": identity,
        "equals_the_mean_of_the_members": max(identity["max_abs_difference"].values())
        <= bar,
        "is_a_proper_prediction": identity["max_change_by_normalising"] <= bar
        and identity["max_gap_of_a_slot_sum_to_one"] <= bar
        and identity["mass_on_slots_without_an_action"] <= bar,
        "member_payloads_unchanged": len(members_stored) == len(labels)
        and all(
            same(entry["payload"], documents[label]["predictor"])
            for entry, label in zip(members_stored, labels)
        ),
        "member_sha256_recorded": all(
            entry["sha256"] == sha256_file(spec[1])
            for entry, spec in zip(members_stored, specs)
        ),
        "weights_stored": [float(value) for value in stored["predictor"]["weights"]],
        "weights_as_built": bool(
            np.allclose(stored["predictor"]["weights"], weights, rtol=0, atol=1e-15)
        ),
        "featurizer_is_the_members": E.payload_hash(stored["featurizer"])
        == E.payload_hash(documents[built.extra["featurizer_of"]]["featurizer"]),
        "no_predictor_failure": not any(
            str(key).startswith(E.FAILURE) for key in again.predictor.counters
        ),
        # What a pair coupling binds to: named, and the same before and after
        # the file.
        "state_sha256": again.predictor.state_sha256,
        "state_named_and_kept": bool(again.predictor.state_sha256)
        and again.predictor.state_sha256 == built.predictor.state_sha256,
        # The own ladder games any member was fine-tuned on are the games the
        # stored artifact says it has seen (none when no member was).
        "seen_own_games": len(seen_alone),
        "seen_games_carried": seen_by(stored.get("extra")) == seen_alone,
    }
    gated = (
        "kind_is_ensemble",
        "no_coupling",
        "equals_the_mean_of_the_members",
        "is_a_proper_prediction",
        "member_payloads_unchanged",
        "member_sha256_recorded",
        "weights_as_built",
        "featurizer_is_the_members",
        "no_predictor_failure",
        "state_named_and_kept",
        "seen_games_carried",
    )
    checks["ok"] = all(bool(checks[key]) for key in gated)
    say(
        f"check on {identity['rows']} validation rows: max |ensemble - mean of the "
        f"members| fine {identity['max_abs_difference']['fine']:.3g}, action "
        f"{identity['max_abs_difference']['action']:.3g}, mega "
        f"{identity['max_abs_difference']['mega']:.3g} (bar {bar:g}); "
        f"{'ok' if checks['ok'] else 'FAILED'}"
    )

    report: dict[str, Any] = {
        "format": FORMAT,
        "version": VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "name": name,
        "kind": E.KIND,
        "combine": E.COMBINE,
        "arguments": {key: plain(value) for key, value in vars(args).items()},
        "dataset": {
            "path": str(dataset),
            "tag": manifest.get("tag"),
            "manifest_sha256": manifest_sha,
            "members_fitted_on_it": {
                label: trained[label] == manifest_sha for label in labels
            },
        },
        "members": {
            label: {
                "path": str(spec[1]),
                "sha256": sha256_file(spec[1]),
                "name": alone[label].name,
                "kind": alone[label].kind,
                "weight": weight,
                "n_parameters": (alone[label].meta.get("extra") or {}).get(
                    "n_parameters"
                ),
                "elo_mode": getattr(alone[label].predictor, "elo_mode", None),
                "describe": built.predictor._member_described(member),
                "featurizer_sha256": E.payload_hash(documents[label]["featurizer"]),
                "featurizer_extras": list(member.extras),
            }
            for label, spec, weight, member in zip(
                labels, specs, weights, built.predictor.members
            )
        },
        "featurizer": {
            "of_member": built.extra["featurizer_of"],
            "layout_version": built.featurizer.layout_version,
            "extras": list(built.featurizer.extras),
            "identical_payloads": len(
                {E.payload_hash(documents[label]["featurizer"]) for label in labels}
            )
            == 1,
            "arrays_the_members_read": list(built.predictor.extra_keys),
        },
        "describe": again.predictor.describe(),
        "own_games": own,
        # None: no member was fine-tuned on own ladder games.
        "finetune": tuned,
        "old_ladder_rows": {
            "read": SET_LADDER in sets,
            "not_read_because": unread or None,
        },
        "sealed_rows_left_out": left_out if SET_LADDER in sets else None,
        "checks": checks,
        "sets": {},
        "artifact": None,
    }
    if not checks["ok"]:
        report["status"] = FAILED
        write_report(out_dir, report)
        failed = [key for key in gated if not checks[key]]
        raise EnsembleError(f"the written artifact failed its checks: {failed}")
    unverified.replace(path)
    report["artifact"] = {"path": str(path), "sha256": sha256_file(path)}
    say(f"VERIFIED {path}")

    # Reported, not gated: every member and the reloaded ensemble on each set.
    for label_of_set, batch in sets.items():
        preds: dict[str, S.Prediction] = {}
        seconds: dict[str, float] = {}
        for label in labels:
            preds[label], seconds[label] = predict(
                alone[label].predictor, batch, f"{label} on {label_of_set}"
            )
        preds[ENSEMBLE], seconds[ENSEMBLE] = predict(
            again.predictor, batch, f"{ENSEMBLE} on {label_of_set}"
        )
        found = score(batch, preds, seconds, int(args.resamples), int(args.seed))
        found["mean_identity"] = mean_identity(
            preds[ENSEMBLE], [preds[label] for label in labels], weights, batch
        )
        report["sets"][label_of_set] = found
        say(
            f"{label_of_set}: fine NLL "
            + ", ".join(
                f"{who} {row['fine_nll']['value']:.4f}"
                for who, row in found["predictors"].items()
            )
        )
    report["predictor_counters"] = dict(again.predictor.counters)
    report["seconds"] = round(time.time() - started, 1)
    report["status"] = DONE
    write_report(out_dir, report)
    return DONE, report


# --- the report -----------------------------------------------------------------


def _number(value: Any, digits: int = 4) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def _level(found: Mapping[str, Any] | None) -> str:
    if not found or found.get("value") is None:
        return "-"
    if found.get("low") is None:
        return _number(found["value"])
    return f"{found['value']:.4f} [{found['low']:.4f}, {found['high']:.4f}]"


def _signed(found: Mapping[str, Any] | None) -> str:
    if not found or found.get("diff") is None:
        return "-"
    if found.get("low") is None:
        return f"{found['diff']:+.4f}"
    text = f"{found['diff']:+.4f} [{found['low']:+.4f}, {found['high']:+.4f}]"
    clear = found["low"] > 0 or found["high"] < 0
    return f"**{text}**" if clear else text


def _percent(value: Any) -> str:
    return "-" if value is None else f"{100.0 * float(value):.2f}%"


def render(report: Mapping[str, Any]) -> str:
    checks = report.get("checks") or {}
    identity = checks.get("mean_identity") or {}
    gaps = identity.get("max_abs_difference") or {}
    lines = [
        f"# Ensemble artifact `{report.get('name')}`",
        "",
        f"Written {report.get('created')} by `training/make_oppmodel_ensemble.py`. "
        f"Status: **{report.get('status')}**. Nothing was trained: the artifact "
        "holds the members' own predictor payloads unchanged and serves the "
        "weighted mean of their fine distributions (`vgc_bench/src/oppmodel/"
        "ensemble.py`).",
        "",
        "## Members",
        "",
        "| label | weight | artifact | sha256 | parameters | temperatures "
        "(action / target) | Elo |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, member in (report.get("members") or {}).items():
        heat = ((member.get("describe") or {}).get("temperatures")) or {}
        lines.append(
            f"| {label} | {_number(member.get('weight'))} | `{member.get('path')}` "
            f"| `{str(member.get('sha256'))[:16]}` | {member.get('n_parameters')} "
            f"| {_number(heat.get('action'))} / {_number(heat.get('target'))} "
            f"| {member.get('elo_mode')} |"
        )
    featurizer = report.get("featurizer") or {}
    artifact = report.get("artifact") or {}
    dataset = report.get("dataset") or {}
    tuned = report.get("finetune") or None
    if tuned:
        each = ", ".join(
            f"`{label}` on {count}" for label, count in tuned["by_member"].items()
        )
        lines += [
            "",
            f"**This ensemble has seen {tuned['seen_own_games']} of the bot's own "
            f"ladder games**: members fine-tuned on them: {each}. Any reading of "
            "it on those games is in-sample; they are listed in the artifact's "
            "`extra['finetune']['seen_games']`, and its dataset manifest is marked "
            "`finetuned-on-ladder-holdout`. The old ladder rows were not read here.",
        ]
    lines += [
        "",
        f"- Featurizer: member `{featurizer.get('of_member')}`'s, layout version "
        f"{featurizer.get('layout_version')}, version-2 arrays "
        f"{featurizer.get('extras')}; the members' payloads are "
        f"{'identical' if featurizer.get('identical_payloads') else 'NOT identical'}.",
        f"- Dataset: `{dataset.get('path')}` (manifest "
        f"`{str(dataset.get('manifest_sha256'))[:16]}`); members fitted on it: "
        f"{dataset.get('members_fitted_on_it')}.",
        f"- Artifact: `{artifact.get('path')}` (sha256 "
        f"`{str(artifact.get('sha256'))[:16]}`), kind `{report.get('kind')}`, "
        f"artifact version {checks.get('artifact_version')}. Code from before this "
        "kind refuses the file (unknown kind).",
        "",
        "## Checks of the written file",
        "",
        f"On {identity.get('rows')} validation rows (a seeded sample), the "
        "reloaded artifact against the members loaded on their own; the bar is "
        f"{checks.get('tolerance')}.",
        "",
        "| check | result |",
        "|---|---|",
        f"| max abs difference to the mean of the members: fine distribution | "
        f"{gaps.get('fine')} |",
        f"| the same: action marginal | {gaps.get('action')} |",
        f"| the same: Mega probability | {gaps.get('mega')} |",
        f"| largest change `normalize_prediction` makes | "
        f"{identity.get('max_change_by_normalising')} |",
        f"| largest gap of an acting slot's fine sum to 1 | "
        f"{identity.get('max_gap_of_a_slot_sum_to_one')} |",
    ]
    for key in (
        "equals_the_mean_of_the_members",
        "is_a_proper_prediction",
        "member_payloads_unchanged",
        "member_sha256_recorded",
        "weights_as_built",
        "featurizer_is_the_members",
        "no_coupling",
        "no_predictor_failure",
        "state_named_and_kept",
        "seen_games_carried",
        "ok",
    ):
        lines.append(f"| {key} | {checks.get(key)} |")
    lines += [
        "",
        f"State `{checks.get('state_sha256')}` (the weights and each member's own "
        "payload, hashed): a pair coupling fitted on this artifact records it and "
        "is refused on any other weights or members.",
    ]
    own = report.get("own_games") or {}
    if tuned:
        ladder = (
            "The ladder holdout was not read at all: a fine-tuned member has "
            "seen its old games, so a reading of them would be in-sample."
        )
    elif report.get("sealed_rows_left_out") is None:
        why = (report.get("old_ladder_rows") or {}).get("not_read_because")
        ladder = f"The ladder holdout was not read: {why}."
    else:
        ladder = (
            f"Own games from {own.get('sealed_from')} on (the sealed set): "
            f"{report.get('sealed_rows_left_out')} rows left out of the ladder "
            "rows by their own times."
        )
    lines += [
        "",
        "## Readings (reported, not gated)",
        "",
        "Fine NLL per scored slot-turn (lower is better), with a 95% interval "
        "from resampling whole games; differences are paired on the same rows "
        "and bold when the interval excludes zero. The split `test` was not "
        f"read. {ladder}",
    ]
    for label, found in (report.get("sets") or {}).items():
        slots = found.get("slots") or {}
        whole = (found.get("mean_identity") or {}).get("max_abs_difference") or {}
        lines += [
            "",
            f"**{label}**: {found.get('examples')} rows, {found.get('games')} games, "
            f"{slots.get('scored')} scored slot-turns ({slots.get('censored')} "
            f"censored); {found.get('resamples')} resamples. Largest difference "
            f"to the mean of the members over all rows: fine {whole.get('fine')}.",
            "",
            "| | fine NLL | top-1 | top-3 | ensemble - this | by account | "
            "microseconds per row |",
            "|---|---|---|---|---|---|---|",
        ]
        for who, row in (found.get("predictors") or {}).items():
            difference = (row.get("ensemble_minus_this") or {}).get("fine")
            lines.append(
                f"| {who} | {_level(row.get('fine_nll'))} "
                f"| {_percent(row.get('fine_top1'))} "
                f"| {_percent(row.get('fine_top3'))} | {_signed(difference)} "
                f"| {_signed(row.get('ensemble_minus_this_by_account'))} "
                f"| {_number(row.get('microseconds_per_example'), 0)} |"
            )
    if tuned:
        untouched = (
            "- The validation split chose each member's epoch and temperatures, "
            "so it is not untouched. The old ladder games are not untouched "
            f"either: members {', '.join(f'`{one}`' for one in tuned['members'])} "
            "were fitted on them, which is why they were not scored here. The "
            "honest numbers of a fine-tuned member on those games are the "
            "out-of-fold ones of its own fine-tune report."
        )
    else:
        untouched = (
            "- The validation split chose each member's epoch and temperatures, "
            "so it is not untouched; the old ladder rows were used for none of "
            "that."
        )
    lines += [
        "",
        "## Not in this file",
        "",
        "- No pair coupling: fit one on this artifact with "
        "`training/fit_oppmodel_coupling.py`.",
        untouched,
        "- A forecast reading only: nothing here says what the mean does to a "
        "decision or a win rate.",
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
    parser.add_argument(
        "--member",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="a member artifact (repeat; the order is the order of --weights)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="the dataset the members were trained on (its val split and old "
        "ladder rows are read; test never)",
    )
    parser.add_argument("--out", required=True, help="a NEW directory")
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="one positive number per member, in member order (default: equal)",
    )
    parser.add_argument("--name", default="", help="default: the name of --out")
    parser.add_argument("--check-rows", type=int, default=DEFAULT_CHECK_ROWS)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=S.DEFAULT_SEED)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--allow-other-dataset",
        action="store_true",
        help="go on when a member was not fitted on --dataset",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import torch

        torch.set_num_threads(max(1, int(args.threads)))
    except Exception:
        pass
    try:
        status, _ = run(args)
    except EnsembleError as exc:
        say(f"{FAILED} {exc}")
        return 1
    say(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
