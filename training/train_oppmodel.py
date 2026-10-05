"""Train OppNet, the neural opponent predictor (OPPONENT_PREDICTOR.md, "Models").

    .venv/bin/python training/train_oppmodel.py --out results_oppmodel/oppnet_v1 \
        --threads 6
    .venv/bin/python training/train_oppmodel.py --out results_oppmodel/oppnet_v1_blind \
        --elo-mode blank --threads 6          # the Elo-blind retrain of reading R2
    .venv/bin/python training/train_oppmodel.py --out results_oppmodel/smoke_oppnet \
        --limit 3000 --epochs 2 --threads 1   # smoke: proves the code runs

Data. ``features.load_dataset`` reads only the ``train`` and ``val`` splits into
memory. The network is fitted on ``train``; ``val`` decides the best epoch, the
early stop and the calibration (a temperature for the action head and for the
target head, a temperature and a bias for the Mega head). Validation leaves out
the battles that have a ladder-holdout opponent (``m_flag`` bit 4): the dataset
keeps those players out of training only, so their other games must not choose
the epoch either. ``test`` and ``ladder_holdout`` are never loaded unless
``--informational`` is given, and then only after the artifact is written, for
one printed line each: the scorecard is the reading, not this.

Augmentation, per example and per epoch: slot ``a`` <-> ``b`` mirrored on the
actor side and, independently, on the other side (``--slot-swap``), and both
ratings blanked (``--elo-drop``) so that "unknown Elo" is in distribution. With
``--elo-mode blank`` every rating is blanked in train and validation, and the
artifact says so: the predictor then blanks ratings at predict time.

Log lines a monitor can follow, all flushed: ``step ...`` every 50 steps,
``EPOCH ...`` after every epoch (epoch 0 is the untrained network, which is the
species prior), and a last line ``TRAIN_DONE`` or ``TRAIN_FAILED <reason>``.

After training: the best epoch's weights (a cloned snapshot) are restored, the
heads are calibrated on validation NLL, the artifact is written through
``oppmodel.artifact.save_artifact`` as ``artifact.pt.unverified``, RELOADED
through ``oppmodel.artifact.load_predictor`` and re-scored on validation with
``features.slot_nll``. The reloaded value must reproduce the recorded best
(within 1e-4 before temperature); only then is the file renamed to
``artifact.pt``. A run that fails this check leaves no ``artifact.pt``.

Outputs under ``--out``: ``artifact.pt``, ``train_report.json`` and
``train_report.md`` (rendered from the JSON). The report is written with
status ``TRAIN_RUNNING`` when the run starts and ends as ``TRAIN_DONE`` or
``TRAIN_FAILED``, so a directory never shows a finished report of another
run. A directory that already holds an artifact or a report is refused
without ``--overwrite``; with it the old files are removed first.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import importlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vgc_bench.src.oppmodel import features as F
from vgc_bench.src.oppmodel import model as M

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "results_oppmodel/v1_ondisk"
SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
INFORMATIONAL_SPLITS: tuple[str, ...] = ("test", "ladder_holdout")
ARTIFACT_NAME = "artifact.pt"
UNVERIFIED_SUFFIX = ".unverified"  # the artifact before its reload check
REPORT_JSON = "train_report.json"
REPORT_MD = "train_report.md"
DONE = "TRAIN_DONE"
FAILED = "TRAIN_FAILED"
RUNNING = "TRAIN_RUNNING"
FLAG_HOLDOUT_BATTLE = 4  # m_flag bit: the battle has a ladder-holdout opponent
MEGA_ROUNDS = 3  # alternations of the Mega head's temperature and bias
RELOAD_TOLERANCE = 1.0e-4
MIN_IMPROVEMENT = 1.0e-4
PROGRESS_EVERY = 50


def say(text: str) -> None:
    print(text, flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out", default=None, help="results_oppmodel/<run>")
    parser.add_argument("--tag", default=None, help="name stored in the artifact")
    parser.add_argument("--elo-mode", choices=(F.ELO_KEEP, F.ELO_BLANK), default="keep")
    parser.add_argument("--elo-drop", type=float, default=0.15)
    parser.add_argument("--slot-swap", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=300, help="warm-up steps")
    parser.add_argument("--clip", type=float, default=1.0, help="gradient norm cap")
    parser.add_argument("--mega-weight", type=float, default=0.2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="smoke: examples")
    parser.add_argument(
        "--informational",
        action="store_true",
        help="after the artifact is written, print test / ladder-holdout NLL once",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the artifact and report that --out already holds",
    )
    return parser.parse_args(argv)


# --- data ---------------------------------------------------------------------


def manifest_sha256(directory: Path | str) -> str:
    return hashlib.sha256((Path(directory) / "manifest.json").read_bytes()).hexdigest()


def load_splits(
    directory: Path | str, names: Sequence[str]
) -> tuple[dict[str, F.Batch], dict[str, Any]]:
    """Only the named splits of a built dataset, each as its own batch."""
    data, manifest = F.load_dataset(directory, splits=list(names))
    codes = list(manifest.get("splits") or [])
    out = {
        name: F.take(data, np.asarray(data["m_split"]) == codes.index(name))
        for name in names
    }
    return out, manifest


def without_holdout_battles(batch: F.Batch) -> tuple[F.Batch, int]:
    """(the batch without battles that have a ladder-holdout opponent, rows left out).

    The dataset keeps a ladder-holdout opponent's other games out of TRAINING
    only; validation chooses the epoch and the calibration, so they are left
    out of it here. A batch without ``m_flag`` is returned as it is.
    """
    if "m_flag" not in batch:
        return batch, 0
    marked = (np.asarray(batch["m_flag"]).astype(np.int64) & FLAG_HOLDOUT_BATTLE) > 0
    if not marked.any():
        return batch, 0
    return F.take(batch, ~marked), int(marked.sum())


def subset(batch: F.Batch, limit: int | None, rng: np.random.Generator) -> F.Batch:
    """At most ``limit`` examples, drawn without replacement, in dataset order."""
    n = int(batch["act_mon"].shape[0])
    if limit is None or limit >= n:
        return batch
    return F.take(batch, np.sort(rng.choice(n, size=max(1, limit), replace=False)))


def augment(
    batch: Mapping[str, np.ndarray],
    rng: np.random.Generator,
    slot_swap: float,
    elo_drop: float,
) -> F.Batch:
    """One epoch's view of the training set; the input is not changed.

    Each example has its actor slots mirrored with probability ``slot_swap``
    and, independently, the other side's; and both ratings blanked with
    probability ``elo_drop``. The three draws are always made, whatever the
    probabilities, so a run with and a run without Elo (same seed) see the
    same mirrors and the same example order: they differ in the Elo inputs only.
    """
    n = int(np.asarray(batch["act_mon"]).shape[0])
    out: F.Batch = dict(batch)
    draws = rng.random((3, n))
    if slot_swap > 0:
        out = M.swap_slots(out, draws[0] < slot_swap, draws[1] < slot_swap)
    if elo_drop > 0:
        drop = (draws[2] < elo_drop)[:, None]
        elo, known = np.asarray(out["elo"]), np.asarray(out["elo_known"])
        out["elo"] = np.where(drop, 0, elo).astype(elo.dtype)
        out["elo_known"] = np.where(drop, 0, known).astype(known.dtype)
    return out


# --- schedule and scoring -----------------------------------------------------


def learning_rate(
    step: int, total: int, warmup: int, base: float, floor: float = 0.05
) -> float:
    """Linear warm-up to ``base``, then a cosine down to ``floor * base``."""
    if warmup > 0 and step < warmup:
        return base * (step + 1) / warmup
    span = max(1, total - warmup)
    progress = min(1.0, max(0.0, (step - warmup) / span))
    return base * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def parameter_groups(net: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Weight decay on matrices only: biases, norms and scalars are left alone."""
    decay = [p for p in net.parameters() if p.requires_grad and p.ndim >= 2]
    plain = [p for p in net.parameters() if p.requires_grad and p.ndim < 2]
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": plain, "weight_decay": 0.0},
    ]


def _mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(values[mask].mean()) if bool(mask.any()) else float("nan")


def _predict_errors(predictor: M.OppNetPredictor) -> int:
    return sum(
        count
        for name, count in predictor.counters.items()
        if name.startswith("predict_error")
    )


def score(
    predictor: M.OppNetPredictor, batch: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """A split's scores through ``features.slot_nll`` (the scorecard's helpers).

    ``fine`` is the mean per scored slot of action + target NLL; ``target`` is
    the target term's share of it (so ``fine = action + target``) and
    ``target_given`` its mean over the slots that have a target label. The
    predictor sees the batch without labels and meta arrays. Raises when the
    predictor fell back to its uniform answer.
    """
    before = _predict_errors(predictor)
    features_only = F.sheet_unknown_as_closed(M.strip_labels(batch))
    pred = predictor.predict(features_only)
    if _predict_errors(predictor) != before:
        raise RuntimeError(
            f"predictor failed while scoring: {dict(predictor.counters)}"
        )
    nll = F.slot_nll(pred, batch)
    scored = nll["fine_scored"]
    fine_hits, fine_scored = F.topk_hits(
        F.fine_probs(pred, batch), F.fine_label(batch), 1
    )
    action_hits, action_scored = F.topk_hits(pred["action"], batch["y_action"], 1)
    slots = int(scored.sum())
    return {
        "fine": _mean(nll["fine"], scored),
        "action": _mean(nll["action"], nll["action_scored"]),
        "target": float(nll["target"].sum() / slots) if slots else float("nan"),
        "target_given": _mean(nll["target"], nll["target_scored"]),
        "mega": _mean(nll["mega"], nll["mega_scored"]),
        "top1": _mean(fine_hits.astype(np.float64), fine_scored),
        "action_top1": _mean(action_hits.astype(np.float64), action_scored),
        "slots": slots,
        "examples": int(np.asarray(batch["act_mon"]).shape[0]),
    }


def calibrate(
    net: M.OppNet, batch: Mapping[str, np.ndarray], device: torch.device
) -> dict[str, float]:
    """The calibration of each head, each minimising its own validation NLL.

    ``action`` and ``target`` are temperatures. The Mega head gets a
    temperature and a bias (``mega``, ``mega_bias``): the best epoch is chosen
    on the fine NLL, by which time the Mega head is past its own best and
    over-confident, and its base rate has drifted too. The two are fitted in
    alternation; each step keeps its value unless it is strictly better.
    """
    outputs = M.collect_outputs(net, M.strip_labels(batch), device)
    labels = M.to_labels(batch)

    def head(name: str, **temperatures: float) -> float:
        terms = M.nll_terms(outputs, labels, **temperatures)
        scored = terms[f"{name}_scored"]
        count = int(scored.sum())
        return float(terms[name][scored].mean()) if count else float("nan")

    action = M.fit_temperature(lambda t: head("action", action_temperature=t))
    target = M.fit_temperature(lambda t: head("target", target_temperature=t))
    mega, bias = 1.0, 0.0
    for _ in range(MEGA_ROUNDS):
        held = bias
        mega = M.fit_temperature(
            lambda t: head("mega", mega_temperature=t, mega_bias=held)
        )
        scale = mega
        bias = M.fit_offset(lambda b: head("mega", mega_temperature=scale, mega_bias=b))
    if not head("mega", mega_temperature=mega, mega_bias=bias) < head("mega"):
        mega, bias = 1.0, 0.0
    return {"action": action, "target": target, "mega": mega, "mega_bias": bias}


# --- training -----------------------------------------------------------------


def _device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("mps was asked for and is not available")
    return torch.device(name)


def _heartbeat(row: Mapping[str, Any], epochs: int, best: Mapping[str, Any]) -> str:
    return (
        f"EPOCH {row['epoch']}/{epochs} train_loss {row['train_loss']:.4f} "
        f"val_fine {row['val_fine']:.4f} = action {row['val_action']:.4f} "
        f"+ target {row['val_target']:.4f} "
        f"(target|label {row['val_target_given']:.4f}) "
        f"val_top1 {row['val_top1']:.4f} action_top1 {row['val_action_top1']:.4f} "
        f"val_mega {row['val_mega']:.4f} secs {row['seconds']:.1f} lr {row['lr']:.2e} "
        f"best {best['val_fine']:.4f}@{best['epoch']}"
    )


def resolve_out(args: argparse.Namespace) -> tuple[str, Path]:
    """(the model's name, the run directory) of a set of arguments."""
    name = args.tag or (Path(args.out).name if args.out else "oppnet")
    out_dir = Path(args.out) if args.out else ROOT / "results_oppmodel" / name
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    return str(name), out_dir


def run_files(out_dir: Path) -> list[Path]:
    """Every file a run writes into its directory."""
    artifact = out_dir / ARTIFACT_NAME
    return [
        artifact,
        artifact.with_name(artifact.name + UNVERIFIED_SUFFIX),
        out_dir / REPORT_JSON,
        out_dir / REPORT_MD,
    ]


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Run one training and return its report. Raises on any failure.

    Refuses a directory that already holds an artifact or a report unless
    ``args.overwrite`` is set (the old files are then removed before anything
    else happens). From then on the directory is this run's: its report says
    ``TRAIN_RUNNING`` until the run ends, and ``TRAIN_FAILED`` with the reason
    if it raises.
    """
    name, out_dir = resolve_out(args)
    found = [path for path in run_files(out_dir) if path.exists()]
    if found and not getattr(args, "overwrite", False):
        raise RuntimeError(
            f"{found[0]} exists; give --overwrite to replace the run in {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in found:
        path.unlink()
    opening = {
        "status": RUNNING,
        "name": name,
        "kind": M.KIND,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "args": {key: value for key, value in sorted(vars(args).items())},
    }
    write_report(out_dir, opening)
    try:
        return _train(args, name, out_dir)
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


def _train(args: argparse.Namespace, name: str, out_dir: Path) -> dict[str, Any]:
    started = time.time()
    torch.set_num_threads(max(1, int(args.threads)))
    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    device = _device(args.device)
    dataset = Path(args.dataset)
    if not dataset.is_absolute():
        dataset = ROOT / dataset

    splits, manifest = load_splits(dataset, (SPLIT_TRAIN, SPLIT_VAL))
    featurizer = F.Featurizer.load(dataset, elo_mode=args.elo_mode)
    if featurizer.signature_diff:
        say(f"WARNING dex signature differs: {featurizer.signature_diff}")
    validation, left_out = without_holdout_battles(splits[SPLIT_VAL])
    train_set = F.sheet_unknown_as_closed(subset(splits[SPLIT_TRAIN], args.limit, rng))
    val_set = F.sheet_unknown_as_closed(subset(validation, args.limit, rng))
    if args.elo_mode == F.ELO_BLANK:
        train_set = F.apply_elo_mode(train_set, F.ELO_BLANK)
        val_set = F.apply_elo_mode(val_set, F.ELO_BLANK)
    n_train = int(train_set["act_mon"].shape[0])
    n_val = int(val_set["act_mon"].shape[0])
    if n_train == 0 or n_val == 0:
        raise RuntimeError(f"empty split: train {n_train}, val {n_val}")
    # The loss skips a target label that the legal-target mask excludes and
    # features.slot_nll does not; the two agree only while there is none.
    masked_targets = {
        SPLIT_TRAIN: M.count_masked_targets(train_set),
        SPLIT_VAL: M.count_masked_targets(val_set),
    }
    if any(masked_targets.values()):
        say(
            f"WARNING target labels outside the legal-target mask: {masked_targets}; "
            "the training loss skips them and features.slot_nll charges them"
        )

    net = M.OppNet.for_featurizer(
        featurizer,
        d_model=int(args.d_model),
        n_layers=int(args.layers),
        n_heads=int(args.heads),
        d_ff=int(args.ff),
        dropout=float(args.dropout),
    ).to(device)
    n_parameters = net.n_parameters()
    batch_size = max(1, int(args.batch))
    steps_per_epoch = math.ceil(n_train / batch_size)
    total_steps = max(1, int(args.epochs) * steps_per_epoch)
    warmup = min(max(0, int(args.warmup)), max(1, total_steps // 10))
    optimizer = torch.optim.AdamW(
        parameter_groups(net, float(args.weight_decay)), lr=float(args.lr)
    )
    live = M.OppNetPredictor(
        net, featurizer, name=name, elo_mode=args.elo_mode, device=device
    )
    say(
        f"TRAIN_START name {name} dataset {manifest.get('tag')} train {n_train} "
        f"val {n_val} (left out: {left_out} rows of holdout battles) "
        f"parameters {n_parameters} batch {batch_size} "
        f"steps/epoch {steps_per_epoch} epochs {args.epochs} device {device} "
        f"threads {torch.get_num_threads()} elo_mode {args.elo_mode}"
    )

    eval_started = time.time()
    first = score(live, val_set)
    eval_seconds = [time.time() - eval_started]
    best = {"epoch": 0, "val_fine": first["fine"]}
    best_state = M.clone_state(net)
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "train_loss": float("nan"),
            "train_fine": float("nan"),
            "lr": 0.0,
            "seconds": eval_seconds[0],
            "step_seconds": 0.0,
            **{f"val_{key}": value for key, value in first.items()},
        }
    ]
    say(_heartbeat(history[0], int(args.epochs), best))

    elo_drop = 0.0 if args.elo_mode == F.ELO_BLANK else float(args.elo_drop)
    step = 0
    stale = 0
    step_seconds_total = 0.0
    stopped_early = False
    lr = 0.0
    for epoch in range(1, int(args.epochs) + 1):
        epoch_started = time.time()
        net.train()
        view = augment(train_set, rng, float(args.slot_swap), elo_drop)
        order = rng.permutation(n_train)
        sums = {"loss": 0.0, "fine": 0.0}
        window_loss, window_steps, window_started = 0.0, 0, time.time()
        epoch_step_seconds = 0.0
        for position in range(steps_per_epoch):
            step_started = time.time()
            index = np.sort(order[position * batch_size : (position + 1) * batch_size])
            lr = learning_rate(step, total_steps, warmup, float(args.lr))
            for group in optimizer.param_groups:
                group["lr"] = lr
            labels = M.to_labels(view, device, index)
            terms = M.nll_terms(net(M.to_tensors(view, device, index)), labels)
            loss, parts = M.total_loss(terms, labels["weight"], float(args.mega_weight))
            if not math.isfinite(parts["loss"]):
                raise RuntimeError(f"non-finite loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), float(args.clip))
            optimizer.step()
            step += 1
            sums["loss"] += parts["loss"]
            sums["fine"] += parts["fine"]
            window_loss += parts["loss"]
            window_steps += 1
            epoch_step_seconds += time.time() - step_started
            if step % PROGRESS_EVERY == 0:
                per_step = (time.time() - window_started) / window_steps
                say(
                    f"  step {step}/{total_steps} epoch {epoch} "
                    f"loss {window_loss / window_steps:.4f} lr {lr:.2e} "
                    f"{per_step:.3f} s/step"
                )
                window_loss, window_steps, window_started = 0.0, 0, time.time()
        step_seconds_total += epoch_step_seconds
        eval_started = time.time()
        val = score(live, val_set)
        eval_seconds.append(time.time() - eval_started)
        row = {
            "epoch": epoch,
            "train_loss": sums["loss"] / steps_per_epoch,
            "train_fine": sums["fine"] / steps_per_epoch,
            "lr": lr,
            "seconds": time.time() - epoch_started,
            "step_seconds": epoch_step_seconds / steps_per_epoch,
            **{f"val_{key}": value for key, value in val.items()},
        }
        history.append(row)
        if val["fine"] < best["val_fine"] - MIN_IMPROVEMENT:
            best = {"epoch": epoch, "val_fine": val["fine"]}
            best_state = M.clone_state(net)
            stale = 0
        else:
            stale += 1
        say(_heartbeat(row, int(args.epochs), best))
        if stale >= int(args.patience):
            stopped_early = True
            say(f"EARLY_STOP epoch {epoch}: no improvement in {stale} epochs")
            break

    # The best epoch, on the CPU, is what gets calibrated, stored and re-read.
    net.load_state_dict(best_state)
    net.to("cpu")
    cpu = torch.device("cpu")
    raw = M.OppNetPredictor(net, featurizer, name=name, elo_mode=args.elo_mode)
    restored = score(raw, val_set)
    temperatures = calibrate(net, val_set, cpu)
    final = raw.with_temperatures(
        temperatures["action"],
        temperatures["target"],
        temperatures["mega"],
        temperatures["mega_bias"],
    )
    tempered = score(final, val_set)
    say(
        f"CALIBRATED action_T {temperatures['action']:.4f} "
        f"target_T {temperatures['target']:.4f} "
        f"val_fine {restored['fine']:.4f} -> {tempered['fine']:.4f} "
        f"mega_T {temperatures['mega']:.4f} mega_bias {temperatures['mega_bias']:+.4f} "
        f"val_mega {restored['mega']:.4f} -> {tempered['mega']:.4f}"
    )

    steps_done = max(1, step)
    report: dict[str, Any] = {
        "status": DONE,
        "name": name,
        "kind": M.KIND,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": {
            "path": str(dataset),
            "tag": manifest.get("tag"),
            "manifest_sha256": manifest_sha256(dataset),
            "formats": list(manifest.get("formats") or []),
            "validation_rows_left_out_holdout_battles": left_out,
        },
        "target_labels_outside_mask": masked_targets,
        "args": {key: value for key, value in sorted(vars(args).items())},
        "config": net.config.to_dict(),
        "n_parameters": n_parameters,
        "examples": {SPLIT_TRAIN: n_train, SPLIT_VAL: n_val},
        "elo_mode": args.elo_mode,
        "history": history,
        "best": best,
        "stopped_early": stopped_early,
        "calibration": {
            "action_temperature": temperatures["action"],
            "target_temperature": temperatures["target"],
            "mega_temperature": temperatures["mega"],
            "mega_bias": temperatures["mega_bias"],
            "val_before": restored,
            "val_after": tempered,
        },
        "timing": {
            "steps": step,
            "steps_per_epoch": steps_per_epoch,
            "seconds_per_step": step_seconds_total / steps_done,
            "seconds_per_validation": float(np.mean(eval_seconds)),
            "seconds_total": time.time() - started,
            "threads": torch.get_num_threads(),
            "device": str(device),
        },
        "torch": str(torch.__version__),
    }

    # Imported here, by name: the artifact module is the only reader and writer
    # of artifact files, and this script must import without it.
    artifact = importlib.import_module("vgc_bench.src.oppmodel.artifact")
    path = out_dir / ARTIFACT_NAME
    # Written under another name first: the file becomes artifact.pt only when
    # its reload reproduces the recorded best, so nothing that looks for
    # artifact.pt can pick up a model this run rejected.
    unverified = path.with_name(path.name + UNVERIFIED_SUFFIX)
    extra = _plain({key: value for key, value in report.items() if key != "status"})
    artifact.save_artifact(
        unverified,
        kind=M.KIND,
        name=name,
        featurizer=featurizer,
        predictor_payload=final.to_payload(),
        extra=extra,
    )
    loaded = artifact.load_predictor(unverified).predictor
    if not isinstance(loaded, M.OppNetPredictor):
        raise RuntimeError(f"artifact reloaded as {type(loaded).__name__}")
    again_tempered = score(loaded, val_set)
    again_raw = score(loaded.with_temperatures(1.0, 1.0), val_set)
    difference = abs(again_raw["fine"] - best["val_fine"])
    report["artifact"] = str(path)
    report["reload"] = {
        "val_fine_raw": again_raw["fine"],
        "val_fine_tempered": again_tempered["fine"],
        "recorded_best": best["val_fine"],
        "difference": difference,
        "tolerance": RELOAD_TOLERANCE,
    }
    say(
        f"RELOAD val_fine raw {again_raw['fine']:.6f} (recorded best "
        f"{best['val_fine']:.6f}, difference {difference:.2e}) "
        f"tempered {again_tempered['fine']:.6f}"
    )
    if not difference <= RELOAD_TOLERANCE:
        report["status"] = FAILED
        report["artifact"] = None
        report["rejected_artifact"] = str(unverified)
        write_report(out_dir, report)
        raise RuntimeError(
            f"reloaded artifact scores {again_raw['fine']:.6f}, "
            f"recorded best {best['val_fine']:.6f}"
        )
    unverified.replace(path)

    if args.informational:
        extra_splits, _ = load_splits(dataset, INFORMATIONAL_SPLITS)
        report["informational"] = {}
        for split in INFORMATIONAL_SPLITS:
            rows = F.sheet_unknown_as_closed(extra_splits[split])
            if int(rows["act_mon"].shape[0]) == 0:
                continue
            scores = score(loaded, rows)
            report["informational"][split] = scores
            say(
                f"INFORMATIONAL (not the reading; see the scorecard) {split} "
                f"fine {scores['fine']:.4f} action {scores['action']:.4f} "
                f"top1 {scores['top1']:.4f} slots {scores['slots']}"
            )
    report["timing"]["seconds_total"] = time.time() - started
    write_report(out_dir, report)
    return report


# --- report -------------------------------------------------------------------


def _plain(value: Any) -> Any:
    """JSON-safe copy: NaN becomes None, numpy scalars become Python numbers."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, str):
        return str(value)  # a plain str, also for a subclass such as a version
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _cell(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(report: Mapping[str, Any]) -> str:
    """The Markdown view of ``train_report.json``."""
    data = _plain(report)
    timing = data.get("timing") or {}
    calibration = data.get("calibration") or {}
    reload = data.get("reload") or {}
    best = data.get("best") or {}
    lines = [
        f"# OppNet training: {data.get('name')}",
        "",
        f"Status: {data.get('status')}"
        + (f" ({data.get('reason')})" if data.get("reason") else "")
        + f". Rendered from {REPORT_JSON}.",
        "",
        f"- dataset: {(data.get('dataset') or {}).get('tag')} "
        f"(manifest sha256 {(data.get('dataset') or {}).get('manifest_sha256')})",
        f"- examples: {data.get('examples')}",
        f"- parameters: {data.get('n_parameters')}",
        f"- Elo mode: {data.get('elo_mode')}",
        f"- best epoch: {best.get('epoch')} with validation fine NLL "
        f"{_cell(best.get('val_fine'))}; stopped early: {data.get('stopped_early')}",
        f"- temperatures: action {_cell(calibration.get('action_temperature'))}, "
        f"target {_cell(calibration.get('target_temperature'))}; validation fine "
        f"NLL {_cell((calibration.get('val_before') or {}).get('fine'))} -> "
        f"{_cell((calibration.get('val_after') or {}).get('fine'))}",
        f"- Mega head: temperature {_cell(calibration.get('mega_temperature'))}, "
        f"bias {_cell(calibration.get('mega_bias'))}; validation Mega NLL "
        f"{_cell((calibration.get('val_before') or {}).get('mega'))} -> "
        f"{_cell((calibration.get('val_after') or {}).get('mega'))}",
        f"- validation rows left out (battles with a ladder-holdout opponent): "
        f"{(data.get('dataset') or {}).get('validation_rows_left_out_holdout_battles')}"
        f"; target labels outside the legal-target mask: "
        f"{data.get('target_labels_outside_mask')}",
        f"- reloaded artifact: raw {_cell(reload.get('val_fine_raw'), 6)}, "
        f"tempered {_cell(reload.get('val_fine_tempered'), 6)}, difference to the "
        f"recorded best {_cell(reload.get('difference'), 8)}",
        f"- timing: {_cell(timing.get('seconds_per_step'))} s/step over "
        f"{timing.get('steps')} steps, {_cell(timing.get('seconds_per_validation'), 1)}"
        f" s per validation pass, {_cell(timing.get('seconds_total'), 1)} s in all "
        f"({timing.get('threads')} threads, {timing.get('device')})",
        "",
        "| epoch | train loss | val fine | val action | val target | top-1 "
        "| action top-1 | val mega | lr | seconds |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in data.get("history") or []:
        lr = row.get("lr")
        lines.append(
            f"| {row.get('epoch')} | {_cell(row.get('train_loss'))} "
            f"| {_cell(row.get('val_fine'))} | {_cell(row.get('val_action'))} "
            f"| {_cell(row.get('val_target'))} | {_cell(row.get('val_top1'))} "
            f"| {_cell(row.get('val_action_top1'))} | {_cell(row.get('val_mega'))} "
            f"| {'-' if lr is None else format(lr, '.2e')} "
            f"| {_cell(row.get('seconds'), 1)} |"
        )
    informational = data.get("informational") or {}
    if informational:
        lines += ["", "Informational only (the scorecard is the reading):", ""]
        for split, scores in informational.items():
            lines.append(
                f"- {split}: fine {_cell(scores.get('fine'))}, "
                f"top-1 {_cell(scores.get('top1'))}, slots {scores.get('slots')}"
            )
    return "\n".join(lines) + "\n"


def write_report(out_dir: Path, report: Mapping[str, Any]) -> None:
    data = _plain(report)
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
        report = train(args)
    except Exception as exc:
        reason = " ".join(f"{type(exc).__name__}: {exc}".split())
        say(f"{FAILED} {reason}")
        return 1
    say(
        f"{DONE} best_epoch {report['best']['epoch']} "
        f"val_fine {report['reload']['val_fine_tempered']:.4f} "
        f"artifact {report['artifact']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
