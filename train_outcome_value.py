"""Train and calibrate the exact planner's terminal-outcome value model."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, Dataset, Subset

from vgc_bench.src.outcome_value import calibration_error, critic_logits


class OutcomeDataset(Dataset):
    def __init__(self, directory: Path):
        self.rows = []
        self.opponents: list[str] = []
        self.styles: list[str] = []
        self.previews: list[bool] = []
        for path in sorted(directory.glob("outcome_*.npz")):
            payload = np.load(path)
            metadata = [json.loads(str(value)) for value in payload["metadata"]]
            for index, meta in enumerate(metadata):
                self.rows.append(
                    (
                        payload["observations"][index].astype(np.float32),
                        payload["action_masks"][index].astype(np.float32),
                        np.float32(payload["targets"][index]),
                        np.float32(payload["critic_values"][index]),
                        np.float32(payload["hybrid_values"][index]),
                    )
                )
                self.opponents.append(str(meta["opponent"]))
                self.styles.append(str(meta.get("opponent_style", "unknown")))
                self.previews.append(bool(meta.get("preview", False)))
        if not self.rows:
            raise ValueError(f"no outcome_*.npz rows in {directory}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def _split(dataset: OutcomeDataset, seed: int):
    opponents = sorted(set(dataset.opponents))
    random.Random(seed).shuffle(opponents)
    if len(opponents) < 3:
        raise ValueError("outcome split requires at least three opponent teams")
    train_end = max(1, int(len(opponents) * 0.80))
    validation_end = max(train_end + 1, int(len(opponents) * 0.90))
    validation_end = min(validation_end, len(opponents) - 1)
    groups = {
        "train": set(opponents[:train_end]),
        "validation": set(opponents[train_end:validation_end]),
        "test": set(opponents[validation_end:]),
    }
    indices = {
        name: [
            index
            for index, opponent in enumerate(dataset.opponents)
            if opponent in selected
        ]
        for name, selected in groups.items()
    }
    if any(not selected for selected in indices.values()):
        raise ValueError("outcome split produced an empty partition")
    return indices, {name: sorted(selected) for name, selected in groups.items()}


def _trainable_value_parameters(policy):
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    modules = [policy.value_net]
    if policy.share_features_extractor:
        modules.append(policy.features_extractor)
    else:
        modules.append(policy.vf_features_extractor)
    critic_net = getattr(policy.mlp_extractor, "value_net", None)
    if critic_net is not None:
        modules.append(critic_net)
    parameters = []
    seen = set()
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            if id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    return parameters


def _probability_metrics(probabilities, targets):
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1 - 1e-6)
    targets = np.asarray(targets, dtype=np.float64)
    return {
        "brier": float(np.mean((probabilities - targets) ** 2)),
        "log_loss": float(
            -np.mean(
                targets * np.log(probabilities)
                + (1 - targets) * np.log(1 - probabilities)
            )
        ),
        "calibration_error": calibration_error(probabilities, targets),
        "accuracy": float(np.mean((probabilities >= 0.5) == (targets >= 0.5))),
        "mean_probability": float(probabilities.mean()),
        "mean_target": float(targets.mean()),
    }


@torch.no_grad()
def _collect(policy, loader, device):
    policy.eval()
    logits = []
    targets = []
    critic_values = []
    hybrid_values = []
    for observations, masks, target, critic, hybrid in loader:
        obs_dict = {
            "observation": observations.to(device),
            "action_mask": masks.to(device),
        }
        logits.append(critic_logits(policy, obs_dict).cpu())
        targets.append(target.cpu())
        critic_values.append(critic.cpu())
        hybrid_values.append(hybrid.cpu())
    return tuple(
        torch.cat(values).numpy()
        for values in (logits, targets, critic_values, hybrid_values)
    )


def _fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    logits_t = torch.as_tensor(logits, dtype=torch.float64)
    targets_t = torch.as_tensor(targets, dtype=torch.float64)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.25, max_iter=80, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(logits_t / temperature, targets_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def _evaluate_arrays(logits, targets, critic, hybrid, temperature):
    learned = 1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -40, 40)))
    critic_probability = (np.clip(critic, -1, 1) + 1.0) * 0.5
    hybrid_probability = (np.clip(hybrid, -1, 1) + 1.0) * 0.5
    return {
        "outcome_model": _probability_metrics(learned, targets),
        "champion_critic": _probability_metrics(critic_probability, targets),
        "hybrid_evaluator": _probability_metrics(hybrid_probability, targets),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="outcome_data_v1")
    parser.add_argument("--checkpoint", default="results_repaired/champion.zip")
    parser.add_argument("--output", default="results_outcome_v1/outcome_value.zip")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--minimum-states", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--holdout-style",
        default="",
        help=(
            "opponent style excluded from TRAINING entirely; the trained model "
            "must beat the champion-critic Brier on that style's rows, which "
            "measures generalization across playing styles rather than teams"
        ),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = OutcomeDataset(Path(args.data))
    if len(dataset) < args.minimum_states:
        raise SystemExit(
            f"only {len(dataset)} outcome states; minimum is {args.minimum_states}"
        )
    indices, opponent_split = _split(dataset, args.seed)
    holdout_indices: list[int] = []
    if args.holdout_style:
        holdout_indices = [
            index
            for index, style in enumerate(dataset.styles)
            if style == args.holdout_style
        ]
        if not holdout_indices:
            raise SystemExit(f"--holdout-style {args.holdout_style} matches no rows")
        excluded = set(holdout_indices)
        before = len(indices["train"])
        indices["train"] = [i for i in indices["train"] if i not in excluded]
        print(
            f"holdout style '{args.holdout_style}': removed "
            f"{before - len(indices['train'])} rows from training; "
            f"{len(holdout_indices)} rows reserved for the style gate",
            flush=True,
        )
    loaders = {
        "train": DataLoader(
            Subset(dataset, indices["train"]), batch_size=args.batch_size, shuffle=True
        ),
        "validation": DataLoader(
            Subset(dataset, indices["validation"]), batch_size=args.batch_size
        ),
        "test": DataLoader(
            Subset(dataset, indices["test"]), batch_size=args.batch_size
        ),
    }
    model = PPO.load(args.checkpoint, device=args.device)
    policy = model.policy
    parameters = _trainable_value_parameters(policy)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-5)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_loss = math.inf
    selected_epoch = 0

    for epoch in range(1, args.epochs + 1):
        policy.train()
        total_loss = 0.0
        seen = 0
        for observations, masks, targets, _critic, _hybrid in loaders["train"]:
            obs_dict = {
                "observation": observations.to(policy.device),
                "action_mask": masks.to(policy.device),
            }
            target = targets.to(policy.device)
            optimizer.zero_grad(set_to_none=True)
            logits = critic_logits(policy, obs_dict)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 0.5)
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            seen += len(target)
        validation = _collect(policy, loaders["validation"], policy.device)
        validation_loss = float(
            F.binary_cross_entropy_with_logits(
                torch.as_tensor(validation[0]), torch.as_tensor(validation[1])
            )
        )
        record = {
            "epoch": epoch,
            "train_log_loss": total_loss / max(1, seen),
            "validation_log_loss": validation_loss,
        }
        history.append(record)
        epoch_path = output.with_name(f"{output.stem}_epoch{epoch}{output.suffix}")
        model.save(epoch_path)
        if validation_loss < best_loss:
            best_loss = validation_loss
            selected_epoch = epoch
            model.save(output)
        print(
            f"epoch {epoch}: train_log_loss={record['train_log_loss']:.4f} "
            f"validation_log_loss={validation_loss:.4f}",
            flush=True,
        )

    selected = PPO.load(output, device=args.device)
    validation = _collect(
        selected.policy, loaders["validation"], selected.policy.device
    )
    temperature = _fit_temperature(validation[0], validation[1])
    test = _collect(selected.policy, loaders["test"], selected.policy.device)
    test_metrics = _evaluate_arrays(*test, temperature)

    def subset_metrics(subset_indices: list[int]) -> dict | None:
        if len(subset_indices) < 50:
            return None
        loader = DataLoader(Subset(dataset, subset_indices), batch_size=args.batch_size)
        arrays = _collect(selected.policy, loader, selected.policy.device)
        return _evaluate_arrays(*arrays, temperature)

    # style axis: the v1 net's headline metrics hid a total blindness to
    # unfamiliar playing styles (and to preview states); report each slice
    test_index_list = list(indices["test"])
    by_style = {}
    for style in sorted(set(dataset.styles)):
        rows = [i for i in test_index_list if dataset.styles[i] == style]
        metrics_for_style = subset_metrics(rows)
        if metrics_for_style is not None:
            by_style[style] = {
                "states": len(rows),
                **{k: v["brier"] for k, v in metrics_for_style.items()},
            }
    preview_rows = [i for i in test_index_list if dataset.previews[i]]
    move_rows = [i for i in test_index_list if not dataset.previews[i]]
    preview_metrics = subset_metrics(preview_rows)
    move_metrics = subset_metrics(move_rows)
    holdout_metrics = subset_metrics(holdout_indices) if holdout_indices else None

    style_gate = all(
        slice_metrics["outcome_model"] <= slice_metrics["champion_critic"]
        for slice_metrics in by_style.values()
    )
    holdout_gate = (
        holdout_metrics is None
        or holdout_metrics["outcome_model"]["brier"]
        <= holdout_metrics["champion_critic"]["brier"]
    )
    strongest_brier = min(
        test_metrics["champion_critic"]["brier"],
        test_metrics["hybrid_evaluator"]["brier"],
    )
    strongest_log_loss = min(
        test_metrics["champion_critic"]["log_loss"],
        test_metrics["hybrid_evaluator"]["log_loss"],
    )
    data_gate = (
        test_metrics["outcome_model"]["brier"] <= strongest_brier * 0.95
        and test_metrics["outcome_model"]["log_loss"] <= strongest_log_loss * 0.95
        and test_metrics["outcome_model"]["calibration_error"] <= 0.05
    )
    metrics = {
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "data": str(Path(args.data).resolve()),
        "states": len(dataset),
        "games": len(list(Path(args.data).glob("outcome_*.npz"))),
        "selected_epoch": selected_epoch,
        "opponent_split": {name: len(value) for name, value in opponent_split.items()},
        "history": history,
        "calibration": {"temperature": temperature},
        "test": test_metrics,
        "test_by_style": by_style,
        "test_preview_states": (
            {"states": len(preview_rows), **preview_metrics["outcome_model"]}
            if preview_metrics
            else None
        ),
        "test_move_states": (
            {"states": len(move_rows), **move_metrics["outcome_model"]}
            if move_metrics
            else None
        ),
        "holdout_style": args.holdout_style or None,
        "holdout_metrics": (
            {
                "states": len(holdout_indices),
                "outcome_model": holdout_metrics["outcome_model"],
                "champion_critic": holdout_metrics["champion_critic"],
            }
            if holdout_metrics
            else None
        ),
        "acceptance": {
            "data_metrics_passed": data_gate,
            "per_style_passed": style_gate,
            "holdout_style_passed": holdout_gate,
            "required_relative_improvement": 0.05,
            "maximum_calibration_error": 0.05,
            "tactical_gate_pending": True,
            "ladder_brier_gate_pending": True,
        },
    }
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        f"saved {output}; selected_epoch={selected_epoch}; "
        f"temperature={temperature:.3f}; data_gate={data_gate}; "
        f"style_gate={style_gate}; holdout_gate={holdout_gate}",
        flush=True,
    )
    if preview_metrics:
        print(
            f"preview states: n={len(preview_rows)} "
            f"brier={preview_metrics['outcome_model']['brier']:.4f} "
            f"mean_p={preview_metrics['outcome_model']['mean_probability']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
