"""Distill exact multi-turn planner rankings into the PPO actor and critic."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, Dataset, Subset


class CounterfactualDataset(Dataset):
    def __init__(
        self,
        directory: Path | Sequence[Path],
        include_truncated: bool = False,
    ):
        directories = (
            [directory] if isinstance(directory, Path) else list(directory)
        )
        if not directories:
            raise ValueError("at least one counterfactual directory is required")
        rows = []
        groups = []
        for directory_index, source in enumerate(directories):
            for path in sorted(source.glob("moves_*.npz")):
                payload = np.load(path)
                metadata = [json.loads(str(value)) for value in payload["metadata"]]
                for index, meta in enumerate(metadata):
                    if (
                        meta.get("truncated")
                        and not meta.get("complete_screen", False)
                        and not include_truncated
                    ):
                        continue
                    rows.append(
                        (
                            payload["observations"][index].astype(np.float32),
                            payload["action_masks"][index].astype(np.float32),
                            payload["candidate_actions"][index].astype(np.int64),
                            payload["planner_scores"][index].astype(np.float32),
                        )
                    )
                    # Game numbers restart in every aggregation round. Keep the
                    # trajectory split disjoint by including the source directory.
                    groups.append(
                        (directory_index, int(meta.get("game", index)))
                    )
        if not rows:
            raise ValueError(
                f"no usable counterfactual move positions in {directories}; "
                "increase generation budget or pass --include-truncated"
            )
        # Each chunk pads its rows to its OWN widest candidate beam. Chunk widths
        # differ once determinization unions vary (first seen in v5h: 10 vs 11),
        # and torch's default collate cannot stack ragged rows. Normalize to the
        # global width with the established (-1, -1)/NaN padding convention that
        # joint_log_prob and the presence masks already handle.
        width = max(row[2].shape[0] for row in rows)
        padded = []
        for observation, mask, actions, scores in rows:
            count = actions.shape[0]
            if count < width:
                actions = np.concatenate(
                    [actions, np.full((width - count, 2), -1, dtype=actions.dtype)]
                )
                scores = np.concatenate(
                    [scores, np.full(width - count, np.nan, dtype=scores.dtype)]
                )
            padded.append((observation, mask, actions, scores))
        self.rows = padded
        self.groups = groups

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def joint_log_prob(policy, logits, mask, actions):
    """Log P(slot 0 action) + log P(slot 1 action | slot 0)."""
    batch, width, _ = actions.shape
    present = (actions >= 0).all(dim=2)
    if not present.any(dim=1).all():
        raise ValueError("every row needs at least one candidate action pair")
    # NPZ rows are padded to the widest candidate beam with (-1, -1). Feeding
    # clamped padding as `(pass, pass)` into the policy's dependent second head can
    # produce an all-masked Categorical before the caller gets a chance to discard
    # that slot. Borrow the row's first real pair solely for padded computation;
    # callers still exclude the placeholder with their original presence mask.
    fallback_index = present.to(torch.int64).argmax(dim=1)
    fallback = actions.gather(
        1, fallback_index[:, None, None].expand(-1, 1, 2)
    )
    safe = torch.where(present[:, :, None], actions, fallback)
    first_dist = policy.get_dist_from_logits(logits, mask).distribution[0]
    first_log = first_dist.logits.gather(1, safe[:, :, 0])

    repeated_logits = logits.repeat_interleave(width, dim=0)
    repeated_mask = mask.repeat_interleave(width, dim=0)
    first_actions = safe[:, :, 0].reshape(-1, 1)
    second_dist = policy.get_dist_from_logits(
        repeated_logits, repeated_mask, first_actions
    ).distribution[1]
    second_log = second_dist.logits.gather(1, safe[:, :, 1].reshape(-1, 1)).reshape(
        batch, width
    )
    return first_log + second_log


def _categorical_kl(current, reference):
    """KL(reference || current) without NaNs from masked-out actions."""
    target = reference.probs
    supported = target > 0
    current_log = torch.where(
        supported, current.logits, torch.zeros_like(current.logits)
    )
    reference_log = torch.where(
        supported, reference.logits, torch.zeros_like(reference.logits)
    )
    return (target * (reference_log - current_log)).sum(dim=1)


def _anchor_kl(policy, reference, logits, obs_dict, actions, valid):
    """Keep behavior outside the searched beam close to the ladder champion."""
    with torch.no_grad():
        reference_logits, _ = reference.get_logits(obs_dict, actor_grad=False)
    current_first = policy.get_dist_from_logits(
        logits, obs_dict["action_mask"]
    ).distribution[0]
    reference_first = reference.get_dist_from_logits(
        reference_logits, obs_dict["action_mask"]
    ).distribution[0]
    first_kl = _categorical_kl(current_first, reference_first).mean()

    width = actions.shape[1]
    safe_first = actions[:, :, 0].clamp_min(0).reshape(-1, 1)
    repeated_mask = obs_dict["action_mask"].repeat_interleave(width, dim=0)
    current_second = policy.get_dist_from_logits(
        logits.repeat_interleave(width, dim=0), repeated_mask, safe_first
    ).distribution[1]
    reference_second = reference.get_dist_from_logits(
        reference_logits.repeat_interleave(width, dim=0), repeated_mask, safe_first
    ).distribution[1]
    second_kl = _categorical_kl(current_second, reference_second)
    valid_flat = valid.reshape(-1)
    if valid_flat.any():
        second_kl = second_kl[valid_flat].mean()
    else:
        second_kl = torch.zeros((), device=logits.device)
    return first_kl + second_kl


def planner_confidence(
    scores: torch.Tensor,
    valid: torch.Tensor,
    minimum_margin: float,
    margin_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return safe scores, best indices, top-two margins, and row confidence."""
    safe_scores = scores.masked_fill(~valid, float("-inf"))
    best = safe_scores.argmax(dim=1)
    top = safe_scores.topk(k=min(2, scores.shape[1]), dim=1).values
    margin = (
        top[:, 0] - top[:, 1]
        if top.shape[1] == 2
        else torch.zeros_like(top[:, 0])
    )
    confidence = (margin / max(margin_scale, 1e-6)).clamp(0.0, 1.0)
    confidence = torch.where(
        margin >= minimum_margin, confidence, torch.zeros_like(confidence)
    )
    return safe_scores, best, margin, confidence


def loss_for_batch(
    policy,
    reference,
    batch,
    device,
    temperature: float,
    value_coef: float,
    anchor_weight: float,
    minimum_margin: float,
    margin_scale: float,
    pairwise_weight: float,
    listwise_weight: float,
):
    observations, masks, actions, scores = [tensor.to(device) for tensor in batch]
    present = (actions >= 0).all(dim=2)
    obs_dict = {"observation": observations, "action_mask": masks}
    logits, values = policy.get_logits(obs_dict, actor_grad=True)
    log_prob = joint_log_prob(policy, logits, masks, actions)
    # Showdown is the source of truth for legal choices, but a few Champions actions
    # cannot be represented by poke-env's older action mask. They have -inf policy
    # probability and previously made the entire batch loss infinite.
    valid = present & torch.isfinite(log_prob) & torch.isfinite(scores)
    row_valid = valid.any(dim=1)
    if not row_valid.any():
        raise RuntimeError(
            "batch contains no planner action representable by the policy"
        )
    valid_fraction = valid.sum() / present.sum().clamp_min(1)
    if not row_valid.all():
        actions = actions[row_valid]
        scores = scores[row_valid]
        masks = masks[row_valid]
        logits = logits[row_valid]
        values = values[row_valid]
        log_prob = log_prob[row_valid]
        valid = valid[row_valid]
        obs_dict = {"observation": observations[row_valid], "action_mask": masks}
    log_prob = log_prob.masked_fill(~valid, 0.0)
    safe_scores, best, margin, confidence = planner_confidence(
        scores, valid, minimum_margin, margin_scale
    )
    target_logits = (scores / temperature).masked_fill(~valid, float("-inf"))
    targets = target_logits.softmax(dim=1)

    # The first dataset treated every planner ordering as equally trustworthy. Its
    # actor rank accuracy stayed flat because near-ties contributed as much gradient
    # as decisive tactical differences. Train primarily on best-vs-alternative pairs,
    # weight each pair by its planner score gap, and ignore rows whose top two choices
    # are effectively tied.
    best_score = safe_scores.gather(1, best[:, None]).squeeze(1)
    best_log_prob = log_prob.gather(1, best[:, None]).squeeze(1)

    gaps = (best_score[:, None] - safe_scores).clamp_min(0.0)
    pair_mask = valid.clone()
    pair_mask.scatter_(1, best[:, None], False)
    pair_weights = (gaps / max(margin_scale, 1e-6)).clamp(0.0, 1.0)
    pair_weights = pair_weights.masked_fill(~pair_mask, 0.0)
    pair_losses = F.softplus(-(best_log_prob[:, None] - log_prob))
    pairwise = (pair_losses * pair_weights).sum(dim=1) / pair_weights.sum(
        dim=1
    ).clamp_min(1e-6)
    listwise = -(targets * log_prob).sum(dim=1)
    actor_per_row = pairwise_weight * pairwise + listwise_weight * listwise
    confidence_total = confidence.sum()
    if float(confidence_total.detach()) > 0:
        actor_loss = (actor_per_row * confidence).sum() / confidence_total
    else:
        actor_loss = actor_per_row.sum() * 0.0

    value_targets = safe_scores.max(dim=1).values
    value_loss = F.mse_loss(values.squeeze(-1), value_targets)
    anchor = _anchor_kl(policy, reference, logits, obs_dict, actions, valid)
    loss = actor_loss + value_coef * value_loss + anchor_weight * anchor

    predicted = log_prob.masked_fill(~valid, float("-inf")).argmax(dim=1)
    desired = safe_scores.argmax(dim=1)
    accuracy = (predicted == desired).float().mean()
    correct = (predicted == desired).float()
    confident_accuracy = (
        (correct * confidence).sum() / confidence_total
        if float(confidence_total.detach()) > 0
        else torch.zeros((), device=device)
    )
    confidence_fraction = (confidence > 0).float().mean()
    return (
        loss,
        actor_loss,
        value_loss,
        accuracy,
        confident_accuracy,
        margin.mean(),
        anchor,
        valid_fraction,
        confidence_fraction,
    )


@torch.no_grad()
def evaluate(
    policy,
    reference,
    loader,
    device,
    temperature,
    value_coef,
    anchor_weight,
    minimum_margin,
    margin_scale,
    pairwise_weight,
    listwise_weight,
):
    policy.eval()
    totals = np.zeros(9, dtype=np.float64)
    count = 0
    for batch in loader:
        metrics = loss_for_batch(
            policy,
            reference,
            batch,
            device,
            temperature,
            value_coef,
            anchor_weight,
            minimum_margin,
            margin_scale,
            pairwise_weight,
            listwise_weight,
        )
        size = len(batch[0])
        totals += np.asarray([float(metric) for metric in metrics]) * size
        count += size
    return totals / max(1, count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        nargs="+",
        default=["counterfactual_data"],
        help="one or more aggregation-round directories",
    )
    parser.add_argument("--checkpoint", default="results_repaired/champion.zip")
    parser.add_argument("--output", default="results_counterfactual/model.zip")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--anchor-weight", type=float, default=0.20)
    parser.add_argument("--minimum-margin", type=float, default=0.005)
    parser.add_argument("--margin-scale", type=float, default=0.10)
    parser.add_argument("--pairwise-weight", type=float, default=0.75)
    parser.add_argument("--listwise-weight", type=float, default=0.25)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--include-truncated", action="store_true")
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = CounterfactualDataset(
        [Path(directory) for directory in args.data], args.include_truncated
    )
    groups = sorted(set(dataset.groups))
    random.shuffle(groups)
    validation_groups = max(1, round(len(groups) * args.validation_fraction))
    if len(groups) == 1:
        train_indices = validation_indices = list(range(len(dataset)))
    else:
        validation_groups = min(validation_groups, len(groups) - 1)
        held_out = set(groups[:validation_groups])
        validation_indices = [
            index for index, group in enumerate(dataset.groups) if group in held_out
        ]
        train_indices = [
            index for index, group in enumerate(dataset.groups) if group not in held_out
        ]
    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices), batch_size=args.batch_size
    )

    model = PPO.load(args.checkpoint, device=args.device)
    policy = model.policy
    reference = copy.deepcopy(policy).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    device = policy.device
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    history = []

    baseline = evaluate(
        policy,
        reference,
        validation_loader,
        device,
        args.temperature,
        args.value_coef,
        args.anchor_weight,
        args.minimum_margin,
        args.margin_scale,
        args.pairwise_weight,
        args.listwise_weight,
    )
    best_key = (float(baseline[4]), float(baseline[3]), -float(baseline[0]))
    selected_epoch = 0
    model.save(output)
    print(
        f"positions={len(dataset)} train={len(train_indices)} "
        f"validation={len(validation_indices)}; "
        f"baseline rank_accuracy={baseline[3]:.3f} "
        f"confident_rank={baseline[4]:.3f} value_mse={baseline[2]:.4f}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        policy.train()
        running = np.zeros(9, dtype=np.float64)
        seen = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            metrics = loss_for_batch(
                policy,
                reference,
                batch,
                device,
                args.temperature,
                args.value_coef,
                args.anchor_weight,
                args.minimum_margin,
                args.margin_scale,
                args.pairwise_weight,
                args.listwise_weight,
            )
            metrics[0].backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            size = len(batch[0])
            running += np.asarray([float(metric.detach()) for metric in metrics]) * size
            seen += size
        train_metrics = running / max(1, seen)
        validation = evaluate(
            policy,
            reference,
            validation_loader,
            device,
            args.temperature,
            args.value_coef,
            args.anchor_weight,
            args.minimum_margin,
            args.margin_scale,
            args.pairwise_weight,
            args.listwise_weight,
        )
        record = {
            "epoch": epoch,
            "train_loss": float(train_metrics[0]),
            "train_rank_accuracy": float(train_metrics[3]),
            "validation_loss": float(validation[0]),
            "validation_rank_accuracy": float(validation[3]),
            "validation_confident_rank_accuracy": float(validation[4]),
            "validation_value_mse": float(validation[2]),
            "validation_mean_margin": float(validation[5]),
            "validation_anchor_kl": float(validation[6]),
            "validation_candidate_coverage": float(validation[7]),
            "validation_confident_fraction": float(validation[8]),
        }
        history.append(record)
        print(
            f"epoch {epoch}: train_loss={train_metrics[0]:.4f} "
            f"train_rank={train_metrics[3]:.3f} val_loss={validation[0]:.4f} "
            f"val_rank={validation[3]:.3f} "
            f"val_confident_rank={validation[4]:.3f} "
            f"val_value_mse={validation[2]:.4f}",
            f"candidate_coverage={validation[7]:.3f} "
            f"confident_fraction={validation[8]:.3f}",
            flush=True,
        )
        epoch_path = output.with_name(f"{output.stem}_epoch{epoch}{output.suffix}")
        model.save(epoch_path)
        key = (
            float(validation[4]),
            float(validation[3]),
            -float(validation[0]),
        )
        if key > best_key:
            best_key = key
            selected_epoch = epoch
            model.save(output)
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "positions": len(dataset),
                "games": len(groups),
                "selected_epoch": selected_epoch,
                "anchor_weight": args.anchor_weight,
                "minimum_margin": args.minimum_margin,
                "margin_scale": args.margin_scale,
                "pairwise_weight": args.pairwise_weight,
                "listwise_weight": args.listwise_weight,
                "baseline": {
                    "rank_accuracy": float(baseline[3]),
                    "confident_rank_accuracy": float(baseline[4]),
                    "value_mse": float(baseline[2]),
                },
                "history": history,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved {output} and {metrics_path}")


if __name__ == "__main__":
    main()
