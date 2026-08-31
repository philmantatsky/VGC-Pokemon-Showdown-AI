"""Train a conservative planner residual while keeping the champion frozen."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, Subset

from training.train_counterfactual import (
    CounterfactualDataset,
    joint_log_prob,
    planner_confidence,
)
from vgc_bench.src.residual_ranker import (
    ResidualConfig,
    ResidualJointRanker,
    candidate_semantic_dim,
    candidate_semantic_features,
    champion_features,
)


def _split(dataset, seed: int, validation_fraction: float):
    groups = sorted(set(dataset.groups))
    random.Random(seed).shuffle(groups)
    validation_count = max(
        1, min(len(groups) - 1, round(len(groups) * validation_fraction))
    )
    held_out = set(groups[:validation_count])
    validation = [
        index for index, group in enumerate(dataset.groups) if group in held_out
    ]
    train = [
        index for index, group in enumerate(dataset.groups) if group not in held_out
    ]
    if not train or not validation:
        raise ValueError("residual split requires at least two trajectory groups")
    return train, validation


def _batch_metrics(
    ranker,
    champion,
    batch,
    device,
    minimum_margin,
    margin_scale,
    confidence_weight,
    residual_weight,
):
    observations, masks, actions, scores = [tensor.to(device) for tensor in batch]
    present = (actions >= 0).all(dim=2) & torch.isfinite(scores)
    obs_dict = {"observation": observations, "action_mask": masks}
    with torch.no_grad():
        logits, _values = champion.get_logits(obs_dict, actor_grad=False)
        champion_log = joint_log_prob(champion, logits, masks, actions)
        features = champion_features(champion, obs_dict)
        action_features = candidate_semantic_features(
            champion, obs_dict, actions
        )
    valid = present & torch.isfinite(champion_log)
    row_valid = valid.sum(dim=1) >= 2
    if not row_valid.any():
        raise RuntimeError("residual batch has no rows with two legal candidates")
    actions = actions[row_valid]
    scores = scores[row_valid]
    valid = valid[row_valid]
    champion_log = champion_log[row_valid]
    features = features[row_valid]
    action_features = action_features[row_valid]
    clean_log = torch.where(
        valid, champion_log.clamp_min(-30.0), torch.full_like(champion_log, -30.0)
    )
    adjusted, confidence, residual = ranker(
        features, actions, clean_log, action_features, valid
    )
    adjusted = adjusted.masked_fill(~valid, float("-inf"))
    safe_scores, best, margin, target_confidence = planner_confidence(
        scores, valid, minimum_margin, margin_scale
    )
    best_score = safe_scores.gather(1, best[:, None]).squeeze(1)
    best_adjusted = adjusted.gather(1, best[:, None]).squeeze(1)
    gaps = (best_score[:, None] - safe_scores).clamp_min(0.0)
    pair_mask = valid.clone()
    pair_mask.scatter_(1, best[:, None], False)
    pair_weight = (gaps / max(margin_scale, 1e-6)).clamp(0.0, 1.0)
    pair_weight = pair_weight.masked_fill(~pair_mask, 0.0)
    pair_loss = F.softplus(-(best_adjusted[:, None] - adjusted))
    pair_loss = (pair_loss * pair_weight).sum(dim=1) / pair_weight.sum(
        dim=1
    ).clamp_min(1e-6)
    confident_rows = target_confidence > 0
    # Deployment is a top-choice decision. Pairwise gap loss alone fell steadily
    # while *both training and validation top-1 accuracy declined*, because it could
    # improve many low-ranked alternatives without placing the teacher's winner
    # first. Optimize the actual best action, retaining a smaller pairwise term for
    # useful ordering among alternatives. Planner margin still suppresses near ties.
    top_choice_loss = F.cross_entropy(adjusted, best, reduction="none")
    ranking_per_row = 0.75 * top_choice_loss + 0.25 * pair_loss
    ranking_loss = (
        (ranking_per_row * target_confidence).sum()
        / target_confidence.sum().clamp_min(1e-6)
        if confident_rows.any()
        else ranking_per_row.sum() * 0.0
    )
    champion_pick = champion_log.masked_fill(~valid, float("-inf")).argmax(dim=1)
    # Confidence means "the champion is decisively wrong here", not merely "the
    # planner produced a non-tied row". The old all-positive target drove the gate
    # from 0% to ~99% application in one epoch and broadly damaged good decisions.
    confidence_target = target_confidence * (champion_pick != best).float()
    confidence_loss = F.binary_cross_entropy(confidence, confidence_target)
    residual_penalty = (residual.masked_fill(~valid, 0.0) ** 2).sum() / valid.sum()
    loss = (
        ranking_loss
        + confidence_weight * confidence_loss
        + residual_weight * residual_penalty
    )

    desired = safe_scores.argmax(dim=1)
    adjusted_pick = adjusted.argmax(dim=1)
    applied = confidence >= ranker.config.confidence_threshold
    deployed_pick = torch.where(applied, adjusted_pick, champion_pick)
    accuracy = (deployed_pick == desired).float().mean()
    champion_accuracy = (champion_pick == desired).float().mean()
    adjusted_accuracy = (adjusted_pick == desired).float().mean()
    changed = (applied & (deployed_pick != champion_pick)).float().mean()
    return (
        loss,
        ranking_loss,
        confidence_loss,
        residual_penalty,
        accuracy,
        champion_accuracy,
        applied.float().mean(),
        changed,
        margin.mean(),
        adjusted_accuracy,
    )


@torch.no_grad()
def _evaluate(ranker, champion, loader, device, args):
    ranker.eval()
    totals = np.zeros(10, dtype=np.float64)
    count = 0
    for batch in loader:
        metrics = _batch_metrics(
            ranker,
            champion,
            batch,
            device,
            args.minimum_margin,
            args.margin_scale,
            args.confidence_weight,
            args.residual_weight,
        )
        size = len(batch[0])
        totals += np.asarray([float(metric) for metric in metrics]) * size
        count += size
    return totals / max(1, count)


@torch.no_grad()
def _calibrate_gate(
    ranker,
    champion,
    loader,
    device,
    max_changed_fraction: float,
    minimum_gain: float,
):
    """Choose a held-out confidence threshold for conservative deployment."""
    rows: list[tuple[float, int, int, int]] = []
    ranker.eval()
    for batch in loader:
        observations, masks, actions, scores = [tensor.to(device) for tensor in batch]
        present = (actions >= 0).all(dim=2) & torch.isfinite(scores)
        obs_dict = {"observation": observations, "action_mask": masks}
        logits, _values = champion.get_logits(obs_dict, actor_grad=False)
        champion_log = joint_log_prob(champion, logits, masks, actions)
        valid = present & torch.isfinite(champion_log)
        row_valid = valid.sum(dim=1) >= 2
        if not row_valid.any():
            continue
        obs_dict = {key: value[row_valid] for key, value in obs_dict.items()}
        actions = actions[row_valid]
        scores = scores[row_valid]
        valid = valid[row_valid]
        champion_log = champion_log[row_valid]
        clean_log = torch.where(
            valid,
            champion_log.clamp_min(-30.0),
            torch.full_like(champion_log, -30.0),
        )
        adjusted, confidence, _residual = ranker(
            champion_features(champion, obs_dict),
            actions,
            clean_log,
            candidate_semantic_features(champion, obs_dict, actions),
            valid,
        )
        desired = scores.masked_fill(~valid, float("-inf")).argmax(dim=1)
        champion_pick = champion_log.masked_fill(
            ~valid, float("-inf")
        ).argmax(dim=1)
        adjusted_pick = adjusted.masked_fill(~valid, float("-inf")).argmax(dim=1)
        rows.extend(
            zip(
                confidence.cpu().tolist(),
                champion_pick.cpu().tolist(),
                adjusted_pick.cpu().tolist(),
                desired.cpu().tolist(),
            )
        )
    if not rows:
        raise RuntimeError("cannot calibrate residual gate without validation rows")
    confidence = np.asarray([row[0] for row in rows], dtype=np.float64)
    champion_pick = np.asarray([row[1] for row in rows])
    adjusted_pick = np.asarray([row[2] for row in rows])
    desired = np.asarray([row[3] for row in rows])
    return _select_gate_threshold(
        confidence,
        champion_pick,
        adjusted_pick,
        desired,
        max_changed_fraction,
        minimum_gain,
    )


def _select_gate_threshold(
    confidence,
    champion_pick,
    adjusted_pick,
    desired,
    max_changed_fraction: float,
    minimum_gain: float,
):
    """Pure threshold selection used by validation and regression tests."""
    confidence = np.asarray(confidence, dtype=np.float64)
    champion_pick = np.asarray(champion_pick)
    adjusted_pick = np.asarray(adjusted_pick)
    desired = np.asarray(desired)
    baseline_accuracy = float(np.mean(champion_pick == desired))
    adjusted_accuracy = float(np.mean(adjusted_pick == desired))
    thresholds = np.unique(
        np.concatenate([confidence, [float(confidence.max()) + 1e-6]])
    )
    options = []
    for threshold in thresholds:
        applied = confidence >= threshold
        changed = applied & (adjusted_pick != champion_pick)
        changed_fraction = float(np.mean(changed))
        if changed_fraction > max_changed_fraction:
            continue
        deployed = np.where(applied, adjusted_pick, champion_pick)
        accuracy = float(np.mean(deployed == desired))
        options.append(
            (
                accuracy,
                -changed_fraction,
                float(threshold),
                float(np.mean(applied)),
                changed_fraction,
            )
        )
    best = max(options)
    accuracy, _negative_changed, threshold, applied_fraction, changed_fraction = best
    if accuracy < baseline_accuracy + minimum_gain:
        threshold = float(confidence.max()) + 1e-6
        accuracy = baseline_accuracy
        applied_fraction = 0.0
        changed_fraction = 0.0
    changed = (confidence >= threshold) & (adjusted_pick != champion_pick)
    return {
        "threshold": threshold,
        "deployed_rank_accuracy": accuracy,
        "champion_rank_accuracy": baseline_accuracy,
        "adjusted_rank_accuracy": adjusted_accuracy,
        "applied_fraction": applied_fraction,
        "changed_fraction": changed_fraction,
        "changed_count": int(changed.sum()),
        "changed_champion_accuracy": (
            float(np.mean(champion_pick[changed] == desired[changed]))
            if changed.any()
            else 0.0
        ),
        "changed_residual_accuracy": (
            float(np.mean(adjusted_pick[changed] == desired[changed]))
            if changed.any()
            else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        nargs="+",
        default=["counterfactual_data"],
        help="one or more aggregation-round directories",
    )
    parser.add_argument("--checkpoint", default="results_repaired/champion.zip")
    parser.add_argument("--output", default="results_residual_v1/residual.pt")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minimum-margin", type=float, default=0.02)
    parser.add_argument("--margin-scale", type=float, default=0.15)
    parser.add_argument("--confidence-weight", type=float, default=0.35)
    parser.add_argument("--residual-weight", type=float, default=0.02)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--minimum-validation-positions",
        type=int,
        default=0,
        help=(
            "hard-fail before training when the held-out split is smaller than "
            "this; the four rejected rounds validated on ~240-376 positions, "
            "far below what their 0.5pp rank-gain gate could resolve"
        ),
    )
    parser.add_argument("--max-changed-fraction", type=float, default=0.15)
    parser.add_argument("--minimum-calibration-gain", type=float, default=0.005)
    parser.add_argument("--include-truncated", action="store_true")
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = CounterfactualDataset(
        [Path(directory) for directory in args.data], args.include_truncated
    )
    train_indices, validation_indices = _split(
        dataset, args.seed, args.validation_fraction
    )
    if len(validation_indices) < args.minimum_validation_positions:
        raise SystemExit(
            f"held-out split has {len(validation_indices)} positions; "
            f"minimum is {args.minimum_validation_positions} -- generate more "
            "games before training (an under-powered validation set cannot "
            "support the rank-gain gate)"
        )
    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices), batch_size=args.batch_size
    )
    champion = PPO.load(args.checkpoint, device=args.device).policy.eval()
    for parameter in champion.parameters():
        parameter.requires_grad_(False)
    feature_dim = int(champion.features_extractor.features_dim)
    ranker = ResidualJointRanker(
        ResidualConfig(
            feature_dim=feature_dim,
            candidate_feature_dim=candidate_semantic_dim(champion),
            contextual_confidence=True,
        )
    ).to(champion.device)
    optimizer = torch.optim.AdamW(
        ranker.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_epochs: list[tuple[tuple[float, float, float], int, Path]] = []
    baseline = _evaluate(ranker, champion, validation_loader, champion.device, args)
    for epoch in range(1, args.epochs + 1):
        ranker.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            metrics = _batch_metrics(
                ranker,
                champion,
                batch,
                champion.device,
                args.minimum_margin,
                args.margin_scale,
                args.confidence_weight,
                args.residual_weight,
            )
            metrics[0].backward()
            torch.nn.utils.clip_grad_norm_(ranker.parameters(), 1.0)
            optimizer.step()
        validation = _evaluate(
            ranker, champion, validation_loader, champion.device, args
        )
        calibration = _calibrate_gate(
            ranker,
            champion,
            validation_loader,
            champion.device,
            args.max_changed_fraction,
            args.minimum_calibration_gain,
        )
        ranker.config = replace(
            ranker.config,
            confidence_threshold=calibration["threshold"],
        )
        record = {
            "epoch": epoch,
            "validation_loss": float(validation[0]),
            "validation_rank_accuracy": calibration["deployed_rank_accuracy"],
            "champion_rank_accuracy": float(validation[5]),
            "applied_fraction": calibration["applied_fraction"],
            "changed_fraction": calibration["changed_fraction"],
            "mean_margin": float(validation[8]),
            "validation_adjusted_rank_accuracy": float(validation[9]),
            "calibration": calibration,
        }
        history.append(record)
        epoch_path = output.with_name(f"{output.stem}_epoch{epoch}{output.suffix}")
        ranker.save(
            epoch_path,
            {"source_checkpoint": str(Path(args.checkpoint).resolve()), **record},
        )
        key = (
            record["validation_rank_accuracy"],
            record["validation_adjusted_rank_accuracy"],
            -record["validation_loss"],
        )
        best_epochs.append((key, epoch, epoch_path))
        print(
            f"epoch {epoch}: val_rank={record['validation_rank_accuracy']:.3f} "
            f"champion={validation[5]:.3f} "
            f"applied={record['applied_fraction']:.3f} "
            f"changed={record['changed_fraction']:.3f} "
            f"gate={calibration['threshold']:.3f}",
            flush=True,
        )
    best_epochs.sort(reverse=True)
    selected = ResidualJointRanker.load(best_epochs[0][2], champion.device)
    selected.save(
        output,
        {
            "source_checkpoint": str(Path(args.checkpoint).resolve()),
            "selected_epoch": best_epochs[0][1],
        },
    )
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "source_checkpoint": str(Path(args.checkpoint).resolve()),
                "positions": len(dataset),
                "train_positions": len(train_indices),
                "validation_positions": len(validation_indices),
                "baseline_rank_accuracy": float(baseline[5]),
                "selected_epoch": best_epochs[0][1],
                "top_three": [
                    {"epoch": epoch, "path": str(path), "key": list(key)}
                    for key, epoch, path in best_epochs[:3]
                ],
                "history": history,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved {output} and {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
