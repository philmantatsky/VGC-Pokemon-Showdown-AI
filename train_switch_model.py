"""Train the high-rated opponent voluntary-switch model."""

from __future__ import annotations

import argparse
import random
import zlib
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from vgc_bench.src.opponent_preview import top_500_rating_floor
from vgc_bench.src.opponent_tactics import (
    SwitchExample,
    SwitchNet,
    load_switch_examples,
)


class SwitchDataset(Dataset):
    def __init__(self, examples: list[SwitchExample], vocab: dict[str, int]):
        self.examples = examples
        self.vocab = vocab

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        def encode(names):
            return torch.tensor([self.vocab[name] for name in names])

        target = (
            -1 if example.switch_to is None else example.roster.index(example.switch_to)
        )
        return (
            encode(example.roster),
            encode(example.opponent_roster),
            encode(example.active),
            encode(example.opponent_active),
            torch.tensor(example.hp, dtype=torch.float32),
            example.actor_slot,
            example.turn,
            target,
        )


def split_examples(examples: list[SwitchExample]):
    train, valid = [], []
    for example in examples:
        bucket = zlib.crc32(example.battle_id.encode()) % 10
        (valid if bucket < 2 else train).append(example)
    return train, valid


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    scores = []
    labels = []
    target1 = target3 = switch_n = 0
    for batch in loader:
        *inputs, target = batch
        inputs = [value.to(device) for value in inputs]
        target = target.to(device)
        switch_logit, target_logits = model(*inputs)
        switched = target >= 0
        scores.extend(torch.sigmoid(switch_logit).cpu().tolist())
        labels.extend(switched.int().cpu().tolist())
        if switched.any():
            switch_n += int(switched.sum())
            target1 += int(
                (target_logits[switched].argmax(1) == target[switched]).sum()
            )
            target3 += int(
                (
                    target_logits[switched].topk(3, dim=1).indices
                    == target[switched, None]
                )
                .any(1)
                .sum()
            )
    predicted = [score >= 0.5 for score in scores]
    tp = sum(p and y for p, y in zip(predicted, labels))
    fp = sum(p and not y for p, y in zip(predicted, labels))
    fn = sum(not p and y for p, y in zip(predicted, labels))
    correct = sum(p == bool(y) for p, y in zip(predicted, labels))
    brier = sum((score - y) ** 2 for score, y in zip(scores, labels)) / max(
        len(labels), 1
    )
    # Rank-based ROC AUC with average ranks for ties (Mann-Whitney U). Unlike a
    # fixed 0.5 threshold, this measures whether likely switches are actually ranked
    # above stays when the calibrated base rate is only about ten percent.
    ranked = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        positive_rank_sum += average_rank * sum(y for _, y in ranked[index:end])
        index = end
    positives = sum(labels)
    negatives = len(labels) - positives
    auc = (positive_rank_sum - positives * (positives + 1) / 2) / max(
        positives * negatives, 1
    )
    budget = max(1, positives)
    top_budget = sorted(zip(scores, labels), reverse=True)[:budget]
    budget_tp = sum(label for _, label in top_budget)
    return {
        "switch_rate": sum(labels) / max(len(labels), 1),
        "switch_accuracy": correct / max(len(labels), 1),
        "switch_precision": tp / max(tp + fp, 1),
        "switch_recall": tp / max(tp + fn, 1),
        "switch_brier": brier,
        "switch_auc": auc,
        "precision_at_base_rate": budget_tp / budget,
        "recall_at_base_rate": budget_tp / max(positives, 1),
        "target_top1": target1 / max(switch_n, 1),
        "target_top3": target3 / max(switch_n, 1),
        "validation_examples": len(labels),
        "validation_switches": switch_n,
    }


@torch.no_grad()
def fit_isotonic_calibration(model, loader, device, initial_bins: int = 20):
    """Pool-adjacent-violators calibration over equal-frequency held-out bins."""
    model.eval()
    values = []
    for batch in loader:
        *inputs, target = batch
        inputs = [value.to(device) for value in inputs]
        switch_logit, _ = model(*inputs)
        values.extend(
            zip(
                torch.sigmoid(switch_logit).cpu().tolist(), (target >= 0).int().tolist()
            )
        )
    values.sort(key=lambda item: item[0])
    chunk = max(1, len(values) // initial_bins)
    blocks = []
    for start in range(0, len(values), chunk):
        group = values[start : start + chunk]
        blocks.append(
            {
                "n": len(group),
                "positive": sum(label for _, label in group),
                "upper": group[-1][0],
            }
        )
        while len(blocks) >= 2:
            left, right = blocks[-2:]
            if left["positive"] / left["n"] <= right["positive"] / right["n"]:
                break
            blocks[-2:] = [
                {
                    "n": left["n"] + right["n"],
                    "positive": left["positive"] + right["positive"],
                    "upper": right["upper"],
                }
            ]
    return {
        "upper_bounds": [float(block["upper"]) for block in blocks],
        # Beta(1,1) smoothing avoids exact zero/one from a small terminal block.
        "rates": [
            float((block["positive"] + 1) / (block["n"] + 2)) for block in blocks
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs",
        type=Path,
        nargs="+",
        default=sorted(Path("battle_logs_top").glob("*regmb*.json")),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/opponent_switch_top500_regmb.pt")
    )
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument(
        "--pretrain_epochs",
        type=int,
        default=8,
        help="representation pretraining on all replay sides before top-500 tuning",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--embed_dim", type=int, default=48)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    all_examples = load_switch_examples(args.logs, top_500_only=False)
    examples = [
        example
        for example in all_examples
        if example.rating is not None
        and example.rating >= top_500_rating_floor(example.battle_id)
    ]
    train_examples, valid_examples = split_examples(examples)
    pretrain_examples, _ = split_examples(all_examples)
    species = sorted(
        {
            species
            for example in all_examples
            for species in (*example.roster, *example.opponent_roster)
        }
    )
    vocab = {"<unknown>": 0, **{name: i + 1 for i, name in enumerate(species)}}
    print(
        f"examples={len(examples)} train={len(train_examples)} "
        f"valid={len(valid_examples)} switches="
        f"{sum(e.switch_to is not None for e in examples)} "
        f"pretrain={len(pretrain_examples)} species={len(species)}",
        flush=True,
    )
    train_loader = DataLoader(
        SwitchDataset(train_examples, vocab), batch_size=args.batch_size, shuffle=True
    )
    pretrain_loader = DataLoader(
        SwitchDataset(pretrain_examples, vocab),
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        SwitchDataset(valid_examples, vocab), batch_size=args.batch_size
    )
    device = torch.device(args.device)
    model = SwitchNet(len(vocab), args.embed_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    binary_loss = nn.BCEWithLogitsLoss()
    target_loss = nn.CrossEntropyLoss()

    def train_epoch(loader):
        model.train()
        running = batches = 0.0
        for batch in loader:
            *inputs, target = batch
            inputs = [value.to(device) for value in inputs]
            target = target.to(device)
            switch_logit, target_logits = model(*inputs)
            switched = target >= 0
            loss = binary_loss(switch_logit, switched.float())
            if switched.any():
                loss = loss + target_loss(target_logits[switched], target[switched])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach())
            batches += 1
        return running / max(batches, 1)

    for epoch in range(1, args.pretrain_epochs + 1):
        loss = train_epoch(pretrain_loader)
        if epoch == 1 or epoch == args.pretrain_epochs:
            metrics = evaluate(model, valid_loader, device)
            print(
                f"pretrain={epoch:02d} loss={loss:.4f} "
                f"top500_auc={metrics['switch_auc']:.3f} "
                f"target@1={metrics['target_top1']:.3f}",
                flush=True,
            )

    best_metrics = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(train_loader)
        metrics = evaluate(model, valid_loader, device)
        score = metrics["switch_auc"] + metrics["target_top1"]
        if best_metrics is None or score > (
            best_metrics["switch_auc"] + best_metrics["target_top1"]
        ):
            best_metrics = metrics
            best_state = {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:02d} loss={loss:.4f} "
                f"switch p/r={metrics['switch_precision']:.3f}/"
                f"{metrics['switch_recall']:.3f} "
                f"auc={metrics['switch_auc']:.3f} "
                f"p@base={metrics['precision_at_base_rate']:.3f} "
                f"target@1/3={metrics['target_top1']:.3f}/"
                f"{metrics['target_top3']:.3f}",
                flush=True,
            )

    assert best_state is not None and best_metrics is not None
    model.load_state_dict(best_state)
    calibration = fit_isotonic_calibration(model, valid_loader, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "vocab": vocab,
            "config": {"embed_dim": args.embed_dim, "hidden_dim": args.hidden_dim},
            "calibration": calibration,
            "metrics": {
                **best_metrics,
                "training_examples": len(train_examples),
                "source_logs": [str(path) for path in args.logs],
                "top_500_only": True,
                "pretraining_examples": len(pretrain_examples),
                "pretrain_epochs": args.pretrain_epochs,
            },
        },
        args.output,
    )
    print(f"saved {args.output}: {best_metrics}")


if __name__ == "__main__":
    main()
