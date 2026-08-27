"""Train and evaluate a fixed-team Team Preview terminal-outcome ranker."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vgc_bench.src.preview_outcome import PreviewOutcomeNet
from vgc_bench.src.set_particles import team_roster


def _metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return {
        "brier": float(np.mean((probabilities - targets) ** 2)),
        "log_loss": float(
            -np.mean(targets * np.log(clipped) + (1 - targets) * np.log(1 - clipped))
        ),
        "accuracy": float(np.mean((probabilities >= 0.5) == targets)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/preview_outcome_examples.jsonl")
    parser.add_argument("--our-team", default="teams/reg_mb/our_team.txt")
    parser.add_argument("--vocab-model", default="data/opponent_preview_top500_regmb.pt")
    parser.add_argument("--output", default="data/preview_outcome_regmb.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines()]
    payload = torch.load(args.vocab_model, map_location="cpu", weights_only=False)
    vocab = dict(payload["vocab"])
    ours = tuple(slot.species for slot in team_roster(Path(args.our_team).read_text()))
    opponents = sorted({row["opponent"] for row in rows})
    random.Random(args.seed).shuffle(opponents)
    train_end = round(0.8 * len(opponents))
    validation_end = round(0.9 * len(opponents))
    split_by_opponent = {
        opponent: "train" if index < train_end else "validation" if index < validation_end else "test"
        for index, opponent in enumerate(opponents)
    }
    device = torch.device(args.device)

    def tensors(selected):
        our_ids, opponent_ids, lead_masks, bring_masks, targets = [], [], [], [], []
        encoded_ours = [vocab.get(species, 0) for species in ours]
        for row in selected:
            our_ids.append(encoded_ours)
            opponent_ids.append([vocab.get(species, 0) for species in row["opponent_roster"]])
            lead = [0.0] * 6
            bring = [0.0] * 6
            for index in row["lead"]:
                lead[index] = 1.0
            for index in row["bring"]:
                bring[index] = 1.0
            lead_masks.append(lead)
            bring_masks.append(bring)
            targets.append(float(row["target"]))
        return tuple(
            torch.tensor(value, device=device, dtype=dtype)
            for value, dtype in (
                (our_ids, torch.long),
                (opponent_ids, torch.long),
                (lead_masks, torch.float32),
                (bring_masks, torch.float32),
                (targets, torch.float32),
            )
        )

    split_rows = {
        split: [row for row in rows if split_by_opponent[row["opponent"]] == split]
        for split in ("train", "validation", "test")
    }
    split_tensors = {split: tensors(selected) for split, selected in split_rows.items()}
    model = PreviewOutcomeNet(len(vocab), embed_dim=48, hidden_dim=160).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    train_targets = split_tensors["train"][-1]
    positives = float(train_targets.sum())
    negative = len(train_targets) - positives
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative / max(1.0, positives), device=device)
    )
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    best = None
    best_loss = math.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_targets), generator=rng)
        for start in range(0, len(order), args.batch_size):
            indexes = order[start : start + args.batch_size].to(device)
            batch = [tensor[indexes] for tensor in split_tensors["train"]]
            optimizer.zero_grad(set_to_none=True)
            logits = model(*batch[:-1])
            loss = loss_fn(logits, batch[-1])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = split_tensors["validation"]
            probabilities = torch.sigmoid(model(*validation[:-1])).cpu().numpy()
        metrics = _metrics(validation[-1].cpu().numpy(), probabilities)
        history.append({"epoch": epoch, **metrics})
        if metrics["log_loss"] < best_loss:
            best_loss = metrics["log_loss"]
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best is not None
    model.load_state_dict(best)
    model.eval()
    metrics = {}
    with torch.no_grad():
        for split, values in split_tensors.items():
            probabilities = torch.sigmoid(model(*values[:-1])).cpu().numpy()
            metrics[split] = _metrics(values[-1].cpu().numpy(), probabilities)
            metrics[split]["rows"] = len(values[-1])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best,
            "vocab": vocab,
            "our_roster": ours,
            "config": {"embed_dim": 48, "hidden_dim": 160},
            "metrics": metrics,
            "history": history,
            "provenance": {"data": str(Path(args.data).resolve()), "seed": args.seed},
        },
        output,
    )
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
