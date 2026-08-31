"""Distill planned bring-four/lead-two rankings into the preview network."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import copy
import json
import random
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from poke_env.data import to_id_str
from torch.utils.data import DataLoader, Dataset, Subset

from vgc_bench.src.opponent_preview import BRING_INDICES, PAIR_INDICES, PreviewNet

PAIR_LOOKUP = {indices: index for index, indices in enumerate(PAIR_INDICES)}
BRING_LOOKUP = {indices: index for index, indices in enumerate(BRING_INDICES)}


def _plan_indices(choice: str) -> tuple[int, int]:
    positions = [
        int(value.strip()) - 1 for value in choice.removeprefix("team ").split(",")
    ]
    if len(positions) != 4:
        raise ValueError(f"expected four preview positions, got {choice!r}")
    lead = cast(tuple[int, int], tuple(sorted((positions[0], positions[1]))))
    bring = cast(
        tuple[int, int, int, int],
        tuple(sorted((positions[0], positions[1], positions[2], positions[3]))),
    )
    return PAIR_LOOKUP[lead], BRING_LOOKUP[bring]


class PreviewCounterfactualDataset(Dataset):
    def __init__(self, path: Path, vocab: dict[str, int]):
        self.rows = []
        seen_games = set()
        for line in path.read_text().splitlines():
            record = json.loads(line)
            game = record.get("game")
            if game in seen_games:
                continue
            seen_games.add(game)
            candidates = []
            for ranking in record["rankings"]:
                try:
                    lead, bring = _plan_indices(ranking["choice"])
                except (KeyError, ValueError):
                    continue
                candidates.append((lead, bring, float(ranking["score"])))
            if len(candidates) < 2:
                continue
            ours = [vocab.get(to_id_str(name), 0) for name in record["our_roster"]]
            theirs = [
                vocab.get(to_id_str(name), 0) for name in record["opponent_roster"]
            ]
            self.rows.append((ours, theirs, candidates))
        if not self.rows:
            raise ValueError(f"no usable preview rankings in {path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        ours, theirs, candidates = self.rows[index]
        return (
            torch.tensor(ours, dtype=torch.long),
            torch.tensor(theirs, dtype=torch.long),
            torch.tensor(candidates, dtype=torch.float32),
        )


def _loss(model, reference, batch, temperature: float, anchor_weight: float):
    ours, theirs, candidates = batch
    lead_logits, bring_logits = model(ours, theirs)
    lead_index = candidates[:, :, 0].long()
    bring_index = candidates[:, :, 1].long()
    scores = candidates[:, :, 2]
    candidate_logits = lead_logits.gather(1, lead_index) + bring_logits.gather(
        1, bring_index
    )
    targets = (scores / temperature).softmax(dim=1)
    ranking_loss = -(targets * candidate_logits.log_softmax(dim=1)).sum(dim=1).mean()
    with torch.no_grad():
        ref_lead, ref_bring = reference(ours, theirs)
    anchor = F.kl_div(
        lead_logits.log_softmax(dim=1), ref_lead.softmax(dim=1), reduction="batchmean"
    ) + F.kl_div(
        bring_logits.log_softmax(dim=1), ref_bring.softmax(dim=1), reduction="batchmean"
    )
    predicted = candidate_logits.argmax(dim=1)
    desired = scores.argmax(dim=1)
    accuracy = (predicted == desired).float().mean()
    return ranking_loss + anchor_weight * anchor, ranking_loss, anchor, accuracy


@torch.no_grad()
def _evaluate(model, reference, loader, temperature, anchor_weight):
    model.eval()
    totals = torch.zeros(4)
    count = 0
    for batch in loader:
        batch = tuple(tensor.to(next(model.parameters()).device) for tensor in batch)
        metrics = _loss(model, reference, batch, temperature, anchor_weight)
        size = len(batch[0])
        totals += torch.tensor([float(metric) for metric in metrics]) * size
        count += size
    return totals / max(1, count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="counterfactual_data/preview_rankings.jsonl")
    parser.add_argument("--checkpoint", default="data/opponent_preview_top500_regmb.pt")
    parser.add_argument("--output", default="data/preview_counterfactual_regmb.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--anchor-weight", type=float, default=0.20)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Loading an entire payload directly onto MPS can leave embedding storage as an
    # unallocated placeholder. Materialize it on CPU, then move the constructed model.
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = PreviewNet(
        len(payload["vocab"]), int(config["embed_dim"]), int(config["hidden_dim"])
    ).to(args.device)
    model.load_state_dict(payload["state_dict"])
    reference = copy.deepcopy(model).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    dataset = PreviewCounterfactualDataset(Path(args.data), payload["vocab"])
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    validation_size = max(1, round(len(indices) * args.validation_fraction))
    if len(indices) == 1:
        train_indices = validation_indices = indices
    else:
        validation_size = min(validation_size, len(indices) - 1)
        validation_indices = indices[:validation_size]
        train_indices = indices[validation_size:]
    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices), batch_size=args.batch_size
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history = []
    baseline = _evaluate(
        model, reference, validation_loader, args.temperature, args.anchor_weight
    )
    print(
        f"previews={len(dataset)} baseline_rank_accuracy={baseline[3]:.3f}", flush=True
    )
    best_key = (float(baseline[3]), -float(baseline[0]))
    best_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.state_dict().items()
    }
    selected_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = tuple(tensor.to(args.device) for tensor in batch)
            optimizer.zero_grad(set_to_none=True)
            loss, _ranking, _anchor, _accuracy = _loss(
                model, reference, batch, args.temperature, args.anchor_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        validation = _evaluate(
            model, reference, validation_loader, args.temperature, args.anchor_weight
        )
        record = {
            "epoch": epoch,
            "validation_loss": float(validation[0]),
            "validation_rank_accuracy": float(validation[3]),
            "anchor_kl": float(validation[2]),
        }
        history.append(record)
        print(
            f"epoch {epoch}: val_loss={validation[0]:.4f} "
            f"val_rank={validation[3]:.3f} anchor_kl={validation[2]:.5f}",
            flush=True,
        )
        key = (float(validation[3]), -float(validation[0]))
        if key > best_key:
            best_key = key
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
            selected_epoch = epoch

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    metrics = dict(payload.get("metrics") or {})
    metrics["counterfactual_distillation"] = {
        "source": args.data,
        "previews": len(dataset),
        "baseline_rank_accuracy": float(baseline[3]),
        "selected_epoch": selected_epoch,
        "history": history,
    }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": payload["vocab"],
            "config": config,
            "metrics": metrics,
        },
        output,
    )
    print(f"saved {output}")


if __name__ == "__main__":
    main()
