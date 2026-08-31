"""Train the opponent/team-preview model on top-player Reg M-B replays."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import random
import zlib
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from vgc_bench.src.opponent_preview import (
    BRING_INDICES,
    PAIR_INDICES,
    PreviewExample,
    PreviewNet,
    load_replay_examples,
)


class PreviewDataset(Dataset):
    def __init__(self, examples: list[PreviewExample], vocab: dict[str, int]):
        self.examples = examples
        self.vocab = vocab

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        roster = torch.tensor([self.vocab[s] for s in example.roster])
        opponent = torch.tensor([self.vocab[s] for s in example.opponent_roster])
        lead = PAIR_INDICES.index(example.lead)
        bring = -1 if example.bring is None else BRING_INDICES.index(example.bring)
        return roster, opponent, lead, bring


def split_examples(examples: list[PreviewExample]):
    train, valid = [], []
    for example in examples:
        # Both sides of one battle stay in the same split.
        bucket = zlib.crc32(example.battle_id.encode()) % 10
        (valid if bucket < 2 else train).append(example)
    return train, valid


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    lead_ok = lead_top3 = bring_ok = bring_top3 = bring_n = total = 0
    for roster, opponent, lead, bring in loader:
        roster, opponent = roster.to(device), opponent.to(device)
        lead, bring = lead.to(device), bring.to(device)
        lead_logits, bring_logits = model(roster, opponent)
        total += len(lead)
        lead_ok += int((lead_logits.argmax(1) == lead).sum())
        lead_top3 += int(
            (lead_logits.topk(3, dim=1).indices == lead[:, None]).any(1).sum()
        )
        known = bring >= 0
        if known.any():
            bring_n += int(known.sum())
            bring_ok += int((bring_logits[known].argmax(1) == bring[known]).sum())
            bring_top3 += int(
                (bring_logits[known].topk(3, dim=1).indices == bring[known, None])
                .any(1)
                .sum()
            )
    return {
        "lead_top1": lead_ok / max(total, 1),
        "lead_top3": lead_top3 / max(total, 1),
        "bring_top1": bring_ok / max(bring_n, 1),
        "bring_top3": bring_top3 / max(bring_n, 1),
        "validation_examples": total,
        "validation_bring_labels": bring_n,
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
        "--output", type=Path, default=Path("data/opponent_preview_regmb.pt")
    )
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-3)
    parser.add_argument("--embed_dim", type=int, default=48)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--top_500_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="retain only sides at or above the source ladder's top-500 Elo floor",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    examples = load_replay_examples(args.logs, top_500_only=args.top_500_only)
    train_examples, valid_examples = split_examples(examples)
    species = sorted(
        {
            name
            for example in examples
            for name in (*example.roster, *example.opponent_roster)
        }
    )
    vocab = {"<unknown>": 0, **{name: i + 1 for i, name in enumerate(species)}}
    print(
        f"examples={len(examples)} train={len(train_examples)} "
        f"valid={len(valid_examples)} exact_bring="
        f"{sum(e.bring is not None for e in examples)} species={len(species)}",
        flush=True,
    )

    train_loader = DataLoader(
        PreviewDataset(train_examples, vocab), batch_size=args.batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        PreviewDataset(valid_examples, vocab), batch_size=args.batch_size
    )
    device = torch.device(args.device)
    model = PreviewNet(len(vocab), args.embed_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_metrics = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = batches = 0.0
        for roster, opponent, lead, bring in train_loader:
            roster, opponent = roster.to(device), opponent.to(device)
            lead, bring = lead.to(device), bring.to(device)
            lead_logits, bring_logits = model(roster, opponent)
            loss = criterion(lead_logits, lead)
            known = bring >= 0
            if known.any():
                loss = loss + criterion(bring_logits[known], bring[known])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach())
            batches += 1
        metrics = evaluate(model, valid_loader, device)
        score = metrics["lead_top1"] + metrics["bring_top1"]
        if best_metrics is None or score > (
            best_metrics["lead_top1"] + best_metrics["bring_top1"]
        ):
            best_metrics = metrics
            best_state = {
                k: value.detach().cpu() for k, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:02d} loss={running / max(batches, 1):.4f} "
                f"lead@1={metrics['lead_top1']:.3f} lead@3={metrics['lead_top3']:.3f} "
                f"bring@1={metrics['bring_top1']:.3f} "
                f"bring@3={metrics['bring_top3']:.3f}",
                flush=True,
            )

    assert best_state is not None and best_metrics is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "vocab": vocab,
            "config": {"embed_dim": args.embed_dim, "hidden_dim": args.hidden_dim},
            "metrics": {
                **best_metrics,
                "training_examples": len(train_examples),
                "exact_bring_examples": sum(e.bring is not None for e in examples),
                "source_logs": [str(path) for path in args.logs],
                "top_500_only": args.top_500_only,
            },
        },
        args.output,
    )
    print(f"saved {args.output}: {best_metrics}")


if __name__ == "__main__":
    main()
