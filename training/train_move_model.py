"""Train a high-rated opponent move/target predictor with move semantics."""

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

from vgc_bench.src.opponent_tactics import MoveExample, MoveNet, load_move_examples


class MoveDataset(Dataset):
    def __init__(
        self,
        examples: list[MoveExample],
        species_vocab: dict[str, int],
        move_vocab: dict[str, int],
    ):
        self.examples = examples
        self.species_vocab = species_vocab
        self.move_vocab = move_vocab

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        def encode(names):
            return torch.tensor([self.species_vocab[name] for name in names])

        return (
            encode(example.roster),
            encode(example.opponent_roster),
            encode(example.active),
            encode(example.opponent_active),
            torch.tensor(example.hp, dtype=torch.float32),
            example.actor_slot,
            example.turn,
            self.move_vocab[example.move_id],
            example.target_class,
        )


def split_examples(examples: list[MoveExample]):
    train, valid = [], []
    for example in examples:
        bucket = zlib.crc32(example.battle_id.encode()) % 10
        (valid if bucket < 2 else train).append(example)
    return train, valid


def repertoire_mask(
    examples: list[MoveExample],
    species_vocab: dict[str, int],
    move_vocab: dict[str, int],
) -> torch.Tensor:
    mask = torch.zeros(len(species_vocab), len(move_vocab), dtype=torch.bool)
    for example in examples:
        actor = example.active[example.actor_slot]
        mask[species_vocab[actor], move_vocab[example.move_id]] = True
    return mask


@torch.no_grad()
def evaluate(model, loader, device, move_mask):
    model.eval()
    total = global1 = pool1 = pool3 = target1 = covered = 0
    for batch in loader:
        *inputs, move, target = batch
        inputs = [value.to(device) for value in inputs]
        move, target = move.to(device), target.to(device)
        move_logits, target_logits = model(*inputs)
        total += len(move)
        global1 += int((move_logits.argmax(1) == move).sum())
        rows = torch.arange(len(move), device=device)
        selected_target_logits = target_logits[rows, move]
        target1 += int((selected_target_logits.argmax(1) == target).sum())
        active, actor_slot = inputs[2], inputs[5]
        actor_species = active[rows, actor_slot]
        allowed = move_mask.to(device)[actor_species]
        is_covered = allowed[rows, move]
        covered += int(is_covered.sum())
        masked = move_logits.masked_fill(~allowed, -1e9)
        pool1 += int(((masked.argmax(1) == move) & is_covered).sum())
        pool3 += int(
            ((masked.topk(3, dim=1).indices == move[:, None]).any(1) & is_covered).sum()
        )
    return {
        "global_move_top1": global1 / max(total, 1),
        "repertoire_move_top1": pool1 / max(covered, 1),
        "repertoire_move_top3": pool3 / max(covered, 1),
        "repertoire_coverage": covered / max(total, 1),
        "target_top1": target1 / max(total, 1),
        "validation_examples": total,
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
        "--output", type=Path, default=Path("data/opponent_move_top500_regmb.pt")
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--embed_dim", type=int, default=48)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    all_examples = load_move_examples(args.logs, top_500_only=False)
    examples = load_move_examples(args.logs, top_500_only=True)
    train_examples, valid_examples = split_examples(examples)
    species = sorted(
        {
            species
            for example in all_examples
            for species in (*example.roster, *example.opponent_roster)
        }
    )
    move_ids = sorted({example.move_id for example in all_examples})
    species_vocab = {
        "<unknown>": 0,
        **{name: index + 1 for index, name in enumerate(species)},
    }
    move_vocab = {
        "<unknown>": 0,
        **{name: index + 1 for index, name in enumerate(move_ids)},
    }
    move_ids_by_index = [""] * len(move_vocab)
    for move_id, index in move_vocab.items():
        move_ids_by_index[index] = move_id
    mask = repertoire_mask(all_examples, species_vocab, move_vocab)
    print(
        f"examples={len(examples)} train={len(train_examples)} "
        f"valid={len(valid_examples)} all={len(all_examples)} "
        f"species={len(species)} moves={len(move_ids)}",
        flush=True,
    )
    train_loader = DataLoader(
        MoveDataset(train_examples, species_vocab, move_vocab),
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        MoveDataset(valid_examples, species_vocab, move_vocab),
        batch_size=args.batch_size,
    )
    device = torch.device(args.device)
    model = MoveNet(
        len(species_vocab), move_ids_by_index, args.embed_dim, args.hidden_dim
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_metrics = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = batches = 0.0
        for batch in train_loader:
            *inputs, move, target = batch
            inputs = [value.to(device) for value in inputs]
            move, target = move.to(device), target.to(device)
            move_logits, target_logits = model(*inputs)
            rows = torch.arange(len(move), device=device)
            selected_target_logits = target_logits[rows, move]
            loss = criterion(move_logits, move) + 0.5 * criterion(
                selected_target_logits, target
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach())
            batches += 1
        metrics = evaluate(model, valid_loader, device, mask)
        score = metrics["repertoire_move_top1"] + 0.25 * metrics["target_top1"]
        if best_metrics is None or score > (
            best_metrics["repertoire_move_top1"] + 0.25 * best_metrics["target_top1"]
        ):
            best_metrics = metrics
            best_state = {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:02d} loss={running / max(batches, 1):.4f} "
                f"move global={metrics['global_move_top1']:.3f} "
                f"pool@1/3={metrics['repertoire_move_top1']:.3f}/"
                f"{metrics['repertoire_move_top3']:.3f} "
                f"coverage={metrics['repertoire_coverage']:.3f} "
                f"target={metrics['target_top1']:.3f}",
                flush=True,
            )

    assert best_state is not None and best_metrics is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "species_vocab": species_vocab,
            "move_vocab": move_vocab,
            "config": {"embed_dim": args.embed_dim, "hidden_dim": args.hidden_dim},
            "metrics": {
                **best_metrics,
                "training_examples": len(train_examples),
                "source_logs": [str(path) for path in args.logs],
                "top_500_only": True,
            },
        },
        args.output,
    )
    print(f"saved {args.output}: {best_metrics}")


if __name__ == "__main__":
    main()
