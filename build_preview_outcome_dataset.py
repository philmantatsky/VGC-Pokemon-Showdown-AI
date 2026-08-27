"""Recover Team Preview choices and terminal labels from outcome_data_v1.

The original outcome files retained final labels but not the preview command. Their
generation is deterministic, so replaying only the RNG schedule and preview model
recovers the exact sampled plan without replaying 10,000 full battles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_outcome_dataset import _game_rng, _opponent_for_game, _seed
from vgc_bench.src.opponent_preview import PreviewPredictor
from vgc_bench.src.set_particles import team_roster


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outcome_data_v1/outcome_manifest.json")
    parser.add_argument("--our-team", default="teams/reg_mb/our_team.txt")
    parser.add_argument("--opponent-dir", default="teams/reg_mb")
    parser.add_argument("--preview-model", default="data/opponent_preview_top500_regmb.pt")
    parser.add_argument("--output", default="data/preview_outcome_examples.jsonl")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    config = manifest["config"]
    seed = int(config["seed"])
    hidden_probability = float(config["hidden_sheet_prob"])
    opponents = sorted(Path(args.opponent_dir).glob("MB*.txt"))
    ours = tuple(slot.species for slot in team_roster(Path(args.our_team).read_text()))
    preview = PreviewPredictor.load(Path(args.preview_model), device="cpu")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for game_text, summary in sorted(
        manifest["completed"].items(), key=lambda item: int(item[0])
    ):
        game = int(game_text)
        rng = _game_rng(seed, game)
        hidden = rng.random() < hidden_probability
        opponent = _opponent_for_game(opponents, seed, game)
        _seed(rng)  # consume the exact battle seed used before preview sampling
        theirs = tuple(
            slot.species for slot in team_roster(opponent.read_text())
        )
        prefix = preview.predict_plans(ours, theirs, top_k=90)[:4]
        plan = rng.choices(
            prefix,
            weights=[max(1e-8, candidate.probability) for candidate in prefix],
            k=1,
        )[0]
        rows.append(
            {
                "game": game,
                "opponent": opponent.name,
                "opponent_roster": theirs,
                "lead": plan.lead_indices,
                "bring": plan.bring_indices,
                "target": float(summary["target"]),
                "hidden_sheets": hidden,
                "opponent_style": summary["opponent_style"],
            }
        )
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    wins = sum(row["target"] for row in rows)
    print(
        f"preview outcomes: rows={len(rows)} win_rate={wins / max(1, len(rows)):.3f} "
        f"opponents={len({row['opponent'] for row in rows})} -> {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
