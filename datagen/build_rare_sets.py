"""Sets for the species the joint set data cannot fill (2026-10-09).

The exact search draws each hidden opposing set from ``data/joint_sets_reg<reg>.json``
(full sets seen on open team sheets). A species that file lacks has no set at all, the
simulator refuses the team ("Set Arbok has no moves"), no world can be created, and
the search is off for the WHOLE game -- every decision falls back to the bot's own
move. Of 415 recorded ladder games 47 (11%) had such a species on the other side (30
species: Mimikyu, Inteleon, Ampharos, ...); the ladder trial of the search met two of
them in its first nine games.

``ParticleDatabase.load`` already reads ``data/rare_sets_reg<reg>.json`` on top of the
joint sets. This writes that file from the opponent predictor's repertoire -- how
often each species used each move in human games (training split only): one set per
missing species, its four most used moves. What the repertoire does not hold is not
invented: no item, and the ability is the species' first listed one (a Pokemon
without an ability line has none in the simulator).

Only the search's worlds read the file (``set_particles.ParticleDatabase``); the
brain's observation has its own set data (``set_priors``) and is not touched.

Usage (repo root):
    .venv/bin/python datagen/build_rare_sets.py --reg mc
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from typing import Any, Mapping  # noqa: E402

MAX_MOVES = 4


def rare_sets(
    counts: Mapping[str, Mapping[str, float]],
    known: set[str],
    abilities: Mapping[str, str],
    min_support: float = 5.0,
    min_moves: int = 2,
) -> dict[str, dict[str, Any]]:
    """One set per species in ``counts`` that ``known`` lacks.

    ``counts[species][move]`` is a weighted number of uses. A species counts when it
    was seen to move at least ``min_support`` times with at least ``min_moves``
    different moves: below that the four "most used" moves are whatever one game
    happened to show. The set is the (up to) four most used moves, by falling use
    and then by name, so the file does not depend on dictionary order.
    """
    out: dict[str, dict[str, Any]] = {}
    for species in sorted(counts):
        if species in known:
            continue
        uses = {m: float(c) for m, c in counts[species].items() if m and c > 0}
        support = sum(uses.values())
        if support < min_support or len(uses) < min_moves:
            continue
        moves = sorted(uses, key=lambda move: (-uses[move], move))[:MAX_MOVES]
        out[species] = {
            "sets": [
                {
                    "ability": abilities.get(species) or None,
                    "count": int(round(support)),
                    "item": None,
                    "moves": moves,
                    "prob": 1.0,
                }
            ]
        }
    return out


def first_abilities(species: list[str]) -> dict[str, str]:
    """Each species' first listed ability (its id), where the dex knows one."""
    from poke_env.data import GenData, to_id_str

    dex = GenData.from_gen(9).pokedex
    out = {}
    for name in species:
        ability = ((dex.get(name) or {}).get("abilities") or {}).get("0")
        if ability:
            out[name] = to_id_str(ability)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reg", default="mc")
    ap.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help="the opponent predictor's artifact (default: the deployed "
        "configuration's opponent_forecast)",
    )
    ap.add_argument("--min-support", type=float, default=5.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    artifact = args.artifact
    if artifact is None:
        from tools.deployed_config import resolve

        artifact = Path(resolve()["FORECAST"] or "")
    if not (ROOT / artifact).is_file():
        raise SystemExit(f"no predictor artifact at {artifact}")
    joint_path = ROOT / "data" / f"joint_sets_reg{args.reg}.json"
    known = set(json.loads(joint_path.read_text())) if joint_path.exists() else set()

    from vgc_bench.src.oppmodel.runtime import OpponentPredictor

    runtime = OpponentPredictor.load(ROOT / artifact)
    featurizer = runtime.featurizer
    if featurizer is None:
        raise SystemExit(f"the artifact did not load: {runtime.load_failure}")
    counts = featurizer.repertoire.counts
    sets = rare_sets(
        counts, known, first_abilities(sorted(counts)), min_support=args.min_support
    )
    out = args.out or ROOT / "data" / f"rare_sets_reg{args.reg}.json"
    out.write_text(json.dumps(sets, indent=1, sort_keys=True) + "\n")
    metadata = {
        "source": {
            "artifact": str(artifact),
            "sha256": hashlib.sha256((ROOT / artifact).read_bytes()).hexdigest(),
            "what": "the predictor's repertoire: weighted move uses per species in "
            "human games, training split only",
        },
        "joint_sets": {
            "path": str(joint_path.relative_to(ROOT)),
            "species": len(known),
        },
        "species": len(sets),
        "min_support": args.min_support,
        "moves": "the four most used; use is not set membership",
        "abilities": "the species' first listed ability; not observed",
        "items": "none: not observed, not fabricated",
        "role": "search worlds only (set_particles.ParticleDatabase); the brain's "
        "observation does not read this file",
    }
    Path(str(out) + ".metadata.json").write_text(json.dumps(metadata, indent=1) + "\n")
    skipped = sorted(s for s in counts if s not in known and s not in sets)
    shown = ", ".join(skipped[:12]) + (" ..." if len(skipped) > 12 else "")
    print(
        f"{len(sets)} species written to {out} ({len(known)} already have joint "
        f"sets; {len(skipped)} below the support floor: {shown})"
    )


if __name__ == "__main__":
    main()
