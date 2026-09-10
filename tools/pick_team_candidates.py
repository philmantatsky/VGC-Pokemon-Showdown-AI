"""Pick concrete team files for the Reg M-C candidate rosters (NEW_BRAIN_PLAN §4).

Each candidate is a roster (six species). Several open-team-sheet files in the
pool may carry that roster with different items, moves or spreads; this picks,
per candidate, the file whose sets agree most with the corpus-wide most common
sets (``results_analysis/mc_meta_stats_<date>.json``), and copies it to
``teams/candidates_mc/<id>.txt`` with the control team as T0. Rosters with no
exact match fall back to the best partial match (>= 5 species) and say so.

Usage (from the repo root):
  .venv/bin/python tools/pick_team_candidates.py \
      --stats results_analysis/mc_meta_stats_20260909.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys as _sys
from pathlib import Path
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from poke_env.data import to_id_str  # noqa: E402
from poke_env.teambuilder import Teambuilder  # noqa: E402

CANDIDATES: dict[str, list[str]] = {
    "T1": [
        "Floette-Eternal",
        "Incineroar",
        "Kingambit",
        "Rillaboom",
        "Salamence",
        "Sneasler",
    ],
    "T2": ["Indeedee-F", "Farigiraf", "Hatterene", "Torkoal", "Gallade", "Kingambit"],
    "T3": [
        "Indeedee-F",
        "Gardevoir",
        "Sneasler",
        "Basculegion",
        "Kingambit",
        "Salamence",
    ],
    "T4": ["Torkoal", "Charizard", "Venusaur", "Incineroar", "Sylveon", "Garchomp"],
    "T5": ["Pelipper", "Archaludon", "Golisopod", "Swampert", "Grimmsnarl", "Venusaur"],
}


def roster_of(path: Path) -> tuple[list[str], list[dict]]:
    mons = Teambuilder.parse_showdown_team(path.read_text())
    species = [to_id_str(m.species or m.nickname or "") for m in mons]
    sets = [
        {
            "species": to_id_str(m.species or m.nickname or ""),
            "item": to_id_str(m.item or ""),
            "moves": {to_id_str(mv) for mv in (m.moves or [])},
        }
        for m in mons
    ]
    return species, sets


def popularity(stats: dict, species: str) -> tuple[str | None, set[str]]:
    """(most common item id, most common move ids) for a species, from the corpus."""
    for name, entry in stats.get("sets", {}).items():
        if to_id_str(name) == species:
            items = entry.get("items") or []
            moves = entry.get("moves") or []
            top_item = to_id_str(items[0][0]) if items else None
            top_moves = {to_id_str(m) for m, _ in moves[:4]}
            return top_item, top_moves
    return None, set()


def score(sets: list[dict], stats: dict) -> float:
    total = 0.0
    for entry in sets:
        top_item, top_moves = popularity(stats, entry["species"])
        if top_item and entry["item"] == top_item:
            total += 1.0
        total += len(entry["moves"] & top_moves) / 4.0
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teams", type=Path, default=Path("teams/reg_mc"))
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("teams/candidates_mc"))
    ap.add_argument("--control", type=Path, default=Path("teams/reg_mc/our_team.txt"))
    args = ap.parse_args()
    stats = json.loads(args.stats.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.control, args.out / "T0.txt")
    print(f"T0 <- {args.control} (control)")

    pool = []
    for path in sorted(args.teams.glob("*.txt")):
        if path.name == "our_team.txt":
            continue
        species, sets = roster_of(path)
        pool.append((path, set(species), sets))

    manifest: dict[str, dict] = {"T0": {"source": str(args.control), "exact": True}}
    for cid, roster in CANDIDATES.items():
        wanted = {to_id_str(s) for s in roster}
        exact = [(p, sets) for p, species, sets in pool if species == wanted]
        partial = sorted(
            ((len(species & wanted), p, sets) for p, species, sets in pool),
            key=lambda item: -item[0],
        )
        if exact:
            best = max(exact, key=lambda item: score(item[1], stats))
            path, note = best[0], "exact"
        elif partial and partial[0][0] >= 5:
            best = max(
                (item for item in partial if item[0] == partial[0][0]),
                key=lambda item: score(item[2], stats),
            )
            path, note = best[1], f"partial ({best[0]}/6 species)"
        else:
            print(f"{cid}: no team in the pool shares >= 5 species; skipped")
            continue
        shutil.copy2(path, args.out / f"{cid}.txt")
        manifest[cid] = {"source": str(path), "exact": note == "exact", "note": note}
        print(f"{cid} <- {path} ({note}, {len(exact)} exact variants)")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
