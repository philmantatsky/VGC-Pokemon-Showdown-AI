"""Measure bounded hidden-set coverage across the complete Reg M-B team pool."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from vgc_bench.src.set_particles import ParticleDatabase, team_roster

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teams", type=Path, default=ROOT / "teams/reg_mb")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results_parity/particle_coverage.json",
    )
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--required", type=float, default=0.85)
    args = parser.parse_args()
    database = ParticleDatabase.load(max_particles=12)
    species = sorted(
        {
            slot.species
            for path in args.teams.glob("*.txt")
            for slot in team_roster(path.read_text())
        }
    )
    rows = [
        {
            "species": name,
            "coverage": database.top_coverage(name, args.width),
            "particles": len(database.particles(name)),
        }
        for name in species
    ]
    mean_coverage = sum(row["coverage"] for row in rows) / max(1, len(rows))
    missing = [row["species"] for row in rows if not row["particles"]]
    payload = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "width": args.width,
        "required_coverage": args.required,
        "species_count": len(rows),
        "mean_coverage": mean_coverage,
        "missing_species": missing,
        "accepted": mean_coverage >= args.required and not missing,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"particle coverage: {mean_coverage:.1%} across {len(rows)} species; "
        f"missing={len(missing)} accepted={payload['accepted']}"
    )
    if not payload["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
