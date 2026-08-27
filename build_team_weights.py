"""Build ladder-representative sampling weights for the bundled Reg M-B teams.

The replay scrape contains team-preview rosters but not always complete sets.  We
therefore weight by the six-species archetype. If several bundled set variants share
one roster, its observed count is divided between them so duplicate files cannot
multiply that archetype's probability.

    python build_team_weights.py
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from poke_env.data import to_id_str
from poke_env.teambuilder import Teambuilder

POKE_LINE = re.compile(r"^\|poke\|(p[12])\|([^|]+)", re.MULTILINE)


def replay_rosters(path: Path) -> Counter[tuple[str, ...]]:
    counts: Counter[tuple[str, ...]] = Counter()
    payload = json.loads(path.read_text())
    for _timestamp, log in payload.values():
        sides: dict[str, list[str]] = {"p1": [], "p2": []}
        for side, details in POKE_LINE.findall(log):
            species = re.split(r", L\d+", details, maxsplit=1)[0]
            sides[side].append(to_id_str(species))
        for roster in sides.values():
            if len(roster) == 6:
                counts[tuple(sorted(roster))] += 1
    return counts


def team_signature(path: Path) -> tuple[str, ...]:
    parsed = Teambuilder.parse_showdown_team(path.read_text())
    return tuple(sorted(to_id_str(mon.species or mon.nickname or "") for mon in parsed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs",
        type=Path,
        nargs="+",
        default=[
            Path("battle_logs_top/logs_gen9championsvgc2026regmb.json"),
            Path("battle_logs_top/logs_gen9championsvgc2026regmbbo3.json"),
        ],
    )
    parser.add_argument("--teams", type=Path, default=Path("teams/reg_mb"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/team_weights_regmb.json")
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=None,
        help="explicit base weight per team (overrides --uniform_mix)",
    )
    parser.add_argument(
        "--uniform_mix",
        type=float,
        default=0.5,
        help="fraction of total sampling mass reserved for uniform full-pool coverage",
    )
    args = parser.parse_args()

    missing = [path for path in args.logs if not path.exists()]
    if missing:
        raise SystemExit(f"missing replay logs: {', '.join(map(str, missing))}")

    roster_counts: Counter[tuple[str, ...]] = Counter()
    for path in args.logs:
        roster_counts.update(replay_rosters(path))

    by_roster: dict[tuple[str, ...], list[Path]] = defaultdict(list)
    for path in sorted(args.teams.rglob("*.txt")):
        by_roster[team_signature(path)].append(path)

    if not 0.0 < args.uniform_mix < 1.0:
        parser.error("--uniform_mix must be strictly between 0 and 1")
    matched_observations = sum(
        roster_counts[roster] for roster in by_roster if roster in roster_counts
    )
    num_teams = sum(len(paths) for paths in by_roster.values())
    smoothing = args.smoothing
    if smoothing is None:
        uniform_total = matched_observations * args.uniform_mix / (1 - args.uniform_mix)
        smoothing = uniform_total / num_teams
    if smoothing < 0:
        parser.error("--smoothing must be non-negative")

    weights: dict[str, float] = {}
    matched_files = 0
    for roster, paths in by_roster.items():
        observed = roster_counts.get(roster, 0)
        if observed:
            matched_files += len(paths)
        per_variant = observed / len(paths)
        for path in paths:
            weights[path.name] = round(smoothing + per_variant, 6)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(sorted(weights.items())), indent=2) + "\n")
    values = list(weights.values())
    effective_teams = sum(values) ** 2 / sum(value * value for value in values)
    actual_uniform_mix = smoothing * num_teams / sum(values)
    print(
        f"wrote {len(weights)} weights to {args.output}; "
        f"{matched_files} team files / {matched_observations} previews matched; "
        f"uniform mix={actual_uniform_mix:.0%}, effective teams={effective_teams:.1f}"
    )


if __name__ == "__main__":
    main()
