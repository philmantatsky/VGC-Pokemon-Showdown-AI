"""Meta snapshot from scraped replays, optionally against a reference window.

Reads the scraper's JSON files ({replay_id: [epoch_seconds, log]}), optionally
restricted to a time window, and reports what the field looks like: species
usage on team sheets (with the win rate of the sheets carrying them), how often
Trick Room and Tailwind go up, and which weather appears. With ``--reference``
the same numbers are computed for a second window and printed as deltas, which
is how a team chosen on older data is checked against the current field.

Usage (from the repo root):
  .venv/bin/python tools/mc_meta_stats.py \\
      --logs battle_logs_top_mc_20260920/logs_*.json --since 1789275733 \\
      --reference battle_logs_top_mc_20260913/logs_*.json --top 25 \\
      --output results_analysis/mc_meta_stats_20260920.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

WEATHERS = {
    "SunnyDay": "sun",
    "DesolateLand": "sun",
    "RainDance": "rain",
    "PrimordialSea": "rain",
    "Sandstorm": "sand",
    "Snowscape": "snow",
    "Snow": "snow",
    "Hail": "snow",
}


def parse_game(log: str) -> dict:
    """Sheets, winner side and field events of one replay log."""
    sheets: dict[str, set[str]] = {"p1": set(), "p2": set()}
    names: dict[str, str] = {}
    winner = None
    trick_room = tailwind = False
    weather: set[str] = set()
    for line in log.split("\n"):
        parts = line.split("|")
        if len(parts) < 3:
            continue
        tag = parts[1]
        if tag == "poke" and len(parts) >= 4 and parts[2] in sheets:
            species = parts[3].split(",")[0].strip().removesuffix("-*")
            sheets[parts[2]].add(species)
        elif tag == "player" and len(parts) >= 4 and parts[2] in sheets and parts[3]:
            names.setdefault(parts[2], parts[3])
        elif tag == "win":
            winner = next((s for s, n in names.items() if n == parts[2]), None)
        elif tag == "-fieldstart" and "Trick Room" in line:
            trick_room = True
        elif tag == "-sidestart" and "Tailwind" in line:
            tailwind = True
        elif tag == "-weather" and parts[2] in WEATHERS:
            weather.add(WEATHERS[parts[2]])
    return {
        "sheets": sheets,
        "winner": winner,
        "trick_room": trick_room,
        "tailwind": tailwind,
        "weather": weather,
    }


def load_games(paths: list[Path], since: int | None, until: int | None) -> list[dict]:
    games: dict[str, dict] = {}
    for path in paths:
        for replay_id, (stamp, log) in json.loads(path.read_text()).items():
            if since is not None and stamp <= since:
                continue
            if until is not None and stamp > until:
                continue
            games.setdefault(replay_id, parse_game(log))
    return list(games.values())


def snapshot(games: list[dict]) -> dict:
    sheets = wins = 0
    usage: Counter[str] = Counter()
    species_wins: Counter[str] = Counter()
    species_decided: Counter[str] = Counter()
    events: Counter[str] = Counter()
    for game in games:
        events["trick_room"] += game["trick_room"]
        events["tailwind"] += game["tailwind"]
        for kind in game["weather"]:
            events[kind] += 1
        for side, roster in game["sheets"].items():
            if not roster:
                continue
            sheets += 1
            usage.update(roster)
            if game["winner"] is not None:
                wins += side == game["winner"]
                for species in roster:
                    species_decided[species] += 1
                    species_wins[species] += side == game["winner"]
    n = max(len(games), 1)
    return {
        "games": len(games),
        "sheets": sheets,
        "usage": {s: c / max(sheets, 1) for s, c in usage.items()},
        "sheet_win_rate": {
            s: species_wins[s] / species_decided[s]
            for s in species_decided
            if species_decided[s]
        },
        "sheet_counts": dict(usage),
        "events": {k: v / n for k, v in events.items()},
    }


def report(current: dict, reference: dict | None, top: int) -> list[str]:
    lines = [f"games {current['games']}  sheets {current['sheets']}"]
    ref_events = reference["events"] if reference else {}
    for key in ("trick_room", "tailwind", "sun", "rain", "sand", "snow"):
        value = 100 * current["events"].get(key, 0.0)
        delta = (
            f"  ({value - 100 * ref_events.get(key, 0.0):+.1f} vs ref)"
            if reference
            else ""
        )
        lines.append(f"  {key:<11} in {value:5.1f}% of games{delta}")
    ranked = sorted(current["usage"], key=lambda s: -current["usage"][s])[:top]
    for species in ranked:
        use = 100 * current["usage"][species]
        win = 100 * current["sheet_win_rate"].get(species, float("nan"))
        delta = ""
        if reference:
            delta = (
                f"  ({use - 100 * reference['usage'].get(species, 0.0):+5.1f} vs ref)"
            )
        lines.append(
            f"  {species:<22} {use:5.1f}% of sheets  sheet win {win:5.1f}%{delta}"
        )
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", type=Path, nargs="+", required=True)
    ap.add_argument(
        "--since", type=int, default=None, help="keep replays after this epoch"
    )
    ap.add_argument(
        "--until", type=int, default=None, help="keep replays up to this epoch"
    )
    ap.add_argument("--reference", type=Path, nargs="*", default=None)
    ap.add_argument("--reference-until", type=int, default=None)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    current = snapshot(load_games(args.logs, args.since, args.until))
    reference = None
    if args.reference:
        reference = snapshot(load_games(args.reference, None, args.reference_until))
    print("\n".join(report(current, reference, args.top)))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"current": current, "reference": reference}, indent=1)
        )


if __name__ == "__main__":
    main()
