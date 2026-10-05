"""What the exact search changed, and how those games went (2026-10-04).

A searched side normally plays the pair the bot would play anyway; the decisions that
matter are the overrides. This lines each run's search audit (``a_decisions.jsonl``:
``champion_actions``, ``actions``, the ranked rows) up with its per-battle results
(``result.json``: ``battles``) and reports:

* how often the search kept the bot's pair, overrode it, or was sent back by the guards;
* the payoff edge and the prior ratio behind each override, and what kind of change
  it was (another move, another target, a Protect, a switch);
* side A's results in games with 0, 1, 2 and 3+ overrides -- descriptive only: a game
  that offers more overrides is a different kind of game, so this is not the search's
  effect (the head-to-head rate is).

Usage (from the repo root):
    .venv/bin/python evaluation/search_override_report.py <run dir> [<run dir> ...]
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.mirror_guard_ab import wilson  # noqa: E402

PROTECTS = {
    "protect",
    "detect",
    "spikyshield",
    "banefulbunker",
    "kingsshield",
    "burningbulwark",
    "silktrap",
    "obstruct",
    "wideguard",
    "quickguard",
}


def _atoms(choice: str) -> list[str]:
    atoms = [atom.strip() for atom in choice.split(",")]
    return (atoms + ["pass", "pass"])[:2]


def change_kind(before: str, after: str) -> str:
    """How one slot's order changed: 'switch', 'protect', 'target' or 'move'."""
    old, new = before.split(), after.split()
    if "switch" in (old[:1] + new[:1]):
        return "switch"
    old_move = old[1] if len(old) > 1 else ""
    new_move = new[1] if len(new) > 1 else ""
    if new_move in PROTECTS or old_move in PROTECTS:
        return "protect"
    return "target" if old_move == new_move else "move"


def _row_for(rankings: list[dict], actions: list[int] | None) -> dict | None:
    if actions is None:
        return None
    return next((r for r in rankings if list(r.get("actions") or []) == actions), None)


def report(runs: list[Path]) -> dict:
    results: dict[str, object] = {}
    for run in runs:
        blocks = json.loads((run / "result.json").read_text())["blocks"]
        for block in blocks:
            for battle in block.get("battles") or []:
                results[battle["battle"]] = battle["a_won"]
    outcomes: collections.Counter[str] = collections.Counter()
    kinds: collections.Counter[str] = collections.Counter()
    swaps: collections.Counter[str] = collections.Counter()
    per_battle: collections.Counter[str] = collections.Counter()
    by_sheet: dict[str, collections.Counter[str]] = {
        "open": collections.Counter(),
        "hidden": collections.Counter(),
    }
    edges: list[float] = []
    ratios: list[float] = []
    turns: list[int] = []
    decisions = 0
    seen_battles: set[str] = set()
    for run in runs:
        for line in (run / "a_decisions.jsonl").read_text().splitlines():
            row = json.loads(line) if line.strip() else {}
            audit = row.get("exact_search")
            if not audit or audit.get("champion_actions") is None:
                continue
            decisions += 1
            seen_battles.add(row["battle"])
            sheet = "open" if audit.get("open_sheet") else "hidden"
            schedule = audit.get("schedule") or {}
            played = audit.get("actions")
            if schedule.get("mode") != "search" or played is None:
                outcomes["fallback"] += 1
                by_sheet[sheet]["fallback"] += 1
                continue
            outcome = (schedule.get("live_guards") or {}).get("champion") or (
                "kept" if played == audit["champion_actions"] else "unanchored"
            )
            outcomes[outcome] += 1
            by_sheet[sheet][outcome] += 1
            if played == audit["champion_actions"]:
                continue
            per_battle[row["battle"]] += 1
            turns.append(int(row.get("turn") or 0))
            rankings = (audit.get("result") or {}).get("rankings") or []
            chosen = _row_for(rankings, played)
            default = _row_for(rankings, audit["champion_actions"])
            if chosen and default:
                edges.append(float(chosen["expected"]) - float(default["expected"]))
                if float(chosen.get("prior") or 0) > 0:
                    ratios.append(float(default["prior"]) / float(chosen["prior"]))
                for old, new in zip(
                    _atoms(default["choice"]), _atoms(chosen["choice"])
                ):
                    if old == new:
                        continue
                    kind = change_kind(old, new)
                    kinds[kind] += 1
                    if kind == "move":
                        swaps[f"{old.split()[1]} -> {new.split()[1]}"] += 1

    def record(tags: list[str]) -> dict:
        known = [results[tag] for tag in tags if tag in results]
        wins = sum(1.0 if won else 0.5 if won is None else 0.0 for won in known)
        low, high = wilson(wins, len(known))
        return {
            "games": len(known),
            "a_wins": wins,
            "a_win_rate": wins / len(known) if known else None,
            "wilson_95": [low, high],
        }

    buckets: dict[str, list[str]] = {"0": [], "1": [], "2": [], "3+": []}
    for tag in sorted(seen_battles):
        count = per_battle.get(tag, 0)
        buckets["3+" if count >= 3 else str(count)].append(tag)
    edges.sort()

    def quantile(values: list[float], q: float) -> float | None:
        return values[min(len(values) - 1, int(q * len(values)))] if values else None

    overrides = sum(per_battle.values())
    return {
        "runs": [str(run) for run in runs],
        "decisions": decisions,
        "outcomes": dict(outcomes),
        "outcomes_by_sheets": {k: dict(v) for k, v in by_sheet.items()},
        "overrides": overrides,
        "override_share": overrides / decisions if decisions else None,
        "overrides_per_game": overrides / len(seen_battles) if seen_battles else None,
        "override_payoff_edge": {
            "p10": quantile(edges, 0.1),
            "median": quantile(edges, 0.5),
            "p90": quantile(edges, 0.9),
            "non_positive": sum(edge <= 0 for edge in edges),
            "of": len(edges),
        },
        "override_prior_ratio_median": statistics.median(ratios) if ratios else None,
        "override_turn_median": statistics.median(turns) if turns else None,
        "changed_slots_by_kind": dict(kinds),
        "top_move_swaps": dict(swaps.most_common(10)),
        "side_a_by_overrides_in_the_game": {k: record(v) for k, v in buckets.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    summary = report(args.runs)
    text = json.dumps(summary, indent=2)
    if args.json is not None:
        args.json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
