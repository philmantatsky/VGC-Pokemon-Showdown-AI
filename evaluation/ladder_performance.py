"""Performance rating of saved ladder games: the rating at which the results against
the opponents actually met are the likeliest ones.

A win rate on a ladder says little by itself: the ladder matches a player with others of
the player's own rating, so whoever is rated right wins about half. What a set of games
does show is a level -- the rating R that solves

    sum over games of (won - 1 / (1 + 10 ** ((opponent's rating - R) / 400))) = 0

with each opponent's rating read from the replay (the ``|player|`` line, the rating at
the start of that game). It does not depend on where our own rating happened to stand,
so two sets of games played from different ratings can be set side by side. What it
does depend on: the opponents' ratings meaning the same strength in both sets. Sets
played weeks apart on a young ladder are not a controlled comparison.

Counted: every finished rated game, a forfeit included (it is a win), except a game
that ended on turn 1 without a faint (a quit or a dead connection before any play).
Intervals: games resampled with replacement, the 2.5th and 97.5th percentile.

Usage (repo root):
  .venv/bin/python evaluation/ladder_performance.py \
      history=ladder_replays_mc_deployed_T6tac,ladder_replays_mc_T6tac_review_guards \
      run=ladder_replays_mc_search_forecast2 [--against history] [--json out.json]
Each ``name=dir[,dir...]`` is one set; ``--against`` adds every other set's difference
to that one (both resampled).
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.ladder_loss_profile import OUR_NAME, extract_log, parse_game  # noqa: E402

RESAMPLES = 2000
SEED = 20261010


def expected(rating: float, opponent: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent - rating) / 400.0))


def read_games(directories: list[Path], ours: str = OUR_NAME) -> list[dict[str, Any]]:
    """One row a counted game: won, the two ratings, the turn it ended on."""
    rows = []
    for directory in directories:
        for path in sorted(directory.glob("*.html"), key=lambda p: p.stat().st_mtime):
            log = extract_log(path)
            game = parse_game(log, ours) if log else None
            if not game or game.get("opp_rating") is None:
                continue
            if (
                int(game.get("turns") or 0) <= 1
                and game.get("first_faint_turn") is None
            ):
                continue  # over before any play
            rows.append(
                {
                    "won": bool(game["won"]),
                    "opponent": int(game["opp_rating"]),
                    "ours": game.get("our_rating"),
                    "turns": int(game.get("turns") or 0),
                    "forfeit": bool(game.get("forfeit")),
                    "directory": directory.name,
                }
            )
    return rows


def performance(rows: list[dict[str, Any]]) -> float | None:
    """The rating that makes the expected score the score; None when every game was
    won or every game lost (no finite rating does)."""
    wins = sum(row["won"] for row in rows)
    if not rows or wins in (0, len(rows)):
        return None
    low, high = 0.0, 4000.0
    for _ in range(60):
        middle = (low + high) / 2
        if sum(expected(middle, row["opponent"]) for row in rows) < wins:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _draws(rows: list[dict[str, Any]], rng: random.Random) -> list[float]:
    out = []
    for _ in range(RESAMPLES):
        value = performance(rng.choices(rows, k=len(rows)))
        if value is not None:
            out.append(value)
    return out


def _interval(values: list[float]) -> tuple[float, float] | None:
    if len(values) < 40:
        return None
    ordered = sorted(values)
    return (
        ordered[int(0.025 * (len(ordered) - 1))],
        ordered[int(0.975 * (len(ordered) - 1))],
    )


def summary(rows: list[dict[str, Any]], seed: int = SEED) -> dict[str, Any]:
    wins = sum(row["won"] for row in rows)
    own = [row["ours"] for row in rows if row["ours"] is not None]
    out: dict[str, Any] = {
        "games": len(rows),
        "wins": wins,
        "forfeit_wins": sum(row["won"] and row["forfeit"] for row in rows),
        "performance": performance(rows),
        "interval": _interval(_draws(rows, random.Random(seed))) if rows else None,
        "opponents_mean": statistics.fmean(r["opponent"] for r in rows)
        if rows
        else None,
        "own_rating": {
            "first": own[0],
            "last": own[-1],
            "mean": statistics.fmean(own),
            "lowest": min(own),
            "highest": max(own),
        }
        if own
        else None,
    }
    if own:
        # how many of these games our rating of the moment expected us to win
        out["expected_wins_at_own_rating"] = sum(
            expected(row["ours"], row["opponent"])
            for row in rows
            if row["ours"] is not None
        )
    return out


def difference(
    rows: list[dict[str, Any]], reference: list[dict[str, Any]], seed: int = SEED
) -> dict[str, Any]:
    """rows minus reference in performance rating, both sets resampled."""
    a, b = performance(rows), performance(reference)
    rng = random.Random(seed)
    one, two = _draws(rows, rng), _draws(reference, rng)
    gaps = [x - y for x, y in zip(one, two)]
    return {
        "difference": None if a is None or b is None else a - b,
        "interval": _interval(gaps),
        "share_above_zero": sum(g > 0 for g in gaps) / len(gaps) if gaps else None,
    }


def render(name: str, block: dict[str, Any]) -> str:
    if not block["games"]:
        return f"{name}: no counted game"
    rating = block["performance"]
    span = block["interval"]
    text = (
        f"{name}: {block['wins']} of {block['games']} "
        f"({100 * block['wins'] / block['games']:.0f}%), "
        f"{block['forfeit_wins']} wins by forfeit; opponents' mean rating "
        f"{block['opponents_mean']:.0f}; performance rating "
        + ("none" if rating is None else f"{rating:.0f}")
        + ("" if span is None else f" [{span[0]:.0f}, {span[1]:.0f}]")
    )
    own = block.get("own_rating")
    if own:
        text += (
            f"; own rating {own['first']} -> {own['last']} (mean {own['mean']:.0f}, "
            f"{own['lowest']} to {own['highest']}), which expected "
            f"{block['expected_wins_at_own_rating']:.1f} wins"
        )
    return text


def main() -> int:
    args = sys.argv[1:]
    against = args[args.index("--against") + 1] if "--against" in args else None
    target = args[args.index("--json") + 1] if "--json" in args else None
    sets: dict[str, list[dict[str, Any]]] = {}
    for arg in args:
        if "=" not in arg or arg.startswith("--"):
            continue
        name, folders = arg.split("=", 1)
        sets[name] = read_games([Path(folder) for folder in folders.split(",")])
    report: dict[str, Any] = {"sets": {}, "differences": {}}
    for name, rows in sets.items():
        report["sets"][name] = summary(rows)
        print(render(name, report["sets"][name]))
    if against is not None:
        for name, rows in sets.items():
            if name == against:
                continue
            gap = difference(rows, sets[against])
            report["differences"][f"{name} - {against}"] = gap
            span = gap["interval"]
            share = gap["share_above_zero"]
            line = f"{name} - {against}: "
            line += "none" if gap["difference"] is None else f"{gap['difference']:+.0f}"
            if span is not None:
                line += f" [{span[0]:+.0f}, {span[1]:+.0f}]"
            if share is not None:
                line += f"; above zero in {100 * share:.0f}% of resamples"
            print(line)
    if target:
        Path(target).write_text(json.dumps(report, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
