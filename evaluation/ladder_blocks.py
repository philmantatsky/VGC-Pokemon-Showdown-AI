"""A ladder run read block by block, in the order it was played.

For every block of games and for all games up to its end: the record, the rating at the
start of its first and last game, the performance rating
(``evaluation/ladder_performance.py``) and, for a run played with the search, the
search's own readings (``evaluation/search_ladder_read.py``): decisions searched,
fallbacks by cause, the opponent's real reply in the table, the overrides.

One configuration plays a whole run, so a block does not differ from the one before it
because the bot got better: it differs by chance and by where the rating stood. A block
of 25 games leaves its performance rating about a hundred points either way -- it is
printed as description; the number with an interval is the one over all games so far.

Usage (repo root):
  .venv/bin/python evaluation/ladder_blocks.py <replay dir> --edges 40,75,100 \
      [--json out.json]
``--edges``: the game counts at which blocks end (the last may be short of the games
there are: the rest is one more block). Without it, every 25 games.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.ladder_performance import read_games, summary  # noqa: E402
from evaluation.search_ladder_read import read  # noqa: E402

LOGS = ("decisions.jsonl", "decisions_champion.jsonl")
BATTLE = re.compile(r"-(\d+)(?:-[a-z0-9]+)?$")


def battle_number(tag: str) -> str:
    """The battle's number, the same in a replay's file name and in the decision logs
    (a private replay's name carries a key after it)."""
    found = BATTLE.search(tag.removesuffix(".html"))
    return found.group(1) if found else tag


def replays_in_order(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.html"), key=lambda path: path.stat().st_mtime)


def edges_for(games: int, wanted: list[int] | None, block: int = 25) -> list[int]:
    """Where blocks end: the wanted counts that have been reached, then the rest."""
    marks = wanted if wanted else list(range(block, games + 1, block))
    reached = sorted({mark for mark in marks if 0 < mark <= games})
    if not reached or reached[-1] < games:
        reached.append(games)
    return reached


def subset(directory: Path, replays: list[Path], target: Path) -> Path:
    """These games as a directory of their own: their replays (linked, so their times
    stay) and their rows of the decision logs."""
    target.mkdir(parents=True, exist_ok=True)
    numbers = {battle_number(path.name) for path in replays}
    for path in replays:
        (target / path.name).symlink_to(path.resolve())
    for name in LOGS:
        source = directory / name
        if not source.exists():
            continue
        with source.open() as lines, (target / name).open("w") as out:
            for line in lines:
                try:
                    tag = str(json.loads(line).get("battle") or "")
                except ValueError:
                    continue
                if battle_number(tag) in numbers:
                    out.write(line)
    return target


def reading(directory: Path) -> dict[str, Any]:
    """What one set of games shows, as plain numbers."""
    level = summary(read_games([directory]))
    out: dict[str, Any] = {
        "games": level["games"],
        "wins": level["wins"],
        "forfeit_wins": level["forfeit_wins"],
        "performance": level["performance"],
        "interval": level["interval"],
        "opponents_mean": level["opponents_mean"],
        "own_rating": level["own_rating"],
    }
    if not (directory / "decisions.jsonl").exists():
        return out
    search = read(directory)
    modes = search["search"]["modes"]
    decisions = int(search["search"]["decisions"])
    edges = [
        o["realized_edge"]
        for o in search["overrides"]
        if o["realized_edge"] is not None
    ]
    hidden = search["reply_table"].get("hidden") or {}
    out.update(
        {
            "all_games": search["record"]["games"],
            "all_wins": search["record"]["wins"],
            "decisions": decisions,
            "searched": int(modes.get("search", 0)),
            "not_searched": search["search"]["reasons"],
            "seconds": search["search"]["seconds"],
            "replies_hidden": int(hidden.get("decisions") or 0),
            "replies_in_table": int(hidden.get("in_some_world") or 0),
            "replies_by_kind": search["reply_table_by_kind"],
            "overrides": len(search["overrides"]),
            "overrides_with_a_searched_reply": len(edges),
            "override_edge_mean": statistics.fmean(edges) if edges else None,
            "override_edge_not_positive": sum(edge <= 0 for edge in edges),
        }
    )
    return out


def blocks(directory: Path, wanted: list[int] | None = None) -> list[dict[str, Any]]:
    replays = replays_in_order(directory)
    rows = []
    start = 0
    with tempfile.TemporaryDirectory() as scratch:
        for end in edges_for(len(replays), wanted):
            rows.append(
                {
                    "games": f"{start + 1}-{end}",
                    "block": reading(
                        subset(directory, replays[start:end], Path(scratch) / f"b{end}")
                    ),
                    "so_far": reading(
                        subset(directory, replays[:end], Path(scratch) / f"c{end}")
                    ),
                }
            )
            start = end
    return rows


def _share(part: int, whole: int) -> str:
    return f"{part} of {whole}" + (f" ({100 * part / whole:.0f}%)" if whole else "")


def _line(label: str, row: dict[str, Any], interval: bool) -> list[str]:
    games = row.get("all_games", row["games"])
    wins = row.get("all_wins", row["wins"])
    own = row.get("own_rating") or {}
    level = row["performance"]
    text = (
        f"{label}: won {_share(wins, games)}, {row['forfeit_wins']} by forfeit; "
        f"rating {own.get('first')} -> {own.get('last')}; opponents' mean "
        + ("-" if row["opponents_mean"] is None else f"{row['opponents_mean']:.0f}")
        + "; performance "
        + ("-" if level is None else f"{level:.0f}")
    )
    if interval and row["interval"]:
        text += f" [{row['interval'][0]:.0f}, {row['interval'][1]:.0f}]"
    lines = [text]
    if "decisions" in row:
        missed = ", ".join(f"{n} {why}" for why, n in row["not_searched"].items())
        edge = row["override_edge_mean"]
        checked = row["overrides_with_a_searched_reply"]
        text = f"    searched {_share(row['searched'], row['decisions'])}"
        if missed:
            text += f" (not searched: {missed})"
        text += "; real reply in the table, hidden sheets: "
        text += _share(row["replies_in_table"], row["replies_hidden"])
        text += f"; overrides {row['overrides']}, {checked} with the reply searched"
        if edge is not None:
            worse = row["override_edge_not_positive"]
            text += f": edge {edge:+.2f}, {worse} not positive"
        lines.append(text)
    return lines


def render(rows: list[dict[str, Any]]) -> list[str]:
    out = []
    for row in rows:
        out += _line(f"games {row['games']}", row["block"], interval=False)
    if rows:
        out.append("")
        total = rows[-1]["so_far"]
        count = total.get("all_games", total["games"])
        out += _line(f"ALL {count} games", total, interval=True)
    return out


def main() -> int:
    args = sys.argv[1:]
    directory = Path(args[0])
    wanted = None
    if "--edges" in args:
        wanted = [int(mark) for mark in args[args.index("--edges") + 1].split(",")]
    rows = blocks(directory, wanted)
    print("\n".join(render(rows)))
    if "--json" in args:
        target = Path(args[args.index("--json") + 1])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(rows, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
