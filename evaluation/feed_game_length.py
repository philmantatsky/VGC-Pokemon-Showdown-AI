"""How long real games are, in turns and in decision steps (2026-10-11; asked for by the
vgczero session to price a self-play run -- its projection assumed 14.5 simulator steps
a game and an untrained policy had played 9.0).

A decision step is a point where the simulator stops for the choice of at least one
side: team preview, the move request of a turn, and every forced-switch request inside
a turn (U-turn, Parting Shot, Eject Button ...) or after it (who replaces a fainted
Pokemon). ``choices`` counts the sides choosing: both at preview and at a move
request, at a forced switch the sides seen switching.

Showdown writes one ``|t:|`` line each time it resumes after such a stop
(``Battle.turnLoop``) and one when the battle is created, so the steps of a game are
its ``|t:|`` lines less one. A request nobody answered -- the game ended by forfeit or
on the timer first -- is not among them; ``pending`` says a game ended on one.

A step is named by where the battle stood when it stopped: before the first turn
(preview), at the start of a turn (``|turn|N`` written, nothing of it played: a move
request), inside a turn (a switch a move or an item asks for), after a turn's
``|upkeep`` (replacements). Lines written between a stop and its answer -- chat, the
timer, a Quick Claw announced as the choices are read -- change nothing in that.

A game ends ``natural`` when its loser has no Pokemon left, by ``forfeit`` or on the
``timer`` when the log says so, ``other`` when it names a winner without either, and is
``unfinished`` without a result. The simulator has no forfeit: ``completed`` adds to a
game that ended early the steps games played to the end still took from a request with
the same Pokemon left to the eventual loser and winner (an estimate: a side that gives
up is likely nearer its end than that).

Usage (repo root; one process, a few minutes for 120,000 games):
  .venv/bin/python evaluation/feed_game_length.py [corpus dir] [--json out.json]
The corpus: ``<dir>/logs/<format>/part-*.jsonl.gz``, one game a line with ``log`` and
``rating`` (``datagen/oppmodel_scrape.py``); default ``battle_logs_feed_mc``.
"""

from __future__ import annotations

import collections
import gzip
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]

BANDS = (("all", None), ("1300+", 1300), ("1500+", 1500))
KINDS = ("preview", "move", "after_turn", "inside_turn")


def _tag(line: str) -> str:
    return line.split("|", 2)[1] if line.startswith("|") and line.count("|") else ""


def game_length(log: str) -> dict[str, Any]:
    """Turns, decision steps by kind, how the game ended, and at every request the
    Pokemon each side had left (for ``completed``)."""
    kinds = collections.Counter()
    size = {"p1": 4, "p2": 4}
    fainted = {"p1": 0, "p2": 0}
    left: list[tuple[int, int]] = []  # (p1, p2) left at each answered request
    turns = resumed = choices = 0
    phase = "preview"
    switching: set[str] | None = None  # the sides seen answering a forced switch
    names: dict[str, str] = {}
    winner = None
    gave_up = None  # the name in a forfeit or timer line
    forfeit = timer = False
    for line in log.split("\n"):
        tag = _tag(line)
        if tag in ("t:", "turn", "upkeep") and switching is not None:
            choices += max(1, len(switching))
            switching = None
        if tag == "t:":
            resumed += 1
            if resumed > 1:  # the first is the battle's creation
                kinds[phase] += 1
                left.append((size["p1"] - fainted["p1"], size["p2"] - fainted["p2"]))
                if phase in ("preview", "move"):
                    choices += 2
                else:
                    switching = set()
                if phase == "move":
                    phase = "inside_turn"
        elif tag == "turn":
            turns = max(turns, int(line.split("|")[2]))
            phase = "move"
        elif tag == "upkeep":
            phase = "after_turn"
        elif tag == "switch":
            if switching is not None:
                switching.add(line.split("|")[2][:2])
        elif tag == "-heal":
            # a Pokemon brought back is no longer among the fainted
            side = line.split("|")[2][:2]
            if line.endswith("[from] move: Revival Blessing") and fainted.get(side):
                fainted[side] -= 1
        elif tag == "faint":
            side = line.split("|")[2][:2]
            if side in fainted:
                fainted[side] += 1
        elif tag == "teamsize":
            parts = line.split("|")
            if parts[2] in size and parts[3].isdigit():
                size[parts[2]] = int(parts[3])
        elif tag == "player":
            parts = line.split("|")
            if len(parts) > 3 and parts[2] in size and parts[3]:
                names[parts[3]] = parts[2]
        elif tag == "win":
            winner = line.split("|", 2)[2].strip()
        elif tag == "tie":
            winner = ""
        elif tag == "-message":
            text = line.split("|", 2)[2]
            for ending in (" forfeited.", " lost due to inactivity."):
                if text.endswith(ending):
                    forfeit |= ending == " forfeited."
                    timer |= ending != " forfeited."
                    gave_up = text[: -len(ending)]
    out = [side for side in ("p1", "p2") if fainted[side] >= size[side]]
    other = {"p1": "p2", "p2": "p1"}
    lost = other.get(names.get(winner or "", "")) or names.get(gave_up or "")
    if lost is None and len(out) == 1:
        lost = out[0]
    if winner is None:
        ended = "unfinished"
    elif forfeit:
        ended = "forfeit"
    elif timer:
        ended = "timer"
    elif winner == "" or out:
        ended = "natural"
    else:
        ended = "other"
    if switching is not None:
        choices += max(1, len(switching))
    steps = max(0, resumed - 1)
    final = (size["p1"] - fainted["p1"], size["p2"] - fainted["p2"])
    return {
        "turns": turns,
        "steps": steps,
        "choices": choices,
        **{kind: kinds[kind] for kind in KINDS},
        "ended": ended,
        "pending": ended in ("forfeit", "timer", "other", "unfinished"),
        "loser": lost,
        "left_at_requests": left,
        "left_at_end": final,
    }


Position = tuple[int, int, bool]  # the loser's Pokemon left, the winner's, in preview


def _position(pair: tuple[int, int], loser: str, preview: bool) -> Position:
    return (pair[0], pair[1], preview) if loser == "p1" else (pair[1], pair[0], preview)


class Remaining:
    """From games played to the end: the steps still to come at a request, by the
    Pokemon the eventual loser and winner had left (team preview apart: the whole
    game is still to come there)."""

    def __init__(self) -> None:
        self.total: collections.Counter[Position] = collections.Counter()
        self.count: collections.Counter[Position] = collections.Counter()

    def add(self, game: dict[str, Any]) -> None:
        if game["ended"] != "natural" or game["loser"] is None:
            return
        steps = len(game["left_at_requests"])
        for index, pair in enumerate(game["left_at_requests"]):
            key = _position(pair, game["loser"], index == 0)
            self.total[key] += steps - index
            self.count[key] += 1

    def expected(self, game: dict[str, Any]) -> float | None:
        """Steps a game that ended early would still have taken, counting the request
        it ended on; None when no finished game was ever in that position."""
        if game["loser"] is None:
            return None
        key = _position(game["left_at_end"], game["loser"], game["steps"] == 0)
        return self.total[key] / self.count[key] if self.count[key] else None


def games(corpus: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    for directory in sorted((corpus / "logs").iterdir()):
        if not directory.is_dir():
            continue
        for part in sorted(directory.glob("part-*.jsonl.gz")):
            with gzip.open(part, "rt") as lines:
                for line in lines:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue  # a row being written
                    if row.get("log"):
                        yield directory.name, row


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _spread(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": _quantile(values, 0.9),
    }


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One cell of the table: these games."""
    n = len(rows)
    finished = [row for row in rows if row["ended"] == "natural"]
    completed = [row["completed"] for row in rows if row["completed"] is not None]
    ends = collections.Counter(row["ended"] for row in rows)
    return {
        "games": n,
        "turns": _spread([row["turns"] for row in rows]),
        "steps": _spread([row["steps"] for row in rows]),
        "choices": _spread([row["choices"] for row in rows]),
        "steps_by_kind": {
            kind: statistics.fmean(row[kind] for row in rows) if n else None
            for kind in KINDS
        },
        "played_to_the_end": {
            "games": len(finished),
            "turns": _spread([row["turns"] for row in finished]),
            "steps": _spread([row["steps"] for row in finished]),
            "choices": _spread([row["choices"] for row in finished]),
            "steps_by_kind": {
                kind: statistics.fmean(row[kind] for row in finished)
                if finished
                else None
                for kind in KINDS
            },
        },
        "ended": {kind: ends[kind] for kind in sorted(ends)},
        "ended_in_preview": sum(row["turns"] == 0 for row in rows),
        "completed_steps": _spread(completed),
        "completed_of": len(completed),
    }


def read(corpus: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    remaining = Remaining()
    for name, row in games(corpus):
        game = game_length(row["log"])
        remaining.add(game)
        del game["left_at_requests"]
        game["format"] = name
        game["rating"] = row.get("rating")
        rows.append(game)
    for game in rows:
        if game["ended"] == "natural":
            game["completed"] = float(game["steps"])
        else:
            more = remaining.expected(game) if game["ended"] != "unfinished" else None
            game["completed"] = None if more is None else game["steps"] + more
    formats = sorted({game["format"] for game in rows})
    table: dict[str, Any] = {}
    for label, chosen in [("all formats", formats), *((f, [f]) for f in formats)]:
        for band, least in BANDS:
            cell = [
                game
                for game in rows
                if game["format"] in chosen
                and (least is None or (game["rating"] or 0) >= least)
            ]
            table[f"{label} | {band}"] = summary(cell)
    return {
        "corpus": str(corpus),
        "games": len(rows),
        "rated": sum(game["rating"] is not None for game in rows),
        "table": table,
        "remaining_by_left": {
            (
                f"{loser} v {winner}" + (" in preview" if preview else "")
            ): remaining.total[key] / remaining.count[key]
            for key in sorted(remaining.count)
            for loser, winner, preview in [key]
        },
    }


def render(reading: dict[str, Any]) -> list[str]:
    out = [
        f"{reading['games']} games of {reading['corpus']} ({reading['rated']} with a "
        "rating)",
        "",
        "mean / median / p90; steps = team preview + move requests + forced-switch "
        "requests answered; choices = the sides choosing, summed over a game's steps",
        "",
    ]

    def three(cell: dict[str, float] | None) -> str:
        if cell is None:
            return "-"
        return f"{cell['mean']:.1f} / {cell['median']:.0f} / {cell['p90']:.0f}"

    def by_kind(cell: dict[str, float | None]) -> str:
        return ", ".join(
            f"{name.replace('_', ' ')} "
            + ("-" if cell[name] is None else f"{cell[name]:.2f}")
            for name in KINDS
        )

    for label, cell in reading["table"].items():
        n = cell["games"]
        if not n:
            continue
        ends = cell["ended"]
        early = n - ends.get("natural", 0)
        done = cell["played_to_the_end"]
        kinds = cell["steps_by_kind"]
        out += [
            f"{label}: {n} games",
            f"   all games            turns {three(cell['turns'])}   steps "
            f"{three(cell['steps'])}   choices {three(cell['choices'])}",
            f"   played to the end    turns {three(done['turns'])}   steps "
            f"{three(done['steps'])}   choices {three(done['choices'])}   "
            f"({done['games']} games)",
            f"   completed (estimate) steps {three(cell['completed_steps'])}   "
            f"({cell['completed_of']} games)",
            "   a game's steps: " + by_kind(kinds),
            "   a played-out game's: " + by_kind(done["steps_by_kind"]),
            f"   ended before the end: {early} ({100 * early / n:.1f}%) -- "
            + ", ".join(
                f"{kind} {count} ({100 * count / n:.1f}%)"
                for kind, count in ends.items()
                if kind != "natural"
            )
            + f"; in team preview {cell['ended_in_preview']}",
            "",
        ]
    out.append(
        "steps still to come at a request in games played to the end, by Pokemon "
        "left (loser v winner):"
    )
    states = [f"{s}: {v:.1f}" for s, v in reading["remaining_by_left"].items()]
    out += ["   " + "  ".join(states[i : i + 8]) for i in range(0, len(states), 8)]
    return out


def main() -> int:
    args = sys.argv[1:]
    target = None
    if "--json" in args:
        at = args.index("--json")
        target = Path(args[at + 1])
        del args[at : at + 2]
    corpus = Path(args[0]) if args else ROOT / "battle_logs_feed_mc"
    reading = read(corpus)
    print("\n".join(render(reading)))
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(reading, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
