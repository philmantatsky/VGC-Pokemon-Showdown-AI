"""The time a decision takes with the reply list at 8, 12 and 16: the reading of the
null-search rehearsals pre-registered in PROJECT_STATUS.md (2026-10-10, 23:06).

A configuration was played in segments -- ``<base>/<name>_a``, ``<base>/<name>_b``:
the replay directories of ``tools/ladder_rehearse.sh``, twelve rosters each
(``<base>/<name>`` alone is read too) -- with the session's log beside each
(``<segment>.log``) and the machine's load average at every segment's start and end in
``<base>/progress.log``.

The rule set before the runs: a width fits when (a) no decision takes 9 s or more, (b)
at most 2% of decisions are not searched, (c) a planned world is lost in at most 5% of
decisions. A planned world is lost when the best-held row of the decision's ranking is
held by less than the whole planned mass: every world's table holds the bot's own pair,
so a row short of the whole mass means a world that never returned a table.

The reply list only bears on MOVE decisions. A forced switch-in (the bot's own two
actions are a pass or a switch: who comes in after a faint or a Parting Shot) is not a
table of replies: the planner searches it two turns deep until the time is up, whatever
the list's width. The two are read apart.

Usage (repo root):
  .venv/bin/python results_analysis/search_worlds_20261010/list_width_timing.py <base> \
      [--json out.json] [name ...]
"""

from __future__ import annotations

import collections
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.search_audit import summarize  # noqa: E402
from evaluation.search_ladder_read import (  # noqa: E402
    NOT_LEGAL,
    decision_kind,
    load_rows,
    quantile,
    read,
)

NAMES = ("particle8", "likely8", "likely12", "likely16")
LOGS = ("decisions.jsonl", "decisions_champion.jsonl")
LATENCY = re.compile(
    r"search latency over (\d+) decisions: p50=(\d+)ms\s+p90=(\d+)ms\s+max=(\d+)ms"
)
MARK = re.compile(r"^(START|END) (\S+) .*?\[([\d:]+)\] load \{ ([\d.]+) ")
SLOW_S, CAP_S = 7.0, 9.0
PLAYED_PRIOR = "sum75"  # the trial's own reply prior: the forecast at weight 0.75
KINDS = (("move decisions", "move"), ("forced switch-ins", "switch_in"))
MOST_UNSEARCHED, MOST_LOST = 0.02, 0.05
BUSY_LOAD = 12.0


def segments(base: Path, name: str) -> list[Path]:
    found = [base / name, *sorted(base.glob(f"{name}_*"))]
    return [path for path in found if (path / "decisions.jsonl").exists()]


def pooled(parts: list[Path], target: Path) -> Path:
    """The segments as one replay directory: their logs end to end, their replays
    linked."""
    target.mkdir(parents=True, exist_ok=True)
    for name in LOGS:
        with (target / name).open("w") as out:
            for part in parts:
                if (part / name).exists():
                    out.write((part / name).read_text())
    for part in parts:
        for replay in part.glob("*.html"):
            (target / f"{part.name} {replay.name}").symlink_to(replay.resolve())
    return target


def loads(progress: Path) -> dict[str, dict[str, Any]]:
    """Per segment: the one-minute load average and the clock at its start and end."""
    marks: dict[str, dict[str, Any]] = collections.defaultdict(dict)
    if not progress.exists():
        return {}
    for line in progress.read_text().splitlines():
        found = MARK.match(line)
        if found:
            edge, name, clock, load = found.groups()
            marks[name][edge.lower()] = {"clock": clock, "load": float(load)}
    return dict(marks)


def latency(log: Path) -> dict[str, int] | None:
    """The session's own line: the whole search call, as the player timed it."""
    found = LATENCY.search(log.read_text()) if log.exists() else None
    if not found:
        return None
    n, p50, p90, most = (int(x) for x in found.groups())
    return {"decisions": n, "p50_ms": p50, "p90_ms": p90, "max_ms": most}


def reading(base: Path, name: str, scratch: Path) -> dict[str, Any] | None:
    parts = segments(base, name)
    if not parts:
        return None
    whole = read(pooled(parts, scratch / name))
    search = whole["search"]
    rows = [
        row["exact_search"]
        for part in parts
        for row in load_rows(part / "decisions.jsonl")
        if "exact_search" in row
    ]
    turns = [
        int(row.get("turn") or 0)
        for part in parts
        for row in load_rows(part / "decisions.jsonl")
        if "exact_search" in row
    ]
    searched = [
        (audit, turn)
        for audit, turn in zip(rows, turns)
        if (audit.get("schedule") or {}).get("mode") == "search"
        and (audit.get("result") or {}).get("elapsed_s") is not None
    ]

    def spent(audit: dict) -> float:
        return float(audit["result"]["elapsed_s"]) + float(
            (audit.get("schedule") or {}).get("preparation_elapsed_s") or 0.0
        )

    def held(audit: dict) -> float:
        ranking = audit["result"].get("rankings") or []
        shares = [float(row.get("depth_coverage") or 0.0) for row in ranking]
        return max(shares, default=0.0)

    lost = [audit for audit, _turn in searched if held(audit) < 0.999]
    reasons = collections.Counter(
        str(audit["result"].get("fallback_reason"))
        for audit, _turn in searched
        if audit["result"].get("truncated")
    )
    widths = collections.Counter(
        int((audit.get("configuration") or {}).get("opponent_width") or 0)
        for audit in rows
    )
    modes = collections.Counter(
        str((audit.get("world_moves") or {}).get("mode")) for audit in rows
    )
    hidden = whole["reply_table"]["hidden"]
    ranks = whole["prior_ranks"]["brain"]
    changed = summarize([part / "decisions.jsonl" for part in parts])[
        "search_changed_champion_action"
    ]
    marks = loads(base / "progress.log")
    by_segment = []
    for part in parts:
        mark = marks.get(part.name, {})
        edge = [mark[e]["load"] for e in ("start", "end") if e in mark]
        by_segment.append(
            {
                "segment": part.name,
                "clock": [mark.get(e, {}).get("clock") for e in ("start", "end")],
                "load": edge,
                "busy": bool(edge) and max(edge) > BUSY_LOAD,
                "latency": latency(base / f"{part.name}.log"),
                "games": len(list(part.glob("*.html"))),
            }
        )
    seconds = [spent(audit) for audit, _turn in searched]
    by_kind = {}
    for name_of_kind in ("move", "switch_in"):
        chosen = [
            audit for audit, _turn in searched if decision_kind(audit) == name_of_kind
        ]
        took = [spent(audit) for audit in chosen]
        by_kind[name_of_kind] = {
            "decisions": len(chosen),
            "seconds": {
                q: quantile(took, p)
                for q, p in (("p50", 0.5), ("p75", 0.75), ("p90", 0.9), ("max", 1.0))
            },
            "at_7s_or_more": sum(s >= SLOW_S for s in took),
            "cut_short": sum(bool(a["result"].get("truncated")) for a in chosen),
            "simulated_branches_mean": (
                sum(float(a["result"].get("nodes") or 0) for a in chosen) / len(chosen)
                if chosen
                else None
            ),
        }
    decisions = int(search["decisions"])
    unsearched = decisions - int(search["modes"].get("search", 0))
    slowest_ms = max(
        (s["latency"]["max_ms"] for s in by_segment if s["latency"]), default=None
    )
    over_cap = int(search["at_9s_or_more"])
    return {
        "name": name,
        "segments": by_segment,
        "games": sum(s["games"] for s in by_segment),
        "width": dict(widths),
        "world_moves": dict(modes),
        "decisions": decisions,
        "not_searched": unsearched,
        "not_searched_why": search["reasons"],
        "seconds": {q: quantile(seconds, p) for q, p in (("p50", 0.5), ("p90", 0.9))}
        | {"max": max(seconds, default=float("nan"))},
        "at_7s_or_more": sum(s >= SLOW_S for s in seconds),
        "at_9s_or_more": over_cap,
        "slowest_as_the_player_timed_it_ms": slowest_ms,
        "by_kind": by_kind,
        "a_world_lost": len(lost),
        "mass_held_when_lost": sorted(round(held(audit), 3) for audit in lost),
        "cut_short": sum(reasons.values()),
        "cut_short_why": dict(reasons),
        "replies_hidden": int(hidden["decisions"]),
        "replies_in_table": int(hidden["in_some_world"]),
        "replies_ranked": len(ranks),
        "replies_legal_in_no_world": sum(rank >= NOT_LEGAL for rank in ranks),
        "replies_within_first": {
            str(k): sum(rank <= k for rank in whole["prior_ranks"][PLAYED_PRIOR])
            for k in (8, 12, 16)
        },
        "null_search_changed": changed,
        "fits": {
            "a_no_decision_at_9s": over_cap == 0
            and (slowest_ms is None or slowest_ms < 1000 * CAP_S),
            "b_not_searched": unsearched <= MOST_UNSEARCHED * decisions,
            "c_worlds_lost": len(lost) <= MOST_LOST * decisions,
        },
    }


def _share(part: int, whole: int) -> str:
    return f"{part} of {whole}" + (f" ({100 * part / whole:.0f}%)" if whole else "")


def render(rows: list[dict[str, Any]]) -> list[str]:
    out = []
    for row in rows:
        n = row["decisions"]
        times = row["seconds"]
        changed = row["null_search_changed"] or {}
        fits = row["fits"]
        out.append(
            f"{row['name']}: list {row['width']}, worlds {row['world_moves']}, "
            f"{row['games']} games, {n} decisions"
        )
        for part in row["segments"]:
            took = part["latency"]
            out.append(
                f"   {part['segment']}: {part['clock'][0]} to {part['clock'][1]}, load "
                f"{part['load']}"
                + (" BUSY" if part["busy"] else "")
                + (
                    f"; the player's own timing p50 {took['p50_ms']} ms, p90 "
                    f"{took['p90_ms']} ms, max {took['max_ms']} ms"
                    if took
                    else "; no latency line"
                )
            )
        out.append(
            f"   not searched {_share(row['not_searched'], n)}"
            + (f" {row['not_searched_why']}" if row["not_searched"] else "")
        )
        out.append(
            f"   seconds a decision (preparation + search): p50 {times['p50']:.1f}  "
            f"p90 {times['p90']:.1f}  max {times['max']:.1f};  7 s or more: "
            f"{_share(row['at_7s_or_more'], n)};  9 s or more: {row['at_9s_or_more']}"
        )
        for label, key in KINDS:
            cell = row["by_kind"][key]
            if not cell["decisions"]:
                continue
            took = cell["seconds"]
            out.append(
                f"   {label}: {cell['decisions']};  p50 {took['p50']:.1f}  p75 "
                f"{took['p75']:.1f}  p90 {took['p90']:.1f}  max {took['max']:.1f};  "
                f"7 s or more: {_share(cell['at_7s_or_more'], cell['decisions'])};  "
                f"cut short: {_share(cell['cut_short'], cell['decisions'])};  "
                f"simulated branches, mean {cell['simulated_branches_mean']:.0f}"
            )
        out.append(
            f"   a planned world lost: {_share(row['a_world_lost'], n)}"
            + (
                f", the mass that held: {row['mass_held_when_lost']}"
                if row["a_world_lost"]
                else ""
            )
            + f";  cut short: {_share(row['cut_short'], n)} {row['cut_short_why']}"
        )
        out.append(
            "   the real reply in the table (hidden sheets): "
            + _share(row["replies_in_table"], row["replies_hidden"])
            + ";  legal in no world: "
            + _share(row["replies_legal_in_no_world"], row["replies_ranked"])
        )
        first = row["replies_within_first"]
        out.append(
            "   the same replies within the first 8 / 12 / 16 of some world's list "
            f"({PLAYED_PRIOR}): "
            + " / ".join(_share(first[k], row["replies_ranked"]) for k in first)
        )
        out.append(
            f"   null search changed: {changed.get('changed')} of {changed.get('of')}"
        )
        verdict = "FITS" if all(fits.values()) else "DOES NOT FIT"
        failed = [rule for rule, passed in fits.items() if not passed]
        out.append(
            f"   {verdict}" + (f" (fails {', '.join(failed)})" if failed else "")
        )
        out.append("")
    return out


def main() -> int:
    args = sys.argv[1:]
    target = None
    if "--json" in args:
        at = args.index("--json")
        target = Path(args[at + 1])
        del args[at : at + 2]
    base = Path(args[0])
    names = args[1:] or list(NAMES)
    with tempfile.TemporaryDirectory() as scratch:
        rows = [
            row
            for name in names
            if (row := reading(base, name, Path(scratch))) is not None
        ]
    print("\n".join(render(rows)))
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(rows, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
