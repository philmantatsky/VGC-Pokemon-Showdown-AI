"""Read a search head-to-head that ran as several arms (2026-10-04, V5).

Each arm is a set of shards written by tools/search_mirror_rounds.sh
(``results_mirror_<name>_r<round>_s<shard>``, each a run of
evaluation/mirror_guard_ab.py). Only complete shards count. For every arm: side A's
pooled record with its Wilson 95% interval, the open- and hidden-sheet halves, and what
the search did (decisions searched, at the time budget, planned on a single random
stream, overrides and guard send-backs, the edge it expected and the edge against the
reply that came). For every ``--factor``: the two pooled groups and the difference of
their rates with a 95% interval (Newcombe's hybrid score interval).

The reading is fixed before the run, not here: an arm wins when its lower bound is
above 50%, loses when its upper bound is below; the factors say which ingredient moved
the rate.

Usage (from the repo root):
    .venv/bin/python evaluation/search_arms_report.py --root ../vgc-bench-v5 \\
        raw=search_nash5 cal=search_nash5cal s4=search_nash5s4 \\
        s4cal=search_nash5s4cal \\
        --factor "streams 4 vs 1=s4,s4cal:raw,cal" \\
        --factor "calibrated vs raw=cal,s4cal:raw,s4" --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.mirror_guard_ab import wilson  # noqa: E402
from evaluation.search_audit import summarize  # noqa: E402

# a decision whose planner used this much of its 8 s is at the budget (the planner
# returns by 7.35 s; the ``truncated`` flag under-counts in nash mode)
AT_BUDGET_S = 7.0


def complete_shards(root: Path, name: str) -> list[Path]:
    """The arm's finished shards, in round and shard order."""
    found = []
    for run in root.glob(f"results_mirror_{name}_r*_s*"):
        tail = run.name[len(f"results_mirror_{name}_r") :]
        round_text, _, shard_text = tail.partition("_s")
        if not (run.is_dir() and round_text.isdigit() and shard_text.isdigit()):
            continue  # another arm sharing the prefix, or a shard set aside
        try:
            done = json.loads((run / "result.json").read_text()).get("complete")
        except (OSError, ValueError):
            done = False
        if done:
            found.append((int(round_text), int(shard_text), run))
    return [run for _round, _shard, run in sorted(found)]


def record(wins: float, games: int) -> dict:
    low, high = wilson(wins, games) if games else (0.0, 1.0)
    return {
        "a_wins": wins,
        "games": games,
        "a_win_rate": wins / games if games else None,
        "wilson_95": [low, high],
    }


def verdict(row: dict) -> str:
    low, high = row["wilson_95"]
    if not row["games"]:
        return "no games"
    return (
        "wins" if low > 0.5 else "loses" if high < 0.5 else "no detectable difference"
    )


def difference(first: dict, second: dict) -> dict:
    """``first`` minus ``second`` with Newcombe's 95% interval for two rates."""
    if not first["games"] or not second["games"]:
        return {"difference": None, "interval_95": None}
    p1, p2 = first["a_win_rate"], second["a_win_rate"]
    (l1, u1), (l2, u2) = first["wilson_95"], second["wilson_95"]
    delta = p1 - p2
    return {
        "difference": delta,
        "interval_95": [
            delta - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            delta + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2),
        ],
    }


def results(runs: list[Path]) -> dict:
    """Side A's record over the shards: overall, by sheet visibility, by round."""
    totals = {"all": [0.0, 0], "open": [0.0, 0], "hidden": [0.0, 0]}
    rounds: dict[str, list[float]] = {}
    for run in runs:
        blocks = json.loads((run / "result.json").read_text())["blocks"]
        round_key = run.name.rsplit("_r", 1)[1].split("_s")[0]
        for block in blocks:
            sheet = "hidden" if block["hidden_sheets"] else "open"
            for key in ("all", sheet):
                totals[key][0] += block["a_wins"]
                totals[key][1] += block["games"]
            row = rounds.setdefault(round_key, [0.0, 0])
            row[0] += block["a_wins"]
            row[1] += block["games"]
    out = {key: record(wins, int(games)) for key, (wins, games) in totals.items()}
    out["by_round"] = {
        key: [wins, int(games)]
        for key, (wins, games) in sorted(rounds.items(), key=lambda kv: int(kv[0]))
    }
    return out


def search_stats(runs: list[Path]) -> dict:
    """What side A's search did, from the shards' decision logs."""
    logs = [run / "a_decisions.jsonl" for run in runs]
    logs = [path for path in logs if path.exists()]
    searched = at_budget = one_stream = 0
    elapsed: list[float] = []
    for path in logs:
        with path.open() as handle:
            for line in handle:
                audit = (json.loads(line) if line.strip() else {}).get("exact_search")
                if not audit:
                    continue
                schedule = audit.get("schedule") or {}
                if schedule.get("mode") != "search":
                    continue
                searched += 1
                seconds = float((audit.get("result") or {}).get("elapsed_s") or 0.0)
                elapsed.append(seconds)
                at_budget += seconds >= AT_BUDGET_S
                streams = int(schedule.get("planning_roots") or 0) * int(
                    schedule.get("chance_samples") or 1
                )
                one_stream += streams <= 1
    elapsed.sort()
    audit = summarize(logs) if logs else {}
    outcomes = audit.get("champion_outcomes") or {}
    decisions = int(audit.get("decisions") or 0)
    modes = audit.get("modes") or {}

    def share(count: float, of: float) -> float | None:
        return count / of if of else None

    return {
        "decisions": decisions,
        "searched": searched,
        "not_searched": {k: v for k, v in modes.items() if k != "search"},
        "seconds_median": elapsed[len(elapsed) // 2] if elapsed else None,
        "seconds_p90": elapsed[int(0.9 * len(elapsed))] if elapsed else None,
        "at_budget_share": share(at_budget, searched),
        "one_stream_share": share(one_stream, searched),
        "override_share": share(outcomes.get("overridden", 0), searched),
        "sent_back_by_guards_share": share(
            outcomes.get("override_vetoed", 0), searched
        ),
        "override_edges": audit.get("override_edges"),
        "reply_was_searched": (audit.get("opponent_reply_was_searched") or {}).get(
            "in_some_world_by_sheets"
        ),
    }


def parse_factor(text: str) -> tuple[str, list[str], list[str]]:
    """``label=a,b:c,d`` -> (label, [a, b], [c, d])."""
    label, _, groups = text.partition("=")
    first, _, second = groups.partition(":")
    names = [
        [n.strip() for n in side.split(",") if n.strip()] for side in (first, second)
    ]
    if not label.strip() or not names[0] or not names[1]:
        raise ValueError(f"--factor needs 'label=a,b:c,d', got {text!r}")
    return label.strip(), names[0], names[1]


def report(root: Path, arms: dict[str, str], factors: list[str]) -> dict:
    out: dict = {"root": str(root), "arms": {}, "factors": {}}
    for label, name in arms.items():
        runs = complete_shards(root, name)
        row = results(runs)
        out["arms"][label] = {
            "name": name,
            "shards": len(runs),
            **row["all"],
            "verdict": verdict(row["all"]),
            "open_sheets": row["open"],
            "hidden_sheets": row["hidden"],
            "by_round": row["by_round"],
            "search": search_stats(runs),
        }

    def pooled(labels: list[str]) -> dict:
        missing = [label for label in labels if label not in out["arms"]]
        if missing:
            raise ValueError(f"--factor names unknown arms: {missing}")
        rows = [out["arms"][label] for label in labels]
        return record(sum(r["a_wins"] for r in rows), sum(r["games"] for r in rows))

    for text in factors:
        label, first, second = parse_factor(text)
        a, b = pooled(first), pooled(second)
        out["factors"][label] = {
            "first": {"arms": first, **a},
            "second": {"arms": second, **b},
            **difference(a, b),
        }
    everything = record(
        sum(arm["a_wins"] for arm in out["arms"].values()),
        sum(arm["games"] for arm in out["arms"].values()),
    )
    out["all_arms"] = {**everything, "verdict": verdict(everything)}
    return out


def _percent(value: float | None) -> str:
    return "  n/a" if value is None else f"{100 * value:5.1f}%"


def _line(row: dict) -> str:
    low, high = row["wilson_95"]
    return (
        f"{row['a_wins']:7.1f} / {row['games']:<5d} = {_percent(row['a_win_rate'])} "
        f"[{100 * low:.1f}, {100 * high:.1f}]"
    )


def render(summary: dict) -> str:
    lines = []
    for label, arm in summary["arms"].items():
        search = arm["search"]
        lines.append(
            f"{label:8s} {_line(arm)}  {arm['verdict']}  ({arm['shards']} shards)"
        )
        lines.append(
            f"         open {_line(arm['open_sheets'])} | hidden "
            f"{_line(arm['hidden_sheets'])}"
        )
        edges = search["override_edges"] or {}
        lines.append(
            f"         searched {search['searched']} decisions: median "
            f"{search['seconds_median'] or 0:.2f} s, at budget "
            f"{_percent(search['at_budget_share'])}, one stream "
            f"{_percent(search['one_stream_share'])}, overrides "
            f"{_percent(search['override_share'])}, sent back "
            f"{_percent(search['sent_back_by_guards_share'])}, edge expected "
            f"{edges.get('mean_expected')} realized {edges.get('mean_realized')}"
        )
    lines.append(
        f"all arms {_line(summary['all_arms'])}  {summary['all_arms']['verdict']}"
    )
    for label, factor in summary["factors"].items():
        if factor["difference"] is None:
            lines.append(f"{label}: no games")
            continue
        low, high = factor["interval_95"]
        lines.append(
            f"{label}: {_percent(factor['first']['a_win_rate'])} "
            f"({factor['first']['games']} games) vs "
            f"{_percent(factor['second']['a_win_rate'])} "
            f"({factor['second']['games']}) -> "
            f"{100 * factor['difference']:+.1f} points "
            f"[{100 * low:+.1f}, {100 * high:+.1f}]"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", nargs="+", help="label=name (results_mirror_<name>_r*_s*)")
    ap.add_argument(
        "--root", type=Path, default=ROOT, help="the checkout that ran them"
    )
    ap.add_argument("--factor", action="append", default=[], help="label=a,b:c,d")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    arms = {}
    for text in args.arms:
        label, _, name = text.partition("=")
        if not label or not name:
            raise ValueError(f"an arm is label=name, got {text!r}")
        arms[label] = name
    summary = report(args.root, arms, args.factor)
    if args.json is not None:
        args.json.write_text(json.dumps(summary, indent=2) + "\n")
    print(render(summary))


if __name__ == "__main__":
    main()
