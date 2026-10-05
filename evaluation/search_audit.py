"""Summarize live exact-search audits (2026-10-04, the matrix search): how many move
decisions were really searched, how often search fell back to the champion and why,
search latency, and how often the searched choice differed from the policy's own
favourite among the searched candidates.

Reads the ``exact_search`` rows that PolicyPlayer appends to its decision log
(evaluation/mirror_guard_ab.py --a-search writes <output>/a_decisions.jsonl).
Usage (from the repo root):
    .venv/bin/python evaluation/search_audit.py <a_decisions.jsonl> [...]
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def summarize(paths: list[Path]) -> dict:
    modes: collections.Counter[str] = collections.Counter()
    reasons: collections.Counter[str] = collections.Counter()
    elapsed: list[float] = []
    changed = searched = 0
    changed_guarded = with_guarded = guard_changed = 0
    # against the pair the bot would have submitted with search off (logged from
    # 2026-10-04 as champion_actions; a fallback plays that pair by construction)
    with_champion = changed_champion = 0
    champion_change_kinds: collections.Counter[str] = collections.Counter()
    champion_outcomes: collections.Counter[str] = collections.Counter()
    # was the opponent's actual reply among those the previous search considered?
    reply_any = reply_n = 0
    reply_mass = 0.0
    reply_by_sheets = {"open": [0, 0], "hidden": [0, 0]}  # [decisions, in some world]
    # overrides: the edge the search expected, and the edge in the cells of the reply
    # the opponent really made (only where that reply had been searched)
    expected_edges: list[float] = []
    realized_edges: list[float] = []
    realized_of_expected: list[float] = []
    overrides_followed = 0
    stages: collections.Counter[str] = collections.Counter()
    weights: list[float] = []
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line) if line.strip() else {}
            audit = row.get("exact_search")
            if not audit:
                continue
            schedule = audit.get("schedule") or {}
            mode = schedule.get("mode", "?")
            modes[mode] += 1
            coverage = audit.get("reply_coverage")
            if coverage:
                reply_n += 1
                reply_any += int(bool(coverage.get("any")))
                reply_mass += float(coverage.get("mass_with_reply") or 0.0)
                sheets = reply_by_sheets[
                    "open" if audit.get("open_sheet") else "hidden"
                ]
                sheets[0] += 1
                sheets[1] += int(bool(coverage.get("any")))
                override = coverage.get("override")
                if override:
                    overrides_followed += 1
                    if override.get("expected_edge") is not None:
                        expected_edges.append(float(override["expected_edge"]))
                    if override.get("realized_edge") is not None:
                        realized_edges.append(float(override["realized_edge"]))
                        if override.get("expected_edge") is not None:
                            realized_of_expected.append(
                                float(override["expected_edge"])
                            )
            champion = audit.get("champion_actions")
            if champion is not None:
                with_champion += 1
                played = audit.get("actions")
                if mode == "search" and played is not None and played != champion:
                    changed_champion += 1
                    live = schedule.get("live_guards") or {}
                    chosen = (audit.get("result") or {}).get("choice")
                    if live.get("policy_choice") not in (None, chosen):
                        kind = "the_search_scores"
                    elif live.get("changed_pick"):
                        kind = "guards_on_the_searched_ranking"
                    else:
                        kind = "rebuilt_view_or_candidate_set"
                    champion_change_kinds[kind] += 1
            if mode != "search":
                for reason in schedule.get("reasons") or ["?"]:
                    reasons[f"{mode}:{reason}"] += 1
                continue
            result = audit.get("result") or {}
            searched += 1
            if result.get("elapsed_s") is not None:
                elapsed.append(float(result["elapsed_s"]))
            rankings = result.get("rankings") or []
            if rankings:
                weights.append(float(rankings[0].get("score", 0.0)))
                favourite = max(rankings, key=lambda r: float(r.get("prior", 0.0)))
                changed += int(favourite.get("choice") != result.get("choice"))
            live_guards = schedule.get("live_guards") or {}
            guard_changed += int(bool(live_guards.get("changed_pick")))
            if live_guards.get("champion"):
                champion_outcomes[live_guards["champion"]] += 1
            stages.update(live_guards.get("stages") or [])
            # the policy's pick after the same guards (logged from 2026-10-04): what
            # the bot would have played without the search's scores
            if live_guards.get("policy_choice") is not None:
                with_guarded += 1
                changed_guarded += int(
                    live_guards["policy_choice"] != result.get("choice")
                )
    elapsed.sort()

    def pct(q: float) -> float | None:
        return (
            elapsed[min(len(elapsed) - 1, int(q * len(elapsed)))] if elapsed else None
        )

    return {
        "decisions": sum(modes.values()),
        "modes": dict(modes),
        "fallback_reasons": dict(reasons.most_common(12)),
        "searched": searched,
        "search_changed_policy_favourite": changed,
        "search_changed_guarded_policy_choice": (
            {"changed": changed_guarded, "of": with_guarded} if with_guarded else None
        ),
        "search_changed_champion_action": (
            {
                "changed": changed_champion,
                "of": with_champion,
                "by": dict(champion_change_kinds),
            }
            if with_champion
            else None
        ),
        # kept: the search agreed with the bot's own pair; overridden: it replaced it
        # and the guards accepted; override_vetoed: the guards sent it back
        "champion_outcomes": dict(champion_outcomes) or None,
        "override_edges": (
            {
                "overrides_followed": overrides_followed,
                "mean_expected": _mean(expected_edges),
                "reply_was_searched": len(realized_edges),
                "mean_realized": _mean(realized_edges),
                "mean_expected_where_realized": _mean(realized_of_expected),
                "realized_not_positive": sum(e <= 0 for e in realized_edges),
            }
            if overrides_followed
            else None
        ),
        "opponent_reply_was_searched": (
            {
                "decisions": reply_n,
                "in_some_world": reply_any / reply_n,
                "mean_world_mass": reply_mass / reply_n,
                # hidden sheets: a move the opponent has not shown yet is in no world
                # unless the sampled set happens to hold it
                "in_some_world_by_sheets": {
                    name: (hit / count if count else None)
                    for name, (count, hit) in reply_by_sheets.items()
                },
            }
            if reply_n
            else None
        ),
        "guards_changed_searched_pick": guard_changed,
        "guard_stages_on_searched_picks": dict(stages.most_common(12)),
        "elapsed_s": {
            "p50": pct(0.5),
            "p90": pct(0.9),
            "max": elapsed[-1] if elapsed else None,
        },
        "mean_weight_of_chosen": sum(weights) / len(weights) if weights else None,
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


if __name__ == "__main__":
    print(json.dumps(summarize([Path(p) for p in sys.argv[1:]]), indent=2))
