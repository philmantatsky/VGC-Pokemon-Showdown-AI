"""Realized-KO counting for the guaranteed_ko promotion bound (Stage C.2).

The promoting guards claim "this pair adds a guaranteed KO" and may cross the
policy's ranking on that claim. Win rate cannot validate a rule this rare, so
we count correctness instead: of the decisions where guaranteed_ko changed the
pick, how often did an opposing Pokemon actually faint on that turn?

Reads the decision logs + replay dirs the overnight Stage-C runs produce, e.g.

    .venv/bin/python tools/count_ko_promotions.py \
        --decisions results_stage_c/ko_off_population_300_champion_policy.jsonl \
        --replays  results_stage_c/ko_off_replays/ko_off_population_300/champion_policy

Compare the off-mode arm against the robust-mode arm: the bound is promotable
when its would-promotions have a materially higher realized-KO rate without
collapsing the firing count to zero.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LOG_EMBED = re.compile(r'class="battle-log-data"[^>]*>(.*?)</script>', re.S)


def opponent_faints_by_turn(log: str, our_role: str) -> dict[int, int]:
    """turn -> count of OPPOSING faints that occurred during that turn."""
    faints: dict[int, int] = {}
    turn = 0
    for line in log.split("\n"):
        parts = line.split("|")
        if len(parts) < 2:
            continue
        if parts[1] == "turn":
            try:
                turn = int(parts[2])
            except (IndexError, ValueError):
                pass
        elif parts[1] == "faint" and len(parts) > 2:
            if parts[2][:2] != our_role:
                faints[turn] = faints.get(turn, 0) + 1
    return faints


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decisions", required=True, help="per-arm decisions .jsonl")
    ap.add_argument("--replays", required=True, help="that arm's replay directory")
    ap.add_argument("--guard", default="guaranteed_ko")
    args = ap.parse_args()

    replays: dict[str, tuple[str, str]] = {}
    for html in Path(args.replays).glob("*.html"):
        raw = html.read_text(encoding="utf-8", errors="replace")
        match = LOG_EMBED.search(raw)
        if match is None:
            continue
        log = match.group(1)
        tag = re.sub(r"^(battle-[a-z0-9]+-\d+).*$", r"\1", html.stem.split(" - ")[-1])
        username = html.name.split(" - ")[0]
        role_match = re.search(rf"\|player\|(p[12])\|{re.escape(username)}\|", log)
        if role_match:
            replays[tag] = (log, role_match.group(1))

    fired = realized = matched = 0
    per_turn_hits: list[dict] = []
    with Path(args.decisions).open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            guards = row.get("guards") or {}
            # audit shape: {"demotions": {...}, "stages": [...], "vetoed": [...]}
            # -- a promotion that changed the pick appears in "stages"
            stages = (
                guards.get("stages", [])
                if isinstance(guards, dict)
                else guards
                if isinstance(guards, list)
                else []
            )
            if not any(args.guard in str(stage) for stage in stages):
                continue
            fired += 1
            tag = row.get("battle", "")
            turn = row.get("turn")
            if tag not in replays or not isinstance(turn, int):
                continue
            matched += 1
            log, role = replays[tag]
            faints = opponent_faints_by_turn(log, role)
            hit = faints.get(turn, 0) > 0
            realized += hit
            per_turn_hits.append({"battle": tag, "turn": turn, "realized": hit})

    print(f"decisions where {args.guard} changed the pick: {fired}")
    print(f"matched to a replay turn: {matched}")
    if matched:
        print(
            f"realized (an opposing faint that turn): {realized}/{matched} "
            f"= {realized / matched * 100:.1f}%"
        )
    out = Path(args.decisions).with_suffix(".ko_counting.json")
    out.write_text(
        json.dumps(
            {
                "guard": args.guard,
                "fired": fired,
                "matched": matched,
                "realized": realized,
                "details": per_turn_hits,
            },
            indent=1,
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
