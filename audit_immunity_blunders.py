"""Count how often WE fire a move into something immune to it, from saved replays.

This is the primary metric the plan asked for and win rate is not: a direct count of
the exact blunder observed on ladder (Dragon Claw into a Fairy). Showdown states it
outright with |-immune|, so no damage model or inference is needed -- the server has
already adjudicated.

    python audit_immunity_blunders.py [--since HHMM]

Attribution is by species, not by player slot: which of p1/p2 we are varies per battle,
so the script finds the side fielding our team and counts only that side's moves.
"""

import argparse
import re
import sys
import time
from pathlib import Path

REPLAYS = Path("ladder_replays")
OUR_TEAM_FILE = Path("teams/reg_mb/our_team.txt")

MOVE_RE = re.compile(r"\|move\|(p[12])[ab]: ([^|]+)\|([^|]+)\|(p[12])[ab]: ([^|]+)")
IMMUNE_RE = re.compile(r"\|-immune\|(p[12])[ab]: ([^|]+)")


def our_species() -> set[str]:
    """Species names only -- every line with an '@' is a Pokemon header."""
    return {
        line.split("@")[0].strip().lower()
        for line in OUR_TEAM_FILE.read_text().splitlines()
        if "@" in line and not line.startswith(("-", " ", "\t"))
    }


def side_that_is_ours(text: str, ours: set[str]) -> str | None:
    """Which player id fields our team. Varies battle to battle."""
    score = {"p1": 0, "p2": 0}
    for m in re.finditer(r"\|(?:switch|drag|replace)\|(p[12])[ab]: ([^|,]+)", text):
        side, name = m.group(1), m.group(2).strip().lower()
        if any(name.startswith(s.split("-")[0]) for s in ours):
            score[side] += 1
    if score["p1"] == score["p2"]:
        return None
    return max(score, key=lambda k: score[k])


def audit(path: Path, ours: set[str]):
    text = path.read_text(errors="ignore")
    me = side_that_is_ours(text, ours)
    if me is None:
        return None
    lines = text.splitlines()
    blunders, our_moves = [], 0
    for i, line in enumerate(lines):
        m = MOVE_RE.search(line)
        if not m:
            continue
        actor, _actor_name, move, tgt_side, tgt_name = m.groups()
        if actor != me:
            continue
        our_moves += 1
        # Scan this move's whole resolution block. A spread move nullified against ONE
        # foe while damaging the other is correct play, not a blunder -- so the test is
        # "something was immune AND nothing on their side took damage", not merely the
        # presence of |-immune|. Counting every |-immune| flagged Earthquake hitting a
        # Flying foe alongside a grounded one, which is exactly the play you want.
        immune_to, damaged = [], False
        for nxt in lines[i + 1 : i + 8]:
            if "|move|" in nxt or "|turn|" in nxt or "|upkeep" in nxt:
                break
            im = IMMUNE_RE.search(nxt)
            if im and im.group(1) != me:
                immune_to.append(im.group(2))
            dm = re.search(r"\|-damage\|(p[12])[ab]:", nxt)
            if dm and dm.group(1) != me:
                damaged = True
        if immune_to and not damaged:
            blunders.append(f"{move} -> {', '.join(immune_to)}")
    return me, our_moves, blunders


def main():
    ap = argparse.ArgumentParser()
    # "MM-DD HH:MM" rather than bare HHMM: ladder sessions run past midnight, and a
    # time-only cutoff silently resolves to a moment in the future once the date rolls.
    ap.add_argument(
        "--since",
        default=None,
        help='"MM-DD HH:MM"; only audit replays modified after it',
    )
    args = ap.parse_args()

    cutoff = None
    if args.since:
        t = time.strptime(args.since, "%m-%d %H:%M")
        cutoff = time.mktime(
            (
                time.localtime().tm_year,
                t.tm_mon,
                t.tm_mday,
                t.tm_hour,
                t.tm_min,
                0,
                0,
                0,
                -1,
            )
        )

    ours = our_species()
    print(f"our team: {sorted(ours)}")
    files = sorted(REPLAYS.glob("*.html"), key=lambda p: p.stat().st_mtime)
    if cutoff:
        files = [f for f in files if f.stat().st_mtime >= cutoff]
    if not files:
        print("no replays in range")
        return

    total_moves = total_blunders = 0
    all_blunders = []
    for f in files:
        res = audit(f, ours)
        if res is None:
            print(f"  ? {f.name[:60]}: could not identify our side")
            continue
        _me, moves, blunders = res
        total_moves += moves
        total_blunders += len(blunders)
        all_blunders += blunders
        flag = "  <-- " + ", ".join(blunders) if blunders else ""
        print(
            f"  {f.name.split('-')[-1][:14]:16} {moves:3d} moves, "
            f"{len(blunders)} immune{flag}"
        )

    print(
        f"\n{len(files)} battles | {total_moves} of our moves | "
        f"{total_blunders} fired into an immunity"
    )
    if total_moves:
        print(f"blunder rate: {total_blunders / total_moves:.2%} of our moves")
    if all_blunders:
        print("\nby move:")
        from collections import Counter

        for k, v in Counter(all_blunders).most_common():
            print(f"  {v:3d}x  {k}")


if __name__ == "__main__":
    sys.exit(main())
