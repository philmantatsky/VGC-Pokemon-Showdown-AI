"""Build JOINT set distributions for Reg M-B from observed team sheets.

Smogon usage stats give marginals: Kingambit runs Black Glasses 40% of the time and
Swords Dance 26% of the time, independently. That cannot answer the question that
actually matters mid-battle -- "given it just used Swords Dance, what item is it
holding?" -- because item and moves are correlated.

Every |showteam| line in our replay corpus is one complete observed set, so counting
them gives the joint distribution directly. This is the same idea as the Random Battle
joint_sets file the Laplace bot uses, built from VGC replays instead.

    python build_joint_sets.py            # writes data/joint_sets_regmb.json
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SOURCES = ["battle_logs", "battle_logs_top"]


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def parse_showteam(line):
    """Parse one ``|showteam|`` protocol line into competitive sets."""
    parts = line.split("|", 3)
    if len(parts) < 4:
        return
    for entry in parts[3].split("]"):
        f = entry.split("|")
        if len(f) < 5 or not f[0].strip():
            continue
        species, item, ability, movestr = f[0], f[2], f[3], f[4]
        moves = tuple(sorted(norm(m) for m in movestr.split(",") if m.strip()))
        if not moves:
            continue
        yield norm(species), norm(item), norm(ability), moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="*regmb*.json")
    ap.add_argument("--out", default="data/joint_sets_regmb.json")
    ap.add_argument(
        "--min_count",
        type=int,
        default=2,
        help="drop sets seen fewer than this many times (noise/typos)",
    )
    args = ap.parse_args()

    joint = defaultdict(Counter)
    seen_logs = 0
    for d in SOURCES:
        for f in Path(d).glob(args.pattern):
            for tag, (ts, log) in json.load(f.open()).items():
                seen_logs += 1
                for line in log.split("\n"):
                    if line.startswith("|showteam|"):
                        for sp, item, ability, moves in parse_showteam(line):
                            joint[sp][(item, ability, moves)] += 1

    out = {}
    dropped = 0
    for sp, counter in joint.items():
        rows = []
        total = sum(counter.values())
        for (item, ability, moves), n in counter.most_common():
            if n < args.min_count:
                dropped += 1
                continue
            rows.append(
                {
                    "item": item or None,
                    "ability": ability or None,
                    "moves": list(moves),
                    "count": n,
                    "prob": round(n / total, 4),
                }
            )
        if rows:
            out[sp] = {"total": total, "sets": rows}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)

    print(f"scanned {seen_logs:,} logs")
    print(f"wrote {len(out)} species -> {args.out}  ({dropped} rare sets dropped)")
    for sp in ("kingambit", "garchomp", "incineroar"):
        if sp in out:
            e = out[sp]
            print(f"\n{sp} ({e['total']} observed, {len(e['sets'])} distinct sets):")
            for r in e["sets"][:3]:
                print(
                    f"   {r['prob']:.0%}  {r['item']:16} {r['ability']:14} "
                    f"{','.join(r['moves'])}"
                )


if __name__ == "__main__":
    main()
