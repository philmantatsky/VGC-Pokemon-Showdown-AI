"""Build a most-likely-set prior for Reg M-B from Smogon usage stats.

Why this exists: Reg M-B's Open Team Sheets rule is opt-in (data/rulesets.ts:1979 shows
an accept/deny button), so on ladder an opponent can refuse. Training ran with sheets
always on, so every opponent slot the model ever saw was populated. When sheets are
denied, the encoder falls back to "null"/"no move" sentinels -- inputs the model has
barely trained against.

Filling those unknown slots with the statistically likely set instead keeps the
observation closer to the training distribution. It does NOT change the observation
shape, so existing checkpoints stay valid.

    python build_movesets.py                 # writes data/movesets_regmb.json
    python build_movesets.py --month 2026-06 --cutoff 1630
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

URL = "https://www.smogon.com/stats/{month}/moveset/{fmt}-{cutoff}.txt"
OUT = Path("data/movesets_regmb.json")


def fetch(month, fmt, cutoff):
    url = URL.format(month=month, fmt=fmt, cutoff=cutoff)
    req = urllib.request.Request(url, headers={"User-Agent": "vgc-bench-research/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def parse(text):
    """Smogon moveset dumps are | delimited blocks per species."""
    entries = {}
    blocks = text.split("+----------------------------------------+")
    species = None
    section = None
    cur = {"abilities": [], "items": [], "moves": [], "spreads": []}

    def flush():
        if species and (cur["moves"] or cur["abilities"]):
            entries[species] = {
                "ability": cur["abilities"][0][0] if cur["abilities"] else None,
                "item": cur["items"][0][0] if cur["items"] else None,
                # top 4 by usage, skipping the aggregate "Other" bucket
                "moves": [m for m, _ in cur["moves"][:4]],
                "spread": cur["spreads"][0][0] if cur["spreads"] else None,
                "ability_pct": cur["abilities"][0][1] if cur["abilities"] else 0.0,
                "move_pcts": {m: p for m, p in cur["moves"][:6]},
            }

    for block in blocks:
        for raw in block.splitlines():
            line = raw.strip().strip("|").strip()
            if not line:
                continue
            if line in (
                "Abilities",
                "Items",
                "Moves",
                "Spreads",
                "Teammates",
                "Checks and Counters",
            ):
                section = line
                continue
            m = re.match(r"^(.*?)\s+([\d.]+)%$", line)
            if m and section in ("Abilities", "Items", "Moves", "Spreads"):
                name, pct = m.group(1).strip(), float(m.group(2))
                if name == "Other":
                    continue
                cur[section.lower()].append((name, pct))
                continue
            if line.startswith(("Raw count", "Avg. weight", "Viability Ceiling")):
                continue
            if not m and section is None or (not m and line and " " not in line[:0]):
                # a bare name line starts a new species block
                if re.match(r"^[A-Z][A-Za-z0-9'\-. :]*$", line) and "%" not in line:
                    flush()
                    species = line
                    section = None
                    cur = {"abilities": [], "items": [], "moves": [], "spreads": []}
    flush()
    return entries


def norm(name):
    """Smogon display name -> poke-env style id (lowercase alphanumeric)."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2026-07")
    ap.add_argument("--fmt", default="gen9championsvgc2026regmb")
    ap.add_argument(
        "--cutoff", default="1760", help="Elo cutoff; higher = stronger play"
    )
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    print(f"fetching {args.fmt} {args.month} cutoff {args.cutoff}...")
    text = fetch(args.month, args.fmt, args.cutoff)
    entries = parse(text)
    print(f"parsed {len(entries)} species")

    out = {}
    for sp, d in entries.items():
        out[norm(sp)] = {
            "display": sp,
            "ability": norm(d["ability"]) if d["ability"] else None,
            "item": norm(d["item"]) if d["item"] else None,
            "moves": [norm(m) for m in d["moves"]],
            "spread": d["spread"],
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"wrote {len(out)} entries -> {args.out}")

    for k in ("kingambit", "garchomp", "incineroar"):
        if k in out:
            e = out[k]
            print(f"  {e['display']:16} {e['ability']:16} {e['item']:16} {e['moves']}")


if __name__ == "__main__":
    main()
