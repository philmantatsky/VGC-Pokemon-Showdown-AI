"""Scrape replays from top-ranked ladder players only.

The bundled scraper (vgc_bench/scrape_logs.py) walks the public replay feed, which is
dominated by low-rated play (median ~1183). This targets the top-N ladder instead, so
behaviour cloning learns from strong players rather than the ladder average.

    python scrape_top_players.py --measure  # count availability, write nothing
    python scrape_top_players.py --top 500          # scrape into battle_logs_top/
"""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FORMATS = [
    "gen9championsvgc2026regma",
    "gen9championsvgc2026regmabo3",
    "gen9championsvgc2026regmb",
    "gen9championsvgc2026regmbbo3",
]
UA = {"User-Agent": "vgc-bench-research/0.1"}
OUT_DIR = Path("battle_logs_top")


def get(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.load(r)
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def ladder_top(fmt, n):
    """Top-n usernames on this format's ladder, best first."""
    d = get(f"https://pokemonshowdown.com/ladder/{fmt}.json") or {}
    return [
        (e.get("username", ""), e.get("elo", 0)) for e in (d.get("toplist") or [])[:n]
    ]


def user_replay_ids(user, fmt):
    """All public replay ids for this user in this format, following pagination."""
    ids, before = [], None
    while True:
        url = (
            f"https://replay.pokemonshowdown.com/search.json"
            f"?user={urllib.parse.quote(user)}&format={fmt}"
        )
        if before:
            url += f"&before={before}"
        page = get(url)
        if not isinstance(page, list) or not page:
            break
        ids += [r["id"] for r in page if "id" in r]
        if len(page) < 51:
            break
        before = min(r["uploadtime"] for r in page if "uploadtime" in r)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=500, help="how many ladder players")
    ap.add_argument("--measure", action="store_true", help="count only, write nothing")
    ap.add_argument("--workers", type=int, default=6, help="concurrent requests")
    args = ap.parse_args()

    grand_ids, grand_elo = {}, {}
    for fmt in FORMATS:
        players = ladder_top(fmt, args.top)
        if not players:
            print(f"{fmt}: no ladder (format may be inactive)")
            continue
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(lambda p: user_replay_ids(p[0], fmt), players))
        ids = {i for lst in results for i in lst}
        have = sum(1 for r in results if r)
        elos = [e for _, e in players]
        grand_ids[fmt] = ids
        grand_elo[fmt] = elos
        print(
            f"{fmt}:\n"
            f"    players {len(players)} (elo {min(elos):.0f}-{max(elos):.0f}, "
            f"median {sorted(elos)[len(elos) // 2]:.0f})\n"
            f"    with >=1 replay: {have}/{len(players)}\n"
            f"    unique replays:  {len(ids):,}"
        )

    total = len({i for s in grand_ids.values() for i in s})
    print(f"\nTOTAL unique replays across formats: {total:,}")
    if args.measure:
        print("(measure only - nothing written)")
        return

    OUT_DIR.mkdir(exist_ok=True)
    for fmt, ids in grand_ids.items():
        if not ids:
            continue
        ids = sorted(ids)
        logs = {}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            fetched = list(
                ex.map(
                    lambda i: (i, get(f"https://replay.pokemonshowdown.com/{i}.json")),
                    ids,
                )
            )
        for ident, d in fetched:
            if d and d.get("log"):
                logs[ident] = (d.get("uploadtime", 0), d["log"])
        path = OUT_DIR / f"logs_{fmt}.json"
        with path.open("w") as f:
            json.dump(logs, f)
        print(f"wrote {len(logs):,} logs -> {path}")


if __name__ == "__main__":
    main()
