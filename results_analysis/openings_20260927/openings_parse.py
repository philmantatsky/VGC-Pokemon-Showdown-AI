"""Scratch: parse Reg M-C logs (human corpus + our ladder replays) into per-game
opening records: rosters, open sheets, brought fours, leads, turn-1/turn-2
actions, field effects, result. Output: games.jsonl (one side-pair per game).

Run from the vgc-bench repo root with .venv/bin/python.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

REPO = Path("/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench")
sys.path.insert(0, str(REPO))
from tools.ladder_loss_profile import extract_log  # noqa: E402

OUR = "antonius1"
OUT = (
    Path(
        os.environ.get(
            "OPENINGS_DATA", Path(tempfile.gettempdir()) / "openings_20260927"
        )
    )
    / "games.jsonl"
)


def sid(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def species_of(details: str) -> str:
    return details.split(",")[0].strip()


MEGA_RE = re.compile(r"-Mega(-[XYZ])?$")


def base(species: str) -> str:
    # Mega formes group with their base species
    return MEGA_RE.sub("", species)


def to_preview(species: str, preview: list[str]) -> str:
    """Map an in-battle species (Mega, re-entry) to its team-preview name."""
    b = base(species)
    if b in preview:
        return b
    for p in preview:
        if p.split("-")[0] == b.split("-")[0]:
            return p
    return b


def parse(log: str) -> dict | None:
    lines = log.split("\n")
    rec: dict = {
        "players": {},
        "rating": {},
        "preview": {"p1": [], "p2": []},
        "sheet": {"p1": {}, "p2": {}},
        "leads": {"p1": [], "p2": []},
        "brought": {"p1": [], "p2": []},
        "actions": {
            1: {"p1": [], "p2": []},
            2: {"p1": [], "p2": []},
            3: {"p1": [], "p2": []},
        },
        "cant": {1: {"p1": [], "p2": []}, 2: {"p1": [], "p2": []}},
        "mega": {"p1": None, "p2": None},
        "faints": {"p1": [], "p2": []},  # (turn, species)
        "tr": [],  # (turn, side_of_setter or None, 'start'/'end')
        "tailwind": [],  # (turn, side)
        "weather": [],  # (turn, weather, side or None, source)
        "terrain": [],
        "switches": {1: {"p1": [], "p2": []}, 2: {"p1": [], "p2": []}},
        "winner_side": None,
        "turns": 0,
        "game_no": None,
        "bo3": False,
    }
    ident: dict[str, str] = {}  # "p1a: Nick" -> species

    class _Ident(dict):
        def get(self, key, default=None):  # tolerate slot swaps (Ally Switch)
            if key in self:
                return self[key]
            nick = key.split(": ", 1)[-1]
            for k, v in self.items():
                if k[:2] == key[:2] and k.split(": ", 1)[-1] == nick:
                    return v
            return default

    ident = _Ident()
    slot_species: dict[str, str] = {}  # "p1a" -> species now
    turn = 0
    started = False
    for raw in lines:
        if not raw.startswith("|"):
            continue
        parts = raw.split("|")
        if len(parts) < 2:
            continue
        tag = parts[1]
        if tag == "player" and len(parts) >= 4 and parts[3]:
            side = parts[2]
            rec["players"][side] = parts[3]
            if len(parts) >= 6 and parts[5].strip().isdigit():
                rec["rating"][side] = int(parts[5])
        elif tag == "uhtml" and "bestof" in raw:
            m = re.search(r"Game (\d)", raw)
            if m:
                rec["game_no"] = int(m.group(1))
            rec["bo3"] = True
        elif tag == "poke" and len(parts) >= 4:
            rec["preview"][parts[2]].append(species_of(parts[3]))
        elif tag == "showteam" and len(parts) >= 4:
            side = parts[2]
            packed = "|".join(parts[3:])
            for mon in packed.split("]"):
                f = mon.split("|")
                if len(f) < 5:
                    continue
                sp = f[1] or f[0]
                rec["sheet"][side][sp] = {
                    "item": f[2],
                    "ability": f[3],
                    "moves": [m for m in f[4].split(",") if m],
                    "nature": f[5] if len(f) > 5 else "",
                }
        elif tag == "start":
            started = True
        elif tag == "turn":
            turn = int(parts[2])
            rec["turns"] = turn
        elif tag in ("switch", "drag", "replace") and len(parts) >= 4:
            who = parts[2]
            side, slot = who[:2], who[:3]
            sp = to_preview(species_of(parts[3]), rec["preview"][side])
            ident[who] = sp
            slot_species[slot] = sp
            if sp not in rec["brought"][side]:
                rec["brought"][side].append(sp)
            if started and turn == 0 and tag == "switch":
                rec["leads"][side].append(sp)
            elif turn in (1, 2) and tag == "switch":
                rec["switches"][turn][side].append(sp)
        elif tag == "-mega" and len(parts) >= 3:
            side = parts[2][:2]
            rec["mega"][side] = (
                turn,
                ident.get(parts[2], parts[3] if len(parts) > 3 else ""),
            )
        elif tag == "move" and len(parts) >= 4 and turn in (1, 2, 3):
            who = parts[2]
            side = who[:2]
            user = ident.get(who, who)
            tgt_raw = parts[4] if len(parts) > 4 else ""
            tgt = ident.get(tgt_raw, "") if tgt_raw.startswith("p") else ""
            tgt_side = tgt_raw[:2] if tgt_raw.startswith("p") else ""
            extra = "|".join(parts[5:])
            if "[from]" in extra and "lockedmove" not in extra:
                continue  # called / reflected moves are not decisions
            rec["actions"][turn][side].append(
                {
                    "user": user,
                    "move": parts[3],
                    "target": tgt,
                    "target_side": tgt_side,
                    "spread": "[spread]" in extra,
                    "miss": "[miss]" in extra,
                    "still": "[still]" in extra,
                }
            )
        elif tag == "cant" and len(parts) >= 4 and turn in (1, 2):
            who = parts[2]
            rec["cant"][turn][who[:2]].append(
                {
                    "user": ident.get(who, who),
                    "reason": parts[3],
                    "move": parts[4] if len(parts) > 4 else "",
                    "of": "|".join(parts[5:]),
                }
            )
        elif tag == "faint" and len(parts) >= 3:
            who = parts[2]
            rec["faints"][who[:2]].append((turn, ident.get(who, who)))
        elif tag == "-fieldstart" and "Trick Room" in raw:
            of = re.search(r"\[of\] (p[12])", raw)
            rec["tr"].append((turn, of.group(1) if of else None, "start"))
        elif tag == "-fieldend" and "Trick Room" in raw:
            rec["tr"].append((turn, None, "end"))
        elif tag == "-fieldstart" and "Terrain" in raw:
            of = re.search(r"\[of\] (p[12])", raw)
            rec["terrain"].append((turn, parts[2], of.group(1) if of else None))
        elif tag == "-sidestart" and "Tailwind" in raw:
            rec["tailwind"].append((turn, parts[2][:2]))
        elif tag == "-weather" and len(parts) >= 3 and "[upkeep]" not in raw:
            of = re.search(r"\[of\] (p[12])", raw)
            src = re.search(r"\[from\] ability: ([^|]+)", raw)
            rec["weather"].append(
                (
                    turn,
                    parts[2],
                    of.group(1) if of else None,
                    src.group(1) if src else "move",
                )
            )
        elif tag == "win" and len(parts) >= 3:
            name = parts[2]
            for s, n in rec["players"].items():
                if n == name:
                    rec["winner_side"] = s
    if not rec["preview"]["p1"] or not rec["leads"]["p1"] or rec["winner_side"] is None:
        return None
    # json keys must be strings
    rec["actions"] = {str(k): v for k, v in rec["actions"].items()}
    rec["cant"] = {str(k): v for k, v in rec["cant"].items()}
    rec["switches"] = {str(k): v for k, v in rec["switches"].items()}
    return rec


def load_corpus() -> dict[str, tuple[str, str]]:
    games: dict[str, tuple[str, str]] = {}
    srcs = [
        (
            "battle_logs_top_mc_merged_20260920/logs_gen9championsvgc2026regmc.json",
            "top",
        ),
        (
            "battle_logs_top_mc_merged_20260920/logs_gen9championsvgc2026regmcbo3.json",
            "top",
        ),
        ("battle_logs_top_mc_merged/logs_gen9championsvgc2026regmc.json", "top"),
        ("battle_logs_top_mc_merged/logs_gen9championsvgc2026regmcbo3.json", "top"),
        ("battle_logs_top_mc_20260920/logs_gen9championsvgc2026regmc.json", "top"),
        ("battle_logs_top_mc_20260920/logs_gen9championsvgc2026regmcbo3.json", "top"),
        ("battle_logs_players_t6like/logs_gen9championsvgc2026regmc.json", "t6like"),
        ("battle_logs_players_t6like/logs_gen9championsvgc2026regmcbo3.json", "t6like"),
        ("battle_logs_player_dksnnfud/logs_gen9championsvgc2026regmc.json", "t6like"),
    ]
    for rel, tag in srcs:
        p = REPO / rel
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for k, v in d.items():
            log = v[1] if isinstance(v, list) else v
            if k not in games or tag == "t6like":
                games[k] = (log, tag)
    return games


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_bad = 0
    with OUT.open("w") as fh:
        for key, (log, tag) in load_corpus().items():
            rec = parse(log)
            if rec is None:
                n_bad += 1
                continue
            rec["id"] = key
            rec["source"] = tag
            fh.write(json.dumps(rec) + "\n")
            n_ok += 1
        # our ladder games
        for d in sorted(REPO.glob("ladder_replays_mc_*")):
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.html")):
                log = extract_log(f)
                if not log:
                    n_bad += 1
                    continue
                rec = parse(log)
                if rec is None:
                    n_bad += 1
                    continue
                m = re.search(r"(gen9championsvgc2026regmc\w*-\d+)", f.name)
                rec["id"] = m.group(1) if m else f.name
                rec["source"] = "ours:" + d.name.replace("ladder_replays_mc_", "")
                rec["our_side"] = next(
                    (s for s, n in rec["players"].items() if sid(n) == sid(OUR)), None
                )
                fh.write(json.dumps(rec) + "\n")
                n_ok += 1
    print("parsed", n_ok, "failed", n_bad, "->", OUT)
    # newer replays (09-10 -> 09-27) fetched by the 2026-09-27 web research agent:
    # battle_logs_web_mc_20260927/top (rated 1429-1678, incl. Bo3) and /low (a
    # 15-hour mid-ladder window, 1150-1399); kept apart as games_web.jsonl
    existing = {json.loads(line)["id"] for line in OUT.open()}
    web = OUT.with_name("games_web.jsonl")
    n_web = 0
    with web.open("w") as fh:
        for sub, tag in (("top", "web_top"), ("low", "web_low")):
            for f in sorted((REPO / "battle_logs_web_mc_20260927" / sub).glob("*.log")):
                if f.stem in existing:
                    continue
                rec = parse(f.read_text(errors="replace"))
                if rec is None:
                    continue
                rec["id"] = f.stem
                rec["source"] = tag
                fh.write(json.dumps(rec) + "\n")
                n_web += 1
    print("web games", n_web, "->", web)


if __name__ == "__main__":
    main()
