"""Build a regulation team pool from open-team-sheet replays.

Why (2026-09-09): Reg M-C went live with no VGCPastes sheet yet, so the only
source of real M-C teams is the ``|showteam|`` line that open-team-sheet
games (all Bo3 games, opt-in Bo1 games) put in their logs. The packed sheet
carries species, item, ability, moves, nature, gender and level but no EV
spread (Champions sheets do not show spreads), so spreads are filled by a
nature-based rule in Champions units (cap 32/stat, budget 66): the nature's
boosted offensive stat + Speed at 32 each (HP instead of Speed for
Speed-lowering natures), remaining 2 to HP. Every team is validated with the
bundled simulator's batch validator; invalid ones are skipped.

Usage (from the repo root):
  .venv/bin/python datagen/extract_ots_teams.py --logs <logs_*.json> \\
      --format gen9championsvgc2026regmc --out teams/reg_mc --prefix MC
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from poke_env.data import GenData  # noqa: E402

ROOT = _Path(__file__).resolve().parents[1]
PHYSICAL = {"Adamant", "Jolly", "Brave", "Lonely", "Naughty", "Impish", "Careful"}
SPECIAL = {"Modest", "Timid", "Quiet", "Mild", "Rash", "Bold", "Calm"}
SPEED_DOWN = {"Brave", "Quiet", "Relaxed", "Sassy"}


def _spread(nature: str, moves: list[str], gen: GenData) -> str:
    """Champions-unit EV line from the nature and the move categories."""
    if nature in PHYSICAL:
        offense = "Atk"
    elif nature in SPECIAL:
        offense = "SpA"
    else:
        cats = [gen.moves.get(m, {}).get("category", "") for m in moves]
        offense = "Atk" if cats.count("Physical") >= cats.count("Special") else "SpA"
    second = "HP" if nature in SPEED_DOWN else "Spe"
    parts = {offense: 32, second: 32}
    leftover = "Def" if second == "HP" else "HP"  # the cap is 32 per stat
    parts[leftover] = parts.get(leftover, 0) + 2
    order = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]
    return " / ".join(f"{parts[s]} {s}" for s in order if parts.get(s))


def unpack_sheet(packed: str, gen: GenData) -> list[str]:
    """One ``|showteam|pX|...`` payload -> Showdown export blocks (one per mon)."""
    blocks = []
    for entry in packed.split("]"):
        fields = entry.split("|")
        if len(fields) < 7 or not fields[1] and not fields[0]:
            continue
        nickname, species_id, item, ability, moves, nature = fields[:6]
        gender = fields[7] if len(fields) > 7 else ""
        level = fields[10] if len(fields) > 10 and fields[10] else "50"
        species = gen.pokedex.get(
            (species_id or nickname).lower().replace("-", "").replace(" ", ""), {}
        ).get("name") or (species_id or nickname)
        move_ids = [m for m in moves.split(",") if m]
        move_names = [gen.moves.get(m.lower(), {}).get("name", m) for m in move_ids]
        head = species + (f" ({gender})" if gender in ("M", "F") else "")
        if item:
            head += f" @ {item}"
        lines = [
            head,
            f"Ability: {ability}",
            f"Level: {level}",
            f"EVs: {_spread(nature, [m.lower() for m in move_ids], gen)}",
        ]
        if nature:
            lines.append(f"{nature} Nature")
        lines += [f"- {m}" for m in move_names]
        blocks.append("\n".join(lines))
    return blocks


def teams_from_logs(paths: list[_Path], gen: GenData) -> list[str]:
    seen: set[str] = set()
    teams: list[str] = []
    for path in paths:
        data = json.loads(path.read_text())
        for _tag, (_ts, log) in data.items():
            for line in log.split("\n"):
                if not line.startswith("|showteam|"):
                    continue
                payload = line.split("|", 3)[3]
                blocks = unpack_sheet(payload, gen)
                if len(blocks) != 6:
                    continue
                text = "\n\n".join(blocks) + "\n"
                key = "\n".join(sorted(b.split("\n")[0] for b in blocks)) + text
                if key in seen:
                    continue
                seen.add(key)
                teams.append(text)
    return teams


def validate(teams: list[str], fmt: str) -> list[bool]:
    payload = "\n".join(json.dumps({"format": fmt, "team": t}) for t in teams) + "\n"
    proc = subprocess.run(
        ["node", "validate-teams-batch.js"],
        cwd=ROOT / "pokemon-showdown",
        input=payload,
        capture_output=True,
        text=True,
        check=False,
    )
    verdicts = []
    for line in proc.stdout.strip().split("\n"):
        try:
            verdicts.append(bool(json.loads(line).get("valid")))
        except json.JSONDecodeError:
            verdicts.append(False)
    verdicts += [False] * (len(teams) - len(verdicts))
    return verdicts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", type=_Path, nargs="+", required=True)
    ap.add_argument("--format", default="gen9championsvgc2026regmc")
    ap.add_argument("--out", type=_Path, default=_Path("teams/reg_mc"))
    ap.add_argument("--prefix", default="MC")
    args = ap.parse_args()
    gen = GenData.from_gen(9)
    teams = teams_from_logs(args.logs, gen)
    verdicts = validate(teams, args.format)
    args.out.mkdir(parents=True, exist_ok=True)
    kept = 0
    for text, ok in zip(teams, verdicts, strict=True):
        if not ok:
            continue
        kept += 1
        (args.out / f"{args.prefix}{kept}.txt").write_text(text)
    print(f"sheets parsed: {len(teams)} unique teams")
    print(f"valid in {args.format}: {kept}; written to {args.out}")


if __name__ == "__main__":
    main()
