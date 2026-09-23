"""Opening audit of our ladder games: how the first turns go, and what kills us.

Complements tools/ladder_loss_profile.py (same replay parsing) with the fields
that the T6 local reviews pointed at (2026-09-21: Farigiraf KO'd before Trick
Room in most sampled losses):

* our plan: which of our Pokemon can set Trick Room (from --team, never
  hardcoded), whether one led, whether it used Trick Room, whether Trick Room
  went up for us and on which turn, and whether our setter fainted before it
  went up;
* turn-1 events: opponent Fake Out (and its target), our faints on turns 1-2;
* knockouts: for each of our faints, the last opposing move that damaged it
  (or the residual cause), aggregated by attacker and move;
* the opposing archetype as REVEALED in the game (full open team sheet when
  present, otherwise what was used or activated): trick_room, rain, sun,
  psychic_terrain, grassy_fakeout, tailwind, balance.

Team-agnostic: species come from the log, setters from the team file; our side
is identified by the account name. Usage (from the repo root):

  .venv/bin/python tools/ladder_opening_audit.py ladder_replays_mc_deployed_T6 \\
      --team teams/candidates_mc/T6.txt --json results_analysis/t6_opening_audit.json
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import collections
import json
import re
from pathlib import Path

from tools.ladder_loss_profile import OUR_NAME, extract_log, parse_game

ARCHETYPES = (
    "trick_room",
    "rain",
    "sun",
    "psychic_terrain",
    "grassy_fakeout",
    "tailwind",
    "balance",
)


def setters_from_team(text: str) -> set[str]:
    """Species on our team sheet that carry Trick Room."""
    setters, species = set(), None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            species = None
            continue
        if species is None and not line.startswith(("-", "Ability", "Level", "EVs")):
            head = line.split(" @ ")[0]
            head = re.sub(r"\s*\((M|F)\)\s*$", "", head).strip()
            match = re.search(r"\(([^()]+)\)\s*$", head)
            species = (match.group(1) if match else head).strip()
        elif species and line.lower() == "- trick room":
            setters.add(species)
    return setters


def _base(species: str) -> str:
    return species.split("-Mega")[0].strip()


def opening_events(
    log: str, ours: str = OUR_NAME, setters: set[str] | frozenset[str] = frozenset()
) -> dict:
    """Second pass over one log: plan execution, turn-1 events, knockout causes."""
    names: dict[str, str] = {}
    ident: dict[str, str] = {}  # "p1a: Nick" -> species
    turn = 0
    last_move: tuple[str, str, str] | None = None  # (attacker ident, move, target)
    our_role = None
    tr_up_turn = None
    tr_by_us_turns: list[int] = []
    our_tr_attempts: list[tuple[int, str]] = []
    setter_faint_turns: list[tuple[int, str]] = []
    knockouts: list[dict] = []
    opp_fake_out: list[tuple[int, str]] = []
    revealed = {"moves": set(), "abilities": set(), "species": set(), "weather": set()}
    sheet_text = ""
    damage_cause: dict[str, tuple[str | None, str]] = {}  # victim ident -> (by, what)
    for line in log.split("\n"):
        parts = line.split("|")
        if len(parts) < 2:
            continue
        kind = parts[1]
        if kind == "player" and len(parts) > 3 and parts[3]:
            names[parts[2]] = parts[3]
            if parts[3] == ours:
                our_role = parts[2]
        elif kind == "showteam" and len(parts) > 3:
            if our_role and parts[2] != our_role:
                sheet_text += "|".join(parts[3:])
        elif kind in ("switch", "drag", "replace") and len(parts) > 3:
            ident[parts[2]] = parts[3].split(",")[0].strip()
            if our_role and not parts[2].startswith(our_role):
                revealed["species"].add(_base(ident[parts[2]]))
        elif kind == "turn":
            try:
                turn = int(parts[2])
            except (IndexError, ValueError):
                pass
        elif kind == "move" and len(parts) > 3:
            attacker = parts[2]
            target = parts[4] if len(parts) > 4 else ""
            last_move = (attacker, parts[3], target)
            species = _base(ident.get(attacker, attacker))
            if our_role and attacker.startswith(our_role):
                if parts[3] == "Trick Room":
                    our_tr_attempts.append((turn, species))
            elif our_role:
                revealed["moves"].add(parts[3])
                if parts[3] == "Fake Out" and turn <= 1:
                    opp_fake_out.append((turn, _base(ident.get(target, target))))
        elif kind == "-fieldstart" and "Trick Room" in line:
            if last_move and our_role and last_move[0].startswith(our_role):
                tr_by_us_turns.append(turn)
                if tr_up_turn is None:
                    tr_up_turn = turn
        elif kind == "-damage" and len(parts) > 2:
            if "[from]" in line:
                damage_cause[parts[2]] = (None, line.split("[from] ")[1].split("|")[0])
            elif last_move:
                damage_cause[parts[2]] = (
                    _base(ident.get(last_move[0], last_move[0])),
                    last_move[1],
                )
        elif kind == "-ability" and len(parts) > 3 and our_role:
            if not parts[2].startswith(our_role):
                revealed["abilities"].add(parts[3])
        elif kind == "-weather" and len(parts) > 2 and "[upkeep]" not in line:
            if "[from] ability:" in line and "[of] " in line and our_role:
                of = line.split("[of] ")[1][:2]
                if of != our_role:
                    revealed["weather"].add(parts[2])
                    revealed["abilities"].add(
                        line.split("[from] ability: ")[1].split("|")[0]
                    )
        elif kind == "-fieldstart" and "Psychic Terrain" in line and our_role:
            if "[of] " in line and line.split("[of] ")[1][:2] != our_role:
                revealed["abilities"].add("Psychic Surge")
        elif kind == "faint" and len(parts) > 2 and our_role:
            victim = parts[2]
            if victim.startswith(our_role):
                species = _base(ident.get(victim, victim))
                attacker_species, cause = damage_cause.get(victim, (None, "unknown"))
                knockouts.append(
                    {
                        "turn": turn,
                        "species": species,
                        "by": attacker_species,
                        "move": cause,
                    }
                )
                if species in setters and tr_up_turn is None:
                    setter_faint_turns.append((turn, species))
    return {
        "our_tr_attempts": our_tr_attempts,
        "our_tr_up_turn": tr_up_turn,
        "our_tr_count": len(tr_by_us_turns),
        "setter_fainted_before_tr": setter_faint_turns,
        "knockouts": knockouts,
        "opp_fake_out_turn1": opp_fake_out,
        "open_sheet": bool(sheet_text),
        "archetype": archetype(sheet_text, revealed),
    }


def archetype(sheet_text: str, revealed: dict) -> str:
    """Opposing archetype from the full sheet if shown, else from revealed play."""
    t = sheet_text.lower()
    moves = {m.lower() for m in revealed["moves"]}
    abilities = {a.lower() for a in revealed["abilities"]}
    species = {s.lower() for s in revealed["species"]}
    if "trickroom" in t or "trick room" in moves:
        return "trick_room"
    if "drizzle" in t or "drizzle" in abilities or "RainDance" in revealed["weather"]:
        return "rain"
    if "drought" in t or "drought" in abilities or "SunnyDay" in revealed["weather"]:
        return "sun"
    if "psychicsurge" in t or "psychic surge" in abilities:
        return "psychic_terrain"
    if "rillaboom" in species and ("fakeout" in t or "fake out" in moves):
        return "grassy_fakeout"
    if "tailwind" in t or "tailwind" in moves:
        return "tailwind"
    return "balance"


def audit(dirs: list[Path], team: Path | None, ours: str = OUR_NAME) -> list[dict]:
    setters = setters_from_team(team.read_text()) if team else set()
    rows = []
    for directory in dirs:
        for path in sorted(directory.glob("*.html"), key=lambda p: p.stat().st_mtime):
            log = extract_log(path)
            if log is None:
                continue
            row = parse_game(log, ours)
            if row is None:
                continue
            row.update(opening_events(log, ours, setters))
            row["replay"] = path.name
            row["setter_led"] = any(s in setters for s in map(_base, row["our_leads"]))
            rows.append(row)
    return rows


def _wl(rows: list[dict]) -> str:
    wins = sum(r["won"] for r in rows)
    return f"{wins}-{len(rows) - wins}"


def report(rows: list[dict]) -> dict:
    out: dict = {"games": len(rows), "record": _wl(rows)}
    lines = [f"games {len(rows)}  record {out['record']}"]

    def split(label: str, predicate) -> None:
        yes = [r for r in rows if predicate(r)]
        no = [r for r in rows if not predicate(r)]
        out[label] = {"yes": _wl(yes), "no": _wl(no)}
        lines.append(f"  {label:<44} yes {_wl(yes):>6}   no {_wl(no):>6}")

    split("setter led", lambda r: r["setter_led"])
    split("our Trick Room went up", lambda r: r["our_tr_up_turn"] is not None)
    split("our Trick Room up by turn 1", lambda r: (r["our_tr_up_turn"] or 99) <= 1)
    split(
        "setter fainted before our Trick Room",
        lambda r: bool(r["setter_fainted_before_tr"]),
    )
    split(
        "we used Trick Room twice or more (reversal risk)",
        lambda r: r["our_tr_count"] >= 2,
    )
    split("first faint ours", lambda r: bool(r["first_faint_ours"]))
    split(
        "we lost a Pokemon on turn 1",
        lambda r: any(k["turn"] <= 1 for k in r["knockouts"]),
    )
    split("opponent Fake Out on turn 1", lambda r: bool(r["opp_fake_out_turn1"]))
    split("opponent set Trick Room", lambda r: r["opp_tr"])
    split("open team sheets", lambda r: r["open_sheet"])
    leads = collections.Counter(tuple(sorted(map(_base, r["our_leads"]))) for r in rows)
    lines.append(
        "  our leads: "
        + ", ".join(f"{'+'.join(k)} x{v}" for k, v in leads.most_common())
    )
    brings = collections.Counter(
        tuple(sorted(map(_base, r["our_brought"]))) for r in rows
    )
    lines.append(
        "  our brings: "
        + ", ".join(f"{'+'.join(k)} x{v}" for k, v in brings.most_common(4))
    )
    arch = collections.defaultdict(list)
    for r in rows:
        arch[r["archetype"]].append(r)
    out["archetypes"] = {a: _wl(arch[a]) for a in ARCHETYPES if arch[a]}
    lines.append(
        "  vs archetype: "
        + ", ".join(f"{a} {out['archetypes'][a]}" for a in out["archetypes"])
    )
    killers = collections.Counter()
    victims = collections.Counter()
    for r in rows:
        for k in r["knockouts"]:
            killers[f"{k['by'] or '-'}: {k['move']}"] += 1
            victims[k["species"]] += 1
    out["knocked_out_by"] = dict(killers.most_common(12))
    out["our_faints_by_species"] = dict(victims.most_common())
    lines.append(
        "  our faints by species: "
        + ", ".join(f"{k} {v}" for k, v in victims.most_common())
    )
    lines.append(
        "  knocked out by: "
        + ", ".join(f"{k} x{v}" for k, v in killers.most_common(10))
    )
    out["lines"] = lines
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--team", type=Path, default=None, help="our team file (setters)")
    ap.add_argument("--ours", default=OUR_NAME)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument(
        "--losses", action="store_true", help="print one digest line per loss"
    )
    args = ap.parse_args()
    rows = audit(args.dirs, args.team, args.ours)
    summary = report(rows)
    print("\n".join(summary["lines"]))
    if args.losses:
        for r in rows:
            if r["won"]:
                continue
            kos = "; ".join(
                f"t{k['turn']} {k['species']} <- {k['by']} {k['move']}"
                for k in r["knockouts"]
            )
            mode = "open" if r["open_sheet"] else "hidden"
            leads = f"{'+'.join(r['our_leads'])} vs {'+'.join(r['opp_leads'])}"
            print(
                f"L {r['replay'][-40:]} vs {r['archetype']} {mode} | leads {leads}"
                f" | our TR {r['our_tr_up_turn']} | turns {r['turns']} | {kos}"
            )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"summary": summary, "games": rows}, indent=1, default=list)
        )


if __name__ == "__main__":
    main()
