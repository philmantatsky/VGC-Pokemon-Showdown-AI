"""Stage-D rule mining: what do winners LEAD and BRING against Trick Room setters?

The Stage-A mining established the direction (denial beats adaptation: TR
never set -> 50.0% for the non-TR side vs 42.1% set; bringing your slowest mon
made things WORSE). This narrows to actionable preview content, restricted to
opponents with a DEDICATED setter (any species whose recorded sets contain
Trick Room >= 90% of the time -- the trigger the rules will use, because
roster-aggregate P(TR) >= 0.6 covers half the meta).

Denial tools are detected from movesets, never species: Encore, Taunt,
Fake Out, Imprison (and own Trick Room as a flip). Human side: from |showteam|
packed teams. Our side: from the team file, so the analysis stays team-agnostic.

Outputs the tables Stage D's rule content is chosen from.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poke_env.data import to_id_str  # noqa: E402

from tools.analyze_ladder_previews import (  # noqa: E402
    load_human_logs,
    parse_log,
    team_species,
    wr,
)
from vgc_bench.src.preview_rules import species_trick_room_rate  # noqa: E402

DENIAL_MOVES = {"encore", "taunt", "fakeout", "imprison"}
FLIP_MOVES = {"trickroom"}
SETTER_RATE = 0.9


def packed_team_moves(packed: str) -> dict[str, set[str]]:
    """species id -> move ids, from a |showteam| packed team string."""
    result: dict[str, set[str]] = {}
    for mon in packed.split("]"):
        fields = mon.split("|")
        if len(fields) < 5:
            continue
        name = fields[0].strip()
        species = fields[1].strip() or name
        moves = {to_id_str(move) for move in fields[4].split(",") if move}
        result[to_id_str(species)] = moves
    return result


def team_file_moves(team_file: Path) -> dict[str, set[str]]:
    """species id -> move ids, from a Showdown export team file."""
    result: dict[str, set[str]] = {}
    for block in team_file.read_text().strip().split("\n\n"):
        lines = block.strip().split("\n")
        name = lines[0].split("@")[0].strip()
        if "(" in name and ")" in name:
            name = name[name.rindex("(") + 1 : name.rindex(")")]
        moves = {
            to_id_str(line[1:].strip())
            for line in lines
            if line.strip().startswith("- ")
        }
        result[to_id_str(name)] = moves
    return result


def dedicated_setters(roster: list[str]) -> list[str]:
    return [s for s in roster if species_trick_room_rate(s) >= SETTER_RATE]


def split(label: str, rows: list[dict], predicate) -> None:
    yes = [row for row in rows if predicate(row)]
    no = [row for row in rows if not predicate(row)]
    yes_wins = sum(row["won"] for row in yes)
    no_wins = sum(row["won"] for row in no)
    print(f"{label:44} yes {wr(yes_wins, len(yes)):18} no {wr(no_wins, len(no))}")


def rate(label: str, rows: list[dict], predicate, field: str) -> None:
    yes = [row for row in rows if predicate(row)]
    no = [row for row in rows if not predicate(row)]
    yes_hits = sum(row[field] for row in yes)
    no_hits = sum(row[field] for row in no)
    print(
        f"{label:44} yes {wr(yes_hits, len(yes)):18} no {wr(no_hits, len(no))}"
    )


def mine_humans(reference: list[str]) -> None:
    logs = load_human_logs()
    rows = []
    skipped = Counter()
    for tag, log in logs.items():
        parsed = parse_log(log)
        if parsed is None:
            skipped["unparseable"] += 1
            continue
        setters = {
            role: dedicated_setters(parsed["rosters"][role]) for role in ("p1", "p2")
        }
        anti = None
        if setters["p1"] and not setters["p2"]:
            anti = "p2"
        elif setters["p2"] and not setters["p1"]:
            anti = "p1"
        if anti is None:
            skipped["not_asymmetric_setter"] += 1
            continue
        showteam = parsed["showteams"].get(anti)
        if not showteam:
            skipped["no_anti_showteam"] += 1
            continue
        moves_by_species = packed_team_moves(showteam)
        denial_holders = {
            s for s, mv in moves_by_species.items() if mv & DENIAL_MOVES
        }
        flip_holders = {s for s, mv in moves_by_species.items() if mv & FLIP_MOVES}
        name = parsed["players"].get(anti, {}).get("name")
        if name is None:
            continue
        leads = set(parsed["leads"][anti])
        appeared = set(parsed["appeared"][anti])
        rows.append(
            {
                "battle": tag,
                "won": parsed["winner"] == name,
                "tr_set": parsed["trick_room_used"],
                "has_denial": bool(denial_holders),
                "led_denial": bool(leads & denial_holders),
                "brought_denial": bool(appeared & denial_holders),
                "led_flip": bool(leads & flip_holders),
                "overlap": len(set(parsed["rosters"][anti]) & set(reference)),
            }
        )

    total = len(rows)
    wins = sum(row["won"] for row in rows)
    print(
        f"\n=== HUMANS vs dedicated setters (rate>={SETTER_RATE}): "
        f"{total} games (skipped {dict(skipped)}) ==="
    )
    print(f"anti-setter side overall {wr(wins, total)}")
    tr_set = sum(row["tr_set"] for row in rows)
    print(f"TR got set in {wr(tr_set, total)} of these games")

    print("\n--- win rate splits ---")
    split("team HAS a denial holder", rows, lambda r: r["has_denial"])
    with_denial = [row for row in rows if row["has_denial"]]
    split("  ...LED a denial holder", with_denial, lambda r: r["led_denial"])
    split("  ...BROUGHT a denial holder", with_denial, lambda r: r["brought_denial"])
    split("led an own-TR flip holder", rows, lambda r: r["led_flip"])

    print("\n--- mechanism: did the lead choice actually deny TR? ---")
    rate(
        "TR-set rate | led denial holder",
        with_denial,
        lambda r: r["led_denial"],
        "tr_set",
    )

    print("\n--- outcome inside each branch ---")
    denied = [row for row in with_denial if not row["tr_set"]]
    not_denied = [row for row in with_denial if row["tr_set"]]
    split("TR never set: led denial", denied, lambda r: r["led_denial"])
    split("TR still set: led denial", not_denied, lambda r: r["led_denial"])

    overlap_rows = [row for row in rows if row["overlap"] >= 3]
    print(f"\n--- rosters sharing >=3 species with ours (n={len(overlap_rows)}) ---")
    split("led a denial holder", overlap_rows, lambda r: r["led_denial"])

    out = ROOT / "results_analysis" / "tr_denial_mining_humans.json"
    out.write_text(json.dumps({"games": rows}, indent=1))
    print(f"\nwrote {out}")


def mine_ladder(team_file: Path) -> None:
    analysis = ROOT / "results_analysis" / "ladder_preview_analysis.json"
    assert analysis.exists(), "run tools/analyze_ladder_previews.py ladder first"
    games = json.loads(analysis.read_text())["games"]
    our_moves = team_file_moves(team_file)
    denial_holders = {s for s, mv in our_moves.items() if mv & DENIAL_MOVES}
    print(
        f"\n=== OUR LADDER vs dedicated setters "
        f"(our denial holders: {sorted(denial_holders)}) ==="
    )
    rows = []
    for game in games:
        setters = dedicated_setters(game["opp_roster"])
        if not setters:
            continue
        leads = set(game["our_leads"])
        appeared = set(game["our_appeared"])
        rows.append(
            {
                "won": game["won"],
                "tr_set": game["trick_room_used"],
                "led_denial": bool(leads & denial_holders),
                "brought_denial": bool(appeared & denial_holders),
                "leads": sorted(leads),
            }
        )
    total = len(rows)
    wins = sum(row["won"] for row in rows)
    print(f"games vs dedicated setters: {wr(wins, total)}")
    tr_set = sum(row["tr_set"] for row in rows)
    print(f"TR got set in {wr(tr_set, total)}")
    split("led our denial holder", rows, lambda r: r["led_denial"])
    split("brought our denial holder", rows, lambda r: r["brought_denial"])
    rate("TR-set rate | led denial", rows, lambda r: r["led_denial"], "tr_set")
    lead_stats: Counter = Counter()
    lead_wins: Counter = Counter()
    for row in rows:
        key = " + ".join(row["leads"])
        lead_stats[key] += 1
        lead_wins[key] += row["won"]
    print("\n--- our lead pairs vs dedicated setters ---")
    for key, count in lead_stats.most_common(8):
        print(f"{key:44} {wr(lead_wins[key], count)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--team", default=str(ROOT / "teams/reg_mb/our_team.txt"))
    args = ap.parse_args()
    reference = team_species(Path(args.team))
    mine_ladder(Path(args.team))
    mine_humans(reference)


if __name__ == "__main__":
    main()
