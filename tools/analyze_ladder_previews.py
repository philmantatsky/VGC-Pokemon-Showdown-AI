"""Mine Team Preview evidence from our ladder replays and from human replays.

Two subcommands, both read-only over stored data:

    ladder   Parse every ladder_replays*/ HTML replay (plus each directory's
             decisions.jsonl when present) into one row per game and print the
             preview tables the 2026-08-23 audit produced by hand: per-batch
             records, lead-pair win rates, species brought/benched splits,
             first-faint outcome splits, and the Trick Room subset.

    humans   Mine the scraped human replay corpora (battle_logs*/logs_*.json)
             for how humans respond to Trick-Room-likely rosters, expressed in
             team-agnostic properties (slowest member brought, frail-fast
             leads, toggle-move holders) rather than species names.

Everything is derived from roster properties and data files, never from
hardcoded species, so the analysis keeps working if our team changes.

Outputs land in results_analysis/ as JSON beside the printed tables.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poke_env.data import GenData, to_id_str  # noqa: E402

from vgc_bench.src.preview_rules import trick_room_probability  # noqa: E402

LOG_EMBED = re.compile(r'class="battle-log-data"[^>]*>(.*?)</script>', re.S)
TR_THRESHOLD = 0.6
TOGGLE_MOVES = {"trickroom", "encore", "taunt", "imprison"}


# ---------------------------------------------------------------------------
# shared parsing


def parse_log(log: str | list[str]) -> dict | None:
    """Extract preview-relevant facts from one Showdown battle log.

    Accepts either a newline-joined string (replay HTML embeds) or a list of
    protocol lines (the scraped battle_logs*/ JSON files store lists).
    """
    lines = log if isinstance(log, list) else log.split("\n")
    players: dict[str, dict] = {}
    rosters: dict[str, list[str]] = {"p1": [], "p2": []}
    leads: dict[str, list[str]] = {"p1": [], "p2": []}
    appeared: dict[str, set[str]] = {"p1": set(), "p2": set()}
    showteams: dict[str, str] = {}
    first_faint_role: str | None = None
    winner: str | None = None
    turns = 0
    trick_room_set_by: str | None = None
    for line in lines:
        parts = line.split("|")
        if len(parts) < 2:
            continue
        kind = parts[1]
        if kind == "player" and len(parts) > 3 and parts[3]:
            rating = None
            if len(parts) > 5 and parts[5].isdigit():
                rating = int(parts[5])
            players.setdefault(parts[2], {"name": parts[3], "rating": rating})
        elif kind == "poke" and len(parts) > 3:
            species = to_id_str(parts[3].split(",")[0])
            rosters[parts[2]].append(species)
        elif kind == "showteam" and len(parts) > 3:
            showteams[parts[2]] = "|".join(parts[3:])
        elif kind in ("switch", "drag") and len(parts) > 3:
            role = parts[2][:2]
            species = to_id_str(parts[3].split(",")[0])
            appeared[role].add(species)
            if turns <= 1 and len(leads[role]) < 2 and species not in leads[role]:
                leads[role].append(species)
        elif kind == "turn":
            try:
                turns = max(turns, int(parts[2]))
            except (IndexError, ValueError):
                pass
        elif kind == "faint" and first_faint_role is None and len(parts) > 2:
            first_faint_role = parts[2][:2]
        elif kind == "-fieldstart" and len(parts) > 2 and "Trick Room" in parts[2]:
            if trick_room_set_by is None and len(parts) > 3:
                of = next((p for p in parts if p.startswith("[of] ")), "")
                trick_room_set_by = of[5:7] if of else "?"
            trick_room_set_by = trick_room_set_by or "?"
        elif kind == "win" and len(parts) > 2:
            winner = parts[2]
    if winner is None or not rosters["p1"] or not rosters["p2"]:
        return None
    return {
        "players": players,
        "rosters": rosters,
        "leads": leads,
        "appeared": {role: sorted(mons) for role, mons in appeared.items()},
        "showteams": showteams,
        "first_faint_role": first_faint_role,
        "winner": winner,
        "turns": turns,
        "trick_room_used": trick_room_set_by is not None,
    }


def wr(wins: int, total: int) -> str:
    return f"{wins}/{total} = {wins / total * 100:5.1f}%" if total else "0/0"


# ---------------------------------------------------------------------------
# part 1: our ladder replays


def load_decision_previews(directory: Path) -> dict[str, dict]:
    """battle tag -> {leads: [...], bring: [...], probabilities: [...]}."""
    path = directory / "decisions.jsonl"
    if not path.exists():
        return {}
    per_battle: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("turn") == 0 and isinstance(row.get("chosen"), dict):
                per_battle[row.get("battle", "")].append(row["chosen"])
    result = {}
    for tag, chosen_rows in per_battle.items():
        stages = [row for row in chosen_rows if row.get("orders")]
        if len(stages) < 2:
            continue
        leads = [order.get("species", "") for order in stages[0]["orders"]]
        back = [order.get("species", "") for order in stages[1]["orders"]]
        result[tag] = {
            "leads": leads,
            "bring": sorted(leads + back),
            "probabilities": [
                stages[0].get("policy_probability"),
                stages[1].get("policy_probability"),
            ],
        }
    return result


def canonical(species: str, roster: list[str]) -> str:
    """Match a decision-log species id (base form) to the replay roster id."""
    if species in roster:
        return species
    for candidate in roster:
        if candidate.startswith(species) or species.startswith(candidate):
            return candidate
    return species


def analyze_ladder(_args: argparse.Namespace) -> None:
    rows = []
    replay_dirs = sorted(
        {p.parent for p in ROOT.glob("ladder_replays*/**/*.html")}
        | {p.parent for p in ROOT.glob("ladder_replays*/*.html")}
    )
    for directory in replay_dirs:
        decisions = load_decision_previews(directory)
        for html in sorted(directory.glob("*.html")):
            username = html.name.split(" - ")[0]
            match = LOG_EMBED.search(html.read_text(encoding="utf-8", errors="replace"))
            if match is None:
                continue
            parsed = parse_log(match.group(1))
            if parsed is None:
                continue
            our_role = next(
                (
                    role
                    for role, info in parsed["players"].items()
                    if info["name"] == username
                ),
                None,
            )
            if our_role is None:
                continue
            their_role = "p2" if our_role == "p1" else "p1"
            tag = html.stem.split(" - ")[-1].split("-kd")[0]
            # replay filenames may carry a password suffix after the numeric id
            tag = re.sub(
                r"^(battle-[a-z0-9]+-\d+).*$", r"\1", html.stem.split(" - ")[-1]
            )
            our_roster = parsed["rosters"][our_role]
            decision = decisions.get(tag)
            rows.append(
                {
                    "batch": str(directory.relative_to(ROOT)),
                    "battle": tag,
                    "won": parsed["winner"] == username,
                    "turns": parsed["turns"],
                    "our_rating": parsed["players"][our_role]["rating"],
                    "opp_rating": parsed["players"][their_role]["rating"],
                    "opponent": parsed["players"][their_role]["name"],
                    "our_leads": sorted(parsed["leads"][our_role]),
                    "our_appeared": parsed["appeared"][our_role],
                    "our_roster": our_roster,
                    "our_bring_logged": (
                        sorted(canonical(s, our_roster) for s in decision["bring"])
                        if decision
                        else None
                    ),
                    "preview_probabilities": (
                        decision["probabilities"] if decision else None
                    ),
                    "opp_roster": parsed["rosters"][their_role],
                    "opp_leads": sorted(parsed["leads"][their_role]),
                    "first_faint": (
                        None
                        if parsed["first_faint_role"] is None
                        else (
                            "ours"
                            if parsed["first_faint_role"] == our_role
                            else "theirs"
                        )
                    ),
                    "opp_tr_probability": round(
                        trick_room_probability(parsed["rosters"][their_role]), 4
                    ),
                    "trick_room_used": parsed["trick_room_used"],
                }
            )

    out_dir = ROOT / "results_analysis"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "ladder_preview_analysis.json"
    out_path.write_text(json.dumps({"games": rows}, indent=1))

    total = len(rows)
    wins = sum(row["won"] for row in rows)
    print(f"\n=== ladder games parsed: {wr(wins, total)} over {total} games ===\n")

    print("--- per batch ---")
    by_batch: dict[str, list] = defaultdict(list)
    for row in rows:
        by_batch[row["batch"]].append(row)
    for batch, batch_rows in sorted(by_batch.items()):
        batch_wins = sum(r["won"] for r in batch_rows)
        ratings = [r["our_rating"] for r in batch_rows if r["our_rating"]]
        span = f"elo {min(ratings)}-{max(ratings)}" if ratings else ""
        print(f"{batch:44} {wr(batch_wins, len(batch_rows)):20} {span}")

    print("\n--- lead pair (ours) ---")
    lead_stats: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        if len(row["our_leads"]) == 2:
            key = tuple(row["our_leads"])
            lead_stats[key][0] += row["won"]
            lead_stats[key][1] += 1
    for key, (lead_wins, lead_total) in sorted(
        lead_stats.items(), key=lambda item: -item[1][1]
    ):
        print(f"{' + '.join(key):40} {wr(lead_wins, lead_total)}")

    print("\n--- our species: appeared vs never appeared ---")
    roster_species = sorted({s for row in rows for s in row["our_roster"]})
    for species in roster_species:
        in_w = in_t = out_w = out_t = 0
        for row in rows:
            if species not in row["our_roster"]:
                continue
            if species in row["our_appeared"]:
                in_w += row["won"]
                in_t += 1
            else:
                out_w += row["won"]
                out_t += 1
        print(f"{species:18} appeared {wr(in_w, in_t):20} absent {wr(out_w, out_t)}")

    logged = [row for row in rows if row["our_bring_logged"]]
    if logged:
        print(f"\n--- our bring-4 (decision-logged, n={len(logged)}) ---")
        bring_stats: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
        for row in logged:
            key = tuple(row["our_bring_logged"])
            bring_stats[key][0] += row["won"]
            bring_stats[key][1] += 1
        for key, (bring_wins, bring_total) in sorted(
            bring_stats.items(), key=lambda item: -item[1][1]
        )[:8]:
            print(f"{' '.join(key):58} {wr(bring_wins, bring_total)}")

    print("\n--- first faint ---")
    for side in ("ours", "theirs", None):
        subset = [row for row in rows if row["first_faint"] == side]
        side_wins = sum(row["won"] for row in subset)
        print(f"first faint {str(side):8} {wr(side_wins, len(subset))}")

    print("\n--- Trick Room ---")
    tr_rows = [row for row in rows if row["opp_tr_probability"] >= TR_THRESHOLD]
    tr_wins = sum(row["won"] for row in tr_rows)
    print(f"opp roster P(TR)>={TR_THRESHOLD}   {wr(tr_wins, len(tr_rows))}")
    used = [row for row in rows if row["trick_room_used"]]
    used_wins = sum(row["won"] for row in used)
    print(f"TR actually set        {wr(used_wins, len(used))}")
    rest = [row for row in rows if not row["trick_room_used"]]
    rest_wins = sum(row["won"] for row in rest)
    print(f"TR never set           {wr(rest_wins, len(rest))}")

    print(f"\nwrote {out_path}")


# ---------------------------------------------------------------------------
# part 2: human replays


def load_human_logs() -> dict[str, str]:
    logs: dict[str, str] = {}
    for path in (
        ROOT / "battle_logs_top" / "logs_gen9championsvgc2026regmb.json",
        ROOT / "battle_logs_top" / "logs_gen9championsvgc2026regmbbo3.json",
        ROOT / "battle_logs" / "logs_gen9championsvgc2026regmb.json",
        ROOT / "battle_logs" / "logs_gen9championsvgc2026regmbbo3.json",
    ):
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for tag, log in data.items():
            # scraped entries are [upload_timestamp, log_string]
            if isinstance(log, list):
                log = next((item for item in log if isinstance(item, str)), "")
            if log:
                logs.setdefault(tag, log)
    return logs


def team_species(team_file: Path) -> list[str]:
    """Species list from a Showdown export team file (team-agnostic)."""
    species = []
    for block in team_file.read_text().strip().split("\n\n"):
        first = block.strip().split("\n")[0]
        name = first.split("@")[0].strip()
        if "(" in name and ")" in name:
            name = name[name.rindex("(") + 1 : name.rindex(")")]
        species.append(to_id_str(name))
    return species


def analyze_humans(args: argparse.Namespace) -> None:
    dex = GenData.from_gen(9).pokedex
    reference = team_species(Path(args.team))

    def base(species: str, stat: str) -> int:
        entry = dex.get(species) or dex.get(species.rstrip("f")) or {}
        return int(entry.get("baseStats", {}).get(stat, 80))

    def bulk(species: str) -> int:
        return base(species, "hp") + base(species, "def") + base(species, "spd")

    def frail_fast(species: str) -> bool:
        return base(species, "spe") >= 100 and bulk(species) <= 255

    logs = load_human_logs()
    print(f"human logs loaded: {len(logs)}")
    stats = Counter()
    response_rows = []
    for tag, log in logs.items():
        parsed = parse_log(log)
        if parsed is None:
            stats["unparseable"] += 1
            continue
        tr_prob = {
            role: trick_room_probability(parsed["rosters"][role])
            for role in ("p1", "p2")
        }
        # clean asymmetric games: one side TR-likely, the other not
        tr_side = None
        if tr_prob["p1"] >= TR_THRESHOLD and tr_prob["p2"] < TR_THRESHOLD:
            tr_side = "p1"
        elif tr_prob["p2"] >= TR_THRESHOLD and tr_prob["p1"] < TR_THRESHOLD:
            tr_side = "p2"
        if tr_side is None:
            stats["not_asymmetric_tr"] += 1
            continue
        anti_role = "p2" if tr_side == "p1" else "p1"
        anti = parsed["rosters"][anti_role]
        anti_name = parsed["players"].get(anti_role, {}).get("name")
        if anti_name is None:
            continue
        won = parsed["winner"] == anti_name
        appeared = set(parsed["appeared"][anti_role])
        slowest = min(anti, key=lambda s: base(s, "spe"))
        leads = parsed["leads"][anti_role]
        toggle_holder_brought = None
        showteam = parsed["showteams"].get(anti_role)
        if showteam:
            packed = to_id_str(showteam)
            toggle_holder_brought = any(move in packed for move in TOGGLE_MOVES)
        response_rows.append(
            {
                "battle": tag,
                "anti_won": won,
                "anti_roster": anti,
                "tr_roster": parsed["rosters"][tr_side],
                "brought_slowest": slowest in appeared,
                "slowest_speed": base(slowest, "spe"),
                "frail_fast_leads": sum(frail_fast(s) for s in leads),
                "overlap_with_reference": len(set(anti) & set(reference)),
                "has_toggle_in_team": toggle_holder_brought,
                "turns": parsed["turns"],
                "trick_room_used": parsed["trick_room_used"],
            }
        )

    total = len(response_rows)
    wins = sum(row["anti_won"] for row in response_rows)
    print(f"\n=== asymmetric TR games: {total} (skipped: {dict(stats)}) ===\n")
    print(f"non-TR side overall     {wr(wins, total)}")

    def split(label: str, predicate) -> None:
        yes = [row for row in response_rows if predicate(row)]
        no = [row for row in response_rows if not predicate(row)]
        yes_wins = sum(row["anti_won"] for row in yes)
        no_wins = sum(row["anti_won"] for row in no)
        print(f"{label:34} yes {wr(yes_wins, len(yes)):20} no {wr(no_wins, len(no))}")

    print("\n--- non-TR side, property splits ---")
    split("brought roster's slowest mon", lambda row: row["brought_slowest"])
    split(
        "slowest mon is genuinely slow (<=60)",
        lambda row: row["brought_slowest"] and row["slowest_speed"] <= 60,
    )
    split("led 2 frail-fast mons", lambda row: row["frail_fast_leads"] >= 2)
    split("led 0 frail-fast mons", lambda row: row["frail_fast_leads"] == 0)
    split("TR actually got set", lambda row: row["trick_room_used"])

    overlap = [row for row in response_rows if row["overlap_with_reference"] >= 3]
    overlap_wins = sum(row["anti_won"] for row in overlap)
    print(
        f"\nrosters sharing >=3 species with {Path(args.team).name}: "
        f"{wr(overlap_wins, len(overlap))}"
    )
    if overlap:
        brought = [row for row in overlap if row["brought_slowest"]]
        brought_wins = sum(row["anti_won"] for row in brought)
        rest = [row for row in overlap if not row["brought_slowest"]]
        rest_wins = sum(row["anti_won"] for row in rest)
        print(f"  ...and brought slowest  {wr(brought_wins, len(brought))}")
        print(f"  ...did not              {wr(rest_wins, len(rest))}")

    out_dir = ROOT / "results_analysis"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "human_tr_response.json"
    out_path.write_text(json.dumps({"games": response_rows}, indent=1))
    print(f"\nwrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("ladder", help="analyze our own ladder replays")
    humans = sub.add_parser("humans", help="analyze scraped human replays")
    humans.add_argument("--team", default=str(ROOT / "teams/reg_mb/our_team.txt"))
    args = ap.parse_args()
    if args.command == "ladder":
        analyze_ladder(args)
    else:
        analyze_humans(args)


if __name__ == "__main__":
    main()
