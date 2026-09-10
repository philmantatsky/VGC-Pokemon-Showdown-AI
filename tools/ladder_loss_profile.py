"""Loss profile of our ladder games: what separates the games we lose from the
games we win?

Reads every replay HTML in the given directories (the log is embedded verbatim
in a ``battle-log-data`` script block) plus the per-decision audit
(``decisions.jsonl``) when present, extracts one feature row per game, and
prints win rate splits by the features that have been suspected of deciding
games: opponent Trick Room, first faint, leads, opponent species, opponent
rating, game length, blowouts, tera usage, hit effectiveness, guard activity.

Team-agnostic: every species comes from the log; our side is identified by
the account name. Usage (from the repo root):

  .venv/bin/python tools/ladder_loss_profile.py ladder_replays_league_canary_20260830 \
      ladder_replays_league_20260906 --json results_analysis/loss_profile.json
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import re
import sys as _sys
from pathlib import Path
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

OUR_NAME = "antonius1"
LOG_BLOCK = re.compile(
    r'<script type="text/plain" class="battle-log-data">(.*?)</script>', re.S
)


def extract_log(path: Path) -> str | None:
    text = path.read_text(errors="replace")
    match = LOG_BLOCK.search(text)
    if not match:
        return None
    return html.unescape(match.group(1))


def _species(field: str) -> str:
    return field.split(",")[0].strip()


def _role(ident: str) -> str:
    return ident[:2]


def parse_game(log: str, ours: str = OUR_NAME) -> dict | None:
    """One feature row per game. Returns None when the log is unusable."""
    lines = log.split("\n")
    players: dict[str, dict] = {}
    rosters: dict[str, list[str]] = {"p1": [], "p2": []}
    leads: dict[str, list[str]] = {"p1": [], "p2": []}
    brought: dict[str, list[str]] = {"p1": [], "p2": []}
    faints: list[tuple[int, str, str]] = []  # (turn, role, species)
    turn = 0
    winner = None
    tr_set: list[tuple[int, str]] = []
    tailwind: list[tuple[int, str]] = []
    weather: list[tuple[int, str, str]] = []
    tera: dict[str, list[tuple[int, str]]] = {"p1": [], "p2": []}
    mega: dict[str, list[tuple[int, str]]] = {"p1": [], "p2": []}
    protects = collections.Counter()
    switches = collections.Counter()
    moves_used = collections.Counter()
    se_hits = collections.Counter()  # attacker role -> count
    resisted_hits = collections.Counter()
    immune_hits = collections.Counter()
    attacks = collections.Counter()
    last_attacker_role: str | None = None
    ident_species: dict[str, str] = {}
    forfeit = False
    for line in lines:
        parts = line.split("|")
        if len(parts) < 2:
            continue
        kind = parts[1]
        if kind == "player" and len(parts) > 3 and parts[3]:
            rating = None
            if len(parts) > 5 and parts[5].strip().isdigit():
                rating = int(parts[5])
            players.setdefault(parts[2], {"name": parts[3], "rating": rating})
        elif kind == "poke" and len(parts) > 3:
            rosters[parts[2]].append(_species(parts[3]))
        elif kind in ("switch", "drag", "replace") and len(parts) > 3:
            role = _role(parts[2])
            species = _species(parts[3])
            ident_species[parts[2].split(":")[0]] = species
            if species not in brought[role]:
                brought[role].append(species)
            if turn <= 1 and len(leads[role]) < 2 and species not in leads[role]:
                leads[role].append(species)
            if kind == "switch" and turn >= 1:
                switches[role] += 1
        elif kind == "turn":
            try:
                turn = int(parts[2])
            except (IndexError, ValueError):
                pass
        elif kind == "move" and len(parts) > 3:
            role = _role(parts[2])
            name = parts[3]
            moves_used[role] += 1
            last_attacker_role = role
            if name == "Protect" or name in ("Detect", "Wide Guard", "Spiky Shield"):
                protects[role] += 1
            if name == "Trick Room":
                pass
        elif kind == "-fieldstart" and len(parts) > 2:
            if "Trick Room" in parts[2]:
                of = next((p for p in parts if p.startswith("[of] ")), "")
                tr_set.append((turn, of[5:7] if of else (last_attacker_role or "?")))
        elif kind == "-sidestart" and len(parts) > 3 and "Tailwind" in parts[3]:
            tailwind.append((turn, _role(parts[2])))
        elif kind == "-weather" and len(parts) > 2 and "[upkeep]" not in line:
            of = next((p for p in parts if p.startswith("[of] ")), "")
            weather.append(
                (turn, parts[2], of[5:7] if of else (last_attacker_role or "?"))
            )
        elif kind == "-terastallize" and len(parts) > 3:
            tera[_role(parts[2])].append((turn, parts[3]))
        elif kind == "-mega" and len(parts) > 2:
            mega[_role(parts[2])].append(
                (turn, _species(parts[3]) if len(parts) > 3 else "")
            )
        elif kind == "-supereffective" and len(parts) > 2:
            target_role = _role(parts[2])
            se_hits["p1" if target_role == "p2" else "p2"] += 1
        elif kind == "-resisted" and len(parts) > 2:
            target_role = _role(parts[2])
            resisted_hits["p1" if target_role == "p2" else "p2"] += 1
        elif kind == "-immune" and len(parts) > 2:
            target_role = _role(parts[2])
            immune_hits["p1" if target_role == "p2" else "p2"] += 1
        elif kind == "-damage" and len(parts) > 2 and "[from]" not in line:
            if last_attacker_role is not None:
                attacks[last_attacker_role] += 1
        elif kind == "faint" and len(parts) > 2:
            role = _role(parts[2])
            species = ident_species.get(parts[2].split(":")[0], parts[2])
            faints.append((turn, role, species))
        elif kind == "win" and len(parts) > 2:
            winner = parts[2].strip()
        elif kind == "-message" and "forfeited" in line:
            forfeit = True
    our_role = next((r for r, p in players.items() if p["name"] == ours), None)
    if our_role is None or winner is None or not rosters["p1"] or not rosters["p2"]:
        return None
    opp_role = "p2" if our_role == "p1" else "p1"
    our_faints = [f for f in faints if f[1] == our_role]
    opp_faints = [f for f in faints if f[1] == opp_role]
    first = faints[0] if faints else None
    return {
        "won": winner == ours,
        "forfeit": forfeit,
        "turns": turn,
        "our_rating": players[our_role]["rating"],
        "opp_rating": players[opp_role]["rating"],
        "opp_name": players[opp_role]["name"],
        "our_roster": rosters[our_role],
        "opp_roster": rosters[opp_role],
        "our_leads": leads[our_role],
        "opp_leads": leads[opp_role],
        "our_brought": brought[our_role],
        "opp_brought": brought[opp_role],
        "first_faint_ours": None if first is None else first[1] == our_role,
        "first_faint_turn": None if first is None else first[0],
        "first_faint_species": None if first is None else first[2],
        "our_faints": len(our_faints),
        "opp_faints": len(opp_faints),
        "our_faint_species": [f[2] for f in our_faints],
        "opp_tr": any(r == opp_role for _, r in tr_set),
        "opp_tr_turn": next((t for t, r in tr_set if r == opp_role), None),
        "our_tr": any(r == our_role for _, r in tr_set),
        "opp_tailwind": any(r == opp_role for _, r in tailwind),
        "our_tailwind": any(r == our_role for _, r in tailwind),
        "our_tailwind_turn": next((t for t, r in tailwind if r == our_role), None),
        "weather": weather[:3],
        "our_tera": tera[our_role][:1],
        "opp_tera": tera[opp_role][:1],
        "our_mega": mega[our_role][:1],
        "opp_mega": mega[opp_role][:1],
        "our_protects": protects[our_role],
        "opp_protects": protects[opp_role],
        "our_switches": switches[our_role],
        "opp_switches": switches[opp_role],
        "our_se": se_hits[our_role],
        "opp_se": se_hits[opp_role],
        "our_resisted": resisted_hits[our_role],
        "opp_resisted": resisted_hits[opp_role],
        "our_immune": immune_hits[our_role],
        "opp_immune": immune_hits[opp_role],
        "our_attacks": attacks[our_role],
        "opp_attacks": attacks[opp_role],
    }


def load_decisions(directory: Path) -> dict[str, dict]:
    """Per-battle guard/confidence summary from decisions.jsonl (if present)."""
    path = directory / "decisions.jsonl"
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        summary = out.setdefault(
            rec.get("battle", ""),
            {"decisions": 0, "demoted_chosen": 0, "guard_changes": 0, "low_conf": 0},
        )
        summary["decisions"] += 1
        chosen = rec.get("chosen") or {}
        if chosen.get("demoted_by"):
            summary["demoted_chosen"] += 1
        guards = rec.get("guards") or {}
        if guards.get("demotions") or guards.get("vetoed"):
            summary["guard_changes"] += 1
        prob = chosen.get("policy_probability")
        if isinstance(prob, (int, float)) and prob < 0.3:
            summary["low_conf"] += 1
    return out


def collect(dirs: list[Path], ours: str) -> list[dict]:
    rows = []
    for directory in dirs:
        decisions = load_decisions(directory)
        for path in sorted(directory.glob("*.html")):
            log = extract_log(path)
            if log is None:
                continue
            row = parse_game(log, ours)
            if row is None:
                continue
            battle_id = path.stem.split(" - ", 1)[-1]
            row["battle"] = battle_id
            row["dir"] = directory.name
            row.update(decisions.get(battle_id, {}))
            rows.append(row)
    return rows


def wr(rows: list[dict]) -> str:
    n = len(rows)
    w = sum(r["won"] for r in rows)
    return f"{w}/{n} = {100 * w / n:5.1f}%" if n else "0/0"


def split(rows: list[dict], label: str, predicate) -> None:
    yes = [r for r in rows if predicate(r)]
    no = [r for r in rows if not predicate(r)]
    print(f"  {label:38s} yes {wr(yes):18s} no {wr(no)}")


def report(rows: list[dict]) -> dict:
    wins = [r for r in rows if r["won"]]
    losses = [r for r in rows if not r["won"]]
    print(f"games {len(rows)}: {wr(rows)}   forfeits {sum(r['forfeit'] for r in rows)}")
    if not rows:
        return {}

    def mean(key, subset):
        vals = [r[key] for r in subset if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else float("nan")

    print("\nmeans                     wins    losses")
    for key in (
        "turns",
        "opp_rating",
        "our_faints",
        "opp_faints",
        "first_faint_turn",
        "our_protects",
        "opp_protects",
        "our_switches",
        "opp_switches",
        "our_se",
        "opp_se",
        "our_resisted",
        "opp_resisted",
        "our_immune",
        "our_attacks",
        "opp_attacks",
        "decisions",
        "demoted_chosen",
        "guard_changes",
        "low_conf",
    ):
        print(f"  {key:22s} {mean(key, wins):7.2f} {mean(key, losses):7.2f}")

    print("\nsplits (win rate when condition holds vs not)")
    split(rows, "first faint ours", lambda r: r["first_faint_ours"] is True)
    split(rows, "opponent set Trick Room", lambda r: r["opp_tr"])
    split(rows, "opponent set Tailwind", lambda r: r["opp_tailwind"])
    split(rows, "we set Tailwind", lambda r: r["our_tailwind"])
    split(
        rows, "we set Tailwind by turn 2", lambda r: (r["our_tailwind_turn"] or 99) <= 2
    )
    split(rows, "we terastallized", lambda r: bool(r["our_tera"]))
    split(rows, "opponent terastallized", lambda r: bool(r["opp_tera"]))
    split(rows, "we mega evolved", lambda r: bool(r["our_mega"]))
    split(rows, "opponent mega evolved", lambda r: bool(r["opp_mega"]))
    split(rows, "opponent rating >= 1200", lambda r: (r["opp_rating"] or 0) >= 1200)
    split(rows, "opponent rating >= 1300", lambda r: (r["opp_rating"] or 0) >= 1300)
    split(rows, "game <= 6 turns", lambda r: r["turns"] <= 6)
    split(rows, "game >= 12 turns", lambda r: r["turns"] >= 12)
    split(rows, "our SE hits >= 2", lambda r: r["our_se"] >= 2)
    split(rows, "our resisted hits >= 2", lambda r: r["our_resisted"] >= 2)
    split(rows, "we switched >= 2 times", lambda r: r["our_switches"] >= 2)
    split(rows, "opp switched >= 2 times", lambda r: r["opp_switches"] >= 2)
    split(rows, "guard changed a pick", lambda r: (r.get("demoted_chosen") or 0) > 0)

    print("\nblowouts: losses by our remaining mons (4 brought - faints)")
    remain = collections.Counter(4 - min(4, r["opp_faints"]) for r in losses)
    print("  opp mons left when we lost:", dict(sorted(remain.items())))
    remain_w = collections.Counter(4 - min(4, r["our_faints"]) for r in wins)
    print("  our mons left when we won:", dict(sorted(remain_w.items())))

    def table(title, key_fn, min_n=4, top=15):
        groups: dict = collections.defaultdict(list)
        for r in rows:
            for k in key_fn(r):
                groups[k].append(r)
        items = [(k, v) for k, v in groups.items() if len(v) >= min_n]
        items.sort(
            key=lambda kv: (sum(x["won"] for x in kv[1]) / len(kv[1]), -len(kv[1]))
        )
        print(f"\n{title} (n >= {min_n}; worst first)")
        for k, v in items[:top]:
            print(f"  {str(k):44s} {wr(v)}")
        return {str(k): [sum(x["won"] for x in v), len(v)] for k, v in items}

    out: dict[str, object] = {"n": len(rows), "wins": len(wins)}
    out["by_opp_species"] = table(
        "win rate when the opponent BROUGHT species", lambda r: r["opp_brought"]
    )
    out["by_opp_lead"] = table(
        "win rate vs opponent lead pair",
        lambda r: [tuple(sorted(r["opp_leads"]))],
        min_n=3,
    )
    out["by_our_lead"] = table(
        "win rate by OUR lead pair",
        lambda r: [tuple(sorted(r["our_leads"]))],
        min_n=3,
        top=12,
    )
    out["by_our_bring"] = table(
        "win rate by OUR brought four",
        lambda r: [tuple(sorted(r["our_brought"]))],
        min_n=3,
        top=12,
    )
    out["by_first_faint_species"] = table(
        "first faint species (ours) -> win rate",
        lambda r: [r["first_faint_species"]] if r["first_faint_ours"] else [],
        min_n=3,
    )
    out["by_our_tera"] = table(
        "our tera species -> win rate", lambda r: [t[1] for t in r["our_tera"]], min_n=3
    )
    out["by_opp_tera"] = table(
        "opponent tera species -> win rate",
        lambda r: [t[1] for t in r["opp_tera"]],
        min_n=3,
    )
    print(
        "\nour lead pairs (frequency):",
        collections.Counter(tuple(sorted(r["our_leads"])) for r in rows).most_common(6),
    )
    print(
        "opponent repeated (>=2 games):",
        [
            (k, v)
            for k, v in collections.Counter(r["opp_name"] for r in rows).most_common()
            if v >= 2
        ][:12],
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--ours", default=OUR_NAME)
    ap.add_argument(
        "--json",
        type=Path,
        default=None,
        help="write the per-game rows and tables here",
    )
    ap.add_argument(
        "--per-dir", action="store_true", help="also report each directory on its own"
    )
    args = ap.parse_args()
    rows = collect(args.dirs, args.ours)
    if args.per_dir:
        for directory in args.dirs:
            print(f"\n===== {directory} =====")
            report([r for r in rows if r["dir"] == directory.name])
    print(f"\n===== pooled: {', '.join(d.name for d in args.dirs)} =====")
    tables = report(rows)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"tables": tables, "games": rows}, indent=1, default=str)
        )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
