"""Read a ladder trial played with the exact search on (2026-10-09).

Per replay directory (one configuration each), from its replays, ``decisions.jsonl``
(the search's audit rows) and ``decisions_champion.jsonl`` (the bot's own decision
rows, which carry the opponent predictor's shadow forecast when the search plays):

  1. does the search run: decisions searched, fallbacks by cause, time per decision
  2. was the opponent's real reply in the search's table (hidden / open sheets)
  3. the overrides: count, expected edge, the edge in the cells of the reply made
  4. the worlds: agreement with what the opponent has shown, redraws, failures
  5. the predictor's forecast on the same decisions: the real reply pair in its eight
     likeliest joint replies, by the search's own matching rule, against the table
  5b. when the run logged it (--search-reply-prior forecast, any weight): where the
      real reply ranks, world by world, under the brain, the forecast and sums of them
  6. each game: result, rating, who lost a Pokemon first, the overrides

Usage (repo root):
    .venv/bin/python evaluation/search_ladder_read.py <replay dir> [--games] [--json F]
    .venv/bin/python evaluation/search_ladder_read.py --pool name=F.json [name=...]
"""

from __future__ import annotations

import collections
import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.ladder_loss_profile import OUR_NAME, extract_log, parse_game  # noqa: E402

SCHEMES = ("brain", "sum25", "sum50", "sum75", "forecast")
WIDTHS = (1, 2, 4, 6, 8, 10, 12, 16, 24)
PROTECTS = frozenset(
    "protect detect spikyshield kingsshield wideguard quickguard banefulbunker "
    "burningbulwark silktrap obstruct".split()
)
NOT_LEGAL = 10**6  # the rank of a reply no world lists


def wilson(wins: float, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.96
    p = wins / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def sign_test(first: int, second: int) -> float:
    """Two-sided exact binomial p of a ``first`` : ``second`` split under even odds
    (the decisions on which two reply priors differ: which one held the reply)."""
    n = first + second
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(first, second) + 1)) / 2**n
    return min(1.0, 2 * tail)


def share(hits: float, total: float) -> str:
    """ "12 of 40 (30%)"; "0 of 0" without a total."""
    if not total:
        return f"{hits:g} of 0"
    return f"{hits:g} of {total:g} ({100 * hits / total:.0f}%)"


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def load_rows(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a row being written
    return rows


def game_rows(directory: Path, ours: str) -> dict[str, dict]:
    """One row per finished game (tools/ladder_loss_profile.parse_game), by tag."""
    games: dict[str, dict] = {}
    replays = sorted(directory.glob("*.html"), key=lambda p: p.stat().st_mtime)
    for path in replays:
        log = extract_log(path)
        if not log:
            continue
        row = parse_game(log, ours) or {}
        row["tag"] = "battle-" + path.stem.split(" - battle-", 1)[-1]
        # either side's clock: which side is in the result of the game
        row["a_clock_ran_out"] = "lost due to inactivity" in log
        games[row["tag"]] = row
    return games


def reply_kind(observed: dict[str, list]) -> str:
    """What the opponent's reply was made of: a switch, else a Protect, else moves."""
    kinds = {
        "switch" if seen[0] == "switch" else "protect" if seen[1] in PROTECTS else "m"
        for seen in observed.values()
    }
    if "switch" in kinds:
        return "with a switch"
    return "with a protect" if "protect" in kinds else "moves only"


def forecast_pairs(record: dict, k: int = 8, mega: bool = True) -> list[tuple]:
    """The k likeliest joint replies of a logged forecast, as (slot a, slot b, who
    Mega-evolves, probability): the product of the two slots' logged marginals (their
    eight likeliest each, which hold the joint's eight likeliest) and, with ``mega``,
    of each slot's Mega marginal -- without what the rules forbid (both switching to
    the same Pokemon, both Mega-evolving, a switch with a Mega Evolution)."""
    nobody = [(("none", "", None), 1.0)]
    per_slot, p_mega = [], []
    for slot in list(record.get("slots") or [])[:2]:
        if not slot:  # nobody stands there
            per_slot.append(nobody)
            p_mega.append(0.0)
            continue
        actions = [
            (
                (
                    a.get("kind"),
                    a.get("move") or a.get("species") or "",
                    a.get("target"),
                ),
                float(a.get("p") or 0.0),
            )
            for a in slot.get("actions") or []
        ]
        per_slot.append(actions or nobody)
        p_mega.append(float(slot.get("p_mega") or 0.0))
    while len(per_slot) < 2:
        per_slot.append(nobody)
        p_mega.append(0.0)
    pairs = []
    for (a, pa), (b, pb) in itertools.product(per_slot[0], per_slot[1]):
        if a[0] == "switch" and b[0] == "switch" and a[1] == b[1]:
            continue
        if not mega:
            pairs.append((a, b, None, pa * pb))
            continue
        may_a = a[0] == "move" and p_mega[0] > 0
        may_b = b[0] == "move" and p_mega[1] > 0
        stay_a = 1 - p_mega[0] if may_a else 1.0
        stay_b = 1 - p_mega[1] if may_b else 1.0
        pairs.append((a, b, (False, False), pa * pb * stay_a * stay_b))
        if may_a:
            pairs.append((a, b, (True, False), pa * pb * p_mega[0] * stay_b))
        if may_b:
            pairs.append((a, b, (False, True), pa * pb * stay_a * p_mega[1]))
    pairs.sort(key=lambda item: -item[3])
    return pairs[:k]


def forecast_matches(action: tuple, observed: list | None, strict: bool) -> bool:
    """Does one slot's forecast action name what that slot was seen to do?
    ``observed`` is the search's record, [kind, identifier, target, mega]; a slot
    that showed nothing (it could not move, or was empty) matches anything."""
    kind, identifier, target = action
    if observed is None:
        return True
    if kind in ("other", "none"):
        return False
    if kind != observed[0] or identifier != observed[1]:
        return False
    seen = observed[2]
    if not strict or kind != "move" or seen is None:
        return True
    if target in (None, "auto", "self", "field"):
        return True
    wanted = {"foe_a": 1, "foe_b": 2}.get(target)
    return seen < 0 if wanted is None else seen == wanted


def _forecast_hit(pairs: list[tuple], observed: dict, strict: bool, megas=None) -> bool:
    return any(
        forecast_matches(a, observed.get(0), strict)
        and forecast_matches(b, observed.get(1), strict)
        and (megas is None or state == megas)
        for a, b, state, _ in pairs
    )


def read(directory: Path, ours: str = OUR_NAME) -> dict[str, Any]:
    """Everything the six readings need from one replay directory, as plain data."""
    games = game_rows(directory, ours)
    audits = [
        r for r in load_rows(directory / "decisions.jsonl") if "exact_search" in r
    ]
    own = [
        r for r in load_rows(directory / "decisions_champion.jsonl") if r.get("turn")
    ]
    by_battle: dict[str, list[dict]] = collections.defaultdict(list)
    for row in audits:
        by_battle[row["battle"]].append(row)
    sheets_of = {
        tag: "open"
        if any(r["exact_search"].get("open_sheet") for r in rows)
        else "hidden"
        for tag, rows in by_battle.items()
    }
    finished = [g for g in games.values() if g.get("won") is not None]
    record: dict[str, Any] = {
        "games": len(finished),
        "wins": sum(bool(g["won"]) for g in finished),
    }
    for kind in ("hidden", "open"):
        subset = [g for g in finished if sheets_of.get(g["tag"]) == kind]
        record[kind] = {
            "games": len(subset),
            "wins": sum(bool(g["won"]) for g in subset),
        }

    # 1. does the search run
    modes: collections.Counter[str] = collections.Counter()
    reasons: collections.Counter[str] = collections.Counter()
    seconds: list[float] = []
    per_game: dict[str, float] = collections.defaultdict(float)
    for row in audits:
        audit = row["exact_search"]
        schedule = audit.get("schedule") or {}
        mode = schedule.get("mode", "?")
        modes[mode] += 1
        result = audit.get("result") or {}
        if mode != "search":
            fallback = audit.get("decision_fallback") or {}
            text = fallback.get("error") or ",".join(schedule.get("reasons") or ["?"])
            reasons[f"{mode}: {str(text).splitlines()[0][:90]}"] += 1
        elif result.get("elapsed_s") is not None:
            spent = float(result["elapsed_s"])
            spent += float(schedule.get("preparation_elapsed_s") or 0.0)
            seconds.append(spent)
            per_game[row["battle"]] += spent
    search = {
        "decisions": sum(modes.values()),
        "modes": dict(modes),
        "reasons": dict(reasons.most_common(10)),
        "seconds": {
            name: quantile(seconds, q)
            for name, q in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("max", 1.0))
        },
        "at_7s_or_more": sum(s >= 7.0 for s in seconds),
        "at_9s_or_more": sum(s >= 9.0 for s in seconds),
        "most_seconds_in_one_game": max(per_game.values()) if per_game else None,
    }

    # 2. the reply table, 3. overrides, 4. worlds, 5b. other priors
    coverage = {"hidden": [0, 0, 0.0], "open": [0, 0, 0.0]}
    kinds: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    overrides: list[dict] = []
    worlds: collections.Counter[str] = collections.Counter()
    errors: collections.Counter[str] = collections.Counter()
    priors: dict[str, Any] = {
        sheets: {
            "decisions": 0,
            "any": collections.Counter(),
            "mass": collections.Counter(),
        }
        for sheets in ("hidden", "open")
    }
    ranks: dict[str, list[int]] = {scheme: [] for scheme in SCHEMES}
    # the table as played against the brain's ranking alone, decision by decision
    paired: collections.Counter[str] = collections.Counter()
    prior_kinds: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for tag, rows in by_battle.items():
        previous: dict = {}
        for index, row in enumerate(rows):
            audit = row["exact_search"]
            schedule = audit.get("schedule") or {}
            sheets = "open" if audit.get("open_sheet") else "hidden"
            searched = schedule.get("mode") == "search"
            cov = audit.get("reply_coverage") or {}
            if cov:
                cell = coverage[sheets]
                cell[0] += 1
                cell[1] += int(bool(cov.get("any")))
                cell[2] += float(cov.get("mass_with_reply") or 0.0)
                if cov.get("observed"):
                    kind = reply_kind(cov["observed"])
                    kinds[kind][0] += 1
                    kinds[kind][1] += int(bool(cov.get("any")))
            logged = cov.get("priors") or {}
            if "top_mass" in logged:
                cell = priors[sheets]
                cell["decisions"] += 1
                cell["any"]["table"] += int(bool(cov.get("any")))
                cell["mass"]["table"] += float(cov.get("mass_with_reply") or 0.0)
                kind = reply_kind(cov.get("observed") or {})
                prior_kinds[kind]["n"] += 1
                played, alone = bool(cov.get("any")), bool(logged["any"].get("brain"))
                paired[
                    ("both", "played only", "brain only", "neither")[
                        (0 if played else 2) + (0 if alone else 1)
                    ]
                ] += 1
                for scheme in SCHEMES:
                    cell["any"][scheme] += int(bool(logged["any"].get(scheme)))
                    cell["mass"][scheme] += float(logged["top_mass"].get(scheme) or 0)
                    ranks[scheme].append(logged["best_rank"].get(scheme) or NOT_LEGAL)
                    prior_kinds[kind][scheme] += int(bool(logged["any"].get(scheme)))
            elif "failed" in logged:
                worlds[f"prior comparison failed: {logged['failed']}"] += 1
            champion, played = audit.get("champion_actions"), audit.get("actions")
            if searched and champion is not None and played not in (None, champion):
                result = audit.get("result") or {}
                later = rows[index + 1]["exact_search"] if index + 1 < len(rows) else {}
                edges = (later.get("reply_coverage") or {}).get("override") or {}
                overrides.append(
                    {
                        "battle": tag,
                        "turn": row.get("turn"),
                        "sheets": sheets,
                        "bots_own": audit.get("champion_choice"),
                        "played": result.get("choice"),
                        "expected_edge": edges.get("expected_edge"),
                        "realized_edge": edges.get("realized_edge"),
                        "replacement": 0 in [int(a) for a in champion]
                        and all(int(a) < 7 for a in played),
                    }
                )
            worlds[f"{sheets}: decisions"] += 1
            agreeing = audit.get("roots_agreeing_with_evidence")
            if sheets == "hidden" and agreeing is not None:
                worlds["hidden: fewer than 4 worlds agree"] += int(agreeing < 4)
                worlds["hidden: no world agrees"] += int(agreeing == 0)
            redrawn = (audit.get("evidence_resamples") or 0) > (
                previous.get("evidence_resamples") or 0
            )
            worlds[f"{sheets}: worlds redrawn"] += int(redrawn)
            if searched and champion is not None and not audit.get("champion_choice"):
                worlds[f"{sheets}: the bot's own pair not expressible"] += 1
            for text in audit.get("last_reconcile_errors") or []:
                text = str(text).split("\n")[0]
                text = re.sub(
                    r"snapshot Pokemon \w+", "snapshot Pokemon <species>", text
                )
                errors[f"{sheets} | {re.sub(r'\\[.*?\\]', '[...]', text)[:100]}"] += 1
            previous = audit

    # 5. the predictor's own eight likeliest joint replies on the same decisions
    forecast_at = {
        (row["battle"], row["turn"]): row["opponent_forecast"]
        for row in own
        if "slots" in (row.get("opponent_forecast") or {})
    }
    tally: collections.Counter[str] = collections.Counter()
    for tag, rows in by_battle.items():
        moves = []  # this battle's move decisions so far, in order
        for row in rows:
            audit = row["exact_search"]
            cov = audit.get("reply_coverage") or {}
            if cov.get("observed") and moves:
                # this row's coverage is about the reply to the move decision before
                sheets = "open" if audit.get("open_sheet") else "hidden"
                tally[f"{sheets}: replies"] += 1
                in_table = bool(cov.get("any"))
                tally[f"{sheets}: table"] += int(in_table)
                forecast = forecast_at.get((tag, moves[-1].get("turn")))
                if forecast is None:
                    tally[f"{sheets}: no forecast"] += 1
                else:
                    observed = {int(k): v for k, v in cov["observed"].items()}
                    megas = tuple(
                        bool((observed.get(s) or [0, 0, 0, 0])[3]) for s in (0, 1)
                    )
                    with_mega = forecast_pairs(forecast, mega=True)
                    plain = forecast_pairs(forecast, mega=False)
                    same = _forecast_hit(with_mega, observed, True, megas)
                    tally[f"{sheets}: forecast"] += int(same)
                    tally[f"{sheets}: forecast, moves only"] += int(
                        _forecast_hit(plain, observed, False)
                    )
                    tally[f"{sheets}: forecast top 1"] += int(
                        _forecast_hit(with_mega[:1], observed, True, megas)
                    )
                    key = ("both", "table only", "forecast only", "neither")[
                        (0 if in_table else 2) + (0 if same else 1)
                    ]
                    tally[f"{sheets}: {key}"] += 1
            champion = audit.get("champion_actions") or [9, 9]
            forced = 0 in [int(a) for a in champion] and all(
                int(a) < 7 for a in champion
            )
            if not forced:
                moves.append(row)

    # 6. the games
    listing = []
    for tag, game in games.items():
        mine = [o for o in overrides if o["battle"] == tag]
        first = game.get("first_faint_ours")
        listing.append(
            {
                "tag": tag.split("-")[2] if tag.count("-") >= 2 else tag,
                "won": game.get("won"),
                "turns": game.get("turns"),
                "sheets": sheets_of.get(tag),
                "our_rating": game.get("our_rating"),
                "opp_rating": game.get("opp_rating"),
                "our_leads": game.get("our_leads"),
                "opp_leads": game.get("opp_leads"),
                "first_faint": None if first is None else "ours" if first else "theirs",
                "first_faint_turn": game.get("first_faint_turn"),
                "opponent_forfeited": bool(game.get("forfeit"))
                and bool(game.get("won")),
                "a_clock_ran_out": game.get("a_clock_ran_out"),
                "searched": len(by_battle.get(tag, [])),
                "overrides": [(o["turn"], o["bots_own"], o["played"]) for o in mine],
            }
        )
    return {
        "directory": str(directory),
        "record": record,
        "search": search,
        "reply_table": {
            sheets: {"decisions": c[0], "in_some_world": c[1], "world_mass": c[2]}
            for sheets, c in coverage.items()
        },
        "reply_table_by_kind": {
            k: {"replies": v[0], "in_table": v[1]} for k, v in kinds.items()
        },
        "overrides": overrides,
        "worlds": dict(worlds),
        "world_errors": dict(errors.most_common(12)),
        "forecast": dict(tally),
        "priors": {
            sheets: {
                "decisions": cell["decisions"],
                "any": dict(cell["any"]),
                "mass": dict(cell["mass"]),
            }
            for sheets, cell in priors.items()
        },
        "prior_ranks": ranks,
        "played_against_brain": dict(paired),
        "priors_by_kind": {kind: dict(c) for kind, c in prior_kinds.items()},
        "games": listing,
    }


def render(reading: dict[str, Any], games: bool = False) -> list[str]:
    out: list[str] = []
    record = reading["record"]
    low, high = wilson(record["wins"], record["games"])
    out.append(
        f"GAMES {share(record['wins'], record['games'])} "
        f"[{100 * low:.1f}, {100 * high:.1f}]"
    )
    for kind in ("hidden", "open"):
        if record[kind]["games"]:
            cell = record[kind]
            out.append(f"  {kind} sheets: {share(cell['wins'], cell['games'])}")
    search = reading["search"]
    modes = search["modes"]
    out.append(f"\n1. SEARCH  decisions {search['decisions']}  modes {modes}")
    times = search["seconds"]
    done = modes.get("search", 0)
    out.append(
        f"   seconds a decision: p50 {times['p50']:.1f}  p90 {times['p90']:.1f}  "
        f"p99 {times['p99']:.1f}  max {times['max']:.1f};  7 s or more: "
        f"{share(search['at_7s_or_more'], done)};  9 s or more: "
        f"{search['at_9s_or_more']}"
    )
    for text, count in search["reasons"].items():
        out.append(f"   {count:4d}  {text}")
    out.append("\n2. THE OPPONENT'S REAL REPLY IN THE SEARCH'S TABLE")
    for sheets, cell in reading["reply_table"].items():
        if cell["decisions"]:
            mass = 100 * cell["world_mass"] / cell["decisions"]
            held = share(cell["in_some_world"], cell["decisions"])
            out.append(
                f"   {sheets:6s} sheets: {held} in some world; mean share of "
                f"worlds holding it {mass:.0f}%"
            )
    for kind, cell in reading["reply_table_by_kind"].items():
        out.append(f"   {kind:15s} {share(cell['in_table'], cell['replies'])}")
    overrides = reading["overrides"]
    expected = [o["expected_edge"] for o in overrides if o["expected_edge"] is not None]
    realized = [o["realized_edge"] for o in overrides if o["realized_edge"] is not None]
    out.append(
        f"\n3. OVERRIDES  {share(len(overrides), done)} of searched decisions, "
        f"{sum(o['replacement'] for o in overrides)} at forced replacements"
    )
    if expected:
        line = (
            f"   expected edge {sum(expected) / len(expected):+.3f} (n {len(expected)})"
        )
        if realized:
            line += (
                f";  reply was in the table for {len(realized)}: realized "
                f"{sum(realized) / len(realized):+.3f}, not positive in "
                f"{sum(e <= 0 for e in realized)}"
            )
        out.append(line)
    out.append("\n4. THE WORLDS")
    for key in sorted(reading["worlds"]):
        out.append(f"   {key:44s} {reading['worlds'][key]}")
    for text, count in reading["world_errors"].items():
        out.append(f"   {count:4d}  {text}")
    tally = reading["forecast"]
    out.append("\n5. THE PREDICTOR'S OWN EIGHT LIKELIEST REPLY PAIRS, SAME DECISIONS")
    for sheets in ("hidden", "open"):
        n = tally.get(f"{sheets}: replies", 0)
        if not n:
            continue
        scored = n - tally.get(f"{sheets}: no forecast", 0)

        def got(key: str, sheets: str = sheets, scored: int = scored) -> str:
            return share(tally.get(f"{sheets}: {key}", 0), scored)

        out.append(
            f"   {sheets:6s}: table {share(tally.get(f'{sheets}: table', 0), n)};  "
            f"forecast {got('forecast')} by the same rule, "
            f"{got('forecast, moves only')} on moves alone, its first "
            f"{got('forecast top 1')}"
        )
        out.append(
            "           "
            + ", ".join(
                f"{key} {tally.get(f'{sheets}: {key}', 0)}"
                for key in ("both", "forecast only", "table only", "neither")
            )
        )
    ranks = reading["prior_ranks"]
    n = len(ranks["brain"])
    if n:
        out.append(
            "\n5b. THE REAL REPLY UNDER OTHER REPLY PRIORS (logged world by world)"
        )
        for sheets, cell in reading["priors"].items():
            if cell["decisions"]:
                d = cell["decisions"]
                cells = "  ".join(
                    f"{name} {100 * cell['any'].get(name, 0) / d:.0f}%"
                    f" (mass {100 * cell['mass'].get(name, 0) / d:.0f}%)"
                    for name in ("table", *SCHEMES)
                )
                out.append(f"   {sheets} n={d}, within the table's width: {cells}")
        out.append("   within the first k replies of some world:")
        out.append("      k   " + " ".join(f"{scheme:>9s}" for scheme in SCHEMES))
        for k in WIDTHS:
            cells = " ".join(
                f"{100 * sum(r <= k for r in ranks[scheme]) / n:8.0f}%"
                for scheme in SCHEMES
            )
            out.append(f"   {k:4d}   {cells}")
        paired = reading.get("played_against_brain") or {}
        only, alone = paired.get("played only", 0), paired.get("brain only", 0)
        out.append(
            "   the table as played against the brain's ranking alone, same decisions: "
            f"both {paired.get('both', 0)}, played only {only}, brain only {alone}, "
            f"neither {paired.get('neither', 0)};  exact two-sided p "
            f"{sign_test(only, alone):.4f}"
        )
        nowhere = sum(r >= NOT_LEGAL for r in ranks["brain"])
        out.append(f"   legal in no world (no prior can list it): {share(nowhere, n)}")
        for kind, cell in reading["priors_by_kind"].items():
            cells = "  ".join(
                f"{scheme} {share(cell.get(scheme, 0), cell['n'])}"
                for scheme in SCHEMES
            )
            out.append(f"   {kind:15s} {cells}")
    listing = [g for g in reading["games"] if g["won"] is not None]
    with_override = [g for g in listing if g["overrides"]]
    without = [g for g in listing if not g["overrides"]]
    out.append("\n6. GAMES")
    if games:
        for game in reading["games"]:
            result = "?" if game["won"] is None else "W" if game["won"] else "L"
            notes = "  opponent forfeited" if game["opponent_forfeited"] else ""
            notes += "  a clock ran out" if game["a_clock_ran_out"] else ""
            out.append(
                f"   {game['tag']} {result} turns {game['turns']} {game['sheets']} "
                f"rating {game['our_rating']} vs {game['opp_rating']}  "
                f"{'+'.join(game['our_leads'] or [])} vs "
                f"{'+'.join(game['opp_leads'] or [])}  first faint "
                f"{game['first_faint']} (turn {game['first_faint_turn']})  searched "
                f"{game['searched']}{notes}"
            )
            for turn, was, now in game["overrides"]:
                out.append(f"        turn {turn}: {was}  ->  {now}")
    out.append(
        f"   games with an override: "
        f"{share(sum(g['won'] for g in with_override), len(with_override))} won;  "
        f"without: {share(sum(g['won'] for g in without), len(without))}"
    )
    return out


def pool(stages: list[tuple[str, dict]]) -> list[str]:
    """Several directories' readings (their --json files) as one trial."""
    out = []
    games = wins = decisions = searched = 0
    sheets = {"hidden": [0, 0], "open": [0, 0]}
    table = {"hidden": [0, 0], "open": [0, 0]}
    forecast: collections.Counter[str] = collections.Counter()
    overrides: list[dict] = []
    listing: list[dict] = []
    for name, reading in stages:
        record, search = reading["record"], reading["search"]
        games += record["games"]
        wins += record["wins"]
        decisions += search["decisions"]
        searched += search["modes"].get("search", 0)
        for kind in sheets:
            sheets[kind][0] += record[kind]["games"]
            sheets[kind][1] += record[kind]["wins"]
            table[kind][0] += reading["reply_table"][kind]["decisions"]
            table[kind][1] += reading["reply_table"][kind]["in_some_world"]
        forecast.update(reading["forecast"])
        overrides += reading["overrides"]
        listing += [g for g in reading["games"] if g["won"] is not None]
        out.append(
            f"{name:10s} {share(record['wins'], record['games'])}   searched "
            f"{share(search['modes'].get('search', 0), search['decisions'])}   "
            f"overrides {len(reading['overrides'])}"
        )
    low, high = wilson(wins, games)
    out.append(f"\nALL  {share(wins, games)} [{100 * low:.1f}, {100 * high:.1f}]")
    for kind, (n, won) in sheets.items():
        if n:
            out.append(f"  {kind} sheets: {share(won, n)}")
    out.append(
        f"  decisions {decisions}, searched {share(searched, decisions)}, overrides "
        f"{share(len(overrides), searched)} of searched "
        f"({sum(o['replacement'] for o in overrides)} at replacements)"
    )
    for kind, (n, hit) in table.items():
        if n:
            scored = forecast[f"{kind}: replies"] - forecast[f"{kind}: no forecast"]
            out.append(
                f"  real reply, {kind}: in the table {share(hit, n)};  in the "
                f"forecast's eight {share(forecast[f'{kind}: forecast'], scored)};  "
                + ", ".join(
                    f"{key} {forecast[f'{kind}: {key}']}"
                    for key in ("both", "forecast only", "table only", "neither")
                )
            )
    for label, subset in (
        ("with an override", [g for g in listing if g["overrides"]]),
        ("without", [g for g in listing if not g["overrides"]]),
        ("first faint ours", [g for g in listing if g["first_faint"] == "ours"]),
        ("first faint theirs", [g for g in listing if g["first_faint"] == "theirs"]),
    ):
        out.append(
            f"  games {label}: {share(sum(g['won'] for g in subset), len(subset))}"
        )
    forfeits = sum(g["opponent_forfeited"] for g in listing)
    out.append(f"  wins by the opponent's forfeit: {forfeits}")
    ratings = [g["our_rating"] for g in listing if g["our_rating"]]
    if ratings:
        out.append(
            f"  our rating at the start of each game: first {ratings[0]}, low "
            f"{min(ratings)}, high {max(ratings)}, last {ratings[-1]}"
        )
    return out


def main() -> None:
    if "--pool" in sys.argv:
        named = [arg.split("=", 1) for arg in sys.argv[sys.argv.index("--pool") + 1 :]]
        stages = [(name, json.loads(Path(path).read_text())) for name, path in named]
        print("\n".join(pool(stages)))
        return
    reading = read(Path(sys.argv[1]))
    print("\n".join(render(reading, games="--games" in sys.argv)))
    if "--json" in sys.argv:
        target = Path(sys.argv[sys.argv.index("--json") + 1])
        target.write_text(json.dumps(reading, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
