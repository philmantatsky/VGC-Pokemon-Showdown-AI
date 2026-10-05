"""Acceptance checks for the opponent predictor's state path (train/serve parity).

The predictor's features come from one event-fed tracker
(``vgc_bench.src.oppmodel.public_state``). This script checks that the offline
path (``drive_log``) and the live path (``LiveShadow`` over
``battle._replay_data``) build the same state, and that the label reader holds
up on real logs. Parts, chosen with ``--parts``:

  a  Every saved own Reg M-C replay (``ladder_replays_mc*/``,
     ``challenge_replays_mc*/``): the player-view stream is pushed through
     ``LiveShadow`` the way the bot would see it (one sync per ``|turn|``, then
     again one sync per event, then again with a mid-game reconnect that replays
     the log), and through ``drive_log`` on the HP-rewritten whole log. Snapshots
     must be identical turn by turn. Also checks that no exact HP of the bot
     survives the rewrite. Both arms share the rewrite rule and the tracker, so
     this part proves the cursor and rebuild logic, NOT the HP rule (parts b
     and c do that). A saved page never shows a team sheet: the bot's decision
     audit (``decisions.jsonl``, ``preview_shadow.open_sheet``) is read to
     report how many of these games were open-sheet in fact.
  b  Robustness and seat symmetry on the human corpus
     (``battle_logs_web_mc_20260927/{top,low}``; ``--union`` adds the JSON
     corpora): ``drive_log`` never raises, counters by reason, ms per log, the
     label census, p1/p2 symmetry (the same log with the sides renamed must
     give the mirrored snapshots and actions), and a synthetic player view of
     each log for each seat (own HP made exact with an invented max HP, by a
     port of the simulator's formula written here) that must give, through
     ``LiveShadow``, the snapshots of ``drive_log`` on the untouched log.
  c  Spectator versus player view of the SAME battle: the bot account's public
     replays are looked up (``--fetch``: at most 3 search requests and 15 replay
     fetches, 1 s apart, stored under ``results_oppmodel/parity/``) and compared
     turn by turn with the saved player-view page of the same battle.
  d  Independent cross-check against poke-env: a ``DoubleBattle`` driven
     straight from the same log must agree on species, HP, status, boosts,
     first-turn flag, faints, Mega use, weather, fields and side conditions.

A part that was asked for and compared nothing is a failure: exit 0 means the
checks ran. All parts are single-threaded CPU work. Usage (from the repo root):

  nice -n 19 .venv/bin/python evaluation/oppmodel_parity.py --parts a,b --sample 1500
  nice -n 19 .venv/bin/python evaluation/oppmodel_parity.py --parts c --fetch
"""

from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path as _Path

_os.environ.setdefault("OMP_NUM_THREADS", "1")
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import gzip
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Sequence

from vgc_bench.src.oppmodel.events import (
    CHATTER_KINDS,
    KIND_HIDDEN,
    KIND_MOVE,
    KIND_NONE,
    KIND_SWITCH,
    SIDES,
    Event,
    SheetSet,
    extract_battle_tag,
    extract_log_from_html,
    parse_ident,
    parse_showteam,
    read_log_actions,
    split_log,
)
from vgc_bench.src.oppmodel.public_state import (
    CONDITION_INDEX,
    DriveResult,
    LiveShadow,
    PublicSnapshot,
    drive_events,
    drive_log,
    rewrite_event,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results_oppmodel" / "parity"
REPLAY_HOST = "https://replay.pokemonshowdown.com"
USER_AGENT = "vgc-bench-research/0.1"
FORMAT_ID = "gen9championsvgc2026regmc"
MAX_SEARCH_REQUESTS = 3
MAX_REPLAY_FETCHES = 15
OWN_GLOBS = ("ladder_replays_mc*/*.html", "challenge_replays_mc*/*.html")
AUDIT_GLOBS = (
    "ladder_replays_mc*/decisions.jsonl",
    "challenge_replays_mc*/decisions.jsonl",
)
WEB_GLOBS = (
    "battle_logs_web_mc_20260927/top/*.log",
    "battle_logs_web_mc_20260927/low/*.log",
)
UNION_JSON = (
    "battle_logs_top_mc_merged_20260920/logs_gen9championsvgc2026regmc.json",
    "battle_logs_top_mc_merged_20260920/logs_gen9championsvgc2026regmcbo3.json",
    "battle_logs_players_t6like/logs_gen9championsvgc2026regmc.json",
    "battle_logs_players_t6like/logs_gen9championsvgc2026regmcbo3.json",
)
_OWN_NAME = re.compile(r"^(?P<player>.+?) - battle-(?P<id>[a-z0-9]+-\d+)")
_SIDE_FIELD = re.compile(r"^(\[of\] )?p([12])([ab]?)(:.*)?$", re.S)
_HP_TOKEN = re.compile(r"^(\d+)/(\d+)")
_PUBLIC_HP = re.compile(r"^(\d+)/100([a-z])?( [a-z]+)?$")
_AUDIT_BATTLE = re.compile(r"battle-([a-z0-9]+-\d+)")
_NO_SIDE_KINDS = CHATTER_KINDS - {"player"}
# Invented max HP values for the synthetic player view: multiples of 20, so the
# colour thresholds (a fifth, a half) are whole numbers, and at least 100, so
# every public percent is reachable.
_MAX_HP_CHOICES = (120, 140, 160, 180, 200, 220, 240, 300, 400)


class FakeBattle:
    """The three attributes ``LiveShadow`` reads from a live battle."""

    def __init__(self, battle_tag: str, player_role: str | None) -> None:
        self.battle_tag = battle_tag
        self.player_role = player_role
        self._replay_data: list[Event] = []


# --- shared helpers -----------------------------------------------------------


def diff_paths(left: Any, right: Any, path: str = "", limit: int = 6) -> list[str]:
    """Paths at which two plain structures differ (first ``limit``)."""
    out: list[str] = []

    def walk(a: Any, b: Any, where: str) -> None:
        if len(out) >= limit:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b), key=str):
                if key not in a or key not in b:
                    out.append(f"{where}.{key} (missing on one side)")
                else:
                    walk(a[key], b[key], f"{where}.{key}")
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                out.append(f"{where} (length {len(a)} vs {len(b)})")
                return
            for index, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{where}[{index}]")
        elif a != b:
            out.append(f"{where}: {a!r} vs {b!r}")

    walk(left, right, path)
    return out


def field_name(path: str) -> str:
    """A diff path with indices and side keys removed, for counting."""
    head = path.split(":", 1)[0].split(" (", 1)[0]
    head = re.sub(r"\[\d+\]", "[]", head)
    return re.sub(r"\.p[12]\b", ".<side>", head)


def own_replays() -> list[tuple[Path, str, str]]:
    """(path, account name, battle id) of every saved own Reg M-C replay page."""
    found: dict[str, tuple[Path, str, str]] = {}
    for pattern in OWN_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            match = _OWN_NAME.match(path.name)
            if match is not None and FORMAT_ID + "-" in path.name:
                found.setdefault(
                    match["id"], (path, match["player"].strip(), match["id"])
                )
    return [found[key] for key in sorted(found)]


def player_role(events: Sequence[Event], account: str) -> str | None:
    for event in events:
        if len(event) > 3 and event[1] == "player" and event[3] == account:
            return event[2]
    return None


def events_text(events: Sequence[Event]) -> str:
    return "\n".join("|".join(event) for event in events)


def live_snapshots(
    events: Sequence[Event],
    tag: str,
    role: str | None,
    every_event: bool = False,
    reconnect_at: int | None = None,
    sheets: dict[str, list[SheetSet]] | None = None,
    sheets_known: bool = False,
) -> tuple[dict[int, PublicSnapshot | None], LiveShadow]:
    """Push a stream through ``LiveShadow`` as the bot would; snapshot per turn.

    ``reconnect_at``: after that many events the server's replay of the log so
    far is appended to the same stream, as a reconnect does. ``sheets``: open
    team sheets handed over before the stream, as the runtime must do (the
    ``showteam`` line never reaches ``battle._replay_data``). ``sheets_known``:
    the runtime has said that every open sheet was handed over.
    """
    fake = FakeBattle(tag, role)
    shadow = LiveShadow()
    for side, sets in (sheets or {}).items():
        shadow.feed_sheet(side, sets)
    if sheets_known:
        shadow.mark_sheets_known()
    out: dict[int, PublicSnapshot | None] = {}
    for index, event in enumerate(events):
        if reconnect_at is not None and index == reconnect_at:
            fake._replay_data.extend(list(e) for e in events[:index])
            shadow.sync(fake)
        fake._replay_data.append(list(event))
        is_turn = len(event) > 2 and event[1] == "turn"
        if every_event or is_turn:
            snapshot = shadow.sync(fake)
            if is_turn:
                out[int(event[2])] = snapshot
    return out, shadow


def condition_tokens(events: Sequence[Event], role: str) -> list[str]:
    """Every condition string of ``role``'s side, in order, with its event type."""
    out: list[str] = []
    for event in events:
        index = CONDITION_INDEX.get(event[1]) if len(event) > 1 else None
        if index is not None and len(event) > index and event[2].startswith(role):
            out.append(f"{event[1]}|{event[2]}|{event[index]}")
    return out


def sheet_audit() -> dict[str, bool]:
    """Battle id -> open-sheet flag the bot recorded at team preview.

    The only record of the sheet state of a saved own game: the page itself
    never carries the ``showteam`` lines.
    """
    found: dict[str, bool] = {}
    for pattern in AUDIT_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
            except OSError:
                continue
            for line in lines:
                if '"open_sheet"' not in line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                shadow = row.get("preview_shadow") if isinstance(row, dict) else None
                match = _AUDIT_BATTLE.search(str(row.get("battle", "")))
                if isinstance(shadow, dict) and match is not None:
                    flag = shadow.get("open_sheet")
                    if isinstance(flag, bool):
                        found[match.group(1)] = flag
    return found


def simulator_public_hp(hp: int, maxhp: int) -> str:
    """Port of ``getHealth`` (pokemon-showdown/sim/pokemon.ts), Champions branch.

    Written here, apart from ``public_state.public_condition``, so the
    synthetic player view below is an independent check of that rule.
    """
    if not hp:
        return "0 fnt"
    percentage = (100 * hp) // maxhp or 1
    shared = f"{percentage}/100"
    if percentage == 20:
        shared += "y" if hp * 5 > maxhp else "r"
    elif percentage == 50:
        shared += "g" if hp * 2 > maxhp else "y"
    return shared


def synthetic_player_view(
    events: Sequence[Event], role: str, rng: random.Random, stats: Counter[str]
) -> list[Event]:
    """A spectator log as ``role`` would have received it.

    ``role``'s HP is made exact with an invented max HP per Pokemon, chosen so
    that the simulator's formula gives back the public string of the log; the
    ``showteam`` lines are dropped, as poke-env keeps them out of the stream.
    """
    maxhp: dict[str, int] = {}
    out: list[Event] = []
    for event in events:
        if len(event) > 1 and event[1] == "showteam":
            continue
        index = CONDITION_INDEX.get(event[1]) if len(event) > 1 else None
        if index is None or len(event) <= index or not event[2].startswith(role):
            out.append(list(event))
            continue
        match = _PUBLIC_HP.match(event[index])
        if match is None:
            out.append(list(event))  # "0 fnt": the same in both views
            continue
        ident = parse_ident(event[2])
        name = ident.name if ident is not None else event[2]
        big = maxhp.setdefault(name, rng.choice(_MAX_HP_CHOICES))
        percent, letter = int(match.group(1)), match.group(2) or ""
        hp = big if percent == 100 else -(-percent * big // 100)
        if percent in (20, 50) and letter in ("y", "g") and percent * big == 100 * hp:
            # The higher colour of a threshold percent: just above the threshold.
            if (percent, letter) != (50, "y"):
                hp += 1
        made = f"{hp}/{big}{match.group(3) or ''}"
        if simulator_public_hp(hp, big) + (match.group(3) or "") != event[index]:
            stats["seat_token_not_reproducible"] += 1
            out.append(list(event))
            continue
        stats["seat_tokens"] += 1
        changed = list(event)
        changed[index] = made
        if rewrite_event(changed, role)[index] != event[index]:
            stats["rewrite_roundtrip_mismatch"] += 1
        out.append(changed)
    return out


# --- part a -------------------------------------------------------------------


def part_a() -> dict[str, Any]:
    audit = sheet_audit()
    stats: Counter[str] = Counter()
    fields: Counter[str] = Counter()
    examples: list[str] = []
    counters: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    for path, account, battle_id in own_replays():
        stats["files"] += 1
        log = extract_log_from_html(path.read_text(encoding="utf-8", errors="replace"))
        if log is None:
            stats["no_log_block"] += 1
            continue
        tag = extract_battle_tag(log) or f"battle-{battle_id}"
        events = split_log(log)
        role = player_role(events, account)
        if role is None:
            stats["account_not_a_player"] += 1
            continue
        roles[role] += 1
        recorded = audit.get(battle_id)
        stats[
            "sheet_audit:no_record"
            if recorded is None
            else f"sheet_audit:open_sheet={recorded}"
        ] += 1
        rewritten = [rewrite_event(event, role) for event in events]
        for event in rewritten:
            index = CONDITION_INDEX.get(event[1]) if len(event) > 1 else None
            if index is None or len(event) <= index or not event[2].startswith(role):
                continue
            token = _HP_TOKEN.match(event[index])
            if token is not None and token.group(2) != "100":
                stats[f"exact_hp_left:{event[1]}"] += 1
        # A player-view stream cannot show a sheet: unknown on both arms.
        offline = drive_log(events_text(rewritten), tag, sheets_known=False)
        counters.update(offline.counters)
        modes = {
            "per_turn": (offline, live_snapshots(events, tag, role)),
            "per_event": (offline, live_snapshots(events, tag, role, every_event=True)),
            "reconnect": (
                offline,
                live_snapshots(events, tag, role, reconnect_at=len(events) // 2),
            ),
        }
        if recorded is False:
            # The audit says closed: both arms are told so.
            modes["told_closed"] = (
                drive_log(events_text(rewritten), tag),
                live_snapshots(events, tag, role, sheets_known=True),
            )
        for name, (reference, (live, shadow)) in modes.items():
            for key, value in shadow.counters.items():
                counters[f"live[{name}]:{key}"] += value
            for record in reference.turns:
                stats[f"turns[{name}]"] += 1
                snapshot = live.get(record.turn)
                if snapshot is None:
                    stats[f"live_none[{name}]"] += 1
                    continue
                diffs = diff_paths(snapshot.to_dict(), record.snapshot.to_dict())
                if diffs:
                    stats[f"turn_mismatch[{name}]"] += 1
                    for item in diffs:
                        fields[field_name(item)] += 1
                    if len(examples) < 8:
                        examples.append(
                            f"{battle_id} t{record.turn} {name}: {diffs[0]}"
                        )
        if not offline.turns:
            stats["no_turns"] += 1
    return {
        "compared": stats["turns[per_turn]"],
        "stats": dict(stats),
        "bot_role": dict(roles),
        "mismatch_fields": dict(fields.most_common()),
        "examples": examples,
        "counters": dict(counters.most_common()),
    }


# --- part b -------------------------------------------------------------------


def corpus(union: bool) -> Iterator[tuple[str, str]]:
    """(battle id, log) for the human Reg M-C corpus, de-duplicated by id."""
    seen: set[str] = set()
    for pattern in WEB_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            if path.stem not in seen:
                seen.add(path.stem)
                yield path.stem, path.read_text(encoding="utf-8", errors="replace")
    if not union:
        return
    for name in UNION_JSON:
        path = ROOT / name
        if not path.exists():
            continue
        for battle_id, row in json.loads(path.read_text(encoding="utf-8")).items():
            if battle_id not in seen and isinstance(row, list) and len(row) > 1:
                seen.add(battle_id)
                yield battle_id, str(row[1])


def swap_sides(events: Sequence[Event]) -> list[Event]:
    """The same log with p1 and p2 renamed into each other."""
    out: list[Event] = []
    for event in events:
        if len(event) < 3 or event[1] in _NO_SIDE_KINDS:
            out.append(list(event))
            continue
        swapped = list(event[:2])
        for part in event[2:]:
            match = _SIDE_FIELD.match(part)
            if match is not None:
                part = (
                    f"{match.group(1) or ''}p{3 - int(match.group(2))}"
                    f"{match.group(3)}{match.group(4) or ''}"
                )
            elif part.startswith("[spread] "):
                part = "[spread] " + ",".join(
                    f"p{3 - int(item[1])}{item[2:]}" if item[:2] in SIDES else item
                    for item in part[9:].split(",")
                )
            swapped.append(part)
        out.append(swapped)
    return out


def flip(value: Any) -> Any:
    """A snapshot / action dict with p1 and p2 exchanged."""
    other = {"p1": "p2", "p2": "p1"}
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for key, item in value.items():
            if key in ("side", "weather_side") and item in other:
                out[key] = other[item]
            elif key == "sides":
                out[key] = {other.get(k, k): flip(v) for k, v in item.items()}
            else:
                out[key] = flip(item)
        return out
    if isinstance(value, (list, tuple)):
        return [flip(item) for item in value]
    return value


def census_update(census: Counter[str], result: DriveResult) -> None:
    for record in result.turns:
        for action in record.actions.values():
            census["slot_turns"] += 1
            census[f"kind:{action.kind}"] += 1
            if action.kind in (KIND_HIDDEN, KIND_NONE):
                census[f"reason:{action.kind}:{action.reason}"] += 1
            if action.kind == KIND_MOVE:
                census[f"target:{action.target}"] += 1
                if action.reason and not action.locked:
                    census["move_named_by_cant"] += 1
                if action.is_protect:
                    census["protect_family"] += 1
                if action.intent is not None:
                    census[f"intent:{action.intent}"] += 1
                else:
                    census["intent:undecided"] += 1
            elif action.kind == KIND_SWITCH:
                census[f"intent:{action.intent}"] += 1
            if action.mega:
                census["mega"] += 1
                if action.kind in (KIND_HIDDEN, KIND_NONE):
                    census["mega_on_censored_slot"] += 1
            if not action.target_trusted:
                census["target_untrusted"] += 1
            if action.target_forced:
                census["target_forced"] += 1
            if action.locked:
                census["locked"] += 1
            if not action.switch_known:
                census["switch_unknown"] += 1
            if not action.protect_known:
                census["protect_unknown"] += 1


def census_table(census: Counter[str]) -> dict[str, Any]:
    total = max(1, census["slot_turns"])

    def share(*keys: str) -> float:
        return round(100.0 * sum(census[key] for key in keys) / total, 2)

    aimed = sum(
        census[f"target:{name}"] for name in ("foe_a", "foe_b", "ally", "self", "auto")
    )
    reasons = {
        key.split(":", 1)[1]: round(100.0 * value / total, 2)
        for key, value in sorted(census.items(), key=lambda item: -item[1])
        if key.startswith("reason:")
    }
    return {
        "slot_turns": census["slot_turns"],
        "move_pct": share("kind:move"),
        "move_target_known_pct": round(100.0 * aimed / total, 2),
        "move_target_blank_pct": share("target:None"),
        "switch_pct": share("kind:switch"),
        "hidden_pct": share("kind:hidden"),
        "none_pct": share("kind:none"),
        "reasons_pct": reasons,
        "mega_pct": share("mega"),
        "mega_on_censored_slot_pct": share("mega_on_censored_slot"),
        "target_untrusted_pct": share("target_untrusted"),
        "target_forced_pct": share("target_forced"),
        "locked_pct": share("locked"),
        "move_named_by_cant_pct": share("move_named_by_cant"),
        "protect_family_pct": share("protect_family"),
        "switch_unknown_pct": share("switch_unknown"),
        "protect_unknown_pct": share("protect_unknown"),
        "intent_pct": {
            key.split(":", 1)[1]: round(100.0 * value / total, 2)
            for key, value in sorted(census.items())
            if key.startswith("intent:")
        },
        "targets_pct": {
            key.split(":", 1)[1]: round(100.0 * value / total, 2)
            for key, value in sorted(census.items())
            if key.startswith("target:")
        },
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def part_b(sample: int, seed: int, union: bool) -> dict[str, Any]:
    logs = list(corpus(union))
    if sample and sample < len(logs):
        logs = random.Random(seed).sample(logs, sample)
    stats: Counter[str] = Counter()
    counters: Counter[str] = Counter()
    census: Counter[str] = Counter()
    label_counters: Counter[str] = Counter()
    symmetry: Counter[str] = Counter()
    seat: Counter[str] = Counter()
    seat_examples: list[str] = []
    rng = random.Random(seed)
    drive_ms: list[float] = []
    label_ms: list[float] = []
    examples: list[str] = []
    for battle_id, log in logs:
        stats["logs"] += 1
        started = time.perf_counter()
        try:
            result = drive_log(log, battle_id)
        except Exception as exc:  # the contract is that this never happens
            stats[f"RAISED:{type(exc).__name__}"] += 1
            continue
        drive_ms.append(1000.0 * (time.perf_counter() - started))
        events = split_log(log)
        started = time.perf_counter()
        read_log_actions(events, label_counters)
        label_ms.append(1000.0 * (time.perf_counter() - started))
        counters.update(result.counters)
        stats["turns"] += len(result.turns)
        stats["sheet_both" if all(result.sheets.values()) else "sheet_not_both"] += 1
        stats["rated_both" if all(result.ratings.values()) else "rated_not_both"] += 1
        if not result.finished:
            stats["no_win_or_tie_line"] += 1
        if not result.usable:
            continue
        stats["usable"] += 1
        census_update(census, result)
        mirrored = drive_events(swap_sides(events), battle_id)
        if len(mirrored.turns) != len(result.turns):
            symmetry["turn_count_differs"] += 1
            continue
        for record, twin in zip(result.turns, mirrored.turns):
            symmetry["turns_compared"] += 1
            diffs = diff_paths(record.snapshot.to_dict(), flip(twin.snapshot.to_dict()))
            if diffs:
                symmetry["snapshot_mismatch"] += 1
                if len(examples) < 6:
                    examples.append(f"{battle_id} t{record.turn}: {diffs[0]}")
            for key, action in record.actions.items():
                symmetry["actions_compared"] += 1
                other = twin.actions.get(("p2" if key[:2] == "p1" else "p1") + key[2:])
                expected = None if other is None else flip(vars(other))
                if expected != vars(action):
                    symmetry["action_mismatch"] += 1
        # The bot seated on either side, receiving its own HP exact.
        sheets = {
            event[2]: parse_showteam("|".join(event[3:]))
            for event in events
            if len(event) > 3 and event[1] == "showteam"
        }
        for role in SIDES:
            view = synthetic_player_view(events, role, rng, seat)
            live, shadow = live_snapshots(
                view, battle_id, role, sheets=sheets, sheets_known=True
            )
            for key, value in shadow.counters.items():
                if key.startswith(("sync_error", "sync_skip")):
                    seat[f"seat_{key}"] += value
            for record in result.turns:
                seat["seat_turns"] += 1
                snapshot = live.get(record.turn)
                if snapshot is None:
                    seat["seat_none"] += 1
                    continue
                diffs = diff_paths(snapshot.to_dict(), record.snapshot.to_dict())
                if diffs:
                    seat["seat_mismatch"] += 1
                    if len(seat_examples) < 6:
                        seat_examples.append(
                            f"{battle_id} t{record.turn} {role}: {diffs[0]}"
                        )
    drive_ms.sort()
    return {
        "compared": stats["usable"],
        "stats": dict(stats),
        "seat": dict(seat),
        "seat_examples": seat_examples,
        "counters": dict(counters.most_common()),
        "label_reader_counters": dict(label_counters.most_common()),
        "census": census_table(census),
        "symmetry": dict(symmetry),
        "symmetry_examples": examples,
        "drive_log_ms": {
            "mean": round(sum(drive_ms) / max(1, len(drive_ms)), 3),
            "median": round(percentile(drive_ms, 0.5), 3),
            "p95": round(percentile(drive_ms, 0.95), 3),
            "max": round(max(drive_ms, default=0.0), 3),
        },
        "labels_only_ms_mean": round(sum(label_ms) / max(1, len(label_ms)), 3),
    }


# --- part c -------------------------------------------------------------------


def http_json(url: str) -> Any:
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
    )
    with urllib.request.urlopen(request, timeout=25) as response:
        body = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
    return json.loads(body.decode("utf-8"))


def public_replay_ids(account: str, fetch: bool, notes: Counter[str]) -> list[str]:
    """Ids of the account's public replays: cached search pages, or the network."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUT_DIR / "search_pages.json"
    if cache.exists():
        rows = json.loads(cache.read_text(encoding="utf-8"))
        notes["search_pages_from_cache"] += 1
    elif not fetch:
        notes["no_search_cache_and_no_fetch_flag"] += 1
        return []
    else:
        rows = []
        before: int | None = None
        for _ in range(MAX_SEARCH_REQUESTS):
            query = {"user": account, "format": FORMAT_ID}
            if before is not None:
                query["before"] = str(before)
            page = http_json(
                f"{REPLAY_HOST}/search.json?{urllib.parse.urlencode(query)}"
            )
            notes["search_requests"] += 1
            time.sleep(1.0)
            if not isinstance(page, list) or not page:
                break
            rows.extend(page)
            if len(page) < 51:
                break
            before = min(int(row["uploadtime"]) for row in page) + 1
        cache.write_text(json.dumps(rows), encoding="utf-8")
    return sorted({str(row["id"]) for row in rows if isinstance(row, dict)})


def part_c(fetch: bool) -> dict[str, Any]:
    saved = {battle_id: (path, account) for path, account, battle_id in own_replays()}
    notes: Counter[str] = Counter()
    if not saved:
        return {"compared": 0, "notes": {"no_saved_replays": 1}}
    account = Counter(name for _, name in saved.values()).most_common(1)[0][0]
    try:
        public_ids = public_replay_ids(account, fetch, notes)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"compared": 0, "notes": {f"search_failed:{type(exc).__name__}": 1}}
    matched = [battle_id for battle_id in public_ids if battle_id in saved]
    notes["public_replays_of_account"] = len(public_ids)
    notes["public_also_saved"] = len(matched)
    fields: Counter[str] = Counter()
    examples: list[str] = []
    stats: Counter[str] = Counter()
    fetched = 0
    for battle_id in matched:
        target = OUT_DIR / "public" / f"{battle_id}.json"
        if not target.exists():
            if not fetch or fetched >= MAX_REPLAY_FETCHES:
                notes["not_fetched"] += 1
                continue
            try:
                payload = http_json(f"{REPLAY_HOST}/{battle_id}.json")
            except (urllib.error.URLError, OSError, ValueError) as exc:
                notes[f"fetch_failed:{type(exc).__name__}"] += 1
                continue
            fetched += 1
            notes["replay_fetches"] += 1
            time.sleep(1.0)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload), encoding="utf-8")
        payload = json.loads(target.read_text(encoding="utf-8"))
        path, name = saved[battle_id]
        log = extract_log_from_html(path.read_text(encoding="utf-8", errors="replace"))
        if log is None or "log" not in payload:
            stats["unreadable"] += 1
            continue
        tag = extract_battle_tag(log) or f"battle-{battle_id}"
        events = split_log(log)
        role = player_role(events, name)
        public_events = split_log(str(payload["log"]))
        sheets = {
            event[2]: parse_showteam("|".join(event[3:]))
            for event in public_events
            if len(event) > 3 and event[1] == "showteam"
        }
        if role is None:
            stats["account_not_a_player"] += 1
            continue
        stats["battles"] += 1
        stats[f"bot_was_{role}"] += 1
        stats["public_log_has_showteam"] += bool(sheets)
        # The rewrite rule itself, token by token: every condition string of the
        # bot's side, rewritten from the saved page, against the public log.
        ours = condition_tokens([rewrite_event(e, role) for e in events], role)
        theirs = condition_tokens(public_events, role)
        stats["bot_side_hp_strings"] += len(theirs)
        if len(ours) != len(theirs):
            stats["bot_hp_token_count_differs"] += 1
        for mine, shown in zip(ours, theirs):
            if mine != shown:
                stats["bot_hp_token_mismatch"] += 1
                if len(examples) < 12:
                    examples.append(f"{battle_id} hp: {mine} vs public {shown}")
        # as_is: the saved page, with nothing said about sheets, against the
        #   public log as stored (the public log knows the sheet state, the
        #   page does not: differences are expected and are not failures).
        # sheet_given: the live shadow is told what the public log shows, the
        #   sheets or that there were none.
        # sheet_withheld: the showteam lines are dropped from the public log
        #   and it is driven as a stream that cannot show a sheet.
        closed = [event for event in public_events if event[1] != "showteam"]
        comparisons = {
            "as_is": (drive_events(public_events, tag), None, False),
            "sheet_given": (drive_events(public_events, tag), sheets, True),
            "sheet_withheld": (
                drive_events(closed, tag, sheets_known=False),
                None,
                False,
            ),
        }
        for mode, (spectator, given, known) in comparisons.items():
            live, shadow = live_snapshots(
                events, tag, role, sheets=given, sheets_known=known
            )
            for key, value in shadow.counters.items():
                stats[f"live[{mode}]:{key}"] += value
            if len(live) != len(spectator.turns):
                stats[f"turn_count_differs[{mode}]"] += 1
            for record in spectator.turns:
                stats[f"turns[{mode}]"] += 1
                snapshot = live.get(record.turn)
                if snapshot is None:
                    stats[f"live_none[{mode}]"] += 1
                    continue
                diffs = diff_paths(
                    snapshot.to_dict(), record.snapshot.to_dict(), limit=40
                )
                if diffs:
                    stats[f"turn_mismatch[{mode}]"] += 1
                    for item in diffs:
                        fields[f"[{mode}] {field_name(item)}"] += 1
                    if len(examples) < 12:
                        examples.append(
                            f"{battle_id} t{record.turn} {mode}: {diffs[0]}"
                        )
    return {
        "compared": stats["battles"],
        "account": account,
        "notes": dict(notes),
        "stats": dict(stats),
        "mismatch_fields": dict(fields.most_common()),
        "examples": examples,
    }


# --- part d -------------------------------------------------------------------


def part_d(sample: int, seed: int) -> dict[str, Any]:
    import logging

    from poke_env.battle import DoubleBattle
    from poke_env.data import to_id_str
    from poke_env.teambuilder import Teambuilder

    from vgc_bench.src import pokeenv_patches

    pokeenv_patches.install()
    player_level = {
        "t:",
        "expire",
        "uhtmlchange",
        "tempnotify",
        "tempnotifyoff",
        "uhtml",
        "request",
        "error",
        "bigerror",
    }
    logger = logging.getLogger("oppmodel_parity")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False

    def showteam(battle: Any, event: Event) -> None:
        team = Teambuilder.parse_packed_team("|".join(event[3:]))
        preview = (
            battle.teampreview_team
            if event[2] == battle.player_role
            else battle.teampreview_opponent_team
        )
        for shown in preview:
            match = [m for m in team if m.nickname and shown.identifies_as(m.nickname)]
            mon = battle.get_pokemon(
                f"{event[2]}: {match[0].nickname}", details=shown._last_details
            )
            mon._update_from_teambuilder(match[0])

    logs = list(corpus(False))
    if sample and sample < len(logs):
        logs = random.Random(seed).sample(logs, sample)
    stats: Counter[str] = Counter()
    fields: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}

    def note(name: str, where: str, mine: Any, theirs: Any) -> None:
        fields[name] += 1
        kept = examples.setdefault(name, [])
        if len(kept) < 6 and not any(where[:40] in item for item in kept):
            kept.append(f"{where}: ours {mine!r} vs poke-env {theirs!r}")

    for battle_id, log in logs:
        ours = drive_log(log, battle_id)
        if not ours.usable:
            stats["skipped_unusable"] += 1
            continue
        stats["logs"] += 1
        by_turn = {record.turn: record.snapshot for record in ours.turns}
        name = ours.players.get("p1") or ""
        battle = DoubleBattle(f"battle-{battle_id}", name, logger, gen=9)
        battle.logger = None
        for event in split_log(log):
            kind = event[1]
            if kind in player_level:
                continue
            try:
                if kind == "showteam":
                    showteam(battle, event)
                elif kind == "win":
                    battle.won_by(event[2])
                elif kind == "tie":
                    battle.tied()
                else:
                    battle.parse_message(event)
            except Exception as exc:
                stats[f"pokeenv_error:{kind}:{type(exc).__name__}"] += 1
            if kind != "turn":
                continue
            snapshot = by_turn.get(int(event[2]))
            if snapshot is None:
                stats["turn_missing_on_our_side"] += 1
                continue
            stats["turns"] += 1
            where = f"{battle_id} t{snapshot.turn}"
            views = {
                "p1": (
                    battle._active_pokemon,
                    battle._team,
                    battle.side_conditions,
                    battle.used_mega_evolve,
                ),
                "p2": (
                    battle._opponent_active_pokemon,
                    battle._opponent_team,
                    battle.opponent_side_conditions,
                    battle.opponent_used_mega_evolve,
                ),
            }
            weather = {to_id_str(w.name) for w in battle.weather}
            if (snapshot.weather is not None) != bool(weather):
                note("weather_present", where, snapshot.weather, sorted(weather))
            theirs_fields = sorted(to_id_str(f.name) for f in battle.fields)
            if sorted(snapshot.fields) != theirs_fields:
                note("fields", where, sorted(snapshot.fields), theirs_fields)
            for side, (active, team, conditions, mega) in views.items():
                view = snapshot.sides[side]
                if view.mega_used != mega:
                    note("mega_used", where, view.mega_used, mega)
                theirs_conditions = sorted(to_id_str(c.name) for c in conditions)
                if sorted(view.conditions) != theirs_conditions:
                    note(
                        "side_conditions",
                        where,
                        sorted(view.conditions),
                        theirs_conditions,
                    )
                fainted = sum(m.fainted for m in team.values())
                if view.n_fainted != fainted:
                    note("n_fainted", where, view.n_fainted, fainted)
                revealed = sum(m.revealed for m in team.values())
                if view.n_revealed != revealed:
                    note("n_revealed", where, view.n_revealed, revealed)
                bench = {to_id_str(m.base_species): m for m in team.values()}
                for mon in view.mons:
                    other = bench.get(mon.species)
                    if other is None or not mon.revealed or mon.slot is not None:
                        continue
                    stats["bench_mons"] += 1
                    hp = 0.0 if other.fainted else other.current_hp_fraction
                    if abs(mon.hp - hp) > 1e-9:
                        note("bench_hp", f"{where} {mon.species}", mon.hp, hp)
                for letter in ("a", "b"):
                    mine = snapshot.mon(side, letter)
                    theirs = active.get(side + letter)
                    if theirs is not None and theirs.fainted:
                        theirs = None
                    if (mine is None) != (theirs is None):
                        note("slot_occupied", f"{where} {side}{letter}", mine, theirs)
                        continue
                    if mine is None or theirs is None:
                        continue
                    stats["active_mons"] += 1
                    spot = f"{where} {side}{letter} {mine.species}"
                    if mine.species != to_id_str(theirs.base_species):
                        note("species", spot, mine.species, theirs.base_species)
                    if abs(mine.hp - theirs.current_hp_fraction) > 1e-9:
                        note("hp", spot, mine.hp, theirs.current_hp_fraction)
                    status = theirs.status.name.lower() if theirs.status else None
                    if mine.status != status:
                        note("status", spot, mine.status, status)
                    if mine.boosts != dict(theirs.boosts):
                        note("boosts", spot, mine.boosts, dict(theirs.boosts))
                    if mine.first_turn != theirs.first_turn:
                        note("first_turn", spot, mine.first_turn, theirs.first_turn)
                    types = tuple(to_id_str(t.name) for t in theirs.types if t)
                    if tuple(sorted(mine.types)) != tuple(sorted(types)):
                        note("types", spot, mine.types, types)
                    if mine.protect_streak != theirs.protect_counter:
                        note(
                            "protect_streak_vs_counter",
                            spot,
                            mine.protect_streak,
                            theirs.protect_counter,
                        )
    return {
        "compared": stats["logs"],
        "stats": dict(stats),
        "disagreements": dict(fields.most_common()),
        "examples_per_field": examples,
    }


def failures(report: dict[str, Any]) -> list[str]:
    """What breaks the acceptance: any mismatch, leak, raise or tracker error.

    Part c's ``as_is`` differences are not failures: the public log knows the
    sheet state, the saved player-view page does not. Part d is a cross-check
    against another implementation: its disagreements are reported only. A part
    that was asked for and compared nothing fails, d included, so that exit 0
    cannot mean "nothing was checked".
    """
    bad_stats = (
        "turn_mismatch",
        "turn_count_differs",
        "live_none",
        "exact_hp_left",
        "RAISED",
        "bot_hp_token",
    )
    bad_seat = ("seat_mismatch", "seat_none", "seat_sync", "rewrite_roundtrip")
    bad_counters = (
        "feed_error",
        "drive_error",
        "last_action_error",
        "reader_error",
        "sync_error",
        "sync_skip",
    )
    out: list[str] = []
    for part, body in report.items():
        if not isinstance(body, dict):
            continue
        if not body.get("compared"):
            out.append(f"{part}: compared nothing")
        if part.startswith("d_"):
            continue
        for key, value in body.get("stats", {}).items():
            if key.startswith(bad_stats) and "[as_is]" not in key and value:
                out.append(f"{part}: {key} = {value}")
        for key, value in body.get("seat", {}).items():
            if key.startswith(bad_seat) and value:
                out.append(f"{part}: {key} = {value}")
        for key, value in body.get("counters", {}).items():
            if any(name in key for name in bad_counters) and value:
                out.append(f"{part}: {key} = {value}")
        for key, value in body.get("symmetry", {}).items():
            if key.endswith(("mismatch", "differs")) and value:
                out.append(f"{part}: symmetry {key} = {value}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--parts", default="a,b", help="comma list of a,b,c,d")
    parser.add_argument("--sample", type=int, default=1500, help="0 = every log")
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--union", action="store_true", help="part b: add JSON corpora")
    parser.add_argument("--fetch", action="store_true", help="part c: use the network")
    parser.add_argument("--json", type=Path, help="write the full report here")
    args = parser.parse_args()
    parts = {part.strip() for part in args.parts.split(",") if part.strip()}
    report: dict[str, Any] = {"argv": _sys.argv[1:]}
    if "a" in parts:
        report["a_own_replays_live_vs_offline"] = part_a()
    if "b" in parts:
        report["b_human_corpus"] = part_b(args.sample, args.seed, args.union)
    if "c" in parts:
        report["c_spectator_vs_player_view"] = part_c(args.fetch)
    if "d" in parts:
        report["d_pokeenv_crosscheck"] = part_d(args.sample, args.seed)
    report["failures"] = failures(report)
    text = json.dumps(report, indent=1, default=str)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
