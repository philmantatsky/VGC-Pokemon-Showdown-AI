"""Acceptance test of the opponent predictor's runtime: "runtime == offline".

``vgc_bench.src.oppmodel.runtime.OpponentPredictor`` is what the bot will call
each turn. This script replays every saved own page (``ladder_replays_mc*/``,
``challenge_replays_mc*/``) into it the way a live battle would arrive and
checks that it says what the offline path says, then measures what it costs.

For each artifact (loaded twice, so the two arms share no object):

  offline   the dataset builder's own-page path: ``Corpus`` reads the page
            (the bot's exact HP rewritten to the public percent, sheet state
            from the decision audit or the cached public replay), ``drive``
            runs the whole log, the builder's ``drop_reason`` decides whether
            the battle is used, ``Featurizer.encode`` makes the opponent's
            example of every turn, ``sheet_unknown_as_closed``, then
            ``predictor.predict`` on the page's examples in one batch (and
            again one example per call).
  runtime   the page's untouched player-view stream is appended event by event
            to a fake battle object (three attributes: ``_replay_data``,
            ``player_role`` from the page, ``battle_tag``), the same sheet
            knowledge is handed over through ``note_sheets`` before the first
            event, and ``predict`` is called at every ``|turn|`` line.

Compared on every turn of every page: the encoded feature arrays (bit for
bit), the predictor's raw ``action`` / ``target`` / ``mega`` output and the
forecast's normalised per-slot arrays (largest absolute difference; the bar is
1e-6). A turn where the offline path has an example and the runtime returned
None is a failure and is listed with the runtime's reason counter; so is the
reverse. A second pass calls ``predict`` after EVERY event: turn starts must
compare the same, and no call after a battle event may return a forecast. A
third pass hands the runtime a REAL poke-env ``DoubleBattle`` that was fed the
page through ``parse_message`` / ``won_by`` / ``tied`` (the object and the
stream the bot will hold), tells it the sheets at the first turn line as the
hook will, and compares the same way; the stream poke-env kept must be the
page's, event for event.

What this does not test. Both arms share the event rewrite, the tracker, the
featurizer and the predictor, so a defect in any of them shows the same on
both sides: this is a train / serve parity check of the serving wrapper, not
an independent check of the state code (the foundation's own audits are
that). Open team sheets reach the runtime from the builder's records here,
never through ``runtime.sheet_sets_from_team`` on poke-env team objects.

Also reported, per artifact: milliseconds per ``predict`` call (p50 / p90 /
p99 / max; one CPU thread) over all turn starts, early and late in long games,
and for a first call late in a long game (a fresh shadow reads the whole
stream: the cost after a restart). The p99 over all turn starts must stay
within ``--p99-budget-ms`` (the largest single call is not gated: on a busy
machine it is a garbage-collector or scheduler pause, not the runtime's own
work). Python memory still held after ``--memory-battles``
battles with ``forget()`` after each one (and after the same battles once
more, which must add nothing), without it under the default bound, and without
any bound; and, when the built dataset is on disk, whether the offline
examples made here equal the rows the dataset stores for the same battles and
turns.

Exit status 0 means every comparison ran and passed. Outputs, rendered from
the JSON: ``<out>/report.json`` and ``<out>/report.md``. A directory that
already holds a report is not written over without ``--overwrite``. From the
repo root:

  OMP_NUM_THREADS=1 nice -n 19 .venv/bin/python evaluation/oppmodel_shadow_replay.py
  OMP_NUM_THREADS=1 nice -n 19 .venv/bin/python evaluation/oppmodel_shadow_replay.py \
      --artifact results_oppmodel/tables_v1/flags_table.pt --limit 20 --out <dir>
  .venv/bin/python evaluation/oppmodel_shadow_replay.py --render-only
"""

from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path as _Path

_os.environ.setdefault("OMP_NUM_THREADS", "1")
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import gc
import json
import logging
import resource
import time
import tracemalloc
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from datagen.oppmodel_build_dataset import (
    DEFAULT_FORMATS,
    Corpus,
    RawBattle,
    drive,
    drop_reason,
)
from vgc_bench.src.oppmodel.artifact import load_predictor
from vgc_bench.src.oppmodel.events import (
    CHATTER_KINDS,
    SIDES,
    Event,
    extract_battle_tag,
    extract_log_from_html,
    split_log,
)
from vgc_bench.src.oppmodel.features import (
    Example,
    Featurizer,
    collate,
    load_dataset,
    normalize_prediction,
    sheet_unknown_as_closed,
    take,
)
from vgc_bench.src.oppmodel.runtime import Forecast, OpponentPredictor

ROOT = Path(__file__).resolve().parents[1]
# The models meant to be served: the count table behind the interface on day
# one, and the trained network (never a smoke artifact).
DEFAULT_ARTIFACTS = (
    "results_oppmodel/tables_v1/flags_table.pt",
    "results_oppmodel/oppnet_v1/artifact.pt",
)
DEFAULT_OUT = "results_oppmodel/shadow_replay"
DEFAULT_DATASET = "results_oppmodel/v1_ondisk"
REPORT_JSON = "report.json"
REPORT_MD = "report.md"
TOLERANCE = 1e-6
# Milliseconds the 99th percentile of a predict call at a turn start may take
# (one thread). Typical is 1-3 ms; the turn's hard cap is 10 s.
DEFAULT_P99_BUDGET_MS = 25.0
PASS_REAL = "real_battle_objects"
_GEN = 9
_QUIET = logging.getLogger("oppmodel_shadow_replay.poke_env")
_QUIET.addHandler(logging.NullHandler())
_QUIET.propagate = False
# Bytes by which "held after the same battles again" may exceed the first
# reading before it counts as growth (allocator and cache noise).
MEMORY_SLACK = 64 * 1024
OWN_SPLITS = ("ladder_holdout", "own_unrated")
HEADS = ("action", "target", "mega")
# (label, last turn of the bucket) for the latency-by-turn table.
TURN_BUCKETS: tuple[tuple[str, int], ...] = (
    ("1", 1),
    ("2-5", 5),
    ("6-9", 9),
    ("10-14", 14),
    ("15+", 10**9),
)
# Event types after which the turn start is over (a battle event, by its type).
_ACTION_KINDS = frozenset(
    {
        "move",
        "switch",
        "drag",
        "replace",
        "detailschange",
        "cant",
        "faint",
        "swap",
        "upkeep",
        "win",
        "tie",
    }
)


def say(text: str) -> None:
    print(text, flush=True)


# --- pages ---------------------------------------------------------------------


class FakeBattle:
    """The three attributes the runtime reads from a live battle."""

    def __init__(self, battle_tag: str, player_role: str) -> None:
        self.battle_tag = battle_tag
        self.player_role = player_role
        self._replay_data: list[Event] = []


@dataclass
class Page:
    """One saved own page: the builder's view of it and its raw stream."""

    battle_id: str
    tag: str
    role: str
    raw: RawBattle
    events: list[Event]
    n_turns: int


def load_pages(
    root: Path = ROOT, limit: int | None = None
) -> tuple[list[Page], set[str], Counter[str]]:
    """Every saved own page, the bot's account ids, and what was left out."""
    corpus = Corpus(root, DEFAULT_FORMATS, ("own",))
    bots = corpus.bot_accounts()
    notes: Counter[str] = Counter()
    pages: list[Page] = []
    for raw in corpus:
        if limit is not None and len(pages) >= limit:
            break
        try:
            html = (root / raw.path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            notes["page_unreadable"] += 1
            continue
        log = extract_log_from_html(html)
        if log is None or raw.bot_side not in SIDES:
            notes["page_without_stream"] += 1
            continue
        events = split_log(log)
        tag = extract_battle_tag(log) or f"battle-{raw.battle_id}"
        turns = sum(1 for event in events if len(event) > 2 and event[1] == "turn")
        pages.append(Page(raw.battle_id, tag, str(raw.bot_side), raw, events, turns))
    notes.update(corpus.notes)
    return pages, bots, notes


def opponent_of(role: str) -> str:
    return SIDES[1 - SIDES.index(role)]


def note_page_sheets(runtime: OpponentPredictor, fake: Any, page: Page) -> None:
    """Hand the runtime the sheet knowledge the builder has for this page."""
    state = page.raw.sheet_audit
    sheets = page.raw.sheets or {}
    if state == "closed":
        runtime.note_sheets(fake, False)
    elif state == "open":
        runtime.note_sheets(
            fake,
            True,
            opponent_sets=sheets.get(opponent_of(page.role)),
            own_sets=sheets.get(page.role),
        )


# --- the offline arm ------------------------------------------------------------


def offline_examples(
    page: Page, featurizer: Featurizer, bots: set[str]
) -> tuple[dict[int, Example], str | None, int]:
    """(turn -> the opponent's example, the builder's drop reason, repeated turns)."""
    result = drive(page.raw)
    reason = drop_reason(page.raw, result, DEFAULT_FORMATS, bots)
    if reason is not None:
        return {}, reason, 0
    other = opponent_of(page.role)
    examples: dict[int, Example] = {}
    repeated = 0
    for record in result.turns:
        example = featurizer.encode(record.snapshot, other)
        if example is None:
            continue
        if record.turn in examples:
            repeated += 1
        examples[record.turn] = example
    return examples, None, repeated


# --- the runtime arm ------------------------------------------------------------


@dataclass
class TurnCall:
    """One ``predict`` at a turn line. ``served`` / ``inner_ms`` outlive the
    forecast, which is dropped once it has been compared."""

    turn: int
    forecast: Forecast | None
    reason: str
    ms: float
    served: bool = False
    inner_ms: float = 0.0

    def light(self) -> "TurnCall":
        return TurnCall(
            self.turn, None, self.reason, self.ms, self.served, self.inner_ms
        )


def replay(
    runtime: OpponentPredictor, page: Page, every_event: bool = False
) -> tuple[list[TurnCall], int, int]:
    """Feed a page event by event. Returns the turn-start calls, the number of
    calls made after a battle event and how many of those gave a forecast."""
    fake = FakeBattle(page.tag, page.role)
    note_page_sheets(runtime, fake, page)
    calls: list[TurnCall] = []
    after_action = leaked = 0
    for event in page.events:
        fake._replay_data.append(list(event))
        kind = event[1] if len(event) > 1 else ""
        if len(event) > 2 and kind == "turn":
            started = time.perf_counter()
            forecast, reason = runtime.predict_with_reason(fake)
            ms = (time.perf_counter() - started) * 1000.0
            try:
                number = int(event[2])
            except ValueError:
                number = -1
            inner = 0.0 if forecast is None else forecast.latency_ms
            calls.append(
                TurnCall(number, forecast, reason, ms, forecast is not None, inner)
            )
        elif every_event:
            forecast = runtime.predict(fake)
            minor = kind.startswith("-") and kind not in CHATTER_KINDS
            if kind in _ACTION_KINDS or minor:
                after_action += 1
                leaked += int(forecast is not None)
    runtime.forget(page.tag)
    return calls, after_action, leaked


def bot_username(page: Page) -> str | None:
    """The bot's display name: the ``|player|`` line of its seat."""
    for event in page.events:
        if len(event) > 3 and event[1] == "player" and event[2] == page.role:
            return event[3] or None
    return None


def replay_real(
    runtime: OpponentPredictor, page: Page, notes: Counter[str]
) -> list[TurnCall] | None:
    """Feed a page through a real poke-env ``DoubleBattle`` and hand THAT object
    to the runtime at every turn line.

    The battle gets every event the way the live player gives it
    (``parse_message``; ``won_by`` / ``tied`` for the two result lines), so
    its ``_replay_data`` is poke-env's own. Sheets are told at the first turn
    line, once poke-env knows the seat, as the bot's hook will. ``notes``
    counts what went differently: ``stream_differs`` (poke-env kept another
    stream than the page's), ``role_differs``, and poke-env's own parse
    errors (which do not stop the stream from growing). None when the page
    names no player or poke-env is not importable.
    """
    try:
        from poke_env.battle import DoubleBattle
    except Exception as exc:
        notes[f"poke_env_missing:{type(exc).__name__}"] += 1
        return None
    name = bot_username(page)
    if name is None:
        notes["no_player_line"] += 1
        return None
    battle: Any = DoubleBattle(page.tag, name, _QUIET, gen=_GEN)
    calls: list[TurnCall] = []
    told = False
    for position, event in enumerate(page.events, start=1):
        kind = event[1] if len(event) > 1 else ""
        try:
            if kind == "win" and len(event) > 2:
                battle.won_by(event[2])
            elif kind == "tie":
                battle.tied()
            else:
                battle.parse_message(list(event))
        except Exception as exc:
            notes[f"poke_env_parse:{kind}:{type(exc).__name__}"] += 1
        stream = battle._replay_data
        if len(stream) != position or list(stream[-1]) != list(event):
            notes["stream_differs"] += 1
            return None
        if not (len(event) > 2 and kind == "turn"):
            continue
        if not told:
            told = True
            if battle.player_role != page.role:
                notes["role_differs"] += 1
                return None
            note_page_sheets(runtime, battle, page)
        started = time.perf_counter()
        forecast, reason = runtime.predict_with_reason(battle)
        ms = (time.perf_counter() - started) * 1000.0
        try:
            number = int(event[2])
        except ValueError:
            number = -1
        inner = 0.0 if forecast is None else forecast.latency_ms
        calls.append(
            TurnCall(number, forecast, reason, ms, forecast is not None, inner)
        )
    runtime.forget(page.tag)
    notes["pages"] += 1
    return calls


def cold_call(runtime: OpponentPredictor, page: Page) -> float | None:
    """Milliseconds of a first ``predict`` at the page's last turn start."""
    last = max(
        (i for i, e in enumerate(page.events) if len(e) > 2 and e[1] == "turn"),
        default=None,
    )
    if last is None:
        return None
    fake = FakeBattle(page.tag, page.role)
    note_page_sheets(runtime, fake, page)
    fake._replay_data.extend(list(event) for event in page.events[: last + 1])
    started = time.perf_counter()
    forecast = runtime.predict(fake)
    ms = (time.perf_counter() - started) * 1000.0
    runtime.forget(page.tag)
    return ms if forecast is not None else None


# --- comparison -----------------------------------------------------------------


class Equality:
    """Running tallies of one pass over the pages."""

    def __init__(self) -> None:
        self.pages = 0
        self.turn_lines = 0
        self.offline_examples = 0
        self.compared = 0
        self.repeated_turns = 0
        self.dropped_offline: Counter[str] = Counter()
        self.max_diff = dict.fromkeys(HEADS, 0.0)
        self.max_diff_single = dict.fromkeys(HEADS, 0.0)
        self.max_diff_normalised = {"action": 0.0, "target": 0.0}
        self.feature_mismatches: Counter[str] = Counter()
        self.slot_mismatches = 0
        self.runtime_none: Counter[str] = Counter()
        self.runtime_only: Counter[str] = Counter()
        self.both_none: Counter[str] = Counter()
        self.examples: list[dict[str, Any]] = []
        self.after_action_calls = 0
        self.after_action_forecasts = 0

    def note(self, kind: str, page: Page, turn: int, detail: str) -> None:
        if len(self.examples) < 20:
            self.examples.append(
                {"kind": kind, "battle": page.battle_id, "turn": turn, "detail": detail}
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pages": self.pages,
            "turn_lines": self.turn_lines,
            "offline_examples": self.offline_examples,
            "compared": self.compared,
            "repeated_turn_numbers": self.repeated_turns,
            "pages_dropped_offline": dict(self.dropped_offline),
            "tolerance": TOLERANCE,
            "max_abs_diff": dict(self.max_diff),
            "max_abs_diff_one_example_per_call": dict(self.max_diff_single),
            "max_abs_diff_normalised_slots": dict(self.max_diff_normalised),
            "feature_arrays_differing": dict(self.feature_mismatches),
            "slot_presence_mismatches": self.slot_mismatches,
            "runtime_none_where_offline_has_an_example": {
                "count": sum(self.runtime_none.values()),
                "reasons": dict(self.runtime_none),
            },
            "runtime_forecast_without_offline_example": {
                "count": sum(self.runtime_only.values()),
                "why_offline_has_none": dict(self.runtime_only),
            },
            "both_none": {
                "count": sum(self.both_none.values()),
                "runtime_reasons": dict(self.both_none),
            },
            "calls_after_a_battle_event": self.after_action_calls,
            "forecasts_after_a_battle_event": self.after_action_forecasts,
            "first_differences": self.examples,
        }


def _diff(left: Any, right: Any) -> float:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    gap = np.abs(a - b)
    return float("inf") if not np.isfinite(gap).all() else float(gap.max())


def compare_page(
    tally: Equality,
    page: Page,
    calls: Sequence[TurnCall],
    examples: Mapping[int, Example],
    dropped: str | None,
    predictor: Any,
) -> None:
    """Compare one page's turn-start calls with the offline arm."""
    tally.pages += 1
    tally.turn_lines += len(calls)
    tally.offline_examples += len(examples)
    if dropped is not None:
        tally.dropped_offline[dropped] += 1
    order = sorted(examples)
    row_of = {turn: row for row, turn in enumerate(order)}
    batch: dict[str, np.ndarray] = {}
    made: dict[str, np.ndarray] = {}
    norm: dict[str, np.ndarray] = {}
    if order:
        batch = sheet_unknown_as_closed(collate([examples[turn] for turn in order]))
        made = predictor.predict(batch)
        norm = normalize_prediction(made, batch)
    seen: set[int] = set()
    for call in calls:
        row = row_of.get(call.turn)
        if call.turn in seen:
            tally.repeated_turns += 1
        seen.add(call.turn)
        forecast = call.forecast
        if row is None:
            if forecast is None:
                tally.both_none[call.reason or "?"] += 1
            else:
                why = dropped or "no_example"
                tally.runtime_only[why] += 1
                tally.note("runtime_only", page, call.turn, why)
            continue
        if forecast is None:
            tally.runtime_none[call.reason or "?"] += 1
            tally.note("runtime_none", page, call.turn, call.reason)
            continue
        tally.compared += 1
        features = forecast.features or {}
        for name in sorted(set(batch) | set(features)):
            same = (
                name in batch
                and name in features
                and np.array_equal(features[name][0], batch[name][row])
            )
            if not same:
                tally.feature_mismatches[name] += 1
                tally.note("feature", page, call.turn, name)
        single = predictor.predict(take(batch, [row]))
        for head in HEADS:
            gap = _diff(forecast.raw.get(head), np.asarray(made[head])[row])
            tally.max_diff[head] = max(tally.max_diff[head], gap)
            if gap > TOLERANCE:
                tally.note("prediction", page, call.turn, f"{head} {gap:.3e}")
            gap = _diff(forecast.raw.get(head), np.asarray(single[head])[0])
            tally.max_diff_single[head] = max(tally.max_diff_single[head], gap)
        active = np.asarray(batch["act_mon"])[row] >= 0
        for index, slot in enumerate(forecast.slots):
            if (slot is not None) != bool(active[index]):
                tally.slot_mismatches += 1
                tally.note("slot", page, call.turn, f"slot {index}")
                continue
            if slot is None:
                continue
            for head, values in (
                ("action", slot.action_probs),
                ("target", slot.target_probs),
            ):
                gap = _diff(values, norm[head][row, index])
                tally.max_diff_normalised[head] = max(
                    tally.max_diff_normalised[head], gap
                )
    for turn in order:
        if turn not in seen:
            tally.runtime_none["no_turn_line"] += 1
            tally.note("runtime_none", page, turn, "no_turn_line")


# --- latency and memory ---------------------------------------------------------


def summary(values: Sequence[float]) -> dict[str, Any]:
    """n, p50, p90, p99, max and mean of a list of milliseconds."""
    if not values:
        return dict.fromkeys(("n", "p50", "p90", "p99", "max", "mean"), None) | {"n": 0}
    data = np.asarray(values, dtype=np.float64)
    return {
        "n": int(data.size),
        "p50": round(float(np.percentile(data, 50)), 4),
        "p90": round(float(np.percentile(data, 90)), 4),
        "p99": round(float(np.percentile(data, 99)), 4),
        "max": round(float(data.max()), 4),
        "mean": round(float(data.mean()), 4),
    }


def latency_report(
    timed: Sequence[tuple[Page, Sequence[TurnCall]]],
    cold: Sequence[float],
    long_turns: int,
    early_until: int,
    late_from: int,
) -> dict[str, Any]:
    everything: list[float] = []
    early: list[float] = []
    late: list[float] = []
    inner: list[float] = []
    by_turn: dict[str, list[float]] = {}
    long_games = 0
    for page, calls in timed:
        is_long = page.n_turns >= long_turns
        long_games += int(is_long)
        for call in calls:
            if not call.served:
                continue
            everything.append(call.ms)
            inner.append(call.inner_ms)
            bucket = next(
                (name for name, last in TURN_BUCKETS if call.turn <= last),
                TURN_BUCKETS[-1][0],
            )
            by_turn.setdefault(bucket, []).append(call.ms)
            if is_long and call.turn <= early_until:
                early.append(call.ms)
            if is_long and call.turn >= late_from:
                late.append(call.ms)
    return {
        "unit": "milliseconds per predict call, wall clock, one CPU thread",
        "all_turn_starts": summary(everything),
        "long_games": {
            "definition": f"at least {long_turns} turns",
            "games": long_games,
            f"early_turns_1_to_{early_until}": summary(early),
            f"late_turns_{late_from}_on": summary(late),
        },
        "by_turn": {
            name: summary(by_turn[name]) for name, _ in TURN_BUCKETS if name in by_turn
        },
        "first_call_at_the_last_turn_of_a_long_game": summary(cold),
        "forecast_latency_ms_field": summary(inner),
    }


def memory_run(
    path: Path, pages: Sequence[Page], mode: str, warm: int = 10
) -> dict[str, Any]:
    """Python memory still held after replaying ``pages`` through one runtime.

    ``mode``: ``forget`` (``forget()`` after every battle), ``bounded`` (never
    forget; the default ``max_battles``) or ``unbounded`` (never forget, no
    effective bound). Forecasts are dropped as they come, as a caller would.
    """
    bound = len(pages) + warm + 1 if mode == "unbounded" else None
    runtime = (
        OpponentPredictor.load(path)
        if bound is None
        else OpponentPredictor.load(path, max_battles=bound)
    )

    def play(page: Page, tag: str) -> None:
        fake = FakeBattle(tag, page.role)
        note_page_sheets(runtime, fake, page)
        for event in page.events:
            fake._replay_data.append(event)
            if len(event) > 2 and event[1] == "turn":
                runtime.predict(fake)
        if mode == "forget":
            runtime.forget(tag)

    for page in pages[:warm]:  # fill the caches that are not per battle
        play(page, page.tag + "-warm")
        runtime.forget(page.tag + "-warm")
    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    started = time.perf_counter()
    for page in pages:
        play(page, page.tag)
    seconds = time.perf_counter() - started
    gc.collect()
    held, peak = tracemalloc.get_traced_memory()
    kept = runtime.n_battles
    evicted = int(runtime.counters.get("evicted", 0))
    again: int | None = None
    if mode == "forget":
        # The same battles once more: what was held must not grow with them.
        for page in pages:
            play(page, page.tag + "-again")
        gc.collect()
        again = max(0, tracemalloc.get_traced_memory()[0] - base)
    tracemalloc.stop()
    held = max(0, held - base)
    return {
        "mode": mode,
        "battles": len(pages),
        "max_battles": runtime.max_battles,
        "shadows_kept": kept,
        "python_bytes_held": int(held),
        "python_bytes_held_after_the_same_battles_again": again,
        "python_bytes_peak": int(max(0, peak - base)),
        "bytes_per_kept_shadow": int(held / kept) if kept else None,
        "evicted": evicted,
        "seconds_under_tracemalloc": round(seconds, 2),
    }


# --- the built dataset ----------------------------------------------------------


def dataset_rows(directory: Path) -> tuple[dict[tuple[str, int], int], dict[str, Any]]:
    """((battle id, turn) -> row, the own-game rows) of a built dataset."""
    batch, _ = load_dataset(directory, splits=OWN_SPLITS)
    ids: dict[int, str] = {}
    with (directory / "battles.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            ids[int(row["index"])] = str(row["id"])
    index: dict[tuple[str, int], int] = {}
    for row, (battle, turn) in enumerate(zip(batch["m_battle"], batch["m_turn"])):
        index[(ids.get(int(battle), ""), int(turn))] = row
    return index, batch


def dataset_check(
    directory: Path,
    rows: tuple[dict[tuple[str, int], int], dict[str, Any]] | None,
    offline: Sequence[tuple[Page, Mapping[int, Example]]],
) -> dict[str, Any]:
    """Do the offline examples made here equal the dataset's stored rows?"""
    if rows is None:
        return {"status": "skipped", "why": f"no readable dataset at {directory}"}
    index, stored = rows
    compared = missing = 0
    differing: Counter[str] = Counter()
    for page, examples in offline:
        for turn, example in examples.items():
            row = index.get((page.battle_id, turn))
            if row is None:
                missing += 1
                continue
            compared += 1
            for name, value in example.items():
                if name not in stored or not np.array_equal(value, stored[name][row]):
                    differing[name] += 1
    made = sum(len(examples) for _, examples in offline)
    return {
        "status": "ran",
        "dataset": str(directory),
        "dataset_rows_of_own_games": int(len(index)),
        "offline_examples": made,
        "compared": compared,
        "offline_examples_not_in_dataset": missing,
        "dataset_rows_not_made_here": int(len(index)) - compared,
        "feature_arrays_differing": dict(differing),
    }


# --- one artifact ---------------------------------------------------------------


def predictor_failures(counters: Any) -> dict[str, int]:
    if not isinstance(counters, Mapping):
        return {}
    return {
        str(name): int(count)
        for name, count in counters.items()
        if "error" in str(name)
    }


def run_artifact(
    path: Path,
    pages: Sequence[Page],
    bots: set[str],
    *,
    dataset: Path | None,
    stored: tuple[dict[tuple[str, int], int], dict[str, Any]] | None,
    memory_battles: int,
    long_turns: int,
    early_until: int,
    late_from: int,
    real_battles: bool = True,
    log: Any = say,
) -> dict[str, Any]:
    """Every check for one artifact."""
    offline = load_predictor(path)
    runtime = OpponentPredictor.load(path, keep_features=True)
    report: dict[str, Any] = {
        "path": str(path),
        "name": runtime.name,
        "kind": runtime.kind,
        "elo_mode": runtime.elo_mode,
        "loaded": runtime.loaded,
        "load_failure": runtime.load_failure,
    }
    if not runtime.serving:
        return report
    log(f"  {runtime.name} ({runtime.kind}): offline arm")
    made: list[tuple[Page, dict[int, Example], str | None]] = []
    repeated = 0
    for page in pages:
        examples, dropped, again = offline_examples(page, offline.featurizer, bots)
        repeated += again
        made.append((page, examples, dropped))

    log("  pass 1: one predict per turn line (timed)")
    per_turn = Equality()
    per_turn.repeated_turns = repeated
    timed: list[tuple[Page, list[TurnCall]]] = []
    for page, examples, dropped in made:
        calls, _, _ = replay(runtime, page)
        compare_page(per_turn, page, calls, examples, dropped, offline.predictor)
        # Keep the timings and whether a forecast came, not the forecasts.
        timed.append((page, [call.light() for call in calls]))
    long_pages = [page for page in pages if page.n_turns >= long_turns]
    fresh = OpponentPredictor(
        runtime.predictor, runtime.featurizer, name=runtime.name, meta=runtime.meta
    )
    cold = [ms for ms in (cold_call(fresh, page) for page in long_pages) if ms]
    latency = latency_report(timed, cold, long_turns, early_until, late_from)
    del timed

    log("  pass 2: one predict per event")
    twin = OpponentPredictor(
        runtime.predictor,
        runtime.featurizer,
        name=runtime.name,
        kind=runtime.kind,
        keep_features=True,
        meta=runtime.meta,
    )
    per_event = Equality()
    for page, examples, dropped in made:
        calls, after, leaked = replay(twin, page, every_event=True)
        compare_page(per_event, page, calls, examples, dropped, offline.predictor)
        per_event.after_action_calls += after
        per_event.after_action_forecasts += leaked

    report["equality"] = {
        "per_turn": per_turn.to_dict(),
        "per_event": per_event.to_dict(),
    }
    if real_battles:
        log("  pass 3: real poke-env battle objects")
        third = OpponentPredictor(
            runtime.predictor,
            runtime.featurizer,
            name=runtime.name,
            kind=runtime.kind,
            keep_features=True,
            meta=runtime.meta,
        )
        real = Equality()
        notes: Counter[str] = Counter()
        for page, examples, dropped in made:
            calls = replay_real(third, page, notes)
            if calls is not None:
                compare_page(real, page, calls, examples, dropped, offline.predictor)
        if notes.get("pages"):
            report["equality"][PASS_REAL] = real.to_dict()
        report["real_battle_notes"] = dict(notes)
        report.setdefault("counters", {})["runtime_pass_3"] = dict(third.counters)
    report["latency_ms"] = latency
    if dataset is not None:
        log("  dataset rows")
        report["dataset_check"] = dataset_check(
            dataset, stored, [(page, examples) for page, examples, _ in made]
        )
    if memory_battles > 0:
        log(f"  memory after {min(memory_battles, len(pages))} battles")
        subset = pages[:memory_battles]
        report["memory"] = [
            memory_run(path, subset, mode)
            for mode in ("forget", "bounded", "unbounded")
        ]
    report["counters"] = {
        **report.get("counters", {}),
        "runtime_pass_1": dict(runtime.counters),
        "runtime_pass_2": dict(twin.counters),
        "featurizer": dict(runtime.diagnostics()["featurizer"]),
        "runtime_predictor": dict(runtime.diagnostics()["predictor"]),
        "offline_predictor": dict(getattr(offline.predictor, "counters", None) or {}),
    }
    return report


# --- the whole run --------------------------------------------------------------


def failures(report: Mapping[str, Any]) -> list[str]:
    """Every acceptance item that did not hold; empty means the run passed."""
    out: list[str] = []
    if not report.get("pages", {}).get("found"):
        out.append("no saved own page was found")
    if not report.get("artifacts"):
        out.append("no artifact was checked")
    for entry in report.get("artifacts", []):
        name = entry.get("name") or entry.get("path")
        if not entry.get("loaded") or "equality" not in entry:
            out.append(f"{name}: not served ({entry.get('load_failure')})")
            continue
        for label, tally in entry["equality"].items():
            where = f"{name} [{label}]"
            if not tally["compared"]:
                out.append(f"{where}: compared nothing")
            for group in (
                "max_abs_diff",
                "max_abs_diff_one_example_per_call",
                "max_abs_diff_normalised_slots",
            ):
                for head, gap in tally[group].items():
                    if not gap <= TOLERANCE:
                        out.append(f"{where}: {group} {head} = {gap:.3e}")
            if tally["feature_arrays_differing"]:
                out.append(
                    f"{where}: features differ {tally['feature_arrays_differing']}"
                )
            if tally["slot_presence_mismatches"]:
                out.append(f"{where}: slot presence differs")
            missing = tally["runtime_none_where_offline_has_an_example"]
            if missing["count"]:
                out.append(f"{where}: runtime None {missing['reasons']}")
            extra = tally["runtime_forecast_without_offline_example"]
            if extra["count"]:
                out.append(f"{where}: runtime only {extra['why_offline_has_none']}")
            if tally["forecasts_after_a_battle_event"]:
                out.append(f"{where}: a forecast was served mid-turn")
        if not entry["equality"]["per_event"]["calls_after_a_battle_event"]:
            out.append(f"{name}: the per-event pass made no mid-turn call")
        notes = entry.get("real_battle_notes")
        if notes is not None:
            for key in ("stream_differs", "role_differs"):
                if notes.get(key):
                    out.append(f"{name}: real battle objects: {key} on {notes[key]}")
            if PASS_REAL not in entry["equality"]:
                out.append(f"{name}: the real-battle pass compared nothing ({notes})")
        budget = report.get("p99_budget_ms")
        slow = (entry.get("latency_ms") or {}).get("all_turn_starts", {}).get("p99")
        if budget and slow is not None and slow > budget:
            out.append(
                f"{name}: p99 of a predict call at a turn start is {slow} ms, "
                f"over the budget of {budget} ms"
            )
        for label, counters in entry.get("counters", {}).items():
            bad = predictor_failures(counters)
            if bad:
                out.append(f"{name}: {label} counted {bad}")
        for block in entry.get("memory") or []:
            mode, kept = block["mode"], block["shadows_kept"]
            if mode == "forget" and kept:
                out.append(f"{name}: {kept} shadows left after forget()")
            if mode == "bounded" and kept > block["max_battles"]:
                out.append(f"{name}: {kept} shadows kept, over the bound")
            again = block.get("python_bytes_held_after_the_same_battles_again")
            if again is not None and again > block["python_bytes_held"] + MEMORY_SLACK:
                out.append(
                    f"{name}: memory grew from {block['python_bytes_held']} to "
                    f"{again} bytes when the same battles were replayed"
                )
        check = entry.get("dataset_check") or {}
        if check.get("status") == "ran":
            if check["feature_arrays_differing"]:
                out.append(
                    f"{name}: dataset rows differ {check['feature_arrays_differing']}"
                )
            if not check["compared"]:
                out.append(f"{name}: no dataset row was compared")
    return out


def run(
    artifacts: Sequence[Path | str],
    out: Path | str,
    *,
    limit: int | None = None,
    dataset: Path | str | None = DEFAULT_DATASET,
    memory_battles: int = 400,
    long_turns: int = 12,
    early_until: int = 3,
    late_from: int = 10,
    root: Path = ROOT,
    log: Any = say,
    overwrite: bool = False,
    p99_budget_ms: float | None = DEFAULT_P99_BUDGET_MS,
    real_battles: bool = True,
) -> dict[str, Any]:
    """Run every check, write ``report.json`` / ``report.md``, return the report.

    Raises ``FileExistsError`` before anything is read when ``out`` already
    holds a report and ``overwrite`` is not set.
    """
    target = Path(out)
    target = target if target.is_absolute() else root / target
    if (target / REPORT_JSON).exists() and not overwrite:
        raise FileExistsError(
            f"{target / REPORT_JSON} exists; give --overwrite to replace that report"
        )
    try:
        import torch

        torch.set_num_threads(1)
        threads = int(torch.get_num_threads())
    except Exception:
        threads = 0
    started = time.time()
    pages, bots, notes = load_pages(root, limit)
    log(f"pages: {len(pages)}")
    seats = Counter(page.role for page in pages)
    sheets = Counter(page.raw.sheet_audit or "unknown" for page in pages)
    report: dict[str, Any] = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "torch_threads": threads,
        "load_average_at_start": [round(value, 2) for value in _os.getloadavg()],
        "p99_budget_ms": float(p99_budget_ms) if p99_budget_ms else None,
        "pages": {
            "found": len(pages),
            "bot_seat": dict(seats),
            "sheet_state_known_to_the_builder": dict(sheets),
            "turn_lines": sum(page.n_turns for page in pages),
            "notes": dict(notes),
            "limit": limit,
        },
        "artifacts": [],
    }
    where = None if dataset is None else Path(dataset)
    if where is not None and not where.is_absolute():
        where = root / where
    stored = None
    if where is not None:
        try:
            stored = dataset_rows(where)
        except (OSError, ValueError, KeyError) as exc:
            log(f"dataset not read: {exc!r}")
    for artifact in artifacts:
        path = Path(artifact)
        path = path if path.is_absolute() else root / path
        log(f"artifact: {path}")
        if not path.is_file():
            report["artifacts"].append(
                {"path": str(path), "loaded": False, "load_failure": "no such file"}
            )
            continue
        report["artifacts"].append(
            run_artifact(
                path,
                pages,
                bots,
                dataset=where,
                stored=stored,
                memory_battles=memory_battles,
                long_turns=long_turns,
                early_until=early_until,
                late_from=late_from,
                real_battles=real_battles,
                log=log,
            )
        )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["peak_rss_mb"] = round(
        peak / (1024 * 1024 if _sys.platform == "darwin" else 1024), 1
    )
    report["load_average_at_end"] = [round(value, 2) for value in _os.getloadavg()]
    report["seconds"] = round(time.time() - started, 1)
    report["failures"] = failures(report)
    target.mkdir(parents=True, exist_ok=True)
    (target / REPORT_JSON).write_text(
        json.dumps(report, indent=1, default=str), encoding="utf-8"
    )
    (target / REPORT_MD).write_text(render_markdown(report), encoding="utf-8")
    return report


# --- rendering ------------------------------------------------------------------


def _row(cells: Sequence[Any]) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _stats(block: Mapping[str, Any]) -> list[Any]:
    return [block.get(key) for key in ("n", "p50", "p90", "p99", "max", "mean")]


def render_markdown(report: Mapping[str, Any]) -> str:
    """The report as markdown. Every number comes from ``report``."""
    pages = report.get("pages", {})
    lines = [
        "# Opponent predictor runtime: shadow replay",
        "",
        "Rendered from report.json by evaluation/oppmodel_shadow_replay.py.",
        "",
        f"- run: {report.get('created')}, {report.get('seconds')} s, "
        f"{report.get('torch_threads')} torch thread(s), load average "
        f"{report.get('load_average_at_start')} -> "
        f"{report.get('load_average_at_end')}, peak RSS "
        f"{report.get('peak_rss_mb')} MB",
        f"- pages: {pages.get('found')} (bot seat {pages.get('bot_seat')}), "
        f"{pages.get('turn_lines')} turn lines; sheet state known to the "
        f"builder {pages.get('sheet_state_known_to_the_builder')}",
        f"- verdict: {'PASS' if not report.get('failures') else 'FAIL'}",
    ]
    for item in report.get("failures", []):
        lines.append(f"  - {item}")
    for entry in report.get("artifacts", []):
        lines += ["", f"## {entry.get('name') or entry.get('path')}", ""]
        lines.append(
            f"- artifact: {entry.get('path')} (kind {entry.get('kind')}, "
            f"Elo mode {entry.get('elo_mode')})"
        )
        if "equality" not in entry:
            lines.append(f"- not served: {entry.get('load_failure')}")
            continue
        lines += ["", "### Runtime against the offline path", ""]
        lines.append(
            _row(
                [
                    "pass",
                    "turn lines",
                    "offline examples",
                    "compared",
                    "max diff action",
                    "target",
                    "mega",
                    "features differing",
                    "runtime None",
                    "runtime only",
                    "mid-turn forecasts / calls",
                ]
            )
        )
        lines.append(_row(["---"] * 11))
        for label, tally in entry["equality"].items():
            gaps = tally["max_abs_diff"]
            lines.append(
                _row(
                    [
                        label,
                        tally["turn_lines"],
                        tally["offline_examples"],
                        tally["compared"],
                        f"{gaps['action']:.2e}",
                        f"{gaps['target']:.2e}",
                        f"{gaps['mega']:.2e}",
                        sum(tally["feature_arrays_differing"].values()),
                        tally["runtime_none_where_offline_has_an_example"]["count"],
                        tally["runtime_forecast_without_offline_example"]["count"],
                        f"{tally['forecasts_after_a_battle_event']} / "
                        f"{tally['calls_after_a_battle_event']}",
                    ]
                )
            )
        first = entry["equality"]["per_turn"]
        single = first["max_abs_diff_one_example_per_call"]
        slots = first["max_abs_diff_normalised_slots"]
        lines += [
            "",
            f"- offline predicted one example per call: max diff action "
            f"{single['action']:.2e}, target {single['target']:.2e}, mega "
            f"{single['mega']:.2e}",
            f"- forecast slot arrays against the normalised offline output: "
            f"action {slots['action']:.2e}, target {slots['target']:.2e}",
            f"- pages the builder drops: {first['pages_dropped_offline']}; turns "
            f"where both arms have nothing: {first['both_none']['runtime_reasons']}",
        ]
        check = entry.get("dataset_check")
        if check:
            if check.get("status") == "ran":
                lines.append(
                    f"- dataset {check['dataset']}: {check['compared']} offline "
                    f"examples compared with the stored rows, arrays differing "
                    f"{check['feature_arrays_differing'] or 'none'}; "
                    f"{check['offline_examples_not_in_dataset']} examples made "
                    f"here are not in the dataset, "
                    f"{check['dataset_rows_not_made_here']} stored rows were "
                    f"not made here"
                )
            else:
                lines.append(f"- dataset check skipped: {check.get('why')}")
        notes = entry.get("real_battle_notes")
        if notes is not None:
            lines.append(
                f"- pass {PASS_REAL}: the battle is a poke-env DoubleBattle fed "
                f"through parse_message; notes {notes or 'none'}"
            )
        latency = entry["latency_ms"]
        lines += [
            "",
            "### Latency (ms per predict call)",
            "",
            f"Budget: p99 over all turn starts at most "
            f"{report.get('p99_budget_ms') or 'not set'} ms. The largest single "
            "call is reported, not gated.",
            "",
        ]
        lines.append(_row(["calls", "n", "p50", "p90", "p99", "max", "mean"]))
        lines.append(_row(["---"] * 7))
        lines.append(_row(["all turn starts", *_stats(latency["all_turn_starts"])]))
        games = latency["long_games"]
        for key, block in games.items():
            if isinstance(block, Mapping):
                lines.append(
                    _row([f"long games ({games['definition']}): {key}", *_stats(block)])
                )
        for key, block in latency["by_turn"].items():
            lines.append(_row([f"turn {key}", *_stats(block)]))
        lines.append(
            _row(
                [
                    "first call at the last turn of a long game",
                    *_stats(latency["first_call_at_the_last_turn_of_a_long_game"]),
                ]
            )
        )
        memory = entry.get("memory")
        if memory:
            lines += ["", "### Memory (Python allocations still held)", ""]
            lines.append(
                _row(
                    [
                        "mode",
                        "battles",
                        "bound",
                        "shadows kept",
                        "KiB held",
                        "KiB per kept shadow",
                        "evicted",
                        "KiB held after the same battles again",
                    ]
                )
            )
            lines.append(_row(["---"] * 8))
            for block in memory:
                each = block["bytes_per_kept_shadow"]
                again = block.get("python_bytes_held_after_the_same_battles_again")
                lines.append(
                    _row(
                        [
                            block["mode"],
                            block["battles"],
                            block["max_battles"],
                            block["shadows_kept"],
                            round(block["python_bytes_held"] / 1024, 1),
                            "-" if each is None else round(each / 1024, 1),
                            block["evicted"],
                            "-" if again is None else round(again / 1024, 1),
                        ]
                    )
                )
        counters = entry.get("counters", {})
        lines += [
            "",
            f"- runtime counters, pass 1: {counters.get('runtime_pass_1')}",
            f"- runtime counters, pass 2: {counters.get('runtime_pass_2')}",
            f"- runtime counters, pass 3: {counters.get('runtime_pass_3')}",
            f"- featurizer counters: {counters.get('featurizer') or 'none'}",
            f"- predictor counters: runtime "
            f"{counters.get('runtime_predictor') or 'none'}, offline "
            f"{counters.get('offline_predictor') or 'none'}",
        ]
    return "\n".join(lines) + "\n"


def _install_poke_env_patches() -> None:
    """The repo's poke-env patches, as the bot installs them before it plays
    (a command-line run only: a library caller decides for its own process)."""
    try:
        from vgc_bench.src import pokeenv_patches

        pokeenv_patches.install()
    except Exception as exc:
        say(f"poke-env patches not installed: {exc!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--artifact",
        action="append",
        default=None,
        help="artifact file; repeat for several (default: the flags table and "
        "the trained network, " + ", ".join(DEFAULT_ARTIFACTS) + ")",
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None, help="first N pages only")
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET, help="built dataset to cross-check"
    )
    parser.add_argument("--no-dataset", action="store_true")
    parser.add_argument(
        "--memory-battles", type=int, default=400, help="0 skips the memory part"
    )
    parser.add_argument("--long-turns", type=int, default=12)
    parser.add_argument("--early-until", type=int, default=3)
    parser.add_argument("--late-from", type=int, default=10)
    parser.add_argument(
        "--render-only", action="store_true", help="rewrite report.md from report.json"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the report that --out already holds",
    )
    parser.add_argument(
        "--p99-budget-ms",
        type=float,
        default=DEFAULT_P99_BUDGET_MS,
        help="fail when the p99 of a predict call at a turn start is above this "
        "(0 switches the check off)",
    )
    parser.add_argument(
        "--no-real-battles",
        action="store_true",
        help="skip the pass over real poke-env battle objects",
    )
    args = parser.parse_args(argv)
    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    if args.render_only:
        report = json.loads((out / REPORT_JSON).read_text(encoding="utf-8"))
        (out / REPORT_MD).write_text(render_markdown(report), encoding="utf-8")
        say(f"rendered {out / REPORT_MD}")
        return 0 if not report.get("failures") else 1
    if not args.no_real_battles:
        _install_poke_env_patches()
    try:
        report = run(
            args.artifact or DEFAULT_ARTIFACTS,
            out,
            limit=args.limit,
            dataset=None if args.no_dataset else args.dataset,
            memory_battles=args.memory_battles,
            long_turns=args.long_turns,
            early_until=args.early_until,
            late_from=args.late_from,
            overwrite=args.overwrite,
            p99_budget_ms=args.p99_budget_ms,
            real_battles=not args.no_real_battles,
        )
    except FileExistsError as exc:
        say(f"FAIL {exc}")
        return 1
    for item in report["failures"]:
        say(f"FAIL {item}")
    say(f"{'PASS' if not report['failures'] else 'FAIL'}: {out / 'report.md'}")
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
