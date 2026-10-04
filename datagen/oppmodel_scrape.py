"""Walk the public replay feed and fetch Reg M-C logs, Elo-stratified and resumable.

Feeds the opponent predictor (OPPONENT_PREDICTOR.md). Three sub-commands, each safe
to kill and restart:

  index   walk search.json?format=<fmt>&before=<cursor> back to the start of the
          format; one row per replay appended to <out>/index_<fmt>.jsonl
  fetch   download <id>.json for indexed ids that are not on disk yet, scarce Elo
          tiers first, into <out>/logs/<fmt>/part-NNNNN.jsonl.gz
  stats   totals of the index and of what is fetched (no network)

    .venv/bin/python datagen/oppmodel_scrape.py index
    .venv/bin/python datagen/oppmodel_scrape.py fetch
    .venv/bin/python datagen/oppmodel_scrape.py index --newer-only   # top-up
    .venv/bin/python datagen/oppmodel_scrape.py stats

What measuring the feed established (2026-10-04), and the rules that follow:

- ``before`` is a strict less-than and about 1.8% of adjacent feed rows share a
  second, so the next cursor is the page's oldest uploadtime + 1 and rows are
  de-duplicated by id. A strict cursor silently drops same-second rows.
- ``page=N`` stops at page 100 and shifts under the walk; ``sort=rating`` timed out
  on the server. Neither is ever requested.
- The row ``rating`` is the minimum post-battle Elo of the two players and is null
  for tournament or challenge games and for the non-final games of a Bo3 set. It
  orders the download; it is not a per-player label (that is the ``|player|`` line).
- Seven accounts sit in a quarter of the feed rows, hence the per-account cap.
- One thread and a fixed gap between requests: the live challenge listener connects
  to Showdown from this machine, so a block costs more than the time saved. A lock
  file keeps a second index or fetch run off the same output directory.

Files in the output directory (default battle_logs_feed_mc/, git-ignored):

  index_<fmt>.jsonl              one row per replay: id, uploadtime, players, rating,
                                 format (the format id), private
  index_<fmt>.state.json         the walk's cursor; a restart resumes from it
  logs/<fmt>/part-NNNNN.jsonl.gz about 2,000 games per shard, one gzip member per
                                 game: id, format, uploadtime, rating, players, log
  fetched_ids.txt                done-set, one id per line
  missing_ids.txt                ids that answered 404 or had no log (id<TAB>reason)
  rate.json, .lock, STOP         last measured pace; run lock; create STOP (or the
                                 --stop-file path) to end a run before its next request

Exit codes: 0 finished or stopped on request (limit, stop file); 2 the server kept
failing or answered something unexpected; 3 another run holds the lock; 1 file error.

Reading the result from another script: ``iter_feed_logs(out_dir)`` yields the shard
records ``{id, format, uploadtime, rating, players, log}``; ``read_index`` the rows.
"""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import fcntl
import gzip
import json
import os
import re
import time
import urllib.error
import urllib.request
import zlib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, BinaryIO, TypeGuard

REPO_ROOT = Path(__file__).resolve().parents[1]
REPLAY_HOST = "https://replay.pokemonshowdown.com"
USER_AGENT = "vgc-bench-research/0.1"
DEFAULT_FORMATS = ("gen9championsvgc2026regmc", "gen9championsvgc2026regmcbo3")
DEFAULT_OUT = REPO_ROOT / "battle_logs_feed_mc"
# Corpora already on disk; their ids are never fetched again.
DEFAULT_HAVE_DIRS = (
    REPO_ROOT / "battle_logs_top_mc_merged_20260920",
    REPO_ROOT / "battle_logs_players_t6like",
    REPO_ROOT / "battle_logs_web_mc_20260927",
)

MIN_DELAY_S = 0.3
DEFAULT_DELAY_S = 1.0
DEFAULT_TIMEOUT_S = 25.0
NOMINAL_LATENCY_S = 0.2  # measured 0.16-0.17 s; used only before a run has a rate
FULL_PAGE = 51  # 50 rows plus the server's "more" sentinel row
HEARTBEAT_EVERY = 100
SHARD_GAMES = 2000
DEFAULT_ACCOUNT_CAP = 400

BACKOFF_BASE_S = 2.0
BACKOFF_CAP_S = 120.0
LONG_PAUSE_EVERY = 5  # consecutive failures
LONG_PAUSE_S = 300.0
MAX_CONSECUTIVE_FAILURES = 20
RETRY_AFTER_STOP_S = 1800.0  # a longer Retry-After ends the run instead of waiting
PAUSE_SLICE_S = 5.0  # long sleeps are cut up so the stop file is still seen
FETCH_ATTEMPTS = 3  # tries per replay before it is left for the next run
MAX_CONSECUTIVE_MISSING = 20

TIER_ORDER = ("1400+", "1300s", "unrated", "1200s", "1100s", "1000s")
BO3_SET_GAP_S = 900
BO3_SET_GAMES = 3
TORN_TAIL_MAX_BYTES = 1 << 20
_SHARD_NAME = re.compile(r"part-(\d+)\.jsonl\.gz")


def _say(text: str) -> None:
    print(text, flush=True)


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def utc(stamp: float | None) -> str:
    if stamp is None:
        return "-"
    return datetime.fromtimestamp(stamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    minutes = int(max(0.0, seconds) // 60)
    if minutes < 60:
        return f"{minutes}m"
    return f"{minutes // 60}h{minutes % 60:02d}m"


# --------------------------------------------------------------------------- HTTP


@dataclass(frozen=True)
class HttpResponse:
    """One HTTP answer. ``status`` 0 means no answer at all (timeout, reset, DNS)."""

    status: int
    body: bytes = b""
    retry_after: str | None = None
    error: str = ""


HttpFn = Callable[[str, float], HttpResponse]


def http_get(url: str, timeout: float) -> HttpResponse:
    """The one function that touches the network. Never raises."""
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            status = int(answer.status)
            raw = answer.read()
            encoding = str(answer.headers.get("Content-Encoding") or "").lower()
            retry_after = answer.headers.get("Retry-After")
        body = gzip.decompress(raw) if encoding == "gzip" else raw
    except urllib.error.HTTPError as err:
        retry = err.headers.get("Retry-After") if err.headers else None
        return HttpResponse(status=int(err.code), retry_after=retry, error=str(err))
    except Exception as err:  # timeout, connection reset, DNS, torn gzip body
        return HttpResponse(status=0, error=repr(err))
    return HttpResponse(status=status, body=body, retry_after=retry_after)


def parse_retry_after(value: str | None, now: float) -> float | None:
    """Seconds asked for by a Retry-After header (delta-seconds or an HTTP date)."""
    if not value:
        return None
    text = value.strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(text)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, when.timestamp() - now)
    except (TypeError, ValueError, OverflowError):
        return None


def backoff_seconds(consecutive_failures: int, retry_after: float | None) -> float:
    """Pause after the n-th failure in a row: 2, 4, 8, 16 s, then five minutes on
    every fifth, capped at two minutes in between; never less than Retry-After."""
    n = max(1, consecutive_failures)
    if n % LONG_PAUSE_EVERY == 0:
        base = LONG_PAUSE_S
    else:
        base = min(BACKOFF_BASE_S * 2.0 ** min(n - 1, 16), BACKOFF_CAP_S)
    return max(base, retry_after or 0.0)


class StopRun(Exception):
    """Ends the current sub-command cleanly; everything written so far is resumable."""

    def __init__(self, reason: str, exit_code: int = 0) -> None:
        super().__init__(reason)
        self.reason = reason
        self.exit_code = exit_code


@dataclass(frozen=True)
class Reply:
    """``ok`` (parsed JSON in ``data``), ``missing`` (a 404) or ``gave_up``."""

    kind: str
    data: Any = None


@dataclass
class Client:
    """Paced, single-threaded JSON GETs with backoff, limits and a heartbeat."""

    http: HttpFn
    delay: float = DEFAULT_DELAY_S
    timeout: float = DEFAULT_TIMEOUT_S
    max_requests: int | None = None
    stop_file: Path | None = None
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    wall: Callable[[], float] = time.time
    say: Callable[[str], None] = _say
    # (detail text, seconds left or None) for the heartbeat line
    progress: Callable[[], tuple[str, float | None]] | None = None
    last: str = "-"
    requests: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    _first_start: float | None = None
    _last_end: float | None = None

    def seconds_per_request(self) -> float:
        """Measured wall time per request, never below the configured gap."""
        if self.requests and self._first_start is not None:
            if self._last_end is not None:
                measured = (self._last_end - self._first_start) / self.requests
                return max(measured, self.delay)
        return self.delay + NOMINAL_LATENCY_S

    def get_json(self, url: str, attempts: int | None = None) -> Reply:
        """GET ``url`` and parse it. A transient failure (429, 5xx, any other
        unexpected status, a timeout, an unparsable body) is retried after
        ``backoff_seconds``; ``attempts`` bounds the tries for this one URL.
        Raises StopRun on the stop file, the request limit, a Retry-After beyond
        ``RETRY_AFTER_STOP_S`` or ``MAX_CONSECUTIVE_FAILURES`` failures in a row."""
        tried = 0
        while True:
            self._gate()
            if self._first_start is None:
                self._first_start = self.clock()
            answer = self.http(url, self.timeout)
            self.requests += 1
            self._last_end = self.clock()
            data: Any = None
            problem = ""
            if answer.status == 200:
                try:
                    data = json.loads(answer.body)
                except ValueError:
                    problem = "unparsable body"
            elif answer.status != 404:
                problem = f"HTTP {answer.status}" if answer.status else "no answer"
                if answer.error:
                    problem += f" ({answer.error})"
            if not problem:
                self.consecutive_failures = 0
            self._heartbeat()
            if not problem:
                if answer.status == 404:
                    return Reply("missing")
                return Reply("ok", data)

            self.failures += 1
            self.consecutive_failures += 1
            tried += 1
            if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                raise StopRun(
                    f"{self.consecutive_failures} consecutive failed requests "
                    f"(last: {problem} on {url}); stopping. Nothing is lost - "
                    "rerun the same command later to resume.",
                    exit_code=2,
                )
            asked = parse_retry_after(answer.retry_after, self.wall())
            if asked is not None and asked > RETRY_AFTER_STOP_S:
                raise StopRun(
                    f"server asked to wait {asked:.0f} s (Retry-After) after "
                    f"{problem}; stopping instead of waiting. Rerun later to resume.",
                    exit_code=2,
                )
            wait = backoff_seconds(self.consecutive_failures, asked)
            self.say(
                f"[retry] {problem} on {url} | failure {self.consecutive_failures} "
                f"in a row | waiting {wait:.0f} s"
            )
            self._pause(wait)
            if attempts is not None and tried >= attempts:
                return Reply("gave_up")

    def _stop_requested(self) -> bool:
        return self.stop_file is not None and self.stop_file.exists()

    def _gate(self) -> None:
        if self._stop_requested():
            raise StopRun(f"stop file {self.stop_file} present (delete it to continue)")
        if self.max_requests is not None and self.requests >= self.max_requests:
            raise StopRun(f"request limit reached ({self.max_requests})")
        if self._last_end is not None:
            self._pause(self.delay - (self.clock() - self._last_end))

    def _pause(self, seconds: float) -> None:
        left = seconds
        while left > 0:
            if self._stop_requested():
                raise StopRun(
                    f"stop file {self.stop_file} present (delete it to continue)"
                )
            step = min(left, PAUSE_SLICE_S)
            self.sleep(step)
            left -= step

    def _heartbeat(self) -> None:
        if self.requests % HEARTBEAT_EVERY:
            return
        detail, eta = self.progress() if self.progress is not None else ("", None)
        elapsed = (self._last_end or 0.0) - (self._first_start or 0.0)
        rate = self.requests / elapsed if elapsed > 0 else 0.0
        self.say(
            f"[hb] requests={self.requests} rate={rate:.2f}/s "
            f"failures={self.failures} last={self.last} eta={fmt_duration(eta)}"
            + (f" | {detail}" if detail else "")
        )


# -------------------------------------------------------------------------- index


@dataclass(frozen=True)
class IndexRow:
    """One feed row. ``format`` is the format id, not the feed's display name."""

    id: str
    uploadtime: int
    players: tuple[str, ...]
    rating: int | None
    format: str
    private: int = 0

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "uploadtime": self.uploadtime,
                "players": list(self.players),
                "rating": self.rating,
                "format": self.format,
                "private": self.private,
            }
        )


def clean_row(raw: object, fmt: str) -> IndexRow | None:
    """A feed row or an index line as an IndexRow; None when it is not a row of
    ``fmt`` (the id must be ``<fmt>-<number>``)."""
    if not isinstance(raw, dict):
        return None
    replay_id, when = raw.get("id"), raw.get("uploadtime")
    if not isinstance(replay_id, str) or not replay_id.startswith(fmt + "-"):
        return None
    if not _is_int(when):
        return None
    players = raw.get("players")
    names: tuple[str, ...] = ()
    if isinstance(players, list):
        names = tuple(p for p in players if isinstance(p, str))
    rating, private = raw.get("rating"), raw.get("private")
    return IndexRow(
        id=replay_id,
        uploadtime=when,
        players=names,
        rating=rating if _is_int(rating) else None,
        format=fmt,
        private=private if _is_int(private) else 0,
    )


def index_paths(out_dir: Path, fmt: str) -> tuple[Path, Path]:
    return out_dir / f"index_{fmt}.jsonl", out_dir / f"index_{fmt}.state.json"


def read_index(path: Path, fmt: str) -> list[IndexRow]:
    """Rows of one index file, de-duplicated by id; a torn last line is skipped."""
    rows: dict[str, IndexRow] = {}
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    row = clean_row(json.loads(line), fmt)
                except ValueError:
                    continue
                if row is not None:
                    rows[row.id] = row
    except OSError:
        return []
    return list(rows.values())


@dataclass
class IndexState:
    """Resume point of one format's walk (``index_<fmt>.state.json``)."""

    format: str
    before: int | None = None  # next cursor of the backward walk; None = newest
    done: bool = False  # the backward walk reached the start of the format
    pages: int = 0
    rows: int = 0
    # An unfinished top-up: it walks from now down to ``topup_target`` and leaves a
    # gap above that second until it completes.
    topup_before: int | None = None
    topup_target: int | None = None
    same_second_overflow: int = 0
    bad_rows: int = 0
    updated: int = 0


def load_state(path: Path, fmt: str) -> IndexState:
    """The saved state, or a fresh one (walk from the newest page; the id
    de-duplication makes that safe) when the file is absent or unreadable."""
    state = IndexState(format=fmt)
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return state
    if not isinstance(raw, dict):
        return state
    for name in ("before", "topup_before", "topup_target"):
        if _is_int(raw.get(name)):
            setattr(state, name, raw[name])
    for name in ("pages", "rows", "same_second_overflow", "bad_rows", "updated"):
        if _is_int(raw.get(name)):
            setattr(state, name, raw[name])
    state.done = raw.get("done") is True
    return state


def _atomic_write(path: Path, text: str) -> None:
    scratch = path.with_name(path.name + ".tmp")
    scratch.write_text(text)
    os.replace(scratch, path)


def save_state(path: Path, state: IndexState) -> None:
    state.updated = int(time.time())
    _atomic_write(path, json.dumps(asdict(state), indent=1) + "\n")


def _open_append(path: Path) -> BinaryIO:
    """Append handle that starts on a fresh line even after a torn last line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torn = False
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            torn = handle.read(1) != b"\n"
    handle = path.open("ab")
    if torn:
        handle.write(b"\n")
    return handle


def _sync(handle: BinaryIO) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def search_url(fmt: str, before: int | None) -> str:
    """The only search request this script makes: format, and ``before`` as cursor."""
    url = f"{REPLAY_HOST}/search.json?format={fmt}"
    return url if before is None else f"{url}&before={before}"


def replay_url(replay_id: str) -> str:
    return f"{REPLAY_HOST}/{replay_id}.json"


@dataclass
class WalkReport:
    format: str
    pages: int = 0
    new_rows: int = 0
    reason: str = ""


def walk_index(
    client: Client,
    fmt: str,
    out_dir: Path,
    *,
    since: int | None = None,
    newer_only: bool = False,
    report: WalkReport | None = None,
) -> WalkReport:
    """Walk one format's feed backwards and append unseen rows to its index.

    Default: continue from the saved cursor until an empty page (the start of the
    format) or until a page's oldest row is older than ``since``; the page that
    crosses ``since`` is kept whole. ``newer_only``: walk from now down to the
    newest uploadtime already indexed, covering that whole second; an interrupted
    top-up is finished first so it cannot leave a gap. With nothing indexed yet a
    top-up is the plain walk. The state is saved after every page; StopRun from
    the client passes through, and ``report`` (when given) holds the counts so far.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path, state_path = index_paths(out_dir, fmt)
    known = read_index(index_path, fmt)
    seen = {row.id for row in known}
    newest = max((row.uploadtime for row in known), default=None)
    state = load_state(state_path, fmt)
    report = report if report is not None else WalkReport(format=fmt)
    started_at = client.clock()
    first_cursor: list[int] = []

    def pages(handle: BinaryIO, topup: bool, floor: int | None) -> str:
        nonlocal newest

        def cursor_now() -> int | None:
            return state.topup_before if topup else state.before

        def progress() -> tuple[str, float | None]:
            cursor = cursor_now()
            detail = f"{fmt} at {utc(cursor)} rows+{report.new_rows}"
            if floor is None or cursor is None or not first_cursor:
                return detail, None
            covered = first_cursor[0] - cursor
            if covered <= 0:
                return detail, None
            spent = client.clock() - started_at
            return detail, max(0, cursor - floor) / covered * spent

        client.progress = progress
        while True:
            cursor = cursor_now()
            if floor is not None and cursor is not None and cursor <= floor:
                return "reached the requested lower bound"
            reply = client.get_json(search_url(fmt, cursor))
            page = reply.data
            if reply.kind != "ok" or not isinstance(page, list):
                raise StopRun(
                    f"{fmt}: the search endpoint answered {reply.kind} "
                    f"({str(page)[:120]!r}) instead of a list of rows; stopping.",
                    exit_code=2,
                )
            state.pages += 1
            report.pages += 1
            if not page:
                if not topup:
                    state.done = True
                save_state(state_path, state)
                return "reached the start of the format (empty page)"
            usable = [r for r in (clean_row(raw, fmt) for raw in page) if r is not None]
            state.bad_rows += len(page) - len(usable)
            times = [r.uploadtime for r in usable]
            if not times or (cursor is not None and max(times) >= cursor):
                # Nothing is written from such a page: the cursor could not move.
                raise StopRun(
                    f"{fmt}: the page at before={cursor} has "
                    + ("no usable row" if not times else "rows at or above the cursor")
                    + " (the feed did not honour before=); stopping.",
                    exit_code=2,
                )
            oldest = min(times)
            fresh: list[IndexRow] = []
            for row in usable:
                if row.id not in seen:
                    seen.add(row.id)
                    fresh.append(row)
            if fresh:
                handle.write("".join(r.to_json() + "\n" for r in fresh).encode())
                _sync(handle)
                newest = max([newest or 0] + [r.uploadtime for r in fresh])
                client.last = fresh[-1].id
            if not first_cursor:
                first_cursor.append(max(times))
            following = oldest + 1
            if cursor is not None and following >= cursor:
                # Every row sits in the one second below the cursor, so "+1" would
                # ask for this page for ever. Step strictly past that second; rows
                # beyond a full page of one second cannot be reached without page=N.
                if len(page) >= FULL_PAGE:
                    state.same_second_overflow += 1
                following = oldest
            if topup:
                state.topup_before = following
            else:
                state.before = following
            state.rows += len(fresh)
            report.new_rows += len(fresh)
            save_state(state_path, state)
            if floor is not None and oldest < floor:
                return "reached the requested lower bound"

    with _open_append(index_path) as handle:
        if newer_only and newest is not None:
            if state.topup_target is not None:
                pages(handle, True, state.topup_target)
            state.topup_before, state.topup_target = None, newest
            save_state(state_path, state)
            pages(handle, True, newest)
            state.topup_before = state.topup_target = None
            save_state(state_path, state)
            report.reason = "top-up reached the rows already indexed"
            if not state.done:
                report.reason += "; the backward walk is still unfinished"
        elif state.done:
            report.reason = "already complete (use --newer-only to top up)"
        else:
            report.reason = pages(handle, False, since)
    return report


# ------------------------------------------------------------------- fetch order


def is_bo3(fmt: str) -> bool:
    # Showdown names a best-of-three format by appending "bo3" to the format id.
    return fmt.endswith("bo3")


def account_id(name: str) -> str:
    """Showdown user id: the name lower-cased with everything but a-z0-9 removed."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def accounts(row: IndexRow) -> tuple[str, ...]:
    return tuple(sorted({a for a in map(account_id, row.players) if a}))


def rating_tier(rating: int | None) -> str:
    """Download tier of a row rating (the minimum post-battle Elo of the game)."""
    if rating is None or rating <= 0:
        return "unrated"
    if rating >= 1400:
        return "1400+"
    if rating >= 1300:
        return "1300s"
    if rating >= 1200:
        return "1200s"
    if rating >= 1100:
        return "1100s"
    return "1000s"


def battle_number(replay_id: str) -> int:
    tail = replay_id.rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def _shuffle_key(replay_id: str) -> tuple[int, str]:
    return zlib.crc32(replay_id.encode()), replay_id


def group_bo3_sets(rows: Iterable[IndexRow]) -> list[list[IndexRow]]:
    """Guess Bo3 sets from index rows alone (the set marker is only in the log).

    Games of the same two accounts in the same format, taken in battle-number
    order (which is game order), are chained into one set while the upload gap
    stays within ``BO3_SET_GAP_S``, up to three games; a rated row closes its set
    because only the final game of a set carries a rating. Checked against the
    set markers of 4,731 on-disk Bo3 logs: 2,106 of 2,132 sets recovered exactly.
    """
    by_pair: dict[tuple[str, tuple[str, ...]], list[IndexRow]] = defaultdict(list)
    for row in rows:
        by_pair[(row.format, accounts(row))].append(row)
    sets: list[list[IndexRow]] = []
    for games in by_pair.values():
        games.sort(key=lambda r: (battle_number(r.id), r.id))
        current: list[IndexRow] = []
        for row in games:
            if current and (
                len(current) >= BO3_SET_GAMES
                or current[-1].rating is not None
                or abs(row.uploadtime - current[-1].uploadtime) > BO3_SET_GAP_S
            ):
                sets.append(current)
                current = []
            current.append(row)
        if current:
            sets.append(current)
    return sets


def fetch_units(rows: Iterable[IndexRow]) -> list[list[IndexRow]]:
    """The download order, as units that are fetched together.

    1. Every single-game format first, one game per unit, taking one game from
       each tier of the row rating in turn (``TIER_ORDER``: 1400+, 1300s, unrated,
       1200s, 1100s, 1000s) until a tier runs out. Any prefix is then balanced
       across the rating bands: the scarce high-Elo games are still exhausted
       early, and the 1000-1200 band the bot's own ladder opponents sit in is not
       left for last.
    2. Then every best-of-three format, one guessed set per unit (see
       ``group_bo3_sets``), by the same tier order applied to the set's rating
       (the rating of its final game; a set with no rated game is "unrated").
       Games inside a set are in game order.

    Inside a tier the order is a fixed pseudo-random one (crc32 of the id), not
    newest-first: ratings drift upward as a format ages, so any prefix of a tier
    is then an unbiased sample of the tier over upload time, and an interrupted
    run does not tie the Elo tier to the date. The order depends only on the rows.
    """
    tier_rank = {tier: rank for rank, tier in enumerate(TIER_ORDER)}
    singles = [row for row in rows if not is_bo3(row.format)]
    bo3 = [row for row in rows if is_bo3(row.format)]
    by_tier: dict[str, list[IndexRow]] = {tier: [] for tier in TIER_ORDER}
    for row in singles:
        by_tier[rating_tier(row.rating)].append(row)
    for tier_rows in by_tier.values():
        tier_rows.sort(key=lambda r: _shuffle_key(r.id))
    units = [
        [tier_rows[depth]]
        for depth in range(max((len(v) for v in by_tier.values()), default=0))
        for tier_rows in by_tier.values()
        if depth < len(tier_rows)
    ]

    def set_key(games: list[IndexRow]) -> tuple[int, tuple[int, str]]:
        rated = [g.rating for g in games if g.rating is not None]
        tier = rating_tier(max(rated) if rated else None)
        return tier_rank[tier], _shuffle_key(games[0].id)

    units.extend(sorted(group_bo3_sets(bo3), key=set_key))
    return units


def fetch_order(rows: Iterable[IndexRow]) -> list[IndexRow]:
    """``fetch_units`` flattened: the order replays are requested in."""
    return [row for unit in fetch_units(rows) for row in unit]


def apply_account_cap(
    units: Sequence[Sequence[IndexRow]],
    held: set[str],
    cap: int,
    skip: set[str] | None = None,
) -> tuple[list[IndexRow], list[IndexRow]]:
    """Split the not-yet-held rows of ``units`` into (to fetch, skipped by the cap).

    An account's count is the number of games it appears in that are already held
    (fetched here or present in another on-disk corpus) or queued earlier in the
    order. A unit is skipped only when EVERY account in it has reached ``cap``; a
    game with one capped player is kept, because the other side is still new data.
    A Bo3 set is decided as a whole, and a set with a held game is always finished.
    ``cap`` <= 0 disables the cap; ids in ``skip`` (known missing) are left out.
    Counts only grow, so a rerun after a partial fetch makes the same decisions.
    """
    skip = skip or set()
    counts: Counter[str] = Counter()
    for unit in units:
        for row in unit:
            if row.id in held:
                counts.update(accounts(row))
    queue: list[IndexRow] = []
    capped: list[IndexRow] = []
    for unit in units:
        todo = [row for row in unit if row.id not in held and row.id not in skip]
        if not todo:
            continue
        names = accounts(todo[0])
        begun = any(row.id in held for row in unit)
        full = cap > 0 and bool(names) and all(counts[a] >= cap for a in names)
        if full and not begun:
            capped.extend(todo)
            continue
        queue.extend(todo)
        for row in todo:
            counts.update(accounts(row))
    return queue, capped


# --------------------------------------------------------------- on-disk corpora


def ondisk_ids(dirs: Iterable[Path]) -> tuple[set[str], list[str]]:
    """Replay ids already held by other corpora, and one note per directory.

    Reads the keys of every ``logs_*.json`` dict and the stem of every ``*.log``
    file in each directory and its immediate sub-directories. An absent directory
    or an unreadable file is noted, never raised.
    """
    found: set[str] = set()
    notes: list[str] = []
    for directory in dirs:
        if not directory.is_dir():
            notes.append(f"{directory}: absent")
            continue
        before = len(found)
        for path in sorted(directory.glob("logs_*.json")):
            try:
                payload = json.loads(path.read_text())
            except (OSError, ValueError):
                notes.append(f"{path}: unreadable, ignored")
                continue
            if isinstance(payload, dict):
                found.update(str(key) for key in payload)
        for pattern in ("*.log", "*/*.log"):
            found.update(path.stem for path in directory.glob(pattern))
        notes.append(f"{directory}: {len(found) - before:,} new ids")
    return found, notes


def read_id_file(path: Path) -> set[str]:
    """First tab-separated field of every line (``fetched_ids.txt`` and the like)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    return {
        line.split("\t", 1)[0].strip() for line in text.splitlines() if line.strip()
    }


# ------------------------------------------------------------------------ shards


def shard_paths(out_dir: Path, fmt: str) -> list[Path]:
    folder = out_dir / "logs" / fmt
    return sorted(
        p for p in folder.glob("part-*.jsonl.gz") if _SHARD_NAME.fullmatch(p.name)
    )


def _gzip_members(data: bytes) -> tuple[list[bytes], int]:
    """Decompressed gzip members of ``data`` and the offset after the last whole
    one. A torn or corrupt member ends the read; nothing is raised."""
    view = memoryview(data)
    members: list[bytes] = []
    good = 0
    while good < len(data):
        decoder = zlib.decompressobj(wbits=31)
        parts: list[bytes] = []
        at = good
        whole = False
        try:
            while at < len(data):
                piece = view[at : at + 65536]
                parts.append(decoder.decompress(piece))
                at += len(piece)
                if decoder.eof:
                    at -= len(decoder.unused_data)
                    whole = True
                    break
        except zlib.error:
            whole = False
        if not whole:
            break
        members.append(b"".join(parts))
        good = at
    return members, good


def read_shard(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Records of one shard and the byte length of its intact prefix."""
    try:
        data = path.read_bytes()
    except OSError:
        return [], 0
    members, good = _gzip_members(data)
    records: list[dict[str, Any]] = []
    for blob in members:
        for line in blob.splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and isinstance(record.get("id"), str):
                records.append(record)
    return records, good


def iter_feed_logs(
    out_dir: Path = DEFAULT_OUT, formats: Sequence[str] = DEFAULT_FORMATS
) -> Iterator[dict[str, Any]]:
    """Every fetched replay ``{id, format, uploadtime, rating, players, log}``, once
    per id, in shard order. Read-only: a torn shard tail is skipped, not repaired."""
    seen: set[str] = set()
    for fmt in formats:
        for path in shard_paths(out_dir, fmt):
            for record in read_shard(path)[0]:
                if record["id"] not in seen:
                    seen.add(record["id"])
                    yield record


class FeedStore:
    """The fetched side of an output directory: shards, done-set and missing-set.

    Each game is one gzip member appended to its format's current shard, so a shard
    is a valid multi-member gzip file after every write and plain ``gzip.open``
    reads it. Write order is shard, then ``fetched_ids.txt``; ``open_format``
    repairs both sides of a kill between the two.
    """

    def __init__(self, out_dir: Path, shard_size: int = SHARD_GAMES) -> None:
        self.out_dir = out_dir
        self.shard_size = max(1, shard_size)
        self.fetched_path = out_dir / "fetched_ids.txt"
        self.missing_path = out_dir / "missing_ids.txt"
        self.fetched = read_id_file(self.fetched_path)
        self.missing = read_id_file(self.missing_path)
        self._shard: dict[str, tuple[int, int]] = {}  # fmt -> (number, games in it)
        self._ids: BinaryIO | None = None

    def _mark_fetched(self, replay_id: str) -> None:
        if self._ids is None:
            self._ids = _open_append(self.fetched_path)
        self._ids.write(replay_id.encode() + b"\n")
        _sync(self._ids)
        self.fetched.add(replay_id)

    def open_format(self, fmt: str) -> list[str]:
        """Find the shard to append to and repair a kill's leftovers: a torn last
        member is cut off (that game is fetched again), and games present in the
        last shard but absent from the done-set are added to it."""
        notes: list[str] = []
        paths = shard_paths(self.out_dir, fmt)
        if not paths:
            self._shard[fmt] = (0, 0)
            return notes
        last = paths[-1]
        match = _SHARD_NAME.fullmatch(last.name)
        number = int(match.group(1)) if match else len(paths) - 1
        records, good = read_shard(last)
        try:
            size = last.stat().st_size
        except OSError:
            size = good
        if good < size:
            if size - good <= TORN_TAIL_MAX_BYTES:
                with last.open("r+b") as handle:
                    handle.truncate(good)
                notes.append(f"{last.name}: cut a torn tail of {size - good:,} bytes")
            else:
                # Not a torn append: leave the file alone and start a new shard.
                notes.append(
                    f"{last.name}: {size - good:,} unreadable bytes, left untouched"
                )
                self._shard[fmt] = (number + 1, 0)
                return notes
        recovered = [r["id"] for r in records if r["id"] not in self.fetched]
        for replay_id in recovered:
            self._mark_fetched(replay_id)
        if recovered:
            notes.append(f"{last.name}: {len(recovered)} ids restored to the done-set")
        self._shard[fmt] = (number, len(records))
        return notes

    def add(self, record: dict[str, Any]) -> Path:
        """Append one game and mark it fetched; both are on disk before returning."""
        fmt = str(record["format"])
        if fmt not in self._shard:
            self.open_format(fmt)
        number, count = self._shard[fmt]
        if count >= self.shard_size:
            number, count = number + 1, 0
        path = self.out_dir / "logs" / fmt / f"part-{number:05d}.jsonl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(record) + "\n").encode()
        with path.open("ab") as handle:
            handle.write(gzip.compress(line, compresslevel=6, mtime=0))
            _sync(handle)
        self._shard[fmt] = (number, count + 1)
        self._mark_fetched(str(record["id"]))
        return path

    def mark_missing(self, replay_id: str, reason: str) -> None:
        with _open_append(self.missing_path) as handle:
            handle.write(f"{replay_id}\t{reason}\n".encode())
            _sync(handle)
        self.missing.add(replay_id)

    def close(self) -> None:
        if self._ids is not None:
            self._ids.close()
            self._ids = None


# ------------------------------------------------------------------------- fetch


def load_rows(out_dir: Path, formats: Sequence[str]) -> list[IndexRow]:
    rows: dict[str, IndexRow] = {}
    for fmt in formats:
        for row in read_index(index_paths(out_dir, fmt)[0], fmt):
            rows[row.id] = row
    return list(rows.values())


@dataclass
class FetchPlan:
    queue: list[IndexRow]  # what to request, in order
    capped: list[IndexRow]  # left out by the account cap
    indexed: int = 0
    on_disk: int = 0  # indexed ids present in another corpus
    fetched: int = 0  # indexed ids already in this directory's shards
    missing: int = 0
    private: int = 0


def plan_fetch(
    rows: Sequence[IndexRow],
    disk: set[str],
    fetched: set[str],
    missing: set[str],
    cap: int,
) -> FetchPlan:
    """What a fetch run would request: the fetch order minus everything held,
    known missing or private, with the account cap applied."""
    public = [row for row in rows if not row.private]
    ids = {row.id for row in public}
    queue, capped = apply_account_cap(
        fetch_units(public), held=disk | fetched, cap=cap, skip=missing
    )
    return FetchPlan(
        queue=queue,
        capped=capped,
        indexed=len(rows),
        on_disk=len(ids & (disk - fetched)),
        fetched=len(ids & fetched),
        missing=len(ids & (missing - disk - fetched)),
        private=len(rows) - len(public),
    )


def replay_record(row: IndexRow, payload: object) -> dict[str, Any] | None:
    """The shard line for a fetched replay; None when the answer has no usable log
    or belongs to another replay. Fields missing from the answer come from the row."""
    if not isinstance(payload, dict):
        return None
    log = payload.get("log")
    if not isinstance(log, str) or not log.strip():
        return None
    if isinstance(payload.get("id"), str) and payload["id"] != row.id:
        return None
    players = payload.get("players")
    names = (
        [p for p in players if isinstance(p, str)] if isinstance(players, list) else []
    )
    when, rating = payload.get("uploadtime"), payload.get("rating")
    return {
        "id": row.id,
        "format": row.format,
        "uploadtime": when if _is_int(when) else row.uploadtime,
        "rating": rating if _is_int(rating) else row.rating,
        "players": names or list(row.players),
        "log": log,
    }


@dataclass
class FetchReport:
    queued: int = 0
    fetched: int = 0
    missing: int = 0
    deferred: int = 0
    capped: int = 0
    reason: str = ""


def fetch_logs(
    client: Client,
    out_dir: Path,
    formats: Sequence[str],
    *,
    have_dirs: Sequence[Path] = DEFAULT_HAVE_DIRS,
    account_cap: int = DEFAULT_ACCOUNT_CAP,
    max_games: int | None = None,
    shard_size: int = SHARD_GAMES,
    report: FetchReport | None = None,
) -> FetchReport:
    """Fetch the planned replays into gzip JSONL shards.

    A 404 or an answer without a log is written to ``missing_ids.txt`` and never
    asked for again. ``MAX_CONSECUTIVE_MISSING`` of them in a row stop the run and
    are NOT written: that pattern means the endpoint changed rather than twenty
    replays vanished, and a poisoned missing-set would hide those games for good.
    A replay that keeps failing transiently is left for the next run after
    ``FETCH_ATTEMPTS`` tries. StopRun from the client passes through, and
    ``report`` (when given) holds the counts so far.
    """
    report = report if report is not None else FetchReport()
    out_dir.mkdir(parents=True, exist_ok=True)
    store = FeedStore(out_dir, shard_size)
    unrecorded: list[tuple[str, str]] = []  # the current run of missing replays

    def record_missing() -> None:
        for replay_id, why in unrecorded:
            store.mark_missing(replay_id, why)
        report.missing += len(unrecorded)
        unrecorded.clear()

    try:
        for fmt in formats:
            for note in store.open_format(fmt):
                client.say(f"[repair] {note}")
        disk, notes = ondisk_ids(have_dirs)
        for note in notes:
            client.say(f"[have] {note}")
        rows = load_rows(out_dir, formats)
        plan = plan_fetch(rows, disk, store.fetched, store.missing, account_cap)
        report.queued, report.capped = len(plan.queue), len(plan.capped)
        client.say(
            f"[plan] indexed {plan.indexed:,} | in other corpora {plan.on_disk:,} | "
            f"fetched {plan.fetched:,} | missing {plan.missing:,} | skipped by the "
            f"account cap ({account_cap}) {len(plan.capped):,} | to fetch "
            f"{len(plan.queue):,}"
        )
        position = 0

        def progress() -> tuple[str, float | None]:
            left = len(plan.queue) - position
            detail = f"fetched {report.fetched:,} this run, {left:,} queued"
            return detail, left * client.seconds_per_request()

        client.progress = progress
        for position, row in enumerate(plan.queue):
            if max_games is not None and report.fetched >= max_games:
                report.reason = f"game limit reached ({max_games})"
                return report
            client.last = row.id
            reply = client.get_json(replay_url(row.id), attempts=FETCH_ATTEMPTS)
            if reply.kind == "gave_up":
                report.deferred += 1
                continue
            record = replay_record(row, reply.data) if reply.kind == "ok" else None
            if record is None:
                unrecorded.append(
                    (row.id, "404" if reply.kind == "missing" else "nolog")
                )
                if len(unrecorded) >= MAX_CONSECUTIVE_MISSING:
                    first, count = unrecorded[0][0], len(unrecorded)
                    unrecorded.clear()
                    raise StopRun(
                        f"{count} replays in a row were missing or had no log "
                        f"({first} .. {row.id}); the endpoint may have changed, so "
                        f"none of them was written to {store.missing_path.name}. "
                        "Stopping.",
                        exit_code=2,
                    )
                continue
            record_missing()
            store.add(record)
            report.fetched += 1
        report.reason = "queue finished"
        if report.deferred:
            report.reason += f" ({report.deferred} deferred replays remain; rerun)"
        return report
    finally:
        record_missing()
        store.close()


# ------------------------------------------------------------------------- stats


def read_rate(out_dir: Path) -> float | None:
    try:
        raw = json.loads((out_dir / "rate.json").read_text())
        value = raw.get("seconds_per_request") if isinstance(raw, dict) else None
    except (OSError, ValueError):
        return None
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


def write_rate(out_dir: Path, client: Client) -> None:
    """Remember the measured pace for ``stats`` (short runs are too noisy to keep)."""
    if client.requests < 10 or not out_dir.is_dir():
        return
    payload = {
        "seconds_per_request": round(client.seconds_per_request(), 4),
        "requests": client.requests,
        "delay": client.delay,
        "updated": int(time.time()),
    }
    _atomic_write(out_dir / "rate.json", json.dumps(payload) + "\n")


def stats_text(
    out_dir: Path,
    formats: Sequence[str],
    have_dirs: Sequence[Path] = DEFAULT_HAVE_DIRS,
    account_cap: int = DEFAULT_ACCOUNT_CAP,
    delay: float = DEFAULT_DELAY_S,
) -> str:
    """The stats report: index totals, tiers, days, heavy accounts, what is held,
    what remains and how long that takes at the last measured rate."""
    lines: list[str] = []
    by_fmt = {fmt: read_index(index_paths(out_dir, fmt)[0], fmt) for fmt in formats}
    rows = [row for fmt in formats for row in by_fmt[fmt]]
    lines.append(f"== index ({out_dir}) ==")
    for fmt in formats:
        mine = by_fmt[fmt]
        state = load_state(index_paths(out_dir, fmt)[1], fmt)
        times = [row.uploadtime for row in mine]
        walk = "walk complete" if state.done else f"walk at {utc(state.before)}"
        if state.topup_target is not None:
            walk += f", top-up unfinished (gap above {utc(state.topup_target)})"
        span = f"{utc(min(times))} .. {utc(max(times))} UTC" if times else "no rows"
        lines.append(
            f"{fmt}: {len(mine):,} rows | {span} | {walk} | pages {state.pages:,} | "
            f"same-second overflows {state.same_second_overflow} | "
            f"bad rows {state.bad_rows}"
        )
    lines.append(f"total rows: {len(rows):,}")

    lines.append("")
    lines.append("== rows per rating tier (row rating = min post-battle Elo) ==")
    wide = max([len(fmt) for fmt in formats] + [16]) + 2
    header = "".join(fmt.rjust(wide) for fmt in formats)
    lines.append("tier".ljust(16) + header)
    for tier in TIER_ORDER:
        cells = ""
        for fmt in formats:
            n = sum(1 for row in by_fmt[fmt] if rating_tier(row.rating) == tier)
            share = 100.0 * n / len(by_fmt[fmt]) if by_fmt[fmt] else 0.0
            cells += f"{n:,} ({share:.1f}%)".rjust(wide)
        lines.append(tier.ljust(16) + cells)
    for floor in (1500, 1600):
        cells = ""
        for fmt in formats:
            n = sum(1 for row in by_fmt[fmt] if (row.rating or 0) >= floor)
            cells += f"{n:,}".rjust(wide)
        lines.append(f"of which {floor}+".ljust(16) + cells)

    lines.append("")
    lines.append("== rows per day (UTC) ==")
    lines.append("day".ljust(16) + header)
    days: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        days[utc(row.uploadtime)[:10]][row.format] += 1
    for day in sorted(days):
        cells = "".join(f"{days[day][fmt]:,}".rjust(wide) for fmt in formats)
        lines.append(day.ljust(16) + cells)

    lines.append("")
    lines.append("== top 20 accounts by row count ==")
    per_account: Counter[str] = Counter()
    for row in rows:
        per_account.update(accounts(row))
    lines.append(f"distinct accounts: {len(per_account):,}")
    for rank, (name, n) in enumerate(per_account.most_common(20), start=1):
        share = 100.0 * n / len(rows) if rows else 0.0
        lines.append(f"{rank:>2}. {name.ljust(24)} {n:>7,}  {share:5.1f}% of rows")

    lines.append("")
    lines.append("== fetch ==")
    disk, notes = ondisk_ids(have_dirs)
    lines.extend(f"have: {note}" for note in notes)
    fetched = read_id_file(out_dir / "fetched_ids.txt")
    missing = read_id_file(out_dir / "missing_ids.txt")
    plan = plan_fetch(rows, disk, fetched, missing, account_cap)
    shards = sum(len(shard_paths(out_dir, fmt)) for fmt in formats)
    lines.append(f"indexed ids: {plan.indexed:,}")
    lines.append(f"already on disk in other corpora: {plan.on_disk:,}")
    lines.append(f"fetched into this directory: {plan.fetched:,} ({shards} shards)")
    lines.append(f"fetched ids in total (done-set): {len(fetched):,}")
    lines.append(f"missing (404 or no log): {plan.missing:,}")
    lines.append(f"skipped by the account cap ({account_cap}): {len(plan.capped):,}")
    lines.append(f"remaining to fetch: {len(plan.queue):,}")
    left: Counter[str] = Counter()
    for row in plan.queue:
        kind = "bo3" if is_bo3(row.format) else f"singles {rating_tier(row.rating)}"
        left[kind] += 1
    for kind in [f"singles {tier}" for tier in TIER_ORDER] + ["bo3"]:
        if left[kind]:
            lines.append(f"  {kind}: {left[kind]:,}")
    measured = read_rate(out_dir)
    pace = measured if measured is not None else delay + NOMINAL_LATENCY_S
    source = "last run" if measured is not None else f"--delay {delay} + latency"
    lines.append(
        f"estimated time for the rest: {fmt_duration(len(plan.queue) * pace)} "
        f"at {pace:.2f} s per request ({source})"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI


def acquire_lock(out_dir: Path) -> BinaryIO | None:
    """Exclusive lock on ``<out>/.lock`` for the life of the process, or None when
    another index or fetch run holds it: two runs would double the request rate
    and fetch the same queue twice. The lock dies with its process, so a kill
    leaves nothing to clean up."""
    out_dir.mkdir(parents=True, exist_ok=True)
    handle = (out_dir / ".lock").open("ab")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def _formats(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser, network: bool) -> None:
        sub.add_argument(
            "--formats",
            type=_formats,
            default=list(DEFAULT_FORMATS),
            help="comma-separated format ids (default: Reg M-C and Reg M-C Bo3)",
        )
        sub.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output dir")
        sub.add_argument(
            "--delay",
            type=float,
            default=DEFAULT_DELAY_S,
            help=f"seconds between requests (default 1.0, minimum {MIN_DELAY_S})",
        )
        if not network:
            return
        sub.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
        sub.add_argument("--max-requests", type=int, default=None)
        sub.add_argument(
            "--stop-file",
            type=Path,
            default=None,
            help="exit cleanly before the next request once this file exists "
            "(default: <out>/STOP)",
        )

    index = commands.add_parser("index", help="walk the feed into index_<fmt>.jsonl")
    common(index, network=True)
    index.add_argument("--since", type=int, default=None, help="unix time to stop at")
    index.add_argument(
        "--newer-only",
        action="store_true",
        help="top-up: walk from now back to the newest row already indexed",
    )

    fetch = commands.add_parser("fetch", help="download logs in Elo-priority order")
    common(fetch, network=True)
    stats = commands.add_parser("stats", help="print totals; no network")
    common(stats, network=False)
    for sub in (fetch, stats):
        sub.add_argument("--account-cap", type=int, default=DEFAULT_ACCOUNT_CAP)
        sub.add_argument(
            "--have-dir",
            type=Path,
            action="append",
            default=None,
            help="corpus directory whose ids are skipped (repeatable; default: "
            "the three Reg M-C corpora in the repo)",
        )
    fetch.add_argument("--max-games", type=int, default=None)
    fetch.add_argument("--shard-size", type=int, default=SHARD_GAMES)
    return parser


def main(
    argv: Sequence[str] | None = None,
    http: HttpFn = http_get,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.delay < MIN_DELAY_S:
        parser.error(f"--delay must be at least {MIN_DELAY_S} s (got {args.delay})")
    out_dir: Path = args.out
    formats: list[str] = args.formats

    if args.command == "stats":
        have = tuple(args.have_dir) if args.have_dir else DEFAULT_HAVE_DIRS
        _say(stats_text(out_dir, formats, have, args.account_cap, args.delay))
        return 0

    if args.command == "index" and args.newer_only and args.since is not None:
        parser.error("--newer-only and --since cannot be combined")
    client = Client(
        http=http,
        delay=args.delay,
        timeout=args.timeout,
        max_requests=args.max_requests,
        stop_file=args.stop_file if args.stop_file is not None else out_dir / "STOP",
        sleep=sleep,
    )
    code = 0
    walks: list[WalkReport] = []
    fetched = FetchReport()
    try:
        lock = acquire_lock(out_dir)
    except OSError as err:
        _say(f"[stop] cannot use {out_dir}: {err!r}")
        return 1
    if lock is None:
        _say(f"[stop] another oppmodel_scrape run holds {out_dir / '.lock'}; exiting")
        return 3
    _say(
        f"[start] {args.command} formats={','.join(formats)} out={out_dir} "
        f"delay={args.delay} max_requests={args.max_requests} "
        f"stop_file={client.stop_file}"
    )
    try:
        if args.command == "index":
            for fmt in formats:
                walks.append(WalkReport(format=fmt))
                walk_index(
                    client,
                    fmt,
                    out_dir,
                    since=args.since,
                    newer_only=args.newer_only,
                    report=walks[-1],
                )
        else:
            have = tuple(args.have_dir) if args.have_dir else DEFAULT_HAVE_DIRS
            fetch_logs(
                client,
                out_dir,
                formats,
                have_dirs=have,
                account_cap=args.account_cap,
                max_games=args.max_games,
                shard_size=args.shard_size,
                report=fetched,
            )
    except StopRun as stop:
        _say(f"[stop] {stop.reason}")
        code = stop.exit_code
    except KeyboardInterrupt:
        _say("[stop] interrupted; rerun the same command to resume")
        code = 130
    except OSError as err:
        _say(f"[stop] file error: {err!r}; rerun the same command to resume")
        code = 1
    finally:
        lock.close()
    for walk in walks:
        _say(
            f"[index] {walk.format}: {walk.pages} pages, +{walk.new_rows:,} rows, "
            f"{walk.reason or 'stopped before the end'}"
        )
    if args.command == "fetch":
        _say(
            f"[fetch] fetched {fetched.fetched:,} | missing {fetched.missing:,} | "
            f"deferred {fetched.deferred:,} | queued at start {fetched.queued:,} | "
            f"{fetched.reason or 'stopped before the end'}"
        )
    _say(
        f"[done] {client.requests} requests, {client.failures} failures, "
        f"{client.seconds_per_request():.2f} s per request"
    )
    write_rate(out_dir, client)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
