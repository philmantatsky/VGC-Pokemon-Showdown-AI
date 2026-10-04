"""Replay scraper: feed pagination, fetch order, account cap, skip set, backoff.

Nothing here touches the network. The scraper's HTTP layer is one injectable
function, so every test drives it with an in-memory feed and a fake clock. The
rules under test came from measuring the real feed: rows share upload seconds, the
cursor is strict, a few accounts dominate, and a long run gets killed and resumed.
"""

import gzip
import json
import random
import re
import urllib.parse
import zlib
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from datagen import oppmodel_scrape as scrape
from datagen.oppmodel_scrape import (
    Client,
    FeedStore,
    HttpResponse,
    IndexRow,
    StopRun,
    apply_account_cap,
    backoff_seconds,
    fetch_logs,
    fetch_order,
    fetch_units,
    group_bo3_sets,
    index_paths,
    iter_feed_logs,
    load_state,
    main,
    ondisk_ids,
    parse_retry_after,
    plan_fetch,
    rating_tier,
    read_index,
    read_shard,
    shard_paths,
    stats_text,
    walk_index,
)

FMT = "gen9championsvgc2026regmc"
BO3 = FMT + "bo3"


def feed_row(
    number: int,
    when: int,
    players: Sequence[str] = ("alice", "bob"),
    rating: int | None = 1200,
    fmt: str = FMT,
) -> dict[str, Any]:
    """A row exactly as search.json returns it."""
    return {
        "uploadtime": when,
        "id": f"{fmt}-{number}",
        "format": "[Gen 9 Champions] display name",
        "players": list(players),
        "rating": rating,
        "private": 0,
        "password": None,
    }


def row(
    number: int,
    when: int = 1000,
    players: Sequence[str] = ("alice", "bob"),
    rating: int | None = 1200,
    fmt: str = FMT,
) -> IndexRow:
    return IndexRow(f"{fmt}-{number}", when, tuple(players), rating, fmt)


class FakeTime:
    def __init__(self):
        self.now = 5000.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeFeed:
    """A replay server in memory: search.json pages and <id>.json replays."""

    def __init__(self, rows=(), page=51):
        self.rows = list(rows)
        self.page = page
        self.urls = []
        self.script = []  # answers returned before the feed itself replies
        self.gone = set()  # replay ids that answer 404
        self.ignore_before = False

    def __call__(self, url, timeout):
        self.urls.append(url)
        if self.script:
            return self.script.pop(0)
        parsed = urllib.parse.urlparse(url)
        if parsed.path == "/search.json":
            query = urllib.parse.parse_qs(parsed.query)
            assert set(query) <= {"format", "before"}, url
            fmt = query["format"][0]
            before = int(query["before"][0]) if "before" in query else None
            if self.ignore_before:
                before = None
            rows = [
                r
                for r in self.rows
                if r["id"].startswith(fmt + "-")
                and (before is None or r["uploadtime"] < before)
            ]
            rows.sort(key=lambda r: (-r["uploadtime"], r["id"]))
            return HttpResponse(200, json.dumps(rows[: self.page]).encode())
        replay_id = parsed.path.strip("/").removesuffix(".json")
        found = [r for r in self.rows if r["id"] == replay_id]
        if not found or replay_id in self.gone:
            return HttpResponse(404, error="not found")
        r = found[0]
        log = (
            f"|player|p1|{r['players'][0]}|1|1210\n"
            f"|player|p2|{r['players'][1]}|2|1195\n|turn|1\n|win|x\n"
        )
        payload = {**r, "log": log, "views": "0", "formatid": FMT}
        return HttpResponse(200, json.dumps(payload).encode())

    def search_urls(self):
        return [u for u in self.urls if "search.json" in u]

    def replay_ids(self):
        return [
            u.rsplit("/", 1)[1].removesuffix(".json")
            for u in self.urls
            if "search.json" not in u
        ]


def make_client(feed, **kwargs):
    clock = FakeTime()
    lines = []
    client = Client(
        http=feed,
        delay=1.0,
        sleep=clock.sleep,
        clock=clock.clock,
        wall=clock.clock,
        say=lines.append,
        **kwargs,
    )
    return client, clock, lines


def index_line_ids(out_dir, fmt=FMT):
    text = index_paths(out_dir, fmt)[0].read_text()
    return [json.loads(line)["id"] for line in text.splitlines() if line.strip()]


def spaced_rows(count, start=10_000, step=10, first_number=1, fmt=FMT):
    """``count`` rows, newest first, one every ``step`` seconds."""
    return [feed_row(first_number + i, start - step * i, fmt=fmt) for i in range(count)]


def failing(status, retry_after=None):
    return HttpResponse(status, retry_after=retry_after, error="boom")


# ---------------------------------------------------------------- index pagination


class TestIndexWalk:
    def test_rows_sharing_a_second_across_a_page_edge_are_all_indexed(self, tmp_path):
        rows = spaced_rows(140)
        # Rows 50, 51 and 52 (newest first) share one second: page one ends on row
        # 50, and a strict cursor would never return rows 51 and 52.
        for i in (51, 52):
            rows[i]["uploadtime"] = rows[50]["uploadtime"]
        feed = FakeFeed(rows)
        client, _, _ = make_client(feed)

        report = walk_index(client, FMT, tmp_path)

        ids = index_line_ids(tmp_path)
        assert sorted(ids) == sorted(r["id"] for r in rows)
        assert len(ids) == len(set(ids))
        assert report.new_rows == 140
        state = load_state(index_paths(tmp_path, FMT)[1], FMT)
        assert state.done and state.rows == 140
        assert state.same_second_overflow == 0
        assert feed.search_urls()[0] == scrape.search_url(FMT, None)
        second = f"before={rows[50]['uploadtime'] + 1}"
        assert feed.search_urls()[1].endswith(second)

    def test_no_request_ever_uses_page_or_sort(self, tmp_path):
        feed = FakeFeed(spaced_rows(120))
        client, _, _ = make_client(feed)
        walk_index(client, FMT, tmp_path)
        walk_index(client, FMT, tmp_path, newer_only=True)
        assert feed.urls
        for url in feed.urls:
            query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            assert set(query) <= {"format", "before"}
            assert "page=" not in url and "sort=" not in url

    def test_index_row_fields(self, tmp_path):
        feed = FakeFeed([feed_row(7, 900, ("Ann A", "bee"), None)])
        client, _, _ = make_client(feed)
        walk_index(client, FMT, tmp_path)
        line = json.loads(index_paths(tmp_path, FMT)[0].read_text().splitlines()[0])
        assert line == {
            "id": f"{FMT}-7",
            "uploadtime": 900,
            "players": ["Ann A", "bee"],
            "rating": None,
            "format": FMT,
            "private": 0,
        }

    def test_resume_continues_from_the_saved_cursor(self, tmp_path):
        rows = spaced_rows(140)
        feed = FakeFeed(rows)
        client, _, _ = make_client(feed, max_requests=2)
        with pytest.raises(StopRun) as stop:
            walk_index(client, FMT, tmp_path)
        assert stop.value.exit_code == 0
        state = load_state(index_paths(tmp_path, FMT)[1], FMT)
        assert not state.done and state.pages == 2
        assert state.before == rows[100]["uploadtime"] + 1

        feed.urls.clear()
        client, _, _ = make_client(feed)
        walk_index(client, FMT, tmp_path)

        assert feed.urls[0].endswith(f"before={state.before}")
        ids = index_line_ids(tmp_path)
        assert sorted(ids) == sorted(r["id"] for r in rows)
        assert len(ids) == len(set(ids))
        assert load_state(index_paths(tmp_path, FMT)[1], FMT).done

    def test_kill_between_row_append_and_state_save_adds_no_duplicate(self, tmp_path):
        rows = spaced_rows(140)
        feed = FakeFeed(rows)
        state_path = index_paths(tmp_path, FMT)[1]
        client, _, _ = make_client(feed, max_requests=1)
        with pytest.raises(StopRun):
            walk_index(client, FMT, tmp_path)
        after_page_one = state_path.read_text()
        client, _, _ = make_client(feed, max_requests=1)
        with pytest.raises(StopRun):
            walk_index(client, FMT, tmp_path)
        # Page two is in the index but its state write "never happened".
        state_path.write_text(after_page_one)

        client, _, _ = make_client(feed)
        walk_index(client, FMT, tmp_path)

        ids = index_line_ids(tmp_path)
        assert len(ids) == len(set(ids)) == 140

    def test_empty_first_page_ends_the_walk(self, tmp_path):
        feed = FakeFeed([])
        client, _, _ = make_client(feed)
        report = walk_index(client, FMT, tmp_path)
        assert report.new_rows == 0 and len(feed.urls) == 1
        assert load_state(index_paths(tmp_path, FMT)[1], FMT).done
        walk_index(client, FMT, tmp_path)
        assert len(feed.urls) == 1  # a finished walk asks for nothing

    def test_walk_stops_only_at_an_empty_page(self, tmp_path):
        feed = FakeFeed(spaced_rows(30))
        client, _, _ = make_client(feed)
        walk_index(client, FMT, tmp_path)
        last = urllib.parse.urlparse(feed.urls[-1])
        before = int(urllib.parse.parse_qs(last.query)["before"][0])
        assert before == min(r["uploadtime"] for r in feed.rows)
        assert len(index_line_ids(tmp_path)) == 30

    def test_a_full_page_inside_one_second_does_not_loop(self, tmp_path):
        newer = spaced_rows(10, start=9_000)
        same = [feed_row(100 + i, 8_000) for i in range(60)]
        older = spaced_rows(10, start=7_000, first_number=200)
        feed = FakeFeed(newer + same + older)
        client, _, _ = make_client(feed, max_requests=50)

        walk_index(client, FMT, tmp_path)

        state = load_state(index_paths(tmp_path, FMT)[1], FMT)
        assert state.done
        assert state.same_second_overflow == 1
        ids = set(index_line_ids(tmp_path))
        assert {r["id"] for r in newer + older} <= ids
        # A page holds 51 rows, so 9 of the 60 same-second rows are unreachable.
        assert len(ids) == 10 + 51 + 10

    def test_since_stops_and_keeps_the_crossing_page_whole(self, tmp_path):
        rows = spaced_rows(140)
        since = rows[70]["uploadtime"]
        feed = FakeFeed(rows)
        client, _, _ = make_client(feed)

        report = walk_index(client, FMT, tmp_path, since=since)

        assert len(feed.urls) == 2 and "lower bound" in report.reason
        ids = set(index_line_ids(tmp_path))
        assert ids == {r["id"] for r in rows[:101]}
        assert not load_state(index_paths(tmp_path, FMT)[1], FMT).done
        walk_index(client, FMT, tmp_path, since=since)
        assert len(feed.urls) == 2  # nothing at or above --since is left
        walk_index(client, FMT, tmp_path)
        assert len(index_line_ids(tmp_path)) == 140

    def test_newer_only_walks_down_to_the_indexed_rows(self, tmp_path):
        old = spaced_rows(100, start=1_990)
        feed = FakeFeed(old)
        client, _, _ = make_client(feed)
        walk_index(client, FMT, tmp_path)
        state_before = load_state(index_paths(tmp_path, FMT)[1], FMT)
        new = spaced_rows(120, start=3_190, first_number=1_000)
        # One late row lands in the newest second that is already indexed.
        late = feed_row(5_000, 1_990)
        feed.rows += new + [late]
        feed.urls.clear()

        report = walk_index(client, FMT, tmp_path, newer_only=True)

        assert report.new_rows == 121 and len(feed.urls) == 3
        ids = index_line_ids(tmp_path)
        assert sorted(ids) == sorted(r["id"] for r in feed.rows)
        assert len(ids) == len(set(ids))
        state = load_state(index_paths(tmp_path, FMT)[1], FMT)
        assert state.done and state.before == state_before.before
        assert state.topup_target is None and state.topup_before is None

    def test_interrupted_top_up_leaves_no_gap(self, tmp_path):
        feed = FakeFeed(spaced_rows(100, start=1_990))
        client, _, _ = make_client(feed)
        walk_index(client, FMT, tmp_path)
        feed.rows += spaced_rows(120, start=3_190, first_number=1_000)
        client, _, _ = make_client(feed, max_requests=1)
        with pytest.raises(StopRun):
            walk_index(client, FMT, tmp_path, newer_only=True)
        state = load_state(index_paths(tmp_path, FMT)[1], FMT)
        assert state.topup_target == 1_990 and state.topup_before == 2_691
        assert len(index_line_ids(tmp_path)) == 151
        feed.rows += spaced_rows(10, start=3_290, first_number=2_000)

        client, _, _ = make_client(feed)
        walk_index(client, FMT, tmp_path, newer_only=True)

        ids = index_line_ids(tmp_path)
        assert sorted(ids) == sorted(r["id"] for r in feed.rows)
        assert len(ids) == len(set(ids)) == 230
        state = load_state(index_paths(tmp_path, FMT)[1], FMT)
        assert state.topup_target is None

    def test_newer_only_with_an_empty_index_is_the_plain_walk(self, tmp_path):
        feed = FakeFeed(spaced_rows(60))
        client, _, _ = make_client(feed)
        walk_index(client, FMT, tmp_path, newer_only=True)
        assert len(index_line_ids(tmp_path)) == 60
        assert load_state(index_paths(tmp_path, FMT)[1], FMT).done

    def test_torn_last_index_line_is_tolerated(self, tmp_path):
        index_path = index_paths(tmp_path, FMT)[0]
        whole = row(1, 500).to_json()
        index_path.write_text(whole + "\n" + row(2, 490).to_json()[:25])
        assert [r.id for r in read_index(index_path, FMT)] == [f"{FMT}-1"]

        feed = FakeFeed(spaced_rows(3, start=480, first_number=10))
        client, _, _ = make_client(feed)
        walk_index(client, FMT, tmp_path)

        assert len(read_index(index_path, FMT)) == 4
        tail = index_path.read_text().splitlines()[2:]
        assert all(json.loads(line)["id"] for line in tail)

    def test_rows_of_another_format_are_not_indexed(self, tmp_path):
        feed = FakeFeed(spaced_rows(3))
        feed.rows.append({"id": "garbage", "uploadtime": "x"})
        client, _, _ = make_client(feed)
        walk_index(client, FMT, tmp_path)
        assert len(index_line_ids(tmp_path)) == 3
        assert read_index(index_paths(tmp_path, FMT)[0], BO3) == []

    def test_a_feed_that_ignores_before_stops_the_walk(self, tmp_path):
        feed = FakeFeed(spaced_rows(120))
        feed.ignore_before = True
        client, _, _ = make_client(feed, max_requests=20)
        with pytest.raises(StopRun) as stop:
            walk_index(client, FMT, tmp_path)
        assert stop.value.exit_code == 2 and len(feed.urls) == 2
        assert "before=" in stop.value.reason

    def test_a_search_answer_that_is_not_a_list_stops_the_walk(self, tmp_path):
        feed = FakeFeed([])
        feed.script.append(HttpResponse(200, b'{"actionerror": "no"}'))
        client, _, _ = make_client(feed)
        with pytest.raises(StopRun) as stop:
            walk_index(client, FMT, tmp_path)
        assert stop.value.exit_code == 2 and "actionerror" in stop.value.reason


# --------------------------------------------------------------------- fetch order


class TestFetchOrder:
    def test_singles_take_one_game_per_tier_in_turn_then_bo3(self):
        ratings = [1050, None, 1650, 1250, 990, 1350, 1150, 1420, 1300, 1399]
        rows = [row(i, rating=r) for i, r in enumerate(ratings)]
        rows += [row(100 + i, rating=1700, fmt=BO3) for i in range(3)]
        random.Random(3).shuffle(rows)

        order = fetch_order(rows)

        singles = [r for r in order if r.format == FMT]
        assert order[: len(singles)] == singles  # every Bo3 game comes after
        tiers = [rating_tier(r.rating) for r in singles]
        # one per tier in TIER_ORDER, then again over the tiers that still have games
        assert tiers == [
            "1400+",
            "1300s",
            "unrated",
            "1200s",
            "1100s",
            "1000s",
            "1400+",
            "1300s",
            "1000s",
            "1300s",
        ]

    def test_tier_edges(self):
        assert [rating_tier(r) for r in (1400, 1399, 1300, 1299, 1200)] == [
            "1400+",
            "1300s",
            "1300s",
            "1200s",
            "1200s",
        ]
        assert [rating_tier(r) for r in (1199, 1100, 1099, 1000, 0, None)] == [
            "1100s",
            "1100s",
            "1000s",
            "1000s",
            "unrated",
            "unrated",
        ]

    def test_order_inside_a_tier_is_a_fixed_shuffle_not_a_time_sort(self):
        rows = [row(i, when=1000 + i, rating=1210 + i % 50) for i in range(200)]
        order = [r.id for r in fetch_order(rows)]
        expected = sorted((r.id for r in rows), key=lambda i: zlib.crc32(i.encode()))
        assert order == expected
        assert order != sorted(order) and order != sorted(order, reverse=True)

    def test_order_does_not_depend_on_input_order(self):
        rows = [row(i, when=1000 + i, rating=1000 + 7 * i) for i in range(80)]
        rows += [
            row(500 + i, when=2000 + 60 * i, players=(f"u{i // 2}", "v"), fmt=BO3)
            for i in range(20)
        ]
        shuffled = list(rows)
        random.Random(11).shuffle(shuffled)
        assert fetch_order(rows) == fetch_order(shuffled)

    def test_bo3_sets_stay_together_in_game_order(self):
        pair, other = ("alice", "bob"), ("carol", "dave")
        rows = [
            # Set A, uploaded out of game order; its final game is rated 1450.
            row(101, 5_050, pair, None, BO3),
            row(102, 5_000, pair, None, BO3),
            row(103, 5_100, pair, 1450, BO3),
            # Set B: two games, final rated 1120.
            row(201, 6_000, other, None, BO3),
            row(202, 6_300, other, 1120, BO3),
            # Set C: the same pair as A, hours later, final rated 1210.
            row(301, 50_000, pair, None, BO3),
            row(302, 50_200, pair, 1210, BO3),
            # A lone unrated game.
            row(401, 7_000, ("erin", "frank"), None, BO3),
        ]
        random.Random(5).shuffle(rows)

        units = [
            [int(r.id.rsplit("-", 1)[1]) for r in unit] for unit in fetch_units(rows)
        ]

        assert units == [[101, 102, 103], [401], [301, 302], [201, 202]]

    def test_a_rated_game_closes_its_set(self):
        pair = ("alice", "bob")
        rows = [
            row(1, 1_000, pair, None, BO3),
            row(2, 1_100, pair, 1300, BO3),
            row(3, 1_160, pair, None, BO3),
            row(4, 1_300, pair, None, BO3),
            row(5, 1_400, pair, None, BO3),
            row(6, 1_500, pair, None, BO3),
        ]
        sets = sorted(
            [int(r.id.rsplit("-", 1)[1]) for r in games]
            for games in group_bo3_sets(rows)
        )
        assert sets == [[1, 2], [3, 4, 5], [6]]

    def test_names_that_differ_only_in_punctuation_are_one_account(self):
        rows = [
            row(1, 1_000, ("Al-ice 1", "BOB"), None, BO3),
            row(2, 1_100, ("alice1", "bob"), 1300, BO3),
        ]
        assert len(group_bo3_sets(rows)) == 1


# --------------------------------------------------------------------- account cap


class TestAccountCap:
    def test_only_games_with_both_players_capped_are_skipped(self):
        units = [
            [row(1, players=("heavy1", "heavy2"))],
            [row(2, players=("heavy1", "heavy2"))],
            [row(3, players=("heavy1", "heavy2"))],
            [row(4, players=("heavy1", "newcomer"))],
            [row(5, players=("heavy2", "heavy1"))],
        ]
        queue, capped = apply_account_cap(units, held=set(), cap=2)
        assert [r.id for r in queue] == [f"{FMT}-1", f"{FMT}-2", f"{FMT}-4"]
        assert [r.id for r in capped] == [f"{FMT}-3", f"{FMT}-5"]

    def test_games_already_held_count_towards_the_cap(self):
        units = [[row(i, players=("heavy1", "heavy2"))] for i in range(1, 5)]
        held = {f"{FMT}-1", f"{FMT}-2"}
        queue, capped = apply_account_cap(units, held=held, cap=2)
        assert queue == []
        assert [r.id for r in capped] == [f"{FMT}-3", f"{FMT}-4"]

    def test_known_missing_ids_are_neither_queued_nor_counted(self):
        units = [[row(i, players=("heavy1", "heavy2"))] for i in range(1, 4)]
        queue, capped = apply_account_cap(units, held=set(), cap=2, skip={f"{FMT}-1"})
        assert [r.id for r in queue] == [f"{FMT}-2", f"{FMT}-3"] and capped == []

    def test_cap_of_zero_disables_the_cap(self):
        units = [[row(i, players=("heavy1", "heavy2"))] for i in range(1, 30)]
        queue, capped = apply_account_cap(units, held=set(), cap=0)
        assert len(queue) == 29 and capped == []

    def test_a_set_is_kept_or_skipped_as_a_whole(self):
        pair = ("heavy1", "heavy2")
        first = [row(i, players=pair, fmt=BO3) for i in (1, 2, 3)]
        second = [row(i, players=pair, fmt=BO3) for i in (4, 5, 6)]
        # At cap 2 the first set starts below the cap and is fetched whole, even
        # though both accounts pass the cap inside it; the next set is skipped whole.
        queue, capped = apply_account_cap([first, second], held=set(), cap=2)
        assert queue == first and capped == second

    def test_a_set_with_a_held_game_is_finished(self):
        pair = ("heavy1", "heavy2")
        filler = [[row(i, players=pair)] for i in (10, 11, 12)]
        begun = [row(i, players=pair, fmt=BO3) for i in (1, 2, 3)]
        held = {r.id for unit in filler for r in unit} | {begun[0].id}
        queue, capped = apply_account_cap(filler + [begun], held=held, cap=2)
        assert queue == begun[1:] and capped == []

    def test_a_rerun_after_a_partial_fetch_decides_the_same(self):
        rng = random.Random(42)
        names = [f"u{i}" for i in range(12)]
        rows = [
            row(i, when=1000 + i, players=rng.sample(names, 2), rating=1000 + i % 500)
            for i in range(300)
        ]
        first = plan_fetch(rows, set(), set(), set(), cap=20)
        assert first.capped and first.queue
        for done in (1, 37, 150, len(first.queue)):
            fetched = {r.id for r in first.queue[:done]}
            again = plan_fetch(rows, set(), fetched, set(), cap=20)
            assert again.queue == first.queue[done:]
            assert {r.id for r in again.capped} == {r.id for r in first.capped}


# ----------------------------------------------------------------- on-disk skip set


class TestSkipSet:
    def test_ids_come_from_json_keys_and_log_file_names(self, tmp_path):
        merged = tmp_path / "merged"
        merged.mkdir()
        (merged / f"logs_{FMT}.json").write_text(
            json.dumps({f"{FMT}-1": [10, "log"], f"{FMT}-2": [11, "log"]})
        )
        (merged / f"logs_{BO3}.json").write_text(json.dumps({f"{BO3}-3": [12, "l"]}))
        (merged / "logs_broken.json").write_text("{not json")
        web = tmp_path / "web"
        (web / "top").mkdir(parents=True)
        (web / "low").mkdir()
        (web / "top" / f"{FMT}-4.log").write_text("x")
        (web / "low" / f"{FMT}-5.log").write_text("x")
        (web / f"{FMT}-6.log").write_text("x")

        ids, notes = ondisk_ids([merged, web, tmp_path / "absent"])

        assert ids == {f"{FMT}-{n}" for n in (1, 2, 4, 5, 6)} | {f"{BO3}-3"}
        assert any("absent" in note for note in notes)
        assert any("unreadable" in note for note in notes)

    def test_fetch_requests_only_what_is_not_held(self, tmp_path):
        out = tmp_path / "out"
        feed = FakeFeed(spaced_rows(7))
        client, _, _ = make_client(feed)
        walk_index(client, FMT, out)
        have = tmp_path / "have"
        (have / "top").mkdir(parents=True)
        (have / f"logs_{FMT}.json").write_text(
            json.dumps({f"{FMT}-1": [1, "l"], f"{FMT}-2": [2, "l"]})
        )
        (have / "top" / f"{FMT}-3.log").write_text("x")
        (out / "fetched_ids.txt").write_text(f"{FMT}-4\n")
        (out / "missing_ids.txt").write_text(f"{FMT}-5\t404\n")
        feed.urls.clear()

        report = fetch_logs(client, out, [FMT], have_dirs=[have])

        assert sorted(feed.replay_ids()) == [f"{FMT}-6", f"{FMT}-7"]
        assert report.fetched == 2 and report.queued == 2
        plan = plan_fetch(
            read_index(index_paths(out, FMT)[0], FMT),
            ondisk_ids([have])[0],
            scrape.read_id_file(out / "fetched_ids.txt"),
            scrape.read_id_file(out / "missing_ids.txt"),
            cap=400,
        )
        assert (plan.indexed, plan.on_disk, plan.fetched, plan.missing) == (7, 3, 3, 1)
        assert plan.queue == []


# -------------------------------------------------------------- backoff and limits


class TestBackoff:
    def test_schedule(self):
        waits = [backoff_seconds(n, None) for n in range(1, 11)]
        assert waits == [2, 4, 8, 16, 300, 64, 120, 120, 120, 300]

    def test_retry_after_is_a_floor_not_a_replacement(self):
        assert backoff_seconds(1, 30.0) == 30.0
        assert backoff_seconds(3, 1.0) == 8.0
        assert backoff_seconds(5, 10.0) == 300.0

    def test_retry_after_parsing(self):
        assert parse_retry_after("7", 0.0) == 7.0
        assert parse_retry_after(" 2.5 ", 0.0) == 2.5
        assert parse_retry_after("-3", 0.0) == 0.0
        assert parse_retry_after(None, 0.0) is None
        assert parse_retry_after("soon", 0.0) is None
        stamp = "Thu, 01 Jan 2026 00:01:00 GMT"
        assert parse_retry_after(stamp, 1767225600.0) == 60.0
        assert parse_retry_after(stamp, 1767229200.0) == 0.0

    def test_429_503_and_timeouts_are_retried_with_backoff(self, tmp_path):
        feed = FakeFeed(spaced_rows(3))
        feed.script = [failing(429, "7"), failing(503), failing(0)]
        client, clock, lines = make_client(feed)

        reply = client.get_json(scrape.search_url(FMT, None))

        assert reply.kind == "ok" and len(reply.data) == 3
        assert client.requests == 4 and client.failures == 3
        assert client.consecutive_failures == 0
        # 7 s (Retry-After beats 2 s), then 4 s, then 8 s; long waits are sliced.
        assert sum(clock.sleeps) == 19.0 and max(clock.sleeps) <= scrape.PAUSE_SLICE_S
        assert sum(line.startswith("[retry]") for line in lines) == 3

    def test_five_failures_in_a_row_pause_five_minutes(self):
        feed = FakeFeed(spaced_rows(3))
        feed.script = [failing(503)] * 5
        client, clock, _ = make_client(feed)
        assert client.get_json(scrape.search_url(FMT, None)).kind == "ok"
        assert sum(clock.sleeps) == 2 + 4 + 8 + 16 + 300

    def test_twenty_failures_in_a_row_stop_with_a_clear_message(self):
        feed = FakeFeed([])
        feed.script = [failing(503)] * 40
        client, clock, _ = make_client(feed)
        with pytest.raises(StopRun) as stop:
            client.get_json(scrape.search_url(FMT, None))
        assert stop.value.exit_code == 2
        assert "20 consecutive failed requests" in stop.value.reason
        assert "resume" in stop.value.reason
        assert client.requests == 20
        assert sum(clock.sleeps) == sum(backoff_seconds(n, None) for n in range(1, 20))

    def test_a_success_resets_the_failure_run(self):
        feed = FakeFeed(spaced_rows(3))
        client, clock, _ = make_client(feed)
        for _ in range(6):
            feed.script = [failing(503)] * 4
            assert client.get_json(scrape.search_url(FMT, None)).kind == "ok"
        assert client.failures == 24 and client.consecutive_failures == 0
        assert LONG_PAUSE not in clock.sleeps

    def test_a_very_long_retry_after_stops_instead_of_waiting(self):
        feed = FakeFeed([])
        feed.script = [failing(429, "7200")]
        client, clock, _ = make_client(feed)
        with pytest.raises(StopRun) as stop:
            client.get_json(scrape.search_url(FMT, None))
        assert stop.value.exit_code == 2 and "Retry-After" in stop.value.reason
        assert clock.sleeps == []

    def test_an_unparsable_200_is_a_failure_and_a_404_is_not(self):
        feed = FakeFeed(spaced_rows(2))
        feed.script = [HttpResponse(200, b"<html>challenge</html>")]
        client, _, _ = make_client(feed)
        assert client.get_json(scrape.search_url(FMT, None)).kind == "ok"
        assert client.failures == 1
        reply = client.get_json(scrape.replay_url(f"{FMT}-999"))
        assert reply.kind == "missing" and client.failures == 1

    def test_requests_are_spaced_by_the_delay(self):
        feed = FakeFeed(spaced_rows(2))
        client, clock, _ = make_client(feed)
        url = scrape.search_url(FMT, None)
        client.get_json(url)
        client.get_json(url)
        clock.now += 0.4  # work between two requests counts towards the gap
        client.get_json(url)
        clock.now += 3.0
        client.get_json(url)
        assert clock.sleeps == pytest.approx([1.0, 0.6])

    def test_max_requests_counts_every_http_call(self):
        feed = FakeFeed([])
        feed.script = [failing(503)] * 10
        client, _, _ = make_client(feed, max_requests=3)
        with pytest.raises(StopRun) as stop:
            client.get_json(scrape.search_url(FMT, None))
        assert stop.value.exit_code == 0 and "request limit" in stop.value.reason
        assert len(feed.urls) == 3

    def test_stop_file_exits_before_the_next_request(self, tmp_path):
        stop_file = tmp_path / "STOP"
        feed = FakeFeed(spaced_rows(140))
        client, _, _ = make_client(feed, stop_file=stop_file)
        client.get_json(scrape.search_url(FMT, None))
        stop_file.write_text("")
        with pytest.raises(StopRun) as stop:
            walk_index(client, FMT, tmp_path)
        assert stop.value.exit_code == 0 and "stop file" in stop.value.reason
        assert len(feed.urls) == 1

    def test_stop_file_is_seen_during_a_long_backoff(self, tmp_path):
        stop_file = tmp_path / "STOP"
        feed = FakeFeed([])
        feed.script = [failing(503)] * 5
        clock = FakeTime()

        def sleep(seconds):
            clock.sleep(seconds)
            if clock.now > 5100:
                stop_file.write_text("")

        client = Client(
            http=feed,
            sleep=sleep,
            clock=clock.clock,
            wall=clock.clock,
            say=lambda _: None,
            stop_file=stop_file,
        )
        with pytest.raises(StopRun):
            client.get_json(scrape.search_url(FMT, None))
        assert sum(clock.sleeps) < 200  # left the five-minute pause early

    def test_heartbeat_every_hundred_requests(self):
        feed = FakeFeed(spaced_rows(2))
        client, _, lines = make_client(feed)
        client.last = f"{FMT}-2"
        for _ in range(250):
            client.get_json(scrape.search_url(FMT, None))
        beats = [line for line in lines if line.startswith("[hb]")]
        assert len(beats) == 2
        assert "requests=100" in beats[0] and "requests=200" in beats[1]
        assert (
            f"last={FMT}-2" in beats[0] and "rate=" in beats[0] and "eta=" in beats[0]
        )

    def test_delay_below_the_minimum_is_refused(self, tmp_path):
        feed = FakeFeed(spaced_rows(3))
        argv = ["index", "--out", str(tmp_path), "--delay", "0.2"]
        with pytest.raises(SystemExit) as stop:
            main(argv, http=feed, sleep=lambda _: None)
        assert stop.value.code == 2 and feed.urls == []

    def test_newer_only_and_since_cannot_be_combined(self, tmp_path):
        feed = FakeFeed(spaced_rows(3))
        argv = ["index", "--out", str(tmp_path), "--newer-only", "--since", "5"]
        with pytest.raises(SystemExit):
            main(argv, http=feed, sleep=lambda _: None)
        assert feed.urls == []


LONG_PAUSE = scrape.LONG_PAUSE_S


# ----------------------------------------------------------------------- fetching


def indexed(tmp_path, rows, fmt=FMT):
    out = tmp_path / "out"
    feed = FakeFeed(rows)
    client, _, _ = make_client(feed)
    walk_index(client, fmt, out)
    feed.urls.clear()
    return out, feed


def shard_lines(path):
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


class TestFetch:
    def test_shards_and_done_set(self, tmp_path):
        out, feed = indexed(tmp_path, spaced_rows(5))
        client, _, _ = make_client(feed)

        report = fetch_logs(client, out, [FMT], have_dirs=[], shard_size=2)

        assert report.fetched == 5 and report.reason == "queue finished"
        paths = shard_paths(out, FMT)
        assert [p.name for p in paths] == [
            "part-00000.jsonl.gz",
            "part-00001.jsonl.gz",
            "part-00002.jsonl.gz",
        ]
        assert [len(shard_lines(p)) for p in paths] == [2, 2, 1]
        record = shard_lines(paths[0])[0]
        assert set(record) == {"id", "format", "uploadtime", "rating", "players", "log"}
        assert record["format"] == FMT and record["players"] == ["alice", "bob"]
        assert "|player|p1|alice|1|1210" in record["log"]
        done = (out / "fetched_ids.txt").read_text().split()
        assert sorted(done) == sorted(r["id"] for r in feed.rows)
        assert len(list(iter_feed_logs(out, [FMT]))) == 5

    def test_resume_does_not_fetch_twice_and_refills_the_open_shard(self, tmp_path):
        out, feed = indexed(tmp_path, spaced_rows(5))
        client, _, _ = make_client(feed)
        first = fetch_logs(client, out, [FMT], have_dirs=[], max_games=3, shard_size=4)
        assert first.fetched == 3 and "game limit" in first.reason

        client, _, _ = make_client(feed)
        fetch_logs(client, out, [FMT], have_dirs=[], shard_size=4)

        assert sorted(feed.replay_ids()) == sorted(r["id"] for r in feed.rows)
        paths = shard_paths(out, FMT)
        assert [len(shard_lines(p)) for p in paths] == [4, 1]

    def test_fetch_follows_the_priority_order(self, tmp_path):
        rows = [
            feed_row(i, 9_000 - i, rating=r)
            for i, r in enumerate([1010, 1450, None, 1320, 1110, 1230])
        ]
        out, feed = indexed(tmp_path, rows)
        client, _, _ = make_client(feed)
        fetch_logs(client, out, [FMT], have_dirs=[])
        by_id = {r["id"]: r["rating"] for r in rows}
        assert [by_id[i] for i in feed.replay_ids()] == [
            1450,
            1320,
            None,
            1230,
            1110,
            1010,
        ]

    def test_a_404_is_recorded_and_never_asked_for_again(self, tmp_path):
        out, feed = indexed(tmp_path, spaced_rows(4))
        feed.gone = {f"{FMT}-2"}
        client, _, _ = make_client(feed)

        report = fetch_logs(client, out, [FMT], have_dirs=[])

        assert report.fetched == 3 and report.missing == 1
        assert (out / "missing_ids.txt").read_text() == f"{FMT}-2\t404\n"
        feed.urls.clear()
        again = fetch_logs(client, out, [FMT], have_dirs=[])
        assert feed.urls == [] and again.queued == 0

    def test_an_answer_without_a_log_is_recorded_as_missing(self, tmp_path):
        out, feed = indexed(tmp_path, spaced_rows(1))
        feed.script = [HttpResponse(200, json.dumps({"id": f"{FMT}-1"}).encode())]
        client, _, _ = make_client(feed)
        report = fetch_logs(client, out, [FMT], have_dirs=[])
        assert report.missing == 1 and report.fetched == 0
        assert "nolog" in (out / "missing_ids.txt").read_text()

    def test_a_replay_that_keeps_failing_is_left_for_the_next_run(self, tmp_path):
        out, feed = indexed(tmp_path, spaced_rows(3))
        feed.script = [failing(503)] * scrape.FETCH_ATTEMPTS
        client, _, _ = make_client(feed)

        report = fetch_logs(client, out, [FMT], have_dirs=[])

        assert report.deferred == 1 and report.fetched == 2
        first = feed.replay_ids()[0]
        assert feed.replay_ids().count(first) == scrape.FETCH_ATTEMPTS
        assert first not in (out / "fetched_ids.txt").read_text()
        assert not (out / "missing_ids.txt").exists()
        assert "deferred" in report.reason

    def test_twenty_missing_in_a_row_stop_the_run(self, tmp_path):
        out, feed = indexed(tmp_path, spaced_rows(30))
        feed.gone = {r["id"] for r in feed.rows}
        client, _, _ = make_client(feed)
        with pytest.raises(StopRun) as stop:
            fetch_logs(client, out, [FMT], have_dirs=[])
        assert stop.value.exit_code == 2 and len(feed.urls) == 20
        assert "endpoint may have changed" in stop.value.reason
        # A systemic failure must not poison the missing-set.
        assert not (out / "missing_ids.txt").exists()

    def test_a_short_run_of_missing_replays_is_recorded_on_a_stop(self, tmp_path):
        out, feed = indexed(tmp_path, spaced_rows(30))
        feed.gone = {r["id"] for r in feed.rows}
        client, _, _ = make_client(feed, max_requests=5)
        with pytest.raises(StopRun):
            fetch_logs(client, out, [FMT], have_dirs=[])
        recorded = (out / "missing_ids.txt").read_text().splitlines()
        assert len(recorded) == 5 and all(line.endswith("\t404") for line in recorded)

    def test_a_torn_shard_tail_is_cut_and_that_game_is_fetched_again(self, tmp_path):
        out, feed = indexed(tmp_path, spaced_rows(4))
        client, _, _ = make_client(feed)
        fetch_logs(client, out, [FMT], have_dirs=[], max_games=3)
        shard = shard_paths(out, FMT)[0]
        whole = shard.read_bytes()
        torn = gzip.compress(b'{"id": "half-written"}\n')[:-9]
        shard.write_bytes(whole + torn)
        assert len(read_shard(shard)[0]) == 3  # readers skip the torn tail

        client, _, lines = make_client(feed)
        fetch_logs(client, out, [FMT], have_dirs=[])

        assert any("torn tail" in line for line in lines)
        records = shard_lines(shard)
        assert sorted(r["id"] for r in records) == sorted(r["id"] for r in feed.rows)

    def test_a_game_in_the_shard_but_not_in_the_done_set_is_not_refetched(
        self, tmp_path
    ):
        out, feed = indexed(tmp_path, spaced_rows(4))
        client, _, _ = make_client(feed)
        fetch_logs(client, out, [FMT], have_dirs=[], max_games=3)
        done_path = out / "fetched_ids.txt"
        done = done_path.read_text().splitlines()
        # Killed after the shard write, in the middle of the done-set write.
        done_path.write_text("\n".join(done[:2]) + "\n" + done[2][:10])
        feed.urls.clear()

        client, _, lines = make_client(feed)
        fetch_logs(client, out, [FMT], have_dirs=[])

        assert len(feed.urls) == 1 and feed.replay_ids()[0] not in done
        assert any("restored to the done-set" in line for line in lines)
        ids = [r["id"] for r in shard_lines(shard_paths(out, FMT)[0])]
        assert len(ids) == len(set(ids)) == 4

    def test_a_large_unreadable_tail_is_left_alone(self, tmp_path, monkeypatch):
        monkeypatch.setattr(scrape, "TORN_TAIL_MAX_BYTES", 8)
        out, feed = indexed(tmp_path, spaced_rows(2))
        client, _, _ = make_client(feed)
        fetch_logs(client, out, [FMT], have_dirs=[], max_games=1)
        shard = shard_paths(out, FMT)[0]
        damaged = shard.read_bytes() + b"x" * 64
        shard.write_bytes(damaged)

        store = FeedStore(out)
        notes = store.open_format(FMT)
        store.add({"id": f"{FMT}-2", "format": FMT, "log": "|turn|1"})
        store.close()

        assert any("left untouched" in note for note in notes)
        assert shard.read_bytes() == damaged
        assert [p.name for p in shard_paths(out, FMT)][-1] == "part-00001.jsonl.gz"

    def test_account_cap_reaches_the_fetch(self, tmp_path):
        rows = [feed_row(i, 9_000 - i, ("heavy1", "heavy2")) for i in range(6)]
        rows += [feed_row(10, 8_000, ("heavy1", "newcomer"))]
        out, feed = indexed(tmp_path, rows)
        client, _, lines = make_client(feed)
        report = fetch_logs(client, out, [FMT], have_dirs=[], account_cap=2)
        assert report.capped == 4 and f"{FMT}-10" in feed.replay_ids()
        assert report.fetched == 3
        assert any(line.startswith("[plan]") for line in lines)


# -------------------------------------------------------------------- stats and CLI


class TestStatsAndCli:
    def test_stats_totals(self, tmp_path):
        rows = [
            feed_row(i, 1_791_000_000 + 3_600 * i, ("heavy", f"p{i}"), r)
            for i, r in enumerate([1450, 1320, None, 1230, 1110, 1010, 1620, 1510])
        ]
        out, feed = indexed(tmp_path, rows)
        client, _, _ = make_client(feed)
        fetch_logs(client, out, [FMT], have_dirs=[], max_games=3)

        text = stats_text(out, [FMT], have_dirs=[], account_cap=400, delay=1.0)

        assert f"{FMT}: 8 rows" in text and "total rows: 8" in text
        assert re.search(r"1400\+\s+3 \(37\.5%\)", text)
        assert re.search(r"unrated\s+1 \(12\.5%\)", text)
        assert re.search(r"of which 1500\+\s+2", text)
        assert re.search(r"1\. heavy\s+8\s+100\.0% of rows", text)
        assert "2026-10-03" in text
        assert "fetched into this directory: 3 (1 shards)" in text
        assert "remaining to fetch: 5" in text
        assert "estimated time for the rest: 0m at 1.20 s per request" in text

    def test_stats_on_an_empty_directory_does_not_fail(self, tmp_path):
        text = stats_text(tmp_path / "nothing", [FMT, BO3], have_dirs=[])
        assert "total rows: 0" in text and "remaining to fetch: 0" in text

    def test_cli_index_fetch_stats(self, tmp_path, capsys):
        out = tmp_path / "out"
        rows = spaced_rows(120) + spaced_rows(8, fmt=BO3, first_number=900)
        feed = FakeFeed(rows)
        common = ["--out", str(out)]
        absent = ["--have-dir", str(tmp_path / "absent")]

        def quiet(_):
            return None

        assert (
            main(["index", *common, "--max-requests", "2"], http=feed, sleep=quiet) == 0
        )
        assert "request limit" in capsys.readouterr().out
        assert len(index_line_ids(out)) == 101
        assert main(["index", *common], http=feed, sleep=quiet) == 0
        assert len(index_line_ids(out)) == 120 and len(index_line_ids(out, BO3)) == 8
        argv = ["fetch", *common, *absent, "--max-games", "4", "--formats", FMT]
        assert main(argv, http=feed, sleep=quiet) == 0
        assert "game limit reached (4)" in capsys.readouterr().out
        assert len(list(iter_feed_logs(out))) == 4
        assert main(["stats", *common, *absent], http=feed) == 0
        printed = capsys.readouterr().out
        assert "total rows: 128" in printed and "remaining to fetch: 124" in printed

    def test_cli_returns_2_when_the_server_keeps_failing(self, tmp_path, capsys):
        feed = FakeFeed([])
        feed.script = [failing(503)] * 40
        argv = ["index", "--out", str(tmp_path), "--formats", FMT]
        assert main(argv, http=feed, sleep=lambda _: None) == 2
        assert "20 consecutive failed requests" in capsys.readouterr().out

    def test_a_second_run_on_the_same_directory_is_refused(self, tmp_path, capsys):
        feed = FakeFeed(spaced_rows(3))
        held = scrape.acquire_lock(tmp_path)
        assert held is not None
        try:
            argv = ["index", "--out", str(tmp_path), "--formats", FMT]
            assert main(argv, http=feed, sleep=lambda _: None) == 3
            assert feed.urls == [] and "another oppmodel_scrape run" in (
                capsys.readouterr().out
            )
            absent = ["--have-dir", str(tmp_path / "absent")]
            assert main(["stats", "--out", str(tmp_path), *absent], http=feed) == 0
        finally:
            held.close()
        assert main(argv, http=feed, sleep=lambda _: None) == 0
        assert len(index_line_ids(tmp_path)) == 3

    def test_default_out_dir_is_git_ignored_by_pattern(self):
        ignore = (scrape.REPO_ROOT / ".gitignore").read_text().split()
        assert scrape.DEFAULT_OUT.name.startswith("battle_logs")
        assert "battle_logs*/" in ignore


# ------------------------------------------------ against the corpora on this disk


class TestOnDiskCorpora:
    def test_default_corpora_hold_the_known_ids(self):
        if not all(d.is_dir() for d in scrape.DEFAULT_HAVE_DIRS):
            pytest.skip("on-disk corpora absent (git-ignored)")
        ids, notes = ondisk_ids(scrape.DEFAULT_HAVE_DIRS)
        assert not any("unreadable" in n or "absent" in n for n in notes)
        singles = [i for i in ids if i.startswith(FMT + "-")]
        bo3 = [i for i in ids if i.startswith(BO3 + "-")]
        assert len(singles) >= 4_000 and len(bo3) >= 4_000
        assert len(singles) + len(bo3) == len(ids)

    def test_bo3_grouping_recovers_the_sets_marked_in_the_logs(self):
        path = scrape.DEFAULT_HAVE_DIRS[0] / f"logs_{BO3}.json"
        if not path.exists():
            pytest.skip("on-disk Bo3 corpus absent (git-ignored)")
        marker = re.compile(r"/game-bestof3-[a-z0-9]+-(\d+)\"")
        player = re.compile(r"^\|player\|(p[12])\|([^|\n]+)\|", re.M)
        change = re.compile(r"rating: \d+ &rarr; <strong>(\d+)</strong>")
        rows, truth = [], defaultdict(set)
        for replay_id, (when, log) in json.loads(path.read_text()).items():
            found = marker.search(log[:4000])
            names = {}
            for match in player.finditer(log[:4000]):
                names.setdefault(match.group(1), match.group(2))
            if not found or len(names) != 2:
                continue
            after = [int(x) for x in change.findall(log)]
            rating = min(after) if after else None
            rows.append(IndexRow(replay_id, when, tuple(names.values()), rating, BO3))
            truth[found.group(1)].add(replay_id)

        guessed = {frozenset(r.id for r in games) for games in group_bo3_sets(rows)}

        exact = sum(1 for games in truth.values() if frozenset(games) in guessed)
        assert len(truth) > 1_000
        assert exact / len(truth) >= 0.97


def test_paths_are_plain_files_under_the_output_directory(tmp_path):
    index_path, state_path = index_paths(Path(tmp_path), FMT)
    assert index_path.name == f"index_{FMT}.jsonl"
    assert state_path.name == f"index_{FMT}.state.json"
