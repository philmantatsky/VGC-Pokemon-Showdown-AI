"""Opponent predictor dataset builder: sources, drops, splits, weights, shards.

Everything runs on a tiny synthetic corpus written under ``tmp_path`` with the
real folder layout. Some tests read the repo's own corpus folders and skip when
those are absent (they are git-ignored). The last section is the
``--human-fraction`` subsample of the learning curve.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from datagen import oppmodel_build_dataset as B
from vgc_bench.src.oppmodel import events as E
from vgc_bench.src.oppmodel import features as F
from vgc_bench.src.oppmodel import public_state as P

ROOT = Path(__file__).resolve().parents[1]
FMT = "gen9championsvgc2026regmc"
BO3 = FMT + "bo3"
BOT = "Test Bot"  # the account the saved pages belong to
ROSTER_1 = ("Incineroar", "Rillaboom", "Charizard", "Farigiraf", "Torkoal", "Venusaur")
ROSTER_2 = ("Garchomp", "Sneasler", "Politoed", "Archaludon", "Indeedee-F", "Pawmot")
SHEETS = (
    "|showteam|p1|Incineroar||SitrusBerry|Intimidate|FakeOut,FlareBlitz,PartingShot,"
    "Protect|Adamant||M|||50|]Rillaboom||Miracle Seed|GrassySurge|WoodHammer,"
    "GrassyGlide,Protect,FakeOut|Adamant||F|||50|\n"
    "|showteam|p2|Garchomp||Garchompite|RoughSkin|Earthquake,DragonClaw,Protect,"
    "RockSlide|Jolly||M|||50|]Sneasler||FocusSash|Unburden|CloseCombat,DireClaw,"
    "FakeOut,Protect|Jolly||M|||50|\n"
)


def account(code: int, taken: set[str], bucket: int | None = None) -> str:
    """A display name whose account id falls in the wanted split."""
    for number in range(10_000):
        name = f"Player {number}"
        uid = E.user_id(name)
        if name in taken or B.player_split(uid) != code:
            continue
        if bucket is not None and B.account_bucket(uid) != bucket:
            continue
        taken.add(name)
        return name
    raise AssertionError("no such name")


def replay_id(number: int, fmt: str = FMT) -> str:
    return f"{fmt}-{number}"


def clone_number(inside: bool, start: int, fmt: str = BO3) -> int:
    """A replay number whose id is (not) in the clones' buckets 5-9."""
    for number in range(start, start + 1000):
        if (B.clone_bucket(replay_id(number, fmt)) in B.CLONE_BUCKETS) == inside:
            return number
    raise AssertionError("no such id")


def make_log(
    p1: str,
    p2: str,
    *,
    ratings: tuple[int | None, int | None] = (1300, 1250),
    rated: bool = True,
    bo3: tuple[str, int] | None = None,
    sheets: bool = False,
    when: int | None = None,
    turns: bool = True,
    extra_species: str | None = None,
    exact: str | None = None,
    p2_move: str = "Dragon Claw",
) -> str:
    """A two-turn doubles log. ``exact`` writes that side's HP as the player sees it."""

    def hp(side: str, percent: int) -> str:
        if percent == 0:
            return "0 fnt"
        if side != exact:
            return f"{percent}/100"
        # An exact value whose true fraction is NOT the public percent, so an
        # example can tell whether the rewrite to public HP happened.
        return f"{-(-percent * 187 // 100)}/187"

    r1 = "" if ratings[0] is None else str(ratings[0])
    r2 = "" if ratings[1] is None else str(ratings[1])
    title = "[Gen 9 Champions] VGC 2026 Reg M-C" + (" (Bo3)" if bo3 else "")
    lines = ["|j|☆" + p1, "|j|☆" + p2]
    if when is not None:
        lines.append(f"|t:|{when}")
    if bo3 is not None:
        lines.append(
            f"|uhtml|bestof|<h2><strong>Game {bo3[1]}</strong> of "
            f'<a href="/game-bestof3-{BO3}-{bo3[0]}">a best-of-3</a></h2>'
        )
    lines += [
        "|gametype|doubles",
        f"|player|p1|{p1}|1|{r1}",
        f"|player|p2|{p2}|2|{r2}",
        "|gen|9",
        f"|tier|{title}",
    ]
    if rated:
        lines.append("|rated|")
    lines.append("|clearpoke")
    roster_1: list[str] = list(ROSTER_1)
    if extra_species:
        roster_1[5] = extra_species
    lines += [f"|poke|p1|{name}, L50, M|" for name in roster_1]
    lines += [f"|poke|p2|{name}, L50, M|" for name in ROSTER_2]
    lines.append("|teampreview|4")
    if sheets:
        lines += SHEETS.strip().split("\n")
    lines += ["|teamsize|p1|4", "|teamsize|p2|4", "|start"]
    lines += [
        f"|switch|p1a: Incineroar|Incineroar, L50, M|{hp('p1', 100)}",
        f"|switch|p1b: Rillaboom|Rillaboom, L50, M|{hp('p1', 100)}",
        f"|switch|p2a: Garchomp|Garchomp, L50, M|{hp('p2', 100)}",
        f"|switch|p2b: Sneasler|Sneasler, L50, M|{hp('p2', 100)}",
    ]
    if turns:
        lines += [
            "|turn|1",
            "|move|p1a: Incineroar|Fake Out|p2b: Sneasler",
            f"|-damage|p2b: Sneasler|{hp('p2', 88)}",
            "|cant|p2b: Sneasler|flinch",
            "|move|p1b: Rillaboom|Protect|p1b: Rillaboom",
            "|-singleturn|p1b: Rillaboom|Protect",
            f"|move|p2a: Garchomp|{p2_move}|p1a: Incineroar",
            f"|-damage|p1a: Incineroar|{hp('p1', 47)}",
            "|upkeep",
            "|turn|2",
            "|move|p2b: Sneasler|Close Combat|p1b: Rillaboom",
            f"|-damage|p1b: Rillaboom|{hp('p1', 20)}",
            "|move|p1a: Incineroar|Flare Blitz|p2a: Garchomp",
            f"|-damage|p2a: Garchomp|{hp('p2', 60)}",
            "|move|p2a: Garchomp|Earthquake|p1a: Incineroar|[spread] p1a,p1b",
            f"|-damage|p1a: Incineroar|{hp('p1', 0)}",
            "|faint|p1a: Incineroar",
            f"|-damage|p1b: Rillaboom|{hp('p1', 0)}",
            "|faint|p1b: Rillaboom",
            "|upkeep",
        ]
    if when is not None:
        lines.append(f"|t:|{when + 300}")
    lines += [f"|win|{p2}", ""]
    return "\n".join(lines)


def page(log: str) -> str:
    return (
        "<html><body>\n"
        '<script type="text/plain" class="battle-log-data">' + log + "</script>\n"
        "</body></html>\n"
    )


class Corpus:
    """A synthetic corpus on disk and the names used to make it."""

    def __init__(self, root: Path) -> None:
        self.root = root
        taken: set[str] = set()
        self.train_a = account(B.SPLIT_TRAIN, taken)
        self.train_b = account(B.SPLIT_TRAIN, taken)
        self.train_d = account(B.SPLIT_TRAIN, taken)
        self.test_c = account(B.SPLIT_TEST, taken)
        self.val_e = account(B.SPLIT_VAL, taken)
        self.opponent = account(B.SPLIT_TRAIN, taken)  # meets the bot on ladder
        self.stranger = account(B.SPLIT_TRAIN, taken)  # meets the bot unrated
        self.in_clone = clone_number(True, 100)
        self.out_clone = clone_number(False, 200)
        a, b, c = self.train_a, self.train_b, self.test_c
        merged = {
            replay_id(1): [1000, make_log(a, b)],
            replay_id(2): [1100, make_log(a, c, p2_move="Stomping Tantrum")],
            replay_id(3): [1200, make_log(a, b, turns=False)],
            replay_id(4): [1300, make_log(a, b, extra_species="Zoroark")],
            replay_id(5): [1400, make_log(a, BOT)],
            replay_id(6): [1500, make_log(a, self.opponent, p2_move="Outrage")],
        }
        series = {
            replay_id(self.in_clone, BO3): [
                1600,
                make_log(self.train_d, self.val_e, bo3=("77", 1), sheets=True),
            ],
            replay_id(self.out_clone, BO3): [
                9000,
                make_log(self.train_d, self.val_e, bo3=("77", 2), sheets=True),
            ],
        }
        self.write_json(B.TOP_MERGED_DIR, FMT, merged)
        self.write_json(B.TOP_MERGED_DIR, BO3, series)
        web = root / B.WEB_DIR
        (web / "top").mkdir(parents=True)
        (web / "low").mkdir(parents=True)
        (web / "top" / f"{replay_id(1)}.log").write_text(merged[replay_id(1)][1])
        (web / "top" / f"{replay_id(20)}.log").write_text(make_log(b, a, when=9500))
        (web / "low" / f"{replay_id(21)}.log").write_text(
            make_log(c, b, when=1700, ratings=(None, None), rated=False)
        )
        shard = root / B.FEED_DIR / "logs" / FMT / "part-00000.jsonl.gz"
        shard.parent.mkdir(parents=True)
        records = [
            {"id": replay_id(30), "format": FMT, "uploadtime": 1800, "rating": 1200,
             "players": [a, b], "log": make_log(a, b)},
            {"id": replay_id(2), "format": FMT, "uploadtime": 1100, "rating": 1200,
             "players": [a, c], "log": merged[replay_id(2)][1]},
        ]  # fmt: skip
        whole = b"".join(
            gzip.compress((json.dumps(record) + "\n").encode(), mtime=0)
            for record in records
        )
        torn = gzip.compress(b'{"id": "never-finished"}\n', mtime=0)[:-9]
        shard.write_bytes(whole + torn)
        self.feed_bytes = len(whole)
        own = root / "ladder_replays_mc_synthetic"
        own.mkdir()
        # The bot is p2 in its ladder game and p1 in the unrated one.
        (own / f"{BOT} - battle-{replay_id(900)}.html").write_text(
            page(make_log(self.opponent, BOT, exact="p2", ratings=(1180, 1210)))
        )
        (own / f"{BOT} - battle-{replay_id(901)}-abcdef0123pw.html").write_text(
            page(make_log(BOT, self.stranger, exact="p1", rated=False))
        )
        (own / "notes.html").write_text("<html></html>")
        audit = [
            {"battle": f"battle-{replay_id(900)}", "turn": 0,
             "preview_shadow": {"open_sheet": False}},
            {"battle": f"battle-{replay_id(900)}", "turn": 1, "candidates": []},
            "not json",
        ]  # fmt: skip
        (own / "decisions.jsonl").write_text(
            "\n".join(r if isinstance(r, str) else json.dumps(r) for r in audit)
        )

    def write_json(self, folder: str, fmt: str, logs: dict[str, Any]) -> None:
        path = self.root / folder / f"logs_{fmt}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(logs))

    def build(self, tag: str = "t", **options: Any) -> dict[str, Any]:
        options.setdefault("log", lambda message: None)
        return B.build(tag, root=self.root, **options)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Corpus:
    return Corpus(tmp_path_factory.mktemp("corpus"))


@pytest.fixture(scope="module")
def built(corpus: Corpus) -> dict[str, Any]:
    return corpus.build("t")


def battles(corpus: Corpus, tag: str = "t") -> dict[str, dict[str, Any]]:
    path = corpus.root / "results_oppmodel" / tag / "battles.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["index"] for row in rows] == list(range(len(rows)))
    return {row["id"]: row for row in rows}


def dataset(corpus: Corpus, tag: str = "t") -> F.Batch:
    return F.load_dataset(corpus.root / "results_oppmodel" / tag)[0]


# --- pure rules ---------------------------------------------------------------


def test_player_split_is_a_fixed_function_of_the_account_id():
    assert [B.account_bucket(name) for name in ("alice", "bob", "carol")] == [
        zlib.crc32(name.encode()) % 10 for name in ("alice", "bob", "carol")
    ]
    assert B.account_bucket("alice") == 5 and B.account_bucket("bob") == 4
    assert B.player_split("alice") == B.SPLIT_TRAIN
    codes = {
        B.account_bucket(f"account{n}"): B.player_split(f"account{n}")
        for n in range(200)
    }
    assert codes == {
        0: B.SPLIT_TEST,
        1: B.SPLIT_VAL,
        **{b: B.SPLIT_TRAIN for b in range(2, 10)},
    }
    # One account, however its name is written.
    assert B.player_split(E.user_id("Some_Name 7")) == B.player_split(
        E.user_id("somename7")
    )
    assert B.account_hash("alice") == zlib.crc32(b"alice")
    assert B.SPLITS[B.SPLIT_LADDER] == "ladder_holdout"


def test_clone_bucket_is_the_function_the_clones_were_split_by():
    tag = "gen9championsvgc2026regmcbo3-2679348520"
    assert B.clone_bucket(tag) == zlib.crc32(tag.encode()) % 10
    assert B.CLONE_BUCKETS == {5, 6, 7, 8, 9}


def test_account_cap_weights():
    weights = B.cap_weights({"heavy": 240, "edge": 60, "light": 3})
    assert weights == {"heavy": 0.25, "edge": 1.0, "light": 1.0}
    assert sum(weights[name] * n for name, n in (("heavy", 240),)) == 60
    assert B.cap_weights({"a": 4}, cap=1) == {"a": 0.25}
    assert B.ACCOUNT_CAP == 60


def test_time_slice_flags_a_series_as_a_whole():
    times: dict[str, int | None] = {f"g{n}": 100 * n for n in range(1, 21)}
    times["undated"] = None
    groups = {name: name for name in times}
    # Games 17 and 19 are one series: the cut (g19) falls between them.
    groups["g17"] = groups["g19"] = "series"
    flagged, cut = B.time_slice_groups(times, groups)
    assert cut == 1900
    assert flagged == {"series", "g20"}
    assert "undated" not in flagged
    assert B.time_slice_groups({"a": None}, {"a": "a"}) == (set(), None)
    late, _ = B.time_slice_groups({"a": 1, "b": 2}, {}, share=0.5)
    assert late == {"b"}


def test_source_names():
    assert B.parse_sources("all") == list(B.SOURCES)
    assert B.parse_sources("own, web") == ["web_top", "web_low", "own"]
    assert B.parse_sources("feed,top_merged") == ["top_merged", "feed"]
    with pytest.raises(SystemExit):
        B.parse_sources("nowhere")


# --- the build ----------------------------------------------------------------


def test_sources_duplicates_and_drops(corpus: Corpus, built: dict[str, Any]):
    counted = built["battles"]
    assert counted["read"] == {
        "top_merged": 8,
        "web_top": 2,
        "web_low": 1,
        "feed": 2,
        "own": 2,
    }
    # The first occurrence of an id wins: web/top and the feed repeat two.
    assert counted["duplicate_ids"] == {"web_top": 1, "feed": 1}
    assert counted["dropped"] == {
        P.SKIP_NO_TURN: 1,
        P.SKIP_ILLUSION: 1,
        "bot_account": 1,
    }
    assert counted["dropped_by_source"] == {"top_merged": counted["dropped"]}
    assert counted["dropped_examples"]["bot_account"] == [replay_id(5)]
    rows = battles(corpus)
    assert set(rows) == {
        replay_id(1),
        replay_id(2),
        replay_id(6),
        replay_id(corpus.in_clone, BO3),
        replay_id(corpus.out_clone, BO3),
        replay_id(20),
        replay_id(21),
        replay_id(30),
        replay_id(900),
        replay_id(901),
    }
    assert rows[replay_id(1)]["source"] == "top_merged"
    assert rows[replay_id(30)]["source"] == "feed"
    assert counted["kept_by_format"] == {FMT: 8, BO3: 2}
    assert built["notes"] == {f"missing:{B.T6LIKE_DIR}/logs_{FMT}.json": 1,
                              f"missing:{B.T6LIKE_DIR}/logs_{BO3}.json": 1}  # fmt: skip
    headline = built["headline"]
    assert headline["human_logs_unique"] == 11 and headline["human_battles_kept"] == 8
    assert headline["own_games_kept"] == 2 and headline["ladder_holdout_games"] == 1
    assert headline["ladder_holdout_opponents"] == 1
    assert headline["ladder_opponents_in_human_corpus"] == 1
    # Every slot-turn of every unique log, dropped ones included.
    assert headline["human_all_logs"]["slot_turns"] == 10 * 8
    assert headline["own_opponents_all_logs"]["slot_turns"] == 2 * 4


def test_the_feed_is_read_up_to_its_last_whole_record(
    corpus: Corpus, built: dict[str, Any]
):
    key = f"{B.FEED_DIR}/logs/{FMT}/part-00000.jsonl.gz"
    entry = built["inputs"][key]
    assert entry["bytes"] == corpus.feed_bytes and entry["n"] == 2
    data = (corpus.root / key).read_bytes()
    assert len(data) > corpus.feed_bytes
    assert entry["sha256"] == B.sha256_bytes(data[: corpus.feed_bytes])


def test_inputs_are_hashed(corpus: Corpus, built: dict[str, Any]):
    inputs = built["inputs"]
    merged = f"{B.TOP_MERGED_DIR}/logs_{FMT}.json"
    assert inputs[merged]["sha256"] == B.sha256_file(corpus.root / merged)
    assert inputs[merged]["n"] == 6
    top = inputs[f"{B.WEB_DIR}/top"]
    assert top == {
        "kind": "folder",
        "n": 2,
        "sha256": B.sha256_ids([replay_id(20), replay_id(1)]),
    }
    own = inputs["ladder_replays_mc_synthetic"]
    assert own["sha256"] == B.sha256_ids([replay_id(900), replay_id(901)])
    assert (
        inputs["ladder_replays_mc_synthetic/decisions.jsonl"]["kind"]
        == "decision_audit"
    )
    assert all(len(entry["sha256"]) == 64 for entry in inputs.values())


def test_splits(corpus: Corpus, built: dict[str, Any]):
    rows = battles(corpus)
    assert rows[replay_id(1)]["split"] == {"p1": "train", "p2": "train"}
    assert rows[replay_id(2)]["split"] == {"p1": "train", "p2": "test"}
    assert rows[replay_id(21)]["split"] == {"p1": "test", "p2": "train"}
    # A battle with a ladder-holdout opponent is kept, tagged, never trained on.
    held = rows[replay_id(6)]
    assert held["holdout_battle"]
    assert held["split"] == {"p1": "excluded_holdout_player"} | {
        "p2": "excluded_holdout_player"
    }
    # Clone hygiene: open sheets, top_merged, bucket 5-9. The validation player
    # of that game stays a validation player.
    seen = rows[replay_id(corpus.in_clone, BO3)]
    unseen = rows[replay_id(corpus.out_clone, BO3)]
    assert seen["clone_bucket"] and not unseen["clone_bucket"]
    assert seen["split"] == {"p1": "excluded_clone", "p2": "val"}
    assert unseen["split"] == {"p1": "train", "p2": "val"}
    assert seen["sheets"] == {"p1": True, "p2": True}
    assert built["battles"]["clone_bucket_battles"] == 1
    assert built["battles"]["holdout_player_battles"] == 1
    assert built["accounts"]["series_straddling_player_split"] == 0
    by_split = built["examples"]["by_split"]
    assert set(by_split) <= set(B.SPLITS)
    assert by_split["ladder_holdout"] == 2 and by_split["own_unrated"] == 2
    assert by_split["excluded_clone"] == 2 and by_split["excluded_holdout_player"] == 4
    assert sum(by_split.values()) == built["headline"]["examples"] == 2 * 2 * 8 + 4


def test_split_assignment_is_deterministic(corpus: Corpus, built: dict[str, Any]):
    again = corpus.build("again")
    first, second = dataset(corpus, "t"), dataset(corpus, "again")
    assert set(first) == set(second)
    for name in first:
        assert np.array_equal(first[name], second[name]), name
    assert battles(corpus, "t") == battles(corpus, "again")
    for key in ("examples", "battles", "accounts", "other_rate", "censored", "inputs"):
        assert again[key] == built[key], key
    root = corpus.root / "results_oppmodel"
    assert (root / "t" / "repertoire.json").read_text() == (
        root / "again" / "repertoire.json"
    ).read_text()


def test_the_bots_side_is_never_an_example(corpus: Corpus, built: dict[str, Any]):
    data = dataset(corpus)
    rows = battles(corpus)
    games = ((900, "p2", "ladder_holdout", "ladder"), (901, "p1", "own_unrated", None))
    for number, bot_side, split, rated in games:
        row = rows[replay_id(number)]
        assert row["own"] and row["bot_side"] == bot_side and row["rated_kind"] == rated
        other = "p1" if bot_side == "p2" else "p2"
        assert row["split"] == {other: split} and row["weight"] == {other: 1.0}
        mine = data["m_battle"] == row["index"]
        assert mine.sum() == 2  # two turns, one actor
        assert set(data["m_side"][mine].tolist()) == {E.SIDES.index(other)}
        assert set(data["m_split"][mine].tolist()) == {B.SPLITS.index(split)}
        assert set(data["m_source"][mine].tolist()) == {B.SOURCES.index("own")}
    bot_hash = B.account_hash(E.user_id(BOT))
    assert bot_hash not in set(data["m_actor"].tolist())
    assert B.account_hash(E.user_id(corpus.opponent)) in set(data["m_actor"].tolist())
    # No battle of the human corpus with the bot's account got in either.
    assert replay_id(5) not in rows
    human = [row for row in rows.values() if not row["own"]]
    assert all(E.user_id(BOT) not in row["accounts"].values() for row in human)


def test_own_pages_are_encoded_as_the_runtime_sees_them(
    corpus: Corpus, built: dict[str, Any]
):
    data = dataset(corpus)
    rows = battles(corpus)
    ladder = rows[replay_id(900)]
    turn_2 = (data["m_battle"] == ladder["index"]) & (data["m_turn"] == 2)
    assert turn_2.sum() == 1
    row = int(np.argmax(turn_2))
    # The actor is p1 (the opponent); the bot's Garchomp is mon row 6. Its page
    # says 187/187 and Sneasler 165/187 (0.882): the features hold the public
    # percent the opponent sees.
    assert data["act_mon"][row].tolist() == [0, 1]
    assert float(data["mon_hp"][row, 6]) == 1.0
    assert float(data["mon_hp"][row, 7]) == pytest.approx(0.88, abs=1e-3)
    assert float(data["mon_hp"][row, 0]) == pytest.approx(0.47, abs=1e-3)
    assert data["elo"][row].tolist() == [1180, 1210]
    # The audit recorded closed sheets for 900 and nothing for 901.
    assert ladder["sheet_audit"] == "closed" and ladder["sheet_evidence"] == "audit"
    assert ladder["sheets"] == {"p1": False, "p2": False}
    assert data["game_flag"][row].tolist() == [0, 1, F.SHEET_CLOSED, F.SHEET_CLOSED]
    assert data["m_sheet"][row] == B.SHEET_STATES.index("closed")
    other = rows[replay_id(901)]
    assert other["sheet_audit"] == "unknown" and other["sheets"] == {
        "p1": None,
        "p2": None,
    }
    unknown = data["m_battle"] == other["index"]
    assert set(data["game_flag"][unknown, F.G_ACTOR_SHEET].tolist()) == {
        F.SHEET_UNKNOWN
    }
    assert set(data["m_sheet"][unknown].tolist()) == {B.SHEET_STATES.index("unknown")}
    assert built["battles"]["own_sheet_audit"] == {"closed": 1, "unknown": 1}
    assert ladder["time_source"] == "mtime" and ladder["time"] > 1_600_000_000
    assert built["examples"]["by_split_actor_sheet"]["ladder_holdout"] == {"closed": 2}


def test_open_sheet_of_an_own_game_comes_from_the_audit_or_a_public_copy(
    tmp_path: Path,
):
    own = tmp_path / "challenge_replays_mc_x"
    own.mkdir()
    for number in (910, 911, 912):
        (own / f"{BOT} - battle-{replay_id(number)}.html").write_text(
            page(make_log("Someone", BOT, exact="p2"))
        )
    their = {
        "incineroar": {
            "ability": "intimidate",
            "item": "sitrusberry",
            "moves": ["fakeout", "flareblitz"],
        }
    }
    audit = [
        {"battle": f"battle-{replay_id(910)}", "preview_shadow": {"open_sheet": True}},
        {
            "battle": f"battle-{replay_id(911)}",
            "preview_shadow": {"open_sheet": True},
            "their_sheet": their,
        },
        {"battle": f"battle-{replay_id(912)}", "preview_shadow": {"open_sheet": False}},
    ]
    (own / "decisions.jsonl").write_text("\n".join(json.dumps(r) for r in audit))
    cache = tmp_path / B.PUBLIC_CACHE
    cache.mkdir(parents=True)
    public = make_log("Someone", BOT, sheets=True)
    (cache / f"{replay_id(912)}.json").write_text(
        json.dumps({"id": replay_id(912), "log": public, "uploadtime": 4242})
    )
    reader = B.Corpus(tmp_path, [FMT], ["own"])
    raws = {raw.battle_id: raw for raw in reader}
    # Open by the audit, text nowhere: the state stays unknown to the features.
    assert raws[replay_id(910)].sheet_audit == "open"
    assert raws[replay_id(910)].sheets is None and not raws[replay_id(910)].sheets_known
    assert B.drive(raws[replay_id(910)]).sheets == {"p1": None, "p2": None}
    # Open, and the audit kept the opponent's sheet: that side is open.
    assert set(raws[replay_id(911)].sheets or {}) == {"p1"}
    assert B.drive(raws[replay_id(911)]).sheets == {"p1": True, "p2": None}
    # A public copy of the same game settles it against the audit.
    settled = raws[replay_id(912)]
    assert settled.sheet_audit == "open" and settled.sheet_evidence == "public_copy"
    assert settled.sheets_known and (settled.time, settled.time_source) == (
        4242,
        "upload",
    )
    assert B.drive(settled).sheets == {"p1": True, "p2": True}
    assert reader.notes["own:sheet_audit_contradicts_public_copy"] == 1
    assert all(raw.bot_side == "p2" and raw.own for raw in raws.values())
    assert reader.bot_accounts() == {E.user_id(BOT)}


def test_account_cap_weight_reaches_the_examples(corpus: Corpus):
    capped = corpus.build("capped", account_cap=1)
    rows = battles(corpus, "capped")
    # train_a plays five kept human battles (1, 2, 6, 20, 30), train_b four.
    assert rows[replay_id(1)]["weight"] == {"p1": 0.2, "p2": 0.25}
    assert rows[replay_id(2)]["weight"] == {"p1": 0.2, "p2": 0.5}
    assert rows[replay_id(900)]["weight"] == {"p1": 1.0}
    data = dataset(corpus, "capped")
    a_hash = B.account_hash(E.user_id(corpus.train_a))
    assert np.allclose(data["m_weight"][data["m_actor"] == a_hash], 0.2)
    assert (data["m_actor"] == a_hash).sum() == 2 * 5
    accounts = capped["accounts"]
    assert accounts["account_cap"] == 1 and accounts["accounts_over_cap"] == 5
    top = accounts["top_accounts"][0]
    assert top["account"] == E.user_id(corpus.train_a)
    assert (top["battles"], top["weight"], top["split"]) == (5, 0.2, "train")
    # The repertoire counts each use with its account's weight.
    stored = json.loads(
        (corpus.root / "results_oppmodel/capped/repertoire.json").read_text()
    )
    # train_a in 1, 2 and 30, train_b in 20, train_d in one bo3 game.
    assert stored["counts"]["incineroar"]["fakeout"] == pytest.approx(
        3 * 0.2 + 0.25 + 0.5
    )
    assert set(np.unique(dataset(corpus, "t")["m_weight"]).tolist()) == {1.0}


def test_repertoire_is_built_from_the_training_split_only(
    corpus: Corpus, built: dict[str, Any]
):
    stored = json.loads(
        (corpus.root / "results_oppmodel/t/repertoire.json").read_text()
    )
    counts = stored["counts"]
    # Garchomp was piloted by train_b (1, 30), test_c (2: its own move), the
    # holdout opponent (6: its own move), val_e (bo3), train_a (20), train_b
    # (21) and the bot or its opponents (900, 901).
    assert counts["garchomp"] == {"dragonclaw": 4.0, "earthquake": 4.0}
    assert "stompingtantrum" not in counts["garchomp"]
    assert "outrage" not in counts["garchomp"]
    # Incineroar: train_a in 1, 2 and 30, train_b in 20, train_d in the bo3
    # game outside the clone bucket. Not 6 (holdout), 21 (test), the clone
    # game or the bot's games.
    assert counts["incineroar"] == {"fakeout": 5.0, "flareblitz": 5.0}
    assert built["repertoire"]["set_keys"] == len(counts) == 4
    featurizer = F.Featurizer.load(corpus.root / "results_oppmodel/t")
    assert featurizer.repertoire.lookup("garchomp")[2] == 8.0


def test_series_share_a_group_and_the_time_slice(corpus: Corpus, built: dict[str, Any]):
    rows = battles(corpus)
    first, second = (
        rows[replay_id(corpus.in_clone, BO3)],
        rows[replay_id(corpus.out_clone, BO3)],
    )
    assert first["group"] == second["group"] == f"game-bestof3-{BO3}-77"
    assert (first["game"], second["game"]) == (1, 2)
    assert first["best_of"] == 3 and rows[replay_id(1)]["group"] == replay_id(1)
    # Eight dated human battles: the cut is the latest one (web/top 20, by its
    # own time stamp). Game 2 of the series is older than the cut, so the
    # series is out as a whole; with a lower cut it is in as a whole.
    assert built["accounts"]["time_cut"] == 9800
    assert (
        rows[replay_id(20)]["time_slice"]
        and rows[replay_id(20)]["time_source"] == "log"
    )
    assert not first["time_slice"] and not second["time_slice"]
    metas = [SimpleNamespace(**row) for row in rows.values() if not row["own"]]
    flagged, cut = B.time_slice_groups(
        {m.id: m.time for m in metas}, {m.id: m.group for m in metas}, share=0.2
    )
    assert cut == 9000 and first["group"] in flagged
    assert first["time"] == 1600 < cut  # flagged by its series, not its own time
    data = dataset(corpus)
    late = (data["m_flag"] & B.FLAG_TIME_SLICE) > 0
    assert set(data["m_battle"][late].tolist()) == {rows[replay_id(20)]["index"]}
    clone = (data["m_flag"] & B.FLAG_CLONE_BUCKET) > 0
    assert set(data["m_battle"][clone].tolist()) == {first["index"]}
    held = (data["m_flag"] & B.FLAG_HOLDOUT_BATTLE) > 0
    assert set(data["m_battle"][held].tolist()) == {rows[replay_id(6)]["index"]}
    is_holdout = (data["m_flag"] & B.FLAG_ACTOR_IS_HOLDOUT) > 0
    assert set(data["m_actor"][is_holdout].tolist()) == {
        B.account_hash(E.user_id(corpus.opponent))
    }
    in_corpus = (data["m_flag"] & B.FLAG_ACTOR_IN_CORPUS) > 0
    assert set(data["m_battle"][in_corpus].tolist()) == {rows[replay_id(900)]["index"]}
    assert set(
        data["game_flag"][data["m_battle"] == first["index"], F.G_BO3].tolist()
    ) == {1}


def test_one_shard_round_trips_and_matches_a_fresh_encode(
    corpus: Corpus, built: dict[str, Any]
):
    out = corpus.root / "results_oppmodel" / "t"
    assert built["shards"] == [{"file": "shard-00000.npz", "examples": 36}]
    shard = F.load_batch(out / "shard-00000.npz")
    featurizer = F.Featurizer.load(out)
    assert featurizer.signature_diff == []
    spec = featurizer.layout()
    for name, entry in spec.items():
        assert shard[name].shape == (36, *entry.shape), name
        assert str(shard[name].dtype) == entry.dtype, name
        assert built["arrays"][name]["shape"] == list(entry.shape)
    assert set(shard) == set(spec) | set(built["meta_arrays"])
    assert shard["m_battle"].dtype == np.int32 and shard["m_weight"].dtype == np.float32
    assert shard["m_actor"].dtype == np.uint32 and shard["m_time"].dtype == np.int64
    # Written, read back, written again: the same bytes of data.
    F.save_batch(out / "copy.npz", shard)
    copy = F.load_batch(out / "copy.npz")
    (out / "copy.npz").unlink()
    assert all(np.array_equal(copy[name], shard[name]) for name in shard)
    # The stored rows are what the stored featurizer encodes from the log.
    row = battles(corpus)[replay_id(2)]
    log = json.loads((corpus.root / B.TOP_MERGED_DIR / f"logs_{FMT}.json").read_text())
    result = P.drive_log(log[replay_id(2)][1], replay_id(2))
    for side in E.SIDES:
        for record in result.turns:
            fresh = featurizer.encode_turn(record.snapshot, record.actions, side)
            assert fresh is not None
            pick = (
                (shard["m_battle"] == row["index"])
                & (shard["m_side"] == E.SIDES.index(side))
                & (shard["m_turn"] == record.turn)
            )
            assert pick.sum() == 1
            for name, value in fresh.items():
                assert np.array_equal(shard[name][pick][0], value), (side, name)
    data, manifest = F.load_dataset(out, splits=["test", "val"])
    assert manifest["tag"] == "t" and len(data["turn"]) == 4 + 4
    assert set(data["m_split"].tolist()) == {B.SPLIT_TEST, B.SPLIT_VAL}
    scores = F.slot_nll(F.uniform_prediction(shard), shard)
    assert scores["action_scored"].sum() > 0


def test_manifest_counts(corpus: Corpus, built: dict[str, Any]):
    slots = built["slot_turns"]["by_split_source_kind"]
    assert slots["ladder_holdout"]["own"] == {"move": 3, "none": 1}
    assert slots["test"]["top_merged"] == {"hidden": 1, "move": 3}
    total = sum(
        count
        for split in slots.values()
        for source in split.values()
        for count in source.values()
    )
    assert total == 36 * 2 == built["censored"]["all"]["slot_turns"]
    assert built["censored"]["ladder_holdout"] == {
        "slot_turns": 4,
        "censored": 1,
        "censored_share": 0.25,
        "censored_with_set_loss": 1,
        "locked_moves": 0,
    }
    reasons = built["slot_turns"]["censored_reasons_by_split"]
    assert reasons["test"] == {"hidden:flinch": 1, "none:fainted_first": 1}
    # Closed-sheet moves: Stomping Tantrum (test) and Outrage (the holdout
    # opponent) are not in a repertoire that never saw them.
    other = built["other_rate"]
    assert other["test"]["closed"]["outside_candidates"] == 1
    assert other["excluded_holdout_player"]["closed"]["outside_candidates"] == 1
    assert other["train"]["closed"]["outside_candidates"] == 0
    assert other["val"]["sheet"] == {"moves": 6, "outside_candidates": 0, "rate": 0.0}
    everything = other["all"]["closed"]
    assert everything["outside_candidates"] == 2
    assert everything["rate"] == pytest.approx(2 / everything["moves"], abs=1e-5)
    histogram = built["elo_histogram_by_split"]
    assert histogram["ladder_holdout"] == {"1100": 2}
    assert histogram["test"] == {"1200": 2, "unknown": 2}
    assert sum(histogram["train"].values()) == built["examples"]["by_split"]["train"]
    assert built["dex_signature"] == E.dex_signature()
    assert built["n_cand"] == 12 and built["n_actions"] == 19
    assert built["splits"] == list(B.SPLITS) and built["sources"] == list(B.SOURCES)
    assert built["featurizer"]["encode_microseconds_per_example"] > 0
    assert built["featurizer"]["counters"] == {}
    on_disk = json.loads((corpus.root / "results_oppmodel/t/manifest.json").read_text())
    assert on_disk["headline"] == built["headline"]


def test_limit_sources_and_overwrite(corpus: Corpus, built: dict[str, Any]):
    with pytest.raises(SystemExit):
        corpus.build("t")
    small = corpus.build("small", sources=["top_merged", "own"], limit=2)
    assert small["battles"]["read"] == {"top_merged": 2, "own": 2}
    assert small["sources_requested"] == ["top_merged", "own"]
    assert small["limit"] == 2 and small["headline"]["human_battles_kept"] == 2
    only_web = corpus.build("web", sources=["web_top"], formats=[FMT])
    assert only_web["battles"]["read"] == {"web_top": 2}
    assert only_web["battles"]["kept_by_source"] == {"web_top": 2}
    rebuilt = corpus.build("small", sources=["own"], overwrite=True)
    assert rebuilt["headline"]["human_battles_kept"] == 0
    assert rebuilt["examples"]["by_split"] == {"ladder_holdout": 2, "own_unrated": 2}
    wrong = corpus.build("bo3only", formats=[BO3])
    assert wrong["battles"]["kept_by_format"] == {BO3: 2}
    assert wrong["battles"]["read"]["top_merged"] == 2


def test_drop_reasons():
    bots = {"testbot"}
    formats = [FMT]

    def reason(log: str, **raw: Any) -> str | None:
        item = B.RawBattle("x", "top_merged", log, **raw)
        return B.drop_reason(item, B.drive(item), formats, bots)

    assert reason(make_log("A", "B")) is None
    assert reason(make_log("A", "B", turns=False)) == P.SKIP_NO_TURN
    assert reason(make_log("A", "B", extra_species="Zorua")) == P.SKIP_ILLUSION
    assert reason(make_log("A", "Test Bot")) == "bot_account"
    assert reason(make_log("A", "TEST_bot")) == "bot_account"  # one account id
    assert reason(make_log("A", "B", bo3=("5", 1))) == "format"
    singles = make_log("A", "B").replace("|gametype|doubles", "|gametype|singles")
    assert reason(singles) == P.SKIP_NOT_DOUBLES
    assert (
        reason(make_log("A", "B").replace("|player|p2|B|2|1250\n", ""))
        == "players_missing"
    )
    assert reason(make_log("A", "B") + "|turn|1\n") == P.SKIP_TURN_ORDER
    assert reason(make_log("A", "Test Bot"), own=True, bot_side="p2") is None
    assert reason("") in ("format", P.SKIP_NO_TURN)


# --- the repo's own corpus (skipped when the folders are absent) --------------


def real_own_pages() -> list[tuple[Path, str, str]]:
    return B.Corpus(ROOT, B.DEFAULT_FORMATS, ["own"]).own_pages()


def test_real_own_pages_encode_like_the_live_shadow():
    pages = real_own_pages()
    if not pages:
        pytest.skip("no saved own replays (ladder_replays_mc*/) on this machine")
    featurizer = F.Featurizer.build()
    reader = B.Corpus(ROOT, B.DEFAULT_FORMATS, ["own"], limit=6)
    compared = 0
    for raw in reader:
        path = ROOT / raw.path
        log = E.extract_log_from_html(
            path.read_text(encoding="utf-8", errors="replace")
        )
        assert log is not None and raw.bot_side in E.SIDES
        actor = E.SIDES[1 - E.SIDES.index(raw.bot_side)]
        offline = {record.turn: record for record in B.drive(raw).turns}
        fake = SimpleNamespace(
            _replay_data=[], player_role=raw.bot_side, battle_tag=raw.battle_id
        )
        shadow = P.LiveShadow()
        for side, sets in (raw.sheets or {}).items():
            shadow.feed_sheet(side, sets)
        if raw.sheets_known:
            shadow.mark_sheets_known()
        for event in E.split_log(log):
            fake._replay_data.append(list(event))
            if len(event) < 3 or event[1] != "turn" or int(event[2]) not in offline:
                continue
            snapshot = shadow.sync(fake)
            assert snapshot is not None
            live = featurizer.encode(snapshot, actor)
            built = featurizer.encode(offline[int(event[2])].snapshot, actor)
            assert (live is None) == (built is None)
            if live is not None and built is not None:
                for name in built:
                    assert np.array_equal(live[name], built[name]), (
                        raw.battle_id,
                        name,
                    )
                compared += 1
    assert compared > 0


def test_real_dataset_keeps_its_contract():
    out = ROOT / "results_oppmodel" / "v1_ondisk"
    if not (out / "manifest.json").exists():
        pytest.skip("results_oppmodel/v1_ondisk has not been built on this machine")
    data, manifest = F.load_dataset(out)
    featurizer = F.Featurizer.load(out)
    assert len(data["turn"]) == manifest["headline"]["examples"]
    for name, entry in featurizer.layout().items():
        assert data[name].shape[1:] == entry.shape, name
        assert str(data[name].dtype) == entry.dtype, name
    assert set(manifest["meta_arrays"]) <= set(data)
    lines = (out / "battles.jsonl").read_text().splitlines()
    own = {row["index"]: row for row in map(json.loads, lines) if row["own"]}
    # The bot's side is never an example, and its games are never trained on.
    for index, row in own.items():
        mine = data["m_battle"] == index
        bot = E.SIDES.index(row["bot_side"])
        assert set(data["m_side"][mine].tolist()) <= {1 - bot}
        assert set(data["m_split"][mine].tolist()) <= {
            B.SPLIT_LADDER,
            B.SPLIT_OWN_UNRATED,
        }
    actors = {
        code: set(data["m_actor"][data["m_split"] == code].tolist())
        for code in range(len(B.SPLITS))
    }
    # A player is on one side of the split only.
    assert not actors[B.SPLIT_TRAIN] & actors[B.SPLIT_VAL]
    assert not actors[B.SPLIT_TRAIN] & actors[B.SPLIT_TEST]
    assert not actors[B.SPLIT_VAL] & actors[B.SPLIT_TEST]
    assert not actors[B.SPLIT_TRAIN] & actors[B.SPLIT_LADDER]
    train = data["m_split"] == B.SPLIT_TRAIN
    hygiene = B.FLAG_CLONE_BUCKET | B.FLAG_HOLDOUT_BATTLE
    assert not (data["m_flag"][train] & hygiene).any()
    assert float(data["m_weight"].max()) <= 1.0 and float(data["m_weight"].min()) > 0
    # Legal labels only: a logged action is always inside its slot's mask.
    visible = data["y_action"] >= 0
    picked = np.take_along_axis(
        data["action_mask"], np.clip(data["y_action"], 0, None)[..., None], axis=2
    )[..., 0]
    assert picked[visible].all()
    assert not (data["y_set"] & (1 - data["action_mask"])).any()


def _tiny_corpus(root: Path, feed_records: list[dict[str, Any]]) -> None:
    """One saved own ladder page (the bot is p2) and a feed shard."""
    shard = root / B.FEED_DIR / "logs" / FMT / "part-00000.jsonl.gz"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(
        b"".join(
            gzip.compress((json.dumps(record) + "\n").encode(), mtime=0)
            for record in feed_records
        )
    )
    own = root / "ladder_replays_mc_synthetic"
    own.mkdir()
    (own / f"{BOT} - battle-{replay_id(900)}.html").write_text(
        page(make_log("Ladder Foe", BOT, exact="p2", ratings=(1180, 1210)))
    )


def test_a_public_copy_of_an_own_game_does_not_take_its_id(tmp_path: Path):
    """2026-10-04: human sources are read before the saved own pages. A public
    copy of an own game in the feed was dropped as the bot's account AND kept the
    id, so the page was then skipped as a duplicate and the game left the ladder
    holdout."""
    public_copy = {
        "id": replay_id(900), "format": FMT, "uploadtime": 1800, "rating": 1180,
        "players": ["Ladder Foe", BOT], "log": make_log("Ladder Foe", BOT),
    }  # fmt: skip
    _tiny_corpus(tmp_path, [public_copy])
    built = B.build("t", root=tmp_path, log=lambda message: None)
    counted = built["battles"]
    assert counted["dropped"] == {"bot_account": 1}
    assert counted["duplicate_ids"] == {}
    assert built["headline"]["ladder_holdout_games"] == 1
    assert built["examples"]["by_split"] == {"ladder_holdout": 2}
    path = tmp_path / "results_oppmodel" / "t" / "battles.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(row["id"], row["source"]) for row in rows] == [(replay_id(900), "own")]
    # the examples are the page's (player view, HP rewritten), not the copy's
    data = F.load_dataset(tmp_path / "results_oppmodel" / "t")[0]
    assert set(data["m_source"].tolist()) == {B.SOURCES.index("own")}


def test_a_holdout_battle_is_neither_trained_nor_validated_on(tmp_path: Path):
    """2026-10-04 review: validation picks table strengths, the stopping epoch
    and the temperatures, so a validation player's battle against one of the
    bot's opponents goes to excluded_holdout_player like a training player's; a
    test player keeps the test split."""
    taken: set[str] = {"Ladder Foe", BOT}
    val_player = account(B.SPLIT_VAL, taken)
    test_player = account(B.SPLIT_TEST, taken)
    records = [
        {"id": replay_id(31), "format": FMT, "uploadtime": 1800, "rating": 1200,
         "players": [val_player, "Ladder Foe"],
         "log": make_log(val_player, "Ladder Foe")},
        {"id": replay_id(32), "format": FMT, "uploadtime": 1900, "rating": 1200,
         "players": [test_player, "Ladder Foe"],
         "log": make_log(test_player, "Ladder Foe")},
    ]  # fmt: skip
    _tiny_corpus(tmp_path, records)
    B.build("t", root=tmp_path, log=lambda message: None)
    path = tmp_path / "results_oppmodel" / "t" / "battles.jsonl"
    rows = {row["id"]: row for row in map(json.loads, path.read_text().splitlines())}
    foe_split = B.SPLITS[B.player_split(E.user_id("Ladder Foe"))]
    expected_foe = (
        "excluded_holdout_player" if foe_split in ("train", "val") else foe_split
    )
    assert rows[replay_id(31)]["split"] == {
        "p1": "excluded_holdout_player",
        "p2": expected_foe,
    }
    assert rows[replay_id(32)]["split"] == {"p1": "test", "p2": expected_foe}


# --- --human-fraction: the learning curve's subsample -------------------------

# With this seed the rule ALONE would drop both saved own games (and the feed's
# copy of the ladder game) at both fractions: whatever keeps them is the code.
SEED = 2
FRACTIONS = (0.25, 0.5)
OFFSET = {FMT: 1000, BO3: 5000}  # replay numbers of the fraction corpus


def feed_record(number: int, log: str, fmt: str = FMT, when: int = 0) -> dict[str, Any]:
    return {"id": replay_id(number, fmt), "format": fmt, "uploadtime": when,
            "rating": 1200, "players": [], "log": log}  # fmt: skip


def write_feed(root: Path, records: list[dict[str, Any]]) -> None:
    """The records as feed shards, one gzip member per game, by format."""
    for fmt in (FMT, BO3):
        mine = [record for record in records if record["format"] == fmt]
        if not mine:
            continue
        shard = root / B.FEED_DIR / "logs" / fmt / "part-00000.jsonl.gz"
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_bytes(
            b"".join(
                gzip.compress((json.dumps(record) + "\n").encode(), mtime=0)
                for record in mine
            )
        )


def write_own(root: Path, foe: str, stranger: str) -> None:
    """A ladder page (the bot is p2) and an unrated one (the bot is p1).

    A page's time is its file's modification time, so that is pinned: two
    corpora written a moment apart must give the same build.
    """
    own = root / "ladder_replays_mc_synthetic"
    own.mkdir(parents=True)
    pages = {
        900: make_log(foe, BOT, exact="p2", ratings=(1180, 1210)),
        901: make_log(BOT, stranger, exact="p1", rated=False),
    }
    for number, log in pages.items():
        path = own / f"{BOT} - battle-{replay_id(number)}.html"
        path.write_text(page(log))
        os.utime(path, (1_790_000_000 + number, 1_790_000_000 + number))


class FractionCorpus:
    """Sixty single games and sixteen series (two or three games each) in the
    feed, a public copy of the bot's ladder game among them, and two own pages."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifests: dict[str, dict[str, Any]] = {}  # filled by the fixture
        taken: set[str] = {BOT}
        train = [account(B.SPLIT_TRAIN, taken) for _ in range(6)]
        others = [
            account(B.SPLIT_VAL, taken),
            account(B.SPLIT_TEST, taken),
            account(B.SPLIT_TRAIN, taken),
        ]
        self.foe = account(B.SPLIT_TRAIN, taken)
        self.stranger = account(B.SPLIT_TRAIN, taken)
        moves = ("Dragon Claw", "Stomping Tantrum", "Outrage")
        records: list[dict[str, Any]] = []
        for n in range(60):
            when = 10_000 + 50 * n
            log = make_log(train[n % 6], others[n % 3], when=when, p2_move=moves[n % 3])
            records.append(feed_record(OFFSET[FMT] + n, log, when=when))
        number = OFFSET[BO3]
        for series in range(16):
            for game in range(1, 3 + series % 2):
                when = 20_000 + 100 * series + game
                log = make_log(
                    train[series % 6],
                    others[series % 3],
                    bo3=(str(700 + series), game),
                    when=when,
                    sheets=True,
                )
                records.append(feed_record(number, log, BO3, when))
                number += 1
        # The public copy of the bot's own ladder game, as the feed holds a few.
        self.copy = feed_record(900, make_log(self.foe, BOT), when=30_000)
        records.insert(7, self.copy)
        self.records = records
        write_feed(root, records)
        write_own(root, self.foe, self.stranger)

    def build(self, tag: str, **options: Any) -> dict[str, Any]:
        options.setdefault("log", lambda message: None)
        return B.build(tag, root=self.root, **options)

    def rows(self, tag: str) -> dict[str, dict[str, Any]]:
        path = self.root / "results_oppmodel" / tag / "battles.jsonl"
        return {
            row["id"]: row for row in map(json.loads, path.read_text().splitlines())
        }

    def human(self, tag: str) -> set[str]:
        return {key for key, row in self.rows(tag).items() if not row["own"]}

    def data(self, tag: str, split: str | None = None) -> F.Batch:
        directory = self.root / "results_oppmodel" / tag
        return F.load_dataset(directory, None if split is None else [split])[0]


@pytest.fixture(scope="module")
def fraction_corpus(tmp_path_factory: pytest.TempPathFactory) -> FractionCorpus:
    corpus = FractionCorpus(tmp_path_factory.mktemp("fraction"))
    corpus.manifests = {"full": corpus.build("full")}
    for fraction in FRACTIONS:
        corpus.manifests[f"f{fraction}"] = corpus.build(
            f"f{fraction}", human_fraction=fraction, fraction_seed=SEED
        )
    return corpus


def manifest_core(manifest: dict[str, Any]) -> dict[str, Any]:
    """A manifest without its name, its clocks and the hashes of its inputs."""
    core = json.loads(json.dumps(manifest))
    for key in ("tag", "created", "seconds", "inputs", "human_fraction"):
        core.pop(key, None)
    core["featurizer"].pop("encode_microseconds_per_example")
    for key in [key for key in core["headline"] if "fraction" in key]:
        del core["headline"][key]
    return core


def test_fraction_position_is_a_fixed_function_in_the_unit_interval():
    group = "gen9championsvgc2026regmc-2690928731"
    digest = hashlib.sha256(f"3:{group}".encode()).digest()
    expected = (int.from_bytes(digest[:8], "big") >> 11) / 2**53
    assert B.fraction_position(group, 3) == expected
    assert B.fraction_position(group, 3) != B.fraction_position(group, 4)
    assert B.fraction_keeps(group, expected + 1e-9, 3)
    assert not B.fraction_keeps(group, expected, 3)  # strictly below the fraction
    positions = [B.fraction_position(f"g{n}", 0) for n in range(4000)]
    assert 0.0 <= min(positions) and max(positions) < 1.0
    # All 53 bits set is the largest position, and it is still below 1.
    assert ((1 << B.FRACTION_BITS) - 1) / float(1 << B.FRACTION_BITS) < 1.0
    assert B.FRACTION_HASH == "sha256" and B.FRACTION_HASH in B.FRACTION_RULE


def test_fraction_subsamples_are_nested_and_one_keeps_everything():
    groups = [f"gen9championsvgc2026regmc-{2_690_000_000 + n}" for n in range(6000)]
    kept = {
        fraction: {g for g in groups if B.fraction_keeps(g, fraction, 7)}
        for fraction in (0.125, 0.25, 0.5, 1.0)
    }
    assert kept[0.125] < kept[0.25] < kept[0.5] < kept[1.0]
    assert kept[1.0] == set(groups)
    for fraction in (0.125, 0.25, 0.5):
        # 6,000 draws: three standard deviations at 0.5 are 0.019.
        assert abs(len(kept[fraction]) / len(groups) - fraction) < 0.02
    other_seed = {g for g in groups if B.fraction_keeps(g, 0.5, 8)}
    assert other_seed != kept[0.5]
    assert 0.2 < len(other_seed & kept[0.5]) / len(groups) < 0.3  # independent seeds


def test_fraction_does_not_follow_the_scrapers_crc32_order():
    """2026-10-05: the feed is downloaded in ascending crc32(replay id) order, so
    a half-finished download holds the low-crc32 ids. A subsample by crc32 of the
    (prefixed) id then keeps almost all or almost none of them: measured on
    v2_feed, 93% of the feed games with seed 0 and 7% with seed 1."""
    ids = [f"gen9championsvgc2026regmc-{2_690_000_000 + n}" for n in range(40_000)]
    fetched = [i for i in ids if zlib.crc32(i.encode()) < 0.15 * 2**32]
    assert 5000 < len(fetched) < 7000
    for seed in (0, 1):
        share = sum(B.fraction_keeps(i, 0.5, seed) for i in fetched) / len(fetched)
        assert abs(share - 0.5) < 0.03
        by_crc = sum(
            zlib.crc32(f"{seed}:{i}".encode()) / 2**32 < 0.5 for i in fetched
        ) / len(fetched)
        assert abs(by_crc - 0.5) > 0.2  # the rule this option must not use


def test_log_group_reads_what_the_driver_reads_without_driving():
    series = f"game-bestof3-{BO3}-77"
    single = make_log("A", "B")
    game = make_log("A", "B", bo3=("77", 2))
    assert B.log_group(single, "one") == "one"
    assert B.log_group(game, "two") == series
    mark = next(line for line in game.split("\n") if line.startswith("|uhtml|bestof|"))
    awkward = {
        "crlf": game.replace("\n", "\r\n"),
        "said in chat": single.replace("|gametype", f"|c|☆A|x{mark}\n|gametype"),
        "two marks, the last one counts": game.replace(
            "|gametype", mark.replace("-77", "-78") + "\n|gametype"
        ),
        "a mark nobody can read": single.replace(
            "|gametype", "|uhtml|bestof|<h2>soon</h2>\n|gametype"
        ),
        "unreadable after a good one": game.replace(
            "|gametype", "|uhtml|bestof|<h2>soon</h2>\n|gametype"
        ),
        "another uhtml": single.replace("|gametype", "|uhtml|other|x\n|gametype"),
        "empty": "",
    }
    expected = {
        "crlf": series,
        "said in chat": "id",
        "two marks, the last one counts": f"game-bestof3-{BO3}-78",
        "a mark nobody can read": "id",
        "unreadable after a good one": series,
        "another uhtml": "id",
        "empty": "id",
    }
    for name, log in awkward.items():
        driven = P.drive_log(log, "id").series_id or "id"
        assert B.log_group(log, "id") == driven == expected[name], name


def test_fraction_keeps_exactly_the_groups_the_rule_names(
    fraction_corpus: FractionCorpus,
):
    full = fraction_corpus.rows("full")
    everything = {key for key, row in full.items() if not row["own"]}
    assert len(everything) == 60 + 40 and len(full) == 102
    kept = {f: fraction_corpus.human(f"f{f}") for f in FRACTIONS}
    for fraction in FRACTIONS:
        by_rule = {
            key
            for key in everything
            if B.fraction_keeps(full[key]["group"], fraction, SEED)
        }
        assert kept[fraction] == by_rule
        assert 0.6 * fraction < len(by_rule) / len(everything) < 1.5 * fraction
    assert kept[0.25] < kept[0.5] < everything  # nested, and really smaller


def test_a_series_is_kept_or_dropped_as_a_whole(fraction_corpus: FractionCorpus):
    full = fraction_corpus.rows("full")
    series: dict[str, list[str]] = {}
    for key, row in full.items():
        if row["series"]:
            series.setdefault(row["group"], []).append(key)
    assert len(series) == 16 and {len(games) for games in series.values()} == {2, 3}
    for fraction in FRACTIONS:
        kept = fraction_corpus.human(f"f{fraction}")
        whole = [all(game in kept for game in games) for games in series.values()]
        none = [not any(game in kept for game in games) for games in series.values()]
        assert all(a or b for a, b in zip(whole, none))
        assert any(whole) and any(none)  # both happen: the check is not empty
        manifest = fraction_corpus.manifests[f"f{fraction}"]
        assert manifest["accounts"]["series"] == sum(whole)


def test_own_games_are_kept_at_every_fraction(fraction_corpus: FractionCorpus):
    tags = ["full", *(f"f{f}" for f in FRACTIONS)]
    own = [
        {key: row for key, row in fraction_corpus.rows(tag).items() if row["own"]}
        for tag in tags
    ]
    assert [sorted(rows) for rows in own] == [[replay_id(900), replay_id(901)]] * 3
    # Not by luck: their ids alone are outside both subsamples.
    assert not any(
        B.fraction_keeps(replay_id(number), max(FRACTIONS), SEED)
        for number in (900, 901)
    )
    reference = fraction_corpus.data("full", "ladder_holdout")
    assert len(reference["turn"]) == 2
    for tag, rows in zip(tags, own):
        manifest = fraction_corpus.manifests[tag]
        assert manifest["headline"]["own_games_kept"] == 2
        assert manifest["headline"]["ladder_holdout_games"] == 1
        assert manifest["examples"]["by_split"]["ladder_holdout"] == 2
        # The feed's copy of the ladder game takes the path it takes at F = 1:
        # dropped as the bot's account, without taking the page's id.
        assert manifest["battles"]["dropped"] == {"bot_account": 1}
        assert rows[replay_id(900)]["source"] == "own"
        data = fraction_corpus.data(tag, "ladder_holdout")
        # What the opponent did is the same; the row numbers of the battles and
        # the candidate lists (a smaller repertoire) may differ.
        for name in ("m_turn", "m_side", "m_actor", "turn", "act_mon", "mon_hp"):
            assert np.array_equal(data[name], reference[name]), (tag, name)
        for name in ("y_kind", "y_reason", "y_mega", "y_intent", "y_attack"):
            assert np.array_equal(data[name], reference[name]), (tag, name)


def test_a_fraction_build_is_the_build_of_the_smaller_corpus(
    fraction_corpus: FractionCorpus, tmp_path: Path
):
    """Decided before anything is counted: the same files, byte for byte, as a
    full build of a corpus that only ever held the kept logs."""
    fraction = 0.5
    groups = {key: row["group"] for key, row in fraction_corpus.rows("full").items()}
    kept = [
        record
        for record in fraction_corpus.records
        if record is fraction_corpus.copy
        or B.fraction_keeps(groups[record["id"]], fraction, SEED)
    ]
    assert len(kept) < len(fraction_corpus.records)
    write_feed(tmp_path, kept)
    write_own(tmp_path, fraction_corpus.foe, fraction_corpus.stranger)
    small = B.build("small", root=tmp_path, log=lambda message: None)
    sampled = fraction_corpus.manifests[f"f{fraction}"]
    there = tmp_path / "results_oppmodel" / "small"
    here = fraction_corpus.root / "results_oppmodel" / f"f{fraction}"
    names = sorted(path.name for path in here.iterdir())
    assert names == sorted(path.name for path in there.iterdir())
    for name in names:
        if name != "manifest.json":
            assert (here / name).read_bytes() == (there / name).read_bytes(), name
    assert manifest_core(sampled) == manifest_core(small)
    assert "human_fraction" not in small
    assert sampled["headline"]["human_battles_kept"] == len(kept) - 1
    assert sampled["battles"]["read"] == {"feed": len(kept), "own": 2}


def test_fraction_one_changes_nothing(fraction_corpus: FractionCorpus):
    full = fraction_corpus.manifests["full"]
    one = fraction_corpus.build("one", human_fraction=1.0, fraction_seed=99)
    assert "human_fraction" not in one and "human_fraction" not in full
    assert not [key for key in one["headline"] if "fraction" in key]
    assert list(one) == list(full) and list(one["headline"]) == list(full["headline"])
    assert manifest_core(one) == manifest_core(full)
    assert one["inputs"] == full["inputs"]
    root = fraction_corpus.root / "results_oppmodel"
    names = sorted(path.name for path in (root / "full").iterdir())
    assert names == sorted(path.name for path in (root / "one").iterdir())
    for name in names:
        if name != "manifest.json":
            assert (root / "one" / name).read_bytes() == (
                root / "full" / name
            ).read_bytes(), name


def test_fraction_is_decided_by_the_seed(fraction_corpus: FractionCorpus):
    again = fraction_corpus.build("again", human_fraction=0.5, fraction_seed=SEED)
    first = fraction_corpus.manifests["f0.5"]
    assert manifest_core(again) == manifest_core(first)
    assert again["human_fraction"] == first["human_fraction"]
    root = fraction_corpus.root / "results_oppmodel"
    for path in (root / "f0.5").iterdir():
        if path.name != "manifest.json":
            assert path.read_bytes() == (root / "again" / path.name).read_bytes()
    other = fraction_corpus.build("other", human_fraction=0.5, fraction_seed=SEED + 1)
    assert other["human_fraction"]["seed"] == SEED + 1
    assert fraction_corpus.human("other") != fraction_corpus.human("f0.5")


def test_the_manifest_records_the_fraction(fraction_corpus: FractionCorpus):
    full = fraction_corpus.manifests["full"]
    offered = full["battles"]["read"]["feed"]
    assert offered == 101
    for fraction in FRACTIONS:
        manifest = fraction_corpus.manifests[f"f{fraction}"]
        chosen = manifest["human_fraction"]
        assert (chosen["fraction"], chosen["seed"]) == (fraction, SEED)
        assert chosen["hash"] == "sha256" and chosen["rule"] == B.FRACTION_RULE
        assert chosen["logs_offered"] == offered
        assert chosen["own_game_copies_exempt"] == 1
        assert (
            chosen["logs_kept"] + chosen["logs_dropped"] + 1 == chosen["logs_offered"]
        )
        assert chosen["logs_kept"] == len(fraction_corpus.human(f"f{fraction}"))
        assert chosen["kept_share"] == pytest.approx(
            chosen["logs_kept"] / 100, abs=1e-6
        )
        assert chosen["by_source"] == {
            "feed": {
                "kept": chosen["logs_kept"],
                "dropped": chosen["logs_dropped"],
                "own_game_copies_exempt": 1,
            }
        }
        assert chosen["groups_kept"] + chosen["groups_dropped"] == 60 + 16
        assert chosen["group_mismatch"] == 0 and chosen["group_mismatch_examples"] == []
        headline = manifest["headline"]
        assert headline["human_fraction"] == fraction
        assert headline["fraction_seed"] == SEED
        assert headline["human_logs_kept_by_fraction"] == chosen["logs_kept"]
        assert headline["human_logs_dropped_by_fraction"] == chosen["logs_dropped"]
        # Every count is the smaller corpus's: a dropped log was never read.
        assert manifest["battles"]["read"] == {
            "feed": chosen["logs_kept"] + 1,
            "own": 2,
        }
        assert headline["human_battles_kept"] == chosen["logs_kept"]
        assert headline["human_logs_unique"] == chosen["logs_kept"] + 1
        assert manifest["accounts"]["distinct_accounts"] <= 9
        on_disk = json.loads(
            (
                fraction_corpus.root
                / "results_oppmodel"
                / f"f{fraction}"
                / "manifest.json"
            ).read_text()
        )
        assert on_disk["human_fraction"] == chosen
        assert list(on_disk)[-1] == "human_fraction"  # appended, nothing reordered


def test_the_repertoire_and_the_weights_are_those_of_the_subsample(
    fraction_corpus: FractionCorpus,
):
    root = fraction_corpus.root / "results_oppmodel"
    full = json.loads((root / "full" / "repertoire.json").read_text())["counts"]
    half = json.loads((root / "f0.5" / "repertoire.json").read_text())["counts"]
    assert set(half) <= set(full)
    assert all(half[key][move] <= full[key][move] for key in half for move in half[key])
    assert sum(map(sum, (m.values() for m in half.values()))) < sum(
        map(sum, (m.values() for m in full.values()))
    )
    # The account cap counts the battles that are there: one account that is
    # over a cap of 8 in the whole corpus weighs more in the half.
    capped = fraction_corpus.build("capped", account_cap=8)
    capped_half = fraction_corpus.build(
        "capped_half", account_cap=8, human_fraction=0.5, fraction_seed=SEED
    )
    weights = {a["account"]: a["weight"] for a in capped["accounts"]["top_accounts"]}
    weights_half = {
        a["account"]: a["weight"] for a in capped_half["accounts"]["top_accounts"]
    }
    assert set(weights_half) <= set(weights)
    assert all(weights_half[name] >= weights[name] for name in weights_half)
    assert any(weights_half[name] > weights[name] for name in weights_half)


def test_the_fraction_is_taken_of_the_limited_logs(fraction_corpus: FractionCorpus):
    # --limit counts logs per source: the 61 single-game records, then the
    # first nine games of the series shard.
    limited = fraction_corpus.build("lim", limit=70)
    assert limited["battles"]["read"] == {"feed": 70, "own": 2}
    rows = fraction_corpus.rows("lim")
    whole = fraction_corpus.human("lim")
    assert len(whole) == 69 and sum(1 for key in whole if rows[key]["series"]) == 9
    kept = {}
    for fraction in FRACTIONS:
        tag = f"lim{fraction}"
        manifest = fraction_corpus.build(
            tag, limit=70, human_fraction=fraction, fraction_seed=SEED
        )
        assert manifest["limit"] == 70
        assert manifest["human_fraction"]["logs_offered"] == 70
        kept[fraction] = fraction_corpus.human(tag)
        assert kept[fraction] == {
            key for key in whole if B.fraction_keeps(rows[key]["group"], fraction, SEED)
        }
    assert kept[0.25] < kept[0.5] < whole


def test_a_fraction_outside_the_unit_interval_is_refused(
    fraction_corpus: FractionCorpus,
):
    for value in (0.0, -0.25, 1.5, float("nan")):
        with pytest.raises(SystemExit):
            fraction_corpus.build("bad", human_fraction=value)
    assert not (fraction_corpus.root / "results_oppmodel" / "bad").exists()
    smallest = fraction_corpus.build("tiny", human_fraction=1e-9, fraction_seed=SEED)
    assert smallest["headline"]["human_battles_kept"] == 0
    assert smallest["examples"]["by_split"] == {"ladder_holdout": 2, "own_unrated": 2}


def test_a_group_read_differently_from_the_driven_one_is_counted(
    fraction_corpus: FractionCorpus, monkeypatch: pytest.MonkeyPatch
):
    """The decision uses the group read off the raw text; the driven battle's
    group is compared with it afterwards, so a reader that went wrong shows."""
    monkeypatch.setattr(B, "log_group", lambda log, battle_id: battle_id)
    manifest = fraction_corpus.build("wrong", human_fraction=0.9, fraction_seed=SEED)
    chosen = manifest["human_fraction"]
    series_games = sum(
        1 for row in fraction_corpus.rows("wrong").values() if row["series"]
    )
    assert series_games > 0
    assert chosen["group_mismatch"] == series_games
    assert 0 < len(chosen["group_mismatch_examples"]) <= B.DROP_EXAMPLES


def test_real_logs_have_the_group_the_driver_finds():
    reader = B.Corpus(ROOT, B.DEFAULT_FORMATS, ["top_merged", "feed"], limit=150)
    compared = series = 0
    for raw in reader:
        driven = B.drive(raw).series_id or raw.battle_id
        assert B.log_group(raw.log, raw.battle_id) == driven, raw.battle_id
        compared += 1
        series += int(driven != raw.battle_id)
    if not compared:
        pytest.skip("no human corpus (battle_logs_*/) on this machine")
    assert series > 0
