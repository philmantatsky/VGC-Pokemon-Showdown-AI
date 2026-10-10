"""Opponent predictor ladder read: pages, sheets, per-artifact scoring, pairing.

Everything runs on a few synthetic saved pages written under ``tmp_path`` with
the real folder layout (the logs of ``test_oppmodel_dataset``) and on two tiny
count-table artifacts with different featurizers. One test at the end reads the
repo's own saved pages and fitted table and skips when they are absent (they
are git-ignored).
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from datagen import oppmodel_build_dataset as B
from evaluation import oppmodel_ladder_read as LR
from evaluation import oppmodel_scorecard as SC
from unit_tests import test_oppmodel_dataset as TD
from unit_tests import test_oppmodel_scorecard as TS
from unit_tests import test_oppmodel_setprior_integration as TSP
from vgc_bench.src.oppmodel import artifact as A
from vgc_bench.src.oppmodel import coupling as C
from vgc_bench.src.oppmodel import events as E
from vgc_bench.src.oppmodel import features as F
from vgc_bench.src.oppmodel import joint as J
from vgc_bench.src.oppmodel import tables as T

ROOT = Path(__file__).resolve().parents[1]
BOT = TD.BOT
FOLDER = "ladder_replays_mc_unit"
TEAM = "teams/unit_team.txt"
CLOSED, OPEN, UNRATED, EMPTY, FALSE_IDENTITY, SECOND = 900, 901, 902, 903, 904, 905
# The synthetic pages are dated before the sealed confirmation set begins, so a
# reading of them needs no ``fresh``.
PLAYED = "2026-09-15"
PLAYED_AT = time.mktime(time.strptime(f"{PLAYED} 12:00", "%Y-%m-%d %H:%M"))
THEIR_SHEET = {
    "Garchomp": {
        "item": "Garchompite",
        "ability": "Rough Skin",
        "moves": ["Dragon Claw", "Earthquake", "Protect", "Rock Slide"],
    },
    "Sneasler": {
        "item": "Focus Sash",
        "ability": "Unburden",
        "moves": ["Close Combat", "Dire Claw", "Fake Out", "Protect"],
    },
}
OWN_TEAM = """Incineroar @ Sitrus Berry
Ability: Intimidate
Level: 50
Adamant Nature
- Fake Out
- Flare Blitz
- Parting Shot
- Protect

Rillaboom @ Miracle Seed
Ability: Grassy Surge
Level: 50
Adamant Nature
- Wood Hammer
- Grassy Glide
- Protect
- Fake Out
"""


def tag(number: int) -> str:
    return f"battle-{TD.replay_id(number)}"


def save_page(folder: Path, number: int, log: str, when: float = PLAYED_AT) -> Path:
    """One saved page of the bot's, dated ``when``."""
    path = folder / f"{BOT} - {tag(number)}.html"
    path.write_text(TD.page(log))
    os.utime(path, (when, when))
    return path


def write_directory(root: Path, team: bool = True) -> Path:
    """Six saved pages and their decision logs, as a ladder session leaves them.

    900 closed (the opponent is p1), 901 open with the opponent's sheet in the
    log (the opponent is p2), 902 unrated, 903 without a turn, 904 with a
    Pokemon whose identity can be false, 905 with a row that says nothing
    about sheets.
    """
    folder = root / FOLDER
    folder.mkdir(parents=True)
    pages = {
        CLOSED: TD.make_log("Foe One", BOT, exact="p2", ratings=(1180, 1210)),
        OPEN: TD.make_log(BOT, "Foe Two", exact="p1", ratings=(1215, 1302)),
        UNRATED: TD.make_log(BOT, "Foe Three", exact="p1", rated=False),
        EMPTY: TD.make_log("Foe Four", BOT, exact="p2", turns=False),
        FALSE_IDENTITY: TD.make_log("Foe Five", BOT, extra_species="Zoroark"),
        SECOND: TD.make_log(BOT, "Foe Six", exact="p1", p2_move="Outrage"),
    }
    for number, log in pages.items():
        save_page(folder, number, log)
    search = [
        {"battle": tag(CLOSED), "turn": 1, "exact_search": {"open_sheet": False}},
        {
            "battle": tag(OPEN),
            "turn": 0,
            "sheet_preview": {},
            "their_sheet": THEIR_SHEET,
        },
        {"battle": tag(OPEN), "turn": 1, "exact_search": {"open_sheet": True}},
        '{"battle": "' + tag(CLOSED) + '", "exact_sea',  # a line being written
    ]
    (folder / "decisions.jsonl").write_text(
        "\n".join(r if isinstance(r, str) else json.dumps(r) for r in search)
    )
    own = [{"battle": tag(SECOND), "turn": 1, "candidates": [], "chosen": 3}]
    (folder / "decisions_champion.jsonl").write_text(
        "\n".join(json.dumps(r) for r in own)
    )
    if team:
        (root / "teams").mkdir(exist_ok=True)
        (root / TEAM).write_text(OWN_TEAM)
        config = {"runs": [{"material": {"our_team": TEAM}}] * 2}
        (folder / LR.RUN_CONFIG).write_text(json.dumps(config))
    return folder


def featurizer_of(root: Path, n_cand: int = F.N_CAND_DEFAULT) -> F.Featurizer:
    """A featurizer whose repertoire is the moves of the directory's own pages."""
    repertoire = F.Repertoire()
    for raw in LR.PageCorpus(root, [root / FOLDER]):
        result = B.drive(raw)
        for _, key, move, auto, terrain in B.free_move_uses(result):
            repertoire.add(
                key, move, 1.0, auto=None if auto < 0 else bool(auto), terrain=terrain
            )
    return F.Featurizer.build(repertoire, n_cand)


def save_table(
    path: Path,
    root: Path,
    kind: Any,
    n_cand: int = F.N_CAND_DEFAULT,
    coupling: Any = None,
) -> Path:
    """Fit a count table on the directory's kept games and store it (with a
    pair coupling beside it when one is given)."""
    featurizer = featurizer_of(root, n_cand)
    games, _ = LR.read_games(root, [root / FOLDER])
    batch = dict(LR.encode_games(games, featurizer))
    batch["m_weight"] = np.ones(len(batch["turn"]), dtype=np.float32)
    table = kind.fit(batch, featurizer=featurizer)
    A.save_artifact(
        path,
        kind=A.KIND_TABLE,
        name=table.name,
        featurizer=featurizer,
        predictor_payload=table.to_payload(),
        extra={"dataset_tag": "unit"},
        coupling=coupling,
    )
    return path


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("ladder_read")
    write_directory(root)
    flags = save_table(root / "flags.pt", root, T.FlagsTable)
    species = save_table(root / "species.pt", root, T.SpeciesTable, n_cand=6)
    return {"root": root, "flags": flags, "species": species}


def read(world: dict[str, Any], out: str, **options: Any) -> dict[str, Any]:
    options.setdefault("log", lambda text: None)
    options.setdefault("resamples", 200)
    return LR.run(
        [f"flags={world['flags']}", f"species={world['species']}"],
        world["root"] / out,
        directories=[FOLDER],
        root=world["root"],
        **options,
    )


# --- pages and sheets -----------------------------------------------------------


def test_games_are_kept_and_skipped_as_the_builder_would(world: dict[str, Any]):
    root = world["root"]
    games, info = LR.read_games(root, [root / FOLDER])
    assert [game.battle_id for game in games] == [
        TD.replay_id(number) for number in (CLOSED, OPEN, SECOND)
    ]
    assert info["pages"] == 6 and info["kept"] == 3
    assert info["skipped"] == {
        "builder:illusion_species": 1,
        "builder:no_turn_1": 1,
        "not_ladder_rated": 1,
    }
    # Every page is kept or counted, and every file found is named.
    assert info["pages_unaccounted"] == 0
    assert info["files"][FOLDER] == {
        "html_files": 6,
        "not_a_saved_game_page": 0,
        "another_format": 0,
        "pages": 6,
    }
    assert info["no_page"]["games"] == 0 and info["sheet_rows_disagree"] == 0
    assert info["bot_accounts"] == [E.user_id(BOT)]
    assert info["decision_logs"][FOLDER]["bad_lines"] == 1
    closed, opened, second = games
    assert (closed.bot_side, closed.opponent) == ("p2", "p1")
    assert (opened.bot_side, opened.opponent) == ("p1", "p2")
    # A game whose rows say nothing about sheets is unknown, not closed.
    assert [game.sheet for game in games] == ["closed", "open", "unknown"]
    assert closed.sheet_evidence == "exact_search"
    assert opened.sheet_evidence == "their_sheet,exact_search"
    assert second.sheet_evidence == "rows_say_nothing"
    assert [game.own_sheet for game in games] == [False, True, False]
    assert [game.both_sheets for game in games] == [False, True, False]
    assert {game.folder for game in games} == {str(root / FOLDER)}
    row = closed.row()
    assert row["battle"] == tag(CLOSED) and row["directory"] == FOLDER
    assert (row["opponent_rating"], row["bot_rating"]) == (1180, 1210)
    assert row["turns"] == 2 and row["finished"] and row["bot_won"] is True
    assert row["date"] == PLAYED and row["time_source"] == "mtime"
    assert opened.row()["opponent_rating"] == 1302
    kept, more = LR.read_games(root, [root / FOLDER], include_unrated=True)
    assert len(kept) == 4 and "not_ladder_rated" not in more["skipped"]


def test_the_examples_are_the_dataset_builders(world: dict[str, Any], tmp_path: Path):
    """A closed game is encoded as the built dataset stores it (bit for bit in
    what a predictor is given), and the bot's exact HP is the public percent."""
    root = world["root"]
    B.build("unit", root=root, out_root=tmp_path, sources=("own",), log=lambda m: None)
    stored, _ = F.load_dataset(tmp_path / "unit")
    featurizer = F.Featurizer.load(tmp_path / "unit")
    ids = {
        json.loads(line)["id"]: json.loads(line)["index"]
        for line in (tmp_path / "unit" / "battles.jsonl").read_text().splitlines()
    }
    games, _ = LR.read_games(root, [root / FOLDER])
    made = LR.encode_games(games, featurizer)
    # The builder reads ``preview_shadow`` alone, so these games are "unknown"
    # to it where the search's rows say closed here (and unknown here too where
    # no row says anything): the one difference, and none once a predictor's
    # view of an unknown sheet (closed) is taken.
    assert set(stored["game_flag"][:, F.G_ACTOR_SHEET].tolist()) == {F.SHEET_UNKNOWN}
    made, stored = F.sheet_unknown_as_closed(made), F.sheet_unknown_as_closed(stored)
    compared = 0
    for game in games:
        if game.sheet == "open":
            continue
        here = made["m_battle"] == game.index
        there = stored["m_battle"] == ids[game.battle_id]
        assert here.sum() == there.sum() == 2
        for name in featurizer.layout():
            assert np.array_equal(made[name][here], stored[name][there]), name
            compared += 1
    assert compared > 40
    # The actor is the opponent (p1); the bot's Sneasler is mon row 7: 165/187
    # on the page, 88% to the opponent.
    turn_2 = (made["m_battle"] == 0) & (made["m_turn"] == 2)
    row = int(np.argmax(turn_2))
    assert float(made["mon_hp"][row, 7]) == pytest.approx(0.88, abs=1e-3)
    assert made["m_sheet"][row] == B.SHEET_STATES.index("closed")
    # The bot's own side is never an example: one per turn and game.
    assert len(made["turn"]) == 6 and sorted(made["m_row"].tolist()) == [
        0,
        0,
        0,
        1,
        1,
        1,
    ]


def test_sheet_state_is_read_from_every_kind_of_row(tmp_path: Path):
    folder = tmp_path / "ladder_replays_mc_rows"
    folder.mkdir()
    forecast = {"slots": [None, None], "sheets_reported": ["open", "open"]}
    rows = [
        {"battle": "battle-fmt-1", "preview_shadow": {"open_sheet": True}},
        {"battle": "battle-fmt-2", "preview_shadow": {"open_sheet": False}},
        {"battle": "battle-fmt-3", "exact_preview": {"open_sheet": True}},
        {"battle": "battle-fmt-4", "turn": 3, "opponent_forecast": forecast},
        {"battle": "battle-fmt-5", "turn": 2, "opponent_forecast": {"stand_down": "x"}},
        {"battle": "battle-fmt-6-pw", "exact_search": {"open_sheet": False}},
        {"battle": "battle-fmt-6-pw", "exact_search": {"open_sheet": True}},
        {"battle": "battle-fmt-7", "their_sheet": THEIR_SHEET},
        {"no": "battle"},
    ]
    (folder / "decisions.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    logs = LR.read_directory_logs(folder)
    states = {battle: note.state for battle, note in logs.sheets.items()}
    assert states == {
        "fmt-1": "open",
        "fmt-2": "closed",
        "fmt-3": "open",
        "fmt-4": "open",
        "fmt-5": "unknown",  # a row, and nothing in it about sheets
        "fmt-6": "open",
        "fmt-7": "open",
    }
    assert logs.sheets["fmt-6"].evidence == "exact_search"
    assert logs.sheets["fmt-6"].disagree and not logs.sheets["fmt-2"].disagree
    assert logs.sheets["fmt-5"].evidence == "rows_say_nothing"
    assert logs.sheets["fmt-2"].evidence == "preview_shadow"
    assert [entry.species for entry in logs.sheets["fmt-7"].sets] == [
        "garchomp",
        "sneasler",
    ]
    assert list(logs.forecasts) == [("fmt-4", 3)] and logs.bad_lines == 0
    # The bytes that were read are hashed.
    stored = (folder / "decisions.jsonl").read_bytes()
    assert logs.hashes == {
        "decisions.jsonl": {
            "sha256": hashlib.sha256(stored).hexdigest(),
            "bytes": len(stored),
        }
    }
    assert LR.read_directory_logs(tmp_path / "nothing").sheets == {}
    assert LR.SheetNote().state == "unknown"


def test_an_open_game_is_given_both_sheets(world: dict[str, Any], tmp_path: Path):
    root = world["root"]
    featurizer = featurizer_of(root)
    games, info = LR.read_games(root, [root / FOLDER])
    team = (root / TEAM).read_bytes()
    assert info["own_sheet"] == {
        "mode": "team-file",
        "open_games_kept": 1,
        "both_sheets": 1,
        "given": 1,
        "from_a_public_copy": 0,
        "not_given_all_pages": {},
        # The team file is the one on disk now: its hash goes with the reading.
        "team_files": {
            FOLDER: {
                "path": str(root / TEAM),
                "sha256": hashlib.sha256(team).hexdigest(),
                "run_config_sha256": hashlib.sha256(
                    (root / FOLDER / LR.RUN_CONFIG).read_bytes()
                ).hexdigest(),
                "sets": 2,
                "why_not": "",
            }
        },
    }
    batch = LR.encode_games(games, featurizer)
    rows = batch["m_battle"] == 1
    flags = batch["game_flag"][rows]
    assert set(flags[:, F.G_ACTOR_SHEET].tolist()) == {F.SHEET_OPEN}
    assert set(flags[:, F.G_OTHER_SHEET].tolist()) == {F.SHEET_OPEN}
    # The opponent's candidates are its sheet's four moves, nothing guessed.
    valid = (batch["cand_flag"][rows] & F.CAND_VALID) > 0
    sheet = (batch["cand_flag"][rows] & F.CAND_SHEET) > 0
    assert valid.sum(-1).tolist() == [[4, 4], [4, 4]] and (valid == sheet).all()
    assert set(batch["m_sheet"][rows].tolist()) == {B.SHEET_STATES.index("open")}
    # Switched off, or without a team file: the builder's way, and it is said.
    plain, info = LR.read_games(root, [root / FOLDER], own_sheet="none")
    assert [game.own_sheet for game in plain] == [False, False, False]
    assert [game.both_sheets for game in plain] == [False, False, False]
    assert info["own_sheet"]["not_given_all_pages"] == {"switched_off": 1}
    assert info["own_sheet"]["both_sheets"] == 0
    batch = LR.encode_games(plain, featurizer)
    flags = batch["game_flag"][batch["m_battle"] == 1]
    assert set(flags[:, F.G_ACTOR_SHEET].tolist()) == {F.SHEET_OPEN}
    assert set(flags[:, F.G_OTHER_SHEET].tolist()) == {F.SHEET_UNKNOWN}
    bare = tmp_path / "bare"
    write_directory(bare, team=False)
    _, info = LR.read_games(bare, [bare / FOLDER])
    assert info["own_sheet"]["given"] == 0
    assert info["own_sheet"]["not_given_all_pages"] == {"no_run_config": 1}
    sets, why = LR.team_file_sets(root, root / FOLDER)
    assert why == "" and [entry.species for entry in sets] == [
        "incineroar",
        "rillaboom",
    ]
    assert sets[0].moves == ("fakeout", "flareblitz", "partingshot", "protect")
    (bare / FOLDER / LR.RUN_CONFIG).write_text(
        json.dumps({"runs": [{"material": {"our_team": name}} for name in "ab"]})
    )
    assert LR.team_file_sets(bare, bare / FOLDER) == ([], "runs_name_several_teams")
    with pytest.raises(LR.LadderReadError, match="own-sheet"):
        LR.read_games(root, [root / FOLDER], own_sheet="maybe")


def test_filters(world: dict[str, Any]):
    root = world["root"]
    one, info = LR.read_games(root, [root / FOLDER], games=tag(SECOND))
    assert [game.battle_id for game in one] == [TD.replay_id(SECOND)]
    assert info["skipped"] == {"filter:not in --games": 5}
    listing = root / "ids.txt"
    listing.write_text(f"{tag(CLOSED)}\n{TD.replay_id(OPEN)}\n")
    two, _ = LR.read_games(root, [root / FOLDER], games=f"@{listing}")
    assert len(two) == 2
    later = time.strftime("%Y-%m-%d", time.localtime(PLAYED_AT + 2 * 86400))
    none, info = LR.read_games(root, [root / FOLDER], since=later)
    assert none == [] and info["skipped"] == {"filter:before --since": 6}
    none, info = LR.read_games(root, [root / FOLDER], until="2020-01-01")
    assert none == [] and info["skipped"] == {"filter:at or after --until": 6}
    every, _ = LR.read_games(root, [root / FOLDER], since="2020-01-01T00:00")
    assert len(every) == 3
    with pytest.raises(LR.LadderReadError, match="--since"):
        LR.read_games(root, [root / FOLDER], since="yesterday")
    with pytest.raises(LR.LadderReadError, match="no such replay directory"):
        LR.read_games(root, [root / "ladder_replays_mc_missing"])
    with pytest.raises(LR.LadderReadError, match="given twice"):
        LR.read_games(root, [root / FOLDER, root / "teams" / ".." / FOLDER])


# --- the reading ----------------------------------------------------------------


def test_reading_holds_the_scorecards_numbers(world: dict[str, Any]):
    reading = read(world, "reading")
    root = world["root"]
    assert reading["format"] == LR.FORMAT and reading["order"] == ["flags", "species"]
    assert reading["settings"]["reference"] == "flags"
    assert reading["settings"]["thresholds"] == {"switch": 0.35, "protect": 0.5}
    games, _ = LR.read_games(root, [root / FOLDER])
    for label, n_cand in (("flags", F.N_CAND_DEFAULT), ("species", 6)):
        found = reading["artifacts"][label]
        assert found["artifact"]["n_cand"] == n_cand
        assert found["artifact"]["kind"] == "table" and found["sanity"]["unchanged"]
        # Scored through the artifact's own featurizer, with the scorecard's functions.
        loaded = A.load_predictor(world[label])
        scored = LR.score_artifact(label, loaded, games)
        assert scored.batch["action_mask"].shape[-1] == n_cand + 1 + F.N_ROSTER
        card = SC.score_set(
            scored.batch,
            {label: scored.pred},
            label,
            tables=loaded.featurizer.tables,
            resamples=50,
        )
        joint = SC.joint_coverage_set(scored.batch, {label: scored.pred}, resamples=50)
        whole = found["splits"]["all"]
        theirs = card["predictors"][label]
        assert whole["examples"] == card["examples"] == 6 and whole["games"] == 3
        assert whole["slot_turns"] == card["slots"]["scored"]
        assert whole["hidden"] == card["slots"]["censored"] > 0
        assert whole["fine_nll"]["value"] == pytest.approx(theirs["fine_nll"]["value"])
        assert whole["fine_nll"]["low"] <= whole["fine_nll"]["value"]
        for key in ("fine_top1", "fine_top3", "intent_accuracy", "fine_labels"):
            assert whole[key] == pytest.approx(theirs[key]), key
        for event in LR.EVENTS:
            mine, card_event = whole["events"][event], theirs["events"][event]
            assert mine["known"] == card_event["slots"]
            assert mine["observed"] == pytest.approx(card_event["observed"])
            assert mine["predicted"] == pytest.approx(card_event["predicted"])
        counted = whole["joint"]
        assert counted["counted"] == joint["counted"]["examples"]
        assert counted["turns"] == 6
        plain = joint["predictors"][label][SC.JOINT_PLAIN]["slices"]
        with_mega = joint["predictors"][label][SC.JOINT_MEGA]["slices"]
        replies = counted["replies"]
        for k in LR.TOP_KS:
            assert replies["all"]["top"][str(k)] == pytest.approx(
                plain["all"]["top"][str(k)]
            )
        assert replies["all"]["top_with_mega"] == pytest.approx(
            with_mega["all"]["top"][str(LR.TOP_K)]
        )
        assert (
            replies[LR.REPLY_UNSHOWN]["examples"]
            == (joint["slices"][SC.REPLY_UNSHOWN]["examples"])
        )
        # The three reply kinds partition the counted turns; so do the two others.
        kinds = (LR.REPLY_SWITCH, LR.REPLY_PROTECT, LR.REPLY_MOVES)
        assert sum(replies[kind]["examples"] for kind in kinds) == counted["counted"]
        assert (
            replies[LR.REPLY_UNSHOWN]["examples"]
            + replies[LR.REPLY_NO_UNSHOWN]["examples"]
            == counted["counted"]
        )
        # Splits add up, and a split inside one game has no interval.
        splits = found["splits"]
        assert (
            splits["sheet closed"]["slot_turns"]
            + splits["sheet open"]["slot_turns"]
            + splits["sheet unknown"]["slot_turns"]
            == whole["slot_turns"]
        )
        assert splits["sheet unknown"]["examples"] == 2  # the game no row tells
        assert splits["sheet unknown"]["games"] == 1
        assert (
            splits["sheet not open"]["slot_turns"]
            == splits["sheet closed"]["slot_turns"]
            + splits["sheet unknown"]["slot_turns"]
        )
        assert (
            splits["turn 1"]["slot_turns"] + splits["turn 2 and later"]["slot_turns"]
            == whole["slot_turns"]
        )
        assert splits["sheet open"]["games"] == 1
        assert splits["sheet open"]["fine_nll"]["low"] is None
        assert splits["sheet open"]["fine_nll"]["value"] is not None
        # The search session's way of counting never has fewer hits than strict.
        rule = counted["search_rule"][LR.PLAIN]
        assert rule[LR.RULE_VISIBLE]["turns"] == counted["counted"]
        assert rule[LR.RULE_ALL_MISS]["turns"] == 6
        assert rule[LR.RULE_ALL_MISS]["hits"] == rule[LR.RULE_VISIBLE]["hits"]
        assert rule[LR.RULE_ALL_ANY]["turns"] == 6 - counted["nothing_shown"]
        assert rule[LR.RULE_ALL_ANY]["hits"] >= rule[LR.RULE_VISIBLE]["hits"]
        # These directories hold no search audit with an observed reply.
        assert rule[LR.RULE_SEARCH] == {"turns": 0, "hits": 0, "share": None}
        assert found["search_decisions_joined"] == 0
        # No coupling in these artifacts: the plain product, and it is said.
        assert found["artifact"]["joint"] == LR.JOINT_PRODUCT
        assert found["artifact"]["coupling"] is None
        assert "top_plain_product" not in replies["all"]
    # One row per game, and the rows add up to the totals.
    rows = reading["games"]
    assert [row["battle"] for row in rows] == [tag(n) for n in (CLOSED, OPEN, SECOND)]
    assert [row["sheet"] for row in rows] == ["closed", "open", "unknown"]
    assert [row["own_sheet_given"] for row in rows] == [False, True, False]
    assert [row["both_sheets"] for row in rows] == [False, True, False]
    for label in ("flags", "species"):
        whole = reading["artifacts"][label]["splits"]["all"]
        cells = [row["artifacts"][label] for row in rows]

        def total(name: str, cells: list[dict[str, Any]] = cells) -> float:
            return sum(cell[name] for cell in cells)

        assert total("slot_turns") == whole["slot_turns"]
        assert total("nll_sum") == pytest.approx(
            whole["fine_nll"]["value"] * whole["slot_turns"]
        )
        # Every headline share can be rebuilt from the per-game sums.
        assert total("fine_labels") == whole["fine_labels"] > 0
        for k in (1, 3):
            assert total(f"fine_top{k}") / total("fine_labels") == pytest.approx(
                whole[f"fine_top{k}"]
            )
        assert total("intent_labels") == whole["intent_labels"]
        assert total("intent_hits") / total("intent_labels") == pytest.approx(
            whole["intent_accuracy"]
        )
        for event in LR.EVENTS:
            mine = whole["events"][event]
            assert total(f"{event}_known") == mine["known"]
            assert total(f"{event}_observed") / mine["known"] == pytest.approx(
                mine["observed"]
            )
            assert total(f"{event}_predicted_sum") / mine["known"] == pytest.approx(
                mine["predicted"]
            )
            assert total(f"{event}_at_threshold") == mine["at_threshold"]["slots"]
        joint = whole["joint"]
        assert total("joint_counted") == joint["counted"]
        for k in LR.TOP_KS:
            assert total(f"joint_top{k}") / joint["counted"] == pytest.approx(
                joint["replies"]["all"]["top"][str(k)]
            )
        for short, reply in LR.GAME_REPLIES:
            row = joint["replies"][reply]
            assert total(f"joint_counted_{short}") == row["examples"]
            if row["examples"]:
                assert total(f"joint_top8_{short}") / row["examples"] == pytest.approx(
                    row["top"]["8"]
                )
    # What produced the reading is in it: the code and the logs that were read.
    code = reading["code"]
    assert len(code["sha256"]) == 64 and code["files"]
    for name in LR.INSTRUMENT_FILES:
        assert code["files"][name] == SC.sha256_file(ROOT / name)
    assert "vgc_bench/src/oppmodel/joint.py" in code["files"]
    read_logs = reading["input"]["decision_logs"][FOLDER]["read"]
    assert set(read_logs) == {"decisions.jsonl", "decisions_champion.jsonl"}
    for name, entry in read_logs.items():
        assert entry["sha256"] == SC.sha256_file(root / FOLDER / name)
    assert reading["input"]["sealed"] == {
        "since": LR.FRESH_SINCE,
        "games": 0,
        "opened_with_fresh": False,
        "directories_found": None,
    }
    assert json.loads((root / "reading" / LR.READ_JSON).read_text()) == reading


def test_paired_differences_are_on_the_same_slot_turns(world: dict[str, Any]):
    reading = read(world, "paired")
    assert list(reading["paired"]) == ["species"]
    found = reading["paired"]["species"]
    assert found["examples_both"] == 6 and found["examples_only_here"] == 0
    assert found["slot_turns_labelled_by_one_only"] == 0
    flags = reading["artifacts"]["flags"]["splits"]
    species = reading["artifacts"]["species"]["splits"]
    for name in ("all", "sheet closed", "sheet open", "turn 1"):
        block = found["splits"][name]
        assert block["fine_nll"]["slots"] == flags[name]["slot_turns"]
        assert block["fine_nll"]["diff"] == pytest.approx(
            species[name]["fine_nll"]["value"] - flags[name]["fine_nll"]["value"]
        )
        top = block["joint_top8"]
        assert top["turns"] == flags[name]["joint"]["counted"]
        if top["turns"]:
            assert top["diff"] == pytest.approx(
                species[name]["joint"]["replies"]["all"]["top"]["8"]
                - flags[name]["joint"]["replies"]["all"]["top"]["8"]
            )
            assert top["only_here"] - top["only_reference"] == pytest.approx(
                top["diff"] * top["turns"]
            )
    assert found["splits"]["all"]["fine_nll"]["games"] == 3
    assert found["splits"]["all"]["fine_nll"]["low"] is not None
    other_way = read(world, "paired_other_way", reference="species")
    # The slot-turns are the same; the labels need not be, and it is counted.
    marks = found["labels"]
    assert marks["slot_turns"] == flags["all"]["slot_turns"]
    assert marks["other_either"] >= max(marks["other_here"], marks["other_reference"])
    assert marks["slots_acting"] == 12
    assert 0 <= marks["slots_with_different_candidates"] <= 12
    named = found["splits"]["all"]["fine_nll_neither_other"]
    assert named["slots"] == marks["slot_turns"] - marks["other_either"]
    if not marks["other_either"]:
        assert named["diff"] == pytest.approx(
            found["splits"]["all"]["fine_nll"]["diff"]
        )
    top = found["splits"]["all"]["joint_top8"]
    assert top["sign_test_p"] == LR.sign_test(top["only_here"], top["only_reference"])
    assert LR.sign_test(0, 0) is None and LR.sign_test(3, 3) == 1.0
    # How often a resample of the games does not favour the artifact: the two
    # directions of one comparison add up to at least the whole.
    for kind in ("fine_nll", "joint_top8"):
        here = found["splits"]["all"][kind]["resamples_not_better"]
        there = other_way["paired"]["flags"]["splits"]["all"][kind]
        assert 0.0 <= here <= 1.0 and here + there["resamples_not_better"] >= 1.0
    assert LR.sign_test(26, 16) == pytest.approx(0.1641, abs=5e-4)
    assert LR.sign_test(5, 0) == LR.sign_test(0, 5) == pytest.approx(2 / 32)
    # The other artifact as the reference turns the sign.
    other = read(world, "paired_back", reference="species")
    back = other["paired"]["flags"]["splits"]["all"]["fine_nll"]
    assert back["diff"] == pytest.approx(-found["splits"]["all"]["fine_nll"]["diff"])
    # A fixed seed: the same reading twice.
    again = read(world, "paired_again")
    assert again["paired"] == reading["paired"]


def test_a_reading_is_not_written_over(world: dict[str, Any], capsys: Any):
    root = world["root"]
    read(world, "kept")
    before = (root / "kept" / LR.READ_JSON).read_text()
    with pytest.raises(FileExistsError, match="--overwrite"):
        read(world, "kept")
    assert (root / "kept" / LR.READ_JSON).read_text() == before
    read(world, "kept", overwrite=True, note="again")
    assert json.loads((root / "kept" / LR.READ_JSON).read_text())["note"] == "again"
    text = (root / "kept" / LR.READ_MD).read_text()
    assert "**again**" in text and "## Paired against flags" in text
    assert "| species | all |" in text and "sheet open" in text
    (root / "kept" / LR.READ_MD).unlink()
    assert LR.main(["--out", str(root / "kept"), "--render-only"]) == 0
    assert (root / "kept" / LR.READ_MD).read_text() == text
    # The command line: a taken directory fails by name, on the last line.
    arguments = ["--artifact", f"flags={world['flags']}", "--out", str(root / "kept")]
    arguments += ["--dir", str(root / FOLDER), "--resamples", "50"]
    assert LR.main(arguments) == 1
    assert (
        capsys.readouterr()
        .out.strip()
        .split("\n")[-1]
        .startswith("LADDER_READ_FAILED FileExistsError")
    )
    assert LR.main([*arguments, "--overwrite"]) == 0
    assert capsys.readouterr().out.strip().split("\n")[-1] == "LADDER_READ_DONE"
    assert LR.main(["--out", str(root / "nothing"), "--render-only"]) == 1


def test_bad_inputs_are_refused(world: dict[str, Any]):
    root = world["root"]
    with pytest.raises(LR.LadderReadError, match="label=path"):
        LR.run([str(world["flags"])], root / "bad", directories=[FOLDER], root=root)
    with pytest.raises(LR.LadderReadError, match="given twice"):
        LR.parse_artifacts(["a=x.pt", "a=y.pt"], root)
    with pytest.raises(LR.LadderReadError, match="no artifact"):
        LR.parse_artifacts([], root)
    with pytest.raises(LR.LadderReadError, match="not among"):
        read(world, "bad", reference="nobody")
    with pytest.raises(LR.LadderReadError, match="no game to score"):
        read(world, "bad", until="2020-01-01")
    with pytest.raises(LR.LadderReadError, match="artifact"):
        LR.run(
            [f"gone={root / 'gone.pt'}"],
            root / "bad",
            directories=[FOLDER],
            root=root,
            log=lambda text: None,
        )
    assert not (root / "bad").exists()


# --- the search session's matching rule -----------------------------------------


def test_lenient_hits_follow_the_search_sessions_rule():
    """Slot a acts alone: a hidden action matches anything, an uncertain target
    matches on the move, a move outside the candidates matches nothing."""
    batch = TS.blank(5)
    for i in range(5):
        TS.put(batch, i, 0, bench=(2,))
    TS.label_move(batch, 0, 0, 0, target=F.T_FOE_B)  # the attack, at foe b
    TS.label_move(batch, 1, 0, 0, target=-1)  # the attack, target not certain
    TS.label_stopped(batch, 2, 0, (0, TS.OTHER))  # hidden
    TS.label_move(batch, 3, 0, TS.OTHER, target=F.T_FOE_A)  # outside the candidates
    TS.label_switch(batch, 4, 0, 2)
    pred = TS.sharp(batch, dict.fromkeys(range(5), 1), strength=0.97)  # Protect
    truth = J.true_replies(batch)
    features = SC.features_only(batch)
    for k, expected in (
        # The Protect alone is listed: only the hidden slot matches.
        (1, [False, False, True, False, False]),
        # Everything possible is listed: all but the move nobody named.
        (64, [True, True, True, False, True]),
    ):
        made = J.joint_replies(pred, features, k=k, truth=truth)
        assert LR.lenient_hits(made, batch, truth).tolist() == expected, k
    # Strict counting needs the whole reply: the uncertain target is not counted.
    assert truth["visible"].tolist() == [True, False, False, True, True]
    # With the Mega bit an entry must name the observed Mega state.
    batch["mon_flag"][:, 0, F.FLAG_MEGA_POSSIBLE] = 1
    batch["y_mega"][:, 0] = 0
    batch["y_mega"][0, 0] = 1
    truth = J.true_replies(batch)
    assert truth["mega"].tolist() == [J.MEGA_A, 0, 0, 0, 0]
    pred["mega"][:] = 0.0  # no entry with a Mega Evolution carries probability
    made = J.joint_replies(pred, features, k=4, mega=True, truth=truth)
    hits = LR.lenient_hits(made, batch, truth).tolist()
    assert hits[0] is False and hits[2] is True
    # An example with nobody on the field is never a hit.
    empty = TS.blank(1)
    nobody = J.joint_replies(
        F.uniform_prediction(empty), SC.features_only(empty), k=2, truth=None
    )
    assert LR.lenient_hits(nobody, empty, J.true_replies(empty)).tolist() == [False]


# --- against the logged forecast ------------------------------------------------


def test_live_check_compares_with_the_logged_forecast(world: dict[str, Any]):
    root = world["root"]
    games, _ = LR.read_games(root, [root / FOLDER])
    loaded = A.load_predictor(world["flags"])
    scored = LR.score_artifact("flags", loaded, games)
    events = F.event_probs(scored.pred, scored.batch)
    logged: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in range(scored.n):
        game = games[int(scored.batch["m_battle"][row])]
        slots = [
            {
                "p_switch": round(float(events["switch"][row, slot]), 4),
                "p_protect": round(float(events["protect"][row, slot]), 4),
            }
            for slot in range(2)
        ]
        told = "open" if game.sheet == "open" else "closed"
        logged[(game.folder, game.battle_id, int(scored.batch["m_turn"][row]))] = {
            "model": loaded.name,
            "sheets": [told, told],
            "slots": slots,
        }
    found = LR.live_check(scored, games, logged, loaded.name)
    assert found["same_model_name"] and found["logged_model_names"] == {loaded.name: 6}
    assert set(found["by_sheet"]) == {"closed", "open", "unknown"}
    for cell in found["by_sheet"].values():
        assert cell["with_logged_forecast"] == cell["turns"]
        assert cell["slots_differing"] == 0 and cell["max_abs_diff"] < 1e-4
        assert cell["sheet_state_differs"] == cell["slot_presence_differs"] == 0
    assert found["by_sheet"]["closed"]["slots_compared"] == 4
    assert found["by_sheet"]["unknown"]["slots_compared"] == 4
    assert found["by_sheet"]["closed"]["actions_listed"] == 0
    # Another value, another sheet state, a missing slot and a missing turn show.
    first = (games[0].folder, games[0].battle_id, 1)
    logged[first]["slots"][0]["p_switch"] = 0.9
    logged[first]["sheets"] = ["open", "open"]
    logged[first]["slots"][1] = None
    del logged[(games[2].folder, games[2].battle_id, 2)]
    found = LR.live_check(scored, games, logged, loaded.name)
    cell = found["by_sheet"]["closed"]
    assert cell["slots_differing"] == 1 and cell["max_abs_diff"] > 0.5
    assert cell["sheet_state_differs"] == 1 and cell["slot_presence_differs"] == 1
    assert cell["with_logged_forecast"] == cell["turns"]
    later = found["by_sheet"]["unknown"]
    assert later["with_logged_forecast"] == later["turns"] - 1
    # A forecast of the same battle id in ANOTHER directory is never compared.
    moved = {("elsewhere", b, t): value for (_, b, t), value in logged.items()}
    assert not LR.live_check(scored, games, moved, loaded.name)["logged_model_names"]
    other = LR.live_check(scored, games, logged, "another_model")
    assert not other["same_model_name"]
    assert not LR.live_check(scored, games, {}, loaded.name)["same_model_name"]


# --- reviewers' findings, 2026-10-10 ---------------------------------------------


def quiet(text: str) -> None:
    del text


def test_a_coupled_artifact_is_ranked_with_its_coupling(world: dict[str, Any]):
    """The joint lists of an artifact that carries a pair coupling are built
    WITH it (as the runtime builds them), never as the plain product under
    the coupled artifact's name; the plain product is kept beside them."""
    root = world["root"]
    rng = np.random.default_rng(3)
    table = C.symmetrise(
        np.exp(rng.normal(0.0, 3.0, (C.N_BUCKET, C.N_CLASS, C.N_CLASS)))
    )
    pair = C.PairCoupling.build(table, name="unit_pair")
    path = save_table(root / "coupled.pt", root, T.FlagsTable, coupling=pair)
    games, _ = LR.read_games(root, [root / FOLDER])
    plain = LR.score_artifact("flags", A.load_predictor(world["flags"]), games)
    loaded = A.load_predictor(path)
    assert loaded.coupling is not None and not loaded.coupling.is_identity
    scored = LR.score_artifact("coupled", loaded, games)
    assert scored.coupling == "unit_pair" and plain.coupling == ""
    # The same predictor: the per-slot numbers do not read a coupling.
    assert np.array_equal(scored.nll["fine"], plain.nll["fine"])
    # The ranks are joint.py's own with the coupling ...
    truth = J.true_replies(scored.batch)
    features = SC.features_only(scored.batch)
    for variant, mega in ((LR.PLAIN, False), (LR.MEGA, True)):
        made = J.joint_replies(
            scored.pred,
            features,
            k=LR.TOP_K,
            mega=mega,
            truth=truth,
            coupling=loaded.coupling,
            move_intent=loaded.featurizer.tables.move_intent,
        )
        assert made.n == scored.n and made.coupled
        assert np.array_equal(scored.rank[variant], np.asarray(made.rank)), variant
    # ... which are not the plain product's; those are kept beside them.
    assert plain.rank_plain is None and scored.rank_plain is not None
    assert np.array_equal(scored.rank_plain, plain.rank[LR.PLAIN])
    assert (scored.rank[LR.PLAIN] != plain.rank[LR.PLAIN]).any()
    assert np.array_equal(scored.hit_plain(LR.TOP_K), plain.hit(LR.TOP_K))

    reading = LR.run(
        [f"flags={world['flags']}", f"coupled={path}"],
        root / "coupled_reading",
        directories=[FOLDER],
        root=root,
        resamples=100,
        log=quiet,
    )
    entry = reading["artifacts"]["coupled"]["artifact"]
    assert entry["joint"] == "coupled: unit_pair"
    assert entry["coupling"]["name"] == "unit_pair"
    assert not entry["coupling"]["is_identity"]
    assert reading["artifacts"]["flags"]["artifact"]["joint"] == LR.JOINT_PRODUCT
    visible = np.asarray(truth["visible"]).astype(bool)
    for name in ("all", "sheet closed", "turn 1"):
        own = reading["artifacts"]["coupled"]["splits"][name]
        base = reading["artifacts"]["flags"]["splits"][name]
        row, other = own["joint"]["replies"]["all"], base["joint"]["replies"]["all"]
        assert row["top_plain_product"] == other["top"], name
        assert row["only_coupled"] - row["only_plain_product"] == pytest.approx(
            (row["top"]["8"] - other["top"]["8"]) * row["examples"]
        )
        assert own["fine_nll"]["value"] == pytest.approx(base["fine_nll"]["value"])
    whole = reading["artifacts"]["coupled"]["splits"]["all"]["joint"]["replies"]["all"]
    for k in LR.TOP_KS:
        assert whole["top"][str(k)] == pytest.approx(scored.hit(k)[visible].mean())
    text = (root / "coupled_reading" / LR.READ_MD).read_text()
    assert "## What the pair coupling changes" in text
    assert "joint lists: coupled: unit_pair" in text
    assert f"joint lists: {LR.JOINT_PRODUCT}" in text
    # A reading without a coupled artifact has no such section.
    assert "pair coupling changes" not in (
        LR.render_markdown(read(world, "uncoupled_reading"))
    )
    # An all-ones coupling is the plain product, on the plain path, and said so.
    same = save_table(
        root / "identity.pt", root, T.FlagsTable, coupling=C.PairCoupling.identity()
    )
    ones = LR.score_artifact("identity", A.load_predictor(same), games)
    assert ones.coupling == "" and ones.rank_plain is None
    assert np.array_equal(ones.rank[LR.PLAIN], plain.rank[LR.PLAIN])


def two_rehearsals(root: Path) -> tuple[Path, Path]:
    """Two replay directories as two local rehearsals leave them: a local
    server restarts its battle numbers, so both hold ids 900 and 901 for
    DIFFERENT games. A's game 901 was cut off: rows (open, with a sheet and a
    logged forecast), no page."""
    first, second = root / "ladder_replays_mc_a", root / "ladder_replays_mc_b"
    first.mkdir(parents=True)
    second.mkdir()
    save_page(first, 900, TD.make_log("Foe A", BOT, exact="p2", ratings=(1100, 1200)))
    forecast = {
        "model": "m",
        "sheets": ["open", "open"],
        "sheets_reported": ["open", "open"],
        "slots": [None, None],
    }
    rows = [
        {"battle": tag(900), "turn": 1, "exact_search": {"open_sheet": False}},
        {"battle": tag(901), "turn": 0, "their_sheet": THEIR_SHEET},
        {
            "battle": tag(901),
            "turn": 1,
            "exact_search": {"open_sheet": True},
            "opponent_forecast": forecast,
        },
    ]
    (first / "decisions.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    save_page(second, 900, TD.make_log("Foe B", BOT, exact="p2", ratings=(1300, 1200)))
    save_page(second, 901, TD.make_log(BOT, "Foe C", exact="p1", ratings=(1200, 1400)))
    rows = [
        {"battle": tag(number), "turn": 1, "exact_search": {"open_sheet": False}}
        for number in (900, 901)
    ]
    (second / "decisions.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (root / "teams").mkdir(exist_ok=True)
    (root / TEAM).write_text(OWN_TEAM)
    for folder in (first, second):
        config = {"runs": [{"material": {"our_team": TEAM}}]}
        (folder / LR.RUN_CONFIG).write_text(json.dumps(config))
    return first, second


def test_a_page_is_joined_to_the_rows_of_its_own_directory(tmp_path: Path):
    first, second = two_rehearsals(tmp_path)
    one, two = TD.replay_id(900), TD.replay_id(901)

    def told(games: list[LR.Game]) -> list[tuple[Any, ...]]:
        return [
            (
                game.directory,
                game.battle_id,
                game.sheet,
                game.sheet_evidence,
                game.own_sheet,
                game.row()["opponent_rating"],
            )
            for game in games
        ]

    games, info = LR.read_games(tmp_path, [first, second])
    # B's game 901 is closed, as B's rows say: A's open sheet and A's forecast
    # belong to another game. Both pages numbered 900 are kept: two games.
    assert told(games) == [
        (first.name, one, "closed", "exact_search", False, 1100),
        (second.name, one, "closed", "exact_search", False, 1300),
        (second.name, two, "closed", "exact_search", False, 1400),
    ]
    assert [game.folder for game in games] == [str(first), str(second), str(second)]
    assert info["pages"] == 3 and info["kept"] == 3 and info["skipped"] == {}
    assert info["pages_found"] == {first.name: 1, second.name: 2}
    assert info["pages_unaccounted"] == 0 and info["notes"] == {}
    # A logged forecast is keyed by its directory.
    assert list(info["_forecasts"]) == [(str(first), two, 1)]
    alone, _ = LR.read_games(tmp_path, [second])
    assert told(alone) == told(games)[1:]
    # The same game saved in two directories is one game: the copy is skipped.
    third = tmp_path / "ladder_replays_mc_c"
    shutil.copytree(first, third)
    games, info = LR.read_games(tmp_path, [first, third, second])
    assert len(games) == 3 and info["pages"] == 4 and info["pages_unaccounted"] == 0
    assert info["skipped"] == {"duplicate:the same game in an earlier directory": 1}
    assert [game.directory for game in games] == [first.name, second.name, second.name]


def test_games_without_a_saved_page_are_counted(world: dict[str, Any], tmp_path: Path):
    """A game with decision rows and no page (cut off) cannot be scored, and
    the reading says how many there are and how many forecasts they hold."""
    first, second = two_rehearsals(tmp_path)
    _, info = LR.read_games(tmp_path, [first, second])
    assert info["no_page"] == {
        "games": 1,
        "decision_rows": 2,
        "logged_forecasts": 1,
        "by_directory": {
            first.name: {
                "games": 1,
                "decision_rows": 2,
                "logged_forecasts": 1,
                "battles": [tag(901)],
            }
        },
    }
    assert info["logged_forecasts"] == {
        "total": 1,
        "in_kept_games": 0,
        "in_games_without_a_page": 1,
        "in_other_games": 0,
    }
    said: list[str] = []
    LR.run(
        [f"flags={world['flags']}"],
        tmp_path / "out",
        directories=[first, second],
        root=tmp_path,
        resamples=50,
        log=said.append,
    )
    assert any("1 games with decision rows have no saved page" in s for s in said)
    text = (tmp_path / "out" / LR.READ_MD).read_text()
    assert "games with decision rows and no saved page: 1** (2 rows, 1 logged" in text
    assert "1 in games without a page" in text


def test_every_page_found_is_kept_or_counted(tmp_path: Path):
    """A page lost before the builder has a game (cut short while written, or
    another account's) is in ``skipped``; a file that is no page is counted."""
    write_directory(tmp_path)
    folder = tmp_path / FOLDER
    cut = folder / f"{BOT} - {tag(910)}.html"
    cut.write_text("<html><body>half a pa")
    stranger = folder / f"Somebody Else - {tag(911)}.html"
    stranger.write_text(TD.page(TD.make_log("Foe Seven", BOT, exact="p2")))
    (folder / "notes.html").write_text("<html></html>")
    (folder / f"{BOT} - battle-gen9randombattle-77.html").write_text("<html></html>")
    games, info = LR.read_games(tmp_path, [folder])
    assert len(games) == 3 and info["pages"] == 8
    assert info["files"][FOLDER] == {
        "html_files": 10,
        "not_a_saved_game_page": 1,
        "another_format": 1,
        "pages": 8,
    }
    assert info["skipped"] == {
        "builder:illusion_species": 1,
        "builder:no_turn_1": 1,
        "not_ladder_rated": 1,
        "page:account_is_not_a_player": 1,
        "page:no_log_block": 1,
    }
    assert info["kept"] + sum(info["skipped"].values()) == info["pages"]
    assert info["pages_unaccounted"] == 0


def test_a_directory_without_sheet_logging_is_unknown_not_closed(tmp_path: Path):
    """Rows that say nothing about sheets leave the game unknown: about one
    ladder game in ten is open, so such a directory's split must not be
    called closed."""
    write_directory(tmp_path)
    folder = tmp_path / FOLDER
    rows = [
        {"battle": tag(number), "turn": 1, "candidates": [], "chosen": 0, "guards": []}
        for number in (CLOSED, OPEN, SECOND)
    ]
    (folder / "decisions.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (folder / "decisions_champion.jsonl").unlink()
    games, info = LR.read_games(tmp_path, [folder])
    assert [game.sheet for game in games] == ["unknown"] * 3
    assert {game.sheet_evidence for game in games} == {"rows_say_nothing"}
    assert info["own_sheet"]["open_games_kept"] == 0
    # Without any row at all: unknown too, with nothing as its evidence.
    (folder / "decisions.jsonl").unlink()
    games, _ = LR.read_games(tmp_path, [folder])
    assert [(game.sheet, game.sheet_evidence) for game in games] == [
        ("unknown", "")
    ] * 3


def test_search_decisions_are_the_ones_a_search_session_counts(
    world: dict[str, Any], tmp_path: Path
):
    """``search_ladder_read.py`` section 5, row for row: an audit row with an
    observed reply answers the game's last earlier MOVE decision and is
    counted when that decision's forecast is in the bot's own log."""
    folder = tmp_path / "ladder_replays_mc_search"
    folder.mkdir()
    seen = {"reply_coverage": {"observed": {"0": ["move", "x", 1, False]}}}

    def audit(battle: int, turn: int, actions: list[int], reply: bool) -> dict:
        body: dict[str, Any] = {"champion_actions": actions}
        return {
            "battle": f"battle-fmt-{battle}",
            "turn": turn,
            "exact_search": {**body, **(seen if reply else {})},
        }

    rows = [
        audit(1, 1, [8, 9], False),  # the first move decision: nothing before it
        audit(1, 2, [8, 9], True),  # the reply to turn 1: counted
        audit(1, 9, [0, 3], True),  # a forced row, the reply to turn 2: counted
        # The forced row is no move decision, whatever turn it carries: this
        # is the reply to turn 2 again, counted again.
        audit(1, 3, [8, 9], True),
        audit(1, 4, [8, 9], True),  # the reply to turn 3, whose forecast stood down
        audit(2, 1, [0, 3], True),  # a reply before any move decision of the game
        audit(2, 2, [8, 9], False),
    ]
    (folder / LR.SEARCH_AUDIT_LOG).write_text("\n".join(json.dumps(r) for r in rows))
    own: list[dict[str, Any]] = [
        {"battle": "battle-fmt-1", "turn": turn, "opponent_forecast": {"slots": []}}
        for turn in (1, 2)
    ]
    own.append(
        {"battle": "battle-fmt-1", "turn": 3, "opponent_forecast": {"stand_down": "x"}}
    )
    (folder / LR.OWN_DECISION_LOG).write_text("\n".join(json.dumps(r) for r in own))
    logs = LR.read_directory_logs(folder)
    counted, replies = LR.search_decisions(logs)
    assert dict(counted) == {("fmt-1", 1): 1, ("fmt-1", 2): 2} and replies == 4
    # A forecast in the search's own audit file is not the session's forecast.
    (folder / LR.OWN_DECISION_LOG).unlink()
    assert not LR.search_decisions(LR.read_directory_logs(folder))[0]

    # In a reading: the rule on exactly those decisions, next to the rule on
    # every turn with an action shown (which is NOT the search session's set).
    root = tmp_path / "world"
    write_directory(root)
    folder = root / FOLDER
    more = [
        {
            "battle": tag(CLOSED),
            "turn": 1,
            "exact_search": {"champion_actions": [8, 9]},
        },
        {
            "battle": tag(CLOSED),
            "turn": 2,
            "exact_search": {"champion_actions": [8, 9], "open_sheet": False, **seen},
        },
    ]
    (folder / LR.SEARCH_AUDIT_LOG).write_text("\n".join(json.dumps(r) for r in more))
    told = {"battle": tag(CLOSED), "turn": 1, "opponent_forecast": {"slots": []}}
    with (folder / LR.OWN_DECISION_LOG).open("a") as handle:
        handle.write("\n" + json.dumps(told))
    games, info = LR.read_games(root, [folder])
    assert info["search_session"] == {"replies": 1, "decisions": 1, "in_kept_games": 1}
    assert dict(info["_search"]) == {(str(folder), TD.replay_id(CLOSED), 1): 1}
    loaded = A.load_predictor(world["flags"])
    scored = LR.score_artifact("flags", loaded, games, info["_search"])
    assert scored.search is not None and scored.search.tolist() == [1, 0, 0, 0, 0, 0]
    reading = LR.run(
        [f"flags={world['flags']}"],
        root / "searched",
        directories=[FOLDER],
        root=root,
        resamples=50,
        log=quiet,
    )
    found = reading["artifacts"]["flags"]
    assert found["search_decisions_joined"] == 1
    for variant in (LR.PLAIN, LR.MEGA):
        hit = int(scored.lenient[variant][0])
        for name, turns in (("all", 1), ("sheet closed", 1), ("sheet open", 0)):
            cell = found["splits"][name]["joint"]["search_rule"][variant]
            assert cell[LR.RULE_SEARCH]["turns"] == turns, name
            assert cell[LR.RULE_SEARCH]["hits"] == (hit if turns else 0)
        whole = found["splits"]["all"]["joint"]["search_rule"][variant]
        assert whole[LR.RULE_ALL_ANY]["turns"] > whole[LR.RULE_SEARCH]["turns"]
    text = (root / "searched" / LR.READ_MD).read_text()
    assert "under a search session's matching rule" in text
    assert "counted as a search session counts it" not in text
    assert "counts 1 decisions in these directories" in text


def test_the_sealed_set_is_read_only_with_fresh(world: dict[str, Any], tmp_path: Path):
    """No default set of games, and a game dated on or after ``FRESH_SINCE``
    is confirmation data: refused before anything is scored unless asked for."""
    flags = [f"flags={world['flags']}"]
    with pytest.raises(LR.LadderReadError, match="--dir"):
        LR.run(flags, tmp_path / "none", root=world["root"], log=quiet)
    assert not (tmp_path / "none").exists()
    root = tmp_path / "sealed"
    write_directory(root)
    new = root / "ladder_replays_mc_new"
    new.mkdir()
    sealed_at = time.mktime(time.strptime(f"{LR.FRESH_SINCE} 09:00", "%Y-%m-%d %H:%M"))
    save_page(new, 950, TD.make_log("Foe New", BOT, exact="p2"), when=sealed_at)
    save_page(new, 951, TD.make_log("Foe Old", BOT, exact="p2"), when=PLAYED_AT)
    both = [FOLDER, new.name]
    with pytest.raises(LR.LadderReadError, match="sealed confirmation set"):
        LR.run(flags, root / "refused", directories=both, root=root, log=quiet)
    assert not (root / "refused").exists()
    # Left out by date: an ordinary reading of the older games.
    reading = LR.run(
        flags,
        root / "older",
        directories=both,
        until=LR.FRESH_SINCE,
        root=root,
        resamples=50,
        log=quiet,
    )
    assert reading["input"]["kept"] == 4 and reading["input"]["sealed"]["games"] == 0
    # Asked for by name.
    reading = LR.run(
        flags,
        root / "named",
        directories=[new.name],
        fresh=True,
        root=root,
        resamples=50,
        log=quiet,
    )
    assert reading["input"]["kept"] == 2
    assert reading["input"]["sealed"] == {
        "since": LR.FRESH_SINCE,
        "games": 1,
        "opened_with_fresh": True,
        "directories_found": None,
    }
    # Asked for as a set: the directories are found, older pages are left out,
    # and a directory added later is found too (no fixed list).
    assert LR.fresh_directories(root) == [new]
    said: list[str] = []
    reading = LR.run(
        flags, root / "found", fresh=True, root=root, resamples=50, log=said.append
    )
    assert said[0].startswith("SEALED confirmation set (--fresh): 1 directories")
    assert reading["input"]["directories"] == [str(new)]
    assert reading["input"]["kept"] == 1
    assert reading["input"]["skipped"] == {"filter:before --since": 1}
    assert reading["input"]["filters"]["since"] == LR.FRESH_SINCE
    assert reading["input"]["sealed"]["directories_found"] == [str(new)]
    assert "**SEALED confirmation set**: 1 of the 1 kept games" in (
        (root / "found" / LR.READ_MD).read_text()
    )
    later = root / "ladder_replays_mc_later"
    later.mkdir()
    save_page(later, 960, TD.make_log("Foe Later", BOT, exact="p2"), when=sealed_at)
    assert LR.fresh_directories(root) == [later, new]
    # Looked for somewhere else; nothing there is an error, not an empty reading.
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(LR.LadderReadError, match="--fresh: no ladder_replays_mc"):
        LR.run(flags, root / "nowhere", fresh=True, fresh_root=empty, root=root)
    # The command line without --dir or --fresh reads nothing.
    assert LR.main(["--artifact", flags[0], "--out", str(root / "cli")]) == 1
    assert not (root / "cli").exists()


def sealed_time(clock: str = "09:00", days: int = 0) -> float:
    """A time on the sealed set's first day (or ``days`` later)."""
    start = time.mktime(time.strptime(f"{LR.FRESH_SINCE} {clock}", "%Y-%m-%d %H:%M"))
    return start + days * 86400


def test_a_direct_read_of_sealed_pages_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """``read_games`` itself is the guard, not only ``run``: a script that
    calls it on a page dated on or after ``FRESH_SINCE`` is refused, and that
    page is never driven (no state and no label is made of it)."""
    root = tmp_path / "sealed"
    write_directory(root)
    new = root / "ladder_replays_mc_new"
    new.mkdir()
    save_page(new, 950, TD.make_log("Foe New", BOT, exact="p2"), when=sealed_time())
    save_page(new, 951, TD.make_log("Foe Old", BOT, exact="p2"), when=PLAYED_AT)
    driven: list[str] = []
    real = LR.drive

    def watched(raw: Any) -> Any:
        driven.append(raw.battle_id)
        return real(raw)

    monkeypatch.setattr(LR, "drive", watched)
    sealed, older = TD.replay_id(950), TD.replay_id(951)
    # THE FAILING CASE: the direct call the first probe of 2026-10-10 made.
    with pytest.raises(LR.LadderReadError, match="sealed confirmation set"):
        LR.read_games(root, [root / FOLDER, new])
    assert sealed not in driven and older in driven
    with pytest.raises(LR.LadderReadError, match="1 of the pages .*fresh=True"):
        LR.read_games(root, [new])
    # The first second of the day is sealed; the second before it is not.
    edge = root / "ladder_replays_mc_edge"
    edge.mkdir()
    page = save_page(edge, 952, TD.make_log("Foe Edge", BOT, exact="p2"))
    midnight = sealed_time("00:00")
    os.utime(page, (midnight, midnight))
    with pytest.raises(LR.LadderReadError, match="sealed confirmation set"):
        LR.read_games(root, [edge])
    os.utime(page, (midnight - 1, midnight - 1))
    assert len(LR.read_games(root, [edge])[0]) == 1
    # Left out by date: an ordinary reading of the older games.
    driven.clear()
    games, info = LR.read_games(root, [root / FOLDER, new], until=LR.FRESH_SINCE)
    assert len(games) == 4 and sealed not in driven
    assert info["skipped"]["filter:at or after --until"] == 1
    # A sealed page that a filter drops is not read, so nothing is refused.
    only, _ = LR.read_games(root, [new], games=tag(951))
    assert [game.battle_id for game in only] == [older] and sealed not in driven
    # Asked for: read.
    games, info = LR.read_games(root, [new], fresh=True)
    assert [game.battle_id for game in games] == [sealed, older]
    assert sealed in driven and info["kept"] == 2


def test_one_day_of_the_sealed_set_is_read_with_until(
    world: dict[str, Any], tmp_path: Path
):
    """``--fresh --until <the next day>`` is the confirmation on the first
    sealed day alone; the help names the exact option."""
    root = tmp_path / "days"
    write_directory(root)
    new = root / "ladder_replays_mc_new"
    new.mkdir()
    save_page(
        new, 950, TD.make_log("Foe Day One", BOT, exact="p2"), sealed_time("21:00")
    )
    save_page(
        new, 951, TD.make_log("Foe Day Two", BOT, exact="p2"), sealed_time("02:00", 1)
    )
    save_page(new, 952, TD.make_log("Foe Before", BOT, exact="p2"), PLAYED_AT)
    next_day = time.strftime("%Y-%m-%d", time.localtime(sealed_time("12:00", 1)))
    assert LR.FRESH_FIRST_DAY_UNTIL == next_day == LR._day_after(LR.FRESH_SINCE)
    flags = [f"flags={world['flags']}"]
    reading = LR.run(
        flags,
        root / "first_day",
        fresh=True,
        until=next_day,
        root=root,
        resamples=50,
        log=quiet,
    )
    assert [row["battle"] for row in reading["games"]] == [tag(950)]
    assert reading["games"][0]["date"] == LR.FRESH_SINCE
    assert reading["input"]["skipped"] == {
        "filter:at or after --until": 1,
        "filter:before --since": 1,
    }
    assert reading["input"]["filters"] == {
        "since": LR.FRESH_SINCE,
        "until": next_day,
        "games": None,
    }
    assert reading["input"]["sealed"]["games"] == reading["input"]["kept"] == 1
    # Without --until: both sealed days.
    both = LR.run(flags, root / "both", fresh=True, root=root, resamples=50, log=quiet)
    assert [row["battle"] for row in both["games"]] == [tag(950), tag(951)]
    # The day alone is not readable without --fresh, whatever the filters.
    with pytest.raises(LR.LadderReadError, match="sealed confirmation set"):
        LR.run(
            flags,
            root / "refused",
            directories=[new.name],
            since=LR.FRESH_SINCE,
            until=next_day,
            root=root,
            log=quiet,
        )
    assert not (root / "refused").exists()
    assert f"--until {next_day}" in LR.HELP_EPILOG
    args = LR.parse_args(["--fresh", "--until", next_day])
    assert (args.fresh, args.until, args.since) == (True, next_day, None)


def test_set_table_contributors_are_counted_and_warned_about(world: dict[str, Any]):
    """An artifact whose set table was built with a scored opponent's own
    sheets is a leak: counted per artifact and per game, and a warning line
    at the top of the report. Never a crash."""
    root = world["root"]
    games, _ = LR.read_games(root, [root / FOLDER])
    opponents = [str(game.result.player_ids[game.opponent]) for game in games]
    assert opponents == [E.user_id(name) for name in ("Foe One", "Foe Two", "Foe Six")]
    base = featurizer_of(root)

    def with_table(name: str, accounts: list[int] | None) -> str:
        """A count table whose featurizer holds a set table said to be built
        from the sheets of ``accounts`` (crc32 of account ids)."""
        fz = F.Featurizer.build(
            base.repertoire,
            base.n_cand,
            layout_version=2,
            set_table=TSP.table(),
            set_accounts=accounts,
        )
        batch = dict(LR.encode_games(games, fz))
        batch["m_weight"] = np.ones(len(batch["turn"]), dtype=np.float32)
        table = T.FlagsTable.fit(batch, featurizer=fz)
        A.save_artifact(
            root / f"{name}.pt",
            kind=A.KIND_TABLE,
            name=table.name,
            featurizer=fz,
            predictor_payload=table.to_payload(),
            extra={"dataset_tag": "unit"},
        )
        return f"{name}={root / f'{name}.pt'}"

    stranger = B.account_hash(E.user_id("Somebody Else"))
    specs = [
        f"flags={world['flags']}",
        with_table("leaky", [B.account_hash(opponents[0]), stranger]),
        with_table("clean", [stranger]),
        with_table("blind", None),
    ]
    said: list[str] = []
    out = root / "set_table"
    reading = LR.run(
        specs, out, directories=[FOLDER], root=root, resamples=50, log=said.append
    )
    found = {
        label: reading["artifacts"][label]["artifact"]["set_table"]
        for label in reading["order"]
    }
    assert found["flags"] is None  # no set table: nothing to ask
    assert found["leaky"]["games"] == {
        "contributors": 1,
        "not_contributors": 2,
        "unknown": 0,
    }
    assert found["leaky"]["opponent_accounts"]["contributors"] == 1
    assert found["leaky"]["leak"] is True and found["leaky"]["accounts_recorded"] == 2
    assert found["leaky"]["contributor_battles"] == [tag(CLOSED)]
    assert found["leaky"]["signature"] == TSP.table().signature()
    assert found["clean"]["games"] == {
        "contributors": 0,
        "not_contributors": 3,
        "unknown": 0,
    }
    assert found["clean"]["leak"] is False and not found["clean"]["contributor_battles"]
    assert found["blind"]["games"] == {
        "contributors": 0,
        "not_contributors": 0,
        "unknown": 3,
    }
    assert found["blind"]["accounts_recorded"] is None
    assert "does not record" in found["blind"]["why_unknown"]
    rows = {row["battle"]: row for row in reading["games"]}
    assert rows[tag(CLOSED)]["set_table_holds"] == {
        "leaky": True,
        "clean": False,
        "blind": None,
    }
    assert rows[tag(OPEN)]["set_table_holds"]["leaky"] is False
    # The warning: logged, stored, and the first thing the report says.
    assert len(reading["warnings"]) == 1
    assert reading["warnings"][0].startswith("WARNING: SET-TABLE LEAK in leaky: ")
    assert "1 of the 3 scored games (1 accounts)" in reading["warnings"][0]
    assert sum(line.startswith("WARNING: SET-TABLE LEAK") for line in said) == 1
    lines = (out / LR.READ_MD).read_text().splitlines()
    assert lines[2].startswith("**WARNING: SET-TABLE LEAK in leaky: ")
    text = "\n".join(lines)
    assert "set table: the opponents of 1 scored games are contributors" in text
    assert f"contributors' games: {tag(CLOSED)}" in text
    assert "0 scored games are contributors (a leak), of 3 are not, 0 unknown" in text
    assert "0 are not, 3 unknown (the set table does not record" in text
    assert "**set_table**" in text
    # The numbers are the reading's own: the leak changes none of them here
    # (a count table reads no set prior), and the run still ends.
    assert reading["artifacts"]["leaky"]["splits"]["all"]["slot_turns"] > 0
    # A reading without such an artifact warns of nothing, and one of an
    # earlier version (no entry at all) still renders.
    plain = read(world, "no_set_table")
    assert plain["warnings"] == [] and "WARNING" not in LR.render_markdown(plain)
    earlier = {key: value for key, value in reading.items() if key != "warnings"}
    for label in earlier["order"]:
        earlier["artifacts"][label]["artifact"].pop("set_table")
    assert "SET-TABLE" not in LR.render_markdown(earlier)
    # A page that names no opponent account is unknown, not a crash.
    nameless = [dataclass_copy(game) for game in games]
    nameless[0].result.player_ids[nameless[0].opponent] = None
    loaded = A.load_predictor(root / "leaky.pt")
    summary, answers = LR.set_table_check(loaded, nameless)
    assert answers == [None, False, False] and summary is not None
    assert summary["games_without_an_opponent_account"] == 1
    assert summary["games"]["unknown"] == 1 and summary["leak"] is False
    assert LR.set_table_check(A.load_predictor(world["flags"]), games) == (None, [])
    assert LR.set_table_warning("x", None) == "" == LR.set_table_warning("x", summary)


def dataclass_copy(game: LR.Game) -> LR.Game:
    """A game whose driven result can be changed without touching the fixture's."""
    result = copy.copy(game.result)
    result.player_ids = dict(game.result.player_ids)
    return dataclasses.replace(game, result=result)


def test_live_check_compares_every_listed_action(world: dict[str, Any]):
    """The logged forecast's ranked list is compared entry by entry, by move
    and target (a wrong candidate-to-move mapping would leave P(switch) and
    P(Protect) unchanged)."""
    root = world["root"]
    games, _ = LR.read_games(root, [root / FOLDER])
    loaded = A.load_predictor(world["flags"])
    scored = LR.score_artifact("flags", loaded, games)
    norm = F.normalize_prediction(scored.pred, scored.batch)
    names = loaded.featurizer.tables.move_ids
    n_cand = loaded.featurizer.n_cand
    logged: dict[tuple[str, str, int], dict[str, Any]] = {}
    listed = 0
    for row in range(scored.n):
        game = games[int(scored.batch["m_battle"][row])]
        slots: list[dict[str, Any]] = []
        for slot in range(2):
            action, target = norm["action"][row, slot], norm["target"][row, slot]
            entries: list[dict[str, Any]] = []
            unnamed = float(action[n_cand])
            for column in range(n_cand):
                move = int(scored.batch["cand_move"][row, slot, column])
                if not scored.batch["cand_flag"][row, slot, column] & F.CAND_VALID:
                    unnamed += float(action[column])
                    continue
                for place, klass in enumerate(E.TARGET_CLASSES):
                    share = float(action[column] * target[column, place])
                    if share > 0:
                        entries.append(
                            {
                                "kind": "move",
                                "p": round(share, 4),
                                "move": names[move],
                                "target": klass,
                            }
                        )
            entries.append({"kind": "other", "p": round(unnamed, 4)})
            for pointer in range(F.N_ROSTER):
                mass = float(action[n_cand + 1 + pointer])
                if mass > 0:
                    entries.append(
                        {"kind": "switch", "p": round(mass, 4), "roster_index": pointer}
                    )
            listed += len(entries)
            slots.append({"p_switch": None, "p_protect": None, "actions": entries})
        key = (game.folder, game.battle_id, int(scored.batch["m_turn"][row]))
        logged[key] = {"model": loaded.name, "sheets": [], "slots": slots}
    found = LR.live_check(scored, games, logged, loaded.name)
    cells = found["by_sheet"].values()
    assert sum(cell["actions_listed"] for cell in cells) == listed > 30
    assert sum(cell["actions_compared"] for cell in cells) == listed
    assert sum(cell["actions_differing"] for cell in cells) == 0
    assert max(cell["actions_max_abs_diff"] for cell in cells) < 1e-4
    # A list written for other candidates (two moves' probabilities exchanged)
    # shows, and so does an entry naming a move the slot's candidates lack.
    first = logged[(games[0].folder, games[0].battle_id, 1)]["slots"][0]["actions"]
    moves = [entry for entry in first if entry["kind"] == "move"]
    high = max(moves, key=lambda entry: entry["p"])
    low = min(moves, key=lambda entry: entry["p"])
    assert high["p"] - low["p"] > 0.01
    high["p"], low["p"] = low["p"], high["p"]
    first.append({"kind": "move", "p": 0.5, "move": "nosuchmove", "target": "auto"})
    found = LR.live_check(scored, games, logged, loaded.name)
    cell = found["by_sheet"][games[0].sheet]
    assert cell["actions_differing"] == 2 and cell["actions_max_abs_diff"] > 0.01
    assert cell["actions_not_among_the_candidates"] == 1
    assert cell["slots_differing"] == 0  # the two aggregates alone would pass


# --- house rules ----------------------------------------------------------------


def test_script_names_nothing_from_the_dex_and_has_no_bare_assert():
    source = (ROOT / "evaluation/oppmodel_ladder_read.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    known = set(E.moves_dex()) | set(E.pokedex())
    found = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        and node.value in known
    }
    assert not found, sorted(found)
    # No default set of games: the sealed set is never read by forgetting --dir.
    assert not hasattr(LR, "DEFAULT_DIRS")
    assert LR.parse_args([]).dir is None and LR.parse_args([]).fresh is False
    assert LR.DEFAULT_RESAMPLES == 2000 and LR.TOP_K == 8


# --- the repo's own pages (skipped when absent) ---------------------------------


def test_real_saved_pages_score_like_the_built_dataset(tmp_path: Path):
    """Old ladder pages (never the fresh set): the table's NLL per game here is
    the one the built dataset gives for the same games."""
    folder = ROOT / "ladder_replays_mc_guards3_20260909"
    artifact = ROOT / "results_oppmodel/tables_v2/flags_table.pt"
    dataset = ROOT / "results_oppmodel/v2_feed"
    if not (folder.is_dir() and artifact.is_file() and dataset.is_dir()):
        pytest.skip("needs the saved replay folders and results_oppmodel/")
    reading = LR.run(
        [f"table={artifact}"],
        tmp_path / "real",
        directories=[folder],
        resamples=100,
        log=lambda text: None,
    )
    assert reading["input"]["kept"] >= 20
    batch, _ = F.load_dataset(dataset, ["ladder_holdout"])
    ids = {}
    for line in (dataset / "battles.jsonl").read_text().splitlines():
        row = json.loads(line)
        ids[f"battle-{row['id']}"] = row["index"]
    loaded = A.load_predictor(artifact)
    pred, _ = SC.guarded_predict(loaded.predictor, batch, name="table")
    nll = F.slot_nll(pred, batch)
    compared = 0
    for row in reading["games"]:
        rows = batch["m_battle"] == ids[row["battle"]]
        labelled = nll["fine_scored"][rows]
        cell = row["artifacts"]["table"]
        assert cell["slot_turns"] == int(labelled.sum()), row["battle"]
        assert cell["nll_sum"] == pytest.approx(
            float(np.where(labelled, nll["fine"][rows], 0.0).sum()), abs=1e-6
        )
        compared += 1
    assert compared == reading["input"]["kept"]
