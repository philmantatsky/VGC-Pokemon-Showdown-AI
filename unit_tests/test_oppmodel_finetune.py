"""Fine-tune on own ladder games: folds, the allowed list, the date seal,
exactness, what the rule does not gate, the artifact and its own record.

A tiny dataset is written under a temporary directory from the hand-written
game of ``test_oppmodel_model``: eight "train" games, four "val" games, one
"test" game and eight ladder-holdout games of which six are ALLOWED (named in
the old-games file, dated before the seal) and two are not (they stand for
the sealed games of a newer build and are dated on the first sealed day). A
tiny network is trained on it by the trainer, and the fine-tune runs on that
artifact.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from evaluation import oppmodel_scorecard as SC
from training import finetune_oppmodel as FT
from training import train_oppmodel as T
from unit_tests import test_oppmodel_model as TM
from vgc_bench.src.oppmodel import artifact as A
from vgc_bench.src.oppmodel import features as F
from vgc_bench.src.oppmodel import model as M

SPLITS = ["train", "val", "test", "ladder_holdout"]
TINY = ["--d-model", "32", "--layers", "1", "--heads", "2", "--ff", "48"]
N_TRAIN, N_VAL, N_ALLOWED, N_SEALED = 8, 4, 6, 2
ROWS = 7  # turns of the hand-written game, one actor side


class Tiny:
    """Paths and facts of the temporary dataset and its trained artifact."""

    dataset: Path
    old_games: Path
    artifact: Path
    allowed_index: set[int]
    sealed_index: set[int]
    test_index: set[int]


def _write_dataset(directory: Path, fz: F.Featurizer) -> Tiny:
    directory.mkdir(parents=True, exist_ok=True)
    mine, theirs = TM.game_batch(fz, ("p2",)), TM.game_batch(fz, ("p1",))
    views = {
        "mine": [
            mine,
            M.swap_slots(mine, True, False),
            M.swap_slots(mine, False, True),
            M.swap_slots(mine, True, True),
        ],
        "theirs": [theirs, M.swap_slots(theirs, True, True)],
    }
    rows: list[F.Batch] = []
    battles: list[dict[str, Any]] = []
    made = Tiny()
    made.allowed_index, made.sealed_index, made.test_index = set(), set(), set()

    def add(source: F.Batch, split: str, ident: str) -> int:
        index = len(battles)
        part = dict(source)
        n = int(part["act_mon"].shape[0])
        part["m_split"] = np.full(n, SPLITS.index(split), dtype=np.uint8)
        part["m_battle"] = np.full(n, index, dtype=np.int32)
        part["m_weight"] = np.linspace(0.5, 1.0, n).astype(np.float32)
        part["m_flag"] = np.zeros(n, dtype=np.uint8)
        part["m_actor"] = np.full(n, 1000 + index // 2, dtype=np.uint32)
        rows.append(part)
        battles.append(
            {
                "index": index,
                "id": ident,
                "split": {"p1": split},
                "time": 1_789_000_000 + 3600 * index,  # 2026-09-09/10: old
            }
        )
        return index

    for game in range(N_TRAIN):
        add(views["mine"][game % 4], "train", f"train-{game}")
    for game in range(N_VAL):
        add(views["theirs"][game % 2], "val", f"val-{game}")
    made.test_index.add(add(theirs, "test", "test-0"))
    for game in range(N_ALLOWED + N_SEALED):
        # allowed and sealed games alternate inside the split, as they would
        # inside the shards of a newer build
        sealed = game in (2, 5)
        ident = f"sealed-{game}" if sealed else f"old-{game}"
        index = add(views["mine"][game % 4], "ladder_holdout", ident)
        (made.sealed_index if sealed else made.allowed_index).add(index)
        if sealed:  # played on the first sealed day
            battles[index]["time"] = int(FT.sealed_start()) + 3600 * game
    data = F.concat_batches(rows)
    F.save_batch(directory / "shard-00000.npz", data)
    manifest = {
        "tag": "tiny",
        "splits": SPLITS,
        "shards": [{"file": "shard-00000.npz", "examples": int(data["turn"].shape[0])}],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "battles.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in battles), encoding="utf-8"
    )
    fz.save(directory)
    made.dataset = directory
    # The old build's battles file: the same ids under OTHER row numbers, and
    # one game of another split, which must not become allowed.
    old = [{"index": 0, "id": "train-0", "split": {"p1": "train", "p2": "val"}}]
    old += [
        {"index": 1 + place, "id": battles[index]["id"], "split": {"p2": SPLITS[3]}}
        for place, index in enumerate(sorted(made.allowed_index))
    ]
    made.old_games = directory.parent / "old_battles.jsonl"
    made.old_games.write_text(
        "".join(json.dumps(row) + "\n" for row in old), encoding="utf-8"
    )
    return made


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> Tiny:
    root = tmp_path_factory.mktemp("finetune")
    fz = F.Featurizer.build(TM.repertoire())
    made = _write_dataset(root / "data", fz)
    out = root / "source"
    argv = ["--dataset", str(made.dataset), "--out", str(out), "--batch", "16"]
    argv += ["--threads", "1", "--warmup", "2", "--seed", "3", "--epochs", "3", *TINY]
    assert T.main(argv) == 0
    made.artifact = out / T.ARTIFACT_NAME
    return made


def arguments(tiny: Tiny, out: Path, *extra: str) -> list[str]:
    argv = ["--artifact", str(tiny.artifact), "--dataset", str(tiny.dataset)]
    argv += ["--old-games", str(tiny.old_games), "--out", str(out)]
    argv += ["--folds", "3", "--batch", "10", "--corpus-rows", "40", "--threads", "1"]
    argv += ["--resamples", "50", "--seed", "11"]
    return [*argv, *extra]


def prepared(tiny: Tiny) -> tuple[Any, FT.Parts, FT.Recipe]:
    """(the loaded source, the parts of the dataset, the recipe) as a run has them."""
    loaded = A.load_predictor(tiny.artifact)
    parts = FT.load_parts(tiny.dataset, FT.allowed_games(tiny.old_games), 40, 11)
    recipe = FT.Recipe.from_training(
        loaded.meta["extra"].get("args"),
        batch=10,
        elo_blind=loaded.predictor.elo_mode == F.ELO_BLANK,
    )
    return loaded, parts, recipe


def same_prediction(first: FT.Prediction, second: FT.Prediction) -> bool:
    return all(np.array_equal(first[name], second[name]) for name in first)


# --- folds ----------------------------------------------------------------------


def test_fold_split_never_puts_one_game_on_both_sides():
    rng = np.random.default_rng(0)
    games = rng.integers(100, 157, size=600)  # 57 games, several rows each
    assignment = FT.game_folds(games, 5, seed=4)
    row_fold = FT.fold_of_rows(games, assignment)
    assert set(assignment) == set(np.unique(games).tolist())
    assert set(assignment.values()) == set(range(5))
    sizes = [sum(1 for fold in assignment.values() if fold == k) for k in range(5)]
    assert max(sizes) - min(sizes) <= 1 and sum(sizes) == len(assignment)
    seen: list[int] = []
    for fold in range(5):
        held = row_fold == fold
        train, test = set(games[~held].tolist()), set(games[held].tolist())
        assert train and test and not train & test
        FT.check_disjoint(games[~held], games[held])
        seen += sorted(test)
    assert sorted(seen) == sorted(assignment)  # every game held out exactly once
    assert FT.game_folds(games, 5, seed=4) == assignment
    assert FT.game_folds(games[::-1], 5, seed=4) == assignment  # row order is no input
    assert FT.game_folds(games, 5, seed=5) != assignment

    with pytest.raises(FT.FinetuneError):
        FT.check_disjoint(np.array([1, 2, 3]), np.array([3, 4]))
    with pytest.raises(FT.FinetuneError):
        FT.game_folds(games, 1, seed=0)
    with pytest.raises(FT.FinetuneError):
        FT.game_folds(np.array([1, 1, 2]), 3, seed=0)


def test_folds_by_account_keep_every_account_on_one_side():
    rng = np.random.default_rng(1)
    games = np.repeat(np.arange(200, 240), 5)  # 40 games, five rows each
    owner = {int(game): int(rng.integers(0, 23)) for game in np.unique(games)}
    accounts = np.asarray([owner[int(game)] for game in games])
    assignment = FT.account_folds(games, accounts, 5, seed=4)
    assert set(assignment) == set(owner) and set(assignment.values()) == set(range(5))
    row_fold = FT.fold_of_rows(games, assignment)
    for account in np.unique(accounts):
        assert np.unique(row_fold[accounts == account]).size == 1, account
    for fold in range(5):
        held = row_fold == fold
        FT.check_disjoint(games[~held], games[held])
        assert not set(accounts[held].tolist()) & set(accounts[~held].tolist())
    per_fold = [
        len({owner[game] for game, place in assignment.items() if place == fold})
        for fold in range(5)
    ]
    assert max(per_fold) - min(per_fold) <= 1 and sum(per_fold) == len(set(accounts))
    assert FT.account_folds(games, accounts, 5, seed=4) == assignment
    assert FT.account_folds(games[::-1], accounts[::-1], 5, seed=4) == assignment
    assert FT.account_folds(games, accounts, 5, seed=5) != assignment
    # The folds by game of the same seed do split some account (that is the
    # difference between the two units).
    by_game = FT.fold_of_rows(games, FT.game_folds(games, 5, seed=4))
    assert any(np.unique(by_game[accounts == a]).size > 1 for a in np.unique(accounts))

    with pytest.raises(FT.FinetuneError, match="two accounts"):
        FT.account_folds(np.array([1, 1, 2]), np.array([7, 8, 9]), 2, seed=0)
    with pytest.raises(FT.FinetuneError):
        FT.account_folds(np.array([1, 2, 3]), np.array([7, 7, 8]), 3, seed=0)
    with pytest.raises(FT.FinetuneError):
        FT.account_folds(np.array([1, 2]), np.array([7, 8]), 1, seed=0)
    with pytest.raises(FT.FinetuneError):
        FT.account_folds(np.array([1, 2]), np.array([7]), 2, seed=0)


# --- the allowed list -----------------------------------------------------------


def test_only_ladder_holdout_games_of_the_old_file_are_allowed(
    tiny: Tiny, tmp_path: Path
):
    allowed = FT.allowed_games(tiny.old_games)
    assert allowed == {f"old-{game}" for game in (0, 1, 3, 4, 6, 7)}
    empty = tmp_path / "none.jsonl"
    empty.write_text(
        json.dumps({"index": 0, "id": "train-0", "split": {"p1": "train"}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(FT.FinetuneError):
        FT.allowed_games(empty)
    # ...and a run with such a file refuses before it writes anything.
    out = tmp_path / "refused"
    assert FT.main(arguments(tiny, out, "--old-games", str(empty))) == 1
    report = json.loads((out / FT.REPORT_JSON).read_text(encoding="utf-8"))
    assert report["status"] == FT.FAILED and "nothing is allowed" in report["reason"]
    assert not (out / FT.ARTIFACT_NAME).exists()


def test_rows_of_a_game_outside_the_list_never_leave_the_shard(tiny: Tiny):
    allowed = FT.allowed_games(tiny.old_games)
    parts = FT.load_parts(tiny.dataset, allowed, corpus_rows=40, seed=1)
    games = set(parts.domain["m_battle"].tolist())
    assert games == tiny.allowed_index and not games & tiny.sealed_index
    assert set(parts.domain_ids.tolist()) == allowed
    assert (parts.domain["m_split"] == SPLITS.index("ladder_holdout")).all()
    assert (parts.corpus["m_split"] == SPLITS.index("train")).all()
    assert (parts.val["m_split"] == SPLITS.index("val")).all()
    assert parts.corpus["act_mon"].shape[0] == 40
    assert len(set(map(bytes, parts.corpus["mon_hp"]))) > 1
    found = parts.counts["domain"]
    assert found["rows_used"] == N_ALLOWED * ROWS and found["games_used"] == N_ALLOWED
    assert found["rows_dropped_game_not_allowed"] == N_SEALED * ROWS
    assert found["games_dropped_not_allowed"] == N_SEALED
    assert found["rows_in_split"] == (N_ALLOWED + N_SEALED) * ROWS
    assert found["used_games_outside_allowed"] == 0
    assert found["allowed_games_without_rows_here"] == 0
    assert parts.counts["corpus"]["rows_in_split"] == N_TRAIN * ROWS
    assert parts.used_ids == sorted(allowed)
    # The seal by the dataset's own dates: both games of the newer day are
    # among the dropped ones and none is used.
    assert found["sealed_since"] == FT.SEALED_SINCE
    assert found["games_in_split_dated_sealed"] == N_SEALED
    assert found["games_dropped_dated_sealed"] == N_SEALED
    assert found["games_used_dated_sealed"] == 0 and found["games_used_undated"] == 0

    # A list with one more game than the build holds: said, not an error.
    more = FT.load_parts(tiny.dataset, allowed | {"old-99"}, corpus_rows=5, seed=1)
    assert more.counts["domain"]["allowed_games_not_in_this_build"] == 1
    assert more.counts["domain"]["rows_used"] == N_ALLOWED * ROWS
    # A list that names none of the build's games is an error.
    with pytest.raises(FT.FinetuneError):
        FT.load_parts(tiny.dataset, {"somewhere-else"}, corpus_rows=5, seed=1)


# --- the date seal ----------------------------------------------------------------


def copy_of(dataset: Path, target: Path) -> Path:
    target.mkdir()
    for path in dataset.iterdir():
        (target / path.name).write_bytes(path.read_bytes())
    return target


def battle_rows(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_the_seal_is_the_ladder_reads_day():
    from evaluation import oppmodel_ladder_read as LR

    assert FT.SEALED_SINCE == LR.FRESH_SINCE
    assert FT.sealed_start() == LR._day_start(LR.FRESH_SINCE, "the seal")
    assert "sealed" not in vars(
        FT.parse_args(["--artifact", "a", "--dataset", "d", "--out", "o"])
    )  # no switch moves it
    assert FT._dated(1.0) == 1.0 and FT._dated(7) == 7.0
    for value in (None, "1790000000", True, 0, -5, float("nan"), float("inf")):
        assert FT._dated(value) is None, value


def test_a_list_that_names_sealed_games_is_refused_before_a_shard_is_opened(
    tiny: Tiny, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # THE FAILING CASE: the newer build's own battles.jsonl given as the list
    # of old games. It names the two games of the sealed day.
    newer = tiny.dataset / "battles.jsonl"
    assert {"sealed-2", "sealed-5"} <= {
        row["id"] for row in FT.holdout_battles(newer).values()
    }
    with pytest.raises(FT.FinetuneError, match=FT.SEALED_SINCE) as caught:
        FT.allowed_games(newer)
    assert "2 ladder_holdout games" in str(caught.value)
    assert "sealed-2" in str(caught.value)

    opened: list[Any] = []
    real_load = F.load_batch

    def load(path: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(path)
        return real_load(path, *args, **kwargs)

    def no_fit(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a fit was started")

    monkeypatch.setattr(F, "load_batch", load)
    monkeypatch.setattr(FT, "finetune", no_fit)
    out = tmp_path / "refused"
    assert FT.main(arguments(tiny, out, "--old-games", str(newer))) == 1
    report = json.loads((out / FT.REPORT_JSON).read_text(encoding="utf-8"))
    assert report["status"] == FT.FAILED
    assert FT.SEALED_SINCE in report["reason"] and "sealed" in report["reason"]
    assert not opened and not (out / FT.ARTIFACT_NAME).exists()

    # The boundary: the first second of the day is sealed, the one before is not.
    seal = int(FT.sealed_start())
    rows = [
        {"index": 0, "id": "a", "split": {"p1": "ladder_holdout"}, "time": seal - 1},
        {"index": 1, "id": "b", "split": {"p2": "ladder_holdout"}, "time": seal - 9},
        {"index": 2, "id": "c", "split": {"p1": "train"}, "time": seal + 99},
    ]
    assert FT.allowed_games(write_rows(tmp_path / "before.jsonl", rows)) == {"a", "b"}
    rows[1]["time"] = seal
    with pytest.raises(FT.FinetuneError, match="1 ladder_holdout games"):
        FT.allowed_games(write_rows(tmp_path / "at.jsonl", rows))


def test_the_dataset_s_own_dates_keep_a_sealed_game_out_whatever_the_list_says(
    tiny: Tiny, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    allowed = FT.allowed_games(tiny.old_games)
    opened: list[Any] = []
    real_load = F.load_batch

    def load(path: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(path)
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(F, "load_batch", load)
    # A list WITHOUT dates that names a sealed game passes the list's own
    # check; the dataset dates the game on the sealed day and stops the run.
    undated_list = battle_rows(tiny.old_games)
    undated_list.append({"index": 99, "id": "sealed-5", "split": {"p1": SPLITS[3]}})
    wrong = write_rows(tmp_path / "wrong.jsonl", undated_list)
    assert FT.allowed_games(wrong) == allowed | {"sealed-5"}
    with pytest.raises(FT.FinetuneError, match="dated on or after") as caught:
        FT.load_parts(tiny.dataset, allowed | {"sealed-5"}, corpus_rows=5, seed=1)
    assert "1 allowed games" in str(caught.value) and "sealed-5" in str(caught.value)
    assert not opened  # refused before a shard was read
    out = tmp_path / "wrong_list"
    assert FT.main(arguments(tiny, out, "--old-games", str(wrong))) == 1
    report = json.loads((out / FT.REPORT_JSON).read_text(encoding="utf-8"))
    assert report["status"] == FT.FAILED and FT.SEALED_SINCE in report["reason"]
    assert not opened

    # An allowed game the dataset cannot date cannot be shown to be old.
    other = copy_of(tiny.dataset, tmp_path / "undated")
    rows = battle_rows(other / "battles.jsonl")
    victim = next(row for row in rows if row["id"] == "old-3")
    del victim["time"]
    write_rows(other / "battles.jsonl", rows)
    with pytest.raises(FT.FinetuneError, match="no date") as caught:
        FT.load_parts(other, allowed, corpus_rows=5, seed=1)
    assert "old-3" in str(caught.value) and not opened
    # ...while a game OUTSIDE the list may lack a date: it is dropped anyway.
    rows = battle_rows(tiny.dataset / "battles.jsonl")
    del next(row for row in rows if row["id"] == "sealed-2")["time"]
    third = copy_of(tiny.dataset, tmp_path / "third")
    write_rows(third / "battles.jsonl", rows)
    parts = FT.load_parts(third, allowed, corpus_rows=5, seed=1)
    assert set(parts.domain["m_battle"].tolist()) == tiny.allowed_index
    assert parts.counts["domain"]["games_in_split_dated_sealed"] == N_SEALED - 1
    assert parts.counts["domain"]["games_dropped_not_allowed"] == N_SEALED


def test_a_run_never_fits_or_predicts_a_game_outside_the_list(
    tiny: Tiny, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    seen: dict[str, set[tuple[int, int]]] = {
        "fit": set(),
        "corpus": set(),
        "pred": set(),
    }
    fits: list[tuple[set[int], set[int]]] = []

    def games_of(batch: Any) -> set[tuple[int, int]]:
        pairs = zip(batch["m_split"].tolist(), batch["m_battle"].tolist())
        return {(int(split), int(game)) for split, game in pairs}

    real_fit, real_predict, real_compare = FT.finetune, FT.predict_rows, FT.compare

    def fit(net: Any, domain: Any, corpus: Any, **kwargs: Any) -> Any:
        seen["fit"] |= games_of(domain)
        seen["corpus"] |= games_of(corpus)
        fits.append(({game for _, game in games_of(domain)}, set()))
        return real_fit(net, domain, corpus, **kwargs)

    def predict(predictor: Any, batch: Any) -> Any:
        seen["pred"] |= games_of(batch)
        if fits and not fits[-1][1]:
            fits[-1][1].update(game for _, game in games_of(batch))
        return real_predict(predictor, batch)

    def compared(batch: Any, *args: Any, **kwargs: Any) -> Any:
        seen["pred"] |= games_of(batch)
        return real_compare(batch, *args, **kwargs)

    monkeypatch.setattr(FT, "finetune", fit)
    monkeypatch.setattr(FT, "predict_rows", predict)
    monkeypatch.setattr(FT, "compare", compared)
    out = tmp_path / "run"
    argv = arguments(tiny, out, "--learning-rates", "1e-3", "--passes", "1", "2")
    assert FT.main(argv) == 0

    train, val, test, holdout = range(4)
    forbidden = tiny.sealed_index | tiny.test_index
    for name, pairs in seen.items():
        assert pairs, name
        assert not {game for _, game in pairs} & forbidden, name
        assert test not in {split for split, _ in pairs}, name
    assert {split for split, _ in seen["fit"]} == {holdout}
    assert {game for _, game in seen["fit"]} == tiny.allowed_index
    assert {split for split, _ in seen["corpus"]} == {train}
    assert {split for split, _ in seen["pred"]} == {val, holdout}
    # Three fold fits (one learning rate) and the fit on all games; a fold's
    # network first predicts exactly the games it was not fitted on.
    assert len(fits) == 4
    for fitted, predicted in fits[:3]:
        assert predicted and not fitted & predicted
        assert fitted | predicted == tiny.allowed_index
    assert fits[3][0] == tiny.allowed_index

    report = json.loads((out / FT.REPORT_JSON).read_text(encoding="utf-8"))
    assert report["status"] == FT.DONE
    assert report["verdict"] in (FT.TAKEN, FT.NOT_TAKEN)
    assert report["grid"]["pre_registered"] is False
    domain = report["data"]["domain"]
    assert domain["rows_used"] == N_ALLOWED * ROWS
    assert domain["rows_dropped_game_not_allowed"] == N_SEALED * ROWS
    assert domain["games_dropped_not_allowed"] == N_SEALED
    assert report["allowed_games"]["sha256"] == FT.sha256_file(tiny.old_games)
    assert report["source"]["sha256"] == FT.sha256_file(tiny.artifact)
    assert report["folds"]["games"] == [2, 2, 2]
    assert [row["key"] for row in report["settings"]] == [
        "lr 0.001 x 1 passes",
        "lr 0.001 x 2 passes",
    ]
    for row in report["settings"]:
        assert row["oof"]["fine"]["n"] == report["base"]["oof_rows"]["scored_slots"]
        assert len(row["oof_fine_diff_by_fold"]) == 3
        assert set(row["oof"]["fine_by_class"]) == set(FT.SLOT_CLASSES)
        assert set(row["oof"]["joint_top8"]) == set(FT.JOINT_SLICES)
        # Validation is read like the old games: by kind of slot and jointly.
        assert set(row["val"]) == set(FT.VAL_ROWS)
        assert set(row["val"]["fine_by_class"]) == set(FT.SLOT_CLASSES)
        assert set(row["val"]["joint_top8"]) == set(FT.JOINT_SLICES)
        assert row["val"]["joint_top8"][FT.JOINT_ALL]["n"] > 0
        assert set(row["in_sample"]) == set(FT.IN_SAMPLE_ROWS)
    whole = report["base"]
    assert whole["val_joint_top8"][FT.JOINT_ALL]["turns"] > 0
    assert sum(whole["val_slots_by_class"].values()) == whole["val_scored_slots"]
    text = (out / FT.REPORT_MD).read_text(encoding="utf-8")
    assert "NOT the pre-registered grid" in text and report["verdict"] in text
    assert report["pre_registered_reading"] is False
    assert report["folds"]["unit"] == FT.FOLD_GAME
    assert report["folds"]["pre_registered"] is False  # three folds, not five
    # Beside the verdict: the best setting's readings the rule does not gate,
    # on both sets, and the gain with the setting chosen without the fold read.
    beside = report["decision"]["not_gated"]
    assert beside["setting"] == report["decision"]["best"]
    assert set(beside["readings"]) == set(FT.PLACES)
    names = {name for name, _ in FT.NOT_GATED_READINGS}
    for place in FT.PLACES:
        assert set(beside["readings"][place]) == names, place
    best = next(r for r in report["settings"] if r["key"] == beside["setting"])
    assert (
        beside["readings"][FT.PLACE_VAL][FT.READ_JOINT_PROTECT]
        == best["val"]["joint_top8"][FT.JOINT_PROTECT]
    )
    assert (
        beside["readings"][FT.PLACE_OOF][FT.READ_NLL_PROTECT]
        == best["oof"]["fine_by_class"][FT.CLASS_PROTECT]
    )
    assert isinstance(beside["costs"], list)
    corrected = beside["selection_corrected"]
    assert len(corrected["chosen_by_fold"]) == 3
    assert set(corrected["chosen_by_fold"]) <= {r["key"] for r in report["settings"]}
    assert corrected["fine"]["n"] == whole["oof_rows"]["scored_slots"]
    assert "Not gated by the rule" in text and FT.READ_JOINT_PROTECT in text
    assert "Setting chosen without the fold it is read on" in text
    seal = report["data"]["domain"]
    assert seal["games_in_split_dated_sealed"] == N_SEALED
    assert seal["games_used_dated_sealed"] == 0

    # The directory is this run's: a second run into it is refused and the
    # finished report stays.
    assert FT.main(argv) == 1
    again = json.loads((out / FT.REPORT_JSON).read_text(encoding="utf-8"))
    assert again["status"] == FT.DONE and again["created"] == report["created"]


def test_an_artifact_of_another_dataset_is_refused(tiny: Tiny, tmp_path: Path):
    other = tmp_path / "other"
    other.mkdir()
    for path in tiny.dataset.iterdir():
        (other / path.name).write_bytes(path.read_bytes())
    manifest = json.loads((other / "manifest.json").read_text(encoding="utf-8"))
    manifest["tag"] = "another build"
    (other / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "out"
    assert FT.main(arguments(tiny, out, "--dataset", str(other))) == 1
    report = json.loads((out / FT.REPORT_JSON).read_text(encoding="utf-8"))
    assert report["status"] == FT.FAILED and "manifest sha256" in report["reason"]


# --- the fine-tune --------------------------------------------------------------


def test_zero_learning_rate_reproduces_the_source_exactly(tiny: Tiny):
    loaded, parts, recipe = prepared(tiny)
    source = loaded.predictor
    domain = F.sheet_unknown_as_closed(parts.domain)
    corpus = F.sheet_unknown_as_closed(parts.corpus)
    want = FT.predict_rows(source, domain)
    before = M.clone_state(source.net)

    net = copy.deepcopy(source.net)
    states, log = FT.finetune(
        net, domain, corpus, lr=0.0, checkpoints=(1, 2), recipe=recipe, seed=5
    )
    assert sorted(states) == [1, 2] and [row["pass"] for row in log] == [1, 2]
    assert log[-1]["steps"] == 2 * int(np.ceil(N_ALLOWED * ROWS / recipe.domain_batch))
    assert not net.training
    for count in (1, 2):
        for name, value in states[count].items():
            assert torch.equal(value, before[name]), name
        scratch = copy.deepcopy(source.net)
        scratch.load_state_dict(states[count])
        got = FT.predict_rows(FT.predictor_of(scratch, source), domain)
        assert same_prediction(got, want)

    # A real learning rate moves the copy, and never the source's own network.
    moved = copy.deepcopy(source.net)
    FT.finetune(moved, domain, corpus, lr=1e-2, checkpoints=(1,), recipe=recipe, seed=5)
    after = FT.predict_rows(FT.predictor_of(moved, source), domain)
    assert not np.array_equal(after["action"], want["action"])
    for name, value in M.clone_state(source.net).items():
        assert torch.equal(value, before[name]), name
    assert same_prediction(FT.predict_rows(source, domain), want)

    with pytest.raises(FT.FinetuneError):
        FT.finetune(net, domain, corpus, lr=0.0, checkpoints=(), recipe=recipe, seed=5)
    with pytest.raises(FT.FinetuneError):
        FT.finetune(
            net, domain, corpus, lr=0.0, checkpoints=(0,), recipe=recipe, seed=5
        )


def test_a_checkpoint_is_the_shorter_run_with_the_same_seed(tiny: Tiny):
    loaded, parts, recipe = prepared(tiny)
    domain, corpus = parts.domain, parts.corpus

    def fit(checkpoints: tuple[int, ...], seed: int) -> dict[int, Any]:
        net = copy.deepcopy(loaded.predictor.net)
        states, _ = FT.finetune(
            net,
            domain,
            corpus,
            lr=1e-3,
            checkpoints=checkpoints,
            recipe=recipe,
            seed=seed,
        )
        return states

    long, short = fit((1, 2, 4), 7), fit((2,), 7)
    assert sorted(long) == [1, 2, 4] and sorted(short) == [2]
    for name, value in short[2].items():
        assert torch.equal(value, long[2][name]), name
    assert any(not torch.equal(long[1][n], long[4][n]) for n in long[1])
    other = fit((2,), 8)  # another seed: other batches and masks
    assert any(not torch.equal(other[2][n], short[2][n]) for n in other[2])


def test_the_two_parts_of_a_batch_carry_equal_loss_weight(tiny: Tiny):
    loaded, parts, _ = prepared(tiny)
    net = TM.jolt(copy.deepcopy(loaded.predictor.net))
    n_domain = 5
    mixed = F.concat_batches(
        [F.take(parts.domain, slice(0, n_domain)), F.take(parts.corpus, slice(0, 20))]
    )
    labels = M.to_labels(mixed)
    with torch.no_grad():
        terms = M.nll_terms(net(M.to_tensors(mixed)), labels)
    weight = FT.balanced_weight(labels["weight"], terms["action_scored"], n_domain)
    assert torch.equal(weight[n_domain:], labels["weight"][n_domain:])
    ratio = weight[:n_domain] / labels["weight"][:n_domain]
    assert torch.allclose(ratio, ratio[0].expand_as(ratio))  # one factor, cap kept
    slots = terms["action_scored"].sum(-1).to(weight.dtype)
    mass = weight * slots
    assert float(mass[:n_domain].sum()) == pytest.approx(float(mass[n_domain:].sum()))

    _, got = M.total_loss(terms, weight)
    fine = terms["action"] + terms["target"]
    means = []
    for part in (slice(0, n_domain), slice(n_domain, None)):
        own = labels["weight"][part, None]
        means.append(
            float((own * fine[part]).sum() / (own * terms["action_scored"][part]).sum())
        )
    assert means[0] != pytest.approx(means[1])
    assert got["fine"] == pytest.approx(0.5 * (means[0] + means[1]), rel=1e-5)
    assert dict(FT._part_means(terms, labels["weight"], n_domain)) == pytest.approx(
        {"domain_fine": means[0], "corpus_fine": means[1]}, rel=1e-5
    )
    # Nothing scored in one part: the weights come back as they were.
    nothing = torch.zeros_like(terms["action_scored"])
    assert torch.equal(
        FT.balanced_weight(labels["weight"], nothing, n_domain), labels["weight"]
    )


def test_recipe_reads_the_source_training_and_blanks_no_rating_twice():
    recipe = FT.Recipe.from_training(
        {"slot_swap": 0.25, "clip": 2.0, "elo_drop": 0.3}, batch=160, elo_blind=False
    )
    assert recipe.domain_batch == 32 and recipe.corpus_per_domain == 4
    assert (recipe.slot_swap, recipe.clip, recipe.elo_drop) == (0.25, 2.0, 0.3)
    defaults = vars(T.parse_args([]))
    assert recipe.mega_weight == defaults["mega_weight"]
    assert recipe.weight_decay == defaults["weight_decay"]
    blind = FT.Recipe.from_training(None, batch=7, elo_blind=True)
    assert blind.elo_drop == 0.0 and blind.domain_batch == 1
    assert blind.slot_swap == defaults["slot_swap"]


# --- scoring --------------------------------------------------------------------


def test_slot_classes_partition_the_scored_slots(tiny: Tiny):
    _, parts, _ = prepared(tiny)
    batch = F.concat_batches([parts.domain, parts.val])
    classes = FT.slot_classes(batch)
    assert tuple(classes) == FT.SLOT_CLASSES
    total = np.stack([mask.astype(int) for mask in classes.values()]).sum(0)
    scored = F.slot_nll(F.uniform_prediction(batch), batch)["fine_scored"]
    assert np.array_equal(total > 0, scored) and total.max() == 1
    assert all(mask.any() for mask in classes.values())
    n_cand = batch["cand_flag"].shape[-1]
    y_action = batch["y_action"].astype(int)
    assert (y_action[classes[FT.CLASS_SWITCH]] > n_cand).all()
    assert (y_action[classes[FT.CLASS_CENSORED]] < 0).all()
    rows, slots = np.nonzero(classes[FT.CLASS_PROTECT])
    flags = batch["cand_flag"][rows, slots, y_action[rows, slots]]
    assert ((flags & F.CAND_PROTECT) > 0).all()


def test_compare_is_the_paired_difference_on_the_scorecard_definitions(tiny: Tiny):
    loaded, parts, _ = prepared(tiny)
    batch = parts.domain
    base = FT.predict_rows(loaded.predictor, batch)
    other_net = TM.jolt(copy.deepcopy(loaded.predictor.net), seed=3)
    new = FT.predict_rows(FT.predictor_of(other_net, loaded.predictor), batch)
    found = FT.compare(batch, base, new, resamples=200, seed=2)

    scores = {"base": F.slot_nll(base, batch), "new": F.slot_nll(new, batch)}
    scored = scores["base"]["fine_scored"]
    fine = found["fine"]
    assert fine["n"] == int(scored.sum()) and fine["games"] == N_ALLOWED
    assert fine["base"] == pytest.approx(scores["base"]["fine"][scored].mean())
    assert fine["new"] == pytest.approx(scores["new"]["fine"][scored].mean())
    assert fine["diff"] == pytest.approx(fine["new"] - fine["base"])
    assert fine["low"] <= fine["diff"] <= fine["high"]
    want = SC.paired_difference(
        scores["new"]["fine"],
        scores["base"]["fine"],
        scored,
        batch["m_battle"],
        resamples=200,
        seed=2,
    )
    assert fine["diff"] == pytest.approx(want["diff"])
    assert (fine["low"], fine["high"]) == pytest.approx((want["low"], want["high"]))
    by_class = found["fine_by_class"]
    assert sum(row["n"] for row in by_class.values()) == fine["n"]
    assert sum(row["n"] * row["diff"] for row in by_class.values()) == pytest.approx(
        fine["n"] * fine["diff"]
    )
    hits, counted = F.topk_hits(F.fine_probs(new, batch), F.fine_label(batch), 1)
    assert found["top1"]["new"] == pytest.approx(hits[counted].mean())
    assert found["top1"]["n"] == int(counted.sum())

    # The joint rows are the scorecard's coverage of the same two predictions.
    card = SC.joint_coverage_set(batch, {"base": base, "new": new}, resamples=10)
    joint = found["joint_top8"][FT.JOINT_ALL]
    assert joint["n"] == card["counted"]["examples"] > 0
    for side in ("base", "new"):
        row = card["predictors"][side][SC.JOINT_PLAIN]["slices"][SC.SLICE_ALL]
        assert joint[side] == pytest.approx(row["top"][str(SC.JOINT_K)])
    assert joint["diff"] == pytest.approx(joint["new"] - joint["base"])
    assert found["joint_top8"][FT.JOINT_SWITCH]["n"] == card["truth"]["involves_switch"]
    assert 0 < found["joint_top8"][FT.JOINT_PROTECT]["n"] < joint["n"]

    same = FT.compare(batch, base, base, resamples=50, seed=2)
    assert same["fine"]["diff"] == 0.0 and same["fine"]["high"] == 0.0
    assert same["joint_top8"][FT.JOINT_ALL]["diff"] == 0.0
    assert "joint_top8" not in FT.compare(batch, base, new, resamples=5, joint=False)


# --- the rule -------------------------------------------------------------------


def setting(
    key: str,
    nll: float,
    fine: tuple[float, float, float],
    joint: tuple[float, float, float] = (0.0, -0.02, 0.02),
    switch: float = 0.0,
) -> dict[str, Any]:
    def row(values: tuple[float, float, float]) -> dict[str, Any]:
        return {"diff": values[0], "low": values[1], "high": values[2], "n": 100}

    return {
        "key": key,
        "oof": {
            "fine": {**row(fine), "new": nll},
            "joint_top8": {
                FT.JOINT_ALL: row(joint),
                FT.JOINT_SWITCH: row((switch, switch - 0.05, switch + 0.05)),
                FT.JOINT_PROTECT: row((0.0, -0.05, 0.05)),
            },
        },
    }


def test_decide_follows_the_pre_registered_rule():
    good = setting("good", 1.40, (-0.03, -0.05, -0.01))
    worse = setting("worse", 1.45, (-0.01, -0.02, -0.001))
    found = FT.decide([worse, good])
    assert found["best"] == "good" and found["verdict"] == FT.TAKEN
    assert all(row["holds"] for row in found["conditions"].values())

    # The best setting alone decides: a passing runner-up does not rescue it.
    crossing = setting("crossing", 1.39, (-0.03, -0.07, 0.01))
    found = FT.decide([good, crossing])
    assert found["best"] == "crossing" and found["verdict"] == FT.NOT_TAKEN
    assert not found["conditions"]["fine_nll_interval_below_zero"]["holds"]
    touching = setting("touching", 1.0, (-0.03, -0.06, 0.0))
    assert FT.decide([touching])["verdict"] == FT.NOT_TAKEN

    # Joint coverage: down by no more than its own half-width (here 0.02).
    name = "joint_top8_not_below_zero_by_more_than_its_half_width"
    inside = setting("j", 1.0, (-0.03, -0.05, -0.01), joint=(-0.02, -0.04, 0.0))
    assert FT.decide([inside])["verdict"] == FT.TAKEN
    beyond = setting("j", 1.0, (-0.03, -0.05, -0.01), joint=(-0.021, -0.041, -0.001))
    found = FT.decide([beyond])
    assert found["verdict"] == FT.NOT_TAKEN and not found["conditions"][name]["holds"]
    assert found["conditions"][name]["half_width"] == pytest.approx(0.02)

    # Replies that hold a switch: down by no more than two points.
    name = "switch_reply_coverage_not_down_more_than_2_points"
    edge = setting("s", 1.0, (-0.03, -0.05, -0.01), switch=-0.02)
    assert FT.decide([edge])["verdict"] == FT.TAKEN
    fallen = setting("s", 1.0, (-0.03, -0.05, -0.01), switch=-0.021)
    found = FT.decide([fallen])
    assert found["verdict"] == FT.NOT_TAKEN and not found["conditions"][name]["holds"]

    # No interval (one game) can never pass; no setting is an error.
    blind = setting("b", 1.0, (-0.03, -0.05, -0.01))
    blind["oof"]["fine"]["high"] = None
    assert FT.decide([blind])["verdict"] == FT.NOT_TAKEN
    with pytest.raises(FT.FinetuneError):
        FT.decide([])
    assert FT.GRID_LEARNING_RATES == (1e-5, 3e-5, 1e-4) and FT.GRID_PASSES == (1, 2, 4)
    assert FT.SWITCH_DROP_LIMIT == 0.02 and FT.DEFAULT_RESAMPLES == 2000


# --- beside the rule ------------------------------------------------------------


def reading(diff: float, low: float, high: float, n: int = 100) -> dict[str, Any]:
    return {
        "base": 0.5,
        "new": 0.5 + diff,
        "diff": diff,
        "low": low,
        "high": high,
        "n": n,
        "games": 50,
    }


def test_protect_costs_are_named_beside_the_verdict_and_gate_nothing():
    quiet = reading(0.0, -0.01, 0.01)
    best = setting("best", 1.40, (-0.03, -0.05, -0.01))
    # Protect replies lose 10 points out of fold and 1.5 on validation, both
    # intervals below zero; the slots where a Protect was clicked cost NLL.
    best["oof"]["joint_top8"][FT.JOINT_PROTECT] = reading(-0.10, -0.15, -0.05)
    best["oof"]["fine_by_class"] = {
        FT.CLASS_PROTECT: reading(0.12, 0.10, 0.14),
        FT.CLASS_SWITCH: reading(-0.015, -0.04, 0.012),
    }
    best["val"] = {
        "joint_top8": {
            FT.JOINT_ALL: quiet,
            FT.JOINT_SWITCH: reading(0.009, 0.004, 0.014),
            FT.JOINT_PROTECT: reading(-0.015, -0.02, -0.01),
        },
        "fine_by_class": {
            FT.CLASS_PROTECT: reading(0.063, 0.060, 0.067),
            FT.CLASS_SWITCH: reading(-0.036, -0.04, -0.03),
        },
    }
    # The pre-registered rule does not read Protect: the verdict stands.
    found = FT.decide([best])
    assert found["verdict"] == FT.TAKEN and "not_gated" not in found
    assert set(found["conditions"]) == {
        "fine_nll_interval_below_zero",
        "joint_top8_not_below_zero_by_more_than_its_half_width",
        "switch_reply_coverage_not_down_more_than_2_points",
    }
    beside = FT.not_gated(best, {"chosen_by_fold": ["best"], "fine": quiet})
    old, val = FT.PLACES[FT.PLACE_OOF], FT.PLACES[FT.PLACE_VAL]
    assert beside["costs"] == [
        f"{old}: {FT.READ_JOINT_PROTECT}",
        f"{old}: {FT.READ_NLL_PROTECT}",
        f"{val}: {FT.READ_JOINT_PROTECT}",
        f"{val}: {FT.READ_NLL_PROTECT}",
    ]
    assert beside["setting"] == "best"
    assert beside["readings"][FT.PLACE_VAL][FT.READ_JOINT_PROTECT]["diff"] == -0.015
    assert beside["selection_corrected"]["chosen_by_fold"] == ["best"]
    assert "not gated" in beside["note"]

    # An interval that reaches zero is not a cost; a missing row is left out.
    best["val"]["joint_top8"][FT.JOINT_PROTECT] = reading(-0.015, -0.03, 0.0)
    best["oof"]["joint_top8"][FT.JOINT_PROTECT] = reading(-0.0095, -0.0312, 0.01)
    del best["val"]["fine_by_class"][FT.CLASS_PROTECT]
    again = FT.not_gated(best)
    assert again["costs"] == [f"{old}: {FT.READ_NLL_PROTECT}"]
    assert FT.READ_NLL_PROTECT not in again["readings"][FT.PLACE_VAL]
    assert again["selection_corrected"] is None
    assert not FT._worse({"diff": -1.0, "low": None, "high": None}, False)
    assert FT._worse(reading(0.1, 0.01, 0.2), True)
    assert not FT._worse(reading(0.1, 0.01, 0.2), False)


def test_the_setting_of_a_fold_is_chosen_without_that_fold(tiny: Tiny):
    scored = np.ones((6, 2), dtype=bool)
    scored[5, 1] = False
    row_fold = np.array([0, 0, 1, 1, 2, 2])
    a = np.full((6, 2), 1.0)
    b = np.full((6, 2), 1.0)
    b[row_fold == 2] = 0.2  # b is far better on fold 2 only
    b[row_fold != 2] = 1.1  # and slightly worse elsewhere
    b[5, 1] = -50.0  # an unscored slot is never read
    fine = {"a": a, "b": b}
    # Fold 2 is read with the setting chosen on folds 0 and 1, where a wins.
    assert FT.choose_without_fold(fine, scored, row_fold, 3) == ["b", "b", "a"]
    assert FT.choose_without_fold({"b": b, "a": a}, scored, row_fold, 3)[2] == "a"
    assert (
        FT.choose_without_fold({"a": a, "also": a.copy()}, scored, row_fold, 3)
        == ["a"] * 3
    )  # equals: the first of the grid
    with pytest.raises(FT.FinetuneError):
        FT.choose_without_fold({}, scored, row_fold, 3)
    with pytest.raises(FT.FinetuneError):
        FT.choose_without_fold(fine, scored, np.zeros(6, dtype=int), 1)

    # On real rows: the pooled prediction takes every fold's rows from the
    # setting chosen without it, and its change is ``compare``'s.
    loaded, parts, _ = prepared(tiny)
    batch = parts.domain
    base = FT.predict_rows(loaded.predictor, batch)
    jolted = TM.jolt(copy.deepcopy(loaded.predictor.net), seed=3)
    new = FT.predict_rows(FT.predictor_of(jolted, loaded.predictor), batch)
    games = batch["m_battle"].astype(np.int64)
    place = FT.fold_of_rows(games, FT.game_folds(games, 3, seed=11))
    oof = {"same": base, "moved": new}
    found = FT.selection_corrected(batch, base, oof, place, 3, resamples=50, seed=2)
    chosen = found["chosen_by_fold"]
    nll = {key: F.slot_nll(pred, batch) for key, pred in oof.items()}
    mask = nll["same"]["fine_scored"]
    assert chosen == FT.choose_without_fold(
        {key: value["fine"] for key, value in nll.items()}, mask, place, 3
    )
    total = sum(
        float(nll[key]["fine"][mask & (place == fold)[:, None]].sum())
        for fold, key in enumerate(chosen)
    )
    assert found["fine"]["new"] == pytest.approx(total / int(mask.sum()))
    assert found["fine"]["n"] == int(mask.sum())
    assert set(found["joint_top8"]) == set(FT.JOINT_SLICES)
    only = FT.selection_corrected(batch, base, {"same": base}, place, 3, resamples=5)
    assert only["chosen_by_fold"] == ["same"] * 3 and only["fine"]["diff"] == 0.0


# --- the artifact ---------------------------------------------------------------


def test_written_artifact_reloads_to_the_same_predictions(
    tiny: Tiny, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    real = FT.decide

    def taken(settings: Any) -> dict[str, Any]:
        found = real(settings)
        found.update({"best": settings[-1]["key"], "verdict": FT.TAKEN})
        return found

    monkeypatch.setattr(FT, "decide", taken)
    out = tmp_path / "taken"
    argv = arguments(tiny, out, "--learning-rates", "1e-3", "--passes", "1", "2")
    assert FT.main(argv) == 0
    report = json.loads((out / FT.REPORT_JSON).read_text(encoding="utf-8"))
    assert report["verdict"] == FT.TAKEN and report["artifact"] == str(
        out / "artifact.pt"
    )
    assert report["reload"]["identical"] is True
    assert max(report["reload"]["max_abs_difference"].values()) == 0.0
    assert max(report["reload"]["max_abs_change_from_source"].values()) > 0.0
    assert not (out / "artifact.pt.unverified").exists()

    source = A.load_predictor(tiny.artifact)
    written = A.load_predictor(out / "artifact.pt")
    assert written.name == f"{source.name}{FT.NAME_SUFFIX}{N_ALLOWED}"
    assert written.name == report["name"] and written.name.endswith("_ft_own6")
    assert written.kind == M.KIND and written.predictor.name == written.name
    assert written.predictor.temperatures == source.predictor.temperatures
    assert written.predictor.elo_mode == source.predictor.elo_mode
    extra, before = written.meta["extra"], source.meta["extra"]
    allowed = FT.allowed_games(tiny.old_games)

    # THE RECORD IS THE FINE-TUNE'S OWN. A reader that compares the artifact
    # with the build's manifest (the scorecard) finds another training set...
    build = T.manifest_sha256(tiny.dataset)
    assert SC._trained_on(before) == build
    marked = SC._trained_on(extra)
    assert marked != build and marked == report["reload"]["trained_on"]
    assert marked == f"{FT.MANIFEST_MARK}:{N_ALLOWED}-games:{build}"
    assert extra["dataset"]["manifest_sha256"] == extra["manifest_sha256"] == marked
    assert extra["dataset"]["source_manifest_sha256"] == build
    entry = SC.load_artifact_entry(str(out / "artifact.pt"), [], build)
    assert entry.info["fitted_on_this_dataset"] is False
    assert entry.info["artifact_name"] == written.name
    assert entry.info["created"] == report["created"] == extra["created"]
    clean = SC.load_artifact_entry(str(tiny.artifact), [], build)
    assert clean.info["fitted_on_this_dataset"] is True
    # ...its name, date and tag are its own, and nothing of the source's
    # training (best epoch, history, calibration) reads as this network's.
    assert extra["name"] == written.name != before["name"]
    assert str(N_ALLOWED) in extra["dataset_tag"]
    assert "in-sample" in extra["dataset_tag"]
    assert extra[FT.KEY_SOURCE_TRAINING] == before
    for key in ("best", "history", "calibration", "examples", "stopped_early"):
        assert key in before and key not in extra, key
    for key in ("config", "args", "elo_mode", "kind"):
        assert extra[key] == before[key], key
    # The formats the runtime stands down by travel with the dataset record.
    assert extra["dataset"].get("formats") == before["dataset"].get("formats")
    assert extra["dataset"]["tag"] == before["dataset"]["tag"]
    # It lists the games it has seen, and a reading on any of them is refused.
    assert FT.seen_games(extra) == allowed == set(extra["finetune"]["seen_games"])
    assert report["reload"]["seen_games"] == N_ALLOWED
    assert report["artifact_record"]["seen_games"] == N_ALLOWED
    assert FT.seen_games(before) == frozenset() == FT.seen_games(None)
    FT.check_unseen(before, allowed)  # never fine-tuned: any game is unseen
    FT.check_unseen(extra, ["sealed-2", "sealed-5", "another-game"])
    with pytest.raises(FT.FinetuneError, match="in-sample"):
        FT.check_unseen(extra, ["another-game", "old-3"], "the ladder holdout")
    text = (out / FT.REPORT_MD).read_text(encoding="utf-8")
    assert "any reading of it on them is in-sample" in text and str(marked) in text
    # A fine-tune over it would cross-validate in-sample: refused.
    again = tmp_path / "again"
    argv_again = [*argv, "--artifact", str(out / "artifact.pt"), "--out", str(again)]
    assert FT.main(argv_again) == 1
    failed = json.loads((again / FT.REPORT_JSON).read_text(encoding="utf-8"))
    assert failed["status"] == FT.FAILED and "itself fine-tuned" in failed["reason"]
    assert not (again / FT.ARTIFACT_NAME).exists()

    record = extra["finetune"]
    assert record["source_artifact_sha256"] == FT.sha256_file(tiny.artifact)
    assert record["allowed_games_sha256"] == FT.sha256_file(tiny.old_games)
    assert record["setting"] == {"lr": 1e-3, "passes": 2}
    assert record["domain_rows"] == N_ALLOWED * ROWS
    assert record["domain_games"] == N_ALLOWED
    assert record["grid_pre_registered"] is False
    assert record["pre_registered_reading"] is False
    assert record["sealed_since"] == FT.SEALED_SINCE
    assert set(record["val"]["joint_top8"]) == set(FT.JOINT_SLICES)
    assert record["not_gated_costs"] == report["decision"]["not_gated"]["costs"]

    # The in-memory model, rebuilt here from the same seed: the fit on all
    # allowed games with the taken setting.
    loaded, parts, recipe = prepared(tiny)
    domain = F.sheet_unknown_as_closed(parts.domain)
    net = copy.deepcopy(loaded.predictor.net)
    states, _ = FT.finetune(
        net,
        domain,
        F.sheet_unknown_as_closed(parts.corpus),
        lr=1e-3,
        checkpoints=(1, 2),
        recipe=recipe,
        seed=FT._fit_seed(11, 3),
    )
    net.load_state_dict(states[2])
    in_memory = FT.predict_rows(FT.predictor_of(net, loaded.predictor), domain)
    reloaded = FT.predict_rows(written.predictor, domain)
    assert same_prediction(reloaded, in_memory)
    assert not same_prediction(reloaded, FT.predict_rows(loaded.predictor, domain))

    # The report's validation rows are ``compare`` on the validation split
    # with the joint rows: the same network, read here again.
    val = F.sheet_unknown_as_closed(T.without_holdout_battles(parts.val)[0])
    want = FT.compare(
        val,
        FT.predict_rows(FT.predictor_of(loaded.predictor.net, loaded.predictor), val),
        FT.predict_rows(written.predictor, val),
        resamples=50,
        seed=11,
    )
    got = report["settings"][-1]["val"]
    for label in FT.JOINT_SLICES:
        for key in ("base", "new", "diff", "n"):
            assert got["joint_top8"][label][key] == pytest.approx(
                want["joint_top8"][label][key]
            ), (label, key)
    for label in FT.SLOT_CLASSES:
        assert got["fine_by_class"][label]["diff"] == pytest.approx(
            want["fine_by_class"][label]["diff"]
        )
    assert got["fine"]["diff"] == pytest.approx(want["fine"]["diff"])


def test_a_reload_that_differs_keeps_the_unverified_name(
    tiny: Tiny, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    loaded, parts, _ = prepared(tiny)
    moved = TM.jolt(copy.deepcopy(loaded.predictor.net), seed=9)
    predictor = FT.predictor_of(moved, loaded.predictor, name="x_ft")
    real = A.load_predictor

    def other(path: Any, *args: Any, **kwargs: Any) -> Any:  # reloads the SOURCE
        return real(tiny.artifact, *args, **kwargs)._replace(name="x_ft")

    monkeypatch.setattr(A, "load_predictor", other)
    with pytest.raises(FT.FinetuneError):
        FT.write_artifact(
            tmp_path,
            name="x_ft",
            featurizer=loaded.featurizer,
            predictor=predictor,
            extra={},
            check_rows=parts.domain,
        )
    assert (tmp_path / "artifact.pt.unverified").exists()
    assert not (tmp_path / "artifact.pt").exists()


def test_a_record_without_the_list_is_read_from_its_file_or_counts_as_all_seen(
    tiny: Tiny, tmp_path: Path
):
    """The first runs (2026-10-10) stored a hash of the games, not the games."""
    import hashlib

    allowed = FT.allowed_games(tiny.old_games)
    digest = hashlib.sha256("\n".join(sorted(allowed)).encode()).hexdigest()
    old = {
        "finetune": {
            "domain_games": N_ALLOWED,
            "allowed_games_file": str(tiny.old_games),
            "domain_used_ids_sha256": digest,
        }
    }
    assert FT.seen_games(old) == allowed
    with pytest.raises(FT.FinetuneError, match="in-sample"):
        FT.check_unseen(old, ["old-0"])
    FT.check_unseen(old, ["a-fresh-game"])
    # The file no longer gives that hash (or is gone): which games were seen
    # cannot be said, and every reading on own ladder games is refused.
    for record in (
        {**old["finetune"], "domain_used_ids_sha256": "0" * 64},
        {**old["finetune"], "allowed_games_file": str(tmp_path / "gone.jsonl")},
        {"domain_games": N_ALLOWED},
    ):
        with pytest.raises(FT.FinetuneError, match="does not list them"):
            FT.seen_games({"finetune": record})
        with pytest.raises(FT.FinetuneError, match="counts as seen"):
            FT.check_unseen({"finetune": record}, ["a-fresh-game"])
    with pytest.raises(FT.FinetuneError, match="must list the games"):
        FT.finetuned_extra({"name": "x"}, name="x_ft", record={"created": "now"})


def test_a_robustness_run_by_account_reports_and_writes_no_artifact(
    tiny: Tiny, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    real = FT.decide

    def taken(settings: Any) -> dict[str, Any]:
        found = real(settings)
        found.update({"verdict": FT.TAKEN})
        return found

    monkeypatch.setattr(FT, "decide", taken)
    out = tmp_path / "by_account"
    argv = arguments(tiny, out, "--learning-rates", "1e-3", "--passes", "1")
    assert FT.main([*argv, "--fold-unit", "account", "--no-artifact"]) == 0
    report = json.loads((out / FT.REPORT_JSON).read_text(encoding="utf-8"))
    assert report["status"] == FT.DONE and report["verdict"] == FT.TAKEN
    assert report["artifact"] is None and report["write_artifact"] is False
    assert not (out / FT.ARTIFACT_NAME).exists()
    assert not (out / (FT.ARTIFACT_NAME + FT.UNVERIFIED_SUFFIX)).exists()
    folds = report["folds"]
    assert folds["unit"] == FT.FOLD_ACCOUNT and folds["pre_registered"] is False
    # Six allowed games of five accounts: no account is on both sides.
    assert folds["accounts"] == 5 and folds["accounts_in_more_than_one_fold"] == 0
    assert sum(folds["games"]) == N_ALLOWED and min(folds["games"]) >= 1
    assert report["pre_registered_reading"] is False
    text = (out / FT.REPORT_MD).read_text(encoding="utf-8")
    assert "NOT the pre-registered folds" in text
    assert "No artifact: a robustness run" in text
    # The same run by game is the pre-registered unit (here with three folds).
    by_game = tmp_path / "by_game"
    assert (
        FT.main(
            [
                *arguments(tiny, by_game),
                "--learning-rates",
                "1e-3",
                "--passes",
                "1",
                "--no-artifact",
            ]
        )
        == 0
    )
    other = json.loads((by_game / FT.REPORT_JSON).read_text(encoding="utf-8"))
    assert other["folds"]["unit"] == FT.FOLD_GAME and other["artifact"] is None
