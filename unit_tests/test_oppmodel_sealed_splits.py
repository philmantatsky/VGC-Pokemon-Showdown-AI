"""Which splits the table fit and the calibration read; a coupled input.

Review fixes of 2026-10-10, second round. Each test states one finding and
fails on the code as it was:

* ``fit_oppmodel_tables.py`` scored the split ``test`` and the whole ladder
  holdout of every dataset and had no way not to: on a dataset of the fourth
  build that reads the sealed own games and the confirmation test split;
* ``calibrate_oppmodel.py`` read the same two splits as "informational";
* it also turned an artifact that carries a pair coupling into a calibrated
  one without it (since the artifact module refuses that write: it fitted
  first and failed at the save);
* the learning-curve driver did not know the trainer's ``--extras``.

An old dataset (no own game from the sealed day on, no ``own_cutoff`` in its
manifest) is read exactly as before: the tests hold that too.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from training import calibrate_oppmodel as K
from training import fit_oppmodel_tables as FT
from training import oppmodel_learning_curve as LC
from unit_tests import test_oppmodel_calibration as TK
from unit_tests import test_oppmodel_model as TM
from unit_tests import test_oppmodel_tables as TB
from vgc_bench.src.oppmodel import artifact as A
from vgc_bench.src.oppmodel import coupling as CP
from vgc_bench.src.oppmodel import features as F
from vgc_bench.src.oppmodel import model as M
from vgc_bench.src.oppmodel import tables as T

SEALED_FROM = F.local_time(F.OWN_SEALED_FROM)
EARLY, LATE = SEALED_FROM - 5 * 86_400, SEALED_FROM + 3_600
LADDER, TEST, SLICE, VAL = FT.LADDER, FT.TEST, FT.TIME_SLICE, FT.VALIDATION
SHARD = "shard-00000.npz"


def date_rows(
    dataset: Path, late: int = 0, cutoff: bool = False, splits: list[str] | None = None
) -> int:
    """Give every row of a one-shard dataset a time; returns the ladder rows.

    Human rows and the ladder rows are dated before the sealed day, except the
    last ``late`` ladder rows, dated on it. ``cutoff`` marks the manifest as a
    build made with ``--own-before``.
    """
    data = F.load_batch(dataset / SHARD)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    names = splits or list(manifest["splits"])
    when = np.full(int(data["m_split"].shape[0]), EARLY, dtype=np.int64)
    ladder = np.flatnonzero(data["m_split"] == names.index("ladder_holdout"))
    if late:
        when[ladder[-late:]] = LATE
    data["m_time"] = when
    F.save_batch(dataset / SHARD, data)
    if cutoff:
        manifest["own_cutoff"] = {"before": F.OWN_SEALED_FROM, "pages_later": 3}
        (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return int(ladder.size)


# --- the table fit ----------------------------------------------------------------


@pytest.fixture(scope="module")
def fz() -> F.Featurizer:
    return F.Featurizer.build(TM.repertoire())


@pytest.fixture()
def small_grids(monkeypatch: pytest.MonkeyPatch) -> None:
    """One or two values per strength, as ``test_oppmodel_tables`` does."""
    monkeypatch.setattr(FT, "KEY_GRID", {"key_strength": (30.0, 100.0)})
    monkeypatch.setattr(FT, "CELL_GRID", {"cell_strength": (30.0, 100.0)})
    monkeypatch.setattr(FT, "USAGE_GRID", {u: {} for u in FT.USAGE_GRID})
    monkeypatch.setattr(FT, "MEGA_GRID", {"mega_strength": (10.0,)})
    monkeypatch.setattr(
        FT, "ELO_EDGE_SETS", ((1300,), (1200, 1400), (1100, 1300, 1500))
    )
    monkeypatch.setattr(FT, "ELO_STRENGTHS", (50.0,))
    monkeypatch.setattr(FT, "ABLATIONS", FT.ABLATIONS[-1:])
    monkeypatch.setattr(FT, "RETUNED", ())


def test_the_plan_of_an_old_dataset_is_everything(tmp_path: Path, fz: F.Featurizer):
    """No own game from the sealed day on, no cut-off: all four splits, every
    row, whether the rows carry times or (a hand-made folder) none at all."""
    plain = tmp_path / "plain"
    TB.write_fit_dataset(plain, fz)
    dated = tmp_path / "dated"
    TB.write_fit_dataset(dated, fz)
    date_rows(dated)
    for dataset, known in ((plain, False), (dated, True)):
        plan = FT.evaluation_plan(dataset)
        assert plan["splits"] == list(FT.EVAL_SPLITS) and plan["not_scored"] == {}
        assert plan["default"] is True and plan["fourth_build"] is False
        assert plan["own_before"] is None and plan["sealed_rows_in_dataset"] == 0
        assert plan["own_time_known"] is known and plan["load_test"] is True
        # The flags change nothing there.
        again = FT.evaluation_plan(dataset, score_test=True, allow_sealed_holdout=True)
        assert again["splits"] == plan["splits"] and again["default"] is True
    # A choice among the four is honoured in the script's own order.
    chosen = FT.evaluation_plan(plain, f"{LADDER}, {VAL}")
    assert chosen["splits"] == [VAL, LADDER] and chosen["default"] is False
    assert chosen["load_test"] is False and set(chosen["not_scored"]) == {TEST, SLICE}
    assert FT.evaluation_plan(plain, [VAL, SLICE])["load_test"] is True
    for bad in ("", "train", f"{TEST},{LADDER}", "val,nope"):
        with pytest.raises(FT.FitError, match="--eval-splits"):
            FT.evaluation_plan(plain, bad)
    with pytest.raises(FT.FitError, match="cannot be read"):
        FT.evaluation_plan(tmp_path / "missing")


def test_the_plan_of_a_fourth_build_dataset_leaves_the_confirmation_data_out(
    tmp_path: Path, fz: F.Featurizer
):
    holds = tmp_path / "holds"
    TB.write_fit_dataset(holds, fz)
    date_rows(holds, late=10)
    plan = FT.evaluation_plan(holds)
    assert plan["splits"] == [VAL, LADDER] and plan["fourth_build"] is True
    assert set(plan["not_scored"]) == {TEST, SLICE}
    assert "--score-test" in plan["not_scored"][TEST]
    assert plan["sealed_rows_in_dataset"] == 10 and plan["sealed_rows_read"] is False
    assert plan["own_before"] == SEALED_FROM and plan["load_test"] is False
    assert plan["default"] is False
    # Each of the two is opened by its own flag, by name.
    test_only = FT.evaluation_plan(holds, score_test=True)
    assert test_only["splits"] == list(FT.EVAL_SPLITS)
    assert test_only["own_before"] == SEALED_FROM and test_only["default"] is False
    rows_only = FT.evaluation_plan(holds, allow_sealed_holdout=True)
    assert rows_only["splits"] == [VAL, LADDER] and rows_only["own_before"] is None
    assert rows_only["sealed_rows_read"] is True
    everything = FT.evaluation_plan(holds, score_test=True, allow_sealed_holdout=True)
    assert everything["default"] is True and everything["fourth_build"] is True
    # Naming test without the flag is refused, never dropped quietly.
    for named in (f"{VAL},{TEST}", f"{VAL},{SLICE},{LADDER}"):
        with pytest.raises(FT.FitError, match="--score-test"):
            FT.evaluation_plan(holds, named)
    assert FT.evaluation_plan(holds, f"{VAL},{TEST}", score_test=True)["splits"] == [
        VAL,
        TEST,
    ]
    # A build made with --own-before holds no sealed game, and its test split
    # is confirmation data all the same.
    cut = tmp_path / "cut"
    TB.write_fit_dataset(cut, fz)
    date_rows(cut, cutoff=True)
    plan = FT.evaluation_plan(cut)
    assert plan["built_with_own_cutoff"] is True and plan["fourth_build"] is True
    assert plan["splits"] == [VAL, LADDER] and plan["own_before"] is None
    assert "--own-before" in plan["not_scored"][TEST]
    assert plan["sealed_rows_in_dataset"] == 0
    assert FT.evaluation_plan(cut, score_test=True)["default"] is True


def test_a_fit_never_loads_what_its_plan_leaves_out(
    tmp_path: Path, fz: F.Featurizer, small_grids: None, monkeypatch: pytest.MonkeyPatch
):
    """THE FAILING CASE: the fit of 2026-10-10 01:11 scored ``test`` and a
    ladder holdout that pooled the sealed games. Now neither is even loaded,
    and the tables are the tables of the fit that reads everything."""
    data = tmp_path / "data"
    sizes = TB.write_fit_dataset(data, fz, late=6)
    ladder_rows = date_rows(data, late=10)
    assert ladder_rows == sizes[LADDER] == 24
    scored: list[int] = []
    real = T._CountTable.predict

    def watched(self: Any, batch: Any) -> Any:
        scored.append(int(np.asarray(batch["turn"]).shape[0]))
        return real(self, batch)

    monkeypatch.setattr(T._CountTable, "predict", watched)
    loaded: list[dict[str, Any]] = []
    real_load = FT.load_dataset

    def recording(directory: Any, *args: Any, **options: Any) -> Any:
        found = real_load(directory, *args, **options)
        loaded.append({"options": options, "rows": int(found[0]["turn"].shape[0])})
        return found

    monkeypatch.setattr(FT, "load_dataset", recording)
    lines: list[str] = []
    out = tmp_path / "fit"
    report = FT.run(data, out, log=lines.append)

    kept = sizes[LADDER] - 10
    assert report["evaluation_splits"] == [VAL, LADDER]
    assert report["dataset"]["evaluation_examples"] == {
        VAL: sizes[VAL] - 2,
        LADDER: kept,
    }
    assert report["dataset"]["splits_not_loaded"] == [TEST]
    assert report["dataset"]["examples"][TEST] == 0  # of the rows that were loaded
    assert report["dataset"]["fit_examples"] == sizes["train"]
    plan = report["evaluation_plan"]
    assert plan["sealed_rows_in_dataset"] == 10 and plan["sealed_rows_read"] is False
    # One load, without the test split and without the sealed rows.
    assert len(loaded) == 1
    assert loaded[0]["options"]["own_before"] == SEALED_FROM
    assert TEST not in loaded[0]["options"]["splits"]
    assert loaded[0]["rows"] == sizes["train"] + sizes[VAL] + kept
    # No predict call was ever handed a batch of the size of what was left out.
    assert max(scored) <= sizes["train"] and sizes[TEST] not in scored
    assert sizes[LADDER] not in scored and kept in scored
    assert any(f"split {TEST} is NOT scored" in line for line in lines)
    assert any("10 rows of own games" in line and "left out" in line for line in lines)
    for label, entry in report["tables"].items():
        assert set(entry["scores"]) == set(entry["calibration"]) == {VAL, LADDER}, label
        extra = A.load_predictor(out / FT.ARTIFACTS[label]).meta["extra"]
        assert set(extra["metrics"]) == set(extra["calibration"]) == {VAL, LADDER}
    for found in report["comparisons"].values():
        if "why" not in found:
            assert set(found) == {VAL, LADDER}
    assert all(set(row["splits"]) == {VAL, LADDER} for row in report["ablations"])
    text = (out / FT.REPORT_MD).read_text(encoding="utf-8")
    stored = json.loads((out / FT.REPORT_JSON).read_text(encoding="utf-8"))
    assert text == FT.render_markdown(stored)
    assert f"Scored: `{VAL}`, `{LADDER}`. Not scored: `{TEST}`" in text
    assert "10 rows of the bot's own games" in text and "never loaded" in text
    assert f"| model | {VAL} | {LADDER} |" in text and f"| {TEST} |" not in text

    # Asked for by name: everything, as the script always read it.
    scored.clear()
    whole = tmp_path / "whole"
    full = FT.run(
        data, whole, log=lines.append, score_test=True, allow_sealed_holdout=True
    )
    assert full["evaluation_splits"] == list(FT.EVAL_SPLITS)
    assert full["dataset"]["evaluation_examples"] == {
        VAL: sizes[VAL] - 2,
        TEST: sizes[TEST],
        SLICE: 6,
        LADDER: sizes[LADDER],
    }
    assert full["evaluation_plan"]["sealed_rows_read"] is True
    assert loaded[1]["options"] == {} and sizes[TEST] in scored
    assert "READ (--allow-sealed-holdout)" in (whole / FT.REPORT_MD).read_text()
    # What is scored changes nothing that is fitted: the same four tables,
    # and on validation and the old ladder games the same numbers.
    probe = TB.features_only(TB.random_batch(31))
    monkeypatch.setattr(T._CountTable, "predict", real)
    for label, file_name in FT.ARTIFACTS.items():
        one = A.load_predictor(out / file_name).predictor
        two = A.load_predictor(whole / file_name).predictor
        assert one.config == two.config, label
        first, second = one.predict(probe), two.predict(probe)
        assert all(np.array_equal(first[head], second[head]) for head in first), label
        assert (
            report["tables"][label]["scores"][VAL]
            == full["tables"][label]["scores"][VAL]
        )
    # The command line: the flags, and a named test split without --score-test.
    args = FT.parse_args(["--eval-splits", "val", "--score-test"])
    assert (args.eval_splits, args.score_test, args.allow_sealed_holdout) == (
        "val",
        True,
        False,
    )
    refused = tmp_path / "refused"
    base = ["--dataset", str(data), "--out", str(refused)]
    assert FT.main([*base, "--eval-splits", f"{VAL},{TEST}"]) == 1
    assert not refused.exists()


def test_an_old_dataset_is_fitted_and_reported_as_before(
    tmp_path: Path, fz: F.Featurizer, small_grids: None, monkeypatch: pytest.MonkeyPatch
):
    data = tmp_path / "data"
    sizes = TB.write_fit_dataset(data, fz, late=6)
    date_rows(data)
    calls: list[dict[str, Any]] = []
    real_load = FT.load_dataset

    def recording(directory: Any, *args: Any, **options: Any) -> Any:
        calls.append({"args": args, "options": options})
        return real_load(directory, *args, **options)

    monkeypatch.setattr(FT, "load_dataset", recording)
    lines: list[str] = []
    report = FT.run(data, tmp_path / "fit", log=lines.append)
    assert calls == [{"args": (), "options": {}}]  # the whole dataset, one call
    # The report of before, key for key: no plan, no list of splits.
    assert "evaluation_splits" not in report and "evaluation_plan" not in report
    assert "splits_not_loaded" not in report["dataset"]
    assert report["dataset"]["evaluation_examples"] == {
        VAL: sizes[VAL] - 2,
        TEST: sizes[TEST],
        SLICE: 6,
        LADDER: sizes[LADDER],
    }
    assert not any("NOT scored" in line or "left out (give" in line for line in lines)
    text = (tmp_path / "fit" / FT.REPORT_MD).read_text(encoding="utf-8")
    assert "Scored:" not in text
    assert "| model | " + " | ".join(FT.EVAL_SPLITS) + " |" in text
    assert FT.scored_splits(report) == list(FT.EVAL_SPLITS)
    for label in FT.ARTIFACTS:
        extra = A.load_predictor(tmp_path / "fit" / FT.ARTIFACTS[label]).meta["extra"]
        assert set(extra["metrics"]) == set(FT.EVAL_SPLITS)


# --- the calibration --------------------------------------------------------------


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory, fz: F.Featurizer) -> TK.World:
    return TK.World(tmp_path_factory.mktemp("sealed_calibration"), fz, tiles=12)


def dated_copy(world: TK.World, target: Path, **options: Any) -> int:
    """The world's dataset in another folder, its rows dated (``date_rows``).

    The manifest is the same file unless a cut-off is written, so the
    artifact is still "trained on this build"."""
    shutil.copytree(world.dataset, target)
    return date_rows(target, splits=TK.SPLITS, **options)


def arguments(*extra: str) -> Any:
    return K.parse_args(["--out", "unused", *extra])


def test_the_informational_plan(world: TK.World, tmp_path: Path):
    both = list(K.INFORMATIONAL_SPLITS)
    # An old dataset: both splits, whole. Rows without times read the same.
    for dataset in (world.dataset, tmp_path / "old"):
        if not dataset.exists():
            dated_copy(world, dataset)
        plan = K.informational_plan(arguments(), dataset)
        assert plan["splits"] == both and plan["not_read"] == {}
        assert plan["default"] is True and plan["own_before"] is None
        assert plan["fourth_build"] is False
    assert K.informational_plan(arguments(), world.dataset)["own_time_known"] is False
    only = K.informational_plan(
        arguments("--informational-splits", "test"), world.dataset
    )
    assert only["splits"] == [K.SPLIT_TEST] and only["default"] is False
    none = K.informational_plan(
        arguments("--informational-splits", "none"), world.dataset
    )
    assert none["splits"] == [] and set(none["not_read"]) == set(both)
    for bad in ("", "val", "test,nope", "none,test"):
        with pytest.raises(K.CalibrationError, match="--informational-splits"):
            K.informational_plan(
                arguments("--informational-splits", bad), world.dataset
            )
    # A dataset that holds sealed own games.
    holds = tmp_path / "holds"
    rows = dated_copy(world, holds, late=24)
    plan = K.informational_plan(arguments(), holds)
    assert plan["splits"] == [K.SPLIT_LADDER] and plan["fourth_build"] is True
    assert "--score-test" in plan["not_read"][K.SPLIT_TEST]
    assert plan["sealed_rows_in_dataset"] == 24 < rows
    assert plan["own_before"] == SEALED_FROM and plan["sealed_rows_read"] is False
    opened = K.informational_plan(
        arguments("--score-test", "--allow-sealed-holdout"), holds
    )
    assert opened["splits"] == both and opened["own_before"] is None
    assert opened["default"] is True and opened["sealed_rows_read"] is True
    with pytest.raises(K.CalibrationError, match="--score-test"):
        K.informational_plan(arguments("--informational-splits", "test"), holds)
    # A build made with --own-before: no sealed row, the test split still shut.
    cut = tmp_path / "cut"
    dated_copy(world, cut, cutoff=True)
    plan = K.informational_plan(arguments(), cut)
    assert plan["splits"] == [K.SPLIT_LADDER] and plan["own_before"] is None
    assert plan["built_with_own_cutoff"] is True
    assert "--own-before" in plan["not_read"][K.SPLIT_TEST]
    with pytest.raises(K.CalibrationError, match="cannot be read"):
        K.informational_plan(arguments(), tmp_path / "missing")


def test_a_calibration_reads_no_confirmation_data_unasked(
    world: TK.World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """THE FAILING CASE: the informational pass scored the whole test split
    and the whole ladder holdout of whatever dataset it was given."""
    holds = tmp_path / "holds"
    ladder_rows = dated_copy(world, holds, late=24)
    lines: list[str] = []
    monkeypatch.setattr(K, "say", lines.append)
    loads: list[tuple[tuple[str, ...], Any]] = []
    real = K.T.load_splits

    def recording(directory: Any, names: Any, own_before: Any = None) -> Any:
        loads.append((tuple(names), own_before))
        return real(directory, names, own_before)

    monkeypatch.setattr(K.T, "load_splits", recording)
    out = tmp_path / "run"
    argv = ["--artifact", str(world.source), "--dataset", str(holds)]
    assert K.main([*argv, "--out", str(out), "--resamples", "100"]) == 0, lines[-1]
    report = json.loads((out / K.REPORT_JSON).read_text(encoding="utf-8"))
    assert report["status"] == K.DONE
    assert set(report["informational"]) == {K.SPLIT_LADDER}
    assert loads == [((K.SPLIT_VAL,), None), ((K.SPLIT_LADDER,), SEALED_FROM)]
    plan = report["informational_plan"]
    assert plan["splits"] == [K.SPLIT_LADDER] and plan["sealed_rows_in_dataset"] == 24
    predicted = [line.strip() for line in lines if "PREDICT" in line]
    assert predicted and not any(" test " in line for line in predicted)
    kept = ladder_rows - 24
    assert any(
        line.startswith(f"PREDICT {K.SPLIT_LADDER} before {kept}/{kept} examples")
        for line in predicted
    )
    assert not any(f"/{ladder_rows} examples" in line for line in predicted)
    assert any(f"INFORMATIONAL {K.SPLIT_TEST}: NOT read" in line for line in lines)
    assert any("24 rows of own games" in line and "left out" in line for line in lines)
    slots = report["informational"][K.SPLIT_LADDER]["events"][K.C.EVENT_SWITCH]["slots"]
    assert 0 < slots <= 2 * kept
    text = (out / K.REPORT_MD).read_text(encoding="utf-8")
    assert f"- informational splits read: `{K.SPLIT_LADDER}`; not read: `test`" in text
    assert "24 rows of the bot's own games" in text and "were left out" in text
    # The calibration itself is fitted on validation alone: the maps are the
    # maps of a run on the dataset without any date.
    lines.clear()
    plain = tmp_path / "plain"
    argv = ["--artifact", str(world.source), "--dataset", str(world.dataset)]
    assert K.main([*argv, "--out", str(plain), "--resamples", "100"]) == 0, lines[-1]
    old = json.loads((plain / K.REPORT_JSON).read_text(encoding="utf-8"))
    assert old["calibration"]["maps"] == report["calibration"]["maps"]
    # ... and an old dataset is read as before: both splits, whole, no plan.
    assert set(old["informational"]) == set(K.INFORMATIONAL_SPLITS)
    assert "informational_plan" not in old
    assert loads[-1] == (tuple(K.INFORMATIONAL_SPLITS), None)
    assert "informational splits read" not in (plain / K.REPORT_MD).read_text()
    # A named test split without --score-test is refused before anything is
    # predicted, and nothing is written.
    lines.clear()
    refused = tmp_path / "refused"
    argv = ["--artifact", str(world.source), "--dataset", str(holds)]
    code = K.main([*argv, "--out", str(refused), "--informational-splits", "test"])
    assert code == 1 and lines[-1].startswith(K.FAILED) and "--score-test" in lines[-1]
    assert not any(line.strip().startswith(("PREDICT", "FIT")) for line in lines)
    assert not refused.exists()


def test_a_coupled_artifact_is_refused_before_anything_is_fitted(
    world: TK.World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """THE FAILING CASE: a coupled input was fitted, and only the save of the
    calibrated copy stopped the run (before the artifact module refused that
    write, the copy was a plain joint under the coupled name)."""
    rng = np.random.default_rng(7)
    table = CP.symmetrise(np.exp(rng.normal(0.0, 0.5, (CP.N_BUCKET, 8, 8))))
    pair = CP.PairCoupling.build(
        table,
        name="unit_pair",
        fitted_after=CP.fitted_after(M.KIND, world.predictor.describe()),
    )
    coupled = tmp_path / "coupled" / "artifact.pt"
    A.save_artifact(
        coupled,
        kind=M.KIND,
        name="tiny_pair",
        featurizer=world.fz,
        predictor_payload=world.predictor.to_payload(),
        extra={"dataset": {"tag": "tiny", "manifest_sha256": world.manifest_sha}},
        coupling=pair,
    )
    assert A.load_predictor(coupled).coupling is not None
    lines: list[str] = []
    monkeypatch.setattr(K, "say", lines.append)
    out = tmp_path / "run"
    assert world.run(out, source=coupled) == 1
    last = lines[-1]
    assert last.startswith(f"{K.FAILED} CalibrationError")
    assert "carries a pair coupling (unit_pair)" in last
    assert "fit the coupling again on the calibrated artifact" in last
    # Before anything was predicted, fitted or written.
    started = ("CAL_START", "PREDICT", "FIT", "SAVED")
    assert not any(line.strip().startswith(started) for line in lines)
    assert not out.exists()
    # Not even --refit (which drops an event calibration) carries it over.
    assert world.run(out, "--refit", source=coupled) == 1
    assert "pair coupling" in lines[-1] and not out.exists()
    # The same artifact without the coupling is calibrated as always.
    lines.clear()
    assert world.run(out, "--limit", "40", "--informational-splits", "none") == 0
    report = json.loads((out / K.REPORT_JSON).read_text(encoding="utf-8"))
    assert report["status"] == K.DONE and report["informational"] == {}
    assert report["informational_plan"]["splits"] == []


# --- the learning curve -----------------------------------------------------------


def test_the_curve_compares_the_extras_mask():
    assert "extras" in LC.HYPER_KEYS and LC.HYPER_UNRECORDED == {"extras": 0}
    # A report from before the flag is a fit without extras.
    assert (
        LC.hyper_value({}, "extras") == 0 == LC.hyper_value({"extras": None}, "extras")
    )
    assert LC.hyper_value({"extras": 7}, "extras") == 7
    assert LC.hyper_value({}, "lr") is None and LC.hyper_value({"lr": 0.1}, "lr") == 0.1
    old = {"epochs": 40, "seed": 1, "lr": 0.001}
    new = {**old, "extras": 0}
    read = {**old, "extras": 15}

    def rows(first: dict[str, Any], second: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {"name": "small", "train_args": dict(first)},
            {"name": "full", "train_args": dict(second)},
        ]

    # An old fit beside a new one without extras: alike.
    assert LC.hyper_warnings(rows(old, new)) == []
    assert LC.hyper_warnings(rows(new, old)) == []
    # THE FAILING CASE: two points fitted with different masks were alike.
    for first, second in ((old, read), (new, read), (read, old)):
        said = LC.hyper_warnings(rows(first, second))
        assert len(said) == 1 and "extras" in said[0], (first, second)
    curve = TLC_curve(rows(new, read))
    assert any("extras 0 vs 15" in line for line in LC.reading_problems(curve))
    assert not any(
        "extras" in line for line in LC.reading_problems(TLC_curve(rows(old, new)))
    )


def TLC_curve(points: list[dict[str, Any]]) -> dict[str, Any]:
    """As much of a curve as ``reading_problems`` reads."""
    return {
        "settings": {
            "smoke": False,
            "build_limit": None,
            "train_limit": None,
            "tables_limit": None,
        },
        "steps": [{"from": "small", "to": "full"}],
        "points": points,
        "corpus": {},
    }
