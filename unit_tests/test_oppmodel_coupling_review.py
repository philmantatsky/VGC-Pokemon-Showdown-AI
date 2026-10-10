"""What the review of the pair coupling found, pinned (2026-10-10).

Each test fails on the code the reviewers read: the margins of a one-target
fit, a coupling dropped by a writer that does not know about couplings, the
coupling lost by ``_replace``, pair failures nobody could see, a load failure
nobody was told about, and the fit script's status, rule record and sealed
games. Synthetic rows and tiny artifacts; nothing here reads a dataset of the
experiment.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from evaluation import oppmodel_scorecard as SC
from training import fit_oppmodel_coupling as FC
from unit_tests.test_oppmodel_coupling import some_coupling, some_table
from unit_tests.test_oppmodel_runtime import (
    filled,
    forecasts,
    hashed,
    repertoire,
    runtime,
)
from unit_tests.test_oppmodel_scorecard import make_card
from vgc_bench.src.oppmodel import artifact as A
from vgc_bench.src.oppmodel import coupling as CP
from vgc_bench.src.oppmodel import features as F
from vgc_bench.src.oppmodel import joint as J
from vgc_bench.src.oppmodel import runtime as R
from vgc_bench.src.oppmodel import tables as T

SPLITS = ["train", "val", "test", "ladder_holdout"]
A_, B_ = CP.C_FOE_A, CP.C_FOE_B


# --- the margins of a one-target fit ----------------------------------------------


def one_target_rows(
    n: int, seed: int, both: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows that face ONE opposing slot: each holds one attack class, never
    the other (``both``: rows that hold both, as with two targets). The true
    pair is drawn from the product times a planted table that has an
    interaction AND a per-class factor on the attacks, so D is far from 1.
    Returns (model, seen, the bucket's real mix with the occupied slot = a).
    """
    rng = np.random.default_rng(seed)
    a = rng.dirichlet(np.full(8, 0.9), n)
    b = rng.dirichlet(np.full(8, 0.9), n)
    gone = np.where(rng.random(n) < 0.5, A_, B_)  # the class of the empty slot
    if not both:
        a[np.arange(n), gone] = 0.0
        b[np.arange(n), gone] = 0.0
        a /= a.sum(1, keepdims=True)
        b /= b.sum(1, keepdims=True)
    model = a[:, :, None] * b[:, None, :]
    per_class = np.ones(8)
    per_class[[A_, B_]] = 1.8
    planted = CP.symmetrise(some_table(41, spread=0.4)[2])
    planted = planted * per_class[:, None] * per_class[None, :]
    tilted = model * planted[None]
    tilted /= tilted.sum((1, 2), keepdims=True)
    flat = tilted.reshape(n, 64)
    cell = (flat.cumsum(1) > rng.random((n, 1))).argmax(1)
    seen = np.zeros((n, 64))
    seen[np.arange(n), cell] = model.reshape(n, 64)[np.arange(n), cell]
    # the mix every row really has: its attack mass on ONE class
    turned = model.copy()
    flip = gone == A_  # the occupied slot is b: rename it a
    turned[flip] = model[flip][:, CP.EXCHANGE][:, :, CP.EXCHANGE]
    real = 0.5 * (turned.sum(2).mean(0) + turned.sum(1).mean(0))
    return model, seen.reshape(n, 8, 8), real


def test_a_one_target_fit_is_margin_neutral_under_the_mix_a_row_has():
    """The later-one bucket: a row holds one attack class, so the cell pairing
    the two has no mass and keeps its prior value. The normal form weighed it
    as if it had half of the attacks, and part of D stayed in the stored
    table (the attack margin read 1.07 on the first real fit)."""
    model, seen, real = one_target_rows(30000, 1)
    assert real[B_] == 0.0 and real[A_] > 0.1
    assert CP.attack_classes_merge(model) and not (model[:, A_, B_] > 0).any()
    bucket = np.full(len(model), CP.BUCKET_ONE)
    table, parts = CP.fit_tables(model, seen, None, bucket, 5.0, False)
    part = parts[CP.BUCKETS[CP.BUCKET_ONE]]
    assert part["attack_classes"] == CP.ATTACKS_MERGED and part["converged"]
    assert part["buckets"] == [CP.BUCKET_ONE]
    stored = table[CP.BUCKET_ONE]
    live = real > 0
    # unit margins under the mix a row has, and under the pooled mix
    assert np.allclose((stored @ real)[live], 1.0, atol=1e-9)
    assert np.allclose(stored @ part["mix"], 1.0, atol=1e-9)
    # the two attack classes are one class of this table
    assert stored[A_, B_] == stored[A_, A_] == stored[B_, B_]
    assert np.array_equal(stored[A_], stored[B_]) and CP.is_symmetric(table)
    assert part["D"][A_] == part["D"][B_] and part["D"][A_] > 1.3  # D is not 1
    assert np.allclose(
        part["D"][:, None] * part["D"][None, :] * part["R"], part["T"], rtol=1e-9
    )
    # the rule of before leaves the attack margin off 1 by a per-class amount
    tilt, _ = CP.fit_tilt(model, seen, None, 5.0)
    before, _ = CP.normal_form(tilt, CP.class_mix(model))
    assert abs(float((before @ real)[A_]) - 1.0) > 0.03
    # and the coupled joint's class marginals move less with the new table
    turned_drift = []
    for rule in (before, stored):
        coupled = model * rule
        coupled /= coupled.sum((1, 2), keepdims=True)
        moved = 0.5 * (coupled.sum(2) + coupled.sum(1)) - 0.5 * (
            model.sum(2) + model.sum(1)
        )
        attack = moved[:, A_] + moved[:, B_]
        turned_drift.append(abs(float(attack.mean())))
    assert turned_drift[1] < turned_drift[0]


def test_a_fit_whose_rows_hold_both_attack_classes_is_the_fit_of_before():
    model, seen, _ = one_target_rows(6000, 2, both=True)
    assert not CP.attack_classes_merge(model)
    bucket = np.where(np.arange(len(model)) % 4 == 0, CP.BUCKET_ONE, CP.BUCKET_TWO)
    table, parts = CP.fit_tables(model, seen, None, bucket, 20.0, False)
    for name, part in parts.items():
        assert part["attack_classes"] == CP.ATTACKS_SEPARATE, name
        rows = np.isin(bucket, part["buckets"])
        tilt, _ = CP.fit_tilt(model[rows], seen[rows], None, 20.0)
        rest, scale = CP.normal_form(tilt, CP.class_mix(model[rows]))
        assert np.array_equal(part["R"], rest) and np.array_equal(part["D"], scale)
        assert np.array_equal(part["T"], tilt)
    assert set(parts) == {"pooled", "later_one"}
    # merged is decided by rows of positive weight, and never without attacks
    weight = np.ones(len(model))
    assert not CP.attack_classes_merge(model, weight)
    none = model.copy()
    none[:, [A_, B_], :] = 0.0
    none[:, :, [A_, B_]] = 0.0
    assert not CP.attack_classes_merge(none) and not CP.attack_classes_merge(none[:0])
    single, _, _ = one_target_rows(50, 3)
    mixed = np.concatenate([single, model[:1]])
    assert not CP.attack_classes_merge(mixed)
    assert CP.attack_classes_merge(mixed, np.r_[np.ones(50), 0.0])


def test_the_full_tilt_is_reported_beside_the_stored_table():
    """What discarding D gives up is a number of every fit, not a surprise."""
    model, seen, _ = one_target_rows(8000, 4, both=True)
    bucket = np.where(np.arange(len(model)) % 5 == 0, CP.BUCKET_ONE, CP.BUCKET_TWO)
    battle = np.arange(len(model)) // 8
    fit = CP.fit_coupling(model, seen, None, bucket, battle, kappas=(20.0,))
    whole = CP.full_tilt(fit["parts"])
    assert whole.shape == (3, 8, 8)
    for part in fit["parts"].values():
        for index in part["buckets"]:
            assert np.array_equal(whole[index], part["T"])
    own = CP.table_gain(whole, model, seen, bucket).mean()
    assert fit["in_sample_gain_full_tilt"] == pytest.approx(own, rel=1e-12)
    # the planted per-class factor is in T and not in R
    assert fit["in_sample_gain_full_tilt"] > fit["in_sample_gain"] + 0.01
    assert CP.full_tilt({}).tolist() == np.ones((3, 8, 8)).tolist()
    # the design's wording of the rule is reported, never the verdict
    assert fit["folds_positive"] == 10 and fit["design_reading"] is True
    none = CP.fit_coupling(model[:4], seen[:4], None, bucket[:4], battle[:4])
    assert none["design_reading"] is False and none["folds_positive"] == 0


# --- a coupling is never dropped without a word -----------------------------------


@pytest.fixture(scope="module")
def fz() -> F.Featurizer:
    return F.Featurizer.build(repertoire())


def moves_by_class(fz: F.Featurizer) -> tuple[int, int]:
    """(an aimed attack, a Protect-family move) as move ids, found by their
    intent rows in the featurizer's own table (never by name)."""
    intent = np.asarray(fz.tables.move_intent).astype(np.int64)
    aimed = np.nonzero((intent[:, 0] == A_) & (intent[:, 1] == B_))[0]
    guard = np.nonzero((intent == CP.C_PROTECT).all(1))[0]
    return int(aimed[0]), int(guard[0])


def paired_rows(fz: F.Featurizer, n: int, seed: int, one_target: int = 0) -> F.Batch:
    """``n`` two-slot rows whose slots choose TOGETHER: both use the
    Protect-family candidate, or both attack. The last ``one_target`` rows
    face one opposing slot."""
    rng = np.random.default_rng(seed)
    attack, guard = moves_by_class(fz)
    width = F.N_CAND_DEFAULT
    other = F.other_index(width)
    batch = {
        name: np.zeros((n, *spec.shape), dtype=spec.dtype)
        for name, spec in fz.layout().items()
    }
    for name in ("act_mon", "foe_mon", "y_kind", "y_action", "y_target", "y_mega"):
        batch[name][:] = -1
    batch["y_intent"][:] = -1
    for i in range(n):
        alone = i >= n - one_target
        batch["turn"][i] = 1 if i % 6 == 0 and not alone else 2 + i % 3
        batch["foe_mon"][i] = (6, -1) if alone else (6, 7)
        together = rng.random() < 0.5
        for slot in range(2):
            batch["act_mon"][i, slot] = slot
            batch["mon_id"][i, slot, F.ID_KEY] = 5 + slot
            batch["cand_move"][i, slot, 0] = attack
            batch["cand_flag"][i, slot, 0] = F.CAND_VALID | F.CAND_DAMAGING
            batch["cand_move"][i, slot, 1] = guard
            batch["cand_flag"][i, slot, 1] = F.CAND_VALID | F.CAND_PROTECT
            aims = 0b00001 if alone else 0b00011
            batch["cand_tmask"][i, slot, 0] = aims
            batch["cand_tmask"][i, slot, 1] = aims
            batch["cand_tmask"][i, slot, other] = 0b11111
            batch["action_mask"][i, slot, [0, 1, other]] = 1
            action = 1 if together else 0
            batch["y_action"][i, slot] = action
            batch["y_set"][i, slot, action] = 1
            batch["y_target"][i, slot] = 0 if alone else rng.integers(0, 2)
    return batch


def with_meta(batch: F.Batch, split: str, first_battle: int, when: int) -> F.Batch:
    n = int(batch["turn"].shape[0])
    out = dict(batch)
    out["m_split"] = np.full(n, SPLITS.index(split), dtype=np.uint8)
    out["m_battle"] = (first_battle + np.arange(n) // 8).astype(np.int32)
    out["m_weight"] = np.ones(n, dtype=np.float32)
    out["m_flag"] = np.zeros(n, dtype=np.uint8)
    out["m_time"] = np.full(n, when, dtype=np.int64)
    return out


def write_table(path: Path, fz: F.Featurizer, table: Any, **more: Any) -> Path:
    more.setdefault("extra", {"dataset_tag": "unit"})
    A.save_artifact(
        path,
        kind=A.KIND_TABLE,
        name=table.name,
        featurizer=fz,
        predictor_payload=table.to_payload(),
        **more,
    )
    return path


def resaved_by_an_unaware_writer(source: Path, target: Path, **more: Any) -> Path:
    """What a calibration or a fine-tune does with a loaded artifact: the
    source's ``extra`` copied, the predictor written again, no coupling."""
    document = A.read_artifact(source)
    loaded = A.load_predictor(source)
    A.save_artifact(
        target,
        kind=loaded.kind,
        name=loaded.name + "_cal",
        featurizer=loaded.featurizer,
        predictor_payload=document["predictor"],
        extra=dict(loaded.meta.get("extra") or {}),
        **more,
    )
    return target


def test_a_writer_that_knows_no_coupling_cannot_drop_one_silently(tmp_path: Path, fz):
    table = T.FlagsTable.fit(paired_rows(fz, 32, 0), featurizer=fz)
    made = CP.PairCoupling.build(
        some_table(51), name="unit_pair", fitted_after=CP.fitted_after("table", None)
    )
    source = write_table(tmp_path / "coupled.pt", fz, table, coupling=made)
    # a coupled file names its coupling in extra, whoever wrote it
    stored = A.read_artifact(source)
    assert stored["version"] == A.VERSION_COUPLED
    assert stored["extra"][A.EXTRA_COUPLING] == {"name": "unit_pair"}
    assert stored["extra"]["dataset_tag"] == "unit"
    own = {"dataset_tag": "unit", A.EXTRA_COUPLING: {"kappa": 5.0}}
    kept = write_table(tmp_path / "own.pt", fz, table, coupling=made, extra=own)
    assert A.read_artifact(kept)["extra"][A.EXTRA_COUPLING] == {"kappa": 5.0}

    # the re-save of before: version 1, no coupling, the coupled record kept
    with pytest.raises(ValueError, match="names a pair coupling and none is passed"):
        resaved_by_an_unaware_writer(source, tmp_path / "dropped.pt")
    assert not (tmp_path / "dropped.pt").exists()
    assert not list(tmp_path.glob("dropped.pt.tmp*"))

    # dropping on purpose is said in the file
    told = resaved_by_an_unaware_writer(
        source, tmp_path / "told.pt", coupling_dropped="fitted after another state"
    )
    document = A.read_artifact(told)
    assert document["version"] == A.VERSION and A.KEY_COUPLING not in document
    assert A.EXTRA_COUPLING not in document["extra"]
    assert document["extra"][A.EXTRA_COUPLING_DROPPED] == {
        "reason": "fitted after another state",
        "was": {"name": "unit_pair"},
    }
    plain = A.load_predictor(told)
    assert type(plain) is A.LoadedPredictor and plain.coupling is None
    # ... and such a file can be re-saved again by anyone
    again = resaved_by_an_unaware_writer(told, tmp_path / "again.pt")
    assert A.EXTRA_COUPLING_DROPPED in A.read_artifact(again)["extra"]

    # passed and dropped at once is a contradiction
    with pytest.raises(ValueError, match="either passed or dropped"):
        write_table(
            tmp_path / "both.pt", fz, table, coupling=made, coupling_dropped="x"
        )
    # nothing to drop: the file of before, key for key
    bare = write_table(tmp_path / "bare.pt", fz, table, coupling_dropped="nothing")
    assert A.read_artifact(bare)["extra"] == {"dataset_tag": "unit"}
    assert A.read_artifact(bare)["version"] == A.VERSION


def test_a_plain_file_under_a_coupled_record_is_refused_by_every_reader(
    tmp_path: Path, fz
):
    """The file the re-save of before wrote (it may exist on a disk): version
    1, no coupling, ``extra`` still naming one. No reader serves it."""
    table = T.FlagsTable.fit(paired_rows(fz, 32, 1), featurizer=fz)
    made = CP.PairCoupling.build(some_table(52), name="unit_pair")
    good = write_table(tmp_path / "coupled.pt", fz, table, coupling=made)
    raw = torch.load(good, map_location="cpu", weights_only=True)
    raw.pop(A.KEY_COUPLING)
    raw["version"] = A.VERSION
    stale = tmp_path / "stale.pt"
    torch.save(raw, stale)
    with pytest.raises(ValueError, match="does not carry"):
        A.read_artifact(stale)
    with pytest.raises(ValueError, match="does not carry"):
        A.load_predictor(stale)
    assert A.try_load_predictor(stale) is None
    served = R.OpponentPredictor.load(stale)
    assert not served.loaded and "does not carry" in str(served.load_failure)
    assert served.counters["load_error:ValueError"] == 1


def test_replace_keeps_the_coupling_or_refuses(tmp_path: Path, fz):
    table = T.FlagsTable.fit(paired_rows(fz, 32, 2), featurizer=fz)
    made = CP.PairCoupling.build(
        some_table(53), name="unit_pair", fitted_after=CP.fitted_after("table", None)
    )
    loaded = A.load_predictor(write_table(tmp_path / "c.pt", fz, table, coupling=made))
    assert isinstance(loaded, A.CoupledPredictor)
    renamed = loaded._replace(name="x")
    assert type(renamed) is A.CoupledPredictor and renamed.name == "x"
    assert renamed.coupling is loaded.coupling and renamed.coupling.same_as(made)
    assert renamed.predictor is loaded.predictor and len(renamed) == 5
    assert copy.copy(loaded).coupling is loaded.coupling
    # the same predictor again fits; another kind does not, and says so
    assert loaded._replace(predictor=loaded.predictor).coupling is loaded.coupling
    with pytest.raises(ValueError, match="does not fit the replaced predictor"):
        loaded._replace(kind=A.KIND_OPPNET)

    class Undescribed:
        def describe(self) -> dict[str, Any]:
            raise RuntimeError("no")

    with pytest.raises(ValueError, match="cannot be described"):
        loaded._replace(predictor=Undescribed())
    # an artifact without a coupling: the tuple's own _replace, as before
    plain = A.load_predictor(write_table(tmp_path / "p.pt", fz, table))
    other = plain._replace(name="y", kind=A.KIND_OPPNET)
    assert type(other) is A.LoadedPredictor and other.coupling is None


# --- pair failures can be seen ----------------------------------------------------


def both(made: R.Forecast | None) -> bool:
    return made is not None and made.a is not None and made.b is not None


def test_a_pair_method_that_cannot_answer_is_counted_where_diagnostics_read(fz):
    rt, _ = runtime(fz, hashed, coupling=some_coupling(54))
    forecast = next(made for made in forecasts(rt, "p1").values() if both(made))
    assert forecast is not None and forecast.joint_inputs is not None
    first = filled(forecast.slots[0]).actions[0]
    second = filled(forecast.slots[1]).actions[0]
    before = dict(rt.diagnostics()["process"])
    healthy = forecast.pair_weight(first, second)
    assert len(forecast.joint_top(8)) == 8
    assert forecast.reply_weight(None, (R.ACTION_SWITCH,)) == 1.0  # a pass
    assert rt.diagnostics()["process"] == before  # nothing failed: nothing counted

    hurt = R.Forecast(
        **{
            **{name: getattr(forecast, name) for name in forecast.__dataclass_fields__},
            "joint_inputs": {
                key: value
                for key, value in forecast.joint_inputs.items()
                if key != CP.KEY_CLASS
            },
        }
    )
    assert hurt.pair is not None
    assert hurt.pair_weight(first, second) == 1.0 != healthy
    assert hurt.joint_top(8) == ()
    assert hurt.reply_weight((R.ACTION_SWITCH,), (R.ACTION_SWITCH,)) == 1.0
    after = rt.diagnostics()["process"]

    def grown(name: str) -> int:
        return after.get(name, 0) - before.get(name, 0)

    assert grown("runtime:" + R.PAIR_UNCLASSED) == 1
    assert grown("runtime:" + R.JOINT_NOT_BUILT) == 1
    assert grown("runtime:" + R.REPLY_UNREAD) == 1
    assert grown("joint:joint_replies:ValueError") == 1
    assert grown("coupling:reply_maps:ValueError") == 1
    assert after == R.process_counters()
    # an entry of another forecast, an unknown target: counted too
    stranger = R.ActionForecast(R.ACTION_MOVE, 0.5, "nosuchmove", "foe_a")
    assert forecast.pair_weight(stranger, second) == 1.0
    named = next(name for name in filled(forecast.slots[0]).moves if name)
    assert forecast.reply_weight(("move", named, "up"), (R.ACTION_SWITCH,)) == 1.0
    last = rt.diagnostics()["process"]
    for name in ("runtime:" + R.PAIR_UNCLASSED, "runtime:" + R.REPLY_UNREAD):
        assert last[name] == after[name] + 1
    # the runtime's own three groups are still there, and its own counters
    assert set(rt.diagnostics()) == {"runtime", "featurizer", "predictor", "process"}


def test_one_without_a_coupling_counts_nothing_for_its_honest_one(fz):
    rt, _ = runtime(fz, hashed)
    forecast = next(made for made in forecasts(rt, "p1").values() if both(made))
    assert forecast is not None and forecast.pair is None
    first = filled(forecast.slots[0]).actions[0]
    second = filled(forecast.slots[1]).actions[0]
    before = R.process_counters()
    assert forecast.pair_weight(first, second) == 1.0
    assert forecast.pair_weight(None, object()) == 1.0
    assert forecast.reply_weight(("move", "x", "up"), (R.ACTION_SWITCH,)) == 1.0
    assert len(forecast.joint_top(4)) == 4
    assert R.process_counters() == before


def test_a_failed_load_is_said_once_and_still_never_raises(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    missing = tmp_path / "none.pt"
    with caplog.at_level(logging.WARNING, logger=R.__name__):
        made = R.OpponentPredictor.load(missing)
    assert not made.loaded and made.counters["load_error:FileNotFoundError"] == 1
    said = [record for record in caplog.records if record.name == R.__name__]
    assert len(said) == 1 and said[0].levelno == logging.WARNING
    assert "NOT loaded" in said[0].getMessage() and "none.pt" in said[0].getMessage()
    with pytest.raises(FileNotFoundError):
        R.OpponentPredictor.load(missing, strict=True)


# --- the fit script ---------------------------------------------------------------


class World:
    dataset: Path
    artifact: Path
    old_rows: int
    sealed_rows: int


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory, fz: F.Featurizer) -> World:
    """A dataset on disk whose two slots choose together, and a count table
    fitted on its training rows. Its ladder holdout holds games of before the
    sealed day and games from it on."""
    root = tmp_path_factory.mktemp("coupling_fit")
    data = root / "data"
    data.mkdir()
    cut = F.local_time(F.OWN_SEALED_FROM)
    made = World()
    made.old_rows, made.sealed_rows = 48, 32
    train = with_meta(paired_rows(fz, 160, 10), "train", 0, cut - 900_000)
    val = with_meta(paired_rows(fz, 640, 11, one_target=80), "val", 100, cut - 800_000)
    old = with_meta(paired_rows(fz, made.old_rows, 12), "ladder_holdout", 300, cut - 9)
    new = with_meta(
        paired_rows(fz, made.sealed_rows, 13), "ladder_holdout", 400, cut + 60
    )
    F.save_batch(data / "shard-00000.npz", F.concat_batches([train, val, old, new]))
    # no "created": the build day says nothing, the rows' times do
    manifest = {
        "tag": "unit",
        "splits": SPLITS,
        "shards": [{"file": "shard-00000.npz", "examples": 880}],
    }
    (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    fz.save(data)
    made.dataset = data
    table = T.FlagsTable.fit(train, featurizer=fz)
    made.artifact = write_table(
        root / "source.pt",
        fz,
        table,
        extra={"manifest_sha256": FC.sha256_file(data / "manifest.json")},
    )
    return made


def fit(world: World, out: Path, *more: str) -> tuple[int, dict[str, Any], str]:
    argv = ["--artifact", str(world.artifact), "--dataset", str(world.dataset)]
    argv += ["--out", str(out), "--resamples", "40", *more]
    code = FC.main(argv)
    path = out / FC.REPORT_JSON
    report = json.loads(path.read_text()) if path.is_file() else {}
    text = (out / FC.REPORT_MD).read_text() if report else ""
    return code, report, text


def test_the_status_never_contradicts_the_verdict(
    world: World, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """``--no-artifact`` with a verdict of taken ended ``CPL_NOT_TAKEN``."""
    code, report, text = fit(world, tmp_path / "none", "--no-artifact")
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert code == 0 and report["verdict"] == CP.VERDICT_TAKEN
    assert report["status"] == last == FC.TAKEN_NO_ARTIFACT != FC.NOT_TAKEN
    assert report["artifact"] is None
    assert not (tmp_path / "none" / "artifact.pt").exists()
    assert "--no-artifact" in report["artifact_withheld"]
    assert "None written: --no-artifact was given." in text
    assert f"Status: {FC.TAKEN_NO_ARTIFACT}." in text

    code, report, _ = fit(world, tmp_path / "written")
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert code == 0 and report["status"] == last == FC.DONE
    assert report["artifact_withheld"] is None and report["checks"]["ok"]
    stored = A.read_artifact(tmp_path / "written" / "artifact.pt")
    assert stored["version"] == A.VERSION_COUPLED
    record = stored["extra"][A.EXTRA_COUPLING]
    assert record["name"] == report["name"] and record["rule_as_fixed"] is True
    loaded = A.load_predictor(tmp_path / "written" / "artifact.pt")
    assert loaded.coupling.info["rule_as_fixed"] is True
    # a second coupling is never stacked; --refit replaces the record too
    code, _, _ = fit(
        world,
        tmp_path / "stacked",
        "--artifact",
        str(tmp_path / "written" / "artifact.pt"),
    )
    assert code == 1 and "CPL_FAILED" in capsys.readouterr().out


def test_the_report_says_which_rule_it_is_and_where_it_was_fixed(
    world: World, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    code, report, text = fit(world, tmp_path / "fixed", "--no-artifact")
    accept = report["acceptance"]
    assert code == 0 and accept["as_fixed"] is True and accept["departures"] == []
    assert "design_joint.md" in accept["fixed"] and "02:23" in accept["fixed"]
    assert "pre-registered" not in json.dumps(report) + text
    assert "Rule (fixed before any fit in" in text
    design = accept["design_reading"]
    assert design["holds"] is True and design["folds_positive"] == accept["folds"]
    assert design["mean_gain"] == pytest.approx(report["fit"]["mean_gain"])
    assert "The design's own wording" in text
    capsys.readouterr()

    # a looser rule: said, and no file unless asked for
    loose = ("--margin", "-1", "--folds", "3", "--fold-share", "0.1")
    code, report, text = fit(world, tmp_path / "loose", *loose)
    last = capsys.readouterr().out.strip().splitlines()[-1]
    accept = report["acceptance"]
    assert code == 0 and report["verdict"] == CP.VERDICT_TAKEN
    assert accept["as_fixed"] is False and len(accept["departures"]) == 3
    assert last == report["status"] == FC.TAKEN_NO_ARTIFACT
    assert "not the fixed ones" in report["artifact_withheld"]
    assert not (tmp_path / "loose" / "artifact.pt").exists()
    assert "Rule (NOT the fixed one: margin -1.0 (fixed: 0.002)" in text
    code, report, _ = fit(world, tmp_path / "asked", *loose, "--allow-other-rule")
    assert code == 0 and report["status"] == FC.DONE
    stored = A.read_artifact(tmp_path / "asked" / "artifact.pt")
    assert stored["extra"][A.EXTRA_COUPLING]["rule_as_fixed"] is False
    # the grid in another order is the same grid
    argv = ["--artifact", "a", "--dataset", "d", "--out", "o", "--kappa-grid"]
    assert FC.rule_departures(FC.parse_args([*argv, "200", "50", "20", "5"])) == []


def test_the_two_counts_of_fully_seen_have_their_own_names(world: World, fz):
    batch = paired_rows(fz, 24, 20)
    other = F.other_index(F.N_CAND_DEFAULT)
    # slot a of row 0 used a move outside the candidates: no target is logged
    batch["y_action"][0, 0] = other
    batch["y_set"][0, 0] = 0
    batch["y_set"][0, 0, other] = 1
    batch["y_target"][0, 0] = -1
    batch = with_meta(batch, "val", 0, 1)
    table = A.load_predictor(world.artifact)
    pred = FC.predict_features(table.predictor, batch, "unit")
    rows = FC.fit_rows(batch, pred, fz.tables.move_intent)
    assert rows["rows"].all() and rows["fully_seen"].all()
    assert rows["counted"].tolist() == [False] + [True] * 23
    assert bool(J.true_replies(batch)["visible"][0]) is False


def test_the_sealed_own_games_are_told_by_the_rows_times(
    world: World, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """The guard of before read the day the dataset was built: a manifest
    without one passed, and the sealed games were read."""
    held = FC.sealed_own_games(world.dataset)
    assert held["late_rows"] == world.sealed_rows
    assert held["sealed_from"] == F.OWN_SEALED_FROM
    args = ("--no-artifact", "--report-splits", "ladder_holdout")
    code, report, text = fit(world, tmp_path / "old_only", *args)
    said = capsys.readouterr().out
    found = report["informational"]["ladder_holdout"]
    assert code == 0 and found["examples"] == world.old_rows
    assert found["sealed_rows_left_out"] == world.sealed_rows
    assert found["own_games"]["read_whole"] is False
    assert f"{world.sealed_rows} rows of own games" in said
    assert f"{world.sealed_rows} rows left out; none is read" in text
    # the whole holdout only when asked, and the report says it was read
    code, report, text = fit(world, tmp_path / "whole", *args, "--allow-fresh-holdout")
    found = report["informational"]["ladder_holdout"]
    assert code == 0 and found["examples"] == world.old_rows + world.sealed_rows
    assert found["sealed_rows_left_out"] == 0 and found["own_games"]["read_whole"]
    assert "THE SEALED OWN GAMES WERE READ" in text
    capsys.readouterr()
    # without the split nothing of the holdout is looked at
    code, report, _ = fit(world, tmp_path / "val_only", "--no-artifact")
    assert code == 0 and report["informational"] == {}
    capsys.readouterr()

    # rows without times cannot be told apart: refused, nothing read
    blind = tmp_path / "blind"
    blind.mkdir()
    batch = F.load_batch(world.dataset / "shard-00000.npz")
    batch.pop("m_time")
    F.save_batch(blind / "shard-00000.npz", batch)
    for name in ("manifest.json", "vocab.json", "tables.npz", "repertoire.json"):
        (blind / name).write_bytes((world.dataset / name).read_bytes())
    argv = ["--artifact", str(world.artifact), "--dataset", str(blind)]
    argv += ["--out", str(tmp_path / "refused"), *args]
    assert FC.main(argv) == 1
    assert "carry no times" in capsys.readouterr().out.strip().splitlines()[-1]
    assert not (tmp_path / "refused").exists()
    assert FC.sealed_own_games(blind)["time_known"] is False
    spelled = FC.parse_args([*argv, "--allow-sealed-holdout"])  # the trainer's name
    assert spelled.allow_fresh_holdout is True


# --- the scorecard ----------------------------------------------------------------


def test_the_scorecard_leaves_the_sealed_own_games_out_of_set_a(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Set (a) of a newer build holds the bot's own games from the sealed day
    on. The card read them like any other row and said nothing."""
    before = make_card(tmp_path / "plain", monkeypatch)  # rows without times
    assert "own_games" not in before["dataset"] and before["warnings"] == []
    assert before["sets"][SC.SET_LADDER]["examples"] == 4

    cut = F.local_time(F.OWN_SEALED_FROM)
    untimed = SC.F.load_dataset  # make_card's stand-in: (batch, manifest)

    def timed(directory: Any, splits: Any = None) -> Any:
        batch, manifest = untimed(directory, splits)
        batch = dict(batch)
        when = np.full(12, cut - 5, dtype=np.int64)
        when[2] = cut  # a ladder row of the sealed day
        when[3] = 0  # a ladder row of unknown time: sealed too
        when[8:] = cut + 5  # test rows are never own games
        batch["m_time"] = when
        return batch, manifest

    monkeypatch.setattr(SC.F, "load_dataset", timed)
    batch, manifest = timed("stand-in")
    marked = SC.sealed_own_rows(batch, manifest["splits"])
    assert marked is not None and np.nonzero(marked)[0].tolist() == [2, 3]
    assert SC.sealed_own_rows(untimed("stand-in")[0], manifest["splits"]) is None

    def card(out: Path, *more: str) -> tuple[dict[str, Any], list[str]]:
        argv = ["--artifact", "blind", "--artifact", "keeps", "--reference", "blind"]
        argv += ["--elo-blind", "blind", "--out", str(out), "--resamples", "100"]
        said: list[str] = []
        made = SC.run(SC.parse_args([*argv, "--shuffles", "2", *more]), log=said.append)
        return made, said

    kept, said = card(tmp_path / "kept")
    assert kept["sets"][SC.SET_LADDER]["examples"] == 2  # the two old rows
    assert kept["sets"][SC.SET_TEST]["examples"] == 8
    assert kept["dataset"]["own_games"] == {
        "sealed_from": F.OWN_SEALED_FROM,
        "sealed_rows_in_dataset": 2,
        "sealed_rows_read": False,
    }
    assert len(kept["warnings"]) == 1 and "left out of set (a)" in kept["warnings"][0]
    assert any("left out of (a)" in line for line in said)

    whole, said = card(tmp_path / "whole", "--allow-sealed-holdout")
    assert whole["sets"][SC.SET_LADDER]["examples"] == 4
    assert whole["dataset"]["own_games"]["sealed_rows_read"] is True
    assert len(whole["warnings"]) == 1 and "SEALED confirmation" in whole["warnings"][0]
    assert any("READ (--allow-sealed-holdout)" in line for line in said)
