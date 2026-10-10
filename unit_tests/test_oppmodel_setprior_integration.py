"""The set prior wired into layout version 2: featurizer, network, builder.

What must hold: ``sp_cand`` is ``SetTable.posterior`` of each candidate given
what the public knows of the Pokemon, quantised; it costs no example and no
version-1 array; the table rides in the featurizer payload (so in a dataset
folder and in an artifact); a network that reads it starts as the network
that does not; the builder reads sheets of TRAINING sides only and never
shows a training row a table that holds its own account.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from datagen import oppmodel_build_dataset as B
from unit_tests import test_oppmodel_dataset as TD
from unit_tests import test_oppmodel_layout_v2 as TL
from unit_tests import test_oppmodel_model as TM
from unit_tests import test_oppmodel_train as TT
from vgc_bench.src.oppmodel import events as E
from vgc_bench.src.oppmodel import features as F
from vgc_bench.src.oppmodel import model as M
from vgc_bench.src.oppmodel import public_state as P
from vgc_bench.src.oppmodel import setprior as SP

MU_NAMES = TL.MU_NAMES
ALL_NAMES = (*MU_NAMES, "sp_cand")
SP_BIT = 8
EVERYTHING = 15
GARCHOMP_A = ("earthquake", "protect", "dragonclaw", "rockslide")
GARCHOMP_B = ("earthquake", "protect", "stompingtantrum", "swordsdance")
SNEASLER = ("closecombat", "fakeout", "direclaw", "protect")
COL = {name: index for index, name in enumerate(SP.COLUMN_NAMES)}


def table(extra: tuple[SP.SetRecord, ...] = ()) -> SP.SetTable:
    """Sets of the hand-written game's opposing side (``TM.GAME``, p2)."""
    records: list[SP.SetRecord] = []
    records += [("garchomp", GARCHOMP_A, "garchompite", "roughskin", 1.0)] * 6
    records += [("garchomp", GARCHOMP_B, "choicescarf", "roughskin", 1.0)] * 6
    records += [("sneasler", SNEASLER, "focussash", "unburden", 1.0)] * 5
    records += list(extra)
    return SP.SetTable.build(records)


@pytest.fixture(scope="module")
def fz1() -> F.Featurizer:
    return F.Featurizer.build(TM.repertoire())


@pytest.fixture(scope="module")
def fzs() -> F.Featurizer:
    return F.Featurizer.build(TM.repertoire(), layout_version=2, set_table=table())


@pytest.fixture(scope="module")
def batch(fzs: F.Featurizer) -> F.Batch:
    return TM.game_batch(fzs)


def row_of(example: F.Example, fz: F.Featurizer, slot: int, move: str) -> np.ndarray:
    column = list(example["cand_move"][slot]).index(fz._move_id(move))
    return example["sp_cand"][slot, column].astype(int)


# --- the table of extra arrays ------------------------------------------------------


def test_the_set_prior_row_of_the_table():
    spec = F.extra_array("sp_cand")
    assert spec is not None and spec is F.EXTRA_ARRAYS[-1]
    assert (
        spec.bit
        == SP_BIT
        == next(bit for bit, name in F.EXTRA_RESERVED_BITS.items() if name == "sp_cand")
    )
    assert spec.kind == F.EXTRA_CAND and spec.foe_axis is None and spec.per_actor_slot
    assert spec.columns == SP.COLUMN_NAMES and spec.width == SP.SET_COLS == 4
    assert spec.scales == (1.0 / F.SP_QUANT,) * 4 and spec.dtype == "int8"
    assert spec.family == F.FAMILY_SETPRIOR != F.FAMILY_MATCHUP
    assert spec.shape(12) == (2, 12, 4)
    assert F.extra_names() == ALL_NAMES
    assert sum(row.bit for row in F.EXTRA_ARRAYS) == EVERYTHING
    assert F.LAYOUT_VERSION == 1  # the default layout never moved


# --- the featurizer -----------------------------------------------------------------


def test_only_a_featurizer_with_a_table_writes_the_array(fzs: F.Featurizer):
    # "All of them" without a table is the matchup arrays alone.
    bare = F.Featurizer.build(TM.repertoire(), layout_version=2)
    assert bare.extras == MU_NAMES and bare.set_table is None
    assert bare.set_prior_signature is None and "set_prior" not in bare.to_payload()
    assert "sp_cand" not in bare.layout()
    with pytest.raises(ValueError):
        F.Featurizer.build(TM.repertoire(), layout_version=2, extras=("sp_cand",))
    with pytest.raises(ValueError):
        F.Featurizer.build(TM.repertoire(), set_table=table())  # layout version 1
    assert fzs.extras == ALL_NAMES and fzs.set_table is not None
    assert fzs.layout()["sp_cand"].shape == (2, 12, 4)
    assert fzs.layout()["sp_cand"].dtype == "int8"
    assert fzs.layout()["sp_cand"].group == "feature"
    assert list(fzs.layout())[-4:] == list(ALL_NAMES)  # after every other array
    # The set prior alone, without the matchup arrays.
    only = F.Featurizer.build(
        TM.repertoire(), layout_version=2, extras=("sp_cand",), set_table=table()
    )
    assert only.extras == ("sp_cand",) and only.matchup_version == 0
    example = only.encode(TM.drive().turns[0].snapshot, "p2")
    assert example is not None and "mu_slot" not in example
    assert example["sp_cand"].any()


def test_every_other_array_is_what_it_was(
    fz1: F.Featurizer, fzs: F.Featurizer, batch: F.Batch
):
    old = TM.game_batch(fz1)
    mu = TM.game_batch(F.Featurizer.build(TM.repertoire(), layout_version=2))
    assert set(batch) - set(old) == set(ALL_NAMES)
    for name, array in old.items():
        assert batch[name].dtype == array.dtype, name
        assert batch[name].tobytes() == array.tobytes(), name
    for name in MU_NAMES:  # the matchup arrays do not depend on the table
        assert batch[name].tobytes() == mu[name].tobytes(), name
    assert batch["sp_cand"].dtype == np.int8 and batch["sp_cand"].shape[1:] == (
        2,
        12,
        4,
    )
    assert batch["sp_cand"].min() >= 0 and batch["sp_cand"].max() <= F.SP_QUANT
    assert not fzs.counters and not fz1.counters


def test_what_the_array_reads(fzs: F.Featurizer):
    turns = TM.drive().turns
    held = fzs.set_table
    assert held is not None
    # Turn 1, p2: Garchomp (slot a) and Sneasler (slot b), nothing shown.
    first = fzs.encode(turns[0].snapshot, "p2")
    assert first is not None
    assert list(row_of(first, fzs, 0, "earthquake")) == [100, 100, 100, 100]
    # In half of the twelve sets: (6 + 2 * 0.5) / (12 + 2) = 0.5.
    assert list(row_of(first, fzs, 0, "dragonclaw")) == [100, 50, 50, 100]
    assert list(row_of(first, fzs, 1, "closecombat")) == [100, 100, 100, 100]
    # Candidates the table never saw on the species: known to be absent.
    flags = first["cand_flag"].astype(int)
    valid = (flags & F.CAND_VALID) > 0
    filler = valid[0] & ((flags[0] & F.CAND_GLOBAL) > 0)
    assert filler.any()
    assert (first["sp_cand"][0][filler][:, COL["p_member"]] == 0).all()
    assert (first["sp_cand"][0][filler][:, COL["has_data"]] == 100).all()
    # OTHER has no row: one per candidate column and no more.
    assert first["sp_cand"].shape[1] == fzs.n_cand
    # Turn 2: Garchomp is a Mega (its stone is public) and has shown one move:
    # the sets with that stone remain, so the other set's moves fall away.
    second = fzs.encode(turns[1].snapshot, "p2")
    assert second is not None
    mon = next(m for m in turns[1].snapshot.sides["p2"].mons if m.slot == "a")
    assert mon.is_mega and mon.item == "garchompite"
    assert [move.id for move in mon.moves] == ["earthquake"]
    assert list(row_of(second, fzs, 0, "earthquake")) == [100, 100, 100, 75]
    # (6 + 2 * 0.5) / (6 + 2) = 0.875 and (0 + 2 * 0.5) / 8 = 0.125.
    assert row_of(second, fzs, 0, "dragonclaw")[COL["p_member"]] in (87, 88)
    assert row_of(second, fzs, 0, "swordsdance")[COL["p_member"]] in (12, 13)
    # Every row is the table's own answer, quantised.
    for example, record in ((first, turns[0]), (second, turns[1])):
        side = record.snapshot.sides["p2"]
        flags = example["cand_flag"].astype(int)
        for slot in range(F.N_SLOT):
            mon = side.mons[int(example["act_mon"][slot])]
            names = [
                fzs.tables.move_ids[int(row)]
                for row, flag in zip(example["cand_move"][slot], flags[slot])
                if flag & F.CAND_VALID
            ]
            want = held.posterior(
                F.set_key(mon.forme, mon.species),
                [move.id for move in mon.moves],
                names,
                item=mon.item or "",
                item_known=bool(mon.item) and mon.item_state != P.ITEM_UNKNOWN,
                ability=mon.ability or "",
                ability_known=bool(mon.ability) and not mon.is_mega,
            )
            got = example["sp_cand"][slot, : len(names)]
            assert np.array_equal(got, np.rint(want * F.SP_QUANT).astype(np.int8))
    # A species the table does not hold: zeros, and no count (not a failure).
    mine = fzs.encode(turns[0].snapshot, "p1")
    assert mine is not None and not mine["sp_cand"].any()
    assert not fzs.counters and held.errors == 0


def test_an_open_sheet_counts_every_sheet_move_as_shown():
    fz = F.Featurizer.build(TM.repertoire(), layout_version=2, set_table=table())
    opened = P.drive_log(TD.make_log("alice", "bob", sheets=True), "battle-sp-1")
    closed = P.drive_log(TD.make_log("alice", "bob"), "battle-sp-2")
    assert opened.usable and opened.sheets == {"p1": True, "p2": True}
    example = fz.encode(opened.turns[0].snapshot, "p2")
    assert example is not None
    assert int(example["game_flag"][F.G_ACTOR_SHEET]) == F.SHEET_OPEN
    flags = example["cand_flag"][0].astype(int)
    sheet = (flags & F.CAND_SHEET) > 0
    assert sheet.sum() == 4 and ((flags & F.CAND_VALID) > 0).sum() == 4
    rows = example["sp_cand"][0][sheet].astype(int)
    assert (rows[:, COL["has_data"]] == 100).all()
    assert (rows[:, COL["p_member"]] == 100).all()
    assert (rows[:, COL["slots_left"]] == 0).all()
    # Padding candidates (eight of the twelve columns here) are zeros.
    assert not example["sp_cand"][0][~sheet].any()
    # The same turn with the sheets closed: nothing shown, four slots left.
    blind = fz.encode(closed.turns[0].snapshot, "p2")
    assert blind is not None
    assert int(blind["game_flag"][F.G_ACTOR_SHEET]) == F.SHEET_CLOSED
    assert row_of(blind, fz, 0, "earthquake")[COL["slots_left"]] == 100
    assert row_of(blind, fz, 0, "swordsdance")[COL["p_member"]] > 0


def test_the_slot_mirror_follows_the_array(fzs: F.Featurizer, batch: F.Batch):
    total = 0
    for record in TM.drive().turns:
        compared, bad = TM.mirror_mismatches(fzs, record)
        assert not bad, bad[:6]
        total += compared
    assert total >= 30
    actor = M.swap_slots(batch, True, False)
    assert np.array_equal(actor["sp_cand"], batch["sp_cand"][:, ::-1])
    assert not np.array_equal(actor["sp_cand"], batch["sp_cand"])
    assert np.array_equal(M.swap_slots(batch, False, True)["sp_cand"], batch["sp_cand"])
    assert actor["sp_cand"].dtype == np.int8
    assert "sp_cand" in M.strip_labels(batch)


def test_the_set_prior_never_costs_an_example(fz1: F.Featurizer):
    snapshot = TM.drive().turns[1].snapshot
    plain = fz1.encode(snapshot, "p2")
    assert plain is not None

    class Broken(SP.SetTable):
        def posterior(self, *args: Any, **kwargs: Any) -> np.ndarray:
            raise RuntimeError("no")

    class Short(SP.SetTable):
        def posterior(self, *args: Any, **kwargs: Any) -> np.ndarray:
            return np.ones((1, 9), dtype=np.float32)

    for kind, counter in (
        (Broken, "setprior_error:RuntimeError"),
        (Short, "setprior_bad_shape"),
    ):
        fz = F.Featurizer.build(
            TM.repertoire(), layout_version=2, set_table=kind(table()._species)
        )
        example = fz.encode(snapshot, "p2")
        assert example is not None and not example["sp_cand"].any()
        assert fz.counters[counter] >= 1
        assert example["mu_slot"].any()  # the other family is untouched
        for name, array in plain.items():
            assert np.array_equal(example[name], array), name
    # An answer the table itself had to zero is counted once per example.
    held = table()
    fz = F.Featurizer.build(TM.repertoire(), layout_version=2, set_table=held)
    held._species["garchomp"].move_ix = None  # type: ignore[assignment]
    example = fz.encode(snapshot, "p2")
    assert example is not None and held.errors >= 1
    assert fz.counters["setprior_posterior_error"] == 1
    assert not example["sp_cand"][0].any() and example["sp_cand"][1].any()
    # A featurizer that lost its table: zeros and a count, never an exception.
    fz = F.Featurizer.build(TM.repertoire(), layout_version=2, set_table=table())
    fz.set_table = None
    example = fz.encode(snapshot, "p2")
    assert example is not None and not example["sp_cand"].any()
    assert fz.counters["setprior_no_table"] == 1


def test_the_override_is_for_one_encode_and_never_stored(fzs: F.Featurizer):
    snapshot = TM.drive().turns[0].snapshot
    fz = F.Featurizer.build(TM.repertoire(), layout_version=2, set_table=table())
    want = fz.encode(snapshot, "p2")
    stored = fz.to_payload()
    fz.set_table_override = SP.SetTable.build([])
    other = fz.encode(snapshot, "p2")
    assert want is not None and other is not None
    assert want["sp_cand"].any() and not other["sp_cand"].any()
    assert fz.to_payload()["set_prior"] == stored["set_prior"]
    assert fz.set_prior_signature == fzs.set_prior_signature
    fz.set_table_override = None
    again = fz.encode(snapshot, "p2")
    assert again is not None and np.array_equal(again["sp_cand"], want["sp_cand"])


# --- storage ------------------------------------------------------------------------


def test_the_table_rides_in_the_payload_and_the_dataset_folder(
    fz1: F.Featurizer, fzs: F.Featurizer, batch: F.Batch, tmp_path: Path
):
    payload = fzs.to_payload()
    held = fzs.set_table
    assert held is not None
    assert payload["extras"] == list(ALL_NAMES)
    assert payload["set_prior"] == held.to_payload()
    assert payload["set_prior_signature"] == held.signature()
    assert json.loads(json.dumps(payload["set_prior"])) == payload["set_prior"]
    again = F.Featurizer.from_payload(payload, strict=True)
    assert again.extras == ALL_NAMES and again.signature_diff == []
    assert again.set_prior_signature == fzs.set_prior_signature
    assert again.set_table is not held
    rebuilt = TM.game_batch(again)
    for name, array in batch.items():
        assert rebuilt[name].tobytes() == array.tobytes(), name
    # A payload that names the array and holds no (or another) table is refused.
    for broken in (
        {key: value for key, value in payload.items() if key != "set_prior"},
        {**payload, "set_prior": {"format": "something else"}},
        {**payload, "set_prior_signature": "0" * 16},
    ):
        with pytest.raises(ValueError):
            F.Featurizer.from_payload(broken)
    # The dataset folder: one more file, and its hash in vocab.json.
    fzs.save(tmp_path / "v2")
    stored = json.loads((tmp_path / "v2" / "vocab.json").read_text())
    assert stored["set_prior_signature"] == held.signature()
    assert "set_prior" not in stored
    on_disk = json.loads((tmp_path / "v2" / F.SET_PRIOR_FILE).read_text())
    assert SP.SetTable.from_payload(on_disk).signature() == held.signature()
    loaded = F.Featurizer.load(tmp_path / "v2", strict=True)
    assert loaded.extras == ALL_NAMES
    assert loaded.set_prior_signature == held.signature()
    (tmp_path / "v2" / F.SET_PRIOR_FILE).unlink()
    with pytest.raises(OSError):
        F.Featurizer.load(tmp_path / "v2")
    # Version 1 and a version-2 featurizer without a table write no such thing.
    assert set(fz1.to_payload()) == set(TL.golden("meta.json")["payload_keys"])
    for name, fz in (
        ("v1", fz1),
        ("mu", F.Featurizer.build(TM.repertoire(), layout_version=2)),
    ):
        fz.save(tmp_path / name)
        assert not (tmp_path / name / F.SET_PRIOR_FILE).exists()
        text = (tmp_path / name / "vocab.json").read_text()
        assert "set_prior" not in text
        assert F.Featurizer.load(tmp_path / name).set_table is None


# --- the network --------------------------------------------------------------------


def test_bit_8_makes_the_network_read_the_set_prior(
    fz1: F.Featurizer, fzs: F.Featurizer, batch: F.Batch
):
    d = TM.SMALL["d_model"]
    assert M.extras_overrides(fzs, SP_BIT) == {"sp_cand": 4}
    assert M.extras_overrides(fzs, EVERYTHING) == {
        "mu_cand": 8,
        "mu_slot": 10,
        "mu_roster": 4,
        "sp_cand": 4,
        "matchup_version": 1,
    }
    bare = F.Featurizer.build(TM.repertoire(), layout_version=2)
    for fz, mask in ((bare, SP_BIT), (bare, EVERYTHING), (fz1, SP_BIT), (fzs, 16)):
        with pytest.raises(ValueError):
            M.extras_overrides(fz, mask)
    plain = TM.jolt(TM.small_net(fzs))
    only = TM.jolt(TL.matchup_net(fzs, SP_BIT))
    full = TM.jolt(TL.matchup_net(fzs, EVERYTHING))
    assert only.extra_keys == ("sp_cand",) and full.extra_keys == ALL_NAMES
    assert set(only.state_dict()) - set(plain.state_dict()) == {
        "sp_cand_act",
        "sp_cand_scale",
    }
    assert only.n_parameters() - plain.n_parameters() == d * 4
    assert full.n_parameters() - plain.n_parameters() == d * (52 + 4)
    assert not only.state_dict()["sp_cand_act"].any()  # created as zeros
    # Fresh weights: every extra on is exactly the network with none.
    assert batch["sp_cand"].any()
    want = TL.logits(plain, batch)
    for net in (only, full):
        got = TL.logits(net, batch)
        for name in want:
            assert torch.equal(want[name], got[name]), name
        one = TL.logits(net, F.take(batch, slice(2, 3)))  # how the runtime calls it
        base = TL.logits(plain, F.take(batch, slice(2, 3)))
        assert all(torch.equal(base[name], one[name]) for name in base)
    # At the default size too: the same start, 512 more numbers for the block.
    torch.manual_seed(0)
    big_plain = M.OppNet.for_featurizer(fzs)
    torch.manual_seed(0)
    big_full = M.OppNet.for_featurizer(fzs, **M.extras_overrides(fzs, EVERYTHING))
    assert big_full.n_parameters() - big_plain.n_parameters() == 6656 + 512
    want_big, got_big = TL.logits(big_plain, batch), TL.logits(big_full, batch)
    assert all(torch.equal(want_big[name], got_big[name]) for name in want_big)
    # Once the matrix is not zero the candidate scores move, and nothing else.
    TL.fill_extras(only)
    moved = TL.logits(only, batch)
    n_cand = fzs.n_cand
    assert not torch.equal(
        moved["action"][:, :, :n_cand], want["action"][:, :, :n_cand]
    )
    assert torch.equal(moved["action"][:, :, n_cand:], want["action"][:, :, n_cand:])
    assert torch.equal(moved["target"], want["target"])
    assert torch.equal(moved["mega"], want["mega"])
    # The network reads the stored integers through the table's scale, and
    # the scale is saved with the weights (see test_oppmodel_review_fixes).
    scale = only.state_dict().get("sp_cand_scale")
    assert scale is not None
    assert torch.equal(scale, torch.full((4,), 1.0 / F.SP_QUANT))
    assert only.extra_scales() == {"sp_cand": [float(x) for x in scale.tolist()]}
    assert M.OppNetConfig.for_featurizer(fzs, sp_cand=4).extra_widths() == {
        "sp_cand": 4
    }
    with pytest.raises(ValueError):
        M.OppNetConfig.for_featurizer(fzs, sp_cand=3).check()


def test_an_artifact_carries_the_table(
    tmp_path: Path, fzs: F.Featurizer, batch: F.Batch
):
    artifact = importlib.import_module(TM.ARTIFACT_MODULE)
    predictor = M.OppNetPredictor(
        TL.fill_extras(TM.jolt(TL.matchup_net(fzs, EVERYTHING))), fzs, name="sp"
    )
    want = predictor.predict(batch)
    target = tmp_path / "artifact.pt"
    artifact.save_artifact(
        target,
        kind=M.KIND,
        name="sp",
        featurizer=fzs,
        predictor_payload=predictor.to_payload(),
    )
    loaded = artifact.load_predictor(target)
    assert loaded.featurizer.extras == ALL_NAMES
    assert loaded.featurizer.set_prior_signature == fzs.set_prior_signature
    assert loaded.featurizer.signature_diff == []
    assert loaded.predictor.net.extra_keys == ALL_NAMES
    got = loaded.predictor.predict(batch)
    assert all(np.array_equal(want[name], got[name]) for name in want)
    # The loaded featurizer encodes what the stored network was fed.
    rebuilt = TM.game_batch(loaded.featurizer)
    assert rebuilt["sp_cand"].tobytes() == batch["sp_cand"].tobytes()
    # A network that reads the array cannot be paired with a featurizer
    # that does not write it.
    bare = F.Featurizer.build(TM.repertoire(), layout_version=2)
    with pytest.raises(ValueError):
        M.from_payload(predictor.to_payload(), bare)
    # The runtime serves it, with nothing counted as a failure.
    runtime_module = importlib.import_module("vgc_bench.src.oppmodel.runtime")
    runtime = runtime_module.OpponentPredictor.load(target, keep_features=True)
    assert runtime.serving, runtime.load_failure
    log = TM.HEADER + TM.GAME
    fake = SimpleNamespace(_replay_data=[], player_role="p1", battle_tag="battle-sp-9")
    served = 0
    offline = P.drive_log(log, "battle-sp-9", sheets_known=False)
    by_turn = {record.turn: record for record in offline.turns}
    for event in E.split_log(log):
        fake._replay_data.append(list(event))
        if len(event) > 2 and event[1] == "turn":
            forecast = runtime.predict(fake)
            assert forecast is not None and forecast.features is not None
            built = loaded.featurizer.encode(by_turn[int(event[2])].snapshot, "p2")
            assert built is not None
            for name in ALL_NAMES:
                assert np.array_equal(forecast.features[name][0], built[name]), name
            served += 1
    assert served == 7
    counted = runtime.diagnostics()
    assert not [
        name
        for group in counted.values()
        for name in group
        if "error" in name or "setprior" in name or "matchup" in name
    ]


def test_the_trainer_takes_bit_8(tmp_path: Path, fzs: F.Featurizer):
    artifact = importlib.import_module(TM.ARTIFACT_MODULE)
    data = tmp_path / "v2"
    TT.write_dataset(data, fzs)
    assert (data / F.SET_PRIOR_FILE).exists()
    assert TT.run(data, tmp_path / "off", "--epochs", "1") == 0
    assert TT.run(data, tmp_path / "sp", "--epochs", "1", "--extras", "8") == 0
    assert TT.run(data, tmp_path / "all", "--epochs", "1", "--extras", "15") == 0
    reports = {
        name: json.loads((tmp_path / name / "train_report.json").read_text())
        for name in ("off", "sp", "all")
    }
    assert all(report["status"] == "TRAIN_DONE" for report in reports.values())
    assert reports["sp"]["config"]["sp_cand"] == 4
    # A field that is off is not written (an old config has the old keys).
    assert "mu_cand" not in reports["sp"]["config"]
    assert not set(reports["off"]["config"]) & {*M.EXTRA_CONFIG_FIELDS}
    assert "matchup_version" not in reports["off"]["config"]
    assert reports["all"]["config"]["sp_cand"] == 4
    assert reports["all"]["config"]["mu_slot"] == 10
    d = 32
    assert reports["sp"]["n_parameters"] - reports["off"]["n_parameters"] == d * 4
    assert reports["all"]["n_parameters"] - reports["off"]["n_parameters"] == d * 56
    # The untrained network on validation is the same number in every arm.
    start = {name: report["history"][0]["val_fine"] for name, report in reports.items()}
    assert start["off"] == start["sp"] == start["all"]
    for report in reports.values():
        assert report["dataset"]["set_prior_signature"] == fzs.set_prior_signature
    loaded = artifact.load_predictor(tmp_path / "all" / "artifact.pt")
    assert loaded.predictor.net.extra_keys == ALL_NAMES
    assert loaded.featurizer.set_prior_signature == fzs.set_prior_signature
    # A version-2 dataset built without a table has no such array to read.
    bare = tmp_path / "mu_only"
    TT.write_dataset(bare, F.Featurizer.build(TM.repertoire(), layout_version=2))
    assert TT.run(bare, tmp_path / "bad", "--epochs", "1", "--extras", "8") != 0
    assert not (tmp_path / "bad" / "artifact.pt").exists()


# --- the builder --------------------------------------------------------------------


def test_shown_sheets_and_the_fold_rule():
    log = TD.make_log("alice", "bob", sheets=True)
    rows = B.shown_sheets(log, ("p1", "p2"))
    assert len(rows) == 4 and {row[0] for row in rows} == {"p1", "p2"}
    garchomp = next(row for row in rows if row[1] == "garchomp")
    assert garchomp == (
        "p2",
        "garchomp",
        ("dragonclaw", "earthquake", "protect", "rockslide"),
        "garchompite",
        "roughskin",
    )
    assert [row[0] for row in B.shown_sheets(log, ("p2",))] == ["p2", "p2"]
    assert B.shown_sheets(TD.make_log("alice", "bob"), ("p1", "p2")) == []
    assert B.shown_sheets("|showteam|p1\n|showteam|", ("p1",)) == []
    folds = [B.set_fold(f"account{number}") for number in range(200)]
    assert set(folds) == set(range(B.SET_FOLDS))
    assert B.set_fold("account7") == B.set_fold("account7")


class SheetCorpus:
    """Five training accounts and one validation account, sheets open.

    Game i: training account i (p1) against the validation account (p2);
    one more game with the validation account as p1. Every training p1
    shows the same two sets, so the full table holds each exactly five times
    (the least it answers for) and no cross-fitting table holds them.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        taken: set[str] = set()
        self.train = [TD.account(B.SPLIT_TRAIN, taken) for _ in range(5)]
        self.val = TD.account(B.SPLIT_VAL, taken)
        web = root / B.WEB_DIR / "top"
        web.mkdir(parents=True)
        for number, name in enumerate(self.train):
            log = TD.make_log(name, self.val, sheets=True, when=1000 + number)
            (web / f"{TD.replay_id(40 + number)}.log").write_text(log)
        log = TD.make_log(self.val, self.train[0], sheets=True, when=2000)
        (web / f"{TD.replay_id(50)}.log").write_text(log)

    def build(self, tag: str, **options: Any) -> dict[str, Any]:
        return B.build(tag, root=self.root, log=lambda message: None, **options)


def test_the_builder_reads_training_sheets_and_cross_fits(tmp_path: Path):
    corpus = SheetCorpus(tmp_path)
    old = corpus.build("v1")
    new = corpus.build("v2", layout_version=2)
    root = tmp_path / "results_oppmodel"
    assert "set_prior" not in old and not (root / "v1" / F.SET_PRIOR_FILE).exists()
    assert not (root / "v1" / B.SET_FOLDS_FILE).exists()
    assert new["extras"] == list(ALL_NAMES)
    summary = new["set_prior"]
    # Five p1 sheets of two Pokemon, and the p2 sheet of training account 0.
    assert summary["train_sides_with_a_shown_sheet"] == 6
    assert summary["contributing_accounts"] == 5
    assert summary["records"] == 5 * 2 + 2
    assert summary["collector"] == {
        "sheet_rows_of_other_splits": 6 * 2,
        "sheet_rows_train": 6 * 2,
    }
    assert summary["table"]["species"] == 2  # the p2 sets were seen once: left out
    assert summary["folds"] == B.SET_FOLDS == len(summary["fold_signatures"])
    featurizer = F.Featurizer.load(root / "v2", strict=True)
    assert featurizer.set_prior_signature == summary["signature"]
    folds = json.loads((root / "v2" / B.SET_FOLDS_FILE).read_text())
    assert [SP.SetTable.from_payload(item).signature() for item in folds] == summary[
        "fold_signatures"
    ]
    # No fold table still answers for a set it holds fewer than five times.
    used = {B.set_fold(E.user_id(name)) for name in corpus.train}
    for fold in used:
        assert len(SP.SetTable.from_payload(folds[fold])) == 0
    before, _ = F.load_dataset(root / "v1")
    after, manifest = F.load_dataset(root / "v2")
    assert manifest["set_prior"]["signature"] == summary["signature"]
    for name, array in before.items():
        assert after[name].tobytes() == array.tobytes(), name
    for name in ("repertoire.json", "tables.npz", "battles.jsonl"):
        assert (root / "v1" / name).read_bytes() == (root / "v2" / name).read_bytes()
    assert "sets" not in (root / "v2" / "battles.jsonl").read_text()
    # A training side never sees a table that holds its own account ...
    sp = after["sp_cand"]
    train = after["m_split"] == B.SPLIT_TRAIN
    val = after["m_split"] == B.SPLIT_VAL
    first_side = after["m_side"] == 0
    assert (train & first_side).sum() > 0 and not sp[train & first_side].any()
    # ... while the validation player's same two sets get the full table.
    rows = sp[val & first_side]
    assert rows.shape[0] > 0 and (rows[:, :, :4, COL["p_member"]] == 100).all()
    assert not sp[val & ~first_side].any()  # sets the table does not hold
    # The run left no override behind and counted no failure.
    assert not [name for name in new["featurizer"]["counters"] if "setprior" in name]
    assert not any("error" in name for name in summary["posterior_counters"])


def test_encode_battle_picks_the_table_by_the_actors_account(tmp_path: Path):
    corpus = SheetCorpus(tmp_path)
    reader = B.Corpus(tmp_path, B.DEFAULT_FORMATS, B.SOURCES)
    plain = B.run_pass1(reader, set(), set())
    assert all(not meta.sets for meta in plain.metas)  # a version-1 pass reads none
    first = B.run_pass1(reader, set(), set(), collect_sets=True)
    assert all(len(meta.sets) == 4 for meta in first.metas)
    info = B.assign_splits(first.metas)
    full, folds, summary = B.build_set_tables(first.metas)
    assert len(full) == 2 and summary["records"] == 12
    fz = F.Featurizer.build(TM.repertoire(), layout_version=2, set_table=full)
    empty = [SP.SetTable.build([]) for _ in range(B.SET_FOLDS)]
    last = next(
        meta for meta in first.metas if meta.accounts["p1"] == E.user_id(corpus.val)
    )
    raw = next(item for item in reader if item.battle_id == last.battle_id)
    result = B.drive(raw)
    crossed = B.encode_battle(fz, last, result, info, empty)
    whole = B.encode_battle(fz, last, result, info)
    assert fz.set_table_override is None
    sides = np.array([int(example["m_side"]) for example in crossed])
    assert set(sides) == {0, 1}
    for one, two in zip(crossed, whole):
        if int(one["m_side"]) == 0:  # the validation account: the full table
            assert one["sp_cand"].any()
            assert np.array_equal(one["sp_cand"], two["sp_cand"])
        else:  # a training account: its fold's table (here an empty one)
            assert not one["sp_cand"].any()
    # One of the bot's own games is always read with the full table.
    last.own = True
    owned = B.encode_battle(fz, last, result, info, empty)
    assert all(
        np.array_equal(one["sp_cand"], two["sp_cand"]) for one, two in zip(owned, whole)
    )
    assert fz.set_table_override is None
