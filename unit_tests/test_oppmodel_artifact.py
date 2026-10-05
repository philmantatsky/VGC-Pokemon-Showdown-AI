"""Opponent predictor artifact file: what is stored, how it is read back."""

from __future__ import annotations

import subprocess
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from vgc_bench.src.oppmodel import artifact as A
from vgc_bench.src.oppmodel import events as E
from vgc_bench.src.oppmodel import features as F
from vgc_bench.src.oppmodel import tables as T

ROOT = Path(__file__).resolve().parents[1]
C = F.N_CAND_DEFAULT


@pytest.fixture(scope="module")
def fz() -> F.Featurizer:
    repertoire = F.Repertoire()
    repertoire.add("alpha", "one", 3.0)
    repertoire.add("alpha", "two", 1.0)
    return F.Featurizer.build(repertoire)


def synthetic_batch(fz: F.Featurizer, n: int = 24, seed: int = 0) -> F.Batch:
    """Random but legal examples in the featurizer's own layout, with labels."""
    rng = np.random.default_rng(seed)
    batch = {
        name: np.zeros((n, *spec.shape), dtype=spec.dtype)
        for name, spec in fz.layout().items()
    }
    for name in ("act_mon", "foe_mon", "y_kind", "y_action", "y_target", "y_mega"):
        batch[name][:] = -1
    batch["y_intent"][:] = -1
    other = F.other_index(C)
    for i in range(n):
        batch["turn"][i] = rng.integers(1, 5)
        batch["act_mon"][i, 0] = 0
        batch["mon_id"][i, 0, F.ID_KEY] = rng.integers(5, 9)
        batch["mon_flag"][i, 0, F.FLAG_FIRST_TURN] = batch["turn"][i] == 1
        batch["elo"][i, 0], batch["elo_known"][i, 0] = rng.integers(1000, 1600), 1
        for column in range(3):
            batch["cand_move"][i, 0, column] = 20 + column
            batch["cand_flag"][i, 0, column] = (
                F.CAND_VALID | F.CAND_DAMAGING | F.CAND_AIMED | F.CAND_REPERTOIRE
            )
            batch["cand_tmask"][i, 0, column] = 0b00111
        batch["cand_tmask"][i, 0, other] = 0b11111
        batch["action_mask"][i, 0, [0, 1, 2, other, F.switch_index(2, C)]] = 1
        batch["switch_mask"][i, 0, 2] = 1
        action = int(rng.choice([0, 1, 2, F.switch_index(2, C)]))
        batch["y_action"][i, 0] = action
        batch["y_set"][i, 0, action] = 1
        if action < 3:
            batch["y_target"][i, 0] = rng.integers(0, 2)
    batch["m_weight"] = np.ones(n, dtype=np.float32)
    return batch


def features_only(batch: F.Batch) -> F.Batch:
    return {k: v for k, v in batch.items() if not k.startswith(("y_", "m_"))}


@pytest.fixture(scope="module")
def table(fz: F.Featurizer) -> T.FlagsTable:
    return T.FlagsTable.fit(synthetic_batch(fz), featurizer=fz)


def save(path: Path, fz: F.Featurizer, table: Any, **extra: Any) -> Path:
    A.save_artifact(
        path,
        kind=A.KIND_TABLE,
        name=table.name,
        featurizer=fz,
        predictor_payload=table.to_payload(),
        extra=extra or None,
    )
    return path


# --- the stored dict ----------------------------------------------------------


def test_artifact_holds_the_documented_keys(tmp_path: Path, fz, table):
    path = save(tmp_path / "deep" / "flags.pt", fz, table, dataset_tag="unit")
    assert [p.name for p in path.parent.iterdir()] == ["flags.pt"]  # no scratch file
    document = A.read_artifact(path)
    assert document["format"] == "oppmodel-artifact" == A.FORMAT
    assert document["version"] == 1 == A.VERSION
    assert document["kind"] == "table" and document["name"] == "flags_table"
    assert document["dex_signature"] == E.dex_signature()
    assert document["dex_signature_diff"] == []
    assert document["extra"] == {"dataset_tag": "unit"}
    assert set(document) == {
        "format",
        "version",
        "kind",
        "name",
        "featurizer",
        "dex_signature",
        "predictor",
        "extra",
        "dex_signature_diff",
    }
    stored = fz.to_payload()
    assert document["featurizer"]["vocab"] == stored["vocab"]
    for name, array in stored["tables"].items():
        again = document["featurizer"]["tables"][name]
        assert again.dtype == array.dtype and np.array_equal(again, array), name
    assert document["predictor"]["table"] == T.TABLE_FLAGS
    assert A.read_artifact(str(path))["name"] == "flags_table"  # str paths too


def test_file_loads_with_weights_only(tmp_path: Path, fz, table):
    """Nothing in the file needs arbitrary unpickling."""
    path = save(tmp_path / "a.pt", fz, table)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert raw["format"] == A.FORMAT and "dex_signature_diff" not in raw

    def check(value: Any, where: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                assert isinstance(key, str), where
                check(item, f"{where}.{key}")
        elif isinstance(value, (list, tuple)):
            for item in value:
                check(item, where)
        else:
            assert value is None or isinstance(
                value, (bool, int, float, str, torch.Tensor)
            ), (where, type(value))

    check(raw, "artifact")


def test_plain_data_round_trip_keeps_every_dtype(tmp_path: Path, fz):
    rng = np.random.default_rng(0)
    arrays = {
        "half": rng.random((3, 2)).astype(np.float16),
        "double": rng.random(5),
        "small": rng.integers(0, 60000, 7).astype(np.uint16),
        "signed": rng.integers(-100, 100, (2, 2, 2)).astype(np.int8),
        "long": np.array([2**40, -(2**40)], dtype=np.int64),
        "flags": np.array([True, False, True]),
        "empty": np.zeros((0, 4), dtype=np.float32),
        "scalar": np.array(3.5),
        "strided": np.arange(12, dtype=np.int32).reshape(3, 4)[:, ::2],
    }
    payload = {
        "arrays": arrays,
        "tensor": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "state": {"weight": torch.ones(2, 2, requires_grad=True)},
        "numbers": [1, 2.5, True, None, "text", float("inf")],
        "tuple": (1, (2, 3)),
        "set": {"b", "a"},
        "counter": Counter(seen=2),
        "path": Path("some/where"),
        "numpy_scalars": [np.float32(1.5), np.int64(7), np.bool_(True)],
    }
    path = tmp_path / "plain.pt"
    A.save_artifact(
        path, kind=A.KIND_OPPNET, name="x", featurizer=fz, predictor_payload=payload
    )
    got = A.read_artifact(path)["predictor"]
    for name, array in arrays.items():
        assert isinstance(got["arrays"][name], np.ndarray), name
        assert got["arrays"][name].dtype == array.dtype, name
        assert got["arrays"][name].shape == array.shape, name
        assert np.array_equal(got["arrays"][name], array), name
    got["arrays"]["double"][0] = 9.0  # a writable copy, not a view of the file
    assert isinstance(got["tensor"], torch.Tensor)
    assert torch.equal(got["tensor"], payload["tensor"])
    assert torch.equal(got["state"]["weight"], torch.ones(2, 2))
    assert not got["state"]["weight"].requires_grad
    assert got["numbers"] == [1, 2.5, True, None, "text", float("inf")]
    assert got["tuple"] == [1, [2, 3]]  # tuples come back as lists
    assert got["set"] == ["a", "b"]
    assert got["counter"] == {"seen": 2}
    assert got["path"] == "some/where"
    assert got["numpy_scalars"] == [1.5, 7, True]
    assert [type(v) for v in got["numpy_scalars"]] == [float, int, bool]


@pytest.mark.parametrize(
    "bad",
    [
        {"object": object()},
        {"array": np.array(["a", None], dtype=object)},
        {1: "integer key"},
        {"nested": [{"deep": {2.5: 1}}]},
        {"set": {1, "a"}},
        {"__ndarray__": True},
        {"function": len},
    ],
    ids=[
        "object",
        "object-array",
        "int-key",
        "float-key",
        "mixed-set",
        "tag",
        "callable",
    ],
)
def test_unstorable_payloads_are_refused_before_anything_is_written(
    tmp_path: Path, fz, bad: dict[Any, Any]
):
    path = tmp_path / "bad.pt"
    with pytest.raises(TypeError):
        A.save_artifact(
            path, kind=A.KIND_TABLE, name="x", featurizer=fz, predictor_payload=bad
        )
    with pytest.raises(TypeError):
        A.save_artifact(
            path,
            kind=A.KIND_TABLE,
            name="x",
            featurizer=fz,
            predictor_payload={},
            extra=bad,
        )
    assert list(tmp_path.iterdir()) == []


def test_save_refuses_an_unknown_kind_or_an_empty_name(tmp_path: Path, fz, table):
    for kind, name in (("forest", "x"), (A.KIND_TABLE, "")):
        with pytest.raises(ValueError):
            A.save_artifact(
                tmp_path / "x.pt",
                kind=kind,
                name=name,
                featurizer=fz,
                predictor_payload=table.to_payload(),
            )
    assert list(tmp_path.iterdir()) == []


def test_an_existing_artifact_survives_a_failed_overwrite(tmp_path: Path, fz, table):
    path = save(tmp_path / "keep.pt", fz, table)
    before = path.read_bytes()
    with pytest.raises(TypeError):
        A.save_artifact(
            path,
            kind=A.KIND_TABLE,
            name="x",
            featurizer=fz,
            predictor_payload={"bad": object()},
        )
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["keep.pt"]


def test_a_failed_write_leaves_no_scratch_file(
    tmp_path: Path, fz, table, monkeypatch: pytest.MonkeyPatch
):
    class Failing:
        Tensor = torch.Tensor
        from_numpy = staticmethod(torch.from_numpy)

        @staticmethod
        def save(document: Any, path: Path) -> None:
            Path(path).write_bytes(b"half a file")
            raise OSError("disk full")

    monkeypatch.setattr(A, "_torch", lambda: Failing)
    with pytest.raises(OSError, match="disk full"):
        save(tmp_path / "x.pt", fz, table)
    assert list(tmp_path.iterdir()) == []


class _Version(str):
    """A str subclass, as ``torch.__version__`` is."""


class _Seconds(float):
    pass


class _Count(int):
    pass


def test_subclasses_of_plain_types_are_stored_as_the_plain_type(tmp_path: Path, fz):
    """np.float64 IS a float and np.str_ / torch.__version__ ARE strs: stored
    as they are, the file saves and then cannot be read back."""
    values = {
        "np_float64": np.float64(1.5),
        "np_mean": np.mean(np.array([1.0, 2.0])),
        "np_str": np.str_("label"),
        "torch_version": torch.__version__,
        "version": _Version("2.1"),
        "seconds": _Seconds(0.25),
        "count": _Count(7),
        "np_float32": np.float32(0.5),
        "np_int64": np.int64(3),
        "np_bool": np.bool_(True),
        "nan": float("nan"),
        "inf": float("inf"),
        "nested": [np.float64(2.5), (np.str_("x"), _Version("y"))],
    }
    path = tmp_path / "plain.pt"
    A.save_artifact(
        path,
        kind=A.KIND_TABLE,
        name="x",
        featurizer=fz,
        predictor_payload={"values": values},
        extra=values,
    )
    torch.load(path, weights_only=True)  # no unpickling of numpy / torch objects
    document = A.read_artifact(path)
    for found in (document["extra"], document["predictor"]["values"]):
        assert type(found["np_float64"]) is float and found["np_float64"] == 1.5
        assert type(found["np_mean"]) is float and found["np_mean"] == 1.5
        assert type(found["np_str"]) is str and found["np_str"] == "label"
        assert type(found["torch_version"]) is str
        assert found["torch_version"] == str(torch.__version__)
        assert type(found["version"]) is str and found["version"] == "2.1"
        assert type(found["seconds"]) is float and found["seconds"] == 0.25
        assert type(found["count"]) is int and found["count"] == 7
        assert type(found["np_float32"]) is float and found["np_float32"] == 0.5
        assert type(found["np_int64"]) is int and found["np_int64"] == 3
        assert found["np_bool"] is True
        assert np.isnan(found["nan"]) and found["inf"] == float("inf")
        assert found["nested"] == [2.5, ["x", "y"]]
        assert type(found["nested"][1][1]) is str


def test_a_file_that_does_not_read_back_never_replaces_an_artifact(
    tmp_path: Path, fz, table, monkeypatch: pytest.MonkeyPatch
):
    """``save_artifact`` reads its temporary file back before the rename."""
    path = save(tmp_path / "keep.pt", fz, table)
    before = path.read_bytes()

    class Leaky:
        """torch, except that ``save`` slips an object past the encoder."""

        Tensor = torch.Tensor
        from_numpy = staticmethod(torch.from_numpy)
        load = staticmethod(torch.load)

        @staticmethod
        def save(document: Any, target: Path) -> None:
            torch.save({**document, "extra": {"version": torch.__version__}}, target)

    monkeypatch.setattr(A, "_torch", lambda: Leaky)
    with pytest.raises(TypeError, match="does not read back"):
        save(path, fz, table)
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["keep.pt"]


def test_reading_never_unpickles_an_object(tmp_path: Path, fz, table):
    """A file in the artifact's own format that holds anything but plain data
    is refused: reading is ``weights_only`` whatever the file claims."""
    path = save(tmp_path / "good.pt", fz, table)
    raw = torch.load(path, weights_only=True)
    for label, smuggled in (
        ("version", torch.__version__),
        ("array", np.arange(3)),
        ("slice", slice(1, 2)),
    ):
        other = tmp_path / f"{label}.pt"
        torch.save({**raw, "extra": {"smuggled": smuggled}}, other)
        with pytest.raises(ValueError, match="not a readable artifact"):
            A.read_artifact(other)
        with pytest.raises(ValueError):
            A.load_predictor(other)
        assert A.try_load_predictor(other) is None


# --- loading a predictor ------------------------------------------------------


@pytest.mark.parametrize("cls", [T.SpeciesTable, T.FlagsTable, T.EloTable])
def test_load_predictor_rebuilds_the_table_and_the_featurizer(
    tmp_path: Path, fz, cls: Any
):
    train = synthetic_batch(fz)
    made = cls.fit(train, featurizer=fz)
    path = save(tmp_path / "t.pt", fz, made, metrics={"fine_nll": 1.25})
    loaded = A.load_predictor(path)
    assert isinstance(loaded, A.LoadedPredictor)
    predictor, featurizer, kind, name, meta = loaded  # unpacks in this order
    assert kind == "table" == predictor.kind
    assert name == cls.table == predictor.name
    assert type(predictor) is cls and isinstance(predictor, F.Predictor)
    assert meta["dex_signature_diff"] == [] and meta["elo_mode"] == "keep"
    assert meta["extra"] == {"metrics": {"fine_nll": 1.25}}
    assert meta["version"] == 1 and meta["path"] == str(path)
    assert featurizer.n_cand == fz.n_cand and featurizer.vocab == fz.vocab
    assert featurizer.repertoire.counts == fz.repertoire.counts
    batch = F.sheet_unknown_as_closed(features_only(synthetic_batch(fz, seed=1)))
    first, second = made.predict(batch), predictor.predict(batch)
    for head in ("action", "target", "mega"):
        assert np.array_equal(first[head], second[head]), head
    assert not loaded.predictor.counters


def test_elo_mode_reaches_the_featurizer(tmp_path: Path, fz, table):
    path = save(tmp_path / "t.pt", fz, table)
    assert A.load_predictor(path).featurizer.elo_mode == F.ELO_KEEP
    blind = A.load_predictor(path, elo_mode=F.ELO_BLANK)
    assert blind.featurizer.elo_mode == F.ELO_BLANK
    assert blind.meta["elo_mode"] == F.ELO_BLANK
    with pytest.raises(ValueError):
        A.load_predictor(path, elo_mode=F.ELO_SHUFFLE)


def test_a_different_dex_is_reported_and_only_strict_raises(
    tmp_path: Path, fz, table, monkeypatch: pytest.MonkeyPatch
):
    other = dict(E.dex_signature(), n_moves=1, first_turn_only_source="elsewhere")
    path = tmp_path / "old.pt"
    with monkeypatch.context() as patch:
        patch.setattr(A, "dex_signature", lambda: other)
        save(path, fz, table)
    document = A.read_artifact(path)
    assert document["dex_signature"] == other
    assert document["dex_signature_diff"] == ["first_turn_only_source", "n_moves"]
    loaded = A.load_predictor(path)
    assert loaded.meta["dex_signature_diff"] == ["first_turn_only_source", "n_moves"]
    assert loaded.meta["dex_signature"] == other
    batch = features_only(synthetic_batch(fz, seed=2))
    assert np.array_equal(
        loaded.predictor.predict(batch)["action"], table.predict(batch)["action"]
    )
    with pytest.raises(ValueError, match="dex signature"):
        A.read_artifact(path, strict=True)
    with pytest.raises(ValueError, match="dex signature"):
        A.load_predictor(path, strict=True)
    assert A.try_load_predictor(path) is not None
    assert A.try_load_predictor(path, strict=True) is None


def test_a_featurizer_from_another_dex_is_also_caught_by_strict(tmp_path: Path, table):
    """The featurizer payload carries its own signature (what the data saw)."""
    old = F.Featurizer.build(F.Repertoire())
    old.signature = dict(old.signature, n_species=3)
    path = save(tmp_path / "old.pt", old, table)
    loaded = A.load_predictor(path)
    assert loaded.meta["dex_signature_diff"] == []  # written under today's dex
    assert loaded.featurizer.signature_diff == ["n_species"]
    with pytest.raises(ValueError):
        A.load_predictor(path, strict=True)


def test_kind_dispatch_is_lazy(tmp_path: Path, fz, monkeypatch: pytest.MonkeyPatch):
    """The module of a kind is imported when an artifact of that kind is loaded."""
    path = tmp_path / "net.pt"
    A.save_artifact(
        path,
        kind=A.KIND_OPPNET,
        name="oppnet_test",
        featurizer=fz,
        predictor_payload={"state": {"w": torch.zeros(2)}, "config": {"width": 8}},
    )
    monkeypatch.setitem(A._MODULES, A.KIND_OPPNET, "vgc_bench.src.oppmodel.no_such")
    with pytest.raises(ValueError, match="no_such"):
        A.load_predictor(path)
    seen: dict[str, Any] = {}

    def from_payload(payload: Any, featurizer: Any) -> Any:
        seen["payload"], seen["featurizer"] = payload, featurizer
        return types.SimpleNamespace(name="oppnet_test", kind="oppnet")

    fake = types.ModuleType("fake_oppnet_module")
    fake.from_payload = from_payload  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fake_oppnet_module", fake)
    monkeypatch.setitem(A._MODULES, A.KIND_OPPNET, "fake_oppnet_module")
    loaded = A.load_predictor(path)
    assert loaded.kind == "oppnet" and loaded.predictor.kind == "oppnet"
    assert isinstance(seen["payload"]["state"]["w"], torch.Tensor)  # tensors stay
    assert seen["payload"]["config"] == {"width": 8}
    assert seen["featurizer"] is loaded.featurizer

    def broken(payload: Any, featurizer: Any) -> Any:
        raise RuntimeError("boom")

    fake.from_payload = broken  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="boom"):
        A.load_predictor(path)


def test_things_that_are_not_artifacts(tmp_path: Path, fz, table):
    with pytest.raises(OSError):
        A.read_artifact(tmp_path / "missing.pt")
    with pytest.raises(OSError):
        A.load_predictor(tmp_path)  # a directory
    text = tmp_path / "text.pt"
    text.write_text("not a torch file", encoding="utf-8")
    with pytest.raises(ValueError):
        A.read_artifact(text)
    plain = tmp_path / "list.pt"
    torch.save([1, 2, 3], plain)
    with pytest.raises(ValueError):
        A.read_artifact(plain)
    path = save(tmp_path / "good.pt", fz, table)
    raw = torch.load(path, weights_only=True)
    for label, change in (
        ("version", {"version": 2}),
        ("format", {"format": "something-else"}),
        ("kind", {"kind": "forest"}),
        ("signature", {"dex_signature": None}),
    ):
        other = tmp_path / f"{label}.pt"
        torch.save({**raw, **change}, other)
        with pytest.raises(ValueError):
            A.read_artifact(other)
    for missing in ("featurizer", "predictor", "extra", "name"):
        other = tmp_path / f"no_{missing}.pt"
        torch.save({k: v for k, v in raw.items() if k != missing}, other)
        with pytest.raises(ValueError):
            A.read_artifact(other)
    damaged = dict(raw)
    damaged["predictor"] = dict(raw["predictor"], reveal_ratio={"__ndarray__": True})
    torch.save(damaged, tmp_path / "damaged.pt")
    with pytest.raises(ValueError):
        A.read_artifact(tmp_path / "damaged.pt")
    short = dict(raw["predictor"]["reveal_ratio"], shape=[999, 4])
    damaged["predictor"] = dict(raw["predictor"], reveal_ratio=short)
    torch.save(damaged, tmp_path / "short.pt")
    with pytest.raises(ValueError):
        A.read_artifact(tmp_path / "short.pt")
    # A sound file whose table payload is not a table.
    wrong = dict(raw, predictor={"table": "flags_table"})
    torch.save(wrong, tmp_path / "wrong.pt")
    with pytest.raises(ValueError):
        A.load_predictor(tmp_path / "wrong.pt")


def test_try_load_predictor_never_raises(tmp_path: Path, fz, table):
    A.COUNTERS.clear()
    assert A.try_load_predictor(tmp_path / "missing.pt") is None
    text = tmp_path / "text.pt"
    text.write_text("junk", encoding="utf-8")
    assert A.try_load_predictor(text) is None
    assert A.try_load_predictor(text, elo_mode="sideways") is None
    assert A.COUNTERS["load_error:FileNotFoundError"] == 1
    assert A.COUNTERS["load_error:ValueError"] == 2
    loaded = A.try_load_predictor(save(tmp_path / "good.pt", fz, table))
    assert loaded is not None and loaded.name == table.name
    assert sum(A.COUNTERS.values()) == 3


def test_importing_the_modules_loads_no_torch():
    code = (
        "import sys; import vgc_bench.src.oppmodel.artifact; "
        "import vgc_bench.src.oppmodel.tables; print('torch' in sys.modules)"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert done.stdout.strip() == "False", done.stdout + done.stderr
