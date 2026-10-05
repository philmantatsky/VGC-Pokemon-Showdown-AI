"""OppNet trainer: schedule, augmentation, snapshot, scoring, the full run.

The full-run tests train a tiny network for a few steps on a dataset written
under ``tmp_path`` from the hand-written game of ``test_oppmodel_model``. They
write and re-read the artifact through ``oppmodel.artifact`` and skip while that
module does not exist.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from training import train_oppmodel as T
from unit_tests import test_oppmodel_model as TM
from vgc_bench.src.oppmodel import features as F
from vgc_bench.src.oppmodel import model as M

SPLITS = ["train", "val", "test", "ladder_holdout"]
TINY = ["--d-model", "32", "--layers", "1", "--heads", "2", "--ff", "48"]


@pytest.fixture(scope="module")
def fz() -> F.Featurizer:
    return F.Featurizer.build(TM.repertoire())


@pytest.fixture(scope="module")
def batch(fz: F.Featurizer) -> F.Batch:
    return TM.game_batch(fz)


def write_dataset(directory: Path, fz: F.Featurizer) -> dict[str, int]:
    """A finished dataset folder: one shard, four splits, the featurizer files."""
    directory.mkdir(parents=True, exist_ok=True)
    mine, theirs = TM.game_batch(fz, ("p2",)), TM.game_batch(fz, ("p1",))
    parts = {
        "train": F.concat_batches(
            [
                mine,
                M.swap_slots(mine, True, False),
                M.swap_slots(mine, False, True),
                M.swap_slots(mine, True, True),
            ]
        ),
        "val": F.concat_batches([theirs, M.swap_slots(theirs, True, True)]),
        "test": theirs,
        "ladder_holdout": mine,
    }
    rows, sizes = [], {}
    for code, name in enumerate(SPLITS):
        part = dict(parts[name])
        n = int(part["act_mon"].shape[0])
        sizes[name] = n
        part["m_split"] = np.full(n, code, dtype=np.uint8)
        part["m_weight"] = np.linspace(0.5, 1.0, n).astype(np.float32)
        rows.append(part)
    data = F.concat_batches(rows)
    F.save_batch(directory / "shard-00000.npz", data)
    manifest = {
        "tag": "tiny",
        "splits": SPLITS,
        "shards": [{"file": "shard-00000.npz", "examples": int(data["turn"].shape[0])}],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    fz.save(directory)
    return sizes


@pytest.fixture()
def dataset(tmp_path: Path, fz: F.Featurizer) -> tuple[Path, dict[str, int]]:
    directory = tmp_path / "data"
    return directory, write_dataset(directory, fz)


@pytest.fixture()
def artifact() -> Any:
    if importlib.util.find_spec(TM.ARTIFACT_MODULE) is None:
        pytest.skip("oppmodel.artifact is not written yet")
    return importlib.import_module(TM.ARTIFACT_MODULE)


def run(dataset: Path, out: Path, *extra: str) -> int:
    argv = ["--dataset", str(dataset), "--out", str(out), "--batch", "16"]
    argv += ["--threads", "1", "--warmup", "2", "--seed", "3", *TINY, *extra]
    return T.main(argv)


def lines_of(capsys: pytest.CaptureFixture[str]) -> list[str]:
    return [line for line in capsys.readouterr().out.splitlines() if line.strip()]


# --- schedule -----------------------------------------------------------------


def test_learning_rate_warms_up_then_follows_a_cosine():
    base, total, warmup = 1e-3, 1000, 100
    rates = [T.learning_rate(step, total, warmup, base) for step in range(total)]
    assert rates[0] == pytest.approx(base / warmup)
    assert all(a < b for a, b in zip(rates[:warmup], rates[1:warmup]))
    assert rates[warmup - 1] == pytest.approx(base)
    assert rates[warmup] == pytest.approx(base)
    assert all(a >= b for a, b in zip(rates[warmup:], rates[warmup + 1 :]))
    assert rates[-1] == pytest.approx(0.05 * base, rel=0.01)
    middle = T.learning_rate(warmup + (total - warmup) // 2, total, warmup, base)
    assert middle == pytest.approx(base * (0.05 + 0.95 * 0.5), rel=1e-3)
    assert T.learning_rate(5000, total, warmup, base) == pytest.approx(0.05 * base)
    assert T.learning_rate(0, 10, 0, base) == pytest.approx(base)
    assert all(math.isfinite(T.learning_rate(s, 1, 0, base)) for s in range(3))


def test_weight_decay_is_on_matrices_only(fz: F.Featurizer):
    net = TM.small_net(fz)
    decay, plain = T.parameter_groups(net, 0.01)
    assert decay["weight_decay"] == 0.01 and plain["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decay["params"])
    assert all(p.ndim < 2 for p in plain["params"])
    held = {id(p) for group in (decay, plain) for p in group["params"]}
    assert held == {id(p) for p in net.parameters()}
    assert any(p is net.switch_bias for p in plain["params"])
    assert any(p is net.emb_species.weight for p in decay["params"])


# --- augmentation -------------------------------------------------------------


def test_augment_mirrors_and_blanks_per_example(batch: F.Batch):
    data = F.concat_batches([batch] * 60)
    n = data["act_mon"].shape[0]
    before = {name: array.copy() for name, array in data.items()}
    view = T.augment(data, np.random.default_rng(1), slot_swap=0.5, elo_drop=0.15)
    assert all(np.array_equal(data[name], before[name]) for name in data)
    assert set(view) == set(data)
    assert all(view[name].dtype == data[name].dtype for name in data)

    dropped = (view["elo_known"] == 0).all(1)
    assert 0.10 < dropped.mean() < 0.20
    assert (view["elo"][dropped] == 0).all()
    assert np.array_equal(view["elo"][~dropped], data["elo"][~dropped])
    assert np.array_equal(view["elo_known"][~dropped], data["elo_known"][~dropped])

    # Every row is one of the four mirrors of its source row, each about a quarter.
    variants = {
        (a, o): M.swap_slots(data, a, o) for a in (False, True) for o in (False, True)
    }
    names = [name for name in data if name not in ("elo", "elo_known")]
    counts = dict.fromkeys(variants, 0)
    two_slots = (data["act_mon"] >= 0).all(1) & (data["foe_mon"] >= 0).all(1)
    for row in np.flatnonzero(two_slots):
        matches = [
            key
            for key, variant in variants.items()
            if all(
                np.array_equal(view[name][row], variant[name][row]) for name in names
            )
        ]
        assert len(matches) == 1, row
        counts[matches[0]] += 1
    total = sum(counts.values())
    assert total > 300
    assert all(0.15 < count / total < 0.35 for count in counts.values()), counts
    # The mirrored view scores like the original: labels moved with the features.
    plain = F.slot_nll(F.uniform_prediction(data), data)
    moved = F.slot_nll(F.uniform_prediction(view), view)
    assert moved["fine"].sum() == pytest.approx(plain["fine"].sum())
    assert n == 14 * 60

    same = T.augment(data, np.random.default_rng(1), slot_swap=0.5, elo_drop=0.15)
    assert all(np.array_equal(same[name], view[name]) for name in view)
    # Same seed, with and without the Elo drop: the same mirrors, and the
    # generator ends in the same state (so the example order stays paired too).
    with_drop, without = np.random.default_rng(9), np.random.default_rng(9)
    first = T.augment(data, with_drop, slot_swap=0.5, elo_drop=0.15)
    second = T.augment(data, without, slot_swap=0.5, elo_drop=0.0)
    assert all(np.array_equal(first[name], second[name]) for name in names)
    assert second["elo"] is data["elo"] and not np.array_equal(
        first["elo"], data["elo"]
    )
    assert with_drop.random() == without.random()

    off = T.augment(data, np.random.default_rng(1), slot_swap=0.0, elo_drop=0.0)
    assert all(off[name] is data[name] for name in data)
    only_elo = T.augment(data, np.random.default_rng(1), slot_swap=0.0, elo_drop=1.0)
    assert not only_elo["elo_known"].any() and not only_elo["elo"].any()
    assert all(only_elo[n_] is data[n_] for n_ in names)


# --- snapshot -----------------------------------------------------------------


def test_snapshot_is_unaffected_by_a_further_optimiser_step(
    fz: F.Featurizer, batch: F.Batch
):
    net = TM.small_net(fz).train()
    optimizer = torch.optim.AdamW(T.parameter_groups(net, 0.01), lr=1e-2)
    labels = M.to_labels(batch)

    def step() -> None:
        loss, _ = M.total_loss(
            M.nll_terms(net(M.to_tensors(batch)), labels), labels["weight"]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    step()
    truth = copy.deepcopy(net.state_dict())
    snapshot = M.clone_state(net)
    alias = {name: value.detach().cpu() for name, value in net.state_dict().items()}
    step()
    live = net.state_dict()
    changed = [name for name in truth if not torch.equal(truth[name], live[name])]
    assert len(changed) > 10  # the step really moved the network
    for name in truth:
        assert torch.equal(snapshot[name], truth[name]), name
        assert torch.equal(alias[name], live[name]), name  # what the old trainers kept
        assert snapshot[name].data_ptr() != live[name].data_ptr()
    # Restoring the snapshot brings the earlier predictions back exactly.
    predictor = M.OppNetPredictor(net, fz)
    after = predictor.predict(batch)["action"].copy()
    net.load_state_dict(snapshot)
    restored = predictor.predict(batch)["action"]
    reference = TM.small_net(fz)
    reference.load_state_dict(truth)
    want = M.OppNetPredictor(reference, fz).predict(batch)["action"]
    assert np.array_equal(restored, want)
    assert not np.array_equal(restored, after)


# --- data ---------------------------------------------------------------------


def test_load_splits_returns_only_the_named_splits(
    dataset: tuple[Path, dict[str, int]], monkeypatch: pytest.MonkeyPatch
):
    directory, sizes = dataset
    asked: list[Any] = []
    real = F.load_dataset

    def recording(path: Any, splits: Any = None) -> Any:
        asked.append(splits)
        return real(path, splits)

    monkeypatch.setattr(F, "load_dataset", recording)
    parts, manifest = T.load_splits(directory, ("train", "val"))
    assert asked == [["train", "val"]]
    assert set(parts) == {"train", "val"} and manifest["tag"] == "tiny"
    assert parts["train"]["act_mon"].shape[0] == sizes["train"] == 28
    assert parts["val"]["act_mon"].shape[0] == sizes["val"] == 14
    assert (parts["train"]["m_split"] == 0).all() and (
        parts["val"]["m_split"] == 1
    ).all()
    assert len(T.manifest_sha256(directory)) == 64

    rng = np.random.default_rng(0)
    few = T.subset(parts["train"], 5, rng)
    assert few["act_mon"].shape[0] == 5 and (few["m_split"] == 0).all()
    assert T.subset(parts["train"], None, rng) is parts["train"]
    assert T.subset(parts["train"], 1000, rng) is parts["train"]


# --- scoring ------------------------------------------------------------------


class Watching(M.OppNetPredictor):
    """A predictor that remembers which arrays it was handed."""

    seen: list[set[str]]

    def predict(self, batch: Any) -> dict[str, np.ndarray]:
        self.seen.append(set(batch))
        return super().predict(batch)


def test_score_goes_through_slot_nll_and_hides_the_labels(
    fz: F.Featurizer, batch: F.Batch
):
    data = dict(batch)
    data["m_weight"] = np.ones(batch["act_mon"].shape[0], dtype=np.float32)
    predictor = Watching(TM.jolt(TM.small_net(fz)), fz)
    predictor.seen = []
    scores = T.score(predictor, data)
    assert len(predictor.seen) == 1
    assert not [name for name in predictor.seen[0] if name.startswith(("y_", "m_"))]

    pred = predictor.predict(M.strip_labels(data))
    nll = F.slot_nll(pred, data)
    scored = nll["fine_scored"]
    assert scores["slots"] == int(scored.sum()) and scores["examples"] == 14
    assert scores["fine"] == pytest.approx(nll["fine"][scored].mean())
    assert scores["action"] == pytest.approx(nll["action"][nll["action_scored"]].mean())
    assert scores["action"] + scores["target"] == pytest.approx(scores["fine"])
    assert scores["target_given"] == pytest.approx(
        nll["target"][nll["target_scored"]].mean()
    )
    hits, counted = F.topk_hits(F.fine_probs(pred, data), F.fine_label(data), 1)
    assert scores["top1"] == pytest.approx(hits[counted].mean())
    assert 0.0 <= scores["action_top1"] <= 1.0 and math.isfinite(scores["mega"])

    # A sheet code "unknown" is scored as closed, for this predictor like any other.
    unknown = dict(data)
    flags = data["game_flag"].copy()
    flags[:, [F.G_ACTOR_SHEET, F.G_OTHER_SHEET]] = F.SHEET_UNKNOWN
    unknown["game_flag"] = flags
    assert T.score(predictor, unknown)["fine"] == pytest.approx(scores["fine"])
    assert "sheet_unknown_as_closed" not in predictor.counters  # mapped before predict

    # A predictor that fell back to uniform is an error, not a score.
    sick = TM.small_net(fz)
    with torch.no_grad():
        sick.mon_proj.weight.fill_(float("nan"))
    with pytest.raises(RuntimeError):
        T.score(M.OppNetPredictor(sick, fz), data)


def test_calibrate_fits_one_temperature_per_head(fz: F.Featurizer, batch: F.Batch):
    net = TM.jolt(TM.small_net(fz))
    with torch.no_grad():  # overconfident on purpose
        net.act_cand.out.weight.mul_(6.0)
        net.tgt_foe.out.weight.mul_(6.0)
    found = T.calibrate(net, batch, torch.device("cpu"))
    assert set(found) == {"action", "target", "mega", "mega_bias"}
    assert 0.2 <= found["action"] <= 5.0 and 0.2 <= found["target"] <= 5.0
    plain = M.OppNetPredictor(net, fz)
    before = T.score(plain, batch)
    after = T.score(plain.with_temperatures(found["action"], found["target"]), batch)
    assert after["action"] <= before["action"] + 1e-9
    assert after["target_given"] <= before["target_given"] + 1e-9
    assert after["fine"] < before["fine"]


def test_each_head_is_calibrated_on_its_own_objective(fz: F.Featurizer, batch: F.Batch):
    """Only the target head is over-confident: only its temperature moves.

    Fitted on the action objective it would stay 1.0 (that objective does not
    depend on it), and the target NLL would not improve.
    """
    data = F.concat_batches([batch] * 4)
    net = TM.jolt(TM.small_net(fz))
    with torch.no_grad():
        net.tgt_foe.out.weight.mul_(6.0)
        net.tgt_foe.out.bias.mul_(6.0)
    found = T.calibrate(net, data, torch.device("cpu"))
    assert found["target"] > 1.5
    plain = M.OppNetPredictor(net, fz)
    before = T.score(plain, data)
    cooled = T.score(plain.with_temperatures(1.0, found["target"]), data)
    assert cooled["target_given"] < before["target_given"] - 0.01
    assert cooled["action"] == pytest.approx(before["action"])
    # The action head alone, the other way round.
    net = TM.jolt(TM.small_net(fz))
    with torch.no_grad():
        net.act_cand.out.weight.mul_(6.0)
    found = T.calibrate(net, data, torch.device("cpu"))
    assert found["action"] > 1.5
    after = T.score(M.OppNetPredictor(net, fz).with_temperatures(found["action"]), data)
    assert after["action"] < T.score(M.OppNetPredictor(net, fz), data)["action"] - 0.01


def test_calibrate_cools_an_overconfident_mega_head(fz: F.Featurizer, batch: F.Batch):
    """A Mega head that is far too sure and biased: scale and bias are fitted
    on the Mega NLL and applied through the predictor."""
    data = F.concat_batches([batch] * 4)
    net = TM.jolt(TM.small_net(fz))
    with torch.no_grad():
        net.mega.out.weight.normal_(
            0.0, 4.0, generator=torch.Generator().manual_seed(2)
        )
        net.mega.out.bias.fill_(3.0)
    plain = M.OppNetPredictor(net, fz)
    before = T.score(plain, data)
    found = T.calibrate(net, data, torch.device("cpu"))
    assert (found["mega"], found["mega_bias"]) != (1.0, 0.0)
    assert 0.2 <= found["mega"] <= 5.0 and -4.0 <= found["mega_bias"] <= 4.0
    final = plain.with_temperatures(
        found["action"], found["target"], found["mega"], found["mega_bias"]
    )
    after = T.score(final, data)
    assert after["mega"] < before["mega"] - 0.05
    # Exactly what the trainer's own objective says it should be.
    outputs = M.collect_outputs(net, M.strip_labels(data))
    terms = M.nll_terms(
        outputs,
        M.to_labels(data),
        mega_temperature=found["mega"],
        mega_bias=found["mega_bias"],
    )
    scored = terms["mega_scored"]
    assert after["mega"] == pytest.approx(float(terms["mega"][scored].mean()), abs=1e-5)
    # A head that is already as good as it gets is left alone.
    with torch.no_grad():
        net.mega.out.weight.zero_()
        net.mega.out.bias.zero_()
    rate = float((data["y_mega"][scored.numpy()] == 1).mean())
    assert 0.05 < rate < 0.95
    level = math.log(rate / (1 - rate))
    with torch.no_grad():
        net.mega.out.bias.fill_(level)
    still = T.calibrate(net, data, torch.device("cpu"))
    assert abs(still["mega_bias"]) < 1e-3
    # A head that is wrong in LEVEL only: a constant logit three above the
    # right one. No temperature can move it across zero; the bias must.
    with torch.no_grad():
        net.mega.out.bias.fill_(level + 3.0)
    shifted = T.calibrate(net, data, torch.device("cpu"))
    fixed = shifted["mega_bias"] + (level + 3.0) / shifted["mega"]
    assert shifted["mega_bias"] < -0.5 and fixed == pytest.approx(level, abs=0.02)
    entropy = -(rate * math.log(rate) + (1 - rate) * math.log(1 - rate))
    level_only = M.OppNetPredictor(net, fz).with_temperatures(
        1.0, 1.0, shifted["mega"], shifted["mega_bias"]
    )
    assert T.score(level_only, data)["mega"] == pytest.approx(entropy, abs=1e-3)
    cooled_only = M.OppNetPredictor(net, fz).with_temperatures(1.0, 1.0, 5.0, 0.0)
    assert T.score(cooled_only, data)["mega"] > entropy + 0.01


# --- the full run -------------------------------------------------------------


def test_run_writes_an_artifact_that_reloads_to_the_best_epoch(
    dataset: tuple[Path, dict[str, int]],
    tmp_path: Path,
    artifact: Any,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    fz: F.Featurizer,
):
    directory, sizes = dataset
    out = tmp_path / "run"
    asked: list[Any] = []
    real = F.load_dataset

    def recording(path: Any, splits: Any = None) -> Any:
        asked.append(splits)
        return real(path, splits)

    monkeypatch.setattr(F, "load_dataset", recording)
    monkeypatch.setattr(T, "PROGRESS_EVERY", 2)
    code = run(directory, out, "--epochs", "3", "--tag", "tiny-net")
    lines = lines_of(capsys)
    assert code == 0, lines[-1]
    assert lines[0].startswith("TRAIN_START") and lines[-1].startswith("TRAIN_DONE")
    assert [line.split()[1] for line in lines if line.startswith("EPOCH")] == [
        "0/3",
        "1/3",
        "2/3",
        "3/3",
    ]
    assert sum(line.startswith("  step ") for line in lines) == 3  # 6 steps, every 2
    assert any(line.startswith("CALIBRATED") for line in lines)
    assert any(line.startswith("RELOAD") for line in lines)
    assert not any(line.startswith("INFORMATIONAL") for line in lines)
    assert asked == [["train", "val"]]  # test and ladder holdout were never read

    report = json.loads((out / "train_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "TRAIN_DONE" and report["name"] == "tiny-net"
    assert report["examples"] == {"train": sizes["train"], "val": sizes["val"]}
    assert report["dataset"]["tag"] == "tiny"
    assert report["dataset"]["manifest_sha256"] == T.manifest_sha256(directory)
    assert report["n_parameters"] > 10_000 and report["elo_mode"] == "keep"
    assert report["timing"]["steps"] == 6 and report["timing"]["seconds_per_step"] > 0
    history = report["history"]
    assert [row["epoch"] for row in history] == [0, 1, 2, 3]
    assert history[0]["train_loss"] is None  # epoch 0 is the untrained prior
    assert all(row["val_fine"] > 0 for row in history)
    assert history[1]["lr"] > 0
    best = report["best"]
    assert best["val_fine"] == pytest.approx(history[best["epoch"]]["val_fine"])
    assert best["val_fine"] <= min(row["val_fine"] for row in history) + 2e-4
    assert report["reload"]["difference"] <= 1e-4
    assert report["reload"]["val_fine_raw"] == pytest.approx(best["val_fine"], abs=1e-4)
    calibration = report["calibration"]
    assert calibration["val_after"]["fine"] <= calibration["val_before"]["fine"] + 1e-9
    assert report["reload"]["val_fine_tempered"] == pytest.approx(
        calibration["val_after"]["fine"], abs=1e-6
    )
    text = (out / "train_report.md").read_text(encoding="utf-8")
    assert "tiny-net" in text and "| 3 |" in text and "TRAIN_DONE" in text

    stored = artifact.read_artifact(out / "artifact.pt")
    assert stored["kind"] == "oppnet" and stored["name"] == "tiny-net"
    extra = stored["extra"]
    assert extra["dataset"]["manifest_sha256"] == report["dataset"]["manifest_sha256"]
    assert extra["config"]["d_model"] == 32 and extra["n_parameters"] > 0
    assert len(extra["history"]) == 4 and "calibration" in extra and "best" in extra
    loaded = artifact.load_predictor(out / "artifact.pt")
    assert (loaded.kind, loaded.name) == ("oppnet", "tiny-net")
    predictor = loaded.predictor
    assert isinstance(predictor, M.OppNetPredictor)
    assert predictor.temperatures["action"] == pytest.approx(
        calibration["action_temperature"]
    )
    val = T.load_splits(directory, ("val",))[0]["val"]
    raw = T.score(predictor.with_temperatures(1.0, 1.0), val)
    assert raw["fine"] == pytest.approx(best["val_fine"], abs=1e-4)
    assert T.score(predictor, val)["fine"] == pytest.approx(
        report["reload"]["val_fine_tempered"], abs=1e-6
    )


def test_best_epoch_is_kept_when_later_epochs_get_worse(
    dataset: tuple[Path, dict[str, int]],
    tmp_path: Path,
    artifact: Any,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    """A gentle first epoch, then a destructive rate: the saved weights must be
    the early ones, which the reload check proves by re-scoring them."""
    directory, _ = dataset
    out = tmp_path / "run"
    steps_per_epoch = 2
    monkeypatch.setattr(
        T,
        "learning_rate",
        lambda step, total, warmup, base, floor=0.05: (
            1e-3 if step < steps_per_epoch else 0.3
        ),
    )
    code = run(directory, out, "--epochs", "6", "--patience", "3")
    lines = lines_of(capsys)
    assert code == 0, lines[-1]
    report = json.loads((out / "train_report.json").read_text(encoding="utf-8"))
    history = report["history"]
    best = report["best"]
    assert best["epoch"] <= 1 < history[-1]["epoch"]
    assert history[-1]["val_fine"] > best["val_fine"] + 0.05  # the end is worse
    assert report["stopped_early"] is True and history[-1]["epoch"] == best["epoch"] + 3
    assert any(line.startswith("EARLY_STOP") for line in lines)
    # Last-epoch weights would have scored history[-1]; the artifact scores the best.
    assert report["reload"]["val_fine_raw"] == pytest.approx(best["val_fine"], abs=1e-4)
    loaded = artifact.load_predictor(out / "artifact.pt").predictor
    val = T.load_splits(directory, ("val",))[0]["val"]
    raw = T.score(loaded.with_temperatures(1.0, 1.0), val)["fine"]
    assert raw == pytest.approx(best["val_fine"], abs=1e-4)
    assert abs(raw - history[-1]["val_fine"]) > 0.05


def test_elo_blind_run_stores_a_blind_predictor(
    dataset: tuple[Path, dict[str, int]],
    tmp_path: Path,
    artifact: Any,
    capsys: pytest.CaptureFixture[str],
):
    directory, _ = dataset
    out = tmp_path / "blind"
    code = run(directory, out, "--epochs", "2", "--elo-mode", "blank")
    lines = lines_of(capsys)
    assert code == 0, lines[-1]
    assert "elo_mode blank" in lines[0]
    report = json.loads((out / "train_report.json").read_text(encoding="utf-8"))
    assert report["elo_mode"] == "blank" and report["args"]["elo_mode"] == "blank"
    stored = artifact.read_artifact(out / "artifact.pt")
    assert stored["predictor"]["elo_mode"] == "blank"
    predictor = artifact.load_predictor(out / "artifact.pt").predictor
    assert predictor.elo_mode == F.ELO_BLANK
    val = T.load_splits(directory, ("val",))[0]["val"]
    features_only = M.strip_labels(val)
    assert (features_only["elo_known"] == 1).all()  # the caller still passes ratings
    want = predictor.predict(features_only)
    for mode in (F.ELO_BLANK, F.ELO_SWAP, F.ELO_SHUFFLE):
        got = predictor.predict(F.apply_elo_mode(features_only, mode))
        assert all(np.array_equal(want[name], got[name]) for name in want), mode
    other = dict(features_only)
    other["elo"] = np.full_like(features_only["elo"], 1800)
    got = predictor.predict(other)
    assert all(np.array_equal(want[name], got[name]) for name in want)


def test_the_network_sees_ratings_only_as_the_mode_allows(
    dataset: tuple[Path, dict[str, int]],
    tmp_path: Path,
    artifact: Any,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    """Blank mode: no rating ever reaches the network, in training or scoring.
    Keep mode: training sees ratings with some blanked, validation sees them all."""
    directory, _ = dataset
    seen: list[tuple[bool, float, float]] = []
    real = M.OppNet.forward

    def watching(self: M.OppNet, x: Any) -> Any:
        known = x["elo_known"].float()
        seen.append((self.training, float(known.mean()), float(x["elo"].abs().max())))
        return real(self, x)

    monkeypatch.setattr(M.OppNet, "forward", watching)
    assert (
        run(directory, tmp_path / "blind", "--epochs", "2", "--elo-mode", "blank") == 0
    )
    assert any(training for training, _, _ in seen)
    assert any(not training for training, _, _ in seen)
    assert all(known == 0 and elo == 0 for _, known, elo in seen)

    seen.clear()
    assert run(directory, tmp_path / "keep", "--epochs", "3", "--elo-drop", "0.5") == 0
    train_known = [known for training, known, _ in seen if training]
    eval_known = [known for training, known, _ in seen if not training]
    assert train_known and eval_known
    assert all(known == 1.0 for known in eval_known)  # validation is never blanked
    assert 0.2 < float(np.mean(train_known)) < 0.8  # about half the examples blanked
    capsys.readouterr()


def test_training_loss_uses_each_example_once_with_its_own_weight(
    dataset: tuple[Path, dict[str, int]],
    tmp_path: Path,
    artifact: Any,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    directory, sizes = dataset
    seen: list[tuple[np.ndarray, float]] = []
    real = M.total_loss

    def watching(terms: Any, weight: Any, mega_weight: float = 0.2) -> Any:
        seen.append((weight.detach().numpy().copy(), mega_weight))
        return real(terms, weight, mega_weight)

    monkeypatch.setattr(M, "total_loss", watching)
    assert (
        run(directory, tmp_path / "out", "--epochs", "1", "--mega-weight", "0.3") == 0
    )
    capsys.readouterr()
    train = T.load_splits(directory, ("train",))[0]["train"]
    used = np.sort(np.concatenate([weight for weight, _ in seen]))
    assert used.shape[0] == sizes["train"] and len(seen) == 2  # one epoch, 16 + 12
    assert np.allclose(used, np.sort(train["m_weight"]))
    assert used.min() < 0.6 and used.max() == pytest.approx(1.0)
    assert all(mega_weight == 0.3 for _, mega_weight in seen)


def test_informational_scores_come_after_the_artifact(
    dataset: tuple[Path, dict[str, int]],
    tmp_path: Path,
    artifact: Any,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    directory, sizes = dataset
    out = tmp_path / "info"
    events: list[str] = []
    real_load, real_save = F.load_dataset, artifact.save_artifact

    def recording(path: Any, splits: Any = None) -> Any:
        events.append("load:" + ",".join(splits or ["all"]))
        return real_load(path, splits)

    def saving(*args: Any, **kwargs: Any) -> Any:
        events.append("save")
        return real_save(*args, **kwargs)

    monkeypatch.setattr(F, "load_dataset", recording)
    monkeypatch.setattr(artifact, "save_artifact", saving)
    code = run(directory, out, "--epochs", "1", "--informational")
    lines = lines_of(capsys)
    assert code == 0, lines[-1]
    assert events == ["load:train,val", "save", "load:test,ladder_holdout"]
    informational = [line for line in lines if line.startswith("INFORMATIONAL")]
    assert len(informational) == 2 and lines[-1].startswith("TRAIN_DONE")
    assert lines.index(informational[0]) > next(
        i for i, line in enumerate(lines) if line.startswith("RELOAD")
    )
    report = json.loads((out / "train_report.json").read_text(encoding="utf-8"))
    assert set(report["informational"]) == {"test", "ladder_holdout"}
    assert report["informational"]["test"]["examples"] == sizes["test"]


def test_limit_keeps_a_smoke_small(
    dataset: tuple[Path, dict[str, int]],
    tmp_path: Path,
    artifact: Any,
    capsys: pytest.CaptureFixture[str],
):
    directory, _ = dataset
    out = tmp_path / "smoke"
    code = run(directory, out, "--epochs", "1", "--limit", "9")
    lines = lines_of(capsys)
    assert code == 0, lines[-1]
    report = json.loads((out / "train_report.json").read_text(encoding="utf-8"))
    assert report["examples"] == {"train": 9, "val": 9}
    assert report["timing"]["steps"] == 1


# --- failures -----------------------------------------------------------------


def test_failure_prints_train_failed_and_returns_non_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    code = run(tmp_path / "no-such-dataset", tmp_path / "out", "--epochs", "1")
    lines = lines_of(capsys)
    assert code == 1
    assert lines[-1].startswith("TRAIN_FAILED FileNotFoundError")
    assert "\n" not in lines[-1]
    assert not (tmp_path / "out" / "artifact.pt").exists()

    code = T.main(["--epochs", "many"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out.strip().splitlines()[-1].startswith("TRAIN_FAILED")


def test_non_finite_loss_stops_the_run(
    dataset: tuple[Path, dict[str, int]],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    directory, _ = dataset
    real = M.total_loss

    def broken(terms: Any, weight: Any, mega_weight: float = 0.2) -> Any:
        loss, parts = real(terms, weight, mega_weight)
        return loss, {**parts, "loss": float("nan")}

    monkeypatch.setattr(M, "total_loss", broken)
    code = run(directory, tmp_path / "out", "--epochs", "1")
    lines = lines_of(capsys)
    assert code == 1 and lines[-1].startswith("TRAIN_FAILED RuntimeError: non-finite")
    assert not (tmp_path / "out" / "artifact.pt").exists()


def test_an_artifact_that_reloads_differently_fails_the_run(
    dataset: tuple[Path, dict[str, int]],
    tmp_path: Path,
    artifact: Any,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    directory, _ = dataset
    out = tmp_path / "out"
    real = artifact.load_predictor

    def tampered(path: Any, *args: Any, **kwargs: Any) -> Any:
        loaded = real(path, *args, **kwargs)
        with torch.no_grad():
            loaded.predictor.net.switch_bias.add_(2.0)
        return loaded

    monkeypatch.setattr(artifact, "load_predictor", tampered)
    code = run(directory, out, "--epochs", "1")
    lines = lines_of(capsys)
    assert code == 1 and lines[-1].startswith("TRAIN_FAILED RuntimeError: reloaded")
    report = json.loads((out / "train_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "TRAIN_FAILED"
    assert report["reload"]["difference"] > 1e-4
    # The model the run rejected is not left where a consumer looks for one.
    assert not (out / "artifact.pt").exists()
    rejected = out / ("artifact.pt" + T.UNVERIFIED_SUFFIX)
    assert rejected.exists() and report["rejected_artifact"] == str(rejected)
    assert report["artifact"] is None and "reloaded artifact" in report["reason"]
    assert "TRAIN_FAILED" in (out / "train_report.md").read_text(encoding="utf-8")


def test_missing_artifact_module_fails_cleanly(
    dataset: tuple[Path, dict[str, int]],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    directory, _ = dataset
    real = importlib.import_module

    def missing(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == TM.ARTIFACT_MODULE:
            raise ModuleNotFoundError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(T.importlib, "import_module", missing)
    code = run(directory, tmp_path / "out", "--epochs", "1")
    lines = lines_of(capsys)
    assert code == 1 and lines[-1].startswith("TRAIN_FAILED ModuleNotFoundError")


# --- the run directory ---------------------------------------------------------


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_run_directory_is_not_written_over_without_overwrite(
    dataset: tuple[Path, dict[str, int]],
    tmp_path: Path,
    artifact: Any,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    directory, _ = dataset
    out = tmp_path / "run"
    assert run(directory, out, "--epochs", "1") == 0
    first = {path.name: digest(path) for path in out.iterdir()}
    assert set(first) == {"artifact.pt", "train_report.json", "train_report.md"}
    capsys.readouterr()
    # The same command again: refused before anything is touched.
    assert run(directory, out, "--epochs", "2", "--seed", "4") == 1
    last = lines_of(capsys)[-1]
    assert last.startswith("TRAIN_FAILED") and "--overwrite" in last
    assert {path.name: digest(path) for path in out.iterdir()} == first
    report = json.loads((out / "train_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "TRAIN_DONE" and report["args"]["epochs"] == 1
    # A report alone (a run that never finished) is refused as well.
    half = tmp_path / "half"
    half.mkdir()
    (half / "train_report.json").write_text("{}", encoding="utf-8")
    assert run(directory, half, "--epochs", "1") == 1
    assert (half / "train_report.json").read_text(encoding="utf-8") == "{}"
    capsys.readouterr()
    # With --overwrite the run replaces what was there.
    assert run(directory, out, "--epochs", "2", "--seed", "4", "--overwrite") == 0
    second = {path.name: digest(path) for path in out.iterdir()}
    assert set(second) == set(first) and second["artifact.pt"] != first["artifact.pt"]
    report = json.loads((out / "train_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "TRAIN_DONE" and report["args"]["epochs"] == 2
    capsys.readouterr()

    # A rerun that fails mid-training must not leave the earlier artifact
    # beside a report that still says TRAIN_DONE.
    real = M.total_loss

    def broken(terms: Any, weight: Any, mega_weight: float = 0.2) -> Any:
        loss, parts = real(terms, weight, mega_weight)
        return loss, {**parts, "loss": float("nan")}

    monkeypatch.setattr(M, "total_loss", broken)
    assert run(directory, out, "--epochs", "1", "--overwrite") == 1
    assert lines_of(capsys)[-1].startswith("TRAIN_FAILED RuntimeError: non-finite")
    assert not (out / "artifact.pt").exists()
    report = json.loads((out / "train_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "TRAIN_FAILED" and "non-finite" in report["reason"]
    assert report["args"]["epochs"] == 1  # this run's report, not the last one's
    assert "TRAIN_FAILED" in (out / "train_report.md").read_text(encoding="utf-8")


def test_the_report_says_running_while_the_run_is_on(
    dataset: tuple[Path, dict[str, int]],
    tmp_path: Path,
    artifact: Any,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    directory, _ = dataset
    out = tmp_path / "run"
    seen: list[dict[str, Any]] = []
    real = T._train

    def watching(args: Any, name: str, out_dir: Path) -> Any:
        seen.append(json.loads((out_dir / "train_report.json").read_text()))
        assert not (out_dir / "artifact.pt").exists()
        return real(args, name, out_dir)

    monkeypatch.setattr(T, "_train", watching)
    assert run(directory, out, "--epochs", "1", "--tag", "named") == 0
    capsys.readouterr()
    assert [report["status"] for report in seen] == ["TRAIN_RUNNING"]
    assert seen[0]["name"] == "named" and seen[0]["args"]["epochs"] == 1
    final = json.loads((out / "train_report.json").read_text(encoding="utf-8"))
    assert final["status"] == "TRAIN_DONE"
    assert T.resolve_out(T.parse_args(["--out", str(out)])) == ("run", out)
    assert T.resolve_out(T.parse_args([]))[1] == T.ROOT / "results_oppmodel" / "oppnet"


# --- which rows choose the epoch and the calibration ------------------------------


def test_calibration_and_early_stopping_read_validation_only(
    dataset: tuple[Path, dict[str, int]],
    tmp_path: Path,
    artifact: Any,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    directory, sizes = dataset
    val_code = SPLITS.index("val")
    calibrated: list[np.ndarray] = []
    scored: list[np.ndarray] = []
    real_calibrate, real_score = T.calibrate, T.score

    def calibrate(net: Any, batch: Any, device: Any) -> Any:
        calibrated.append(np.asarray(batch["m_split"]).copy())
        return real_calibrate(net, batch, device)

    def score(predictor: Any, batch: Any) -> Any:
        scored.append(np.asarray(batch["m_split"]).copy())
        return real_score(predictor, batch)

    monkeypatch.setattr(T, "calibrate", calibrate)
    monkeypatch.setattr(T, "score", score)
    assert run(directory, tmp_path / "run", "--epochs", "2") == 0
    capsys.readouterr()
    assert len(calibrated) == 1 and calibrated[0].shape[0] == sizes["val"]
    assert np.unique(calibrated[0]).tolist() == [val_code]
    assert len(scored) >= 5  # epoch 0, 1, 2, the restored best, the reload
    assert all(np.unique(rows).tolist() == [val_code] for rows in scored)


def test_validation_leaves_out_battles_with_a_holdout_opponent(
    tmp_path: Path, fz: F.Featurizer, artifact: Any, capsys: pytest.CaptureFixture[str]
):
    batch = TM.game_batch(fz)
    n = int(batch["act_mon"].shape[0])
    kept, left = T.without_holdout_battles(batch)
    assert kept is batch and left == 0  # no m_flag: nothing to go by
    flagged = dict(batch)
    flagged["m_flag"] = np.zeros(n, dtype=np.uint8)
    assert T.without_holdout_battles(flagged)[1] == 0
    flagged["m_flag"][[0, 3, 5]] = [T.FLAG_HOLDOUT_BATTLE, 12, 1]  # 4, 4 + 8, time
    kept, left = T.without_holdout_battles(flagged)
    assert left == 2 and int(kept["act_mon"].shape[0]) == n - 2
    assert kept["m_flag"].tolist() == [0] * 3 + [1] + [0] * (n - 6)

    # In a run: the marked validation rows choose nothing.
    directory = tmp_path / "data"
    sizes = write_dataset(directory, fz)
    data = F.load_batch(directory / "shard-00000.npz")
    flags = np.zeros(int(data["turn"].shape[0]), dtype=np.uint8)
    val_rows = np.flatnonzero(data["m_split"] == SPLITS.index("val"))
    train_rows = np.flatnonzero(data["m_split"] == SPLITS.index("train"))
    flags[val_rows[:5]] = T.FLAG_HOLDOUT_BATTLE
    flags[train_rows[:7]] = T.FLAG_HOLDOUT_BATTLE  # never in a real build; kept
    data["m_flag"] = flags
    F.save_batch(directory / "shard-00000.npz", data)
    out = tmp_path / "run"
    assert run(directory, out, "--epochs", "1") == 0
    lines = lines_of(capsys)
    report = json.loads((out / "train_report.json").read_text(encoding="utf-8"))
    assert report["examples"] == {"train": sizes["train"], "val": sizes["val"] - 5}
    assert report["dataset"]["validation_rows_left_out_holdout_battles"] == 5
    assert "left out: 5 rows" in lines[0]
    assert report["history"][0]["val_examples"] == sizes["val"] - 5


def test_target_labels_outside_the_mask_are_counted_and_reported(
    tmp_path: Path, fz: F.Featurizer, artifact: Any, capsys: pytest.CaptureFixture[str]
):
    directory = tmp_path / "data"
    write_dataset(directory, fz)
    out = tmp_path / "clean"
    assert run(directory, out, "--epochs", "1") == 0
    lines = lines_of(capsys)
    assert not any(line.startswith("WARNING target labels") for line in lines)
    report = json.loads((out / "train_report.json").read_text(encoding="utf-8"))
    assert report["target_labels_outside_mask"] == {"train": 0, "val": 0}

    # Spoil two training labels: the target is one the legal mask excludes.
    data = F.load_batch(directory / "shard-00000.npz")
    legal = F.expand_target_mask(data["cand_tmask"])
    train = data["m_split"] == SPLITS.index("train")
    aimed = np.argwhere(
        train[:, None] & (data["y_target"] >= 0) & (data["y_action"] >= 0)
    )
    for row, slot in aimed[:2]:
        action = int(data["y_action"][row, slot])
        data["y_target"][row, slot] = np.flatnonzero(~legal[row, slot, action])[0]
    F.save_batch(directory / "shard-00000.npz", data)
    out = tmp_path / "spoiled"
    assert run(directory, out, "--epochs", "1") == 0
    lines = lines_of(capsys)
    warning = [line for line in lines if line.startswith("WARNING target labels")]
    assert len(warning) == 1 and "'train': 2" in warning[0]
    report = json.loads((out / "train_report.json").read_text(encoding="utf-8"))
    assert report["target_labels_outside_mask"] == {"train": 2, "val": 0}
    assert "{'train': 2, 'val': 0}" in (out / "train_report.md").read_text()


# --- the optimisation loop ---------------------------------------------------------


def test_one_epoch_equals_a_loop_written_by_hand(
    dataset: tuple[Path, dict[str, int]],
    tmp_path: Path,
    artifact: Any,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    """Seeds, example order, gradient reset, clipping and the optimiser step.

    The trainer's weights after one epoch must equal those of the loop below,
    which names every step: a seed that is not used, an order that is not
    shuffled, gradients that are not zeroed or not clipped each change them.
    """
    directory, sizes = dataset
    seed, batch_size, base, clip, decay, mega_weight = 3, 16, 1e-3, 0.05, 0.01, 0.2
    states: list[dict[str, torch.Tensor]] = []
    real_score = T.score

    def recording(predictor: Any, batch: Any) -> Any:
        states.append(M.clone_state(predictor.net))
        return real_score(predictor, batch)

    monkeypatch.setattr(T, "score", recording)
    code = run(
        directory,
        tmp_path / "out",
        "--epochs",
        "1",
        "--dropout",
        "0",
        "--clip",
        str(clip),
    )
    assert code == 0, lines_of(capsys)[-1]
    untrained, trained = states[0], states[1]  # before training, after epoch 1

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    splits, _ = T.load_splits(directory, ("train",))
    train_set = F.sheet_unknown_as_closed(splits["train"])
    n = sizes["train"]
    net = M.OppNet.for_featurizer(
        F.Featurizer.load(directory),
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=48,
        dropout=0.0,
    )
    assert all(torch.equal(untrained[k], v) for k, v in net.state_dict().items())
    optimizer = torch.optim.AdamW(T.parameter_groups(net, decay), lr=base)
    steps = math.ceil(n / batch_size)
    warmup = min(2, max(1, steps // 10))
    net.train()
    view = T.augment(train_set, rng, 0.5, 0.15)
    order = rng.permutation(n)
    assert not np.array_equal(order, np.arange(n))
    norms: list[float] = []
    for position in range(steps):
        index = np.sort(order[position * batch_size : (position + 1) * batch_size])
        for group in optimizer.param_groups:
            group["lr"] = T.learning_rate(position, steps, warmup, base)
        labels = M.to_labels(view, "cpu", index)
        terms = M.nll_terms(net(M.to_tensors(view, "cpu", index)), labels)
        loss, _ = M.total_loss(terms, labels["weight"], mega_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norms.append(float(torch.nn.utils.clip_grad_norm_(net.parameters(), clip)))
        optimizer.step()
    assert steps >= 2 and min(norms) > clip  # the cap was at work in every step
    mine = net.state_dict()
    assert set(mine) == set(trained)
    worst = max(float((mine[k] - trained[k]).abs().max()) for k in mine)
    assert worst <= 1e-7, worst
    moved = max(float((trained[k] - untrained[k]).abs().max()) for k in mine)
    assert moved > 1e-4  # and the epoch did train


def test_the_seed_decides_the_run(
    dataset: tuple[Path, dict[str, int]],
    tmp_path: Path,
    artifact: Any,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    directory, sizes = dataset
    orders: dict[str, list[np.ndarray]] = {}
    real = M.to_labels

    def recording(batch: Any, device: Any = None, index: Any = None) -> Any:
        if index is not None and current:
            orders.setdefault(current[0], []).append(np.asarray(index).copy())
        return real(batch, device, index)

    monkeypatch.setattr(M, "to_labels", recording)
    current: list[str] = []
    weights: dict[str, dict[str, torch.Tensor]] = {}
    for label, seed in (("a", "3"), ("b", "3"), ("c", "4")):
        current[:] = [label]
        out = tmp_path / label
        code = T.main(
            ["--dataset", str(directory), "--out", str(out), "--batch", "16"]
            + [
                "--threads",
                "1",
                "--warmup",
                "2",
                "--seed",
                seed,
                "--epochs",
                "2",
                *TINY,
            ]
        )
        assert code == 0, lines_of(capsys)[-1]
        payload = artifact.read_artifact(out / "artifact.pt")["predictor"]
        weights[label] = payload["state_dict"]
    capsys.readouterr()
    same = all(torch.equal(weights["a"][k], weights["b"][k]) for k in weights["a"])
    assert same  # same seed: the same model, bit for bit
    assert any(not torch.equal(weights["a"][k], weights["c"][k]) for k in weights["a"])
    # The example order follows the seed, and changes from epoch to epoch.
    steps = math.ceil(sizes["train"] / 16)
    assert steps >= 2 and len(orders["a"]) == 2 * steps  # two epochs
    assert all(np.array_equal(x, y) for x, y in zip(orders["a"], orders["b"]))
    assert any(not np.array_equal(x, y) for x, y in zip(orders["a"], orders["c"]))
    first, second = orders["a"][:steps], orders["a"][steps:]
    assert any(not np.array_equal(x, y) for x, y in zip(first, second))
    for epoch in (first, second):  # every example once per epoch
        assert sorted(np.concatenate(epoch).tolist()) == list(range(sizes["train"]))
    # Not the dataset's own order cut into batches.
    assert not np.array_equal(first[0], np.arange(16))


def test_the_untrained_network_is_kept_when_training_only_hurts(
    dataset: tuple[Path, dict[str, int]],
    tmp_path: Path,
    artifact: Any,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    """A destructive rate from the first step: epoch 0 (the prior) is the best
    epoch and it is what the artifact holds."""
    directory, _ = dataset
    out = tmp_path / "run"
    monkeypatch.setattr(
        T, "learning_rate", lambda step, total, warmup, base, floor=0.05: 0.3
    )
    code = run(directory, out, "--epochs", "3", "--patience", "5")
    lines = lines_of(capsys)
    assert code == 0, lines[-1]
    report = json.loads((out / "train_report.json").read_text(encoding="utf-8"))
    history = report["history"]
    assert report["best"]["epoch"] == 0
    assert all(row["val_fine"] > history[0]["val_fine"] + 0.05 for row in history[1:])
    assert report["reload"]["val_fine_raw"] == pytest.approx(
        history[0]["val_fine"], abs=1e-6
    )
    # The stored weights are the untrained ones: the species prior.
    torch.manual_seed(3)
    fresh = M.OppNet.for_featurizer(
        F.Featurizer.load(directory), d_model=32, n_layers=1, n_heads=2, d_ff=48
    )
    stored = artifact.read_artifact(out / "artifact.pt")["predictor"]["state_dict"]
    assert all(torch.equal(stored[k], v) for k, v in fresh.state_dict().items())


def test_every_fitted_calibration_value_reaches_the_artifact(
    dataset: tuple[Path, dict[str, int]],
    tmp_path: Path,
    artifact: Any,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    directory, _ = dataset
    out = tmp_path / "run"
    fixed = {"action": 1.3, "target": 0.7, "mega": 1.9, "mega_bias": -0.3}
    monkeypatch.setattr(T, "calibrate", lambda net, batch, device: dict(fixed))
    assert run(directory, out, "--epochs", "1") == 0
    lines = lines_of(capsys)
    calibrated = next(line for line in lines if line.startswith("CALIBRATED"))
    assert "mega_T 1.9000" in calibrated and "mega_bias -0.3000" in calibrated
    assert "val_mega" in calibrated
    predictor = artifact.load_predictor(out / "artifact.pt").predictor
    assert predictor.temperatures == {"action": 1.3, "target": 0.7}
    assert predictor.mega_calibration == {"temperature": 1.9, "bias": -0.3}
    report = json.loads((out / "train_report.json").read_text(encoding="utf-8"))
    calibration = report["calibration"]
    assert calibration["mega_temperature"] == 1.9 and calibration["mega_bias"] == -0.3
    # The reported "after" Mega NLL is the stored predictor's own.
    val = T.load_splits(directory, ("val",))[0]["val"]
    assert T.score(predictor, val)["mega"] == pytest.approx(
        calibration["val_after"]["mega"], abs=1e-6
    )
    assert calibration["val_after"]["mega"] != pytest.approx(
        calibration["val_before"]["mega"], abs=1e-4
    )
