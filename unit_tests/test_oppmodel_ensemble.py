"""The ensemble artifact kind: the mean of the members' fine distributions.

What must hold: the prediction IS the weighted mean of the members' fine
distributions and a proper prediction; one member alone is that member; the
order of the members does not matter; members share one featurizer or are
refused; a coupled member is refused; a member that fails fails the call
(never a mean of the rest); the artifact round-trips through
``artifact.load_predictor`` and the runtime serves it; old kinds are what
they were and an old reader refuses the new kind; a pair coupling is bound to
the weights and to each member's own payload (also when the members'
temperatures are equal); the own ladder games a fine-tuned member has seen
are the ensemble's record, and are not scored by the script.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evaluation import oppmodel_scorecard as SC
from training import make_oppmodel_ensemble as MK
from unit_tests import test_oppmodel_coupling_review as TR
from unit_tests import test_oppmodel_model as TM
from unit_tests import test_oppmodel_runtime as TRT
from unit_tests.test_oppmodel_coupling import some_table
from vgc_bench.src.oppmodel import artifact as A
from vgc_bench.src.oppmodel import coupling as CP
from vgc_bench.src.oppmodel import ensemble as EN
from vgc_bench.src.oppmodel import events as E
from vgc_bench.src.oppmodel import features as F
from vgc_bench.src.oppmodel import model as M
from vgc_bench.src.oppmodel import runtime as R
from vgc_bench.src.oppmodel import tables as T

ROOT = Path(__file__).resolve().parents[1]
MU_NAMES = ("mu_cand", "mu_slot", "mu_roster")
HEADS = ("action", "target", "mega")
# The format of the hand-made battle of ``test_oppmodel_runtime``.
PLAYED = "gen9championsvgc2026regmc"
FORECAST_KEYS = {
    "turn",
    "model",
    "kind",
    "elo",
    "own_elo",
    "sheets",
    "sheets_reported",
    "latency_ms",
    "slots",
}


@pytest.fixture(scope="module")
def fz() -> F.Featurizer:
    return F.Featurizer.build(TRT.repertoire())


@pytest.fixture(scope="module")
def fz2() -> F.Featurizer:
    return F.Featurizer.build(TRT.repertoire(), layout_version=2, extras=MU_NAMES)


@pytest.fixture(scope="module")
def batch(fz: F.Featurizer) -> F.Batch:
    return TM.game_batch(fz)


@pytest.fixture(scope="module")
def batch2(fz2: F.Featurizer) -> F.Batch:
    return TM.game_batch(fz2)


def net(fz: F.Featurizer, seed: int, **more: Any) -> M.OppNetPredictor:
    """A small network with random heads and its own temperatures."""
    made = TM.jolt(TM.small_net(fz, seed=seed), seed=seed + 100)
    more.setdefault("action_temperature", 1.0 + 0.1 * seed)
    more.setdefault("target_temperature", 1.0 + 0.05 * seed)
    return M.OppNetPredictor(made, fz, name=f"net{seed}", **more)


def matchup_net(fz2: F.Featurizer, seed: int) -> M.OppNetPredictor:
    """A network that reads the three matchup arrays, with weights in them."""
    made = TM.jolt(
        TM.small_net(fz2, seed=seed, **M.extras_overrides(fz2, 7)), seed=seed + 100
    )
    generator = np.random.default_rng(seed)
    import torch

    with torch.no_grad():
        for name, value in made.named_parameters():
            if name.startswith("mu_"):
                value.copy_(
                    torch.from_numpy(generator.normal(size=tuple(value.shape))).float()
                )
    return M.OppNetPredictor(made, fz2, name=f"mu{seed}")


def member(label: str, predictor: Any, kind: str = M.KIND) -> EN.Member:
    return EN.Member(label, kind, predictor)


def held(predictor: Any, fz: F.Featurizer, kind: str = M.KIND) -> A.LoadedPredictor:
    return A.LoadedPredictor(predictor, fz, kind, predictor.name, {"extra": {}})


def mean_of_fines(
    preds: list[dict[str, np.ndarray]], batch: F.Batch, weights: list[float]
) -> np.ndarray:
    out = np.zeros_like(F.fine_probs(preds[0], batch))
    for weight, pred in zip(weights, preds):
        out += weight * F.fine_probs(pred, batch)
    return out


def equal(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> bool:
    return all(np.array_equal(first[head], second[head]) for head in HEADS)


def save_net(path: Path, predictor: M.OppNetPredictor, fz: F.Featurizer, **more: Any):
    A.save_artifact(
        path,
        kind=A.KIND_OPPNET,
        name=predictor.name,
        featurizer=fz,
        predictor_payload=predictor.to_payload(),
        **more,
    )
    return path


# --- the mean ---------------------------------------------------------------------


@pytest.mark.parametrize("weights", [None, [0.2, 0.3, 0.5], [3.0, 1.0, 1.0]])
def test_predict_is_the_mean_of_the_members_fine_distributions(
    fz: F.Featurizer, batch: F.Batch, weights: list[float] | None
):
    nets = [net(fz, seed) for seed in (1, 2, 3)]
    ensemble = EN.EnsemblePredictor(
        [member(f"m{i}", made) for i, made in enumerate(nets)], fz, weights=weights
    )
    share = list(ensemble.weights)
    assert abs(sum(share) - 1.0) < 1e-15
    if weights is None:
        assert share == [1 / 3] * 3
    else:
        assert np.allclose(share, np.asarray(weights) / sum(weights), atol=1e-15)
    alone = [made.predict(batch) for made in nets]
    assert not equal(alone[0], alone[1])  # the members disagree
    made = ensemble.predict(batch)
    assert all(made[head].dtype == np.float64 for head in HEADS)
    want = mean_of_fines(alone, batch, share)
    assert np.abs(F.fine_probs(made, batch) - want).max() < 1e-12
    norms = [F.normalize_prediction(pred, batch) for pred in alone]
    for head in ("action", "mega"):
        mean = sum(w * norm[head] for w, norm in zip(share, norms))
        assert np.abs(F.normalize_prediction(made, batch)[head] - mean).max() < 1e-12
    # the pure function gives the same arrays
    again = EN.mix_predictions(alone, batch, weights)
    assert equal(made, again)
    assert not ensemble.counters or not any(
        name.startswith(EN.FAILURE) for name in ensemble.counters
    )


def test_the_result_is_a_proper_distribution(fz: F.Featurizer, batch: F.Batch):
    nets = [net(fz, seed) for seed in (4, 5)]
    ensemble = EN.EnsemblePredictor([member("a", nets[0]), member("b", nets[1])], fz)
    made = ensemble.predict(batch)
    norm = F.normalize_prediction(made, batch)
    for head in HEADS:
        assert np.abs(norm[head] - made[head]).max() < 1e-12
        assert (made[head] >= 0).all() and np.isfinite(made[head]).all()
    mask = np.asarray(batch["action_mask"]).astype(bool)
    legal = F.expand_target_mask(batch["cand_tmask"])
    assert not made["action"][~mask].any() and not made["target"][~legal].any()
    acting = mask.any(-1)
    sums = F.fine_probs(made, batch).sum(-1)
    assert acting.any() and np.abs(sums[acting] - 1.0).max() < 1e-12
    assert not sums[~acting].any()
    # the target array is the mixture's conditional, not the mean of the factors
    one, two = (F.normalize_prediction(n.predict(batch), batch) for n in nets)
    n_move = one["target"].shape[2]
    top = 0.5 * one["action"][:, :, :n_move, None] * one["target"]
    top += 0.5 * two["action"][:, :, :n_move, None] * two["target"]
    bottom = top.sum(-1, keepdims=True)
    factors = 0.5 * one["target"] + 0.5 * two["target"]
    want = np.divide(top, bottom, out=factors.copy(), where=bottom > 0)
    assert np.abs(made["target"] - want).max() < 1e-15
    assert np.abs(made["target"] - factors).max() > 1e-4
    # the runtime's scalars read it like any prediction
    assert (
        F.event_probs(made, batch)
        and F.intent_probs(made, batch, fz.tables) is not None
    )


def test_one_member_is_that_member_exactly(fz: F.Featurizer, batch: F.Batch):
    alone = net(fz, 6)
    ensemble = EN.EnsemblePredictor([member("only", alone)], fz, name="one")
    want, made = alone.predict(batch), ensemble.predict(batch)
    assert equal(want, made)
    assert all(made[head].dtype == want[head].dtype for head in HEADS)
    assert ensemble.weights == (1.0,) and ensemble.kind == EN.KIND
    assert ensemble.name == "one" and ensemble.labels == ("only",)
    assert ensemble.temperatures == alone.temperatures


def test_member_order_does_not_matter(fz: F.Featurizer, batch: F.Batch):
    one, two, three = (net(fz, seed) for seed in (7, 8, 9))
    first = EN.EnsemblePredictor([member("a", one), member("b", two)], fz)
    second = EN.EnsemblePredictor([member("b", two), member("a", one)], fz)
    assert equal(first.predict(batch), second.predict(batch))
    weighted = EN.EnsemblePredictor(
        [member("a", one), member("b", two)], fz, weights=[0.7, 0.3]
    )
    turned = EN.EnsemblePredictor(
        [member("b", two), member("a", one)], fz, weights=[0.3, 0.7]
    )
    assert equal(weighted.predict(batch), turned.predict(batch))
    assert not equal(weighted.predict(batch), first.predict(batch))
    # three members: the sums are taken in another order, so to rounding
    members = [member("a", one), member("b", two), member("c", three)]
    made = EN.EnsemblePredictor(members, fz).predict(batch)
    other = EN.EnsemblePredictor(members[::-1], fz).predict(batch)
    assert all(np.abs(made[head] - other[head]).max() < 1e-15 for head in HEADS)


def test_chunks_and_threads_change_nothing(fz: F.Featurizer, batch: F.Batch):
    # A network's float32 output depends on the rows it is batched with (in
    # the last bits), so a chunk is a whole number of the members' own
    # batches: then each member sees the batches it sees alone.
    members = [
        member("a", net(fz, 1, batch_size=2)),
        member("b", net(fz, 2, batch_size=2)),
    ]
    whole = EN.EnsemblePredictor(members, fz).predict(batch)
    assert EN.CHUNK % 1024 == 0 and EN.EnsemblePredictor(members, fz).chunk == EN.CHUNK
    small = EN.EnsemblePredictor(members, fz, chunk=4)
    assert batch["turn"].shape[0] > 3 * small.chunk
    assert equal(whole, small.predict(batch))
    odd = EN.EnsemblePredictor(members, fz, chunk=3).predict(batch)
    assert all(np.abs(odd[head] - whole[head]).max() < 1e-6 for head in HEADS)
    empty = small.predict(F.take(batch, slice(0, 0)))
    assert empty["action"].shape[0] == 0 and not small.counters
    import threading

    found: list[dict[str, np.ndarray]] = []
    threads = [
        threading.Thread(target=lambda: found.append(small.predict(batch)))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(found) == 4 and all(equal(whole, made) for made in found)


def test_what_the_runtime_and_the_tools_read_is_there(fz: F.Featurizer, fz2):
    blind, seeing = net(fz, 1, elo_mode=F.ELO_BLANK), net(fz, 2)
    mixed = EN.EnsemblePredictor([member("a", blind), member("b", seeing)], fz)
    assert isinstance(mixed, F.Predictor)
    assert mixed.elo_mode == F.ELO_KEEP and mixed.extra_keys == ()
    assert isinstance(mixed.counters, Counter) and not mixed.counters
    both = EN.EnsemblePredictor(
        [member("a", blind), member("b", net(fz, 3, elo_mode=F.ELO_BLANK))], fz
    )
    assert both.elo_mode == F.ELO_BLANK
    told = both.describe()
    json.dumps(told, allow_nan=False)
    assert told["kind"] == EN.KIND and set(told["members"]) == {"a", "b"}
    assert told["event_calibrated"] is False
    for head in ("action", "target"):
        mean = (
            0.5 * blind.temperatures[head]
            + 0.5 * both.members[1].predictor.temperatures[head]
        )
        assert told["temperatures"][head] == pytest.approx(mean, abs=1e-15)
    assert R.extras_read(mixed) == ()
    wide = EN.EnsemblePredictor(
        [member("a", net(fz2, 1)), member("b", matchup_net(fz2, 2))], fz2
    )
    assert wide.extra_keys == MU_NAMES and R.extras_read(wide) == MU_NAMES
    assert wide.n_parameters == sum(
        m.predictor.net.n_parameters() for m in wide.members
    )


def test_bad_members_and_weights_are_refused(fz: F.Featurizer):
    one, two = net(fz, 1), net(fz, 2)
    with pytest.raises(ValueError, match="at least one member"):
        EN.EnsemblePredictor([], fz)
    with pytest.raises(ValueError, match="repeat"):
        EN.EnsemblePredictor([member("a", one), member("a", two)], fz)
    with pytest.raises(ValueError, match="not nested"):
        EN.EnsemblePredictor([member("a", one, kind=EN.KIND)], fz)
    with pytest.raises(ValueError, match="label"):
        EN.EnsemblePredictor([member("a:b", one)], fz)
    for weights in ([1.0], [1.0, 0.0], [1.0, -1.0], [1.0, float("nan")], ["x", 1]):
        with pytest.raises(ValueError):
            EN.EnsemblePredictor(
                [member("a", one), member("b", two)], fz, weights=weights
            )


# --- one featurizer ---------------------------------------------------------------


def test_the_extra_keys_are_the_ones_a_version_2_payload_adds(fz, fz2):
    old, new = fz.to_payload(), fz2.to_payload()
    added = set(new) - set(old)
    assert added and all(EN.is_extra_key(key) for key in added)
    changed = {
        key for key in old if EN.payload_hash(old[key]) != EN.payload_hash(new[key])
    }
    assert changed == {"layout_version"}
    assert EN.payload_hash(EN.core_payload(old)) == EN.payload_hash(
        EN.core_payload(new)
    )
    assert set(EN.core_payload(old)) == set(old) - {"layout_version"}
    # what narrowed() drops is what the rule leaves out
    assert EN.payload_hash(fz2.narrowed(()).to_payload()) == EN.payload_hash(old)
    assert EN.payload_extras(new) == MU_NAMES and EN.payload_extras(old) == ()


def test_members_with_different_featurizers_are_refused(fz: F.Featurizer, fz2):
    other = F.Featurizer.build(F.Repertoire())
    with pytest.raises(ValueError, match="different featurizers") as raised:
        EN.shared_featurizer({"a": fz.to_payload(), "b": other.to_payload()})
    assert "repertoire" in str(raised.value)
    fewer = F.Featurizer.build(TRT.repertoire(), n_cand=8)
    with pytest.raises(ValueError, match="n_cand"):
        EN.shared_featurizer({"a": fz.to_payload(), "b": fewer.to_payload()})
    with pytest.raises(ValueError, match="different featurizers"):
        EN.from_loaded({"a": held(net(fz, 1), fz), "b": held(net(other, 2), other)})
    # identical payloads, and payloads that differ by the version-2 arrays only
    assert EN.shared_featurizer({"a": fz.to_payload(), "b": fz.to_payload()}) == (
        "a",
        {"a": (), "b": ()},
    )
    richest, extras = EN.shared_featurizer(
        {"a": fz.to_payload(), "b": fz2.to_payload()}
    )
    assert richest == "b" and extras == {"a": (), "b": MU_NAMES}
    part = fz2.narrowed(("mu_cand",))
    richest, extras = EN.shared_featurizer(
        {"a": part.to_payload(), "b": fz2.to_payload(), "c": fz.to_payload()}
    )
    assert richest == "b" and extras["a"] == ("mu_cand",) and extras["c"] == ()
    # no one member writes every array
    with pytest.raises(ValueError, match="no one featurizer"):
        EN.shared_featurizer(
            {"a": part.to_payload(), "b": fz2.narrowed(("mu_slot",)).to_payload()}
        )
    # the same arrays under other definitions
    changed = part.to_payload()
    changed["matchup"] = {**(changed["matchup"] or {}), "version": 999}
    with pytest.raises(ValueError, match="narrowed"):
        EN.shared_featurizer({"a": changed, "b": fz2.to_payload()})


def test_a_member_fed_the_richer_batch_gives_its_stand_alone_output(
    fz: F.Featurizer, fz2: F.Featurizer, batch: F.Batch, batch2: F.Batch, tmp_path: Path
):
    plain, reads = net(fz, 1), matchup_net(fz2, 2)
    alone = [plain.predict(batch), reads.predict(batch2)]
    built = EN.from_loaded({"plain": held(plain, fz), "reads": held(reads, fz2)})
    assert built.featurizer.layout_version == 2
    assert (
        built.featurizer.extras == MU_NAMES and built.extra["featurizer_of"] == "reads"
    )
    ensemble = built.predictor
    assert [m.extras for m in ensemble.members] == [(), MU_NAMES]
    assert ensemble.extra_keys == MU_NAMES
    # the ensemble's featurizer writes what each member's own did
    fed = TM.game_batch(built.featurizer)
    assert all(np.array_equal(fed[name], batch2[name]) for name in batch2)
    assert all(np.array_equal(fed[name], batch[name]) for name in batch)
    # each member inside the ensemble, on the ensemble's batch
    for position, want in enumerate(alone):
        got = ensemble.members[position].predictor.predict(fed)
        assert equal(want, got), ensemble.labels[position]
    made = ensemble.predict(fed)
    want = 0.5 * F.fine_probs(alone[0], batch) + 0.5 * F.fine_probs(alone[1], batch2)
    assert np.abs(F.fine_probs(made, fed) - want).max() < 1e-12
    # and through files: each member's stored payload, the richer featurizer
    paths = [("plain", save_net(tmp_path / "plain.pt", plain, fz))]
    paths += [("reads", save_net(tmp_path / "reads.pt", reads, fz2))]
    stored = EN.from_artifacts(paths)
    assert equal(stored.predictor.predict(fed), made)
    assert stored.featurizer.extras == MU_NAMES
    target = tmp_path / "both.pt"
    A.save_artifact(
        target,
        kind=EN.KIND,
        name="both",
        featurizer=stored.featurizer,
        predictor_payload=stored.predictor.to_payload(),
        extra=stored.extra,
    )
    loaded = A.load_predictor(target)
    assert loaded.featurizer.extras == MU_NAMES
    assert equal(loaded.predictor.predict(fed), made)
    assert R.OpponentPredictor.load(target).extras_read == MU_NAMES
    # a member whose arrays the stored featurizer does not write is refused
    payload = stored.predictor.to_payload()
    with pytest.raises(ValueError, match="does not write"):
        EN.from_payload(payload, fz)


# --- a coupled member ---------------------------------------------------------------


def test_a_coupled_member_is_refused(fz: F.Featurizer, tmp_path: Path):
    one, two = net(fz, 1), net(fz, 2)
    coupling = CP.PairCoupling.build(
        some_table(3), name="unit", fitted_after=CP.fitted_after(M.KIND, two.describe())
    )
    plain = save_net(tmp_path / "plain.pt", one, fz)
    coupled = save_net(tmp_path / "coupled.pt", two, fz, coupling=coupling)
    assert A.load_predictor(coupled).coupling is not None
    with pytest.raises(ValueError, match="pair coupling"):
        EN.from_artifacts([("a", plain), ("b", coupled)])
    with pytest.raises(ValueError, match="pair coupling"):
        EN.from_loaded({"a": A.load_predictor(plain), "b": A.load_predictor(coupled)})
    # the same member twice, a nested ensemble, a repeated label
    with pytest.raises(ValueError, match="same file"):
        EN.from_artifacts([("a", plain), ("b", plain)])
    with pytest.raises(ValueError, match="twice"):
        EN.from_artifacts([("a", plain), ("a", coupled)])
    with pytest.raises((OSError, ValueError)):
        EN.from_artifacts([("a", plain), ("b", tmp_path / "nothing.pt")])


# --- a failing member ---------------------------------------------------------------


class Fake:
    """A member made of a function; it counts like a real predictor."""

    kind = M.KIND
    name = "fake"
    elo_mode = F.ELO_KEEP

    def __init__(self, make: Any) -> None:
        self.make = make
        self.counters: Counter[str] = Counter()

    def predict(self, batch: F.Batch) -> Any:
        return self.make(self, batch)


def _raises(fake: Fake, batch: F.Batch) -> Any:
    raise RuntimeError("no")


def _degrades(fake: Fake, batch: F.Batch) -> Any:
    fake.counters["predict_error:ValueError"] += 1
    return F.uniform_prediction(batch)


def _no_mega(fake: Fake, batch: F.Batch) -> Any:
    made = F.uniform_prediction(batch)
    return {"action": made["action"], "target": made["target"]}


def _wrong_shape(fake: Fake, batch: F.Batch) -> Any:
    made = F.uniform_prediction(batch)
    return {**made, "action": made["action"][:, :, :-1]}


def _not_finite(fake: Fake, batch: F.Batch) -> Any:
    made = F.uniform_prediction(batch)
    return {**made, "mega": np.full_like(made["mega"], np.nan)}


def _no_target(fake: Fake, batch: F.Batch) -> Any:
    made = F.uniform_prediction(batch)
    return {**made, "target": np.zeros_like(made["target"])}


@pytest.mark.parametrize(
    "make, counter",
    [
        (_raises, EN.MEMBER_RAISED),
        (_degrades, EN.MEMBER_DEGRADED),
        (_no_mega, EN.MEMBER_MALFORMED),
        (_wrong_shape, EN.MEMBER_MALFORMED),
        (_not_finite, EN.MEMBER_MALFORMED),
        (_no_target, EN.MEMBER_IMPROPER),
    ],
)
@pytest.mark.parametrize("bad_first", [False, True])
def test_a_failing_member_gives_the_failure_value_and_a_counter(
    fz: F.Featurizer, batch: F.Batch, make: Any, counter: str, bad_first: bool
):
    good, bad = net(fz, 1), Fake(make)
    members = [member("good", good), member("bad", bad)]
    ensemble = EN.EnsemblePredictor(members[::-1] if bad_first else members, fz)
    made = ensemble.predict(batch)
    # the failure value of every predictor, never the member that is left
    assert equal(made, F.uniform_prediction(batch))
    assert not equal(made, good.predict(batch))
    assert ensemble.counters[counter + "bad"] == 1
    assert EN.predictor_failures(ensemble) == 1 and SC.predict_errors(ensemble) == 1
    if make is _degrades:
        assert ensemble.counters["member:bad:predict_error:ValueError"] == 1
    # the tools take it for a failure: no forecast, no score
    with pytest.raises(SC.ScorecardError, match="fell back"):
        SC.guarded_predict(ensemble, batch, name="ens")
    rt = R.OpponentPredictor(ensemble, fz)
    live = TRT.forecasts(rt, "p1")
    assert set(live.values()) == {None}
    assert rt.counters[R.FAILURE + "predictor_degraded"] >= 1
    # it is the call that failed, not the ensemble
    bad.make = lambda fake, rows: good.predict(rows)
    before = EN.predictor_failures(ensemble)
    again = ensemble.predict(batch)
    assert EN.predictor_failures(ensemble) == before
    assert (
        np.abs(
            F.fine_probs(again, batch) - F.fine_probs(good.predict(batch), batch)
        ).max()
        < 1e-12
    )


def test_one_failing_member_alone_and_a_broken_batch_never_raise(
    fz: F.Featurizer, batch: F.Batch
):
    alone = EN.EnsemblePredictor([member("bad", Fake(_raises))], fz)
    assert equal(alone.predict(batch), F.uniform_prediction(batch))
    assert alone.counters[EN.MEMBER_RAISED + "bad"] == 1
    ensemble = EN.EnsemblePredictor(
        [member("a", net(fz, 1)), member("b", net(fz, 2))], fz
    )
    # real networks on a batch without an array they read: each falls back
    broken = {name: value for name, value in batch.items() if name != "cand_move"}
    made = ensemble.predict(broken)
    assert equal(made, F.uniform_prediction(broken))
    assert ensemble.counters[EN.MEMBER_DEGRADED + "a"] == 1
    assert ensemble.counters["member:a:predict_error:KeyError"] == 1
    assert EN.MEMBER_DEGRADED + "b" not in ensemble.counters  # stopped at the first
    # nothing to stand on at all
    assert ensemble.predict({}) == {}
    assert ensemble.counters["predict_error:KeyError"] == 1
    assert ensemble.member_counters()["a"]["predict_error:KeyError"] == 1
    assert equal(
        ensemble.predict(batch),
        EN.mix_predictions(
            [m.predictor.predict(batch) for m in ensemble.members], batch
        ),
    )


def test_mix_predictions_refuses_what_is_not_a_distribution(fz: F.Featurizer, batch):
    good = net(fz, 1).predict(batch)
    with pytest.raises(ValueError, match="no legal target"):
        EN.mix_predictions([good, _no_target(Fake(_no_target), batch)], batch)
    with pytest.raises(ValueError):
        EN.mix_predictions([good, _wrong_shape(Fake(_wrong_shape), batch)], batch)
    with pytest.raises(ValueError, match="non-finite"):
        EN.mix_predictions([good, _not_finite(Fake(_not_finite), batch)], batch)
    with pytest.raises(ValueError):
        EN.mix_predictions([], batch)


# --- the artifact -----------------------------------------------------------------


def two_files(fz: F.Featurizer, tmp_path: Path) -> list[tuple[str, Path]]:
    extra = {
        "args": {"tag": "run_one"},
        "dataset": {"tag": "unit", "manifest_sha256": "abc", "formats": [PLAYED]},
    }
    first = save_net(tmp_path / "one.pt", net(fz, 1), fz, extra=extra)
    second = save_net(
        tmp_path / "two.pt",
        net(fz, 2, elo_mode=F.ELO_BLANK),
        fz,
        extra={**extra, "args": {}},
    )
    return [("base", first), ("wide", second)]


def test_artifact_round_trip(fz: F.Featurizer, batch: F.Batch, tmp_path: Path):
    sources = two_files(fz, tmp_path)
    built = EN.from_artifacts(sources, weights=[2.0, 1.0], name="unit_ens")
    want = built.predictor.predict(batch)
    alone = [A.load_predictor(path).predictor.predict(batch) for _, path in sources]
    fine = mean_of_fines(alone, batch, [2 / 3, 1 / 3])
    assert np.abs(F.fine_probs(want, batch) - fine).max() < 1e-12
    path = tmp_path / "ensemble.pt"
    A.save_artifact(
        path,
        kind=A.KIND_ENSEMBLE,
        name="unit_ens",
        featurizer=built.featurizer,
        predictor_payload=built.predictor.to_payload(),
        extra=built.extra,
    )
    document = A.read_artifact(path)
    assert (
        document["version"] == A.VERSION and document["kind"] == EN.KIND == "ensemble"
    )
    stored = document["predictor"]
    assert stored["format"] == EN.PAYLOAD_FORMAT and stored["combine"] == EN.COMBINE
    assert stored["weights"] == [2 / 3, 1 / 3]
    for entry, (label, source) in zip(stored["members"], sources):
        own = A.read_artifact(source)
        # the member's own payload, unchanged: weights, temperatures, all of it
        assert MK.same(entry["payload"], own["predictor"])
        assert entry["label"] == label and entry["kind"] == A.KIND_OPPNET
        assert entry["sha256"] == EN.sha256_file(source) and entry["path"] == str(
            source
        )
        assert entry["name"] == own["name"] and entry["extras"] == []
    assert [entry["tag"] for entry in stored["members"]] == ["run_one", "net2"]
    assert EN.payload_hash(document["featurizer"]) == EN.payload_hash(
        A.read_artifact(sources[0][1])["featurizer"]
    )
    assert document["extra"]["dataset"] == {
        "tag": "unit",
        "manifest_sha256": "abc",
        "formats": [PLAYED],
    }
    assert SC._trained_on(document["extra"]) == "abc"

    loaded = A.load_predictor(path)
    assert isinstance(loaded.predictor, EN.EnsemblePredictor)
    assert (loaded.kind, loaded.name, loaded.coupling) == (EN.KIND, "unit_ens", None)
    assert loaded.predictor.labels == ("base", "wide")
    assert loaded.predictor.weights == (2 / 3, 1 / 3)
    assert loaded.predictor.elo_mode == F.ELO_KEEP  # one member reads ratings
    assert equal(loaded.predictor.predict(batch), want)
    assert A.try_load_predictor(path) is not None
    # saving what was loaded gives the same payload again
    assert MK.same(
        A._decode(A._encode(loaded.predictor.to_payload(), "p")), document["predictor"]
    )

    # the runtime serves it; the forecast's dict has the keys it always had
    rt = R.OpponentPredictor.load(path, keep_features=True)
    assert rt.loaded and rt.serving and (rt.name, rt.kind) == ("unit_ens", EN.KIND)
    assert rt._formats == frozenset({PLAYED})
    served = 0
    for forecast in TRT.forecasts(rt, "p1").values():
        if forecast is None:
            continue
        served += 1
        assert forecast.features is not None
        offline = loaded.predictor.predict(forecast.features)
        for head in HEADS:
            assert np.array_equal(forecast.raw[head], offline[head][0])
        told = forecast.to_dict()
        assert set(told) == FORECAST_KEYS and told["kind"] == EN.KIND
        json.dumps(told, allow_nan=False)
    assert served >= 3 and not rt.diagnostics()["predictor"]
    # the members' formats are the ensemble's: another game is not served
    elsewhere = TRT.LOG.replace(
        "|tier|[Gen 9 Champions] VGC 2026 Reg M-C", "|tier|[Gen 9] Doubles OU"
    )
    assert elsewhere != TRT.LOG
    away = R.OpponentPredictor.load(path)
    assert set(TRT.forecasts(away, "p1", elsewhere).values()) == {None}
    assert away.counters["stand_down:format"] >= 1

    # damaged payloads are a ValueError, never a half-built ensemble
    for change in (
        {"format": "other"},
        {"version": 2},
        {"combine": "median"},
        {"weights": [1.0]},
        {"members": []},
        {"members": [{**stored["members"][0], "kind": EN.KIND}]},
        {"members": [{**stored["members"][0], "payload": {"format": "no"}}]},
    ):
        with pytest.raises(ValueError):
            EN.from_payload({**stored, **change}, loaded.featurizer)
    with pytest.raises(ValueError):
        EN.from_payload({"format": EN.PAYLOAD_FORMAT}, loaded.featurizer)


def test_a_coupling_is_fitted_on_the_ensemble_and_bound_to_its_state(
    fz: F.Featurizer, batch: F.Batch, tmp_path: Path
):
    built = EN.from_artifacts(two_files(fz, tmp_path), name="unit_ens")
    described = built.predictor.describe()
    coupling = CP.PairCoupling.build(
        some_table(5), name="on_ens", fitted_after=CP.fitted_after(EN.KIND, described)
    )
    assert coupling.fitted_after["kind"] == EN.KIND
    path = tmp_path / "coupled.pt"
    payload = built.predictor.to_payload()

    def write(target: Path, pair: CP.PairCoupling, **changes: Any) -> Path:
        A.save_artifact(
            target,
            kind=EN.KIND,
            name="unit_ens_cpl",
            featurizer=built.featurizer,
            predictor_payload={**payload, **changes},
            extra=built.extra,
            coupling=pair,
        )
        return target

    loaded = A.load_predictor(write(path, coupling))
    assert A.read_artifact(path)["version"] == A.VERSION_COUPLED
    assert loaded.coupling is not None and loaded.coupling.same_as(coupling)
    assert equal(loaded.predictor.predict(batch), built.predictor.predict(batch))
    rt = R.OpponentPredictor.load(path)
    assert rt.serving and rt.coupling is not None
    # other weights are another predictor state: the coupling no longer fits
    with pytest.raises(ValueError, match="does not fit"):
        A.load_predictor(write(tmp_path / "moved.pt", coupling, weights=[0.9, 0.1]))
    # and one fitted after a member alone does not fit the ensemble
    member_state = built.predictor.members[0].predictor.describe()
    alone = CP.PairCoupling.build(
        some_table(5), name="on_one", fitted_after=CP.fitted_after(M.KIND, member_state)
    )
    with pytest.raises(ValueError, match="does not fit"):
        A.load_predictor(write(tmp_path / "wrong.pt", alone))


def coupled_file(
    path: Path, built: EN.Built, pair: CP.PairCoupling, **changes: Any
) -> Path:
    """``built``'s ensemble with a coupling, its payload changed by ``changes``."""
    A.save_artifact(
        path,
        kind=EN.KIND,
        name="e_cpl",
        featurizer=built.featurizer,
        predictor_payload={**built.predictor.to_payload(), **changes},
        extra=built.extra,
        coupling=pair,
    )
    return path


def test_a_coupling_is_bound_to_the_weights_and_to_each_members_payload(
    fz: F.Featurizer, batch: F.Batch, tmp_path: Path
):
    # Members with EQUAL temperatures: the mean temperature is the same for
    # every weighting, so only the state binds the weights and the members.
    flat = {"action_temperature": 1.0, "target_temperature": 1.0}
    one, two, three = (net(fz, seed, **flat) for seed in (1, 2, 9))
    sources = [
        ("a", save_net(tmp_path / "a.pt", one, fz)),
        ("b", save_net(tmp_path / "b.pt", two, fz)),
    ]
    built = EN.from_artifacts(sources, name="e")
    described = built.predictor.describe()
    after = CP.fitted_after(EN.KIND, described)
    assert after["temperatures"] == {"action": 1.0, "target": 1.0}
    coupling = CP.PairCoupling.build(some_table(5), name="on_e", fitted_after=after)
    want = built.predictor.predict(batch)

    # the state it was fitted on: loaded and served
    loaded = A.load_predictor(coupled_file(tmp_path / "same.pt", built, coupling))
    assert loaded.coupling is not None and loaded.coupling.same_as(coupling)
    assert equal(loaded.predictor.predict(batch), want)
    rt = R.OpponentPredictor.load(tmp_path / "same.pt")
    assert rt.serving and rt.coupling is not None

    # other weights: refused at load, and the runtime does not serve the file
    moved = coupled_file(tmp_path / "moved.pt", built, coupling, weights=[0.9, 0.1])
    with pytest.raises(ValueError, match="does not fit.*predictor state"):
        A.load_predictor(moved)
    assert not R.OpponentPredictor.load(moved).loaded
    # another network in a member's place. The recorded sha256 of the member's
    # file is left as it was: the state is read from the payload, not from it.
    members = list(built.predictor.to_payload()["members"])
    swapped = [members[0], {**members[1], "payload": three.to_payload()}]
    assert swapped[1]["sha256"] == members[1]["sha256"] != ""
    with pytest.raises(ValueError, match="does not fit.*predictor state"):
        A.load_predictor(
            coupled_file(tmp_path / "swapped.pt", built, coupling, members=swapped)
        )

    # what the record holds: the state, beside what it always held
    state = described[EN.STATE_KEY]
    assert EN.STATE_KEY == CP.KEY_STATE == "state_sha256"
    assert isinstance(state, str) and len(state) == 64
    assert state == built.predictor.state_sha256 == loaded.predictor.state_sha256
    assert after == {
        "kind": EN.KIND,
        "temperatures": {"action": 1.0, "target": 1.0},
        "event_calibrated": False,
        CP.KEY_STATE: state,
    }
    # weights that agree to rounding are the same state
    near = coupled_file(
        tmp_path / "near.pt", built, coupling, weights=[0.5, 0.5 + 1e-15]
    )
    assert A.load_predictor(near).coupling is not None
    # the same member with another temperature, a member less
    hot = {**members[1], "payload": net(fz, 2).to_payload()}
    for name, change in (
        ("hot", {"members": [members[0], hot]}),
        ("less", {"members": members[:1], "weights": [1.0]}),
    ):
        with pytest.raises(ValueError, match="does not fit"):
            A.load_predictor(
                coupled_file(tmp_path / f"{name}.pt", built, coupling, **change)
            )

    # a loaded coupled artifact keeps its coupling only on its own state
    other = EN.from_artifacts(sources, weights=[0.9, 0.1], name="e").predictor
    assert other.state_sha256 != state
    with pytest.raises(ValueError, match="does not fit"):
        loaded._replace(predictor=other)
    again = EN.from_artifacts(sources, name="another_name").predictor
    assert again.state_sha256 == state  # names and labels are not the state
    assert loaded._replace(predictor=again).coupling is loaded.coupling

    # a record that names no state (a coupling fitted before the state was
    # recorded) is read as before: bound by kind and temperatures alone
    legacy = {key: value for key, value in after.items() if key != CP.KEY_STATE}
    weak = CP.PairCoupling.build(some_table(5), name="old", fitted_after=legacy)
    assert A.load_predictor(
        coupled_file(tmp_path / "legacy.pt", built, weak, weights=[0.9, 0.1])
    ).coupling.same_as(weak)

    # a single network's record is what it always was, and a record that
    # names a state does not fit a predictor that names none
    assert set(CP.fitted_after(M.KIND, one.describe())) == {
        "kind",
        "temperatures",
        "event_calibrated",
    }
    bound = CP.PairCoupling.build(
        some_table(5), name="x", fitted_after={"kind": M.KIND, CP.KEY_STATE: state}
    )
    assert "predictor state" in bound.fits(M.KIND, one.describe())
    assert "predictor state" in bound.fits(M.KIND, None)
    assert coupling.fits(EN.KIND, described) == ""


def test_a_coupling_on_count_tables_is_bound_to_the_weights(
    fz: F.Featurizer, batch: F.Batch, tmp_path: Path
):
    # Count tables have no temperatures at all: every weighting reads 1.0.
    sources = []
    for position, rows in enumerate((slice(0, None, 2), slice(1, None, 2))):
        table = T.FlagsTable.fit(F.take(batch, rows), featurizer=fz)
        target = tmp_path / f"t{position}.pt"
        A.save_artifact(
            target,
            kind=A.KIND_TABLE,
            name=f"t{position}",
            featurizer=fz,
            predictor_payload=table.to_payload(),
        )
        sources.append((f"t{position}", target))
    built = EN.from_artifacts(sources, name="tables")
    after = CP.fitted_after(EN.KIND, built.predictor.describe())
    assert after["temperatures"] == {"action": 1.0, "target": 1.0}
    coupling = CP.PairCoupling.build(some_table(4), name="on_t", fitted_after=after)
    assert (
        A.load_predictor(coupled_file(tmp_path / "t.pt", built, coupling)).coupling
        is not None
    )
    with pytest.raises(ValueError, match="does not fit.*predictor state"):
        A.load_predictor(
            coupled_file(tmp_path / "tw.pt", built, coupling, weights=[0.9, 0.1])
        )
    assert after[CP.KEY_STATE] == built.predictor.state_sha256
    turned = EN.from_artifacts(sources[::-1], name="tables").predictor
    assert turned.state_sha256 != built.predictor.state_sha256  # the stored order


def test_payload_hash_reads_every_number_and_survives_the_file(
    fz: F.Featurizer, batch: F.Batch, tmp_path: Path
):
    import torch

    big = torch.zeros(4000)
    other = big.clone()
    other[2000] = 1e-3
    assert repr(big) == repr(other)  # what a hash of the repr would compare
    assert EN.payload_hash({"w": big}) != EN.payload_hash({"w": other})
    assert EN.payload_hash(big) == EN.payload_hash(big.numpy())
    assert EN.payload_hash(big) != EN.payload_hash(big.double())
    assert EN.payload_hash(big) != EN.payload_hash(big.reshape(40, 100))
    # a member's payload in memory and the same payload read back from a file
    alone = net(fz, 4)
    path = save_net(tmp_path / "n.pt", alone, fz)
    stored = A.read_artifact(path)["predictor"]
    assert EN.payload_hash(alone.to_payload()) == EN.payload_hash(stored)
    assert EN.payload_hash(net(fz, 5).to_payload()) != EN.payload_hash(stored)
    table = T.FlagsTable.fit(batch, featurizer=fz)
    target = tmp_path / "t.pt"
    A.save_artifact(
        target,
        kind=A.KIND_TABLE,
        name="t",
        featurizer=fz,
        predictor_payload=table.to_payload(),
    )
    assert EN.payload_hash(table.to_payload()) == EN.payload_hash(
        A.read_artifact(target)["predictor"]
    )
    # what the artifact stores in another type hashes as the stored value
    odd = {
        "set": {3, 1, 2},
        "path": Path("a") / "b",
        "float": np.float64(0.5),
        "text": np.str_("x"),
        "tuple": (1, 2.0, True, None),
        "array": np.arange(6, dtype=np.uint16).reshape(2, 3),
    }
    assert EN.payload_hash(odd) == EN.payload_hash(A._decode(A._encode(odd, "odd")))
    # so the state of an ensemble built in memory is its state after the file
    built = EN.from_loaded({"a": held(net(fz, 1), fz), "b": held(net(fz, 2), fz)})
    state = built.predictor.state_sha256
    direct = EN.EnsemblePredictor(
        [member("a", net(fz, 1)), member("b", net(fz, 2))], fz
    )
    assert direct.members[0].payload is None and direct.state_sha256 == state
    file = tmp_path / "e.pt"
    A.save_artifact(
        file,
        kind=EN.KIND,
        name="e",
        featurizer=built.featurizer,
        predictor_payload=built.predictor.to_payload(),
    )
    loaded = A.load_predictor(file).predictor
    assert loaded.state_sha256 == state and not loaded.counters
    told = loaded.describe()
    assert [told["members"][label]["payload_sha256"] for label in ("a", "b")] == list(
        loaded.member_states() or ()
    )
    # a member whose payload cannot be made has no state, and nothing raises
    broken = EN.EnsemblePredictor([member("f", Fake(_raises))], fz)
    assert broken.state_sha256 is None and broken.describe()[EN.STATE_KEY] is None
    assert broken.counters[f"{EN.STATE_UNREAD}:AttributeError"] >= 1
    assert EN.predictor_failures(broken) == 0  # not a failed prediction
    assert EN.STATE_KEY not in CP.fitted_after(EN.KIND, broken.describe())


def test_a_reader_from_before_the_kind_refuses_the_file(
    fz: F.Featurizer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    built = EN.from_artifacts(two_files(fz, tmp_path))
    path = tmp_path / "ensemble.pt"
    A.save_artifact(
        path,
        kind=EN.KIND,
        name="e",
        featurizer=built.featurizer,
        predictor_payload=built.predictor.to_payload(),
    )
    old_kinds = tuple(kind for kind in A.KINDS if kind != EN.KIND)
    assert old_kinds == (A.KIND_TABLE, A.KIND_OPPNET)
    monkeypatch.setattr(A, "KINDS", old_kinds)
    with pytest.raises(ValueError, match="unknown kind 'ensemble'"):
        A.read_artifact(path)
    with pytest.raises(ValueError, match="unknown kind 'ensemble'"):
        A.load_predictor(path)
    rt = R.OpponentPredictor.load(path)
    assert not rt.loaded and "unknown kind" in str(rt.load_failure)
    with pytest.raises(ValueError):
        A.save_artifact(
            tmp_path / "no.pt",
            kind=EN.KIND,
            name="e",
            featurizer=fz,
            predictor_payload={},
        )


def test_old_kinds_are_unaffected(fz: F.Featurizer, batch: F.Batch, tmp_path: Path):
    assert A.KINDS[:2] == (A.KIND_TABLE, A.KIND_OPPNET) == ("table", "oppnet")
    assert A._MODULES[A.KIND_TABLE] == EN.MEMBER_MODULES["table"]
    assert A._MODULES[A.KIND_OPPNET] == EN.MEMBER_MODULES["oppnet"]
    assert EN.MEMBER_KINDS == ("table", "oppnet") and EN.KIND not in EN.MEMBER_KINDS
    keys = {
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
    alone = net(fz, 4)
    path = save_net(tmp_path / "net.pt", alone, fz)
    document = A.read_artifact(path)
    assert set(document) == keys and document["version"] == A.VERSION == 1
    assert document["kind"] == A.KIND_OPPNET
    loaded = A.load_predictor(path)
    assert isinstance(loaded.predictor, M.OppNetPredictor) and loaded.coupling is None
    assert equal(loaded.predictor.predict(batch), alone.predict(batch))
    table = T.FlagsTable.fit(batch, featurizer=fz)
    target = tmp_path / "table.pt"
    A.save_artifact(
        target,
        kind=A.KIND_TABLE,
        name="t",
        featurizer=fz,
        predictor_payload=table.to_payload(),
    )
    document = A.read_artifact(target)
    assert set(document) == keys and document["kind"] == A.KIND_TABLE
    features = SC.features_only(batch)
    assert equal(
        A.load_predictor(target).predictor.predict(features), table.predict(features)
    )
    coupling = CP.PairCoupling.build(
        some_table(2), name="c", fitted_after=CP.fitted_after(M.KIND, alone.describe())
    )
    coupled = A.load_predictor(
        save_net(tmp_path / "c.pt", alone, fz, coupling=coupling)
    )
    assert A.read_artifact(tmp_path / "c.pt")["version"] == A.VERSION_COUPLED == 2
    assert coupled.coupling is not None and coupled.kind == A.KIND_OPPNET
    assert equal(coupled.predictor.predict(batch), alone.predict(batch))
    # a count table can be a member too: the same contract
    mixed = EN.from_artifacts([("net", path), ("table", target)])
    made = mixed.predictor.predict(features)
    want = 0.5 * F.fine_probs(alone.predict(features), batch)
    want += 0.5 * F.fine_probs(table.predict(features), batch)
    assert np.abs(F.fine_probs(made, batch) - want).max() < 1e-12


# --- fine-tuned members -------------------------------------------------------------


def tuned_extra(source: dict[str, Any], name: str, games: list[str]) -> dict[str, Any]:
    """The ``extra`` the fine-tune script writes for an artifact fitted on
    ``games`` (its own function, so the record is the real one)."""
    from training import finetune_oppmodel as FT

    return FT.finetuned_extra(
        source, name=name, record={"created": "2026-10-10", "seen_games": games}
    )


def test_the_games_a_fine_tuned_member_has_seen_are_the_ensembles_record(
    fz: F.Featurizer, tmp_path: Path
):
    from training import finetune_oppmodel as FT

    source = {
        "args": {"tag": "run"},
        "dataset": {"tag": "unit", "manifest_sha256": "abc", "formats": [PLAYED]},
    }
    first, second = ["g1", "g2", "g3"], ["g3", "g4"]
    plain = save_net(tmp_path / "plain.pt", net(fz, 1), fz, extra=source)
    one = save_net(
        tmp_path / "one.pt", net(fz, 2), fz, extra=tuned_extra(source, "n2_ft", first)
    )
    two = save_net(
        tmp_path / "two.pt", net(fz, 3), fz, extra=tuned_extra(source, "n3_ft", second)
    )
    assert FT.seen_games(A.read_artifact(one)["extra"]) == set(first)

    # a plain member and a fine-tuned one: the ensemble has seen its games,
    # and its manifest is not the build's (nor missing)
    mixed = EN.from_artifacts([("base", plain), ("tuned", one)], name="mixed")
    marked = f"{FT.MANIFEST_MARK}:3-games:abc"
    assert FT.seen_games(mixed.extra) == set(first)
    assert SC._trained_on(mixed.extra) == marked != "abc"
    assert (EN.KEY_FINETUNE, EN.KEY_SEEN) == (FT.KEY_FINETUNE, FT.KEY_SEEN)
    assert EN.MANIFEST_MARK == FT.MANIFEST_MARK

    # no fine-tuned member: the record of before, and no fine-tune keys
    never = EN.from_artifacts(
        [
            ("a", plain),
            ("b", save_net(tmp_path / "p2.pt", net(fz, 4), fz, extra=source)),
        ]
    )
    assert EN.KEY_FINETUNE not in never.extra and "dataset_tag" not in never.extra
    assert SC._trained_on(never.extra) == "abc" and FT.seen_games(never.extra) == set()
    assert [m["finetuned_on_own_games"] for m in never.extra["members"].values()] == [
        0,
        0,
    ]

    assert mixed.extra["dataset"]["source_manifest_sha256"] == "abc"
    assert mixed.extra["dataset"]["formats"] == [PLAYED]
    assert "3 own ladder games" in mixed.extra["dataset_tag"]
    record = mixed.extra[EN.KEY_FINETUNE]
    assert record["finetuned_members"] == ["tuned"] and set(record["members"]) == {
        "tuned"
    }
    # the member's own record, whole
    assert record["members"]["tuned"] == A.read_artifact(one)["extra"][EN.KEY_FINETUNE]
    assert mixed.extra["members"]["base"]["finetuned_on_own_games"] == 0
    assert mixed.extra["members"]["tuned"]["finetuned_on_own_games"] == 3
    FT.check_unseen(mixed.extra, ["another-game"])
    with pytest.raises(FT.FinetuneError, match="in-sample"):
        FT.check_unseen(mixed.extra, ["another-game", "g2"])

    # two fine-tuned members: the union of their games
    both = EN.from_artifacts([("x", one), ("y", two)])
    assert FT.seen_games(both.extra) == {"g1", "g2", "g3", "g4"}
    assert SC._trained_on(both.extra) == f"{FT.MANIFEST_MARK}:4-games:abc"
    assert both.extra[EN.KEY_FINETUNE]["finetuned_members"] == ["x", "y"]

    # the record is in the file, the runtime still knows the formats, and
    # members already in memory give the same record
    path = tmp_path / "mixed.pt"
    A.save_artifact(
        path,
        kind=EN.KIND,
        name="mixed",
        featurizer=mixed.featurizer,
        predictor_payload=mixed.predictor.to_payload(),
        extra=mixed.extra,
    )
    loaded = A.load_predictor(path)
    assert FT.seen_games(loaded.meta["extra"]) == set(first)
    assert SC._trained_on(loaded.meta["extra"]) == marked
    assert R.OpponentPredictor.load(path)._formats == frozenset({PLAYED})
    again = EN.from_loaded(
        {"base": A.load_predictor(plain), "tuned": A.load_predictor(one)}
    )
    assert FT.seen_games(again.extra) == set(first)
    assert SC._trained_on(again.extra) == marked

    # members of different builds: marked all the same, no one build named
    elsewhere = {**source, "dataset": {**source["dataset"], "manifest_sha256": "xyz"}}
    far = save_net(tmp_path / "far.pt", net(fz, 5), fz, extra=elsewhere)
    apart = EN.from_artifacts([("far", far), ("tuned", one)])
    assert SC._trained_on(apart.extra) == f"{FT.MANIFEST_MARK}:3-games:several-builds"
    assert apart.extra["dataset"]["source_manifest_sha256"] is None
    assert FT.seen_games(apart.extra) == set(first)

    # a fine-tuned member that cannot name its games is refused
    silent = dict(tuned_extra(source, "n6_ft", first))
    silent[EN.KEY_FINETUNE] = {"created": "2026-10-10", "domain_games": 3}
    nameless = save_net(tmp_path / "nameless.pt", net(fz, 6), fz, extra=silent)
    with pytest.raises(ValueError, match="does not list them"):
        EN.from_artifacts([("base", plain), ("tuned", nameless)])
    stripped = {key: value for key, value in silent.items() if key != EN.KEY_FINETUNE}
    bare = save_net(tmp_path / "bare.pt", net(fz, 7), fz, extra=stripped)
    with pytest.raises(ValueError, match="does not list them"):
        EN.from_artifacts([("base", plain), ("tuned", bare)])


# --- the script -------------------------------------------------------------------


def tiny_dataset(fz: F.Featurizer, data: Path, times: bool = True) -> str:
    """A dataset of five parts (train, val, old and sealed ladder rows, test)
    under ``data``; the sha256 of its manifest. ``times=False`` stores the
    rows without ``m_time`` (a build from before the rows carried times)."""
    data.mkdir()
    cut = F.local_time(F.OWN_SEALED_FROM)
    parts = [
        TR.with_meta(TR.paired_rows(fz, 80, 10), "train", 0, cut - 900_000),
        TR.with_meta(
            TR.paired_rows(fz, 240, 11, one_target=40), "val", 100, cut - 800_000
        ),
        TR.with_meta(TR.paired_rows(fz, 48, 12), "ladder_holdout", 300, cut - 9),
        TR.with_meta(TR.paired_rows(fz, 32, 13), "ladder_holdout", 400, cut + 60),
        TR.with_meta(TR.paired_rows(fz, 24, 14), "test", 500, cut - 700_000),
    ]
    rows = F.concat_batches(parts)
    if not times:
        rows = {name: value for name, value in rows.items() if name != "m_time"}
    F.save_batch(data / "shard-00000.npz", rows)
    manifest = {
        "tag": "unit",
        "splits": TR.SPLITS,
        "shards": [{"file": "shard-00000.npz", "examples": 424}],
    }
    assert "test" in TR.SPLITS
    (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    fz.save(data)
    return MK.sha256_file(data / "manifest.json")


def test_the_script_does_not_score_ladder_rows_a_member_was_fine_tuned_on(
    fz: F.Featurizer, tmp_path: Path, capsys
):
    from training import finetune_oppmodel as FT

    data = tmp_path / "data"
    sha = tiny_dataset(fz, data)
    extra = {"dataset": {"tag": "unit", "manifest_sha256": sha}, "n_parameters": 1}
    games = ["old-1", "old-2"]
    one = save_net(tmp_path / "one.pt", net(fz, 1), fz, extra=extra)
    two = save_net(
        tmp_path / "two.pt", net(fz, 2), fz, extra=tuned_extra(extra, "two_ft", games)
    )
    argv = ["--member", f"base={one}", "--member", f"tuned={two}"]
    argv += ["--dataset", str(data), "--resamples", "40", "--check-rows", "100"]
    # by its record the fine-tuned member is not fitted on the build: refused
    assert MK.main([*argv, "--out", str(tmp_path / "no")]) == 1
    assert "not fitted on this dataset" in capsys.readouterr().out
    assert not (tmp_path / "no").exists()
    out = tmp_path / "out"
    assert MK.main([*argv, "--out", str(out), "--allow-other-dataset"]) == 0
    said = capsys.readouterr().out
    assert said.strip().splitlines()[-1] == MK.DONE
    report = json.loads((out / MK.REPORT_JSON).read_text())
    assert report["status"] == MK.DONE and report["checks"]["ok"] is True
    # validation is read; the old ladder rows are not, and the report says why
    assert set(report["sets"]) == {MK.SET_VAL}
    assert "fine-tuned on 2 own ladder games" in said
    assert report["old_ladder_rows"] == {
        "read": False,
        "not_read_because": "a fine-tuned member has seen its old games",
    }
    assert report["finetune"] == {
        "members": ["tuned"],
        "seen_own_games": 2,
        "by_member": {"tuned": 2},
    }
    assert report["checks"]["seen_games_carried"] is True
    assert report["checks"]["seen_own_games"] == 2
    assert report["dataset"]["members_fitted_on_it"] == {"base": True, "tuned": False}
    text = (out / MK.REPORT_MD).read_text()
    assert "has seen 2 of the bot's own ladder games" in text
    assert "The ladder holdout was not read at all" in text
    assert "were used for none of that" not in text
    assert f"**{MK.SET_VAL}**" in text and f"**{MK.SET_LADDER}**" not in text
    # the artifact carries the record a reader of own ladder games checks
    stored = A.load_predictor(out / MK.ARTIFACT_NAME).meta["extra"]
    assert FT.seen_games(stored) == set(games)
    assert SC._trained_on(stored) == f"{FT.MANIFEST_MARK}:2-games:{sha}" != sha
    with pytest.raises(FT.FinetuneError):
        FT.check_unseen(stored, ["old-2"])


def test_the_script_writes_checks_and_reports(fz: F.Featurizer, tmp_path: Path, capsys):
    data = tmp_path / "data"
    sha = tiny_dataset(fz, data)
    extra = {"dataset": {"tag": "unit", "manifest_sha256": sha}, "n_parameters": 1}
    one = save_net(tmp_path / "one.pt", net(fz, 1), fz, extra=extra)
    two = save_net(tmp_path / "two.pt", net(fz, 2), fz, extra=extra)
    out = tmp_path / "out"
    argv = ["--member", f"base={one}", "--member", f"wide={two}"]
    argv += ["--dataset", str(data), "--resamples", "40", "--check-rows", "100"]
    assert MK.main([*argv, "--out", str(out)]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == MK.DONE
    report = json.loads((out / MK.REPORT_JSON).read_text())
    assert report["status"] == MK.DONE and report["checks"]["ok"] is True
    assert report["checks"]["rows_checked"] == 100
    gaps = report["checks"]["mean_identity"]["max_abs_difference"]
    assert max(gaps.values()) < 1e-12
    assert report["sealed_rows_left_out"] == 32
    sets = report["sets"]
    assert sets[MK.SET_VAL]["examples"] == 240
    assert sets[MK.SET_LADDER]["examples"] == 48  # the sealed 32 and test never read
    for found in sets.values():
        assert set(found["predictors"]) == {"base", "wide", MK.ENSEMBLE}
        assert found["predictors"][MK.ENSEMBLE]["ensemble_minus_this"] is None
        for who in ("base", "wide"):
            row = found["predictors"][who]
            want = found["predictors"][MK.ENSEMBLE]["fine_nll"]["value"]
            want -= row["fine_nll"]["value"]
            assert row["ensemble_minus_this"]["fine"]["diff"] == pytest.approx(want)
        assert max(found["mean_identity"]["max_abs_difference"].values()) < 1e-12
    text = (out / MK.REPORT_MD).read_text()
    assert "ensemble - this" in text and "32 rows left out" in text
    assert not (out / (MK.ARTIFACT_NAME + MK.UNVERIFIED_SUFFIX)).exists()
    loaded = A.load_predictor(out / MK.ARTIFACT_NAME)
    assert loaded.kind == EN.KIND and loaded.name == "out"
    assert SC._trained_on(loaded.meta["extra"]) == sha
    # no fine-tuned member: the old ladder rows are unseen, and read
    assert report["finetune"] is None and report["old_ladder_rows"]["read"] is True
    assert EN.KEY_FINETUNE not in loaded.meta["extra"]
    assert "were used for none of that" in text and "has seen" not in text
    assert report["checks"]["seen_games_carried"] is True
    assert report["checks"]["seen_own_games"] == 0
    # the state a coupling binds to is in the report, and is the file's
    assert report["checks"]["state_named_and_kept"] is True
    state = loaded.predictor.state_sha256
    assert report["checks"]["state_sha256"] == state == report["describe"][EN.STATE_KEY]
    assert str(state) in text
    # an existing directory is never written over; a foreign member is refused
    assert MK.main([*argv, "--out", str(out)]) == 1
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith(MK.FAILED)
    three = save_net(tmp_path / "three.pt", net(fz, 3), fz, extra={"dataset": {}})
    bad = ["--member", f"base={one}", "--member", f"other={three}"]
    bad += ["--dataset", str(data), "--out", str(tmp_path / "out2")]
    assert MK.main(bad) == 1
    assert "not fitted on this dataset" in capsys.readouterr().out
    assert not (tmp_path / "out2").exists()
    with pytest.raises(MK.EnsembleError):
        MK.member_specs(["nolabel"])
    # one member is that member: refused before anything is written
    lone = ["--member", f"base={one}", "--dataset", str(data)]
    assert MK.main([*lone, "--out", str(tmp_path / "out3")]) == 1
    assert "at least two --member" in capsys.readouterr().out
    assert not (tmp_path / "out3").exists()


def test_the_script_reads_validation_only_when_the_rows_carry_no_times(
    fz: F.Featurizer, tmp_path: Path, capsys
):
    data = tmp_path / "data"
    sha = tiny_dataset(fz, data, times=False)
    extra = {"dataset": {"tag": "unit", "manifest_sha256": sha}, "n_parameters": 1}
    one = save_net(tmp_path / "one.pt", net(fz, 1), fz, extra=extra)
    two = save_net(tmp_path / "two.pt", net(fz, 2), fz, extra=extra)
    out = tmp_path / "out"
    argv = ["--member", f"base={one}", "--member", f"wide={two}"]
    argv += ["--dataset", str(data), "--resamples", "40", "--check-rows", "100"]
    assert MK.main([*argv, "--out", str(out)]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == MK.DONE
    report = json.loads((out / MK.REPORT_JSON).read_text())
    assert report["status"] == MK.DONE and report["checks"]["ok"] is True
    # the own games cannot be told from the sealed ones: none of them is read
    assert set(report["sets"]) == {MK.SET_VAL}
    assert report["sets"][MK.SET_VAL]["examples"] == 240
    assert report["own_games"]["time_known"] is False
    assert report["old_ladder_rows"] == {
        "read": False,
        "not_read_because": "the dataset's rows carry no times",
    }
    text = (out / MK.REPORT_MD).read_text()
    assert "The ladder holdout was not read: the dataset's rows carry no times" in text
    assert not (out / (MK.ARTIFACT_NAME + MK.UNVERIFIED_SUFFIX)).exists()
    # and a call that would read own games without their times is refused
    own = {"time_known": False, "sealed_cut": 0}
    assert set(MK.load_sets(data, own)) == {MK.SET_VAL}


# --- house rules ------------------------------------------------------------------


def test_the_ensemble_module_loads_no_torch():
    code = (
        "import sys\n"
        "from vgc_bench.src.oppmodel import ensemble\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
        "print(ensemble.KIND)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ensemble"


def test_library_names_nothing_from_the_dex_and_has_no_bare_assert():
    vocab = F.Vocab.build()
    known = set(E.moves_dex()) | set(E.pokedex())
    known |= set(vocab.items[3:]) | set(vocab.abilities[3:])
    for name in (
        "vgc_bench/src/oppmodel/ensemble.py",
        "training/make_oppmodel_ensemble.py",
    ):
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            )
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        }
        found = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and node.value in known
        }
        assert not found, (name, sorted(found))
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.Assert)], name
