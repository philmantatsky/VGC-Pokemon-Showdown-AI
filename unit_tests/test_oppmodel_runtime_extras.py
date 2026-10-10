"""The runtime and the version-2 inputs; what an artifact says differs.

Review fixes of 2026-10-10, second round. Each test states one finding and
fails on the code as it was:

* a layout-version-2 featurizer whose matchup or set-prior block failed wrote
  zeros and counted a name, and the runtime SERVED a forecast computed from
  those zeros. It now stands down for that call when the predictor reads the
  failed block, and only then;
* ``meta['dex_signature_diff']`` of a layout-version-2 artifact did not name a
  changed matchup / set-prior definition, so the tools that print the field
  never showed it;
* a slot's ``other`` entry that holds a candidate the coupling classes
  otherwise was answered with the class of the OTHER bucket.

The battle is the hand-made one of ``test_oppmodel_runtime``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evaluation import oppmodel_scorecard as SC
from unit_tests import test_oppmodel_layout_v2 as TL
from unit_tests import test_oppmodel_model as TM
from unit_tests import test_oppmodel_setprior_integration as TS
from unit_tests.test_oppmodel_coupling import some_coupling
from unit_tests.test_oppmodel_runtime import (
    LOG,
    Hand,
    battle,
    filled,
    forecasts,
    hashed,
    private_view,
    repertoire,
    runtime,
)
from vgc_bench.src.oppmodel import artifact as A
from vgc_bench.src.oppmodel import coupling as CP
from vgc_bench.src.oppmodel import features as F
from vgc_bench.src.oppmodel import matchup as MU
from vgc_bench.src.oppmodel import model as M
from vgc_bench.src.oppmodel import runtime as R
from vgc_bench.src.oppmodel import setprior as SP

ROOT = Path(__file__).resolve().parents[1]
LIVE_ARTIFACT = ROOT / "results_oppmodel/oppnet_v2_blind/artifact.pt"
MU_NAMES = TL.MU_NAMES
ALL_NAMES = (*MU_NAMES, "sp_cand")
MATCHUP_ONLY, EVERYTHING = 7, 15
DEGRADED = R.FAILURE + R.EXTRAS_DEGRADED
HEADS = ("action", "target", "mega")


@pytest.fixture(scope="module")
def fz1() -> F.Featurizer:
    return F.Featurizer.build(repertoire())


@pytest.fixture()
def fzs() -> F.Featurizer:
    """Layout version 2, every array: a fresh one per test (its counters and
    caches are part of what is tested)."""
    return F.Featurizer.build(repertoire(), layout_version=2, set_table=TS.table())


def network(fz: F.Featurizer, mask: int, name: str = "v2") -> M.OppNetPredictor:
    net = TM.jolt(TL.matchup_net(fz, mask) if mask else TM.small_net(fz))
    return M.OppNetPredictor(TL.fill_extras(net), fz, name=name)


def serve(fz: F.Featurizer, predictor: Any) -> R.OpponentPredictor:
    return R.OpponentPredictor(predictor, fz, keep_features=True)


def told(
    rt: R.OpponentPredictor, role: str = "p1", turns: tuple[int, ...] = ()
) -> dict[int, tuple[R.Forecast | None, str]]:
    """(forecast, reason) at every turn line of the hand-made game."""
    fake = battle(role)
    out: dict[int, tuple[R.Forecast | None, str]] = {}
    for event in private_view(LOG, role):
        fake._replay_data.append(list(event))
        if event[1] == "turn" and (not turns or int(event[2]) in turns):
            out[int(event[2])] = rt.predict_with_reason(fake)
    return out


def raising(kind: type[Exception]) -> Any:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise kind("made to fail")

    return broken


def raw_equal(first: R.Forecast | None, second: R.Forecast | None) -> bool:
    assert first is not None and second is not None
    return all(np.array_equal(first.raw[head], second.raw[head]) for head in HEADS)


# --- the stand-down ---------------------------------------------------------------


def test_what_a_predictor_reads(fz1: F.Featurizer, fzs: F.Featurizer):
    assert R.extras_read(network(fzs, EVERYTHING)) == ALL_NAMES
    assert R.extras_read(network(fzs, MATCHUP_ONLY)) == MU_NAMES
    assert R.extras_read(network(fzs, 8)) == ("sp_cand",)
    assert R.extras_read(network(fzs, 0)) == ()
    assert R.extras_read(network(fz1, 0)) == ()
    # A predictor that names none (a count table, a hand-made one) reads none.
    assert R.extras_read(Hand(hashed)) == ()
    assert R.extras_read(None) == () and R.extras_read(object()) == ()
    declared = Hand(hashed)
    setattr(declared, "extra_keys", ["sp_cand", "mu_slot", "not_an_array"])
    assert R.extras_read(declared) == ("mu_slot", "sp_cand")  # table order
    rt = serve(fzs, network(fzs, MATCHUP_ONLY))
    assert rt.extras_read == MU_NAMES
    assert rt._extra_families == frozenset({F.FAMILY_MATCHUP})
    assert R._failed_extras({"matchup_error:KeyError": 1}, {F.FAMILY_MATCHUP}) == [
        F.FAMILY_MATCHUP
    ]
    both = {F.FAMILY_MATCHUP, F.FAMILY_SETPRIOR}
    assert R._failed_extras({"setprior_no_table": 2}, both) == [F.FAMILY_SETPRIOR]
    # Not a failure of the block: every dataset holds such rows as they are.
    assert R._failed_extras({"matchup_no_stats": 3, "unknown_move": 1}, both) == []
    assert R._failed_extras({"matchup_error:X": 1, "setprior_bad_shape": 1}, both) == [
        F.FAMILY_MATCHUP,
        F.FAMILY_SETPRIOR,
    ]
    assert R._failed_extras({"matchup_error:X": 1}, set()) == []


def test_a_healthy_version_2_runtime_serves_as_before(fzs: F.Featurizer):
    rt = serve(fzs, network(fzs, EVERYTHING))
    made = told(rt)
    assert len(made) == 7
    served = {turn: found for turn, (found, why) in made.items() if found is not None}
    assert len(served) >= 5 and all(made[turn][1] == "" for turn in served)
    assert not any(name.startswith(R.EXTRAS_DEGRADED) for name in rt.counters)
    assert DEGRADED not in rt.counters
    assert not any(name.startswith("predict_error") for name in rt.predictor.counters)
    forecast = served[1]
    assert forecast.features is not None and set(ALL_NAMES) <= set(forecast.features)
    assert np.abs(forecast.features["mu_slot"]).sum() > 0
    assert np.abs(forecast.features["sp_cand"]).sum() > 0


@pytest.mark.parametrize(
    ("family", "counter"),
    [
        (F.FAMILY_MATCHUP, "matchup_error:ZeroDivisionError"),
        (F.FAMILY_SETPRIOR, "setprior_error:RuntimeError"),
    ],
)
def test_a_failed_block_gives_no_forecast_and_a_counter(
    fzs: F.Featurizer, monkeypatch: pytest.MonkeyPatch, family: str, counter: str
):
    """THE FAILING CASE of the review: the block raises, the featurizer writes
    zeros and counts, and the forecast used to be served with reason ''."""
    healthy = told(serve(fzs, network(fzs, EVERYTHING)))
    rt = serve(fzs, network(fzs, EVERYTHING))
    fake = battle("p1")
    events = private_view(LOG, "p1")
    first = next(i for i, e in enumerate(events) if e[1] == "turn" and e[2] == "1")
    second = next(i for i, e in enumerate(events) if e[1] == "turn" and e[2] == "2")
    fake._replay_data.extend(list(event) for event in events[: first + 1])
    before = dict(fzs.counters)
    with monkeypatch.context() as patch:
        if family == F.FAMILY_MATCHUP:
            patch.setattr(MU, "hit", raising(ZeroDivisionError))
        else:
            patch.setattr(SP.SetTable, "posterior", raising(RuntimeError))
        assert rt.predict_with_reason(fake) == (None, DEGRADED)
        # The featurizer counted the block's own name, the runtime the family.
        assert fzs.counters[counter] == before.get(counter, 0) + 1
        assert rt.counters[DEGRADED] == 1
        assert rt.counters[f"{R.EXTRAS_DEGRADED}:{family}"] == 1
        other = {F.FAMILY_MATCHUP, F.FAMILY_SETPRIOR} - {family}
        assert not any(f"{R.EXTRAS_DEGRADED}:{name}" in rt.counters for name in other)
        assert rt.counters["forecasts"] == 0
        # Remembered for the turn, like every other failure: not recomputed.
        assert rt.predict_with_reason(fake) == (None, DEGRADED)
        assert rt.counters["cache_hit"] == 1 and rt.counters[DEGRADED] == 1
        assert rt.predict(fake) is None
    told_of = rt.diagnostics()
    assert told_of["runtime"][DEGRADED] == 1
    assert told_of["featurizer"][counter] >= 1
    assert not told_of["predictor"]  # the predictor was never asked
    # For THAT call only: the next turn, healthy again, is served, and it is
    # the forecast of a runtime that never met the failure.
    fake._replay_data.extend(list(event) for event in events[first + 1 : second + 1])
    again, why = rt.predict_with_reason(fake)
    assert why == "" and raw_equal(again, healthy[2][0])
    assert rt.counters[DEGRADED] == 1 and rt.counters["forecasts"] == 1


def test_a_predictor_that_does_not_read_the_block_is_served(
    fz1: F.Featurizer, fzs: F.Featurizer, monkeypatch: pytest.MonkeyPatch
):
    """An ``--extras 0`` network on a version-2 featurizer, a count table, the
    matchup-only network when the set prior fails: none of them reads the
    zeros, so none stands down, and each gives its healthy forecast."""
    plain, matchup = network(fzs, 0, "plain"), network(fzs, MATCHUP_ONLY, "mu")
    table = Hand(hashed)
    want = {
        "plain": told(serve(fzs, plain), turns=(1,))[1][0],
        "matchup": told(serve(fzs, matchup), turns=(1,))[1][0],
    }
    with monkeypatch.context() as patch:
        patch.setattr(SP.SetTable, "posterior", raising(RuntimeError))
        for name, predictor in (("plain", plain), ("matchup", matchup)):
            rt = serve(fzs, predictor)
            found, why = told(rt, turns=(1,))[1]
            assert why == "" and raw_equal(found, want[name]), name
            assert DEGRADED not in rt.counters
        found, why = told(serve(fzs, table), turns=(1,))[1]
        assert why == "" and found is not None
        # ... and the network that DOES read the set prior stands down.
        assert told(serve(fzs, network(fzs, 8)), turns=(1,))[1] == (None, DEGRADED)
    assert fzs.counters["setprior_error:RuntimeError"] >= 4
    with monkeypatch.context() as patch:
        patch.setattr(MU, "hit", raising(ZeroDivisionError))
        rt = serve(fzs, plain)
        found, why = told(rt, turns=(1,))[1]
        assert why == "" and raw_equal(found, want["plain"])
        assert told(serve(fzs, network(fzs, 8)), turns=(1,))[1][1] == ""
        assert told(serve(fzs, matchup), turns=(1,))[1] == (None, DEGRADED)
        # A predictor that says what it reads is held to it, whatever its kind.
        declared = Hand(hashed)
        setattr(declared, "extra_keys", ("mu_cand",))
        assert told(serve(fzs, declared), turns=(1,))[1] == (None, DEGRADED)
        assert declared.batches == []  # never asked
        # A version-1 featurizer computes no block at all.
        assert told(serve(fz1, network(fz1, 0)), turns=(1,))[1][1] == ""


def test_a_table_that_answers_zeros_for_something_it_cannot_read_stands_down(
    fzs: F.Featurizer, monkeypatch: pytest.MonkeyPatch
):
    """Not only an exception: every name the set-prior block counts when it
    wrote zeros it should not have (a wrong shape, the table's own error)."""
    for broken, counter in (
        (lambda *a, **k: np.zeros((1, 1)), "setprior_bad_shape"),
        (None, "setprior_no_table"),
    ):
        rt = serve(fzs, network(fzs, 8))
        with monkeypatch.context() as patch:
            if broken is None:
                patch.setattr(fzs, "set_table", None)
            else:
                patch.setattr(SP.SetTable, "posterior", broken)
            assert told(rt, turns=(1,))[1] == (None, DEGRADED), counter
        assert fzs.counters[counter] >= 1
        assert rt.counters[f"{R.EXTRAS_DEGRADED}:{F.FAMILY_SETPRIOR}"] == 1


def test_the_live_artifact_reads_no_extras_and_is_served_as_before(
    monkeypatch: pytest.MonkeyPatch,
):
    if not LIVE_ARTIFACT.is_file():
        pytest.skip("the live artifact is not here")
    rt = R.OpponentPredictor.load(LIVE_ARTIFACT, keep_features=True)
    assert rt.serving and rt.extras_read == () and rt._extra_families == frozenset()
    assert rt.featurizer is not None and rt.featurizer.layout_version == 1
    want = told(rt, turns=(1, 2))
    assert all(found is not None and why == "" for found, why in want.values())
    keys = {"turn", "model", "kind", "elo", "own_elo", "sheets", "sheets_reported"}
    assert set(filled_dict(want[1][0])) == keys | {"latency_ms", "slots"}
    # Nothing the version-2 code does can stand it down: both blocks broken.
    again = R.OpponentPredictor.load(LIVE_ARTIFACT, keep_features=True)
    with monkeypatch.context() as patch:
        patch.setattr(MU, "hit", raising(ZeroDivisionError))
        patch.setattr(SP.SetTable, "posterior", raising(RuntimeError))
        got = told(again, turns=(1, 2))
    for turn, (found, why) in got.items():
        assert why == "" and raw_equal(found, want[turn][0])
        one, two = filled_dict(found), filled_dict(want[turn][0])
        one.pop("latency_ms")
        two.pop("latency_ms")
        assert one == two
    assert not any(R.EXTRAS_DEGRADED in name for name in again.counters)


def filled_dict(made: R.Forecast | None) -> dict[str, Any]:
    assert made is not None
    return made.to_dict()


# --- what an artifact says differs ------------------------------------------------


def save(path: Path, fz: F.Featurizer, mask: int) -> Path:
    A.save_artifact(
        path,
        kind=M.KIND,
        name="unit_v2",
        featurizer=fz,
        predictor_payload=network(fz, mask).to_payload(),
        extra={"dataset": "unit"},
    )
    return path


def test_meta_names_a_changed_matchup_or_set_prior_definition(
    fz1: F.Featurizer,
    fzs: F.Featurizer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = save(tmp_path / "v2.pt", fzs, EVERYTHING)
    old = save(tmp_path / "v1.pt", fz1, 0)
    assert A.load_predictor(path, strict=True).meta["dex_signature_diff"] == []
    assert A.load_predictor(old, strict=True).meta["dex_signature_diff"] == []
    was = fzs.to_payload()["matchup"]["assumptions"]["proxy_power"]
    with monkeypatch.context() as patch:
        # A later edit of the matchup definitions, without a version bump.
        patch.setattr(MU, "PROXY_POWER", was + 10)
        loaded = A.load_predictor(path)
        assert loaded.featurizer.signature_diff == [F.FAMILY_MATCHUP]
        # THE FAILING CASE: the artifact-level list alone was [].
        assert A.read_artifact(path)["dex_signature_diff"] == []
        assert loaded.meta["dex_signature_diff"] == [F.FAMILY_MATCHUP]
        # ... so the tools that print the field show it: the scorecard's entry
        # and its warning, and the runtime's refusal.
        entry = SC.load_artifact_entry(str(path), [], None)
        assert entry.info["dex_signature_diff"] == [F.FAMILY_MATCHUP]
        card = {"dataset": {}, "sets": {}, "predictors": {"unit_v2": entry.info}}
        said = [line for line in SC.warnings_of(card) if "another dex" in line]
        assert len(said) == 1 and "version-2 input arrays" in said[0]
        rt = R.OpponentPredictor.load(path)
        assert rt.loaded and not rt.serving
        assert rt.load_failure == "dex signature differs: ['matchup']"
        # The version-1 artifact holds no such definition: nothing to differ.
        assert A.load_predictor(old).meta["dex_signature_diff"] == []
    with monkeypatch.context() as patch:
        patch.setattr(F, "SP_QUANT", 50)
        assert A.load_predictor(path).meta["dex_signature_diff"] == [F.FAMILY_SETPRIOR]
        assert A.load_predictor(old).meta["dex_signature_diff"] == []
    with monkeypatch.context() as patch:
        # Both places at once: the union, sorted, each name once.
        other = dict(A.dex_signature(), n_moves=3)
        patch.setattr(A, "dex_signature", lambda: other)
        patch.setattr(MU, "PROXY_POWER", was + 10)
        assert A.load_predictor(path).meta["dex_signature_diff"] == [
            F.FAMILY_MATCHUP,
            "n_moves",
        ]
        # A version-1 artifact: the artifact-level difference, as always.
        assert A.load_predictor(old).meta["dex_signature_diff"] == ["n_moves"]
        info = {"dex_signature_diff": ["n_moves"]}
        said = SC.warnings_of({"dataset": {}, "sets": {}, "predictors": {"old": info}})
        assert said == ["old was written with another dex: ['n_moves']."]


def test_a_version_1_artifact_reports_what_it_always_did(tmp_path: Path):
    """The featurizer's own stored signature is not added for layout version
    1 (``test_oppmodel_artifact`` holds the same for a count table)."""
    odd = F.Featurizer.build(repertoire())
    odd.signature = dict(odd.signature, n_species=3)
    path = save(tmp_path / "odd.pt", odd, 0)
    loaded = A.load_predictor(path)
    assert loaded.featurizer.layout_version == 1
    assert loaded.featurizer.signature_diff == ["n_species"]
    assert loaded.meta["dex_signature_diff"] == []
    # The same stored signature in a version-2 featurizer IS added.
    newer = F.Featurizer.build(repertoire(), layout_version=2, extras=MU_NAMES)
    newer.signature = dict(newer.signature, n_species=3)
    loaded = A.load_predictor(save(tmp_path / "newer.pt", newer, MATCHUP_ONLY))
    assert loaded.meta["dex_signature_diff"] == ["n_species"]
    if LIVE_ARTIFACT.is_file():
        live = A.load_predictor(LIVE_ARTIFACT)
        assert live.meta["dex_signature_diff"] == live.featurizer.signature_diff == []
        assert (
            live.meta["dex_signature_diff"]
            == (A.read_artifact(LIVE_ARTIFACT)["dex_signature_diff"])
        )


# --- an "other" entry that holds two classes --------------------------------------


def paired_forecast(fz: F.Featurizer, seed: int) -> R.Forecast:
    rt, _ = runtime(fz, hashed, coupling=some_coupling(seed))
    return next(
        made
        for made in forecasts(rt, "p1").values()
        if made is not None and made.a is not None and made.b is not None
    )


def without_name(forecast: R.Forecast, column: int) -> R.Forecast:
    """The forecast with one candidate of slot a left without a name, as the
    runtime leaves a candidate whose move id it cannot name."""
    slot = filled(forecast.slots[0])
    moves = list(slot.moves)
    moves[column] = ""
    changed = dataclasses.replace(slot, moves=tuple(moves))
    return dataclasses.replace(forecast, slots=(changed, forecast.slots[1]))


def classed_column(forecast: R.Forecast) -> int:
    """A named candidate of slot a with mass that the coupling does not class
    ``unassigned`` at a target it can be aimed at."""
    assert forecast.joint_inputs is not None
    classes = np.asarray(forecast.joint_inputs[CP.KEY_CLASS])[0, 0]
    slot = filled(forecast.a)
    for column, name in enumerate(slot.moves):
        aimed = np.asarray(slot.target_probs[column]) > 0.0
        kinds = classes[column * F.N_TARGET : (column + 1) * F.N_TARGET]
        if name and float(slot.action_probs[column]) > 0.0 and aimed.any():
            if (kinds[aimed] != CP.C_UNASSIGNED).any():
                return column
    raise AssertionError("no classed candidate in slot a")


def test_an_other_entry_of_one_class_agrees_with_the_joint_list(fz1: F.Featurizer):
    """With the tables a featurizer builds, every entry of ``joint_top``
    (the OTHER bucket's included) stands in the ratio ``pair_weight`` says."""
    forecast = paired_forecast(fz1, 41)
    assert forecast.pair is not None and forecast.joint_inputs is not None
    classes = np.asarray(forecast.joint_inputs[CP.KEY_CLASS])[0]
    before = dict(R.COUNTERS)
    ratios, others = [], 0
    for entry in forecast.joint_top(10_000):
        first, second = entry.slots
        assert first is not None and second is not None
        weight = forecast.pair_weight(first, second)
        assert weight == float(
            forecast.pair.weight[
                classes[0, entry.replies[0]], classes[1, entry.replies[1]]
            ]
        )
        others += R.ACTION_OTHER in (first.kind, second.kind)
        ratios.append(
            entry.probability / (first.probability * second.probability * weight)
        )
    assert others > 5 and np.allclose(ratios, ratios[0], rtol=1e-9)
    slot_a, slot_b = filled(forecast.a), filled(forecast.b)
    assert forecast._other_class(0) == forecast._other_class(1) == CP.C_UNASSIGNED
    other = next(a for a in slot_a.actions if a.kind == R.ACTION_OTHER)
    assert forecast.action_class(0, other) == CP.C_UNASSIGNED
    switch = (R.ACTION_SWITCH,)
    assert forecast.reply_weight((R.ACTION_OTHER,), switch) == float(
        forecast.pair.weight[CP.C_UNASSIGNED, CP.C_SWITCH]
    )
    # A candidate without a name that carries no mass: still one class.
    column = classed_column(forecast)
    places = np.arange(len(slot_a.action_probs))
    quiet = dataclasses.replace(
        slot_a, action_probs=np.where(places == column, 0.0, slot_a.action_probs)
    )
    silent = without_name(dataclasses.replace(forecast, slots=(quiet, slot_b)), column)
    assert silent._other_class(0) == CP.C_UNASSIGNED
    # One whose own row is unassigned too (the row of "no move" in every table
    # a featurizer builds): one class, and the joint list agrees with it.
    inputs = dict(forecast.joint_inputs)
    kinds = np.array(inputs[CP.KEY_CLASS])
    kinds[0, 0, column * F.N_TARGET : (column + 1) * F.N_TARGET] = CP.C_UNASSIGNED
    inputs[CP.KEY_CLASS] = kinds
    blank = dataclasses.replace(without_name(forecast, column), joint_inputs=inputs)
    assert blank._other_class(0) == CP.C_UNASSIGNED
    ratios, unnamed = [], 0
    for entry in blank.joint_top(10_000):
        first, second = entry.slots
        assert first is not None and second is not None
        weight = blank.pair_weight(first, second)
        assert weight == float(
            forecast.pair.weight[
                kinds[0, 0, entry.replies[0]], kinds[0, 1, entry.replies[1]]
            ]
        )
        unnamed += entry.replies[0] // F.N_TARGET == column
        ratios.append(
            entry.probability / (first.probability * second.probability * weight)
        )
    assert unnamed > 0 and np.allclose(ratios, ratios[0], rtol=1e-9)
    for name in (R.PAIR_OTHER_MIXED, R.PAIR_UNCLASSED, R.REPLY_UNREAD):
        assert R.COUNTERS[name] == before.get(name, 0), name


def test_an_other_entry_of_two_classes_is_counted_not_answered(fz1: F.Featurizer):
    """THE CASE of the review: a candidate without a name that the coupling
    classes by its own row. ``joint_top`` applies that class; the ``other``
    entry cannot say one, so ``pair_weight`` answers 1.0 and counts."""
    forecast = paired_forecast(fz1, 42)
    assert forecast.pair is not None and forecast.joint_inputs is not None
    slot_a, slot_b = filled(forecast.a), filled(forecast.b)
    column = classed_column(forecast)
    mixed = without_name(forecast, column)
    other = R.ActionForecast(R.ACTION_OTHER, 0.1)
    partner = slot_b.actions[0]
    before = dict(R.COUNTERS)
    assert mixed._other_class(0) is None and mixed.action_class(0, other) is None
    assert mixed.pair_weight(other, partner) == 1.0
    assert mixed.reply_weight((R.ACTION_OTHER,), (R.ACTION_SWITCH,)) == 1.0
    assert R.COUNTERS[R.PAIR_OTHER_MIXED] == before.get(R.PAIR_OTHER_MIXED, 0) + 4
    assert R.COUNTERS[R.PAIR_UNCLASSED] == before.get(R.PAIR_UNCLASSED, 0) + 1
    assert R.COUNTERS[R.REPLY_UNREAD] == before.get(R.REPLY_UNREAD, 0) + 1
    assert R.process_counters()[f"runtime:{R.PAIR_OTHER_MIXED}"] >= 4
    # Slot b is untouched, and so is every entry that is not ``other``.
    assert mixed._other_class(1) == CP.C_UNASSIGNED
    named = next(a for a in slot_a.actions if a.kind == R.ACTION_MOVE and a.move)
    if named.move != slot_a.moves[column]:
        assert mixed.pair_weight(named, partner) == forecast.pair_weight(named, partner)
    # A move the forecast does not name is still the OTHER bucket's own move.
    assert mixed.reply_weight(
        (R.ACTION_MOVE, "nosuchmove", "foe_a"), (R.ACTION_SWITCH,)
    ) == float(forecast.pair.weight[CP.C_UNASSIGNED, CP.C_SWITCH])
    # The joint list itself is what it was: the mismatch is in the entry only.
    assert [e.replies for e in mixed.joint_top(16)] == [
        e.replies for e in forecast.joint_top(16)
    ]
    # Without a coupling nothing is classed and nothing is counted.
    plain = next(
        made
        for made in forecasts(runtime(fz1, hashed)[0], "p1").values()
        if made is not None and made.a is not None and made.b is not None
    )
    count = R.COUNTERS[R.PAIR_OTHER_MIXED]
    assert without_name(plain, column).pair_weight(other, partner) == 1.0
    assert R.COUNTERS[R.PAIR_OTHER_MIXED] == count
