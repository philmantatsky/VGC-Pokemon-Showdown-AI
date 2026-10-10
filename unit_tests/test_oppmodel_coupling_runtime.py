"""The pair coupling where it is read: the runtime's forecast and the scorecard.

The runtime tests drive the hand-made battle of ``test_oppmodel_runtime``; the
scorecard tests use the seven-example batch of ``test_oppmodel_scorecard``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evaluation import oppmodel_scorecard as SC
from unit_tests.test_oppmodel_coupling import intent_table, some_coupling, some_table
from unit_tests.test_oppmodel_runtime import (
    filled,
    forecasts,
    hashed,
    other,
    reference,
    repertoire,
    runtime,
    served,
)
from unit_tests.test_oppmodel_scorecard import coverage_batch, dyadic
from vgc_bench.src.oppmodel import artifact as A
from vgc_bench.src.oppmodel import coupling as CP
from vgc_bench.src.oppmodel import events as E
from vgc_bench.src.oppmodel import features as F
from vgc_bench.src.oppmodel import joint as J
from vgc_bench.src.oppmodel import runtime as R
from vgc_bench.src.oppmodel import tables as T

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
    return F.Featurizer.build(repertoire())


def both(made: R.Forecast) -> bool:
    return made.a is not None and made.b is not None


def kept(made: R.Forecast) -> dict[str, np.ndarray]:
    """The features a runtime with ``keep_features`` attached."""
    assert made.features is not None
    return made.features


def without_latency(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key != "latency_ms"}


# --- the runtime ----------------------------------------------------------------


def test_without_a_coupling_the_forecast_is_the_forecast_of_before(fz: F.Featurizer):
    made = forecasts(runtime(fz, hashed)[0], "p1")
    seen = 0
    for forecast in made.values():
        if forecast is None:
            continue
        seen += 1
        assert forecast.pair is None and forecast.coupling is None
        told = forecast.to_dict()
        assert set(told) == FORECAST_KEYS  # no new key: the bot logs this dict
        json.dumps(told, allow_nan=False)
        assert forecast.joint_inputs is not None
        assert CP.KEY_CLASS not in forecast.joint_inputs
        assert forecast.pair_class_weight(E.INTENT_PROTECT, E.INTENT_PROTECT) == 1.0
        if both(forecast):
            first = filled(forecast.slots[0]).actions[0]
            second = filled(forecast.slots[1]).actions[0]
            assert forecast.pair_weight(first, second) == 1.0
            assert forecast.action_class(0, first) is None
        # the lazy joint is joint_replies on the kept features: the plain product
        for mega in (False, True):
            top = forecast.joint_top(8, mega=mega)
            offline = J.joint_replies(forecast.raw, kept(forecast), k=8, mega=mega)
            assert offline.n == 1 and len(top) == int((offline.prob[0] > 0).sum())
            for position, entry in enumerate(top):
                assert not entry.coupled
                assert entry.replies == tuple(
                    int(v) for v in offline.reply[0, position]
                )
                assert entry.probability == float(offline.prob[0, position])
                state = int(offline.mega[0, position])
                assert entry.mega == (None if state == J.MEGA_NONE else state - 1)
        assert forecast.joint_top(8, coupled=False) == forecast.joint_top(8)
    assert seen >= 3


def test_a_coupling_changes_no_per_slot_number_and_adds_the_pair(fz: F.Featurizer):
    made = some_coupling(33)
    plain = forecasts(runtime(fz, hashed)[0], "p1")
    rt, _ = runtime(fz, hashed, coupling=made)
    assert rt.coupling is made
    coupled = forecasts(rt, "p1")
    assert plain.keys() == coupled.keys()
    pairs = 0
    for turn, before in plain.items():
        after = coupled[turn]
        assert (before is None) == (after is None)
        if before is None or after is None:
            continue
        # every per-slot field is the uncoupled marginal
        for index in range(2):
            old, new = before.slots[index], after.slots[index]
            assert (old is None) == (new is None)
            if old is None or new is None:
                continue
            assert np.array_equal(old.action_probs, new.action_probs)
            assert np.array_equal(old.target_probs, new.target_probs)
            assert old.actions == new.actions and old.intents == new.intents
            for name in ("p_switch", "p_protect", "p_fake_out", "p_attacks", "p_mega"):
                assert getattr(old, name) == getattr(new, name)
        for head in ("action", "target", "mega"):
            assert np.array_equal(before.raw[head], after.raw[head])
        assert before.p_attacked == after.p_attacked
        told = after.to_dict()
        assert after.coupling is made
        if not both(after):
            assert after.pair is None and set(told) == FORECAST_KEYS
            assert without_latency(told) == without_latency(before.to_dict())
            assert after.pair_class_weight(0, 0) == 1.0
            continue
        pairs += 1
        pair = after.pair
        assert pair is not None and pair.coupling == made.name
        bucket = int(CP.reply_buckets(kept(after))[0])
        assert pair.bucket == CP.BUCKETS[bucket] and pair.classes == CP.CLASSES
        assert np.array_equal(pair.weight, made.table[bucket])
        assert not pair.weight.flags.writeable
        # to_dict: the dict of before plus one small entry
        assert told.pop("pair") == {"coupling": made.name, "bucket": pair.bucket}
        assert without_latency(told) == without_latency(before.to_dict())
        json.dumps(after.to_dict(), allow_nan=False)
        assert after.pair_class_weight(E.INTENT_SWITCH, E.INTENT_PROTECT) == float(
            made.table[bucket, CP.C_SWITCH, CP.C_PROTECT]
        )
        assert after.pair_class_weight(CP.C_PROTECT, CP.C_PROTECT) == float(
            made.table[bucket, CP.C_PROTECT, CP.C_PROTECT]
        )
        # joint_top is the offline coupled joint on the same example
        intents = fz.tables.move_intent
        top = after.joint_top(10_000)
        offline = J.joint_replies(
            after.raw, kept(after), k=10_000, coupling=made, move_intent=intents
        )
        assert offline.n == 1 and offline.coupled
        assert len(top) == int(offline.size[0])
        for position, entry in enumerate(top):
            assert entry.coupled
            assert entry.replies == tuple(int(v) for v in offline.reply[0, position])
            assert entry.probability == float(offline.prob[0, position])
        assert sum(entry.probability for entry in top) == pytest.approx(1.0)
        # coupled=False is the plain product, the same as without a coupling
        assert after.joint_top(8, coupled=False) == before.joint_top(8)
        # entries stand in the ratio p_a * p_b * pair_weight says
        classes = CP.reply_classes(kept(after), intents)[0]
        scaled = []
        for entry in top:
            first, second = entry.slots
            if first is None or second is None:
                continue
            if R.ACTION_OTHER in (first.kind, second.kind):
                continue  # a candidate without a name reads "other"
            weight = after.pair_weight(first, second)
            assert weight == float(
                made.table[
                    bucket, classes[0, entry.replies[0]], classes[1, entry.replies[1]]
                ]
            )
            scaled.append(
                entry.probability / (first.probability * second.probability * weight)
            )
        assert len(scaled) > 20 and np.allclose(scaled, scaled[0], rtol=1e-9)
    assert pairs >= 2


def test_pair_methods_never_raise_on_nonsense(fz: F.Featurizer):
    rt, _ = runtime(fz, hashed, coupling=some_coupling(34))
    forecast = next(
        made for made in forecasts(rt, "p1").values() if made is not None and both(made)
    )
    slot_a, slot_b = filled(forecast.slots[0]), filled(forecast.slots[1])
    action = slot_a.actions[0]
    for wrong in (None, 3, "x", object(), [action]):
        assert forecast.pair_weight(wrong, action) == 1.0
        assert forecast.pair_weight(action, wrong) == 1.0
        assert forecast.action_class(0, wrong) is None
    for wrong in (None, "x", object(), 8, -1, 1.5, [0]):
        assert forecast.pair_class_weight(wrong, 0) == 1.0
        assert forecast.pair_class_weight(0, wrong) == 1.0
    assert forecast.action_class(5, action) is None
    stranger = R.ActionForecast(R.ACTION_MOVE, 0.5, "nosuchmove", E.TARGET_FOE_A)
    assert forecast.pair_weight(stranger, action) == 1.0
    # plain replies, as a search holding choice strings has them
    second = slot_b.actions[0]
    for entry in slot_a.actions:
        if entry.kind == R.ACTION_MOVE:
            told = (R.ACTION_MOVE, entry.move, entry.target)
        else:
            told = (entry.kind,)
        other_told = (
            (R.ACTION_MOVE, second.move, second.target)
            if second.kind == R.ACTION_MOVE
            else (second.kind,)
        )
        assert forecast.reply_weight(told, other_told) == forecast.pair_weight(
            entry, second
        )
    assert forecast.pair is not None and forecast.joint_inputs is not None
    weight = forecast.pair.weight
    switch = (R.ACTION_SWITCH,)
    assert forecast.reply_weight(switch, R.ACTION_SWITCH) == float(
        weight[CP.C_SWITCH, CP.C_SWITCH]
    )
    assert forecast.reply_weight((R.ACTION_MOVE, "nosuchmove", "foe_a"), switch) == (
        float(weight[CP.C_UNASSIGNED, CP.C_SWITCH])
    )
    named = next(name for name in slot_a.moves if name)
    assert forecast.reply_weight((R.ACTION_MOVE, named, "field"), switch) == (
        forecast.reply_weight((R.ACTION_MOVE, named, None), switch)
    )
    for wrong in (
        None,
        (),
        ("pass",),
        7,
        object(),
        (R.ACTION_MOVE,),
        ("move", named, "up"),
    ):
        assert forecast.reply_weight(wrong, switch) == 1.0
        assert forecast.reply_weight(switch, wrong) == 1.0
    many: Any = "many"
    not_a_flag: Any = np.zeros(2)
    assert forecast.joint_top(many) == ()
    assert forecast.joint_top(8, mega=not_a_flag) == ()
    assert len(forecast.joint_top(0)) == 1  # a length below 1 is read as 1
    # a coupling the forecast cannot apply gives nothing, never the plain list
    broken = R.Forecast(
        **{
            **{name: getattr(forecast, name) for name in forecast.__dataclass_fields__},
            "joint_inputs": {
                key: value
                for key, value in forecast.joint_inputs.items()
                if key not in (CP.KEY_CLASS, CP.KEY_BUCKET)
            },
        }
    )
    assert broken.joint_top(8) == () and len(broken.joint_top(8, coupled=False)) == 8
    assert broken.pair_weight(action, slot_b.actions[0]) == 1.0


def test_load_hands_an_artifacts_coupling_to_the_runtime(tmp_path: Path, fz):
    rows = []
    for role in ("p1", "p2"):
        for record in reference(role).turns:
            rows.append(fz.encode_turn(record.snapshot, record.actions, other(role)))
    table = T.FlagsTable.fit(F.collate([row for row in rows if row]), featurizer=fz)
    made = CP.PairCoupling.build(
        some_table(35), name="tiny_pair", fitted_after=CP.fitted_after("table", None)
    )
    for name, coupling in (("plain.pt", None), ("coupled.pt", made)):
        A.save_artifact(
            tmp_path / name,
            kind=A.KIND_TABLE,
            name="tiny_flags",
            featurizer=fz,
            predictor_payload=table.to_payload(),
            extra={"dataset": "unit"},
            coupling=coupling,
        )
    plain = R.OpponentPredictor.load(tmp_path / "plain.pt", keep_features=True)
    coupled = R.OpponentPredictor.load(tmp_path / "coupled.pt", keep_features=True)
    assert plain.coupling is None and coupled.coupling.same_as(made)
    before, after = forecasts(plain, "p1"), forecasts(coupled, "p1")
    pairs = 0
    for turn, old in before.items():
        new = after[turn]
        if old is None or new is None:
            assert old is None and new is None
            continue
        assert old.pair is None and "pair" not in old.to_dict()
        told = new.to_dict()
        told.pop("pair", None)
        assert without_latency(told) == without_latency(old.to_dict())
        if both(new):
            pairs += 1
            assert served(new).pair is not None
            assert new.joint_top(8)[0].coupled and not old.joint_top(8)[0].coupled
    assert pairs >= 2 and not coupled.diagnostics()["predictor"]


# --- the scorecard --------------------------------------------------------------


def card_batch() -> tuple[F.Batch, F.Tables]:
    batch = coverage_batch()
    batch["cand_move"][:, :, 0] = 1  # the aimed attack of intent_table
    batch["cand_move"][:, :, 1] = 2  # its Protect-family row
    # the intent classes of the counted two-slot turns (0, 1 and 5)
    batch["y_intent"][0] = (CP.C_FOE_A, CP.C_PROTECT)
    batch["y_intent"][1] = (CP.C_SWITCH, CP.C_FOE_B)
    batch["y_intent"][5] = (CP.C_FOE_A, CP.C_FOE_A)
    table = intent_table()
    tables = F.Tables(
        np.zeros((1, 1), np.float32),
        np.zeros((len(table), 1), np.float32),
        table,
        np.zeros(len(table), np.uint16),
        ("x",),
        tuple(f"m{i}" for i in range(len(table))),
    )
    return batch, tables


def test_a_card_without_a_coupling_is_the_card_of_before():
    batch, tables = card_batch()
    preds = {"dyadic": dyadic(batch), "uniform": F.uniform_prediction(batch)}
    base = SC.joint_coverage_set(batch, preds, resamples=200, seed=1, tables=tables)
    for empty in (None, {}):
        again = SC.joint_coverage_set(
            batch, preds, resamples=200, seed=1, tables=tables, couplings=empty
        )
        assert json.dumps(SC._plain(again), sort_keys=True) == json.dumps(
            SC._plain(base), sort_keys=True
        )
    assert SC.REPLY_HOLDS_SWITCH not in base["slices"]
    assert all(
        set(made) == set(SC.JOINT_VARIANTS) for made in base["predictors"].values()
    )
    section = SC.joint_coverage(
        {SC.SET_LADDER: batch}, {SC.SET_LADDER: preds}, resamples=50, tables=tables
    )
    assert set(section["variants"]) == set(SC.JOINT_VARIANTS)
    assert "coupled" not in section and "coupled" not in section["definitions"]
    with pytest.raises(SC.ScorecardError):
        SC.joint_coverage_set(batch, preds, couplings={"dyadic": some_coupling(1)})


def test_a_coupled_predictor_gets_a_second_row_and_a_paired_change():
    batch, tables = card_batch()
    preds = {"dyadic": dyadic(batch), "uniform": F.uniform_prediction(batch)}
    base = SC.joint_coverage_set(batch, preds, resamples=200, seed=1, tables=tables)
    # an identity coupling: equal rows, a change of exactly 0
    same = SC.joint_coverage_set(
        batch,
        preds,
        resamples=200,
        seed=1,
        tables=tables,
        couplings={"dyadic": CP.PairCoupling.identity()},
    )
    assert set(same["predictors"]["dyadic"]) == {*SC.JOINT_VARIANTS, SC.JOINT_COUPLED}
    assert set(same["predictors"]["uniform"]) == set(SC.JOINT_VARIANTS)
    plain = same["predictors"]["dyadic"][SC.JOINT_PLAIN]
    twin = same["predictors"]["dyadic"][SC.JOINT_COUPLED]
    for label, row in plain["slices"].items():
        other_row = twin["slices"][label]
        assert (
            other_row["top"] == row["top"] and other_row["log_prob"] == row["log_prob"]
        )
        assert other_row["top_interval"] == row["top_interval"]
        if row["examples"]:
            for key in ("top", "log_prob"):
                change = other_row["change"][key]
                assert change["diff"] == 0.0
                assert change["low"] in (None, 0.0) and change["high"] in (None, 0.0)
    # the plain rows, and every number of the predictor without a coupling,
    # are those of a card without any coupling
    for name in preds:
        for variant in SC.JOINT_VARIANTS:
            assert json.dumps(
                SC._plain(same["predictors"][name][variant])
            ) == json.dumps(
                SC._plain(
                    {
                        **base["predictors"][name][variant],
                        "slices": {
                            **base["predictors"][name][variant]["slices"],
                            **{
                                label: same["predictors"][name][variant]["slices"][
                                    label
                                ]
                                for label in (
                                    SC.REPLY_HOLDS_SWITCH,
                                    SC.REPLY_HOLDS_PROTECT,
                                )
                            },
                        },
                    }
                )
            )
    # the two new slices: example 1 holds a switch, example 0 a Protect
    assert same["slices"][SC.REPLY_HOLDS_SWITCH]["examples"] == 1
    assert same["slices"][SC.REPLY_HOLDS_PROTECT]["examples"] == 1

    # a real coupling: the change is coupled minus plain on the same examples
    made = some_coupling(36)
    found = SC.joint_coverage_set(
        batch, preds, resamples=200, seed=1, tables=tables, couplings={"dyadic": made}
    )
    plain = found["predictors"]["dyadic"][SC.JOINT_PLAIN]
    coupled = found["predictors"]["dyadic"][SC.JOINT_COUPLED]
    moved = False
    for label, row in coupled["slices"].items():
        if not row["examples"]:
            continue
        before = plain["slices"][label]
        key = str(SC.JOINT_K)
        assert row["change"]["top"]["diff"] == pytest.approx(
            row["top"][key] - before["top"][key], abs=1e-12
        )
        assert row["change"]["log_prob"]["diff"] == pytest.approx(
            row["log_prob"] - before["log_prob"], abs=1e-12
        )
        moved |= abs(row["change"]["log_prob"]["diff"]) > 1e-6
    assert moved
    one = coupled["slices"][SC.SLOTS_ONE]
    assert one["change"]["log_prob"]["diff"] == 0.0  # one acting slot: no pair
    # the dependence table says what the coupled joint puts on each pair event
    events = found["dependence"]["predictors"]
    assert all("coupled" in row for row in events["dyadic"].values())
    assert all("coupled" not in row for row in events["uniform"].values())
    assert all(
        "coupled" not in row
        for row in base["dependence"]["predictors"]["dyadic"].values()
    )

    # the section and its rendering
    section = SC.joint_coverage(
        {SC.SET_LADDER: batch},
        {SC.SET_LADDER: preds},
        resamples=50,
        tables=tables,
        couplings={"dyadic": made},
    )
    assert section["coupled"] == ["dyadic"]
    assert SC.JOINT_COUPLED in section["variants"]
    card = {
        "order": ["dyadic", "uniform"],
        "set_labels": {},
        SC.JOINT_KEY: SC._plain(section),
    }
    text = "\n".join(SC.joint_lines(card))
    assert "with its pair coupling, by slice" in text
    assert SC.REPLY_HOLDS_PROTECT in text and "; coupled " in text
    bare = SC.joint_coverage(
        {SC.SET_LADDER: batch}, {SC.SET_LADDER: preds}, resamples=50, tables=tables
    )
    plain_text = "\n".join(SC.joint_lines({**card, SC.JOINT_KEY: SC._plain(bare)}))
    assert "pair coupling" not in plain_text and "coupled" not in plain_text


def test_the_scorecard_keeps_an_artifacts_coupling_unless_told_not_to(
    tmp_path: Path, fz
):
    rows = []
    for record in reference("p1").turns:
        rows.append(fz.encode_turn(record.snapshot, record.actions, "p2"))
    table = T.FlagsTable.fit(F.collate([row for row in rows if row]), featurizer=fz)
    made = CP.PairCoupling.build(some_table(37), name="tiny_pair")
    for name, coupling in (("plain.pt", None), ("coupled.pt", made)):
        A.save_artifact(
            tmp_path / name,
            kind=A.KIND_TABLE,
            name="tiny_flags",
            featurizer=fz,
            predictor_payload=table.to_payload(),
            extra={},
            coupling=coupling,
        )
    plain = SC.load_artifact_entry(str(tmp_path / "plain.pt"), [], None)
    assert plain.coupling is None and "pair_coupling" not in plain.info
    entry = SC.load_artifact_entry(str(tmp_path / "coupled.pt"), [], None)
    assert entry.coupling.same_as(made)
    assert entry.info["pair_coupling"] == {
        "name": "tiny_pair",
        "used": True,
        "fitted_after": {},
    }
    SC.drop_coupling(entry)
    assert entry.coupling is None and entry.info["pair_coupling"]["used"] is False
    assert SC.drop_coupling(plain) is plain and "pair_coupling" not in plain.info
    assert SC.parse_args(["--dataset", "d", "--no-coupling"]).no_coupling is True
