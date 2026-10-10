"""The pair coupling of two slots: the object, the joint, the fit, the artifact.

Synthetic batches and hand-made marginals (the helpers of
``test_oppmodel_joint``); nothing here reads a dataset.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from unit_tests.test_oppmodel_joint import FAR, C, as_dict, brute_force, random_case
from vgc_bench.src.oppmodel import artifact as A
from vgc_bench.src.oppmodel import coupling as CP
from vgc_bench.src.oppmodel import events as E
from vgc_bench.src.oppmodel import features as F
from vgc_bench.src.oppmodel import joint as J
from vgc_bench.src.oppmodel import tables as T

ROOT = Path(__file__).resolve().parents[1]
ARRAYS = (
    "reply",
    "mega",
    "prob",
    "size",
    "kept",
    "ok",
    "rank",
    "prob_true",
    "above",
    "ties",
    "tilt",
)
PER_SLOT = ("act_mon", "action_mask", "cand_flag", "cand_tmask", "cand_move")


def some_table(seed: int = 0, spread: float = 0.6) -> np.ndarray:
    """A positive ``[3, 8, 8]`` table with both symmetries, far from ones."""
    rng = np.random.default_rng(seed)
    return CP.symmetrise(np.exp(rng.normal(0.0, spread, (3, 8, 8))))


def some_coupling(seed: int = 0) -> CP.PairCoupling:
    return CP.PairCoupling.build(some_table(seed), name=f"unit{seed}")


def intent_table() -> np.ndarray:
    """A move-intent table whose rows behave like the dex's under a renaming
    of the faced side's slots: (foe a, foe b, ally, self, auto)."""
    a, b, protect = CP.C_FOE_A, CP.C_FOE_B, CP.C_PROTECT
    spread = E.INTENT_CLASSES.index(E.INTENT_SPREAD)
    status = E.INTENT_CLASSES.index(E.INTENT_STATUS_FOE)
    support = E.INTENT_CLASSES.index(E.INTENT_SUPPORT)
    rows = [
        [-1, -1, -1, -1, -1],  # nothing the dex can decide
        [a, b, support, support, spread],  # an aimed attack
        [protect] * 5,  # a Protect-family move
        [status, status, support, support, support],  # a status move
        [spread] * 5,  # a spread attack
        [a, b, support, -1, spread],
    ]
    return np.array(rows, dtype=np.int8)


def with_context(batch: F.Batch, seed: int) -> F.Batch:
    """Random candidate ids, turns and faced slots: every bucket occurs."""
    rng = np.random.default_rng(seed)
    out = dict(batch)
    n = len(batch["turn"])
    out["cand_move"] = rng.integers(0, len(intent_table()), (n, 2, C)).astype(np.uint16)
    out["turn"] = rng.integers(1, 4, n).astype(np.uint8)
    foes = np.zeros((n, 2), dtype=np.int8)
    foes[rng.random(n) < 0.3, 1] = -1
    out["foe_mon"] = foes
    return out


def case(
    n: int, seed: int
) -> tuple[F.Batch, dict[str, np.ndarray], dict[str, np.ndarray]]:
    batch, pred, truth = random_case(n, seed)
    return with_context(batch, seed + 1), pred, truth


def same_arrays(first: J.JointReplies, second: J.JointReplies) -> bool:
    return first.n == second.n and all(
        np.array_equal(getattr(first, name), getattr(second, name)) for name in ARRAYS
    )


class OnesThroughChunk:
    """An all-ones coupling that does not take the short cut: its factor is
    pushed through ``_chunk``."""

    is_identity = False

    def __init__(self) -> None:
        self.inner = CP.PairCoupling.identity()

    def reply_maps(self, batch: Any, move_intent: Any = None) -> Any:
        return self.inner.reply_maps(batch, move_intent)


# --- the object -----------------------------------------------------------------


def test_identity_payload_round_trip_and_refusals():
    assert CP.N_ORBIT == 29 and int(CP.ORBIT_SIZE.sum()) == 64
    assert CP.PairCoupling.identity().is_identity
    made = CP.PairCoupling.build(
        some_table(1), name="unit", fitted_after={"kind": "table"}, info={"rows": 3}
    )
    assert not made.is_identity and not made.table.flags.writeable
    again = CP.PairCoupling.from_payload(made.to_payload())
    assert again.same_as(made) and again.info == {"rows": 3}
    assert made.padded().shape == (3, 9, 9)
    assert (made.padded()[:, 8, :] == 1).all() and (made.padded()[:, :, 8] == 1).all()
    later = made.weights("later_two")
    assert later is not None and np.array_equal(later, made.table[CP.BUCKET_TWO])
    assert made.weights(7) is None and made.weights("nowhere") is None

    def changed(**change: Any) -> dict[str, Any]:
        return {**made.to_payload(), **change}

    bad_tables = {
        "shape": np.ones((3, 8, 7)),
        "zero": np.zeros((3, 8, 8)),
        "nan": np.full((3, 8, 8), np.nan),
    }
    lopsided = some_table(1).copy()
    lopsided[0, 0, 1] *= 2.0  # not symmetric
    bad_tables["asymmetric"] = lopsided
    renamed = some_table(1).copy()
    renamed[1, CP.C_FOE_A, CP.C_FOE_A] *= 2.0  # differs from (foe b, foe b)
    bad_tables["exchange"] = renamed
    for name, table in bad_tables.items():
        with pytest.raises(ValueError):
            CP.PairCoupling.from_payload(changed(table=table))
        with pytest.raises(ValueError):
            CP.PairCoupling.build(table, name=name)
    for change in (
        {"format": "something"},
        {"version": 2},
        {"class_scheme": "intent9"},
        {"bucket_scheme": "none"},
        {"classes": list(CP.CLASSES[::-1])},
        {"buckets": ["a", "b", "c"]},
        {"marginals": "kept"},
    ):
        with pytest.raises(ValueError):
            CP.PairCoupling.from_payload(changed(**change))
    with pytest.raises(ValueError):
        CP.PairCoupling.from_payload("not a dict")
    with pytest.raises(ValueError):
        CP.PairCoupling.from_payload({"format": CP.FORMAT})


def test_class_index_reads_names_and_numbers_only():
    assert CP.class_index(E.INTENT_PROTECT) == CP.C_PROTECT
    assert CP.class_index(CP.UNASSIGNED) == CP.C_UNASSIGNED == 7
    assert CP.class_index(3) == 3 and CP.class_index(np.int64(0)) == 0
    for wrong in ("nothing", 8, -1, 1.5, None, True, [1], object()):
        assert CP.class_index(wrong) is None


def test_fitted_after_must_match_the_predictor_state():
    described = {
        "temperatures": {"action": 1.2, "target": 0.9, "mega": 1.0},
        "event_calibrated": False,
    }
    after = CP.fitted_after("oppnet", described)
    made = CP.PairCoupling.build(some_table(), fitted_after=after)
    assert made.fits("oppnet", described) == ""
    assert made.fits("table", described)  # another kind
    hotter = {**described, "temperatures": {"action": 1.3, "target": 0.9}}
    assert "action temperature" in made.fits("oppnet", hotter)
    assert "event calibration" in made.fits(
        "oppnet", {**described, "event_calibrated": True}
    )
    plain = CP.PairCoupling.build(some_table(), fitted_after=CP.fitted_after("t", None))
    assert plain.fits("t", None) == "" and plain.fits("u", None)
    assert CP.PairCoupling.build(some_table()).fits("anything", described) == ""


# --- classes and buckets --------------------------------------------------------


def test_reply_classes_follow_the_intent_table_and_sum_to_intent_probs():
    batch, pred, _ = case(60, 3)
    table = intent_table()
    classes = CP.reply_classes(batch, table)
    n_cand = C
    assert classes.shape == (60, 2, J.reply_size(n_cand))
    assert (classes[:, :, J.other_reply(n_cand)] == CP.C_UNASSIGNED).all()
    assert (classes[:, :, J.other_reply(n_cand) + 1 :] == CP.C_SWITCH).all()
    moves = batch["cand_move"].astype(int)
    for (i, s, c), move in np.ndenumerate(moves):
        for t in range(F.N_TARGET):
            want = table[move, t]
            got = classes[i, s, J.move_reply(c, t)]
            assert got == (CP.C_UNASSIGNED if want < 0 else want)
        if move == 2:  # the Protect-family row: one class whatever the target
            held = classes[i, s, c * F.N_TARGET : (c + 1) * F.N_TARGET]
            assert (held == CP.C_PROTECT).all()
    # summed by class, a slot's reply probabilities are intent_probs
    prob, _ = J.slot_replies(pred, batch)
    tables = F.Tables(
        np.zeros((1, 1), np.float32),
        np.zeros((len(table), 1), np.float32),
        table,
        np.zeros(len(table), np.uint16),
        ("x",),
        tuple(f"m{i}" for i in range(len(table))),
    )
    intents = F.intent_probs(pred, batch, tables)
    assert intents is not None
    by_class = CP.class_mass(prob, classes)
    # a candidate whose legal targets carry no mass loses it in the reply
    # split; intent_probs counts that as unassigned. Compare the seven classes.
    assert np.allclose(by_class[..., :7], intents[..., :7], atol=1e-12)
    with pytest.raises(ValueError):
        CP.reply_classes({"cand_move": moves + 100}, table)


def test_buckets_read_turn_and_faced_slots_and_no_label():
    batch, _, _ = case(40, 5)
    batch["turn"][:4] = (1, 1, 2, 9)
    batch["foe_mon"][:4] = ((0, 1), (0, -1), (0, 1), (-1, 1))
    found = CP.reply_buckets(batch)
    assert found[:4].tolist() == [
        CP.BUCKET_TURN1,
        CP.BUCKET_TURN1,
        CP.BUCKET_TWO,
        CP.BUCKET_ONE,
    ]
    rng = np.random.default_rng(0)
    scrambled = {
        name: (rng.permutation(value) if name.startswith(("y_", "m_")) else value)
        for name, value in batch.items()
    }
    bare = {k: v for k, v in batch.items() if not k.startswith(("y_", "m_"))}
    assert np.array_equal(CP.reply_buckets(scrambled), found)
    assert np.array_equal(CP.reply_buckets(bare), found)
    assert np.array_equal(
        CP.reply_classes(bare, intent_table()),
        CP.reply_classes(scrambled, intent_table()),
    )


# --- the joint ------------------------------------------------------------------


@pytest.mark.parametrize("mega", [False, True])
def test_an_all_ones_coupling_is_exactly_the_joint_of_today(mega: bool):
    batch, pred, truth = case(300, 11)
    table = intent_table()
    plain = J.joint_replies(pred, batch, k=12, mega=mega, truth=truth)
    assert plain.n == 300 and not plain.coupled and (plain.tilt == 1).all()
    none = J.joint_replies(
        pred, batch, k=12, mega=mega, truth=truth, coupling=None, move_intent=table
    )
    identity = J.joint_replies(
        pred,
        batch,
        k=12,
        mega=mega,
        truth=truth,
        coupling=CP.PairCoupling.identity(),
        move_intent=table,
    )
    pushed = J.joint_replies(
        pred,
        batch,
        k=12,
        mega=mega,
        truth=truth,
        coupling=OnesThroughChunk(),
        move_intent=table,
    )
    assert same_arrays(plain, none) and same_arrays(plain, identity)
    assert not identity.coupled  # an identity takes the plain path
    assert pushed.coupled and same_arrays(plain, pushed)  # x 1.0 is exact
    # the identity needs nothing of the batch: no candidate ids, no table
    bare = {k: batch[k] for k in ("action_mask", "cand_tmask", "act_mon", "mon_flag")}
    assert same_arrays(
        plain,
        J.joint_replies(
            pred,
            bare,
            k=12,
            mega=mega,
            truth=truth,
            coupling=CP.PairCoupling.identity(),
        ),
    )


@pytest.mark.parametrize("mega", [False, True])
def test_coupled_joint_is_normalised_and_agrees_with_plain_loops(mega: bool):
    batch, pred, truth = case(120, 21)
    table = intent_table()
    made = some_coupling(4)
    classes = CP.reply_classes(batch, table)
    bucket = CP.reply_buckets(batch)
    full = J.joint_replies(
        pred, batch, k=J.MAX_K, mega=mega, truth=truth, coupling=made, move_intent=table
    )
    plain = J.joint_replies(pred, batch, k=J.MAX_K, mega=mega, truth=truth)
    assert full.n == 120 and full.coupled
    assert (full.prob >= 0).all()
    assert np.allclose(full.prob.sum(1)[full.ok], 1.0, atol=1e-12)
    assert np.array_equal(full.kept, plain.kept)  # the plain mass, as before
    assert np.array_equal(full.size, plain.size) and np.array_equal(full.ok, plain.ok)
    first_switch = J.other_reply(C) + 1
    for i in range(120):
        listed, total = brute_force(pred, batch, i, mega)
        tilted = []
        for weight, state, a, b in listed:
            factor = 1.0
            if a >= 0 and b >= 0:
                factor = made.table[bucket[i], classes[i, 0, a], classes[i, 1, b]]
            tilted.append((weight * factor, state, a, b))
        tilted.sort(
            key=lambda x: (
                -x[0],
                x[1],
                x[2] if x[2] >= 0 else FAR,
                x[3] if x[3] >= 0 else FAR,
            )
        )
        mass = sum(x[0] for x in tilted)
        got = as_dict(full, i)
        if mass <= 0:
            assert not full.ok[i] and not got
            continue
        assert len(got) == len(tilted) == int(full.size[i])
        for weight, state, a, b in tilted:
            assert got[(state, a, b)] == pytest.approx(weight / mass, abs=1e-12)
            # both slots to the same bench Pokemon has no entry
            assert not (a >= first_switch and a == b)
        assert full.tilt[i] == pytest.approx(mass / total, rel=1e-12)
        # the order: by probability, the listed order of the plain loops
        order = [(int(full.mega[i, p]), *map(int, full.reply[i, p])) for p in range(3)]
        want = [(state, a, b) for _, state, a, b in tilted[:3]]
        probs = [x[0] for x in tilted[:4]]
        if len(set(probs)) == len(probs):  # no tie among the first: order is forced
            assert order[: len(want)] == want


def mirror_actor(batch: F.Batch, pred: dict[str, np.ndarray]) -> tuple[Any, Any]:
    """The same turns with the actor's two slots renamed."""
    flipped = dict(batch)
    for name in PER_SLOT:
        flipped[name] = np.asarray(batch[name])[:, ::-1].copy()
    turned = {name: np.asarray(value)[:, ::-1].copy() for name, value in pred.items()}
    return flipped, turned


def mirror_faced(batch: F.Batch, pred: dict[str, np.ndarray]) -> tuple[Any, Any]:
    """The same turns with the faced side's two slots renamed."""
    flipped = dict(batch)
    bits = np.asarray(batch["cand_tmask"]).astype(np.int64)
    a, b = (bits >> F.T_FOE_A) & 1, (bits >> F.T_FOE_B) & 1
    rest = bits & ~((1 << F.T_FOE_A) | (1 << F.T_FOE_B))
    flipped["cand_tmask"] = (rest | (b << F.T_FOE_A) | (a << F.T_FOE_B)).astype(
        batch["cand_tmask"].dtype
    )
    flipped["foe_mon"] = np.asarray(batch["foe_mon"])[:, ::-1].copy()
    turned = dict(pred)
    order = np.arange(F.N_TARGET)
    order[F.T_FOE_A], order[F.T_FOE_B] = F.T_FOE_B, F.T_FOE_A
    turned["target"] = np.asarray(pred["target"])[..., order].copy()
    return flipped, turned


def renamed_reply(reply: int) -> int:
    if 0 <= reply < J.other_reply(C):
        column, aim = divmod(reply, F.N_TARGET)
        aim = {F.T_FOE_A: F.T_FOE_B, F.T_FOE_B: F.T_FOE_A}.get(aim, aim)
        return J.move_reply(column, aim)
    return reply


def test_the_coupled_joint_is_symmetric_under_both_renamings():
    batch, pred, _ = case(80, 31)
    table = intent_table()
    made = some_coupling(6)

    def full(b: Any, p: Any, coupling: Any = made) -> J.JointReplies:
        found = J.joint_replies(p, b, k=J.MAX_K, coupling=coupling, move_intent=table)
        assert found.n == 80
        return found

    base = full(batch, pred)
    actor = full(*mirror_actor(batch, pred))
    faced = full(*mirror_faced(batch, pred))
    for i in range(80):
        here = as_dict(base, i)
        there = {(s, b, a): p for (s, a, b), p in as_dict(actor, i).items()}
        assert here.keys() == there.keys()
        assert all(here[key] == pytest.approx(there[key], abs=1e-13) for key in here)
        other = {
            (s, renamed_reply(a), renamed_reply(b)): p
            for (s, a, b), p in as_dict(faced, i).items()
        }
        assert here.keys() == other.keys()
        assert all(here[key] == pytest.approx(other[key], abs=1e-13) for key in here)
    # a table without the symmetry breaks the first equality (test-only: the
    # public constructor refuses such a table)
    lopsided = some_table(6).copy()
    lopsided[:, CP.C_SWITCH, :] *= 3.0
    crooked = CP.PairCoupling(lopsided, "crooked")
    base = full(batch, pred, crooked)
    actor = full(*mirror_actor(batch, pred), crooked)
    differs = 0
    for i in range(80):
        here = as_dict(base, i)
        there = {(s, b, a): p for (s, a, b), p in as_dict(actor, i).items()}
        differs += any(abs(here[key] - there[key]) > 1e-9 for key in here)
    assert differs > 10


def test_one_acting_slot_is_untouched_and_ratios_inside_a_class_pair_stay():
    batch, pred, truth = case(200, 41)
    table = intent_table()
    made = some_coupling(8)
    plain = J.joint_replies(pred, batch, k=J.MAX_K, truth=truth)
    coupled = J.joint_replies(
        pred, batch, k=J.MAX_K, truth=truth, coupling=made, move_intent=table
    )
    one = (batch["act_mon"] >= 0).sum(-1) == 1
    assert one.any() and (~one).any()
    for name in ARRAYS:
        assert np.array_equal(getattr(plain, name)[one], getattr(coupled, name)[one])
    assert (coupled.tilt[one] == 1).all()
    classes = CP.reply_classes(batch, table)
    bucket = CP.reply_buckets(batch)
    checked = 0
    for i in np.nonzero(~one)[0][:60]:
        before, after = as_dict(plain, i), as_dict(coupled, i)
        assert before.keys() == after.keys()
        blocks: dict[tuple[int, int], list[float]] = {}
        for (_, a, b), p in before.items():
            if p <= 0:
                continue
            pair = (int(classes[i, 0, a]), int(classes[i, 1, b]))
            blocks.setdefault(pair, []).append(after[(0, a, b)] / p)
        for (c, d), ratios in blocks.items():
            # one ratio per class pair: R[c, d] over the row's coupled total
            want = made.table[bucket[i], c, d] / coupled.tilt[i]
            assert np.allclose(ratios, want, rtol=1e-9)
            checked += 1
    assert checked > 100


def test_a_coupling_that_cannot_be_applied_is_a_failure_never_the_plain_list():
    batch, pred, _ = case(20, 51)
    table = intent_table()
    made = some_coupling(2)
    assert J.joint_replies(pred, batch, coupling=made, move_intent=table).n == 20
    before = dict(J.COUNTERS)
    lacking = [
        ({k: v for k, v in batch.items() if k != "cand_move"}, table),
        ({k: v for k, v in batch.items() if k != "turn"}, table),
        ({k: v for k, v in batch.items() if k != "foe_mon"}, table),
        (batch, None),
        (batch, table[:2]),  # candidate ids outside the table
        (batch, np.zeros((6, 4), dtype=np.int8)),
    ]
    for held, intents in lacking:
        found = J.joint_replies(pred, held, coupling=made, move_intent=intents)
        assert found.n == 0 and not found.coupled
    other_scheme = CP.PairCoupling(some_table(2), "x", class_scheme="intent9")
    assert J.joint_replies(pred, batch, coupling=other_scheme, move_intent=table).n == 0
    assert J.joint_replies(pred, batch, coupling=object(), move_intent=table).n == 0
    assert sum(J.COUNTERS.values()) - sum(before.values()) == len(lacking) + 2
    assert any(name.startswith("reply_maps") for name in CP.COUNTERS)
    # maps handed over ready-made need neither the ids nor the table
    ready = {k: v for k, v in batch.items() if k not in ("cand_move", "turn")}
    ready[CP.KEY_CLASS] = CP.reply_classes(batch, table)
    ready[CP.KEY_BUCKET] = CP.reply_buckets(batch)
    direct = J.joint_replies(pred, batch, coupling=made, move_intent=table)
    assert same_arrays(direct, J.joint_replies(pred, ready, coupling=made))


def test_chunking_changes_nothing_and_inputs_are_not_changed():
    batch, pred, truth = case(90, 61)
    table = intent_table()
    made = some_coupling(3)
    kept_batch = {name: np.array(value) for name, value in batch.items()}
    kept_pred = {name: np.array(value) for name, value in pred.items()}
    whole = J.joint_replies(
        pred, batch, k=10, mega=True, truth=truth, coupling=made, move_intent=table
    )
    small = J.joint_replies(
        pred,
        batch,
        k=10,
        mega=True,
        truth=truth,
        coupling=made,
        move_intent=table,
        chunk_bytes=1,
    )
    assert whole.n == 90 and same_arrays(whole, small)
    assert all(np.array_equal(batch[name], kept_batch[name]) for name in batch)
    assert all(np.array_equal(pred[name], kept_pred[name]) for name in pred)
    assert np.array_equal(made.table, some_table(3))


def test_true_replies_mark_a_protect_family_candidate():
    batch, _, _ = case(6, 71)
    batch["cand_flag"][:, :, 1] |= F.CAND_PROTECT
    batch["act_mon"][:] = (0, 1)
    batch["y_action"][:] = (
        (0, 1),
        (1, 0),
        (C, 1),
        (F.switch_index(2, C), 1),
        (1, 1),
        (0, 0),
    )
    batch["y_target"][:] = 0
    truth = J.true_replies(batch)
    assert truth[E.INTENT_PROTECT].tolist() == [
        [False, True],
        [True, False],
        [False, True],
        [False, True],
        [True, True],
        [False, False],
    ]


# --- the masses and the fit -----------------------------------------------------


def test_consistent_replies_follow_the_labels():
    batch, _, _ = case(4, 81)
    for name in ("y_action", "y_target"):
        batch[name][:] = -1
    batch["y_set"][:] = 0
    batch["act_mon"][:] = (0, 1)
    batch["action_mask"][:] = 0
    batch["action_mask"][
        :, :, [0, 1, C, F.switch_index(2, C), F.switch_index(3, C)]
    ] = 1
    foes = (1 << F.T_FOE_A) | (1 << F.T_FOE_B)
    batch["cand_tmask"][:, :, 0] = foes
    batch["cand_tmask"][:, :, 1] = 1 << F.T_AUTO
    # row 0: a seen aimed move with a certain target / a seen switch
    batch["y_action"][0] = (0, F.switch_index(3, C))
    batch["y_target"][0, 0] = F.T_FOE_B
    batch["y_set"][0, 0, 0] = batch["y_set"][0, 1, F.switch_index(3, C)] = 1
    # row 1: a seen aimed move whose target is not certain / a hidden slot
    batch["y_action"][1, 0] = 0
    batch["y_set"][1, 0, 0] = 1
    batch["y_set"][1, 1, [0, 1, C]] = 1
    # row 2: a switch whose destination is not known / a forced slot (empty)
    batch["y_set"][2, 0, [F.switch_index(2, C), F.switch_index(3, C)]] = 1
    # row 3: a set entry outside the mask is ignored
    batch["y_set"][3, 0, 2] = 1
    found = CP.consistent_replies(batch)
    assert found.shape == (4, 2, J.reply_size(C))

    def replies(i: int, s: int) -> list[int]:
        return np.nonzero(found[i, s])[0].tolist()

    assert replies(0, 0) == [J.move_reply(0, F.T_FOE_B)]
    assert replies(0, 1) == [J.switch_reply(3, C)]
    assert replies(1, 0) == [J.move_reply(0, F.T_FOE_A), J.move_reply(0, F.T_FOE_B)]
    assert replies(1, 1) == [
        J.move_reply(0, F.T_FOE_A),
        J.move_reply(0, F.T_FOE_B),
        J.move_reply(1, F.T_AUTO),
        J.other_reply(C),
    ]
    assert replies(2, 0) == [J.switch_reply(2, C), J.switch_reply(3, C)]
    assert replies(2, 1) == [] and replies(3, 0) == []


def test_pair_masses_are_the_class_sums_of_the_joint():
    batch, pred, _ = case(80, 91)
    table = intent_table()
    prob, _ = J.slot_replies(pred, batch)
    classes = CP.reply_classes(batch, table)
    masses = CP.pair_masses(prob, classes)
    plain = J.joint_replies(pred, batch, k=J.MAX_K)
    two = (batch["act_mon"] >= 0).all(-1)
    for i in np.nonzero(two & plain.ok)[0][:30]:
        want = np.zeros((8, 8))
        for (_, a, b), p in as_dict(plain, i).items():
            want[classes[i, 0, a], classes[i, 1, b]] += p
        assert np.allclose(masses["model"][i], want, atol=1e-12)
        assert masses["kept"][i] == pytest.approx(plain.kept[i], rel=1e-12)
    # restricted to everything, the seen mass is the model mass
    everything = np.ones(prob.shape, dtype=bool)
    again = CP.pair_masses(prob, classes, everything)
    assert np.allclose(again["seen"], again["model"])


def draw_rows(
    n: int, truth: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows with random class marginals whose true pair is drawn from
    ``a x b x truth``: (model, seen, the drawn cell)."""
    rng = np.random.default_rng(seed)
    a = rng.dirichlet(np.full(8, 0.7), n)
    b = rng.dirichlet(np.full(8, 0.7), n)
    model = a[:, :, None] * b[:, None, :]
    tilted = model * truth[None]
    tilted /= tilted.sum((1, 2), keepdims=True)
    flat = tilted.reshape(n, 64)
    cell = (flat.cumsum(1) > rng.random((n, 1))).argmax(1)
    seen = np.zeros((n, 64))
    seen[np.arange(n), cell] = model.reshape(n, 64)[np.arange(n), cell]
    return model, seen.reshape(n, 8, 8), cell


def test_the_fit_recovers_a_planted_interaction_and_rejects_none():
    planted = some_table(12, spread=0.5)[1]
    model, seen, _ = draw_rows(40000, planted, 1)
    tilt, record = CP.fit_tilt(model, seen, None, kappa=1.0, tolerance=1e-7)
    assert record["converged"] and CP.is_symmetric(tilt)
    mix = CP.class_mix(model)
    found, scale = CP.normal_form(tilt, mix)
    wanted, _ = CP.normal_form(planted, mix)
    assert np.abs(np.log(found) - np.log(wanted)).max() < 0.15
    assert np.allclose(found @ mix, 1.0, atol=1e-9)  # unit margins
    assert np.allclose(scale[:, None] * scale[None, :] * found, tilt, rtol=1e-9)
    assert CP.is_symmetric(found)  # exactly, both ways

    size = 8000
    battle = np.arange(size) // 6
    bucket = np.full(size, CP.BUCKET_TWO)
    grid = (20.0, 200.0)
    fit = CP.fit_coupling(model[:size], seen[:size], None, bucket, battle, kappas=grid)
    assert fit["verdict"] == CP.VERDICT_TAKEN and fit["folds_passed"] == 10
    assert fit["folds"] == 10 and len(fit["fold_gain"]) == 10
    assert fit["table"].shape == (3, 8, 8) and CP.is_symmetric(fit["table"])
    assert np.array_equal(fit["table"][0], fit["table"][1]) or fit["bucketed"]
    assert (fit["table"][CP.BUCKET_ONE] == 1).all()  # no row there: ones

    flat_model, flat_seen, _ = draw_rows(size, np.ones((8, 8)), 2)
    none = CP.fit_coupling(flat_model, flat_seen, None, bucket, battle, kappas=grid)
    assert none["verdict"] == CP.VERDICT_NOT_TAKEN and none["folds_passed"] == 0
    assert np.abs(np.log(none["table"])).max() < 0.15
    # same inputs and seed: the same answer
    again = CP.fit_coupling(flat_model, flat_seen, None, bucket, battle, kappas=grid)
    assert np.array_equal(none["table"], again["table"])
    assert np.array_equal(none["fold_gain"], again["fold_gain"])


def test_shrinkage_and_rows_that_say_nothing():
    planted = some_table(13)[0]
    model, seen, _ = draw_rows(3000, planted, 3)
    # a class nobody has: nothing expected, nothing observed -> exactly 1
    model[:, CP.C_UNASSIGNED, :] = 0
    model[:, :, CP.C_UNASSIGNED] = 0
    seen[:, CP.C_UNASSIGNED, :] = 0
    seen[:, :, CP.C_UNASSIGNED] = 0
    keep = seen.sum((1, 2)) > 0
    model, seen = model[keep], seen[keep]
    tilt, _ = CP.fit_tilt(model, seen, None, kappa=5.0)
    assert (tilt[CP.C_UNASSIGNED] == 1).all() and (tilt[:, CP.C_UNASSIGNED] == 1).all()
    huge, _ = CP.fit_tilt(model, seen, None, kappa=1e9)
    assert np.abs(np.log(huge)).max() < 1e-4
    spans = [
        np.abs(np.log(CP.fit_tilt(model, seen, None, kappa=k)[0])).max()
        for k in (1.0, 10.0, 100.0, 1000.0)
    ]
    assert spans == sorted(spans, reverse=True)
    # a row whose set allows everything adds nothing
    extra_model = np.concatenate([model, model[:500]])
    extra_seen = np.concatenate([seen, model[:500]])
    wider, _ = CP.fit_tilt(extra_model, extra_seen, None, kappa=5.0)
    assert np.allclose(wider, tilt, rtol=1e-6)
    # weights: a row of weight 2 is that row twice
    weight = np.ones(len(model))
    weight[:300] = 2.0
    weighted, _ = CP.fit_tilt(model, seen, weight, kappa=5.0)
    doubled, _ = CP.fit_tilt(
        np.concatenate([model, model[:300]]),
        np.concatenate([seen, seen[:300]]),
        None,
        kappa=5.0,
    )
    assert np.allclose(weighted, doubled, rtol=1e-6)
    # no row at all: ones
    empty, record = CP.fit_tilt(model[:0], seen[:0], None, kappa=5.0)
    assert (empty == 1).all() and record["rows"] == 0


def test_selection_by_each_slots_own_class_leaves_the_normal_form():
    """What justifies discarding D: rows deleted by a rule that depends on
    each slot's own class change T by a per-class factor and R not at all."""
    planted = some_table(14, spread=0.4)[2]
    model, seen, cell = draw_rows(80000, planted, 4)
    rng = np.random.default_rng(5)
    visible = np.array([1.0, 1.0, 0.45, 0.45, 0.6, 0.8, 0.7, 0.5])  # by class
    c, d = np.divmod(cell, 8)
    kept = rng.random(len(cell)) < visible[c] * visible[d]
    mix = CP.class_mix(model)
    every, _ = CP.fit_tilt(model, seen, None, kappa=1.0, tolerance=1e-7)
    part, _ = CP.fit_tilt(model[kept], seen[kept], None, kappa=1.0, tolerance=1e-7)
    full_form, _ = CP.normal_form(every, mix)
    part_form, scale = CP.normal_form(part, mix)
    assert np.abs(np.log(part) - np.log(every)).max() > 0.5  # T moved a lot
    assert np.abs(np.log(part_form) - np.log(full_form)).max() < 0.2
    assert scale.max() / scale.min() > 1.5  # and D took the selection


def test_only_the_rows_given_are_read_and_the_folds_hold_whole_battles():
    planted = some_table(15)[1]
    model, seen, _ = draw_rows(4000, planted, 6)
    bucket = np.where(np.arange(4000) % 5 == 0, CP.BUCKET_ONE, CP.BUCKET_TWO)
    battle = np.arange(4000) // 4
    table, parts = CP.fit_tables(model, seen, None, bucket, 20.0, False)
    assert set(parts) == {"pooled", "later_one"}
    assert np.array_equal(table[0], table[1]) and not np.array_equal(table[1], table[2])
    split, parts = CP.fit_tables(model, seen, None, bucket, 20.0, True)
    assert set(parts) == {"turn1", "later_two", "later_one"}
    assert (split[CP.BUCKET_TURN1] == 1).all()  # no turn-1 row: ones
    fold, used = CP.fold_index(battle, 10, 7)
    assert used == 10
    for game in range(0, 1000, 37):
        assert len(set(fold[battle == game].tolist())) == 1
    assert CP.fold_index(battle[:4], 10, 7)[1] == 0  # one battle: no folds
    lonely = CP.fit_coupling(
        model[:4], seen[:4], None, bucket[:4], battle[:4], kappas=(20.0,)
    )
    assert lonely["verdict"] == CP.VERDICT_NOT_TAKEN and lonely["folds"] == 0
    gain = CP.table_gain(table, model, seen, bucket)
    assert gain.shape == (4000,) and np.isfinite(gain).all()
    flat = CP.table_gain(np.ones((3, 8, 8)), model, seen, bucket)
    assert np.abs(flat).max() < 1e-12


# --- the artifact ---------------------------------------------------------------


@pytest.fixture(scope="module")
def fz() -> F.Featurizer:
    repertoire = F.Repertoire()
    repertoire.add("alpha", "one", 3.0)
    return F.Featurizer.build(repertoire)


def tiny_batch(fz: F.Featurizer, n: int = 16) -> F.Batch:
    rng = np.random.default_rng(0)
    width = F.N_CAND_DEFAULT
    batch = {
        name: np.zeros((n, *spec.shape), dtype=spec.dtype)
        for name, spec in fz.layout().items()
    }
    for name in ("act_mon", "foe_mon", "y_kind", "y_action", "y_target", "y_mega"):
        batch[name][:] = -1
    batch["y_intent"][:] = -1
    other = F.other_index(width)
    for i in range(n):
        batch["turn"][i] = 1 + i % 3
        batch["foe_mon"][i] = (6, 7)
        for slot in range(2):
            batch["act_mon"][i, slot] = slot
            batch["mon_id"][i, slot, F.ID_KEY] = 5 + slot
            for column in range(3):
                batch["cand_move"][i, slot, column] = 20 + column + 3 * slot
                batch["cand_flag"][i, slot, column] = F.CAND_VALID | F.CAND_DAMAGING
                batch["cand_tmask"][i, slot, column] = 0b00011
            batch["cand_tmask"][i, slot, other] = 0b11111
            legal = [0, 1, 2, other, F.switch_index(2 + slot, width)]
            batch["action_mask"][i, slot, legal] = 1
            action = int(rng.choice(legal[:3]))
            batch["y_action"][i, slot] = action
            batch["y_set"][i, slot, action] = 1
            batch["y_target"][i, slot] = rng.integers(0, 2)
    batch["m_weight"] = np.ones(n, dtype=np.float32)
    return batch


def features_only(batch: F.Batch) -> F.Batch:
    return {k: v for k, v in batch.items() if not k.startswith(("y_", "m_"))}


def write(path: Path, fz: F.Featurizer, table: Any, **more: Any) -> Path:
    A.save_artifact(
        path,
        kind=A.KIND_TABLE,
        name=table.name,
        featurizer=fz,
        predictor_payload=table.to_payload(),
        extra={"dataset_tag": "unit"},
        **more,
    )
    return path


def test_an_artifact_without_a_coupling_is_the_file_of_before(tmp_path: Path, fz):
    table = T.FlagsTable.fit(tiny_batch(fz), featurizer=fz)
    old = write(tmp_path / "old.pt", fz, table)
    explicit = write(tmp_path / "explicit.pt", fz, table, coupling=None)
    raw = torch.load(old, map_location="cpu", weights_only=True)
    assert set(raw) == {
        "format",
        "version",
        "kind",
        "name",
        "featurizer",
        "dex_signature",
        "predictor",
        "extra",
    }
    assert raw["version"] == 1 == A.VERSION
    first, second = A.read_artifact(old), A.read_artifact(explicit)
    assert first.keys() == second.keys() and "coupling" not in first

    def equal(a: Any, b: Any) -> bool:
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, list):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        if isinstance(a, np.ndarray):
            return a.dtype == b.dtype and np.array_equal(a, b)
        if isinstance(a, torch.Tensor):
            return bool(torch.equal(a, b))
        return a == b

    assert equal(first, second)
    loaded = A.load_predictor(old)
    assert type(loaded) is A.LoadedPredictor and loaded.coupling is None
    assert len(loaded) == 5 and "coupling" not in loaded.meta
    assert loaded.meta["version"] == 1


def test_a_coupled_artifact_is_version_two_and_predicts_as_its_source(
    tmp_path: Path, fz
):
    batch = tiny_batch(fz)
    table = T.FlagsTable.fit(batch, featurizer=fz)
    made = CP.PairCoupling.build(
        some_table(21), name="tiny_pair", fitted_after=CP.fitted_after("table", None)
    )
    source = write(tmp_path / "source.pt", fz, table)
    target = write(tmp_path / "coupled.pt", fz, table, coupling=made)
    stored = A.read_artifact(target)
    assert stored["version"] == 2 == A.VERSION_COUPLED
    assert set(stored) - set(A.read_artifact(source)) == {"coupling"}
    loaded, plain = A.load_predictor(target), A.load_predictor(source)
    assert isinstance(loaded, A.LoadedPredictor) and len(loaded) == 5
    predictor, featurizer, kind, name, meta = loaded  # still unpacks as before
    assert kind == A.KIND_TABLE and name == table.name and meta["version"] == 2
    assert loaded.coupling.same_as(made) and loaded.coupling.info == made.info
    assert meta["coupling"]["name"] == "tiny_pair"
    feats = F.sheet_unknown_as_closed(features_only(batch))
    ours, theirs = predictor.predict(feats), plain.predictor.predict(feats)
    for head in ("action", "target", "mega"):
        assert np.array_equal(ours[head], theirs[head])
    # the payload form is accepted too, and a damaged one never reaches a file
    write(tmp_path / "payload.pt", fz, table, coupling=made.to_payload())
    assert A.load_predictor(tmp_path / "payload.pt").coupling.same_as(made)
    broken = {**made.to_payload(), "table": np.zeros((3, 8, 8))}
    with pytest.raises(ValueError):
        write(tmp_path / "broken.pt", fz, table, coupling=broken)
    assert not (tmp_path / "broken.pt").exists()
    # the joint: without the coupling the source's, with it one application
    intents = featurizer.tables.move_intent
    assert same_arrays(
        J.joint_replies(ours, feats, k=8), J.joint_replies(theirs, feats, k=8)
    )
    coupled = J.joint_replies(
        ours, feats, k=8, coupling=loaded.coupling, move_intent=intents
    )
    direct = J.joint_replies(theirs, feats, k=8, coupling=made, move_intent=intents)
    assert coupled.n == len(batch["turn"]) and coupled.coupled
    assert same_arrays(coupled, direct)


def test_version_and_content_must_agree_and_a_misfit_is_an_error(tmp_path: Path, fz):
    table = T.FlagsTable.fit(tiny_batch(fz), featurizer=fz)
    made = CP.PairCoupling.build(some_table(22), name="p")
    good = write(tmp_path / "good.pt", fz, table, coupling=made)
    raw = torch.load(good, map_location="cpu", weights_only=True)

    def store(name: str, **change: Any) -> Path:
        document = {**raw, **change}
        for key, value in change.items():
            if value is None and key == "coupling_dropped":
                document.pop("coupling")
        document.pop("coupling_dropped", None)
        torch.save(document, tmp_path / name)
        return tmp_path / name

    with pytest.raises(ValueError, match="must carry a coupling"):
        A.read_artifact(store("two_without.pt", coupling_dropped=None))
    with pytest.raises(ValueError, match="must not carry a coupling"):
        A.read_artifact(store("one_with.pt", version=1))
    with pytest.raises(ValueError, match="version"):
        A.read_artifact(store("three.pt", version=3))
    # a coupling fitted after another predictor state is an error, not dropped
    other = CP.PairCoupling.build(
        some_table(22), name="p", fitted_after=CP.fitted_after("oppnet", None)
    )
    write(tmp_path / "misfit.pt", fz, table, coupling=other)
    with pytest.raises(ValueError, match="does not fit"):
        A.load_predictor(tmp_path / "misfit.pt")
    before = sum(A.COUNTERS.values())
    assert A.try_load_predictor(tmp_path / "misfit.pt") is None
    damaged = dict(raw["coupling"])
    damaged["format"] = "something else"
    assert A.try_load_predictor(store("damaged.pt", coupling=damaged)) is None
    assert sum(A.COUNTERS.values()) == before + 2
    assert A.try_load_predictor(good) is not None


def test_the_coupling_module_loads_no_torch():
    code = (
        "import sys\n"
        "from vgc_bench.src.oppmodel import coupling\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
        "print(coupling.N_ORBIT)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "29"


# --- house rules ----------------------------------------------------------------


def test_library_names_nothing_from_the_dex_and_has_no_bare_assert():
    vocab = F.Vocab.build()
    known = set(E.moves_dex()) | set(E.pokedex())
    known |= set(vocab.items[3:]) | set(vocab.abilities[3:])
    for name in ("coupling.py", "joint.py"):
        path = ROOT / "vgc_bench/src/oppmodel" / name
        tree = ast.parse(path.read_text(encoding="utf-8"))
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
        assert not [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assert)]
    # what a runtime calls in coupling.py holds no raise of its own
    tree = ast.parse((ROOT / "vgc_bench/src/oppmodel/coupling.py").read_text("utf-8"))
    never = {"reply_maps", "weights", "fits", "class_index", "describe", "padded"}
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in never:
            seen.add(node.name)
            handlers = [n for n in ast.walk(node) if isinstance(n, ast.Try)]
            raises = [n for n in ast.walk(node) if isinstance(n, ast.Raise)]
            assert not raises or handlers, node.name
    assert seen == never
