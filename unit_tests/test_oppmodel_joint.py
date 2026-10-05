"""Joint replies of two slots: construction, exclusions, order, truth, failures.

Inputs are synthetic batches and hand-made marginals written inline. One test
at the end reads the built dataset and a fitted table and skips when those are
absent (they are git-ignored).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vgc_bench.src.oppmodel import events as E
from vgc_bench.src.oppmodel import features as F
from vgc_bench.src.oppmodel import joint as J

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "results_oppmodel" / "v1_ondisk"
FITTED = ROOT / "results_oppmodel" / "tables_v1"

C = 3  # candidates per slot in the synthetic batches
A = F.action_size(C)
OTHER = F.other_index(C)
FOES = (1 << F.T_FOE_A) | (1 << F.T_FOE_B)
AUTO = 1 << F.T_AUTO
SW2, SW3 = J.switch_reply(2, C), J.switch_reply(3, C)
FAR = 10**6  # sorts "no action" (-1) after every reply, as the table does


# --- synthetic batches ----------------------------------------------------------


def blank(n: int) -> F.Batch:
    """``n`` examples with nothing on the field, labels included."""
    batch = {
        name: np.zeros((n, *spec.shape), dtype=spec.dtype)
        for name, spec in F.layout(C).items()
    }
    for name in ("act_mon", "foe_mon", "y_kind", "y_action", "y_target", "y_mega"):
        batch[name][:] = -1
    return batch


def put(
    batch: F.Batch,
    i: int,
    slot: int,
    *,
    targets: tuple[int, ...] = (FOES, AUTO),
    bench: tuple[int, ...] = (),
    mega: bool = False,
    flags: tuple[int, ...] = (),
) -> None:
    """Slot ``slot`` of example ``i``: one candidate per target mask, a bench."""
    batch["act_mon"][i, slot] = slot
    batch["mon_flag"][i, slot, F.FLAG_PRESENT] = 1
    batch["mon_flag"][i, slot, F.FLAG_MEGA_POSSIBLE] = int(mega)
    for column, bits in enumerate(targets):
        extra = flags[column] if column < len(flags) else 0
        batch["cand_flag"][i, slot, column] = F.CAND_VALID | extra
        batch["cand_tmask"][i, slot, column] = bits
        batch["action_mask"][i, slot, column] = 1
    batch["action_mask"][i, slot, OTHER] = 1
    batch["cand_tmask"][i, slot, OTHER] = (1 << F.N_TARGET) - 1
    for roster in bench:
        batch["switch_mask"][i, slot, roster] = 1
        batch["action_mask"][i, slot, F.switch_index(roster, C)] = 1


def empty_prediction(batch: F.Batch) -> dict[str, np.ndarray]:
    n = len(batch["turn"])
    return {
        "action": np.zeros((n, 2, A)),
        "target": np.zeros((n, 2, C + 1, F.N_TARGET)),
        "mega": np.zeros((n, 2)),
    }


def two_slots() -> tuple[F.Batch, dict[str, np.ndarray]]:
    """One example, both slots on the field, marginals written by hand.

    Slot a: an aimed move (0.5: 0.75 at foe a, 0.25 at foe b), a move that is
    not aimed (0.25), OTHER (0.1), a switch to roster 2 (0.15), to 3 (0).
    Slot b: an aimed move (0.5: 0.6 / 0.4), OTHER (0.1), a switch to roster 2
    (0.25), to 3 (0.15).
    """
    batch = blank(1)
    put(batch, 0, 0, targets=(FOES, AUTO), bench=(2, 3))
    put(batch, 0, 1, targets=(FOES,), bench=(2, 3))
    pred = empty_prediction(batch)
    pred["action"][0, 0, [0, 1, OTHER, F.switch_index(2, C)]] = (0.5, 0.25, 0.1, 0.15)
    pred["target"][0, 0, 0, [F.T_FOE_A, F.T_FOE_B]] = (0.75, 0.25)
    pred["target"][0, 0, 1, F.T_AUTO] = 1.0
    pred["action"][0, 1, 0] = 0.5
    pred["action"][0, 1, OTHER] = 0.1
    pred["action"][0, 1, [F.switch_index(2, C), F.switch_index(3, C)]] = (0.25, 0.15)
    pred["target"][0, 1, 0, [F.T_FOE_A, F.T_FOE_B]] = (0.6, 0.4)
    return batch, pred


def as_dict(made: J.JointReplies, row: int = 0) -> dict[tuple[int, int, int], float]:
    """(mega state, reply a, reply b) -> probability, of the listed replies."""
    out: dict[tuple[int, int, int], float] = {}
    for position in range(made.k):
        a, b = (int(v) for v in made.reply[row, position])
        if J.UNKNOWN in (a, b):
            break
        out[(int(made.mega[row, position]), a, b)] = float(made.prob[row, position])
    return out


def brute_force(
    pred: dict[str, np.ndarray], batch: F.Batch, i: int, with_mega: bool
) -> tuple[list[tuple[float, int, int, int]], float]:
    """Every possible joint reply of example ``i`` by plain loops, in order.

    Returns ``[(weight, mega state, reply a, reply b), ...]`` sorted as the
    module promises, and the total weight (the mass kept).
    """
    norm = F.normalize_prediction(pred, batch)
    legal = F.expand_target_mask(batch["cand_tmask"])
    mask = np.asarray(batch["action_mask"]).astype(bool)
    slots: list[list[tuple[int, float, bool]]] = []
    can: list[bool] = []
    q: list[float] = []
    for s in range(2):
        row = int(batch["act_mon"][i, s])
        if row < 0:
            slots.append([(J.NO_ACTION, 1.0, False)])
            can.append(False)
            q.append(0.0)
            continue
        found: list[tuple[int, float, bool]] = []
        for c in range(C):
            for t in range(F.N_TARGET):
                if mask[i, s, c] and legal[i, s, c, t]:
                    weight = norm["action"][i, s, c] * norm["target"][i, s, c, t]
                    found.append((J.move_reply(c, t), float(weight), True))
        found.append((J.other_reply(C), float(norm["action"][i, s, OTHER]), True))
        for r in range(F.N_ROSTER):
            if mask[i, s, F.switch_index(r, C)]:
                weight = norm["action"][i, s, F.switch_index(r, C)]
                found.append((J.switch_reply(r, C), float(weight), False))
        slots.append(found)
        possible = bool(batch["mon_flag"][i, row, F.FLAG_MEGA_POSSIBLE]) and with_mega
        can.append(possible)
        q.append(float(np.clip(pred["mega"][i, s], 0, 1)) if possible else 0.0)
    scale = [(1 - q[0]) * (1 - q[1]), q[0] * (1 - q[1]), (1 - q[0]) * q[1]]
    out: list[tuple[float, int, int, int]] = []
    for state in (J.MEGA_NONE, J.MEGA_A, J.MEGA_B):
        if state != J.MEGA_NONE and not can[state - 1]:
            continue
        for a, pa, a_moves in slots[0]:
            for b, pb, b_moves in slots[1]:
                if a >= J.other_reply(C) + 1 and a == b:
                    continue  # both to the same bench Pokemon
                if (state == J.MEGA_A and not a_moves) or (
                    state == J.MEGA_B and not b_moves
                ):
                    continue  # a switch cannot carry a Mega Evolution
                out.append((pa * pb * scale[state], state, a, b))
    out.sort(
        key=lambda x: (
            -x[0],
            x[1],
            x[2] if x[2] >= 0 else FAR,
            x[3] if x[3] >= 0 else FAR,
        )
    )
    return out, float(sum(x[0] for x in out))


def random_case(
    n: int, seed: int
) -> tuple[F.Batch, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Random legal masks, predictions and true replies (some not possible)."""
    rng = np.random.default_rng(seed)
    batch = blank(n)
    for i in range(n):
        both = rng.random() < 0.8
        bench = tuple(
            int(r) for r in np.nonzero(rng.random(F.N_ROSTER) < 0.4)[0] if r > 1
        )
        for slot in range(2):
            if not both and slot == int(rng.integers(2)):
                continue
            count = int(rng.integers(1, C + 1))
            masks = tuple(
                int(rng.choice([FOES, AUTO, FOES | 1 << F.T_ALLY, FOES | AUTO]))
                for _ in range(count)
            )
            put(batch, i, slot, targets=masks, bench=bench, mega=rng.random() < 0.5)
        if (batch["act_mon"][i] < 0).all():
            put(batch, i, 0)
    pred = {
        "action": rng.random((n, 2, A)) ** 3,
        "target": rng.random((n, 2, C + 1, F.N_TARGET)),
        "mega": rng.random((n, 2)),
    }
    # exact zeros and exact ties, so the order of equal replies is exercised
    pred["action"][rng.random((n, 2, A)) < 0.2] = 0.0
    pred["action"][: n // 4] = np.round(pred["action"][: n // 4], 1)
    pred["target"][: n // 4] = 1.0
    # truths: mostly a legal reply of each slot, sometimes anything at all
    size = J.reply_size(C)
    legal = J.slot_replies(pred, batch)[1]
    reply = rng.integers(-1, size, size=(n, 2))
    for i in range(n):
        for slot in range(2):
            if rng.random() < 0.85:
                options = np.nonzero(legal[i, slot])[0]
                reply[i, slot] = rng.choice(options) if options.size else J.NO_ACTION
    truth = {"reply": reply, "mega": rng.integers(0, 3, size=n)}
    return batch, pred, truth


# --- the reply index ------------------------------------------------------------


def test_reply_indices_and_their_description():
    assert J.reply_size(C) == C * F.N_TARGET + 1 + F.N_ROSTER == 22
    assert J.move_reply(2, F.T_FOE_B) == 2 * F.N_TARGET + F.T_FOE_B
    assert J.other_reply(C) == 15 and J.switch_reply(0, C) == 16 and SW3 == 19
    assert J.describe_reply(J.move_reply(1, F.T_ALLY), C) == {
        "kind": E.KIND_MOVE,
        "action": 1,
        "target": E.TARGET_ALLY,
    }
    assert J.describe_reply(15, C) == {"kind": J.KIND_OTHER, "action": OTHER}
    assert J.describe_reply(SW2, C) == {
        "kind": E.KIND_SWITCH,
        "action": F.switch_index(2, C),
        "roster": 2,
    }
    for bad in (J.NO_ACTION, J.UNKNOWN, 22, "x", None):
        assert J.describe_reply(bad, C) is None  # type: ignore[arg-type]


def test_slot_replies_split_a_move_over_its_targets_and_keep_other_whole():
    batch, pred = two_slots()
    pred["target"][0, 0, OTHER] = 0.2  # OTHER's target row is not read
    pred["action"][0, 0, 2] = 0.5  # a candidate that is not legal: removed
    prob, legal = J.slot_replies(pred, batch)
    assert prob.shape == legal.shape == (1, 2, J.reply_size(C))
    a = prob[0, 0]
    assert a[J.move_reply(0, F.T_FOE_A)] == pytest.approx(0.375)
    assert a[J.move_reply(0, F.T_FOE_B)] == pytest.approx(0.125)
    assert a[J.move_reply(1, F.T_AUTO)] == pytest.approx(0.25)
    assert a[J.other_reply(C)] == pytest.approx(0.1)
    assert a[SW2] == pytest.approx(0.15) and a[SW3] == 0.0
    assert a.sum() == pytest.approx(1.0) and prob[0, 1].sum() == pytest.approx(1.0)
    # possible: two targets of the aimed move, one of the other, OTHER, two switches
    assert int(legal[0, 0].sum()) == 6 and int(legal[0, 1].sum()) == 5
    assert legal[0, 0, SW3] and not legal[0, 0, J.switch_reply(4, C)]
    assert not legal[0, 0, J.move_reply(2, F.T_FOE_A)]
    assert (prob[~legal] == 0).all()


# --- the joint: products, exclusions, renormalisation, the truth ---------------


def test_joint_probabilities_multiply_and_the_same_bench_pokemon_is_excluded():
    batch, pred = two_slots()
    made = J.joint_replies(pred, batch, k=64)
    assert made.n == 1 and made.ok[0] and not made.with_mega
    found = as_dict(made)
    # 6 x 5 replies, minus both to roster 2 and both to roster 3
    assert int(made.size[0]) == len(found) == 28
    kept = 1.0 - 0.15 * 0.25  # the product's mass on "both to roster 2"
    assert made.kept[0] == pytest.approx(kept)
    a_first, b_first = J.move_reply(0, F.T_FOE_A), J.move_reply(0, F.T_FOE_A)
    assert found[(J.MEGA_NONE, a_first, b_first)] == pytest.approx(0.375 * 0.3 / kept)
    assert found[(J.MEGA_NONE, a_first, SW2)] == pytest.approx(0.375 * 0.25 / kept)
    assert found[(J.MEGA_NONE, SW2, SW3)] == pytest.approx(0.15 * 0.15 / kept)
    assert found[(J.MEGA_NONE, J.other_reply(C), J.other_reply(C))] == pytest.approx(
        0.01 / kept
    )
    # the two exclusions of the same bench Pokemon, and nothing illegal
    assert (J.MEGA_NONE, SW2, SW2) not in found
    assert (J.MEGA_NONE, SW3, SW3) not in found
    assert all(state == J.MEGA_NONE for state, _, _ in found)
    # renormalised: the possible replies hold everything
    assert sum(found.values()) == pytest.approx(1.0)
    assert made.mass()[0] == pytest.approx(1.0)
    # most probable first; the first one is the product of the two modes
    assert (np.diff(made.prob[0, :28]) <= 1e-15).all()
    assert tuple(made.reply[0, 0]) == (a_first, b_first)
    # padding after the last possible reply
    assert (made.reply[0, 28:] == J.UNKNOWN).all() and (made.prob[0, 28:] == 0).all()


def test_the_rank_of_the_truth_is_its_place_in_the_list():
    batch, pred = two_slots()
    listed = J.joint_replies(pred, batch, k=64)
    order = [tuple(int(v) for v in pair) for pair in listed.reply[0, :28]]
    for place in (0, 1, 5, 17, 27):
        truth = {"reply": np.array([order[place]])}
        made = J.joint_replies(pred, batch, k=2, truth=truth)
        assert int(made.rank[0]) == place
        assert made.prob_true[0] == pytest.approx(listed.prob[0, place])
        assert int(made.above[0]) <= place <= int(made.above[0] + made.ties[0])
    # a reply that is not possible has no rank: the excluded pair, an illegal
    # move, a missing slot, an unknown action
    for pair in (
        (SW2, SW2),
        (J.move_reply(2, F.T_FOE_A), SW2),
        (J.NO_ACTION, SW2),
        (J.UNKNOWN, SW2),
        (J.reply_size(C), 0),
    ):
        made = J.joint_replies(pred, batch, truth={"reply": np.array([pair])})
        assert int(made.rank[0]) == -1 and made.prob_true[0] == 0.0, pair
    # without a truth nothing is ranked
    assert int(J.joint_replies(pred, batch).rank[0]) == -1


def test_one_slot_on_the_field_gives_its_own_distribution():
    batch, pred = two_slots()
    batch["act_mon"][0, 1] = -1
    batch["action_mask"][0, 1] = 0
    batch["cand_tmask"][0, 1] = 0
    truth = {"reply": np.array([[J.move_reply(0, F.T_FOE_B), J.NO_ACTION]])}
    made = J.joint_replies(pred, batch, k=8, truth=truth)
    found = as_dict(made)
    assert int(made.size[0]) == len(found) == 6 and made.kept[0] == pytest.approx(1.0)
    assert all(b == J.NO_ACTION for _, _, b in found)
    assert found[(J.MEGA_NONE, J.move_reply(0, F.T_FOE_A), J.NO_ACTION)] == (
        pytest.approx(0.375)
    )
    assert found[(J.MEGA_NONE, SW2, J.NO_ACTION)] == pytest.approx(0.15)
    # 0.375, 0.25, 0.15, then 0.125: the truth is fourth
    assert int(made.rank[0]) == 3 and made.prob_true[0] == pytest.approx(0.125)
    listing = made.listing(0)
    assert listing[0]["slots"][1] is None and listing[0]["mega"] is None
    assert listing[0]["slots"][0] == J.describe_reply(J.move_reply(0, F.T_FOE_A), C)
    # the other way round: slot a is the empty one
    batch, pred = two_slots()
    batch["act_mon"][0, 0] = -1
    batch["action_mask"][0, 0] = 0
    made = J.joint_replies(pred, batch, k=8)
    assert int(made.size[0]) == 5
    assert (made.reply[0, :5, 0] == J.NO_ACTION).all()
    assert tuple(made.reply[0, 0]) == (J.NO_ACTION, J.move_reply(0, F.T_FOE_A))


def mega_case() -> tuple[F.Batch, dict[str, np.ndarray]]:
    """Slot a: a move 0.7, OTHER 0.1, a switch 0.2. Slot b: a move 1.0."""
    batch = blank(1)
    put(batch, 0, 0, targets=(AUTO,), bench=(2,), mega=True)
    put(batch, 0, 1, targets=(AUTO,), mega=True)
    pred = empty_prediction(batch)
    pred["action"][0, 0, [0, OTHER, F.switch_index(2, C)]] = (0.7, 0.1, 0.2)
    pred["target"][0, :, 0, F.T_AUTO] = 1.0
    pred["action"][0, 1, 0] = 1.0
    pred["mega"][0] = (0.4, 0.6)
    return batch, pred


def test_mega_bit_two_megas_and_a_switch_with_a_mega_are_excluded():
    batch, pred = mega_case()
    made = J.joint_replies(pred, batch, k=64, mega=True)
    found = as_dict(made)
    move, other = J.move_reply(0, F.T_AUTO), J.other_reply(C)
    none, a, b = 0.6 * 0.4, 0.4 * 0.4, 0.6 * 0.6  # nobody, slot a, slot b
    kept = none + a * 0.8 + b
    assert made.with_mega and made.kept[0] == pytest.approx(kept)
    # removed: both Mega-evolve (0.4 * 0.6) and "a switches and Mega-evolves"
    assert kept == pytest.approx(1 - 0.4 * 0.6 - 0.2 * 0.4 * 0.4)
    assert found[(J.MEGA_NONE, move, move)] == pytest.approx(none * 0.7 / kept)
    assert found[(J.MEGA_A, move, move)] == pytest.approx(a * 0.7 / kept)
    assert found[(J.MEGA_B, move, move)] == pytest.approx(b * 0.7 / kept)
    assert found[(J.MEGA_B, SW2, move)] == pytest.approx(b * 0.2 / kept)
    assert found[(J.MEGA_A, other, move)] == pytest.approx(a * 0.1 / kept)
    assert (J.MEGA_A, SW2, move) not in found  # a switch with a Mega Evolution
    assert sum(found.values()) == pytest.approx(1.0)
    # slot a: 3 replies, slot b: move and OTHER -> 6 pairs; x (nobody, b) and
    # 4 pairs where a moves for "a Mega-evolves"
    assert int(made.size[0]) == len(found) == 6 + 4 + 6
    assert tuple(made.reply[0, 0]) == (move, move) and made.mega[0, 0] == J.MEGA_B
    assert made.listing(0)[0]["mega"] == 1
    # the truth must name the Mega state as well
    truth = {"reply": np.array([[move, move]]), "mega": np.array([J.MEGA_A])}
    ranked = J.joint_replies(pred, batch, k=64, mega=True, truth=truth)
    assert int(ranked.rank[0]) == 2  # after "b" (0.252) and "nobody" (0.168)
    assert ranked.prob_true[0] == pytest.approx(a * 0.7 / kept)
    # without the Mega bit the same call ignores both the Mega head and the state
    plain = J.joint_replies(pred, batch, k=64, truth=truth)
    assert int(plain.size[0]) == 6 and int(plain.rank[0]) == 0
    assert plain.prob_true[0] == pytest.approx(0.7) and (plain.mega == 0).all()
    # a truth without a Mega state has no rank when the bit is on
    no_state = J.joint_replies(pred, batch, mega=True, truth={"reply": truth["reply"]})
    assert int(no_state.rank[0]) == -1


def test_a_mega_the_public_state_rules_out_is_not_a_reply():
    batch, pred = mega_case()
    batch["mon_flag"][0, 1, F.FLAG_MEGA_POSSIBLE] = 0  # slot b cannot
    made = J.joint_replies(pred, batch, k=64, mega=True)
    found = as_dict(made)
    move = J.move_reply(0, F.T_AUTO)
    kept = 0.6 + 0.4 * 0.8  # b's 0.6 is ignored: nobody or a, and a must move
    assert made.kept[0] == pytest.approx(kept)
    assert all(state != J.MEGA_B for state, _, _ in found) and len(found) == 6 + 4
    assert found[(J.MEGA_A, move, move)] == pytest.approx(0.4 * 0.7 / kept)
    truth = {"reply": np.array([[move, move]]), "mega": np.array([J.MEGA_B])}
    assert int(J.joint_replies(pred, batch, mega=True, truth=truth).rank[0]) == -1
    # nobody can: the list is the list without the Mega bit
    batch["mon_flag"][0, 0, F.FLAG_MEGA_POSSIBLE] = 0
    with_bit = J.joint_replies(pred, batch, k=16, mega=True)
    without = J.joint_replies(pred, batch, k=16)
    assert np.array_equal(with_bit.reply, without.reply)
    assert np.allclose(with_bit.prob, without.prob) and (with_bit.mega == 0).all()
    assert int(J.joint_replies(pred, batch, mega=True, truth=truth).rank[0]) == -1
    # "mega_possible" given outright replaces the feature arrays
    small = {
        "action_mask": batch["action_mask"],
        "cand_tmask": batch["cand_tmask"],
        "mega_possible": np.array([[True, False]]),
    }
    again = J.joint_replies(pred, small, k=64, mega=True)
    assert again.kept[0] == pytest.approx(kept) and int(again.size[0]) == 10


def test_top_k_is_monotone_in_k():
    batch, pred, truth = random_case(60, seed=3)
    for with_mega in (False, True):
        lists = {
            k: J.joint_replies(pred, batch, k=k, mega=with_mega, truth=truth)
            for k in (1, 2, 4, 8, 16, 32)
        }
        previous, hits_before = None, None
        for k, made in lists.items():
            assert made.n == 60 and made.k == k
            if previous is not None:
                width = previous.k
                # a shorter list is the beginning of a longer one
                assert np.array_equal(made.reply[:, :width], previous.reply)
                assert np.array_equal(made.mega[:, :width], previous.mega)
                assert np.array_equal(made.prob[:, :width], previous.prob)
                assert (made.mass() >= previous.mass() - 1e-15).all()
                assert np.array_equal(made.rank, previous.rank)
            hits = (made.rank >= 0) & (made.rank < k)
            if hits_before is not None:
                assert (hits | ~hits_before).all()  # covered at k stays covered
            # "among the first k" and "rank below k" are one statement
            for i in range(made.n):
                listed = [
                    (int(made.mega[i, p]), *(int(v) for v in made.reply[i, p]))
                    for p in range(k)
                ]
                true = (
                    int(truth["mega"][i]) if with_mega else 0,
                    *(int(v) for v in truth["reply"][i]),
                )
                assert (true in listed) == bool(hits[i]), (k, i)
            assert (made.mass(1) == made.prob[:, 0]).all()
            assert (made.mass(99) == made.mass()).all()
            previous, hits_before = made, hits


def test_the_vectorised_table_agrees_with_plain_loops():
    batch, pred, truth = random_case(80, seed=11)
    for with_mega in (False, True):
        made = J.joint_replies(pred, batch, k=12, mega=with_mega, truth=truth)
        assert made.n == 80
        ranked = 0
        for i in range(80):
            want, kept = brute_force(pred, batch, i, with_mega)
            assert int(made.size[i]) == len(want), i
            assert made.kept[i] == pytest.approx(kept, abs=1e-12)
            if kept <= 0:
                assert not made.ok[i] and int(made.rank[i]) == -1
                continue
            for place, (weight, state, a, b) in enumerate(want[:12]):
                assert int(made.mega[i, place]) == state, (i, place)
                assert tuple(int(v) for v in made.reply[i, place]) == (a, b)
                assert made.prob[i, place] == pytest.approx(weight / kept, abs=1e-12)
            true = (
                int(truth["mega"][i]) if with_mega else 0,
                int(truth["reply"][i, 0]),
                int(truth["reply"][i, 1]),
            )
            place = next(
                (p for p, x in enumerate(want) if (x[1], x[2], x[3]) == true), -1
            )
            assert int(made.rank[i]) == place, (i, true)
            if place >= 0:
                ranked += 1
                weight = want[place][0]
                assert int(made.above[i]) == sum(x[0] > weight for x in want)
                assert int(made.ties[i]) == sum(x[0] == weight for x in want) - 1
                assert made.prob_true[i] == pytest.approx(weight / kept, abs=1e-12)
        assert ranked >= 20  # the random truths are possible often enough


def test_equal_probabilities_keep_the_order_of_the_index():
    batch = blank(1)
    put(batch, 0, 0, targets=(FOES, AUTO), bench=(2,), mega=True)
    put(batch, 0, 1, targets=(FOES,), bench=(2,))
    pred = F.uniform_prediction(batch)
    for with_mega in (False, True):
        made = J.joint_replies(pred, batch, k=64, mega=with_mega)
        size = int(made.size[0])
        keys = [
            (int(made.mega[0, p]), int(made.reply[0, p, 0]), int(made.reply[0, p, 1]))
            for p in range(size)
        ]
        # within one probability the keys rise: Mega state, slot a, slot b
        for p in range(size - 1):
            if made.prob[0, p] == made.prob[0, p + 1]:
                assert keys[p] < keys[p + 1], (with_mega, p)
        assert (np.diff(made.prob[0, :size]) <= 0).all()
        for place in (0, 3, size - 1):
            state, a, b = keys[place]
            truth = {"reply": np.array([[a, b]]), "mega": np.array([state])}
            ranked = J.joint_replies(pred, batch, mega=with_mega, truth=truth)
            assert int(ranked.rank[0]) == place
            assert int(ranked.above[0]) <= place
            assert place <= int(ranked.above[0] + ranked.ties[0])
    # a flat product: each slot has four replies at 0.25, so every one of the
    # 4 x 4 - 1 joint replies ties with the others
    batch = blank(1)
    for slot in range(2):
        put(batch, 0, slot, targets=(AUTO, AUTO), bench=(2,))
    truth = {"reply": np.array([[SW2, J.other_reply(C)]])}
    flat_pred = F.uniform_prediction(batch)
    flat = J.joint_replies(flat_pred, batch, k=64, truth=truth)
    assert int(flat.size[0]) == 15 and np.allclose(flat.prob[0, :15], 1 / 15)
    assert int(flat.above[0]) == 0 and int(flat.ties[0]) == 14
    # (switch, OTHER): after the 3 x 4 pairs of slot a's two moves and OTHER,
    # and after (switch, move) twice; (switch, switch) is not a reply
    assert int(flat.rank[0]) == 14
    first = {"reply": np.array([[J.move_reply(0, F.T_AUTO)] * 2])}
    assert int(J.joint_replies(flat_pred, batch, truth=first).rank[0]) == 0


def test_an_oracle_is_always_first_and_other_is_only_a_bucket():
    batch, _, _ = random_case(40, seed=5)
    rng = np.random.default_rng(0)
    pred = empty_prediction(batch)
    legal = F.expand_target_mask(batch["cand_tmask"])
    reply = np.full((40, 2), J.NO_ACTION, dtype=np.int64)
    for i in range(40):
        for s in range(2):
            if batch["act_mon"][i, s] < 0:
                continue
            options = np.nonzero(batch["action_mask"][i, s])[0]
            if s == 1 and reply[i, 0] > J.other_reply(C):  # not the same bench
                taken = reply[i, 0] - J.other_reply(C) - 1 + C + 1
                options = options[options != taken]
            action = int(rng.choice(options))
            pred["action"][i, s, action] = 1.0
            if action < C:
                target = int(rng.choice(np.nonzero(legal[i, s, action])[0]))
                pred["target"][i, s, action, target] = 1.0
                reply[i, s] = J.move_reply(action, target)
            elif action == OTHER:
                reply[i, s] = J.other_reply(C)
            else:
                reply[i, s] = J.switch_reply(action - C - 1, C)
    made = J.joint_replies(pred, batch, k=1, truth={"reply": reply})
    assert (made.rank == 0).all() and np.allclose(made.prob_true, 1.0)
    assert np.array_equal(made.reply[:, 0], reply) and np.allclose(made.mass(), 1.0)
    # OTHER is a place in the list like any other, and described as a bucket
    assert (reply == J.other_reply(C)).any()
    assert J.describe_reply(J.other_reply(C), C) == {"kind": "other", "action": OTHER}


# --- inputs: one example, chunks, failures --------------------------------------


def test_one_example_without_the_leading_axis_equals_its_batch_row():
    batch, pred, truth = random_case(6, seed=2)
    whole = J.joint_replies(pred, batch, k=8, mega=True, truth=truth)
    for i in (0, 5):
        one = J.joint_replies(
            {name: value[i] for name, value in pred.items()},
            {name: value[i] for name, value in batch.items()},
            k=8,
            mega=True,
            truth={name: value[i] for name, value in truth.items()},
        )
        assert one.n == 1
        assert np.array_equal(one.reply[0], whole.reply[i])
        assert np.array_equal(one.prob[0], whole.prob[i])
        assert int(one.rank[0]) == int(whole.rank[i])
        assert one.listing() == whole.listing(i)
        # one example's prediction with a batch of one (the runtime's shapes)
        mixed = J.joint_replies(
            {name: value[i] for name, value in pred.items()},
            {name: value[i : i + 1] for name, value in batch.items()},
            k=8,
            mega=True,
        )
        assert mixed.n == 1 and mixed.listing() == whole.listing(i)
    prob, legal = J.slot_replies(
        {name: value[0] for name, value in pred.items()},
        {name: value[0] for name, value in batch.items()},
    )
    assert prob.shape == (1, 2, J.reply_size(C)) and legal.shape == prob.shape


def test_chunking_does_not_change_anything():
    batch, pred, truth = random_case(50, seed=8)
    for with_mega in (False, True):
        whole = J.joint_replies(pred, batch, k=8, mega=with_mega, truth=truth)
        pieces = J.joint_replies(
            pred, batch, k=8, mega=with_mega, truth=truth, chunk_bytes=1
        )
        for name in ("reply", "mega", "prob", "size", "kept", "ok", "rank", "ties"):
            assert np.array_equal(getattr(whole, name), getattr(pieces, name)), name
        assert np.array_equal(whole.prob_true, pieces.prob_true)
        assert np.array_equal(whole.above, pieces.above)


def test_k_larger_than_the_table_is_padded():
    batch = blank(2)
    put(batch, 0, 0, targets=(AUTO,))
    put(batch, 1, 0, targets=(AUTO,))
    pred = F.uniform_prediction(batch)
    # The joint table of three candidates: (3 x 5 + OTHER + 6 switches + "no
    # action") squared. A list is padded up to k, and never beyond the table.
    table = (J.reply_size(3) + 1) ** 2
    assert table == 529
    made = J.joint_replies(pred, batch, k=400)
    # one slot: a move and OTHER
    assert made.k == 400 and (made.size == 2).all()
    assert made.reply.shape == (2, 400, 2) and made.prob.shape == (2, 400)
    assert np.allclose(made.prob[:, :2], 0.5) and (made.prob[:, 2:] == 0).all()
    assert (made.reply[:, 2:] == J.UNKNOWN).all() and len(made.listing(1)) == 2
    wide = J.joint_replies(pred, batch, k=1000)
    assert wide.k == table and wide.reply.shape == (2, table, 2)
    assert np.array_equal(wide.prob[:, :400], made.prob)
    assert np.array_equal(wide.reply[:, :400], made.reply)
    assert (wide.prob[:, 2:] == 0).all() and len(wide.listing(1)) == 2


def test_a_huge_k_asks_for_no_more_than_the_table():
    """A list length is a caller's number: one that is far too large must not
    make the result larger than the joint table it is cut from (k = 10**9 on a
    few examples used to ask for gigabytes)."""
    batch, pred, truth = random_case(6, seed=3)
    for with_mega in (False, True):
        table = (J.reply_size(C) + 1) ** 2 * (J.N_MEGA_STATE if with_mega else 1)
        want = J.joint_replies(pred, batch, k=table, mega=with_mega, truth=truth)
        for huge in (table + 1, 10**9, 10**30, np.int64(2**40)):
            made = J.joint_replies(
                pred,
                batch,
                k=huge,  # type: ignore[arg-type]
                mega=with_mega,
                truth=truth,
            )
            assert made.n == 6 and made.k == table, (with_mega, huge)
            assert made.prob.shape == (6, table) and made.mega.shape == (6, table)
            assert made.reply.shape == (6, table, 2)
            assert made.reply.nbytes + made.prob.nbytes + made.mega.nbytes < 1 << 20
            for name in ("reply", "mega", "prob", "size", "rank", "above", "ties"):
                assert np.array_equal(getattr(made, name), getattr(want, name)), name
        # the first eight of the whole table are the list of eight
        short = J.joint_replies(pred, batch, k=8, mega=with_mega, truth=truth)
        assert np.array_equal(want.reply[:, :8], short.reply)
        assert np.array_equal(want.rank, short.rank)
    # ... and a failure with a huge k is small as well
    J.COUNTERS.clear()
    failed = J.joint_replies({}, batch, k=10**12)
    assert failed.n == 0 and failed.k <= J.MAX_K and failed.prob.nbytes == 0
    J.COUNTERS.clear()


def test_a_mega_argument_that_is_not_a_flag_does_not_raise():
    """A caller's slip: the Mega ARRAY handed over where the flag goes. An
    array has no truth value, and the failure branch itself used to ask for
    one, so the function that never raises raised."""
    batch, pred = two_slots()
    J.COUNTERS.clear()
    slips: list[Any] = [pred["mega"], np.zeros((1, 2)), np.array([True, False])]
    slips += ["yes", 1, 0, None, [True]]
    for slip in slips:
        for k in (8, None, "many"):
            made = J.joint_replies(pred, batch, k=k, mega=slip)  # type: ignore[arg-type]
            assert made.n == 0 and made.listing(0) == [], (slip, k)
            assert made.with_mega is False and made.k == 8
    assert J.COUNTERS["joint_replies:TypeError"] == 3 * len(slips)
    # numpy's own flag is a flag
    flagged = J.joint_replies(pred, batch, mega=np.bool_(True))  # type: ignore[arg-type]
    assert flagged.with_mega is True and flagged.n == 1
    assert J.joint_replies(pred, batch, mega=np.bool_(False)).n == 1  # type: ignore[arg-type]
    # mass() answers for any k as well
    made = J.joint_replies(pred, batch, k=4)
    before = sum(J.COUNTERS.values())
    assert made.mass("x").tolist() == [0.0]  # type: ignore[arg-type]
    assert made.mass(np.zeros((2, 2))).tolist() == [0.0]  # type: ignore[arg-type]
    assert sum(J.COUNTERS.values()) == before + 2
    assert made.mass(2)[0] == pytest.approx(made.prob[0, :2].sum())
    assert made.mass(99)[0] == pytest.approx(made.prob[0].sum())
    J.COUNTERS.clear()


def test_an_example_without_any_probability_is_not_ok():
    batch, pred = two_slots()
    pred["action"][0, 1] = 0.0  # slot b: no mass on anything legal
    truth = {"reply": np.array([[J.move_reply(0, F.T_FOE_A), SW2]])}
    made = J.joint_replies(pred, batch, k=4, truth=truth)
    assert made.n == 1 and not made.ok[0] and made.kept[0] == 0.0
    assert int(made.size[0]) == 28  # the replies are still possible
    assert (made.prob == 0).all() and (made.reply == J.UNKNOWN).all()
    assert int(made.rank[0]) == -1 and made.listing(0) == []
    # zero probability on a possible truth is still a rank, after the others
    batch, pred = two_slots()
    truth = {"reply": np.array([[SW3, J.other_reply(C)]])}  # slot a: 0 on roster 3
    made = J.joint_replies(pred, batch, k=4, truth=truth)
    assert made.ok[0] and made.prob_true[0] == 0.0
    assert int(made.above[0]) == 28 - 4 and int(made.ties[0]) == 3
    assert int(made.above[0]) <= int(made.rank[0]) <= 27


def test_nothing_raises_and_failures_are_counted():
    batch, pred = two_slots()
    J.COUNTERS.clear()
    bad: list[tuple[Any, Any]] = [
        ({}, batch),
        (pred, {}),
        ({**pred, "action": pred["action"][:, :, :5]}, batch),
        ({**pred, "target": pred["target"][..., :3]}, batch),
        ({**pred, "action": np.full_like(pred["action"], np.nan)}, batch),
        (pred, {**batch, "action_mask": batch["action_mask"][:, :1]}),
        (None, None),
    ]
    for made_pred, made_batch in bad:
        made = J.joint_replies(made_pred, made_batch, k=8)
        assert made.n == 0 and made.k == 8 and made.listing(0) == []
        prob, legal = J.slot_replies(made_pred, made_batch)
        assert prob.shape[0] == 0 and legal.shape[0] == 0
    assert sum(J.COUNTERS.values()) == 2 * len(bad)
    assert all(":" in name for name in J.COUNTERS)
    # the Mega bit needs a finite Mega head and a way to tell who can
    assert J.joint_replies({**pred, "mega": np.array([[np.inf, 0]])}, batch).n == 1
    for broken in (np.array([[np.inf, 0]]), np.zeros((1, 3))):
        assert J.joint_replies({**pred, "mega": broken}, batch, mega=True).n == 0
    no_flags = {key: batch[key] for key in ("action_mask", "cand_tmask")}
    assert J.joint_replies(pred, no_flags).n == 1
    assert J.joint_replies(pred, no_flags, mega=True).n == 0
    # a truth of the wrong shape, an odd k
    assert J.joint_replies(pred, batch, truth={"reply": np.zeros(3)}).n == 0
    assert J.joint_replies(pred, batch, truth={}).n == 0
    assert J.joint_replies(pred, batch, k=0).k == 1
    assert J.joint_replies(pred, batch, k="many").n == 0  # type: ignore[arg-type]
    assert J.true_replies({}) == {}
    assert J.true_replies({"act_mon": np.zeros((1, 2))}) == {}
    assert J.joint_replies(pred, batch).listing(7) == []
    # an empty batch is not a failure
    before = sum(J.COUNTERS.values())
    none = J.joint_replies(
        {name: value[:0] for name, value in pred.items()},
        {name: value[:0] for name, value in batch.items()},
        mega=True,
        truth=J.true_replies({name: value[:0] for name, value in batch.items()}),
    )
    assert none.n == 0 and sum(J.COUNTERS.values()) == before
    J.COUNTERS.clear()


def test_the_inputs_are_not_changed():
    batch, pred, truth = random_case(10, seed=4)
    kept = {name: value.copy() for name, value in {**batch, **pred, **truth}.items()}
    J.joint_replies(pred, batch, k=8, mega=True, truth=truth)
    J.slot_replies(pred, batch)
    for name, value in {**batch, **pred, **truth}.items():
        assert np.array_equal(value, kept[name]), name
    # and the list reads four feature arrays, nothing else of the batch
    whole = J.joint_replies(pred, batch, k=8, mega=True)
    needed = ("action_mask", "cand_tmask", "act_mon", "mon_flag")
    bare = J.joint_replies(pred, {name: batch[name] for name in needed}, k=8, mega=True)
    assert bare.n == 10 and np.array_equal(bare.reply, whole.reply)
    assert np.array_equal(bare.prob, whole.prob)
    scrambled = dict(batch)
    for name in batch:
        if name.startswith("y_"):
            scrambled[name] = np.roll(batch[name], 3, axis=0) + 1
    again = J.joint_replies(pred, scrambled, k=8, mega=True)
    assert np.array_equal(again.reply, whole.reply)
    assert np.array_equal(again.prob, whole.prob)


# --- the labels -----------------------------------------------------------------


def labelled() -> F.Batch:
    """Seven examples, one label situation each (see the test below)."""
    batch = blank(7)
    shown = (F.CAND_REVEALED, F.CAND_SHEET, 0)
    for i in range(7):
        for slot in range(2):
            put(
                batch,
                i,
                slot,
                targets=(FOES, AUTO, FOES),
                bench=(2, 3),
                flags=shown,
                mega=(slot == 0),
            )

    def move(i: int, slot: int, action: int, target: int = -1, mega: int = 0) -> None:
        batch["y_kind"][i, slot] = F.Y_KIND_MOVE
        batch["y_action"][i, slot] = action
        batch["y_target"][i, slot] = target
        batch["y_mega"][i, slot] = mega

    def switch(i: int, slot: int, roster: int) -> None:
        batch["y_kind"][i, slot] = F.Y_KIND_SWITCH
        batch["y_action"][i, slot] = F.switch_index(roster, C)
        batch["y_mega"][i, slot] = 0

    move(0, 0, 0, F.T_FOE_B, mega=1)  # a shown move, with a Mega Evolution
    move(0, 1, 1, F.T_AUTO)  # a move from the sheet
    move(1, 0, 2, F.T_FOE_A)  # a guessed candidate
    switch(1, 1, 3)
    move(2, 0, OTHER, F.T_FOE_A)  # outside the candidates
    move(2, 1, 0, F.T_FOE_A)
    move(3, 0, 0, -1)  # an aimed move whose target is not certain
    move(3, 1, 1, F.T_AUTO)
    batch["y_kind"][4, 0] = F.Y_KIND_NONE  # hidden: fainted before it moved
    move(4, 1, 0, F.T_FOE_A)
    batch["act_mon"][5, 1] = -1  # one slot on the field
    batch["action_mask"][5, 1] = 0
    switch(5, 0, 2)
    move(6, 0, 0, F.T_FOE_A, mega=1)  # both slots named: not a possible state
    move(6, 1, 0, F.T_FOE_A, mega=1)
    return batch


def test_true_replies_read_the_labels():
    batch = labelled()
    truth = J.true_replies(batch)
    assert truth["reply"].tolist() == [
        [J.move_reply(0, F.T_FOE_B), J.move_reply(1, F.T_AUTO)],
        [J.move_reply(2, F.T_FOE_A), SW3],
        [J.other_reply(C), J.move_reply(0, F.T_FOE_A)],
        [J.UNKNOWN, J.move_reply(1, F.T_AUTO)],
        [J.UNKNOWN, J.move_reply(0, F.T_FOE_A)],
        [SW2, J.NO_ACTION],
        [J.move_reply(0, F.T_FOE_A), J.move_reply(0, F.T_FOE_A)],
    ]
    assert truth["visible"].tolist() == [True, True, True, False, False, True, True]
    # exactly the examples where every slot on the field has a fine label
    fine = F.fine_label(batch)
    active = batch["act_mon"] >= 0
    assert np.array_equal(truth["visible"], ((fine >= 0) | ~active).all(-1))
    assert np.array_equal(truth["active"], active)
    assert truth["mega"].tolist() == [
        J.MEGA_A,
        J.MEGA_NONE,
        J.MEGA_NONE,
        J.MEGA_NONE,
        J.UNKNOWN,  # the hidden slot could have Mega-evolved: the log cannot tell
        J.MEGA_NONE,
        J.UNKNOWN,  # both slots named
    ]
    assert truth["other"].tolist()[2] == [True, False] and truth["other"].sum() == 1
    assert (
        truth["switch"].sum() == 2 and truth["switch"][1, 1] and truth["switch"][5, 0]
    )
    # shown in battle, or on the sheet; a guessed candidate and OTHER are not
    assert truth["shown"].tolist()[:3] == [[True, True], [False, False], [False, True]]
    assert truth["unshown"].tolist()[:3] == [
        [False, False],
        [True, False],
        [True, False],
    ]
    assert not truth["unshown"][5].any() and not truth["shown"][5].any()
    # a hidden slot that cannot Mega-evolve leaves the state known
    batch["mon_flag"][4, 0, F.FLAG_MEGA_POSSIBLE] = 0
    assert int(J.true_replies(batch)["mega"][4]) == J.MEGA_NONE
    # the truth of a visible example is a possible joint reply
    made = J.joint_replies(
        F.uniform_prediction(batch), batch, k=4, mega=True, truth=truth
    )
    assert (made.rank[[0, 1, 2, 5]] >= 0).all()
    assert (made.rank[[3, 4, 6]] == -1).all()


# --- house rules ----------------------------------------------------------------


def test_library_names_nothing_from_the_dex_and_has_no_bare_assert():
    vocab = F.Vocab.build()
    known = set(E.moves_dex()) | set(E.pokedex())
    known |= set(vocab.items[3:]) | set(vocab.abilities[3:])
    path = ROOT / "vgc_bench/src/oppmodel/joint.py"
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
    assert not found, sorted(found)
    assert not [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assert)]
    # every raise sits in a private helper that a public function catches
    public = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        or (isinstance(node, ast.FunctionDef) and not node.name.startswith("_"))
    ]
    assert len(public) >= 9
    for node in public:
        assert not [n for n in ast.walk(node) if isinstance(n, ast.Raise)], node.name


# --- the real dataset and a fitted table (skipped when absent) ------------------


@pytest.mark.skipif(
    not (DATASET / "manifest.json").is_file()
    or not (FITTED / "flags_table.pt").is_file(),
    reason="results_oppmodel/v1_ondisk or results_oppmodel/tables_v1 is absent",
)
def test_real_labels_are_possible_joint_replies():
    from vgc_bench.src.oppmodel.artifact import load_predictor

    J.COUNTERS.clear()
    batch, _ = F.load_dataset(DATASET, splits=["ladder_holdout"])
    truth = J.true_replies(batch)
    active = np.asarray(batch["act_mon"]) >= 0
    fine = F.fine_label(batch)
    assert np.array_equal(truth["visible"], ((fine >= 0) | ~active).all(-1))
    visible = truth["visible"]
    assert 0.5 < visible.mean() < 0.9
    features = F.sheet_unknown_as_closed(
        {k: v for k, v in batch.items() if not k.startswith(("y_", "m_"))}
    )
    pred = load_predictor(FITTED / "flags_table.pt").predictor.predict(features)
    part = F.take(batch, visible)
    own = {name: np.asarray(value)[visible] for name, value in pred.items()}
    told = {name: value[visible] for name, value in truth.items()}
    assert (told["mega"] != J.UNKNOWN).all()
    n_cand = np.asarray(batch["cand_flag"]).shape[-1]
    for with_mega in (False, True):
        made = J.joint_replies(own, part, k=8, mega=with_mega, truth=told)
        assert made.n == int(visible.sum()) and made.ok.all()
        # every reply a human made is one the table calls possible
        assert (made.rank >= 0).all()
        assert (made.prob_true > 0).all() and (made.mass() <= 1 + 1e-9).all()
        assert (made.kept <= 1 + 1e-9).all() and (made.kept > 0).all()
        assert (made.size >= 2).all()
        if not with_mega:
            assert (made.mega == J.MEGA_NONE).all()
            assert int(made.size.max()) <= (J.reply_size(n_cand)) ** 2
    assert not J.COUNTERS
