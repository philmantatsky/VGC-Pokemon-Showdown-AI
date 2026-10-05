"""Opponent predictor scorecard: bootstrap, label hygiene, events, verdicts.

Inputs are synthetic batches written inline. Two tests at the end read the
built dataset and the fitted tables and skip when those are absent (they are
git-ignored).
"""

from __future__ import annotations

import ast
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evaluation import oppmodel_scorecard as SC
from vgc_bench.src.oppmodel import events as E
from vgc_bench.src.oppmodel import features as F

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "results_oppmodel" / "v1_ondisk"
FITTED = ROOT / "results_oppmodel" / "tables_v1"

C = 4  # candidates per slot in the synthetic batches
OTHER = F.other_index(C)
ATTACK = F.CAND_DAMAGING | F.CAND_AIMED
SPREAD = F.CAND_DAMAGING | F.CAND_SPREAD
GUARD = F.CAND_PROTECT
FOES = (1 << F.T_FOE_A) | (1 << F.T_FOE_B)
AUTO = 1 << F.T_AUTO
LN = math.log


# --- synthetic batches ----------------------------------------------------------


def blank(n: int, games: Any = None) -> F.Batch:
    """``n`` examples with nothing on the field; labels and meta arrays included."""
    batch = {
        name: np.zeros((n, *spec.shape), dtype=spec.dtype)
        for name, spec in F.layout(C).items()
    }
    for name in ("act_mon", "foe_mon", "y_kind", "y_action", "y_target", "y_mega"):
        batch[name][:] = -1
    batch["y_intent"][:] = -1
    batch["y_attack"][:] = -1
    batch["turn"][:] = 2
    batch["m_battle"] = (
        np.arange(n, dtype=np.int32) if games is None else np.asarray(games, np.int32)
    )
    batch["m_sheet"] = np.zeros(n, dtype=np.uint8)
    batch["m_flag"] = np.zeros(n, dtype=np.uint8)
    batch["m_weight"] = np.ones(n, dtype=np.float32)
    return batch


def put(
    batch: F.Batch,
    i: int,
    slot: int = 0,
    *,
    cands: tuple[tuple[int, int], ...] = ((ATTACK, FOES), (GUARD, AUTO)),
    bench: tuple[int, ...] = (),
    foes: tuple[bool, bool] = (True, True),
) -> None:
    """Slot ``slot`` of example ``i``: candidates (class bits, target bits), bench."""
    batch["act_mon"][i, slot] = slot
    batch["mon_flag"][i, slot, F.FLAG_PRESENT] = 1
    for column in (F.ID_SPECIES, F.ID_FORME, F.ID_KEY):
        batch["mon_id"][i, slot, column] = 7
    for position, up in enumerate(foes):
        if up:
            batch["foe_mon"][i, position] = F.N_ROSTER + position
    for column, (bits, targets) in enumerate(cands):
        batch["cand_move"][i, slot, column] = 11 + column
        batch["cand_flag"][i, slot, column] = F.CAND_VALID | bits
        batch["cand_tmask"][i, slot, column] = targets
        batch["action_mask"][i, slot, column] = 1
    batch["action_mask"][i, slot, OTHER] = 1
    batch["cand_tmask"][i, slot, OTHER] = (1 << F.N_TARGET) - 1
    for roster in bench:
        batch["switch_mask"][i, slot, roster] = 1
        batch["action_mask"][i, slot, F.switch_index(roster, C)] = 1


def label_move(
    batch: F.Batch,
    i: int,
    slot: int,
    action: int,
    *,
    target: int = -1,
    attack: tuple[int, int] = (0, 0),
    protect: bool = False,
) -> None:
    batch["y_kind"][i, slot] = F.Y_KIND_MOVE
    batch["y_action"][i, slot] = action
    batch["y_set"][i, slot, action] = 1
    batch["y_target"][i, slot] = target
    batch["y_attack"][i, slot] = attack
    batch["y_flag"][i, slot, [F.Y_SWITCH_KNOWN, F.Y_PROTECT_KNOWN]] = 1
    batch["y_flag"][i, slot, F.Y_PROTECTED] = int(protect)


def label_switch(batch: F.Batch, i: int, slot: int, roster: int) -> None:
    action = F.switch_index(roster, C)
    batch["y_kind"][i, slot] = F.Y_KIND_SWITCH
    batch["y_action"][i, slot] = action
    batch["y_set"][i, slot, action] = 1
    batch["y_attack"][i, slot] = (0, 0)
    batch["y_flag"][i, slot, [F.Y_SWITCHED, F.Y_SWITCH_KNOWN, F.Y_PROTECT_KNOWN]] = 1


def label_stopped(
    batch: F.Batch,
    i: int,
    slot: int,
    allowed: tuple[int, ...],
    *,
    kind: int = F.Y_KIND_NONE,
    reason: str = E.REASON_FAINTED_FIRST,
    protect_known: bool = True,
) -> None:
    """A hidden action: the log allows ``allowed`` and proves it was no switch."""
    batch["y_kind"][i, slot] = kind
    batch["y_set"][i, slot, list(allowed)] = 1
    batch["y_reason"][i, slot] = F.REASON_NAMES.index(reason)
    batch["y_flag"][i, slot, F.Y_SWITCH_KNOWN] = 1
    batch["y_flag"][i, slot, F.Y_PROTECT_KNOWN] = int(protect_known)


def small_batch() -> F.Batch:
    """Four examples in two games; slot a acts, both foes stand.

    Candidates of every slot: an aimed attack (two legal targets), a Protect.
    0: the attack at foe a.  1: Protect.  2: a switch.  3: fainted first.
    """
    batch = blank(4, games=[0, 0, 1, 1])
    for i in range(4):
        put(batch, i, 0, bench=(2, 3))
    label_move(batch, 0, 0, 0, target=F.T_FOE_A, attack=(1, 0))
    label_move(batch, 1, 0, 1, target=F.T_AUTO, protect=True)
    label_switch(batch, 2, 0, 2)
    label_stopped(batch, 3, 0, (0, OTHER))
    return batch


def sharp(batch: F.Batch, action: dict[int, int], strength: float = 0.9) -> Any:
    """The uniform prediction with ``strength`` moved onto one action per example."""
    pred = F.uniform_prediction(batch)
    for i, index in action.items():
        legal = np.asarray(batch["action_mask"][i, 0]) > 0
        row = np.where(legal, (1.0 - strength) / (legal.sum() - 1), 0.0)
        row[index] = strength
        pred["action"][i, 0] = row
    return pred


class Recorder:
    """A predictor that keeps what it was given and answers uniformly."""

    def __init__(self) -> None:
        self.seen: list[F.Batch] = []
        self.counters: Counter[str] = Counter()

    def predict(self, batch: F.Batch) -> dict[str, np.ndarray]:
        self.seen.append(dict(batch))
        return F.uniform_prediction(batch)


# --- the clustered bootstrap ----------------------------------------------------


def test_identical_predictors_give_exactly_zero_and_a_zero_width_interval():
    rng = np.random.default_rng(0)
    nll = rng.random((60, 2))
    scored = rng.random((60, 2)) > 0.2
    games = np.repeat(np.arange(12), 5)
    got = SC.paired_difference(nll, nll.copy(), scored, games, resamples=400, seed=3)
    assert got["diff"] == 0.0 and got["low"] == 0.0 and got["high"] == 0.0
    assert got["slots"] == int(scored.sum()) and got["games"] == 12


def test_a_constant_difference_is_recovered_exactly():
    rng = np.random.default_rng(1)
    second = rng.random((80, 2))
    scored = rng.random((80, 2)) > 0.3
    games = rng.integers(0, 15, size=80)
    got = SC.paired_difference(second + 0.25, second, scored, games, resamples=300)
    assert got["diff"] == pytest.approx(0.25, abs=1e-12)
    assert got["low"] == pytest.approx(0.25, abs=1e-9)
    assert got["high"] == pytest.approx(0.25, abs=1e-9)


def test_a_known_difference_is_recovered_with_a_game_level_interval():
    """Per-game differences with a known mean and spread, equal game sizes."""
    rng = np.random.default_rng(2)
    n_games, per_game = 200, 10
    delta = rng.normal(0.10, 0.05, size=n_games)
    games = np.repeat(np.arange(n_games), per_game)
    second = rng.random((n_games * per_game, 2))
    first = second + delta[games][:, None]
    scored = np.ones_like(first, dtype=bool)
    got = SC.paired_difference(first, second, scored, games, resamples=4000, seed=5)
    assert got["diff"] == pytest.approx(delta.mean(), abs=1e-12)
    assert got["low"] < 0.10 < got["high"]
    expected = 1.96 * delta.std(ddof=1) / math.sqrt(n_games)
    half = (got["high"] - got["low"]) / 2
    assert half == pytest.approx(expected, rel=0.15)
    # resampling slots instead of games would be sqrt(20) times narrower
    assert half > 3 * expected / math.sqrt(2 * per_game)


def test_the_estimator_is_a_ratio_of_sums_not_a_mean_of_game_means():
    games = np.array([0, 0, 0, 0, 1])
    first = np.array([[1.0, 1.0]] * 4 + [[5.0, 0.0]])
    second = np.zeros_like(first)
    scored = np.array([[True, True]] * 4 + [[True, False]])
    got = SC.paired_difference(first, second, scored, games, resamples=50)
    assert got["diff"] == pytest.approx(13.0 / 9.0)  # not (1 + 5) / 2
    assert got["slots"] == 9 and got["games"] == 2


def test_one_set_of_resampled_games_serves_every_column():
    rng = np.random.default_rng(4)
    games = rng.integers(0, 9, size=40)
    values = rng.random((40, 2))
    mask = rng.random((40, 2)) > 0.5
    sums = SC.GameSums(games)
    sums.add("a", values, mask)
    sums.add("b", values, mask)
    sums.add("double", 2 * values, mask)
    sums.count("n", mask)
    draws = sums.draws(200, seed=11)
    assert draws.resamples == 200 and draws.n_games == len(np.unique(games))
    assert np.array_equal(draws.samples[:, 0], draws.samples[:, 1])
    assert np.allclose(draws.samples[:, 2], 2 * draws.samples[:, 0])
    assert draws.total("n") == mask.sum()
    assert draws.total("a") == pytest.approx(values[mask].sum())
    # every resample holds as many games as the data, drawn with replacement
    ones = SC.GameSums(games)
    ones.add("games", np.ones(40) / np.bincount(games)[games])
    assert np.allclose(ones.draws(50, seed=1).samples[:, 0], draws.n_games)
    again = sums.draws(200, seed=11)
    assert np.array_equal(again.samples, draws.samples)
    assert not np.array_equal(sums.draws(200, seed=12).samples, draws.samples)


def test_ratio_drops_resamples_with_an_empty_denominator():
    games = np.arange(6)
    fired = np.array([1.0, 0, 0, 0, 0, 0])
    sums = SC.GameSums(games)
    sums.add("hit", fired)
    sums.add("fired", fired)
    found = sums.draws(500, seed=2).ratio("hit", "fired")
    assert found["value"] == 1.0 and found["low"] == 1.0 and found["high"] == 1.0
    sums = SC.GameSums(games)
    sums.add("hit", np.zeros(6))
    sums.add("fired", np.zeros(6))
    assert sums.draws(100, seed=2).ratio("hit", "fired") == {
        "value": None,
        "low": None,
        "high": None,
    }


def test_no_interval_from_fewer_than_two_games_and_bad_columns_are_refused():
    got = SC.paired_difference(
        np.ones((3, 2)), np.zeros((3, 2)), np.ones((3, 2), bool), np.zeros(3)
    )
    assert got["diff"] == 1.0 and got["low"] is None and got["high"] is None
    empty = SC.paired_difference(
        np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 2), bool), np.zeros(0)
    )
    assert empty["diff"] is None and empty["slots"] == 0 and empty["games"] == 0
    sums = SC.GameSums(np.arange(3))
    sums.add("a", np.ones(3))
    with pytest.raises(ValueError):
        sums.add("a", np.ones(3))
    with pytest.raises(ValueError):
        sums.add("b", np.ones(4))


# --- label stripping and the guard around predict -------------------------------


def test_features_only_drops_every_label_and_meta_array():
    batch = small_batch()
    features = SC.features_only(batch)
    assert features and not [k for k in features if k.startswith(("y_", "m_"))]
    assert set(batch) - set(features) == {
        k for k in batch if k.startswith(("y_", "m_"))
    }
    assert all(features[k] is batch[k] for k in features)


def test_a_predictor_never_sees_labels_and_unknown_sheets_read_closed():
    class Strict(Recorder):
        def predict(self, batch: F.Batch) -> dict[str, np.ndarray]:
            leaked = [k for k in batch if k.startswith(("y_", "m_"))]
            if leaked:
                raise AssertionError(f"labels reached the predictor: {leaked}")
            return super().predict(batch)

    batch = small_batch()
    batch["game_flag"][:, F.G_ACTOR_SHEET] = F.SHEET_UNKNOWN
    batch["game_flag"][1, F.G_OTHER_SHEET] = F.SHEET_OPEN
    strict = Strict()
    pred, seconds = SC.guarded_predict(strict, batch, name="strict")
    assert seconds >= 0 and set(pred) == {"action", "target", "mega"}
    seen = strict.seen[0]["game_flag"]
    assert (seen[:, F.G_ACTOR_SHEET] == F.SHEET_CLOSED).all()
    assert seen[1, F.G_OTHER_SHEET] == F.SHEET_OPEN
    assert (batch["game_flag"][:, F.G_ACTOR_SHEET] == F.SHEET_UNKNOWN).all()


def test_a_predictor_that_peeks_at_labels_is_caught():
    """Both ways a peek can end: an exception, or a counted fallback to uniform."""

    def peek(batch: F.Batch) -> dict[str, np.ndarray]:
        return {"action": batch["y_set"], "target": batch["y_target"]}

    class Raises:
        def predict(self, batch: F.Batch) -> dict[str, np.ndarray]:
            return peek(batch)

    class FallsBack(Recorder):
        def predict(self, batch: F.Batch) -> dict[str, np.ndarray]:
            try:
                return peek(batch)
            except Exception as exc:
                self.counters[f"predict_error:{type(exc).__name__}"] += 1
                return F.uniform_prediction(batch)

    batch = small_batch()
    with pytest.raises(SC.ScorecardError, match="predict raised"):
        SC.guarded_predict(Raises(), batch, name="peek")
    with pytest.raises(SC.ScorecardError, match="fell back"):
        SC.guarded_predict(FallsBack(), batch, name="peek")
    assert SC.predict_errors(Recorder()) == 0
    assert SC.predict_errors(object()) == 0


def test_a_prediction_that_cannot_be_scored_is_refused():
    batch = small_batch()

    class Fixed:
        def __init__(self, change: Any) -> None:
            self.change = change

        def predict(self, features: F.Batch) -> Any:
            pred = F.uniform_prediction(features)
            return self.change(pred)

    def with_nan(pred: Any) -> Any:
        pred["mega"][0, 0] = np.nan
        return pred

    def wrong_shape(pred: Any) -> Any:
        pred["action"] = pred["action"][:, :, :-1]
        return pred

    def missing(pred: Any) -> Any:
        return {"action": pred["action"]}

    for change, text in (
        (with_nan, "non-finite"),
        (wrong_shape, "cannot be scored"),
        (missing, "did not return"),
    ):
        with pytest.raises(SC.ScorecardError, match=text):
            SC.guarded_predict(Fixed(change), batch)


# --- Elo modes ------------------------------------------------------------------


def test_elo_modes_touch_only_the_elo_arrays():
    batch = blank(6)
    for i in range(6):
        put(batch, i, 0)
    batch["elo"][:] = [[1000 + 10 * i, 1500 + 10 * i] for i in range(6)]
    batch["elo_known"][:] = 1
    batch["elo_known"][5] = (0, 1)
    recorder = Recorder()
    SC.guarded_predict(recorder, batch)
    kept = recorder.seen[0]
    for mode in SC.ELO_MODES:
        recorder.seen.clear()
        rng = np.random.default_rng(9)
        SC.guarded_predict(recorder, batch, elo_mode=mode, rng=rng)
        seen = recorder.seen[0]
        assert set(seen) == set(kept)
        for name in kept:
            if name in ("elo", "elo_known"):
                continue
            assert np.array_equal(seen[name], kept[name]), (mode, name)
        if mode == F.ELO_BLANK:
            assert not seen["elo"].any() and not seen["elo_known"].any()
        elif mode == F.ELO_SWAP:
            assert np.array_equal(seen["elo"], kept["elo"][:, ::-1])
            assert np.array_equal(seen["elo_known"], kept["elo_known"][:, ::-1])
        else:  # shuffle: whole rows move together, nothing is invented
            order = np.random.default_rng(9).permutation(6)
            assert np.array_equal(seen["elo"], kept["elo"][order])
            assert np.array_equal(seen["elo_known"], kept["elo_known"][order])
            assert not np.array_equal(seen["elo"], kept["elo"])
    assert np.array_equal(batch["elo"][:, 0], 1000 + 10 * np.arange(6))


# --- events and their denominators ----------------------------------------------


def events_batch() -> F.Batch:
    """Eight examples, both actor slots filled, both foes up unless said.

    0: a attacks foe a (visible), b uses Protect.
    1: a switches, b's turn never resolved (nothing known).
    2: a fainted first (stopped), b attacks foe b.
    3: a flinched and Protect is not ruled out, b's click was overridden.
    4: a attacks the only foe (a: the target is forced), slot b is empty.
    5: a's move has an unknown target, b attacks foe a.
    6: a fainted first, slot b is empty, foe b is absent.
    7: a uses Protect, slot b is empty.
    8: a's aimed attack LANDED on foe a but its target is not certain (a
       redirector stood, or it was retargeted); b does not attack.
    9: a's click is hidden (fainted first), b visibly does not attack.
    """
    batch = blank(10)
    for i in range(10):
        put(batch, i, 0)
        if i not in (4, 6, 7):
            put(batch, i, 1)
    batch["foe_mon"][[4, 6], 1] = -1
    label_move(batch, 0, 0, 0, target=F.T_FOE_A, attack=(1, 0))
    label_move(batch, 0, 1, 1, protect=True)
    label_switch(batch, 1, 0, 2)
    batch["y_kind"][1, 1] = F.Y_KIND_NONE
    batch["y_reason"][1, 1] = F.REASON_NAMES.index(E.REASON_UNRESOLVED)
    label_stopped(batch, 2, 0, (0, OTHER))
    label_move(batch, 2, 1, 0, target=F.T_FOE_B, attack=(0, 1))
    label_stopped(
        batch,
        3,
        0,
        (0, 1, OTHER),
        kind=F.Y_KIND_HIDDEN,
        reason=E.REASON_FLINCH,
        protect_known=False,
    )
    label_stopped(
        batch, 3, 1, (0, 1, OTHER), kind=F.Y_KIND_HIDDEN, reason=E.REASON_OVERRIDDEN
    )
    label_move(batch, 4, 0, 0, target=F.T_FOE_A, attack=(1, -1))
    label_move(batch, 5, 0, 0, attack=(-1, -1))
    label_move(batch, 5, 1, 0, target=F.T_FOE_A, attack=(1, 0))
    label_stopped(batch, 6, 0, (0, OTHER))
    label_move(batch, 7, 0, 1, protect=True)
    label_move(batch, 8, 0, 0, attack=(1, 0))  # y_target stays -1
    label_move(batch, 8, 1, 1, protect=True)
    label_stopped(batch, 9, 0, (0, OTHER))
    label_move(batch, 9, 1, 1, protect=True)
    return batch


def test_event_denominators():
    labels = SC.event_labels(events_batch())
    known, happened = labels[SC.EVENT_SWITCH]
    # every slot whose switch state the log tells: all but the unresolved one
    assert known.tolist() == [
        [True, True],
        [True, False],
        [True, True],
        [True, True],
        [True, False],
        [True, True],
        [True, False],
        [True, False],
        [True, True],
        [True, True],
    ]
    assert (known & happened).sum() == 1 and happened[1, 0]
    known, happened = labels[SC.EVENT_PROTECT]
    assert known.sum() == 15 and not known[3, 0] and not known[1, 1]
    assert (known & happened).sum() == 4 and happened[0, 1] and happened[7, 0]

    click, hit = labels[SC.EVENT_ATTACK]
    assert click.shape == (10, 2, 2)
    # visible clicks only: 0a, 0b, 1a, 2b, 4a (foe a only), 5b, 7a, 8b, 9b
    assert click.sum() == 2 + 2 + 2 + 2 + 1 + 2 + 2 + 2 + 2
    assert not click[2, 0].any() and not click[3].any() and not click[5, 0].any()
    assert hit.sum() == 4 and hit[0, 0, 0] and hit[2, 1, 1] and hit[4, 0, 0]
    # 8a: the label says where the move LANDED, not what was clicked. Neither
    # "attacked foe a" nor "did not attack foe b" is known of that slot.
    assert not click[8, 0].any() and not hit[8].any() and click[8, 1].all()

    arrived, hit_arrived = labels[SC.EVENT_ATTACK_ARRIVED]
    # adds the slots stopped before acting (2a, 6a and 9a fainted first, 3a
    # flinched) as "no attack" on every foe slot that holds a Pokemon, but not
    # the overridden click (3b) and not the unresolved turn (1b)
    assert arrived.sum() == click.sum() + 2 + 2 + 1 + 2
    assert arrived[2, 0].all() and arrived[3, 0].all()
    assert arrived[6, 0].tolist() == [True, False]
    assert not arrived[3, 1].any() and not arrived[1, 1].any()
    assert not arrived[8, 0].any()  # a landed-only target is not "stopped"
    assert np.array_equal(hit_arrived, hit)
    assert not (hit_arrived & ~click).any()


def test_a_landed_target_counts_only_when_it_is_certain():
    """One slot, one aimed attack that hit foe a; only ``y_target`` differs."""
    batch = blank(4)
    for i in range(4):
        put(batch, i, 0)
    label_move(batch, 0, 0, 0, target=F.T_FOE_A, attack=(1, 0))  # trusted
    label_move(batch, 1, 0, 0, attack=(1, 0))  # landed there, not certain
    label_move(batch, 2, 0, OTHER, attack=(1, 0))  # the same, outside the candidates
    label_move(batch, 3, 0, 1, protect=True)  # no target, attacks nobody: known
    click, hit = SC.event_labels(batch)[SC.EVENT_ATTACK]
    assert click[:, 0].tolist() == [[True, True], [False] * 2, [False] * 2, [True] * 2]
    assert hit.sum() == 1 and hit[0, 0, 0]
    known, yes = SC.event_labels(batch)[SC.EVENT_EITHER]
    assert known.tolist() == [[True, True], [False] * 2, [False] * 2, [True, True]]
    assert yes.tolist() == [[True, False], [False] * 2, [False] * 2, [False] * 2]


def test_attacked_by_either_needs_every_attacker_known_whatever_the_answer():
    labels = SC.event_labels(events_batch())
    known, happened = labels[SC.EVENT_EITHER]
    assert known.shape == (10, 2)
    assert known[0].tolist() == [True, True] and happened[0].tolist() == [True, False]
    assert not known[1].any()  # b unknown and a did not attack
    # 2: a's click is hidden and b visibly hits foe b. Keeping that row because
    # the answer is yes would keep hidden-attacker rows only when it is yes.
    assert not known[2].any() and not happened[2].any()
    assert not known[3].any()
    assert known[4].tolist() == [True, False]  # empty slot b settles; foe b absent
    assert happened[4].tolist() == [True, False]
    # 5: a's target is unknown; b's visible hit on foe a does not settle the row
    assert not known[5].any() and not happened[5].any()
    assert not known[6].any()  # the only slot's click is hidden
    assert known[7].tolist() == [True, True] and not happened[7].any()
    assert not known[8].any()  # a's target is where the move landed
    # 9: a hidden attacker and a visible non-attacker: unknown, both foes
    assert not known[9].any()
    assert known.sum() == 2 + 1 + 2 and (known & happened).sum() == 2
    known, happened = labels[SC.EVENT_EITHER_ARRIVED]
    assert known[6].tolist() == [True, False] and not happened[6].any()
    assert known[2].tolist() == [True, True] and happened[2].tolist() == [False, True]
    assert known[9].tolist() == [True, True] and not happened[9].any()
    assert not known[3].any()  # the overridden click stays unknown
    assert not known[5].any() and not known[8].any()
    # The outcome is never True where the row is unknown.
    for name in SC.EVENTS:
        found, did = labels[name]
        assert found.shape == did.shape and not (did & ~found).any(), name


def test_attack_rows_split_by_the_number_of_opposing_pokemon():
    batch = events_batch()
    labels = SC.event_labels(batch)
    two = (batch["foe_mon"] >= 0).all(-1)
    assert two.tolist() == [True] * 4 + [False, True, False] + [True] * 3
    click, hit = labels[SC.EVENT_ATTACK]
    wide, wide_hit = labels[SC.EVENT_ATTACK_TWO]
    lone, lone_hit = labels[SC.EVENT_ATTACK_ONE]
    assert np.array_equal(wide | lone, click) and not (wide & lone).any()
    assert np.array_equal(wide_hit | lone_hit, hit) and not (wide_hit & lone_hit).any()
    assert lone.sum() == 1 and lone[4, 0, 0] and lone_hit.sum() == 1
    # 0a at foe a, 2b at foe b and 5b at foe a; 4a's hit is the one-target row
    assert wide.sum() == click.sum() - 1 and wide_hit.sum() == 3
    either, yes = labels[SC.EVENT_EITHER]
    wide, _ = labels[SC.EVENT_EITHER_TWO]
    lone, _ = labels[SC.EVENT_EITHER_ONE]
    assert np.array_equal(wide | lone, either) and not (wide & lone).any()
    assert lone.tolist() == [[False] * 2] * 4 + [[True, False]] + [[False] * 2] * 5
    # The split events read the same prediction as the undivided one.
    got = SC.event_predictions(F.uniform_prediction(batch), batch)
    assert got[SC.EVENT_ATTACK_TWO] is got[SC.EVENT_ATTACK] is got[SC.EVENT_ATTACK_ONE]
    assert got[SC.EVENT_EITHER_TWO] is got[SC.EVENT_EITHER] is got[SC.EVENT_EITHER_ONE]


def test_an_event_that_cannot_happen_is_told_apart():
    """A slot with no legal switch target, or no Protect candidate, is answered
    0 by every predictor: the mask's doing, not the predictor's."""
    batch = blank(6, games=[0, 0, 1, 1, 2, 2])
    for i in range(6):
        cands = ((ATTACK, FOES), (GUARD, AUTO)) if i < 5 else ((ATTACK, FOES),)
        put(batch, i, 0, bench=(2, 3) if i < 3 else (), cands=cands)
    label_switch(batch, 0, 0, 2)
    for i in range(1, 5):
        label_move(batch, i, 0, 0, target=F.T_FOE_A, attack=(1, 0))
    label_move(batch, 5, 0, 0, target=F.T_FOE_A, attack=(1, 0))
    can = SC.event_possible(batch)
    assert set(can) == set(SC.MASKED_EVENTS)
    assert can[SC.EVENT_SWITCH][:, 0].tolist() == [True] * 3 + [False] * 3
    assert can[SC.EVENT_PROTECT][:, 0].tolist() == [True] * 5 + [False]
    assert not can[SC.EVENT_SWITCH][:, 1].any()  # an empty slot
    pred = F.uniform_prediction(batch)
    found = SC.score_set(batch, {"u": pred}, "u", resamples=50, seed=3)
    switch = found["predictors"]["u"]["events"][SC.EVENT_SWITCH]
    assert switch["slots"] == 6 and switch["impossible"] == 3
    assert switch["slots_where_possible"] == 3
    assert switch["observed"] == pytest.approx(1 / 6)
    assert switch["observed_where_possible"] == pytest.approx(1 / 3)
    assert found["events"][SC.EVENT_SWITCH]["impossible"] == 3
    # uniform over two moves, OTHER and two pointers: P(switch) = 0.4 where a
    # switch is legal and exactly 0 where it is not
    p = np.array([0.4, 0.4, 0.4, 0.0, 0.0, 0.0])
    y = np.array([1.0, 0, 0, 0, 0, 0])
    overall = 1 - ((p - y) ** 2).mean() / ((1 / 6) * (5 / 6))
    inside = 1 - ((p[:3] - y[:3]) ** 2).mean() / ((1 / 3) * (2 / 3))
    assert switch["skill"]["value"] == pytest.approx(overall)
    assert switch["skill_where_possible"]["value"] == pytest.approx(inside)
    assert inside < overall  # part of the reported skill was the mask
    decision = {d["threshold"]: d for d in switch["decisions"]}
    assert decision[0.35]["slots"] == 3 and decision[0.35]["share"] == 0.5
    assert decision[0.35]["share_where_possible"] == 1.0
    assert decision[0.35]["predicted"] == pytest.approx(0.4)
    protect = found["predictors"]["u"]["events"][SC.EVENT_PROTECT]
    assert protect["impossible"] == 1 and protect["slots_where_possible"] == 5
    # An event the mask does not gate: nothing is impossible, one skill.
    attack = found["predictors"]["u"]["events"][SC.EVENT_ATTACK]
    assert attack["impossible"] == 0
    assert attack["skill_where_possible"] == attack["skill"]
    json.dumps(SC._plain(found), allow_nan=False)


def test_differences_are_also_given_with_accounts_as_the_resampling_unit():
    """Eight games of two accounts: resampling accounts gives a wider interval
    than resampling games when the difference depends on the account."""
    n = 16
    batch = blank(n, games=np.repeat(np.arange(8), 2))
    for i in range(n):
        put(batch, i, 0, bench=(2, 3))
        label_move(batch, i, 0, 0, target=F.T_FOE_A, attack=(1, 0))
    batch["m_actor"] = np.repeat(np.array([11, 22], dtype=np.uint32), n // 2)
    uniform = F.uniform_prediction(batch)
    # Sharp on the first account's examples only.
    better = sharp(batch, {i: 0 for i in range(n // 2)})
    found = SC.score_set(batch, {"u": uniform, "s": better}, "u", resamples=400, seed=5)
    assert found["clusters"] == {SC.CLUSTER_GAME: 8, SC.CLUSTER_ACCOUNT: 2}
    row = found["predictors"]["s"]
    games, people = row["difference"]["fine"], row["difference_by_account"]
    assert people["diff"] == pytest.approx(games["diff"]) and games["diff"] < 0
    assert games["high"] < 0  # eight games: clearly below zero
    # Two accounts, one of which carries all of the difference: a resample of
    # the other one alone gives exactly zero.
    assert people["high"] == pytest.approx(0.0, abs=1e-12)
    assert people["high"] - people["low"] > games["high"] - games["low"]
    assert found["predictors"]["u"]["difference_by_account"] is None
    # Without the account column the card says so and nothing else changes.
    bare = {k: v for k, v in batch.items() if k != "m_actor"}
    plain = SC.score_set(bare, {"u": uniform, "s": better}, "u", resamples=400, seed=5)
    assert plain["clusters"] == {SC.CLUSTER_GAME: 8, SC.CLUSTER_ACCOUNT: None}
    assert plain["predictors"]["s"]["difference_by_account"] is None
    assert plain["predictors"]["s"]["difference"] == row["difference"]


def test_event_predictions_follow_event_probs():
    batch = events_batch()
    pred = F.uniform_prediction(batch)
    events = F.event_probs(pred, batch)
    got = SC.event_predictions(pred, batch)
    assert np.array_equal(got[SC.EVENT_SWITCH], events[E.INTENT_SWITCH])
    assert np.array_equal(got[SC.EVENT_PROTECT], events[E.INTENT_PROTECT])
    assert got[SC.EVENT_ATTACK] is got[SC.EVENT_ATTACK_ARRIVED]
    attack = events["attack"]
    either = 1 - (1 - attack[:, 0]) * (1 - attack[:, 1])
    assert np.allclose(got[SC.EVENT_EITHER], either)
    # one attack candidate of three legal actions, two legal targets
    assert attack[0, 0].tolist() == pytest.approx([1 / 6, 1 / 6])
    assert got[SC.EVENT_EITHER][4, 1] == 0.0  # no foe in slot b
    with pytest.raises(SC.ScorecardError):
        SC.event_predictions({"action": np.zeros((2, 2, 3))}, batch)


def test_thresholds_hold_the_pre_registered_value_and_the_common_ones():
    assert SC.thresholds_for(SC.EVENT_SWITCH) == (0.35, 0.6, 0.8, 0.9)
    assert SC.thresholds_for(SC.EVENT_PROTECT) == (0.5, 0.6, 0.8, 0.9)
    assert SC.thresholds_for(SC.EVENT_ATTACK) == (0.6, 0.7, 0.8, 0.9)
    assert set(SC.PRE_REGISTERED) == set(SC.EVENTS) == set(SC.EVENT_TEXT)


# --- reliability, Brier, decisions ----------------------------------------------


def test_reliability_table_arithmetic():
    p = np.array([0.05, 0.05, 0.25, 0.5, 0.95, 1.0, 0.0])
    y = np.array([0, 1, 1, 0, 1, 1, 0])
    table = SC.reliability_table(p, y)
    assert len(table) == 10 and sum(row["slots"] for row in table) == 7
    assert [row["slots"] for row in table] == [3, 0, 1, 0, 0, 1, 0, 0, 0, 2]
    assert table[0]["from"] == 0.0 and table[0]["to"] == 0.1
    assert table[0]["predicted"] == pytest.approx(0.1 / 3)
    assert table[0]["observed"] == pytest.approx(1 / 3)
    assert table[1]["predicted"] is None and table[1]["observed"] is None
    assert table[5]["observed"] == 0.0  # 0.5 belongs to the bin that starts there
    assert table[9]["predicted"] == pytest.approx(0.975) and table[9]["observed"] == 1
    assert SC.reliability_table(np.array([]), np.array([]))[3]["slots"] == 0


def test_brier_summary_arithmetic():
    p = np.array([0.8, 0.2, 0.2, 0.2])
    y = np.array([1, 0, 0, 1])
    got = SC.brier_summary(p, y)
    brier = (0.04 + 0.04 + 0.04 + 0.64) / 4
    assert got["slots"] == 4 and got["observed"] == 0.5
    assert got["predicted"] == pytest.approx(0.35)
    assert got["brier"] == pytest.approx(brier)
    assert got["brier_base_rate"] == 0.25
    assert got["skill"] == pytest.approx(1 - brier / 0.25)
    assert SC.brier_summary(np.full(4, 0.5), y)["skill"] == pytest.approx(0.0)
    assert SC.brier_summary(p, np.zeros(4))["skill"] is None  # no variance
    assert SC.brier_summary(np.array([]), np.array([]))["brier"] is None


def test_decision_row_arithmetic():
    p = np.array([0.9, 0.7, 0.7, 0.4, 0.1])
    y = np.array([1, 1, 0, 1, 0])
    got = SC.decision_row(p, y, 0.7)
    assert got == {
        "threshold": 0.7,
        "slots": 3,
        "share": 0.6,
        "precision": 2 / 3,
        "predicted": pytest.approx((0.9 + 0.7 + 0.7) / 3),
        "recall": 2 / 3,
    }
    none = SC.decision_row(p, y, 0.95)
    assert none["slots"] == 0 and none["precision"] is None and none["recall"] == 0
    assert none["predicted"] is None


# --- verdict logic --------------------------------------------------------------


def difference(diff: float, low: float | None, high: float | None) -> dict[str, Any]:
    return {"diff": diff, "low": low, "high": high}


def test_r1_passes_only_with_both_intervals_entirely_below_zero():
    below = difference(-0.2, -0.3, -0.1)
    touching = difference(-0.2, -0.4, 0.0)
    above = difference(0.1, 0.05, 0.2)
    ladder, test, late = SC.SET_LADDER, SC.SET_TEST, SC.SET_TIME
    assert SC.r1_verdict({ladder: below, test: below})["verdict"] == SC.VERDICT_PASS
    # the time slice is reported, never gated
    assert (
        SC.r1_verdict({ladder: below, test: below, late: above})["verdict"]
        == SC.VERDICT_PASS
    )
    for a, b in ((below, touching), (touching, below), (below, above), (above, below)):
        got = SC.r1_verdict({ladder: a, test: b})
        assert got["verdict"] == SC.VERDICT_FAIL
    got = SC.r1_verdict({ladder: below, test: touching})
    assert got["interval_below_zero"] == {ladder: True, test: False}
    # a point estimate below zero without an interval cannot pass
    no_interval = difference(-0.5, None, None)
    assert SC.r1_verdict({ladder: below, test: no_interval})["verdict"] == "fails"
    assert SC.r1_verdict({ladder: below})["verdict"] == SC.VERDICT_FAIL
    assert SC.r1_verdict({ladder: below, test: None})["verdict"] == SC.VERDICT_FAIL


def test_r2_elo_stays_only_with_every_gain_at_the_bar_on_both_sets():
    ladder, test = SC.SET_LADDER, SC.SET_TEST
    big = {ladder: 0.03, test: 0.02}
    small = {ladder: 0.03, test: 0.019}
    assert SC.ELO_GAIN_BAR == 0.02
    assert SC.r2_verdict(big, big)["verdict"] == SC.ELO_STAYS
    assert SC.r2_verdict(big)["verdict"] == SC.ELO_OPEN  # the retrain is missing
    assert SC.r2_verdict(small)["verdict"] == SC.ELO_GOES
    assert SC.r2_verdict(small, big)["verdict"] == SC.ELO_GOES
    assert SC.r2_verdict(big, small)["verdict"] == SC.ELO_GOES
    assert SC.r2_verdict({ladder: 0.05, test: None})["verdict"] == SC.ELO_GOES
    assert SC.r2_verdict({ladder: 0.05})["verdict"] == SC.ELO_GOES
    assert SC.r2_verdict({ladder: -0.01, test: 0.5})["verdict"] == SC.ELO_GOES
    got = SC.r2_verdict(big, small)
    assert got["reaches_bar"] == {"shuffle": True, "blind_retrain": False}
    assert SC.r2_verdict(big, big, bar=0.05)["verdict"] == SC.ELO_GOES


# --- slices ---------------------------------------------------------------------


def test_slices_partition_the_examples_within_each_family():
    batch = blank(8)
    for i in range(8):
        put(batch, i, 0)
        put(batch, i, 1)
    batch["m_sheet"][:] = [0, 0, 1, 1, 2, 2, 2, 0]
    batch["game_flag"][:4, F.G_BO3] = 1
    batch["turn"][:] = [1, 1, 2, 3, 1, 9, 2, 2]
    batch["elo"][:, 0] = [1099, 1100, 1199, 1250, 1399, 1400, 1800, 1234]
    batch["elo_known"][:, 0] = [1, 1, 1, 1, 1, 1, 1, 0]
    batch["foe_mon"][7, 1] = -1
    masks = SC.slice_masks(batch)
    assert list(masks)[0] == SC.SLICE_ALL and masks[SC.SLICE_ALL].all()
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    assert counts["sheet closed"] == 3 and counts["sheet open"] == 2
    assert counts["sheet unknown"] == 3
    assert counts["bo1"] == 4 and counts["bo3"] == 4
    assert counts["turn 1"] == 3 and counts["turn 2 and later"] == 5
    assert counts["actor Elo below 1100"] == 1
    assert counts["actor Elo 1100-1199"] == 2
    assert counts["actor Elo 1200-1299"] == 1
    assert counts["actor Elo 1300-1399"] == 1
    assert counts["actor Elo 1400 and above"] == 2
    assert counts["actor Elo unknown"] == 1
    assert counts["four active Pokemon"] == 7
    assert counts["fewer than four active"] == 1
    families = ("sheet", "bo", "turn", "actor Elo", "f")
    for prefix in families:
        total = sum(c for name, c in counts.items() if name.startswith(prefix))
        assert total == 8, prefix
    # without the meta array the sheet slices fall back to the feature
    del batch["m_sheet"]
    batch["game_flag"][:, F.G_ACTOR_SHEET] = F.SHEET_OPEN
    assert SC.slice_masks(batch)["sheet open"].all()


def test_time_slice_flag_matches_the_dataset_builder():
    from datagen import oppmodel_build_dataset as B

    assert SC.FLAG_TIME_SLICE == B.FLAG_TIME_SLICE
    assert (F.SHEET_CLOSED, F.SHEET_OPEN, F.SHEET_UNKNOWN) == tuple(
        B.SHEET_STATES.index(name) for name in ("closed", "open", "unknown")
    )


# --- one evaluation set, by hand ------------------------------------------------


def test_score_set_matches_a_hand_calculation():
    batch = small_batch()
    uniform = F.uniform_prediction(batch)
    better = sharp(batch, {0: 0, 1: 1, 2: F.switch_index(2, C)})
    found = SC.score_set(
        batch, {"uniform": uniform, "sharp": better}, "uniform", resamples=200, seed=1
    )
    assert found["examples"] == 4 and found["games"] == 2
    assert found["slot_turns"] == 4
    assert found["slots"] == {
        "scored": 4,
        "visible": 3,
        "censored": 1,
        "target_labels": 2,
    }
    # five legal actions: two candidates, OTHER, two switch destinations
    row = found["predictors"]["uniform"]
    action = (3 * LN(5) + LN(5 / 2)) / 4
    target = LN(2) / 4  # one aimed move with two legal targets; Protect has one
    assert row["parts"]["action"] == pytest.approx(action)
    assert row["parts"]["target"] == pytest.approx(target)
    assert row["fine_nll"]["value"] == pytest.approx(action + target)
    assert row["parts"]["visible"] == pytest.approx((3 * LN(5) + LN(2)) / 3)
    assert row["parts"]["censored"] == pytest.approx(LN(5 / 2))
    assert row["target_nll_per_label"] == pytest.approx(LN(2) / 2)
    assert row["action_nll_visible"] == pytest.approx(LN(5))
    assert row["difference"] is None
    assert row["intent_accuracy"] is None and row["intent_labels"] == 0
    assert row["fine_labels"] == 3 and row["action_labels"] == 3

    row = found["predictors"]["sharp"]
    sharp_action = (3 * -LN(0.9) + LN(5 / 2)) / 4
    assert row["parts"]["action"] == pytest.approx(sharp_action)
    assert row["fine_nll"]["value"] == pytest.approx(sharp_action + target)
    assert row["difference"]["fine"]["diff"] == pytest.approx(sharp_action - action)
    assert row["difference"]["target"]["diff"] == pytest.approx(0.0, abs=1e-12)
    assert row["difference"]["censored"]["diff"] == pytest.approx(0.0, abs=1e-12)
    assert row["difference"]["visible"]["diff"] == pytest.approx(-LN(0.9) - LN(5))
    assert row["difference"]["fine"]["high"] < 0
    assert row["action_top1"] == 1.0 and row["fine_top3"] == 1.0
    # game 0 holds examples 0 and 1, game 1 the switch and the hidden slot
    assert found["slices"][SC.SLICE_ALL] == {"slots": 4, "games": 2}
    assert found["slices"]["turn 2 and later"] == {"slots": 4, "games": 2}
    assert found["slices"]["turn 1"] == {"slots": 0, "games": 0}
    assert row["slices"]["turn 1"]["fine_nll"] is None
    assert row["slices"]["bo1"]["difference"]["diff"] == pytest.approx(
        sharp_action - action
    )

    switch = row["events"][SC.EVENT_SWITCH]
    assert switch["slots"] == 4 and switch["observed"] == 0.25
    # 0.9 on one action leaves 0.025 on each of the other four legal ones
    assert switch["predicted"] == pytest.approx((0.05 + 0.05 + 0.925 + 0.4) / 4)
    decision = {d["threshold"]: d for d in switch["decisions"]}
    assert decision[0.35]["pre_registered"] and not decision[0.6]["pre_registered"]
    assert decision[0.35]["slots"] == 2 and decision[0.35]["share"] == 0.5
    assert decision[0.35]["precision"]["value"] == 0.5
    assert decision[0.9]["slots"] == 1 and decision[0.9]["precision"]["value"] == 1.0
    assert decision[0.9]["recall"] == 1.0
    assert sum(r["slots"] for r in switch["reliability"]) == 4
    brier = (0.05**2 + 0.05**2 + 0.075**2 + 0.4**2) / 4
    assert switch["brier"] == pytest.approx(brier)
    assert switch["skill"]["value"] == pytest.approx(1 - brier / (0.25 * 0.75))
    assert switch["skill"]["low"] <= switch["skill"]["value"] <= switch["skill"]["high"]
    attack = found["predictors"]["uniform"]["events"]
    assert attack[SC.EVENT_ATTACK]["slots"] == 6  # three visible clicks, two foes
    assert attack[SC.EVENT_ATTACK_ARRIVED]["slots"] == 8  # plus the fainted slot
    assert attack[SC.EVENT_ATTACK]["observed"] == pytest.approx(1 / 6)
    assert attack[SC.EVENT_ATTACK_ARRIVED]["observed"] == pytest.approx(1 / 8)
    assert found["events"][SC.EVENT_PROTECT] == {
        "known": 4,
        "observed": 0.25,
        "impossible": 0,
    }
    # every slot can switch and holds a Protect: nothing is impossible
    assert switch["impossible"] == 0 and switch["slots_where_possible"] == 4
    assert switch["skill_where_possible"]["value"] == pytest.approx(
        switch["skill"]["value"]
    )
    assert decision[0.35]["predicted"] == pytest.approx((0.925 + 0.4) / 2)
    assert decision[0.35]["share_where_possible"] == 0.5
    json.dumps(SC._plain(found), allow_nan=False)


def test_a_probability_exactly_at_the_threshold_counts_as_at_or_above():
    batch = small_batch()
    pred = F.uniform_prediction(batch)
    row = np.zeros(F.action_size(C))
    row[[0, 1]] = 0.2
    row[[F.switch_index(2, C), F.switch_index(3, C)]] = 0.3
    pred["action"][2, 0] = row  # the example that switches: P(switch) = 0.6
    found = SC.score_set(batch, {"p": pred}, "p", resamples=50)
    decisions = found["predictors"]["p"]["events"][SC.EVENT_SWITCH]["decisions"]
    at = {d["threshold"]: d for d in decisions}
    assert at[0.6]["slots"] == 1 and at[0.6]["precision"]["value"] == 1.0
    assert at[0.35]["slots"] == 4 and at[0.35]["precision"]["value"] == 0.25
    assert at[0.8]["slots"] == 0 and at[0.8]["precision"]["value"] is None


def test_intent_accuracy_and_nll_are_taken_over_the_labelled_slots():
    batch = small_batch()
    switch, protect, foe_a = (
        E.INTENT_CLASSES.index(name)
        for name in (E.INTENT_SWITCH, E.INTENT_PROTECT, E.INTENT_ATTACK_FOE_A)
    )
    batch["y_intent"][:3, 0] = (foe_a, protect, switch)
    intent = np.full((16, F.N_TARGET), -1, dtype=np.int8)
    intent[11, F.T_FOE_A] = foe_a  # candidate 0 of put(): the aimed attack
    intent[11, F.T_FOE_B] = E.INTENT_CLASSES.index(E.INTENT_ATTACK_FOE_B)
    intent[12, F.T_AUTO] = protect  # candidate 1: the Protect
    tables = F.Tables(
        np.zeros((1, 1), np.float32),
        np.zeros((16, 1), np.float32),
        intent,
        np.zeros(16, np.uint16),
        ("",),
        tuple(str(i) for i in range(16)),
    )
    better = sharp(batch, {0: 0, 1: 1, 2: F.switch_index(2, C)})
    found = SC.score_set(batch, {"sharp": better}, "sharp", tables=tables, resamples=0)
    row = found["predictors"]["sharp"]
    assert row["intent_labels"] == 3 and row["intent_accuracy"] == 1.0
    # the attack splits 0.9 over two foes; Protect keeps 0.9; a switch 0.925
    assert row["intent_nll"] == pytest.approx(-(LN(0.45) + LN(0.9) + LN(0.925)) / 3)
    uniform = F.uniform_prediction(batch)
    found = SC.score_set(batch, {"u": uniform}, "u", tables=tables, resamples=0)
    # uniform: switching (two of five actions) is the first choice everywhere
    assert found["predictors"]["u"]["intent_accuracy"] == pytest.approx(1 / 3)


def test_score_set_reads_elo_modes_and_blind_pairs():
    batch = small_batch()
    uniform = F.uniform_prediction(batch)
    better = sharp(batch, {0: 0, 1: 1, 2: F.switch_index(2, C)})
    worse = sharp(batch, {0: 0, 1: 1, 2: F.switch_index(2, C)}, strength=0.6)
    fine = {
        name: F.slot_nll(pred, batch)["fine"]
        for name, pred in (("uniform", uniform), ("worse", worse))
    }
    found = SC.score_set(
        batch,
        {"uniform": uniform, "sharp": better, "blind": worse},
        "uniform",
        elo={"sharp": {F.ELO_SHUFFLE: [fine["uniform"], fine["worse"]]}},
        pairs=[("sharp", "blind")],
        resamples=100,
    )
    gap = 3 * (-LN(0.9) + LN(0.6)) / 4  # sharp minus worse
    row = found["predictors"]["sharp"]["elo"][F.ELO_SHUFFLE]
    by_seed = row["keep_minus_mode_by_seed"]
    assert by_seed[1] == pytest.approx(gap)
    assert row["keep_minus_mode"]["diff"] == pytest.approx(np.mean(by_seed))
    assert row["fine_nll"] == pytest.approx(
        (fine["uniform"].sum() + fine["worse"].sum()) / 8
    )
    assert "elo" not in found["predictors"]["uniform"]
    assert len(found["pairs"]) == 1
    assert (found["pairs"][0]["model"], found["pairs"][0]["blind"]) == (
        "sharp",
        "blind",
    )
    assert found["pairs"][0]["model_minus_blind"]["diff"] == pytest.approx(gap)
    with pytest.raises(SC.ScorecardError):
        SC.score_set(batch, {"sharp": better}, "uniform")


def test_sanity_block_measures_what_normalising_changes():
    batch = small_batch()
    clean = F.uniform_prediction(batch)
    got = SC.sanity_block(clean, batch, seconds=0.002)
    assert got["unchanged"] and max(got["max_abs_change"].values()) == 0.0
    assert got["non_finite"] == 0 and got["floor_hits"] == {"action": 0, "target": 0}
    assert got["microseconds_per_example"] == pytest.approx(500.0)
    leaky = {name: array.copy() for name, array in clean.items()}
    leaky["action"][0, 0, 3] = 0.2  # an illegal candidate
    leaky["target"][0, 0, 0, F.T_FOE_A] = 0.0  # the label's target gets nothing
    leaky["action"][2, 0, F.switch_index(2, C)] = 0.0  # the label's action too
    got = SC.sanity_block(leaky, batch)
    assert not got["unchanged"] and got["max_abs_change"]["action"] > 0.03
    assert got["floor_hits"] == {"action": 1, "target": 1}
    assert "predict_seconds" not in got


# --- the whole run --------------------------------------------------------------


def make_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *more: str
) -> dict[str, Any]:
    """``run`` on a synthetic dataset and two synthetic predictors."""
    batch = blank(12, games=np.repeat(np.arange(6), 2))
    for i in range(12):
        put(batch, i, 0, bench=(2, 3))
        if i % 3 == 0:
            label_move(batch, i, 0, 0, target=F.T_FOE_A, attack=(1, 0))
        elif i % 3 == 1:
            label_move(batch, i, 0, 1, target=F.T_AUTO, protect=True)
        else:
            label_switch(batch, i, 0, 2)
    batch["m_split"] = np.array([0] * 4 + [1] * 8, dtype=np.uint8)
    # Three of the eight test examples are in the time slice: not half, so
    # "the slice" and "everything but the slice" cannot be confused.
    batch["m_flag"][9:] = SC.FLAG_TIME_SLICE
    batch["m_actor"] = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], dtype=np.uint32)
    batch["elo"][:, 0] = 1000 + 50 * np.arange(12)
    batch["elo_known"][:, 0] = 1
    manifest = {"splits": [SC.SET_LADDER, SC.SET_TEST], "tag": "synthetic"}

    class Skewed(Recorder):
        """Likes the first candidate; more so in every other rating band."""

        kind = "table"

        def __init__(self, name: str, reads_elo: bool) -> None:
            super().__init__()
            self.name, self.reads_elo = name, reads_elo

        def predict(self, batch: F.Batch) -> dict[str, np.ndarray]:
            pred = F.uniform_prediction(batch)
            band = (batch["elo"][:, 0] // 100) % 2 == 1
            lift = 0.3 + (0.3 * band if self.reads_elo else 0)
            pred["action"][:, 0, 0] += lift
            return F.normalize_prediction(pred, batch)

    made = {"keeps": Skewed("keeps", True), "blind": Skewed("blind", False)}

    def fake_dataset(directory: Any, splits: Any = None) -> Any:
        return batch, manifest

    def fake_entry(spec: str, taken: Any, sha: Any) -> SC.Entry:
        return SC.Entry(spec, "table", made[spec], {"path": spec, "sha256": "0" * 64})

    monkeypatch.setattr(SC.F, "load_dataset", fake_dataset)
    monkeypatch.setattr(SC.F.Featurizer, "load", lambda directory: _Featurizer())
    monkeypatch.setattr(SC, "sha256_file", lambda path: "f" * 64)
    monkeypatch.setattr(SC, "load_artifact_entry", fake_entry)
    args = SC.parse_args(
        [
            "--artifact",
            "blind",
            "--artifact",
            "keeps",
            "--reference",
            "blind",
            "--elo-blind",
            "blind",
            "--out",
            str(tmp_path),
            "--resamples",
            "100",
            "--shuffles",
            "2",
            "--note",
            "a synthetic run",
            *more,
        ]
    )
    return SC.run(args, log=lambda text: None)


class _Featurizer:
    """Stand-in for the dataset's featurizer: no dex signature difference."""

    signature_diff: list[str] = []
    tables = None


def test_run_writes_a_card_and_a_readme_rendered_from_it(tmp_path, monkeypatch):
    card = make_card(tmp_path, monkeypatch)
    stored = json.loads((tmp_path / SC.CARD_JSON).read_text(encoding="utf-8"))
    assert stored == card and card["format"] == SC.FORMAT
    assert card["order"] == ["blind", "keeps"]
    assert card["elo_pair"] == {"model": "keeps", "blind": "blind"}
    assert [card["sets"][name]["examples"] for name in SC.SETS] == [4, 8, 3]
    # The time slice is the FLAGGED test examples (rows 9, 10, 11: one move at
    # foe a, one Protect, one switch), not the five unflagged ones.
    late = card["sets"][SC.SET_TIME]
    assert late["slots"]["scored"] == 3 and late["games"] == 2
    assert late["events"][SC.EVENT_SWITCH] == {
        "known": 3,
        "observed": pytest.approx(1 / 3),
        "impossible": 0,
    }
    assert card["set_labels"][SC.SET_TIME].endswith("not an out-of-time test)")
    assert card["dataset"]["accounts_in_both_gated_sets"] == {
        "accounts": 0,
        "ladder_holdout_examples": 0,
        "test_examples": 0,
    }
    assert card["sets"][SC.SET_TEST]["clusters"] == {"game": 4, "account": 4}
    assert card["predictors"]["keeps"]["reads_elo"] is True
    assert card["predictors"]["blind"]["reads_elo"] is False
    assert card["predictors"]["blind"]["elo_blind_retrain"] is True
    assert card["warnings"] == []
    assert card["r2"]["does_not_read_elo"] == ["blind"]
    found = card["r2"]["predictors"]["keeps"]
    assert set(found["shuffle_gain"]) == set(SC.SETS)
    assert found["blind_retrain"]["blind"] == "blind"
    assert found["verdict"] in (SC.ELO_STAYS, SC.ELO_GOES)
    # the retrain's gain is minus the paired difference of the two models
    pair = card["sets"][SC.SET_TEST]["pairs"][0]["model_minus_blind"]["diff"]
    assert found["blind_retrain"]["gain"][SC.SET_TEST] == pytest.approx(-pair)
    row = card["sets"][SC.SET_TEST]["predictors"]["keeps"]
    assert row["difference"]["fine"]["diff"] == pytest.approx(pair)
    assert set(row["elo"]) == set(SC.ELO_MODES)
    shuffled = row["elo"][F.ELO_SHUFFLE]["keep_minus_mode"]["diff"]
    assert shuffled != 0 and found["shuffle_gain"][SC.SET_TEST] == -shuffled
    assert len(row["elo"][F.ELO_SHUFFLE]["keep_minus_mode_by_seed"]) == 2
    assert "sanity" in row and row["sanity"]["unchanged"]
    assert (
        "predict_seconds"
        not in card["sets"][SC.SET_TIME]["predictors"]["keeps"]["sanity"]
    )
    assert card["r1"]["predictors"]["keeps"]["verdict"] in ("passes", "fails")
    assert "blind" not in card["r1"]["predictors"]  # the reference itself
    ladder = card["sets"][SC.SET_LADDER]["predictors"]
    scores = {name: ladder[name]["fine_nll"]["value"] for name in ladder}
    assert scores["keeps"] != scores["blind"]
    assert card["r3"]["best"] == min(scores, key=lambda name: scores[name])

    readme = (tmp_path / SC.CARD_MD).read_text(encoding="utf-8")
    assert readme == SC.render_readme(card)
    assert "**a synthetic run**" in readme and "R1, quality" in readme
    assert "No neural model was scored" in readme
    for text in (
        "accounts resampled",
        "cannot happen",
        "skill where it can [95%]",
        "mean predicted",
        "two opposing Pokemon on the field",
        "not an out-of-time test",
    ):
        assert text in readme, text
    by_account = card["r1"]["predictors"]["keeps"]["difference_by_account"]
    assert set(by_account) == set(SC.SETS)
    assert by_account[SC.SET_TEST]["diff"] == pytest.approx(pair)
    (tmp_path / SC.CARD_MD).unlink()
    assert SC.main(["--out", str(tmp_path), "--render-only"]) == 0
    assert (tmp_path / SC.CARD_MD).read_text(encoding="utf-8") == readme
    assert SC.main(["--out", str(tmp_path / "missing"), "--render-only"]) == 1


def test_warnings_name_what_weakens_a_reading(tmp_path, monkeypatch):
    card = make_card(tmp_path, monkeypatch)
    assert SC.warnings_of(card) == [] and "Warnings: none." in SC.render_readme(card)
    card["dataset"]["dex_signature_diff"] = ["n_moves"]
    card["predictors"]["keeps"]["fitted_on_this_dataset"] = False
    card["predictors"]["keeps"]["dex_signature_diff"] = ["n_species"]
    card["predictors"]["blind"]["reads_elo"] = True
    card["predictors"]["keeps"]["reads_elo"] = False
    sanity = card["sets"][SC.SET_TEST]["predictors"]["keeps"]["sanity"]
    sanity["unchanged"], sanity["max_abs_change"]["action"] = False, 0.25
    found = SC.warnings_of(card)
    assert len(found) == 6
    for text in ("another dex than", "another build", "n_species", "2.5e-01"):
        assert any(text in line for line in found), text
    assert any("blind was given as the Elo-blind retrain" in line for line in found)
    assert any("keeps is paired with" in line for line in found)
    card["warnings"] = found
    readme = SC.render_readme(card)
    assert "Warnings: none." not in readme and f"- {found[0]}" in readme


def test_an_existing_card_is_not_written_over_without_overwrite(
    tmp_path, monkeypatch, capsys
):
    card = make_card(tmp_path, monkeypatch)
    stored = (tmp_path / SC.CARD_JSON).read_bytes()
    readme = (tmp_path / SC.CARD_MD).read_bytes()
    with pytest.raises(SC.ScorecardError, match="--overwrite"):
        make_card(tmp_path, monkeypatch)
    assert (tmp_path / SC.CARD_JSON).read_bytes() == stored
    assert (tmp_path / SC.CARD_MD).read_bytes() == readme
    # The command line says so on its last line and writes nothing.
    base = ["--artifact", "keeps", "--reference", "keeps", "--out", str(tmp_path)]
    assert SC.main([*base, "--resamples", "10"]) == 1
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.startswith("SCORECARD_FAILED") and "--overwrite" in last
    assert (tmp_path / SC.CARD_JSON).read_bytes() == stored
    # --render-only reads the card and is not a run: no flag needed.
    assert SC.main(["--out", str(tmp_path), "--render-only"]) == 0
    # With the flag the card is replaced.
    again = make_card(tmp_path, monkeypatch, "--overwrite", "--note", "second run")
    assert again["note"] == "second run" != card["note"]
    assert json.loads((tmp_path / SC.CARD_JSON).read_text())["note"] == "second run"


def test_any_failure_ends_in_one_scorecard_failed_line(tmp_path, monkeypatch, capsys):
    make_card(tmp_path, monkeypatch)
    capsys.readouterr()
    base = ["--artifact", "keeps", "--reference", "keeps", "--resamples", "10"]

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise ZeroDivisionError("an unforeseen\nfailure")

    # Not a ScorecardError: still one last line, no traceback, nothing left.
    monkeypatch.setattr(SC, "score_set", explode)
    out = tmp_path / "deep" / "run"
    assert SC.main([*base, "--out", str(out)]) == 1
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1] == "SCORECARD_FAILED ZeroDivisionError: an unforeseen failure"
    assert not (tmp_path / "deep").exists()
    # A path that cannot be made fails before anything is predicted.
    blocker = tmp_path / "a_file"
    blocker.write_text("x", encoding="utf-8")
    calls: list[int] = []
    monkeypatch.setattr(
        SC, "guarded_predict", lambda *a, **k: calls.append(1) or explode()
    )
    assert SC.main([*base, "--out", str(blocker / "x")]) == 1
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.startswith("SCORECARD_FAILED cannot write to") and not calls
    # An unreadable card in --render-only as well.
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / SC.CARD_JSON).write_text("{}", encoding="utf-8")
    assert SC.main(["--out", str(broken), "--render-only"]) == 1
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.startswith("SCORECARD_FAILED cannot render")


def test_players_in_both_gated_sets_are_counted_and_warned_about(tmp_path, monkeypatch):
    ladder = {"m_actor": np.array([1, 1, 2, 3], dtype=np.uint32)}
    test = {"m_actor": np.array([3, 4, 4, 1, 5, 1], dtype=np.uint32)}
    assert SC.shared_accounts(ladder, test) == {
        "accounts": 2,
        "ladder_holdout_examples": 3,
        "test_examples": 3,
    }
    assert SC.shared_accounts({}, test) is None
    card = make_card(tmp_path, monkeypatch)
    assert SC.warnings_of(card) == []
    card["dataset"]["accounts_in_both_gated_sets"] = SC.shared_accounts(ladder, test)
    found = SC.warnings_of(card)
    assert len(found) == 1 and found[0].startswith("2 accounts act in both (a) and (b)")
    assert "3 examples of (a), 3 of (b)" in found[0]


def test_run_refuses_an_unknown_reference_and_an_empty_run(tmp_path, monkeypatch):
    make_card(tmp_path, monkeypatch)
    base = ["--artifact", "keeps", "--out", str(tmp_path / "x"), "--resamples", "10"]
    with pytest.raises(SC.ScorecardError, match="reference"):
        SC.run(SC.parse_args(base), log=lambda text: None)
    with pytest.raises(SC.ScorecardError, match="no predictor"):
        SC.run(SC.parse_args(["--out", str(tmp_path / "y")]), log=lambda text: None)
    with pytest.raises(SC.ScorecardError, match="--elo-model"):
        SC.run(
            SC.parse_args([*base, "--reference", "keeps", "--elo-blind", "keeps"]),
            log=lambda text: None,
        )
    assert not (tmp_path / "x").exists() and not (tmp_path / "y").exists()


def test_a_failing_predictor_stops_the_run(tmp_path, monkeypatch):
    make_card(tmp_path, monkeypatch)

    class Broken(Recorder):
        def predict(self, batch: F.Batch) -> dict[str, np.ndarray]:
            self.counters["predict_error:KeyError"] += 1
            return F.uniform_prediction(batch)

    monkeypatch.setattr(
        SC,
        "load_artifact_entry",
        lambda spec, taken, sha: SC.Entry(spec, "table", Broken(), {"path": spec}),
    )
    out = tmp_path / "broken"
    code = SC.main(["--artifact", "b", "--reference", "b", "--out", str(out)])
    assert code == 1 and not out.exists()


# --- hygiene --------------------------------------------------------------------


def test_script_and_adapter_hold_no_dex_name_and_no_bare_assert():
    vocab = F.Vocab.build()
    known = set(E.moves_dex()) | set(E.pokedex())
    known |= set(vocab.items[3:]) | set(vocab.abilities[3:])
    for path in (
        ROOT / "evaluation/oppmodel_scorecard.py",
        ROOT / "vgc_bench/src/oppmodel/legacy.py",
    ):
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
        assert not found, (path.name, sorted(found))
        asserts = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assert)]
        assert not asserts, (path.name, asserts)


# --- the real dataset and tables (skipped when absent) --------------------------


@pytest.mark.skipif(
    not (DATASET / "manifest.json").is_file()
    or not (FITTED / "flags_table.pt").is_file(),
    reason="results_oppmodel/v1_ondisk or results_oppmodel/tables_v1 is absent",
)
def test_real_tables_through_the_whole_scorecard(tmp_path):
    code = SC.main(
        [
            "--dataset",
            str(DATASET),
            "--artifact",
            str(FITTED / "species_table.pt"),
            "--artifact",
            str(FITTED / "flags_table.pt"),
            "--artifact",
            str(FITTED / "elo_table.pt"),
            "--elo-blind",
            str(FITTED / "flags_table.pt"),
            "--elo-model",
            "elo_table",
            "--limit",
            "400",
            "--resamples",
            "200",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    card = json.loads((tmp_path / SC.CARD_JSON).read_text(encoding="utf-8"))
    assert card["order"] == ["species_table", "flags_table", "elo_table"]
    assert card["settings"]["reference"] == "flags_table"
    assert card["dataset"]["limit_per_set"] == 400
    for name in SC.GATED_SETS:
        found = card["sets"][name]
        assert found["examples"] == 400 and found["games"] >= 2
        flags = found["predictors"]["flags_table"]
        species = found["predictors"]["species_table"]
        assert flags["difference"] is None and flags["intent_accuracy"] is not None
        gap = species["fine_nll"]["value"] - flags["fine_nll"]["value"]
        assert species["difference"]["fine"]["diff"] == pytest.approx(gap)
        assert species["difference"]["fine"]["low"] <= gap
        assert gap <= species["difference"]["fine"]["high"]
        parts = flags["parts"]
        assert parts["action"] + parts["target"] == pytest.approx(
            flags["fine_nll"]["value"]
        )
        for row in found["predictors"].values():
            assert row["sanity"]["unchanged"] and row["sanity"]["non_finite"] == 0
    assert all(info["fitted_on_this_dataset"] for info in card["predictors"].values())
    assert card["predictors"]["elo_table"]["reads_elo"] is True
    assert card["predictors"]["flags_table"]["reads_elo"] is False
    assert card["r2"]["predictors"]["elo_table"]["blind_retrain"]["blind"] == (
        "flags_table"
    )
    assert "legacy" not in card["predictors"]
    readme = (tmp_path / SC.CARD_MD).read_text(encoding="utf-8")
    assert "first 400 examples" in readme and "## R3" in readme


@pytest.mark.skipif(
    not (DATASET / "manifest.json").is_file()
    or not (FITTED / "fit_report.json").is_file(),
    reason="results_oppmodel/v1_ondisk or results_oppmodel/tables_v1 is absent",
)
def test_fine_nll_agrees_with_the_fit_report_on_the_ladder_holdout():
    """The harness reproduces the number the table fit wrote for the same set."""
    from vgc_bench.src.oppmodel.artifact import load_predictor

    report = json.loads((FITTED / "fit_report.json").read_text(encoding="utf-8"))
    batch, _ = F.load_dataset(DATASET, splits=[SC.SET_LADDER])
    table = load_predictor(FITTED / "flags_table.pt").predictor
    pred, _ = SC.guarded_predict(table, batch, name="flags_table")
    found = SC.score_set(batch, {"flags_table": pred}, "flags_table", resamples=0)
    wanted = report["tables"]["flags_table"]["scores"][SC.SET_LADDER]
    row = found["predictors"]["flags_table"]
    assert row["fine_nll"]["value"] == pytest.approx(wanted["fine_nll"], abs=1e-9)
    assert row["fine_nll"]["low"] is None
    assert found["slots"]["scored"] == wanted["slots_scored"]
    assert found["slots"]["censored"] == wanted["slots_censored"]
    assert row["fine_top1"] == pytest.approx(wanted["fine_top1"], abs=1e-9)
