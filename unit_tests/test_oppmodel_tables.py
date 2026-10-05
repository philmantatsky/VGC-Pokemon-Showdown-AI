"""Opponent predictor count tables: counts, shrinkage, legality, storage."""

from __future__ import annotations

import ast
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from vgc_bench.src.oppmodel import events as E
from vgc_bench.src.oppmodel import features as F
from vgc_bench.src.oppmodel import tables as T

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "results_oppmodel" / "v1_ondisk"
FITTED = ROOT / "results_oppmodel" / "tables_v1"

C = 4  # candidates per slot in the synthetic batches
OTHER = F.other_index(C)
INF = math.inf
TINY = 1e-9

ATTACK = F.CAND_DAMAGING | F.CAND_AIMED
GUARD = F.CAND_PROTECT
FIRST = F.CAND_DAMAGING | F.CAND_AIMED | F.CAND_FIRST_TURN
STATUS = F.CAND_AIMED
FOES = (1 << F.T_FOE_A) | (1 << F.T_FOE_B)
FOES_ALLY = FOES | (1 << F.T_ALLY)
SELF_ONLY = 1 << F.T_AUTO
MOVE_A, MOVE_B, MOVE_G, MOVE_F = 11, 12, 13, 14
KEY = 7

# Raw counts, nothing learned on top: every strength 0, no offsets, no EM.
RAW = T.TableConfig(
    usage=T.USAGE_SHARE,
    move_strength=0.0,
    key_strength=0.0,
    cell_strength=0.0,
    switch_key_strength=0.0,
    switch_cell_strength=0.0,
    flag_effect_strength=INF,
    target_key_strength=0.0,
    target_move_strength=0.0,
    mega_strength=0.0,
    elo_strength=INF,
    other_boost=1.0,
    unseen_count=0.0,
    floor=TINY,
    iterations=0,
    reveal_offsets=False,
    class_offsets=False,
    target_context_strength=0.0,
)


def blank(n: int) -> F.Batch:
    """``n`` examples with nothing on the field, labels and meta arrays included."""
    batch = {
        name: np.zeros((n, *spec.shape), dtype=spec.dtype)
        for name, spec in F.layout(C).items()
    }
    for name in ("act_mon", "foe_mon", "y_kind", "y_action", "y_target", "y_mega"):
        batch[name][:] = -1
    batch["y_intent"][:] = -1
    batch["turn"][:] = 3
    batch["m_weight"] = np.ones(n, dtype=np.float32)
    batch["m_split"] = np.zeros(n, dtype=np.uint8)
    batch["m_battle"] = np.arange(n, dtype=np.int32)
    return batch


def put(
    batch: F.Batch,
    i: int,
    slot: int = 0,
    *,
    key: int = KEY,
    cands: tuple[tuple[int, int, int], ...] = (),
    turn: int | None = None,
    first_turn: bool = False,
    protected_last: bool = False,
    bench: tuple[tuple[int, bool], ...] = (),
    action: int | None = None,
    allowed: tuple[int, ...] | None = None,
    target: int | None = None,
    mega: int | None = None,
    mega_possible: bool = False,
    elo: int | None = None,
) -> None:
    """One slot of example ``i``: candidates (move id, class bits, target bits),
    legal bench pointers (roster row, shown), and its label."""
    if turn is not None:
        batch["turn"][i] = turn
    row = slot
    batch["act_mon"][i, slot] = row
    for column in (F.ID_SPECIES, F.ID_FORME, F.ID_KEY):
        batch["mon_id"][i, row, column] = key
    batch["mon_flag"][i, row, F.FLAG_PRESENT] = 1
    batch["mon_flag"][i, row, F.FLAG_FIRST_TURN] = int(first_turn)
    batch["mon_flag"][i, row, F.FLAG_PROTECTED_LAST] = int(protected_last)
    batch["mon_flag"][i, row, F.FLAG_MEGA_POSSIBLE] = int(mega_possible)
    for column, (move, bits, targets) in enumerate(cands):
        batch["cand_move"][i, slot, column] = move
        batch["cand_flag"][i, slot, column] = F.CAND_VALID | bits
        batch["cand_tmask"][i, slot, column] = targets
        batch["action_mask"][i, slot, column] = 1
    batch["action_mask"][i, slot, OTHER] = 1
    batch["cand_tmask"][i, slot, OTHER] = (1 << F.N_TARGET) - 1
    for roster, shown in bench:
        batch["switch_mask"][i, slot, roster] = 1
        batch["action_mask"][i, slot, F.switch_index(roster, C)] = 1
        batch["mon_flag"][i, roster, F.FLAG_PRESENT] = 1
        batch["mon_flag"][i, roster, F.FLAG_REVEALED] = int(shown)
    if action is not None:
        batch["y_action"][i, slot] = action
        batch["y_set"][i, slot, action] = 1
        batch["y_kind"][i, slot] = F.Y_KIND_SWITCH if action > OTHER else F.Y_KIND_MOVE
        guarded = action < len(cands) and bool(cands[action][1] & F.CAND_PROTECT)
        batch["y_flag"][i, slot, [F.Y_SWITCH_KNOWN, F.Y_PROTECT_KNOWN]] = 1
        batch["y_flag"][i, slot, F.Y_SWITCHED] = int(action > OTHER)
        batch["y_flag"][i, slot, F.Y_PROTECTED] = int(guarded)
    elif allowed is not None:
        batch["y_set"][i, slot, list(allowed)] = 1
        batch["y_kind"][i, slot] = F.Y_KIND_NONE
        batch["y_flag"][i, slot, [F.Y_SWITCH_KNOWN, F.Y_PROTECT_KNOWN]] = 1
    if target is not None:
        batch["y_target"][i, slot] = target
    if mega is not None:
        batch["y_mega"][i, slot] = mega
    if elo is not None:
        batch["elo"][i, 0] = elo
        batch["elo_known"][i, 0] = 1


def rows(specs: list[dict[str, Any]]) -> F.Batch:
    batch = blank(len(specs))
    for i, spec in enumerate(specs):
        put(batch, i, **spec)
    return batch


def features_only(batch: F.Batch) -> F.Batch:
    return {k: v for k, v in batch.items() if not k.startswith(("y_", "m_"))}


TWO = ((MOVE_A, ATTACK, FOES_ALLY), (MOVE_B, ATTACK, FOES_ALLY))
ONE_BENCH = ((2, True),)


def random_batch(seed: int, n: int = 160) -> F.Batch:
    """A messy but self-consistent batch: both slots, every label kind."""
    rng = np.random.default_rng(seed)
    pool = [
        (MOVE_A, ATTACK, FOES_ALLY),
        (MOVE_B, ATTACK | F.CAND_SPREAD, SELF_ONLY),
        (MOVE_G, GUARD, SELF_ONLY),
        (MOVE_F, FIRST, FOES),
        (15, STATUS, FOES_ALLY),
        (16, ATTACK, FOES_ALLY | SELF_ONLY),
        (17, 0, SELF_ONLY),
    ]
    batch = blank(n)
    for i in range(n):
        turn = int(rng.integers(1, 6))
        for slot in range(2):
            if slot == 1 and rng.random() < 0.3:
                continue
            picks = rng.choice(
                len(pool), size=int(rng.integers(1, C + 1)), replace=False
            )
            sheet = rng.random() < 0.3
            cands = []
            for pick in picks:
                move, bits, targets = pool[int(pick)]
                extra = F.CAND_SHEET if sheet else F.CAND_REPERTOIRE
                if rng.random() < 0.4:
                    extra = (extra & F.CAND_SHEET) | F.CAND_REVEALED
                cands.append((move, bits | extra, targets))
            bench = tuple(
                (int(r), bool(rng.random() < 0.5))
                for r in (2, 3, 4)
                if rng.random() < 0.6
            )
            legal = list(range(len(cands))) + [OTHER]
            legal += [F.switch_index(r, C) for r, _ in bench]
            draw = rng.random()
            action = allowed = target = None
            if draw < 0.7:
                action = int(rng.choice(legal))
                if action < len(cands):
                    options = [
                        t for t in range(F.N_TARGET) if cands[action][2] >> t & 1
                    ]
                    if rng.random() < 0.8:
                        target = int(rng.choice(options))
            elif draw < 0.9:
                allowed = tuple(c for c in range(len(cands))) + (OTHER,)
            mega_possible = rng.random() < 0.3
            put(
                batch,
                i,
                slot,
                key=int(rng.integers(5, 9)),
                cands=tuple(cands),
                turn=turn,
                first_turn=turn == 1 or rng.random() < 0.3,
                protected_last=turn > 1 and rng.random() < 0.2,
                bench=bench,
                action=action,
                allowed=allowed,
                target=target,
                mega=int(rng.random() < 0.5) if mega_possible else None,
                mega_possible=mega_possible,
                elo=int(rng.integers(1000, 1700)) if rng.random() < 0.8 else None,
            )
        batch["m_weight"][i] = rng.uniform(0.2, 1.0)
    return batch


# --- known counts -------------------------------------------------------------


def test_share_table_gives_the_counted_frequencies():
    specs = (
        [{"cands": TWO, "bench": ONE_BENCH, "action": 0}] * 6
        + [{"cands": TWO, "bench": ONE_BENCH, "action": 1}] * 2
        + [{"cands": TWO, "bench": ONE_BENCH, "action": F.switch_index(2, C)}] * 2
    )
    batch = rows(specs)
    table = T.SpeciesTable.fit(batch, config=RAW)
    pred = table.predict(features_only(batch))
    action = pred["action"][0, 0]
    assert action[0] == pytest.approx(0.6, abs=1e-6)  # 6 of 8 moves, 8 of 10 slots
    assert action[1] == pytest.approx(0.2, abs=1e-6)
    assert action[F.switch_index(2, C)] == pytest.approx(0.2, abs=1e-6)
    assert action[OTHER] < 1e-6 and action[OTHER] > 0  # legal: floored, not zero
    assert action.sum() == pytest.approx(1.0)
    assert not pred["action"][:, 1].any()  # the empty slot
    assert table.fit_info["slots_visible"] == 10
    assert table.fit_info["set_keys"] == 1 and table.fit_info["moves"] == 2


def test_example_weights_are_the_counts():
    batch = rows([{"cands": TWO, "action": 0}, {"cands": TWO, "action": 1}])
    batch["m_weight"][:] = (3.0, 1.0)
    pred = T.SpeciesTable.fit(batch, config=RAW).predict(batch)
    assert pred["action"][0, 0, 0] == pytest.approx(0.75, abs=1e-6)
    del batch["m_weight"]  # without weights every example counts once
    pred = T.SpeciesTable.fit(batch, config=RAW).predict(batch)
    assert pred["action"][0, 0, 0] == pytest.approx(0.5, abs=1e-6)


def test_rate_table_counts_clicks_over_exposure_per_state():
    shown_a = (
        (MOVE_A, ATTACK | F.CAND_REVEALED, FOES_ALLY),
        (MOVE_B, ATTACK | F.CAND_REPERTOIRE, FOES_ALLY),
    )
    hidden = (
        (MOVE_A, ATTACK | F.CAND_REPERTOIRE, FOES_ALLY),
        (MOVE_B, ATTACK | F.CAND_REPERTOIRE, FOES_ALLY),
    )
    specs = [{"cands": shown_a, "action": a} for a in (0, 0, 0, 1)]
    specs += [{"cands": hidden, "action": a} for a in (0, 1, 1, 1)]
    batch = rows(specs)
    table = T.SpeciesTable.fit(batch, config=replace(RAW, usage=T.USAGE_RATE))
    # A turn spent switching is no exposure: the same counts with two more rows.
    away = {"cands": hidden, "bench": ONE_BENCH, "action": F.switch_index(2, C)}
    more = T.SpeciesTable.fit(
        rows(specs + [away, away]), config=replace(RAW, usage=T.USAGE_RATE)
    )
    assert np.array_equal(more.move_exposure, table.move_exposure)
    assert np.array_equal(more.move_counts, table.move_counts)
    # A known: 3 clicks in 4 turns. A not known: 1 in 4. B not known: 4 in 8.
    assert table.move_counts[T.KNOWN, 0, 0].tolist() == [3.0, 0.0, 0.0]
    assert table.move_exposure[T.KNOWN, 0, 0].tolist() == [4.0, 0.0, 0.0]
    assert table.move_counts[T.NOT_KNOWN, 0, 0].tolist() == [1.0, 4.0, 0.0]
    assert table.move_exposure[T.NOT_KNOWN, 0, 0].tolist() == [4.0, 8.0, 8.0]
    pred = table.predict(features_only(batch))["action"]
    assert pred[0, 0, :2] == pytest.approx([0.75 / 1.25, 0.5 / 1.25], abs=1e-6)
    assert pred[4, 0, :2] == pytest.approx([0.25 / 0.75, 0.5 / 0.75], abs=1e-6)


def test_a_shown_rare_move_is_likely_under_rates_and_not_under_shares():
    """The reason for the rates: rare in the species, common once it is shown."""
    common = ((MOVE_A, ATTACK | F.CAND_REPERTOIRE, FOES), (MOVE_B, ATTACK, FOES))
    carried = ((MOVE_B, ATTACK | F.CAND_REVEALED, FOES), (MOVE_A, ATTACK, FOES))
    specs = [{"cands": common, "action": 0}] * 40
    specs += [{"cands": carried, "action": 0}] * 3 + [{"cands": carried, "action": 1}]
    batch = rows(specs)
    query = features_only(rows([{"cands": carried}]))
    share = T.SpeciesTable.fit(batch, config=RAW).predict(query)["action"][0, 0]
    table = T.SpeciesTable.fit(batch, config=replace(RAW, usage=T.USAGE_RATE))
    rate = table.predict(query)["action"][0, 0]
    assert share[0] == pytest.approx(3 / 44, abs=1e-6)  # its share of all clicks
    # 3 clicks in its 4 known turns, against A's 41 in 44 turns as a candidate.
    assert rate[0] == pytest.approx(0.75 / (0.75 + 41 / 44), abs=1e-6)
    assert rate[0] > 6 * share[0]


# --- shrinkage ----------------------------------------------------------------


def test_shrink_limits():
    counts = np.array([[3.0, 1.0], [0.0, 0.0]])
    total = counts.sum(-1, keepdims=True)
    prior = np.array([0.5, 0.5])
    raw = T._shrink(counts, total, prior, 0.0)
    assert raw[0].tolist() == [0.75, 0.25]
    assert raw[1].tolist() == [0.5, 0.5]  # an empty row falls back to its prior
    assert T._shrink(counts, total, prior, INF).tolist() == [[0.5, 0.5]] * 2
    assert T._shrink(counts, total, prior, 4.0)[0].tolist() == [0.625, 0.375]
    assert T._ratio(np.array([4.0]), np.array([2.0]), 0.0).tolist() == [2.0]
    assert T._ratio(np.array([4.0]), np.array([2.0]), INF).tolist() == [1.0]
    assert T._ratio(np.array([4.0]), np.array([0.0]), 0.0).tolist() == [1.0]


def flag_rows() -> F.Batch:
    """Key 7 clicks A on turn 1 and B later, four times each."""
    specs = [{"cands": TWO, "action": 0, "turn": 1, "first_turn": True}] * 4
    specs += [{"cands": TWO, "action": 1, "turn": 3}] * 4
    return rows(specs)


def test_flags_table_between_its_cell_and_the_set_key_row():
    batch = flag_rows()
    query = features_only(batch)
    cell = T.FlagsTable.fit(batch, config=RAW).predict(query)["action"]
    assert cell[0, 0, 0] == pytest.approx(1.0, abs=1e-6)  # strength 0: the cell
    assert cell[4, 0, 1] == pytest.approx(1.0, abs=1e-6)
    species = T.SpeciesTable.fit(batch, config=RAW).predict(query)
    assert species["action"][0, 0, :2] == pytest.approx([0.5, 0.5], abs=1e-6)
    pooled = T.FlagsTable.fit(batch, config=replace(RAW, cell_strength=INF))
    for name, array in pooled.predict(query).items():
        if name != "mega":  # the Mega head keeps its own cells
            assert np.array_equal(array, species[name]), name
    half = T.FlagsTable.fit(batch, config=replace(RAW, cell_strength=4.0))
    assert half.predict(query)["action"][0, 0, :2] == pytest.approx(
        [(4 + 4 * 0.5) / 8, (0 + 4 * 0.5) / 8], abs=1e-6
    )
    # The same limits for usage rates.
    rate = replace(RAW, usage=T.USAGE_RATE)
    assert T.FlagsTable.fit(batch, config=rate).predict(query)["action"][
        0, 0, 0
    ] == pytest.approx(1.0, abs=1e-6)
    pooled = T.FlagsTable.fit(batch, config=replace(rate, cell_strength=INF))
    species = T.SpeciesTable.fit(batch, config=rate).predict(query)
    assert np.array_equal(pooled.predict(query)["action"], species["action"])


def test_flags_cells_use_all_three_flags():
    pair = ((MOVE_A, ATTACK, FOES), (MOVE_G, GUARD, SELF_ONLY))
    specs = [{"cands": pair, "action": 1}] * 3 + [{"cands": pair, "action": 0}]
    specs += [{"cands": pair, "action": 0, "protected_last": True}] * 4
    specs += [{"cands": pair, "action": 0, "first_turn": True}] * 2
    specs += [{"cands": pair, "action": 1, "first_turn": True}] * 2
    specs += [{"cands": pair, "action": 1, "turn": 1, "first_turn": True}] * 4
    batch = rows(specs)
    guard = T.FlagsTable.fit(batch, config=RAW).predict(features_only(batch))["action"]
    assert guard[0, 0, 1] == pytest.approx(0.75, abs=1e-6)  # no flag set
    assert guard[4, 0, 1] < 1e-6  # protected last turn
    assert guard[8, 0, 1] == pytest.approx(0.5, abs=1e-6)  # first turn on the field
    assert guard[12, 0, 1] == pytest.approx(1.0, abs=1e-6)  # turn 1
    pooled = T.SpeciesTable.fit(batch, config=RAW).predict(features_only(batch))
    assert pooled["action"][4, 0, 1] == pytest.approx(9 / 16, abs=1e-6)


def test_sheet_moves_are_known_and_a_complete_moveset_has_no_other():
    sheet = (
        (MOVE_A, ATTACK | F.CAND_SHEET, FOES),
        (MOVE_B, ATTACK | F.CAND_SHEET | F.CAND_REVEALED, FOES),
    )
    closed = ((MOVE_A, ATTACK | F.CAND_REPERTOIRE, FOES),)
    specs = [{"cands": sheet, "action": a} for a in (0, 0, 0, 1)]
    specs += [{"cands": closed, "action": a} for a in (0, 0, OTHER, OTHER)]
    batch = rows(specs)
    config = replace(RAW, usage=T.USAGE_RATE)
    table = T.SpeciesTable.fit(batch, config=config)
    # On a sheet both moves are known; OTHER is offered on the open moveset only.
    assert table.move_counts[T.KNOWN, 0, 0].tolist() == [3.0, 1.0, 0.0]
    assert table.move_exposure[T.KNOWN, 0, 0].tolist() == [4.0, 4.0, 0.0]
    assert table.move_counts[T.NOT_KNOWN, 0, 0].tolist() == [2.0, 0.0, 2.0]
    assert table.move_exposure[T.NOT_KNOWN, 0, 0].tolist() == [4.0, 0.0, 4.0]
    action = table.predict(features_only(batch))["action"]
    assert action[0, 0, :2] == pytest.approx([0.75, 0.25], abs=1e-6)
    assert 0 < action[0, 0, OTHER] < 1e-6  # complete: only the floor
    assert action[4, 0, [0, OTHER]] == pytest.approx([0.5, 0.5], abs=1e-6)
    boosted = table.with_config(other_boost=3.0).predict(features_only(batch))["action"]
    assert boosted[4, 0, [0, OTHER]] == pytest.approx([0.25, 0.75], abs=1e-6)
    assert boosted[0, 0, OTHER] < 1e-6  # a boost of nothing is nothing
    share = T.SpeciesTable.fit(batch, config=RAW).with_config(other_boost=3.0)
    action = share.predict(features_only(batch))["action"]
    # Shares: A 5 of 8, B 1 of 8, OTHER 2 of 8; the closed slot offers A only,
    # so OTHER stands for the 3 of 8 outside it, times the boost.
    assert action[4, 0, [0, OTHER]] == pytest.approx(
        [5 / (5 + 9), 9 / (5 + 9)], abs=1e-6
    )
    assert action[0, 0, OTHER] < 1e-6


def test_reveal_and_class_offsets_restore_the_counted_shares():
    """What a share cannot hold: the click depends on what has been shown."""
    one_shown = (
        (MOVE_A, ATTACK | F.CAND_REVEALED, FOES),
        (MOVE_B, ATTACK | F.CAND_REPERTOIRE, FOES),
    )
    none_shown = (
        (MOVE_A, ATTACK | F.CAND_REPERTOIRE, FOES),
        (MOVE_B, ATTACK | F.CAND_REPERTOIRE, FOES),
    )
    specs = [{"cands": one_shown, "action": 0}] * 8
    specs += [{"cands": none_shown, "action": a} for a in (0, 1) * 4]
    batch = rows(specs)
    query = features_only(batch)
    plain = T.SpeciesTable.fit(batch, config=RAW).predict(query)["action"]
    assert plain[0, 0, 0] == pytest.approx(0.75, abs=1e-6)  # 12 of 16, shown or not
    config = replace(RAW, iterations=8, reveal_offsets=True, offset_strength=0.0)
    table = T.SpeciesTable.fit(batch, config=config)
    assert table.predict(query)["action"][0, 0, 0] > 0.99  # always A once shown
    assert (
        table.reveal_ratio[1, T.STATUS_SHOWN]
        > 3 * table.reveal_ratio[1, T.STATUS_NOT_SHOWN]
    )

    # Class offsets: the Protect share differs between open and closed sheets.
    def pair(extra: int) -> tuple[tuple[int, int, int], ...]:
        return ((MOVE_A, ATTACK | extra, FOES), (MOVE_G, GUARD | extra, SELF_ONLY))

    specs = [{"cands": pair(F.CAND_REPERTOIRE), "action": a} for a in (0, 0, 0, 1) * 2]
    specs += [{"cands": pair(F.CAND_SHEET), "action": a} for a in (0, 1, 1, 1) * 2]
    batch = rows(specs)
    query = features_only(batch)
    plain = T.SpeciesTable.fit(batch, config=RAW).predict(query)["action"]
    assert plain[0, 0, 1] == pytest.approx(0.5, abs=1e-6)
    config = replace(RAW, iterations=8, class_offsets=True, offset_strength=0.0)
    table = T.SpeciesTable.fit(batch, config=config)
    guard = table.predict(query)["action"][:, 0, 1]
    assert guard[0] == pytest.approx(0.25, abs=0.01)  # closed sheet
    assert guard[8] == pytest.approx(0.75, abs=0.01)  # open sheet
    assert table.class_ratio.shape == (2, T.N_MOVE_CLASS)
    # A strong pseudo-count keeps the ratios at 1: nothing moves.
    still = T.SpeciesTable.fit(batch, config=replace(config, offset_strength=1e9))
    assert still.predict(query)["action"][0, 0, 1] == pytest.approx(0.5, abs=1e-4)


def test_with_config_changes_strengths_without_refitting():
    batch = flag_rows()
    query = features_only(batch)
    table = T.FlagsTable.fit(batch, config=RAW)
    pooled = table.with_config(cell_strength=INF)
    assert pooled.predict(query)["action"][0, 0, :2] == pytest.approx(
        [0.5, 0.5], abs=1e-6
    )
    assert table.predict(query)["action"][0, 0, 0] == pytest.approx(1.0, abs=1e-6)
    assert pooled.move_counts is table.move_counts  # the counts are shared


def test_set_key_row_between_its_counts_and_the_global_row():
    specs = [{"cands": TWO, "action": 0, "key": 7}] * 3
    specs += [{"cands": TWO, "action": 1, "key": 8}]
    batch = rows(specs)
    query = features_only(batch)
    own = T.SpeciesTable.fit(batch, config=RAW).predict(query)["action"]
    assert own[0, 0, 0] == pytest.approx(1.0, abs=1e-6)
    assert own[3, 0, 1] == pytest.approx(1.0, abs=1e-6)
    pooled = T.SpeciesTable.fit(batch, config=replace(RAW, key_strength=INF))
    assert pooled.predict(query)["action"][3, 0, :2] == pytest.approx(
        [0.75, 0.25], abs=1e-6
    )


def test_unseen_set_key_reads_the_global_row():
    specs = [{"cands": TWO, "bench": ONE_BENCH, "action": 0, "key": 7}] * 3
    specs += [{"cands": TWO, "bench": ONE_BENCH, "action": 1, "key": 8}]
    specs += [{"cands": TWO, "bench": ONE_BENCH, "action": F.switch_index(2, C)}] * 4
    batch = rows(specs)
    query = features_only(
        rows(
            [
                {"cands": TWO, "bench": ONE_BENCH, "key": 999},
                {"cands": TWO, "key": 60000},
            ]
        )
    )
    for usage in (T.USAGE_SHARE, T.USAGE_RATE):
        for cls in (T.SpeciesTable, T.FlagsTable, T.EloTable):
            table = cls.fit(batch, config=replace(RAW, usage=usage))
            pred = table.predict(query)
            assert not table.counters
            action = pred["action"][0, 0]
            # Global: A 3 of 4 moves, B 1 of 4; switches 4 of 8 slots.
            assert action[:2] == pytest.approx([0.375, 0.125], abs=1e-6), (usage, cls)
            assert action[F.switch_index(2, C)] == pytest.approx(0.5, abs=1e-6)
            assert pred["action"][1, 0, :2] == pytest.approx([0.75, 0.25], abs=1e-6)


def test_unseen_candidate_move_gets_a_small_positive_mass():
    batch = rows([{"cands": TWO, "action": 0}] * 5)
    novel = ((MOVE_A, ATTACK, FOES_ALLY), (901, ATTACK, FOES_ALLY))
    query = features_only(rows([{"cands": novel}]))
    config = replace(RAW, unseen_count=0.5, key_strength=1.0)
    action = T.SpeciesTable.fit(batch, config=config).predict(query)["action"][0, 0]
    assert 0 < action[1] < 0.1 < action[0]
    assert action.sum() == pytest.approx(1.0)


def test_flags_ratio_teaches_a_rare_set_key_what_the_flags_mean():
    """A first-turn-only move: gone later for every key, also one seen once."""
    opener = ((MOVE_F, FIRST, FOES), (MOVE_A, ATTACK, FOES))
    specs = [{"cands": opener, "action": 0, "turn": 1, "first_turn": True}] * 30
    specs += [{"cands": opener, "action": 1, "turn": 3}] * 30
    specs += [{"cands": opener, "action": 0, "turn": 1, "first_turn": True, "key": 8}]
    batch = rows(specs)
    query = features_only(rows([{"cands": opener, "turn": 3, "key": 8}]))
    config = replace(RAW, cell_strength=5.0, key_strength=1.0, unseen_count=0.1)
    without = T.FlagsTable.fit(batch, config=config).predict(query)["action"][0, 0, 0]
    learned = T.FlagsTable.fit(
        batch, config=replace(config, flag_effect_strength=1.0)
    ).predict(query)["action"][0, 0, 0]
    assert without > 0.5  # all the key has shown is the opener
    assert learned < 0.2 < without


# --- censored slots -----------------------------------------------------------


def censored_rows() -> F.Batch:
    pair = ((MOVE_A, ATTACK, FOES), (MOVE_G, GUARD, SELF_ONLY))
    base = {"cands": pair, "bench": ONE_BENCH}
    specs = [{**base, "action": 0}] * 4 + [{**base, "action": 1}] * 2
    specs += [{**base, "action": F.switch_index(2, C)}] * 2
    # Fainted before acting: some move, not a switch, not a Protect.
    specs += [{**base, "allowed": (0, OTHER)}] * 2
    return rows(specs)


def test_dropping_censored_slots_over_predicts_switch_and_protect():
    batch = censored_rows()
    query = features_only(batch)
    config = replace(RAW, iterations=2, censored=T.CENSORED_DROP)
    dropped = T.SpeciesTable.fit(batch, config=config).predict(query)
    events = F.event_probs(dropped, batch)
    assert events[E.INTENT_SWITCH][0, 0] == pytest.approx(2 / 8, abs=1e-6)
    assert events[E.INTENT_PROTECT][0, 0] == pytest.approx(2 / 6 * 6 / 8, abs=1e-6)
    for mode in (T.CENSORED_FRACTIONAL, T.CENSORED_SLOT):
        table = T.SpeciesTable.fit(batch, config=replace(config, censored=mode))
        events = F.event_probs(table.predict(query), batch)
        # Ten slots, two switched; eight moved, two of them protected.
        assert events[E.INTENT_SWITCH][0, 0] == pytest.approx(0.2, abs=1e-6), mode
        assert events[E.INTENT_PROTECT][0, 0] == pytest.approx(0.2, abs=1e-6), mode
        assert table.fit_info["slots_censored_with_set"] == 2
        assert table.fit_info["slots_counted"] == 10
    assert T.SpeciesTable.fit(batch, config=config).fit_info["slots_counted"] == 8


def test_fractional_counts_keep_the_rows_visible_mix():
    """Row mode: censored weight scales the group, it does not pick the move."""
    trio = ((MOVE_A, ATTACK, FOES), (MOVE_B, ATTACK, FOES), (MOVE_G, GUARD, SELF_ONLY))
    specs = [{"cands": trio, "action": 0}] * 3 + [{"cands": trio, "action": 1}]
    specs += [{"cands": trio, "action": 2}] * 2
    # The censored slot only offers A (B is not among its candidates).
    specs += [{"cands": (trio[0], trio[2]), "allowed": (0, OTHER)}] * 4
    batch = rows(specs)
    config = replace(RAW, iterations=3)
    row = T.SpeciesTable.fit(batch, config=config)
    counts = row.move_counts.sum((0, 1))[0]
    assert counts == pytest.approx([6.0, 2.0, 2.0, 0.0], abs=1e-6)  # A:B stays 3:1
    slot = T.SpeciesTable.fit(batch, config=replace(config, censored=T.CENSORED_SLOT))
    counts = slot.move_counts.sum((0, 1))[0]
    # Per slot the weight goes to what the slot offers: A, and OTHER (which
    # stands for B, the move outside its candidates). B itself gets nothing.
    assert counts == pytest.approx([6.0, 1.0, 2.0, 1.0], abs=1e-4)


# --- switch destinations, targets, Mega ---------------------------------------


def test_switch_destination_follows_the_counted_preference():
    bench = ((2, True), (3, False), (4, False), (5, False))
    base = {"cands": TWO, "bench": bench}
    specs = [{**base, "action": 0}] * 4
    specs += [{**base, "action": F.switch_index(2, C)}] * 3
    specs += [{**base, "action": F.switch_index(4, C)}]
    batch = rows(specs)
    query = features_only(batch)
    table = T.SpeciesTable.fit(batch, config=RAW)
    action = table.predict(query)["action"][0, 0]
    assert action[F.switch_index(2, C)] == pytest.approx(0.5 * 0.75, abs=1e-6)
    for roster in (3, 4, 5):  # uniform among the not yet shown
        assert action[F.switch_index(roster, C)] == pytest.approx(0.5 * 0.25 / 3)
    assert table.fit_info["switch_dest_odds_shown"] == pytest.approx(9.0)
    even = table.with_config(switch_dest=T.DEST_UNIFORM).predict(query)["action"][0, 0]
    for roster in (2, 3, 4, 5):
        assert even[F.switch_index(roster, C)] == pytest.approx(0.125, abs=1e-6)
    # A pattern the fit never saw uses the pooled odds: 9 to 1 per pointer.
    other = features_only(rows([{"cands": TWO, "bench": ((2, True), (3, False))}]))
    action = table.predict(other)["action"][0, 0]
    assert action[F.switch_index(2, C)] == pytest.approx(0.5 * 0.9, abs=1e-6)


def test_no_legal_switch_means_no_switch_mass():
    batch = rows(
        [{"cands": TWO, "bench": ONE_BENCH, "action": F.switch_index(2, C)}] * 3
    )
    query = features_only(rows([{"cands": TWO}]))
    action = T.SpeciesTable.fit(batch, config=RAW).predict(query)["action"][0, 0]
    assert not action[OTHER + 1 :].any()
    assert action.sum() == pytest.approx(1.0)


def test_targets_back_off_from_key_to_move_to_class_to_uniform():
    aimed = ((MOVE_A, ATTACK, FOES_ALLY),)
    specs = [{"cands": aimed, "action": 0, "target": F.T_FOE_A}] * 3
    specs += [{"cands": aimed, "action": 0, "target": F.T_FOE_B}]
    batch = rows(specs)
    table = T.SpeciesTable.fit(batch, config=RAW)
    target = table.predict(features_only(batch))["target"][0, 0, 0]
    assert target[:2] == pytest.approx([0.75, 0.25], abs=1e-6)
    assert 0 < target[F.T_ALLY] < 1e-6  # legal and never seen: the floor
    assert not target[F.T_SELF] and not target[F.T_AUTO]  # illegal: zero

    query = rows(
        [
            {"cands": ((MOVE_A, ATTACK, (1 << F.T_FOE_B) | (1 << F.T_ALLY)),)},
            {"cands": aimed, "key": 999},  # unseen key: the move's row
            {"cands": ((901, ATTACK, FOES_ALLY),)},  # unseen move: its class row
            {"cands": ((902, STATUS, FOES_ALLY),)},  # unseen class: uniform
        ]
    )
    target = table.predict(features_only(query))["target"][:, 0, 0]
    assert target[0, F.T_FOE_B] == pytest.approx(1.0, abs=1e-6)  # the legal mask
    # A second set key keeps its own row: it always aims at the other foe.
    both = rows(
        specs + [{"cands": aimed, "action": 0, "target": F.T_FOE_B, "key": 8}] * 4
    )
    own = T.SpeciesTable.fit(both, config=RAW).predict(features_only(both))["target"]
    assert own[0, 0, 0, :2] == pytest.approx([0.75, 0.25], abs=1e-6)
    assert own[4, 0, 0, F.T_FOE_B] == pytest.approx(1.0, abs=1e-6)
    assert target[1, :2] == pytest.approx([0.75, 0.25], abs=1e-6)
    by_class = np.array([3.2, 1.2, 0.2]) / 4.6  # (count + 1 * 0.2) / 5, then masked
    assert target[2, :3] == pytest.approx(by_class, abs=1e-6)
    assert target[3, :3] == pytest.approx([1 / 3] * 3, abs=1e-6)
    assert target.sum(-1) == pytest.approx(1.0)


# --- which of two foes: the target context ------------------------------------


@pytest.fixture(scope="module")
def fz() -> F.Featurizer:
    return F.Featurizer.build(F.Repertoire())


def type_id(fz: F.Featurizer, name: str) -> int:
    return 1 + fz.vocab.types.index(name)


def move_of_type(fz: F.Featurizer, name: str, damaging: bool = True) -> int:
    """Vocabulary id of some move of that type, read off the numeric tables."""
    columns = F.move_columns(fz.vocab)
    typed = fz.tables.move_num[:, columns.index(f"type:{name}")] > 0
    hurts = fz.tables.move_num[:, columns.index("damaging")] > 0
    return int(np.flatnonzero(typed & (hurts == damaging))[0])


def face(
    batch: F.Batch,
    i: int,
    first: tuple[int, ...],
    second: tuple[int, ...],
    hp: tuple[float, float] = (1.0, 1.0),
) -> None:
    """Put two opposing Pokemon with these type ids and HP in front of example ``i``."""
    for position, types in enumerate((first, second)):
        row = F.N_ROSTER + position
        batch["foe_mon"][i, position] = row
        batch["mon_flag"][i, row, F.FLAG_PRESENT] = 1
        batch["mon_type"][i, row, : len(types)] = types
        batch["mon_hp"][i, row] = hp[position]


def facing(specs: list[tuple[dict[str, Any], tuple[Any, ...]]]) -> F.Batch:
    batch = rows([spec for spec, _ in specs])
    for i, (_, foes) in enumerate(specs):
        face(batch, i, *foes)
    return batch


def test_which_of_two_foes_follows_type_effectiveness(fz: F.Featurizer):
    """Set key and move cannot tell the two foes apart; the context can."""
    fire, water, grass = (type_id(fz, name) for name in ("fire", "water", "grass"))
    burn = move_of_type(fz, "fire")
    aimed = ((burn, ATTACK, FOES_ALLY),)
    hit_a = {"cands": aimed, "action": 0, "target": F.T_FOE_A}
    hit_b = {"cands": aimed, "action": 0, "target": F.T_FOE_B}
    weak_a, weak_b = ((grass,), (water,)), ((water,), (grass,))
    # Nine times in ten the fire move goes into the grass type, in either slot.
    specs = [(hit_a, weak_a)] * 9 + [(hit_b, weak_a)] + [(hit_b, weak_b)] * 9
    specs += [(hit_a, weak_b)]
    batch = facing(specs)
    table = T.SpeciesTable.fit(batch, config=RAW, featurizer=fz)
    assert table.fit_info["target_context_labels"] == 20
    assert table.type_names == list(fz.vocab.types)
    assert table.move_type.tolist() == [fire]
    assert table.type_chart[fire, grass] == 2.0 and table.type_chart[fire, water] == 0.5
    assert table.type_chart[0].tolist() == [1.0] * len(table.type_chart)
    target = table.predict(features_only(batch))["target"]
    assert target[0, 0, 0, :2] == pytest.approx([0.9, 0.1], abs=1e-6)
    assert target[19, 0, 0, :2] == pytest.approx([0.1, 0.9], abs=1e-6)
    # Without the context the two slots are a coin flip: 10 labels each.
    off = table.with_config(target_context=False).predict(features_only(batch))
    assert off["target"][0, 0, 0, :2] == pytest.approx([0.5, 0.5], abs=1e-6)
    assert off["target"][19, 0, 0, :2] == pytest.approx([0.5, 0.5], abs=1e-6)
    assert np.array_equal(off["action"], table.predict(features_only(batch))["action"])
    # A refit without it stores neutral ratios and says the same as switching off.
    plain = T.SpeciesTable.fit(
        batch, config=replace(RAW, target_context=False), featurizer=fz
    )
    assert plain.target_ratio.tolist() == [1.0] * T.N_TARGET_CELL
    assert plain.fit_info["target_context_labels"] == 0
    assert_same_prediction(plain, table.with_config(target_context=False), batch)
    # New positions: two foes of one kind tell nothing; a neutral pair neither.
    query = facing(
        [
            ({"cands": aimed}, ((grass,), (grass,))),
            ({"cands": aimed}, ((fire,), (fire,))),
            ({"cands": aimed}, ((grass, water), (water,))),  # 2 x 0.5 = neutral here
        ]
    )
    asked = table.predict(features_only(query))["target"][:, 0, 0]
    assert asked[0, :2] == pytest.approx([0.5, 0.5], abs=1e-6)
    assert asked[1, :2] == pytest.approx([0.5, 0.5], abs=1e-6)
    # neutral against a, resisted by b: a cell with no label, ratio 1 on both
    assert asked[2, :2] == pytest.approx([0.5, 0.5], abs=1e-6)
    assert not table.counters


def test_the_context_only_moves_mass_between_two_legal_foes(fz: F.Featurizer):
    fire, water, grass = (type_id(fz, name) for name in ("fire", "water", "grass"))
    burn = move_of_type(fz, "fire")
    wide = ((burn, ATTACK, FOES_ALLY),)
    lone = ((burn, ATTACK, (1 << F.T_FOE_A) | (1 << F.T_ALLY)),)
    hit_a = {"cands": wide, "action": 0, "target": F.T_FOE_A}
    hit_b = {"cands": wide, "action": 0, "target": F.T_FOE_B}
    ally = {"cands": wide, "action": 0, "target": F.T_ALLY}
    weak_a, weak_b = ((grass,), (water,)), ((water,), (grass,))
    specs = [(hit_a, weak_a)] * 6 + [(hit_b, weak_b)] * 6 + [(ally, weak_a)] * 4
    batch = facing(specs)
    table = T.SpeciesTable.fit(batch, config=RAW, featurizer=fz)
    on = table.predict(features_only(batch))["target"]
    off = table.with_config(target_context=False).predict(features_only(batch))[
        "target"
    ]
    # Four of sixteen labels went to the ally: that share is not the context's.
    assert off[0, 0, 0, :3] == pytest.approx([0.375, 0.375, 0.25], abs=1e-6)
    assert on[0, 0, 0, F.T_ALLY] == pytest.approx(0.25, abs=1e-6)
    assert on[0, 0, 0, :2].sum() == pytest.approx(0.75, abs=1e-6)
    assert on[0, 0, 0, F.T_FOE_A] > 0.74 and on[6, 0, 0, F.T_FOE_B] > 0.74
    # OTHER's row (a move nobody knows) is left alone.
    assert np.array_equal(on[:, :, OTHER], off[:, :, OTHER])
    # One legal foe: nothing to split, whatever stands there.
    query = facing([({"cands": lone}, weak_b), ({"cands": lone}, weak_a)])
    asked = table.predict(features_only(query))["target"][:, 0, 0]
    still = table.with_config(target_context=False).predict(features_only(query))
    assert np.array_equal(asked, still["target"][:, 0, 0])


def test_a_lone_foe_teaches_the_context_nothing(fz: F.Featurizer):
    """With one foe on the field the target is not a choice. Such labels must
    not enter the context ratios: the lone foe would always look like "the
    one with more HP" that everybody aims at."""
    fire, grass = type_id(fz, "fire"), type_id(fz, "grass")
    aimed = ((MOVE_A, ATTACK, FOES),)
    alone = ((MOVE_A, ATTACK, 1 << F.T_FOE_A),)
    hit_a = {"cands": aimed, "action": 0, "target": F.T_FOE_A}
    hit_b = {"cands": aimed, "action": 0, "target": F.T_FOE_B}
    forced = {"cands": alone, "action": 0, "target": F.T_FOE_A}
    same = ((fire,), (grass,))
    low_a, low_b = (*same, (0.3, 1.0)), (*same, (1.0, 0.3))
    choices = [(hit_a, low_a)] * 8 + [(hit_b, low_a)] * 2
    choices += [(hit_b, low_b)] * 8 + [(hit_a, low_b)] * 2
    plain = T.SpeciesTable.fit(facing(choices), config=RAW)
    batch = facing(choices + [(forced, low_b)] * 30)
    batch["foe_mon"][20:, 1] = -1  # the forced rows: foe b is not there
    crowded = T.SpeciesTable.fit(batch, config=RAW)
    assert crowded.fit_info["target_labels"] == plain.fit_info["target_labels"] + 30
    assert crowded.fit_info["target_context_labels"] == 20
    assert np.allclose(crowded.target_ratio, plain.target_ratio)
    assert crowded.target_ratio[13 - 1] == pytest.approx(1.6)  # neutral, lower HP
    assert crowded.target_ratio[13 + 1] == pytest.approx(0.4)  # neutral, higher HP
    # Forty of fifty labels went to slot a, which the move's own row keeps
    # (0.8 / 0.2); the context says "the weaker of the two" on top of it.
    target = crowded.predict(features_only(batch))["target"]
    assert target[0, 0, 0, :2] == pytest.approx([1.28 / 1.36, 0.08 / 1.36], abs=1e-6)
    assert target[10, 0, 0, :2] == pytest.approx([0.5, 0.5], abs=1e-6)
    assert target[20, 0, 0, F.T_FOE_A] == pytest.approx(1.0, abs=1e-6)


def test_hp_relation_is_all_a_typeless_table_reads(fz: F.Featurizer):
    """Fitted without a featurizer there is no type chart: only who has less HP."""
    fire, grass = type_id(fz, "fire"), type_id(fz, "grass")
    aimed = ((MOVE_A, ATTACK, FOES),)
    hit_a = {"cands": aimed, "action": 0, "target": F.T_FOE_A}
    hit_b = {"cands": aimed, "action": 0, "target": F.T_FOE_B}
    same = ((fire,), (grass,))
    low_a, low_b = (*same, (0.3, 1.0)), (*same, (1.0, 0.3))
    specs = [(hit_a, low_a)] * 8 + [(hit_b, low_a)] * 2
    specs += [(hit_b, low_b)] * 8 + [(hit_a, low_b)] * 2
    batch = facing(specs)
    table = T.SpeciesTable.fit(batch, config=RAW)
    assert table.type_names == [] and table.move_type.tolist() == [0]
    target = table.predict(features_only(batch))["target"]
    assert target[0, 0, 0, :2] == pytest.approx([0.8, 0.2], abs=1e-6)
    assert target[10, 0, 0, :2] == pytest.approx([0.2, 0.8], abs=1e-6)
    # HP within a tenth of each other counts as the same.
    near = facing([({"cands": aimed}, (*same, (0.52, 0.6)))])
    assert table.predict(features_only(near))["target"][0, 0, 0, :2] == pytest.approx(
        [0.5, 0.5], abs=1e-6
    )
    # A move that does no damage has its own three cells.
    soft = ((MOVE_B, STATUS, FOES),)
    calm_a = {"cands": soft, "action": 0, "target": F.T_FOE_A}
    calm_b = {"cands": soft, "action": 0, "target": F.T_FOE_B}
    # Status moves go to the HEALTHIER foe here, attacks (above) to the weaker.
    calm = [(calm_b, low_a)] * 9 + [(calm_a, low_a)]
    calm += [(calm_a, low_b)] * 9 + [(calm_b, low_b)]
    mixed = facing(specs + calm)
    both = T.SpeciesTable.fit(mixed, config=RAW)
    target = both.predict(features_only(mixed))["target"]
    assert target[0, 0, 0, :2] == pytest.approx([0.8, 0.2], abs=1e-6)
    assert target[20, 0, 0, :2] == pytest.approx([0.1, 0.9], abs=1e-6)
    assert target[30, 0, 0, :2] == pytest.approx([0.9, 0.1], abs=1e-6)
    assert both.target_ratio[T._STATUS_CELL :] == pytest.approx([0.2, 1.0, 1.8])
    # Without the context each move is a coin flip between the two slots.
    off = both.with_config(target_context=False).predict(features_only(mixed))
    assert off["target"][[0, 20], 0, 0, :2] == pytest.approx(0.5, abs=1e-6)


def test_context_ratios_are_shrunk_and_survive_storage(fz: F.Featurizer):
    fire, water, grass = (type_id(fz, name) for name in ("fire", "water", "grass"))
    burn = move_of_type(fz, "fire")
    aimed = ((burn, ATTACK, FOES_ALLY),)
    hit_a = {"cands": aimed, "action": 0, "target": F.T_FOE_A}
    hit_b = {"cands": aimed, "action": 0, "target": F.T_FOE_B}
    weak_a, weak_b = ((grass,), (water,)), ((water,), (grass,))
    batch = facing([(hit_a, weak_a)] * 2 + [(hit_b, weak_b)] * 2)
    raw = T.FlagsTable.fit(batch, config=RAW, featurizer=fz)
    held = T.FlagsTable.fit(
        batch, config=replace(RAW, target_context_strength=50.0), featurizer=fz
    )
    assert (
        raw.predict(features_only(batch))["target"][0, 0, 0, 0]
        > held.predict(features_only(batch))["target"][0, 0, 0, 0]
    )
    assert abs(held.target_ratio - 1.0).max() < 0.05 < abs(raw.target_ratio - 1.0).max()
    # Storage: the ratios, the move types, the chart and the type names.
    payload = held.to_payload()
    assert payload["target_ratio"].shape == (T.N_TARGET_CELL,)
    assert payload["type_names"] == list(fz.vocab.types)
    again = T.from_payload(payload, fz)
    assert_same_prediction(held, again, features_only(batch))
    assert np.array_equal(again.target_ratio, held.target_ratio)
    assert np.array_equal(again.type_chart, held.type_chart)
    # A payload written before the context existed: every ratio reads 1.
    old = {
        key: value
        for key, value in raw.to_payload().items()
        if key not in ("target_ratio", "move_type", "type_chart", "type_names")
    }
    before = T.from_payload(old, fz)
    assert before.target_ratio.tolist() == [1.0] * T.N_TARGET_CELL
    assert_same_prediction(
        before, raw.with_config(target_context=False), features_only(batch)
    )
    with pytest.raises(ValueError):
        T.from_payload({**payload, "target_ratio": payload["target_ratio"][:-1]})
    with pytest.raises(ValueError):
        T.from_payload({**payload, "type_chart": np.ones((3, 4))})


def test_context_follows_type_names_into_a_reordered_vocabulary(fz: F.Featurizer):
    fire, water, grass = (type_id(fz, name) for name in ("fire", "water", "grass"))
    burn = move_of_type(fz, "fire")
    aimed = ((burn, ATTACK, FOES),)
    hit_a = {"cands": aimed, "action": 0, "target": F.T_FOE_A}
    hit_b = {"cands": aimed, "action": 0, "target": F.T_FOE_B}
    weak_a, weak_b = ((grass,), (water,)), ((water,), (grass,))
    batch = facing([(hit_a, weak_a)] * 5 + [(hit_b, weak_b)] * 5)
    table = T.SpeciesTable.fit(batch, config=RAW, featurizer=fz)
    names = list(fz.vocab.types)
    turned = SimpleNamespace(
        tables=fz.tables, vocab=SimpleNamespace(types=tuple(reversed(names)))
    )
    moved = T.from_payload(table.to_payload(), turned)
    assert moved.type_names == list(reversed(names))
    new_id = {1 + i: 1 + list(reversed(names)).index(n) for i, n in enumerate(names)}
    assert moved.move_type.tolist() == [new_id[fire]]
    assert moved.type_chart[new_id[fire], new_id[grass]] == 2.0
    query = features_only(batch)
    recoded = dict(query)
    recoded["mon_type"] = np.vectorize(lambda t: new_id.get(int(t), 0))(
        query["mon_type"]
    ).astype(query["mon_type"].dtype)
    first, second = table.predict(query), moved.predict(recoded)
    assert first["target"][0, 0, 0, 0] > 0.9
    assert np.allclose(first["target"], second["target"])
    # A vocabulary that lacks the types reads them as "other": neutral.
    bare = SimpleNamespace(tables=fz.tables, vocab=SimpleNamespace(types=("x", "y")))
    lost = T.from_payload(table.to_payload(), bare)
    assert lost.move_type.tolist() == [0] and (lost.type_chart == 1.0).all()


def test_a_batch_without_the_foes_types_is_predicted_without_the_context(
    fz: F.Featurizer,
):
    fire, water, grass = (type_id(fz, name) for name in ("fire", "water", "grass"))
    burn = move_of_type(fz, "fire")
    aimed = ((burn, ATTACK, FOES),)
    hit_a = {"cands": aimed, "action": 0, "target": F.T_FOE_A}
    hit_b = {"cands": aimed, "action": 0, "target": F.T_FOE_B}
    weak_a, weak_b = ((grass,), (water,)), ((water,), (grass,))
    batch = facing([(hit_a, weak_a)] * 5 + [(hit_b, weak_b)] * 5)
    table = T.SpeciesTable.fit(batch, config=RAW, featurizer=fz)
    query = features_only(batch)
    for missing in T._CONTEXT_FEATURES:
        thin = {name: array for name, array in query.items() if name != missing}
        table.counters.clear()
        made = table.predict(thin)
        assert not [name for name in table.counters if name.startswith("predict_error")]
        assert table.counters["target_context_unread"] == 1
        assert made["target"][0, 0, 0, :2] == pytest.approx([0.5, 0.5], abs=1e-6)
    # Fitting such a batch works too and leaves the ratios neutral.
    thin_fit = {k: v for k, v in batch.items() if k != "mon_type"}
    assert T.SpeciesTable.fit(thin_fit, config=RAW).target_ratio.tolist() == [1.0] * 30


def test_spread_share_of_an_aimed_move_comes_from_the_batch():
    wide = ((MOVE_A, ATTACK, FOES | (1 << F.T_AUTO)),)
    specs = [{"cands": wide, "action": 0, "target": F.T_FOE_A}] * 3
    specs += [{"cands": wide, "action": 0, "target": F.T_FOE_B}]
    batch = rows(specs)
    batch["cand_auto"][:, 0, 0] = 0.8
    table = T.SpeciesTable.fit(batch, config=RAW)
    target = table.predict(features_only(batch))["target"][0, 0, 0]
    assert target[F.T_AUTO] == pytest.approx(0.8, abs=1e-3)
    assert target[:2] == pytest.approx([0.15, 0.05], abs=1e-3)
    plain = table.with_config(use_cand_auto=False).predict(features_only(batch))
    assert plain["target"][0, 0, 0, F.T_AUTO] < 1e-6  # never counted as a spread


def test_mega_rate_is_counted_where_possible_and_scaled_by_moving():
    base = {"cands": TWO, "mega_possible": True}
    specs = [
        {**base, "action": 0, "turn": 1, "first_turn": True, "mega": m}
        for m in (1, 1, 1, 0)
    ]
    specs += [{**base, "action": 0, "turn": 3, "mega": 0}] * 4
    batch = rows(specs)
    for cls in (T.SpeciesTable, T.FlagsTable):
        mega = cls.fit(batch, config=RAW).predict(features_only(batch))["mega"]
        assert mega[0, 0] == pytest.approx(0.75, abs=1e-6), cls
        assert mega[4, 0] == pytest.approx(TINY)  # never on a later turn
        assert mega[0, 1] == pytest.approx(TINY)  # the empty slot
    # Half the slots switch: a Mega needs a move, so the rate is halved.
    bench = {**base, "bench": ONE_BENCH, "turn": 1, "first_turn": True}
    specs = [{**bench, "action": 0, "mega": 1}] * 2
    specs += [{**bench, "action": F.switch_index(2, C), "mega": 0}] * 2
    batch = rows(specs)
    mega = T.SpeciesTable.fit(batch, config=RAW).predict(features_only(batch))["mega"]
    assert mega[0, 0] == pytest.approx(0.5, abs=1e-6)
    impossible = rows([{"cands": TWO, "turn": 1, "first_turn": True}])
    table = T.SpeciesTable.fit(batch, config=RAW)
    assert table.predict(features_only(impossible))["mega"][0, 0] == pytest.approx(TINY)


# --- Elo ----------------------------------------------------------------------


def elo_rows() -> F.Batch:
    base = {"cands": TWO[:1], "bench": ONE_BENCH}
    switch = F.switch_index(2, C)
    specs = [{**base, "action": switch, "elo": 1000}] * 4
    specs += [{**base, "action": 0, "elo": 1000}] * 4
    specs += [{**base, "action": 0, "elo": 1500}] * 8
    return rows(specs)


def test_elo_offset_moves_the_mix_and_stands_down_without_a_rating():
    batch = elo_rows()
    query = rows(
        [
            {"cands": TWO[:1], "bench": ONE_BENCH, "elo": 1000},
            {"cands": TWO[:1], "bench": ONE_BENCH, "elo": 1500},
            {"cands": TWO[:1], "bench": ONE_BENCH},
        ]
    )
    config = replace(RAW, elo_edges=(1300,), elo_strength=0.0)
    flags = T.FlagsTable.fit(batch, config=config).predict(features_only(query))
    table = T.EloTable.fit(batch, config=config)
    assert table.fit_info["elo_slots"] == 16
    assert table.elo_observed[:, T.CLASS_SWITCH].tolist() == [4.0, 0.0]
    assert table.elo_expected[:, T.CLASS_SWITCH] == pytest.approx([2.0, 2.0], abs=1e-6)
    switch = F.event_probs(table.predict(features_only(query)), query)[E.INTENT_SWITCH]
    assert switch[0, 0] == pytest.approx(0.5, abs=1e-6)  # the low band's own rate
    assert switch[1, 0] < 1e-6  # the high band never switched
    assert switch[2, 0] == pytest.approx(0.25, abs=1e-6)  # unknown: the flags table
    off = table.with_config(elo_strength=INF).predict(features_only(query))
    for name, array in off.items():
        assert np.array_equal(array, flags[name]), name
    blank_elo = F.apply_elo_mode(features_only(query), F.ELO_BLANK)
    assert np.array_equal(table.predict(blank_elo)["action"], flags["action"])
    with pytest.raises(ValueError):
        table.with_config(elo_edges=(1200, 1400))


# --- contract: legality, labels, masks, storage -------------------------------


@pytest.mark.parametrize("cls", [T.SpeciesTable, T.FlagsTable, T.EloTable])
@pytest.mark.parametrize(
    "config",
    [
        T.TableConfig(),
        T.TableConfig(usage=T.USAGE_SHARE),
        replace(RAW, iterations=3, reveal_offsets=True, class_offsets=True),
        replace(RAW, usage=T.USAGE_RATE, elo_strength=0.0, floor=1e-4),
    ],
    ids=["default", "share", "raw-share", "raw-rate"],
)
def test_probabilities_are_finite_legal_and_floored(cls: Any, config: T.TableConfig):
    train = random_batch(1)
    table = cls.fit(train, config=config)
    batch = random_batch(2)
    batch["mon_id"][:5, :, F.ID_KEY] = 500  # some unseen set keys
    batch["cand_move"][5:10] += 300  # and unseen moves
    pred = table.predict(features_only(batch))
    assert not table.counters
    assert set(pred) == {"action", "target", "mega"}
    action, target, mega = pred["action"], pred["target"], pred["mega"]
    legal = batch["action_mask"].astype(bool)
    assert action.shape == legal.shape
    assert np.isfinite(action).all() and np.isfinite(target).all()
    assert np.isfinite(mega).all()
    assert not action[~legal].any()
    assert (action[legal] >= config.floor / 2).all()  # the floor, renormalised
    active = batch["act_mon"] >= 0
    assert action.sum(-1)[active] == pytest.approx(1.0)
    assert not action[~active].any()
    classes = F.expand_target_mask(batch["cand_tmask"])
    assert target.shape == classes.shape
    assert not target[~classes].any()
    assert (target[classes] > 0).all()
    assert target.sum(-1)[classes.any(-1)] == pytest.approx(1.0)
    assert ((mega > 0) & (mega < 1)).all()
    # The shared scoring code accepts it and no label costs the floor's worth.
    nll = F.slot_nll(pred, batch)
    assert nll["fine_scored"].any()
    assert nll["fine"][nll["fine_scored"]].max() < -2 * math.log(config.floor / 2)
    # Normalising changes nothing: the mass was already on the masks.
    norm = F.normalize_prediction(pred, batch)
    assert np.allclose(norm["action"], action) and np.allclose(norm["target"], target)


@pytest.mark.parametrize("cls", [T.SpeciesTable, T.FlagsTable, T.EloTable])
def test_predict_reads_no_label_and_no_meta_array(cls: Any):
    table = cls.fit(random_batch(3))
    batch = random_batch(4)
    full = table.predict(batch)
    bare = table.predict(features_only(batch))
    scrambled = dict(batch)
    rng = np.random.default_rng(0)
    for name in batch:
        if name.startswith(("y_", "m_")):
            scrambled[name] = rng.permutation(batch[name])
    again = table.predict(scrambled)
    for name in full:
        assert np.array_equal(full[name], bare[name]), name
        assert np.array_equal(full[name], again[name]), name
    assert not table.counters


@pytest.mark.parametrize("cls", [T.SpeciesTable, T.FlagsTable, T.EloTable])
def test_fit_reads_only_the_masked_rows(cls: Any):
    batch = random_batch(5, n=120)
    mask = np.zeros(120, dtype=bool)
    mask[::3] = True
    clean = cls.fit(F.take(batch, mask))
    poisoned = {name: array.copy() for name, array in batch.items()}
    rng = np.random.default_rng(1)
    for name, array in poisoned.items():
        if name == "m_weight":
            array[~mask] = np.nan  # one read of these rows would spoil a count
        elif array.dtype.kind == "f":
            array[~mask] = 7.0
        else:
            array[~mask] = rng.integers(0, 3, size=array[~mask].shape).astype(
                array.dtype
            )
    masked = cls.fit(poisoned, mask=mask)
    a, b = clean.to_payload(), masked.to_payload()
    assert a["fit_info"] == b["fit_info"]
    for name in ("key_ids", "move_ids", "reveal_ratio", "class_ratio", "target_ratio"):
        assert np.array_equal(a[name], b[name]), name
    for name in ("move_counts", "move_exposure", "switch_counts", "target_counts"):
        assert np.array_equal(a[name]["index"], b[name]["index"]), name
        assert np.array_equal(a[name]["value"], b[name]["value"]), name
    with pytest.raises(ValueError):
        cls.fit(batch, mask=np.ones(7, dtype=bool))
    with pytest.raises(ValueError):
        cls.fit(batch, mask=np.ones(120, dtype=np.int64))


def assert_same_prediction(first: Any, second: Any, batch: F.Batch) -> None:
    a, b = first.predict(batch), second.predict(batch)
    assert set(a) == set(b) == {"action", "target", "mega"}
    for name in a:
        assert np.array_equal(a[name], b[name]), name


@pytest.mark.parametrize("cls", [T.SpeciesTable, T.FlagsTable, T.EloTable])
def test_payload_round_trip_and_determinism(cls: Any):
    train, batch = random_batch(6), features_only(random_batch(7))
    config = replace(T.TableConfig(), flag_effect_strength=INF, elo_edges=(1200, 1400))
    table = cls.fit(train, config=config)
    assert_same_prediction(table, cls.fit(train, config=config), batch)  # deterministic
    payload = table.to_payload()
    assert payload["kind"] == "table" and payload["table"] == cls.table
    again = T.from_payload(payload)
    assert type(again) is cls and again.name == table.name == cls.table
    assert again.kind == "table" and again.config == config
    assert again.fit_info == table.fit_info
    assert_same_prediction(table, again, batch)
    assert_same_prediction(table, cls.from_payload(again.to_payload()), batch)
    assert isinstance(again, F.Predictor)


def test_payload_is_plain_data():
    payload = T.EloTable.fit(random_batch(8)).to_payload()

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
                value, (bool, int, float, str, np.ndarray)
            ), (where, type(value))

    check(payload, "payload")


def test_bad_payloads_are_refused():
    payload = T.FlagsTable.fit(random_batch(9)).to_payload()
    with pytest.raises(ValueError):
        T.from_payload({})
    with pytest.raises(ValueError):
        T.from_payload({**payload, "table": "something_else"})
    with pytest.raises(ValueError):
        T.SpeciesTable.from_payload(payload)  # a flags payload
    with pytest.raises(ValueError):
        T.from_payload({**payload, "payload_version": 99})
    broken = {k: v for k, v in payload.items() if k != "switch_counts"}
    with pytest.raises(ValueError):
        T.from_payload(broken)
    with pytest.raises(ValueError):
        T.from_payload({**payload, "move_bits": payload["move_bits"][:-1]})
    with pytest.raises(ValueError):
        T.FlagsTable.fit(random_batch(9), config=replace(RAW, censored="sometimes"))
    with pytest.raises(ValueError):
        T.FlagsTable.fit(random_batch(9), config=replace(RAW, usage="guess"))
    with pytest.raises(ValueError):
        T.FlagsTable.fit({"turn": np.zeros(3)})


def fake_featurizer(species: list[str], moves: list[str]) -> Any:
    return SimpleNamespace(tables=SimpleNamespace(species_ids=species, move_ids=moves))


def test_payload_follows_names_into_a_reordered_vocabulary():
    species = [f"s{i}" for i in range(40)]
    moves = [f"m{i}" for i in range(40)]
    train = random_batch(10)
    table = T.FlagsTable.fit(train, featurizer=fake_featurizer(species, moves))
    assert table.key_names == [species[i] for i in table.key_ids]
    assert table.move_names == [moves[i] for i in table.move_ids]
    batch = features_only(random_batch(11))
    # A vocabulary with one more entry in front: every id moves up by one.
    moved = T.from_payload(
        table.to_payload(), fake_featurizer(["new"] + species, ["new"] + moves)
    )
    assert moved.key_ids.tolist() == (table.key_ids + 1).tolist()
    shifted = dict(batch)
    shifted["mon_id"] = batch["mon_id"].copy()
    shifted["mon_id"][..., F.ID_KEY] += 1
    shifted["cand_move"] = np.where(batch["cand_move"] > 0, batch["cand_move"] + 1, 0)
    first, second = table.predict(batch), moved.predict(shifted)
    for name in first:
        assert np.allclose(first[name], second[name]), name
    # A name the new vocabulary lacks keeps its row and is never read.
    gone = T.from_payload(table.to_payload(), fake_featurizer(species[:6], moves))
    assert gone.key_ids.size == table.key_ids.size
    assert not gone.counters and gone.predict(batch)["action"].shape[0] == 160
    # Without names the stored ids stand.
    assert T.from_payload(table.to_payload()).key_ids.tolist() == table.key_ids.tolist()


def test_predict_never_raises():
    table = T.FlagsTable.fit(random_batch(12))
    assert table.predict({}) == {}
    assert table.counters["predict_error:KeyError"] == 1
    batch = features_only(random_batch(13, n=8))
    broken = {k: v for k, v in batch.items() if k != "cand_auto"}
    pred = table.predict(broken)
    uniform = F.uniform_prediction(batch)
    assert np.array_equal(pred["action"], uniform["action"])  # degraded, not raised
    assert table.counters["predict_error:KeyError"] == 2
    wrong = dict(batch)
    wrong["cand_flag"] = batch["cand_flag"][:, :, :2]
    assert np.array_equal(table.predict(wrong)["action"], uniform["action"])
    assert table.counters["predict_error:ValueError"] == 1
    assert sum(table.counters.values()) == 3
    empty = F.take(batch, np.zeros(8, dtype=bool))
    pred = table.predict(empty)  # no examples is a batch like any other
    assert pred["action"].shape == (0, 2, F.action_size(C))
    assert pred["target"].shape == (0, 2, C + 1, F.N_TARGET)
    assert pred["mega"].shape == (0, 2)
    assert sum(table.counters.values()) == 3


@pytest.mark.parametrize("usage", [T.USAGE_RATE, T.USAGE_SHARE])
@pytest.mark.parametrize("cls", [T.SpeciesTable, T.FlagsTable, T.EloTable])
def test_a_table_fitted_on_nothing_still_predicts(cls: Any, usage: str):
    batch = random_batch(16, n=12)
    table = cls.fit(
        batch, mask=np.zeros(12, dtype=bool), config=T.TableConfig(usage=usage)
    )
    assert table.fit_info["slots_counted"] == 0 and table.fit_info["set_keys"] == 0
    pred = table.predict(features_only(batch))
    assert not table.counters
    legal = batch["action_mask"].astype(bool)
    assert np.isfinite(pred["action"]).all() and not pred["action"][~legal].any()
    assert (pred["action"][legal] > 0).all()
    assert pred["action"].sum(-1)[batch["act_mon"] >= 0] == pytest.approx(1.0)
    again = T.from_payload(table.to_payload())
    assert np.array_equal(again.predict(batch)["action"], pred["action"])


def test_another_candidate_count_at_predict_time():
    """The table is keyed by move id, so the candidate axis may differ."""
    table = T.SpeciesTable.fit(rows([{"cands": TWO, "action": 0}] * 3), config=RAW)
    wide = {
        name: np.zeros((1, *spec.shape), dtype=spec.dtype)
        for name, spec in F.layout(6).items()
        if spec.group == "feature"
    }
    wide["act_mon"][:] = (0, -1)
    wide["mon_id"][0, 0, F.ID_KEY] = KEY
    wide["cand_move"][0, 0, 5] = MOVE_A
    wide["cand_flag"][0, 0, 5] = F.CAND_VALID | ATTACK
    wide["cand_tmask"][0, 0, 5] = FOES
    wide["action_mask"][0, 0, [5, 6]] = 1
    action = table.predict(wide)["action"]
    assert action.shape == (1, 2, F.action_size(6))
    assert action[0, 0, 5] == pytest.approx(1.0, abs=1e-6)


def test_config_round_trip_and_coarse_classes():
    config = T.TableConfig(cell_strength=INF, elo_edges=(1200, 1400), floor=1e-3)
    again = T.TableConfig.from_dict({**config.to_dict(), "not_a_setting": 1})
    assert again == config and math.isinf(again.cell_strength)
    assert T.TableConfig.from_dict({}) == T.TableConfig()
    bits = np.array(
        [GUARD, FIRST, ATTACK | F.CAND_SPREAD, ATTACK, F.CAND_DAMAGING, STATUS, 0]
    )
    assert T.coarse_class(bits).tolist() == list(range(7))
    assert len(T.CLASS_NAMES) == T.N_CLASS == T.CLASS_SWITCH + 1
    assert set(T.TABLES) == {T.TABLE_SPECIES, T.TABLE_FLAGS, T.TABLE_ELO}


# --- the fit script's helpers -------------------------------------------------


def test_fit_script_helpers():
    from training import fit_oppmodel_tables as FT

    train, batch = random_batch(14), random_batch(15)
    table = T.FlagsTable.fit(train)
    pred = FT.predict(table, batch)
    assert np.array_equal(pred["action"], table.predict(features_only(batch))["action"])
    got = FT.score(pred, batch)
    assert got["slots_scored"] == got["slots_visible"] + got["slots_censored"]
    assert 0 < got["fine_nll"] and got["action_nll"] <= got["fine_nll"]
    assert 0 <= got["action_top1"] <= got["action_top3"] <= 1
    uniform = FT.score(F.uniform_prediction(features_only(batch)), batch)
    assert uniform["fine_nll"] > 0
    events = FT.calibration(pred, batch)
    for name in (E.INTENT_SWITCH, E.INTENT_PROTECT):
        assert sum(b["slots"] for b in events[name]["bins"]) == events[name]["slots"]
        assert 0 <= events[name]["predicted"] <= 1
    tuned, history = FT.tune(table, batch, {"cell_strength": (0.0, 5.0, INF)}, "fine")
    assert FT.objective(tuned, batch) <= FT.objective(table, batch) + 1e-12
    assert history and history[0]["name"] == "cell_strength"
    assert set(history[0]["nll"]) >= {"0.0", "5.0", "inf"}
    assert FT._plain({"a": INF, "b": (1, np.float32(2.0))}) == {
        "a": "inf",
        "b": [1, 2.0],
    }
    assert set(FT.grid_for(T.TableConfig(), cells=True)) > set(
        FT.grid_for(T.TableConfig(), cells=False)
    )


def test_paired_difference_clusters_by_game():
    from training import fit_oppmodel_tables as FT

    first = np.array([[1.0, 2.0], [3.0, 0.0], [5.0, 5.0]])
    second = np.zeros_like(first)
    scored = np.array([[True, True], [True, False], [True, True]])
    got = FT.paired_difference(first, second, scored, np.array([0, 0, 1]))
    assert got["slots"] == 5 and got["games"] == 2
    assert got["diff"] == pytest.approx(16 / 5)
    # Clusters: sums 6 and 10 over 3 and 2 slots; linearised ratio variance.
    residual = np.array([6 - 3.2 * 3, 10 - 3.2 * 2])
    assert got["se"] == pytest.approx(math.sqrt((residual**2).sum() * 2) / 5)
    assert got["low"] < got["diff"] < got["high"]
    assert (
        FT.paired_difference(first, second, ~scored & scored, np.arange(3))["diff"]
        is None
    )


# --- the fit script, end to end ------------------------------------------------

FIT_SPLITS = ["train", "val", "test", "ladder_holdout"]
STRAY_KEY = 23  # a set key that only evaluation rows carry


def write_fit_dataset(
    directory: Path, fz: F.Featurizer, late: int = 0
) -> dict[str, int]:
    """A finished dataset folder for the fit script: four splits in one shard.

    A third of the val / test / ladder rows carry ``STRAY_KEY``, which no train
    row has. ``late`` test rows are flagged as the time slice (0: an empty
    slice). Two val rows belong to a battle with a ladder-holdout opponent.
    """
    from training import fit_oppmodel_tables as FT

    directory.mkdir(parents=True, exist_ok=True)
    parts, sizes = [], {}
    for code, (name, n) in enumerate(zip(FIT_SPLITS, (90, 36, 30, 24))):
        part = random_batch(20 + code, n=n)
        part["m_split"][:] = code
        part["m_battle"] = (1000 * code + np.arange(n) // 2).astype(np.int32)
        part["m_flag"] = np.zeros(n, dtype=np.uint8)
        if name != "train":
            part["mon_id"][::3, :, F.ID_KEY] = STRAY_KEY
        if name == "test":
            part["m_flag"][:late] = FT.FLAG_TIME_SLICE
        if name == "val":
            part["m_flag"][:2] = FT.FLAG_HOLDOUT_BATTLE
        parts.append(part)
        sizes[name] = n
    data = F.concat_batches(parts)
    F.save_batch(directory / "shard-00000.npz", data)
    manifest = {
        "tag": "tiny",
        "splits": FIT_SPLITS,
        "formats": ["someformat"],
        "shards": [{"file": "shard-00000.npz", "examples": int(data["turn"].shape[0])}],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    fz.save(directory)
    return sizes


@pytest.fixture()
def small_grids(monkeypatch: pytest.MonkeyPatch) -> None:
    """One value per strength: the run's plumbing, not its tuning, is under test."""
    from training import fit_oppmodel_tables as FT

    monkeypatch.setattr(FT, "KEY_GRID", {"key_strength": (30.0, 100.0)})
    monkeypatch.setattr(FT, "CELL_GRID", {"cell_strength": (30.0, 100.0)})
    monkeypatch.setattr(FT, "USAGE_GRID", {u: {} for u in FT.USAGE_GRID})
    monkeypatch.setattr(FT, "MEGA_GRID", {"mega_strength": (10.0,)})
    monkeypatch.setattr(
        FT, "ELO_EDGE_SETS", ((1300,), (1200, 1400), (1100, 1300, 1500))
    )
    monkeypatch.setattr(FT, "ELO_STRENGTHS", (50.0,))
    monkeypatch.setattr(FT, "ABLATIONS", FT.ABLATIONS[-1:])
    monkeypatch.setattr(FT, "RETUNED", ())


def test_run_fits_on_train_only_and_survives_an_empty_split(
    tmp_path: Path, fz: F.Featurizer, small_grids: None, monkeypatch: pytest.MonkeyPatch
):
    """The whole script on a tiny dataset whose time slice is empty.

    Pins the split hygiene: the tables are counted on the train rows and
    nothing else, validation leaves out the holdout battles, and no predict
    call is handed a label or a meta array.
    """
    from training import fit_oppmodel_tables as FT
    from vgc_bench.src.oppmodel.artifact import load_predictor

    sizes = write_fit_dataset(tmp_path / "data", fz)
    seen: list[set[str]] = []
    real = T._CountTable.predict

    def watched(self: Any, batch: Any) -> Any:
        seen.append(set(batch))
        return real(self, batch)

    monkeypatch.setattr(T._CountTable, "predict", watched)
    lines: list[str] = []
    out = tmp_path / "fit"
    report = FT.run(tmp_path / "data", out, log=lines.append)

    # Nothing but features ever reached a table.
    assert len(seen) > 20
    leaked = {name for names in seen for name in names if name.startswith(("y_", "m_"))}
    assert leaked == set()
    # Counted on the train rows only: the stray key of the other splits is unknown.
    data = report["dataset"]
    assert data["fit_examples"] == sizes["train"] == 90
    assert data["evaluation_examples"] == {
        "val": sizes["val"] - 2,
        "test": sizes["test"],
        "test_time_slice": 0,
        "ladder_holdout": sizes["ladder_holdout"],
    }
    assert data["validation_rows_left_out_holdout_battles"] == 2
    assert set(report["tables"]) == set(FT.ARTIFACTS)
    for label, file_name in FT.ARTIFACTS.items():
        assert report["tables"][label]["fit_info"]["examples"] == 90, label
        loaded = load_predictor(out / file_name)
        assert loaded.name == label and loaded.predictor.fit_info["examples"] == 90
        assert STRAY_KEY not in loaded.predictor.key_ids.tolist()
        assert KEY in loaded.predictor.key_ids.tolist()
        assert loaded.meta["extra"]["formats"] == ["someformat"]
        assert (
            report["artifacts"][label]["reloaded_max_abs_difference_on_validation"] == 0
        )
    # The empty time slice has no score; the others do.
    for label in FT.ARTIFACTS:
        scores = report["tables"][label]["scores"]
        assert scores["test_time_slice"]["fine_nll"] is None
        assert scores["test_time_slice"]["slots_scored"] == 0
        assert all(scores[name]["fine_nll"] > 0 for name in ("val", "test"))
    assert any(
        line.startswith("floors (fine NLL)") and " / - / " in line for line in lines
    )
    # The fourth artifact is the flags table with its target context off.
    plain = load_predictor(out / FT.ARTIFACTS[FT.TABLE_FLAGS_PLAIN]).predictor
    flags = load_predictor(out / FT.ARTIFACTS[T.TABLE_FLAGS]).predictor
    assert type(plain) is T.FlagsTable and not plain.config.target_context
    assert flags.config.target_context
    assert replace(plain.config, target_context=True) == flags.config
    probe = features_only(random_batch(31))
    assert_same_prediction(plain, flags.with_config(target_context=False), probe)
    assert "flags_minus_flags_plain_targets" in report["comparisons"]
    # Both reports are written and the markdown is the JSON's rendering.
    stored = json.loads((out / FT.REPORT_JSON).read_text(encoding="utf-8"))
    assert (out / FT.REPORT_MD).read_text(encoding="utf-8") == FT.render_markdown(
        stored
    )
    assert "Target context of the flags table" in (out / FT.REPORT_MD).read_text()


def test_fit_script_refuses_to_write_over_a_fit(
    tmp_path: Path,
    fz: F.Featurizer,
    small_grids: None,
    capsys: pytest.CaptureFixture[str],
):
    from training import fit_oppmodel_tables as FT

    write_fit_dataset(tmp_path / "data", fz, late=6)
    out = tmp_path / "fit"
    base = ["--dataset", str(tmp_path / "data"), "--out", str(out)]
    assert FT.main(base) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == FT.DONE
    report = json.loads((out / FT.REPORT_JSON).read_text(encoding="utf-8"))
    assert report["dataset"]["evaluation_examples"]["test_time_slice"] == 6
    before = {path.name: path.read_bytes() for path in out.iterdir()}
    assert set(FT.ARTIFACTS.values()) <= set(before)
    # A second run into the same directory is refused and changes nothing.
    assert FT.main(base) == 1
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.startswith(FT.FAILED) and "--overwrite" in last
    assert {path.name: path.read_bytes() for path in out.iterdir()} == before
    with pytest.raises(FT.FitError, match="--overwrite"):
        FT.run(tmp_path / "data", out, log=lambda text: None)
    # One leftover table file is enough to refuse.
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / FT.ARTIFACTS[T.TABLE_FLAGS]).write_bytes(b"the bar")
    assert FT.main(["--dataset", str(tmp_path / "data"), "--out", str(partial)]) == 1
    assert (partial / FT.ARTIFACTS[T.TABLE_FLAGS]).read_bytes() == b"the bar"
    assert [path.name for path in partial.iterdir()] == [FT.ARTIFACTS[T.TABLE_FLAGS]]
    capsys.readouterr()
    # --render-only reads the report and is not a fit: no flag needed.
    assert FT.main(["--out", str(out), "--render-only"]) == 0
    # With --overwrite the fit is replaced.
    assert FT.main([*base, "--overwrite", "--limit", "20"]) == 0
    again = json.loads((out / FT.REPORT_JSON).read_text(encoding="utf-8"))
    assert again["dataset"]["limit_per_split"] == 20
    assert again["dataset"]["fit_examples"] == 20


def test_a_smoke_fit_cannot_land_on_the_default_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    from training import fit_oppmodel_tables as FT

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the fit must not start")

    monkeypatch.setattr(FT, "run", never)
    assert FT.main(["--limit", "3000"]) == 1
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.startswith(FT.FAILED) and "--out" in last and FT.DEFAULT_OUT in last
    # Any failure ends in one FIT_FAILED line, never in a traceback.
    monkeypatch.setattr(FT, "run", lambda *a, **k: 1 / 0)
    assert FT.main(["--out", str(tmp_path / "x")]) == 1
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.startswith(f"{FT.FAILED} ZeroDivisionError")
    assert FT.main(["--out", str(tmp_path / "missing"), "--render-only"]) == 1
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith(FT.FAILED)
    assert FT.main(["--no-such-flag"]) == 2


def test_split_masks_keep_train_validation_and_the_time_slice_apart():
    from training import fit_oppmodel_tables as FT

    names = ["train", "val", "test", "ladder_holdout", "own_unrated"]
    data = {
        "m_split": np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 4], dtype=np.uint8),
        "m_flag": np.array([1, 4, 0, 4, 12, 1, 0, 1, 16, 0], dtype=np.uint8),
    }
    train, masks, left_out = FT.split_masks(data, names)
    assert train.tolist() == [True, True] + [False] * 8
    assert set(masks) == set(FT.EVAL_SPLITS)
    assert masks["val"].tolist() == [False, False, True] + [False] * 7
    assert left_out == 2
    assert masks["test"].tolist() == [False] * 5 + [True] * 3 + [False] * 2
    assert (
        masks["test_time_slice"].tolist()
        == [False] * 5 + [True, False, True] + [False] * 2
    )
    assert masks["ladder_holdout"].tolist() == [False] * 8 + [True, False]
    # No row is in train and in an evaluation split.
    assert not any((train & mask).any() for mask in masks.values())


# --- team-agnostic rule -------------------------------------------------------


def test_library_holds_no_species_move_item_or_ability_name():
    """String literals of the tables, the artifact code and the fit script."""
    vocab = F.Vocab.build()
    known = set(E.moves_dex()) | set(E.pokedex())
    known |= set(vocab.items[3:]) | set(vocab.abilities[3:])
    for path in (
        ROOT / "vgc_bench/src/oppmodel/tables.py",
        ROOT / "vgc_bench/src/oppmodel/artifact.py",
        ROOT / "training/fit_oppmodel_tables.py",
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


# --- the fitted tables on the real dataset (skipped when absent) ---------------


@pytest.mark.skipif(
    not (DATASET / "manifest.json").is_file()
    or not (FITTED / "fit_report.json").is_file(),
    reason="results_oppmodel/v1_ondisk or results_oppmodel/tables_v1 is absent",
)
def test_written_artifacts_reproduce_the_reported_numbers():
    import json

    from training import fit_oppmodel_tables as FT
    from vgc_bench.src.oppmodel.artifact import load_predictor

    report = json.loads((FITTED / "fit_report.json").read_text(encoding="utf-8"))
    batch, manifest = F.load_dataset(DATASET, splits=["val"])
    # Validation as the fit script takes it: without the holdout battles.
    _, masks, left_out = FT.split_masks(batch, list(manifest["splits"]))
    batch = F.take(batch, masks["val"])
    data = report["dataset"]
    assert left_out == data["validation_rows_left_out_holdout_battles"] > 0
    assert len(batch["turn"]) == data["evaluation_examples"]["val"]
    found: dict[str, float] = {}
    for label, file_name in FT.ARTIFACTS.items():
        loaded = load_predictor(FITTED / file_name)
        assert loaded.kind == "table" and loaded.name == label
        assert loaded.meta["dex_signature_diff"] == []
        nll = F.slot_nll(FT.predict(loaded.predictor, batch), batch)
        found[label] = float(nll["fine"][nll["fine_scored"]].mean())
        stored = report["tables"][label]["scores"]["val"]["fine_nll"]
        assert found[label] == pytest.approx(stored, abs=1e-9), label
        assert not loaded.predictor.counters
    floors = report["floors"]
    assert (
        found[T.TABLE_FLAGS]
        < found[FT.TABLE_FLAGS_PLAIN]  # the bar without its target context
        < floors["prior"]["val"]["fine_nll"]
        < floors["uniform"]["val"]["fine_nll"]
    )
    assert found[T.TABLE_FLAGS] < found[T.TABLE_SPECIES]
