"""The Earth Power lesson (2026-10-04, the user on T6e: "just use ep when its super
effective on a pokemon and it does better damage than the rest of the moves and
theres no better switch in"): the data marks the legal Earth Power actions, and the
focus lesson moves the attack mass to the hardest-hitting attack only in those
positions, leaving Protect / switch mass and every other position to the brain."""

from __future__ import annotations

import json

import numpy as np
import pytest

from training.gen_tactical_data import save_shard
from training.tactical_sft import build_targets, focus_rows, load_data
from training.tactical_teacher import focus_actions, position_facts, target_distribution
from unit_tests.ladder_position import move_action, position
from vgc_bench.src.utils import act_len

LEGAL = np.zeros(act_len, dtype=np.int8)
LEGAL[7:47] = 1
EP = frozenset({"earthpower"})


def _rain_vs_incineroar():
    """Torkoal (T6e) at full HP after Pelipper's rain took the sun: Fire moves are
    halved and Incineroar resists them; Earth Power hits it super effectively."""
    return position(
        [
            "|switch|p1a: Torkoal|Torkoal, L50, M|177/177",
            "|switch|p1b: Farigiraf|Farigiraf, L50, F|227/227",
            "|switch|p2a: Incineroar|Incineroar, L50, M|100/100",
            "|switch|p2b: Pelipper|Pelipper, L50, M|100/100",
            "|-weather|SunnyDay|[from] ability: Drought|[of] p1a: Torkoal",
            "|-weather|RainDance|[from] ability: Drizzle|[of] p2b: Pelipper",
            "|turn|1",
        ],
        team="teams/candidates_mc/T6e.txt",
    )


def test_focus_marks_the_earth_power_actions_only():
    battle = _rain_vs_incineroar()
    focus = focus_actions(battle, 0, LEGAL, EP)
    assert focus[move_action(battle, 0, "earthpower", 1)]
    assert not focus[move_action(battle, 0, "eruption", 0)]
    assert not focus_actions(battle, 1, LEGAL, EP).any()  # Farigiraf has none
    assert not focus_actions(battle, 0, LEGAL, frozenset()).any()


def test_super_effective_earth_power_out_values_the_fire_moves():
    battle = _rain_vs_incineroar()
    flags, values = position_facts(battle, 0, LEGAL, None)
    ep = move_action(battle, 0, "earthpower", 1)
    for move in ("eruption", "heatwave"):
        assert values[ep] > values[move_action(battle, 0, move, 0)]
    assert flags[move_action(battle, 0, "earthpower", 2)]  # Pelipper flies: immune


def test_the_lesson_moves_attack_mass_to_earth_power_and_keeps_protect():
    battle = _rain_vs_incineroar()
    flags, values = position_facts(battle, 0, LEGAL, None)
    ep = move_action(battle, 0, "earthpower", 1)
    eruption = move_action(battle, 0, "eruption", 0)
    heat_wave = move_action(battle, 0, "heatwave", 0)
    protect = move_action(battle, 0, "protect", 0)
    probs = np.zeros(act_len)
    probs[[eruption, heat_wave, ep, protect]] = [0.5, 0.2, 0.1, 0.2]
    target = target_distribution(probs, flags, values, tau=0.1)
    assert target is not None
    attacks = [eruption, heat_wave, ep]
    assert target[ep] == max(target[a] for a in attacks)
    assert target[ep] > 0.4
    assert target[protect] == pytest.approx(0.2)  # whether to attack stays its own


def test_build_targets_only_teaches_the_focus_positions():
    rows = 2
    p = np.full((rows, act_len), 0.0)
    p[:, [7, 8, 9]] = [0.5, 0.3, 0.2]
    values = np.full((rows, 2, act_len), np.nan, dtype=np.float32)
    values[:, :, 7] = 0.1
    values[:, :, 8] = 0.9
    useless = np.zeros((rows, 2, act_len), dtype=bool)
    only = np.array([[True, False], [False, False]])
    q0, q1, lesson = build_targets(p, p.copy(), useless, values, 0.1, only=only)
    assert q0[0, 8] > p[0, 8]  # taught: the stronger attack gains
    np.testing.assert_allclose(q0[1], p[1])  # untaught rows are the brain's own
    np.testing.assert_allclose(q1, p)
    assert lesson.tolist() == [[True, False], [False, False]]


def test_focus_rows_need_a_valued_focus_move():
    values = np.full((2, 2, act_len), np.nan, dtype=np.float32)
    focus = np.zeros((2, 2, act_len), dtype=bool)
    focus[0, 0, 17] = focus[1, 1, 17] = True
    values[0, 0, 17] = 0.4  # row 0: Earth Power valued; row 1: it is not
    rows = focus_rows({"focus": focus, "values": values})
    assert rows.tolist() == [[True, False], [False, False]]


def test_shards_record_focus_and_mixed_shards_are_refused(tmp_path):
    record = {
        "obs": np.zeros(3, dtype=np.float32),
        "mask": np.zeros(act_len, dtype=np.int8),
        "played": np.zeros(2, dtype=np.int16),
        "useless": np.zeros((2, act_len), dtype=bool),
        "values": np.full((2, act_len), np.nan, dtype=np.float32),
        "doomed": np.zeros(2, dtype=np.float32),
        "protect": np.zeros((2, act_len), dtype=bool),
        "wasted": np.zeros((2, act_len), dtype=bool),
        "drain": np.zeros((2, act_len), dtype=bool),
        "receive": np.zeros((2, act_len), dtype=bool),
        "focus": np.zeros((2, act_len), dtype=bool),
        "turn": 1,
        "battle": "battle-x",
    }
    assert save_shard(tmp_path / "a.npz", [record]) == 1
    assert "focus" in load_data(tmp_path)
    np.savez_compressed(tmp_path / "b.npz", obs=np.zeros((1, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="different fields"):
        load_data(tmp_path)


def _log(tmp_path, before, after, rows=400):
    def row(epoch, agreement, ce):
        return {
            "epoch": epoch,
            "cross_entropy": ce,
            "focus_rows": rows,
            "focus_agreement": agreement,
            "focus_best_share": 0.5,
            "focus_worse_share": 0.2,
        }

    log = {"val_before": row(0, before, 9.0), "epochs": [row(1, after, 1.0)]}
    (tmp_path / "log.json").write_text(json.dumps(log))
    return tmp_path


def test_the_lesson_gate_needs_the_disagreements_halved(tmp_path):
    from training.t6e_ep_gate import lesson_holds

    assert lesson_holds(_log(tmp_path, 0.80, 0.90)) == []  # 20% -> 10% left
    assert lesson_holds(_log(tmp_path, 0.80, 0.89)) == ["lesson did not take"]
    assert "unmeasured" in lesson_holds(_log(tmp_path, 0.5, 0.9, rows=50))[0]
