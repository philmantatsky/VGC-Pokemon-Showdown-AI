"""Brain v1 potential-based reward shaping (2026-09-10).

reward += Phi(s') - Phi(s), Phi := 0 at the terminal state. With gamma 1 every
episode's shaped return telescopes to exactly the terminal +-1, so the optimal
policy is unchanged; both weights at 0 reproduce the original signal bit for bit.
"""

from __future__ import annotations

from types import SimpleNamespace as NS
from typing import Any, cast

import pytest
from poke_env.battle import AbstractBattle

from vgc_bench.src.env import ShowdownEnv


class _Env:
    """Only what calc_reward touches, bound to the real methods."""

    material_potential = ShowdownEnv.material_potential
    calc_reward = ShowdownEnv.calc_reward

    def __init__(self, faint: float, hp: float):
        self._shaping_faint = faint
        self._shaping_hp = hp
        self._potentials: dict[str, float] = {}


def _mon(hp: float = 1.0, fainted: bool = False) -> NS:
    return NS(current_hp_fraction=0.0 if fainted else hp, fainted=fainted)


def _battle(
    ours, theirs, finished=False, won=False, lost=False, tag="b1"
) -> AbstractBattle:
    fake: Any = NS(
        battle_tag=tag,
        finished=finished,
        won=won,
        lost=lost,
        team={f"p1: m{i}": m for i, m in enumerate(ours)},
        opponent_team={f"p2: m{i}": m for i, m in enumerate(theirs)},
    )
    return cast(AbstractBattle, fake)


def test_off_by_default_is_the_terminal_signal():
    env = _Env(0.0, 0.0)
    ours, theirs = [_mon()] * 6, [_mon(0.2)] * 4
    assert env.calc_reward(_battle(ours, theirs)) == 0.0
    assert env.calc_reward(_battle(ours, theirs, finished=True, won=True)) == 1.0
    assert env.calc_reward(_battle(ours, theirs, finished=True, lost=True)) == -1.0
    assert env._potentials == {}


def test_potential_is_zero_at_the_start_whatever_is_revealed():
    env = _Env(0.25, 0.1)
    assert env.material_potential(_battle([_mon()] * 6, [_mon()] * 6)) == 0.0
    assert env.material_potential(_battle([_mon()] * 6, [_mon()] * 4)) == 0.0
    assert env.material_potential(_battle([_mon()] * 6, [])) == 0.0


def test_shaped_return_telescopes_to_the_terminal_reward():
    env = _Env(0.25, 0.1)
    states = [
        _battle([_mon()] * 6, [_mon()] * 4),  # turn 0
        _battle([_mon(0.5)] + [_mon()] * 5, [_mon(0.7)] + [_mon()] * 3),
        _battle([_mon(0.5)] + [_mon()] * 5, [_mon(fainted=True)] + [_mon()] * 3),
        _battle(
            [_mon(fainted=True)] + [_mon()] * 5,
            [_mon(fainted=True), _mon(0.1)] + [_mon()] * 2,
        ),
        _battle(
            [_mon(fainted=True)] + [_mon()] * 5,
            [_mon(fainted=True)] * 4,
            finished=True,
            won=True,
        ),
    ]
    rewards = [env.calc_reward(s) for s in states]
    assert rewards[0] == 0.0
    # taking a KO is rewarded on the turn it happens
    assert rewards[2] > 0
    # losing a Pokemon is punished even while the game goes on
    assert rewards[3] < rewards[2]
    assert sum(rewards) == pytest.approx(1.0)
    assert env._potentials == {}  # terminal state cleaned up


def test_lost_game_also_telescopes_and_battles_are_independent():
    env = _Env(0.25, 0.1)
    a = [
        _battle([_mon()] * 6, [_mon()] * 4, tag="a"),
        _battle([_mon(fainted=True)] + [_mon()] * 5, [_mon()] * 4, tag="a"),
    ]
    b = [
        _battle([_mon()] * 6, [_mon()] * 4, tag="b"),
        _battle([_mon()] * 6, [_mon(fainted=True)] + [_mon()] * 3, tag="b"),
    ]
    ra = [env.calc_reward(s) for s in a]
    rb = [env.calc_reward(s) for s in b]
    assert ra[1] < 0 < rb[1]
    a_end = _battle(
        [_mon(fainted=True)] * 4 + [_mon()] * 2,
        [_mon()] * 4,
        finished=True,
        lost=True,
        tag="a",
    )
    b_end = _battle(
        [_mon()] * 6, [_mon(fainted=True)] * 4, finished=True, won=True, tag="b"
    )
    ra.append(env.calc_reward(a_end))
    rb.append(env.calc_reward(b_end))
    assert sum(ra) == pytest.approx(-1.0)
    assert sum(rb) == pytest.approx(1.0)


def test_negative_weights_are_refused():
    bare: Any = NS()
    with pytest.raises(ValueError):
        ShowdownEnv.__init__(bare, shaping_faint=-1.0)
