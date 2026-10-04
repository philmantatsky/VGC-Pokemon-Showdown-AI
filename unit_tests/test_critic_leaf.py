"""The critic leaf value (vgc_bench/src/critic_leaf.py): the brain's PPO critic plus
the shaping potential it was trained with, so leaves compare on the [-1, 1] result
scale (2026-10-04, the matrix search)."""

from __future__ import annotations

import threading
from types import SimpleNamespace as NS

import pytest
import torch

from vgc_bench.src.critic_leaf import CriticLeafEvaluator, material_potential


def _battle(ours, theirs):
    def mons(spec):
        return {
            str(i): NS(fainted=hp == 0, current_hp_fraction=hp)
            for i, hp in enumerate(spec)
        }

    return NS(team=mons(ours), opponent_team=mons(theirs))


def test_potential_matches_the_training_formula():
    # our six: one fainted, one at half; their four brought: two fainted
    battle = _battle([0, 0.5, 1, 1, 1, 1], [0, 0, 1, 1])
    faints = 2 - 1
    hp = (0 + 0.5 + 4) - (0 + 0 + 2 + 2)  # their two unbrought count as full
    assert material_potential(battle) == pytest.approx(0.10 * faints + 0.05 * hp)
    assert material_potential(_battle([1] * 6, [1] * 4)) == pytest.approx(0.0)


def test_leaf_adds_the_potential_back_and_clips():
    battle = _battle([1] * 6, [0, 0, 0, 1])  # three of theirs down
    phi = material_potential(battle)
    for raw, expected in ((0.2, 0.2 + phi), (0.95, 1.0), (-1.5, -1.0)):
        adapter = NS(
            _inputs=lambda *_args: (battle, None, None, {}),
            inference_lock=threading.Lock(),
            policy=NS(
                get_logits=lambda _obs, actor_grad=False, raw=raw: (
                    torch.zeros(1),
                    torch.tensor([[raw]]),
                )
            ),
        )
        node = NS(state={}, requests=[{}, {}])
        assert CriticLeafEvaluator(adapter)(node, "p1") == pytest.approx(expected)
