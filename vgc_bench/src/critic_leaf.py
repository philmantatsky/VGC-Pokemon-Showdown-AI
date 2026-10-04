"""The brain's own critic as the exact search's leaf value (2026-10-04, the matrix
search; RESEARCH_TOP_BOTS.md).

The one-turn nash search scores every child position. The outcome net it used
(results_outcome_v2h) was calibrated in August on the Reg M-B champion and MB430,
and in the first 4-game smoke run the search overrode T6ep's own pick in 18 of 40
decisions and lost all four games. mikumiku37 scores leaves with the value head
trained alongside its policy; this does the same with the deployed brain's PPO critic.

That critic was trained with potential-based shaping (vgc_bench/src/env.py: reward +=
Phi(s') - Phi(s), Phi(terminal) = 0, gamma 1), so it estimates E[result] - Phi(s).
Adding Phi(s) back gives the expected result on the planner's [-1, 1] scale.
"""

from __future__ import annotations

from typing import Any

import torch

# the shaping weights of the T6-family training runs (--shaping_faint / --shaping_hp)
SHAPING_FAINT = 0.10
SHAPING_HP = 0.05


def material_potential(
    battle: Any, faint: float = SHAPING_FAINT, hp: float = SHAPING_HP
) -> float:
    """Phi(s) exactly as ShowdownEnv.material_potential computes it."""
    ours = list(battle.team.values())
    theirs = list(battle.opponent_team.values())
    faints = sum(m.fainted for m in theirs) - sum(m.fainted for m in ours)
    our_hp = sum(m.current_hp_fraction for m in ours) + (6 - len(ours))
    their_hp = sum(m.current_hp_fraction for m in theirs) + (6 - len(theirs))
    return faint * faints + hp * (our_hp - their_hp)


class CriticLeafEvaluator:
    """Leaf value = the brain's critic + the shaping potential, clipped to [-1, 1]."""

    def __init__(
        self, adapter: Any, faint: float = SHAPING_FAINT, hp: float = SHAPING_HP
    ):
        self.adapter = adapter
        self.faint = faint
        self.hp = hp

    def __call__(self, node: Any, role: str) -> float:
        battle, _obs, _mask, obs_dict = self.adapter._inputs(
            node.state, node.requests, role
        )
        with self.adapter.inference_lock, torch.no_grad():
            _logits, value = self.adapter.policy.get_logits(obs_dict, actor_grad=False)
        estimate = float(value.item()) + material_potential(battle, self.faint, self.hp)
        return max(-1.0, min(1.0, estimate))
