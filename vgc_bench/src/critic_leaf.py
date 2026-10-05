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

import json
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
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


@dataclass(frozen=True)
class LeafCalibration:
    """Win probability as a function of the raw leaf value, one monotone map per
    phase of the game (``evaluation/leaf_calibration.py`` fits it from games).

    The critic was trained against opponents the brain usually beats, so against an
    equal its raw value is optimistic and badly scaled: 2026-10-04, mirror games,
    raw +0.6..+0.8 won 40-52% and anything from -0.8 to +0.6 won 25-43%; on turns
    1-2 it carried no information at all (AUC 0.50). A calibrated leaf turns payoff
    differences into differences in win probability, so the planner's margins mean
    the same thing early and late.

    ``phases`` are ``(last turn of the phase or None, knots)`` in turn order; knots
    are ``(raw value, win probability)`` with both ascending.
    """

    phases: tuple[tuple[int | None, tuple[tuple[float, float], ...]], ...]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "LeafCalibration":
        phases = []
        for phase in payload["phases"]:
            knots = tuple((float(raw), float(p)) for raw, p in phase["knots"])
            if not knots:
                raise ValueError("a calibration phase needs at least one knot")
            if any(b[0] < a[0] or b[1] < a[1] for a, b in zip(knots, knots[1:])):
                raise ValueError("calibration knots must ascend in both coordinates")
            if not all(0.0 <= p <= 1.0 for _raw, p in knots):
                raise ValueError("calibrated win probabilities must be in [0, 1]")
            through = phase.get("through_turn")
            phases.append((None if through is None else int(through), knots))
        if not phases or phases[-1][0] is not None:
            raise ValueError("the last calibration phase must be open-ended")
        return cls(tuple(phases))

    @classmethod
    def load(cls, path: str | Path) -> "LeafCalibration":
        return cls.from_payload(json.loads(Path(path).read_text()))

    def win_probability(self, raw: float, turn: int) -> float:
        knots = next(
            knots
            for through, knots in self.phases
            if through is None or turn <= through
        )
        if raw <= knots[0][0]:
            return knots[0][1]
        if raw >= knots[-1][0]:
            return knots[-1][1]
        index = bisect_right([knot[0] for knot in knots], raw)
        (x0, y0), (x1, y1) = knots[index - 1], knots[index]
        return y0 + (y1 - y0) * (raw - x0) / (x1 - x0) if x1 > x0 else y1


def raw_leaf_value(
    policy: Any,
    obs_dict: Any,
    battle: Any,
    faint: float = SHAPING_FAINT,
    hp: float = SHAPING_HP,
) -> float:
    """The critic plus the shaping potential, before any clipping or calibration."""
    with torch.no_grad():
        _logits, value = policy.get_logits(obs_dict, actor_grad=False)
    return float(value.item()) + material_potential(battle, faint, hp)


class CriticLeafEvaluator:
    """Leaf value = the brain's critic + the shaping potential, clipped to [-1, 1];
    with a ``calibration``, twice the calibrated win probability minus one."""

    def __init__(
        self,
        adapter: Any,
        faint: float = SHAPING_FAINT,
        hp: float = SHAPING_HP,
        calibration: LeafCalibration | None = None,
    ):
        self.adapter = adapter
        self.faint = faint
        self.hp = hp
        self.calibration = calibration

    def __call__(self, node: Any, role: str) -> float:
        battle, _obs, _mask, obs_dict = self.adapter._inputs(
            node.state, node.requests, role
        )
        with self.adapter.inference_lock:
            estimate = raw_leaf_value(
                self.adapter.policy, obs_dict, battle, self.faint, self.hp
            )
        if self.calibration is not None:
            turn = int(getattr(battle, "turn", 0) or 0)
            return 2.0 * self.calibration.win_probability(estimate, turn) - 1.0
        return max(-1.0, min(1.0, estimate))
