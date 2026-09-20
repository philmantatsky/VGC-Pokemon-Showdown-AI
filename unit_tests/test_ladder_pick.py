"""tools/ladder_pick.py: the pre-registered ladder-candidate rule evaluated
lazily -- human arms first, PPO arms only for the leader, fall through on a
disqualification, generalist when no specialist qualifies, nothing below the
cross-team bar."""

from __future__ import annotations

import json
from pathlib import Path

from tools.ladder_pick import next_action

SAVES = Path("saves")
GENERALIST = Path("v1/19660800.zip")


def _arm(root: Path, stem: str, name: str, candidate: float, baseline: float) -> None:
    path = root / stem / "screening"
    path.mkdir(parents=True, exist_ok=True)
    arms = {
        "distilled_policy": {"win_rate": candidate},
        "champion_policy": {"win_rate": baseline},
    }
    (path / f"battery_{name}.json").write_text(json.dumps({"arms": arms}))


def _ppo(root: Path, stem: str, deltas: tuple[float, float, float]) -> None:
    for name, delta in zip(("frozen", "rotation1", "rotation2"), deltas):
        _arm(root, stem, name, 0.85 + delta / 100, 0.85)


def _act(root: Path, finalists: list[str]) -> str:
    return next_action(root, SAVES, GENERALIST, finalists)


def test_human_arms_come_first_then_ppo_for_the_leader(tmp_path: Path) -> None:
    assert _act(tmp_path, ["a", "b"]) == "RUN_HUMAN a"
    _arm(tmp_path, "a", "human_bc", 0.86, 0.84)
    assert _act(tmp_path, ["a", "b"]) == "RUN_HUMAN b"
    _arm(tmp_path, "b", "human_bc", 0.88, 0.84)
    assert _act(tmp_path, ["a", "b"]) == "RUN_PPO b"
    _ppo(tmp_path, "b", (1.0, -1.5, 0.5))
    action = _act(tmp_path, ["a", "b"])
    assert action.startswith("PICK saves/b.zip specialist b: human 88.0% (+4.0pp")


def test_ppo_breach_falls_through_to_the_next_finalist(tmp_path: Path) -> None:
    _arm(tmp_path, "a", "human_bc", 0.86, 0.84)
    _arm(tmp_path, "b", "human_bc", 0.88, 0.84)
    _ppo(tmp_path, "b", (1.0, -2.5, 0.5))
    assert _act(tmp_path, ["a", "b"]) == "RUN_PPO a"
    _ppo(tmp_path, "a", (0.0, 0.0, -1.9))
    assert _act(tmp_path, ["a", "b"]).startswith("PICK saves/a.zip specialist a")


def test_generalist_goes_when_no_specialist_qualifies(tmp_path: Path) -> None:
    _arm(tmp_path, "a", "human_bc", 0.82, 0.84)  # -2.0pp: below the human floor
    _arm(tmp_path, "b", "human_bc", 0.86, 0.84)
    _ppo(tmp_path, "b", (-3.0, 0.0, 0.0))
    action = _act(tmp_path, ["a", "b"])
    assert action.startswith(f"PICK {GENERALIST} generalist (84.0% human)")
    assert "a human -2.0pp" in action and "b frozen -3.0pp" in action


def test_nothing_goes_below_the_cross_team_bar(tmp_path: Path) -> None:
    _arm(tmp_path, "a", "human_bc", 0.78, 0.77)  # fine on delta, not on the bar
    action = _act(tmp_path, ["a"])
    assert action.startswith("PICK_NONE") and "below the cross-team bar" in action
