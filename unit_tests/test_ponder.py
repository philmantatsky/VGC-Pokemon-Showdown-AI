from __future__ import annotations

import threading

import pytest

from vgc_bench.src import ponder
from vgc_bench.src.exact_planner import ExactNode
from vgc_bench.src.ponder import (
    BackgroundPonder,
    PonderConfig,
    PonderRoot,
    _diverse_choices,
)


def _node(identifier="root"):
    return ExactNode(
        state={
            "id": identifier,
            "field": {"weather": "", "terrain": "", "pseudoWeather": {}},
            "sides": [
                {"pokemon": [], "sideConditions": {}},
                {"pokemon": [], "sideConditions": {}},
            ],
        },
        requests=[{}, {}],
        turn=1,
        request_state="move",
    )


class _FakeBridge:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def choices(self, _state, role):
        return ["ours"] if role == "p1" else ["reply-a", "reply-b"]

    def simulate_batch(self, state, branches, timeout_s=None):
        del state, timeout_s
        rows = []
        for branch in branches:
            score = 0.4 if branch["p2_choice"] == "reply-a" else -0.2
            child = _node(branch["p2_choice"])
            child.state["score"] = score
            rows.append(
                {
                    "state": child.state,
                    "requests": child.requests,
                    "turn": 2,
                    "request_state": "move",
                    "ended": False,
                    "winner": None,
                }
            )
        return rows


def test_ponder_config_rejects_unbounded_values():
    with pytest.raises(ValueError):
        PonderConfig(budget_s=20)
    with pytest.raises(ValueError):
        PonderConfig(chance_samples=3)


def test_diverse_choice_cap_covers_both_slots_move_families():
    choices = [
        "move tackle +1, move protect",
        "move tackle +2, move tailwind",
        "move protect, move protect",
        "switch 3, move tailwind",
    ]
    selected = _diverse_choices(choices, 3)
    families = [set(), set()]
    for choice in selected:
        for slot, family in enumerate(ponder._families(choice)):
            families[slot].add(family)
    assert "move tackle" in families[0]
    assert "move protect" in families[0] or "switch 3" in families[0]
    assert {"move protect", "move tailwind"}.issubset(families[1])


def test_diverse_choice_cap_keeps_likely_replies_first():
    choices = [
        "move tackle +1, move protect",
        "move tackle +2, move tailwind",
        "switch 3, move tailwind",
        "move protect, move protect",
    ]
    likely = choices[-1]
    selected = _diverse_choices(choices, 2, preferred=[likely])
    assert selected[0] == likely
    assert len(selected) == 2


def test_background_ponder_expands_replies_in_isolated_bridge(monkeypatch):
    monkeypatch.setattr(ponder, "ExactShowdownBridge", _FakeBridge)
    monkeypatch.setattr(
        ponder, "strategic_value", lambda node, _role: float(node.state["score"])
    )
    job = BackgroundPonder(
        [PonderRoot(_node(), 1.0, "world")],
        "ours",
        generation=7,
        config=PonderConfig(budget_s=1, max_opponent_choices=8),
    )
    job.start()
    job.join(1)
    result = job.result_if_done()
    assert result is not None
    assert result.error is None
    assert result.roots_completed == 1
    assert result.choices_simulated == 2
    assert {row.opponent_choice for row in result.outcomes} == {
        "reply-a",
        "reply-b",
    }
    assert result.reference_values["world"] == pytest.approx(-0.2)


def test_background_ponder_result_poll_never_waits(monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    class SlowBridge(_FakeBridge):
        def simulate_batch(self, state, branches, timeout_s=None):
            entered.set()
            release.wait(1)
            return super().simulate_batch(state, branches, timeout_s)

    monkeypatch.setattr(ponder, "ExactShowdownBridge", SlowBridge)
    monkeypatch.setattr(ponder, "strategic_value", lambda _node, _role: 0.0)
    job = BackgroundPonder(
        [PonderRoot(_node(), 1.0, "world")],
        "ours",
        generation=8,
        config=PonderConfig(budget_s=1),
    )
    job.start()
    assert entered.wait(0.5)
    assert job.result_if_done() is None
    job.cancel()
    release.set()
    job.join(1)
    result = job.result_if_done()
    assert result is not None
    assert result.cancelled
