"""Three options of the matrix search added on 2026-10-05, all off by default:

* ``nash_confidence``: a candidate is credited with its edge over the bot's own pair
  minus that many standard errors of the edge across the random streams -- half the
  overrides measured on 2026-10-04 had an edge under two standard errors;
* ``nash_opponent_replacement_leaf``: a turn that leaves only the opponent owing a
  replacement is valued as it stands (0.006 from the resolved value on 1,826
  positions) instead of ranking and simulating their bench;
* ``LiveExactSession(search_replacements=False)``: a forced replacement is the bot's
  own pick -- the search's two values for the same last Pokemon into either slot
  differ with sd 0.21.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from vgc_bench.src.exact_observation import RankedChoice
from vgc_bench.src.exact_planner import (
    ExactDeterminizationPlanner,
    ExactNode,
    PlannerConfig,
    WeightedExactNode,
    _owes_replacement,
    paired_edge,
)
from vgc_bench.src.live_exact import LiveExactSession


def test_paired_edge_is_a_weighted_mean_with_its_standard_error():
    mean, error = paired_edge([(1.0, 0.8), (1.0, -0.4), (1.0, 0.8), (1.0, -0.4)])
    assert mean == pytest.approx(0.2)
    # sample sd 0.6928 over four streams
    assert error == pytest.approx(math.sqrt(4 * 0.36 / 3) / 2)
    steady, none = paired_edge([(0.25, 0.2)] * 4)
    assert steady == pytest.approx(0.2) and none == pytest.approx(0.0)
    # one stream, or all the weight on one: an edge without a standard error
    assert paired_edge([(1.0, 0.3)]) == (0.3, None)
    assert paired_edge([]) == (None, None)
    lopsided_mean, lopsided_error = paired_edge([(1.0, 0.3), (0.0, 9.0)])
    assert lopsided_mean == pytest.approx(0.3) and lopsided_error is None
    # weights are world shares: a heavier world counts for more
    weighted, _ = paired_edge([(0.75, 0.4), (0.25, 0.0)])
    assert weighted == pytest.approx(0.3)


class _StreamBridge:
    """A world whose payoff for (ours, theirs) depends on the random stream."""

    def __init__(self, own_by_stream, alt_by_stream):
        self.own, self.alt = own_by_stream, alt_by_stream
        self.seeds: list[str | None] = []

    def choices(self, state, role):
        return ["own", "alt"] if role == "p1" else ["x", "y"]

    def simulate_batch(self, state, branches, timeout_s=None):
        del state, timeout_s
        out = []
        for branch in branches:
            seed = branch.get("rng_seed")
            if seed not in self.seeds:
                self.seeds.append(seed)
            stream = self.seeds.index(seed)
            values = self.own if branch["p1_choice"] == "own" else self.alt
            out.append(
                {
                    "state": {"score": values[stream], "sides": [{}, {}]},
                    "requests": [{}, {}],
                    "turn": 2,
                    "request_state": "move",
                    "ended": False,
                    "winner": None,
                    "log": [],
                }
            )
        return out


class _EvenPrior:
    def rank(self, _state, _requests, role, choices):
        return [
            RankedChoice(c, (i, i), 1.0 / len(choices)) for i, c in enumerate(choices)
        ]


def _score(node: ExactNode, role: str) -> float:
    del role
    return float(node.state["score"])


def _root() -> ExactNode:
    return ExactNode(
        state={"id": "w", "prng": "1,2,3,4", "sides": [{}, {}]},
        requests=[{}, {}],
        turn=1,
        request_state="move",
    )


def _plan(own, alt, streams=4, **config):
    bridge: Any = _StreamBridge(own, alt)
    prior: Any = _EvenPrior()
    planner = ExactDeterminizationPlanner(
        bridge,
        prior,
        evaluator=_score,
        config=PlannerConfig(
            solution="nash",
            nash_sample=False,
            nash_anchor=0.07,
            time_budget_s=5.0,
            chance_samples=streams,
            volatile_chance_samples=max(2, streams),
            **config,
        ),
    )
    return planner.plan([WeightedExactNode(_root(), 1.0, "w")], include=["own"])


OWN = [0.1, 0.1, 0.1, 0.1]
STEADY = [0.3, 0.3, 0.3, 0.3]  # +0.2 in every stream
LUCKY = [0.9, -0.3, 0.9, -0.3]  # +0.2 on average: +0.8, -0.4, +0.8, -0.4


def test_an_edge_that_holds_in_every_stream_overrides_with_or_without_the_gate():
    for confidence in (0.0, 1.0, 3.0):
        result = _plan(OWN, STEADY, nash_confidence=confidence)
        assert result.choice == "alt"
    row = next(r for r in _plan(OWN, STEADY).rankings if r.choice == "alt")
    assert row.edge == pytest.approx(0.2) and row.edge_se == pytest.approx(0.0)
    own = next(r for r in _plan(OWN, STEADY).rankings if r.choice == "own")
    assert own.edge is None and own.edge_se is None  # nothing to compare it with


def test_an_edge_the_streams_disagree_on_overrides_only_without_the_gate():
    assert _plan(OWN, LUCKY).choice == "alt"  # the mean edge, 0.2: as before
    gated = _plan(OWN, LUCKY, nash_confidence=1.0)
    assert gated.choice == "own"
    row = next(r for r in gated.rankings if r.choice == "alt")
    # the report still carries the mean, the edge and its standard error
    assert row.expected == pytest.approx(0.3)
    assert row.edge == pytest.approx(0.2)
    assert row.edge_se == pytest.approx(math.sqrt(4 * 0.36 / 3) / 2)
    # the gate is a subtraction, not a veto: a small enough multiple still overrides
    assert _plan(OWN, LUCKY, nash_confidence=0.2).choice == "alt"


def test_one_stream_cannot_override_under_the_gate():
    assert _plan(OWN, STEADY, streams=1).choice == "alt"
    gated = _plan(OWN, STEADY, streams=1, nash_confidence=1.0)
    assert gated.choice == "own"
    row = next(r for r in gated.rankings if r.choice == "alt")
    assert row.edge == pytest.approx(0.2) and row.edge_se is None


def test_the_gate_must_not_be_negative():
    with pytest.raises(ValueError, match="nash_confidence"):
        PlannerConfig(nash_confidence=-0.1)


def _node(state, requests, request_state):
    return {
        "state": state,
        "requests": requests,
        "turn": 2,
        "request_state": request_state,
        "ended": False,
        "winner": None,
        "log": [],
    }


class _KoBridge:
    """Our "ko" leaves a foe fainted (they owe a replacement); "trade" leaves us one
    to make."""

    def __init__(self):
        self.replacement_batches = 0

    def choices(self, state, role):
        if state.get("id") == "pending":
            return ["switch 3", "switch 4"] if state["owes"] == role else [""]
        return ["ko", "trade"] if role == "p1" else ["x"]

    def simulate_batch(self, state, branches, timeout_s=None):
        del timeout_s
        if state.get("id") == "pending":
            self.replacement_batches += 1
            return [
                _node({"score": 0.9, "sides": [{}, {}]}, [{}, {}], "move")
                for _ in branches
            ]
        out = []
        for branch in branches:
            owes = "p2" if branch["p1_choice"] == "ko" else "p1"
            wait, forced = {"wait": True}, {"forceSwitch": [True, False]}
            out.append(
                _node(
                    {"id": "pending", "owes": owes, "score": 0.3, "sides": [{}, {}]},
                    [wait, forced] if owes == "p2" else [forced, wait],
                    "switch",
                )
            )
        return out


class _KoPrior:
    def rank(self, _state, _requests, _role, choices):
        return [
            RankedChoice(c, (i, i), 1.0 / len(choices)) for i, c in enumerate(choices)
        ]


def _ko_plan(**config):
    bridge = _KoBridge()
    prior: Any = _KoPrior()
    planner = ExactDeterminizationPlanner(
        bridge,  # type: ignore[arg-type]
        prior,
        evaluator=_score,
        config=PlannerConfig(
            solution="nash", nash_sample=False, time_budget_s=5.0, **config
        ),
    )
    root = ExactNode(
        state={"id": "root", "sides": [{}, {}]},
        requests=[{}, {}],
        turn=1,
        request_state="move",
    )
    result = planner.plan([WeightedExactNode(root, 1.0, "w")])
    return bridge, {row.choice: row.expected for row in result.rankings}


def test_who_owes_a_replacement_is_read_from_the_request():
    wait, forced = {"wait": True}, {"forceSwitch": [False, True]}
    node = ExactNode(state={}, requests=[wait, forced], turn=3, request_state="switch")
    assert not _owes_replacement(node, "p1") and _owes_replacement(node, "p2")
    both = ExactNode(
        state={}, requests=[forced, forced], turn=3, request_state="switch"
    )
    assert _owes_replacement(both, "p1") and _owes_replacement(both, "p2")
    bare = ExactNode(state={}, requests=[None, None], turn=3, request_state="switch")
    assert not _owes_replacement(bare, "p1")


def test_only_the_opponents_pending_replacement_is_valued_as_it_stands():
    bridge, expected = _ko_plan()
    # as before: both kinds of pending replacement are played out (value 0.9)
    assert bridge.replacement_batches == 2
    assert expected == {"ko": pytest.approx(0.9), "trade": pytest.approx(0.9)}
    bridge, expected = _ko_plan(nash_opponent_replacement_leaf=True)
    # their replacement: the position as it stands (0.3), no bench ranked or played;
    # ours: still played out -- which Pokemon we send in is ours to choose
    assert bridge.replacement_batches == 1
    assert expected == {"ko": pytest.approx(0.3), "trade": pytest.approx(0.9)}


def _session(search_replacements: bool) -> Any:
    session: Any = object.__new__(LiveExactSession)
    session.search_replacements = search_replacements
    session.champion_actions = None
    session.skipped_searches = 0
    session.prepare = lambda battle: None
    session._reply_coverage = lambda: None

    def beyond(_battle):
        raise RuntimeError("the search went on")

    session._reuse_contingent_plan = beyond
    return session


def test_a_forced_replacement_is_the_bots_own_when_the_search_is_told_so():
    replacing: Any = SimpleNamespace(force_switch=[True, False])
    moving: Any = SimpleNamespace(force_switch=[False, False])
    session = _session(search_replacements=False)
    assert session.plan(replacing) is None
    assert session.last_schedule["mode"] == "skip_replacement"
    assert session.last_schedule["reasons"] == ["a_forced_replacement_is_the_bots_own"]
    assert session.skipped_searches == 1 and session.last_result is None
    # a move decision is searched as ever
    with pytest.raises(RuntimeError, match="the search went on"):
        session.plan(moving)
    # ... and so is a replacement by default (every run before 2026-10-05)
    with pytest.raises(RuntimeError, match="the search went on"):
        _session(search_replacements=True).plan(replacing)
