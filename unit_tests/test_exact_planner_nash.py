"""The nash solution mode of the exact planner (2026-10-04, the user: "start the matrix
search"): one turn per world as a zero-sum matrix game of our candidates x the
opponent's likely replies, equilibrium strategies averaged over worlds and sampled --
the recipe of the two bots that topped Reg M-C (RESEARCH_TOP_BOTS.md)."""

from __future__ import annotations

import collections
from typing import Any

import pytest

from vgc_bench.src.exact_observation import RankedChoice
from vgc_bench.src.exact_planner import (
    ExactDeterminizationPlanner,
    ExactMultiTurnPlanner,
    ExactNode,
    PlannerConfig,
    WeightedExactNode,
)

PENNIES = {("a", "x"): 1.0, ("a", "y"): -1.0, ("b", "x"): -1.0, ("b", "y"): 1.0}
A_DOMINATES = {("a", "x"): 0.6, ("a", "y"): 0.4, ("b", "x"): 0.1, ("b", "y"): -0.2}
B_DOMINATES = {("a", "x"): -0.2, ("a", "y"): 0.1, ("b", "x"): 0.4, ("b", "y"): 0.6}


class _TableBridge:
    """Children score the payoff of the (p1, p2) pair in the world's table."""

    def __init__(self, tables):
        self.tables = tables

    def choices(self, state, role):
        return ["a", "b"] if role == "p1" else ["x", "y"]

    def simulate_batch(self, state, branches, timeout_s=None):
        del timeout_s
        table = self.tables[state["id"]]
        return [
            {
                "state": {
                    "score": table[(branch["p1_choice"], branch["p2_choice"])],
                    "sides": [{}, {}],
                },
                "requests": [{}, {}],
                "turn": 2,
                "request_state": "move",
                "ended": False,
                "winner": None,
                "log": [],
            }
            for branch in branches
        ]


class _Prior:
    def __init__(self, ours=None):
        self.ours = ours or {"a": 0.6, "b": 0.4}

    def rank(self, _state, _requests, role, choices):
        probabilities = self.ours if role == "p1" else {"x": 0.5, "y": 0.5}
        return [
            RankedChoice(choice, (i, i), probabilities[choice])
            for i, choice in enumerate(choices)
        ]


def _score(node: ExactNode, role: str) -> float:
    del role
    return float(node.state["score"])


def _root(world: str) -> ExactNode:
    return ExactNode(
        state={"id": world, "sides": [{}, {}]},
        requests=[{}, {}],
        turn=1,
        request_state="move",
    )


def _planner(tables, prior=None, **config):
    bridge: Any = _TableBridge(tables)
    ranker: Any = prior or _Prior()
    return ExactMultiTurnPlanner(
        bridge,
        ranker,
        evaluator=_score,
        config=PlannerConfig(solution="nash", nash_sample=False, **config),
    )


def test_matching_pennies_is_an_even_mix_and_fully_searched():
    result = _planner({"w": PENNIES}).plan(_root("w"))
    weights = {row.choice: row.score for row in result.rankings}
    assert weights["a"] == pytest.approx(0.5, abs=0.03)
    assert weights["b"] == pytest.approx(0.5, abs=0.03)
    assert set(result.deepened_choices) == {"a", "b"}
    assert result.completed_depth == 1


def test_a_dominant_action_takes_the_whole_strategy():
    result = _planner({"w": A_DOMINATES}).plan(_root("w"))
    assert result.choice == "a"
    assert result.rankings[0].score > 0.99
    assert result.rankings[0].expected == pytest.approx(0.4, abs=0.02)


def test_candidates_below_the_prior_ratio_stay_out_of_the_table():
    planner = _planner({"w": PENNIES}, prior=_Prior({"a": 0.95, "b": 0.05}))
    result = planner.plan(_root("w"))
    assert [row.choice for row in result.rankings] == ["a"]


def _worlds(**config):
    bridge: Any = _TableBridge({"w1": A_DOMINATES, "w2": B_DOMINATES})
    ranker: Any = _Prior()
    planner = ExactDeterminizationPlanner(
        bridge,
        ranker,
        evaluator=_score,
        config=PlannerConfig(solution="nash", time_budget_s=5.0, **config),
    )
    roots = [
        WeightedExactNode(_root("w1"), 0.75, "w1"),
        WeightedExactNode(_root("w2"), 0.25, "w2"),
    ]
    return planner.plan(roots)


def test_strategies_are_averaged_over_worlds_by_their_probability():
    result = _worlds(nash_sample=False)
    weights = {row.choice: row.score for row in result.rankings}
    assert weights["a"] == pytest.approx(0.75, abs=0.02)
    assert weights["b"] == pytest.approx(0.25, abs=0.02)
    assert result.choice == "a"
    assert all(row.depth_coverage == pytest.approx(1.0) for row in result.rankings)
    assert result.selected_depth_coverage == pytest.approx(1.0)


def test_sampling_follows_the_averaged_strategy():
    picks = collections.Counter(
        _worlds(nash_sample=True, nash_seed=seed).choice for seed in range(400)
    )
    assert picks["a"] + picks["b"] == 400
    assert 0.65 < picks["a"] / 400 < 0.85  # the averaged weight of "a" is 0.75
    first = _worlds(nash_sample=True, nash_seed=7)
    assert first.choice == _worlds(nash_sample=True, nash_seed=7).choice


def test_unknown_solution_is_refused():
    with pytest.raises(ValueError, match="solution"):
        PlannerConfig(solution="minimax")


def test_prior_mix_pulls_the_played_strategy_toward_the_policy():
    pure = {row.choice: row.score for row in _worlds(nash_sample=False).rankings}
    mixed = {
        row.choice: row.score
        for row in _worlds(nash_sample=False, nash_prior_mix=0.5).rankings
    }
    # policy prior a 0.6 / b 0.4; equilibrium average a 0.75 / b 0.25
    assert mixed["a"] == pytest.approx(0.5 * pure["a"] + 0.5 * 0.6, abs=0.02)
    assert mixed["b"] == pytest.approx(0.5 * pure["b"] + 0.5 * 0.4, abs=0.02)
    with pytest.raises(ValueError, match="prior_mix"):
        PlannerConfig(nash_prior_mix=1.5)


def test_hidden_world_sets_without_a_spread_get_a_real_one():
    """2026-10-04: Reg M-C set data has no spreads; hidden-world opponents were built
    with zero stat points (the search lost ~70% of hidden-sheet games)."""
    from vgc_bench.src.set_particles import (
        SetParticle,
        TeamSlot,
        default_spread,
        determination_team_text,
    )

    torkoal = ("eruption", "earthpower", "heatwave", "protect")
    assert default_spread("torkoal", torkoal) == "Serious:32/0/0/32/0/2"
    fake_out = ("fakeout", "flareblitz", "partingshot", "throatchop")
    assert default_spread("incineroar", fake_out) == "Serious:32/32/0/0/0/2"
    # no damaging move: the higher base attacking stat decides (Farigiraf: SpA)
    assert default_spread("farigiraf", ("trickroom", "helpinghand")).endswith(
        "0/32/0/2"
    )
    particle = SetParticle("torkoal", "drought", "charcoal", torkoal, None, 1.0, "t")
    text = determination_team_text(
        [TeamSlot("torkoal", "Torkoal")], {"torkoal": particle}
    )
    assert "EVs: 32 HP / 32 SpA / 2 Spe" in text
    known = SetParticle(
        "torkoal", "drought", "charcoal", torkoal, "Quiet:32/0/2/32/0/0", 1.0, "t"
    )
    text = determination_team_text([TeamSlot("torkoal", "Torkoal")], {"torkoal": known})
    assert "EVs: 32 HP / 2 Def / 32 SpA" in text and "Quiet Nature" in text
