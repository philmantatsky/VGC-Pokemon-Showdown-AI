"""Exact Team Preview repair: multi-world aggregation and champion injection.

The 20260823 re-gate showed two structural holes in the original planner: one
sampled hidden world decided every ranking, and the candidate list could omit the
champion policy's own pick entirely. These tests pin the pure pieces of the
repair -- the injection prior, the clean-worlds acceptance rule, and the shared
risk aggregation -- without launching a simulator.
"""

import time
from types import SimpleNamespace

import pytest

from vgc_bench.src import live_preview
from vgc_bench.src.exact_observation import RankedChoice
from vgc_bench.src.exact_planner import (
    ActionScore,
    PlannerConfig,
    PlanResult,
    aggregate_plans,
)
from vgc_bench.src.live_preview import (
    ChampionInjectedPrior,
    LivePreviewPlanner,
    _command,
    _team_choice_key,
)


def _score(choice: str, score: float) -> ActionScore:
    return ActionScore(
        choice=choice,
        actions=None,
        score=score,
        expected=score,
        cvar=score,
        worst=score,
        standard_deviation=0.0,
        prior=0.5,
        opponent_branches=6,
    )


def _plan(rankings: list[ActionScore], truncated: bool = False) -> PlanResult:
    return PlanResult(
        choice=rankings[0].choice,
        actions=None,
        score=rankings[0].score,
        rankings=tuple(rankings),
        nodes=100,
        elapsed_s=1.0,
        completed_depth=1,
        truncated=truncated,
        screened_actions=len(rankings),
        deepened_actions=len(rankings),
        deepened_choices=tuple(row.choice for row in rankings),
    )


class _FakePrior:
    def __init__(self, ranked: list[RankedChoice]):
        self.ranked = ranked

    def rank(self, state, requests, role, choices):
        return list(self.ranked)


PREVIEW_CHOICES = [
    "team 1, 2, 3, 4",
    "team 1, 3, 2, 4",
    "team 2, 4, 1, 3",
    "team 5, 6, 1, 2",
]


class TestTeamChoiceKey:
    def test_formats_normalize_identically(self):
        assert _team_choice_key("team 1, 2, 3, 4") == "team 1,2,3,4"
        assert _team_choice_key("team 1,2,3,4") == "team 1,2,3,4"
        assert _team_choice_key("team 1 2 3 4") == "team 1,2,3,4"


class TestChampionInjectedPrior:
    def test_absent_champion_is_injected_at_front(self):
        base = _FakePrior(
            [
                RankedChoice("team 1, 2, 3, 4", None, 0.5),
                RankedChoice("team 1, 3, 2, 4", None, 0.3),
            ]
        )
        prior = ChampionInjectedPrior(base, "p1", (2, 4, 1, 3), width=2)
        ranked = prior.rank({}, [None, None], "p1", PREVIEW_CHOICES)
        assert ranked[0].choice == "team 2, 4, 1, 3"
        assert ranked[0].probability > 0
        assert len(ranked) == 3
        assert prior.injections == 1 and prior.promotions == 0

    def test_champion_beyond_width_is_promoted_to_front(self):
        base = _FakePrior(
            [
                RankedChoice("team 1, 2, 3, 4", None, 0.5),
                RankedChoice("team 1, 3, 2, 4", None, 0.3),
                RankedChoice("team 2, 4, 1, 3", None, 0.2),
            ]
        )
        prior = ChampionInjectedPrior(base, "p1", (2, 4, 1, 3), width=2)
        ranked = prior.rank({}, [None, None], "p1", PREVIEW_CHOICES)
        assert [item.choice for item in ranked] == [
            "team 2, 4, 1, 3",
            "team 1, 2, 3, 4",
            "team 1, 3, 2, 4",
        ]
        # probability preserved from the predictor's own ranking
        assert ranked[0].probability == 0.2
        assert prior.promotions == 1 and prior.injections == 0

    def test_champion_within_width_leaves_ranking_untouched(self):
        rows = [
            RankedChoice("team 2, 4, 1, 3", None, 0.5),
            RankedChoice("team 1, 2, 3, 4", None, 0.3),
        ]
        prior = ChampionInjectedPrior(_FakePrior(rows), "p1", (2, 4, 1, 3), width=2)
        assert prior.rank({}, [None, None], "p1", PREVIEW_CHOICES) == rows

    def test_opponent_role_passes_through(self):
        rows = [RankedChoice("team 1, 2, 3, 4", None, 1.0)]
        prior = ChampionInjectedPrior(_FakePrior(rows), "p1", (2, 4, 1, 3), width=2)
        assert prior.rank({}, [None, None], "p2", PREVIEW_CHOICES) == rows

    def test_move_choices_pass_through(self):
        rows = [RankedChoice("move earthquake, move protect", None, 1.0)]
        prior = ChampionInjectedPrior(_FakePrior(rows), "p1", (2, 4, 1, 3), width=2)
        choices = ["move earthquake, move protect"]
        assert prior.rank({}, [None, None], "p1", choices) == rows

    def test_illegal_champion_choice_passes_through(self):
        rows = [RankedChoice("team 1, 2, 3, 4", None, 1.0)]
        prior = ChampionInjectedPrior(_FakePrior(rows), "p1", (6, 5, 4, 3), width=2)
        # (6,5,4,3) is not among the legal choices below, so nothing to inject
        assert prior.rank({}, [None, None], "p1", PREVIEW_CHOICES) == rows


class TestAggregatePlans:
    def test_risk_blend_across_two_worlds(self):
        config = PlannerConfig()
        world_a = _plan([_score("team 1,2,3,4", 0.5), _score("team 5,6,1,2", 0.2)])
        world_b = _plan([_score("team 1,2,3,4", 0.1), _score("team 5,6,1,2", 0.4)])
        aggregate = aggregate_plans(
            [(0.6, world_a), (0.2, world_b)], 2.0, config=config
        )
        # masses normalize to 0.75/0.25; X = [0.5@0.75, 0.1@0.25]:
        # expected 0.4, cvar(0.25) 0.1, worst 0.1 -> 0.6*0.4+0.3*0.1+0.1*0.1
        assert aggregate.choice == "team 1,2,3,4"
        assert aggregate.score == pytest.approx(0.28)
        by_choice = {row.choice: row for row in aggregate.rankings}
        assert by_choice["team 5,6,1,2"].score == pytest.approx(0.23)

    def test_choice_missing_from_one_world_is_penalized(self):
        config = PlannerConfig()
        world_a = _plan([_score("team 1,2,3,4", 0.5), _score("team 5,6,1,2", 0.2)])
        world_b = _plan([_score("team 1,2,3,4", 0.1)])
        aggregate = aggregate_plans(
            [(0.6, world_a), (0.2, world_b)], 2.0, config=config
        )
        by_choice = {row.choice: row for row in aggregate.rankings}
        # Y only in world A (mass 0.75): values [0.2@0.75, -1.0@0.25]
        # expected -0.1, cvar -1, worst -1 -> 0.6*-0.1 + 0.3*-1 + 0.1*-1
        assert by_choice["team 5,6,1,2"].score == pytest.approx(-0.46)


def _planner(open_sheet: bool = False) -> LivePreviewPlanner:
    planner = object.__new__(LivePreviewPlanner)
    planner.open_sheet = open_sheet
    return planner


class TestDecide:
    def test_no_clean_world_returns_truncated_stand_down(self):
        decision = _planner()._decide(
            [],
            worlds_requested=3,
            worlds_failed=3,
            started=time.monotonic(),
            nodes=42,
            config=PlannerConfig(),
            champion_key="team 1,2,3,4",
        )
        assert decision.truncated
        assert decision.command == ""
        assert decision.worlds_requested == 3
        assert decision.worlds_clean == 0
        assert decision.worlds_failed == 3
        assert decision.champion_choice == "team 1,2,3,4"

    def test_clean_worlds_aggregate_and_report_champion_margin(self):
        world_a = _plan([_score("team 1,2,3,4", 0.5), _score("team 5,6,1,2", 0.2)])
        world_b = _plan([_score("team 1,2,3,4", 0.1), _score("team 5,6,1,2", 0.4)])
        decision = _planner()._decide(
            [(0.6, world_a), (0.2, world_b)],
            worlds_requested=3,
            worlds_failed=1,
            started=time.monotonic(),
            nodes=200,
            config=PlannerConfig(),
            champion_key="team 5,6,1,2",
        )
        assert not decision.truncated
        assert decision.command == "/team 1234"
        assert decision.worlds_clean == 2 and decision.worlds_failed == 1
        assert decision.champion_rank == 2
        assert decision.override_margin == pytest.approx(0.28 - 0.23)

    def test_champion_chosen_reports_zero_margin(self):
        world = _plan([_score("team 5,6,1,2", 0.4), _score("team 1,2,3,4", 0.1)])
        decision = _planner()._decide(
            [(1.0, world)],
            worlds_requested=1,
            worlds_failed=0,
            started=time.monotonic(),
            nodes=50,
            config=PlannerConfig(),
            champion_key="team 5,6,1,2",
        )
        assert decision.champion_rank == 1
        assert decision.override_margin == pytest.approx(0.0)


class TestConstructorContract:
    @pytest.fixture(autouse=True)
    def _fake_particles(self, monkeypatch):
        monkeypatch.setattr(
            live_preview,
            "ParticleDatabase",
            SimpleNamespace(load=lambda max_particles=12: object()),
        )

    def _build(self, **overrides):
        kwargs: dict = dict(
            policy=None,
            our_team_text="",
            outcome_evaluator=None,
            preview_predictor=None,
            open_sheet=False,
        )
        kwargs.update(overrides)
        return LivePreviewPlanner(**kwargs)

    def test_budget_may_exceed_move_turn_ceiling(self):
        assert self._build(budget_s=24.0).budget_s == 24.0
        with pytest.raises(ValueError):
            self._build(budget_s=61.0)
        with pytest.raises(ValueError):
            self._build(budget_s=0.05)

    def test_determinizations_bounds(self):
        assert self._build(determinizations=3).determinizations == 3
        with pytest.raises(ValueError):
            self._build(determinizations=0)
        with pytest.raises(ValueError):
            self._build(determinizations=9)

    def test_open_sheet_forces_single_world(self):
        planner = self._build(open_sheet=True, determinizations=4)
        assert planner.determinizations == 1

    def test_champion_order_validation(self):
        planner = self._build(champion_order=(2, 4, 1, 3))
        assert planner.champion_order == (2, 4, 1, 3)
        for bad in ((1, 2, 3), (1, 2, 3, 3), (0, 2, 3, 4), (1, 2, 3, 7)):
            with pytest.raises(ValueError):
                self._build(champion_order=bad)


class TestCommand:
    def test_round_trip(self):
        assert _command("team 2,4,1,3") == "/team 2413"
        assert _command("team 2, 4, 1, 3") == "/team 2413"
        with pytest.raises(ValueError):
            _command("team 1,2,3")
        with pytest.raises(ValueError):
            _command("team 1,2,3,3")
        with pytest.raises(ValueError):
            _command("team 0,2,3,7")
