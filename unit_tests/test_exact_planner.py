from __future__ import annotations

import copy
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from poke_env.battle import Move
from poke_env.data import to_id_str

from vgc_bench.src.exact_observation import (
    ExactPolicyAdapter,
    RankedChoice,
    choice_to_actions,
    opponent_choice_likelihood,
    state_to_battle,
)
from vgc_bench.src.exact_planner import (
    ActionScore,
    ExactDeterminizationPlanner,
    ExactMultiTurnPlanner,
    ExactNode,
    HybridEvaluator,
    PlannerConfig,
    PlanResult,
    WeightedExactNode,
    _choice_has_volatile_accuracy,
    _diverse_prefix,
    _lost_unexecuted_move_slots,
)
from vgc_bench.src.exact_sim import ExactShowdownBridge, ExactSimulatorError
from vgc_bench.src.opponent_tactics import MovePrediction, SwitchPrediction

ROOT = Path(__file__).resolve().parents[1]


def test_exact_adapters_serialize_shared_stateful_policy_inference():
    class Policy:
        device = torch.device("cpu")

        def __init__(self):
            self.guard = threading.Lock()
            self.active = 0
            self.maximum = 0

        def get_logits(self, _obs, actor_grad=False):
            del actor_grad
            with self.guard:
                self.active += 1
                self.maximum = max(self.maximum, self.active)
            time.sleep(0.03)
            with self.guard:
                self.active -= 1
            return torch.zeros((1, 2)), torch.tensor([[0.25]])

    policy = Policy()
    inference_lock = threading.RLock()
    adapters = [
        ExactPolicyAdapter(policy, inference_lock=inference_lock)
        for _ in range(2)
    ]
    inputs = ({}, np.zeros(1), np.ones(2), {"observation": torch.zeros((1, 1))})
    for adapter in adapters:
        adapter._inputs = lambda *_args: inputs
    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(lambda adapter: adapter.value({}, [None, None]), adapters))
    assert values == [0.25, 0.25]
    assert policy.maximum == 1


def _without_timestamps(state):
    normalized = copy.deepcopy(state)
    normalized["log"] = [
        line for line in normalized.get("log", []) if not line.startswith("|t:|")
    ]
    return normalized


def test_choice_to_actions_preserves_targets_and_mega():
    request = {
        "active": [
            {"moves": [{"id": "earthquake"}, {"id": "dragonclaw"}]},
            {"moves": [{"id": "protect"}, {"id": "heatwave"}]},
        ]
    }
    assert choice_to_actions("move earthquake, move heatwave mega", request) == (9, 34)


def test_choice_to_actions_uses_stable_known_move_slot_for_locked_move():
    request = {
        "active": [
            {"moves": [{"id": "lightscreen"}]},
            {"moves": [{"id": "electroshot"}]},
        ]
    }
    battle = NS(
        active_pokemon=[
            NS(moves={"lightscreen": Move("lightscreen", 9)}),
            NS(
                moves={
                    move: Move(move, 9)
                    for move in ("flashcannon", "dragonpulse", "electroshot", "protect")
                }
            ),
        ],
        available_moves=[
            [Move("lightscreen", 9)],
            [Move("electroshot", 9)],
        ],
    )
    assert choice_to_actions(
        "move lightscreen, move electroshot +1", request, battle=battle
    ) == (9, 20)


def test_choice_to_actions_normalizes_forced_recharge_target():
    request = {
        "active": [
            {"moves": [{"id": "moonblast"}]},
            {"moves": [{"id": "recharge"}]},
        ]
    }
    battle = NS(
        active_pokemon=[
            NS(moves={"moonblast": Move("moonblast", 9)}),
            NS(moves={"hyperbeam": Move("hyperbeam", 9)}),
        ],
        available_moves=[[Move("moonblast", 9)], [Move("recharge", 9)]],
    )
    assert choice_to_actions(
        "move moonblast -2, move recharge +1", request, battle=battle
    ) == (7, 9)


class _UniformPolicy:
    device = torch.device("cpu")

    def get_logits(self, obs_dict, actor_grad=False):
        batch = obs_dict["action_mask"].shape[0]
        return torch.zeros((batch, 220)), torch.zeros((batch, 1))

    def _update_mask(self, mask, _first):
        return mask

    def get_dist_from_logits(self, logits, _mask, _first=None):
        batch = logits.shape[0]
        probs = torch.full((batch, 110), 1.0 / 110.0)
        distribution = [NS(probs=probs), NS(probs=probs)]
        return NS(distribution=distribution)


def test_policy_rank_accepts_showdown_forced_pass_with_false_policy_mask():
    adapter = ExactPolicyAdapter(_UniformPolicy())
    move = Move("protect", 9)
    battle = NS(
        active_pokemon=[None, NS(moves={"protect": move})],
        available_moves=[[], [move]],
    )
    mask = np.zeros(220, dtype=np.float32)
    mask[1] = 1.0  # surrogate used only to condition the second action head
    mask[110 + 9] = 1.0
    obs_dict = {
        "observation": torch.zeros((1, 1)),
        "action_mask": torch.as_tensor(mask).unsqueeze(0),
    }
    adapter._inputs = lambda *_args: (battle, np.zeros(1), mask, obs_dict)
    request = {
        "active": [None, {"moves": [{"id": "protect"}]}],
    }
    ranked = adapter.rank(
        {"sides": [{"pokemon": []}, {"pokemon": []}]},
        [request, request],
        "p1",
        ["pass, move protect"],
    )
    assert ranked[0].actions == (0, 9)
    assert ranked[0].probability == pytest.approx(1.0)


def test_policy_rank_restores_showdown_omitted_leading_pass():
    adapter = ExactPolicyAdapter(_UniformPolicy())
    move = Move("moonblast", 9)
    battle = NS(
        active_pokemon=[None, NS(moves={"moonblast": move}, fainted=False)],
        available_moves=[[], [move]],
    )
    mask = np.zeros(220, dtype=np.float32)
    mask[1] = 1.0
    mask[110 + 9] = 1.0
    obs_dict = {
        "observation": torch.zeros((1, 1)),
        "action_mask": torch.as_tensor(mask).unsqueeze(0),
    }
    adapter._inputs = lambda *_args: (battle, np.zeros(1), mask, obs_dict)
    request = {"active": [None, {"moves": [{"id": "moonblast"}]}]}

    ranked = adapter.rank(
        {"sides": [{"pokemon": []}, {"pokemon": []}]},
        [request, request],
        "p1",
        ["move moonblast"],
    )

    assert ranked[0].actions == (0, 9)


def test_policy_rank_omits_showdown_target_variant_for_empty_ally_slot():
    adapter = ExactPolicyAdapter(_UniformPolicy())
    move = Move("weatherball", 9)
    battle = NS(
        active_pokemon=[None, NS(moves={"weatherball": move}, fainted=False)],
        opponent_active_pokemon=[NS(fainted=False), None],
        available_moves=[[], [move]],
    )
    mask = np.zeros(220, dtype=np.float32)
    mask[1] = 1.0
    mask[110 + 10] = 1.0
    obs_dict = {
        "observation": torch.zeros((1, 1)),
        "action_mask": torch.as_tensor(mask).unsqueeze(0),
    }
    adapter._inputs = lambda *_args: (battle, np.zeros(1), mask, obs_dict)
    request = {"active": [None, {"moves": [{"id": "weatherball"}]}]}

    ranked = adapter.rank(
        {"sides": [{"pokemon": []}, {"pokemon": []}]},
        [request, request],
        "p1",
        ["pass, move weatherball -1", "pass, move weatherball +1"],
    )

    assert [row.choice for row in ranked] == ["pass, move weatherball +1"]
    assert ranked[0].actions == (0, 10)


def test_switch_action_follows_identity_after_showdown_reorders_party():
    with ExactShowdownBridge() as bridge:
        root = bridge.create(
            formatid="gen9championsvgc2026regmb",
            seed=[101, 102, 103, 104],
            p1_team_text=(ROOT / "teams/reg_mb/our_team.txt").read_text(),
            p2_team_text=(ROOT / "teams/reg_mb/MB11.txt").read_text(),
            p1_preview="team 1234",
            p2_preview="team 1234",
        )
        switched = bridge.simulate(
            root["state"],
            "switch 3, move protect mega",
            "move protect, move protect",
            rng_seed="11,12,13,14",
        )
        choices = [
            choice for choice in bridge.choices(switched["state"], "p1")
            if "switch" in choice
        ]
    node = ExactNode.from_result(switched)
    battle = state_to_battle(node.state, node.requests, "p1", True)
    # The target assertion below is the important invariant: each exact switch maps
    # back to the same Pokemon in poke-env even after Showdown moved Whimsicott active.
    for choice in choices[:20]:
        actions = choice_to_actions(
            choice,
            node.requests[0],
            state=node.state,
            role="p1",
            battle=battle,
        )
        atoms = [atom.strip() for atom in choice.split(",")]
        for slot, atom in enumerate(atoms):
            if not atom.startswith("switch "):
                continue
            exact_index = int(atom.split()[1]) - 1
            exact_species = node.state["sides"][0]["pokemon"][exact_index]["set"][
                "species"
            ]
            mapped = list(battle.team.values())[actions[slot] - 1]
            assert to_id_str(mapped.species) == to_id_str(exact_species)


def test_imposter_request_preserves_transformed_move_vocabulary():
    with ExactShowdownBridge() as bridge:
        root = bridge.create(
            formatid="gen9championsvgc2026regmb",
            seed=[201, 202, 203, 204],
            p1_team_text=(ROOT / "teams/reg_mb/our_team.txt").read_text(),
            p2_team_text=(ROOT / "teams/reg_mb/MB571.txt").read_text(),
            p1_preview="team 1234",
            p2_preview="team 1234",
        )
        choices = bridge.choices(root["state"], "p2")
    battle = state_to_battle(root["state"], root["requests"], "p2", True)
    assert "transform" not in battle.active_pokemon[0].moves
    request = root["requests"][1]
    assert request is not None
    for choice in choices:
        choice_to_actions(
            choice,
            request,
            state=root["state"],
            role="p2",
            battle=battle,
        )


def test_diverse_beam_guarantees_every_move_family_even_beyond_soft_width():
    ranked = [
        RankedChoice("move dragonclaw +1, move protect", (1, 1), 0.60),
        RankedChoice("move dragonclaw +1, move heatwave", (1, 1), 0.50),
        RankedChoice("move earthquake, move protect", (1, 1), 0.40),
        RankedChoice("move earthquake, move heatwave", (1, 1), 0.30),
        RankedChoice("move rocktomb +1, move solarbeam +1", (1, 1), 0.02),
        RankedChoice("move protect, move weatherball +1", (1, 1), 0.01),
    ]
    selected = _diverse_prefix(ranked, 3)
    slot_families = [set(), set()]
    for item in selected:
        for slot, atom in enumerate(item.choice.split(",")):
            slot_families[slot].add(atom.strip().split()[1])
    assert slot_families[0] == {"dragonclaw", "earthquake", "rocktomb", "protect"}
    assert slot_families[1] == {"protect", "heatwave", "solarbeam", "weatherball"}
    assert len(selected) == 4  # hard coverage expands past the nominal width of three


def test_diverse_beam_does_not_cover_earthquake_only_with_a_bad_partner():
    ranked = [
        RankedChoice("move rocktomb +1, move heatwave", (1, 1), 0.90),
        RankedChoice("move earthquake, move heatwave", (1, 1), 0.80),
        RankedChoice("move protect, move weatherball +1", (1, 1), 0.70),
        RankedChoice("move earthquake, move weatherball +1", (1, 1), 0.10),
    ]
    selected = _diverse_prefix(ranked, 2)
    choices = {item.choice for item in selected}
    assert "move earthquake, move heatwave" in choices
    assert "move earthquake, move weatherball +1" not in choices


def test_move_accuracy_marks_only_volatile_moves():
    assert _choice_has_volatile_accuracy("move zapcannon +1, move protect", 0.90)
    assert not _choice_has_volatile_accuracy(
        "move earthquake, move protect", 0.90
    )


def test_lost_unexecuted_move_slots_uses_event_order():
    log = [
        "|move|p2a: Raichu|Sucker Punch|p1a: Basculegion",
        "|faint|p1a: Basculegion",
        "|move|p1b: Whimsicott|Tailwind|p1: Whimsicott",
        "|faint|p1b: Whimsicott",
    ]
    assert _lost_unexecuted_move_slots(
        log, "p1", "move wavecrash +1, move tailwind"
    ) == (0,)


class _Prior:
    def rank(self, _state, _requests, role, choices):
        probabilities = {
            "p1": {"greedy": 0.8, "setup": 0.2, "finish": 1.0},
            "p2": {"reply": 1.0},
        }[role]
        return [
            RankedChoice(choice, (index, index), probabilities.get(choice, 1.0))
            for index, choice in enumerate(choices)
        ]


class _TreeBridge:
    def choices(self, state, role):
        if role == "p2":
            return ["reply"]
        return {
            "root": ["greedy", "setup"],
            "greedy": ["finish"],
            "setup": ["finish"],
        }.get(state["id"], [])

    def simulate_batch(self, state, branches):
        results = []
        for branch in branches:
            choice = branch["p1_choice"]
            if state["id"] == "root":
                child = choice
                score = 0.6 if choice == "greedy" else 0.2
                turn = 2
            elif state["id"] == "greedy":
                child, score, turn = "greedy_leaf", -0.9, 3
            else:
                child, score, turn = "setup_leaf", 0.7, 3
            results.append(
                {
                    "state": {"id": child, "score": score, "sides": [{}, {}]},
                    "requests": [{}, {}],
                    "turn": turn,
                    "request_state": "move",
                    "ended": False,
                    "winner": None,
                }
            )
        return results


class _ChanceRecordingBridge:
    def __init__(self, our_choice):
        self.our_choice = our_choice
        self.branches = []

    def choices(self, _state, role):
        return [self.our_choice] if role == "p1" else ["move protect"]

    def simulate_batch(self, _state, branches, timeout_s=None):
        del timeout_s
        self.branches.extend(branches)
        return [
            {
                "state": {"score": 0.0, "sides": [{}, {}]},
                "requests": [{}, {}],
                "turn": 2,
                "request_state": "move",
                "ended": False,
                "winner": None,
                "log": [],
            }
            for _branch in branches
        ]


@pytest.mark.parametrize(
    ("choice", "expected_samples"),
    [
        ("move zapcannon +1", 2),
        ("move earthquake", 1),
    ],
)
def test_deep_scoring_adapts_chance_samples_to_move_accuracy(
    choice, expected_samples
):
    bridge = _ChanceRecordingBridge(choice)
    root = ExactNode(
        state={"score": 0.0, "sides": [{}, {}]},
        requests=[{}, {}],
        turn=1,
        request_state="move",
    )
    planner = ExactMultiTurnPlanner(
        bridge,
        evaluator=lambda node, _role: float(node.state["score"]),
        config=PlannerConfig(
            depth=1,
            root_width=1,
            opponent_width=1,
            expected_weight=1.0,
            cvar_weight=0.0,
            worst_weight=0.0,
            time_budget_s=5,
            chance_samples=1,
            volatile_chance_samples=2,
        ),
    )
    planner.plan(root)
    assert len(bridge.branches) == expected_samples
    assert sum("rng_seed" in branch for branch in bridge.branches) == (
        expected_samples if expected_samples > 1 else 0
    )


class _PreMoveKoBridge:
    def choices(self, _state, role):
        if role == "p2":
            return ["move suckerpunch +1"]
        return [
            "move dragonclaw +1, move heatwave",
            "move protect, move heatwave",
        ]

    def simulate_batch(self, _state, branches, timeout_s=None):
        del timeout_s
        results = []
        for branch in branches:
            risky = branch["p1_choice"].startswith("move dragonclaw")
            log = (
                [
                    "|move|p2a: Kingambit|Sucker Punch|p1a: Garchomp",
                    "|faint|p1a: Garchomp",
                    "|move|p1b: Charizard|Heat Wave|p2a: Kingambit",
                ]
                if risky
                else [
                    "|move|p1a: Garchomp|Protect|p1a: Garchomp",
                    "|move|p1b: Charizard|Heat Wave|p2a: Kingambit",
                ]
            )
            results.append(
                {
                    "state": {"score": 0.2, "sides": [{}, {}]},
                    "requests": [{}, {}],
                    "turn": 2,
                    "request_state": "move",
                    "ended": False,
                    "winner": None,
                    "log": log,
                }
            )
        return results


def test_planner_penalizes_action_lost_to_a_pre_move_ko():
    root = ExactNode(
        state={"score": 0.0, "sides": [{}, {}]},
        requests=[{}, {}],
        turn=1,
        request_state="move",
    )
    planner = ExactMultiTurnPlanner(
        _PreMoveKoBridge(),
        evaluator=lambda node, _role: float(node.state["score"]),
        config=PlannerConfig(
            depth=1,
            root_width=2,
            opponent_width=1,
            expected_weight=1.0,
            cvar_weight=0.0,
            worst_weight=0.0,
            time_budget_s=5,
            pre_move_ko_penalty=0.22,
        ),
    )
    result = planner.plan(root)
    assert result.choice == "move protect, move heatwave"
    assert result.rankings[0].score > result.rankings[1].score


def test_two_turn_planner_prefers_position_with_better_future():
    root = ExactNode(
        state={"id": "root", "score": 0.0, "sides": [{}, {}]},
        requests=[{}, {}],
        turn=1,
        request_state="move",
    )

    def evaluator(node, _role):
        return float(node.state["score"])

    planner = ExactMultiTurnPlanner(
        _TreeBridge(),
        prior=_Prior(),
        evaluator=evaluator,
        config=PlannerConfig(
            depth=2,
            root_width=2,
            opponent_width=1,
            continuation_width=1,
            expected_weight=1.0,
            cvar_weight=0.0,
            worst_weight=0.0,
            time_budget_s=5,
        ),
    )
    result = planner.plan(root)
    assert result.choice == "setup"
    assert result.rankings[0].score > result.rankings[1].score


def test_two_turn_planner_preserves_next_action_as_contingent_plan():
    root = ExactNode(
        state={"id": "root", "score": 0.0, "sides": [{}, {}]},
        requests=[{}, {}],
        turn=1,
        request_state="move",
    )
    planner = ExactMultiTurnPlanner(
        _TreeBridge(),
        prior=_Prior(),
        evaluator=lambda node, _role: float(node.state["score"]),
        config=PlannerConfig(
            depth=2,
            root_width=2,
            opponent_width=1,
            continuation_width=1,
            expected_weight=1.0,
            cvar_weight=0.0,
            worst_weight=0.0,
            time_budget_s=5,
        ),
    )
    result = planner.plan(root)
    assert result.choice == "setup"
    assert planner.continuations
    assert {row.root_choice for row in planner.continuations} == {"greedy", "setup"}
    assert {row.next_choice for row in planner.continuations} == {"finish"}
    assert planner.outcomes
    assert {row.root_choice for row in planner.outcomes} == {"greedy", "setup"}


class _ScreenTimeoutPlanner(ExactMultiTurnPlanner):
    def _score_actions(self, *_args, **_kwargs):
        time.sleep(0.03)
        return []


def test_anytime_screen_timeout_returns_complete_prior_fallback():
    root = ExactNode(
        state={"id": "root", "score": 0.0, "sides": [{}, {}]},
        requests=[{}, {}],
        turn=1,
        request_state="move",
    )
    planner = _ScreenTimeoutPlanner(
        _TreeBridge(),
        prior=_Prior(),
        evaluator=lambda node, _role: float(node.state["score"]),
        config=PlannerConfig(
            depth=2,
            root_width=2,
            opponent_width=1,
            continuation_width=1,
            time_budget_s=0.05,
            screen_budget_s=0.01,
            anytime=True,
        ),
    )
    result = planner.plan(root)
    assert {row.choice for row in result.rankings} == {"greedy", "setup"}
    assert result.choice == "greedy"
    assert result.truncated
    assert result.fallback_reason == "screen_timeout"


def test_determinization_planner_survives_one_invalid_root():
    good = ExactNode(
        state={"id": "root", "score": 0.0, "sides": [{}, {}]},
        requests=[{}, {}],
        turn=1,
        request_state="move",
    )
    bad = ExactNode(
        state={"id": "bad", "score": 0.0, "sides": [{}, {}]},
        requests=[{}, {}],
        turn=1,
        request_state="move",
    )
    planner = ExactDeterminizationPlanner(
        _TreeBridge(),
        prior=_Prior(),
        evaluator=lambda node, _role: float(node.state["score"]),
        config=PlannerConfig(
            depth=2,
            root_width=2,
            opponent_width=1,
            continuation_width=1,
            time_budget_s=1.0,
            screen_budget_s=0.2,
            anytime=True,
        ),
    )
    result = planner.plan(
        [
            WeightedExactNode(bad, 0.5, "bad"),
            WeightedExactNode(good, 0.5, "good"),
        ]
    )
    assert result.rankings
    assert result.truncated
    assert result.fallback_reason == "partial_root_failure"


def test_aggregation_never_selects_choice_illegal_in_sampled_real_root():
    def plan_result(choice, score):
        row = ActionScore(
            choice=choice,
            actions=(1, 1),
            score=score,
            expected=score,
            cvar=score,
            worst=score,
            standard_deviation=0.0,
            prior=1.0,
            opponent_branches=1,
        )
        return PlanResult(
            choice=choice,
            actions=(1, 1),
            score=score,
            rankings=(row,),
            nodes=1,
            elapsed_s=0.01,
            completed_depth=2,
            truncated=False,
        )

    planner = ExactDeterminizationPlanner(_TreeBridge())
    result = planner._aggregate(
        [(0.5, plan_result("safe", 0.1)), (0.5, plan_result("fantasy", 1.0))],
        0.02,
        required_choices={"safe"},
    )
    assert result.choice == "safe"


def test_continuation_timeout_falls_back_to_leaf_instead_of_crashing():
    root = ExactNode(
        state={"id": "root", "score": 0.25, "sides": [{}, {}]},
        requests=[{}, {}],
        turn=1,
        request_state="move",
    )
    planner = ExactMultiTurnPlanner(
        _TreeBridge(),
        prior=_Prior(),
        evaluator=lambda node, _role: float(node.state["score"]),
        config=PlannerConfig(
            depth=2,
            root_width=2,
            opponent_width=1,
            continuation_width=1,
            time_budget_s=5,
        ),
    )
    planner._deadline = 0.0
    planner._nodes = 0
    planner._truncated = False
    assert planner._search(root, "p1", 2) == pytest.approx(0.25)
    assert planner._truncated


class _RecordingContinuationBridge:
    def __init__(self):
        self.batch_sizes = []

    def choices(self, state, role):
        if state["id"] == "root":
            return ["start"] if role == "p1" else ["reply"]
        return ["a", "b", "c", "d"] if role == "p1" else ["x", "y", "z"]

    def simulate_batch(self, state, branches):
        self.batch_sizes.append(len(branches))
        child = "branch" if state["id"] == "root" else "leaf"
        turn = 2 if state["id"] == "root" else 3
        return [
            {
                "state": {
                    "id": child,
                    "score": 0.1,
                    "sides": [{}, {}],
                },
                "requests": [{}, {}],
                "turn": turn,
                "request_state": "move",
                "ended": False,
                "winner": None,
            }
            for _branch in branches
        ]


def test_second_turn_search_branches_across_several_continuations():
    bridge = _RecordingContinuationBridge()
    root = ExactNode(
        state={"id": "root", "score": 0.0, "sides": [{}, {}]},
        requests=[{}, {}],
        turn=1,
        request_state="move",
    )
    planner = ExactMultiTurnPlanner(
        bridge,
        evaluator=lambda node, _role: float(node.state["score"]),
        config=PlannerConfig(
            depth=2,
            root_width=1,
            opponent_width=1,
            continuation_width=3,
            expected_weight=1.0,
            cvar_weight=0.0,
            worst_weight=0.0,
            time_budget_s=5,
        ),
    )
    planner.plan(root)
    assert bridge.batch_sizes == [1, 9]


class _ForcedReplacementBridge:
    def __init__(self):
        self.visited = []

    def choices(self, state, role):
        self.visited.append(("choices", state["id"], role))
        return ["only"]

    def simulate_batch(self, state, branches):
        self.visited.append(("simulate", state["id"], len(branches)))
        if state["id"] == "root":
            child, turn, request_state = "replace", 1, "switch"
        elif state["id"] == "replace":
            child, turn, request_state = "next_turn", 2, "move"
        else:
            child, turn, request_state = "too_deep", 3, "move"
        return [
            {
                "state": {
                    "id": child,
                    "score": 0.2,
                    "sides": [{}, {}],
                },
                "requests": [{}, {}],
                "turn": turn,
                "request_state": request_state,
                "ended": False,
                "winner": None,
            }
            for _branch in branches
        ]


def test_ko_counts_as_a_turn_before_same_turn_forced_replacement():
    bridge = _ForcedReplacementBridge()
    root = ExactNode(
        state={"id": "root", "score": 0.0, "sides": [{}, {}]},
        requests=[{}, {}],
        turn=1,
        request_state="move",
    )
    planner = ExactMultiTurnPlanner(
        bridge,
        evaluator=lambda node, _role: float(node.state["score"]),
        config=PlannerConfig(
            depth=1,
            root_width=1,
            opponent_width=1,
            continuation_width=1,
            replacement_width=1,
            expected_weight=1.0,
            cvar_weight=0.0,
            worst_weight=0.0,
            time_budget_s=5,
        ),
    )
    planner.plan(root)
    simulated_states = [
        event[1] for event in bridge.visited if event[0] == "simulate"
    ]
    assert simulated_states == ["root", "replace"]


def test_exact_batch_branches_are_isolated_and_order_invariant():
    with ExactShowdownBridge() as bridge:
        root = bridge.create(
            formatid="gen9championsvgc2026regmb",
            seed=[1, 2, 3, 4],
            p1_team_text=(ROOT / "teams/reg_mb/our_team.txt").read_text(),
            p2_team_text=(ROOT / "teams/reg_mb/MB11.txt").read_text(),
            p1_preview="team 1234",
            p2_preview="team 1234",
        )
        p1 = bridge.choices(root["state"], "p1")[:2]
        p2 = bridge.choices(root["state"], "p2")[:2]
        branches = [
            {"p1_choice": p1[0], "p2_choice": p2[0]},
            {"p1_choice": p1[1], "p2_choice": p2[1]},
        ]
        forward = bridge.simulate_batch(root["state"], branches)
        reverse = bridge.simulate_batch(root["state"], list(reversed(branches)))
        individual = [
            bridge.simulate(root["state"], branch["p1_choice"], branch["p2_choice"])
            for branch in branches
        ]
    assert (
        _without_timestamps(forward[0]["state"])
        == _without_timestamps(individual[0]["state"])
        == _without_timestamps(reverse[1]["state"])
    )
    assert (
        _without_timestamps(forward[1]["state"])
        == _without_timestamps(individual[1]["state"])
        == _without_timestamps(reverse[0]["state"])
    )


def test_hidden_exact_observation_does_not_leak_opponent_sets():
    with ExactShowdownBridge() as bridge:
        root = bridge.create(
            formatid="gen9championsvgc2026regmb",
            seed=[5, 6, 7, 8],
            p1_team_text=(ROOT / "teams/reg_mb/our_team.txt").read_text(),
            p2_team_text=(ROOT / "teams/reg_mb/MB11.txt").read_text(),
            p1_preview="team 1234",
            p2_preview="team 1234",
        )
    open_view = state_to_battle(
        root["state"], root["requests"], "p1", reveal_opponent_sets=True
    )
    hidden_view = state_to_battle(
        root["state"], root["requests"], "p1", reveal_opponent_sets=False
    )
    assert sum(len(mon.moves) for mon in open_view.opponent_team.values()) >= 16
    assert sum(len(mon.moves) for mon in hidden_view.opponent_team.values()) == 0


def test_opponent_choice_likelihood_uses_moves_targets_and_switches():
    move_predictions = (
        MovePrediction(
            (("thunderbolt", 0.8), ("protect", 0.2)),
            (),
            (
                ("thunderbolt", "foe_a", 0.64),
                ("thunderbolt", "foe_b", 0.16),
                ("protect", "self", 0.20),
            ),
        ),
        MovePrediction((("rockslide", 0.9), ("protect", 0.1)), (), ()),
    )
    switches = (
        SwitchPrediction(0.25, (("incineroar", 0.8), ("pelipper", 0.2))),
        SwitchPrediction(0.10, (("pelipper", 0.8), ("incineroar", 0.2))),
    )
    roster = ("raichu", "tyranitar", "incineroar", "pelipper", "", "")

    likely = opponent_choice_likelihood(
        "move thunderbolt +1, move rockslide",
        move_predictions,
        switches,
        roster,
    )
    wrong_target = opponent_choice_likelihood(
        "move thunderbolt +2, move rockslide",
        move_predictions,
        switches,
        roster,
    )
    assert likely > wrong_target
    assert opponent_choice_likelihood(
        "switch 3, move rockslide", move_predictions, switches, roster
    ) > opponent_choice_likelihood(
        "switch 4, move rockslide", move_predictions, switches, roster
    )
    assert opponent_choice_likelihood(
        "switch 3, pass",
        None,
        switches,
        roster,
        forced_switch=True,
    ) > 0


def test_mechanics_evaluator_is_finite_on_exact_move_state():
    with ExactShowdownBridge() as bridge:
        result = bridge.create(
            formatid="gen9championsvgc2026regmb",
            seed=[9, 10, 11, 12],
            p1_team_text=(ROOT / "teams/reg_mb/our_team.txt").read_text(),
            p2_team_text=(ROOT / "teams/reg_mb/MB11.txt").read_text(),
            p1_preview="team 1234",
            p2_preview="team 1234",
        )
    value = HybridEvaluator(None)(ExactNode.from_result(result), "p1")
    assert -1.0 <= value <= 1.0


class _AnytimePrior:
    def rank(self, _state, _requests, _role, choices):
        probability = 1.0 / len(choices)
        return [
            RankedChoice(choice, (index, 0), probability)
            for index, choice in enumerate(choices)
        ]


class _AnytimeBridge:
    def __init__(self, fail_after=None):
        self.calls = 0
        self.fail_after = fail_after
        self.screened = []

    def choices(self, state, role):
        if role == "p2":
            return ["reply"]
        if state["id"] == "root":
            return ["move 1", "move 2", "move 3", "move 4"]
        return ["continue 1", "continue 2"]

    def simulate_batch(self, state, branches, timeout_s=None):
        del timeout_s
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise ExactSimulatorError("planned timeout")
        if state["id"] == "root" and len(branches) > 1:
            self.screened = [branch["p1_choice"] for branch in branches]
        results = []
        for branch in branches:
            choice = branch["p1_choice"]
            move = int(choice.split()[-1]) if choice.startswith("move") else 1
            results.append(
                {
                    "state": {
                        "id": "child" if state["id"] == "root" else "leaf",
                        "score": move / 10,
                        "sides": [{}, {}],
                    },
                    "requests": [{}, {}],
                    "turn": 2 if state["id"] == "root" else 3,
                    "request_state": "move",
                    "ended": False,
                    "winner": None,
                }
            )
        return results


def _anytime_planner(bridge):
    return ExactMultiTurnPlanner(
        bridge,
        prior=_AnytimePrior(),
        evaluator=lambda node, _role: float(node.state["score"]),
        config=PlannerConfig(
            depth=2,
            root_width=4,
            opponent_width=1,
            continuation_width=2,
            chance_samples=1,
            expected_weight=1.0,
            cvar_weight=0.0,
            worst_weight=0.0,
            time_budget_s=5,
            anytime=True,
            screen_budget_s=1,
            screen_opponent_width=1,
            deep_root_width=2,
        ),
    )


def test_anytime_search_screens_all_roots_before_deepening_prefix():
    bridge = _AnytimeBridge()
    root = ExactNode(
        state={"id": "root", "score": 0.0, "sides": [{}, {}]},
        requests=[{}, {}],
        turn=1,
        request_state="move",
    )
    result = _anytime_planner(bridge).plan(root)
    assert set(bridge.screened) == {"move 1", "move 2", "move 3", "move 4"}
    assert result.screened_actions == 4
    assert result.deepened_actions == 2
    assert result.elapsed_s <= 5


def test_anytime_search_keeps_complete_screening_on_deep_timeout():
    bridge = _AnytimeBridge(fail_after=1)
    root = ExactNode(
        state={"id": "root", "score": 0.0, "sides": [{}, {}]},
        requests=[{}, {}],
        turn=1,
        request_state="move",
    )
    result = _anytime_planner(bridge).plan(root)
    assert len(result.rankings) == 4
    assert result.deepened_actions == 0
    assert result.truncated
    assert result.fallback_reason == "simulator_timeout"


def _score(choice, value):
    return ActionScore(choice, (1, 1), value, value, value, value, 0.0, 0.5, 1)


def _plan_result(*scores):
    best = max(scores, key=lambda score: score.score)
    return PlanResult(
        best.choice,
        best.actions,
        best.score,
        tuple(scores),
        nodes=1,
        elapsed_s=0.1,
        completed_depth=2,
        truncated=False,
    )


def test_determinization_aggregation_penalizes_hidden_set_downside():
    planner = ExactDeterminizationPlanner(
        _AnytimeBridge(),
        config=PlannerConfig(
            expected_weight=0.6,
            cvar_weight=0.3,
            worst_weight=0.1,
            time_budget_s=5,
        ),
    )
    first = _plan_result(_score("safe", 0.4), _score("risky", 0.9))
    second = _plan_result(_score("safe", 0.4), _score("risky", -0.9))
    result = planner._aggregate([(0.5, first), (0.5, second)], elapsed_s=0.2)
    assert result.choice == "safe"
    assert result.rankings[0].worst == pytest.approx(0.4)


def test_determinization_prefers_best_action_with_required_future_coverage():
    planner = ExactDeterminizationPlanner(_AnytimeBridge())
    high_mass = _plan_result(_score("covered", 0.40), _score("shallow", 0.90))
    high_mass = PlanResult(
        **{**high_mass.__dict__, "deepened_choices": ("covered",)}
    )
    low_mass = _plan_result(_score("covered", 0.40), _score("shallow", 0.90))
    low_mass = PlanResult(
        **{**low_mass.__dict__, "deepened_choices": ("shallow",)}
    )
    result = planner._aggregate(
        [(0.60, high_mass), (0.40, low_mass)],
        elapsed_s=0.2,
        minimum_depth_coverage=0.50,
    )
    assert result.choice == "covered"
    assert result.selected_depth_coverage == pytest.approx(0.60)
