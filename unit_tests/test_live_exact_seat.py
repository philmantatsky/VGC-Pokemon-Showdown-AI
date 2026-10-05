"""The exact search's live bookkeeping must not depend on which seat the server gave
us (2026-10-04). The exact worlds always seat us as p1; the raw protocol names every
actor by the server's seat. Until this date the readers assumed we were p1, mirrored
seat 2's targets, and let a searched pick skip the player's opt-in guards.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vgc_bench.src import guards
from vgc_bench.src.exact_planner import ActionScore
from vgc_bench.src.exact_sim import ExactShowdownBridge
from vgc_bench.src.live_exact import (
    LiveExactSession,
    ObservedAction,
    _move_order_from_events,
    _move_order_from_log,
    _order_error,
    _target_location,
    observed_opponent_actions,
)
from vgc_bench.src.live_snapshot import live_roles, public_snapshot
from vgc_bench.src.set_particles import TeamSlot

ROOT = Path(__file__).resolve().parents[1]


def test_live_roles_follow_the_server_seat():
    assert live_roles(SimpleNamespace(player_role="p1")) == ("p1", "p2")
    assert live_roles(SimpleNamespace(player_role="p2")) == ("p2", "p1")
    # before the |player| line arrives, and for a bare test double: the old reading
    assert live_roles(SimpleNamespace(player_role=None)) == ("p1", "p2")
    assert live_roles(SimpleNamespace()) == ("p1", "p2")


@pytest.mark.parametrize("role, foe", [("p1", "p2"), ("p2", "p1")])
def test_opposing_targets_number_the_same_for_both_seats(role, foe):
    assert _target_location(f"{foe}a: Torkoal", role, 0) == 1
    assert _target_location(f"{foe}b: Farigiraf", role, 0) == 2
    assert _target_location(f"{role}b: Venusaur", role, 0) == -2
    assert _target_location(f"{role}a: Blastoise", role, 0) is None


def test_simulator_agrees_with_the_target_numbering():
    """The fact the mapping rests on, asked of the simulator itself."""
    team = (ROOT / "teams/candidates_mc/T6e.txt").read_text()
    with ExactShowdownBridge() as bridge:
        root = bridge.create(
            formatid="gen9championsvgc2026regmc",
            p1_team_text=team,
            p2_team_text=team,
            p1_preview="team 1234",
            p2_preview="team 1234",
        )
        state = root["state"]
        for role, foe in (("p1", "p2"), ("p2", "p1")):
            passive = next(
                choice
                for choice in bridge.choices(state, foe)
                if "protect" not in choice and "fakeout" not in choice
            )
            for location, slot in (("+1", "a"), ("+2", "b")):
                choice = next(
                    c
                    for c in bridge.choices(state, role)
                    if c.split(",")[0].strip() == f"move waterpulse {location}"
                )
                result = bridge.simulate(
                    state,
                    rng_seed="1,2,3,4",
                    **{f"{role}_choice": choice, f"{foe}_choice": passive},
                )
                line = next(
                    entry
                    for entry in result["log"]
                    if entry.startswith(f"|move|{role}a:")
                )
                target = line.split("|")[4]
                assert target.startswith(f"{foe}{slot}:"), line
                assert _target_location(target, role, 0) == int(location)


def test_observed_actions_read_the_opponent_in_seat_one():
    # we are p2: the opponent's Pokemon are p1a / p1b, ours p2a / p2b
    events = [
        ["", "switch", "p1b: Venusaur", "Venusaur, L50", "100/100"],
        ["", "-mega", "p2a: Charizard", "Charizard", "Charizardite Y"],
        ["", "move", "p2a: Charizard", "Heat Wave", "p1a: Blastoise", "[spread] p1a"],
        ["", "move", "p1a: Blastoise", "Ice Beam", "p2b: Farigiraf"],
        ["", "move", "p2b: Farigiraf", "Trick Room", "p2b: Farigiraf"],
    ]
    assert observed_opponent_actions(events, "p1") == {
        0: ObservedAction("move", "icebeam", 2, False),
        1: ObservedAction("switch", "venusaur"),
    }
    # the old default read our own turn as theirs
    assert observed_opponent_actions(events)[0].identifier == "heatwave"


def test_move_order_keeps_same_named_pokemon_on_each_side_apart():
    # a mirror in seat 2: their Torkoal (p1b) moves before ours (p2a)
    events = [
        ["", "move", "p1b: Torkoal", "Eruption", "p2a: Torkoal"],
        ["", "move", "p2a: Torkoal", "Protect", "p2a: Torkoal"],
        ["", "-damage", "p2b: Farigiraf", "50/100"],
    ]
    observed = _move_order_from_events(events, "p2")
    assert observed == ["p2:torkoal", "p1:torkoal"]
    same = ["|move|p2b: Torkoal|Eruption|p1a: Torkoal", "|move|p1a: Torkoal|Protect"]
    swapped = list(reversed(same))
    assert _move_order_from_log(same) == ["p2:torkoal", "p1:torkoal"]
    assert _order_error(observed, _move_order_from_log(same)) == 0.0
    assert _order_error(observed, _move_order_from_log(swapped)) == 1.0


def _mon(species, **extra):
    fields = dict(
        max_hp=100,
        current_hp=100,
        current_hp_fraction=1.0,
        fainted=False,
        species=species,
        base_species=species,
        status=None,
        boosts={},
        ability=None,
        item=None,
        effects={},
        first_turn=False,
        moves={},
        revealed=True,
        selected_in_teampreview=True,
    )
    fields.update(extra)
    return SimpleNamespace(**fields)


def _mirror_battle(player_role) -> Any:
    """One Charizard each; only the SERVER's p1 has Mega Evolved and moved."""
    ours, theirs = _mon("Charizard"), _mon("Charizard")
    foe = "p1" if player_role == "p2" else "p2"
    return SimpleNamespace(
        player_role=player_role,
        turn=2,
        team={f"{player_role}: Charizard": ours},
        opponent_team={f"{foe}: Charizard": theirs},
        active_pokemon=[ours, None],
        opponent_active_pokemon=[theirs, None],
        side_conditions={},
        opponent_side_conditions={},
        weather={},
        fields={},
        _replay_data=[
            ["", "-mega", "p1a: Charizard", "Charizard", "Charizardite Y"],
            ["", "move", "p1a: Charizard", "Heat Wave", "p2a: Charizard"],
        ],
    )


@pytest.mark.parametrize("player_role", ["p1", "p2"])
def test_snapshot_reads_each_side_from_its_own_seat(player_role):
    snapshot = public_snapshot(_mirror_battle(player_role), None, request_state="move")
    ours, theirs = snapshot["sides"]
    we_acted = player_role == "p1"
    assert ours["mechanic_usage"]["mega_used"] is we_acted
    assert theirs["mechanic_usage"]["mega_used"] is not we_acted
    assert ours["pokemon"][0]["last_move"] == ("heatwave" if we_acted else None)
    assert theirs["pokemon"][0]["last_move"] == (None if we_acted else "heatwave")


def test_revealed_evidence_reads_the_opponent_in_seat_one():
    session = object.__new__(LiveExactSession)
    session.open_sheet = False
    session.opponent_roster_species = {"torkoal"}
    session.opponent_species_by_nickname = {}
    torkoal = _mon("Torkoal")
    battle: Any = SimpleNamespace(
        player_role="p2",
        opponent_team={"p1: Torkoal": torkoal},
        _replay_data=[
            ["", "move", "p1a: Torkoal", "Earth Power", "p2a: Torkoal"],
            # our own Torkoal's move must not be credited to theirs
            ["", "move", "p2a: Torkoal", "Yawn", "p1a: Torkoal"],
            ["", "-enditem", "p1a: Torkoal", "Charcoal"],
        ],
    )
    assert session._revealed_evidence(battle) == {
        "torkoal": {"moves": {"earthpower"}, "item": "charcoal", "ability": None}
    }


def _row(choice, actions, prior):
    return ActionScore(
        choice=choice,
        actions=actions,
        score=prior,
        expected=prior,
        cvar=prior,
        worst=prior,
        standard_deviation=0.0,
        prior=prior,
        opponent_branches=1,
    )


def _enabled_guards(monkeypatch, configure):
    seen = {}

    def fake_apply(_battle, candidates, enabled):
        seen.update(enabled)
        return candidates, guards.GuardReport()

    monkeypatch.setattr(guards, "apply_guards", fake_apply)
    session = object.__new__(LiveExactSession)
    session.last_live_guards = {"strict_rejections": []}
    configure(session)
    rows = (_row("move a", (1, 2), 0.6), _row("move b", (3, 4), 0.4))
    battle: Any = SimpleNamespace()
    assert session._apply_live_hard_guards(battle, rows) == rows
    return {name for name, on in seen.items() if on}


def test_searched_pick_meets_hard_guards_only_until_the_player_hands_its_own(
    monkeypatch,
):
    assert _enabled_guards(monkeypatch, lambda session: None) == set(guards.HARD_GUARDS)


def test_searched_pick_meets_the_players_opt_in_guards(monkeypatch):
    # the deployed shape: an explicit hard-only profile plus the opted-in extras
    flags = {name: name in guards.HARD_GUARDS for name in guards.GUARDS}
    flags.update({"threat_first2": True, "wasted_fake_out": True})
    enabled = _enabled_guards(
        monkeypatch, lambda session: session.set_player_guards(flags)
    )
    assert enabled == set(guards.HARD_GUARDS) | {"threat_first2", "wasted_fake_out"}


def test_player_guards_never_switch_a_hard_guard_off(monkeypatch):
    flags = {name: False for name in guards.GUARDS}
    enabled = _enabled_guards(
        monkeypatch, lambda session: session.set_player_guards(flags)
    )
    assert enabled == set(guards.HARD_GUARDS)


def test_full_profile_player_runs_every_guard_on_a_searched_pick(monkeypatch):
    enabled = _enabled_guards(
        monkeypatch, lambda session: session.set_player_guards(None)
    )
    assert enabled == set(guards.GUARDS)


def test_player_hands_its_guard_switches_to_the_search(monkeypatch):
    from collections import Counter

    from vgc_bench.src.policy_player import PolicyPlayer

    player = object.__new__(PolicyPlayer)  # skip Player.__init__ (no websocket)
    player.guard_overrides = {"threat_first2": True}
    player.exact_player_guards = True
    player.decision_log_path = None
    handed = []
    session = SimpleNamespace(
        set_player_guards=handed.append,
        plan=lambda _battle, _champion: None,
        audit=dict,
    )
    monkeypatch.setattr(player, "_live_exact_session", lambda _battle: session)
    monkeypatch.setattr(PolicyPlayer, "guard_fire_counts", Counter())
    monkeypatch.setattr(PolicyPlayer, "use_knowledge_guards", True)
    monkeypatch.setattr(
        PolicyPlayer, "guard_flags", {"zero_damage": True, "threat_first2": False}
    )
    battle: Any = SimpleNamespace(battle_tag="battle-test-1", turn=1)

    assert player._exact_search_action(battle) is None
    # exactly what the unsearched path hands apply_guards
    assert handed == [{"zero_damage": True, "threat_first2": True}]
    assert handed[0] == player._effective_guard_flags()

    handed.clear()
    player.exact_player_guards = False  # every search run before 2026-10-04
    assert player._exact_search_action(battle) is None
    assert handed == []

    player.exact_player_guards = True
    monkeypatch.setattr(PolicyPlayer, "use_knowledge_guards", False)
    assert player._exact_search_action(battle) is None
    assert handed == []


def test_audit_keeps_the_guarded_policy_pick_as_its_reference(monkeypatch):
    from vgc_bench.src.exact_planner import PlanResult

    def demote_first_of_prior_order(_battle, candidates, _enabled):
        # a guard that vetoes the policy's raw favourite, (1, 2)
        kept = [c for c in candidates if c.actions != (1, 2)]
        vetoed = [c for c in candidates if c.actions == (1, 2)]
        report = guards.GuardReport()
        if candidates[0].actions == (1, 2):
            report.note("dominated_attack")
        return kept + vetoed, report

    monkeypatch.setattr(guards, "apply_guards", demote_first_of_prior_order)
    session = object.__new__(LiveExactSession)
    actions = {"favourite": (1, 2), "second": (3, 4), "searched": (5, 6)}
    monkeypatch.setattr(
        session, "_live_actions", lambda choice, _battle: actions[choice]
    )
    monkeypatch.setattr(
        session, "_strict_live_rejection_reason", lambda _battle, _actions: None
    )
    rows = (
        _row("searched", None, 0.1),
        _row("favourite", None, 0.6),
        _row("second", None, 0.3),
    )
    result = PlanResult(
        choice="searched",
        actions=None,
        score=0.9,
        rankings=rows,
        nodes=1,
        elapsed_s=0.1,
        completed_depth=1,
        truncated=False,
    )
    battle: Any = SimpleNamespace()
    ranked, illegal = session._live_legal_rankings(result, battle)
    assert illegal == [] and ranked[0].choice == "searched"
    # the search's own pass: nothing fired, the searched pick stood
    assert session.last_live_guards["changed_pick"] is False
    assert session.last_live_guards["stages"] == []
    # without the search the bot plays "second": the guard vetoes the raw favourite
    assert session.last_live_guards["policy_choice"] == "second"


def test_search_audit_counts_changes_against_the_guarded_policy_pick(tmp_path):
    import json

    from evaluation.search_audit import summarize

    def row(choice, policy_choice, stages):
        return {
            "battle": "b",
            "turn": 1,
            "exact_search": {
                "schedule": {
                    "mode": "search",
                    "live_guards": {
                        "changed_pick": bool(stages),
                        "stages": stages,
                        "policy_choice": policy_choice,
                    },
                },
                "result": {
                    "choice": choice,
                    "elapsed_s": 1.0,
                    "rankings": [
                        {"choice": "favourite", "prior": 0.6, "score": 0.2},
                        {"choice": "second", "prior": 0.3, "score": 0.7},
                    ],
                },
            },
        }

    log = tmp_path / "a_decisions.jsonl"
    rows = [
        row("second", "second", ["dominated_attack"]),  # the guards' doing
        row("second", "favourite", []),  # the search's doing
        row("favourite", "favourite", []),
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    summary = summarize([log])
    assert summary["search_changed_policy_favourite"] == 2
    assert summary["search_changed_guarded_policy_choice"] == {"changed": 1, "of": 3}
    assert summary["guards_changed_searched_pick"] == 1
    assert summary["guard_stages_on_searched_picks"] == {"dominated_attack": 1}


def test_a_pair_a_guard_adds_is_ranked_but_never_counts_as_searched(monkeypatch):
    """2026-10-04: with the player's guards on a searched ranking, a guard that adds
    a pair of its own raised KeyError and cost the whole search step."""

    def add_a_pair(_battle, candidates, _enabled):
        report = guards.GuardReport()
        report.note("guaranteed_ko")
        return [guards.Candidate(actions=(0, 25), prob=0.4)] + list(candidates), report

    monkeypatch.setattr(guards, "apply_guards", add_a_pair)
    session = object.__new__(LiveExactSession)
    session.min_deep_coverage = 0.0  # even a session that demands no depth
    session.last_live_guards = {"strict_rejections": []}
    rows = (_row("move a", (1, 2), 0.6), _row("move b", (3, 4), 0.4))
    battle: Any = SimpleNamespace()
    ranked = session._apply_live_hard_guards(battle, rows)
    assert [row.actions for row in ranked] == [(0, 25), (1, 2), (3, 4)]
    assert ranked[0].choice.startswith("guard-added")
    assert not session._reaches_required_depth(ranked[0])
    assert session.last_live_guards["changed_pick"] is True
    assert session.last_live_guards["stages"] == ["guaranteed_ko"]


def test_champion_reference_is_the_unsearched_pick_and_leaves_no_trace(monkeypatch):
    from collections import Counter

    import numpy as np

    from vgc_bench.src.policy_player import PolicyPlayer

    player = object.__new__(PolicyPlayer)
    player.enable_search = True
    player.decision_log_path = Path("decisions.jsonl")
    player._exact_sessions = {"battle-test-1": "the live session"}
    seen: dict[str, Any] = {}

    def unsearched(battle, obs_dict, mask):
        # what the real path would see while the reference is taken
        seen.update(
            search=player.enable_search,
            log=player.decision_log_path,
            sessions=dict(player._exact_sessions),
        )
        PolicyPlayer.guard_fire_counts["guards_ran"] += 1
        PolicyPlayer._decisions_seen += 1
        return np.array([12, 34])

    monkeypatch.setattr(player, "_guarded_action", unsearched)
    monkeypatch.setattr(PolicyPlayer, "guard_fire_counts", Counter(guards_ran=7))
    monkeypatch.setattr(PolicyPlayer, "_decisions_seen", 40)
    battle: Any = SimpleNamespace(battle_tag="battle-test-1")

    assert player._champion_reference(battle, {}, None) == [12, 34]
    # its own decision audit goes to a file beside the search's
    assert seen == {
        "search": False,
        "log": Path("decisions_champion.jsonl"),
        "sessions": {},
    }
    assert player.enable_search is True
    assert player.decision_log_path == Path("decisions.jsonl")
    assert player._exact_sessions == {"battle-test-1": "the live session"}
    assert PolicyPlayer.guard_fire_counts == Counter(guards_ran=7)
    assert PolicyPlayer._decisions_seen == 40

    def broken(battle, obs_dict, mask):
        raise RuntimeError("no policy")

    monkeypatch.setattr(player, "_guarded_action", broken)
    assert player._champion_reference(battle, {}, None) is None
    assert player.enable_search is True


def test_search_audit_counts_changes_against_the_champion_action(tmp_path):
    import json

    from evaluation.search_audit import summarize

    def row(mode, actions, champion, policy_choice="x", changed_pick=False):
        return {
            "exact_search": {
                "schedule": {
                    "mode": mode,
                    "live_guards": {
                        "changed_pick": changed_pick,
                        "stages": [],
                        "policy_choice": policy_choice,
                    },
                },
                "result": {"choice": "x", "elapsed_s": 1.0, "rankings": []},
                "actions": actions,
                "champion_actions": champion,
            }
        }

    rows = [
        row("search", [1, 2], [1, 2]),
        row("search", [3, 4], [1, 2], policy_choice="y"),  # the search's scores
        row("search", [3, 4], [1, 2], changed_pick=True),  # guards on its ranking
        row("search", [3, 4], [1, 2]),  # same ranking, different view of the game
        row("search_fallback", None, [1, 2]),  # plays the champion's pair
    ]
    log = tmp_path / "a_decisions.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert summarize([log])["search_changed_champion_action"] == {
        "changed": 3,
        "of": 5,
        "by": {
            "the_search_scores": 1,
            "guards_on_the_searched_ranking": 1,
            "rebuilt_view_or_candidate_set": 1,
        },
    }


def test_champion_pair_is_spelled_per_root(monkeypatch):
    spellings = {"w1": {"move a": (1, 2), "switch 3, move b": (5, 6)}, "w2": {}}
    spellings["w2"] = {"move a": (1, 2), "switch 4, move b": (5, 6)}
    session = object.__new__(LiveExactSession)
    monkeypatch.setattr(
        session,
        "bridge",
        SimpleNamespace(choices=lambda state, _role: list(spellings[state["id"]])),
        raising=False,
    )
    monkeypatch.setattr(
        LiveExactSession,
        "_root_live_actions",
        staticmethod(lambda root, choice, _battle: spellings[root.label][choice]),
    )
    roots: Any = [
        SimpleNamespace(label=label, node=SimpleNamespace(state={"id": label}))
        for label in ("w1", "w2")
    ]
    battle: Any = SimpleNamespace()
    # a switch is numbered by each root's own party order
    assert session._champion_choices(roots, [5, 6], battle) == [
        "switch 3, move b",
        "switch 4, move b",
    ]
    assert session._champion_choices(roots, [9, 9], battle) == [None, None]
    assert session._champion_choices(roots, None, battle) is None


def test_an_attack_into_an_empty_slot_is_respelled_at_the_opponent_left():
    """The policy may aim at an empty opposing slot (the simulator re-aims the move);
    the search ranks only the real target. 2026-10-04: 8 of 9 remaining differences
    between a null search and the bot's own pick were this spelling."""
    from vgc_bench.src.live_exact import canonical_target_actions

    foe = SimpleNamespace(fainted=False)
    gone = SimpleNamespace(fainted=True)
    only_b: Any = SimpleNamespace(opponent_active_pokemon=[None, foe])
    only_a: Any = SimpleNamespace(opponent_active_pokemon=[foe, gone])
    both: Any = SimpleNamespace(opponent_active_pokemon=[foe, foe])
    none: Any = SimpleNamespace(opponent_active_pokemon=[None, None])
    # action = 7 + move slot x 5 + (target + 2), +20 with a Mega
    assert canonical_target_actions(only_b, [15, 21]) == (16, 21)  # +1 -> +2
    assert canonical_target_actions(only_b, [35, 0]) == (36, 0)  # the Mega variant
    assert canonical_target_actions(only_a, [16, 26]) == (15, 25)  # +2 -> +1
    assert canonical_target_actions(both, [15, 26]) == (15, 26)
    assert canonical_target_actions(none, [15, 26]) == (15, 26)
    # no target, an ally target, a switch, a pass: untouched
    assert canonical_target_actions(only_b, [14, 13]) == (14, 13)
    assert canonical_target_actions(only_b, [3, 0]) == (3, 0)


def _ranked_session(monkeypatch, champion, guard):
    from vgc_bench.src.exact_planner import PlanResult

    monkeypatch.setattr(guards, "apply_guards", guard)
    session = object.__new__(LiveExactSession)
    session.champion_actions = champion
    pairs = {"searched": (5, 6), "own": (1, 2), "third": (3, 4)}
    monkeypatch.setattr(session, "_live_actions", lambda choice, _battle: pairs[choice])
    monkeypatch.setattr(
        session, "_strict_live_rejection_reason", lambda _battle, _actions: None
    )

    def rank(order):
        rows = tuple(_row(choice, None, 0.3) for choice in order)
        result = PlanResult(
            choice=order[0],
            actions=None,
            score=0.9,
            rankings=rows,
            nodes=1,
            elapsed_s=0.1,
            completed_depth=1,
            truncated=False,
        )
        battle: Any = SimpleNamespace()
        ranked, _illegal = session._live_legal_rankings(result, battle)
        return [row.choice for row in ranked]

    return session, rank


def test_the_bots_own_pick_is_not_put_through_the_guards_a_second_time(monkeypatch):
    def must_not_run(_battle, _candidates, _enabled):
        raise AssertionError("the guards ran on the bot's own pair")

    session, rank = _ranked_session(monkeypatch, (1, 2), must_not_run)
    assert rank(["own", "searched", "third"]) == ["own", "searched", "third"]
    assert session.last_live_guards["champion"] == "kept"


def test_an_override_the_guards_accept_is_played(monkeypatch):
    def accept(_battle, candidates, _enabled):
        return candidates, guards.GuardReport()

    session, rank = _ranked_session(monkeypatch, (1, 2), accept)
    assert rank(["searched", "own", "third"])[0] == "searched"
    assert session.last_live_guards["champion"] == "overridden"


def test_an_override_the_guards_reject_gives_way_to_the_bots_own_pick(monkeypatch):
    def reject_the_top(_battle, candidates, _enabled):
        report = guards.GuardReport()
        report.note("dominated_attack")
        # the stack would promote the THIRD pair among these few, not the bot's own
        return [candidates[2], candidates[1], candidates[0]], report

    session, rank = _ranked_session(monkeypatch, (1, 2), reject_the_top)
    assert rank(["searched", "own", "third"]) == ["own", "third", "searched"]
    assert session.last_live_guards["champion"] == "override_vetoed"
    assert session.last_live_guards["stages"] == ["dominated_attack"]


def test_reply_coverage_holds_the_searched_replies_against_what_happened():
    from vgc_bench.src.exact_planner import BranchOutcome, ExactNode

    def node():
        return ExactNode(
            state={"sides": [{"pokemon": []}, {"pokemon": []}]},
            requests=[None, None],
            turn=1,
            request_state="move",
        )

    def outcome(label, reply):
        return BranchOutcome(
            root_choice="move a",
            opponent_choice=reply,
            value=0.0,
            probability=0.5,
            predicted_node=node(),
            searched_depth=1,
            root_label=label,
        )

    session = object.__new__(LiveExactSession)
    roots: Any = [
        SimpleNamespace(label="w1", node=node(), probability=0.75),
        SimpleNamespace(label="w2", node=node(), probability=0.25),
        SimpleNamespace(label="unsearched", node=node(), probability=0.5),
    ]
    session._record_reply_tables(
        [
            outcome("w1", "move icebeam +1, move trickroom"),
            outcome("w1", "move waterspout, move trickroom"),
            outcome("w2", "move waterpulse +2, move protect"),
        ],
        roots[:2],
    )
    session.last_observed_actions = {
        0: ObservedAction("move", "waterspout", 1, False),  # a spread move's log target
        1: ObservedAction("move", "trickroom", None, False),
    }
    coverage = session._reply_coverage()
    assert coverage is not None and coverage["any"] and coverage["worlds"] == 2
    assert coverage["mass_with_reply"] == pytest.approx(0.75)
    assert coverage["observed"]["0"] == ["move", "waterspout", 1, False]
    # consumed: the same tables are never held against a second turn
    assert session._reply_coverage() is None

    session._record_reply_tables(
        [outcome("w1", "move icebeam +1, move protect")], roots
    )
    session.last_observed_actions = {0: ObservedAction("move", "fakeout", 2, False)}
    missed = session._reply_coverage()
    assert missed is not None and not missed["any"]
    assert missed["mass_with_reply"] == 0.0
    session._record_reply_tables(
        [outcome("w1", "move icebeam +1, move protect")], roots
    )
    session.last_observed_actions = {}  # nothing seen (the first decision)
    assert session._reply_coverage() is None


def _hidden_session(particles) -> Any:
    session: Any = object.__new__(LiveExactSession)
    session.open_sheet = False
    session.oracle_opponent_team_text = None
    session.database = SimpleNamespace(particles=lambda species: particles[species])
    session.opponent_roster_species = set(particles)
    session.opponent_species_by_nickname = {}
    return session


def test_hidden_worlds_are_drawn_from_sets_that_hold_what_was_shown():
    """2026-10-04: a world recreated mid-battle was sampled from the untouched prior,
    so the opponent's Blastoise could lack the Water Pulse it had already used."""
    import random

    from vgc_bench.src.set_particles import SetParticle

    def particle(moves, probability, item="blastoisinite"):
        return SetParticle(
            "blastoise", "raindish", item, moves, None, probability, "test"
        )

    common = particle(("shellsmash", "darkpulse", "protect", "waterspout"), 0.9)
    ours = particle(("waterpulse", "icebeam", "waterspout", "fakeout"), 0.1)
    session = _hidden_session({"blastoise": (common, ours)})
    roster = (TeamSlot("blastoise", "Blastoise"),)
    blastoise = _mon("Blastoise")

    def battle(*events) -> Any:
        return SimpleNamespace(
            player_role="p1",
            opponent_team={"p2: Blastoise": blastoise},
            _replay_data=[list(event) for event in events],
        )

    def drawn(live):
        belief = session._belief(live, roster)
        worlds = belief.sample_determinizations(8, random.Random(0))
        return {world["blastoise"].moves for world in worlds}

    assert common.moves in drawn(battle())  # nothing shown: the prior
    shown = battle(("", "move", "p2a: Blastoise", "Water Pulse", "p1a: Torkoal"))
    assert drawn(shown) == {ours.moves}
    # a move no stored set has: one set built around what was seen
    novel = battle(
        ("", "move", "p2a: Blastoise", "Aura Sphere", "p1a: Torkoal"),
        ("", "move", "p2a: Blastoise", "Struggle", "p1a: Torkoal"),  # not a set move
    )
    (moves,) = drawn(novel)
    assert "aurasphere" in moves and "struggle" not in moves and len(moves) == 4


def test_contradicted_hidden_worlds_are_drawn_again_once_per_new_evidence(monkeypatch):
    session: Any = object.__new__(LiveExactSession)
    session.open_sheet = False
    session.oracle_opponent_team_text = None
    session.search_determinizations = 2
    session._ponder_job = None
    session.root_refreshes = 0
    session.reconciliations = 0
    session.last_reconcile_errors = []
    session.roots = ["old-1", "old-2", "old-3"]
    session.pending_our_choice = "move a"
    session.roots_agreeing_with_evidence = 1  # fewer than the two it searches
    session.evidence_signature = (("blastoise", ("waterpulse",), None, None),)
    created = []

    def create(_battle, _snapshot):
        created.append(1)
        session.roots = ["new-1", "new-2"]

    def condition(_battle):
        session.roots_agreeing_with_evidence = len(session.roots)

    monkeypatch.setattr(session, "_create_roots", create, raising=False)
    monkeypatch.setattr(session, "_condition_roots", condition, raising=False)
    battle: Any = SimpleNamespace(turn=4)

    session._resample_contradicted_roots(battle, {})
    assert session.roots == ["new-1", "new-2"] and len(created) == 1
    assert session.root_refreshes == 1 and session.evidence_resamples == 1
    assert session.pending_our_choice is None  # nothing searched on the old worlds

    # still contradicted by the SAME evidence (a set no particle can express): once
    session.roots_agreeing_with_evidence = 0
    session._resample_contradicted_roots(battle, {})
    assert len(created) == 1
    # a new reveal: again
    session.evidence_signature += (("torkoal", ("earthpower",), None, None),)
    session._resample_contradicted_roots(battle, {})
    assert len(created) == 2

    # a failed redraw leaves the reconciled worlds it had
    def broken(_battle, _snapshot):
        raise ValueError("no legal determinization")

    monkeypatch.setattr(session, "_create_roots", broken, raising=False)
    session.roots = ["kept"]
    session.roots_agreeing_with_evidence = 0
    session.evidence_signature += (("incineroar", ("uturn",), None, None),)
    session._resample_contradicted_roots(battle, {})
    assert session.roots == ["kept"]
    assert "evidence resample" in session.last_reconcile_errors[-1]

    # open sheets name the sets: never
    session.open_sheet = True
    session.evidence_signature += (("farigiraf", ("psychic",), None, None),)
    monkeypatch.setattr(session, "_create_roots", create, raising=False)
    session._resample_contradicted_roots(battle, {})
    assert len(created) == 2


def test_a_megas_ability_is_not_evidence_about_its_set():
    session = _hidden_session({"gyarados": ()})
    gyarados = _mon("Gyarados")
    battle: Any = SimpleNamespace(
        player_role="p1",
        opponent_team={"p2: Gyarados": gyarados},
        _replay_data=[
            ["", "-ability", "p2a: Gyarados", "Intimidate", "boost"],
            ["", "-mega", "p2a: Gyarados", "Gyarados", "Gyaradosite"],
            ["", "-ability", "p2a: Gyarados", "Mold Breaker"],
        ],
    )
    assert session._revealed_evidence(battle) == {
        "gyarados": {"moves": set(), "item": "gyaradosite", "ability": "intimidate"}
    }
    # an open sheet: the live Pokemon reports the Mega's ability once it has evolved
    session.open_sheet = True
    mega = _mon("Gyarados", ability="moldbreaker", forme_change_ability="moldbreaker")
    mega.moves = {"waterfall": None}
    sheet: Any = SimpleNamespace(
        player_role="p1", opponent_team={"p2: Gyarados": mega}, _replay_data=[]
    )
    assert session._revealed_evidence(sheet)["gyarados"]["ability"] is None
