from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from poke_env.battle import SideCondition

from vgc_bench.src.exact_observation import (
    _stale_branch_move_conflict,
    choice_to_actions,
    state_to_battle,
)
from vgc_bench.src.exact_planner import (
    ActionScore,
    BranchContinuation,
    BranchOutcome,
    ExactNode,
    PlanResult,
)
from vgc_bench.src.exact_sim import ExactShowdownBridge, live_snapshot_supported
from vgc_bench.src.live_exact import (
    LiveExactSession,
    LiveRoot,
    ObservedAction,
    _opponent_previews,
    _our_preview,
    choice_matches_observation,
    observed_opponent_actions,
)
from vgc_bench.src.live_snapshot import (
    apply_public_snapshot,
    _public_effect_move,
    _public_item,
    _public_move_state,
    public_snapshot,
)
from vgc_bench.src.ponder import PonderOutcome
from vgc_bench.src.set_particles import SetParticle, TeamSlot
from verify_live_parity import (
    _canonical_private_targets,
    _private_target_counter,
)

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "gen9championsvgc2026regmb"


def _battle_with_events(events):
    return SimpleNamespace(_replay_data=events)


def test_public_move_state_keeps_same_nickname_sides_separate():
    events = [
        ["", "switch", "p2a: Garchomp", "Garchomp, L50", "100/100"],
        ["", "move", "p2a: Garchomp", "Breaking Swipe", "p1a: Kingambit"],
        ["", "switch", "p1a: Garchomp", "Garchomp, L50", "191/191"],
    ]
    assert _public_move_state(_battle_with_events(events), "garchomp", "p2") == (
        "breakingswipe",
        False,
        None,
    )


def test_only_stale_fifth_move_assertion_is_tolerated():
    conflict = AssertionError(
        "Error with move solarbeam. Expected self.moves to contain copycat"
    )
    assert _stale_branch_move_conflict(
        ["", "move", "p2a: Charizard", "Solar Beam", "p1a: Garchomp"],
        conflict,
    )
    assert not _stale_branch_move_conflict(["", "switch"], conflict)
    assert not _stale_branch_move_conflict(
        ["", "move"], AssertionError("unrelated mechanics assertion")
    )


def test_public_move_state_distinguishes_charge_from_weather_shortcut():
    charge = [
        ["", "move", "p2a: Archaludon", "Electro Shot", "", "[still]"],
        ["", "-prepare", "p2a: Archaludon", "Electro Shot"],
    ]
    instant = charge + [
        ["", "-anim", "p2a: Archaludon", "Electro Shot", "p1a: Floette"]
    ]
    assert _public_move_state(
        _battle_with_events(charge), "archaludon", "p2"
    )[2] == "electroshot"
    assert (
        _public_move_state(
            _battle_with_events(instant), "archaludon", "p2"
        )[2]
        is None
    )


def test_parity_allows_only_hidden_opponent_charge_target_difference():
    snapshot = {
        "sides": [
            {"pokemon": []},
            {
                "pokemon": [
                    {
                        "active_slot": 0,
                        "effects": {
                            "twoturnmove": {"move": "electroshot", "duration": 1}
                        },
                    }
                ]
            },
        ]
    }
    source = ["move electroshot +1, move protect"]
    shadow = ["move electroshot +2, move protect"]
    assert _private_target_counter(source, snapshot) == _private_target_counter(
        shadow, snapshot
    )
    assert _canonical_private_targets(source[0], {}) == source[0]
    # Multiplicity still catches a shadow that incorrectly offers several targets.
    assert _private_target_counter(source, snapshot) != _private_target_counter(
        shadow * 2, snapshot
    )


def test_opponent_snapshot_does_not_expose_or_refill_private_pp():
    move = SimpleNamespace(id="protect", current_pp=16)
    pokemon = SimpleNamespace(
        max_hp=100,
        current_hp=100,
        current_hp_fraction=1.0,
        fainted=False,
        species="Floette-Eternal",
        base_species="Floette",
        status=None,
        boosts={},
        ability="flowerveil",
        item=None,
        effects={},
        first_turn=False,
        moves={"protect": move},
    )
    from vgc_bench.src.live_snapshot import _pokemon_snapshot

    row = _pokemon_snapshot(
        pokemon,
        ident="p2: Floette",
        battle=SimpleNamespace(turn=1, _replay_data=[]),
        active_slot=0,
        own=False,
        request_moves=None,
    )
    assert row["moves"] == [{"id": "protect"}]


def test_public_snapshot_rebuilds_fixed_right_active_slot():
    def mon(species):
        return SimpleNamespace(
            base_species=species,
            _active=False,
            _max_hp=100,
            _current_hp=100,
            _status=None,
            _boosts={},
            _ability=None,
            _item=None,
            _effects={},
        )

    floette = mon("Floette-Eternal")
    garchomp = mon("Garchomp")
    staraptor = mon("Staraptor")
    tyranitar = mon("Tyranitar")
    battle = SimpleNamespace(
        _turn=7,
        turn=7,
        # Deliberately use Showdown's post-switch order. The snapshot below carries
        # the stable live Team Preview order and must restore it.
        team={"p1: Garchomp": garchomp, "p1: Floette": floette},
        opponent_team={"p2: Staraptor": staraptor, "p2: Tyranitar": tyranitar},
        _active_pokemon={"p1a": floette, "p2a": staraptor},
        _side_conditions={},
        _opponent_side_conditions={},
        _weather={},
        _fields={},
    )
    snapshot = {
        "turn": 7,
        "weather": None,
        "terrain": None,
        "pseudo_weather": {},
        "sides": [
            {
                "pokemon": [
                    {
                        "nickname": "floette",
                        "base_species": "floetteeternal",
                        "active_slot": 1,
                        "maxhp": 151,
                        "hp": 76,
                        "status": "",
                        "boosts": {},
                        "effects": {},
                    },
                    {
                        "nickname": "garchomp",
                        "base_species": "garchomp",
                        "active_slot": None,
                        "maxhp": 191,
                        "hp": 191,
                        "status": "",
                        "boosts": {},
                        "effects": {},
                    },
                ],
                "side_conditions": {},
            },
            {
                "pokemon": [
                    {
                        "nickname": "tyranitar",
                        "base_species": "tyranitar",
                        "active_slot": None,
                        "hp_fraction": 1.0,
                        "status": "",
                        "boosts": {},
                        "effects": {},
                    },
                    {
                        "nickname": "staraptor",
                        "base_species": "staraptor",
                        "active_slot": 0,
                        "hp_fraction": 0.37,
                        "status": "",
                        "boosts": {},
                        "effects": {},
                    }
                ],
                "side_conditions": {},
            },
        ],
    }

    apply_public_snapshot(battle, snapshot)

    assert "p1a" not in battle._active_pokemon
    assert battle._active_pokemon["p1b"] is floette
    assert battle._active_pokemon["p2a"] is staraptor
    assert list(battle.team.values()) == [floette, garchomp]
    assert list(battle.opponent_team.values()) == [staraptor, tyranitar]


def test_public_encore_keeps_the_move_captured_at_start():
    battle = _battle_with_events(
        [
            ["", "move", "p2a: Incineroar", "Fake Out", "p1a: Floette"],
            ["", "-start", "p2a: Incineroar", "Encore"],
            ["", "move", "p2a: Incineroar", "Struggle", "p1a: Floette"],
        ]
    )
    assert _public_effect_move(battle, "incineroar", "p2", "encore") == "fakeout"


def test_public_item_tracks_trick_and_consumption_over_stale_model_field():
    traded = _battle_with_events(
        [["", "-item", "p2a: Gholdengo", "Life Orb", "[from] move: Trick"]]
    )
    assert _public_item(
        traded, "gholdengo", "p2", "choicescarf", own=False
    ) == "lifeorb"
    consumed = _battle_with_events(
        [["", "-enditem", "p1a: Garchomp", "Sitrus Berry", "[eat]"]]
    )
    assert _public_item(
        consumed, "garchomp", "p1", "sitrusberry", own=True
    ) == ""


def test_observed_actions_match_move_target_and_mega():
    events = [
        ["", "-mega", "p2a: Charizard", "Charizard", "Charizardite Y"],
        ["", "move", "p2a: Charizard", "Weather Ball", "p1b: Floette"],
        ["", "move", "p2b: Whimsicott", "Tailwind", "p2b: Whimsicott"],
    ]
    observed = observed_opponent_actions(events)
    assert observed == {
        0: ObservedAction("move", "weatherball", 1, True),
        1: ObservedAction("move", "tailwind", None, False),
    }
    node = ExactNode(
        state={"sides": [{"pokemon": []}, {"pokemon": []}]},
        requests=[None, None],
        turn=1,
        request_state="move",
    )
    assert choice_matches_observation(
        node,
        "p2",
        "move weatherball +1 mega, move tailwind",
        observed,
    )
    assert not choice_matches_observation(
        node,
        "p2",
        "move weatherball +2 mega, move tailwind",
        observed,
    )


def test_observed_spread_target_matches_targetless_showdown_command():
    events = [
        ["", "move", "p2a: Tyranitar", "Protect", "p2a: Tyranitar"],
        ["", "move", "p2b: Excadrill", "Rock Slide", "p1b: Charizard"],
    ]
    observed = observed_opponent_actions(events)
    node = ExactNode(
        state={"sides": [{"pokemon": []}, {"pokemon": []}]},
        requests=[None, None],
        turn=1,
        request_state="move",
    )
    assert observed[1].target == 1
    assert choice_matches_observation(
        node,
        "p2",
        "move protect, move rockslide",
        observed,
    )


def test_divergent_rng_reconcile_preserves_both_sides_legal_actions():
    our_team = (ROOT / "teams/reg_mb/our_team.txt").read_text()
    their_team = (ROOT / "teams/reg_mb/MB1.txt").read_text()
    with ExactShowdownBridge() as bridge:
        source = bridge.create(
            formatid=FORMAT,
            seed=[101, 102, 103, 104],
            p1_team_text=our_team,
            p2_team_text=their_team,
            p1_preview="team 1,2,3,4",
            p2_preview="team 1,2,3,4",
        )
        shadow = bridge.create(
            formatid=FORMAT,
            seed=[101, 102, 103, 104],
            p1_team_text=our_team,
            p2_team_text=their_team,
            p1_preview="team 1,2,3,4",
            p2_preview="team 1,2,3,4",
        )
        p1_choice = bridge.choices(source["state"], "p1")[0]
        p2_choice = bridge.choices(source["state"], "p2")[0]
        source = bridge.simulate(
            source["state"], p1_choice, p2_choice, "11,12,13,14"
        )
        shadow = bridge.simulate(
            shadow["state"], p1_choice, p2_choice, "51,52,53,54"
        )
        battle = state_to_battle(
            source["state"], source["requests"], "p1", True
        )
        snapshot = public_snapshot(
            battle,
            source["requests"][0],
            request_state=source["request_state"],
            side_requests=source["requests"],
        )
        repaired = bridge.reconcile(shadow["state"], snapshot)
        for role in ("p1", "p2"):
            assert set(bridge.choices(source["state"], role)) == set(
                bridge.choices(repaired["state"], role)
            )


def test_reconcile_uses_active_identity_when_shadow_has_duplicate_species():
    """A stale duplicate must not consume every hidden-set search root."""
    our_team = (ROOT / "teams/reg_mb/our_team.txt").read_text()
    their_team = (ROOT / "teams/reg_mb/MB1.txt").read_text()
    with ExactShowdownBridge() as bridge:
        root = bridge.create(
            formatid=FORMAT,
            seed=[105, 106, 107, 108],
            p1_team_text=our_team,
            p2_team_text=their_team,
            p1_preview="team 1,2,3,4",
            p2_preview="team 1,2,3,4",
        )
        battle = state_to_battle(root["state"], root["requests"], "p1", True)
        snapshot = public_snapshot(
            battle,
            root["requests"][0],
            request_state=root["request_state"],
            side_requests=root["requests"],
        )
        shadow = deepcopy(root["state"])
        # Mimic the stale roster shape observed in the local loss: an active species
        # also appears in a bench record, while side.active still points to the live
        # identity that should win the disambiguation.
        shadow["sides"][1]["pokemon"][2] = deepcopy(
            shadow["sides"][1]["pokemon"][0]
        )

        repaired = bridge.reconcile(shadow, snapshot)

        assert bridge.choices(repaired["state"], "p1")
        assert bridge.choices(repaired["state"], "p2")


def test_reconcile_spends_side_mega_resource_and_all_choices_round_trip():
    """A non-Mega shadow must not offer a second Mega after live Floette used it."""
    our_team = (ROOT / "teams/reg_mb/our_team.txt").read_text()
    their_team = (ROOT / "teams/reg_mb/MB1.txt").read_text()
    with ExactShowdownBridge() as bridge:
        source = bridge.create(
            formatid=FORMAT,
            seed=[111, 112, 113, 114],
            p1_team_text=our_team,
            p2_team_text=their_team,
            p1_preview="team 1,2,3,4",
            p2_preview="team 1,2,3,4",
        )
        shadow = bridge.create(
            formatid=FORMAT,
            seed=[111, 112, 113, 114],
            p1_team_text=our_team,
            p2_team_text=their_team,
            p1_preview="team 1,2,3,4",
            p2_preview="team 1,2,3,4",
        )
        source = bridge.simulate(
            source["state"],
            "move protect mega, move protect",
            "move protect, move protect",
            "121,122,123,124",
        )
        shadow = bridge.simulate(
            shadow["state"],
            "move protect, move protect",
            "move protect, move protect",
            "121,122,123,124",
        )
        assert any("mega" in choice for choice in bridge.choices(shadow["state"], "p1"))

        battle = state_to_battle(source["state"], source["requests"], "p1", True)
        snapshot = public_snapshot(
            battle,
            source["requests"][0],
            request_state=source["request_state"],
            side_requests=source["requests"],
        )
        assert snapshot["sides"][0]["mechanic_usage"]["mega_used"]
        repaired = bridge.reconcile(shadow["state"], snapshot)
        source_choices = set(bridge.choices(source["state"], "p1"))
        repaired_choices = set(bridge.choices(repaired["state"], "p1"))

        assert source_choices == repaired_choices
        assert not any("mega" in choice for choice in repaired_choices)

        repaired_battle = state_to_battle(
            repaired["state"], repaired["requests"], "p1", True
        )
        encoded = {
            choice_to_actions(
                choice,
                repaired["requests"][0],
                state=repaired["state"],
                role="p1",
                battle=repaired_battle,
            )
            for choice in repaired_choices
        }
        assert encoded


def test_public_snapshot_marker_is_not_reapplied_to_exact_child():
    our_team = (ROOT / "teams/reg_mb/our_team.txt").read_text()
    their_team = (ROOT / "teams/reg_mb/MB1.txt").read_text()
    with ExactShowdownBridge() as bridge:
        root = bridge.create(
            formatid=FORMAT,
            seed=[301, 302, 303, 304],
            p1_team_text=our_team,
            p2_team_text=their_team,
            p1_preview="team 1,2,3,4",
            p2_preview="team 1,2,3,4",
        )
        battle = state_to_battle(root["state"], root["requests"], "p1", True)
        snapshot = public_snapshot(
            battle,
            root["requests"][0],
            request_state="move",
            side_requests=root["requests"],
        )
        reconciled = bridge.reconcile(root["state"], snapshot)
        switched = bridge.simulate(
            reconciled["state"],
            "switch 3, move protect",
            "move protect, move protect",
            "31,32,33,34",
        )
    child = state_to_battle(
        switched["state"], switched["requests"], "p1", True
    )
    assert child.active_pokemon[0].species == "whimsicott"


def test_live_snapshot_gate_requires_passing_thousand_state_report():
    assert live_snapshot_supported()


def test_hidden_snapshot_accepts_unknown_opponent_items():
    our_team = (ROOT / "teams/reg_mb/our_team.txt").read_text()
    their_team = (ROOT / "teams/reg_mb/MB1.txt").read_text()
    with ExactShowdownBridge() as bridge:
        root = bridge.create(
            formatid=FORMAT,
            seed=[401, 402, 403, 404],
            p1_team_text=our_team,
            p2_team_text=their_team,
            p1_preview="team 1,2,3,4",
            p2_preview="team 1,2,3,4",
        )
    battle = state_to_battle(
        root["state"], root["requests"], "p1", reveal_opponent_sets=False
    )
    snapshot = public_snapshot(
        battle,
        root["requests"][0],
        request_state="move",
        side_requests=root["requests"],
    )
    assert len(snapshot["sides"][1]["pokemon"]) == 2


def test_refreshed_opponent_previews_keep_every_revealed_pokemon():
    def mon(species, revealed):
        return SimpleNamespace(
            species=species,
            base_species=species,
            revealed=revealed,
        )

    active_a = mon("Garchomp", True)
    active_b = mon("Whimsicott", True)
    revealed_back = mon("Tyranitar", True)
    hidden = [mon("Excadrill", False), mon("Rotom-Wash", False), mon("Sylveon", False)]
    battle = SimpleNamespace(
        opponent_active_pokemon=[active_a, active_b],
        opponent_team={
            "p2: Garchomp": active_a,
            "p2: Whimsicott": active_b,
            "p2: Tyranitar": revealed_back,
            "p2: Excadrill": hidden[0],
            "p2: Rotom": hidden[1],
            "p2: Sylveon": hidden[2],
        },
    )
    roster = tuple(
        TeamSlot(species.lower().replace("-", ""), species)
        for species in (
            "Garchomp",
            "Whimsicott",
            "Tyranitar",
            "Excadrill",
            "Rotom-Wash",
            "Sylveon",
        )
    )

    previews = _opponent_previews(battle, roster)

    assert len(previews) == 3
    assert all(
        brought[:3] == ("garchomp", "whimsicott", "tyranitar")
        for _, brought in previews
    )
    assert all(len(set(brought)) == 4 for _, brought in previews)


def test_refreshed_our_preview_maps_mega_back_to_roster_species():
    def mon(species, base_species, selected=True):
        return SimpleNamespace(
            species=species,
            base_species=base_species,
            selected_in_teampreview=selected,
            _selected_in_teampreview=selected,
        )

    charizard = mon("Charizard-Mega-Y", "Charizard")
    garchomp = mon("Garchomp", "Garchomp")
    whimsicott = mon("Whimsicott", "Whimsicott")
    basculegion = mon("Basculegion", "Basculegion")
    battle = SimpleNamespace(
        active_pokemon=[charizard, garchomp],
        team={
            "p1: Charizard": charizard,
            "p1: Garchomp": garchomp,
            "p1: Whimsicott": whimsicott,
            "p1: Basculegion": basculegion,
        },
    )
    roster = tuple(
        TeamSlot(species.lower().replace("-", ""), species)
        for species in (
            "Floette-Eternal",
            "Charizard",
            "Whimsicott",
            "Garchomp",
            "Basculegion",
            "Kingambit",
        )
    )

    assert _our_preview(battle, roster) == "team 2,4,3,5"


def test_refreshed_our_preview_preserves_mega_nickname_identity():
    mega = SimpleNamespace(
        species="Floette-Mega",
        base_species="Floette-Mega",
        selected_in_teampreview=True,
        _selected_in_teampreview=True,
    )
    charizard = SimpleNamespace(
        species="Charizard",
        base_species="Charizard",
        selected_in_teampreview=True,
        _selected_in_teampreview=True,
    )
    whimsicott = SimpleNamespace(
        species="Whimsicott",
        base_species="Whimsicott",
        selected_in_teampreview=True,
        _selected_in_teampreview=True,
    )
    garchomp = SimpleNamespace(
        species="Garchomp",
        base_species="Garchomp",
        selected_in_teampreview=True,
        _selected_in_teampreview=True,
    )
    battle = SimpleNamespace(
        active_pokemon=[mega, charizard],
        team={
            "p1: Floette": mega,
            "p1: Charizard": charizard,
            "p1: Whimsicott": whimsicott,
            "p1: Garchomp": garchomp,
        },
    )
    roster = tuple(
        TeamSlot(species, species)
        for species in (
            "floetteeternal",
            "charizard",
            "whimsicott",
            "garchomp",
            "basculegion",
            "kingambit",
        )
    )

    assert _our_preview(
        battle, roster, {"floette": "floetteeternal"}
    ) == "team 1,2,3,4"


def test_refreshed_opponent_preview_preserves_transformed_identity():
    transformed_ditto = SimpleNamespace(
        species="Metagross", base_species="Metagross", revealed=True
    )
    metagross = SimpleNamespace(
        species="Metagross", base_species="Metagross", revealed=True
    )
    reserves = [
        SimpleNamespace(species=name, base_species=name, revealed=False)
        for name in ("Whimsicott", "Charizard", "Incineroar", "Kommo-o")
    ]
    battle = SimpleNamespace(
        opponent_active_pokemon=[transformed_ditto, metagross],
        opponent_team={
            "p2: Ditto": transformed_ditto,
            "p2: Metagross": metagross,
            **{f"p2: {mon.species}": mon for mon in reserves},
        },
    )
    roster = tuple(
        TeamSlot(species, species)
        for species in (
            "ditto",
            "metagross",
            "whimsicott",
            "charizard",
            "incineroar",
            "kommoo",
        )
    )

    previews = _opponent_previews(
        battle,
        roster,
        {"ditto": "ditto", "metagross": "metagross"},
    )

    assert len(previews) == 6
    assert all(brought[:2] == ("ditto", "metagross") for _, brought in previews)


def test_redundant_tailwind_is_strictly_rejected_after_exact_search():
    our_team = (ROOT / "teams/reg_mb/our_team.txt").read_text()
    their_team = (ROOT / "teams/reg_mb/MB1.txt").read_text()
    with ExactShowdownBridge() as bridge:
        root = bridge.create(
            formatid=FORMAT,
            seed=[451, 452, 453, 454],
            p1_team_text=our_team,
            p2_team_text=their_team,
            p1_preview="team 3,5,1,2",
            p2_preview="team 1,2,3,4",
        )
    battle = state_to_battle(root["state"], root["requests"], "p1", True)
    battle.side_conditions[SideCondition.TAILWIND] = 1
    actions = choice_to_actions(
        "move tailwind, move wavecrash +1",
        root["requests"][0],
        state=root["state"],
        role="p1",
        battle=battle,
    )

    assert (
        LiveExactSession._strict_live_rejection_reason(battle, actions)
        == "redundant_side_condition"
    )


def test_cached_heat_wave_is_rejected_when_sun_weather_ball_dominates(monkeypatch):
    from vgc_bench.src import guards as guards_module

    monkeypatch.setattr(
        guards_module,
        "candidate_gives_catastrophic_free_setup",
        lambda _battle, _candidate: False,
    )
    monkeypatch.setattr(
        guards_module,
        "candidate_repeats_solo_protect",
        lambda _battle, _candidate: False,
    )
    monkeypatch.setattr(
        guards_module,
        "candidate_exposes_protect_to_encore",
        lambda _battle, _candidate: False,
    )
    monkeypatch.setattr(
        guards_module,
        "candidate_uses_dominated_weather_ball",
        lambda _battle, _candidate: False,
    )
    monkeypatch.setattr(
        guards_module,
        "candidate_uses_dominated_single_target_heat_wave",
        lambda _battle, _candidate: True,
    )

    assert (
        LiveExactSession._strict_live_rejection_reason(SimpleNamespace(), (9, 20))
        == "single_target_weather_ball"
    )


def test_hidden_reveal_prunes_impossible_brought_four_world():
    particle = SetParticle(
        species="garchomp",
        ability="roughskin",
        item="sitrusberry",
        moves=("earthquake",),
        spread=None,
        probability=1.0,
        source="test",
    )
    node = ExactNode(
        state={"sides": [{"pokemon": []}, {"pokemon": []}]},
        requests=[{}, {}],
        turn=2,
        request_state="move",
    )
    session = object.__new__(LiveExactSession)
    session.open_sheet = False
    session.eliminated_roots = 0
    session.roots = [
        LiveRoot(node, 0.5, {"garchomp": particle}, ("a", "b", "garchomp", "d"), "has"),
        LiveRoot(node, 0.5, {"garchomp": particle}, ("a", "b", "c", "d"), "missing"),
    ]
    session._revealed_evidence = lambda _battle: {
        "garchomp": {"moves": {"earthquake"}, "item": None, "ability": None}
    }
    session._condition_roots(SimpleNamespace())
    assert [root.label for root in session.roots] == ["has"]
    assert session.roots[0].probability == pytest.approx(1.0)
    assert session.eliminated_roots == 1


class _ChoicesBridge:
    def choices(self, _state, _role):
        return ["next"]


def _empty_exact_state(weather=""):
    return {
        "field": {"weather": weather, "terrain": "", "pseudoWeather": {}},
        "sides": [
            {"pokemon": [], "sideConditions": {}},
            {"pokemon": [], "sideConditions": {}},
        ],
    }


def test_selective_search_reuses_only_consensus_matching_continuation():
    parent = ExactNode(_empty_exact_state(), [{}, {}], 1, "move")
    predicted = ExactNode(_empty_exact_state(), [{}, {}], 2, "move")
    continuation = BranchContinuation(
        root_choice="ours",
        opponent_choice="theirs",
        next_choice="next",
        value=0.4,
        margin=0.2,
        probability=1.0,
        predicted_node=predicted,
        root_label="world",
    )
    session = object.__new__(LiveExactSession)
    session.selective_search = True
    session.planned_continuations = (continuation,)
    session.plan_parent_nodes = {"world": parent}
    session.last_executed_choice = "ours"
    session.planned_root_margin = 0.1
    session.last_snapshot = {"sides": [{"pokemon": []}, {"pokemon": []}]}
    session.last_observed_actions = {}
    session.roots = [LiveRoot(predicted, 1.0, {}, (), "world")]
    session.bridge = _ChoicesBridge()
    session._live_actions = lambda _choice, _battle: (7, 9)
    session.reused_plans = 0
    session.plan_parent_nodes = {"world": parent}

    result = session._reuse_contingent_plan(SimpleNamespace())

    assert result is not None
    assert result.choice == "next"
    assert result.actions == (7, 9)
    assert result.fallback_reason == "reused_contingent_plan"
    assert session.last_schedule["mode"] == "reuse"


def test_selective_search_rejects_continuation_after_strategic_divergence():
    parent = ExactNode(_empty_exact_state(), [{}, {}], 1, "move")
    predicted = ExactNode(_empty_exact_state("raindance"), [{}, {}], 2, "move")
    actual = ExactNode(_empty_exact_state("sunnyday"), [{}, {}], 2, "move")
    continuation = BranchContinuation(
        "ours", "theirs", "next", 0.4, 0.2, 1.0, predicted, "world"
    )
    session = object.__new__(LiveExactSession)
    session.selective_search = True
    session.planned_continuations = (continuation,)
    session.plan_parent_nodes = {"world": parent}
    session.last_executed_choice = "ours"
    session.planned_root_margin = 0.1
    session.last_snapshot = {"sides": [{"pokemon": []}, {"pokemon": []}]}
    session.last_observed_actions = {}
    session.roots = [LiveRoot(actual, 1.0, {}, (), "world")]
    session.bridge = _ChoicesBridge()

    assert session._reuse_contingent_plan(SimpleNamespace()) is None


def test_selective_search_recognizes_acceptable_evaluated_position():
    parent = ExactNode(_empty_exact_state(), [{}, {}], 1, "move")
    predicted = ExactNode(_empty_exact_state(), [{}, {}], 2, "move")
    outcome = BranchOutcome(
        root_choice="ours",
        opponent_choice="theirs",
        value=0.35,
        probability=1.0,
        predicted_node=predicted,
        searched_depth=2,
        root_label="world",
    )
    session = object.__new__(LiveExactSession)
    session.selective_search = True
    session.planned_outcomes = (outcome,)
    session.plan_parent_nodes = {"world": parent}
    session.last_executed_choice = "ours"
    session.planned_root_margin = 0.1
    session.planned_reference_score = 0.4
    session.last_snapshot = {"sides": [{"pokemon": []}, {"pokemon": []}]}
    session.last_observed_actions = {}
    session.roots = [LiveRoot(predicted, 1.0, {}, (), "world")]

    match = session._matching_planned_outcome()

    assert match is not None
    assert match["coverage"] == pytest.approx(1.0)
    assert match["predicted_value"] == pytest.approx(0.35)


def test_selective_search_recognizes_safe_background_ponder_branch():
    parent = ExactNode(_empty_exact_state(), [{}, {}], 1, "move")
    predicted = ExactNode(_empty_exact_state(), [{}, {}], 2, "move")
    outcome = PonderOutcome(
        root_label="world",
        root_choice="ours",
        opponent_choice="theirs",
        predicted_node=predicted,
        value=0.1,
        probability=1.0,
    )
    session = object.__new__(LiveExactSession)
    session.enable_ponder = True
    session.pondered_outcomes = (outcome,)
    session.ponder_parent_nodes = {"world": parent}
    session.ponder_reference_values = {"world": 0.0}
    session.last_executed_choice = "ours"
    session.last_snapshot = {"sides": [{"pokemon": []}, {"pokemon": []}]}
    session.last_observed_actions = {}
    session.roots = [LiveRoot(predicted, 1.0, {}, (), "world")]
    session.ponder_matches = 0

    match = session._matching_pondered_outcome()

    assert match is not None
    assert match["coverage"] == pytest.approx(1.0)
    assert session.ponder_matches == 1


def test_selective_search_skips_quiet_turn_but_periodically_refreshes():
    node = ExactNode(_empty_exact_state(), [{}, {}], 2, "move")
    session = object.__new__(LiveExactSession)
    session.selective_search = True
    session.last_recent_events = [
        ["", "move", "p2a: Foo", "Tackle", "p1a: Bar"],
        ["", "-damage", "p1a: Bar", "80/100"],
    ]
    session.last_search_turn = 1
    session.last_snapshot = {"sides": [{"pokemon": []}, {"pokemon": []}]}
    session.roots = [LiveRoot(node, 1.0, {}, (), "world")]

    assert session._importance_reasons(
        SimpleNamespace(turn=2, force_switch=[False, False])
    ) == []
    assert session._importance_reasons(
        SimpleNamespace(turn=3, force_switch=[False, False])
    ) == []
    assert "periodic_refresh" in session._importance_reasons(
        SimpleNamespace(turn=4, force_switch=[False, False])
    )


def test_selective_search_ignores_weather_upkeep_but_not_a_real_change():
    upkeep = [["", "-weather", "RainDance", "[upkeep]"]]
    changed = [["", "-weather", "SunnyDay", "[from] ability: Drought"]]

    assert LiveExactSession._critical_event_reasons(upkeep) == []
    assert LiveExactSession._critical_event_reasons(changed) == ["weather_changed"]


def test_planned_opponent_action_coverage_ignores_damage_roll_drift():
    parent = ExactNode(_empty_exact_state(), [{}, {}], 1, "move")
    predicted = ExactNode(_empty_exact_state(), [{}, {}], 2, "move")
    outcome = BranchOutcome(
        root_choice="ours",
        opponent_choice="move tackle +1, move protect",
        value=0.2,
        probability=1.0,
        predicted_node=predicted,
        searched_depth=2,
        root_label="world",
    )
    session = object.__new__(LiveExactSession)
    session.last_executed_choice = "ours"
    session.last_observed_actions = {0: ObservedAction("move", "tackle", 1, False)}
    session.plan_parent_nodes = {"world": parent}
    session.planned_outcomes = (outcome,)
    session.planned_continuations = ()
    session.roots = [LiveRoot(predicted, 1.0, {}, (), "world")]

    assert session._planned_opponent_action_coverage() == pytest.approx(1.0)

    session.last_observed_actions = {0: ObservedAction("move", "growl", 1, False)}
    assert session._planned_opponent_action_coverage() == 0.0


def test_planned_coverage_is_relative_to_representative_searched_worlds():
    parent = ExactNode(_empty_exact_state(), [{}, {}], 1, "move")
    predicted = ExactNode(_empty_exact_state(), [{}, {}], 2, "move")
    outcome = BranchOutcome(
        root_choice="ours",
        opponent_choice="move tackle +1, move protect",
        value=0.2,
        probability=1.0,
        predicted_node=predicted,
        searched_depth=2,
        root_label="searched",
    )
    session = object.__new__(LiveExactSession)
    session.last_executed_choice = "ours"
    session.last_observed_actions = {0: ObservedAction("move", "tackle", 1, False)}
    session.plan_parent_nodes = {"searched": parent}
    session.planned_outcomes = (outcome,)
    session.planned_continuations = ()
    session.roots = [
        LiveRoot(predicted, 0.25, {}, (), "searched"),
        LiveRoot(predicted, 0.75, {}, (), "belief-only"),
    ]

    assert session._planned_opponent_action_coverage() == pytest.approx(1.0)
    assert session._planned_opponent_family_coverage() == pytest.approx(1.0)
    assert session._planned_position_coverage() == pytest.approx(1.0)

    session.last_observed_actions = {0: ObservedAction("move", "growl", 1, False)}
    assert session._planned_opponent_action_coverage() == 0.0
    assert session._planned_opponent_family_coverage() == 0.0
    assert session._planned_position_coverage() == 0.0


def test_family_coverage_accepts_individually_searched_joint_moves():
    parent = ExactNode(_empty_exact_state(), [{}, {}], 1, "move")
    predicted = ExactNode(_empty_exact_state(), [{}, {}], 2, "move")

    def outcome(choice):
        return BranchOutcome(
            root_choice="ours",
            opponent_choice=choice,
            value=0.2,
            probability=0.5,
            predicted_node=predicted,
            searched_depth=2,
            root_label="world",
        )

    session = object.__new__(LiveExactSession)
    session.last_executed_choice = "ours"
    session.last_observed_actions = {
        0: ObservedAction("move", "tackle", 1, False),
        1: ObservedAction("move", "protect", None, False),
    }
    session.plan_parent_nodes = {"world": parent}
    session.planned_outcomes = (
        outcome("move tackle +1, move growl"),
        outcome("move scratch +1, move protect"),
    )
    session.planned_continuations = ()
    session.roots = [LiveRoot(predicted, 1.0, {}, (), "world")]

    assert session._planned_opponent_action_coverage() == 0.0
    assert session._planned_opponent_family_coverage() == pytest.approx(1.0)


def test_expected_searched_action_skips_only_soft_board_updates():
    assert LiveExactSession._expected_action_is_quiet(
        ["stat_stage_changed", "weather_changed"], 0.90
    )
    assert not LiveExactSession._expected_action_is_quiet(
        ["stat_stage_changed", "faint"], 0.90
    )
    assert not LiveExactSession._expected_action_is_quiet(
        ["stat_stage_changed"], 0.74
    )


def test_refresh_turn_uses_one_foreground_world_then_restores_normal_count():
    session = object.__new__(LiveExactSession)
    session.search_determinizations = 2
    session.last_root_refresh_turn = 6
    assert session._foreground_determinizations(6) == 1
    assert session._foreground_determinizations(7) == 2


def test_shallow_low_prior_override_requires_deeper_coverage_or_tactical_reason(
    monkeypatch,
):
    def score(choice, prior, coverage):
        return ActionScore(
            choice=choice,
            actions=(1, 1),
            score=0.8,
            expected=0.8,
            cvar=0.8,
            worst=0.8,
            standard_deviation=0.0,
            prior=prior,
            opponent_branches=1,
            depth_coverage=coverage,
        )

    session = object.__new__(LiveExactSession)
    session.low_prior_override_ratio = 0.10
    session.low_prior_override_min_coverage = 0.90
    session.last_live_guards = {"changed_pick": False}
    winner = score("move aquajet +1, move moonblast +1", 0.004, 0.73)
    policy_line = score("move wavecrash +1, move moonblast +1", 0.58, 0.73)
    monkeypatch.setattr(
        session,
        "_priority_ko_justifies_low_prior_override",
        lambda *_args: False,
    )

    rejection = session._low_prior_override_rejection(
        SimpleNamespace(), winner, (winner, policy_line)
    )
    assert rejection is not None
    assert rejection["reason"] == "shallow_low_prior_override"

    deep = replace(winner, depth_coverage=0.95)
    deep_rejection = session._low_prior_override_rejection(
        SimpleNamespace(), deep, (deep, policy_line)
    )
    assert deep_rejection is not None
    assert deep_rejection["reason"] == "unsupported_low_prior_override"

    supported = replace(deep, prior=0.10)
    assert (
        session._low_prior_override_rejection(
            SimpleNamespace(), supported, (supported, policy_line)
        )
        is None
    )
    monkeypatch.setattr(
        session,
        "_priority_ko_justifies_low_prior_override",
        lambda *_args: True,
    )
    assert (
        session._low_prior_override_rejection(
            SimpleNamespace(), winner, (winner, policy_line)
        )
        is None
    )


def test_live_safety_replacement_must_retain_future_depth_coverage():
    session = object.__new__(LiveExactSession)
    session.min_deep_coverage = 0.50
    shallow = ActionScore(
        choice="move wavecrash +1, move protect",
        actions=(10, 24),
        score=0.8,
        expected=0.8,
        cvar=0.8,
        worst=0.8,
        standard_deviation=0.0,
        prior=0.4,
        opponent_branches=1,
        depth_coverage=0.0,
    )

    assert not session._reaches_required_depth(shallow)
    assert session._reaches_required_depth(replace(shallow, depth_coverage=0.50))


def test_live_ranking_discards_stale_trapped_switch_and_keeps_next_choice():
    def score(choice, value):
        return ActionScore(
            choice=choice,
            actions=None,
            score=value,
            expected=value,
            cvar=value,
            worst=value,
            standard_deviation=0.0,
            prior=0.5,
            opponent_branches=1,
        )

    result = PlanResult(
        choice="switch 3, move protect",
        actions=None,
        score=0.4,
        rankings=(
            score("switch 3, move protect", 0.4),
            score("move earthquake, move protect", 0.2),
        ),
        nodes=10,
        elapsed_s=1.0,
        completed_depth=2,
        truncated=False,
    )
    session = object.__new__(LiveExactSession)

    def map_choice(choice, _battle):
        if choice.startswith("switch"):
            raise ValueError("trapped")
        return (9, 24)

    session._live_actions = map_choice
    rankings, illegal = session._live_legal_rankings(result, SimpleNamespace())
    assert [row.choice for row in rankings] == ["move earthquake, move protect"]
    assert rankings[0].actions == (9, 24)
    assert illegal == ["switch 3, move protect"]


def test_live_ranking_discards_redundant_side_condition_choice():
    def score(choice, value):
        return ActionScore(
            choice=choice,
            actions=None,
            score=value,
            expected=value,
            cvar=value,
            worst=value,
            standard_deviation=0.0,
            prior=0.5,
            opponent_branches=1,
        )

    result = PlanResult(
        choice="move tailwind, move wavecrash +1",
        actions=None,
        score=0.9,
        rankings=(
            score("move tailwind, move wavecrash +1", 0.9),
            score("move moonblast +1, move wavecrash +1", 0.4),
        ),
        nodes=10,
        elapsed_s=1.0,
        completed_depth=2,
        truncated=False,
    )
    session = object.__new__(LiveExactSession)
    session._live_actions = lambda choice, _battle: (
        (11, 12) if "tailwind" in choice else (21, 22)
    )
    session._strict_live_rejection_reason = lambda _battle, actions: (
        "redundant_side_condition" if actions == (11, 12) else None
    )
    session._apply_live_hard_guards = lambda _battle, rankings: rankings

    rankings, illegal = session._live_legal_rankings(result, SimpleNamespace())

    assert [row.choice for row in rankings] == [
        "move moonblast +1, move wavecrash +1"
    ]
    assert illegal == []
    assert session.last_live_guards["strict_rejections"] == [
        "move tailwind, move wavecrash +1 [redundant_side_condition]"
    ]
