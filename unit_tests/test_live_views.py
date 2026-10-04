"""Live-anchored views (2026-10-04): the exact search sees our side of a shadow as
the live battle (at the root) or a copy of it that watched the shadow's new protocol
(a child), not a battle rebuilt from the shadow's own log. The rebuilt view matched
the live one in 0 of 77 probed decisions."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from poke_env.battle import Pokemon

from vgc_bench.src.exact_observation import (
    LiveAnchor,
    _restore_battle_state,
    _state_species_id,
    choice_to_actions,
    lines_since_reconcile,
    live_view,
    state_to_battle,
    to_live_line,
)
from vgc_bench.src.exact_sim import ExactShowdownBridge
from vgc_bench.src.live_snapshot import _public_item, public_snapshot

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "gen9championsvgc2026regmc"


def _anchor(player_role, names=None) -> LiveAnchor:
    battle: Any = SimpleNamespace(player_role=player_role)
    return LiveAnchor(battle, names or {})


def test_shadow_lines_keep_their_seats_when_we_are_seat_one():
    line = "|move|p1a: Blastoise|Ice Beam|p2b: Farigiraf"
    assert to_live_line(line, _anchor("p1")) == line


def test_shadow_lines_swap_seats_when_the_server_seated_us_second():
    anchor = _anchor("p2")
    assert (
        to_live_line("|move|p1a: Blastoise|Ice Beam|p2b: Farigiraf", anchor)
        == "|move|p2a: Blastoise|Ice Beam|p1b: Farigiraf"
    )
    assert (
        to_live_line(
            "|-weather|SunnyDay|[from] ability: Drought|[of] p2a: Torkoal", anchor
        )
        == "|-weather|SunnyDay|[from] ability: Drought|[of] p1a: Torkoal"
    )
    assert (
        to_live_line("|-sidestart|p1: planner|move: Tailwind", anchor)
        == "|-sidestart|p2: planner|move: Tailwind"
    )
    # nothing that is not a seat is touched
    assert to_live_line("|-damage|p2b: Farigiraf|48/100", anchor) == (
        "|-damage|p1b: Farigiraf|48/100"
    )
    assert to_live_line("|turn|4", anchor) == "|turn|4"
    assert to_live_line("|-fieldstart|move: Trick Room", anchor) == (
        "|-fieldstart|move: Trick Room"
    )


def test_shadow_lines_use_the_opponents_live_nicknames():
    anchor = _anchor("p1", {"incineroar": "Bob", "rotomwash": "Wash Day"})
    assert (
        to_live_line("|switch|p2a: Incineroar|Incineroar, L50, M|100/100", anchor)
        == "|switch|p2a: Bob|Incineroar, L50, M|100/100"
    )
    assert (
        to_live_line("|move|p2b: Rotom-Wash|Hydro Pump|p1a: Incineroar", anchor)
        == "|move|p2b: Wash Day|Hydro Pump|p1a: Incineroar"  # ours keeps its name
    )
    flipped = _anchor("p2", {"incineroar": "Bob"})
    assert to_live_line("|faint|p2a: Incineroar", flipped) == "|faint|p1a: Bob"


def test_lines_since_reconcile_start_after_the_last_marker():
    log = [
        "|turn|1",
        "|vgcsnapshot|{}",
        "|move|x",
        "|vgcsnapshot|{}",
        "|move|y",
        "|turn|3",
    ]
    assert lines_since_reconcile(log) == ["|move|y", "|turn|3"]
    assert lines_since_reconcile(log[:4]) == []
    assert lines_since_reconcile(["|turn|1"]) is None


def test_species_ids_come_out_of_the_serialized_form():
    assert _state_species_id("[Species:blastoisemega]") == "blastoisemega"
    assert _state_species_id("Rotom-Wash") == "rotomwash"
    assert _state_species_id(None) == ""


def test_sheet_reset_does_not_undo_a_mega_or_return_a_spent_item():
    blastoise = Pokemon(gen=9, species="blastoise")
    blastoise._item = "blastoisinite"
    _restore_battle_state(
        blastoise,
        {
            "species": "[Species:blastoisemega]",
            "baseSpecies": "[Species:blastoisemega]",
            "item": "blastoisinite",
            "ability": "megalauncher",
        },
    )
    # exactly what poke-env does with the -mega line of a live opponent: the Mega's
    # stats, types and ability under the name it already had
    assert blastoise.species == "blastoise"
    assert blastoise.base_stats["spa"] == 135
    assert blastoise.ability == "megalauncher"

    farigiraf = Pokemon(gen=9, species="farigiraf")
    farigiraf._item = "sitrusberry"  # the sheet's item, eaten two turns ago
    _restore_battle_state(
        farigiraf,
        {"species": "[Species:farigiraf]", "item": "", "ability": "armortail"},
    )
    assert farigiraf.species == "farigiraf" and farigiraf.item is None

    # a battle-only forme changes stats and types, not the name poke-env keeps
    aegislash = Pokemon(gen=9, species="aegislash")
    _restore_battle_state(
        aegislash,
        {"species": "[Species:aegislashblade]", "baseSpecies": "[Species:aegislash]"},
    )
    assert aegislash.species == "aegislash" and aegislash.base_stats["atk"] == 140


def test_an_unrevealed_item_is_not_passed_to_the_shadow_as_an_item():
    """2026-10-04: poke-env's "unknown_item" placeholder overwrote the sampled item
    of every active opponent in every hidden-sheet shadow."""
    quiet: Any = SimpleNamespace(_replay_data=[])
    assert _public_item(quiet, "torkoal", "p2", "unknown_item", own=False) is None
    assert _public_item(quiet, "torkoal", "p2", None, own=False) is None
    assert _public_item(quiet, "torkoal", "p2", "charcoal", own=False) == "charcoal"
    eaten: Any = SimpleNamespace(
        _replay_data=[["", "-enditem", "p2a: Torkoal", "Sitrus Berry", "[eat]"]]
    )
    assert _public_item(eaten, "torkoal", "p2", "unknown_item", own=False) == ""


def _reconciled_root(bridge, seat):
    """A live battle in ``seat`` and a shadow just reconciled to it (a T6e mirror)."""
    team = (ROOT / "teams/candidates_mc/T6e.txt").read_text()

    def create(seed):
        return bridge.create(
            formatid=FORMAT,
            seed=seed,
            p1_team_text=team,
            p2_team_text=team,
            p1_preview="team 1234",
            p2_preview="team 1234",
        )

    source = create([1, 2, 3, 4])  # the "server": we sit in ``seat``
    live = state_to_battle(source["state"], source["requests"], seat, False)
    request = source["requests"][int(seat[1]) - 1]
    snapshot = public_snapshot(live, request, request_state="move")
    shadow = bridge.reconcile(create([5, 6, 7, 8])["state"], snapshot)
    return live, shadow


@pytest.mark.parametrize("seat", ["p1", "p2"])
def test_root_view_is_the_live_battle_and_a_child_is_a_copy_that_watched(seat):
    with ExactShowdownBridge() as bridge:
        live, shadow = _reconciled_root(bridge, seat)
        anchor = LiveAnchor(live, {})
        assert live.player_role == seat
        assert live_view(anchor, shadow["state"], shadow["requests"][0]) is live

        ours = next(
            c
            for c in bridge.choices(shadow["state"], "p1")
            if c.startswith("move icebeam +2") and "trickroom" in c
        )
        theirs = next(
            c
            for c in bridge.choices(shadow["state"], "p2")
            if c.startswith("move waterpulse +2") and "helpinghand" in c
        )
        child = bridge.simulate(shadow["state"], ours, theirs, "9,9,9,9")
        view: Any = live_view(anchor, child["state"], child["requests"][0])

    assert view is not None and view is not live
    assert live.turn == 1 and view.turn == 2  # the live battle itself is untouched
    assert view.player_role == seat
    # the same Pokemon in the same order as the live battle holds them
    assert list(view.team) == list(live.team)
    assert list(view.team)[0].startswith(seat)
    exact = child["state"]["sides"]
    # our Farigiraf took the Water Pulse: exact HP, as our own side always has
    ours_hit = next(
        p for p in exact[0]["pokemon"] if p["set"]["species"] == "Farigiraf"
    )
    mine = view.active_pokemon[1]
    assert mine is not None and mine.species == "farigiraf"
    assert mine.current_hp == ours_hit["hp"] < ours_hit["maxhp"]
    # their Farigiraf took the Ice Beam: a public percentage
    theirs_hit = next(
        p for p in exact[1]["pokemon"] if p["set"]["species"] == "Farigiraf"
    )
    foe = view.opponent_active_pokemon[1]
    assert foe is not None and foe.species == "farigiraf"
    assert foe.current_hp_fraction == pytest.approx(
        theirs_hit["hp"] / theirs_hit["maxhp"], abs=0.011
    )
    assert foe.current_hp_fraction < 1.0
    assert "icebeam" in view.active_pokemon[0].moves
    assert not view.active_pokemon[0].first_turn


def test_a_state_that_was_never_reconciled_has_no_live_view():
    with ExactShowdownBridge() as bridge:
        team = (ROOT / "teams/candidates_mc/T6e.txt").read_text()
        fresh = bridge.create(
            formatid=FORMAT,
            p1_team_text=team,
            p2_team_text=team,
            p1_preview="team 1234",
            p2_preview="team 1234",
        )
        live = state_to_battle(fresh["state"], fresh["requests"], "p1", False)
    anchor = LiveAnchor(live, {})
    assert live_view(anchor, fresh["state"], fresh["requests"][0]) is None
    assert live_view(anchor, {"log": ["|vgcsnapshot|{}"]}, None) is None


def _t6e_root(bridge):
    team = (ROOT / "teams/candidates_mc/T6e.txt").read_text()
    root = bridge.create(
        formatid="gen9championsvgc2026regmc",
        p1_team_text=team,
        p2_team_text=team,
        p1_preview="team 1234",
        p2_preview="team 1234",
    )
    battle = state_to_battle(root["state"], root["requests"], "p1", True)
    snapshot = public_snapshot(
        battle,
        root["requests"][0],
        request_state="move",
        side_requests=root["requests"],
    )
    return root, snapshot


def test_reconcile_moves_our_last_pokemon_to_the_other_slot_without_cloning_it():
    """2026-10-04: a Pokemon that stood in the shadow's slot a and is live in slot b
    was written into BOTH active slots. Every joint choice then gave it two moves,
    none could be encoded, and the search failed -- in about 3% of all decisions, all
    of them one-Pokemon endgames."""
    with ExactShowdownBridge() as bridge:
        root, snapshot = _t6e_root(bridge)
        for row in snapshot["sides"][0]["pokemon"]:
            if row["active_slot"] == 0:
                row["active_slot"] = 1  # our last Pokemon, live in slot b
            else:
                row.update(active_slot=None, fainted=True, hp=0, hp_fraction=0.0)
                row["moves"] = []
        repaired = bridge.reconcile(root["state"], snapshot)
        state = repaired["state"]
        assert state["sides"][0]["active"] == ["[Pokemon:p1a]", "[Pokemon:p1b]"]
        choices = bridge.choices(state, "p1")
        assert choices and all(choice.startswith("pass, move ") for choice in choices)
        battle: Any = state_to_battle(state, repaired["requests"], "p1", True)
        assert battle.active_pokemon[0] is None
        assert battle.active_pokemon[1].species == "blastoise"
        encoded = {
            choice_to_actions(
                choice, repaired["requests"][0], state=state, role="p1", battle=battle
            )
            for choice in choices
        }
        assert encoded and all(pair[0] == 0 and pair[1] != 0 for pair in encoded)


def test_reconcile_swaps_two_live_actives_and_keeps_their_slots_consistent():
    with ExactShowdownBridge() as bridge:
        root, snapshot = _t6e_root(bridge)
        for row in snapshot["sides"][0]["pokemon"]:
            if row["active_slot"] is not None:
                row["active_slot"] = 1 - row["active_slot"]
        repaired = bridge.reconcile(root["state"], snapshot)
        side = repaired["state"]["sides"][0]
        assert side["active"] == ["[Pokemon:p1a]", "[Pokemon:p1b]"]
        assert [mon["set"]["species"] for mon in side["pokemon"][:2]] == [
            "Farigiraf",
            "Blastoise",
        ]
        assert [mon["position"] for mon in side["pokemon"]] == [0, 1, 2, 3]
        choice = next(
            c
            for c in bridge.choices(repaired["state"], "p1")
            if c.split(",")[1].strip() == "move icebeam +1"
        )
        result = bridge.simulate(
            repaired["state"], choice, bridge.choices(repaired["state"], "p2")[0]
        )
        assert any(
            line.startswith("|move|p1b: Blastoise|Ice Beam|p2a:")
            for line in result["log"]
        )


def test_reconcile_gives_a_shadow_the_mega_it_never_played():
    """2026-10-04: a shadow recreated mid-battle got the Mega's stats but stayed
    "Blastoise, L50" in every request, so the policy and the critic saw the base
    forme at that root and in all its children, and it reverted on its first switch.
    """
    with ExactShowdownBridge() as bridge:
        root, snapshot = _t6e_root(bridge)
        for side in snapshot["sides"]:
            for row in side["pokemon"]:
                if row["nickname"] == "blastoise":
                    row["species"] = "blastoisemega"
                    row["ability"] = "megalauncher"
            side["mechanic_usage"]["mega_used"] = True
        repaired = bridge.reconcile(root["state"], snapshot)
        state = repaired["state"]
        for side in state["sides"]:
            mon = next(p for p in side["pokemon"] if p["set"]["species"] == "Blastoise")
            assert mon["species"] == "[Species:blastoisemega]"
            assert mon["baseSpecies"] == "[Species:blastoisemega]"  # survives a switch
            assert mon["details"].startswith("Blastoise-Mega")
            assert mon["ability"] == mon["baseAbility"] == "megalauncher"
            assert mon["hp"] == mon["maxhp"]
        assert not any("mega" in choice for choice in bridge.choices(state, "p1"))
        for role in ("p1", "p2"):  # each side's own view of its Mega
            view: Any = state_to_battle(state, repaired["requests"], role, True)
            assert view.active_pokemon[0].species == "blastoisemega"
            assert view.active_pokemon[0].ability == "megalauncher"
            # ... and of the other side's, as poke-env shows a live opponent's Mega
            assert view.opponent_active_pokemon[0].base_stats["spa"] == 135
        ours = next(c for c in bridge.choices(state, "p1") if "waterpulse +2" in c)
        child = bridge.simulate(state, ours, bridge.choices(state, "p2")[0], "1,2,3,4")
        after: Any = state_to_battle(child["state"], child["requests"], "p1", True)
        assert after.active_pokemon[0].species == "blastoisemega"
