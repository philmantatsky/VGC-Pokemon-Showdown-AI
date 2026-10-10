"""Worlds rebuilt from the public snapshot on a turn a Pokemon has no choice.

Sixty-five ladder games with the search on 2026-10-10: nine decisions fell back to the
bot's own move on an error, eight of them of one kind -- a move a Pokemon is forced
into, spelled without a target in a rebuilt world, which Showdown rejects when the
choice is submitted by id. The recharge turn had been closed the same morning
(``test_bridge_recharge.py``); these are the others.

* An opponent between the two turns of a move (Phantom Force three times, Solar Beam,
  Electro Shot). Its charge turn shows no target, so the snapshot names none. And the
  snapshot names one volatile where Showdown keeps two: without the one named after the
  move, the Pokemon can be hit while it has vanished and starts to charge again where
  it should strike.
* One of our own Pokemon left with Struggle (Encore, then Disable on the encored move):
  the snapshot named Struggle as a charged move.
* And one decision ended on poke-env refusing a world's request outright.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from poke_env.battle import DoubleBattle

from vgc_bench.src.exact_observation import (
    ActionEncodingError,
    choice_to_actions,
    state_to_battle,
)
from vgc_bench.src.exact_sim import ExactShowdownBridge
from vgc_bench.src.live_exact import _spread_charge_targets
from vgc_bench.src.live_snapshot import public_snapshot

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "gen9championsvgc2026regmc"
CHARGERS = """Dragapult @ Life Orb
Ability: Clear Body
Level: 50
- Phantom Force
- Protect
- Dragon Darts
- U-turn

Venusaur @ Black Sludge
Ability: Overgrow
Level: 50
- Solar Beam
- Protect
- Sludge Bomb
- Sleep Powder

Garchomp @ Life Orb
Ability: Rough Skin
Level: 50
- Earthquake
- Dragon Claw
- Rock Slide
- Protect

Milotic @ Leftovers
Ability: Competitive
Level: 50
- Scald
- Ice Beam
- Recover
- Protect
"""
WALLS = """Kingambit @ Black Glasses
Ability: Defiant
Level: 50
- Protect
- Sucker Punch
- Iron Head
- Kowtow Cleave

Milotic @ Leftovers
Ability: Competitive
Level: 50
- Recover
- Protect
- Scald
- Ice Beam

Garchomp @ Life Orb
Ability: Rough Skin
Level: 50
- Earthquake
- Dragon Claw
- Rock Slide
- Protect

Farigiraf @ Sitrus Berry
Ability: Armor Tail
Level: 50
- Protect
- Trick Room
- Psychic
- Helping Hand
"""
LOCKED = """Garchomp @ Life Orb
Ability: Rough Skin
Level: 50
- Dragon Claw
- Protect
- Earthquake
- Rock Slide

Milotic @ Leftovers
Ability: Competitive
Level: 50
- Recover
- Protect
- Scald
- Ice Beam

Incineroar @ Safety Goggles
Ability: Intimidate
Level: 50
- Protect
- Flare Blitz
- Parting Shot
- Fake Out

Farigiraf @ Sitrus Berry
Ability: Armor Tail
Level: 50
- Protect
- Trick Room
- Psychic
- Helping Hand
"""
LOCKERS = """Politoed @ Leftovers
Ability: Damp
Level: 50
- Encore
- Protect
- Scald
- Icy Wind

Gengar @ Black Sludge
Ability: Cursed Body
Level: 50
- Disable
- Protect
- Shadow Ball
- Sludge Bomb

Garchomp @ Life Orb
Ability: Rough Skin
Level: 50
- Earthquake
- Dragon Claw
- Rock Slide
- Protect

Milotic @ Leftovers
Ability: Competitive
Level: 50
- Scald
- Ice Beam
- Recover
- Protect
"""


def _atoms(choice: str) -> list[str]:
    return [atom.strip() for atom in choice.split(",")]


def _moves(choices: list[str]) -> set[str]:
    """What the first slot may do other than switch."""
    return {_atoms(c)[0] for c in choices if _atoms(c)[0].startswith("move")}


def _pick(choices: list[str], first: str, second: str) -> str:
    return next(choice for choice in choices if _atoms(choice) == [first, second])


def _world(bridge, ours: str, theirs: str, seed: int = 61):
    return bridge.create(
        formatid=FORMAT,
        seed=[seed, seed + 1, seed + 2, seed + 3],
        p1_team_text=ours,
        p2_team_text=theirs,
        p1_preview="team 1,2,3,4",
        p2_preview="team 1,2,3,4",
    )


def _snapshot(world):
    battle = state_to_battle(world["state"], world["requests"], "p1", True)
    return public_snapshot(
        battle,
        world["requests"][0],
        request_state=world["request_state"],
        side_requests=world["requests"],
    )


def _volatiles(world, side: int, species: str) -> dict:
    return next(
        pokemon["volatiles"]
        for pokemon in world["state"]["sides"][side]["pokemon"]
        if species in str(pokemon.get("set", {}).get("species", "")).lower()
    )


@pytest.fixture(scope="module")
def charging():
    """Their Dragapult has vanished (Phantom Force at our slot b) and their Venusaur
    takes in light (Solar Beam at our slot a); we protected and recovered."""
    with ExactShowdownBridge() as bridge:
        live, shadow = _world(bridge, WALLS, CHARGERS), _world(bridge, WALLS, CHARGERS)
        charge = _pick(
            bridge.choices(live["state"], "p2"),
            "move phantomforce +2",
            "move solarbeam +1",
        )
        wait = _pick(
            bridge.choices(live["state"], "p1"), "move protect", "move recover"
        )
        live = bridge.simulate(live["state"], wait, charge, "71,72,73,74")
        assert live["request_state"] == "move"
        for line in ("|-prepare|p2a: Dragapult|Phantom Force", "Solar Beam||[still]"):
            assert any(line in entry for entry in live["log"]), live["log"]
        yield bridge, live, shadow, _snapshot(live)


def test_the_charge_turn_shows_no_target_and_the_snapshot_names_none(charging):
    bridge, live, _, snapshot = charging
    # the real game's choice, which the rebuilt world has to be able to spell
    assert bridge.choices(live["state"], "p2") == [
        "move phantomforce +2, move solarbeam +1"
    ]
    effects = {
        record["species"]: record["effects"]
        for record in snapshot["sides"][1]["pokemon"]
    }
    for species, move in (("dragapult", "phantomforce"), ("venusaur", "solarbeam")):
        assert effects[species]["twoturnmove"]["move"] == move
        assert effects[species]["twoturnmove"].get("target_loc") is None
    # the worlds are told one slot each, ours is left alone, nothing else is touched
    for index, slot in ((0, 1), (1, 2), (2, 1), (5, 2)):
        told = _spread_charge_targets(snapshot, index)
        aims = {
            record["species"]: record["effects"]["twoturnmove"]["target_loc"]
            for record in told["sides"][1]["pokemon"]
            if "twoturnmove" in record["effects"]
        }
        assert aims == {"dragapult": slot, "venusaur": slot}
        assert told["sides"][0] is snapshot["sides"][0]
    assert effects["dragapult"]["twoturnmove"].get("target_loc") is None  # no edit
    quiet = _snapshot(_world(bridge, WALLS, CHARGERS))
    assert _spread_charge_targets(quiet, 3) is quiet


@pytest.mark.parametrize("index", [0, 1, None])
def test_a_charging_opponent_is_rebuilt_whole_and_strikes(charging, index):
    """``index``: the world's number, which decides the slot it is told; None: a
    snapshot handed over as it is -- the simulator's own default still spells a
    choice the rules accept."""
    bridge, _, shadow, snapshot = charging
    told = snapshot if index is None else _spread_charge_targets(snapshot, index)
    slot = 1 if index is None else 1 + index % 2
    rebuilt = bridge.reconcile(shadow["state"], told)  # a world that never saw it
    locked = bridge.choices(rebuilt["state"], "p2")
    assert locked == [f"move phantomforce +{slot}, move solarbeam +{slot}"]
    # both volatiles of a move between its turns, for both Pokemon
    assert {"twoturnmove", "phantomforce"} <= set(_volatiles(rebuilt, 1, "dragapult"))
    assert {"twoturnmove", "solarbeam"} <= set(_volatiles(rebuilt, 1, "venusaur"))
    # the opponent's seat can be read and its choice encoded (before: "maps slot 0
    # to masked action 19 for p2")
    seat = state_to_battle(rebuilt["state"], rebuilt["requests"], "p2", True)
    choice_to_actions(
        locked[0],
        rebuilt["requests"][1],
        state=rebuilt["state"],
        role="p2",
        battle=seat,
    )
    ours = next(
        choice
        for choice in bridge.choices(rebuilt["state"], "p1")
        if _atoms(choice) == ["move protect", "move recover"]
    )
    played = bridge.simulate(rebuilt["state"], ours, locked[0], "81,82,83,84")
    log = "\n".join(played["log"])
    target = {1: "p1a: Kingambit", 2: "p1b: Milotic"}[slot]
    # they strike the slot they were told, and do not start to charge again
    assert f"|move|p2a: Dragapult|Phantom Force|{target}" in log, log
    assert f"|move|p2b: Venusaur|Solar Beam|{target}" in log, log
    assert "|-prepare|" not in log


def test_a_vanished_pokemon_is_out_of_reach_until_it_strikes(charging):
    """Sucker Punch goes first: at the Dragapult that is gone it must miss -- in the
    game, and in a world that was rebuilt from the snapshot."""
    bridge, live, shadow, snapshot = charging
    for world in (
        live,
        bridge.reconcile(shadow["state"], _spread_charge_targets(snapshot, 1)),
    ):
        theirs = bridge.choices(world["state"], "p2")[0]
        ours = next(
            choice
            for choice in bridge.choices(world["state"], "p1")
            if _atoms(choice) == ["move suckerpunch +1", "move recover"]
        )
        log = "\n".join(bridge.simulate(world["state"], ours, theirs, "5,6,7,8")["log"])
        assert "|move|p1a: Kingambit|Sucker Punch|p2a: Dragapult|[miss]" in log, log
        assert "|move|p2a: Dragapult|Phantom Force|" in log


def test_one_of_ours_left_with_struggle_is_rebuilt_playable():
    """Encore on our Garchomp's Dragon Claw, then Disable on Dragon Claw: the rules
    leave it Struggle, in the game and in a world rebuilt from the snapshot."""
    with ExactShowdownBridge() as bridge:
        live, shadow = _world(bridge, LOCKED, LOCKERS), _world(bridge, LOCKED, LOCKERS)
        live = bridge.simulate(
            live["state"],
            _pick(
                bridge.choices(live["state"], "p1"),
                "move dragonclaw +1",
                "move protect",
            ),
            _pick(
                bridge.choices(live["state"], "p2"), "move encore +1", "move protect"
            ),
            "71,72,73,74",
        )
        assert any("|-start|p1a: Garchomp|Encore" in line for line in live["log"])
        again = next(
            choice
            for choice in bridge.choices(live["state"], "p1")
            if _atoms(choice) == ["move dragonclaw +1", "move recover"]
        )
        live = bridge.simulate(
            live["state"],
            again,
            _pick(
                bridge.choices(live["state"], "p2"), "move protect", "move disable +1"
            ),
            "75,76,77,78",
        )
        assert any(
            "|-start|p1a: Garchomp|Disable|Dragon Claw" in x for x in live["log"]
        )
        assert live["request_state"] == "move"
        # the game itself: no move but Struggle (it may still switch out)
        assert _moves(bridge.choices(live["state"], "p1")) == {"move struggle"}

        snapshot = _snapshot(live)
        ours = next(
            record
            for record in snapshot["sides"][0]["pokemon"]
            if record["species"] == "garchomp"
        )
        assert "twoturnmove" not in ours["effects"]  # before: a charged "struggle"
        assert {"encore", "disable"} <= set(ours["effects"])

        rebuilt = bridge.reconcile(shadow["state"], snapshot)
        choices = [
            choice
            for choice in bridge.choices(rebuilt["state"], "p1")
            if _atoms(choice)[0] == "move struggle"
        ]
        assert _moves(bridge.choices(rebuilt["state"], "p1")) == {"move struggle"}
        theirs = next(
            choice
            for choice in bridge.choices(rebuilt["state"], "p2")
            if _atoms(choice) == ["move protect", "move protect"]
        )
        for choice in choices[:3]:  # before: every one of them threw
            played = bridge.simulate(rebuilt["state"], choice, theirs, "81,82,83,84")
            assert any("|move|p1a: Garchomp|Struggle|" in x for x in played["log"])


def test_a_request_the_parser_refuses_fails_its_world_and_not_the_decision(monkeypatch):
    """poke-env asserts on a request it cannot square with the Pokemon it holds; the
    planner sets aside a world that raises a ValueError and lets an AssertionError
    through."""
    with ExactShowdownBridge() as bridge:
        world = _world(bridge, WALLS, CHARGERS)

    def refuse(self, request):
        raise AssertionError("Error with move chillyreception. Expected ...")

    monkeypatch.setattr(DoubleBattle, "parse_request", refuse)
    with pytest.raises(ActionEncodingError, match="chillyreception") as caught:
        state_to_battle(world["state"], world["requests"], "p2", True)
    assert isinstance(caught.value, ValueError)
