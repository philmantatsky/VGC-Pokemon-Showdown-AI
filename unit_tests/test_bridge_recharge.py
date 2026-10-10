"""A search world with a Pokemon that must recharge must be playable.

After Hyper Beam and its kind the Pokemon is locked into "recharge". Showdown's own
spelling of that choice carries the target of the move before it (``move recharge +2``):
submitted by id, ``Side.chooseMove`` validates a target before it reaches the
locked-move branch, and takes "normal" for a request entry that names none. A world
rebuilt from the public snapshot had the ``mustrecharge`` volatile but no last target,
so its own choice list said ``move recharge`` and every joint choice with it failed --
"[Invalid choice] Can't move: recharge needs a target". Each world of the turn failed
the same way and the decision fell back to the bot's own move: the one turn on which an
opponent certainly does nothing was not searched. Found by a ladder rehearsal with the
null search on 2026-10-10 (a heuristic's Blast Burn); in none of the 80 ladder games.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vgc_bench.src.exact_observation import state_to_battle
from vgc_bench.src.exact_sim import ExactShowdownBridge
from vgc_bench.src.live_snapshot import public_snapshot

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "gen9championsvgc2026regmc"
BEAMERS = """Dragonite @ Lum Berry
Ability: Multiscale
Level: 50
- Hyper Beam
- Protect
- Dragon Claw
- Extreme Speed

Incineroar @ Safety Goggles
Ability: Intimidate
Level: 50
- Fake Out
- Flare Blitz
- Parting Shot
- Protect

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
WALLS = """Farigiraf @ Sitrus Berry
Ability: Armor Tail
Level: 50
- Protect
- Trick Room
- Psychic
- Helping Hand

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

Incineroar @ Safety Goggles
Ability: Intimidate
Level: 50
- Fake Out
- Flare Blitz
- Parting Shot
- Protect
"""


def _atoms(choice: str) -> list[str]:
    return [atom.strip() for atom in choice.split(",")]


def _pick(choices: list[str], first: str, second: str) -> str:
    return next(
        choice
        for choice in choices
        if _atoms(choice)[0] == first and _atoms(choice)[1] == second
    )


@pytest.mark.parametrize("beamer", ["p2", "p1"])
@pytest.mark.parametrize("history", ["fresh", "protected"])
def test_a_world_with_a_recharging_pokemon_is_rebuilt_playable(beamer, history):
    """``beamer`` p2: the opponent recharges (the case the rehearsal met). p1: we do --
    the deployed team has no such move, another team may. ``history`` protected: the
    world being rebuilt played its own turn, in which that Pokemon used Protect -- the
    last target it remembers is not a foe's."""
    waller = "p1" if beamer == "p2" else "p2"
    teams = {beamer: BEAMERS, waller: WALLS}
    with ExactShowdownBridge() as bridge:

        def world():
            return bridge.create(
                formatid=FORMAT,
                seed=[61, 62, 63, 64],
                p1_team_text=teams["p1"],
                p2_team_text=teams["p2"],
                p1_preview="team 1,2,3,4",
                p2_preview="team 1,2,3,4",
            )

        def step(state, beam: str, wall: str, seed: str):
            if beamer == "p1":
                return bridge.simulate(state, beam, wall, seed)
            return bridge.simulate(state, wall, beam, seed)

        live, shadow = world(), world()
        # Hyper Beam into the slot that does not protect; nothing else happens
        beam = _pick(
            bridge.choices(live["state"], beamer), "move hyperbeam +2", "move protect"
        )
        wall = _pick(
            bridge.choices(live["state"], waller), "move protect", "move recover"
        )
        live = step(live["state"], beam, wall, "71,72,73,74")
        assert live["request_state"] == "move"
        side = f"{beamer}a: Dragonite"
        assert f"|-mustrecharge|{side}" in live["log"], live["log"]
        # the real game's spelling carries the target the beam had
        real = {_atoms(c)[0] for c in bridge.choices(live["state"], beamer)}
        assert real == {"move recharge +2"}

        battle = state_to_battle(live["state"], live["requests"], "p1", True)
        snapshot = public_snapshot(
            battle,
            live["requests"][0],
            request_state=live["request_state"],
            side_requests=live["requests"],
        )
        record = next(
            pokemon
            for pokemon in snapshot["sides"][int(beamer[1]) - 1]["pokemon"]
            if pokemon["species"] == "dragonite"
        )
        assert "mustrecharge" in record["effects"]

        # a world created now, as on a redraw: it never saw the beam
        if history == "protected":
            other = _pick(
                bridge.choices(shadow["state"], beamer), "move protect", "move protect"
            )
            shadow = step(shadow["state"], other, wall, "71,72,73,74")
            assert shadow["request_state"] == "move"
        rebuilt = bridge.reconcile(shadow["state"], snapshot)
        locked = bridge.choices(rebuilt["state"], beamer)
        forms = {_atoms(choice)[0] for choice in locked}
        assert len(forms) == 1 and forms.pop().startswith("move recharge ")
        rest = next(
            choice
            for choice in bridge.choices(rebuilt["state"], waller)
            if "switch" not in choice
        )
        for choice in locked:  # before: every one of them threw
            played = step(rebuilt["state"], choice, rest, "81,82,83,84")
            assert f"|cant|{side}|recharge" in played["log"], played["log"]
        # and the turn after, the Pokemon chooses freely again
        after = step(rebuilt["state"], locked[0], rest, "81,82,83,84")
        if after["request_state"] == "move":
            free = {_atoms(c)[0] for c in bridge.choices(after["state"], beamer)}
            assert any(atom.startswith("move hyperbeam") for atom in free)
