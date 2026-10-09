"""A search world with an Imprison user must survive reconciliation.

Showdown's Imprison reads ``effectState.source.moveSlots``. The bridge rebuilds a
world's volatiles from the public snapshot, and a rebuilt Imprison had no source: every
reconciliation threw "Cannot read properties of undefined (reading 'moveSlots')" and the
search was off from the turn Imprison was used to the end of the game -- the roster run
of 2026-10-06 (one roster against two opponents) and two games of the ladder trial of
2026-10-09 (15 of that stage's 138 decisions).
"""

from __future__ import annotations

from pathlib import Path

from vgc_bench.src.exact_observation import state_to_battle
from vgc_bench.src.exact_sim import ExactShowdownBridge
from vgc_bench.src.live_snapshot import public_snapshot

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "gen9championsvgc2026regmc"
THEIRS = """Farigiraf @ Sitrus Berry
Ability: Armor Tail
Level: 50
- Imprison
- Protect
- Trick Room
- Psychic

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


def _first(atom: str) -> str:
    return atom.split(",")[0].strip()


def test_a_world_with_an_imprison_user_reconciles_and_keeps_the_effect():
    ours = (ROOT / "teams/candidates_mc/T6e.txt").read_text()
    with ExactShowdownBridge() as bridge:

        def world():
            return bridge.create(
                formatid=FORMAT,
                seed=[31, 32, 33, 34],
                p1_team_text=ours,
                p2_team_text=THEIRS,
                p1_preview="team 3,4,2,5",  # Charizard (it knows Protect), Venusaur
                p2_preview="team 1,2,3,4",
            )

        live, shadow = world(), world()
        before = bridge.choices(live["state"], "p1")
        assert any(_first(choice).startswith("move protect") for choice in before)
        imprison = next(
            choice
            for choice in bridge.choices(live["state"], "p2")
            if _first(choice) == "move imprison" and "protect" in choice.split(",")[1]
        )
        ours_turn = next(
            choice for choice in before if choice.count("move sleeppowder") == 0
        )
        live = bridge.simulate(live["state"], ours_turn, imprison, "41,42,43,44")
        assert live["request_state"] == "move"
        battle = state_to_battle(live["state"], live["requests"], "p1", True)
        snapshot = public_snapshot(
            battle,
            live["requests"][0],
            request_state=live["request_state"],
            side_requests=live["requests"],
        )
        user = next(
            record
            for record in snapshot["sides"][1]["pokemon"]
            if record["species"] == "farigiraf"
        )
        assert "imprison" in user["effects"]
        # a world created now, as on a redraw: it never saw Imprison being used
        repaired = bridge.reconcile(shadow["state"], snapshot)  # before: it threw
        after = bridge.choices(repaired["state"], "p1")
        assert after
        # Charizard's Protect is sealed, as its request says ...
        assert not any(_first(choice).startswith("move protect") for choice in after)
        # ... and Venusaur's can still be picked, as in the real game: the request
        # hides a sealed move from the last Pokemon to choose. Picking it shows that
        # the rebuilt effect knows who set it -- the move fails to Imprison.
        sealed = next(c for c in after if c.split(",")[1].strip() == "move protect")
        foe = next(
            choice
            for choice in bridge.choices(repaired["state"], "p2")
            if "switch" not in choice
        )
        played = bridge.simulate(repaired["state"], sealed, foe, "51,52,53,54")
        assert any(
            line.startswith("|cant|p1b: Venusaur|move: Imprison")
            for line in played["log"]
        ), played["log"]
