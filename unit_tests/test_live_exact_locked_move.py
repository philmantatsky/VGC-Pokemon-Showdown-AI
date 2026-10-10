"""The bot's own pair on a turn one of our Pokemon is locked into a move.

2026-10-10, a ladder rehearsal with the null search: our Charizard charged Solar Beam
in the opponent's rain and fired it the turn after. That turn the simulator offers the
slot one command, the move with the target it was first given (``move solarbeam +1``).
The bot's own action for the same turn named the other target -- its mask allows both
and the server takes the lock's either way. The search looked for the bot's pair among
a world's choices number for number, did not find it, planned the turn without the pair
it is anchored to, and answered with the lock's spelling: the null search read one
changed decision of 32. Nothing was played differently that time; with a real anchor
the partner's slot had lost its default.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from poke_env.environment import DoublesEnv

from vgc_bench.src.exact_observation import choice_to_actions, state_to_battle
from vgc_bench.src.exact_planner import ExactNode
from vgc_bench.src.exact_sim import ExactShowdownBridge
from vgc_bench.src.live_exact import (
    LiveExactSession,
    LiveRoot,
    _forced_slots,
    _same_pair,
)

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "gen9championsvgc2026regmc"
RAIN = """Pelipper @ Damp Rock
Ability: Drizzle
Level: 50
- Hurricane
- Protect
- Tailwind
- Weather Ball

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


def _atoms(choice: str) -> list[str]:
    return [atom.strip() for atom in choice.split(",")]


def test_a_slot_with_one_command_is_forced_and_only_there_the_target_is_free():
    legal = ["move solarbeam +1, move protect", "move solarbeam +1, move psychic +2"]
    assert _forced_slots(legal) == (True, False)
    assert _forced_slots(["move a +1, pass", "move a +2, pass"]) == (False, True)
    assert _forced_slots([]) == (False, False)
    # 15 and 16: the second move at the first and at the second foe
    assert _same_pair((15, 9), (16, 9), (True, False))
    assert _same_pair((15, 9), (15, 9), (False, False))
    # a slot that does choose: the target is the choice
    assert not _same_pair((15, 9), (16, 9), (False, False))
    assert not _same_pair((15, 9), (15, 10), (True, False))
    # never another move, another gimmick (+20: Mega Evolution), a switch or a pass
    assert not _same_pair((15, 9), (21, 9), (True, False))
    assert not _same_pair((15, 9), (35, 9), (True, False))
    assert not _same_pair((15, 9), (3, 9), (True, False))
    assert not _same_pair((0, 9), (16, 9), (True, False))


@pytest.fixture(scope="module")
def locked():
    """Our Charizard charged Solar Beam at the first foe in the rain: it is locked."""
    ours = (ROOT / "teams/candidates_mc/T6e.txt").read_text()
    with ExactShowdownBridge() as bridge:
        world = bridge.create(
            formatid=FORMAT,
            seed=[91, 92, 93, 94],
            p1_team_text=ours,
            p2_team_text=RAIN,
            p1_preview="team 3,4,2,5",  # Charizard in slot a, Venusaur in slot b
            p2_preview="team 1,2,3,4",
        )
        charge = next(
            choice
            for choice in bridge.choices(world["state"], "p1")
            if _atoms(choice) == ["move solarbeam +1", "move protect"]
        )
        wait = next(
            choice
            for choice in bridge.choices(world["state"], "p2")
            if _atoms(choice) == ["move protect", "move protect"]
        )
        world = bridge.simulate(world["state"], charge, wait, "95,96,97,98")
        assert "|-prepare|p1a: Charizard|Solar Beam" in world["log"], world["log"]
        assert world["request_state"] == "move"
        battle = state_to_battle(world["state"], world["requests"], "p1", True)
        root = LiveRoot(ExactNode.from_result(world), 1.0, {}, (), "the world")
        yield SimpleNamespace(bridge=bridge, root=root, battle=battle)


def _session(locked, champion=None) -> Any:
    """Just what the three methods read of a session."""
    return SimpleNamespace(
        bridge=locked.bridge,
        roots=[locked.root],
        champion_actions=champion,
        current_battle=locked.battle,
        last_result=None,
        fallbacks=0,
        _root_live_actions=LiveExactSession._root_live_actions,
    )


def _both_spellings(locked) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """A legal choice of the world, the pair it maps to, and the bot's other way of
    naming the same pair: the locked move at the other foe."""
    legal = locked.bridge.choices(locked.root.node.state, "p1")
    assert {_atoms(choice)[0] for choice in legal} == {"move solarbeam +1"}
    assert _forced_slots(legal) == (True, False)
    choice = next(c for c in legal if _atoms(c)[1].startswith("move sludgebomb"))
    request = locked.root.node.requests[0]
    simulators = choice_to_actions(
        choice, request, state=locked.root.node.state, role="p1", battle=locked.battle
    )
    bots = (simulators[0] + 1, simulators[1])  # the same move at the second foe
    # the bot's mask really allows it: the live request takes the pair as it stands
    DoublesEnv.action_to_order(
        np.asarray(bots, dtype=np.int64), locked.battle, fake=False, strict=True
    )
    # and no command of the world maps to it number for number -- the old test
    mapped = {
        choice_to_actions(
            other,
            request,
            state=locked.root.node.state,
            role="p1",
            battle=locked.battle,
        )
        for other in legal
    }
    assert simulators in mapped and bots not in mapped
    return choice, simulators, bots


def test_the_bots_pair_is_found_among_a_worlds_choices(locked):
    choice, _, bots = _both_spellings(locked)
    found = LiveExactSession._champion_choices(
        _session(locked), [locked.root], bots, locked.battle
    )
    assert found == [choice]  # before: [None], and the turn had no anchor


def test_the_search_hands_the_bots_pair_back_as_the_bot_spells_it(locked):
    choice, simulators, bots = _both_spellings(locked)
    session = _session(locked, champion=bots)
    # the pair the search is anchored to: number for number the bot's own
    assert LiveExactSession._live_actions(session, choice, locked.battle) == bots
    # another pair keeps the simulator's spelling of the locked slot
    legal = locked.bridge.choices(locked.root.node.state, "p1")
    other = next(c for c in legal if _atoms(c)[1] != _atoms(choice)[1])
    played = LiveExactSession._live_actions(session, other, locked.battle)
    assert played[0] == simulators[0] and played[1] != bots[1]
    # and without a pair to compare with nothing is rewritten
    assert (
        LiveExactSession._live_actions(_session(locked), choice, locked.battle)
        == simulators
    )


def test_what_the_bot_played_is_written_back_into_the_world(locked):
    choice, _, bots = _both_spellings(locked)
    session = _session(locked)
    LiveExactSession.record_actions(session, bots)
    assert session.pending_our_choice == choice  # before: None, and a fallback counted
    assert session.fallbacks == 0
