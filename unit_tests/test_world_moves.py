"""The moves a search world gives the opponent's active Pokemon ("likely" worlds).

A world drew one of at most twelve set families a species, and a move outside the four
it drew could not be the opponent's reply in that world. On 145 ladder games a fifth of
the opponent's real replies were legal in no planning world (147 of 690), four in five
of those for a move: in none of the species' families (31%), in no family that also
held what the Pokemon had shown (15%), or in a family none of the four worlds had drawn
(36%). The retrained predictor ranked replies better and the table stayed at 62%.

Now a world can be told the moves: what the Pokemon has shown, then the likeliest of
what it has not, the last free slot going round the candidates from world to world.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vgc_bench.src.exact_observation import (
    RootForecast,
    choice_to_actions,
    state_to_battle,
)
from vgc_bench.src.exact_sim import ExactShowdownBridge
from vgc_bench.src.live_exact import (
    LiveExactSession,
    _spread_likely_moves,
    likely_moves,
)
from vgc_bench.src.live_snapshot import public_snapshot
from vgc_bench.src.opponent_tactics import MovePrediction
from vgc_bench.src.set_particles import ParticleDatabase

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "gen9championsvgc2026regmc"
THEIRS = """Dragapult @ Life Orb
Ability: Clear Body
Level: 50
- Dragon Darts
- Protect
- Phantom Force
- U-turn

Incineroar @ Safety Goggles
Ability: Intimidate
Level: 50
- Fake Out
- Flare Blitz
- Parting Shot
- Throat Chop

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
OURS = """Milotic @ Leftovers
Ability: Competitive
Level: 50
- Recover
- Protect
- Scald
- Ice Beam

Garchomp @ Life Orb
Ability: Rough Skin
Level: 50
- Protect
- Dragon Claw
- Rock Slide
- Earthquake

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
RANKED = ["shadowball", "dracometeor", "protect", "willowisp", "dragonpulse", "uturn"]


def test_shown_moves_stay_and_the_last_free_slot_goes_round_the_worlds():
    # nothing shown: the worlds agree on three and hold the next four between them
    assert [likely_moves([], "abcdefgh", index) for index in range(5)] == [
        list("abcd"),
        list("abce"),
        list("abcf"),
        list("abcg"),
        list("abcd"),
    ]
    # one shown: it stays first, two are fixed, the fourth goes round
    assert likely_moves(["x"], ["a", "x", "b", "c", "d"], 0) == ["x", "a", "b", "c"]
    assert likely_moves(["x"], ["a", "x", "b", "c", "d"], 1) == ["x", "a", "b", "d"]
    assert likely_moves(["x"], ["a", "x", "b", "c", "d"], 2) == ["x", "a", "b", "c"]
    # three shown: one slot, a different candidate a world
    last = [(likely_moves("xyz", "abcde", index) or [""])[-1] for index in range(4)]
    assert last == list("abcd")
    # fewer candidates than free slots: what there is, and the world fills the rest
    assert likely_moves(["x"], ["a"], 3) == ["x", "a"]
    # nothing to choose
    assert likely_moves("wxyz", "abc", 0) is None
    assert likely_moves("vwxyz", "abc", 0) is None  # no set has five moves
    assert likely_moves(["x"], [], 0) is None and likely_moves(["x"], ["x"], 0) is None


def test_the_species_full_move_list_is_longer_than_its_set_families():
    database = ParticleDatabase.load(max_particles=12, formatid=FORMAT)
    usage = dict(database.move_usage("incineroar"))
    in_families = {m for p in database.particles("incineroar") for m in p.moves}
    assert usage["fakeout"] > 0.9 and 0 < usage["snarl"] < 0.1
    assert "snarl" in usage and "snarl" not in in_families  # the ladder's example
    ranked = [move for move, _ in database.move_usage("incineroar")]
    assert ranked[0] == "fakeout" and ranked == sorted(ranked, key=lambda m: -usage[m])
    assert database.move_usage("no such pokemon") == ()


@pytest.fixture(scope="module")
def played():
    """One turn: their Dragapult has shown Dragon Darts and nothing else."""
    with ExactShowdownBridge() as bridge:

        def world():
            return bridge.create(
                formatid=FORMAT,
                seed=[41, 42, 43, 44],
                p1_team_text=OURS,
                p2_team_text=THEIRS,
                p1_preview="team 1,2,3,4",
                p2_preview="team 1,2,3,4",
            )

        live, shadow = world(), world()
        ours = next(
            choice
            for choice in bridge.choices(live["state"], "p1")
            if choice == "move recover, move protect"
        )
        theirs = next(
            choice
            for choice in bridge.choices(live["state"], "p2")
            if choice == "move dragondarts +1, move flareblitz +1"
        )
        live = bridge.simulate(live["state"], ours, theirs, "51,52,53,54")
        assert live["request_state"] == "move"
        # our view of it: only what was seen
        battle = state_to_battle(live["state"], live["requests"], "p1", False)
        snapshot = public_snapshot(
            battle,
            live["requests"][0],
            request_state=live["request_state"],
            side_requests=live["requests"],
        )
        yield bridge, live, shadow, snapshot


def _record(snapshot: dict[str, Any], species: str) -> dict[str, Any]:
    return next(r for r in snapshot["sides"][1]["pokemon"] if r["species"] == species)


def _moves(world: dict[str, Any], species: str) -> list[str]:
    mon = next(
        pokemon
        for pokemon in world["state"]["sides"][1]["pokemon"]
        if species in str(pokemon.get("set", {}).get("species", "")).lower()
    )
    return [slot["id"] for slot in mon["moveSlots"]]


def test_a_world_is_told_moves_for_the_opponents_active_pokemon_only(played):
    _, _, _, snapshot = played
    assert [m["id"] for m in _record(snapshot, "dragapult")["moves"]] == ["dragondarts"]
    told = [_spread_likely_moves(snapshot, i, {0: RANKED}) for i in range(4)]
    assert [_record(t, "dragapult")["set_moves"] for t in told] == [
        ["dragondarts", "shadowball", "dracometeor", "protect"],
        ["dragondarts", "shadowball", "dracometeor", "willowisp"],
        ["dragondarts", "shadowball", "dracometeor", "dragonpulse"],
        ["dragondarts", "shadowball", "dracometeor", "uturn"],
    ]
    for world in told:
        # the other active Pokemon had no candidates, the bench and our side none
        assert world["sides"][0] is snapshot["sides"][0]
        for record in world["sides"][1]["pokemon"]:
            assert ("set_moves" in record) == (record["species"] == "dragapult")
    assert "set_moves" not in _record(
        snapshot, "dragapult"
    )  # the snapshot is not edited
    assert _spread_likely_moves(snapshot, 0, {}) is snapshot


def test_a_rebuilt_world_has_the_moves_it_was_told_and_can_play_them(played):
    bridge, live, shadow, snapshot = played
    told = _spread_likely_moves(snapshot, 1, {0: RANKED})
    assert _moves(shadow, "dragapult") == [
        "dragondarts",
        "protect",
        "phantomforce",
        "uturn",
    ]
    rebuilt = bridge.reconcile(shadow["state"], told)
    assert _moves(rebuilt, "dragapult") == [
        "dragondarts",
        "shadowball",
        "dracometeor",
        "willowisp",
    ]
    assert _moves(rebuilt, "incineroar") == _moves(shadow, "incineroar")  # untouched
    replies = bridge.choices(rebuilt["state"], "p2")
    first = {choice.split(",")[0].strip().split()[1] for choice in replies}
    assert {"shadowball", "dracometeor", "willowisp", "dragondarts"} <= first
    assert "phantomforce" not in first
    # the opponent's seat reads the world as it now is (poke-env refuses a fifth move)
    seat = state_to_battle(rebuilt["state"], rebuilt["requests"], "p2", True)
    reply = next(
        c for c in replies if c.startswith("move shadowball +1, move flareblitz")
    )
    choice_to_actions(
        reply, rebuilt["requests"][1], state=rebuilt["state"], role="p2", battle=seat
    )
    ours = next(
        c
        for c in bridge.choices(rebuilt["state"], "p1")
        if c.startswith("move recover")
    )
    played_out = bridge.simulate(rebuilt["state"], ours, reply, "61,62,63,64")
    assert any("|move|p2a: Dragapult|Shadow Ball|" in x for x in played_out["log"])

    # a world that did see the turn keeps the PP it spent on the move it had shown
    def darts(world: dict[str, Any]) -> dict[str, Any]:
        mon = world["state"]["sides"][1]["pokemon"][0]
        return next(slot for slot in mon["moveSlots"] if slot["id"] == "dragondarts")

    spent = darts(live)["pp"]
    assert spent < darts(shadow)["pp"]
    kept = bridge.reconcile(live["state"], told)
    assert darts(kept)["pp"] == spent and _moves(kept, "dragapult")[1] == "shadowball"
    # a shorter list is filled from what the Pokemon had; four moves, always
    short = _spread_likely_moves(snapshot, 0, {0: ["shadowball"]})
    assert _record(short, "dragapult")["set_moves"] == ["dragondarts", "shadowball"]
    filled = _moves(bridge.reconcile(shadow["state"], short), "dragapult")
    assert filled[:2] == ["dragondarts", "shadowball"] and len(filled) == 4
    assert set(filled[2:]) <= {"protect", "phantomforce", "uturn"}


def _session(played, forecast: RootForecast | None, mode: str = "likely") -> Any:
    return SimpleNamespace(
        prior=SimpleNamespace(root_forecast=forecast),
        database=ParticleDatabase.load(max_particles=12, formatid=FORMAT),
        world_moves=mode,
        world_move_sources={},
        open_sheet=False,
        oracle_opponent_team_text=None,
        _likely_candidates=None,
    )


def _forecast(species: str) -> RootForecast:
    """The predictor on Dragapult's slot: Will-O-Wisp likeliest, then Draco Meteor."""
    slot = MovePrediction(
        (("dracometeor", 0.2), ("willowisp", 0.5), ("dragondarts", 0.1)), ()
    )
    return RootForecast(2, (slot, MovePrediction((), ())), None, None, (species, ""))


def test_the_candidates_are_the_forecasts_then_the_species_most_used(played):
    _, _, _, snapshot = played
    usage = [
        m for m, _ in ParticleDatabase.load(formatid=FORMAT).move_usage("dragapult")
    ]
    session = _session(played, _forecast("Dragapult"))
    ranked = LiveExactSession._likely_candidates(session, snapshot)
    # by how likely each is to be used now, then by how many sets hold it
    assert ranked[0][:3] == ["willowisp", "dracometeor", "dragondarts"]
    assert ranked[0][3:] == usage and session.world_move_sources["forecast"] == 1
    # the other slot: no forecast moves, the species' list alone
    assert ranked[1][0] == "fakeout" and session.world_move_sources["usage"] == 1
    # a forecast about another Pokemon (it switched) is not read for this one
    other = _session(played, _forecast("garchomp"))
    assert LiveExactSession._likely_candidates(other, snapshot)[0] == usage
    assert (
        LiveExactSession._likely_candidates(_session(played, None), snapshot)[0]
        == usage
    )


def test_only_likely_worlds_with_hidden_sheets_are_told_moves(played):
    _, _, _, snapshot = played

    def told(session: Any, index: int = 0) -> dict[str, Any]:
        session._likely_candidates = lambda snap: LiveExactSession._likely_candidates(
            session, snap
        )
        return LiveExactSession._world_snapshot(session, snapshot, index)

    likely = told(_session(played, _forecast("dragapult")))
    assert _record(likely, "dragapult")["set_moves"][:3] == [
        "dragondarts",
        "willowisp",
        "dracometeor",
    ]
    assert "set_moves" in _record(likely, "incineroar")
    assert told(_session(played, None, mode="particle")) is snapshot
    for name, value in (("open_sheet", True), ("oracle_opponent_team_text", "x")):
        session = _session(played, _forecast("dragapult"))
        setattr(session, name, value)
        assert told(session) is snapshot
