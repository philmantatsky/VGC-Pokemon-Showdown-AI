"""A search world created while one of the two active slots holds a fainted Pokemon.

2026-10-09, the first rehearsal of a ladder trial with the search on (the ladder
script itself on a local server, null search): at a forced replacement the search sent
in Torkoal where the bot sends Farigiraf. The opponent's move that fainted Charizard
had contradicted seven of the eight hidden-set worlds, so they were drawn again at
that request -- and a world created there put the next Pokemon of our four in team
order (Farigiraf, healthy, on the bench) into the fainted Charizard's slot, because
poke-env reports a fainted Pokemon's slot as empty. Reconciliation then marked
Farigiraf as the one to be replaced; the only replacement the world offered was
Torkoal, and the bot's own pick could not be expressed.

The head-to-heads of 10-04 / 10-05 could not show it (0 of 1,459 such redraws): there
both sides lead the first two Pokemon of the team file, and the next of the four in
file order is then always the fainted lead itself. On other rosters it did happen (16
of 234 redraws at a replacement in the roster run), and on ladder three games in ten
open with another lead pair.

The same moment on the opponent's side raised "opponent leads unavailable" (no world
could be created; 5.8% of the hidden-sheet decisions of the 10-05 head-to-head carry
it) and the redraw was then never tried again for that evidence.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vgc_bench.src.exact_observation import state_to_battle
from vgc_bench.src.exact_sim import ExactShowdownBridge
from vgc_bench.src.live_exact import (
    LiveExactSession,
    _opponent_previews,
    _our_preview,
    _slot_occupants,
)
from vgc_bench.src.live_snapshot import public_snapshot
from vgc_bench.src.set_particles import TeamSlot

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "gen9championsvgc2026regmc"
# our team file's order; we brought Farigiraf, Charizard, Venusaur and Torkoal, and
# Charizard and Venusaur lead
TEAM = ("Blastoise", "Farigiraf", "Charizard", "Venusaur", "Torkoal", "Incineroar")
BROUGHT = {"Farigiraf", "Charizard", "Venusaur", "Torkoal"}
ROSTER = tuple(TeamSlot(name.lower(), name) for name in TEAM)
RIGHT = "team 3,4,2,5"  # Charizard in slot a, Venusaur in slot b


def _mon(name: str, *, selected: bool = True, fainted: bool = False) -> Any:
    return SimpleNamespace(
        species=name,
        base_species=name,
        selected_in_teampreview=selected,
        _selected_in_teampreview=selected,
        fainted=fainted,
        revealed=True,
    )


def _our_battle(*fainted: str, role: str = "p2", slots: bool = True) -> Any:
    """The live battle's shape: all six in team order, four of them brought."""
    mons = {n: _mon(n, selected=n in BROUGHT, fainted=n in fainted) for n in TEAM}
    held = {f"{role}a": mons["Charizard"], f"{role}b": mons["Venusaur"]}
    battle = SimpleNamespace(
        team={f"{role}: {name}": mon for name, mon in mons.items()},
        # poke-env: a fainted Pokemon's slot reads as empty
        active_pokemon=[None if mon.fainted else mon for mon in held.values()],
    )
    if slots:
        battle.player_role = role
        battle._active_pokemon = held
    return battle


def test_both_standing_is_the_order_it_always_was():
    assert _our_preview(_our_battle(), ROSTER) == RIGHT


@pytest.mark.parametrize(
    "fainted",
    [("Charizard",), ("Venusaur",), ("Charizard", "Venusaur")],
    ids=["slot a", "slot b", "both"],
)
def test_a_fainted_pokemon_keeps_its_slot_in_a_world_created_then(fainted):
    """Before: the next of the four in team order took the slot -- "team 4,2,3,5"
    with Charizard down (Farigiraf in a slot it never stood in)."""
    assert _our_preview(_our_battle(*fainted), ROSTER) == RIGHT
    for role in ("p1", "p2"):
        assert _our_preview(_our_battle(*fainted, role=role), ROSTER) == RIGHT


def test_a_slot_nobody_is_known_to_stand_in_goes_to_a_fainted_pokemon():
    """Without poke-env's slot table (an older battle object) the fainted Pokemon is
    still the one to fill the empty slot: never a healthy one, which the request
    offers as a replacement."""
    assert _our_preview(_our_battle("Charizard", slots=False), ROSTER) == RIGHT
    # an earlier faint on the bench serves as well: it cannot be sent in either
    battle = _our_battle("Charizard", "Torkoal", slots=False)
    preview = _our_preview(battle, ROSTER)
    assert preview.startswith(("team 3,4,", "team 5,4,"))
    assert "2" not in preview.removeprefix("team ").split(",")[:2]


def test_slot_occupants_reads_the_fainted_pokemon_of_either_side():
    battle = _our_battle("Charizard")
    occupants = _slot_occupants(battle, own=True)
    assert [mon and mon.species for mon in occupants] == ["Charizard", "Venusaur"]
    # a slot whose last Pokemon left without fainting stays empty
    battle._active_pokemon["p2a"] = _mon("Charizard")
    assert _slot_occupants(battle, own=True)[0] is None
    # no slot table: what poke-env reports
    bare = _our_battle("Charizard", slots=False)
    assert _slot_occupants(bare, own=True) == bare.active_pokemon


def _their_battle(*fainted: str, slots: bool = True) -> Any:
    mons = {n: _mon(n, fainted=n in fainted) for n in TEAM}
    for name in ("Blastoise", "Farigiraf", "Torkoal", "Incineroar"):
        mons[name].revealed = False
    held = {"p1a": mons["Charizard"], "p1b": mons["Venusaur"]}
    battle = SimpleNamespace(
        opponent_team={f"p1: {name}": mon for name, mon in mons.items()},
        opponent_active_pokemon=[None if m.fainted else m for m in held.values()],
    )
    if slots:
        battle.opponent_role = "p1"
        battle._opponent_active_pokemon = held
    return battle


def test_worlds_can_be_created_while_an_opposing_slot_is_empty():
    """Before: ValueError("opponent leads unavailable: ['venusaur']")."""
    previews = _opponent_previews(_their_battle("Charizard"), ROSTER)
    assert len(previews) == 6  # every pair of the four they have not shown
    for preview, brought in previews:
        assert brought[:2] == ("charizard", "venusaur")
        assert preview.startswith("team 3,4,")
    # without the slot table nothing says who stood there: still refused
    with pytest.raises(ValueError, match="opponent leads unavailable"):
        _opponent_previews(_their_battle("Charizard", slots=False), ROSTER)


def _pending_replacement_snapshot(bridge: ExactShowdownBridge) -> dict:
    """The live board the rehearsal reached: our Charizard fainted in slot a on turn
    one, Venusaur stands in slot b, Farigiraf and Torkoal are on the bench."""
    live = bridge.create(
        formatid=FORMAT,
        seed=[11, 12, 13, 14],
        p1_team_text=(ROOT / "teams/candidates_mc/T6e.txt").read_text(),
        p2_team_text=(ROOT / "teams/reg_mc/MC2566.txt").read_text(),
        p1_preview=RIGHT,
        p2_preview="team 2,4,5,1",
    )
    battle = state_to_battle(live["state"], live["requests"], "p1", True)
    snapshot = public_snapshot(
        battle,
        live["requests"][0],
        request_state=live["request_state"],
        side_requests=live["requests"],
    )
    snapshot = deepcopy(snapshot)
    charizard = snapshot["sides"][0]["pokemon"][0]
    assert charizard["species"] == "charizard" and charizard["active_slot"] == 0
    # as the live snapshot writes a fainted Pokemon: no slot of its own
    charizard.update(
        active_slot=None, hp=0, hp_fraction=0.0, fainted=True, status="fnt"
    )
    snapshot["request_state"] = "switch"
    snapshot["sides"][0]["force_switch"] = [True, False]
    return snapshot


def _replacements(bridge: ExactShowdownBridge, snapshot: dict, preview: str) -> set:
    """Who a world created now with ``preview`` lets us send in."""
    world = bridge.create(
        formatid=FORMAT,
        seed=[21, 22, 23, 24],
        p1_name="planner",
        p2_name="opponent",
        p1_team_text=(ROOT / "teams/candidates_mc/T6e.txt").read_text(),
        p2_team_text=(ROOT / "teams/reg_mc/MC2566.txt").read_text(),
        p1_preview=preview,
        p2_preview="team 2,4,5,1",
    )
    repaired = bridge.reconcile(world["state"], snapshot)
    party = [
        str(mon["speciesState"]["id"])
        for mon in repaired["state"]["sides"][0]["pokemon"]
    ]
    sent = set()
    for choice in bridge.choices(repaired["state"], "p1"):
        first = choice.split(",")[0].strip()
        assert first.startswith("switch "), choice
        sent.add(party[int(first.split()[1]) - 1])
    return sent


def test_a_world_created_at_a_pending_replacement_offers_every_replacement():
    with ExactShowdownBridge() as bridge:
        snapshot = _pending_replacement_snapshot(bridge)
        live = _our_battle("Charizard")
        assert _replacements(bridge, snapshot, _our_preview(live, ROSTER)) == {
            "farigiraf",
            "torkoal",
        }
        # the order such a world was created with until 2026-10-09 (Venusaur, then
        # the four in team order): Farigiraf stands in Charizard's slot and is the
        # one "replaced" -- only Torkoal can be sent in
        assert _replacements(bridge, snapshot, "team 4,2,3,5") == {"torkoal"}


def _session(create) -> Any:
    session: Any = object.__new__(LiveExactSession)
    session.open_sheet = False
    session.oracle_opponent_team_text = None
    session.roots = ["old"] * 8
    session.search_determinizations = 4
    session.roots_agreeing_with_evidence = 0
    session.evidence_signature = (("sneasler", ("rockslide",), None, None),)
    session.root_refreshes = 0
    session.reconciliations = 0
    session.last_root_refresh_turn = 0
    session.last_reconcile_errors = []
    session._forget_plans = lambda: None
    session._condition_roots = lambda battle: None
    session._create_roots = create
    return session


def test_a_redraw_that_failed_is_tried_again_and_one_that_happened_is_not():
    battle: Any = SimpleNamespace(turn=3)
    attempts = []

    def create(_battle, _snapshot, fail=True):
        attempts.append(fail)
        if fail:
            raise ValueError("opponent leads unavailable: ['glimmora']")
        session.roots = ["new"] * 8

    session = _session(create)
    session._resample_contradicted_roots(battle, {})
    # the old worlds stay, the failure is on record -- and the evidence is still open
    assert session.roots == ["old"] * 8
    assert session.last_reconcile_errors == [
        "evidence resample: ValueError: opponent leads unavailable: ['glimmora']"
    ]
    session._resample_contradicted_roots(battle, {})
    assert attempts == [True, True]  # before: one attempt, then never again
    session._create_roots = lambda b, s: create(b, s, fail=False)
    session._resample_contradicted_roots(battle, {})
    assert session.roots == ["new"] * 8 and session.evidence_resamples == 1
    # answered: the same evidence does not redraw the worlds again (a set no particle
    # can express must not loop)
    session._resample_contradicted_roots(battle, {})
    assert attempts == [True, True, False]
