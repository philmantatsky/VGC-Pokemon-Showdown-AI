"""evaluation/oppmodel_oracle.py: which audit records are move decisions, how a
true action becomes a certain prediction (and which slot it lands on, from either
seat), what each arm feeds the reranker, the hindsight terms, the game-clustered
bootstrap and the R4 reading."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import torch
from poke_env.teambuilder import Teambuilder

from evaluation import oppmodel_oracle as O
from vgc_bench.src.guards import Candidate
from vgc_bench.src.oppmodel import events as E
from vgc_bench.src.opponent_tactics import MovePrediction, SwitchPrediction
from vgc_bench.src.policy_player import PolicyPlayer

ROOT = Path(__file__).resolve().parents[1]

TEAM = """Charizard @ Charcoal
Ability: Blaze
Level: 50
EVs: 2 HP / 32 SpA / 32 Spe
Modest Nature
- Heat Wave
- Air Slash
- Protect
- Solar Beam

Garchomp @ Sitrus Berry
Ability: Rough Skin
Level: 50
EVs: 32 HP / 32 Atk / 2 Spe
Adamant Nature
- Dragon Claw
- Earthquake
- Protect
- Rock Slide

Incineroar @ Safety Goggles
Ability: Intimidate
Level: 50
EVs: 32 HP / 32 Def / 2 SpD
Impish Nature
- Fake Out
- Flare Blitz
- Parting Shot
- Knock Off

Rillaboom @ Miracle Seed
Ability: Grassy Surge
Level: 50
EVs: 32 HP / 32 Atk / 2 Spe
Adamant Nature
- Fake Out
- Wood Hammer
- Grassy Glide
- Protect
"""

# The bot is Alice (p1). Turn 1: their slot b switches out, so that by turn 2
# one of their benched Pokemon has been seen and one has not.
LOG = """|gametype|doubles
|player|p1|Alice|1|1300
|player|p2|Bob|2|1250
|gen|9
|tier|[Gen 9 Champions] VGC 2026 Reg M-C
|clearpoke
|poke|p1|Charizard, L50, M|
|poke|p1|Garchomp, L50, M|
|poke|p1|Incineroar, L50, M|
|poke|p1|Rillaboom, L50, F|
|poke|p2|Tyranitar, L50, M|
|poke|p2|Sneasler, L50, M|
|poke|p2|Politoed, L50, M|
|poke|p2|Archaludon, L50, M|
|teampreview|4
|teamsize|p1|4
|teamsize|p2|4
|start
|switch|p1a: Charizard|Charizard, L50, M|155/155
|switch|p1b: Garchomp|Garchomp, L50, M|215/215
|switch|p2a: Tyranitar|Tyranitar, L50, M|100/100
|switch|p2b: Sneasler|Sneasler, L50, M|100/100
|turn|1
|switch|p2b: Politoed|Politoed, L50, M|100/100
|move|p1a: Charizard|Protect|p1a: Charizard
|-singleturn|p1a: Charizard|Protect
|move|p1b: Garchomp|Protect|p1b: Garchomp
|-singleturn|p1b: Garchomp|Protect
|move|p2a: Tyranitar|Stone Edge|p1a: Charizard
|-activate|p1a: Charizard|move: Protect
|upkeep
|turn|2
"""
_SWAP = re.compile(r"\bp([12])(?=[ab:|]|$)", re.M)


def mirrored(log: str) -> str:
    """The same game with the sides renamed: the bot becomes p2."""
    return _SWAP.sub(lambda m: "p" + ("2" if m.group(1) == "1" else "1"), log)


def position_at(log: str, player: str, role: str) -> O.Position:
    sets = Teambuilder.parse_showdown_team(TEAM)
    position = O.Position("battle-gen9championsvgc2026regmc-1", player, role, sets)
    counters: Counter[str] = Counter()
    for event in E.split_log(log):
        if position.feed(event, counters) == "turn":
            position.at_turn_start()
    assert not counters
    return position


def action_for(battle: Any, slot: int, move: str, target: int = 0) -> int:
    for action in range(7, 27):
        order = PolicyPlayer._audit_order(battle, action, slot)
        if order.get("id") == move and order.get("target") == target:
            return action
    raise AssertionError(f"no action for {move} -> {target}")


def pair(battle: Any, first: tuple[str, int], second: tuple[str, int]) -> Candidate:
    return Candidate(
        (action_for(battle, 0, *first), action_for(battle, 1, *second)), 0.5
    )


def move_action(side: str, slot: str, move: str, target: str | None) -> E.SlotAction:
    return E.SlotAction(E.KIND_MOVE, move=move, target=target, side=side, slot=slot)


def config(sticky: bool = False) -> O.DirectoryConfig:
    return O.DirectoryConfig(
        name="test",
        team=Path("unused"),
        move_model=None,
        switch_model=None,
        preview_model=None,
        use_opponent=True,
        use_tempo=True,
        sticky=sticky,
        moveset_prior=True,
        set_prior_reg="mc",
        mixing="off",
    )


# --- decision filtering --------------------------------------------------------------


def record(turn: int, kinds: list[list[str]], battle: str = "battle-f-1") -> dict:
    candidates = [
        {
            "actions": [index, index],
            "orders": [{"kind": kind} for kind in orders],
            "policy_probability": 0.5,
            "demoted_by": None,
        }
        for index, orders in enumerate(kinds)
    ]
    return {
        "battle": battle,
        "turn": turn,
        "candidates": candidates,
        "chosen": candidates[0] if candidates else None,
    }


def test_classify_record_follows_the_critics_rule():
    assert O.classify_record(record(0, [["move", "move"]])) == O.RECORD_PREVIEW
    assert O.classify_record(record(3, [["switch", "pass"]])) == O.RECORD_FORCED
    assert O.classify_record(record(3, [["pass", "pass"]])) == O.RECORD_FORCED
    assert O.classify_record(record(3, [["switch", "switch"]])) == O.RECORD_FORCED
    # one order that is a move makes it a real decision, whatever the others are
    mixed = record(3, [["switch", "pass"], ["move", "switch"]])
    assert O.classify_record(mixed) == O.RECORD_MOVE
    assert O.classify_record(record(1, [["move", "move"]])) == O.RECORD_MOVE
    assert O.classify_record({"battle": "b", "turn": 0, "playbook": {}}) == (
        O.RECORD_OTHER
    )
    assert O.classify_record(record(2, [])) == O.RECORD_OTHER
    assert O.classify_record({"candidates": [{}], "turn": "x"}) == O.RECORD_OTHER


def test_read_decisions_census_superseded_and_sheet_flags():
    lines = [
        json.dumps({"battle": "battle-f-1", "turn": 0, "preview_shadow": {}}),
        json.dumps(record(0, [["move", "move"]])),
        json.dumps(record(1, [["move", "move"]])),
        "not json",
        json.dumps(record(2, [["move", "move"]])),  # refused by the server ...
        json.dumps(record(2, [["move", "switch"]])),  # ... and replaced
        json.dumps(record(2, [["switch", "pass"]])),
        "",
        json.dumps(
            {"battle": "battle-f-2", "turn": 0, "preview_shadow": {"open_sheet": True}}
        ),
        json.dumps(record(1, [["move", "move"]], battle="battle-f-2")),
        json.dumps({"battle": "battle-f-3", "turn": 0, "their_sheet": {"x": {}}}),
        json.dumps(
            {"battle": "battle-f-4", "turn": 0, "preview_shadow": {"open_sheet": False}}
        ),
        json.dumps([1, 2]),
    ]
    decisions, census, sheets = O.read_decisions(lines, "folder")
    assert census == {
        O.RECORD_OTHER: 4,
        O.RECORD_PREVIEW: 1,
        O.RECORD_MOVE: 4,
        O.RECORD_FORCED: 1,
        "bad_json": 2,
    }
    assert [(d.room, d.turn, d.line, d.superseded) for d in decisions] == [
        ("battle-f-1", 1, 2, False),
        ("battle-f-1", 2, 4, True),
        ("battle-f-1", 2, 5, False),
        ("battle-f-2", 1, 9, False),
    ]
    assert all(decision.folder == "folder" for decision in decisions)
    assert sheets == {"battle-f-2": True, "battle-f-3": True, "battle-f-4": False}


def test_public_battle_id_drops_a_private_room_suffix():
    room = "battle-gen9championsvgc2026regmc-2692579470-abcdefghijklmnopqrstuvwxpw"
    assert O.public_battle_id(room) == "gen9championsvgc2026regmc-2692579470"
    assert O.public_battle_id("battle-gen9x-12") == "gen9x-12"
    assert O.public_battle_id("something else") == "something else"


def test_find_replays_and_bot_role(tmp_path):
    room = "battle-gen9championsvgc2026regmc-77-abcpw"
    (tmp_path / f"Alice Smith - {room}.html").write_text("x")
    (tmp_path / "notes.html").write_text("x")
    found = O.find_replays(tmp_path)
    assert list(found) == [room]
    assert found[room][1] == "Alice Smith"
    events = E.split_log("|player|p1|Bob|1|1200\n|player|p2|alice smith|2|1300\n")
    assert O.bot_role(events, "Alice Smith") == "p2"
    assert O.bot_role(events, "Carol") is None


def test_read_run_config_takes_the_team_from_the_directory(tmp_path):
    material = {
        "our_team": "teams/x.txt",
        "move_model": "data/m.pt",
        "switch_model": "",
        "preview_model": "data/p.pt",
        "opponent_aware": True,
        "tempo_aware": False,
        "sticky_corrections": True,
        "moveset_prior": True,
        "set_prior_reg": "mc",
    }
    older = dict(material, our_team="teams/y.txt")
    runs = {"runs": [{"material": older}, {"material": material}]}
    (tmp_path / "run_config.json").write_text(json.dumps(runs))
    loaded = O.read_run_config(tmp_path)
    assert loaded is not None
    assert loaded.team == O.ROOT / "teams/x.txt"
    assert loaded.move_model == O.ROOT / "data/m.pt" and loaded.switch_model is None
    assert (loaded.use_opponent, loaded.use_tempo, loaded.sticky) == (True, False, True)
    assert loaded.set_prior_reg == "mc" and loaded.varies
    del material["set_prior_reg"], material["sticky_corrections"]
    (tmp_path / "run_config.json").write_text(json.dumps({"runs": [runs["runs"][1]]}))
    plain = O.read_run_config(tmp_path)
    assert plain is not None and plain.set_prior_reg == "mb"
    assert not plain.sticky and not plain.varies
    (tmp_path / "run_config.json").write_text("{}")
    assert O.read_run_config(tmp_path) is None
    assert O.read_run_config(tmp_path / "absent") is None


def test_rebuild_candidates_restores_the_order_the_reranker_received():
    def entry(actions, probability, demoted=None):
        return {
            "actions": actions,
            "policy_probability": probability,
            "demoted_by": demoted,
        }

    logged = [
        entry([3, 3], 0.2),  # the reranker's pick
        entry([1, 1], 0.1),  # the guard's pick, first before the reranker
        entry([2, 2], 0.5),
        entry([4, 4], 0.05),
        entry([9, 9], 0.9, "zero_damage"),
    ]
    row = {"candidates": logged, "reranker": {"before": [1, 1], "after": [3, 3]}}
    order = [candidate.actions for candidate in O.rebuild_candidates(row)]
    assert order == [(1, 1), (2, 2), (3, 3), (4, 4), (9, 9)]
    untouched = O.rebuild_candidates({"candidates": logged})
    assert [c.actions for c in untouched] == [tuple(e["actions"]) for e in logged]
    assert untouched[-1].demoted_by == "zero_damage" and untouched[0].prob == 0.2
    lost = {"candidates": logged, "reranker": {"before": [7, 7]}}
    assert [c.actions for c in O.rebuild_candidates(lost)][0] == (3, 3)


# --- a true action as a certain prediction -------------------------------------------


def test_a_true_move_becomes_one_certain_move_on_the_slot_it_landed_on():
    made = O.certain_predictions(move_action("p2", "a", "stoneedge", E.TARGET_FOE_B))
    assert made is not None
    moves, switches = made
    assert moves.moves == (("stoneedge", 1.0),)
    assert moves.actions == (("stoneedge", "foe_b", 1.0),)
    assert moves.targets == (("foe_b", 1.0),) and moves.reliability == 1.0
    assert switches == SwitchPrediction(0.0, ())
    spread = O.certain_predictions(move_action("p2", "a", "rockslide", E.TARGET_AUTO))
    assert spread is not None and spread[0].actions == (("rockslide", "field", 1.0),)
    gated = O.certain_predictions(
        move_action("p2", "a", "stoneedge", E.TARGET_FOE_A), reliability=0.25
    )
    assert gated is not None and gated[0].reliability == 0.25


def test_a_true_switch_becomes_a_certain_switch_and_no_move():
    action = E.SlotAction(E.KIND_SWITCH, switch_to="politoed", side="p2", slot="b")
    made = O.certain_predictions(action)
    assert made is not None
    assert made[0] == MovePrediction((), (), (), reliability=1.0)
    assert made[1] == SwitchPrediction(1.0, (("politoed", 1.0),))


def test_what_the_log_does_not_show_is_not_stated_as_fact():
    fainted = E.SlotAction(E.KIND_NONE, reason=E.REASON_FAINTED_FIRST)
    flinch = E.SlotAction(E.KIND_HIDDEN, reason=E.REASON_FLINCH)
    forced = E.SlotAction(
        E.KIND_HIDDEN, reason=E.REASON_OVERRIDDEN, forced_move="protect"
    )
    for action in (None, fainted, flinch, forced):
        assert not O.oracle_known(action)
        assert O.certain_predictions(action) is None
    # an aimed attack whose target was never logged: the consumer needs the slot
    blank = move_action("p2", "a", "stoneedge", None)
    assert not O.oracle_known(blank) and O.certain_predictions(blank) is None
    # a move nobody aims is known without a target
    status = O.certain_predictions(move_action("p2", "a", "trickroom", None))
    assert status is not None and status[0].moves == (("trickroom", 1.0),)
    assert status[0].actions == ()


def test_certain_predictions_from_a_read_log():
    actions = dict(E.read_log_actions(E.split_log(LOG))[0][1])
    attack = O.certain_predictions(actions["p2a"])
    assert attack is not None
    assert attack[0].actions == (("stoneedge", "foe_a", 1.0),)
    switch = O.certain_predictions(actions["p2b"])
    assert switch is not None and switch[1].targets == (("politoed", 1.0),)
    assert actions["p1a"].is_protect
    assert O.truth_class([actions["p2a"], actions["p2b"]]) == O.TRUTH_SWITCH
    assert O.truth_class([actions["p2a"], None]) == O.TRUTH_MOVE_ONLY
    assert O.truth_class([None, E.SlotAction(E.KIND_NONE)]) == O.TRUTH_NONE


def test_prescaling_keeps_the_weight_and_opens_the_gate():
    logged = MovePrediction(
        (("a", 0.6), ("b", 0.4)),
        (("foe_a", 1.0),),
        (("a", "foe_a", 0.6), ("b", "foe_b", 0.4)),
        reliability=0.5,
    )
    scaled = O.prescaled(logged)
    assert scaled.reliability == 1.0
    assert scaled.moves == (("a", 0.3), ("b", 0.2))
    assert scaled.actions == (("a", "foe_a", 0.3), ("b", "foe_b", 0.2))
    for (_, _, before), (_, _, after) in zip(logged.actions, scaled.actions):
        assert before * logged.reliability == pytest.approx(after * scaled.reliability)


def test_oracle_inputs_by_arm():
    logged_moves = (
        MovePrediction((("x", 1.0),), (), (("x", "foe_a", 1.0),), reliability=0.25),
        MovePrediction((("y", 1.0),), (), (("y", "foe_b", 1.0),), reliability=0.5),
    )
    logged_switches = (SwitchPrediction(0.1, ()), SwitchPrediction(0.2, ()))
    attack = move_action("p2", "a", "stoneedge", E.TARGET_FOE_A)
    switch = E.SlotAction(E.KIND_SWITCH, switch_to="politoed", side="p2", slot="a")
    hidden = E.SlotAction(E.KIND_HIDDEN, reason=E.REASON_FLINCH)

    # B: known slot certain; the unknown slot keeps the logged prediction as it is
    oracle = O.oracle_inputs(
        [attack, hidden], logged_moves, logged_switches, gated=False
    )
    assert oracle.moves is not None and oracle.switches is not None
    assert oracle.moves[0].reliability == 1.0
    assert oracle.moves[0].actions == (("stoneedge", "foe_a", 1.0),)
    assert oracle.moves[1] is logged_moves[1]
    assert oracle.switches == (O.NO_SWITCH, logged_switches[1]) and not oracle.prescaled

    # B with a known switch: the unknown slot must not close the all-slots gate
    opened = O.oracle_inputs(
        [switch, hidden], logged_moves, logged_switches, gated=False
    )
    assert opened.moves is not None and opened.switches is not None
    assert opened.prescaled and opened.moves[1].reliability == 1.0
    assert opened.moves[1].actions == (("y", "foe_b", 0.5),)
    assert opened.switches[0].switch_probability == 1.0
    assert opened.switches[1] == O.NO_SWITCH
    assert all(m.reliability >= O.SWITCH_EVIDENCE_GATE for m in opened.moves)

    # C: the same content at the logged reliability, nothing where the bot had none
    gated = O.oracle_inputs([switch, hidden], logged_moves, logged_switches, gated=True)
    assert gated.moves is not None and gated.switches is not None
    assert gated.moves[0].reliability == 0.25 and gated.moves[1] is logged_moves[1]
    assert gated.switches[0].switch_probability == 1.0 and not gated.prescaled
    assert O.oracle_inputs([attack, hidden], None, None, gated=True).moves is None

    # B without logged predictions (fewer than two actives): certain plus empty
    bare = O.oracle_inputs([attack, None], None, None, gated=False)
    assert bare.moves is not None and bare.switches is not None
    assert bare.moves[1] == O.EMPTY_MOVES and bare.switches[1] == O.NO_SWITCH

    # N: the gates as open as in B, the known slot silent
    null = O.oracle_inputs(
        [attack, hidden], logged_moves, logged_switches, gated=False, content=False
    )
    assert null.moves is not None and null.switches is not None
    assert null.moves[0] == O.EMPTY_MOVES and null.switches[0] == O.NO_SWITCH
    assert null.moves[1] is logged_moves[1]

    truth_moves, truth_switches = O.truth_inputs([attack, hidden])
    assert truth_moves[1] == O.EMPTY_MOVES and truth_switches[1] == O.NO_SWITCH
    assert truth_moves[0].actions == (("stoneedge", "foe_a", 1.0),)


# --- the consumer on a rebuilt position ----------------------------------------------


@pytest.mark.parametrize("seat", ["p1", "p2"])
def test_a_certain_attack_lands_on_the_slot_the_log_names_from_either_seat(seat):
    """``foe_a`` is OUR slot a whichever side the bot sits on."""
    log, player = (LOG, "Alice") if seat == "p1" else (mirrored(LOG), "Bob")
    theirs = "p2" if seat == "p1" else "p1"
    battle = position_at(log, player, seat).battle
    assert battle.player_role == seat and battle.turn == 2
    assert [mon.species for mon in battle.active_pokemon if mon] == [
        "charizard",
        "garchomp",
    ]
    attack = pair(battle, ("heatwave", 0), ("dragonclaw", 1))
    shield_a = pair(battle, ("protect", 0), ("dragonclaw", 1))
    shield_b = pair(battle, ("heatwave", 0), ("protect", 0))

    into_a = [move_action(theirs, "a", "stoneedge", E.TARGET_FOE_A), None]
    moves, switches = O.truth_inputs(into_a)
    assert O.primary_score(battle, attack, moves, switches)[1] > 0.3
    assert O.primary_score(battle, shield_a, moves, switches)[1] == 0.0
    assert O.primary_score(battle, shield_b, moves, switches)[1] > 0.3

    into_b = [move_action(theirs, "a", "stoneedge", E.TARGET_FOE_B), None]
    moves, switches = O.truth_inputs(into_b)
    assert O.primary_score(battle, shield_a, moves, switches)[1] > 0.0
    assert O.primary_score(battle, shield_b, moves, switches)[1] == 0.0

    censored = O.truth_inputs([E.SlotAction(E.KIND_NONE), None])
    assert O.primary_score(battle, attack, *censored) == (0.0, 0.0)


def test_the_oracle_flips_the_pick_only_when_the_gate_is_open():
    battle = position_at(LOG, "Alice", "p1").battle
    attack = pair(battle, ("heatwave", 0), ("dragonclaw", 1))
    shield = pair(battle, ("protect", 0), ("dragonclaw", 1))
    attack.prob, shield.prob = 0.5, 0.45
    candidates = [attack, shield]
    truth = [move_action("p2", "a", "stoneedge", E.TARGET_FOE_A), None]
    no_evidence = (
        MovePrediction((), (), (), reliability=0.0),
        MovePrediction((), (), (), reliability=0.0),
    )
    quiet = (O.NO_SWITCH, O.NO_SWITCH)

    def pick(inputs: O.OracleInputs) -> tuple[int, int]:
        return O.run_consumer(
            battle, candidates, inputs.moves, inputs.switches, config(), False
        ).pick

    assert pick(O.OracleInputs(no_evidence, quiet)) == attack.actions
    assert pick(O.OracleInputs(None, None)) == attack.actions
    oracle = O.oracle_inputs(truth, no_evidence, quiet, gated=False)
    assert pick(oracle) == shield.actions
    gated = O.oracle_inputs(truth, no_evidence, quiet, gated=True)
    assert pick(gated) == attack.actions  # reliability 0: the reranker hears nothing
    null = O.oracle_inputs(truth, no_evidence, quiet, gated=False, content=False)
    assert pick(null) == attack.actions


def test_opening_the_gate_alone_undoes_a_guard_promotion_unless_sticky():
    """What control arm N measures: evidence of any kind makes the reranker
    re-sort by policy probability, so a guard's low-probability pick is lost."""
    battle = position_at(LOG, "Alice", "p1").battle
    promoted = pair(battle, ("airslash", 1), ("dragonclaw", 1))
    favourite = pair(battle, ("heatwave", 0), ("dragonclaw", 1))
    promoted.prob, favourite.prob = 0.2, 0.6
    candidates = [promoted, favourite]  # the guard put its pick first
    silent = O.oracle_inputs(
        [move_action("p2", "a", "trickroom", E.TARGET_AUTO), None],
        None,
        None,
        gated=False,
        content=False,
    )

    def result(sticky: bool, moves: Any, switches: Any) -> O.ArmResult:
        return O.run_consumer(battle, candidates, moves, switches, config(sticky), True)

    assert result(False, None, None).pick == promoted.actions
    undone = result(False, silent.moves, silent.switches)
    assert undone.pick == favourite.actions and not undone.sticky_kept
    kept = result(True, silent.moves, silent.switches)
    assert kept.pick == promoted.actions and kept.sticky_kept


def test_dealt_damage_follows_the_true_board():
    battle = position_at(LOG, "Alice", "p1").battle
    # their slot a: Tyranitar; slot b: Politoed; Sneasler seen on the bench
    into_a = pair(battle, ("airslash", 1), ("dragonclaw", 1))
    into_b = pair(battle, ("airslash", 2), ("dragonclaw", 2))
    stays = [move_action("p2", "a", "stoneedge", E.TARGET_FOE_A), None]
    base, unscored = O.dealt_damage(battle, into_a, stays)
    assert base > 0 and unscored == 0

    protects = [move_action("p2", "a", "protect", E.TARGET_SELF), None]
    assert O.dealt_damage(battle, into_a, protects) == (0.0, 0)
    assert O.dealt_damage(battle, into_b, protects)[0] > 0

    seen = [E.SlotAction(E.KIND_SWITCH, switch_to="sneasler"), None]
    swapped, unscored = O.dealt_damage(battle, into_a, seen)
    assert unscored == 0 and swapped > 0 and swapped != base

    unseen = [E.SlotAction(E.KIND_SWITCH, switch_to="archaludon"), None]
    assert O.dealt_damage(battle, into_a, unseen) == (0.0, 2)
    assert O.dealt_damage(battle, into_b, unseen)[1] == 0

    spread = pair(battle, ("heatwave", 0), ("protect", 0))
    both, _ = O.dealt_damage(battle, spread, stays)
    only_b, _ = O.dealt_damage(battle, spread, protects)
    assert both > only_b > 0


def test_decode_check_compares_with_the_logged_orders():
    battle = position_at(LOG, "Alice", "p1").battle
    action = action_for(battle, 0, "heatwave")
    entry = {
        "actions": [action, 2],
        "orders": [
            PolicyPlayer._audit_order(battle, action, 0),
            PolicyPlayer._audit_order(battle, 2, 1),
        ],
    }
    assert entry["orders"][1] == {"kind": "switch", "species": "garchomp"}
    assert O.decode_matches(battle, {"candidates": [entry]})
    wrong = dict(entry, orders=[{"kind": "move", "id": "struggle"}, entry["orders"][1]])
    assert not O.decode_matches(battle, {"candidates": [wrong]})


def test_change_kind_names_what_a_flip_changes():
    def move(name: str, target: int = 0) -> dict:
        return {"kind": "move", "id": name, "target": target}

    played = [move("heatwave"), move("dragonclaw", 1)]
    assert O.change_kind(played, [move("protect"), move("dragonclaw", 1)]) == "protect"
    assert O.change_kind(played, [move("heatwave"), move("dragonclaw", 2)]) == (
        "retarget"
    )
    switched = [{"kind": "switch", "species": "rillaboom"}, move("rockslide")]
    assert O.change_kind(played, switched) == "move+switch"
    assert O.change_kind(played, played) == "none"


# --- bootstrap -----------------------------------------------------------------------


def test_bootstrap_point_is_the_ratio_of_sums_and_the_interval_brackets_it():
    sums, counts = O.by_cluster(
        [("g1", 1.0), ("g1", 3.0), ("g2", -1.0), ("g3", 2.0), ("g3", 2.0), ("g3", 5.0)]
    )
    assert sums == [4.0, -1.0, 9.0] and counts == [2.0, 1.0, 3.0]
    interval = O.cluster_bootstrap(sums, counts, resamples=2000, seed=1)
    assert interval.point == pytest.approx(12.0 / 6.0)
    assert interval.low is not None and interval.high is not None
    assert interval.low <= 2.0 <= interval.high
    assert (interval.clusters, interval.total, interval.resamples) == (3, 6.0, 2000)
    again = O.cluster_bootstrap(sums, counts, resamples=2000, seed=1)
    assert again == interval
    wide_sums = [float((7 * game) % 11 - 5) for game in range(40)]
    wide = O.cluster_bootstrap(wide_sums, [1.0] * 40, resamples=2000, seed=1)
    assert O.cluster_bootstrap(wide_sums, [1.0] * 40, resamples=2000, seed=2) != wide
    assert O.cluster_bootstrap(wide_sums, [1.0] * 40, resamples=2000, seed=1) == wide


def test_bootstrap_has_no_width_without_variation_and_handles_empty_input():
    flat = O.cluster_bootstrap([2.0, 4.0, 6.0], [1.0, 2.0, 3.0], resamples=500)
    assert (flat.point, flat.low, flat.high) == (2.0, 2.0, 2.0)
    single = O.cluster_bootstrap([3.0], [4.0], resamples=200)
    assert (single.point, single.low, single.high) == (0.75, 0.75, 0.75)
    empty = O.cluster_bootstrap([], [])
    assert empty.point is None and empty.low is None and empty.resamples == 0
    assert O.cluster_bootstrap([1.0], [0.0]).point is None
    only_point = O.cluster_bootstrap([1.0, 2.0], [1.0, 1.0], resamples=0)
    assert only_point.point == 1.5 and only_point.low is None
    with pytest.raises(ValueError):
        O.cluster_bootstrap([1.0, 2.0], [1.0])
    # clusters without observations are resampled too; all-empty draws are dropped
    sparse = O.cluster_bootstrap([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], resamples=1000)
    assert sparse.point == 1.0 and 0 < sparse.resamples < 1000
    assert (sparse.low, sparse.high) == (1.0, 1.0)


def test_bootstrap_resamples_games_not_observations():
    """Ten games that each agree with themselves: the game-clustered interval
    must be far wider than one that treats 200 observations as independent."""
    values = [(f"g{game}", 1.0 if game % 2 else -1.0) for game in range(10)] * 20
    sums, counts = O.by_cluster(values)
    clustered = O.cluster_bootstrap(sums, counts, resamples=4000, seed=3)
    independent = O.cluster_bootstrap(
        [value for _, value in values], [1.0] * len(values), resamples=4000, seed=3
    )
    assert clustered.point == independent.point == 0.0
    assert clustered.low is not None and clustered.high is not None
    assert independent.low is not None and independent.high is not None
    assert clustered.high - clustered.low > 3 * (independent.high - independent.low)
    assert clustered.high - clustered.low == pytest.approx(1.2, abs=0.25)


# --- the reading ---------------------------------------------------------------------


def synthetic_row(index: int, flip: bool, diff: float = 0.5, **over: Any) -> dict:
    """One rebuilt decision with every field ``summarise`` reads."""
    played, other = [7, 7], [8, 8]

    def arm(flipped: bool) -> dict:
        entry: dict[str, Any] = {
            "pick": other if flipped else played,
            "flip": flipped,
            "report": True,
        }
        if flipped:
            entry["orders"] = [{"kind": "move", "id": "protect", "target": 0}] * 2
            entry["hindsight"] = {"primary_diff": diff, "exchange_diff": diff / 2}
        return entry

    row = {
        "dir": "d",
        "battle": f"g{index % 25}",
        "line": index,
        "turn": 1 + index % 3,
        "status": "ok",
        "superseded": False,
        "sheet": O.SHEET_CLOSED,
        "guard_stages": [],
        "sticky": True,
        "mixing_logged": False,
        "played": {"actions": played, "orders": [{"kind": "move", "id": "tackle"}] * 2},
        "truth": [
            {"slot": "a", "kind": "move", "known": True, "target_trusted": True},
            {"slot": "b", "kind": None, "known": False},
        ],
        "truth_class": O.TRUTH_MOVE_ONLY,
        "known_slots": 1,
        "n_live": 3,
        "n_eligible": 2,
        "logged": {"reranker": None},
        "arms": {
            O.ARM_LOGGED: arm(False),
            O.ARM_ORACLE: arm(flip),
            O.ARM_GATED: arm(False),
            O.ARM_WIDE: arm(flip),
            O.ARM_NULL: arm(False),
            "sticky_on": {"oracle_changes_pick": flip},
        },
        "agree": {"pick": True, "report": True},
        "own_action_matches_log": True,
        "flags": [],
    }
    row.update(over)
    return row


def reading(rows: list[dict]) -> dict:
    census = {O.RECORD_MOVE: len(rows)}
    return O.summarise(rows, census, {}, resamples=400)


def test_reading_closes_the_consumer_below_five_percent_of_flips():
    rows = [synthetic_row(index, flip=index < 3) for index in range(100)]
    rows.append({**synthetic_row(100, False), "status": "no_replay"})
    summary = reading(rows)
    verdict = summary["reading_R4"]
    assert verdict["flip_share"] == pytest.approx(3 / 101)
    assert verdict["flips_below_threshold"] and verdict["flips_better_in_hindsight"]
    assert verdict["verdict"] == "consumer_cannot_use_a_predictor"
    arm = summary["arms"][O.ARM_ORACLE]
    assert arm["flips"] == 3 and arm["share_of_rebuilt"] == pytest.approx(0.03)
    assert arm["upper_bound_share"] == pytest.approx(4 / 101)
    assert summary["rebuild"]["status"] == {"ok": 100, "no_replay": 1}
    assert summary["post_hoc"]["B_flips_that_need_the_true_action"] == 3
    assert summary["post_hoc"]["what_those_flips_change"] == {"protect+protect": 3}
    assert "consumer_cannot_use_a_predictor" in O.render_readme(summary)


def test_reading_needs_flips_that_are_better_not_just_many():
    many = [synthetic_row(index, flip=index % 5 == 0) for index in range(200)]
    kept = reading(many)["reading_R4"]
    assert kept["flip_share"] == pytest.approx(0.2)
    assert kept["verdict"] == "not_falsified"
    # three games whose flips all help, two whose flips all hurt
    mixed = [
        synthetic_row(
            index, flip=index % 5 == 0, diff=0.5 if (index % 25) % 10 == 0 else -0.5
        )
        for index in range(200)
    ]
    noisy = reading(mixed)["reading_R4"]
    assert not noisy["flips_below_threshold"]
    assert not noisy["flips_better_in_hindsight"]
    assert noisy["verdict"] == "consumer_cannot_use_a_predictor"
    assert noisy["hindsight_interval"][0] < 0 < noisy["hindsight_interval"][1]


def test_reading_is_refused_when_the_logged_arm_does_not_reproduce():
    rows = [
        synthetic_row(index, flip=False, agree={"pick": index % 5 != 0, "report": True})
        for index in range(100)
    ]
    verdict = reading(rows)["reading_R4"]
    assert verdict["arm_A_pick_rate"] == pytest.approx(0.8)
    assert verdict["verdict"] == "not_read"


def test_flips_are_split_into_unreproduced_opening_and_true_action():
    rows = [synthetic_row(index, flip=False) for index in range(60)]
    rows[0] = synthetic_row(0, flip=True, agree={"pick": False, "report": True})
    opening = synthetic_row(1, flip=True, guard_stages=["some_guard"])
    opening["arms"][O.ARM_NULL] = dict(opening["arms"][O.ARM_ORACLE])
    rows[1] = opening
    rows[2] = synthetic_row(2, flip=True)
    unscored = synthetic_row(3, flip=True)
    unscored["arms"][O.ARM_ORACLE]["hindsight"]["exchange_diff"] = None
    rows[3] = unscored
    summary = reading(rows)
    post = summary["post_hoc"]
    assert post["B_flips"] == 4
    assert post["B_flips_where_arm_A_does_not_reproduce_the_log"] == 1
    assert post["B_flips_the_control_arm_N_makes_too"] == 1
    assert post["of_those_with_a_guard_fired"] == 1
    assert post["B_flips_that_need_the_true_action"] == 2
    arm = summary["arms"][O.ARM_ORACLE]
    assert arm["flips_where_arm_A_reproduces_the_played_pick"] == 3
    assert arm["hindsight_primary"]["n"] == 4
    assert arm["hindsight_exchange"]["n"] == 3
    assert arm["hindsight_exchange"]["left_out"] == 1
    assert arm["by_guard_fired"]["True"] == {"decisions": 1, "flips": 1, "share": 1.0}
    assert summary["gates"]["B_only_blocked_by_the_gates"] == 4


def test_main_writes_the_three_files_even_with_nothing_to_read(tmp_path):
    threads = torch.get_num_threads()
    try:  # the script pins torch to one thread; give the test session its own back
        code = O.main(["--dirs", "no_such_replay_directory*", "--out", str(tmp_path)])
    finally:
        torch.set_num_threads(threads)
    assert code == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["enumeration"][O.RECORD_MOVE] == 0
    assert summary["reading_R4"]["verdict"] == "not_read"
    assert (tmp_path / "decisions.jsonl").read_text() == ""
    assert "not_read" in (tmp_path / "README.md").read_text()


# --- the saved ladder games (git-ignored: skipped when absent) -----------------------


def _ladder_folders() -> list[Path]:
    return sorted(
        path
        for path in ROOT.glob(O.DEFAULT_GLOB)
        if path.is_dir() and (path / "decisions.jsonl").is_file()
    )


def test_census_of_the_saved_audits_adds_up():
    folders = _ladder_folders()
    if not folders:
        pytest.skip("ladder_replays_mc* is not on this machine")
    total: Counter[str] = Counter()
    moves = 0
    for folder in folders:
        text = (folder / "decisions.jsonl").read_text(errors="replace")
        decisions, census, _ = O.read_decisions(text.split("\n"), folder.name)
        assert len(decisions) == census[O.RECORD_MOVE]
        assert all(decision.turn > 0 for decision in decisions)
        total.update(census)
        moves += len(decisions)
    assert total["bad_json"] == 0 and moves > 0
    # the three kinds the critic separated are all present and disjoint
    assert total[O.RECORD_PREVIEW] > 0 and total[O.RECORD_FORCED] > 0


def test_one_saved_directory_rebuilds_and_reproduces_the_played_picks():
    folders = [
        folder
        for folder in _ladder_folders()
        if 3 <= len(list(folder.glob("*.html"))) <= 20
    ]
    usable = []
    for folder in folders:
        loaded = O.read_run_config(folder)
        wanted = (
            [loaded.team, loaded.move_model, loaded.switch_model, loaded.preview_model]
            if loaded is not None
            else [None]
        )
        if all(path is not None and path.is_file() for path in wanted):
            usable.append(folder)
    if not usable:
        pytest.skip("no saved ladder directory with its team and models on disk")
    counters: Counter[str] = Counter()
    rows, census = O.process_directory(usable[0], counters)
    assert len(rows) == census[O.RECORD_MOVE] > 0
    rebuilt = [row for row in rows if row["status"] == "ok"]
    saved = [row for row in rows if row["status"] != "no_replay"]
    assert len(rebuilt) >= 0.9 * len(saved) > 0
    assert not [name for name in counters if "error" in name]
    reproduced = sum(row["agree"]["pick"] for row in rebuilt)
    assert reproduced >= 0.9 * len(rebuilt)
    for row in rebuilt:
        assert set(row["arms"]) >= {"A", "B", "C", "D", "N", "sticky_on"}
        assert len(row["truth"]) == 2
        for name in ("B", "C", "D", "N"):
            arm = row["arms"][name]
            assert ("hindsight" in arm) == arm["flip"]
        assert "pw" not in row["battle"].rsplit("-", 1)[-1]
    summary = O.summarise(rows, census, counters, resamples=200)
    assert summary["rebuild"]["rebuilt"] == len(rebuilt)
    assert (
        summary["arms"]["B"]["flips"]
        >= summary["post_hoc"]["B_flips_that_need_the_true_action"]
    )
