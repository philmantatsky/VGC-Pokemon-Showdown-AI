"""Opponent predictor features: candidates, labels, censored sets, scoring."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vgc_bench.src.oppmodel import events as E
from vgc_bench.src.oppmodel import features as F
from vgc_bench.src.oppmodel import public_state as P

ROOT = Path(__file__).resolve().parents[1]

HEADER = """|j|☆Alice
|j|☆Bob
|gametype|doubles
|player|p1|Alice|1|1300
|player|p2|Bob|2|1250
|gen|9
|tier|[Gen 9 Champions] VGC 2026 Reg M-C
|rated|
|clearpoke
|poke|p1|Incineroar, L50, M|
|poke|p1|Rillaboom, L50, F|
|poke|p1|Charizard, L50, M|
|poke|p1|Farigiraf, L50, F|
|poke|p1|Torkoal, L50, M|
|poke|p1|Venusaur, L50, M|
|poke|p2|Garchomp, L50, M|
|poke|p2|Sneasler, L50, M|
|poke|p2|Politoed, L50, M|
|poke|p2|Archaludon, L50, M|
|poke|p2|Indeedee-F, L50, F|
|poke|p2|Pawmot, L50, M|
|teampreview|4
|teamsize|p1|4
|teamsize|p2|4
|start
|switch|p1a: Incineroar|Incineroar, L50, M|100/100
|switch|p1b: Rillaboom|Rillaboom, L50, F|100/100
|switch|p2a: Garchomp|Garchomp, L50, M|100/100
|switch|p2b: Sneasler|Sneasler, L50, M|100/100
"""

# Seven turns that between them hold every label kind (see the label tests).
GAME = """|turn|1
|detailschange|p2a: Garchomp|Garchomp-Mega, L50, M
|-mega|p2a: Garchomp|Garchomp|Garchompite
|move|p1a: Incineroar|Fake Out|p2b: Sneasler
|-damage|p2b: Sneasler|88/100
|cant|p2b: Sneasler|flinch
|move|p1b: Rillaboom|Protect|p1b: Rillaboom
|-singleturn|p1b: Rillaboom|Protect
|move|p2a: Garchomp|Earthquake|p1a: Incineroar|[spread] p1a,p1b
|-activate|p1b: Rillaboom|move: Protect
|-damage|p1a: Incineroar|41/100
|upkeep
|turn|2
|switch|p1a: Charizard|Charizard, L50, M|100/100
|move|p2b: Sneasler|Close Combat|p1b: Rillaboom
|-damage|p1b: Rillaboom|0 fnt
|faint|p1b: Rillaboom
|move|p2a: Garchomp|Rock Slide|p1a: Charizard|[spread] p1a
|-damage|p1a: Charizard|30/100
|upkeep
|switch|p1b: Torkoal|Torkoal, L50, M|100/100
|turn|3
|move|p2b: Sneasler|Dire Claw|p1a: Charizard
|-damage|p1a: Charizard|0 fnt
|faint|p1a: Charizard
|cant|p1b: Torkoal|slp
|move|p2a: Garchomp|Protect|p2a: Garchomp
|-singleturn|p2a: Garchomp|Protect
|upkeep
|switch|p1a: Incineroar|Incineroar, L50, M|41/100
|turn|4
|move|p1a: Incineroar|Flare Blitz|p2b: Sneasler
|-damage|p2b: Sneasler|0 fnt
|faint|p2b: Sneasler
|move|p1b: Torkoal|Eruption|p2a: Garchomp|[spread] p2a
|-damage|p2a: Garchomp|60/100
|move|p2a: Garchomp|Dragon Claw|p1a: Incineroar
|-damage|p1a: Incineroar|5/100
|upkeep
|switch|p2b: Politoed|Politoed, L50, M|100/100
|turn|5
|move|p2a: Garchomp|Dragon Claw|p1a: Incineroar
|-damage|p1a: Incineroar|0 fnt
|faint|p1a: Incineroar
|move|p2b: Politoed|Icy Wind|p1b: Torkoal|[spread] p1b
|-damage|p1b: Torkoal|80/100
|move|p1b: Torkoal|Heat Wave|p2a: Garchomp|[spread] p2a,p2b
|-damage|p2a: Garchomp|30/100
|-damage|p2b: Politoed|70/100
|upkeep
|turn|6
|move|p2a: Garchomp|Dragon Claw|p1b: Torkoal
|-damage|p1b: Torkoal|40/100
|move|p2b: Politoed|Encore|p1b: Torkoal
|-fail|p1b: Torkoal
|move|p1b: Torkoal|Protect||[still]
|-fail|p1b: Torkoal
|upkeep
|turn|7
|-message|Alice forfeited.
|win|Bob
"""

SHEET_P2 = (
    "|showteam|p2|Garchomp||Garchompite|RoughSkin|Earthquake,DragonClaw,Protect,"
    "RockSlide|Jolly||M|||50|]Sneasler||FocusSash|Unburden|CloseCombat,DireClaw,"
    "FakeOut,Protect|Jolly||M|||50|]Politoed||Leftovers|Drizzle|WeatherBall,"
    "Protect,IcyWind,Encore|Calm||M|||50|"
)

USES = {
    "garchomp": {
        "earthquake": 50,
        "protect": 30,
        "dragonclaw": 10,
        "rockslide": 5,
        "stompingtantrum": 3,
        "swordsdance": 2,
    },
    "sneasler": {"closecombat": 40, "fakeout": 30, "direclaw": 20, "protect": 10},
    "politoed": {"icywind": 5, "encore": 3},
    "whimsicott": {"tailwind": 8},
}


def repertoire() -> F.Repertoire:
    out = F.Repertoire()
    for key, moves in USES.items():
        for move, count in moves.items():
            out.add(key, move, count)
    return out


@pytest.fixture(scope="module")
def fz() -> F.Featurizer:
    return F.Featurizer.build(repertoire())


def drive(body: str = GAME, header: str = HEADER) -> P.DriveResult:
    result = P.drive_log(header + body, "battle-test-1")
    assert result.usable, result.skip_reason
    return result


def turn(result: P.DriveResult, number: int) -> P.TurnRecord:
    return next(record for record in result.turns if record.turn == number)


def encode(fz: F.Featurizer, record: P.TurnRecord, side: str) -> F.Example:
    example = fz.encode_turn(record.snapshot, record.actions, side)
    assert example is not None
    return example


def names(fz: F.Featurizer, example: F.Example, slot: int) -> list[str]:
    return [
        fz.vocab.moves[int(move)]
        for move, flag in zip(example["cand_move"][slot], example["cand_flag"][slot])
        if int(flag) & F.CAND_VALID
    ]


def chosen(fz: F.Featurizer, example: F.Example, slot: int) -> str:
    index = int(example["y_action"][slot])
    assert 0 <= index < fz.n_cand
    return fz.vocab.moves[int(example["cand_move"][slot][index])]


def set_names(fz: F.Featurizer, example: F.Example, slot: int) -> set[str]:
    """The censored set as move names, 'OTHER' and 'switch:<roster index>'."""
    out: set[str] = set()
    for index, on in enumerate(example["y_set"][slot]):
        if not on:
            continue
        if index < fz.n_cand:
            out.add(fz.vocab.moves[int(example["cand_move"][slot][index])])
        elif index == fz.n_cand:
            out.add("OTHER")
        else:
            out.add(f"switch:{index - fz.n_cand - 1}")
    return out


# --- layout -------------------------------------------------------------------


def test_every_array_matches_the_documented_layout(fz: F.Featurizer):
    spec = fz.layout()
    for record in drive().turns:
        for side in E.SIDES:
            example = encode(fz, record, side)
            assert set(example) == set(spec)
            for name, array in example.items():
                assert array.shape == spec[name].shape, name
                assert str(array.dtype) == spec[name].dtype, name
    assert fz.n_actions == F.action_size(12) == 19
    assert F.other_index(12) == 12 and F.switch_index(0, 12) == 13
    groups = {entry.group for entry in spec.values()}
    assert groups == {"feature", "label"}
    turns = [(record.snapshot, record.actions) for record in drive().turns]
    assert len(F.encode_many(fz, turns)) == 2 * 7
    assert len(F.encode_many(fz, turns, sides=["p2"])) == 7
    assert not fz.counters


def test_actor_roster_comes_first_and_targets_stay_absolute(fz: F.Featurizer):
    record = turn(drive(), 2)
    first, second = encode(fz, record, "p1"), encode(fz, record, "p2")
    for name in ("mon_id", "mon_move", "mon_hp", "mon_cat", "mon_flag", "mon_type"):
        assert np.array_equal(first[name][:6], second[name][6:]), name
        assert np.array_equal(first[name][6:], second[name][:6]), name
    assert first["elo"].tolist() == [1300, 1250]
    assert second["elo"].tolist() == [1250, 1300]
    assert first["elo_known"].tolist() == [1, 1]
    species = fz.vocab.species
    assert species[int(second["mon_id"][0, F.ID_SPECIES])] == "garchomp"
    assert species[int(second["mon_id"][0, F.ID_FORME])] == "garchompmega"
    assert species[int(second["mon_id"][0, F.ID_KEY])] == "garchomp"
    assert second["act_mon"].tolist() == [0, 1]
    # p1's slots at turn 2: Incineroar (roster 0) and Rillaboom (roster 1).
    assert second["foe_mon"].tolist() == [6, 7]
    assert first["game_flag"].tolist() == [0, 1, F.SHEET_CLOSED, F.SHEET_CLOSED]
    assert int(first["turn"]) == 2


def test_public_state_reaches_the_mon_rows(fz: F.Featurizer):
    result = drive()
    example = encode(fz, turn(result, 2), "p2")
    garchomp, incineroar = example["mon_flag"][0], example["mon_flag"][6]
    assert garchomp[F.FLAG_IS_MEGA] == 1 and garchomp[F.FLAG_MEGA_POSSIBLE] == 0
    assert garchomp[F.FLAG_FIRST_TURN] == 0 and garchomp[F.FLAG_LAST_MEGA] == 1
    assert example["mon_cat"][0, F.CAT_SLOT] == F.SLOT_A
    assert example["mon_cat"][2, F.CAT_SLOT] == F.SLOT_BENCH
    assert example["mon_cat"][2, F.CAT_BROUGHT] == F.BROUGHT_UNKNOWN
    assert incineroar[F.FLAG_REVEALED] == 1
    assert float(example["mon_hp"][6]) == pytest.approx(0.41, abs=1e-3)
    assert float(example["mon_hp"][2]) == 1.0  # never seen
    moves = fz.vocab.moves
    assert moves[int(example["mon_id"][0, F.ID_LAST_MOVE])] == "earthquake"
    assert example["mon_cat"][0, F.CAT_LAST_KIND] == 1 + F.Y_KIND_MOVE
    assert example["mon_cat"][0, F.CAT_LAST_TARGET] == 1 + F.T_AUTO
    assert example["mon_cat"][1, F.CAT_LAST_KIND] == 1 + F.Y_KIND_HIDDEN
    # Rillaboom protected on turn 1.
    assert example["mon_flag"][7, F.FLAG_PROTECTED_LAST] == 1
    assert example["mon_flag"][7, F.FLAG_LAST_PROTECT] == 1
    assert moves[int(example["mon_move"][0, 0])] == "earthquake"
    assert example["mon_move_flag"][0, 0] == F.MOVE_REVEALED
    stone = fz.vocab.items[int(example["mon_id"][0, F.ID_ITEM])]
    assert stone == "garchompite"
    assert example["mon_cat"][0, F.CAT_ITEM_STATE] == F.ITEM_STATE_CODES[P.ITEM_KNOWN]
    unknown = fz.vocab.items[int(example["mon_id"][1, F.ID_ITEM])]
    assert unknown == "unknown_item"
    types = [fz.vocab.types[int(t) - 1] for t in example["mon_type"][0] if t]
    assert types == ["dragon", "ground"]
    side = example["side_cnt"]
    assert side[0].tolist() == [1, 0, 2, 3, 4]  # mega used, 2 revealed, 2 unknown
    assert side[1].tolist() == [0, 0, 2, 3, 4]


# --- candidates ---------------------------------------------------------------


def test_closed_sheet_candidates_are_revealed_then_repertoire_then_global(
    fz: F.Featurizer,
):
    result = drive()
    opening = encode(fz, turn(result, 1), "p2")
    assert names(fz, opening, 0) == [
        "earthquake",
        "protect",
        "dragonclaw",
        "rockslide",
        "stompingtantrum",
        "swordsdance",
        # the species has six known moves: the rest is the overall ranking
        "closecombat",
        "fakeout",
        "direclaw",
        "tailwind",
        "icywind",
        "encore",
    ]
    flags = opening["cand_flag"][0]
    assert all(int(f) & F.CAND_REPERTOIRE for f in flags[:6])
    assert all(int(f) & F.CAND_GLOBAL for f in flags[6:])
    assert not any(int(f) & (F.CAND_REVEALED | F.CAND_SHEET) for f in flags)
    assert opening["cand_prior"][0, :2].tolist() == pytest.approx([0.5, 0.3], abs=1e-3)
    assert opening["cand_rank"][0].tolist() == [1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0]
    assert float(opening["other_prior"][0]) == 0.0
    assert float(opening["slot_support"][0]) == pytest.approx(np.log1p(100), abs=1e-2)
    # Properties of the move itself, by the dex.
    by_name = dict(zip(names(fz, opening, 0), (int(f) for f in flags)))
    assert by_name["protect"] & F.CAND_PROTECT
    assert not by_name["protect"] & (F.CAND_DAMAGING | F.CAND_AIMED)
    assert (
        by_name["earthquake"] & F.CAND_DAMAGING
        and by_name["earthquake"] & F.CAND_SPREAD
    )
    assert (
        by_name["dragonclaw"] & F.CAND_AIMED
        and not by_name["dragonclaw"] & F.CAND_SPREAD
    )
    assert by_name["fakeout"] & F.CAND_FIRST_TURN

    # Turn 3: Garchomp has shown Earthquake and Rock Slide, in that order.
    third = encode(fz, turn(result, 3), "p2")
    assert names(fz, third, 0)[:4] == [
        "earthquake",
        "rockslide",
        "protect",
        "dragonclaw",
    ]
    assert third["cand_flag"][0, 0] & F.CAND_REVEALED
    assert third["cand_flag"][0, 1] & F.CAND_REVEALED
    assert not third["cand_flag"][0, 2] & F.CAND_REVEALED


def test_four_shown_moves_close_the_candidate_set(fz: F.Featurizer):
    example = encode(fz, turn(drive(), 5), "p2")
    assert names(fz, example, 0) == ["earthquake", "rockslide", "protect", "dragonclaw"]
    mask = example["action_mask"][0]
    assert mask[:4].tolist() == [1, 1, 1, 1] and not mask[4:12].any()
    assert mask[12] == 1  # OTHER stays legal
    assert float(example["other_prior"][0]) == 0.0


def test_open_sheet_candidates_are_the_sheet_moves(fz: F.Featurizer):
    header = HEADER.replace("|teamsize|p1|4", SHEET_P2 + "\n|teamsize|p1|4")
    record = turn(drive(GAME, header), 1)
    example = encode(fz, record, "p2")
    assert names(fz, example, 0) == ["earthquake", "dragonclaw", "protect", "rockslide"]
    assert names(fz, example, 1) == ["closecombat", "direclaw", "fakeout", "protect"]
    assert all(int(f) & F.CAND_SHEET for f in example["cand_flag"][0, :4])
    assert not any(int(f) & F.CAND_REVEALED for f in example["cand_flag"][0, :4])
    assert example["game_flag"].tolist() == [0, 1, F.SHEET_OPEN, F.SHEET_CLOSED]
    assert example["mon_move_flag"][0, 0] == F.MOVE_FROM_SHEET
    # The other actor sees the sheet as the OTHER side's, and has none itself.
    closed = encode(fz, record, "p1")
    assert closed["game_flag"].tolist() == [0, 1, F.SHEET_CLOSED, F.SHEET_OPEN]
    assert len(names(fz, closed, 0)) == 12
    # A sheet move that was then shown carries both flags.
    later = encode(fz, turn(drive(GAME, header), 2), "p2")
    assert later["cand_flag"][0, 0] & F.CAND_REVEALED
    assert later["cand_flag"][0, 0] & F.CAND_SHEET
    assert chosen(fz, later, 0) == "rockslide"


def test_a_species_the_training_split_never_saw_gets_the_overall_ranking():
    blind = F.Featurizer.build(repertoire())
    example = encode(blind, turn(drive(), 1), "p1")  # Incineroar, Rillaboom: unseen
    assert names(blind, example, 0)[:3] == ["earthquake", "closecombat", "protect"]
    assert all(int(f) & F.CAND_GLOBAL for f in example["cand_flag"][0])
    assert float(example["other_prior"][0]) == 1.0
    assert float(example["slot_support"][0]) == 0.0
    assert example["y_flag"][0, F.Y_OTHER] == 0  # Fake Out is in the ranking
    empty = F.Featurizer.build()
    bare = encode(empty, turn(drive(), 1), "p1")
    assert names(empty, bare, 0) == []
    assert int(bare["y_action"][0]) == F.other_index(12)
    assert bare["y_flag"][0, F.Y_OTHER] == 1


def test_switch_pointer_mask(fz: F.Featurizer):
    result = drive()
    first = encode(fz, turn(result, 1), "p2")
    # Two shown, two more were brought: any unseen roster Pokemon may come in.
    assert first["switch_mask"].tolist() == [[0, 0, 1, 1, 1, 1]] * 2
    assert first["action_mask"][0, 13:].tolist() == [0, 0, 1, 1, 1, 1]
    fifth = encode(fz, turn(result, 5), "p2")  # Sneasler fainted, Politoed is in
    assert fifth["switch_mask"][0].tolist() == [0, 0, 0, 1, 1, 1]
    # p1 at turn 3: all four shown, so the two never-seen Pokemon are out.
    third = encode(fz, turn(result, 3), "p1")
    assert third["act_mon"].tolist() == [2, 4]
    assert third["switch_mask"][0].tolist() == [1, 0, 0, 0, 0, 0]
    # p1 at turn 6: only Torkoal stands; the empty slot has no action at all.
    sixth = encode(fz, turn(result, 6), "p1")
    assert sixth["act_mon"].tolist() == [-1, 4]
    assert not sixth["action_mask"][0].any() and not sixth["switch_mask"][0].any()
    assert not sixth["action_mask"][1, 13:].any()


def test_legal_targets(fz: F.Featurizer):
    result = drive()
    both = encode(fz, turn(result, 1), "p2")
    mask = dict(zip(names(fz, both, 0), both["cand_tmask"][0].tolist()))
    foes_and_ally = 1 << F.T_FOE_A | 1 << F.T_FOE_B | 1 << F.T_ALLY
    assert mask["dragonclaw"] == foes_and_ally
    assert mask["earthquake"] == mask["protect"] == 1 << F.T_AUTO
    assert both["cand_tmask"][0, 12] == 0b11111  # OTHER: nothing is ruled out
    # Turn 6: only p1b stands, so a foe-aimed move has one foe to go to.
    lone = encode(fz, turn(result, 6), "p2")
    mask = dict(zip(names(fz, lone, 0), lone["cand_tmask"][0].tolist()))
    assert mask["dragonclaw"] == 1 << F.T_FOE_B | 1 << F.T_ALLY
    expanded = F.expand_target_mask(lone["cand_tmask"])
    assert expanded.shape == (2, 13, 5)
    assert expanded[0, 3].tolist() == [False, True, True, False, False]


# --- labels -------------------------------------------------------------------


def test_visible_move_labels(fz: F.Featurizer):
    result = drive()
    second = encode(fz, turn(result, 2), "p2")
    assert second["y_kind"].tolist() == [F.Y_KIND_MOVE, F.Y_KIND_MOVE]
    assert (
        chosen(fz, second, 0) == "rockslide" and chosen(fz, second, 1) == "closecombat"
    )
    assert second["y_set"].sum(-1).tolist() == [1, 1]
    assert second["y_target"].tolist() == [F.T_AUTO, F.T_FOE_B]
    intents = [E.INTENT_CLASSES[i] for i in second["y_intent"]]
    assert intents == [E.INTENT_SPREAD, E.INTENT_ATTACK_FOE_B]
    # A spread move covers both foe slots, an aimed one its target only.
    assert second["y_attack"].tolist() == [[1, 1], [0, 1]]
    assert second["y_mega"].tolist() == [0, 0]
    assert second["y_flag"][:, F.Y_SWITCH_KNOWN].tolist() == [1, 1]
    assert second["y_flag"][:, F.Y_SWITCHED].tolist() == [0, 0]
    assert second["y_reason"].tolist() == [0, 0]

    third = encode(fz, turn(result, 3), "p2")
    assert chosen(fz, third, 0) == "protect"
    assert (
        third["y_flag"][0, F.Y_PROTECTED] == 1
        and third["y_flag"][1, F.Y_PROTECTED] == 0
    )
    assert E.INTENT_CLASSES[int(third["y_intent"][0])] == E.INTENT_PROTECT
    assert third["y_attack"].tolist() == [[0, 0], [1, 0]]


def test_mega_is_read_with_the_move(fz: F.Featurizer):
    first = encode(fz, turn(drive(), 1), "p2")
    assert first["y_mega"].tolist() == [1, 0]
    assert chosen(fz, first, 0) == "earthquake"
    view = F.slot_view(F.collate([first]))
    assert view["mega_possible"].tolist() == [[True, False]]
    assert view["first_turn"].tolist() == [[True, True]]
    assert view["turn1"].tolist() == [[True, True]]
    assert fz.vocab.species[int(view["key"][0, 0])] == "garchomp"


def test_voluntary_switch_label(fz: F.Featurizer):
    example = encode(fz, turn(drive(), 2), "p1")
    assert example["y_kind"][0] == F.Y_KIND_SWITCH
    assert int(example["y_action"][0]) == F.switch_index(2, 12)  # Charizard
    assert set_names(fz, example, 0) == {"switch:2"}
    assert example["y_flag"][0].tolist() == [1, 1, 0, 1, 0, 0]
    assert E.INTENT_CLASSES[int(example["y_intent"][0])] == E.INTENT_SWITCH
    assert example["y_target"][0] == -1 and example["y_mega"][0] == 0
    assert example["y_attack"][0].tolist() == [0, 0]


def test_flinch_and_faint_exclude_switch_and_protect(fz: F.Featurizer):
    result = drive()
    flinched = encode(fz, turn(result, 1), "p2")
    assert flinched["y_kind"][1] == F.Y_KIND_HIDDEN
    assert F.REASON_NAMES[int(flinched["y_reason"][1])] == E.REASON_FLINCH
    allowed = set_names(fz, flinched, 1)
    assert "protect" not in allowed and "OTHER" in allowed
    assert {"closecombat", "fakeout", "direclaw"} <= allowed
    assert not any(name.startswith("switch") for name in allowed)
    assert flinched["y_action"][1] == -1 and flinched["y_intent"][1] == -1
    assert flinched["y_flag"][1].tolist() == [0, 1, 0, 1, 0, 0]
    assert flinched["y_attack"][1].tolist() == [-1, -1]

    fainted = encode(fz, turn(result, 4), "p2")
    assert fainted["y_kind"][1] == F.Y_KIND_NONE
    assert F.REASON_NAMES[int(fainted["y_reason"][1])] == E.REASON_FAINTED_FIRST
    assert set_names(fz, fainted, 1) == allowed
    assert fainted["y_mega"][1] == 0  # Megas resolve before the knock-out


def test_sleep_allows_any_move_protect_included(fz: F.Featurizer):
    example = encode(fz, turn(drive(), 3), "p1")
    assert example["y_kind"][1] == F.Y_KIND_HIDDEN
    assert F.REASON_NAMES[int(example["y_reason"][1])] == "status"
    allowed = set_names(fz, example, 1)
    assert "protect" in allowed and "OTHER" in allowed
    assert not any(name.startswith("switch") for name in allowed)
    assert example["y_flag"][1, F.Y_SWITCH_KNOWN] == 1
    assert example["y_flag"][1, F.Y_PROTECT_KNOWN] == 0


def test_unresolved_turn_is_fully_masked(fz: F.Featurizer):
    example = encode(fz, turn(drive(), 7), "p2")
    assert example["y_kind"].tolist() == [F.Y_KIND_NONE, F.Y_KIND_NONE]
    assert not example["y_set"].any()
    assert example["y_mega"].tolist() == [-1, -1]
    assert not example["y_flag"].any()
    assert example["y_attack"].tolist() == [[-1, -1], [-1, -1]]
    names_ = [F.REASON_NAMES[int(r)] for r in example["y_reason"]]
    assert names_ == [E.REASON_UNRESOLVED] * 2
    scores = F.slot_nll(
        F.uniform_prediction(F.collate([example])), F.collate([example])
    )
    assert not scores["action_scored"].any() and not scores["mega_scored"].any()


def test_lone_foe_target_is_kept_under_the_legal_mask(fz: F.Featurizer):
    record = turn(drive(), 6)
    assert record.actions["p2a"].target_forced
    assert not record.actions["p2a"].target_trusted
    example = encode(fz, record, "p2")
    assert chosen(fz, example, 0) == "dragonclaw" and chosen(fz, example, 1) == "encore"
    assert example["y_target"].tolist() == [F.T_FOE_B, F.T_FOE_B]
    assert example["foe_mon"].tolist() == [-1, 10]
    # The empty foe slot has no "attacked" label; the standing one has.
    assert example["y_attack"].tolist() == [[-1, 1], [-1, 0]]
    assert E.INTENT_CLASSES[int(example["y_intent"][1])] == E.INTENT_STATUS_FOE


def test_redirected_target_is_not_a_target_label(fz: F.Featurizer):
    body = """|turn|1
|move|p1b: Rillaboom|Follow Me|p1b: Rillaboom
|-singleturn|p1b: Rillaboom|move: Follow Me
|move|p2a: Garchomp|Dragon Claw|p1b: Rillaboom
|-damage|p1b: Rillaboom|60/100
|move|p2b: Sneasler|Fake Out|p1a: Incineroar
|-damage|p1a: Incineroar|90/100
|cant|p1a: Incineroar|flinch
|upkeep
|turn|2
"""
    record = turn(drive(body), 1)
    assert not record.actions["p2a"].target_trusted
    example = encode(fz, record, "p2")
    assert chosen(fz, example, 0) == "dragonclaw"
    assert example["y_target"].tolist() == [-1, F.T_FOE_A]
    # Where it landed is still what happened to the foe's slots.
    assert example["y_attack"].tolist() == [[0, 1], [1, 0]]
    scores = F.slot_nll(
        F.uniform_prediction(F.collate([example])), F.collate([example])
    )
    assert scores["target_scored"].tolist() == [[False, True]]
    assert scores["action_scored"].tolist() == [[True, True]]


def test_forced_continuation_is_not_a_choice(fz: F.Featurizer):
    body = """|turn|1
|move|p2a: Garchomp|Outrage|p1a: Incineroar
|-damage|p1a: Incineroar|40/100
|move|p2b: Sneasler|Protect|p2b: Sneasler
|-singleturn|p2b: Sneasler|Protect
|upkeep
|turn|2
|move|p2a: Garchomp|Outrage|p1b: Rillaboom|[from] lockedmove
|-damage|p1b: Rillaboom|30/100
|cant|p2b: Sneasler|recharge
|upkeep
|turn|3
"""
    result = drive(body)
    assert turn(result, 2).actions["p2a"].locked
    example = encode(fz, turn(result, 2), "p2")
    assert example["y_kind"].tolist() == [F.Y_KIND_MOVE, F.Y_KIND_NONE]
    assert not example["y_set"].any()
    assert example["y_action"].tolist() == [-1, -1]
    assert example["y_flag"][:, F.Y_LOCKED].tolist() == [1, 1]
    assert example["y_intent"].tolist() == [-1, -1]
    assert example["y_attack"][0].tolist() == [-1, -1]  # a random target
    first = encode(fz, turn(result, 1), "p2")
    # Outrage is outside Garchomp's twelve candidates here.
    assert int(first["y_action"][0]) == F.other_index(12)
    assert first["y_flag"][0, F.Y_OTHER] == 1
    assert set_names(fz, first, 0) == {"OTHER"}


def test_mega_on_a_censored_slot_proves_a_move(fz: F.Featurizer):
    body = """|turn|1
|detailschange|p2a: Garchomp|Garchomp-Mega, L50, M
|-mega|p2a: Garchomp|Garchomp|Garchompite
|move|p1a: Incineroar|Flare Blitz|p2a: Garchomp
|-damage|p2a: Garchomp|0 fnt
|faint|p2a: Garchomp
|-message|Bob forfeited.
|win|Alice
"""
    record = turn(drive(body), 1)
    example = encode(fz, record, "p2")
    assert example["y_kind"].tolist() == [F.Y_KIND_NONE, F.Y_KIND_NONE]
    assert example["y_mega"].tolist() == [1, 0]
    allowed = set_names(fz, example, 0)
    assert "OTHER" in allowed and "protect" not in allowed
    assert not any(name.startswith("switch") for name in allowed)
    assert F.REASON_NAMES[int(example["y_reason"][1])] == E.REASON_GAME_ENDED


def test_switch_to_a_pokemon_outside_the_mask_keeps_the_set(fz: F.Featurizer):
    record = turn(drive(), 2)
    example = fz.encode(record.snapshot, "p1")
    assert example is not None
    action = record.actions["p1a"]
    odd = {"p1a": E.SlotAction(**{**action.__dict__, "switch_to": "rillaboom"})}
    before = fz.counters["switch_outside_mask"]
    labels = fz.encode_labels(example, record.snapshot, "p1", odd)
    assert labels is not None
    assert labels["y_action"][0] == -1
    assert labels["y_set"][0, 13:].tolist() == example["switch_mask"][0].tolist()
    assert not labels["y_set"][0, :13].any()
    assert fz.counters["switch_outside_mask"] == before + 1
    fz.counters.pop("switch_outside_mask")
    fz.counters.pop("slot_without_label", None)


# --- Elo ----------------------------------------------------------------------


def test_elo_modes_touch_only_the_elo_fields(fz: F.Featurizer):
    result = drive()
    examples = [encode(fz, record, side) for record in result.turns for side in E.SIDES]
    batch = F.collate(examples)
    assert batch["elo"][0].tolist() == [1300, 1250]
    for mode in F.ELO_MODES:
        changed = F.apply_elo_mode(batch, mode, np.random.default_rng(3))
        assert set(changed) == set(batch)
        for name in batch:
            if name not in ("elo", "elo_known"):
                assert changed[name] is batch[name], (mode, name)
    blank = F.apply_elo_mode(batch, F.ELO_BLANK)
    assert not blank["elo"].any() and not blank["elo_known"].any()
    assert batch["elo"].any()  # the input is not written to
    swapped = F.apply_elo_mode(batch, F.ELO_SWAP)
    assert swapped["elo"][0].tolist() == [1250, 1300]
    shuffled = F.apply_elo_mode(batch, F.ELO_SHUFFLE, np.random.default_rng(1))
    assert sorted(map(tuple, shuffled["elo"].tolist())) == sorted(
        map(tuple, batch["elo"].tolist())
    )
    assert not np.array_equal(shuffled["elo"], batch["elo"])
    assert F.apply_elo_mode(batch, F.ELO_KEEP)["elo"] is batch["elo"]
    assert F.apply_elo_mode(batch, "no such mode")["elo"] is batch["elo"]
    assert F.apply_elo_mode({}, F.ELO_BLANK) == {}


def test_elo_blind_featurizer_and_unknown_rating():
    record = turn(drive(), 1)
    blind = F.Featurizer.build(repertoire(), elo_mode=F.ELO_BLANK)
    seen = F.Featurizer.build(repertoire())
    one, two = blind.encode(record.snapshot, "p2"), seen.encode(record.snapshot, "p2")
    assert one is not None and two is not None
    assert one["elo"].tolist() == [0, 0] and one["elo_known"].tolist() == [0, 0]
    for name in two:
        if name not in ("elo", "elo_known"):
            assert np.array_equal(one[name], two[name]), name
    unrated = HEADER.replace("|player|p2|Bob|2|1250", "|player|p2|Bob|2|")
    example = seen.encode(turn(drive(GAME, unrated), 1).snapshot, "p2")
    assert example is not None
    assert example["elo"].tolist() == [0, 1300]
    assert example["elo_known"].tolist() == [0, 1]


# --- robustness ---------------------------------------------------------------


def test_encode_never_raises():
    fz = F.Featurizer.build(repertoire())
    record = turn(drive(), 1)
    assert fz.encode(None, "p1") is None  # type: ignore[arg-type]
    assert fz.encode(record.snapshot, "p3") is None
    assert fz.encode(SimpleNamespace(sides={}), "p1") is None  # type: ignore[arg-type]
    assert fz.encode_labels({}, record.snapshot, "p1", record.actions) is None
    assert fz.encode_turn(record.snapshot, None, "p1") is None  # type: ignore[arg-type]
    counted = set(fz.counters)
    assert "encode_skip:bad_side" in counted
    assert "encode_skip:side_missing" in counted
    assert any(name.startswith("encode_error:") for name in counted)
    assert any(name.startswith("label_error:") for name in counted)
    # No Pokemon of the actor on the field: nothing to predict.
    preview = P.drive_log(HEADER.split("|start")[0] + "|start\n|turn|1\n", "x")
    assert fz.encode(preview.turns[0].snapshot, "p1") is None
    assert fz.counters["encode_skip:no_active"] == 1
    assert F.collate([]) == {} and F.collate([{"a": np.zeros(2)}, {}])["a"].shape == (
        1,
        2,
    )
    assert F.collate([{"a": np.zeros(2)}, {"a": np.zeros(3)}]) == {}
    assert fz.collate([{"a": np.zeros(2)}, {"a": np.zeros(3)}]) == {}
    assert fz.counters["collate_failed"] == 1
    assert fz.densify({"x": np.zeros(1)}) == {"x": np.zeros(1)}
    assert any(name.startswith("densify_error:") for name in fz.counters)
    # The array helpers a runtime may call degrade to an empty result.
    before = sum(F.COUNTERS.values())
    assert F.slot_view({}) == {} and F.uniform_prediction({}) == {}
    assert F.event_probs({}, {}) == {}
    assert F.intent_probs({}, {}, fz.tables) is None
    assert sum(F.COUNTERS.values()) == before + 4
    assert any(name.startswith("event_probs:") for name in F.COUNTERS)


def test_odd_snapshot_contents_degrade_to_unknown_ids():
    fz = F.Featurizer.build(repertoire())
    body = """|turn|1
|-weather|StrangeSky
|-fieldstart|move: Odd Room
|-sidestart|p1: Alice|move: Odd Wall
|-start|p2a: Garchomp|Odd Curse
|-start|p2a: Garchomp|perish3
|move|p2a: Garchomp|Made Up Move|p1a: Incineroar
|upkeep
|turn|2
"""
    record = turn(drive(body), 2)
    example = fz.encode(record.snapshot, "p2")
    assert example is not None
    assert example["ctx_cat"][F.C_WEATHER] == len(fz.vocab.weathers) + 1  # "other"
    assert fz.counters["field_unknown"] == 1
    assert fz.counters["side_condition_unknown"] == 1
    assert fz.counters["unknown_move"] >= 1
    vols = [int(v) for v in example["mon_vol"][0] if v]
    assert len(vols) == 2 and all(0 < v < fz.vocab.sizes()["volatiles"] for v in vols)
    assert example["mon_move"][0, 0] == 0 and example["cand_move"][0, 0] == 0
    labels = fz.encode_labels(example, record.snapshot, "p2", {})
    assert labels is not None and labels["y_kind"].tolist() == [-1, -1]


def test_weather_field_and_side_condition_ages(fz: F.Featurizer):
    body = """|turn|1
|move|p2a: Garchomp|Sunny Day|p2a: Garchomp
|-weather|SunnyDay
|move|p2b: Sneasler|Tailwind|p2b: Sneasler
|-sidestart|p2: Bob|move: Tailwind
|move|p1a: Incineroar|Trick Room|p1a: Incineroar
|-fieldstart|move: Trick Room|[of] p1a: Incineroar
|move|p1b: Rillaboom|Grassy Terrain|p1b: Rillaboom
|-fieldstart|move: Grassy Terrain
|upkeep
|-weather|SunnyDay|[upkeep]
|turn|2
|upkeep
|turn|3
"""
    result = drive(body)
    before = encode(fz, turn(result, 1), "p2")
    assert before["ctx_cat"].tolist() == [0, 0, F.SETTER_UNKNOWN]
    assert not before["field_age"].any() and not before["side_age"].any()
    second = encode(fz, turn(result, 2), "p2")
    vocab = fz.vocab
    assert vocab.weathers[int(second["ctx_cat"][F.C_WEATHER]) - 1] == "sunnyday"
    assert vocab.terrains[int(second["ctx_cat"][F.C_TERRAIN]) - 1] == "grassyterrain"
    assert second["ctx_cat"][F.C_WEATHER_SETTER] == F.SETTER_ACTOR
    room = 2 + vocab.pseudo.index("trickroom")
    assert second["field_age"][0] == 2 and second["field_age"][1] == 2
    assert second["field_age"][room] == 2  # one end of turn lived
    wind = vocab.side_conditions.index("tailwind")
    assert second["side_age"][0, wind] == 2 and second["side_age"][1, wind] == 0
    third = encode(fz, turn(result, 3), "p1")
    assert third["ctx_cat"][F.C_WEATHER_SETTER] == F.SETTER_OTHER
    assert third["field_age"][room] == 3 and third["side_age"][1, wind] == 3


# --- vocabulary, tables, storage ----------------------------------------------


def test_vocabulary_and_tables(fz: F.Featurizer):
    vocab, tables = fz.vocab, fz.tables
    assert vocab.species[0] == "<unknown>" and "garchompmega" in vocab.species
    assert vocab.moves[0] == "no move"
    assert vocab.items[:3] == ("null", "", "unknown_item")
    assert vocab.abilities[0] == "null"
    assert tables.species_num.shape == (
        len(vocab.species),
        len(F.species_columns(vocab)),
    )
    assert tables.move_num.shape == (len(vocab.moves), len(F.move_columns(vocab)))
    assert len(F.move_columns(vocab)) >= 84 + 6
    assert not tables.species_num[0].any() and not tables.move_num[0].any()
    columns = F.species_columns(vocab)
    base, mega = vocab.species.index("garchomp"), vocab.species.index("garchompmega")
    attack = columns.index("base:atk")
    assert tables.species_num[mega, attack] > tables.species_num[base, attack]
    assert tables.species_num[mega, columns.index("is_mega_forme")] == 1.0
    assert tables.species_num[base, columns.index("has_mega_forme")] == 1.0
    assert tables.species_num[base, columns.index("type:dragon")] == 1.0
    moves = F.move_columns(vocab)
    for name in (
        "protect",
        "fakeout",
        "earthquake",
        "dragonclaw",
        "followme",
        "wideguard",
    ):
        row = tables.move_num[vocab.moves.index(name)]
        assert row[moves.index("protect_family")] == float(E.is_protect_family(name))
        assert row[moves.index("first_turn_only")] == float(E.is_first_turn_only(name))
        assert row[moves.index("spread")] == float(E.is_spread(name))
        assert row[moves.index("damaging")] == float(E.is_damaging(name))
        assert row[moves.index("aimed")] == float(bool(E.is_choosable_target(name)))
        assert row[moves.index("redirection")] == float(E.is_redirection(name))
        assert row[moves.index("known")] == 1.0
    fake_out = tables.move_num[vocab.moves.index("fakeout")]
    assert fake_out[moves.index("priority")] == pytest.approx(3 / 5)
    assert fake_out[moves.index("category:physical")] == 1.0
    assert fake_out[moves.index("target:normal")] == 1.0
    assert fake_out[moves.index("type:normal")] == 1.0
    claw = vocab.moves.index("dragonclaw")
    intents = [int(i) for i in tables.move_intent[claw]]
    assert E.INTENT_CLASSES[intents[F.T_FOE_A]] == E.INTENT_ATTACK_FOE_A
    assert E.INTENT_CLASSES[intents[F.T_AUTO]] == E.INTENT_SPREAD
    assert tables.move_class[claw] == F.CAND_DAMAGING | F.CAND_AIMED
    assert vocab.sizes()["species"] == len(vocab.species)
    assert "trickroom" in vocab.pseudo and "tailwind" in vocab.side_conditions


def test_set_key_follows_the_moveset():
    assert F.set_key("garchompmega", "garchomp") == "garchomp"
    assert F.set_key("charizardmegay", "charizard") == "charizard"
    assert F.set_key("ninetalesalola", "ninetales") == "ninetalesalola"
    assert F.set_key("rotomwash", "rotom") == "rotomwash"
    assert F.set_key("aegislashblade", "aegislash") == "aegislash"
    assert F.set_key("not a forme", "fallback") == "fallback"


def test_repertoire_ranking_and_aim_statistics():
    rep = repertoire()
    order, table, support = rep.lookup("garchomp")
    assert order[:2] == ("earthquake", "protect") and support == 100
    assert table["dragonclaw"] == (0.1, 3)
    assert rep.lookup("nobody") == ((), {}, 0.0)
    assert rep.global_ranked()[:3] == ("earthquake", "closecombat", "protect")
    assert not rep.auto_seen("expandingforce")
    for _ in range(3):
        rep.add("indeedeef", "expandingforce", auto=True, terrain=True)
    rep.add("indeedeef", "expandingforce", auto=False, terrain=True)
    rep.add("indeedeef", "expandingforce", auto=False, terrain=False)
    assert rep.auto_seen("expandingforce")
    assert rep.auto_share("expandingforce", True) == 0.75
    assert rep.auto_share("expandingforce", False) == 0.0
    assert rep.auto_share("dragonclaw", True) == 0.0
    again = F.Repertoire.from_dict(json.loads(json.dumps(rep.to_dict())))
    assert again.lookup("garchomp") == rep.lookup("garchomp")
    assert again.aim == rep.aim
    rep.add("", "earthquake")
    rep.add("garchomp", "earthquake", 0.0)
    assert rep.lookup("garchomp")[2] == 100


def test_terrain_spread_move_is_flagged_from_the_repertoire():
    rep = repertoire()
    for _ in range(4):
        rep.add("indeedeef", "expandingforce", auto=True, terrain=True)
    rep.add("indeedeef", "expandingforce", auto=False, terrain=False)
    fz = F.Featurizer.build(rep)
    lead = HEADER.replace(
        "|switch|p2b: Sneasler|Sneasler, L50, M|100/100",
        "|switch|p2b: Indeedee|Indeedee-F, L50, F|100/100",
    )
    body = """|turn|1
|move|p2b: Indeedee|Expanding Force|p1a: Incineroar
|-damage|p1a: Incineroar|70/100
|move|p2a: Garchomp|Protect|p2a: Garchomp
|-singleturn|p2a: Garchomp|Protect
|-fieldstart|move: Psychic Terrain
|upkeep
|turn|2
|move|p2b: Indeedee|Expanding Force|p1a: Incineroar|[spread] p1a,p1b
|-damage|p1a: Incineroar|30/100
|-damage|p1b: Rillaboom|60/100
|upkeep
|turn|3
"""
    result = drive(body, lead)
    plain, terrain = (
        encode(fz, turn(result, 1), "p2"),
        encode(fz, turn(result, 2), "p2"),
    )
    assert names(fz, plain, 1)[0] == "expandingforce"
    assert float(plain["cand_auto"][1, 0]) == 0.0
    assert float(terrain["cand_auto"][1, 0]) == 1.0
    assert F.has_terrain(turn(result, 2).snapshot)
    assert not F.has_terrain(turn(result, 1).snapshot)
    # 'auto' is a legal target of this aimed move, and the label uses it.
    assert terrain["cand_tmask"][1, 0] >> F.T_AUTO & 1
    assert plain["y_target"][1] == F.T_FOE_A and terrain["y_target"][1] == F.T_AUTO
    assert terrain["y_attack"][1].tolist() == [1, 1]
    protect = names(fz, plain, 0).index("protect")
    assert float(plain["cand_auto"][0, protect]) == 1.0  # never aimed


def test_save_load_round_trip(tmp_path: Path, fz: F.Featurizer):
    fz.save(tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {
        "vocab.json",
        "tables.npz",
        "repertoire.json",
    }
    loaded = F.Featurizer.load(tmp_path)
    assert loaded.signature_diff == [] and loaded.n_cand == fz.n_cand
    assert loaded.vocab.to_dict() == fz.vocab.to_dict()
    assert np.array_equal(loaded.tables.move_num, fz.tables.move_num)
    for record in drive().turns:
        one, two = encode(fz, record, "p2"), encode(loaded, record, "p2")
        for name in one:
            assert np.array_equal(one[name], two[name]), name
    stored = json.loads((tmp_path / "vocab.json").read_text())
    assert stored["dex_signature"] == E.dex_signature()
    assert stored["layout_version"] == F.LAYOUT_VERSION
    # A dataset built with another dex is reported, not silently used.
    payload = fz.to_payload()
    payload["dex_signature"] = {**payload["dex_signature"], "n_moves": 1}
    assert F.Featurizer.from_payload(payload).signature_diff == ["n_moves"]
    with pytest.raises(ValueError):
        F.Featurizer.from_payload(payload, strict=True)
    assert F.Featurizer.load(tmp_path, strict=True).signature_diff == []
    with pytest.raises(ValueError):
        F.Featurizer.from_payload({**fz.to_payload(), "layout_version": 99})
    with pytest.raises(ValueError):
        F.Featurizer.from_payload({"layout_version": F.LAYOUT_VERSION})


def test_ids_the_dex_knows_but_the_vocabulary_lacks_keep_their_numerics():
    full = F.Featurizer.build(repertoire())
    payload = full.to_payload()
    species = [s for s in payload["vocab"]["species"] if s != "sneasler"]
    moves = [m for m in payload["vocab"]["moves"] if m != "closecombat"]
    keep_species = [
        i for i, s in enumerate(payload["vocab"]["species"]) if s != "sneasler"
    ]
    keep_moves = [
        i for i, m in enumerate(payload["vocab"]["moves"]) if m != "closecombat"
    ]
    payload["vocab"] = {**payload["vocab"], "species": species, "moves": moves}
    tables = payload["tables"]
    payload["tables"] = {
        "species_num": tables["species_num"][keep_species],
        "move_num": tables["move_num"][keep_moves],
        "move_intent": tables["move_intent"][keep_moves],
        "move_class": tables["move_class"][keep_moves],
    }
    old = F.Featurizer.from_payload(payload)
    assert old.n_species == len(species) and old.n_moves == len(moves)
    assert old.tables.species_num.shape[0] == old.n_species + 1  # one extension row
    assert old.tables.move_num.shape[0] == old.n_moves + 1
    example = old.encode_turn(turn(drive(), 2).snapshot, turn(drive(), 2).actions, "p2")
    assert example is not None
    assert int(example["mon_id"][1, F.ID_SPECIES]) == old.n_species
    column = names_with_rows(old, example, 1).index("closecombat")
    assert int(example["cand_move"][1, column]) == old.n_moves
    assert int(example["y_action"][1]) == column  # still its own candidate
    dense = old.densify(F.collate([example]), moves=True)
    assert dense["mon_id"][0, 1, F.ID_SPECIES] == 0  # unknown to an embedding
    assert dense["cand_move"][0, 1, column] == 0
    assert dense["mon_id"].dtype == np.uint16 and dense["cand_move"].dtype == np.uint16
    reference = full.densify(
        F.collate([encode(full, turn(drive(), 2), "p2")]), moves=True
    )
    assert np.array_equal(dense["mon_dex"], reference["mon_dex"])
    assert np.array_equal(dense["cand_num"], reference["cand_num"])
    assert np.array_equal(dense["mon_move_num"], reference["mon_move_num"])
    assert dense["mon_dex"][0, 1].any() and dense["cand_num"][0, 1, column].any()
    assert dense["mon_dex"].shape == (1, 12, len(F.species_columns(old.vocab)))
    assert dense["cand_num"].shape == (1, 2, 12, len(F.move_columns(old.vocab)))
    assert dense["mon_move_num"].shape[:3] == (1, 12, 4)


def names_with_rows(fz: F.Featurizer, example: F.Example, slot: int) -> list[str]:
    return [fz.tables.move_ids[int(row)] for row in example["cand_move"][slot]]


def test_batches_shards_and_selection(tmp_path: Path, fz: F.Featurizer):
    examples = [encode(fz, r, side) for r in drive().turns for side in E.SIDES]
    batch = F.collate(examples)
    assert batch["mon_id"].shape == (len(examples), 12, 6)
    F.save_batch(tmp_path / "shard-00000.npz", batch)
    assert [p.name for p in tmp_path.iterdir()] == ["shard-00000.npz"]
    loaded = F.load_batch(tmp_path / "shard-00000.npz")
    assert set(loaded) == set(batch)
    for name in batch:
        assert loaded[name].dtype == batch[name].dtype, name
        assert np.array_equal(loaded[name], batch[name]), name
    half = F.take(batch, np.arange(len(examples)) % 2 == 0)
    assert len(half["turn"]) == (len(examples) + 1) // 2
    joined = F.concat_batches([half, F.take(batch, np.arange(len(examples)) % 2 == 1)])
    assert sorted(joined["turn"].tolist()) == sorted(batch["turn"].tolist())
    assert F.concat_batches([]) == {}


# --- predictor contract and scoring -------------------------------------------


def one_hot_prediction(batch: F.Batch) -> dict[str, np.ndarray]:
    """All mass on the logged action and target; uniform where there is none."""
    pred = F.uniform_prediction(batch)
    for row, slot in np.argwhere(batch["y_action"] >= 0):
        action = int(batch["y_action"][row, slot])
        pred["action"][row, slot] = 0.0
        pred["action"][row, slot, action] = 1.0
        target = int(batch["y_target"][row, slot])
        if target >= 0:
            pred["target"][row, slot, action] = 0.0
            pred["target"][row, slot, action, target] = 1.0
    pred["mega"] = (batch["y_mega"] == 1).astype(np.float64)
    return pred


def test_uniform_prediction_and_set_loss(fz: F.Featurizer):
    examples = [encode(fz, r, "p2") for r in drive().turns]
    batch = F.collate(examples)
    assert isinstance(F.UniformPredictor(), F.Predictor)
    pred = F.UniformPredictor().predict(batch)
    assert pred["action"].shape == (7, 2, 19)
    assert pred["target"].shape == (7, 2, 13, 5) and pred["mega"].shape == (7, 2)
    legal = batch["action_mask"].sum(-1)
    assert np.allclose(pred["action"].sum(-1), 1.0)
    scores = F.slot_nll(pred, batch)
    # Turn 2 (row 1): two visible moves, each one of the legal actions.
    assert scores["action"][1].tolist() == pytest.approx(np.log(legal[1]).tolist())
    assert scores["action_scored"][1].all() and not scores["censored"][1].any()
    # Rock Slide is not aimed: no target loss. Close Combat: foe a, foe b or ally.
    assert scores["target"][1].tolist() == pytest.approx([0.0, np.log(3)])
    assert scores["fine"][1].tolist() == pytest.approx(
        [np.log(legal[1, 0]), np.log(legal[1, 1]) + np.log(3)]
    )
    # Turn 1, slot b: flinch. 13 moves minus Protect, out of 17 legal actions.
    assert scores["censored"][0].tolist() == [False, True]
    assert scores["informative"][0].tolist() == [True, True]
    assert scores["action"][0, 1] == pytest.approx(-np.log(12 / 17))
    # Turn 7: nothing is scored, and unscored entries are 0.
    assert not scores["action_scored"][6].any() and not scores["action"][6].any()
    # Mega: only a Pokemon that could still Mega-evolve is scored.
    assert scores["mega_scored"][0].tolist() == [True, False]
    assert scores["mega"][0, 0] == pytest.approx(np.log(2))
    assert not scores["mega_scored"][1:].any()


def test_a_perfect_prediction_scores_zero_and_a_wrong_one_hits_the_floor(
    fz: F.Featurizer,
):
    batch = F.collate([encode(fz, r, "p2") for r in drive().turns])
    perfect = F.slot_nll(one_hot_prediction(batch), batch)
    visible = batch["y_action"] >= 0
    assert np.allclose(perfect["fine"][visible], 0.0, atol=1e-5)
    assert np.allclose(perfect["mega"][perfect["mega_scored"]], 0.0, atol=1e-5)
    wrong = one_hot_prediction(batch)
    wrong["action"][1, 0] = 0.0
    wrong["action"][1, 0, F.other_index(12)] = 1.0
    scores = F.slot_nll(wrong, batch)
    assert scores["action"][1, 0] == pytest.approx(-np.log(F.PROBABILITY_FLOOR))
    assert np.isfinite(scores["fine"]).all()
    # Mass on an illegal action is removed before scoring.
    leaky = F.uniform_prediction(batch)
    leaky["action"][:, :, 13] = 5.0  # an active Pokemon is not a switch target
    assert np.allclose(
        F.slot_nll(leaky, batch)["action"],
        F.slot_nll(F.uniform_prediction(batch), batch)["action"],
    )
    norm = F.normalize_prediction(leaky, batch)
    assert not norm["action"][:, :, 13].any()
    with pytest.raises(ValueError):
        F.slot_nll({**leaky, "action": leaky["action"][:, :, :5]}, batch)
    with pytest.raises(ValueError):
        F.slot_nll({**leaky, "target": leaky["target"][:, :, :, :3]}, batch)
    bad = F.uniform_prediction(batch)
    bad["action"][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        F.slot_nll(bad, batch)


def test_fine_labels_and_top_k(fz: F.Featurizer):
    batch = F.collate([encode(fz, r, "p2") for r in drive().turns])
    labels = F.fine_label(batch)
    probs = F.fine_probs(one_hot_prediction(batch), batch)
    assert probs.shape == (7, 2, 13 * 5 + 6)
    assert np.allclose(probs.sum(-1)[batch["action_mask"].any(-1)], 1.0)
    # Turn 2: Rock Slide (candidate 3, auto) and Close Combat (candidate 0, foe b).
    assert labels[1].tolist() == [3 * 5 + F.T_AUTO, 0 * 5 + F.T_FOE_B]
    assert labels[0].tolist() == [0 * 5 + F.T_AUTO, -1]  # the flinch has none
    hits, scored = F.topk_hits(probs, labels, 1)
    assert scored.sum() == (labels >= 0).sum() and hits[scored].all()
    hits, _ = F.topk_hits(F.fine_probs(F.uniform_prediction(batch), batch), labels, 3)
    assert not hits[~scored].any()
    switch = F.collate([encode(fz, turn(drive(), 2), "p1")])
    assert F.fine_label(switch)[0, 0] == 13 * 5 + 2
    action_hits, action_scored = F.topk_hits(
        one_hot_prediction(switch)["action"], switch["y_action"], 1
    )
    assert action_hits[action_scored].all()


def test_event_and_intent_probabilities(fz: F.Featurizer):
    batch = F.collate([encode(fz, r, "p2") for r in drive().turns])
    events = F.event_probs(one_hot_prediction(batch), batch)
    # Turn 3: Garchomp protects, Sneasler claws foe a.
    assert events["protect"][2].tolist() == pytest.approx([1.0, 0.0])
    assert np.allclose(events["attack"][2], [[0.0, 0.0], [1.0, 0.0]])
    # Turn 2: Rock Slide spreads over both foes; Close Combat hits foe b.
    assert np.allclose(events["attack"][1], [[1.0, 1.0], [0.0, 1.0]])
    assert events["switch"][1].tolist() == pytest.approx([0.0, 0.0])
    # Turn 6: foe slot a is empty, so nothing is said to be aimed there.
    assert events["attack"][5, 0].tolist() == pytest.approx([0.0, 1.0])
    uniform = F.event_probs(F.uniform_prediction(batch), batch)
    assert uniform["switch"][0, 0] == pytest.approx(4 / 17)
    assert uniform["protect"][0, 0] == pytest.approx(1 / 17)
    intents = F.intent_probs(one_hot_prediction(batch), batch, fz.tables)
    assert intents is not None and intents.shape == (7, 2, 8)
    for row, slot in np.argwhere(batch["y_intent"] >= 0):
        assert intents[row, slot, batch["y_intent"][row, slot]] == pytest.approx(1.0)
    spread = F.intent_probs(F.uniform_prediction(batch), batch, fz.tables)
    assert spread is not None
    assert np.allclose(spread.sum(-1)[batch["action_mask"].any(-1)], 1.0)
    assert spread[0, 0, E.INTENT_CLASSES.index(E.INTENT_SWITCH)] == pytest.approx(
        4 / 17
    )
    assert spread[0, 0, 7] >= 1 / 17  # the OTHER bucket is not assigned
    switched = F.collate([encode(fz, turn(drive(), 2), "p1")])
    assert F.event_probs(one_hot_prediction(switched), switched)["switch"][
        0, 0
    ] == pytest.approx(1.0)


# --- train / serve parity -----------------------------------------------------


def test_live_shadow_gives_the_examples_of_the_offline_path(fz: F.Featurizer):
    """A player-view stream through LiveShadow encodes like the rewritten log."""
    exact = HEADER + GAME
    for public, private in (("100/100", "187/187"), ("41/100", "77/187")):
        exact = exact.replace(
            f"|p1a: Incineroar|{public}", f"|p1a: Incineroar|{private}"
        )
        exact = exact.replace(
            f"|Incineroar, L50, M|{public}", f"|Incineroar, L50, M|{private}"
        )
    events = E.split_log(exact)
    assert any("77/187" in "|".join(event) for event in events)
    rewritten = "\n".join("|".join(P.rewrite_event(e, "p1")) for e in events)
    offline = P.drive_log(rewritten, "battle-test-1", sheets_known=False)
    fake = SimpleNamespace(
        _replay_data=[], player_role="p1", battle_tag="battle-test-1"
    )
    shadow = P.LiveShadow()
    compared = 0
    for event in events:
        fake._replay_data.append(list(event))
        if event[1] != "turn":
            continue
        snapshot = shadow.sync(fake)
        assert snapshot is not None
        record = turn(offline, int(event[2]))
        live, built = fz.encode(snapshot, "p2"), fz.encode(record.snapshot, "p2")
        assert live is not None and built is not None
        for name in built:
            assert np.array_equal(live[name], built[name]), (event[2], name)
        compared += 1
    assert compared == 7
    example = fz.encode(turn(offline, 2).snapshot, "p2")
    assert example is not None
    # The bot's Incineroar reads the public 41 percent, never 77/187.
    assert float(example["mon_hp"][6]) == pytest.approx(0.41, abs=1e-3)
    assert example["game_flag"].tolist() == [0, 1, F.SHEET_UNKNOWN, F.SHEET_UNKNOWN]
    batch = F.collate([example])
    closed = F.sheet_unknown_as_closed(batch)
    assert closed["game_flag"].tolist() == [[0, 1, F.SHEET_CLOSED, F.SHEET_CLOSED]]
    assert batch["game_flag"][0, F.G_ACTOR_SHEET] == F.SHEET_UNKNOWN  # not in place
    assert all(closed[n] is batch[n] for n in batch if n != "game_flag")
    assert F.sheet_unknown_as_closed({}) == {}


# --- team-agnostic rule -------------------------------------------------------


def test_library_holds_no_species_move_item_or_ability_name(fz: F.Featurizer):
    """String literals of the library and the builder name nothing from the dex."""
    known = set(E.moves_dex()) | set(E.pokedex())
    known |= set(fz.vocab.items[3:]) | set(fz.vocab.abilities[3:])
    for path in (
        ROOT / "vgc_bench/src/oppmodel/features.py",
        ROOT / "datagen/oppmodel_build_dataset.py",
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            )
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        }
        found = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and node.value in known
        }
        assert not found, (path.name, sorted(found))
