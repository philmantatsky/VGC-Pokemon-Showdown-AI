"""Reading rules of the opponent predictor's label reader, one excerpt per rule."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from vgc_bench.src.oppmodel import events as E

ROOT = Path(__file__).resolve().parents[1]

HEADER = """|gametype|doubles
|player|p1|Alice|1|1300
|player|p2|Bob|2|1250
|gen|9
|tier|[Gen 9 Champions] VGC 2026 Reg M-C
|clearpoke
|poke|p1|Incineroar, L50, M|
|poke|p1|Rillaboom, L50, F|
|poke|p1|Charizard, L50, M|
|poke|p1|Farigiraf, L50, F|
|poke|p2|Garchomp, L50, M|
|poke|p2|Sneasler, L50, M|
|poke|p2|Politoed, L50, M|
|poke|p2|Archaludon, L50, M|
|teampreview|4
|start
|switch|p1a: Incineroar|Incineroar, L50, M|100/100
|switch|p1b: Rillaboom|Rillaboom, L50, F|100/100
|switch|p2a: Garchomp|Garchomp, L50, M|100/100
|switch|p2b: Sneasler|Sneasler, L50, M|100/100
"""


def read(body: str, counters: Counter[str] | None = None) -> dict[int, dict]:
    """Actions per turn for HEADER + body."""
    pairs = E.read_log_actions(E.split_log(HEADER + body), counters)
    return {segment.turn: actions for segment, actions in pairs}


def test_split_log_keeps_only_pipe_lines():
    text = "Alice rejected open team sheets.\n|move|p1a: X|Protect|p1a: X\r\n\n|\n>tag"
    assert E.split_log(text) == [["", "move", "p1a: X", "Protect", "p1a: X"], ["", ""]]


def test_extract_log_from_html_and_tag():
    page = (
        '<h1>replay</h1>\n<script type="text/plain" class="battle-log-data">\n'
        "    >battle-gen9championsvgc2026regmc-123\n|init|battle\n|turn|1\n"
        "    </script>\n<script>other()</script>"
    )
    log = E.extract_log_from_html(page)
    assert log is not None
    assert E.extract_battle_tag(log) == "battle-gen9championsvgc2026regmc-123"
    assert [event[1] for event in E.split_log(log)] == ["init", "turn"]
    assert E.extract_log_from_html("<html>no log here</html>") is None


def test_ident_parsing_by_slot_letter():
    ident = E.parse_ident("p2b: end of the horizon")
    assert ident is not None and (ident.side, ident.slot, ident.key) == (
        "p2",
        "b",
        "p2b",
    )
    bench = E.parse_ident("p1: Salamence")
    assert bench is not None and bench.slot is None and bench.key is None
    assert E.parse_ident("SunnyDay") is None


def test_base_species_matches_repo_canonical_id():
    preview = pytest.importorskip("vgc_bench.src.opponent_preview")
    samples = [
        "Indeedee-F, L50, F",
        "Charizard-Mega-Y, L50, M",
        "Gengar-Mega, L50, F, shiny",
        "Urshifu-Rapid-Strike, L50, M",
        "Sinistcha, L50",
        "Vivillon-Fancy, L50, M",
        "Zoroark-Hisui, L50, M",
        "Floette-Eternal, L50, F",
        "Not A Real Species, L50",
    ]
    samples += [f"{entry['name']}, L50" for entry in list(E.pokedex().values())[::7]]
    for details in samples:
        assert E.species_from_details(details) == preview.species_id(details), details


def test_move_data_matches_poke_env():
    data = pytest.importorskip("poke_env.data")
    assert E.moves_dex() == data.GenData.from_gen(9).moves


def test_voluntary_switch_before_first_move():
    turns = read(
        """|turn|1
|switch|p2a: Politoed|Politoed, L50, M|100/100
|-weather|RainDance|[from] ability: Drizzle|[of] p2a: Politoed
|move|p1a: Incineroar|Flare Blitz|p2a: Politoed
|-damage|p2a: Politoed|80/100
|move|p2b: Sneasler|Close Combat|p1a: Incineroar
|move|p1b: Rillaboom|Wood Hammer|p2a: Politoed
|upkeep
|turn|2
"""
    )
    action = turns[1]["p2a"]
    assert (action.kind, action.switch_to, action.species) == (
        "switch",
        "politoed",
        "garchomp",
    )
    assert action.intent == E.INTENT_SWITCH
    # The Pokemon that came in is not a turn-start occupant: no label for it.
    assert set(turns[1]) == {"p1a", "p1b", "p2a", "p2b"}
    assert turns[2]["p2a"].species == "politoed"


def test_pivot_drag_and_replacement_are_not_voluntary():
    turns = read(
        """|turn|1
|move|p1a: Incineroar|U-turn|p2a: Garchomp
|-damage|p2a: Garchomp|90/100
|switch|p1a: Charizard|Charizard, L50, M|100/100|[from] U-turn
|move|p2a: Garchomp|Dragon Tail|p1b: Rillaboom
|-damage|p1b: Rillaboom|70/100
|drag|p1b: Farigiraf|Farigiraf, L50, F|100/100
|move|p2b: Sneasler|Close Combat|p1a: Charizard
|-damage|p1a: Charizard|0 fnt
|faint|p1a: Charizard
|upkeep
|switch|p1a: Incineroar|Incineroar, L50, M|100/100
|turn|2
"""
    )
    assert turns[1]["p1a"].kind == "move" and turns[1]["p1a"].move == "uturn"
    dragged = turns[1]["p1b"]
    assert (dragged.kind, dragged.reason) == ("none", E.REASON_LEFT_FIELD)
    assert dragged.switch_known and dragged.protect_known
    assert all(action.kind != "switch" for action in turns[1].values())
    assert turns[2]["p1a"].species == "incineroar"


def test_eject_after_a_hit_is_not_a_choice():
    turns = read(
        """|turn|1
|move|p1a: Incineroar|Fake Out|p2b: Sneasler
|-damage|p2b: Sneasler|93/100
|-enditem|p2b: Sneasler|Eject Button
|switch|p2b: Politoed|Politoed, L50, M|100/100
|move|p2a: Garchomp|Earthquake|p1a: Incineroar|[spread] p1a,p1b
|move|p1b: Rillaboom|Grassy Glide|p2a: Garchomp
|upkeep
"""
    )
    ejected = turns[1]["p2b"]
    assert (ejected.kind, ejected.reason) == ("none", E.REASON_LEFT_FIELD)
    assert ejected.switch_known and ejected.protect_known


def test_untagged_switch_after_the_first_move_is_not_voluntary():
    turns = read(
        """|turn|1
|move|p1a: Incineroar|Flare Blitz|p2b: Sneasler
|-damage|p2b: Sneasler|39/100
|-activate|p2b: Sneasler|ability: Emergency Exit
|switch|p2b: Politoed|Politoed, L50, M|100/100
|move|p2a: Garchomp|Earthquake|p1a: Incineroar|[spread] p1a,p1b
|move|p1b: Rillaboom|Grassy Glide|p2a: Garchomp
|upkeep
"""
    )
    left = turns[1]["p2b"]
    assert (left.kind, left.reason, left.switch_to) == ("none", "left_field", None)
    assert left.switch_known and left.protect_known


def test_eject_in_the_switch_phase_is_not_a_voluntary_switch():
    turns = read(
        """|turn|1
|switch|p1a: Charizard|Charizard, L50, M|100/100
|-ability|p1a: Charizard|Intimidate|boost
|-unboost|p2a: Garchomp|atk|1
|-enditem|p2a: Garchomp|Eject Pack
|switch|p2a: Politoed|Politoed, L50, M|100/100
|move|p2b: Sneasler|Close Combat|p1b: Rillaboom
|move|p1b: Rillaboom|Wood Hammer|p2b: Sneasler
|upkeep
"""
    )
    assert turns[1]["p1a"].kind == "switch"
    ejected = turns[1]["p2a"]
    assert (ejected.kind, ejected.reason) == ("none", E.REASON_LEFT_FIELD)
    # Its own click (a move or a switch) was cancelled before it was shown.
    assert not ejected.switch_known and not ejected.protect_known


def test_only_an_untagged_switch_in_the_switch_phase_is_voluntary():
    turns = read(
        """|turn|1
|drag|p1a: Charizard|Charizard, L50, M|100/100
|switch|p1b: Farigiraf|Farigiraf, L50, F|100/100|[from] Eject Pack
|-mega|p2a: Garchomp|Garchomp|Garchompite
|switch|p2b: Politoed|Politoed, L50, M|100/100
|upkeep
|turn|2
|cant|p1a: Charizard|slp
|switch|p1b: Rillaboom|Rillaboom, L50, F|100/100
|upkeep
|turn|3
|detailschange|p1a: Charizard|Charizard-Mega-Y, L50, M
|switch|p1b: Incineroar|Incineroar, L50, M|100/100
|upkeep
"""
    )
    # Pulled out or pushed out before anything was shown: its click is unknown.
    for key in ("p1a", "p1b"):
        gone = turns[1][key]
        assert (gone.kind, gone.reason) == ("none", E.REASON_LEFT_FIELD)
        assert not gone.switch_known and not gone.protect_known
    # Every voluntary switch resolves before any Mega line and any cant line.
    after_mega = turns[1]["p2b"]
    assert (after_mega.kind, after_mega.reason) == ("none", E.REASON_LEFT_FIELD)
    assert after_mega.switch_known and turns[1]["p2a"].mega
    after_cant = turns[2]["p1b"]
    assert (after_cant.kind, after_cant.species) == ("none", "farigiraf")
    assert after_cant.reason == E.REASON_LEFT_FIELD and after_cant.switch_known
    # A forme line into a Mega forme ends the switch phase like the Mega line.
    after_forme = turns[3]["p1b"]
    assert (after_forme.kind, after_forme.species) == ("none", "rillaboom")
    assert after_forme.switch_known and turns[3]["p1a"].mega


def test_swap_line_that_names_the_current_position_changes_nothing():
    turns = read(
        """|turn|1
|swap|p2a: Garchomp|0
|move|p2a: Garchomp|Dragon Claw|p1a: Incineroar
|move|p2b: Sneasler|Dire Claw|p1b: Rillaboom
|upkeep
|turn|2
"""
    )
    assert (turns[1]["p2a"].species, turns[1]["p2a"].move) == ("garchomp", "dragonclaw")
    assert (turns[1]["p2b"].species, turns[1]["p2b"].move) == ("sneasler", "direclaw")
    assert turns[2]["p2a"].species == "garchomp"
    assert turns[2]["p2b"].species == "sneasler"


def test_item_with_a_follow_up_line_does_not_look_like_an_ejection():
    turns = read(
        """|turn|1
|switch|p1a: Farigiraf|Farigiraf, L50, F|100/100
|-fieldstart|move: Psychic Terrain|[from] ability: Psychic Surge|[of] p1a: Farigiraf
|-enditem|p2a: Garchomp|Psychic Seed
|-boost|p2a: Garchomp|spd|1|[from] item: Psychic Seed
|switch|p2a: Politoed|Politoed, L50, M|100/100
|move|p2b: Sneasler|Close Combat|p1b: Rillaboom
|upkeep
"""
    )
    assert turns[1]["p2a"].kind == "switch"


def test_first_untagged_move_and_instruct_repeat_ignored():
    counters: Counter[str] = Counter()
    turns = read(
        """|turn|1
|move|p1b: Rillaboom|Wood Hammer|p2a: Garchomp
|-damage|p2a: Garchomp|40/100
|move|p1a: Incineroar|Instruct|p1b: Rillaboom
|-singleturn|p1b: Rillaboom|move: Instruct|[of] p1a: Incineroar
|move|p1b: Rillaboom|Wood Hammer|p2b: Sneasler
|-damage|p2b: Sneasler|30/100
|move|p2a: Garchomp|Protect||[still]
|-fail|p2a: Garchomp
|move|p2b: Sneasler|Dire Claw|p1a: Incineroar
|upkeep
""",
        counters,
    )
    action = turns[1]["p1b"]
    assert (action.move, action.target) == ("woodhammer", "foe_a")
    assert counters["repeat_move_ignored"] == 1
    # A failed Protect is still a Protect click, and its target is not a choice.
    failed = turns[1]["p2a"]
    assert failed.is_protect and failed.target == "auto"
    assert failed.intent == E.INTENT_PROTECT


def test_a_second_untagged_move_line_is_never_the_choice():
    counters: Counter[str] = Counter()
    turns = read(
        """|turn|1
|move|p2b: Sneasler|Dire Claw|p1a: Incineroar
|move|p2b: Sneasler|Close Combat|p1b: Rillaboom
|upkeep
""",
        counters,
    )
    action = turns[1]["p2b"]
    assert (action.move, action.target) == ("direclaw", "foe_a")
    assert counters["repeat_move_ignored"] == 1


def test_instruct_before_the_target_moved_skips_the_forced_repeat():
    counters: Counter[str] = Counter()
    turns = read(
        """|turn|1
|move|p1a: Incineroar|Protect|p1a: Incineroar
|-singleturn|p1a: Incineroar|Protect
|move|p2b: Sneasler|Helping Hand|p2a: Garchomp
|-singleturn|p2a: Garchomp|Helping Hand|[of] p2b: Sneasler
|move|p2a: Garchomp|Dragon Claw|p1b: Rillaboom
|move|p1b: Rillaboom|Instruct|p1a: Incineroar
|-fail|p1b: Rillaboom
|upkeep
|turn|2
|move|p1b: Rillaboom|Instruct|p1a: Incineroar
|-singleturn|p1a: Incineroar|move: Instruct|[of] p1b: Rillaboom
|move|p1a: Incineroar|Protect|p1a: Incineroar
|-singleturn|p1a: Incineroar|Protect
|move|p1a: Incineroar|Expanding Force|p2a: Garchomp|[spread] p2a,p2b
|move|p2a: Garchomp|Dragon Claw|p1b: Rillaboom
|move|p2b: Sneasler|Instruct|p2a: Garchomp
|-singleturn|p2a: Garchomp|move: Instruct|[of] p2b: Sneasler
|move|p2a: Garchomp|Dragon Claw|p1b: Rillaboom
|upkeep
|turn|3
|move|p1b: Rillaboom|Instruct|p1a: Incineroar
|-singleturn|p1a: Incineroar|move: Instruct|[of] p1b: Rillaboom
|cant|p1a: Incineroar|par
|move|p1a: Incineroar|Flare Blitz|p2b: Sneasler
|upkeep
""",
        counters,
    )
    # A single-turn effect given by a partner is not a repeat: the helped
    # Pokemon's next move line is its own choice.
    assert turns[1]["p2a"].move == "dragonclaw"
    # The repeat of LAST turn's move came first; the real action follows it.
    real = turns[2]["p1a"]
    assert (real.kind, real.move, real.target) == ("move", "expandingforce", "auto")
    assert not real.is_protect
    # Instructed after it had moved: first line is the choice, as before.
    assert turns[2]["p2a"].move == "dragonclaw" and turns[2]["p2a"].target == "foe_b"
    # The inserted repeat was stopped by paralysis; its own move still came.
    stopped = turns[3]["p1a"]
    assert (stopped.kind, stopped.move, stopped.reason) == ("move", "flareblitz", None)
    # Three repeats skipped, two of them ahead of the Pokemon's own action.
    assert counters["repeat_move_ignored"] == 3
    assert counters["repeat_before_own_move"] == 2


def test_locked_move_and_recharge_are_flagged_locked():
    turns = read(
        """|turn|1
|move|p1a: Incineroar|Outrage|p2a: Garchomp
|move|p2a: Garchomp|Hyper Beam|p1a: Incineroar
|-mustrecharge|p2a: Garchomp
|move|p1b: Rillaboom|Protect|p1b: Rillaboom
|move|p2b: Sneasler|Protect|p2b: Sneasler
|upkeep
|turn|2
|move|p1a: Incineroar|Outrage|p2b: Sneasler|[from] lockedmove
|cant|p2a: Garchomp|recharge
|move|p1b: Rillaboom|Wood Hammer|p2a: Garchomp
|move|p2b: Sneasler|Dire Claw|p1b: Rillaboom
|upkeep
"""
    )
    assert not turns[1]["p1a"].locked and turns[1]["p1a"].target == "auto"
    locked = turns[2]["p1a"]
    assert locked.kind == "move" and locked.locked and locked.reason == E.REASON_FORCED
    recharge = turns[2]["p2a"]
    assert (recharge.kind, recharge.reason, recharge.locked) == ("none", "forced", True)
    assert not turns[2]["p1b"].locked


def test_round_partner_is_a_real_choice_and_called_moves_are_not():
    turns = read(
        """|turn|1
|move|p1a: Incineroar|Round|p2a: Garchomp
|move|p1b: Rillaboom|Round|p2a: Garchomp|[from] move: Round
|move|p2a: Garchomp|Will-O-Wisp|p1a: Incineroar
|move|p1a: Incineroar|Will-O-Wisp|p2a: Garchomp|[from] ability: Magic Bounce
|move|p2b: Sneasler|Copycat|p2b: Sneasler
|move|p2b: Sneasler|Round|p1a: Incineroar|[from] move: Copycat
|upkeep
"""
    )
    partner = turns[1]["p1b"]
    assert (partner.kind, partner.move, partner.target) == ("move", "round", "foe_a")
    assert not partner.locked
    assert turns[1]["p1a"].move == "round"
    assert turns[1]["p2b"].move == "copycat"


def test_blank_still_target_filled_from_anim():
    counters: Counter[str] = Counter()
    turns = read(
        """|turn|1
|move|p2a: Garchomp|Electro Shot||[still]
|-prepare|p2a: Garchomp|Electro Shot
|-boost|p2a: Garchomp|spa|1
|-anim|p2a: Garchomp|Electro Shot|p1b: Rillaboom
|-damage|p1b: Rillaboom|50/100
|move|p1a: Incineroar|Sucker Punch||[still]
|-fail|p1a: Incineroar
|move|p1b: Rillaboom|Wood Hammer|p2a: Garchomp
|move|p2b: Sneasler|Dire Claw|p1a: Incineroar
|upkeep
""",
        counters,
    )
    assert turns[1]["p2a"].target == "foe_b"
    assert counters["target_from_anim"] == 1
    # Nothing tells where the failed move was aimed: unknown, not a guess.
    assert turns[1]["p1a"].kind == "move" and turns[1]["p1a"].target is None
    assert turns[1]["p1a"].intent is None


def test_blank_still_target_filled_from_next_turn_locked_line():
    body = """|turn|1
|move|p2a: Garchomp|Solar Beam||[still]
|-prepare|p2a: Garchomp|Solar Beam
|move|p1a: Incineroar|Protect|p1a: Incineroar
|move|p1b: Rillaboom|Protect|p1b: Rillaboom
|move|p2b: Sneasler|Protect|p2b: Sneasler
|upkeep
|turn|2
|move|p2a: Garchomp|Solar Beam|p1a: Incineroar|[from] lockedmove
|-damage|p1a: Incineroar|60/100
|move|p1a: Incineroar|Flare Blitz|p2a: Garchomp
|move|p1b: Rillaboom|Wood Hammer|p2a: Garchomp
|move|p2b: Sneasler|Dire Claw|p1a: Incineroar
|upkeep
"""
    turns = read(body)
    assert turns[1]["p2a"].target == "foe_a" and not turns[1]["p2a"].locked
    assert turns[2]["p2a"].locked
    # Without the next turn (the serve-time previous-action feature) it stays unknown.
    segments = E.segment_turns(E.split_log(HEADER + body))
    assert E.read_turn_actions(segments[0])["p2a"].target is None


def test_target_class_from_move_data_and_a_spread_tag_on_an_aimed_move():
    counters: Counter[str] = Counter()
    turns = read(
        """|turn|1
|move|p1a: Incineroar|Heat Wave|p2a: Garchomp
|-damage|p2a: Garchomp|70/100
|move|p1b: Rillaboom|Expanding Force|p2b: Sneasler|[spread] p2a,p2b
|-damage|p2a: Garchomp|27/100
|-damage|p2b: Sneasler|12/100
|move|p2a: Garchomp|Earthquake|p1a: Incineroar|[spread] p1a,p1b
|move|p2b: Sneasler|Helping Hand|p2a: Garchomp
|upkeep
|turn|2
|move|p1a: Incineroar|Trick Room|p1a: Incineroar
|move|p1b: Rillaboom|Tailwind|p1b: Rillaboom
|move|p2a: Garchomp|Swords Dance|p2a: Garchomp
|move|p2b: Sneasler|Acupressure|p2b: Sneasler
|upkeep
|turn|3
|move|p2b: Sneasler|Protect|p2b: Sneasler
|-singleturn|p2b: Sneasler|Protect
|move|p1b: Rillaboom|Expanding Force|p2b: Sneasler|[spread] p2a
|-activate|p2b: Sneasler|move: Protect
|-damage|p2a: Garchomp|5/100
|move|p1a: Incineroar|Expanding Force|p2a: Garchomp
|-damage|p2a: Garchomp|1/100
|upkeep
""",
        counters,
    )
    # A spread move with a single foe hit carries no [spread] tag: still 'auto'.
    assert turns[1]["p1a"].target == "auto"
    assert turns[1]["p1a"].intent == E.INTENT_SPREAD
    # Aimed in the move data but tagged [spread]: the simulator made it a spread
    # hit and the slot it names is a random pick, not the click.
    boosted = turns[1]["p1b"]
    assert (boosted.move, boosted.target) == ("expandingforce", "auto")
    assert boosted.intent == E.INTENT_SPREAD
    assert turns[1]["p2a"].target == "auto"
    assert turns[1]["p2b"].target == "ally"
    assert turns[1]["p2b"].intent == E.INTENT_SUPPORT
    assert [turns[2][key].target for key in ("p1a", "p1b", "p2a")] == ["auto"] * 3
    assert turns[2]["p2b"].target == "self"
    # One of the two foes protected: the tag lists one slot and is still a tag.
    assert turns[3]["p1b"].target == "auto"
    assert turns[3]["p1b"].intent == E.INTENT_SPREAD
    # The same move without the tag was aimed: the logged slot is the click.
    aimed = turns[3]["p1a"]
    assert (aimed.target, aimed.target_trusted) == ("foe_a", True)
    assert aimed.intent == E.INTENT_ATTACK_FOE_A
    assert counters["aimed_move_became_spread"] == 2
    assert E.move_intent("expandingforce", "auto") == E.INTENT_SPREAD
    assert E.move_intent("expandingforce", "foe_b") == E.INTENT_ATTACK_FOE_B
    assert not E.is_spread("expandingforce")  # the dex alone cannot tell


def test_same_turn_encore_hides_the_click():
    counters: Counter[str] = Counter()
    turns = read(
        """|turn|1
|move|p1b: Rillaboom|Ancient Power|p2b: Sneasler
|move|p1a: Incineroar|Protect|p1a: Incineroar
|-singleturn|p1a: Incineroar|Protect
|move|p2a: Garchomp|Swords Dance|p2a: Garchomp
|move|p2b: Sneasler|Dire Claw|p1b: Rillaboom
|upkeep
|turn|2
|move|p2b: Sneasler|Encore|p1b: Rillaboom
|-start|p1b: Rillaboom|Encore
|move|p2a: Garchomp|Encore|p1a: Incineroar
|-start|p1a: Incineroar|Encore
|move|p1b: Rillaboom|Ancient Power|p2b: Sneasler
|-damage|p2b: Sneasler|60/100
|move|p1a: Incineroar|Protect||[still]
|-fail|p1a: Incineroar
|upkeep
|turn|3
|move|p1b: Rillaboom|Ancient Power|p2a: Garchomp
|move|p1a: Incineroar|Flare Blitz|p2a: Garchomp
|move|p2a: Garchomp|Encore|p1a: Incineroar
|-start|p1a: Incineroar|Encore
|upkeep
""",
        counters,
    )
    forced = turns[2]["p1b"]
    # The line shows the encored move with a random target, not what was clicked.
    assert (forced.kind, forced.reason) == ("hidden", E.REASON_OVERRIDDEN)
    assert forced.move is None and forced.target is None
    assert forced.forced_move == "ancientpower"
    assert forced.intent is None
    # It did not switch, and a Protect click would have resolved before the
    # lower-priority move that overrode it.
    assert forced.switch_known and forced.protect_known
    # A Protect line behind that move is the forced repeat, never a click.
    shielded = turns[2]["p1a"]
    assert (shielded.kind, shielded.reason) == ("hidden", E.REASON_OVERRIDDEN)
    assert shielded.forced_move == "protect" and not shielded.is_protect
    assert counters["overridden_move"] == 2
    # Encored after it had moved: this turn's click stands.
    free = turns[3]["p1a"]
    assert (free.kind, free.move, free.target) == ("move", "flareblitz", "foe_a")
    assert free.forced_move is None
    # Still encored from last turn: no start line this turn, a normal label.
    assert turns[3]["p1b"].kind == "move" and turns[3]["p1b"].move == "ancientpower"


def test_same_turn_encore_then_a_cant_line_or_a_cure():
    turns = read(
        """|turn|1
|move|p2a: Garchomp|Encore|p1a: Incineroar
|-start|p1a: Incineroar|Encore
|cant|p1a: Incineroar|Disable|Flare Blitz
|move|p2b: Sneasler|Encore|p1b: Rillaboom
|-start|p1b: Rillaboom|Encore
|-end|p1b: Rillaboom|Encore
|move|p1b: Rillaboom|Wood Hammer|p2a: Garchomp
|upkeep
"""
    )
    # A disabled move cannot have been selected: it is the encored one.
    blocked = turns[1]["p1a"]
    assert (blocked.kind, blocked.reason) == ("hidden", E.REASON_OVERRIDDEN)
    assert blocked.move is None and blocked.forced_move == "flareblitz"
    # Cured before it acted: the click stands.
    cured = turns[1]["p1b"]
    assert (cured.kind, cured.move, cured.target) == ("move", "woodhammer", "foe_a")
    assert E.overrides_action("encore") and not E.overrides_action("taunt")
    assert not E.overrides_action("notaneffect")


def test_mega_line_marks_the_slot_even_when_the_move_is_hidden():
    turns = read(
        """|turn|1
|detailschange|p2a: Garchomp|Garchomp-Mega, L50, M
|-mega|p2a: Garchomp|Garchomp|Garchompite
|detailschange|p1a: Incineroar|Charizard-Mega-Y, L50, M
|move|p1b: Rillaboom|Fake Out|p2a: Garchomp
|-damage|p2a: Garchomp|90/100
|cant|p2a: Garchomp|flinch
|move|p1a: Incineroar|Flare Blitz|p2b: Sneasler
|-damage|p2b: Sneasler|0 fnt
|faint|p2b: Sneasler
|upkeep
"""
    )
    flinched = turns[1]["p2a"]
    assert (flinched.kind, flinched.reason, flinched.mega) == ("hidden", "flinch", True)
    assert flinched.switch_known and flinched.protect_known
    # detailschange into a Mega forme alone is enough.
    assert turns[1]["p1a"].mega
    assert not turns[1]["p1b"].mega
    assert turns[1]["p2b"].reason == E.REASON_FAINTED_FIRST


def test_cant_reasons_and_named_move():
    turns = read(
        """|turn|1
|cant|p1a: Incineroar|slp
|cant|p1b: Rillaboom|Disable|Wood Hammer
|cant|p2a: Garchomp|move: Throat Chop
|cant|p2b: Sneasler|par
|upkeep
"""
    )
    asleep = turns[1]["p1a"]
    assert (asleep.kind, asleep.reason) == ("hidden", "slp")
    # A sleeping Pokemon may have clicked Protect; it did not switch.
    assert asleep.switch_known and not asleep.protect_known
    named = turns[1]["p1b"]
    assert (named.kind, named.move, named.target, named.reason) == (
        "move",
        "woodhammer",
        None,
        "disable",
    )
    assert turns[1]["p2a"].reason == "throatchop"
    assert turns[1]["p2b"].reason == "par"


def test_cant_with_of_names_the_ability_holder_not_the_blocked_attacker():
    counters: Counter[str] = Counter()
    turns = read(
        """|turn|1
|move|p1a: Incineroar|Fake Out||[still]
|cant|p2a: Garchomp|ability: Armor Tail|Fake Out|[of] p1a: Incineroar
|move|p2a: Garchomp|Earthquake|p1a: Incineroar|[spread] p1a,p1b
|move|p1b: Rillaboom|Wood Hammer|p2a: Garchomp
|move|p2b: Sneasler|Dire Claw|p1a: Incineroar
|upkeep
""",
        counters,
    )
    holder = turns[1]["p2a"]
    assert (holder.kind, holder.move) == ("move", "earthquake")
    attacker = turns[1]["p1a"]
    assert (attacker.kind, attacker.move, attacker.target) == ("move", "fakeout", None)
    assert counters["cant_blocked_by_ability"] == 1


def test_confusion_self_hit_has_no_cant_line():
    turns = read(
        """|turn|1
|-activate|p1b: Rillaboom|confusion
|-damage|p1b: Rillaboom|88/100|[from] confusion
|-activate|p1a: Incineroar|confusion
|move|p1a: Incineroar|Flare Blitz|p2a: Garchomp
|move|p2a: Garchomp|Protect|p2a: Garchomp
|move|p2b: Sneasler|Dire Claw|p1a: Incineroar
|upkeep
"""
    )
    hit_itself = turns[1]["p1b"]
    assert (hit_itself.kind, hit_itself.reason) == ("hidden", E.REASON_CONFUSION)
    assert hit_itself.switch_known and not hit_itself.protect_known
    assert turns[1]["p1a"].kind == "move"


def test_fainted_before_acting():
    turns = read(
        """|turn|1
|move|p2b: Sneasler|Close Combat|p1b: Rillaboom
|-damage|p1b: Rillaboom|0 fnt
|faint|p1b: Rillaboom
|move|p1a: Incineroar|Flare Blitz|p2b: Sneasler
|move|p2a: Garchomp|Earthquake|p1a: Incineroar|[spread] p1a
|upkeep
|switch|p1b: Charizard|Charizard, L50, M|100/100
|turn|2
"""
    )
    fainted = turns[1]["p1b"]
    assert (fainted.kind, fainted.reason) == ("none", E.REASON_FAINTED_FIRST)
    assert fainted.switch_known and fainted.protect_known
    assert fainted.intent is None


def test_game_ended_mid_turn_versus_turn_never_resolved():
    ended = read(
        """|turn|1
|move|p1a: Incineroar|Flare Blitz|p2a: Garchomp
|-damage|p2a: Garchomp|0 fnt
|faint|p2a: Garchomp
|win|Alice
"""
    )
    waiting = ended[1]["p2b"]
    assert (waiting.kind, waiting.reason) == ("none", E.REASON_GAME_ENDED)
    assert waiting.switch_known and waiting.protect_known
    assert ended[1]["p2a"].reason == E.REASON_FAINTED_FIRST
    forfeit = read(
        """|turn|1
|inactive|Bob has 30 seconds left.
|-message|Bob forfeited.
|
|win|Alice
"""
    )
    assert set(forfeit[1]) == {"p1a", "p1b", "p2a", "p2b"}
    for action in forfeit[1].values():
        assert (action.kind, action.reason) == ("none", E.REASON_UNRESOLVED)
        assert not action.switch_known and not action.protect_known
    # Ended, or knocked out, while the switches were still resolving: the other
    # slots' clicks, a switch included, were never shown.
    early = read(
        """|turn|1
|switch|p1a: Charizard|Charizard, L50, M|100/100
|-damage|p2a: Garchomp|0 fnt|[from] Stealth Rock
|faint|p2a: Garchomp
|win|Alice
"""
    )
    assert early[1]["p1a"].kind == "switch"
    fainted = early[1]["p2a"]
    assert fainted.reason == E.REASON_FAINTED_FIRST
    assert not fainted.switch_known and not fainted.protect_known
    for key in ("p1b", "p2b"):
        assert early[1][key].reason == E.REASON_GAME_ENDED
        assert not early[1][key].switch_known and not early[1][key].protect_known


def test_ally_switch_keeps_actions_with_the_turn_start_slot():
    turns = read(
        """|turn|1
|move|p2a: Garchomp|Ally Switch|p2a: Garchomp
|swap|p2a: Garchomp|1|[from] move: Ally Switch
|move|p1a: Incineroar|Flare Blitz|p2a: Sneasler
|move|p2a: Sneasler|Close Combat|p1b: Rillaboom
|move|p1b: Rillaboom|Wood Hammer|p2b: Garchomp
|upkeep
|turn|2
"""
    )
    assert (
        turns[1]["p2a"].move == "allyswitch" and turns[1]["p2a"].species == "garchomp"
    )
    # Sneasler started the turn in slot b and moved from slot a after the swap.
    moved = turns[1]["p2b"]
    assert (moved.species, moved.move, moved.target) == (
        "sneasler",
        "closecombat",
        "foe_b",
    )
    # Targets are positions: the slot letter as logged.
    assert turns[1]["p1a"].target == "foe_a"
    assert turns[2]["p2a"].species == "sneasler"
    assert turns[2]["p2b"].species == "garchomp"


def test_replace_keeps_the_slot_and_nicknames_may_repeat():
    turns = read(
        """|turn|1
|switch|p2a: Sneasler|Sneasler, L50, M|100/100
|move|p1a: Incineroar|Flare Blitz|p2a: Sneasler
|replace|p2a: Zoroark|Zoroark-Hisui, L50, M
|-end|p2a: Zoroark|Illusion
|move|p2b: Sneasler|Close Combat|p1a: Incineroar
|move|p1b: Rillaboom|Wood Hammer|p2b: Sneasler
|upkeep
|turn|2
|move|p2a: Zoroark|Shadow Ball|p1a: Incineroar
|move|p2b: Sneasler|Dire Claw|p1b: Rillaboom
|upkeep
"""
    )
    # Two Pokemon showed the same nickname in turn 1; slots keep them apart.
    assert turns[1]["p2a"].kind == "switch" and turns[1]["p2b"].move == "closecombat"
    assert turns[2]["p2a"].species == "zoroark" and turns[2]["p2a"].move == "shadowball"
    assert turns[2]["p2b"].species == "sneasler"


def test_revived_pokemon_is_alive_again_after_it_re_enters():
    turns = read(
        """|turn|1
|move|p2a: Garchomp|Earthquake|p1a: Incineroar|[spread] p1a,p1b
|-damage|p1a: Incineroar|0 fnt
|faint|p1a: Incineroar
|move|p1b: Rillaboom|Protect|p1b: Rillaboom
|move|p2b: Sneasler|Protect|p2b: Sneasler
|upkeep
|switch|p1a: Charizard|Charizard, L50, M|100/100
|turn|2
|move|p1a: Charizard|Revival Blessing|p1a: Charizard
|-heal|p1: Incineroar|50/100y|[from] move: Revival Blessing
|move|p1b: Rillaboom|Wood Hammer|p2a: Garchomp
|move|p2a: Garchomp|Protect|p2a: Garchomp
|move|p2b: Sneasler|Dire Claw|p1b: Rillaboom
|upkeep
|turn|3
|switch|p1b: Incineroar|Incineroar, L50, M|50/100y
|move|p1a: Charizard|Heat Wave|p2a: Garchomp|[spread] p2a,p2b
|move|p2a: Garchomp|Earthquake|p1a: Charizard|[spread] p1a,p1b
|move|p2b: Sneasler|Dire Claw|p1b: Incineroar
|upkeep
|turn|4
|move|p1b: Incineroar|Flare Blitz|p2a: Garchomp
|upkeep
"""
    )
    assert "p1a" in turns[2] and turns[2]["p1a"].species == "charizard"
    assert (
        turns[3]["p1b"].kind == "switch" and turns[3]["p1b"].switch_to == "incineroar"
    )
    assert turns[4]["p1b"].species == "incineroar"
    assert turns[4]["p1b"].move == "flareblitz"


def test_target_untrusted_after_redirection_move():
    turns = read(
        """|turn|1
|move|p2b: Sneasler|Follow Me|p2b: Sneasler
|-singleturn|p2b: Sneasler|move: Follow Me
|move|p1a: Incineroar|Flare Blitz|p2b: Sneasler
|move|p1b: Rillaboom|Heat Wave|p2a: Garchomp|[spread] p2a,p2b
|move|p2a: Garchomp|Dragon Claw|p1a: Incineroar
|upkeep
|turn|2
|move|p2b: Sneasler|Rage Powder|p2b: Sneasler
|-singleturn|p2b: Sneasler|move: Rage Powder
|move|p1a: Incineroar|Flare Blitz|p2a: Garchomp
|upkeep
"""
    )
    redirected = turns[1]["p1a"]
    assert redirected.target == "foe_b" and not redirected.target_trusted
    assert turns[1]["p1b"].target_trusted and turns[1]["p2a"].target_trusted
    # The move did not land on the redirecting Pokemon: the logged slot is the click.
    assert turns[2]["p1a"].target == "foe_a" and turns[2]["p1a"].target_trusted


def test_target_untrusted_after_redirecting_ability_and_after_a_faint():
    counters: Counter[str] = Counter()
    turns = read(
        """|turn|1
|move|p1a: Incineroar|Thunderbolt|p2a: Garchomp
|-activate|p2a: Garchomp|ability: Lightning Rod
|-ability|p2a: Garchomp|Lightning Rod|boost
|move|p1b: Rillaboom|Wood Hammer|p2b: Sneasler
|-damage|p2b: Sneasler|0 fnt
|faint|p2b: Sneasler
|move|p2a: Garchomp|Dragon Claw|p1a: Incineroar
|upkeep
|switch|p2b: Politoed|Politoed, L50, M|100/100
|turn|2
|move|p2a: Garchomp|Dragon Claw|p1a: Incineroar
|-damage|p1a: Incineroar|0 fnt
|faint|p1a: Incineroar
|move|p2b: Politoed|Weather Ball|p1b: Rillaboom
|move|p1b: Rillaboom|Wood Hammer|p2b: Politoed
|upkeep
""",
        counters,
    )
    assert not turns[1]["p1a"].target_trusted
    assert counters["target_untrusted_ability"] == 1
    assert turns[1]["p1b"].target_trusted
    # A foe had already fainted this turn: the move may have been retargeted.
    assert not turns[2]["p2b"].target_trusted
    assert turns[2]["p2a"].target_trusted and turns[2]["p1b"].target_trusted


def test_redirecting_ability_distrust_survives_the_anim_line_and_covers_a_partner():
    counters: Counter[str] = Counter()
    turns = read(
        """|turn|1
|move|p1b: Rillaboom|Electro Shot||[still]
|-activate|p2a: Garchomp|ability: Lightning Rod
|-prepare|p1b: Rillaboom|Electro Shot
|-anim|p1b: Rillaboom|Electro Shot|p2a: Garchomp
|move|p2b: Sneasler|Electro Shot||[still]
|-activate|p2a: Garchomp|ability: Lightning Rod
|-anim|p2b: Sneasler|Electro Shot|p2a: Garchomp
|move|p1a: Incineroar|Electro Shot||[still]
|-prepare|p1a: Incineroar|Electro Shot
|-anim|p1a: Incineroar|Electro Shot|p2b: Sneasler
|move|p2a: Garchomp|Thunderbolt|p1a: Incineroar
|-activate|p1a: Incineroar|ability: Lightning Rod
|upkeep
|turn|2
|move|p1a: Incineroar|Flare Blitz|p2b: Sneasler
|-damage|p2b: Sneasler|39/100
|-activate|p2b: Sneasler|ability: Emergency Exit
|move|p1b: Rillaboom|Wood Hammer|p2a: Garchomp
|-activate|p1b: Rillaboom|ability: Battle Bond
|move|p2a: Garchomp|Taunt|p1a: Incineroar
|-activate|p1b: Rillaboom|ability: Aroma Veil|[of] p1a: Incineroar
|upkeep
""",
        counters,
    )
    # The target came from the -anim line; the ability line before it still counts.
    pulled = turns[1]["p1b"]
    assert pulled.target == "foe_a" and not pulled.target_trusted
    # A redirecting ability pulls its partner's move too.
    partner = turns[1]["p2b"]
    assert partner.target == "ally" and not partner.target_trusted
    # No ability line: the -anim target is the click.
    assert turns[1]["p1a"].target == "foe_b" and turns[1]["p1a"].target_trusted
    assert not turns[1]["p2a"].target_trusted
    assert counters["target_untrusted_ability"] == 3
    # Not directly after the move line; the mover's own ability; an ability of a
    # Pokemon the move did not land on: none of them is a redirection.
    assert turns[2]["p1a"].target_trusted
    assert turns[2]["p1b"].target_trusted
    assert turns[2]["p2a"].target == "foe_a" and turns[2]["p2a"].target_trusted


def test_target_forced_when_one_foe_stood_at_turn_start():
    turns = read(
        """|turn|1
|move|p1a: Incineroar|Flare Blitz|p2a: Garchomp
|-damage|p2a: Garchomp|0 fnt
|faint|p2a: Garchomp
|move|p1b: Rillaboom|Wood Hammer|p2b: Sneasler
|move|p2b: Sneasler|Dire Claw|p1a: Incineroar
|upkeep
|turn|2
|move|p1a: Incineroar|Flare Blitz|p2b: Sneasler
|move|p1b: Rillaboom|Helping Hand|p1a: Incineroar
|move|p2b: Sneasler|Dire Claw|p1b: Rillaboom
|upkeep
"""
    )
    # Two foes stood when the turn began: the first label is a click; the second
    # came after a faint (may be retargeted) but was not forced at decision time.
    assert turns[1]["p1a"].target_trusted and not turns[1]["p1a"].target_forced
    assert not turns[1]["p1b"].target_trusted and not turns[1]["p1b"].target_forced
    assert turns[1]["p2b"].target_trusted and not turns[1]["p2b"].target_forced
    # One foe stood at the start of turn 2: every foe-aimed move goes to it.
    lone = turns[2]["p1a"]
    assert (lone.target, lone.target_forced, lone.target_trusted) == (
        "foe_b",
        True,
        False,
    )
    assert lone.intent == E.INTENT_ATTACK_FOE_B  # where it landed is certain
    # An ally target is still a choice, and the side with two foes is untouched.
    assert turns[2]["p1b"].target == "ally" and not turns[2]["p1b"].target_forced
    assert turns[2]["p1b"].target_trusted
    assert turns[2]["p2b"].target_trusted and not turns[2]["p2b"].target_forced


def test_backfilled_target_is_untrusted_when_one_foe_was_left_to_take_it():
    turns = read(
        """|turn|1
|move|p2a: Garchomp|Solar Beam||[still]
|-prepare|p2a: Garchomp|Solar Beam
|move|p2b: Sneasler|Close Combat|p1b: Rillaboom
|-damage|p1b: Rillaboom|0 fnt
|faint|p1b: Rillaboom
|upkeep
|turn|2
|move|p2a: Garchomp|Solar Beam|p1a: Incineroar|[from] lockedmove
|upkeep
"""
    )
    charged = turns[1]["p2a"]
    # Aimed with two foes standing, landed when one was left: not forced at the
    # click, and the slot it hit says nothing about the click.
    assert charged.target == "foe_a"
    assert not charged.target_trusted and not charged.target_forced
    assert turns[2]["p2a"].locked and turns[2]["p2a"].target_forced


def test_struggle_is_forced_and_a_fainted_partner_target_is_an_ally():
    turns = read(
        """|turn|1
|move|p1a: Incineroar|Struggle|p2a: Garchomp
|move|p1b: Rillaboom|Instruct|p1: Charizard|[notarget]
|-fail|p1b: Rillaboom
|move|p2a: Garchomp|Taunt|p1: Charizard|[notarget]
|-fail|p2a: Garchomp
|move|p2b: Sneasler|Dire Claw|p1a: Incineroar
|upkeep
"""
    )
    struggle = turns[1]["p1a"]
    assert struggle.locked and struggle.reason == E.REASON_FORCED
    assert not turns[1]["p2b"].locked
    # A slot-less reference on the mover's own side is its fainted partner,
    # which the simulator keeps as the target.
    instruct = turns[1]["p1b"]
    assert instruct.target == "ally" and instruct.intent == E.INTENT_SUPPORT
    # On the foe side the simulator writes a stand-in, not the click: unknown.
    assert turns[1]["p2a"].target is None


def test_predicates_come_from_dex_properties():
    assert E.is_protect_family("protect") and E.is_protect_family("detect")
    assert E.is_protect_family("spikyshield")
    assert not E.is_protect_family("endure") and not E.is_protect_family("wideguard")
    assert E.is_stalling("endure") and not E.is_stalling("wideguard")
    # The stall counter: stalling moves, and the side guards that add to it.
    assert E.raises_stall_counter("protect") and E.raises_stall_counter("endure")
    assert E.raises_stall_counter("wideguard") and E.raises_stall_counter("quickguard")
    assert not E.raises_stall_counter("craftyshield")
    assert not E.raises_stall_counter("tailwind")
    assert not E.raises_stall_counter("notamove")
    assert E.is_redirection("followme") and E.is_redirection("ragepowder")
    assert not E.is_redirection("helpinghand")
    assert E.is_spread("heatwave") and E.is_spread("earthquake")
    assert not E.is_spread("flareblitz") and not E.is_spread("tailwind")
    assert E.is_choosable_target("flareblitz") is True
    assert E.is_choosable_target("protect") is False
    assert E.is_choosable_target("notamove") is None
    assert E.is_mega_forme("charizardmegay") and not E.is_mega_forme("charizard")


def test_user_id_is_the_account_not_the_display_name():
    assert E.user_id("Piedi nei piedi") == E.user_id("piedi nei piedi")
    assert E.user_id("The_AG_Is_Real") == "theagisreal"
    assert E.user_id(" Some-Name 42 ") == "somename42"
    # Not the Pokemon id rule: the server drops every non-ASCII character.
    assert E.user_id("Algod\u00e3o") == "algodo"
    assert E.to_id("Algod\u00e3o") == "algod\u00e3o"


def test_dex_signature_names_what_the_rules_were_built_from():
    signature = E.dex_signature()
    assert E.dex_available() and signature["dex_available"]
    assert signature["n_moves"] > 500 and signature["n_species"] > 1000
    assert signature["first_turn_only_source"] == E.first_turn_only_source()
    assert signature["first_turn_only_moves"] == sorted(E.first_turn_only_moves())


def test_first_turn_only_flag():
    assert E.is_first_turn_only("fakeout") and E.is_fake_out_like("fakeout")
    assert E.is_fake_out_like("firstimpression")
    assert not E.is_fake_out_like("suckerpunch") and not E.is_fake_out_like("protect")
    assert not E.is_fake_out_like("extremespeed")
    assert E.first_turn_only_source() in ("sim_source", "dex_heuristic")
    if (ROOT / "pokemon-showdown" / "data" / "moves.ts").exists():
        assert E.first_turn_only_source() == "sim_source"
        # Gated on the target's action, not on the user's first turn.
        assert not E.is_first_turn_only("upperhand")


def test_intent_classes():
    assert len(E.INTENT_CLASSES) == 7
    cases = {
        ("protect", "auto"): E.INTENT_PROTECT,
        ("closecombat", "foe_a"): E.INTENT_ATTACK_FOE_A,
        ("closecombat", "foe_b"): E.INTENT_ATTACK_FOE_B,
        ("closecombat", None): None,
        ("heatwave", "auto"): E.INTENT_SPREAD,
        ("outrage", "auto"): E.INTENT_SPREAD,
        ("expandingforce", "auto"): E.INTENT_SPREAD,
        ("instruct", "ally"): E.INTENT_SUPPORT,
        ("willowisp", "foe_b"): E.INTENT_STATUS_FOE,
        ("taunt", None): E.INTENT_STATUS_FOE,
        ("icywind", "auto"): E.INTENT_SPREAD,
        ("trickroom", "auto"): E.INTENT_SUPPORT,
        ("tailwind", "auto"): E.INTENT_SUPPORT,
        ("swordsdance", "auto"): E.INTENT_SUPPORT,
        ("helpinghand", "ally"): E.INTENT_SUPPORT,
        ("pollenpuff", "ally"): E.INTENT_SUPPORT,
        ("healpulse", None): E.INTENT_SUPPORT,
        ("followme", "auto"): E.INTENT_SUPPORT,
        ("notamove", "foe_a"): None,
    }
    for (move, target), expected in cases.items():
        assert E.move_intent(move, target) == expected, (move, target)
    switch = E.SlotAction(kind="switch", switch_to="politoed")
    assert E.intent_class(switch) == E.INTENT_SWITCH
    assert E.intent_class(E.SlotAction(kind="hidden", reason="flinch")) is None


def test_segments_carry_turn_start_occupants():
    segments = E.segment_turns(
        E.split_log(
            HEADER
            + """|turn|1
|move|p1a: Incineroar|Flare Blitz|p2a: Garchomp
|-damage|p2a: Garchomp|0 fnt
|faint|p2a: Garchomp
|upkeep
|turn|2
|win|Alice
"""
        )
    )
    assert [segment.turn for segment in segments] == [1, 2]
    assert set(segments[0].start) == {"p1a", "p1b", "p2a", "p2b"}
    # The fainted slot was not refilled: nobody stands there at turn 2.
    assert set(segments[1].start) == {"p1a", "p1b", "p2b"}
    assert segments[1].ended and not segments[0].ended
    assert segments[0].start["p2b"].species == "sneasler"
    # A turn's start table is a copy: the faint later in the turn is not in it.
    assert not segments[0].start["p2a"].fainted


def test_parse_showteam():
    payload = (
        "Venusaur||FocusSash|Chlorophyll|SludgeBomb,LeafStorm,SleepPowder,Protect"
        "|Modest||M|||50|]Mrs. Fluffy|Indeedee-F|RockyHelmet|PsychicSurge"
        "|Psychic,FollowMe,HelpingHand,Protect|Bold||F|||50|"
    )
    sets = E.parse_showteam(payload)
    assert [entry.species for entry in sets] == ["venusaur", "indeedee"]
    assert sets[0].item == "focussash" and sets[0].ability == "chlorophyll"
    assert sets[0].moves == ("sludgebomb", "leafstorm", "sleeppowder", "protect")
    assert sets[1].forme == "indeedeef" and sets[1].nickname == "Mrs. Fluffy"
    assert E.parse_showteam("garbage") == []


def test_reader_survives_malformed_lines():
    turns = read(
        """|turn|1
|move
|move|p1a: Incineroar
|switch|p9z: Nobody
|cant|p2a: Garchomp
|-anim|p1a
|swap|p1a: Incineroar
|faint|
|turn|not-a-number
|upkeep
"""
    )
    assert set(turns[1]) == {"p1a", "p1b", "p2a", "p2b"}
    assert turns[1]["p2a"].kind == "hidden"


def test_reader_never_raises_on_non_string_fields():
    segment = E.TurnSegment(
        1,
        {"p1a": E.Occupant("incineroar", "incineroar")},
        [["", "move", None, 3]],  # type: ignore[list-item]
    )
    counters: Counter[str] = Counter()
    actions = E.read_turn_actions(segment, None, counters)
    assert actions["p1a"].kind == "none" and not actions["p1a"].switch_known
    assert actions["p1a"].reason == E.REASON_UNRESOLVED
    assert sum(v for k, v in counters.items() if k.startswith("reader_error")) == 1
    # The fallback itself must hold when the turn-start table is broken too.
    broken = E.TurnSegment(
        2,
        {"p1a": None, "p2b": E.Occupant("garchomp", "garchomp"), "zz": 7},  # type: ignore[dict-item]
        [["", "move", None, 3]],  # type: ignore[list-item]
    )
    actions = E.read_turn_actions(broken, None, counters)
    assert set(actions) == {"p1a", "p2b"}
    assert actions["p1a"].species == "" and actions["p2b"].species == "garchomp"
    assert all(action.kind == "none" for action in actions.values())
    assert E.read_turn_actions(E.TurnSegment(3, None)) == {}  # type: ignore[arg-type]


def test_called_move_before_the_users_own_move_is_not_its_choice():
    counters: Counter[str] = Counter()
    turns = read(
        """|turn|1
|move|p1a: Incineroar|Will-O-Wisp|p2a: Garchomp
|move|p2a: Garchomp|Will-O-Wisp|p1a: Incineroar|[from] ability: Magic Bounce
|move|p2a: Garchomp|Earthquake|p1a: Incineroar|[spread] p1a,p1b
|upkeep
""",
        counters,
    )
    assert turns[1]["p2a"].move == "earthquake"
    assert counters["called_move_ignored"] == 1


def test_blank_target_is_not_filled_from_a_free_choice_next_turn():
    turns = read(
        """|turn|1
|move|p1a: Incineroar|Sucker Punch||[still]
|-fail|p1a: Incineroar
|upkeep
|turn|2
|move|p1a: Incineroar|Sucker Punch|p2a: Garchomp
|upkeep
"""
    )
    assert turns[1]["p1a"].target is None
    assert turns[2]["p1a"].target == "foe_a"


def test_events_of_a_pokemon_that_entered_this_turn_do_not_reach_the_old_slot():
    turns = read(
        """|turn|1
|move|p2a: Garchomp|Dragon Tail|p1b: Rillaboom
|-damage|p1b: Rillaboom|70/100
|drag|p1b: Farigiraf|Farigiraf, L50, F|100/100
|move|p2b: Sneasler|Close Combat|p1b: Farigiraf
|-damage|p1b: Farigiraf|0 fnt
|faint|p1b: Farigiraf
|move|p1a: Incineroar|Flare Blitz|p2a: Garchomp
|upkeep
"""
    )
    dragged = turns[1]["p1b"]
    assert (dragged.species, dragged.kind, dragged.reason) == (
        "rillaboom",
        "none",
        E.REASON_LEFT_FIELD,
    )


def _corpus_logs(limit: int) -> list[Path]:
    folder = ROOT / "battle_logs_web_mc_20260927" / "top"
    return sorted(folder.glob("*.log"))[:limit] if folder.is_dir() else []


def test_reader_on_real_logs_every_living_slot_gets_one_action():
    paths = _corpus_logs(40)
    if not paths:
        pytest.skip("battle_logs_web_mc_20260927 is not on this machine")
    kinds: Counter[str] = Counter()
    for path in paths:
        events = E.split_log(path.read_text(encoding="utf-8", errors="replace"))
        for segment, actions in E.read_log_actions(events):
            assert set(actions) == set(segment.start)
            for key, action in actions.items():
                assert action.key == key and action.side == key[:2]
                assert action.kind in ("move", "switch", "hidden", "none")
                if action.target is not None:
                    assert action.target in E.TARGET_CLASSES
                kinds[action.kind] += 1
    total = sum(kinds.values())
    assert total > 1000 and kinds["move"] / total > 0.6
