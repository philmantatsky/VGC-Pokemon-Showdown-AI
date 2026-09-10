"""tools/ladder_loss_profile.py parses one game into the feature row the loss
profile splits on: result, sides, leads, faints, Trick Room, Tailwind, hits."""

from __future__ import annotations

from pathlib import Path

from tools.ladder_loss_profile import extract_log, parse_game

LOG = "\n".join(
    [
        "|player|p1|antonius1|170|1257",
        "|player|p2|rival|caitlin|1214",
        "|poke|p1|Floette-Eternal, L50, F|",
        "|poke|p1|Charizard, L50, M|",
        "|poke|p1|Whimsicott, L50, M|",
        "|poke|p1|Garchomp, L50, F|",
        "|poke|p2|Rillaboom, L50, M|",
        "|poke|p2|Sneasler, L50, F|",
        "|poke|p2|Farigiraf, L50, M|",
        "|poke|p2|Kingambit, L50, F|",
        "|start",
        "|switch|p1a: Charizard|Charizard, L50, M|100/100",
        "|switch|p1b: Garchomp|Garchomp, L50, F|100/100",
        "|switch|p2a: Rillaboom|Rillaboom, L50, M|100/100",
        "|switch|p2b: Farigiraf|Farigiraf, L50, M|100/100",
        "|turn|1",
        "|move|p2a: Rillaboom|Fake Out|p1a: Charizard",
        "|-damage|p1a: Charizard|80/100",
        "|move|p2b: Farigiraf|Trick Room|p2b: Farigiraf",
        "|-fieldstart|move: Trick Room|[of] p2b: Farigiraf",
        "|move|p1b: Garchomp|Earthquake|p2a: Rillaboom|[spread] p2a,p2b",
        "|-resisted|p2a: Rillaboom",
        "|-damage|p2a: Rillaboom|70/100",
        "|-supereffective|p2b: Farigiraf",
        "|-damage|p2b: Farigiraf|10/100",
        "|turn|2",
        "|move|p1a: Charizard|Protect|p1a: Charizard",
        "|move|p2a: Rillaboom|Wood Hammer|p1b: Garchomp",
        "|-damage|p1b: Garchomp|0 fnt",
        "|faint|p1b: Garchomp",
        "|switch|p1b: Whimsicott|Whimsicott, L50, M|100/100",
        "|turn|3",
        "|move|p1b: Whimsicott|Tailwind|p1b: Whimsicott",
        "|-sidestart|p1: antonius1|move: Tailwind",
        "|move|p1a: Charizard|Heat Wave|p2a: Rillaboom|[spread] p2a,p2b",
        "|-supereffective|p2a: Rillaboom",
        "|-damage|p2a: Rillaboom|0 fnt",
        "|faint|p2a: Rillaboom",
        "|win|rival",
    ]
)


def test_parse_game_extracts_the_split_features():
    row = parse_game(LOG)
    assert row is not None
    assert row["won"] is False
    assert row["opp_name"] == "rival" and row["opp_rating"] == 1214
    assert row["our_leads"] == ["Charizard", "Garchomp"]
    assert row["opp_leads"] == ["Rillaboom", "Farigiraf"]
    assert row["our_brought"] == ["Charizard", "Garchomp", "Whimsicott"]
    assert row["first_faint_ours"] is True
    assert row["first_faint_turn"] == 2
    assert row["first_faint_species"] == "Garchomp"
    assert row["our_faints"] == 1 and row["opp_faints"] == 1
    assert row["opp_tr"] is True and row["opp_tr_turn"] == 1
    assert row["our_tailwind"] is True and row["our_tailwind_turn"] == 3
    assert row["our_se"] == 2 and row["our_resisted"] == 1
    assert row["opp_se"] == 0
    assert row["our_protects"] == 1
    assert row["our_switches"] == 1  # the post-faint replacement counts
    assert row["turns"] == 3


def test_parse_game_needs_our_side_and_a_winner():
    assert parse_game(LOG.replace("antonius1", "someone")) is None
    assert parse_game(LOG.replace("|win|rival", "")) is None


def test_extract_log_reads_the_embedded_block(tmp_path: Path):
    html = (
        '<html><script type="text/plain" class="battle-log-data">\n'
        "|player|p1|antonius1|170|1257\n|poke|p1|Garchomp, L50, F|&amp;\n"
        "</script></html>"
    )
    path = tmp_path / "game.html"
    path.write_text(html)
    log = extract_log(path)
    assert log is not None and "|player|p1|antonius1" in log
    assert "&" in log and "&amp;" not in log
