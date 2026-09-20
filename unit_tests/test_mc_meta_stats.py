"""tools/mc_meta_stats.py: sheets and the winner side from a replay log, the
field events, time-window filtering with replay de-duplication, and usage /
sheet-win-rate arithmetic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.mc_meta_stats import load_games, parse_game, report, snapshot

LOG_A = "\n".join(
    [
        "|player|p1|Alice|a|1500",
        "|player|p2|Bob|b|1480",
        "|poke|p1|Charizard, L50, M|",
        "|poke|p1|Venusaur, L50, F|",
        "|poke|p2|Hatterene, L50, F|",
        "|poke|p2|Urshifu-*, L50, M|",
        "|-weather|SunnyDay|[from] ability: Drought|[of] p1a: Charizard",
        "|-fieldstart|move: Trick Room|[of] p2a: Hatterene",
        "|win|Alice",
    ]
)
LOG_B = "\n".join(
    [
        "|player|p1|Cara|c|1400",
        "|player|p2|Dan|d|1410",
        "|poke|p1|Charizard, L50, M|",
        "|poke|p2|Pelipper, L50, M|",
        "|-sidestart|p2: Dan|move: Tailwind",
        "|-weather|RainDance|[from] ability: Drizzle",
        "|win|Dan",
    ]
)


def test_parse_game_reads_sheets_winner_and_field() -> None:
    game = parse_game(LOG_A)
    assert game["sheets"] == {
        "p1": {"Charizard", "Venusaur"},
        "p2": {"Hatterene", "Urshifu"},
    }
    assert game["winner"] == "p1"
    assert game["trick_room"] and not game["tailwind"]
    assert game["weather"] == {"sun"}


def test_window_filter_and_deduplication(tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps({"r1": [100, LOG_A], "r2": [200, LOG_B]}))
    second.write_text(json.dumps({"r2": [200, LOG_B]}))
    assert len(load_games([first, second], None, None)) == 2
    assert len(load_games([first, second], 100, None)) == 1
    assert len(load_games([first, second], None, 100)) == 1


def test_snapshot_usage_and_sheet_win_rate() -> None:
    snap = snapshot([parse_game(LOG_A), parse_game(LOG_B)])
    assert snap["games"] == 2 and snap["sheets"] == 4
    assert snap["usage"]["Charizard"] == pytest.approx(0.5)
    assert snap["sheet_win_rate"]["Charizard"] == pytest.approx(0.5)
    assert snap["sheet_win_rate"]["Pelipper"] == pytest.approx(1.0)
    assert snap["events"]["trick_room"] == pytest.approx(0.5)
    assert snap["events"]["rain"] == pytest.approx(0.5)
    lines = report(snap, snap, top=2)
    assert lines[0] == "games 2  sheets 4"
    assert any("Charizard" in line and "+0.0" in line for line in lines)
