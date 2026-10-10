"""evaluation/ladder_blocks.py: a ladder run read block by block, as it was played."""

from __future__ import annotations

import json
import os

from evaluation.ladder_blocks import battle_number, blocks, edges_for, render

PAGE = """<script type="text/plain" class="battle-log-data">
|player|p1|antonius1|1|{ours}
|player|p2|someone|2|{theirs}
|poke|p1|Charizard, L50|
|poke|p2|Garchomp, L50|
|switch|p1a: Charizard|Charizard, L50|100/100
|switch|p2a: Garchomp|Garchomp, L50|100/100
|turn|1
|turn|2
|faint|p2a: Garchomp
|win|{winner}
</script>"""


def test_blocks_end_where_asked_and_the_rest_is_one_more():
    assert edges_for(40, [40, 75, 100]) == [40]
    assert edges_for(81, [40, 75, 100]) == [40, 75, 81]
    assert edges_for(60, None) == [25, 50, 60]
    assert edges_for(0, [40]) == [0]
    # a replay's name and the decision log name the same battle
    assert battle_number("antonius1 - battle-gen9championsvgc2026regmc-2695726662.html")
    assert battle_number(
        "antonius1 - battle-gen9championsvgc2026regmc-2695726662-hj3y6jdq8vqc.html"
    ) == battle_number("battle-gen9championsvgc2026regmc-2695726662")


def _audit(number: int, turn: int, mode: str = "search") -> dict:
    return {
        "battle": f"battle-gen9championsvgc2026regmc-{number}",
        "turn": turn,
        "exact_search": {
            "schedule": {"mode": mode, "reasons": ["a reason"]},
            "result": {"elapsed_s": 2.0, "choice": "move a, move b"},
            "actions": [9, 9],
            "champion_actions": [9, 9],
        },
    }


def test_a_run_is_split_by_game_and_each_block_gets_its_own_rows(tmp_path):
    results = ["antonius1", "someone", "antonius1", "antonius1", "someone"]
    rows = []
    for index, winner in enumerate(results):
        number = 5000 + index
        name = f"antonius1 - battle-gen9championsvgc2026regmc-{number}.html"
        page = tmp_path / name
        page.write_text(PAGE.format(ours=1300 + index, theirs=1320, winner=winner))
        os.utime(page, (1_000_000 + index, 1_000_000 + index))
        rows.append(_audit(number, 1))
        # the fourth game's second decision was not searched
        rows.append(_audit(number, 2, "search_fallback" if index == 3 else "search"))
    (tmp_path / "decisions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    read = blocks(tmp_path, [2, 4])
    assert [row["games"] for row in read] == ["1-2", "3-4", "5-5"]
    first, second, last = (row["block"] for row in read)
    assert (first["wins"], first["games"]) == (1, 2)
    assert (second["wins"], second["games"]) == (2, 2)
    assert (first["decisions"], first["searched"]) == (4, 4)
    assert (second["decisions"], second["searched"]) == (4, 3)
    assert second["not_searched"] == {"search_fallback: a reason": 1}
    assert (
        second["own_rating"]["first"] == 1302 and second["own_rating"]["last"] == 1303
    )
    so_far = read[-1]["so_far"]
    assert (so_far["wins"], so_far["games"], so_far["decisions"]) == (3, 5, 10)
    assert so_far["interval"] is not None and last["performance"] is None  # one loss
    text = "\n".join(render(read))
    assert "games 3-4: won 2 of 2 (100%)" in text and "ALL 5 games: won 3 of 5" in text
    assert "searched 3 of 4 (75%)" in text
