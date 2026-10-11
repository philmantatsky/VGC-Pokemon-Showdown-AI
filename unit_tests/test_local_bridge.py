"""The local bridge (evaluation/local_bridge.py): serial games between the deployed
bot and an external client on a local server, in blocks of who challenges x sheets
open or closed. The server-side loop is tested here with a stand-in for the bot."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from checks.local_bridge_standin import parse as standin_args
from evaluation.local_bridge import (
    Block,
    _Remembering,
    _run,
    claim_line,
    client_command,
    parse_blocks,
    read_games,
    recorded,
    render,
    tally,
)


def test_a_block_says_who_challenges_what_the_bot_answers_and_how_many():
    blocks = parse_blocks("they:closed:75, they:open:75,we:closed:75,we:open:75")
    assert [(b.index, b.who, b.sheets, b.games, b.same) for b in blocks] == [
        (0, "they", "closed", 75, False),
        (1, "they", "open", 75, False),
        (2, "we", "closed", 75, False),
        (3, "we", "open", 75, False),
    ]
    assert blocks[0].label == "the client challenges, sheets closed"
    same = parse_blocks("we:open:20:same")[0]
    assert same.same and same.spec == "we:open:20:same"
    assert same.label == "the bot challenges, sheets open, our team on both sides"
    for bad in (
        "",
        "they:closed",
        "us:open:5",
        "we:shut:5",
        "we:open:0",
        "we:open:5:x",
    ):
        with pytest.raises(ValueError):
            parse_blocks(bad)


def test_the_client_is_told_its_side_of_a_block():
    template = (
        "client {port} {opponent} {bot} {mode} {games} {sheets} [{team}] {format}"
    )
    they, same = parse_blocks("they:closed:5,we:open:3:same")
    assert client_command(template, they, 7650, "vgczero", "pfb1", "/t/T6e.txt") == (
        "client 7650 vgczero pfb1 challenge 5 closed [] gen9championsvgc2026regmc"
    )
    assert client_command(template, same, 7650, "vgczero", "pfb1", "/t/T6e.txt") == (
        "client 7650 vgczero pfb1 accept 3 open [/t/T6e.txt] gen9championsvgc2026regmc"
    )


def test_better_is_claimed_at_the_plans_count():
    """The plan: 168 of 300 claims "better" (one-sided 2.5% under even odds)."""
    assert claim_line(300) == 168
    assert claim_line(6) == 6 and claim_line(5) is None
    assert claim_line(100) == 61 and claim_line(0) is None


def _row(block: Block, won, seen=None, turns=8):
    return {
        "block": block.index,
        "spec": block.spec,
        "label": block.label,
        "who": block.who,
        "sheets": block.sheets,
        "same": block.same,
        "won": won,
        "turns": turns,
        "sheets_seen": seen or block.sheets,
        "sheets_as_asked": (seen or block.sheets) == block.sheets,
    }


def test_the_tally_keeps_blocks_and_arms_apart():
    they, we, same = parse_blocks("they:closed:4,we:open:3,we:closed:2:same")
    rows = (
        [_row(they, False), _row(they, False), _row(they, True), _row(they, None)]
        + [_row(we, False), _row(we, True, seen="closed"), _row(we, True)]
        + [_row(same, False), _row(same, False)]
    )
    reading = tally(rows)
    first = reading["blocks"]["the client challenges, sheets closed"]
    assert (first["games"], first["client_wins"], first["bot_wins"], first["ties"]) == (
        4,
        2,
        1,
        1,
    )
    assert first["client_share"] == pytest.approx(2 / 3)  # the tie is not a game won
    second = reading["blocks"]["the bot challenges, sheets open"]
    assert second["sheets_not_as_asked"] == 1
    # the client brought our six in a block of its own team: counted, not hidden
    mirrored = dict(_row(we, True), same_team_seen=True)
    assert tally([mirrored])["blocks"][we.label]["team_not_as_asked"] == 1
    assert "the client's team not as asked in 1" in "\n".join(render(tally([mirrored])))
    assert first["team_not_as_asked"] == 0
    own = reading["arms"]["the client's own team"]
    assert (own["games"], own["client_wins"], own["bot_wins"]) == (7, 3, 3)
    assert reading["arms"]["our team on both sides"]["client_wins"] == 2
    # the claim is about the client's own team, over decided games
    assert reading["claim"] == {
        "decided": 6,
        "client_wins": 3,
        "needed": 6,
        "better": False,
    }
    text = "\n".join(render(reading))
    assert "the client won 2, the bot 1, 1 tied" in text
    assert "sheets not as asked in 1" in text
    assert "better needs 6 of 6: not claimed" in text


def test_a_resumed_run_counts_what_is_recorded_and_refuses_another_block_list():
    they, we = parse_blocks("they:closed:4,we:open:3")
    rows = [_row(they, True), _row(they, False), _row(we, True)]
    assert recorded(rows, [they, we]) == {0: 2, 1: 1}
    with pytest.raises(ValueError, match="not this block list"):
        recorded(rows, parse_blocks("they:open:4,we:open:3"))
    with pytest.raises(ValueError, match="not this block list"):
        recorded(rows, parse_blocks("they:closed:4"))


class _Bot:
    """What the bridge touches of the bot: a name, its finished battles, the set the
    real bot keeps of battles with open sheets (and empties as each battle ends)."""

    def __init__(self, results, open_tags=(), hang_after=None):
        self.username = "pfbtest"
        self.battles = {}
        self._open_sheet_battles = set()
        self._accept_open_team_sheet = True
        self.answers = []
        self.results = list(results)
        self.open_tags = set(open_tags)
        self.hang_after = hang_after
        self.ps_client = SimpleNamespace(wait_for_login=self._login, loop=None)

    async def _login(self, wait_for=0):
        return None

    @property
    def n_won_battles(self):
        return sum(b.won is True for b in self.battles.values())

    @property
    def n_lost_battles(self):
        return sum(b.won is False for b in self.battles.values())

    async def accept_challenges(self, opponent, n):
        assert opponent == "vgczero" and n == 1
        if self.hang_after is not None and len(self.battles) >= self.hang_after:
            await asyncio.sleep(3600)
        tag = f"battle-gen9championsvgc2026regmc-{len(self.battles) + 1}"
        self.answers.append(self._accept_open_team_sheet)
        if tag in self.open_tags:
            self._open_sheet_battles.add(tag)
        team = [SimpleNamespace(species=s) for s in ("torkoal", "venusaur")]
        self.battles[tag] = SimpleNamespace(
            finished=True,
            won=self.results.pop(0),
            turn=7,
            teampreview_opponent_team=team,
            teampreview_team=team,
        )
        self._open_sheet_battles.discard(tag)  # as the bot does when a battle ends


def _drive(bot, blocks, out, timeout=5.0):
    state: dict = {}
    asyncio.run(
        _run(bot, blocks, "vgczero", out, 7650, None, "/t/T6e.txt", timeout, 1.0, state)
    )
    return state


def test_games_are_logged_as_they_end_with_the_sheets_they_really_had(tmp_path):
    blocks = parse_blocks("they:closed:2,they:open:2")
    open_tags = {"battle-gen9championsvgc2026regmc-3"}  # the client refused once
    bot = _Bot([True, False, None, False], open_tags)
    assert _drive(bot, blocks, tmp_path) == {}
    rows = read_games(tmp_path)
    assert [row["won"] for row in rows] == [True, False, None, False]
    assert [row["block"] for row in rows] == [0, 0, 1, 1]
    # the bot answered the sheet question as each block says ...
    assert bot.answers == [False, False, True, True]
    # ... and what a game really had is kept though the bot forgets it at the end
    assert [row["sheets_seen"] for row in rows] == [
        "closed",
        "closed",
        "open",
        "closed",
    ]
    assert [row["sheets_as_asked"] for row in rows] == [True, True, True, False]
    assert rows[0]["same_team_seen"] and rows[0]["client"] == "vgczero"
    assert rows[0]["spec"] == "they:closed:2" and rows[0]["turns"] == 7


def test_a_run_that_stops_is_resumed_where_it_stopped(tmp_path):
    blocks = parse_blocks("they:closed:2,they:open:2")
    state = _drive(_Bot([True, True, True], hang_after=3), blocks, tmp_path, 0.3)
    assert "no finished game in 0 s" in state["abort"] and "block 1" in state["abort"]
    assert len(read_games(tmp_path)) == 3
    bot = _Bot([False])
    assert _drive(bot, blocks, tmp_path) == {}
    rows = read_games(tmp_path)
    assert [row["block"] for row in rows] == [0, 0, 1, 1] and bot.answers == [True]
    assert json.loads((tmp_path / "games.jsonl").read_text().splitlines()[-1])[
        "won"
    ] is (False)


def test_open_sheets_are_remembered_past_the_battle():
    seen = _Remembering()
    seen.add("battle-1")
    seen.discard("battle-1")
    assert "battle-1" not in seen and "battle-1" in seen.ever


def test_the_stand_in_client_reads_its_command_line():
    spec = standin_args(
        ["7650", "vgczero", "pfb1", "accept", "5", "open", "--team", ""]
    )
    assert spec == {
        "port": 7650,
        "own": "vgczero",
        "bot": "pfb1",
        "mode": "accept",
        "games": 5,
        "sheets": "open",
        "team": "",
        "player": "deployed",
    }
    for bad in (
        ["7650", "a", "b", "wait", "5", "open"],
        ["7650", "a", "b", "accept", "5", "ajar"],
        ["7650", "a", "b", "accept", "5"],
        ["7650", "a", "b", "accept", "5", "open", "--player", "random"],
    ):
        with pytest.raises(ValueError):
            standin_args(bad)
