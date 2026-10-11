"""Serial games on a LOCAL Showdown server between the deployed bot and an external
client (2026-10-11; the vgczero plan's bridge to the real game: at least 300 games in
four blocks, who challenges x sheets open or closed, and an arm on our own team).

Our side is ``ladder_ourteam.main()`` itself -- the script the ladder launcher runs,
with the argument list it would pass (``tools/local_bridge.sh`` builds it from the
deployed configuration) -- pointed at the local server. Nothing touches the real
server or the real account: the bot's name is a throwaway guest and no password is set
or read.

The external client is any program that speaks Showdown's protocol. All it is asked
to do: log into ``ws://localhost:<port>/showdown/websocket`` under the name given here,
play ``gen9championsvgc2026regmc``, and either challenge the bot or accept the bot's
challenges, answering the open-team-sheet question as the block says. This script
never imports it. ``checks/local_bridge_standin.py`` is such a client (the deployed
bot without search, or a heuristic player) for rehearsing the harness.

A block is ``<who>:<sheets>:<games>[:same]``, blocks are joined by commas:
  who     ``they`` the client challenges the bot; ``we`` the bot challenges the client
  sheets  ``open`` the bot accepts open team sheets (they are open only if the client
          accepts too); ``closed`` the bot rejects them (closed whatever the client
          does). What a game really was is read from the battle and logged.
  games   how many, one after another
  same    the arm on our own team: the client is asked to play the bot's team file
          (a game whose client showed other species, or ours in another block, is
          counted as "team not as asked")

With ``--client`` the client is started by this script once a block, from a command
template with these fields: ``{port} {opponent} {bot} {mode} {games} {sheets} {team}
{format} {block}`` -- ``{mode}`` is ``challenge`` in a ``they`` block and ``accept``
in a ``we`` block, ``{team}`` the bot's team file in a ``same`` block and empty
otherwise. Without it the block is announced (``BRIDGE_BLOCK``) and the client is
expected to be there: start it by hand with the same block list.

Usage (repo root; a local server on <port>):
  .venv/bin/python evaluation/local_bridge.py <port> <opponent account> <blocks> \
      <out dir> [--client "<template>"] [--bot-name NAME] [--game-timeout 900] \
      [--retry 20] [--resume] -- <ladder_ourteam.py arguments>
  .venv/bin/python evaluation/local_bridge.py --tally <out dir>
Written to <out dir>: ``games.jsonl`` (a row a game, as it ends), ``tally.txt`` and
``tally.json``, ``client_<block>.log``, ``replays/`` (the ladder script's replay
directory: its run configuration, replays and decision logs).
Exit status: 0 all blocks played; 2 refused or stopped (``BRIDGE_ABORT`` says why).

Limits. Games are serial and untimed (no battle timer in challenges): a client that
stops answering ends the run at ``--game-timeout``. A ``we`` challenge the client has
not taken after ``--retry`` seconds is withdrawn and sent again, so the client may be
started late. The claim line follows the plan's rule: the client is "better" when its
wins have a one-sided chance of at most 2.5% under even odds (168 of 300).
"""

from __future__ import annotations

import asyncio
import collections
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.search_ladder_read import wilson  # noqa: E402

FORMAT = "gen9championsvgc2026regmc"
WHO = {"they": "the client challenges", "we": "the bot challenges"}
CLAIM_P = 0.025


@dataclass(frozen=True)
class Block:
    index: int
    who: str  # "they" or "we": who sends the challenge
    sheets: str  # "open" or "closed": what the bot answers
    games: int
    same: bool = False  # the client is asked to play the bot's own team

    @property
    def label(self) -> str:
        arm = ", our team on both sides" if self.same else ""
        return f"{WHO[self.who]}, sheets {self.sheets}{arm}"

    @property
    def spec(self) -> str:
        return f"{self.who}:{self.sheets}:{self.games}" + (":same" if self.same else "")


def parse_blocks(text: str) -> list[Block]:
    blocks = []
    for index, part in enumerate(p for p in text.split(",") if p.strip()):
        fields = part.strip().split(":")
        same = len(fields) == 4 and fields[3] == "same"
        if (
            len(fields) not in (3, 4)
            or (len(fields) == 4 and not same)
            or fields[0] not in WHO
            or fields[1] not in ("open", "closed")
            or not fields[2].isdigit()
            or int(fields[2]) < 1
        ):
            raise ValueError(
                f"block {part!r} is not <they|we>:<open|closed>:<games>[:same]"
            )
        blocks.append(Block(index, fields[0], fields[1], int(fields[2]), same))
    if not blocks:
        raise ValueError("no blocks given")
    return blocks


def client_command(
    template: str, block: Block, port: int, opponent: str, bot: str, team: str
) -> str:
    """The client's command line for one block."""
    return template.format(
        port=port,
        opponent=opponent,
        bot=bot,
        mode="challenge" if block.who == "they" else "accept",
        games=block.games,
        sheets=block.sheets,
        team=team if block.same else "",
        format=FORMAT,
        block=block.index,
    )


def claim_line(n: int) -> int | None:
    """The fewest wins of ``n`` decided games whose one-sided chance under even odds
    is at most ``CLAIM_P`` (168 of 300); None when no count of ``n`` reaches it."""
    tail = 0.0
    for wins in range(n, -1, -1):
        tail += math.comb(n, wins) / 2**n
        if tail > CLAIM_P:
            return wins + 1 if wins < n else None
    return 0


def _cell(rows: list[dict[str, Any]]) -> dict[str, Any]:
    games = len(rows)
    bot = sum(row["won"] is True for row in rows)
    client = sum(row["won"] is False for row in rows)
    decided = bot + client
    low, high = wilson(client, decided)
    return {
        "games": games,
        "bot_wins": bot,
        "client_wins": client,
        "ties": games - decided,
        "client_share": client / decided if decided else None,
        "client_interval": [low, high] if decided else None,
        "sheets_not_as_asked": sum(not row["sheets_as_asked"] for row in rows),
        # the client on our six in a block of its own team, or the other way round
        "team_not_as_asked": sum(
            row.get("same_team_seen", row["same"]) != row["same"] for row in rows
        ),
        "turns_mean": (sum(row["turns"] for row in rows) / games if games else None),
    }


def tally(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The games by block (who challenges x sheets, the two arms apart), each arm's
    total, and whether the client's wins on its own team claim "better"."""
    by_block: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_block[row["label"]].append(row)
    arms = {
        "the client's own team": [row for row in rows if not row["same"]],
        "our team on both sides": [row for row in rows if row["same"]],
    }
    out: dict[str, Any] = {
        "games": len(rows),
        "blocks": {label: _cell(cell) for label, cell in by_block.items()},
        "arms": {name: _cell(cell) for name, cell in arms.items() if cell},
    }
    main = out["arms"].get("the client's own team")
    if main:
        decided = main["bot_wins"] + main["client_wins"]
        needed = claim_line(decided)
        out["claim"] = {
            "decided": decided,
            "client_wins": main["client_wins"],
            "needed": needed,
            "better": needed is not None and main["client_wins"] >= needed,
        }
    return out


def render(reading: dict[str, Any]) -> list[str]:
    def line(label: str, cell: dict[str, Any]) -> str:
        text = (
            f"{label}: {cell['games']} games, the client won {cell['client_wins']}, "
            f"the bot {cell['bot_wins']}"
        )
        if cell["ties"]:
            text += f", {cell['ties']} tied"
        if cell["client_interval"]:
            low, high = cell["client_interval"]
            text += (
                f"; the client's share {100 * cell['client_share']:.0f}% "
                f"[{100 * low:.0f}, {100 * high:.0f}]"
            )
        if cell["sheets_not_as_asked"]:
            text += f"; sheets not as asked in {cell['sheets_not_as_asked']}"
        if cell["team_not_as_asked"]:
            text += f"; the client's team not as asked in {cell['team_not_as_asked']}"
        return text

    out = [f"{reading['games']} games"]
    out += [line(f"  {label}", cell) for label, cell in reading["blocks"].items()]
    out += [line(f"ARM {name}", cell) for name, cell in reading["arms"].items()]
    claim = reading.get("claim")
    if claim:
        if claim["needed"] is None:
            verdict = "too few games for any count to claim better"
        else:
            verdict = f"better needs {claim['needed']} of {claim['decided']}: " + (
                "CLAIMED" if claim["better"] else "not claimed"
            )
        out.append(
            f"CLAIM (the client's own team): {claim['client_wins']} of "
            f"{claim['decided']} decided games; {verdict}"
        )
    return out


def read_games(out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / "games.jsonl"
    rows = []
    if path.exists():
        for text in path.read_text().splitlines():
            try:
                rows.append(json.loads(text))
            except ValueError:
                continue  # a row being written
    return rows


def write_tally(out_dir: Path) -> list[str]:
    reading = tally(read_games(out_dir))
    lines = render(reading)
    (out_dir / "tally.json").write_text(json.dumps(reading, indent=1) + "\n")
    (out_dir / "tally.txt").write_text("\n".join(lines) + "\n")
    return lines


def _flag(args: list[str], name: str, default: str | None = None) -> str | None:
    if name in args:
        at = args.index(name)
        value = args[at + 1]
        del args[at : at + 2]
        return value
    return default


class _Remembering(set):
    """The bot forgets a battle's open sheets when the battle ends; this keeps them."""

    def __init__(self) -> None:
        super().__init__()
        self.ever: set[str] = set()

    def add(self, item: str) -> None:
        self.ever.add(item)
        super().add(item)


def game_row(
    block: Block, tag: str, battle: Any, sheets_seen: str, opponent: str, seconds: float
) -> dict[str, Any]:
    """One finished game as a row of ``games.jsonl``."""
    theirs = sorted(str(mon.species) for mon in battle.teampreview_opponent_team)
    ours = sorted(str(mon.species) for mon in battle.teampreview_team)
    return {
        "block": block.index,
        "spec": block.spec,
        "label": block.label,
        "who": block.who,
        "sheets": block.sheets,
        "same": block.same,
        "tag": tag,
        "won": battle.won,
        "turns": int(battle.turn),
        "sheets_seen": sheets_seen,
        "sheets_as_asked": sheets_seen == block.sheets,
        "same_team_seen": theirs == ours,
        "client": opponent,
        "client_species": theirs,
        "seconds": round(seconds, 1),
        "ended": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def recorded(rows: list[dict[str, Any]], blocks: list[Block]) -> dict[int, int]:
    """Games already played, by block; refuses rows of another block list."""
    done: collections.Counter[int] = collections.Counter()
    for row in rows:
        index = row["block"]
        if index >= len(blocks) or blocks[index].spec != row["spec"]:
            raise ValueError(
                f"games.jsonl holds block {index} as {row['spec']!r}: not this "
                "block list"
            )
        done[index] += 1
    return dict(done)


async def _challenge(agent: Any, opponent: str, retry_s: float) -> None:
    """One game the bot asks for (in poke-env's own loop): the challenge is sent
    again while the client has not taken it, then the game is awaited."""
    await agent.ps_client.logged_in.wait()
    known = len(agent.battles)
    while True:
        await agent.ps_client.challenge(opponent, agent._format, agent.next_team)
        try:
            await asyncio.wait_for(agent._battle_semaphore.acquire(), retry_s)
            break
        except asyncio.TimeoutError:
            if len(agent.battles) > known:  # taken as the wait ran out
                await agent._battle_semaphore.acquire()
                break
            await agent.ps_client.send_message(f"/cancelchallenge {opponent}")
    await agent._battle_count_queue.join()


async def _play(agent: Any, play: Any, client: Any, game_timeout: float) -> str | None:
    """Await one game. Returns why it did not end: the time ran out, or the client
    this script started is gone with no game in progress."""
    task = asyncio.ensure_future(play)
    started = time.monotonic()
    gone_since = None
    while not task.done():
        await asyncio.wait({task}, timeout=2.0)
        if task.done():
            break
        if time.monotonic() - started > game_timeout:
            task.cancel()
            return f"no finished game in {game_timeout:.0f} s"
        idle = all(battle.finished for battle in agent.battles.values())
        if client is not None and client.returncode is not None and idle:
            gone_since = gone_since or time.monotonic()
            if time.monotonic() - gone_since > 15:
                task.cancel()
                return f"the client ended (status {client.returncode}) with games left"
        else:
            gone_since = None
    task.result()
    return None


async def _run(
    agent: Any,
    blocks: list[Block],
    opponent: str,
    out_dir: Path,
    port: int,
    template: str | None,
    team: str,
    game_timeout: float,
    retry_s: float,
    state: dict[str, Any],
) -> None:
    from poke_env.concurrency import handle_threaded_coroutines

    await agent.ps_client.wait_for_login(wait_for=60)
    print(f"BRIDGE_READY bot={agent.username} format={FORMAT} port={port}", flush=True)
    done = recorded(read_games(out_dir), blocks)
    opened = _Remembering()
    agent._open_sheet_battles = opened
    log = (out_dir / "games.jsonl").open("a")
    for block in blocks:
        left = block.games - done.get(block.index, 0)
        if left <= 0:
            continue
        mode = "challenge" if block.who == "they" else "accept"
        print(
            f"BRIDGE_BLOCK {block.index} {block.spec} games={left} "
            f"client={opponent} mode={mode} sheets={block.sheets} "
            f"team={team if block.same else '-'}",
            flush=True,
        )
        client = None
        if template:
            rest = Block(block.index, block.who, block.sheets, left, block.same)
            command = client_command(
                template, rest, port, opponent, agent.username, team
            )
            sink = (out_dir / f"client_{block.index}.log").open("a")
            sink.write(f"$ {command}\n")
            sink.flush()
            client = await asyncio.create_subprocess_shell(
                command, stdout=sink, stderr=asyncio.subprocess.STDOUT
            )
        # poke-env reads this when a battle starts: the bot's answer to the question
        agent._accept_open_team_sheet = block.sheets == "open"
        for _ in range(left):
            before = set(agent.battles)
            started = time.monotonic()
            play = (
                agent.accept_challenges(opponent, 1)
                if block.who == "they"
                else handle_threaded_coroutines(
                    _challenge(agent, opponent, retry_s), agent.ps_client.loop
                )
            )
            failed = await _play(agent, play, client, game_timeout)
            tags = [tag for tag in agent.battles if tag not in before]
            if failed is None and (
                len(tags) != 1 or not agent.battles[tags[0]].finished
            ):
                failed = f"one play left these games: {tags}"
            if failed is not None:
                state["abort"] = f"block {block.index} ({block.spec}): {failed}"
                break
            battle = agent.battles[tags[0]]
            seen = "open" if tags[0] in opened.ever else "closed"
            row = game_row(
                block, tags[0], battle, seen, opponent, time.monotonic() - started
            )
            log.write(json.dumps(row) + "\n")
            log.flush()
            winner = "bot" if battle.won else "tie" if battle.won is None else "client"
            print(
                f"BRIDGE_GAME block={block.index} {winner} turns={row['turns']} "
                f"sheets={seen} seconds={row['seconds']:.0f} "
                f"bot_record={agent.n_won_battles}-{agent.n_lost_battles}",
                flush=True,
            )
        if client is not None:
            try:
                await asyncio.wait_for(client.wait(), 30)
            except asyncio.TimeoutError:
                client.terminate()
        if state.get("abort"):
            break
    log.close()


async def drive(args: list[str]) -> int:
    split = args.index("--")
    own, ladder_args = args[:split], args[split + 1 :]
    template = _flag(own, "--client")
    name = _flag(own, "--bot-name") or "pfb" + uuid4().hex[:9]
    game_timeout = float(_flag(own, "--game-timeout", "900") or 900)
    retry_s = float(_flag(own, "--retry", "20") or 20)
    resume = "--resume" in own
    own = [arg for arg in own if arg != "--resume"]
    if len(own) != 4:
        print(__doc__)
        return 2
    port, opponent, out_dir = int(own[0]), own[1], Path(own[3])
    try:
        blocks = parse_blocks(own[2])
    except ValueError as error:
        print(f"BRIDGE_ABORT {error}")
        return 2
    if (out_dir / "games.jsonl").exists() and not resume:
        print(f"BRIDGE_ABORT {out_dir}/games.jsonl exists: --resume, or a new out dir")
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        recorded(read_games(out_dir), blocks)
    except ValueError as error:
        print(f"BRIDGE_ABORT {error}")
        return 2
    team = ""
    if "--our_team" in ladder_args:
        team = str((ROOT / ladder_args[ladder_args.index("--our_team") + 1]).resolve())

    # a throwaway local guest; the real credentials are never loaded into this process
    os.environ["SHOWDOWN_USERNAME"] = name
    os.environ.pop("SHOWDOWN_PASSWORD", None)

    from poke_env.ps_client import ServerConfiguration

    import ladder_ourteam

    ladder_ourteam.ShowdownServerConfiguration = ServerConfiguration(
        f"ws://localhost:{port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )  # the one substitution
    state: dict[str, Any] = {}
    original = ladder_ourteam.play_until_disconnect

    async def bridged(agent, awaitable, *more, **named):
        awaitable.close()  # the ladder script's own "accept anybody's challenge"
        schedule = _run(
            agent,
            blocks,
            opponent,
            out_dir,
            port,
            template,
            team,
            game_timeout,
            retry_s,
            state,
        )
        return await original(agent, schedule, *more, **named)

    ladder_ourteam.play_until_disconnect = bridged
    total = sum(block.games for block in blocks)
    sys.argv = [
        "ladder_ourteam.py",
        *ladder_args,
        "--challenges",
        "--n_games",
        str(total),
        "--replay_dir",
        str(out_dir / "replays"),
    ]
    await ladder_ourteam.main()
    print("\n".join(write_tally(out_dir)), flush=True)
    if state.get("abort"):
        print(f"BRIDGE_ABORT {state['abort']}", flush=True)
        return 2
    print("BRIDGE_DONE", flush=True)
    return 0


def main() -> int:
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == "--tally":
        print("\n".join(write_tally(Path(args[1]))))
        return 0
    if "--" not in args:
        print(__doc__)
        return 2
    return asyncio.run(drive(args))


if __name__ == "__main__":
    sys.exit(main())
