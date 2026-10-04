"""Minimal Pokemon Showdown websocket client for vgczero.

Modes
  accept     accept challenges in the format (like mikumiku37)
  ladder     search the ladder repeatedly
  challenge  challenge one user repeatedly (local testing / exhibitions)

Each battle gets a Tracker (protocol -> engine snapshot) and the shared
Agent. Teams rotate among a list of pool teams (e.g. the top-10 from
training). Open team sheets are rejected by default, as mikumiku37 did.

Usage (see scripts/play.py):
  python scripts/play.py --checkpoint runs/base1/checkpoints/latest.pt \
      --username mybot --password ... --mode accept --teams MC1051,MC2001
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import websockets

from .agent import Agent
from .sets import pool, to_id
from .tracker import Tracker

FORMAT = "gen9championsvgc2026regmc"
OFFICIAL_WS = "wss://sim3.psim.us/showdown/websocket"
LOGIN_URL = "https://play.pokemonshowdown.com/api/login"
ASSERT_URL = "https://play.pokemonshowdown.com/api/getassertion"

log = logging.getLogger("vgczero.live")


@dataclass
class BattleSession:
    room: str
    tracker: Tracker
    team_index: int
    rqid_done: int = -1
    acted_turn: int = -1
    pending: dict | None = None
    started: float = field(default_factory=time.time)
    lines: list = field(default_factory=list)


class Client:
    def __init__(self, agent: Agent, username: str, password: str | None = None, server: str = OFFICIAL_WS,
                 mode: str = "accept", teams: list[int] | None = None, fmt: str = FORMAT, opponent: str | None = None,
                 max_battles: int = 0, ots: str = "reject", log_dir: str | None = None, timer: bool = False,
                 max_concurrent: int = 1):
        self.agent = agent
        self.username = username
        self.password = password
        self.server = server
        self.mode = mode
        self.fmt = fmt
        self.opponent = opponent
        self.max_battles = max_battles
        self.ots = ots
        self.timer = timer
        self.max_concurrent = max_concurrent
        self.pool = pool()
        self.teams = teams or [0]
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.battles: dict[str, BattleSession] = {}
        self.finished = 0
        self.results: list[dict] = []
        self.ws = None
        self.logged_in = asyncio.Event()
        self.searching = False
        self.current_team: int | None = None

    # ---- connection -------------------------------------------------------------------------

    async def send(self, msg: str) -> None:
        log.debug(">> %s", msg[:300])
        await self.ws.send(msg)

    async def run(self) -> None:
        async with websockets.connect(self.server, max_size=2**23, ping_interval=None) as ws:
            self.ws = ws
            async for raw in ws:
                await self.handle(raw)
                if self.max_battles and self.finished >= self.max_battles and not self.battles:
                    break

    async def login(self, challstr: str) -> None:
        name = self.username
        if "localhost" in self.server or "127.0.0.1" in self.server:
            await self.send(f"|/trn {name},0,")
            return
        loop = asyncio.get_running_loop()

        def fetch() -> str:
            if self.password:
                data = urllib.parse.urlencode({"name": name, "pass": self.password, "challstr": challstr}).encode()
                with urllib.request.urlopen(LOGIN_URL, data=data, timeout=30) as r:
                    body = r.read().decode()
                return json.loads(body[1:])["assertion"]
            q = urllib.parse.urlencode({"userid": to_id(name), "challstr": challstr})
            with urllib.request.urlopen(f"{ASSERT_URL}?{q}", timeout=30) as r:
                return r.read().decode()

        assertion = await loop.run_in_executor(None, fetch)
        await self.send(f"|/trn {name},0,{assertion}")

    def pick_team(self) -> int:
        self.current_team = random.choice(self.teams)
        return self.current_team

    async def set_team(self) -> None:
        t = self.pool.team(self.pick_team())
        await self.send(f"|/utm {t['packed']}")

    async def start_search(self) -> None:
        if self.mode == "ladder" and not self.searching and len(self.battles) < self.max_concurrent:
            await self.set_team()
            await self.send(f"|/search {self.fmt}")
            self.searching = True
        elif self.mode == "challenge" and self.opponent and len(self.battles) < self.max_concurrent:
            await self.send(f"|/cancelchallenge {self.opponent}")
            await self.set_team()
            await self.send(f"|/challenge {self.opponent}, {self.fmt}")

    # ---- messages -------------------------------------------------------------------------------

    async def handle(self, raw: str) -> None:
        room = ""
        lines = raw.split("\n")
        if lines and lines[0].startswith(">"):
            room = lines[0][1:].strip()
            lines = lines[1:]
        if room.startswith("battle-"):
            await self.handle_battle(room, lines)
            return
        for line in lines:
            if not line.startswith("|"):
                continue
            parts = line.split("|")
            cmd = parts[1] if len(parts) > 1 else ""
            if cmd == "challstr":
                await self.login("|".join(parts[2:]))
            elif cmd == "updateuser":
                if len(parts) > 3 and parts[3] == "1" and to_id(parts[2].strip()) == to_id(self.username):
                    self.logged_in.set()
                    log.info("logged in as %s", self.username)
                    await self.start_search()
            elif cmd == "updatechallenges":
                await self.on_challenges(json.loads("|".join(parts[2:])))
            elif cmd == "popup":
                log.warning("popup: %s", "|".join(parts[2:])[:300])
            elif cmd == "pm" and len(parts) > 4 and parts[4].startswith("/challenge"):
                # |pm| SENDER| RECEIVER|/challenge FORMAT|FORMAT|...
                sender, receiver = to_id(parts[2]), to_id(parts[3])
                fmt = parts[4].split(" ", 1)[1].strip() if " " in parts[4] else ""
                if receiver == to_id(self.username) and sender != receiver and fmt:
                    await self.on_challenges({"challengesFrom": {sender: fmt}})
            else:
                log.debug("<< %s", line[:200])

    async def on_challenges(self, data: dict) -> None:
        if self.mode != "accept":
            return
        for user, fmt in (data.get("challengesFrom") or {}).items():
            if fmt == self.fmt and len(self.battles) < self.max_concurrent:
                await self.set_team()
                await self.send(f"|/accept {user}")
            elif fmt != self.fmt:
                await self.send(f"|/reject {user}")

    async def handle_battle(self, room: str, lines: list[str]) -> None:
        sess = self.battles.get(room)
        if sess is None:
            if not any(l.startswith("|init|battle") for l in lines):
                return
            self.searching = False
            team = self.current_team if self.current_team is not None else self.pick_team()
            sess = BattleSession(room, Tracker(self.pool.team(team)), team)
            self.battles[room] = sess
            if self.timer:
                await self.send(f"{room}|/timer on")
        act_now = False
        for line in lines:
            if not line.startswith("|"):
                continue
            sess.lines.append(line)
            if line.startswith("|request|"):
                body = line[len("|request|"):]
                if body.strip():
                    req = json.loads(body)
                    sess.tracker.feed(line)
                    sess.pending = req
                    if req.get("teamPreview") or req.get("forceSwitch"):
                        act_now = True
                    elif req.get("active") and sess.tracker.turn > sess.acted_turn:
                        # The turn's log already arrived (this server sends the
                        # request after it); otherwise wait for |turn|.
                        act_now = True
                continue
            if line.startswith("|uhtml|otsrequest"):
                cmd = "/acceptopenteamsheets" if self.ots == "accept" else "/rejectopenteamsheets"
                await self.send(f"{room}|{cmd}")
                continue
            if line.startswith("|error|"):
                log.warning("%s %s", room, line)
                if "[Invalid choice]" in line or "[Unavailable choice]" in line:
                    sess.tracker.log.append(line)
                    await self.send(f"{room}|/choose default|{sess.rqid_done}")
                continue
            sess.tracker.feed(line)
            if line.startswith("|turn|") or line.startswith("|teampreview"):
                act_now = True
            if line.startswith("|win|") or line == "|tie" or line.startswith("|tie|"):
                await self.finish(sess)
                return
        if sess.pending is not None and (act_now or sess.tracker.turn > 0 and sess.pending.get("forceSwitch")):
            await self.act(sess)

    async def act(self, sess: BattleSession) -> None:
        req = sess.pending
        if req is None or req.get("wait"):
            return
        rqid = req.get("rqid", 0)
        if rqid <= sess.rqid_done:
            return
        sess.rqid_done = rqid
        sess.pending = None
        if req.get("active"):
            sess.acted_turn = sess.tracker.turn
        choice = self.agent.decide_safe(sess.tracker)
        await self.send(f"{sess.room}|/choose {choice}|{rqid}")

    async def finish(self, sess: BattleSession) -> None:
        tr = sess.tracker
        me_name = tr.sides[tr.me].name if tr.me else self.username
        won = tr.winner is not None and to_id(tr.winner) == to_id(me_name)
        res = {"room": sess.room, "won": won, "winner": tr.winner, "turns": tr.turn, "team": self.pool.team(sess.team_index)["name"],
               "opponent": tr.sides[tr.opp].name if tr.me else "", "disagreements": len(self.agent.disagreements),
               "tracker_errors": tr.log[-5:]}
        self.results.append(res)
        log.info("%s %s in %d turns vs %s (team %s)", sess.room, "WON" if won else "lost", tr.turn, res["opponent"], res["team"])
        if self.log_dir:
            (self.log_dir / f"{sess.room}.log").write_text("\n".join(sess.lines))
            with open(self.log_dir / "results.jsonl", "a") as f:
                f.write(json.dumps(res) + "\n")
        self.battles.pop(sess.room, None)
        self.finished += 1
        await self.send(f"|/leave {sess.room}")
        if not self.max_battles or self.finished < self.max_battles:
            await self.start_search()
