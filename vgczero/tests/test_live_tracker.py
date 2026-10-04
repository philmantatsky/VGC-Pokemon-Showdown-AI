"""Replay a recorded live Showdown battle through the tracker and check that
every request becomes a consistent engine battle."""

import json
from pathlib import Path

import numpy as np

from vgczero import data as D
from vgczero.live.agent import Agent
from vgczero.live.sets import pool
from vgczero.live.tracker import Tracker, parse_hp

LOG = Path(__file__).parent / "fixtures" / "live_battle.log"


def test_replay_recorded_battle():
    p = pool()
    team = p.team(p.team_index("MC1000"))
    tr = Tracker(team)
    agent = Agent(model=None)  # only uses request_masks here
    checked = 0
    for line in LOG.read_text().splitlines():
        tr.feed(line)
        if not line.startswith("|request|") or not line[9:].strip():
            continue
        req = json.loads(line[9:])
        if req.get("wait"):
            continue
        b = D.E.Battle.from_snapshot(tr.snapshot_json(), 0)
        viewer = 0 if tr.me == "p1" else 1
        # Our HP in the engine matches the server's request exactly.
        obs = b.observe(viewer)
        floats = np.asarray(obs[1])[0]
        for i, mon in enumerate(tr.sides[tr.me].mons):
            sp = req["side"]["pokemon"]
            entry = next((x for x in sp if p.base_id(x["details"].split(",")[0]) == mon.base), None)
            if entry is None:
                continue
            hp, mx, _, fnt = parse_hp(entry["condition"])
            want = 0.0 if fnt else hp / mx
            assert abs(floats[i, 0] - want) < 1e-5, (i, floats[i, 0], want)
        if not req.get("teamPreview"):
            rm = agent.request_masks(tr, b)
            assert rm.any(axis=1).all()
        checked += 1
    assert checked >= 5
    assert tr.ended
    assert not tr.log, tr.log[:3]
