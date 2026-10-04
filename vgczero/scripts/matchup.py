#!/usr/bin/env python
"""Matchup lab: how does team A do against team B, and what should each bring and lead?

  python scripts/matchup.py --checkpoint runs/base1/checkpoints/latest.pt --a my_team.txt --b MC2001 --games 400

Teams are pool names (e.g. MC2001) or Showdown paste files (validated and
compiled through Showdown, so PS_DIR / showdown/ps must exist for pastes).
Reports A's win rate with the trained policy piloting both sides, each side's
most frequent bring/lead choices with their win rates, and the team-preview
game solved by search: a mixed strategy over A's best bring/lead options
against B's, with open team sheets (both teams known).
"""

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from vgczero import data as D  # noqa: E402
from vgczero.live.sets import pool  # noqa: E402
from vgczero.matrix_game import solve  # noqa: E402
from vgczero.model import load_model, to_torch  # noqa: E402


def resolve_team(spec: str) -> tuple[int, dict]:
    p = pool()
    path = Path(spec)
    if path.exists():
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / (path.stem + ".txt")).write_text(path.read_text())
            out = Path(td) / "team.json"
            subprocess.run(["node", str(ROOT / "showdown" / "compile_teams.js"), td, "--out", str(out)], check=True,
                           capture_output=True, text=True)
            data = json.loads(out.read_text())
        if not data["teams"]:
            raise SystemExit(f"{spec}: invalid team: {data['rejected']}")
        t = data["teams"][0]
        idx = D.E.register_team(json.dumps({"name": t["name"], "mons": t["mons"]}), allow_unsupported=True)
        reasons = D.E.team_unsupported_reasons(idx)
        if reasons:
            print(f"warning: {spec} uses mechanics the engine does not simulate yet: {reasons[:5]}")
        return idx, t
    i = p.team_index(spec)
    return i, p.team(i)


def preview_name(team: dict, opt: np.ndarray) -> str:
    sp = [m["species"] for m in team["mons"]]
    return f"lead {sp[opt[0]]}+{sp[opt[1]]} / back {sp[opt[2]]}+{sp[opt[3]]}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--k", type=int, default=8, help="bring/lead options per side in the preview game")
    ap.add_argument("--worlds", type=int, default=32, help="dice samples per preview payoff cell")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    D.load()
    model = load_model(args.checkpoint, device=args.device)
    ia, ta = resolve_team(args.a)
    ib, tb = resolve_team(args.b)
    opts = D.preview_options()

    # 1. Self-play games, A on both sides alternately.
    n = min(args.envs, args.games)
    wins, games = 0.0, 0
    prev_stats = [defaultdict(lambda: [0, 0.0]), defaultdict(lambda: [0, 0.0])]  # side A/B: option -> [n, wins for that side]
    for a_side in (0, 1):
        env = D.E.VecEnv(n, seed=args.seed + a_side, team_indices=[ia, ib], open_sheet_prob=1.0)
        env.set_team_weights(a_side, np.array([1.0, 0.0]))
        env.set_team_weights(1 - a_side, np.array([0.0, 1.0]))
        first = np.ones(n, bool)  # first episodes used uniform teams
        chosen = np.full((n, 2), -1)
        target = args.games // 2
        got = 0
        while got < target:
            obs = env.observe()
            o = to_torch(*obs, device=args.device)
            acts, _, _ = model.act(o)
            acts = acts.reshape(n, 2, 3).cpu().numpy()
            req = obs[4][:, :, 0]
            for s in range(2):
                pv = req[:, s] == D.REQ_PREVIEW
                chosen[pv, s] = acts[pv, s, 0]
            _, dones, _ = env.step(acts.astype(np.uint8))
            if dones.any():
                _, winner, _, _ = env.last_infos()
                for e in np.nonzero(dones)[0]:
                    if first[e]:
                        first[e] = False
                        continue
                    if got >= target:
                        break
                    w = int(winner[e])
                    score_a = 1.0 if w == a_side else (0.5 if w == 2 else 0.0)
                    wins += score_a
                    got += 1
                    for who, side in ((0, a_side), (1, 1 - a_side)):
                        c = int(chosen[e, side])
                        if c >= 0:
                            st = prev_stats[who][c]
                            st[0] += 1
                            st[1] += score_a if who == 0 else 1 - score_a
                    chosen[e] = -1
        games += got
    print(f"\n{ta['name']} vs {tb['name']}: A wins {wins / max(1, games):.3f} over {games} games (policy on both sides, open sheets)\n")
    for who, team in ((0, ta), (1, tb)):
        print(f"{'A' if who == 0 else 'B'} ({team['name']}) most played bring/leads:")
        rows = sorted(prev_stats[who].items(), key=lambda kv: -kv[1][0])[:5]
        for c, (k, w) in rows:
            print(f"  {k:4d}x  win {w / k:.2f}  {preview_name(team, opts[c])}")
        print()

    # 2. The team-preview game, solved with both teams known.
    battle = D.E.Battle(ia, ib, seed=args.seed, sheets=(True, True))
    ca, pa = model.top_joint(to_torch(*battle.observe(0), device=args.device), args.k)
    cb, pb = model.top_joint(to_torch(*battle.observe(1), device=args.device), args.k)
    ca, cb = ca[0].cpu().numpy(), cb[0].cpu().numpy()
    copies = D.E.BattleBatch.copies(battle, args.worlds, args.seed)
    children = copies.expand(0, ca.astype(np.uint8), np.repeat(cb[None], args.worlds, 0).astype(np.uint8))
    cobs = to_torch(*children.observe(0), device=args.device)
    H, _ = model.encode(cobs["ints"], cobs["floats"], cobs["field"])
    v = model.value(H).detach().float().cpu().numpy().reshape(args.worlds, len(ca), len(cb)).mean(0)
    x, y, val = solve(v, 4000)
    print(f"Team preview game (value for A {val:+.3f}, +1 = A always wins):")
    print("  A should mix:")
    for i in np.argsort(-x):
        if x[i] > 0.01:
            print(f"    {x[i]:.2f}  {preview_name(ta, opts[ca[i, 0]])}  (prior {pa[0, i]:.2f})")
    print("  B's best response mix:")
    for j in np.argsort(-y):
        if y[j] > 0.01:
            print(f"    {y[j]:.2f}  {preview_name(tb, opts[cb[j, 0]])}")


if __name__ == "__main__":
    main()
