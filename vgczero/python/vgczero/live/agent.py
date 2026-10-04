"""Decide a live Showdown choice from the tracked battle.

The tracker's snapshot becomes an engine battle; the policy (or search) picks
an action in the engine's encoding, restricted to what the server's request
allows (the request is ground truth -- any disagreement with the engine's own
legality is logged, which doubles as a live parity check), and the action is
turned into a Showdown choice string.
"""

from __future__ import annotations

import json
import random

import numpy as np
import torch

from .. import data as D
from ..model import VGCNet, to_torch
from ..search import SearchConfig, search
from .sets import to_id
from .tracker import Tracker, details_species

T_FOE0, T_FOE1, T_ALLY = 0, 1, 2


def decode(a: int):
    if a == 0:
        return ("pass",)
    if 1 <= a <= 6:
        return ("switch", a - 1)
    k = a - 7
    mega = k % 2 == 1
    mt = k // 2
    return ("move", mt // 3, mt % 3, mega)


def encode_move(slot: int, target: int, mega: bool) -> int:
    return 7 + (slot * 3 + target) * 2 + int(mega)


class Agent:
    def __init__(self, model: VGCNet, search_cfg: SearchConfig | None = None, device="cpu", seed: int = 0, deterministic: bool = False):
        self.model = model
        self.search_cfg = search_cfg
        self.device = device
        self.rng = random.Random(seed)
        self.deterministic = deterministic
        self.disagreements: list[str] = []

    # ---- request-derived legality ------------------------------------------------------

    def _team_index_of(self, tr: Tracker, p: dict) -> int | None:
        base = tr.pool.base_id(details_species(p.get("details", "")))
        for i, m in enumerate(tr.sides[tr.me].mons):
            if m.base == base:
                return i
        return None

    def request_masks(self, tr: Tracker, battle) -> np.ndarray:
        """[2, 31] bool from the request alone (targets from the engine)."""
        req = tr.request or {}
        side = req.get("side", {})
        pokemon = side.get("pokemon", [])
        mask = np.zeros((2, 31), bool)
        bench = []
        for p in pokemon:
            if p.get("active"):
                continue
            if p.get("condition", "").endswith("fnt"):
                continue
            ti = self._team_index_of(tr, p)
            if ti is not None:
                bench.append(ti)
        engine = np.asarray(battle.legal_masks(1 if tr.me == "p2" else 0), dtype=bool)
        if req.get("forceSwitch"):
            for k, need in enumerate(req["forceSwitch"][:2]):
                if need and bench:
                    for ti in bench:
                        mask[k, 1 + ti] = True
                else:
                    mask[k, 0] = True
            return mask
        actives = [p for p in pokemon if p.get("active")]
        for k, act in enumerate((req.get("active") or [])[:2]):
            if k >= len(actives) or actives[k].get("condition", "").endswith("fnt"):
                mask[k, 0] = True
                continue
            moves = act.get("moves", [])
            mon_moves = None
            ti = self._team_index_of(tr, actives[k])
            if ti is not None:
                mon_moves = tr.sides[tr.me].mons[ti].set["moves"]
            can_mega = bool(act.get("canMegaEvo"))
            for mv in moves:
                if mv.get("disabled"):
                    continue
                mid = mv.get("id") or to_id(mv.get("move", ""))
                if mid == "struggle":
                    slots = [0]
                elif mon_moves is not None and mid in mon_moves:
                    slots = [mon_moves.index(mid)]
                else:
                    continue
                s = slots[0]
                tgt_ok = [bool(engine[k, encode_move(s, t, False)] or engine[k, encode_move(s, t, True)]) for t in range(3)]
                if not any(tgt_ok):
                    tgt_ok = [True, False, False]
                for t in range(3):
                    if tgt_ok[t]:
                        mask[k, encode_move(s, t, False)] = True
                        if can_mega:
                            mask[k, encode_move(s, t, True)] = True
            if not (act.get("trapped") or act.get("maybeTrapped")):
                for ti in bench:
                    mask[k, 1 + ti] = True
            if not mask[k].any():
                mask[k, 0] = True
        return mask

    # ---- choice strings ----------------------------------------------------------------

    def choice_string(self, tr: Tracker, acts: tuple[int, int, int]) -> str:
        req = tr.request or {}
        if req.get("teamPreview"):
            row = D.preview_options()[acts[0]]
            return "team " + "".join(str(int(i) + 1) for i in row)
        pokemon = req.get("side", {}).get("pokemon", [])

        def switch_pos(ti: int) -> int:
            for j, p in enumerate(pokemon):
                if self._team_index_of(tr, p) == ti:
                    return j + 1
            return 0

        parts = []
        n_slots = len(req.get("forceSwitch") or req.get("active") or [])
        for k in range(min(2, n_slots)):
            d = decode(int(acts[1 + k]))
            if d[0] == "pass":
                parts.append("pass")
            elif d[0] == "switch":
                parts.append(f"switch {switch_pos(d[1])}")
            else:
                _, ms, t, mega = d
                active = (req.get("active") or [{}])[k]
                moves = active.get("moves", [])
                actives = [p for p in pokemon if p.get("active")]
                ti = self._team_index_of(tr, actives[k]) if k < len(actives) else None
                # Request move order matches the set's move order except for Struggle.
                if len(moves) == 1 and (moves[0].get("id") == "struggle"):
                    num = 1
                    target_type = "randomNormal"
                else:
                    mid = tr.sides[tr.me].mons[ti].set["moves"][ms] if ti is not None else ""
                    ids = [m.get("id") for m in moves]
                    num = ids.index(mid) + 1 if mid in ids else ms + 1
                    target_type = moves[num - 1].get("target", "normal") if num - 1 < len(moves) else "normal"
                s = f"move {num}"
                if target_type in ("normal", "any", "adjacentFoe", "adjacentAlly", "adjacentAllyOrSelf"):
                    if t == T_ALLY or target_type == "adjacentAlly":
                        s += f" -{2 - k}"
                    elif target_type == "adjacentAllyOrSelf":
                        s += f" -{k + 1}"
                    else:
                        s += f" {t + 1}"
                if mega:
                    s += " mega"
                parts.append(s)
        return ", ".join(parts) if parts else "default"

    # ---- deciding ------------------------------------------------------------------------

    @torch.no_grad()
    def decide(self, tr: Tracker) -> str:
        snap = tr.snapshot_json()
        battle = D.E.Battle.from_snapshot(snap, self.rng.getrandbits(48))
        viewer = 0 if tr.me == "p1" else 1
        req = tr.request or {}
        if req.get("teamPreview"):
            if self.search_cfg is not None and self.search_cfg.search_preview:
                r = search(self.model, battle, viewer, self.search_cfg, seed=self.rng.getrandbits(32), device=self.device)
                return self.choice_string(tr, r.choice)
            o = to_torch(*battle.observe(viewer), device=self.device)
            a, _, _ = self.model.act(o, deterministic=self.deterministic)
            return self.choice_string(tr, tuple(int(x) for x in a[0]))
        rmask = self.request_masks(tr, battle)
        emask = np.asarray(battle.legal_masks(viewer), dtype=bool)
        for k in range(2):
            if (rmask[k] != emask[k]).any():
                self.disagreements.append(
                    f"turn {tr.turn} slot {k}: request-only {np.nonzero(rmask[k] & ~emask[k])[0].tolist()} engine-only {np.nonzero(emask[k] & ~rmask[k])[0].tolist()}"
                )
        choice = None
        if self.search_cfg is not None and not req.get("forceSwitch"):
            r = search(self.model, battle, viewer, self.search_cfg, seed=self.rng.getrandbits(32), device=self.device)
            c = r.choice
            if rmask[0, c[1]] and rmask[1, c[2]]:
                choice = c
        if choice is None:
            o = to_torch(*battle.observe(viewer), device=self.device)
            o["mask"] = torch.as_tensor(rmask[None], device=self.device)
            a, _, _ = self.model.act(o, deterministic=self.deterministic)
            choice = tuple(int(x) for x in a[0])
        return self.choice_string(tr, choice)

    def decide_safe(self, tr: Tracker) -> str:
        try:
            return self.decide(tr)
        except Exception as e:  # never forfeit on a bug: let the server pick
            self.disagreements.append(f"decide error turn {tr.turn}: {e!r}")
            return "default"

    def log_state(self, tr: Tracker) -> str:
        return json.dumps(tr.snapshot())[:2000]
