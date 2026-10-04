"""PPO self-play against a league of past versions, terminal rewards only.

Each env has two players. Side 0 is always the learner. Side 1 is either the
current policy (self-play; both sides then produce training data) or a past
snapshot from the league, chosen per episode with PFSP weights. Rewards are
+1/-1 at the end of the game and nothing else.

A player only acts at its own decision points (it waits while the opponent
picks a mid-turn replacement, for example), so advantages are computed over
each player's own sequence of decisions with GAE, bootstrapping from the value
of the next decision state when a rollout ends mid-game.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .league import SELF, League
from .model import ModelConfig, VGCNet, load_model, save, to_torch
from .teams import TeamStats


@dataclass
class TrainConfig:
    run_dir: str = "runs/default"
    model: str = "base"
    device: str = "auto"
    seed: int = 0
    n_envs: int = 512
    rollout_steps: int = 32
    gamma: float = 1.0
    gae_lambda: float = 0.95
    lr: float = 3e-4
    lr_final: float = 3e-5
    warmup_updates: int = 20
    total_updates: int = 200_000
    epochs: int = 2
    minibatch: int = 4096
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 1.0
    target_kl: float = 0.03
    # League.
    self_play_frac: float = 0.5
    league_every: int = 50
    league_max: int = 64
    league_active: int = 4
    league_refresh: int = 10
    pfsp_power: float = 2.0
    # Teams.
    team_focus: float = 0.0
    team_update_every: int = 20
    open_sheet_prob: float = 0.5
    turn_limit: int = 100
    teams: str = "supported"
    # Team evolution during training (0 = off): every N updates, evolve the
    # best teams and add survivors to the training pool.
    evolve_every: int = 0
    evolve_population: int = 8
    evolve_children: int = 2
    evolve_games: int = 200
    evolve_field: int = 64
    # Periodic evaluation against the greedy baseline (0 = off).
    eval_every: int = 0
    eval_games: int = 400
    # Bookkeeping.
    save_every: int = 25
    log_every: int = 1
    bf16: bool = True


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Rollout:
    """Storage for one rollout: [T, N, 2] players."""

    def __init__(self, T: int, N: int):
        n_mons, mon_int, mon_float, field_float, _, n_act, _ = D.E.dims()
        self.T, self.N = T, N
        self.ints = np.zeros((T, N, 2, n_mons, mon_int), np.int32)
        self.floats = np.zeros((T, N, 2, n_mons, mon_float), np.float32)
        self.field = np.zeros((T, N, 2, field_float), np.float32)
        self.mask = np.zeros((T, N, 2, 2, n_act), np.uint8)
        self.req = np.zeros((T, N, 2, 3), np.int32)
        self.acts = np.zeros((T, N, 2, 3), np.int64)
        self.logp = np.zeros((T, N, 2), np.float32)
        self.value = np.zeros((T, N, 2), np.float32)
        self.valid = np.zeros((T, N, 2), bool)
        self.reward = np.zeros((T, N, 2), np.float32)
        self.done = np.zeros((T, N), bool)


def compute_gae(ro: Rollout, last_value: np.ndarray, gamma: float, lam: float):
    """Advantages/returns over each player's own decision sequence.

    last_value [N,2]: value of the state after the rollout (bootstrap).
    Rewards arrive at env steps (terminal only); they are credited to the
    player's most recent decision before them.
    """
    T, N = ro.T, ro.N
    adv = np.zeros((T, N, 2), np.float32)
    next_v = last_value.astype(np.float32).copy()
    next_adv = np.zeros((N, 2), np.float32)
    pend_r = np.zeros((N, 2), np.float32)
    pend_done = np.zeros((N, 2), bool)
    for t in range(T - 1, -1, -1):
        d = ro.done[t]
        if d.any():
            # The episode ended after this step's actions.
            next_v[d] = 0.0
            next_adv[d] = 0.0
            pend_r[d] = ro.reward[t][d]
            pend_done[d] = True
        v = ro.value[t]
        valid = ro.valid[t]
        nonterm = (~pend_done).astype(np.float32)
        delta = pend_r + gamma * next_v * nonterm - v
        a = delta + gamma * lam * nonterm * next_adv
        adv[t] = np.where(valid, a, 0.0)
        next_v = np.where(valid, v, next_v)
        next_adv = np.where(valid, a, next_adv)
        pend_r = np.where(valid, 0.0, pend_r)
        pend_done = np.where(valid, False, pend_done)
    ret = adv + ro.value
    return adv, ret


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        D.load()
        self.dir = Path(cfg.run_dir)
        (self.dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.dir / "cfg.json").write_text(json.dumps(asdict(cfg), indent=1))
        self.device = pick_device(cfg.device)
        torch.manual_seed(cfg.seed)
        self.rng = random.Random(cfg.seed)
        self.np_rng = np.random.default_rng(cfg.seed)
        # Teams.
        if cfg.teams == "supported":
            pool = D.supported_teams()
        else:
            names = D.team_names()
            want = [x.strip() for x in Path(cfg.teams).read_text().split() if x.strip()]
            pool = [names.index(w) for w in want]
        # Evolved teams from earlier sessions are re-registered in the same
        # order, so their engine indices (used by TeamStats) are stable.
        self.evolved_path = self.dir / "evolved_teams.json"
        self.evolved: list[dict] = json.loads(self.evolved_path.read_text()) if self.evolved_path.exists() else []
        for t in self.evolved:
            idx = D.E.register_team(json.dumps({"name": t["name"], "mons": t["mons"]}))
            if idx not in pool:
                pool.append(idx)
        self.pool = pool
        self.env = D.E.VecEnv(cfg.n_envs, seed=cfg.seed, team_indices=pool, turn_limit=cfg.turn_limit, open_sheet_prob=cfg.open_sheet_prob)
        self.team_stats = TeamStats(list(pool))
        self.evolver = None
        # Model.
        latest = self.dir / "checkpoints" / "latest.pt"
        self.update = 0
        if latest.exists():
            ck = torch.load(latest, map_location=self.device, weights_only=False)
            self.model = VGCNet(ModelConfig(**ck["config"])).to(self.device)
            self.model.load_state_dict(ck["state"])
            self.update = ck["extra"].get("update", 0)
            ts = self.dir / "team_stats.json"
            if ts.exists():
                self.team_stats.load_state(json.loads(ts.read_text()))
        else:
            self.model = VGCNet(ModelConfig.preset(cfg.model)).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=0.0, eps=1e-5)
        opt_path = self.dir / "checkpoints" / "optim.pt"
        if latest.exists() and opt_path.exists():
            self.opt.load_state_dict(torch.load(opt_path, map_location=self.device, weights_only=False))
        self.league = League(self.dir / "league", cfg.league_max, cfg.pfsp_power)
        if not self.league.members:
            self.league.snapshot(self.model, self.update, tags=["anchor"])
        self.active: list[int] = []
        self.refresh_active()
        # Opponent per env (sticky for the episode).
        self.opp = np.array([self.sample_opp() for _ in range(cfg.n_envs)], dtype=np.int64)
        self.obs = self.env.observe()
        self.metrics_f = open(self.dir / "metrics.jsonl", "a")
        self.games_done = 0
        self.t_start = time.time()
        print(f"vgczero: {self.model.n_params()/1e6:.2f}M params on {self.device}; {len(pool)} teams; "
              f"{cfg.n_envs} envs; league {len(self.league.members)}")

    # ---- league ---------------------------------------------------------------------

    def refresh_active(self) -> None:
        self.active = sorted(set(self.league.sample(self.rng, self.cfg.league_active)))

    def sample_opp(self) -> int:
        if not self.active or self.rng.random() < self.cfg.self_play_frac:
            return SELF
        return self.rng.choice(self.active)

    def lr_at(self, u: int) -> float:
        c = self.cfg
        if u < c.warmup_updates:
            return c.lr * (u + 1) / c.warmup_updates
        frac = min(1.0, u / max(1, c.total_updates))
        return c.lr_final + 0.5 * (c.lr - c.lr_final) * (1 + math.cos(math.pi * frac))

    # ---- rollout ----------------------------------------------------------------------

    @torch.no_grad()
    def collect(self) -> tuple[Rollout, dict]:
        c, N, T = self.cfg, self.cfg.n_envs, self.cfg.rollout_steps
        ro = Rollout(T, N)
        self.model.eval()
        stats = {"games": 0, "learner_wins": 0.0, "turns": 0, "illegal": 0, "self_games": 0}
        for t in range(T):
            ints, floats, field, mask, req = self.obs
            ro.ints[t], ro.floats[t], ro.field[t], ro.mask[t], ro.req[t] = ints, floats, field, mask, req
            acts = np.zeros((N, 2, 3), np.int64)
            # Learner rows: side 0 everywhere, side 1 where the opponent is SELF.
            learner = np.zeros((N, 2), bool)
            learner[:, 0] = True
            learner[:, 1] = self.opp == SELF
            o = to_torch(ints, floats, field, mask, req, self.device)
            flat_learner = np.nonzero(learner.reshape(-1))[0]
            if len(flat_learner):
                sub = {k: v[flat_learner] for k, v in o.items()}
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=c.bf16 and self.device.type == "cuda"):
                    a, lp, v = self.model.act(sub)
                av = acts.reshape(-1, 3)
                av[flat_learner] = a.cpu().numpy()
                ro.logp[t].reshape(-1)[flat_learner] = lp.float().cpu().numpy()
                ro.value[t].reshape(-1)[flat_learner] = v.float().cpu().numpy()
            for mid in np.unique(self.opp[self.opp != SELF]):
                envs = np.nonzero(self.opp == mid)[0]
                rows = envs * 2 + 1
                sub = {k: v[rows] for k, v in o.items()}
                a, _, _ = self.league.model(int(mid), self.device).act(sub)
                acts.reshape(-1, 3)[rows] = a.cpu().numpy()
            ro.acts[t] = acts
            ro.valid[t] = learner & (req[..., 0] != D.REQ_WAIT)
            rewards, dones, illegal = self.env.step(acts.astype(np.uint8))
            stats["illegal"] += int(illegal)
            ro.reward[t] = rewards
            ro.done[t] = dones.astype(bool)
            if dones.any():
                done_i, winner, turns, team = self.env.last_infos()
                for e in np.nonzero(dones)[0]:
                    w = int(winner[e])
                    score0 = 1.0 if w == 0 else (0.0 if w == 1 else 0.5)
                    opp = int(self.opp[e])
                    stats["games"] += 1
                    stats["turns"] += int(turns[e])
                    if opp == SELF:
                        stats["self_games"] += 1
                        self.team_stats.record(int(team[e, 0]), score0)
                        self.team_stats.record(int(team[e, 1]), 1.0 - score0)
                    else:
                        stats["learner_wins"] += score0
                        self.league.record(opp, score0)
                        self.team_stats.record(int(team[e, 0]), score0)
                    self.opp[e] = self.sample_opp()
                self.games_done += int(dones.sum())
            self.obs = self.env.observe()
        # Bootstrap values for the state after the rollout.
        ints, floats, field, mask, req = self.obs
        o = to_torch(ints, floats, field, mask, req, self.device)
        H, _ = self.model.encode(o["ints"], o["floats"], o["field"])
        last_v = self.model.value(H).float().cpu().numpy().reshape(N, 2)
        stats["last_value"] = last_v
        return ro, stats

    # ---- update -----------------------------------------------------------------------

    def ppo_update(self, ro: Rollout, last_v: np.ndarray) -> dict:
        c = self.cfg
        adv, ret = compute_gae(ro, last_v, c.gamma, c.gae_lambda)
        sel = ro.valid.reshape(-1)
        idx = np.nonzero(sel)[0]
        if len(idx) == 0:
            return {}
        flat = lambda x: x.reshape(-1, *x.shape[3:])[idx]
        data = {
            "ints": torch.as_tensor(flat(ro.ints), dtype=torch.long),
            "floats": torch.as_tensor(flat(ro.floats)),
            "field": torch.as_tensor(flat(ro.field)),
            "mask": torch.as_tensor(flat(ro.mask), dtype=torch.bool),
            "req": torch.as_tensor(flat(ro.req), dtype=torch.long),
        }
        acts = torch.as_tensor(ro.acts.reshape(-1, 3)[idx])
        old_logp = torch.as_tensor(ro.logp.reshape(-1)[idx])
        advs = torch.as_tensor(adv.reshape(-1)[idx])
        rets = torch.as_tensor(ret.reshape(-1)[idx])
        old_v = torch.as_tensor(ro.value.reshape(-1)[idx])
        n = len(idx)
        lr = self.lr_at(self.update)
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.model.train()
        logs = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "kl": 0.0, "clipfrac": 0.0, "n": n}
        steps = 0
        stop = False
        for _ in range(c.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, c.minibatch):
                mb = perm[s : s + c.minibatch]
                o = {k: v[mb].to(self.device, non_blocking=True) for k, v in data.items()}
                a = acts[mb].to(self.device)
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=c.bf16 and self.device.type == "cuda"):
                    logp, ent, v = self.model.evaluate(o, a)
                logp, ent, v = logp.float(), ent.float(), v.float()
                adv_mb = advs[mb].to(self.device)
                adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
                ratio = torch.exp(logp - old_logp[mb].to(self.device))
                pg = -torch.min(ratio * adv_mb, ratio.clamp(1 - c.clip, 1 + c.clip) * adv_mb).mean()
                ret_mb = rets[mb].to(self.device)
                ov = old_v[mb].to(self.device)
                v_clip = ov + (v - ov).clamp(-c.clip, c.clip)
                v_loss = 0.5 * torch.max((v - ret_mb) ** 2, (v_clip - ret_mb) ** 2).mean()
                loss = pg + c.vf_coef * v_loss - c.ent_coef * ent.mean()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    lr_ = logp - old_logp[mb].to(self.device)
                    kl = ((torch.exp(lr_) - 1) - lr_).mean().item()
                logs["pi_loss"] += pg.item()
                logs["v_loss"] += v_loss.item()
                logs["entropy"] += ent.mean().item()
                logs["kl"] += kl
                logs["clipfrac"] += ((ratio - 1).abs() > c.clip).float().mean().item()
                steps += 1
                if c.target_kl and kl > 1.5 * c.target_kl:
                    stop = True
                    break
            if stop:
                break
        for k in ("pi_loss", "v_loss", "entropy", "kl", "clipfrac"):
            logs[k] /= max(1, steps)
        logs["lr"] = lr
        logs["early_stop"] = stop
        # Explained variance of the value head.
        y, yp = rets.numpy(), old_v.numpy()
        logs["explained_var"] = float(1 - np.var(y - yp) / (np.var(y) + 1e-8))
        return logs

    # ---- main loop ----------------------------------------------------------------------

    def save_checkpoint(self) -> None:
        ck = self.dir / "checkpoints"
        save(self.model, str(ck / "latest.pt"), {"update": self.update})
        torch.save(self.opt.state_dict(), ck / "optim.pt")
        self.team_stats.save(self.dir / "team_stats.json")
        self.league.save_index()

    def evolve_teams(self) -> None:
        """One evolution generation; survivors join the training pool."""
        from .evolve import Evolver

        c = self.cfg
        self.model.eval()
        ranked = [t for t, _, _ in self.team_stats.top(max(c.evolve_population, c.evolve_field), min_games=20)]
        ranked += [t for t in self.pool if t not in ranked]
        if self.evolver is None:
            self.evolver = Evolver(self.model, ranked[: c.evolve_field], str(self.dir / "evolve"), c.evolve_population,
                                   c.evolve_children, c.evolve_games, self.device, c.seed + self.update)
            self.evolver.seed_population(ranked[: c.evolve_population])
        else:
            self.evolver.model = self.model
            self.evolver.field = ranked[: c.evolve_field]
        self.evolver.step()
        added = 0
        for e in self.evolver.pop:
            if e.index not in self.team_stats.pos:
                self.team_stats.add(e.index)
                self.pool.append(e.index)
                self.evolved.append({"name": e.team["name"], "mons": e.team["mons"]})
                added += 1
        if added:
            self.evolved_path.write_text(json.dumps(self.evolved))
            # Rebuild the env with the larger pool (running games restart).
            self.env = D.E.VecEnv(c.n_envs, seed=c.seed + self.update, team_indices=self.pool, turn_limit=c.turn_limit,
                                  open_sheet_prob=c.open_sheet_prob)
            if c.team_focus > 0:
                self.env.set_team_weights(0, self.team_stats.sampling_weights(c.team_focus))
            self.opp = np.array([self.sample_opp() for _ in range(c.n_envs)], dtype=np.int64)
            self.obs = self.env.observe()
        print(f"  evolve u{self.update}: {added} new teams in the training pool ({len(self.pool)} total)")

    def train(self, n_updates: int | None = None) -> None:
        c = self.cfg
        end = self.update + (n_updates if n_updates is not None else c.total_updates)
        while self.update < end:
            t0 = time.time()
            ro, st = self.collect()
            t1 = time.time()
            logs = self.ppo_update(ro, st.pop("last_value"))
            t2 = time.time()
            self.update += 1
            if self.update % c.league_every == 0:
                self.league.snapshot(self.model, self.update)
            if self.update % c.league_refresh == 0:
                self.refresh_active()
            if c.team_focus > 0 and self.update % c.team_update_every == 0:
                self.env.set_team_weights(0, self.team_stats.sampling_weights(c.team_focus))
            if self.update % c.save_every == 0:
                self.save_checkpoint()
            if c.evolve_every and self.update % c.evolve_every == 0:
                self.evolve_teams()
            if c.eval_every and self.update % c.eval_every == 0:
                from .evaluate import play_match

                self.model.eval()
                r = play_match(self.model, "greedy", c.eval_games, min(256, c.eval_games), team_indices=self.pool,
                               seed=self.update, device=self.device)
                ev = {"update": self.update, "games_total": self.games_done, "vs_greedy": r.win_rate, "vs_greedy_n": r.games,
                      "elapsed_h": round((time.time() - self.t_start) / 3600, 3)}
                with open(self.dir / "eval.jsonl", "a") as f:
                    f.write(json.dumps(ev) + "\n")
                print(f"  eval u{self.update}: vs greedy {r}")
            decisions = int(ro.valid.sum())
            rec = {
                "update": self.update,
                "games_total": self.games_done,
                "games": st["games"],
                "vs_league_wr": st["learner_wins"] / max(1, st["games"] - st["self_games"]),
                "turns_per_game": st["turns"] / max(1, st["games"]),
                "illegal": st["illegal"],
                "decisions": decisions,
                "collect_s": round(t1 - t0, 3),
                "update_s": round(t2 - t1, 3),
                "env_steps_per_s": round(c.n_envs * c.rollout_steps / max(1e-9, t1 - t0)),
                "games_per_s": round(st["games"] / max(1e-9, t2 - t0), 1),
                "elapsed_h": round((time.time() - self.t_start) / 3600, 3),
                **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in logs.items()},
            }
            self.metrics_f.write(json.dumps(rec) + "\n")
            self.metrics_f.flush()
            if self.update % c.log_every == 0:
                print(
                    f"u{self.update} games {self.games_done} ({rec['games_per_s']}/s) "
                    f"vsLeague {rec['vs_league_wr']:.3f} turns {rec['turns_per_game']:.1f} "
                    f"ent {rec.get('entropy', 0):.3f} kl {rec.get('kl', 0):.4f} ev {rec.get('explained_var', 0):.2f} "
                    f"| collect {rec['collect_s']}s update {rec['update_s']}s | league {len(self.league.members)}"
                )
        self.save_checkpoint()


def load_policy(path: str, device: str = "cpu") -> VGCNet:
    return load_model(path, device=device)
