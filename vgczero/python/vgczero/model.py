"""Entity transformer policy/value network.

Tokens: [CLS] + 12 Pokemon (own team 0..5, opponent 0..5) + field = 14.
Each Pokemon token combines learned embeddings (species, ability, item, four
moves, last move, status, position, owner) with static dex features (types,
base stats, move data) and the numeric observation. Nothing beyond the game
state is given: no damage calculator, usage statistics or speed resolver.

Heads
  value    tanh scalar, expected outcome for the player observing
  preview  90 bring/lead options, scored from the chosen Pokemon's tokens
  slots    two autoregressive heads over 31 actions each; the second slot
           sees an embedding of the first slot's chosen action, so plans like
           double-targeting, Protect + attack and coordinated switches can be
           represented jointly
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data as D

NEG = -1e9


def pick_device(name: str = "auto") -> torch.device:
    """'auto' picks CUDA, then Apple's MPS, then CPU; anything else is passed to torch."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class ModelConfig:
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1280
    d_species: int = 64
    d_ability: int = 32
    d_item: int = 32
    d_move: int = 64
    d_small: int = 8

    @staticmethod
    def preset(name: str) -> "ModelConfig":
        if name == "tiny":  # tests / CPU smoke runs
            return ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, d_species=16, d_ability=8, d_item=8, d_move=16)
        if name == "small":  # laptops
            return ModelConfig(d_model=192, n_layers=4, n_heads=6, d_ff=768)
        if name == "base":  # ~8.7M parameters, the mikumiku37 scale
            return ModelConfig()
        raise ValueError(name)


def mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class VGCNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D.load()
        self.cfg = cfg
        n_sp, n_mv, n_it, n_ab, n_status, n_pos = D.E.vocab()
        n_mons, mon_int, mon_float, field_float, _, n_act, n_prev = D.E.dims()
        tab = D.static_tables()
        self.register_buffer("sp_tab", torch.tensor(tab["species"], dtype=torch.float32), persistent=False)
        self.register_buffer("mv_tab", torch.tensor(tab["moves"], dtype=torch.float32), persistent=False)
        self.register_buffer("it_tab", torch.tensor(tab["items"], dtype=torch.float32), persistent=False)
        self.register_buffer("preview_idx", torch.tensor(D.preview_options(), dtype=torch.long), persistent=False)
        d = cfg.d_model
        s = cfg.d_small
        self.sp_emb = nn.Embedding(n_sp, cfg.d_species)
        self.ab_emb = nn.Embedding(n_ab, cfg.d_ability)
        self.it_emb = nn.Embedding(n_it, cfg.d_item)
        self.mv_emb = nn.Embedding(n_mv, cfg.d_move)
        self.status_emb = nn.Embedding(n_status, s)
        self.pos_emb = nn.Embedding(n_pos, s)
        self.owner_emb = nn.Embedding(2, s)
        self.move_enc = mlp(cfg.d_move + self.mv_tab.shape[1], 2 * cfg.d_move, cfg.d_move)
        mon_in = (
            cfg.d_species + self.sp_tab.shape[1] + cfg.d_ability + cfg.d_item + self.it_tab.shape[1]
            + 5 * cfg.d_move + 3 * s + mon_float
        )
        self.mon_proj = mlp(mon_in, 2 * d, d)
        self.field_proj = mlp(field_float, 2 * d, d)
        self.cls = nn.Parameter(torch.zeros(d))
        self.type_emb = nn.Parameter(torch.zeros(4, d))  # cls / own / opp / field
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, cfg.d_ff, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d)
        # Heads.
        self.value_head = mlp(d, d, 1)
        self.q_proj = nn.Linear(d, d)
        self.self_proj = nn.Linear(d, d)
        self.pass_vec = nn.Parameter(torch.zeros(d))
        self.sw_proj = nn.Linear(d, d)
        self.mv_act_proj = nn.Linear(cfg.d_move, d)
        self.tgt_proj = nn.Linear(d, d)
        self.no_tgt = nn.Parameter(torch.zeros(d))
        self.mega_emb = nn.Parameter(torch.zeros(2, d))
        self.slot_emb = nn.Parameter(torch.zeros(2, d))
        self.cond_proj = nn.Linear(d, d)
        self.act_mlp = mlp(d, d, 1)
        self.lead_proj = nn.Linear(d, d)
        self.back_proj = nn.Linear(d, d)
        self.prev_mlp = mlp(d, d, 1)
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.type_emb, std=0.02)
        nn.init.normal_(self.pass_vec, std=0.02)
        nn.init.normal_(self.mega_emb, std=0.02)
        nn.init.normal_(self.slot_emb, std=0.02)
        nn.init.normal_(self.no_tgt, std=0.02)

    # ---- encoder ---------------------------------------------------------------------

    def encode(self, ints: torch.Tensor, floats: torch.Tensor, field: torch.Tensor):
        """ints [B,12,11] long, floats [B,12,F], field [B,Ff] -> (H [B,14,d], move_vecs [B,12,4,dm])."""
        B = ints.shape[0]
        sp = ints[..., D.I_SPECIES]
        mv_ids = ints[..., D.I_MOVE0 : D.I_MOVE0 + 4]
        mv = torch.cat([self.mv_emb(mv_ids), self.mv_tab[mv_ids]], -1)
        mv = self.move_enc(mv)  # [B,12,4,dm]
        last = ints[..., D.I_LAST_MOVE]
        last_v = self.move_enc(torch.cat([self.mv_emb(last), self.mv_tab[last]], -1))
        it = ints[..., D.I_ITEM]
        x = torch.cat(
            [
                self.sp_emb(sp),
                self.sp_tab[sp],
                self.ab_emb(ints[..., D.I_ABILITY]),
                self.it_emb(it),
                self.it_tab[it],
                mv.flatten(-2),
                last_v,
                self.status_emb(ints[..., D.I_STATUS]),
                self.pos_emb(ints[..., D.I_POSITION]),
                self.owner_emb(ints[..., D.I_OWNER]),
                floats,
            ],
            -1,
        )
        mons = self.mon_proj(x)  # [B,12,d]
        mons = mons + torch.cat([self.type_emb[1].expand(6, -1), self.type_emb[2].expand(6, -1)], 0)
        fld = (self.field_proj(field) + self.type_emb[3]).unsqueeze(1)
        cls = (self.cls + self.type_emb[0]).expand(B, 1, -1)
        h = self.encoder(torch.cat([cls, mons, fld], 1))
        return self.out_norm(h), mv

    def value(self, H: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.value_head(H[:, 0])).squeeze(-1)

    # ---- action features -------------------------------------------------------------

    def _slot_tokens(self, H, mv, ints):
        """Own active / foe tokens and own active move vectors."""
        own = H[:, 1:7]
        opp = H[:, 7:13]
        pos = ints[..., D.I_POSITION]
        own_pos, opp_pos = pos[:, :6], pos[:, 6:]

        def pick(pos6, code, tok6):
            oh = (pos6 == code).float()
            return torch.einsum("bj,bjd->bd", oh, tok6), oh.sum(-1) > 0, oh

        s0, e0, oh0 = pick(own_pos, D.POS_ACTIVE0, own)
        s1, e1, oh1 = pick(own_pos, D.POS_ACTIVE1, own)
        f0, fe0, _ = pick(opp_pos, D.POS_ACTIVE0, opp)
        f1, fe1, _ = pick(opp_pos, D.POS_ACTIVE1, opp)
        m0 = torch.einsum("bj,bjmd->bmd", oh0, mv[:, :6])
        m1 = torch.einsum("bj,bjmd->bmd", oh1, mv[:, :6])
        return own, (s0, s1), (e0, e1), (f0, f1), (fe0, fe1), (m0, m1)

    def action_features(self, H, mv, ints):
        """Per-slot action feature vectors A [B,2,31,d] and slot queries Q [B,2,d]."""
        own, slot_tok, slot_exists, foe_tok, foe_exists, slot_moves = self._slot_tokens(H, mv, ints)
        B, d = H.shape[0], H.shape[-1]
        sw = self.sw_proj(own)  # [B,6,d]
        As, Qs = [], []
        no_tgt = self.no_tgt.expand(B, d)
        tf0 = torch.where(foe_exists[0].unsqueeze(-1), self.tgt_proj(foe_tok[0]), no_tgt)
        tf1 = torch.where(foe_exists[1].unsqueeze(-1), self.tgt_proj(foe_tok[1]), no_tgt)
        for k in range(2):
            ally_ok = slot_exists[1 - k].unsqueeze(-1)
            ta = torch.where(ally_ok, self.tgt_proj(slot_tok[1 - k]), no_tgt)
            tgts = torch.stack([tf0, tf1, ta], 1)  # [B,3,d]
            mvp = self.mv_act_proj(slot_moves[k])  # [B,4,d]
            # index = (m*3 + t)*2 + g
            mov = mvp[:, :, None, None, :] + tgts[:, None, :, None, :] + self.mega_emb[None, None, None, :, :]
            mov = mov.reshape(B, 24, d)
            a = torch.cat([self.pass_vec.expand(B, 1, d), sw, mov], 1)  # [B,31,d]
            As.append(a)
            Qs.append(self.q_proj(H[:, 0]) + self.self_proj(slot_tok[k]) + self.slot_emb[k])
        return torch.stack(As, 1), torch.stack(Qs, 1)

    def slot_logits(self, A_k, q_k):
        """A_k [B,31,d], q_k [B,d] -> logits [B,31]."""
        return self.act_mlp(F.gelu(A_k + q_k.unsqueeze(1))).squeeze(-1)

    def preview_logits(self, H):
        own = H[:, 1:7]
        idx = self.preview_idx  # [90,4]
        lead = own[:, idx[:, 0]] + own[:, idx[:, 1]]
        back = own[:, idx[:, 2]] + own[:, idx[:, 3]]
        x = self.lead_proj(lead) + self.back_proj(back) + self.q_proj(H[:, 0]).unsqueeze(1)
        return self.prev_mlp(F.gelu(x)).squeeze(-1)

    # ---- acting / evaluating ------------------------------------------------------------

    @staticmethod
    def cond_mask(mask1: torch.Tensor, a0: torch.Tensor) -> torch.Tensor:
        """Slot-1 legality given slot 0's action: no shared switch target, one Mega."""
        m = mask1.clone()
        B = a0.shape[0]
        ar = torch.arange(B, device=a0.device)
        sw = (a0 >= 1) & (a0 <= 6)
        m[ar[sw], a0[sw]] = False
        mega0 = (a0 >= 7) & (((a0 - 7) % 2) == 1)
        if mega0.any():
            mega_cols = torch.arange(8, 31, 2, device=a0.device)
            m[mega0.nonzero(as_tuple=True)[0][:, None], mega_cols[None, :]] = False
        # Never leave a row empty.
        empty = ~m.any(-1)
        if empty.any():
            m[empty, 0] = True
        return m

    def forward_all(self, ints, floats, field):
        H, mv = self.encode(ints, floats, field)
        A, Q = self.action_features(H, mv, ints)
        return H, A, Q

    def act(self, obs: dict[str, torch.Tensor], deterministic: bool = False):
        """Sample actions. Returns (actions [B,3] long, logp [B], value [B])."""
        ints, floats, field, mask, req = obs["ints"], obs["floats"], obs["field"], obs["mask"], obs["req"]
        H, A, Q = self.forward_all(ints, floats, field)
        v = self.value(H)
        B = ints.shape[0]
        ar = torch.arange(B, device=ints.device)
        pl = self.preview_logits(H)
        pdist = torch.distributions.Categorical(logits=pl)
        p = pl.argmax(-1) if deterministic else pdist.sample()
        l0 = self.slot_logits(A[:, 0], Q[:, 0]).masked_fill(~mask[:, 0], NEG)
        d0 = torch.distributions.Categorical(logits=l0)
        a0 = l0.argmax(-1) if deterministic else d0.sample()
        q1 = Q[:, 1] + self.cond_proj(A[ar, 0, a0])
        m1 = self.cond_mask(mask[:, 1], a0)
        l1 = self.slot_logits(A[:, 1], q1).masked_fill(~m1, NEG)
        d1 = torch.distributions.Categorical(logits=l1)
        a1 = l1.argmax(-1) if deterministic else d1.sample()
        kind = req[:, 0]
        logp = torch.where(kind == D.REQ_PREVIEW, pdist.log_prob(p), d0.log_prob(a0) + d1.log_prob(a1))
        acts = torch.stack([p, a0, a1], -1)
        return acts, logp, v

    def evaluate(self, obs: dict[str, torch.Tensor], acts: torch.Tensor):
        """Log-prob, entropy and value of given actions (teacher-forced slot 0)."""
        ints, floats, field, mask, req = obs["ints"], obs["floats"], obs["field"], obs["mask"], obs["req"]
        H, A, Q = self.forward_all(ints, floats, field)
        v = self.value(H)
        B = ints.shape[0]
        ar = torch.arange(B, device=ints.device)
        p, a0, a1 = acts[:, 0], acts[:, 1], acts[:, 2]
        pl = self.preview_logits(H)
        pdist = torch.distributions.Categorical(logits=pl)
        l0 = self.slot_logits(A[:, 0], Q[:, 0]).masked_fill(~mask[:, 0], NEG)
        d0 = torch.distributions.Categorical(logits=l0)
        q1 = Q[:, 1] + self.cond_proj(A[ar, 0, a0])
        m1 = self.cond_mask(mask[:, 1], a0)
        l1 = self.slot_logits(A[:, 1], q1).masked_fill(~m1, NEG)
        d1 = torch.distributions.Categorical(logits=l1)
        is_prev = req[:, 0] == D.REQ_PREVIEW
        logp = torch.where(is_prev, pdist.log_prob(p), d0.log_prob(a0) + d1.log_prob(a1))
        ent = torch.where(is_prev, pdist.entropy(), d0.entropy() + d1.entropy())
        return logp, ent, v

    @torch.no_grad()
    def top_joint(self, obs: dict[str, torch.Tensor], k: int):
        """Top-k joint actions per sample with probabilities.

        Returns (choices [B,k,3] long, probs [B,k]) where choices are
        (preview, slot0, slot1). For team preview the 90 options are ranked.
        Rows with fewer than k legal joint actions are padded with the best
        one (probability 0).
        """
        ints, floats, field, mask, req = obs["ints"], obs["floats"], obs["field"], obs["mask"], obs["req"]
        H, A, Q = self.forward_all(ints, floats, field)
        B, dev = ints.shape[0], ints.device
        pl = F.log_softmax(self.preview_logits(H), -1)
        l0 = F.log_softmax(self.slot_logits(A[:, 0], Q[:, 0]).masked_fill(~mask[:, 0], NEG), -1)  # [B,31]
        # Slot 1 conditioned on every possible slot-0 action: [B,31,31].
        q1 = Q[:, 1].unsqueeze(1) + self.cond_proj(A[:, 0])  # [B,31,d]
        l1 = self.act_mlp(F.gelu(A[:, 1].unsqueeze(1) + q1.unsqueeze(2))).squeeze(-1)  # [B,31(a0),31(a1)]
        m1 = mask[:, 1].unsqueeze(1).expand(B, 31, 31).clone()
        a0s = torch.arange(31, device=dev)
        m1[:, a0s[1:7], a0s[1:7]] = False  # same switch target
        mega_rows = a0s[(a0s >= 7) & ((a0s - 7) % 2 == 1)]
        mega_cols = mega_rows
        m1[:, mega_rows[:, None], mega_cols[None, :]] = False
        empty = ~m1.any(-1)
        m1[..., 0] = m1[..., 0] | empty
        l1 = F.log_softmax(l1.masked_fill(~m1, NEG), -1)
        joint = l0.unsqueeze(-1) + l1  # [B,31,31]
        joint = joint.masked_fill(~mask[:, 0].unsqueeze(-1), NEG)
        flat = joint.reshape(B, -1)
        kk = min(k, flat.shape[1])
        lp, ix = flat.topk(kk, -1)
        a0, a1 = ix // 31, ix % 31
        pp, pix = pl.topk(min(k, 90), -1)
        if pix.shape[1] < kk:
            pix = F.pad(pix, (0, kk - pix.shape[1]), value=0)
            pp = F.pad(pp, (0, kk - pp.shape[1]), value=NEG)
        is_prev = (req[:, 0] == D.REQ_PREVIEW).unsqueeze(-1)
        choices = torch.stack([torch.where(is_prev, pix[:, :kk], 0), torch.where(is_prev, 0, a0), torch.where(is_prev, 0, a1)], -1)
        probs = torch.where(is_prev, pp[:, :kk].exp(), lp.exp())
        return choices, probs

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config_dict(self) -> dict:
        return asdict(self.cfg)


def to_torch(ints, floats, field, mask, req, device) -> dict[str, torch.Tensor]:
    """Engine numpy observation -> model input dict (any leading shape flattened)."""
    n = int(np.prod(ints.shape[:-2]))
    return {
        "ints": torch.as_tensor(np.ascontiguousarray(ints).reshape(n, *ints.shape[-2:]), dtype=torch.long, device=device),
        "floats": torch.as_tensor(np.ascontiguousarray(floats).reshape(n, *floats.shape[-2:]), device=device),
        "field": torch.as_tensor(np.ascontiguousarray(field).reshape(n, field.shape[-1]), device=device),
        "mask": torch.as_tensor(np.ascontiguousarray(mask).reshape(n, 2, mask.shape[-1]), dtype=torch.bool, device=device),
        "req": torch.as_tensor(np.ascontiguousarray(req).reshape(n, 3), dtype=torch.long, device=device),
    }


def atomic_torch_save(obj, path) -> None:
    """torch.save through a temp file and a rename (a kill mid-save keeps the old file)."""
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, str(path))


def save(model: VGCNet, path: str, extra: dict | None = None) -> None:
    atomic_torch_save({"config": model.config_dict(), "state": model.state_dict(), "extra": extra or {}}, path)


def load_model(path: str, device="cpu") -> VGCNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = VGCNet(ModelConfig(**ck["config"])).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    return m
