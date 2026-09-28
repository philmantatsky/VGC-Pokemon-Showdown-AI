"""Tactical fine-tune: teach the deployed brain the move and damage facts the
guards patch at play time (training/tactical_teacher.py), on positions from
training/gen_tactical_data.py.

For every recorded decision and both positions the target is the brain's OWN
distribution with the teacher's corrections: certainly-useless actions lose
their mass, and the mass the brain puts on plain attacks is re-spread over them
by value (temperature --tau). Everything else -- attack vs Protect vs switch vs
support, Mega timing -- keeps the brain's own proportions, so the loss
(cross-entropy to that target = KL + constant) also anchors the rest of the
policy. Slot 2 is trained conditioned on slot 1's played action (the joint
head). Only the actor trains: the critic has its own feature extractor and is
frozen, so the value function is untouched.

The start checkpoint is never modified: saves go to --output as new files
(SB3 zips like any training save, loadable by every player). Validation is a
10% split by battle. Local comparison only: never promotion or ladder.

Usage (from the repo root):
  .venv/bin/python training/tactical_sft.py --data results_tactical1/data \
      --output results_tactical1/sft
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import copy
import hashlib
import json
import math
import time

import numpy as np
import torch
from stable_baselines3 import PPO

from training.tactical_teacher import doomed_target, pair_target, target_distribution


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(data_dir: Path) -> dict[str, np.ndarray]:
    parts = sorted(data_dir.glob("*.npz"))
    if not parts:
        raise FileNotFoundError(f"no shards in {data_dir}")
    arrays: dict[str, list[np.ndarray]] = {}
    for part in parts:
        with np.load(part, allow_pickle=False) as z:
            for key in z.files:
                arrays.setdefault(key, []).append(z[key])
    return {k: np.concatenate(v) for k, v in arrays.items()}


def split_by_battle(battles: np.ndarray, fraction: float = 0.1) -> np.ndarray:
    """True for validation rows: a stable hash of the battle tag."""
    return np.array(
        [
            int(hashlib.sha256(b.encode()).hexdigest()[:8], 16) % 1000 < fraction * 1000
            for b in battles
        ]
    )


@torch.no_grad()
def policy_probs(policy, obs, mask, played0, device, batch=512):
    """(p0, p1 | played slot-1 action) for every row, as float64 numpy."""
    p0s, p1s = [], []
    for start in range(0, len(obs), batch):
        o = torch.as_tensor(obs[start : start + batch], device=device)
        m = torch.as_tensor(mask[start : start + batch].astype(np.int64), device=device)
        a0 = torch.as_tensor(
            played0[start : start + batch].astype(np.int64), device=device
        )[:, None]
        logits, _, latent = policy.logits_with_latent(
            {"observation": o, "action_mask": m}, False
        )
        d0 = policy.get_dist_from_logits(logits, m)
        d1 = policy.get_dist_from_logits(logits, m, a0, latent)
        p0s.append(d0.distribution[0].probs.cpu().double().numpy())
        p1s.append(d1.distribution[1].probs.cpu().double().numpy())
    return np.concatenate(p0s), np.concatenate(p1s)


def build_targets(
    p0,
    p1,
    useless,
    values,
    tau,
    doomed=None,
    protect=None,
    wasted=None,
    drain=None,
    receive=None,
):
    """Targets for both positions and which rows carry a lesson. With the doomed
    facts (data since 2026-09-27) a Pokemon likely knocked out before it moves
    also has that share of its wasted moves' mass moved onto Protect; with the
    pairing facts (since 2026-09-28) slot 2's Protect beside a Fake Out, or its
    Fake Out beside a Protect, gives its mass to its other moves."""
    q0 = p0.copy()
    q1 = p1.copy()
    lesson = np.zeros((len(p0), 2), dtype=bool)
    for i in range(len(p0)):
        for pos, (p, q) in enumerate(((p0, q0), (p1, q1))):
            t = target_distribution(p[i], useless[i, pos], values[i, pos], tau)
            if (
                doomed is not None
                and protect is not None
                and wasted is not None
                and doomed[i, pos] > 0
            ):
                d = doomed_target(
                    p[i] if t is None else t,
                    float(doomed[i, pos]),
                    protect[i, pos],
                    wasted[i, pos],
                )
                t = d if d is not None else t
            if drain is not None and receive is not None and drain[i, pos].any():
                r = pair_target(
                    p[i] if t is None else t, drain[i, pos], receive[i, pos]
                )
                t = r if r is not None else t
            if t is not None:
                q[i] = t
                lesson[i, pos] = True
    return q0.astype(np.float32), q1.astype(np.float32), lesson


def masked_log_probs(policy, obs, mask, played0):
    logits, _, latent = policy.logits_with_latent(
        {"observation": obs, "action_mask": mask}, True
    )
    d0 = policy.get_dist_from_logits(logits, mask)
    d1 = policy.get_dist_from_logits(logits, mask, played0, latent)
    lp0 = torch.log_softmax(d0.distribution[0].logits, dim=-1)
    lp1 = torch.log_softmax(d1.distribution[1].logits, dim=-1)
    return lp0, lp1


def cross_entropy(q: torch.Tensor, logp: torch.Tensor) -> torch.Tensor:
    return -torch.where(q > 0, q * logp, torch.zeros_like(q)).sum(-1)


def evaluate(policy, data, rows, q0, q1, lesson, device, batch=512) -> dict[str, float]:
    """Mean cross-entropy to the targets, teacher agreement on lesson rows, the
    probability mass left on useless actions, and drift on lesson-free rows."""
    ce_sum = 0.0
    agree = total = 0
    useless_mass = useless_rows = 0.0
    drift_sum = drift_rows = 0.0
    doomed_mass = doomed_rows = 0.0
    pair_mass = pair_rows = 0.0
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            idx = rows[start : start + batch]
            o = torch.as_tensor(data["obs"][idx], device=device)
            m = torch.as_tensor(data["mask"][idx].astype(np.int64), device=device)
            a0 = torch.as_tensor(
                data["played"][idx, 0].astype(np.int64), device=device
            )[:, None]
            lp0, lp1 = masked_log_probs(policy, o, m, a0)
            t0 = torch.as_tensor(q0[idx], device=device)
            t1 = torch.as_tensor(q1[idx], device=device)
            ce_sum += float((cross_entropy(t0, lp0) + cross_entropy(t1, lp1)).sum())
            for pos, lp in enumerate((lp0, lp1)):
                probs = lp.exp().cpu().double().numpy()
                vals = data["values"][idx, pos]
                flags = data["useless"][idx, pos]
                for j in range(len(idx)):
                    row_lesson = lesson[idx[j], pos]
                    if "drain" in data and data["drain"][idx[j], pos].any():
                        pair_mass += float(probs[j][data["drain"][idx[j], pos]].sum())
                        pair_rows += 1
                    if "doomed" in data and data["doomed"][idx[j], pos] > 0:
                        doomed_mass += float(
                            probs[j][data["wasted"][idx[j], pos]].sum()
                        )
                        doomed_rows += 1
                    if flags[j].any():
                        useless_mass += float(probs[j][flags[j]].sum())
                        useless_rows += 1
                    valued = ~np.isnan(vals[j]) & (probs[j] > 0)
                    if row_lesson and valued.sum() >= 2:
                        cand = np.flatnonzero(valued)
                        best = cand[np.argmax(vals[j][cand])]
                        top = cand[np.argmax(probs[j][cand])]
                        # agreement: the brain's favourite attack is the teacher's
                        # best, or within 0.02 of its value (a near-tie)
                        agree += int(
                            top == best or vals[j][best] - vals[j][top] <= 0.02
                        )
                        total += 1
                    if not row_lesson:
                        target = (q0 if pos == 0 else q1)[idx[j]]
                        keep = target > 0
                        drift_sum += float(
                            np.sum(
                                target[keep]
                                * (
                                    np.log(target[keep])
                                    - np.log(np.maximum(probs[j][keep], 1e-12))
                                )
                            )
                        )
                        drift_rows += 1
    return {
        "cross_entropy": ce_sum / max(1, len(rows)),
        "teacher_agreement": agree / max(1, total),
        "lesson_rows": total,
        "useless_mass": useless_mass / max(1, useless_rows),
        "drift_kl_no_lesson": drift_sum / max(1, drift_rows),
        # the probability still put on moves a likely knocked-out Pokemon wastes
        "doomed_wasted_mass": doomed_mass / max(1, doomed_rows),
        "doomed_rows": doomed_rows,
        # slot 2's Protect beside a Fake Out / Fake Out beside a Protect
        "pair_drain_mass": pair_mass / max(1, pair_rows),
        "pair_rows": pair_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="results_deployed/champion_mc_T6ctx.zip")
    ap.add_argument("--data", type=Path, default=Path("results_tactical1/data"))
    ap.add_argument("--output", type=Path, default=Path("results_tactical1/sft"))
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20926)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    start_sha = sha256(ROOT / args.checkpoint)

    data = load_data(args.data)
    val = split_by_battle(data["battle"])
    train_rows = np.flatnonzero(~val)
    val_rows = np.flatnonzero(val)
    model = PPO.load(ROOT / args.checkpoint, device=device)
    policy = model.policy
    policy.eval()
    frozen = copy.deepcopy(policy).eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    p0, p1 = policy_probs(
        frozen, data["obs"], data["mask"], data["played"][:, 0], device
    )
    q0, q1, lesson = build_targets(
        p0,
        p1,
        data["useless"],
        data["values"],
        args.tau,
        data.get("doomed"),
        data.get("protect"),
        data.get("wasted"),
        data.get("drain"),
        data.get("receive"),
    )

    critic = (
        list(policy.vf_features_extractor.parameters())
        + list(policy.mlp_extractor.value_net.parameters())
        + list(policy.value_net.parameters())
    )
    critic_ids = {id(p) for p in critic}
    for p in critic:
        p.requires_grad_(False)
    actor = [
        p for p in policy.parameters() if id(p) not in critic_ids and p.requires_grad
    ]
    optimizer = torch.optim.Adam(actor, lr=args.lr)

    log = {
        "start_checkpoint": args.checkpoint,
        "start_sha256": start_sha,
        "data": str(args.data),
        "rows": int(len(data["obs"])),
        "train_rows": int(len(train_rows)),
        "val_rows": int(len(val_rows)),
        "lesson_rows_pos0": int(lesson[:, 0].sum()),
        "lesson_rows_pos1": int(lesson[:, 1].sum()),
        "doomed_rows": int((data["doomed"] > 0).sum()) if "doomed" in data else 0,
        "args": {k: str(v) for k, v in vars(args).items()},
        "epochs": [],
    }
    before = evaluate(policy, data, val_rows, q0, q1, lesson, device)
    log["val_before"] = before
    print(json.dumps({"epoch": 0, **before}), flush=True)
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        order = rng.permutation(train_rows)
        running = 0.0
        for start in range(0, len(order), args.batch):
            idx = np.sort(order[start : start + args.batch])
            o = torch.as_tensor(data["obs"][idx], device=device)
            m = torch.as_tensor(data["mask"][idx].astype(np.int64), device=device)
            a0 = torch.as_tensor(
                data["played"][idx, 0].astype(np.int64), device=device
            )[:, None]
            lp0, lp1 = masked_log_probs(policy, o, m, a0)
            t0 = torch.as_tensor(q0[idx], device=device)
            t1 = torch.as_tensor(q1[idx], device=device)
            loss = (cross_entropy(t0, lp0) + cross_entropy(t1, lp1)).mean()
            if not math.isfinite(float(loss)):
                raise RuntimeError("non-finite loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor, 0.5)
            optimizer.step()
            running += float(loss) * len(idx)
        metrics = evaluate(policy, data, val_rows, q0, q1, lesson, device)
        row = {
            "epoch": epoch,
            "train_loss": running / len(order),
            **metrics,
            "minutes": round((time.time() - started) / 60, 1),
        }
        log["epochs"].append(row)
        print(json.dumps(row), flush=True)
        save = args.output / f"tactical_e{epoch}.zip"
        model.save(save)
        (args.output / f"tactical_e{epoch}.zip.metadata.json").write_text(
            json.dumps(
                {
                    "role": "candidate",
                    "method": "tactical fine-tune (training/tactical_sft.py)",
                    "start_checkpoint": args.checkpoint,
                    "start_sha256": start_sha,
                    "epoch": epoch,
                    "tau": args.tau,
                    "lr": args.lr,
                    "sha256": sha256(save),
                },
                indent=2,
            )
            + "\n"
        )
    if sha256(ROOT / args.checkpoint) != start_sha:
        raise RuntimeError("the start checkpoint changed during fine-tuning")
    (args.output / "log.json").write_text(json.dumps(log, indent=2) + "\n")


if __name__ == "__main__":
    main()
