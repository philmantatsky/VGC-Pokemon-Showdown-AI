"""Team-preview entropy boost for PPO, applied without touching vgc_bench/.

Why (2026-09-23): the deployed T6 brain plays one fixed script. At team preview
it leads Farigiraf + Torkoal with probability ~0.998 / ~0.915 and brings
Blastoise + Charizard ~0.99, whatever the opponent shows (ladder decision log,
ladder_replays_mc_deployed_T6/decisions.jsonl); Codex's local runs saw the same
lead in 300/300 and 6,204/6,204 games. PPO's entropy floor (ent_coef 0.02 late
in training, vgc_bench/src/callback.py) did not keep the preview distribution
from collapsing, so alternative openings stopped being explored and the brain
never learned which opening fits which matchup.

This multiplies the entropy term of TEAM-PREVIEW samples by (1 + boost) inside
MaskedActorCriticPolicy.evaluate_actions, i.e. an on-policy, preview-only
entropy bonus: every other decision, the value loss and inference (forward /
predict, used at deployment) are unchanged. The preview flag is the
`teampreview` feature of embed_global, the first block of every token.

It is a launcher-level patch on purpose: vgc_bench/ stays byte-identical, so
the pinned deployed-T6 arm of results_t6_vs_deployed_v1 remains a valid
reference for the comparison. Integrate it into policy.py only if it helps.

Usage (from the repo root; everything after "--" goes to vgc_bench.train):
  .venv/bin/python -u training/preview_entropy.py --boost 4 -- --fictitious_play ...
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import runpy

import torch
from poke_env.battle import Field, Weather

from vgc_bench.src.policy import MaskedActorCriticPolicy
from vgc_bench.src.utils import glob_obs_len

# embed_global: [*weather, *fields, champions_format, teampreview, reviving, ...]
PREVIEW_INDEX = len(Weather) + len(Field) + 1
if PREVIEW_INDEX >= glob_obs_len:  # never silently read the wrong feature
    raise RuntimeError("team-preview flag is outside the global block")


def preview_mask(observation: torch.Tensor) -> torch.Tensor:
    """1.0 for samples whose observation is a team-preview decision, else 0.0."""
    flat = observation.reshape(observation.shape[0], -1)
    return (flat[:, PREVIEW_INDEX] > 0.5).to(flat.dtype)


def install(boost: float) -> None:
    """Patch evaluate_actions so preview entropy counts (1 + boost) times."""
    if boost < 0:
        raise ValueError("boost must be >= 0")
    original = MaskedActorCriticPolicy.evaluate_actions
    if getattr(original, "_preview_entropy_boost", None) is not None:
        raise RuntimeError("preview entropy boost already installed")

    def evaluate_actions(self, obs, actions):
        values, log_prob, entropy = original(self, obs, actions)
        weight = 1.0 + boost * preview_mask(obs["observation"]).to(entropy.dtype)
        return values, log_prob, entropy * weight

    evaluate_actions._preview_entropy_boost = boost  # type: ignore[attr-defined]
    MaskedActorCriticPolicy.evaluate_actions = evaluate_actions  # type: ignore[method-assign]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boost", type=float, required=True)
    ap.add_argument("train_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    train_args = (
        args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args
    )
    install(args.boost)
    print(
        f"preview entropy boost {args.boost} installed (flag index {PREVIEW_INDEX})",
        flush=True,
    )
    sys.argv = ["vgc_bench.train", *train_args]
    runpy.run_module("vgc_bench.train", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
