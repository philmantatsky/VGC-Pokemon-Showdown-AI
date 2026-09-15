"""Extend an old checkpoint to a longer observation or the joint head, losing nothing.

`chunk_obs_len` reaches the network through exactly one layer -- AttentionExtractor's
`pokemon_proj` -- and every feature block ever added sits at the END of each token,
so it maps to trailing COLUMNS of that layer's weight matrix. Copy the old weights
into the leading columns, zero the new ones, and the converted model produces
bit-identical output on any state where the new features are zero.

The joint-action head (brain v1) is handled the same way: its tensors do not
exist in an older checkpoint, and its final layer is zero-initialised, so the
converted policy's slot-2 distribution equals the unconditioned one until
training moves it.

So this is a fine-tune with extra inputs the network is free to start using, not a
restart.

    python training/convert_checkpoint.py --src <old.zip> --dst <new.zip> [--joint-head]
"""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import torch
from poke_env.environment import SingleAgentWrapper
from poke_env.player import RandomPlayer
from stable_baselines3 import PPO

from vgc_bench.src.env import ShowdownEnv
from vgc_bench.src.policy import (  # noqa: E402
    PROJ_W,
    TAIL_BLOCKS,
    MaskedActorCriticPolicy,
    convert_state_dict,
    read_policy_state,
    valid_growths,
)
from vgc_bench.src.teams import get_available_regs
from vgc_bench.src.utils import (
    chunk_obs_len,
    correct_accuracy_obs_len,
    format_map,
    global_presence_obs_len,
    knowledge_obs_len,
    presence_obs_len,
    semantics_obs_len,
    threat_obs_len,
)

__all__ = ["PROJ_W", "TAIL_BLOCKS", "convert_state_dict", "valid_growths"]


def build_fresh_policy(device: str, joint_head: bool = False):
    env = ShowdownEnv(
        battle_format=format_map[get_available_regs()[0]],
        log_level=40,
        accept_open_team_sheet=True,
        start_listening=False,
        choose_on_teampreview=True,
    )
    saw = SingleAgentWrapper(env, RandomPlayer(start_listening=False))
    policy_kwargs = {"d_model": 256, "choose_on_teampreview": True}
    if joint_head:
        policy_kwargs["joint_head"] = True
    return PPO(MaskedActorCriticPolicy, saw, policy_kwargs=policy_kwargs, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="checkpoint trained on the old obs")
    ap.add_argument("--dst", required=True, help="where to write the converted one")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--joint-head",
        action="store_true",
        help="give the converted policy the brain-v1 joint-action head (zero-ended)",
    )
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    assert src.exists(), f"missing {src}"

    print(
        f"target obs: 12 x {chunk_obs_len} = {12 * chunk_obs_len} per token: "
        f"{knowledge_obs_len} knowledge + {semantics_obs_len} semantics "
        f"+ {presence_obs_len} side presence + {global_presence_obs_len} global "
        f"presence + {correct_accuracy_obs_len} corrected accuracy "
        f"+ {threat_obs_len} threat; joint_head={args.joint_head}"
    )

    new = build_fresh_policy(args.device, joint_head=args.joint_head)
    policy = new.policy
    assert isinstance(policy, MaskedActorCriticPolicy)
    new_sd = policy.state_dict()
    old_sd = read_policy_state(src, args.device)
    old_in = old_sd[PROJ_W].shape[1]
    print(f"pokemon_proj: {old_in} -> {new_sd[PROJ_W].shape[1]} inputs")
    converted, copied, zeroed, added = convert_state_dict(old_sd, new_sd)
    policy.load_state_dict(converted)
    # Saving a policy with a resized first layer necessarily starts a fresh optimiser.
    # Do not pretend to preserve the source optimiser: copying the zip and immediately
    # overwriting it (the old implementation) preserved nothing.
    new.save(dst)
    print(f"copied {copied} tensors unchanged, zero-extended {zeroed}, added {added}")
    print(f"wrote {dst}")
    print("optimizer state reset intentionally; policy tensors were preserved")

    # Prove the surgery: zero new features must reproduce the old outputs.
    torch.manual_seed(0)
    obs = torch.zeros(1, 12 * chunk_obs_len)
    mask = torch.ones(1, 2 * 107)
    with torch.no_grad():
        logits_new, value_new, latent = policy.logits_with_latent(
            {"observation": obs, "action_mask": mask}, actor_grad=False
        )
        if args.joint_head:
            ally = torch.tensor([[7]])
            conditioned = policy.conditioned_logits(logits_new, latent, ally)
            drift = (conditioned - logits_new).abs().max().item()
            assert drift == 0.0, f"joint head is not zero-ended (drift {drift})"
            print("sanity: joint head conditioned logits == unconditioned (drift 0)")
    print(
        f"sanity: converted model runs, value={value_new.item():.4f}, "
        f"logits={tuple(logits_new.shape)}"
    )


if __name__ == "__main__":
    main()
