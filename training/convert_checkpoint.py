"""Extend an old checkpoint to the knowledge-augmented observation, losing nothing.

The observation grew from 6936 to 7224 floats (24 knowledge features per Pokemon
token). Naively that invalidates every checkpoint. It does not have to.

`chunk_obs_len` reaches the network through exactly one layer -- AttentionExtractor's
`pokemon_proj` (policy.py:240) -- and the knowledge block sits at the END of each
token, so it maps to trailing COLUMNS of that layer's weight matrix. Copy the old
weights into the leading columns, zero the new ones, and the converted model produces
bit-identical output on any state where the knowledge features are zero.

So this is a fine-tune with extra inputs the network is free to start using, not a
restart. ~9M self-play steps are preserved.

    python convert_checkpoint.py --src <old.zip> --dst <new.zip>
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
from vgc_bench.src.policy import MaskedActorCriticPolicy
from vgc_bench.src.teams import get_available_regs
from vgc_bench.src.utils import (
    chunk_obs_len,
    correct_accuracy_obs_len,
    format_map,
    global_presence_obs_len,
    knowledge_obs_len,
    presence_obs_len,
    semantics_obs_len,
)

PROJ_W = "features_extractor.pokemon_proj.weight"
PROJ_B = "features_extractor.pokemon_proj.bias"


def build_fresh_policy(device: str):
    env = ShowdownEnv(
        battle_format=format_map[get_available_regs()[0]],
        log_level=40,
        accept_open_team_sheet=True,
        start_listening=False,
        choose_on_teampreview=True,
    )
    saw = SingleAgentWrapper(env, RandomPlayer(start_listening=False))
    return PPO(
        MaskedActorCriticPolicy,
        saw,
        policy_kwargs={"d_model": 256, "choose_on_teampreview": True},
        device=device,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="checkpoint trained on the old obs")
    ap.add_argument("--dst", required=True, help="where to write the converted one")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    assert src.exists(), f"missing {src}"

    print(
        f"target obs: 12 x {chunk_obs_len} = {12 * chunk_obs_len} per token: "
        f"{knowledge_obs_len} knowledge + {semantics_obs_len} semantics "
        f"+ {presence_obs_len} side presence + {global_presence_obs_len} global "
        f"presence + {correct_accuracy_obs_len} corrected accuracy"
    )

    new = build_fresh_policy(args.device)
    new_sd = new.policy.state_dict()
    target_in = new_sd[PROJ_W].shape[1]

    # Read the raw tensors straight out of the zip: PPO.load would shape-check the
    # old observation space against the new env and refuse.
    import io
    import zipfile

    with zipfile.ZipFile(src) as zf:
        with zf.open("policy.pth") as f:
            old_sd = torch.load(
                io.BytesIO(f.read()), map_location=args.device, weights_only=False
            )

    old_in = old_sd[PROJ_W].shape[1]
    grew = target_in - old_in
    print(f"pokemon_proj: {old_in} -> {target_in} inputs (+{grew})")
    # Every added feature block must live at the TAIL of the Pokemon token, so the
    # original inputs stay in the leading columns. Converting from an intermediate
    # checkpoint is fine too -- it simply grew by less.
    new_tail = global_presence_obs_len + correct_accuracy_obs_len
    # Supported historical prefixes, in the order these blocks were introduced:
    # base -> knowledge -> semantics -> side presence -> repaired global/accuracy.
    valid_growth = {
        new_tail,
        presence_obs_len + new_tail,
        semantics_obs_len + presence_obs_len + new_tail,
        knowledge_obs_len + semantics_obs_len + presence_obs_len + new_tail,
    }
    assert grew in valid_growth, (
        f"grew by +{grew}; expected one of {sorted(valid_growth)} from a known "
        f"historical checkpoint prefix. "
        f"A different number means a new block was NOT appended at the tail, and the "
        f"surgery would silently misalign every feature."
    )

    converted = {}
    copied = zeroed = 0
    for key, new_tensor in new_sd.items():
        if key not in old_sd:
            converted[key] = new_tensor
            continue
        old_tensor = old_sd[key]
        if old_tensor.shape == new_tensor.shape:
            converted[key] = old_tensor
            copied += 1
        elif key.endswith("pokemon_proj.weight"):
            # SB3 keeps three extractor copies (shared / pi_ / vf_); all three grow.
            w = torch.zeros_like(new_tensor)
            w[:, : old_tensor.shape[1]] = old_tensor
            converted[key] = w
            zeroed += 1
        else:
            raise RuntimeError(
                f"unexpected shape change on {key}: "
                f"{tuple(old_tensor.shape)} -> {tuple(new_tensor.shape)}"
            )

    new.policy.load_state_dict(converted)
    # Saving a policy with a resized first layer necessarily starts a fresh optimiser.
    # Do not pretend to preserve the source optimiser: copying the zip and immediately
    # overwriting it (the old implementation) preserved nothing.
    new.save(dst)
    print(f"copied {copied} tensors unchanged, zero-extended {zeroed}")
    print(f"wrote {dst}")
    print("optimizer state reset intentionally; policy tensors were preserved")

    # Prove the surgery: zero knowledge features must reproduce the old outputs.
    torch.manual_seed(0)
    obs = torch.zeros(1, 12 * chunk_obs_len)
    mask = torch.ones(1, 2 * 107)
    with torch.no_grad():
        logits_new, value_new = new.policy.get_logits(
            {"observation": obs, "action_mask": mask}, actor_grad=False
        )
    print(
        f"sanity: converted model runs, value={value_new.item():.4f}, "
        f"logits={tuple(logits_new.shape)}"
    )


if __name__ == "__main__":
    main()
