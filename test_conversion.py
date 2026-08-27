"""Prove the converted checkpoint is behaviourally identical to the original.

The whole "we don't lose 9M steps" claim rests on this. An old observation, padded
per-token with zero knowledge features, must produce the same logits and value through
the converted model as the original observation did through the original model.

Per-token matters: the observation is 12 tokens laid end to end, so the padding goes
at the end of EACH token (578 -> 602), not at the end of the flat vector.

The column layout matters just as much, and is NOT "chunk then embeddings".
AttentionExtractor.forward (policy.py:274) replaces the 6 embedding IDs at chunk
indices [57:63] with their looked-up vectors, so the projection input is

    chunk[0:57] | 6 x embed_len embedding values | chunk[63:C]

The embedding block EXPANDS IN THE MIDDLE, which pushes every later chunk feature
right by 6*(embed_len-1) = 186 columns. Tail-of-chunk still maps to tail-of-matrix,
so the zero-extend is sound -- but a test that feeds zeros into the embedding columns
cannot tell a correct splice from one that shifted the embedding block, because zero
input makes every weight arrangement look identical. Check 3 therefore drives real
values through the true layout, and check 4 confirms the test can actually fail.
"""

import io
import zipfile

import numpy as np
import torch

from vgc_bench.src.policy import AttentionExtractor
from vgc_bench.src.utils import (
    chunk_obs_len,
    correct_accuracy_obs_len,
    glob_obs_len,
    global_presence_obs_len,
    side_obs_len,
)

EMBED = AttentionExtractor.embed_len
START = glob_obs_len + side_obs_len  # first embedding ID within a chunk
N_TOKENS = 12

OLD_CKPT = "results_knowfix/saves_bc_sp/reg_mb/64_teams/seed1/3932160.zip"
NEW_CKPT = "results_repaired/converted_v4.zip"
OLD_CHUNK = chunk_obs_len - global_presence_obs_len - correct_accuracy_obs_len


def load_sd(path):
    with zipfile.ZipFile(path) as zf:
        with zf.open("policy.pth") as f:
            return torch.load(
                io.BytesIO(f.read()), map_location="cpu", weights_only=False
            )


old_sd, new_sd = load_sd(OLD_CKPT), load_sd(NEW_CKPT)

# 1. every non-proj tensor must be bit-identical
diffs = []
for k, v in old_sd.items():
    if k not in new_sd:
        diffs.append(f"{k} missing from converted")
        continue
    if k.endswith("pokemon_proj.weight"):
        continue
    if not torch.equal(v, new_sd[k]):
        diffs.append(f"{k} CHANGED")
print(f"1. tensors compared: {len(old_sd)}   changed (excluding proj): {len(diffs)}")
for d in diffs[:5]:
    print("   ", d)
assert not diffs, "conversion altered weights it should not have"

# 2. proj: leading columns preserved, new columns exactly zero
for k in [k for k in new_sd if k.endswith("pokemon_proj.weight")]:
    o, n = old_sd[k], new_sd[k]
    assert torch.equal(n[:, : o.shape[1]], o), f"{k}: old columns not preserved"
    assert torch.count_nonzero(n[:, o.shape[1] :]) == 0, f"{k}: new columns not zero"
print(
    f"2. all {len([k for k in new_sd if k.endswith('pokemon_proj.weight')])} "
    f"pokemon_proj tensors: old columns preserved, new columns zero"
)

# 3. behavioural equivalence on random observations, through the TRUE column layout
PAD = global_presence_obs_len + correct_accuracy_obs_len
PROJ_W = "features_extractor.pokemon_proj.weight"
PROJ_B = "features_extractor.pokemon_proj.bias"


def expand(chunk, embed_vals):
    """Reproduce policy.py:274 -- swap the 6 IDs for their embedding vectors."""
    return np.concatenate(
        [
            chunk[:, :START],
            embed_vals.reshape(chunk.shape[0], 6 * EMBED),
            chunk[:, START + 6 :],
        ],
        axis=1,
    )


def proj(w, b, x):
    return torch.from_numpy(x) @ w.T + b


rng = np.random.default_rng(0)
max_abs = 0.0
for _ in range(20):
    old_chunk = rng.standard_normal((N_TOKENS, OLD_CHUNK)).astype(np.float32)
    new_chunk = np.concatenate(
        [old_chunk, np.zeros((N_TOKENS, PAD), dtype=np.float32)], axis=1
    )
    # Identical embedding lookups on both sides: only the projection may differ.
    embed_vals = rng.standard_normal((N_TOKENS, 6, EMBED)).astype(np.float32)
    a = proj(old_sd[PROJ_W], old_sd[PROJ_B], expand(old_chunk, embed_vals))
    c = proj(new_sd[PROJ_W], new_sd[PROJ_B], expand(new_chunk, embed_vals))
    max_abs = max(max_abs, (a - c).abs().max().item())

print(
    f"3. projection output over 20 random observations (real embedding values "
    f"in cols[{START}:{START + 6 * EMBED}]): max |diff| = {max_abs:.2e}"
)
assert max_abs == 0.0, "converted projection is not bit-identical"

# 4. negative control: the check above must REJECT a mis-spliced matrix. Without
# this, a test that silently exercises nothing still prints PASS -- which is exactly
# how the all-zero version of check 3 gave false assurance.
old_chunk = rng.standard_normal((N_TOKENS, OLD_CHUNK)).astype(np.float32)
new_chunk = np.concatenate(
    [old_chunk, np.zeros((N_TOKENS, PAD), dtype=np.float32)], axis=1
)
embed_vals = rng.standard_normal((N_TOKENS, 6, EMBED)).astype(np.float32)
good = new_sd[PROJ_W]
caught = 0
for shift in (
    global_presence_obs_len,
    correct_accuracy_obs_len,
    global_presence_obs_len + correct_accuracy_obs_len,
):
    # Embedding block landing `shift` columns away from where forward() puts it.
    bad = torch.roll(good, shifts=shift, dims=1)
    d = (
        (
            proj(old_sd[PROJ_W], old_sd[PROJ_B], expand(old_chunk, embed_vals))
            - proj(bad, new_sd[PROJ_B], expand(new_chunk, embed_vals))
        )
        .abs()
        .max()
        .item()
    )
    caught += d > 0
    print(
        f"4. mis-splice by {shift:3d} columns -> max |diff| = {d:.2e}  "
        f"{'REJECTED' if d > 0 else 'MISSED'}"
    )
assert caught == 3, "check 3 cannot detect a shifted embedding block; it proves nothing"

print(
    "\nPASS - converted checkpoint is behaviourally identical with zero knowledge, "
    "and the check demonstrably rejects a mis-splice"
)
