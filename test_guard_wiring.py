"""Both decision paths must actually reach the guard stack.

The ladder path (PolicyPlayer.choose_move) and the eval path
(BatchPolicyPlayer._choose_move) build their arguments independently, and they
disagreed: one passed the numpy action mask, the other the tensor. build_candidates
needs the tensor, so on ladder EVERY decision raised and _guarded_action silently fell
back to plain sampling -- 465 errors in 465 decisions. Nothing caught it because the
eval path was the only one under test, and the failure is a silent fallback.

So this test asserts the property that matters -- the guards RUN -- on both paths.
"""

import inspect

import numpy as np
import torch

from vgc_bench.src.policy_player import PolicyPlayer

fails = []


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: got {got!r}, want {want!r}")
    if not ok:
        fails.append(label)


print("1. both call sites hand _guarded_action something usable")
src_plain = inspect.getsource(PolicyPlayer.choose_move)
check(
    "ladder path does NOT pass the bare numpy mask",
    "_guarded_action(battle, obs_dict, mask)" in src_plain,
    False,
)

print("2. _guarded_action normalises whatever it is given")


class StubPolicy:
    device = torch.device("cpu")

    def get_logits(self, obs_dict, actor_grad=False):
        return torch.zeros(1, 2, 107), torch.zeros(1)

    def get_dist_from_logits(self, logits, mask, prev=None):
        assert isinstance(mask, torch.Tensor), "mask reached the policy as non-tensor"
        probs = mask.float()[0]
        probs = probs / probs.sum()
        d = type("D", (), {})()
        d.distribution = [type("P", (), {"probs": probs.unsqueeze(0)})()] * 2
        return d

    def forward(self, obs_dict, deterministic=False):
        return torch.zeros(1, 2, dtype=torch.long), None, None


player = PolicyPlayer.__new__(PolicyPlayer)
player.policy = StubPolicy()
player.deterministic = True

obs_dict = {"observation": torch.zeros(1, 8), "action_mask": torch.ones(1, 214)}

prev_search, prev_guards = PolicyPlayer.use_search, PolicyPlayer.use_knowledge_guards
PolicyPlayer.use_search = False
PolicyPlayer.use_knowledge_guards = True
try:
    for label, mask in [
        ("tensor (eval path)", torch.ones(1, 214)),
        ("numpy 1-D (the ladder bug)", np.ones(214, dtype=np.float32)),
        ("numpy 2-D", np.ones((1, 214), dtype=np.float32)),
    ]:
        PolicyPlayer.guard_fire_counts.clear()
        # battle=None: the guards raise internally and are counted, but the point is
        # that build_candidates got far enough to be CALLED with a real tensor.
        player._guarded_action(None, obs_dict, mask)
        errs = [
            k
            for k in PolicyPlayer.guard_fire_counts
            if k.startswith(("error:AttributeError", "error:TypeError"))
        ]
        check(f"{label} -> mask reached the policy as a tensor", errs, [])
finally:
    PolicyPlayer.use_search = prev_search
    PolicyPlayer.use_knowledge_guards = prev_guards
    PolicyPlayer.guard_fire_counts.clear()

print("3. failure counters name the exception, not just 'error'")
src = inspect.getsource(PolicyPlayer._guarded_action)
check("guard errors are typed", 'f"error:{type(exc).__name__}"' in src, True)
check("search errors are typed", "search_error:{type(exc).__name__}" in src, True)

print()
if fails:
    raise SystemExit(f"FAILED {len(fails)}: {fails}")
print("PASS - both decision paths reach the guard stack")
