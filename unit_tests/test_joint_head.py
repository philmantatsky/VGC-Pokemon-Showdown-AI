"""Brain v1 joint-action head (2026-09-10).

Slot 2 is conditioned on slot 1's CHOSEN action. The head ends in a
zero-initialised layer, so a converted checkpoint plays bit-identically until
training moves it; after that, slot 2's distribution genuinely depends on slot
1's choice while slot 1's own distribution is untouched.
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from training.convert_checkpoint import PROJ_W, convert_state_dict, valid_growths
from vgc_bench.src import guards as G
from vgc_bench.src.policy import MaskedActorCriticPolicy, act_len, upgrade_policy
from vgc_bench.src.utils import chunk_obs_len, moves, threat_obs_len

OBS_SPACE = spaces.Dict(
    {
        "observation": spaces.Box(
            -1, len(moves), shape=(12 * chunk_obs_len,), dtype=np.float32
        ),
        "action_mask": spaces.Box(0, 1, shape=(2 * act_len,), dtype=np.int64),
    }
)
ACT_SPACE = spaces.MultiDiscrete([act_len, act_len])


def _policy(joint: bool, seed: int = 0) -> MaskedActorCriticPolicy:
    torch.manual_seed(seed)
    return MaskedActorCriticPolicy(
        OBS_SPACE,
        ACT_SPACE,
        lambda _: 3e-4,
        d_model=32,
        choose_on_teampreview=True,
        joint_head=joint,
    )


def _pair() -> tuple[MaskedActorCriticPolicy, MaskedActorCriticPolicy]:
    """A plain policy and a joint-head policy sharing every common tensor."""
    plain, joint = _policy(False), _policy(True, seed=1)
    merged = joint.state_dict()
    for key, value in plain.state_dict().items():
        merged[key] = value
    joint.load_state_dict(merged)
    return plain, joint


def _obs(batch: int = 3, seed: int = 7) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {
        "observation": torch.rand(batch, 12 * chunk_obs_len) * 0.5,
        "action_mask": torch.ones(batch, 2 * act_len),
    }


def _perturb(joint: MaskedActorCriticPolicy) -> None:
    last = joint.joint_cond[-1]
    assert isinstance(last, nn.Linear)
    with torch.no_grad():
        last.weight.normal_(0, 0.5)
        last.bias.normal_(0, 0.5)


def test_head_parameters_are_in_the_optimiser() -> None:
    joint = _policy(True)
    in_optimiser = sum(
        p.numel() for group in joint.optimizer.param_groups for p in group["params"]
    )
    assert in_optimiser == sum(p.numel() for p in joint.parameters())
    assert any(name.startswith("joint_cond") for name, _ in joint.named_parameters())


def test_zero_ended_head_is_the_identity() -> None:
    plain, joint = _pair()
    obs = _obs()
    with torch.no_grad():
        a_plain, v_plain, lp_plain = plain.forward(obs, deterministic=True)
        a_joint, v_joint, lp_joint = joint.forward(obs, deterministic=True)
    assert torch.equal(a_plain, a_joint)
    assert torch.allclose(v_plain, v_joint)
    assert torch.allclose(lp_plain, lp_joint)
    single = {k: v[:1] for k, v in obs.items()}
    plain_cands, _ = G.build_candidates(plain, single, single["action_mask"])
    joint_cands, _ = G.build_candidates(joint, single, single["action_mask"])
    assert [(c.actions, round(c.prob, 6)) for c in plain_cands] == [
        (c.actions, round(c.prob, 6)) for c in joint_cands
    ]


def test_trained_head_conditions_slot_two_only() -> None:
    plain, joint = _pair()
    _perturb(joint)
    obs = _obs()
    with torch.no_grad():
        logits, _value, latent = joint.logits_with_latent(obs, actor_grad=False)
        first = (
            joint.get_dist_from_logits(logits, obs["action_mask"])
            .distribution[0]
            .probs.clone()
        )
        first_plain = (
            plain.get_dist_from_logits(logits, obs["action_mask"])
            .distribution[0]
            .probs.clone()
        )
        ally_a = torch.full((3, 1), 7)
        ally_b = torch.full((3, 1), 12)
        # SB3's proba_distribution reuses ONE distribution object per policy, so
        # read each result out before asking for the next one.
        second_a = (
            joint.get_dist_from_logits(logits, obs["action_mask"], ally_a, latent)
            .distribution[1]
            .probs.clone()
        )
        second_b = (
            joint.get_dist_from_logits(logits, obs["action_mask"], ally_b, latent)
            .distribution[1]
            .probs.clone()
        )
    assert torch.allclose(first, first_plain)
    assert not torch.allclose(second_a, second_b)
    # the live candidate builder sees the conditional too
    single = {k: v[:1] for k, v in obs.items()}
    plain_cands, _ = G.build_candidates(plain, single, single["action_mask"])
    joint_cands, _ = G.build_candidates(joint, single, single["action_mask"])
    plain_probs = {c.actions: c.prob for c in plain_cands}
    joint_probs = {c.actions: c.prob for c in joint_cands}
    shared = set(plain_probs) & set(joint_probs)
    assert shared
    assert any(abs(plain_probs[a] - joint_probs[a]) > 1e-6 for a in shared)


def test_conditioned_distribution_requires_the_latent() -> None:
    joint = _policy(True)
    obs = _obs(batch=1)
    with torch.no_grad():
        logits, _value = joint.get_logits(obs, actor_grad=False)
    try:
        joint.get_dist_from_logits(logits, obs["action_mask"], torch.tensor([[7]]))
    except ValueError as exc:
        assert "latent" in str(exc)
    else:  # pragma: no cover - the silent fallback is the bug this guards
        raise AssertionError(
            "a joint-head policy conditioned slot 2 without the latent"
        )


def test_evaluate_actions_matches_forward_log_prob() -> None:
    _plain, joint = _pair()
    _perturb(joint)
    obs = _obs()
    with torch.no_grad():
        actions, _values, log_prob = joint.forward(obs, deterministic=False)
        _values2, log_prob2, entropy = joint.evaluate_actions(obs, actions)
    assert torch.allclose(log_prob, log_prob2, atol=1e-5)
    assert torch.isfinite(entropy).all()


def test_convert_state_dict_extends_projection_and_adds_head() -> None:
    plain, joint = _pair()
    # an "older" checkpoint: same weights, token shorter by the threat block,
    # no joint head
    old_sd = {}
    proj_keys = []
    for key, value in plain.state_dict().items():
        if key.endswith("pokemon_proj.weight"):
            old_sd[key] = value[:, :-threat_obs_len].clone()
            proj_keys.append(key)
        else:
            old_sd[key] = value.clone()
    converted, copied, zeroed, added = convert_state_dict(old_sd, joint.state_dict())
    assert zeroed == len(proj_keys) and zeroed >= 1
    assert added == sum(1 for k in joint.state_dict() if k not in old_sd) and added > 0
    assert copied == len(old_sd) - zeroed
    for key in proj_keys:
        assert torch.equal(converted[key][:, :-threat_obs_len], old_sd[key])
        assert torch.count_nonzero(converted[key][:, -threat_obs_len:]) == 0
    fresh = _policy(True, seed=5)
    fresh.load_state_dict(converted)
    # zero the plain policy's trailing columns too, then equal outputs on any obs
    # whose threat features are zero prove the leading columns survived intact
    with torch.no_grad():
        for key in proj_keys:
            plain.state_dict()[key][:, -threat_obs_len:] = 0
    obs = _obs(batch=2)
    view = obs["observation"].view(2, 12, chunk_obs_len)
    view[:, :, -threat_obs_len:] = 0
    with torch.no_grad():
        a_old, v_old, lp_old = plain.forward(obs, deterministic=True)
        a_new, v_new, lp_new = fresh.forward(obs, deterministic=True)
    assert torch.equal(a_old, a_new)
    assert torch.allclose(v_old, v_new)
    assert torch.allclose(lp_old, lp_new)


def test_convert_state_dict_refuses_mid_token_growth() -> None:
    plain, joint = _pair()
    old_sd = {
        k: (v[:, :-3].clone() if k.endswith("pokemon_proj.weight") else v.clone())
        for k, v in plain.state_dict().items()
    }
    try:
        convert_state_dict(old_sd, joint.state_dict())
    except RuntimeError as exc:
        assert "tail" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a +3 growth is not a known block and must be refused")


def test_valid_growths_cover_the_threat_block_and_the_full_history() -> None:
    growths = valid_growths()
    assert threat_obs_len in growths
    assert 0 in growths
    assert max(growths) > threat_obs_len
    assert PROJ_W.endswith("pokemon_proj.weight")


def test_upgrade_policy_brings_an_older_policy_to_the_current_layout() -> None:
    """A policy whose pokemon_proj is shorter (older token) is rebuilt in memory
    with its weights in the leading columns; play on zero-tail states is
    unchanged and the joint head can be added on the way."""
    plain = _policy(False)
    # shrink the projection as an older checkpoint would have it
    proj_keys = [k for k in plain.state_dict() if k.endswith("pokemon_proj.weight")]
    # SB3 aliases features_extractor to pi_features_extractor: collect the
    # distinct modules first, then shrink each exactly once
    targets: dict[int, tuple[str, nn.Linear]] = {}
    for key in proj_keys:
        module_path, _ = key.rsplit(".", 1)
        module = plain.get_submodule(module_path)
        assert isinstance(module, nn.Linear)
        targets.setdefault(id(module), (module_path, module))
    for module_path, module in targets.values():
        shorter = nn.Linear(module.in_features - threat_obs_len, module.out_features)
        with torch.no_grad():
            shorter.weight.copy_(module.weight[:, :-threat_obs_len])
            shorter.bias.copy_(module.bias)
        parent = plain.get_submodule(module_path.rsplit(".", 1)[0])
        setattr(parent, module_path.rsplit(".", 1)[1], shorter)
    same, changed = upgrade_policy(_policy(False))
    assert not changed and same is not None
    upgraded, changed = upgrade_policy(plain, joint_head=True)
    assert changed and upgraded.joint_head
    obs = _obs(batch=2)
    view = obs["observation"].view(2, 12, chunk_obs_len)
    view[:, :, -threat_obs_len:] = 0
    short_obs = {
        "observation": view[:, :, :-threat_obs_len].reshape(2, -1).clone(),
        "action_mask": obs["action_mask"],
    }
    with torch.no_grad():
        a_old, v_old, lp_old = plain.forward(short_obs, deterministic=True)
        a_new, v_new, lp_new = upgraded.forward(obs, deterministic=True)
    assert torch.equal(a_old, a_new)
    assert torch.allclose(v_old, v_new)
    assert torch.allclose(lp_old, lp_new)
