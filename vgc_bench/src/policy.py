"""
Neural network policy module for VGC-Bench.

Implements the actor-critic policy architecture with attention-based feature
extraction for Pokemon VGC battles. Uses action masking to ensure only legal
moves are selected.

Joint-action head (2026-09-10, brain v1): with ``joint_head=True`` the second
slot's logits are conditioned on the first slot's CHOSEN action. Without it the
policy emits one logit vector for both slots and slot 2 only ever sees slot 1
through the legality mask, so "both attack the same foe because only the pair
is a KO" or "spread move plus Protect" can only be marginal habits. The
conditioning path ends in a zero-initialised layer, so a converted checkpoint
plays bit-identically until training moves those weights.
"""

from pathlib import Path
from typing import Any

import torch
from gymnasium import Space
from stable_baselines3.common.distributions import MultiCategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs
from torch import nn

from vgc_bench.src.utils import (
    abilities,
    act_len,
    chunk_obs_len,
    correct_accuracy_obs_len,
    glob_obs_len,
    global_presence_obs_len,
    items,
    knowledge_obs_len,
    moves,
    presence_obs_len,
    semantics_obs_len,
    side_obs_len,
    threat_obs_len,
)

action_map = (
    ["pass", "switch 1", "switch 2", "switch 3", "switch 4", "switch 5", "switch 6"]
    + [f"move {i} target {j}" for i in range(1, 5) for j in range(-2, 3)]
    + [f"move {i} target {j} mega" for i in range(1, 5) for j in range(-2, 3)]
    + [f"move {i} target {j} zmove" for i in range(1, 5) for j in range(-2, 3)]
    + [f"move {i} target {j} dynamax" for i in range(1, 5) for j in range(-2, 3)]
    + [f"move {i} target {j} tera" for i in range(1, 5) for j in range(-2, 3)]
)

# Width of the embedding of slot 1's chosen action fed to the joint head.
JOINT_EMBED_LEN = 64


class MaskedActorCriticPolicy(ActorCriticPolicy):
    """
    Actor-critic policy with action masking for Pokemon VGC.

    Extends SB3's ActorCriticPolicy with action masking to enforce legal
    moves and uses an attention-based feature extractor for processing
    Pokemon battle observations.

    Attributes:
        choose_on_teampreview: Whether policy controls teampreview decisions.
        joint_head: Whether slot 2's logits are conditioned on slot 1's action.
        actor_grad: Whether to compute gradients for actor during evaluation.
        debug: Whether to print debug information during forward pass.
    """

    def __init__(
        self,
        *args: Any,
        d_model: int,
        choose_on_teampreview: bool,
        joint_head: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize the masked actor-critic policy.

        Args:
            d_model: Hidden size for policy/value networks and attention extractor.
            choose_on_teampreview: Whether policy controls teampreview.
            joint_head: Condition slot 2 on slot 1's chosen action (brain v1).
            *args: Additional arguments for ActorCriticPolicy.
            **kwargs: Additional keyword arguments for ActorCriticPolicy.
        """
        self.choose_on_teampreview = choose_on_teampreview
        self.joint_head = bool(joint_head)
        self._d_model = d_model
        self.actor_grad = True
        self.debug = False
        super().__init__(
            *args,
            **kwargs,
            net_arch=[],
            activation_fn=torch.nn.ReLU,
            features_extractor_class=AttentionExtractor,
            features_extractor_kwargs={
                "d_model": d_model,
                "choose_on_teampreview": choose_on_teampreview,
            },
            share_features_extractor=False,
        )

    def _build(self, lr_schedule) -> None:
        """Create the joint head BEFORE SB3 builds the optimiser over parameters().

        SB3 constructs the optimiser inside ``_build``; a module added after
        ``super().__init__`` returns would never receive a gradient step. The
        final layer starts at zero so the conditioned distribution equals the
        unconditioned one on day one (checkpoint conversion relies on this).
        """
        if self.joint_head:
            self.ally_action_embed = nn.Embedding(act_len, JOINT_EMBED_LEN)
            self.joint_cond = nn.Sequential(
                nn.Linear(self._d_model + JOINT_EMBED_LEN, self._d_model),
                nn.ReLU(),
                nn.Linear(self._d_model, act_len),
            )
            last = self.joint_cond[-1]
            assert isinstance(last, nn.Linear)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        super()._build(lr_schedule)

    def forward(self, obs: PyTorchObs, deterministic=False):
        assert isinstance(obs, dict)
        action_logits, value_logits, latent = self.logits_with_latent(
            obs, actor_grad=True
        )
        distribution = self.get_dist_from_logits(action_logits, obs["action_mask"])
        actions = distribution.get_actions(deterministic=deterministic)
        distribution2 = self.get_dist_from_logits(
            action_logits, obs["action_mask"], actions[:, :1], latent
        )
        actions2 = distribution2.get_actions(deterministic=deterministic)
        distribution.distribution[1] = distribution2.distribution[1]
        actions[:, 1] = actions2[:, 1]
        if self.debug:
            print("value:", value_logits[0][0].item())
            action_dist1 = {
                action_map[i]: f"{p.item():.3e}"
                for i, p in enumerate(distribution.distribution[0].probs[0])
                if p > 0
            }
            action_dist1 = dict(
                sorted(action_dist1.items(), key=lambda x: float(x[1]), reverse=True)
            )
            print("action1 dist:", action_dist1)
            action_dist2 = {
                action_map[i]: f"{p.item():.3e}"
                for i, p in enumerate(distribution.distribution[1].probs[0])
                if p > 0
            }
            action_dist2 = dict(
                sorted(action_dist2.items(), key=lambda x: float(x[1]), reverse=True)
            )
            print("action2 dist:", action_dist2)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, value_logits, log_prob

    def evaluate_actions(self, obs, actions):
        assert isinstance(obs, dict)
        action_logits, value_logits, latent = self.logits_with_latent(
            obs, self.actor_grad
        )
        distribution = self.get_dist_from_logits(action_logits, obs["action_mask"])
        distribution2 = self.get_dist_from_logits(
            action_logits, obs["action_mask"], actions[:, :1], latent
        )
        distribution.distribution[1] = distribution2.distribution[1]
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return value_logits, log_prob, entropy

    def logits_with_latent(
        self, obs: dict[str, torch.Tensor], actor_grad: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract features; return (action logits, value logits, actor latent).

        The actor latent is what the joint head conditions on. Callers that
        enumerate slot-2 distributions for several slot-1 choices (candidate
        building, search) must pass it back through ``get_dist_from_logits``.
        """
        actor_context = torch.enable_grad() if actor_grad else torch.no_grad()
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            with actor_context:
                latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        with actor_context:
            action_logits = self.action_net(latent_pi)
        value_logits = self.value_net(latent_vf)
        return action_logits, value_logits, latent_pi

    def get_logits(
        self, obs: dict[str, torch.Tensor], actor_grad: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract features and compute action/value logits (no latent)."""
        action_logits, value_logits, _latent = self.logits_with_latent(obs, actor_grad)
        return action_logits, value_logits

    def conditioned_logits(
        self,
        action_logits: torch.Tensor,
        latent: torch.Tensor,
        ally_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Slot-2 logits given slot 1's chosen action; identity without the head."""
        if not self.joint_head:
            return action_logits
        ally = ally_actions.reshape(-1).long()
        delta = self.joint_cond(
            torch.cat([latent, self.ally_action_embed(ally)], dim=-1)
        )
        return torch.cat(
            [action_logits[:, :act_len], action_logits[:, act_len:] + delta], dim=1
        )

    def get_dist_from_logits(
        self,
        action_logits: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor | None = None,
        latent: torch.Tensor | None = None,
    ) -> MultiCategoricalDistribution:
        """Create masked action distribution from logits.

        With ``action`` (slot 1's choice) the second half is masked for legality
        and, when the joint head is on, conditioned on that choice -- which
        needs the actor ``latent`` from ``logits_with_latent``. Asking a
        joint-head policy for a conditioned distribution without the latent is
        an error rather than a silent fallback to the unconditioned head.
        """
        if action is not None:
            mask = self._update_mask(mask, action)
            if self.joint_head:
                if latent is None:
                    raise ValueError(
                        "joint-head policy needs the actor latent to condition "
                        "slot 2; use logits_with_latent()"
                    )
                action_logits = self.conditioned_logits(action_logits, latent, action)
        mask = torch.where(mask == 1, 0, float("-inf"))
        distribution = self.action_dist.proba_distribution(action_logits + mask)
        assert isinstance(distribution, MultiCategoricalDistribution)
        return distribution

    @staticmethod
    def _update_mask(mask: torch.Tensor, ally_actions: torch.Tensor) -> torch.Tensor:
        """
        Update action mask based on ally's already-chosen action.

        Prevents illegal combinations like both Pokemon switching to the same
        slot, both passing when not forced, or both terastallizing.

        Args:
            mask: Current action mask tensor of shape (batch, 2*act_len).
            ally_actions: Ally's chosen actions of shape (batch, 1).

        Returns:
            Updated mask tensor with illegal actions disabled.
        """
        indices = (
            torch.arange(act_len, device=ally_actions.device)
            .unsqueeze(0)
            .expand(len(ally_actions), -1)
        )
        ally_passed = ally_actions == 0
        ally_force_passed = (
            (mask[:, 0] == 1) & (mask[:, :act_len].sum(1) == 1)
        ).unsqueeze(1)
        ally_switched = (1 <= ally_actions) & (ally_actions <= 6)
        ally_mega_evolved = (26 < ally_actions) & (ally_actions <= 46)
        ally_z_moved = (46 < ally_actions) & (ally_actions <= 66)
        ally_dynamaxed = (66 < ally_actions) & (ally_actions <= 86)
        ally_terastallized = (86 < ally_actions) & (ally_actions <= 106)
        updated_half = mask[:, act_len:] * ~(
            ((indices == 0) & ally_passed & ~ally_force_passed)
            | ((indices == ally_actions) & ally_switched)
            | ((26 < indices) & (indices <= 46) & ally_mega_evolved)
            | ((46 < indices) & (indices <= 66) & ally_z_moved)
            | ((66 < indices) & (indices <= 86) & ally_dynamaxed)
            | ((86 < indices) & (indices <= 106) & ally_terastallized)
        )
        return torch.cat([mask[:, :act_len], updated_half], dim=1)


class AttentionExtractor(BaseFeaturesExtractor):
    """
    Attention-based feature extractor for Pokemon battle observations.

    Processes Pokemon observations using embeddings for abilities, items, and
    moves, then applies transformer attention to produce a fixed-size feature
    vector.

    Class Attributes:
        embed_len: Dimension of embedding vectors for abilities/items/moves.
        num_heads: Number of attention heads in transformer layers.
        embed_layers: Number of transformer encoder layers.
    """

    embed_len: int = 32
    num_heads: int = 4
    embed_layers: int = 3

    def __init__(
        self, observation_space: Space[Any], d_model: int, choose_on_teampreview: bool
    ):
        """
        Initialize the attention-based feature extractor.

        Args:
            observation_space: Gymnasium observation space specification.
            d_model: Hidden size for token projection and transformer layers.
            choose_on_teampreview: Whether policy controls teampreview decisions.
        """
        super().__init__(observation_space, features_dim=d_model)
        self.choose_on_teampreview = choose_on_teampreview
        self.ability_embed = nn.Embedding(
            len(abilities), self.embed_len, max_norm=self.embed_len**0.5
        )
        self.item_embed = nn.Embedding(
            len(items), self.embed_len, max_norm=self.embed_len**0.5
        )
        self.move_embed = nn.Embedding(
            len(moves), self.embed_len, max_norm=self.embed_len**0.5
        )
        self.pokemon_proj = nn.Linear(chunk_obs_len + 6 * (self.embed_len - 1), d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pokemon_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=self.num_heads,
                dim_feedforward=d_model,
                dropout=0,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=self.embed_layers,
            enable_nested_tensor=False,
        )

    def pokemon_tokens(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the frozen per-Pokemon projections before team attention."""
        x = obs_dict["observation"]
        batch_size = x.size(0)
        pokemon_obs = x.view(batch_size, 12, -1)
        # embedding
        start = glob_obs_len + side_obs_len
        pokemon_obs = torch.cat(
            [
                pokemon_obs[:, :, :start],
                self.ability_embed(pokemon_obs[:, :, start].long()),
                self.item_embed(pokemon_obs[:, :, start + 1].long()),
                self.move_embed(pokemon_obs[:, :, start + 2].long()),
                self.move_embed(pokemon_obs[:, :, start + 3].long()),
                self.move_embed(pokemon_obs[:, :, start + 4].long()),
                self.move_embed(pokemon_obs[:, :, start + 5].long()),
                pokemon_obs[:, :, start + 6 :],
            ],
            dim=-1,
        )
        # pokemon encoder
        return self.pokemon_proj(pokemon_obs)

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract features from battle observation.

        Embeds Pokemon attributes and applies transformer attention across all
        12 Pokemon (6 per side).

        Args:
            obs_dict: Dict with an ``"observation"`` tensor of shape
                ``(batch, 12 * chunk_obs_len)``.

        Returns:
            Feature tensor of shape (batch, d_model).
        """
        pokemon_tokens = self.pokemon_tokens(obs_dict)
        batch_size = pokemon_tokens.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_token, pokemon_tokens], dim=1)
        return self.pokemon_encoder(tokens)[:, 0, :]


# ---------------------------------------------------------------------------
# Checkpoint upgrades: every feature block ever added sits at the END of the
# Pokemon token, so an older checkpoint's pokemon_proj maps onto the leading
# columns of the current one and the new columns start at zero. Upgrading in
# memory keeps every stored artifact (the deployed brain, clones, lineage
# checkpoints) loadable and bit-identical in play after the observation grows.
# ---------------------------------------------------------------------------

PROJ_W = "features_extractor.pokemon_proj.weight"

# Feature blocks in the order they were appended to the token tail. A checkpoint
# from any point in this history grew by the sum of the blocks after it.
TAIL_BLOCKS = [
    knowledge_obs_len,
    semantics_obs_len,
    presence_obs_len,
    global_presence_obs_len + correct_accuracy_obs_len,
    threat_obs_len,
]


def valid_growths() -> set[int]:
    """Every observation growth a known historical checkpoint can show."""
    return {sum(TAIL_BLOCKS[i:]) for i in range(len(TAIL_BLOCKS) + 1)}


def convert_state_dict(
    old_sd: dict[str, torch.Tensor], new_sd: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], int, int, int]:
    """Map an older policy state dict onto a fresh policy's shapes.

    Returns (converted, copied, zero_extended, added). Raises on any shape change
    that is not a tail growth of pokemon_proj, because a mid-token insertion would
    silently misalign every feature.
    """
    old_in = old_sd[PROJ_W].shape[1]
    target_in = new_sd[PROJ_W].shape[1]
    grew = target_in - old_in
    if grew not in valid_growths():
        raise RuntimeError(
            f"pokemon_proj grew by +{grew}; expected one of {sorted(valid_growths())} "
            "from a known historical checkpoint prefix. A different number means a "
            "new block was NOT appended at the tail."
        )
    converted: dict[str, torch.Tensor] = {}
    copied = zeroed = added = 0
    for key, new_tensor in new_sd.items():
        if key not in old_sd:
            # e.g. the joint head: fresh (zero-ended) modules the old policy lacked
            converted[key] = new_tensor
            added += 1
            continue
        old_tensor = old_sd[key]
        if old_tensor.shape == new_tensor.shape:
            converted[key] = old_tensor
            copied += 1
        elif key.endswith("pokemon_proj.weight"):
            # SB3 keeps three extractor copies (shared / pi_ / vf_); all three grow.
            w = torch.zeros_like(new_tensor)
            w[:, : old_tensor.shape[1]] = old_tensor.to(new_tensor.device)
            converted[key] = w
            zeroed += 1
        else:
            raise RuntimeError(
                f"unexpected shape change on {key}: "
                f"{tuple(old_tensor.shape)} -> {tuple(new_tensor.shape)}"
            )
    extra = set(old_sd) - set(new_sd)
    if extra:
        raise RuntimeError(f"source checkpoint has tensors the target lacks: {extra}")
    return converted, copied, zeroed, added


def observation_spaces() -> tuple[Space[Any], Space[Any]]:
    """The (observation, action) spaces the policy is built against."""
    import numpy as np
    from gymnasium import spaces

    obs_space = spaces.Dict(
        {
            "observation": spaces.Box(
                -1, len(moves), shape=(12 * chunk_obs_len,), dtype=np.float32
            ),
            "action_mask": spaces.Box(0, 1, shape=(2 * act_len,), dtype=np.int64),
        }
    )
    return obs_space, spaces.MultiDiscrete([act_len, act_len])


def build_policy(
    d_model: int = 256,
    choose_on_teampreview: bool = True,
    joint_head: bool = False,
    device: str | torch.device = "cpu",
) -> "MaskedActorCriticPolicy":
    """A fresh policy at the CURRENT observation length, no environment needed."""
    obs_space, act_space = observation_spaces()
    policy = MaskedActorCriticPolicy(
        obs_space,
        act_space,
        lambda _progress: 3e-4,
        d_model=d_model,
        choose_on_teampreview=choose_on_teampreview,
        joint_head=joint_head,
    )
    return policy.to(device)


def expected_proj_inputs() -> int:
    return chunk_obs_len + 6 * (AttentionExtractor.embed_len - 1)


def upgrade_policy(
    policy: "MaskedActorCriticPolicy", joint_head: bool | None = None
) -> tuple["MaskedActorCriticPolicy", bool]:
    """Return a policy at the current observation length (and head layout).

    The loaded policy is returned unchanged when it already matches. Otherwise a
    fresh policy is built, the old tensors are copied into it (tail columns of
    pokemon_proj zeroed, the joint head zero-ended), and (policy, True) is
    returned. ``joint_head`` overrides the loaded policy's head layout.
    """
    wants_head = policy.joint_head if joint_head is None else joint_head
    current_in = policy.features_extractor.pokemon_proj.in_features  # type: ignore[union-attr]
    if current_in == expected_proj_inputs() and wants_head == policy.joint_head:
        return policy, False
    fresh = build_policy(
        d_model=policy._d_model,
        choose_on_teampreview=policy.choose_on_teampreview,
        joint_head=wants_head,
        device=policy.device,
    )
    converted, _copied, _zeroed, _added = convert_state_dict(
        policy.state_dict(), fresh.state_dict()
    )
    fresh.load_state_dict(converted)
    fresh.actor_grad = policy.actor_grad
    fresh.eval() if not policy.training else fresh.train()
    return fresh, True


def read_policy_state(
    path: "str | Path", device: "str | torch.device"
) -> dict[str, torch.Tensor]:
    """The raw policy tensors of an SB3 zip, without SB3's shape checks."""
    import io
    import zipfile

    with zipfile.ZipFile(path) as zf:
        with zf.open("policy.pth") as f:
            return torch.load(
                io.BytesIO(f.read()), map_location=device, weights_only=True
            )


def load_state_dict_upgraded(
    target: nn.Module, old_sd: dict[str, torch.Tensor]
) -> tuple[int, int, int]:
    """Load ``old_sd`` (possibly from an older checkpoint) into ``target``.

    Used by the trainers' resume/init paths, which otherwise load strictly.
    Returns (copied, zero_extended, added).
    """
    converted, copied, zeroed, added = convert_state_dict(old_sd, target.state_dict())
    target.load_state_dict(converted)
    return copied, zeroed, added
