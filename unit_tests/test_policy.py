"""Unit tests for vgc_bench.src.policy action map and masking logic."""

import numpy as np
import torch
from poke_env.battle import Pokemon

from vgc_bench.src.policy import MaskedActorCriticPolicy, act_len, action_map
from vgc_bench.src.policy_player import PolicyPlayer


class TestActionMap:
    def test_length(self):
        assert len(action_map) == act_len

    def test_first_is_pass(self):
        assert action_map[0] == "pass"

    def test_switches(self):
        for i in range(1, 7):
            assert action_map[i] == f"switch {i}"


class TestUpdateMask:
    def test_ally_switch_blocks_same_switch(self):
        mask = torch.ones(1, 2 * act_len)
        ally_action = torch.tensor([[3]])  # switch 3
        updated = MaskedActorCriticPolicy._update_mask(mask, ally_action)
        # Second half should have switch 3 blocked
        assert updated[0, act_len + 3] == 0

    def test_ally_tera_blocks_all_tera(self):
        mask = torch.ones(1, 2 * act_len)
        ally_action = torch.tensor([[87]])  # a tera action (87 < x <= 106)
        updated = MaskedActorCriticPolicy._update_mask(mask, ally_action)
        # All tera actions (indices 87-106) should be blocked in second half
        for i in range(87, 107):
            assert updated[0, act_len + i] == 0

    def test_ally_pass_blocks_pass_when_not_forced(self):
        mask = torch.ones(1, 2 * act_len)
        ally_action = torch.tensor([[0]])  # pass
        updated = MaskedActorCriticPolicy._update_mask(mask, ally_action)
        # Pass should be blocked for second mon when ally voluntarily passed
        assert updated[0, act_len + 0] == 0

    def test_ally_force_pass_does_not_block_pass(self):
        # Force pass scenario: only pass is available (mask sum = 1, pass = 1)
        mask = torch.zeros(1, 2 * act_len)
        mask[0, 0] = 1  # only pass available for first mon
        mask[0, act_len:] = 1  # all available for second mon
        ally_action = torch.tensor([[0]])
        updated = MaskedActorCriticPolicy._update_mask(mask, ally_action)
        # Pass should NOT be blocked because ally was forced to pass
        assert updated[0, act_len + 0] == 1

    def test_batch_processing(self):
        batch = 4
        mask = torch.ones(batch, 2 * act_len)
        ally_actions = torch.tensor([[1], [2], [3], [87]])
        updated = MaskedActorCriticPolicy._update_mask(mask, ally_actions)
        assert updated.shape == (batch, 2 * act_len)


def test_embed_own_pokemon_completes_temporarily_missing_stats():
    pokemon = Pokemon(gen=9, species="garchomp")
    pokemon._selected_in_teampreview = True

    embedded = PolicyPlayer.embed_pokemon(
        pokemon,
        0,
        from_opponent=False,
        active_a=False,
        active_b=False,
    )

    assert np.isfinite(embedded).all()
    assert all(value is not None for value in pokemon.stats.values())
