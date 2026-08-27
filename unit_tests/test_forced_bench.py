"""Stage-E.3: the generic bring-selection (forced bench) preview mechanism.

Masks the named species' team slots out of both preview picks so the policy
drafts around them -- the causal test for observational bring/bench splits
(Garchomp is merely this team's first test case). Team-agnostic: species are
resolved against whatever roster the battle carries, and the mechanism stands
down rather than break preview when the roster lacks the species or benching
would leave fewer than four picks.
"""

from types import SimpleNamespace

import numpy as np

from vgc_bench.src.policy_player import PolicyPlayer
from vgc_bench.src.utils import act_len


def _battle(species: list[str]):
    team = {f"p1: {name}": SimpleNamespace(base_species=name) for name in species}
    return SimpleNamespace(team=team, teampreview=True)


def _player(bench: tuple[str, ...]):
    player = object.__new__(PolicyPlayer)  # skip Player.__init__ (no websocket)
    player.forced_bench_species = tuple(bench)
    return player


ROSTER = [
    "floetteeternal",
    "charizard",
    "whimsicott",
    "garchomp",
    "basculegion",
    "kingambit",
]


class TestForcedBench:
    def test_masks_both_heads_for_named_species(self):
        player = _player(("garchomp",))
        mask = np.ones(2 * act_len, dtype=np.float32)
        adjusted = player._apply_forced_bench(_battle(ROSTER), mask)
        slot = ROSTER.index("garchomp") + 1
        assert adjusted[slot] == 0
        assert adjusted[act_len + slot] == 0
        # everything else untouched
        assert adjusted.sum() == mask.sum() - 2
        assert mask[slot] == 1  # original not mutated

    def test_missing_species_stands_down(self):
        player = _player(("notonroster",))
        mask = np.ones(2 * act_len, dtype=np.float32)
        adjusted = player._apply_forced_bench(_battle(ROSTER), mask)
        assert (adjusted == mask).all()

    def test_over_restriction_stands_down(self):
        player = _player(("garchomp", "charizard", "whimsicott"))
        mask = np.ones(2 * act_len, dtype=np.float32)
        adjusted = player._apply_forced_bench(_battle(ROSTER), mask)
        assert (adjusted == mask).all()  # 6 - 3 < 4 picks

    def test_team_agnostic_on_other_rosters(self):
        other = [
            "farigiraf",
            "torkoal",
            "hatterene",
            "indeedee",
            "ursaluna",
            "gothitelle",
        ]
        player = _player(("hatterene",))
        mask = np.ones(2 * act_len, dtype=np.float32)
        adjusted = player._apply_forced_bench(_battle(other), mask)
        slot = other.index("hatterene") + 1
        assert adjusted[slot] == 0 and adjusted[act_len + slot] == 0
