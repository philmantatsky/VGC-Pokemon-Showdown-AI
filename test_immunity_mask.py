"""Verify the immunity mask blocks Ghost-at-Normal, the bug seen on ladder."""

import numpy as np
from poke_env.battle import Move, Pokemon
from poke_env.battle.move import MoveSet

from vgc_bench.src.policy_player import PolicyPlayer
from vgc_bench.src.utils import act_len


class FakeBattle:
    """Minimal stand-in exposing only what mask_immune_actions reads."""

    def __init__(self, attacker, foes):
        self._a = attacker
        self._f = foes

    @property
    def active_pokemon(self):
        return [self._a, None]

    @property
    def opponent_active_pokemon(self):
        return self._f


attacker = Pokemon(gen=9, species="gholdengo")
# slot 0 move = Shadow Ball (Ghost), slot 1 = Make It Rain (Steel)
attacker._moves = MoveSet(
    {"shadowball": Move("shadowball", gen=9), "makeitrain": Move("makeitrain", gen=9)}
)
snorlax = Pokemon(gen=9, species="snorlax")  # pure Normal -> immune to Ghost
garchomp = Pokemon(gen=9, species="garchomp")  # Dragon/Ground -> takes Ghost fine

battle = FakeBattle(attacker, [snorlax, garchomp])
mask = np.ones(2 * act_len, dtype=np.int64)
out = PolicyPlayer.mask_immune_actions(battle, mask)


def idx(move_i, target):
    return 7 + move_i * 5 + (target + 2)


print(f"Shadow Ball -> Snorlax  (Normal, immune): allowed={bool(out[idx(0, 1)])}")
print(f"Shadow Ball -> Garchomp (not immune)    : allowed={bool(out[idx(0, 2)])}")
print(f"Make It Rain -> Snorlax (not immune)    : allowed={bool(out[idx(1, 1)])}")
print(f"total actions masked off: {int((mask - out).sum())}")

assert not out[idx(0, 1)], "FAIL: Ghost at Normal still allowed"
assert out[idx(0, 2)], "FAIL: wrongly blocked a legal Ghost target"
assert out[idx(1, 1)], "FAIL: wrongly blocked a legal Steel move"

# Showdown auto-retargets a move aimed at a fainted/empty slot when only one foe is
# left. The old mask inspected the empty slot and allowed Shadow Ball to redirect
# into Normal/Psychic Farigiraf, reproducing the Turn 10 ladder failure.
farigiraf = Pokemon(gen=9, species="farigiraf")
retargeted = PolicyPlayer.mask_immune_actions(
    FakeBattle(attacker, [None, farigiraf]), mask
)
assert not retargeted[idx(0, 1)], "FAIL: auto-retargeted Ghost immunity was missed"
assert not retargeted[idx(0, 2)], "FAIL: direct Ghost target at Farigiraf was missed"
print("\nPASS")
