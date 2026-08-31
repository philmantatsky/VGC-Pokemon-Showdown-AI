"""Regression tests for observation features that previously encoded false facts."""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from types import SimpleNamespace

import numpy as np
from poke_env.battle import Field, Move, Weather

from vgc_bench.src.move_semantics import ABILITY_GROUPS, ability_semantics
from vgc_bench.src.policy_player import PolicyPlayer
from vgc_bench.src.utils import correct_accuracy_obs_len, global_presence_obs_len

battle = SimpleNamespace(weather={Weather.RAINDANCE: 8}, fields={Field.TRICK_ROOM: 8})
presence = PolicyPlayer.embed_global_presence(battle)
assert presence.shape == (global_presence_obs_len,)
assert presence[list(Weather).index(Weather.RAINDANCE)] == 1.0
field_start = len(Weather)
assert presence[field_start + list(Field).index(Field.TRICK_ROOM)] == 1.0

thunder = Move("thunder", gen=9)
mon = SimpleNamespace(moves={"thunder": thunder})
accuracies = PolicyPlayer.embed_move_accuracies(mon, from_opponent=False)
assert accuracies.shape == (correct_accuracy_obs_len,)
assert np.isclose(accuracies[0], 0.7), accuracies
# The old column remains in place solely so old weights do not shift.
assert np.isclose(PolicyPlayer.embed_move(thunder)[1], 0.007)

snow_group = next(i for i, group in enumerate(ABILITY_GROUPS) if "snowwarning" in group)
speed_group = next(i for i, group in enumerate(ABILITY_GROUPS) if "slushrush" in group)
assert snow_group != speed_group
assert ability_semantics("slushrush")[speed_group] == 1.0
assert ability_semantics("slushrush")[snow_group] == 0.0

print("PASS - weather/field presence, accuracy, and Slush Rush semantics repaired")
