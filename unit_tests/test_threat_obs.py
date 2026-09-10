"""Brain v1 threat block (2026-09-10): what the enemies across the field can do to
THIS active Pokemon, who moves first, and whether Fake Out / priority is on.

The damage calculator and the speed model are stubbed; these tests pin the
layout and the logic that turns their answers into the eight floats.
"""

from __future__ import annotations

from types import SimpleNamespace as NS
from typing import Any, cast

import pytest
from poke_env.battle import DoubleBattle, Field, Move, Pokemon, Status

from vgc_bench.src import tempo_reranker as T
from vgc_bench.src import vgc_knowledge as K
from vgc_bench.src.utils import threat_obs_len

SPEEDS = {"sneasler": 120.0, "garchomp": 102.0, "rillaboom": 85.0, "incineroar": 60.0}
DAMAGE = {
    # (attacker species, move id) -> (min, max) fraction of the defender's HP
    ("sneasler", "closecombat"): (0.55, 0.65),
    ("sneasler", "fakeout"): (0.08, 0.10),
    ("rillaboom", "woodhammer"): (0.90, 1.05),
    ("rillaboom", "grassyglide"): (0.40, 0.48),
    ("garchomp", "earthquake"): (0.30, 0.36),
}


def _mon(species: str, hp: float = 1.0, first_turn: bool = False) -> Pokemon:
    mon = Pokemon(gen=9, species=species)
    mon._max_hp = 100
    mon._current_hp = int(round(hp * 100))
    mon._active_turns = 1 if first_turn else 2
    return mon


def _moves(*ids: str) -> list[Move]:
    return [Move(i, gen=9) for i in ids]


def _battle(fields=None, trick_room: int = 0) -> DoubleBattle:
    fake: Any = NS(
        fields=fields or {},
        weather={},
        side_conditions={},
        opponent_side_conditions={},
        turn=1,
        battle_tag="battle-test",
        _tr=trick_room,
    )
    return cast(DoubleBattle, fake)


@pytest.fixture(autouse=True)
def _stubs(monkeypatch):
    def damage(battle, attacker, defender, move):
        return DAMAGE.get((attacker.species, move.id))

    monkeypatch.setattr(K, "damage_fraction", damage)
    monkeypatch.setattr(
        T, "effective_speed", lambda battle, mon, ours: SPEEDS.get(mon.species)
    )
    monkeypatch.setattr(T, "trick_room_turns", lambda battle: battle._tr)


def test_layout_length():
    assert K.THREAT_LEN == threat_obs_len == 8


def test_incoming_damage_ko_and_speed_order():
    garchomp = _mon("garchomp", hp=0.6)
    sneasler, rillaboom = _mon("sneasler"), _mon("rillaboom")
    out = K.threat_knowledge(
        _battle(),
        garchomp,
        _moves("earthquake", "dragonclaw", "rocktomb", "protect"),
        [sneasler, rillaboom],
        [
            _moves("closecombat", "fakeout", "protect"),
            _moves("woodhammer", "grassyglide"),
        ],
        ours=True,
    )
    assert len(out) == threat_obs_len
    # best expected incoming: Close Combat 0.60 x acc 1.0; Wood Hammer 0.975
    assert out[0] == pytest.approx(0.60)
    assert out[1] == pytest.approx(0.975)
    # KO flags: Close Combat max 0.65 >= 0.6 HP -> KO; Wood Hammer -> KO
    assert out[2] == 1.0 and out[3] == 1.0
    # speed: garchomp 102 < sneasler 120 -> enemy first; > rillaboom 85 -> we first
    assert out[4] == 0.0 and out[5] == 1.0


def test_trick_room_reverses_speed_order_and_unknown_speed_is_half():
    garchomp = _mon("garchomp")
    sneasler = _mon("sneasler")
    unknown = _mon("pelipper")  # no stubbed speed
    out = K.threat_knowledge(
        _battle(trick_room=3),
        garchomp,
        _moves("earthquake"),
        [sneasler, unknown],
        [_moves("closecombat"), _moves("hurricane")],
        ours=True,
    )
    assert out[4] == 1.0  # slower moves first under Trick Room
    assert out[5] == 0.5


def test_no_ko_when_max_roll_falls_short():
    garchomp = _mon("garchomp", hp=1.0)
    sneasler = _mon("sneasler")
    out = K.threat_knowledge(
        _battle(),
        garchomp,
        _moves("earthquake"),
        [sneasler],
        [_moves("closecombat")],
        True,
    )
    assert out[0] == pytest.approx(0.60) and out[2] == 0.0
    assert out[1] == 0.0 and out[3] == 0.0  # no second enemy


def test_fake_out_only_on_the_first_turn_and_priority_flag():
    fresh = _mon("sneasler", first_turn=True)
    stale = _mon("sneasler", first_turn=False)
    kit = _moves("closecombat", "fakeout", "direclaw", "protect")
    garchomp = _mon("garchomp")
    fresh_out = K.threat_knowledge(
        _battle(), fresh, kit, [garchomp], [_moves("earthquake")], ours=False
    )
    stale_out = K.threat_knowledge(
        _battle(), stale, kit, [garchomp], [_moves("earthquake")], ours=False
    )
    assert fresh_out[6] == 1.0 and stale_out[6] == 0.0
    assert fresh_out[7] == 1.0  # Fake Out is a damaging priority move
    quiet = K.threat_knowledge(
        _battle(), stale, _moves("closecombat", "protect"), [garchomp], [[]], False
    )
    assert quiet[7] == 0.0


def test_grassy_glide_counts_as_priority_only_under_grassy_terrain():
    rillaboom = _mon("rillaboom")
    garchomp = _mon("garchomp")
    kit = _moves("grassyglide", "woodhammer")
    plain = K.threat_knowledge(_battle(), rillaboom, kit, [garchomp], [[]], False)
    terrain = K.threat_knowledge(
        _battle(fields={Field.GRASSY_TERRAIN: 1}),
        rillaboom,
        kit,
        [garchomp],
        [[]],
        False,
    )
    assert plain[7] == 0.0 and terrain[7] == 1.0


def test_fainted_and_absent_enemies_give_zeros():
    fainted = _mon("garchomp")
    fainted._status = Status.FNT
    assert (
        K.threat_knowledge(_battle(), fainted, [], [_mon("sneasler")], [[]], True)
        == [0.0] * threat_obs_len
    )
    alive = _mon("garchomp")
    nobody: list[Pokemon | None] = [None, None]
    out = K.threat_knowledge(
        _battle(), alive, _moves("earthquake"), nobody, [[], []], True
    )
    assert out[:6] == [0.0] * 6


def test_only_two_enemies_are_read():
    garchomp = _mon("garchomp")
    enemies = [_mon("sneasler"), _mon("rillaboom"), _mon("incineroar")]
    kits = [_moves("closecombat"), _moves("woodhammer"), _moves("flareblitz")]
    out = K.threat_knowledge(
        _battle(), garchomp, _moves("earthquake"), enemies, kits, True
    )
    assert out[0] == pytest.approx(0.60) and out[1] == pytest.approx(0.975)
