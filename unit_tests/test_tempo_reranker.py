from types import SimpleNamespace as NS

import pytest
from poke_env.battle import Field, Move, SideCondition, Weather

from vgc_bench.src.exact_planner import speed_position_value
from vgc_bench.src.guards import Candidate
from vgc_bench.src.opponent_reranker import rerank_candidates
from vgc_bench.src.opponent_tactics import MovePrediction
from vgc_bench.src.tempo_reranker import (
    acts_before,
    effective_speed,
    score_candidates,
    speed_control_snapshot,
    tailwind_turns,
    trick_room_turns,
)


def mon(
    speed: int,
    *,
    hp: float = 1.0,
    ability: str = "",
    item: str = "",
    status=None,
    last_move=None,
    moves=(),
    base_speed: int | None = None,
    possible_abilities=(),
):
    known = {move.id: move for move in moves}
    if last_move is not None:
        known[last_move.id] = last_move
    return NS(
        stats={"spe": speed},
        boosts={"spe": 0},
        status=status,
        ability=ability,
        item=item,
        effects={},
        current_hp_fraction=hp,
        fainted=False,
        first_turn=False,
        last_move=last_move,
        moves=known,
        base_stats={"spe": base_speed} if base_speed is not None else {},
        possible_abilities=list(possible_abilities),
    )


def battle(ours, foes, *, turn=5, fields=None, side=None, weather=None):
    return NS(
        gen=9,
        turn=turn,
        fields=fields or {},
        side_conditions=side or {},
        opponent_side_conditions={},
        weather=weather or {},
        active_pokemon=list(ours),
        opponent_active_pokemon=list(foes),
    )


def order(move_id: str, target: int = 0):
    return NS(order=Move(move_id, gen=9), move_target=target)


def test_tailwind_changes_speed_before_trick_room_inverts_order():
    ours = mon(100)
    foe = mon(150)
    state = battle(
        [ours, mon(90)],
        [foe, mon(80)],
        turn=2,
        fields={Field.TRICK_ROOM: 1},
        side={SideCondition.TAILWIND: 1},
    )
    assert effective_speed(state, ours, True) == 200
    assert tailwind_turns(state, True) == 3
    # Tailwind makes us faster numerically, therefore Trick Room makes us act later.
    assert (
        acts_before(state, ours, Move("splash", 9), True, foe, Move("splash", 9), False)
        is False
    )


def test_sand_rush_is_included_in_effective_speed():
    excadrill = mon(140, ability="Sand Rush")
    state = battle(
        [mon(100), mon(90)], [excadrill, mon(80)], weather={Weather.SANDSTORM: 1}
    )
    assert effective_speed(state, excadrill, False) == 280


def test_leaf_speed_value_inverts_only_when_trick_room_is_active():
    ours = mon(150)
    foe = mon(100)
    normal = battle([ours, mon(140)], [foe, mon(90)])
    room = battle(
        [ours, mon(140)], [foe, mon(90)], fields={Field.TRICK_ROOM: 1}
    )
    assert speed_position_value(normal) == pytest.approx(1.0)
    assert speed_position_value(room) == pytest.approx(-1.0)


def test_hidden_speed_range_stands_down_when_order_can_flip():
    ours = mon(130)
    hidden_foe = mon(122, item="", base_speed=100)
    state = battle([ours, mon(90)], [hidden_foe, mon(80)])
    assert (
        acts_before(
            state, ours, Move("splash", 9), True, hidden_foe, Move("splash", 9), False
        )
        is None
    )


def test_turn_five_good_trick_room_penalizes_double_protect():
    # Healthy Sand Rush Excadrill is faster than both of ours outside room. The very
    # low Tyranitar is slower, but its 5% HP makes it the less important comparison.
    state = battle(
        [mon(100), mon(150)],
        [mon(50, hp=0.05), mon(180, ability="Sand Rush")],
        fields={Field.TRICK_ROOM: 1},
        weather={Weather.SANDSTORM: 1},
    )
    snapshot = speed_control_snapshot(state)
    assert trick_room_turns(state) == 1
    assert snapshot.trick_room_advantage > 0.8

    double_protect = Candidate(
        (1, 1), 0.55, orders=(order("protect"), order("protect"))
    )
    keep_attacking = Candidate(
        (2, 2), 0.50, orders=(order("dazzlinggleam"), order("dragonclaw", 1))
    )
    ranked, report = rerank_candidates(
        state,
        [double_protect, keep_attacking],
        None,
        None,
        use_opponent=False,
        use_tempo=True,
    )
    assert ranked[0] is keep_attacking
    assert report is not None
    assert dict(report.tempo_factors_before)["double_protect_in_good_room"] < 0


def test_protect_plus_earthquake_is_recognized_as_coordinated_progress():
    state = battle(
        [mon(150), mon(140)], [mon(50), mon(60)], turn=4, fields={Field.TRICK_ROOM: 1}
    )
    immediate_damage = Candidate(
        (1, 1), 0.55, orders=(order("dazzlinggleam"), order("dragonclaw", 1))
    )
    protect_earthquake = Candidate(
        (2, 2), 0.50, orders=(order("protect"), order("earthquake"))
    )
    scores = score_candidates(state, [protect_earthquake], None, None)
    factors = scores.factors[protect_earthquake.actions]
    assert factors["protected_spread_in_bad_room"] > 0
    assert factors["protect_stalls_bad_room"] > 0

    ranked, _ = rerank_candidates(
        state,
        [immediate_damage, protect_earthquake],
        None,
        None,
        use_opponent=False,
        use_tempo=True,
    )
    assert ranked[0] is protect_earthquake


def test_encore_does_not_credit_stale_trick_room_after_faster_rage_powder():
    trick_room = Move("trickroom", 9)
    whimsicott = mon(150, ability="Prankster")
    sinistcha = mon(50, last_move=trick_room, moves=(Move("ragepowder", 9),))
    state = battle(
        [whimsicott, mon(140)],
        [sinistcha, mon(60)],
        turn=2,
        fields={Field.TRICK_ROOM: 1},
    )
    candidate = Candidate(
        (1, 1), 1.0, orders=(order("encore", 1), order("moonblast", 1))
    )
    rage_powder = (
        MovePrediction((("ragepowder", 1.0),), (), (), reliability=1.0),
        MovePrediction((("rockslide", 1.0),), (), (), reliability=1.0),
    )
    score = score_candidates(state, [candidate], rage_powder, None)
    # Rage Powder is +2; Prankster Encore is +1. The target acts first, so Encore
    # locks Rage Powder, not the previous turn's Trick Room.
    assert score.factors[candidate.actions]["encore"] == pytest.approx(0.03)


def test_encore_can_force_trick_room_toggle_when_it_really_moves_first():
    trick_room = Move("trickroom", 9)
    whimsicott = mon(150, ability="Prankster")
    sinistcha = mon(50, last_move=trick_room, moves=(Move("matchagotcha", 9),))
    state = battle(
        [whimsicott, mon(140)],
        [sinistcha, mon(60)],
        turn=2,
        fields={Field.TRICK_ROOM: 1},
    )
    candidate = Candidate(
        (1, 1), 1.0, orders=(order("encore", 1), order("moonblast", 1))
    )
    normal_attack = (
        MovePrediction((("matchagotcha", 1.0),), (), (), reliability=1.0),
        MovePrediction((("rockslide", 1.0),), (), (), reliability=1.0),
    )
    score = score_candidates(state, [candidate], normal_attack, None)
    assert score.snapshot.trick_room_advantage == pytest.approx(-1.0)
    assert score.factors[candidate.actions]["encore"] == pytest.approx(0.85)
