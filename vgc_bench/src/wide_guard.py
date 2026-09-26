"""wide_guard: do not throw spread moves into a Wide Guard the opponent has shown.

Ladder 2026-09-26 (game 2 of the Throat Chop read, a loss): Aerodactyl showed
Wide Guard on turn 2; on turn 3 Mega Blastoise used Water Spout into it again,
with Farigiraf's Helping Hand, and on turn 5 Torkoal used Eruption into it, with
Helping Hand again -- every one blocked. Wide Guard (+3 priority) has no
consecutive-use penalty, and that opponent used it on 3 of its 4 turns. The
user: "it needs to be taught wide guard if its gonna use water spout and
eruption too".

Demotes pairs that use a damaging spread move (all adjacent foes, or all
adjacent) while an active foe that can act this turn -- not asleep, frozen or
Taunted -- has shown Wide Guard (used it, or it is on its open team sheet).
Stands down for a pair whose other half is a Fake Out that certainly flinches
that foe before it moves (Fake Out and Wide Guard are both +3, so our partner
must be certainly faster, or certainly slower under Trick Room), and when
nothing else is legal. Runs with the factual vetoes, before the attack guards,
so they refine the single-target pair that takes over. Lives outside guards.py
because it needs the tempo reranker's speed bounds (which import guards.py).
Opt-in: not in HARD_GUARDS until it passes its A/B.
"""

from __future__ import annotations

from poke_env.battle import (
    DoubleBattle,
    Effect,
    Field,
    Move,
    MoveCategory,
    Pokemon,
    Status,
    Target,
)

from vgc_bench.src import guards as G
from vgc_bench.src import vgc_knowledge as K
from vgc_bench.src.tempo_reranker import effective_speed_bounds
from vgc_bench.src.trick_room_guard import _fake_out_lands

NAME = "wide_guard"
SPREAD = {Target.ALL_ADJACENT_FOES, Target.ALL_ADJACENT}
_CANNOT_ACT = (Status.SLP, Status.FRZ)


def spread_attack(order: object) -> bool:
    """A damaging move that hits several Pokemon: what Wide Guard blocks."""
    move = getattr(order, "order", None)
    return (
        isinstance(move, Move)
        and move.category != MoveCategory.STATUS
        and float(move.base_power or 0) > 0
        and move.target in SPREAD
    )


def wide_guard_users(battle: DoubleBattle) -> list[Pokemon]:
    """Active foes that have shown Wide Guard and can use it this turn."""
    taunt = getattr(Effect, "TAUNT", None)
    users = []
    for foe in battle.opponent_active_pokemon:
        if foe is None or foe.fainted or foe.status in _CANNOT_ACT:
            continue
        if taunt is not None and taunt in (foe.effects or {}):
            continue
        if G._known_move(foe, "wideguard") is not None:
            users.append(foe)
    return users


def _fake_out_first(
    battle: DoubleBattle, candidate: G.Candidate, pos: int, foe: Pokemon
) -> bool:
    """The other half of the pair is a Fake Out that flinches ``foe`` before its
    Wide Guard: both are +3, so speed (reversed under Trick Room) must be sure."""
    partner_pos = 1 - pos
    partner = battle.active_pokemon[partner_pos]
    if partner is None or partner.fainted:
        return False
    order = G._decode(battle, candidate.actions[partner_pos], partner_pos)
    move = getattr(order, "order", None)
    if (
        not isinstance(move, Move)
        or move.id != "fakeout"
        or foe not in G.resolved_foe_targets(battle, order, move)
        or not _fake_out_lands(battle, partner, foe, move)
    ):
        return False
    K.ensure_stats(foe)
    ours = effective_speed_bounds(battle, partner, True)
    theirs = effective_speed_bounds(battle, foe, False)
    if ours is None or theirs is None:
        return False
    if Field.TRICK_ROOM in battle.fields:
        return ours[1] < theirs[0]
    return ours[0] > theirs[1]


def guard_wide_guard(battle, cands, report) -> list[G.Candidate]:
    """Demote spread attacks into a shown Wide Guard (see the module doc)."""
    users = wide_guard_users(battle)
    if not users:
        return cands
    dead: set[int] = set()
    for i, candidate in enumerate(cands):
        for pos, action in enumerate(candidate.actions):
            attacker = battle.active_pokemon[pos]
            if attacker is None or attacker.fainted:
                continue
            if not spread_attack(G._decode(battle, action, pos)):
                continue
            if all(_fake_out_first(battle, candidate, pos, foe) for foe in users):
                continue
            dead.add(i)
            break
    return G._demote(cands, dead, NAME, report)
