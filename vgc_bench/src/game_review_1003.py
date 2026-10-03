"""Three guards from the user's review of a challenge game (2026-10-03, open team
sheets on both sides, vs Mega Raichu Y / Sylveon / Volcarona / Rillaboom;
challenge_replays_mc_deployed_T6tac_guards12, battle 2692336128).

wasted_fake_out -- turn 2: Incineroar Faked Out a Raichu that had fallen asleep the
turn before, beside a Sylveon that had to recharge after Hyper Beam. Neither foe
could act: a Pokemon that has just fallen asleep cannot wake on its next move
attempt (sleep lasts at least two attempts; Champions 2-3, poke-env counts the
failed ones in ``status_counter``) and a recharging one skips its turn. The user:
"this fakeout made no sense since raichu was already sleeping so we couldve just
used flare blitz". When every foe the top pair's Fake Out hits cannot act this
turn, the best-ranked pair that keeps the partner's action and gives the same
Pokemon its strongest plain attack (calculator-scored, any foe) is promoted.

throat_chop_main_threat -- turn 6: Incineroar, slower than both foes, Throat
Chopped a Volcarona whose sheet shows Bug Buzz, while its Heat Wave was the threat;
Flare Blitz hit harder. dominated_throat_chop keeps Throat Chop for its two-turn
sound block whenever the target has any sound move. Here the block counts only
when a sound move is the target's strongest attack on our side, or it carries
Parting Shot / Perish Song; otherwise Throat Chop is compared as a plain attack
(dominated_attack's scoring and margins, the policy's target kept).

hp_move_after_hits -- turn 7: Torkoal, slower than both foes, used Eruption at full
HP, took a Heat Wave first and hit at half power; the user: "if we heat waved we
wouldve won ... we gotta predict our opponent will damage us". Eruption and Water
Spout scale with the user's HP when it moves. For each foe that moves first (a
certain order counts fully, an uncertain one half), its strongest attack on our
Pokemon is expected to land -- a spread attack fully, a single-target one half,
since it may aim at our partner -- and the HP move is re-scored at the HP left,
against the same Pokemon's other attacks in the ranked pairs (same partner action
and Mega/Tera choice); the best is promoted at dominated_attack's margins.

All three promote only ranked pairs (always legal). Opt-in: not in HARD_GUARDS
until they pass their A/B.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from poke_env.battle import DoubleBattle, Field, Move, MoveCategory, Pokemon, Status
from poke_env.battle.target import Target

from vgc_bench.src import guards as G
from vgc_bench.src import threat_first as T
from vgc_bench.src import vgc_knowledge as K
from vgc_bench.src.tempo_reranker import effective_speed

HP_MOVES = frozenset({"eruption", "waterspout", "dragonenergy"})
KEY_SOUND_STATUS = frozenset({"partingshot", "perishsong"})
SPREAD = (Target.ALL_ADJACENT_FOES, Target.ALL_ADJACENT)


# ---------------------------------------------------------------- wasted_fake_out


def cannot_act(battle: DoubleBattle, foe: Pokemon | None) -> bool:
    """The foe certainly takes no action this turn: it must recharge, or it fell
    asleep and has not yet failed a move attempt (no sleep ends on the first
    attempt; Early Bird and Sleep Talk excepted)."""
    if foe is None or foe.fainted:
        return False
    if getattr(foe, "must_recharge", False):
        return True
    if foe.status != Status.SLP:
        return False
    if int(getattr(foe, "status_counter", 0) or 0) > 0:
        return False
    if "earlybird" in {G._norm(a) for a in (foe.ability, *foe.possible_abilities)}:
        return False
    return "sleeptalk" not in {m.id for m in _known_moves(battle, foe)}


def guard_wasted_fake_out(battle, cands, report) -> list[G.Candidate]:
    """Attack instead of Faking Out foes that cannot act this turn."""
    name = "wasted_fake_out"
    live = [candidate for candidate in cands if candidate.demoted_by is None]
    if len(live) < 2:
        return cands
    top = live[0]
    for pos in (0, 1):
        me = battle.active_pokemon[pos]
        if me is None or me.fainted:
            continue
        order = G._decode(battle, top.actions[pos], pos)
        move, hit = G._move_and_targets(battle, order, pos)
        if move is None or move.id != "fakeout" or not hit:
            continue
        if not all(cannot_act(battle, foe) for foe in hit):
            continue
        already = G._partner_damage(battle, top.actions, pos)
        best, best_value = None, 0.0
        for candidate in live[1:]:
            if candidate.actions[1 - pos] != top.actions[1 - pos]:
                continue
            alternative = G._decode(battle, candidate.actions[pos], pos)
            alt_move, alt_hit = G._move_and_targets(battle, alternative, pos)
            if (
                alt_move is None
                or alt_move.id == "fakeout"
                or not alt_hit
                or not G._plain_attack(alt_move)
                or (getattr(alternative, "move_target", 0) or 0) < 0
                or G._gimmicks(alternative) != G._gimmicks(order)
            ):
                continue
            value = G._attack_value(battle, me, alternative, pos, already)
            if value is not None and value > best_value:
                best, best_value = candidate, value
        if best is None:
            continue
        best.prob = max(best.prob, top.prob)  # see resisted_target
        report.demotions[f"{name}:promoted"] += 1
        return G._promote_candidate(cands, best, name, report)
    return cands


# -------------------------------------------------------- throat_chop_main_threat


def _known_moves(battle: DoubleBattle, foe: Pokemon) -> list[Move]:
    """A full sheet's moves (open sheets, or all four seen), else revealed plus the
    species' most-used set."""
    seen = [m for m in (foe.moves or {}).values() if isinstance(m, Move)]
    return seen if len(seen) >= 4 else T._threat_moves(battle, foe)


def _best_damage(battle: DoubleBattle, foe: Pokemon, move: Move) -> float:
    """The move's mean damage fraction on the worse-hit of our active Pokemon."""
    best = 0.0
    for mon in battle.active_pokemon:
        if mon is None or mon.fainted:
            continue
        fraction = K.damage_fraction(battle, foe, mon, move)
        if fraction is not None:
            best = max(best, (fraction[0] + fraction[1]) / 2)
    return best


def sound_main_threat(battle: DoubleBattle, foe: Pokemon) -> bool:
    """Throat Chop's sound block matters: the foe carries Parting Shot or Perish
    Song, or a sound move is its strongest attack on our active Pokemon."""
    if not G._sound_threat(battle, foe):
        return False
    K.ensure_stats(foe)
    moves = _known_moves(battle, foe)
    if any(m.id in KEY_SOUND_STATUS for m in moves):
        return True
    sound = other = 0.0
    for move in moves:
        if move.category == MoveCategory.STATUS or not move.base_power:
            continue
        damage = _best_damage(battle, foe, move)
        if move.id in G.SOUND_MOVES:
            sound = max(sound, damage)
        else:
            other = max(other, damage)
    return sound > 0 and sound >= other


def _throat_chop_swappable(battle: DoubleBattle, move: Move, hit) -> bool:
    if G._plain_attack(move):
        return True
    return (
        move.id == G.THROAT_CHOP
        and bool(hit)
        and not any(sound_main_threat(battle, foe) for foe in hit)
    )


def guard_throat_chop_main_threat(battle, cands, report) -> list[G.Candidate]:
    """dominated_throat_chop with the block counted only against a main threat."""
    name = "throat_chop_main_threat"
    live = [candidate for candidate in cands if candidate.demoted_by is None]
    if not live:
        return cands
    top = live[0]
    best = top
    for pos in (0, 1):
        attacker = battle.active_pokemon[pos]
        if attacker is None or attacker.fainted:
            continue
        order = G._decode(battle, best.actions[pos], pos)
        move, aimed = G._move_and_targets(battle, order, pos)
        if move is None or move.id != G.THROAT_CHOP:
            continue
        if not _throat_chop_swappable(battle, move, aimed):
            continue
        already = G._partner_damage(battle, best.actions, pos)
        current = G._attack_value(battle, attacker, order, pos, already)
        if current is None:
            report.demotions[f"{name}:no_calc"] += 1
            continue
        winner, winner_value = None, current
        for candidate in live:
            if (
                candidate.actions[1 - pos] != best.actions[1 - pos]
                or candidate.actions[pos] == best.actions[pos]
            ):
                continue
            alternative = G._decode(battle, candidate.actions[pos], pos)
            alt_move, hit = G._move_and_targets(battle, alternative, pos)
            if (
                alt_move is None
                or not G._plain_attack(alt_move)
                or alt_move.target == Target.ALL_ADJACENT
                or (getattr(alternative, "move_target", 0) or 0) < 0
                or G._gimmicks(alternative) != G._gimmicks(order)
            ):
                continue
            # move choice, not target choice (as dominated_attack)
            if not all(any(foe is other for other in hit) for foe in aimed):
                continue
            value = G._attack_value(battle, attacker, alternative, pos, already)
            if value is not None and value > winner_value:
                winner, winner_value = candidate, value
        if (
            winner is not None
            and winner_value >= current * G.DOMINATED_ATTACK_RATIO
            and winner_value - current >= G.DOMINATED_ATTACK_MIN_GAIN
        ):
            best = winner
    if best is top:
        return cands
    best.prob = max(best.prob, top.prob)
    report.demotions[f"{name}:promoted"] += 1
    return G._promote_candidate(cands, best, name, report)


# ------------------------------------------------------------- hp_move_after_hits


def _moves_first(
    battle: DoubleBattle, me: Pokemon, my_move: Move, foe: Pokemon, foe_move: Move
) -> float:
    """1.0 if the foe's move certainly resolves before ours, 0.0 if certainly
    after, 0.5 when the speeds overlap."""
    mine, theirs = T._priority(battle, me, my_move), T._priority(battle, foe, foe_move)
    if mine != theirs:
        return 1.0 if theirs > mine else 0.0
    speed = effective_speed(battle, me, True)
    bounds = T.foe_speed_bounds(battle, foe)
    if speed is None or bounds is None:
        return 0.5
    low, high = bounds
    if Field.TRICK_ROOM in getattr(battle, "fields", {}):
        return 1.0 if high < speed else 0.0 if low > speed else 0.5
    return 1.0 if low > speed else 0.0 if high < speed else 0.5


def expected_loss(battle: DoubleBattle, me: Pokemon, my_move: Move) -> float:
    """Expected share of our max HP lost to foes that move before us this turn."""
    loss = 0.0
    for foe in battle.opponent_active_pokemon:
        if foe is None or foe.fainted or cannot_act(battle, foe):
            continue
        if foe.status in T._CANNOT_ACT:
            continue
        K.ensure_stats(foe)
        best = 0.0
        for move in _known_moves(battle, foe):
            if move.category == MoveCategory.STATUS or not move.base_power:
                continue
            first = _moves_first(battle, me, my_move, foe, move)
            if first <= 0:
                continue
            fraction = K.damage_fraction(battle, foe, me, move)
            if fraction is None:
                continue
            accuracy = move.accuracy if isinstance(move.accuracy, float) else 1.0
            aimed = 1.0 if move.target in SPREAD else 0.5
            mean = (fraction[0] + fraction[1]) / 2
            best = max(best, first * aimed * min(accuracy, 1.0) * mean)
        loss += best
    return loss


@contextmanager
def at_hp(mon: Pokemon, fraction: float) -> Generator[None]:
    """Temporarily set our Pokemon's HP to ``fraction`` of its max (restored)."""
    saved = mon._current_hp
    mon._current_hp = max(1, int(round(fraction * (mon.max_hp or 0))))
    try:
        yield
    finally:
        mon._current_hp = saved


def guard_hp_move_after_hits(battle, cands, report) -> list[G.Candidate]:
    """Score Eruption / Water Spout at the HP we will have when we move."""
    name = "hp_move_after_hits"
    live = [candidate for candidate in cands if candidate.demoted_by is None]
    if len(live) < 2:
        return cands
    top = live[0]
    for pos in (0, 1):
        me = battle.active_pokemon[pos]
        if me is None or me.fainted or not me.max_hp:
            continue
        order = G._decode(battle, top.actions[pos], pos)
        move = getattr(order, "order", None)
        if not isinstance(move, Move) or move.id not in HP_MOVES:
            continue
        try:
            loss = expected_loss(battle, me, move)
        except Exception:
            report.demotions[f"{name}_error"] += 1
            return cands
        hp_then = float(me.current_hp_fraction or 0.0) - loss
        if loss <= 0 or hp_then <= 0:
            continue  # nothing lands first, or we fall before moving anyway
        already = G._partner_damage(battle, top.actions, pos)
        with at_hp(me, hp_then):
            current = G._attack_value(battle, me, order, pos, already)
            if current is None:
                report.demotions[f"{name}:no_calc"] += 1
                continue
            winner, winner_value = None, current
            for candidate in live[1:]:
                if candidate.actions[1 - pos] != top.actions[1 - pos]:
                    continue
                alternative = G._decode(battle, candidate.actions[pos], pos)
                alt_move, hit = G._move_and_targets(battle, alternative, pos)
                if (
                    alt_move is None
                    or alt_move.id in HP_MOVES
                    or not hit
                    or not G._plain_attack(alt_move)
                    or alt_move.target == Target.ALL_ADJACENT
                    or (getattr(alternative, "move_target", 0) or 0) < 0
                    or G._gimmicks(alternative) != G._gimmicks(order)
                ):
                    continue
                value = G._attack_value(battle, me, alternative, pos, already)
                if value is not None and value > winner_value:
                    winner, winner_value = candidate, value
        if (
            winner is not None
            and winner_value >= current * G.DOMINATED_ATTACK_RATIO
            and winner_value - current >= G.DOMINATED_ATTACK_MIN_GAIN
        ):
            winner.prob = max(winner.prob, top.prob)
            report.demotions[f"{name}:promoted"] += 1
            return G._promote_candidate(cands, winner, name, report)
    return cands
