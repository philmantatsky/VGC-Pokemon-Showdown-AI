"""threat_first2: threat_first, aware of Mega formes, with Fake Out as an answer.

Two gaps found by replaying threat_first and fake_out_threat over our ladder
decisions (2026-10-03, results_analysis/threat_first_20261003/):
* On turn 1 a foe holding its Mega stone Mega Evolves before anyone moves, but the
  guard judged it as its base forme: Raichu's Zap Cannon "could not" knock out our
  Mega Blastoise, while Mega Raichu Y (No Guard, 160 base Special Attack) did, three
  times. Here a foe that can still Mega Evolve -- its side has not, its revealed item
  is the stone or, unrevealed, its most-used sets carry the stone -- threatens with
  the stronger of its two formes, and our knockout of it must hold against both.
* A Pokemon on its first turn can answer a faster threat with Fake Out. When the top
  pair does not answer the threat, a pair that keeps the partner's action and gives
  the same Pokemon a landing Fake Out into the threat (fake_out_threat.flinches; the
  threat's move below +3 priority) answers it too -- replacing one of our plain
  attacks (not one aimed at a threatening foe) or a Fake Out aimed at a harmless foe.
Everything else is threat_first: act only when it saves the Pokemon, only ranked
pairs, partner and Mega/Tera choice unchanged. Opt-in until it passes its A/B.
"""

from __future__ import annotations

import re
from collections.abc import Generator
from contextlib import contextmanager

from poke_env.battle import DoubleBattle, Move, MoveCategory, Pokemon
from poke_env.data import GenData, to_id_str

from vgc_bench.src import guards as G
from vgc_bench.src import threat_first as T
from vgc_bench.src import vgc_knowledge as K
from vgc_bench.src.fake_out_threat import flinches

NAME = "threat_first2"
STONE_SHARE = 0.5  # an unrevealed item: the stone if this share of sets carry it


def likely_mega(battle: DoubleBattle, foe: Pokemon) -> str | None:
    """The Mega forme (species id) the foe will become this turn, if likely."""
    if getattr(battle, "_opponent_used_mega_evolve", False):
        return None
    root = to_id_str(foe.species or "")
    if not root or "mega" in root:
        return None
    dex = GenData.from_gen(9).pokedex
    megas = {
        key: to_id_str(entry.get("requiredItem") or "")
        for key, entry in dex.items()
        if key.startswith(root + "mega") and entry.get("requiredItem")
    }
    if not megas:
        return None
    item = G._norm(foe.item)
    if item and item != "unknownitem":
        return next((key for key, stone in megas.items() if stone == item), None)
    from vgc_bench.src.playbook import set_share

    tag = re.match(r"battle-([a-z0-9]+)-", getattr(battle, "battle_tag", "") or "")
    fmt = battle.format or (tag.group(1) if tag else None)
    share, key = max(
        (set_share(foe.species, item=stone, formatid=fmt), key)
        for key, stone in megas.items()
    )
    return key if share >= STONE_SHARE else None


@contextmanager
def as_mega(
    battle: DoubleBattle, foe: Pokemon, species: str
) -> Generator[Pokemon | None]:
    """Temporarily put the foe's Mega forme in its place (restored on exit)."""
    key = K.identifier(battle, foe)
    if key is None or key not in battle.opponent_team:
        yield None
        return
    mega = Pokemon(gen=9, species=species)
    entry = GenData.from_gen(9).pokedex[species]
    mega._ability = to_id_str(entry["abilities"]["0"])
    mega._level = foe._level  # a fresh Pokemon is level 100
    mega._item = foe._item
    mega._current_hp = foe._current_hp
    mega._max_hp = foe._max_hp
    mega._status = foe._status
    mega._boosts = dict(foe._boosts)
    mega._moves = foe._moves
    K.ensure_stats(mega)
    battle.opponent_team[key] = mega
    try:
        yield mega
    finally:
        battle.opponent_team[key] = foe


def _forms(battle: DoubleBattle, foe: Pokemon) -> list[str | None]:
    mega = likely_mega(battle, foe)
    return [None, mega] if mega else [None]


def threats(battle: DoubleBattle) -> list[tuple[float, Pokemon, int, Move]]:
    """threat_first.threats with each foe's likely Mega forme counted."""
    out = []
    for foe in battle.opponent_active_pokemon:
        if foe is None or foe.fainted or foe.status in T._CANNOT_ACT:
            continue
        K.ensure_stats(foe)
        moves = T._threat_moves(battle, foe)
        best: dict[int, tuple[float, Move]] = {}
        for form in _forms(battle, foe):
            if form is None:
                _score(battle, foe, foe, moves, best)
            else:
                with as_mega(battle, foe, form) as mega:
                    if mega is not None:
                        _score(battle, foe, mega, moves, best)
        for pos, (chance, move) in best.items():
            if chance >= T.THREAT_KO:
                out.append((chance, foe, pos, move))
    return sorted(out, key=lambda t: -t[0])


def _score(battle, foe, attacker, moves, best) -> None:
    for pos, me in enumerate(battle.active_pokemon):
        if me is None or me.fainted:
            continue
        hp = float(me.current_hp_fraction or 0.0)
        for move in moves:
            if move.category == MoveCategory.STATUS or not move.base_power:
                continue
            priority = T._priority(battle, attacker, move)
            if priority > 0 and G._priority_is_blocked(battle, attacker, me, priority):
                continue
            if move.id in G.FIRST_TURN_ONLY and not foe.first_turn:
                continue
            fraction = K.damage_fraction(battle, attacker, me, move)
            if fraction is None:
                continue
            accuracy = move.accuracy if isinstance(move.accuracy, float) else 1.0
            if G._norm(attacker.ability) == "noguard":
                accuracy = 1.0
            chance = T._ko_chance(fraction, hp) * min(accuracy, 1.0)
            if pos not in best or chance > best[pos][0]:
                best[pos] = (chance, move)


def knocks_out_first(
    battle: DoubleBattle, pos: int, order, foe: Pokemon, foe_move: Move
) -> bool:
    """threat_first.knocks_out_first against every likely forme of the foe."""
    if not T.knocks_out_first(battle, pos, order, foe, foe_move):
        return False
    for form in _forms(battle, foe)[1:]:
        with as_mega(battle, foe, form) as mega:  # type: ignore[arg-type]
            if mega is None:
                return False
            # the Mega's Speed is already inside threat_first's bounds
            if not T.knocks_out_first(battle, pos, order, mega, foe_move):
                return False
    return True


def _fake_out_into(battle: DoubleBattle, order, pos: int, foe: Pokemon) -> bool:
    move, hit = G._move_and_targets(battle, order, pos)
    return move is not None and move.id == "fakeout" and any(f is foe for f in hit)


def _answers(battle: DoubleBattle, pos: int, order, threat) -> bool:
    _, foe, _, foe_move = threat
    attacker = battle.active_pokemon[pos]
    if attacker is None:
        return False
    if _fake_out_into(battle, order, pos, foe):
        return T._priority(battle, foe, foe_move) < 3 and flinches(
            battle, attacker, foe
        )
    return knocks_out_first(battle, pos, order, foe, foe_move)


def _handled(battle: DoubleBattle, top: G.Candidate, threat) -> bool:
    _, _, victim_pos, _ = threat
    victim_order = G._decode(battle, top.actions[victim_pos], victim_pos)
    victim_move = getattr(victim_order, "order", None)
    if not isinstance(victim_move, Move) or victim_move.id in G.PROTECT_MOVES:
        return True
    return any(
        _answers(battle, pos, G._decode(battle, top.actions[pos], pos), threat)
        for pos in (0, 1)
    )


def guard_threat_first2(battle, cands, report) -> list[G.Candidate]:
    """Promote the attack or Fake Out that answers a knockout threat."""
    live = [candidate for candidate in cands if candidate.demoted_by is None]
    if len(live) < 2:
        return cands
    top = live[0]
    try:
        found = threats(battle)
        threatening = {id(t[1]) for t in found}
        open_threats = [t for t in found if not _handled(battle, top, t)]
    except Exception:
        report.demotions[f"{NAME}_error"] += 1
        return cands
    for threat in open_threats:
        _, foe, victim_pos, _ = threat
        if any(t[2] == victim_pos and t[1] is not foe for t in open_threats):
            continue  # another foe still knocks the same Pokemon out
        for pos in (0, 1):
            attacker = battle.active_pokemon[pos]
            if attacker is None or attacker.fainted:
                continue
            order = G._decode(battle, top.actions[pos], pos)
            move = getattr(order, "order", None)
            if not isinstance(move, Move):
                continue
            stray_fake_out = move.id == "fakeout" and not any(
                _fake_out_into(battle, order, pos, f)
                for f in battle.opponent_active_pokemon
                if f is not None and id(f) in threatening
            )
            plain = (
                G._plain_attack(move)
                and move.id not in G.FIRST_TURN_ONLY
                and not T._single_target_on(battle, order, pos, threatening)
            )
            if not (plain or stray_fake_out):
                continue
            for candidate in live[1:]:
                if candidate.actions[1 - pos] != top.actions[1 - pos]:
                    continue
                alternative = G._decode(battle, candidate.actions[pos], pos)
                if G._gimmicks(alternative) != G._gimmicks(order):
                    continue
                try:
                    good = _answers(battle, pos, alternative, threat)
                except Exception:
                    report.demotions[f"{NAME}_error"] += 1
                    return cands
                if not good:
                    continue
                candidate.prob = max(candidate.prob, top.prob)  # see resisted_target
                report.demotions[f"{NAME}:promoted"] += 1
                return G._promote_candidate(cands, candidate, NAME, report)
    return cands
