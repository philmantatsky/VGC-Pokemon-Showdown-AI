"""playbook_opening: play the chosen plan card's turn-1 script (opt-in guard).

The handoff from the playbook planner (vgc_bench/src/playbook.py) to the battle:
at team preview the player attaches the chosen card to the battle
(``battle._vgc_playbook``: the card's audit dict), and on turn 1 this guard puts
the card's script on top of the policy's ranking --

* ``fake_out``: that Pokemon Fake Outs their Trick Room or Tailwind setter if one
  leads (a revealed move, else the species' Reg M-C set rate >= 0.5), otherwise
  the lead that threatens our Trick Room setter most (the damage calculator over
  the foe's revealed or most-used moves) -- among the foes Fake Out can reach
  (not a Ghost, not grounded under Psychic Terrain, no Armor Tail / Dazzling /
  Queenly Majesty on their side); none reachable -> no Fake Out step;
* ``setter``: that Pokemon uses the named move (Trick Room);
* ``status``: that Pokemon uses the named status move on the named kind of target;
* ``mega``: that Pokemon Mega Evolves with its scripted move.

It runs FIRST, so every factual veto after it still applies: a Fake Out blocked by
Psychic Terrain, Armor Tail or a Ghost type is demoted and the policy's own pick
comes back. It never acts after turn 1, when Trick Room is already up, or when the
card's Pokemon did not lead. A ``script_only`` card (PolicyPlayer
playbook_script_only: our usual preview chose the four and leads) is played only
when both of the card's leads are out. A scripted pair the policy already ranked
is promoted with the top pick's weight; otherwise it is built from the script
plus the policy's own choice for any unscripted slot.
"""

from __future__ import annotations

import os

from poke_env.battle import (
    DoubleBattle,
    Field,
    Move,
    MoveCategory,
    Pokemon,
    PokemonType,
)
from poke_env.data import to_id_str

from vgc_bench.src import guards as G
from vgc_bench.src import vgc_knowledge as K
from vgc_bench.src.playbook import set_share
from vgc_bench.src.preview_rules import species_trick_room_rate

NAME = "playbook_opening"
SETTER_RATE = 0.5
MEGA_OFFSET = 20  # the Mega band repeats the move actions 20 higher


def _species(mon: Pokemon | None) -> str:
    if mon is None:
        return ""
    return to_id_str(getattr(mon, "base_species", "") or mon.species)


def _setter_rate(battle: DoubleBattle, foe: Pokemon) -> float:
    """How likely this lead sets Trick Room (weighted 1.0) or Tailwind (0.9)."""
    if G._known_move(foe, "trickroom") is not None:
        return 1.0
    if G._known_move(foe, "tailwind") is not None:
        return 0.9
    tr = species_trick_room_rate(foe.species, battle.format)
    tw = set_share(_species(foe), move="tailwind", formatid=battle.format)
    return max(tr, 0.9 * tw)


def _likely_moves(battle: DoubleBattle, foe: Pokemon) -> list[Move]:
    """Revealed moves, else the moves of the species' most-used set."""
    known = [m for m in (foe.moves or {}).values() if isinstance(m, Move)]
    if known:
        return known
    from vgc_bench.src.playbook import _sets

    entry = _sets(battle.format).get(_species(foe)) or {}
    top = max(entry.get("sets", []), key=lambda s: s.get("prob", 0), default=None)
    moves = []
    for move_id in (top or {}).get("moves", []):
        try:
            moves.append(Move(to_id_str(move_id), gen=9))
        except Exception:
            continue
    return moves


def _threat_to(battle: DoubleBattle, foe: Pokemon, target: Pokemon) -> float:
    best = 0.0
    for move in _likely_moves(battle, foe):
        if move.category == MoveCategory.STATUS or float(move.base_power or 0) <= 0:
            continue
        fraction = K.damage_fraction(battle, foe, target, move)
        if fraction is not None:
            best = max(best, (fraction[0] + fraction[1]) / 2)
    return best


def fake_out_reachable(
    battle: DoubleBattle, foe: Pokemon, attacker: Pokemon | None = None
) -> bool:
    """Whether our Fake Out can flinch this foe at all (2026-09-27: many Trick Room
    setters cannot be Faked Out). Not a Ghost (immune to a Normal move unless we
    have Scrappy / Mind's Eye), not grounded under Psychic Terrain, and no Armor
    Tail / Dazzling / Queenly Majesty on their side -- known, or near-certain by
    usage, the same facts and threshold as the priority_block guard."""
    ability = G._norm(getattr(attacker, "ability", None))
    if PokemonType.GHOST in foe.types and ability not in ("scrappy", "mindseye"):
        return False
    if Field.PSYCHIC_TERRAIN in getattr(battle, "fields", {}) and G._is_grounded(
        battle, foe
    ):
        return False
    foes = [
        f for f in battle.opponent_active_pokemon if f is not None and not f.fainted
    ]
    if any(G._norm(f.ability) in G.PRIORITY_BLOCK_ABILITIES for f in foes):
        return False
    try:
        prior = G._priority_block_prior_probability(battle)
    except (AttributeError, TypeError):
        prior = 0.0
    return prior < float(os.environ.get("VGC_PRIORITY_BLOCK_PRIOR_THRESHOLD", "0.99"))


def fake_out_target(
    battle: DoubleBattle, protect: Pokemon | None, attacker: Pokemon | None = None
) -> int | None:
    """Slot (1 or 2) of the foe the script Fake Outs, or None when Fake Out can
    reach neither foe (that Pokemon then keeps the policy's own action)."""
    foes = [
        (slot, foe)
        for slot, foe in enumerate(battle.opponent_active_pokemon, start=1)
        if foe is not None
        and not foe.fainted
        and fake_out_reachable(battle, foe, attacker)
    ]
    if not foes:
        return None
    setters = [(slot, _setter_rate(battle, foe)) for slot, foe in foes]
    slot, rate = max(setters, key=lambda item: item[1])
    if rate >= SETTER_RATE:
        return slot
    if protect is None:
        return foes[0][0]
    return max(foes, key=lambda item: _threat_to(battle, item[1], protect))[0]


def trick_room_setter_target(battle: DoubleBattle) -> int | None:
    best = None
    for slot, foe in enumerate(battle.opponent_active_pokemon, start=1):
        if foe is None or foe.fainted:
            continue
        rate = _setter_rate(battle, foe)
        if rate >= SETTER_RATE and (best is None or rate > best[1]):
            best = (slot, rate)
    return best[0] if best else None


def _legal(battle: DoubleBattle, action: int, pos: int) -> bool:
    """Whether the live request accepts this action (a scripted pair is built,
    not drawn from the masked policy). Positions built without a request (test
    fixtures) list no available moves and are not checked."""
    try:
        if not battle.available_moves[pos]:
            return True
        valid = {str(order) for order in battle.valid_orders[pos]}
    except (AttributeError, IndexError, TypeError):
        return True
    order = G._decode(battle, action, pos)
    return order is not None and str(order) in valid


def _can_mega(battle: DoubleBattle, pos: int) -> bool:
    """Whether the live request allows this Pokemon to Mega Evolve (positions
    built without a request -- test fixtures -- are not checked)."""
    try:
        if not battle.available_moves[pos]:
            return True
        return bool(battle.can_mega_evolve[pos])
    except (AttributeError, IndexError, TypeError):
        return True


def _find_action(
    battle: DoubleBattle, pos: int, move_id: str, target: int | None, mega: bool
) -> int | None:
    """The action index for (move, target) in the plain or Mega band; a move
    with no scripted target uses target 0, the code Showdown expects for it. A
    Mega the request does not allow is dropped; an illegal move is not scripted."""
    wanted = 0 if target is None else target
    for action in range(7, 27):
        order = G._decode(battle, action, pos)
        move = getattr(order, "order", None)
        if not isinstance(move, Move) or move.id != move_id:
            continue
        if (getattr(order, "move_target", 0) or 0) != wanted:
            continue
        if mega and _legal(battle, action + MEGA_OFFSET, pos):
            return action + MEGA_OFFSET
        return action if _legal(battle, action, pos) else None
    return None


def _matches(
    battle: DoubleBattle, action: int, pos: int, move_id: str, target: int | None
) -> bool:
    order = G._decode(battle, action, pos)
    move = getattr(order, "order", None)
    if not isinstance(move, Move) or move.id != move_id:
        return False
    return target is None or getattr(order, "move_target", None) == target


def scripted_steps(
    battle: DoubleBattle, plan: dict
) -> dict[int, tuple[str, int | None]]:
    """{our position: (move id, target slot or None)} for this battle's turn 1."""
    script = plan.get("turn1") or {}
    ours = {
        _species(mon): pos
        for pos, mon in enumerate(battle.active_pokemon)
        if mon is not None and not mon.fainted
    }
    steps: dict[int, tuple[str, int | None]] = {}
    setter = script.get("setter") or {}
    setter_pos = ours.get(setter.get("user", ""))
    fake_out = script.get("fake_out") or {}
    if fake_out.get("user") in ours:
        protect = battle.active_pokemon[setter_pos] if setter_pos is not None else None
        user_pos = ours[fake_out["user"]]
        target = fake_out_target(battle, protect, battle.active_pokemon[user_pos])
        if target is not None:
            steps[user_pos] = ("fakeout", target)
    if setter_pos is not None and setter.get("move"):
        steps[setter_pos] = (to_id_str(setter["move"]), None)
    status = script.get("status") or {}
    if status.get("user") in ours and status.get("move"):
        target = (
            trick_room_setter_target(battle)
            if status.get("target") == "trick_room_setter"
            else None
        )
        if status.get("target") != "trick_room_setter" or target is not None:
            steps[ours[status["user"]]] = (to_id_str(status["move"]), target)
    return steps


def guard_playbook_opening(battle, cands, report) -> list[G.Candidate]:
    """Put the chosen card's turn-1 script on top (see the module doc)."""
    plan = getattr(battle, "_vgc_playbook", None)
    if not plan or battle.turn != 1 or Field.TRICK_ROOM in battle.fields:
        return cands
    live = [c for c in cands if c.demoted_by is None]
    if not live:
        return cands
    ours = {
        _species(mon): pos
        for pos, mon in enumerate(battle.active_pokemon)
        if mon is not None
    }
    if plan.get("script_only") and not {
        to_id_str(s) for s in plan.get("lead") or ()
    } <= set(ours):
        return cands  # our own preview chose other leads: not the card's opening
    steps = scripted_steps(battle, plan)
    if not steps:
        return cands
    mega_pos = ours.get((plan.get("turn1") or {}).get("mega", ""))
    # the card's Mega is part of the script: a ranked pair without it is not the
    # script (ladder 2026-09-27, playbook trial game 12 promoted a non-Mega Fake Out)
    wants_mega = mega_pos in steps and _can_mega(battle, mega_pos)
    for candidate in live:
        if wants_mega and not 27 <= candidate.actions[mega_pos] <= 46:
            continue
        if all(
            _matches(battle, candidate.actions[pos], pos, move, target)
            for pos, (move, target) in steps.items()
        ):
            if candidate is live[0]:
                return cands
            candidate.prob = max(candidate.prob, live[0].prob)
            report.demotions[f"{NAME}:promoted"] += 1
            return G._promote_candidate(cands, candidate, NAME, report)
    actions = list(live[0].actions)
    for pos, (move, target) in steps.items():
        action = _find_action(battle, pos, move, target, mega=pos == mega_pos)
        if action is None:
            report.demotions[f"{NAME}:unscriptable"] += 1
            return cands
        actions[pos] = action
    if mega_pos is not None:  # one Mega per turn: the scripted one
        for pos in range(2):
            if pos != mega_pos and 27 <= actions[pos] <= 46:
                actions[pos] -= MEGA_OFFSET
    scripted = G.Candidate(tuple(actions), live[0].prob)
    report.demotions[f"{NAME}:injected"] += 1
    report.note(NAME)
    return [scripted] + list(cands)
