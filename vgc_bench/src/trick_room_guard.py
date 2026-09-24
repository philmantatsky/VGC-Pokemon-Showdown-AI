"""Trick Room direction: set, keep or reverse Trick Room by whom it helps.

The user, 2026-09-24, on ladder game 2 (our Mega Blastoise + Farigiraf against
Farigiraf + Incineroar from a Hatterene / Indeedee / Camerupt / Gallade team):
"its sometimes a good move to predict an opponents trick room and undo it
because they have a heavy trick room team ... sending out our fast pokemon but
with farafarig for the sole purpose of countering their TR". The bot set Trick
Room on turns 1 and 4 while our side was the faster one; it only worked out
because the opponent's own Trick Room reversed ours each time. Nothing decided
whether Trick Room helps us or them: the tempo reranker compares speeds only to
choose Protect while Trick Room is already up.

Rules on the policy's ranked pairs (the opt-in guard `trick_room_direction`):

* Trick Room up (>= 2 turns left) and favouring them: promote the best ranked
  pair that reverses it.
* Trick Room up and favouring us: demote pairs that would undo it.
* Trick Room down and it would favour them: if a likely setter of theirs is
  active (Trick Room revealed, or a species set rate >= 0.5), both sides using
  it this turn cancels theirs -- promote the best ranked pair that does (the
  counter the user described); with no setter active, demote pairs that would
  set it, since that only hands them the room.

"Favouring" is the threat-weighted share of the four active Pokemon's speed
comparisons each side wins under Trick Room (the tempo reranker's measure).
Against rosters with >= 2 likely setters their Pokemon are taken at minimum
Speed, as Trick Room teams run it: the plain Speed range of a hidden spread
(72-123 for a base-60 Pokemon) is too wide to order anything.
"""

from __future__ import annotations

from poke_env.battle import DoubleBattle, Field, Move, Pokemon
from poke_env.data import to_id_str

from vgc_bench.src import guards as G
from vgc_bench.src import vgc_knowledge as K
from vgc_bench.src.preview_rules import TRICK_ROOM_MOVE, species_trick_room_rate
from vgc_bench.src.tempo_reranker import effective_speed_bounds, trick_room_turns

SETTER_RATE = 0.5
HEAVY_SETTERS = 2
CLEAR_EDGE = 0.2
NAME = "trick_room_direction"


def _format(battle: DoubleBattle) -> str | None:
    """The battle's format id; test fixtures only carry it in the battle tag."""
    fmt = getattr(battle, "format", None)
    if fmt:
        return fmt
    parts = str(getattr(battle, "battle_tag", "")).split("-")
    return parts[1] if len(parts) > 2 else None


def _their_roster(battle: DoubleBattle) -> list[str]:
    preview = getattr(battle, "teampreview_opponent_team", None) or []
    mons = list(preview) or list(battle.opponent_team.values())
    return [to_id_str(mon.species) for mon in mons]


def heavy_trick_room(battle: DoubleBattle) -> bool:
    """At least two likely Trick Room setters on the opponent's roster."""
    fmt = _format(battle)
    setters = sum(
        species_trick_room_rate(species, fmt) >= SETTER_RATE
        for species in _their_roster(battle)
    )
    return setters >= HEAVY_SETTERS


def likely_setter(battle: DoubleBattle, foe: Pokemon) -> bool:
    """An active foe that sets Trick Room: revealed, or its species usually does."""
    if foe is None or foe.fainted:
        return False
    if G._known_move(foe, TRICK_ROOM_MOVE) is not None:
        return True
    return species_trick_room_rate(foe.species, _format(battle)) >= SETTER_RATE


def trick_room_edge(battle: DoubleBattle) -> tuple[float, int]:
    """(+1 .. -1, comparisons): under Trick Room, +1 = every comparison favours
    us, -1 = every one favours them; HP-weighted as in the tempo reranker."""
    heavy = heavy_trick_room(battle)
    score = total = 0.0
    known = 0
    for ours in battle.active_pokemon:
        if ours is None or ours.fainted:
            continue
        our_bounds = effective_speed_bounds(battle, ours, True)
        if our_bounds is None:
            continue
        for foe in battle.opponent_active_pokemon:
            if foe is None or foe.fainted:
                continue
            K.ensure_stats(foe)  # the damage calculator's stat estimate
            foe_bounds = effective_speed_bounds(battle, foe, False)
            if foe_bounds is None:
                continue
            if heavy:
                foe_bounds = (foe_bounds[0], foe_bounds[0])  # minimum Speed
            if our_bounds[1] < foe_bounds[0]:
                favours_us = True  # we are slower: we move first under Trick Room
            elif our_bounds[0] > foe_bounds[1]:
                favours_us = False
            else:
                continue
            weight = max(0.02, float(foe.current_hp_fraction or 0.0)) * max(
                0.20, float(ours.current_hp_fraction or 0.0)
            )
            score += weight if favours_us else -weight
            total += weight
            known += 1
    return (score / total if total else 0.0), known


def _uses_trick_room(battle: DoubleBattle, candidate: G.Candidate) -> bool:
    for pos, action in enumerate(candidate.actions):
        move = getattr(G._decode(battle, action, pos), "order", None)
        if isinstance(move, Move) and move.id == TRICK_ROOM_MOVE:
            return True
    return False


def _promote(battle, cands, report, reason: str) -> list[G.Candidate]:
    live = [c for c in cands if c.demoted_by is None]
    if not live or _uses_trick_room(battle, live[0]):
        return cands
    best = next((c for c in live if _uses_trick_room(battle, c)), None)
    if best is None:
        report.demotions[f"{NAME}:{reason}_unranked"] += 1
        return cands
    best.prob = max(best.prob, live[0].prob)  # survive the opponent reranker
    report.demotions[f"{NAME}:{reason}"] += 1
    return G._promote_candidate(cands, best, NAME, report)


def _demote(battle, cands, report, reason: str) -> list[G.Candidate]:
    dead = {i for i, c in enumerate(cands) if _uses_trick_room(battle, c)}
    before = report.demotions[NAME]
    out = G._demote(cands, dead, NAME, report)
    if report.demotions[NAME] > before:
        report.demotions[f"{NAME}:{reason}"] += 1
    return out


def guard_trick_room_direction(battle, cands, report) -> list[G.Candidate]:
    """Set, keep or reverse Trick Room by whom it helps (see the module doc)."""
    if not any(_uses_trick_room(battle, c) for c in cands if c.demoted_by is None):
        return cands
    edge, known = trick_room_edge(battle)
    if not known or abs(edge) < CLEAR_EDGE:
        return cands
    favours_us = edge > 0
    if Field.TRICK_ROOM in battle.fields:
        if favours_us:
            return _demote(battle, cands, report, "keep_ours")
        if trick_room_turns(battle) >= 2:
            return _promote(battle, cands, report, "reverse")
        return cands
    if favours_us:
        return cands
    if any(likely_setter(battle, foe) for foe in battle.opponent_active_pokemon):
        return _promote(battle, cands, report, "counter")
    return _demote(battle, cands, report, "no_gift")
