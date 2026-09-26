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

from collections import Counter

from poke_env.battle import (
    DoubleBattle,
    Effect,
    Field,
    Move,
    MoveCategory,
    Pokemon,
    Status,
)
from poke_env.data import to_id_str

from vgc_bench.src import guards as G
from vgc_bench.src import vgc_knowledge as K
from vgc_bench.src.preview_rules import TRICK_ROOM_MOVE, species_trick_room_rate
from vgc_bench.src.tempo_reranker import (
    effective_speed,
    effective_speed_bounds,
    tailwind_turns,
    trick_room_turns,
)

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


# ---- trick_room_counter ---------------------------------------------------------

COUNTER = "trick_room_counter"
_CANNOT_ACT = (Status.SLP, Status.FRZ)
_FLINCH_PROOF_ABILITIES = {"innerfocus", "shielddust"}


def _fake_out_lands(
    battle: DoubleBattle, partner: Pokemon, foe: Pokemon, move: Move
) -> bool:
    """Our Fake Out flinches ``foe``: our first turn out, no priority blocker on
    their side, no Psychic Terrain under a grounded foe, no flinch immunity."""
    if not partner.first_turn:
        return False
    if any(
        mon is not None
        and not mon.fainted
        and G._norm(mon.ability) in G.PRIORITY_BLOCK_ABILITIES
        for mon in battle.opponent_active_pokemon
    ):
        return False
    psychic_terrain = getattr(Field, "PSYCHIC_TERRAIN", None)
    if (
        psychic_terrain is not None
        and psychic_terrain in battle.fields
        and G._is_grounded(battle, foe)
    ):
        return False
    if (
        G._norm(foe.ability) in _FLINCH_PROOF_ABILITIES
        or G._norm(foe.item) == "covertcloak"
    ):
        return False
    return not K.deals_no_damage(battle, partner, foe, move)


def _partner_stops(
    battle: DoubleBattle,
    candidate: G.Candidate,
    pos: int,
    foe: Pokemon,
    before: int | None,
) -> bool:
    """Our partner's half of the pair keeps ``foe`` from acting this turn: a Fake
    Out that flinches it, or a certain knockout that lands first. Everything at
    normal priority lands before Trick Room's -7; ``before`` is the priority of
    a foe's move the partner must beat instead (Taunt), where a tie needs our
    partner certainly faster."""
    partner_pos = 1 - pos
    partner = battle.active_pokemon[partner_pos]
    if partner is None or partner.fainted:
        return False
    order = G._decode(battle, candidate.actions[partner_pos], partner_pos)
    move = getattr(order, "order", None)
    if not isinstance(move, Move) or foe not in G.resolved_foe_targets(
        battle, order, move
    ):
        return False
    priority = G._effective_priority(partner, move)
    if move.id == "fakeout":
        landed = _fake_out_lands(battle, partner, foe, move)
    elif move.category == MoveCategory.STATUS:
        return False
    else:
        landed = K.guaranteed_ko(battle, partner, foe, move)
    if not landed or before is None:
        return landed
    if priority != before:
        return priority > before
    K.ensure_stats(foe)
    ours = effective_speed_bounds(battle, partner, True)
    theirs = effective_speed_bounds(battle, foe, False)
    return ours is not None and theirs is not None and ours[0] > theirs[1]


def _taunt_lands(battle: DoubleBattle, taunter: Pokemon, setter: Pokemon) -> bool:
    """A shown Taunt that would reach our setter."""
    taunt = G._known_move(taunter, "taunt")
    if taunt is None:
        return False
    priority = G._effective_priority(taunter, taunt)
    if G._priority_is_blocked(battle, taunter, setter, priority):
        return False  # Prankster into Armor Tail / Dazzling / a Dark setter
    if G._norm(setter.ability) == "oblivious" or G._norm(setter.item) == "mentalherb":
        return False
    return not any(
        mon is not None and not mon.fainted and G._norm(mon.ability) == "aromaveil"
        for mon in battle.active_pokemon
    )


def _fast_mode(battle: DoubleBattle, pos: int) -> bool:
    """Our other active certainly outruns every active foe without Trick Room --
    against their fastest plausible spread, Scarf aside -- so the room would slow
    us: pressing it as they press theirs is the counter the user described."""
    partner = battle.active_pokemon[1 - pos]
    if partner is None or partner.fainted:
        return False
    mine = effective_speed(battle, partner, True)
    if mine is None:
        return False
    for foe in battle.opponent_active_pokemon:
        if foe is None or foe.fainted:
            continue
        base = (getattr(foe, "base_stats", None) or {}).get("spe")
        if not base:
            return False
        ceiling = int((base + 52) * 1.1) * G._BOOST_MULT.get(
            foe.boosts.get("spe", 0), 1.0
        )
        if tailwind_turns(battle, False):
            ceiling *= 2
        if not mine > ceiling:
            return False
    return True


def _counter(
    battle: DoubleBattle, candidate: G.Candidate, pos: int, setter: Pokemon
) -> str | None:
    """Why this pair's Trick Room would be stopped this turn, if it would."""
    imprison = getattr(Effect, "IMPRISON", None)
    fmt = _format(battle)
    for foe in battle.opponent_active_pokemon:
        if foe is None or foe.fainted or foe.status in _CANNOT_ACT:
            continue
        knows_room = G._known_move(foe, TRICK_ROOM_MOVE) is not None
        # Imprison disables the move at selection (Showdown refuses the choice
        # outright), so even a partner knocking the Imprison user out first
        # cannot rescue it -- ladder game 11, turn 2.
        if (
            imprison is not None
            and imprison in (foe.effects or {})
            and (knows_room or species_trick_room_rate(foe.species, fmt) >= SETTER_RATE)
        ):
            return "imprison"
        taunt = G._known_move(foe, "taunt")
        if (
            taunt is not None
            and _taunt_lands(battle, foe, setter)
            and not _partner_stops(
                battle, candidate, pos, foe, G._effective_priority(foe, taunt)
            )
        ):
            return "taunt"
        if (
            knows_room
            and not _fast_mode(battle, pos)
            and not _partner_stops(battle, candidate, pos, foe, None)
        ):
            return "reverser"
    return None


def guard_trick_room_counter(battle, cands, report) -> list[G.Candidate]:
    """Do not press Trick Room into a counter the opponent has already shown.

    Ladder T6 reads (75 games), 2026-09-25: game 5 pressed Trick Room again the
    turn after Cofagrigus's own Trick Room cancelled ours; Cofagrigus reversed
    it the next turn and Farigiraf fell without a lasting room. (The Imprison
    and Taunt that stopped the first attempt in games 11 and 19 were not shown
    before that turn -- nothing on the board could have flagged them.)

    With Trick Room down, demotes pairs that set it while an active foe that can
    act this turn (not asleep or frozen) has shown a counter:
      * Imprison up and Trick Room known, or its species' set rate >= 0.5 --
        Showdown refuses the choice itself (game 11, turn 2: the policy's top
        six pairs all pressed Trick Room into Indeedee's Imprison);
      * Taunt that would land (not Prankster into a priority blocker or a Dark
        setter; no Oblivious, Aroma Veil or Mental Herb);
      * its own Trick Room: it can cancel ours this turn or reverse it the next,
        and if it wants the room it will set it itself. Not in our fast mode --
        our other active outruns both foes' fastest plausible spreads -- where
        pressing Trick Room as they press theirs is the user's counter.
    For Taunt and a reverser, stands down when our partner's half of the pair
    stops that foe first (a Fake Out that flinches it, or a certain knockout
    landing before it) -- which also promotes such a pair over a Protect beside
    the room. Stands down when nothing else is legal. Opt-in: not in HARD_GUARDS
    until it passes its A/B.
    """
    if Field.TRICK_ROOM in battle.fields:
        return cands  # setting only; reversing stays the policy's call
    dead: set[int] = set()
    reasons: Counter[str] = Counter()
    for i, candidate in enumerate(cands):
        if candidate.demoted_by is not None:
            continue
        for pos, action in enumerate(candidate.actions):
            move = getattr(G._decode(battle, action, pos), "order", None)
            setter = battle.active_pokemon[pos]
            if (
                not isinstance(move, Move)
                or move.id != TRICK_ROOM_MOVE
                or setter is None
                or setter.fainted
            ):
                continue
            reason = _counter(battle, candidate, pos, setter)
            if reason is not None:
                dead.add(i)
                reasons[reason] += 1
    before = report.demotions[COUNTER]
    out = G._demote(cands, dead, COUNTER, report)
    if report.demotions[COUNTER] > before:
        for reason in reasons:
            report.demotions[f"{COUNTER}:{reason}"] += 1
    return out
