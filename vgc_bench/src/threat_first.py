"""threat_first: knock out the foe that is about to knock out one of ours.

Ladder 2026-10-03 (the user: "what do u think for the losing a pokemon problem"):
over the 236 T6-era ladder games we lose a Pokemon first in 107 and win 21% of
them (71% when the opponent loses one first); 92 of those first faints come on
turns 1-3, 73 of them our two leads, which have no Protect, and 52 of the 92 under
our own Trick Room. The top killer is Rillaboom's Wood Hammer (13), a foe our
Mega Blastoise outspeeds under Trick Room and knocks out with Ice Beam -- while it
was often spending the turn on a spread Water Spout instead
(results_analysis/early_faints_20261003/).

For each foe that threatens to knock out one of our active Pokemon this turn --
its best likely damaging move (revealed moves plus the species' most-used set)
does so with at least THREAT_KO of the damage roll, accuracy included, from our
current HP -- the guard checks whether the top pair already deals with it (one
of our attacks knocks it out before it moves, a Fake Out hits it, or the
threatened Pokemon Protects or switches). If not, it promotes the best-ranked pair
that keeps the partner's action and replaces one of our plain attacks with an
attack of the same Pokemon (same Mega/Tera choice) that knocks the threat out
with at least FINISH_KO before the threat's move: higher priority, or the same
priority and certainly faster (certainly slower under Trick Room). Foes at full
HP that may hold a Focus Sash are never "knocked out". Only ranked pairs; stands
down when the calculator cannot evaluate. Lives outside guards.py because it uses
the tempo reranker's speed model (which imports guards.py). Opt-in: not in
HARD_GUARDS until it passes its A/B.
"""

from __future__ import annotations

import re
from functools import lru_cache

from poke_env.battle import DoubleBattle, Field, Move, MoveCategory, Pokemon, Status
from poke_env.data import GenData, to_id_str

from vgc_bench.src import guards as G
from vgc_bench.src import vgc_knowledge as K
from vgc_bench.src.tempo_reranker import effective_speed

NAME = "threat_first"
THREAT_KO = 0.5  # a foe threatens one of ours when its best move KOs with >= this
FINISH_KO = 0.85  # our attack must knock the threat out at least this surely
SASH_SHARE = 0.3  # an unrevealed item at full HP: a likely Focus Sash blocks the KO
_CANNOT_ACT = (Status.SLP, Status.FRZ)
_WEATHER_SPEED = {
    "chlorophyll": ("sunnyday", "desolateland"),
    "swiftswim": ("raindance", "primordialsea"),
    "sandrush": ("sandstorm",),
    "slushrush": ("snowscape", "hail"),
}


def _threat_moves(battle: DoubleBattle, foe: Pokemon) -> list[Move]:
    """Revealed moves plus the species' most-used set (a foe rarely shows all four)."""
    from vgc_bench.src.playbook import _sets
    from vgc_bench.src.playbook_opening import _species

    moves = {m.id: m for m in (foe.moves or {}).values() if isinstance(m, Move)}
    tag = re.match(r"battle-([a-z0-9]+)-", getattr(battle, "battle_tag", "") or "")
    fmt = battle.format or (tag.group(1) if tag else None)
    entry = _sets(fmt).get(_species(foe)) or {}
    top = max(entry.get("sets", []), key=lambda s: s.get("prob", 0), default=None)
    for move_id in (top or {}).get("moves", []):
        key = to_id_str(move_id)
        if key in moves:
            continue
        try:
            moves[key] = Move(key, gen=9)
        except Exception:
            continue
    return list(moves.values())


def _priority(battle: DoubleBattle, mon: Pokemon, move: Move) -> int:
    priority = G._effective_priority(mon, move)
    if move.id == "grassyglide" and Field.GRASSY_TERRAIN in getattr(
        battle, "fields", {}
    ):
        priority += 1
    return priority


def _ko_chance(fraction: tuple[float, float], hp: float) -> float:
    """Share of the damage roll range that knocks out ``hp`` (max-HP fractions)."""
    lo, hi = fraction
    if lo >= hp:
        return 1.0
    if hi < hp or hi <= lo:
        return 0.0
    return (hi - hp) / (hi - lo)


@lru_cache(maxsize=None)
def _mega_base_speed(root: str) -> float | None:
    if "mega" in root:
        return None
    speeds = [
        float(entry.get("baseStats", {}).get("spe", 0))
        for key, entry in GenData.from_gen(9).pokedex.items()
        if key.startswith(root + "mega")
    ]
    return max(speeds) if speeds else None


def _mega_speed(battle: DoubleBattle, foe: Pokemon) -> float | None:
    """The base Speed of the foe's fastest Mega forme while its side can still Mega."""
    if getattr(battle, "_opponent_used_mega_evolve", False):
        return None
    return _mega_base_speed(to_id_str(foe.species or ""))


def foe_speed_bounds(battle: DoubleBattle, foe: Pokemon) -> tuple[float, float] | None:
    """(slowest, fastest) Speed of a foe over every Champions build of its species
    (0-32 points, -/+ nature), its Mega if its side can still Mega, its stages,
    Tailwind, paralysis, weather abilities and items: a revealed Choice Scarf /
    Iron Ball counts, an unrevealed item may be a Scarf (the upper bound only)."""
    base = float((getattr(foe, "base_stats", None) or {}).get("spe") or 0)
    if not base:
        return None
    low, high = (base + 20) * 0.9, (base + 52) * 1.1
    mega = _mega_speed(battle, foe)
    if mega is not None:
        low = min(low, (mega + 20) * 0.9)
        high = max(high, (mega + 52) * 1.1)
    stage = G._BOOST_MULT.get(foe.boosts.get("spe", 0), 1.0)
    low, high = low * stage, high * stage
    if G.SideCondition.TAILWIND in battle.opponent_side_conditions:
        low, high = low * 2, high * 2
    if foe.status == Status.PAR and G._norm(foe.ability) != "quickfeet":
        low, high = low * 0.5, high * 0.5
    item = G._norm(foe.item)
    if item == "choicescarf":
        low, high = low * 1.5, high * 1.5
    elif item == "ironball":
        low, high = low * 0.5, high * 0.5
    elif not item or item == "unknownitem":
        high *= 1.5
    weather = {str(w).split(".")[-1].lower() for w in getattr(battle, "weather", {})}
    abilities = (
        {G._norm(foe.ability)}
        if foe.ability
        else {G._norm(a) for a in getattr(foe, "possible_abilities", ())}
    )
    for ability in abilities:
        if weather & {w.replace("_", "") for w in _WEATHER_SPEED.get(ability, ())}:
            if foe.ability:
                low *= 2
            high *= 2
            break
    return low, high


def acts_before(
    battle: DoubleBattle, me: Pokemon, my_move: Move, foe: Pokemon, foe_move: Move
) -> bool:
    """Our move certainly resolves before the foe's move this turn."""
    mine, theirs = _priority(battle, me, my_move), _priority(battle, foe, foe_move)
    if mine != theirs:
        return mine > theirs
    speed = effective_speed(battle, me, True)
    bounds = foe_speed_bounds(battle, foe)
    if speed is None or bounds is None:
        return False
    if Field.TRICK_ROOM in getattr(battle, "fields", {}):
        return speed < bounds[0]
    return speed > bounds[1]


def threats(battle: DoubleBattle) -> list[tuple[float, Pokemon, int, Move]]:
    """(KO chance, foe, our position, its move) for every foe that can knock out
    one of our active Pokemon this turn with at least THREAT_KO, best first."""
    out = []
    for foe in battle.opponent_active_pokemon:
        if foe is None or foe.fainted or foe.status in _CANNOT_ACT:
            continue
        K.ensure_stats(foe)
        moves = _threat_moves(battle, foe)
        for pos, me in enumerate(battle.active_pokemon):
            if me is None or me.fainted:
                continue
            hp = float(me.current_hp_fraction or 0.0)
            best: tuple[float, Move] | None = None
            for move in moves:
                if move.category == MoveCategory.STATUS or not move.base_power:
                    continue
                priority = _priority(battle, foe, move)
                if priority > 0 and G._priority_is_blocked(battle, foe, me, priority):
                    continue
                if move.id in G.FIRST_TURN_ONLY and not foe.first_turn:
                    continue
                fraction = K.damage_fraction(battle, foe, me, move)
                if fraction is None:
                    continue
                accuracy = move.accuracy if isinstance(move.accuracy, float) else 1.0
                chance = _ko_chance(fraction, hp) * min(accuracy, 1.0)
                if best is None or chance > best[0]:
                    best = (chance, move)
            if best is not None and best[0] >= THREAT_KO:
                out.append((best[0], foe, pos, best[1]))
    return sorted(out, key=lambda t: -t[0])


def _sash_likely(battle: DoubleBattle, foe: Pokemon) -> bool:
    if (foe.current_hp_fraction or 0.0) < 0.999:
        return False
    item = G._norm(foe.item)
    if item == "focussash":
        return True
    if item and item != "unknownitem":
        return False
    from vgc_bench.src.playbook import set_share

    tag = re.match(r"battle-([a-z0-9]+)-", getattr(battle, "battle_tag", "") or "")
    fmt = battle.format or (tag.group(1) if tag else None)
    return set_share(foe.species, item="focussash", formatid=fmt) >= SASH_SHARE


def knocks_out_first(
    battle: DoubleBattle, pos: int, order, foe: Pokemon, foe_move: Move
) -> bool:
    """The order's attack knocks ``foe`` out (>= FINISH_KO) before ``foe_move``."""
    me = battle.active_pokemon[pos]
    move, hit = G._move_and_targets(battle, order, pos)
    if me is None or move is None or move.category == MoveCategory.STATUS:
        return False
    if all(f is not foe for f in hit) or _sash_likely(battle, foe):
        return False
    if not acts_before(battle, me, move, foe, foe_move):
        return False
    fraction = K.damage_fraction(battle, me, foe, move)
    if fraction is None:
        return False
    accuracy = move.accuracy if isinstance(move.accuracy, float) else 1.0
    hp = float(foe.current_hp_fraction or 0.0)
    return _ko_chance(fraction, hp) * min(accuracy, 1.0) >= FINISH_KO


def _handled(battle: DoubleBattle, top: G.Candidate, threat) -> bool:
    """The top pair already answers this threat."""
    _, foe, victim_pos, foe_move = threat
    victim_order = G._decode(battle, top.actions[victim_pos], victim_pos)
    victim_move = getattr(victim_order, "order", None)
    if not isinstance(victim_move, Move):
        return True  # a switch (or no order): the threatened Pokemon leaves
    if victim_move.id in G.PROTECT_MOVES:
        return True
    for pos in (0, 1):
        order = G._decode(battle, top.actions[pos], pos)
        move, hit = G._move_and_targets(battle, order, pos)
        if move is None or all(f is not foe for f in hit):
            continue
        if move.id == "fakeout":
            return True  # the factual vetoes already removed Fake Outs that miss
        if knocks_out_first(battle, pos, order, foe, foe_move):
            return True
    return False


def _single_target_on(battle: DoubleBattle, order, pos: int, foes: set[int]) -> bool:
    """A single-target attack already aimed at one of ``foes`` (by id)."""
    move, hit = G._move_and_targets(battle, order, pos)
    if move is None or move.target in (
        G.Target.ALL_ADJACENT_FOES,
        G.Target.ALL_ADJACENT,
    ):
        return False
    return any(id(f) in foes for f in hit)


def guard_threat_first(battle, cands, report) -> list[G.Candidate]:
    """Promote the attack that removes a foe threatening a KO (see the module doc).

    Two refinements from replaying it over our ladder decisions (2026-10-03): it
    acts only when removing the threat saves the Pokemon (no other unanswered foe
    can knock that Pokemon out this turn), and it never moves a single-target
    attack off a foe that is itself a threat -- only spread attacks and attacks
    into harmless foes are redirected."""
    live = [candidate for candidate in cands if candidate.demoted_by is None]
    if len(live) < 2:
        return cands
    top = live[0]
    try:
        found = threats(battle)
    except Exception:
        report.demotions[f"{NAME}_error"] += 1
        return cands
    threatening = {id(t[1]) for t in found}
    open_threats = [t for t in found if not _handled(battle, top, t)]
    for threat in open_threats:
        _, foe, victim_pos, foe_move = threat
        if any(t[2] == victim_pos and t[1] is not foe for t in open_threats):
            continue  # another foe still knocks the same Pokemon out
        for pos in (0, 1):
            attacker = battle.active_pokemon[pos]
            if attacker is None or attacker.fainted:
                continue
            order = G._decode(battle, top.actions[pos], pos)
            move = getattr(order, "order", None)
            if (
                not isinstance(move, Move)
                or not G._plain_attack(move)
                or move.id in G.FIRST_TURN_ONLY
                or _single_target_on(battle, order, pos, threatening)
            ):
                continue  # only a spread attack or one into a harmless foe moves
            for candidate in live[1:]:
                if candidate.actions[1 - pos] != top.actions[1 - pos]:
                    continue
                alternative = G._decode(battle, candidate.actions[pos], pos)
                if G._gimmicks(alternative) != G._gimmicks(order):
                    continue
                if not knocks_out_first(battle, pos, alternative, foe, foe_move):
                    continue
                candidate.prob = max(candidate.prob, top.prob)  # see resisted_target
                report.demotions[f"{NAME}:promoted"] += 1
                return G._promote_candidate(cands, candidate, NAME, report)
    return cands
