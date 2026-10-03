"""doomed_switch: switch out a Pokemon that will be knocked out before it moves.

Ladder 2026-10-03: in 66 of the 67 turn 1-3 first faints with a known killing
move, one of our two benched Pokemon would have survived that hit
(results_analysis/early_faints_20261003/switch_counterfactual.txt), and half of
those Pokemon fainted before acting -- so their own action was lost anyway.

For one of our active Pokemon, the guard looks for a foe that knocks it out this
turn with at least DOOMED of the roll (threat_first2.threats: revealed moves plus
the most-used set, the foe's likely Mega forme counted) before it can move
(threat_first.acts_before is false for its top-pair action), that cannot also knock
out our partner (so our Pokemon is its natural target), and that the top pair does
not answer (threat_first2). If the Pokemon has no usable Protect and its top-pair
action is an attack (a status move such as Trick Room stays: switching would
abandon the plan, not just a lost attack), the best-ranked
pair that keeps the partner's action and switches it to a benched Pokemon taking at
most SWITCH_MAX of its current HP from that move (both formes) is promoted. Only
ranked pairs (always legal). Opt-in: not in HARD_GUARDS until it passes its A/B.
"""

from __future__ import annotations

from poke_env.battle import Move, MoveCategory, Pokemon

from vgc_bench.src import guards as G
from vgc_bench.src import threat_first as T
from vgc_bench.src import threat_first2 as T2
from vgc_bench.src import vgc_knowledge as K

NAME = "doomed_switch"
DOOMED = 0.9  # the threat knocks our Pokemon out at least this surely
SWITCH_MAX = 0.6  # the switch-in takes at most this share of its current HP


def _damage_ceiling(battle, foe: Pokemon, mon: Pokemon, move: Move) -> float | None:
    """The highest roll of ``move`` from ``foe`` (either forme) on ``mon``."""
    worst = None
    for form in T2._forms(battle, foe):
        if form is None:
            fraction = K.damage_fraction(battle, foe, mon, move)
        else:
            with T2.as_mega(battle, foe, form) as mega:
                if mega is None:
                    return None
                fraction = K.damage_fraction(battle, mega, mon, move)
        if fraction is None:
            return None
        worst = fraction[1] if worst is None else max(worst, fraction[1])
    return worst


def _has_protect(mon: Pokemon) -> bool:
    if int(getattr(mon, "protect_counter", 0) or 0) > 0:
        return False
    return any(
        isinstance(m, Move) and m.id in G.PROTECT_MOVES
        for m in (mon.moves or {}).values()
    )


def guard_doomed_switch(battle, cands, report) -> list[G.Candidate]:
    """Switch a Pokemon that is knocked out before it moves (see the module doc)."""
    live = [candidate for candidate in cands if candidate.demoted_by is None]
    if len(live) < 2:
        return cands
    top = live[0]
    try:
        found = T2.threats(battle)
    except Exception:
        report.demotions[f"{NAME}_error"] += 1
        return cands
    for pos in (0, 1):
        me = battle.active_pokemon[pos]
        partner_pos = 1 - pos
        if me is None or me.fainted or _has_protect(me):
            continue
        order = G._decode(battle, top.actions[pos], pos)
        my_move = getattr(order, "order", None)
        if not isinstance(my_move, Move) or my_move.category == MoveCategory.STATUS:
            continue  # already switching, or a status move (Trick Room...) we keep
        for threat in found:
            chance, foe, victim, foe_move = threat
            if victim != pos or chance < DOOMED:
                continue
            if T.acts_before(battle, me, my_move, foe, foe_move):
                continue  # it moves first: keep its action
            if any(t[1] is foe and t[2] == partner_pos for t in found):
                continue  # the foe may as well aim at our partner
            if T2._handled(battle, top, threat):
                continue
            for candidate in live[1:]:
                if candidate.actions[partner_pos] != top.actions[partner_pos]:
                    continue
                incoming = getattr(
                    G._decode(battle, candidate.actions[pos], pos), "order", None
                )
                if not isinstance(incoming, Pokemon) or incoming.fainted:
                    continue
                hp = incoming.current_hp_fraction or 1.0
                ceiling = _damage_ceiling(battle, foe, incoming, foe_move)
                if ceiling is None or ceiling > SWITCH_MAX * hp:
                    continue
                candidate.prob = max(candidate.prob, top.prob)  # see resisted_target
                report.demotions[f"{NAME}:promoted"] += 1
                return G._promote_candidate(cands, candidate, NAME, report)
    return cands
