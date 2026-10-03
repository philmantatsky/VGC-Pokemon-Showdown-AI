"""fake_out_threat: on its first turn, Fake Out the foe about to knock one of ours out.

Ladder 2026-10-03: since the learned preview, our turn-1 losses of a Pokemon are
mostly Mega Blastoise knocked out before it moves by a faster attacker -- Mega
Raichu's Zap Cannon three times, Rillaboom's Wood Hammer, Sneasler's Close Combat
-- while Blastoise, which carries Fake Out (+3), spent the turn on Water Spout or a
Fake Out into the other foe (results_analysis/early_faints_20261003/). threat_first
answers a threat only with an attack that knocks it out first, which a slower
Pokemon cannot do outside Trick Room; a landing Fake Out answers it for the turn.

For each foe that threatens a knockout this turn (threat_first.threats) and that
the top pair does not already answer, where removing it saves the Pokemon (no other
unanswered foe threatens it) and its threatening move has priority below +3: the
best-ranked pair that keeps the partner's action and replaces one of our plain
attacks (not one already aimed at a threatening foe) with a Fake Out of the same
Pokemon (same Mega/Tera choice) into that foe is promoted -- only when the Fake Out
lands: our Pokemon's first turn, the foe not a Ghost (without Scrappy / Mind's
Eye), not grounded under Psychic Terrain, no Armor Tail / Dazzling / Queenly
Majesty on its side, no Inner Focus / Shield Dust / Covert Cloak known or likely.
Only ranked pairs. Opt-in: not in HARD_GUARDS until it passes its A/B.
"""

from __future__ import annotations

import re

from poke_env.battle import DoubleBattle, Move, Pokemon

from vgc_bench.src import guards as G
from vgc_bench.src import threat_first as T

NAME = "fake_out_threat"
NO_FLINCH_ABILITIES = {"innerfocus", "shielddust"}
NO_FLINCH_SHARE = 0.5  # an unrevealed ability / item this likely blocks the flinch


def flinches(battle: DoubleBattle, attacker: Pokemon, foe: Pokemon) -> bool:
    """Our Fake Out lands on ``foe`` and makes it flinch."""
    from vgc_bench.src.playbook import set_share
    from vgc_bench.src.playbook_opening import fake_out_reachable

    if not attacker.first_turn or not fake_out_reachable(battle, foe, attacker):
        return False
    ability = G._norm(foe.ability)
    if ability in NO_FLINCH_ABILITIES:
        return False
    item = G._norm(foe.item)
    if item == "covertcloak":
        return False
    tag = re.match(r"battle-([a-z0-9]+)-", getattr(battle, "battle_tag", "") or "")
    fmt = battle.format or (tag.group(1) if tag else None)
    if not ability:
        for blocker in NO_FLINCH_ABILITIES:
            if set_share(foe.species, ability=blocker, formatid=fmt) >= NO_FLINCH_SHARE:
                return False
    if not item or item == "unknownitem":
        if set_share(foe.species, item="covertcloak", formatid=fmt) >= NO_FLINCH_SHARE:
            return False
    return True


def guard_fake_out_threat(battle, cands, report) -> list[G.Candidate]:
    """Promote a Fake Out into the foe threatening a KO (see the module doc)."""
    live = [candidate for candidate in cands if candidate.demoted_by is None]
    if len(live) < 2:
        return cands
    top = live[0]
    try:
        found = T.threats(battle)
    except Exception:
        report.demotions[f"{NAME}_error"] += 1
        return cands
    threatening = {id(t[1]) for t in found}
    open_threats = [t for t in found if not T._handled(battle, top, t)]
    for threat in open_threats:
        _, foe, victim_pos, foe_move = threat
        if any(t[2] == victim_pos and t[1] is not foe for t in open_threats):
            continue  # another foe still knocks the same Pokemon out
        if T._priority(battle, foe, foe_move) >= 3:
            continue  # its own Fake Out-speed move goes first or ties
        for pos in (0, 1):
            attacker = battle.active_pokemon[pos]
            if (
                attacker is None
                or attacker.fainted
                or not flinches(battle, attacker, foe)
            ):
                continue
            order = G._decode(battle, top.actions[pos], pos)
            move = getattr(order, "order", None)
            if (
                not isinstance(move, Move)
                or not G._plain_attack(move)
                or move.id in G.FIRST_TURN_ONLY
                or T._single_target_on(battle, order, pos, threatening)
            ):
                continue
            for candidate in live[1:]:
                if candidate.actions[1 - pos] != top.actions[1 - pos]:
                    continue
                alternative = G._decode(battle, candidate.actions[pos], pos)
                if G._gimmicks(alternative) != G._gimmicks(order):
                    continue
                alt_move, hit = G._move_and_targets(battle, alternative, pos)
                if alt_move is None or alt_move.id != "fakeout":
                    continue
                if all(f is not foe for f in hit):
                    continue
                candidate.prob = max(candidate.prob, top.prob)  # see resisted_target
                report.demotions[f"{NAME}:promoted"] += 1
                return G._promote_candidate(cands, candidate, NAME, report)
    return cands
