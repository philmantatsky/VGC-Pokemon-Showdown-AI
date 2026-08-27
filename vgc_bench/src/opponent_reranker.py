"""Prediction-conditioned soft reranking of the policy's strongest joint actions.

This is intentionally narrower than search. It does not claim to simulate a turn.
Instead, it corrects two concrete blind spots while exact Showdown state parity is
being built:

* attacks are valued against both the current target and likely switch-ins;
* predicted incoming damage is valued against our post-choice board (including a
  switch or Protect).

The learned policy remains the prior. Only a near-tied, non-vetoed prefix can move,
and an unplanned Tera can never be promoted. Any missing calculation simply
contributes no tactical evidence.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from poke_env.battle import DoubleBattle, Move, MoveCategory, Pokemon, Target
from poke_env.data import to_id_str

from vgc_bench.src import guards
from vgc_bench.src import vgc_knowledge as K
from vgc_bench.src.guards import PROTECT_MOVES, Candidate
from vgc_bench.src.opponent_tactics import MovePrediction, SwitchPrediction
from vgc_bench.src.tempo_reranker import SpeedControlSnapshot, score_candidates

_SPREAD_TARGETS = {Target.ALL, Target.ALL_ADJACENT, Target.ALL_ADJACENT_FOES}


@dataclass(frozen=True)
class OpponentRerankReport:
    before: tuple[int, int]
    after: tuple[int, int]
    evaluated: int
    policy_score_before: float
    tactical_score_before: float
    tactical_score_after: float
    tempo_score_before: float = 0.0
    tempo_score_after: float = 0.0
    tempo_factors_before: tuple[tuple[str, float], ...] = ()
    tempo_factors_after: tuple[tuple[str, float], ...] = ()
    tempo_factor_totals: tuple[tuple[str, float], ...] = ()
    speed_control: SpeedControlSnapshot | None = None
    special_reason: str | None = None

    @property
    def changed(self) -> bool:
        return self.before != self.after


def _orders(battle: DoubleBattle, candidate: Candidate):
    orders = []
    for pos, action in enumerate(candidate.actions):
        cached = candidate.orders[pos]
        orders.append(
            cached if cached is not None else guards._decode(battle, action, pos)
        )
    return orders


def _is_tera(candidate: Candidate) -> bool:
    return any(86 < action <= 106 for action in candidate.actions)


def _damage(
    battle: DoubleBattle,
    attacker: Pokemon,
    target: Pokemon,
    move: Move,
    cache: dict[tuple[int, int, str], float],
) -> float:
    key = (id(attacker), id(target), move.id)
    if key in cache:
        return cache[key]
    fraction = K.damage_fraction(battle, attacker, target, move)
    expected = (
        min(1.0, (fraction[0] + fraction[1]) / 2 * (move.accuracy or 1.0))
        if fraction
        else 0.0
    )
    cache[key] = expected
    return expected


def _target_slots(move: Move, order) -> list[int]:
    if move.target in _SPREAD_TARGETS:
        return [0, 1]
    target = int(getattr(order, "move_target", 0) or 0)
    return [target - 1] if target in {1, 2} else []


def _bench_by_species(battle: DoubleBattle) -> dict[str, Pokemon]:
    active = {id(mon) for mon in battle.opponent_active_pokemon if mon is not None}
    return {
        to_id_str(mon.base_species): mon
        for mon in battle.opponent_team.values()
        if id(mon) not in active and not mon.fainted
    }


def _switch_damage(
    battle: DoubleBattle,
    attacker: Pokemon,
    move: Move,
    prediction: SwitchPrediction | None,
    bench: dict[str, Pokemon],
    cache: dict[tuple[int, int, str], float],
) -> tuple[float, float]:
    """Return (switch probability with usable targets, conditional switch damage)."""
    if prediction is None or prediction.switch_probability <= 0 or not bench:
        return 0.0, 0.0
    weighted = [
        (bench[species], probability)
        for species, probability in prediction.targets
        if species in bench and probability > 0
    ]
    total = sum(probability for _, probability in weighted)
    if total <= 0:
        return 0.0, 0.0
    expected = sum(
        probability / total * _damage(battle, attacker, target, move, cache)
        for target, probability in weighted
    )
    # The switch model is only moderately calibrated. Capping prevents one uncertain
    # prediction from overwhelming the policy while still catching obvious pivots.
    return min(0.65, max(0.0, prediction.switch_probability)), expected


def _outgoing_damage(
    battle: DoubleBattle,
    candidate: Candidate,
    switches: tuple[SwitchPrediction, SwitchPrediction] | None,
    cache: dict[tuple[int, int, str], float],
) -> float:
    """Return only the damage adjustment caused by predicted switches.

    The PPO already ranks ordinary damage versus the current board. Adding that raw
    damage a second time pushed support and positioning moves down for no new reason.
    The opponent model should contribute only information absent from the policy's
    visible state: how the value changes if a target switches.
    """
    foes = battle.opponent_active_pokemon
    bench = _bench_by_species(battle)
    by_slot = [0.0, 0.0]
    for pos, order in enumerate(_orders(battle, candidate)):
        move = getattr(order, "order", None)
        attacker = battle.active_pokemon[pos]
        if (
            not isinstance(move, Move)
            or attacker is None
            or move.category == MoveCategory.STATUS
            or move.base_power <= 0
        ):
            continue
        for slot in _target_slots(move, order):
            if slot >= len(foes):
                continue
            foe = foes[slot]
            if foe is None or foe.fainted:
                continue
            current = _damage(battle, attacker, foe, move, cache)
            prediction = switches[slot] if switches is not None else None
            switch_probability, switched = _switch_damage(
                battle, attacker, move, prediction, bench, cache
            )
            by_slot[slot] += switch_probability * (switched - current)
    return sum(max(-1.0, min(1.0, adjustment)) for adjustment in by_slot)


def _our_post_choice_board(battle: DoubleBattle, candidate: Candidate):
    targets: list[Pokemon | None] = list(battle.active_pokemon)
    protected = [False, False]
    for pos, order in enumerate(_orders(battle, candidate)):
        chosen = getattr(order, "order", None)
        if isinstance(chosen, Pokemon):
            targets[pos] = chosen
        elif isinstance(chosen, Move) and chosen.id in PROTECT_MOVES:
            protected[pos] = True
    return targets, protected


def _move_for_prediction(attacker: Pokemon, move_id: str, gen: int) -> Move | None:
    move = attacker.moves.get(move_id)
    if move is not None:
        return move
    try:
        return Move(move_id, gen=gen)
    except Exception:
        return None


def _incoming_damage(
    battle: DoubleBattle,
    candidate: Candidate,
    moves: tuple[MovePrediction, MovePrediction] | None,
    cache: dict[tuple[int, int, str], float],
) -> float:
    if moves is None:
        return 0.0
    targets, protected = _our_post_choice_board(battle, candidate)
    by_slot = [0.0, 0.0]
    for foe_slot, attacker in enumerate(battle.opponent_active_pokemon):
        if attacker is None or attacker.fainted:
            continue
        prediction = moves[foe_slot]
        reliability = min(1.0, max(0.0, prediction.reliability))
        if reliability <= 0:
            continue
        move_probabilities = dict(prediction.moves)
        action_probabilities: dict[tuple[str, str], float] = {
            (move_id, target): probability
            for move_id, target, probability in prediction.actions
        }
        for move_id, move_probability in move_probabilities.items():
            move = _move_for_prediction(attacker, move_id, battle.gen)
            if (
                move is None
                or move.category == MoveCategory.STATUS
                or move.base_power <= 0
            ):
                continue
            if move.target in _SPREAD_TARGETS:
                target_weights = [(0, move_probability), (1, move_probability)]
            else:
                target_weights = [
                    (0, action_probabilities.get((move_id, "foe_a"), 0.0)),
                    (1, action_probabilities.get((move_id, "foe_b"), 0.0)),
                ]
            for slot, probability in target_weights:
                probability *= reliability
                target = targets[slot]
                if probability <= 0 or target is None or protected[slot]:
                    continue
                by_slot[slot] += probability * _damage(
                    battle, attacker, target, move, cache
                )
    return sum(min(1.0, damage) for damage in by_slot)


def _own_recoil(
    battle: DoubleBattle,
    candidate: Candidate,
    cache: dict[tuple[int, int, str], float],
) -> tuple[float, float]:
    """Expected recoil as a fraction of each attacker's maximum HP."""
    recoil = [0.0, 0.0]
    foes = battle.opponent_active_pokemon
    for pos, order in enumerate(_orders(battle, candidate)):
        move = getattr(order, "order", None)
        attacker = battle.active_pokemon[pos]
        if (
            not isinstance(move, Move)
            or attacker is None
            or move.category == MoveCategory.STATUS
            or move.base_power <= 0
        ):
            continue
        dealt = 0.0
        for slot in _target_slots(move, order):
            if slot >= len(foes):
                continue
            target = foes[slot]
            if target is None or target.fainted:
                continue
            fraction = _damage(battle, attacker, target, move, cache)
            try:
                hp_ratio = float(target.max_hp) / float(attacker.max_hp)
            except (TypeError, ValueError, ZeroDivisionError):
                hp_ratio = 1.0
            dealt += fraction * hp_ratio
        if dealt <= 0:
            continue
        recoil[pos] += dealt * float(move.recoil or 0.0)
        if to_id_str(attacker.item or "") == "lifeorb":
            recoil[pos] += 0.10
    return recoil[0], recoil[1]


def _predicted_lethal_risk(
    battle: DoubleBattle,
    candidate: Candidate,
    moves: tuple[MovePrediction, MovePrediction],
    cache: dict[tuple[int, int, str], float],
) -> tuple[float, float]:
    """Conservative chance/pressure that each post-choice slot is knocked out.

    This deliberately uses the move prior even before a move is revealed. The normal
    soft reranker correctly discounts hidden sets, but that made its opening-turn
    survival score exactly zero in 24/25 ladder games. Target-head predictions are
    not trusted here either: an unrevealed single-target attack is tested against
    both legal slots, preventing labels such as Electro Shot -> ``field`` from hiding
    an otherwise obvious lethal threat.
    """
    targets, protected = _our_post_choice_board(battle, candidate)
    recoil = _own_recoil(battle, candidate, cache)
    risk = [0.0, 0.0]
    for slot, target in enumerate(targets):
        if target is None or protected[slot]:
            continue
        hp = max(0.0, float(target.current_hp_fraction or 0.0) - recoil[slot])
        if hp <= 0:
            risk[slot] = 1.0
            continue
        for foe_slot, attacker in enumerate(battle.opponent_active_pokemon):
            if attacker is None or attacker.fainted:
                continue
            for move_id, probability in moves[foe_slot].moves:
                if probability <= 0:
                    continue
                move = _move_for_prediction(attacker, move_id, battle.gen)
                if (
                    move is None
                    or move.category == MoveCategory.STATUS
                    or move.base_power <= 0
                ):
                    continue
                fraction = K.damage_fraction(battle, attacker, target, move)
                if fraction is None:
                    continue
                multiplier = K.type_multiplier(move, target)
                if multiplier is None or multiplier <= 1.0:
                    continue
                accuracy = float(move.accuracy or 0.0)
                if max(0.0, fraction[0]) >= hp:
                    # Do not add unrelated guesses together until they look certain.
                    # The rejected broad version treated several mediocre priors as
                    # one overwhelming threat and fired 123 times per 100 battles.
                    risk[slot] = max(
                        risk[slot], min(1.0, float(probability) * accuracy)
                    )
    return risk[0], risk[1]


def _emergency_survival_candidate(
    battle: DoubleBattle,
    live: list[Candidate],
    moves: tuple[MovePrediction, MovePrediction] | None,
    cache: dict[tuple[int, int, str], float],
) -> Candidate | None:
    """Promote defense only for a likely unprofitable knockout of one of our slots."""
    if (
        moves is None
        or len(live) < 2
        or os.environ.get("VGC_PREDICTED_SURVIVAL", "0") == "0"
        or not hasattr(battle, "active_pokemon")
        or not hasattr(battle, "opponent_active_pokemon")
        or not hasattr(battle, "gen")
        or int(getattr(battle, "turn", 1) or 1)
        > int(os.environ.get("VGC_SURVIVAL_MAX_TURN", "2"))
    ):
        return None
    top = live[0]
    top_risk = _predicted_lethal_risk(battle, top, moves, cache)
    risk_threshold = float(os.environ.get("VGC_SURVIVAL_RISK_THRESHOLD", "0.60"))
    protected_species = {
        to_id_str(name)
        for name in os.environ.get("VGC_SURVIVAL_SPECIES", "basculegion").split(",")
        if name.strip()
    }
    top_orders = _orders(battle, top)
    threatened = set()
    for slot, risk in enumerate(top_risk):
        mon = battle.active_pokemon[slot]
        move = getattr(top_orders[slot], "order", None)
        if (
            risk >= risk_threshold
            and mon is not None
            and to_id_str(mon.base_species) in protected_species
            and isinstance(move, Move)
            and move.category != MoveCategory.STATUS
            and move.base_power > 0
        ):
            threatened.add(slot)
    if not threatened:
        return None
    top_kos, _, top_valid = guards._candidate_guaranteed_progress(battle, top)
    if not top_valid or top_kos > 0:
        return None

    minimum_ratio = float(os.environ.get("VGC_SURVIVAL_MIN_RATIO", "0.002"))
    minimum_reduction = float(os.environ.get("VGC_SURVIVAL_MIN_REDUCTION", "0.30"))
    alternatives = []
    for candidate in live[1:]:
        if candidate.prob < top.prob * minimum_ratio:
            continue
        if _is_tera(candidate) and not _is_tera(top):
            continue
        orders = _orders(battle, candidate)
        defended = {
            pos
            for pos, order in enumerate(orders)
            if isinstance(getattr(order, "order", None), Pokemon)
            or (
                isinstance(getattr(order, "order", None), Move)
                and getattr(order, "order").id in PROTECT_MOVES
            )
        }
        if not (defended & threatened):
            continue
        if any(
            guards._order_signature(orders[pos])
            != guards._order_signature(top_orders[pos])
            for pos in range(2)
            if pos not in threatened
        ):
            continue
        kos, _, valid = guards._candidate_guaranteed_progress(battle, candidate)
        if not valid or kos < top_kos:
            continue
        candidate_risk = _predicted_lethal_risk(battle, candidate, moves, cache)
        reduction = sum(top_risk[pos] - candidate_risk[pos] for pos in threatened)
        if reduction < minimum_reduction:
            continue
        alternatives.append(
            (
                reduction,
                -sum(candidate_risk),
                candidate.prob,
                candidate,
            )
        )
    if not alternatives:
        return None
    return max(alternatives, key=lambda entry: entry[:3])[3]


def rerank_candidates(
    battle: DoubleBattle,
    candidates: list[Candidate],
    opponent_moves: tuple[MovePrediction, MovePrediction] | None,
    opponent_switches: tuple[SwitchPrediction, SwitchPrediction] | None,
    *,
    weight: float | None = None,
    tempo_weight: float | None = None,
    min_policy_ratio: float | None = None,
    max_candidates: int = 8,
    use_opponent: bool = True,
    use_tempo: bool = False,
) -> tuple[list[Candidate], OpponentRerankReport | None]:
    """Blend opponent predictions and tempo mechanics into one near-tied rerank.

    Both evidence sources must be scored together. Running two independent rerankers
    in sequence lets the second silently erase the first because candidate
    probabilities do not contain the first layer's adjustment.
    """
    live = [candidate for candidate in candidates if candidate.demoted_by is None]
    demoted = [
        candidate for candidate in candidates if candidate.demoted_by is not None
    ]
    if len(live) < 2:
        return candidates, None
    original = live[0]
    cache: dict[tuple[int, int, str], float] = {}
    emergency = (
        _emergency_survival_candidate(battle, live, opponent_moves, cache)
        if use_opponent
        else None
    )

    # A switch-in matchup is only trustworthy when the opponent's actual moves are
    # known (normally through OTS). In hidden-sheet games the move prior fills all
    # four slots, but treating those guesses as facts made the reranker materially
    # worse. Revealed moves can still provide defensive evidence on their own.
    sets_known = opponent_moves is not None and all(
        prediction.reliability >= 0.999 for prediction in opponent_moves
    )
    effective_switches = opponent_switches if sets_known and use_opponent else None
    has_move_evidence = (
        use_opponent
        and opponent_moves is not None
        and any(prediction.reliability > 0 for prediction in opponent_moves)
    )
    if weight is None:
        weight = float(os.environ.get("VGC_OPPONENT_RERANK_WEIGHT", "0.65"))
    if tempo_weight is None:
        tempo_weight = float(os.environ.get("VGC_TEMPO_RERANK_WEIGHT", "0.85"))
    if min_policy_ratio is None:
        min_policy_ratio = float(
            os.environ.get("VGC_OPPONENT_RERANK_MIN_RATIO", "0.30")
        )
    top_probability = max(live[0].prob, 1e-12)
    eligible = [
        candidate
        for candidate in live
        if candidate.prob >= top_probability * min_policy_ratio
        and (not candidate.strategic_only or candidate is live[0])
    ][:max_candidates]
    if emergency is not None and emergency not in eligible:
        eligible.append(emergency)
    if len(eligible) < 2:
        return candidates, None

    opponent_utility = {
        candidate.actions: (
            _outgoing_damage(battle, candidate, effective_switches, cache)
            - _incoming_damage(battle, candidate, opponent_moves, cache)
            if has_move_evidence or effective_switches is not None
            else 0.0
        )
        for candidate in eligible
    }
    tempo = None
    if use_tempo:
        tempo = score_candidates(battle, eligible, opponent_moves, opponent_switches)
        tempo_utility = tempo.utility
    else:
        tempo_utility = {candidate.actions: 0.0 for candidate in eligible}
    if (
        not has_move_evidence
        and effective_switches is None
        and not (tempo is not None and tempo.has_evidence)
        and emergency is None
    ):
        return candidates, None

    def score(candidate: Candidate) -> float:
        base = math.log(max(candidate.prob, 1e-12) / top_probability)
        # Like the KO tiebreak, never promote a Tera the policy did not choose itself.
        may_promote = (
            0.0
            if _is_tera(candidate) and candidate.actions != original.actions
            else 1.0
        )
        return (
            base
            + may_promote * weight * opponent_utility[candidate.actions]
            + may_promote * tempo_weight * tempo_utility[candidate.actions]
        )

    reranked = sorted(eligible, key=score, reverse=True)
    if emergency is not None:
        reranked = [emergency] + [
            candidate for candidate in reranked if candidate is not emergency
        ]
    remaining = [candidate for candidate in live if candidate not in eligible]
    out = reranked + remaining + demoted
    report = OpponentRerankReport(
        before=original.actions,
        after=out[0].actions,
        evaluated=len(eligible),
        policy_score_before=math.log(max(original.prob, 1e-12) / top_probability),
        tactical_score_before=opponent_utility[original.actions],
        tactical_score_after=opponent_utility[out[0].actions],
        tempo_score_before=tempo_utility[original.actions],
        tempo_score_after=tempo_utility[out[0].actions],
        tempo_factors_before=(
            tuple(tempo.factors[original.actions].items()) if tempo is not None else ()
        ),
        tempo_factors_after=(
            tuple(tempo.factors[out[0].actions].items()) if tempo is not None else ()
        ),
        tempo_factor_totals=(
            tuple(tempo.factor_totals.items()) if tempo is not None else ()
        ),
        speed_control=tempo.snapshot if tempo is not None else None,
        special_reason=("predicted_ko_survival" if emergency is not None else None),
    )
    return out, report
