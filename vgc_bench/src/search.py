"""Legacy approximate one-ply search, disabled by default.

This is the piece that makes the bot reason about a move instead of pattern-matching
it. The knowledge features let it SEE that Tailwind sets a side condition; this lets it
work out that doing so raises its win probability.

Shape, following the Laplace bot:

    for each (our joint action, their joint action):
        simulate one turn      -> successor battle state
        evaluate the successor -> the CRITIC's win-probability estimate
    solve the resulting payoff matrix -> a strategy robust to whatever they pick

Two things make it worth doing despite being only one turn deep:

* The evaluation is LEARNED, not hand-written. Reward is +1 win / -1 loss with
  gamma=1, so the critic's output is an estimate of "how likely am I to win from
  here". Tailwind scores well because positions with Tailwind up won more often in
  ~9M self-play games -- nobody had to write down that Tailwind is good.
* Moves are SIMULTANEOUS, so there is often no single best action, only a best
  mixture. Taking the argmax of a payoff matrix is exploitable; solving it is not.
  nashpy is already a dependency (callback.py uses it for double-oracle).

This module's hand-mutated forward model is retained for reproducible experiments, but
it is not safe for ladder decisions: it does not model switches, Protect, Mega/Tera,
weather, ability/item activation, or enough move effects for our team. The exact
Showdown bridge lives in ``src/exact_sim.py``. Live search remains gated until a
poke-env snapshot synchronizer passes parity tests against that bridge.
"""

from __future__ import annotations

import os
import time
from copy import deepcopy

import numpy as np
import torch
from nashpy import Game
from poke_env.battle import DoubleBattle, Move, MoveCategory, Pokemon, SideCondition

from vgc_bench.src import guards as _guards
from vgc_bench.src import vgc_knowledge as K

# Keep the matrix small enough to stay well inside a ladder turn timer.
OUR_K = 6
THEIR_K = 6

# Wall-clock budget per decision. The search is ANYTIME: rows of the payoff matrix are
# filled one at a time, and when the budget runs out we solve on the rows completed so
# far rather than abandoning the work. Fewer than two rows and we hand back to the
# guard stack, which is fast and (since the type-chart fallback) no longer blind.
#
# Two budgets because the costs are not alike. The opening decision is made once and
# sets up the whole game, so it can afford real thought; every later turn pays its
# cost twelve or so times over and burns the ladder clock. A previous "6 ms/decision"
# estimate was measured on replayed states, where `pass` is the only legal action and
# the matrix collapses to a single row -- it measured nothing. Hence the telemetry
# below: never trust an unmeasured latency claim about this function again.
#
# Measured on 20 live battles / 160 decisions: p50=435ms, p90=3118ms, max=3276ms.
# The median is cheap and the tail is what burns the clock, so 2s clips roughly the
# slowest tenth while leaving most searches untouched. Candidates are ordered by policy
# probability, so a truncated matrix drops the LEAST likely actions first -- graceful
# degradation rather than an arbitrary cut.
SEARCH_BUDGET_S = float(os.environ.get("VGC_SEARCH_BUDGET_S", "2"))
SEARCH_OPENING_BUDGET_S = float(os.environ.get("VGC_SEARCH_OPENING_BUDGET_S", "20"))
ALLOW_APPROXIMATE_SEARCH = os.environ.get("VGC_ALLOW_APPROX_SEARCH") == "1"


def backend_status() -> tuple[bool, str]:
    """Whether a search backend is safe for live decisions, plus an explanation."""
    from vgc_bench.src.exact_sim import live_snapshot_supported

    if live_snapshot_supported():
        return True, "exact Pokemon Showdown snapshot backend"
    if ALLOW_APPROXIMATE_SEARCH:
        return True, "UNSAFE legacy approximate backend explicitly enabled"
    return (
        False,
        "exact Showdown bridge exists, but live poke-env state parity is not yet "
        "established; approximate search is disabled",
    )


# Per-decision latencies in ms, for reporting. Bounded so a long ladder run cannot
# grow it without limit.
SEARCH_LATENCIES_MS: list[float] = []


def _record_latency(ms: float) -> None:
    if len(SEARCH_LATENCIES_MS) < 5000:
        SEARCH_LATENCIES_MS.append(ms)


def latency_summary() -> str:
    """p50/p90/max of observed search latency, or a note that none was observed."""
    if not SEARCH_LATENCIES_MS:
        return "search latency: no decisions recorded"
    xs = sorted(SEARCH_LATENCIES_MS)

    def pct(p: float) -> float:
        return xs[min(len(xs) - 1, int(len(xs) * p))]

    return (
        f"search latency over {len(xs)} decisions: "
        f"p50={pct(0.5):.0f}ms  p90={pct(0.9):.0f}ms  max={xs[-1]:.0f}ms"
    )


def _apply_damage(defender: Pokemon, fraction: float) -> None:
    """Reduce HP by a fraction of max, marking a faint at zero."""
    if defender.fainted:
        return
    cur = defender.current_hp_fraction or 0.0
    new = max(0.0, cur - fraction)
    max_hp = defender.max_hp or 100
    defender._current_hp = int(round(new * max_hp))
    if new <= 0:
        defender._status = None
        defender._current_hp = 0


def _apply_move_effects(battle: DoubleBattle, user: Pokemon, move: Move, ours: bool):
    """Apply the non-damage part of a move: side conditions and self boosts.

    Deliberately narrow. These two cover the cases that change a position's value
    without dealing damage -- Tailwind, screens, Swords Dance -- which is exactly the
    class the bot was blind to.
    """
    entry = getattr(move, "entry", None) or {}
    side_cond = entry.get("sideCondition")
    if side_cond:
        try:
            cond = SideCondition.from_showdown_message(side_cond)
            target = battle.side_conditions if ours else battle.opponent_side_conditions
            target.setdefault(cond, battle.turn)
        except Exception:
            pass
    boosts = entry.get("boosts") or {}
    if boosts and entry.get("target") == "self":
        for stat, amount in boosts.items():
            if stat in user.boosts:
                user._boosts[stat] = max(-6, min(6, user.boosts[stat] + int(amount)))


def _effective_speed(mon: Pokemon, battle: DoubleBattle, ours: bool) -> float:
    if mon is None or mon.fainted:
        return -1.0
    spe = (mon.stats or {}).get("spe") or 0
    boost = mon.boosts.get("spe", 0)
    spe *= (2 + boost) / 2 if boost >= 0 else 2 / (2 - boost)
    conds = battle.side_conditions if ours else battle.opponent_side_conditions
    if any(c.name == "TAILWIND" for c in conds):
        spe *= 2
    return spe


def simulate_turn(battle: DoubleBattle, ours, theirs) -> DoubleBattle:
    """Play one turn forward under both sides' chosen actions.

    `ours`/`theirs` are lists of (attacker, move, targets) resolved by the caller.
    Actions resolve in speed order so a KO can deny the slower side its move -- the
    single most important thing a damage-only model gets wrong.
    """
    sim = deepcopy(battle)

    def remap(mon, table):
        # deepcopy breaks identity, so re-resolve every Pokemon by its key.
        key = K.identifier(battle, mon)
        return table.get(key) if key else None

    acts = []
    for attacker, move, targets, is_ours in [(a, m, t, True) for a, m, t in ours] + [
        (a, m, t, False) for a, m, t in theirs
    ]:
        table = sim.team if is_ours else sim.opponent_team
        opp_table = sim.opponent_team if is_ours else sim.team
        s_att = remap(attacker, table)
        s_tgt = [remap(t, opp_table) for t in targets]
        if s_att is None:
            continue
        acts.append(
            (
                s_att,
                move,
                [t for t in s_tgt if t],
                is_ours,
                move.priority,
                _effective_speed(s_att, sim, is_ours),
            )
        )

    acts.sort(key=lambda a: (a[4], a[5]), reverse=True)

    for s_att, move, s_targets, is_ours, _prio, _spe in acts:
        if s_att.fainted:
            continue
        if move.category == MoveCategory.STATUS:
            _apply_move_effects(sim, s_att, move, is_ours)
            continue
        for tgt in s_targets:
            frac = K.damage_fraction(sim, s_att, tgt, move)
            if frac is None:
                continue
            expected = (frac[0] + frac[1]) / 2 * (move.accuracy or 1.0)
            _apply_damage(tgt, expected)
    return sim


def _evaluate(policy, battle: DoubleBattle) -> float:
    """Critic's win-probability estimate for this position."""
    from vgc_bench.src.policy_player import PolicyPlayer

    obs = PolicyPlayer.embed_battle(battle, fake_rating=2000)
    with torch.no_grad():
        obs_dict = {
            "observation": torch.as_tensor(obs, device=policy.device).unsqueeze(0),
            "action_mask": torch.ones(1, 214, device=policy.device),
        }
        _logits, value = policy.get_logits(obs_dict, actor_grad=False)
    return float(value.item())


def _resolve(battle: DoubleBattle, actions, ours: bool):
    """(slot0, slot1) action ints -> [(attacker, move, targets)]."""
    out = []
    actives = battle.active_pokemon if ours else battle.opponent_active_pokemon
    for pos, action in enumerate(actions):
        attacker = actives[pos] if pos < len(actives) else None
        if attacker is None or attacker.fainted:
            continue
        order = _guards._decode(battle, action, pos)
        move, hit = _guards._move_and_targets(battle, order, pos)
        if move is None:
            continue
        out.append((attacker, move, hit))
    return out


def _their_candidates(battle: DoubleBattle, k: int, move_predictions=None):
    """Plausible opponent actions ranked by learned human choices when available."""
    from vgc_bench.src.policy_player import PolicyPlayer

    foes = [f for f in battle.opponent_active_pokemon if f and not f.fainted]
    ours = [m for m in battle.active_pokemon if m and not m.fainted]
    if not foes or not ours:
        return []

    per_slot = []
    for slot, foe in enumerate(foes):
        known = list(foe.moves.values())[:4]
        prior = (
            PolicyPlayer._moveset_prior(foe)
            if PolicyPlayer.moveset_prior_enabled()
            else None
        )
        if prior and len(known) < 4:
            seen = {m.id for m in known}
            for mid in prior.get("moves", []):
                if len(known) >= 4:
                    break
                if mid not in seen:
                    try:
                        known.append(Move(mid, gen=9))
                        seen.add(mid)
                    except Exception:
                        pass
        scored = []
        predicted_moves = (
            dict(move_predictions[slot].moves) if move_predictions is not None else {}
        )
        predicted_targets = (
            dict(move_predictions[slot].targets) if move_predictions is not None else {}
        )
        for mv in known:
            if mv.category == MoveCategory.STATUS:
                learned = predicted_moves.get(mv.id, 0.0)
                scored.append((learned if predicted_moves else 0.15, mv, [ours[0]]))
                continue
            for target_slot, target in enumerate(ours):
                frac = K.damage_fraction(battle, foe, target, mv)
                val = ((frac[0] + frac[1]) / 2 * (mv.accuracy or 1.0)) if frac else 0.0
                if predicted_moves:
                    # The replay model scores the actual move and its intended target;
                    # damage is only a tiny tie-break. This replaces the old hidden
                    # assumption that every opponent greedily maximizes immediate HP.
                    target_name = "foe_a" if target_slot == 0 else "foe_b"
                    val = (
                        predicted_moves.get(mv.id, 0.0)
                        * predicted_targets.get(target_name, 0.5)
                        + val * 1e-3
                    )
                scored.append((val, mv, [target]))
        scored.sort(key=lambda s: -s[0])
        per_slot.append([(foe, mv, tg) for _v, mv, tg in scored[:k]])

    combos = []
    if len(per_slot) == 1:
        combos = [[a] for a in per_slot[0]]
    else:
        for a in per_slot[0][:k]:
            for b in per_slot[1][:k]:
                combos.append([a, b])
    return combos[: k * k]


def search_action(
    policy,
    battle: DoubleBattle,
    obs_dict,
    mask,
    opponent_moves=None,
    opponent_switches=None,
) -> np.ndarray | None:
    """Pick a joint action by one-ply matrix-game search. None if not applicable.

    Thin wrapper so every exit path is timed, including the early returns and the
    exception path -- an untimed bail-out is how a latency claim goes unverified.
    """
    if not ALLOW_APPROXIMATE_SEARCH:
        return None
    start = time.perf_counter()
    try:
        return _search_action(
            policy, battle, obs_dict, mask, start, opponent_moves, opponent_switches
        )
    finally:
        _record_latency((time.perf_counter() - start) * 1000)


def _search_action(
    policy,
    battle: DoubleBattle,
    obs_dict,
    mask,
    start: float,
    opponent_moves=None,
    opponent_switches=None,
) -> np.ndarray | None:
    # The exact backend will add voluntary-switch branches with this prior. The
    # legacy hand-mutated simulator cannot model switching and remains disabled.
    _ = opponent_switches
    cands, _v = _guards.build_candidates(policy, obs_dict, mask, top_k=OUR_K)
    if len(cands) < 2:
        return None
    cands, _report = _guards.apply_guards(battle, cands)
    # Guards DEMOTE rather than delete, so a vetoed pair is still in the list -- and a
    # positional slice would hand it back to the search, which then ranks purely on
    # critic value and can pick it. That silently undoes every hard veto. Search is a
    # soft evaluator; the vetoes constrain it, not the other way round.
    live = [c for c in cands if c.demoted_by is None]
    ours_top = (live or cands)[:OUR_K]  # stand down if everything was vetoed
    theirs = _their_candidates(battle, THEIR_K, opponent_moves)
    if not theirs:
        return np.array(ours_top[0].actions, dtype=np.int64)

    # Opening decisions are made once and shape the whole game; later turns pay their
    # cost every turn and burn the ladder clock, which is the asymmetry that made the
    # bot sit there thinking on turn after turn.
    deadline = start + (
        SEARCH_OPENING_BUDGET_S if battle.turn <= 1 else SEARCH_BUDGET_S
    )
    payoff = np.zeros((len(ours_top), len(theirs)), dtype=np.float64)
    rows = 0
    for i, cand in enumerate(ours_top):
        # Anytime: stop starting new rows once out of time, but never below the two
        # rows a matrix game needs. A partly-filled row would be scored against
        # zeros, so the check sits between rows rather than inside one.
        if rows >= 2 and time.perf_counter() >= deadline:
            from vgc_bench.src.policy_player import PolicyPlayer

            PolicyPlayer.guard_fire_counts["search_truncated"] += 1
            break
        our_resolved = _resolve(battle, cand.actions, ours=True)
        for j, their_resolved in enumerate(theirs):
            successor = simulate_turn(battle, our_resolved, their_resolved)
            payoff[i, j] = _evaluate(policy, successor)
        rows += 1
    payoff = payoff[:rows]
    ours_top = ours_top[:rows]
    if rows < 2:
        return None  # hand back to the guard stack rather than guess off one row

    # Simultaneous moves: solve for a strategy rather than taking an argmax, which
    # a competent opponent could read and punish.
    try:
        strategy = Game(payoff).linear_program()[0]
        strategy = np.asarray(strategy, dtype=np.float64).ravel()
        if strategy.size != len(ours_top) or not np.isfinite(strategy).all():
            raise ValueError("degenerate solve")
        strategy = np.clip(strategy, 0, None)
        total = strategy.sum()
        strategy = strategy / total if total > 0 else None
    except Exception:
        strategy = None

    if strategy is None:  # fall back to maximin: best worst case
        idx = int(np.argmax(payoff.min(axis=1)))
    else:
        idx = int(np.argmax(strategy))
    return np.array(ours_top[idx].actions, dtype=np.int64)
