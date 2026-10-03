"""Team preview that reads the opponent's open team sheet (the user, 2026-10-03: "i
need it to also factor in things it knows from open team sheets when it has them").

The deployed preview model (PreviewPredictor, data/preview_t6_focus_*.pt) ranks
our four and leads from the twelve species alone; it learned from human choices,
so it already reflects the sets humans usually face. What an open sheet adds is
how THIS opponent's sets differ from those usual sets (Choice Specs instead of a
Fairy Feather, no Fake Out, a Wide Guard where few carry it, ...). So the sheet
moves a plan only by that difference:

  for each of the model's top TOP_K plans (all practised openings)
      delta = sum over their likely lead pairs (the opponent model's weights) of
              exchange(our leads, their leads | their sheet sets)
            - exchange(our leads, their leads | their usual sets)
      score = log(model probability) + WEIGHT * delta

and the best score is played. ``exchange`` is a turn-1 damage race: each lead's
strongest attack (calculator; spread moves at 0.75 and both foes, capped at a
foe's HP), ours minus theirs, with a foe's Fake Out costing the hit side half of
its best lead (unless an Armor Tail / Queenly Majesty / Dazzling ally blocks it)
and Wide Guard halving the other side's spread damage. Scoring the whole race
directly would undervalue support openings (a Trick Room lead deals little on
turn 1); the difference cancels that bias and keeps only the sheet's
information. When the sheet matches the usual sets the model's plan stands.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any

from poke_env.battle import DoubleBattle, Move, MoveCategory, Pokemon, Weather
from poke_env.battle.move import MoveSet
from poke_env.battle.target import Target
from poke_env.data import to_id_str

from vgc_bench.src import vgc_knowledge as K
from vgc_bench.src.opponent_preview import PreviewPlan
from vgc_bench.src.playbook import _sets

TOP_K = 6  # only the model's top plans: openings the brain has practised
WEIGHT = 2.0  # log-probability per unit of turn-1 HP race
SPREAD = (Target.ALL_ADJACENT_FOES, Target.ALL_ADJACENT)
PRIORITY_BLOCKERS = frozenset({"armortail", "queenlymajesty", "dazzling"})
WEATHER_ABILITIES = {
    "drought": Weather.SUNNYDAY,
    "drizzle": Weather.RAINDANCE,
    "sandstream": Weather.SANDSTORM,
    "snowwarning": Weather.SNOW,
}
FAKE_OUT_SHARE = 0.5  # a Fake Out costs the hit side half of its best lead's hit
WIDE_GUARD_SHARE = 0.5  # Wide Guard halves the other side's spread damage


def _format(battle: DoubleBattle) -> str | None:
    tag = re.match(r"battle-([a-z0-9]+)-", getattr(battle, "battle_tag", "") or "")
    return battle.format or (tag.group(1) if tag else None)


def sheet_open(battle: DoubleBattle) -> bool:
    """Every opposing Pokemon shows an ability and four moves (an open sheet)."""
    team = list(battle.opponent_team.values())
    return len(team) == 6 and all(
        mon.ability and len(mon.moves or {}) >= 4 for mon in team
    )


def _usual_set(battle: DoubleBattle, mon: Pokemon) -> dict[str, Any] | None:
    entry = _sets(_format(battle)).get(to_id_str(mon.base_species)) or {}
    return max(entry.get("sets", []), key=lambda s: s.get("prob", 0), default=None)


@contextmanager
def usual_sets(battle: DoubleBattle) -> Generator[None]:
    """Temporarily give every opposing Pokemon its species' most-used set."""
    saved = []
    for mon in battle.opponent_team.values():
        usual = _usual_set(battle, mon)
        if usual is None:
            continue
        moves = {}
        for move_id in usual.get("moves") or []:
            try:
                moves[to_id_str(move_id)] = Move(to_id_str(move_id), gen=9)
            except Exception:
                continue
        saved.append((mon, mon._moves, mon._item, mon._ability))
        mon._moves = MoveSet(moves)
        mon._item = to_id_str(usual.get("item") or "") or mon._item
        mon._ability = to_id_str(usual.get("ability") or "") or mon._ability
    try:
        yield
    finally:
        for mon, moves, item, ability in saved:
            mon._moves, mon._item, mon._ability = moves, item, ability


def _speed(mon: Pokemon) -> float:
    return float((mon.stats or {}).get("spe") or mon.base_stats.get("spe", 0))


def lead_weather(ours: Sequence[Pokemon], theirs: Sequence[Pokemon]) -> Weather | None:
    """The weather after both pairs switch in: setters activate fastest first, so
    the slowest setter's weather stays."""
    setters = [
        (_speed(mon), WEATHER_ABILITIES[to_id_str(mon.ability or "")])
        for mon in (*ours, *theirs)
        if to_id_str(mon.ability or "") in WEATHER_ABILITIES
    ]
    return min(setters, key=lambda s: s[0])[1] if setters else None


@contextmanager
def weather(battle: DoubleBattle, kind: Weather | None) -> Generator[None]:
    saved = battle._weather
    battle._weather = {kind: 0} if kind is not None else {}
    try:
        yield
    finally:
        battle._weather = saved


def _attacks(mon: Pokemon) -> list[Move]:
    return [
        m
        for m in (mon.moves or {}).values()
        if isinstance(m, Move) and m.category != MoveCategory.STATUS and m.base_power
    ]


def _has(mon: Pokemon, move_id: str) -> bool:
    return move_id in (mon.moves or {})


def _offense(battle: DoubleBattle, attacker: Pokemon, foes: Sequence[Pokemon]):
    """(best turn-1 damage, the spread part of it) of one lead into a foe pair."""
    K.ensure_stats(attacker)
    best = (0.0, 0.0)
    for move in _attacks(attacker):
        spread = move.target in SPREAD
        hits = []
        for foe in foes:
            K.ensure_stats(foe)
            fraction = K.damage_fraction(battle, attacker, foe, move)
            if fraction is None:
                continue
            mean = (fraction[0] + fraction[1]) / 2 * (0.75 if spread else 1.0)
            accuracy = move.accuracy if isinstance(move.accuracy, float) else 1.0
            hits.append(min(mean, 1.0) * min(accuracy, 1.0))
        if not hits:
            continue
        value = sum(hits) if spread else max(hits)
        if value > best[0]:
            best = (value, value if spread else 0.0)
    return best


def _side(battle, attackers, defenders) -> float:
    """A side's turn-1 damage into the other pair, after the other side's Fake Out
    and Wide Guard."""
    offense = [_offense(battle, mon, defenders) for mon in attackers]
    total = sum(value for value, _ in offense)
    if any(_has(mon, "wideguard") for mon in defenders):
        total -= WIDE_GUARD_SHARE * sum(spread for _, spread in offense)
    blocked = any(
        to_id_str(mon.ability or "") in PRIORITY_BLOCKERS for mon in attackers
    )
    if not blocked and any(_has(mon, "fakeout") for mon in defenders):
        total -= FAKE_OUT_SHARE * max((value for value, _ in offense), default=0.0)
    return total


def exchange(battle: DoubleBattle, ours: Sequence[Pokemon], theirs: Sequence[Pokemon]):
    """Our turn-1 damage race against their leads (ours minus theirs)."""
    with weather(battle, lead_weather(ours, theirs)):
        return _side(battle, ours, theirs) - _side(battle, theirs, ours)


def their_lead_weights(
    their_plans: Sequence[PreviewPlan],
) -> dict[tuple[int, int], float]:
    weights: dict[tuple[int, int], float] = defaultdict(float)
    for plan in their_plans:
        weights[tuple(sorted(plan.lead_indices))] += plan.probability  # type: ignore[index]
    total = sum(weights.values()) or 1.0
    return {lead: w / total for lead, w in weights.items()}


def rerank(
    battle: DoubleBattle,
    our_plans: Sequence[PreviewPlan],
    their_plans: Sequence[PreviewPlan],
    top_k: int = TOP_K,
    weight: float = WEIGHT,
) -> tuple[PreviewPlan, dict[str, Any]]:
    """The model plan the open sheet favours most (see the module doc), with an
    audit of every plan's model log-probability and sheet difference."""
    ours = list(battle.team.values())
    theirs = list(battle.opponent_team.values())
    plans = list(our_plans[:top_k])
    leads = sorted(their_lead_weights(their_plans).items(), key=lambda kv: -kv[1])[:8]
    norm = sum(w for _, w in leads) or 1.0
    audit: dict[str, Any] = {"plans": []}
    best, best_score = plans[0], -math.inf
    for plan in plans:
        mine = [ours[i] for i in plan.lead_indices]
        delta = 0.0
        for (a, b), w in leads:
            pair = [theirs[a], theirs[b]]
            on_sheet = exchange(battle, mine, pair)
            with usual_sets(battle):
                usual = exchange(battle, mine, pair)
            delta += w / norm * (on_sheet - usual)
        score = math.log(max(plan.probability, 1e-9)) + weight * delta
        audit["plans"].append(
            {
                "lead": [to_id_str(ours[i].species) for i in plan.lead_indices],
                "back": [to_id_str(ours[i].species) for i in plan.back_indices],
                "probability": round(plan.probability, 4),
                "sheet_delta": round(delta, 4),
                "score": round(score, 4),
            }
        )
        if score > best_score:
            best, best_score = plan, score
    audit["chosen_rank"] = plans.index(best) + 1
    return best, audit
