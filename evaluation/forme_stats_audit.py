"""How much the stale forme stats matter on our own ladder games (2026-10-04).

``vgc_knowledge.ensure_stats`` estimates an opposing Pokemon's stats once, in the
forme it is first seen in; a foe that Mega-evolves keeps its pre-Mega numbers
(vgc_bench/src/forme_stats.py). This audit measures that on every saved Reg M-C game
before any behaviour changes. Each game is rebuilt turn by turn as the live bot held
it (evaluation/oppmodel_oracle.py's ``Position``: the opposing actives get their
estimate at every turn start, which is what freezes the old line), from the saved
player-view page; nothing heavier than the CPU damage calculator and the bot's three
small prediction nets runs. No brain, no server, no MPS.

Three readings:

1. EXPOSURE. Turns and logged move decisions with an opposing Pokemon on the field
   whose stat line was made for another of its formes, by species.

2. ACCURACY, against what really happened. Every clean damaging hit of the game
   that such a Pokemon dealt or took is compared with the calculator's range at the
   start of that turn, once with the stored line and once with the line for the
   forme it is in (``forme_stats``). A hit is clean when nothing the calculator
   could not know at the turn start touched it: no critical hit, no multi-hit, no
   knock-out (the damage is cut off), no resist berry, no Helping Hand, no boost /
   status / item / forme change on either Pokemon earlier in the turn, no switch,
   faint, weather, terrain or screen change before the move, and for a move whose
   power follows the user's HP, no HP change of the user before it. Hits between
   our Pokemon and opposing Pokemon whose line is NOT stale are the control: the
   same estimate (32 HP, 32 in the better attack, neutral nature) where the forme
   is right.

3. DECISIONS. The logged candidates of every move decision (the audit keeps the
   first eight after the whole pipeline) are put back in the policy's order and go
   through the guard stack with today's deployed profile, on the stored numbers
   and with ``forme_stats`` on: picks that differ, which guards' verdicts moved,
   each guard alone, then the reranker and sticky corrections after it (on the
   stored numbers as deployed, and on the right ones), and how far the
   observation's knowledge and threat blocks would move if they were recomputed
   too (they are NOT touched by ``forme_stats``; the number says what the brain
   would see if they were).

Limits, stated once: eight candidates of up to 64 (a guard that would promote or
release a pair outside the eight is invisible, so pick changes are undercounted);
``strategic_only`` is not logged; a saved page never carries a team sheet, so every
game is rebuilt as a hidden-sheet game; the stack's own profile reproduces the
logged pick only as often as the summary says.

Outputs under ``results_analysis/forme_stats_20261004/``: ``summary.json``,
``decisions.jsonl`` (exposed move decisions), ``hits.jsonl``, ``report.txt``.
Usage, from the repo root, one core:

    nice -n 19 .venv/bin/python evaluation/forme_stats_audit.py
"""

from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path as _Path

_os.environ.setdefault("OMP_NUM_THREADS", "1")
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import logging
import math
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from poke_env.battle import DoubleBattle, Move, Pokemon
from poke_env.data import GenData, to_id_str
from poke_env.teambuilder import Teambuilder

from evaluation import oppmodel_oracle as O
from vgc_bench.src import forme_stats as F
from vgc_bench.src import guards as G
from vgc_bench.src import vgc_knowledge as K
from vgc_bench.src.oppmodel import events as E
from vgc_bench.src.policy_player import PolicyPlayer

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results_analysis" / "forme_stats_20261004"
GLOBS = ("ladder_replays_mc*", "challenge_replays_mc*")
DEPLOYED = ROOT / "results_deployed" / "DEPLOYED.json"

# power follows the user's remaining HP: the turn-start range is wrong once it is hit
USER_HP_MOVES = frozenset(
    {"waterspout", "eruption", "dragonenergy", "flail", "reversal"}
)
# ... or the target's
TARGET_HP_MOVES = frozenset({"brine", "crushgrip", "hardpress", "wringout"})
_BOOST_KINDS = frozenset(
    {
        "-boost",
        "-unboost",
        "-setboost",
        "-clearboost",
        "-clearpositiveboost",
        "-clearnegativeboost",
        "-copyboost",
        "-swapboost",
        "-invertboost",
    }
)
# anything here on a Pokemon earlier in the turn: its numbers are not the turn start's
_CHANGE_KINDS = _BOOST_KINDS | frozenset(
    {
        "-status",
        "-curestatus",
        "-item",
        "-enditem",
        "-mega",
        "-primal",
        "detailschange",
        "-formechange",
        "-transform",
        "-start",
        "-end",
        "-singleturn",
        "-singlemove",
        "-endability",
        "-terastallize",
    }
)
_BOARD_KINDS = frozenset(
    {
        "switch",
        "drag",
        "replace",
        "faint",
        "-fieldstart",
        "-fieldend",
        "-sidestart",
        "-sideend",
        "-swapsideconditions",
        "-clearallboost",
    }
)
_HARMLESS_FIELDS = frozenset({"trickroom"})
# flags of the two observation blocks (0/1 facts, not magnitudes)
_KNOWLEDGE_FLAGS = tuple(m * 5 + k for m in range(4) for k in (2, 3)) + (21, 22)
_THREAT_FLAGS = (2, 3, 4, 5)


@lru_cache(maxsize=None)
def _forme_index() -> dict[str, list[tuple[str, dict[str, int]]]]:
    out: dict[str, list[tuple[str, dict[str, int]]]] = defaultdict(list)
    for key, entry in GenData.from_gen(9).pokedex.items():
        if entry.get("baseStats"):
            root = to_id_str(entry.get("baseSpecies") or key)
            out[root].append((key, entry["baseStats"]))
    return out


def forme_key(mon: Pokemon) -> str:
    """The pokedex id of the forme ``mon`` is in (poke-env keeps the species it
    Mega-evolved from until it next switches in)."""
    for key, base in _forme_index().get(mon.base_species, ()):
        if base == mon.base_stats:
            return key
    return mon.species


# --- hits of a turn -----------------------------------------------------------------


@dataclass
class Hit:
    """One clean damaging hit: who, with what, and the share of the defender's
    maximum HP it really took."""

    attacker: str
    defender: str
    move: str
    realized: float


def team_key(ident: str) -> str | None:
    """``p2a: Raichu`` -> ``p2: Raichu``, the key of poke-env's team dicts."""
    parsed = E.parse_ident(ident)
    if parsed is None or not parsed.name:
        return None
    return f"{parsed.side}: {parsed.name}"


def _hp(text: str) -> tuple[int, int] | None:
    """``120/186 par`` -> (120, 186); ``0 fnt`` -> (0, 0)."""
    head = text.split(" ")[0]
    if head == "0":
        return 0, 0
    if "/" not in head:
        return None
    try:
        current, maximum = head.split("/")
        return int(current), int(maximum)
    except ValueError:
        return None


@dataclass
class _Pending:
    attacker: str
    move: str
    usable: bool
    damage: dict[str, list[tuple[tuple[int, int], tuple[int, int], bool]]] = field(
        default_factory=dict
    )
    spoiled: set[str] = field(default_factory=set)
    multi: bool = False


def read_hits(
    events: Sequence[Sequence[str]], start: int, hp: Mapping[str, tuple[int, int]]
) -> tuple[list[Hit], Counter[str]]:
    """The clean hits of the turn that begins at ``events[start]`` (its ``|turn|``
    line). ``hp`` holds (current, maximum) of every Pokemon on the field at that
    moment, by team key; only hits between two of those are returned."""
    present = set(hp)
    now = dict(hp)
    changed: set[str] = set()  # numbers moved this turn
    touched: set[str] = set()  # HP moved this turn
    board = False  # the field is no longer the turn start's
    hits: list[Hit] = []
    why: Counter[str] = Counter()
    pending: _Pending | None = None

    def flush() -> None:
        nonlocal pending
        if pending is None:
            return
        for defender, rows in pending.damage.items():
            if not pending.usable:
                why["attacker_or_board_changed"] += 1
            elif pending.multi or len(rows) != 1:
                why["multi_hit"] += 1
            elif defender in pending.spoiled or rows[0][2]:
                why["defender_changed_or_crit"] += 1
            elif defender not in present:
                why["defender_not_there_at_turn_start"] += 1
            else:
                before, after, _ = rows[0]
                if after[0] <= 0:
                    why["knock_out"] += 1
                elif before[1] <= 0 or before[0] <= after[0]:
                    why["no_hp_drop"] += 1
                else:
                    hits.append(
                        Hit(
                            pending.attacker,
                            defender,
                            pending.move,
                            (before[0] - after[0]) / before[1],
                        )
                    )
        pending = None

    for event in events[start + 1 :]:
        kind = event[1] if len(event) > 1 else ""
        if kind in ("turn", "win", "tie"):
            break
        key = team_key(event[2]) if len(event) > 2 else None
        if kind == "move":
            flush()
            if key is None or len(event) < 4:
                continue
            move = E.to_id(event[3])
            usable = (
                key in present
                and key not in changed
                and not board
                and not (move in USER_HP_MOVES and key in touched)
            )
            pending = _Pending(key, move, usable)
        elif kind in ("-damage", "-heal", "-sethp"):
            new = _hp(event[3]) if len(event) > 3 else None
            if key is None or new is None:
                continue
            old = now.get(key)
            direct = kind == "-damage" and E.tag_value(event, "[from]") is None
            if direct and pending is not None and old is not None:
                unclean = key in changed or (
                    pending.move in TARGET_HP_MOVES and key in touched
                )
                pending.damage.setdefault(key, []).append((old, new, unclean))
            now[key] = new if new[1] else (0, old[1] if old else 0)
            touched.add(key)
        elif kind == "-crit" or kind == "-hitcount":
            if pending is not None:
                if kind == "-hitcount":
                    pending.multi = True
                elif key is not None:
                    pending.spoiled.add(key)
        elif kind in _CHANGE_KINDS:
            if key is not None:
                changed.add(key)
                if pending is not None and kind == "-enditem":
                    pending.spoiled.add(key)  # a resist berry, eaten before the hit
            if kind in ("-swapboost", "-copyboost") and len(event) > 3:
                other = team_key(event[3])
                if other is not None:
                    changed.add(other)
        elif kind == "-weather":
            if not any(part.startswith("[upkeep]") for part in event[3:]):
                flush()
                board = True
        elif kind in _BOARD_KINDS:
            if kind in ("-fieldstart", "-fieldend") and (
                E.effect_id(event[2]) in _HARMLESS_FIELDS
            ):
                continue
            if kind != "faint":
                flush()
            board = True
            if kind in ("switch", "drag", "replace") and key is not None:
                present.discard(key)
    flush()
    return hits, why


# --- rebuilding ----------------------------------------------------------------------


def field_hp(battle: DoubleBattle) -> dict[str, tuple[int, int]]:
    """(current, maximum) HP of the Pokemon on the field, by team key."""
    out: dict[str, tuple[int, int]] = {}
    for table in (battle.team, battle.opponent_team):
        for key, mon in table.items():
            if mon.active and not mon.fainted and mon.max_hp:
                out[key] = (int(mon.current_hp or 0), int(mon.max_hp))
    return out


def _mon(battle: DoubleBattle, key: str) -> Pokemon | None:
    return battle.team.get(key) or battle.opponent_team.get(key)


def _range(
    battle: DoubleBattle, attacker: Pokemon, defender: Pokemon, move_id: str
) -> tuple[float, float] | None:
    move = attacker.moves.get(move_id)
    if move is None:
        try:
            move = Move(move_id, gen=9)
        except Exception:
            return None
    fraction = K.damage_fraction(battle, attacker, defender, move)
    if fraction is None or fraction[1] <= 0:
        return None
    return fraction


def score_hits(
    battle: DoubleBattle,
    hits: Sequence[Hit],
    stale: Sequence[tuple[Pokemon, F.Stats]],
    ours: str,
) -> list[dict[str, Any]]:
    """Each hit between one of ours and one of theirs with the calculator's range on
    the stored numbers and, where a stale line is involved, on the right ones."""
    stale_ids = {id(mon) for mon, _ in stale}
    rows: list[dict[str, Any]] = []
    for hit in hits:
        attacker, defender = _mon(battle, hit.attacker), _mon(battle, hit.defender)
        if attacker is None or defender is None:
            continue
        ours_attacks = hit.attacker.startswith(ours)
        if ours_attacks == hit.defender.startswith(ours):
            continue  # an ally hit by a spread move: no opposing estimate in it
        foe = defender if ours_attacks else attacker
        stored = _range(battle, attacker, defender, hit.move)
        if stored is None:
            continue
        row: dict[str, Any] = {
            "direction": "into_foe" if ours_attacks else "by_foe",
            "foe": forme_key(foe),
            "move": hit.move,
            "realized": hit.realized,
            "stored": list(stored),
            "stale": id(foe) in stale_ids,
        }
        if row["stale"]:
            with F.swapped(list(stale)):
                fresh = _range(battle, attacker, defender, hit.move)
            if fresh is None:
                continue
            row["fresh"] = list(fresh)
        rows.append(row)
    return rows


# --- one decision --------------------------------------------------------------------


def _copies(cands: Sequence[G.Candidate]) -> list[G.Candidate]:
    return [replace(c) for c in cands]


def _signature(cands: Sequence[G.Candidate]) -> list[tuple[tuple[int, int], Any]]:
    return [(c.actions, c.demoted_by) for c in cands]


def _label(battle: DoubleBattle, actions: Sequence[int]) -> list[str]:
    out = []
    for pos, action in enumerate(actions):
        order = G._decode(battle, int(action), pos)
        what = getattr(order, "order", None)
        name = getattr(what, "id", None) or getattr(what, "species", None) or "?"
        target = getattr(order, "move_target", 0)
        foes = battle.opponent_active_pokemon
        foe = foes[target - 1] if target in (1, 2) and target - 1 < len(foes) else None
        aim = ">" + foe.species if foe is not None else ""
        mega = "+mega" if getattr(order, "mega", False) else ""
        out.append(f"{name}{mega}{aim}")
    return out


def _blocks(battle: DoubleBattle) -> dict[str, list[float]]:
    """The observation's knowledge and threat blocks of the Pokemon on the field."""
    PolicyPlayer._knowledge_cache.clear()
    PolicyPlayer._threat_cache.clear()
    knowledge = PolicyPlayer._knowledge_for(battle)
    threat = PolicyPlayer._threat_for(battle)
    out: dict[str, list[float]] = {}
    mons = [*battle.active_pokemon, *battle.opponent_active_pokemon]
    for index, mon in enumerate(mons):
        if mon is None:
            continue
        if id(mon) in knowledge:
            out[f"k{index}"] = [float(v) for v in knowledge[id(mon)]]
        if id(mon) in threat:
            out[f"t{index}"] = [float(v) for v in threat[id(mon)]]
    PolicyPlayer._knowledge_cache.clear()
    PolicyPlayer._threat_cache.clear()
    return out


def observation_shift(
    battle: DoubleBattle, stale: Sequence[tuple[Pokemon, F.Stats]]
) -> dict[str, Any]:
    """How the two blocks would differ if they were computed on the right lines."""
    stored = _blocks(battle)
    with F.swapped(list(stale)):
        fresh = _blocks(battle)
    floats = moved = flags = 0
    largest = total = 0.0
    for name, old in stored.items():
        new = fresh.get(name)
        if new is None or len(new) != len(old):
            continue
        flag_index = _KNOWLEDGE_FLAGS if name.startswith("k") else _THREAT_FLAGS
        for index, (a, b) in enumerate(zip(old, new)):
            floats += 1
            if abs(a - b) > 1e-6:
                moved += 1
                largest = max(largest, abs(a - b))
                total += abs(a - b)
                if index in flag_index:
                    flags += 1
    ours = []
    for index, mon in enumerate(battle.active_pokemon):
        if mon is None or mon.fainted:
            continue
        threat = [stored.get(f"t{index}"), fresh.get(f"t{index}")]
        know = [stored.get(f"k{index}"), fresh.get(f"k{index}")]
        if None in threat:
            continue
        ours.append(
            {
                "key": K.identifier(battle, mon),
                # threat block [2..3]: a foe has a move whose MAX roll removes it
                "can_be_koed": [
                    max((block or [0.0] * 4)[2:4]) > 0.5 for block in threat
                ],
                # knowledge block [21]: a foe's best move removes it on average
                "expected_ko": [
                    bool(block is not None and block[21] > 0.5) for block in know
                ],
            }
        )
    return {
        "floats": floats,
        "moved": moved,
        "flag_flips": flags,
        "largest": largest,
        "sum_abs": total,
        "ours": ours,
    }


def faints(events: Sequence[Sequence[str]], start: int) -> set[str]:
    """Team keys of the Pokemon that fainted in the turn beginning at ``start``."""
    out: set[str] = set()
    for event in events[start + 1 :]:
        kind = event[1] if len(event) > 1 else ""
        if kind in ("turn", "win", "tie"):
            break
        if kind == "faint" and len(event) > 2:
            key = team_key(event[2])
            if key is not None:
                out.add(key)
    return out


def attack_details(
    battle: DoubleBattle,
    actions: Sequence[int],
    stale: Sequence[tuple[Pokemon, F.Stats]],
) -> list[dict[str, Any]]:
    """Each attack of a pair into the foes it hits: the foe's HP and the damage
    range on the stored numbers and on the right ones."""
    rows = []
    for pos, action in enumerate(actions):
        me = battle.active_pokemon[pos]
        order = G._decode(battle, int(action), pos)
        move, hit = G._move_and_targets(battle, order, pos)
        if me is None or move is None:
            continue
        for foe in hit:
            if foe is None or foe in battle.active_pokemon:
                continue
            stored = K.damage_fraction(battle, me, foe, move)
            with F.swapped(list(stale)):
                fresh = K.damage_fraction(battle, me, foe, move)
            rows.append(
                {
                    "move": move.id,
                    "target": forme_key(foe),
                    "hp": round(float(foe.current_hp_fraction or 0.0), 3),
                    "stored": None if stored is None else [round(v, 3) for v in stored],
                    "fresh": None if fresh is None else [round(v, 3) for v in fresh],
                }
            )
    return rows


def ko_claims(
    battle: DoubleBattle,
    actions: Sequence[int],
    stale: Sequence[tuple[Pokemon, F.Stats]],
    fainted: set[str],
) -> list[dict[str, Any]]:
    """For each stale foe a pair attacks: the pair's summed MINIMUM rolls into it on
    the stored numbers and on the right ones, its HP, and whether it went down."""
    rows = []
    for foe, _ in stale:
        if not foe.active or foe.fainted:
            continue
        totals = []
        for swap in (False, True):
            total, aimed = 0.0, False
            with F.swapped(list(stale) if swap else []):
                for pos, action in enumerate(actions):
                    me = battle.active_pokemon[pos]
                    order = G._decode(battle, int(action), pos)
                    move, hit = G._move_and_targets(battle, order, pos)
                    if me is None or move is None or all(f is not foe for f in hit):
                        continue
                    fraction = K.damage_fraction(battle, me, foe, move)
                    if fraction is not None and fraction[1] > 0:
                        total, aimed = total + fraction[0], True
            totals.append(total if aimed else None)
        if totals[0] is None or totals[1] is None:
            continue
        rows.append(
            {
                "foe": forme_key(foe),
                "hp": float(foe.current_hp_fraction or 0.0),
                "stored_min": totals[0],
                "right_min": totals[1],
                "fainted": K.identifier(battle, foe) in fainted,
            }
        )
    return rows


def analyse_decision(
    battle: DoubleBattle,
    decision: O.Decision,
    config: O.DirectoryConfig,
    own_profile: dict[str, bool],
    deployed_profile: dict[str, bool],
    logged: O.LoggedPredictions,
    stale: Sequence[tuple[Pokemon, F.Stats]],
    timings: dict[str, list[float]],
    fainted: set[str],
) -> dict[str, Any]:
    row = decision.row
    out: dict[str, Any] = {}
    if not O.decode_matches(battle, row):
        return {"status": "decode_mismatch"}
    logged_order = O.rebuild_candidates(row)
    raw = sorted(
        (G.Candidate(actions=c.actions, prob=c.prob) for c in logged_order),
        key=lambda c: -c.prob,
    )
    if len(raw) < 2:
        return {"status": "one_candidate"}
    out["status"] = "ok"

    # 1. does the replayed stack (the directory's own profile, stored numbers)
    #    reproduce what the guards handed on in the real game?
    report = row.get("reranker")
    handed_on = tuple(
        (report or {}).get("before") or row["chosen"]["actions"]  # no reranker report
    )
    own, own_report = G.apply_guards(battle, _copies(raw), own_profile)
    out["guard_stage_reproduced"] = own[0].actions == handed_on
    out["logged_stages"] = list((row.get("guards") or {}).get("stages") or [])
    out["replayed_stages"] = list(own_report.stages)

    active = [mon for mon, _ in stale if mon.active and not mon.fainted]
    out["stale_active"] = [forme_key(mon) for mon in active]
    out["stale_bench"] = [forme_key(mon) for mon, _ in stale if mon not in active]
    if not stale:
        return out

    # 2. today's deployed profile, stored numbers against the right ones
    started = time.perf_counter()
    old, old_report = G.apply_guards(battle, _copies(raw), deployed_profile)
    timings["stack_plain"].append(time.perf_counter() - started)
    switched = {**deployed_profile, G.FORME_STATS: True}
    started = time.perf_counter()
    new, new_report = G.apply_guards(battle, _copies(raw), switched)
    timings["stack_forme_stats"].append(time.perf_counter() - started)
    changed = new[0].actions != old[0].actions
    out["guard_pick_changed"] = changed
    out["marker_agrees"] = changed == (G.FORME_STATS in new_report.stages)
    out["old_stages"] = list(old_report.stages)
    out["new_stages"] = [s for s in new_report.stages if s != G.FORME_STATS]
    out["old_pick"] = list(old[0].actions)
    out["new_pick"] = list(new[0].actions)
    out["old_label"] = _label(battle, old[0].actions)
    out["new_label"] = _label(battle, new[0].actions)
    out["demotions_moved"] = sorted(
        name
        for name in set(old_report.demotions) | set(new_report.demotions)
        if not name.startswith(G.FORME_STATS)
        and old_report.demotions[name] != new_report.demotions[name]
    )

    # 3. each enabled guard alone on the policy's order
    alone_any, alone_top = [], []
    for name in G.GUARD_ORDER:
        if name == G.FORME_STATS or not deployed_profile.get(name, False):
            continue
        try:
            a = G.GUARDS[name](battle, _copies(raw), G.GuardReport())
            with F.swapped(list(stale)):
                b = G.GUARDS[name](battle, _copies(raw), G.GuardReport())
        except Exception:
            out.setdefault("guard_errors", []).append(name)
            continue
        if _signature(a) != _signature(b):
            alone_any.append(name)
            if a[0].actions != b[0].actions:
                alone_top.append(name)
    out["guards_verdict_moved"] = alone_any
    out["guards_top_moved"] = alone_top

    # 4. the reranker and sticky corrections after the guards, as deployed
    today = replace(config, sticky=True)
    moves, switches = logged.predict(battle)

    def final(cands: list[G.Candidate], stages: Sequence[str]) -> tuple[int, int]:
        fired = any(stage != G.FORME_STATS for stage in stages)
        return O.run_consumer(battle, cands, moves, switches, today, fired).pick

    try:
        p0 = final(old, old_report.stages)
        p1 = final(new, new_report.stages)
        with F.swapped(list(stale)):
            p2 = final(_copies(new), new_report.stages)
        out["final_pick_changed"] = p1 != p0
        out["final_old_label"] = _label(battle, p0)
        out["final_new_label"] = _label(battle, p1)
        out["reranker_would_also_move"] = p2 != p1
        out["final_both_label"] = _label(battle, p2)
    except Exception as exc:
        out["reranker_error"] = type(exc).__name__

    # 5. what the brain would see if the observation were recomputed too, and
    #    whether the Pokemon it describes as safe went down that turn
    if active:
        try:
            shift = observation_shift(battle, stale)
            for mine in shift["ours"]:
                mine["fainted"] = mine["key"] in fainted
            out["observation"] = shift
        except Exception as exc:
            out["observation_error"] = type(exc).__name__

    # 6. the pair that was really played: where it claims a knock-out on a stale foe
    try:
        out["played_ko_claims"] = ko_claims(
            battle, row["chosen"]["actions"], stale, fainted
        )
    except Exception as exc:
        out["ko_claims_error"] = type(exc).__name__

    # 7. for reading a changed pick by hand: what was played and what happened
    if changed:
        out["played_label"] = _label(battle, row["chosen"]["actions"])
        out["fainted_this_turn"] = sorted(fainted)
        try:
            out["old_attacks"] = attack_details(battle, old[0].actions, stale)
            out["new_attacks"] = attack_details(battle, new[0].actions, stale)
        except Exception as exc:
            out["attack_details_error"] = type(exc).__name__
    return out


# --- one directory -------------------------------------------------------------------


def _profile(extras: Sequence[str]) -> dict[str, bool]:
    return {
        name: (name in G.HARD_GUARDS or name in extras) and name != G.FORME_STATS
        for name in G.GUARDS
    }


def process_directory(
    folder: Path,
    deployed_profile: dict[str, bool],
    counters: Counter[str],
    timings: dict[str, list[float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """(turn rows, hit rows, decision rows) of one replay directory."""
    config = O.read_run_config(folder)
    if config is None:
        counters["directory_without_run_config"] += 1
        return [], [], []
    try:
        material = json.loads((folder / "run_config.json").read_text())["runs"][-1][
            "material"
        ]
        sets = Teambuilder.parse_showdown_team(config.team.read_text())
    except Exception as exc:
        counters[f"directory_error:{type(exc).__name__}"] += 1
        return [], [], []
    extras = [g for g in str(material.get("guards_extra") or "").split(",") if g]
    own_profile = _profile(extras)
    decisions: list[O.Decision] = []
    sheets: dict[str, bool] = {}
    audit = folder / "decisions.jsonl"
    if audit.exists():
        text = audit.read_text(encoding="utf-8", errors="replace")
        decisions, _, sheets = O.read_decisions(text.split("\n"), folder.name)
    by_room: dict[str, list[O.Decision]] = defaultdict(list)
    for decision in decisions:
        by_room[decision.room].append(decision)

    turns: list[dict[str, Any]] = []
    hit_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    previous = (PolicyPlayer.use_moveset_prior, _os.environ.get("VGC_SET_PRIOR_REG"))
    PolicyPlayer.use_moveset_prior = config.moveset_prior
    _os.environ["VGC_SET_PRIOR_REG"] = config.set_prior_reg
    try:
        logged = O.LoggedPredictions(config, counters)
        for room, (path, player) in sorted(O.find_replays(folder).items()):
            try:
                log = E.extract_log_from_html(
                    path.read_text(encoding="utf-8", errors="replace")
                )
            except OSError:
                log = None
            if not log:
                counters["replay_without_log"] += 1
                continue
            events = E.split_log(log)
            role = O.bot_role(events, player)
            if role is None:
                counters["bot_side_not_found"] += 1
                continue
            game = O.public_battle_id(room)
            won = any(
                len(event) > 2
                and event[1] == "win"
                and E.user_id(event[2]) == E.user_id(player)
                for event in events
            )
            try:
                position = O.Position("battle-" + game, player, role, sets)
            except Exception as exc:
                counters[f"rebuild_error:{type(exc).__name__}"] += 1
                continue
            battle = position.battle
            pending: dict[int, list[O.Decision]] = defaultdict(list)
            for decision in by_room.get(room, []):
                pending[decision.turn].append(decision)
            counters["games"] += 1
            base = {
                "dir": folder.name,
                "game": game,
                "won": won,
                "sheet": {True: "open", False: "closed"}.get(
                    sheets.get(room),  # type: ignore[arg-type]
                    "unknown",
                ),
            }
            for index, event in enumerate(events):
                kind = position.feed(event, counters)
                if kind == "teampreview":
                    logged.at_preview(battle)
                if kind != "turn":
                    continue
                try:
                    turn = int(event[2])
                    position.at_turn_start()
                    logged.observe(battle)
                    stale = F.find(battle)
                except Exception as exc:
                    counters[f"turn_start_error:{type(exc).__name__}"] += 1
                    continue
                active = [m for m, _ in stale if m.active and not m.fainted]
                turns.append(
                    dict(base, turn=turn, stale_active=[forme_key(m) for m in active])
                )
                try:
                    hits, why = read_hits(events, index, field_hp(battle))
                    for name, count in why.items():
                        counters[f"hit_dropped:{name}"] += count
                    for row in score_hits(battle, hits, stale, role):
                        hit_rows.append(dict(base, turn=turn, **row))
                except Exception as exc:
                    counters[f"hit_error:{type(exc).__name__}"] += 1
                fainted = faints(events, index)
                for decision in pending.pop(turn, []):
                    try:
                        row = analyse_decision(
                            battle,
                            decision,
                            config,
                            own_profile,
                            deployed_profile,
                            logged,
                            stale,
                            timings,
                            fainted,
                        )
                    except Exception as exc:
                        row = {"status": f"error:{type(exc).__name__}"}
                    decision_rows.append(
                        dict(
                            base,
                            turn=turn,
                            line=decision.line,
                            superseded=decision.superseded,
                            **row,
                        )
                    )
            for leftovers in pending.values():
                for decision in leftovers:
                    decision_rows.append(
                        dict(
                            base,
                            turn=decision.turn,
                            line=decision.line,
                            superseded=decision.superseded,
                            status="turn_not_in_replay",
                        )
                    )
            logged.done(battle)
    finally:
        PolicyPlayer.use_moveset_prior = previous[0]
        if previous[1] is None:
            _os.environ.pop("VGC_SET_PRIOR_REG", None)
        else:
            _os.environ["VGC_SET_PRIOR_REG"] = previous[1]
    return turns, hit_rows, decision_rows


# --- summary -------------------------------------------------------------------------


def _share(count: float, total: float) -> float | None:
    return count / total if total else None


def accuracy(rows: Sequence[Mapping[str, Any]], line: str) -> dict[str, Any] | None:
    """How the real damage sits against the calculator's range on one stat line."""
    ratios, inside, low, high = [], 0, 0, 0
    for row in rows:
        lo, hi = row[line]
        realized = row["realized"]
        ratios.append(realized / ((lo + hi) / 2))
        # an opposing Pokemon's HP is shown in whole percent
        slack = 0.011 if row["direction"] == "into_foe" else 0.004
        if realized < lo - slack:
            low += 1
        elif realized > hi + slack:
            high += 1
        else:
            inside += 1
    if not ratios:
        return None
    return {
        "hits": len(ratios),
        "median_real_over_predicted": statistics.median(ratios),
        "mean_abs_log_error": sum(abs(math.log(r)) for r in ratios) / len(ratios),
        "inside_range": inside / len(ratios),
        "real_below_range": low / len(ratios),
        "real_above_range": high / len(ratios),
    }


def summarise(
    turns: Sequence[Mapping[str, Any]],
    hits: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    counters: Counter[str],
    timings: Mapping[str, Sequence[float]],
    deployed_extras: Sequence[str],
) -> dict[str, Any]:
    games = {(row["dir"], row["game"]) for row in turns}
    exposed_turns = [row for row in turns if row["stale_active"]]
    exposed_games = {(row["dir"], row["game"]) for row in exposed_turns}
    species: Counter[str] = Counter()
    for row in exposed_turns:
        species.update(row["stale_active"])
    species_games: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in exposed_turns:
        for name in row["stale_active"]:
            species_games[name].add((row["dir"], row["game"]))

    out: dict[str, Any] = {
        "games": len(games),
        "turns": len(turns),
        "exposure": {
            "games_with_a_stale_foe_on_the_field": len(exposed_games),
            "share_of_games": _share(len(exposed_games), len(games)),
            "turns_with_a_stale_foe_on_the_field": len(exposed_turns),
            "share_of_turns": _share(len(exposed_turns), len(turns)),
            "by_species": {
                name: {"turns": count, "games": len(species_games[name])}
                for name, count in species.most_common()
            },
        },
    }

    # accuracy
    acc: dict[str, Any] = {}
    for direction in ("by_foe", "into_foe"):
        mine = [row for row in hits if row["direction"] == direction]
        stale_rows = [row for row in mine if row["stale"]]
        control = [row for row in mine if not row["stale"]]
        closer = sum(
            abs(math.log(r["realized"] / (sum(r["fresh"]) / 2)))
            < abs(math.log(r["realized"] / (sum(r["stored"]) / 2)))
            for r in stale_rows
        )
        acc[direction] = {
            "stale_foe_stored_line": accuracy(stale_rows, "stored"),
            "stale_foe_right_line": accuracy(stale_rows, "fresh"),
            "right_line_closer": _share(closer, len(stale_rows)),
            "control_foe_not_stale": accuracy(control, "stored"),
        }
    out["accuracy"] = acc

    # decisions
    ok = [row for row in decisions if row.get("status") == "ok"]
    status = Counter(str(row.get("status")) for row in decisions)
    reproduced = sum(bool(row.get("guard_stage_reproduced")) for row in ok)
    fired = [row for row in ok if row.get("logged_stages")]
    exposed = [row for row in ok if row.get("stale_active")]
    with_stale = [row for row in ok if "guard_pick_changed" in row]

    def block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        exposed_rows = [row for row in rows if row.get("stale_active")]
        measured = [row for row in rows if "guard_pick_changed" in row]
        changed = [row for row in measured if row["guard_pick_changed"]]
        final = [row for row in measured if row.get("final_pick_changed")]
        also = [row for row in measured if row.get("reranker_would_also_move")]
        moved: Counter[str] = Counter()
        top: Counter[str] = Counter()
        stages: Counter[str] = Counter()
        for row in measured:
            moved.update(row.get("guards_verdict_moved") or [])
            top.update(row.get("guards_top_moved") or [])
        for row in changed:
            stages.update(set(row["old_stages"]) ^ set(row["new_stages"]))
        shifts = [row["observation"] for row in exposed_rows if "observation" in row]
        # of our Pokemon on the field in an exposed decision: did it go down that
        # turn, by what the block said on the stored line and on the right one
        went_down: dict[str, dict[str, list[int]]] = {}
        for row in exposed_rows:
            if row.get("superseded") or "observation" not in row:
                continue
            for mine in row["observation"]["ours"]:
                for fact in ("can_be_koed", "expected_ko"):
                    cell = "stored %s, right %s" % tuple(
                        "yes" if flag else "no" for flag in mine[fact]
                    )
                    tally = went_down.setdefault(fact, {}).setdefault(cell, [0, 0])
                    tally[0] += 1
                    tally[1] += bool(mine["fainted"])
        claims: dict[str, list[int]] = {}
        for row in exposed_rows:
            if row.get("superseded"):
                continue
            for claim in row.get("played_ko_claims") or []:
                cell = "stored %s, right %s" % tuple(
                    "KO" if claim[name] >= claim["hp"] else "no KO"
                    for name in ("stored_min", "right_min")
                )
                tally = claims.setdefault(cell, [0, 0])
                tally[0] += 1
                tally[1] += bool(claim["fainted"])
        n = len(rows)
        return {
            "move_decisions": n,
            "games": len({(row["dir"], row["game"]) for row in rows}),
            "exposed": len(exposed_rows),
            "exposed_share": _share(len(exposed_rows), n),
            "guard_pick_changed": len(changed),
            "guard_pick_changed_share_of_all": _share(len(changed), n),
            "guard_pick_changed_share_of_exposed": _share(
                len(changed), len(exposed_rows)
            ),
            "guard_pick_changed_per_game": _share(
                len(changed), len({(row["dir"], row["game"]) for row in rows})
            ),
            "final_pick_changed_after_reranker_and_sticky": len(final),
            "reranker_would_also_move_if_it_used_the_right_lines": len(also),
            "stages_that_differ_on_changed_picks": dict(stages.most_common()),
            "each_guard_alone_verdict_moved": dict(moved.most_common()),
            "each_guard_alone_top_moved": dict(top.most_common()),
            "our_pokemon_knocked_out_that_turn": {
                fact: {
                    cell: {
                        "pokemon": tally[0],
                        "fainted": tally[1],
                        "rate": _share(tally[1], tally[0]),
                    }
                    for cell, tally in sorted(cells.items())
                }
                for fact, cells in went_down.items()
            },
            "stale_foe_knocked_out_that_turn_by_minimum_roll_claim": {
                cell: {
                    "foes": tally[0],
                    "fainted": tally[1],
                    "rate": _share(tally[1], tally[0]),
                }
                for cell, tally in sorted(claims.items())
            },
            "observation_if_recomputed": None
            if not shifts
            else {
                "decisions": len(shifts),
                "any_float_moved": _share(
                    sum(s["moved"] > 0 for s in shifts), len(shifts)
                ),
                "any_flag_flipped": _share(
                    sum(s["flag_flips"] > 0 for s in shifts), len(shifts)
                ),
                "mean_floats_moved": statistics.mean(s["moved"] for s in shifts),
                "mean_floats_in_the_blocks": statistics.mean(
                    s["floats"] for s in shifts
                ),
                "mean_flag_flips": statistics.mean(s["flag_flips"] for s in shifts),
                "mean_largest_move": statistics.mean(s["largest"] for s in shifts),
            },
        }

    t6 = [row for row in ok if "T4" not in row["dir"] and "guards3" not in row["dir"]]
    today = [row for row in ok if "t6e" in row["dir"].lower() or "T6ep" in row["dir"]]
    marker_wrong = sum(not row.get("marker_agrees", True) for row in with_stale)
    out["decisions"] = {
        "records": dict(status),
        "replayed_guard_stage_equals_logged": {
            "all": _share(reproduced, len(ok)),
            "where_a_guard_fired_live": _share(
                sum(bool(row.get("guard_stage_reproduced")) for row in fired),
                len(fired),
            ),
            "decisions": len(ok),
        },
        "marker_disagreements": marker_wrong,
        "deployed_extras": list(deployed_extras),
        "all_directories": block(ok),
        "t6_family_directories": block(t6),
        "todays_team_t6e_directories": block(today),
        "exposed_only": {"count": len(exposed)},
    }
    out["timings_ms"] = {
        name: {
            "n": len(values),
            "median": 1000 * statistics.median(values),
            "p90": 1000 * sorted(values)[int(0.9 * (len(values) - 1))],
        }
        for name, values in timings.items()
        if values
    }
    out["counters"] = dict(sorted(counters.items()))
    return out


def render(summary: Mapping[str, Any], decisions: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{100 * value:.1f}%"

    exposure = summary["exposure"]
    lines.append(
        f"games {summary['games']}, turns {summary['turns']}; a stale opposing "
        f"forme on the field in {exposure['games_with_a_stale_foe_on_the_field']} "
        f"games ({pct(exposure['share_of_games'])}) and "
        f"{exposure['turns_with_a_stale_foe_on_the_field']} turns "
        f"({pct(exposure['share_of_turns'])})"
    )
    for name, row in list(exposure["by_species"].items())[:25]:
        lines.append(f"    {name:16s} {row['turns']:5d} turns in {row['games']} games")
    lines.append("")
    lines.append("accuracy: real damage / middle of the calculator's range")
    for direction, label in (
        ("by_foe", "hits BY the foe on ours"),
        ("into_foe", "our hits INTO the foe"),
    ):
        block = summary["accuracy"][direction]
        lines.append(f"  {label}:")
        for key, name in (
            ("stale_foe_stored_line", "stale forme, stored line"),
            ("stale_foe_right_line", "stale forme, right line "),
            ("control_foe_not_stale", "control (forme right)   "),
        ):
            a = block[key]
            if a is None:
                lines.append(f"    {name}: no hits")
                continue
            lines.append(
                f"    {name}: n={a['hits']:4d} "
                f"median {a['median_real_over_predicted']:.2f}  "
                f"|log error| {a['mean_abs_log_error']:.3f}  "
                f"inside range {pct(a['inside_range'])}  "
                f"real above {pct(a['real_above_range'])}  "
                f"below {pct(a['real_below_range'])}"
            )
        lines.append(f"    right line closer: {pct(block['right_line_closer'])}")
    lines.append("")
    d = summary["decisions"]
    r = d["replayed_guard_stage_equals_logged"]
    lines.append(
        f"replayed guard stage == logged: {pct(r['all'])} of {r['decisions']} "
        f"decisions ({pct(r['where_a_guard_fired_live'])} where a guard fired live)"
    )
    for key in (
        "all_directories",
        "t6_family_directories",
        "todays_team_t6e_directories",
    ):
        b = d[key]
        also = b["reranker_would_also_move_if_it_used_the_right_lines"]
        lines.append(
            f"{key}: {b['move_decisions']} move decisions in {b['games']} games, "
            f"exposed {b['exposed']} ({pct(b['exposed_share'])}); "
            f"guard pick changes {b['guard_pick_changed']} "
            f"({pct(b['guard_pick_changed_share_of_all'])} of all, "
            f"{pct(b['guard_pick_changed_share_of_exposed'])} of exposed, "
            f"{(b['guard_pick_changed_per_game'] or 0):.2f} per game); final pick "
            f"changes {b['final_pick_changed_after_reranker_and_sticky']}; "
            f"reranker would also move {also}"
        )
        lines.append(f"    stages: {b['stages_that_differ_on_changed_picks']}")
        lines.append(f"    alone, verdict: {b['each_guard_alone_verdict_moved']}")
        lines.append(f"    alone, top:     {b['each_guard_alone_top_moved']}")
        lines.append(f"    observation:    {b['observation_if_recomputed']}")
        lines.append(
            "    the stale foe our played pair attacked went down that turn, by "
            "whether the pair's minimum rolls remove its HP:"
        )
        cells = b["stale_foe_knocked_out_that_turn_by_minimum_roll_claim"]
        for cell, tally in cells.items():
            lines.append(
                f"        {cell:26s} {tally['fainted']:4d} of {tally['foes']:4d}"
                f"  ({pct(tally['rate'])})"
            )
        for fact, cells in b["our_pokemon_knocked_out_that_turn"].items():
            lines.append(f"    our Pokemon went down that turn, by '{fact}':")
            for cell, tally in cells.items():
                lines.append(
                    f"        {cell:24s} {tally['fainted']:4d} of {tally['pokemon']:4d}"
                    f"  ({pct(tally['rate'])})"
                )
    lines.append(f"timings (ms): {summary['timings_ms']}")
    lines.append("")
    lines.append("every guard-stage pick change (dir, game, turn, result, stale foes):")
    for row in decisions:
        if not row.get("guard_pick_changed"):
            continue
        foes = ",".join(row["stale_active"] + row["stale_bench"])
        stages = sorted(set(row["old_stages"]) ^ set(row["new_stages"]))
        lines.append(
            f"  {row['dir'][-30:]:30s} {row['game'][-10:]} T{row['turn']:<2d} "
            f"{'W' if row['won'] else 'L'} {foes:22s} "
            f"{row['old_label']} -> {row['new_label']}  {stages}"
            + ("" if row.get("final_pick_changed") else "  (undone after the guards)")
        )
        lines.append(
            f"      played {row.get('played_label')}; fainted that turn: "
            f"{row.get('fainted_this_turn')}"
        )
        for name in ("old_attacks", "new_attacks"):
            for attack in row.get(name) or []:
                lines.append(
                    f"      {name[:3]}: {attack['move']} into {attack['target']} "
                    f"(HP {attack['hp']}): stored {attack['stored']}, "
                    f"right {attack['fresh']}"
                )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--only", default=None, help="substring of the directory names")
    ap.add_argument("--output", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    _os.chdir(ROOT)
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(1)
    extras = [
        g
        for g in json.loads(DEPLOYED.read_text())["deployed"]["guards_extra"].split(",")
        if g
    ]
    deployed_profile = _profile(extras)
    saved = (PolicyPlayer.use_knowledge_obs, PolicyPlayer.use_moveset_prior)
    PolicyPlayer.use_knowledge_obs = True
    counters: Counter[str] = Counter()
    timings: dict[str, list[float]] = defaultdict(list)
    turns: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    try:
        folders = sorted(
            folder
            for pattern in GLOBS
            for folder in ROOT.glob(pattern)
            if folder.is_dir() and (args.only is None or args.only in folder.name)
        )
        for folder in folders:
            started = time.perf_counter()
            t, h, d = process_directory(folder, deployed_profile, counters, timings)
            turns += t
            hits += h
            decisions += d
            print(
                f"{folder.name}: {len({r['game'] for r in t})} games, {len(t)} turns, "
                f"{len(h)} hits, {len(d)} decisions "
                f"({time.perf_counter() - started:.0f}s)",
                flush=True,
            )
    finally:
        PolicyPlayer.use_knowledge_obs, PolicyPlayer.use_moveset_prior = saved
    summary = summarise(turns, hits, decisions, counters, timings, extras)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    with (args.output / "decisions.jsonl").open("w") as handle:
        for row in decisions:
            if row.get("stale_active") or row.get("stale_bench"):
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (args.output / "hits.jsonl").open("w") as handle:
        for row in hits:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report = render(summary, decisions)
    (args.output / "report.txt").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
