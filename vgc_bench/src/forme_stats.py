"""Stat estimates that follow an opposing Pokemon's forme.

An opponent's real stats are never known. ``vgc_knowledge.ensure_stats`` gives each
opposing Pokemon an estimate from its base stats (32 HP, 32 in the better attack,
2 Speed, neutral nature) the first time a calculation needs one, and leaves it alone
afterwards: the stats are "already usable". poke-env updates ``base_stats`` when the
Pokemon changes forme (``-mega``, ``detailschange``, ``-formechange``) but not
``stats``. The live bot runs the damage calculator against every active foe at every
decision, so a foe is always estimated in the forme it came in with, and from the
turn after it Mega-evolves every calculation by or against it uses its old forme's
numbers: Mega Blastoise's Special Attack 137 for 187, Mega Charizard Y's 161 for 211,
Mega Gengar's Speed 132 for 152 (found 2026-10-04; open and hidden sheets alike).

The estimate is deliberately NOT repaired where it is stored. The observation's
knowledge and threat blocks are computed from the same numbers and every brain so far
was trained on them; inputs a brain never trained on have hurt it even when they were
more accurate (PROJECT_STATUS 2026-09-23). ``swapped`` instead puts the right
estimate in for the length of a ``with`` block and the old line back afterwards, so
only the code inside the block sees it. ``guards.apply_guards`` runs the guard stack
inside it when the opt-in profile entry ``forme_stats`` is on.

Only an opponent's line is ever replaced, and only when it is recognisably this
project's own stale work: exactly ``ensure_stats``' estimate for ANOTHER forme of the
same Pokemon. Our own stats come from the server's request and already follow the
forme. A transformed Pokemon (temporary base stats, its own HP) is left alone.

Two kinds of line exist. Without a team sheet ``ensure_stats`` builds the estimate
from nothing, with a neutral nature. With an open sheet poke-env first writes the
sheet's numbers -- no stat points, but the NATURE is on the sheet and in them -- and
``ensure_stats`` tops those up, keeping each stat's nature multiplier. Both are
recognised; the replacement keeps the nature the stale line shows.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from copy import copy
from functools import lru_cache

from poke_env.battle import DoubleBattle, Pokemon
from poke_env.data import GenData, to_id_str

from vgc_bench.src import vgc_knowledge as K

Stats = dict[str, int | None]
_STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")
_NATURE_MULTS = (0.9, 1.0, 1.1)


def estimate(
    mon: Pokemon,
    base_stats: dict[str, int] | None = None,
    nature: dict[str, float] | None = None,
) -> Stats | None:
    """``ensure_stats``' line for ``mon`` as if it were first seen now -- or, with
    ``base_stats``, as if those were its base stats.

    ``nature`` (stat -> 0.9 / 1.0 / 1.1) is for a Pokemon on an open team sheet: the
    copy starts from what poke-env computes from a sheet without stat points, as it
    does in a real game, and ``ensure_stats`` tops that up.

    Computed by ``ensure_stats`` itself on a copy, so the two cannot drift apart.
    ``mon`` is not touched.
    """
    probe = copy(mon)
    if base_stats is not None:
        probe._base_stats = base_stats
        probe._temporary_base_stats = None
    base = probe.base_stats
    line: Stats = {key: None for key in _STAT_KEYS}
    if nature is not None:
        for key in _STAT_KEYS:
            # level 50, 31 IVs, no stat points: base + 75 HP, (base + 20) x nature
            line[key] = (
                base[key] + 75
                if key == "hp"
                else int((base[key] + 20) * nature.get(key, 1.0))
            )
    probe.stats = line
    if not K.ensure_stats(probe):
        return None
    return probe.stats


def _nature(stats: Stats, neutral: Stats) -> dict[str, float] | None:
    """The nature multipliers that turn the neutral estimate ``neutral`` into
    ``stats``; None when no nature does (then ``stats`` is not that estimate)."""
    if stats.get("hp") != neutral.get("hp"):
        return None
    nature: dict[str, float] = {}
    for key in _STAT_KEYS[1:]:
        value, plain = stats.get(key), neutral.get(key)
        if value is None or plain is None:
            return None
        mult = next((m for m in _NATURE_MULTS if int(plain * m) == value), None)
        if mult is None:
            return None
        nature[key] = mult
    raised = sum(mult > 1.0 for mult in nature.values())
    lowered = sum(mult < 1.0 for mult in nature.values())
    return nature if (raised, lowered) in ((0, 0), (1, 1)) else None


def _line_of(mon: Pokemon, base_stats: dict[str, int] | None) -> dict | None:
    """The nature with which ``mon``'s line is exactly the estimate for
    ``base_stats`` (None: its own, current ones); None if it is not that estimate."""
    stats = mon.stats
    neutral = estimate(mon, base_stats)
    nature = None if neutral is None else _nature(stats, neutral)
    if nature is None or estimate(mon, base_stats, nature) != stats:
        return None
    return nature


@lru_cache(maxsize=None)
def _formes(gen: int) -> dict[str, tuple[dict[str, int], ...]]:
    """Every distinct base-stat line of each species' formes, by base species id."""
    out: dict[str, list[dict[str, int]]] = {}
    for key, entry in GenData.from_gen(gen).pokedex.items():
        base = entry.get("baseStats")
        if not base:
            continue
        lines = out.setdefault(to_id_str(entry.get("baseSpecies") or key), [])
        if base not in lines:
            lines.append(base)
    return {root: tuple(lines) for root, lines in out.items()}


def stale_estimate(mon: Pokemon) -> Stats | None:
    """The estimate for the forme ``mon`` is in now, when the line it carries is the
    estimate for another of its formes. None when its line is right, is not an
    estimate of ours, or is absent."""
    stats, base = mon.stats, mon.base_stats
    if not stats or not base or any(stats.get(key) is None for key in _STAT_KEYS):
        return None
    if getattr(mon, "_temporary_base_stats", None) is not None:
        return None  # Transform: another Pokemon's base stats, not a forme of this one
    if _line_of(mon, None) is not None:
        return None  # the estimate for the forme it is in
    for other in _formes(mon.gen).get(mon.base_species, ()):
        if other == base:
            continue
        nature = _line_of(mon, other)
        if nature is not None:
            fresh = estimate(mon, None, nature)
            return fresh if fresh != stats else None
    return None


def find(battle: DoubleBattle) -> list[tuple[Pokemon, Stats]]:
    """The opponent's Pokemon whose stat estimate belongs to another forme, each
    with the estimate for the forme it is in. Changes nothing."""
    found: list[tuple[Pokemon, Stats]] = []
    for mon in battle.opponent_team.values():
        fresh = stale_estimate(mon)
        if fresh is not None:
            found.append((mon, fresh))
    return found


@contextmanager
def swapped(found: list[tuple[Pokemon, Stats]]) -> Generator[None]:
    """Inside the block each Pokemon of ``find``'s result carries its new line; the
    line it had (the same dict object) is back on exit, also after an exception."""
    old = [(mon, mon.stats) for mon, _ in found]
    try:
        for mon, fresh in found:
            mon.stats = fresh
        yield
    finally:
        for mon, stats in old:
            mon.stats = stats


@contextmanager
def current_forme(battle: DoubleBattle) -> Generator[list[Pokemon]]:
    """``swapped(find(battle))``; yields the Pokemon whose line was replaced."""
    found = find(battle)
    with swapped(found):
        yield [mon for mon, _ in found]
