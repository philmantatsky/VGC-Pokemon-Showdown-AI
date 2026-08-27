"""Team-agnostic Team Preview knowledge.

Every function here takes rosters as input and derives everything from data
files and mechanics -- no species may be hardcoded. The same code must produce
sensible output for any team, so the layer keeps working if our team changes.

Currently: Trick Room likelihood from the replay-counted joint sets. The ladder
evidence that motivated this (2026-08-23 audit): opponents whose roster carried
a likely Trick Room setter beat the bot 75% of the time (n=48), and P(TR) is a
pure data lookup (e.g. Farigiraf runs Trick Room in ~97% of recorded sets).

Stage D of the replan extends this module with the preview rule engine
(`apply_preview_rules`); shadow logging uses the probabilities alone first so
rule content can be derived from data rather than intuition.
"""

import json
from pathlib import Path

from poke_env.data import to_id_str

TRICK_ROOM_MOVE = "trickroom"

_JOINT_SETS: dict | None = None
_TR_RATE_CACHE: dict[str, float] = {}


def _joint_sets() -> dict:
    """Load the replay-counted joint sets once (same pattern as guards.py)."""
    global _JOINT_SETS
    if _JOINT_SETS is None:
        try:
            root = Path(__file__).resolve().parents[2]
            _JOINT_SETS = json.loads(
                (root / "data" / "joint_sets_regmb.json").read_text()
            )
        except (OSError, ValueError):
            _JOINT_SETS = {}
    return _JOINT_SETS or {}


def species_trick_room_rate(species: str) -> float:
    """P(this species' set contains Trick Room), from counted joint sets.

    Count-weighted over the species' recorded sets; 0.0 for species without
    data (unknown species cannot claim Trick Room evidence).
    """
    key = to_id_str(species)
    if key in _TR_RATE_CACHE:
        return _TR_RATE_CACHE[key]
    entry = _joint_sets().get(key)
    rate = 0.0
    if entry:
        sets = entry.get("sets", [])
        total = sum(int(candidate.get("count", 0)) for candidate in sets)
        if total > 0:
            with_tr = sum(
                int(candidate.get("count", 0))
                for candidate in sets
                if TRICK_ROOM_MOVE in candidate.get("moves", ())
            )
            rate = with_tr / total
    _TR_RATE_CACHE[key] = rate
    return rate


def species_trick_room_rates(roster: tuple[str, ...] | list[str]) -> dict[str, float]:
    """Per-species Trick Room set rates for a roster, rounded for logging."""
    return {
        to_id_str(species): round(species_trick_room_rate(species), 4)
        for species in roster
    }


def trick_room_probability(roster: tuple[str, ...] | list[str]) -> float:
    """P(at least one roster member's set contains Trick Room).

    Treats set choices as independent across the roster, which overstates
    slightly (a team built around TR correlates its sets), so treat this as an
    upper-leaning indicator. The per-species maximum is also informative: a
    single >0.9 species is a dedicated setter, while several 0.2s are not.
    """
    no_tr = 1.0
    for species in roster:
        no_tr *= 1.0 - species_trick_room_rate(species)
    return 1.0 - no_tr
