"""Ladder positions for guard tests.

Protocol lines go through poke-env's own parser (with pokeenv_patches installed,
as on ladder), our sets come from teams/candidates_mc/T6.txt, and all six of our
Pokemon are registered first in team-file order, as the live request lists them,
so switch actions 1-6 decode to the same Pokemon they did on ladder, with the
stats the Champions server sends (poke-env's teambuilder applies the main-series
formula: Charizard's Speed 124 instead of 152). The bot is p1; positions from
games where it was p2 are written with the sides swapped.
"""

from __future__ import annotations

import logging
from pathlib import Path

from poke_env.battle import DoubleBattle, Move
from poke_env.data import GenData, to_id_str
from poke_env.teambuilder import Teambuilder

from vgc_bench.src import guards as G
from vgc_bench.src import pokeenv_patches

ROOT = Path(__file__).resolve().parents[1]
T6 = Teambuilder.parse_showdown_team((ROOT / "teams/candidates_mc/T6.txt").read_text())
SETS = {to_id_str(mon.species or mon.nickname): mon for mon in T6}
HEADER = ["|player|p1|antonius1|1|1100", "|player|p2|rival|1|1100", "|gen|9"]
STATS = ("hp", "atk", "def", "spa", "spd", "spe")


def champions_stats(mon) -> dict[str, int | None]:
    """Champions' stat formula: HP = base + EVs + 75, others (base + EVs + 20) x
    nature, EVs 0-32 per stat."""
    nature = GenData.from_gen(9).natures[
        to_id_str(SETS[to_id_str(mon.base_species)].nature)
    ]
    evs = SETS[to_id_str(mon.base_species)].evs or [0] * len(STATS)
    stats: dict[str, int | None] = {}
    for stat, ev in zip(STATS, evs):
        base = mon.base_stats[stat]
        if stat == "hp":
            stats[stat] = base + ev + 75
        else:
            stats[stat] = int((base + ev + 20) * nature.get(stat, 1))
    return stats


def position(lines: list[str]) -> DoubleBattle:
    pokeenv_patches.install()
    battle = DoubleBattle(
        "battle-gen9championsvgc2026regmc-fixture",
        "antonius1",
        logging.getLogger("ladder_position"),
        gen=9,
    )
    battle._player_role = "p1"
    for mon in T6:  # sets first: the protocol's Mega forms must survive them
        species = mon.species or mon.nickname
        ours = battle.get_pokemon(f"p1: {species}", details=f"{species}, L50")
        ours._update_from_teambuilder(mon)
    for line in [*HEADER, *lines]:
        battle.parse_message(line.split("|"))
    for mon in battle.team.values():
        if mon.species != to_id_str(mon.base_species):  # a Mega: its own ability
            mon._ability = to_id_str(mon.possible_abilities[0])
        mon.stats = champions_stats(mon)
    return battle


def move_action(battle: DoubleBattle, pos: int, move_id: str, target: int) -> int:
    for action in range(7, 27):
        order = G._decode(battle, action, pos)
        move = getattr(order, "order", None)
        if (
            isinstance(move, Move)
            and move.id == move_id
            and getattr(order, "move_target", None) == target
        ):
            return action
    raise AssertionError(f"no action for {move_id} -> {target}")


def switch_action(battle: DoubleBattle, species: str) -> int:
    for action, mon in enumerate(battle.team.values(), start=1):
        if to_id_str(mon.species) == species:
            return action
    raise AssertionError(f"{species} is not on the team")


def run(guard, battle: DoubleBattle, pairs: list[tuple[tuple[int, int], float]]):
    cands = [G.Candidate(actions, prob) for actions, prob in pairs]
    report = G.GuardReport()
    return guard(battle, cands, report), report
