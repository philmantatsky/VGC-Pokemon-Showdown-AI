"""Tests for the matchup estimate (vgc_bench/src/oppmodel/matchup.py).

The damage chain is checked twice: against integers worked out by hand in the
comments (from the simulator lines the spec cites: base damage
sim/battle-actions.ts:1718, modifier order data/mods/champions/scripts.ts
197-313, ``modify`` sim/battle.ts:2332-2343, stage table sim/pokemon.ts
583-589), and against a second implementation written here in the
simulator's own float-and-truncate style. Both are readings of the same
source lines: neither is a run of the simulator.
"""

from __future__ import annotations

import ast
import math
import random
from pathlib import Path

import pytest

from vgc_bench.src.oppmodel import events as E
from vgc_bench.src.oppmodel import features as F
from vgc_bench.src.oppmodel import matchup as MU

ROOT = Path(__file__).resolve().parents[1]
CHART = F._type_chart()


def line(hp=175, atk=120, dfn=100, spa=100, spd=90, spe=102, base_spe=80):
    return {
        "hp": hp,
        "atk": atk,
        "def": dfn,
        "spa": spa,
        "spd": spd,
        "spe": spe,
        "base_spe": base_spe,
        "physical": int(atk >= spa),
    }


def fighter(types=("normal",), boosts=None, status=None, hp=1.0, **stats):
    return MU.Fighter(line(**stats), tuple(types), dict(boosts or {}), status, hp)


def facts(move_id: str) -> MU.MoveFacts:
    made = MU.move_facts(E.moves_dex()[move_id])
    assert made is not None, move_id
    return made


def board(weather: str = "", reverse: bool = False) -> MU.Board:
    return MU.Board(CHART, weather, reverse)


def plain(kind: str = "normal", physical: bool = True, power: int = 80, **changes):
    made = MU.proxy(kind, physical)
    return MU.MoveFacts(**{**made.__dict__, "power": power, **changes})


# --- the integer chain, by hand -------------------------------------------------


def test_modify_rounds_half_down_like_the_simulator():
    # modify(v, f) = tr((tr(v * tr(f * 4096)) + 2047) / 4096)
    assert MU.modify(44, 1.5) == 66  # 270336 + 2047 = 272383 -> 66
    assert MU.modify(3, 1.5) == 4  # 4.5: 18432 + 2047 = 20479 < 5 * 4096
    assert MU.modify(5, 1.5) == 7  # 7.5: 30720 + 2047 = 32767 < 8 * 4096
    assert MU.modify(45, 0.5) == 22  # 22.5: 92160 + 2047 = 94207 < 23 * 4096
    assert MU.modify(59, 0.75) == 44  # 44.25
    assert MU.modify(44, 0.5) == 22


def test_stages_truncate_like_the_simulator():
    # boostTable 1, 1.5, 2, ...: floor(stat * table) up, floor(stat / table) down.
    assert MU.staged(120, 2) == 240
    assert MU.staged(100, -1) == 66  # 100 / 1.5
    assert MU.staged(101, 1) == 151  # 151.5
    assert MU.staged(100, 9) == 400 and MU.staged(100, -9) == 25  # clipped to 6
    assert MU.stage_factor(1) == 1.5 and MU.stage_factor(-2) == 0.5


def test_base_damage_by_hand():
    # P 80, A 120, D 100: 22 * 80 * 120 = 211200; / 100 = 2112; / 50 = 42; + 2.
    assert MU.max_damage(80, 120, 100) == 44
    # P 95, A 133, D 97: 277970 / 97 = 2865 (r 65); / 50 = 57; + 2 = 59.
    assert MU.max_damage(95, 133, 97) == 59
    # P 40, A 100, D 150 (the spec's weak hit): 88000 / 150 = 586; / 50 = 11; + 2.
    assert MU.max_damage(40, 100, 150) == 13
    # ... with weather x1.5, STAB x1.5 and 4x: 13 -> 19 -> 28 -> 112 (the
    # real-number chain gives 123.6: this is why the chain is in integers).
    assert MU.max_damage(40, 100, 150, weather=1.5, stab=True, steps=2) == 112


def test_each_modifier_by_hand():
    assert MU.max_damage(80, 120, 100, stab=True) == 66  # modify(44, 1.5)
    assert MU.max_damage(80, 120, 100, steps=2) == 176  # 44 * 2 * 2
    assert MU.max_damage(80, 120, 100, steps=-2) == 11  # tr(tr(44 / 2) / 2)
    assert MU.max_damage(80, 120, 100, spread=True) == 33  # modify(44, 0.75) = 33
    assert MU.max_damage(80, 120, 100, weather=1.5) == 66
    assert MU.max_damage(80, 120, 100, weather=0.5) == 22
    assert MU.max_damage(80, 120, 100, crit=True) == 66  # tr(44 * 1.5)
    assert MU.max_damage(80, 120, 100, burn=True) == 22
    # All at once, in the simulator's order. 59 -> spread 44 -> weather 66 ->
    # crit 99 -> STAB modify(99, 1.5) = 148 (148.5 rounds down) -> x2 296 ->
    # burn modify(296, 0.5) = 148.
    assert (
        MU.max_damage(
            95, 133, 97, spread=True, weather=1.5, crit=True, stab=True, steps=1
        )
        == 296
    )
    assert (
        MU.max_damage(
            95,
            133,
            97,
            spread=True,
            weather=1.5,
            crit=True,
            stab=True,
            steps=1,
            burn=True,
        )
        == 148
    )
    # The order matters: STAB before the type steps. P 40, A 118, D 150 gives
    # 15; STAB 22 (22.5 rounds down), halved 11. Halving first would give
    # 7, then 10.
    assert MU.max_damage(40, 118, 150, stab=True, steps=-1) == 11
    assert MU.max_damage(1, 1, 500, steps=-3) == 1  # never below 1


def test_hit_cases_by_hand():
    user = fighter(types=("fire",))
    foe = fighter(types=("normal",))
    b = board()
    # Plain neutral hit: 44 of 175 HP at the maximum roll.
    move = plain("dark")
    assert MU.max_hit(move, user, foe, b) == 44
    est, immune, steps, share = MU.hit(move, user, foe, b)
    assert (est, immune, steps) == (1, 0, 0)
    assert share == pytest.approx(0.925 * 44 / 175)
    # STAB: the move's type is one of the user's current types.
    assert MU.max_hit(plain("fire"), user, foe, b) == 66
    # 4x (ice on dragon / flying), 0.25x (grass on fire / flying), immune.
    assert MU.max_hit(plain("ice"), user, fighter(types=("dragon", "flying")), b) == 176
    assert MU.max_hit(plain("grass"), user, fighter(types=("fire", "flying")), b) == 11
    assert MU.hit(plain("ground"), user, fighter(types=("fire", "flying")), b) == (
        1,
        1,
        0,
        0.0,
    )
    assert MU.max_hit(plain("ground"), user, fighter(types=("flying",)), b) == 0
    assert MU.hit(plain("ice"), user, fighter(types=("dragon", "flying")), b)[2] == 2
    assert MU.hit(plain("grass"), user, fighter(types=("fire", "flying")), b)[2] == -2
    # An unknown type counts as 1; a 2x and a 0.5x cancel.
    assert MU.max_hit(plain("dark"), user, fighter(types=("mystery",)), b) == 44
    assert MU.max_hit(plain("ice"), user, fighter(types=("dragon", "fire")), b) == 44


def test_spread_needs_two_targets():
    user, foe, b = fighter(types=("fire",)), fighter(), board()
    foes = plain("dark", spread=MU.SPREAD_FOES)
    everyone = plain("dark", spread=MU.SPREAD_ALL)
    assert MU.max_hit(foes, user, foe, b, n_opposing=2, n_allies=1) == 33
    assert MU.max_hit(foes, user, foe, b, n_opposing=1, n_allies=1) == 44
    # ``allAdjacent`` also counts the user's partner.
    assert MU.max_hit(everyone, user, foe, b, n_opposing=1, n_allies=1) == 33
    assert MU.max_hit(everyone, user, foe, b, n_opposing=1, n_allies=0) == 44
    assert MU.max_hit(everyone, user, foe, b, n_opposing=2, n_allies=0) == 33


def test_weather_rules_by_the_type_of_the_weather_move():
    user, foe = fighter(types=("normal",)), fighter()
    fire, water = plain("fire"), plain("water")
    assert MU.max_hit(fire, user, foe, board("fire")) == 66
    assert MU.max_hit(water, user, foe, board("fire")) == 22
    assert MU.max_hit(water, user, foe, board("water")) == 66
    assert MU.max_hit(fire, user, foe, board("water")) == 22
    assert MU.max_hit(fire, user, foe, board("rock")) == 44
    assert MU.max_hit(plain("dark"), user, foe, board("fire")) == 44
    # A Rock defender's special bulk in the Rock-type weather. Special P 80,
    # A 100, D 90: 176000 / 90 = 1955; / 50 = 39; + 2 = 41. Guarded:
    # D = modify(90, 1.5) = 135: 176000 / 135 = 1303; / 50 = 26; + 2 = 28.
    special = plain("dark", physical=False)
    rock = fighter(types=("rock", "ground"))
    assert MU.max_hit(special, user, rock, board()) == 41
    assert MU.max_hit(special, user, rock, board("rock")) == 28
    assert MU.max_hit(special, user, fighter(types=("ground",)), board("rock")) == 41
    assert MU.max_hit(plain("dark"), user, rock, board("rock")) == 44  # physical
    # The Ice-type weather guards the physical side of an Ice defender:
    # D = modify(100, 1.5) = 150: 211200 / 150 = 1408; / 50 = 28; + 2 = 30.
    ice = fighter(types=("ice",))
    assert MU.max_hit(plain("dark"), user, ice, board("ice")) == 30
    assert MU.max_hit(special, user, ice, board("ice")) == 41


def test_burn_and_stages():
    b = board()
    burned = fighter(types=("fire",), status="brn")
    assert MU.max_hit(plain("dark"), burned, fighter(), b) == 22
    assert MU.max_hit(plain("dark", physical=False), burned, fighter(), b) == 41
    # +2 attack against -1 defence: A = 240, D = 66: 422400 / 66 = 6400;
    # / 50 = 128; + 2 = 130.
    up = fighter(types=("fire",), boosts={"atk": 2})
    down = fighter(boosts={"def": -1})
    assert MU.max_hit(plain("dark"), up, down, b) == 130
    # A special move reads the special stages, not these.
    assert MU.max_hit(plain("dark", physical=False), up, down, b) == 41


def test_dex_moves_with_special_rules():
    b = board()
    user = fighter(types=("ghost",), dfn=150, boosts={"def": 1})
    foe = fighter(types=("ghost",), atk=130, dfn=100, spd=200, boosts={"atk": 1})
    # Defence as the attack stat (Fighting on Ghost is immune: use a Normal foe).
    press = facts("bodypress")
    assert press.attack_stat == "def" and press.est
    normal = fighter(types=("psychic",), dfn=100)
    # Fighting on Psychic: 0.5x. A = staged(150, +1) = 225: 22 * 80 * 225 =
    # 396000 / 100 = 3960; / 50 = 79; + 2 = 81; tr(81 / 2) = 40.
    assert MU.max_hit(press, user, normal, b) == 40
    # The target's attack (and its stages) as the attack stat. Dark on Ghost
    # 2x, P 95: A = staged(130, +1) = 195: 407550 / 100 = 4075; / 50 = 81;
    # + 2 = 83; x 2 = 166.
    foul = facts("foulplay")
    assert foul.from_target
    assert MU.max_hit(foul, user, foe, b) == 166
    # A special move against the physical defence. P 80, A 100 (spa), D 100
    # (def, not spd 200): 176000 / 100 = 1760; / 50 = 35; + 2 = 37; Psychic
    # on Ghost 1x.
    shock = facts("psyshock")
    assert shock.defence_stat == "def" and not shock.physical
    assert MU.max_hit(shock, user, foe, b) == 37
    # A fixed two-hit move: two hits of the one-hit damage.
    twice = facts("dualwingbeat")
    assert twice.hits == 2.0
    one = MU.max_hit(twice, user, normal, b)
    assert one is not None
    assert MU.hit(twice, user, normal, b)[3] == pytest.approx(0.925 * 2 * one / 175)
    assert facts("bulletseed").hits == pytest.approx(3.1)  # the 2-5 range
    assert facts("tripleaxel").hits == 3.0
    # An always-crit move: x1.5, truncated, before STAB. Flying 60 on Psychic:
    # 22 * 60 * 120 = 158400 / 100 = 1584; / 50 = 31; + 2 = 33; crit 49.
    assert facts("stormthrow").crit
    crit = MU.MoveFacts(**{**plain("flying", power=60).__dict__, "crit": True})
    assert MU.max_hit(crit, user, normal, b) == 49
    # A move whose type a callback decides: typeless. No STAB, no weather, no
    # immunity, no type step; the nominal power.
    ball = facts("weatherball")
    assert ball.type == "" and ball.est and ball.power == 50
    ghost = fighter(types=("ghost",))
    assert MU.hit(ball, fighter(types=("normal",)), ghost, board("fire"))[:3] == (
        1,
        0,
        0,
    )
    # Special 50, A 100, D 90: 110000 / 90 = 1222; / 50 = 24; + 2 = 26.
    assert MU.max_hit(ball, fighter(types=("normal",)), ghost, board("fire")) == 26
    # A Status move has no facts; fixed damage, one-hit KO and a zero-power
    # callback move have no estimate, but their type facts are set.
    assert MU.move_facts(E.moves_dex()["protect"]) is None
    assert MU.move_facts(None) is None and MU.move_facts({}) is None
    for name in ("seismictoss", "nightshade", "sheercold", "lowkick", "superfang"):
        assert not facts(name).est, name
    assert MU.hit(facts("lowkick"), user, normal, b) == (0, 0, -1, 0.0)
    assert MU.hit(facts("lowkick"), user, ghost, b) == (0, 1, 0, 0.0)
    assert MU.max_hit(facts("lowkick"), user, normal, b) is None
    # A callback move WITH a nominal power uses it.
    assert facts("facade").est and facts("facade").power == 70
    # Priority is a dex fact the candidate block reads.
    assert facts("fakeout").priority > 0 and facts("suckerpunch").priority > 0
    assert facts("dragontail").priority < 0 and facts("earthquake").priority == 0
    assert facts("earthquake").spread == MU.SPREAD_ALL
    assert facts("rockslide").spread == MU.SPREAD_FOES and facts("crunch").spread == ""


def test_no_stat_line_means_no_number():
    known, b = fighter(), board()
    unknown = MU.Fighter(None, ("fire",), {}, None, 1.0, (plain("fire"),))
    assert MU.hit(plain("dark"), unknown, known, b) == (0, 0, 0, 0.0)
    assert MU.hit(plain("dark"), known, unknown, b) == (0, 0, 0, 0.0)
    assert MU.best(unknown, known, b) == 0.0
    assert MU.speeds(unknown) is None
    assert MU.order(MU.speeds(known), None) == (0, 0, 0.0)
    assert MU.stat_line(None) is None and MU.stat_line({"hp": 1}) is None
    assert (
        MU.stat_line({"hp": "x", "atk": 1, "def": 1, "spa": 1, "spd": 1, "spe": 1})
        is None
    )


# --- the chain against a second implementation ---------------------------------


def _tr(value: float) -> int:
    return math.trunc(value)


def _sim_modify(value: int, factor: float) -> int:
    # sim/battle.ts:2332-2343
    modifier = _tr(factor * 4096)
    return _tr((_tr(value * modifier) + 2048 - 1) / 4096)


_BOOST_TABLE = [1, 1.5, 2, 2.5, 3, 3.5, 4]  # sim/pokemon.ts:583-589


def _sim_stat(stat: int, boost: int) -> int:
    boost = max(-6, min(6, boost))
    if boost >= 0:
        return math.floor(stat * _BOOST_TABLE[boost])
    return math.floor(stat / _BOOST_TABLE[-boost])


def simulator_style(entry, a_line, a_types, a_boosts, a_status, d_line, d_types,
                    d_boosts, weather_type, n_opposing, n_allies):  # fmt: skip
    """One hit at the maximum roll, written from the simulator's lines."""
    category = entry["category"]
    kind = E.to_id(entry["type"])
    attack_name = entry.get("overrideOffensiveStat") or (
        "atk" if category == "Physical" else "spa"
    )
    defence_name = entry.get("overrideDefensiveStat") or (
        "def" if category == "Physical" else "spd"
    )
    from_target = entry.get("overrideOffensivePokemon") == "target"
    source_line, source_boosts = (
        (d_line, d_boosts) if from_target else (a_line, a_boosts)
    )
    attack = _sim_stat(source_line[attack_name], source_boosts.get(attack_name, 0))
    def_boost = 0 if entry.get("ignoreDefensive") else d_boosts.get(defence_name, 0)
    defence = _sim_stat(d_line[defence_name], def_boost)
    if weather_type == "rock" and "rock" in d_types and defence_name == "spd":
        defence = _sim_modify(defence, 1.5)
    if weather_type == "ice" and "ice" in d_types and defence_name == "def":
        defence = _sim_modify(defence, 1.5)
    type_mod = 0
    for defender_type in d_types:
        value = CHART[kind].get(defender_type, 1.0)
        if value == 0:
            return 0
        type_mod += {2.0: 1, 0.5: -1}.get(value, 0)
    power = entry["basePower"]
    damage = _tr(_tr(_tr(_tr(2 * 50 / 5 + 2) * power * attack) / defence) / 50)
    damage += 2
    target = entry["target"]
    if (target == "allAdjacentFoes" and n_opposing > 1) or (
        target == "allAdjacent" and n_opposing + n_allies > 1
    ):
        damage = _sim_modify(damage, 0.75)
    if weather_type == "fire" and kind == "fire":
        damage = _sim_modify(damage, 1.5)
    if weather_type == "fire" and kind == "water":
        damage = _sim_modify(damage, 0.5)
    if weather_type == "water" and kind == "water":
        damage = _sim_modify(damage, 1.5)
    if weather_type == "water" and kind == "fire":
        damage = _sim_modify(damage, 0.5)
    if entry.get("willCrit"):
        damage = _tr(damage * 1.5)
    if kind in a_types:
        damage = _sim_modify(damage, 1.5)
    if type_mod > 0:
        for _ in range(type_mod):
            damage *= 2
    if type_mod < 0:
        for _ in range(-type_mod):
            damage = _tr(damage / 2)
    if a_status == "brn" and category == "Physical":
        damage = _sim_modify(damage, 0.5)
    return damage or 1


def test_chain_equals_a_second_reading_of_the_simulator_on_random_cases():
    rng = random.Random(20261010)
    dex, moves = E.pokedex(), E.moves_dex()
    formes = sorted(name for name, entry in dex.items() if entry.get("baseStats"))
    damaging = sorted(
        name
        for name, entry in moves.items()
        if (made := MU.move_facts(entry)) is not None and made.est and made.type
    )
    assert len(formes) > 1000 and len(damaging) > 400
    stats = ("atk", "def", "spa", "spd")
    checked = nonzero = modifiers = 0
    for _ in range(2500):
        a_name, d_name = rng.choice(formes), rng.choice(formes)
        a_entry, d_entry = dex[a_name], dex[d_name]
        a_line = MU.stat_line(a_entry["baseStats"])
        d_line = MU.stat_line(d_entry["baseStats"])
        assert a_line is not None and d_line is not None
        a_types = tuple(E.to_id(t) for t in a_entry["types"])
        d_types = tuple(E.to_id(t) for t in d_entry["types"])
        a_boosts = {s: rng.randint(-6, 6) if rng.random() < 0.4 else 0 for s in stats}
        d_boosts = {s: rng.randint(-6, 6) if rng.random() < 0.4 else 0 for s in stats}
        status = rng.choice([None, None, "brn", "par"])
        weather = rng.choice(["", "", "fire", "water", "rock", "ice"])
        n_opposing, n_allies = rng.choice([1, 2]), rng.choice([0, 1])
        name = rng.choice(damaging)
        want = simulator_style(
            moves[name], a_line, a_types, a_boosts, status, d_line, d_types,
            d_boosts, weather, n_opposing, n_allies,
        )  # fmt: skip
        user = MU.Fighter(a_line, a_types, a_boosts, status, 1.0)
        foe = MU.Fighter(d_line, d_types, d_boosts, None, 1.0)
        got = MU.max_hit(
            facts(name), user, foe, MU.Board(CHART, weather), n_opposing, n_allies
        )
        assert got == want, (name, a_name, d_name, a_boosts, d_boosts, status, weather)
        checked += 1
        nonzero += int(want > 0)
        modifiers += int(weather != "" or status == "brn" or n_opposing > 1)
    assert checked == 2500 and nonzero > 2000 and modifiers > 1500


# --- properties ------------------------------------------------------------------


def test_damage_is_monotonic_and_forme_blind():
    b = board()
    foe = fighter(types=("water",))
    last = 0
    for power in range(10, 200, 10):
        now = MU.max_hit(plain("dark", power=power), fighter(), foe, b)
        assert now is not None and now >= last
        last = now
    last = 0
    for stage in range(-6, 7):
        now = MU.max_hit(plain("dark"), fighter(boosts={"atk": stage}), foe, b)
        assert now is not None and now >= last
        last = now
    last = 10**9
    for stage in range(-6, 7):
        now = MU.max_hit(
            plain("dark"),
            fighter(),
            fighter(types=("water",), boosts={"def": stage}),
            b,
        )
        assert now is not None and now <= last
        last = now
    # Two formes with equal base stats and types give equal numbers: only
    # the stat line and the types are read.
    dex = E.pokedex()
    by_line: dict[tuple, list[str]] = {}
    for name, entry in dex.items():
        if entry.get("baseStats"):
            key = (
                tuple(sorted(entry["baseStats"].items())),
                tuple(entry.get("types", ())),
            )
            by_line.setdefault(key, []).append(name)
    twins = next(names for names in by_line.values() if len(set(names)) >= 2)
    lines = [MU.stat_line(dex[name]["baseStats"]) for name in twins[:2]]
    assert lines[0] == lines[1]


def test_stat_line_is_the_projects_imputation():
    """vgc_knowledge.ensure_stats (lines 210-218): 32 HP points, 32 in the
    better attack stat (atk on a tie), 2 in speed, neutral nature; HP =
    base + points + 75, every other stat = base + points + 20."""
    rows = {
        # hp, atk, def, spa, spd, spe  ->  the same order
        (78, 84, 78, 109, 85, 100): (185, 104, 98, 161, 105, 122),
        (108, 130, 95, 80, 85, 102): (215, 182, 115, 100, 105, 124),
        (95, 95, 95, 95, 95, 59): (202, 147, 115, 115, 115, 81),  # tie: atk
        (70, 85, 140, 85, 70, 20): (177, 137, 160, 105, 90, 42),
    }
    for base, want in rows.items():
        made = MU.stat_line(dict(zip(MU.STATS, base)))
        assert made is not None
        assert tuple(made[name] for name in MU.STATS) == want
        assert made["base_spe"] == base[5]
        assert made["physical"] == int(base[1] >= base[3])


def test_best_takes_the_strongest_known_hit():
    b = board()
    weak, strong = plain("dark", power=40), plain("ice", power=90)
    foe = fighter(types=("dragon",))
    user = MU.Fighter(line(), ("fire",), {}, None, 1.0, (weak, strong))
    assert MU.best(user, foe, b) == MU.hit(strong, user, foe, b)[3]
    nobody = MU.Fighter(line(), ("fire",), {}, None, 1.0, ())
    assert MU.best(nobody, foe, b) == 0.0
    ground_only = MU.Fighter(line(), ("fire",), {}, None, 1.0, (plain("ground"),))
    assert MU.best(ground_only, fighter(types=("flying",)), b) == 0.0


# --- speed ----------------------------------------------------------------------


def test_speed_order_is_antisymmetric_and_reverses():
    fast = fighter(base_spe=100)
    slow = fighter(base_spe=60)
    mine, theirs = MU.speeds(fast), MU.speeds(slow)
    assert mine is not None and theirs is not None
    # slowest (base + 20) * 0.9, assumed base + 22, fastest (base + 52) * 1.1
    assert mine == pytest.approx((108.0, 122.0, 167.2))
    assert theirs == pytest.approx((72.0, 82.0, 123.2))
    first, sure, spd = MU.order(mine, theirs)
    assert (first, sure) == (1, 0)  # 108 is not above 123.2: not sure
    assert spd == pytest.approx(math.log2(122 / 82))
    back = MU.order(theirs, mine)
    assert back == (-first, -sure, pytest.approx(-spd))
    assert MU.order(mine, theirs, reverse=True) == (-1, 0, pytest.approx(-spd))
    assert MU.order(mine, mine) == (0, 0, 0.0)
    # Sure only when the ranges do not overlap: base 150 against base 30.
    quick, crawl = MU.speeds(fighter(base_spe=150)), MU.speeds(fighter(base_spe=30))
    assert MU.order(quick, crawl) == (1, 1, 1.0)  # ratio clipped at 2x
    assert MU.order(crawl, quick) == (-1, -1, -1.0)
    assert MU.order(quick, crawl, reverse=True) == (-1, -1, -1.0)


def test_speed_modifiers():
    base = MU.speeds(fighter(base_spe=80))
    assert base is not None
    para = MU.speeds(fighter(base_spe=80, status="par"))
    up = MU.speeds(fighter(base_spe=80, boosts={"spe": 1}))
    down = MU.speeds(fighter(base_spe=80, boosts={"spe": -1}))
    wind = MU.speeds(MU.Fighter(line(base_spe=80), (), {}, None, 1.0, (), 2.0))
    assert para == pytest.approx(tuple(0.5 * v for v in base))
    assert up == pytest.approx(tuple(1.5 * v for v in base))
    assert down == pytest.approx(tuple(v / 1.5 for v in base))
    assert wind == pytest.approx(tuple(2.0 * v for v in base))
    assert MU.order(wind, base)[0] == 1 and MU.order(para, base)[0] == -1


def test_rules_are_found_by_dex_property_one_id_each():
    found = MU.rules(E.moves_dex())
    assert len(found.speed_double) == 1 and len(found.speed_reverse) == 1
    assert found.speed_double == ("tailwind",)
    assert found.speed_reverse == ("trickroom",)
    # One type per weather; the four types the rule table knows are all set
    # by some weather of the installed dex.
    assert found.weather_types["sunnyday"] == "fire"
    assert found.weather_types["raindance"] == "water"
    assert found.weather_types["sandstorm"] == "rock"
    assert found.weather_types["snowscape"] == "ice"
    assert set(found.weather_types.values()) == set(MU.WEATHER_MOVE) | set(
        MU.WEATHER_GUARD
    )
    # A weather two Status moves give different types is dropped; junk is survived.
    clash = {
        "a": {"category": "Status", "weather": "Odd", "type": "Fire"},
        "b": {"category": "Status", "weather": "odd", "type": "Water"},
        "c": {"category": "Status", "weather": "Fine", "type": "Rock"},
        "d": "junk",
    }
    assert MU.rules(clash).weather_types == {"fine": "rock"}
    assert MU.rules({}) == MU.Rules({}, (), ())
    assert MU.signature(found)["version"] == MU.MATCHUP_VERSION


def test_quantisation():
    assert [MU.q(v) for v in (-1.0, 0.0, 0.004, 0.01, 0.5, 1.0, 2.0, 9.0)] == [
        0,
        0,
        0,
        1,
        25,
        50,
        100,
        100,
    ]
    assert [MU.qlog(v) for v in (-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0)] == [
        -50,
        -50,
        -25,
        0,
        25,
        50,
        50,
    ]
    for value in (0.013, 0.26, 0.77):
        assert MU.qlog(-value) == -MU.qlog(value)
    assert MU.kills(0.5, 0.5) and MU.kills(0.6, 0.5) and not MU.kills(0.49, 0.5)
    assert not MU.kills(0.0, 0.0)  # an immune or absent hit never "kills"
    assert MU.ko_ratio(0.3, 0.0) == pytest.approx(30.0)
    assert len(MU.MU_CAND_COLUMNS) == len(MU.MU_CAND_SCALES) == 8
    assert len(MU.MU_SLOT_COLUMNS) == len(MU.MU_SLOT_SCALES) == 10
    assert len(MU.MU_ROSTER_COLUMNS) == len(MU.MU_ROSTER_SCALES) == 4


# --- library rules ----------------------------------------------------------------


def test_matchup_module_names_nothing_from_the_dex_and_asserts_nothing():
    path = ROOT / "vgc_bench/src/oppmodel/matchup.py"
    fz = F.Featurizer.build()
    known = set(E.moves_dex()) | set(E.pokedex())
    known |= set(fz.vocab.items[3:]) | set(fz.vocab.abilities[3:])
    known |= set(fz.vocab.weathers) | set(fz.vocab.side_conditions)
    known |= set(fz.vocab.pseudo) | set(fz.vocab.terrains)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    found = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        and node.value in known
    }
    assert not found, sorted(found)
    asserts = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    assert not asserts, asserts
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    # Standard library only: no torch, no numpy, no poke-env, no battle object.
    assert imported <= {"__future__", "math", "dataclasses", "typing"}, imported
