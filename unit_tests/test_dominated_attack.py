"""dominated_attack (ladder 2026-09-24): the policy picked the weaker of its own
attacks and nothing compared one move with another -- Leaf Storm at -2 into
Mega Emboar over Sludge Bomb, Ice Beam into Farigiraf over a full-HP Water Spout,
Water Pulse into Mega Camerupt over a Water Spout that also hit Hatterene.

The ladder positions are rebuilt from protocol lines with poke-env's own parser,
our sets come from teams/candidates_mc/T6.txt, and the real damage calculator
scores them; the candidate pairs and their probabilities are the ones the bot
logged in those turns."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from poke_env.battle import DoubleBattle, Move, Pokemon
from poke_env.data import to_id_str
from poke_env.teambuilder import Teambuilder

from vgc_bench.src import guards as G

ROOT = Path(__file__).resolve().parents[1]
SETS = {
    to_id_str(mon.species or mon.nickname): mon
    for mon in Teambuilder.parse_showdown_team(
        (ROOT / "teams/candidates_mc/T6.txt").read_text()
    )
}


def _position(lines: list[str], mega_launcher: bool = False) -> DoubleBattle:
    battle = DoubleBattle(
        "battle-gen9championsvgc2026regmc-fixture",
        "antonius1",
        logging.getLogger("dominated_attack"),
        gen=9,
    )
    battle._player_role = "p1"
    header = ["|player|p1|antonius1|1|1100", "|player|p2|rival|1|1100", "|gen|9"]
    for line in [*header, *lines]:
        battle.parse_message(line.split("|"))
    for mon in battle.team.values():
        mon._update_from_teambuilder(SETS[to_id_str(mon.base_species)])
    if mega_launcher:  # the live request reports the Mega's ability; the set cannot
        mega = battle.active_pokemon[0]
        assert mega is not None
        mega._ability = "megalauncher"
    return battle


def _action(battle: DoubleBattle, pos: int, move_id: str, target: int) -> int:
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


def _run(battle, pairs):
    cands = [G.Candidate(actions, prob) for actions, prob in pairs]
    report = G.GuardReport()
    return G.guard_dominated_attack(battle, cands, report), report


def test_sludge_bomb_replaces_minus_two_leaf_storm_into_mega_emboar():
    battle = _position(
        [
            "|switch|p1a: Venusaur|Venusaur, L50, M|1/100",
            "|switch|p2a: Emboar|Emboar, L50, M|30/100",
            "|detailschange|p2a: Emboar|Emboar-Mega, L50, M",
            "|-weather|SunnyDay",
            "|-unboost|p1a: Venusaur|spa|2",
            "|-unboost|p2a: Emboar|spd|1",
        ]
    )
    leaf = _action(battle, 0, "leafstorm", 1)
    sleep = _action(battle, 0, "sleeppowder", 1)
    sludge = _action(battle, 0, "sludgebomb", 1)
    out, report = _run(
        battle, [((leaf, 0), 0.755), ((sleep, 0), 0.114), ((sludge, 0), 0.069)]
    )
    assert out[0].actions == (sludge, 0)
    assert out[0].prob == pytest.approx(0.755)  # survives the opponent reranker
    assert report.demotions["dominated_attack:promoted"] == 1
    assert "dominated_attack" in report.stages


def test_water_spout_replaces_turn_one_ice_beam_into_farigiraf():
    battle = _position(
        [
            "|switch|p1a: Blastoise|Blastoise, L50, M|100/100",
            "|detailschange|p1a: Blastoise|Blastoise-Mega, L50, M",
            "|switch|p1b: Farigiraf|Farigiraf, L50, M|100/100",
            "|switch|p2a: Farigiraf|Farigiraf, L50, M|100/100",
            "|switch|p2b: Incineroar|Incineroar, L50, M|100/100",
        ],
        mega_launcher=True,
    )
    trick_room = _action(battle, 1, "trickroom", 0)
    ice = _action(battle, 0, "icebeam", 1)
    spout = _action(battle, 0, "waterspout", 0)
    pulse = _action(battle, 0, "waterpulse", 2)
    out, _ = _run(
        battle,
        [
            ((ice, trick_room), 0.222),
            ((spout, trick_room), 0.188),
            ((pulse, trick_room), 0.143),
        ],
    )
    assert out[0].actions == (spout, trick_room)


def test_water_spout_replaces_water_pulse_into_mega_camerupt_in_rain():
    battle = _position(
        [
            "|switch|p1a: Blastoise|Blastoise, L50, M|100/100",
            "|detailschange|p1a: Blastoise|Blastoise-Mega, L50, M",
            "|switch|p1b: Farigiraf|Farigiraf, L50, M|60/100",
            "|switch|p2a: Camerupt|Camerupt, L50, M|100/100",
            "|detailschange|p2a: Camerupt|Camerupt-Mega, L50, M",
            "|switch|p2b: Hatterene|Hatterene, L50, F|100/100",
            "|-weather|RainDance",
        ],
        mega_launcher=True,
    )
    pulse = _action(battle, 0, "waterpulse", 1)
    spout = _action(battle, 0, "waterspout", 0)
    psychic_2 = _action(battle, 1, "psychic", 2)
    psychic_1 = _action(battle, 1, "psychic", 1)
    out, _ = _run(
        battle,
        [
            ((pulse, psychic_2), 0.263),
            ((pulse, psychic_1), 0.263),
            ((spout, psychic_1), 0.200),
            ((spout, psychic_2), 0.068),
        ],
    )
    # Psychic stays on Hatterene: Water Spout already knocks Camerupt out, so
    # moving Psychic onto it (neutral instead of resisted) would be wasted damage.
    assert out[0].actions == (spout, psychic_2)


def _sun_blastoise(blastoise_hp: int, partner: str, snorlax_hp: int, sinistcha_hp: int):
    return _position(
        [
            f"|switch|p1a: Blastoise|Blastoise, L50, M|{blastoise_hp}/100",
            "|detailschange|p1a: Blastoise|Blastoise-Mega, L50, M",
            f"|switch|p1b: {partner}|{partner}, L50, M|100/100",
            f"|switch|p2a: Snorlax|Snorlax, L50, M|{snorlax_hp}/100",
            f"|switch|p2b: Sinistcha|Sinistcha, L50|{sinistcha_hp}/100",
            "|-weather|SunnyDay",
        ],
        mega_launcher=True,
    )


def test_super_effective_ice_beam_replaces_sun_halved_water_pulse():
    battle = _sun_blastoise(50, "Farigiraf", 100, 100)
    trick_room = _action(battle, 1, "trickroom", 0)
    pulse = _action(battle, 0, "waterpulse", 2)
    ice = _action(battle, 0, "icebeam", 2)
    out, _ = _run(battle, [((pulse, trick_room), 0.38), ((ice, trick_room), 0.09)])
    assert out[0].actions == (ice, trick_room)


def test_target_choice_stays_with_the_policy():
    """A bigger hit on the OTHER foe is not promoted: which foe to hit is the
    policy's strategic call (threat, focus fire), not a damage comparison."""
    battle = _sun_blastoise(50, "Farigiraf", 100, 100)
    trick_room = _action(battle, 1, "trickroom", 0)
    pulse_snorlax = _action(battle, 0, "waterpulse", 1)
    ice_sinistcha = _action(battle, 0, "icebeam", 2)
    out, report = _run(
        battle,
        [((pulse_snorlax, trick_room), 0.38), ((ice_sinistcha, trick_room), 0.09)],
    )
    assert out[0].actions == (pulse_snorlax, trick_room)
    assert not report.stages


def test_nothing_to_gain_when_the_partner_already_knocks_both_out():
    """Game 1, turn 4 as played: Torkoal's sun Eruption KO'd Snorlax (1%) and
    Sinistcha (~30%), so Blastoise's choice could not matter."""
    battle = _sun_blastoise(8, "Torkoal", 1, 30)
    eruption = _action(battle, 1, "eruption", 0)
    pulse = _action(battle, 0, "waterpulse", 2)
    ice = _action(battle, 0, "icebeam", 2)
    out, report = _run(battle, [((pulse, eruption), 0.384), ((ice, eruption), 0.09)])
    assert out[0].actions == (pulse, eruption)
    assert not report.stages


def test_a_policy_attack_that_is_already_best_is_left_alone():
    battle = _position(
        [
            "|switch|p1a: Blastoise|Blastoise, L50, M|100/100",
            "|detailschange|p1a: Blastoise|Blastoise-Mega, L50, M",
            "|switch|p1b: Farigiraf|Farigiraf, L50, M|100/100",
            "|switch|p2a: Farigiraf|Farigiraf, L50, M|100/100",
            "|switch|p2b: Incineroar|Incineroar, L50, M|100/100",
        ],
        mega_launcher=True,
    )
    trick_room = _action(battle, 1, "trickroom", 0)
    spout = _action(battle, 0, "waterspout", 0)
    ice = _action(battle, 0, "icebeam", 1)
    out, report = _run(battle, [((spout, trick_room), 0.4), ((ice, trick_room), 0.2)])
    assert out[0].actions == (spout, trick_room)
    assert not report.stages


# ---- rule boundaries on stubbed states --------------------------------------

MOVES = {
    0: ["fakeout", "earthquake", "icebeam", "hydropump"],
    1: ["protect", "psychic", "trickroom", "helpinghand"],
}


def _stub_decode(_battle, action, pos):
    action = int(action)
    if action < 7:
        return NS(order=None, move_target=0)
    band, offset = (action - 7) // 20, (action - 7) % 20
    return NS(
        order=Move(MOVES[pos][offset // 5], gen=9),
        move_target=offset % 5 - 2,
        mega=band == 1,
    )


def _stub_action(pos, move_id, target, mega=False):
    return 7 + 20 * mega + 5 * MOVES[pos].index(move_id) + (target + 2)


@pytest.fixture
def stub(monkeypatch):
    foes = [Pokemon(gen=9, species="garchomp"), Pokemon(gen=9, species="amoonguss")]
    for foe in foes:  # a bare Pokemon object reports 0 HP
        foe._current_hp, foe._max_hp = 100, 100
    battle = NS(
        active_pokemon=[
            Pokemon(gen=9, species="blastoise"),
            Pokemon(gen=9, species="farigiraf"),
        ],
        opponent_active_pokemon=foes,
        weather={},
        fields={},
    )
    damage = {
        "icebeam": (0.50, 0.60),
        "hydropump": (0.30, 0.36),
        "fakeout": (0.05, 0.06),
    }
    monkeypatch.setattr(G, "_decode", _stub_decode)
    monkeypatch.setattr(
        G.K, "damage_fraction", lambda b, a, d, m: damage.get(m.id, (0.8, 0.9))
    )
    return battle


def test_fake_out_is_never_swapped_for_damage(stub):
    top = G.Candidate((_stub_action(0, "fakeout", 1), 7), 0.6)
    ice = G.Candidate((_stub_action(0, "icebeam", 1), 7), 0.1)
    out, _ = _run_stub(stub, [top, ice])
    assert out[0] is top


def test_close_comparison_stands_down(stub, monkeypatch):
    monkeypatch.setattr(
        G.K,
        "damage_fraction",
        # Ice Beam 0.36 expected vs Hydro Pump 0.8 x 0.38 = 0.30: under 1.25x
        lambda b, a, d, m: {"icebeam": (0.34, 0.38), "hydropump": (0.36, 0.40)}[m.id],
    )
    top = G.Candidate((_stub_action(0, "hydropump", 1), 7), 0.6)
    ice = G.Candidate((_stub_action(0, "icebeam", 1), 7), 0.1)
    out, _ = _run_stub(stub, [top, ice])
    assert out[0] is top


def test_clear_domination_promotes_and_keeps_the_partner(stub):
    top = G.Candidate((_stub_action(0, "hydropump", 1), 7), 0.6)
    other_partner = G.Candidate((_stub_action(0, "icebeam", 1), 8), 0.2)
    same_partner = G.Candidate((_stub_action(0, "icebeam", 1), 7), 0.1)
    out, _ = _run_stub(stub, [top, other_partner, same_partner])
    assert out[0] is same_partner


def test_ally_hitting_and_other_mega_choices_are_not_promoted(stub):
    top = G.Candidate((_stub_action(0, "hydropump", 1), 7), 0.6)
    quake = G.Candidate((_stub_action(0, "earthquake", 0), 7), 0.2)
    mega_ice = G.Candidate((_stub_action(0, "icebeam", 1, mega=True), 7), 0.1)
    out, _ = _run_stub(stub, [top, quake, mega_ice])
    assert out[0] is top


def test_stands_down_without_the_calculator(stub, monkeypatch):
    monkeypatch.setattr(G.K, "damage_fraction", lambda b, a, d, m: None)
    top = G.Candidate((_stub_action(0, "hydropump", 1), 7), 0.6)
    ice = G.Candidate((_stub_action(0, "icebeam", 1), 7), 0.1)
    out, report = _run_stub(stub, [top, ice])
    assert out[0] is top
    assert report.demotions["dominated_attack:no_calc"] == 1


def _run_stub(battle, cands):
    report = G.GuardReport()
    return G.guard_dominated_attack(battle, cands, report), report


def test_capped_roll_expectation_and_knockout_chance():
    assert G._capped_roll(0.2, 0.4, 1.0) == pytest.approx((0.3, 0.0))
    assert G._capped_roll(0.6, 0.8, 0.5) == pytest.approx((0.5, 1.0))
    expected, p_ko = G._capped_roll(0.2, 0.6, 0.4)
    assert p_ko == pytest.approx(0.5)
    assert expected == pytest.approx(0.35)  # half the rolls at 0.2-0.4, half capped
    assert G._capped_roll(0.3, 0.5, 0.0) == (0.0, 0.0)


def test_registered_but_opt_in():
    assert "dominated_attack" in G.GUARDS
    assert "dominated_attack" not in G.HARD_GUARDS
    order = G.GUARD_ORDER
    assert order.index("dominated_attack") < order.index("resisted_target")


def test_the_partner_s_knockout_is_not_counted_twice(stub, monkeypatch):
    """Slot 1 finishing a foe makes slot 0's attack into that foe worth nothing."""
    damage = {"icebeam": (0.50, 0.60), "hydropump": (0.30, 0.36), "psychic": (1.2, 1.4)}
    monkeypatch.setattr(G.K, "damage_fraction", lambda b, a, d, m: damage[m.id])
    psychic_1 = 7 + 5 * MOVES[1].index("psychic") + 3
    top = G.Candidate((_stub_action(0, "hydropump", 1), psychic_1), 0.6)
    ice_same_foe = G.Candidate((_stub_action(0, "icebeam", 1), psychic_1), 0.1)
    out, _ = _run_stub(stub, [top, ice_same_foe])
    assert out[0] is top  # Psychic already KOs that foe: neither attack adds value
