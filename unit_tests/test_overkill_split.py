"""overkill_split guard (2026-09-06): two attacks stacked on a foe one of them
already KOs waste the second and hand its redirect to an immune foe (Last
Respects into Oranguru after a 1%-HP Whimsicott died first). Facts only, opt-in."""

from types import SimpleNamespace as NS

import pytest
from poke_env.battle import Move, Pokemon, PokemonType

from vgc_bench.src import guards as G

# our slot 0 = Garchomp, our slot 1 = Basculegion
MOVES = {
    0: ["earthquake", "dragonclaw", "rocktomb", "protect"],
    1: ["wavecrash", "lastrespects", "aquajet", "protect"],
}


def _decode(_battle, action, pos):
    action = int(action)
    if action < 7:
        return NS(order=None, move_target=0)
    band, offset = (action - 7) // 5, (action - 7) % 5
    return NS(order=Move(MOVES[pos][band % 4], gen=9), move_target=offset - 2)


def _action(pos, move_id, target):
    return 7 + 5 * MOVES[pos].index(move_id) + (target + 2)


@pytest.fixture
def position(monkeypatch):
    whimsicott, oranguru = (
        Pokemon(gen=9, species="whimsicott"),
        Pokemon(gen=9, species="oranguru"),
    )
    battle = NS(
        active_pokemon=[
            Pokemon(gen=9, species="garchomp"),
            Pokemon(gen=9, species="basculegion"),
        ],
        opponent_active_pokemon=[whimsicott, oranguru],  # slot 1 = Whimsicott at 1%
        weather={},
        fields={},
    )
    monkeypatch.setattr(G, "_decode", _decode)
    monkeypatch.setattr(G.K, "damage_fraction", lambda *a, **k: None)
    monkeypatch.setattr(
        G.K, "guaranteed_ko", lambda b, a, d, m: m.id == "rocktomb" and d is whimsicott
    )
    monkeypatch.setattr(
        G.K,
        "deals_no_damage",
        lambda b, a, d, m: m.type == PokemonType.GHOST and d is oranguru,
    )
    return battle


def _run(battle, cands):
    report = G.GuardReport()
    return G.guard_overkill_split(battle, cands, report), report


def test_the_ladder_position_promotes_rock_tomb_plus_wave_crash_into_oranguru(position):
    top = G.Candidate((_action(0, "rocktomb", 1), _action(1, "lastrespects", 1)), 0.183)
    protect = G.Candidate(
        (_action(0, "protect", 0), _action(1, "lastrespects", 1)), 0.126
    )
    same = G.Candidate((_action(0, "rocktomb", 1), _action(1, "wavecrash", 1)), 0.119)
    split = G.Candidate((_action(0, "rocktomb", 1), _action(1, "wavecrash", 2)), 0.117)
    out, report = _run(position, [top, protect, same, split])
    assert out[0] is split and len(out) == 4
    assert report.demotions["overkill_split:promoted"] == 1
    assert "overkill_split" in report.stages


def test_no_ranked_split_and_immune_twin_stands_down(position):
    top = G.Candidate((_action(0, "rocktomb", 1), _action(1, "lastrespects", 1)), 0.2)
    protect = G.Candidate(
        (_action(0, "protect", 0), _action(1, "lastrespects", 1)), 0.1
    )
    out, report = _run(position, [top, protect])
    assert out[0] is top and not report.stages
    assert report.demotions["overkill_split:no_alternative"] == 1


def test_no_ranked_split_injects_the_re_aimed_move_when_it_does_damage(position):
    top = G.Candidate((_action(0, "rocktomb", 1), _action(1, "wavecrash", 1)), 0.2)
    out, report = _run(position, [top])
    assert out[0].actions == (_action(0, "rocktomb", 1), _action(1, "wavecrash", 2))
    assert out[0].prob == top.prob and out[1] is top
    assert report.demotions["overkill_split:injected"] == 1


def test_without_a_guaranteed_ko_the_pair_is_left_alone(position, monkeypatch):
    monkeypatch.setattr(G.K, "guaranteed_ko", lambda *a, **k: False)
    top = G.Candidate((_action(0, "rocktomb", 1), _action(1, "lastrespects", 1)), 0.2)
    split = G.Candidate((_action(0, "rocktomb", 1), _action(1, "wavecrash", 2)), 0.1)
    out, report = _run(position, [top, split])
    assert out[0] is top and not report.stages


def test_attacks_on_different_foes_or_spread_moves_are_ignored(position):
    different = G.Candidate(
        (_action(0, "rocktomb", 1), _action(1, "wavecrash", 2)), 0.2
    )
    out, report = _run(position, [different])
    assert out[0] is different and not report.stages
    spread = G.Candidate(
        (_action(0, "earthquake", 0), _action(1, "lastrespects", 1)), 0.2
    )
    out, report = _run(position, [spread])
    assert out[0] is spread and not report.stages


def test_registered_but_opt_in():
    assert "overkill_split" in G.GUARDS and "overkill_split" in G.GUARD_ORDER
    assert "overkill_split" not in G.HARD_GUARDS


def test_promoted_split_inherits_the_top_probability(position):
    top = G.Candidate((_action(0, "rocktomb", 1), _action(1, "lastrespects", 1)), 0.183)
    split = G.Candidate((_action(0, "rocktomb", 1), _action(1, "wavecrash", 2)), 0.117)
    out, _ = _run(position, [top, split])
    assert out[0] is split and split.prob == 0.183
