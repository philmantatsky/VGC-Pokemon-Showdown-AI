from types import SimpleNamespace as NS

import torch
from poke_env.battle import Effect, Move, Pokemon, SideCondition, Weather
from poke_env.environment import DoublesEnv

from vgc_bench.src import guards as G
from vgc_bench.src.opponent_reranker import rerank_candidates
from vgc_bench.src.opponent_tactics import MovePrediction
from vgc_bench.src.policy_player import PolicyPlayer


def test_live_legality_gate_rejects_an_invalid_trapped_switch(monkeypatch):
    def reject_switch(actions, _battle, **_kwargs):
        if int(actions[0]) in range(1, 7):
            raise ValueError("trapped switch")
        return NS()

    monkeypatch.setattr(DoublesEnv, "action_to_order", reject_switch)
    assert not PolicyPlayer._action_pair_is_legal(NS(), (1, 9))
    assert PolicyPlayer._action_pair_is_legal(NS(), (9, 9))


def test_live_exact_hard_stack_keeps_consecutive_protect_guard():
    assert "protect_spam" in G.HARD_GUARDS


def _order(move_id: str, target: int = 0):
    return NS(order=Move(move_id, gen=9), move_target=target)


def test_empty_slot_retargets_to_the_only_live_foe():
    farigiraf = Pokemon(gen=9, species="farigiraf")
    battle = NS(opponent_active_pokemon=[None, farigiraf])

    targets = G.resolved_foe_targets(battle, _order("lastrespects", 1))

    assert targets == [farigiraf]


def test_revealed_armor_tail_demotes_sucker_punch(monkeypatch):
    kingambit = Pokemon(gen=9, species="kingambit")
    farigiraf = Pokemon(gen=9, species="farigiraf")
    farigiraf.ability = "Armor Tail"
    battle = NS(
        active_pokemon=[kingambit, None],
        opponent_active_pokemon=[None, farigiraf],
        fields={},
    )
    candidates = [G.Candidate((1, 0), 0.55), G.Candidate((2, 0), 0.45)]

    def decode(_battle, action, _pos):
        if action == 1:
            return _order("suckerpunch", 1)
        if action == 2:
            return _order("kowtowcleave", 1)
        return NS(order=None, move_target=0)

    monkeypatch.setattr(G, "_decode", decode)
    report = G.GuardReport()

    ranked = G.guard_priority_block(battle, candidates, report)

    assert ranked[0].actions == (2, 0)
    assert ranked[-1].demoted_by == "priority_block"
    assert report.stages == ["priority_block"]


def test_hidden_farigiraf_usage_prior_avoids_the_first_sucker_punch(monkeypatch):
    kingambit = Pokemon(gen=9, species="kingambit")
    farigiraf = Pokemon(gen=9, species="farigiraf")
    battle = NS(
        active_pokemon=[kingambit, None],
        opponent_active_pokemon=[None, farigiraf],
        fields={},
    )
    candidates = [G.Candidate((1, 0), 0.55), G.Candidate((2, 0), 0.45)]

    def decode(_battle, action, _pos):
        if action == 1:
            return _order("suckerpunch", 1)
        if action == 2:
            return _order("kowtowcleave", 1)
        return NS(order=None, move_target=0)

    monkeypatch.setattr(G, "_decode", decode)
    monkeypatch.setattr(PolicyPlayer, "use_moveset_prior", True)
    report = G.GuardReport()

    assert G.candidate_uses_blocked_priority(battle, candidates[0])
    ranked = G.guard_priority_block(battle, candidates, report)

    assert ranked[0].actions == (2, 0)
    assert ranked[-1].demoted_by == "priority_block_prior"
    assert report.stages == ["priority_block_prior"]


def test_safe_earthquake_combo_can_overcome_a_large_policy_gap(monkeypatch):
    garchomp = NS(fainted=False, current_hp_fraction=1.0)
    charizard = NS(fainted=False, current_hp_fraction=1.0)
    tyranitar = NS(fainted=False, current_hp_fraction=0.58)
    battle = NS(
        active_pokemon=[garchomp, charizard], opponent_active_pokemon=[tyranitar, None]
    )
    dragon_weather = G.Candidate((1, 2), 0.30)
    quake_weather = G.Candidate((3, 2), 0.05)

    def decode(_battle, action, pos):
        if pos == 0 and action == 1:
            return _order("dragonclaw", 1)
        if pos == 0 and action == 3:
            return _order("earthquake")
        if pos == 1 and action == 2:
            return _order("weatherball", 1)
        return NS(order=None, move_target=0)

    def damage(_battle, _attacker, defender, move):
        assert defender is tyranitar
        minimum = {"dragonclaw": 0.20, "weatherball": 0.22, "earthquake": 0.42}[move.id]
        return minimum, minimum

    monkeypatch.setattr(G, "_decode", decode)
    monkeypatch.setattr(G.K, "damage_fraction", damage)
    monkeypatch.setattr(
        G.K,
        "deals_no_damage",
        lambda _battle, _attacker, defender, move: (
            defender is charizard and move.id == "earthquake"
        ),
    )
    report = G.GuardReport()

    ranked = G.guard_safe_spread_ko(battle, [dragon_weather, quake_weather], report)

    assert ranked[0] is quake_weather
    assert report.stages == ["guaranteed_ko"]


def test_candidate_prefix_preserves_switch_and_non_mega_alternatives():
    probabilities = torch.zeros(107)
    for action, probability in zip(range(27, 33), (0.30, 0.25, 0.20, 0.15, 0.06, 0.03)):
        probabilities[action] = probability
    probabilities[4] = 0.005
    probabilities[14] = 0.004

    prefix = {
        action: (probability, strategic)
        for probability, action, strategic in G._candidate_action_prefix(
            probabilities, 6
        )
    }

    assert 4 in prefix, "best legal switch must survive the policy top-k"
    assert 14 in prefix, "best ordinary move must survive a Mega-heavy top-k"
    assert prefix[4][1] is True
    assert prefix[14][1] is True


def test_solo_guaranteed_earthquake_beats_resisted_dragon_claw(monkeypatch):
    garchomp = NS(fainted=False, current_hp_fraction=1.0)
    archaludon = NS(fainted=False, current_hp_fraction=0.43)
    battle = NS(
        active_pokemon=[garchomp, None], opponent_active_pokemon=[archaludon, None]
    )
    dragon = G.Candidate((1, 0), 0.56)
    earthquake = G.Candidate((2, 0), 0.32)

    def decode(_battle, action, _pos):
        if action == 1:
            return _order("dragonclaw", 1)
        if action == 2:
            return _order("earthquake")
        return NS(order=None, move_target=0)

    monkeypatch.setattr(G, "_decode", decode)
    monkeypatch.setattr(
        G.K,
        "damage_fraction",
        lambda _battle, _attacker, _defender, move: (
            (0.19, 0.21) if move.id == "dragonclaw" else (0.60, 0.70)
        ),
    )
    monkeypatch.setattr(G, "_has_resisted_attack", lambda *_args: True)
    report = G.GuardReport()

    ranked = G.guard_guaranteed_ko(battle, [dragon, earthquake], report)

    assert ranked[0] is earthquake
    assert report.stages == ["guaranteed_ko"]


def test_last_respects_uses_fainted_ally_base_power(monkeypatch):
    attacker = NS(fainted=False)
    fainted_one = NS(fainted=True)
    fainted_two = NS(fainted=True)
    defender = NS(fainted=False)
    battle = NS(
        team={"p1: Basculegion": attacker, "p1: A": fainted_one, "p1: B": fainted_two},
        opponent_team={"p2: Pelipper": defender},
    )
    original = Move("lastrespects", gen=9)

    monkeypatch.setattr(G.K, "ensure_stats", lambda _mon: None)

    def calculate(attacker_id, defender_id, move, _battle):
        assert attacker_id == "p1: Basculegion"
        assert defender_id == "p2: Pelipper"
        assert move.base_power == 150
        return 100, 120

    monkeypatch.setattr(G.K, "calculate_damage", calculate)

    assert G.K.damage_range(battle, attacker, defender, original) == (100, 120)
    assert original.base_power == 50, "the shared move object must not be mutated"


def test_last_respects_ko_beats_rain_aqua_jet_at_observed_policy_gap(monkeypatch):
    basculegion = NS(fainted=False, current_hp_fraction=0.50)
    charizard = NS(fainted=False, current_hp_fraction=1.0)
    pelipper = NS(fainted=False, current_hp_fraction=0.62)
    sneasler = NS(fainted=False, current_hp_fraction=1.0)
    battle = NS(
        active_pokemon=[basculegion, charizard],
        opponent_active_pokemon=[pelipper, sneasler],
    )
    aqua_jet = G.Candidate((1, 3), 0.256)
    last_respects = G.Candidate((2, 3), 0.156)

    def decode(_battle, action, pos):
        if pos == 0 and action == 1:
            return _order("aquajet", 1)
        if pos == 0 and action == 2:
            return _order("lastrespects", 1)
        if pos == 1 and action == 3:
            return _order("heatwave")
        return NS(order=None, move_target=0)

    def damage(_battle, _attacker, defender, move):
        if move.id == "aquajet" and defender is pelipper:
            return 0.38, 0.45
        if move.id == "lastrespects" and defender is pelipper:
            return 0.75, 0.88
        return None

    monkeypatch.setattr(G, "_decode", decode)
    monkeypatch.setattr(G.K, "damage_fraction", damage)
    monkeypatch.setattr(G, "_has_resisted_attack", lambda *_args: True)
    report = G.GuardReport()

    ranked = G.guard_guaranteed_ko(battle, [aqua_jet, last_respects], report)

    assert ranked[0] is last_respects
    assert report.stages == ["guaranteed_ko"]


def test_generic_reranker_ignores_preserved_strategic_candidate():
    top = G.Candidate((1, 1), 0.55)
    preserved_switch = G.Candidate((2, 1), 0.40, strategic_only=True)

    ranked, report = rerank_candidates(
        NS(), [top, preserved_switch], opponent_moves=None, opponent_switches=None
    )

    assert ranked == [top, preserved_switch]
    assert report is None


def test_two_on_one_focus_attacks_with_both_slots(monkeypatch):
    garchomp = NS(fainted=False, current_hp_fraction=0.4)
    kingambit = NS(fainted=False, current_hp_fraction=0.3)
    staraptor = NS(fainted=False, current_hp_fraction=0.43)
    battle = NS(
        active_pokemon=[garchomp, kingambit], opponent_active_pokemon=[staraptor, None]
    )
    protect_sucker = G.Candidate((1, 2), 0.224)
    dragon_sucker = G.Candidate((3, 2), 0.195)

    def decode(_battle, action, pos):
        if pos == 0 and action == 1:
            return _order("protect")
        if pos == 0 and action == 3:
            return _order("dragonclaw", 1)
        if pos == 1 and action == 2:
            return _order("suckerpunch", 1)
        return NS(order=None, move_target=0)

    monkeypatch.setattr(G, "_decode", decode)
    monkeypatch.setattr(
        G.K,
        "damage_fraction",
        lambda _battle, _attacker, _defender, move: {
            "dragonclaw": (0.16, 0.18),
            "suckerpunch": (0.12, 0.14),
        }.get(move.id),
    )
    report = G.GuardReport()

    ranked = G.guard_two_on_one_focus(battle, [protect_sucker, dragon_sucker], report)

    assert ranked[0] is dragon_sucker
    assert report.stages == ["two_on_one_focus"]


def test_hidden_blastoise_prior_recognizes_shell_smash():
    blastoise = Pokemon(gen=9, species="blastoise")

    probability = G._hidden_move_probability(blastoise, G.CATASTROPHIC_SETUP_MOVES)

    assert probability > 0.85


def test_double_protect_is_demoted_into_likely_shell_smash(monkeypatch):
    basculegion = Pokemon(gen=9, species="basculegion")
    floette = Pokemon(gen=9, species="floetteeternal")
    sneasler = Pokemon(gen=9, species="sneasler")
    blastoise = Pokemon(gen=9, species="blastoise")
    battle = NS(
        active_pokemon=[basculegion, floette],
        opponent_active_pokemon=[sneasler, blastoise],
        fields={},
        side_conditions={},
        opponent_side_conditions={},
    )
    double_protect = G.Candidate((1, 2), 0.79)
    contest_setup = G.Candidate((3, 4), 0.21)

    def decode(_battle, action, pos):
        if action in {1, 2}:
            return _order("protect")
        if pos == 0:
            return _order("wavecrash", 1)
        return _order("dazzlinggleam")

    monkeypatch.setattr(G, "_decode", decode)
    monkeypatch.setattr(
        G.K,
        "damage_fraction",
        lambda _battle, _attacker, defender, move: (
            (0.20, 0.24)
            if defender is blastoise and move.id == "dazzlinggleam"
            else (0.40, 0.48)
        ),
    )
    report = G.GuardReport()

    ranked = G.guard_free_catastrophic_setup(
        battle, [double_protect, contest_setup], report
    )

    assert ranked[0] is contest_setup
    assert report.stages == ["free_catastrophic_setup"]


def test_double_protect_can_stall_asymmetric_tailwind(monkeypatch):
    basculegion = Pokemon(gen=9, species="basculegion")
    floette = Pokemon(gen=9, species="floetteeternal")
    sneasler = Pokemon(gen=9, species="sneasler")
    blastoise = Pokemon(gen=9, species="blastoise")
    battle = NS(
        active_pokemon=[basculegion, floette],
        opponent_active_pokemon=[sneasler, blastoise],
        fields={},
        side_conditions={},
        opponent_side_conditions={SideCondition.TAILWIND: 1},
    )
    double_protect = G.Candidate((1, 2), 0.79)
    contest_setup = G.Candidate((3, 4), 0.21)

    monkeypatch.setattr(
        G,
        "_decode",
        lambda _battle, action, _pos: (
            _order("protect") if action in {1, 2} else _order("dazzlinggleam")
        ),
    )
    monkeypatch.setattr(G.K, "damage_fraction", lambda *_args: (0.20, 0.24))
    report = G.GuardReport()

    ranked = G.guard_free_catastrophic_setup(
        battle, [double_protect, contest_setup], report
    )

    assert ranked[0] is double_protect
    assert report.stages == []


def test_known_prankster_encore_demotes_first_protect(monkeypatch):
    charizard = Pokemon(gen=9, species="charizard")
    garchomp = Pokemon(gen=9, species="garchomp")
    whimsicott = Pokemon(gen=9, species="whimsicott")
    incineroar = Pokemon(gen=9, species="incineroar")
    whimsicott.ability = "prankster"
    whimsicott.moves["encore"] = Move("encore", gen=9)
    battle = NS(
        active_pokemon=[charizard, garchomp],
        opponent_active_pokemon=[whimsicott, incineroar],
        fields={},
    )
    protect = G.Candidate((1, 3), 0.70)
    attack = G.Candidate((2, 3), 0.25)

    def decode(_battle, action, pos):
        if pos == 0 and action == 1:
            return _order("protect")
        if pos == 0 and action == 2:
            return _order("heatwave")
        if pos == 1 and action == 3:
            return _order("dragonclaw", 2)
        return NS(order=None, move_target=0)

    monkeypatch.setattr(G, "_decode", decode)
    monkeypatch.setattr(G.K, "guaranteed_ko", lambda *_args: False)
    report = G.GuardReport()

    ranked = G.guard_encore_exposure(battle, [protect, attack], report)

    assert ranked[0] is attack
    assert report.stages == ["encore_exposure"]


def test_protect_is_kept_when_partner_guarantees_encore_user_ko(monkeypatch):
    charizard = Pokemon(gen=9, species="charizard")
    floette = Pokemon(gen=9, species="floetteeternal")
    whimsicott = Pokemon(gen=9, species="whimsicott")
    incineroar = Pokemon(gen=9, species="incineroar")
    whimsicott.ability = "prankster"
    whimsicott.moves["encore"] = Move("encore", gen=9)
    battle = NS(
        active_pokemon=[charizard, floette],
        opponent_active_pokemon=[whimsicott, incineroar],
        fields={},
    )
    protect_ko = G.Candidate((1, 3), 0.70)
    attack_ko = G.Candidate((2, 3), 0.25)

    def decode(_battle, action, pos):
        if pos == 0 and action == 1:
            return _order("protect")
        if pos == 0 and action == 2:
            return _order("heatwave")
        if pos == 1 and action == 3:
            return _order("moonblast", 1)
        return NS(order=None, move_target=0)

    monkeypatch.setattr(G, "_decode", decode)
    monkeypatch.setattr(
        G.K,
        "guaranteed_ko",
        lambda _battle, attacker, defender, move: (
            attacker is floette and defender is whimsicott and move.id == "moonblast"
        ),
    )
    report = G.GuardReport()

    ranked = G.guard_encore_exposure(battle, [protect_ko, attack_ko], report)

    assert ranked[0] is protect_ko
    assert report.stages == []


def test_prankster_encore_does_not_threaten_a_dark_type(monkeypatch):
    kingambit = Pokemon(gen=9, species="kingambit")
    garchomp = Pokemon(gen=9, species="garchomp")
    whimsicott = Pokemon(gen=9, species="whimsicott")
    whimsicott.ability = "prankster"
    whimsicott.moves["encore"] = Move("encore", gen=9)
    battle = NS(
        active_pokemon=[kingambit, garchomp],
        opponent_active_pokemon=[whimsicott, None],
        fields={},
    )
    protect = G.Candidate((1, 2), 0.70)
    attack = G.Candidate((3, 2), 0.25)

    monkeypatch.setattr(
        G,
        "_decode",
        lambda _battle, action, _pos: (
            _order("protect") if action == 1 else _order("kowtowcleave", 1)
        ),
    )
    report = G.GuardReport()

    ranked = G.guard_encore_exposure(battle, [protect, attack], report)

    assert ranked[0] is protect
    assert report.stages == []


def test_no_weather_weather_ball_is_demoted_when_heat_wave_dominates(monkeypatch):
    charizard = Pokemon(gen=9, species="charizard")
    garchomp = Pokemon(gen=9, species="garchomp")
    whimsicott = Pokemon(gen=9, species="whimsicott")
    opposing_garchomp = Pokemon(gen=9, species="garchomp")
    charizard.moves["weatherball"] = Move("weatherball", gen=9)
    charizard.moves["heatwave"] = Move("heatwave", gen=9)
    battle = NS(
        active_pokemon=[charizard, garchomp],
        opponent_active_pokemon=[opposing_garchomp, whimsicott],
        weather={},
    )
    weather_ball = G.Candidate((1, 3), 0.001)
    heat_wave = G.Candidate((2, 4), 0.20)

    def decode(_battle, action, pos):
        if pos == 0 and action == 1:
            return _order("weatherball", 2)
        if pos == 0 and action == 2:
            return _order("heatwave")
        return _order("dragonclaw", 1)

    monkeypatch.setattr(G, "_decode", decode)
    monkeypatch.setattr(
        G.K,
        "damage_fraction",
        lambda _battle, _attacker, _defender, move: (
            (0.22, 0.24) if move.id == "weatherball" else (0.45, 0.49)
        ),
    )
    report = G.GuardReport()

    ranked = G.guard_dominated_weather_ball(battle, [weather_ball, heat_wave], report)

    assert ranked[0] is heat_wave
    assert report.stages == ["dominated_weather_ball"]


def test_no_weather_weather_ball_is_kept_when_heat_wave_is_not_better(monkeypatch):
    charizard = Pokemon(gen=9, species="charizard")
    garchomp = Pokemon(gen=9, species="garchomp")
    tyranitar = Pokemon(gen=9, species="tyranitar")
    charizard.moves["weatherball"] = Move("weatherball", gen=9)
    charizard.moves["heatwave"] = Move("heatwave", gen=9)
    battle = NS(
        active_pokemon=[charizard, garchomp],
        opponent_active_pokemon=[tyranitar, None],
        weather={},
    )
    weather_ball = G.Candidate((1, 3), 0.60)
    heat_wave = G.Candidate((2, 3), 0.30)

    def decode(_battle, action, pos):
        if pos == 0 and action == 1:
            return _order("weatherball", 1)
        if pos == 0 and action == 2:
            return _order("heatwave")
        return _order("dragonclaw", 1)

    monkeypatch.setattr(G, "_decode", decode)
    monkeypatch.setattr(
        G.K,
        "damage_fraction",
        lambda _battle, _attacker, _defender, move: (
            (0.30, 0.34) if move.id == "weatherball" else (0.25, 0.29)
        ),
    )
    report = G.GuardReport()

    ranked = G.guard_dominated_weather_ball(battle, [weather_ball, heat_wave], report)

    assert ranked[0] is weather_ball
    assert report.stages == []


def test_sun_weather_ball_beats_heat_wave_with_one_delphox(monkeypatch):
    garchomp = Pokemon(gen=9, species="garchomp")
    charizard = Pokemon(gen=9, species="charizard")
    delphox = Pokemon(gen=9, species="delphoxmega")
    charizard.moves["weatherball"] = Move("weatherball", gen=9)
    charizard.moves["heatwave"] = Move("heatwave", gen=9)
    battle = NS(
        active_pokemon=[garchomp, charizard],
        opponent_active_pokemon=[delphox, None],
        weather={Weather.SUNNYDAY: 3},
    )
    rock_heat = G.Candidate((20, 9), 0.103)
    rock_weather = G.Candidate((20, 15), 0.005)

    def decode(_battle, action, pos):
        if pos == 0:
            return _order("rocktomb", 1)
        if action == 9:
            return _order("heatwave")
        return _order("weatherball", 1)

    monkeypatch.setattr(G, "_decode", decode)
    monkeypatch.setattr(
        G.K,
        "damage_fraction",
        lambda _battle, _attacker, _defender, move: (
            (0.22, 0.24) if move.id == "heatwave" else (0.28, 0.31)
        ),
    )
    report = G.GuardReport()

    ranked = G.guard_single_target_weather_ball(
        battle, [rock_heat, rock_weather], report
    )

    assert ranked[0] is rock_weather
    assert rock_heat.demoted_by == "single_target_weather_ball"
    assert report.stages == ["single_target_weather_ball"]


def test_weather_ball_fallback_uses_its_dynamic_sun_profile(monkeypatch):
    charizard = Pokemon(gen=9, species="charizard")
    delphox = Pokemon(gen=9, species="delphoxmega")
    battle = NS(
        opponent_active_pokemon=[delphox, None],
        weather={Weather.SUNNYDAY: 3},
    )
    monkeypatch.setattr(G.K, "damage_fraction", lambda *_args: None)

    heat = G._expected_damage_value(
        battle, charizard, delphox, Move("heatwave", gen=9)
    )
    weather = G._expected_damage_value(
        battle, charizard, delphox, Move("weatherball", gen=9)
    )

    assert heat is not None and weather is not None
    assert weather >= heat * 1.10


def test_delphox_two_on_one_near_tie_attacks_with_both_slots(monkeypatch):
    garchomp = NS(fainted=False, current_hp_fraction=0.88)
    charizard = NS(fainted=False, current_hp_fraction=0.16)
    delphox = NS(fainted=False, current_hp_fraction=0.21)
    battle = NS(
        active_pokemon=[garchomp, charizard],
        opponent_active_pokemon=[delphox, None],
    )
    protect_solar = G.Candidate((24, 20), 0.000020)
    rock_weather = G.Candidate((20, 15), 0.000295)

    def decode(_battle, action, pos):
        if pos == 0 and action == 24:
            return _order("protect")
        if pos == 0:
            return _order("rocktomb", 1)
        if action == 20:
            return _order("solarbeam", 1)
        return _order("weatherball", 1)

    monkeypatch.setattr(G, "_decode", decode)
    monkeypatch.setattr(
        G.K,
        "damage_fraction",
        lambda _battle, _attacker, _defender, move: {
            "solarbeam": (0.08, 0.10),
            "rocktomb": (0.13, 0.15),
            "weatherball": (0.09, 0.11),
        }.get(move.id),
    )
    report = G.GuardReport()

    ranked = G.guard_two_on_one_focus(
        battle, [protect_solar, rock_weather], report
    )

    assert ranked[0] is rock_weather
    assert report.stages == ["two_on_one_focus"]


def test_yawn_switch_values_the_next_turn(monkeypatch):
    whimsicott = Pokemon(gen=9, species="whimsicott")
    floette = Pokemon(gen=9, species="floetteeternal")
    floette._max_hp = 100
    floette._current_hp = 100
    floette._effects[Effect.YAWN] = 0
    charizard = Pokemon(gen=9, species="charizard")
    foes = [Pokemon(gen=9, species="vaporeon"), Pokemon(gen=9, species="meganium")]
    bench = [
        Pokemon(gen=9, species="lycanrocdusk"),
        Pokemon(gen=9, species="ninetalesalola"),
    ]
    battle = NS(
        active_pokemon=[whimsicott, floette],
        opponent_active_pokemon=foes,
        opponent_team={str(i): mon for i, mon in enumerate([*foes, *bench])},
    )
    stay = G.Candidate((1, 2), 0.87)
    switch = G.Candidate((1, 3), 0.01)

    def decode(_battle, action, pos):
        if pos == 0:
            return _order("moonblast", 1)
        if action == 2:
            return _order("dazzlinggleam")
        if action == 3:
            return NS(order=charizard, move_target=0)
        return NS(order=None, move_target=0)

    monkeypatch.setattr(G, "_decode", decode)
    monkeypatch.setattr(G.K, "damage_fraction", lambda *_args: None)
    report = G.GuardReport()

    ranked = G.guard_yawn_switch(battle, [stay, switch], report)

    assert ranked[0] is switch
    assert report.stages == ["yawn_switch"]


def test_minus_two_physical_attacker_gets_a_switch_candidate(monkeypatch):
    garchomp = Pokemon(gen=9, species="garchomp")
    garchomp._max_hp = 100
    garchomp._current_hp = 100
    garchomp._boosts["atk"] = -2
    charizard = Pokemon(gen=9, species="charizard")
    basculegion = Pokemon(gen=9, species="basculegion")
    incineroar = Pokemon(gen=9, species="incineroar")
    staraptor = Pokemon(gen=9, species="staraptor")
    for foe in (incineroar, staraptor):
        foe._max_hp = 100
        foe._current_hp = 100
    battle = NS(
        active_pokemon=[garchomp, charizard],
        opponent_active_pokemon=[staraptor, incineroar],
    )
    stay = G.Candidate((1, 2), 0.47)
    switch = G.Candidate((3, 2), 0.01)

    def decode(_battle, action, pos):
        if pos == 0 and action == 1:
            return _order("earthquake")
        if pos == 0 and action == 3:
            return NS(order=basculegion, move_target=0)
        if pos == 1 and action == 2:
            return _order("heatwave")
        return NS(order=None, move_target=0)

    monkeypatch.setattr(G, "_decode", decode)
    monkeypatch.setattr(G.K, "damage_fraction", lambda *_args: (0.10, 0.15))
    monkeypatch.setattr(G.K, "deals_no_damage", lambda *_args: True)
    report = G.GuardReport()

    ranked = G.guard_severe_attack_drop_switch(battle, [stay, switch], report)

    assert ranked[0] is switch
    assert report.stages == ["severe_attack_drop_switch"]


def test_rain_reserves_mega_charizard_y(monkeypatch):
    garchomp = Pokemon(gen=9, species="garchomp")
    floette = Pokemon(gen=9, species="floetteeternal")
    charizard = Pokemon(gen=9, species="charizard")
    charizard.item = "charizarditey"
    charizard._selected_in_teampreview = True
    swampert = Pokemon(gen=9, species="swampert")
    pelipper = Pokemon(gen=9, species="pelipper")
    battle = NS(
        active_pokemon=[garchomp, floette],
        opponent_active_pokemon=[pelipper, swampert],
        team={"g": garchomp, "f": floette, "z": charizard},
        weather={Weather.RAINDANCE: 1},
    )
    mega_floette = G.Candidate((1, 2), 0.60)
    save_mega = G.Candidate((1, 3), 0.02)

    def decode(_battle, action, pos):
        if pos == 0:
            return _order("dragonclaw", 2)
        if action == 2:
            return NS(order=Move("dazzlinggleam", gen=9), move_target=0, mega=True)
        if action == 3:
            return NS(order=Move("dazzlinggleam", gen=9), move_target=0, mega=False)
        return NS(order=None, move_target=0, mega=False)

    monkeypatch.setattr(G, "_decode", decode)
    monkeypatch.setattr(G.K, "damage_fraction", lambda *_args: None)
    report = G.GuardReport()

    ranked = G.guard_reserve_weather_mega(battle, [mega_floette, save_mega], report)

    assert ranked[0] is save_mega
    assert report.stages == ["reserve_weather_mega"]


def test_visible_rain_core_reserves_mega_before_pelipper_switches(monkeypatch):
    garchomp = Pokemon(gen=9, species="garchomp")
    floette = Pokemon(gen=9, species="floetteeternal")
    charizard = Pokemon(gen=9, species="charizard")
    charizard.item = "charizarditey"
    charizard._selected_in_teampreview = True
    swampert = Pokemon(gen=9, species="swampert")
    sneasler = Pokemon(gen=9, species="sneasler")
    pelipper = Pokemon(gen=9, species="pelipper")
    battle = NS(
        active_pokemon=[garchomp, floette],
        opponent_active_pokemon=[sneasler, swampert],
        opponent_team={"s": sneasler, "w": swampert, "p": pelipper},
        team={"g": garchomp, "f": floette, "z": charizard},
        weather={},
    )
    mega_floette = G.Candidate((1, 2), 0.60)
    save_mega = G.Candidate((1, 3), 0.02, strategic_only=True)

    def decode(_battle, action, pos):
        if pos == 0:
            return _order("dragonclaw", 2)
        if action == 2:
            return NS(order=Move("dazzlinggleam", gen=9), move_target=0, mega=True)
        if action == 3:
            return NS(order=Move("dazzlinggleam", gen=9), move_target=0, mega=False)
        return NS(order=None, move_target=0, mega=False)

    monkeypatch.setattr(G, "_decode", decode)
    monkeypatch.setattr(G.K, "damage_fraction", lambda *_args: None)
    report = G.GuardReport()

    ranked = G.guard_reserve_weather_mega(battle, [mega_floette, save_mega], report)

    assert ranked[0] is save_mega
    assert report.stages == ["reserve_weather_mega"]


def test_rain_reservation_survives_a_changed_floette_move(monkeypatch):
    garchomp = Pokemon(gen=9, species="garchomp")
    floette = Pokemon(gen=9, species="floetteeternal")
    charizard = Pokemon(gen=9, species="charizard")
    charizard.item = "charizarditey"
    charizard._selected_in_teampreview = True
    archaludon = Pokemon(gen=9, species="archaludon")
    pelipper = Pokemon(gen=9, species="pelipper")
    battle = NS(
        active_pokemon=[garchomp, floette],
        opponent_active_pokemon=[pelipper, archaludon],
        opponent_team={"p": pelipper, "a": archaludon},
        team={"g": garchomp, "f": floette, "z": charizard},
        weather={Weather.RAINDANCE: 1},
    )
    mega_moonblast = G.Candidate((1, 2), 0.60)
    ordinary_dazzling_gleam = G.Candidate((1, 3), 0.02, strategic_only=True)

    def decode(_battle, action, pos):
        if pos == 0:
            return _order("dragonclaw", 2)
        if action == 2:
            return NS(order=Move("moonblast", gen=9), move_target=2, mega=True)
        if action == 3:
            return NS(order=Move("dazzlinggleam", gen=9), move_target=0, mega=False)
        return NS(order=None, move_target=0, mega=False)

    monkeypatch.setattr(G, "_decode", decode)
    monkeypatch.setattr(G.K, "damage_fraction", lambda *_args: None)
    report = G.GuardReport()

    ranked = G.guard_reserve_weather_mega(
        battle, [mega_moonblast, ordinary_dazzling_gleam], report
    )

    assert ranked[0] is ordinary_dazzling_gleam
    assert report.stages == ["reserve_weather_mega"]


def test_rain_reservation_does_not_replace_an_attack_with_double_protect(monkeypatch):
    basculegion = Pokemon(gen=9, species="basculegion")
    floette = Pokemon(gen=9, species="floetteeternal")
    charizard = Pokemon(gen=9, species="charizard")
    charizard.item = "charizarditey"
    charizard._selected_in_teampreview = True
    pelipper = Pokemon(gen=9, species="pelipper")
    opposing_basculegion = Pokemon(gen=9, species="basculegion")
    battle = NS(
        active_pokemon=[basculegion, floette],
        opponent_active_pokemon=[pelipper, opposing_basculegion],
        opponent_team={"p": pelipper, "b": opposing_basculegion},
        team={"b": basculegion, "f": floette, "z": charizard},
        weather={Weather.RAINDANCE: 1},
    )
    attack = G.Candidate((1, 2), 0.36)
    double_protect = G.Candidate((1, 3), 0.02)

    def decode(_battle, action, pos):
        if pos == 0:
            return _order("protect")
        if action == 2:
            return NS(order=Move("dazzlinggleam", gen=9), move_target=0, mega=True)
        return NS(order=Move("protect", gen=9), move_target=0, mega=False)

    monkeypatch.setattr(G, "_decode", decode)
    monkeypatch.setattr(G.K, "damage_fraction", lambda *_args: (0.20, 0.25))
    report = G.GuardReport()

    ranked = G.guard_reserve_weather_mega(
        battle, [attack, double_protect], report
    )

    assert ranked[0] is attack
    assert report.stages == []


def test_visible_pelipper_archaludon_core_reserves_charizard_y(monkeypatch):
    garchomp = Pokemon(gen=9, species="garchomp")
    floette = Pokemon(gen=9, species="floetteeternal")
    charizard = Pokemon(gen=9, species="charizard")
    charizard.item = "charizarditey"
    charizard._selected_in_teampreview = True
    archaludon = Pokemon(gen=9, species="archaludon")
    venusaur = Pokemon(gen=9, species="venusaur")
    pelipper = Pokemon(gen=9, species="pelipper")
    battle = NS(
        active_pokemon=[garchomp, floette],
        opponent_active_pokemon=[venusaur, archaludon],
        opponent_team={"v": venusaur, "a": archaludon, "p": pelipper},
        team={"g": garchomp, "f": floette, "z": charizard},
        weather={},
    )
    mega_floette = G.Candidate((1, 2), 0.60)
    save_mega = G.Candidate((1, 3), 0.02, strategic_only=True)

    def decode(_battle, action, pos):
        if pos == 0:
            return _order("dragonclaw", 2)
        if action == 2:
            return NS(order=Move("moonblast", gen=9), move_target=2, mega=True)
        if action == 3:
            return NS(order=Move("dazzlinggleam", gen=9), move_target=0, mega=False)
        return NS(order=None, move_target=0, mega=False)

    monkeypatch.setattr(G, "_decode", decode)
    monkeypatch.setattr(G.K, "damage_fraction", lambda *_args: None)
    report = G.GuardReport()

    ranked = G.guard_reserve_weather_mega(battle, [mega_floette, save_mega], report)

    assert ranked[0] is save_mega
    assert report.stages == ["reserve_weather_mega"]


def _final_two_on_two_battle():
    ours = [
        Pokemon(gen=9, species="floetteeternal"),
        Pokemon(gen=9, species="garchomp"),
    ]
    foes = [Pokemon(gen=9, species="tyranitar"), Pokemon(gen=9, species="excadrill")]
    for mon in (*ours, *foes):
        mon._max_hp = 100
        mon._current_hp = 100
        mon._revealed = True
    return NS(
        active_pokemon=ours,
        opponent_active_pokemon=foes,
        team={str(i): mon for i, mon in enumerate(ours)},
        opponent_team={str(i): mon for i, mon in enumerate(foes)},
        max_team_size=2,
        fields={},
        side_conditions={},
        opponent_side_conditions={},
    )


def test_final_two_on_two_double_protect_makes_progress(monkeypatch):
    battle = _final_two_on_two_battle()
    double_protect = G.Candidate((1, 2), 0.60)
    one_attack = G.Candidate((1, 3), 0.156)
    both_attack = G.Candidate((4, 3), 0.0023)

    def decode(_battle, action, pos):
        if (pos, action) in {(0, 1), (1, 2)}:
            return _order("protect")
        if pos == 0 and action == 4:
            return _order("dazzlinggleam")
        if pos == 1 and action == 3:
            return _order("rocktomb", 1)
        return NS(order=None, move_target=0)

    monkeypatch.setattr(G, "_decode", decode)
    monkeypatch.setattr(G.K, "damage_fraction", lambda *_args: (0.20, 0.25))
    report = G.GuardReport()

    ranked = G.guard_endgame_progress(
        battle, [double_protect, one_attack, both_attack], report
    )

    assert ranked[0] is both_attack
    assert report.stages == ["endgame_progress"]


def test_final_two_on_two_repeated_protect_attacks(monkeypatch):
    battle = _final_two_on_two_battle()
    battle.active_pokemon[0]._protect_counter = 1
    repeat_protect = G.Candidate((1, 2), 0.456)
    both_attack = G.Candidate((3, 2), 0.125)

    def decode(_battle, action, pos):
        if pos == 0 and action == 1:
            return _order("protect")
        if pos == 0 and action == 3:
            return _order("heatwave")
        if pos == 1 and action == 2:
            return _order("rocktomb", 1)
        return NS(order=None, move_target=0)

    monkeypatch.setattr(G, "_decode", decode)
    monkeypatch.setattr(G.K, "damage_fraction", lambda *_args: (0.20, 0.25))
    report = G.GuardReport()

    ranked = G.guard_endgame_progress(battle, [repeat_protect, both_attack], report)

    assert ranked[0] is both_attack
    assert report.stages == ["endgame_progress"]


def test_last_pokemon_does_not_repeat_protect_without_a_stall_goal(monkeypatch):
    charizard = Pokemon(gen=9, species="charizard")
    charizard._max_hp = 100
    charizard._current_hp = 100
    charizard._protect_counter = 1
    blastoise = Pokemon(gen=9, species="blastoise")
    sneasler = Pokemon(gen=9, species="sneasler")
    for foe in (blastoise, sneasler):
        foe._max_hp = 100
        foe._current_hp = 100
    battle = NS(
        active_pokemon=[charizard, None],
        opponent_active_pokemon=[blastoise, sneasler],
        team={"charizard": charizard},
        opponent_team={"blastoise": blastoise, "sneasler": sneasler},
        fields={},
        side_conditions={},
        opponent_side_conditions={},
    )
    repeat_protect = G.Candidate((1, 0), 0.14)
    heat_wave = G.Candidate((2, 0), 0.86)

    monkeypatch.setattr(
        G,
        "_decode",
        lambda _battle, action, _pos: (
            _order("protect") if action == 1 else _order("heatwave")
        ),
    )
    monkeypatch.setattr(G.K, "damage_fraction", lambda *_args: (0.20, 0.25))
    report = G.GuardReport()

    ranked = G.guard_endgame_progress(battle, [repeat_protect, heat_wave], report)

    assert ranked[0] is heat_wave
    assert report.stages == ["endgame_progress"]


def test_endgame_progress_stands_down_for_asymmetric_tailwind(monkeypatch):
    battle = _final_two_on_two_battle()
    battle.opponent_side_conditions = {SideCondition.TAILWIND: 1}
    double_protect = G.Candidate((1, 2), 0.60)
    both_attack = G.Candidate((3, 4), 0.20)
    monkeypatch.setattr(G, "_decode", lambda *_args: _order("protect"))
    report = G.GuardReport()

    ranked = G.guard_endgame_progress(battle, [double_protect, both_attack], report)

    assert ranked[0] is double_protect
    assert report.stages == []


def test_hidden_move_prior_prevents_an_unprofitable_predicted_ko(monkeypatch):
    monkeypatch.setenv("VGC_PREDICTED_SURVIVAL", "1")
    basculegion = Pokemon(gen=9, species="basculegion")
    whimsicott = Pokemon(gen=9, species="whimsicott")
    venusaur = Pokemon(gen=9, species="venusaur")
    archaludon = Pokemon(gen=9, species="archaludon")
    for mon in (basculegion, whimsicott, venusaur, archaludon):
        mon._max_hp = 100
        mon._current_hp = 100
    battle = NS(
        gen=9,
        active_pokemon=[basculegion, whimsicott],
        opponent_active_pokemon=[venusaur, archaludon],
    )
    attack = G.Candidate(
        (1, 2), 0.90, orders=(_order("wavecrash", 1), _order("tailwind"))
    )
    protect = G.Candidate(
        (3, 2),
        0.01,
        orders=(_order("protect"), _order("tailwind")),
        strategic_only=True,
    )
    predictions = (
        MovePrediction(
            moves=(("gigadrain", 0.70), ("protect", 0.30)), targets=(), reliability=0.0
        ),
        MovePrediction(moves=(("protect", 1.0),), targets=(), reliability=0.0),
    )

    def damage(_battle, attacker, target, move):
        if attacker is venusaur and target is basculegion and move.id == "gigadrain":
            return 1.05, 1.20
        if attacker is venusaur and target is whimsicott and move.id == "gigadrain":
            return 0.10, 0.12
        if attacker is basculegion and move.id == "wavecrash":
            return 0.30, 0.35
        return None

    monkeypatch.setattr(G.K, "damage_fraction", damage)
    monkeypatch.setattr(
        G, "_candidate_guaranteed_progress", lambda *_args: (0, 0.0, True)
    )

    ranked, report = rerank_candidates(
        battle,
        [attack, protect],
        predictions,
        opponent_switches=None,
        use_opponent=True,
        use_tempo=False,
    )

    assert ranked[0] is protect
    assert report is not None
    assert report.special_reason == "predicted_ko_survival"


def test_predicted_ko_survival_allows_a_profitable_trade(monkeypatch):
    monkeypatch.setenv("VGC_PREDICTED_SURVIVAL", "1")
    basculegion = Pokemon(gen=9, species="basculegion")
    whimsicott = Pokemon(gen=9, species="whimsicott")
    venusaur = Pokemon(gen=9, species="venusaur")
    archaludon = Pokemon(gen=9, species="archaludon")
    for mon in (basculegion, whimsicott, venusaur, archaludon):
        mon._max_hp = 100
        mon._current_hp = 100
    battle = NS(
        gen=9,
        active_pokemon=[basculegion, whimsicott],
        opponent_active_pokemon=[venusaur, archaludon],
    )
    attack = G.Candidate(
        (1, 2), 0.90, orders=(_order("wavecrash", 1), _order("tailwind"))
    )
    protect = G.Candidate((3, 2), 0.01, orders=(_order("protect"), _order("tailwind")))
    predictions = (
        MovePrediction(
            moves=(("gigadrain", 0.70), ("protect", 0.30)), targets=(), reliability=0.0
        ),
        MovePrediction(moves=(("protect", 1.0),), targets=(), reliability=0.0),
    )

    monkeypatch.setattr(
        G.K,
        "damage_fraction",
        lambda _battle, attacker, target, move: (
            (1.05, 1.20)
            if attacker is venusaur and target is basculegion and move.id == "gigadrain"
            else ((0.30, 0.35) if attacker is basculegion else None)
        ),
    )
    monkeypatch.setattr(
        G,
        "_candidate_guaranteed_progress",
        lambda _battle, candidate: (1, 1.0, True)
        if candidate is attack
        else (0, 0.0, True),
    )

    ranked, report = rerank_candidates(
        battle,
        [attack, protect],
        predictions,
        opponent_switches=None,
        use_opponent=True,
        use_tempo=False,
    )

    assert ranked[0] is attack
    assert report is None
