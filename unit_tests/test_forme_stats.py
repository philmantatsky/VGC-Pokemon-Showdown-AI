"""forme_stats (2026-10-04): an opposing Pokemon gets its stat estimate the first
time a calculation needs one and never again, so a foe that Mega-evolves keeps its
pre-Mega numbers for the rest of the game.

The first tests reproduce that in the default path, which is unchanged. The rest
cover the opt-in repair: with the profile entry ``forme_stats`` on, the guard stack
runs on the estimate for the forme each foe is in, and the stored line -- what the
observation reads, and what every brain was trained on -- is back afterwards.
"""

from __future__ import annotations

import json

import pytest
from poke_env.battle import DoubleBattle, Move, Pokemon

from unit_tests.ladder_position import move_action, position
from vgc_bench.src import forme_stats as F
from vgc_bench.src import guards as G
from vgc_bench.src import vgc_knowledge as K
from vgc_bench.src.policy_player import PolicyPlayer

T6E = "teams/candidates_mc/T6e.txt"
OURS = [
    "|switch|p1a: Blastoise|Blastoise, L50, M|170/186",
    "|switch|p1b: Farigiraf|Farigiraf, L50, F|227/227",
]
MEGAS = [
    ("Blastoise, L50, M", "Blastoise-Mega, L50, M", "Blastoisinite"),
    ("Charizard, L50, M", "Charizard-Mega-Y, L50, M", "Charizardite Y"),
    ("Charizard, L50, M", "Charizard-Mega-X, L50, M", "Charizardite X"),
    # its Mega's base species entry (Floette) is not the forme it evolves from
    ("Floette-Eternal, L50, F", "Floette-Mega, L50, F", "Floettite"),
    ("Gengar, L50, M", "Gengar-Mega, L50, M", "Gengarite"),
]


def _foe(details: str) -> Pokemon:
    """An opposing Pokemon as a hidden-sheet game creates it: no stats."""
    return Pokemon(gen=9, details=details)


def _evolved(details: str, forme: str, stone: str) -> Pokemon:
    """Estimated in its first forme, then Mega-evolved, as on ladder."""
    mon = _foe(details)
    assert K.ensure_stats(mon) is True
    mon.forme_change(forme)  # |detailschange|
    mon.mega_evolve(stone)  # |-mega|
    return mon


def _first_seen_as(forme: str) -> Pokemon:
    mon = _foe(forme)
    K.ensure_stats(mon)
    return mon


def _live(before: list[str], after: list[str]) -> DoubleBattle:
    """The position as the live bot holds it: its observation ran the damage
    calculator against every active foe at the decision before ``after`` happened."""
    battle = position(before, T6E)
    for foe in battle.opponent_active_pokemon:
        if foe is not None:
            K.ensure_stats(foe)
    for line in after:
        battle.parse_message(line.split("|"))
    return battle


def _only(*names: str) -> dict[str, bool]:
    return {name: name in names for name in G.GUARDS}


def _raichu() -> DoubleBattle:
    """Turn 2 under our Trick Room: the foe's Raichu Mega-evolved (Raichu-Mega-Y)
    on turn 1, has shown Thunderbolt and stands at 28%."""
    return _live(
        [
            *OURS,
            "|switch|p2a: Raichu|Raichu, L50, F|100/100",
            "|switch|p2b: Incineroar|Incineroar, L50, F|100/100",
            "|turn|1",
        ],
        [
            "|detailschange|p2a: Raichu|Raichu-Mega-Y, L50, F",
            "|-mega|p2a: Raichu|Raichu|Raichunite Y",
            "|move|p2a: Raichu|Thunderbolt|p1b: Farigiraf",
            "|-damage|p1b: Farigiraf|150/227",
            "|move|p1b: Farigiraf|Trick Room|p1b: Farigiraf",
            "|-fieldstart|move: Trick Room|[of] p1b: Farigiraf",
            "|move|p1a: Blastoise|Water Pulse|p2a: Raichu",
            "|-damage|p2a: Raichu|28/100",
            "|turn|2",
        ],
    )


def _raichu_pairs(battle: DoubleBattle) -> list[G.Candidate]:
    psychic = move_action(battle, 1, "psychic", 2)
    return [
        G.Candidate((move_action(battle, 0, "waterpulse", 2), psychic), 0.5),
        G.Candidate((move_action(battle, 0, "waterpulse", 1), psychic), 0.2),
        G.Candidate((move_action(battle, 0, "icebeam", 2), psychic), 0.1),
    ]


def _gengar() -> DoubleBattle:
    """Turn 2: the foe's Gengar Mega-evolved on turn 1 and stands at 80%."""
    return _live(
        [
            *OURS,
            "|switch|p2a: Gengar|Gengar, L50, M|100/100",
            "|switch|p2b: Metagross|Metagross, L50|100/100",
            "|turn|1",
        ],
        [
            "|detailschange|p2a: Gengar|Gengar-Mega, L50, M",
            "|-mega|p2a: Gengar|Gengar|Gengarite",
            "|-damage|p2a: Gengar|80/100",
            "|turn|2",
        ],
    )


def _gengar_pairs(battle: DoubleBattle) -> list[G.Candidate]:
    water_pulse = move_action(battle, 0, "waterpulse", 2)
    return [
        # Psychic into the Metagross that resists it: the policy's favourite
        G.Candidate((water_pulse, move_action(battle, 1, "psychic", 2)), 0.5),
        G.Candidate((water_pulse, move_action(battle, 1, "psychic", 1)), 0.3),
    ]


# --- the defect, in the default path -------------------------------------------------


def test_ensure_stats_keeps_the_pre_mega_line():
    mon = _foe("Blastoise, L50, M")
    assert K.ensure_stats(mon) is True
    before = dict(mon.stats)
    assert before["spa"] == 137  # base 85 + 32 + 20
    mon.forme_change("Blastoise-Mega, L50, M")
    mon.mega_evolve("Blastoisinite")
    assert mon.base_stats["spa"] == 135  # poke-env follows the forme ...
    assert K.ensure_stats(mon) is False  # ... the estimate is "already usable"
    assert mon.stats == before
    assert _first_seen_as("Blastoise-Mega, L50, M").stats["spa"] == 187
    # and the from-absent detector no longer recognises the line as an estimate
    assert K.stats_were_synthesized(mon) is False


def test_the_same_position_gives_two_answers_by_when_the_foe_was_first_estimated():
    """Mega Raichu Y's Thunderbolt into our Blastoise: at most a two-hit knockout
    for the live bot, a likely one-hit knockout when the position is rebuilt in one
    go (the replay audits) and the Mega is the first forme the calculator sees."""
    live = _raichu()
    lines = [
        *OURS,
        "|switch|p2a: Raichu|Raichu, L50, F|100/100",
        "|switch|p2b: Incineroar|Incineroar, L50, F|100/100",
        "|detailschange|p2a: Raichu|Raichu-Mega-Y, L50, F",
        "|-mega|p2a: Raichu|Raichu|Raichunite Y",
        "|turn|2",
    ]
    rebuilt = position(lines, T6E)
    thunderbolt = Move("thunderbolt", gen=9)
    numbers = []
    for battle in (live, rebuilt):
        raichu, blastoise = battle.opponent_active_pokemon[0], battle.active_pokemon[0]
        assert raichu is not None and blastoise is not None
        numbers.append(K.damage_fraction(battle, raichu, blastoise, thunderbolt))
    assert numbers[0] is not None and numbers[1] is not None
    assert numbers[0][1] < 0.6 and numbers[1][0] > 0.9


# --- recognising a stale line --------------------------------------------------------


@pytest.mark.parametrize("details, forme, stone", MEGAS)
def test_a_stale_line_is_replaced_by_what_the_mega_would_get_first_seen(
    details, forme, stone
):
    mon = _evolved(details, forme, stone)
    carried, before = mon.stats, dict(mon.stats)
    fresh = F.stale_estimate(mon)
    assert fresh == _first_seen_as(forme).stats and fresh != before
    assert mon.stats is carried and mon.stats == before  # looking changes nothing


def test_a_mega_that_left_and_came_back_is_still_recognised():
    mon = _evolved(*MEGAS[0])
    mon.switch_out({})
    mon.switch_in("Blastoise-Mega, L50, M")  # its details now name the Mega
    assert mon.species == "blastoisemega"
    assert F.stale_estimate(mon) == _first_seen_as("Blastoise-Mega, L50, M").stats


def test_a_forme_that_changes_back_and_forth_is_followed_both_ways():
    mon = _foe("Aegislash, L50, M")
    K.ensure_stats(mon)
    assert F.stale_estimate(mon) is None
    mon.forme_change("Aegislash-Blade")
    blade = F.stale_estimate(mon)
    assert blade is not None and blade["atk"] == 192 and blade["def"] == 70
    mon.forme_change("Aegislash")
    assert F.stale_estimate(mon) is None  # the stored line is the shield's again


def test_lines_that_are_not_a_stale_estimate_are_left_alone():
    first_seen_in_this_forme = _first_seen_as("Rotom-Wash, L50")
    assert F.stale_estimate(first_seen_in_this_forme) is None
    no_stats = _foe("Blastoise, L50, M")
    no_stats.forme_change("Blastoise-Mega, L50, M")
    assert F.stale_estimate(no_stats) is None
    not_our_estimate = _foe("Blastoise, L50, M")
    K.ensure_stats(not_our_estimate)
    bulkier = dict(not_our_estimate.stats)
    bulkier["def"] = (bulkier["def"] or 0) + 30  # e.g. a real spread from a sheet
    not_our_estimate.stats = bulkier
    not_our_estimate.forme_change("Blastoise-Mega, L50, M")
    assert F.stale_estimate(not_our_estimate) is None
    transformed = _foe("Ditto, L50")
    K.ensure_stats(transformed)
    transformed.transform(_foe("Blastoise, L50, M"))
    assert F.stale_estimate(transformed) is None


def test_only_the_opponents_pokemon_are_looked_at():
    battle = _raichu()
    ours = battle.active_pokemon[0]
    assert ours is not None
    line = F.estimate(ours)
    assert line is not None
    ours.stats = line  # dress our Blastoise up as a stale estimate
    ours.forme_change("Blastoise-Mega, L50, M")
    assert F.stale_estimate(ours) is not None
    assert [mon.species for mon, _ in F.find(battle)] == ["raichu"]


# --- the temporary swap --------------------------------------------------------------


def test_the_stored_line_is_back_after_the_block_and_after_an_error():
    battle = _raichu()
    raichu = battle.opponent_active_pokemon[0]
    assert raichu is not None
    carried, before = raichu.stats, dict(raichu.stats)
    with F.current_forme(battle) as changed:
        assert changed == [raichu]
        assert raichu.stats == _first_seen_as("Raichu-Mega-Y, L50, F").stats
        with F.current_forme(battle) as again:  # nothing left to replace
            assert again == []
    assert raichu.stats is carried and raichu.stats == before
    with pytest.raises(RuntimeError):
        with F.current_forme(battle):
            raise RuntimeError("a calculation failed")
    assert raichu.stats is carried and raichu.stats == before


def test_damage_by_and_against_the_mega_changes_only_inside_the_block():
    battle = _gengar()
    gengar, farigiraf = battle.opponent_active_pokemon[0], battle.active_pokemon[1]
    assert gengar is not None and farigiraf is not None
    psychic, sludge_bomb = Move("psychic", gen=9), Move("sludgebomb", gen=9)
    assert K.guaranteed_ko(battle, farigiraf, gengar, psychic) is True
    old_hit = K.damage_fraction(battle, gengar, farigiraf, sludge_bomb)
    with F.current_forme(battle):
        assert K.guaranteed_ko(battle, farigiraf, gengar, psychic) is False
        new_hit = K.damage_fraction(battle, gengar, farigiraf, sludge_bomb)
    assert old_hit is not None and new_hit is not None and new_hit[0] > old_hit[1]
    assert K.guaranteed_ko(battle, farigiraf, gengar, psychic) is True


# --- the guard stack -----------------------------------------------------------------


def test_registered_as_an_opt_in_profile_entry():
    assert G.FORME_STATS in G.GUARDS and G.FORME_STATS not in G.HARD_GUARDS
    assert set(G.GUARD_ORDER) == set(G.GUARDS)
    cands = [G.Candidate((7, 7), 0.6), G.Candidate((8, 7), 0.4)]
    assert G.guard_forme_stats(_raichu(), cands, G.GuardReport()) is cands


def test_every_guard_does_not_mean_the_switch():
    battle = _raichu()
    _, report = G.apply_guards(battle, _raichu_pairs(battle))  # enabled=None
    assert G.FORME_STATS not in report.stages and report.forme_stats is None
    assert f"{G.FORME_STATS}:corrected" not in report.demotions


@pytest.mark.parametrize(
    "entry", [{G.FORME_STATS: False}, {}], ids=["off", "absent from the profile"]
)
def test_without_the_entry_the_stack_runs_on_the_stored_numbers(entry):
    battle = _raichu()
    profile = {n: n == "threat_first2" for n in G.GUARDS if n != G.FORME_STATS}
    cands = _raichu_pairs(battle)
    out, report = G.apply_guards(battle, cands, profile | entry)
    assert out[0] is cands[0] and not report.stages  # no threat seen: 48-58%
    assert report.forme_stats is None and not report.demotions


def test_threat_first2_answers_the_evolved_raichu():
    """Thunderbolt from Mega Raichu Y knocks our Blastoise out (94-111% of its HP;
    the stored line says 48-58%). Under our Trick Room Blastoise moves first and
    Water Pulse finishes the Raichu at 28%."""
    battle = _raichu()
    raichu = battle.opponent_active_pokemon[0]
    assert raichu is not None
    carried = raichu.stats
    cands = _raichu_pairs(battle)
    out, report = G.apply_guards(battle, cands, _only("threat_first2", G.FORME_STATS))
    assert out[0] is cands[1]  # the caller's own object, not the comparison's copy
    assert report.stages == ["threat_first2", G.FORME_STATS]
    assert report.demotions[f"{G.FORME_STATS}:corrected"] == 1
    assert report.forme_stats == {
        "corrected": ["raichu"],
        "old_pick": list(cands[0].actions),
    }
    assert raichu.stats is carried  # the observation's numbers are untouched


def test_a_knockout_the_mega_survives_is_no_longer_promoted():
    """Psychic into Gengar at 80%: a guaranteed knockout against the stored line
    (84-101%), not against Mega Gengar's (69-83%)."""
    battle = _gengar()
    plain, plain_report = G.apply_guards(
        battle, _gengar_pairs(battle), _only("guaranteed_ko")
    )
    cands = _gengar_pairs(battle)
    assert plain[0].actions == cands[1].actions
    assert plain_report.stages == ["guaranteed_ko"]
    out, report = G.apply_guards(battle, cands, _only("guaranteed_ko", G.FORME_STATS))
    assert out[0] is cands[0] and report.stages == [G.FORME_STATS]
    assert report.forme_stats == {
        "corrected": ["gengar"],
        "old_pick": list(cands[1].actions),
    }


def test_no_stale_line_nothing_to_do():
    battle = position(
        [
            *OURS,
            "|switch|p2a: Raichu|Raichu, L50, F|100/100",
            "|switch|p2b: Incineroar|Incineroar, L50, F|100/100",
            "|turn|1",
        ],
        T6E,
    )
    cands = _raichu_pairs(battle)
    out, report = G.apply_guards(battle, cands, _only("threat_first2", G.FORME_STATS))
    reference, reference_report = G.apply_guards(
        battle, _raichu_pairs(battle), _only("threat_first2")
    )
    assert [c.actions for c in out] == [c.actions for c in reference]
    assert report.stages == reference_report.stages
    assert report.forme_stats is None and G.FORME_STATS not in report.stages
    assert f"{G.FORME_STATS}:corrected" not in report.demotions


def test_a_correction_that_leaves_the_pick_alone_is_counted_but_not_noted():
    battle = _gengar()
    cands = _gengar_pairs(battle)
    out, report = G.apply_guards(battle, cands, _only("zero_damage", G.FORME_STATS))
    assert out[0] is cands[0] and G.FORME_STATS not in report.stages
    assert report.demotions[f"{G.FORME_STATS}:corrected"] == 1
    assert report.forme_stats == {"corrected": ["gengar"]}


def test_a_guard_that_raises_inside_the_block_still_leaves_the_stored_line(monkeypatch):
    battle = _gengar()
    gengar = battle.opponent_active_pokemon[0]
    assert gengar is not None
    carried = gengar.stats
    seen = []

    def broken(_battle, _cands, _report):
        seen.append(dict(gengar.stats))
        raise RuntimeError("boom")

    monkeypatch.setitem(G.GUARDS, "guaranteed_ko", broken)
    cands = _gengar_pairs(battle)
    out, report = G.apply_guards(battle, cands, _only("guaranteed_ko", G.FORME_STATS))
    assert out[0] is cands[0] and "guaranteed_ko_error" in report.stages
    # once on the stored numbers (the comparison), once on Mega Gengar's
    assert seen == [dict(carried), _first_seen_as("Gengar-Mega, L50, M").stats]
    assert gengar.stats is carried


def test_if_the_lookup_fails_the_plain_stack_still_runs(monkeypatch):
    battle = _gengar()

    def broken(_battle):
        raise KeyError("unknown species")

    monkeypatch.setattr(F, "find", broken)
    cands = _gengar_pairs(battle)
    out, report = G.apply_guards(battle, cands, _only("guaranteed_ko", G.FORME_STATS))
    assert out[0] is cands[1]  # the stored numbers' pick: the stack was not skipped
    assert report.stages == [f"{G.FORME_STATS}_error", "guaranteed_ko"]
    assert report.forme_stats is None


# --- what the brain sees -------------------------------------------------------------


def test_the_observations_knowledge_and_threat_blocks_do_not_move():
    battle = _raichu()
    blastoise, raichu = battle.active_pokemon[0], battle.opponent_active_pokemon[0]
    assert blastoise is not None and raichu is not None

    def blocks() -> tuple[list[float], list[float]]:
        return (
            K.pokemon_knowledge(battle, blastoise, is_ours=True),
            K.threat_knowledge(
                battle,
                blastoise,
                list(blastoise.moves.values()),
                [raichu],
                [list(raichu.moves.values())],
                ours=True,
            ),
        )

    before = blocks()
    G.apply_guards(battle, _raichu_pairs(battle), _only("threat_first2", G.FORME_STATS))
    assert blocks() == before
    with F.current_forme(battle):  # the test can tell the two lines apart
        inside = blocks()
    assert inside != before
    assert before[1][2] == 0.0 and inside[1][2] == 1.0  # "this foe can KO me"


def test_the_decision_audit_records_the_correction(tmp_path):
    battle = _gengar()
    cands = _gengar_pairs(battle)
    out, report = G.apply_guards(battle, cands, _only("guaranteed_ko", G.FORME_STATS))
    player = PolicyPlayer.__new__(PolicyPlayer)
    player.decision_log_path = tmp_path / "decisions.jsonl"
    player._audit_decision(battle, out, None, report)
    row = json.loads(player.decision_log_path.read_text())
    assert row["guards"]["stages"] == [G.FORME_STATS]
    assert row["guards"]["forme_stats"] == {
        "corrected": ["gengar"],
        "old_pick": list(cands[1].actions),
    }
    # a decision without a correction keeps the record it always had
    plain, plain_report = G.apply_guards(
        battle, _gengar_pairs(battle), _only("guaranteed_ko")
    )
    player._audit_decision(battle, plain, None, plain_report)
    last = json.loads(player.decision_log_path.read_text().splitlines()[-1])
    assert "forme_stats" not in last["guards"]
