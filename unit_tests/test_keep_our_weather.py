"""keep_our_weather (ladder 2026-09-25, game 10): Farigiraf's Rain Dance was up
for Mega Blastoise when Farigiraf fainted; Torkoal came in (the policy 99.9%)
and Drought's sun halved Blastoise's Water Spouts. Also the weather tracker in
pokeenv_patches it relies on: poke-env forgets who set the weather and rewrites
its start turn every upkeep."""

from __future__ import annotations

from unit_tests.ladder_position import move_action, position, run, switch_action
from vgc_bench.src import guards as G


def _replacement(partner: str, weather: list[str], decision_turn: int = 2):
    """Our slot b fainted at the end of a turn; slot a (``partner``) stays."""
    lines = [
        f"|switch|p1a: {partner}|{partner}, L50, M|186/186",
        "|switch|p1b: Farigiraf|Farigiraf, L50, F|227/227",
        "|switch|p2a: Rillaboom|Rillaboom, L50, M|100/100",
        "|switch|p2b: Milotic|Milotic, L50, F|74/100",
        "|turn|2",
        *weather,
    ]
    if decision_turn != 2:
        lines.append(f"|turn|{decision_turn}")
    lines += ["|faint|p1b: Farigiraf", "|faint|p2a: Rillaboom"]
    battle = position(lines)
    torkoal, venusaur = (switch_action(battle, s) for s in ("torkoal", "venusaur"))
    return battle, [((0, torkoal), 0.999), ((0, venusaur), 0.001)]


OUR_RAIN = ["|move|p1b: Farigiraf|Rain Dance|p1b: Farigiraf", "|-weather|RainDance"]


def test_venusaur_replaces_torkoal_beside_blastoise_in_our_rain():
    battle, pairs = _replacement("Blastoise", OUR_RAIN)
    out, report = run(G.guard_keep_our_weather, battle, pairs)
    assert out[0].actions == pairs[1][0]
    assert report.demotions["keep_our_weather"] == 1
    assert "keep_our_weather" in report.stages


def test_a_sun_partner_takes_torkoal():
    """Charizard's Heat Wave wants the sun Torkoal brings."""
    battle, pairs = _replacement("Charizard", OUR_RAIN)
    out, report = run(G.guard_keep_our_weather, battle, pairs)
    assert out[0].actions == pairs[0][0]
    assert not report.stages


def test_their_rain_is_theirs_to_lose():
    """Rain from the opponent's Drizzle: overwriting it is the usual answer."""
    weather = [
        "|switch|p2a: Pelipper|Pelipper, L50, M|100/100",
        "|-weather|RainDance|[from] ability: Drizzle|[of] p2a: Pelipper",
    ]
    battle, pairs = _replacement("Blastoise", weather)
    out, report = run(G.guard_keep_our_weather, battle, pairs)
    assert out[0].actions == pairs[0][0]
    assert not report.stages


def test_rain_on_its_last_turn_is_not_worth_keeping():
    battle, pairs = _replacement("Blastoise", OUR_RAIN, decision_turn=6)
    out, report = run(G.guard_keep_our_weather, battle, pairs)
    assert out[0].actions == pairs[0][0]
    assert not report.stages


def test_a_voluntary_switch_is_judged_the_same_way():
    battle = position(
        [
            "|switch|p1a: Blastoise|Blastoise, L50, M|186/186",
            "|switch|p1b: Farigiraf|Farigiraf, L50, F|227/227",
            "|switch|p2a: Rillaboom|Rillaboom, L50, M|100/100",
            "|switch|p2b: Milotic|Milotic, L50, F|74/100",
            "|turn|2",
            *OUR_RAIN,
            "|turn|3",
        ]
    )
    spout = move_action(battle, 0, "waterspout", 0)
    psychic = move_action(battle, 1, "psychic", 2)
    torkoal = switch_action(battle, "torkoal")
    out, _ = run(
        G.guard_keep_our_weather,
        battle,
        [((spout, torkoal), 0.6), ((spout, psychic), 0.4)],
    )
    assert out[0].actions == (spout, psychic)


def test_an_indifferent_partner_takes_torkoal():
    """Farigiraf's Psychic is the same in rain and sun."""
    battle = position(
        [
            "|switch|p1a: Blastoise|Blastoise, L50, M|186/186",
            "|switch|p1b: Farigiraf|Farigiraf, L50, F|227/227",
            "|switch|p2a: Rillaboom|Rillaboom, L50, M|100/100",
            "|turn|2",
            *OUR_RAIN,
            "|faint|p1a: Blastoise",
        ]
    )
    torkoal, venusaur = (switch_action(battle, s) for s in ("torkoal", "venusaur"))
    out, report = run(
        G.guard_keep_our_weather,
        battle,
        [((torkoal, 0), 0.999), ((venusaur, 0), 0.001)],
    )
    assert out[0].actions == (torkoal, 0)
    assert not report.stages


def _tracked(battle) -> tuple[object, object]:
    return (
        getattr(battle, "_vgc_weather_side", None),
        getattr(battle, "_vgc_weather_start", None),
    )


def _feed(battle, line: str) -> None:
    battle.parse_message(line.split("|"))


def test_the_tracker_records_the_setter_side_and_turn():
    battle, _ = _replacement("Blastoise", OUR_RAIN)
    assert _tracked(battle) == ("p1", 2)
    _feed(battle, "|turn|3")
    _feed(battle, "|-weather|RainDance|[upkeep]")
    assert _tracked(battle) == ("p1", 2)  # poke-env itself rewrites the turn here
    _feed(battle, "|-weather|SunnyDay|[from] ability: Drought|[of] p2a: Torkoal")
    assert _tracked(battle) == ("p2", 3)
    _feed(battle, "|-weather|none")
    assert _tracked(battle) == (None, None)


def test_registered_but_opt_in():
    assert "keep_our_weather" in G.GUARDS
    assert "keep_our_weather" not in G.HARD_GUARDS
    assert G.GUARD_ORDER.index("keep_our_weather") < G.GUARD_ORDER.index("protect_spam")
