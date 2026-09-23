"""tools/ladder_opening_audit.py: setters from the team sheet, our Trick Room
plan (setter led, fainted before Trick Room, turn it went up), turn-1 Fake Out,
knockout attribution (move vs residual), and the revealed archetype."""

from __future__ import annotations

from tools.ladder_opening_audit import archetype, opening_events, setters_from_team

TEAM = """Farigiraf (F) @ Sitrus Berry
Ability: Armor Tail
- Trick Room
- Psychic

Torkoal (M) @ Charcoal
Ability: Drought
- Eruption
- Protect
"""

LOSS = "\n".join(
    [
        "|player|p1|antonius1|1|1245",
        "|player|p2|rival|2|1260",
        "|switch|p1a: Farigiraf|Farigiraf, L50, F|100/100",
        "|switch|p1b: Torkoal|Torkoal, L50, M|100/100",
        "|switch|p2a: Rillaboom|Rillaboom, L50, M|100/100",
        "|switch|p2b: Garchomp|Garchomp, L50, F|100/100",
        "|turn|1",
        "|move|p2a: Rillaboom|Fake Out|p1b: Torkoal",
        "|-damage|p1b: Torkoal|90/100",
        "|move|p2b: Garchomp|Rock Slide|p1a: Farigiraf|[spread] p1a,p1b",
        "|-damage|p1a: Farigiraf|0 fnt",
        "|-damage|p1b: Torkoal|60/100",
        "|faint|p1a: Farigiraf",
        "|turn|2",
        "|move|p2b: Garchomp|Tailwind|p2b: Garchomp",
        "|-damage|p1b: Torkoal|0 fnt|[from] Sandstorm",
        "|faint|p1b: Torkoal",
        "|win|rival",
    ]
)

WIN = "\n".join(
    [
        "|player|p1|rival|1|1200",
        "|player|p2|antonius1|2|1245",
        "|switch|p2a: Farigiraf|Farigiraf, L50, F|100/100",
        "|switch|p2b: Torkoal|Torkoal, L50, M|100/100",
        "|switch|p1a: Pelipper|Pelipper, L50, M|100/100",
        "|-weather|RainDance|[from] ability: Drizzle|[of] p1a: Pelipper",
        "|turn|1",
        "|move|p2a: Farigiraf|Trick Room|p2a: Farigiraf",
        "|-fieldstart|move: Trick Room|[of] p2a: Farigiraf",
        "|win|antonius1",
    ]
)


def test_setters_come_from_the_team_sheet() -> None:
    assert setters_from_team(TEAM) == {"Farigiraf"}


def test_setter_knocked_out_before_trick_room() -> None:
    events = opening_events(LOSS, setters={"Farigiraf"})
    assert events["our_tr_up_turn"] is None
    assert events["setter_fainted_before_tr"] == [(1, "Farigiraf")]
    assert events["opp_fake_out_turn1"] == [(1, "Torkoal")]
    assert events["knockouts"] == [
        {"turn": 1, "species": "Farigiraf", "by": "Garchomp", "move": "Rock Slide"},
        {"turn": 2, "species": "Torkoal", "by": None, "move": "Sandstorm"},
    ]
    assert events["archetype"] == "grassy_fakeout"  # outranks Tailwind


def test_trick_room_up_and_rain_archetype_when_we_are_p2() -> None:
    events = opening_events(WIN, setters={"Farigiraf"})
    assert events["our_tr_up_turn"] == 1 and events["our_tr_count"] == 1
    assert events["our_tr_attempts"] == [(1, "Farigiraf")]
    assert events["setter_fainted_before_tr"] == [] and events["knockouts"] == []
    assert events["archetype"] == "rain"


def test_archetype_prefers_the_full_sheet() -> None:
    empty = {"moves": set(), "abilities": set(), "species": set(), "weather": set()}
    assert archetype("Indeedee||PsychicSurge|...", empty) == "psychic_terrain"
    assert archetype("", empty) == "balance"
