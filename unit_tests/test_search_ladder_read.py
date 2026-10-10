"""evaluation/search_ladder_read.py: the readings of a ladder trial played with the
search -- above all the comparison "was the opponent's real reply among the eight the
predictor expects?", which has to follow the same matching rule as the search's own
"was it in the table" (move, target where the command names one, Mega Evolution)."""

from __future__ import annotations

import itertools
import json
from typing import Any, cast

import pytest

from evaluation.search_ladder_read import (
    _forecast_hit,
    forecast_matches,
    forecast_pairs,
    pool,
    read,
    render,
    reply_kind,
    share,
    sign_test,
    wilson,
)
from vgc_bench.src.live_exact import ObservedAction, choice_matches_observation
from vgc_bench.src.oppmodel.runtime import ActionForecast


def _slot(*actions, p_mega=0.0):
    return {"actions": list(actions), "p_mega": p_mega}


def _move(move, p, target="auto"):
    return ActionForecast("move", p, move=move, target=target).to_dict()


def _switch(species, p, roster_index=0):
    # the runtime's own record of a switch, as decisions_champion.jsonl holds it
    action = ActionForecast("switch", p, switch_to=species, roster_index=roster_index)
    return action.to_dict()


FORECAST = {
    "slots": [
        _slot(_move("rockslide", 0.6), _switch("pelipper", 0.3), p_mega=0.8),
        _slot(_move("powergem", 0.5, "foe_b"), _switch("pelipper", 0.4)),
    ]
}


def test_joint_replies_are_products_without_what_the_rules_forbid():
    plain = forecast_pairs(FORECAST, mega=False)
    assert [(a[1], b[1]) for a, b, _, _ in plain] == [
        ("rockslide", "powergem"),
        ("rockslide", "pelipper"),
        ("pelipper", "powergem"),
    ]  # both switching to Pelipper is not a reply
    assert plain[0][3] == pytest.approx(0.30) and plain[0][2] is None
    with_mega = forecast_pairs(FORECAST, mega=True)
    # Rock Slide with the Mega Evolution (0.8) comes before Rock Slide without it
    assert with_mega[0][2] == (True, False)
    assert with_mega[0][3] == pytest.approx(0.6 * 0.5 * 0.8)
    unevolved = next(
        pair
        for pair in with_mega
        if pair[:3] == (plain[0][0], plain[0][1], (False, False))
    )
    assert unevolved[3] == pytest.approx(0.6 * 0.5 * 0.2)
    assert [round(pair[3], 3) for pair in with_mega] == [0.24, 0.192, 0.15, 0.06, 0.048]
    # a switch does not Mega-evolve; slot b's Pokemon cannot (p_mega 0)
    assert all(state in ((False, False), (True, False)) for _, _, state, _ in with_mega)
    assert not any(a[0] == "switch" and state[0] for a, _, state, _ in with_mega)
    assert len(forecast_pairs(FORECAST, k=2)) == 2
    # an empty slot is a slot that does nothing
    alone = forecast_pairs({"slots": [FORECAST["slots"][0], None]}, mega=False)
    assert [b[0] for _, b, _, _ in alone] == ["none", "none"]
    assert forecast_pairs({}) == [
        (("none", "", None), ("none", "", None), (False, False), 1.0)
    ]


def test_a_logged_switch_is_read_under_the_runtimes_own_key():
    """Until 2026-10-10 the destination was read under a key the runtime never writes:
    no forecast switch could match the reply seen, and two switches both read as ""
    were dropped as "the same Pokemon" -- the predictor's own pairs were undercounted
    on every reply holding a switch."""
    record = {
        "slots": [
            _slot(_switch("pelipper", 0.7, 2), _move("protect", 0.3)),
            _slot(_switch("archaludon", 0.6, 3), _move("protect", 0.4)),
        ]
    }
    assert record["slots"][0]["actions"][0] == {
        "kind": "switch",
        "p": 0.7,
        "switch_to": "pelipper",
        "roster_index": 2,
    }
    pairs = forecast_pairs(record, mega=False)
    # two different switches are a reply, and the likeliest one here
    assert (pairs[0][0], pairs[0][1]) == (
        ("switch", "pelipper", None),
        ("switch", "archaludon", None),
    )
    assert forecast_matches(pairs[0][0], ["switch", "pelipper", None, False], True)
    assert forecast_matches(pairs[0][1], ["switch", "archaludon", None, False], True)
    assert not forecast_matches(pairs[0][0], ["switch", "archaludon", None, 0], True)


def test_a_forecast_action_matches_by_the_searchs_own_rule():
    gem_at_b = ("move", "powergem", "foe_b")
    assert forecast_matches(gem_at_b, ["move", "powergem", 2, False], strict=True)
    assert not forecast_matches(gem_at_b, ["move", "powergem", 1, False], strict=True)
    assert forecast_matches(gem_at_b, ["move", "powergem", 1, False], strict=False)
    # a spread move's log names one target, the forecast none: the move decides
    assert forecast_matches(
        ("move", "rockslide", "auto"), ["move", "rockslide", 2, 0], True
    )
    assert forecast_matches(
        ("move", "helpinghand", "ally"), ["move", "helpinghand", -1, 0], True
    )
    assert not forecast_matches(
        ("move", "psychic", "ally"), ["move", "psychic", 1, 0], True
    )
    assert forecast_matches(
        ("switch", "pelipper", None), ["switch", "pelipper", None, 0], True
    )
    assert not forecast_matches(
        ("switch", "pelipper", None), ["move", "protect", None, 0], True
    )
    # the bucket "some other move" names nothing; a slot that showed nothing fits all
    assert not forecast_matches(("other", "", None), ["move", "protect", None, 0], True)
    assert forecast_matches(("other", "", None), None, True)


def test_the_reading_and_the_search_share_one_matching_rule():
    """Reading 5 sets the predictor's own list beside the search's table "by the same
    rule". For replies made of moves that rule is the search's own
    ``choice_matches_observation``, and the reading agrees with it case by case: a
    target is compared where the command names one; a Mega Evolution that was SEEN must
    be named, one that is named but was not seen is not held against the entry; a slot
    that showed nothing fits anything. (Until 2026-10-10 the reading asked for the exact
    Mega state of both slots, a silent slot counted as not Mega-evolving -- stricter
    than the table's own check, against the predictor.)"""
    command = {"auto": "", "foe_a": " +1", "foe_b": " +2"}
    entries = [
        ("rockslide", "auto"),
        ("powergem", "foe_a"),
        ("powergem", "foe_b"),
        ("protect", "auto"),
    ]
    sightings = [
        None,  # the slot showed nothing
        ("rockslide", 2, False),
        ("rockslide", 1, True),
        ("powergem", 1, False),
        ("powergem", 2, True),
        ("powergem", None, False),
        ("protect", None, False),
    ]
    checked = 0
    for a, b in itertools.product(entries, repeat=2):
        for state in ((False, False), (True, False), (False, True)):
            choice = ", ".join(
                f"move {move}{command[target]}{' mega' if evolves else ''}"
                for (move, target), evolves in zip((a, b), state)
            )
            pair = [(("move", *a), ("move", *b), state, 1.0)]
            for seen in itertools.product(sightings, repeat=2):
                actions = {
                    slot: ObservedAction("move", *what)
                    for slot, what in enumerate(seen)
                    if what is not None
                }
                logged = {
                    slot: [act.kind, act.identifier, act.target, act.mega]
                    for slot, act in actions.items()
                }
                searchs = choice_matches_observation(
                    cast(Any, None), "p2", choice, actions
                )
                assert _forecast_hit(pair, logged, True, True) == searchs, (
                    choice,
                    logged,
                )
                checked += 1
    assert checked == 16 * 3 * 49
    # the two cases the old rule got wrong, spelled out
    silent_partner = {0: ["move", "protect", None, False]}
    protect, gem = ("move", "protect", "auto"), ("move", "powergem", "foe_b")
    evolving_partner = [(protect, gem, (False, True), 1.0)]
    assert _forecast_hit(evolving_partner, silent_partner, True, True)
    unseen_mega = [(protect, gem, (True, False), 1.0)]
    assert _forecast_hit(unseen_mega, silent_partner, True, True)
    seen_mega = {0: ["move", "protect", None, True]}
    assert not _forecast_hit(evolving_partner, seen_mega, True, True)


def test_replies_are_told_apart_by_what_they_are_made_of():
    assert reply_kind({"0": ["move", "rockslide", 2, False]}) == "moves only"
    assert reply_kind({"0": ["move", "spikyshield", None, False]}) == "with a protect"
    both = {"0": ["switch", "pelipper", None, False], "1": ["move", "protect", None, 0]}
    assert reply_kind(both) == "with a switch"


def test_small_helpers():
    assert share(12, 40) == "12 of 40 (30%)" and share(0, 0) == "0 of 0"
    # the decisions on which two priors differ: 16 to 0 is no accident, 5 to 4 is
    assert sign_test(16, 0) == pytest.approx(2 / 2**16)
    assert sign_test(5, 4) == 1.0 and sign_test(0, 0) == 1.0
    assert sign_test(9, 1) == pytest.approx(22 / 1024)
    low, high = wilson(22, 40)
    assert (round(100 * low, 1), round(100 * high, 1)) == (39.8, 69.3)


def _audit(turn, actions, champion, coverage=None, mode="search", **more):
    return {
        "battle": "battle-gen9championsvgc2026regmc-1-x",
        "turn": turn,
        "exact_search": {
            "schedule": {"mode": mode, "preparation_elapsed_s": 0.25},
            "result": {"elapsed_s": 2.0, "choice": "move a, move b"},
            "actions": actions,
            "champion_actions": champion,
            "champion_choice": "move c, move d",
            "reply_coverage": coverage,
            "roots_agreeing_with_evidence": 8,
            **more,
        },
    }


def test_a_directory_is_read_from_its_logs_alone(tmp_path):
    """No replays: the search's own rows carry readings 1 to 5b."""
    observed = {
        "0": ["move", "rockslide", 2, True],
        "1": ["move", "powergem", 2, False],
    }
    priors = {
        "any": {
            "brain": False,
            "sum25": True,
            "sum50": True,
            "sum75": True,
            "forecast": True,
        },
        "top_mass": {
            "brain": 0.0,
            "sum25": 0.5,
            "sum50": 0.5,
            "sum75": 1.0,
            "forecast": 1.0,
        },
        "best_rank": {"brain": 12, "sum25": 7, "sum50": 4, "sum75": 2, "forecast": 1},
    }
    rows = [
        _audit(1, [9, 9], [9, 9]),
        _audit(
            2,
            [9, 10],
            [9, 9],
            {
                "any": False,
                "mass_with_reply": 0.0,
                "observed": observed,
                "priors": priors,
            },
        ),
        _audit(3, None, [9, 9], mode="error_fallback"),
    ]
    rows[2]["exact_search"]["decision_fallback"] = {
        "error": "Set Arbok has no moves\nat"
    }
    (tmp_path / "decisions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    own = [{"battle": rows[0]["battle"], "turn": 1, "opponent_forecast": FORECAST}]
    (tmp_path / "decisions_champion.jsonl").write_text(json.dumps(own[0]) + "\n")
    reading = read(tmp_path)
    assert reading["search"]["modes"] == {"search": 2, "error_fallback": 1}
    assert reading["search"]["reasons"] == {"error_fallback: Set Arbok has no moves": 1}
    assert reading["search"]["seconds"]["max"] == 2.25
    assert reading["reply_table"]["hidden"] == {
        "decisions": 1,
        "in_some_world": 0,
        "world_mass": 0.0,
    }
    assert [o["turn"] for o in reading["overrides"]] == [2]
    # the predictor's first pair was the real reply, Mega Evolution included
    assert reading["forecast"]["hidden: forecast"] == 1
    assert reading["forecast"]["hidden: forecast top 1"] == 1
    assert reading["forecast"]["hidden: forecast only"] == 1
    assert reading["prior_ranks"]["forecast"] == [1] and reading["prior_ranks"][
        "brain"
    ] == [12]
    assert reading["priors"]["hidden"]["mass"]["sum75"] == 1.0
    # the table as played missed the reply, and so would the brain's ranking alone
    assert reading["played_against_brain"] == {"neither": 1}
    text = "\n".join(render(reading, games=True))
    assert "5b. THE REAL REPLY UNDER OTHER REPLY PRIORS" in text
    assert "Set Arbok has no moves" in text
    pooled = "\n".join(pool([("one", reading), ("two", reading)]))
    assert "searched 4 of 6 (67%)" in pooled and "overrides 2 of 4" in pooled
