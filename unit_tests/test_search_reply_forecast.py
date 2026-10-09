"""The opponent predictor's forecast as the search's reply prior (2026-10-09, opt-in).

The matrix search values each of our candidates against the opponent's eight likeliest
replies. Until now "likeliest" was the brain's own prior from the opponent's seat. On
the first ladder games played with the search (2026-10-09) the reply the opponent
really made was in that table 9 times in 24 with open sheets -- against bots it is 93
to 96 -- while the predictor's eight likeliest joint replies held it 19 times.

``PolicyPlayer(exact_reply_forecast=True)`` / ``--search-reply-prior forecast`` hands
the search the forecast for the decision it plans; ``OpponentModelPrior`` then ranks
the opponent's replies to that decision with it. Everything else -- forced
replacements, any other turn, our own candidates -- is ranked as before, and without
the option nothing reads the forecast (shadow mode's promise, pinned in
test_opponent_forecast_shadow.py).
"""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ladder_ourteam import (
    SEARCH_FLAG_DEFAULTS,
    build_parser,
    forecast_status,
    material_config,
)
from vgc_bench.src.exact_observation import (
    OpponentModelPrior,
    RankedChoice,
    RootForecast,
    _mega_likelihood,
    _state_roster,
    replies_by_mixture,
)
from vgc_bench.src.live_exact import LiveExactSession, ObservedAction
from vgc_bench.src.oppmodel.events import TARGET_CLASSES
from vgc_bench.src.opponent_tactics import MovePrediction, SwitchPrediction
from vgc_bench.src.policy_player import PolicyPlayer

MOVE_A = "move rockslide"
REPLIES = [
    f"{MOVE_A}, move powergem +1",
    f"{MOVE_A}, move powergem +2",
    "move direclaw +1, move powergem +1",
    "move direclaw +2, move spikyshield",
    "move protect, move earthpower +1",
    "switch 3, move powergem +1",
    "switch 4, move powergem +1",
]


class _Brain:
    """The base prior: the replies in the order given, most probable first."""

    reveal_opponent_sets = False

    def __init__(self):
        self.calls = 0

    def rank(self, _state, _requests, _role, choices):
        self.calls += 1
        weights = [2.0 ** -(index + 1) for index in range(len(choices))]
        total = sum(weights)
        return [
            RankedChoice(choice, (index, index), weight / total)
            for index, (choice, weight) in enumerate(zip(choices, weights))
        ]


def _state(turn: int = 3) -> dict:
    party = ["Sneasler", "Glimmora", "Swampert", "Pelipper"]
    return {
        "turn": turn,
        "sides": [
            {"pokemon": []},
            {"pokemon": [{"set": {"species": name}} for name in party]},
        ],
    }


MOVE_REQUEST = [{}, {"active": [{}, {}]}]
FORCED = [{}, {"forceSwitch": [True, False]}]


def _forecast(turn: int = 3, **changes: Any) -> RootForecast:
    """Sneasler: Dire Claw into our slot b, hardly ever Rock Slide; Glimmora: Spiky
    Shield; a switch to Pelipper is the only one considered."""
    slot_a = MovePrediction(
        moves=(("direclaw", 0.70), ("rockslide", 0.05), ("protect", 0.10)),
        targets=(("foe_b", 0.9), ("foe_a", 0.1)),
        actions=(
            ("direclaw", "foe_b", 0.63),
            ("direclaw", "foe_a", 0.07),
            ("rockslide", "field", 0.05),
            ("protect", "self", 0.10),
        ),
        reliability=1.0,
    )
    slot_b = MovePrediction(
        moves=(("spikyshield", 0.60), ("powergem", 0.25), ("earthpower", 0.05)),
        targets=(("self", 1.0),),
        actions=(
            ("spikyshield", "self", 0.60),
            ("powergem", "foe_a", 0.20),
            ("powergem", "foe_b", 0.05),
            ("earthpower", "foe_a", 0.05),
        ),
        reliability=1.0,
    )
    switches = (
        SwitchPrediction(0.15, (("pelipper", 0.9), ("swampert", 0.1))),
        SwitchPrediction(0.10, (("swampert", 1.0),)),
    )
    values: dict[str, Any] = dict(
        turn=turn, moves=(slot_a, slot_b), switches=switches, mega=None
    )
    values.update(changes)
    return RootForecast(**values)


def _prior(**options: Any) -> tuple[OpponentModelPrior, _Brain]:
    brain = _Brain()
    return OpponentModelPrior(brain, controlled_role="p1", **options), brain  # type: ignore[arg-type]


def _order(ranked) -> list[str]:
    return [item.choice for item in ranked]


def test_without_a_forecast_the_brain_ranks_the_replies_as_it_always_did():
    prior, _ = _prior()
    assert _order(prior.rank(_state(), MOVE_REQUEST, "p2", REPLIES)) == REPLIES
    assert prior.forecast_rankings == 0


def test_the_forecast_ranks_the_replies_to_the_decision_it_was_made_for():
    prior, _ = _prior()
    assert (prior.forecast_weight, prior.forecast_mix) == (0.5, "sum")
    prior.set_root_forecast(_forecast())
    ranked = prior.rank(_state(), MOVE_REQUEST, "p2", REPLIES)
    # what the predictor expects comes first, whatever the brain thought of it ...
    assert ranked[0].choice == "move direclaw +2, move spikyshield"
    assert REPLIES.index(ranked[0].choice) == 3
    # ... and what only the brain expects stays next: the sum keeps both lists
    assert ranked[1].choice == REPLIES[0]
    assert sum(item.probability for item in ranked) == pytest.approx(1.0)
    assert prior.forecast_rankings == 1
    # a switch goes to the Pokemon the forecast names: party position 4 is Pelipper
    order = _order(ranked)
    assert order.index("switch 4, move powergem +1") < order.index(
        "switch 3, move powergem +1"
    )


@pytest.mark.parametrize(
    "state, requests, role",
    [
        (_state(turn=4), MOVE_REQUEST, "p2"),  # a later turn inside the search
        (_state(), FORCED, "p2"),  # their replacement after a faint
        (_state(), MOVE_REQUEST, "p1"),  # our own candidates
    ],
    ids=["another turn", "a forced replacement", "our side"],
)
def test_everything_else_is_ranked_as_before(state, requests, role):
    prior, _ = _prior()
    prior.set_root_forecast(_forecast())
    assert _order(prior.rank(state, requests, role, REPLIES)) == REPLIES
    assert prior.forecast_rankings == 0


def test_a_side_with_one_legal_reply_is_not_a_ranking():
    """While we replace a fainted Pokemon the opponent waits: one reply, ""."""
    prior, _ = _prior()
    prior.set_root_forecast(_forecast())
    assert _order(prior.rank(_state(), [{}, {"wait": True}], "p2", [""])) == [""]
    assert prior.forecast_rankings == 0 and prior.reply_log == {}


def test_a_forecast_is_for_one_decision():
    prior, _ = _prior()
    prior.set_root_forecast(_forecast())
    prior.rank(_state(), MOVE_REQUEST, "p2", REPLIES)
    prior.set_root_forecast(None)
    assert _order(prior.rank(_state(), MOVE_REQUEST, "p2", REPLIES)) == REPLIES
    assert prior.forecast_rankings == 0
    # one that names nothing is no forecast
    prior.set_root_forecast(_forecast(moves=None, switches=None))
    assert _order(prior.rank(_state(), MOVE_REQUEST, "p2", REPLIES)) == REPLIES


def test_a_product_keeps_only_what_both_expect():
    """As the move / switch models are blended. The brain's favourite needs Rock
    Slide (0.05 for the forecast) with Power Gem at our slot a: under the product it
    falls behind a reply both rate, under the sum it stays second."""
    prior, _ = _prior(forecast_weight=0.8, forecast_mix="product")
    prior.set_root_forecast(_forecast())
    order = _order(prior.rank(_state(), MOVE_REQUEST, "p2", REPLIES))
    assert order[0] == "move direclaw +2, move spikyshield"
    assert order.index(REPLIES[0]) > order.index(REPLIES[2])
    with pytest.raises(ValueError, match="forecast_mix"):
        _prior(forecast_mix="average")


def test_weight_zero_is_the_brains_ranking_and_still_logs_both_priors():
    """The measuring mode: the search plays as with the brain's prior alone, and the
    two probabilities of every legal reply are kept for the next decision."""
    prior, brain = _prior(forecast_weight=0.0)
    prior.set_root_forecast(_forecast())
    ranked = prior.rank(_state(), MOVE_REQUEST, "p2", REPLIES)
    untouched = brain.rank(None, None, "p2", REPLIES)
    assert [(r.choice, r.actions, r.probability) for r in ranked] == [
        (r.choice, r.actions, r.probability) for r in untouched
    ]
    assert prior.forecast_rankings == 1
    (entry,) = prior.reply_log.values()
    assert set(entry) == set(REPLIES)
    assert sum(b for b, _ in entry.values()) == pytest.approx(1.0)
    assert sum(f for _, f in entry.values()) == pytest.approx(1.0)
    assert max(entry, key=lambda reply: entry[reply][1]) == REPLIES[3]
    assert max(entry, key=lambda reply: entry[reply][0]) == REPLIES[0]
    # how any mixture would have ranked them, from the log alone
    assert replies_by_mixture(entry, 0.0) == REPLIES
    assert replies_by_mixture(entry, 1.0)[0] == REPLIES[3]
    assert replies_by_mixture(entry, 0.5)[:2] == [REPLIES[3], REPLIES[0]]
    assert replies_by_mixture(entry, 0.8, "product")[0] == REPLIES[3]
    # the next decision: what was logged becomes "the decision before"
    prior.set_root_forecast(None)
    assert prior.reply_log == {} and list(prior.previous_reply_log.values()) == [entry]
    prior.set_root_forecast(None)
    assert prior.previous_reply_log == {}


def test_the_weight_is_the_forecasts_share_against_the_brain():
    brain_only, _ = _prior(forecast_weight=0.0)
    brain_only.set_root_forecast(_forecast())
    assert _order(brain_only.rank(_state(), MOVE_REQUEST, "p2", REPLIES)) == REPLIES
    # equal forecast probability: the brain's order decides (a stable sort)
    alone, _ = _prior(forecast_weight=1.0)
    twins = [
        "move rockslide, move spikyshield",
        "move rockslide mega, move spikyshield",
    ]
    alone.set_root_forecast(_forecast())
    assert _order(alone.rank(_state(), MOVE_REQUEST, "p2", twins)) == twins
    with pytest.raises(ValueError, match="forecast_weight"):
        _prior(forecast_weight=1.5)


def test_the_mega_probability_splits_the_two_spellings_of_a_move():
    twins = [
        "move rockslide, move spikyshield",
        "move rockslide mega, move spikyshield",
    ]
    for probability, first in ((0.9, twins[1]), (0.1, twins[0])):
        prior, _ = _prior(forecast_weight=1.0)
        prior.set_root_forecast(_forecast(mega=(probability, None)))
        assert prior.rank(_state(), MOVE_REQUEST, "p2", twins)[0].choice == first
    assert _mega_likelihood("move megahorn +1, move protect", (0.9, None)) == (
        pytest.approx(0.1)
    )  # "mega" in a move's name is not the event
    assert _mega_likelihood("move megahorn +1 mega, pass", (0.9, 0.5)) == (
        pytest.approx(0.9)
    )
    assert _mega_likelihood("switch 3, move protect mega", (0.9, 1.0)) == (
        pytest.approx(0.98)
    )  # clamped: one confident head does not erase the other spelling
    assert _mega_likelihood("move protect, move protect", None) == 1.0


def test_the_party_is_read_from_the_exact_state():
    assert _state_roster(_state(), "p2") == (
        "sneasler",
        "glimmora",
        "swampert",
        "pelipper",
    )
    assert _state_roster({}, "p2") == ()
    assert _state_roster(
        {"sides": [{}, {"pokemon": [None, {"species": "X"}]}]}, "p2"
    ) == ("", "x")


# --- the session: a predictor's Forecast becomes the prior's RootForecast -----------


def _slot(moves, action, target, *, p_mega=0.0, roster=("sneasler", "glimmora")):
    n = len(moves)
    full = np.zeros(n + 1 + 6)
    full[: len(action)] = action
    targets = np.zeros((n + 1, len(TARGET_CLASSES)))
    for row, shares in enumerate(target):
        targets[row, : len(shares)] = shares
    return SimpleNamespace(
        moves=tuple(moves),
        roster=tuple(roster) + ("",) * (6 - len(roster)),
        action_probs=full,
        target_probs=targets,
        known_moves=0,
        p_mega=p_mega,
    )


def _session() -> Any:
    session: Any = object.__new__(LiveExactSession)
    session.prior, _ = _prior()
    return session


def test_the_session_reads_a_forecast_at_face_value_and_names_its_turn():
    # foe_a, foe_b, ally, self, auto
    a = _slot(["direclaw", "rockslide"], [0.7, 0.2], [[0.1, 0.9], [0, 0, 0, 0, 1]])
    b = _slot(["spikyshield"], [0.9], [[0, 0, 0, 1]], p_mega=0.75)
    forecast = SimpleNamespace(a=a, b=b, turn=3, slots=(a, b))
    session = _session()
    session.set_reply_forecast(forecast)
    assert session.reply_forecast_status == "set"
    root = session.prior.root_forecast
    assert root.turn == 3 and root.mega == (0.0, 0.75)
    # hidden sheets, nothing shown yet: still the forecast's own numbers, not a
    # blend with a uniform (the adaptor's default reliability is known moves / 4)
    assert [m.reliability for m in root.moves] == [1.0, 1.0]
    assert dict(root.moves[0].moves) == pytest.approx(
        {"direclaw": 0.7, "rockslide": 0.2}
    )
    assert ("direclaw", "foe_b", pytest.approx(0.63)) in root.moves[0].actions
    ranked = session.prior.rank(_state(), MOVE_REQUEST, "p2", REPLIES)
    assert ranked[0].choice == "move direclaw +2, move spikyshield"
    # an empty opposing slot is no obstacle
    session.set_reply_forecast(SimpleNamespace(a=a, b=None, turn=5, slots=(a, None)))
    assert session.reply_forecast_status == "set"
    assert session.prior.root_forecast.mega == (0.0, None)
    # the turn can be given by the caller
    session.set_reply_forecast(forecast, turn=9)
    assert session.prior.root_forecast.turn == 9


def test_no_forecast_or_a_broken_one_leaves_the_ordinary_prior():
    session = _session()
    session.set_reply_forecast(SimpleNamespace(a=None, b=None, turn=3, slots=()))
    session.set_reply_forecast(None)
    assert session.reply_forecast_status == "none"
    assert session.prior.root_forecast is None
    session.set_reply_forecast(object())  # nothing a forecast has
    assert session.prior.root_forecast is None
    assert session.reply_forecast_status in ("unreadable", "failed:AttributeError")
    assert _order(session.prior.rank(_state(), MOVE_REQUEST, "p2", REPLIES)) == REPLIES


def test_the_session_holds_other_priors_against_the_real_reply():
    """What the weight-0 mode is for: where would the reply the opponent really made
    have ranked under the brain, the forecast and sums of the two?"""
    session = _session()
    session.config = SimpleNamespace(opponent_width=2)  # a table two replies wide
    # world b's sampled set has no Spiky Shield: the real reply is not legal there
    other = [reply for reply in REPLIES if "spikyshield" not in reply]
    world_a, world_b = _state(), _state()
    session.prior.set_root_forecast(_forecast())
    session.prior.rank(world_a, MOVE_REQUEST, "p2", REPLIES)
    session.prior.rank(world_b, MOVE_REQUEST, "p2", other)
    session.prior.set_root_forecast(None)  # the next decision begins
    observed = {
        0: ObservedAction("move", "direclaw", 2, False),
        1: ObservedAction("move", "spikyshield", None, False),
    }
    tables = [
        (SimpleNamespace(state=world_a), 0.75, {}, None),
        (SimpleNamespace(state=world_b), 0.25, {}, None),
    ]
    out = session._prior_coverage(tables, observed)
    assert (out["worlds"], out["width"]) == (2, 2)
    # REPLIES[3]: fourth for the brain, first for the forecast
    assert out["best_rank"] == {
        "brain": 4,
        "sum25": 2,
        "sum50": 1,
        "sum75": 1,
        "forecast": 1,
    }
    assert out["top_mass"] == {
        "brain": 0.0,
        "sum25": 0.75,
        "sum50": 0.75,
        "sum75": 0.75,
        "forecast": 0.75,
    }
    assert out["any"]["brain"] is False and out["any"]["forecast"] is True
    # nothing was logged (no forecast was handed over): nothing to say
    session.prior.set_root_forecast(None)
    assert session._prior_coverage(tables, observed) is None
    # and it cannot break a decision
    session.prior.previous_reply_log = {id(world_a): {"move tackle": (1.0, 1.0)}}
    del session.config
    assert session._prior_coverage(tables, observed) == {"failed": "AttributeError"}


def test_two_worlds_with_the_same_legal_replies_are_logged_apart():
    """Two worlds often give the Pokemon on the field the same four moves, and once
    advanced on shared random streams they carry the same seed -- and the brain
    still ranks their replies differently (the bench differs). Keyed by the legal
    replies, or by replies and seed, one world's ranking stood in for another's:
    in the rehearsals the table as played held the real reply while "the brain's
    prior" ranked it fifteenth. Two equal states are two worlds."""
    session = _session()
    session.config = SimpleNamespace(opponent_width=1)
    world_a = {**_state(), "prng": [1, 2, 3, 4]}
    world_b = {**_state(), "prng": [1, 2, 3, 4]}
    assert world_a == world_b
    session.prior.set_root_forecast(_forecast())
    session.prior.rank(world_a, MOVE_REQUEST, "p2", REPLIES)
    # in world b the brain ranks the replies the other way round
    session.prior.base.rank = lambda *_: [  # type: ignore[method-assign]
        RankedChoice(reply, (i, i), 2.0 ** -(len(REPLIES) - i))
        for i, reply in enumerate(REPLIES)
    ]
    session.prior.rank(world_b, MOVE_REQUEST, "p2", REPLIES)
    session.prior.set_root_forecast(None)
    assert len(session.prior.previous_reply_log) == 2
    observed = {  # REPLIES[6]: last for the brain in world a, first in world b
        0: ObservedAction("switch", "pelipper", None, False),
        1: ObservedAction("move", "powergem", 1, False),
    }
    tables = [
        (SimpleNamespace(state=world_a), 0.5, {}, None),
        (SimpleNamespace(state=world_b), 0.5, {}, None),
    ]
    out = session._prior_coverage(tables, observed)
    assert out["worlds"] == 2 and out["top_mass"]["brain"] == 0.5
    assert out["best_rank"]["brain"] == 1


def test_the_log_of_a_decision_is_there_at_the_next_one():
    """Handing over the next forecast moves the log once, not twice (the rehearsal
    of 2026-10-09 had the comparison only after replacements: the session cleared
    the prior and then set it, and the second step emptied what the first kept)."""
    a = _slot(["direclaw", "rockslide"], [0.7, 0.2], [[0.1, 0.9], [0, 0, 0, 0, 1]])
    b = _slot(["spikyshield"], [0.9], [[0, 0, 0, 1]])
    forecast = SimpleNamespace(a=a, b=b, turn=3, slots=(a, b))
    session = _session()
    session.set_reply_forecast(forecast)
    session.prior.rank(_state(), MOVE_REQUEST, "p2", REPLIES)
    assert len(session.prior.reply_log) == 1
    session.set_reply_forecast(SimpleNamespace(a=a, b=b, turn=4, slots=(a, b)))
    assert len(session.prior.previous_reply_log) == 1 and not session.prior.reply_log
    assert session.prior.root_forecast.turn == 4
    # a broken forecast still moves it once
    session.prior.rank(_state(turn=4), MOVE_REQUEST, "p2", REPLIES)
    session.set_reply_forecast(object())
    assert len(session.prior.previous_reply_log) == 1
    assert session.prior.root_forecast is None


# --- the player: off by default, and never in the way -------------------------------


class _Runtime:
    def __init__(self, answer):
        self.answer, self.asked = answer, 0

    def predict_with_reason(self, _battle):
        self.asked += 1
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


@pytest.fixture
def counts():
    saved = PolicyPlayer.guard_fire_counts.copy()
    PolicyPlayer.guard_fire_counts.clear()
    yield PolicyPlayer.guard_fire_counts
    PolicyPlayer.guard_fire_counts.clear()
    PolicyPlayer.guard_fire_counts.update(saved)


def test_the_player_asks_the_predictor_only_when_the_option_is_on(counts: Counter):
    runtime = _Runtime(("the forecast", ""))
    off: Any = SimpleNamespace(_opponent_forecaster=runtime)
    assert PolicyPlayer._reply_forecast(off, object()) is None
    assert runtime.asked == 0 and not counts
    on: Any = SimpleNamespace(_opponent_forecaster=runtime, exact_reply_forecast=True)
    assert PolicyPlayer._reply_forecast(on, object()) == "the forecast"
    assert runtime.asked == 1 and counts["reply_forecast"] == 1


def test_a_predictor_that_declines_or_fails_costs_the_search_nothing(counts: Counter):
    battle = object()
    declined: Any = SimpleNamespace(
        _opponent_forecaster=_Runtime((None, "stand_down:mid_turn")),
        exact_reply_forecast=True,
    )
    assert PolicyPlayer._reply_forecast(declined, battle) is None
    failing: Any = SimpleNamespace(
        _opponent_forecaster=_Runtime(RuntimeError("stream")), exact_reply_forecast=True
    )
    assert PolicyPlayer._reply_forecast(failing, battle) is None
    unloaded: Any = SimpleNamespace(exact_reply_forecast=True)
    assert PolicyPlayer._reply_forecast(unloaded, battle) is None
    assert counts == Counter(
        {
            "reply_forecast:stand_down:mid_turn": 1,
            "reply_forecast_failed:RuntimeError": 1,
            "reply_forecast:not_loaded": 1,
        }
    )


# --- the launcher --------------------------------------------------------------------


def test_the_ladder_flag_and_its_weight():
    assert SEARCH_FLAG_DEFAULTS["search_forecast_weight"] == 0.5
    assert SEARCH_FLAG_DEFAULTS["search_forecast_mix"] == "sum"
    args = build_parser().parse_args(
        ["--search", "--search-reply-prior", "forecast", "--opponent-forecast", "f.pt"]
    )
    assert (args.search_reply_prior, args.search_forecast_weight) == ("forecast", 0.5)
    assert args.search_forecast_mix == "sum"
    assert build_parser().parse_args([]).search_reply_prior == "models"


def test_the_forecast_is_configuration_once_the_search_reads_it(tmp_path):
    """Shadow mode's artifact is left out of a replay directory's configuration (it
    changes no decision). Read by the search, the artifact is recorded with its hash."""
    artifact = tmp_path / "forecast.pt"
    artifact.write_bytes(b"weights")
    base = [
        "--checkpoint",
        "c.zip",
        "--reg",
        "mc",
        "--opponent-forecast",
        str(artifact),
    ]
    shadow = material_config(build_parser().parse_args(base), "sha", "hard")
    assert not [key for key in shadow if "forecast" in key]
    args = build_parser().parse_args(
        [*base, "--search", "--search-reply-prior", "forecast"]
    )
    read = material_config(args, "sha", "hard")
    assert read["search_reply_prior"] == "forecast"
    assert read["search_reply_forecast"] == str(artifact)
    assert len(read["search_reply_forecast_sha256"]) == 64
    assert "search_forecast_weight" not in read  # at its default
    assert "search_forecast_mix" not in read
    measuring = build_parser().parse_args(
        [*base, "--search", "--search-reply-prior", "forecast"]
        + ["--search-forecast-weight", "0", "--search-forecast-mix", "product"]
    )
    recorded = material_config(measuring, "sha", "hard")
    assert recorded["search_forecast_weight"] == 0.0
    assert recorded["search_forecast_mix"] == "product"


def test_the_launch_line_says_when_the_search_reads_the_forecast():
    serving = SimpleNamespace(serving=True, load_failure=None)
    shadow = SimpleNamespace(_opponent_forecaster=serving)
    assert "no decision reads it" in forecast_status(shadow, "a.pt")
    read = SimpleNamespace(
        _opponent_forecaster=serving,
        exact_reply_forecast=True,
        exact_reply_forecast_weight=0.5,
        exact_reply_forecast_mix="sum",
    )
    line = forecast_status(read, "a.pt")
    assert "READ BY THE SEARCH" in line and "sum at weight 0.5" in line
    assert "no decision reads it" not in line
    measuring = SimpleNamespace(
        _opponent_forecaster=serving,
        exact_reply_forecast=True,
        exact_reply_forecast_weight=0.0,
    )
    line = forecast_status(measuring, "a.pt")
    assert "weight 0" in line and "brain's prior alone" in line
    assert "READ BY THE SEARCH" not in line
    broken = SimpleNamespace(
        _opponent_forecaster=SimpleNamespace(serving=False, load_failure="OSError"),
        exact_reply_forecast=True,
    )
    assert "NOT SERVING" in forecast_status(broken, "a.pt")
    assert "brain alone" in forecast_status(broken, "a.pt")
