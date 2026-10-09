"""Which opponent model ranks the replies in the search (2026-10-09).

The search's opponent prior (``OpponentModelPrior``) blends the brain's own prior with
the opponent move / switch models whenever the player has them. The local search
measurements of 2026-10-04 to 10-06 (the head-to-heads, the roster run) played with
players that carry no such models: their search ranked replies with the brain alone.
The ladder bot does carry them -- they are the reranker's -- so the ladder trial
launched on 10-09 gave the search a reply prior no local run had measured (0.6 of the
log weight on those models with hidden sheets, 0.4 with open ones). Found by reading
the code while the trial's first games played.

``PolicyPlayer(exact_opponent_models=False)`` / ``--search-reply-prior brain`` keeps
them out of the search; the reranker still reads them.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ladder_ourteam import SEARCH_FLAG_DEFAULTS, build_parser
from vgc_bench.src import live_exact
from vgc_bench.src.policy_player import PolicyPlayer


class _Recorder:
    made: list[dict] = []

    def __init__(self, **kwargs: Any):
        _Recorder.made.append(kwargs)


def _player(tmp_path, **attributes: Any) -> Any:
    team = tmp_path / "team.txt"
    team.write_text("Blastoise @ Blastoisinite\n")
    return SimpleNamespace(
        _exact_sessions={},
        exact_team_path=team,
        outcome_value_path=tmp_path / "outcome.zip",
        policy=SimpleNamespace(device="cpu"),
        residual_ranker_path=None,
        _residual_ranker=None,
        opponent_move_predictions=lambda battle: None,
        opponent_switch_predictions=lambda battle: None,
        format="gen9championsvgc2026regmc",
        _open_sheet_battles=set(),
        _outcome_evaluator=None,
        _preview_predictor=None,
        _move_predictor="the reranker's move model",
        _switch_predictor="the reranker's switch model",
        exact_search_config=None,
        exact_max_determinizations=8,
        exact_search_determinizations=4,
        exact_min_deep_coverage=0.5,
        exact_selective_search=False,
        exact_enable_ponder=False,
        exact_ponder_config=None,
        _exact_policy_lock=None,
        exact_leaf="critic",
        exact_oracle_opponent_team=None,
        **attributes,
    )


@pytest.fixture
def sessions(monkeypatch):
    _Recorder.made = []
    monkeypatch.setattr(live_exact, "LiveExactSession", _Recorder)
    return _Recorder.made


def _models(made: list[dict]) -> tuple[Any, Any]:
    return made[-1]["move_predictor"], made[-1]["switch_predictor"]


def test_the_search_reads_the_players_opponent_models_unless_told_not_to(
    tmp_path, sessions
):
    battle: Any = SimpleNamespace(battle_tag="battle-1")
    # as before the option existed (a player built without it), and with it on
    for player in (_player(tmp_path), _player(tmp_path, exact_opponent_models=True)):
        PolicyPlayer._live_exact_session(player, battle)
        assert _models(sessions) == (
            "the reranker's move model",
            "the reranker's switch model",
        )
    # off: the search ranks replies with the brain alone ...
    player = _player(tmp_path, exact_opponent_models=False)
    session = PolicyPlayer._live_exact_session(player, battle)
    assert _models(sessions) == (None, None)
    # ... the player keeps its models (the reranker's), and the session is kept
    assert player._move_predictor == "the reranker's move model"
    assert PolicyPlayer._live_exact_session(player, battle) is session
    assert len(sessions) == 3


def test_the_ladder_flag_defaults_to_what_ladder_searches_played_before():
    assert SEARCH_FLAG_DEFAULTS["search_reply_prior"] == "models"
    assert build_parser().parse_args([]).search_reply_prior == "models"
    args = build_parser().parse_args(["--search", "--search-reply-prior", "brain"])
    assert args.search_reply_prior == "brain"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--search-reply-prior", "predictor"])
