import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from types import SimpleNamespace

from vgc_bench.src.guards import Candidate
from vgc_bench.src.opponent_reranker import rerank_candidates
from vgc_bench.src.opponent_tactics import MovePrediction, SwitchPrediction


def test_reranker_stands_down_without_predictions():
    candidates = [Candidate((1, 1), 0.6), Candidate((2, 2), 0.4)]
    out, report = rerank_candidates(
        SimpleNamespace(), candidates, None, None, weight=1.0
    )
    assert out == candidates
    assert report is None


def test_reranker_ignores_switch_guesses_when_sets_are_hidden(monkeypatch):
    candidates = [Candidate((1, 1), 0.6), Candidate((2, 2), 0.4)]
    hidden_moves = (
        MovePrediction((("protect", 1.0),), (), (), reliability=0.0),
        MovePrediction((("protect", 1.0),), (), (), reliability=0.0),
    )
    switch_guesses = (
        SwitchPrediction(0.9, (("incineroar", 1.0),)),
        SwitchPrediction(0.9, (("rillaboom", 1.0),)),
    )
    monkeypatch.setattr(
        "vgc_bench.src.opponent_reranker._outgoing_damage", lambda *_args: 10.0
    )
    out, report = rerank_candidates(
        SimpleNamespace(), candidates, hidden_moves, switch_guesses, weight=100.0
    )
    assert out == candidates
    assert report is None


def test_reranker_never_promotes_demoted_candidate(monkeypatch):
    candidates = [
        Candidate((1, 1), 0.5),
        Candidate((2, 2), 0.4),
        Candidate((3, 3), 0.9, demoted_by="factual_guard"),
    ]
    utilities = {(1, 1): 0.0, (2, 2): 1.0}
    monkeypatch.setattr(
        "vgc_bench.src.opponent_reranker._outgoing_damage",
        lambda _battle, candidate, _switches, _cache: utilities[candidate.actions],
    )
    monkeypatch.setattr(
        "vgc_bench.src.opponent_reranker._incoming_damage", lambda *_args: 0.0
    )
    prediction = (
        MovePrediction((("protect", 1.0),), (), ()),
        MovePrediction((("protect", 1.0),), (), ()),
    )
    out, report = rerank_candidates(
        SimpleNamespace(), candidates, prediction, None, weight=1.0
    )
    assert out[0].actions == (2, 2)
    assert out[-1].actions == (3, 3)
    assert report is not None and report.changed


def test_reranker_respects_near_tie_boundary(monkeypatch):
    candidates = [Candidate((1, 1), 0.8), Candidate((2, 2), 0.1)]
    prediction = (
        MovePrediction((("protect", 1.0),), (), ()),
        MovePrediction((("protect", 1.0),), (), ()),
    )
    monkeypatch.setattr(
        "vgc_bench.src.opponent_reranker._outgoing_damage", lambda *_args: 10.0
    )
    monkeypatch.setattr(
        "vgc_bench.src.opponent_reranker._incoming_damage", lambda *_args: 0.0
    )
    out, report = rerank_candidates(
        SimpleNamespace(),
        candidates,
        prediction,
        None,
        weight=100.0,
        min_policy_ratio=0.3,
    )
    assert out == candidates
    assert report is None
