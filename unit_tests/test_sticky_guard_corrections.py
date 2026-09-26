"""keep_guard_correction (ladder 2026-09-26, game 6): the opponent/tempo reranker
runs after the guards and may reorder near-ties; a guard's promoted pair only ties
the pick it corrected, so the reranker put Leaf Storm back into a 4x-resisting
Archaludon three turns running. With sticky corrections on, the exact pair a guard
corrected away cannot come back on top; any other reranker choice stands."""

from __future__ import annotations

from vgc_bench.src.guards import Candidate
from vgc_bench.src.policy_player import keep_guard_correction

RAW = (15, 12)  # Psychic + Leaf Storm into the resisting slot
FIXED = (15, 13)  # the guard's twin: Leaf Storm into the other slot
OTHER = (15, 20)


def _cands(*order):
    return [Candidate(actions, 0.4) for actions in order]


def test_the_corrected_pair_is_put_back_on_top():
    out, kept = keep_guard_correction(_cands(RAW, FIXED, OTHER), RAW, FIXED, None)
    assert kept
    assert out[0].actions == FIXED
    assert [c.actions for c in out[1:]] == [RAW, OTHER]


def test_another_reranker_choice_stands():
    cands = _cands(OTHER, FIXED, RAW)
    out, kept = keep_guard_correction(cands, RAW, FIXED, None)
    assert not kept and out is cands


def test_the_predicted_ko_survival_pick_is_exempt():
    cands = _cands(RAW, FIXED)
    out, kept = keep_guard_correction(cands, RAW, FIXED, "predicted_ko_survival")
    assert not kept and out is cands


def test_nothing_to_keep_when_no_guard_changed_the_pick():
    cands = _cands(RAW, OTHER)
    out, kept = keep_guard_correction(cands, RAW, RAW, None)
    assert not kept and out is cands


def test_off_by_default():
    import inspect

    from vgc_bench.src.policy_player import PolicyPlayer

    default = (
        inspect.signature(PolicyPlayer.__init__)
        .parameters["sticky_guard_corrections"]
        .default
    )
    assert default is False
