"""Mixed-strategy play (2026-09-06). A player that never mixes is maximally
predictable in a simultaneous-move game; the exploiter's 60% and the 1300+
opening punishes are the symptom. The switch must be off by default, gated to
the opening when asked, never play a guard-demoted or strategic-only pair,
count everything it does, and never raise (poke-env swallows exceptions and
the battle stalls forever)."""

import threading
from types import SimpleNamespace

import numpy as np

from vgc_bench.src.guards import Candidate
from vgc_bench.src.policy_player import PolicyPlayer


def _stub(mode="opening", k=3, temperature=1.0, last_turn=2, seed=0):
    return SimpleNamespace(
        mixing_mode=mode,
        mixing_top_k=k,
        mixing_temperature=temperature,
        mixing_last_turn=last_turn,
        _mixing_rng=np.random.default_rng(seed),
        _mixing_lock=threading.Lock(),
    )


def _battle(turn=1, teampreview=False):
    return SimpleNamespace(turn=turn, teampreview=teampreview)


def _cands(probs, demoted=()):
    return [
        Candidate(actions=(i, i), prob=p, demoted_by=("rule" if i in demoted else None))
        for i, p in enumerate(probs)
    ]


def test_off_mode_leaves_ranking_untouched() -> None:
    PolicyPlayer.guard_fire_counts.clear()
    cands = _cands([0.5, 0.3, 0.2])
    out, report = PolicyPlayer._apply_mixing(_stub(mode="off"), _battle(), cands)
    assert out is cands and report is None
    assert not any(k.startswith("mixing") for k in PolicyPlayer.guard_fire_counts)


def test_opening_mode_is_turn_gated() -> None:
    stub = _stub(mode="opening", last_turn=2)
    assert PolicyPlayer._mixing_active(stub, _battle(turn=0, teampreview=True))
    assert PolicyPlayer._mixing_active(stub, _battle(turn=2))
    assert not PolicyPlayer._mixing_active(stub, _battle(turn=3))
    assert PolicyPlayer._mixing_active(_stub(mode="always"), _battle(turn=9))
    out, report = PolicyPlayer._apply_mixing(stub, _battle(turn=7), _cands([0.6, 0.4]))
    assert report is None and out[0].prob == 0.6


def test_samples_by_policy_probability_and_rotates_choice_front() -> None:
    PolicyPlayer.guard_fire_counts.clear()
    stub = _stub(seed=1)
    cands = _cands([0.5, 0.3, 0.2, 0.1])
    picks = []
    for _ in range(2000):
        out, report = PolicyPlayer._apply_mixing(stub, _battle(), cands)
        assert report is not None
        assert out[0] is cands[report["chosen_rank"]]
        assert len(out) == len(cands)
        assert {id(c) for c in out} == {id(c) for c in cands}
        picks.append(report["chosen_rank"])
    freq = np.bincount(picks, minlength=4) / len(picks)
    # k=3: the fourth pair is never played; weights 0.5/0.3/0.2 renormalised
    assert freq[3] == 0
    assert abs(freq[0] - 0.5) < 0.05
    assert abs(freq[1] - 0.3) < 0.05
    assert abs(freq[2] - 0.2) < 0.05
    assert PolicyPlayer.guard_fire_counts["mixing_ran"] == 2000
    assert 0 < PolicyPlayer.guard_fire_counts["mixing_changed_pick"] < 2000


def test_demoted_and_strategic_only_candidates_are_never_sampled() -> None:
    PolicyPlayer.guard_fire_counts.clear()
    stub = _stub(seed=2)
    cands = _cands([0.4, 0.35, 0.25], demoted={1})
    cands[2] = Candidate(actions=(2, 2), prob=0.25, strategic_only=True)
    for _ in range(200):
        out, report = PolicyPlayer._apply_mixing(stub, _battle(), cands)
        assert report is None and out is cands  # one eligible pair: nothing to mix
    assert PolicyPlayer.guard_fire_counts["mixing_skipped:single_candidate"] == 200
    # a demoted pair inside the top-k is skipped, the rest still mix
    cands = _cands([0.4, 0.35, 0.25], demoted={0})
    ranks = set()
    for _ in range(300):
        _, report = PolicyPlayer._apply_mixing(stub, _battle(), cands)
        assert report is not None
        ranks.add(report["chosen_rank"])
    assert ranks == {1, 2}


def test_bad_temperature_and_internal_errors_count_not_raise() -> None:
    PolicyPlayer.guard_fire_counts.clear()
    out, report = PolicyPlayer._apply_mixing(
        _stub(temperature=0.0), _battle(), _cands([0.6, 0.4])
    )
    assert report is None
    assert PolicyPlayer.guard_fire_counts["mixing_skipped:bad_temperature"] == 1
    broken = _stub()
    broken._mixing_rng = None
    cands = _cands([0.6, 0.4])
    out, report = PolicyPlayer._apply_mixing(broken, _battle(), cands)
    assert report is None and out is cands
    assert any(k.startswith("mixing_error:") for k in PolicyPlayer.guard_fire_counts)


def test_low_temperature_collapses_to_top_pick() -> None:
    stub = _stub(temperature=0.02, seed=3)
    cands = _cands([0.5, 0.3, 0.2])
    ranks = {
        PolicyPlayer._apply_mixing(stub, _battle(), cands)[1]["chosen_rank"]
        for _ in range(300)
    }
    assert ranks == {0}


def test_constructor_rejects_unknown_mode() -> None:
    import pytest

    with pytest.raises(ValueError):
        PolicyPlayer.__init__(SimpleNamespace(), mixing_mode="sometimes")
