"""The paired-preview ledger must never raise inside a battle handler.

A serial search arm and an 8-way control arm do not reproduce each other's
matchup order exactly; when replay runs out of recorded choices for a
matchup, the player falls back to its own preview and the drift is counted
(observed 2026-09-05: seven RuntimeErrors froze seven battles of the search
re-gate).
"""

from evaluation.eval_counterfactual import PreviewLedger


def test_replay_returns_recorded_choices_in_order() -> None:
    ledger = PreviewLedger()
    key = ("a", "b", "__vs__", "c")
    ledger.record(key, "/team 1234")
    ledger.record(key, "/team 2341")
    ledger.begin_replay()
    assert ledger.next(key) == "/team 1234"
    assert ledger.next(key) == "/team 2341"
    assert ledger.replayed == 2


def test_drift_is_counted_not_raised() -> None:
    ledger = PreviewLedger()
    key = ("a", "b", "__vs__", "c")
    ledger.record(key, "/team 1234")
    ledger.begin_replay()
    assert ledger.next(key) == "/team 1234"
    assert ledger.next(key) is None  # exhausted: fall back, do not raise
    assert ledger.next(("never", "recorded")) is None
    assert ledger.mismatched == 2
