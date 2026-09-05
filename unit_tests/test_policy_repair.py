"""Decision paths must never assert on the policy object (poke-env swallows the
exception and the battle stalls forever). ``PolicyPlayer._repair_policy`` unwraps
a wrapped policy in place, or counts the event so the caller can play a safe
default order. Observed 2026-09-05: the search re-gate's forced-switch request
tripped a bare isinstance assert and stalled the serial arm.
"""

from types import SimpleNamespace
from unittest.mock import create_autospec

from vgc_bench.src.policy import MaskedActorCriticPolicy
from vgc_bench.src.policy_player import PolicyPlayer


def _stub(policy):
    return SimpleNamespace(policy=policy, username="stub")


def test_real_policy_is_left_alone() -> None:
    real = create_autospec(MaskedActorCriticPolicy, instance=True)
    stub = _stub(real)
    PolicyPlayer._repair_policy(stub)
    assert stub.policy is real


def test_wrapped_policy_is_unwrapped_in_place() -> None:
    PolicyPlayer.guard_fire_counts.clear()
    real = create_autospec(MaskedActorCriticPolicy, instance=True)
    stub = _stub(SimpleNamespace(policy=real))
    PolicyPlayer._repair_policy(stub)
    assert stub.policy is real
    assert PolicyPlayer.guard_fire_counts["policy_unwrapped"] == 1


def test_unusable_policy_is_counted_not_raised() -> None:
    PolicyPlayer.guard_fire_counts.clear()
    PolicyPlayer._policy_type_reported = False
    stub = _stub(None)
    PolicyPlayer._repair_policy(stub)  # must not raise
    assert stub.policy is None
    assert PolicyPlayer.guard_fire_counts["policy_unavailable"] == 1


def test_decision_waits_for_a_loading_policy() -> None:
    """The construction-order race: a decision arrives while set_policy is
    still inside PPO.load. The decision must wait, not play a default order."""
    import threading
    import time

    PolicyPlayer.guard_fire_counts.clear()
    real = create_autospec(MaskedActorCriticPolicy, instance=True)
    stub = _stub(None)
    stub._policy_loading = True

    def finish_loading() -> None:
        time.sleep(0.2)
        stub.policy = real

    threading.Thread(target=finish_loading, daemon=True).start()
    old = PolicyPlayer.policy_wait_s
    PolicyPlayer.policy_wait_s = 5.0
    try:
        PolicyPlayer._repair_policy(stub)
    finally:
        PolicyPlayer.policy_wait_s = old
    assert stub.policy is real
    assert PolicyPlayer.guard_fire_counts["policy_waited"] == 1
    assert PolicyPlayer.guard_fire_counts["policy_unavailable"] == 0
