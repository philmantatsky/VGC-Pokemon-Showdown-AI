import numpy as np

from vgczero.ppo import Rollout, compute_gae


def make(T, N=1):
    ro = Rollout.__new__(Rollout)
    ro.T, ro.N = T, N
    ro.value = np.zeros((T, N, 2), np.float32)
    ro.valid = np.zeros((T, N, 2), bool)
    ro.reward = np.zeros((T, N, 2), np.float32)
    ro.done = np.zeros((T, N), bool)
    return ro


def test_terminal_reward_reaches_every_decision():
    ro = make(3)
    ro.valid[:, 0, 0] = True
    ro.done[2, 0] = True
    ro.reward[2, 0] = [1.0, -1.0]
    adv, ret = compute_gae(ro, np.zeros((1, 2)), gamma=1.0, lam=0.95)
    assert np.allclose(adv[:, 0, 0], [0.95**2, 0.95, 1.0])


def test_waiting_steps_are_skipped():
    # Side 1 decides only at t=0 then waits; the loss at t=2 is credited to t=0.
    ro = make(3)
    ro.valid[:, 0, 0] = True
    ro.valid[0, 0, 1] = True
    ro.done[2, 0] = True
    ro.reward[2, 0] = [1.0, -1.0]
    adv, _ = compute_gae(ro, np.zeros((1, 2)), 1.0, 0.95)
    assert np.isclose(adv[0, 0, 1], -1.0)
    assert np.all(adv[1:, 0, 1] == 0)


def test_episode_boundary_and_bootstrap():
    ro = make(4)
    ro.valid[:, 0, 0] = True
    ro.value[:, 0, 0] = 0.5
    ro.done[1, 0] = True
    ro.reward[1, 0] = [-1.0, 1.0]
    adv, _ = compute_gae(ro, np.full((1, 2), 0.25), 1.0, 1.0)
    # First episode: returns -1 at t=0 and t=1.
    assert np.allclose(adv[:2, 0, 0], [-1.5, -1.5])
    # Second episode bootstraps from the final value 0.25.
    assert np.allclose(adv[2:, 0, 0], [-0.25, -0.25])
