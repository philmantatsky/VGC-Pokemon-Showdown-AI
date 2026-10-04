import numpy as np
import torch

from vgczero import data as D
from vgczero.model import ModelConfig, VGCNet, to_torch


def test_vecenv_random_play_is_legal():
    env = D.E.VecEnv(32, seed=11)
    for _ in range(60):
        a = np.zeros((32, 2, 3), np.uint8)
        for s in range(2):
            a[:, s] = np.asarray(env.baseline_actions(s, "random", 3))
        _, _, illegal = env.step(a)
        assert illegal == 0


def test_act_and_evaluate_agree():
    torch.manual_seed(0)
    m = VGCNet(ModelConfig.preset("tiny"))
    env = D.E.VecEnv(16, seed=2)
    for _ in range(12):
        o = to_torch(*env.observe(), device="cpu")
        a, lp, _ = m.act(o)
        lp2, ent, _ = m.evaluate(o, a)
        assert torch.allclose(lp, lp2, atol=1e-4)
        assert torch.all(ent >= -1e-5)
        _, _, illegal = env.step(a.reshape(16, 2, 3).numpy().astype(np.uint8))
        assert illegal == 0


def test_top_joint_is_legal():
    m = VGCNet(ModelConfig.preset("tiny"))
    env = D.E.VecEnv(8, seed=5)
    for _ in range(8):
        obs = env.observe()
        o = to_torch(*obs, device="cpu")
        ch, pr = m.top_joint(o, 8)
        mask = o["mask"]
        for b in range(ch.shape[0]):
            if o["req"][b, 0] in (D.REQ_MOVE, D.REQ_SWITCH):
                for k in range(ch.shape[1]):
                    if pr[b, k] > 1e-6:
                        a0, a1 = int(ch[b, k, 1]), int(ch[b, k, 2])
                        assert mask[b, 0, a0] and mask[b, 1, a1]
                        if 1 <= a0 <= 6:
                            assert a0 != a1
        a, _, _ = m.act(o)
        env.step(a.reshape(8, 2, 3).numpy().astype(np.uint8))


def test_determinize_keeps_viewer_observation():
    """Sampled worlds must look identical to the viewer (no hidden-info leak)."""
    env = D.E.VecEnv(16, seed=9, open_sheet_prob=0.3)
    checked = 0
    for step in range(30):
        a = np.zeros((16, 2, 3), np.uint8)
        for s in range(2):
            a[:, s] = np.asarray(env.baseline_actions(s, "greedy", step))
        env.step(a)
        if step % 3:
            continue
        for e in range(16):
            b = env.battle(e)
            for viewer in (0, 1):
                o1 = b.observe(viewer)
                for seed in range(3):
                    o2 = b.determinize(viewer, 1000 * step + 10 * e + seed).observe(viewer)
                    for x, y in zip(o1, o2):
                        assert np.array_equal(np.asarray(x), np.asarray(y))
                    checked += 1
    assert checked > 500
