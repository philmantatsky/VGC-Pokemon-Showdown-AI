"""Stage-C.1: the prior-based reliability floor for opening turns.

The opponent/tempo layers scale their soft evidence by prediction reliability,
which was revealed-moves-only (0 on turn 1 of hidden-sheet games -- a complete
no-op on exactly the turns that decide the format). The floor lets the
evidence-conditioned set posterior lend BOUNDED reliability. Default off:
VGC_PRIOR_RELIABILITY_FLOOR unset must reproduce historical behavior exactly.
"""

import os

from vgc_bench.src.policy_player import PolicyPlayer


class TestFlooredReliability:
    def test_zero_cap_is_historical_behavior(self):
        prior = {"moves": ["trickroom"], "prob": 0.97}
        assert PolicyPlayer._floored_reliability(0.0, prior, 0.0) == 0.0
        assert PolicyPlayer._floored_reliability(0.25, prior, 0.0) == 0.25
        assert PolicyPlayer._floored_reliability(1.0, prior, 0.0) == 1.0

    def test_concentrated_prior_lifts_to_cap(self):
        prior = {"moves": ["trickroom"], "prob": 0.97}
        assert PolicyPlayer._floored_reliability(0.0, prior, 0.35) == 0.35

    def test_diffuse_prior_lends_only_its_posterior(self):
        prior = {"moves": ["protect"], "prob": 0.12}
        assert PolicyPlayer._floored_reliability(0.0, prior, 0.35) == 0.12

    def test_revealed_evidence_always_wins(self):
        prior = {"moves": [], "prob": 0.05}
        assert PolicyPlayer._floored_reliability(0.75, prior, 0.35) == 0.75
        assert PolicyPlayer._floored_reliability(1.0, prior, 0.35) == 1.0

    def test_no_prior_means_no_floor(self):
        assert PolicyPlayer._floored_reliability(0.0, {}, 0.35) == 0.0

    def test_missing_prob_key_treated_as_zero(self):
        assert PolicyPlayer._floored_reliability(0.0, {"moves": ["x"]}, 0.35) == 0.0


class TestFloorCapEnv:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("VGC_PRIOR_RELIABILITY_FLOOR", raising=False)
        assert PolicyPlayer._reliability_floor_cap() == 0.0

    def test_env_value_parsed(self, monkeypatch):
        monkeypatch.setenv("VGC_PRIOR_RELIABILITY_FLOOR", "0.35")
        assert PolicyPlayer._reliability_floor_cap() == 0.35

    def test_garbage_env_value_is_off(self, monkeypatch):
        monkeypatch.setenv("VGC_PRIOR_RELIABILITY_FLOOR", "banana")
        assert PolicyPlayer._reliability_floor_cap() == 0.0


class TestMovesetPriorPosterior:
    def test_prob_present_and_concentrated_for_dedicated_setter(self):
        os.environ["VGC_KNOWLEDGE_OBS"] = "1"

        class FakeMon:
            base_species = "farigiraf"
            moves: dict = {}
            item = None
            ability = None

        PolicyPlayer.use_moveset_prior = True
        try:
            prior = PolicyPlayer._moveset_prior(FakeMon())  # type: ignore[arg-type]
            assert prior is not None
            assert "prob" in prior
            assert 0.0 <= prior["prob"] <= 1.0
        finally:
            PolicyPlayer.use_moveset_prior = None
