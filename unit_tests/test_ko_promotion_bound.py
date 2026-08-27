"""Stage-C.2: robust KO bound for the promoting guards under fabricated stats.

When an opponent denies team sheets, ensure_stats synthesizes an offense-heavy,
zero-defense spread; min-roll "guaranteed" KOs computed against it can promote
moves that do not actually KO a real bulky set. The bound scales those claims
to the worst plausible defensive spread. Mode defaults to "off" (historical
behavior) until the realized-KO counting validation promotes it.
"""

from poke_env.battle import Move, Pokemon

from vgc_bench.src.guards import _ko_promotion_mode
from vgc_bench.src.vgc_knowledge import (
    CHAMPIONS_EV_CAP,
    ensure_stats,
    robust_ko_scale,
    stats_were_synthesized,
)


def _mon(species: str) -> Pokemon:
    return Pokemon(gen=9, species=species)


class TestSynthesisDetection:
    def test_from_absent_synthesis_detected(self):
        mon = _mon("farigiraf")
        assert mon.stats is None or any(v is None for v in mon.stats.values())
        assert ensure_stats(mon) is True
        assert stats_were_synthesized(mon) is True

    def test_real_looking_stats_not_flagged(self):
        mon = _mon("farigiraf")
        ensure_stats(mon)
        assert mon.stats is not None
        mon.stats["def"] = mon.stats["def"] + 30  # a genuinely bulky spread
        assert stats_were_synthesized(mon) is False

    def test_missing_stats_not_flagged(self):
        assert stats_were_synthesized(_mon("farigiraf")) is False


class TestRobustScale:
    def test_scale_strictly_below_one(self):
        mon = _mon("farigiraf")
        ensure_stats(mon)
        physical = Move("rockslide", gen=9)
        special = Move("heatwave", gen=9)
        for move in (physical, special):
            scale = robust_ko_scale(mon, move)
            assert 0.0 < scale < 1.0

    def test_scale_matches_formula(self):
        mon = _mon("farigiraf")
        ensure_stats(mon)
        move = Move("rockslide", gen=9)  # physical -> def
        base = mon.base_stats["def"]
        expected = (base + 20) / ((base + CHAMPIONS_EV_CAP + 20) * 1.1)
        assert abs(robust_ko_scale(mon, move) - expected) < 1e-9


class TestModeEnv:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("VGC_KO_PROMOTION_MODE", raising=False)
        assert _ko_promotion_mode() == "off"

    def test_modes_parse(self, monkeypatch):
        for mode in ("robust", "skip", "off"):
            monkeypatch.setenv("VGC_KO_PROMOTION_MODE", mode)
            assert _ko_promotion_mode() == mode

    def test_garbage_is_off(self, monkeypatch):
        monkeypatch.setenv("VGC_KO_PROMOTION_MODE", "banana")
        assert _ko_promotion_mode() == "off"
