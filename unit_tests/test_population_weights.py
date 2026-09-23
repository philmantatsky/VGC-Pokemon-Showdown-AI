from pathlib import Path

import pytest

from vgc_bench.src.callback import fixed_population_weights


@pytest.mark.parametrize("lineage_count", [1, 8, 100])
def test_human_mass_does_not_decay(lineage_count):
    paths = [Path(f"{s}.zip") for s in [100, 200, *range(1000, 1000 + lineage_count)]]
    weights = fixed_population_weights(paths, {100, 200}, 0.2)
    assert sum(weights) == pytest.approx(1)
    assert sum(weights[:2]) == pytest.approx(0.2)


def test_missing_human_opponent_refuses_silent_reweighting():
    with pytest.raises(ValueError):
        fixed_population_weights([Path("100.zip"), Path("1000.zip")], {100, 200}, 0.2)
