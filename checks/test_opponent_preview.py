import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from pathlib import Path

import torch

from vgc_bench.src.opponent_preview import (
    OpponentBelief,
    PreviewNet,
    PreviewPlan,
    PreviewPredictor,
    parse_preview_examples,
    plan_to_showdown_order,
)

LOG = """|gametype|doubles
|poke|p1|Charizard, L50|
|poke|p1|Garchomp, L50|
|poke|p1|Whimsicott, L50|
|poke|p1|Kingambit, L50|
|poke|p1|Basculegion, L50|
|poke|p1|Floette-Eternal, L50|
|poke|p2|Pelipper, L50|
|poke|p2|Archaludon, L50|
|poke|p2|Sinistcha, L50|
|poke|p2|Swampert, L50|
|poke|p2|Grimmsnarl, L50|
|poke|p2|Sneasler, L50|
|teampreview|4
|start
|switch|p1a: Zard|Charizard, L50|100/100
|switch|p1b: Cotton|Whimsicott, L50|100/100
|switch|p2a: Rain|Pelipper, L50|100/100
|switch|p2b: Bridge|Archaludon, L50|100/100
|turn|1
|switch|p1b: Chomp|Garchomp, L50|100/100
|switch|p2b: Tea|Sinistcha, L50|100/100
|turn|2
|switch|p1a: Gambit|Kingambit, L50|100/100
|switch|p2a: Pert|Swampert, L50|100/100
|turn|3
|win|p1
"""


def test_parser_extracts_exact_lead_and_bring():
    examples = parse_preview_examples("battle-1", LOG)
    assert len(examples) == 2
    assert examples[0].lead == (0, 2)
    assert examples[0].bring == (0, 1, 2, 3)
    assert examples[1].lead == (0, 1)
    assert examples[1].bring == (0, 1, 2, 3)


def test_predicted_plans_are_coherent(tmp_path: Path):
    vocab = {"<unknown>": 0, **{f"mon{i}": i + 1 for i in range(12)}}
    model = PreviewNet(len(vocab), embed_dim=8, hidden_dim=16)
    path = tmp_path / "preview.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "config": {"embed_dim": 8, "hidden_dim": 16},
        },
        path,
    )
    predictor = PreviewPredictor.load(path)
    plans = predictor.predict_plans(
        [f"mon{i}" for i in range(6)], [f"mon{i}" for i in range(6, 12)]
    )
    assert plans
    assert abs(sum(plan.probability for plan in plans) - 1.0) < 1e-6
    for plan in plans:
        assert set(plan.lead_indices).issubset(plan.bring_indices)
        assert len(set(plan_to_showdown_order(plan))) == 4


def test_belief_filters_on_reveals_and_leads():
    plans = [
        PreviewPlan((0, 1), (0, 1, 2, 3), 0.6),
        PreviewPlan((0, 2), (0, 2, 4, 5), 0.4),
    ]
    belief = OpponentBelief(tuple(f"mon{i}" for i in range(6)), plans)
    belief.observe(["mon4"], ["mon0", "mon2"])
    assert len(belief.plans) == 1
    assert belief.plans[0].bring_indices == (0, 2, 4, 5)
    assert belief.plans[0].probability == 1.0
