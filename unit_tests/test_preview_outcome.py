from types import SimpleNamespace

import pytest
import torch

from vgc_bench.src.opponent_preview import PreviewPlan
from vgc_bench.src.policy_player import PolicyPlayer
from vgc_bench.src.preview_outcome import PreviewOutcomeNet, PreviewOutcomePredictor

OUR_TEAM = (
    "floetteeternal",
    "charizard",
    "whimsicott",
    "garchomp",
    "basculegion",
    "kingambit",
)
THEIR_TEAM = (
    "tyranitar",
    "excadrill",
    "primarina",
    "gholdengo",
    "sneasler",
    "gengar",
)


def test_outcome_ranker_stays_inside_supported_candidate_plans():
    vocab = {"<pad>": 0, **{name: i + 1 for i, name in enumerate(OUR_TEAM + THEIR_TEAM)}}
    model = PreviewOutcomeNet(len(vocab), embed_dim=4, hidden_dim=8)
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.0)
    predictor = PreviewOutcomePredictor(model, vocab, OUR_TEAM)
    candidates = [
        PreviewPlan((0, 1), (0, 1, 2, 3), 0.6),
        PreviewPlan((2, 3), (1, 2, 3, 4), 0.4),
    ]

    ranked = predictor.rank(OUR_TEAM, THEIR_TEAM, candidates=candidates)

    assert {item.plan for item in ranked} == set(candidates)
    assert all(item.win_probability == pytest.approx(0.5) for item in ranked)


def test_outcome_preview_falls_back_to_champion_when_sheet_is_open():
    PolicyPlayer.guard_fire_counts.clear()
    player = object.__new__(PolicyPlayer)
    player.use_outcome_teampreview = True
    player.preview_outcome_model_path = object()
    player._open_sheet_battles = {"battle-open"}
    battle = SimpleNamespace(battle_tag="battle-open")

    assert player._outcome_teampreview(battle) is None
    assert PolicyPlayer.guard_fire_counts["outcome_preview_open_sheet_fallback"] == 1


def test_outcome_ranker_rejects_another_fixed_team():
    vocab = {"<pad>": 0, **{name: i + 1 for i, name in enumerate(OUR_TEAM + THEIR_TEAM)}}
    predictor = PreviewOutcomePredictor(
        PreviewOutcomeNet(len(vocab), embed_dim=4, hidden_dim=8), vocab, OUR_TEAM
    )

    with pytest.raises(ValueError, match="specialized to another team"):
        predictor.rank(tuple(reversed(OUR_TEAM)), THEIR_TEAM)
