import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from pathlib import Path

import torch

from vgc_bench.src.opponent_tactics import (
    MoveNet,
    MovePredictor,
    SwitchNet,
    SwitchPredictor,
    parse_move_examples,
    parse_switch_examples,
)


def test_parser_separates_voluntary_forced_and_pivot_switches():
    log = """|player|p1|Top|avatar|1700
|player|p2|Foe|avatar|1700
|poke|p1|Charizard, L50|
|poke|p1|Floette, L50|
|poke|p1|Garchomp, L50|
|poke|p1|Kingambit, L50|
|poke|p1|Whimsicott, L50|
|poke|p1|Basculegion, L50|
|poke|p2|Sylveon, L50|
|poke|p2|Raichu, L50|
|poke|p2|Incineroar, L50|
|poke|p2|Pelipper, L50|
|poke|p2|Sneasler, L50|
|poke|p2|Farigiraf, L50|
|start
|switch|p1a: Charizard|Charizard, L50|100/100
|switch|p1b: Floette|Floette, L50|100/100
|switch|p2a: Sylveon|Sylveon, L50|100/100
|switch|p2b: Raichu|Raichu, L50|100/100
|turn|1
|switch|p2a: Incineroar|Incineroar, L50|100/100
|move|p1a: Charizard|Heat Wave|p2a: Incineroar
|move|p1b: Floette|Moonblast|p2b: Raichu
|move|p2b: Raichu|Volt Switch|p1a: Charizard
|switch|p2b: Pelipper|Pelipper, L50|100/100
|faint|p1a: Charizard
|switch|p1a: Garchomp|Garchomp, L50|100/100
|turn|2
|move|p1a: Garchomp|Earthquake|p2a: Incineroar
|move|p1b: Floette|Moonblast|p2b: Pelipper
|move|p2a: Incineroar|Knock Off|p1b: Floette
|move|p2b: Pelipper|Hurricane|p1b: Floette
"""
    examples = parse_switch_examples("gen9championsvgc2026regmb-test", log)
    turn_one_p2 = [e for e in examples if e.turn == 1 and e.rating == 1700][2:]
    assert turn_one_p2[0].switch_to == "incineroar"
    assert turn_one_p2[1].switch_to is None  # Volt Switch is not a chosen switch.
    turn_two_p1 = [e for e in examples if e.turn == 2][:2]
    assert turn_two_p1[0].active[0] == "garchomp"  # forced replacement was tracked
    moves = parse_move_examples("gen9championsvgc2026regmb-test", log)
    heat_wave = next(example for example in moves if example.move_id == "heatwave")
    assert heat_wave.active == ("charizard", "floette")
    assert heat_wave.opponent_active == ("sylveon", "raichu")
    assert heat_wave.target_class == 2


def test_predictor_returns_normalized_targets(tmp_path: Path):
    vocab = {"<unknown>": 0, **{f"mon{i}": i + 1 for i in range(12)}}
    model = SwitchNet(len(vocab), embed_dim=8, hidden_dim=16)
    path = tmp_path / "switch.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "config": {"embed_dim": 8, "hidden_dim": 16},
        },
        path,
    )
    predictor = SwitchPredictor.load(path)
    result = predictor.predict(
        [f"mon{i}" for i in range(6)],
        [f"mon{i}" for i in range(6, 12)],
        ["mon0", "mon1"],
        ["mon6", "mon7"],
        [1.0, 0.5, 1.0, 1.0],
        actor_slot=0,
        turn=3,
        bring_marginals={f"mon{i}": 1.0 for i in range(6)},
    )
    assert 0 <= result.switch_probability <= 1
    assert abs(sum(probability for _, probability in result.targets) - 1) < 1e-6
    assert dict(result.targets)["mon0"] < 1e-6


def test_move_predictor_masks_to_available_moves(tmp_path: Path):
    species_vocab = {"<unknown>": 0, **{f"mon{i}": i + 1 for i in range(12)}}
    move_vocab = {"<unknown>": 0, "protect": 1, "tailwind": 2, "dragonclaw": 3}
    move_ids = [""] * len(move_vocab)
    for move_id, index in move_vocab.items():
        move_ids[index] = move_id
    model = MoveNet(len(species_vocab), move_ids, embed_dim=8, hidden_dim=16)
    path = tmp_path / "moves.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "species_vocab": species_vocab,
            "move_vocab": move_vocab,
            "config": {"embed_dim": 8, "hidden_dim": 16},
        },
        path,
    )
    predictor = MovePredictor.load(path)
    result = predictor.predict(
        [f"mon{i}" for i in range(6)],
        [f"mon{i}" for i in range(6, 12)],
        ["mon0", "mon1"],
        ["mon6", "mon7"],
        [1.0, 1.0, 1.0, 1.0],
        actor_slot=0,
        turn=1,
        available_moves=["protect", "tailwind"],
    )
    assert {move for move, _ in result.moves} == {"protect", "tailwind"}
    assert abs(sum(probability for _, probability in result.moves) - 1) < 1e-6
    assert abs(sum(probability for _, _, probability in result.actions) - 1) < 1e-6
