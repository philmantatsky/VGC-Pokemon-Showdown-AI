"""The shipped opponent models behind the predictor contract (oppmodel.legacy).

Two tiny networks with random weights stand in for the shipped files; batches
are written inline with real dex ids. One test reads the shipped files and a
slice of the built dataset and skips when the dataset is absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from vgc_bench.src.oppmodel import features as F
from vgc_bench.src.oppmodel import legacy as L
from vgc_bench.src.opponent_tactics import (
    MoveNet,
    MovePredictor,
    SwitchNet,
    SwitchPredictor,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "results_oppmodel" / "v1_ondisk"
SHIPPED_MOVE = ROOT / "data" / "opponent_move_top500_regmc.pt"
SHIPPED_SWITCH = ROOT / "data" / "opponent_switch_top500_regmc.pt"

C = 4  # candidates per slot in the synthetic batches
OTHER = F.other_index(C)
ACTOR = ("garchomp", "sneasler", "politoed", "archaludon", "indeedee", "pawmot")
FOE = ("incineroar", "rillaboom", "charizard", "farigiraf", "torkoal", "venusaur")
STRANGER = "pawmot"  # in the dex, not in the old models' species vocabulary
MOVES = ("earthquake", "protect", "dragonclaw", "closecombat", "fakeout")
NEW_MOVE = "stompingtantrum"  # in the dex, not in the old move vocabulary
ATTACK = F.CAND_DAMAGING | F.CAND_AIMED
SPREAD = F.CAND_DAMAGING | F.CAND_SPREAD
GUARD = F.CAND_PROTECT
AIM = (1 << F.T_FOE_A) | (1 << F.T_FOE_B) | (1 << F.T_ALLY)
AUTO = 1 << F.T_AUTO
CALIBRATION = {"upper_bounds": [0.3, 0.5, 0.7, 1.0], "rates": [0.05, 0.1, 0.2, 0.4]}


# --- fixtures -------------------------------------------------------------------


@pytest.fixture(scope="module")
def fz() -> F.Featurizer:
    return F.Featurizer.build()


def tiny_models(seed: int = 0) -> tuple[MovePredictor, SwitchPredictor]:
    """Random small networks with the shipped classes and a 5-move vocabulary."""
    torch.manual_seed(seed)
    known = [name for name in (*ACTOR, *FOE) if name != STRANGER]
    species = {"<unknown>": 0, **{name: i + 1 for i, name in enumerate(known)}}
    moves = {"<unknown>": 0, **{name: i + 1 for i, name in enumerate(MOVES)}}
    move_net = MoveNet(len(species), ["<unknown>", *MOVES], 8, 16)
    switch_net = SwitchNet(len(species), 8, 16)
    return (
        MovePredictor(move_net, species, moves),
        SwitchPredictor(switch_net, dict(species), dict(CALIBRATION)),
    )


@pytest.fixture()
def legacy(fz: F.Featurizer) -> L.LegacyPredictor:
    move, switch = tiny_models()
    return L.LegacyPredictor(move, switch, fz)


def species_id(fz: F.Featurizer, name: str) -> int:
    return fz.tables.species_ids.index(name)


def move_id(fz: F.Featurizer, name: str) -> int:
    return fz.tables.move_ids.index(name)


def scene(fz: F.Featurizer) -> F.Batch:
    """Three examples, features only.

    0: turn 3, four active. Slot a (roster 0) has a spread move, a Protect, an
       aimed attack and an aimed attack the old vocabulary lacks; it can still
       Mega-evolve. Slot b (roster 1) has two aimed attacks and a Protect.
       Roster 2 and 3 are legal switch destinations.
    1: turn 5, only slot a (roster 2) and foe a stand; no legal switch.
    2: turn 2, slot a is the species the old models do not know.
    """
    batch = {
        name: np.zeros((3, *spec.shape), dtype=spec.dtype)
        for name, spec in F.layout(C).items()
        if spec.group == "feature"
    }
    batch["act_mon"][:] = -1
    batch["foe_mon"][:] = -1
    for row, name in enumerate((*ACTOR, *FOE)):
        batch["mon_id"][:, row, F.ID_SPECIES] = species_id(fz, name)
        batch["mon_flag"][:, row, F.FLAG_PRESENT] = 1
    batch["mon_hp"][:] = 1.0

    def slot(i: int, s: int, roster: int, cands: tuple[tuple[str, int, int], ...]):
        batch["act_mon"][i, s] = roster
        for column, (name, bits, targets) in enumerate(cands):
            batch["cand_move"][i, s, column] = move_id(fz, name)
            batch["cand_flag"][i, s, column] = F.CAND_VALID | bits
            batch["cand_tmask"][i, s, column] = targets
            batch["action_mask"][i, s, column] = 1
        batch["action_mask"][i, s, OTHER] = 1
        batch["cand_tmask"][i, s, OTHER] = (1 << F.N_TARGET) - 1

    batch["turn"][:] = (3, 5, 2)
    slot(
        0,
        0,
        0,
        (
            ("earthquake", SPREAD, AUTO),
            ("protect", GUARD, AUTO),
            ("dragonclaw", ATTACK, AIM),
            (NEW_MOVE, ATTACK, AIM),
        ),
    )
    slot(
        0,
        1,
        1,
        (
            ("closecombat", ATTACK, AIM),
            ("fakeout", ATTACK | F.CAND_FIRST_TURN, AIM),
            ("protect", GUARD, AUTO),
        ),
    )
    batch["foe_mon"][0] = (6, 7)
    batch["mon_hp"][0, [0, 1, 6, 7]] = (0.55, 0.88, 0.6, 0.7)
    batch["mon_flag"][0, 0, F.FLAG_MEGA_POSSIBLE] = 1
    batch["other_prior"][0] = (0.07, 0.0)
    for roster in (2, 3):
        batch["switch_mask"][0, :, roster] = 1
        batch["action_mask"][0, :, F.switch_index(roster, C)] = 1

    slot(1, 0, 2, (("protect", GUARD, AUTO), ("earthquake", SPREAD, AUTO)))
    batch["foe_mon"][1, 0] = 8
    batch["mon_hp"][1, [2, 8]] = (0.4, 0.25)
    batch["other_prior"][1, 0] = 0.9

    slot(2, 0, 5, (("closecombat", ATTACK, AIM), ("protect", GUARD, AUTO)))
    slot(2, 1, 1, (("fakeout", ATTACK, AIM),))
    batch["foe_mon"][2] = (6, 9)
    return batch


def names(batch: F.Batch, fz: F.Featurizer, i: int) -> dict[str, Any]:
    """Example ``i`` as the arguments the shipped ``predict`` methods take."""

    def name(row: int) -> str:
        if row < 0:
            return ""
        found = fz.tables.species_ids[int(batch["mon_id"][i, row, F.ID_SPECIES])]
        return "" if found == STRANGER else found

    def hp(row: int) -> float:
        return float(batch["mon_hp"][i, row]) if row >= 0 else 0.0

    act = [int(row) for row in batch["act_mon"][i]]
    foe = [int(row) for row in batch["foe_mon"][i]]
    return {
        "roster": [name(row) for row in range(6)],
        "opponent_roster": [name(row) for row in range(6, 12)],
        "active": [name(row) for row in act],
        "opponent_active": [name(row) for row in foe],
        "hp": [hp(row) for row in (*act, *foe)],
        "turn": int(batch["turn"][i]),
    }


# --- the contract ---------------------------------------------------------------


def test_prediction_is_a_clean_probability_table(legacy, fz):
    batch = scene(fz)
    pred = legacy.predict(batch)
    contract: object = legacy
    assert isinstance(contract, F.Predictor) and legacy.kind == "legacy"
    assert pred["action"].shape == (3, 2, F.action_size(C))
    assert pred["target"].shape == (3, 2, C + 1, F.N_TARGET)
    assert pred["mega"].shape == (3, 2)
    norm = F.normalize_prediction(pred, batch)
    for key in pred:
        assert np.isfinite(pred[key]).all()
        assert np.abs(norm[key] - pred[key]).max() < 1e-12, key
    active = batch["act_mon"] >= 0
    assert np.allclose(pred["action"].sum(-1), active.astype(float))
    assert not pred["action"][1, 1].any() and pred["mega"][1, 1] == 0.0
    assert not any(k.startswith("predict_error") for k in legacy.counters)
    assert legacy.counters["slots"] == 5
    assert legacy.counters["examples_with_an_empty_slot"] == 1
    assert legacy.counters["acting_species_unknown_to_move_model"] == 1
    assert legacy.counters["candidates_unknown_to_move_model"] == 1
    assert legacy.counters["candidates"] == 4 + 3 + 2 + 2 + 1


def test_moves_are_the_shipped_softmax_over_the_known_candidates(legacy, fz):
    batch = scene(fz)
    pred = legacy.predict(batch)
    for i, slot, offered in (
        (0, 0, ["earthquake", "protect", "dragonclaw"]),
        (0, 1, ["closecombat", "fakeout", "protect"]),
        (1, 0, ["protect", "earthquake"]),
        (2, 0, ["closecombat", "protect"]),
    ):
        want = dict(
            legacy.move.predict(
                **names(batch, fz, i), actor_slot=slot, available_moves=offered
            ).moves
        )
        got = pred["action"][i, slot, : len(offered)]
        got = got / got.sum()
        assert got.tolist() == pytest.approx([want[m] for m in offered], abs=1e-6)


def test_switch_probability_and_destination_are_the_shipped_ones(legacy, fz):
    batch = scene(fz)
    pred = legacy.predict(batch)
    for slot in (0, 1):
        want = legacy.switch.predict(**names(batch, fz, 0), actor_slot=slot)
        pointers = pred["action"][0, slot, C + 1 :]
        assert pointers.sum() == pytest.approx(want.switch_probability, abs=1e-9)
        assert want.switch_probability in CALIBRATION["rates"]
        weight = dict(want.targets)
        share = weight["politoed"] / (weight["politoed"] + weight["archaludon"])
        assert pointers[2] / pointers.sum() == pytest.approx(share, abs=1e-6)
        assert pointers[[0, 1, 4, 5]].sum() == 0.0  # not legal: no mass
    # nothing legal to switch to: the whole row is moves
    assert pred["action"][1, 0, C + 1 :].sum() == 0.0
    assert pred["action"][1, 0, : C + 1].sum() == pytest.approx(1.0)


def test_calibration_matches_the_shipped_lookup(legacy):
    raw = np.array([0.0, 0.1, 0.3, 0.30001, 0.5, 0.69, 0.7, 0.9, 1.0])
    want = [legacy.switch.calibrate(float(value)) for value in raw]
    assert legacy._calibrated(raw).tolist() == want
    legacy.switch.calibration = None
    assert legacy._calibrated(raw) is raw
    legacy.switch.calibration = {"upper_bounds": [0.5], "rates": []}
    assert legacy._calibrated(raw) is raw


def test_unknown_candidate_floor_and_other_mass(legacy, fz):
    batch = scene(fz)
    tuned = legacy.with_config(unknown_rate=0.03)
    pred = tuned.predict(batch)
    stay = 1.0 - pred["action"][0, 0, C + 1 :].sum()
    # the candidate the old vocabulary lacks gets the floor, OTHER its prior
    assert pred["action"][0, 0, 3] == pytest.approx(0.03 * stay)
    assert pred["action"][0, 0, OTHER] == pytest.approx(0.07 * stay, rel=1e-3)
    assert pred["action"][0, 0, :3].sum() == pytest.approx(
        (1 - 0.07 - 0.03) * stay,
        rel=1e-3,  # the prior is stored as float16
    )
    # a prior of 0 is floored, a prior of 0.9 is capped
    stay = 1.0 - pred["action"][0, 1, C + 1 :].sum()
    assert pred["action"][0, 1, OTHER] == pytest.approx(1e-3 * stay)
    assert pred["action"][1, 0, OTHER] == pytest.approx(0.5)
    # the floor is capped in total, and the cap is a setting
    capped = legacy.with_config(unknown_rate=0.9, unknown_cap=0.25).predict(batch)
    stay = 1.0 - capped["action"][0, 0, C + 1 :].sum()
    assert capped["action"][0, 0, 3] == pytest.approx(0.25 * stay)
    # a slot whose every candidate is unknown shares what OTHER leaves
    batch["cand_move"][2, 1, 0] = move_id(fz, NEW_MOVE)
    lone = legacy.predict(batch)["action"][2, 1]
    assert lone[0] + lone[OTHER] == pytest.approx(1.0) and lone[0] > lone[OTHER]


def test_vocabulary_scope_gives_other_what_the_model_leaves(legacy, fz):
    batch = scene(fz)
    wide = legacy.with_config(scope=L.SCOPE_VOCABULARY, unknown_rate=0.0)
    assert wide.config.scope == "vocabulary" and legacy.config.scope == "candidates"
    pred = wide.predict(batch)
    with torch.no_grad():
        args = names(batch, fz, 0)
        logits, _ = legacy.move.model(
            legacy.move._encode(args["roster"]).unsqueeze(0),
            legacy.move._encode(args["opponent_roster"]).unsqueeze(0),
            legacy.move._encode(args["active"]).unsqueeze(0),
            legacy.move._encode(args["opponent_active"]).unsqueeze(0),
            torch.tensor([args["hp"]], dtype=torch.float32),
            torch.tensor([0]),
            torch.tensor([args["turn"]]),
        )
    whole = logits[0].softmax(0).numpy()
    stay = 1.0 - pred["action"][0, 0, C + 1 :].sum()
    index = [legacy.move.move_vocab[m] for m in ("earthquake", "protect", "dragonclaw")]
    assert pred["action"][0, 0, :3] == pytest.approx(whole[index] * stay, abs=1e-6)
    assert pred["action"][0, 0, 3] == 0.0  # unknown, and its floor is 0 here
    left = 1.0 - whole[index].sum()
    assert left > 0.05  # two vocabulary moves are not candidates
    assert pred["action"][0, 0, OTHER] == pytest.approx(left * stay, abs=1e-6)
    assert L.LegacyConfig(scope="nonsense").checked().scope == L.SCOPE_CANDIDATES


def test_temperatures_flatten_the_two_softmaxes(legacy, fz):
    batch = scene(fz)
    cold = legacy.predict(batch)
    warm = legacy.with_config(move_temperature=20, destination_temperature=20)
    assert warm.config.move_temperature == 20.0
    hot = warm.predict(batch)

    def spread(pred: Any) -> float:
        row = pred["action"][0, 0, :3]
        return float(row.max() / row.sum())

    assert spread(hot) < spread(cold) and spread(hot) == pytest.approx(1 / 3, abs=0.02)
    pointers = hot["action"][0, 0, C + 1 :]
    assert pointers[2] / pointers.sum() == pytest.approx(0.5, abs=0.02)
    # the calibrated switch probability and the targets do not move
    assert hot["action"][0, 0, C + 1 :].sum() == pytest.approx(
        cold["action"][0, 0, C + 1 :].sum()
    )
    assert np.array_equal(hot["target"], cold["target"])


def test_targets_are_read_by_name_and_kept_inside_the_legal_mask(legacy, fz):
    assert L._target_permutation(MovePredictor.TARGET_NAMES) == [2, 3, 1, 0, 4]
    assert L._target_permutation(("foe_b", "foe_a")) == [1, 0, -1, -1, -1]
    assert L._target_permutation(("x", "self")) == [-1, -1, -1, 1, 0]
    batch = scene(fz)
    pred = legacy.predict(batch)
    floor = legacy.config.target_floor
    args = names(batch, fz, 0)
    joint = {
        (move, target): value
        for move, target, value in legacy.move.predict(
            **args,
            actor_slot=0,
            available_moves=["earthquake", "protect", "dragonclaw"],
        ).actions
    }
    # the aimed attack: foe a, foe b and ally keep the model's proportions
    claw = pred["target"][0, 0, 2]
    total = sum(
        joint[("dragonclaw", t)] for t in ("self", "ally", "foe_a", "foe_b", "field")
    )
    want = np.array(
        [joint[("dragonclaw", t)] / total + floor for t in ("foe_a", "foe_b", "ally")]
    )
    assert claw[[F.T_FOE_A, F.T_FOE_B, F.T_ALLY]] == pytest.approx(
        want / want.sum(), abs=1e-6
    )
    assert claw[F.T_SELF] == 0.0 and claw[F.T_AUTO] == 0.0
    # one legal class: certain, whatever the model's head says
    assert pred["target"][0, 0, 0].tolist() == [0, 0, 0, 0, 1]
    assert pred["target"][0, 0, 1].tolist() == [0, 0, 0, 0, 1]
    # a move the model cannot score, and OTHER: uniform over what is legal
    assert pred["target"][0, 0, 3].tolist() == pytest.approx(
        [1 / 3, 1 / 3, 1 / 3, 0, 0]
    )
    assert pred["target"][0, 0, OTHER].tolist() == pytest.approx([0.2] * 5)
    # a head rigged towards its 'field' class moves the mass to our 'auto'
    batch["cand_tmask"][0, 0, 2] = AIM | AUTO
    with torch.no_grad():
        legacy.move.model.target_head.weight.zero_()
        legacy.move.model.target_head.bias.copy_(torch.tensor([0, 0, 0, 0, 9.0]))
    rigged = legacy.predict(batch)["target"][0, 0, 2]
    assert rigged[F.T_AUTO] > 0.99 and rigged[F.T_SELF] == 0.0


def test_mega_is_one_rate_where_a_mega_is_still_possible(legacy, fz):
    batch = scene(fz)
    pred = legacy.with_config(mega_rate=0.31).predict(batch)
    assert pred["mega"].tolist() == [[0.31, 0.0], [0.0, 0.0], [0.0, 0.0]]
    assert legacy.predict(batch)["mega"][0, 0] == 0.5


def test_an_empty_slot_is_fed_as_a_blank(legacy, fz):
    """Example 1: the partner and one foe are gone; compare with a direct call."""
    batch = scene(fz)
    args = names(batch, fz, 1)
    assert args["active"] == ["politoed", ""] and args["opponent_active"][1] == ""
    assert args["hp"][1] == 0.0 and args["hp"][3] == 0.0
    want = dict(
        legacy.move.predict(
            **args, actor_slot=0, available_moves=["protect", "earthquake"]
        ).moves
    )
    row = legacy.predict(batch)["action"][1, 0, :2]
    assert (row / row.sum()).tolist() == pytest.approx(
        [want["protect"], want["earthquake"]], abs=1e-6
    )


def test_predict_never_raises_and_reads_no_label(legacy, fz):
    assert legacy.predict({}) == {}
    assert legacy.counters["predict_error:KeyError"] == 1
    batch = scene(fz)
    broken = dict(batch)
    del broken["mon_hp"]
    fallback = legacy.predict(broken)
    assert legacy.counters["predict_error:KeyError"] == 2
    uniform = F.uniform_prediction(broken)
    assert all(np.array_equal(fallback[k], uniform[k]) for k in uniform)
    assert not any(name.startswith(("y_", "m_")) for name in batch)
    legacy.predict(batch)  # features alone are enough
    assert legacy.counters["predict_error:KeyError"] == 2
    assert legacy.candidate_moves({}).shape == (0, 2, 0)
    empty = {name: array[:0] for name, array in batch.items()}
    assert legacy.predict(empty)["action"].shape == (0, 2, F.action_size(C))


def test_ids_outside_the_tables_read_as_unknown(legacy, fz):
    batch = scene(fz)
    batch["mon_id"][2, 5, F.ID_SPECIES] = 65000
    batch["cand_move"][2, 0, 0] = 65000
    pred = legacy.predict(batch)
    assert not any(k.startswith("predict_error") for k in legacy.counters)
    assert pred["action"][2, 0].sum() == pytest.approx(1.0)
    assert legacy.candidate_moves(batch)[2, 0].tolist() == [
        0,
        legacy.move.move_vocab["protect"],
        0,
        0,
    ]


# --- settings and base rates ----------------------------------------------------


def test_config_is_forced_into_range_and_unknown_keys_are_ignored(legacy):
    wild = L.LegacyConfig(
        unknown_rate=7.0,
        unknown_cap=2.0,
        other_floor=-1.0,
        other_cap=float("nan"),
        mega_rate=-3.0,
        move_temperature=0.0,
        destination_temperature=float("inf"),
        batch_size=0,
    ).checked()
    assert wild.unknown_rate == 1.0 and wild.unknown_cap == 0.9
    assert 0 < wild.other_floor <= 1e-9 and 0 < wild.other_cap <= 1e-9
    assert wild.mega_rate == 0.0 and wild.batch_size == 1
    assert wild.move_temperature == 0.05 and wild.destination_temperature == 0.05
    twin = legacy.with_config(name="twin", mega_rate=0.2, no_such_setting=1)
    assert twin.name == "twin" and twin.config.mega_rate == 0.2
    assert legacy.name == "legacy" and legacy.config.mega_rate == 0.5
    assert twin.move is legacy.move and twin._lock is legacy._lock
    found = twin.describe()
    json.dumps(found)
    assert found["config"]["mega_rate"] == 0.2
    assert found["moves_known_to_move_model"] == len(MOVES)
    assert found["species_known_to_move_model"] == 11
    assert found["target_columns"] == {
        "foe_a": "foe_a",
        "foe_b": "foe_b",
        "ally": "ally",
        "self": "self",
        "auto": "field",
    }
    assert found["mapping"] == list(L.MAPPING_NOTES) and found["disadvantages"]


def test_training_rates_are_weighted_base_rates(legacy, fz):
    batch = scene(fz)
    batch = F.take(batch, [0, 0, 0, 0])
    batch["mon_flag"][:, 1, F.FLAG_MEGA_POSSIBLE] = (0, 0, 1, 1)
    batch["y_mega"] = np.array([[1, 0], [0, 1], [1, 0], [-1, 0]], dtype=np.int8)
    # slot a has one unknown candidate (column 3), chosen in the first example;
    # slot b of the third example has two, and its move is a known one
    batch["cand_move"][2, 1, :2] = move_id(fz, NEW_MOVE)
    batch["y_action"] = np.array([[3, 0], [0, 1], [OTHER, 2], [-1, 0]], dtype=np.int8)
    batch["m_weight"] = np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float32)
    rates = L.training_rates(batch, legacy)
    # Mega: slot a in examples 0-2 (weights 1, 2, 3; yes, no, yes), slot b
    # where possible (examples 2 and 3: no, no; weights 3 and 5)
    assert rates["mega_rate"] == pytest.approx((1 + 3) / (1 + 2 + 3 + 3 + 5))
    # offered: one candidate on slot a of the three visible examples (weights
    # 1, 2, 3) and two on slot b of the third (weight 3); made once (weight 1)
    assert rates["unknown_rate"] == pytest.approx(1 / (1 + 2 + 3 + 2 * 3))
    assert set(rates) == {"mega_rate", "unknown_rate"}
    fitted = legacy.with_config(**rates)
    assert fitted.config.unknown_rate == pytest.approx(1 / 12)
    del batch["m_weight"]
    assert L.training_rates(batch, legacy)["mega_rate"] == pytest.approx(2 / 5)
    # nothing to measure: the rate is left out, and a broken batch gives nothing
    batch["y_mega"][:] = -1
    batch["cand_move"][:, 0, 3] = move_id(fz, "protect")
    batch["cand_move"][2, 1, :2] = move_id(fz, "closecombat")
    assert L.training_rates(batch, legacy) == {}
    before = sum(L.COUNTERS.values())
    assert L.training_rates({}, legacy) == {}
    assert sum(L.COUNTERS.values()) == before + 1


# --- loading --------------------------------------------------------------------


def write_files(tmp_path: Path) -> tuple[Path, Path]:
    """The two tiny models in the layout of the shipped files."""
    move, switch = tiny_models()
    config = {"embed_dim": 8, "hidden_dim": 16}
    move_path, switch_path = tmp_path / "move.pt", tmp_path / "switch.pt"
    torch.save(
        {
            "state_dict": move.model.state_dict(),
            "species_vocab": move.species_vocab,
            "move_vocab": move.move_vocab,
            "config": config,
        },
        move_path,
    )
    torch.save(
        {
            "state_dict": switch.model.state_dict(),
            "vocab": switch.vocab,
            "config": config,
            "calibration": CALIBRATION,
        },
        switch_path,
    )
    return move_path, switch_path


def test_load_reads_files_in_the_shipped_layout(tmp_path, fz, legacy):
    move_path, switch_path = write_files(tmp_path)
    loaded = L.LegacyPredictor.load(move_path, switch_path, fz, name="from_files")
    assert loaded.name == "from_files"
    assert loaded.files == {"move": str(move_path), "switch": str(switch_path)}
    batch = scene(fz)
    first, second = legacy.predict(batch), loaded.predict(batch)
    assert all(np.allclose(first[k], second[k], atol=1e-9) for k in first)


def test_load_failures(tmp_path, fz):
    move_path, switch_path = write_files(tmp_path)
    with pytest.raises(FileNotFoundError):
        L.LegacyPredictor.load(tmp_path / "missing.pt", switch_path, fz)
    junk = tmp_path / "junk.pt"
    torch.save({"not": "a model"}, junk)
    with pytest.raises(ValueError, match="did not load"):
        L.LegacyPredictor.load(junk, switch_path, fz)
    before = dict(L.COUNTERS)
    assert L.LegacyPredictor.try_load(junk, switch_path, fz) is None
    assert L.LegacyPredictor.try_load(tmp_path / "missing.pt", switch_path, fz) is None
    assert (
        L.COUNTERS["load_error:ValueError"]
        == before.get("load_error:ValueError", 0) + 1
    )
    assert (
        L.COUNTERS["load_error:FileNotFoundError"]
        == before.get("load_error:FileNotFoundError", 0) + 1
    )
    assert L.LegacyPredictor.try_load(move_path, switch_path, fz) is not None


# --- the shipped files on real examples (skipped when the dataset is absent) ----


@pytest.mark.skipif(
    not (DATASET / "manifest.json").is_file()
    or not SHIPPED_MOVE.is_file()
    or not SHIPPED_SWITCH.is_file(),
    reason="results_oppmodel/v1_ondisk or the shipped models are absent",
)
def test_shipped_models_on_real_examples():
    featurizer = F.Featurizer.load(DATASET)
    data, _ = F.load_dataset(DATASET, splits=["val"])
    batch = {
        name: array[:300]
        for name, array in data.items()
        if not name.startswith(("y_", "m_"))
    }
    shipped = L.LegacyPredictor.load(SHIPPED_MOVE, SHIPPED_SWITCH, featurizer)
    pred = shipped.predict(batch)
    assert not any(k.startswith("predict_error") for k in shipped.counters)
    norm = F.normalize_prediction(pred, batch)
    assert max(float(np.abs(norm[k] - pred[k]).max()) for k in pred) < 1e-9
    active = batch["act_mon"] >= 0
    assert np.allclose(pred["action"].sum(-1), active.astype(float))
    # every species and nearly every candidate of real play is in its vocabulary
    assert shipped.counters["acting_species_unknown_to_move_model"] <= 5
    share = (
        shipped.counters["candidates_unknown_to_move_model"]
        / shipped.counters["candidates"]
    )
    assert share < 0.05
    # the same numbers as the shipped predict, on the first four-active slots
    species = featurizer.tables.species_ids
    moves = featurizer.tables.move_ids
    legacy_moves = shipped.candidate_moves(batch)
    full = np.flatnonzero(active.all(1) & (batch["foe_mon"] >= 0).all(1))[:5]
    assert len(full) == 5
    for i in full:

        def name(row: int) -> str:
            return species[int(batch["mon_id"][i, row, F.ID_SPECIES])]

        act = [int(row) for row in batch["act_mon"][i]]
        foe = [int(row) for row in batch["foe_mon"][i]]
        common = {
            "roster": [name(row) for row in range(6)],
            "opponent_roster": [name(row) for row in range(6, 12)],
            "active": [name(row) for row in act],
            "opponent_active": [name(row) for row in foe],
            "hp": [float(batch["mon_hp"][i, row]) for row in (*act, *foe)],
            "turn": int(batch["turn"][i]),
        }
        for slot in (0, 1):
            columns = np.flatnonzero(legacy_moves[i, slot] > 0)
            offered = [moves[int(batch["cand_move"][i, slot, c])] for c in columns]
            want = dict(
                shipped.move.predict(
                    **common, actor_slot=slot, available_moves=offered
                ).moves
            )
            got = pred["action"][i, slot, columns]
            assert (got / got.sum()).tolist() == pytest.approx(
                [want[m] for m in offered], abs=1e-5
            )
            want_switch = shipped.switch.predict(**common, actor_slot=slot)
            pointers = pred["action"][i, slot, batch["cand_move"].shape[-1] + 1 :]
            if batch["switch_mask"][i, slot].any():
                assert pointers.sum() == pytest.approx(
                    want_switch.switch_probability, abs=1e-9
                )
            else:
                assert pointers.sum() == 0.0
