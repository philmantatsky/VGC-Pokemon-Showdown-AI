"""OppNet: shapes, masks, the prior at init, the loss, Elo, slot mirror, payload.

Inputs are a hand-written log driven through the real foundation and encoded
by the real ``Featurizer``. A few tests also read a slice of the built dataset
or a corpus folder and skip when those are absent (they are git-ignored).
"""

from __future__ import annotations

import ast
import importlib.util
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from vgc_bench.src.oppmodel import events as E
from vgc_bench.src.oppmodel import features as F
from vgc_bench.src.oppmodel import model as M
from vgc_bench.src.oppmodel import public_state as P

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "results_oppmodel" / "v1_ondisk"
CORPUS = ROOT / "battle_logs_web_mc_20260927" / "top"
ARTIFACT_MODULE = "vgc_bench.src.oppmodel.artifact"

HEADER = """|j|☆Alice
|j|☆Bob
|gametype|doubles
|player|p1|Alice|1|1300
|player|p2|Bob|2|1250
|gen|9
|tier|[Gen 9 Champions] VGC 2026 Reg M-C
|rated|
|clearpoke
|poke|p1|Incineroar, L50, M|
|poke|p1|Rillaboom, L50, F|
|poke|p1|Charizard, L50, M|
|poke|p1|Farigiraf, L50, F|
|poke|p1|Torkoal, L50, M|
|poke|p1|Venusaur, L50, M|
|poke|p2|Garchomp, L50, M|
|poke|p2|Sneasler, L50, M|
|poke|p2|Politoed, L50, M|
|poke|p2|Archaludon, L50, M|
|poke|p2|Indeedee-F, L50, F|
|poke|p2|Pawmot, L50, M|
|teampreview|4
|teamsize|p1|4
|teamsize|p2|4
|start
|switch|p1a: Incineroar|Incineroar, L50, M|100/100
|switch|p1b: Rillaboom|Rillaboom, L50, F|100/100
|switch|p2a: Garchomp|Garchomp, L50, M|100/100
|switch|p2b: Sneasler|Sneasler, L50, M|100/100
"""

# Seven turns: a Mega, a flinch, Protect, spread and aimed attacks on both foe
# slots, an ally-aimed move, voluntary switches, a faint before acting, a turn
# with one actor slot empty, and a forfeit.
GAME = """|turn|1
|detailschange|p2a: Garchomp|Garchomp-Mega, L50, M
|-mega|p2a: Garchomp|Garchomp|Garchompite
|move|p1a: Incineroar|Fake Out|p2b: Sneasler
|-damage|p2b: Sneasler|88/100
|cant|p2b: Sneasler|flinch
|move|p1b: Rillaboom|Protect|p1b: Rillaboom
|-singleturn|p1b: Rillaboom|Protect
|move|p2a: Garchomp|Earthquake|p1a: Incineroar|[spread] p1a,p1b
|-activate|p1b: Rillaboom|move: Protect
|-damage|p1a: Incineroar|41/100
|upkeep
|turn|2
|switch|p1a: Charizard|Charizard, L50, M|100/100
|move|p2b: Sneasler|Close Combat|p1b: Rillaboom
|-damage|p1b: Rillaboom|35/100
|move|p1b: Rillaboom|Helping Hand|p1a: Charizard
|-singleturn|p1a: Charizard|Helping Hand|[of] p1b: Rillaboom
|move|p2a: Garchomp|Rock Slide|p1a: Charizard|[spread] p1a,p1b
|-damage|p1a: Charizard|30/100
|-damage|p1b: Rillaboom|20/100
|upkeep
|turn|3
|move|p2a: Garchomp|Protect|p2a: Garchomp
|-singleturn|p2a: Garchomp|Protect
|move|p2b: Sneasler|Dire Claw|p1a: Charizard
|-damage|p1a: Charizard|0 fnt
|faint|p1a: Charizard
|move|p1b: Rillaboom|Wood Hammer|p2b: Sneasler
|-damage|p2b: Sneasler|40/100
|upkeep
|switch|p1a: Torkoal|Torkoal, L50, M|100/100
|turn|4
|move|p2b: Sneasler|Close Combat|p1a: Torkoal
|-damage|p1a: Torkoal|35/100
|move|p2a: Garchomp|Dragon Claw|p1b: Rillaboom
|-damage|p1b: Rillaboom|0 fnt
|faint|p1b: Rillaboom
|move|p1a: Torkoal|Eruption|p2a: Garchomp|[spread] p2a,p2b
|-damage|p2a: Garchomp|60/100
|-damage|p2b: Sneasler|0 fnt
|faint|p2b: Sneasler
|upkeep
|switch|p2b: Politoed|Politoed, L50, M|100/100
|switch|p1b: Incineroar|Incineroar, L50, M|41/100
|turn|5
|switch|p2a: Archaludon|Archaludon, L50, M|100/100
|move|p1b: Incineroar|Flare Blitz|p2a: Archaludon
|-damage|p2a: Archaludon|80/100
|move|p2b: Politoed|Icy Wind|p1a: Torkoal|[spread] p1a,p1b
|-damage|p1a: Torkoal|20/100
|-damage|p1b: Incineroar|0 fnt
|faint|p1b: Incineroar
|move|p1a: Torkoal|Heat Wave|p2a: Archaludon|[spread] p2a,p2b
|-damage|p2a: Archaludon|60/100
|-damage|p2b: Politoed|70/100
|upkeep
|turn|6
|move|p2b: Politoed|Protect|p2b: Politoed
|-singleturn|p2b: Politoed|Protect
|move|p2a: Archaludon|Dragon Pulse|p1a: Torkoal
|-damage|p1a: Torkoal|5/100
|move|p1a: Torkoal|Heat Wave|p2a: Archaludon|[spread] p2a,p2b
|-activate|p2b: Politoed|move: Protect
|-damage|p2a: Archaludon|30/100
|upkeep
|turn|7
|-message|Alice forfeited.
|win|Bob
"""

USES = {
    "garchomp": {
        "earthquake": 50,
        "protect": 30,
        "dragonclaw": 10,
        "rockslide": 5,
        "stompingtantrum": 3,
        "swordsdance": 2,
    },
    "sneasler": {"closecombat": 40, "fakeout": 30, "direclaw": 20, "protect": 10},
    "politoed": {"icywind": 5, "encore": 3, "protect": 2},
    "archaludon": {"dragonpulse": 6, "electroshot": 9, "protect": 5},
    "incineroar": {"fakeout": 30, "flareblitz": 20, "partingshot": 25, "protect": 5},
    "rillaboom": {"woodhammer": 20, "grassyglide": 30, "helpinghand": 2, "protect": 9},
    "torkoal": {"eruption": 30, "heatwave": 12, "protect": 10, "earthpower": 6},
    "charizard": {"heatwave": 25, "protect": 12, "weatherball": 9},
}
FOE_FLIP = {E.TARGET_FOE_A: E.TARGET_FOE_B, E.TARGET_FOE_B: E.TARGET_FOE_A}
SMALL = {"d_model": 32, "n_layers": 1, "n_heads": 2, "d_ff": 48, "d_embed": 16}


# --- helpers (also used by the trainer's tests) ---------------------------------


def repertoire() -> F.Repertoire:
    out = F.Repertoire()
    for key, moves in USES.items():
        for move, count in moves.items():
            out.add(key, move, count)
    return out


def drive(body: str = GAME, header: str = HEADER) -> P.DriveResult:
    result = P.drive_log(header + body, "battle-model-1")
    assert result.usable, result.skip_reason
    return result


def game_examples(
    fz: F.Featurizer, sides: tuple[str, ...] = E.SIDES
) -> list[F.Example]:
    """Every turn of the hand-written game, for the listed actor sides."""
    examples = [
        fz.encode_turn(record.snapshot, record.actions, side)
        for record in drive().turns
        for side in sides
    ]
    return [example for example in examples if example is not None]


def game_batch(fz: F.Featurizer, sides: tuple[str, ...] = E.SIDES) -> F.Batch:
    batch = F.collate(game_examples(fz, sides))
    assert batch
    return batch


def small_net(fz: F.Featurizer, seed: int = 0, **overrides: Any) -> M.OppNet:
    torch.manual_seed(seed)
    return M.OppNet.for_featurizer(fz, **{**SMALL, **overrides})


def jolt(net: M.OppNet, seed: int = 1) -> M.OppNet:
    """Random weights in the zero-initialised output layers: a non-prior network."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for scorer in (
            net.act_cand,
            net.act_other,
            net.act_switch,
            net.tgt_foe,
            net.tgt_ally,
            net.tgt_own,
        ):
            scorer.out.weight.copy_(
                torch.randn(scorer.out.weight.shape, generator=generator)
            )
            scorer.out.bias.copy_(
                torch.randn(scorer.out.bias.shape, generator=generator)
            )
    return net


def real_slice(n: int = 64) -> F.Batch:
    """The first ``n`` validation examples of the built dataset, or skip."""
    if not (DATASET / "manifest.json").is_file():
        pytest.skip("results_oppmodel/v1_ondisk is not built here")
    batch, _ = F.load_dataset(DATASET, splits=["val"])
    return F.take(batch, slice(0, n))


def flip(letter: str | None) -> str | None:
    return {"a": "b", "b": "a"}.get(letter or "", letter)


def mirror_action(action: E.SlotAction | None, side: str) -> E.SlotAction | None:
    """The action after ``side``'s slots were renamed a <-> b."""
    if action is None:
        return None
    if action.side == side:
        return replace(action, slot=flip(action.slot) or "")
    return replace(action, target=FOE_FLIP.get(action.target or "", action.target))


def mirror_snapshot(snapshot: P.PublicSnapshot, side: str) -> P.PublicSnapshot:
    """The snapshot of the same game with ``side``'s two slots renamed."""
    sides = {}
    for name, team in snapshot.sides.items():
        mons = tuple(
            replace(
                mon,
                slot=flip(mon.slot) if name == side else mon.slot,
                last_action=mirror_action(mon.last_action, side),
            )
            for mon in team.mons
        )
        active = dict(team.active)
        if name == side:
            active = {str(flip(letter)): index for letter, index in team.active.items()}
        sides[name] = replace(team, mons=mons, active=active)
    return replace(snapshot, sides=sides)


def mirror_actions(
    actions: dict[str, E.SlotAction], side: str
) -> dict[str, E.SlotAction]:
    out: dict[str, E.SlotAction] = {}
    for key, action in actions.items():
        mirrored = mirror_action(action, side)
        assert mirrored is not None
        out[key[:2] + str(flip(key[2:])) if key.startswith(side) else key] = mirrored
    return out


def mirror_mismatches(fz: F.Featurizer, record: P.TurnRecord) -> tuple[int, list[str]]:
    """Compare ``swap_slots`` with re-encoding a mirrored snapshot, every way."""
    compared, bad = 0, []
    for actor in E.SIDES:
        base = fz.encode_turn(record.snapshot, record.actions, actor)
        if base is None:
            continue
        other = E.SIDES[1 - E.SIDES.index(actor)]
        for which in (("p1",), ("p2",), ("p1", "p2")):
            snapshot, actions = record.snapshot, dict(record.actions)
            for side in which:
                snapshot = mirror_snapshot(snapshot, side)
                actions = mirror_actions(actions, side)
            want = fz.encode_turn(snapshot, actions, actor)
            assert want is not None
            got = M.swap_slots(
                F.collate([base]), actor=actor in which, other=other in which
            )
            compared += 1
            for name, array in want.items():
                same = np.array_equal(array, got[name][0])
                if not same or array.dtype != got[name].dtype:
                    bad.append(f"turn {record.turn} {actor} {which} {name}")
            twice = M.swap_slots(got, actor=actor in which, other=other in which)
            for name, array in base.items():
                if not np.array_equal(array, twice[name][0]):
                    bad.append(f"twice: turn {record.turn} {actor} {which} {name}")
    return compared, bad


@pytest.fixture(scope="module")
def fz() -> F.Featurizer:
    return F.Featurizer.build(repertoire())


@pytest.fixture(scope="module")
def batch(fz: F.Featurizer) -> F.Batch:
    return game_batch(fz)


# --- the synthetic batch covers what the tests rely on -------------------------


def test_synthetic_batch_holds_every_case(batch: F.Batch):
    assert batch["act_mon"].shape[0] == 14  # 7 turns x 2 actors
    assert (batch["act_mon"] < 0).any()  # an empty actor slot
    assert (batch["foe_mon"] < 0).any()
    assert (batch["elo_known"] == 1).all() and set(batch["elo"][0]) == {1300, 1250}
    targets = set(batch["y_target"][batch["y_target"] >= 0].tolist())
    assert targets == {F.T_FOE_A, F.T_FOE_B, F.T_ALLY, F.T_AUTO}
    assert (batch["y_action"] >= F.switch_index(0)).any()  # a voluntary switch
    assert (batch["y_mega"] == 1).any()
    censored = (batch["y_action"] < 0) & batch["y_set"].any(-1)
    assert censored.any()
    assert not batch["y_set"].any(-1).all()  # the forfeit turn carries no set


# --- shapes and masks ---------------------------------------------------------


def test_forward_shapes(fz: F.Featurizer, batch: F.Batch):
    net = small_net(fz).eval()
    n = batch["act_mon"].shape[0]
    x = M.to_tensors(batch)
    assert set(x) == set(M.FEATURE_KEYS)
    with torch.no_grad():
        hidden = net.encode(x)
        out = net(x)
    assert hidden.shape == (n, M.N_TOKENS, 32) and M.N_TOKENS == 15
    assert out["action"].shape == (n, 2, fz.n_actions)
    assert out["target"].shape == (n, 2, fz.n_cand + 1, F.N_TARGET)
    assert out["mega"].shape == (n, 2)
    assert out["target_mask"].dtype == torch.bool
    assert np.array_equal(
        out["target_mask"].numpy(), F.expand_target_mask(batch["cand_tmask"])
    )
    assert np.array_equal(out["active"].numpy(), batch["act_mon"] >= 0)

    pred = M.OppNetPredictor(net, fz).predict(batch)
    assert pred["action"].shape == (n, 2, fz.n_actions)
    assert pred["target"].shape == (n, 2, fz.n_cand + 1, F.N_TARGET)
    assert pred["mega"].shape == (n, 2)
    assert all(value.dtype == np.float32 for value in pred.values())
    norm = F.normalize_prediction(pred, batch)  # raises on a shape or NaN problem
    for name in ("action", "target", "mega"):
        assert np.allclose(norm[name], pred[name], atol=1e-6), name
    active = batch["act_mon"] >= 0
    assert np.allclose(pred["action"].sum(-1)[active], 1.0, atol=1e-5)
    assert isinstance(M.OppNetPredictor(net, fz), F.Predictor)


def test_default_network_has_the_designed_shape(fz: F.Featurizer):
    net = M.OppNet.for_featurizer(fz)
    config = net.config
    assert (config.d_model, config.n_layers, config.n_heads) == (128, 3, 4)
    assert (config.d_ff, config.dropout, config.pre_norm) == (256, 0.1, True)
    assert len(net.blocks) == 3 and net.emb_species.num_embeddings == fz.n_species
    assert net.emb_move.num_embeddings == fz.n_moves
    assert net.emb_elo.num_embeddings == config.elo_buckets + 1
    assert 500_000 < net.n_parameters() < 1_500_000
    # The dex numerics are buffers outside the state dict, and not trainable.
    assert "species_num" not in net.state_dict() and "move_num" not in net.state_dict()
    assert net.species_num.shape == fz.tables.species_num.shape
    assert not net.species_num.requires_grad and not net.move_num.requires_grad
    post = M.OppNet.for_featurizer(fz, pre_norm=False, **SMALL).eval()
    with torch.no_grad():
        assert torch.isfinite(post(M.to_tensors(game_batch(fz)))["action"]).all()


def test_illegal_actions_and_targets_get_zero_probability(
    fz: F.Featurizer, batch: F.Batch
):
    net = jolt(small_net(fz))
    pred = M.OppNetPredictor(net, fz).predict(batch)
    mask = batch["action_mask"].astype(bool)
    legal = F.expand_target_mask(batch["cand_tmask"])
    assert (~mask).any() and (~legal).any()
    assert (pred["action"][~mask] == 0).all()
    assert (pred["target"][~legal] == 0).all()
    assert (pred["action"][mask] > 0).all()
    assert (pred["target"][legal] > 0).all()
    empty = batch["act_mon"] < 0
    assert empty.any()
    assert (pred["action"][empty] == 0).all() and (pred["target"][empty] == 0).all()
    # Mega: a probability where the Pokemon can Mega-evolve, exactly 0 elsewhere.
    can_mega = np.asarray(F.slot_view(batch)["mega_possible"]).astype(bool)
    assert (
        can_mega.any() and (~can_mega & ~empty).any() and not (can_mega & empty).any()
    )
    assert (pred["mega"][~can_mega] == 0).all()
    assert ((pred["mega"][can_mega] > 0) & (pred["mega"][can_mega] < 1)).all()
    assert (batch["y_mega"][~can_mega] != 1).all()  # no Mega label under the mask
    # The jolted network is not the prior, so the masks were really applied.
    plain = M.OppNetPredictor(small_net(fz), fz).predict(batch)
    assert not np.allclose(plain["action"], pred["action"], atol=1e-3)


def test_heads_read_the_tokens_the_design_names(fz: F.Featurizer, batch: F.Batch):
    """With no encoder layer a token is its own Pokemon only, so changing one
    Pokemon shows exactly which outputs read its token."""
    net = jolt(small_net(fz, n_layers=0)).eval()
    full = (batch["act_mon"] >= 0).all(1) & (batch["foe_mon"] >= 0).all(1)
    one = F.take(
        batch, slice(int(np.flatnonzero(full)[0]), int(np.flatnonzero(full)[0]) + 1)
    )
    n_cand = fz.n_cand
    slot_a, slot_b = (int(row) for row in one["act_mon"][0])
    foe_a, foe_b = (int(row) for row in one["foe_mon"][0])
    bench = next(r for r in range(F.N_ROSTER) if r not in (slot_a, slot_b))

    def outputs(changed_row: int | None) -> dict[str, np.ndarray]:
        data = dict(one)
        if changed_row is not None:
            hp = one["mon_hp"].copy()
            hp[0, changed_row] = 0.07 if hp[0, changed_row] > 0.5 else 0.93
            boost = one["mon_boost"].copy()
            boost[0, changed_row] += 2
            data["mon_hp"], data["mon_boost"] = hp, boost
        with torch.no_grad():
            out = net(M.to_tensors(data))
        return {name: out[name].numpy() for name in ("action", "target", "mega")}

    base = outputs(None)

    def moved(changed_row: int) -> dict[str, np.ndarray]:
        out = outputs(changed_row)
        return {name: np.abs(out[name] - base[name]) for name in out}

    others = [F.T_FOE_A, F.T_ALLY, F.T_SELF, F.T_AUTO]
    diff = moved(foe_b)  # the Pokemon in the foe's slot b: only the foe_b target
    assert diff["target"][..., F.T_FOE_B].min() > 1e-5
    assert diff["target"][..., others].max() < 1e-6
    assert diff["action"].max() < 1e-6 and diff["mega"].max() < 1e-6
    diff = moved(foe_a)
    assert diff["target"][..., F.T_FOE_A].min() > 1e-5
    assert diff["target"][..., [F.T_FOE_B, F.T_ALLY, F.T_SELF, F.T_AUTO]].max() < 1e-6

    pointer = F.switch_index(bench, n_cand)  # a benched roster Pokemon: its pointer
    diff = moved(bench)
    assert diff["action"][0, :, pointer].min() > 1e-5
    rest = np.delete(diff["action"], pointer, axis=-1)
    assert (
        rest.max() < 1e-6 and diff["target"].max() < 1e-6 and diff["mega"].max() < 1e-6
    )

    diff = moved(slot_b)  # the partner: slot a's ally target and its own pointer
    assert diff["target"][0, 0, :, F.T_ALLY].min() > 1e-5
    assert (
        diff["target"][0, 0][:, [F.T_FOE_A, F.T_FOE_B, F.T_SELF, F.T_AUTO]].max() < 1e-6
    )
    own_pointer = F.switch_index(slot_b, n_cand)
    assert np.delete(diff["action"][0, 0], own_pointer).max() < 1e-6
    assert diff["mega"][0, 0] < 1e-6
    # ... and everything of slot b itself, whose query it is.
    assert diff["action"][0, 1, :n_cand].min() > 1e-5 and diff["mega"][0, 1] > 1e-5
    assert diff["target"][0, 1].min() > 1e-5

    # With encoder layers the foe's state reaches the action logits as well.
    deep = jolt(small_net(fz, n_layers=1)).eval()
    with torch.no_grad():
        before = deep(M.to_tensors(one))["action"].numpy()
        hp = one["mon_hp"].copy()
        hp[0, foe_b] = 0.07 if hp[0, foe_b] > 0.5 else 0.93
        after = deep(M.to_tensors({**one, "mon_hp": hp}))["action"].numpy()
    assert np.abs(after - before)[0, :, :n_cand].max() > 1e-5


def test_an_absent_roster_row_is_masked_out_of_the_attention(
    fz: F.Featurizer, batch: F.Batch
):
    """A roster shorter than six leaves padded rows; nothing may attend to them."""
    net = jolt(small_net(fz, n_layers=2)).eval()
    absent = F.N_MON - 1  # the other side's last roster row, on the bench
    assert not (batch["foe_mon"] == absent).any()
    flags = batch["mon_flag"].copy()
    flags[:, absent, F.FLAG_PRESENT] = 0
    short = {**batch, "mon_flag": flags}
    noisy = dict(short)
    noisy["mon_hp"] = batch["mon_hp"].copy()
    noisy["mon_hp"][:, absent] = 0.31
    noisy["mon_id"] = batch["mon_id"].copy()
    noisy["mon_id"][:, absent] = 7
    noisy["mon_boost"] = batch["mon_boost"].copy()
    noisy["mon_boost"][:, absent] = 3
    predictor = M.OppNetPredictor(net, fz)
    want, got = predictor.predict(short), predictor.predict(noisy)
    for name in want:
        assert np.allclose(want[name], got[name], atol=1e-6), name
    # The same noise on a present row does reach the output.
    present = dict(batch)
    present["mon_hp"], present["mon_id"] = noisy["mon_hp"], noisy["mon_id"]
    moved = predictor.predict(present)["action"]
    assert not np.allclose(moved, predictor.predict(batch)["action"], atol=1e-4)


def test_no_mega_label_sits_outside_the_mega_mask_in_the_built_dataset():
    """The Mega head is gated by ``mega possible``: no label may contradict it."""
    if not (DATASET / "manifest.json").is_file():
        pytest.skip("results_oppmodel/v1_ondisk is not built here")
    data, _ = F.load_dataset(DATASET, splits=["train", "val"])
    possible = np.asarray(F.slot_view(data)["mega_possible"]).astype(bool)
    megas = data["y_mega"] == 1
    assert megas.sum() > 1000 and not (megas & ~possible).any()


def test_real_dataset_slice_runs_and_respects_its_masks():
    data = real_slice(64)
    fz = F.Featurizer.load(DATASET)
    net = jolt(small_net(fz))
    pred = M.OppNetPredictor(net, fz).predict(M.strip_labels(data))
    assert pred["action"].shape == (64, 2, fz.n_actions)
    assert (pred["action"][~data["action_mask"].astype(bool)] == 0).all()
    assert (pred["target"][~F.expand_target_mask(data["cand_tmask"])] == 0).all()
    scores = F.slot_nll(pred, data)
    assert np.isfinite(scores["fine"]).all() and scores["fine_scored"].any()


# --- the prior at init --------------------------------------------------------


def expected_prior(batch: F.Batch, eps: float, switch_logit: float) -> np.ndarray:
    mask = batch["action_mask"].astype(np.float64)
    weights = np.concatenate(
        [
            batch["cand_prior"].astype(np.float64) + eps,
            batch["other_prior"].astype(np.float64)[..., None] + eps,
            np.full(batch["switch_mask"].shape, np.exp(switch_logit)),
        ],
        -1,
    )
    weights = weights * mask
    total = weights.sum(-1, keepdims=True)
    return np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)


def test_untrained_network_reproduces_the_prior(fz: F.Featurizer, batch: F.Batch):
    net = small_net(fz)
    eps = net.config.prior_eps
    predictor = M.OppNetPredictor(net, fz)
    pred = predictor.predict(batch)
    want = expected_prior(batch, eps, net.switch_bias.item())
    assert net.switch_bias.item() == pytest.approx(net.config.switch_logit_init)
    assert np.allclose(pred["action"], want, atol=1e-6)
    # The switch offset is one learned scalar: move it and the prior follows.
    with torch.no_grad():
        net.switch_bias.fill_(-1.0)
    moved = predictor.predict(batch)["action"]
    assert np.allclose(moved, expected_prior(batch, eps, -1.0), atol=1e-6)
    assert not np.allclose(moved, want, atol=1e-3)
    # Targets start uniform over what is legal.
    uniform = F.uniform_prediction(batch)["target"]
    assert np.allclose(pred["target"], uniform, atol=1e-6)
    # The default-size network does the same (every residual scorer ends in zeros).
    big = M.OppNet.for_featurizer(fz)
    again = M.OppNetPredictor(big, fz).predict(batch)["action"]
    assert np.allclose(
        again, expected_prior(batch, eps, big.switch_bias.item()), atol=1e-6
    )


# --- loss ---------------------------------------------------------------------


def random_logits(batch: F.Batch, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    n = batch["act_mon"].shape[0]
    n_action = batch["action_mask"].shape[-1]
    n_move = batch["cand_tmask"].shape[-1]
    return {
        "action": 2.0
        * torch.randn((n, 2, n_action), generator=generator, dtype=torch.float64),
        "target": 2.0
        * torch.randn(
            (n, 2, n_move, F.N_TARGET), generator=generator, dtype=torch.float64
        ),
        "mega": torch.randn((n, 2), generator=generator, dtype=torch.float64),
        "action_mask": torch.from_numpy(batch["action_mask"] > 0),
        "target_mask": torch.from_numpy(F.expand_target_mask(batch["cand_tmask"])),
        "mega_mask": torch.from_numpy(
            np.asarray(F.slot_view(batch)["mega_possible"]).astype(bool)
        ),
        "active": torch.from_numpy(batch["act_mon"] >= 0),
    }


def check_loss_equals_slot_nll(batch: F.Batch, seed: int) -> None:
    out = random_logits(batch, seed)
    labels = M.to_labels(batch)
    terms = {name: value.numpy() for name, value in M.nll_terms(out, labels).items()}
    pred = {name: value.numpy() for name, value in M.probabilities(out).items()}
    want = F.slot_nll(pred, batch)
    assert want["action_scored"].any() and want["target_scored"].any()
    assert np.array_equal(terms["action_scored"], want["action_scored"])
    assert np.array_equal(terms["target_scored"], want["target_scored"])
    assert np.allclose(terms["action"], want["action"], atol=1e-5)
    assert np.allclose(terms["target"], want["target"], atol=1e-5)
    assert np.allclose(terms["action"] + terms["target"], want["fine"], atol=1e-5)
    # Mega: scored where the label is known and the Pokemon can Mega-evolve.
    assert np.array_equal(terms["mega_scored"], want["mega_scored"])
    assert np.allclose(terms["mega"], want["mega"], atol=1e-5)
    # Censored slots are in the action term, with their set.
    censored = want["censored"]
    assert (terms["action"][censored] > 0).any() or not censored.any()

    weight = labels["weight"].double()
    loss, parts = M.total_loss(M.nll_terms(out, labels), weight, mega_weight=0.25)
    per_slot = weight.numpy()[:, None]
    slots = (per_slot * want["action_scored"]).sum()
    fine = (per_slot * want["fine"]).sum() / slots
    mega = (per_slot * terms["mega"]).sum() / (per_slot * terms["mega_scored"]).sum()
    assert float(loss) == pytest.approx(fine + 0.25 * mega, abs=1e-6)
    assert parts["fine"] == pytest.approx(fine, abs=1e-6)
    assert parts["action"] + parts["target"] == pytest.approx(parts["fine"], abs=1e-9)


def test_loss_equals_slot_nll_on_random_logits(batch: F.Batch):
    weighted = dict(batch)
    weighted["m_weight"] = np.linspace(0.2, 1.0, batch["act_mon"].shape[0]).astype(
        np.float32
    )
    for seed in range(3):
        check_loss_equals_slot_nll(weighted, seed)
    assert (batch["y_mega"] >= 0).any()


def test_loss_equals_slot_nll_on_the_real_dataset():
    check_loss_equals_slot_nll(real_slice(256), 7)


def test_loss_ignores_slots_without_information(batch: F.Batch):
    out = random_logits(batch, 3)
    labels = M.to_labels(batch)
    terms = M.nll_terms(out, labels)
    no_set = ~torch.from_numpy(batch["y_set"].astype(bool)).any(-1)
    assert no_set.any()
    assert (terms["action"][no_set] == 0).all() and (terms["target"][no_set] == 0).all()
    assert not terms["action_scored"][no_set].any()
    # A label outside the legal targets is not scored (and stays finite).
    broken = dict(labels)
    row = batch["y_action"].clip(0, batch["cand_tmask"].shape[-1] - 1)
    legal = np.take_along_axis(
        F.expand_target_mask(batch["cand_tmask"]), row[..., None, None], 2
    )[:, :, 0]
    wrong = np.argmin(legal, -1)  # an illegal class where there is one
    broken["y_target"] = torch.from_numpy(wrong.astype(np.int64))
    again = M.nll_terms(out, broken)
    illegal = ~np.take_along_axis(legal, wrong[..., None], 2)[..., 0]
    assert illegal.any()
    assert not again["target_scored"].numpy()[illegal].any()
    assert torch.isfinite(again["target"]).all()
    # An all-masked row and gradients: nothing is NaN.
    logits = {name: value.clone() for name, value in out.items()}
    logits["action"].requires_grad_(True)
    loss, _ = M.total_loss(M.nll_terms(logits, labels), labels["weight"].double())
    loss.backward()
    gradient = logits["action"].grad
    assert gradient is not None and torch.isfinite(gradient).all()


# --- Elo ----------------------------------------------------------------------


def with_elo(batch: F.Batch, elo: Any, known: Any) -> F.Batch:
    out = dict(batch)
    out["elo"] = np.broadcast_to(np.asarray(elo, dtype=np.int16), batch["elo"].shape)
    out["elo_known"] = np.broadcast_to(
        np.asarray(known, dtype=np.uint8), batch["elo_known"].shape
    )
    return out


def elo_dependent_network(fz: F.Featurizer, batch: F.Batch) -> tuple[M.OppNet, int]:
    """A network trained so that slot 0 picks one of two actions by the rating."""
    one = F.take(batch, slice(0, 1))
    first, second = np.flatnonzero(one["action_mask"][0, 0])[:2]
    high, low = with_elo(one, 1700, 1), with_elo(one, 1000, 1)
    data = F.concat_batches([high, low])
    y_set = np.zeros_like(data["y_set"])
    y_set[0, 0, first] = 1
    y_set[1, 0, second] = 1
    data["y_set"] = y_set
    data["y_action"] = np.full_like(data["y_action"], -1)
    data["y_target"] = np.full_like(data["y_target"], -1)
    data["y_mega"] = np.full_like(data["y_mega"], -1)
    net = small_net(fz, dropout=0.0).train()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-3)
    x, labels = M.to_tensors(data), M.to_labels(data)
    for _ in range(60):
        loss, _ = M.total_loss(M.nll_terms(net(x), labels), labels["weight"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return net.eval(), int(first)


def test_trained_network_reads_elo_and_a_blanked_one_cannot(
    fz: F.Featurizer, batch: F.Batch
):
    net, first = elo_dependent_network(fz, batch)
    one = F.take(batch, slice(0, 1))
    high, low = with_elo(one, 1700, 1), with_elo(one, 1000, 1)
    keep = M.OppNetPredictor(net, fz)
    p_high = keep.predict(high)["action"][0, 0, first]
    p_low = keep.predict(low)["action"][0, 0, first]
    assert p_high > p_low + 0.5  # the output depends on Elo

    blind = M.OppNetPredictor(net, fz, elo_mode=F.ELO_BLANK)
    assert blind.elo_mode == F.ELO_BLANK
    reference = blind.predict(with_elo(one, 0, 0))
    for elo, known in (
        (1700, 1),
        (1000, 1),
        (2500, 1),
        (1300, 0),
        ((900, 1900), (1, 0)),
    ):
        changed = with_elo(one, elo, known)
        before = {name: array.copy() for name, array in changed.items()}
        again = blind.predict(changed)
        for name in ("action", "target", "mega"):
            assert np.array_equal(again[name], reference[name]), (elo, known, name)
        # Blanking happens on a copy: the caller's batch is untouched.
        assert all(np.array_equal(changed[name], before[name]) for name in before)
    # The blind answer is the keep-mode answer for an unknown rating, exactly.
    unknown = keep.predict(with_elo(one, 0, 0))
    assert np.array_equal(unknown["action"], reference["action"])
    # A featurizer built Elo-blind makes the predictor blind as well.
    blind_fz = F.Featurizer.build(repertoire(), elo_mode=F.ELO_BLANK)
    assert M.OppNetPredictor(net, blind_fz).elo_mode == F.ELO_BLANK


def test_rating_is_read_only_where_it_is_known(fz: F.Featurizer, batch: F.Batch):
    net, _ = elo_dependent_network(fz, batch)
    keep = M.OppNetPredictor(net, fz)
    hidden_low = keep.predict(with_elo(batch, 1000, 0))
    hidden_high = keep.predict(with_elo(batch, 1900, 0))
    assert np.array_equal(hidden_low["action"], hidden_high["action"])
    # One player known, the other not: only the known one's value matters.
    a = keep.predict(with_elo(batch, (1500, 1000), (1, 0)))
    b = keep.predict(with_elo(batch, (1500, 1900), (1, 0)))
    c = keep.predict(with_elo(batch, (1100, 1900), (1, 0)))
    assert np.array_equal(a["action"], b["action"])
    assert not np.allclose(a["action"], c["action"], atol=1e-4)


def test_elo_buckets_have_a_dedicated_unknown_row(fz: F.Featurizer):
    net = small_net(fz)
    c = net.config
    elo = torch.tensor([[0.0, 1249.0], [1250.0, 9000.0], [1300.0, 1300.0]])
    known = torch.tensor([[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]])
    seen: list[torch.Tensor] = []
    handle = net.emb_elo.register_forward_hook(
        lambda _module, args, _out: seen.append(args[0].clone())
    )
    with torch.no_grad():
        features = net._elo(elo, known)
    handle.remove()
    buckets = seen[0]
    assert buckets[0, 0] == 0 and buckets[2, 0] == 0  # unknown, whatever the value
    assert buckets[0, 1] == 1 + 1249 // c.elo_bucket
    assert buckets[1, 0] == 1 + 1250 // c.elo_bucket == buckets[0, 1] + 1
    assert buckets[1, 1] == c.elo_buckets  # everything above the last bucket
    assert features.shape == (3, 2, 2 * c.d_cat + 2)
    scaled, flag = features[..., -2], features[..., -1]
    assert scaled[0, 0] == 0 and scaled[2, 0] == 0 and flag[0, 0] == 0
    assert float(scaled[1, 0]) == pytest.approx((1250 - c.elo_center) / c.elo_scale)
    assert flag[0, 1] == 1


# --- slot mirror --------------------------------------------------------------


def test_swap_slots_equals_encoding_the_mirrored_snapshot(fz: F.Featurizer):
    compared = 0
    for record in drive().turns:
        count, bad = mirror_mismatches(fz, record)
        assert not bad, bad
        compared += count
    assert compared == 14 * 3
    assert not M.COUNTERS


def test_swap_slots_equals_the_mirrored_snapshot_on_real_logs():
    paths = sorted(CORPUS.glob("*.log"))[:40] if CORPUS.is_dir() else []
    if not paths or not (DATASET / "vocab.json").is_file():
        pytest.skip("corpus logs or the built dataset are absent")
    fz = F.Featurizer.load(DATASET)
    compared = 0
    for path in paths:
        result = P.drive_log(path.read_text(encoding="utf-8"), path.stem)
        if not result.usable:
            continue
        for record in result.turns:
            count, bad = mirror_mismatches(fz, record)
            assert not bad, (path.name, bad[:5])
            compared += count
    assert compared > 500


def test_swap_slots_properties(fz: F.Featurizer, batch: F.Batch):
    data = dict(batch)
    data["m_weight"] = np.ones(batch["act_mon"].shape[0], dtype=np.float32)
    before = {name: array.copy() for name, array in data.items()}
    for actor, other in ((True, False), (False, True), (True, True)):
        swapped = M.swap_slots(data, actor=actor, other=other)
        assert set(swapped) == set(data)
        assert all(swapped[n].dtype == data[n].dtype for n in data)
        assert all(swapped[n].shape == data[n].shape for n in data)
        assert all(swapped[n].flags["C_CONTIGUOUS"] for n in data)
        twice = M.swap_slots(swapped, actor=actor, other=other)
        assert all(np.array_equal(twice[n], data[n]) for n in data), (actor, other)
        assert not all(np.array_equal(swapped[n], data[n]) for n in data)
        # Arrays that carry no slot are shared, not copied.
        for name in ("mon_id", "mon_hp", "elo", "turn", "m_weight", "side_cnt"):
            assert swapped[name] is data[name], name
    assert all(np.array_equal(data[n], before[n]) for n in data)  # input untouched
    both = M.swap_slots(M.swap_slots(data, True, False), False, True)
    direct = M.swap_slots(data, True, True)
    assert all(np.array_equal(both[n], direct[n]) for n in data)
    same = M.swap_slots(data, actor=False, other=False)
    assert all(same[n] is data[n] for n in data)

    # What moves where, on one example with both slots and both foes up.
    row = int(
        np.flatnonzero((batch["act_mon"] >= 0).all(1) & (batch["foe_mon"] >= 0).all(1))[
            0
        ]
    )
    one = F.take(batch, slice(row, row + 1))
    actor_only = M.swap_slots(one, actor=True, other=False)
    assert actor_only["act_mon"][0].tolist() == one["act_mon"][0, ::-1].tolist()
    assert np.array_equal(actor_only["foe_mon"], one["foe_mon"])
    assert np.array_equal(actor_only["cand_move"][0, 0], one["cand_move"][0, 1])
    assert np.array_equal(actor_only["y_set"][0, 1], one["y_set"][0, 0])
    a_row, b_row = one["act_mon"][0]
    assert actor_only["mon_cat"][0, a_row, F.CAT_SLOT] == F.SLOT_B
    assert actor_only["mon_cat"][0, b_row, F.CAT_SLOT] == F.SLOT_A
    other_only = M.swap_slots(one, actor=False, other=True)
    assert other_only["foe_mon"][0].tolist() == one["foe_mon"][0, ::-1].tolist()
    assert np.array_equal(other_only["act_mon"], one["act_mon"])
    assert np.array_equal(other_only["y_attack"], one["y_attack"][..., ::-1])
    legal = F.expand_target_mask(one["cand_tmask"])
    flipped = F.expand_target_mask(other_only["cand_tmask"])
    assert np.array_equal(flipped[..., F.T_FOE_A], legal[..., F.T_FOE_B])
    assert np.array_equal(flipped[..., F.T_FOE_B], legal[..., F.T_FOE_A])
    assert np.array_equal(flipped[..., F.T_ALLY :], legal[..., F.T_ALLY :])


def test_swap_slots_per_example_flags(batch: F.Batch):
    n = batch["act_mon"].shape[0]
    rng = np.random.default_rng(5)
    actor, other = rng.random(n) < 0.5, rng.random(n) < 0.5
    assert actor.any() and other.any() and not actor.all() and not other.all()
    mixed = M.swap_slots(batch, actor, other)
    for row in range(n):
        one = F.take(batch, slice(row, row + 1))
        want = M.swap_slots(one, actor=bool(actor[row]), other=bool(other[row]))
        for name in batch:
            assert np.array_equal(mixed[name][row], want[name][0]), (row, name)
    assert all(mixed[name].dtype == batch[name].dtype for name in batch)
    # A prediction of the mirrored batch is still well-formed for its labels.
    scores = F.slot_nll(F.uniform_prediction(mixed), mixed)
    plain = F.slot_nll(F.uniform_prediction(batch), batch)
    assert scores["fine"].sum() == pytest.approx(plain["fine"].sum())
    assert scores["target_scored"].sum() == plain["target_scored"].sum()


def test_swap_slots_never_raises(batch: F.Batch):
    M.COUNTERS.clear()
    assert M.swap_slots({}) == {}
    example = {name: array[0] for name, array in batch.items()}  # not a batch
    back = M.swap_slots(example)
    assert all(back[name] is example[name] for name in example)
    assert sum(M.COUNTERS.values()) == 1
    odd = M.swap_slots(batch, actor=np.zeros(3, dtype=bool) | True, other=False)
    assert all(odd[name] is batch[name] for name in batch)  # flags of the wrong size
    assert sum(M.COUNTERS.values()) == 2
    M.COUNTERS.clear()


def test_mirrored_training_view_keeps_the_prior_consistent(
    fz: F.Featurizer, batch: F.Batch
):
    """The untrained network is the prior, so its output mirrors with the batch."""
    predictor = M.OppNetPredictor(small_net(fz), fz)
    plain = predictor.predict(batch)
    mirrored = predictor.predict(M.swap_slots(batch, True, True))
    assert np.allclose(mirrored["action"], plain["action"][:, ::-1], atol=1e-6)


# --- predictor ----------------------------------------------------------------


class Recording(dict):
    """A batch that remembers which arrays were read."""

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.read: set[str] = set()

    def __getitem__(self, key: str) -> Any:
        self.read.add(key)
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        self.read.add(key)
        return super().get(key, default)


def test_predictor_reads_no_label_and_no_meta_array(fz: F.Featurizer, batch: F.Batch):
    data = dict(batch)
    data["m_weight"] = np.ones(batch["act_mon"].shape[0], dtype=np.float32)
    data["m_split"] = np.zeros(batch["act_mon"].shape[0], dtype=np.uint8)
    recording = Recording(data)
    predictor = M.OppNetPredictor(jolt(small_net(fz)), fz)
    pred = predictor.predict(recording)
    assert recording.read and not predictor.counters
    assert not [name for name in recording.read if name.startswith(("y_", "m_"))]
    stripped = M.strip_labels(data)
    assert not [name for name in stripped if name.startswith(("y_", "m_"))]
    assert set(M.FEATURE_KEYS) <= set(stripped)
    again = predictor.predict(stripped)
    assert all(np.array_equal(pred[name], again[name]) for name in pred)
    # Changing every label changes nothing.
    scrambled = dict(data)
    for name in data:
        if name.startswith("y_"):
            scrambled[name] = np.zeros_like(data[name])
    third = predictor.predict(scrambled)
    assert all(np.array_equal(pred[name], third[name]) for name in pred)


def test_predictor_never_raises_on_a_malformed_batch(fz: F.Featurizer, batch: F.Batch):
    predictor = M.OppNetPredictor(jolt(small_net(fz)), fz)
    good = predictor.predict(batch)
    assert not predictor.counters
    assert predictor.predict({}) == {}
    assert predictor.predict(None) == {}  # type: ignore[arg-type]
    assert predictor.counters["predict_error:KeyError"] == 1
    assert predictor.counters["predict_error:TypeError"] == 1

    uniform = F.uniform_prediction(batch)
    missing = {name: array for name, array in batch.items() if name != "mon_id"}
    truncated = dict(batch)
    truncated["cand_move"] = batch["cand_move"][:, :, :5]
    ragged = dict(batch)
    ragged["mon_hp"] = batch["mon_hp"][:3]
    wrong_type = dict(batch)
    wrong_type["mon_boost"] = np.array([["x"]])
    for broken in (missing, truncated, ragged, wrong_type):
        before = sum(predictor.counters.values())
        pred = predictor.predict(broken)
        assert sum(predictor.counters.values()) == before + 1
        assert set(pred) == {"action", "target", "mega"}
        assert np.array_equal(pred["action"], uniform["action"])
        F.normalize_prediction(pred, batch)

    # An empty batch is not an error: empty arrays of the right shape.
    before = dict(predictor.counters)
    nothing = predictor.predict(F.take(batch, slice(0, 0)))
    assert nothing["action"].shape == (0, 2, fz.n_actions)
    assert nothing["target"].shape == (0, 2, fz.n_cand + 1, F.N_TARGET)
    assert nothing["mega"].shape == (0, 2) and dict(predictor.counters) == before

    # Ids no table holds read row 0 instead of failing.
    huge = dict(batch)
    for name in ("mon_id", "mon_move", "cand_move"):
        huge[name] = np.full_like(batch[name], 60_000)
    for name in ("mon_cat", "mon_type", "mon_vol", "ctx_cat", "side_cnt"):
        huge[name] = np.full_like(batch[name], 250)
    huge["turn"] = np.full_like(batch["turn"], 255)
    huge["elo"] = np.full_like(batch["elo"], 32_000)
    before = dict(predictor.counters)
    pred = predictor.predict(huge)
    assert dict(predictor.counters) == before
    assert np.isfinite(pred["action"]).all()
    assert (pred["action"][~batch["action_mask"].astype(bool)] == 0).all()

    # A network that produces NaN degrades to the uniform answer.
    sick = small_net(fz)
    with torch.no_grad():
        sick.mon_proj.weight.fill_(float("nan"))
    sick_predictor = M.OppNetPredictor(sick, fz)
    pred = sick_predictor.predict(batch)
    assert sick_predictor.counters["predict_error:ValueError"] == 1
    assert np.array_equal(pred["action"], uniform["action"])
    # ... and the healthy predictor still answers as before.
    again = predictor.predict(batch)
    assert all(np.array_equal(good[name], again[name]) for name in good)


def test_unknown_sheet_code_is_read_as_closed(fz: F.Featurizer, batch: F.Batch):
    predictor = M.OppNetPredictor(jolt(small_net(fz)), fz)
    closed = dict(batch)
    flags = batch["game_flag"].copy()
    flags[:, [F.G_ACTOR_SHEET, F.G_OTHER_SHEET]] = F.SHEET_CLOSED
    closed["game_flag"] = flags
    unknown = dict(batch)
    flags = flags.copy()
    flags[:, [F.G_ACTOR_SHEET, F.G_OTHER_SHEET]] = F.SHEET_UNKNOWN
    unknown["game_flag"] = flags
    want = predictor.predict(closed)
    assert not predictor.counters
    got = predictor.predict(unknown)
    assert predictor.counters["sheet_unknown_as_closed"] == 1
    assert all(np.array_equal(want[name], got[name]) for name in want)
    assert (unknown["game_flag"][:, F.G_ACTOR_SHEET] == F.SHEET_UNKNOWN).all()
    opened = dict(batch)
    flags = flags.copy()
    flags[:, [F.G_ACTOR_SHEET, F.G_OTHER_SHEET]] = F.SHEET_OPEN
    opened["game_flag"] = flags
    assert not np.allclose(
        predictor.predict(opened)["action"], want["action"], atol=1e-5
    )


def test_predictor_leaves_the_training_mode_as_it_found_it(
    fz: F.Featurizer, batch: F.Batch
):
    net = jolt(small_net(fz, dropout=0.5)).train()
    predictor = M.OppNetPredictor(net, fz)
    first = predictor.predict(batch)
    second = predictor.predict(batch)
    assert net.training  # restored for the trainer
    assert all(
        np.array_equal(first[name], second[name]) for name in first
    )  # no dropout
    net.eval()
    predictor.predict(batch)
    assert not net.training
    outputs = M.collect_outputs(net.train(), batch, batch_size=4)
    assert net.training and outputs["action"].shape[0] == batch["act_mon"].shape[0]
    whole = M.probabilities(outputs)["action"].numpy().astype(np.float32)
    assert np.allclose(whole, first["action"], atol=1e-6)
    chunked = M.OppNetPredictor(net, fz, batch_size=3).predict(batch)
    assert np.allclose(chunked["action"], first["action"], atol=1e-6)


def test_concurrent_predict_calls_agree(fz: F.Featurizer, batch: F.Batch):
    predictor = M.OppNetPredictor(jolt(small_net(fz)), fz, batch_size=4)
    want = predictor.predict(batch)
    results: list[dict[str, np.ndarray]] = []
    errors: list[BaseException] = []

    def work() -> None:
        try:
            for _ in range(5):
                results.append(predictor.predict(batch))
        except BaseException as exc:  # pragma: no cover - the test then fails
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not errors and len(results) == 20 and not predictor.counters
    for got in results:
        assert all(np.array_equal(want[name], got[name]) for name in want)


def test_temperatures_apply_per_head(fz: F.Featurizer, batch: F.Batch):
    net = jolt(small_net(fz))
    plain = M.OppNetPredictor(net, fz)
    hot = plain.with_temperatures(action=2.0, target=0.5)
    assert hot.net is net and hot.temperatures == {"action": 2.0, "target": 0.5}
    assert plain.temperatures == {"action": 1.0, "target": 1.0}
    base, tempered = plain.predict(batch), hot.predict(batch)
    out = M.collect_outputs(net, batch)
    mask = out["action_mask"]
    want = torch.softmax((out["action"] / 2.0).masked_fill(~mask, M.NEG), -1) * mask
    assert np.allclose(tempered["action"], want.numpy(), atol=1e-6)
    assert not np.allclose(tempered["action"], base["action"], atol=1e-3)
    assert not np.allclose(tempered["target"], base["target"], atol=1e-3)
    assert np.array_equal(tempered["mega"], base["mega"])
    active = batch["act_mon"] >= 0
    entropy = lambda p: -(p * np.log(np.maximum(p, 1e-12))).sum(-1)[active].mean()  # noqa: E731
    assert entropy(tempered["action"]) > entropy(base["action"])
    # A nonsense temperature is ignored rather than served.
    for bad in (0.0, -1.0, float("nan"), None, "x"):
        assert plain.with_temperatures(action=bad).temperatures["action"] == 1.0  # type: ignore[arg-type]


def test_mega_head_has_its_own_temperature_and_bias(fz: F.Featurizer, batch: F.Batch):
    net = jolt(small_net(fz))
    with torch.no_grad():  # a Mega head that says something
        net.mega.out.weight.normal_(
            0.0, 1.0, generator=torch.Generator().manual_seed(4)
        )
        net.mega.out.bias.fill_(0.7)
    plain = M.OppNetPredictor(net, fz)
    assert plain.mega_calibration == {"temperature": 1.0, "bias": 0.0}
    cool = plain.with_temperatures(mega=2.0, mega_bias=-0.5)
    assert cool.mega_calibration == {"temperature": 2.0, "bias": -0.5}
    assert cool.temperatures == {"action": 1.0, "target": 1.0}
    base, moved = plain.predict(batch), cool.predict(batch)
    out = M.collect_outputs(net, batch)
    can = out["mega_mask"].numpy()
    assert can.any() and not can.all()
    want = torch.sigmoid(out["mega"] / 2.0 - 0.5).numpy()
    assert np.allclose(moved["mega"][can], want[can], atol=1e-6)
    assert not np.allclose(moved["mega"][can], base["mega"][can], atol=1e-3)
    # Still exactly zero where the public state rules a Mega out.
    assert not moved["mega"][~can].any() and not base["mega"][~can].any()
    # The other two heads do not move.
    assert np.array_equal(moved["action"], base["action"])
    assert np.array_equal(moved["target"], base["target"])
    # The loss sees the same calibrated logit as the probability.
    labels = M.to_labels(batch)
    terms = M.nll_terms(out, labels, mega_temperature=2.0, mega_bias=-0.5)
    scored = terms["mega_scored"].numpy()
    assert scored.any()
    y = labels["y_mega"].numpy() == 1
    p = np.clip(want, 1e-12, 1 - 1e-12)
    by_hand = -np.where(y, np.log(p), np.log(1 - p))
    assert np.allclose(terms["mega"].numpy()[scored], by_hand[scored], atol=1e-5)
    nll = F.slot_nll(moved, batch)
    assert np.array_equal(nll["mega_scored"], scored)
    assert np.allclose(nll["mega"][scored], by_hand[scored], atol=1e-5)
    # Stored and read back; a payload written before it existed reads 1 and 0.
    payload = cool.to_payload()
    assert payload["temperatures"] == {
        "action": 1.0,
        "target": 1.0,
        "mega": 2.0,
        "mega_bias": -0.5,
    }
    again = M.from_payload(payload, fz)
    assert again.mega_calibration == {"temperature": 2.0, "bias": -0.5}
    assert np.array_equal(again.predict(batch)["mega"], moved["mega"])
    old = dict(payload, temperatures={"action": 1.0, "target": 1.0})
    before = M.from_payload(old, fz)
    assert before.mega_calibration == {"temperature": 1.0, "bias": 0.0}
    assert np.array_equal(before.predict(batch)["mega"], base["mega"])
    # Nonsense is ignored rather than served.
    for bad in (0.0, -1.0, float("nan"), None, "x"):
        twin = plain.with_temperatures(mega=bad, mega_bias=bad)  # type: ignore[arg-type]
        assert twin.mega_temperature == 1.0
        assert twin.mega_bias in (0.0, -1.0)


def test_fit_offset_finds_the_shift_and_keeps_zero_otherwise():
    assert M.fit_offset(lambda b: (b - 1.25) ** 2) == pytest.approx(1.25, abs=1e-4)
    assert M.fit_offset(lambda b: (b + 0.4) ** 2) == pytest.approx(-0.4, abs=1e-4)
    assert M.fit_offset(lambda b: b**2) == 0.0  # zero is already the best
    assert M.fit_offset(lambda b: float("nan")) == 0.0
    assert M.fit_offset(lambda b: (b - 9.0) ** 2, low=-4.0, high=4.0) == pytest.approx(
        4.0, abs=1e-3
    )


def test_count_masked_targets_is_where_the_loss_and_slot_nll_part(batch: F.Batch):
    """``nll_terms`` skips a target label that the legal mask excludes;
    ``features.slot_nll`` charges it. The counter finds exactly those."""
    assert M.count_masked_targets(batch) == 0
    labels, legal = M.to_labels(batch), F.expand_target_mask(batch["cand_tmask"])
    aimed = np.argwhere((batch["y_target"] >= 0) & (batch["y_action"] >= 0))
    assert len(aimed) >= 2
    spoiled = {name: array.copy() for name, array in batch.items()}
    for row, slot in aimed[:2]:
        action = int(batch["y_action"][row, slot])
        illegal = np.flatnonzero(~legal[row, slot, action])
        assert illegal.size
        spoiled["y_target"][row, slot] = illegal[0]
    assert M.count_masked_targets(spoiled) == 2
    out = random_logits(batch, 5)
    ours = M.nll_terms(out, M.to_labels(spoiled))["target_scored"].numpy()
    pred = {k: v.numpy() for k, v in M.probabilities(out).items()}
    theirs = F.slot_nll(pred, spoiled)["target_scored"]
    assert int((theirs & ~ours).sum()) == 2 and not (ours & ~theirs).any()
    assert labels["y_target"].shape == batch["y_target"].shape
    with pytest.raises(KeyError):
        M.count_masked_targets(M.strip_labels(batch))


def test_fit_temperature_finds_the_scale_of_overconfident_logits(batch: F.Batch):
    out = random_logits(batch, 11)
    labels = M.to_labels(batch)
    # Labels drawn from the logits' own distribution, then the logits sharpened.
    probs = M.probabilities(out)["action"]
    generator = torch.Generator().manual_seed(3)
    y_set = torch.zeros_like(labels["y_set"])
    big = {name: value.repeat_interleave(200, 0) for name, value in out.items()}
    draws = torch.multinomial(
        probs.repeat_interleave(200, 0).flatten(0, 1).clamp(min=1e-12),
        1,
        generator=generator,
    ).view(-1, 2)
    y_set = torch.zeros_like(big["action_mask"]).scatter(2, draws[..., None], True)
    y_set &= big["action_mask"]
    big_labels = {
        "y_set": y_set,
        "y_action": torch.full(draws.shape, -1),
        "y_target": torch.full(draws.shape, -1),
        "y_mega": torch.full(draws.shape, -1),
    }
    sharp = dict(big)
    sharp["action"] = big["action"] * 3.0

    def objective(temperature: float) -> float:
        terms = M.nll_terms(sharp, big_labels, action_temperature=temperature)
        return float(terms["action"][terms["action_scored"]].mean())

    found = M.fit_temperature(objective)
    assert found == pytest.approx(3.0, rel=0.1)
    assert objective(found) < objective(1.0) - 0.1
    # An objective that 1.0 already minimises, or a broken one, gives 1.0.
    assert M.fit_temperature(lambda t: (np.log(t)) ** 2) == 1.0
    assert M.fit_temperature(lambda t: float("nan")) == 1.0
    assert M.fit_temperature(lambda t: (np.log(t) - np.log(0.5)) ** 2) == pytest.approx(
        0.5, rel=1e-3
    )


# --- payload ------------------------------------------------------------------


def plain_data(value: Any) -> bool:
    if isinstance(value, dict):
        return all(isinstance(k, str) and plain_data(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(plain_data(item) for item in value)
    return value is None or isinstance(value, (str, int, float, bool, torch.Tensor))


def test_payload_round_trip_gives_identical_probabilities(
    fz: F.Featurizer, batch: F.Batch
):
    net = jolt(small_net(fz))
    predictor = M.OppNetPredictor(
        net, fz, name="unit", action_temperature=1.3, target_temperature=0.8
    )
    payload = predictor.to_payload()
    assert plain_data(payload)
    assert payload["config"] == net.config.to_dict()
    assert payload["n_parameters"] == net.n_parameters()
    assert not any(
        "species_num" in key or "move_num" in key for key in payload["state_dict"]
    )
    again = M.from_payload(payload, fz)
    assert isinstance(again, F.Predictor)
    assert (again.kind, again.name) == ("oppnet", "unit")
    assert again.temperatures == {"action": 1.3, "target": 0.8}
    assert again.elo_mode == F.ELO_KEEP and not again.net.training
    want, got = predictor.predict(batch), again.predict(batch)
    for name in want:
        assert np.array_equal(want[name], got[name]), name
    # The payload is a copy: training the source afterwards does not move it.
    with torch.no_grad():
        net.act_cand.out.weight.add_(1.0)
    still = M.from_payload(payload, fz).predict(batch)
    assert all(np.array_equal(got[name], still[name]) for name in got)
    assert not np.allclose(predictor.predict(batch)["action"], got["action"], atol=1e-4)

    blind = M.OppNetPredictor(net, fz, elo_mode=F.ELO_BLANK).to_payload()
    assert blind["elo_mode"] == F.ELO_BLANK
    assert M.from_payload(blind, fz).elo_mode == F.ELO_BLANK
    # An Elo-blind featurizer wins over a payload that says keep.
    blind_fz = F.Featurizer.build(repertoire(), elo_mode=F.ELO_BLANK)
    assert M.from_payload(payload, blind_fz).elo_mode == F.ELO_BLANK


def test_payload_survives_torch_save_and_a_rebuilt_featurizer(
    tmp_path: Path, fz: F.Featurizer, batch: F.Batch
):
    predictor = M.OppNetPredictor(jolt(small_net(fz)), fz, name="saved")
    path = tmp_path / "payload.pt"
    torch.save(predictor.to_payload(), path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    rebuilt = F.Featurizer.from_payload(fz.to_payload())
    again = M.from_payload(payload, rebuilt)
    want, got = predictor.predict(batch), again.predict(batch)
    assert all(np.array_equal(want[name], got[name]) for name in want)


def test_bad_payloads_raise_value_error(fz: F.Featurizer):
    payload = M.OppNetPredictor(small_net(fz), fz).to_payload()
    for change in (
        {"format": "something-else"},
        {"version": 99},
        {"config": {**payload["config"], "n_cand": 5}},
        {"config": {**payload["config"], "d_model": 33}},
        {"config": {**payload["config"], "move_width": 3}},
        {"config": "nonsense"},
        {"state_dict": {}},
        {"state_dict": None},
    ):
        with pytest.raises(ValueError):
            M.from_payload({**payload, **change}, fz)
    with pytest.raises(ValueError):
        M.from_payload({}, fz)
    with pytest.raises(ValueError):
        M.OppNetConfig.for_featurizer(fz, d_model=30, n_heads=4).check()
    with pytest.raises(ValueError):
        M.OppNet(M.OppNetConfig(), np.zeros((2, 3)), np.zeros((2, 3)))
    config = M.OppNetConfig.for_featurizer(fz, **SMALL)
    assert M.OppNetConfig.from_dict({**config.to_dict(), "unknown_field": 1}) == config


def test_artifact_round_trip_gives_identical_probabilities(
    tmp_path: Path, fz: F.Featurizer, batch: F.Batch
):
    if importlib.util.find_spec(ARTIFACT_MODULE) is None:
        pytest.skip("oppmodel.artifact is not written yet")
    artifact = pytest.importorskip(ARTIFACT_MODULE)
    predictor = M.OppNetPredictor(
        jolt(small_net(fz)), fz, name="unit", action_temperature=1.2
    )
    path = tmp_path / "artifact.pt"
    artifact.save_artifact(
        path,
        kind=M.KIND,
        name="unit",
        featurizer=fz,
        predictor_payload=predictor.to_payload(),
        extra={"note": "unit test"},
    )
    stored = artifact.read_artifact(path)
    assert stored["kind"] == "oppnet" and stored["name"] == "unit"
    loaded = artifact.load_predictor(path)
    assert (loaded.kind, loaded.name) == ("oppnet", "unit")
    assert isinstance(loaded.predictor, M.OppNetPredictor)
    assert loaded.predictor.temperatures["action"] == pytest.approx(1.2)
    want, got = predictor.predict(batch), loaded.predictor.predict(batch)
    for name in want:
        assert np.array_equal(want[name], got[name]), name


def test_ids_of_extension_rows_keep_their_numerics(batch: F.Batch):
    """A vocabulary older than the dex: new ids read row 0 of the embeddings but
    their own row of the numeric tables."""
    full = F.Featurizer.build(repertoire())
    payload = full.to_payload()
    dropped_move = "closecombat"
    keep = [i for i, m in enumerate(payload["vocab"]["moves"]) if m != dropped_move]
    payload["vocab"] = {
        **payload["vocab"],
        "moves": [m for m in payload["vocab"]["moves"] if m != dropped_move],
    }
    tables = payload["tables"]
    payload["tables"] = {
        **tables,
        "move_num": tables["move_num"][keep],
        "move_intent": tables["move_intent"][keep],
        "move_class": tables["move_class"][keep],
    }
    old = F.Featurizer.from_payload(payload)
    assert old.tables.move_num.shape[0] == old.n_moves + 1  # one extension row
    net = small_net(old)
    assert net.emb_move.num_embeddings == old.n_moves
    assert net.move_num.shape[0] == old.n_moves + 1
    assert int(net.move_rows[-1]) == 0 and int(net.move_rows[-2]) == old.n_moves - 1
    data = game_batch(old)
    assert (data["cand_move"] == old.n_moves).any()  # the extension id is in use
    table = net._move_table()
    assert not torch.allclose(table[-1], table[0])  # its numerics, not row 0's
    pred = M.OppNetPredictor(jolt(net), old).predict(data)
    F.normalize_prediction(pred, data)
    # The state dict does not depend on the extension rows.
    rebuilt = M.from_payload(M.OppNetPredictor(net, old).to_payload(), old)
    assert rebuilt.net.move_num.shape == net.move_num.shape


def test_clone_state_is_a_real_copy(fz: F.Featurizer):
    net = small_net(fz)
    alias = {name: value.detach().cpu() for name, value in net.state_dict().items()}
    snapshot = M.clone_state(net)
    with torch.no_grad():
        net.emb_species.weight.add_(1.0)
    name = "emb_species.weight"
    assert torch.equal(alias[name], net.state_dict()[name])  # the old trainers' bug
    assert not torch.equal(snapshot[name], net.state_dict()[name])
    assert set(snapshot) == set(net.state_dict())


# --- library rules ------------------------------------------------------------


def library_paths() -> list[Path]:
    return [
        ROOT / "vgc_bench/src/oppmodel/model.py",
        ROOT / "training/train_oppmodel.py",
    ]


def test_library_holds_no_species_move_item_or_ability_name(fz: F.Featurizer):
    """String literals of the model and the trainer name nothing from the dex."""
    known = set(E.moves_dex()) | set(E.pokedex())
    known |= set(fz.vocab.items[3:]) | set(fz.vocab.abilities[3:])
    for path in library_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            )
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        }
        found = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and node.value in known
        }
        assert not found, (path.name, sorted(found))


def test_library_has_no_bare_assert():
    for path in library_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        lines = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert)]
        assert not lines, (path.name, lines)
