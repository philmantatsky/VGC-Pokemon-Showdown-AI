"""Layout version 2 of the opponent predictor (the matchup arrays).

What must hold: a version-1 featurizer and a network without the new config
fields are exactly what they were (goldens made from the tree before the
change); a version-2 example holds every version-1 array bit for bit; the new
arrays follow the slot mirror; a network that reads them starts as the
network that does not; and every wrong pairing is refused or counted.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from datagen import oppmodel_build_dataset as B
from training import train_oppmodel as T
from unit_tests import test_oppmodel_dataset as TD
from unit_tests import test_oppmodel_features as TF
from unit_tests import test_oppmodel_model as TM
from unit_tests import test_oppmodel_train as TT
from vgc_bench.src.oppmodel import events as E
from vgc_bench.src.oppmodel import features as F
from vgc_bench.src.oppmodel import matchup as MU
from vgc_bench.src.oppmodel import model as M
from vgc_bench.src.oppmodel import public_state as P

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_V1 = ROOT / "unit_tests" / "data" / "oppmodel_golden_v1"
GOLDEN_LIVE = ROOT / "results_oppmodel/analysis_20261010/golden_oppnet_v2_blind"
LIVE_ARTIFACT = ROOT / "results_oppmodel/oppnet_v2_blind/artifact.pt"
MU_NAMES = ("mu_cand", "mu_slot", "mu_roster")
ALL = 7  # the three matchup blocks of the trainer's bit mask


@pytest.fixture(scope="module")
def fz1() -> F.Featurizer:
    return F.Featurizer.build(TM.repertoire())


@pytest.fixture(scope="module")
def fz2() -> F.Featurizer:
    # The three matchup arrays by name: a later row of ``F.EXTRA_ARRAYS``
    # (the reserved ``sp_cand``) does not change what these tests build.
    return F.Featurizer.build(TM.repertoire(), layout_version=2, extras=MU_NAMES)


@pytest.fixture(scope="module")
def batch1(fz1: F.Featurizer) -> F.Batch:
    return TM.game_batch(fz1)


@pytest.fixture(scope="module")
def batch2(fz2: F.Featurizer) -> F.Batch:
    return TM.game_batch(fz2)


def golden(name: str) -> Any:
    path = GOLDEN_V1 / name
    if not path.is_file():
        pytest.skip(f"golden fixture {path} is not here")
    if name.endswith(".json"):
        return json.loads(path.read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def logits(net: M.OppNet, batch: F.Batch) -> dict[str, torch.Tensor]:
    out = M.collect_outputs(net, batch)
    return {name: out[name] for name in ("action", "target", "mega")}


def matchup_net(fz: F.Featurizer, mask: int = ALL, **overrides: Any) -> M.OppNet:
    return TM.small_net(fz, **{**M.extras_overrides(fz, mask), **overrides})


def fill_extras(net: M.OppNet, seed: int = 5) -> M.OppNet:
    """Random values in the zero-initialised version-2 matrices."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, value in net.named_parameters():
            if name.startswith(("mu_", "sp_")):
                value.copy_(torch.randn(value.shape, generator=generator))
    return net


# --- old artifacts and the layout -------------------------------------------------


def test_default_featurizer_reproduces_the_golden_encode(
    fz1: F.Featurizer, batch1: F.Batch
):
    """G1: arrays the code wrote BEFORE layout version 2 existed, byte for byte."""
    want, meta = golden("encode.npz"), golden("meta.json")
    assert set(batch1) == set(want)
    for name, array in want.items():
        assert batch1[name].dtype == array.dtype, name
        assert np.array_equal(batch1[name], array), name
    assert sorted(fz1.to_payload()) == meta["payload_keys"]
    assert fz1.layout_version == 1 and fz1.extras == () and fz1.matchup_version == 0
    assert fz1.matchup_signature() is None
    assert not any(name in fz1.layout() for name in MU_NAMES)
    assert F.LAYOUT_VERSION == 1


def test_the_v1_network_reproduces_the_golden_outputs(
    fz1: F.Featurizer, batch1: F.Batch
):
    """G2: the network rebuilt under the same seeds gives the stored logits.

    No weight is stored: equal outputs also show that construction draws the
    same random numbers as before the change.
    """
    want, meta = golden("outputs.npz"), golden("meta.json")
    net = TM.jolt(TM.small_net(fz1))
    assert sorted(net.state_dict()) == meta["state_keys"]
    assert net.n_parameters() == meta["n_parameters"]
    assert net.extra_keys == () and net.matchup_keys == ()
    made = {f"logit__{k}": v.numpy() for k, v in logits(net, batch1).items()}
    probs = M.OppNetPredictor(net, fz1).predict(batch1)
    made.update({f"prob__{k}": np.asarray(v) for k, v in probs.items()})
    assert set(made) == set(want)
    exact = torch.__version__ == meta["torch"]
    for name, array in want.items():
        if exact:
            assert np.array_equal(made[name], array), name
        else:
            assert np.allclose(made[name], array, atol=1e-6), (
                name,
                f"torch {torch.__version__} is not the golden's {meta['torch']}: "
                "compared with atol 1e-6, not bit for bit",
            )


def test_the_live_artifact_still_gives_its_golden_predictions():
    """G3, machine-local: the artifact the bot runs, on rows encoded at 01:14.

    The full check (every array the featurizer encodes, the stored holdout
    rows, the forecast hash) is the golden's own script; this is its fast
    part, as a batch of one like the runtime calls it. Skipped when the
    artifact or the golden is not on this machine.
    """
    if not LIVE_ARTIFACT.is_file() or not (GOLDEN_LIVE / "golden.npz").is_file():
        pytest.skip("results_oppmodel/oppnet_v2_blind or its golden is not here")
    spec = importlib.util.find_spec(TM.ARTIFACT_MODULE)
    if spec is None:
        pytest.skip("oppmodel.artifact is not here")
    artifact = importlib.import_module(TM.ARTIFACT_MODULE)
    torch.set_num_threads(1)
    loaded = artifact.load_predictor(LIVE_ARTIFACT)
    assert loaded.featurizer.layout_version == 1 and loaded.featurizer.extras == ()
    assert loaded.predictor.net.extra_keys == ()
    assert loaded.predictor.net.config.extra_widths() == {}
    rows = 160
    with np.load(GOLDEN_LIVE / "golden.npz", allow_pickle=False) as data:
        batch = {
            name[len("feat__") :]: data[name][:rows]
            for name in data.files
            if name.startswith("feat__")
        }
        want = {head: data[f"single__{head}"][:rows] for head in ("action", "mega")}
    batch = F.sheet_unknown_as_closed(batch)
    got: dict[str, list[np.ndarray]] = {head: [] for head in want}
    for row in range(rows):
        one = {name: array[row : row + 1] for name, array in batch.items()}
        made = loaded.predictor.predict(one)
        for head in want:
            got[head].append(np.asarray(made[head]))
    assert not loaded.predictor.counters
    for head, array in want.items():
        assert np.array_equal(np.concatenate(got[head]), array), head


def test_every_v1_array_is_unchanged_in_a_v2_example(
    fz1: F.Featurizer, fz2: F.Featurizer
):
    compared = 0
    for record in TM.drive().turns:
        for side in E.SIDES:
            old = fz1.encode_turn(record.snapshot, record.actions, side)
            new = fz2.encode_turn(record.snapshot, record.actions, side)
            assert old is not None and new is not None
            assert set(new) - set(old) == set(MU_NAMES)
            features = fz2.encode(record.snapshot, side)
            assert features is not None
            # Made after every version-1 feature array.
            assert list(features)[-3:] == list(MU_NAMES)
            assert list(features)[:-3] == list(fz1.encode(record.snapshot, side) or {})
            for name, array in old.items():
                assert new[name].dtype == array.dtype, name
                assert new[name].tobytes() == array.tobytes(), (record.turn, name)
            compared += 1
    assert compared == 14 and not fz2.counters


def test_payload_versions(fz1: F.Featurizer, fz2: F.Featurizer, tmp_path: Path):
    old, new = fz1.to_payload(), fz2.to_payload()
    assert set(new) - set(old) == {"extras", "matchup"}
    assert old["layout_version"] == 1 and new["layout_version"] == 2
    assert new["extras"] == list(MU_NAMES)
    assert new["matchup"]["version"] == MU.MATCHUP_VERSION == fz2.matchup_version
    json.dumps(new["matchup"])  # plain data: it goes into vocab.json
    assert set(new["matchup"]) == {
        "version",
        "columns",
        "assumptions",
        "rules",
        "type_chart_crc32",
    }
    assert new["matchup"]["rules"]["speed_double"] == ["tailwind"]
    for fz, payload in ((fz1, old), (fz2, new)):
        again = F.Featurizer.from_payload(payload, strict=True)
        assert again.layout_version == fz.layout_version
        assert again.extras == fz.extras and again.signature_diff == []
        assert list(again.layout()) == list(fz.layout())
        fz.save(tmp_path / f"v{fz.layout_version}")
        loaded = F.Featurizer.load(tmp_path / f"v{fz.layout_version}", strict=True)
        assert loaded.layout_version == fz.layout_version
        assert loaded.extras == fz.extras
        assert sorted(loaded.to_payload()) == sorted(payload)
    for version in (0, 3, 99):
        with pytest.raises(ValueError):
            F.Featurizer.from_payload({**old, "layout_version": version})
    # A version-2 payload must say which arrays it wrote, and only known ones.
    with pytest.raises(ValueError):
        F.Featurizer.from_payload({k: v for k, v in new.items() if k != "extras"})
    with pytest.raises(ValueError):
        F.Featurizer.from_payload({**new, "extras": ["mu_cand", "not_an_array"]})
    # Matchup definitions other than this code's: left in signature_diff
    # (the runtime refuses to serve on it), an error under ``strict``.
    changed = {**new, "matchup": {**new["matchup"], "version": 99}}
    assert F.Featurizer.from_payload(changed).signature_diff == ["matchup"]
    with pytest.raises(ValueError):
        F.Featurizer.from_payload(changed, strict=True)
    missing = {k: v for k, v in new.items() if k != "matchup"}
    assert F.Featurizer.from_payload(missing).signature_diff == ["matchup"]
    # Construction: the extras exist only at version 2.
    with pytest.raises(ValueError):
        F.Featurizer.build(TM.repertoire(), layout_version=3)
    with pytest.raises(ValueError):
        F.Featurizer.build(TM.repertoire(), extras=("mu_cand",))
    with pytest.raises(ValueError):
        F.Featurizer.build(TM.repertoire(), layout_version=2, extras=("nope",))


def test_v2_layout_shapes_dtypes_and_ranges(fz2: F.Featurizer, batch2: F.Batch):
    spec = fz2.layout()
    assert list(spec)[-3:] == list(MU_NAMES)
    assert spec["mu_cand"].shape == (2, 12, 2, 8)
    assert spec["mu_slot"].shape == (2, 2, 10)
    assert spec["mu_roster"].shape == (6, 2, 4)
    for record in TM.drive().turns:
        example = fz2.encode_turn(record.snapshot, record.actions, "p1")
        assert example is not None and set(example) == set(spec)
        for name in MU_NAMES:
            assert example[name].shape == spec[name].shape
            assert example[name].dtype == np.dtype(spec[name].dtype) == np.int8
    cand, slot, roster = (batch2[name].astype(int) for name in MU_NAMES)
    col = {name: i for i, name in enumerate(MU.MU_CAND_COLUMNS)}
    for flag in ("est", "immune", "kill", "kill_first"):
        assert set(np.unique(cand[..., col[flag]])) <= {0, 1}
    assert cand[..., col["eff"]].min() >= -3 and cand[..., col["eff"]].max() <= 3
    for share in ("dmg", "ko"):
        assert cand[..., col[share]].min() >= 0 and cand[..., col[share]].max() <= 100
    assert set(np.unique(cand[..., col["first"]])) <= {-1, 0, 1}
    assert (cand[..., col["kill_first"]] <= cand[..., col["kill"]]).all()
    assert (cand[..., col["dmg"]][cand[..., col["est"]] == 0] == 0).all()
    assert (cand[..., col["dmg"]][cand[..., col["immune"]] == 1] == 0).all()
    col = {name: i for i, name in enumerate(MU.MU_SLOT_COLUMNS)}
    assert set(np.unique(slot[..., col["first"]])) <= {-1, 0, 1}
    assert set(np.unique(slot[..., col["first_sure"]])) <= {-1, 0, 1}
    assert abs(slot[..., col["spd"]]).max() <= 50
    for share in ("in_dmg", "in_ko", "out_dmg", "out_ko"):
        assert slot[..., col[share]].min() >= 0 and slot[..., col[share]].max() <= 100
    assert roster.min() >= 0 and roster.max() <= 100
    # Something was computed: the game has damage on both sides.
    assert cand[..., 3].max() > 0 and slot[..., 3].max() > 0 and roster.max() > 0
    # Zero wherever the slot, the candidate, the foe or the roster row is absent.
    absent_slot = batch2["act_mon"] < 0
    assert (cand[absent_slot] == 0).all() and (slot[absent_slot] == 0).all()
    no_foe = batch2["foe_mon"] < 0
    assert (np.swapaxes(slot, 1, 2)[no_foe] == 0).all()
    assert (np.swapaxes(roster, 1, 2)[no_foe] == 0).all()
    assert (np.moveaxis(cand, 3, 1)[no_foe] == 0).all()
    legal = batch2["switch_mask"].max(1) > 0  # [N, 6]
    assert (roster[~legal] == 0).all()
    invalid = (batch2["cand_flag"].astype(int) & F.CAND_VALID) == 0
    assert (cand[invalid] == 0).all()
    # A subset of the arrays can be asked for.
    only = F.Featurizer.build(TM.repertoire(), layout_version=2, extras=("mu_slot",))
    assert only.extras == ("mu_slot",) and only.matchup_version == MU.MATCHUP_VERSION
    example = only.encode(TM.drive().turns[0].snapshot, "p1")
    assert example is not None and "mu_slot" in example and "mu_cand" not in example
    assert np.array_equal(example["mu_slot"], batch2["mu_slot"][0])


def test_the_table_of_extra_arrays_is_consistent():
    bits = [spec.bit for spec in F.EXTRA_ARRAYS]
    assert bits[:3] == [1, 2, 4] and len(set(bits)) == len(bits)
    assert [spec.name for spec in F.EXTRA_ARRAYS[:3]] == list(MU_NAMES)
    assert all(bit > 0 and bit & (bit - 1) == 0 for bit in bits)  # one bit each
    # Bit 8 belongs to ``sp_cand``, whether or not its row exists yet.
    assert F.EXTRA_RESERVED_BITS.get(8) == "sp_cand"
    for spec in F.EXTRA_ARRAYS:
        assert F.EXTRA_RESERVED_BITS.get(spec.bit, spec.name) == spec.name
        assert spec.kind in F.EXTRA_KINDS
        assert len(spec.columns) == len(spec.scales) == spec.width
        assert spec.name in M.EXTRA_CONFIG_FIELDS
        assert not spec.name.startswith(("y_", "m_"))  # strip_labels would drop it
        assert F.extra_array(spec.name) is spec
    for name in F.EXTRA_RESERVED_BITS.values():
        assert name in M.EXTRA_CONFIG_FIELDS
    assert F.extra_array("nope") is None
    assert F.extra_names(["mu_roster", "mu_cand"]) == ("mu_cand", "mu_roster")


# --- the featurizer ---------------------------------------------------------------


def with_mon(
    snapshot: P.PublicSnapshot, side: str, index: int, **changes: Any
) -> P.PublicSnapshot:
    team = snapshot.sides[side]
    mons = tuple(
        replace(mon, **changes) if mon.index == index else mon for mon in team.mons
    )
    return replace(snapshot, sides={**snapshot.sides, side: replace(team, mons=mons)})


def mu(fz: F.Featurizer, snapshot: P.PublicSnapshot, side: str = "p2") -> F.Example:
    example = fz.encode(snapshot, side)
    assert example is not None
    return {name: example[name].astype(int) for name in MU_NAMES}


def same(a: F.Example, b: F.Example) -> bool:
    return all(np.array_equal(a[name], b[name]) for name in MU_NAMES)


def test_what_the_matchup_arrays_read(fz2: F.Featurizer):
    """Turn 2 of the hand-written game, actor p2 (Garchomp-Mega a, Sneasler b)
    against p1's Charizard (a, row 2) and Rillaboom (b, row 1)."""
    snapshot = TM.drive().turns[2].snapshot
    actor, foe = snapshot.sides["p2"], snapshot.sides["p1"]
    assert actor.active == {"a": 0, "b": 1} and foe.active == {"a": 2, "b": 1}
    base = mu(fz2, snapshot)
    # Not read: item, ability, volatiles, last action, protect flags, ratings, turn.
    for side, index in (("p1", 2), ("p1", 1), ("p2", 0), ("p2", 1)):
        quiet = with_mon(
            snapshot,
            side,
            index,
            item="choicescarf",
            item_state=P.ITEM_KNOWN,
            ability="levitate",
            ability_known=True,
            volatiles={"substitute": 1},
            last_action=None,
            protected_last_turn=True,
            protect_streak=2,
            first_turn=True,
            turns_on_field=5,
            mega_possible=True,
        )
        assert same(mu(fz2, quiet), base), (side, index)
    rated = replace(
        snapshot,
        turn=17,
        sides={
            name: replace(team, rating=1900, player="Somebody")
            for name, team in snapshot.sides.items()
        },
    )
    assert same(mu(fz2, rated), base)

    def moved(changed: F.Example) -> dict[str, np.ndarray]:
        return {name: changed[name] != base[name] for name in MU_NAMES}

    # Read: a foe's type. Only entries about foe slot a (Charizard) move.
    diff = moved(mu(fz2, with_mon(snapshot, "p1", 2, types=("water",))))
    assert diff["mu_cand"][:, :, 0].any() and not diff["mu_cand"][:, :, 1].any()
    assert diff["mu_slot"][:, 0].any() and not diff["mu_slot"][:, 1].any()
    assert not diff["mu_roster"][:, 1].any()
    # A foe's HP: the knock-out columns of that foe, nothing of the other.
    diff = moved(mu(fz2, with_mon(snapshot, "p1", 2, hp=1.0)))
    assert diff["mu_cand"][:, :, 0].any() and not diff["mu_cand"][:, :, 1].any()
    assert diff["mu_slot"][:, 0].any() and not diff["mu_slot"][:, 1].any()
    assert not diff["mu_roster"].any()
    # The actor's own stage: that slot's rows only.
    boosts = {**actor.mons[0].boosts, "atk": 2, "spe": -1}
    diff = moved(mu(fz2, with_mon(snapshot, "p2", 0, boosts=boosts)))
    assert diff["mu_cand"][0].any() and not diff["mu_cand"][1].any()
    assert diff["mu_slot"][0].any() and not diff["mu_slot"][1].any()
    assert not diff["mu_roster"].any()
    # Status: a burn on the actor's physical attacker, paralysis on a foe.
    diff = moved(mu(fz2, with_mon(snapshot, "p2", 0, status="brn")))
    assert diff["mu_cand"][0].any() and not diff["mu_cand"][1].any()
    diff = moved(mu(fz2, with_mon(snapshot, "p1", 2, status="par")))
    assert diff["mu_slot"][:, 0, :3].any() and not diff["mu_slot"][:, 1].any()
    # The weather, a side condition and a field, by the ids the tracker uses.
    assert not same(mu(fz2, replace(snapshot, weather="sunnyday")), base)
    start = P.EffectStart(1, False)
    wind = replace(actor, conditions={"tailwind": start})
    blown = mu(fz2, replace(snapshot, sides={**snapshot.sides, "p2": wind}))
    first = MU.MU_SLOT_COLUMNS.index("first")
    assert (blown["mu_slot"][..., first] == 1).all()  # doubled: all four pairs
    room = mu(fz2, replace(snapshot, fields={"trickroom": start}))
    assert np.array_equal(room["mu_slot"][..., :3], -base["mu_slot"][..., :3])
    assert not same(room, base)
    # A newly known move of a foe changes what it threatens, not what it takes.
    known = foe.mons[2].moves + (P.MovePublic("icebeam", True, False),)
    diff = moved(mu(fz2, with_mon(snapshot, "p1", 2, moves=known)))
    in_dmg = MU.MU_SLOT_COLUMNS.index("in_dmg")
    assert diff["mu_slot"][0, 0, in_dmg] and not diff["mu_slot"][:, 1].any()
    assert not diff["mu_slot"][:, :, 7:].any() and not diff["mu_cand"].any()
    assert not diff["mu_roster"][:, 1].any() and not diff["mu_roster"][:, :, 3].any()
    # ... and what it threatens on the actor's bench (Ground on Steel / Dragon).
    known = foe.mons[2].moves + (P.MovePublic("earthpower", True, False),)
    diff = moved(mu(fz2, with_mon(snapshot, "p1", 2, moves=known)))
    assert actor.mons[3].species == "archaludon"
    assert diff["mu_roster"][3, 0, 0] and not diff["mu_roster"][:, 1].any()
    assert not diff["mu_roster"][:, :, 3].any() and not diff["mu_cand"].any()


def test_matchup_never_costs_an_example(
    fz1: F.Featurizer, monkeypatch: pytest.MonkeyPatch
):
    fz = F.Featurizer.build(TM.repertoire(), layout_version=2, extras=MU_NAMES)
    snapshot = TM.drive().turns[2].snapshot
    want = fz1.encode(snapshot, "p2")
    assert want is not None
    odd = [
        with_mon(snapshot, "p1", 2, forme="notapokemon", species="alsonot"),
        with_mon(snapshot, "p2", 0, forme="notapokemon", species="alsonot"),
        with_mon(snapshot, "p1", 2, types=()),
        with_mon(snapshot, "p1", 2, types=("fire", "water", "grass", "ice")),
        with_mon(snapshot, "p1", 2, hp=0.0),
        with_mon(snapshot, "p1", 2, moves=(P.MovePublic("notamove", True, False),)),
        with_mon(snapshot, "p2", 0, boosts={}),
        with_mon(snapshot, "p2", 0, boosts={"atk": 99, "spe": -99}),
    ]
    for changed in odd:
        example = fz.encode(changed, "p2")
        assert example is not None and all(name in example for name in MU_NAMES)
    assert fz.counters["matchup_no_stats"] >= 2
    assert not any(name.startswith("matchup_error") for name in fz.counters)
    # A Pokemon without a stat line zeroes every entry it is part of.
    blind = mu(fz, odd[0])
    assert not blind["mu_cand"][:, :, 0].any() and not blind["mu_slot"][:, 0].any()
    assert blind["mu_slot"][:, 1].any()
    blind = mu(fz, odd[1])
    assert not blind["mu_cand"][0].any() and not blind["mu_slot"][0].any()
    assert blind["mu_slot"][1].any()
    # One foe absent, and nobody on the bench.
    team = snapshot.sides["p1"]
    alone = replace(
        team,
        active={"a": 2, "b": None},
        mons=tuple(
            replace(mon, slot=None) if mon.index == 1 else mon for mon in team.mons
        ),
    )
    lone = mu(fz, replace(snapshot, sides={**snapshot.sides, "p1": alone}))
    assert not lone["mu_slot"][:, 1].any() and lone["mu_slot"][:, 0].any()
    actor = snapshot.sides["p2"]
    spent = replace(
        actor,
        mons=tuple(
            replace(mon, fainted=mon.slot is None, brought=True) for mon in actor.mons
        ),
    )
    none = mu(fz, replace(snapshot, sides={**snapshot.sides, "p2": spent}))
    assert not none["mu_roster"].any() and none["mu_slot"].any()

    # The estimate itself fails: the example survives with every version-1
    # array as it was, the three arrays zero, and a named count.
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise ZeroDivisionError("unit test")

    monkeypatch.setattr(MU, "hit", broken)
    example = fz.encode(snapshot, "p2")
    assert example is not None
    for name, array in want.items():
        assert example[name].tobytes() == array.tobytes(), name
    assert all(not example[name].any() for name in MU_NAMES)
    assert fz.counters["matchup_error:ZeroDivisionError"] == 1
    assert not any(name.startswith("encode_error") for name in fz.counters)


def test_live_shadow_gives_the_v2_examples_of_the_offline_path(fz2: F.Featurizer):
    """Train / serve parity of the new arrays, and no private number in them.

    The body of the version-1 test with a version-2 featurizer: a player-view
    stream (the bot is p1, with its exact HP) through ``LiveShadow`` against
    the rewritten log the dataset builder reads.
    """
    exact = TF.HEADER + TF.GAME
    for public, private in (("100/100", "187/187"), ("41/100", "77/187")):
        exact = exact.replace(
            f"|p1a: Incineroar|{public}", f"|p1a: Incineroar|{private}"
        )
        exact = exact.replace(
            f"|Incineroar, L50, M|{public}", f"|Incineroar, L50, M|{private}"
        )
    events = E.split_log(exact)
    assert any("77/187" in "|".join(event) for event in events)
    rewritten = "\n".join("|".join(P.rewrite_event(e, "p1")) for e in events)
    offline = P.drive_log(rewritten, "battle-test-1", sheets_known=False)
    fake = SimpleNamespace(
        _replay_data=[], player_role="p1", battle_tag="battle-test-1"
    )
    shadow = P.LiveShadow()
    compared = 0
    for event in events:
        fake._replay_data.append(list(event))
        if event[1] != "turn":
            continue
        snapshot = shadow.sync(fake)
        assert snapshot is not None
        record = TF.turn(offline, int(event[2]))
        live, built = fz2.encode(snapshot, "p2"), fz2.encode(record.snapshot, "p2")
        assert live is not None and built is not None
        assert all(name in built for name in MU_NAMES)
        for name in built:
            assert np.array_equal(live[name], built[name]), (event[2], name)
        compared += 1
    assert compared == 7
    # The knock-out ratio of a hit into the bot's Incineroar is made from the
    # public 41 percent, not from 77 / 187 (0.4118): out_ko = q(share / 0.41).
    snapshot = TF.turn(offline, 2).snapshot
    bot = snapshot.sides["p1"]
    slot = next(mon for mon in bot.mons if mon.species == "incineroar")
    assert slot.hp == pytest.approx(0.41, abs=1e-9) and slot.hp != 77 / 187
    if slot.slot is not None:
        f = "ab".index(slot.slot)
        example = fz2.encode(snapshot, "p2")
        assert example is not None
        out_dmg, out_ko = (
            MU.MU_SLOT_COLUMNS.index("out_dmg"),
            MU.MU_SLOT_COLUMNS.index("out_ko"),
        )
        rows = example["mu_slot"][:, f].astype(int)
        for row in rows[rows[:, out_dmg] > 0]:
            # q() of the two are consistent with hp 0.41 within rounding.
            low = (row[out_dmg] - 0.5) / 50 / 0.41 * 50 - 0.5
            high = (row[out_dmg] + 0.5) / 50 / 0.41 * 50 + 0.5
            assert low <= row[out_ko] <= high or row[out_ko] == 100


# --- the slot mirror --------------------------------------------------------------


def test_swap_slots_mirrors_the_v2_arrays(fz2: F.Featurizer, batch2: F.Batch):
    total = 0
    for record in TM.drive().turns:
        compared, bad = TM.mirror_mismatches(fz2, record)
        assert not bad, bad[:6]
        total += compared
        example = fz2.encode_turn(record.snapshot, record.actions, "p1")
        assert example is not None and all(name in example for name in MU_NAMES)
    assert total >= 30
    # The mirror is not the identity on them, and each side moves its own axis.
    actor = M.swap_slots(batch2, True, False)
    other = M.swap_slots(batch2, False, True)
    assert np.array_equal(actor["mu_cand"], batch2["mu_cand"][:, ::-1])
    assert np.array_equal(actor["mu_slot"], batch2["mu_slot"][:, ::-1])
    assert np.array_equal(actor["mu_roster"], batch2["mu_roster"])
    assert np.array_equal(other["mu_cand"], batch2["mu_cand"][:, :, :, ::-1])
    assert np.array_equal(other["mu_slot"], batch2["mu_slot"][:, :, ::-1])
    assert np.array_equal(other["mu_roster"], batch2["mu_roster"][:, :, ::-1])
    assert not np.array_equal(actor["mu_slot"], batch2["mu_slot"])
    assert not np.array_equal(other["mu_roster"], batch2["mu_roster"])
    # Per-example flags: only the chosen rows are mirrored.
    n = batch2["mu_slot"].shape[0]
    pick = np.arange(n) % 2 == 0
    some = M.swap_slots(batch2, pick, ~pick)
    assert np.array_equal(some["mu_slot"][pick], actor["mu_slot"][pick])
    assert np.array_equal(some["mu_slot"][~pick], other["mu_slot"][~pick])
    for name in MU_NAMES:
        assert some[name].dtype == batch2[name].dtype
    # A version-1 batch has none of them and mirrors as before.
    assert not any(name in M.swap_slots(M.strip_labels(batch2)) for name in ())
    assert all(name in M.strip_labels(batch2) for name in MU_NAMES)


# --- the network ------------------------------------------------------------------


def test_a_config_without_extras_builds_the_old_network(
    fz1: F.Featurizer, fz2: F.Featurizer
):
    old, new = TM.small_net(fz1), TM.small_net(fz2)
    assert list(old.state_dict()) == list(new.state_dict())
    assert new.extra_keys == () and new.matchup_keys == () and new.extra_blocks == ()
    assert not [name for name, _ in new.named_buffers() if name.endswith("_scale")]
    assert new.config.extra_widths() == {}
    assert M.extras_overrides(fz2, 0) == {} and M.extras_overrides(fz1, 0) == {}
    assert set(M.to_tensors(TM.game_batch(fz2))) == set(M.FEATURE_KEYS)
    # A stored config from before these fields reads them as off.
    stored = {
        key: value
        for key, value in old.config.to_dict().items()
        if key not in (*M.EXTRA_CONFIG_FIELDS, "matchup_version")
    }
    assert M.OppNetConfig.from_dict(stored) == old.config
    assert M.matchup_overrides is M.extras_overrides


def test_overrides_follow_the_bit_mask(fz1: F.Featurizer, fz2: F.Featurizer):
    assert M.extras_overrides(fz2, 7) == {
        "mu_cand": 8,
        "mu_slot": 10,
        "mu_roster": 4,
        "matchup_version": MU.MATCHUP_VERSION,
    }
    assert M.extras_overrides(fz2, 2) == {"mu_slot": 10, "matchup_version": 1}
    assert M.extras_overrides(fz2, 5) == {
        "mu_cand": 8,
        "mu_roster": 4,
        "matchup_version": 1,
    }
    # 8 is ``sp_cand``'s bit: this featurizer does not write it (and until
    # its table row exists no array has the bit at all).
    for mask in (8, 15, 1 << 20, -1):
        with pytest.raises(ValueError):
            M.extras_overrides(fz2, mask)
    with pytest.raises(ValueError):
        M.extras_overrides(fz1, 7)  # a version-1 featurizer writes none
    only = F.Featurizer.build(TM.repertoire(), layout_version=2, extras=("mu_slot",))
    assert M.extras_overrides(only, 2) == {"mu_slot": 10, "matchup_version": 1}
    with pytest.raises(ValueError):
        M.extras_overrides(only, 3)
    config = M.OppNetConfig.for_featurizer(fz2)
    for bad in (
        {"mu_cand": 7, "matchup_version": 1},  # not the table's width
        {"mu_cand": 8},  # no matchup version
        {"mu_slot": -1},
        {"sp_cand": 99},  # no array of that width
    ):
        with pytest.raises(ValueError):
            replace(config, **bad).check()
    if F.extra_array("sp_cand") is None:  # reserved: no such array yet
        with pytest.raises(ValueError):
            replace(config, sp_cand=4).check()


def test_a_fresh_matchup_network_is_the_network_without_it(
    fz2: F.Featurizer, batch2: F.Batch
):
    """Seed 0 for construction, seed 1 for the jolt: extras 7 against extras 0."""
    plain = TM.jolt(TM.small_net(fz2))
    full = TM.jolt(matchup_net(fz2))
    assert full.extra_keys == MU_NAMES
    new = {"mu_cand_act", "mu_cand_tgt", "mu_slot_in", "mu_roster_in"}
    # The scales each array is multiplied by are stored with the weights.
    scales = {f"{name}_scale" for name in MU_NAMES}
    shared, extra = plain.state_dict(), full.state_dict()
    assert set(extra) - set(shared) == new | scales
    assert {name for name, _ in full.named_parameters()} - set(shared) == new
    for name, value in shared.items():
        assert torch.equal(value, extra[name]), name
    for name in new:
        assert not extra[name].any()
    d = full.config.d_model
    assert {name: tuple(extra[name].shape) for name in sorted(new)} == {
        "mu_cand_act": (d, 16),
        "mu_cand_tgt": (d, 8),
        "mu_roster_in": (d, 8),
        "mu_slot_in": (d, 20),
    }
    assert full.n_parameters() - plain.n_parameters() == d * 52
    assert batch2["mu_cand"].any()  # the inputs are not zero: the weights are
    want, got = logits(plain, batch2), logits(full, batch2)
    for name in want:
        assert torch.equal(want[name], got[name]), name
    # Each block alone, and a batch of one (how the runtime calls it).
    for mask in (1, 2, 4, 3, 6):
        part = TM.jolt(matchup_net(fz2, mask))
        got = logits(part, F.take(batch2, slice(3, 4)))
        want = logits(plain, F.take(batch2, slice(3, 4)))
        assert all(torch.equal(want[name], got[name]) for name in want), mask
    # At the default size: 6,656 more parameters, the same start.
    torch.manual_seed(0)
    big_plain = M.OppNet.for_featurizer(fz2)
    torch.manual_seed(0)
    big_full = M.OppNet.for_featurizer(fz2, **M.extras_overrides(fz2, ALL))
    assert big_full.n_parameters() - big_plain.n_parameters() == 6656
    want, got = logits(big_plain, batch2), logits(big_full, batch2)
    assert all(torch.equal(want[name], got[name]) for name in want)
    # An untrained matchup network is still the prior.
    net = matchup_net(fz2)
    prediction = M.OppNetPredictor(net, fz2).predict(batch2)
    again = M.OppNetPredictor(TM.small_net(fz2), fz2).predict(batch2)
    assert all(np.array_equal(prediction[name], again[name]) for name in again)
    assert np.allclose(
        prediction["target"], F.uniform_prediction(batch2)["target"], atol=1e-6
    )


def test_an_old_state_dict_warm_starts_a_matchup_network(
    fz2: F.Featurizer, batch2: F.Batch
):
    old = TM.jolt(TM.small_net(fz2, seed=4))
    new = matchup_net(fz2, seed=9)
    result = new.load_state_dict(old.state_dict(), strict=False)
    assert sorted(result.missing_keys) == [
        "mu_cand_act",
        "mu_cand_scale",
        "mu_cand_tgt",
        "mu_roster_in",
        "mu_roster_scale",
        "mu_slot_in",
        "mu_slot_scale",
    ]
    assert not result.unexpected_keys
    want, got = logits(old, batch2), logits(new, batch2)
    assert all(torch.equal(want[name], got[name]) for name in want)


def test_each_block_reaches_the_heads_it_should(fz2: F.Featurizer, batch2: F.Batch):
    net = fill_extras(TM.jolt(matchup_net(fz2, n_layers=0)))
    one = F.take(batch2, slice(0, 1))
    assert (one["act_mon"] >= 0).all() and (one["foe_mon"] >= 0).all()
    base = logits(net, one)

    def changed(name: str, index: tuple) -> dict[str, np.ndarray]:
        moved = dict(one)
        array = one[name].copy()
        array[index] = array[index] + 3
        moved[name] = array
        after = logits(net, moved)
        return {head: (after[head] != base[head]).numpy()[0] for head in base}

    s, c, f, n_cand = 1, 2, 0, fz2.n_cand
    # mu_cand[s, c, f]: the action logit of (s, c), the target logit of
    # (s, c, foe f); nothing else anywhere.
    diff = changed("mu_cand", (0, s, c, f))
    want = np.zeros_like(diff["action"])
    want[s, c] = True
    assert np.array_equal(diff["action"], want)
    want = np.zeros_like(diff["target"])
    want[s, c, [F.T_FOE_A, F.T_FOE_B][f]] = True
    assert np.array_equal(diff["target"], want)
    assert not diff["mega"].any()
    # mu_roster[r]: the switch logit of pointer r, in both slots.
    r = 3
    diff = changed("mu_roster", (0, r))
    want = np.zeros_like(diff["action"])
    want[:, n_cand + 1 + r] = True
    assert np.array_equal(diff["action"], want)
    assert not diff["target"].any() and not diff["mega"].any()
    # mu_slot[s]: every output of slot s, and of the partner only the
    # ally-target logits (they read the partner's query).
    diff = changed("mu_slot", (0, s))
    assert diff["action"][s].all() and not diff["action"][1 - s].any()
    assert diff["mega"][s] and not diff["mega"][1 - s]
    assert diff["target"][s].all()
    partner = diff["target"][1 - s]
    assert partner[:, F.T_ALLY].all()
    others = [t for t in range(F.N_TARGET) if t != F.T_ALLY]
    assert not partner[:, others].any()


def test_a_per_candidate_block_without_a_foe_axis_is_wired(
    fz2: F.Featurizer, batch2: F.Batch, monkeypatch: pytest.MonkeyPatch
):
    """The reserved kind (``sp_cand``): one table row makes the network read it."""
    spec = F.ExtraArray(
        "sp_cand",
        8,
        F.EXTRA_CAND,
        ("a", "b", "c", "d"),
        (1.0,) * 4,
        "float16",
        "set",
        "",
    )
    assert spec.shape(12) == (2, 12, 4) and spec.foe_axis is None
    table = tuple(row for row in F.EXTRA_ARRAYS if row.name != "sp_cand")
    monkeypatch.setattr(F, "EXTRA_ARRAYS", (*table, spec))
    monkeypatch.setattr(M, "EXTRA_ARRAYS", (*table, spec))
    n = batch2["act_mon"].shape[0]
    rng = np.random.default_rng(0)
    batch = dict(batch2)
    batch["sp_cand"] = rng.random((n, 2, 12, 4)).astype(np.float16)
    plain = TM.jolt(TM.small_net(fz2))
    net = TM.jolt(TM.small_net(fz2, sp_cand=4))
    assert net.extra_keys == ("sp_cand",)
    assert set(net.state_dict()) - set(plain.state_dict()) == {
        "sp_cand_act",
        "sp_cand_scale",
    }
    assert tuple(net.state_dict()["sp_cand_act"].shape) == (net.config.d_model, 4)
    want, got = logits(plain, batch), logits(net, batch)
    assert all(torch.equal(want[name], got[name]) for name in want)
    fill_extras(net)
    moved = logits(net, batch)
    assert not torch.equal(moved["action"], want["action"])
    assert torch.equal(moved["action"][:, :, 12:], want["action"][:, :, 12:])
    assert torch.equal(moved["target"], want["target"])
    # The mirror follows the actor's slot axis.
    swapped = M.swap_slots(batch, True, False)
    assert np.array_equal(swapped["sp_cand"], batch["sp_cand"][:, ::-1])
    assert np.array_equal(M.swap_slots(batch, False, True)["sp_cand"], batch["sp_cand"])


def test_the_new_matrices_learn(fz2: F.Featurizer, batch2: F.Batch):
    torch.manual_seed(0)
    net = matchup_net(fz2, dropout=0.0)
    names = ("mu_cand_act", "mu_cand_tgt", "mu_slot_in", "mu_roster_in")
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    labels = M.to_labels(batch2)
    net.train()
    for _ in range(2):
        tensors = M.to_tensors(batch2, extra=net.extra_keys)
        terms = M.nll_terms(net(tensors), labels)
        loss, _ = M.total_loss(terms, labels["weight"], 0.2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    state = net.state_dict()
    for name in names:
        assert state[name].abs().sum() > 0, name
    # They are matrices: the trainer's weight-decay group holds them.
    decay = T.parameter_groups(net, 0.01)[0]["params"]
    assert all(any(p is getattr(net, name) for p in decay) for name in names)


def test_wrong_pairings_are_refused_or_counted(
    fz1: F.Featurizer, fz2: F.Featurizer, batch1: F.Batch, batch2: F.Batch
):
    predictor = M.OppNetPredictor(matchup_net(fz2), fz2)
    # A matchup network on a batch without the arrays: a counted failure and
    # the uniform floor, never a silent version-1 answer.
    out = predictor.predict(batch1)
    assert predictor.counters["predict_error:KeyError"] == 1
    floor = F.uniform_prediction(batch1)
    assert all(np.allclose(out[name], floor[name]) for name in floor)
    with pytest.raises(KeyError):
        M.to_tensors(batch1, extra=predictor.net.extra_keys)
    # Arrays of another width.
    narrow = dict(batch2)
    narrow["mu_slot"] = batch2["mu_slot"][..., :9]
    predictor.predict(narrow)
    assert sum(predictor.counters.values()) == 2
    assert predictor.predict(batch2)["action"].shape == out["action"].shape
    assert sum(predictor.counters.values()) == 2
    # Loading: the featurizer must write the arrays, at the same definitions.
    payload = predictor.to_payload()
    assert M.from_payload(payload, fz2).net.extra_keys == MU_NAMES
    with pytest.raises(ValueError):
        M.from_payload(payload, fz1)
    other = {**payload["config"], "matchup_version": MU.MATCHUP_VERSION + 1}
    with pytest.raises(ValueError):
        M.from_payload({**payload, "config": other}, fz2)
    only = F.Featurizer.build(TM.repertoire(), layout_version=2, extras=("mu_slot",))
    with pytest.raises(ValueError):
        M.from_payload(payload, only)
    # An old-style network on a version-2 batch is allowed, and exact.
    plain = TM.jolt(TM.small_net(fz1))
    want, got = logits(plain, batch1), logits(plain, batch2)
    assert all(torch.equal(want[name], got[name]) for name in want)


def test_matchup_predictor_round_trips(
    tmp_path: Path, fz2: F.Featurizer, batch2: F.Batch
):
    predictor = M.OppNetPredictor(
        fill_extras(TM.jolt(matchup_net(fz2))), fz2, name="mu", action_temperature=1.2
    )
    want = predictor.predict(batch2)
    plain = M.OppNetPredictor(TM.jolt(TM.small_net(fz2)), fz2).predict(batch2)
    assert not np.array_equal(want["action"], plain["action"])  # the blocks are read
    path = tmp_path / "payload.pt"
    torch.save(predictor.to_payload(), path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    rebuilt = F.Featurizer.from_payload(fz2.to_payload())
    again = M.from_payload(payload, rebuilt)
    got = again.predict(batch2)
    assert all(np.array_equal(want[name], got[name]) for name in want)
    assert payload["version"] == M.PAYLOAD_VERSION
    assert (
        payload["config"]["mu_cand"] == 8 and payload["config"]["matchup_version"] == 1
    )
    if importlib.util.find_spec(TM.ARTIFACT_MODULE) is None:
        pytest.skip("oppmodel.artifact is not here")
    artifact = importlib.import_module(TM.ARTIFACT_MODULE)
    target = tmp_path / "artifact.pt"
    artifact.save_artifact(
        target,
        kind=M.KIND,
        name="mu",
        featurizer=fz2,
        predictor_payload=predictor.to_payload(),
    )
    loaded = artifact.load_predictor(target)
    assert loaded.featurizer.layout_version == 2
    assert loaded.featurizer.extras == MU_NAMES
    assert loaded.featurizer.signature_diff == []
    assert loaded.predictor.net.extra_keys == MU_NAMES
    got = loaded.predictor.predict(batch2)
    assert all(np.array_equal(want[name], got[name]) for name in want)
    # The loaded featurizer encodes the arrays the stored network reads.
    example = loaded.featurizer.encode(TM.drive().turns[0].snapshot, "p1")
    assert example is not None
    assert np.array_equal(example["mu_cand"], batch2["mu_cand"][0])


# --- the trainer and the builder --------------------------------------------------


def test_trainer_extras_flag(
    tmp_path: Path,
    fz1: F.Featurizer,
    fz2: F.Featurizer,
    capsys: pytest.CaptureFixture[str],
):
    if importlib.util.find_spec(TM.ARTIFACT_MODULE) is None:
        pytest.skip("oppmodel.artifact is not here")
    artifact = importlib.import_module(TM.ARTIFACT_MODULE)
    assert T.parse_args([]).extras == 0
    assert T.parse_args(["--matchup", "7"]).extras == 7
    assert T.parse_args(["--extras", "5"]).extras == 5
    old_data, new_data = tmp_path / "v1", tmp_path / "v2"
    TT.write_dataset(old_data, fz1)
    TT.write_dataset(new_data, fz2)
    # All three blocks on a version-2 dataset: an artifact that reloads.
    assert TT.run(new_data, tmp_path / "on", "--epochs", "1", "--extras", "7") == 0
    report_on = json.loads((tmp_path / "on" / "train_report.json").read_text())
    assert report_on["status"] == "TRAIN_DONE" and report_on["args"]["extras"] == 7
    assert report_on["config"]["mu_cand"] == 8 and report_on["config"]["mu_slot"] == 10
    loaded = artifact.load_predictor(tmp_path / "on" / "artifact.pt")
    assert loaded.predictor.net.extra_keys == MU_NAMES
    assert loaded.featurizer.layout_version == 2
    # ... refused on a version-1 dataset, and nothing is written as a model.
    capsys.readouterr()
    assert TT.run(old_data, tmp_path / "bad", "--epochs", "1", "--matchup", "7") != 0
    assert "TRAIN_FAILED" in capsys.readouterr().out
    assert not (tmp_path / "bad" / "artifact.pt").exists()
    # Off (the default), the version-2 dataset trains the model the
    # version-1 dataset trains: the same numbers, epoch by epoch.
    assert TT.run(new_data, tmp_path / "off2", "--epochs", "2") == 0
    assert TT.run(old_data, tmp_path / "off1", "--epochs", "2", "--extras", "0") == 0
    off1 = json.loads((tmp_path / "off1" / "train_report.json").read_text())
    off2 = json.loads((tmp_path / "off2" / "train_report.json").read_text())
    assert off1["n_parameters"] == off2["n_parameters"]
    assert report_on["n_parameters"] - off2["n_parameters"] == 32 * 52
    keys = ("val_fine", "val_action", "val_target", "train_loss", "train_fine")
    for row1, row2 in zip(off1["history"], off2["history"]):
        for key in keys:
            if key in row1 and row1[key] == row1[key]:  # skip NaN of epoch 0
                assert row1[key] == row2[key], (row1["epoch"], key)
    assert len(off1["history"]) == len(off2["history"]) == 3
    one = artifact.load_predictor(tmp_path / "off1" / "artifact.pt").predictor
    two = artifact.load_predictor(tmp_path / "off2" / "artifact.pt").predictor
    for name, value in one.net.state_dict().items():
        assert torch.equal(value, two.net.state_dict()[name]), name
    # Both arms start from the same shared weights: the epoch-0 line (the
    # untrained network on validation) is the same number.
    assert report_on["history"][0]["val_fine"] == off2["history"][0]["val_fine"]


def test_builder_layout_version(tmp_path: Path):
    corpus = TD.Corpus(tmp_path)
    old = corpus.build("v1")
    new = corpus.build("v2", layout_version=2)
    assert old["layout_version"] == 1 and new["layout_version"] == 2
    assert "matchup" not in old and "extras" not in old
    assert new["extras"] == list(F.extra_names()) and set(MU_NAMES) <= set(
        new["extras"]
    )
    assert new["matchup"]["version"] == MU.MATCHUP_VERSION
    assert set(new["arrays"]) - set(old["arrays"]) == set(new["extras"])
    assert new["arrays"]["mu_cand"]["shape"] == [2, 12, 2, 8]
    root = tmp_path / "results_oppmodel"
    before, _ = F.load_dataset(root / "v1")
    after, manifest = F.load_dataset(root / "v2")
    assert manifest["layout_version"] == 2
    assert set(after) - set(before) == set(new["extras"])
    for name, array in before.items():
        assert after[name].dtype == array.dtype, name
        assert after[name].tobytes() == array.tobytes(), name
    assert after["mu_slot"].any() and after["mu_cand"].dtype == np.int8
    assert F.Featurizer.load(root / "v2", strict=True).layout_version == 2
    assert F.Featurizer.load(root / "v1", strict=True).layout_version == 1
    stored = json.loads((root / "v1" / "vocab.json").read_text())
    assert "extras" not in stored and "matchup" not in stored
    assert B.LAYOUT_VERSION == 1
