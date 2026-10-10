"""Review fixes of 2026-10-10 on the features side of the opponent predictor.

Each test states one finding of the review and fails on the code as it was:

* the config of a network without version-2 arrays keeps its old keys, so the
  payload of an old artifact is re-made key for key (the calibration tool
  compares them);
* the per-column scales a network multiplies the version-2 arrays by are
  stored with its weights, and a state dict without them is refused;
* a featurizer that was loaded writes back the version-2 definitions it was
  LOADED with, so a difference to today's code survives every re-save;
* the quantiser of the set-prior array is pinned like the matchup definitions;
* a matchup featurizer is not built on a type chart that cannot be read;
* an artifact carries the featurizer of the arrays its network reads;
* a version-2 build needs the own source, and the set table says whose
  sheets built it;
* ``--own-before`` leaves the later own games out and keeps their opponents
  out of training; a reader tells the later rows by their own time;
* the trainer's ``--informational`` never scores the sealed games unasked.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from poke_env.data import GenData

from datagen import oppmodel_build_dataset as B
from training import train_oppmodel as T
from unit_tests import test_oppmodel_dataset as TD
from unit_tests import test_oppmodel_layout_v2 as TL
from unit_tests import test_oppmodel_model as TM
from unit_tests import test_oppmodel_setprior_integration as TS
from unit_tests import test_oppmodel_train as TT
from vgc_bench.src.oppmodel import events as E
from vgc_bench.src.oppmodel import features as F
from vgc_bench.src.oppmodel import matchup as MU
from vgc_bench.src.oppmodel import model as M

ROOT = Path(__file__).resolve().parents[1]
LIVE_ARTIFACT = ROOT / "results_oppmodel/oppnet_v2_blind/artifact.pt"
MU_NAMES = TL.MU_NAMES
ALL_NAMES = (*MU_NAMES, "sp_cand")
EVERYTHING = 15
V2_FIELDS = (*M.EXTRA_CONFIG_FIELDS, "matchup_version")
HEADS = ("action", "target", "mega")
CUT_DAY = "2026-10-09"


@pytest.fixture(scope="module")
def fz1() -> F.Featurizer:
    return F.Featurizer.build(TM.repertoire())


@pytest.fixture(scope="module")
def fz2() -> F.Featurizer:
    return F.Featurizer.build(TM.repertoire(), layout_version=2, extras=MU_NAMES)


@pytest.fixture(scope="module")
def fzs() -> F.Featurizer:
    return F.Featurizer.build(
        TM.repertoire(), layout_version=2, set_table=TS.table(), set_accounts=[7, 3, 7]
    )


@pytest.fixture(scope="module")
def batch(fzs: F.Featurizer) -> F.Batch:
    return TM.game_batch(fzs)


def artifact_module() -> Any:
    if importlib.util.find_spec(TM.ARTIFACT_MODULE) is None:
        pytest.skip("oppmodel.artifact is not here")
    return importlib.import_module(TM.ARTIFACT_MODULE)


def same_tensors(one: dict[str, Any], two: dict[str, Any]) -> bool:
    return list(one) == list(two) and all(
        torch.equal(one[name], two[name]) for name in one
    )


# --- the network config -----------------------------------------------------------


def test_a_config_without_extras_has_the_keys_it_always_had(
    fz1: F.Featurizer, fz2: F.Featurizer
):
    plain = TM.small_net(fz2)
    told = plain.config.to_dict()
    assert not set(told) & set(V2_FIELDS)
    assert M.OppNetConfig.from_dict(told) == plain.config
    # The payload of a network from before the fields existed is re-made
    # with exactly its stored keys (what the calibration tool compares).
    predictor = M.OppNetPredictor(TM.jolt(TM.small_net(fz1)), fz1, name="old")
    stored = predictor.to_payload()
    old_config = {k: v for k, v in stored["config"].items() if k not in V2_FIELDS}
    again = M.from_payload({**stored, "config": old_config}, fz1).to_payload()
    assert again["config"] == old_config
    assert same_tensors(again["state_dict"], stored["state_dict"])
    # A network that reads some arrays writes those fields and no other.
    part = TL.matchup_net(fz2, 2).config.to_dict()
    assert {key: part[key] for key in part if key in V2_FIELDS} == {
        "mu_slot": 10,
        "matchup_version": MU.MATCHUP_VERSION,
    }
    assert M.OppNetConfig.from_dict(part) == TL.matchup_net(fz2, 2).config
    assert M.OppNetConfig.from_dict(part).extra_widths() == {"mu_slot": 10}


def test_the_live_artifact_is_remade_key_for_key():
    if not LIVE_ARTIFACT.is_file():
        pytest.skip("the live artifact is not here")
    artifact = artifact_module()
    document = artifact.read_artifact(LIVE_ARTIFACT)
    loaded = artifact.load_predictor(LIVE_ARTIFACT)
    old, new = document["predictor"], loaded.predictor.to_payload()
    # (``event_calibration`` is absent in a payload from before it existed.)
    assert set(old) | {"event_calibration"} == set(new)
    assert new["config"] == old["config"] and len(old["config"]) == 29
    assert list(new["state_dict"]) == list(old["state_dict"])
    for name, value in old["state_dict"].items():
        assert np.array_equal(np.asarray(value), new["state_dict"][name].numpy()), name
    assert new["temperatures"] == old["temperatures"]
    # ... and so is its featurizer payload (a version-1 one).
    kept, made = document["featurizer"], loaded.featurizer.to_payload()
    assert sorted(kept) == sorted(made) and made["layout_version"] == 1
    assert made["dex_signature"] == kept["dex_signature"]


# --- the scales -------------------------------------------------------------------


def test_the_scales_travel_with_the_weights(
    fzs: F.Featurizer, batch: F.Batch, monkeypatch: pytest.MonkeyPatch
):
    net = TL.fill_extras(TM.jolt(TL.matchup_net(fzs, EVERYTHING)))
    state = net.state_dict()
    for spec in F.EXTRA_ARRAYS:
        stored = state[f"{spec.name}{M.SCALE_SUFFIX}"]
        assert torch.equal(stored, torch.tensor(spec.scales, dtype=torch.float32))
    assert net.extra_scales()["mu_slot"] == pytest.approx(list(MU.MU_SLOT_SCALES))
    predictor = M.OppNetPredictor(net, fzs, name="scaled")
    want = predictor.predict(batch)
    payload = predictor.to_payload()
    assert f"sp_cand{M.SCALE_SUFFIX}" in payload["state_dict"]
    # The table in the code changes (as an edit of a scale constant would).
    changed = tuple(
        dataclasses.replace(spec, scales=tuple(2.0 * x for x in spec.scales))
        for spec in F.EXTRA_ARRAYS
    )
    monkeypatch.setattr(F, "EXTRA_ARRAYS", changed)
    monkeypatch.setattr(M, "EXTRA_ARRAYS", changed)
    # A stored network keeps the scales it was fitted with: the same numbers.
    again = M.from_payload(payload, fzs)
    got = again.predict(batch)
    assert not again.counters
    for head in HEADS:
        assert np.array_equal(want[head], got[head]), head
    assert again.net.extra_scales() == net.extra_scales()
    # A NEW network takes the table's, and they are what it stores.
    fresh = TL.matchup_net(fzs, EVERYTHING)
    assert fresh.extra_scales()["mu_slot"] == pytest.approx(
        [2.0 * x for x in MU.MU_SLOT_SCALES]
    )
    fresh.load_state_dict(
        {k: v for k, v in state.items() if not k.endswith(M.SCALE_SUFFIX)}, strict=False
    )
    moved = M.OppNetPredictor(fresh, fzs).predict(batch)
    assert not np.array_equal(moved["action"], want["action"])


def test_a_state_dict_without_its_scales_is_refused(
    fz1: F.Featurizer, fzs: F.Featurizer
):
    predictor = M.OppNetPredictor(TL.matchup_net(fzs, EVERYTHING), fzs)
    payload = predictor.to_payload()
    config = M.OppNetConfig.from_dict(payload["config"])
    assert M.extra_state_problems(config, payload["state_dict"]) == []
    widths = {spec.name: spec.width for spec in F.EXTRA_ARRAYS}
    for name in ALL_NAMES:
        key = f"{name}{M.SCALE_SUFFIX}"
        state = {k: v for k, v in payload["state_dict"].items() if k != key}
        assert len(M.extra_state_problems(config, state)) == 1
        with pytest.raises(ValueError, match=key):
            M.from_payload({**payload, "state_dict": state}, fzs)
        for broken in (torch.zeros(3), torch.full((widths[name],), float("nan"))):
            damaged = {**payload["state_dict"], key: broken}
            with pytest.raises(ValueError):
                M.from_payload({**payload, "state_dict": damaged}, fzs)
    # A network without version-2 arrays has no such tensor and needs none.
    plain = M.OppNetPredictor(TM.small_net(fz1), fz1).to_payload()
    assert not [k for k in plain["state_dict"] if k.endswith(M.SCALE_SUFFIX)]
    assert M.from_payload(plain, fz1).net.extra_keys == ()


# --- the stored definitions -------------------------------------------------------


def test_a_resave_keeps_the_matchup_definitions_it_was_loaded_with(
    fz2: F.Featurizer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    stored = fz2.to_payload()
    assert stored["matchup"] == fz2.matchup_signature()
    assert fz2.stored_matchup_signature() == fz2.matchup_signature()
    was = stored["matchup"]["assumptions"]["proxy_power"]
    # A later edit of the definitions, without a version bump.
    monkeypatch.setattr(MU, "PROXY_POWER", was + 10)
    loaded = F.Featurizer.from_payload(stored)
    assert loaded.signature_diff == [F.FAMILY_MATCHUP]
    today = loaded.matchup_signature()
    assert today is not None and today["assumptions"]["proxy_power"] == was + 10
    assert loaded.stored_matchup_signature() == stored["matchup"]
    # Written again (a tool that copies an artifact does this), it still says
    # what the arrays and the weights were made under ...
    copied = loaded.to_payload()
    assert copied["matchup"] == stored["matchup"]
    assert copied["matchup"]["assumptions"]["proxy_power"] == was
    # ... so the copy is refused or left unserved exactly like the original.
    assert F.Featurizer.from_payload(copied).signature_diff == [F.FAMILY_MATCHUP]
    with pytest.raises(ValueError):
        F.Featurizer.from_payload(copied, strict=True)
    loaded.save(tmp_path / "copy")
    assert F.Featurizer.load(tmp_path / "copy").signature_diff == [F.FAMILY_MATCHUP]
    # A payload that stored no definitions keeps saying so.
    bare = {key: value for key, value in stored.items() if key != "matchup"}
    again = F.Featurizer.from_payload(F.Featurizer.from_payload(bare).to_payload())
    assert again.signature_diff == [F.FAMILY_MATCHUP]
    # Through an artifact file: the copy does not load strictly either.
    artifact = artifact_module()
    net = TL.matchup_net(fz2)
    payload = M.OppNetPredictor(net, fz2, name="mu").to_payload()
    target = tmp_path / "copy.pt"
    artifact.save_artifact(
        target, kind=M.KIND, name="mu", featurizer=loaded, predictor_payload=payload
    )
    assert artifact.load_predictor(target).featurizer.signature_diff == [
        F.FAMILY_MATCHUP
    ]
    with pytest.raises(ValueError):
        artifact.load_predictor(target, strict=True)
    monkeypatch.undo()
    assert F.Featurizer.from_payload(copied, strict=True).signature_diff == []


def test_the_set_prior_quantiser_is_pinned(
    fzs: F.Featurizer, monkeypatch: pytest.MonkeyPatch
):
    stored = fzs.to_payload()
    assert stored["set_prior_quant"] == F.SP_QUANT == fzs.set_prior_quant
    assert F.Featurizer.from_payload(stored, strict=True).signature_diff == []
    # A payload from before the key existed was written with 100: it loads
    # clean and is written back as it was stored.
    old = {key: value for key, value in stored.items() if key != "set_prior_quant"}
    legacy = F.Featurizer.from_payload(old, strict=True)
    assert legacy.set_prior_quant == F.SP_QUANT_UNSTORED == 100
    assert sorted(legacy.to_payload()) == sorted(old)
    for wrong in (0, -5, 2.5, "100", True):
        with pytest.raises(ValueError):
            F.Featurizer.from_payload({**stored, "set_prior_quant": wrong})
    # The quantiser changes in the code: both payloads are now another layout.
    monkeypatch.setattr(F, "SP_QUANT", 50)
    for payload in (stored, old):
        changed = F.Featurizer.from_payload(payload)
        assert changed.signature_diff == [F.FAMILY_SETPRIOR]
        assert changed.set_prior_quant == 100
        assert changed.to_payload().get("set_prior_quant") == payload.get(
            "set_prior_quant"
        )
        with pytest.raises(ValueError):
            F.Featurizer.from_payload(payload, strict=True)
    # A featurizer without the array has no quantiser to differ.
    bare = F.Featurizer.build(TM.repertoire(), layout_version=2)
    assert bare.set_prior_quant is None
    assert "set_prior_quant" not in bare.to_payload()
    assert F.Featurizer.from_payload(bare.to_payload()).signature_diff == []


def test_the_trainer_refuses_arrays_written_under_other_definitions(
    fzs: F.Featurizer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    artifact_module()
    data = tmp_path / "v2"
    TT.write_dataset(data, fzs)
    monkeypatch.setattr(F, "SP_QUANT", 50)
    capsys.readouterr()
    assert TT.run(data, tmp_path / "sp", "--epochs", "1", "--extras", "8") != 0
    assert "set_prior" in TT.lines_of(capsys)[-1]
    assert not (tmp_path / "sp" / "artifact.pt").exists()
    # The matchup blocks do not read that array: they still train.
    assert TT.run(data, tmp_path / "mu", "--epochs", "1", "--extras", "7") == 0


# --- the type chart ---------------------------------------------------------------


def test_no_matchup_featurizer_on_a_type_chart_that_cannot_be_read(
    fz1: F.Featurizer, fz2: F.Featurizer, monkeypatch: pytest.MonkeyPatch
):
    healthy = F._type_chart()
    assert healthy and fz2.matchup_signature() is not None
    stored = fz2.to_payload()

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("no data")

    F._type_chart_cached.cache_clear()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(GenData, "from_gen", broken)
            assert F._type_chart() == {}
            with pytest.raises(ValueError, match="type chart"):
                F.Featurizer.build(TM.repertoire(), layout_version=2, extras=MU_NAMES)
            with pytest.raises(ValueError, match="type chart"):
                F.Featurizer.from_payload(stored)
            # A featurizer that writes no matchup array does not need it.
            assert F.Featurizer.build(TM.repertoire()).layout_version == 1
            assert F.Featurizer.from_payload(fz1.to_payload()).signature_diff == []
        # The failure was not cached: the next read is the real chart.
        assert F._type_chart() == healthy
    finally:
        F._type_chart_cached.cache_clear()
    assert F.Featurizer.from_payload(stored, strict=True).signature_diff == []
    # A rule group the scan did not find is named in the counters.
    with monkeypatch.context() as patch:
        patch.setattr(MU, "rules", lambda moves: MU.Rules({}, (), ()))
        blind = F.Featurizer.build(TM.repertoire(), layout_version=2, extras=MU_NAMES)
    assert dict(blind.counters) == {
        "matchup_rule_not_found:weather_types": 1,
        "matchup_rule_not_found:speed_double": 1,
        "matchup_rule_not_found:speed_reverse": 1,
    }
    assert not fz2.counters or not any(
        name.startswith("matchup_rule_not_found") for name in fz2.counters
    )


# --- the featurizer an artifact carries -------------------------------------------


def test_narrowed_writes_only_the_arrays_that_are_asked(
    fz1: F.Featurizer, fzs: F.Featurizer, batch: F.Batch
):
    assert fzs.narrowed(ALL_NAMES) is fzs
    with pytest.raises(ValueError):
        fz1.narrowed(("mu_slot",))
    with pytest.raises(ValueError):
        fzs.narrowed(("nope",))
    # None left: the version-1 featurizer, with the version-1 payload keys.
    plain = fzs.narrowed(())
    assert plain.layout_version == 1 and plain.extras == ()
    assert sorted(plain.to_payload()) == sorted(fz1.to_payload())
    assert plain.set_table is None and plain.matchup_signature() is None
    want = TM.game_batch(fz1)
    got = TM.game_batch(plain)
    assert list(got) == list(want)
    for name, array in want.items():
        assert got[name].tobytes() == array.tobytes(), name
    # The set prior alone: no matchup definitions stored, the table kept.
    only = fzs.narrowed(("sp_cand",))
    payload = only.to_payload()
    assert only.extras == ("sp_cand",) and "matchup" not in payload
    assert payload["set_prior_signature"] == fzs.set_prior_signature
    assert payload["set_prior_accounts"] == [3, 7]
    assert payload["set_prior_quant"] == F.SP_QUANT
    made = TM.game_batch(only)
    assert made["sp_cand"].tobytes() == batch["sp_cand"].tobytes()
    assert "mu_slot" not in made
    # The matchup arrays alone: no table carried.
    blocks = fzs.narrowed(MU_NAMES)
    assert blocks.extras == MU_NAMES and blocks.set_table is None
    assert not [key for key in blocks.to_payload() if key.startswith("set_prior")]
    assert blocks.to_payload()["matchup"] == fzs.to_payload()["matchup"]
    made = TM.game_batch(blocks)
    for name in MU_NAMES:
        assert made[name].tobytes() == batch[name].tobytes(), name
    # A loaded featurizer keeps its stored definitions through it.
    loaded = F.Featurizer.from_payload(fzs.to_payload())
    assert (
        loaded.narrowed(MU_NAMES).to_payload()["matchup"]
        == (fzs.to_payload()["matchup"])
    )


def test_an_artifact_carries_the_featurizer_of_what_its_network_reads(
    fz1: F.Featurizer, fzs: F.Featurizer, tmp_path: Path
):
    artifact = artifact_module()
    v1, v2 = tmp_path / "v1", tmp_path / "v2"
    TT.write_dataset(v1, fz1)
    TT.write_dataset(v2, fzs)
    assert TT.run(v1, tmp_path / "one", "--epochs", "1") == 0
    assert TT.run(v2, tmp_path / "off", "--epochs", "1") == 0
    assert TT.run(v2, tmp_path / "sp", "--epochs", "1", "--extras", "8") == 0
    assert TT.run(v2, tmp_path / "mu", "--epochs", "1", "--extras", "2") == 0
    docs = {
        name: artifact.read_artifact(tmp_path / name / "artifact.pt")
        for name in ("one", "off", "sp", "mu")
    }
    # --extras 0 on the version-2 dataset: the artifact of the version-1 fit.
    one, off = docs["one"], docs["off"]
    assert off["featurizer"]["layout_version"] == 1
    assert sorted(off["featurizer"]) == sorted(one["featurizer"])
    assert off["predictor"]["config"] == one["predictor"]["config"]
    for name, value in one["predictor"]["state_dict"].items():
        assert np.array_equal(value, off["predictor"]["state_dict"][name]), name
    # The dataset's set table reaches only the artifact whose network reads it.
    assert docs["sp"]["featurizer"]["extras"] == ["sp_cand"]
    assert "matchup" not in docs["sp"]["featurizer"]
    assert docs["sp"]["featurizer"]["set_prior_accounts"] == [3, 7]
    assert docs["mu"]["featurizer"]["extras"] == ["mu_slot"]
    assert not [k for k in docs["mu"]["featurizer"] if k.startswith("set_prior")]
    report = json.loads((tmp_path / "sp" / "train_report.json").read_text())
    assert report["extra_scales"] == {"sp_cand": pytest.approx([0.01] * 4)}
    assert report["artifact_extras"] == ["sp_cand"]
    assert report["dataset"]["extras"] == list(ALL_NAMES)
    assert report["dataset"]["set_prior_accounts"] == 2
    loaded = artifact.load_predictor(tmp_path / "sp" / "artifact.pt", strict=True)
    assert loaded.featurizer.extras == ("sp_cand",) == loaded.predictor.net.extra_keys
    assert loaded.featurizer.set_table_holds("x") is False


# --- whose sheets built the set table ---------------------------------------------


def test_the_set_table_says_whose_sheets_built_it(fzs: F.Featurizer, tmp_path: Path):
    assert fzs.set_accounts == (3, 7)
    code = B.account_hash("somebody")
    held = F.Featurizer.build(
        TM.repertoire(), layout_version=2, set_table=TS.table(), set_accounts=[code]
    )
    assert held.set_table_holds("somebody") is True
    assert held.set_table_holds("somebodyelse") is False
    payload = held.to_payload()
    assert payload["set_prior_accounts"] == [code]
    again = F.Featurizer.from_payload(json.loads(json.dumps(_plain(payload))))
    assert again.set_accounts == (code,) and again.set_table_holds("somebody") is True
    held.save(tmp_path / "ds")
    assert F.Featurizer.load(tmp_path / "ds").set_table_holds("somebody") is True
    # Not recorded (a table from before this was stored): nothing can be said.
    old = {key: value for key, value in payload.items() if key != "set_prior_accounts"}
    legacy = F.Featurizer.from_payload(old)
    assert legacy.set_accounts is None and legacy.set_table_holds("somebody") is None
    assert "set_prior_accounts" not in legacy.to_payload()
    # No table, no answer.
    assert F.Featurizer.build(TM.repertoire()).set_table_holds("somebody") is None


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# --- the builder ------------------------------------------------------------------


class LaterCorpus(TD.Corpus):
    """``TD.Corpus`` plus one later ladder game and open sheets of both opponents.

    ``self.opponent`` (the old ladder opponent) and ``self.later`` (the
    opponent of a ladder page saved after the cut) each show an open sheet in
    a human log. The old pages are dated before the cut, the later one after.
    """

    LATER_FOLDER = "ladder_replays_mc_zz_later"

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        taken = {
            self.train_a,
            self.train_b,
            self.train_d,
            self.test_c,
            self.val_e,
            self.opponent,
            self.stranger,
        }
        self.later = TD.account(B.SPLIT_TRAIN, taken)
        a, b = self.train_a, self.train_b
        self.sheet_logs = {
            TD.replay_id(40): [2000, TD.make_log(a, self.opponent, sheets=True)],
            TD.replay_id(41): [2100, TD.make_log(self.later, b, sheets=True)],
            TD.replay_id(42): [2200, TD.make_log(b, a, sheets=True)],
        }
        self.write_json(B.T6LIKE_DIR, TD.FMT, self.sheet_logs)
        folder = root / self.LATER_FOLDER
        folder.mkdir()
        self.later_id = TD.replay_id(950)
        self.later_page = folder / f"{TD.BOT} - battle-{self.later_id}.html"
        self.later_page.write_text(
            TD.page(TD.make_log(self.later, TD.BOT, exact="p2", ratings=(1300, 1290)))
        )
        (folder / "decisions.jsonl").write_text(
            json.dumps(
                {
                    "battle": f"battle-{self.later_id}",
                    "turn": 0,
                    "preview_shadow": {"open_sheet": False},
                }
            )
        )
        before = F.local_time("2026-10-04T12:00")
        after = F.local_time("2026-10-09T10:30")
        for path in (root / "ladder_replays_mc_synthetic").glob("*.html"):
            os.utime(path, (before, before))
        os.utime(self.later_page, (after, after))
        self.before, self.after = before, after


@pytest.fixture(scope="module")
def later(tmp_path_factory: pytest.TempPathFactory) -> LaterCorpus:
    return LaterCorpus(tmp_path_factory.mktemp("later"))


@pytest.fixture(scope="module")
def whole(later: LaterCorpus) -> dict[str, Any]:
    return later.build("whole", layout_version=2)


@pytest.fixture(scope="module")
def early(later: LaterCorpus) -> dict[str, Any]:
    return later.build("early", layout_version=2, own_before=CUT_DAY)


def crc(name: str) -> int:
    return B.account_hash(E.user_id(name))


def test_a_version_2_build_needs_the_own_source(
    later: LaterCorpus, whole: dict[str, Any]
):
    human = [name for name in B.SOURCES if name != "own"]
    with pytest.raises(SystemExit) as refused:
        later.build("no_own", layout_version=2, sources=human)
    assert "own source" in str(refused.value)
    assert not (later.root / "results_oppmodel" / "no_own" / "manifest.json").exists()
    # A version-1 build has no set table and is not refused.
    assert later.build("no_own_v1", sources=human)["layout_version"] == 1
    # With the own source nobody the bot met contributes a sheet ...
    kept = whole["set_prior"]
    assert kept["own_source_read"] is True and kept["own_pages_not_offered"] == 0
    listed = kept["contributing_account_crc32"]
    assert listed == sorted(listed) and len(listed) == kept["contributing_accounts"]
    assert crc(later.opponent) not in listed and crc(later.later) not in listed
    assert crc(later.train_a) in listed and crc(later.train_b) in listed
    # ... and the featurizer (so an artifact) can be asked.
    folder = later.root / "results_oppmodel"
    with_own = F.Featurizer.load(folder / "whole", strict=True)
    assert with_own.set_accounts == tuple(listed)
    assert with_own.set_table_holds(E.user_id(later.opponent)) is False
    assert with_own.set_table_holds(E.user_id(later.train_a)) is True
    # Told to go on without it, the build says what it is: both opponents'
    # own sheets are in the table, and a reader can see it.
    smoke = later.build(
        "smoke", layout_version=2, sources=human, allow_no_own_source=True
    )
    leaky = smoke["set_prior"]
    assert leaky["own_source_read"] is False and leaky["own_pages_not_offered"] == 3
    assert crc(later.opponent) in leaky["contributing_account_crc32"]
    assert crc(later.later) in leaky["contributing_account_crc32"]
    without = F.Featurizer.load(folder / "smoke")
    assert without.set_table_holds(E.user_id(later.opponent)) is True
    assert without.set_table_holds(E.user_id(later.later)) is True
    assert leaky["records"] > kept["records"]


def test_own_before_leaves_the_later_games_out(
    later: LaterCorpus, whole: dict[str, Any], early: dict[str, Any]
):
    folder = later.root / "results_oppmodel"
    all_rows, cut_rows = TD.battles(later, "whole"), TD.battles(later, "early")
    assert later.later_id in all_rows and later.later_id not in cut_rows
    assert set(all_rows) - set(cut_rows) == {later.later_id}
    assert whole["headline"]["ladder_holdout_games"] == 2
    assert early["headline"]["ladder_holdout_games"] == 1
    block = early["own_cutoff"]
    assert block["before"] == CUT_DAY and block["before_unix"] == F.local_time(CUT_DAY)
    assert block["pages_read"] == 2 and block["pages_later"] == 1
    assert block["later_pages_by_folder"] == {later.LATER_FOLDER: 1}
    assert block["later_opponents"] == 1
    assert block["later_pages_without_an_opponent"] == 0
    assert block["later_pages_opened_for_the_opponent"] is True
    assert block["later_opponents_kept_out"] is True
    assert early["set_prior"]["own_opponents_not_kept_out"] == 0
    assert early["headline"]["own_pages_left_out_by_cutoff"] == 1
    assert "own_cutoff" not in whole
    # Nothing of the later folder was read as an input: no page list, no audit.
    assert any(key.startswith(later.LATER_FOLDER) for key in whole["inputs"])
    assert not any(key.startswith(later.LATER_FOLDER) for key in early["inputs"])
    # What each build holds, by the games' own times.
    assert whole["own_time"]["ladder_holdout"]["latest"] == later.after
    assert early["own_time"]["ladder_holdout"]["latest"] == later.before
    assert early["own_time"]["ladder_holdout"]["games"] == 1
    assert early["own_time"]["own_unrated"]["games"] == 1
    # The later opponent is still a holdout account: every human battle has
    # the split it has in the whole build ...
    assert early["accounts"]["later_own_opponents"] == 1
    assert "later_own_opponents" not in whole["accounts"]
    for battle_id, row in cut_rows.items():
        assert row["split"] == all_rows[battle_id]["split"], battle_id
        assert row["holdout_battle"] == all_rows[battle_id]["holdout_battle"]
    met = TD.replay_id(41)
    assert cut_rows[met]["holdout_battle"] is True
    assert cut_rows[met]["split"] == {
        "p1": "excluded_holdout_player",
        "p2": "excluded_holdout_player",
    }
    # ... the set table is the same table, without the later opponent's sheet ...
    assert early["set_prior"]["signature"] == whole["set_prior"]["signature"]
    assert (
        early["set_prior"]["fold_signatures"] == whole["set_prior"]["fold_signatures"]
    )
    assert (
        early["set_prior"]["contributing_account_crc32"]
        == whole["set_prior"]["contributing_account_crc32"]
    )
    assert crc(later.later) not in early["set_prior"]["contributing_account_crc32"]
    assert early["set_prior"]["own_pages_not_offered"] == 0
    # ... and the examples are the whole build's minus the later game's rows.
    everything, _ = F.load_dataset(folder / "whole")
    smaller, _ = F.load_dataset(folder / "early")
    gone = everything["m_battle"] == all_rows[later.later_id]["index"]
    assert gone.any() and smaller["m_battle"].shape[0] == int((~gone).sum())
    for name, array in everything.items():
        assert smaller[name].tobytes() == array[~gone].tobytes(), name
    # Without the hygiene the later opponent's sheet WOULD be a training sheet.
    naive = B.assign_splits(B.run_pass1(*pass1_args(later), collect_sets=True).metas)
    assert E.user_id(later.later) not in naive["holdout_accounts"]


def pass1_args(corpus: LaterCorpus) -> tuple[Any, Any, Any]:
    reader = B.Corpus(
        corpus.root, B.DEFAULT_FORMATS, B.SOURCES, own_before=F.local_time(CUT_DAY)
    )
    return reader, reader.bot_accounts(), reader.clone_corpus_ids()


def test_a_later_page_is_never_driven_and_its_audit_never_read(
    later: LaterCorpus, monkeypatch: pytest.MonkeyPatch
):
    cut = F.local_time(CUT_DAY)
    reader = B.Corpus(later.root, B.DEFAULT_FORMATS, B.SOURCES, own_before=cut)
    read, late = reader.own_split()
    assert [battle_id for _, _, battle_id in late] == [later.later_id]
    assert len(read) == 2
    opened: list[str] = []
    real = Path.read_text

    def recording(self: Path, *args: Any, **kwargs: Any) -> str:
        opened.append(self.name)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", recording)
    yielded = [raw.battle_id for raw in reader if raw.own]
    audit = reader.sheet_audit()
    monkeypatch.undo()
    assert later.later_id not in yielded and len(yielded) == 2
    assert later.later_page.name not in opened
    assert later.later_id not in audit and TD.replay_id(900) in audit
    assert opened.count("decisions.jsonl") == 1  # the old folder's alone
    # The one thing taken of it: its opponent's account id.
    assert reader.late_opponents() == ({E.user_id(later.later)}, 0)
    # A page whose time cannot be told apart counts as later; one at the cut too.
    exact = B.Corpus(later.root, B.DEFAULT_FORMATS, B.SOURCES, own_before=later.after)
    assert len(exact.own_split()[1]) == 1
    loose = B.Corpus(
        later.root, B.DEFAULT_FORMATS, B.SOURCES, own_before=later.after + 1
    )
    assert loose.own_split()[1] == [] and loose.late_opponents() == (set(), 0)
    # Told not to open a later page at all: no name is read, and it says so.
    shut = B.Corpus(
        later.root, B.DEFAULT_FORMATS, B.SOURCES, own_before=cut, read_later_names=False
    )
    opened.clear()
    monkeypatch.setattr(Path, "read_text", recording)
    assert shut.late_opponents() == (set(), 1)
    assert [raw.battle_id for raw in shut if raw.own] == yielded
    monkeypatch.undo()
    assert later.later_page.name not in opened
    unread = later.build(
        "unread", layout_version=2, own_before=CUT_DAY, own_later_unread=True
    )
    block = unread["own_cutoff"]
    assert block["later_pages_opened_for_the_opponent"] is False
    assert block["later_opponents_kept_out"] is False
    assert block["later_opponents"] == 0 and block["pages_later"] == 1
    assert unread["set_prior"]["own_opponents_not_kept_out"] == 1
    # ... and a reader can see whose sheet got in.
    listed = unread["set_prior"]["contributing_account_crc32"]
    assert crc(later.later) in listed and crc(later.opponent) not in listed
    with pytest.raises(SystemExit):
        later.build("refused", own_later_unread=True)
    # Without a cut everything is read, as always.
    plain = B.Corpus(later.root, B.DEFAULT_FORMATS, B.SOURCES)
    assert len(plain.own_split()[0]) == 3 and plain.own_split()[1] == []
    assert later.later_id in plain.sheet_audit()
    # Bad arguments are refused before anything is written.
    for options in (
        {"own_before": "next week"},
        {"own_before": CUT_DAY, "sources": ["top_merged"]},
    ):
        with pytest.raises(SystemExit):
            later.build("refused", **options)
    assert not (later.root / "results_oppmodel" / "refused").exists()


def test_a_reader_tells_the_later_rows_by_their_own_time(
    later: LaterCorpus, whole: dict[str, Any], early: dict[str, Any]
):
    folder = later.root / "results_oppmodel"
    cut = F.local_time(CUT_DAY)
    assert (
        F.local_time("2026-10-09T00:00") == cut == F.local_time("2026-10-09T00:00:00")
    )
    assert time.localtime(cut)[:5] == (2026, 10, 9, 0, 0)
    with pytest.raises(ValueError):
        F.local_time("09/10/2026")
    held = F.own_time_range(folder / "whole", cut)
    assert held["has_time"] is True and held["late_rows"] > 0
    ladder = held["splits"]["ladder_holdout"]
    assert ladder["latest"] == later.after and ladder["earliest"] == later.before
    assert ladder["late_rows"] == held["late_rows"] and ladder["rows_without_time"] == 0
    assert F.own_time_range(folder / "early", cut)["late_rows"] == 0
    assert (
        "late_rows" not in F.own_time_range(folder / "whole")["splits"]["own_unrated"]
    )
    # Filtering the whole build's rows gives the early build's holdout.
    names = ["ladder_holdout", "own_unrated"]
    want, _ = F.load_dataset(folder / "early", splits=names)
    got, _ = F.load_dataset(folder / "whole", splits=names, own_before=cut)
    everything, _ = F.load_dataset(folder / "whole", splits=names)
    assert everything["m_time"].shape[0] > got["m_time"].shape[0] > 0
    for name, array in want.items():
        assert got[name].tobytes() == array.tobytes(), name
    # Human rows are never touched by it, whatever their time.
    human, _ = F.load_dataset(folder / "whole", splits=["train", "val", "test"])
    kept, _ = F.load_dataset(
        folder / "whole", splits=["train", "val", "test"], own_before=1
    )
    assert kept["m_time"].shape[0] == human["m_time"].shape[0] > 0
    # An unknown time counts as later.
    codes = [3, 4]
    assert F.late_own_rows([3, 3, 4, 0], [0, cut - 1, cut, 0], codes, cut).tolist() == [
        True,
        False,
        True,
        False,
    ]


# --- the trainer's informational reading ------------------------------------------


def dated_dataset(
    directory: Path, fz: F.Featurizer, early: int, late: int | None = None
) -> dict[str, int]:
    """``TT.write_dataset`` with ``m_time``.

    Every ladder row is dated ``early``; with ``late`` the second half of
    them is dated ``late`` instead (``sizes["late"]`` rows).
    """
    sizes = TT.write_dataset(directory, fz)
    shard = directory / "shard-00000.npz"
    data = F.load_batch(shard)
    when = np.full(int(data["m_split"].shape[0]), 1_000, dtype=np.int64)
    ladder = np.flatnonzero(data["m_split"] == TT.SPLITS.index("ladder_holdout"))
    when[ladder] = early
    sizes["late"] = 0
    if late is not None:
        when[ladder[len(ladder) // 2 :]] = late
        sizes["late"] = int(len(ladder) - len(ladder) // 2)
    data["m_time"] = when
    F.save_batch(shard, data)
    return sizes


def test_informational_never_scores_the_sealed_games_unasked(
    fz1: F.Featurizer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    artifact_module()
    sealed_from = F.local_time(F.OWN_SEALED_FROM)
    data = tmp_path / "data"
    sizes = dated_dataset(data, fz1, sealed_from - 86_400, sealed_from + 3_600)
    assert sizes["late"] > 0 and sizes["ladder_holdout"] > sizes["late"]
    plan = T.informational_plan(T.parse_args(["--informational"]), data)
    assert plan is not None and plan["splits"] == ["test", "ladder_holdout"]
    assert plan["sealed_rows_in_dataset"] == sizes["late"]
    assert plan["own_before"] == sealed_from and plan["sealed_rows_read"] is False
    assert T.informational_plan(T.parse_args([]), data) is None
    capsys.readouterr()
    assert TT.run(data, tmp_path / "info", "--epochs", "1", "--informational") == 0
    lines = TT.lines_of(capsys)
    report = json.loads((tmp_path / "info" / "train_report.json").read_text())
    assert any("sealed" in line and "left out" in line for line in lines)
    scored = report["informational"]["ladder_holdout"]["examples"]
    assert scored == sizes["ladder_holdout"] - sizes["late"]
    assert report["informational"]["test"]["examples"] == sizes["test"]
    assert report["informational_plan"]["sealed_rows_in_dataset"] == sizes["late"]
    # One split alone can be asked for; the test split is then not read.
    assert (
        TT.run(
            data,
            tmp_path / "ladder",
            "--epochs",
            "1",
            "--informational",
            "--informational-splits",
            "ladder_holdout",
        )
        == 0
    )
    only = json.loads((tmp_path / "ladder" / "train_report.json").read_text())
    assert set(only["informational"]) == {"ladder_holdout"}
    assert only["informational"]["ladder_holdout"]["examples"] == scored
    # Reading them takes the explicit flag.
    assert (
        TT.run(
            data,
            tmp_path / "all",
            "--epochs",
            "1",
            "--informational",
            "--allow-sealed-holdout",
        )
        == 0
    )
    everything = json.loads((tmp_path / "all" / "train_report.json").read_text())
    assert (
        everything["informational"]["ladder_holdout"]["examples"]
        == sizes["ladder_holdout"]
    )
    assert everything["informational_plan"]["sealed_rows_read"] is True
    # A split that is not offered is refused before any training.
    capsys.readouterr()
    assert (
        TT.run(
            data,
            tmp_path / "bad",
            "--epochs",
            "1",
            "--informational",
            "--informational-splits",
            "train",
        )
        != 0
    )
    lines = TT.lines_of(capsys)
    assert lines[-1].startswith("TRAIN_FAILED") and not any(
        line.startswith("EPOCH") for line in lines
    )
    # A dataset without later games is read whole, with nothing to say.
    old = tmp_path / "old"
    dated_dataset(old, fz1, sealed_from - 86_400)
    plan = T.informational_plan(T.parse_args(["--informational"]), old)
    assert plan is not None and plan["sealed_rows_in_dataset"] == 0
    assert plan["own_before"] is None and plan["own_time_known"] is True


def test_copy_of_the_extras_table_is_not_shared_state():
    # The tests above patch ``EXTRA_ARRAYS``; the module's own table is intact.
    assert [spec.name for spec in F.EXTRA_ARRAYS] == list(ALL_NAMES)
    assert copy.deepcopy(F.EXTRA_ARRAYS) == F.EXTRA_ARRAYS == M.EXTRA_ARRAYS
