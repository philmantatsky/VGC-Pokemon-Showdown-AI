"""Learning-curve driver: names, owned directories, pairing, the corpus check,
the verdict sentence, and one whole run on a synthetic corpus.

The pairing tests build two "builds" by hand: the same things happened, but the
candidate lists and the row order differ, as they do between two real builds.
The whole-run tests use the synthetic feed of ``test_oppmodel_dataset`` (with a
few more saved own games), tiny one-epoch fits and the real stages.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from evaluation import oppmodel_scorecard as SC
from training import oppmodel_learning_curve as LC
from unit_tests import test_oppmodel_dataset as TD
from vgc_bench.src.oppmodel import features as F

TINY_FIT = "--d-model 32 --layers 1 --heads 2 --ff 48 --batch 64 --warmup 2"


# --- names ----------------------------------------------------------------------


def test_fractions_are_parsed_sorted_and_checked():
    assert LC.parse_fractions("0.5, 0.125,0.25,0.5") == [0.125, 0.25, 0.5]
    assert LC.parse_fractions("1") == [1.0]
    for bad in ("", "0", "1.5", "-0.1", "half", "0.5,,nope"):
        with pytest.raises(LC.CurveError):
            LC.parse_fractions(bad)
    assert [LC.percent_tag(f) for f in (0.125, 0.25, 0.5, 1.0, 0.0625)] == [
        "12p5",
        "25",
        "50",
        "100",
        "6p25",
    ]
    assert LC.percent_text(0.125) == "12.5%" and LC.percent_text(1.0) == "100%"


def arguments(tmp_path: Path, *extra: str) -> Any:
    return LC.resolve(
        LC.parse_args(
            [
                "--base-tag",
                "c_x",
                "--results-root",
                str(tmp_path / "results"),
                "--repo-root",
                str(tmp_path),
                *extra,
            ]
        )
    )


def test_default_plan_reads_the_given_full_point_and_builds_the_fractions(
    tmp_path: Path,
):
    args = arguments(tmp_path)
    assert args.fraction_list == [0.125, 0.25, 0.5] and args.elo_mode == "blank"
    assert args.threads == 1 and args.resamples == SC.DEFAULT_RESAMPLES
    assert args.build_limit is None and args.train_limit is None and args.epochs == 40
    points = LC.plan_points(args)
    results = (tmp_path / "results").resolve()
    assert [(p.name, p.built) for p in points] == [
        ("c_x_f12p5", True),
        ("c_x_f25", True),
        ("c_x_f50", True),
        ("v2_feed", False),
    ]
    assert points[0].dataset == results / "c_x_f12p5"
    assert points[0].artifact == results / "c_x_f12p5_oppnet" / "artifact.pt"
    assert points[0].tables is None and points[0].tables_dir is None
    full = points[-1]
    assert full.fraction == 1.0 and full.fit_dir is None
    assert full.dataset == tmp_path.resolve() / "results_oppmodel" / "v2_feed"
    assert full.artifact == (
        tmp_path.resolve() / "results_oppmodel" / "oppnet_v2_blind" / "artifact.pt"
    )
    assert Path(args.out) == tmp_path.resolve() / "results" / "c_x_curve"
    with_tables = LC.plan_points(arguments(tmp_path, "--with-tables"))
    assert with_tables[0].tables == results / "c_x_f12p5_tables" / "flags_table.pt"
    assert with_tables[-1].tables == (
        tmp_path.resolve() / "results_oppmodel" / "tables_v2" / "flags_table.pt"
    )


def test_smoke_is_tiny_and_builds_its_own_full_point(tmp_path: Path):
    args = arguments(tmp_path, "--smoke", "--fractions", "0.25,0.5")
    assert args.fraction_list == [0.25, 0.5, 1.0]
    assert (args.build_limit, args.train_limit, args.epochs) == (300, 3000, 1)
    assert args.tables_limit == 3000 and args.resamples == LC.SMOKE_RESAMPLES
    points = LC.plan_points(args)
    assert [(p.name, p.built) for p in points] == [
        ("c_x_f25", True),
        ("c_x_f50", True),
        ("c_x_f100", True),
    ]
    # A fraction of 1.0 on the command line does the same without --smoke.
    assert [
        p.built for p in LC.plan_points(arguments(tmp_path, "--fractions", "0.5,1"))
    ] == [True, True]


# --- the directories a run may write --------------------------------------------


def test_owned_dir_refuses_everything_that_is_not_the_base_tags(tmp_path: Path):
    results = tmp_path / "results"
    results.mkdir()
    for name in ("c_x_f25", "c_x_f25_oppnet", "c_x_curve", "c_x_"):
        assert LC.owned_dir(results / name, "c_x", results) == results.resolve() / name
    refused = [
        results / "v2_feed",  # another experiment's dataset
        results / "c_x",  # the bare tag: no directory of the run has that name
        results / "c_xy_f25",  # another tag that begins the same
        results / "c_y_f25",
        results / "c_x_f25" / "inner",  # not directly in the results folder
        results,
        tmp_path / "elsewhere" / "c_x_f25",
        results / "c_x_f25" / ".." / "v2_feed",  # resolves to a foreign directory
        results / ".." / "c_x_f25",
    ]
    for path in refused:
        with pytest.raises(LC.CurveError, match="refusing to touch"):
            LC.owned_dir(path, "c_x", results)
    for tag in ("", ".", "..", "../c_x", "a/b", "_x", "c x", "c_x/"):
        with pytest.raises(LC.CurveError, match="--base-tag"):
            LC.owned_dir(results / "anything", tag, results)
        with pytest.raises(LC.CurveError):
            LC.check_base_tag(tag)
    # A link from an owned name to a foreign directory is the foreign directory.
    (results / "v2_feed").mkdir()
    os.symlink(results / "v2_feed", results / "c_x_link")
    with pytest.raises(LC.CurveError, match="refusing to touch"):
        LC.owned_dir(results / "c_x_link", "c_x", results)


def tree(root: Path) -> dict[str, str]:
    """Every file under ``root`` with the hash of its bytes."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def run_main(*argv: str) -> tuple[int, list[str]]:
    """``LC.main`` with its printed lines (the trainer's included)."""
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        code = LC.main(list(argv))
    return code, [line for line in stream.getvalue().splitlines() if line.strip()]


def test_a_run_refuses_an_output_directory_that_is_not_its_own(tmp_path: Path):
    results = tmp_path / "results"
    for name in ("v2_feed", "scorecard_v2", "c_y_curve"):
        (results / name).mkdir(parents=True)
        (results / name / "manifest.json").write_text('{"tag": "theirs"}')
    before = tree(tmp_path)
    base = ["--results-root", str(results), "--repo-root", str(tmp_path), "--smoke"]
    cases = [
        ["--base-tag", "c_x", "--out", str(results / "scorecard_v2")],
        ["--base-tag", "c_x", "--out", str(results / "c_y_curve")],
        ["--base-tag", "c_x", "--out", str(tmp_path / "c_x_curve")],
        ["--base-tag", "c_x", "--out", str(results / "c_x_curve" / "deeper")],
        # Named after the base tag, but it is somebody's dataset.
        ["--base-tag", "v2", "--out", str(results / "v2_feed")],
        ["--base-tag", "../c_x"],
    ]
    for case in cases:
        code, lines = run_main(*base, *case)
        assert code == 1 and lines[-1].startswith(LC.FAILED), case
        assert "refusing" in lines[-1] or "--base-tag" in lines[-1], lines[-1]
        # Nothing was made, built, or changed.
        assert tree(tmp_path) == before
        assert sorted(p.name for p in results.iterdir()) == [
            "c_y_curve",
            "scorecard_v2",
            "v2_feed",
        ]


def test_a_dot_file_in_the_curve_directory_does_not_block_a_rescoring(tmp_path: Path):
    """The results folder is in iCloud Drive: opening the curve's directory in
    the Finder leaves a .DS_Store, and the documented "--overwrite" must still
    score again. Anything else that is not a curve is still refused."""
    args = arguments(tmp_path, "--overwrite")
    out = Path(args.out)
    out.mkdir(parents=True)
    (out / LC.CURVE_JSON).write_text("{}")
    (out / LC.CURVE_MD).write_text("old")
    assert LC.prepare_out(args) == (out.resolve(), False)
    (out / ".DS_Store").write_bytes(b"finder")
    assert LC.prepare_out(args) == (out.resolve(), False)
    assert (out / ".DS_Store").read_bytes() == b"finder"  # and it is left alone
    # without --overwrite an earlier curve is still not scored over
    with pytest.raises(LC.CurveError, match="--overwrite"):
        LC.prepare_out(arguments(tmp_path))
    (out / "notes.txt").write_text("somebody's")
    with pytest.raises(LC.CurveError, match="it holds"):
        LC.prepare_out(args)


@pytest.mark.parametrize(
    "extra, named",
    [
        ("--out {victim} --overwrite", "out"),
        ("--out {victim}", "out"),
        ("--overwrite", "overwrite"),
        ("--dataset {victim}", "dataset"),
        ("--tag other", "tag"),
        ("--device mps", "device"),
        ("--epochs 3", "epochs"),
        ("--limit 10", "limit"),
        ("--seed 1", "seed"),
        ("--elo-mode keep", "elo_mode"),
    ],
)
def test_train_extra_cannot_send_the_fit_elsewhere(
    tmp_path: Path, extra: str, named: str
):
    """``--train-extra`` is appended to the trainer's command line, where a
    later option wins: ``--out X --overwrite`` made the trainer remove X's
    artifact and write a fit there. It may add options, never change what the
    driver sets, and it is refused before anything is touched."""
    victim = tmp_path / "victim_model"
    victim.mkdir()
    (victim / "artifact.pt").write_bytes(b"somebody else's weights")
    (victim / "train_report.json").write_text('{"status": "TRAIN_DONE"}')
    before = tree(tmp_path)
    results = tmp_path / "results"
    code, lines = run_main(
        "--base-tag",
        "c_x",
        "--fractions",
        "0.5,1.0",
        "--results-root",
        str(results),
        "--repo-root",
        str(tmp_path),
        # one argument: a lone "--overwrite" would be read as the driver's own
        "--train-extra=" + extra.format(victim=victim),
    )
    assert code == 1 and lines[-1].startswith(f"{LC.FAILED} CurveError")
    assert "--train-extra may not change" in lines[-1] and named in lines[-1]
    assert tree(tmp_path) == before and not results.exists()


def test_train_extra_may_add_what_the_driver_does_not_set(tmp_path: Path):
    args = arguments(tmp_path, "--train-extra", TINY_FIT)
    fit = tmp_path / "results" / "c_x_f50_oppnet"
    argv, wanted = LC.checked_trainer_args(tmp_path / "data", fit, args)
    assert argv[: len(LC.trainer_argv(tmp_path / "data", fit, args))] == (
        LC.trainer_argv(tmp_path / "data", fit, args)
    )
    assert argv[-len(TINY_FIT.split()) :] == TINY_FIT.split()
    assert (wanted["d_model"], wanted["batch"], wanted["device"]) == (32, 64, "cpu")
    assert wanted["out"] == str(fit) and wanted["tag"] == "c_x_f50_oppnet"
    assert wanted["overwrite"] is False and wanted["epochs"] == 40
    assert set(LC.PINNED_TRAIN_ARGS) <= set(wanted)
    with pytest.raises(LC.CurveError, match="bad trainer arguments"):
        with contextlib.redirect_stderr(io.StringIO()):
            LC.checked_trainer_args(
                "d", "o", arguments(tmp_path, "--train-extra", "--no-such-option 1")
            )
    with pytest.raises(LC.CurveError, match="cannot be read"):
        LC.checked_trainer_args("d", "o", arguments(tmp_path, "--train-extra", "'"))


# --- pairing two builds, by hand --------------------------------------------------

MOVES = ("", "alpha", "beta", "gamma", "delta", "epsilon")  # a tiny vocabulary
N_CAND = 3
OTHER = N_CAND
NO_FLAGS = (0, 1, 0, 1, 0, 0)  # switch known, protect known


def slot(
    move: str | None = None,
    *,
    switch: int | None = None,
    kind: int | None = None,
    target: int = -1,
    intent: int = -1,
    mega: int = 0,
    reason: int = 0,
    attack: tuple[int, int] = (-1, -1),
    flags: tuple[int, ...] = NO_FLAGS,
    fine: float = 1.0,
    scored: bool = True,
) -> dict[str, Any]:
    """What one slot did, said without any candidate list."""
    if kind is None:
        kind = (
            F.Y_KIND_SWITCH
            if switch is not None
            else (F.Y_KIND_MOVE if move is not None else F.Y_KIND_NONE)
        )
    return {
        "move": move,
        "switch": switch,
        "kind": kind,
        "target": target,
        "intent": intent,
        "mega": mega,
        "reason": reason,
        "attack": attack,
        "flags": list(flags),
        "fine": fine,
        "scored": scored,
    }


def view(
    examples: list[tuple[LC.Key, tuple[dict[str, Any], dict[str, Any]]]],
    candidates: tuple[str, ...],
) -> LC.EvalView:
    """A build's view of ``examples`` under its own candidate list.

    A move that is not among ``candidates`` becomes the OTHER action with the
    "outside the candidates" flag, as the featurizer encodes it.
    """
    n = len(examples)
    labels = {
        "y_kind": np.full((n, 2), F.Y_KIND_ABSENT, dtype=np.int8),
        "y_action": np.full((n, 2), -1, dtype=np.int8),
        "y_target": np.full((n, 2), -1, dtype=np.int8),
        "y_mega": np.full((n, 2), -1, dtype=np.int8),
        "y_intent": np.full((n, 2), -1, dtype=np.int8),
        "y_reason": np.zeros((n, 2), dtype=np.uint8),
        "y_attack": np.full((n, 2, 2), -1, dtype=np.int8),
        "y_flag": np.zeros((n, 2, F.N_Y_FLAG), dtype=np.uint8),
        "cand_move": np.zeros((n, 2, N_CAND), dtype=np.uint16),
    }
    fine = np.zeros((n, 2))
    scored = np.zeros((n, 2), dtype=bool)
    for row, (_, slots) in enumerate(examples):
        for index, what in enumerate(slots):
            labels["cand_move"][row, index] = [MOVES.index(m) for m in candidates]
            labels["y_kind"][row, index] = what["kind"]
            labels["y_target"][row, index] = what["target"]
            labels["y_mega"][row, index] = what["mega"]
            labels["y_intent"][row, index] = what["intent"]
            labels["y_reason"][row, index] = what["reason"]
            labels["y_attack"][row, index] = what["attack"]
            labels["y_flag"][row, index] = what["flags"]
            if what["switch"] is not None:
                labels["y_action"][row, index] = N_CAND + 1 + what["switch"]
            elif what["move"] is not None and what["move"] in candidates:
                labels["y_action"][row, index] = candidates.index(what["move"])
            elif what["move"] is not None:
                labels["y_action"][row, index] = OTHER
                labels["y_flag"][row, index, F.Y_OTHER] = 1
            fine[row, index] = what["fine"]
            scored[row, index] = what["scored"]
    return LC.EvalView(
        keys=[key for key, _ in examples],
        labels=labels,
        moves=MOVES,
        n_cand=N_CAND,
        fine=fine,
        scored=scored,
    )


def games() -> list[tuple[LC.Key, tuple[dict[str, Any], dict[str, Any]]]]:
    """Three games, six examples: moves, a switch, a hidden slot, an empty one."""
    return [
        (("g1", 1, 0), (slot("alpha", target=1, intent=2), slot("beta", intent=4))),
        (("g1", 2, 0), (slot(switch=3, intent=0), slot("gamma", target=0, intent=3))),
        (("g2", 1, 1), (slot("beta", attack=(1, 0)), slot(kind=F.Y_KIND_HIDDEN))),
        (("g2", 2, 1), (slot("alpha", mega=1), slot(scored=False))),
        (("g3", 1, 0), (slot("delta", target=2), slot("alpha", reason=3))),
        (("g3", 2, 0), (slot(switch=5), slot("beta", flags=(0, 1, 1, 1, 0, 0)))),
    ]


def with_fine(
    examples: list[tuple[LC.Key, tuple[dict[str, Any], dict[str, Any]]]],
    values: list[tuple[float, float]],
) -> list[tuple[LC.Key, tuple[dict[str, Any], dict[str, Any]]]]:
    out = []
    for (key, slots), pair in zip(examples, values):
        out.append((key, tuple({**s, "fine": v} for s, v in zip(slots, pair))))
    return out  # type: ignore[return-value]


SMALL_FINE = [(2.0, 1.0), (3.0, 1.5), (2.5, 0.5), (1.0, 9.0), (2.0, 2.0), (4.0, 1.0)]
LARGE_FINE = [(1.5, 1.0), (2.0, 1.0), (2.5, 1.0), (0.5, 7.0), (1.0, 2.5), (3.0, 0.5)]


def test_two_builds_pair_by_key_whatever_their_rows_and_candidates():
    small = view(with_fine(games(), SMALL_FINE), ("alpha", "beta", "gamma"))
    # The larger build: other rows first, another candidate order, one more move
    # named ("delta" is a candidate here and OTHER in the small build).
    order = [4, 2, 0, 5, 3, 1]
    shuffled = [with_fine(games(), LARGE_FINE)[i] for i in order]
    large = view(shuffled, ("delta", "alpha", "beta"))
    back = np.argsort(order)  # large's rows in the small build's order
    assert [large.keys[i] for i in back] == small.keys
    assert not np.array_equal(small.labels["y_action"], large.labels["y_action"][back])
    found = LC.paired_step(small, large, resamples=400, seed=5)
    assert found["error"] is None
    assert found["examples_matched"] == 6
    assert (found["examples_only_smaller"], found["examples_only_larger"]) == (0, 0)
    labels = found["labels"]
    assert labels["slots_compared"] == 12 and labels["slots_mismatched"] == 0
    assert labels["mismatches"] == {} and labels["examples"] == []
    # "delta" is named only by the large build, "gamma" only by the small one.
    assert labels["move_named_in_one_build"] == 2
    # Larger minus smaller over the eleven slots both score (g2 turn 2 slot b
    # is scored by neither), paired by key.
    deltas = [
        large_value - small_value
        for small_pair, large_pair, (_, slots) in zip(SMALL_FINE, LARGE_FINE, games())
        for small_value, large_value, what in zip(small_pair, large_pair, slots)
        if what["scored"]
    ]
    assert len(deltas) == 11
    difference = found["difference"]
    assert difference["diff"] == pytest.approx(sum(deltas) / 11)
    assert (difference["slots"], difference["games"]) == (11, 3)
    assert difference["low"] <= difference["diff"] <= difference["high"]
    # It is the scorecard's own paired difference on the matched rows.
    direct = SC.paired_difference(
        large.fine[back],
        small.fine,
        small.scored & large.scored[back],
        np.array([0, 0, 1, 1, 2, 2]),
        resamples=400,
        seed=5,
    )
    assert difference == direct


def test_the_two_players_of_one_turn_are_two_examples():
    """The key holds the side: both players' examples of one turn of one game
    (opposite seats, as among the test players) are matched each to its own."""
    both = [
        (("g1", 1, 0), (slot("alpha", fine=1.0), slot("beta", fine=2.0))),
        (("g1", 1, 1), (slot("gamma", fine=3.0), slot("alpha", fine=4.0))),
        (("g2", 3, 0), (slot("beta", fine=5.0), slot("beta", fine=6.0))),
        (("g2", 3, 1), (slot("alpha", fine=7.0), slot("gamma", fine=8.0))),
    ]
    candidates = ("alpha", "beta", "gamma")
    small = view(both, candidates)
    drops = [(1.0, 0.5), (2.0, 0.25), (4.0, 1.0), (3.0, 0.5)]
    lowered = [
        (key, tuple({**w, "fine": w["fine"] - less} for w, less in zip(slots, pair)))
        for (key, slots), pair in zip(both, drops)
    ]
    large = view([lowered[i] for i in (3, 0, 2, 1)], candidates)  # type: ignore[misc]
    found = LC.paired_step(small, large, resamples=0)
    assert found["error"] is None
    assert found["examples_matched"] == 4 and found["examples_repeated_key"] == 0
    assert (found["examples_only_smaller"], found["examples_only_larger"]) == (0, 0)
    assert found["labels"]["slots_compared"] == 8
    assert found["labels"]["slots_mismatched"] == 0
    assert found["difference"]["slots"] == 8 and found["difference"]["games"] == 2
    assert found["difference"]["diff"] == pytest.approx(
        -sum(less for pair in drops for less in pair) / 8
    )
    # Without the side the two seats of a turn share a key: neither row could
    # be told from the other, and nothing would be matched.
    blind = [(battle, turn, 0) for battle, turn, _ in small.keys]
    match = LC.match_keys(blind, [(b, t, 0) for b, t, _ in large.keys])
    assert match["first"].size == 0 and match["repeated_first"] == 4
    # ... and eval_view takes the side from m_side.
    batch = {
        "m_battle": np.array([0, 0, 1, 1]),
        "m_turn": np.array([1, 1, 3, 3]),
        "m_side": np.array([0, 1, 0, 1]),
    }

    def no_scores(pred: Any, data: Any) -> dict[str, np.ndarray]:
        return {"fine": np.zeros((4, 2)), "fine_scored": np.ones((4, 2), dtype=bool)}

    featurizer: Any = SimpleNamespace(vocab=SimpleNamespace(moves=MOVES), n_cand=N_CAND)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(LC.F, "slot_nll", no_scores)
        made = LC.eval_view(batch, ["g1", "g2"], featurizer, {})
    assert made.keys == [key for key, _ in both] and len(set(made.keys)) == 4


def control_view(
    examples: list[tuple[LC.Key, tuple[dict[str, Any], dict[str, Any]]]],
    candidates: tuple[str, ...],
) -> LC.EvalView:
    """A build under a predictor that knows nothing: a move the build names
    costs 2.0, one that falls in OTHER 2.5 (another label space), a switch
    1.0, a slot without a visible move 3.0."""
    priced = []
    for key, slots in examples:
        row = []
        for what in slots:
            if what["switch"] is not None:
                cost = 1.0
            elif what["move"] is None:
                cost = 3.0
            else:
                cost = 2.0 if what["move"] in candidates else 2.5
            row.append({**what, "fine": cost})
        priced.append((key, (row[0], row[1])))
    return view(priced, candidates)


def test_a_step_carries_the_label_space_control():
    """A predictor that knows nothing "improves" or "gets worse" between two
    builds through their candidate lists alone. The step reports that control,
    the model's step net of it, and the model's step on the slots both builds
    label alike."""
    names = ("alpha", "beta", "gamma")
    small = view(with_fine(games(), SMALL_FINE), names)
    plain_small = control_view(games(), names)
    # The model's step, slot by slot, on the eleven slots both score.
    steps = {
        (key, index): large_value - small_value
        for (key, slots), small_pair, large_pair in zip(games(), SMALL_FINE, LARGE_FINE)
        for index, (what, small_value, large_value) in enumerate(
            zip(slots, small_pair, large_pair)
        )
        if what["scored"]
    }
    assert len(steps) == 11

    # 1. The larger build names "delta" (g3/1 slot a) and loses "gamma" (g1/2
    #    slot b): the control moves by -0.5 and +0.5 there, 0 in all.
    other = ("delta", "alpha", "beta")
    large = view(with_fine(games(), LARGE_FINE), other)
    plain_large = control_view(games(), other)
    bare = LC.paired_step(small, large, resamples=300, seed=4)
    assert bare["label_space_only"] is None and bare["net_of_label_space"] is None
    found = LC.paired_step(
        small, large, resamples=300, seed=4, control=(plain_small, plain_large)
    )
    assert found["difference"] == bare["difference"]
    assert found["same_label_space"] == bare["same_label_space"]
    assert found["difference"]["diff"] == pytest.approx(sum(steps.values()) / 11)
    control = found["label_space_only"]
    assert control["slots"] == 11 and control["diff"] == pytest.approx(0.0)
    assert found["net_of_label_space"]["diff"] == pytest.approx(
        found["difference"]["diff"]
    )
    alike = found["same_label_space"]
    unlike = [(("g1", 2, 0), 1), (("g3", 1, 0), 0)]
    assert alike["slots"] == 9 and found["labels"]["move_named_in_one_build"] == 2
    assert alike["diff"] == pytest.approx(
        (sum(steps.values()) - sum(steps[where] for where in unlike)) / 9
    )

    # 2. The larger build names "delta" and "gamma" but not "beta" (three
    #    slots): the control gets WORSE by 3 x 0.5 - 0.5 over eleven slots, and
    #    the model's step net of it is better than it looks.
    other = ("delta", "gamma", "alpha")
    large = view(with_fine(games(), LARGE_FINE), other)
    plain_large = control_view(games(), other)
    found = LC.paired_step(
        small, large, resamples=300, seed=4, control=(plain_small, plain_large)
    )
    control, net = found["label_space_only"], found["net_of_label_space"]
    assert control["diff"] == pytest.approx(1.0 / 11)
    assert control["slots"] == net["slots"] == 11 and net["games"] == 3
    assert net["diff"] == pytest.approx(found["difference"]["diff"] - 1.0 / 11)
    assert net["low"] <= net["diff"] <= net["high"]
    assert found["same_label_space"]["slots"] == 11 - 4
    assert found["labels"]["move_named_in_one_build"] == 4
    # The control is the same examples under another predictor, nothing else.
    with pytest.raises(LC.CurveError, match="same examples"):
        LC.paired_step(
            small, large, control=(control_view(games()[::-1], names), plain_large)
        )


def test_a_reversed_order_without_keys_would_give_another_number():
    """The control of the test above: matching by row number is not the same."""
    small = view(with_fine(games(), SMALL_FINE), ("alpha", "beta", "gamma"))
    large = view(with_fine(games(), LARGE_FINE)[::-1], ("alpha", "beta", "gamma"))
    by_key = LC.paired_step(small, large, resamples=0)["difference"]["diff"]
    by_row = float((large.fine - small.fine)[small.scored & large.scored].mean())
    assert abs(by_key - by_row) > 0.05


MUTATIONS: dict[str, tuple[str, Any]] = {
    "y_kind": ("kind", F.Y_KIND_HIDDEN),
    "y_reason": ("reason", 2),
    "y_mega": ("mega", 1),
    "y_intent": ("intent", 5),
    "y_attack": ("attack", (0, 1)),
    "y_flag": ("flags", (1, 1, 0, 1, 0, 0)),
    "move": ("move", "beta"),
    "y_target": ("target", 3),
    "switch_to": ("switch", 2),
    "acted": ("move", None),
}


@pytest.mark.parametrize("field", sorted(MUTATIONS))
def test_a_label_that_differs_between_builds_is_detected_and_named(field: str):
    small = view(games(), ("alpha", "beta", "gamma"))
    changed = games()
    name, value = MUTATIONS[field]
    # Row 0 slot a is "alpha at target 1"; row 1 slot a is a switch to roster 3.
    row = 1 if field == "switch_to" else 0
    key, slots = changed[row]
    slots[0][name] = value
    if field == "acted":
        slots[0]["kind"] = F.Y_KIND_MOVE  # same kind, but no visible choice
    large = view(changed, ("alpha", "beta", "gamma"))
    found = LC.paired_step(small, large, resamples=50)
    assert found["error"] == "label_mismatch"
    assert found["difference"] is None  # never averaged over
    labels = found["labels"]
    assert labels["slots_mismatched"] == 1 and field in labels["mismatches"]
    example = next(e for e in labels["examples"] if e["field"] == field)
    assert (example["battle"], example["turn"], example["side"]) == key
    assert example["slot"] == 0 and example["first"] != example["second"]
    # The step is a failure of the run, not a warning.
    steps = [{"from": "s", "to": "l", "sets": {LC.SET_LADDER: found}}]
    failures, _ = LC.step_problems(steps)
    assert len(failures) == 1 and "label mismatch" in failures[0]
    assert field in failures[0]


def test_what_only_one_build_can_express_is_counted_not_a_mismatch():
    small = view(games(), ("alpha", "epsilon", "gamma"))  # "beta" is OTHER here
    changed = games()
    changed[0][1][0]["target"] = -1  # a target the larger build could not give
    large = view(changed, ("alpha", "beta", "gamma"))
    assert small.labels["y_flag"][0, 1, F.Y_OTHER] == 1
    assert large.labels["y_flag"][0, 1, F.Y_OTHER] == 0
    found = LC.paired_step(small, large, resamples=50)
    assert found["error"] is None and found["difference"] is not None
    labels = found["labels"]
    assert labels["slots_mismatched"] == 0
    # beta in g1/1, g2/1 and g3/2 (named by the large build only), delta nowhere.
    assert labels["move_named_in_one_build"] == 3
    assert labels["target_in_one_build"] == 1
    # A slot one build scores and the other does not is left out, and counted.
    large.scored[2, 0] = False
    again = LC.paired_step(small, large, resamples=50)
    assert again["labels"]["scored_in_one_build"] == 1
    assert again["difference"]["slots"] == found["difference"]["slots"] - 1


def test_examples_of_one_build_only_and_repeated_keys_are_counted():
    first = [("a", 1, 0), ("a", 2, 0), ("b", 1, 1), ("c", 1, 0), ("c", 1, 0)]
    second = [("b", 1, 1), ("a", 1, 0), ("d", 1, 0), ("c", 1, 0), ("e", 1, 0)]
    match = LC.match_keys(first, second)
    assert match["first"].tolist() == [0, 2] and match["second"].tolist() == [1, 0]
    assert (match["only_first"], match["only_second"]) == (1, 2)
    # ("c", 1, 0) is twice in the first build: neither row can be matched.
    assert (match["repeated_first"], match["repeated_second"]) == (2, 1)
    assert len(first) == 2 + 1 + 2 and len(second) == 2 + 2 + 1
    small = view(games(), ("alpha", "beta", "gamma"))
    large = view(
        games()[2:] + [(("g9", 1, 0), (slot("alpha"), slot("beta")))], MOVES[1:4]
    )
    found = LC.paired_step(small, large, resamples=50)
    assert found["examples_matched"] == 4
    assert (found["examples_only_smaller"], found["examples_only_larger"]) == (2, 1)
    assert found["difference"]["games"] == 2
    steps = [{"from": "s", "to": "l", "sets": {LC.SET_LADDER: found}}]
    failures, warnings = LC.step_problems(steps)
    assert not failures
    assert len(warnings) == 1 and "not the same examples" in warnings[0]
    nothing = LC.paired_step(
        small, view([(("z", 1, 0), (slot("alpha"), slot()))], MOVES[1:4])
    )
    assert nothing["error"] == "no_common_examples" and nothing["difference"] is None


# --- one corpus -------------------------------------------------------------------


def manifest(feed: dict[str, int], *, fraction: float | None = None) -> dict[str, Any]:
    """A manifest with one feed shard per entry (name -> records read)."""
    inputs: dict[str, Any] = {
        f"battle_logs_feed_mc/logs/x/{name}": {
            "kind": "feed_shard",
            "sha256": f"{name}:{count}",
            "n": count,
        }
        for name, count in feed.items()
    }
    inputs["battle_logs_web_mc_20260927/top"] = {
        "kind": "folder",
        "sha256": "w",
        "n": 5,
    }
    inputs["ladder_replays_mc_x"] = {"kind": "folder", "sha256": str(fraction), "n": 9}
    inputs["ladder_replays_mc_x/decisions.jsonl"] = {
        "kind": "decision_audit",
        "sha256": f"audit{fraction}",
    }
    offered = sum(feed.values()) + 5
    out: dict[str, Any] = {
        "inputs": inputs,
        "battles": {"read": {"feed": sum(feed.values()), "web_top": 5, "own": 9}},
    }
    if fraction is not None:
        kept = int(offered * fraction)
        out["battles"]["read"] = {"feed": kept, "own": 9}
        out["human_fraction"] = {"fraction": fraction, "logs_offered": offered}
    return out


def test_the_corpus_check_tells_one_corpus_from_two():
    full = manifest({"part-0": 1000, "part-1": 995})
    same = manifest({"part-0": 1000, "part-1": 995}, fraction=0.25)
    report = LC.corpus_report({"f25": same, "full": full}, "full")
    # Own games and the decision audit are not the human corpus.
    assert set(LC.human_inputs(full)) == {
        "battle_logs_feed_mc/logs/x/part-0",
        "battle_logs_feed_mc/logs/x/part-1",
        "battle_logs_web_mc_20260927/top",
    }
    assert report["identical_inputs"] and report["largest_drift"] == 0.0
    assert (
        report["points"]["f25"]["human_logs_offered"] == 2000 == LC.logs_offered(full)
    )
    assert LC.check_corpus(report, 0.02) == []
    # The feed grew by ten games between the two builds: tolerated, and said.
    grown = manifest({"part-0": 1000, "part-1": 1005}, fraction=0.25)
    report = LC.corpus_report({"f25": grown, "full": full}, "full")
    assert not report["identical_inputs"]
    assert report["points"]["f25"]["inputs_differing"] == [
        "battle_logs_feed_mc/logs/x/part-1"
    ]
    assert report["largest_drift"] == pytest.approx(10 / 2000)
    warnings = LC.check_corpus(report, 0.02)
    assert len(warnings) == 1 and "Not exactly one corpus" in warnings[0]
    assert "2,010" in warnings[0] and "2,000" in warnings[0]
    # A full point built on a much older download: the run stops.
    later = manifest({"part-0": 1000, "part-1": 1000, "part-2": 300}, fraction=0.25)
    report = LC.corpus_report({"f25": later, "full": full}, "full")
    assert report["largest_drift"] == pytest.approx(305 / 2000)
    with pytest.raises(LC.CurveError, match="not one corpus"):
        LC.check_corpus(report, 0.02)
    assert len(LC.check_corpus(report, 0.5)) == 1  # accepted on request, and said


# --- the sentence -----------------------------------------------------------------


def curve_with(difference: dict[str, Any] | None, battles: tuple[int, int]) -> Any:
    step = {
        "from": "b_f50",
        "to": "full",
        "from_label": "50%",
        "to_label": "100%",
        "human_battles": list(battles),
        "battle_ratio": battles[1] / battles[0],
        "sets": {
            LC.SET_LADDER: {
                "difference": difference,
                "error": None if difference else "label_mismatch",
            }
        },
    }
    return {"steps": [step]}


def test_the_sentence_says_whether_the_last_doubling_helps_and_by_how_much():
    helps = {"diff": -0.0412, "low": -0.06, "high": -0.02, "slots": 4810, "games": 397}
    text = LC.verdict(curve_with(helps, (13_570, 26_847)))
    assert text.startswith("The last doubling, 50% -> 100% of the corpus")
    assert "(13,570 -> 26,847 human battles)" in text
    assert "-0.0412 [-0.0600, -0.0200] nats" in text and "397 games" in text
    assert text.endswith("more data still helps, by 0.041 nats.")
    flat = {"diff": -0.004, "low": -0.02, "high": 0.012, "slots": 4810, "games": 397}
    text = LC.verdict(curve_with(flat, (13_570, 26_847)))
    assert "includes zero" in text and "still helps" not in text
    worse = {"diff": 0.03, "low": 0.01, "high": 0.05, "slots": 4810, "games": 397}
    assert "WORSE by 0.030 nats" in LC.verdict(curve_with(worse, (13_570, 26_847)))
    # Not a doubling: said as a step, with the rate per doubling.
    text = LC.verdict(curve_with(helps, (3_821, 26_847)))
    assert text.startswith("The last step (7.03 times the battles)")
    assert "per doubling" in text and "The last doubling" not in text
    assert "no reading" in LC.verdict(curve_with(None, (13_570, 26_847)))
    assert "label_mismatch" in LC.verdict(curve_with(None, (13_570, 26_847)))
    assert "no step to judge" in LC.verdict({"steps": []})


ALIKE = {key: None for key in LC.HYPER_KEYS} | {"epochs": 40, "seed": 20261004}
HELPS = {"diff": -0.0412, "low": -0.06, "high": -0.02, "slots": 4810, "games": 397}
JUDGEMENTS = ("still helps", "WORSE", "includes zero", "not beyond")


def whole_curve(
    difference: dict[str, Any] = HELPS,
    battles: tuple[int, int] = (13_570, 26_847),
    *,
    settings: dict[str, Any] | None = None,
    args: tuple[dict[str, Any], dict[str, Any]] = (ALIKE, ALIKE),
    drift: float | None = 0.0,
    weighted: tuple[float, float] | None = None,
    control: dict[str, Any] | None = None,
    net: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A curve as the driver writes it, as far as the sentence reads it."""
    curve = curve_with(difference, battles)
    found = curve["steps"][0]["sets"][LC.SET_LADDER]
    found["label_space_only"], found["net_of_label_space"] = control, net
    if weighted is not None:
        curve["steps"][0]["weighted_ratio"] = weighted[1] / weighted[0]
    curve["settings"] = {
        "smoke": False,
        "build_limit": None,
        "train_limit": None,
        "tables_limit": None,
        **(settings or {}),
    }
    curve["points"] = [
        {"name": "b_f50", "train_args": dict(args[0])},
        {"name": "full", "train_args": dict(args[1])},
    ]
    curve["corpus"] = {"largest_drift": drift}
    return curve


def test_the_sentence_needs_the_step_to_be_beyond_the_label_space_control():
    flat = {"diff": -0.0063, "low": -0.0095, "high": -0.0034, "slots": 4810, "games": 9}
    net = {"diff": -0.0349, "low": -0.05, "high": -0.01, "slots": 4810, "games": 397}
    curve = whole_curve(control=flat, net=net)
    assert LC.reading_problems(curve) == []
    text = LC.verdict(curve)
    assert text.startswith("The last doubling, 50% -> 100% of the corpus")
    assert "a predictor that knows nothing moves by -0.0063 [-0.0095, -0.0034]" in text
    assert "net of that the step is -0.0349 [-0.0500, -0.0100]" in text
    assert text.endswith("more data still helps, by 0.041 nats.")
    # The model's step is below zero, but not by more than the control shows.
    weak = {"diff": -0.0100, "low": -0.03, "high": 0.004, "slots": 4810, "games": 397}
    text = LC.verdict(whole_curve(control=flat, net=weak))
    assert "still helps, by" not in text and "WORSE" not in text
    assert text.endswith("so it does not show that more data still helps.")
    assert "not beyond what the two candidate lists alone do" in text
    # A net step without an interval is not "below zero" either.
    open_net = {"diff": -0.03, "low": None, "high": None, "slots": 10, "games": 1}
    assert "not beyond" in LC.verdict(whole_curve(control=flat, net=open_net))
    # An interval that includes zero, or a worse step, is said as before.
    zero = {"diff": -0.004, "low": -0.02, "high": 0.012, "slots": 4810, "games": 397}
    assert "includes zero" in LC.verdict(whole_curve(zero, control=flat, net=zero))


def test_per_doubling_is_counted_in_what_the_loss_sees():
    # 1.98 times the battles, but 1.65 times the cap-weighted training examples.
    curve = whole_curve(weighted=(126_543.0, 208_568.0))
    ratio = 208_568.0 / 126_543.0
    text = LC.verdict(curve)
    assert text.startswith(
        "The last step (1.98 times the battles, 1.65 times the cap-weighted "
        "training examples), 50% -> 100% of the corpus"
    )
    rate = 0.0412 / math.log2(ratio)
    assert text.endswith(
        f"more data still helps, by 0.041 nats, {rate:.3f} nats per doubling of "
        "the cap-weighted training examples at this rate."
    )
    assert "The last doubling" not in text
    # a doubling of both is called one, and says both numbers
    text = LC.verdict(whole_curve(weighted=(100_000.0, 201_000.0)))
    assert text.startswith(
        "The last doubling (1.98 times the battles, 2.01 times the cap-weighted "
        "training examples), 50% -> 100%"
    )
    assert "per doubling" not in text


@pytest.mark.parametrize(
    "change, head, reason",
    [
        ({"settings": {"smoke": True}}, "SMOKE, NO READING: ", "a smoke run"),
        (
            {"settings": {"train_limit": 3000}},
            "NO READING: ",
            "the fits were limited to 3,000 training examples",
        ),
        (
            {"settings": {"build_limit": 300}},
            "NO READING: ",
            "the builds were limited to 300 logs per source",
        ),
        (
            {"settings": {"tables_limit": 3000}},
            "NO READING: ",
            "the table fits were limited to 3,000 examples per split",
        ),
        (
            {"args": ({**ALIKE, "epochs": 1, "limit": 3000}, ALIKE)},
            "NO READING: ",
            "b_f50 and full were not trained alike "
            "(epochs 1 vs 40, limit 3000 vs None)",
        ),
        (
            {"args": ({**ALIKE, "lr": 0.01}, {**ALIKE, "lr": 0.001})},
            "NO READING: ",
            "were not trained alike (lr 0.01 vs 0.001)",
        ),
        (
            {"args": ({}, ALIKE)},
            "NO READING: ",
            "the training arguments of b_f50 are not recorded",
        ),
        (
            {"args": (ALIKE, {})},
            "NO READING: ",
            "the training arguments of full are not recorded",
        ),
        (
            {"drift": 0.1269},
            "NO READING: ",
            "did not read one corpus (they differ by 12.69% of the human logs",
        ),
    ],
)
def test_points_that_were_not_made_alike_give_no_reading(
    change: dict[str, Any], head: str, reason: str
):
    """The paired step of two fits that were not made alike, or not in full, is
    a number and not a finding: the sentence says so instead of concluding."""
    curve = whole_curve(**change)
    problems = LC.reading_problems(curve)
    assert len(problems) == 1 and reason in problems[0]
    text = LC.verdict(curve)
    assert text.startswith(head) and reason in text
    assert "-0.0412 [-0.0600, -0.0200] nats" in text  # the number is still given
    assert not any(word in text for word in JUDGEMENTS), text
    assert "says nothing about whether more data helps" in text
    # whichever way the number points
    for other in (
        {"diff": 0.03, "low": 0.01, "high": 0.05, "slots": 9, "games": 4},
        {"diff": -0.004, "low": -0.02, "high": 0.012, "slots": 9, "games": 4},
    ):
        again = LC.verdict(whole_curve(other, **change))
        assert again.startswith(head) and not any(w in again for w in JUDGEMENTS)


def test_the_curve_that_was_on_disk_is_no_reading():
    """results_oppmodel/c_curve_real_curve: a 3,000-example one-epoch fit
    against the full 23-epoch one across a 12.69% corpus drift. Its file said
    "more data still helps, by 0.625 nats"."""
    big = {"diff": -0.625, "low": -0.6709, "high": -0.5804, "slots": 4810, "games": 397}
    curve = whole_curve(
        big,
        (3_821, 26_847),
        settings={"train_limit": 3000, "epochs": 1, "max_corpus_drift": 1.0},
        args=({**ALIKE, "epochs": 1, "limit": 3000}, ALIKE),
        drift=0.1269,
    )
    problems = LC.reading_problems(curve)
    assert len(problems) == 3
    text = LC.verdict(curve)
    assert text.startswith("NO READING: the fits were limited to 3,000 training")
    assert "epochs 1 vs 40, limit 3000 vs None" in text and "12.69%" in text
    assert "still helps" not in text and "per doubling" not in text
    # a drift the default limit accepts is not a reason, and neither are equal
    # arguments; a curve without points (older files, hand-made ones) has
    # nothing to compare
    assert LC.reading_problems(whole_curve(drift=LC.DEFAULT_DRIFT)) == []
    assert LC.reading_problems(whole_curve(drift=None)) == []
    assert LC.reading_problems(curve_with(HELPS, (13_570, 26_847))) == []


def test_the_cap_is_read_for_one_set_of_accounts_at_every_point(tmp_path: Path):
    """An account over the cap in the full build is capped less, or not at
    all, in a smaller one: its examples weigh relatively more there. Counting
    "accounts over the cap in this build" hides that (the small build has
    fewer of them), so the heaviest accounts are one fixed set."""

    def build(name: str, rows: list[tuple[int, int, float]]) -> tuple[Path, dict]:
        """A build from (split code, account code, weight) per example."""
        directory = tmp_path / name
        directory.mkdir()
        half = len(rows) // 2
        files = []
        for index, part in enumerate((rows[:half], rows[half:])):
            split, actor, weight = (np.array(column) for column in zip(*part))
            np.savez_compressed(
                directory / f"shard-{index:05d}.npz",
                m_split=split.astype(np.uint8),
                m_actor=actor.astype(np.uint32),
                m_weight=weight.astype(np.float32),
            )
            files.append({"file": f"shard-{index:05d}.npz", "examples": len(part)})
        manifest = {"splits": ["train", "val"], "shards": files, "account_cap": 2}
        return directory, manifest

    # The full build: account 7 has eight training examples at weight 1/4,
    # account 9 four at weight 1/2, accounts 1-4 one each at weight 1; and two
    # validation examples that must not count.
    full_rows = (
        [(0, 7, 0.25)] * 8 + [(0, 9, 0.5)] * 4 + [(0, a, 1.0) for a in (1, 2, 3, 4)]
    )
    full_rows += [(1, 7, 0.25), (1, 5, 1.0)]
    full, full_manifest = build("full", full_rows)
    # The half build: account 7 has four examples (weight 1/2), account 9 two
    # (weight 1: no longer over the cap), accounts 1 and 2 one each.
    half_rows = [(0, 7, 0.5)] * 4 + [(0, 9, 1.0)] * 2 + [(0, 1, 1.0), (0, 2, 1.0)]
    half_rows += [(1, 7, 0.5), (1, 5, 1.0)]
    half, half_manifest = build("half", half_rows)

    heavy = LC.over_cap_accounts(full, full_manifest)
    assert heavy.tolist() == [7, 9]
    assert LC.over_cap_accounts(half, half_manifest).tolist() == [7]
    whole = LC.cap_weighting(full, full_manifest, heavy, "full")
    assert whole["train_examples"] == 16 and whole["account_cap"] == 2
    assert whole["train_examples_weighted"] == pytest.approx(8 * 0.25 + 4 * 0.5 + 4)
    assert (
        whole["examples_of_accounts_over_cap"] == 12 and whole["accounts_over_cap"] == 2
    )
    assert whole["share_over_cap"] == pytest.approx(12 / 16)
    assert whole["share_over_cap_weighted"] == pytest.approx(4 / 8)
    assert whole["heaviest_accounts"] == {
        "over_the_cap_in": "full",
        "accounts": 2,
        "share": pytest.approx(12 / 16),
        "share_weighted": pytest.approx(4 / 8),
    }
    small = LC.cap_weighting(half, half_manifest, heavy, "full")
    assert small["train_examples"] == 8
    assert small["train_examples_weighted"] == pytest.approx(4 * 0.5 + 2 + 2)
    # In this build only account 7 is over the cap ...
    assert small["accounts_over_cap"] == 1
    assert small["share_over_cap_weighted"] == pytest.approx(2 / 6)
    # ... but the SAME two accounts hold two thirds of what the loss sees here,
    # against one half in the full build: the mix is not the full build's.
    assert small["heaviest_accounts"]["share"] == pytest.approx(6 / 8)
    assert small["heaviest_accounts"]["share_weighted"] == pytest.approx(4 / 6)
    assert (
        small["heaviest_accounts"]["share_weighted"]
        > (whole["heaviest_accounts"]["share_weighted"])
    )
    # 16 -> 8 examples is a halving; in what the loss sees it is 8 -> 6.
    step = LC.make_steps(
        [
            {"name": "h", "label": "50%", "human_battles": 10, "cap": small},
            {"name": "f", "label": "100%", "human_battles": 20, "cap": whole},
        ],
        [{}, {}],
        resamples=0,
        seed=0,
    )[0]
    assert step["battle_ratio"] == 2.0
    assert step["weighted_ratio"] == pytest.approx(8 / 6)
    assert step["train_examples_weighted"] == [pytest.approx(6.0), pytest.approx(8.0)]
    # without the set, no such block; without a training split, nothing
    assert "heaviest_accounts" not in LC.cap_weighting(full, full_manifest)
    assert LC.cap_weighting(full, {"splits": ["val"], "shards": []}) == {}
    assert LC.over_cap_accounts(full, {"splits": ["val"]}).size == 0
    gone = {**full_manifest, "shards": [{"file": "shard-00009.npz"}]}
    with pytest.raises(LC.CurveError, match="shard-00009"):
        LC.cap_weighting(full, gone)


# --- builds on disk ---------------------------------------------------------------


def test_dataset_state_and_what_makes_a_build_another_build(tmp_path: Path):
    assert LC.dataset_state(tmp_path / "missing") == ("absent", None)
    directory = tmp_path / "c_x_f25"
    directory.mkdir()
    (directory / "shard-00000.npz").write_bytes(b"half written")
    assert LC.dataset_state(directory) == ("partial", None)
    written = {
        "tag": "c_x_f25",
        "limit": None,
        "sources_requested": list(LC.B.SOURCES),
        "shards": [{"file": "shard-00000.npz", "examples": 1}],
        "human_fraction": {"fraction": 0.25, "seed": 0, "hash": LC.B.FRACTION_HASH},
    }
    (directory / "manifest.json").write_text(json.dumps(written))
    assert LC.dataset_state(directory)[0] == "damaged"  # featurizer files missing
    for name in LC.DATASET_FILES:
        (directory / name).write_text("x")
    state, found = LC.dataset_state(directory)
    assert state == "complete" and found == written
    args = arguments(tmp_path, "--fractions", "0.25")
    point = LC.plan_points(args)[0]
    assert LC.build_differences(written, point, args) == []
    other = arguments(
        tmp_path, "--fractions", "0.25", "--seed", "4", "--build-limit", "9"
    )
    assert LC.build_differences(written, point, other) == ["seed 0", "limit None"]
    assert LC.build_differences({**written, "tag": "v2_feed"}, point, args) == [
        "tag 'v2_feed'"
    ]
    plain = {key: value for key, value in written.items() if key != "human_fraction"}
    assert LC.build_differences(plain, point, args) == ["not a subsample"]
    crc = {**written, "human_fraction": {"fraction": 0.25, "seed": 0, "hash": "crc32"}}
    assert LC.build_differences(crc, point, args) == ["subsample hash 'crc32'"]
    whole = LC.plan_points(arguments(tmp_path, "--fractions", "1.0"))[0]
    assert LC.build_differences(written, whole, args)[:2] == [
        "tag 'c_x_f25'",
        "a subsample",
    ]


# --- a whole run on the synthetic corpus -------------------------------------------

LADDER_FOES = 4


class Run:
    """One finished run of the driver and where it lives."""

    def __init__(self, root: Path) -> None:
        self.corpus = TD.FractionCorpus(root)
        # A few more saved ladder games, each against another opponent, so the
        # ladder holdout has games to resample.
        own = root / "ladder_replays_mc_synthetic"
        for number in range(902, 902 + LADDER_FOES):
            foe = f"Ladder Foe {number}"  # an account of no human battle
            path = own / f"{TD.BOT} - battle-{TD.replay_id(number)}.html"
            moves = ("Dragon Claw", "Outrage", "Stomping Tantrum", "Earthquake")
            log = TD.make_log(
                foe, TD.BOT, exact="p2", ratings=(1200, 1210), p2_move=moves[number % 4]
            )
            path.write_text(TD.page(log))
            os.utime(path, (1_790_000_000 + number, 1_790_000_000 + number))
        self.root = root
        self.results = root / "results"
        self.argv = [
            "--base-tag",
            "t",
            "--fractions",
            "0.5,1.0",
            "--seed",
            str(TD.SEED),
            "--repo-root",
            str(root),
            "--results-root",
            str(self.results),
            "--epochs",
            "1",
            "--train-extra",
            TINY_FIT,
            "--resamples",
            "60",
            "--with-tables",
        ]
        self.code, self.lines = run_main(*self.argv)
        self.out = self.results / "t_curve"

    def curve(self) -> dict[str, Any]:
        return json.loads((self.out / LC.CURVE_JSON).read_text())


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory) -> Run:
    return Run(tmp_path_factory.mktemp("curve"))


def test_a_run_builds_fits_scores_and_pairs(finished: Run):
    assert finished.code == 0, finished.lines[-3:]
    assert finished.lines[-1] == f"{LC.DONE} {finished.out.resolve() / LC.CURVE_JSON}"
    curve = finished.curve()
    assert curve["format"] == LC.FORMAT and curve["status"] == LC.DONE
    assert curve["settings"]["subsample_hash"] == "sha256"
    assert curve["settings"]["elo_mode"] == "blank"
    assert curve["settings"]["seed"] == TD.SEED
    # Every directory the run made is named after its base tag.
    assert sorted(p.name for p in finished.results.iterdir()) == [
        "t_curve",
        "t_f100",
        "t_f100_oppnet",
        "t_f100_tables",
        "t_f50",
        "t_f50_oppnet",
        "t_f50_tables",
    ]
    half, full = curve["points"]
    assert (half["name"], full["name"]) == ("t_f50", "t_f100")
    assert (half["label"], full["label"]) == ("50%", "100%")
    assert half["built_by_this_run"] and full["built_by_this_run"]
    manifests = {
        name: json.loads((finished.results / name / "manifest.json").read_text())
        for name in ("t_f50", "t_f100")
    }
    assert "human_fraction" not in manifests["t_f100"]
    assert manifests["t_f50"]["human_fraction"]["fraction"] == 0.5
    for row in (half, full):
        headline = manifests[row["name"]]["headline"]
        assert row["human_battles"] == headline["human_battles_kept"]
        by_split = manifests[row["name"]]["examples"]["by_split"]
        assert row["train_examples_in_dataset"] == by_split["train"]
        assert row["train_examples_used"] == by_split["train"]
        assert row["elo_mode"] == "blank" and row["best_epoch"] in (0, 1)
        assert row["ladder_holdout_games"] == 1 + LADDER_FOES
        for set_name in LC.SETS:
            found = row["sets"][set_name]
            for who in (LC.MODEL, LC.BAR):
                fine = found[who]["fine_nll"]
                assert fine["low"] <= fine["value"] <= fine["high"]
                assert found[who]["action_nll"] + found[who]["target_nll"] == (
                    pytest.approx(fine["value"])
                )
                assert 0.0 <= found[who]["top1"] <= found[who]["top3"] <= 1.0
                assert 0.0 <= found[who]["intent_accuracy"] <= 1.0
            assert found[LC.BAR]["minus_bar"] is None
            assert found[LC.MODEL]["minus_bar"]["diff"] == pytest.approx(
                found[LC.MODEL]["fine_nll"]["value"]
                - found[LC.BAR]["fine_nll"]["value"]
            )
    assert half["human_battles"] < full["human_battles"] == 100
    ladder = [row["sets"][LC.SET_LADDER] for row in (half, full)]
    assert ladder[0]["examples"] == ladder[1]["examples"] == 2 * (1 + LADDER_FOES)
    assert ladder[0]["games"] == ladder[1]["games"] == 1 + LADDER_FOES
    assert half["sets"][LC.SET_TEST]["examples"] < full["sets"][LC.SET_TEST]["examples"]
    (step,) = curve["steps"]
    assert (step["from"], step["to"]) == ("t_f50", "t_f100")
    assert step["human_battles"] == [half["human_battles"], 100]
    on_ladder = step["sets"][LC.SET_LADDER]
    assert on_ladder["error"] is None and on_ladder["examples_matched"] == 10
    assert on_ladder["examples_only_smaller"] == on_ladder["examples_only_larger"] == 0
    assert on_ladder["labels"]["slots_mismatched"] == 0
    difference = on_ladder["difference"]
    assert difference["games"] == 1 + LADDER_FOES
    assert difference["low"] <= difference["diff"] <= difference["high"]
    # The unpaired means of the two points differ by the paired difference:
    # the same slot-turns are scored under both.
    assert difference["diff"] == pytest.approx(
        full["sets"][LC.SET_LADDER][LC.MODEL]["fine_nll"]["value"]
        - half["sets"][LC.SET_LADDER][LC.MODEL]["fine_nll"]["value"]
    )
    # (b): the smaller build's test examples are all in the larger build.
    on_test = step["sets"][LC.SET_TEST]
    assert on_test["examples_matched"] == half["sets"][LC.SET_TEST]["examples"]
    assert on_test["examples_only_smaller"] == 0 and on_test["examples_only_larger"] > 0
    assert on_test["labels"]["slots_mismatched"] == 0
    assert curve["corpus"]["identical_inputs"] and curve["corpus"]["largest_drift"] == 0
    assert curve["sentence"] == LC.verdict(curve)
    assert curve["sentence"] in finished.lines
    assert "50% -> 100% of the corpus" in curve["sentence"]
    # Both points were built and trained by the run, alike and in full: the
    # sentence is a reading.
    assert curve["reading"] == {"valid": True, "problems": []}
    assert LC.NO_READING not in curve["sentence"]
    assert (
        half["train_args"] == full["train_args"] and half["train_args"]["epochs"] == 1
    )
    # The account cap, read from each build's own shards.
    for row in (half, full):
        cap = row["cap"]
        data, _ = F.load_dataset(finished.results / row["name"], ["train"])
        assert cap["train_examples"] == row["train_examples_in_dataset"]
        assert cap["train_examples_weighted"] == pytest.approx(
            float(data["m_weight"].astype(np.float64).sum())
        )
        assert 0 < cap["train_examples_weighted"] <= cap["train_examples"]
        over = data["m_weight"] < 1.0
        assert cap["examples_of_accounts_over_cap"] == int(over.sum())
        assert cap["account_cap"] == manifests[row["name"]]["account_cap"]
        # One set of accounts for both points: those the 100% point caps.
        assert cap["heaviest_accounts"]["over_the_cap_in"] == "t_f100"
    assert step["train_examples_weighted"] == [
        half["cap"]["train_examples_weighted"],
        full["cap"]["train_examples_weighted"],
    ]
    assert step["weighted_ratio"] == pytest.approx(
        full["cap"]["train_examples_weighted"] / half["cap"]["train_examples_weighted"]
    )
    # The label-space control of every step, on the same slot-turns.
    for found in (on_ladder, on_test):
        control, net = found["label_space_only"], found["net_of_label_space"]
        assert control["slots"] == net["slots"] == found["difference"]["slots"]
        assert net["diff"] == pytest.approx(
            found["difference"]["diff"] - control["diff"]
        )
        assert found["same_label_space"]["slots"] <= found["difference"]["slots"]
    # ... which is the uniform predictor scored on each build's own labels.
    plain = []
    for name in ("t_f50", "t_f100"):
        data, _ = F.load_dataset(finished.results / name, [LC.SET_LADDER])
        nll = F.slot_nll(F.uniform_prediction(SC.features_only(data)), data)
        plain.append(float(nll["fine"][nll["fine_scored"]].mean()))
    assert on_ladder["label_space_only"]["diff"] == pytest.approx(plain[1] - plain[0])


def test_the_ladder_holdout_rows_sit_at_other_row_numbers_in_each_build(finished: Run):
    """Why the pairing goes through battles.jsonl: own games come after the human
    ones, so the same game has another ``m_battle`` in a smaller build."""
    rows = {}
    for name in ("t_f50", "t_f100"):
        data, found = F.load_dataset(finished.results / name, [LC.SET_LADDER])
        ids = LC.battle_ids(finished.results / name)
        assert len(ids) == found["headline"]["human_battles_kept"] + 2 + LADDER_FOES
        rows[name] = (sorted(set(data["m_battle"].tolist())), ids)
    assert rows["t_f50"][0] != rows["t_f100"][0]
    named = [{ids[i] for i in numbers} for numbers, ids in rows.values()]
    assert named[0] == named[1] and len(named[0]) == 1 + LADDER_FOES


def test_the_report_is_rendered_from_the_json_alone(finished: Run):
    text = (finished.out / LC.CURVE_MD).read_text()
    curve = finished.curve()
    assert text == LC.render(curve)
    assert f"**{curve['sentence']}**" in text
    for needle in ("## Points", "## Paired steps", "## One corpus", "## Timings"):
        assert needle in text
    assert "| 50% | `t_f50` |" in text and "| 100% | `t_f100` |" in text
    assert "FlagsTable fitted on that build" in text and "SMOKE RUN" not in text
    assert LC.NO_READING not in text
    for needle in ("| cap-weighted |", "knows nothing [95%]", "same label space"):
        assert needle in text, needle
    # The sentence is computed from the curve when the file is rendered: a
    # sentence stored by an older driver cannot contradict the data under it.
    stale = dict(curve, sentence="more data still helps, by 9.999 nats.")
    assert LC.render(stale) == text
    limited = json.loads(json.dumps(curve))
    limited["settings"]["train_limit"] = 3000
    told = LC.render(limited)
    assert f"**{LC.NO_READING} OF THE LEARNING CURVE: the fits were limited" in told
    assert f"**{LC.NO_READING}: the fits were limited to 3,000" in told
    assert f"**{curve['sentence']}**" not in told
    (finished.out / LC.CURVE_MD).write_text("stale")
    code, lines = run_main(*finished.argv, "--render-only")
    assert code == 0 and lines[-1].startswith(LC.DONE)
    assert (finished.out / LC.CURVE_MD).read_text() == text


def test_a_second_run_reuses_every_build_and_fit(finished: Run):
    before = tree(finished.results)
    code, lines = run_main(*finished.argv)
    assert code == 1 and lines[-1].startswith(LC.FAILED)
    assert "--overwrite" in lines[-1]
    assert tree(finished.results) == before
    code, lines = run_main(*finished.argv, "--overwrite")
    assert code == 0, lines[-3:]
    reused = [line for line in lines if line.endswith("reused")]
    assert len(reused) == 6  # two builds, two fits, two table fits
    assert not any(line.startswith("TRAIN_START") for line in lines)
    after = tree(finished.results)
    changed = sorted(name for name in after if after[name] != before.get(name))
    assert changed in ([], ["t_curve/curve.json", "t_curve/curve.md"])
    assert set(after) == set(before)
    timings = finished.curve()["points"][0]["timings"]
    assert (
        timings["build_reused"] and timings["fit_reused"] and timings["tables_reused"]
    )


def test_a_stopped_fit_is_done_again_and_a_finished_one_is_not(
    finished: Run, tmp_path: Path
):
    """Resuming: leftovers without an artifact are this run's to redo."""
    results = tmp_path / "results"
    argv = [a if a != str(finished.results) else str(results) for a in finished.argv]
    argv = [a for a in argv if a != "--with-tables"]
    fit = results / "t_f50_oppnet"
    fit.mkdir(parents=True)
    (fit / "train_report.json").write_text('{"status": "TRAIN_RUNNING"}')
    (fit / "artifact.pt.unverified").write_bytes(b"cut short")
    code, lines = run_main(*argv)
    assert code == 0, lines[-3:]
    assert (fit / "artifact.pt").is_file()
    assert not (fit / "artifact.pt.unverified").exists()
    assert json.loads((fit / "train_report.json").read_text())["status"] == "TRAIN_DONE"
    assert sum(line.startswith("TRAIN_START") for line in lines) == 2


def test_a_limited_run_ends_with_no_reading_in_every_place_a_reader_looks(
    finished: Run, tmp_path: Path
):
    """A run with --train-limit is a pipeline check. It still ends CURVE_DONE,
    but the sentence, the last line, curve.json and curve.md all say that its
    step is no reading (a file of such a run used to open with "more data
    still helps")."""
    results = tmp_path / "results"
    argv = [a if a != str(finished.results) else str(results) for a in finished.argv]
    argv = [a for a in argv if a != "--with-tables"]
    code, lines = run_main(*argv, "--train-limit", "40")
    assert code == 0, lines[-3:]
    out = results / "t_curve"
    assert lines[-1] == (
        f"{LC.DONE} {out.resolve() / LC.CURVE_JSON} ({LC.NO_READING}: see the sentence)"
    )
    assert lines[-2].startswith(
        f"{LC.NO_READING}: the fits were limited to 40 training examples. "
    )
    curve = json.loads((out / LC.CURVE_JSON).read_text())
    assert curve["status"] == LC.DONE and curve["sentence"] == lines[-2]
    assert curve["reading"] == {
        "valid": False,
        "problems": ["the fits were limited to 40 training examples"],
    }
    assert [row["train_examples_used"] for row in curve["points"]] == [40, 40]
    assert not any(word in curve["sentence"] for word in JUDGEMENTS)
    text = (out / LC.CURVE_MD).read_text()
    assert f"**{LC.NO_READING} OF THE LEARNING CURVE: the fits were limited" in text
    assert f"**{curve['sentence']}**" in text
    # The same builds and fits under the settings of a real run are refused,
    # not passed off as its points.
    code, lines = run_main(*argv, "--overwrite")
    assert code == 1 and "was trained with other settings" in lines[-1]
    assert "limit 40, not None" in lines[-1]


def test_a_foreign_build_or_fit_under_the_runs_name_is_refused(
    finished: Run, tmp_path: Path
):
    results = tmp_path / "results"
    argv = [a if a != str(finished.results) else str(results) for a in finished.argv]
    foreign = results / "t_f50"
    foreign.mkdir(parents=True)
    theirs = {"tag": "t_f50", "limit": 7, "sources_requested": ["feed"], "shards": []}
    (foreign / "manifest.json").write_text(json.dumps(theirs))
    for name in LC.DATASET_FILES:
        (foreign / name).write_text("theirs")
    before = tree(results)
    code, lines = run_main(*argv)
    assert code == 1 and "holds another build" in lines[-1]
    assert "not a subsample" in lines[-1] and "limit 7" in lines[-1]
    assert tree(results) == before
    assert sorted(p.name for p in results.iterdir()) == ["t_f50"]  # no t_curve left
    # A build that lost a shard is not rebuilt over either.
    theirs["shards"] = [{"file": "shard-00000.npz", "examples": 3}]
    (foreign / "manifest.json").write_text(json.dumps(theirs))
    code, lines = run_main(*argv)
    assert code == 1 and "not all of its files" in lines[-1]
    # An artifact under the run's name that is not the fit the run would make:
    # trained on another build, without a finished report, or with other
    # settings. It is never replaced and never scored.
    other = tmp_path / "other"
    argv = [a if a != str(finished.results) else str(other) for a in finished.argv]
    fit = other / "t_f50_oppnet"
    fit.mkdir(parents=True)
    (fit / "artifact.pt").write_bytes(b"someone else's weights")
    done = json.loads((finished.results / "t_f50_oppnet/train_report.json").read_text())

    def refused(report: dict[str, Any] | None, needle: str) -> None:
        if report is None:
            (fit / "train_report.json").unlink()
        else:
            (fit / "train_report.json").write_text(json.dumps(report))
        code, lines = run_main(*argv)
        assert code == 1 and needle in lines[-1], (needle, lines[-1])
        assert "it is not replaced" in lines[-1]
        assert (fit / "artifact.pt").read_bytes() == b"someone else's weights"
        assert not any(line.startswith("TRAIN_START") for line in lines)

    # The report of the finished run names that run's build, not this one's.
    refused(done, "trained on another build")
    here = {
        **done,
        "dataset": {
            "manifest_sha256": LC.sha256_file(other / "t_f50" / "manifest.json")
        },
    }
    refused({**here, "status": "TRAIN_RUNNING"}, "no finished train_report.json")
    for key, value, needle in (
        ("limit", 3000, "limit 3000, not None"),
        ("epochs", 7, "epochs 7, not 1"),
        ("d_model", 128, "d_model 128, not 32"),
        ("elo_mode", "keep", "elo_mode 'keep', not 'blank'"),
    ):
        refused({**here, "args": {**done["args"], key: value}}, needle)
    refused(None, "no finished train_report.json")


def test_a_table_fit_of_something_else_is_refused(finished: Run, tmp_path: Path):
    results = tmp_path / "results"
    argv = [a if a != str(finished.results) else str(results) for a in finished.argv]
    tables = results / "t_f50_tables"
    tables.mkdir(parents=True)
    (tables / "flags_table.pt").write_bytes(b"another fit")
    done = json.loads((finished.results / "t_f50_tables/fit_report.json").read_text())
    # The finished run's report names that run's build of the dataset.
    (tables / "fit_report.json").write_text(json.dumps(done))
    code, lines = run_main(*argv)
    assert code == 1
    assert "holds another table fit (another build of the dataset)" in lines[-1]
    assert (tables / "flags_table.pt").read_bytes() == b"another fit"
    sha = LC.sha256_file(results / "t_f50" / "manifest.json")
    here = {**done["dataset"], "manifest_sha256": sha}
    stale = {**done, "dataset": {**here, "limit_per_split": 3000}}
    (tables / "fit_report.json").write_text(json.dumps(stale))
    code, lines = run_main(*argv)
    assert code == 1 and "holds another table fit (limit 3000)" in lines[-1]
    assert (tables / "flags_table.pt").read_bytes() == b"another fit"
    # Table files without a report are a fit that was stopped: done again.
    (tables / "fit_report.json").unlink()
    code, lines = run_main(*argv)
    assert code == 0, lines[-3:]
    assert (tables / "flags_table.pt").read_bytes() != b"another fit"
    assert (tables / "fit_report.json").is_file()


def test_a_label_mismatch_between_two_builds_fails_the_run(
    finished: Run, tmp_path: Path
):
    """Reported, not ignored: the curve is written, says CURVE_FAILED, names the
    field and the game, and gives no paired number for that step."""
    results = tmp_path / "results"
    results.mkdir()
    for directory in finished.results.iterdir():
        if directory.name == "t_curve":
            continue
        (results / directory.name).mkdir()
        for path in directory.iterdir():
            (results / directory.name / path.name).write_bytes(path.read_bytes())
    # One reason code of one ladder-holdout slot changes in the smaller build.
    shard = results / "t_f50" / "shard-00000.npz"
    data = F.load_batch(shard)
    ladder = np.flatnonzero(data["m_split"] == TD.B.SPLIT_LADDER)
    row = int(ladder[0])
    data["y_reason"] = data["y_reason"].copy()
    data["y_reason"][row, 0] = (int(data["y_reason"][row, 0]) + 1) % len(F.REASON_NAMES)
    F.save_batch(shard, data)
    game = LC.battle_ids(results / "t_f50")[int(data["m_battle"][row])]
    argv = [a if a != str(finished.results) else str(results) for a in finished.argv]
    code, lines = run_main(*argv)
    assert code == 1
    assert lines[-1].startswith(f"{LC.FAILED} label mismatch, t_f50 -> t_f100")
    assert "y_reason" in lines[-1]
    curve = json.loads((results / "t_curve" / LC.CURVE_JSON).read_text())
    assert curve["status"] == LC.FAILED and "label mismatch" in curve["reason"]
    found = curve["steps"][0]["sets"][LC.SET_LADDER]
    assert found["error"] == "label_mismatch" and found["difference"] is None
    assert found["labels"]["mismatches"] == {"y_reason": 1}
    (example,) = found["labels"]["examples"]
    assert example["battle"] == game and example["slot"] == 0
    assert example["turn"] == int(data["m_turn"][row])
    # The points themselves are still reported; the step and the sentence are not.
    assert len(curve["points"]) == 2
    assert "no reading" in curve["sentence"]
    text = (results / "t_curve" / LC.CURVE_MD).read_text()
    assert f"**{LC.FAILED}**" in text and "(label_mismatch)" in text
