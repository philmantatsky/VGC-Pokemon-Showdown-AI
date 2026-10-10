"""Set prior: the table of full movesets and its posterior over a set's moves.

Every input is a hand-made record list written inline; ids are made-up strings
(the module compares ids, it knows no species and no move). One test at the end
checks that the module imports neither torch nor poke-env.
"""

from __future__ import annotations

import ast
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vgc_bench.src.oppmodel import setprior as S

HAS, MEMBER, PRIOR, LEFT = (
    S.COL_HAS_DATA,
    S.COL_P_MEMBER,
    S.COL_P_PRIOR,
    S.COL_SLOTS_LEFT,
)

# One species with two archetypes that share one move:
#   "fast" (60 records): m_shared, m_fast1, m_fast2, m_fast3     item i_fast
#   "bulk" (40 records): m_shared, m_bulk1, m_bulk2, m_bulk3     item i_bulk
FAST = ("m_shared", "m_fast1", "m_fast2", "m_fast3")
BULK = ("m_shared", "m_bulk1", "m_bulk2", "m_bulk3")
ALL_MOVES = ("m_shared", "m_fast1", "m_fast2", "m_fast3", "m_bulk1", "m_bulk2")


def two_archetypes() -> list[S.SetRecord]:
    records: list[S.SetRecord] = []
    records += [("sp_two", FAST, "i_fast", "a_one", 1.0)] * 60
    records += [("sp_two", BULK, "i_bulk", "a_two", 1.0)] * 40
    # A second species whose every set holds m_always.
    for index in range(30):
        records.append(
            ("sp_always", ("m_always", f"m_x{index % 5}"), "i_any", "a_any", 1.0)
        )
    return records


def table(**options: Any) -> S.SetTable:
    return S.SetTable.build(two_archetypes(), **options)


# --- contract ---------------------------------------------------------------------


def test_the_columns_are_the_contract() -> None:
    assert S.SET_COLS == 4
    assert S.COLUMN_NAMES == ("has_data", "p_member", "p_prior", "slots_left")
    out = table().posterior("sp_two", (), ALL_MOVES)
    assert out.shape == (len(ALL_MOVES), S.SET_COLS)
    assert out.dtype == np.float32


def test_a_shown_move_is_a_member_with_probability_exactly_one() -> None:
    out = table().posterior("sp_two", ("m_fast1",), ALL_MOVES)
    assert out[ALL_MOVES.index("m_fast1"), MEMBER] == 1.0
    # ... and keeps its unconditional rate in the prior column.
    assert out[ALL_MOVES.index("m_fast1"), PRIOR] == pytest.approx(0.6)


def test_every_column_is_a_probability() -> None:
    t = table()
    for species in ("sp_two", "sp_always", "sp_unknown"):
        for shown in ((), ("m_shared",), ("m_fast1", "m_fast2"), FAST, ("m_nowhere",)):
            out = t.posterior(species, shown, ALL_MOVES + ("m_nowhere", ""))
            assert np.isfinite(out).all()
            assert (out >= 0.0).all() and (out <= 1.0).all()


def test_an_unknown_species_gives_zeros() -> None:
    out = table().posterior("sp_unknown", ("m_shared",), ALL_MOVES)
    assert out.shape == (len(ALL_MOVES), S.SET_COLS)
    assert not out.any()


def test_a_shown_move_no_recorded_set_holds_gives_zeros() -> None:
    out = table().posterior("sp_two", ("m_nowhere",), ALL_MOVES)
    assert not out.any()


def test_without_shown_moves_the_posterior_is_the_prior() -> None:
    out = table(smoothing=0.0).posterior("sp_two", (), ALL_MOVES)
    assert (out[:, HAS] == 1.0).all()
    np.testing.assert_allclose(out[:, MEMBER], out[:, PRIOR], atol=1e-6)
    np.testing.assert_allclose(out[:, PRIOR], [1.0, 0.6, 0.6, 0.6, 0.4, 0.4], atol=1e-6)
    assert (out[:, LEFT] == 1.0).all()


def test_slots_left_counts_the_shown_moves() -> None:
    t = table()
    for shown, left in (((), 1.0), (FAST[:1], 0.75), (FAST[:2], 0.5), (FAST, 0.0)):
        out = t.posterior("sp_two", shown, ALL_MOVES)
        assert (out[:, LEFT] == left).all()
    # A move given twice is one shown move.
    out = t.posterior("sp_two", ("m_fast1", "m_fast1"), ALL_MOVES)
    assert (out[:, LEFT] == 0.75).all()


def test_a_move_every_set_holds_is_near_one() -> None:
    out = table().posterior("sp_always", ("m_x0",), ("m_always", "m_x1"))
    assert out[0, HAS] == 1.0
    assert out[0, MEMBER] > 0.99
    assert out[0, PRIOR] == pytest.approx(1.0)
    # Sets here hold two moves: m_x1 never comes with m_x0.
    assert out[1, MEMBER] < 0.1


def test_conditioning_tells_two_archetypes_apart() -> None:
    t = table()
    fast = t.posterior("sp_two", ("m_fast1",), ALL_MOVES)
    bulk = t.posterior("sp_two", ("m_bulk1",), ALL_MOVES)
    ix = {move: row for row, move in enumerate(ALL_MOVES)}
    # After one fast move the other fast moves are near certain and the bulk
    # ones near excluded; the prior alone said 0.6 / 0.4.
    assert fast[ix["m_fast2"], MEMBER] > 0.95 > fast[ix["m_fast2"], PRIOR]
    assert fast[ix["m_bulk1"], MEMBER] < 0.05 < fast[ix["m_bulk1"], PRIOR]
    assert bulk[ix["m_bulk2"], MEMBER] > 0.95
    assert bulk[ix["m_fast2"], MEMBER] < 0.05
    # The shared move says nothing: the posterior stays at the prior.
    shared = t.posterior("sp_two", ("m_shared",), ALL_MOVES)
    np.testing.assert_allclose(shared[1:, MEMBER], shared[1:, PRIOR], atol=1e-6)


def test_smoothing_pulls_a_thin_condition_toward_the_prior() -> None:
    records: list[S.SetRecord] = [("sp", ("m_a", "m_b"), "", "", 1.0)] * 5
    records += [("sp", ("m_c", "m_d"), "", "", 1.0)] * 95
    sharp = S.SetTable.build(records, smoothing=0.0)
    soft = S.SetTable.build(records, smoothing=5.0)
    # Given m_a: 5 consistent sets, all with m_b, none with m_c.
    assert sharp.posterior("sp", ("m_a",), ("m_b", "m_c"))[:, MEMBER].tolist() == [
        1.0,
        0.0,
    ]
    out = soft.posterior("sp", ("m_a",), ("m_b", "m_c"))
    assert out[0, MEMBER] == pytest.approx((5 + 5 * 0.05) / 10)
    assert out[1, MEMBER] == pytest.approx((0 + 5 * 0.95) / 10)


def test_a_known_item_narrows_the_sets() -> None:
    t = table()
    ix = {move: row for row, move in enumerate(ALL_MOVES)}
    blind = t.posterior("sp_two", (), ALL_MOVES)
    told = t.posterior("sp_two", (), ALL_MOVES, item="i_bulk", item_known=True)
    assert blind[ix["m_bulk1"], MEMBER] == pytest.approx(0.4, abs=1e-6)
    assert told[ix["m_bulk1"], MEMBER] > 0.95
    assert told[ix["m_fast1"], MEMBER] < 0.05
    # The item is ignored unless the caller says it is known.
    quiet = t.posterior("sp_two", (), ALL_MOVES, item="i_bulk")
    np.testing.assert_array_equal(quiet, blind)
    # The same with the ability.
    by_ability = t.posterior(
        "sp_two", (), ALL_MOVES, ability="a_one", ability_known=True
    )
    assert by_ability[ix["m_fast1"], MEMBER] > 0.95


def test_an_item_or_ability_no_set_has_is_dropped_unless_strict() -> None:
    t = table()
    blind = t.posterior("sp_two", ("m_fast1",), ALL_MOVES)
    odd = t.posterior(
        "sp_two",
        ("m_fast1",),
        ALL_MOVES,
        item="i_fast",
        item_known=True,
        ability="a_of_a_mega",
        ability_known=True,
    )
    # The ability goes first; the item still agrees with the fast sets.
    np.testing.assert_allclose(odd, blind, atol=1e-6)
    assert t.counters["posterior_backed_off"] >= 1
    strict = t.posterior(
        "sp_two",
        ("m_fast1",),
        ALL_MOVES,
        ability="a_of_a_mega",
        ability_known=True,
        strict=True,
    )
    assert not strict.any()
    # An item that contradicts the shown move: no set has both, so back off to
    # the moves alone rather than answer from nothing.
    clash = t.posterior(
        "sp_two", ("m_fast1",), ALL_MOVES, item="i_bulk", item_known=True
    )
    np.testing.assert_allclose(clash, blind, atol=1e-6)


def test_min_sets_decides_has_data() -> None:
    records: list[S.SetRecord] = [("sp", ("m_a", "m_b"), "", "", 1.0)] * 3
    records += [("sp", ("m_c", "m_d"), "", "", 1.0)] * 20
    t = S.SetTable.build(records, min_sets=5)
    assert not t.posterior("sp", ("m_a",), ("m_b",)).any()  # 3 consistent sets
    assert t.posterior("sp", ("m_c",), ("m_d",))[0, HAS] == 1.0
    # A species with fewer records than min_sets is not stored at all.
    thin = S.SetTable.build([("sp", ("m_a",), "", "", 1.0)] * 4, min_sets=5)
    assert len(thin) == 0 and "sp" not in thin
    assert thin.build_counts["species_below_min_sets"] == 1


def test_weights_are_used_and_counts_gate() -> None:
    records: list[S.SetRecord] = [("sp", ("m_a", "m_b"), "", "", 0.25)] * 8
    records += [("sp", ("m_a", "m_c"), "", "", 1.0)] * 8
    t = S.SetTable.build(records, smoothing=0.0)
    out = t.posterior("sp", ("m_a",), ("m_b", "m_c"))
    assert out[0, MEMBER] == pytest.approx(2.0 / 10.0)
    assert out[1, MEMBER] == pytest.approx(8.0 / 10.0)
    assert t.n_records("sp") == 16 and t.n_distinct_sets("sp") == 2


def test_max_sets_keeps_the_heaviest_and_the_prior_counts_everything() -> None:
    records: list[S.SetRecord] = [("sp", ("m_a", "m_b"), "", "", 1.0)] * 10
    records += [("sp", ("m_a", "m_c"), "", "", 1.0)] * 6
    records += [("sp", ("m_a", "m_d"), "", "", 1.0)] * 4
    t = S.SetTable.build(records, max_sets_per_species=2, smoothing=0.0)
    assert t.n_distinct_sets("sp") == 2 and t.n_records("sp") == 20
    assert [held for held, *_ in t.sets("sp")] == [("m_a", "m_b"), ("m_a", "m_c")]
    out = t.posterior("sp", (), ("m_b", "m_c", "m_d"))
    np.testing.assert_allclose(out[:, PRIOR], [0.5, 0.3, 0.2], atol=1e-6)
    # The dropped set is out of the conditional only.
    np.testing.assert_allclose(out[:, MEMBER], [10 / 16, 6 / 16, 0.0], atol=1e-6)
    assert t.build_counts["distinct_sets_dropped"] == 1


# --- storage -----------------------------------------------------------------------


def test_payload_round_trip_is_exact_and_deterministic() -> None:
    first = table()
    payload = first.to_payload()
    text = json.dumps(payload, sort_keys=True)
    second = S.SetTable.from_payload(json.loads(text))
    assert second.to_payload() == payload
    assert json.dumps(second.to_payload(), sort_keys=True) == text
    assert second.signature() == first.signature()
    for shown in ((), ("m_fast1",), ("m_bulk1", "m_bulk2")):
        np.testing.assert_array_equal(
            first.posterior("sp_two", shown, ALL_MOVES),
            second.posterior("sp_two", shown, ALL_MOVES),
        )


def test_the_table_does_not_depend_on_the_order_of_the_records() -> None:
    records = two_archetypes()
    shuffled = [records[i] for i in np.random.default_rng(0).permutation(len(records))]
    one, two = S.SetTable.build(records), S.SetTable.build(shuffled)
    assert one.to_payload() == two.to_payload()
    assert one.signature() == two.signature()
    assert len(one.signature()) == S.SIGNATURE_LENGTH
    # Another table, another signature.
    assert S.SetTable.build(records[:-1]).signature() != one.signature()


def test_a_foreign_payload_is_refused() -> None:
    payload = table().to_payload()
    for broken in (
        {**payload, "format": "something-else"},
        {**payload, "version": 99},
        {**payload, "columns": ["has_data"]},
        {**payload, "species": [{"id": "sp"}]},
        {},
    ):
        with pytest.raises(ValueError):
            S.SetTable.from_payload(broken)


def test_an_empty_table_answers_zeros() -> None:
    empty = S.SetTable.build([])
    assert len(empty) == 0
    assert not empty.posterior("sp_two", ("m_a",), ("m_a", "m_b")).any()
    again = S.SetTable.from_payload(empty.to_payload())
    assert again.to_payload() == empty.to_payload()


# --- never raises ------------------------------------------------------------------


def test_bad_records_are_skipped_and_counted() -> None:
    records: list[Any] = [
        ("sp", ("m_a", "m_b"), "", "", 1.0),
        ("", ("m_a",), "", "", 1.0),  # no species
        ("sp", (), "", "", 1.0),  # no move
        ("sp", ("a", "b", "c", "d", "e"), "", "", 1.0),  # five moves
        ("sp", ("m_a",), "", "", 0.0),  # no weight
        ("sp", ("m_a",), "", "", float("nan")),
        ("sp", ("m_a",), "", "", "heavy"),  # weight is not a number
        ("sp", ("m_a",)),  # too short
        None,
        ("sp", ("m_a", "m_a", "m_b", ""), None, None, 2),  # duplicates, None ids
    ]
    t = S.SetTable.build(records, min_sets=1)
    counts = t.build_counts
    assert counts["records"] == 2
    assert counts["record_without_species_or_move"] == 2
    assert counts["record_with_too_many_moves"] == 1
    assert counts["record_without_weight"] == 2
    assert counts["record_malformed"] == 3
    assert t.sets("sp") == [(("m_a", "m_b"), "", "", 3.0, 2)]


def test_garbage_input_never_raises() -> None:
    t = table()
    garbage: list[Any] = [None, 0, 3.5, "", "sp_two", [], {}, object()]
    garbage.append(["m_fast1", None])
    for species in garbage:
        for shown in garbage:
            for candidates in garbage:
                out = t.posterior(
                    species,
                    shown,
                    candidates,
                    item=object(),
                    ability=[],
                    item_known=True,
                    ability_known=True,
                )
                assert isinstance(out, np.ndarray)
                assert out.ndim == 2 and out.shape[1] == S.SET_COLS
                assert np.isfinite(out).all()
    # Something that cannot be counted as candidates gives an empty answer.
    assert t.posterior("sp_two", (), None).shape == (0, S.SET_COLS)
    assert t.posterior("sp_two", (), 7).shape == (0, S.SET_COLS)
    assert t.counters["posterior_error:candidates"] >= 2
    # A lone string is one shown move, not its letters.
    one = t.posterior("sp_two", "m_fast1", ALL_MOVES)
    np.testing.assert_array_equal(one, t.posterior("sp_two", ("m_fast1",), ALL_MOVES))


def test_integer_ids_work_when_asked_with_the_same_integers() -> None:
    records: list[S.SetRecord] = [(7, (1, 2, 3, 4), 11, 21, 1.0)] * 6
    records += [(7, (1, 5, 6, 8), 12, 22, 1.0)] * 6
    t = S.SetTable.build(records)
    out = t.posterior(7, (2,), (1, 3, 5), item=11, item_known=True)
    assert out[:, HAS].tolist() == [1.0, 1.0, 1.0]
    # (6 + 2 * 0.5) / (6 + 2) against (0 + 2 * 0.5) / (6 + 2).
    assert out[1, MEMBER] == pytest.approx(0.875)
    assert out[2, MEMBER] == pytest.approx(0.125)
    assert S.SetTable.from_payload(t.to_payload()).signature() == t.signature()


def test_an_empty_candidate_row_is_zeros() -> None:
    out = table().posterior("sp_two", (), ("m_shared", "", "m_fast1"))
    assert out[0, HAS] == 1.0 and out[2, HAS] == 1.0
    assert not out[1].any()


# --- cost --------------------------------------------------------------------------


def test_a_call_is_far_under_the_budget() -> None:
    rng = np.random.default_rng(1)
    moves = [f"m{i}" for i in range(60)]
    records: list[S.SetRecord] = []
    for index in range(4000):
        held = tuple(str(m) for m in rng.choice(moves, size=4, replace=False))
        records.append(("sp", held, f"i{index % 9}", f"a{index % 3}", 1.0))
    t = S.SetTable.build(records)
    candidates = moves[:12]
    shown = [tuple(moves[i : i + k]) for k in range(4) for i in range(40)]
    started = time.perf_counter()
    for held in shown:  # every call is a cache miss
        t.posterior("sp", held, candidates, item="i1", item_known=True)
    cold = (time.perf_counter() - started) / len(shown)
    started = time.perf_counter()
    for _ in range(5):
        for held in shown:
            t.posterior("sp", held, candidates, item="i1", item_known=True)
    warm = (time.perf_counter() - started) / (5 * len(shown))
    # Budget 0.3 ms per call; the bounds leave room for a loaded machine.
    assert cold < 3e-3, cold
    assert warm < 1e-3, warm


def test_the_cache_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(S, "_CACHE_LIMIT", 4)
    t = table()
    for index in range(20):
        t.posterior("sp_two", (), ALL_MOVES, item=f"i{index}", item_known=True)
    assert len(t._cache) <= 4


# --- the module stays light -----------------------------------------------------------


def test_the_module_imports_no_torch_no_poke_env_and_holds_no_assert() -> None:
    source = Path(S.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        assert not isinstance(node, ast.Assert), "an assert in the runtime path"
    assert imported <= {
        "hashlib",
        "json",
        "math",
        "collections",
        "dataclasses",
        "typing",
        "numpy",
    }
