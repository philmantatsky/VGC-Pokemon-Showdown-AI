import json

import pytest

from evaluation.opening_study import append_cell, resolve_plan, roster_split


def test_opening_identity_mapping_and_distinct_four():
    roster = [
        "blastoise",
        "farigiraf",
        "charizard",
        "venusaur",
        "torkoal",
        "incineroar",
    ]
    assert (
        resolve_plan(roster, ["charizard", "venusaur", "blastoise", "farigiraf"])
        == "/team 3412"
    )
    with pytest.raises(ValueError):
        resolve_plan(roster, ["blastoise"] * 4)


def test_alternate_sheets_cannot_cross_roster_split():
    one = "Rillaboom @ Assault Vest\nAbility: Grassy Surge\n- Fake Out"
    two = one.replace("Assault Vest", "Miracle Seed").replace("Fake Out", "Wood Hammer")
    assert roster_split(one) == roster_split(two)


def test_cell_append_is_complete(tmp_path):
    path = tmp_path / "study.jsonl"
    append_cell(path, [{"target": 1}, {"target": 0}])
    append_cell(path, [{"target": 1}])
    assert [json.loads(s)["target"] for s in path.read_text().splitlines()] == [1, 0, 1]
    assert not path.with_suffix(".jsonl.tmp").exists()
