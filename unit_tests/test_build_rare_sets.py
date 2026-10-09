"""datagen/build_rare_sets.py: a set for each species the joint set data lacks, from
the opponent predictor's repertoire, so that the search can build a world when one
is on the other side (2026-10-09: "Set Arbok has no moves" turned the search off for
whole ladder games, 11% of them by the recorded rosters)."""

from __future__ import annotations

import json
import random

import pytest

from datagen.build_rare_sets import first_abilities, rare_sets
from vgc_bench.src.set_particles import (
    ParticleDatabase,
    TeamBelief,
    determination_team_text,
    team_roster,
)

COUNTS = {
    "inteleon": {"snipeshot": 46.0, "protect": 18.0, "focusenergy": 11.0}
    | {"icebeam": 9.0, "soak": 3.0, "muddywater": 3.0},
    "blastoise": {"waterspout": 500.0, "protect": 100.0},  # has joint sets already
    "stunfisk": {"earthpower": 3.0, "painsplit": 2.0},
    "furfrou": {"cottonguard": 2.0, "snarl": 1.0},  # seen three times: too thin
    "unown": {"hiddenpower": 40.0},  # one move is not a set
    "tied": {"b": 5.0, "a": 5.0, "c": 5.0, "": 9.0, "gone": 0.0},
}


def test_one_set_of_the_four_most_used_moves_for_each_species_the_data_lacks():
    sets = rare_sets(COUNTS, {"blastoise"}, {"inteleon": "torrent"}, min_support=5.0)
    assert sorted(sets) == ["inteleon", "stunfisk", "tied"]
    assert sets["inteleon"] == {
        "sets": [
            {
                "ability": "torrent",
                "count": 90,
                "item": None,
                "moves": ["snipeshot", "protect", "focusenergy", "icebeam"],
                "prob": 1.0,
            }
        ]
    }
    # fewer than four moves seen: the set is what was seen, nothing is invented
    assert sets["stunfisk"]["sets"][0]["moves"] == ["earthpower", "painsplit"]
    assert sets["stunfisk"]["sets"][0]["ability"] is None
    # equal use: by name, so the file does not depend on dictionary order
    assert sets["tied"]["sets"][0]["moves"] == ["a", "b", "c"]
    assert rare_sets(COUNTS, set(), {}, min_support=1000.0) == {}


def test_the_ability_is_the_species_first_listed_one():
    found = first_abilities(["mimikyu", "inteleon", "notapokemon"])
    assert found == {"mimikyu": "disguise", "inteleon": "torrent"}


def test_the_search_can_draw_a_world_with_such_a_species(tmp_path):
    """Before: no particle for the species, so no team the simulator accepts."""
    joint = {
        "blastoise": {
            "sets": [
                {
                    "ability": "torrent",
                    "count": 3,
                    "item": "blastoisinite",
                    "moves": ["waterspout", "protect", "icebeam", "fakeout"],
                    "prob": 1.0,
                }
            ]
        }
    }
    (tmp_path / "joint_sets_regmc.json").write_text(json.dumps(joint))
    roster = team_roster("Blastoise @ Blastoisinite\n\nInteleon\n")
    names = [slot.species for slot in roster]
    rng = random.Random(7)
    bare = ParticleDatabase.load(tmp_path, formatid="gen9championsvgc2026regmc")
    assert bare.particles("inteleon") == ()
    with pytest.raises(ValueError, match="could not sample"):
        TeamBelief.from_roster(bare, names).sample_determinizations(8, rng)
    ParticleDatabase.particles.cache_clear()
    rare = rare_sets(COUNTS, set(joint), first_abilities(sorted(COUNTS)))
    (tmp_path / "rare_sets_regmc.json").write_text(json.dumps(rare))
    database = ParticleDatabase.load(tmp_path, formatid="gen9championsvgc2026regmc")
    (particle,) = database.particles("inteleon")
    assert particle.moves == ("snipeshot", "protect", "focusenergy", "icebeam")
    assert (particle.ability, particle.item) == ("torrent", None)
    # the joint data's own species are untouched
    assert database.particles("blastoise")[0].item == "blastoisinite"
    worlds = TeamBelief.from_roster(database, names).sample_determinizations(8, rng)
    text = determination_team_text(roster, worlds[0])
    assert "Inteleon\nAbility: torrent" in text and "- snipeshot" in text
    # a move it then shows that the set lacks: the shown move leads a novel set
    belief = TeamBelief.from_roster(database, names)
    belief.condition("inteleon", moves=["uturn"])
    (novel,) = belief.beliefs["inteleon"].posterior()
    assert novel.moves == ("uturn", "snipeshot", "protect", "focusenergy")
    ParticleDatabase.particles.cache_clear()
