from __future__ import annotations

import random

import pytest

from vgc_bench.src.set_particles import (
    ParticleDatabase,
    RevealEvidence,
    SpeciesBelief,
    TeamBelief,
    determination_team_text,
    team_roster,
)


def _database(max_particles=12):
    joint = {
        "alpha": {
            "sets": [
                {
                    "ability": "fast",
                    "item": "alphite",
                    "moves": ["one", "two", "three", "protect"],
                    "prob": 0.55,
                    "count": 55,
                },
                {
                    "ability": "slow",
                    "item": "sitrusberry",
                    "moves": ["one", "four", "five", "protect"],
                    "prob": 0.35,
                    "count": 35,
                },
                {
                    "ability": "slow",
                    "item": "leftovers",
                    "moves": ["six", "seven", "eight", "protect"],
                    "prob": 0.10,
                    "count": 10,
                },
            ]
        },
        "beta": {
            "sets": [
                {
                    "ability": "plain",
                    "item": "betite",
                    "moves": ["a", "b", "c", "protect"],
                    "prob": 0.7,
                    "count": 7,
                },
                {
                    "ability": "plain",
                    "item": "sitrusberry",
                    "moves": ["a", "d", "e", "protect"],
                    "prob": 0.3,
                    "count": 3,
                },
            ]
        },
    }
    marginal = {
        "alpha": {
            "ability": "fast",
            "item": "alphite",
            "moves": ["one", "two", "three", "protect"],
            "spread": "Jolly:0/32/0/0/0/32",
        },
        "beta": {
            "ability": "plain",
            "item": "betite",
            "moves": ["a", "b", "c", "protect"],
            "spread": "Adamant:0/32/0/0/0/32",
        },
    }
    return ParticleDatabase(joint, marginal, max_particles=max_particles)


def test_database_caps_and_normalizes_particles():
    particles = _database(max_particles=2).particles("Alpha")
    assert len(particles) == 2
    assert sum(p.probability for p in particles) == pytest.approx(1.0)
    assert all(p.spread == "Jolly:0/32/0/0/0/32" for p in particles)
    assert _database().top_coverage("alpha", 2) == pytest.approx(0.9)


def test_weighted_medoids_cover_related_tail_sets():
    joint = {
        "alpha": {
            "sets": [
                {
                    "ability": "a",
                    "item": "i",
                    "moves": ["core1", "core2", "core3", f"option{index}"],
                    "prob": 0.2,
                }
                for index in range(5)
            ]
        }
    }
    database = ParticleDatabase(joint, {}, max_particles=1)
    particles = database.particles("alpha")
    assert len(particles) == 1
    assert particles[0].probability == pytest.approx(1.0)
    assert database.top_coverage("alpha", 1) == pytest.approx(1.0)


def test_reveals_eliminate_impossible_particles():
    belief = SpeciesBelief.from_database(_database(), "alpha")
    belief.condition(RevealEvidence.build(moves=["four"], item="Sitrus Berry"))
    posterior = belief.posterior()
    assert len(posterior) == 1
    assert posterior[0].ability == "slow"
    assert posterior[0].item == "sitrusberry"


def test_novel_reveal_does_not_resurrect_impossible_sets():
    belief = SpeciesBelief.from_database(_database(), "alpha")
    belief.condition(RevealEvidence.build(moves=["never-seen"], item="choiceband"))
    posterior = belief.posterior()
    assert len(posterior) == 1
    assert posterior[0].novel
    assert "neverseen" in posterior[0].moves
    assert posterior[0].item == "choiceband"


def test_numeric_evidence_updates_particle_weights():
    belief = SpeciesBelief.from_database(_database(), "alpha")
    fast, slow, other = belief.particles
    predicted = {
        fast.signature: (0.70, 0.80),
        slow.signature: (0.25, 0.35),
        other.signature: (0.10, 0.20),
    }
    belief.update_numeric_interval(0.30, predicted)
    posterior = belief.posterior()
    assert max(posterior, key=lambda p: p.probability).signature == slow.signature


def test_team_samples_obey_item_clause_without_erasing_mega_options():
    team = TeamBelief.from_roster(_database(), ["alpha", "beta"])
    samples = team.sample_determinizations(8, random.Random(7))
    assert 1 <= len(samples) <= 8
    for sample in samples:
        items = [p.item for p in sample.values() if p.item]
        assert len(items) == len(set(items))
    assert any(sum(p.mega for p in sample.values()) == 2 for sample in samples)


def test_open_sheet_uses_one_determinization():
    team = TeamBelief.from_roster(_database(), ["alpha", "beta"])
    assert len(team.sample_determinizations(8, random.Random(3), open_sheet=True)) == 1


def test_roster_parser_drops_private_set_information():
    roster = team_roster(
        "Nickname (Alpha-Mega) (F) @ Secret Item\nAbility: Secret\n- Move\n\n"
        "Beta (M)\n- B"
    )
    assert [(slot.species, slot.display) for slot in roster] == [
        ("alphamega", "Alpha-Mega"),
        ("beta", "Beta"),
    ]


def test_determination_renders_showdown_team():
    database = _database()
    roster = team_roster("Alpha\n- one\n\nBeta\n- a")
    sampled = TeamBelief.from_roster(database, ["alpha", "beta"])
    determination = sampled.sample_determinizations(1, random.Random(11))[0]
    text = determination_team_text(roster, determination)
    assert "Secret" not in text
    assert "Level: 50" in text
    assert "Nature" in text
    assert text.count("\n\n") == 1


def test_placeholder_moves_are_stripped_from_particles():
    """Showdown's "[nothing]" placeholder crashed every Ditto determinization.

    The scraped joint table recorded pre-Transform Ditto as having the move
    "nothing" with all its probability mass; rendering that set crashed
    poke-env's battle view and aborted counterfactual generation (2026-08-25).
    """
    database = ParticleDatabase(
        joint_sets={
            "ditto": {
                "sets": [
                    {
                        "ability": "imposter",
                        "item": "choicescarf",
                        "moves": ["nothing", "transform"],
                        "prob": 1.0,
                        "count": 40,
                    }
                ]
            }
        },
        marginals={
            "ditto": {"moves": ["struggle", "transform"], "item": "choicescarf"}
        },
    )
    particles = database.particles("ditto")
    assert particles
    for particle in particles:
        assert particle.moves == ("transform",)
