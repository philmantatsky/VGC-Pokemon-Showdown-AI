"""training/playbook_preview.py: during training the playbook's own team previews
by its plan cards (sometimes another practised card), every other team keeps the
wrapped human-model preview (the user, 2026-09-27: "start the practice cycle")."""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from poke_env.battle import Pokemon

from training import playbook_preview as P
from vgc_bench.src.playbook import Playbook

FORMAT = "gen9championsvgc2026regmc"
OURS = ["blastoise", "farigiraf", "charizard", "venusaur", "torkoal", "incineroar"]
DEFAULT = [
    "pelipper",
    "archaludon",
    "basculegion",
    "whimsicott",
    "gholdengo",
    "maushold",
]
SAND = ["tyranitar", "excadrill", "indeedeef", "corviknight", "gholdengo", "sinistcha"]


def _battle(ours: list[str], theirs: list[str]) -> Any:
    return SimpleNamespace(
        format=FORMAT,
        team={f"p1: {s}": Pokemon(gen=9, species=s) for s in ours},
        opponent_team={f"p2: {s}": Pokemon(gen=9, species=s) for s in theirs},
    )


@pytest.fixture(scope="module")
def playbook():
    return Playbook("data/playbook_t6.json")


def _picks(order: str) -> list[str]:
    return [OURS[int(i) - 1] for i in order.removeprefix("/team ")]


def test_our_team_plays_the_card_the_rules_pick(playbook):
    battle = _battle(OURS, DEFAULT)
    picked = P.playbook_order(playbook, battle, 0.0, random.Random(1))
    assert picked is not None
    card, order = picked
    assert card == "water_room"
    assert _picks(order)[:2] == ["blastoise", "farigiraf"]
    brought = [
        s for s, m in zip(OURS, battle.team.values()) if m._selected_in_teampreview
    ]
    assert sorted(brought) == sorted(_picks(order))


def test_exploration_practises_another_card_never_an_experimental_one(playbook):
    seen = set()
    for seed in range(40):
        picked = P.playbook_order(
            playbook, _battle(OURS, DEFAULT), 1.0, random.Random(seed)
        )
        assert picked is not None
        card, order = picked
        assert card.endswith("*") and card != "water_room*"
        seen.add(card)
        picks = _picks(order)
        assert len(set(picks)) == 4
        spec = next(c for c in playbook.cards if c["name"] == card[:-1])
        assert picks[:2] == spec["lead"]
    assert seen == {"sun_room*", "support_room*"}  # fast_sun is experimental


def test_an_explored_card_keeps_its_own_back_line_rules(playbook):
    water = next(c for c in playbook.cards if c["name"] == "water_room")
    feats = __import__("vgc_bench.src.playbook", fromlist=["x"]).opponent_features(
        SAND, {}, FORMAT
    )
    order = P.card_order(water, feats, OURS)
    assert [OURS[i - 1] for i in order[2:]] == ["incineroar", "venusaur"]  # sand


def test_only_the_playbook_team_is_patched(monkeypatch: pytest.MonkeyPatch):
    from poke_env.player.player import Player

    monkeypatch.setattr(Player, "random_teampreview", lambda self, battle: "/team 6543")
    assert P.install(Path("data/playbook_t6.json"), 0.0, seed=3)
    patched: Any = Player.random_teampreview
    other = ["garchomp", "kingambit", "amoonguss", "scizor", "sinistcha", "dragonite"]
    assert patched(None, _battle(other, DEFAULT)) == "/team 6543"  # human model
    assert patched(None, _battle(OURS, DEFAULT)).startswith("/team 12")
    assert not P.install(Path("data/playbook_t6.json"), 0.0)  # idempotent
    assert patched._playbook_counts["water_room"] == 1


def test_a_broken_battle_falls_back_to_the_wrapped_preview(
    monkeypatch: pytest.MonkeyPatch,
):
    from poke_env.player.player import Player

    monkeypatch.setattr(Player, "random_teampreview", lambda self, battle: "/team 1234")
    P.install(Path("data/playbook_t6.json"), 0.0, seed=4)
    broken = _battle(OURS, DEFAULT)
    broken.opponent_team = {"p2: x": Pokemon(gen=9, species="pelipper")}  # not six
    preview: Any = Player.random_teampreview
    assert preview(None, broken) == "/team 1234"
    counts = preview._playbook_counts
    assert counts["fallback"] == 1
