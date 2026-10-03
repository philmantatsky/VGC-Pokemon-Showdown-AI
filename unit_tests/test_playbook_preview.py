"""The playbook at team preview (PolicyPlayer playbook_path): OUR four and leads
come from our own plan card, which is attached to the battle for the turn-1
guard and written to the decision log with its reasons; a broken playbook falls
back to the normal preview instead of stalling it."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from poke_env.battle import Pokemon
from poke_env.teambuilder import Teambuilder

from vgc_bench.src.opponent_preview import PreviewPlan
from vgc_bench.src.policy_player import PolicyPlayer

FORMAT = "gen9championsvgc2026regmc"
OURS = ["blastoise", "farigiraf", "charizard", "venusaur", "torkoal", "incineroar"]


def _player(tmp_path: Path, playbook: str = "data/playbook_t6.json") -> PolicyPlayer:
    player = PolicyPlayer.__new__(PolicyPlayer)
    player.playbook_path = Path(playbook)
    player._playbook = None
    player.preview_model_path = None
    player._battle_plans = {}
    player.decision_log_path = tmp_path / "decisions.jsonl"
    return player


def _battle(theirs: list[Pokemon]) -> Any:  # stands in for a DoubleBattle
    return SimpleNamespace(
        battle_tag="battle-gen9championsvgc2026regmc-1",
        format=FORMAT,
        team={f"p1: {s}": Pokemon(gen=9, species=s) for s in OURS},
        opponent_team={f"p2: {m.species}": m for m in theirs},
    )


def _mons(*species: str) -> list[Pokemon]:
    return [Pokemon(gen=9, species=s) for s in species]


@pytest.fixture(autouse=True)
def _counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PolicyPlayer, "guard_fire_counts", Counter())


def test_the_card_picks_our_four_and_leads(tmp_path: Path) -> None:
    player = _player(tmp_path)
    battle = _battle(
        _mons(
            "pelipper",
            "archaludon",
            "basculegion",
            "whimsicott",
            "gholdengo",
            "maushold",
        )
    )
    order = player._learned_teampreview(battle)  # no preview model: the playbook alone
    assert order is not None and order.startswith("/team ")
    picks = [OURS[int(i) - 1] for i in order.removeprefix("/team ")]
    assert picks[:2] == ["blastoise", "farigiraf"]  # Water Room, the default card
    assert sorted(picks[2:]) == ["incineroar", "venusaur"]  # Pelipper's Wide Guard
    brought = [
        s for s, m in zip(OURS, battle.team.values()) if m._selected_in_teampreview
    ]
    assert sorted(brought) == sorted(picks)
    assert battle._vgc_playbook["card"] == "water_room"
    assert battle._vgc_playbook["turn1"]["setter"]["move"] == "trickroom"
    row = json.loads((tmp_path / "decisions.jsonl").read_text().splitlines()[0])
    assert row["turn"] == 0 and row["playbook"]["card"] == "water_room"
    assert any("wide guard (pelipper)" in r for r in row["playbook"]["reasons"])
    assert PolicyPlayer.guard_fire_counts["playbook:water_room"] == 1


def test_an_open_sheet_is_read_over_the_usage_guess(tmp_path: Path) -> None:
    """Their sheet (poke-env's |showteam| path) shows Pelipper without Wide Guard;
    Pokemon whose sets are hidden are left out."""
    player = _player(tmp_path)
    pelipper, *rest = _mons(
        "pelipper", "kingambit", "amoonguss", "scizor", "sinistcha", "garchomp"
    )
    shown = "Pelipper @ Damp Rock\nAbility: Drizzle\n- Hurricane\n- Protect\n"
    pelipper._update_from_teambuilder(Teambuilder.parse_showdown_team(shown)[0])
    sheets = player._playbook_sheets(_battle([pelipper, *rest]))
    assert list(sheets) == ["pelipper"]
    assert sheets["pelipper"].ability == "drizzle"
    assert set(sheets["pelipper"].moves) == {"hurricane", "protect"}


def test_a_broken_playbook_falls_back_to_the_normal_preview(tmp_path: Path) -> None:
    player = _player(tmp_path, playbook=str(tmp_path / "missing.json"))
    battle = _battle(
        _mons(
            "pelipper",
            "archaludon",
            "basculegion",
            "whimsicott",
            "gholdengo",
            "maushold",
        )
    )
    assert player._learned_teampreview(battle) is None
    assert not hasattr(battle, "_vgc_playbook")
    assert PolicyPlayer.guard_fire_counts["playbook_error:FileNotFoundError"] == 1


class _Model:
    """A stand-in preview model: leads Torkoal + Incineroar, brings Blastoise and
    Farigiraf behind them."""

    def predict_plans(self, roster, other, top_k):
        return [PreviewPlan((4, 5), (4, 5, 0, 1), 1.0)]


def test_script_only_keeps_our_preview_and_attaches_the_card(tmp_path: Path) -> None:
    """playbook_script_only (2026-10-03): the model's four and leads are played;
    the card is attached (marked script_only) for its turn-1 script alone."""
    player = _player(tmp_path, playbook="data/playbook_t6_trial.json")
    player.playbook_script_only = True
    player.preview_model_path = Path("model.pt")
    player._preview_predictor = cast(Any, _Model())
    player.use_learned_teampreview = True
    battle = _battle(
        _mons(
            "pelipper",
            "archaludon",
            "basculegion",
            "whimsicott",
            "gholdengo",
            "maushold",
        )
    )
    assert player._learned_teampreview(battle) == "/team 5612"  # the model's plan
    brought = [
        s for s, m in zip(OURS, battle.team.values()) if m._selected_in_teampreview
    ]
    assert sorted(brought) == ["blastoise", "farigiraf", "incineroar", "torkoal"]
    assert battle._vgc_playbook["card"] == "water_room"
    assert battle._vgc_playbook["script_only"] is True
    assert PolicyPlayer.guard_fire_counts["learned_preview"] == 1
    assert PolicyPlayer.guard_fire_counts["playbook:water_room"] == 1


def test_script_only_without_a_preview_model_leaves_the_preview_to_the_policy(
    tmp_path: Path,
) -> None:
    player = _player(tmp_path, playbook="data/playbook_t6_trial.json")
    player.playbook_script_only = True
    battle = _battle(
        _mons(
            "pelipper",
            "archaludon",
            "basculegion",
            "whimsicott",
            "gholdengo",
            "maushold",
        )
    )
    assert player._learned_teampreview(battle) is None
    assert battle._vgc_playbook["script_only"] is True
    assert not any(
        getattr(m, "_selected_in_teampreview", False) for m in battle.team.values()
    )


def test_script_only_needs_a_playbook() -> None:
    with pytest.raises(ValueError, match="needs a playbook_path"):
        PolicyPlayer(playbook_script_only=True)
