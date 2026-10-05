"""Runtime of the opponent predictor: forecasts from a fake battle's stream.

A fake battle is three attributes (``_replay_data``, ``player_role``,
``battle_tag``) fed from a short hand-written log, from either seat, with the
seat's own HP made exact as a player receives it.
"""

from __future__ import annotations

import ast
import json
import threading
import zlib
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

from vgc_bench.src.oppmodel import artifact as A
from vgc_bench.src.oppmodel import events as E
from vgc_bench.src.oppmodel import features as F
from vgc_bench.src.oppmodel import public_state as P
from vgc_bench.src.oppmodel import runtime as R
from vgc_bench.src.oppmodel import tables as T
from vgc_bench.src.opponent_tactics import MovePrediction, MovePredictor

ROOT = Path(__file__).resolve().parents[1]
TAG = "battle-test-1"

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

# Seven turns: every kind of action from both sides, a faint with a
# replacement, a slot left empty (p1a from turn 6) and a forfeit.
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
|-damage|p1b: Rillaboom|0 fnt
|faint|p1b: Rillaboom
|move|p2a: Garchomp|Rock Slide|p1a: Charizard|[spread] p1a
|-damage|p1a: Charizard|30/100
|upkeep
|switch|p1b: Torkoal|Torkoal, L50, M|100/100
|turn|3
|move|p2b: Sneasler|Dire Claw|p1a: Charizard
|-damage|p1a: Charizard|0 fnt
|faint|p1a: Charizard
|cant|p1b: Torkoal|slp
|move|p2a: Garchomp|Protect|p2a: Garchomp
|-singleturn|p2a: Garchomp|Protect
|upkeep
|switch|p1a: Incineroar|Incineroar, L50, M|41/100
|turn|4
|move|p1a: Incineroar|Flare Blitz|p2b: Sneasler
|-damage|p2b: Sneasler|0 fnt
|faint|p2b: Sneasler
|move|p1b: Torkoal|Eruption|p2a: Garchomp|[spread] p2a
|-damage|p2a: Garchomp|60/100
|move|p2a: Garchomp|Dragon Claw|p1a: Incineroar
|-damage|p1a: Incineroar|5/100
|upkeep
|switch|p2b: Politoed|Politoed, L50, M|100/100
|turn|5
|move|p2a: Garchomp|Dragon Claw|p1a: Incineroar
|-damage|p1a: Incineroar|0 fnt
|faint|p1a: Incineroar
|move|p2b: Politoed|Icy Wind|p1b: Torkoal|[spread] p1b
|-damage|p1b: Torkoal|80/100
|move|p1b: Torkoal|Heat Wave|p2a: Garchomp|[spread] p2a,p2b
|-damage|p2a: Garchomp|30/100
|-damage|p2b: Politoed|70/100
|upkeep
|turn|6
|move|p2a: Garchomp|Dragon Claw|p1b: Torkoal
|-damage|p1b: Torkoal|40/100
|move|p2b: Politoed|Encore|p1b: Torkoal
|-fail|p1b: Torkoal
|move|p1b: Torkoal|Protect||[still]
|-fail|p1b: Torkoal
|upkeep
|turn|7
|-message|Alice forfeited.
|win|Bob
"""

LOG = HEADER + GAME
LEADS = {"p1": ["incineroar", "rillaboom"], "p2": ["garchomp", "sneasler"]}

SHEET_P1 = (
    "Incineroar||SitrusBerry|Intimidate|FakeOut,FlareBlitz,PartingShot,KnockOff"
    "|Careful||M|||50|]Rillaboom||MiracleSeed|GrassySurge|GrassyGlide,WoodHammer,"
    "FakeOut,Protect|Adamant||F|||50|]Charizard||Charcoal|Blaze|HeatWave,AirSlash,"
    "Protect,SolarBeam|Modest||M|||50|]Torkoal||Leftovers|Drought|Eruption,"
    "HeatWave,Protect,EarthPower|Quiet||M|||50|"
)
SHEET_P2 = (
    "Garchomp||Garchompite|RoughSkin|Earthquake,DragonClaw,Protect,RockSlide"
    "|Jolly||M|||50|]Sneasler||FocusSash|Unburden|CloseCombat,DireClaw,FakeOut,"
    "Protect|Jolly||M|||50|]Politoed||Leftovers|Drizzle|WeatherBall,Protect,"
    "IcyWind,Encore|Calm||M|||50|"
)

USES = {
    "garchomp": {"earthquake": 50, "protect": 30, "dragonclaw": 10, "rockslide": 5},
    "sneasler": {"closecombat": 40, "fakeout": 30, "direclaw": 20, "protect": 10},
    "politoed": {"icywind": 5, "encore": 3, "protect": 2, "fakeout": 1},
    "incineroar": {"fakeout": 40, "flareblitz": 30, "partingshot": 20, "knockoff": 9},
    "rillaboom": {"grassyglide": 30, "protect": 20, "woodhammer": 10, "fakeout": 5},
    "charizard": {"heatwave": 30, "protect": 20, "airslash": 5},
    "torkoal": {"eruption": 30, "heatwave": 20, "protect": 10, "earthpower": 5},
}

Maker = Callable[[dict[str, np.ndarray]], Any]


def other(role: str) -> str:
    return "p2" if role == "p1" else "p1"


def private_view(log: str, role: str) -> list[E.Event]:
    """The log as ``role`` receives it: its own HP exact, out of 187."""
    out: list[E.Event] = []
    for event in E.split_log(log):
        index = P.CONDITION_INDEX.get(event[1])
        ident = E.parse_ident(event[2]) if len(event) > 2 else None
        if index is not None and ident is not None and ident.side == role:
            head, _, tail = event[index].partition(" ")
            if "/" in head:
                exact = -(-int(head.split("/")[0]) * 187 // 100)
                event = list(event)
                event[index] = f"{exact}/187" + (f" {tail}" if tail else "")
        out.append(event)
    return out


def reference(
    role: str,
    log: str = LOG,
    sheets: dict[str, list[E.SheetSet]] | None = None,
    sheets_known: bool = False,
) -> P.DriveResult:
    """The dataset builder's own-page path: HP rewritten, then the whole log."""
    public = "\n".join(
        "|".join(P.rewrite_event(e, role)) for e in private_view(log, role)
    )
    result = P.drive_log(public, TAG, sheets=sheets, sheets_known=sheets_known)
    assert result.usable, result.skip_reason
    return result


def battle(role: str | None, tag: str = TAG) -> SimpleNamespace:
    return SimpleNamespace(_replay_data=[], player_role=role, battle_tag=tag)


class Hand:
    """A predictor made of a function of the batch; it keeps every batch."""

    kind = "hand"
    name = "hand_made"

    def __init__(self, make: Maker) -> None:
        self.make = make
        self.batches: list[dict[str, np.ndarray]] = []
        self.counters: Counter[str] = Counter()

    def predict(self, batch: dict[str, np.ndarray]) -> Any:
        self.batches.append(dict(batch))
        return self.make(batch)


def hashed(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Probabilities that depend on every feature value: any changed input shows."""
    mask = np.asarray(batch["action_mask"])
    legal = F.expand_target_mask(batch["cand_tmask"])
    action = np.zeros(mask.shape)
    target = np.zeros(legal.shape)
    mega = np.zeros(mask.shape[:2])
    for row in range(mask.shape[0]):
        raw = b"".join(
            np.ascontiguousarray(batch[name][row]).tobytes() for name in sorted(batch)
        )
        rng = np.random.default_rng(zlib.crc32(raw))
        action[row] = rng.random(mask.shape[1:]) + 0.05
        target[row] = rng.random(legal.shape[1:]) + 0.05
        mega[row] = rng.random(2)
    return {"action": action * mask, "target": target * legal, "mega": mega}


def aimed_at(n_cand: int, klass: int) -> Maker:
    """All mass on an aimed attack of each slot, aimed at target class ``klass``."""

    def make(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        mask = np.asarray(batch["action_mask"]).astype(bool)
        flags = np.asarray(batch["cand_flag"]).astype(int)
        legal = F.expand_target_mask(batch["cand_tmask"])
        action = np.zeros(mask.shape)
        target = np.zeros(legal.shape)
        wanted = F.CAND_VALID | F.CAND_DAMAGING | F.CAND_AIMED
        for row, slot in np.ndindex(mask.shape[:2]):
            columns = [
                c
                for c in range(n_cand)
                if flags[row, slot, c] & wanted == wanted and legal[row, slot, c, klass]
            ]
            if columns:
                action[row, slot, columns[0]] = 1.0
            elif mask[row, slot].any():
                action[row, slot, n_cand] = 1.0
        target[..., klass] = 1.0
        return {"action": action, "target": target, "mega": np.zeros(mask.shape[:2])}

    return make


def repertoire() -> F.Repertoire:
    out = F.Repertoire()
    for key, moves in USES.items():
        for move, count in moves.items():
            # One aimed move is on record as having hit as a spread move, so
            # its "auto" target class is legal (an invented statistic).
            seen_spread = (key, move) == ("politoed", "fakeout")
            out.add(key, move, count, auto=True if seen_spread else None)
    return out


@pytest.fixture(scope="module")
def fz() -> F.Featurizer:
    return F.Featurizer.build(repertoire())


def runtime(
    fz: F.Featurizer, make: Maker, **options: Any
) -> tuple[R.OpponentPredictor, Hand]:
    hand = Hand(make)
    options.setdefault("keep_features", True)
    return R.OpponentPredictor(hand, fz, **options), hand


def forecasts(
    rt: R.OpponentPredictor,
    role: str,
    log: str = LOG,
    tag: str = TAG,
    every_event: bool = False,
) -> dict[int, R.Forecast | None]:
    """Feed the seat's stream event by event; the forecast at each turn line."""
    fake = battle(role, tag)
    out: dict[int, R.Forecast | None] = {}
    for event in private_view(log, role):
        fake._replay_data.append(list(event))
        if event[1] == "turn":
            out[int(event[2])] = rt.predict(fake)
        elif every_event:
            rt.predict(fake)
    return out


def turn_line(events: list[E.Event], number: int) -> int:
    return next(
        i for i, e in enumerate(events) if e[1] == "turn" and e[2] == str(number)
    )


def served(made: R.Forecast | None) -> R.Forecast:
    assert made is not None
    return made


def filled(slot: R.SlotForecast | None) -> R.SlotForecast:
    assert slot is not None
    return slot


# --- turn starts ----------------------------------------------------------------


@pytest.mark.parametrize("role", ["p1", "p2"])
def test_a_forecast_exists_at_a_turn_start_and_not_between(fz: F.Featurizer, role: str):
    rt, hand = runtime(fz, F.uniform_prediction)
    events = private_view(LOG, role)
    first, second = turn_line(events, 1), turn_line(events, 2)
    fake = battle(role)
    fake._replay_data.extend(events[:first])
    assert rt.predict_with_reason(fake) == (None, "stand_down:no_turn_yet")
    fake._replay_data.append(events[first])
    made = served(rt.predict(fake))
    assert (made.turn, made.battle_tag) == (1, TAG)
    assert (made.player_role, made.opponent_role) == (role, other(role))
    assert (made.model_name, made.model_kind) == ("hand_made", "hand")
    assert [filled(slot).species for slot in made.slots] == LEADS[other(role)]
    assert [filled(slot).slot for slot in made.slots] == ["a", "b"]
    assert made.a is made.slots[0] and made.b is made.slots[1]
    assert made.slot("b") is made.b and made.slot(0) is made.a
    assert made.latency_ms > 0
    # Chatter after the turn line does not end the turn start.
    fake._replay_data.append(["", "inactive", "Time left: 120 sec this turn"])
    assert rt.predict(fake) is made
    # The first battle event of the turn does.
    fake._replay_data.append(events[first + 1])
    assert rt.predict_with_reason(fake) == (None, "stand_down:mid_turn")
    fake._replay_data.extend(events[first + 2 : second])
    assert rt.predict(fake) is None
    fake._replay_data.append(events[second])
    later = served(rt.predict(fake))
    assert later.turn == 2 and later is not made
    assert rt.counters["forecasts"] == 2 and len(hand.batches) == 2
    assert rt.counters["stand_down:mid_turn"] == 2
    assert not [name for name in rt.counters if "error" in name]


class Spy:
    """A battle that records every attribute anybody reads from it."""

    def __init__(self, role: str, tag: str) -> None:
        self.read: set[str] = set()
        self.battle_tag = tag
        self.player_role = role
        self._replay_data: list[E.Event] = []
        self.team = self.opponent_team = self._players = self.active_pokemon = None
        self.opponent_active_pokemon = self.rating = self.opponent_rating = None

    def __getattribute__(self, name: str) -> Any:
        if not name.startswith("__") and name != "read":
            object.__getattribute__(self, "read").add(name)
        return object.__getattribute__(self, name)


def test_only_three_attributes_of_the_battle_are_read(fz: F.Featurizer):
    rt, _ = runtime(fz, hashed)
    spy = Spy("p2", TAG)
    stream = object.__getattribute__(spy, "_replay_data")
    spy.read.clear()
    assert rt.note_sheets(spy, False)
    served_turns = 0
    for event in private_view(LOG, "p2"):
        stream.append(list(event))
        served_turns += int(rt.predict(spy) is not None and event[1] == "turn")
    assert rt.forget(spy) == 1 and served_turns == 7
    assert spy.read == {"_replay_data", "player_role", "battle_tag"}


def test_an_empty_opposing_slot_is_none(fz: F.Featurizer):
    rt, _ = runtime(fz, hashed)
    made = served(forecasts(rt, "p2")[6])  # the opponent (p1) has only slot b left
    assert made.a is None and made.b is not None and made.b.species == "torkoal"
    assert made.to_dict()["slots"][0] is None
    assert R.to_move_predictions(made) is None
    assert R.to_switch_predictions(made) is None
    loose = R.to_move_predictions(made, require_both=False)
    assert loose is not None and loose[0] == R.EMPTY_MOVES and loose[1].moves
    switches = R.to_switch_predictions(made, require_both=False)
    assert switches is not None and switches[0] == R.NO_SWITCH
    assert R.to_move_predictions(None) is None and R.to_switch_predictions(None) is None
    # From the other seat that empty slot is OURS: nobody can be attacked there.
    mine = served(forecasts(runtime(fz, hashed)[0], "p1")[6])
    for slot in mine.slots:
        assert filled(slot).p_attacks[0] == 0.0 and filled(slot).p_attacks[1] > 0.0
        assert filled(slot).p_fake_out_at[0] == 0.0
    assert mine.p_attacked[0] == 0.0


# --- orientation ----------------------------------------------------------------


@pytest.mark.parametrize("role", ["p1", "p2"])
def test_mass_on_foe_a_is_our_slot_a_from_either_seat(fz: F.Featurizer, role: str):
    rt, _ = runtime(fz, aimed_at(fz.n_cand, F.T_FOE_A), top_k=None)
    made = served(forecasts(rt, role)[1])
    for slot in map(filled, made.slots):
        assert slot.p_attacks == (1.0, 0.0)
        top = slot.top
        assert top is not None and top.kind == R.ACTION_MOVE
        assert top.target == E.TARGET_FOE_A and top.our_slot == 0
        assert slot.intents[E.INTENT_ATTACK_FOE_A] == pytest.approx(1.0)
        assert R.to_move_prediction(slot).actions[0][1] == "foe_a"
    assert made.p_attacked == (1.0, 0.0)
    # foe_a is the Pokemon in OUR slot a, whichever seat we have.
    features = made.features
    assert features is not None
    row = int(features["foe_mon"][0, 0])
    species = fz.vocab.species[int(features["mon_id"][0, row, F.ID_SPECIES])]
    assert species == LEADS[role][0]
    rt, _ = runtime(fz, aimed_at(fz.n_cand, F.T_FOE_B))
    made = served(forecasts(rt, role)[1])
    assert [filled(slot).p_attacks for slot in made.slots] == [(0.0, 1.0)] * 2
    assert made.p_attacked == (0.0, 1.0)
    assert filled(made.a).top is not None and filled(made.a).actions[0].our_slot == 1


def test_fake_out_mass_and_where_it_is_aimed(fz: F.Featurizer):
    # Seated p2, the opponent's slot a is an Incineroar whose first candidate
    # is its first-turn flinching move.
    rt, _ = runtime(fz, aimed_at(fz.n_cand, F.T_FOE_B))
    made = served(forecasts(rt, "p2")[1])
    first = filled(made.a)
    assert first.moves[0] == "fakeout" and first.p_fake_out == 1.0
    assert first.p_fake_out_at == (0.0, 1.0)
    assert filled(made.b).p_fake_out == 0.0  # its attack is another move
    assert R.is_first_turn_flinch("fakeout")
    assert not R.is_first_turn_flinch("firstimpression")  # first turn, no flinch
    assert not R.is_first_turn_flinch("protect")
    assert not R.is_first_turn_flinch("no such move")
    assert not R.is_first_turn_flinch("ironhead")  # may flinch, any turn
    if E.first_turn_only_source() == "sim_source":
        # Always flinches, but gated on the target's action, not on the turn.
        assert not R.is_first_turn_flinch("upperhand")


def test_first_turn_flinch_is_read_off_the_dex_entry(monkeypatch: pytest.MonkeyPatch):
    entries = {
        "always": {"secondary": {"chance": 100, "volatileStatus": "flinch"}},
        "listed": {"secondaries": [{"chance": 100, "volatileStatus": "flinch"}]},
        "sure": {"secondary": {"volatileStatus": "flinch"}},
        "sometimes": {"secondary": {"chance": 30, "volatileStatus": "flinch"}},
        "elsewise": {"secondary": {"chance": 100, "volatileStatus": "confusion"}},
        "plain": {"secondary": None},
        "anyturn": {"secondary": {"chance": 100, "volatileStatus": "flinch"}},
    }
    monkeypatch.setattr(R, "move_entry", entries.get)
    monkeypatch.setattr(R, "is_fake_out_like", lambda move: move != "anyturn")
    R._first_turn_flinch.cache_clear()
    try:
        answers = {name: R.is_first_turn_flinch(name) for name in entries}
        # Not a move id: False, and nothing raises (an unhashable argument
        # would raise inside a cache before the function's own guard).
        odd: list[Any] = [[], {}, None, 5, ["always"], object()]
        assert [R.is_first_turn_flinch(value) for value in odd] == [False] * 6
    finally:
        R._first_turn_flinch.cache_clear()
    assert answers == {
        "always": True,
        "listed": True,
        "sure": True,
        "sometimes": False,
        "elsewise": False,
        "plain": False,
        "anyturn": False,
    }


def test_a_move_the_dex_does_not_know_is_counted_with_other(fz: F.Featurizer):
    new = "|move|p2a: Garchomp|Earthquake|p1a: Incineroar|[spread] p1a,p1b"
    assert new in LOG
    log = LOG.replace(new, "|move|p2a: Garchomp|Brand New Move|p1a: Incineroar")
    rt, _ = runtime(fz, hashed, top_k=None)
    slot = filled(served(forecasts(rt, "p1", log)[2]).a)
    n = fz.n_cand
    # The move it showed is a candidate, but there is no id to name it by.
    assert slot.moves[0] == "" and slot.action_probs[0] > 0
    other_entries = [e for e in slot.actions if e.kind == R.ACTION_OTHER]
    assert len(other_entries) == 1
    both = slot.action_probs[n] + slot.action_probs[0]
    assert other_entries[0].probability == pytest.approx(both)
    assert sum(entry.probability for entry in slot.actions) == pytest.approx(1.0)
    assert slot.p_other == pytest.approx(slot.action_probs[n])
    assert slot.p_unassigned == pytest.approx(both)
    moves = R.to_move_prediction(slot)
    assert "" not in dict(moves.moves)
    assert sum(p for _, p in moves.moves) == pytest.approx(1 - slot.p_switch - both)
    assert rt.diagnostics()["featurizer"]["unknown_move"] >= 1


# --- scalars --------------------------------------------------------------------


def test_scalars_are_sums_over_the_action_and_target_arrays(fz: F.Featurizer):
    n = fz.n_cand
    checked = empty_ours = 0
    for role in ("p1", "p2"):
        rt, _ = runtime(fz, hashed, top_k=None)
        for made in forecasts(rt, role).values():
            made = served(made)
            features = made.features
            assert features is not None
            ours_up = features["foe_mon"][0] >= 0
            empty_ours += int(not ours_up.all())
            for index, slot in enumerate(made.slots):
                if slot is None:
                    continue
                a, t = slot.action_probs, slot.target_probs
                assert a.shape == (n + 7,) and t.shape == (n + 1, 5)
                assert not a.flags.writeable and not t.flags.writeable
                assert a.sum() == pytest.approx(1.0)
                assert slot.p_switch == pytest.approx(a[n + 1 :].sum())
                assert slot.p_other == pytest.approx(a[n])
                named = [(c, m) for c, m in enumerate(slot.moves) if m]
                protect = sum(a[c] for c, m in named if E.is_protect_family(m))
                assert slot.p_protect == pytest.approx(protect) and protect > 0
                fake = [c for c, m in named if m == "fakeout"]
                assert slot.p_fake_out == pytest.approx(sum(a[c] for c in fake))
                for ours, klass in enumerate((F.T_FOE_A, F.T_FOE_B)):
                    attack = flinch = 0.0
                    for c, m in named:
                        hit = t[c, klass] + t[c, F.T_AUTO]
                        if not E.is_choosable_target(m):
                            hit = float(E.is_spread(m))
                        attack += a[c] * E.is_damaging(m) * hit
                        if m == "fakeout":
                            flinch += a[c] * (t[c, klass] + t[c, F.T_AUTO])
                    if not ours_up[ours]:
                        attack = flinch = 0.0
                    assert slot.p_attacks[ours] == pytest.approx(attack)
                    assert slot.p_fake_out_at[ours] == pytest.approx(flinch)
                assert set(slot.intents) == set(E.INTENT_CLASSES)
                total = sum(slot.intents.values()) + slot.p_unassigned
                assert total == pytest.approx(1.0)
                assert slot.intents[E.INTENT_SWITCH] == pytest.approx(slot.p_switch)
                assert slot.intents[E.INTENT_PROTECT] == pytest.approx(slot.p_protect)
                assert slot.p_unassigned >= slot.p_other - 1e-12
                # The ranked list is the whole distribution, most probable first.
                masses = [entry.probability for entry in slot.actions]
                assert sum(masses) == pytest.approx(1.0)
                assert masses == sorted(masses, reverse=True)
                for entry in slot.actions:
                    if entry.kind == R.ACTION_SWITCH:
                        assert entry.roster_index is not None
                        assert entry.switch_to == slot.roster[entry.roster_index]
                        mass = a[n + 1 + entry.roster_index]
                    elif entry.kind == R.ACTION_OTHER:
                        assert entry.move is None and entry.target is None
                        mass = a[n]
                    else:
                        c = slot.moves.index(str(entry.move))
                        mass = a[c] * t[c, E.TARGET_CLASSES.index(str(entry.target))]
                    assert entry.probability == pytest.approx(mass)
                assert slot.move_probabilities() == tuple(
                    sorted(((m, a[c]) for c, m in named), key=lambda item: -item[1])
                )
                # Mega: the predictor's value where the public state allows one.
                can = features["mon_flag"][0, slot.roster_index, F.FLAG_MEGA_POSSIBLE]
                expected = float(made.raw["mega"][index]) if can else 0.0
                assert slot.p_mega == pytest.approx(expected)
                assert slot.roster[slot.roster_index] == slot.species
                assert slot.support == pytest.approx(
                    sum(USES[slot.species].values()), rel=2e-3
                )
                assert slot.support_log == pytest.approx(np.log1p(slot.support))
                checked += 1
            union = [1.0, 1.0]
            for slot in made.slots:
                for ours in range(2):
                    union[ours] *= 1.0 - (slot.p_attacks[ours] if slot else 0.0)
            assert made.p_attacked == pytest.approx((1 - union[0], 1 - union[1]))
    assert checked == 26 and empty_ours == 2


def test_top_k_cuts_the_ranked_list_only(fz: F.Featurizer):
    whole = served(forecasts(runtime(fz, hashed, top_k=None)[0], "p1")[2])
    short = served(forecasts(runtime(fz, hashed, top_k=3)[0], "p1")[2])
    default = served(forecasts(runtime(fz, hashed)[0], "p1")[2])
    for index in range(2):
        full = filled(whole.slots[index])
        assert len(full.actions) > R.DEFAULT_TOP_K
        assert filled(short.slots[index]).actions == full.actions[:3]
        assert filled(default.slots[index]).actions == full.actions[: R.DEFAULT_TOP_K]
        assert np.array_equal(
            filled(short.slots[index]).action_probs, full.action_probs
        )
        assert filled(short.slots[index]).p_switch == full.p_switch


def test_to_dict_is_plain_json(fz: F.Featurizer):
    made = served(forecasts(runtime(fz, hashed)[0], "p1")[1])
    data = json.loads(json.dumps(made.to_dict()))
    assert data["turn"] == 1 and data["model"] == "hand_made" and data["elo"] == 1250
    assert data["sheets"] == ["closed", "closed"]
    first = data["slots"][0]
    assert first["species"] == "garchomp" and len(first["actions"]) == R.DEFAULT_TOP_K
    assert set(first["intents"]) == set(E.INTENT_CLASSES)
    assert first["p_attacks"] == [round(v, 4) for v in filled(made.a).p_attacks]
    assert {"kind", "p"} <= set(first["actions"][0])


# --- Elo --------------------------------------------------------------------------


def test_known_unknown_and_blanked_elo(fz: F.Featurizer):
    rt, hand = runtime(fz, F.uniform_prediction)
    made = served(forecasts(rt, "p1")[1])
    assert (made.elo, made.own_elo) == (1250, 1300) and rt.elo_mode == F.ELO_KEEP
    assert hand.batches[0]["elo"].tolist() == [[1250, 1300]]
    assert hand.batches[0]["elo_known"].tolist() == [[1, 1]]
    rt, hand = runtime(fz, F.uniform_prediction)
    made = served(forecasts(rt, "p2")[1])
    assert (made.elo, made.own_elo) == (1300, 1250)
    # A challenge: the player lines carry no rating.
    unrated = LOG.replace("|Alice|1|1300", "|Alice|1|").replace(
        "|Bob|2|1250", "|Bob|2|"
    )
    rt, hand = runtime(fz, F.uniform_prediction)
    made = served(forecasts(rt, "p1", unrated)[1])
    assert (made.elo, made.own_elo) == (None, None)
    assert hand.batches[0]["elo_known"].tolist() == [[0, 0]]
    # Only the opponent's rating is missing.
    rt, hand = runtime(fz, F.uniform_prediction)
    made = served(forecasts(rt, "p1", LOG.replace("|Bob|2|1250", "|Bob|2|"))[1])
    assert (made.elo, made.own_elo) == (None, 1300)
    assert hand.batches[0]["elo"].tolist() == [[0, 1300]]
    # An Elo-blind featurizer never encodes a rating.
    blind = F.Featurizer.build(repertoire(), elo_mode=F.ELO_BLANK)
    rt, hand = runtime(blind, F.uniform_prediction)
    made = served(forecasts(rt, "p1")[1])
    assert (made.elo, made.own_elo) == (None, None) and rt.elo_mode == F.ELO_BLANK
    assert hand.batches[0]["elo_known"].tolist() == [[0, 0]]
    # An Elo-blind predictor blanks the ratings itself: none was "used".
    hand = Hand(F.uniform_prediction)
    hand.elo_mode = F.ELO_BLANK  # type: ignore[attr-defined]
    rt = R.OpponentPredictor(hand, fz)
    made = served(forecasts(rt, "p1")[1])
    assert (made.elo, made.own_elo) == (None, None) and rt.elo_mode == F.ELO_BLANK


# --- cache, forget, bound ---------------------------------------------------------


def test_a_second_call_in_the_same_turn_returns_the_same_forecast(fz: F.Featurizer):
    rt, hand = runtime(fz, hashed)
    events = private_view(LOG, "p1")
    fake = battle("p1")
    fake._replay_data.extend(events[: turn_line(events, 1) + 1])
    first = served(rt.predict(fake))
    assert rt.predict(fake) is first and rt.predict(fake) is first
    assert len(hand.batches) == 1 and rt.counters["cache_hit"] == 2
    # The other seat of the same battle tag is another battle to the runtime.
    events = private_view(LOG, "p2")
    twin = battle("p2")
    twin._replay_data.extend(events[: turn_line(events, 1) + 1])
    second = served(rt.predict(twin))
    assert second is not first and second.opponent_role == "p1" and rt.n_battles == 2
    assert rt.predict(fake) is first and rt.predict(twin) is second
    # Sheet knowledge that arrives at the turn start makes a new forecast.
    assert rt.note_sheets(fake, False)
    again = served(rt.predict(fake))
    assert again is not first and again.sheets_reported == ("closed", "closed")
    assert first.sheets_reported == ("unknown", "unknown")
    assert len(hand.batches) == 3


def test_a_failure_is_remembered_for_the_turn(fz: F.Featurizer):
    def broken(batch: dict[str, np.ndarray]) -> Any:
        raise RuntimeError("no")

    rt, hand = runtime(fz, broken)
    events = private_view(LOG, "p1")
    fake = battle("p1")
    fake._replay_data.extend(events[: turn_line(events, 1) + 1])
    reason = "opp_predictor_error:predictor:RuntimeError"
    assert rt.predict_with_reason(fake) == (None, reason)
    assert rt.predict_with_reason(fake) == (None, reason)
    assert len(hand.batches) == 1 and rt.counters[reason] == 1
    assert rt.counters["cache_hit"] == 1


def test_forget_and_the_bound_on_kept_battles(fz: F.Featurizer):
    rt, hand = runtime(fz, hashed, max_battles=2)
    events = private_view(LOG, "p1")
    fakes = [battle("p1", f"battle-test-{index}") for index in range(3)]
    for fake in fakes:
        fake._replay_data.extend(events[: turn_line(events, 2) + 1])
    first = served(rt.predict(fakes[0]))
    second = served(rt.predict(fakes[1]))
    assert rt.n_battles == 2 and "evicted" not in rt.counters
    assert rt.predict(fakes[0]) is first  # touched: battle 1 is now the oldest
    served(rt.predict(fakes[2]))
    assert rt.n_battles == 2 and rt.counters["evicted"] == 1
    assert rt.predict(fakes[0]) is first
    # The evicted battle is rebuilt from its stream and says the same.
    again = served(rt.predict(fakes[1]))
    assert again is not second and again.turn == 2
    assert np.array_equal(again.raw["action"], second.raw["action"])
    assert rt.counters["evicted"] == 2 and rt.counters["battles_tracked"] == 4
    assert rt.forget("battle-test-1") == 1 and rt.forget("battle-test-1") == 0
    assert rt.forget(fakes[2]) == 0  # evicted when battle 1 came back
    assert rt.forget(fakes[0]) == 1  # the battle object works too
    assert rt.forget("never seen") == 0 and rt.forget(None) == 0
    assert rt.n_battles == 0 and rt.counters["forgotten"] == 2
    # forget drops both seats of a battle.
    rt, _ = runtime(fz, hashed)
    forecasts(rt, "p1")
    forecasts(rt, "p2")
    assert rt.n_battles == 2 and rt.forget(TAG) == 2 and rt.n_battles == 0
    # A battle that was forgotten mid-game catches up from the whole stream.
    rt, _ = runtime(fz, hashed)
    fake = battle("p1")
    fake._replay_data.extend(events[: turn_line(events, 3) + 1])
    before = served(rt.predict(fake))
    rt.forget(TAG)
    after = served(rt.predict(fake))
    assert after is not before and after.turn == 3
    assert all(np.array_equal(after.raw[k], before.raw[k]) for k in before.raw)


# --- nothing raises -----------------------------------------------------------------


def test_garbage_battles_and_streams_never_raise(fz: F.Featurizer):
    rt, _ = runtime(fz, F.uniform_prediction)
    assert rt.predict_with_reason(None) == (None, "stand_down:no_battle_tag")
    assert rt.predict(object()) is None and rt.predict(SimpleNamespace()) is None
    no_role = SimpleNamespace(battle_tag="x", player_role="p3", _replay_data=[])
    assert rt.predict_with_reason(no_role) == (None, "stand_down:role_unknown")
    assert rt.predict(SimpleNamespace(battle_tag=5, player_role="p1")) is None
    text = SimpleNamespace(battle_tag="y", player_role="p1", _replay_data="a log")
    assert rt.predict_with_reason(text) == (None, "stand_down:no_stream")
    junk = [None, 5, ["x"], ["", "turn"], ["", "turn", "x"], ["", "move"], {"a": 1}]
    junk += [["", "-damage", None, None], ["", ["turn"], "4"], [], ["", "poke", "p1"]]
    odd = SimpleNamespace(battle_tag="z", player_role="p2", _replay_data=junk)
    assert rt.predict(odd) is None
    assert rt.note_sheets(None, True) is False
    assert (
        rt.note_sheets(SimpleNamespace(battle_tag="q", player_role=None), False)
        is False
    )
    assert rt.counters["sheets_ignored:no_battle_tag"] == 1
    assert rt.counters["sheets_ignored:role_unknown"] == 1
    # A real stream with fields dropped and events shuffled, called at every step.
    rng = np.random.default_rng(20261004)
    events = private_view(LOG, "p1")
    for trial in range(25):
        fake = battle("p1", f"battle-fuzz-{trial}")
        for event in events:
            event = list(event)
            roll = rng.random()
            if roll < 0.08:
                event = event[: int(rng.integers(0, len(event) + 1))]
            elif roll < 0.12:
                event[int(rng.integers(0, len(event)))] = "?"
            elif roll < 0.15:
                event = list(events[int(rng.integers(0, len(events)))])
            fake._replay_data.append(event)
            made = rt.predict(fake)
            assert made is None or isinstance(made, R.Forecast)
    assert all(isinstance(name, str) for name in rt.counters)
    assert rt.counters["forecasts"] > 0


def test_a_battle_the_dataset_drops_is_not_served(fz: F.Featurizer):
    # An event the tracker fails on: the state can no longer be vouched for.
    rt, hand = runtime(fz, F.uniform_prediction)
    events = private_view(LOG, "p1")
    first, second = turn_line(events, 1), turn_line(events, 2)
    fake = battle("p1")
    fake._replay_data.extend(events[: first + 1])
    assert rt.predict(fake) is not None
    fake._replay_data.append(["", "-damage", "p2a: Garchomp"])
    fake._replay_data.extend(events[first + 1 : second + 1])
    assert rt.predict_with_reason(fake) == (None, "stand_down:state_unreliable")
    assert rt.counters["state:feed_error:-damage:IndexError"] == 1
    assert len(hand.batches) == 1
    # A roster with a Pokemon whose logged identity can be false.
    masked = LOG.replace("|poke|p2|Pawmot, L50, M|", "|poke|p2|Zoroark, L50, M|")
    rt, hand = runtime(fz, F.uniform_prediction)
    made = forecasts(rt, "p1", masked)
    assert set(made.values()) == {None} and not hand.batches
    assert rt.counters["stand_down:illusion_species"] == 7
    assert reference("p1").usable and P.drive_log(masked).skip_reason == P.SKIP_ILLUSION


def test_a_failing_predictor_gives_none_and_a_counter(fz: F.Featurizer):
    events = private_view(LOG, "p1")

    def at_turn_one(rt: R.OpponentPredictor) -> tuple[R.Forecast | None, str]:
        fake = battle("p1")
        fake._replay_data.extend(events[: turn_line(events, 1) + 1])
        return rt.predict_with_reason(fake)

    def broken(batch: dict[str, np.ndarray]) -> Any:
        raise ZeroDivisionError

    rt, _ = runtime(fz, broken)
    reason = "opp_predictor_error:predictor:ZeroDivisionError"
    assert at_turn_one(rt) == (None, reason) and rt.counters[reason] == 1

    # A predictor that catches its own failure and answers uniformly is not served.
    hand = Hand(F.uniform_prediction)

    def degraded(batch: dict[str, np.ndarray]) -> Any:
        hand.counters["predict_error:ValueError"] += 1
        return F.uniform_prediction(batch)

    hand.make = degraded
    rt = R.OpponentPredictor(hand, fz)
    assert at_turn_one(rt) == (None, "opp_predictor_error:predictor_degraded")

    good = F.uniform_prediction
    bad_outputs: list[Maker] = [
        lambda batch: None,
        lambda batch: {},
        lambda batch: {"action": 1.0, "target": 2.0, "mega": 3.0},
        lambda batch: good(batch) | {"action": good(batch)["action"][:, :, :-1]},
        lambda batch: good(batch) | {"target": good(batch)["target"][:, :, :, :-1]},
        lambda batch: good(batch) | {"mega": np.zeros((1, 3))},
        lambda batch: good(batch) | {"action": good(batch)["action"] * np.nan},
        lambda batch: good(batch) | {"mega": np.array([[np.nan, 0.5]])},
    ]
    for make in bad_outputs:
        rt, _ = runtime(fz, make)
        assert at_turn_one(rt) == (None, "opp_predictor_error:prediction_malformed")

    # A failure on one turn does not silence the next.
    def first_turn_only(batch: dict[str, np.ndarray]) -> Any:
        if int(batch["turn"][0]) == 1:
            raise RuntimeError("no")
        return F.uniform_prediction(batch)

    rt, _ = runtime(fz, first_turn_only)
    made = forecasts(rt, "p1")
    assert made[1] is None and all(made[turn] is not None for turn in range(2, 8))
    assert rt.counters["forecasts"] == 6

    # Nothing loaded at all.
    rt = R.OpponentPredictor(None, None)
    assert not rt.loaded and at_turn_one(rt) == (None, "stand_down:not_loaded")
    assert rt.diagnostics()["runtime"] == {
        "predict_calls": 1,
        "stand_down:not_loaded": 1,
    }


def test_a_differing_dex_is_refused_unless_allowed(fz: F.Featurizer):
    meta = {"dex_signature_diff": ["first_turn_only_moves"]}
    rt, hand = runtime(fz, F.uniform_prediction, meta=meta)
    assert (
        rt.loaded and not rt.serving and "first_turn_only_moves" in str(rt.load_failure)
    )
    assert set(forecasts(rt, "p1").values()) == {None} and not hand.batches
    assert rt.counters["stand_down:dex_signature"] == 7
    rt, hand = runtime(fz, F.uniform_prediction, meta=meta, allow_dex_difference=True)
    assert rt.serving and forecasts(rt, "p1")[1] is not None
    assert rt.counters["dex_signature_differs"] == 1


def test_why_no_example_was_made_is_named():
    assert R._encode_reason({"encode_skip:no_active": 1}) == "stand_down:no_active"
    failed = R._encode_reason({"unknown_move": 2, "encode_error:KeyError": 1})
    assert failed == "opp_predictor_error:encode:KeyError"
    assert R._encode_reason({}) == "stand_down:no_example"
    assert R._predictor_failures(SimpleNamespace(counters={"predict_error:X": 2})) == 2
    assert R._predictor_failures(SimpleNamespace(counters={"other": 2})) == 0
    assert R._predictor_failures(object()) == 0


# --- two battles ------------------------------------------------------------------


def _alone(fz: F.Featurizer, role: str, log: str, tag: str) -> dict[int, R.Forecast]:
    made = forecasts(runtime(fz, hashed)[0], role, log, tag)
    return {turn: served(forecast) for turn, forecast in made.items()}


def _same(left: R.Forecast | None, right: R.Forecast) -> bool:
    return (
        left is not None
        and left.battle_tag == right.battle_tag
        and left.opponent_role == right.opponent_role
        and all(np.array_equal(left.raw[k], right.raw[k]) for k in right.raw)
        and left.features is not None
        and right.features is not None
        and all(
            np.array_equal(left.features[k], right.features[k]) for k in right.features
        )
    )


def test_two_interleaved_battles_do_not_mix(fz: F.Featurizer):
    lower = LOG.replace("|Bob|2|1250", "|Bob|2|1011")
    games = [
        ("p1", LOG, "battle-x"),
        ("p2", lower, "battle-y"),
        ("p1", lower, "battle-z"),
    ]
    alone = [_alone(fz, *game) for game in games]
    assert not _same(alone[0][1], alone[1][1]) and not _same(alone[0][1], alone[2][1])
    rt, _ = runtime(fz, hashed)
    fakes = [battle(role, tag) for role, _, tag in games]
    streams = [private_view(log, role) for role, log, _ in games]
    compared = 0
    for step in range(max(len(stream) for stream in streams)):
        for index, stream in enumerate(streams):
            if step >= len(stream):
                continue
            fakes[index]._replay_data.append(list(stream[step]))
            made = rt.predict(fakes[index])
            if stream[step][1] == "turn":
                assert _same(made, alone[index][int(stream[step][2])])
                compared += 1
    assert compared == 21 and rt.n_battles == 3


def test_two_threads_two_battles(fz: F.Featurizer):
    games = [("p1", LOG, "battle-x"), ("p2", LOG, "battle-y")]
    alone = [_alone(fz, *game) for game in games]
    rt, _ = runtime(fz, hashed)
    gate = threading.Barrier(len(games))
    wrong: list[str] = []

    def play(index: int) -> None:
        role, log, tag = games[index]
        try:
            gate.wait(timeout=10)
            for repeat in range(6):
                name = f"{tag}-{repeat}"
                made = forecasts(rt, role, log, name, every_event=True)
                for turn, forecast in made.items():
                    expected = alone[index][turn]
                    if forecast is None or not all(
                        np.array_equal(forecast.raw[k], expected.raw[k])
                        for k in expected.raw
                    ):
                        wrong.append(f"{name} turn {turn}")
                rt.forget(name)
        except Exception as exc:  # a thread must not die silently
            wrong.append(repr(exc))

    threads = [threading.Thread(target=play, args=(i,)) for i in range(len(games))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not wrong and not any(thread.is_alive() for thread in threads)
    assert rt.counters["forecasts"] == 2 * 6 * 7 and rt.n_battles == 0
    assert not [name for name in rt.counters if "error" in name]


# --- runtime == offline -------------------------------------------------------------


@pytest.mark.parametrize("role", ["p1", "p2"])
@pytest.mark.parametrize("every_event", [False, True])
def test_runtime_equals_the_offline_path(
    fz: F.Featurizer, role: str, every_event: bool
):
    rt, _ = runtime(fz, hashed)
    live = forecasts(rt, role, every_event=every_event)
    offline = reference(role)
    assert sorted(live) == [record.turn for record in offline.turns]
    assert any("187" in "|".join(event) for event in private_view(LOG, role))
    for record in offline.turns:
        example = fz.encode(record.snapshot, other(role))
        assert example is not None
        batch = F.sheet_unknown_as_closed(F.collate([example]))
        expected = hashed(batch)
        norm = F.normalize_prediction(expected, batch)
        made = served(live[record.turn])
        assert made.features is not None and set(made.features) == set(batch)
        for name in batch:
            assert np.array_equal(made.features[name], batch[name]), name
        for head in ("action", "target", "mega"):
            assert np.array_equal(made.raw[head], expected[head][0]), head
        for index, slot in enumerate(made.slots):
            assert (slot is not None) == bool(example["act_mon"][index] >= 0)
            if slot is not None:
                assert np.array_equal(slot.action_probs, norm["action"][0, index])
                assert np.array_equal(slot.target_probs, norm["target"][0, index])
        # The seat's exact HP never reaches a feature: whole percents only.
        hp = made.features["mon_hp"][0].astype(np.float64) * 100
        assert np.allclose(hp, np.round(hp), atol=0.03)


# --- sheets -------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["p1", "p2"])
def test_sheet_knowledge_reaches_the_model(fz: F.Featurizer, role: str):
    sheets = {"p1": E.parse_showteam(SHEET_P1), "p2": E.parse_showteam(SHEET_P2)}
    theirs, ours = sheets[other(role)], sheets[role]
    events = private_view(LOG, role)

    def first_turn(
        *told: Any, **sets: Any
    ) -> tuple[R.Forecast, Hand, R.OpponentPredictor]:
        rt, hand = runtime(fz, hashed)
        fake = battle(role)
        if told:
            assert rt.note_sheets(fake, *told, **sets) is True
        fake._replay_data.extend(events[: turn_line(events, 1) + 1])
        return served(rt.predict(fake)), hand, rt

    def offline(given: dict[str, list[E.SheetSet]] | None, known: bool) -> F.Batch:
        record = reference(role, sheets=given, sheets_known=known).turns[0]
        example = fz.encode(record.snapshot, other(role))
        assert example is not None
        return F.sheet_unknown_as_closed(F.collate([example]))

    def agrees(made: R.Forecast, batch: F.Batch) -> bool:
        return made.features is not None and all(
            np.array_equal(made.features[name], batch[name]) for name in batch
        )

    # Nothing said: unknown, served as closed.
    made, hand, _ = first_turn()
    assert made.sheets_reported == ("unknown", "unknown")
    assert made.sheets == ("closed", "closed")
    flags = hand.batches[0]["game_flag"][0]
    assert [flags[F.G_ACTOR_SHEET], flags[F.G_OTHER_SHEET]] == [F.SHEET_CLOSED] * 2
    assert agrees(made, offline(None, False))
    # Known to be closed: the same inputs, now as a fact.
    closed, _, rt = first_turn(False)
    assert closed.sheets_reported == ("closed", "closed") == closed.sheets
    assert rt.counters["sheets:closed"] == 1 and agrees(closed, offline(None, True))
    assert np.array_equal(closed.raw["action"], made.raw["action"])
    # Open, both sheets handed over: the candidates are the sheet's moves.
    both, _, rt = first_turn(True, opponent_sets=theirs, own_sets=ours)
    assert both.sheets_reported == ("open", "open") == both.sheets
    lead = filled(both.a)
    sheet = next(entry for entry in theirs if entry.species == lead.species)
    assert lead.moves[:4] == sheet.moves and set(lead.moves[4:]) == {""}
    assert lead.p_other == 0.0 or lead.action_probs[fz.n_cand] > 0
    assert rt.counters["sheets:open"] == 1
    assert agrees(both, offline({other(role): theirs, role: ours}, True))
    assert not np.array_equal(both.raw["action"], made.raw["action"])
    # Open, only the opponent's sheet: our own stays unknown (served as closed).
    half, _, rt = first_turn(True, opponent_sets=theirs)
    assert half.sheets_reported == ("open", "unknown")
    assert half.sheets == ("open", "closed")
    assert rt.counters["sheets:open_partial"] == 1
    assert agrees(half, offline({other(role): theirs}, False))
    # Open with nothing handed over is no knowledge at all.
    none, _, _ = first_turn(True)
    assert none.sheets_reported == ("unknown", "unknown")
    # Entries that are not sheet sets are left out, not fed.
    mixed: list[Any] = [None, "text", *theirs]
    junk, _, rt = first_turn(True, opponent_sets=mixed, own_sets=ours)
    assert agrees(junk, offline({other(role): theirs, role: ours}, True))
    assert not [name for name in rt.counters if "error" in name]
    # "Closed" with sheets attached: the sheets are ignored and counted.
    odd, _, rt = first_turn(False, opponent_sets=theirs)
    assert odd.sheets_reported == ("closed", "closed")
    assert rt.counters["sheets_ignored:closed_with_sets"] == 1
    assert filled(odd.a).moves == filled(closed.a).moves


def test_sheet_sets_from_a_poke_env_team():
    def mon(species: str, item: Any, ability: Any, moves: list[str]) -> Any:
        return SimpleNamespace(
            species=species, item=item, ability=ability, moves=dict.fromkeys(moves)
        )

    four = ["earthquake", "dragonclaw", "protect", "rockslide"]
    team = {
        "p2: Garchomp": mon("garchomp", "garchompite", "roughskin", four),
        "p2: Indeedee": mon("indeedeef", None, "psychicsurge", ["followme"]),
        "p2: Sneasler": mon("sneasler", "unknown_item", None, []),
        "p2: Politoed": mon("politoed", "unknown_item", None, ["icywind"]),
    }
    sets = R.sheet_sets_from_team(team)
    assert sets[0] == E.SheetSet(
        "garchomp", "garchomp", "garchompite", "roughskin", tuple(four)
    )
    assert sets[1] == E.SheetSet(
        "indeedee", "indeedeef", None, "psychicsurge", ("followme",)
    )
    # No moves: no sheet was shown for it. An unseen item is "no item known".
    assert [entry.species for entry in sets] == ["garchomp", "indeedee", "politoed"]
    assert sets[2].item is None and sets[2].ability is None
    assert R.sheet_sets_from_team(team.values()) == sets
    assert R.sheet_sets_from_team(team, sort_moves=True)[0].moves == tuple(sorted(four))
    for junk in (None, 5, [object()], {"a": None}, "text"):
        assert R.sheet_sets_from_team(junk) == []
    # Real poke-env objects built from a packed sheet give the sheet back.
    from poke_env.battle import Pokemon
    from poke_env.teambuilder import Teambuilder

    packed = Teambuilder.parse_packed_team(SHEET_P2)
    real = R.sheet_sets_from_team([Pokemon(gen=9, teambuilder=tb) for tb in packed])
    shown = E.parse_showteam(SHEET_P2)
    assert [(s.species, s.forme, s.item, s.ability, s.moves) for s in real] == [
        (s.species, s.forme, s.item, s.ability, s.moves) for s in shown
    ]


# --- the existing prediction shapes -------------------------------------------------


def _certain(
    fz: F.Featurizer,
    batch: dict[str, np.ndarray],
    actions: dict[str, E.SlotAction],
    side: str,
    roster: list[str],
) -> tuple[dict[str, np.ndarray], list[bool]]:
    """One-hot arrays for the true actions of ``side``'s slots, built here from
    the label reader's ``SlotAction`` (not by the runtime), and which slots
    could be stated."""
    mask = np.asarray(batch["action_mask"]).astype(bool)
    legal = F.expand_target_mask(batch["cand_tmask"])
    action = np.zeros(mask.shape)
    target = np.zeros(legal.shape)
    stated = [False, False]
    for index, letter in enumerate(E.SLOT_LETTERS):
        truth = actions.get(side + letter)
        if truth is None or not mask[0, index].any():
            continue
        if truth.kind == E.KIND_SWITCH:
            pointer = fz.n_cand + 1 + roster.index(str(truth.switch_to))
            if mask[0, index, pointer]:
                action[0, index, pointer] = 1.0
                stated[index] = True
        elif truth.kind == E.KIND_MOVE and truth.move and truth.target:
            names = [
                fz.tables.move_ids[int(row)] for row in batch["cand_move"][0, index]
            ]
            klass = E.TARGET_CLASSES.index(truth.target)
            if truth.move in names and legal[0, index, names.index(truth.move), klass]:
                action[0, index, names.index(truth.move)] = 1.0
                target[0, index, names.index(truth.move), klass] = 1.0
                stated[index] = True
        if not stated[index]:
            action[0, index] = mask[0, index] / mask[0, index].sum()
            target[0, index] = legal[0, index]
    return {"action": action, "target": target, "mega": np.zeros((1, 2))}, stated


def test_adapters_match_the_oracle_on_one_hot_forecasts(fz: F.Featurizer):
    from evaluation import oppmodel_oracle as O

    assert R.LEGACY_TARGET_NAMES == O._TARGET_NAMES
    assert set(R.LEGACY_TARGET_NAMES.values()) == set(MovePredictor.TARGET_NAMES)
    seen: Counter[str] = Counter()
    for role in ("p1", "p2"):
        side = other(role)
        offline = reference(role)
        truth = {record.turn: record.actions for record in offline.turns}
        roster = [mon.species for mon in offline.turns[0].snapshot.sides[side].mons]
        stated: dict[int, list[bool]] = {}

        def make(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            turn = int(batch["turn"][0])
            arrays, stated[turn] = _certain(fz, batch, truth[turn], side, roster)
            return arrays

        rt, _ = runtime(fz, make, top_k=None)
        for turn, made in forecasts(rt, role).items():
            for index, letter in enumerate(E.SLOT_LETTERS):
                slot = served(made).slots[index]
                action = truth[turn].get(side + letter)
                certain = O.certain_predictions(action)
                if slot is None or certain is None or not stated[turn][index]:
                    continue
                assert action is not None
                used = str(action.move)
                # The oracle states a certainty: reliability 1.0, given
                # explicitly (the default is the known share of the moveset).
                stated_move = R.to_move_prediction(slot, 1.0)
                assert stated_move == certain[0], (role, turn, letter)
                assert R.to_switch_prediction(slot) == certain[1], (role, turn, letter)
                gated = O.certain_predictions(action, reliability=0.25)
                assert gated is not None
                assert R.to_move_prediction(slot, 0.25) == gated[0]
                assert len(slot.actions) == 1 and slot.actions[0].probability == 1.0
                kind = str(action.target if action.kind == E.KIND_MOVE else action.kind)
                seen[kind] += 1
                if action.kind == E.KIND_MOVE:
                    assert slot.actions[0].move == action.move
                    assert slot.actions[0].target == action.target
                    assert slot.p_switch == 0.0
                    hits = [slot.p_attacks[0] == 1.0, slot.p_attacks[1] == 1.0]
                    if action.target == E.TARGET_FOE_A and E.is_damaging(used):
                        assert hits == [True, False]
                    if action.target == E.TARGET_FOE_B and E.is_damaging(used):
                        assert hits == [False, True]
                else:
                    assert slot.actions[0].switch_to == action.switch_to
                    assert slot.p_switch == 1.0 and slot.p_attacks == (0.0, 0.0)
    # Every kind the oracle states was met: both foes, a spread or unaimed
    # move, and a voluntary switch.
    assert seen[E.TARGET_FOE_A] >= 2 and seen[E.TARGET_FOE_B] >= 2
    assert seen[E.TARGET_AUTO] >= 3 and seen[E.KIND_SWITCH] == 1
    # The stream itself: on turn 2 the opponent's slot b hit p1b, OUR slot b
    # when we sit at p1.
    assert truth_of("p1", 2, "p2b").target == E.TARGET_FOE_B


def truth_of(role: str, turn: int, key: str) -> E.SlotAction:
    record = next(r for r in reference(role).turns if r.turn == turn)
    return record.actions[key]


def test_adapters_on_a_spread_out_forecast(fz: F.Featurizer):
    rt, _ = runtime(fz, hashed, top_k=None)
    made = served(forecasts(rt, "p1")[2])
    slot = filled(made.a)
    moves = R.to_move_prediction(slot)
    switch = R.to_switch_prediction(slot)
    n = fz.n_cand
    # Absolute probabilities: what is left after a switch and the OTHER bucket.
    assert moves.moves == slot.move_probabilities()
    # Reliability, by default, is the share of the moveset that is known.
    assert made.features is not None
    flags = made.features["cand_flag"][0, 0].astype(int)
    shown = int(((flags & F.CAND_VALID) > 0)[(flags & F.CAND_REVEALED) > 0].sum())
    assert 0 < shown < 4 and slot.known_moves == shown
    assert moves.reliability == shown / 4 == R.known_set_reliability(slot)
    assert R.to_move_prediction(slot, 1.0).reliability == 1.0
    assert R.to_move_prediction(slot, 7.0).reliability == 1.0  # clipped
    assert R.to_move_prediction(slot, -1.0).reliability == 0.0
    total = sum(p for _, p in moves.moves)
    assert total == pytest.approx(1.0 - slot.p_switch - slot.p_other)
    assert sum(p for _, _, p in moves.actions) == pytest.approx(total)
    assert [p for _, _, p in moves.actions] == sorted(
        (p for _, _, p in moves.actions), reverse=True
    )
    by_move: dict[str, float] = {}
    for move, name, p in moves.actions:
        assert name in MovePredictor.TARGET_NAMES
        by_move[move] = by_move.get(move, 0.0) + p
    assert dict(by_move) == pytest.approx(dict(moves.moves))
    # ``targets`` is the top move's own target distribution.
    top = slot.moves.index(moves.moves[0][0])
    assert dict(moves.targets) == pytest.approx(
        {
            R.LEGACY_TARGET_NAMES[klass]: slot.target_probs[top, k]
            for k, klass in enumerate(E.TARGET_CLASSES)
            if slot.target_probs[top, k] > 0
        }
    )
    assert sum(p for _, p in moves.targets) == pytest.approx(1.0)
    # The old predictor's shares among moves only.
    shares = R.to_move_prediction(slot, 0.5, conditional=True)
    assert sum(p for _, p in shares.moves) == pytest.approx(1.0)
    assert shares.reliability == 0.5
    assert [m for m, _ in shares.moves] == [m for m, _ in moves.moves]
    # Switch: the probability, and the destinations given a switch.
    assert switch.switch_probability == pytest.approx(slot.p_switch)
    assert sum(p for _, p in switch.targets) == pytest.approx(1.0)
    for species, p in switch.targets:
        pointer = slot.roster.index(species)
        assert p == pytest.approx(slot.action_probs[n + 1 + pointer] / slot.p_switch)
    assert (
        R.to_move_prediction(None) == R.EMPTY_MOVES == MovePrediction((), (), (), 0.0)
    )
    assert R.to_switch_prediction(None) == R.NO_SWITCH
    pair = R.to_move_predictions(made, 0.75)
    assert pair is not None and pair[1] == R.to_move_prediction(made.b, 0.75)
    default = R.to_move_predictions(made)
    assert default is not None and default == (
        R.to_move_prediction(made.a),
        R.to_move_prediction(made.b),
    )
    both = R.to_switch_predictions(made)
    assert both is not None and both[0] == switch
    # A broken slot degrades to "says nothing" and is counted.
    broken: Any = SimpleNamespace(action_probs=None, target_probs=None, moves=("x",))
    before = sum(R.COUNTERS.values())
    assert R.to_move_prediction(broken) == R.EMPTY_MOVES
    assert R.to_switch_prediction(broken) == R.NO_SWITCH
    assert sum(R.COUNTERS.values()) == before + 2
    assert R.known_set_reliability(None) == 0.0
    assert R.known_set_reliability(broken) == 0.0


def test_default_reliability_keeps_the_known_set_gate_closed(fz: F.Featurizer):
    """``MovePrediction.reliability`` is "how much of the set is known" to the
    reranker, whose switch-evidence gate opens at 0.999 on both slots. A
    closed-sheet forecast must not open it by default."""
    sheets = {"p1": E.parse_showteam(SHEET_P1), "p2": E.parse_showteam(SHEET_P2)}
    events = private_view(LOG, "p1")

    def at_turn(turn: int, open_sheets: bool) -> R.Forecast:
        rt, _ = runtime(fz, hashed, top_k=None)
        fake = battle("p1")
        if open_sheets:
            assert rt.note_sheets(
                fake, True, opponent_sets=sheets["p2"], own_sets=sheets["p1"]
            )
        else:
            assert rt.note_sheets(fake, False)
        fake._replay_data.extend(events[: turn_line(events, turn) + 1])
        return served(rt.predict(fake))

    for turn in (1, 2, 3):
        closed = at_turn(turn, False)
        assert closed.sheets == ("closed", "closed")
        pair = R.to_move_predictions(closed)
        assert pair is not None
        for slot, moves in zip(closed.slots, pair):
            slot = filled(slot)
            assert slot.known_moves < 4 and len(moves.moves) > 4
            assert moves.reliability == slot.known_moves / 4 < 0.999
        assert not all(moves.reliability >= 0.999 for moves in pair)
        assert (
            closed.to_dict()["slots"][0]["known_moves"] == filled(closed.a).known_moves
        )
    # Nothing shown yet: nothing is known, whatever the model says.
    assert filled(at_turn(1, False).a).known_moves == 0
    # An open sheet is a known set: the gate may open.
    opened = at_turn(2, True)
    pair = R.to_move_predictions(opened)
    assert pair is not None and [moves.reliability for moves in pair] == [1.0, 1.0]
    assert [filled(slot).known_moves for slot in opened.slots] == [4, 4]


def test_plural_adapters_never_raise_on_a_wrong_object():
    before = Counter(R.COUNTERS)
    wrong: list[Any] = [5, "x", object(), [], {}, 1.5]
    for value in wrong:
        assert R.to_move_predictions(value) is None
        assert R.to_switch_predictions(value) is None
        assert R.to_move_predictions(value, require_both=False) is None
        assert R.to_switch_predictions(value, require_both=False) is None
        assert R.to_move_prediction(value) == R.EMPTY_MOVES
        assert R.to_switch_prediction(value) == R.NO_SWITCH
    fresh = Counter(R.COUNTERS) - before
    assert fresh["to_move_predictions:AttributeError"] == 2 * len(wrong)
    assert fresh["to_switch_predictions:AttributeError"] == 2 * len(wrong)


# --- a prediction that says nothing -------------------------------------------------


def _wrapped(change: Callable[[dict[str, np.ndarray], F.Batch], None]) -> Maker:
    """The ``hashed`` prediction with one fault put into it."""

    def make(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        made = {name: value.copy() for name, value in hashed(batch).items()}
        change(made, batch)
        return made

    return make


def test_a_prediction_with_no_legal_mass_is_not_served(fz: F.Featurizer):
    """Normalising turns these into rows of zeros; served, the forecast would
    read "this slot does nothing and attacks nobody" with no counter raised."""
    events = private_view(LOG, "p1")

    def at_turn_two(make: Maker) -> tuple[R.Forecast | None, str, R.OpponentPredictor]:
        rt, _ = runtime(fz, make, top_k=None)
        fake = battle("p1")
        fake._replay_data.extend(events[: turn_line(events, 2) + 1])
        return (*rt.predict_with_reason(fake), rt)

    def negative(made: dict[str, np.ndarray], batch: F.Batch) -> None:
        made["action"] *= -1.0

    def all_zero(made: dict[str, np.ndarray], batch: F.Batch) -> None:
        made["action"][:] = 0.0

    def one_slot_zero(made: dict[str, np.ndarray], batch: F.Batch) -> None:
        made["action"][:, 1] = 0.0

    def mass_on_illegal_only(made: dict[str, np.ndarray], batch: F.Batch) -> None:
        made["action"] = 1.0 - np.asarray(batch["action_mask"]).astype(np.float64)

    def zero_target(made: dict[str, np.ndarray], batch: F.Batch) -> None:
        made["target"][:] = 0.0

    def one_target_row_zero(made: dict[str, np.ndarray], batch: F.Batch) -> None:
        made["target"][0, 0, 0] = 0.0

    def one_negative_entry(made: dict[str, np.ndarray], batch: F.Batch) -> None:
        column = int(np.flatnonzero(batch["action_mask"][0, 0])[0])
        made["action"][0, 0, column] = -0.2

    def negative_mega(made: dict[str, np.ndarray], batch: F.Batch) -> None:
        made["mega"][:] = -0.5

    healthy, reason, rt = at_turn_two(hashed)
    assert healthy is not None and reason == ""
    # The healthy forecast is what the faults below would have silenced.
    assert all(len(filled(slot).actions) > 3 for slot in healthy.slots)
    assert sum(sum(filled(slot).p_attacks) for slot in healthy.slots) > 0.1
    assert filled(healthy.a).action_probs[0] > 0  # the row one_target_row_zero hits
    assert "opp_predictor_error:prediction_malformed" not in rt.counters
    for fault in (
        negative,
        all_zero,
        one_slot_zero,
        mass_on_illegal_only,
        zero_target,
        one_target_row_zero,
        one_negative_entry,
        negative_mega,
    ):
        made, reason, rt = at_turn_two(_wrapped(fault))
        assert made is None, fault.__name__
        assert reason == "opp_predictor_error:prediction_malformed", fault.__name__
        assert rt.counters[reason] == 1 and "forecasts" not in rt.counters

    # Mass on illegal entries next to legal mass is renormalised away, as the
    # scorecard does it: that is still a probability table.
    def some_illegal(made: dict[str, np.ndarray], batch: F.Batch) -> None:
        made["action"] += 0.01
        made["target"] += 0.01

    made, reason, _ = at_turn_two(_wrapped(some_illegal))
    assert made is not None and reason == ""
    for slot in made.slots:
        assert filled(slot).action_probs.sum() == pytest.approx(1.0, abs=1e-9)


# --- battles the model was not fitted for, and damaged streams ------------------------


def test_a_battle_of_another_format_is_not_served(fz: F.Featurizer):
    """The dataset builder drops a battle whose format is not one of its own;
    an artifact that names its formats is served on those only."""
    played = "gen9championsvgc2026regmc"
    elsewhere = LOG.replace(
        "|tier|[Gen 9 Champions] VGC 2026 Reg M-C", "|tier|[Gen 9] Doubles OU"
    )
    assert reference("p1").format_id == played
    assert P.drive_log(elsewhere).format_id == "gen9doublesou"

    def meta(*formats: str) -> dict[str, Any]:
        return {"extra": {"formats": list(formats)}}

    rt, hand = runtime(fz, hashed, meta=meta(played, played + "bo3"))
    assert all(made is not None for made in forecasts(rt, "p1").values())
    rt, hand = runtime(fz, hashed, meta=meta(played, played + "bo3"))
    made = forecasts(rt, "p1", elsewhere)
    assert set(made.values()) == {None} and not hand.batches
    assert rt.counters["stand_down:format"] == 7
    # Nested under the dataset block, as the trainer's report stores it.
    nested = {"extra": {"dataset": {"formats": [played]}}}
    rt, hand = runtime(fz, hashed, meta=nested)
    assert set(forecasts(rt, "p1", elsewhere).values()) == {None}
    assert rt.counters["stand_down:format"] == 7
    # An artifact that names no format is served everywhere, as before.
    for loose in ({}, {"extra": {}}, {"extra": {"formats": []}}, {"extra": None}):
        rt, _ = runtime(fz, hashed, meta=loose)
        assert all(m is not None for m in forecasts(rt, "p1", elsewhere).values())
    # A stream that never names its format is served.
    silent = LOG.replace("|tier|[Gen 9 Champions] VGC 2026 Reg M-C\n", "")
    assert silent != LOG
    rt, _ = runtime(fz, hashed, meta=meta(played))
    assert all(made is not None for made in forecasts(rt, "p1", silent).values())


def test_a_lost_turn_line_or_a_skipped_entry_stands_down_for_the_battle(
    fz: F.Featurizer,
):
    events = private_view(LOG, "p1")
    third = turn_line(events, 3)

    def play(stream: list[Any]) -> tuple[dict[int, Any], R.OpponentPredictor, Hand]:
        rt, hand = runtime(fz, hashed)
        fake = battle("p1")
        out: dict[int, Any] = {}
        for event in stream:
            fake._replay_data.append(event)
            if isinstance(event, list) and len(event) > 2 and event[1] == "turn":
                out[int(event[2])] = rt.predict_with_reason(fake)
        return out, rt, hand

    clean, rt, hand = play([list(event) for event in events])
    assert all(made is not None for made, _ in clean.values()) and len(clean) == 7
    assert not [name for name in rt.counters if name.startswith("stand_down")]
    # The line "turn 3" is lost: turns 2 and 3 are read as one turn.
    lost = [list(event) for index, event in enumerate(events) if index != third]
    made, rt, hand = play(lost)
    assert sorted(made) == [1, 2, 4, 5, 6, 7]
    assert made[1][0] is not None and made[2][0] is not None
    for turn in (4, 5, 6, 7):
        assert made[turn] == (None, "stand_down:turn_gap"), turn
    assert rt.counters["stand_down:turn_gap"] == 4 and len(hand.batches) == 2
    # A fresh runtime that meets the damaged stream late sees the gap as well.
    rt, hand = runtime(fz, hashed)
    fake = battle("p1")
    fake._replay_data.extend(lost)
    fake._replay_data[:] = lost[: turn_line(lost, 5) + 1]
    assert rt.predict_with_reason(fake) == (None, "stand_down:turn_gap")
    assert not hand.batches
    # An entry that is not an event was skipped: something is missing.
    for junk in (None, 5, ["only one field"], {"a": 1}, ["", "move", 3]):
        damaged: list[Any] = [list(event) for event in events]
        damaged.insert(third + 2, junk)
        made, rt, hand = play(damaged)
        assert made[1][0] is not None and made[2][0] is not None
        assert made[3][0] is not None  # the entry came after the turn-3 line
        for turn in (4, 5, 6, 7):
            assert made[turn] == (None, "stand_down:stream_damaged"), (junk, turn)
        assert rt.counters["sync_skip:not_an_event"] == 1
        assert len(hand.batches) == 3


def test_sheets_survive_the_eviction_of_a_battle(fz: F.Featurizer):
    """A shadow rebuilt after an eviction gets the sheets it was told again:
    the model's input must not change from open to closed mid-battle."""
    sheets = {"p1": E.parse_showteam(SHEET_P1), "p2": E.parse_showteam(SHEET_P2)}
    events = private_view(LOG, "p1")
    second, third = turn_line(events, 2), turn_line(events, 3)

    def run(max_battles: int, *told: Any, **sets: Any) -> tuple[Any, Any, Any]:
        rt, _ = runtime(fz, hashed, max_battles=max_battles)
        fake, rival = battle("p1"), battle("p1", "battle-test-2")
        assert rt.note_sheets(fake, *told, **sets)
        fake._replay_data.extend(events[: second + 1])
        before = served(rt.predict(fake))
        rival._replay_data.extend(events[: second + 1])
        served(rt.predict(rival))  # with a bound of one, this evicts the first
        fake._replay_data.extend(events[second + 1 : third + 1])
        return before, served(rt.predict(fake)), rt

    for told, sets, state in (
        ((False,), {}, ("closed", "closed")),
        (
            (True,),
            {"opponent_sets": sheets["p2"], "own_sets": sheets["p1"]},
            ("open",) * 2,
        ),
        ((True,), {"opponent_sets": sheets["p2"]}, ("open", "unknown")),
    ):
        _, kept, rt = run(8, *told, **sets)
        assert "evicted" not in rt.counters and kept.sheets_reported == state
        before, after, rt = run(1, *told, **sets)
        assert rt.counters["evicted"] == 2 and rt.counters["sheets_restored"] == 1
        assert before.sheets_reported == state == after.sheets_reported
        assert after.features is not None and kept.features is not None
        assert all(
            np.array_equal(after.features[name], kept.features[name])
            for name in kept.features
        )
        assert filled(after.a).moves == filled(kept.a).moves
    # forget() drops the note with the battle: the next game under the same
    # tag starts from "unknown".
    rt, _ = runtime(fz, hashed)
    fake = battle("p1")
    assert rt.note_sheets(fake, False)
    fake._replay_data.extend(events[: second + 1])
    assert served(rt.predict(fake)).sheets_reported == ("closed", "closed")
    assert rt.forget(TAG) == 1
    assert served(rt.predict(fake)).sheets_reported == ("unknown", "unknown")
    assert "sheets_restored" not in rt.counters
    # A later note replaces the earlier one.
    rt, _ = runtime(fz, hashed, max_battles=1)
    fake, rival = battle("p1"), battle("p1", "battle-test-2")
    assert rt.note_sheets(fake, False)
    assert rt.note_sheets(fake, True, opponent_sets=sheets["p2"], own_sets=sheets["p1"])
    rival._replay_data.extend(events[: second + 1])
    served(rt.predict(rival))
    fake._replay_data.extend(events[: second + 1])
    assert served(rt.predict(fake)).sheets_reported == ("open", "open")


# --- artifacts ----------------------------------------------------------------------


def test_load_serves_an_artifact_and_stands_down_without_one(
    tmp_path: Path, fz: F.Featurizer
):
    rows = []
    for role in ("p1", "p2"):
        for record in reference(role).turns:
            rows.append(fz.encode_turn(record.snapshot, record.actions, other(role)))
    table = T.FlagsTable.fit(F.collate([row for row in rows if row]), featurizer=fz)
    path = tmp_path / "tiny.pt"
    A.save_artifact(
        path,
        kind=A.KIND_TABLE,
        name="tiny_flags",
        featurizer=fz,
        predictor_payload=table.to_payload(),
        extra={"dataset": "unit"},
    )
    rt = R.OpponentPredictor.load(path, keep_features=True)
    assert rt.loaded and rt.serving and rt.load_failure is None
    assert (rt.name, rt.kind, rt.elo_mode) == ("tiny_flags", A.KIND_TABLE, F.ELO_KEEP)
    assert rt.meta["extra"] == {"dataset": "unit"} and rt.featurizer is not None
    offline = A.load_predictor(path)
    live = forecasts(rt, "p1")
    for record in reference("p1").turns:
        example = offline.featurizer.encode(record.snapshot, "p2")
        assert example is not None
        batch = F.sheet_unknown_as_closed(F.collate([example]))
        expected = offline.predictor.predict(batch)
        made = served(live[record.turn])
        assert (made.model_name, made.model_kind) == ("tiny_flags", A.KIND_TABLE)
        for head in ("action", "target", "mega"):
            assert np.array_equal(made.raw[head], expected[head][0])
    assert not rt.diagnostics()["predictor"]
    blind = R.OpponentPredictor.load(path, elo_mode=F.ELO_BLANK)
    assert (
        blind.elo_mode == F.ELO_BLANK and served(forecasts(blind, "p1")[1]).elo is None
    )
    assert blind.predict(battle("p1")) is None  # features are not kept by default
    assert served(forecasts(blind, "p1", tag="battle-test-9")[1]).features is None

    missing = R.OpponentPredictor.load(tmp_path / "nothing.pt")
    assert not missing.loaded and not missing.serving
    assert "FileNotFoundError" in str(missing.load_failure)
    assert missing.counters["load_error:FileNotFoundError"] == 1
    assert set(forecasts(missing, "p1").values()) == {None}
    assert missing.counters["stand_down:not_loaded"] == 7
    assert missing.note_sheets(battle("p1"), False) is True  # harmless
    (tmp_path / "junk.pt").write_bytes(b"not an artifact")
    assert not R.OpponentPredictor.load(tmp_path / "junk.pt").loaded
    with pytest.raises((OSError, ValueError)):
        R.OpponentPredictor.load(tmp_path / "nothing.pt", strict=True)
    with pytest.raises(ValueError):
        R.OpponentPredictor.load(tmp_path / "junk.pt", strict=True)
    assert not R.OpponentPredictor.load(path, elo_mode="no such mode").loaded


# --- house rules --------------------------------------------------------------------


def test_library_names_nothing_from_the_dex_and_has_no_bare_assert(fz: F.Featurizer):
    known = set(E.moves_dex()) | set(E.pokedex())
    known |= set(fz.vocab.items[3:]) | set(fz.vocab.abilities[3:])
    for name in (
        "vgc_bench/src/oppmodel/runtime.py",
        "evaluation/oppmodel_shadow_replay.py",
    ):
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
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
        assert not found, (name, sorted(found))
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.Assert)], name
    tree = ast.parse((ROOT / "vgc_bench/src/oppmodel/runtime.py").read_text("utf-8"))
    raises = [node for node in ast.walk(tree) if isinstance(node, ast.Raise)]
    # The one raise: load(strict=True) passing the loader's error on.
    assert len(raises) == 1 and raises[0].exc is None


# --- the acceptance script ----------------------------------------------------------


def _page(S: Any, role: str) -> Any:
    events = private_view(LOG, role)
    public = "\n".join("|".join(P.rewrite_event(event, role)) for event in events)
    raw = S.RawBattle(
        "test-1", "own", public, own=True, bot_side=role, sheets_known=False
    )
    return S.Page("test-1", TAG, role, raw, events, 7)


def test_shadow_replay_compares_and_reports(fz: F.Featurizer):
    from evaluation import oppmodel_shadow_replay as S

    reports = []
    for role in ("p1", "p2"):
        page = _page(S, role)
        examples, dropped, repeated = S.offline_examples(page, fz, set())
        assert (
            dropped is None and repeated == 0 and sorted(examples) == list(range(1, 8))
        )
        rt, hand = runtime(fz, hashed)
        calls, after, leaked = S.replay(rt, page)
        assert [call.turn for call in calls] == list(range(1, 8)) and after == 0
        assert all(call.served and call.ms > 0 and call.inner_ms > 0 for call in calls)
        assert rt.n_battles == 0  # the replay forgets the battle
        tally = S.Equality()
        S.compare_page(tally, page, calls, examples, dropped, Hand(hashed))
        calls, after, leaked = S.replay(rt, page, every_event=True)
        assert after > 40 and leaked == 0
        every = S.Equality()
        S.compare_page(every, page, calls, examples, dropped, Hand(hashed))
        every.after_action_calls, every.after_action_forecasts = after, leaked
        for made in (tally.to_dict(), every.to_dict()):
            assert made["compared"] == 7 and made["turn_lines"] == 7
            assert set(made["max_abs_diff"].values()) == {0.0}
            assert set(made["max_abs_diff_one_example_per_call"].values()) == {0.0}
            assert not made["feature_arrays_differing"]
            assert made["runtime_none_where_offline_has_an_example"]["count"] == 0
        assert S.cold_call(rt, page) is not None
        light = [(page, [call.light() for call in calls])]
        latency = S.latency_report(light, [1.5], 5, 3, 5)
        assert latency["all_turn_starts"]["n"] == 7
        assert latency["long_games"]["early_turns_1_to_3"]["n"] == 3
        assert latency["long_games"]["late_turns_5_on"]["n"] == 3
        assert list(latency["by_turn"]) == ["1", "2-5", "6-9"]
        reports.append(
            {
                "name": f"hand_{role}",
                "path": "memory",
                "kind": "hand",
                "elo_mode": "keep",
                "loaded": True,
                "load_failure": None,
                "equality": {"per_turn": tally.to_dict(), "per_event": every.to_dict()},
                "latency_ms": latency,
                "counters": {"runtime_pass_1": dict(rt.counters)},
            }
        )
    report: dict[str, Any] = {"pages": {"found": 1}, "artifacts": reports}
    assert S.failures(report) == []
    text = S.render_markdown(report | {"failures": []})
    assert "PASS" in text and "hand_p1" in text and "per_event" in text
    # A runtime that says something else, or nothing, fails by name.
    page = _page(S, "p1")
    examples, dropped, _ = S.offline_examples(page, fz, set())
    rt, _ = runtime(fz, F.uniform_prediction)
    calls, _, _ = S.replay(rt, page)
    wrong = S.Equality()
    S.compare_page(wrong, page, calls, examples, dropped, Hand(hashed))
    assert wrong.max_diff["action"] > 0.01 and wrong.feature_mismatches == {}
    silent = S.Equality()
    none = [S.TurnCall(call.turn, None, "stand_down:mid_turn", 0.1) for call in calls]
    S.compare_page(silent, page, none, examples, dropped, Hand(hashed))
    assert silent.runtime_none == {"stand_down:mid_turn": 7} and silent.compared == 0
    extra = S.Equality()
    S.compare_page(extra, page, calls, {}, "illusion_species", Hand(hashed))
    assert extra.runtime_only == {"illusion_species": 7}
    bad = dict(reports[0])
    bad["equality"] = {"per_turn": wrong.to_dict(), "per_event": silent.to_dict()}
    bad["counters"] = {"runtime_pass_1": {"opp_predictor_error:predict:KeyError": 1}}
    found = S.failures({"pages": {"found": 1}, "artifacts": [bad]})
    assert any("max_abs_diff action" in item for item in found)
    assert any("runtime None" in item for item in found)
    assert any("compared nothing" in item for item in found)
    assert any("counted" in item for item in found)
    assert any("no mid-turn call" in item for item in found)
    assert S.failures({"pages": {"found": 0}, "artifacts": []}) != []
    leaky = dict(reports[0])
    leaky["memory"] = [
        {
            "mode": "forget",
            "shadows_kept": 2,
            "max_battles": 64,
            "python_bytes_held": 1000,
            "python_bytes_held_after_the_same_battles_again": 500_000,
        },
        {"mode": "bounded", "shadows_kept": 70, "max_battles": 64},
    ]
    found_memory = S.failures({"pages": {"found": 1}, "artifacts": [leaky]})
    assert len(found_memory) == 3 and any("grew" in item for item in found_memory)
    assert "FAIL" in S.render_markdown({"failures": found, "artifacts": [bad]})
    assert S.summary([]) == {
        "n": 0,
        "p50": None,
        "p90": None,
        "p99": None,
        "max": None,
        "mean": None,
    }
    assert S.summary([1.0, 2.0, 3.0, 10.0])["max"] == 10.0
    spread = S.summary([1.0] * 99 + [50.0])
    assert spread["p50"] == 1.0 and spread["max"] == 50.0
    assert 1.0 < spread["p99"] < 50.0  # one slow call in a hundred is not the p99
    # The p99 over all turn starts is gated; the largest single call is not.
    good = {"pages": {"found": 1}, "artifacts": reports, "p99_budget_ms": 25.0}
    assert S.failures(good) == []
    slow = dict(reports[0])
    slow["latency_ms"] = {
        **slow["latency_ms"],
        "all_turn_starts": {"n": 7, "p50": 2.0, "p90": 3.0, "p99": 31.5, "max": 40.0},
    }
    found = S.failures({**good, "artifacts": [slow]})
    assert len(found) == 1 and "p99" in found[0] and "31.5" in found[0]
    spike = dict(reports[0])
    spike["latency_ms"] = {
        **spike["latency_ms"],
        "all_turn_starts": {"n": 7, "p50": 2.0, "p90": 3.0, "p99": 4.0, "max": 900.0},
    }
    assert S.failures({**good, "artifacts": [spike]}) == []
    assert S.failures({**good, "artifacts": [slow], "p99_budget_ms": None}) == []
    text = S.render_markdown({**good, "failures": []})
    assert "| calls | n | p50 | p90 | p99 | max | mean |" in text and "Budget" in text
    # The default artifacts are the models meant to be served, never a smoke.
    assert not [path for path in S.DEFAULT_ARTIFACTS if "smoke" in path]
    assert any("tables" in path for path in S.DEFAULT_ARTIFACTS)


def test_shadow_replay_hands_over_a_real_poke_env_battle(fz: F.Featurizer):
    """The third pass: the object is a poke-env ``DoubleBattle`` and the stream
    is the one poke-env's own ``parse_message`` kept."""
    from evaluation import oppmodel_shadow_replay as S

    for role in ("p1", "p2"):
        page = _page(S, role)
        assert S.bot_username(page) == {"p1": "Alice", "p2": "Bob"}[role]
        examples, dropped, _ = S.offline_examples(page, fz, set())
        rt, hand = runtime(fz, hashed)
        notes: Counter[str] = Counter()
        calls = S.replay_real(rt, page, notes)
        assert calls is not None and notes["pages"] == 1
        assert "stream_differs" not in notes and "role_differs" not in notes
        assert [call.turn for call in calls] == list(range(1, 8))
        assert all(call.served for call in calls) and rt.n_battles == 0
        tally = S.Equality()
        S.compare_page(tally, page, calls, examples, dropped, Hand(hashed))
        made = tally.to_dict()
        assert made["compared"] == 7 and not made["feature_arrays_differing"]
        assert set(made["max_abs_diff"].values()) == {0.0}
        assert made["runtime_none_where_offline_has_an_example"]["count"] == 0
        # The same forecasts as from the three-attribute stand-in.
        plain, _ = runtime(fz, hashed)
        fake_calls, _, _ = S.replay(plain, page)
        for real, fake in zip(calls, fake_calls):
            assert real.forecast is not None and fake.forecast is not None
            assert np.array_equal(
                real.forecast.raw["action"], fake.forecast.raw["action"]
            )
            assert real.forecast.player_role == role == fake.forecast.player_role
    # A page that names no player cannot be given to poke-env: said, not raised.
    nameless = _page(S, "p1")
    nameless.events = [e for e in nameless.events if e[1] != "player"]
    notes = Counter()
    assert S.replay_real(runtime(fz, hashed)[0], nameless, notes) is None
    assert notes == {"no_player_line": 1}
    # What the acceptance list makes of the notes.
    entry = {
        "name": "x",
        "loaded": True,
        "equality": {
            "per_turn": made,
            "per_event": {**made, "calls_after_a_battle_event": 5},
            S.PASS_REAL: made,
        },
        "real_battle_notes": {"pages": 1},
    }
    report = {"pages": {"found": 1}, "artifacts": [entry]}
    assert S.failures(report) == []
    differs = {**entry, "real_battle_notes": {"stream_differs": 2, "pages": 1}}
    found = S.failures({**report, "artifacts": [differs]})
    assert len(found) == 1 and "stream_differs on 2" in found[0]
    lost = {**entry, "real_battle_notes": {"no_player_line": 1}}
    lost["equality"] = {k: v for k, v in entry["equality"].items() if k != S.PASS_REAL}
    found = S.failures({**report, "artifacts": [lost]})
    assert len(found) == 1 and "real-battle pass compared nothing" in found[0]


def test_shadow_replay_does_not_write_over_a_report(tmp_path: Path):
    from evaluation import oppmodel_shadow_replay as S

    (tmp_path / S.REPORT_JSON).write_text('{"failures": []}', encoding="utf-8")
    with pytest.raises(FileExistsError, match="--overwrite"):
        S.run([], tmp_path, limit=0, dataset=None, memory_battles=0, root=tmp_path)
    assert (tmp_path / S.REPORT_JSON).read_text(encoding="utf-8") == '{"failures": []}'
    assert S.main(["--out", str(tmp_path), "--limit", "0", "--no-dataset"]) == 1
    assert (tmp_path / S.REPORT_JSON).read_text(encoding="utf-8") == '{"failures": []}'
    # --render-only reads the report and is not a run.
    assert S.main(["--out", str(tmp_path), "--render-only"]) == 0
    assert (tmp_path / S.REPORT_MD).exists()


def test_shadow_replay_on_saved_pages(tmp_path: Path):
    from evaluation import oppmodel_shadow_replay as S

    artifact = ROOT / "results_oppmodel/tables_v1/flags_table.pt"
    pages = [
        p for g in ("ladder_replays_mc*", "challenge_replays_mc*") for p in ROOT.glob(g)
    ]
    if not artifact.is_file() or not pages:
        pytest.skip("needs results_oppmodel/tables_v1 and the saved replay folders")
    lines: list[str] = []
    report = S.run(
        [artifact],
        tmp_path,
        limit=6,
        dataset=None,
        memory_battles=3,
        log=lines.append,
        p99_budget_ms=None,  # timing is not this test's business
    )
    assert report["failures"] == [] and report["pages"]["found"] == 6
    entry = report["artifacts"][0]
    assert entry["equality"]["per_turn"]["compared"] > 20
    # The pass over real poke-env battle objects ran and said the same.
    real = entry["equality"][S.PASS_REAL]
    assert real["compared"] == entry["equality"]["per_turn"]["compared"]
    assert entry["real_battle_notes"]["pages"] == 6
    assert "stream_differs" not in entry["real_battle_notes"]
    with pytest.raises(FileExistsError):
        S.run([artifact], tmp_path, limit=1, dataset=None, memory_battles=0)
    assert [block["mode"] for block in entry["memory"]] == [
        "forget",
        "bounded",
        "unbounded",
    ]
    assert entry["memory"][0]["shadows_kept"] == 0
    assert entry["memory"][2]["shadows_kept"] == 3
    assert json.loads((tmp_path / "report.json").read_text())["failures"] == []
    assert "PASS" in (tmp_path / "report.md").read_text()
    assert S.main(["--out", str(tmp_path), "--render-only"]) == 0
