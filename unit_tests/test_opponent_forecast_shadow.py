"""Shadow mode for the opponent predictor (2026-10-05, the user: "turn on shadow
mode"): the deployed bot writes the forecast into the decision audit at every move
decision, and no decision reads it. Nothing it does may stop the bot."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ladder_ourteam import (
    build_parser,
    forecast_status,
    material_config,
    record_run_config,
)
from tools.deployed_config import resolve
from unit_tests.test_deployed_config import _deployment, _digest
from unit_tests.test_run_config_hardening import DEPLOYED_LISTENER_ARGV
from vgc_bench.src.policy_player import PolicyPlayer

ROOT = Path(__file__).resolve().parents[1]
TABLE_ARTIFACT = ROOT / "results_oppmodel/tables_v2/flags_table.pt"


class StubForecast:
    def __init__(self, turn: int) -> None:
        self.turn = turn

    def to_dict(self) -> dict[str, Any]:
        return {"turn": self.turn, "model": "stub", "slots": [None, None]}


class StubRuntime:
    """What the hook calls on vgc_bench.src.oppmodel.runtime.OpponentPredictor."""

    serving = True
    load_failure = None

    def __init__(self, reason: str = "", fail: Exception | None = None) -> None:
        self.reason = reason
        self.fail = fail
        self.sheet_calls: list[tuple[str, bool, Any, Any]] = []
        self.predict_calls = 0
        self.forgotten: list[str] = []

    def note_sheets(self, battle, known_open, opponent_sets=None, own_sets=None):
        self.sheet_calls.append(
            (battle.battle_tag, known_open, opponent_sets, own_sets)
        )
        return True

    def predict_with_reason(self, battle):
        self.predict_calls += 1
        if self.fail is not None:
            raise self.fail
        if self.reason:
            return None, self.reason
        return StubForecast(battle.turn), ""

    def forget(self, battle_tag):
        self.forgotten.append(battle_tag)
        return 1


def bare_player(runtime: Any = None, open_tags: tuple[str, ...] = ()) -> PolicyPlayer:
    """A player without its network or connection: the hook must work on one."""
    player = PolicyPlayer.__new__(PolicyPlayer)
    if runtime is not None:
        player._opponent_forecaster = runtime
    player._open_sheet_battles = set(open_tags)
    return player


def battle(tag: str = "battle-gen9championsvgc2026regmc-1", turn: int = 3, **more):
    fields = {"battle_tag": tag, "turn": turn, "teampreview": False}
    return SimpleNamespace(**(fields | more))


@pytest.fixture
def counts():
    """The class-wide counter, restored afterwards."""
    before = PolicyPlayer.guard_fire_counts.copy()
    PolicyPlayer.guard_fire_counts.clear()
    yield PolicyPlayer.guard_fire_counts
    PolicyPlayer.guard_fire_counts.clear()
    PolicyPlayer.guard_fire_counts.update(before)


# --- the hook --------------------------------------------------------------------


def test_off_by_default(counts: Counter):
    assert bare_player()._opponent_forecast_record(battle()) is None
    assert not counts


def test_nothing_at_team_preview(counts: Counter):
    runtime = StubRuntime()
    record = bare_player(runtime)._opponent_forecast_record(battle(teampreview=True))
    assert record is None and runtime.predict_calls == 0 and not runtime.sheet_calls


def test_a_closed_game_is_told_closed_once_and_every_decision_is_logged(
    counts: Counter,
):
    runtime = StubRuntime()
    player = bare_player(runtime)
    first = player._opponent_forecast_record(battle(turn=1))
    second = player._opponent_forecast_record(battle(turn=2))
    assert first == {"turn": 1, "model": "stub", "slots": [None, None]}
    assert second is not None and second["turn"] == 2
    assert runtime.sheet_calls == [(battle().battle_tag, False, None, None)]
    assert counts["opponent_forecast"] == 2


def test_an_open_game_hands_over_both_sheets_or_neither(
    counts: Counter, monkeypatch: pytest.MonkeyPatch
):
    """One sheet alone is a state no training game has; and our own sheet is
    public only when the game is open-sheet."""
    from vgc_bench.src.oppmodel import runtime as R

    tag = "battle-gen9championsvgc2026regmc-2"
    monkeypatch.setattr(R, "sheet_sets_from_team", lambda team: list(team))
    both = StubRuntime()
    player = bare_player(both, open_tags=(tag,))
    player._opponent_forecast_record(
        battle(tag, team=["ours"], opponent_team=["theirs"])
    )
    assert both.sheet_calls == [(tag, True, ["theirs"], ["ours"])]
    unread = StubRuntime()
    player = bare_player(unread, open_tags=(tag,))
    record = player._opponent_forecast_record(
        battle(tag, team=["ours"], opponent_team=[])
    )
    assert unread.sheet_calls == [] and record is not None
    assert counts["opponent_forecast_sheets_unread"] == 1


def test_a_stand_down_is_logged_by_name(counts: Counter):
    runtime = StubRuntime(reason="stand_down:mid_turn")
    record = bare_player(runtime)._opponent_forecast_record(battle())
    assert record == {"stand_down": "stand_down:mid_turn"}
    assert counts["opponent_forecast:stand_down:mid_turn"] == 1
    assert counts["opponent_forecast"] == 0


def test_a_failing_predictor_costs_nothing(counts: Counter):
    runtime = StubRuntime(fail=RuntimeError("boom"))
    assert bare_player(runtime)._opponent_forecast_record(battle()) is None
    assert counts["opponent_forecast_failed:RuntimeError"] == 1
    # a battle object the hook cannot even read
    assert bare_player(StubRuntime())._opponent_forecast_record(object()) is None
    assert counts["opponent_forecast_failed:AttributeError"] == 1


def test_a_finished_battle_is_forgotten(counts: Counter):
    runtime = StubRuntime()
    player = bare_player(runtime)
    tag = battle().battle_tag
    player._opponent_forecast_record(battle())
    assert tag in player._forecast_sheets_noted
    player._forget_opponent_forecast(tag)
    assert runtime.forgotten == [tag] and tag not in player._forecast_sheets_noted
    # off, or a runtime whose forget raises: still nothing reaches the caller
    bare_player()._forget_opponent_forecast(tag)
    runtime.forget = lambda battle_tag: 1 / 0  # type: ignore[method-assign]
    player._forget_opponent_forecast(tag)
    assert counts["opponent_forecast_failed:ZeroDivisionError"] == 1


def test_the_audit_carries_the_forecast_only_when_there_is_one(
    tmp_path: Path, counts: Counter
):
    player = bare_player()
    player.decision_log_path = tmp_path / "decisions.jsonl"
    record = {"turn": 3, "model": "stub", "slots": [None, None]}
    player._audit_decision(battle(), [], None, None, None, record)
    player._audit_decision(battle(turn=4), [], None)
    rows = [
        json.loads(line)
        for line in player.decision_log_path.read_text().split("\n")
        if line
    ]
    assert rows[0]["opponent_forecast"] == record and rows[0]["turn"] == 3
    assert "opponent_forecast" not in rows[1]
    assert not [name for name in counts if "audit" in name]


def test_no_decision_reads_the_forecast():
    """Shadow mode is a promise about the code: the record is made, passed to the
    audit and written. A future consumer has to change this test on purpose."""
    source = (ROOT / "vgc_bench/src/policy_player.py").read_text()
    uses = re.findall(r"\bforecast_record\b", source)
    # made in _guarded_action, passed to the audit, the audit's parameter, its
    # None check and the payload line
    assert len(uses) == 5, uses
    assert source.count('payload["opponent_forecast"] = forecast_record') == 1
    assert source.count("self._opponent_forecast_record(battle)") == 1
    for module in sorted((ROOT / "vgc_bench/src").glob("*.py")):
        if module.name == "policy_player.py":
            continue
        text = module.read_text()
        assert "opponent_forecast" not in text, module.name
        assert "_opponent_forecaster" not in text, module.name
    # 2026-10-09, changed on purpose (the search session): there is now ONE consumer,
    # the search's reply prior, behind PolicyPlayer(exact_reply_forecast=True) -- off
    # by default (unit_tests/test_search_reply_forecast.py). It asks the predictor
    # itself; the shadow record above is still read by nothing, and with the option
    # off the search is never handed a forecast.
    assert source.count("runtime.predict_with_reason(battle)") == 2
    assert source.count('if not getattr(self, "exact_reply_forecast", False):') == 1
    assert source.count('if getattr(self, "exact_reply_forecast", False):') == 1
    assert source.count("session.set_reply_forecast(") == 1


# --- the launcher ----------------------------------------------------------------


def test_the_flag_is_not_part_of_the_play_configuration(tmp_path: Path):
    """It changes no decision, so a replay directory keeps accepting runs with and
    without it; each run records which artifact it logged with."""
    artifact = tmp_path / "forecast.pt"
    artifact.write_bytes(b"weights")
    off = build_parser().parse_args(DEPLOYED_LISTENER_ARGV)
    on = build_parser().parse_args(
        DEPLOYED_LISTENER_ARGV + ["--opponent-forecast", str(artifact)]
    )
    assert off.opponent_forecast == "" and on.opponent_forecast == str(artifact)
    assert material_config(off, "sha", "hard") == material_config(on, "sha", "hard")
    assert "opponent_forecast" not in material_config(on, "sha", "hard")
    replay_dir = tmp_path / "replays"
    record_run_config(replay_dir, off, "sha", "hard")
    record_run_config(replay_dir, on, "sha", "hard")
    gone = build_parser().parse_args(
        DEPLOYED_LISTENER_ARGV + ["--opponent-forecast", str(tmp_path / "gone.pt")]
    )
    path = record_run_config(replay_dir, gone, "sha", "hard")
    runs = json.loads(path.read_text())["runs"]
    assert "opponent_forecast" not in runs[0]
    assert runs[1]["opponent_forecast"] == {
        "path": str(artifact),
        "sha256": _digest(artifact),
    }
    assert runs[2]["opponent_forecast"]["sha256"] == "missing"
    assert runs[0]["material"] == runs[1]["material"] == runs[2]["material"]


def test_the_launch_line_says_whether_shadow_mode_serves():
    assert forecast_status(SimpleNamespace(), "") == "off"
    serving = SimpleNamespace(_opponent_forecaster=StubRuntime())
    assert "shadow mode on" in forecast_status(serving, "a.pt")
    broken = StubRuntime()
    broken.serving = False  # type: ignore[misc]
    broken.load_failure = "OSError: no such file"  # type: ignore[assignment]
    line = forecast_status(SimpleNamespace(_opponent_forecaster=broken), "a.pt")
    assert "NOT SERVING" in line and "no such file" in line
    assert "NOT SERVING" in forecast_status(SimpleNamespace(), "a.pt")


# --- the manifest and the launch scripts -------------------------------------------


def test_the_forecast_is_off_unless_deployed_and_verified(tmp_path: Path):
    assert resolve(_deployment(tmp_path), tmp_path)["FORECAST"] == ""
    artifact = tmp_path / "results_deployed/opponent_forecast.pt"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"weights")
    name = "results_deployed/opponent_forecast.pt"
    on = _deployment(
        tmp_path, opponent_forecast=name, opponent_forecast_sha256=_digest(artifact)
    )
    assert resolve(on, tmp_path)["FORECAST"] == name
    wrong = _deployment(
        tmp_path, opponent_forecast=name, opponent_forecast_sha256="0" * 64
    )
    with pytest.raises(ValueError, match="opponent_forecast"):
        resolve(wrong, tmp_path)
    with pytest.raises(KeyError):
        resolve(_deployment(tmp_path, opponent_forecast=name), tmp_path)
    artifact.unlink()
    with pytest.raises(ValueError, match="missing"):
        resolve(on, tmp_path)


@pytest.mark.parametrize(
    "script", ["tools/ladder_read_loop.sh", "tools/challenges_deployed.sh"]
)
def test_launchers_pass_the_forecast(script: str):
    text = (ROOT / script).read_text()
    assert '--opponent-forecast "$FORECAST"' in text


@pytest.mark.parametrize("name", ["ladder_deployed.sh", "ladder_trial.sh"])
def test_the_ladder_wrappers_export_the_forecast(name: str):
    assert re.search(r"export\b.*\bFORECAST\b", (ROOT / "tools" / name).read_text())


def test_a_ladder_session_refuses_while_the_roster_search_plays():
    """The search session's harness is latency-sensitive (2026-10-05)."""
    text = (ROOT / "tools/ladder_read_loop.sh").read_text()
    line = next(s for s in text.splitlines() if s.startswith("HEAVY="))
    heavy = re.compile(line.removeprefix("HEAVY='").removesuffix("'"))
    assert heavy.search(".venv/bin/python -u evaluation/search_roster_ab.py --shard 1")
    assert heavy.search(".venv/bin/python evaluation/mirror_guard_ab.py --guard x")


# --- end to end on saved games -------------------------------------------------------


def test_the_hook_on_real_battle_objects_logs_what_the_predictor_says(counts: Counter):
    """Saved own pages fed through real poke-env battles: at every turn line the
    record is the runtime's own forecast, JSON-ready and small."""
    pytest.importorskip("poke_env")
    if not TABLE_ARTIFACT.is_file():
        pytest.skip("no fitted table artifact (results_oppmodel is git-ignored)")
    from poke_env.battle import DoubleBattle

    from evaluation.oppmodel_shadow_replay import bot_username, load_pages
    from vgc_bench.src import pokeenv_patches
    from vgc_bench.src.oppmodel.runtime import OpponentPredictor

    pages, _, _ = load_pages(ROOT, limit=3)
    if not pages:
        pytest.skip("no saved own pages")
    pokeenv_patches.install()
    quiet = logging.getLogger("test_opponent_forecast_shadow")
    quiet.addHandler(logging.NullHandler())
    quiet.propagate = False
    player = bare_player(OpponentPredictor.load(TABLE_ARTIFACT))
    reference = OpponentPredictor.load(TABLE_ARTIFACT)
    assert player._opponent_forecaster.serving
    compared = 0
    for page in pages:
        name = bot_username(page)
        if name is None:
            continue
        live: Any = DoubleBattle(page.tag, name, quiet, gen=9)
        for event in page.events:
            kind = event[1] if len(event) > 1 else ""
            try:
                if kind == "win" and len(event) > 2:
                    live.won_by(event[2])
                elif kind == "tie":
                    live.tied()
                else:
                    live.parse_message(list(event))
            except Exception:
                pass  # poke-env's own parse errors do not stop the stream
            if kind != "turn":
                continue
            record = player._opponent_forecast_record(live)
            reference.note_sheets(live, False)
            expected = reference.predict(live)
            if expected is None:
                assert record is not None and "stand_down" in record
                continue
            assert record is not None and "stand_down" not in record
            want = expected.to_dict()
            assert {k: v for k, v in record.items() if k != "latency_ms"} == {
                k: v for k, v in want.items() if k != "latency_ms"
            }
            assert len(json.dumps(record)) < 4096
            compared += 1
        player._forget_opponent_forecast(page.tag)
        reference.forget(page.tag)
    assert compared >= 3
    assert not [name for name in counts if "failed" in name]
    assert player._opponent_forecaster.n_battles == 0
