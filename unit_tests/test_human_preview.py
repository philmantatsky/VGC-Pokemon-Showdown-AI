"""training/human_preview.py: temperature weights, the sampled '/team' order and
its selection flags, the incomplete-roster fallback, and idempotent install."""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from training import human_preview
from vgc_bench.src.opponent_preview import PreviewPlan


def test_plan_weights_follow_the_temperature() -> None:
    assert human_preview.plan_weights([0.75, 0.25], 1.0) == pytest.approx([0.75, 0.25])
    sharp = human_preview.plan_weights([0.75, 0.25], 0.5)  # squares, renormalised
    assert sharp == pytest.approx([0.9, 0.1])
    assert human_preview.plan_weights([0.0, 0.0], 1.0) == [0.5, 0.5]
    with pytest.raises(ValueError):
        human_preview.plan_weights([1.0], 0.0)


class _Mon:
    def __init__(self, species: str) -> None:
        self.base_species = species
        self._selected_in_teampreview = False


def _battle(ours: list[str], theirs: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        team={f"p1: {s}": _Mon(s) for s in ours},
        opponent_team={f"p2: {s}": _Mon(s) for s in theirs},
    )


class _Predictor:
    def __init__(self, plans: list[PreviewPlan]) -> None:
        self.plans = plans
        self.calls: list[tuple[list[str], list[str]]] = []

    def predict_plans(self, ours, theirs, top_k=90):
        self.calls.append((list(ours), list(theirs)))
        return self.plans


SIX = ["Blastoise", "Farigiraf", "Charizard", "Venusaur", "Torkoal", "Incineroar"]
THEM = ["Tyranitar", "Excadrill", "Rillaboom", "Sneasler", "Salamence", "Milotic"]


def test_human_order_returns_leads_then_back_and_flags_the_four() -> None:
    plan = PreviewPlan(lead_indices=(0, 1), bring_indices=(0, 1, 4, 5), probability=1.0)
    predictor = _Predictor([plan])
    battle = _battle(SIX, THEM)
    order = human_preview.human_order(predictor, battle, 1.0, random.Random(0))
    assert order == "/team 1256"  # Blastoise, Farigiraf lead; Torkoal, Incineroar back
    assert (
        predictor.calls[0][0][0] == "blastoise"
        and predictor.calls[0][1][0] == "tyranitar"
    )
    flagged = [
        m.base_species for m in battle.team.values() if m._selected_in_teampreview
    ]
    assert flagged == ["Blastoise", "Farigiraf", "Torkoal", "Incineroar"]


def test_human_order_samples_by_probability_and_refuses_incomplete_rosters() -> None:
    a = PreviewPlan(lead_indices=(0, 1), bring_indices=(0, 1, 2, 3), probability=0.9)
    b = PreviewPlan(lead_indices=(2, 3), bring_indices=(2, 3, 4, 5), probability=0.1)
    predictor = _Predictor([a, b])
    rng = random.Random(1)
    picks = []
    for _ in range(400):
        order = human_preview.human_order(predictor, _battle(SIX, THEM), 1.0, rng)
        assert order is not None
        picks.append(order[6:8])
    assert 0.8 < picks.count("12") / 400 < 0.97
    assert (
        human_preview.human_order(predictor, _battle(SIX[:5], THEM), 1.0, rng) is None
    )


def test_install_is_idempotent_and_refuses_missing_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from poke_env.player.player import Player

    monkeypatch.setattr(Player, "random_teampreview", Player.random_teampreview)
    monkeypatch.setattr(
        "vgc_bench.src.opponent_preview.PreviewPredictor.load",
        classmethod(lambda cls, path, device="cpu": _Predictor([])),
    )
    with pytest.raises(FileNotFoundError):
        human_preview.install(tmp_path / "missing.pt")
    model = tmp_path / "preview.pt"
    model.write_bytes(b"x")
    assert human_preview.install(model) is True
    assert human_preview.install(model) is False  # already installed
    fallback: Any = Player.random_teampreview  # empty plans -> random order
    battle = _battle(SIX, THEM)
    battle.format = "gen9championsvgc2026regmc"
    order = fallback(SimpleNamespace(), battle)
    assert order.startswith("/team ") and len(order) == len("/team ") + 4


FORK_SCRIPT = """
import json, multiprocessing, os, sys
sys.path.insert(0, {root!r})
os.environ["VGC_HUMAN_PREVIEW_MODEL"] = {model!r}
from training import human_preview  # installs here, as the real launcher does


def child(queue):
    import asyncio
    import threading

    import poke_env.concurrency as concurrency
    from poke_env.player.player import Player

    alive = concurrency._t.is_alive()
    ran = None
    if alive:
        job = asyncio.run_coroutine_threadsafe(
            asyncio.sleep(0, result="ran"), concurrency.POKE_LOOP
        )
        ran = job.result(timeout=20)
    marker = getattr(Player.random_teampreview, "_human_preview_model", None)
    queue.put({{
        "pid": os.getpid(),
        "patched": bool(marker),
        "loop_alive": alive,
        "ran": ran,
    }})


if __name__ == "__main__":
    if sys.argv[1] == "clean":
        human_preview.prepare_workers()
    ctx = multiprocessing.get_context("forkserver")
    queue = ctx.Queue()
    workers = [ctx.Process(target=child, args=(queue,)) for _ in range(2)]
    for worker in workers:
        worker.start()
    results = [queue.get(timeout=120) for _ in workers]
    for worker in workers:
        worker.join(timeout=30)
    print(json.dumps(results))
"""


def _fork_workers(tmp_path: Path, mode: str) -> list[dict]:
    import json
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    script = tmp_path / "launcher.py"  # path-run __main__, like human_preview.py
    script.write_text(
        FORK_SCRIPT.format(
            root=str(root), model=str(root / "data/preview_t6_focus_20260923.pt")
        )
    )
    done = subprocess.run(
        [sys.executable, str(script), mode],
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_forkserver_workers_install_in_their_own_clean_process(tmp_path: Path) -> None:
    """2026-09-24: preloading the launcher in the forkserver imported poke-env and
    torch there; every forked worker got a dead event loop and macOS killed them.
    With prepare_workers() each worker installs the patch itself and its loop runs."""
    clean = _fork_workers(tmp_path, "clean")
    assert len({w["pid"] for w in clean}) == 2
    assert all(w["patched"] and w["loop_alive"] and w["ran"] == "ran" for w in clean)
    # The negative control: the old way leaves every worker's event loop dead.
    dirty = _fork_workers(tmp_path, "dirty")
    assert all(w["patched"] and not w["loop_alive"] for w in dirty)


def test_main_empties_the_forkserver_preload_before_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import multiprocessing.forkserver as forkserver

    seen: dict[str, Any] = {}
    server: Any = forkserver._forkserver  # private stdlib state, untyped
    monkeypatch.setattr(server, "_preload_modules", ["__main__"], raising=True)
    monkeypatch.setenv(human_preview.MODEL_ENV, "unused.pt")
    monkeypatch.setattr(
        "sys.argv", ["human_preview.py", "--", "--no_teampreview", "--reg", "mc"]
    )
    monkeypatch.setattr(
        human_preview.runpy,
        "run_module",
        lambda name, **kw: seen.update(
            name=name, preload=list(server._preload_modules), **kw
        ),
    )
    human_preview.main()
    assert seen["name"] == "vgc_bench.train" and seen["alter_sys"] is False
    assert seen["preload"] == []
