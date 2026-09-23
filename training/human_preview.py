"""Human-style team preview during training (both sides), sampled from a model
trained on top-player replays; the battle policy learns to play those openings.

Why (2026-09-23): the T6 brain leads Farigiraf + Torkoal every game, and human
pilots of this kind of team vary their four and their leads by opponent
(gankyburner, 19 games: 7 lead pairs, 15-4; Farigiraf + Incineroar 11 times;
dksnnfud: Blastoise + Farigiraf). A preview-only entropy bonus loosened our back
row but never the lead, because the battle policy only knows how to play the
Trick Room line. Here the preview comes from the human-trained model
(data/preview_t6_focus_*.pt) and the battle policy trains on the openings it
picks; at play time the same model chooses the preview (ladder --learned_preview).

Training runs with vgc_bench.train --no_teampreview, so poke-env calls
Player.random_teampreview for both sides and the preview is not part of the
policy's trajectory. This module replaces random_teampreview with a sample from
the model's plan distribution (probability ** (1/temperature)); anything that
fails falls back to poke-env's random order.

It must be active in every process. Training envs run in SubprocVecEnv workers
started by a forkserver, which re-imports the main script; this file is the
main script (it runs vgc_bench.train with runpy and alter_sys=False, so
sys.modules["__main__"] stays this file), and it installs the patch at import
time whenever VGC_HUMAN_PREVIEW_MODEL is set. vgc_bench/ stays byte-identical.

Usage (from the repo root; everything after "--" goes to vgc_bench.train):
  VGC_HUMAN_PREVIEW_MODEL=data/preview_t6_focus_20260923.pt \\
      .venv/bin/python -u training/human_preview.py -- --no_teampreview ...
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import random
import runpy
from collections.abc import Sequence

from poke_env.data import to_id_str

MODEL_ENV = "VGC_HUMAN_PREVIEW_MODEL"
TEMPERATURE_ENV = "VGC_HUMAN_PREVIEW_TEMPERATURE"
LOG_FIRST = 3


def plan_weights(probabilities: Sequence[float], temperature: float) -> list[float]:
    """Sampling weights: probability ** (1 / temperature), normalised."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    raw = [max(p, 0.0) ** (1.0 / temperature) for p in probabilities]
    total = sum(raw)
    if total <= 0:
        return [1.0 / len(raw)] * len(raw)
    return [w / total for w in raw]


def human_order(
    predictor, battle, temperature: float, rng: random.Random
) -> str | None:
    """A sampled human-style '/team ...' order for this battle, or None."""
    from vgc_bench.src.opponent_preview import plan_to_showdown_order

    team = list(battle.team.values())
    ours = [to_id_str(p.base_species) for p in team]
    theirs = [to_id_str(p.base_species) for p in battle.opponent_team.values()]
    if len(ours) != 6 or len(theirs) != 6:
        return None
    plans = predictor.predict_plans(ours, theirs, top_k=90)
    if not plans:
        return None
    weights = plan_weights([p.probability for p in plans], temperature)
    plan = rng.choices(plans, weights=weights, k=1)[0]
    order = plan_to_showdown_order(plan)
    for index in order:
        team[index - 1]._selected_in_teampreview = True
    return "/team " + "".join(str(i) for i in order)


def install(model: Path, temperature: float = 1.0, seed: int | None = None) -> bool:
    """Replace Player.random_teampreview with human-style sampling (idempotent)."""
    from poke_env.player.player import Player

    from vgc_bench.src.opponent_preview import PreviewPredictor

    if getattr(Player.random_teampreview, "_human_preview_model", None):
        return False
    if not model.exists():
        raise FileNotFoundError(model)
    predictor = PreviewPredictor.load(model)
    rng = random.Random(seed if seed is not None else os.getpid())
    original = Player.random_teampreview
    logged = [0]

    def random_teampreview(self, battle) -> str:
        try:
            order = human_order(predictor, battle, temperature, rng)
        except Exception as exc:  # never block a battle on the model
            print(f"human preview fallback: {type(exc).__name__}", file=sys.stderr)
            order = None
        if order is None:
            return original(self, battle)
        if logged[0] < LOG_FIRST:
            logged[0] += 1
            names = [p.base_species for p in battle.team.values()]
            picks = [names[int(c) - 1] for c in order.removeprefix("/team ")]
            print(
                f"human preview pid {os.getpid()}: {picks}", file=sys.stderr, flush=True
            )
        return order

    random_teampreview._human_preview_model = str(model)  # type: ignore[attr-defined]
    Player.random_teampreview = random_teampreview  # type: ignore[method-assign]
    print(
        f"human preview installed pid {os.getpid()} model {model} T={temperature}",
        file=sys.stderr,
        flush=True,
    )
    return True


if os.environ.get(MODEL_ENV):  # every process, including forkserver workers
    install(Path(os.environ[MODEL_ENV]), float(os.environ.get(TEMPERATURE_ENV, "1.0")))


def main() -> None:
    if not os.environ.get(MODEL_ENV):
        raise SystemExit(f"set {MODEL_ENV} to the preview model")
    args = sys.argv[1:]
    if args[:1] == ["--"]:
        args = args[1:]
    if "--no_teampreview" not in args:
        raise SystemExit("run with --no_teampreview so the preview is not learned")
    sys.argv = ["vgc_bench.train", *args]
    # alter_sys=False keeps this file as __main__, so workers re-import it.
    runpy.run_module("vgc_bench.train", run_name="__main__", alter_sys=False)


if __name__ == "__main__":
    main()
