"""Run evaluation/opening_study.py with OUR team preview chosen by a learned,
human-trained preview model (vgc_bench.src.opponent_preview.PreviewPredictor).

Why (2026-09-23): the T6 brain leads Farigiraf + Torkoal every game; human pilots
of this kind of team vary their four and their leads by opponent (gankyburner:
7 lead pairs in 19 games, 15-4). data/preview_t6_focus_*.pt learned those
choices from replays. This wrapper lets the study's own player (StudyPlayer)
use that model for its preview and nothing else:

* only StudyPlayer is patched; the opponent's preview stays its own policy's;
* the model's belief about the OPPONENT's plan is discarded after preview, so
  the in-battle behaviour is exactly the reference arm's (which ran without a
  preview model); only the opening differs.

It patches at runtime so evaluation/opening_study.py and vgc_bench/ stay
byte-identical (the pinned reference arms of results_t6_vs_deployed_v1 remain
valid). Usage (from the repo root):

  .venv/bin/python evaluation/learned_preview_study.py --preview-model <model.pt> \\
      [--extra-guards name,...] -- <opening_study arguments>

--playbook (2026-09-27) makes OUR four and leads come from our own playbook
(vgc_bench/src/playbook.py); the preview model then only predicts THEIR plan.
With --playbook-script-only (2026-10-03) the model keeps choosing our four and
leads, and the chosen card supplies only its turn-1 script.

--sheet-preview (2026-10-03) lets the opponent's open team sheet choose among the
model's top plans for our player (PolicyPlayer sheet_preview); hidden-sheet games
are unchanged.

--extra-guards (2026-09-24) also turns on opt-in guards for OUR player only
(a per-player override; the opponents keep the study's class-level set). The
arm manifest records the class-level flags, so the harness records the extras.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse

from evaluation import opening_study
from vgc_bench.src.guards import GUARDS, HARD_GUARDS
from vgc_bench.src.policy_player import PolicyPlayer


def install(model: Path) -> None:
    """StudyPlayer previews with ``model``; the opponent belief is dropped."""
    if not model.exists():
        raise FileNotFoundError(model)
    if getattr(opening_study.StudyPlayer, "_learned_preview_model", None):
        raise RuntimeError("learned preview already installed")
    original_init = opening_study.StudyPlayer.__init__
    original_learned = PolicyPlayer._learned_teampreview

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.preview_model_path = model
        self.use_learned_teampreview = True

    def _learned_teampreview(self, battle):
        choice = original_learned(self, battle)
        if isinstance(self, opening_study.StudyPlayer):
            # Keep only our opening: in-battle code sees no plan state, exactly as
            # in the reference arm, so the arms differ in the preview alone.
            self._battle_plans.pop(battle.battle_tag, None)
        return choice

    opening_study.StudyPlayer.__init__ = __init__  # type: ignore[method-assign]
    opening_study.StudyPlayer._learned_preview_model = str(model)  # type: ignore[attr-defined]
    PolicyPlayer._learned_teampreview = _learned_teampreview  # type: ignore[method-assign]


def enable_guards(names: list[str]) -> None:
    """Opt-in guards for OUR player only, on top of the study's own set.

    opening_study sets PolicyPlayer.guard_flags on the class, which the PPO
    opponents share; a per-player override keeps the opponents exactly as in the
    reference arm.
    """
    invalid = [name for name in names if name not in GUARDS or name in HARD_GUARDS]
    if invalid:
        raise ValueError(f"not opt-in guards: {invalid}")
    original_init = opening_study.StudyPlayer.__init__
    extra = {name: True for name in names}

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.guard_overrides = {
            **(getattr(self, "guard_overrides", None) or {}),
            **extra,
        }

    opening_study.StudyPlayer.__init__ = __init__  # type: ignore[method-assign]


def enable_playbook(path: Path, script_only: bool = False) -> None:
    """OUR preview from our own playbook (vgc_bench/src/playbook.py); the preview
    model installed above keeps predicting the opponent's plan. ``script_only``
    keeps the model's own four and leads and takes only the card's turn-1 script
    (PolicyPlayer playbook_script_only)."""
    if not path.exists():
        raise FileNotFoundError(path)
    original_init = opening_study.StudyPlayer.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.playbook_path = path
        self.playbook_script_only = script_only
        self._playbook = None

    opening_study.StudyPlayer.__init__ = __init__  # type: ignore[method-assign]


def enable_sheet_preview() -> None:
    """Our player reads the opponent's open sheet at preview (sheet_preview.py)."""
    original_init = opening_study.StudyPlayer.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.sheet_preview = True

    opening_study.StudyPlayer.__init__ = __init__  # type: ignore[method-assign]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preview-model", type=Path, required=True)
    ap.add_argument(
        "--playbook", type=Path, default=None, help="our preview from this playbook"
    )
    ap.add_argument(
        "--playbook-script-only",
        action="store_true",
        help="with --playbook: the model's own preview, the card's turn-1 script only",
    )
    ap.add_argument("--extra-guards", default="", help="comma-separated opt-in guards")
    ap.add_argument(
        "--sheet-preview",
        action="store_true",
        help="our preview reads the opponent's open team sheet",
    )
    ap.add_argument("study_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if args.playbook_script_only and args.playbook is None:
        ap.error("--playbook-script-only needs --playbook")
    rest = args.study_args[1:] if args.study_args[:1] == ["--"] else args.study_args
    install(args.preview_model)
    extra = [name for name in args.extra_guards.split(",") if name]
    if extra:
        enable_guards(extra)
        print(f"extra guards for our side: {', '.join(extra)}", flush=True)
    if args.playbook is not None:
        enable_playbook(args.playbook, args.playbook_script_only)
        if args.playbook_script_only:
            print(f"playbook turn-1 script for our side: {args.playbook}", flush=True)
        else:
            print(f"playbook preview for our side: {args.playbook}", flush=True)
    if args.sheet_preview:
        enable_sheet_preview()
        print("open-sheet preview for our side", flush=True)
    print(f"learned preview for our side: {args.preview_model}", flush=True)
    sys.argv = ["evaluation/opening_study.py", *rest]
    opening_study.main()


if __name__ == "__main__":
    main()
