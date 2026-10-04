#!/usr/bin/env python
"""Train a vgczero policy with PPO self-play against a league of past versions.

Examples
  # laptop smoke run
  python scripts/train.py --run-dir runs/smoke --model tiny --n-envs 64 --rollout-steps 16 --updates 20
  # the real thing (one GPU)
  python scripts/train.py --run-dir runs/base1 --model base --n-envs 2048 --rollout-steps 32
Re-running with the same --run-dir resumes from runs/<name>/checkpoints/latest.pt.
"""

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from vgczero.ppo import TrainConfig, Trainer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in dataclasses.fields(TrainConfig):
        flag = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        else:
            typ = {"int": int, "float": float, "str": str}.get(f.type if isinstance(f.type, str) else f.type.__name__, str)
            ap.add_argument(flag, type=typ, default=f.default)
    ap.add_argument("--updates", type=int, default=None, help="stop after this many updates (default: total_updates)")
    args = ap.parse_args()
    cfg = TrainConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(TrainConfig)})
    Trainer(cfg).train(args.updates)


if __name__ == "__main__":
    main()
