"""Reproduce the priority_block guard's exceptions (2026-10-04: 7 in the 22,116
practice decisions of results_tactical_t6e_ep1, counted as priority_block_error but
never logged).

Plays the deployed setup against the training-role opponents exactly as
training/gen_tactical_data.py does (its main() runs, into a scratch output dir),
with the guard wrapped to write each exception's traceback, the position and the
candidates to priority_block_errors.txt next to this script, then re-raise so
apply_guards counts it as before. Nothing in the bot is changed.
Run from the repo root (a Showdown server must listen on --port):
    .venv/bin/python results_analysis/guard_errors_20261004/priority_block_repro.py \\
        --games-per-cell 125 --port 7631 --output <scratch dir>
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vgc_bench.src import guards as G  # noqa: E402

LOG = Path(__file__).with_name("priority_block_errors.txt")


def describe(battle, cands) -> str:
    lines = [f"=== {battle.battle_tag} turn {battle.turn}"]
    for side, mons in (
        ("ours", battle.active_pokemon),
        ("foes", battle.opponent_active_pokemon),
    ):
        lines.append(
            f"{side}: "
            + ", ".join(
                "-"
                if m is None
                else f"{m.species}{' (fainted)' if m.fainted else ''} {m.ability}"
                for m in mons
            )
        )
    lines.append(f"fields={dict(battle.fields)} weather={dict(battle.weather)}")
    for c in cands[:8]:
        decoded = []
        for pos, action in enumerate(c.actions):
            try:
                decoded.append(str(G._decode(battle, action, pos)))
            except Exception as exc:
                decoded.append(f"<decode {type(exc).__name__}>")
        lines.append(f"cand {tuple(c.actions)} p={c.prob:.3f} -> {decoded}")
    return "\n".join(lines)


def wrap() -> None:
    original = G.GUARDS["priority_block"]

    def logged(battle, cands, report):
        try:
            return original(battle, cands, report)
        except Exception:
            with LOG.open("a") as f:
                f.write(describe(battle, cands) + "\n" + traceback.format_exc() + "\n")
            raise

    G.GUARDS["priority_block"] = logged


if __name__ == "__main__":
    wrap()
    from training import gen_tactical_data

    gen_tactical_data.main()
