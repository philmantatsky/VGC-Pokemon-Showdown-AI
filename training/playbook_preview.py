"""Practise the playbook: our team previews by its plan cards during training.

Why (2026-09-27): our own playbook (data/playbook_t6.json, vgc_bench/src/playbook.py)
picks a plan card per opponent -- our four, our leads, each Pokemon's job. Played
by the deployed brain it LOST locally (battery -8.03pp; Support Room -14.6, Sun Room
-5.2, Water Room -2.7), while the turn-1 script alone was about neutral: the loss
is the openings themselves, which the brain never practised (it trained with the
human-model previews of training/human_preview.py). The user: "start the practice
cycle". Here the side playing the playbook's team takes its preview from the
playbook -- the card the rules pick for this opponent, or with probability
VGC_PLAYBOOK_EXPLORE another non-experimental card, so every opening is practised
-- and every other side keeps the human-model preview (training/human_preview.py,
installed first; this patch wraps it).

Like human_preview.py this is the main script (runpy, alter_sys=False) and installs
the patch at import time in every process, forkserver workers included; the
forkserver must preload nothing. vgc_bench/ stays byte-identical. Anything that
fails falls back to the wrapped preview.

Usage (from the repo root; everything after "--" goes to vgc_bench.train):
  VGC_HUMAN_PREVIEW_MODEL=data/preview_t6_focus_20260923.pt \\
  VGC_PLAYBOOK=data/playbook_t6.json VGC_PLAYBOOK_EXPLORE=0.15 \\
      .venv/bin/python -u training/playbook_preview.py -- --no_teampreview ...
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import os
import random
import runpy
from collections import Counter
from collections.abc import Sequence

from poke_env.data import to_id_str

from training import human_preview  # installs the human-model preview first

PLAYBOOK_ENV = "VGC_PLAYBOOK"
EXPLORE_ENV = "VGC_PLAYBOOK_EXPLORE"
LOG_FIRST = 3
LOG_EVERY = 500


def team_species(path: Path) -> frozenset[str]:
    """The six species of the team a playbook is written for."""
    from poke_env.teambuilder import Teambuilder

    team = Teambuilder.parse_showdown_team((ROOT / path).read_text())
    return frozenset(to_id_str(mon.species or mon.nickname) for mon in team)


def open_sheets(battle) -> dict:
    """Their open team sheet (ability + moves known at preview), else {}."""
    from vgc_bench.src.playbook import OpenSet

    sheets = {}
    for mon in battle.opponent_team.values():
        moves = tuple(getattr(mon, "moves", {}) or {})
        if mon.ability and moves:
            sheets[to_id_str(mon.base_species)] = OpenSet(
                to_id_str(mon.ability), to_id_str(mon.item or ""), moves
            )
    return sheets


def card_order(card: dict, feats: dict, ours: Sequence[str]) -> list[int]:
    """1-based /team order (lead, lead, back, back) for one card, with its
    back_if swaps applied to these opponent features."""
    from vgc_bench.src.playbook import rule_matches

    back = list(card["back"])
    for alt in card.get("back_if", []):
        if rule_matches(alt, feats):
            back = list(alt["back"])
            break
    index = {species: i for i, species in enumerate(ours)}
    return [index[s] + 1 for s in (*card["lead"], *back)]


def playbook_order(
    playbook, battle, explore: float, rng: random.Random, formatid: str | None = None
) -> tuple[str, str] | None:
    """(card name, '/team ...') for the playbook's team, else None. The set data
    follow the battle's format (the player's own when the battle has none yet:
    without it the planner would read the old regulation's data)."""
    from vgc_bench.src.opponent_preview import plan_to_showdown_order
    from vgc_bench.src.playbook import opponent_features

    team = list(battle.team.values())
    ours = [to_id_str(p.base_species) for p in team]
    theirs = [to_id_str(p.base_species) for p in battle.opponent_team.values()]
    if len(ours) != 6 or len(theirs) != 6:
        return None
    sheets = open_sheets(battle)
    formatid = getattr(battle, "format", None) or formatid
    choice = playbook.choose(ours, theirs, sheets, formatid)
    card, order = choice.card, plan_to_showdown_order(choice.plan)
    others = [
        c for c in playbook.cards if c["name"] != card and not c.get("experimental")
    ]
    if others and rng.random() < explore:
        other = rng.choice(others)
        feats = opponent_features(theirs, sheets, formatid)
        card, order = other["name"] + "*", card_order(other, feats, ours)
    for index in order:
        team[index - 1]._selected_in_teampreview = True
    return card, "/team " + "".join(str(i) for i in order)


def install(path: Path, explore: float, seed: int | None = None) -> bool:
    """Wrap Player.random_teampreview: the playbook for its own team (idempotent)."""
    from poke_env.player.player import Player

    from vgc_bench.src.playbook import Playbook

    if getattr(Player.random_teampreview, "_playbook_preview", None):
        return False
    if not 0.0 <= explore <= 1.0:
        raise ValueError("VGC_PLAYBOOK_EXPLORE must be within [0, 1]")
    playbook = Playbook(ROOT / path)
    ours = team_species(Path(playbook.team))
    rng = random.Random(seed if seed is not None else os.getpid() + 7)
    wrapped = Player.random_teampreview
    counts: Counter[str] = Counter()

    def random_teampreview(self, battle) -> str:
        species = frozenset(to_id_str(p.base_species) for p in battle.team.values())
        if species != ours:
            return wrapped(self, battle)
        try:
            picked = playbook_order(
                playbook, battle, explore, rng, getattr(self, "format", None)
            )
        except Exception as exc:  # never block a battle on the planner
            counts[f"error:{type(exc).__name__}"] += 1
            picked = None
        if picked is None:
            counts["fallback"] += 1
            return wrapped(self, battle)
        card, order = picked
        counts[card] += 1
        total = sum(counts.values())
        if total <= LOG_FIRST or total % LOG_EVERY == 0:
            print(
                f"playbook preview pid {os.getpid()}: {card} {order} "
                f"counts {json.dumps(dict(counts), sort_keys=True)}",
                file=sys.stderr,
                flush=True,
            )
        return order

    random_teampreview._playbook_preview = str(path)  # type: ignore[attr-defined]
    random_teampreview._playbook_counts = counts  # type: ignore[attr-defined]
    Player.random_teampreview = random_teampreview  # type: ignore[method-assign]
    print(
        f"playbook preview installed pid {os.getpid()} playbook {path} "
        f"explore={explore} team={sorted(ours)}",
        file=sys.stderr,
        flush=True,
    )
    return True


if os.environ.get(PLAYBOOK_ENV):  # every process, including forkserver workers
    install(Path(os.environ[PLAYBOOK_ENV]), float(os.environ.get(EXPLORE_ENV, "0.15")))


def main() -> None:
    if not os.environ.get(PLAYBOOK_ENV) or not os.environ.get(human_preview.MODEL_ENV):
        raise SystemExit(f"set {PLAYBOOK_ENV} and {human_preview.MODEL_ENV}")
    args = sys.argv[1:]
    if args[:1] == ["--"]:
        args = args[1:]
    if "--no_teampreview" not in args:
        raise SystemExit("run with --no_teampreview so the preview is not learned")
    sys.argv = ["vgc_bench.train", *args]
    human_preview.prepare_workers()  # before vgc_bench.train starts the forkserver
    # alter_sys=False keeps this file as __main__, so workers re-import it.
    runpy.run_module("vgc_bench.train", run_name="__main__", alter_sys=False)


if __name__ == "__main__":
    main()
