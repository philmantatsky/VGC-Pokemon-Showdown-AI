"""Engine loading and static data shared by the model, trainer and search."""

from __future__ import annotations

import functools
import os
from pathlib import Path

import numpy as np

import vgczero_engine as E

ROOT = Path(__file__).resolve().parents[2]
DEX = ROOT / "data" / "dex_regmc.json.gz"
TEAMS = ROOT / "data" / "teams_regmc.json.gz"

# Observation layout (must match engine/src/obs.rs).
I_SPECIES, I_ABILITY, I_ITEM, I_MOVE0, I_LAST_MOVE, I_STATUS, I_POSITION, I_OWNER = 0, 1, 2, 3, 7, 8, 9, 10
POS_ACTIVE0, POS_ACTIVE1 = 1, 2
PASS = 0
N_SLOT_ACTIONS = 31
N_PREVIEW = 90
REQ_WAIT, REQ_PREVIEW, REQ_MOVE, REQ_SWITCH = 0, 1, 2, 3


def write_text_atomic(path, text: str) -> None:
    """Write through a temp file and a rename, so a killed process never leaves a half-written file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


@functools.lru_cache(maxsize=1)
def load() -> int:
    """Load dex and team pool into the engine (idempotent). Returns #teams."""
    return E.load(str(DEX), str(TEAMS))


@functools.lru_cache(maxsize=1)
def static_tables() -> dict[str, np.ndarray]:
    load()
    sp, mv, it = E.static_tables()
    return {"species": np.asarray(sp), "moves": np.asarray(mv), "items": np.asarray(it)}


@functools.lru_cache(maxsize=1)
def preview_options() -> np.ndarray:
    load()
    return np.asarray(E.preview_options(), dtype=np.int64)  # [90, 4]


def supported_teams() -> list[int]:
    load()
    return [i for i, ok in enumerate(E.team_supported()) if ok]


def team_names() -> list[str]:
    load()
    return E.team_names()


# Slot action decoding helpers (actions.rs).
def is_switch(a: np.ndarray | int):
    return (a >= 1) & (a <= 6)


def is_mega(a: np.ndarray | int):
    return (a >= 7) & (((a - 7) % 2) == 1)
