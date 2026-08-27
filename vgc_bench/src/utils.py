"""
Utility module for VGC-Bench.

Contains shared constants, enums, and helper functions used throughout the
codebase. Defines observation space dimensions, loads Pokemon game data,
and provides training configuration utilities.
"""

import json
import os
import random
from enum import Enum, auto, unique

import numpy as np
import torch
from poke_env.battle import (
    Effect,
    Field,
    MoveCategory,
    PokemonGender,
    PokemonType,
    SideCondition,
    Status,
    Target,
    Weather,
)


@unique
class LearningStyle(Enum):
    """
    Training paradigm options for reinforcement learning.

    Defines different self-play and opponent sampling strategies used
    during PPO training for Pokemon VGC agents.

    Values:
        EXPLOITER: Train against a fixed opponent policy.
        PURE_SELF_PLAY: Train against current policy (both players identical).
        FICTITIOUS_PLAY: Sample historical checkpoints uniformly as opponents.
        DOUBLE_ORACLE: Sample checkpoints based on Nash equilibrium distribution.
    """

    EXPLOITER = auto()
    PURE_SELF_PLAY = auto()
    FICTITIOUS_PLAY = auto()
    DOUBLE_ORACLE = auto()

    @property
    def is_self_play(self) -> bool:
        """Check if this style involves any form of self-play training."""
        return self in {
            LearningStyle.PURE_SELF_PLAY,
            LearningStyle.FICTITIOUS_PLAY,
            LearningStyle.DOUBLE_ORACLE,
        }

    @property
    def abbrev(self) -> str:
        """Get two-letter abbreviation for logging and file naming."""
        match self:
            case LearningStyle.EXPLOITER:
                return "ex"
            case LearningStyle.PURE_SELF_PLAY:
                return "sp"
            case LearningStyle.FICTITIOUS_PLAY:
                return "fp"
            case LearningStyle.DOUBLE_ORACLE:
                return "do"


def set_global_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Integer seed to use for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# observation length constants
act_len = 107
glob_obs_len = len(Field) + len(Weather) + 4
side_obs_len = len(SideCondition) + 5
move_obs_len = len(MoveCategory) + len(Target) + len(PokemonType) + 12
# Explicit battle knowledge appended to every Pokemon token: per move slot, the
# damage fraction and guaranteed-KO flag against each of the two foes plus the type
# multiplier against the primary target (4 x 5), then four per-mon scalars. Zeros
# unless PolicyPlayer.use_knowledge_obs is set.
#
# These MUST stay at the END of the token: AttentionExtractor concatenates the chunk
# into one vector, so trailing features map to trailing columns of pokemon_proj --
# which is what lets an old checkpoint be extended with zero-initialised weights and
# still produce bit-identical output. See convert_checkpoint.py.
knowledge_obs_len = 4 * 5 + 4

# What moves and abilities DO (see src/move_semantics.py). Static per id, so these
# cost a dict lookup at encode time. Appended after the knowledge block, keeping the
# whole "explicit knowledge" region at the tail of the token so checkpoints can be
# zero-extended (convert_checkpoint.py).
from vgc_bench.src.move_semantics import ABILITY_SEM_LEN, MOVE_SEM_LEN  # noqa: E402

semantics_obs_len = 4 * MOVE_SEM_LEN + ABILITY_SEM_LEN

# Side-condition PRESENCE flags, appended at the token tail (see
# PolicyPlayer.embed_side_presence). The age encoding in embed_side cannot express
# "set this turn", which made every setup move look like a no-op.
presence_obs_len = len(SideCondition)

# These repair the same set-this-turn ambiguity for global weather/field effects and
# the legacy move-accuracy scaling bug. They are appended AFTER the existing presence
# block so every column in converted checkpoints keeps its old meaning.
global_presence_obs_len = len(Weather) + len(Field)
correct_accuracy_obs_len = 4

pokemon_obs_len = (
    4 * move_obs_len
    + len(Effect)
    + len(PokemonGender)
    + 2 * len(PokemonType)
    + len(Status)
    + 45
    + knowledge_obs_len
    + semantics_obs_len
)
chunk_obs_len = (
    glob_obs_len
    + side_obs_len
    + pokemon_obs_len
    + presence_obs_len
    + global_presence_obs_len
    + correct_accuracy_obs_len
)

# pokemon data
format_map = {"ma": "gen9championsvgc2026regma", "mb": "gen9championsvgc2026regmb"}


def get_reg_from_format(fmt: str) -> str:
    """Extract the regulation identifier from a VGC format string"""
    assert fmt.startswith("gen9championsvgc"), f"not a valid VGC format: {fmt}"
    return fmt.removesuffix("bo3").split("reg")[-1]


with open("data/abilities.json") as f:
    abilities: list[str] = json.load(f)
with open("data/items.json") as f:
    items: list[str] = json.load(f)
with open("data/moves.json") as f:
    moves: list[str] = json.load(f)


def refuse_eval_only_checkpoint(path: "str | os.PathLike") -> None:
    """Hard-fail when a checkpoint stamped ``role: eval_only`` enters training.

    The held-out human-imitation arm (bc_eval_B) exists to measure candidates
    against a population nothing was fit on. Letting it into any data
    generation or training mix would quietly destroy that property, so every
    pipeline that consumes an opponent checkpoint calls this first. Sidecars
    are written by stamp_checkpoint_metadata.py.
    """
    from pathlib import Path as _Path

    sidecar = _Path(str(path) + ".metadata.json")
    if not sidecar.exists():
        return
    try:
        role = json.loads(sidecar.read_text()).get("role")
    except (OSError, ValueError):
        return
    if role == "eval_only":
        raise SystemExit(
            f"{path} is stamped role=eval_only (a held-out evaluation arm). "
            "It must never appear in training or data generation; use "
            "bc_mix_A or a frozen PPO opponent instead."
        )
