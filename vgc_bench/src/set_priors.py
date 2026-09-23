"""Format-specific set evidence, shared by observations and tactical helpers."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"


def prior_reg(formatid: str | None = None) -> str:
    """An explicit override permits controlled old/new-information comparisons."""
    override = os.environ.get("VGC_SET_PRIOR_REG")
    if override:
        if override not in {"ma", "mb", "mc"}:
            raise ValueError("VGC_SET_PRIOR_REG must be ma, mb or mc")
        return override
    match = re.search(r"reg(m[a-z])(?:bo3)?$", formatid or "")
    return match.group(1) if match else "mb"


@lru_cache(maxsize=8)
def _load(reg: str) -> tuple[dict, dict]:
    def read(kind: str) -> dict:
        path = DATA / f"{kind}_reg{reg}.json"
        if not path.exists():
            # Missing current-format evidence is unknown, never old-format fact.
            return {}
        return json.loads(path.read_text())

    return read("joint_sets"), read("movesets")


def load_set_priors(formatid: str | None = None) -> tuple[dict, dict]:
    return _load(prior_reg(formatid))
