import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import pytest  # noqa: E402

from vgczero import data as D  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def engine():
    D.load()
    return D.E
