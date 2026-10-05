"""The search head-to-head launchers (tools/search_mirror_chain.sh, one round;
tools/search_mirror_rounds.sh, several): they parse, survive macOS bash 3.2, never
delete anything, and the rounds script pools only shards that finished."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    ROOT / "tools/search_mirror_chain.sh",
    ROOT / "tools/search_mirror_rounds.sh",
]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_parses_and_is_executable(script):
    subprocess.run(["bash", "-n", str(script)], check=True)
    assert script.stat().st_mode & 0o111


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_nothing_is_deleted(script):
    assert not re.search(r"(^|[;&|]\s*)rm\s", script.read_text(), flags=re.M)


def test_rounds_script_empty_arrays_are_bash32_safe():
    text = (ROOT / "tools/search_mirror_rounds.sh").read_text()
    for name in re.findall(r'"\$\{(\w+)\[@\]\}"', text):
        guarded = '${%s[@]+"${%s[@]}"}' % (name, name)
        bare = text.replace(guarded, "")
        if f'"${{{name}[@]}}"' in bare:
            # an unguarded expansion is fine only behind an explicit non-empty check
            assert re.search(r"\[ \$\{#%s\[@\]\} -gt 0 \]" % name, text), name


def test_rounds_script_pools_only_finished_shards_and_sets_leftovers_aside():
    text = (ROOT / "tools/search_mirror_rounds.sh").read_text()
    pool = text[text.index("pool() {") : text.index('say "ROUNDS_START')]
    assert '"complete": true' in pool and "pool_mirrors.py" in pool
    assert "_unfinished_" in text and "mv " in text
    assert "caffeinate -is" in text
    # every round has its own seeds
    assert "100000 * r + 1000 * k" in text
