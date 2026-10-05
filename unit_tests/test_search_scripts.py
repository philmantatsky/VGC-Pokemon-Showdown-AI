"""The search launchers (tools/search_mirror_chain.sh, one round of the head-to-head;
tools/search_mirror_rounds.sh, several; tools/search_roster_arms.sh, the held-out
battery's opponents): they parse, survive macOS bash 3.2, never delete anything, and
pool only what finished."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    ROOT / "tools/search_mirror_chain.sh",
    ROOT / "tools/search_mirror_rounds.sh",
    ROOT / "tools/search_roster_arms.sh",
]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_parses_and_is_executable(script):
    subprocess.run(["bash", "-n", str(script)], check=True)
    assert script.stat().st_mode & 0o111


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_nothing_is_deleted(script):
    assert not re.search(r"(^|[;&|]\s*)rm\s", script.read_text(), flags=re.M)


@pytest.mark.parametrize("script", SCRIPTS[1:], ids=lambda p: p.name)
def test_empty_arrays_are_bash32_safe(script):
    text = script.read_text()
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


def test_roster_script_gives_every_opponent_its_own_server_and_pools_the_finished():
    text = (ROOT / "tools/search_roster_arms.sh").read_text()
    assert "evaluation/search_roster_ab.py" in text and "caffeinate -is" in text
    assert "port=$((PORT + k))" in text  # one battle at a time, one server each
    assert "from evaluation.run_t6_confirmation import OPPONENTS" in text
    assert "--deterministic-opponent" in text
    pooled = text[text.index("done_runs=()") :]
    assert '"complete": true' in pooled and "--pool" in pooled
    # it runs in the checkout it is called from, like the rounds script
    assert 'cd "$(dirname "$SELF")/.."' in text
