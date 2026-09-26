"""The shell launchers that can put the bot online: they parse, survive macOS
bash 3.2 (an empty "${array[@]}" is an unbound-variable error under set -u,
which crashed the default challenge listener), and refuse while any heavy local
job runs -- including the human-preview training and evaluation jobs and the
guard mirror / A/B runners."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = [
    ROOT / "tools/ladder_read_loop.sh",
    ROOT / "tools/ladder_deployed.sh",
    ROOT / "tools/challenges_deployed.sh",
    ROOT / "exhibition_mode.sh",
]
REFUSING = [p for p in LAUNCHERS if p.name != "ladder_deployed.sh"]
HEAVY_JOBS = [
    ".venv/bin/python -m vgc_bench.train --reg mc",
    ".venv/bin/python -u training/human_preview.py -- --no_teampreview --reg mc",
    ".venv/bin/python -u training/run_t6_human_preview_trial.py",
    ".venv/bin/python -u training/preview_entropy.py --boost 9 -- --reg mc",
    "caffeinate -is .venv/bin/python -u evaluation/run_candidate_vs_t6.py --label x",
    "python evaluation/learned_preview_study.py --preview-model m.pt -- --port 7610",
    "python evaluation/opening_study.py --checkpoint c.zip",
    "bash /scratch/human_preview_chain.sh",
    ".venv/bin/python evaluation/mirror_guard_ab.py --guard focus_boosted",
    ".venv/bin/python evaluation/run_guard_ab.py --guards focus_boosted",
]


@pytest.mark.parametrize("script", LAUNCHERS, ids=lambda p: p.name)
def test_launcher_parses(script):
    subprocess.run(["bash", "-n", str(script)], check=True)


@pytest.mark.parametrize("script", LAUNCHERS, ids=lambda p: p.name)
def test_array_expansions_are_bash32_safe(script):
    text = script.read_text()
    for name in re.findall(r'"\$\{(\w+)\[@\]\}"', text):
        guarded = '${%s[@]+"${%s[@]}"}' % (name, name)
        bare = text.replace(guarded, "")
        assert f'"${{{name}[@]}}"' not in bare, f"{script.name}: bare {name}[@]"


def _heavy(script: Path) -> re.Pattern:
    line = next(s for s in script.read_text().splitlines() if s.startswith("HEAVY="))
    return re.compile(line.removeprefix("HEAVY='").removesuffix("'"))


@pytest.mark.parametrize("script", REFUSING, ids=lambda p: p.name)
def test_launchers_refuse_during_heavy_jobs(script):
    heavy = _heavy(script)
    for command in HEAVY_JOBS:
        assert heavy.search(command), f"{script.name} would not refuse: {command}"
    ladder = ".venv/bin/python -u ladder_ourteam.py --checkpoint c.zip --reg mc"
    assert not heavy.search(ladder)
