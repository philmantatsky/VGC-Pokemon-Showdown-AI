"""The shell launchers that can put the bot online: they parse, survive macOS
bash 3.2 (an empty "${array[@]}" is an unbound-variable error under set -u,
which crashed the default challenge listener), and refuse while any heavy local
job runs -- including the human-preview training and evaluation jobs and the
guard mirror / A/B runners."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = [
    ROOT / "tools/ladder_read_loop.sh",
    ROOT / "tools/ladder_deployed.sh",
    ROOT / "tools/ladder_trial.sh",
    ROOT / "tools/challenges_deployed.sh",
    ROOT / "exhibition_mode.sh",
]
# the two ladder wrappers exec tools/ladder_read_loop.sh, which refuses and
# passes the flags for them
WRAPPERS = ("ladder_deployed.sh", "ladder_trial.sh")
REFUSING = [p for p in LAUNCHERS if p.name not in WRAPPERS]
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
    ".venv/bin/python -u training/gen_tactical_data.py --focus-moves earthpower",
    ".venv/bin/python -u training/tactical_sft.py --lessons focus",
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


@pytest.mark.parametrize("script", REFUSING, ids=lambda p: p.name)
def test_launchers_pass_the_deployed_mixing_flags(script):
    """DEPLOYED.json's mixing reaches ladder_ourteam.py (empty = off)."""
    assert "MIXING_ARGS" in script.read_text()


def test_the_deployed_ladder_exports_mixing():
    assert "export MIXING" in (ROOT / "tools/ladder_deployed.sh").read_text()


@pytest.mark.parametrize("script", REFUSING, ids=lambda p: p.name)
def test_launchers_pass_the_deployed_sticky_flag(script):
    """DEPLOYED.json's sticky_guard_corrections reaches ladder_ourteam.py."""
    text = script.read_text()
    assert "STICKY" in text and "--sticky-corrections" in text


def test_the_deployed_ladder_exports_sticky():
    assert "export STICKY" in (ROOT / "tools/ladder_deployed.sh").read_text()


@pytest.mark.parametrize("script", REFUSING, ids=lambda p: p.name)
def test_launchers_pass_the_playbook(script):
    """DEPLOYED.json's playbook (or a trial's) reaches ladder_ourteam.py."""
    text = script.read_text()
    assert "PLAYBOOK" in text and "--playbook" in text


@pytest.mark.parametrize("name", WRAPPERS)
def test_the_ladder_wrappers_export_what_the_read_loop_reads(name):
    text = (ROOT / "tools" / name).read_text()
    for variable in ("PREVIEW_MODEL", "MIXING", "STICKY", "PLAYBOOK"):
        assert re.search(rf"export\b.*\b{variable}\b", text), variable


def test_a_trial_is_the_verified_deployment_plus_its_additions():
    """Checked by reading, never by running: a launcher that got past its checks
    would log in and play real ladder games."""
    text = (ROOT / "tools/ladder_trial.sh").read_text()
    assert "tools/deployed_config.py" in text and 'eval "$CFG"' in text
    assert 'ladder_replays_mc_deployed_"$REPLAY_TAG"' in text  # its own dir only
    assert 'GUARDS="$GUARDS,$TRIAL_GUARDS"' in text  # added, never replacing
    assert "exec ./tools/ladder_read_loop.sh" in text
    assert "CKPT=$TRIAL_CHECKPOINT" in text  # a candidate brain, the user's word
    # a team variant only with a brain practised on it, and only the same six
    assert "TRIAL_TEAM needs TRIAL_CHECKPOINT" in text
    assert "not a set variant of the deployed team" in text
    assert "TEAM=$TRIAL_TEAM" in text


def test_a_trial_can_take_the_turn1_script_alone():
    """TRIAL_SCRIPT_ONLY (2026-10-03): the deployed preview plus the card's turn-1
    script; the read loop passes --playbook-script-only, never without a playbook."""
    trial = (ROOT / "tools/ladder_trial.sh").read_text()
    assert "TRIAL_SCRIPT_ONLY needs TRIAL_PLAYBOOK" in trial
    assert re.search(r"export\b.*\bPLAYBOOK_SCRIPT_ONLY\b", trial)
    loop = (ROOT / "tools/ladder_read_loop.sh").read_text()
    assert "PLAYBOOK_ARGS+=(--playbook-script-only)" in loop
    assert "PLAYBOOK_SCRIPT_ONLY needs PLAYBOOK" in loop


@pytest.mark.parametrize(
    "script",
    [
        ROOT / "tools/ladder_read_loop.sh",
        ROOT / "tools/challenges_deployed.sh",
        ROOT / "exhibition_mode.sh",
    ],
    ids=lambda p: p.name,
)
def test_launchers_pass_the_sheet_preview(script):
    """DEPLOYED.json's sheet_preview (2026-10-03) reaches ladder_ourteam.py."""
    assert "--sheet-preview" in script.read_text()


@pytest.mark.parametrize("name", WRAPPERS)
def test_the_ladder_wrappers_export_the_sheet_preview(name):
    text = (ROOT / "tools" / name).read_text()
    assert re.search(r"export\b.*\bSHEET_PREVIEW\b", text)


def test_a_trial_can_add_the_search_and_the_deployed_launcher_never_does():
    """TRIAL_SEARCH (2026-10-04): exact-search flags reach ladder_ourteam.py only
    through a trial; they must turn the search on and put the networks on cpu."""
    trial = (ROOT / "tools/ladder_trial.sh").read_text()
    assert "TRIAL_SEARCH must turn the search on (--search)" in trial
    assert "TRIAL_SEARCH must run the networks on cpu (--device cpu)" in trial
    assert re.search(r"export\b.*\bSEARCH\b", trial)
    loop = (ROOT / "tools/ladder_read_loop.sh").read_text()
    assert 'read -r -a SEARCH_ARGS <<< "$SEARCH"' in loop
    # an empty array under `set -u` on macOS bash 3.2
    assert '${SEARCH_ARGS[@]+"${SEARCH_ARGS[@]}"}' in loop
    deployed = (ROOT / "tools/ladder_deployed.sh").read_text()
    assert re.search(r"^SEARCH=$", deployed, re.M)
    assert re.search(r"^export SEARCH$", deployed, re.M)


def test_the_trials_search_example_is_the_configuration_that_was_measured():
    """The flags written in tools/ladder_trial.sh parse, and they are the roster
    run's configuration (evaluation/search_roster_ab.py: raw leaf, streams) -- its
    reply prior included: the local players carry no opponent models, so the search
    ranked replies with the brain alone (found 2026-10-09, with the ladder trial
    already playing the blend)."""
    import shlex

    from ladder_ourteam import build_parser, search_planner_config

    trial = (ROOT / "tools/ladder_trial.sh").read_text()
    example = re.search(r'#\s+TRIAL_SEARCH="(.*?)"', trial, re.S)
    assert example is not None
    flags = shlex.split(re.sub(r"\\\n#", " ", example.group(1)))
    args = build_parser().parse_args(flags)
    assert args.search and args.search_every_turn and args.device == "cpu"
    assert (args.search_leaf, args.search_streams) == ("critic", 4)
    assert args.search_determinizations == 4
    assert args.search_leaf_calibration == ""
    assert args.search_reply_prior == "brain"
    config = search_planner_config(args)
    assert (config.solution, config.nash_anchor, config.nash_sample) == (
        "nash",
        0.07,
        False,
    )
    assert config.opponent_width == 8


def test_the_rehearsal_plays_what_the_ladder_loop_plays_and_reads_no_credentials():
    """tools/ladder_rehearse.sh exists to put the ladder loop's own command line on a
    local server before rated games (2026-10-09: its first run caught a search fault
    the unit tests and the local harnesses could not see)."""
    rehearse = (ROOT / "tools/ladder_rehearse.sh").read_text()
    loop = (ROOT / "tools/ladder_read_loop.sh").read_text()
    for flags in (
        '--checkpoint "$CKPT" --reg mc --our_team "$TEAM"',
        '--guards-extra "$GUARDS"',
        '--learned_preview --preview_model "$PREVIEW_MODEL"',
        "--sticky-corrections",
        "--sheet-preview",
        '--opponent-forecast "$FORECAST"',
        '--playbook "$PLAYBOOK"',
    ):
        assert flags in rehearse and flags in loop, flags
    assert "tools/deployed_config.py" in rehearse
    assert 'export VGC_SET_PRIOR_REG="$SET_PRIOR"' in rehearse
    driver = (ROOT / "checks/ladder_rehearsal.py").read_text()
    for text in (rehearse, driver):
        # the credentials file of the real launchers
        assert "Laplace-Pokemon-Showdown-AI" not in text and "/.env" not in text
    assert "SHOWDOWN" not in rehearse and "source " not in rehearse
    # the driver names a throwaway local guest and makes sure no password is set
    assert 'os.environ.pop("SHOWDOWN_PASSWORD", None)' in driver
    assert "ladder_ourteam.ShowdownServerConfiguration = SERVER" in driver


# --- the lid ----------------------------------------------------------------------
# 2026-10-09: the lid was closed in the middle of a rated game. The launcher's restart
# logged in and queued in a dark wake seconds before the next sleep: a second rated
# game abandoned.

LID = '  |   "AppleClamshellState" = {}\n'
SLEEPS = '  |   "AppleClamshellCausesSleep" = {}\n'
OPEN = SLEEPS.format("Yes") + LID.format("No")
CLOSED = SLEEPS.format("Yes") + LID.format("Yes")


def _script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)
    return path


def _fake_tools(tmp_path: Path) -> dict[str, str]:
    """An ioreg that says what a file says, and no heavy job, on a PATH of its own."""
    fake = tmp_path / "bin"
    _script(fake / "ioreg", 'cat "$FAKE_IOREG" 2>/dev/null\n')
    _script(fake / "pgrep", "exit 1\n")
    _script(fake / "caffeinate", 'shift\nexec "$@"\n')
    return {
        "PATH": f"{fake}:/usr/bin:/bin",
        "FAKE_IOREG": str(tmp_path / "registry"),
        "FAKE_PID": str(tmp_path / "session.pid"),
        "TICK": "1",
    }


@pytest.mark.parametrize(
    "registry, closed",
    [
        (OPEN, False),
        (CLOSED, True),
        # a closed lid with a display and power does not sleep: clamshell mode
        (SLEEPS.format("No") + LID.format("Yes"), False),
        # closed, and nothing says it stays awake
        (LID.format("Yes"), True),
        # a desktop has no lid; an ioreg that answers nothing is no closed lid
        ("", False),
        (None, False),
    ],
)
def test_the_lid_helper_reads_the_power_managers_two_keys(tmp_path, registry, closed):
    helper = ROOT / "tools/lid_closed.sh"
    subprocess.run(["bash", "-n", str(helper)], check=True)
    env = _fake_tools(tmp_path)
    if registry is not None:
        Path(env["FAKE_IOREG"]).write_text(registry)
    done = subprocess.run(["bash", str(helper)], env=env, capture_output=True)
    assert (done.returncode == 0) is closed
    assert done.stdout == b""


def _ladder_tree(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """The read loop and its lid helper in a tree of their own: a session that logs in
    and then only waits, and credentials that are not the real ones -- the loop sources
    ``../Laplace-Pokemon-Showdown-AI/.env`` relative to where IT lies."""
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    for name in ("ladder_read_loop.sh", "lid_closed.sh"):
        shutil.copy2(ROOT / "tools" / name, repo / "tools" / name)
    assert (
        'source "../Laplace-Pokemon-Showdown-AI/.env"'
        in (repo / "tools/ladder_read_loop.sh").read_text()
    )
    (tmp_path / "Laplace-Pokemon-Showdown-AI").mkdir()
    (tmp_path / "Laplace-Pokemon-Showdown-AI/.env").write_text("SHOWDOWN_USERNAME=x\n")
    _script(
        repo / ".venv/bin/python",
        'echo "account : nobody"\necho $$ > "$FAKE_PID"\n'
        "trap 'exit 0' TERM\nwhile :; do sleep 0.2; done\n",
    )
    (repo / "brain.zip").write_text("")
    (repo / "team.txt").write_text("")
    return repo, _fake_tools(tmp_path)


def _run_loop(repo: Path, env: dict[str, str], during=None) -> tuple[int, str]:
    """The read loop for one game in the tree, in a process group of its own so that
    nothing of it outlives the test: a loop that never ends fails, it does not hang."""
    loop = subprocess.Popen(
        ["bash", str(repo / "tools/ladder_read_loop.sh"), "brain.zip", "team.txt"]
        + ["1", "replays"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        if during is not None:
            during(loop)
        out, _ = loop.communicate(timeout=40)
        return loop.returncode, out
    finally:
        try:
            os.killpg(loop.pid, 9)
        except ProcessLookupError:
            pass


def test_the_read_loop_starts_no_session_with_the_lid_closed(tmp_path):
    repo, env = _ladder_tree(tmp_path)
    Path(env["FAKE_IOREG"]).write_text(CLOSED)
    status, out = _run_loop(repo, env)
    assert status == 3, out
    assert "LADDER_ABORT" in out and "the lid is closed" in out
    assert "SESSION_START" not in out and "LADDER_DONE" not in out
    assert not Path(env["FAKE_PID"]).exists()  # nothing logged in


def test_the_read_loop_stops_a_playing_session_when_the_lid_closes(tmp_path):
    repo, env = _ladder_tree(tmp_path)
    registry, pid_file = Path(env["FAKE_IOREG"]), Path(env["FAKE_PID"])
    registry.write_text(OPEN)

    def close_the_lid(loop: subprocess.Popen) -> None:
        deadline = time.monotonic() + 20
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert pid_file.exists(), "the session never started with the lid open"
        time.sleep(1.5)  # it plays; the loop has looked at it at least once
        assert loop.poll() is None
        registry.write_text(CLOSED)

    status, out = _run_loop(repo, env, close_the_lid)
    assert status == 3, out
    assert out.count("SESSION_START") == 1
    assert "LID_CLOSED session=1" in out
    assert "the lid was closed during session 1" in out and "LADDER_DONE" not in out
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)  # the session is gone


def test_the_read_loop_looks_at_a_session_every_twenty_seconds_unless_told():
    text = (ROOT / "tools/ladder_read_loop.sh").read_text()
    assert "TICK=${TICK:-20}" in text and 'sleep "$TICK"' in text
    # both lid checks come before anything is started or restarted
    start = text.index('echo "SESSION_START')
    assert text.index("./tools/lid_closed.sh") < start
    assert text.count("./tools/lid_closed.sh") == 2
