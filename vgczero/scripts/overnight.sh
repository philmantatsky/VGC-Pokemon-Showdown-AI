#!/bin/bash
# One command for an unattended overnight vgczero training run on a Mac (works on Linux too).
#
#   bash scripts/overnight.sh                  # 8 hours into runs/overnight
#   bash scripts/overnight.sh --hours 10       # takes any scripts/overnight.py option
#
# The first time it creates .venv (Python 3.10+), installs maturin/numpy/torch/pytest/websockets,
# builds the Rust engine and runs the tests (5-10 minutes); later runs skip what is already done.
# It needs the Xcode command line tools and Rust 1.80+ (rustup), and says how to get them if missing.
# Ctrl-C stops cleanly: the trainer saves, then REPORT.md is written in the run directory.
set -eu
cd "$(dirname "$0")/.."

# Keep the Mac awake from the start: setup (first run: package install + engine build) and the
# speed probes take a while, and a sleeping Mac would wake up in the morning past the deadline.
# `exec` below keeps this PID, so this also covers the training run.
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -i -s -w $$ &
fi

say() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# 1. Python 3.10+ in vgczero/.venv (macOS's own /usr/bin/python3 is 3.9 and cannot load the engine).
if [ ! -x .venv/bin/python ]; then
  BASE=""
  for c in python3.12 python3.13 python3.11 python3.10 /opt/homebrew/bin/python3 /usr/local/bin/python3 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      BASE="$c"
      break
    fi
  done
  [ -n "$BASE" ] || die "no Python 3.10+ found. Install one with: brew install python@3.12"
  say "creating .venv with $("$BASE" --version 2>&1)"
  "$BASE" -m venv .venv
fi
. .venv/bin/activate
# A conda base environment left active makes maturin refuse to build ("Both VIRTUAL_ENV and CONDA_PREFIX are set").
unset CONDA_PREFIX CONDA_DEFAULT_ENV
python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
  || die ".venv has Python < 3.10; delete vgczero/.venv and run this again"
if ! python -c 'import numpy, torch, pytest, websockets' >/dev/null 2>&1 || ! command -v maturin >/dev/null 2>&1; then
  say "installing Python packages (torch is the big one)"
  python -m pip install --upgrade pip
  python -m pip install maturin numpy torch pytest websockets
fi

# 2. Rust 1.80+ and (on a Mac) the Xcode command line tools, needed to build the engine.
if ! command -v cargo >/dev/null 2>&1 && [ -x "$HOME/.cargo/bin/cargo" ]; then
  PATH="$HOME/.cargo/bin:$PATH"
fi
if [ "$(uname)" = "Darwin" ] && ! xcode-select -p >/dev/null 2>&1; then
  die "the Xcode command line tools are missing. Run: xcode-select --install   (then run this again)"
fi
command -v cargo >/dev/null 2>&1 || die "Rust is not installed. Install it with:
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
then open a new terminal and run this again."
RV="$(rustc --version 2>/dev/null | awk '{print $2}')"
case "$RV" in
  [0-9]*.[0-9]*) ;;
  *) die "rustc did not report a version. Run: rustup default stable" ;;
esac
MAJ="${RV%%.*}"
REST="${RV#*.}"
MIN="${REST%%.*}"
if [ "$MAJ" -lt 1 ] || { [ "$MAJ" -eq 1 ] && [ "$MIN" -lt 80 ]; }; then
  die "Rust $RV is too old (1.80+ needed). Run: rustup update stable"
fi

# 3. The engine: (re)build when it is missing or its sources changed since the last build.
STAMP=.venv/.engine_built
if ! python -c 'import vgczero_engine' >/dev/null 2>&1 || [ ! -f "$STAMP" ] \
  || [ -n "$(find engine/src engine/Cargo.toml engine/Cargo.lock -newer "$STAMP" -print 2>/dev/null | head -n 1)" ]; then
  say "building the Rust engine (the first build takes a few minutes)"
  (cd engine && maturin develop --release)
  touch "$STAMP"
fi

# 4. Tests (a few seconds), so a broken setup fails now rather than at 2 am.
say "running the tests"
python -m pytest -q -x -p no:cacheprovider || die "tests failed; not starting a night of training on a broken setup"

# 5. Train.
say "starting the overnight run (Ctrl-C stops it cleanly and writes the report)"
exec python scripts/overnight.py "$@"
