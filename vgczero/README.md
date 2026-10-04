# vgczero

Self-play reinforcement learning for **[Gen 9 Champions] VGC 2026 Reg M-C**
doubles, modelled on mikumiku37 and Jaxcalibur: a Rust battle engine ~690x
faster than Showdown, a ~8M-parameter entity transformer trained from scratch
with PPO against a league of its own past versions (terminal rewards only), and
a one-turn simultaneous-move search over sampled hidden-information worlds.
See [DESIGN.md](DESIGN.md) for the full picture and status.

## Setup

Needs Python ≥ 3.10 and Rust ≥ 1.80 (via rustup).

```bash
cd vgczero
python3.12 -m venv .venv && . .venv/bin/activate   # any Python >= 3.10
pip install maturin numpy torch pytest websockets
(cd engine && maturin develop --release)           # builds the vgczero_engine module; re-run after engine changes
pytest && (cd engine && cargo test --release)
```

Node ≥ 22.18 is only needed for the Showdown tooling: regenerating data, parity runs, compiling
team pastes for `matchup.py`, validating evolved teams and running a local server.
`showdown/setup.sh` clones and builds the pinned upstream Showdown into `showdown/ps` (or set `PS_DIR`).

## On a Mac (Apple Silicon)

A MacBook can run all of it: the engine, tests, parity runs, training of the `tiny` and `small`
presets, evaluation, search, the matchup lab, team evolution and live play. The full `base` run is a
GPU job (see the throughput notes in DESIGN.md).

### Overnight training in one command

```bash
cd "<your clone>"                       # e.g. ~/Desktop/pokemon\ showdown\ bot/vgc-bench
git fetch origin && git worktree add ../vgczero-wt claude/nice-bohr-mvs9zh
cd ../vgczero-wt/vgczero
bash scripts/overnight.sh               # 8 hours; --hours N to change
```

The first time, it creates `.venv`, installs the Python packages, builds the engine and runs the
tests (5-10 minutes; it tells you how to install Rust or the Xcode tools if they are missing). Then it:

* checks memory, power and disk space, and refuses to start while antonius1's batteries, mirrors
  or training are running (`--force` overrides). A challenge listener can stay up.
* trains the `small` model with a minibatch that fits in memory, and probes a few updates on the
  CPU and the GPU (MPS) to keep the faster device.
* keeps the Mac awake with `caffeinate`, restarts from the last checkpoint if the trainer crashes,
  and drops to the CPU if MPS fails twice.
* evaluates against the greedy baseline about every 30 minutes, then writes
  `runs/overnight/REPORT.md` with a final evaluation and the best teams.

Keep it plugged in with the lid open (the display can sleep). Ctrl-C stops cleanly: the trainer
saves, then the report is written. On later nights, `cd ../vgczero-wt/vgczero && git pull && bash
scripts/overnight.sh` keeps improving the same model. Pass `--model tiny` for a faster, smaller model.

* Python: `/usr/bin/python3` is 3.9 and cannot install the engine. Use Homebrew's, python.org's or
  uv's Python 3.10+. Run `xcode-select --install` and rustup before `maturin develop`.
* If this clone is also the working antonius1 checkout, give vgczero its own worktree so the
  deployed bot's files stay as they are:
  `git fetch origin && git worktree add ../vgczero-wt claude/nice-bohr-mvs9zh && cd ../vgczero-wt/vgczero`.
* `train.py`'s defaults are the GPU recipe: `base` model, minibatch 4096, and `--device auto`, which
  picks MPS on a Mac. Peak memory of a training step, measured on CPU:

  | model | minibatch 4096 | 2048 | 1024 |
  |---|---|---|---|
  | tiny | 2.7 GB | | |
  | small | 7.3 GB | 4.4 GB | |
  | base | 13.8 GB | 8.3 GB | 4.7 GB |

  On 8 GB, train `tiny`, or `small` with `--minibatch 1024`. On 16 GB, `small` fits, or `base` at
  `--minibatch 1024`-`2048`. On a CPU, smaller minibatches were also faster per sample.
* CPU or MPS: for `tiny`, pass `--device cpu`. For `small`/`base`, run 5 updates with each device and
  keep the faster one. On the first MPS run, check that `illegal` in `metrics.jsonl` stays 0.
* Measured on a 4-vCPU x86 VM (a MacBook CPU should be similar or faster): `tiny` self-play runs at
  55-120 games/s and reached 45-50% against `greedy` in about 45 minutes; `small` runs at 10-13
  games/s and `base` at about 6.
* Run vgczero while the antonius1 batteries or training are idle: competing processes slowed
  search 4-24x in testing. To cap threads, set `OMP_NUM_THREADS` / `RAYON_NUM_THREADS`, or use
  `play.py --threads`.
* Run the local Showdown server on a port other than 8000 (vgc_bench uses 8000), e.g.
  `node pokemon-showdown start 8123 --no-security` in `showdown/ps`.
* Ladder or accept challenges with a different account from antonius1. The Reg M-C timer is a
  420 s bank with nothing added per turn (55 s max per turn). Default matrix search with `base` takes
  about 1 s per decision. Tree search at 2048 simulations takes about 9 s on 4 threads, and about 25 s
  on 1 thread, which can run the bank out in a long game.
* For short runs, use `top_teams.py --min-games 30`.

## Use

```bash
# Engine speed + which teams are fully supported
(cd engine && cargo run --release --bin bench -- 20000)

# Train (resumes if the run dir exists). Laptop smoke run:
python scripts/train.py --run-dir runs/smoke --model tiny --device cpu --n-envs 64 --rollout-steps 16 --updates 20
# Real run on a GPU box:
python scripts/train.py --run-dir runs/base1 --model base --n-envs 2048 --rollout-steps 32

# Evaluate head to head (checkpoints or 'random' / 'greedy')
PYTHONPATH=python python -m vgczero.evaluate runs/base1/checkpoints/latest.pt greedy --games 2000
```

Play on Showdown (accept challenges with the ten best training teams, search on):

```bash
python scripts/top_teams.py --run-dir runs/base1 --k 10
python scripts/play.py --checkpoint runs/base1/checkpoints/latest.pt --username NAME --password PASS \
    --mode accept --team-file runs/base1/top_teams.json --team-top 10
```

Evolve teams with a trained policy:

```bash
python scripts/evolve.py --checkpoint runs/base1/checkpoints/latest.pt --run-dir runs/base1 --generations 20
node showdown/validate_team.js runs/base1/evolve/population_latest.json
```

Matchup lab (win rate, each side's bring/lead choices, solved team-preview game):

```bash
python scripts/matchup.py --checkpoint runs/base1/checkpoints/latest.pt --a my_team.txt --b MC2001 --games 400
```

Tests: `pytest` (Python, from `vgczero/`) and `cargo test --release` (engine, including Showdown parity
fixtures). Full parity runs against a live Showdown build: see `showdown/parity/README.md`.

Search from Python:

```python
from vgczero import data as D
from vgczero.model import load_model
from vgczero.search import search, SearchConfig
D.load()
model = load_model("runs/base1/checkpoints/latest.pt")
battle = D.E.Battle(team_a=0, team_b=1, seed=0)     # pool team indices
result = search(model, battle, viewer=0, cfg=SearchConfig(worlds=16, k_self=8, k_opp=8))
print(result.choice, result.strategy)               # (preview, slot0, slot1) and the mixed strategy
```

## Regenerating data

```bash
node showdown/export_dex.js                          # -> data/dex_regmc.json.gz
node showdown/compile_teams.js ../teams/reg_mc       # -> data/teams_regmc.json.gz
```

## Layout

| path | what |
|---|---|
| `showdown/` | Node scripts that treat Showdown as the source of truth (export, parity, speed reference) |
| `engine/` | Rust engine, observation encoder, batched env, world sampling, baselines, Python bindings |
| `python/vgczero/` | model, PPO + league, team stats, search, evaluation |
| `scripts/` | CLIs |
| `data/` | exported dex and compiled team pool (gzipped JSON) |
