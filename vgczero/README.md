# vgczero

Self-play reinforcement learning for **[Gen 9 Champions] VGC 2026 Reg M-C**
doubles, modelled on mikumiku37 and Jaxcalibur: a Rust battle engine ~690x
faster than Showdown, a ~8M-parameter entity transformer trained from scratch
with PPO against a league of its own past versions (terminal rewards only), and
a one-turn simultaneous-move search over sampled hidden-information worlds.
See [DESIGN.md](DESIGN.md) for the full picture and status.

## Setup

```bash
cd vgczero
python3 -m venv .venv && . .venv/bin/activate
pip install maturin numpy torch pytest
(cd engine && maturin develop --release)        # builds the vgczero_engine module
```

Rust ≥ 1.75 is needed for the engine. Node is only needed to regenerate data
or run the parity tests: `showdown/setup.sh` clones and builds the pinned
upstream Showdown into `showdown/ps` (or set `PS_DIR`).

## Use

```bash
# Engine speed + which teams are fully supported
(cd engine && cargo run --release --bin bench -- 20000)

# Train (resumes if the run dir exists). Laptop smoke run:
python scripts/train.py --run-dir runs/smoke --model tiny --n-envs 64 --rollout-steps 16 --updates 20
# Real run on a GPU box:
python scripts/train.py --run-dir runs/base1 --model base --n-envs 2048 --rollout-steps 32

# Evaluate head to head (checkpoints or 'random' / 'greedy')
PYTHONPATH=python python -m vgczero.evaluate runs/base1/checkpoints/latest.pt greedy --games 2000
```

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
