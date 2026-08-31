#!/usr/bin/env bash
set -euo pipefail

# Intentionally refuses to run until check_training_gate.py creates this marker.
gate="results_repaired/training_gate.pass"
if [[ ! -f "$gate" ]]; then
  echo "Training refused: $gate is missing. Run open, hidden, and learned-population evaluations plus check_training_gate.py first." >&2
  exit 2
fi

seed_dir="results_repaired/saves_fp_hs_wt/reg_mb/seed1"
seed_checkpoint="$seed_dir/3932160.zip"
mkdir -p "$seed_dir"
if [[ ! -f "$seed_checkpoint" ]]; then
  cp results_repaired/converted_v4.zip "$seed_checkpoint"
fi

exec .venv/bin/python -m vgc_bench.train \
  --fictitious_play \
  --reg mb \
  --run_id 1 \
  --our_team teams/reg_mb/our_team.txt \
  --knowledge_obs \
  --hidden_sheet_prob 0.50 \
  --team_weights data/team_weights_regmb.json \
  --results_suffix repaired \
  --num_envs 8 \
  --num_eval_workers 8 \
  --port 7400 \
  --device mps \
  --learning_rate 0.00003 \
  --n_epochs 3 \
  --target_kl 0.02 \
  --total_steps 7864320
