#!/usr/bin/env bash
set -euo pipefail

# League round 6 (2026-09-10): the first Reg M-C round. Single variable vs
# round 1's recipe: the DATA -- M-C opponent team pool (open-team-sheet teams,
# `data/team_weights_regmc_league6.json`), a Reg M-C human clone as the human
# backbone, no adversary (four adversary rounds all overfit). Our side stays
# MB430 for continuity until the team tournament picks the M-C team. Pool
# copies are listed in training/league6_config.json (built by after_ladder_mc.sh).
#
# Prereqs (manual): Showdown server on the port below --
#   cd pokemon-showdown && node pokemon-showdown start 7700 --no-security &
# Launch (house rules: AC power, lid open):
#   nohup caffeinate -is ./training/run_league6_training.sh > league6_$(date +%H%M%S).log 2>&1 &

.venv/bin/python training/build_league.py --config training/league6_config.json --verify-only || {
  echo "Training refused: league-6 pool failed verification. Run training/build_league.py --config training/league6_config.json first." >&2
  exit 2
}

exec .venv/bin/python -u -m vgc_bench.train \
  --fictitious_play \
  --reg mc \
  --run_id 1 \
  --our_team teams/reg_mc/our_team.txt \
  --knowledge_obs \
  --hidden_sheet_prob 0.50 \
  --team_weights data/team_weights_regmc_league6.json \
  --results_suffix league6 \
  --num_envs 8 \
  --num_eval_workers 8 \
  --port 7700 \
  --device mps \
  --learning_rate 0.00003 \
  --n_epochs 3 \
  --target_kl 0.02 \
  --total_steps 17694720
