#!/usr/bin/env bash
set -euo pipefail

# League round 3: continue from the deployed league champion (12,779,520)
# with the EXPLOITER in the pool -- the adversary that beats the champion 60.2%
# via the opening exchange and long games (the same holes 1300+ humans use).
# Pool: champion (resume) + its league-1 history + the old champion + bc_mix_A
# x3 + exploiter final x3 + exploiter interval-4 x1. No Trick Room boost.
# Same hyperparameters as rounds 0-2.
#
# Prereqs (manual): Showdown server on the port below --
#   cd pokemon-showdown && node pokemon-showdown start 7700 --no-security &
# Launch (house rules: AC power, lid open):
#   nohup caffeinate -is ./run_league2_training.sh > league2_$(date +%H%M%S).log 2>&1 &

.venv/bin/python training/build_league.py --config training/league3_config.json --verify-only || {
  echo "Training refused: league-2 pool failed verification. Run training/build_league.py --config training/league3_config.json first." >&2
  exit 2
}

exec .venv/bin/python -m vgc_bench.train \
  --fictitious_play \
  --reg mb \
  --run_id 1 \
  --our_team teams/reg_mb/our_team.txt \
  --knowledge_obs \
  --hidden_sheet_prob 0.50 \
  --team_weights data/team_weights_regmb_league3.json \
  --results_suffix league3 \
  --num_envs 8 \
  --num_eval_workers 8 \
  --port 7700 \
  --device mps \
  --learning_rate 0.00003 \
  --n_epochs 3 \
  --target_kl 0.02 \
  --total_steps 17694720
