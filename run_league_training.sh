#!/usr/bin/env bash
set -euo pipefail

# League fine-tune of the deployed champion: same flags as the champion run
# except the opponent pool (build_league.py seeds bc_mix_A at 3/8 share) and a
# league team-weights file that stops double-counting our_team.txt. Refuses to
# run until the league pool passes every content/role check.
#
# Prereqs (manual): Showdown server on the port below --
#   cd pokemon-showdown && node pokemon-showdown start 7700 --no-security &
# Launch (house rules: AC power, lid open):
#   nohup caffeinate -is ./run_league_training.sh > league_$(date +%H%M%S).log 2>&1 &

.venv/bin/python build_league.py --verify-only || {
  echo "Training refused: league pool failed verification. Run build_league.py first." >&2
  exit 2
}

exec .venv/bin/python -m vgc_bench.train \
  --fictitious_play \
  --reg mb \
  --run_id 1 \
  --our_team teams/reg_mb/our_team.txt \
  --knowledge_obs \
  --hidden_sheet_prob 0.50 \
  --team_weights data/team_weights_regmb_league.json \
  --results_suffix league \
  --num_envs 8 \
  --num_eval_workers 8 \
  --port 7700 \
  --device mps \
  --learning_rate 0.00003 \
  --n_epochs 3 \
  --target_kl 0.02 \
  --total_steps 12779520
