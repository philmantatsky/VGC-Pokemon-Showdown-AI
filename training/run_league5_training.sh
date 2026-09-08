#!/usr/bin/env bash
set -euo pipefail

# League round 4 (2026-09-07): the current-meta round -- pool copies from the meta-game equilibrium (make_league5_config.py), September human clones, exploiter capped, TR rosters boosted in the opponent team pool. Round 3 (exploiter at
# 31% of the pool) flipped the exploit (+28..+32pp) but both finalists paid
# 2-6pp on the general arms -- overfitting to the sparring partner. Same pool
# otherwise (deployed champion at resume, league-1 history, old champion,
# bc_mix_A x3) with ONE copy of the final exploiter (~10% initial, decaying).
#
# Prereqs (manual): Showdown server on the port below --
#   cd pokemon-showdown && node pokemon-showdown start 7700 --no-security &
# Launch (house rules: AC power, lid open):
#   nohup caffeinate -is ./run_league2_training.sh > league2_$(date +%H%M%S).log 2>&1 &

.venv/bin/python training/build_league.py --config training/league5_config.json --verify-only || {
  echo "Training refused: league-2 pool failed verification. Run training/build_league.py --config training/league5_config.json first." >&2
  exit 2
}

exec .venv/bin/python -m vgc_bench.train \
  --fictitious_play \
  --reg mb \
  --run_id 1 \
  --our_team teams/reg_mb/our_team.txt \
  --knowledge_obs \
  --hidden_sheet_prob 0.50 \
  --team_weights data/team_weights_regmb_league5.json \
  --results_suffix league5 \
  --num_envs 8 \
  --num_eval_workers 8 \
  --port 7700 \
  --device mps \
  --learning_rate 0.00003 \
  --n_epochs 3 \
  --target_kl 0.02 \
  --total_steps 17694720
