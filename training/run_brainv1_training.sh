#!/usr/bin/env bash
set -euo pipefail

# Brain v1 training round (NEW_BRAIN_PLAN §3, M3): the deployed lineage continued
# with the joint-action head, the threat block and potential-based shaping, on
# Reg M-C data, TEAM-AGNOSTIC -- our side is drawn from the candidate teams
# (teams/candidates_mc/T0..T5) so the team tournament that follows is fair.
#
# The pool directory is built by build_league.py from training/brainv1_config.json
# (written by the launcher that knows the round-6 finalist); older checkpoints in
# the pool need no conversion -- every load path upgrades them in memory.
#
# Env: INIT_STEM (resume stem in the pool dir), TOTAL_STEPS (default +8 intervals),
#      OUR_TEAMS (comma list; default all candidates), SHAPING_FAINT/SHAPING_HP.
# Prereqs: Showdown server on 7700. Launch under caffeinate, AC power, lid open.

CONFIG=${CONFIG:-training/brainv1_config.json}
TOTAL_STEPS=${TOTAL_STEPS:-20643840}
OUR_TEAMS=${OUR_TEAMS:-$(ls teams/candidates_mc/T*.txt | paste -sd, -)}
SHAPING_FAINT=${SHAPING_FAINT:-0.10}
SHAPING_HP=${SHAPING_HP:-0.05}

.venv/bin/python training/build_league.py --config "$CONFIG" --verify-only || {
  echo "Training refused: brain-v1 pool failed verification. Run training/build_league.py --config $CONFIG first." >&2
  exit 2
}

exec .venv/bin/python -u -m vgc_bench.train \
  --fictitious_play \
  --reg mc \
  --run_id 1 \
  --our_team "$OUR_TEAMS" \
  --joint_head \
  --shaping_faint "$SHAPING_FAINT" \
  --shaping_hp "$SHAPING_HP" \
  --knowledge_obs \
  --hidden_sheet_prob 0.50 \
  --team_weights data/team_weights_regmc_brainv1.json \
  --results_suffix brainv1 \
  --num_envs 8 \
  --num_eval_workers 8 \
  --port 7700 \
  --device mps \
  --learning_rate 0.00003 \
  --n_epochs 3 \
  --target_kl 0.02 \
  --total_steps "$TOTAL_STEPS"
