#!/usr/bin/env bash
# Confirmation reads for the team pick (pre-registered 2026-09-19): one PILOT
# alone on each named team against the eval-only Reg M-C human clone, fresh
# seed, under the stall watchdog. Output: <out_dir>/confirm_<team>.json
# (arm champion_policy = the pilot). Completed reads are skipped.
# Usage: HUMAN_BC=<eval_mcB ckpt> ./evaluation/run_team_confirmation.sh <pilot.zip> <out_dir> <n> <seed> <team files...>
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
PILOT=$1; OUT=$2; N=$3; SEED=$4; shift 4
PORT=${PORT:-7600}
HUMAN_BC=${HUMAN_BC:?eval-only Reg M-C clone checkpoint}
mkdir -p "$OUT"
for T in "$@"; do
  L=$(basename "$T" .txt)
  echo "== confirm $L [$(date '+%H:%M:%S')]"
  ./evaluation/supervised_eval.sh "$OUT/confirm_$L.json" $PORT -- \
    --baseline "$PILOT" --candidate "$PILOT" --baseline-only \
    --reg mc --team-weights data/team_weights_regmc.json --our-team "$T" \
    --opponent-checkpoint "$HUMAN_BC" --opponent-stochastic \
    --n-battles "$N" --hidden-sheets --seed "$SEED" --workers 8
done
echo "CONFIRM_DONE [$(date '+%H:%M:%S')]"
