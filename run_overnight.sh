#!/bin/bash
# Overnight self-play run that survives an idle Mac.
#
# macOS suspends processes during sleep, which silently stalls training. `caffeinate -is`
# blocks idle + system sleep for the lifetime of this command only; settings revert when
# it exits, so nothing is changed permanently.
#
# IMPORTANT: caffeinate does NOT override lid-close sleep. Leave the lid OPEN (the display
# may sleep, that's fine), or use clamshell mode with an external display on AC power.
#
# Usage: ./run_overnight.sh [total_steps] [num_teams] [style] [reg]
#   e.g. ./run_overnight.sh 10000000 4 self_play mb

total_steps="${1:-10000000}"
num_teams="${2:-4}"
style="${3:-self_play}"
reg="${4:-mb}"

stamp=$(date +%Y%m%d_%H%M%S)
log="overnight_${reg}_${num_teams}teams_${stamp}.log"

if [ "$(pmset -g ps | head -1 | grep -c 'AC Power')" -eq 0 ]; then
    echo "WARNING: not on AC power. On battery this Mac idle-sleeps after 1 minute."
    echo "Plug in before starting a long run. Continuing in 10s (Ctrl-C to abort)..."
    sleep 10
fi

echo "Logging to $log"
echo "Sleep blocked for the duration of this run (lid must stay open)."

caffeinate -is .venv/bin/python -m vgc_bench.train \
    --run_id 1 \
    --reg "$reg" \
    --num_teams "$num_teams" \
    --num_envs "${NUM_ENVS:-8}" \
    --num_eval_workers "${NUM_ENVS:-8}" \
    --port "${PORT:-7200}" \
    --device "${DEVICE:-mps}" \
    --behavior_clone \
    --"$style" \
    --no_teampreview \
    --results_suffix "${reg}_${num_teams}teams" \
    --total_steps "$total_steps" \
    >> "$log" 2>&1

status=$?
echo "finished with status $status at $(date)" | tee -a "$log"
