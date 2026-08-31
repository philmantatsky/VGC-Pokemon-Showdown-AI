#!/bin/bash
# Apple Silicon self-play training.
#
# Upstream train.sh runs 4 concurrent jobs (cuda:0-3), 24 envs each, for 98,304,000
# steps. That is a multi-GPU budget. Here we run one job sized for this machine and
# default to a short horizon so there is early signal rather than a multi-day
# commitment. Checkpoints land every 983,040 steps regardless, and training resumes
# from the newest checkpoint automatically, so a short run can simply be re-run with a
# larger --total_steps to continue.
#
# Usage: ./train_mac.sh [total_steps] [num_teams] [style]
#   style: self_play (default) | fictitious_play | double_oracle

total_steps="${1:-1966080}"     # 2 checkpoint intervals
num_teams="${2:-1}"
style="${3:-self_play}"
run_id=1
port=7200
device="${DEVICE:-mps}"
num_envs="${NUM_ENVS:-8}"

start_showdown() {
    (
        cd pokemon-showdown
        node pokemon-showdown start "$1" --no-security > /dev/null 2>&1 &
        echo $!
    )
}

echo "Starting Showdown server on port $port..."
showdown_pid=$(start_showdown "$port")
sleep 5
echo "Self-play: style=$style teams=$num_teams envs=$num_envs device=$device steps=$total_steps"
.venv/bin/python -m vgc_bench.train \
    --run_id $run_id \
    --num_teams "$num_teams" \
    --num_envs "$num_envs" \
    --num_eval_workers "$num_envs" \
    --port $port \
    --device "$device" \
    --behavior_clone \
    --"$style" \
    --total_steps "$total_steps"
status=$?
[ $status -ne 0 ] && echo "Training died with exit status $status" || echo "Training finished!"
kill $showdown_pid 2>/dev/null
