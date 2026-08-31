#!/bin/bash
# Apple Silicon variant of eval.sh.
#
# Upstream fans 4 evals across cuda:0-3 concurrently; this Mac has one MPS device,
# so the team-count sweep runs sequentially instead. That sweep is the point of the
# benchmark: it measures how much play degrades as the agent must cover more teams.
#
# Usage: ./eval_mac.sh [reg] [team_counts...]      e.g. ./eval_mac.sh ma 1 4 16

reg="${1:-ma}"
shift 2>/dev/null
team_counts=("${@:-1 4 16 64}")
[ $# -eq 0 ] && team_counts=(1 4 16 64)
port=8010
device="${DEVICE:-mps}"

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

for num_teams in "${team_counts[@]}"; do
    echo "=== eval: reg=$reg num_teams=$num_teams device=$device ==="
    .venv/bin/python -m vgc_bench.eval \
        --reg "$reg" --num_teams "$num_teams" --port "$port" --device "$device" \
        > "debug_eval_${reg}_${num_teams}.log" 2>&1
    status=$?
    [ $status -ne 0 ] && echo "  eval (num_teams=$num_teams) died with status $status" \
                      || echo "  eval (num_teams=$num_teams) finished"
done

kill $showdown_pid 2>/dev/null
echo "All evals done."
