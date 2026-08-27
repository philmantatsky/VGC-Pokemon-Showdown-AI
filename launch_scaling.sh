#!/bin/bash
# Overnight scaling run: our fixed team (MB430) vs varied opponent teams.
#
# This is the ladder objective. Warm-starts from the Reg M-B baseline's newest
# checkpoint, staged as 100.zip so --behavior_clone finds ours instead of silently
# downloading the researchers' model from HuggingFace.
#
# Usage: ./launch_scaling.sh [num_opponent_teams] [total_steps]

set -e
OPP="${1:-8}"
STEPS="${2:-10000000}"
SRC_DIR="results_mb_1teams/saves_bc_sp/reg_mb/1_teams/seed1"
SUFFIX="mb_ourteam_${OPP}opp"
DST_DIR="results_${SUFFIX}/saves_bc_sp/reg_mb/${OPP}_teams/seed1"

newest=$(ls "$SRC_DIR"/*.zip 2>/dev/null | sed 's/.*\///;s/\.zip//' | sort -n | tail -1)
if [ -z "$newest" ]; then
    echo "ERROR: no baseline checkpoint in $SRC_DIR"; exit 1
fi
echo "warm-starting from $SRC_DIR/${newest}.zip"

mkdir -p "$DST_DIR"
cp "$SRC_DIR/${newest}.zip" "$DST_DIR/100.zip"

pkill -f "pokemon-showdown start 7200" 2>/dev/null || true
sleep 2
(cd pokemon-showdown && node pokemon-showdown start 7200 --no-security > /dev/null 2>&1 &)
sleep 5

log="scaling_${SUFFIX}_$(date +%Y%m%d_%H%M%S).log"
echo "our team: teams/reg_mb/our_team.txt | opponents: $OPP teams | steps: $STEPS"
echo "logging to $log"

nohup caffeinate -is .venv/bin/python -m vgc_bench.train \
    --run_id 1 \
    --reg mb \
    --our_team teams/reg_mb/our_team.txt \
    --num_teams "$OPP" \
    --num_envs 8 \
    --num_eval_workers 8 \
    --port 7200 \
    --device mps \
    --behavior_clone \
    --self_play \
    --results_suffix "$SUFFIX" \
    --total_steps "$STEPS" \
    > "$log" 2>&1 &

sleep 60
if grep -qi "Downloading" "$log"; then
    echo "!! WARNING: downloaded their model instead of ours -- check method dir naming"
else
    echo "OK: warm-started from our checkpoint"
fi
tail -3 "$log" | grep -viE 'warn|deprecat' || true
