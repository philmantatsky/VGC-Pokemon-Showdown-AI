#!/usr/bin/env bash
# The matrix-search head-to-head (the user, 2026-10-04: "start the matrix search";
# pre-registered in PROJECT_STATUS 2026-10-04 before this ran): side A = the deployed
# T6ep + nash exact search (critic leaf, $WORLDS worlds, ${BUDGET}s, every move turn,
# sampled), side B = the deployed T6ep without search, both on T6e with the 14
# deployed guards. Searched games run one at a time per process, so $SHARDS processes
# with different seeds play $GAMES games each on one local server; the shards are then
# pooled (evaluation/pool_mirrors.py). DEPLOYED.json is not touched; no ladder.
# Usage (repo root, AC power): nohup ./tools/search_mirror_chain.sh > /dev/null 2>&1 &
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
if [ -z "${UNDER_CAFFEINATE:-}" ]; then UNDER_CAFFEINATE=1 exec caffeinate -is "$0" "$@"; fi
NAME=${NAME:-search_nash1}
SHARDS=${SHARDS:-4}
GAMES=${GAMES:-200}
WORLDS=${WORLDS:-4}
BUDGET=${BUDGET:-8}
PORT=${PORT:-7612}
# extra side-A search flags for a variant, e.g. "--a-search-argmax --a-search-prior-mix 0.5"
EXTRA=${EXTRA:-}
OUT=results_mirror_$NAME
LOG=${OUT}_chain.log
stamp() { date '+%m-%d %H:%M:%S'; }
say() { echo "[$(stamp)] $*" >> "$LOG"; }
up() { lsof -nP -iTCP:$1 -sTCP:LISTEN >/dev/null 2>&1; }
start_server() { up $1 && return 0; (cd pokemon-showdown && nohup node pokemon-showdown start $1 --no-security > /dev/null 2>&1 &); for _ in $(seq 1 60); do up $1 && return 0; sleep 1; done; return 1; }
stop_server() { for p in $(lsof -nP -t -iTCP:$1 -sTCP:LISTEN 2>/dev/null); do kill $p 2>/dev/null; done; pkill -f "pokemon-showdown start $1" 2>/dev/null; sleep 2; true; }
say "CHAIN_START shards=$SHARDS games=$GAMES worlds=$WORLDS budget=$BUDGET extra=$EXTRA"
start_server "$PORT" || { say "CHAIN_FAILED server"; exit 3; }
pids=()
runs=()
for k in $(seq 1 "$SHARDS"); do
  run="${OUT}_s$k"
  runs+=("$run")
  if grep -q '"complete": true' "$run/result.json" 2>/dev/null; then
    say "SHARD_DONE $k (skipped)"
    continue
  fi
  .venv/bin/python -u evaluation/mirror_guard_ab.py --a-search nash --a-search-leaf critic \
    --a-search-worlds "$WORLDS" --a-search-budget "$BUDGET" --concurrency 1 $EXTRA \
    --games "$GAMES" --seed $((20924 + 1000 * k)) --port "$PORT" --output "$run" \
    > "$run.log" 2>&1 &
  pids+=("$!")
  say "SHARD_START $k pid=$! -> $run"
done
rc=0
for pid in ${pids[@]+"${pids[@]}"}; do
  wait "$pid" || rc=$?
done
stop_server "$PORT"
say "SHARDS_END rc=$rc"
.venv/bin/python evaluation/pool_mirrors.py "${runs[@]}" --json "${OUT}_pooled.json" >> "$LOG" 2>&1 \
  && say "POOLED ${OUT}_pooled.json" || say "POOL_FAILED"
say "CHAIN_COMPLETE"
