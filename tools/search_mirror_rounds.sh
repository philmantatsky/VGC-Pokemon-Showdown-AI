#!/usr/bin/env bash
# A long search head-to-head as several rounds of parallel shards (2026-10-04).
# tools/search_mirror_chain.sh plays one round; a shard that dies there takes its whole
# share of the games with it (one of six did, a minute after launch, with no traceback).
# Here each round is short and has its own seeds, a dead shard costs only that round's
# share, and everything that finished is pooled at the end (and after every round).
# Side A = the deployed bot + nash exact search (critic leaf); side B = the deployed bot.
# DEPLOYED.json is not touched; no ladder.
# Usage (repo root, AC power):
#   NAME=search_nash5 ROUNDS=4 SHARDS=3 GAMES=250 PORT=7612 \
#     EXTRA="--a-search-argmax --a-search-anchor 0.07" \
#     nohup ./tools/search_mirror_rounds.sh > /dev/null 2>&1 &
# Never edit this file while it runs: bash reads a running script by offset.
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
if [ -z "${UNDER_CAFFEINATE:-}" ]; then UNDER_CAFFEINATE=1 exec caffeinate -is "$0" "$@"; fi
NAME=${NAME:?set NAME}
ROUNDS=${ROUNDS:-4}
SHARDS=${SHARDS:-3}
GAMES=${GAMES:-250}
WORLDS=${WORLDS:-4}
BUDGET=${BUDGET:-8}
PORT=${PORT:-7612}
EXTRA=${EXTRA:-}
OUT=results_mirror_$NAME
LOG=${OUT}_rounds.log
stamp() { date '+%m-%d %H:%M:%S'; }
say() { echo "[$(stamp)] $*" >> "$LOG"; }
up() { lsof -nP -iTCP:$1 -sTCP:LISTEN >/dev/null 2>&1; }
start_server() { up $1 && return 0; (cd pokemon-showdown && nohup node pokemon-showdown start $1 --no-security > /dev/null 2>&1 &); for _ in $(seq 1 60); do up $1 && return 0; sleep 1; done; return 1; }
stop_server() { for p in $(lsof -nP -t -iTCP:$1 -sTCP:LISTEN 2>/dev/null); do kill $p 2>/dev/null; done; pkill -f "pokemon-showdown start $1" 2>/dev/null; sleep 2; true; }
pool() {
  local done_runs=()
  for run in "${OUT}"_r*_s*; do
    [ -d "$run" ] && grep -q '"complete": true' "$run/result.json" 2>/dev/null && done_runs+=("$run")
  done
  [ ${#done_runs[@]} -gt 0 ] || { say "POOL_SKIPPED nothing complete"; return 0; }
  .venv/bin/python evaluation/pool_mirrors.py "${done_runs[@]}" --json "${OUT}_pooled.json" > /dev/null 2>> "$LOG" \
    && say "POOLED ${#done_runs[@]} shards -> ${OUT}_pooled.json: $(.venv/bin/python -c "
import json; d = json.load(open('${OUT}_pooled.json'))
print(f\"A {d['a_wins']:.1f}/{d['games']} = {100 * d['a_win_rate']:.1f}% [{100 * d['wilson_95'][0]:.1f}, {100 * d['wilson_95'][1]:.1f}]\")")" \
    || say "POOL_FAILED"
}
say "ROUNDS_START rounds=$ROUNDS shards=$SHARDS games=$GAMES worlds=$WORLDS budget=$BUDGET port=$PORT extra=$EXTRA"
for r in $(seq 1 "$ROUNDS"); do
  [ -e "${OUT}_STOP" ] && { say "STOP file found before round $r"; break; }
  start_server "$PORT" || { say "ROUND_FAILED $r server"; break; }
  pids=()
  for k in $(seq 1 "$SHARDS"); do
    run="${OUT}_r${r}_s$k"
    if grep -q '"complete": true' "$run/result.json" 2>/dev/null; then
      say "SHARD_DONE r$r s$k (skipped)"
      continue
    fi
    # an unfinished shard from an earlier launch: keep it, out of the way (its
    # decision log would otherwise be appended to)
    if [ -e "$run" ] || [ -e "$run.log" ]; then
      aside="${run}_unfinished_$(date +%s)"
      mkdir -p "$aside" && mv "$run" "$run.log" "$aside"/ 2>/dev/null
      say "SHARD_SET_ASIDE r$r s$k -> $aside"
    fi
    .venv/bin/python -u evaluation/mirror_guard_ab.py --a-search nash --a-search-leaf critic \
      --a-search-worlds "$WORLDS" --a-search-budget "$BUDGET" --concurrency 1 $EXTRA \
      --games "$GAMES" --seed $((20924 + 100000 * r + 1000 * k)) --port "$PORT" --output "$run" \
      > "$run.log" 2>&1 &
    pids+=("$!")
    say "SHARD_START r$r s$k pid=$! -> $run"
    # stagger the launches: the shard that died on 10-04 was one of six started in
    # the same second
    sleep 5
  done
  rc=0
  for pid in ${pids[@]+"${pids[@]}"}; do
    wait "$pid" || { rc=$?; say "SHARD_EXIT pid=$pid rc=$rc"; }
  done
  stop_server "$PORT"
  say "ROUND_END $r rc=$rc"
  pool
done
say "ROUNDS_COMPLETE"
