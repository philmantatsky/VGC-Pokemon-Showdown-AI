#!/usr/bin/env bash
# The exact search against the held-out battery (2026-10-05): the deployed bot with and
# without the search against each of the battery's six opponents on its 47 held-out
# rosters (evaluation/search_roster_ab.py), one process and one server per opponent --
# the searching side plays one battle at a time. Cells are appended whole, so a
# relaunch with the same NAME continues where the last one stopped. Pooled when every
# opponent has finished. DEPLOYED.json is not touched; no ladder.
# Usage (AC power; it runs in the checkout it is called from):
#   NAME=rosters1 REPEATS=6 PORT=7630 \
#     EXTRA="--a-search-streams 4 --a-search-leaf-calibration results_leaf_calibration_T6ep/calibration.json" \
#     nohup ./tools/search_roster_arms.sh > /dev/null 2>&1 &
# POPULATIONS="frozen heuristic" plays a subset. `touch results_search_rosters_<NAME>/STOP`
# ends every process at its next cell (a roster x sheets x arm, a few minutes); what
# is finished stays, and a relaunch without the file resumes.
# Never edit this file while it runs: bash reads a running script by offset.
set -uo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."
if [ -z "${UNDER_CAFFEINATE:-}" ]; then UNDER_CAFFEINATE=1 exec caffeinate -is "$SELF" "$@"; fi
NAME=${NAME:?set NAME}
REPEATS=${REPEATS:-11}
PORT=${PORT:-7630}
EXTRA=${EXTRA:-}
POPULATIONS=${POPULATIONS:-human_new frozen rotation1 rotation2 human_previous heuristic}
OUT=results_search_rosters_$NAME
LOG=$OUT/arms.log
mkdir -p "$OUT"
stamp() { date '+%m-%d %H:%M:%S'; }
say() { echo "[$(stamp)] $*" >> "$LOG"; }
up() { lsof -nP -iTCP:$1 -sTCP:LISTEN >/dev/null 2>&1; }
start_server() { up $1 && return 0; (cd pokemon-showdown && nohup node pokemon-showdown start $1 --no-security > /dev/null 2>&1 &); for _ in $(seq 1 60); do up $1 && return 0; sleep 1; done; return 1; }
stop_server() { for p in $(lsof -nP -t -iTCP:$1 -sTCP:LISTEN 2>/dev/null); do kill $p 2>/dev/null; done; pkill -f "pokemon-showdown start $1" 2>/dev/null; sleep 2; true; }
# "<checkpoint or heuristic> <1 if the opponent plays its top action>" for a population
opponent_of() {
  .venv/bin/python - "$1" <<'PY'
import sys
sys.path.insert(0, ".")
from evaluation.run_t6_confirmation import OPPONENTS
path, deterministic = OPPONENTS[sys.argv[1]]
print(path, int(deterministic))
PY
}
say "ARMS_START repeats=$REPEATS port=$PORT populations=$POPULATIONS extra=$EXTRA at $(pwd) $(git rev-parse --short HEAD 2>/dev/null)"
pids=()
ports=()
runs=()
k=0
for pop in $POPULATIONS; do
  k=$((k + 1))
  port=$((PORT + k))
  read -r opponent deterministic <<< "$(opponent_of "$pop")"
  [ -n "${opponent:-}" ] || { say "ARM_FAILED $pop is not one of the battery's opponents"; continue; }
  start_server "$port" || { say "ARM_FAILED $pop server on $port"; continue; }
  flag=()
  [ "$deterministic" = 1 ] && flag=(--deterministic-opponent)
  .venv/bin/python -u evaluation/search_roster_ab.py --opponent "$opponent" ${flag[@]+"${flag[@]}"} \
    --output "$OUT/$pop" --repeats "$REPEATS" --port "$port" $EXTRA \
    >> "$OUT/$pop.log" 2>&1 &
  pids+=("$!")
  ports+=("$port")
  runs+=("$OUT/$pop")
  say "ARM_START $pop pid=$! port=$port opponent=$opponent"
  # one of six processes started in the same second died without a trace on 10-04
  sleep 5
done
rc=0
i=0
while [ "$i" -lt "${#pids[@]}" ]; do
  wait "${pids[$i]}" || { rc=$?; say "ARM_EXIT pid=${pids[$i]} rc=$rc (${runs[$i]})"; }
  stop_server "${ports[$i]}"
  i=$((i + 1))
done
done_runs=()
for run in ${runs[@]+"${runs[@]}"}; do
  grep -q '"complete": true' "$run/result.json" 2>/dev/null && done_runs+=("$run")
done
if [ ${#done_runs[@]} -gt 0 ]; then
  .venv/bin/python evaluation/search_roster_ab.py --pool "${done_runs[@]}" --json "$OUT/pooled.json" > /dev/null 2>> "$LOG" \
    && say "POOLED ${#done_runs[@]} opponents -> $OUT/pooled.json: $(.venv/bin/python -c "
import json; d = json.load(open('$OUT/pooled.json'))['search_minus_plain']['overall']
print(f\"search - plain = {100 * d['delta']:+.1f} points [{100 * d['bootstrap_95'][0]:+.1f}, {100 * d['bootstrap_95'][1]:+.1f}] over {d['rosters']} rosters, {d['games_per_arm']} games an arm ({100 * d['plain_win_rate']:.1f}% -> {100 * d['search_win_rate']:.1f}%)\")")" \
    || say "POOL_FAILED"
else
  say "POOL_SKIPPED nothing complete"
fi
say "ARMS_COMPLETE rc=$rc"
