#!/usr/bin/env bash
# Run one eval_counterfactual invocation under a stall watchdog.
#   supervised_eval.sh <output.json> <port> -- <eval args...>
# Skips if the output exists. Kills and retries (fresh server) when the eval's
# log has not grown for STALL_MIN minutes (frozen-battle class, 2026-09-08).
set -uo pipefail
OUT=$1; PORT=$2; shift 2; [ "${1:-}" = "--" ] && shift
STALL_MIN=${STALL_MIN:-20}; RETRIES=${RETRIES:-2}
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
[ -f "$OUT" ] && { echo "SKIP $OUT exists"; exit 0; }
mkdir -p "$(dirname "$OUT")"
LOG="${OUT%.json}.log"
restart_server() {
  if pids=$(lsof -nP -t -iTCP:$PORT -sTCP:LISTEN 2>/dev/null); then kill $pids 2>/dev/null || true; sleep 3; fi
  (cd pokemon-showdown && node pokemon-showdown start $PORT --no-security > /dev/null 2>&1 &)
  for _ in $(seq 1 30); do lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && return 0; sleep 1; done
  echo "SERVER_FAILED $PORT"; return 1
}
for attempt in $(seq 0 $RETRIES); do
  restart_server || exit 2
  : > "$LOG"
  .venv/bin/python evaluation/eval_counterfactual.py "$@" --port "$PORT" --output "$OUT" >> "$LOG" 2>&1 &
  PID=$!
  last_size=0; stalled_for=0
  while kill -0 $PID 2>/dev/null; do
    sleep 60
    size=$(stat -f %z "$LOG" 2>/dev/null || echo 0)
    if [ "$size" -gt "$last_size" ]; then last_size=$size; stalled_for=0; else stalled_for=$((stalled_for + 1)); fi
    if [ "$stalled_for" -ge "$STALL_MIN" ]; then
      echo "STALL_KILL attempt=$attempt $OUT [$(date '+%H:%M:%S')]"; kill $PID 2>/dev/null; sleep 5; kill -9 $PID 2>/dev/null; break
    fi
  done
  wait $PID 2>/dev/null; rc=$?
  if [ -f "$OUT" ]; then echo "DONE $OUT attempt=$attempt"; exit 0; fi
  echo "RETRY $OUT after attempt=$attempt (rc=$rc)"
done
echo "GAVE_UP $OUT"; exit 3
