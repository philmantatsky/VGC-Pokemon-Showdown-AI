#!/usr/bin/env bash
set -euo pipefail
trap 'echo "CHAIN_FAILED at line $LINENO (exit $?)"' ERR

# League-3 gate chain for one candidate checkpoint (pre-registered 2026-09-05):
#   1. exploit re-measure: candidate vs the final exploiter (stochastic), n=1,000
#      -- the champion read 39.8%; the candidate must reach >= 44.8% (+5pp)
#   2. standard 5-arm screening battery vs the DEPLOYED league champion
#      -- no arm below baseline by >2pp, weighted >= 0, memorization clean
#   3. mix_A memorization diagnostic
# Usage: ./evaluation/run_league3_gates.sh <candidate.zip> <label>
# Restarts the eval server first (stale rooms from a killed run stall the next).

CAND=${1:?candidate checkpoint}
LABEL=${2:?label}
BASE=results_league/league_champion.zip
EXP=results_exploiter/saves_ex_hs_wt/reg_mb/seed1/17694720.zip
PORT=7600
OUT=results_gate_battery_league3/$LABEL
mkdir -p "$OUT"

if pids=$(lsof -nP -t -iTCP:$PORT -sTCP:LISTEN 2>/dev/null); then kill $pids 2>/dev/null || true; sleep 3; fi
(cd pokemon-showdown && node pokemon-showdown start $PORT --no-security > /dev/null 2>&1 &)
for _ in $(seq 1 30); do lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && break; sleep 1; done
lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 || { echo "CHAIN_FAILED: eval server on $PORT did not start"; exit 2; }

echo "[$(date '+%H:%M')] 1/3 exploit re-measure ($LABEL)"
.venv/bin/python evaluation/eval_counterfactual.py \
  --baseline $BASE --candidate "$CAND" \
  --opponent-checkpoint $EXP --opponent-stochastic \
  --n-battles 1000 --hidden-sheets --seed 83 --port $PORT --workers 8 \
  --output "$OUT/exploit_remeasure_1000.json"

echo "[$(date '+%H:%M')] 2/3 screening battery vs deployed champion ($LABEL)"
.venv/bin/python evaluation/run_gate_battery.py --baseline $BASE --candidate "$CAND" \
  --tier screening --port $PORT --out-dir "$OUT"

echo "[$(date '+%H:%M')] 3/3 mix_A memorization diagnostic ($LABEL)"
.venv/bin/python evaluation/run_gate_battery.py --baseline $BASE --candidate "$CAND" \
  --tier screening --arms human_bc --human-bc results_bc/mix_A/saves_bc/seed1/30.zip \
  --port $PORT --out-dir "${OUT}_mixA"
echo "L3_GATES_COMPLETE $LABEL"
