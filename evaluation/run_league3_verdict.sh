#!/usr/bin/env bash
set -euo pipefail
trap 'echo "CHAIN_FAILED at line $LINENO (exit $?)"' ERR

# League-3 verdict orchestrator (pre-registered bars, 2026-09-05):
#   step 1 for EVERY finalist: exploit re-measure vs the final exploiter,
#           n=1,000 stochastic -- champion read 39.8%; a finalist qualifies at
#           >= 44.8% (+5pp). Cheap (~20 min), so it runs first for all.
#   step 2 for QUALIFIERS only: 5-arm screening battery vs the DEPLOYED
#           champion (no arm below by >2pp, weighted >= 0) + mix_A diagnostic.
# Usage: ./evaluation/run_league3_verdict.sh <ckpt> [<ckpt> ...]

BASE=results_league/league_champion.zip
EXP=results_exploiter/saves_ex_hs_wt/reg_mb/seed1/17694720.zip
PORT=7600
ROOT=${VERDICT_ROOT:-results_gate_battery_league3}
mkdir -p "$ROOT"

if pids=$(lsof -nP -t -iTCP:$PORT -sTCP:LISTEN 2>/dev/null); then kill $pids 2>/dev/null || true; sleep 3; fi
(cd pokemon-showdown && node pokemon-showdown start $PORT --no-security > /dev/null 2>&1 &)
for _ in $(seq 1 30); do lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && break; sleep 1; done
lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 || { echo "CHAIN_FAILED: eval server on $PORT did not start"; exit 2; }

for CAND in "$@"; do
  LABEL=$(basename "$CAND" .zip)
  mkdir -p "$ROOT/$LABEL"
  echo "[$(date '+%H:%M')] exploit re-measure: $LABEL"
  .venv/bin/python evaluation/eval_counterfactual.py \
    --baseline $BASE --candidate "$CAND" \
    --opponent-checkpoint $EXP --opponent-stochastic \
    --n-battles 1000 --hidden-sheets --seed 83 --port $PORT --workers 8 \
    --output "$ROOT/$LABEL/exploit_remeasure_1000.json"
done

for CAND in "$@"; do
  LABEL=$(basename "$CAND" .zip)
  RATE=$(.venv/bin/python - "$ROOT/$LABEL/exploit_remeasure_1000.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"{d['arms']['distilled_policy']['win_rate']:.4f}")
PY
)
  BASE_RATE=$(.venv/bin/python - "$ROOT/$LABEL/exploit_remeasure_1000.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"{d['arms']['champion_policy']['win_rate']:.4f}")
PY
)
  echo "EXPLOIT_REMEASURE $LABEL candidate=$RATE deployed=$BASE_RATE"
  if .venv/bin/python -c "import sys; sys.exit(0 if float('$RATE') >= 0.448 else 1)"; then
    echo "[$(date '+%H:%M')] $LABEL qualifies (>= 44.8%): screening battery vs deployed champion"
    .venv/bin/python evaluation/run_gate_battery.py --baseline $BASE --candidate "$CAND" \
      --tier screening --port $PORT --out-dir "$ROOT/$LABEL"
    echo "[$(date '+%H:%M')] $LABEL mix_A memorization diagnostic"
    .venv/bin/python evaluation/run_gate_battery.py --baseline $BASE --candidate "$CAND" \
      --tier screening --arms human_bc --human-bc results_bc/mix_A/saves_bc/seed1/30.zip \
      --port $PORT --out-dir "$ROOT/${LABEL}_mixA"
    echo "GATES_DONE $LABEL"
  else
    echo "EXPLOIT_TARGET_MISSED $LABEL (candidate=$RATE < 0.448): battery skipped"
  fi
done
echo "L3_VERDICT_COMPLETE"
