#!/usr/bin/env bash
set -euo pipefail
trap 'echo "CHAIN_FAILED at line $LINENO (exit $?)"' ERR

# Opt-in guard probe (pre-registered 2026-09-06): the DEPLOYED brain with one
# extra guard enabled (--candidate-guards) vs the DEPLOYED brain as-is, paired.
#   step 1  exploit meter: n=1,000 vs the final exploiter (stochastic), informative.
#   step 2  the 5-arm screening battery vs the deployed brain, judged by
#           scorecard_verdict.py (advisory tier + confirmation clause).
#   Before believing anything: the candidate arm's telemetry must show the guard
#   firing (guard_fire_counts[<guard>] > 0) and no <guard>_error counts.
# Usage: ./evaluation/run_guard_probe.sh <guard_name>

GUARD=${1:?guard name}
BASE=results_league/league_champion.zip
EXP=results_exploiter/saves_ex_hs_wt/reg_mb/seed1/17694720.zip
PORT=7600
ROOT=results_guard_probe/$GUARD
mkdir -p "$ROOT"

if pids=$(lsof -nP -t -iTCP:$PORT -sTCP:LISTEN 2>/dev/null); then kill $pids 2>/dev/null || true; sleep 3; fi
(cd pokemon-showdown && node pokemon-showdown start $PORT --no-security > /dev/null 2>&1 &)
for _ in $(seq 1 30); do lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && break; sleep 1; done
lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 || { echo "CHAIN_FAILED: eval server on $PORT did not start"; exit 2; }

echo "[$(date '+%H:%M')] step 1: exploit meter, guard=$GUARD"
.venv/bin/python evaluation/eval_counterfactual.py \
  --baseline $BASE --candidate $BASE \
  --opponent-checkpoint $EXP --opponent-stochastic \
  --n-battles 1000 --hidden-sheets --seed 83 --port $PORT --workers 8 \
  --candidate-guards "$GUARD" --output "$ROOT/exploit_1000.json"
.venv/bin/python - "$ROOT/exploit_1000.json" "$GUARD" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); g = sys.argv[2]
c, b = d["arms"]["distilled_policy"], d["arms"]["champion_policy"]
t = c.get("telemetry", {})
print(f"EXPLOIT_GUARD guard={g} candidate={c['win_rate']:.4f} deployed={b['win_rate']:.4f} "
      f"delta={100*(c['win_rate']-b['win_rate']):+.1f}pp fired={t.get(g, 0)} "
      f"injected={t.get(g + ':injected:demoted', 0)} promoted={t.get(g + ':promoted:demoted', 0)} "
      f"errors={t.get(g + '_error', 0)} mismatched={d.get('opponent_preview_pairing', {}).get('mismatched')}")
PY

echo "[$(date '+%H:%M')] step 2: screening battery (candidate = deployed + $GUARD)"
.venv/bin/python evaluation/run_gate_battery.py --baseline $BASE --candidate $BASE \
  --tier screening --port $PORT --out-dir "$ROOT" --candidate-guards "$GUARD"
.venv/bin/python evaluation/scorecard_verdict.py "$ROOT/screening" --json "$ROOT/screening/verdict.json"
echo "GUARD_PROBE_COMPLETE $ROOT"
