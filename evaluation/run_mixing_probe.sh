#!/usr/bin/env bash
set -euo pipefail
trap 'echo "CHAIN_FAILED at line $LINENO (exit $?)"' ERR

# Mixed-strategy probe (pre-registered 2026-09-06, PROJECT_STATUS):
#   the DEPLOYED brain with --mixing opening (top-3, T=1, preview + turns 1-2)
#   vs the DEPLOYED brain as-is, paired.
#   step 1  exploitability meter: n=1,000 vs the final exploiter (stochastic).
#           Deployed reads ~40%. Success bar: mixed arm >= deployed + 5pp.
#   step 2  cost side: the 5-arm screening battery vs the deployed brain
#           (candidate = deployed + mixing), judged by scorecard_verdict.py
#           (advisory tier: borderline arms go to a fresh-seed confirmation).
#   Before believing anything: the mixed arm's telemetry must show
#   mixing_ran > 0 and mixing_changed_pick > 0 (silence is not success).
# Usage: ./evaluation/run_mixing_probe.sh [mode] [top_k] [temperature] [last_turn]

MODE=${1:-opening}; TOPK=${2:-3}; TEMP=${3:-1.0}; LAST=${4:-2}
BASE=results_league/league_champion.zip
EXP=results_exploiter/saves_ex_hs_wt/reg_mb/seed1/17694720.zip
PORT=7600
ROOT=results_mixing_probe/${MODE}_k${TOPK}_t${TEMP}_l${LAST}
mkdir -p "$ROOT"

if pids=$(lsof -nP -t -iTCP:$PORT -sTCP:LISTEN 2>/dev/null); then kill $pids 2>/dev/null || true; sleep 3; fi
(cd pokemon-showdown && node pokemon-showdown start $PORT --no-security > /dev/null 2>&1 &)
for _ in $(seq 1 30); do lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && break; sleep 1; done
lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 || { echo "CHAIN_FAILED: eval server on $PORT did not start"; exit 2; }

MIX=(--candidate-mixing "$MODE" --mixing-top-k "$TOPK" --mixing-temperature "$TEMP" --mixing-last-turn "$LAST")

echo "[$(date '+%H:%M')] step 1: exploit re-measure, mixing=$MODE k=$TOPK T=$TEMP last=$LAST"
.venv/bin/python evaluation/eval_counterfactual.py \
  --baseline $BASE --candidate $BASE \
  --opponent-checkpoint $EXP --opponent-stochastic \
  --n-battles 1000 --hidden-sheets --seed 83 --port $PORT --workers 8 \
  "${MIX[@]}" --output "$ROOT/exploit_1000.json"
.venv/bin/python - "$ROOT/exploit_1000.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
c, b = d["arms"]["distilled_policy"], d["arms"]["champion_policy"]
t = c.get("telemetry", {})
print(f"EXPLOIT_MIXING mixed={c['win_rate']:.4f} deployed={b['win_rate']:.4f} "
      f"delta={100*(c['win_rate']-b['win_rate']):+.1f}pp "
      f"mixing_ran={t.get('mixing_ran', 0)} changed={t.get('mixing_changed_pick', 0)} "
      f"mismatched={d.get('opponent_preview_pairing', {}).get('mismatched')}")
PY

echo "[$(date '+%H:%M')] step 2: screening battery (candidate = deployed + mixing)"
.venv/bin/python evaluation/run_gate_battery.py --baseline $BASE --candidate $BASE \
  --tier screening --port $PORT --out-dir "$ROOT" "${MIX[@]}"
.venv/bin/python evaluation/scorecard_verdict.py "$ROOT/screening" --json "$ROOT/screening/verdict.json"
echo "MIXING_PROBE_COMPLETE $ROOT"
