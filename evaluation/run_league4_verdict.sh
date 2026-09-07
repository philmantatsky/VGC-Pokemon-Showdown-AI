#!/usr/bin/env bash
set -euo pipefail
trap 'echo "CHAIN_FAILED at line $LINENO (exit $?)"' ERR

# Round-4 verdict: the league-3 chain (exploit re-measure for every finalist,
# 5-arm screening battery + mix_A diagnostic for qualifiers; same
# pre-registered bars) plus one extra diagnostic arm per finalist against
# eval_D, the September-meta human clone (baseline read separately).
# Usage: ./evaluation/run_league4_verdict.sh <ckpt> [<ckpt> ...]

export VERDICT_ROOT=results_gate_battery_league4
./evaluation/run_league3_verdict.sh "$@"
BASE=results_league/league_champion.zip
for CAND in "$@"; do
  LABEL=$(basename "$CAND" .zip)
  echo "[$(date '+%H:%M')] $LABEL vs eval_D (September human clone), n=1,000"
  .venv/bin/python evaluation/run_gate_battery.py --baseline $BASE --candidate "$CAND" \
    --tier screening --arms human_bc --human-bc results_bc/eval_D/saves_bc/seed2/4.zip \
    --port 7600 --out-dir "$VERDICT_ROOT/${LABEL}_evalD"
done
echo "L4_VERDICT_COMPLETE"
