#!/usr/bin/env bash
# After the round-2 ladder step: continue threat_first2's ladder trial to 40 games
# unless it won 6 or fewer of its first 20 (pre-registered in PROJECT_STATUS
# 2026-10-03 08:00, before its first ladder game). No promotion.
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
if [ -z "${UNDER_CAFFEINATE:-}" ]; then UNDER_CAFFEINATE=1 exec caffeinate -is "$0" "$@"; fi
R2L=results_analysis/threat_first_20261003/round2_ladder.log
LOG=results_analysis/threat_first_20261003/round2_ladder.log
DIR=ladder_replays_mc_T6tac_threat_first2
stamp() { date '+%m-%d %H:%M:%S'; }
until grep -q "R2LADDER_COMPLETE" "$R2L" 2>/dev/null; do sleep 60; done
sleep 30
[ -d "$DIR" ] || { echo "[$(stamp)] TF2_CONTINUE_SKIP no trial" >> "$LOG"; exit 0; }
if .venv/bin/python evaluation/guard_ladder_gate.py continue "$DIR" 20 6 >> "$LOG" 2>&1; then
  echo "[$(stamp)] TF2_CONTINUE 40" >> "$LOG"
  TRIAL_GUARDS=threat_first2 ./tools/ladder_trial.sh 40 "$DIR" >> "$LOG" 2>&1
  echo "[$(stamp)] TF2_CONTINUE_END rc=$?" >> "$LOG"
else
  echo "[$(stamp)] TF2_CONTINUE_STOP" >> "$LOG"
fi
