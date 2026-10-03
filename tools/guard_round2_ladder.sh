#!/usr/bin/env bash
# After round 2 (tools/guard_round2_chain.sh): a 20-game serial ladder trial for each
# guard that passed the pre-registered gate (ROUND2_PASS in its log), in order
# threat_first2 then doomed_switch, each alone on top of the deployed bot, in its own
# replay dir (pre-registered in PROJECT_STATUS 2026-10-03 06:30). No promotion.
# Usage (repo root): nohup ./tools/guard_round2_ladder.sh > /dev/null 2>&1 &
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
if [ -z "${UNDER_CAFFEINATE:-}" ]; then UNDER_CAFFEINATE=1 exec caffeinate -is "$0" "$@"; fi
R2=results_analysis/threat_first_20261003/round2.log
LOG=results_analysis/threat_first_20261003/round2_ladder.log
stamp() { date '+%m-%d %H:%M:%S'; }
say() { echo "[$(stamp)] $*" >> "$LOG"; }
say "R2LADDER_WAIT"
until grep -q "ROUND2_COMPLETE" "$R2" 2>/dev/null; do sleep 60; done
sleep 30
for G in threat_first2 doomed_switch; do
  if grep -q "ROUND2_PASS $G" "$R2"; then
    say "R2LADDER_START $G 20"
    TRIAL_GUARDS=$G ./tools/ladder_trial.sh 20 "ladder_replays_mc_T6tac_$G" >> "$LOG" 2>&1
    say "R2LADDER_END $G rc=$?"
  else
    say "R2LADDER_SKIP $G (did not pass)"
  fi
done
say "R2LADDER_COMPLETE"
