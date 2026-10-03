#!/usr/bin/env bash
# Round 2 of the early-faint guards (2026-10-03, under the user's 8-hour delegation;
# pre-registered in PROJECT_STATUS): threat_first2 (threat_first aware of Mega formes,
# with a first-turn Fake Out as an answer) and doomed_switch (switch out a Pokemon
# knocked out before it moves), each tested alone on top of the deployed bot:
# 2,000-game mirror + held-out battery vs the deployed bot with its 11 guards, then
# the pre-registered gate (evaluation/guard_ladder_gate.py) -- read only, no ladder
# from this script. Finished steps are skipped. Waits for any running chain.
# Usage (repo root, AC power): nohup ./tools/guard_round2_chain.sh > /dev/null 2>&1 &
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
if [ -z "${UNDER_CAFFEINATE:-}" ]; then UNDER_CAFFEINATE=1 exec caffeinate -is "$0" "$@"; fi
mkdir -p results_analysis/threat_first_20261003
LOG=results_analysis/threat_first_20261003/round2.log
stamp() { date '+%m-%d %H:%M:%S'; }
say() { echo "[$(stamp)] $*" >> "$LOG"; }
up() { lsof -nP -iTCP:$1 -sTCP:LISTEN >/dev/null 2>&1; }
start_server() { up $1 && return 0; (cd pokemon-showdown && nohup node pokemon-showdown start $1 --no-security > /dev/null 2>&1 &); for _ in $(seq 1 60); do up $1 && return 0; sleep 1; done; return 1; }
stop_server() { pids=$(lsof -nP -t -iTCP:$1 -sTCP:LISTEN 2>/dev/null) && [ -n "$pids" ] && kill $pids 2>/dev/null; pkill -f "pokemon-showdown start $1" 2>/dev/null; sleep 2; true; }
run() { say "START $*"; "$@" >> "$LOG" 2>&1; local rc=$?; say "END rc=$rc"; return $rc; }
idle() { while pgrep -f "ladder_ourteam[.]py|ladder_read_loop[.]sh|run_guard_ab[.]py|mirror_guard_ab[.]py|vgc_bench[.]train|eval_counterfactual[.]py|threat_first_chain" >/dev/null 2>&1; do sleep 30; done; }
say "ROUND2_START"
idle
start_server 7610 || { say "ROUND2_FAILED server"; exit 3; }
for G in threat_first2 doomed_switch; do
  grep -q '"complete": true' "results_mirror_$G/result.json" 2>/dev/null \
    || run .venv/bin/python evaluation/mirror_guard_ab.py --guard $G --games 2000 --output "results_mirror_$G" \
    || say "ROUND2_FAILED mirror $G"
  grep -q complete_review_required "results_guard_ab_$G/status.json" 2>/dev/null \
    || run .venv/bin/python evaluation/run_guard_ab.py --guards $G \
         --without-arm results_guard_ab_review_guards_0928 --output "results_guard_ab_$G" \
    || say "ROUND2_FAILED battery $G"
  if run .venv/bin/python evaluation/guard_ladder_gate.py go "results_guard_ab_$G" "results_mirror_$G" $G; then
    say "ROUND2_PASS $G"
  else
    say "ROUND2_HOLD $G"
  fi
done
stop_server 7610
say "ROUND2_COMPLETE"
