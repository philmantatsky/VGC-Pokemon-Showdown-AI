#!/usr/bin/env bash
# threat_first (2026-10-03, the user: "build the guard and test it, then run ladder if
# tests are good"; pre-registered in PROJECT_STATUS 2026-10-03): the 2,000-game
# mirror and the held-out battery against the deployed bot with its 11 guards, the
# pre-registered gate (evaluation/guard_ladder_gate.py), then a serial ladder trial
# of T6tac + threat_first: 20 games, continued to 40 unless it wins 6 or fewer.
# No promotion. Finished steps are skipped. Usage (repo root, AC power):
#   nohup ./tools/threat_first_chain.sh > /dev/null 2>&1 &
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
if [ -z "${UNDER_CAFFEINATE:-}" ]; then UNDER_CAFFEINATE=1 exec caffeinate -is "$0" "$@"; fi
G=threat_first
MIRROR=results_mirror_$G
BATTERY=results_guard_ab_$G
LADDER=ladder_replays_mc_T6tac_$G
mkdir -p results_analysis/threat_first_20261003
LOG=results_analysis/threat_first_20261003/chain.log
stamp() { date '+%m-%d %H:%M:%S'; }
say() { echo "[$(stamp)] $*" >> "$LOG"; }
up() { lsof -nP -iTCP:$1 -sTCP:LISTEN >/dev/null 2>&1; }
start_server() { up $1 && return 0; (cd pokemon-showdown && nohup node pokemon-showdown start $1 --no-security > /dev/null 2>&1 &); for _ in $(seq 1 60); do up $1 && return 0; sleep 1; done; return 1; }
stop_server() { pids=$(lsof -nP -t -iTCP:$1 -sTCP:LISTEN 2>/dev/null) && [ -n "$pids" ] && kill $pids 2>/dev/null; pkill -f "pokemon-showdown start $1" 2>/dev/null; sleep 2; true; }
run() { say "START $*"; "$@" >> "$LOG" 2>&1; local rc=$?; say "END rc=$rc"; return $rc; }
fail() { say "CHAIN_FAILED $*"; stop_server 7610; exit 3; }
idle() { while pgrep -f "ladder_ourteam[.]py|ladder_read_loop[.]sh|run_guard_ab[.]py|mirror_guard_ab[.]py|vgc_bench[.]train|eval_counterfactual[.]py" >/dev/null 2>&1; do sleep 30; done; }
say "CHAIN_START"
idle
start_server 7610 || fail "server 7610"
grep -q '"complete": true' "$MIRROR/result.json" 2>/dev/null \
  || run .venv/bin/python evaluation/mirror_guard_ab.py --guard $G --games 2000 --output "$MIRROR" \
  || fail "mirror"
grep -q complete_review_required "$BATTERY/status.json" 2>/dev/null \
  || run .venv/bin/python evaluation/run_guard_ab.py --guards $G \
       --without-arm results_guard_ab_review_guards_0928 --output "$BATTERY" \
  || fail "battery $(tr -d '\n' < "$BATTERY/status.json" 2>/dev/null)"
stop_server 7610
idle
if run .venv/bin/python evaluation/guard_ladder_gate.py go "$BATTERY" "$MIRROR" $G; then
  say "LADDER_START 20"
  TRIAL_GUARDS=$G ./tools/ladder_trial.sh 20 "$LADDER" >> results_analysis/threat_first_20261003/ladder.log 2>&1
  say "LADDER_END rc=$?"
  if run .venv/bin/python evaluation/guard_ladder_gate.py continue "$LADDER" 20 6; then
    say "LADDER_CONTINUE 40"
    TRIAL_GUARDS=$G ./tools/ladder_trial.sh 40 "$LADDER" >> results_analysis/threat_first_20261003/ladder.log 2>&1
    say "LADDER_END rc=$?"
  else
    say "LADDER_STOP"
  fi
else
  say "GATE_HOLD no ladder"
fi
say "CHAIN_COMPLETE"
