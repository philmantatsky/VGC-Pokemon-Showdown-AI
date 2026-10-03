#!/usr/bin/env bash
# The experts' turn 1 alone (2026-10-03, under the user's delegation; pre-registered
# in PROJECT_STATUS before this ran): the Water Room card's turn-1 script (Mega
# Blastoise Fake Out + Farigiraf Trick Room) on our usual preview, played only when
# the preview leads Blastoise + Farigiraf (PolicyPlayer playbook_script_only).
# 1. Held-out battery vs the deployed bot with its 11 guards.
# 2. Gate (evaluation/guard_ladder_gate.py script, 5% minimum firing share).
# 3. If it holds up: serial ladder A/B in alternating 10-game blocks, script (S) and
#    the unchanged deployed configuration (C), order S C C S C S S C. A STOP file
#    in the analysis folder ends it at the next block boundary. No promotion.
# Usage (repo root, AC power): nohup ./tools/turn1_script_chain.sh > /dev/null 2>&1 &
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
if [ -z "${UNDER_CAFFEINATE:-}" ]; then UNDER_CAFFEINATE=1 exec caffeinate -is "$0" "$@"; fi
OUT=results_analysis/turn1_script_20261003
mkdir -p "$OUT"
LOG=$OUT/chain.log
BATTERY=results_guard_ab_turn1_script
PLAYBOOK=data/playbook_t6_trial.json
SCRIPT_DIR=ladder_replays_mc_T6tac_turn1_script
CONTROL_DIR=ladder_replays_mc_T6tac_turn1_control
stamp() { date '+%m-%d %H:%M:%S'; }
say() { echo "[$(stamp)] $*" >> "$LOG"; }
up() { lsof -nP -iTCP:$1 -sTCP:LISTEN >/dev/null 2>&1; }
start_server() { up $1 && return 0; (cd pokemon-showdown && nohup node pokemon-showdown start $1 --no-security > /dev/null 2>&1 &); for _ in $(seq 1 60); do up $1 && return 0; sleep 1; done; return 1; }
stop_server() { pids=$(lsof -nP -t -iTCP:$1 -sTCP:LISTEN 2>/dev/null) && [ -n "$pids" ] && kill $pids 2>/dev/null; pkill -f "pokemon-showdown start $1" 2>/dev/null; sleep 2; true; }
run() { say "START $*"; "$@" >> "$LOG" 2>&1; local rc=$?; say "END rc=$rc"; return $rc; }
idle() { while pgrep -f "ladder_ourteam[.]py|ladder_read_loop[.]sh|run_guard_ab[.]py|mirror_guard_ab[.]py|vgc_bench[.]train|eval_counterfactual[.]py" >/dev/null 2>&1; do sleep 30; done; }
games() { ls "$1"/*.html 2>/dev/null | wc -l | tr -d ' '; }
say "TURN1_START"
idle
if ! grep -q complete_review_required "$BATTERY/status.json" 2>/dev/null; then
  start_server 7610 || { say "TURN1_FAILED server"; exit 3; }
  run .venv/bin/python evaluation/run_guard_ab.py --guards playbook_opening \
      --playbook "$PLAYBOOK" --playbook-script-only \
      --without-arm results_guard_ab_review_guards_0928 --output "$BATTERY" \
    || say "TURN1_FAILED battery"
  stop_server 7610
fi
if ! run .venv/bin/python evaluation/guard_ladder_gate.py script "$BATTERY" 0.05; then
  say "TURN1_HOLD no ladder"
  exit 0
fi
say "TURN1_PASS ladder A/B"
for ARM in S C C S C S S C; do
  [ -f "$OUT/STOP" ] && { say "TURN1_STOPPED (STOP file)"; exit 0; }
  if [ "$ARM" = S ]; then DIR=$SCRIPT_DIR; else DIR=$CONTROL_DIR; fi
  TARGET=$(( $(games "$DIR") + 10 ))
  say "BLOCK $ARM to $TARGET games in $DIR"
  if [ "$ARM" = S ]; then
    TRIAL_PLAYBOOK=$PLAYBOOK TRIAL_SCRIPT_ONLY=1 TRIAL_GUARDS=playbook_opening \
      ./tools/ladder_trial.sh "$TARGET" "$DIR" >> "$LOG" 2>&1
  else
    ./tools/ladder_trial.sh "$TARGET" "$DIR" >> "$LOG" 2>&1
  fi
  say "BLOCK_END $ARM rc=$? games=$(games "$DIR")"
done
say "TURN1_LADDER_COMPLETE script=$(games $SCRIPT_DIR) control=$(games $CONTROL_DIR)"
