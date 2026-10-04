#!/usr/bin/env bash
# The rare review guards together (pre-registered in PROJECT_STATUS 2026-10-03 before
# this ran): wasted_fake_out, throat_chop_main_threat and the narrow
# hp_move_after_spread -- 2,000-game mirror + held-out battery vs the deployed bot with
# its 12 guards, read by evaluation/review1003_gate.py guards (battery deploy-eligible,
# mirror upper >= 50%, each guard's errors <= 1%). Waits for the review1003 chain.
# DEPLOYED.json is not touched; no ladder. Finished steps are skipped.
# Usage (repo root, AC power): nohup ./tools/rare_trio_chain.sh > /dev/null 2>&1 &
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
if [ -z "${UNDER_CAFFEINATE:-}" ]; then UNDER_CAFFEINATE=1 exec caffeinate -is "$0" "$@"; fi
OUT=results_analysis/review1003
LOG=$OUT/rare_trio.log
TRIO=wasted_fake_out,throat_chop_main_threat,hp_move_after_spread
REF=results_guard_ab_threat_first2
mkdir -p "$OUT"
stamp() { date '+%m-%d %H:%M:%S'; }
say() { echo "[$(stamp)] $*" >> "$LOG"; }
up() { lsof -nP -iTCP:$1 -sTCP:LISTEN >/dev/null 2>&1; }
start_server() { up $1 && return 0; (cd pokemon-showdown && nohup node pokemon-showdown start $1 --no-security > /dev/null 2>&1 &); for _ in $(seq 1 60); do up $1 && return 0; sleep 1; done; return 1; }
stop_server() { pids=$(lsof -nP -t -iTCP:$1 -sTCP:LISTEN 2>/dev/null) && [ -n "$pids" ] && kill $pids 2>/dev/null; pkill -f "pokemon-showdown start $1" 2>/dev/null; sleep 2; true; }
run() { say "START $*"; "$@" >> "$LOG" 2>&1; local rc=$?; say "END rc=$rc"; return $rc; }
idle() { while pgrep -f "review1003_chain|run_guard_ab[.]py|mirror_guard_ab[.]py|vgc_bench[.]train|human_preview" >/dev/null 2>&1; do sleep 30; done; }
say "RARE_TRIO_START"
idle
start_server 7610 || { say "RARE_TRIO_FAILED server"; exit 3; }
grep -q '"complete": true' results_mirror_rare_trio/result.json 2>/dev/null \
  || run .venv/bin/python evaluation/mirror_guard_ab.py --guard "$TRIO" --games 2000 \
       --output results_mirror_rare_trio || say "RARE_TRIO_FAILED mirror"
grep -q complete_review_required results_guard_ab_rare_trio/status.json 2>/dev/null \
  || run .venv/bin/python evaluation/run_guard_ab.py --guards "$TRIO" --without-arm "$REF" \
       --output results_guard_ab_rare_trio || say "RARE_TRIO_FAILED battery"
stop_server 7610
if run .venv/bin/python evaluation/review1003_gate.py guards results_guard_ab_rare_trio \
     results_mirror_rare_trio "$TRIO"; then say "RARE_TRIO_PASS"; else say "RARE_TRIO_HOLD"; fi
say "RARE_TRIO_COMPLETE"
