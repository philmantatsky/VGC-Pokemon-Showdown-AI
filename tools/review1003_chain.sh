#!/usr/bin/env bash
# The user's 2026-10-03 requests, measured in order (pre-registered in PROJECT_STATUS
# 2026-10-03 before any of these runs; readings in evaluation/review1003_gate.py):
#   1. the three review guards together (wasted_fake_out, throat_chop_main_threat,
#      hp_move_after_hits): 2,000-game mirror + held-out battery vs the deployed bot
#      with its 12 guards;
#   2. T6e (Torkoal Earth Power for Weather Ball) on the deployed brain, unpractised:
#      head-to-head vs the same brain on T6 + battery;
#   3. the open-sheet preview (PolicyPlayer sheet_preview): battery.
# Waits for the T6tac practice chain. DEPLOYED.json is not touched; no ladder.
# Finished steps are skipped, so a stopped chain can be relaunched.
# Usage (repo root, AC power): nohup ./tools/review1003_chain.sh > /dev/null 2>&1 &
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
if [ -z "${UNDER_CAFFEINATE:-}" ]; then UNDER_CAFFEINATE=1 exec caffeinate -is "$0" "$@"; fi
OUT=results_analysis/review1003
LOG=$OUT/chain.log
GUARDS3=wasted_fake_out,throat_chop_main_threat,hp_move_after_hits
REF=results_guard_ab_threat_first2  # the deployed brain with all 12 guards
BRAIN=results_deployed/champion_mc_T6tac.zip
mkdir -p "$OUT"
stamp() { date '+%m-%d %H:%M:%S'; }
say() { echo "[$(stamp)] $*" >> "$LOG"; }
up() { lsof -nP -iTCP:$1 -sTCP:LISTEN >/dev/null 2>&1; }
start_server() { up $1 && return 0; (cd pokemon-showdown && nohup node pokemon-showdown start $1 --no-security > /dev/null 2>&1 &); for _ in $(seq 1 60); do up $1 && return 0; sleep 1; done; return 1; }
stop_server() { pids=$(lsof -nP -t -iTCP:$1 -sTCP:LISTEN 2>/dev/null) && [ -n "$pids" ] && kill $pids 2>/dev/null; pkill -f "pokemon-showdown start $1" 2>/dev/null; sleep 2; true; }
run() { say "START $*"; "$@" >> "$LOG" 2>&1; local rc=$?; say "END rc=$rc"; return $rc; }
done_mirror() { grep -q '"complete": true' "$1/result.json" 2>/dev/null; }
done_battery() { grep -q complete_review_required "$1/status.json" 2>/dev/null; }
idle() { while pgrep -f "t6tac_practice_chain|ladder_read_loop[.]sh|run_guard_ab[.]py|mirror_guard_ab[.]py|vgc_bench[.]train|human_preview|gen_tactical_data" >/dev/null 2>&1; do sleep 30; done; }
say "REVIEW1003_START"
idle
start_server 7610 || { say "REVIEW1003_FAILED server"; exit 3; }

# 1. the three review guards
done_mirror results_mirror_review1003 \
  || run .venv/bin/python evaluation/mirror_guard_ab.py --guard "$GUARDS3" --games 2000 \
       --output results_mirror_review1003 || say "REVIEW1003_FAILED guards mirror"
done_battery results_guard_ab_review1003 \
  || run .venv/bin/python evaluation/run_guard_ab.py --guards "$GUARDS3" --without-arm "$REF" \
       --output results_guard_ab_review1003 || say "REVIEW1003_FAILED guards battery"
if run .venv/bin/python evaluation/review1003_gate.py guards results_guard_ab_review1003 \
     results_mirror_review1003 "$GUARDS3"; then say "GUARDS_PASS"; else say "GUARDS_HOLD"; fi

# 2. T6e on the deployed brain, unpractised
done_mirror results_mirror_t6e_unpractised \
  || run .venv/bin/python evaluation/mirror_guard_ab.py --a-checkpoint "$BRAIN" \
       --a-team teams/candidates_mc/T6e.txt --games 2000 \
       --output results_mirror_t6e_unpractised || say "REVIEW1003_FAILED t6e mirror"
done_battery results_brain_ab_t6e_unpractised \
  || run .venv/bin/python evaluation/run_guard_ab.py --candidate "$BRAIN" \
       --candidate-plans data/opening_plans_t6e.json --label t6e_unpractised \
       --without-arm "$REF" || say "REVIEW1003_FAILED t6e battery"
if run .venv/bin/python evaluation/review1003_gate.py team results_brain_ab_t6e_unpractised \
     results_mirror_t6e_unpractised; then say "T6E_PASS"; else say "T6E_HOLD"; fi

# 3. the open-sheet preview
done_battery results_guard_ab_sheet_preview \
  || run .venv/bin/python evaluation/run_guard_ab.py --sheet-preview --without-arm "$REF" \
       --output results_guard_ab_sheet_preview || say "REVIEW1003_FAILED sheet battery"
if run .venv/bin/python evaluation/review1003_gate.py sheet results_guard_ab_sheet_preview; then
  say "SHEET_PASS"; else say "SHEET_HOLD"; fi

stop_server 7610
say "REVIEW1003_COMPLETE"
