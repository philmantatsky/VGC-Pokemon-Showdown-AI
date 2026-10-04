#!/usr/bin/env bash
# The Earth Power lesson on T6e (the user, 2026-10-04: "start the earth power practice
# training but like its pretty obvious just use ep when its super effective on a
# pokemon and it does better damage than the rest of the moves and theres no better
# switch in"; pre-registered in PROJECT_STATUS 2026-10-04 before this ran):
#   4,000 practice games by the deployed bot on T6e that record where Earth Power is
#   legal -> fine-tune of the deployed brain with the focus lesson (only those
#   positions are taught: attack mass to the hardest-hitting attack; Protect / switch
#   mass and every other position stay the brain's own) -> head-to-head vs the
#   deployed bot -> held-out battery vs the deployed brain on T6e -> the gate
#   (training/t6e_ep_gate.py).
# DEPLOYED.json is not touched; no ladder. The challenge listener may keep running
# (unranked games). Finished steps are skipped, so a stopped chain can be relaunched.
# Usage (repo root, AC power): nohup ./training/t6e_ep_chain.sh > /dev/null 2>&1 &
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
if [ -z "${UNDER_CAFFEINATE:-}" ]; then UNDER_CAFFEINATE=1 exec caffeinate -is "$0" "$@"; fi
OUT=results_tactical_t6e_ep1
BRAIN=results_deployed/champion_mc_T6tac.zip
TEAM=teams/candidates_mc/T6e.txt
REF=results_brain_ab_t6e_unpractised  # the deployed brain on T6e (12 guards both sides)
H2H=results_mirror_t6e_ep1
LABEL=t6e_ep1
BATTERY=results_brain_ab_$LABEL
LOG=$OUT/chain.log
mkdir -p "$OUT"
stamp() { date '+%m-%d %H:%M:%S'; }
say() { echo "[$(stamp)] $*" >> "$LOG"; }
up() { lsof -nP -iTCP:$1 -sTCP:LISTEN >/dev/null 2>&1; }
start_server() { up $1 && return 0; (cd pokemon-showdown && nohup node pokemon-showdown start $1 --no-security > /dev/null 2>&1 &); for _ in $(seq 1 60); do up $1 && return 0; sleep 1; done; return 1; }
stop_server() { pids=$(lsof -nP -t -iTCP:$1 -sTCP:LISTEN 2>/dev/null) && [ -n "$pids" ] && kill $pids 2>/dev/null; pkill -f "pokemon-showdown start $1" 2>/dev/null; sleep 2; true; }
run() { say "START $*"; "$@" >> "$LOG" 2>&1; local rc=$?; say "END rc=$rc"; return $rc; }
fail() { say "CHAIN_FAILED $*"; stop_server 7630; stop_server 7610; exit 3; }
# ladder sessions and other heavy local jobs -- not the (unranked) challenge listener
idle() { while pgrep -f "ladder_read_loop[.]sh|run_guard_ab[.]py|mirror_guard_ab[.]py|vgc_bench[.]train|human_preview[.]py|gen_tactical_data[.]py|tactical_sft[.]py" >/dev/null 2>&1; do sleep 30; done; }
say "CHAIN_START"
idle

# 1. practice games on T6e, recording where Earth Power is legal
if [ ! -f "$OUT/data/summary.json" ]; then
  start_server 7630 || fail "server 7630"
  run .venv/bin/python -u training/gen_tactical_data.py --checkpoint "$BRAIN" \
    --team "$TEAM" --games-per-cell 500 --focus-moves earthpower \
    --output "$OUT/data" --port 7630 || fail "practice games"
  stop_server 7630
fi
FOCUS=$(.venv/bin/python -c "
from pathlib import Path
from training.tactical_sft import focus_rows, load_data
print(int(focus_rows(load_data(Path('$OUT/data'))).sum()))") || fail "focus count"
say "EARTH_POWER_POSITIONS $FOCUS"
[ "$FOCUS" -ge 300 ] || fail "too few Earth Power positions to teach ($FOCUS < 300)"

# 2. the focus lesson on the deployed brain
if [ ! -f "$OUT/sft/log.json" ]; then
  run .venv/bin/python -u training/tactical_sft.py --checkpoint "$BRAIN" \
    --data "$OUT/data" --output "$OUT/sft" --lr 1e-4 --epochs 4 --lessons focus \
    || fail "fine-tune"
fi
BEST=$(.venv/bin/python -c "
import json; log = json.load(open('$OUT/sft/log.json'))
print(min(log['epochs'], key=lambda e: e['cross_entropy'])['epoch'])") || fail "epoch pick"
CAND=$OUT/sft/tactical_e$BEST.zip
say "CANDIDATE $CAND"

# 3. head-to-head vs the deployed bot (both on T6e with the deployed guards)
start_server 7610 || fail "server 7610"
grep -q '"complete": true' "$H2H/result.json" 2>/dev/null \
  || run .venv/bin/python evaluation/mirror_guard_ab.py --a-checkpoint "$CAND" \
       --games 2000 --output "$H2H" || fail "head-to-head"

# 4. the held-out battery vs the deployed brain on T6e
grep -q complete_review_required "$BATTERY/status.json" 2>/dev/null \
  || run .venv/bin/python evaluation/run_guard_ab.py --candidate "$CAND" \
       --candidate-plans data/opening_plans_t6e.json --label "$LABEL" \
       --without-arm "$REF" || fail "battery $(tr -d '\n' < "$BATTERY/status.json" 2>/dev/null)"
stop_server 7610

# 5. the pre-registered gate (no ladder, no promotion)
run .venv/bin/python training/t6e_ep_gate.py "$OUT/sft" "$BATTERY" "$H2H"
case $? in
  0) say "GATE_GO $CAND" ;;
  4) say "GATE_NEUTRAL $CAND" ;;
  *) say "GATE_HOLD $CAND" ;;
esac
say "CHAIN_COMPLETE"
