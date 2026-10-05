#!/usr/bin/env bash
# forme_stats (2026-10-04; pre-registered in PROJECT_STATUS 2026-10-04 21:00, amended
# 21:20 before either run below existed): its gate. This chain waits until the machine
# is free of every other heavy or latency-sensitive job (the search session's overnight
# V5 above all), then plays
#   1. the 2,000-game mirror again -- the first one (results_mirror_forme_stats, 10-04
#      evening) ran before the entry recognised open-sheet stat lines, so its two
#      open-sheet blocks compared the bot with itself;
#   2. the held-out battery of the deployed brain on the deployed team with the
#      guard-profile entry forme_stats against the same brain and team without it;
# and prints the pre-registered verdict (evaluation/guard_ladder_gate.py go). No
# ladder, no promotion, DEPLOYED.json is not touched. Finished steps are skipped.
#
# The without side is results_brain_ab_t6e_ep1: the deployed T6ep on T6e with the 12
# guards deployed when it was played (2026-10-04 03:00; wasted_fake_out and
# throat_chop_main_threat came later and are off on both sides). run_guard_ab.py plays
# a plain guard arm on the reference study's team (T6), so the with side goes through
# its candidate path: the same brain file and --candidate-plans for T6e. The two arms
# then differ in forme_stats alone. The port must be the reference arm's (7610): the
# arms' manifests are compared field by field.
# Usage (repo root, AC power):  nohup ./tools/forme_stats_chain.sh > /dev/null 2>&1 &
# Stop it while it waits:       touch results_analysis/forme_stats_20261004/STOP
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
if [ -z "${UNDER_CAFFEINATE:-}" ]; then UNDER_CAFFEINATE=1 exec caffeinate -is "$0" "$@"; fi
G=forme_stats
PORT=7610
MIRROR=results_mirror_${G}2
BATTERY=results_guard_ab_$G
WITHOUT=results_brain_ab_t6e_ep1
BRAIN=results_tactical_t6e_ep1/sft/tactical_e4.zip  # the deployed brain's own file (sha f92248c6)
PLANS=data/opening_plans_t6e.json
OUT=results_analysis/forme_stats_20261004
LOG=$OUT/chain.log
QUIET_MINUTES=${QUIET_MINUTES:-10}
BUSY='mirror_guard_ab[.]py|search_mirror_|run_guard_ab[.]py|learned_preview_study|leaf_calibration|vgc_bench[.]train|eval_counterfactual[.]py|run_gate_battery|gen_tactical_data|tactical_sft|train_oppmodel|ladder_read_loop[.]sh'
stamp() { date '+%m-%d %H:%M:%S'; }
say() { echo "[$(stamp)] $*" >> "$LOG"; }
up() { lsof -nP -iTCP:$1 -sTCP:LISTEN >/dev/null 2>&1; }
stop_server() { for p in $(lsof -nP -t -iTCP:$1 -sTCP:LISTEN 2>/dev/null); do kill "$p" 2>/dev/null; done; sleep 2; true; }
# A fresh server: one left up by an earlier run may still hold that run's rooms.
start_server() { up $1 && stop_server $1; (cd pokemon-showdown && nohup node pokemon-showdown start $1 --no-security > /dev/null 2>&1 &); for _ in $(seq 1 60); do up $1 && return 0; sleep 1; done; return 1; }
run() { say "START $*"; "$@" >> "$LOG" 2>&1; local rc=$?; say "END rc=$rc"; return $rc; }
# Free means free for QUIET_MINUTES in a row: a study that runs in rounds has no
# process of its own for a moment between two of them.
wait_until_free() {
  local quiet=0
  while [ "$quiet" -lt "$QUIET_MINUTES" ]; do
    [ -e "$OUT/STOP" ] && { say "CHAIN_STOPPED by $OUT/STOP"; exit 0; }
    if pgrep -f "$BUSY" >/dev/null 2>&1; then quiet=0; else quiet=$((quiet + 1)); fi
    sleep 60
  done
}
mkdir -p "$OUT"
say "CHAIN_START (waiting for a free machine)"
if ! grep -q '"complete": true' "$MIRROR/result.json" 2>/dev/null; then
  wait_until_free
  start_server "$PORT" || { say "CHAIN_FAILED the server on $PORT did not start"; exit 3; }
  run .venv/bin/python evaluation/mirror_guard_ab.py --guard $G --games 2000 \
    --port "$PORT" --output "$MIRROR"
  rc=$?
  stop_server "$PORT"
  [ $rc -eq 0 ] || { say "CHAIN_FAILED mirror"; exit 3; }
fi
# the result without its 2,000 per-battle rows
.venv/bin/python - "$MIRROR/result.json" "$OUT/mirror_result.json" <<'PY' >> "$LOG" 2>&1
import json, sys
result = json.load(open(sys.argv[1]))
for block in result["blocks"]:
    block.pop("battles", None)
json.dump(result, open(sys.argv[2], "w"), indent=1)
PY
if ! grep -q complete_review_required "$BATTERY/status.json" 2>/dev/null; then
  wait_until_free
  start_server "$PORT" || { say "CHAIN_FAILED the server on $PORT did not start"; exit 3; }
  run .venv/bin/python evaluation/run_guard_ab.py --guards $G --candidate "$BRAIN" \
    --label $G --candidate-plans "$PLANS" --without-arm "$WITHOUT" \
    --port "$PORT" --output "$BATTERY"
  rc=$?
  stop_server "$PORT"
  [ $rc -eq 0 ] || { say "CHAIN_FAILED battery $(tr -d '\n' < "$BATTERY/status.json" 2>/dev/null)"; exit 3; }
fi
cp "$BATTERY/scorecard.json" "$OUT/battery_scorecard.json" 2>/dev/null
if run .venv/bin/python evaluation/guard_ladder_gate.py go "$BATTERY" "$MIRROR" $G; then
  say "GATE_GO $G may have a ladder trial (the user's word): TRIAL_GUARDS=$G ./tools/ladder_trial.sh 20 <replay dir>"
else
  say "GATE_HOLD $G stays off"
fi
say "CHAIN_COMPLETE"
