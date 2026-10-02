#!/usr/bin/env bash
# The T6m practice cycle (the user, 2026-10-01: "add the guards to the bot and start
# the practice run ... if it does well start 15 ladder games and go from there on
# your own"; pre-registered in PROJECT_STATUS 2026-10-02):
#   training on T6m -> tactical re-fit on its own T6m games -> both head-to-heads vs
#   the deployed bot -> the battery of the better one -> the pre-registered gate ->
#   a 15-game ladder trial (continued to 40 unless it starts 4-11 or worse).
# No promotion: DEPLOYED.json is not touched. Finished steps are skipped, so a
# stopped chain can be relaunched.
# Usage (repo root, AC power): nohup ./training/t6m_practice_chain.sh &
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
if [ -z "${UNDER_CAFFEINATE:-}" ]; then UNDER_CAFFEINATE=1 exec caffeinate -is "$0" "$@"; fi
NAME=brainv1_t6m_practice1
OUT=results_$NAME
TAC=results_tactical_t6m1
TEAM=teams/candidates_mc/T6m.txt
INIT=results_tactical1/sft/tactical_e4.zip  # = the deployed T6tac (same sha)
REF=results_guard_ab_review_guards_0928     # the deployed brain with all 11 guards
LADDER=ladder_replays_mc_t6m1
LOG=$OUT/chain.log
mkdir -p "$OUT"
stamp() { date '+%m-%d %H:%M:%S'; }
say() { echo "[$(stamp)] $*" >> "$LOG"; }
up() { lsof -nP -iTCP:$1 -sTCP:LISTEN >/dev/null 2>&1; }
start_server() { up $1 && return 0; (cd pokemon-showdown && nohup node pokemon-showdown start $1 --no-security > /dev/null 2>&1 &); for _ in $(seq 1 60); do up $1 && return 0; sleep 1; done; return 1; }
stop_server() { pids=$(lsof -nP -t -iTCP:$1 -sTCP:LISTEN 2>/dev/null) && [ -n "$pids" ] && kill $pids 2>/dev/null; pkill -f "pokemon-showdown start $1" 2>/dev/null; sleep 2; true; }
run() { say "START $*"; "$@" >> "$LOG" 2>&1; local rc=$?; say "END rc=$rc"; return $rc; }
fail() { say "CHAIN_FAILED $*"; stop_server 7700; stop_server 7630; stop_server 7610; exit 3; }
idle() { while pgrep -f "ladder_ourteam[.]py|ladder_read_loop[.]sh|run_guard_ab[.]py|mirror_guard_ab[.]py|vgc_bench[.]train|eval_counterfactual[.]py" >/dev/null 2>&1; do sleep 30; done; }
say "CHAIN_START"
idle

# 1. training: T6tac + 983,040 steps with T6m as our team
SAVES=$OUT/saves_fp_xt_hs_wt/reg_mc/seed1
if ! grep -q training_complete_evaluation_next "$OUT/status.json" 2>/dev/null; then
  start_server 7700 || fail "server 7700"
  run .venv/bin/python -u training/run_t6_teamvariant_trial.py --init "$INIT" \
    || fail "training $(tr -d '\n' < "$OUT/status.json" 2>/dev/null)"
  stop_server 7700
fi
FINAL=$(ls "$SAVES"/*.zip | sed 's#.*/##; s#\.zip##' | sort -n | tail -1)
CAND_RL=$SAVES/$FINAL.zip
say "RL_SAVE $CAND_RL"

# 2. tactical re-fit on the practised brain's own T6m games (base lessons)
if [ ! -f "$TAC/data/summary.json" ]; then
  start_server 7630 || fail "server 7630"
  run .venv/bin/python -u training/gen_tactical_data.py --checkpoint "$CAND_RL" \
    --team "$TEAM" --games-per-cell 250 --output "$TAC/data" --port 7630 \
    || fail "tactical data"
  stop_server 7630
fi
if [ ! -f "$TAC/sft/log.json" ]; then
  run .venv/bin/python -u training/tactical_sft.py --checkpoint "$CAND_RL" \
    --data "$TAC/data" --output "$TAC/sft" --lr 1e-4 --epochs 4 --lessons base \
    || fail "tactical fine-tune"
fi
BEST=$(.venv/bin/python -c "
import json; log = json.load(open('$TAC/sft/log.json'))
print(min(log['epochs'], key=lambda e: e['cross_entropy'])['epoch'])") || fail "epoch pick"
CAND_TAC=$TAC/sft/tactical_e$BEST.zip
say "TACTICAL_REFIT $CAND_TAC"

# 3. head-to-head vs the deployed bot (T6tac on T6), 2,000 games each
start_server 7610 || fail "server 7610"
[ -f results_mirror_t6m1_rl/result.json ] && grep -q '"complete": true' results_mirror_t6m1_rl/result.json \
  || run .venv/bin/python evaluation/mirror_guard_ab.py --a-checkpoint "$CAND_RL" \
       --a-team "$TEAM" --games 2000 --output results_mirror_t6m1_rl || fail "head-to-head rl"
[ -f results_mirror_t6m1_tactical/result.json ] && grep -q '"complete": true' results_mirror_t6m1_tactical/result.json \
  || run .venv/bin/python evaluation/mirror_guard_ab.py --a-checkpoint "$CAND_TAC" \
       --a-team "$TEAM" --games 2000 --output results_mirror_t6m1_tactical || fail "head-to-head tactical"
# pre-registered: the battery plays whichever of the two won more of its head-to-head
PICK=$(.venv/bin/python -c "
import json
a = json.load(open('results_mirror_t6m1_rl/result.json'))['a_win_rate']
b = json.load(open('results_mirror_t6m1_tactical/result.json'))['a_win_rate']
print('rl' if a > b else 'tactical')") || fail "pick"
if [ "$PICK" = rl ]; then CAND=$CAND_RL; LABEL="t6m1_$FINAL"; else CAND=$CAND_TAC; LABEL="t6m1_${FINAL}_tactical"; fi
say "BATTERY_PICK $PICK $CAND"

# 4. the held-out battery of the pick on T6m vs the deployed brain on T6
BATTERY=results_brain_ab_$LABEL
if ! grep -q complete_review_required "$BATTERY/status.json" 2>/dev/null; then
  run .venv/bin/python evaluation/run_guard_ab.py --candidate "$CAND" \
    --candidate-plans data/opening_plans_t6m.json --label "$LABEL" --without-arm "$REF" \
    || fail "battery $(tr -d '\n' < "$BATTERY/status.json" 2>/dev/null)"
fi
stop_server 7610
idle

# 5. the gate, then the ladder trial (serial; credentials are sourced by the launcher)
if run .venv/bin/python training/t6m_practice_gate.py go "$BATTERY" "results_mirror_t6m1_$PICK"; then
  .venv/bin/python - "$CAND" "$TEAM" >> "$LOG" 2>&1 <<'PY' || fail "sidecar"
import hashlib, json, sys
from pathlib import Path
ckpt, team = Path(sys.argv[1]), sys.argv[2]
side = Path(str(ckpt) + ".metadata.json")
meta = json.loads(side.read_text()) if side.exists() else {}
meta.update(
    sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
    requires_knowledge_obs=True,
    role="candidate",
    team=team,
)
side.write_text(json.dumps(meta, indent=2) + "\n")
print("stamped", side)
PY
  say "LADDER_START 15 $CAND"
  TRIAL_CHECKPOINT=$CAND TRIAL_TEAM=$TEAM ./tools/ladder_trial.sh 15 "$LADDER" >> "$OUT/ladder.log" 2>&1
  say "LADDER_END rc=$?"
  if run .venv/bin/python training/t6m_practice_gate.py continue "$LADDER"; then
    say "LADDER_CONTINUE 40"
    TRIAL_CHECKPOINT=$CAND TRIAL_TEAM=$TEAM ./tools/ladder_trial.sh 40 "$LADDER" >> "$OUT/ladder.log" 2>&1
    say "LADDER_END rc=$?"
  else
    say "LADDER_STOP"
  fi
else
  say "GATE_HOLD no ladder"
fi
say "CHAIN_COMPLETE"
