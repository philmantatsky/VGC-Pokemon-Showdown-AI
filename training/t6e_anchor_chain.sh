#!/usr/bin/env bash
# Anchored practice on T6e (the user, 2026-10-04: "do more training afterwards too ...
# use your best judgement through the night"; pre-registered in PROJECT_STATUS
# 2026-10-04 before this ran):
#   the T6ctx practice recipe (+983,040 steps) on T6e from START with a KL anchor to
#   START (training/run_t6e_anchor_trial.py, target 0.10 nats) -> its drift from
#   START on the T6e positions -> head-to-head vs the deployed bot -> held-out
#   battery vs the deployed brain on T6e -> the gate (training/t6tac_practice_gate.py
#   go: head-to-head won AND battery deploy-eligible).
# START = $1, chosen by the pre-registered rule (the Earth Power candidate unless its
# head-to-head was lost or its battery unsafe, else the deployed T6tac).
# Launch it only after the ladder trial ends (ladder and heavy local runs never share
# the machine); it also waits while any ladder session or heavy job runs.
# DEPLOYED.json is not touched; no ladder. Finished steps are skipped.
# Usage (repo root, AC power): nohup ./training/t6e_anchor_chain.sh <start.zip> > /dev/null 2>&1 &
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
if [ -z "${UNDER_CAFFEINATE:-}" ]; then UNDER_CAFFEINATE=1 exec caffeinate -is "$0" "$@"; fi
START=${1:?start brain}
NAME=brainv1_t6e_anchor1
OUT=results_$NAME
SAVES=$OUT/saves_fp_xt_hs_wt/reg_mc/seed1
REF=results_brain_ab_t6e_unpractised  # the deployed brain on T6e (12 guards both sides)
H2H=results_mirror_t6e_anchor1
LABEL=t6e_anchor1
BATTERY=results_brain_ab_$LABEL
LOG=$OUT/chain.log
mkdir -p "$OUT"
stamp() { date '+%m-%d %H:%M:%S'; }
say() { echo "[$(stamp)] $*" >> "$LOG"; }
up() { lsof -nP -iTCP:$1 -sTCP:LISTEN >/dev/null 2>&1; }
start_server() { up $1 && return 0; (cd pokemon-showdown && nohup node pokemon-showdown start $1 --no-security > /dev/null 2>&1 &); for _ in $(seq 1 60); do up $1 && return 0; sleep 1; done; return 1; }
stop_server() { pids=$(lsof -nP -t -iTCP:$1 -sTCP:LISTEN 2>/dev/null) && [ -n "$pids" ] && kill $pids 2>/dev/null; pkill -f "pokemon-showdown start $1" 2>/dev/null; sleep 2; true; }
run() { say "START $*"; "$@" >> "$LOG" 2>&1; local rc=$?; say "END rc=$rc"; return $rc; }
fail() { say "CHAIN_FAILED $*"; stop_server 7700; stop_server 7610; exit 3; }
# ladder sessions and other heavy local jobs -- not the (unranked) challenge listener
idle() { while pgrep -f "ladder_read_loop[.]sh|run_guard_ab[.]py|mirror_guard_ab[.]py|vgc_bench[.]train|human_preview[.]py|gen_tactical_data[.]py|tactical_sft[.]py" >/dev/null 2>&1; do sleep 30; done; }
say "CHAIN_START start=$START"
idle

# 1. anchored practice on T6e
if ! grep -q training_complete_evaluation_next "$OUT/status.json" 2>/dev/null; then
  start_server 7700 || fail "server 7700"
  run .venv/bin/python -u training/run_t6e_anchor_trial.py --init "$START" \
    || fail "training $(tr -d '\n' < "$OUT/status.json" 2>/dev/null)"
  stop_server 7700
fi
FINAL=$(ls "$SAVES"/*.zip | sed 's#.*/##; s#\.zip##' | sort -n | tail -1)
CAND=$SAVES/$FINAL.zip
say "CANDIDATE $CAND"

# 2. how far it moved from its start (the anchor's job), on the T6e positions
run .venv/bin/python - "$START" "$CAND" <<'PY' || say "DRIFT_REPORT_FAILED"
import sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, torch
from stable_baselines3 import PPO
from training.tactical_sft import load_data, policy_probs
data = load_data(Path("results_tactical_t6e_ep1/data"))
rows = np.random.default_rng(0).choice(len(data["obs"]), 4000, replace=False)
args = (data["obs"][rows], data["mask"][rows], data["played"][rows, 0], torch.device("cpu"))
p = [policy_probs(PPO.load(f, device="cpu").policy.eval(), *args) for f in sys.argv[1:3]]
def kl(a, b):
    return float(np.where(a > 0, a * (np.log(np.maximum(a, 1e-12)) - np.log(np.maximum(b, 1e-12))), 0).sum(-1).mean())
print(f"DRIFT KL(start || candidate) slot1 {kl(p[0][0], p[1][0]):.3f} slot2 {kl(p[0][1], p[1][1]):.3f}"
      f" same top slot1 {100 * (p[0][0].argmax(-1) == p[1][0].argmax(-1)).mean():.1f}%")
PY

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
if run .venv/bin/python training/t6tac_practice_gate.py go "$BATTERY" "$H2H"; then
  say "GATE_GO $CAND"
else
  say "GATE_HOLD $CAND"
fi
say "CHAIN_COMPLETE"
