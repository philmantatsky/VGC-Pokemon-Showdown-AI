#!/usr/bin/env bash
set -uo pipefail

# Exhibition mode: keep the deployed bot online to accept challenges from
# anyone (the README invites portfolio visitors to fight it) whenever this
# machine is otherwise idle. It yields to real work automatically: between
# short 3-challenge sessions it checks for training / battery / ladder-batch
# processes and waits while any are running. A session already in progress
# finishes its games first -- and note a ladder batch started on the same
# account will kick the exhibition login, which is the intended priority.
#
# Watchdog (2026-09-10): after a network hiccup the Showdown websocket dies
# while the Python process stays alive and deaf (keepalive ping timeout,
# ConnectionClosedError, "Can't assign requested address"), and a DNS failure
# at login ("nodename nor servname provided") leaves it waiting for a login
# that never comes. Each session writes its own log; when a failure signature
# appears, or the session has not reached "awaiting" within LOGIN_GRACE
# seconds, the session is killed and a fresh one logs in (marker lines below).
#
# Run it (credentials are sourced locally, per house rules):
#   nohup ./exhibition_mode.sh > exhibition_mode.log 2>&1 &
# Stop with pkill -f exhibition_mode.sh (then pkill -f ladder_ourteam.py).
# Exhibition games land in ladder_replays_exhibition_<reg>/, kept separate
# from the measured ladder corpus. The checkpoint and team default to the
# deployed configuration in results_deployed/DEPLOYED.json (sha-verified;
# T6 since 2026-09-23), Reg M-C, the three opt-in targeting guards
# (CHECKPOINT / TEAM / REG / GUARDS env override). poke-env only accepts a challenge sent in the bot's own format,
# so challengers must pick "[Gen 9 Champions] VGC 2026 Reg M-C".

cd "$(dirname "$0")"
set -a; source "../Laplace-Pokemon-Showdown-AI/.env"; set +a
REG=${REG:-mc}
# DEP_* = the deployed configuration, sha-verified (checkpoint, team, preview model)
CFG=$(.venv/bin/python tools/deployed_config.py --prefix DEP_) || {
  echo "EXHIBITION_REFUSED the deployed configuration failed verification"; exit 2
}
eval "$CFG"
CHECKPOINT=${CHECKPOINT:-$DEP_CKPT}
# the deployed brain reads the set data it was trained with; any other checkpoint
# uses SET_PRIOR (env) or the format default
if [ "$CHECKPOINT" = "$DEP_CKPT" ]; then SET_PRIOR=${SET_PRIOR:-$DEP_SET_PRIOR}; fi
if [ -n "${SET_PRIOR:-}" ]; then export VGC_SET_PRIOR_REG="$SET_PRIOR"; fi
TEAM=${TEAM:-$DEP_TEAM}
# the deployed preview model (if any) belongs to the deployed brain on the deployed team
PREVIEW_ARGS=()
MIXING_ARGS=()
STICKY_ARGS=()
PLAYBOOK_ARGS=()
SHEET_ARGS=()
TAG=$(basename "$TEAM" .txt)
DEFAULT_GUARDS=resisted_target,overkill_split,dominated_weather_ball_weather
if [ "$CHECKPOINT" = "$DEP_CKPT" ] && [ "$TEAM" = "$DEP_TEAM" ]; then
  TAG=$DEP_REPLAY_TAG
  DEFAULT_GUARDS=$DEP_GUARDS  # the deployed brain plays with the deployed guards
  if [ -n "$DEP_PREVIEW_MODEL" ]; then
    PREVIEW_ARGS=(--learned_preview --preview_model "$DEP_PREVIEW_MODEL")
  fi
  if [ -n "$DEP_MIXING" ]; then
    read -r -a MIXING_ARGS <<< "$DEP_MIXING"
  fi
  if [ -n "$DEP_STICKY" ]; then
    STICKY_ARGS=(--sticky-corrections)
  fi
  if [ -n "$DEP_PLAYBOOK" ]; then
    PLAYBOOK_ARGS=(--playbook "$DEP_PLAYBOOK")
  fi
  if [ -n "${DEP_SHEET_PREVIEW:-}" ]; then
    SHEET_ARGS=(--sheet-preview)
  fi
fi
GUARDS=${GUARDS:-$DEFAULT_GUARDS}
LOGIN_GRACE=${LOGIN_GRACE:-120}
HEAVY='vgc_bench[.]train|run_gate_battery|eval_counterfactual[.]py|run_counterfactual_pipeline|generate_counterfactuals|vgc_bench[.]pretrain|logs2trajs|run_team_tournament|opening_study[.]py|run_t6_|run_set_prior_ablation|run_candidate_vs_t6|learned_preview_study|human_preview|preview_entropy[.]py|mirror_guard_ab|run_guard_ab'
DEAD='keepalive ping timeout|ConnectionClosedError|TimeoutError: timed out while closing|Errno 49|Errno 54|Errno 60|Errno 8\]|nodename nor servname|gaierror|ConnectionRefusedError'
mkdir -p exhibition_logs

echo "exhibition mode: accepting challenges in reg $REG (pkill -f exhibition_mode.sh to stop)"
session=0
while true; do
  if pgrep -f "$HEAVY" >/dev/null 2>&1 \
     || pgrep -af "ladder_ourteam[.]py" 2>/dev/null | grep -v -- "--challenges" | grep -q .; then
    echo "$(date '+%H:%M') heavy job running; exhibition waiting..."
    sleep 120
    continue
  fi
  session=$((session + 1))
  LOG="exhibition_logs/session_$(date '+%Y%m%d_%H%M%S').log"
  echo "SESSION_START $session [$(date '+%H:%M:%S')] log=$LOG"
  caffeinate -is .venv/bin/python -u ladder_ourteam.py \
    --checkpoint "$CHECKPOINT" \
    --reg "$REG" --our_team "$TEAM" \
    --guards-extra "$GUARDS" \
    --challenges --n_games 3 \
    --replay_dir "ladder_replays_exhibition_${REG}_${TAG}" \
    ${PREVIEW_ARGS[@]+"${PREVIEW_ARGS[@]}"} \
    ${MIXING_ARGS[@]+"${MIXING_ARGS[@]}"} \
    ${STICKY_ARGS[@]+"${STICKY_ARGS[@]}"} \
    ${PLAYBOOK_ARGS[@]+"${PLAYBOOK_ARGS[@]}"} \
    ${SHEET_ARGS[@]+"${SHEET_ARGS[@]}"} > "$LOG" 2>&1 &
  PID=$!
  started=$(date +%s)
  while kill -0 $PID 2>/dev/null; do
    sleep 15
    if grep -qE "$DEAD" "$LOG"; then
      echo "SOCKET_DEAD session=$session [$(date '+%H:%M:%S')]; restarting"
      kill $PID 2>/dev/null; sleep 5; kill -9 $PID 2>/dev/null; break
    fi
    if ! grep -q "awaiting" "$LOG" && [ $(( $(date +%s) - started )) -gt "$LOGIN_GRACE" ]; then
      echo "LOGIN_STALLED session=$session [$(date '+%H:%M:%S')]; restarting"
      kill $PID 2>/dev/null; sleep 5; kill -9 $PID 2>/dev/null; break
    fi
  done
  wait $PID 2>/dev/null
  grep -h "^record:" "$LOG" | sed "s/^/SESSION_RECORD $session /"
  echo "SESSION_END $session [$(date '+%H:%M:%S')]"
  sleep 20
done
