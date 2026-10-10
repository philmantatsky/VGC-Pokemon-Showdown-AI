#!/usr/bin/env bash
# Serial ladder measurement read of ONE checkpoint on ONE team (Reg M-C, the
# three opt-in targeting guards) under a dead-socket restart loop.
# Usage: tools/ladder_read_loop.sh <checkpoint.zip> <team.txt> <n_games_total> <replay_dir>
#
# - Ladder play needs the user's explicit word; this script is only the launcher.
# - Refuses to start while training / batteries / another ladder or exhibition
#   session run: ladder and heavy local runs never share the machine, and the
#   ladder is serial (one game at a time).
# - Credentials are sourced HERE, shell-side, from the Laplace .env. They are
#   never passed as arguments, never printed, never read by any other tool.
# - <n_games_total> counts the games already in <replay_dir> (one .html per
#   finished game), so relaunching with a larger total continues the same read:
#   a 10-game canary, an audit, then "25" for the remaining 15. No session is
#   ever killed mid-game except on a dead socket or a 45-minute idle queue.
# - A replay dir is single-config (ladder_ourteam.py refuses a changed config).
# - No session starts, and a playing one is stopped, while the lid is closed and a
#   closed lid means sleep (tools/lid_closed.sh): a laptop asleep under its lid wakes
#   for seconds at a time, long enough to log in and queue, never to play. The launcher
#   then ends with LADDER_ABORT and status 3; it is launched again by hand.
# - PREVIEW_MODEL (env, set by ladder_deployed.sh from DEPLOYED.json): a learned,
#   human-trained preview model chooses our four and leads (--learned_preview).
# - MIXING (env, same source): ladder_ourteam.py mixed-strategy flags, or empty.
# - STICKY (env, same source): --sticky-corrections, or empty.
# - SHEET_PREVIEW (env, same source): --sheet-preview, or empty.
# - PLAYBOOK (env, same source or tools/ladder_trial.sh): our own plan cards at
#   team preview (--playbook data/playbook_<team>.json), or empty.
# - PLAYBOOK_SCRIPT_ONLY (env, tools/ladder_trial.sh): non-empty adds
#   --playbook-script-only (our usual preview; the card's turn-1 script only).
# - SEARCH (env, tools/ladder_trial.sh; tools/ladder_deployed.sh sets it empty):
#   ladder_ourteam.py exact-search flags, or empty. The deployed bot does not search.
# - FORECAST (env, set by ladder_deployed.sh from DEPLOYED.json): the opponent
#   predictor's artifact for shadow mode (--opponent-forecast; logged, read by no
#   decision), or empty.
set -uo pipefail
cd "$(dirname "$0")/.."
CKPT=${1:?checkpoint}; TEAM=${2:?team file}; N=${3:?total games}; DIR=${4:?replay dir}
GUARDS=${GUARDS:-resisted_target,overkill_split,dominated_weather_ball_weather}
MAX_SESSIONS=${MAX_SESSIONS:-6}; LOGIN_GRACE=${LOGIN_GRACE:-180}; IDLE_MIN=${IDLE_MIN:-45}
TICK=${TICK:-20}  # seconds between two looks at a playing session
HEAVY='vgc_bench[.]train|run_gate_battery|eval_counterfactual[.]py|run_counterfactual_pipeline|generate_counterfactuals|vgc_bench[.]pretrain|logs2trajs|run_team_tournament|run_team_grid|run_team_confirmation|opening_study[.]py|run_t6_|run_set_prior_ablation|run_candidate_vs_t6|learned_preview_study|human_preview|preview_entropy[.]py|mirror_guard_ab|search_roster_ab|run_guard_ab|gen_tactical_data[.]py|tactical_sft[.]py'
DEAD='keepalive ping timeout|ConnectionClosedError|TimeoutError: timed out while closing|Errno 49|Errno 54|Errno 60|Errno 8\]|nodename nor servname|gaierror|ConnectionRefusedError'
stamp() { date '+%H:%M:%S'; }
pgrep -f "$HEAVY" >/dev/null 2>&1 && { echo "LADDER_REFUSED [$(stamp)] a heavy local job is running"; exit 2; }
pgrep -f "ladder_ourteam[.]py" >/dev/null 2>&1 && { echo "LADDER_REFUSED [$(stamp)] another ladder/exhibition session is running"; exit 2; }
[ -f "$CKPT" ] && [ -f "$TEAM" ] || { echo "LADDER_REFUSED [$(stamp)] missing checkpoint or team file"; exit 2; }
PREVIEW_ARGS=()
if [ -n "${PREVIEW_MODEL:-}" ]; then
  [ -f "$PREVIEW_MODEL" ] || { echo "LADDER_REFUSED [$(stamp)] missing preview model $PREVIEW_MODEL"; exit 2; }
  PREVIEW_ARGS=(--learned_preview --preview_model "$PREVIEW_MODEL")
fi
MIXING_ARGS=()
if [ -n "${MIXING:-}" ]; then
  read -r -a MIXING_ARGS <<< "$MIXING"
fi
STICKY_ARGS=()
if [ -n "${STICKY:-}" ]; then
  STICKY_ARGS=(--sticky-corrections)
fi
SHEET_ARGS=()
if [ -n "${SHEET_PREVIEW:-}" ]; then
  SHEET_ARGS=(--sheet-preview)
fi
SEARCH_ARGS=()
if [ -n "${SEARCH:-}" ]; then
  read -r -a SEARCH_ARGS <<< "$SEARCH"
fi
FORECAST_ARGS=()
if [ -n "${FORECAST:-}" ]; then
  [ -f "$FORECAST" ] || { echo "LADDER_REFUSED [$(stamp)] missing opponent forecast artifact $FORECAST"; exit 2; }
  FORECAST_ARGS=(--opponent-forecast "$FORECAST")
fi
PLAYBOOK_ARGS=()
if [ -n "${PLAYBOOK:-}" ]; then
  [ -f "$PLAYBOOK" ] || { echo "LADDER_REFUSED [$(stamp)] missing playbook $PLAYBOOK"; exit 2; }
  PLAYBOOK_ARGS=(--playbook "$PLAYBOOK")
  [ -n "${PLAYBOOK_SCRIPT_ONLY:-}" ] && PLAYBOOK_ARGS+=(--playbook-script-only)
elif [ -n "${PLAYBOOK_SCRIPT_ONLY:-}" ]; then
  echo "LADDER_REFUSED [$(stamp)] PLAYBOOK_SCRIPT_ONLY needs PLAYBOOK"; exit 2
fi
set -a; source "../Laplace-Pokemon-Showdown-AI/.env"; set +a
mkdir -p "$DIR"
games_done() { ls "$DIR"/*.html 2>/dev/null | wc -l | tr -d ' '; }
wins_done() {  # our account name is the replay filename's prefix before " - battle-"
  local n=0 f who
  for f in "$DIR"/*.html; do
    [ -f "$f" ] || continue
    who=$(basename "$f" | sed 's/ - battle-.*//')
    grep -qi "|win|$who" "$f" && n=$((n + 1))
  done
  echo $n
}
echo "LADDER_START [$(stamp)] checkpoint=$CKPT team=$TEAM preview=${PREVIEW_MODEL:-policy} mixing=${MIXING:-off} sticky=${STICKY:+on} playbook=${PLAYBOOK:-off}${PLAYBOOK_SCRIPT_ONLY:+ (script only)} search=${SEARCH:-off} forecast=${FORECAST:-off} total=$N dir=$DIR games_done=$(games_done)"
session=0
while :; do
  done_n=$(games_done); remaining=$((N - done_n))
  [ "$remaining" -le 0 ] && break
  if bash ./tools/lid_closed.sh; then
    echo "LADDER_ABORT [$(stamp)] the lid is closed: no session starts on a laptop that is going to sleep (it would queue a game it cannot play); games_done=$done_n -- open the lid and launch again"
    exit 3
  fi
  session=$((session + 1))
  [ "$session" -gt "$MAX_SESSIONS" ] && { echo "LADDER_ABORT [$(stamp)] $MAX_SESSIONS sessions used; games_done=$done_n"; break; }
  LOG="${DIR%/}_$(date +%Y%m%d_%H%M%S)_session$session.log"  # never reuse a name: a relaunch must not overwrite an earlier report
  echo "SESSION_START $session [$(stamp)] games_done=$done_n remaining=$remaining log=$LOG"
  caffeinate -is .venv/bin/python -u ladder_ourteam.py --checkpoint "$CKPT" --reg mc --our_team "$TEAM" \
    --guards-extra "$GUARDS" --n_games "$remaining" --replay_dir "$DIR" ${PREVIEW_ARGS[@]+"${PREVIEW_ARGS[@]}"} \
    ${MIXING_ARGS[@]+"${MIXING_ARGS[@]}"} \
    ${STICKY_ARGS[@]+"${STICKY_ARGS[@]}"} \
    ${SHEET_ARGS[@]+"${SHEET_ARGS[@]}"} \
    ${SEARCH_ARGS[@]+"${SEARCH_ARGS[@]}"} \
    ${FORECAST_ARGS[@]+"${FORECAST_ARGS[@]}"} \
    ${PLAYBOOK_ARGS[@]+"${PLAYBOOK_ARGS[@]}"} > "$LOG" 2>&1 &
  PID=$!; started=$(date +%s); last_games=$done_n; last_change=$started; lid=
  while kill -0 $PID 2>/dev/null; do
    sleep "$TICK"
    now=$(date +%s)
    if bash ./tools/lid_closed.sh; then
      echo "LID_CLOSED session=$session [$(stamp)]; stopping the session: a laptop asleep under its lid abandons the game it is in and queues for the next"
      kill $PID 2>/dev/null; sleep 5; kill -9 $PID 2>/dev/null; lid=1; break
    fi
    if grep -qE "$DEAD" "$LOG"; then
      echo "SOCKET_DEAD session=$session [$(stamp)]; restarting"
      kill $PID 2>/dev/null; sleep 5; kill -9 $PID 2>/dev/null; break
    fi
    if ! grep -q "^account" "$LOG" && [ $((now - started)) -gt "$LOGIN_GRACE" ]; then
      echo "LOGIN_STALLED session=$session [$(stamp)]; restarting"
      kill $PID 2>/dev/null; sleep 5; kill -9 $PID 2>/dev/null; break
    fi
    g=$(games_done)
    if [ "$g" -ne "$last_games" ]; then
      last_games=$g; last_change=$now
      echo "GAME $g/$N wins=$(wins_done) [$(stamp)]"
    elif [ $((now - last_change)) -gt $((IDLE_MIN * 60)) ]; then
      echo "LADDER_IDLE session=$session [$(stamp)] no finished game for $IDLE_MIN min; restarting"
      kill $PID 2>/dev/null; sleep 5; kill -9 $PID 2>/dev/null; break
    fi
  done
  wait $PID 2>/dev/null
  grep -h "^record:\|^win rate:\|parse errors" "$LOG" | sed "s/^/SESSION_RECORD $session /"
  echo "SESSION_END $session [$(stamp)] games_done=$(games_done)"
  if [ -n "$lid" ]; then
    echo "LADDER_ABORT [$(stamp)] the lid was closed during session $session; games_done=$(games_done) -- open the lid and launch again"
    exit 3
  fi
  sleep 15
done
echo "LADDER_DONE [$(stamp)] games=$(games_done) wins=$(wins_done)"
