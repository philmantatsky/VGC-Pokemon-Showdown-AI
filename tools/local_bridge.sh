#!/usr/bin/env bash
# Serial games on a LOCAL server between the deployed configuration -- as
# tools/deployed_config.py resolves it, through ladder_ourteam.py with the argument list
# tools/ladder_read_loop.sh would build, the search off -- and an external client that
# logs in under a name of its own (evaluation/local_bridge.py: the blocks, the client
# contract, what is written). No credentials are sourced and nothing reaches the real
# server.
# Usage (a local Showdown server on <port>: not 7200 / 7400, which training and
# evaluation use):
#   tools/local_bridge.sh <port> <out dir> <client account> <blocks> [bridge flags] \
#       [-- extra ladder_ourteam.py flags]
# <blocks>: "they:closed:75,they:open:75,we:closed:75,we:open:75"; an entry ending ":same"
# is the arm on our own team. Bridge flags: --client "<command template>", --bot-name,
# --game-timeout, --retry, --resume. <out dir> outside the repository.
# A rehearsal with the stand-in client (the deployed bot without search):
#   tools/local_bridge.sh 7650 <out dir> standin they:closed:5,they:open:5,we:closed:5,we:open:5 \
#     --client ".venv/bin/python checks/local_bridge_standin.py {port} {opponent} {bot} {mode} {games} {sheets} --team '{team}'"
# The tally of a run, again: .venv/bin/python evaluation/local_bridge.py --tally <out dir>
set -uo pipefail
cd "$(dirname "$0")/.."
PORT=${1:?port}; OUT=${2:?out dir}; CLIENT=${3:?client account}; BLOCKS=${4:?blocks}; shift 4
BRIDGE_ARGS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do BRIDGE_ARGS+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
case "$PORT" in 7200|7400) echo "BRIDGE_REFUSED port $PORT belongs to training and evaluation"; exit 2 ;; esac
nc -z localhost "$PORT" 2>/dev/null || { echo "BRIDGE_REFUSED no server on port $PORT: (cd pokemon-showdown && node pokemon-showdown start $PORT --no-security)"; exit 2; }
pgrep -f "ladder_ourteam[.]py" >/dev/null 2>&1 && { echo "BRIDGE_REFUSED a ladder or exhibition session is running"; exit 2; }
CFG=$(.venv/bin/python tools/deployed_config.py) || { echo "BRIDGE_REFUSED the deployed configuration failed verification"; exit 2; }
eval "$CFG"
export VGC_SET_PRIOR_REG="$SET_PRIOR"
PREVIEW_ARGS=()
[ -n "${PREVIEW_MODEL:-}" ] && PREVIEW_ARGS=(--learned_preview --preview_model "$PREVIEW_MODEL")
MIXING_ARGS=(); [ -n "${MIXING:-}" ] && read -r -a MIXING_ARGS <<< "$MIXING"
STICKY_ARGS=(); [ -n "${STICKY:-}" ] && STICKY_ARGS=(--sticky-corrections)
SHEET_ARGS=(); [ -n "${SHEET_PREVIEW:-}" ] && SHEET_ARGS=(--sheet-preview)
FORECAST_ARGS=(); [ -n "${FORECAST:-}" ] && FORECAST_ARGS=(--opponent-forecast "$FORECAST")
PLAYBOOK_ARGS=(); [ -n "${PLAYBOOK:-}" ] && PLAYBOOK_ARGS=(--playbook "$PLAYBOOK")
exec .venv/bin/python -u evaluation/local_bridge.py "$PORT" "$CLIENT" "$BLOCKS" "$OUT" \
  ${BRIDGE_ARGS[@]+"${BRIDGE_ARGS[@]}"} -- \
  --checkpoint "$CKPT" --reg mc --our_team "$TEAM" \
  --guards-extra "$GUARDS" \
  ${PREVIEW_ARGS[@]+"${PREVIEW_ARGS[@]}"} \
  ${MIXING_ARGS[@]+"${MIXING_ARGS[@]}"} \
  ${STICKY_ARGS[@]+"${STICKY_ARGS[@]}"} \
  ${SHEET_ARGS[@]+"${SHEET_ARGS[@]}"} \
  "$@" \
  ${FORECAST_ARGS[@]+"${FORECAST_ARGS[@]}"} \
  ${PLAYBOOK_ARGS[@]+"${PLAYBOOK_ARGS[@]}"}
