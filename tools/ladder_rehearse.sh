#!/usr/bin/env bash
# Dress rehearsal of a ladder trial on a LOCAL server: the deployed configuration as
# tools/deployed_config.py resolves it and the ladder_ourteam.py argument list
# tools/ladder_read_loop.sh would build, played by checks/ladder_rehearsal.py. No
# credentials are sourced and nothing reaches the real server.
# Usage (a local Showdown server on <port>):
#   tools/ladder_rehearse.sh <port> <out dir> <games> <extra ladder_ourteam.py flags>
# <games>: "closed:MC2610,open:MC2566" (a heuristic on those rosters, closed or open
# sheets) or "external:<n>" (see checks/ladder_rehearsal.py). <out dir> is the replay
# directory: give one outside the repository. Before a search trial, twice: once with
# the trial's flags and --search-anchor 1e12 (the null search: evaluation/
# search_audit.py must read 0 changed decisions), once with the trial's own anchor.
set -uo pipefail
cd "$(dirname "$0")/.."
PORT=${1:?port}; OUT=${2:?out dir}; GAMES=${3:?games}; shift 3
EXTRA="$*"
CFG=$(.venv/bin/python tools/deployed_config.py) || { echo "REHEARSAL_REFUSED the deployed configuration failed verification"; exit 2; }
eval "$CFG"
export VGC_SET_PRIOR_REG="$SET_PRIOR"
EXTRA_ARGS=()
[ -n "$EXTRA" ] && read -r -a EXTRA_ARGS <<< "$EXTRA"
case "$GAMES" in external:*) N=${GAMES#external:} ;; *) N=$(echo "$GAMES" | tr ',' '\n' | wc -l | tr -d ' ') ;; esac
PREVIEW_ARGS=()
[ -n "${PREVIEW_MODEL:-}" ] && PREVIEW_ARGS=(--learned_preview --preview_model "$PREVIEW_MODEL")
STICKY_ARGS=(); [ -n "${STICKY:-}" ] && STICKY_ARGS=(--sticky-corrections)
SHEET_ARGS=(); [ -n "${SHEET_PREVIEW:-}" ] && SHEET_ARGS=(--sheet-preview)
FORECAST_ARGS=(); [ -n "${FORECAST:-}" ] && FORECAST_ARGS=(--opponent-forecast "$FORECAST")
PLAYBOOK_ARGS=(); [ -n "${PLAYBOOK:-}" ] && PLAYBOOK_ARGS=(--playbook "$PLAYBOOK")
mkdir -p "$OUT"
exec .venv/bin/python -u checks/ladder_rehearsal.py "$PORT" "$GAMES" -- \
  --checkpoint "$CKPT" --reg mc --our_team "$TEAM" \
  --guards-extra "$GUARDS" --n_games "$N" --replay_dir "$OUT" \
  ${PREVIEW_ARGS[@]+"${PREVIEW_ARGS[@]}"} \
  ${STICKY_ARGS[@]+"${STICKY_ARGS[@]}"} \
  ${SHEET_ARGS[@]+"${SHEET_ARGS[@]}"} \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  ${FORECAST_ARGS[@]+"${FORECAST_ARGS[@]}"} \
  ${PLAYBOOK_ARGS[@]+"${PLAYBOOK_ARGS[@]}"}
