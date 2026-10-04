#!/usr/bin/env bash
# Run the challenge listener and restart it after a lost server connection
# (2026-10-03: Showdown dropped the listener's connection at 17:52 and it sat deaf
# for hours). ladder_ourteam.py exits 75 when its connection drops and appends
# {"unfinished": <battle room or "">} to <replay_dir>/disconnects.jsonl; this loop
# then reruns the same command, adding --rejoin-battle for a battle left unfinished,
# after a pause (REJOIN_PAUSE=5 s when a battle waits, else RECONNECT_PAUSE=30 s).
# Any other exit status ends it (Ctrl-C, or stopping the python process), and so do
# MAX_QUICK=10 short runs in a row (under QUICK_SECONDS=60 s each: the server is
# down or refuses us).
# Usage: tools/reconnect_loop.sh <replay_dir> <rejoin_room_or_empty> <command> [args...]
set -uo pipefail
DIR=$1
ROOM=$2
shift 2
stamp() { date '+%m-%d %H:%M:%S'; }
quick=0
while :; do
  REJOIN=()
  [ -n "$ROOM" ] && REJOIN=(--rejoin-battle "$ROOM")
  started=$(date +%s)
  # guarded: macOS bash 3.2 calls an empty array unbound under set -u
  "$@" ${REJOIN[@]+"${REJOIN[@]}"}
  rc=$?
  [ "$rc" -eq 75 ] || exit "$rc"
  ROOM=$(tail -n 1 "$DIR/disconnects.jsonl" 2>/dev/null \
    | sed -n 's/.*"unfinished": "\(battle-gen9championsvgc2026regmc-[a-z0-9-]*\)".*/\1/p')
  if [ $(($(date +%s) - started)) -lt "${QUICK_SECONDS:-60}" ]; then
    quick=$((quick + 1))
  else
    quick=0
  fi
  if [ "$quick" -ge "${MAX_QUICK:-10}" ]; then
    echo "RECONNECT_GAVE_UP [$(stamp)] $quick short runs in a row"
    exit 75
  fi
  if [ -n "$ROOM" ]; then pause=${REJOIN_PAUSE:-5}; else pause=${RECONNECT_PAUSE:-30}; fi
  echo "RECONNECT [$(stamp)] the server connection was lost; restarting in ${pause}s${ROOM:+, then rejoining $ROOM}"
  sleep "$pause"
done
