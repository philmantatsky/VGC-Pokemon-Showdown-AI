#!/usr/bin/env bash
# Ladder the DEPLOYED configuration (results_deployed/DEPLOYED.json) serially.
# Usage: tools/ladder_deployed.sh <n_games_total> [replay_dir]
# <n_games_total> counts the games already in the replay dir, so "75" after a
# 50-game day plays 25 more. Ladder play needs the user's explicit word; never
# while training or a battery runs (the launcher refuses).
set -uo pipefail
cd "$(dirname "$0")/.."
N=${1:?total games}; M=results_deployed/DEPLOYED.json
read -r CKPT TEAM GUARDS SHA <<<"$(.venv/bin/python -c "
import json; d=json.load(open('$M'))['deployed']; print(d['checkpoint'], d['team'], d['guards_extra'], d['sha256'])")"
[ "$(shasum -a 256 "$CKPT" | cut -d' ' -f1)" = "$SHA" ] || { echo "LADDER_REFUSED the deployed checkpoint does not match the manifest sha256"; exit 2; }
DIR=${2:-ladder_replays_mc_deployed_$(basename "$TEAM" .txt)}
GUARDS="$GUARDS" exec ./tools/ladder_read_loop.sh "$CKPT" "$TEAM" "$N" "$DIR"
