#!/usr/bin/env bash
# Ladder the DEPLOYED configuration (results_deployed/DEPLOYED.json) serially.
# Usage: tools/ladder_deployed.sh <n_games_total> [replay_dir]
# <n_games_total> counts the games already in the replay dir, so "75" after a
# 50-game day plays 25 more. Ladder play needs the user's explicit word; never
# while training or a battery runs (the launcher refuses).
set -uo pipefail
cd "$(dirname "$0")/.."
N=${1:?total games}
# checkpoint, team and (when deployed) preview model are sha-verified by the helper
CFG=$(.venv/bin/python tools/deployed_config.py) || { echo "LADDER_REFUSED the deployed configuration failed verification"; exit 2; }
eval "$CFG"
[ "$REG" = mc ] || { echo "LADDER_REFUSED the deployed configuration is not Reg M-C"; exit 2; }
# the opponent set data this brain was trained with (DEPLOYED.json); recorded per replay dir
export VGC_SET_PRIOR_REG="$SET_PRIOR"
# non-empty only when DEPLOYED.json lets a learned, human-trained model choose our preview
export PREVIEW_MODEL
DIR=${2:-ladder_replays_mc_deployed_$REPLAY_TAG}
GUARDS="$GUARDS" exec ./tools/ladder_read_loop.sh "$CKPT" "$TEAM" "$N" "$DIR"
