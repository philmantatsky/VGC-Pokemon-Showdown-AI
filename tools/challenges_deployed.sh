#!/usr/bin/env bash
# Accept direct Pokemon Showdown challenges using the immutable deployed Reg M-C
# configuration. These are unranked challenge battles, not ladder matchmaking.
#
# Usage: tools/challenges_deployed.sh [n_challenges] [replay_dir] [rejoin_battle]
# Defaults to a long-lived 1000-challenge listener. Stop it normally with Ctrl-C.
set -euo pipefail
cd "$(dirname "$0")/.."

N=${1:-1000}
DIR_ARG=${2:-}
REJOIN=${3:-}
MANIFEST=results_deployed/DEPLOYED.json

case "$N" in
  ''|*[!0-9]*) echo "CHALLENGE_REFUSED n_challenges must be a positive integer"; exit 2 ;;
esac
[ "$N" -gt 0 ] || { echo "CHALLENGE_REFUSED n_challenges must be positive"; exit 2; }

read -r CKPT TEAM GUARDS CKPT_SHA TEAM_SHA REG FORMAT SET_PRIOR <<<"$(.venv/bin/python -c "
import json
d=json.load(open('$MANIFEST'))['deployed']
print(d['checkpoint'], d['team'], d['guards_extra'], d['sha256'], d['team_sha256'], d['reg'], d['format'], d.get('set_prior_reg', 'mc'))")"
# the opponent set data this brain was trained with (DEPLOYED.json)
export VGC_SET_PRIOR_REG="$SET_PRIOR"

# default replay dir follows the deployed team (T4 until 2026-09-23, then T6)
DIR=${DIR_ARG:-challenge_replays_mc_deployed_$(basename "$TEAM" .txt)}

[ "$REG" = mc ] && [ "$FORMAT" = gen9championsvgc2026regmc ] || {
  echo "CHALLENGE_REFUSED deployed configuration is not Reg M-C"
  exit 2
}
[ "$(shasum -a 256 "$CKPT" | cut -d' ' -f1)" = "$CKPT_SHA" ] || {
  echo "CHALLENGE_REFUSED deployed checkpoint hash mismatch"
  exit 2
}
[ "$(shasum -a 256 "$TEAM" | cut -d' ' -f1)" = "$TEAM_SHA" ] || {
  echo "CHALLENGE_REFUSED deployed team hash mismatch"
  exit 2
}

HEAVY='vgc_bench[.]train|run_gate_battery|eval_counterfactual[.]py|run_counterfactual_pipeline|generate_counterfactuals|vgc_bench[.]pretrain|logs2trajs|run_team_tournament|run_team_grid|run_team_confirmation|run_t6_confirmation|run_t6_vs_deployed|opening_study[.]py|run_t6_|run_set_prior_ablation|run_candidate_vs_t6|learned_preview_study|human_preview|preview_entropy[.]py'
pgrep -f "$HEAVY" >/dev/null 2>&1 && {
  echo "CHALLENGE_REFUSED a heavy local job is running"
  exit 2
}
pgrep -f "ladder_ourteam[.]py" >/dev/null 2>&1 && {
  echo "CHALLENGE_REFUSED another ladder/challenge session is running"
  exit 2
}

# Credentials stay shell-side and are never printed or passed as arguments.
set -a
source "../Laplace-Pokemon-Showdown-AI/.env"
set +a

mkdir -p "$DIR"
echo "CHALLENGE_LISTENER format=$FORMAT team=$(basename "$TEAM") limit=$N replays=$DIR"
EXTRA=()
if [ -n "$REJOIN" ]; then
  case "$REJOIN" in
    battle-gen9championsvgc2026regmc-*) EXTRA=(--rejoin-battle "$REJOIN") ;;
    *) echo "CHALLENGE_REFUSED rejoin room is not Reg M-C"; exit 2 ;;
  esac
fi
GUARDS="$GUARDS" exec caffeinate -is .venv/bin/python -u ladder_ourteam.py \
  --checkpoint "$CKPT" \
  --reg mc \
  --our_team "$TEAM" \
  --guards-extra "$GUARDS" \
  --challenges \
  --n_games "$N" \
  --replay_dir "$DIR" \
  "${EXTRA[@]}"
