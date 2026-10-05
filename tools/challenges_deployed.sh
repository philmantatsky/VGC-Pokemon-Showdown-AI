#!/usr/bin/env bash
# Accept direct Pokemon Showdown challenges using the immutable deployed Reg M-C
# configuration. These are unranked challenge battles, not ladder matchmaking.
#
# Usage: [ALLOW_HEAVY=1] tools/challenges_deployed.sh [n_challenges] [replay_dir] [rejoin_battle]
# Defaults to a long-lived 1000-challenge listener. Stop it normally with Ctrl-C
# (detached: stop its ladder_ourteam.py process). After a lost server connection it
# reconnects and rejoins an unfinished battle (tools/reconnect_loop.sh; each restart
# accepts up to n_challenges again).
# It refuses while a heavy local job runs unless ALLOW_HEAVY=1.
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

# checkpoint, team and (when deployed) preview model are sha-verified by the helper
CFG=$(.venv/bin/python tools/deployed_config.py --manifest "$MANIFEST") || {
  echo "CHALLENGE_REFUSED the deployed configuration failed verification"
  exit 2
}
eval "$CFG"
# the opponent set data this brain was trained with (DEPLOYED.json)
export VGC_SET_PRIOR_REG="$SET_PRIOR"

# default replay dir follows the deployed configuration (replay_tag, else the team)
DIR=${DIR_ARG:-challenge_replays_mc_deployed_$REPLAY_TAG}

[ "$REG" = mc ] && [ "$FORMAT" = gen9championsvgc2026regmc ] || {
  echo "CHALLENGE_REFUSED deployed configuration is not Reg M-C"
  exit 2
}

HEAVY='vgc_bench[.]train|run_gate_battery|eval_counterfactual[.]py|run_counterfactual_pipeline|generate_counterfactuals|vgc_bench[.]pretrain|logs2trajs|run_team_tournament|run_team_grid|run_team_confirmation|run_t6_confirmation|run_t6_vs_deployed|opening_study[.]py|run_t6_|run_set_prior_ablation|run_candidate_vs_t6|learned_preview_study|human_preview|preview_entropy[.]py|mirror_guard_ab|run_guard_ab|gen_tactical_data[.]py|tactical_sft[.]py'
if pgrep -f "$HEAVY" >/dev/null 2>&1; then
  # ALLOW_HEAVY=1 (the user's word, 2026-10-03): challenge games are unranked, so a
  # listener may share the machine with training; its moves may come more slowly.
  if [ "${ALLOW_HEAVY:-}" = 1 ]; then
    echo "CHALLENGE_WARNING a heavy local job is running; ALLOW_HEAVY=1 starts the listener anyway"
  else
    echo "CHALLENGE_REFUSED a heavy local job is running (ALLOW_HEAVY=1 overrides)"
    exit 2
  fi
fi
pgrep -f "ladder_ourteam[.]py" >/dev/null 2>&1 && {
  echo "CHALLENGE_REFUSED another ladder/challenge session is running"
  exit 2
}

# Credentials stay shell-side and are never printed or passed as arguments.
set -a
source "../Laplace-Pokemon-Showdown-AI/.env"
set +a

mkdir -p "$DIR"
echo "CHALLENGE_LISTENER format=$FORMAT team=$(basename "$TEAM") preview=${PREVIEW_MODEL:-policy} forecast=${FORECAST:-off} limit=$N replays=$DIR"
EXTRA=()
# a learned, human-trained model chooses our preview when DEPLOYED.json says so
if [ -n "$PREVIEW_MODEL" ]; then
  EXTRA+=(--learned_preview --preview_model "$PREVIEW_MODEL")
fi
# mixed-strategy play when DEPLOYED.json turns it on
if [ -n "$MIXING" ]; then
  read -r -a MIXING_ARGS <<< "$MIXING"
  EXTRA+=(${MIXING_ARGS[@]+"${MIXING_ARGS[@]}"})
fi
# guard corrections the reranker may not undo, when DEPLOYED.json says so
if [ -n "${STICKY:-}" ]; then
  EXTRA+=(--sticky-corrections)
fi
# their open team sheet shapes our preview, when DEPLOYED.json says so
if [ -n "${SHEET_PREVIEW:-}" ]; then
  EXTRA+=(--sheet-preview)
fi
# our own plan cards at team preview, when DEPLOYED.json has them
if [ -n "${PLAYBOOK:-}" ]; then
  EXTRA+=(--playbook "$PLAYBOOK")
fi
# shadow mode: log the opponent forecast with every decision, when DEPLOYED.json has it
if [ -n "${FORECAST:-}" ]; then
  EXTRA+=(--opponent-forecast "$FORECAST")
fi
if [ -n "$REJOIN" ]; then
  case "$REJOIN" in
    battle-gen9championsvgc2026regmc-*) ;;  # the loop passes --rejoin-battle once
    *) echo "CHALLENGE_REFUSED rejoin room is not Reg M-C"; exit 2 ;;
  esac
fi
GUARDS="$GUARDS" exec tools/reconnect_loop.sh "$DIR" "$REJOIN" \
  caffeinate -is .venv/bin/python -u ladder_ourteam.py \
  --checkpoint "$CKPT" \
  --reg mc \
  --our_team "$TEAM" \
  --guards-extra "$GUARDS" \
  --challenges \
  --n_games "$N" \
  --replay_dir "$DIR" \
  ${EXTRA[@]+"${EXTRA[@]}"}  # guarded: macOS bash 3.2 calls an empty array unbound under set -u
