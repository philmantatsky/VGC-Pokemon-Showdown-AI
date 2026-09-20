#!/usr/bin/env bash
# Fast path to the ladder candidate. The pre-registered rule lives in
# tools/ladder_pick.py; this runner executes only the arms the rule asks for,
# in the order it asks (human arm of every finalist, then the PPO arms of the
# leader, falling through on a disqualification). Arm files land where the full
# verdict expects them, so a later run_brainv1_verdict_supervised.sh fills in
# the scripted and diagnostic arms without repeating anything.
# Usage: OUR_TEAM=<team.txt> HUMAN_BC=<eval clone> MIX_BC=<pool clone> \
#          ./evaluation/run_ladder_pick.sh <finalist stem> [<finalist stem> ...]
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
ROOT=${ROOT:-results_gate_battery_brainv1_spec}
DEST=${DEST:-results_brainv1_spec/saves_fp_hs_wt/reg_mc/seed1}
BASE=${BASE:-results_brainv1/saves_fp_hs_wt/reg_mc/seed1/19660800.zip}
OUR_TEAM=${OUR_TEAM:?team file}
HUMAN_BC=${HUMAN_BC:?eval-only Reg M-C clone checkpoint}
MIX_BC=${MIX_BC:?pool Reg M-C clone checkpoint}
STEMS=("$@"); [ ${#STEMS[@]} -gt 0 ] || { echo "DECISION_FAILED no finalists"; exit 3; }
mkdir -p "$ROOT"
last=""
for i in $(seq 1 12); do
  act=$(.venv/bin/python tools/ladder_pick.py --root "$ROOT" --save-dir "$DEST" --generalist "$BASE" --finalists "${STEMS[@]}") || { echo "DECISION_FAILED rule error"; exit 3; }
  echo "DECISION[$i] $act [$(date '+%H:%M:%S')]"
  case "$act" in
    RUN_HUMAN*) arms="human_bc";;
    RUN_PPO*) arms="frozen rotation1 rotation2";;
    PICK*) echo "$act" > "$ROOT/ladder_pick.txt"; echo "LADDER_PICK_DONE [$(date '+%H:%M:%S')]"; exit 0;;
    *) echo "DECISION_FAILED unknown action"; exit 3;;
  esac
  [ "$act" = "$last" ] && { echo "DECISION_FAILED the same action repeated (an arm produced no result): $act"; exit 3; }
  last="$act"
  stem=$(echo "$act" | awk '{print $2}')
  ARMS="$arms" ROOT="$ROOT" BASE="$BASE" OUR_TEAM="$OUR_TEAM" HUMAN_BC="$HUMAN_BC" MIX_BC="$MIX_BC" \
    ./evaluation/run_brainv1_verdict_supervised.sh "$DEST/$stem.zip"
done
echo "DECISION_FAILED too many iterations"; exit 3
