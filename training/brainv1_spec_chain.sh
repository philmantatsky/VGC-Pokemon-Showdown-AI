#!/usr/bin/env bash
# Specialist round (NEW_BRAIN_PLAN section 4 step 3; pre-registered in
# PROJECT_STATUS 2026-09-19): the brain-v1 generalist (save 7) continues with
# our side fixed to ONE team. Separate artifacts (results_brainv1_spec/); the
# brain-v1 saves are never touched. Pool: mc_A clone x2, old champion, league-1
# history, deployed x2, save 8, save 7 x2 + resume. Then triage and the paired
# verdict against save 7 ON THE SAME TEAM (does specialising help?); the arms'
# absolute rates give the cross-team read against the deployed brain on MB430.
# Relaunching with saves beyond the resume stem keeps the pool and resumes.
# Env overrides (a later round on the same team, e.g. with a fresh human clone):
#   INIT (learner's start; default brain-v1 save 7), SWAP (save at stem 700),
#   RESUME (INIT's step count), TOTAL_STEPS, SUFFIX (artifacts in results_<SUFFIX>/),
#   CLONE (pool human clone), HUMAN (eval-only clone for the verdict), HUMAN2
#   (a second eval-only clone: extra human arm under <root>_human2), EVAL_ONLY_ROOTS
#   (space-separated clone roots banned from the pool), VERDICT_BASE (paired baseline).
# Usage: ./training/brainv1_spec_chain.sh teams/candidates_mc/<T>.txt
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
TEAM=${1:?team file}; [ -f "$TEAM" ] || { echo "SPEC_FAILED[no_team_file] $TEAM"; exit 2; }
L=$(basename "$TEAM" .txt)
PY=.venv/bin/python
V1=results_brainv1/saves_fp_hs_wt/reg_mc/seed1
INIT=${INIT:-$V1/19660800.zip}; SAVE8=${SWAP:-$V1/20643840.zip}
RESUME=${RESUME:-19660800}; FINAL=${TOTAL_STEPS:-$((RESUME + 8 * 983040))}
SUFFIX=${SUFFIX:-brainv1_spec}; DEST=results_$SUFFIX/saves_fp_hs_wt/reg_mc/seed1
CONFIG=training/${SUFFIX}_config.json; WEIGHTS=data/team_weights_regmc_$SUFFIX.json
EVAL_ONLY_ROOTS=${EVAL_ONLY_ROOTS:-results_bc/eval_mcB results_bc/eval_mcB_20260913}
stamp() { date '+%H:%M:%S'; }
fail() { echo "SPEC_FAILED[$1] [$(stamp)]"; exit 3; }
restart_server() {
  if pids=$(lsof -nP -t -iTCP:$1 -sTCP:LISTEN 2>/dev/null); then kill $pids 2>/dev/null || true; sleep 3; fi
  (cd pokemon-showdown && node pokemon-showdown start $1 --no-security > /dev/null 2>&1 &)
  for _ in $(seq 1 30); do lsof -nP -iTCP:$1 -sTCP:LISTEN >/dev/null 2>&1 && return 0; sleep 1; done
  return 1
}

pgrep -f "vgc_bench[.]train" >/dev/null 2>&1 && fail already_training
[ -f "$INIT" ] && [ -f "$SAVE8" ] || fail missing_brainv1_saves
CLONE=${CLONE:-$(cat results_bc/mc_A_20260913/BEST.txt)}; HUMAN=${HUMAN:-$(cat results_bc/eval_mcB_20260913/BEST.txt)}
VERDICT_BASE=${VERDICT_BASE:-$INIT}
BAN=(); for r in $EVAL_ONLY_ROOTS; do BAN+=(--eval-only-clone "$r"); done
echo "SPEC_START [$(stamp)] team=$L ($TEAM) init=$INIT to=$FINAL"
if [ -f results_$SUFFIX/TEAM.txt ] && [ "$(cat results_$SUFFIX/TEAM.txt)" != "$TEAM" ]; then fail "team_mismatch_$(cat results_$SUFFIX/TEAM.txt)"; fi
if ls $DEST/*.zip 2>/dev/null | sed 's#.*/##; s/\.zip$//' | awk -v r=$RESUME '$1 > r' | grep -q .; then
  echo "SPEC_RESUME [$(stamp)] saves beyond $RESUME exist; pool kept"
else
  $PY training/make_brainv1_config.py --init $INIT --clone "$CLONE" --resume-stem $RESUME --dest $DEST \
    --weights-dest $WEIGHTS --swap-deployed $SAVE8 "${BAN[@]}" --out $CONFIG || fail config
  rm -rf results_$SUFFIX/saves_fp_hs_wt
  $PY training/build_league.py --config $CONFIG > ${SUFFIX}_build.log 2>&1 || { tail -5 ${SUFFIX}_build.log; fail build; }
fi
$PY training/build_league.py --config $CONFIG --verify-only || fail verify
echo "$TEAM" > results_$SUFFIX/TEAM.txt
restart_server 7700 || fail server_7700
LOG=${SUFFIX}_$(date +%H%M%S).log
echo "SPEC_TRAINING_START [$(stamp)] log=$LOG team=$L"
CONFIG=$CONFIG RESULTS_SUFFIX=$SUFFIX TEAM_WEIGHTS=$WEIGHTS OUR_TEAMS="$TEAM" TOTAL_STEPS=$FINAL \
  caffeinate -is ./training/run_brainv1_training.sh > "$LOG" 2>&1
rc=$?
echo "SPEC_TRAINING_EXITED [$(stamp)] exit=$rc final=$([ -f $DEST/$FINAL.zip ] && echo yes || echo NO)"

$PY training/triage_league_log.py "$LOG" --save-dir $DEST --resume $RESUME --results-dir results_$SUFFIX --from-tensorboard | tee ${SUFFIX}_triage.txt
FINALISTS=$(grep -E "\.zip$" ${SUFFIX}_triage.txt | tr '\n' ' ')
[ -n "$FINALISTS" ] || fail no_finalists
echo "SPEC_VERDICT_START [$(stamp)] baseline=$VERDICT_BASE team=$L $FINALISTS"
ROOT=results_gate_battery_$SUFFIX BASE=$VERDICT_BASE OUR_TEAM="$TEAM" HUMAN_BC="$HUMAN" MIX_BC="$CLONE" \
  caffeinate -is ./evaluation/run_brainv1_verdict_supervised.sh $FINALISTS > ${SUFFIX}_verdict.log 2>&1 && echo "SPEC_VERDICT_OK" || echo "SPEC_VERDICT_FAILED"
if [ -n "${HUMAN2:-}" ]; then  # a second eval-only human clone as an extra human arm
  ARMS=human_bc ROOT=results_gate_battery_${SUFFIX}_human2 BASE=$VERDICT_BASE OUR_TEAM="$TEAM" HUMAN_BC="$HUMAN2" MIX_BC="$CLONE" \
    caffeinate -is ./evaluation/run_brainv1_verdict_supervised.sh $FINALISTS >> ${SUFFIX}_verdict.log 2>&1
fi
grep -E "VERDICT\[|DIAG\[" ${SUFFIX}_verdict.log
echo "SPEC_CHAIN_COMPLETE [$(stamp)]"
