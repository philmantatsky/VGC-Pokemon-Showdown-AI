#!/usr/bin/env bash
# Specialist round (NEW_BRAIN_PLAN section 4 step 3; pre-registered in
# PROJECT_STATUS 2026-09-19): the brain-v1 generalist (save 7) continues with
# our side fixed to ONE team. Separate artifacts (results_brainv1_spec/); the
# brain-v1 saves are never touched. Pool: mc_A clone x2, old champion, league-1
# history, deployed x2, save 8, save 7 x2 + resume. Then triage and the paired
# verdict against save 7 ON THE SAME TEAM (does specialising help?); the arms'
# absolute rates give the cross-team read against the deployed brain on MB430.
# Relaunching with saves beyond the resume stem keeps the pool and resumes.
# Usage: ./training/brainv1_spec_chain.sh teams/candidates_mc/<T>.txt
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
TEAM=${1:?team file}; [ -f "$TEAM" ] || { echo "SPEC_FAILED[no_team_file] $TEAM"; exit 2; }
L=$(basename "$TEAM" .txt)
PY=.venv/bin/python
V1=results_brainv1/saves_fp_hs_wt/reg_mc/seed1
INIT=$V1/19660800.zip; SAVE8=$V1/20643840.zip
RESUME=19660800; FINAL=${TOTAL_STEPS:-27525120}
SUFFIX=brainv1_spec; DEST=results_$SUFFIX/saves_fp_hs_wt/reg_mc/seed1
CONFIG=training/brainv1_spec_config.json; WEIGHTS=data/team_weights_regmc_brainv1_spec.json
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
CLONE=$(cat results_bc/mc_A_20260913/BEST.txt); HUMAN=$(cat results_bc/eval_mcB_20260913/BEST.txt)
echo "SPEC_START [$(stamp)] team=$L ($TEAM) init=$INIT to=$FINAL"
if [ -f results_$SUFFIX/TEAM.txt ] && [ "$(cat results_$SUFFIX/TEAM.txt)" != "$TEAM" ]; then fail "team_mismatch_$(cat results_$SUFFIX/TEAM.txt)"; fi
if ls $DEST/*.zip 2>/dev/null | sed 's#.*/##; s/\.zip$//' | awk -v r=$RESUME '$1 > r' | grep -q .; then
  echo "SPEC_RESUME [$(stamp)] saves beyond $RESUME exist; pool kept"
else
  $PY training/make_brainv1_config.py --init $INIT --clone "$CLONE" --resume-stem $RESUME --dest $DEST \
    --weights-dest $WEIGHTS --swap-deployed $SAVE8 --out $CONFIG || fail config
  rm -rf results_$SUFFIX/saves_fp_hs_wt
  $PY training/build_league.py --config $CONFIG > brainv1_spec_build.log 2>&1 || { tail -5 brainv1_spec_build.log; fail build; }
fi
$PY training/build_league.py --config $CONFIG --verify-only || fail verify
echo "$TEAM" > results_$SUFFIX/TEAM.txt
restart_server 7700 || fail server_7700
LOG=brainv1_spec_$(date +%H%M%S).log
echo "SPEC_TRAINING_START [$(stamp)] log=$LOG team=$L"
CONFIG=$CONFIG RESULTS_SUFFIX=$SUFFIX TEAM_WEIGHTS=$WEIGHTS OUR_TEAMS="$TEAM" TOTAL_STEPS=$FINAL \
  caffeinate -is ./training/run_brainv1_training.sh > "$LOG" 2>&1
rc=$?
echo "SPEC_TRAINING_EXITED [$(stamp)] exit=$rc final=$([ -f $DEST/$FINAL.zip ] && echo yes || echo NO)"

$PY training/triage_league_log.py "$LOG" --save-dir $DEST --resume $RESUME --results-dir results_$SUFFIX --from-tensorboard | tee brainv1_spec_triage.txt
FINALISTS=$(grep -E "\.zip$" brainv1_spec_triage.txt | tr '\n' ' ')
[ -n "$FINALISTS" ] || fail no_finalists
echo "SPEC_VERDICT_START [$(stamp)] baseline=$INIT team=$L $FINALISTS"
ROOT=results_gate_battery_$SUFFIX BASE=$INIT OUR_TEAM="$TEAM" HUMAN_BC="$HUMAN" MIX_BC="$CLONE" \
  caffeinate -is ./evaluation/run_brainv1_verdict_supervised.sh $FINALISTS > brainv1_spec_verdict.log 2>&1 && echo "SPEC_VERDICT_OK" || echo "SPEC_VERDICT_FAILED"
grep -E "VERDICT\[|DIAG\[" brainv1_spec_verdict.log
echo "SPEC_CHAIN_COMPLETE [$(stamp)]"
