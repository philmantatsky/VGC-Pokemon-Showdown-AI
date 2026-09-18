#!/usr/bin/env bash
# Resume of the brain-v1 round (paused 2026-09-16 09:39 at 4 of 8 saves).
# Fresh 7700, then training continues from the newest save in results_brainv1
# to TOTAL_STEPS (train.py resumes from the max stem and SB3 keeps the run's
# tensorboard directory; Adam restarts, as it did at the original launch).
# Afterwards: triage across BOTH events files of the run, the supervised
# battery on the finalists (MB430 arms, deployed-brain reference), and the
# brain tournament of each finalist on the six candidate teams (NEW_BRAIN_PLAN
# section 4). Markers: BRAINV1_RESUME_START / TRAINING_START / TRAINING_EXITED /
# VERDICT_* / TOURNAMENT_START / CHAIN_COMPLETE; BRAINV1_RESUME_FAILED[step].
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
PY=.venv/bin/python
SAVES=results_brainv1/saves_fp_hs_wt/reg_mc/seed1
FINAL=${TOTAL_STEPS:-20643840}
stamp() { date '+%H:%M:%S'; }
fail() { echo "BRAINV1_RESUME_FAILED[$1] [$(stamp)]"; exit 3; }
restart_server() {
  if pids=$(lsof -nP -t -iTCP:$1 -sTCP:LISTEN 2>/dev/null); then kill $pids 2>/dev/null || true; sleep 3; fi
  (cd pokemon-showdown && node pokemon-showdown start $1 --no-security > /dev/null 2>&1 &)
  for _ in $(seq 1 30); do lsof -nP -iTCP:$1 -sTCP:LISTEN >/dev/null 2>&1 && return 0; sleep 1; done
  return 1
}

pgrep -f "vgc_bench[.]train" >/dev/null 2>&1 && fail already_training
LAST=$(ls $SAVES/*.zip | sed 's#.*/##; s/\.zip$//' | sort -n | tail -1)
echo "BRAINV1_RESUME_START [$(stamp)] from=$LAST to=$FINAL"
[ "$LAST" -lt "$FINAL" ] || fail already_final
$PY training/build_league.py --config training/brainv1_config.json --verify-only || fail verify
restart_server 7700 || fail server_7700
LOG=brainv1_resume_$(date +%H%M%S).log
echo "BRAINV1_TRAINING_START [$(stamp)] log=$LOG"
TOTAL_STEPS=$FINAL caffeinate -is ./training/run_brainv1_training.sh > "$LOG" 2>&1
rc=$?
echo "BRAINV1_TRAINING_EXITED [$(stamp)] exit=$rc final=$([ -f $SAVES/$FINAL.zip ] && echo yes || echo NO)"

# ---- triage + verdict ---------------------------------------------------------
$PY training/triage_league_log.py "$LOG" --save-dir $SAVES --resume 12779520 --results-dir results_brainv1 --from-tensorboard | tee brainv1_triage.txt
FINALISTS=$(grep -E "\.zip$" brainv1_triage.txt | tr '\n' ' ')
[ -n "$FINALISTS" ] || fail no_finalists
CLONE=$(cat results_bc/mc_A_20260913/BEST.txt)
HUMAN=$(cat results_bc/eval_mcB_20260913/BEST.txt)
echo "BRAINV1_VERDICT_START [$(stamp)] $FINALISTS"
HUMAN_BC="$HUMAN" MIX_BC="$CLONE" caffeinate -is ./evaluation/run_brainv1_verdict_supervised.sh $FINALISTS > brainv1_verdict.log 2>&1 && echo "BRAINV1_VERDICT_OK" || echo "BRAINV1_VERDICT_FAILED"
grep -E "VERDICT\[|DIAG\[" brainv1_verdict.log

# ---- brain tournament: each finalist pilots T0..T5 (n=300 per team, as the deployed brain did)
for F in $FINALISTS; do
  L=$(basename "$F" .zip)
  echo "BRAINV1_TOURNAMENT_START [$(stamp)] $L"
  PORT=7600 caffeinate -is ./evaluation/run_team_tournament.sh "$F" 300 "results_team_tournament/brainv1_$L" > "team_tournament_brainv1_$L.log" 2>&1
  grep -E "^TOURNAMENT" "team_tournament_brainv1_$L.log" | sed "s/^/[$L] /"
done
echo "BRAINV1_CHAIN_COMPLETE [$(stamp)]"
