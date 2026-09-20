#!/usr/bin/env bash
# Overnight chain after the first T4 ladder read (2026-09-20). Single variable
# against the specialist round: the HUMAN CLONE. The specialist's human-clone arm
# was flat (+0.1pp vs the generalist) while the field moved (Gholdengo +12,
# Raichu +12 points of usage in a week), and a human-like pool opponent is the
# only lever that ever transferred to ladder.
#   1. deferred specialist arms (scripted + mc_A / eval_D diagnostics);
#   2. merged corpus incl. the 09-20 scrape -> NEW dated dir (nothing existing
#      is modified; the battery's team pool and weights stay frozen);
#   3. trajectories A (buckets 0-4) / B (5-9) at the current token length;
#   4. clones mc_A_<DAY> (training_opponent) and eval_mcB_<DAY> (eval_only),
#      best epoch by cross-bucket agreement;
#   5. smoke of the new eval clone under the stall watchdog;
#   6. round "spec2": the T4 specialist continues (+8 intervals) with the new
#      clone in the pool; verdict paired vs the specialist on T4 with BOTH
#      eval-only clones as human arms.
# Refuses to start while a ladder session runs (they never share the machine).
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
PY=.venv/bin/python
DAY=${DAY:-20260920}
TEAM=${TEAM:-teams/candidates_mc/T4.txt}
SPEC=results_brainv1_spec/saves_fp_hs_wt/reg_mc/seed1/27525120.zip
GENERALIST=results_brainv1/saves_fp_hs_wt/reg_mc/seed1/19660800.zip
MERGED=battle_logs_top_mc_merged_$DAY
stamp() { date '+%H:%M:%S'; }
fail() { echo "REFRESH_FAILED[$1] [$(stamp)]"; exit 3; }
pgrep -f "ladder_ourteam[.]py" >/dev/null 2>&1 && fail ladder_running
pgrep -f "vgc_bench[.]train" >/dev/null 2>&1 && fail already_training
echo "REFRESH_START [$(stamp)] day=$DAY"

# ---- 1. deferred arms of the first specialist verdict --------------------------
OLD_A=$(cat results_bc/mc_A_20260913/BEST.txt); OLD_B=$(cat results_bc/eval_mcB_20260913/BEST.txt)
ROOT=results_gate_battery_brainv1_spec BASE=$GENERALIST OUR_TEAM="$TEAM" HUMAN_BC="$OLD_B" MIX_BC="$OLD_A" \
  caffeinate -is ./evaluation/run_brainv1_verdict_supervised.sh $SPEC > brainv1_spec_verdict_full.log 2>&1 || fail deferred_arms
grep -E "VERDICT\[|DIAG\[" brainv1_spec_verdict_full.log
echo "REFRESH_DEFERRED_ARMS_DONE [$(stamp)]"

# ---- 2-3. corpus and trajectories ---------------------------------------------
$PY datagen/merge_battle_logs.py --out $MERGED battle_logs_top_mc_20260909 battle_logs_top_mc_20260910 battle_logs_top_mc_20260913 battle_logs_top_mc_$DAY > merge_mc_$DAY.log 2>&1 || { tail -3 merge_mc_$DAY.log; fail merge; }
tail -2 merge_mc_$DAY.log
LOGS="$MERGED/logs_gen9championsvgc2026regmc.json $MERGED/logs_gen9championsvgc2026regmcbo3.json"
echo "REFRESH_TRAJS_START [$(stamp)]"
$PY -m vgc_bench.logs2trajs --logs $LOGS --out_dir trajs_regmc_human_A_$DAY --buckets 0,1,2,3,4 --num_workers 4 > logs2trajs_mc_A_$DAY.log 2>&1 || fail trajs_A
$PY -m vgc_bench.logs2trajs --logs $LOGS --out_dir trajs_regmc_human_B_$DAY --buckets 5,6,7,8,9 --num_workers 4 > logs2trajs_mc_B_$DAY.log 2>&1 || fail trajs_B
echo "REFRESH_TRAJS_DONE [$(stamp)] A=$(ls trajs_regmc_human_A_$DAY | wc -l | tr -d ' ') B=$(ls trajs_regmc_human_B_$DAY | wc -l | tr -d ' ')"

# ---- 4. clones -----------------------------------------------------------------
agree() { $PY evaluation/eval_bc_agreement.py --policy "$1" --trajs_dir "$2" --sample 800 2>&1 | grep -E "top-1 per-slot" | grep -oE "[0-9]+\.[0-9]+" | head -1; }
A_DIR=results_bc/mc_A_$DAY; B_DIR=results_bc/eval_mcB_$DAY
echo "REFRESH_CLONES_START [$(stamp)]"
caffeinate -is $PY -m vgc_bench.pretrain --run_id 1 --device mps --num_epochs 12 --div_frac 1.0 \
  --init_from results_bc/foundation_converted.zip --trajs_dir trajs_regmc_human_A_$DAY \
  --output_dir $A_DIR --eval_every 0 --port 7611 > pretrain_mc_A_$DAY.log 2>&1 || { tail -5 pretrain_mc_A_$DAY.log; fail pretrain_mc_A; }
BEST_A=""; BEST_A_SCORE=0
for e in 2 3 4 5 6 8 10 12; do
  [ -f $A_DIR/saves_bc/seed1/$e.zip ] || continue
  s=$(agree $A_DIR/saves_bc/seed1/$e.zip trajs_regmc_human_B_$DAY); echo "AGREE mc_A_$DAY epoch $e on B: top-1 $s%"
  if [ -n "$s" ] && [ "$(echo "$s > $BEST_A_SCORE" | bc)" = 1 ]; then BEST_A_SCORE=$s; BEST_A=$A_DIR/saves_bc/seed1/$e.zip; fi
done
[ -n "$BEST_A" ] || fail mc_A_select
$PY tools/stamp_checkpoint_metadata.py "$BEST_A" --role training_opponent || fail stamp_A
caffeinate -is $PY -m vgc_bench.pretrain --run_id 2 --device mps --num_epochs 8 --div_frac 1.0 \
  --init_from results_bc/foundation_converted.zip --trajs_dir trajs_regmc_human_B_$DAY \
  --output_dir $B_DIR --eval_every 0 --port 7611 > pretrain_eval_mcB_$DAY.log 2>&1 || { tail -5 pretrain_eval_mcB_$DAY.log; fail pretrain_eval_mcB; }
BEST_B=""; BEST_B_SCORE=0
for e in 2 3 4 5 6 8; do
  [ -f $B_DIR/saves_bc/seed2/$e.zip ] || continue
  s=$(agree $B_DIR/saves_bc/seed2/$e.zip trajs_regmc_human_A_$DAY); echo "AGREE eval_mcB_$DAY epoch $e on A: top-1 $s%"
  if [ -n "$s" ] && [ "$(echo "$s > $BEST_B_SCORE" | bc)" = 1 ]; then BEST_B_SCORE=$s; BEST_B=$B_DIR/saves_bc/seed2/$e.zip; fi
done
[ -n "$BEST_B" ] || fail eval_mcB_select
$PY tools/stamp_checkpoint_metadata.py "$BEST_B" --role eval_only || fail stamp_B
echo "$BEST_A" > $A_DIR/BEST.txt; echo "$BEST_B" > $B_DIR/BEST.txt
# how the OLD clones agree with the NEW data (did the meta move for them?)
echo "AGREE old mc_A_20260913 on new B: top-1 $(agree "$OLD_A" trajs_regmc_human_B_$DAY)%"
echo "REFRESH_CLONES_DONE [$(stamp)] mc_A=$BEST_A ($BEST_A_SCORE%) eval_mcB=$BEST_B ($BEST_B_SCORE%)"

# ---- 5. smoke of the new eval clone --------------------------------------------
mkdir -p results_gate_battery_mc_smoke
STALL_MIN=5 RETRIES=2 ./evaluation/supervised_eval.sh results_gate_battery_mc_smoke/spec_vs_evalmcB_$DAY.json 7600 -- \
  --baseline "$SPEC" --candidate "$SPEC" --baseline-only --reg mc --team-weights data/team_weights_regmc.json \
  --our-team "$TEAM" --opponent-checkpoint "$BEST_B" --opponent-stochastic --n-battles 40 --hidden-sheets --seed 83 --workers 8 || fail smoke
[ -f results_gate_battery_mc_smoke/spec_vs_evalmcB_$DAY.json ] || fail smoke_missing
$PY -c "
import json; a=json.load(open('results_gate_battery_mc_smoke/spec_vs_evalmcB_$DAY.json'))['arms']['champion_policy']
print(f\"REFRESH_SMOKE specialist vs new eval clone: {a['win_rate']:.3f} ({a['wins']}/{a['battles']})\")"

# ---- 6. round spec2 ---------------------------------------------------------------
echo "REFRESH_SPEC2_LAUNCH [$(stamp)]"
INIT=$SPEC RESUME=27525120 SUFFIX=brainv1_spec2 CLONE="$BEST_A" HUMAN="$BEST_B" HUMAN2="$OLD_B" \
  EVAL_ONLY_ROOTS="results_bc/eval_mcB results_bc/eval_mcB_20260913 results_bc/eval_mcB_$DAY" \
  ./training/brainv1_spec_chain.sh "$TEAM"
echo "REFRESH_CHAIN_COMPLETE [$(stamp)]"
