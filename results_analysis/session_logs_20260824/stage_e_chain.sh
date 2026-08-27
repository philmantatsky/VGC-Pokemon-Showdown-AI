#!/bin/zsh
# Stage-E overnight chain: train the human-grounded value net, gate it, then
# re-gate the exact preview teacher with it, then the Garchomp bring A/B.
set -x
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
PY=.venv/bin/python
NEW=results_outcome_v2h/outcome_value.zip

# 1. train (mps), historical style held out of training as the style gate
$PY train_outcome_value.py --data outcome_data_v2h --output $NEW \
  --holdout-style historical --device mps --minimum-states 25000 || exit 1

# 2. stamp + ladder-Brier gate (the standing instrument; must beat 0.2141)
$PY stamp_checkpoint_metadata.py $NEW --role outcome_value
VGC_KNOWLEDGE_OBS=1 $PY calibrate_vs_ladder.py value --outcome-value $NEW \
  > results_outcome_v2h/ladder_calibration.txt 2>&1
cat results_outcome_v2h/ladder_calibration.txt | tail -12

# 3. tactical stack regressions (decision stack unchanged, but prove it)
$PY evaluate_tactical_gate.py > results_outcome_v2h/tactical_gate.txt 2>&1 || true
tail -3 results_outcome_v2h/tactical_gate.txt

# 4. exact preview re-gate with the NEW net: champion preview vs exact preview,
#    n=300 hidden (the old net lost this 44/60 vs 53/60 at n=60)
$PY eval_counterfactual.py --candidate results_repaired/champion.zip \
  --include-exact-preview --preview-comparison-only \
  --n-battles 300 --hidden-sheets --port 7600 --seed 303 \
  --outcome-value $NEW \
  --output results_outcome_v2h/exact_preview_regate_hidden300.json || true

# 5. Garchomp bring A/B, local-first: vs learned population and vs the
#    quarantined human arm (stochastic), n=500 each
$PY eval_counterfactual.py --candidate results_repaired/champion.zip \
  --candidate-bench-species garchomp \
  --n-battles 500 --hidden-sheets --port 7600 --seed 404 \
  --opponent-checkpoint results_repaired/opponents/64opp_3932160_v4.zip \
  --output results_stage_e/bench_garchomp_population_500.json || true
$PY eval_counterfactual.py --candidate results_repaired/champion.zip \
  --candidate-bench-species garchomp \
  --n-battles 500 --hidden-sheets --port 7600 --seed 404 \
  --opponent-checkpoint results_bc/eval_B/saves_bc/seed2/30.zip --opponent-stochastic \
  --output results_stage_e/bench_garchomp_humanbc_500.json || true
echo "STAGE E CHAIN COMPLETE"
