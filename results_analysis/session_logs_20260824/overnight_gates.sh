#!/bin/zsh
# Overnight local gates: Stage C.1 reliability floor + C.2 KO promotion counting.
# All arms vs the learned population (the arm that catches overfitting) plus one
# heuristic pass; server on 7600 must be running.
set -x
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
PY=.venv/bin/python
CH=results_repaired/champion.zip
OUT=results_stage_c
mkdir -p $OUT

# --- C.1 reliability floor: paired arms in ONE process (per-arm floor flag) ---
$PY eval_counterfactual.py --candidate $CH --candidate-reliability-floor 0.35 \
  --n-battles 300 --hidden-sheets --port 7600 --seed 101 \
  --output $OUT/floor035_heuristic_300.json
$PY eval_counterfactual.py --candidate $CH --candidate-reliability-floor 0.35 \
  --n-battles 300 --hidden-sheets --port 7600 --seed 101 \
  --opponent-checkpoint results_repaired/opponents/64opp_3932160_v4.zip \
  --output $OUT/floor035_population_300.json
$PY eval_counterfactual.py --candidate $CH --candidate-reliability-floor 0.20 \
  --n-battles 300 --hidden-sheets --port 7600 --seed 101 \
  --opponent-checkpoint results_repaired/opponents/64opp_3932160_v4.zip \
  --output $OUT/floor020_population_300.json
$PY eval_counterfactual.py --candidate $CH --candidate-reliability-floor 0.50 \
  --n-battles 300 --hidden-sheets --port 7600 --seed 101 \
  --opponent-checkpoint results_repaired/opponents/64opp_3932160_v4.zip \
  --output $OUT/floor050_population_300.json
# open-sheet no-op check: floor must change nothing when everything is revealed
$PY eval_counterfactual.py --candidate $CH --candidate-reliability-floor 0.35 \
  --n-battles 300 --port 7600 --seed 101 \
  --output $OUT/floor035_open_300.json

# --- C.2 KO promotion: off vs robust, logged + replayed for realized-KO counting ---
VGC_KO_PROMOTION_MODE=off $PY eval_counterfactual.py --candidate $CH \
  --n-battles 300 --hidden-sheets --port 7600 --seed 202 \
  --opponent-checkpoint results_repaired/opponents/64opp_3932160_v4.zip \
  --replay-dir $OUT/ko_off_replays \
  --output $OUT/ko_off_population_300.json
VGC_KO_PROMOTION_MODE=robust $PY eval_counterfactual.py --candidate $CH \
  --n-battles 300 --hidden-sheets --port 7600 --seed 202 \
  --opponent-checkpoint results_repaired/opponents/64opp_3932160_v4.zip \
  --replay-dir $OUT/ko_robust_replays \
  --output $OUT/ko_robust_population_300.json
echo "OVERNIGHT GATES COMPLETE"
