#!/usr/bin/env bash
set -euo pipefail

# Evaluator retrain for the new-brain era (v3h), chained after the exploiter:
#   1. measure the exploit (champion vs the final exploiter checkpoint, n=1,000)
#   2. generate the outcome dataset with the Stage-E recipe (35/25/25/15
#      human_prior / human_bc / historical / model, 25% our-side-from-pool,
#      preview states labeled) -- our side = the deployed league champion, and
#      the historical rotation now includes the OLD champion and the EXPLOITER
#      so the net learns which positions are the leaky ones
#   3. train the value net with the historical holdout-style gate
#   4. score it on our own ladder games (the standing sim-to-real instrument)
# Runs from the repo root; needs the eval server on 7600.

EXP=results_exploiter/saves_ex_hs_wt/reg_mb/seed1/17694720.zip
CHAMP=results_league/league_champion.zip
OPP=results_repaired/opponents

echo "[$(date '+%H:%M')] 1/4 exploit measurement"
.venv/bin/python evaluation/eval_counterfactual.py \
  --baseline $CHAMP --candidate $CHAMP \
  --opponent-checkpoint $EXP --opponent-stochastic \
  --n-battles 1000 --hidden-sheets --seed 83 --port 7600 --workers 8 \
  --output results_exploiter/exploit_eval_1000.json

echo "[$(date '+%H:%M')] 2/4 outcome dataset"
.venv/bin/python datagen/generate_outcome_dataset.py \
  --checkpoint $CHAMP \
  --opponent-checkpoint $OPP/64opp_3932160_v4.zip $OPP/8opp_4915200_v4.zip $OPP/tuned_983040_v4.zip results_repaired/champion.zip $EXP \
  --human-checkpoint results_bc/mix_A/saves_bc/seed1/30.zip \
  --styles human_prior,human_prior,human_prior,human_prior,human_prior,human_prior,human_prior,human_bc,human_bc,human_bc,human_bc,human_bc,historical,historical,historical,historical,historical,model,model,model \
  --our-team-pool-prob 0.25 --games 20000 --minimum-games 15000 \
  --hidden-sheet-prob 0.5 --workers 8 --seed 20260903 \
  --output outcome_data_v3h

echo "[$(date '+%H:%M')] 3/4 train value net"
mkdir -p results_outcome_v3h
.venv/bin/python training/train_outcome_value.py \
  --data outcome_data_v3h --checkpoint $CHAMP \
  --output results_outcome_v3h/outcome_value.zip \
  --holdout-style historical --seed 20260903

echo "[$(date '+%H:%M')] 4/4 ladder-replay calibration (incumbent vs v3h)"
.venv/bin/python tools/calibrate_vs_ladder.py value \
  --outcome-value results_outcome_v2h/outcome_value.zip > results_outcome_v3h/ladder_calibration_v2h_incumbent.txt 2>&1
.venv/bin/python tools/calibrate_vs_ladder.py value \
  --outcome-value results_outcome_v3h/outcome_value.zip > results_outcome_v3h/ladder_calibration_v3h.txt 2>&1
echo "EVALUATOR_CHAIN_COMPLETE"
