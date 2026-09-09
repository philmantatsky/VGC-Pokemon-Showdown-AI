#!/usr/bin/env bash
# Round-5 verdict, arm by arm under the stall watchdog. Exploit re-measure is
# INFORMATIVE (the pool held no adversary); every finalist gets the full
# battery + mix_A + eval_D. Completed arms are skipped on relaunch.
# Usage: ./evaluation/run_league5_verdict_supervised.sh <ckpt> [<ckpt> ...]
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
ROOT=results_gate_battery_league5; BASE=results_league/league_champion.zip; PORT=7600
EXP=results_exploiter/saves_ex_hs_wt/reg_mb/seed1/17694720.zip
COMMON=(--baseline $BASE --n-battles 1000 --hidden-sheets --seed 83 --workers 8)
for CAND in "$@"; do
  L=$(basename "$CAND" .zip)
  echo "== $L [$(date '+%H:%M:%S')]"
  ./evaluation/supervised_eval.sh "$ROOT/$L/exploit_remeasure_1000.json" $PORT -- "${COMMON[@]}" --candidate "$CAND" --opponent-checkpoint $EXP --opponent-stochastic
  ./evaluation/supervised_eval.sh "$ROOT/$L/screening/battery_heuristic.json" $PORT -- "${COMMON[@]}" --candidate "$CAND"
  ./evaluation/supervised_eval.sh "$ROOT/$L/screening/battery_frozen.json" $PORT -- "${COMMON[@]}" --candidate "$CAND" --opponent-checkpoint results_repaired/opponents/64opp_3932160_v4.zip
  ./evaluation/supervised_eval.sh "$ROOT/$L/screening/battery_rotation1.json" $PORT -- "${COMMON[@]}" --candidate "$CAND" --opponent-checkpoint results_repaired/opponents/8opp_4915200_v4.zip
  ./evaluation/supervised_eval.sh "$ROOT/$L/screening/battery_rotation2.json" $PORT -- "${COMMON[@]}" --candidate "$CAND" --opponent-checkpoint results_repaired/opponents/tuned_983040_v4.zip
  ./evaluation/supervised_eval.sh "$ROOT/$L/screening/battery_human_bc.json" $PORT -- "${COMMON[@]}" --candidate "$CAND" --opponent-checkpoint results_bc/eval_B/saves_bc/seed2/30.zip --opponent-stochastic
  ./evaluation/supervised_eval.sh "$ROOT/${L}_mixA/screening/battery_human_bc.json" $PORT -- "${COMMON[@]}" --candidate "$CAND" --opponent-checkpoint results_bc/mix_A/saves_bc/seed1/30.zip --opponent-stochastic
  ./evaluation/supervised_eval.sh "$ROOT/${L}_evalD/screening/battery_human_bc.json" $PORT -- "${COMMON[@]}" --candidate "$CAND" --opponent-checkpoint results_bc/eval_D/saves_bc/seed2/4.zip --opponent-stochastic
  .venv/bin/python - "$ROOT/$L/exploit_remeasure_1000.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); c, b = d["arms"]["distilled_policy"], d["arms"]["champion_policy"]
print(f"EXPLOIT_INFO {sys.argv[1]} candidate={c['win_rate']:.3f} deployed={b['win_rate']:.3f} delta={100*(c['win_rate']-b['win_rate']):+.1f}pp")
PY
  .venv/bin/python evaluation/scorecard_verdict.py "$ROOT/$L/screening" --json "$ROOT/$L/screening/verdict.json" | sed "s/^/VERDICT[$L] /"
  for X in mixA evalD; do .venv/bin/python - "$ROOT/${L}_$X/screening/battery_human_bc.json" "$X" "$L" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); c, b = d["arms"]["distilled_policy"], d["arms"]["champion_policy"]
print(f"DIAG[{sys.argv[3]}] {sys.argv[2]}: candidate={c['win_rate']:.3f} deployed={b['win_rate']:.3f} delta={100*(c['win_rate']-b['win_rate']):+.1f}pp")
PY
  done
done
echo "L5_SUPERVISED_COMPLETE"
