#!/usr/bin/env bash
# Brain-v1 verdict (NEW_BRAIN_PLAN M3; same M-C-anchored arms as round 6), arm by arm under the stall watchdog.
# Every arm is anchored to Reg M-C: the M-C opponent team pool and weights,
# our team file, and the Reg M-C eval-only human clone as the human arm. The
# frozen/rotation PPOs are Reg M-B-trained populations piloting M-C teams --
# still never-trained-against, which is what the arms are for. Diagnostic arms:
# the pool clone mc_A (memorisation check) and eval_D (Reg M-B humans).
# Completed arms are skipped on relaunch.
# Env overrides (the specialist round: baseline = the brain-v1 generalist on the
# chosen team): ROOT (output root), BASE (baseline checkpoint), OUR_TEAM, PORT.
# Usage: HUMAN_BC=<eval_mcB ckpt> MIX_BC=<mc_A ckpt> ./evaluation/run_brainv1_verdict_supervised.sh <ckpt> [<ckpt> ...]
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
ROOT=${ROOT:-results_gate_battery_brainv1}; BASE=${BASE:-results_league/league_champion.zip}; PORT=${PORT:-7600}
OUR_TEAM=${OUR_TEAM:-teams/reg_mc/our_team.txt}
HUMAN_BC=${HUMAN_BC:?eval-only Reg M-C clone checkpoint}
MIX_BC=${MIX_BC:?pool Reg M-C clone checkpoint}
COMMON=(--baseline $BASE --n-battles 1000 --hidden-sheets --seed 83 --workers 8
        --reg mc --team-weights data/team_weights_regmc.json --our-team "$OUR_TEAM")
for CAND in "$@"; do
  L=$(basename "$CAND" .zip)
  echo "== $L [$(date '+%H:%M:%S')]"
  ./evaluation/supervised_eval.sh "$ROOT/$L/screening/battery_heuristic.json" $PORT -- "${COMMON[@]}" --candidate "$CAND"
  ./evaluation/supervised_eval.sh "$ROOT/$L/screening/battery_frozen.json" $PORT -- "${COMMON[@]}" --candidate "$CAND" --opponent-checkpoint results_repaired/opponents/64opp_3932160_v4.zip
  ./evaluation/supervised_eval.sh "$ROOT/$L/screening/battery_rotation1.json" $PORT -- "${COMMON[@]}" --candidate "$CAND" --opponent-checkpoint results_repaired/opponents/8opp_4915200_v4.zip
  ./evaluation/supervised_eval.sh "$ROOT/$L/screening/battery_rotation2.json" $PORT -- "${COMMON[@]}" --candidate "$CAND" --opponent-checkpoint results_repaired/opponents/tuned_983040_v4.zip
  ./evaluation/supervised_eval.sh "$ROOT/$L/screening/battery_human_bc.json" $PORT -- "${COMMON[@]}" --candidate "$CAND" --opponent-checkpoint "$HUMAN_BC" --opponent-stochastic
  ./evaluation/supervised_eval.sh "$ROOT/${L}_mcA/screening/battery_human_bc.json" $PORT -- "${COMMON[@]}" --candidate "$CAND" --opponent-checkpoint "$MIX_BC" --opponent-stochastic
  ./evaluation/supervised_eval.sh "$ROOT/${L}_evalD/screening/battery_human_bc.json" $PORT -- "${COMMON[@]}" --candidate "$CAND" --opponent-checkpoint results_bc/eval_D/saves_bc/seed2/4.zip --opponent-stochastic
  .venv/bin/python evaluation/scorecard_verdict.py "$ROOT/$L/screening" --json "$ROOT/$L/screening/verdict.json" | sed "s/^/VERDICT[$L] /"
  for X in mcA evalD; do .venv/bin/python - "$ROOT/${L}_$X/screening/battery_human_bc.json" "$X" "$L" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); c, b = d["arms"]["distilled_policy"], d["arms"]["champion_policy"]
print(f"DIAG[{sys.argv[3]}] {sys.argv[2]}: candidate={c['win_rate']:.3f} baseline={b['win_rate']:.3f} delta={100*(c['win_rate']-b['win_rate']):+.1f}pp")
PY
  done
done
echo "BRAINV1_SUPERVISED_COMPLETE"
