#!/usr/bin/env bash
# Brain-v1 verdict (NEW_BRAIN_PLAN M3; same M-C-anchored arms as round 6), arm by arm under the stall watchdog.
# Every arm is anchored to Reg M-C: the M-C opponent team pool and weights,
# our team file, and the Reg M-C eval-only human clone as the human arm. The
# frozen/rotation PPOs are Reg M-B-trained populations piloting M-C teams --
# still never-trained-against, which is what the arms are for. Diagnostic arms:
# the pool clone mc_A (memorisation check) and eval_D (Reg M-B humans).
# Completed arms are skipped on relaunch.
# ARMS (space-separated subset of: heuristic frozen rotation1 rotation2 human_bc mcA
# evalD) runs only those arms; completed arms are always skipped, so a later full run
# fills in the rest.
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
ARMS=${ARMS:-heuristic frozen rotation1 rotation2 human_bc mcA evalD}
want() { case " $ARMS " in *" $1 "*) return 0;; esac; return 1; }
for CAND in "$@"; do
  L=$(basename "$CAND" .zip); S="$ROOT/$L/screening"
  echo "== $L arms: $ARMS [$(date '+%H:%M:%S')]"
  want heuristic && ./evaluation/supervised_eval.sh "$S/battery_heuristic.json" $PORT -- "${COMMON[@]}" --candidate "$CAND"
  want frozen && ./evaluation/supervised_eval.sh "$S/battery_frozen.json" $PORT -- "${COMMON[@]}" --candidate "$CAND" --opponent-checkpoint results_repaired/opponents/64opp_3932160_v4.zip
  want rotation1 && ./evaluation/supervised_eval.sh "$S/battery_rotation1.json" $PORT -- "${COMMON[@]}" --candidate "$CAND" --opponent-checkpoint results_repaired/opponents/8opp_4915200_v4.zip
  want rotation2 && ./evaluation/supervised_eval.sh "$S/battery_rotation2.json" $PORT -- "${COMMON[@]}" --candidate "$CAND" --opponent-checkpoint results_repaired/opponents/tuned_983040_v4.zip
  want human_bc && ./evaluation/supervised_eval.sh "$S/battery_human_bc.json" $PORT -- "${COMMON[@]}" --candidate "$CAND" --opponent-checkpoint "$HUMAN_BC" --opponent-stochastic
  want mcA && ./evaluation/supervised_eval.sh "$ROOT/${L}_mcA/screening/battery_human_bc.json" $PORT -- "${COMMON[@]}" --candidate "$CAND" --opponent-checkpoint "$MIX_BC" --opponent-stochastic
  want evalD && ./evaluation/supervised_eval.sh "$ROOT/${L}_evalD/screening/battery_human_bc.json" $PORT -- "${COMMON[@]}" --candidate "$CAND" --opponent-checkpoint results_bc/eval_D/saves_bc/seed2/4.zip --opponent-stochastic
  # the scorecard needs all five gate arms; a partial run (ARMS subset) skips it
  if ls "$S"/battery_{heuristic,frozen,rotation1,rotation2,human_bc}.json >/dev/null 2>&1; then
    .venv/bin/python evaluation/scorecard_verdict.py "$S" --json "$S/verdict.json" | sed "s/^/VERDICT[$L] /"
  fi
  for X in mcA evalD; do
    [ -f "$ROOT/${L}_$X/screening/battery_human_bc.json" ] || continue
    .venv/bin/python - "$ROOT/${L}_$X/screening/battery_human_bc.json" "$X" "$L" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); c, b = d["arms"]["distilled_policy"], d["arms"]["champion_policy"]
print(f"DIAG[{sys.argv[3]}] {sys.argv[2]}: candidate={c['win_rate']:.3f} baseline={b['win_rate']:.3f} delta={100*(c['win_rate']-b['win_rate']):+.1f}pp")
PY
  done
done
echo "BRAINV1_SUPERVISED_COMPLETE"
