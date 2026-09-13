#!/usr/bin/env bash
# Round-6b verdict, reordered after the mcA diagnostic arm ran at 79
# decisions/min (2h19m for 11,000 decisions; the gate arms ran ~1,900/min).
# 1) finalist 1's fresh-seed confirmation of its one CONFIRM arm (heuristic),
# 2) finalist 2's five gate arms + any confirmations, 3) eval_D diagnostics,
# 4) the mcA diagnostic last and bounded to 300 battles. Completed outputs skip.
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
ROOT=results_gate_battery_league6; BASE=results_league/league_champion.zip; PORT=7600
SAVES=results_league6/saves_fp_hs_wt/reg_mc/seed1
HUMAN_BC=$(cat results_bc/eval_mcB_20260913/BEST.txt); MIX_BC=$(cat results_bc/mc_A_20260913/BEST.txt)
COMMON=(--baseline $BASE --hidden-sheets --workers 8 --reg mc --team-weights data/team_weights_regmc.json --our-team teams/reg_mc/our_team.txt)
arm_extra() {
  case "$1" in
    heuristic) echo "" ;;
    frozen) echo "--opponent-checkpoint results_repaired/opponents/64opp_3932160_v4.zip" ;;
    rotation1) echo "--opponent-checkpoint results_repaired/opponents/8opp_4915200_v4.zip" ;;
    rotation2) echo "--opponent-checkpoint results_repaired/opponents/tuned_983040_v4.zip" ;;
    human_bc) echo "--opponent-checkpoint $HUMAN_BC --opponent-stochastic" ;;
  esac
}
screen_and_confirm() {  # $1 = stem
  local L=$1 CAND=$SAVES/$1.zip
  echo "== $L gate arms [$(date '+%H:%M:%S')]"
  for ARM in heuristic frozen rotation1 rotation2 human_bc; do
    ./evaluation/supervised_eval.sh "$ROOT/$L/screening/battery_$ARM.json" $PORT -- "${COMMON[@]}" --candidate "$CAND" --n-battles 1000 --seed 83 $(arm_extra $ARM)
  done
  .venv/bin/python evaluation/scorecard_verdict.py "$ROOT/$L/screening" --json "$ROOT/$L/screening/verdict.json" | sed "s/^/VERDICT[$L] /"
  local ARMS
  ARMS=$(.venv/bin/python -c "import json; print(' '.join(json.load(open('$ROOT/$L/screening/verdict.json')).get('confirm_needed', [])))")
  echo "CONFIRM_ARMS $L '${ARMS}'"
  local FLAGS=()
  for ARM in $ARMS; do
    OUT="$ROOT/$L/confirm_${ARM}_1500_seed8302.json"
    ./evaluation/supervised_eval.sh "$OUT" $PORT -- "${COMMON[@]}" --candidate "$CAND" --n-battles 1500 --seed 8302 $(arm_extra $ARM)
    [ -f "$OUT" ] && FLAGS+=(--confirm "$ARM=$OUT")
  done
  if [ ${#FLAGS[@]} -gt 0 ]; then
    .venv/bin/python evaluation/scorecard_verdict.py "$ROOT/$L/screening" "${FLAGS[@]}" --json "$ROOT/$L/screening/verdict_confirmed.json" | sed "s/^/FINAL[$L] /"
  else
    echo "FINAL[$L] no confirmation needed; screening verdict stands"
  fi
}
diag() {  # $1 = stem, $2 = label, $3 = ckpt, $4 = n
  local L=$1
  ./evaluation/supervised_eval.sh "$ROOT/${L}_$2/screening/battery_human_bc.json" $PORT -- "${COMMON[@]}" --candidate "$SAVES/$L.zip" --n-battles "$4" --seed 83 --opponent-checkpoint "$3" --opponent-stochastic
  .venv/bin/python - "$ROOT/${L}_$2/screening/battery_human_bc.json" "$2" "$L" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1])); c, b = d["arms"]["distilled_policy"], d["arms"]["champion_policy"]
    print(f"DIAG[{sys.argv[3]}] {sys.argv[2]}: candidate={c['win_rate']:.3f} deployed={b['win_rate']:.3f} delta={100*(c['win_rate']-b['win_rate']):+.1f}pp n={c['battles']}")
except Exception as exc:
    print(f"DIAG[{sys.argv[3]}] {sys.argv[2]}: missing ({exc})")
PY
}
screen_and_confirm 13762560
screen_and_confirm 17694720
for L in 13762560 17694720; do diag $L evalD results_bc/eval_D/saves_bc/seed2/4.zip 1000; done
rm -f "$ROOT/13762560_mcA/screening/battery_human_bc.json"
for L in 13762560 17694720; do diag $L mcA "$MIX_BC" 300; done
echo "L6_VERDICT_COMPLETE [$(date '+%H:%M:%S')]"
