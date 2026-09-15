#!/usr/bin/env bash
# Brain v1 round (NEW_BRAIN_PLAN M3), launched only once the round-6 verdict
# and the deployed-brain team tournament have finished (the merge changes the
# observation length for every process that imports the package):
#   merge brain-v1 -> full unit suite in the main checkout -> baseline of the
#   deployed brain on the candidate teams (kill-rule reference) -> pool build
#   -> fresh 7700 -> team-agnostic training (+8 intervals) -> triage ->
#   supervised battery on the M-C arms.
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
PY=.venv/bin/python
stamp() { date '+%H:%M:%S'; }
fail() { echo "BRAINV1_FAILED[$1] [$(stamp)]"; exit 3; }
restart_server() {
  if pids=$(lsof -nP -t -iTCP:$1 -sTCP:LISTEN 2>/dev/null); then kill $pids 2>/dev/null || true; sleep 3; fi
  (cd pokemon-showdown && node pokemon-showdown start $1 --no-security > /dev/null 2>&1 &)
  for _ in $(seq 1 30); do lsof -nP -iTCP:$1 -sTCP:LISTEN >/dev/null 2>&1 && return 0; sleep 1; done
  return 1
}

echo "[$(stamp)] waiting for the round-6 verdict and the team tournament to finish"
until grep -q "L6_VERDICT_COMPLETE" league6_verdict2.log 2>/dev/null && grep -q "TOURNAMENT_DONE" team_tournament_deployed.log 2>/dev/null; do sleep 120; done
while pgrep -f "eval_counterfactual[.]py" >/dev/null 2>&1; do sleep 30; done
echo "BRAINV1_START [$(stamp)]"

# ---- merge between runs ------------------------------------------------------
git status --short | grep -v "^ m" | grep -q "^ M\|^M " && fail dirty_tree
git merge --no-ff brain-v1 -m "Merge brain-v1: joint-action head, threat block, reward shaping, upgrade-on-load, team-agnostic training" > merge_brainv1.log 2>&1 || { tail -5 merge_brainv1.log; fail merge; }
echo "BRAINV1_MERGED [$(stamp)] $(git log --oneline -1)"
$PY -m pytest unit_tests -q > brainv1_suite.log 2>&1; tail -2 brainv1_suite.log | sed 's/^/SUITE /'
grep -q " failed" brainv1_suite.log && { grep -E "^FAILED" brainv1_suite.log | head -8; fail suite; }

# ---- kill-rule reference: deployed brain vs the scripted opponent per candidate team
restart_server 7610 || fail server_7610
echo "BRAINV1_BASELINE_START [$(stamp)]"
mkdir -p results_brainv1
for T in teams/candidates_mc/T*.txt; do
  L=$(basename "$T" .txt)
  $PY evaluation/eval_counterfactual.py --baseline results_league/league_champion.zip --candidate results_league/league_champion.zip --baseline-only \
    --reg mc --team-weights data/team_weights_regmc.json --our-team "$T" --n-battles 150 --hidden-sheets --seed 83 --workers 8 --port 7610 \
    --output "results_brainv1/baseline_heuristic_$L.json" > "results_brainv1/baseline_heuristic_$L.log" 2>&1 || fail "baseline_$L"
done
$PY - <<'PY'
import json, glob
rows = []
for f in sorted(glob.glob("results_brainv1/baseline_heuristic_T*.json")):
    a = json.load(open(f))["arms"]["champion_policy"]; rows.append((f.split("_")[-1][:-5], a["win_rate"]))
mean = sum(w for _, w in rows) / len(rows)
print("BRAINV1_BASELINE " + " ".join(f"{t}={w:.3f}" for t, w in rows) + f" mean={mean:.3f} kill_line={mean-0.10:.3f}")
json.dump({"per_team": dict(rows), "mean": mean, "kill_line": mean - 0.10}, open("results_brainv1/baseline_heuristic.json", "w"), indent=1)
PY

# ---- pool + training --------------------------------------------------------------
CLONE=$(cat results_bc/mc_A_20260913/BEST.txt)
$PY training/make_brainv1_config.py --init results_league/league_champion.zip --clone "$CLONE" || fail config
rm -rf results_brainv1/saves_fp_hs_wt
$PY training/build_league.py --config training/brainv1_config.json > brainv1_build.log 2>&1 || { tail -5 brainv1_build.log; fail build; }
$PY training/build_league.py --config training/brainv1_config.json --verify-only || fail verify
restart_server 7700 || fail server_7700
LOG=brainv1_$(date +%H%M%S).log
echo "BRAINV1_TRAINING_START [$(stamp)] log=$LOG"
caffeinate -is ./training/run_brainv1_training.sh > "$LOG" 2>&1
echo "BRAINV1_TRAINING_EXITED [$(stamp)] exit=$? final=$([ -f results_brainv1/saves_fp_hs_wt/reg_mc/seed1/20643840.zip ] && echo yes || echo NO)"
$PY training/triage_league_log.py "$LOG" --save-dir results_brainv1/saves_fp_hs_wt/reg_mc/seed1 --resume 12779520 --results-dir results_brainv1 | tee brainv1_triage.txt
FINALISTS=$(grep -E "\.zip$" brainv1_triage.txt | tr '\n' ' ')
[ -n "$FINALISTS" ] || fail no_finalists
echo "BRAINV1_VERDICT_START [$(stamp)] $FINALISTS"
HUMAN_BC=$(cat results_bc/eval_mcB_20260913/BEST.txt) MIX_BC="$CLONE" caffeinate -is ./evaluation/run_brainv1_verdict_supervised.sh $FINALISTS > brainv1_verdict.log 2>&1 && echo "BRAINV1_VERDICT_OK" || echo "BRAINV1_VERDICT_FAILED"
grep -E "VERDICT\[|DIAG\[" brainv1_verdict.log
echo "BRAINV1_CHAIN_COMPLETE [$(stamp)]"
