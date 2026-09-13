#!/usr/bin/env bash
# Resume of the Reg M-C chain (paused 2026-09-10 01:20 at the battery smoke).
# Refreshes the data layer with today's scrape (the M-C corpus doubles every
# day or two right now), rebuilds priors/trajectories/clones into DATED
# artifacts (the 09-10 clones stay untouched -- the clone tournament's pilot
# keeps its provenance), runs the battery smoke UNDER THE STALL WATCHDOG, then
# builds and launches league round 6 and its supervised verdict.
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
PY=.venv/bin/python
DAY=20260913
MERGED=battle_logs_top_mc_merged
stamp() { date '+%H:%M:%S'; }
fail() { echo "MC_CHAIN_FAILED[$1] [$(stamp)]"; exit 3; }
ensure_server() {  # $1 = port
  lsof -nP -iTCP:$1 -sTCP:LISTEN >/dev/null 2>&1 && return 0
  (cd pokemon-showdown && node pokemon-showdown start $1 --no-security > /dev/null 2>&1 &)
  for _ in $(seq 1 30); do lsof -nP -iTCP:$1 -sTCP:LISTEN >/dev/null 2>&1 && return 0; sleep 1; done
  return 1
}

echo "MC_RESUME_START [$(stamp)]"
# ---- M1 refresh -------------------------------------------------------------
echo "MC_SCRAPE_START [$(stamp)]"
$PY datagen/scrape_top_players.py --formats gen9championsvgc2026regmc,gen9championsvgc2026regmcbo3 --out-dir battle_logs_top_mc_$DAY > scrape_mc_$DAY.log 2>&1 || fail scrape
tail -4 scrape_mc_$DAY.log
$PY datagen/merge_battle_logs.py --out $MERGED battle_logs_top_mc_20260909 battle_logs_top_mc_20260910 battle_logs_top_mc_$DAY || fail merge
LOGS="$MERGED/logs_gen9championsvgc2026regmc.json $MERGED/logs_gen9championsvgc2026regmcbo3.json"
echo "MC_TEAMS_START [$(stamp)]"
rm -f teams/reg_mc/MC*.txt
$PY datagen/extract_ots_teams.py --logs $LOGS --format gen9championsvgc2026regmc --out teams/reg_mc --prefix MC > extract_ots_mc_$DAY.log 2>&1 || fail extract_teams
tail -2 extract_ots_mc_$DAY.log
$PY datagen/build_team_weights.py --logs $LOGS --teams teams/reg_mc --output data/team_weights_regmc.json > team_weights_mc_$DAY.log 2>&1 || fail team_weights
echo "MC_TEAMS_DONE [$(stamp)] teams=$(ls teams/reg_mc/MC*.txt | wc -l | tr -d ' ')"
echo "MC_PRIORS_START [$(stamp)]"
$PY training/train_preview_model.py --logs $LOGS --output data/opponent_preview_top500_regmc.pt > prior_preview_mc.log 2>&1 || fail prior_preview
$PY training/train_move_model.py --logs $LOGS --output data/opponent_move_top500_regmc.pt > prior_move_mc.log 2>&1 || fail prior_move
$PY training/train_switch_model.py --logs $LOGS --output data/opponent_switch_top500_regmc.pt > prior_switch_mc.log 2>&1 || fail prior_switch
for f in prior_preview_mc.log prior_move_mc.log prior_switch_mc.log; do echo "PRIOR $f: $(grep -E 'examples=|saved' $f | tail -1 | cut -c1-200)"; done
echo "MC_PRIORS_DONE [$(stamp)]"
echo "MC_TRAJS_START [$(stamp)]"
$PY -m vgc_bench.logs2trajs --logs $LOGS --out_dir trajs_regmc_human_A_$DAY --buckets 0,1,2,3,4 --num_workers 4 > logs2trajs_mc_A_$DAY.log 2>&1 || fail trajs_A
$PY -m vgc_bench.logs2trajs --logs $LOGS --out_dir trajs_regmc_human_B_$DAY --buckets 5,6,7,8,9 --num_workers 4 > logs2trajs_mc_B_$DAY.log 2>&1 || fail trajs_B
echo "MC_TRAJS_DONE [$(stamp)] A=$(ls trajs_regmc_human_A_$DAY | wc -l | tr -d ' ') B=$(ls trajs_regmc_human_B_$DAY | wc -l | tr -d ' ')"

agree() { $PY evaluation/eval_bc_agreement.py --policy "$1" --trajs_dir "$2" --sample 800 2>&1 | grep -E "top-1 per-slot" | grep -oE "[0-9]+\.[0-9]+" | head -1; }
echo "MC_CLONES_START [$(stamp)]"
A_DIR=results_bc/mc_A_$DAY; B_DIR=results_bc/eval_mcB_$DAY
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
echo "MC_CLONES_DONE [$(stamp)] mc_A=$BEST_A ($BEST_A_SCORE%) eval_mcB=$BEST_B ($BEST_B_SCORE%)"

# ---- smoke under the watchdog ------------------------------------------------
echo "MC_SMOKE_START [$(stamp)]"
mkdir -p results_gate_battery_mc_smoke
STALL_MIN=5 RETRIES=2 ./evaluation/supervised_eval.sh results_gate_battery_mc_smoke/deployed_vs_evalmcB_$DAY.json 7600 -- \
  --baseline results_league/league_champion.zip --candidate results_league/league_champion.zip --baseline-only \
  --reg mc --team-weights data/team_weights_regmc.json --our-team teams/reg_mc/our_team.txt \
  --opponent-checkpoint "$BEST_B" --opponent-stochastic --n-battles 40 --hidden-sheets --seed 83 --workers 8 || fail smoke
[ -f results_gate_battery_mc_smoke/deployed_vs_evalmcB_$DAY.json ] || fail smoke_missing
$PY - "$DAY" <<'PY'
import json, sys
d=json.load(open(f"results_gate_battery_mc_smoke/deployed_vs_evalmcB_{sys.argv[1]}.json"))
a=d["arms"]["champion_policy"]; f=a.get("resolved_flags",{})
print(f"MC_SMOKE deployed_vs_evalmcB={a['win_rate']:.3f} ({a['wins']}/{a['battles']}) knowledge={f.get('use_knowledge_obs')} preview={a.get('preview_model')}")
PY
echo "MC_SMOKE_DONE [$(stamp)]"

# ---- round 6 ----------------------------------------------------------------
cat > training/league6_config.json <<EOF
{
  "comment": "Round 6 ($DAY): the first Reg M-C round. Single variable vs the league-1 recipe: the DATA. Opponent pool = Reg M-C open-team-sheet teams with replay-derived weights; human backbone = the Reg M-C clone mc_A_$DAY x4 (36%); self-lineage = old champion, league-1 history x2, deployed x2 + resume. No adversary. Our side stays MB430 until the team tournament. eval_B, eval_D, eval_mcB and eval_mcB_$DAY banned by content.",
  "dest": "results_league6/saves_fp_hs_wt/reg_mc/seed1",
  "resume_stem": 12779520,
  "sources": {
    "100": "$BEST_A",
    "200": "$BEST_A",
    "300": "$BEST_A",
    "400": "$BEST_A",
    "500": "results_repaired/champion.zip",
    "600": "results_league/saves_fp_hs_wt/reg_mb/seed1/8847360.zip",
    "700": "results_league/saves_fp_hs_wt/reg_mb/seed1/11796480.zip",
    "800": "results_league/league_champion.zip",
    "900": "results_league/league_champion.zip",
    "12779520": "results_league/league_champion.zip"
  },
  "weights_source": "data/team_weights_regmc.json",
  "weights_dest": "data/team_weights_regmc_league6.json",
  "tr_boost": 1.0,
  "eval_only_roots": ["results_bc/eval_B", "results_bc/eval_D", "results_bc/eval_mcB", "results_bc/eval_mcB_$DAY"]
}
EOF
echo "MC_LEAGUE6_BUILD [$(stamp)]"
rm -rf results_league6
$PY training/build_league.py --config training/league6_config.json > league6_build.log 2>&1 || { tail -8 league6_build.log; fail league6_build; }
$PY training/build_league.py --config training/league6_config.json --verify-only || fail league6_verify
ensure_server 7700 || fail server_7700
LOG=league6_$(date +%H%M%S).log
echo "ROUND6_TRAINING_START [$(stamp)] log=$LOG"
caffeinate -is ./training/run_league6_training.sh > "$LOG" 2>&1
echo "ROUND6_TRAINING_EXITED [$(stamp)] exit=$? final=$([ -f results_league6/saves_fp_hs_wt/reg_mc/seed1/17694720.zip ] && echo yes || echo NO)"
$PY training/triage_league_log.py "$LOG" --save-dir results_league6/saves_fp_hs_wt/reg_mc/seed1 --resume 12779520 --results-dir results_league6 | tee round6_triage.txt
FINALISTS=$(grep -E "\.zip$" round6_triage.txt | tr '\n' ' ')
[ -n "$FINALISTS" ] || fail round6_no_finalists
echo "ROUND6_VERDICT_START [$(stamp)] $FINALISTS"
HUMAN_BC="$BEST_B" MIX_BC="$BEST_A" caffeinate -is ./evaluation/run_league6_verdict_supervised.sh $FINALISTS > league6_verdict.log 2>&1 && echo "ROUND6_VERDICT_OK" || echo "ROUND6_VERDICT_FAILED"
grep -E "VERDICT\[|DIAG\[" league6_verdict.log
echo "MC_CHAIN_COMPLETE [$(stamp)]"
