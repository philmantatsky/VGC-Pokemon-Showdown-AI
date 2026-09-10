#!/usr/bin/env bash
# After the M-C ladder read: tally it, build the Reg M-C data layer (scrape,
# team pool, priors, human clones), smoke the M-C battery wiring, then launch
# league round 6 (first M-C round) and its supervised verdict. Every stage
# prints a marker; a failed stage stops the chain.
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
PY=.venv/bin/python
DAY=20260910
DIR=ladder_replays_mc_guards3_20260909
MERGED=battle_logs_top_mc_merged
stamp() { date '+%H:%M:%S'; }
fail() { echo "MC_CHAIN_FAILED[$1] [$(stamp)]"; exit 3; }

echo "[$(stamp)] waiting for LADDER_DONE"
until grep -q LADDER_DONE ladder_read_mc_loop.log 2>/dev/null; do sleep 60; done
while pgrep -f "ladder_ourteam.py" >/dev/null 2>&1; do sleep 10; done

# ---- M0 tally --------------------------------------------------------------
echo "MC_TALLY_START [$(stamp)]"
grep -h "^record:" ladder_read_mc_session*.log | sed 's/^/LADDER_RECORD /'
$PY tools/ladder_loss_profile.py "$DIR" --json results_analysis/loss_profile_mc_read_$DAY.json > results_analysis/loss_profile_mc_read_$DAY.txt 2>&1 && head -60 results_analysis/loss_profile_mc_read_$DAY.txt
$PY tools/hit_effectiveness.py "$DIR" 2>&1 | tail -12
$PY - "$DIR/decisions.jsonl" <<'PY'
import json, sys, collections
fired = collections.Counter(); changed = collections.Counter(); n = 0
for raw in open(sys.argv[1]):
    try: r = json.loads(raw)
    except Exception: continue
    n += 1
    g = r.get("guards") or {}
    for name in (g.get("stages") or []): fired[name] += 1
    for name in (g.get("demotions") or {}): changed[name] += 1
    if (r.get("chosen") or {}).get("demoted_by"): changed["chosen_demoted_by:" + str(r["chosen"]["demoted_by"])] += 1
print(f"GUARDS decisions={n} stages={dict(fired)} demotions={dict(changed)}")
PY
echo "MC_TALLY_DONE [$(stamp)]"

# ---- M1 data layer ---------------------------------------------------------
echo "MC_SCRAPE_START [$(stamp)]"
$PY datagen/scrape_top_players.py --formats gen9championsvgc2026regmc,gen9championsvgc2026regmcbo3 --out-dir battle_logs_top_mc_$DAY > scrape_mc_$DAY.log 2>&1 || fail scrape
tail -4 scrape_mc_$DAY.log
$PY datagen/merge_battle_logs.py --out $MERGED battle_logs_top_mc_20260909 battle_logs_top_mc_$DAY || fail merge
LOGS="$MERGED/logs_gen9championsvgc2026regmc.json $MERGED/logs_gen9championsvgc2026regmcbo3.json"
echo "MC_TEAMS_START [$(stamp)]"
rm -f teams/reg_mc/MC*.txt
$PY datagen/extract_ots_teams.py --logs $LOGS --format gen9championsvgc2026regmc --out teams/reg_mc --prefix MC > extract_ots_mc_$DAY.log 2>&1 || fail extract_teams
tail -3 extract_ots_mc_$DAY.log
$PY datagen/build_team_weights.py --logs $LOGS --teams teams/reg_mc --output data/team_weights_regmc.json > team_weights_mc_$DAY.log 2>&1 || fail team_weights
echo "MC_TEAMS_DONE [$(stamp)] teams=$(ls teams/reg_mc/MC*.txt | wc -l | tr -d ' ')"
echo "MC_PRIORS_START [$(stamp)]"
$PY training/train_preview_model.py --logs $LOGS --output data/opponent_preview_top500_regmc.pt > prior_preview_mc.log 2>&1 || fail prior_preview
$PY training/train_move_model.py --logs $LOGS --output data/opponent_move_top500_regmc.pt > prior_move_mc.log 2>&1 || fail prior_move
$PY training/train_switch_model.py --logs $LOGS --output data/opponent_switch_top500_regmc.pt > prior_switch_mc.log 2>&1 || fail prior_switch
for f in prior_preview_mc.log prior_move_mc.log prior_switch_mc.log; do echo "PRIOR $f: $(grep -E 'examples=|valid|acc|top' $f | tail -2 | tr '\n' ' ')"; done
echo "MC_PRIORS_DONE [$(stamp)]"
echo "MC_TRAJS_START [$(stamp)]"
$PY -m vgc_bench.logs2trajs --logs $LOGS --out_dir trajs_regmc_human_A --buckets 0,1,2,3,4 --num_workers 4 > logs2trajs_mc_A.log 2>&1 || fail trajs_A
$PY -m vgc_bench.logs2trajs --logs $LOGS --out_dir trajs_regmc_human_B --buckets 5,6,7,8,9 --num_workers 4 > logs2trajs_mc_B.log 2>&1 || fail trajs_B
echo "MC_TRAJS_DONE [$(stamp)] A=$(ls trajs_regmc_human_A | wc -l | tr -d ' ') B=$(ls trajs_regmc_human_B | wc -l | tr -d ' ')"
grep -E "wrote|prefiltered|read failures" logs2trajs_mc_A.log | tail -3

agree() { $PY evaluation/eval_bc_agreement.py --policy "$1" --trajs_dir "$2" --sample 800 2>&1 | grep -E "top-1 per-slot" | grep -oE "[0-9]+\.[0-9]+" | head -1; }
echo "MC_CLONES_START [$(stamp)]"
caffeinate -is $PY -m vgc_bench.pretrain --run_id 1 --device mps --num_epochs 30 --div_frac 1.0 \
  --init_from results_bc/foundation_converted.zip --trajs_dir trajs_regmc_human_A \
  --output_dir results_bc/mc_A --eval_every 0 --port 7611 > pretrain_mc_A.log 2>&1 || { tail -5 pretrain_mc_A.log; fail pretrain_mc_A; }
BEST_A=""; BEST_A_SCORE=0
for e in 2 4 6 8 10 15 20 30; do
  [ -f results_bc/mc_A/saves_bc/seed1/$e.zip ] || continue
  s=$(agree results_bc/mc_A/saves_bc/seed1/$e.zip trajs_regmc_human_B); echo "AGREE mc_A epoch $e on B: top-1 $s%"
  if [ -n "$s" ] && [ "$(echo "$s > $BEST_A_SCORE" | bc)" = 1 ]; then BEST_A_SCORE=$s; BEST_A=results_bc/mc_A/saves_bc/seed1/$e.zip; fi
done
[ -n "$BEST_A" ] || fail mc_A_select
$PY tools/stamp_checkpoint_metadata.py "$BEST_A" --role training_opponent || fail stamp_A
caffeinate -is $PY -m vgc_bench.pretrain --run_id 2 --device mps --num_epochs 10 --div_frac 1.0 \
  --init_from results_bc/foundation_converted.zip --trajs_dir trajs_regmc_human_B \
  --output_dir results_bc/eval_mcB --eval_every 0 --port 7611 > pretrain_eval_mcB.log 2>&1 || { tail -5 pretrain_eval_mcB.log; fail pretrain_eval_mcB; }
BEST_B=""; BEST_B_SCORE=0
for e in 2 3 4 5 6 8 10; do
  [ -f results_bc/eval_mcB/saves_bc/seed2/$e.zip ] || continue
  s=$(agree results_bc/eval_mcB/saves_bc/seed2/$e.zip trajs_regmc_human_A); echo "AGREE eval_mcB epoch $e on A: top-1 $s%"
  if [ -n "$s" ] && [ "$(echo "$s > $BEST_B_SCORE" | bc)" = 1 ]; then BEST_B_SCORE=$s; BEST_B=results_bc/eval_mcB/saves_bc/seed2/$e.zip; fi
done
[ -n "$BEST_B" ] || fail eval_mcB_select
$PY tools/stamp_checkpoint_metadata.py "$BEST_B" --role eval_only || fail stamp_B
echo "MC_CLONES_DONE [$(stamp)] mc_A=$BEST_A ($BEST_A_SCORE%) eval_mcB=$BEST_B ($BEST_B_SCORE%)"
echo "$BEST_A" > results_bc/mc_A/BEST.txt; echo "$BEST_B" > results_bc/eval_mcB/BEST.txt

# ---- M1 smoke: M-C battery wiring (deployed vs itself, 40 battles/arm) ------
echo "MC_SMOKE_START [$(stamp)]"
mkdir -p results_gate_battery_mc_smoke
$PY evaluation/eval_counterfactual.py --baseline results_league/league_champion.zip --candidate results_league/league_champion.zip \
  --baseline-only --reg mc --team-weights data/team_weights_regmc.json --our-team teams/reg_mc/our_team.txt \
  --opponent-checkpoint "$BEST_B" --opponent-stochastic --n-battles 40 --hidden-sheets --seed 83 --port 7600 --workers 8 \
  --output results_gate_battery_mc_smoke/deployed_vs_evalmcB_40.json > results_gate_battery_mc_smoke/smoke.log 2>&1 || { tail -8 results_gate_battery_mc_smoke/smoke.log; fail smoke; }
$PY - <<'PY'
import json; d=json.load(open("results_gate_battery_mc_smoke/deployed_vs_evalmcB_40.json"))
a=d["arms"]["champion_policy"]; f=d.get("resolved_flags",{})
print(f"MC_SMOKE deployed_vs_evalmcB={a['win_rate']:.3f} ({a['wins']}/{a['battles']}) preview={f.get('preview_model')} move={f.get('move_model')} switch={f.get('switch_model')}")
PY
echo "MC_SMOKE_DONE [$(stamp)]"

# ---- M2 league round 6 ------------------------------------------------------
cat > training/league6_config.json <<EOF
{
  "comment": "Round 6 ($DAY): the first Reg M-C round. Single variable vs the league-1 recipe: the DATA. Opponent pool = Reg M-C open-team-sheet teams with replay-derived weights; human backbone = the Reg M-C clone mc_A x4 (36%); self-lineage = old champion, league-1 history x2, deployed x2 + resume. No adversary. Our side stays MB430 until the team tournament. eval_B, eval_D and eval_mcB banned by content.",
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
  "eval_only_roots": ["results_bc/eval_B", "results_bc/eval_D", "results_bc/eval_mcB"]
}
EOF
echo "MC_LEAGUE6_BUILD [$(stamp)]"
$PY training/build_league.py --config training/league6_config.json > league6_build.log 2>&1 || { tail -8 league6_build.log; fail league6_build; }
$PY training/build_league.py --config training/league6_config.json --verify-only || fail league6_verify
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
