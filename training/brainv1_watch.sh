#!/usr/bin/env bash
# Brain-v1 checkpoint watcher: triage line per save; at save 1 the pre-registered
# rule -- kill when eval/heuristic is more than 10pp below the deployed brain's
# own read on the candidate teams (results_brainv1/baseline_heuristic.json,
# "kill_line") AND a paired 200-battle read on MB430 vs the deployed brain is
# worse than -5pp. Later saves are triaged only.
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
CHAIN=brainv1_chain.log
SAVES=results_brainv1/saves_fp_hs_wt/reg_mc/seed1
echo "[$(date '+%H:%M:%S')] waiting for BRAINV1_TRAINING_START in $CHAIN"
until grep -q "BRAINV1_TRAINING_START\|BRAINV1_FAILED" $CHAIN 2>/dev/null; do sleep 120; done
grep -q BRAINV1_FAILED $CHAIN && { echo "V1WATCH_SKIPPED chain failed"; exit 2; }
LOG=$(grep BRAINV1_TRAINING_START $CHAIN | tail -1 | sed 's/.*log=//')
KILL=$(.venv/bin/python -c "import json; print(round(json.load(open('results_brainv1/baseline_heuristic.json'))['kill_line'], 3))" 2>/dev/null || echo 0.75)
echo "V1WATCH_ARMED [$(date '+%H:%M:%S')] log=$LOG kill_line=$KILL"
seen=""
for i in $(seq 1 600); do
  sleep 120
  if ! pgrep -f "vgc_bench[.]train" >/dev/null 2>&1; then
    if grep -q "BRAINV1_TRAINING_EXITED" $CHAIN; then echo "V1WATCH_TRAINING_EXITED"; break; fi
  fi
  for f in $(ls $SAVES/*.zip 2>/dev/null | sort -t/ -k5 -n); do
    stem=$(basename "$f" .zip)
    [ "$stem" -gt 12779520 ] 2>/dev/null || continue
    case " $seen " in *" $stem "*) continue;; esac
    seen="$seen $stem"
    echo "V1SAVE $stem [$(date '+%H:%M:%S')]"
    triage=$(.venv/bin/python training/triage_league_log.py "$LOG" --save-dir "$SAVES" --resume 12779520 --results-dir results_brainv1 2>&1 | grep -E "^save")
    echo "$triage" | sed 's/^/V1TRIAGE /'
    if [ "$stem" = "13762560" ]; then
      h=$(echo "$triage" | grep -E "^save 1 " | grep -oE "heuristic [0-9.]+" | grep -oE "[0-9.]+$")
      if [ -n "$h" ] && [ "$(echo "$h < $KILL" | bc)" = 1 ]; then
        echo "V1FIRSTSAVE heuristic=$h below kill line $KILL; running the paired 200-battle read on MB430"
        .venv/bin/python evaluation/eval_counterfactual.py --baseline results_league/league_champion.zip --candidate "$f" \
          --reg mc --team-weights data/team_weights_regmc.json --our-team teams/reg_mc/our_team.txt \
          --n-battles 200 --hidden-sheets --seed 83 --workers 8 --port 7610 \
          --output results_brainv1/first_save_vs_deployed_200.json > results_brainv1/first_save_vs_deployed_200.log 2>&1
        d=$(.venv/bin/python -c "
import json; d=json.load(open('results_brainv1/first_save_vs_deployed_200.json'))['arms']; print(round(100*(d['distilled_policy']['win_rate']-d['champion_policy']['win_rate']),1))" 2>/dev/null)
        echo "V1FIRSTSAVE paired delta vs deployed on MB430: ${d}pp"
        if [ -n "$d" ] && [ "$(echo "$d < -5" | bc)" = 1 ]; then
          echo "V1_KILLED [$(date '+%H:%M:%S')] first save heuristic=$h paired=${d}pp"
          pkill -f "brainv1_chain.sh"; sleep 1; pkill -f "vgc_bench[.]train"; sleep 5; pkill -9 -f "vgc_bench[.]train" 2>/dev/null
          echo "BRAINV1_KILLED_BY_FIRST_SAVE_RULE [$(date '+%H:%M:%S')] heuristic=$h paired=${d}pp" >> $CHAIN
          exit 0
        fi
      else
        echo "V1FIRSTSAVE heuristic=$h passes kill line $KILL; continuing"
      fi
    fi
  done
  grep -qiE "Traceback" "$LOG" 2>/dev/null && { grep -iE "Traceback|Error" "$LOG" | tail -2 | sed 's/^/V1LOGERR /'; }
  # shaped-return invariant: a potential-based shaped episode return telescopes
  # to the terminal +-1; a mean outside [-1.05, 1.05] is a reward bug, never
  # learning (the 2026-09-15 attempt farmed +6.9). Checked every tick.
  rew=$(.venv/bin/python - <<'PY' 2>/dev/null
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
fs = sorted(Path("results_brainv1/logs_fp_hs_wt/reg_mc").rglob("events.out.tfevents.*"))
if fs:
    ea = EventAccumulator(str(fs[-1]), size_guidance={"scalars": 0}); ea.Reload()
    if "rollout/ep_rew_mean" in ea.Tags()["scalars"]:
        ev = ea.Scalars("rollout/ep_rew_mean")
        if len(ev) >= 5: print(f"{ev[-1].value:.3f} {len(ev)}")
PY
)
  if [ -n "$rew" ]; then
    val=${rew%% *}; cnt=${rew##* }
    if [ $((i % 15)) -eq 0 ]; then echo "V1REWARD rollouts=$cnt ep_rew_mean=$val [$(date '+%H:%M:%S')]"; fi
    if [ "$(echo "$val > 1.05 || $val < -1.05" | bc)" = 1 ]; then
      echo "V1_KILLED_REWARD_INVARIANT [$(date '+%H:%M:%S')] ep_rew_mean=$val after $cnt rollouts"
      pkill -f "brainv1_chain.sh"; sleep 1; pkill -f "vgc_bench[.]train"; sleep 5; pkill -9 -f "vgc_bench[.]train" 2>/dev/null
      echo "BRAINV1_KILLED_REWARD_INVARIANT [$(date '+%H:%M:%S')] ep_rew_mean=$val" >> $CHAIN
      exit 0
    fi
  fi
done
echo "V1WATCH_END [$(date '+%H:%M:%S')]"
