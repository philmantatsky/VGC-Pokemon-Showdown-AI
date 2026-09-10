#!/usr/bin/env bash
set -euo pipefail

# Exhibition mode: keep the deployed bot online to accept challenges from
# anyone (the README invites portfolio visitors to fight it) whenever this
# machine is otherwise idle. It yields to real work automatically: between
# short 3-challenge sessions it checks for training / battery / ladder-batch
# processes and waits while any are running. A session already in progress
# finishes its games first -- and note a ladder batch started on the same
# account will kick the exhibition login, which is the intended priority.
#
# Run it (credentials are sourced locally, per house rules):
#   nohup ./exhibition_mode.sh > exhibition_mode.log 2>&1 &
# Stop with Ctrl-C / pkill -f exhibition_mode.sh. Exhibition games land in
# ladder_replays_exhibition_<reg>/, kept separate from the measured ladder
# corpus. Since 2026-09-10 the deployed configuration is Reg M-C with the
# three opt-in targeting guards (REG / GUARDS env override). poke-env only
# accepts a challenge sent in the bot's own format, so challengers must pick
# "[Gen 9 Champions] VGC 2026 Reg M-C".

cd "$(dirname "$0")"
set -a; source "../Laplace-Pokemon-Showdown-AI/.env"; set +a
REG=${REG:-mc}
GUARDS=${GUARDS:-resisted_target,overkill_split,dominated_weather_ball_weather}
HEAVY='vgc_bench[.]train|run_gate_battery|eval_counterfactual[.]py|run_counterfactual_pipeline|generate_counterfactuals|vgc_bench[.]pretrain|logs2trajs|run_team_tournament'

echo "exhibition mode: accepting challenges in reg $REG (Ctrl-C to stop)"
while true; do
  if pgrep -f "$HEAVY" >/dev/null 2>&1 \
     || pgrep -af "ladder_ourteam[.]py" 2>/dev/null | grep -v -- "--challenges" | grep -q .; then
    echo "$(date '+%H:%M') heavy job running; exhibition waiting..."
    sleep 120
    continue
  fi
  caffeinate -is .venv/bin/python -u ladder_ourteam.py \
    --checkpoint results_league/league_champion.zip \
    --reg "$REG" --our_team "teams/reg_$REG/our_team.txt" \
    --guards-extra "$GUARDS" \
    --challenges --n_games 3 \
    --replay_dir "ladder_replays_exhibition_$REG" || sleep 60
  sleep 5
done
