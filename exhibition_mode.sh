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
# Run it yourself (credentials are sourced locally, per house rules):
#   ./exhibition_mode.sh
# Stop with Ctrl-C. Exhibition games land in ladder_replays_exhibition/,
# kept separate from the measured ladder corpus.

cd "$(dirname "$0")"
set -a; source "../Laplace-Pokemon-Showdown-AI/.env"; set +a

echo "exhibition mode: accepting challenges (Ctrl-C to stop)"
while true; do
  if pgrep -f "vgc_bench.train|run_gate_battery|eval_counterfactual|run_counterfactual_pipeline|generate_counterfactuals" >/dev/null 2>&1 \
     || pgrep -af "ladder_ourteam.py" 2>/dev/null | grep -v -- "--challenges" | grep -q .; then
    echo "$(date '+%H:%M') heavy job running; exhibition waiting..."
    sleep 120
    continue
  fi
  caffeinate -is .venv/bin/python ladder_ourteam.py \
    --checkpoint results_league/league_champion.zip \
    --challenges --n_games 3 \
    --replay_dir ladder_replays_exhibition || sleep 60
  sleep 5
done
