#!/usr/bin/env bash
# The T6 set-variant clone tournament, pre-registered 2026-09-28 11:02 (commit
# a4d098c; reading and decision rule in PROJECT_STATUS): T6 / T6m / T6mAS, 1,000
# games each, on the full Reg M-C pool and the sand-only pool; then the
# pre-registered reading. Finished arms are skipped (their JSON exists), so a
# paused run resumes where it stopped (a half-played arm restarts). Local only.
# Usage (repo root, on AC power): ./evaluation/run_t6variants_20260928.sh
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
# keep the Mac awake for the whole run
if [ -z "${UNDER_CAFFEINATE:-}" ]; then UNDER_CAFFEINATE=1 exec caffeinate -is "$0" "$@"; fi
PILOT=results_bc/mc_A_20260920/saves_bc/seed1/2.zip
TEAMS=(teams/candidates_mc/T6.txt teams/candidates_mc/T6m.txt teams/candidates_mc/T6mAS.txt)
LOG=results_team_tournament/t6variants_20260928.log
mkdir -p results_team_tournament
{
  echo "START $(date)"
  PORT=7600 ./evaluation/run_team_tournament.sh "$PILOT" 1000 \
    results_team_tournament/clone_t6variants_20260928 "${TEAMS[@]}"
  PORT=7600 TEAM_WEIGHTS=data/team_weights_regmc_sand.json \
    ./evaluation/run_team_tournament.sh "$PILOT" 1000 \
    results_team_tournament/clone_t6variants_sand_20260928 "${TEAMS[@]}"
  if pids=$(lsof -nP -t -iTCP:7600 -sTCP:LISTEN 2>/dev/null); then kill $pids; fi
  .venv/bin/python results_analysis/t6variants_20260928/read_tournament.py
  echo "END $(date)"
} >> "$LOG" 2>&1
