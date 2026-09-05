#!/usr/bin/env bash
set -euo pipefail
trap 'echo "GATE_FAILED at line $LINENO (exit $?)"' ERR

# Search re-gate on the promoted brain: paired n=300 hidden-sheet battles vs
# the HELD-OUT human-imitation opponent (bc_eval_B, stochastic; baseline ~84%
# leaves headroom, unlike the heuristic where the champion sits at ~91%),
# no-search champion vs the same champion with selective
# exact search (the Aug-22 production configuration: 8s turn budget, 2s screen,
# up to 8 determinizations, selective, no pondering). Search arms force serial
# play, so expect ~6-8 hours.
#
# Pre-registered bar (written before any result): the search arm must BEAT the
# no-search arm by >= 3pp with `move_search.decisions` > 0 and latency p90
# under 9s; a tie is not a pass (house rule since Aug 22). The user's standing
# order: the 25-game ladder test runs ONLY if this gate passes.

VALUE=${1:-results_outcome_v3h/outcome_value.zip}
PORT=7600

# Pre-flight: a fresh eval server. A server still holding a killed run's
# half-finished rooms hands the next run stale mid-battle messages under the
# same player names (sheet-handshake errors, embed crashes, silent stalls --
# observed repeatedly on 2026-09-05). The eval port is dedicated, so restart it.
if pids=$(lsof -nP -t -iTCP:$PORT -sTCP:LISTEN 2>/dev/null); then
  kill $pids 2>/dev/null || true
  sleep 3
fi
(cd pokemon-showdown && node pokemon-showdown start $PORT --no-security > /dev/null 2>&1 &)
for _ in $(seq 1 30); do lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 && break; sleep 1; done
lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1 || { echo "GATE_FAILED: eval server on $PORT did not start"; exit 2; }
CHAMP=results_league/league_champion.zip
mkdir -p results_search_v3h

.venv/bin/python evaluation/eval_counterfactual.py \
  --baseline $CHAMP --candidate $CHAMP \
  --opponent-checkpoint results_bc/eval_B/saves_bc/seed2/30.zip --opponent-stochastic \
  --include-live-search --search-comparison-only \
  --outcome-value "$VALUE" \
  --search-budget 8 --screen-budget 2 --chance-samples 1 \
  --determinizations 8 --selective-search \
  --n-battles 300 --hidden-sheets --seed 303 --port $PORT --workers 8 \
  --output results_search_v3h/search_regate_hidden300.json
echo "SEARCH_REGATE_COMPLETE"
