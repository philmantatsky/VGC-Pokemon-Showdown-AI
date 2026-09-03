#!/usr/bin/env bash
set -euo pipefail

# Search re-gate on the promoted brain with the v3h evaluator: paired n=300
# hidden-sheet battles, no-search champion vs the same champion with selective
# exact search (the Aug-22 production configuration: 8s turn budget, 2s screen,
# up to 8 determinizations, selective, no pondering). Search arms force serial
# play, so expect ~6-8 hours.
#
# Pre-registered bar (written before any result): the search arm must BEAT the
# no-search arm by >= 3pp with `move_search.decisions` > 0 and latency p90
# under 9s; a tie is not a pass (house rule since Aug 22). The user's standing
# order: the 25-game ladder test runs ONLY if this gate passes.

VALUE=${1:-results_outcome_v3h/outcome_value.zip}
CHAMP=results_league/league_champion.zip
mkdir -p results_search_v3h

.venv/bin/python evaluation/eval_counterfactual.py \
  --baseline $CHAMP --candidate $CHAMP \
  --include-live-search --search-comparison-only \
  --outcome-value "$VALUE" \
  --search-budget 8 --screen-budget 2 --chance-samples 1 \
  --determinizations 8 --selective-search \
  --n-battles 300 --hidden-sheets --seed 303 --port 7600 --workers 8 \
  --output results_search_v3h/search_regate_hidden300.json
echo "SEARCH_REGATE_COMPLETE"
