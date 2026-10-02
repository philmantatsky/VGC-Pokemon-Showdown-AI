#!/usr/bin/env bash
# A serial ladder TRIAL on top of the DEPLOYED configuration: the same brain, team,
# preview model, guards, sticky corrections and set data as
# results_deployed/DEPLOYED.json (verified by tools/deployed_config.py), plus the
# trial's own additions, into the trial's own replay dir. DEPLOYED.json is not
# changed: a trial is not a promotion.
# Usage: TRIAL_PLAYBOOK=<playbook.json> TRIAL_GUARDS=<guard,...> \
#        TRIAL_CHECKPOINT=<brain.zip> TRIAL_TEAM=<team.txt> \
#          tools/ladder_trial.sh <n_games_total> <replay_dir>
# - TRIAL_PLAYBOOK: our own plan cards at team preview (--playbook), replacing the
#   deployed playbook if there is one.
# - TRIAL_GUARDS: opt-in guards added to the deployed ones (e.g. playbook_opening,
#   the playbook's turn-1 script).
# - TRIAL_CHECKPOINT: a candidate brain instead of the deployed one (the user's
#   word; its sidecar must carry sha256 + requires_knowledge_obs, which
#   ladder_ourteam.py checks). Everything else stays the deployed configuration.
# - TRIAL_TEAM: a set variant of the deployed team, for a candidate brain practised
#   on it (needs TRIAL_CHECKPOINT; the same six species, so the deployed preview
#   model still applies).
# Ladder play needs the user's explicit word; tools/ladder_read_loop.sh refuses
# while a heavy local job or another ladder session runs.
set -uo pipefail
cd "$(dirname "$0")/.."
N=${1:?total games}
DIR=${2:?replay dir (never the deployed one)}
CFG=$(.venv/bin/python tools/deployed_config.py) || { echo "LADDER_REFUSED the deployed configuration failed verification"; exit 2; }
eval "$CFG"
[ "$REG" = mc ] || { echo "LADDER_REFUSED the deployed configuration is not Reg M-C"; exit 2; }
case "$DIR" in
  ladder_replays_mc_deployed_"$REPLAY_TAG"|ladder_replays_mc_deployed_"$REPLAY_TAG"/)
    echo "LADDER_REFUSED a trial needs its own replay dir"; exit 2 ;;
esac
if [ -n "${TRIAL_CHECKPOINT:-}" ]; then
  [ -f "$TRIAL_CHECKPOINT" ] || { echo "LADDER_REFUSED missing checkpoint $TRIAL_CHECKPOINT"; exit 2; }
  CKPT=$TRIAL_CHECKPOINT
fi
if [ -n "${TRIAL_TEAM:-}" ]; then
  [ -n "${TRIAL_CHECKPOINT:-}" ] || { echo "LADDER_REFUSED TRIAL_TEAM needs TRIAL_CHECKPOINT (a brain practised on it)"; exit 2; }
  [ -f "$TRIAL_TEAM" ] || { echo "LADDER_REFUSED missing team $TRIAL_TEAM"; exit 2; }
  .venv/bin/python - "$TEAM" "$TRIAL_TEAM" <<'PY' || { echo "LADDER_REFUSED $TRIAL_TEAM is not a set variant of the deployed team"; exit 2; }
import sys
from pathlib import Path
sys.path.insert(0, ".")
from vgc_bench.src.set_particles import team_roster
a, b = ({p.species for p in team_roster(Path(f).read_text())} for f in sys.argv[1:3])
sys.exit(0 if a == b else 1)
PY
  TEAM=$TRIAL_TEAM
fi
if [ -n "${TRIAL_PLAYBOOK:-}" ]; then
  [ -f "$TRIAL_PLAYBOOK" ] || { echo "LADDER_REFUSED missing playbook $TRIAL_PLAYBOOK"; exit 2; }
  PLAYBOOK=$TRIAL_PLAYBOOK
fi
if [ -n "${TRIAL_GUARDS:-}" ]; then
  GUARDS="$GUARDS,$TRIAL_GUARDS"
fi
export VGC_SET_PRIOR_REG="$SET_PRIOR"
export PREVIEW_MODEL MIXING STICKY PLAYBOOK
GUARDS="$GUARDS" exec ./tools/ladder_read_loop.sh "$CKPT" "$TEAM" "$N" "$DIR"
