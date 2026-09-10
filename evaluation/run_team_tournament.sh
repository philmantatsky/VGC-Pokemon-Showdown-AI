#!/usr/bin/env bash
# Team tournament (NEW_BRAIN_PLAN §4): one PILOT checkpoint plays every candidate
# team in teams/candidates_mc/ against the weighted Reg M-C opponent pool, which
# the same pilot plays back (stochastic). Ranks teams by "how well this pilot does
# with the team in this meta". The clone tournament uses the Reg M-C human clone
# as the pilot (team-agnostic by construction); the brain tournament reuses this
# script with the new brain. Arms run under the stall watchdog and are skipped
# when their JSON exists.
# Usage: ./evaluation/run_team_tournament.sh <pilot.zip> <n_battles> <out_dir> [team files...]
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
PILOT=$1; N=$2; OUT=$3; shift 3
PORT=${PORT:-7600}
TEAMS=("$@"); [ ${#TEAMS[@]} -gt 0 ] || TEAMS=(teams/candidates_mc/T*.txt)
mkdir -p "$OUT"
for T in "${TEAMS[@]}"; do
  L=$(basename "$T" .txt)
  echo "== $L <- $T [$(date '+%H:%M:%S')]"
  ./evaluation/supervised_eval.sh "$OUT/$L.json" $PORT -- \
    --baseline "$PILOT" --candidate "$PILOT" --baseline-only \
    --reg mc --team-weights data/team_weights_regmc.json --our-team "$T" \
    --opponent-checkpoint "$PILOT" --opponent-stochastic \
    --n-battles "$N" --hidden-sheets --seed 83 --workers 8
done
.venv/bin/python - "$OUT" "$PILOT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); rows = []
for path in sorted(out.glob("T*.json")):
    d = json.load(open(path)); a = d["arms"]["champion_policy"]
    lo, hi = a.get("wilson_95", [float("nan"), float("nan")])
    shape = a.get("loss_shape", {})
    rows.append((path.stem, a["wins"], a["battles"], a["win_rate"], lo, hi, shape.get("first_faint_ours")))
rows.sort(key=lambda r: -r[3])
print(f"TOURNAMENT pilot={sys.argv[2]}")
for team, w, n, wr, lo, hi, ff in rows:
    print(f"TOURNAMENT {team}: {w}/{n} = {100*wr:5.1f}%  [{100*lo:4.1f}, {100*hi:4.1f}]  first_faint_ours={ff}")
json.dump({"pilot": sys.argv[2], "teams": [dict(team=r[0], wins=r[1], battles=r[2], win_rate=r[3], wilson_95=[r[4], r[5]]) for r in rows]}, open(out / "summary.json", "w"), indent=1)
print("TOURNAMENT_DONE")
PY
