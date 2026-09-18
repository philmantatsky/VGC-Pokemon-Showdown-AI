#!/usr/bin/env bash
# Team grid (NEW_BRAIN_PLAN section 4, M4): one CANDIDATE and the deployed
# brain pilot every candidate team in a PAIRED read against (a) the scripted
# opponent and (b) the eval-only Reg M-C human clone, so the per-team deltas
# are cross-pilot comparable. (The team tournament is not: its opponent side is
# the pilot itself, so a pilot that is weak with unfamiliar teams inflates its
# own numbers.) Arms run under the stall watchdog; completed arms are skipped.
# Usage: HUMAN_BC=<eval_mcB ckpt> ./evaluation/run_team_grid.sh <candidate.zip> <out_dir> [n_battles] [team files...]
set -uo pipefail
cd "/Users/phillipmantatsky/Desktop/pokemon showdown bot/vgc-bench"
CAND=$1; OUT=$2; N=${3:-300}; shift $(( $# < 3 ? $# : 3 ))
PORT=${PORT:-7600}; BASE=${BASE:-results_league/league_champion.zip}
HUMAN_BC=${HUMAN_BC:?eval-only Reg M-C clone checkpoint}
TEAMS=("$@"); [ ${#TEAMS[@]} -gt 0 ] || TEAMS=(teams/candidates_mc/T*.txt)
mkdir -p "$OUT"
COMMON=(--baseline "$BASE" --candidate "$CAND" --reg mc --team-weights data/team_weights_regmc.json
        --n-battles "$N" --hidden-sheets --seed 83 --workers 8)
for T in "${TEAMS[@]}"; do
  L=$(basename "$T" .txt)
  echo "== $L heuristic [$(date '+%H:%M:%S')]"
  ./evaluation/supervised_eval.sh "$OUT/${L}_heuristic.json" $PORT -- "${COMMON[@]}" --our-team "$T"
  echo "== $L human [$(date '+%H:%M:%S')]"
  ./evaluation/supervised_eval.sh "$OUT/${L}_human.json" $PORT -- "${COMMON[@]}" --our-team "$T" --opponent-checkpoint "$HUMAN_BC" --opponent-stochastic
done
.venv/bin/python - "$OUT" "$CAND" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); rows = []
for arm in ("heuristic", "human"):
    for path in sorted(out.glob(f"T*_{arm}.json")):
        d = json.load(open(path)); a = d["arms"]; c, b = a["distilled_policy"], a["champion_policy"]
        rows.append(dict(team=path.stem.split("_")[0], arm=arm, candidate=c["win_rate"], deployed=b["win_rate"],
                         n=c["battles"], delta_pp=100 * (c["win_rate"] - b["win_rate"]),
                         first_faint_ours=[c.get("loss_shape", {}).get("first_faint_ours"), b.get("loss_shape", {}).get("first_faint_ours")]))
print(f"GRID candidate={sys.argv[2]} deployed=results_league/league_champion.zip (paired, per team)")
for arm in ("heuristic", "human"):
    sub = [r for r in rows if r["arm"] == arm]
    for r in sub:
        print(f"GRID {r['team']} {arm:<9}: candidate {100*r['candidate']:5.1f}%  deployed {100*r['deployed']:5.1f}%  delta {r['delta_pp']:+5.1f}pp  (n={r['n']})")
    if sub:
        print(f"GRID mean {arm:<9}: candidate {100*sum(r['candidate'] for r in sub)/len(sub):5.1f}%  deployed {100*sum(r['deployed'] for r in sub)/len(sub):5.1f}%  delta {sum(r['delta_pp'] for r in sub)/len(sub):+5.1f}pp")
json.dump({"candidate": sys.argv[2], "rows": rows}, open(out / "summary.json", "w"), indent=1)
print("GRID_DONE")
PY
