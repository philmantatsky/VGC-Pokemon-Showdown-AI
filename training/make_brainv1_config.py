"""Write training/brainv1_config.json for the brain-v1 round (NEW_BRAIN_PLAN M3).

The pool continues the deployed lineage from ``--init`` (the round-6 finalist,
or the deployed brain when round 6 fails) with the Reg M-C human clone as the
human backbone and the self-lineage as history. Older checkpoints need no
conversion: every load path upgrades them to the current token length in
memory (policy.upgrade_policy). Our side is drawn from the candidate teams
(see run_brainv1_training.sh), so the opponent weights only zero our_team.txt.

Usage (from the repo root):
  .venv/bin/python training/make_brainv1_config.py --init <ckpt.zip> \
      --clone results_bc/mc_A/saves_bc/seed1/<epoch>.zip
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEPLOYED = "results_league/league_champion.zip"
OLD_CHAMPION = "results_repaired/champion.zip"
LEAGUE1_HISTORY = "results_league/saves_fp_hs_wt/reg_mb/seed1/11796480.zip"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", required=True, help="checkpoint the learner resumes from")
    ap.add_argument("--clone", required=True, help="Reg M-C human clone checkpoint")
    ap.add_argument("--resume-stem", type=int, default=12779520)
    ap.add_argument("--out", type=Path, default=Path("training/brainv1_config.json"))
    ap.add_argument("--weights-source", default="data/team_weights_regmc.json")
    ap.add_argument("--eval-only-clone", default="results_bc/eval_mcB")
    args = ap.parse_args()

    init = str(args.init)
    clone = str(args.clone)
    config = {
        "comment": (
            "Brain v1 round (NEW_BRAIN_PLAN M3): joint-action head + threat block + "
            "potential-based shaping, team-agnostic our side (teams/candidates_mc), "
            "Reg M-C data only. Pool: init x2 + resume, deployed, old champion, "
            "league-1 history, Reg M-C clone x4. No adversary. Older checkpoints are "
            "upgraded in memory at load."
        ),
        "dest": "results_brainv1/saves_fp_hs_wt/reg_mc/seed1",
        "resume_stem": args.resume_stem,
        "sources": {
            "100": clone,
            "200": clone,
            "300": clone,
            "400": clone,
            "500": OLD_CHAMPION,
            "600": LEAGUE1_HISTORY,
            "700": DEPLOYED,
            "800": init,
            "900": init,
            str(args.resume_stem): init,
        },
        "weights_source": args.weights_source,
        "weights_dest": "data/team_weights_regmc_brainv1.json",
        "tr_boost": 1.0,
        "eval_only_roots": [
            "results_bc/eval_B",
            "results_bc/eval_D",
            args.eval_only_clone,
        ],
    }
    args.out.write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {args.out}")
    for stem, source in config["sources"].items():
        print(f"  {stem:>9} <- {source}")


if __name__ == "__main__":
    main()
