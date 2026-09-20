"""Write a brain-v1 league config (NEW_BRAIN_PLAN M3, and section 4 step 3).

The pool continues the deployed lineage from ``--init`` (the round-6 finalist,
the deployed brain, or a brain-v1 save for the specialist round) with the Reg
M-C human clone as the human backbone and the self-lineage as history. Older
checkpoints need no conversion: every load path upgrades them to the current
token length in memory (policy.upgrade_policy). Our side is chosen by the
launcher (run_brainv1_training.sh, OUR_TEAMS), so the opponent weights only
zero our_team.txt.

Usage (from the repo root):
  .venv/bin/python training/make_brainv1_config.py --init <ckpt.zip> \
      --clone results_bc/mc_A/saves_bc/seed1/<epoch>.zip
  # specialist round: separate artifacts, another brain-v1 save in place of
  # one deployed copy (the deployed brain is a weak pilot of the pool's teams)
  .venv/bin/python training/make_brainv1_config.py --init <save7.zip> \
      --clone <mc_A.zip> --resume-stem 19660800 \
      --dest results_brainv1_spec/saves_fp_hs_wt/reg_mc/seed1 \
      --weights-dest data/team_weights_regmc_brainv1_spec.json \
      --swap-deployed <save8.zip> --out training/brainv1_spec_config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEPLOYED = "results_league/league_champion.zip"
OLD_CHAMPION = "results_repaired/champion.zip"
LEAGUE1_HISTORY = "results_league/saves_fp_hs_wt/reg_mb/seed1/11796480.zip"
DEFAULT_DEST = "results_brainv1/saves_fp_hs_wt/reg_mc/seed1"
DEFAULT_WEIGHTS_DEST = "data/team_weights_regmc_brainv1.json"
DEFAULT_EVAL_ONLY_CLONES = ["results_bc/eval_mcB", "results_bc/eval_mcB_20260913"]


def build_config(
    init: str,
    clone: str,
    resume_stem: int = 12779520,
    weights_source: str = "data/team_weights_regmc.json",
    dest: str = DEFAULT_DEST,
    weights_dest: str = DEFAULT_WEIGHTS_DEST,
    swap_deployed: str | None = None,
    eval_only_clones: list[str] | None = None,
) -> dict:
    """The league description build_league.py consumes."""
    return {
        "comment": (
            "Brain v1 league (NEW_BRAIN_PLAN M3 / section 4): joint-action head + "
            "threat block + potential-based shaping, Reg M-C data only; our side "
            "set by the launcher. Pool: init x2 + resume, deployed x3 (x2 when "
            "--swap-deployed puts another save at stem 700), old champion, "
            "league-1 history, Reg M-C clone x2 (deployed-heavy: round 6's "
            "lesson). No adversary. Older checkpoints are upgraded in memory."
        ),
        "dest": dest,
        "resume_stem": resume_stem,
        "sources": {
            # Round 6 (2026-09-13) showed a clone-heavy pool is too soft after a
            # format change (first save -10pp paired); the deployed brain at 6/10
            # copies held (first save 0.89). Same shape here.
            "100": clone,
            "200": clone,
            "300": OLD_CHAMPION,
            "400": LEAGUE1_HISTORY,
            "500": DEPLOYED,
            "600": DEPLOYED,
            "700": swap_deployed or DEPLOYED,
            "800": init,
            "900": init,
            str(resume_stem): init,
        },
        "weights_source": weights_source,
        "weights_dest": weights_dest,
        "tr_boost": 1.0,
        "eval_only_roots": [
            "results_bc/eval_B",
            "results_bc/eval_D",
            *(eval_only_clones or DEFAULT_EVAL_ONLY_CLONES),
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", required=True, help="checkpoint the learner resumes from")
    ap.add_argument("--clone", required=True, help="Reg M-C human clone checkpoint")
    ap.add_argument("--resume-stem", type=int, default=12779520)
    ap.add_argument("--out", type=Path, default=Path("training/brainv1_config.json"))
    ap.add_argument("--weights-source", default="data/team_weights_regmc.json")
    ap.add_argument("--dest", default=DEFAULT_DEST, help="pool/save directory")
    ap.add_argument("--weights-dest", default=DEFAULT_WEIGHTS_DEST)
    ap.add_argument(
        "--swap-deployed",
        default=None,
        help="checkpoint that replaces the third deployed copy (stem 700)",
    )
    ap.add_argument(
        "--eval-only-clone",
        action="append",
        default=None,
        help="eval-only clone roots banned from the pool (repeatable)",
    )
    args = ap.parse_args()

    config = build_config(
        init=str(args.init),
        clone=str(args.clone),
        resume_stem=args.resume_stem,
        weights_source=args.weights_source,
        dest=args.dest,
        weights_dest=args.weights_dest,
        swap_deployed=args.swap_deployed,
        eval_only_clones=args.eval_only_clone,
    )
    args.out.write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {args.out}")
    for stem, source in config["sources"].items():
        print(f"  {stem:>9} <- {source}")


if __name__ == "__main__":
    main()
