"""Resolve results_deployed/DEPLOYED.json for the shell launchers, verified.

Prints shell assignments (shlex-quoted, for eval) of the deployed configuration:
CKPT, TEAM, GUARDS, REG, FORMAT, SET_PRIOR, PREVIEW_MODEL and REPLAY_TAG.

PREVIEW_MODEL is empty unless the deployment lets a learned, human-trained model
choose our team preview (fields learned_preview: true, preview_model,
preview_model_sha256; 2026-09-24, for a brain trained on human openings --
training/human_preview.py). REPLAY_TAG names the replay directories
(ladder_replays_mc_deployed_<tag>, challenge_replays_mc_deployed_<tag>): the
manifest's replay_tag, else the team file's stem, so a new configuration can get
fresh single-config directories without changing the current ones.

The checkpoint, team and preview model are checked against their manifest
sha256. Any mismatch or missing field exits 2 with the reason on stderr and
prints nothing, so no launcher can start an unverified configuration.

Usage (from the repo root):
  CFG=$(.venv/bin/python tools/deployed_config.py) || exit 2; eval "$CFG"
  (--prefix DEP_ prints DEP_CKPT=... so a launcher's own TEAM/REG/GUARDS env
  overrides survive the eval)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MANIFEST = Path("results_deployed/DEPLOYED.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(manifest: Path = MANIFEST, root: Path = ROOT) -> dict[str, str]:
    """The deployed configuration as shell variables; raises when unverified."""
    deployed = json.loads((root / manifest).read_text())["deployed"]
    learned = deployed.get("learned_preview", False)
    if not isinstance(learned, bool):
        raise ValueError("learned_preview must be true or false")
    checks = [("checkpoint", "sha256"), ("team", "team_sha256")]
    if learned:
        checks.append(("preview_model", "preview_model_sha256"))
    elif deployed.get("preview_model"):
        raise ValueError("preview_model is set but learned_preview is not true")
    from vgc_bench.src.guards import GUARDS

    unknown = [g for g in deployed["guards_extra"].split(",") if g and g not in GUARDS]
    if unknown:  # the bot would silently ignore a misspelled guard
        raise ValueError(f"unknown guards in guards_extra: {unknown}")
    for field, sha_field in checks:
        path = root / deployed[field]
        if not path.is_file():
            raise ValueError(f"deployed {field} is missing: {deployed[field]}")
        if sha256(path) != deployed[sha_field]:
            raise ValueError(f"deployed {field} does not match {sha_field}")
    return {
        "CKPT": deployed["checkpoint"],
        "TEAM": deployed["team"],
        "GUARDS": deployed["guards_extra"],
        "REG": deployed["reg"],
        "FORMAT": deployed["format"],
        "SET_PRIOR": deployed.get("set_prior_reg", "mc"),
        "PREVIEW_MODEL": deployed["preview_model"] if learned else "",
        "REPLAY_TAG": deployed.get("replay_tag") or Path(deployed["team"]).stem,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--root", type=Path, default=ROOT, help="repo root (tests)")
    ap.add_argument(
        "--prefix",
        default="",
        help="prefix every variable (DEP_ keeps a launcher's own env overrides)",
    )
    args = ap.parse_args()
    try:
        config = resolve(args.manifest, args.root)
    except (OSError, KeyError, ValueError) as exc:
        print(f"deployed configuration refused: {exc!r}", file=sys.stderr)
        raise SystemExit(2) from exc
    for key, value in config.items():
        print(f"{args.prefix}{key}={shlex.quote(value)}")


if __name__ == "__main__":
    main()
