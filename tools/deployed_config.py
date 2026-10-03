"""Resolve results_deployed/DEPLOYED.json for the shell launchers, verified.

Prints shell assignments (shlex-quoted, for eval) of the deployed configuration:
CKPT, TEAM, GUARDS, REG, FORMAT, SET_PRIOR, PREVIEW_MODEL, REPLAY_TAG, MIXING,
STICKY, PLAYBOOK and SHEET_PREVIEW.

SHEET_PREVIEW is "--sheet-preview" when the deployment has "sheet_preview": true
(2026-10-03: with the opponent's open team sheet the learned preview chooses among
its top plans by the sheet, and re-plans if the sheet arrives late), else empty.

PLAYBOOK is empty unless the deployment has our own plan cards (fields playbook,
playbook_sha256; 2026-09-27, vgc_bench/src/playbook.py): the launchers then pass
--playbook, and the file is sha-checked like the others.

STICKY is "--sticky-corrections" when the deployment has
"sticky_guard_corrections": true (2026-09-26: the reranker may not put back a
pair a guard corrected away), else empty; a non-boolean value refuses.

MIXING is empty unless the deployment has a "mixing" field (2026-09-26, the
user's near-tie "wheel"): {"mode": "always"|"opening", "top_k", "temperature",
"min_ratio", "keep_corrections"}; it prints the matching ladder_ourteam.py
flags. Invalid values refuse the whole configuration.

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


def mixing_args(deployed: dict) -> str:
    """ladder_ourteam.py flags for the deployment's mixed-strategy play."""
    mixing = deployed.get("mixing")
    if not mixing:
        return ""
    mode = mixing.get("mode", "off")
    if mode not in ("off", "opening", "always"):
        raise ValueError(f"unknown mixing mode {mode!r}")
    if mode == "off":
        return ""
    top_k = mixing.get("top_k", 3)
    temperature = float(mixing.get("temperature", 1.0))
    ratio = float(mixing.get("min_ratio", 0.0))
    keep = mixing.get("keep_corrections", False)
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("mixing top_k must be a positive integer")
    if not temperature > 0 or not 0 <= ratio < 1 or not isinstance(keep, bool):
        raise ValueError("mixing temperature/min_ratio/keep_corrections invalid")
    flags = [
        "--mixing",
        mode,
        "--mixing-top-k",
        str(top_k),
        "--mixing-temperature",
        f"{temperature:g}",
        "--mixing-min-ratio",
        f"{ratio:g}",
    ]
    if keep:
        flags.append("--mixing-keep-corrections")
    return " ".join(flags)


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
    if deployed.get("playbook"):
        checks.append(("playbook", "playbook_sha256"))
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
    sticky = deployed.get("sticky_guard_corrections", False)
    if not isinstance(sticky, bool):
        raise ValueError("sticky_guard_corrections must be true or false")
    sheet = deployed.get("sheet_preview", False)
    if not isinstance(sheet, bool):
        raise ValueError("sheet_preview must be true or false")
    return {
        "CKPT": deployed["checkpoint"],
        "TEAM": deployed["team"],
        "GUARDS": deployed["guards_extra"],
        "REG": deployed["reg"],
        "FORMAT": deployed["format"],
        "SET_PRIOR": deployed.get("set_prior_reg", "mc"),
        "PREVIEW_MODEL": deployed["preview_model"] if learned else "",
        "REPLAY_TAG": deployed.get("replay_tag") or Path(deployed["team"]).stem,
        "MIXING": mixing_args(deployed),
        "STICKY": "--sticky-corrections" if sticky else "",
        "PLAYBOOK": deployed.get("playbook") or "",
        "SHEET_PREVIEW": "--sheet-preview" if sheet else "",
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
