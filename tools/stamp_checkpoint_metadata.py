"""Stamp checkpoint zips with sidecar metadata that launchers can hard-check.

Writes `<checkpoint>.metadata.json` next to each zip:

    {"sha256": ..., "requires_knowledge_obs": true, "obs_len": 7224,
     "stamped_at": ..., "role": "production"}

`ladder_ourteam.py` refuses to start when the knowledge_obs requirement cannot be
resolved from an explicit flag or this sidecar, and when the sidecar's sha256 no
longer matches the file. That closes the failure mode where the champion (which
was fine-tuned WITH the 24 knowledge features) silently laddered with them zeroed
because a store_true flag defaulted off.

Usage:
    python stamp_checkpoint_metadata.py <ckpt.zip> [--no-knowledge-obs] [--role NAME]
    python stamp_checkpoint_metadata.py --defaults

--defaults stamps the known production artifacts (champion, converted_v4, the
frozen opponents, and the outcome value net), all of which are v4-observation
checkpoints. Freshly converted-but-never-fine-tuned checkpoints are indifferent
to the flag (their appended weight columns are zero), so requiring it for them is
safe; the fine-tuned champion genuinely requires it.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("results_repaired/champion.zip", "production"),
    ("results_repaired/converted_v4.zip", "seed"),
    ("results_repaired/opponents/64opp_3932160_v4.zip", "eval_opponent"),
    ("results_repaired/opponents/8opp_4915200_v4.zip", "eval_opponent"),
    ("results_repaired/opponents/tuned_983040_v4.zip", "eval_opponent"),
    ("results_outcome_v1/outcome_value.zip", "outcome_value"),
)


def observation_length(ckpt: Path) -> int | None:
    """Best-effort observation length from an SB3 zip; None when unreadable."""
    try:
        from stable_baselines3.common.save_util import load_from_zip_file

        data, _, _ = load_from_zip_file(str(ckpt), load_data=True, device="cpu")
        space = data.get("observation_space")
        # PolicyPlayer uses a Dict space: {"observations": Box, "action_mask": ...}.
        subspaces = getattr(space, "spaces", None)
        if subspaces and "observation" in subspaces:
            space = subspaces["observation"]
        shape = getattr(space, "shape", None)
        if shape:
            return int(shape[-1])
    except Exception:
        return None
    return None


def stamp(ckpt: Path, requires_knowledge_obs: bool, role: str) -> Path:
    assert ckpt.exists(), f"checkpoint not found: {ckpt}"
    sidecar = Path(str(ckpt) + ".metadata.json")
    meta = {
        "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "requires_knowledge_obs": requires_knowledge_obs,
        "obs_len": observation_length(ckpt),
        "role": role,
        "stamped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    sidecar.write_text(json.dumps(meta, indent=1) + "\n")
    return sidecar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="*", help="zip files to stamp")
    ap.add_argument(
        "--no-knowledge-obs",
        dest="requires_knowledge_obs",
        action="store_false",
        help="mark the checkpoint as requiring the knowledge features OFF "
        "(pre-conversion checkpoints only)",
    )
    ap.set_defaults(requires_knowledge_obs=True)
    ap.add_argument("--role", default="production")
    ap.add_argument(
        "--defaults",
        action="store_true",
        help="stamp the known production artifacts instead of a positional list",
    )
    args = ap.parse_args()

    if args.defaults:
        targets = [(Path(p), role) for p, role in DEFAULT_ARTIFACTS]
    else:
        if not args.checkpoints:
            raise SystemExit("pass checkpoint paths or --defaults")
        targets = [(Path(p), args.role) for p in args.checkpoints]

    for ckpt, role in targets:
        if not ckpt.exists():
            print(f"SKIP (missing): {ckpt}")
            continue
        sidecar = stamp(ckpt, args.requires_knowledge_obs, role)
        meta = json.loads(sidecar.read_text())
        print(
            f"stamped {sidecar}  sha={meta['sha256'][:12]}...  "
            f"requires_knowledge_obs={meta['requires_knowledge_obs']}  "
            f"obs_len={meta['obs_len']}"
        )


if __name__ == "__main__":
    main()
