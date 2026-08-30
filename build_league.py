"""Build (and verify) the league fine-tune opponent pool.

The fictitious-play trainer treats every file in its save dir as an opponent,
and every filename there must be a bare integer stem -- so checkpoints cannot
carry their .metadata.json sidecars into the pool. This builder is the only
sanctioned way to seed a league: it verifies each source against its sidecar
(sha256 + role), refuses eval-only holdouts by role AND by content, copies the
checkpoints under their league stems, and writes a league_manifest.json at the
league root that verify_league_dir() (called by vgc_bench.train on every
launch) re-checks before any training step runs.

League shape (defaults): the deployed champion resumes at its own step count,
its four lineage checkpoints keep the historical FP pool, and bc_mix_A -- the
human-imitation policy -- enters at three low stems for a 3/8 initial share
that decays as new self-saves join the pool. Stem 100 doubles as the slot
callback.py auto-evaluates against each save interval (eval/bc*).

Usage:
    .venv/bin/python build_league.py                # build the pool
    .venv/bin/python build_league.py --verify-only  # re-check an existing pool
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

from vgc_bench.src.utils import (
    _sha256_file,
    refuse_eval_only_checkpoint,
    verify_league_dir,
)

RESUME_STEM = 7864320
DEFAULT_DEST = Path("results_league/saves_fp_hs_wt/reg_mb/seed1")
DEFAULT_EVAL_ONLY_ROOT = Path("results_bc/eval_B")
DEFAULT_WEIGHTS_SOURCE = Path("data/team_weights_regmb.json")
DEFAULT_WEIGHTS_DEST = Path("data/team_weights_regmb_league.json")
ALLOWED_ROLES = {"training_opponent", "production", "seed"}

DEFAULT_SOURCES: dict[int, str] = {
    100: "results_bc/mix_A/saves_bc/seed1/30.zip",
    200: "results_bc/mix_A/saves_bc/seed1/30.zip",
    300: "results_bc/mix_A/saves_bc/seed1/30.zip",
    3932160: "results_repaired/saves_fp_hs_wt/reg_mb/seed1/3932160.zip",
    4915200: "results_repaired/saves_fp_hs_wt/reg_mb/seed1/4915200.zip",
    5898240: "results_repaired/saves_fp_hs_wt/reg_mb/seed1/5898240.zip",
    6881280: "results_repaired/saves_fp_hs_wt/reg_mb/seed1/6881280.zip",
    RESUME_STEM: "results_repaired/champion.zip",
}


def check_source(path: Path) -> tuple[str, str]:
    """Verify one source checkpoint; return (sha256, role).

    The sidecar, when present, must match the file's current content (a stale
    sidecar means the checkpoint changed after stamping) and carry an allowed
    role. Sidecar-less sources (the raw lineage checkpoints) are admitted with
    role "lineage" -- their shas still get pinned in the manifest.
    """
    if not path.exists():
        raise SystemExit(f"league source missing: {path}")
    refuse_eval_only_checkpoint(path)
    digest = _sha256_file(path)
    sidecar = Path(str(path) + ".metadata.json")
    if not sidecar.exists():
        return digest, "lineage"
    metadata = json.loads(sidecar.read_text())
    stamped = metadata.get("sha256")
    if stamped != digest:
        raise SystemExit(
            f"{path} does not match its sidecar sha256 (stamped {stamped}, "
            f"actual {digest}); re-stamp or investigate before building a league."
        )
    role = metadata.get("role", "")
    if role not in ALLOWED_ROLES:
        raise SystemExit(
            f"{path} has role={role!r}; only {sorted(ALLOWED_ROLES)} may enter "
            "a training pool (eval arms must stay unfit populations)."
        )
    return digest, role


def collect_banned_shas(eval_only_root: Path) -> dict[str, str]:
    """Hash every checkpoint under the eval-only tree into a content ban list.

    Banning only the stamped epoch would leave 30 sibling epochs admissible;
    the holdout property belongs to the whole training run, so every .zip
    under the tree is banned by content.
    """
    banned: dict[str, str] = {}
    for path in sorted(eval_only_root.rglob("*.zip")):
        banned[_sha256_file(path)] = f"{path} (eval-only holdout)"
    if not banned:
        raise SystemExit(
            f"no checkpoints found under {eval_only_root}; refusing to build a "
            "league without a content ban list for the holdout arm."
        )
    return banned


def _team_species(team_file: Path) -> list[str]:
    """Species ids from a Showdown export team file (team-agnostic)."""
    from poke_env.data import to_id_str

    species = []
    for block in team_file.read_text().strip().split("\n\n"):
        first = block.strip().split("\n")[0]
        name = first.split("@")[0].strip()
        if "(" in name and ")" in name:
            name = name[name.rindex("(") + 1 : name.rindex(")")]
        species.append(to_id_str(name))
    return species


def write_league_team_weights(
    source: Path,
    dest: Path,
    tr_boost: float = 1.0,
    teams_dir: Path = Path("teams/reg_mb"),
) -> int:
    """Write the league copy of the team weights; returns boosted-team count.

    our_team.txt is byte-identical to MB430.txt and both sit in the sampled
    pool, silently double-weighting the mirror matchup during training. The
    weight must be an explicit 0.0 -- deleting the key would hand the file the
    sampler's default weight of 1.0.

    ``tr_boost`` > 1 multiplies the weight of every team whose roster reads
    P(TrickRoom) >= 0.6 from the joint sets -- a curriculum knob for a league
    round that targets a measured TR weakness. Property-derived, no species
    hardcoding.
    """
    weights = json.loads(source.read_text())
    if "our_team.txt" not in weights:
        raise SystemExit(f"{source} has no our_team.txt entry; wrong weights file?")
    weights["our_team.txt"] = 0.0
    boosted = 0
    if tr_boost != 1.0:
        from vgc_bench.src.preview_rules import trick_room_probability

        for name, value in weights.items():
            if not value:
                continue
            team_file = teams_dir / name
            if not team_file.exists():
                continue
            if trick_room_probability(_team_species(team_file)) >= 0.6:
                weights[name] = value * tr_boost
                boosted += 1
    dest.write_text(json.dumps(weights, indent=1, sort_keys=True) + "\n")
    return boosted


def build_league(
    dest: Path,
    sources: dict[int, str],
    resume_stem: int,
    eval_only_root: Path,
    weights_source: Path,
    weights_dest: Path,
    tr_boost: float = 1.0,
) -> Path:
    """Seed the league pool and write its manifest; return the manifest path."""
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(
            f"{dest} already contains files; a league is seeded exactly once. "
            "Use --verify-only to re-check it, or remove the directory to rebuild."
        )
    if max(sources) != resume_stem:
        raise SystemExit(
            f"resume stem {resume_stem} must be the highest seeded stem "
            f"(got max {max(sources)}); training resumes from the max stem."
        )
    banned = collect_banned_shas(eval_only_root)
    entries: dict[str, dict[str, str]] = {}
    staged: list[tuple[int, Path, str, str]] = []
    for stem, source in sorted(sources.items()):
        source_path = Path(source)
        digest, role = check_source(source_path)
        if digest in banned:
            raise SystemExit(
                f"{source_path} matches a banned checkpoint by content "
                f"({banned[digest]}); it cannot enter a training pool."
            )
        staged.append((stem, source_path, digest, role))
    dest.mkdir(parents=True, exist_ok=True)
    for stem, source_path, digest, role in staged:
        shutil.copy2(source_path, dest / f"{stem}.zip")
        entries[str(stem)] = {
            "sha256": digest,
            "source": str(source_path),
            "role": role,
        }
    boosted = write_league_team_weights(weights_source, weights_dest, tr_boost)
    manifest_path = _league_root(dest) / "league_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_at": datetime.datetime.now(datetime.timezone.utc)
                .isoformat(timespec="seconds"),
                "resume_stem": resume_stem,
                "banned_sha256": banned,
                "entries": entries,
                "team_weights": str(weights_dest),
                "tr_boost": tr_boost,
                "tr_boosted_teams": boosted,
            },
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    return manifest_path


def _league_root(dest: Path) -> Path:
    """The results_* ancestor of the save dir (where the manifest lives)."""
    for candidate in [dest, *dest.parents]:
        if candidate.name.startswith("results_"):
            return candidate
    return dest.parent


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build (and verify) the league fine-tune opponent pool."
    )
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "JSON league description: {dest, resume_stem, sources: {stem: path}, "
            "weights_source, weights_dest, tr_boost}. Omitted keys fall back to "
            "the round-1 defaults."
        ),
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="re-run every league check on an existing pool without copying",
    )
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text()) if args.config else {}
    dest = args.dest or Path(config.get("dest", DEFAULT_DEST))
    sources = {int(k): v for k, v in config.get("sources", DEFAULT_SOURCES).items()}
    resume_stem = int(config.get("resume_stem", RESUME_STEM))
    weights_dest = Path(config.get("weights_dest", DEFAULT_WEIGHTS_DEST))
    if args.verify_only:
        if not dest.exists() or not any(dest.iterdir()):
            raise SystemExit(f"nothing to verify: {dest} is missing or empty")
        verify_league_dir(dest)
        if not weights_dest.exists():
            raise SystemExit(f"missing {weights_dest}; rebuild the league")
        league_weights = json.loads(weights_dest.read_text())
        if league_weights.get("our_team.txt") != 0.0:
            raise SystemExit(
                f"{weights_dest} does not zero our_team.txt; rebuild it"
            )
        print(f"league pool verified: {dest}", flush=True)
        return
    manifest_path = build_league(
        dest=dest,
        sources=sources,
        resume_stem=resume_stem,
        eval_only_root=Path(config.get("eval_only_root", DEFAULT_EVAL_ONLY_ROOT)),
        weights_source=Path(config.get("weights_source", DEFAULT_WEIGHTS_SOURCE)),
        weights_dest=weights_dest,
        tr_boost=float(config.get("tr_boost", 1.0)),
    )
    verify_league_dir(dest)
    print(f"league built: {dest}\nmanifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
