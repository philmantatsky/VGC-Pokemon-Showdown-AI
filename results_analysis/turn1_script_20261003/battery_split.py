"""Descriptive split of the turn-1 script battery (not part of the gate; PROJECT_STATUS
2026-10-03 10:40). Both arms take our four and leads from the same preview model, so
cells (population x roster x sheets) where we did NOT lead Blastoise + Farigiraf play
identically configured bots -- a built-in A/A check -- and the script can only act in
the Blastoise + Farigiraf cells. Reports the with-minus-without delta (mean over cells,
95% cell bootstrap) for: Blastoise + Farigiraf cells, other cells, cells where the
script changed turn 1 / never did / 3+ times; plus "we lost a Pokemon first" per arm.
Run from the repo root:
    .venv/bin/python results_analysis/turn1_script_20261003/battery_split.py \
        [results_guard_ab_turn1_script]
"""

from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WATER = {"blastoise", "farigiraf"}


def arm_files(run: Path) -> tuple[str, Path, str]:
    manifest = json.loads((run / "manifest.json").read_text())
    ref = manifest["reference_arm"]  # "<dir>/<population>_<label>.jsonl"
    ref_dir, ref_pattern = ref.split("/", 1)
    return manifest["arm_label"], ROOT / ref_dir, ref_pattern.split("_", 1)[1]


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def cells(run: Path) -> list[dict]:
    label, ref_dir, ref_suffix = arm_files(run)
    out = []
    for file in sorted(run.glob(f"*_{label}.jsonl")):
        if file.name.endswith(".telemetry.jsonl"):
            continue
        pop = file.name.removesuffix(f"_{label}.jsonl")
        cand = read(file)
        ref = read(ref_dir / f"{pop}_{ref_suffix}")
        fired = collections.Counter()
        for cell in read(file.with_suffix(".telemetry.jsonl")):
            key = (cell["opponent"], bool(cell["hidden_sheets"]))
            fired[key] += int(
                (cell.get("guard_counts") or {}).get("playbook_opening", 0)
            )
        by = collections.defaultdict(lambda: {"ref": [], "cand": []})
        for side, rows in (("ref", ref), ("cand", cand)):
            for r in rows:
                by[(r["opponent"], bool(r["hidden_sheets"]))][side].append(r)
        for key, sides in by.items():
            if not sides["ref"] or not sides["cand"]:
                continue
            leads = {frozenset(r.get("lead_species") or ()) for r in sides["cand"]}
            out.append(
                {
                    "pop": pop,
                    "cell": key,
                    "water": all(lead == WATER for lead in leads),
                    "fired": fired[key],
                    "ref": sum(r["target"] for r in sides["ref"]) / len(sides["ref"]),
                    "cand": sum(r["target"] for r in sides["cand"])
                    / len(sides["cand"]),
                    "ref_first": sum(
                        r.get("first_faint") == "ours" for r in sides["ref"]
                    )
                    / len(sides["ref"]),
                    "cand_first": sum(
                        r.get("first_faint") == "ours" for r in sides["cand"]
                    )
                    / len(sides["cand"]),
                }
            )
    return out


def summary(group: list[dict], seed: int = 20923) -> str:
    if not group:
        return "no cells"
    deltas = [c["cand"] - c["ref"] for c in group]
    rng = random.Random(seed)
    draws = sorted(
        sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(4000)
    )
    mean = sum(deltas) / len(deltas)
    first = sum(c["cand_first"] - c["ref_first"] for c in group) / len(group)
    return (
        f"{len(group):3d} cells  delta {100 * mean:+.2f}pp "
        f"[{100 * draws[100]:+.2f}, {100 * draws[3899]:+.2f}]  "
        f"lost a Pokemon first {100 * first:+.1f}pp"
    )


def main() -> None:
    run = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "results_guard_ab_turn1_script")
    data = cells(run)
    print(f"{run.name}: {len(data)} cells")
    groups = {
        "led Blastoise + Farigiraf": [c for c in data if c["water"]],
        "other leads (A/A check)": [c for c in data if not c["water"]],
        "script changed turn 1": [c for c in data if c["fired"] > 0],
        "script never changed it": [c for c in data if c["fired"] == 0],
        "changed it 3+ times": [c for c in data if c["fired"] >= 3],
    }
    for name, group in groups.items():
        print(f"  {name:28s} {summary(group)}")
    print("\nper population, Blastoise + Farigiraf cells:")
    for pop in sorted({c["pop"] for c in data}):
        group = [c for c in data if c["pop"] == pop and c["water"]]
        print(f"  {pop:16s} {summary(group)}")


if __name__ == "__main__":
    main()
