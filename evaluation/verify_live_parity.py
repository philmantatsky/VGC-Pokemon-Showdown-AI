"""Stress the public live-state reconciler against exact Showdown source battles.

The shadow deliberately receives unrelated RNG every turn. It must recover from only
the source's poke-env-visible snapshot and then expose exactly the same legal joint
actions. This catches stale HP, active slots, forced switches, PP, statuses, fields,
and volatile state without cheating by copying the source's serialized state.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from vgc_bench.src.exact_observation import state_to_battle
from vgc_bench.src.exact_sim import ExactShowdownBridge
from vgc_bench.src.live_snapshot import public_snapshot

ROOT = Path(__file__).resolve().parent
FORMAT = "gen9championsvgc2026regmb"


def _hidden_charge_slots(snapshot: dict, side_index: int) -> dict[int, str]:
    """Return active slots whose charged target is private to that player.

    Showdown publicly announces that Electro Shot/Solar Beam is charging, but does
    not reveal the chosen target until the attack resolves. Different exact hidden
    particles may therefore retain different target locks while representing the
    same public state. That is uncertainty to search over, not a parity failure.
    """
    slots: dict[int, str] = {}
    for row in snapshot["sides"][side_index].get("pokemon", []):
        slot = row.get("active_slot")
        charge = row.get("effects", {}).get("twoturnmove", {})
        if slot is None or not charge.get("move") or "target_loc" in charge:
            continue
        slots[int(slot)] = str(charge["move"])
    return slots


def _canonical_private_targets(choice: str, hidden_slots: dict[int, str]) -> str:
    atoms = [atom.strip() for atom in choice.split(",")]
    for slot, move in hidden_slots.items():
        if slot >= len(atoms):
            continue
        tokens = atoms[slot].split()
        if len(tokens) < 2 or tokens[0] != "move" or tokens[1] != move:
            continue
        tokens = [
            "<private-target>"
            if index >= 2 and token.lstrip("+-").isdigit()
            else token
            for index, token in enumerate(tokens)
        ]
        atoms[slot] = " ".join(tokens)
    return ", ".join(atoms)


def _private_target_counter(choices: list[str], snapshot: dict) -> Counter[str]:
    hidden = _hidden_charge_slots(snapshot, 1)
    return Counter(_canonical_private_targets(choice, hidden) for choice in choices)


def _seed(rng: random.Random) -> list[int]:
    return [rng.randrange(1, 65536) for _ in range(4)]


def _rng_text(rng: random.Random) -> str:
    return ",".join(str(value) for value in _seed(rng))


def _preview(rng: random.Random) -> str:
    order = rng.sample(range(1, 7), 4)
    return "team " + ",".join(str(value) for value in order)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=1000)
    parser.add_argument("--max-games", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--output", type=Path, default=Path("results_parity/live_exact_parity.json")
    )
    args = parser.parse_args()
    opponents = sorted((ROOT / "teams/reg_mb").glob("*.txt"))
    our_team = (ROOT / "teams/reg_mb/our_team.txt").read_text()
    opponents = [path for path in opponents if path.name != "our_team.txt"]
    rng = random.Random(args.seed)
    checked = 0
    mismatches: list[dict] = []
    accepted_private_target_differences = 0
    started = time.monotonic()
    with ExactShowdownBridge() as bridge:
        for game in range(args.max_games):
            if checked >= args.states:
                break
            opponent = opponents[game % len(opponents)]
            battle_seed = _seed(rng)
            p1_preview = _preview(rng)
            p2_preview = _preview(rng)
            source = bridge.create(
                formatid=FORMAT,
                seed=battle_seed,
                p1_team_text=our_team,
                p2_team_text=opponent.read_text(),
                p1_preview=p1_preview,
                p2_preview=p2_preview,
            )
            shadow = bridge.create(
                formatid=FORMAT,
                seed=battle_seed,
                p1_team_text=our_team,
                p2_team_text=opponent.read_text(),
                p1_preview=p1_preview,
                p2_preview=p2_preview,
            )
            pending_p1_choice = None
            for _turn_step in range(40):
                if source["ended"] or checked >= args.states:
                    break
                try:
                    battle = state_to_battle(
                        source["state"], source["requests"], "p1", True
                    )
                    snapshot = public_snapshot(
                        battle,
                        source["requests"][0],
                        request_state=source["request_state"],
                        side_requests=source["requests"],
                        pending_our_choice=pending_p1_choice,
                    )
                    shadow = bridge.reconcile(shadow["state"], snapshot)
                    source_choice_lists = {
                        role: bridge.choices(source["state"], role)
                        for role in ("p1", "p2")
                    }
                    shadow_choice_lists = {
                        role: bridge.choices(shadow["state"], role)
                        for role in ("p1", "p2")
                    }
                    source_choices = {
                        role: set(choices)
                        for role, choices in source_choice_lists.items()
                    }
                    shadow_choices = {
                        role: set(choices)
                        for role, choices in shadow_choice_lists.items()
                    }
                    raw_p2_differs = source_choices["p2"] != shadow_choices["p2"]
                    choices_match = (
                        source_choices["p1"] == shadow_choices["p1"]
                        and _private_target_counter(
                            source_choice_lists["p2"], snapshot
                        )
                        == _private_target_counter(
                            shadow_choice_lists["p2"], snapshot
                        )
                    )
                    if not choices_match:
                        raise AssertionError(
                            {
                                role: {
                                    "missing": sorted(
                                        source_choices[role] - shadow_choices[role]
                                    )[:10],
                                    "extra": sorted(
                                        shadow_choices[role] - source_choices[role]
                                    )[:10],
                                }
                                for role in ("p1", "p2")
                            }
                        )
                    if raw_p2_differs:
                        accepted_private_target_differences += 1
                    checked += 1
                    p1_choice = rng.choice(sorted(source_choices["p1"]))
                    p2_choice = rng.choice(sorted(source_choices["p2"]))
                    if source["request_state"] == "move":
                        pending_p1_choice = p1_choice
                    source = bridge.simulate(
                        source["state"], p1_choice, p2_choice, _rng_text(rng)
                    )
                    shadow = bridge.simulate(
                        shadow["state"], p1_choice, p2_choice, _rng_text(rng)
                    )
                except Exception as error:
                    mismatches.append(
                        {
                            "game": game,
                            "opponent": opponent.name,
                            "checked_state": checked,
                            "turn": source.get("turn"),
                            "request_state": source.get("request_state"),
                            "error": f"{type(error).__name__}: {error}",
                            "source_requests": source.get("requests"),
                            "shadow_requests": shadow.get("requests"),
                            "snapshot": locals().get("snapshot"),
                            "public_events": getattr(
                                locals().get("battle"), "_replay_data", []
                            )[-80:],
                        }
                    )
                    break
    payload = {
        "schema": 1,
        "snapshot_schema": 2,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": args.seed,
        "required_states": args.states,
        "checked_states": checked,
        "mismatch_count": len(mismatches),
        "accepted_private_target_differences": (
            accepted_private_target_differences
        ),
        "mismatches": mismatches[:25],
        "elapsed_seconds": time.monotonic() - started,
        "accepted": checked >= args.states and not mismatches,
        "method": "different-RNG shadow reconciled only from poke-env-visible state",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"live parity: states={checked}/{args.states} mismatches={len(mismatches)} "
        f"elapsed={payload['elapsed_seconds']:.1f}s",
        flush=True,
    )
    if not payload["accepted"]:
        for mismatch in mismatches[:5]:
            print(
                {
                    key: mismatch[key]
                    for key in (
                        "game",
                        "opponent",
                        "checked_state",
                        "turn",
                        "request_state",
                        "error",
                    )
                },
                flush=True,
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
