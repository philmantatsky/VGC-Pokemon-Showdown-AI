"""Controlled opening trials: same turn policy, matchup and opponent preview.

These are local development labels, NOT a promotion test. Server battle RNG is
independent between arms; repeated games estimate outcomes, not exact counterfactuals.
Evaluation-only opponents are refused when producing training labels.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import asyncio
import hashlib
import json
import os
import random
import time
from collections import Counter
from uuid import uuid4

import torch
from poke_env.data import to_id_str
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from evaluation.eval_counterfactual import (
    DelayedShapeAwareBatchPolicyPlayer,
    DelayedSimpleHeuristicsPlayer,
    PreviewLedger,
    _apply_preview_choice,
    _first_faint_side,
    _preview_fingerprint,
)
from vgc_bench.src import pokeenv_patches
from vgc_bench.src.guards import GUARDS, HARD_GUARDS
from vgc_bench.src.policy_player import BatchPolicyPlayer, PolicyPlayer
from vgc_bench.src.set_particles import team_roster
from vgc_bench.src.set_priors import DATA, prior_reg
from vgc_bench.src.teams import RandomTeamBuilder
from vgc_bench.src.utils import chunk_obs_len, format_map, refuse_eval_only_checkpoint


def resolve_plan(roster, names):
    index = {to_id_str(s): i + 1 for i, s in enumerate(roster)}
    order = tuple(index[to_id_str(s)] for s in names)
    if len(order) != 4 or len(set(order)) != 4:
        raise ValueError("an opening needs four distinct team members")
    return "/team " + "".join(map(str, order))


def fresh_local_account():
    # Reusing guest names after interruption rejoins abandoned local battles,
    # including requests arriving before set_policy has finished loading.
    return AccountConfiguration("T6" + uuid4().hex[:12], None)


class StudyPlayer(BatchPolicyPlayer):
    forced_plan = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preview_records = {}

    async def _teampreview(self, battle):
        if self.forced_plan:
            choice = resolve_plan(
                [p.base_species for p in battle.team.values()], self.forced_plan
            )
            _apply_preview_choice(self, battle, choice)
        else:
            choice = await super()._teampreview(battle)
        order = [int(x) - 1 for x in choice.removeprefix("/team ")]
        ours = [p.base_species for p in battle.team.values()]
        self.preview_records[battle.battle_tag] = {
            "choice": choice,
            "lead": order[:2],
            "bring": sorted(order),
            "lead_species": [ours[i] for i in order[:2]],
            "bring_species": [ours[i] for i in order],
            "opponent_roster": [p.base_species for p in battle.opponent_team.values()],
        }
        return choice


class StudyOpponent(DelayedShapeAwareBatchPolicyPlayer):
    async def teampreview(self, battle):
        if not hasattr(self, "preview_fingerprints"):
            self.preview_fingerprints = {}
        self.preview_fingerprints[battle.battle_tag] = _preview_fingerprint(battle)
        choice = await super().teampreview(battle)
        if not hasattr(self, "preview_records"):
            self.preview_records = {}
        self.preview_records[battle.battle_tag] = choice
        return choice


class StudyHeuristic(DelayedSimpleHeuristicsPlayer):
    async def teampreview(self, battle):
        if not hasattr(self, "preview_fingerprints"):
            self.preview_fingerprints = {}
        self.preview_fingerprints[battle.battle_tag] = _preview_fingerprint(battle)
        choice = await super().teampreview(battle)
        if not hasattr(self, "preview_records"):
            self.preview_records = {}
        self.preview_records[battle.battle_tag] = choice
        return choice


def category(text):
    t = text.lower()
    if "- trick room" in t:
        return "trick_room"
    if "ability: drizzle" in t:
        return "rain"
    if "ability: chlorophyll" in t and (
        "charizardite y" in t or "ability: drought" in t
    ):
        return "sun"
    if "ability: psychic surge" in t:
        return "psychic_terrain"
    if "rillaboom" in t and "- fake out" in t:
        return "grassy_fakeout"
    if "- tailwind" in t:
        return "tailwind"
    return "balance"


def roster_split(text):
    roster = tuple(sorted(p.species for p in team_roster(text)))
    bucket = int(hashlib.sha256("|".join(roster).encode()).hexdigest()[:8], 16) % 5
    return roster, "heldout" if bucket == 0 else "train"


def select_matchups(seed, per_category, split, exclude_manifest=""):
    paths = sorted(Path("teams/reg_mc").glob("MC*.txt"))
    random.Random(seed).shuffle(paths)
    counts = Counter()
    selected = []
    seen = set()
    if exclude_manifest:
        prior = json.loads(Path(exclude_manifest).read_text())
        seen.update(
            roster_split(Path(m["path"]).read_text())[0] for m in prior["matchups"]
        )
    for path in paths:
        text = path.read_text()
        digest = hashlib.sha256(text.encode()).hexdigest()
        roster, team_split = roster_split(text)
        if team_split != split or roster in seen:
            continue
        group = category(text)
        if counts[group] >= per_category:
            continue
        counts[group] += 1
        seen.add(roster)
        selected.append({"path": str(path), "category": group, "sha256": digest})
    return selected


def append_cell(path, rows):
    """Atomic whole-file replacement: a crash cannot leave a partial cell."""
    old = path.read_text() if path.exists() else ""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(old)
        stream.write("".join(json.dumps(row) + "\n" for row in rows))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


async def play_cell(ours, foe, repeats, timeout):
    await asyncio.wait_for(ours.battle_against(foe, n_battles=repeats), timeout)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--opponent", required=True, help="checkpoint path or heuristic")
    ap.add_argument("--deterministic-opponent", action="store_true")
    ap.add_argument(
        "--pair-from", default="", help="reuse opponent previews from a control JSONL"
    )
    ap.add_argument(
        "--audit-dir", default="", help="audit first roster per category/mode"
    )
    ap.add_argument(
        "--exclude-matchups",
        default="",
        help="exclude rosters in a prior study manifest",
    )
    ap.add_argument("--plans", default="data/opening_plans_t6.json")
    ap.add_argument("--per-category", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20921)
    ap.add_argument("--split", choices=["train", "heldout"], default="train")
    ap.add_argument("--port", type=int, default=7610)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--arms", default="all", help="all or comma-separated arm names")
    ap.add_argument("--cell-timeout", type=float, default=300)
    args = ap.parse_args()
    if args.split == "train" and args.opponent != "heuristic":
        refuse_eval_only_checkpoint(args.opponent)
    pokeenv_patches.install()
    torch.set_num_threads(1)
    PolicyPlayer.use_knowledge_obs = PolicyPlayer.use_moveset_prior = True
    PolicyPlayer.use_knowledge_guards = PolicyPlayer.mask_immunities = True
    extras = {"resisted_target", "overkill_split", "dominated_weather_ball_weather"}
    PolicyPlayer.guard_flags = {
        name: name in HARD_GUARDS or name in extras for name in GUARDS
    }
    spec = json.loads(Path(args.plans).read_text())
    all_arms = {"policy": None, **spec["plans"]}
    wanted = list(all_arms) if args.arms == "all" else args.arms.split(",")
    if not wanted or wanted[0] != "policy" or not set(wanted) <= set(all_arms):
        raise ValueError("arms must start with policy and name existing plans")
    if args.repeats < 1 or args.per_category < 1 or args.cell_timeout <= 0:
        raise ValueError("study sizes and timeout must be positive")
    matchups = select_matchups(
        args.seed, args.per_category, args.split, args.exclude_matchups
    )
    config = {
        **vars(args),
        "output": str(args.output),
        "matchups": matchups,
        "checkpoint_sha256": hashlib.sha256(
            Path(args.checkpoint).read_bytes()
        ).hexdigest(),
        "opponent_sha256": (
            hashlib.sha256(Path(args.opponent).read_bytes()).hexdigest()
            if args.opponent != "heuristic"
            else None
        ),
        "pair_source_sha256": (
            hashlib.sha256(Path(args.pair_from).read_bytes()).hexdigest()
            if args.pair_from
            else None
        ),
        "plans": spec,
        "battle_rng_paired": False,
        "chunk_obs_len": chunk_obs_len,
        "set_prior_reg": prior_reg(format_map["mc"]),
        "set_prior_sha256": hashlib.sha256(
            (DATA / f"joint_sets_reg{prior_reg(format_map['mc'])}.json").read_bytes()
        ).hexdigest(),
        "team_sha256": hashlib.sha256(Path(spec["team"]).read_bytes()).hexdigest(),
        "guard_flags": PolicyPlayer.guard_flags,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = args.output.with_suffix(".manifest.json")
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("output belongs to another study")
    manifest.write_text(json.dumps(config, indent=2) + "\n")
    existing = (
        [json.loads(s) for s in args.output.read_text().splitlines()]
        if args.output.exists()
        else []
    )
    paired = (
        [json.loads(s) for s in Path(args.pair_from).read_text().splitlines()]
        if args.pair_from
        else []
    )
    done = Counter((r["opponent"], r["arm"], r["hidden_sheets"]) for r in existing)
    server = ServerConfiguration(
        f"ws://localhost:{args.port}/showdown/websocket",
        "https://play.pokemonshowdown.com/action.php?",
    )
    players = []
    for hidden in (True, False):
        common = dict(
            server_configuration=server,
            battle_format=format_map["mc"],
            log_level=40,
            max_concurrent_battles=8,
            accept_open_team_sheet=not hidden,
            open_timeout=None,
        )
        ours = StudyPlayer(
            account_configuration=fresh_local_account(),
            deterministic=True,
            team=RandomTeamBuilder(args.seed, 1, "mc", [Path(spec["team"])]),
            **common,
        )
        ours.set_policy(Path(args.checkpoint), torch.device(args.device))
        if args.opponent == "heuristic":
            foe = StudyHeuristic(account_configuration=fresh_local_account(), **common)
        else:
            foe = StudyOpponent(
                deterministic=args.deterministic_opponent,
                account_configuration=fresh_local_account(),
                **common,
            )
            foe.set_policy(Path(args.opponent), torch.device("cpu"))
        players.append((ours, foe))
    arms = {arm: all_arms[arm] for arm in wanted}
    audited = set()
    for matchup in matchups:
        path = Path(matchup["path"])
        for hidden, (ours, foe) in zip((True, False), players):
            ledger = PreviewLedger()
            for row in paired or existing:
                if (row["opponent"], row["arm"], row["hidden_sheets"]) == (
                    path.name,
                    "policy",
                    hidden,
                ):
                    ledger.record(
                        tuple(row["opponent_fingerprint"]), row["opponent_preview"]
                    )
            for arm, plan in arms.items():
                key = (path.name, arm, hidden)
                if done[key] not in (0, args.repeats):
                    raise ValueError(
                        f"incomplete or duplicated cell {key}: {done[key]}"
                    )
                if done[key] == args.repeats:
                    continue
                # Append full cells only; resumption cannot mix incomplete trials.
                ours.forced_plan = plan
                ours.preview_records.clear()
                foe._team = RandomTeamBuilder(args.seed, 1, "mc", [path])
                foe.preview_ledger = ledger
                foe.replay_previews = bool(args.pair_from) or (
                    arm != "policy" and ledger.recorded > 0
                )
                if foe.replay_previews:
                    ledger.begin_replay()
                PolicyPlayer.guard_fire_counts.clear()
                PolicyPlayer._threat_cache.clear()
                PolicyPlayer._knowledge_cache.clear()
                random.seed(args.seed)
                torch.manual_seed(args.seed)
                audit_key = (matchup["category"], hidden, arm)
                audit_dir = None
                if args.audit_dir and audit_key not in audited:
                    audit_dir = Path(args.audit_dir) / f"{path.stem}_{hidden}_{arm}"
                    audit_dir.mkdir(parents=True, exist_ok=True)
                    audited.add(audit_key)
                ours.decision_log_path = (
                    audit_dir / "decisions.jsonl" if audit_dir else None
                )
                started = time.monotonic()
                asyncio.run(play_cell(ours, foe, args.repeats, args.cell_timeout))
                failures = {
                    k: v
                    for k, v in PolicyPlayer.guard_fire_counts.items()
                    if "error:" in k or "fallback_replaced" in k
                }
                if failures or ledger.mismatched:
                    raise RuntimeError(
                        f"unusable opening cell: {failures}; "
                        f"pairing drift {ledger.mismatched}"
                    )
                rows = []
                for b in ours.battles.values():
                    if not b.finished or b.won is None:
                        raise RuntimeError(
                            "unfinished/tied battle needs explicit handling"
                        )
                    rows.append(
                        {
                            "opponent": path.name,
                            "battle": b.battle_tag,
                            "category": matchup["category"],
                            "arm": arm,
                            "hidden_sheets": hidden,
                            "target": float(bool(b.won)),
                            "first_faint": _first_faint_side(b),
                            "opponent_preview": foe.preview_records[b.battle_tag],
                            # End-of-game opponent_team contains the revealed four,
                            # not the preview six. Capture before choosing, not here.
                            "opponent_fingerprint": foe.preview_fingerprints[
                                b.battle_tag
                            ],
                            "turns": b.turn,
                            "split": args.split,
                            **ours.preview_records[b.battle_tag],
                        }
                    )
                    if audit_dir:
                        b.save_replay(audit_dir / f"{b.battle_tag}.html")
                        (audit_dir / f"{b.battle_tag}.json").write_text(
                            json.dumps(
                                {
                                    "battle": b.battle_tag,
                                    "our_role": b.player_role,
                                    "won": b.won,
                                    "protocol": b._replay_data,
                                }
                            )
                            + "\n"
                        )
                if len(rows) != args.repeats:
                    raise RuntimeError(
                        f"expected {args.repeats} games, got {len(rows)}"
                    )
                append_cell(args.output, rows)
                append_cell(
                    args.output.with_suffix(".telemetry.jsonl"),
                    [
                        {
                            "opponent": path.name,
                            "arm": arm,
                            "hidden_sheets": hidden,
                            "games": len(rows),
                            "elapsed_s": time.monotonic() - started,
                            "guard_counts": dict(PolicyPlayer.guard_fire_counts),
                            "preview_pairing_mismatches": ledger.mismatched,
                            "audited": bool(audit_dir),
                        }
                    ],
                )
                print(
                    f"{path.name} {matchup['category']} hidden={hidden} {arm}: "
                    f"{sum(r['target'] for r in rows):.0f}/{len(rows)}",
                    flush=True,
                )
                ours.reset_battles()
                foe.reset_battles()


if __name__ == "__main__":
    main()
