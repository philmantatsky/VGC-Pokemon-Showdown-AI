"""Ladder a trained checkpoint on real Pokemon Showdown with our fixed team.

Differs from vgc_bench/play.py in three ways that matter here:
  * credentials come from the environment, never a CLI arg (command lines leak into
    `ps` output and shell history)
  * our side always brings --our_team, matching what training optimized for, instead
    of drawing from the team pool
  * device is configurable (play.py hardcodes cuda:0) and per-turn debug is off

Credentials (set these in the environment, do not pass them as flags):
    SHOWDOWN_USERNAME, SHOWDOWN_PASSWORD

    set -a && source /path/to/.env && set +a
    python ladder_ourteam.py --checkpoint <path.zip> --n_games 50
"""

import argparse
import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from poke_env import AccountConfiguration, ShowdownServerConfiguration
from torch import device

from vgc_bench.src.policy import MaskedActorCriticPolicy
from vgc_bench.src.policy_player import PolicyPlayer
from vgc_bench.src.teams import RandomTeamBuilder
from vgc_bench.src.utils import format_map

# Arguments that may differ between runs sharing one replay directory; everything
# else is "material" and must stay identical so each directory is single-config.
VOLATILE_ARGS = {
    "n_games",
    "challenges",
    "debug",
    "replay_dir",
    "decision_log",
    "deployment",
}


def resolve_knowledge_obs(explicit: bool | None, ckpt: Path, ckpt_sha: str) -> bool:
    """Resolve the knowledge-feature requirement: flag -> sidecar -> hard fail.

    A silent default burned a whole ladder era -- the champion requires these
    features and omitting the flag zeroed them with no error of any kind.
    """
    sidecar = Path(str(ckpt) + ".metadata.json")
    sidecar_meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    if sidecar_meta.get("sha256") and sidecar_meta["sha256"] != ckpt_sha:
        raise SystemExit(
            f"{sidecar} was stamped for a different file "
            f"(sha256 {sidecar_meta['sha256'][:12]}... vs {ckpt_sha[:12]}...).\n"
            f"The checkpoint changed since stamping; restamp it:\n"
            f"  .venv/bin/python stamp_checkpoint_metadata.py '{ckpt}'"
        )
    sidecar_requires = sidecar_meta.get("requires_knowledge_obs")
    if explicit is None:
        if sidecar_requires is None:
            raise SystemExit(
                f"knowledge_obs is unresolved for {ckpt}.\n"
                f"No {sidecar.name} sidecar exists and neither --knowledge_obs nor "
                "--no_knowledge_obs was passed. Stamp the checkpoint once:\n"
                f"  .venv/bin/python stamp_checkpoint_metadata.py '{ckpt}'\n"
                "or pass the flag explicitly."
            )
        return bool(sidecar_requires)
    if sidecar_requires is not None and bool(sidecar_requires) != explicit:
        flag = "--knowledge_obs" if explicit else "--no_knowledge_obs"
        raise SystemExit(
            f"{flag} contradicts {sidecar} "
            f"(requires_knowledge_obs={sidecar_requires}). Fix the flag or restamp."
        )
    return explicit


def record_run_config(
    replay_dir: Path, args: argparse.Namespace, ckpt_sha: str, guard_profile: str
) -> Path:
    """Append this run's resolved configuration to <replay_dir>/run_config.json.

    Refuses a replay directory whose recorded material configuration differs, so
    every directory stays single-config and later analysis can trust it.
    """
    replay_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = replay_dir / "run_config.json"
    material = {k: v for k, v in vars(args).items() if k not in VOLATILE_ARGS}
    material["checkpoint_sha256"] = ckpt_sha
    material["guard_profile_resolved"] = guard_profile
    material["mask_immunities"] = not args.no_immunity_mask
    material["moveset_prior"] = not args.no_moveset_prior
    run_config = (
        json.loads(run_config_path.read_text())
        if run_config_path.exists()
        else {"runs": []}
    )
    prior_runs = run_config.get("runs") or []
    if prior_runs and prior_runs[-1].get("material") != material:
        prior_material = prior_runs[-1].get("material") or {}
        changed = sorted(
            key
            for key in set(prior_material) | set(material)
            if prior_material.get(key) != material.get(key)
        )
        raise SystemExit(
            f"{run_config_path} already records a different configuration "
            f"(changed: {', '.join(changed)}).\n"
            "Use a fresh --replay_dir so each directory stays single-config."
        )
    run_config.setdefault("runs", []).append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_games": args.n_games,
            "challenges": args.challenges,
            "material": material,
        }
    )
    run_config_path.write_text(json.dumps(run_config, indent=1, default=str))
    return run_config_path


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default="",
        help=(
            "path to the .zip policy; when omitted, use a passed counterfactual "
            "deployment or fall back to results_repaired/champion.zip"
        ),
    )
    ap.add_argument(
        "--deployment",
        default="results_counterfactual/deployment.json",
        help="promotion manifest produced by run_counterfactual_pipeline.py",
    )
    ap.add_argument("--our_team", default="teams/reg_mb/our_team.txt")
    ap.add_argument("--reg", default="mb")
    ap.add_argument("--n_games", type=int, default=50)
    ap.add_argument("--device", default="mps")
    ap.add_argument(
        "--opening-wait",
        "--opening_wait",
        dest="opening_wait",
        type=float,
        default=20.0,
        help=(
            "seconds to wait at Team Preview for the opponent's Open Team Sheets "
            "answer before choosing anyway (default: 20)"
        ),
    )
    ap.add_argument(
        "--preview_model",
        default="",
        help=(
            "optional replay-trained bring/lead model; when omitted, retain the "
            "checkpoint's original two-stage team preview"
        ),
    )
    ap.add_argument(
        "--learned_preview",
        action="store_true",
        help=(
            "let --preview_model choose our four/leads; experimental because the "
            "specialized policy is still stronger in local A/B"
        ),
    )
    outcome_preview_group = ap.add_mutually_exclusive_group()
    outcome_preview_group.add_argument(
        "--outcome-preview",
        dest="outcome_preview",
        action="store_true",
        help=(
            "EXPERIMENTAL rejected candidate: use the terminal self-play outcome "
            "model when team sheets are hidden; open-sheet battles retain champion "
            "Team Preview"
        ),
    )
    outcome_preview_group.add_argument(
        "--no-outcome-preview",
        dest="outcome_preview",
        action="store_false",
        help="retain champion Team Preview in every battle",
    )
    ap.set_defaults(outcome_preview=False)
    ap.add_argument("--preview-outcome-model", default="data/preview_outcome_regmb.pt")
    lead_group = ap.add_mutually_exclusive_group()
    lead_group.add_argument(
        "--stable-lead",
        dest="stable_lead",
        action="store_true",
        help=(
            "lead Whimsicott + Basculegion while the policy chooses the back two "
            "(experimental; the observational ladder split did not survive A/B)"
        ),
    )
    lead_group.add_argument(
        "--adaptive-lead",
        dest="stable_lead",
        action="store_false",
        help="let the checkpoint choose both leads as before",
    )
    ap.set_defaults(stable_lead=False)
    ap.add_argument(
        "--switch_model",
        default="",
        help="optional top-player switch predictor used by opponent-aware reranking",
    )
    ap.add_argument(
        "--move_model",
        default="",
        help=(
            "optional top-player move/target predictor used by opponent-aware reranking"
        ),
    )
    ap.add_argument(
        "--residual_ranker",
        default="",
        help="optional confidence-gated joint-action residual checkpoint",
    )
    opponent_group = ap.add_mutually_exclusive_group()
    opponent_group.add_argument(
        "--opponent-aware",
        "--opponent_aware",
        dest="opponent_aware",
        action="store_true",
        help="enable the learned opponent-planning layer (the default)",
    )
    opponent_group.add_argument(
        "--no-opponent-aware",
        "--no_opponent_aware",
        dest="opponent_aware",
        action="store_false",
        help="disable all learned opponent priors for a baseline/A-B run",
    )
    ap.set_defaults(opponent_aware=True)
    tempo_group = ap.add_mutually_exclusive_group()
    tempo_group.add_argument(
        "--tempo-aware",
        dest="tempo_aware",
        action="store_true",
        help=(
            "enable Trick Room/Tailwind speed order, Protect coordination, and "
            "timing-aware Encore scoring (the default)"
        ),
    )
    tempo_group.add_argument(
        "--no-tempo-aware",
        dest="tempo_aware",
        action="store_false",
        help="disable the mechanics-aware tempo layer for an A/B run",
    )
    ap.set_defaults(tempo_aware=True)
    ap.add_argument(
        "--bench-species",
        default="",
        help=(
            "comma-separated species to force OUT of the bring-four "
            "(bring-selection experiment; stands down if the roster lacks them)"
        ),
    )
    ap.add_argument(
        "--challenges",
        action="store_true",
        help="accept challenges instead of laddering (for testing)",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="print the full action distribution each turn (very noisy)",
    )
    ap.add_argument(
        "--replay_dir",
        default="ladder_replays",
        help="save every finished battle here as HTML for later analysis",
    )
    ap.add_argument(
        "--decision_log",
        default="",
        help=(
            "JSONL decision audit; defaults to <replay_dir>/decisions.jsonl so a "
            "replay preserves why each joint action was chosen"
        ),
    )
    ap.add_argument(
        "--search",
        action="store_true",
        help=("two-turn risk-aware exact Showdown search with a hard turn budget"),
    )
    ap.add_argument(
        "--outcome_value",
        default="results_outcome_v2h/outcome_value.zip",
        help="calibrated terminal-outcome checkpoint used at exact search leaves",
    )
    # Live observation/policy bookkeeping added roughly one second beyond the exact
    # planner's own timer in the first serial ladder gate. Eight seconds keeps the
    # complete decision comfortably below ten without weakening the two-second root
    # screen.
    ap.add_argument("--search_budget", type=float, default=8.0)
    ap.add_argument("--screen_budget", type=float, default=2.0)
    ap.add_argument("--chance_samples", type=int, default=1)
    ap.add_argument("--deep-root-width", type=int, default=4)
    ap.add_argument("--determinizations", type=int, default=8)
    ap.add_argument(
        "--search-determinizations",
        type=int,
        default=2,
        help=(
            "belief worlds that receive foreground deep-search time; all hidden "
            "worlds are still tracked and conditioned"
        ),
    )
    ap.add_argument("--min-deep-coverage", type=float, default=0.50)
    preview_search_group = ap.add_mutually_exclusive_group()
    preview_search_group.add_argument(
        "--planned-preview",
        dest="planned_preview",
        action="store_true",
        help="experimentally use exact first-turn simulation for Team Preview",
    )
    preview_search_group.add_argument(
        "--no-planned-preview",
        dest="planned_preview",
        action="store_false",
        help="retain the promoted champion Team Preview (default)",
    )
    ap.set_defaults(planned_preview=False)
    ap.add_argument("--preview-search-budget", type=float, default=8.0)
    ap.add_argument(
        "--preview-determinizations",
        type=int,
        default=1,
        help=(
            "hidden-set worlds the exact preview planner samples and merges; "
            "the VGC Timer allows 90s at Team Preview against a 420s bank, so "
            "several full worlds fit without threatening move-turn time"
        ),
    )
    ap.add_argument(
        "--search-every-turn",
        action="store_true",
        help=(
            "disable contingent-plan reuse and force a fresh exact search on every "
            "move request (A/B control; selective search is the default)"
        ),
    )
    ponder_group = ap.add_mutually_exclusive_group()
    ponder_group.add_argument(
        "--ponder",
        dest="ponder",
        action="store_true",
        help=(
            "experimentally expand opponent responses while waiting for the "
            "next request"
        ),
    )
    ponder_group.add_argument(
        "--no-ponder",
        dest="ponder",
        action="store_false",
        help="disable background pondering (default)",
    )
    ap.set_defaults(ponder=False)
    ap.add_argument("--ponder-budget", type=float, default=6.0)
    ap.add_argument("--ponder-choices", type=int, default=96)
    ap.add_argument("--ponder-chance-samples", type=int, default=1)
    ap.add_argument(
        "--no_guards",
        action="store_true",
        help="disable the knowledge guard stack entirely (for A/B)",
    )
    ap.add_argument(
        "--guard_profile",
        choices=("hard", "all", "none"),
        default="hard",
        help=(
            "hard factual guards (default), all experimental guards, or none; "
            "the full stack underperformed in learned-population evaluation"
        ),
    )
    ap.add_argument(
        "--no_immunity_mask",
        action="store_true",
        help="allow damaging moves the target is immune to (for A/B)",
    )
    ap.add_argument(
        "--no_moveset_prior",
        action="store_true",
        help="disable filling unknown opponent sets from Smogon usage "
        "stats (only matters when an opponent denies team sheets)",
    )
    knowledge_group = ap.add_mutually_exclusive_group()
    knowledge_group.add_argument(
        "--knowledge_obs",
        dest="knowledge_obs",
        action="store_true",
        help="populate the 24 knowledge features per Pokemon token; "
        "REQUIRED for checkpoints fine-tuned with --knowledge_obs, "
        "and wrong for older ones",
    )
    knowledge_group.add_argument(
        "--no_knowledge_obs",
        dest="knowledge_obs",
        action="store_false",
        help="force the 24 knowledge features to stay zero "
        "(only correct for pre-conversion checkpoints)",
    )
    # None means "unresolved": the checkpoint's sidecar metadata must decide, and
    # startup hard-fails if it cannot. A silent default burned a whole ladder era --
    # the champion requires these features and omitting the flag zeroed them with no
    # error of any kind.
    ap.set_defaults(knowledge_obs=None)
    args = ap.parse_args()

    deployment_label = "not used"
    if not args.checkpoint:
        deployment_path = Path(args.deployment)
        if deployment_path.exists():
            deployment = json.loads(deployment_path.read_text())
            if deployment.get("passed"):
                args.checkpoint = deployment["checkpoint"]
                args.preview_model = deployment["preview_model"]
                args.learned_preview = bool(deployment.get("learned_preview"))
                args.search = bool(deployment.get("use_search", False))
                args.residual_ranker = deployment.get("residual_ranker") or ""
                if (
                    args.knowledge_obs is None
                    and "requires_knowledge_obs" in deployment
                ):
                    args.knowledge_obs = bool(deployment["requires_knowledge_obs"])
                deployment_label = str(deployment_path)
        if not args.checkpoint:
            args.checkpoint = "results_repaired/champion.zip"
            deployment_label = "current repaired champion (promotion not ready)"

    if args.search:
        from vgc_bench.src.search import backend_status

        ready, reason = backend_status()
        if not ready:
            raise SystemExit(f"Search refused: {reason}")
        if not 0 < args.search_budget <= 9:
            raise SystemExit("--search_budget must be in (0, 9]")
        if not 0 < args.screen_budget <= args.search_budget:
            raise SystemExit("--screen_budget must be within the search budget")
        if not 1 <= args.chance_samples <= 4:
            raise SystemExit("--chance_samples must be between 1 and 4")
        if not 1 <= args.deep_root_width <= 6:
            raise SystemExit("--deep-root-width must be between 1 and 6")
        if not 1 <= args.determinizations <= 8:
            raise SystemExit("--determinizations must be between 1 and 8")
        if not 1 <= args.search_determinizations <= args.determinizations:
            raise SystemExit(
                "--search-determinizations must be between 1 and --determinizations"
            )
        if not 0 <= args.min_deep_coverage <= 1:
            raise SystemExit("--min-deep-coverage must be within [0, 1]")
        # Preview allowance is the VGC Timer's 90-second first request, not the
        # 55-second move-turn clock; 60 leaves a two-thirds safety margin.
        if not 0.1 <= args.preview_search_budget <= 60:
            raise SystemExit("--preview-search-budget must be within [0.1, 60]")
        if not 1 <= args.preview_determinizations <= 8:
            raise SystemExit("--preview-determinizations must be within [1, 8]")
        if not 0.1 <= args.ponder_budget <= 15:
            raise SystemExit("--ponder-budget must be between 0.1 and 15 seconds")
        if args.ponder_choices < 1:
            raise SystemExit("--ponder-choices must be positive")
        if not 1 <= args.ponder_chance_samples <= 2:
            raise SystemExit("--ponder-chance-samples must be 1 or 2")
        print(f"search backend: {reason}", flush=True)

    # A raise inside poke-env's message handler stalls the battle permanently, which
    # on ladder means burning the clock and losing. See the module docstring.
    from vgc_bench.src import pokeenv_patches

    pokeenv_patches.install()

    username = os.environ.get("SHOWDOWN_USERNAME")
    password = os.environ.get("SHOWDOWN_PASSWORD")
    if not username:
        raise SystemExit(
            "SHOWDOWN_USERNAME is not set.\n"
            "Load your .env first, e.g.:\n"
            "  set -a && source '<path to>/.env' && set +a"
        )
    if not password:
        print("WARNING: SHOWDOWN_PASSWORD not set; connecting as an unregistered name.")

    # Reg M-B team sheets are opt-in, so opponents can deny them. Training always had
    # sheets; this keeps unknown opponent slots plausible rather than blank.
    PolicyPlayer.use_moveset_prior = not args.no_moveset_prior
    # Type effectiveness is absent from the observation, so the network would have to
    # learn all 18x18 interactions from self-play; immunities are rare and learned
    # last. Enforce them as a hard rule instead.
    # Kept ON even with the guards. The old rationale -- "calculate_damage supersedes a
    # type-chart lookup" -- holds only while the calc can actually run, and it cannot
    # when an opponent denies Open Team Sheets. That combination left the bot with no
    # type knowledge whatsoever in exactly those games. This mask needs no stats, so it
    # is the one immunity check that always works.
    PolicyPlayer.mask_immunities = not args.no_immunity_mask
    # Laplace-style guard stack: the policy ranks joint action pairs, explicit
    # knowledge reorders a prefix. See src/guards.py.
    guard_profile = "none" if args.no_guards else args.guard_profile
    PolicyPlayer.use_knowledge_guards = guard_profile != "none"
    if guard_profile == "hard":
        from vgc_bench.src.guards import GUARDS, HARD_GUARDS

        PolicyPlayer.guard_flags = {name: name in HARD_GUARDS for name in GUARDS}
    else:
        PolicyPlayer.guard_flags = None
    PolicyPlayer.use_search = args.search

    team_path = Path(args.our_team)
    assert team_path.exists(), f"team not found: {team_path}"
    ckpt = Path(args.checkpoint)
    assert ckpt.exists(), f"checkpoint not found: {ckpt}"
    ckpt_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()

    args.knowledge_obs = resolve_knowledge_obs(args.knowledge_obs, ckpt, ckpt_sha)
    # Set the env var too, for anything that imports fresh in another process; clear
    # it when off so a stale export cannot re-enable the features.
    if args.knowledge_obs:
        os.environ["VGC_KNOWLEDGE_OBS"] = "1"
    else:
        os.environ.pop("VGC_KNOWLEDGE_OBS", None)
    PolicyPlayer.use_knowledge_obs = args.knowledge_obs
    print(
        f"knowledge_obs={args.knowledge_obs}  guards={guard_profile}  "
        f"search={args.search}",
        flush=True,
    )
    if (
        sum(
            bool(value)
            for value in (
                args.learned_preview,
                args.outcome_preview,
                args.planned_preview,
            )
        )
        > 1
    ):
        raise SystemExit("choose only one Team Preview controller")
    if args.opponent_aware or args.outcome_preview or args.planned_preview:
        args.preview_model = (
            args.preview_model or "data/opponent_preview_top500_regmb.pt"
        )
    if args.opponent_aware:
        args.switch_model = args.switch_model or "data/opponent_switch_top500_regmb.pt"
        args.move_model = args.move_model or "data/opponent_move_top500_regmb.pt"
    preview_model = Path(args.preview_model) if args.preview_model else None
    if preview_model is not None:
        assert preview_model.exists(), f"preview model not found: {preview_model}"
    preview_outcome_model = Path(args.preview_outcome_model)
    if args.outcome_preview:
        assert preview_outcome_model.exists(), (
            f"preview outcome model not found: {preview_outcome_model}"
        )
    switch_model = Path(args.switch_model) if args.switch_model else None
    if switch_model is not None:
        assert switch_model.exists(), f"switch model not found: {switch_model}"
    move_model = Path(args.move_model) if args.move_model else None
    if move_model is not None:
        assert move_model.exists(), f"move model not found: {move_model}"
    residual_ranker = Path(args.residual_ranker) if args.residual_ranker else None
    if residual_ranker is not None:
        assert residual_ranker.exists(), f"residual model not found: {residual_ranker}"
    outcome_value = Path(args.outcome_value)
    if args.search:
        assert outcome_value.exists(), f"outcome model not found: {outcome_value}"

    exact_search_config = None
    ponder_config = None
    if args.search:
        from vgc_bench.src.exact_planner import PlannerConfig
        from vgc_bench.src.ponder import PonderConfig

        exact_search_config = PlannerConfig(
            depth=2,
            root_width=6,
            opponent_width=6,
            continuation_width=3,
            replacement_width=2,
            chance_samples=args.chance_samples,
            deep_root_width=args.deep_root_width,
            anytime=True,
            screen_budget_s=args.screen_budget,
            time_budget_s=args.search_budget,
            max_nodes=5000,
        )
        ponder_config = PonderConfig(
            budget_s=args.ponder_budget,
            max_opponent_choices=args.ponder_choices,
            chance_samples=args.ponder_chance_samples,
            max_roots=args.determinizations,
        )

    # Persist the resolved configuration next to the replays so every batch is
    # auditable from disk (stdout was previously the only record).
    run_config_path = record_run_config(
        Path(args.replay_dir), args, ckpt_sha, guard_profile
    )

    agent = PolicyPlayer(
        account_configuration=AccountConfiguration(username, password),
        battle_format=format_map[args.reg],
        log_level=40,
        max_concurrent_battles=1,
        server_configuration=ShowdownServerConfiguration,
        accept_open_team_sheet=True,
        start_timer_on_battle_start=not args.challenges,
        save_replays=args.replay_dir,
        team=RandomTeamBuilder(1, 1, args.reg, [team_path]),
        preview_model_path=preview_model,
        preview_outcome_model_path=(
            preview_outcome_model if args.outcome_preview else None
        ),
        switch_model_path=switch_model,
        move_model_path=move_model,
        residual_ranker_path=residual_ranker,
        use_learned_teampreview=args.learned_preview,
        use_outcome_teampreview=args.outcome_preview,
        forced_lead_species=("whimsicott", "basculegion")
        if args.stable_lead and not args.learned_preview and not args.outcome_preview
        else None,
        forced_bench_species=(
            tuple(
                name.strip() for name in args.bench_species.split(",") if name.strip()
            )
            or None
        )
        if args.bench_species
        else None,
        use_opponent_reranker=args.opponent_aware,
        use_tempo_reranker=args.tempo_aware,
        team_sheet_wait_timeout=args.opening_wait,
        decision_log_path=(
            Path(args.decision_log)
            if args.decision_log
            else Path(args.replay_dir) / "decisions.jsonl"
        ),
        exact_team_path=team_path,
        outcome_value_path=(
            outcome_value if args.search or args.planned_preview else None
        ),
        exact_search_config=exact_search_config,
        exact_max_determinizations=args.determinizations,
        exact_search_determinizations=args.search_determinizations,
        exact_min_deep_coverage=args.min_deep_coverage,
        exact_preview_search=args.search and args.planned_preview,
        exact_preview_budget=args.preview_search_budget,
        exact_preview_determinizations=args.preview_determinizations,
        exact_selective_search=not args.search_every_turn,
        exact_enable_ponder=args.search and args.ponder,
        exact_ponder_config=ponder_config,
        enable_search=args.search,
    )
    agent.set_policy(ckpt, device(args.device))
    assert isinstance(agent.policy, MaskedActorCriticPolicy)
    agent.policy.debug = args.debug

    print(f"account : {username}")
    print(f"format  : {format_map[args.reg]}")
    print(f"team    : {team_path.name}")
    print(f"policy  : {ckpt}")
    print(f"deploy  : {deployment_label}")
    print(f"opening : {args.opening_wait:g}s max for the team-sheet reply")
    if args.search:
        search_mode = (
            "every turn" if args.search_every_turn else "selective + plan reuse"
        )
        print(f"planning: {search_mode}")
        print(
            "ponder  : "
            + (
                f"on, {args.ponder_budget:g}s / {args.ponder_choices} replies / "
                f"{args.ponder_chance_samples} RNG"
                if args.ponder
                else "off"
            )
        )
    preview_selector = (
        "terminal outcome"
        if args.outcome_preview
        else "learned"
        if args.learned_preview
        else (
            "stable Whimsicott/Basculegion lead + policy backline"
            if args.stable_lead
            else "checkpoint policy"
        )
    )
    print(f"preview : {preview_selector}; opponent model={preview_model or 'off'}")
    print(f"switch  : {switch_model or 'off'}")
    print(f"moves   : {move_model or 'off'}")
    print(f"residual: {residual_ranker or 'off'}")
    print(f"rerank  : {'on' if args.opponent_aware else 'off'}")
    print(f"tempo   : {'on' if args.tempo_aware else 'off'}")
    if args.search:
        print(
            "exact   : "
            f"depth=2 budget={args.search_budget:g}s "
            f"screen={args.screen_budget:g}s rng={args.chance_samples} "
            f"hidden={args.determinizations} "
            f"deep_worlds={args.search_determinizations} "
            f"min_deep={args.min_deep_coverage:.2f} "
            f"preview={'exact' if args.planned_preview else 'champion'} "
            f"outcome={outcome_value}"
        )
    print(
        "audit   : "
        f"{args.decision_log or str(Path(args.replay_dir) / 'decisions.jsonl')}"
    )
    print(f"config  : recorded in {run_config_path}")

    if args.challenges:
        print(f"awaiting {args.n_games} challenge(s)...")
        await agent.accept_challenges(opponent=None, n_challenges=args.n_games)
    else:
        print(f"laddering {args.n_games} games...")
        await agent.ladder(n_games=args.n_games)

    counts = dict(PolicyPlayer.guard_fire_counts)
    if counts:
        print("\nguard firings (times a guard changed the top pick):")
        for k, v in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {k:26} {v}")
    if args.search:
        from vgc_bench.src.search import latency_summary

        print(latency_summary())
    print(pokeenv_patches.report())

    wins = agent.n_won_battles
    losses = agent.n_lost_battles
    ties = agent.n_tied_battles
    total = wins + losses + ties
    print(f"\nrecord: {wins}-{losses}-{ties}")
    if total:
        print(f"win rate: {wins / total * 100:.0f}% over {total} games")


if __name__ == "__main__":
    asyncio.run(main())
