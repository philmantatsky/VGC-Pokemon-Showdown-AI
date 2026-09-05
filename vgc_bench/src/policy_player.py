"""
Policy-based player module for VGC-Bench.

Provides player implementations that use neural network policies to make
battle decisions, including synchronous and batched asynchronous variants.
Also implements the battle state embedding used for policy observations.
"""

import asyncio
import io
import json
import os
import threading
import time
import traceback
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable

import numpy as np
import numpy.typing as npt
import torch
from poke_env.battle import (
    AbstractBattle,
    DoubleBattle,
    Effect,
    Field,
    Move,
    MoveCategory,
    Pokemon,
    PokemonGender,
    PokemonType,
    SideCondition,
    Status,
    Target,
    Weather,
)
from poke_env.data import GenData, to_id_str
from poke_env.environment import DoublesEnv
from poke_env.player import BattleOrder, DefaultBattleOrder, Player
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy

from vgc_bench.src.move_semantics import MOVE_SEM_LEN, ability_semantics, move_semantics
from vgc_bench.src.opponent_preview import (
    BattlePlanState,
    OpponentBelief,
    PreviewPlan,
    PreviewPredictor,
    plan_to_showdown_order,
)
from vgc_bench.src.opponent_tactics import (
    MovePrediction,
    MovePredictor,
    SwitchPrediction,
    SwitchPredictor,
)
from vgc_bench.src.policy import MaskedActorCriticPolicy
from vgc_bench.src.teams import RandomTeamBuilder
from vgc_bench.src.utils import (
    abilities,
    act_len,
    correct_accuracy_obs_len,
    get_reg_from_format,
    global_presence_obs_len,
    items,
    knowledge_obs_len,
    move_obs_len,
    moves,
    pokemon_obs_len,
)

_ZERO_KNOWLEDGE = np.zeros(knowledge_obs_len, dtype=np.float32)
_ZERO_MOVE_SEM = np.zeros(MOVE_SEM_LEN, dtype=np.float32)


class PolicyPlayer(Player):
    """
    A Pokemon VGC player that uses a neural network policy for decisions.

    Handles battle state embedding and action masking to ensure only legal
    moves are selected.

    Attributes:
        policy: The neural network policy used for action selection.
    """

    policy: BasePolicy | None

    # Fill unknown OPPONENT ability/item/moves from Smogon usage stats when team
    # sheets are unavailable. Off by default so training behaviour is unchanged;
    # ladder scripts opt in. See _moveset_prior and build_movesets.py.
    # Tri-state for the same spawn reason as use_knowledge_obs below. Training with
    # hidden sheets sets VGC_MOVESET_PRIOR before workers are created; ladder/eval can
    # still assign an explicit bool.
    use_moveset_prior: bool | None = None
    _prior_cache: dict[str, Any] | None = None
    _type_chart: dict[str, dict[str, float]] | None = None
    # Forbid single-target damaging moves the foe is immune to. Type
    # effectiveness is absent from the observation, so this is a hard rule
    # rather than something learned. Off during training by default.
    mask_immunities: bool = False
    # Knowledge guard stack (see src/guards.py). Off by default so training is
    # untouched; ladder/eval scripts opt in. guard_flags can disable one at a time
    # for A/B testing, and guard_fire_counts records which guards actually earn
    # their keep rather than leaving it to guesswork.
    use_knowledge_guards: bool = False
    # One-ply simultaneous-move search over the joint action matrix, evaluated by the
    # critic (src/search.py). Inference only -- far too slow for self-play.
    use_search: bool = False
    guard_flags: dict[str, bool] | None = None
    guard_fire_counts: "Counter[str]" = Counter()
    # Populate the 24 knowledge floats per Pokemon token (damage, KO flags, type
    # multiplier, incoming threat) instead of leaving them zero. Requires a
    # checkpoint converted by convert_checkpoint.py. Off by default: an old
    # checkpoint with zeros here behaves exactly as it always did.
    # Tri-state: None means "consult VGC_KNOWLEDGE_OBS at call time", a bool is an
    # explicit override (the eval and test scripts assign one directly).
    #
    # The env var exists because self-play spawns worker processes (macOS uses spawn,
    # not fork) and a class attribute assigned in __main__ would never reach them.
    # But it must NOT be resolved at import time: train.py imports this module at
    # line 19 and only sets the env var at line 307, and PURE_SELF_PLAY builds its
    # envs in the PARENT process (train.py:84) instead of spawning. An import-time
    # read therefore sees a stale False in the one process that matters, and the run
    # silently trains on zeroed knowledge features. Resolve lazily instead --
    # see knowledge_obs_enabled().
    use_knowledge_obs: bool | None = None
    _knowledge_cache: dict[Any, dict[int, Any]] = {}

    def __init__(
        self,
        policy: BasePolicy | None = None,
        accept_all_formats: bool = False,
        deterministic: bool = False,
        invitee: str | None = None,
        preview_model_path: str | Path | None = None,
        preview_outcome_model_path: str | Path | None = None,
        switch_model_path: str | Path | None = None,
        move_model_path: str | Path | None = None,
        residual_ranker_path: str | Path | None = None,
        use_learned_teampreview: bool = False,
        use_outcome_teampreview: bool = False,
        use_opponent_reranker: bool = False,
        use_tempo_reranker: bool = False,
        forced_lead_species: tuple[str, str] | None = None,
        forced_bench_species: tuple[str, ...] | None = None,
        decision_log_path: str | Path | None = None,
        team_sheet_wait_timeout: float | None = None,
        exact_team_path: str | Path | None = None,
        outcome_value_path: str | Path | None = None,
        exact_search_config: Any | None = None,
        exact_max_determinizations: int = 8,
        exact_search_determinizations: int = 2,
        exact_min_deep_coverage: float = 0.50,
        exact_preview_search: bool = False,
        exact_preview_budget: float = 8.0,
        exact_preview_determinizations: int = 1,
        exact_selective_search: bool = False,
        exact_enable_ponder: bool = False,
        exact_ponder_config: Any | None = None,
        enable_search: bool | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initialize the policy player.

        Args:
            policy: Neural network policy (can be set later via set_policy).
            accept_all_formats: If True, accept challenges in any recognized
                VGC format instead of only ``battle_format``. Requires the
                team builder to be in multi-reg mode (``reg=None``) so the
                correct regulation's teams are yielded.
            deterministic: If True, always pick the highest-probability action
                instead of sampling from the distribution.
            preview_model_path: Optional learned bring/lead predictor used to track
                an opponent-plan belief throughout each battle.
            preview_outcome_model_path: Fixed-team terminal-outcome preview ranker.
            use_learned_teampreview: Also let that predictor replace the policy's
                two-stage preview. Off until it wins the local preview A/B.
            use_outcome_teampreview: Rerank the supported preview candidates by
                terminal self-play outcomes.
            switch_model_path: Optional high-rated turn-level switch predictor. Its
                output is advisory state for search, never a hard veto.
            move_model_path: Optional high-rated move/target predictor used to rank
                opponent action branches during future search.
            residual_ranker_path: Optional conservative joint-action residual. The
                frozen champion remains the policy and the residual only applies
                above its learned confidence threshold.
            use_opponent_reranker: Let the opponent priors softly rerank near-tied
                policy candidates. Factual guards remain hard constraints.
            use_tempo_reranker: Score exact speed order, timed Trick Room/Tailwind,
                coordinated Protect/spread turns, and timing-aware Encore use.
            forced_lead_species: Optional fixed lead pair. The policy still chooses
                the two back Pokemon after these leads are marked as selected.
            decision_log_path: Optional JSONL audit of every live joint decision.
            team_sheet_wait_timeout: Maximum seconds to wait at Team Preview for an
                opponent to accept or reject Open Team Sheets. ``None`` preserves
                poke-env's normal unlimited wait.
            exact_team_path: Fixed Showdown team export used to seed live exact roots.
            outcome_value_path: Calibrated terminal-outcome evaluator for search.
            exact_search_config: Optional ``PlannerConfig`` for live anytime search.
            exact_max_determinizations: Maximum concrete hidden worlds (at most 8).
            exact_search_determinizations: Belief worlds receiving foreground deep
                search time on one move (the full belief remains tracked).
            exact_min_deep_coverage: Minimum searched-world mass in which the chosen
                action must finish the requested future depth before it is trusted.
            exact_preview_search: Use bounded exact first-turn simulation to choose
                the bring-four and lead-two at Team Preview.
            exact_preview_budget: Wall-clock budget for the opening planner. The
                VGC Timer allows 90 seconds at Team Preview, so this may exceed
                the per-move-turn search budget.
            exact_preview_determinizations: Hidden-set worlds the opening planner
                samples and merges (at most 8; open sheets always use one).
            exact_selective_search: Reuse a matching searched continuation and skip
                quiet turns instead of starting a fresh exact search every request.
            exact_enable_ponder: Expand opponent responses in an isolated simulator
                after submitting our action, without delaying the live turn.
            exact_ponder_config: Optional ``PonderConfig`` for background expansion.
            enable_search: Per-player exact-search switch. ``None`` inherits the
                class default; evaluations use this to keep opponent players on
                their own policy while searching several controlled battles.
            *args: Additional arguments for Player base class.
            **kwargs: Additional keyword arguments for Player base class.
        """
        super().__init__(*args, **kwargs)
        self.policy = policy
        # SB3's MultiCategoricalDistribution is stateful. Local exact searches run
        # in worker threads, so all accesses to this shared policy need one lock.
        self._exact_policy_lock = threading.RLock()
        self._accept_all_formats = accept_all_formats
        self.deterministic = deterministic
        self.invitee = invitee
        self.preview_model_path = (
            Path(preview_model_path) if preview_model_path is not None else None
        )
        self._preview_predictor: PreviewPredictor | None = None
        self.preview_outcome_model_path = (
            Path(preview_outcome_model_path)
            if preview_outcome_model_path is not None
            else None
        )
        self._preview_outcome_predictor = None
        self._battle_plans: dict[str, BattlePlanState] = {}
        self.use_learned_teampreview = use_learned_teampreview
        self.use_outcome_teampreview = use_outcome_teampreview
        self.switch_model_path = (
            Path(switch_model_path) if switch_model_path is not None else None
        )
        self._switch_predictor: SwitchPredictor | None = None
        self._switch_prediction_cache: dict[
            tuple[Any, ...], tuple[SwitchPrediction, SwitchPrediction]
        ] = {}
        self.move_model_path = (
            Path(move_model_path) if move_model_path is not None else None
        )
        self._move_predictor: MovePredictor | None = None
        self._move_prediction_cache: dict[
            tuple[Any, ...], tuple[MovePrediction, MovePrediction]
        ] = {}
        self.residual_ranker_path = (
            Path(residual_ranker_path) if residual_ranker_path is not None else None
        )
        self._residual_ranker = None
        self.use_opponent_reranker = use_opponent_reranker
        self.use_tempo_reranker = use_tempo_reranker
        self.forced_lead_species = (
            tuple(to_id_str(species) for species in forced_lead_species)
            if forced_lead_species is not None
            else None
        )
        # Bring-selection experiment mechanism (works for any species on any
        # roster): the named species' team slots are masked out of BOTH preview
        # picks, so the policy drafts around them. Stands down when the roster
        # lacks the species or benching would leave fewer than four picks.
        self.forced_bench_species = (
            tuple(to_id_str(species) for species in forced_bench_species)
            if forced_bench_species
            else None
        )
        self.decision_log_path = (
            Path(decision_log_path) if decision_log_path is not None else None
        )
        self.team_sheet_wait_timeout = team_sheet_wait_timeout
        self.exact_team_path = (
            Path(exact_team_path) if exact_team_path is not None else None
        )
        self.outcome_value_path = (
            Path(outcome_value_path) if outcome_value_path is not None else None
        )
        self._outcome_evaluator = None
        self.exact_search_config = exact_search_config
        self.exact_max_determinizations = min(
            8, max(1, int(exact_max_determinizations))
        )
        self.exact_search_determinizations = min(
            self.exact_max_determinizations, max(1, int(exact_search_determinizations))
        )
        self.exact_min_deep_coverage = float(exact_min_deep_coverage)
        self.exact_preview_search = bool(exact_preview_search)
        self.exact_preview_budget = float(exact_preview_budget)
        self.exact_preview_determinizations = min(
            8, max(1, int(exact_preview_determinizations))
        )
        self.exact_selective_search = bool(exact_selective_search)
        self.exact_enable_ponder = bool(exact_enable_ponder)
        self.exact_ponder_config = exact_ponder_config
        self.enable_search = (
            PolicyPlayer.use_search if enable_search is None else bool(enable_search)
        )
        self._exact_sessions: dict[str, Any] = {}
        self._open_sheet_battles: set[str] = set()
        self._preview_requests_submitted: set[tuple[str, int | None]] = set()
        self._exact_preview_decisions: dict[str, str] = {}
        self._team_sheet_wait_tasks: dict[
            tuple[str, int | None], asyncio.Task[None]
        ] = {}

    @staticmethod
    def _preview_request_key(battle: AbstractBattle) -> tuple[str, int | None]:
        """Identify one Team Preview request so OTS and timeout cannot submit twice."""
        rqid = battle.last_request.get("rqid")
        return battle.battle_tag, rqid if isinstance(rqid, int) else None

    async def _handle_battle_request(
        self, battle: AbstractBattle, maybe_default_order: bool = False
    ):
        """Deduplicate Team Preview orders when the OTS fallback races a reply."""
        key: tuple[str, int | None] | None = None
        was_teampreview = bool(battle.teampreview)
        if was_teampreview and not maybe_default_order:
            key = self._preview_request_key(battle)
            if key in self._preview_requests_submitted:
                return
            # Mark before awaiting the websocket send. A late |showteam| message and
            # the timeout task can otherwise enter this method in the same event-loop
            # tick and submit two /team orders.
            self._preview_requests_submitted.add(key)
        try:
            await super()._handle_battle_request(battle, maybe_default_order)
            if not was_teampreview:
                session = self._exact_sessions.get(battle.battle_tag)
                if session is not None:
                    session.start_pending_ponder()
        except Exception:
            if key is not None:
                self._preview_requests_submitted.discard(key)
            raise

    async def _team_sheet_wait_fallback(
        self, battle: AbstractBattle, key: tuple[str, int | None]
    ) -> None:
        """Proceed with Team Preview if the opponent leaves the OTS prompt open."""
        assert self.team_sheet_wait_timeout is not None
        await asyncio.sleep(self.team_sheet_wait_timeout)
        if key in self._preview_requests_submitted:
            return
        if not battle.teampreview or self._preview_request_key(battle) != key:
            return
        PolicyPlayer.guard_fire_counts["team_sheet_wait_timeout"] += 1
        await self._handle_battle_request(battle)

    async def _handle_battle_message(self, split_messages: list[list[str]]):
        """Add an opening-only timeout to poke-env's Open Team Sheets handshake."""
        battle_tag = split_messages[0][0].removeprefix(">")
        saw_opponent_sheet = any(
            len(message) >= 3 and message[1] == "showteam" and message[2] == "p2"
            for message in split_messages[1:]
        )
        battle_ended = any(
            len(message) >= 2 and message[1] in {"win", "tie"}
            for message in split_messages[1:]
        )
        # A local/server chunk can contain both |showteam| and the first actionable
        # request. Record OTS before the parent handles that request, otherwise the
        # exact session is created in hidden-sheet mode for an open-sheet battle.
        if saw_opponent_sheet:
            self._open_sheet_battles.add(battle_tag)
        await super()._handle_battle_message(split_messages)
        if battle_ended:
            session = self._exact_sessions.pop(battle_tag, None)
            if session is not None:
                session.close()
            self._open_sheet_battles.discard(battle_tag)
        timeout = self.team_sheet_wait_timeout
        if timeout is None or not self.accept_open_team_sheet or self.format_is_bestof:
            return

        saw_preview_request = False
        for split_message in split_messages[1:]:
            if len(split_message) >= 3 and split_message[1] == "request":
                try:
                    request = json.loads(split_message[2]) if split_message[2] else {}
                except json.JSONDecodeError:
                    continue
                if request.get("teamPreview"):
                    saw_preview_request = True
                    break
        if not saw_preview_request:
            return

        battle = self._battles.get(battle_tag)
        if battle is None:
            return
        key = self._preview_request_key(battle)
        if (
            key in self._preview_requests_submitted
            or key in self._team_sheet_wait_tasks
        ):
            return
        task = asyncio.create_task(self._team_sheet_wait_fallback(battle, key))
        self._team_sheet_wait_tasks[key] = task
        task.add_done_callback(
            lambda _task, k=key: self._team_sheet_wait_tasks.pop(k, None)
        )

    async def _handle_challenge_request(self, split_message: list[str]):
        """Accept challenge requests, optionally for any recognized format."""
        if not self._accept_all_formats:
            return await super()._handle_challenge_request(split_message)
        challenging_player = split_message[2].strip()
        if challenging_player != self.username:
            if len(split_message) >= 6:
                fmt = split_message[5]
                if fmt.startswith("gen9championsvgc"):
                    await self._challenge_queue.put((challenging_player, fmt))

    async def _update_challenges(self, split_message: list[str]):
        """Queue challenges, optionally accepting any recognized format."""
        if not self._accept_all_formats:
            return await super()._update_challenges(split_message)
        challenges = json.loads(split_message[2]).get("challengesFrom", {})
        for user, fmt in challenges.items():
            if fmt.startswith("gen9championsvgc"):
                await self._challenge_queue.put((user, fmt))

    async def _accept_challenges(
        self,
        opponent: str | list[str] | None,
        n_challenges: int,
        packed_team: str | None,
    ):
        """Accept challenges, setting format and team reg before each."""
        if not self._accept_all_formats:
            return await super()._accept_challenges(opponent, n_challenges, packed_team)
        if opponent:
            if isinstance(opponent, list):
                opponent = [to_id_str(o) for o in opponent]
            else:
                opponent = to_id_str(opponent)
        await self.ps_client.logged_in.wait()

        for _ in range(n_challenges):
            while True:
                username, fmt = await self._challenge_queue.get()
                username = to_id_str(username)
                if (
                    (opponent is None)
                    or (opponent == username)
                    or (isinstance(opponent, list) and (username in opponent))
                ):
                    self._format = fmt
                    if (
                        isinstance(self._team, RandomTeamBuilder)
                        and self._team.available_regs is not None
                    ):
                        self._team.current_reg = get_reg_from_format(fmt)
                    team = packed_team or self.next_team
                    await self.ps_client.accept_challenge(username, team)
                    await self._battle_semaphore.acquire()
                    break
        await self._battle_count_queue.join()

    async def _create_battle(self, split_message: list[str]):
        """Create a battle, accepting any recognized format if configured."""
        if not self._accept_all_formats:
            battle = await super()._create_battle(split_message)
        elif split_message[1].startswith("gen9championsvgc"):
            saved = self.format
            self._format = split_message[1]
            try:
                battle = await super()._create_battle(split_message)
            finally:
                self._format = saved
        else:
            battle = await super()._create_battle(split_message)
        if self.invitee is not None and "bo3" not in self.format:
            await self.ps_client.send_message(
                f"/invite {self.invitee}", battle.battle_tag
            )
        return battle

    async def _handle_bestof_message(self, split_messages):
        """Handle best-of series messages, inviting spectator to the lobby."""
        if self.invitee is not None:
            game_tag = split_messages[0][0][1:]  # strip >
            for split_message in split_messages[1:]:
                if len(split_message) >= 2 and split_message[1] == "init":
                    await self.ps_client.send_message(
                        f"/invite {self.invitee}", room=game_tag
                    )
                    break
        await super()._handle_bestof_message(split_messages)

    def set_policy(self, policy_file: str | Path, device: torch.device):
        """
        Load or update the policy from a checkpoint file.

        Args:
            policy_file: Path to the saved PPO checkpoint.
            device: PyTorch device for model placement.
        """
        # poke-env's listener already runs on its own thread by the time this is
        # called, so battle requests can arrive while the checkpoint (and the
        # outcome evaluator) are still loading. Decisions wait on this flag in
        # _repair_policy instead of failing.
        self._policy_loading = True
        try:
            self._set_policy_impl(policy_file, device)
        finally:
            self._policy_loading = False

    def _set_policy_impl(self, policy_file: str | Path, device: torch.device):
        if self.policy is None:
            self.policy = PPO.load(policy_file, device=device).policy
        else:
            # Bypass SB3's leaky set_parameters - load state dict directly from zip
            with zipfile.ZipFile(policy_file, "r") as zf:
                with zf.open("policy.pth") as f:
                    state_dict = torch.load(
                        io.BytesIO(f.read()), map_location=device, weights_only=True
                    )
            self.policy.load_state_dict(state_dict)
        if self.outcome_value_path is not None and self._outcome_evaluator is None:
            from vgc_bench.src.outcome_value import OutcomeValueEvaluator

            self._outcome_evaluator = OutcomeValueEvaluator.load(
                self.outcome_value_path, device=device, mechanics_weight=0.10
            )

    _policy_type_reported = False
    # Seconds a decision may wait for set_policy to finish loading. Ladder
    # entry points should keep this under the 10s turn clock.
    policy_wait_s: float = 30.0

    def _repair_policy(self) -> None:
        """Make ``self.policy`` usable for per-decision inference, or count why not.

        A bare ``assert isinstance(self.policy, MaskedActorCriticPolicy)`` in a
        decision path stalls the battle forever -- poke-env swallows the exception
        and never sends another order. Observed 2026-09-05 in the search re-gate
        (learned opponent + live search) on a forced-switch request. If something
        wrapped the policy, unwrap it in place; otherwise report the type once,
        count the event, and let the caller play a safe default order.
        """
        if isinstance(self.policy, MaskedActorCriticPolicy):
            return
        if self.policy is None and getattr(self, "_policy_loading", False):
            # Construction-order race (observed 2026-09-05 in the search re-gate):
            # the listener thread requested a decision while set_policy was still
            # inside PPO.load. Wait briefly rather than play a default order.
            deadline = time.monotonic() + PolicyPlayer.policy_wait_s
            while self.policy is None and time.monotonic() < deadline:
                time.sleep(0.05)
            if isinstance(self.policy, MaskedActorCriticPolicy):
                PolicyPlayer.guard_fire_counts["policy_waited"] += 1
                return
        inner = getattr(self.policy, "policy", None)
        if isinstance(inner, MaskedActorCriticPolicy):
            self.policy = inner
            PolicyPlayer.guard_fire_counts["policy_unwrapped"] += 1
            return
        PolicyPlayer.guard_fire_counts["policy_unavailable"] += 1
        if not PolicyPlayer._policy_type_reported:
            PolicyPlayer._policy_type_reported = True
            print(
                f"WARNING: policy unavailable for decisions: {type(self.policy)!r} "
                f"({getattr(self, 'username', '?')}); playing default orders",
                flush=True,
            )

    @staticmethod
    def _preview_roster(team: dict[str, Pokemon]) -> tuple[str, ...]:
        return tuple(to_id_str(mon.base_species) for mon in team.values())

    def _learned_teampreview(self, battle: DoubleBattle) -> str | None:
        """Choose a coherent lead-two/bring-four plan from replay-trained priors."""
        if self.preview_model_path is None:
            return None
        try:
            if self._preview_predictor is None:
                self._preview_predictor = PreviewPredictor.load(self.preview_model_path)
            ours = self._preview_roster(battle.team)
            theirs = self._preview_roster(battle.opponent_team)
            if len(ours) != 6 or len(theirs) != 6:
                raise ValueError(
                    f"preview rosters incomplete: ours={len(ours)} theirs={len(theirs)}"
                )
            their_plans = self._preview_predictor.predict_plans(theirs, ours, top_k=90)
            self._battle_plans[battle.battle_tag] = BattlePlanState(
                own_plan=None, opponent_belief=OpponentBelief(theirs, their_plans)
            )
            if not self.use_learned_teampreview:
                # Production only wants the opponent belief; our own plan is chosen
                # by the champion policy, so computing it here would be waste.
                return None
            own_plan = self._preview_predictor.predict_plans(ours, theirs, top_k=1)[0]
            self._battle_plans[battle.battle_tag].own_plan = own_plan
            selected = set(own_plan.bring_indices)
            for index, pokemon in enumerate(battle.team.values()):
                pokemon._selected_in_teampreview = index in selected
            order = plan_to_showdown_order(own_plan)
            PolicyPlayer.guard_fire_counts["learned_preview"] += 1
            return "/team " + "".join(str(index) for index in order)
        except Exception as exc:
            # Preview must still complete if a checkpoint is missing or a roster uses
            # an unexpected form. Fall back to the already-tested policy preview.
            PolicyPlayer.guard_fire_counts[
                f"learned_preview_error:{type(exc).__name__}"
            ] += 1
            return None

    def _champion_preview_order(
        self, battle: DoubleBattle
    ) -> tuple[int, int, int, int] | None:
        """The champion policy's own two-stage preview pick, without lasting mutation.

        Replays the exact fallback preview path (forced leads, immunity mask,
        forced bench, guards) so the exact preview planner can guarantee the
        champion's plan a candidate slot. Team-preview selection flags are restored
        afterwards; the planner or the fallback path sets them for real later.
        """
        assert self.policy is not None
        if not self.policy.choose_on_teampreview:
            return None
        team = list(battle.team.values())
        saved = [bool(pokemon._selected_in_teampreview) for pokemon in team]
        # This is a what-would-the-champion-do probe, not a played decision:
        # suppress the audit rows _guarded_action would write, so turn-0 analyzers
        # keep seeing exactly the decisions that were actually submitted.
        saved_log_path = self.decision_log_path
        self.decision_log_path = None
        try:
            action1 = self._forced_lead_actions(battle)
            if action1 is None:
                # The base synchronous implementation explicitly: the batched
                # subclass overrides choose_move with a coroutine, but its guarded
                # preview path is this same code.
                order1 = PolicyPlayer.choose_move(self, battle)
                assert not isinstance(order1, Awaitable)
                action1 = DoublesEnv.order_to_action(order1, battle)
            team[int(action1[0]) - 1]._selected_in_teampreview = True
            team[int(action1[1]) - 1]._selected_in_teampreview = True
            order2 = PolicyPlayer.choose_move(self, battle)
            assert not isinstance(order2, Awaitable)
            action2 = DoublesEnv.order_to_action(order2, battle)
            return (
                int(action1[0]),
                int(action1[1]),
                int(action2[0]),
                int(action2[1]),
            )
        finally:
            self.decision_log_path = saved_log_path
            for pokemon, flag in zip(team, saved):
                pokemon._selected_in_teampreview = flag

    def _planned_teampreview(self, battle: DoubleBattle) -> str | None:
        """Choose a coherent matchup plan by exact first-turn simulation."""
        if not self.exact_preview_search:
            return None
        cached = self._exact_preview_decisions.get(battle.battle_tag)
        if cached is not None:
            return cached
        if (
            self.policy is None
            or self.exact_team_path is None
            or self._outcome_evaluator is None
            or self.preview_model_path is None
        ):
            PolicyPlayer.guard_fire_counts["exact_preview_missing_dependency"] += 1
            return None
        try:
            from vgc_bench.src.live_preview import LivePreviewPlanner

            if self._preview_predictor is None:
                self._preview_predictor = PreviewPredictor.load(
                    self.preview_model_path, device=self.policy.device
                )
            champion_order = None
            try:
                champion_order = self._champion_preview_order(battle)
            except Exception as exc:
                # The planner still works without the injection; it just cannot
                # guarantee the champion's plan a candidate slot.
                PolicyPlayer.guard_fire_counts[
                    f"exact_preview_champion_order_error:{type(exc).__name__}"
                ] += 1
            decision = LivePreviewPlanner(
                policy=self.policy,
                our_team_text=self.exact_team_path.read_text(),
                outcome_evaluator=self._outcome_evaluator,
                preview_predictor=self._preview_predictor,
                open_sheet=battle.battle_tag in self._open_sheet_battles,
                budget_s=self.exact_preview_budget,
                determinizations=self.exact_preview_determinizations,
                champion_order=champion_order,
                policy_inference_lock=self._exact_policy_lock,
            ).choose(battle)
            if decision.truncated:
                PolicyPlayer.guard_fire_counts["exact_preview_truncated"] += 1
                return None

            order = [int(char) for char in decision.command.removeprefix("/team ")]
            selected = {index - 1 for index in order}
            for index, pokemon in enumerate(battle.team.values()):
                pokemon._selected_in_teampreview = index in selected
            ours = self._preview_roster(battle.team)
            theirs = self._preview_roster(battle.opponent_team)
            their_plans = self._preview_predictor.predict_plans(theirs, ours, top_k=90)
            bring = tuple(sorted(index - 1 for index in order))
            own_plan = PreviewPlan(
                lead_indices=(order[0] - 1, order[1] - 1),
                bring_indices=bring,
                probability=1.0,
            )
            self._battle_plans[battle.battle_tag] = BattlePlanState(
                own_plan=own_plan, opponent_belief=OpponentBelief(theirs, their_plans)
            )
            self._exact_preview_decisions[battle.battle_tag] = decision.command
            PolicyPlayer.guard_fire_counts["exact_preview"] += 1
            if decision.champion_rank == 1:
                PolicyPlayer.guard_fire_counts["exact_preview_agrees_champion"] += 1
            elif decision.champion_rank is not None:
                PolicyPlayer.guard_fire_counts["exact_preview_overrides_champion"] += 1
            if self.decision_log_path is not None:
                self.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.decision_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "battle": battle.battle_tag,
                                "turn": 0,
                                "exact_preview": {
                                    "choice": decision.showdown_choice,
                                    "command": decision.command,
                                    "elapsed_s": decision.elapsed_s,
                                    "nodes": decision.nodes,
                                    "truncated": decision.truncated,
                                    "score": decision.score,
                                    "open_sheet": decision.open_sheet,
                                    "worlds_requested": decision.worlds_requested,
                                    "worlds_clean": decision.worlds_clean,
                                    "worlds_failed": decision.worlds_failed,
                                    "champion_choice": decision.champion_choice,
                                    "champion_rank": decision.champion_rank,
                                    "override_margin": decision.override_margin,
                                    "rankings": list(decision.rankings),
                                },
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
            return decision.command
        except Exception as exc:
            PolicyPlayer.guard_fire_counts[
                f"exact_preview_error:{type(exc).__name__}"
            ] += 1
            return None

    def _outcome_teampreview(self, battle: DoubleBattle) -> str | None:
        """Select hidden-sheet plans using terminal self-play outcomes.

        The outcome model improved both hidden-sheet validation modes, but lost
        three percentage points with open sheets.  Open sheets therefore fall
        through to the checkpoint's original Team Preview policy, which can use
        the fully revealed roster without this model's hidden-set assumptions.
        """
        if not self.use_outcome_teampreview or self.preview_outcome_model_path is None:
            return None
        if battle.battle_tag in self._open_sheet_battles:
            PolicyPlayer.guard_fire_counts["outcome_preview_open_sheet_fallback"] += 1
            return None
        try:
            from vgc_bench.src.preview_outcome import PreviewOutcomePredictor

            if self._preview_predictor is None:
                if self.preview_model_path is None:
                    raise ValueError("outcome preview requires the candidate prior")
                self._preview_predictor = PreviewPredictor.load(
                    self.preview_model_path, device=self.policy.device
                )
            if self._preview_outcome_predictor is None:
                self._preview_outcome_predictor = PreviewOutcomePredictor.load(
                    self.preview_outcome_model_path, device=self.policy.device
                )
            ours = self._preview_roster(battle.team)
            theirs = self._preview_roster(battle.opponent_team)
            # Training explored the replay prior's first four plans. Do not let the
            # outcome network extrapolate to unsupported orders merely because they
            # receive an accidentally optimistic score.
            candidates = self._preview_predictor.predict_plans(ours, theirs, top_k=4)
            ranked = self._preview_outcome_predictor.rank(
                ours, theirs, candidates=candidates
            )
            chosen = ranked[0]
            selected = set(chosen.plan.bring_indices)
            for index, pokemon in enumerate(battle.team.values()):
                pokemon._selected_in_teampreview = index in selected
            their_plans = self._preview_predictor.predict_plans(theirs, ours, top_k=90)
            self._battle_plans[battle.battle_tag] = BattlePlanState(
                own_plan=chosen.plan,
                opponent_belief=OpponentBelief(theirs, their_plans),
            )
            order = plan_to_showdown_order(chosen.plan)
            PolicyPlayer.guard_fire_counts["outcome_preview"] += 1
            return "/team " + "".join(str(index) for index in order)
        except Exception as exc:
            PolicyPlayer.guard_fire_counts[
                f"outcome_preview_error:{type(exc).__name__}"
            ] += 1
            return None

    def _record_own_preview(
        self, battle: DoubleBattle, leads: tuple[int, int], backline: tuple[int, int]
    ) -> None:
        """Record the actual policy draft alongside the opponent-plan belief."""
        state = self._battle_plans.get(battle.battle_tag)
        if state is None:
            return
        brought = sorted(index - 1 for index in (*leads, *backline))
        state.own_plan = PreviewPlan(
            lead_indices=(leads[0] - 1, leads[1] - 1),
            bring_indices=(brought[0], brought[1], brought[2], brought[3]),
            probability=1.0,
        )

    def _shadow_preview_record(
        self, battle: DoubleBattle, leads: tuple[int, int], backline: tuple[int, int]
    ) -> None:
        """Append one consolidated Team Preview record to the decision log.

        Pure instrumentation for the preview workstream: what we chose, what the
        replay-trained predictor would rank for both sides, and the opponent's
        Trick Room likelihood. Never raises and never changes the choice. The
        per-candidate policy probabilities are already in the two adjacent
        turn-0 audit records written by ``_guarded_action``.
        """
        if self.decision_log_path is None:
            return
        try:
            from vgc_bench.src.preview_rules import (
                species_trick_room_rates,
                trick_room_probability,
            )

            ours = self._preview_roster(battle.team)
            theirs = self._preview_roster(battle.opponent_team)
            our_mons = list(battle.team.values())
            shadow: dict = {
                "open_sheet": battle.battle_tag in self._open_sheet_battles,
                "roster_ours": list(ours),
                "roster_theirs": list(theirs),
                "lead_slots": [int(leads[0]), int(leads[1])],
                "lead_species": [
                    to_id_str(our_mons[leads[0] - 1].base_species),
                    to_id_str(our_mons[leads[1] - 1].base_species),
                ],
                "bring_species": sorted(
                    to_id_str(our_mons[slot - 1].base_species)
                    for slot in (*leads, *backline)
                ),
                "opponent_trick_room_rates": species_trick_room_rates(theirs),
                "opponent_trick_room_probability": round(
                    trick_room_probability(theirs), 4
                ),
            }
            if self.preview_model_path is not None:
                try:
                    if self._preview_predictor is None:
                        self._preview_predictor = PreviewPredictor.load(
                            self.preview_model_path
                        )
                    predictor = self._preview_predictor

                    def plans(us, them):
                        return [
                            {
                                "leads": [int(i) for i in plan.lead_indices],
                                "bring": [int(i) for i in plan.bring_indices],
                                "probability": round(float(plan.probability), 4),
                            }
                            for plan in predictor.predict_plans(us, them, top_k=5)
                        ]

                    shadow["predictor_top5_ours"] = plans(ours, theirs)
                    shadow["predictor_top5_theirs"] = plans(theirs, ours)
                except Exception as exc:
                    shadow["predictor_error"] = type(exc).__name__
            self.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.decision_log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "battle": battle.battle_tag,
                            "turn": 0,
                            "preview_shadow": shadow,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        except Exception as exc:
            PolicyPlayer.guard_fire_counts[
                f"preview_shadow_error:{type(exc).__name__}"
            ] += 1

    def _apply_forced_bench(self, battle: DoubleBattle, mask: np.ndarray) -> np.ndarray:
        """Zero the benched species' team-slot actions in a PREVIEW mask.

        Preview actions are one-based team slots on both action heads. Never
        raises: an absent species or an over-restricted draft (fewer than four
        pickable slots) stands the experiment down with a counter instead of
        breaking preview.
        """
        assert self.forced_bench_species is not None
        slots = [
            index
            for index, mon in enumerate(battle.team.values(), start=1)
            if to_id_str(mon.base_species) in self.forced_bench_species
        ]
        if not slots:
            PolicyPlayer.guard_fire_counts["forced_bench_missing"] += 1
            return mask
        if len(battle.team) - len(slots) < 4:
            PolicyPlayer.guard_fire_counts["forced_bench_too_restrictive"] += 1
            return mask
        adjusted = mask.copy()
        half = adjusted.shape[-1] // 2
        for slot in slots:
            adjusted[slot] = 0
            adjusted[half + slot] = 0
        PolicyPlayer.guard_fire_counts["forced_bench"] += 1
        return adjusted

    def _forced_lead_actions(self, battle: DoubleBattle) -> np.ndarray | None:
        """Resolve an optional fixed lead pair to Showdown's one-based team slots."""
        if self.forced_lead_species is None:
            return None
        wanted = self.forced_lead_species
        if len(wanted) != 2 or wanted[0] == wanted[1]:
            return None
        by_species = {
            to_id_str(mon.base_species): index
            for index, mon in enumerate(battle.team.values(), start=1)
        }
        if any(species not in by_species for species in wanted):
            PolicyPlayer.guard_fire_counts["forced_lead_missing"] += 1
            return None
        PolicyPlayer.guard_fire_counts["forced_lead"] += 1
        return np.array([by_species[wanted[0]], by_species[wanted[1]]], dtype=np.int64)

    def _update_battle_plan(self, battle: DoubleBattle) -> None:
        """Condition the opponent's predicted plan on every observed switch-in."""
        state = self._battle_plans.get(battle.battle_tag)
        if state is None:
            return
        active = [
            to_id_str(mon.base_species)
            for mon in battle.opponent_active_pokemon
            if mon is not None
        ]
        if active:
            state.observe_active(active)

    def battle_plan(self, battle: DoubleBattle) -> BattlePlanState | None:
        """Expose the current plan belief to future turn search and diagnostics."""
        self._update_battle_plan(battle)
        return self._battle_plans.get(battle.battle_tag)

    def opponent_switch_predictions(
        self, battle: DoubleBattle
    ) -> tuple[SwitchPrediction, SwitchPrediction] | None:
        """Predict each opposing slot's switch probability and likely replacement."""
        if self.switch_model_path is None:
            return None
        state = self.battle_plan(battle)
        if state is None:
            return None
        our_active = [mon for mon in battle.active_pokemon if mon is not None]
        their_active = [
            mon for mon in battle.opponent_active_pokemon if mon is not None
        ]
        if len(our_active) != 2 or len(their_active) != 2:
            return None
        ours = self._preview_roster(battle.team)
        theirs = state.opponent_belief.roster
        active_names = tuple(to_id_str(mon.base_species) for mon in their_active)
        our_active_names = tuple(to_id_str(mon.base_species) for mon in our_active)
        hp = tuple(
            float(mon.current_hp_fraction or 0.0)
            for mon in (*their_active, *our_active)
        )
        key = (battle.battle_tag, battle.turn, active_names, our_active_names, hp)
        cached = self._switch_prediction_cache.get(key)
        if cached is not None:
            return cached
        try:
            if self._switch_predictor is None:
                self._switch_predictor = SwitchPredictor.load(self.switch_model_path)
            marginals = state.opponent_belief.bring_marginals()
            predictions = tuple(
                self._switch_predictor.predict(
                    theirs,
                    ours,
                    active_names,
                    our_active_names,
                    hp,
                    actor_slot=slot,
                    turn=battle.turn,
                    bring_marginals=marginals,
                )
                for slot in range(2)
            )
            assert len(predictions) == 2
            result = (predictions[0], predictions[1])
            if len(self._switch_prediction_cache) > 256:
                self._switch_prediction_cache.clear()
            self._switch_prediction_cache[key] = result
            return result
        except Exception as exc:
            PolicyPlayer.guard_fire_counts[
                f"switch_prediction_error:{type(exc).__name__}"
            ] += 1
            return None

    @staticmethod
    def _reliability_floor_cap() -> float:
        """VGC_PRIOR_RELIABILITY_FLOOR, defaulting to 0.0 (historical behavior)."""
        try:
            return float(os.environ.get("VGC_PRIOR_RELIABILITY_FLOOR", "0"))
        except ValueError:
            return 0.0

    @staticmethod
    def _floored_reliability(
        revealed_fraction: float, prior: dict[str, Any], floor_cap: float
    ) -> float:
        """Reliability for a move pool: revealed evidence, floored by the prior.

        Revealed moves stay the primary signal (fraction of four slots seen).
        When the floor is enabled and the pool was prior-filled, the
        evidence-conditioned set posterior may lift reliability, bounded by the
        cap -- so a 97%-one-set species lends bounded evidence on turn 1 while
        a 30-way-split species lends almost none. floor_cap=0 reproduces the
        historical revealed-moves-only behavior exactly.
        """
        if floor_cap <= 0.0 or not prior:
            return revealed_fraction
        posterior = float(prior.get("prob") or 0.0)
        return max(revealed_fraction, min(floor_cap, posterior))

    def opponent_move_predictions(
        self, battle: DoubleBattle
    ) -> tuple[MovePrediction, MovePrediction] | None:
        """Rank each opposing active's plausible moves and targets from replay data."""
        if self.move_model_path is None:
            return None
        state = self.battle_plan(battle)
        if state is None:
            return None
        our_active = [mon for mon in battle.active_pokemon if mon is not None]
        their_active = [
            mon for mon in battle.opponent_active_pokemon if mon is not None
        ]
        if len(our_active) != 2 or len(their_active) != 2:
            return None
        ours = self._preview_roster(battle.team)
        theirs = state.opponent_belief.roster
        active_names = tuple(to_id_str(mon.base_species) for mon in their_active)
        our_active_names = tuple(to_id_str(mon.base_species) for mon in our_active)
        hp = tuple(
            float(mon.current_hp_fraction or 0.0)
            for mon in (*their_active, *our_active)
        )
        # Revealed moves alone left this layer inert on turns 1-2 of hidden-sheet
        # games -- exactly the turns that decide a 6-turn format. With the floor
        # enabled, an evidence-conditioned set posterior can lend BOUNDED
        # reliability to a prior-filled pool: capped by VGC_PRIOR_RELIABILITY_FLOOR
        # (default 0.0 = exact historical behavior) and scaled by how concentrated
        # the species' surviving sets actually are. Reliability only scales soft
        # reranker evidence; the >=0.999 switch-evidence gate is untouched.
        # Per-instance override lets an A/B run give each arm its own floor in
        # one process; the env var is the production/global path.
        instance_floor = getattr(self, "reliability_floor", None)
        floor_cap = (
            float(instance_floor)
            if instance_floor is not None
            else PolicyPlayer._reliability_floor_cap()
        )
        move_pools = []
        move_reliability = []
        for mon in their_active:
            move_ids = list(mon.moves)
            revealed_fraction = min(1.0, len(move_ids) / 4.0)
            prior: dict[str, Any] = {}
            if len(move_ids) < 4 and PolicyPlayer.moveset_prior_enabled():
                prior = PolicyPlayer._moveset_prior(mon) or {}
                move_ids.extend(
                    move_id
                    for move_id in prior.get("moves", [])
                    if move_id not in move_ids
                )
            move_reliability.append(
                PolicyPlayer._floored_reliability(revealed_fraction, prior, floor_cap)
            )
            move_pools.append(tuple(move_ids[:4]))
        key = (
            battle.battle_tag,
            battle.turn,
            active_names,
            our_active_names,
            hp,
            tuple(move_pools),
        )
        cached = self._move_prediction_cache.get(key)
        if cached is not None:
            return cached
        try:
            if self._move_predictor is None:
                self._move_predictor = MovePredictor.load(self.move_model_path)
            raw_predictions = tuple(
                self._move_predictor.predict(
                    theirs,
                    ours,
                    active_names,
                    our_active_names,
                    hp,
                    actor_slot=slot,
                    turn=battle.turn,
                    available_moves=move_pools[slot],
                )
                for slot in range(2)
            )
            predictions = tuple(
                MovePrediction(
                    prediction.moves,
                    prediction.targets,
                    prediction.actions,
                    reliability=move_reliability[slot],
                )
                for slot, prediction in enumerate(raw_predictions)
            )
            assert len(predictions) == 2
            result = (predictions[0], predictions[1])
            if len(self._move_prediction_cache) > 256:
                self._move_prediction_cache.clear()
            self._move_prediction_cache[key] = result
            return result
        except Exception as exc:
            PolicyPlayer.guard_fire_counts[
                f"move_prediction_error:{type(exc).__name__}"
            ] += 1
            return None

    def choose_move(
        self, battle: AbstractBattle
    ) -> BattleOrder | Awaitable[BattleOrder]:
        """
        Choose the next move using the neural network policy.

        Args:
            battle: Current battle state.

        Returns:
            The chosen battle order.
        """
        assert isinstance(battle, DoubleBattle)
        if not isinstance(self.policy, MaskedActorCriticPolicy):
            self._repair_policy()
        if not isinstance(self.policy, MaskedActorCriticPolicy):
            return DefaultBattleOrder()
        if battle._wait:
            return DefaultBattleOrder()
        self._update_battle_plan(battle)
        obs = self.embed_battle(battle, fake_rating=2000)
        mask = np.array(DoublesEnv.get_action_mask(battle))
        if PolicyPlayer.mask_immunities:
            mask = PolicyPlayer.mask_immune_actions(battle, mask)
        if battle.teampreview and self.forced_bench_species:
            mask = self._apply_forced_bench(battle, mask)
        with torch.no_grad():
            obs_dict = {
                "observation": torch.as_tensor(
                    obs, device=self.policy.device
                ).unsqueeze(0),
                "action_mask": torch.as_tensor(
                    mask, device=self.policy.device
                ).unsqueeze(0),
            }
            if (
                PolicyPlayer.use_knowledge_guards
                or self.enable_search
                or self.residual_ranker_path is not None
                or getattr(self, "use_opponent_reranker", False)
                or getattr(self, "use_tempo_reranker", False)
            ):
                # The TENSOR, not the numpy `mask` above: build_candidates feeds this
                # straight into policy.get_dist_from_logits. Passing the array raised
                # on every single decision, and because _guarded_action catches
                # everything and falls back to plain sampling, the ladder ran with the
                # guards and search silently disabled -- 465 errors in 465 decisions,
                # visible only once the failure counters were printed.
                action = self._guarded_action(battle, obs_dict, obs_dict["action_mask"])
            else:
                action, _, _ = self.policy.forward(
                    obs_dict, deterministic=self.deterministic
                )
                action = action.cpu().numpy()[0]
        return DoublesEnv.action_to_order(action, battle)

    def _live_exact_session(self, battle: DoubleBattle):
        """Create or return the public-state exact-search session for this battle."""
        from vgc_bench.src.live_exact import LiveExactSession

        session = self._exact_sessions.get(battle.battle_tag)
        if session is not None:
            return session
        if self.exact_team_path is None or self.outcome_value_path is None:
            raise ValueError("live exact search requires team and outcome checkpoints")
        assert self.policy is not None
        if self.residual_ranker_path is not None and self._residual_ranker is None:
            from vgc_bench.src.residual_ranker import ResidualJointRanker

            self._residual_ranker = ResidualJointRanker.load(
                self.residual_ranker_path, self.policy.device
            )
        # Trigger lazy loading so the exact opponent prior uses the same predictors
        # as the fast reranker. A missing model remains a safe uniform component.
        self.opponent_move_predictions(battle)
        self.opponent_switch_predictions(battle)
        session = LiveExactSession(
            battle_tag=battle.battle_tag,
            policy=self.policy,
            our_team_text=self.exact_team_path.read_text(),
            formatid=self.format,
            open_sheet=battle.battle_tag in self._open_sheet_battles,
            outcome_value_path=self.outcome_value_path,
            outcome_evaluator=self._outcome_evaluator,
            residual_ranker=self._residual_ranker,
            preview_predictor=self._preview_predictor,
            move_predictor=self._move_predictor,
            switch_predictor=self._switch_predictor,
            device=str(self.policy.device),
            config=self.exact_search_config,
            max_determinizations=self.exact_max_determinizations,
            search_determinizations=self.exact_search_determinizations,
            min_deep_coverage=self.exact_min_deep_coverage,
            selective_search=self.exact_selective_search,
            enable_ponder=self.exact_enable_ponder,
            ponder_config=self.exact_ponder_config,
            policy_inference_lock=self._exact_policy_lock,
        )
        self._exact_sessions[battle.battle_tag] = session
        return session

    def _exact_search_action(
        self, battle: DoubleBattle
    ) -> npt.NDArray[np.int64] | None:
        """Run bounded exact planning and preserve a complete decision audit."""
        from vgc_bench.src import search as _search
        from vgc_bench.src.live_exact import append_exact_audit

        session = self._live_exact_session(battle)
        started = time.monotonic()
        result = session.plan(battle)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        _search._record_latency(elapsed_ms)
        append_exact_audit(self.decision_log_path, battle, session.audit())
        if result is None:
            PolicyPlayer.guard_fire_counts["exact_search_skipped"] += 1
            return None
        if result.actions is None:
            return None
        actions = np.asarray(result.actions, dtype=np.int64)
        session.record_actions(actions)
        PolicyPlayer.guard_fire_counts["exact_search"] += 1
        if result.fallback_reason == "reused_contingent_plan":
            PolicyPlayer.guard_fire_counts["exact_search_reused_plan"] += 1
        PolicyPlayer.guard_fire_counts["exact_search_nodes"] += result.nodes
        PolicyPlayer.guard_fire_counts[
            "exact_search_truncated" if result.truncated else "exact_search_complete"
        ] += 1
        if result.fallback_reason:
            PolicyPlayer.guard_fire_counts[
                f"exact_search_partial:{result.fallback_reason}"
            ] += 1
        return actions

    def _record_exact_fallback(
        self, battle: DoubleBattle, actions: npt.NDArray[np.int64]
    ) -> None:
        session = self._exact_sessions.get(battle.battle_tag)
        if session is None:
            return
        try:
            session.record_actions(actions)
        except Exception as exc:
            PolicyPlayer.guard_fire_counts[
                f"exact_record_error:{type(exc).__name__}"
            ] += 1

    def _guarded_action(self, battle, obs_dict, mask) -> npt.NDArray[np.int64]:
        """Rank joint action pairs, let knowledge guards reorder them, take the best.

        Mirrors Laplace's pipeline: the learned policy supplies the ranking, explicit
        knowledge permutes a prefix. Falls back to plain sampling if anything fails --
        a guard error must never cost us the turn.
        """
        from vgc_bench.src import guards as _guards

        assert self.policy is not None

        # Accept either a tensor or the numpy mask and normalise here, so a caller
        # passing the wrong one cannot disable the entire stack again. The two call
        # sites disagreed for the whole project's life and nothing caught it, because
        # the failure path below is a silent fallback to plain sampling.
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, device=self.policy.device)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)

        # Exact multi-turn search, when enabled. The public-state synchronizer is
        # parity-gated; every simulator error or timeout falls through to the repaired
        # champion plus hard guards, preserving a valid submission.
        # Team preview calls choose_move twice to select leads/backline, but there
        # are no active opponent leads to seed a turn simulator yet. Preview stays
        # with the champion (or a separately trained preview model) until genuine
        # planner preview labels clear their own promotion gate.
        if self.enable_search and not battle.teampreview:
            try:
                picked = self._exact_search_action(battle)
                if picked is not None:
                    self._maybe_report_guards()
                    return picked
            except Exception as exc:
                PolicyPlayer.guard_fire_counts[
                    f"exact_search_error:{type(exc).__name__}"
                ] += 1
                from vgc_bench.src.live_exact import append_exact_audit

                session = self._exact_sessions.get(battle.battle_tag)
                payload = (
                    session.audit()
                    if session is not None
                    else {"backend": "live-exact-showdown"}
                )
                # ``session.audit()`` retains the last successful schedule. Without
                # replacing it here, an exception on the current turn is reported as
                # another completed/reused search from the previous turn.
                payload["schedule"] = {
                    "mode": "error_fallback",
                    "reasons": ["exact_search_error"],
                    "error_type": type(exc).__name__,
                }
                payload["decision_fallback"] = {
                    "to": "champion_plus_guards",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=24),
                }
                append_exact_audit(self.decision_log_path, battle, payload)

        cands = []
        try:
            with self._exact_policy_lock:
                cands, _value = _guards.build_candidates(self.policy, obs_dict, mask)
            if not cands:
                raise ValueError("no candidates")
            if self.residual_ranker_path is not None:
                from vgc_bench.src.residual_ranker import ResidualJointRanker
                from vgc_bench.src.residual_ranker import (
                    rerank_candidates as residual_rerank,
                )

                if self._residual_ranker is None:
                    self._residual_ranker = ResidualJointRanker.load(
                        self.residual_ranker_path, self.policy.device
                    )
                with self._exact_policy_lock:
                    cands, residual_report = residual_rerank(
                        self._residual_ranker, self.policy, obs_dict, cands
                    )
                PolicyPlayer.guard_fire_counts["residual_ran"] += 1
                if residual_report.applied:
                    PolicyPlayer.guard_fire_counts["residual_applied"] += 1
                if residual_report.changed:
                    PolicyPlayer.guard_fire_counts["residual_changed_pick"] += 1
            guard_report = None
            if PolicyPlayer.use_knowledge_guards:
                cands, guard_report = _guards.apply_guards(
                    battle, cands, PolicyPlayer.guard_flags
                )
                PolicyPlayer.guard_fire_counts["guards_ran"] += 1
                if guard_report.stages:
                    PolicyPlayer.guard_fire_counts.update(guard_report.stages)
                # Separate "changed the pick" from "demoted something that was not
                # going to be picked anyway: only telemetry distinguishes an inert
                # rule from one that changes lower-ranked candidates.
                for _name, _n in guard_report.demotions.items():
                    PolicyPlayer.guard_fire_counts[f"{_name}:demoted"] += _n
            use_opponent = getattr(self, "use_opponent_reranker", False)
            use_tempo = getattr(self, "use_tempo_reranker", False)
            opponent_report = None
            if use_opponent or use_tempo:
                from vgc_bench.src.opponent_reranker import rerank_candidates

                cands, opponent_report = rerank_candidates(
                    battle,
                    cands,
                    self.opponent_move_predictions(battle),
                    self.opponent_switch_predictions(battle),
                    use_opponent=use_opponent,
                    use_tempo=use_tempo,
                )
                if opponent_report is not None:
                    if opponent_report.special_reason is not None:
                        PolicyPlayer.guard_fire_counts[
                            opponent_report.special_reason
                        ] += 1
                    if use_opponent:
                        PolicyPlayer.guard_fire_counts["opponent_reranker_ran"] += 1
                        PolicyPlayer.guard_fire_counts[
                            "opponent_reranker_candidates"
                        ] += opponent_report.evaluated
                    if use_tempo and opponent_report.speed_control is not None:
                        PolicyPlayer.guard_fire_counts["tempo_reranker_ran"] += 1
                        for name, total in opponent_report.tempo_factor_totals:
                            if total:
                                PolicyPlayer.guard_fire_counts[
                                    f"tempo_factor:{name}"
                                ] += 1
                        if (
                            opponent_report.tempo_score_before
                            != opponent_report.tempo_score_after
                        ):
                            PolicyPlayer.guard_fire_counts[
                                "tempo_reranker_influenced"
                            ] += 1
                    if opponent_report.changed:
                        PolicyPlayer.guard_fire_counts["combined_reranker"] += 1
                        if use_opponent:
                            PolicyPlayer.guard_fire_counts["opponent_reranker"] += 1
            legal_cands = [
                candidate
                for candidate in cands
                if self._action_pair_is_legal(battle, candidate.actions)
            ]
            removed = len(cands) - len(legal_cands)
            if removed:
                PolicyPlayer.guard_fire_counts["illegal_candidate_filtered"] += removed
            cands = legal_cands
            if not cands:
                raise ValueError("no live-legal candidate pair")
            self._audit_decision(battle, cands, opponent_report, guard_report)
            self._maybe_report_guards()
            action = np.array(cands[0].actions, dtype=np.int64)
            self._record_exact_fallback(battle, action)
            return action
        except Exception as exc:
            PolicyPlayer.guard_fire_counts[f"error:{type(exc).__name__}"] += 1
            self._maybe_report_guards()
            with self._exact_policy_lock:
                action, _, _ = self.policy.forward(
                    obs_dict, deterministic=self.deterministic
                )
            result = action.cpu().numpy()[0]
            if not self._action_pair_is_legal(battle, result):
                PolicyPlayer.guard_fire_counts["illegal_policy_fallback_replaced"] += 1
                # This should be unreachable with a correct action mask. It is still
                # better to submit poke-env's own legal fallback than crash or time
                # out if a future mask regression reaches production.
                result = DoublesEnv.order_to_action(
                    Player.choose_random_doubles_move(battle),
                    battle,
                    fake=False,
                    strict=True,
                )
            self._record_exact_fallback(battle, result)
            return result

    @staticmethod
    def _action_pair_is_legal(battle: DoubleBattle, actions: Any) -> bool:
        try:
            DoublesEnv.action_to_order(
                np.asarray(actions, dtype=np.int64), battle, fake=False, strict=True
            )
            return True
        except (AssertionError, IndexError, TypeError, ValueError):
            return False

    @staticmethod
    def _audit_order(
        battle: DoubleBattle, action: int, position: int
    ) -> dict[str, Any]:
        """Human-readable form of one encoded action for post-battle diagnosis."""
        from vgc_bench.src import guards as _guards

        decoded = _guards._decode(battle, action, position)
        selected = getattr(decoded, "order", None)
        if isinstance(selected, Move):
            return {
                "kind": "move",
                "id": selected.id,
                "target": int(getattr(decoded, "move_target", 0) or 0),
                "mega": bool(getattr(decoded, "mega", False)),
                "tera": bool(getattr(decoded, "terastallize", False)),
            }
        if isinstance(selected, Pokemon):
            return {"kind": "switch", "species": to_id_str(selected.base_species)}
        return {"kind": "pass"}

    def _audit_decision(self, battle, candidates, report, guard_report=None) -> None:
        """Append the chosen pair and its tactical evidence to a JSONL audit.

        Replays show what happened but not why the policy preferred it. This record
        preserves the top alternatives, their probabilities, every hard demotion,
        and the speed/Encore factors that changed the final ranking.
        """
        decision_log_path = getattr(self, "decision_log_path", None)
        if decision_log_path is None:
            return
        try:
            top = []
            for candidate in candidates[:8]:
                top.append(
                    {
                        "actions": list(candidate.actions),
                        "orders": [
                            self._audit_order(battle, action, position)
                            for position, action in enumerate(candidate.actions)
                        ],
                        "policy_probability": float(candidate.prob),
                        "demoted_by": candidate.demoted_by,
                    }
                )
            payload: dict[str, Any] = {
                "battle": battle.battle_tag,
                "turn": int(battle.turn),
                "chosen": top[0] if top else None,
                "candidates": top,
            }
            if guard_report is not None:
                payload["guards"] = {
                    "stages": list(guard_report.stages),
                    "demotions": dict(guard_report.demotions),
                    "vetoed": [
                        list(actions) for actions in sorted(guard_report.vetoed)
                    ],
                }
            if report is not None:
                payload["reranker"] = {
                    "before": list(report.before),
                    "after": list(report.after),
                    "special_reason": report.special_reason,
                    "opponent_score_before": report.tactical_score_before,
                    "opponent_score_after": report.tactical_score_after,
                    "tempo_score_before": report.tempo_score_before,
                    "tempo_score_after": report.tempo_score_after,
                    "tempo_factors_before": dict(report.tempo_factors_before),
                    "tempo_factors_after": dict(report.tempo_factors_after),
                    "tempo_factor_totals": dict(report.tempo_factor_totals),
                }
                speed = report.speed_control
                if speed is not None:
                    payload["speed_control"] = {
                        "trick_room_turns": speed.trick_room_turns,
                        "our_tailwind_turns": speed.our_tailwind_turns,
                        "their_tailwind_turns": speed.their_tailwind_turns,
                        "trick_room_advantage": speed.trick_room_advantage,
                        "known_comparisons": speed.known_comparisons,
                    }
            decision_log_path.parent.mkdir(parents=True, exist_ok=True)
            with decision_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        except Exception as exc:
            PolicyPlayer.guard_fire_counts[
                f"decision_audit_error:{type(exc).__name__}"
            ] += 1

    def teampreview(self, battle: AbstractBattle) -> str | Awaitable[str]:
        """
        Select Pokemon for teampreview.

        Uses random teampreview when policy-controlled teampreview is disabled.

        Args:
            battle: Current battle state during team preview.

        Returns:
            Team order string for Pokemon Showdown.
        """
        assert isinstance(battle, DoubleBattle)
        if not isinstance(self.policy, MaskedActorCriticPolicy):
            self._repair_policy()
        if not isinstance(self.policy, MaskedActorCriticPolicy):
            return self.random_teampreview(battle)
        if self.exact_preview_search:
            # Multi-world preview budgets exceed the websocket keepalive window
            # (poke-env pings every 20s and waits 20s), so the planner must not
            # block the event loop. poke-env awaits an Awaitable teampreview.
            return self._offloaded_teampreview(battle)
        return self._fallback_teampreview(battle)

    async def _offloaded_teampreview(self, battle: DoubleBattle) -> str:
        """Run the exact preview planner in a worker thread, then fall back."""
        planned = await asyncio.to_thread(self._planned_teampreview, battle)
        if planned is not None:
            return planned
        return self._fallback_teampreview(battle)

    def _fallback_teampreview(self, battle: DoubleBattle) -> str:
        """The promoted preview chain: outcome model, learned model, two-stage."""
        if not isinstance(self.policy, MaskedActorCriticPolicy):
            self._repair_policy()
        if not isinstance(self.policy, MaskedActorCriticPolicy):
            return self.random_teampreview(battle)
        outcome = self._outcome_teampreview(battle)
        if outcome is not None:
            return outcome
        learned = self._learned_teampreview(battle)
        if learned is not None:
            return learned
        if not self.policy.choose_on_teampreview:
            return self.random_teampreview(battle)
        action1 = self._forced_lead_actions(battle)
        if action1 is None:
            order1 = self.choose_move(battle)
            assert not isinstance(order1, Awaitable)
            action1 = DoublesEnv.order_to_action(order1, battle)
        list(battle.team.values())[action1[0] - 1]._selected_in_teampreview = True
        list(battle.team.values())[action1[1] - 1]._selected_in_teampreview = True
        order2 = self.choose_move(battle)
        assert not isinstance(order2, Awaitable)
        action2 = DoublesEnv.order_to_action(order2, battle)
        list(battle.team.values())[action2[0] - 1]._selected_in_teampreview = True
        list(battle.team.values())[action2[1] - 1]._selected_in_teampreview = True
        self._record_own_preview(
            battle,
            (int(action1[0]), int(action1[1])),
            (int(action2[0]), int(action2[1])),
        )
        self._shadow_preview_record(
            battle,
            (int(action1[0]), int(action1[1])),
            (int(action2[0]), int(action2[1])),
        )
        return f"/team {action1[0]}{action1[1]}{action2[0]}{action2[1]}"

    @staticmethod
    def embed_battle(
        battle: AbstractBattle, fake_rating: int | None = None
    ) -> npt.NDArray[np.float32]:
        """
        Convert a battle state to a feature vector observation.

        Creates a fixed-size numpy array encoding the full battle state including
        action masks, global effects, side conditions, and all Pokemon information.

        Args:
            battle: The battle state to embed.
            fake_rating: Optional raw rating override for the player side.
                If provided, opponent rating is masked to 0.

        Returns:
            Numpy array observation for the policy network.
        """
        assert isinstance(battle, DoubleBattle)
        glob = PolicyPlayer.embed_global(battle)
        side = PolicyPlayer.embed_side(battle, fake_rating)
        opp_fake_rating = None if fake_rating is None else 0
        opp_side = PolicyPlayer.embed_side(battle, opp_fake_rating, opp=True)
        a1, a2 = battle.active_pokemon
        o1, o2 = battle.opponent_active_pokemon
        know = PolicyPlayer._knowledge_for(battle)
        our_mons = list(battle.team.values())
        pokemons = [
            PolicyPlayer.embed_pokemon(
                p,
                i,
                from_opponent=False,
                active_a=a1 is not None and p.name == a1.name,
                active_b=a2 is not None and p.name == a2.name,
                knowledge=know.get(id(p)),
            )
            for i, p in enumerate(our_mons)
        ]
        accuracies = [PolicyPlayer.embed_move_accuracies(p, False) for p in our_mons]
        pokemons += [np.zeros(pokemon_obs_len, dtype=np.float32)] * (6 - len(pokemons))
        accuracies += [np.zeros(correct_accuracy_obs_len, dtype=np.float32)] * (
            6 - len(accuracies)
        )
        opp_mons = list(battle.opponent_team.values())
        opp_pokemons = [
            PolicyPlayer.embed_pokemon(
                p,
                i,
                from_opponent=True,
                active_a=o1 is not None and p.name == o1.name,
                active_b=o2 is not None and p.name == o2.name,
            )
            for i, p in enumerate(opp_mons)
        ]
        opp_accuracies = [PolicyPlayer.embed_move_accuracies(p, True) for p in opp_mons]
        opp_pokemons += [np.zeros(pokemon_obs_len, dtype=np.float32)] * (
            6 - len(opp_pokemons)
        )
        opp_accuracies += [np.zeros(correct_accuracy_obs_len, dtype=np.float32)] * (
            6 - len(opp_accuracies)
        )
        pres = PolicyPlayer.embed_side_presence(battle)
        opp_pres = PolicyPlayer.embed_side_presence(battle, opp=True)
        global_pres = PolicyPlayer.embed_global_presence(battle)
        return np.concatenate(
            [
                np.concatenate([glob, side, p, pres, global_pres, acc])
                for p, acc in zip(pokemons, accuracies)
            ]
            + [
                np.concatenate([glob, opp_side, p, opp_pres, global_pres, acc])
                for p, acc in zip(opp_pokemons, opp_accuracies)
            ],
            dtype=np.float32,
        )

    @staticmethod
    def embed_global(battle: DoubleBattle) -> npt.NDArray[np.float32]:
        """Embed global battle state (weather, fields, etc)."""
        weather = [
            (min(battle.turn - battle.weather[w], 8) / 8 if w in battle.weather else 0)
            for w in Weather
        ]
        fields = [
            min(battle.turn - battle.fields[f], 8) / 8 if f in battle.fields else 0
            for f in Field
        ]
        champions_format = float(
            battle.format is not None and "champions" in battle.format
        )
        teampreview = float(battle.teampreview)
        reviving = float(battle.reviving)
        commanding = float(battle.commanding)
        return np.array(
            [*weather, *fields, champions_format, teampreview, reviving, commanding],
            dtype=np.float32,
        )

    @staticmethod
    def embed_side(
        battle: DoubleBattle, fake_rating: int | None, opp: bool = False
    ) -> npt.NDArray[np.float32]:
        """
        Embed side-specific state (side conditions, gimmick availability, rating).

        Args:
            battle: Current doubles battle state.
            fake_rating: Optional raw rating override for this side.
                If None, read rating from battle player metadata.
            opp: Whether to embed the opponent side.
        """
        gims = [
            battle.can_mega_evolve[0],
            battle.can_z_move[0],
            battle.can_dynamax[0],
            battle.can_tera[0],
        ]
        opp_gims = [
            battle.opponent_used_mega_evolve,
            battle.opponent_used_z_move,
            battle.opponent_used_dynamax,
            battle.opponent_used_tera,
        ]
        side_conds = battle.opponent_side_conditions if opp else battle.side_conditions
        side_conditions = [
            (
                0
                if s not in side_conds
                else (
                    1
                    if s == SideCondition.STEALTH_ROCK
                    else (
                        side_conds[s] / 2
                        if s == SideCondition.TOXIC_SPIKES
                        else (
                            side_conds[s] / 3
                            if s == SideCondition.SPIKES
                            else min(battle.turn - side_conds[s], 8) / 8
                        )
                    )
                )
            )
            for s in SideCondition
        ]
        gims = opp_gims if opp else gims
        gimmicks = [float(g) for g in gims]
        if fake_rating is not None:
            rating = fake_rating / 2000
        else:
            player = battle.opponent_role if opp else battle.player_role
            rat = [p for p in battle._players if p["player"] == player][0].get(
                "rating", "0"
            )
            rating = int(rat or "0") / 2000
        return np.array([*side_conditions, *gimmicks, rating], dtype=np.float32)

    @staticmethod
    def embed_side_presence(battle: DoubleBattle, opp: bool = False):
        """Plain presence flags for side conditions.

        embed_side encodes them as `min(turn - set_turn, 8) / 8` -- their AGE -- so one
        set THIS turn reads 0, bit-identical to not having it. Setup moves therefore
        looked worthless both to the model and to any search asking "what if I set this
        now". These flags are appended at the TOKEN TAIL rather than folded into
        embed_side, because inserting mid-token would shift every Pokemon feature and
        silently misalign the trained weights.
        """
        conds = battle.opponent_side_conditions if opp else battle.side_conditions
        return np.array([float(s in conds) for s in SideCondition], dtype=np.float32)

    @staticmethod
    def embed_global_presence(battle: DoubleBattle) -> npt.NDArray[np.float32]:
        """Presence flags for weather and fields, including their setup turn."""
        out = np.array(
            [float(w in battle.weather) for w in Weather]
            + [float(f in battle.fields) for f in Field],
            dtype=np.float32,
        )
        assert out.shape == (global_presence_obs_len,)
        return out

    @staticmethod
    def mask_immune_actions(battle: DoubleBattle, mask: np.ndarray) -> np.ndarray:
        """Zero out single-target damaging moves that the target is immune to.

        The observation gives the move's type and the target's types as separate
        one-hots but never their interaction, so the network would have to learn the
        whole 18x18 chart from self-play. Immunities are rare, so they are learned
        last -- which is why it will happily fire a Ghost move at a Normal type for
        zero damage.

        This is a hard constraint rather than a learned one, so it needs no retraining.
        Deliberately conservative: only single-target damaging moves aimed at a foe,
        and never Tera Blast (whose type changes on terastallization).

        Action layout (poke_env doubles_env): 7..106 are moves, in 5 gimmick bands of
        20; within a band, 4 moves x 5 targets (-2,-1,0,1,2); positive targets are foes.
        """
        if PolicyPlayer._type_chart is None:
            PolicyPlayer._type_chart = GenData.from_gen(9).type_chart
        chart = PolicyPlayer._type_chart
        from vgc_bench.src.guards import resolved_foe_targets

        out = mask.copy()
        for pos in range(2):
            base = pos * act_len
            active = battle.active_pokemon[pos]
            if active is None:
                continue
            # [:4] not [-4:] -- poke-env's action decoder slices the FRONT
            # (doubles_env.py:200,342,393), so any 5th stored move (Struggle,
            # Instruct/Dancer copies, Transform) would desync this mask from the
            # action it actually blocks.
            move_list = list(active.moves.values())[:4]
            for a in range(7, act_len):
                if not out[base + a]:
                    continue
                off = a - 7
                within = off % 20
                mi, target = within // 5, (within % 5) - 2
                if target <= 0 or mi >= len(move_list):
                    continue  # not aimed at a foe, or move slot absent
                move = move_list[mi]
                if move.category == MoveCategory.STATUS or move.id == "terablast":
                    continue
                order = Player.create_order(move, move_target=target)
                targets = resolved_foe_targets(battle, order, move)
                if not targets:
                    continue
                multipliers = []
                for foe in targets:
                    if not foe.types:
                        continue
                    try:
                        multipliers.append(
                            move.type.damage_multiplier(*foe.types, type_chart=chart)
                        )
                    except Exception:
                        pass
                if multipliers and all(multiplier == 0 for multiplier in multipliers):
                    out[base + a] = 0
            # never mask away every option
            if not out[base : base + act_len].any():
                out[base : base + act_len] = mask[base : base + act_len]
        return out

    _decisions_seen: int = 0

    def _maybe_report_guards(self, every: int = 50) -> None:
        """Print the guard counters periodically during a long run.

        These counters only ever printed at the END of a run, which is how a 50-game
        ladder session completed with the guard stack raising on all 465 decisions
        before anyone could see it. A dead stack should be obvious within a minute,
        not after three hours.
        """
        PolicyPlayer._decisions_seen += 1
        if PolicyPlayer._decisions_seen % every:
            return
        counts = PolicyPlayer.guard_fire_counts
        bad = sum(v for k, v in counts.items() if "error" in k)
        summary = "  ".join(
            f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:6]
        )
        alarm = (
            "  <-- GUARDS ARE FAILING" if bad >= PolicyPlayer._decisions_seen else ""
        )
        print(
            f"[guards @ {PolicyPlayer._decisions_seen} decisions] {summary}{alarm}",
            flush=True,
        )

    @staticmethod
    def knowledge_obs_enabled() -> bool:
        """Whether the 24 knowledge floats are populated, resolved at call time.

        An explicit `PolicyPlayer.use_knowledge_obs = <bool>` wins; otherwise the
        VGC_KNOWLEDGE_OBS env var decides. Deliberately not cached at import -- see
        the comment on use_knowledge_obs for the failure this prevents.
        """
        if PolicyPlayer.use_knowledge_obs is not None:
            return PolicyPlayer.use_knowledge_obs
        return os.environ.get("VGC_KNOWLEDGE_OBS") == "1"

    @staticmethod
    def moveset_prior_enabled() -> bool:
        """Whether hidden opponent sets are imputed from competitive usage data."""
        if PolicyPlayer.use_moveset_prior is not None:
            return PolicyPlayer.use_moveset_prior
        return os.environ.get("VGC_MOVESET_PRIOR") == "1"

    @staticmethod
    def _knowledge_for(battle: DoubleBattle) -> dict[int, npt.NDArray[np.float32]]:
        """Knowledge vectors for our two actives, memoised within a battle state.

        embed_battle is called more than once on the same state, and the damage calcs
        cost ~44% of an embed, so recomputing is the single biggest avoidable expense.

        The cache key includes an HP/species fingerprint, not just (tag, turn):
        state changes mid-turn (a faint, a switch-in on a KO) and a turn-only key
        would serve stale damage numbers for the rest of that turn.

        Only our actives are computed. Benched mons aren't on the field, so their
        damage numbers are speculative, and scoping to actives is the difference
        between a 29% and a 54% throughput hit (measured).
        """
        if not PolicyPlayer.knowledge_obs_enabled():
            return {}
        # A fainted Pokemon remains in poke-env's active slot until its replacement
        # request resolves, but the damage calculator correctly rejects it as an
        # attacker. Planning visits these mid-turn forced-switch states frequently;
        # excluding fainted slots prevents thousands of invalid damage probes and
        # keeps the observation cache tied to actual battlefield actors.
        actives = [p for p in battle.active_pokemon if p is not None and not p.fainted]
        foes = [
            p for p in battle.opponent_active_pokemon if p is not None and not p.fainted
        ]
        if not actives or not foes:
            return {}
        fingerprint = (
            battle.battle_tag,
            battle.turn,
            tuple((p.species, p.current_hp_fraction) for p in actives + foes),
        )
        cached = PolicyPlayer._knowledge_cache.get(fingerprint)
        if cached is not None:
            return cached

        from vgc_bench.src import vgc_knowledge as _vk

        out = {
            id(p): np.array(
                _vk.pokemon_knowledge(battle, p, is_ours=True), dtype=np.float32
            )
            for p in actives
        }
        # Bounded: parallel self-play envs would otherwise leak one entry per turn.
        if len(PolicyPlayer._knowledge_cache) > 256:
            PolicyPlayer._knowledge_cache.clear()
        PolicyPlayer._knowledge_cache[fingerprint] = out
        return out

    @staticmethod
    def _load_priors() -> tuple[dict[str, Any], dict[str, Any]]:
        """Lazily load (joint sets, Smogon marginals)."""
        if PolicyPlayer._prior_cache is None:
            root = Path(__file__).resolve().parents[2] / "data"

            def _read(name):
                try:
                    return json.loads((root / name).read_text())
                except Exception:
                    return {}

            PolicyPlayer._prior_cache = {
                "joint": _read("joint_sets_regmb.json"),
                "marginal": _read("movesets_regmb.json"),
            }
        return PolicyPlayer._prior_cache["joint"], PolicyPlayer._prior_cache["marginal"]

    @staticmethod
    def _moveset_prior(pokemon: Pokemon) -> dict[str, Any] | None:
        """Infer this opponent's likely full set, conditioned on what it has revealed.

        Marginal usage stats say Kingambit holds Black Glasses 40% of the time
        regardless of context. But item and moves are correlated -- its Swords Dance
        set runs Black Glasses while its Low Kick set runs Focus Sash -- so once a move
        is revealed the marginal is the wrong answer.

        Here we keep only the observed sets consistent with everything revealed so far
        (moves seen, plus item/ability if those leaked) and take the most likely
        survivor. Each reveal narrows the candidates, so the guess sharpens as the
        battle goes on. Falls back to the marginal for species we have no joint data
        for, and to the unconditioned best set if observations rule everything out
        (a genuinely novel set).
        """
        if not PolicyPlayer.moveset_prior_enabled():
            return None
        joint, marginal = PolicyPlayer._load_priors()
        species = to_id_str(pokemon.base_species)
        entry = joint.get(species)
        if entry is None:
            return marginal.get(species)

        seen_moves = {m for m in pokemon.moves}
        seen_item = pokemon.item if pokemon.item not in (None, "unknown_item") else None
        seen_ability = pokemon.ability

        candidates = entry["sets"]
        consistent = [
            s
            for s in candidates
            if seen_moves.issubset(set(s["moves"]))
            and (seen_item is None or s["item"] == seen_item)
            and (seen_ability is None or s["ability"] == seen_ability)
        ]
        pool = consistent or candidates
        best = pool[0]
        # Renormalized posterior mass of the chosen set within the surviving
        # pool. A Farigiraf whose consistent sets are 97% one shape is a
        # trustworthy guess; a species split 30 ways is not. Consumers (the
        # turn-1/2 reliability floor) use this to scale how much soft evidence
        # a prior-filled move pool may contribute.
        total_mass = sum(float(s.get("prob", 0.0)) for s in pool)
        posterior = float(best.get("prob", 0.0)) / total_mass if total_mass else 0.0
        return {
            "ability": best["ability"],
            "item": best["item"],
            "moves": best["moves"],
            "prob": round(posterior, 4),
        }

    @staticmethod
    def _resolved_moves(pokemon: Pokemon, from_opponent: bool) -> list[Move]:
        """The four move slots used by both the action decoder and observation."""
        move_list = list(pokemon.moves.values())[:4]
        prior = PolicyPlayer._moveset_prior(pokemon) if from_opponent else None
        if prior and len(move_list) < 4:
            known = {move.id for move in move_list}
            for move_id in prior.get("moves", []):
                if len(move_list) >= 4:
                    break
                if move_id in known or move_id not in moves:
                    continue
                try:
                    move_list.append(Move(move_id, gen=9))
                    known.add(move_id)
                except Exception:
                    continue
        return move_list

    @staticmethod
    def embed_move_accuracies(
        pokemon: Pokemon, from_opponent: bool
    ) -> npt.NDArray[np.float32]:
        """Correct 0..1 move accuracies appended at the token tail.

        poke-env already exposes accuracy as a fraction. The original encoder divided
        it by 100 again; retaining that legacy column preserves checkpoint alignment,
        while this appended block gives the model the correctly scaled signal.
        """
        values = [
            float(move.accuracy)
            for move in PolicyPlayer._resolved_moves(pokemon, from_opponent)
        ]
        values += [0.0] * (correct_accuracy_obs_len - len(values))
        return np.array(values, dtype=np.float32)

    @staticmethod
    def embed_pokemon(
        pokemon: Pokemon,
        pos: int,
        from_opponent: bool,
        active_a: bool,
        active_b: bool,
        knowledge: npt.NDArray[np.float32] | None = None,
    ) -> npt.NDArray[np.float32]:
        """Embed a Pokemon's stats, moves, status, and effects."""
        # Invariant: one of OUR revealed Pokemon must have been drafted at teampreview.
        # It is violated by a race -- with Open Team Sheets the server can reveal our
        # side before _teampreview() has recorded its picks -- and a bare assert here
        # is NOT survivable. poke-env catches the exception inside its battle-message
        # handler, so the battle never receives an order, never finishes, and the
        # evaluation blocks on it forever. That is what stalled the 200-battle search
        # eval: 24 assertions in the opening milliseconds, then a permanent hang.
        #
        # A revealed Pokemon is on the field, so it was self-evidently drafted. Repair
        # the flag (which is what `in_draft` below reads), count it, and carry on.
        if (
            not from_opponent
            and pokemon.revealed
            and not pokemon.selected_in_teampreview
        ):
            pokemon._selected_in_teampreview = True
            PolicyPlayer.guard_fire_counts["teampreview_flag_repaired"] += 1
        # Reg M-B's Open Team Sheets rule is opt-in, so an opponent can deny it.
        # Training always had sheets, so opponent slots were populated; without them
        # the sentinels below are near-unseen inputs. When the prior is enabled we fill
        # unknown opponent fields with the most common competitive set instead, keeping
        # the observation nearer the training distribution. Shape is unchanged, so
        # existing checkpoints stay valid. Off by default -> training is unaffected.
        prior = PolicyPlayer._moveset_prior(pokemon) if from_opponent else None
        # (mostly) stable fields
        ability = pokemon.ability
        if ability is None and prior and prior.get("ability"):
            ability = prior["ability"]
        ability_id = abilities.index(ability if ability in abilities else "null")
        item = pokemon.item
        if item in (None, "unknown_item") and prior and prior.get("item"):
            item = prior["item"]
        item_id = items.index(item if item in items else "null")
        # [:4] to match poke-env's action decoder (doubles_env.py:200,342,393);
        # [-4:] silently shifted which move each action index referred to whenever
        # more than four were stored.
        move_list = PolicyPlayer._resolved_moves(pokemon, from_opponent)
        move_ids = [moves.index(move.id) for move in move_list]
        move_ids += [0] * (4 - len(move_ids))
        move_embeds = [PolicyPlayer.embed_move(move) for move in move_list]
        move_embeds += [np.zeros(move_obs_len, dtype=np.float32)] * (
            4 - len(move_embeds)
        )
        move_embeds = np.concatenate(move_embeds)
        move_sem = np.concatenate(
            [move_semantics(m.id) for m in move_list]
            + [_ZERO_MOVE_SEM] * (4 - len(move_list))
        )
        types = [float(t in pokemon.base_types) for t in PokemonType]
        tera_type = [float(t == pokemon.tera_type) for t in PokemonType]
        base_stats = [s / 255 for s in pokemon.base_stats.values()]
        if from_opponent:
            stats = [-1] * 6
        else:
            stat_names = ("hp", "atk", "def", "spa", "spd", "spe")
            raw_stats = pokemon.stats or {}
            if any(raw_stats.get(name) is None for name in stat_names):
                # poke-env can temporarily expose an unselected or newly restored
                # teammate without its request-side stats during a forced-switch
                # message.  Crashing here leaves the battle waiting forever.  Use
                # the same conservative Champions stat completion as the mechanics
                # layer; complete request stats remain bit-for-bit unchanged.
                from vgc_bench.src.vgc_knowledge import ensure_stats

                if ensure_stats(pokemon):
                    PolicyPlayer.guard_fire_counts["own_stats_imputed"] += 1
                raw_stats = pokemon.stats or {}
            stats = [float(raw_stats[name]) / 255 for name in stat_names]
        gender = [float(g == pokemon.gender) for g in PokemonGender]
        weight = pokemon.weight / 1000
        # volatile fields
        hp_frac = pokemon.current_hp_fraction
        revealed = float(pokemon.revealed)
        in_draft = float(pokemon.selected_in_teampreview)
        status = [float(s == pokemon.status) for s in Status]
        status_counter = pokemon.status_counter / 16
        boosts = [b / 6 for b in pokemon.boosts.values()]
        effects = [
            (min(pokemon.effects[e], 8) / 8 if e in pokemon.effects else 0)
            for e in Effect
        ]
        first_turn = float(pokemon.first_turn)
        protect_counter = pokemon.protect_counter / 5
        must_recharge = float(pokemon.must_recharge)
        preparing = float(pokemon.preparing)
        gimmicks = [float(s) for s in [pokemon.is_dynamaxed, pokemon.is_terastallized]]
        pos_onehot = [float(pos == i) for i in range(6)]
        return np.array(
            [
                ability_id,
                item_id,
                *move_ids,
                *move_embeds,
                *types,
                *tera_type,
                *base_stats,
                *stats,
                *gender,
                weight,
                hp_frac,
                revealed,
                in_draft,
                *status,
                status_counter,
                *boosts,
                *effects,
                first_turn,
                protect_counter,
                must_recharge,
                preparing,
                *gimmicks,
                float(active_a),
                float(active_b),
                *pos_onehot,
                float(from_opponent),
                # knowledge block LAST -- see utils.knowledge_obs_len for why order
                # matters for checkpoint compatibility
                *(knowledge if knowledge is not None else _ZERO_KNOWLEDGE),
                # what each move / this ability actually does
                *move_sem,
                *ability_semantics(ability),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def embed_move(move: Move) -> npt.NDArray[np.float32]:
        """Embed a move's power, accuracy, type, and special properties."""
        power = move.base_power / 250
        # Legacy feature retained for checkpoint compatibility. The correctly scaled
        # value is appended by embed_move_accuracies at the end of each token.
        acc = move.accuracy / 100
        category = [float(c == move.category) for c in MoveCategory]
        target = [float(t == move.target) for t in Target]
        priority = (move.priority + 7) / 12
        crit_ratio = move.crit_ratio
        drain = move.drain
        force_switch = float(move.force_switch)
        recoil = move.recoil
        self_destruct = float(move.self_destruct is not None)
        self_switch = float(move.self_switch is not False)
        pp = move.max_pp / 64
        pp_frac = move.current_pp / move.max_pp
        is_last_used = float(move.is_last_used)
        move_type = [float(t == move.type) for t in PokemonType]
        return np.array(
            [
                power,
                acc,
                *category,
                *target,
                priority,
                crit_ratio,
                drain,
                force_switch,
                recoil,
                self_destruct,
                self_switch,
                pp,
                pp_frac,
                is_last_used,
                *move_type,
            ]
        )


@dataclass
class _BatchReq:
    """Internal request object for batched inference."""

    obs: npt.NDArray[np.float32]
    mask: npt.NDArray[np.int64]
    event: asyncio.Event
    result: npt.NDArray[np.int64] | None = None


class BatchPolicyPlayer(PolicyPlayer):
    """
    A policy player that batches inference requests for efficiency.

    Collects multiple battle observations and runs them through the policy
    network together, improving GPU utilization when managing many concurrent
    battles.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the batch policy player with an inference queue."""
        super().__init__(*args, **kwargs)
        self._q: asyncio.Queue[_BatchReq] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    def choose_move(self, battle: AbstractBattle) -> Awaitable[BattleOrder]:
        """Return an awaitable that resolves to the chosen battle order."""
        return self._choose_move(battle)

    async def _choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """Queue an observation for batched inference and await the result."""
        assert isinstance(battle, DoubleBattle)
        if battle._wait:
            return DefaultBattleOrder()
        self._update_battle_plan(battle)
        obs = self.embed_battle(battle, fake_rating=2000)
        mask = np.array(DoublesEnv.get_action_mask(battle))
        if PolicyPlayer.mask_immunities:
            mask = PolicyPlayer.mask_immune_actions(battle, mask)
        if battle.teampreview and self.forced_bench_species:
            mask = self._apply_forced_bench(battle, mask)
        if (
            PolicyPlayer.use_knowledge_guards
            or self.enable_search
            or self.residual_ranker_path is not None
            or getattr(self, "use_opponent_reranker", False)
            or getattr(self, "use_tempo_reranker", False)
        ):
            # Guards need the battle object and a per-battle candidate ranking, which
            # the shared batch loop cannot provide (it only returns sampled actions).
            # Batching is a throughput optimisation for training; eval and ladder can
            # afford one forward pass per decision.
            if not isinstance(self.policy, MaskedActorCriticPolicy):
                self._repair_policy()
            if not isinstance(self.policy, MaskedActorCriticPolicy):
                return DefaultBattleOrder()
            with torch.no_grad():
                obs_dict = {
                    "observation": torch.as_tensor(
                        obs, device=self.policy.device
                    ).unsqueeze(0),
                    "action_mask": torch.as_tensor(
                        mask, device=self.policy.device
                    ).unsqueeze(0),
                }
                if self.enable_search:
                    # Each battle owns a separate exact bridge/session, so local
                    # evaluation can search several games concurrently without
                    # blocking poke-env's websocket event loop. Ladder remains
                    # max_concurrent_battles=1.
                    action = await asyncio.to_thread(
                        self._guarded_action, battle, obs_dict, obs_dict["action_mask"]
                    )
                else:
                    action = self._guarded_action(
                        battle, obs_dict, obs_dict["action_mask"]
                    )
            return DoublesEnv.action_to_order(action, battle)
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._inference_loop())
        req = _BatchReq(obs=obs, mask=mask, event=asyncio.Event())
        await self._q.put(req)
        await req.event.wait()
        assert req.result is not None
        action = req.result
        return DoublesEnv.action_to_order(action, battle)

    def teampreview(self, battle: AbstractBattle) -> Awaitable[str]:
        """Return an awaitable that resolves to the team order string."""
        return self._teampreview(battle)

    async def _teampreview(self, battle: AbstractBattle) -> str:
        """Async teampreview implementation with random fallback when disabled."""
        assert isinstance(battle, DoubleBattle)
        if not isinstance(self.policy, MaskedActorCriticPolicy):
            self._repair_policy()
        if not isinstance(self.policy, MaskedActorCriticPolicy):
            return self.random_teampreview(battle)
        # Off-loop for the same reason as the sync class: multi-world preview
        # budgets exceed the 20s websocket keepalive window.
        planned = await asyncio.to_thread(self._planned_teampreview, battle)
        if planned is not None:
            return planned
        outcome = self._outcome_teampreview(battle)
        if outcome is not None:
            return outcome
        learned = self._learned_teampreview(battle)
        if learned is not None:
            return learned
        if not self.policy.choose_on_teampreview:
            return self.random_teampreview(battle)
        action1 = self._forced_lead_actions(battle)
        if action1 is None:
            order1 = await self.choose_move(battle)
            action1 = DoublesEnv.order_to_action(order1, battle)
        list(battle.team.values())[action1[0] - 1]._selected_in_teampreview = True
        list(battle.team.values())[action1[1] - 1]._selected_in_teampreview = True
        order2 = await self.choose_move(battle)
        action2 = DoublesEnv.order_to_action(order2, battle)
        list(battle.team.values())[action2[0] - 1]._selected_in_teampreview = True
        list(battle.team.values())[action2[1] - 1]._selected_in_teampreview = True
        self._record_own_preview(
            battle,
            (int(action1[0]), int(action1[1])),
            (int(action2[0]), int(action2[1])),
        )
        self._shadow_preview_record(
            battle,
            (int(action1[0]), int(action1[1])),
            (int(action2[0]), int(action2[1])),
        )
        return f"/team {action1[0]}{action1[1]}{action2[0]}{action2[1]}"

    async def _inference_loop(self) -> None:
        """Background task that batches and processes inference requests."""
        if not isinstance(self.policy, MaskedActorCriticPolicy):
            self._repair_policy()
        while True:
            # gather requests
            requests = [await self._q.get()]
            just_slept = False
            while len(requests) < self._max_concurrent_battles:
                try:
                    req = self._q.get_nowait()
                    requests.append(req)
                    just_slept = False
                except asyncio.QueueEmpty:
                    if just_slept:
                        break
                    await asyncio.sleep(0.005)
                    just_slept = True

            # run inference
            obs = np.stack([r.obs for r in requests], axis=0)
            masks = np.stack([r.mask for r in requests], axis=0)
            if isinstance(self.policy, MaskedActorCriticPolicy):
                with torch.no_grad():
                    obs_dict = {
                        "observation": torch.as_tensor(obs, device=self.policy.device),
                        "action_mask": torch.as_tensor(
                            masks, device=self.policy.device
                        ),
                    }
                    actions, _, _ = self.policy.forward(
                        obs_dict, deterministic=self.deterministic
                    )
                actions = actions.cpu().numpy()
            else:
                # No usable policy (see _repair_policy): play the first legal
                # action per slot so batched battles keep moving instead of
                # stalling on a dead worker task. Counted, never silent.
                PolicyPlayer.guard_fire_counts["policy_unavailable_batch"] += len(
                    requests
                )
                actions = np.argmax(masks, axis=-1)

            # dispatch
            for req, act in zip(requests, actions):
                req.result = act
                req.event.set()
