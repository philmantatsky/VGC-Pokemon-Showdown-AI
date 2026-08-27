"""
Gymnasium environment module for VGC-Bench.

Provides a custom Gymnasium environment wrapping poke-env's DoublesEnv for
training reinforcement learning agents on Pokemon VGC battles.
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
import supersuit as ss
from gymnasium import Env
from gymnasium.spaces import Box
from poke_env.battle import AbstractBattle
from poke_env.environment import DoublesEnv, SingleAgentWrapper
from poke_env.ps_client import ServerConfiguration
from stable_baselines3.common.monitor import Monitor

from vgc_bench.src.policy_player import PolicyPlayer
from vgc_bench.src.teams import RandomTeamBuilder, TeamToggle, get_available_regs
from vgc_bench.src.utils import LearningStyle, chunk_obs_len, format_map, moves


class ShowdownEnv(DoublesEnv):
    """
    Gymnasium environment for Pokemon VGC doubles battles.

    Extends poke-env's DoublesEnv with custom observation embedding,
    reward calculation, and support for various training paradigms.
    """

    def __init__(
        self,
        *args: Any,
        hidden_sheet_prob: float = 0.0,
        sheet_seed: int = 0,
        **kwargs: Any,
    ):
        """
        Initialize the ShowdownEnv.
        """
        if not 0.0 <= hidden_sheet_prob <= 1.0:
            raise ValueError("hidden_sheet_prob must be between 0 and 1")
        self._hidden_sheet_prob = hidden_sheet_prob
        self._sheet_seed = sheet_seed
        self._sheet_rng = random.Random(sheet_seed)
        self._hidden_sheet_battles = 0
        self._sheet_battles = 0
        self._our_builder: RandomTeamBuilder | None = None
        self._opp_builder: RandomTeamBuilder | None = None
        super().__init__(*args, **kwargs)
        self.observation_spaces = {
            agent: Box(-1, len(moves), shape=(12 * chunk_obs_len,), dtype=np.float32)
            for agent in self.possible_agents
        }

    @classmethod
    def create_env(
        cls,
        reg: str | None,
        run_id: int,
        num_teams: int | None,
        num_envs: int,
        log_level: int,
        port: int,
        learning_style: LearningStyle,
        allow_mirror_match: bool,
        choose_on_teampreview: bool,
        team_paths: list[Path] | None = None,
        our_team_paths: list[Path] | None = None,
        hidden_sheet_prob: float = 0.0,
        team_weights_path: Path | None = None,
    ) -> Env:
        """
        Factory method to create a properly wrapped training environment.

        Creates the base ShowdownEnv and applies appropriate wrappers based
        on the learning style (vectorization for self-play, single-agent
        wrapper for other paradigms).

        Args:
            reg: VGC regulation identifier (e.g. 'ma', 'mb'), or None for all.
            run_id: Training run identifier.
            num_teams: Number of teams to train with, or None for all.
            num_envs: Number of parallel environments.
            log_level: Logging verbosity for Showdown clients.
            port: Port for the Pokemon Showdown server.
            learning_style: Training paradigm to use.
            allow_mirror_match: Whether to allow same-team matchups.
            choose_on_teampreview: Whether policy controls teampreview.
            team_paths: Optional list of team file paths for matchup solving.
            our_team_paths: If given, agent1 is locked to these team(s) while agent2
                draws from the full pool. This expresses the ladder objective -- we
                bring one team and face many -- which neither num_teams nor
                allow_mirror_match can state, since poke-env hands the same builder
                to both agents.
            hidden_sheet_prob: Fraction of training battles where agent2 denies Open
                Team Sheets. If either player denies, both sides play without sheets.
            team_weights_path: Optional ladder-derived opponent sampling weights.

        Returns:
            Wrapped Gymnasium environment ready for training.
        """
        toggle = None if allow_mirror_match else TeamToggle()
        if reg is None:
            battle_format = format_map[get_available_regs()[0]]
        else:
            battle_format = format_map[reg]
        opp_builder = RandomTeamBuilder(
            run_id, num_teams, reg, team_paths, toggle, weights_path=team_weights_path
        )
        our_builder = (
            RandomTeamBuilder(run_id, 1, reg, our_team_paths, None)
            if our_team_paths
            else None
        )
        env = cls(
            server_configuration=ServerConfiguration(
                f"ws://localhost:{port}/showdown/websocket",
                "https://play.pokemonshowdown.com/action.php?",
            ),
            battle_format=battle_format,
            log_level=log_level,
            accept_open_team_sheet=True,
            open_timeout=None,
            team=opp_builder,
            choose_on_teampreview=choose_on_teampreview,
            hidden_sheet_prob=hidden_sheet_prob,
            sheet_seed=run_id,
        )
        if our_builder is not None:
            # Stashed on the env so they survive pickling into subprocess workers;
            # __setstate__ below re-applies them, because poke-env's own __setstate__
            # rebuilds both agents from self._team and would silently drop this.
            env._our_builder = our_builder
            env._opp_builder = opp_builder
            env._apply_per_agent_teams()
        if learning_style == LearningStyle.PURE_SELF_PLAY:
            env = ss.pettingzoo_env_to_vec_env_v1(env)
            env = ss.concat_vec_envs_v1(
                env,
                num_vec_envs=num_envs,
                num_cpus=num_envs,
                base_class="stable_baselines3",
            )
            return env
        else:
            opponent = PolicyPlayer(start_listening=False)
            env = SingleAgentWrapper(env, opponent)
            env = Monitor(env)
            return env

    def _apply_per_agent_teams(self) -> None:
        """Lock agent1 to our team and agent2 to the opponent pool."""
        our_builder = self._our_builder
        opp_builder = self._opp_builder
        if our_builder is None:
            return
        assert opp_builder is not None
        self.agent1._team = our_builder
        self.agent2._team = opp_builder

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore per-agent teams after unpickling into a subprocess worker.

        poke-env's __setstate__ rebuilds agent1/agent2 with team=self._team, which
        would revert both sides to the shared pool. That failure is silent -- training
        would run against the wrong objective for hours -- so re-apply here.
        """
        super().__setstate__(state)
        # Spawned workers should not all replay the identical visible/hidden sequence.
        self._sheet_rng = random.Random(self._sheet_seed * 1_000_003 + os.getpid())
        self._apply_per_agent_teams()

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Reset the environment, updating battle format if multi-reg."""
        self._configure_open_team_sheets()
        if getattr(self, "_our_builder", None) is not None:
            # Loud check for the silent-pickling failure described in __setstate__.
            assert self.agent1._team is not self.agent2._team, (
                "per-agent team builders were lost (likely unpickling); both sides "
                "would share a pool and training the wrong objective"
            )
        assert isinstance(self._team, RandomTeamBuilder)
        if self._team.available_regs is not None:
            assert self._team.current_reg is not None
            self._team.pick_reg()
            fmt = format_map[self._team.current_reg]
            self.agent1._format = fmt
            self.agent2._format = fmt
        return super().reset(seed=seed, options=options)

    def _configure_open_team_sheets(self) -> bool:
        """Sample and apply this battle's sheet mode; return True when hidden."""
        hidden = self._sheet_rng.random() < self._hidden_sheet_prob
        # If one side accepts while the other rejects, poke-env can race: the accepter
        # waits for showteam after the server has already closed the sheet choice. Both
        # rejecting produces the identical observable state (no sheets for either side)
        # without that permanent wait.
        self.agent1._accept_open_team_sheet = not hidden
        self.agent2._accept_open_team_sheet = not hidden
        if hidden:
            self._hidden_sheet_battles += 1
        else:
            self._sheet_battles += 1
        return hidden

    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Calculate reward for the current battle state.

        Returns:
            1 if won, -1 if lost, 0 otherwise.
        """
        if not battle.finished:
            return 0
        elif battle.won:
            return 1
        elif battle.lost:
            return -1
        else:
            return 0

    def embed_battle(self, battle: AbstractBattle) -> npt.NDArray[np.float32]:
        """
        Convert the battle state to a feature vector observation.

        Args:
            battle: Current battle state.

        Returns:
            Numpy array observation for the policy network.
        """
        return PolicyPlayer.embed_battle(battle, fake_rating=2000)
