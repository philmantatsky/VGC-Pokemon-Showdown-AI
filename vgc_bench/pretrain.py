"""
Pretraining module for VGC-Bench.

Implements behavior cloning (BC) pretraining using trajectory data extracted
from human battle logs. The pretrained policy can then be fine-tuned using
reinforcement learning.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from imitation.algorithms.bc import BC
from imitation.data.types import DictObs, Trajectory
from imitation.util.logger import configure
from poke_env.environment import SingleAgentWrapper
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client import ServerConfiguration
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, Dataset

from vgc_bench.src.callback import Callback
from vgc_bench.src.env import ShowdownEnv
from vgc_bench.src.policy import MaskedActorCriticPolicy
from vgc_bench.src.policy_player import BatchPolicyPlayer
from vgc_bench.src.teams import RandomTeamBuilder, get_available_regs
from vgc_bench.src.utils import act_len, format_map, set_global_seed


class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset for loading trajectory data from pickle files.

    Loads pre-extracted trajectories from data/trajs/.

    Attributes:
        files: List of trajectory file paths.
    """

    def __init__(self, trajs_dir: str = "trajs"):
        """Initialize the dataset by discovering trajectory files."""
        directory = Path(trajs_dir)
        self.files = [file for file in directory.iterdir() if file.suffix == ".pkl"]

    def __len__(self):
        """Return the number of trajectories in the dataset."""
        return len(self.files)

    def __getitem__(self, idx):
        """
        Load and return a trajectory by index.

        Wraps raw numpy observations into DictObs with a synthetic action
        mask so the trajectory matches the policy's Dict observation space.

        Args:
            idx: Index of the trajectory to load.

        Returns:
            Trajectory object with DictObs observations.
        """
        file_path = self.files[idx]
        with file_path.open("rb") as f:
            traj = pickle.load(f)
        obs = traj.obs
        n_steps = obs.shape[0]
        action_mask = np.ones((n_steps, 2 * act_len), dtype=np.float32)
        action_mask[:, 47:87] = 0
        action_mask[:, act_len + 47 : act_len + 87] = 0
        dict_obs = DictObs({"observation": obs, "action_mask": action_mask})
        return Trajectory(
            obs=dict_obs, acts=traj.acts, infos=traj.infos, terminal=traj.terminal
        )


def _identity_collate(batch):
    """Pass trajectories through unbatched.

    Must be a module-level function, not a lambda: macOS/Windows spawn DataLoader
    workers rather than forking them, and spawned workers pickle collate_fn.
    """
    return batch


def pretrain(
    run_id: int,
    port: int,
    device: str,
    div_frac: float,
    num_epochs: int = 100,
    init_from: str | None = None,
    trajs_dir: str = "trajs",
    output_dir: str = "results",
    eval_every: int = 1,
):
    """
    Pretrain a policy using behavior cloning on human gameplay data.

    Trains a neural network policy to imitate human players using trajectory
    data, periodically evaluating against a SimpleHeuristics opponent.
    Evaluates across all available VGC regulations using all available teams.

    Args:
        run_id: Training run identifier for saving checkpoints.
        port: Port for the Pokemon Showdown server.
        device: CUDA device for training.
        div_frac: Fraction of dataset to load per training iteration.
        num_epochs: Number of outer training epochs.
    """
    out_root = Path(output_dir)
    log_dir = out_root / "logs_bc" / f"seed{run_id}"
    save_dir = out_root / "saves_bc" / f"seed{run_id}"
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    battle_format = format_map[get_available_regs()[0]]
    env = ShowdownEnv(
        battle_format=battle_format,
        log_level=40,
        accept_open_team_sheet=True,
        start_listening=False,
        choose_on_teampreview=True,
    )
    opponent = RandomPlayer(start_listening=False)
    single_agent_env = SingleAgentWrapper(env, opponent)
    ppo = PPO(
        MaskedActorCriticPolicy,
        single_agent_env,
        policy_kwargs={"d_model": 256, "choose_on_teampreview": True},
        device=device,
    )
    if init_from is not None:
        # Continue from an existing BC checkpoint instead of random init, so a second
        # pass over a smaller high-quality set fine-tunes the model rather than
        # retraining it from scratch on far too little data.
        print(f"initializing policy from {init_from}", flush=True)
        ppo.set_parameters(str(init_from), device=ppo.device)
    dataset = TrajectoryDataset(trajs_dir)
    div_count = int(1 / div_frac)
    dataloader = DataLoader(
        dataset,
        batch_size=len(dataset) // div_count,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        collate_fn=_identity_collate,
    )
    bc = BC(
        observation_space=ppo.observation_space,
        action_space=ppo.action_space,
        rng=np.random.default_rng(run_id),
        policy=ppo.policy,
        batch_size=1024,
        device=device,
        custom_logger=configure(str(log_dir), ["tensorboard"]),
    )
    if eval_every <= 0:
        # No live evals: never construct the eval players, whose websocket
        # clients would otherwise try to reach a server that may not exist.
        ppo.save(save_dir / "0")
        for i in range(1, num_epochs + 1):
            data = iter(dataloader)
            for _ in range(div_count):
                demos = next(data)
                bc.set_demonstrations(demos)
                bc.train(n_epochs=1)
            ppo.save(save_dir / str(i))
        bc.train(n_epochs=1)
        return
    eval_agent = BatchPolicyPlayer(
        policy=ppo.policy,
        server_configuration=ServerConfiguration(
            f"ws://localhost:{port}/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?",
        ),
        battle_format=battle_format,
        log_level=40,
        max_concurrent_battles=10,
        accept_open_team_sheet=True,
        team=RandomTeamBuilder(run_id, None, None),
    )
    eval_opponent = SimpleHeuristicsPlayer(
        server_configuration=ServerConfiguration(
            f"ws://localhost:{port}/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?",
        ),
        battle_format=battle_format,
        log_level=40,
        max_concurrent_battles=10,
        accept_open_team_sheet=True,
        team=RandomTeamBuilder(run_id, None, None),
    )
    if eval_every > 0:
        win_rates = Callback.compare(eval_agent, eval_opponent, 100)
        for label, wr in win_rates.items():
            bc.logger.record(f"eval/heuristic{label}", wr)
    ppo.save(save_dir / "0")
    for i in range(1, num_epochs + 1):
        data = iter(dataloader)
        for _ in range(div_count):
            demos = next(data)
            bc.set_demonstrations(demos)
            bc.train(n_epochs=1)
        # 100 live battles per epoch dominate wall-clock on a small corpus;
        # eval_every thins them (0 disables live evals entirely, so no local
        # server is needed).
        if eval_every > 0 and i % eval_every == 0:
            win_rates = Callback.compare(eval_agent, eval_opponent, 100)
            for label, wr in win_rates.items():
                bc.logger.record(f"eval/heuristic{label}", wr)
        ppo.save(save_dir / str(i))
    bc.train(n_epochs=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretrain a policy using behavior cloning"
    )
    parser.add_argument(
        "--div_frac",
        type=float,
        default=0.01,
        help=(
            "fraction of total dataset to load at a given time during training (must be"
            " <1 when dataset is large)"
        ),
    )
    parser.add_argument(
        "--run_id", type=int, default=1, help="run ID for the training session"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="port to run showdown server on"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="device to use for pretraining"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="number of outer training epochs"
    )
    parser.add_argument(
        "--init_from",
        type=str,
        default=None,
        help=(
            "path to an existing BC checkpoint (.zip) to start from, instead of random"
            " init; use to fine-tune on a smaller high-quality dataset"
        ),
    )
    parser.add_argument(
        "--trajs_dir",
        type=str,
        default="trajs",
        help="trajectory pickle directory (default: trajs/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="root for logs_bc/ and saves_bc/ (default: results/)",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=1,
        help=(
            "run the 100-battle heuristic eval every N epochs; 0 disables live "
            "evals so no local server is required"
        ),
    )
    args = parser.parse_args()
    set_global_seed(args.run_id)
    pretrain(
        args.run_id,
        args.port,
        args.device,
        args.div_frac,
        args.num_epochs,
        args.init_from,
        args.trajs_dir,
        args.output_dir,
        args.eval_every,
    )
