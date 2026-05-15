"""
PPO agent wrapper for Blackjack (via Stable-Baselines3).

Wraps SB3's PPO implementation with project-specific configuration,
logging, and evaluation interfaces. Designed for the finite-deck
environment where the extended state space enables implicit card counting.

Reference: Schulman et al. (2017), "Proximal Policy Optimization Algorithms"
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env

from src.utils.config import PPOConfig

logger = logging.getLogger(__name__)


class PPOBlackjackAgent:
    """
    PPO agent for Blackjack using Stable-Baselines3.

    PPO's clipped surrogate objective prevents catastrophic weight updates
    from lucky streaks, making it uniquely stable for the high-variance
    Blackjack environment. Combined with the extended state space, PPO can
    discover card counting strategies autonomously.
    """

    def __init__(
        self,
        env_id: str = "BlackjackFiniteDeck-v0",
        config: PPOConfig | None = None,
        env_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize PPO agent.

        Args:
            env_id: Gymnasium environment ID.
            config: PPO hyperparameters.
            env_kwargs: Additional kwargs passed to the environment constructor.
        """
        self.config = config or PPOConfig()
        self._name = "PPO"
        self.env_id = env_id

        # Resolve device
        device = self.config.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create environment
        self.env = make_vec_env(env_id, n_envs=1, env_kwargs=env_kwargs or {})

        # Initialize SB3 PPO
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            policy_kwargs={"net_arch": self.config.net_arch},
            verbose=1,
            device=device,
            tensorboard_log=None,  # We handle TB logging separately
        )

        logger.info(
            "PPOBlackjackAgent initialized: env=%s, device=%s, "
            "net_arch=%s, lr=%.1e, n_steps=%d, clip=%.2f, ent_coef=%.3f",
            env_id, device, self.config.net_arch,
            self.config.learning_rate, self.config.n_steps,
            self.config.clip_range, self.config.ent_coef,
        )

    @property
    def name(self) -> str:
        return self._name

    def train(
        self,
        total_timesteps: int | None = None,
        tb_log_dir: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        eval_env_id: str | None = None,
        eval_freq: int | None = None,
    ) -> None:
        """
        Train the PPO agent.

        Args:
            total_timesteps: Override config timesteps if provided.
            tb_log_dir: TensorBoard log directory.
            checkpoint_dir: Directory for periodic checkpoints.
            eval_env_id: Environment ID for evaluation callback.
            eval_freq: Steps between evaluations.
        """
        timesteps = total_timesteps or self.config.total_timesteps
        callbacks = []

        # Set up TensorBoard logging
        if tb_log_dir:
            self.model.tensorboard_log = str(tb_log_dir)

        # Checkpoint callback
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                CheckpointCallback(
                    save_freq=self.config.checkpoint_interval,
                    save_path=str(checkpoint_dir),
                    name_prefix="ppo_blackjack",
                )
            )

        # Evaluation callback
        if eval_env_id and eval_freq:
            eval_env = make_vec_env(eval_env_id, n_envs=1)
            callbacks.append(
                EvalCallback(
                    eval_env,
                    eval_freq=eval_freq,
                    n_eval_episodes=1000,
                    best_model_save_path=str(checkpoint_dir or Path("models")),
                    log_path=str(checkpoint_dir or Path("results/logs")),
                    deterministic=True,
                )
            )

        logger.info("Starting PPO training: %d timesteps", timesteps)

        self.model.learn(
            total_timesteps=timesteps,
            callback=callbacks if callbacks else None,
            log_interval=self.config.log_interval,
            progress_bar=True,
        )

        logger.info("PPO training complete: %d timesteps", timesteps)

    def get_action(
        self,
        state: tuple | np.ndarray,
        greedy: bool = False,
    ) -> int:
        """
        Select action using the learned policy.

        Args:
            state: Observation from the environment.
            greedy: If True, use deterministic policy.

        Returns:
            Action integer.
        """
        if isinstance(state, tuple):
            # Convert basic Blackjack tuple to array
            player_sum, dealer_card, usable_ace = state
            obs = np.array(
                [player_sum / 31.0, dealer_card / 10.0, float(usable_ace)],
                dtype=np.float32,
            )
        else:
            obs = np.asarray(state, dtype=np.float32)

        action, _ = self.model.predict(obs, deterministic=greedy)
        return int(action)

    def get_policy(self) -> dict[tuple[int, int, bool], int]:
        """
        Extract greedy policy for standard Blackjack states.

        Note: For finite-deck environments, this only covers the base
        3-dimensional states (ignoring card counts) for comparison purposes.
        """
        policy: dict[tuple[int, int, bool], int] = {}
        for player_sum in range(4, 22):
            for dealer_card in range(1, 11):
                for usable_ace in [False, True]:
                    state = (player_sum, dealer_card, usable_ace)
                    policy[state] = self.get_action(state, greedy=True)
        return policy

    def save(self, filepath: str | Path) -> None:
        """Save the PPO model."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(filepath))
        logger.info("PPO model saved to %s", filepath)

    def load(self, filepath: str | Path) -> None:
        """Load a saved PPO model."""
        filepath = Path(filepath)
        self.model = PPO.load(str(filepath), env=self.env)
        logger.info("PPO model loaded from %s", filepath)
