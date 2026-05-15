"""
Centralized configuration system for Blackjack AI.

All hyperparameters, environment settings, and training configs are defined here
as dataclasses for type safety and IDE autocompletion.
"""

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "results" / "logs"

# Ensure directories exist at import time
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def resolve_device(preference: str = "auto") -> str:
    """
    Resolve the actual usable device, testing CUDA before trusting it.

    torch.cuda.is_available() returns True even when the GPU's compute
    capability is unsupported (e.g., GTX 1050 Ti SM 6.1 on PyTorch 2.12+).
    This helper runs a real tensor operation to verify CUDA actually works.

    Args:
        preference: "auto", "cpu", or "cuda".

    Returns:
        "cuda" if GPU works, "cpu" otherwise.
    """
    if preference == "cpu":
        return "cpu"

    try:
        import torch

        if not torch.cuda.is_available():
            logger.info("CUDA not available. Using CPU.")
            return "cpu"

        # Actually test a tensor op — this is where SM 6.1 GPUs crash
        test_tensor = torch.zeros(1, device="cuda")
        _ = test_tensor + 1
        del test_tensor
        torch.cuda.empty_cache()

        gpu_name = torch.cuda.get_device_name(0)
        logger.info("CUDA verified working. Using GPU: %s", gpu_name)
        return "cuda"

    except Exception as e:
        logger.warning(
            "CUDA reported available but failed verification: %s. "
            "Falling back to CPU. This is normal for older GPUs "
            "(GTX 1050 Ti, etc.) with PyTorch 2.12+.",
            e,
        )
        return "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Tabular RL Configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TabularConfig:
    """Hyperparameters for tabular RL agents (Monte Carlo, SARSA, Q-Learning)."""

    # Training
    num_episodes: int = 500_000
    eval_episodes: int = 100_000

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_episodes: int = 400_000

    # Learning
    learning_rate: float = 0.01  # α — step size for TD updates
    discount_factor: float = 1.0  # γ — episodic task, no discounting needed

    # Environment
    use_natural_blackjack: bool = True  # +1.5 reward for natural BJ
    use_sab: bool = True  # Sutton & Barto ruleset alignment

    # Logging
    log_interval: int = 10_000  # Log metrics every N episodes
    checkpoint_interval: int = 100_000  # Save Q-table every N episodes

    def get_epsilon(self, episode: int) -> float:
        """Calculate decaying epsilon for a given episode number."""
        return max(
            self.epsilon_end,
            self.epsilon_start
            - (self.epsilon_start - self.epsilon_end)
            * (episode / self.epsilon_decay_episodes),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Deep RL Configuration (DQN variants)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class DQNConfig:
    """Hyperparameters for DQN, Double DQN, and Dueling DQN agents."""

    # Network architecture
    hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    input_dim: int = 3  # (player_sum, dealer_card, usable_ace)
    output_dim: int = 2  # Hit, Stand

    # Training
    total_timesteps: int = 1_000_000
    eval_episodes: int = 100_000
    batch_size: int = 128
    learning_rate: float = 5e-4
    discount_factor: float = 1.0

    # Experience Replay
    replay_buffer_size: int = 100_000
    min_replay_size: int = 1_000  # Minimum transitions before training starts

    # Target Network — use soft updates (Polyak) when tau > 0, hard updates otherwise
    target_update_freq: int = 1_000  # Hard update frequency (ignored when soft updates are on)
    soft_update_tau: float = 0.005   # τ=0 disables soft updates, τ=1 is equivalent to hard copy

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 700_000  # Slower decay → more thorough exploration

    # Device
    device: str = "auto"  # "auto", "cpu", or "cuda"

    # Logging
    log_interval: int = 10_000
    checkpoint_interval: int = 100_000

    def get_epsilon(self, step: int) -> float:
        """Calculate decaying epsilon for a given timestep."""
        return max(
            self.epsilon_end,
            self.epsilon_start
            - (self.epsilon_start - self.epsilon_end)
            * (step / self.epsilon_decay_steps),
        )


# ──────────────────────────────────────────────────────────────────────────────
# PPO Configuration (Stable-Baselines3)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PPOConfig:
    """Hyperparameters for PPO agent via Stable-Baselines3."""

    # Training
    total_timesteps: int = 10_000_000
    eval_episodes: int = 100_000

    # PPO-specific
    learning_rate: float = 1e-4   # Lower LR → more stable policy updates
    n_steps: int = 4096           # Larger rollout → better return estimates for episodic task
    batch_size: int = 128
    n_epochs: int = 10  # Optimization epochs per rollout
    gamma: float = 1.0  # Episodic, no discounting
    clip_range: float = 0.2  # PPO clipping threshold
    ent_coef: float = 0.005  # Reduced entropy coef — agent is mostly converged on basic play
    vf_coef: float = 0.5  # Value function loss coefficient
    max_grad_norm: float = 0.5  # Gradient clipping

    # Network — deeper arch to capture card-count correlations across 13-dim state
    net_arch: list[int] = field(default_factory=lambda: [256, 256, 128])

    # Device
    device: str = "auto"

    # Logging
    log_interval: int = 10
    eval_freq: int = 100_000
    checkpoint_interval: int = 500_000


# ──────────────────────────────────────────────────────────────────────────────
# Finite Deck Environment Configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class FiniteDeckConfig:
    """Configuration for the custom finite-deck Blackjack environment."""

    num_decks: int = 1
    penetration: float = 0.75  # Reshuffle when 75% of shoe is dealt
    natural_bonus: float = 1.5  # Reward multiplier for natural Blackjack
    surrender_penalty: float = -0.5  # Reward for surrender action


# ──────────────────────────────────────────────────────────────────────────────
# Curriculum Learning Configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class CurriculumConfig:
    """Configuration for phased curriculum learning."""

    # Phase boundaries (cumulative timesteps)
    phase1_end: int = 2_000_000  # Hit/Stand only
    phase2_end: int = 5_000_000  # + Double Down
    phase3_end: int = 10_000_000  # + Split/Surrender

    # PPO config per phase (inherits from PPOConfig)
    ppo_config: PPOConfig = field(default_factory=PPOConfig)
