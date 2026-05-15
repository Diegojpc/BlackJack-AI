"""
Gymnasium environment wrappers for Blackjack.

Provides observation transformations and reward modifications needed by
different agent architectures (tabular vs. deep RL).
"""

import logging

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)


class FlattenObservationWrapper(gym.ObservationWrapper):
    """
    Converts Blackjack-v1's tuple observation (player_sum, dealer_card, usable_ace)
    into a flat numpy array suitable for neural network input.

    The default Gymnasium Blackjack-v1 returns a Tuple space of three Discrete values.
    Neural networks require a contiguous vector, so this wrapper normalizes the
    observation into a Box space with float32 values.

    Normalization:
        - player_sum: [0, 31] → [0.0, 1.0] (divided by 31)
        - dealer_card: [1, 10] → [0.0, 1.0] (divided by 10)
        - usable_ace: {0, 1} → {0.0, 1.0} (cast to float)
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        logger.info(
            "FlattenObservationWrapper initialized: original_space=%s, new_space=%s",
            env.observation_space,
            self.observation_space,
        )

    def observation(self, observation: tuple[int, int, int]) -> np.ndarray:
        """
        Transform tuple observation to normalized float array.

        Args:
            observation: Raw (player_sum, dealer_card, usable_ace) tuple.

        Returns:
            Normalized numpy array of shape (3,).
        """
        player_sum, dealer_card, usable_ace = observation
        return np.array(
            [player_sum / 31.0, dealer_card / 10.0, float(usable_ace)],
            dtype=np.float32,
        )


def make_blackjack_env(
    flatten: bool = False,
    natural: bool = True,
    sab: bool = True,
    render_mode: str | None = None,
) -> gym.Env:
    """
    Factory function to create a configured Blackjack-v1 environment.

    Args:
        flatten: If True, wraps with FlattenObservationWrapper for neural nets.
        natural: If True, natural Blackjack pays 1.5x instead of 1.0x.
        sab: If True, uses Sutton & Barto ruleset.
        render_mode: Gymnasium render mode ("human", "rgb_array", or None).

    Returns:
        Configured Gymnasium environment.
    """
    logger.info(
        "Creating Blackjack-v1 env: flatten=%s, natural=%s, sab=%s, render_mode=%s",
        flatten, natural, sab, render_mode,
    )
    env = gym.make(
        "Blackjack-v1",
        natural=natural,
        sab=sab,
        render_mode=render_mode,
    )
    if flatten:
        env = FlattenObservationWrapper(env)
        logger.info("Applied FlattenObservationWrapper")

    return env
