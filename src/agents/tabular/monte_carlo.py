"""
First-Visit Monte Carlo Control agent for Blackjack.

Learns action-value function Q(s,a) by averaging returns observed after
the first visit to each state-action pair across complete episodes.
Uses ε-greedy exploration with decaying epsilon.

Reference: Sutton & Barto, "Reinforcement Learning: An Introduction", Ch. 5
"""

import logging
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.utils.config import TabularConfig

logger = logging.getLogger(__name__)


class MonteCarloAgent:
    """
    First-Visit Monte Carlo Control with ε-greedy exploration.

    The agent plays out complete episodes, then updates Q-values by averaging
    the returns observed from each state-action pair's first occurrence.
    This is mathematically unbiased but has high variance due to the
    stochasticity of Blackjack's deck.
    """

    def __init__(self, config: TabularConfig | None = None) -> None:
        """
        Initialize Monte Carlo agent.

        Args:
            config: Hyperparameter configuration. Uses defaults if None.
        """
        self.config = config or TabularConfig()
        self._name = "Monte Carlo (First-Visit)"

        # Q-table: maps (player_sum, dealer_card, usable_ace, action) → value
        self.q_table: dict[tuple[int, int, bool, int], float] = defaultdict(float)

        # Visit counts for incremental mean calculation
        self.returns_count: dict[tuple[int, int, bool, int], int] = defaultdict(int)

        # Training statistics
        self.training_rewards: list[float] = []
        self.episode_count: int = 0

        logger.info(
            "MonteCarloAgent initialized: episodes=%d, epsilon_start=%.2f, "
            "epsilon_end=%.4f, decay_episodes=%d",
            self.config.num_episodes,
            self.config.epsilon_start,
            self.config.epsilon_end,
            self.config.epsilon_decay_episodes,
        )

    @property
    def name(self) -> str:
        return self._name

    def get_action(self, state: tuple[int, int, bool], greedy: bool = False) -> int:
        """
        Select an action using ε-greedy policy.

        Args:
            state: (player_sum, dealer_card, usable_ace).
            greedy: If True, always select the best action (epsilon=0).

        Returns:
            Action: 0 (Stand) or 1 (Hit).
        """
        if not greedy:
            epsilon = self.config.get_epsilon(self.episode_count)
            if np.random.random() < epsilon:
                return np.random.randint(0, 2)

        # Greedy selection: pick action with highest Q-value
        player_sum, dealer_card, usable_ace = state
        q_stand = self.q_table[(player_sum, dealer_card, usable_ace, 0)]
        q_hit = self.q_table[(player_sum, dealer_card, usable_ace, 1)]

        if q_stand == q_hit:
            return np.random.randint(0, 2)  # Break ties randomly
        return 0 if q_stand > q_hit else 1

    def update(self, episode: list[tuple[tuple[int, int, bool], int, float]]) -> None:
        """
        Update Q-table using first-visit Monte Carlo returns.

        For each state-action pair encountered for the first time in the episode,
        compute the return (cumulative reward from that point) and update the
        running average in the Q-table.

        Args:
            episode: List of (state, action, reward) tuples from a complete episode.
        """
        visited: set[tuple[int, int, bool, int]] = set()
        G = 0.0  # Return accumulator

        # Process episode backwards to compute returns efficiently
        for state, action, reward in reversed(episode):
            G = self.config.discount_factor * G + reward
            player_sum, dealer_card, usable_ace = state
            sa_key = (player_sum, dealer_card, usable_ace, action)

            # First-visit check: only update on first occurrence (reversed order)
            if sa_key not in visited:
                visited.add(sa_key)
                self.returns_count[sa_key] += 1
                n = self.returns_count[sa_key]

                # Incremental mean update: Q = Q + (1/n)(G - Q)
                self.q_table[sa_key] += (G - self.q_table[sa_key]) / n

    def get_policy(self) -> dict[tuple[int, int, bool], int]:
        """
        Extract the greedy policy from the current Q-table.

        Returns:
            Dictionary mapping states → optimal actions.
        """
        policy: dict[tuple[int, int, bool], int] = {}
        for player_sum in range(4, 22):
            for dealer_card in range(1, 11):
                for usable_ace in [False, True]:
                    q_stand = self.q_table[(player_sum, dealer_card, usable_ace, 0)]
                    q_hit = self.q_table[(player_sum, dealer_card, usable_ace, 1)]
                    policy[(player_sum, dealer_card, usable_ace)] = (
                        0 if q_stand >= q_hit else 1
                    )
        return policy

    def save(self, filepath: str | Path) -> None:
        """Save Q-table and agent state to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "q_table": dict(self.q_table),
            "returns_count": dict(self.returns_count),
            "episode_count": self.episode_count,
            "training_rewards": self.training_rewards,
            "config": self.config,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        logger.info("MonteCarloAgent saved to %s (episodes=%d)", filepath, self.episode_count)

    def load(self, filepath: str | Path) -> None:
        """Load Q-table and agent state from disk."""
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        self.q_table = defaultdict(float, state["q_table"])
        self.returns_count = defaultdict(int, state["returns_count"])
        self.episode_count = state["episode_count"]
        self.training_rewards = state["training_rewards"]
        logger.info(
            "MonteCarloAgent loaded from %s (episodes=%d, q_entries=%d)",
            filepath, self.episode_count, len(self.q_table),
        )
