"""
Off-Policy Q-Learning agent for Blackjack.

Learns the optimal action-value function Q(s,a) using the maximum future
Q-value regardless of the action actually taken. This aggressive off-policy
approach converges fastest to the theoretical optimum but is susceptible
to overestimation bias in highly stochastic environments like Blackjack.

Reference: Watkins & Dayan (1992), Sutton & Barto Ch. 6
"""

import logging
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.utils.config import TabularConfig

logger = logging.getLogger(__name__)


class QLearningAgent:
    """
    Off-Policy Q-Learning (tabular) agent.

    Key difference from SARSA: Q-learning uses max_a Q(s', a) in updates,
    not the actual action chosen by the behavior policy. This decouples
    exploration from evaluation, enabling faster convergence to the optimal
    policy at the cost of potential overestimation in stochastic environments.
    """

    def __init__(self, config: TabularConfig | None = None) -> None:
        """
        Initialize Q-Learning agent.

        Args:
            config: Hyperparameter configuration. Uses defaults if None.
        """
        self.config = config or TabularConfig()
        self._name = "Q-Learning (Off-Policy)"

        # Q-table: maps (player_sum, dealer_card, usable_ace, action) → value
        self.q_table: dict[tuple[int, int, bool, int], float] = defaultdict(float)

        # Training statistics
        self.training_rewards: list[float] = []
        self.episode_count: int = 0
        self.total_steps: int = 0

        logger.info(
            "QLearningAgent initialized: episodes=%d, alpha=%.4f, gamma=%.2f, "
            "epsilon_start=%.2f, epsilon_end=%.4f",
            self.config.num_episodes,
            self.config.learning_rate,
            self.config.discount_factor,
            self.config.epsilon_start,
            self.config.epsilon_end,
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

        player_sum, dealer_card, usable_ace = state
        q_stand = self.q_table[(player_sum, dealer_card, usable_ace, 0)]
        q_hit = self.q_table[(player_sum, dealer_card, usable_ace, 1)]

        if q_stand == q_hit:
            return np.random.randint(0, 2)
        return 0 if q_stand > q_hit else 1

    def update(
        self,
        state: tuple[int, int, bool],
        action: int,
        reward: float,
        next_state: tuple[int, int, bool],
        done: bool,
    ) -> None:
        """
        Perform Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)].

        The max operator is what makes Q-learning off-policy and aggressive:
        it always assumes optimal future play regardless of the exploratory
        action actually taken.

        Args:
            state: Current state (player_sum, dealer_card, usable_ace).
            action: Action taken in current state.
            reward: Reward received.
            next_state: State transitioned to.
            done: Whether the episode has terminated.
        """
        player_sum, dealer_card, usable_ace = state
        sa_key = (player_sum, dealer_card, usable_ace, action)

        current_q = self.q_table[sa_key]

        if done:
            target = reward
        else:
            np_sum, np_card, np_ace = next_state
            # Off-policy max: take the BEST possible next action value
            max_next_q = max(
                self.q_table[(np_sum, np_card, np_ace, 0)],
                self.q_table[(np_sum, np_card, np_ace, 1)],
            )
            target = reward + self.config.discount_factor * max_next_q

        td_error = target - current_q
        self.q_table[sa_key] = current_q + self.config.learning_rate * td_error
        self.total_steps += 1

    def get_policy(self) -> dict[tuple[int, int, bool], int]:
        """Extract the greedy policy from the current Q-table."""
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
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
            "training_rewards": self.training_rewards,
            "config": self.config,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        logger.info("QLearningAgent saved to %s (episodes=%d)", filepath, self.episode_count)

    def load(self, filepath: str | Path) -> None:
        """Load Q-table and agent state from disk."""
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        self.q_table = defaultdict(float, state["q_table"])
        self.episode_count = state["episode_count"]
        self.total_steps = state["total_steps"]
        self.training_rewards = state["training_rewards"]
        logger.info(
            "QLearningAgent loaded from %s (episodes=%d, q_entries=%d)",
            filepath, self.episode_count, len(self.q_table),
        )
