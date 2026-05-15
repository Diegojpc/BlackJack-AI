"""
Unit tests for Blackjack AI — Phase 1 (Tabular RL).

Tests environment wrappers, agent initialization, training mechanics,
Basic Strategy reference, and evaluation harness.
"""

import gymnasium as gym
import numpy as np
import pytest

from src.agents.tabular.monte_carlo import MonteCarloAgent
from src.agents.tabular.q_learning import QLearningAgent
from src.agents.tabular.sarsa import SarsaAgent
from src.environments.wrappers import FlattenObservationWrapper, make_blackjack_env
from src.evaluation.basic_strategy import (
    calculate_strategy_agreement,
    get_basic_strategy_action,
)
from src.evaluation.evaluator import BasicStrategyAgent, evaluate_agent
from src.evaluation.metrics import EvaluationResult, compute_confidence_interval
from src.utils.config import TabularConfig


# ──────────────────────────────────────────────────────────────────────────────
# Environment Tests
# ──────────────────────────────────────────────────────────────────────────────
class TestEnvironmentWrappers:
    """Test Gymnasium environment wrappers."""

    def test_make_blackjack_env_default(self) -> None:
        """Default env returns tuple observations."""
        env = make_blackjack_env(flatten=False)
        state, _ = env.reset(seed=42)
        assert isinstance(state, tuple)
        assert len(state) == 3
        env.close()

    def test_make_blackjack_env_flattened(self) -> None:
        """Flattened env returns normalized numpy arrays."""
        env = make_blackjack_env(flatten=True)
        state, _ = env.reset(seed=42)
        assert isinstance(state, np.ndarray)
        assert state.shape == (3,)
        assert state.dtype == np.float32
        assert np.all(state >= 0.0) and np.all(state <= 1.0)
        env.close()

    def test_flatten_observation_values(self) -> None:
        """Verify normalization math is correct."""
        wrapper = FlattenObservationWrapper(
            gym.make("Blackjack-v1", natural=True, sab=True)
        )
        # player_sum=15, dealer_card=10, usable_ace=False
        obs = wrapper.observation((15, 10, 0))
        np.testing.assert_allclose(obs, [15 / 31.0, 10 / 10.0, 0.0], rtol=1e-5)

        # player_sum=21, dealer_card=1, usable_ace=True
        obs = wrapper.observation((21, 1, 1))
        np.testing.assert_allclose(obs, [21 / 31.0, 1 / 10.0, 1.0], rtol=1e-5)
        wrapper.close()


# ──────────────────────────────────────────────────────────────────────────────
# Basic Strategy Tests
# ──────────────────────────────────────────────────────────────────────────────
class TestBasicStrategy:
    """Test the reference Basic Strategy chart."""

    def test_always_stand_on_hard_20(self) -> None:
        """Player with 20 should always stand."""
        for dealer_card in range(1, 11):
            assert get_basic_strategy_action(20, dealer_card, False) == 0  # Stand

    def test_always_hit_on_hard_5(self) -> None:
        """Player with 5 should always hit."""
        for dealer_card in range(1, 11):
            assert get_basic_strategy_action(5, dealer_card, False) == 1  # Hit

    def test_hit_hard_12_against_dealer_2(self) -> None:
        """Player with hard 12 against dealer 2 should hit."""
        assert get_basic_strategy_action(12, 2, False) == 1  # Hit

    def test_stand_hard_12_against_dealer_5(self) -> None:
        """Player with hard 12 against dealer 5 should stand."""
        assert get_basic_strategy_action(12, 5, False) == 0  # Stand

    def test_soft_18_against_dealer_9(self) -> None:
        """Player with soft 18 against dealer 9 should hit."""
        assert get_basic_strategy_action(18, 9, True) == 1  # Hit

    def test_soft_18_against_dealer_7(self) -> None:
        """Player with soft 18 against dealer 7 should stand."""
        assert get_basic_strategy_action(18, 7, True) == 0  # Stand

    def test_strategy_agreement_perfect(self) -> None:
        """Basic Strategy agent should agree 100% with itself."""
        agent = BasicStrategyAgent()
        agreement = calculate_strategy_agreement(agent.get_policy())
        assert agreement["total_agreement"] == 100.0


# ──────────────────────────────────────────────────────────────────────────────
# Agent Tests
# ──────────────────────────────────────────────────────────────────────────────
class TestMonteCarloAgent:
    """Test Monte Carlo agent mechanics."""

    def test_initialization(self) -> None:
        agent = MonteCarloAgent()
        assert agent.name == "Monte Carlo (First-Visit)"
        assert agent.episode_count == 0
        assert len(agent.q_table) == 0

    def test_get_action_returns_valid(self) -> None:
        agent = MonteCarloAgent()
        for _ in range(100):
            action = agent.get_action((15, 7, False))
            assert action in [0, 1]

    def test_update_changes_q_table(self) -> None:
        agent = MonteCarloAgent()
        trajectory = [
            ((15, 7, False), 1, 0.0),  # Hit on 15 vs 7
            ((18, 7, False), 0, 1.0),  # Stand on 18 vs 7, win
        ]
        agent.update(trajectory)
        # Q-value for (15, 7, False, 1) should now be non-zero
        assert agent.q_table[(15, 7, False, 1)] != 0.0

    def test_greedy_action_after_training(self) -> None:
        """After updating, greedy action should reflect learned values."""
        agent = MonteCarloAgent()
        # Simulate: hitting on (20, 7, False) leads to bust (-1)
        for _ in range(50):
            agent.update([((20, 7, False), 1, -1.0)])
        # Simulate: standing on (20, 7, False) leads to win (+1)
        for _ in range(50):
            agent.update([((20, 7, False), 0, 1.0)])
        # Greedy should choose Stand
        assert agent.get_action((20, 7, False), greedy=True) == 0


class TestSarsaAgent:
    """Test SARSA agent mechanics."""

    def test_initialization(self) -> None:
        agent = SarsaAgent()
        assert agent.name == "SARSA (On-Policy TD)"

    def test_update_changes_q_table(self) -> None:
        agent = SarsaAgent()
        agent.update(
            state=(15, 7, False),
            action=1,
            reward=-1.0,
            next_state=(15, 7, False),
            next_action=0,
            done=True,
        )
        assert agent.q_table[(15, 7, False, 1)] != 0.0


class TestQLearningAgent:
    """Test Q-Learning agent mechanics."""

    def test_initialization(self) -> None:
        agent = QLearningAgent()
        assert agent.name == "Q-Learning (Off-Policy)"

    def test_update_uses_max_q(self) -> None:
        """Q-learning should use max next Q-value, not actual action."""
        agent = QLearningAgent()
        # Set up a known state
        agent.q_table[(20, 7, False, 0)] = 1.0  # Stand is great in next state
        agent.q_table[(20, 7, False, 1)] = -1.0  # Hit is terrible

        # Update from previous state: should use max(1.0, -1.0) = 1.0
        agent.update(
            state=(15, 7, False),
            action=1,
            reward=0.0,
            next_state=(20, 7, False),
            done=False,
        )
        # Q(15,7,False,1) should be updated towards 0 + γ*1.0 = 1.0
        assert agent.q_table[(15, 7, False, 1)] > 0


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation Tests
# ──────────────────────────────────────────────────────────────────────────────
class TestEvaluation:
    """Test evaluation harness and metrics."""

    def test_evaluation_result_metrics(self) -> None:
        result = EvaluationResult(
            agent_name="Test",
            num_hands=100,
            wins=42,
            losses=50,
            draws=8,
            busts=20,
            natural_blackjacks=5,
            total_reward=-8.0,
        )
        assert result.win_rate == 42.0
        assert result.loss_rate == 50.0
        assert result.draw_rate == 8.0
        assert result.bust_rate == 20.0
        assert result.avg_reward == -0.08
        assert result.natural_rate == 5.0

    def test_basic_strategy_evaluation(self) -> None:
        """Basic Strategy agent should win approximately 42-43%."""
        agent = BasicStrategyAgent()
        result = evaluate_agent(agent, num_hands=10_000, show_progress=False)
        # Reasonable range for 10K hands
        assert 38.0 < result.win_rate < 48.0, f"Win rate {result.win_rate}% outside expected range"
        assert result.avg_reward < 0, "Basic Strategy should have negative EV in infinite deck"

    def test_confidence_interval(self) -> None:
        """Test CI computation with known data."""
        rewards = np.array([1.0, -1.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0])
        ci_lower, ci_upper = compute_confidence_interval(rewards)
        assert ci_lower < 0.0 < ci_upper  # Mean is 0.0, CI should straddle it


# ──────────────────────────────────────────────────────────────────────────────
# Config Tests
# ──────────────────────────────────────────────────────────────────────────────
class TestConfig:
    """Test configuration system."""

    def test_epsilon_decay(self) -> None:
        config = TabularConfig(epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_episodes=100)
        assert config.get_epsilon(0) == 1.0
        assert config.get_epsilon(100) == pytest.approx(0.01)
        assert config.get_epsilon(200) == 0.01  # Clamped to min
        assert config.get_epsilon(50) == pytest.approx(0.505)
