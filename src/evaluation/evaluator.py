"""
Standardized evaluation harness for all Blackjack agents.

Runs a configurable number of hands with greedy policy (epsilon=0)
and computes comprehensive performance metrics against the environment.
"""

import logging
from typing import Protocol

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from src.evaluation.basic_strategy import (
    calculate_strategy_agreement,
    get_basic_strategy_action,
)
from src.evaluation.metrics import EvaluationResult, compute_confidence_interval

logger = logging.getLogger(__name__)


class Agent(Protocol):
    """Protocol defining the interface all agents must implement."""

    @property
    def name(self) -> str: ...

    def get_action(self, state: tuple[int, int, bool], greedy: bool = False) -> int: ...

    def get_policy(self) -> dict[tuple[int, int, bool], int]: ...


class BasicStrategyAgent:
    """
    Reference agent that plays perfect Basic Strategy.
    Used as the theoretical ceiling for infinite-deck performance.
    """

    @property
    def name(self) -> str:
        return "Basic Strategy (Optimal)"

    def get_action(self, state: tuple[int, int, bool], greedy: bool = False) -> int:
        player_sum, dealer_card, usable_ace = state
        return get_basic_strategy_action(player_sum, dealer_card, usable_ace)

    def get_policy(self) -> dict[tuple[int, int, bool], int]:
        policy = {}
        for player_sum in range(4, 22):
            for dealer_card in range(1, 11):
                for usable_ace in [False, True]:
                    policy[(player_sum, dealer_card, usable_ace)] = (
                        get_basic_strategy_action(player_sum, dealer_card, usable_ace)
                    )
        return policy


def evaluate_agent(
    agent: Agent,
    num_hands: int = 100_000,
    natural: bool = True,
    sab: bool = True,
    seed: int | None = 42,
    show_progress: bool = True,
) -> EvaluationResult:
    """
    Evaluate an agent over a specified number of hands using greedy policy.

    Args:
        agent: Any agent implementing the Agent protocol.
        num_hands: Number of hands to play.
        natural: Whether natural Blackjack pays 1.5x.
        sab: Whether to use Sutton & Barto ruleset.
        seed: Random seed for reproducibility.
        show_progress: Whether to show tqdm progress bar.

    Returns:
        EvaluationResult with comprehensive metrics.
    """
    logger.info(
        "Starting evaluation: agent=%s, num_hands=%d, natural=%s, sab=%s, seed=%s",
        agent.name, num_hands, natural, sab, seed,
    )

    env = gym.make("Blackjack-v1", natural=natural, sab=sab)
    rewards_list: list[float] = []
    wins = 0
    losses = 0
    draws = 0
    busts = 0
    natural_bjs = 0

    iterator = range(num_hands)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Evaluating {agent.name}", unit="hands")

    for i in iterator:
        state, info = env.reset(seed=seed + i if seed is not None else None)
        done = False
        episode_reward = 0.0
        player_busted = False

        while not done:
            action = agent.get_action(state, greedy=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

            # Detect bust: if player hit and the game ended with negative reward
            if action == 1 and done and reward < 0:
                player_busted = True

            state = next_state

        rewards_list.append(episode_reward)

        if episode_reward > 0:
            wins += 1
            if episode_reward == 1.5:
                natural_bjs += 1
        elif episode_reward < 0:
            losses += 1
        else:
            draws += 1

        if player_busted:
            busts += 1

    env.close()

    rewards_array = np.array(rewards_list)
    total_reward = float(np.sum(rewards_array))
    ci_lower, ci_upper = compute_confidence_interval(rewards_array)

    result = EvaluationResult(
        agent_name=agent.name,
        num_hands=num_hands,
        wins=wins,
        losses=losses,
        draws=draws,
        busts=busts,
        natural_blackjacks=natural_bjs,
        total_reward=total_reward,
    )

    logger.info(result.summary())
    logger.info(
        "95%% CI for avg reward: (%.4f, %.4f)", ci_lower, ci_upper,
    )

    return result


def evaluate_basic_strategy(num_hands: int = 100_000) -> EvaluationResult:
    """
    Convenience function to evaluate the reference Basic Strategy agent.

    Args:
        num_hands: Number of hands to evaluate.

    Returns:
        EvaluationResult for the optimal Basic Strategy.
    """
    agent = BasicStrategyAgent()
    return evaluate_agent(agent, num_hands=num_hands)
