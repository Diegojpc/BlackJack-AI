"""
Training loop for tabular RL agents (Monte Carlo, SARSA, Q-Learning).

Handles the training lifecycle: environment interaction, agent updates,
periodic logging, checkpointing, and evaluation.

Usage:
    python -m src.training.train_tabular --agent monte_carlo --episodes 500000
    python -m src.training.train_tabular --agent sarsa --episodes 500000
    python -m src.training.train_tabular --agent q_learning --episodes 500000
"""

import argparse
import logging
import time

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from src.agents.tabular.monte_carlo import MonteCarloAgent
from src.agents.tabular.q_learning import QLearningAgent
from src.agents.tabular.sarsa import SarsaAgent
from src.evaluation.basic_strategy import calculate_strategy_agreement
from src.evaluation.evaluator import evaluate_agent
from src.utils.config import MODELS_DIR, RESULTS_DIR, TabularConfig
from src.utils.logger import MetricsCSVWriter, setup_logger

logger = setup_logger(__name__)

AGENT_MAP = {
    "monte_carlo": MonteCarloAgent,
    "sarsa": SarsaAgent,
    "q_learning": QLearningAgent,
}


def train_monte_carlo(agent: MonteCarloAgent, env: gym.Env, config: TabularConfig) -> None:
    """
    Train a Monte Carlo agent for a specified number of episodes.

    Monte Carlo requires full episode trajectories before updating,
    so we collect the complete (state, action, reward) trajectory first.

    Args:
        agent: MonteCarloAgent instance.
        env: Gymnasium Blackjack-v1 environment.
        config: Training configuration.
    """
    logger.info("Starting Monte Carlo training: %d episodes", config.num_episodes)
    csv_writer = MetricsCSVWriter(
        RESULTS_DIR / "monte_carlo_training.csv",
        ["episode", "avg_reward", "epsilon", "strategy_agreement"],
    )

    reward_window: list[float] = []
    start_time = time.time()

    for episode in tqdm(range(config.num_episodes), desc="Monte Carlo Training", unit="ep"):
        state, _ = env.reset()
        trajectory: list[tuple[tuple[int, int, bool], int, float]] = []
        done = False
        episode_reward = 0.0

        # Collect full episode trajectory
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            trajectory.append((state, action, reward))
            episode_reward += reward
            done = terminated or truncated
            state = next_state

        # Update Q-table with complete episode
        agent.update(trajectory)
        agent.episode_count += 1
        agent.training_rewards.append(episode_reward)
        reward_window.append(episode_reward)

        # Periodic logging
        if (episode + 1) % config.log_interval == 0:
            avg_reward = np.mean(reward_window[-config.log_interval:])
            epsilon = config.get_epsilon(episode)
            agreement = calculate_strategy_agreement(agent.get_policy())

            logger.info(
                "Episode %d/%d | Avg Reward: %.4f | Epsilon: %.4f | "
                "Strategy Agreement: %.1f%%",
                episode + 1, config.num_episodes, avg_reward, epsilon,
                agreement["total_agreement"],
            )
            csv_writer.write_row({
                "episode": episode + 1,
                "avg_reward": f"{avg_reward:.4f}",
                "epsilon": f"{epsilon:.4f}",
                "strategy_agreement": f"{agreement['total_agreement']:.1f}",
            })

        # Periodic checkpointing
        if (episode + 1) % config.checkpoint_interval == 0:
            checkpoint_path = MODELS_DIR / f"monte_carlo_ep{episode + 1}.pkl"
            agent.save(checkpoint_path)

    elapsed = time.time() - start_time
    logger.info(
        "Monte Carlo training complete: %d episodes in %.1f seconds (%.0f ep/s)",
        config.num_episodes, elapsed, config.num_episodes / elapsed,
    )


def train_sarsa(agent: SarsaAgent, env: gym.Env, config: TabularConfig) -> None:
    """
    Train a SARSA agent for a specified number of episodes.

    SARSA is on-policy TD(0): it updates after every step using the
    action actually chosen in the next state.

    Args:
        agent: SarsaAgent instance.
        env: Gymnasium Blackjack-v1 environment.
        config: Training configuration.
    """
    logger.info("Starting SARSA training: %d episodes", config.num_episodes)
    csv_writer = MetricsCSVWriter(
        RESULTS_DIR / "sarsa_training.csv",
        ["episode", "avg_reward", "epsilon", "strategy_agreement"],
    )

    reward_window: list[float] = []
    start_time = time.time()

    for episode in tqdm(range(config.num_episodes), desc="SARSA Training", unit="ep"):
        state, _ = env.reset()
        action = agent.get_action(state)
        done = False
        episode_reward = 0.0

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            if not done:
                next_action = agent.get_action(next_state)
            else:
                next_action = 0  # Terminal, action doesn't matter

            # SARSA update: uses actual next_action (on-policy)
            agent.update(state, action, reward, next_state, next_action, done)
            state = next_state
            action = next_action

        agent.episode_count += 1
        agent.training_rewards.append(episode_reward)
        reward_window.append(episode_reward)

        # Periodic logging
        if (episode + 1) % config.log_interval == 0:
            avg_reward = np.mean(reward_window[-config.log_interval:])
            epsilon = config.get_epsilon(episode)
            agreement = calculate_strategy_agreement(agent.get_policy())

            logger.info(
                "Episode %d/%d | Avg Reward: %.4f | Epsilon: %.4f | "
                "Strategy Agreement: %.1f%%",
                episode + 1, config.num_episodes, avg_reward, epsilon,
                agreement["total_agreement"],
            )
            csv_writer.write_row({
                "episode": episode + 1,
                "avg_reward": f"{avg_reward:.4f}",
                "epsilon": f"{epsilon:.4f}",
                "strategy_agreement": f"{agreement['total_agreement']:.1f}",
            })

        if (episode + 1) % config.checkpoint_interval == 0:
            agent.save(MODELS_DIR / f"sarsa_ep{episode + 1}.pkl")

    elapsed = time.time() - start_time
    logger.info(
        "SARSA training complete: %d episodes in %.1f seconds (%.0f ep/s)",
        config.num_episodes, elapsed, config.num_episodes / elapsed,
    )


def train_q_learning(agent: QLearningAgent, env: gym.Env, config: TabularConfig) -> None:
    """
    Train a Q-Learning agent for a specified number of episodes.

    Q-Learning is off-policy TD(0): it updates using the maximum future
    Q-value regardless of the action actually taken.

    Args:
        agent: QLearningAgent instance.
        env: Gymnasium Blackjack-v1 environment.
        config: Training configuration.
    """
    logger.info("Starting Q-Learning training: %d episodes", config.num_episodes)
    csv_writer = MetricsCSVWriter(
        RESULTS_DIR / "q_learning_training.csv",
        ["episode", "avg_reward", "epsilon", "strategy_agreement"],
    )

    reward_window: list[float] = []
    start_time = time.time()

    for episode in tqdm(range(config.num_episodes), desc="Q-Learning Training", unit="ep"):
        state, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # Q-learning update: uses max Q(s', a') — off-policy
            agent.update(state, action, reward, next_state, done)
            state = next_state

        agent.episode_count += 1
        agent.training_rewards.append(episode_reward)
        reward_window.append(episode_reward)

        # Periodic logging
        if (episode + 1) % config.log_interval == 0:
            avg_reward = np.mean(reward_window[-config.log_interval:])
            epsilon = config.get_epsilon(episode)
            agreement = calculate_strategy_agreement(agent.get_policy())

            logger.info(
                "Episode %d/%d | Avg Reward: %.4f | Epsilon: %.4f | "
                "Strategy Agreement: %.1f%%",
                episode + 1, config.num_episodes, avg_reward, epsilon,
                agreement["total_agreement"],
            )
            csv_writer.write_row({
                "episode": episode + 1,
                "avg_reward": f"{avg_reward:.4f}",
                "epsilon": f"{epsilon:.4f}",
                "strategy_agreement": f"{agreement['total_agreement']:.1f}",
            })

        if (episode + 1) % config.checkpoint_interval == 0:
            agent.save(MODELS_DIR / f"q_learning_ep{episode + 1}.pkl")

    elapsed = time.time() - start_time
    logger.info(
        "Q-Learning training complete: %d episodes in %.1f seconds (%.0f ep/s)",
        config.num_episodes, elapsed, config.num_episodes / elapsed,
    )


def main() -> None:
    """Entry point for tabular agent training."""
    parser = argparse.ArgumentParser(description="Train tabular RL agents for Blackjack")
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=list(AGENT_MAP.keys()),
        help="Agent type to train",
    )
    parser.add_argument("--episodes", type=int, default=500_000, help="Number of training episodes")
    parser.add_argument("--eval-episodes", type=int, default=100_000, help="Number of evaluation episodes")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (alpha)")
    parser.add_argument("--no-eval", action="store_true", help="Skip post-training evaluation")
    args = parser.parse_args()

    logger.info("="*60)
    logger.info("Blackjack AI — Tabular RL Training")
    logger.info("="*60)
    logger.info("Agent: %s | Episodes: %d | LR: %.4f", args.agent, args.episodes, args.lr)

    # Configure
    config = TabularConfig(
        num_episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        learning_rate=args.lr,
    )

    # Create environment
    env = gym.make("Blackjack-v1", natural=config.use_natural_blackjack, sab=config.use_sab)

    # Create and train agent
    agent_class = AGENT_MAP[args.agent]
    agent = agent_class(config=config)

    train_fn_map = {
        "monte_carlo": train_monte_carlo,
        "sarsa": train_sarsa,
        "q_learning": train_q_learning,
    }
    train_fn_map[args.agent](agent, env, config)

    # Save final model
    final_path = MODELS_DIR / f"{args.agent}_final.pkl"
    agent.save(final_path)

    # Post-training evaluation
    if not args.no_eval:
        logger.info("Running post-training evaluation with %d hands...", config.eval_episodes)
        result = evaluate_agent(agent, num_hands=config.eval_episodes)
        logger.info(result.summary())

        # Compare against Basic Strategy
        agreement = calculate_strategy_agreement(agent.get_policy())
        logger.info(
            "Strategy Agreement — Hard: %.1f%%, Soft: %.1f%%, Total: %.1f%%",
            agreement["hard_agreement"],
            agreement["soft_agreement"],
            agreement["total_agreement"],
        )

    env.close()
    logger.info("Training pipeline complete.")


if __name__ == "__main__":
    main()
