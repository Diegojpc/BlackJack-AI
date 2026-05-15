"""
Training loop for Deep RL agents (DQN, Double DQN, Dueling DQN).

Handles environment interaction, experience collection, periodic training,
target network updates, TensorBoard logging, and evaluation.

Usage:
    python -m src.training.train_deep --agent dqn --timesteps 1000000
    python -m src.training.train_deep --agent double_dqn --timesteps 1000000
    python -m src.training.train_deep --agent dueling_dqn --timesteps 1000000
"""

import argparse
import logging
import time

import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.agents.deep.dqn import DQNAgent
from src.agents.deep.double_dqn import DoubleDQNAgent
from src.agents.deep.dueling_dqn import DuelingDQNAgent
from src.evaluation.basic_strategy import calculate_strategy_agreement
from src.evaluation.evaluator import evaluate_agent
from src.utils.config import LOGS_DIR, MODELS_DIR, RESULTS_DIR, DQNConfig
from src.utils.logger import MetricsCSVWriter, setup_logger

logger = setup_logger(__name__)

AGENT_MAP = {
    "dqn": DQNAgent,
    "double_dqn": DoubleDQNAgent,
    "dueling_dqn": DuelingDQNAgent,
}


def train_dqn_agent(
    agent: DQNAgent,
    config: DQNConfig,
    agent_key: str,
) -> None:
    """
    Unified training loop for all DQN variants.

    The training flow:
    1. Collect transitions via ε-greedy exploration
    2. Store in replay buffer
    3. Sample mini-batches and perform gradient updates
    4. Periodically hard-copy to target network
    5. Log metrics to TensorBoard and CSV

    Args:
        agent: Any DQN-family agent (DQN, Double DQN, Dueling DQN).
        config: DQN hyperparameters.
        agent_key: Agent identifier string for file naming.
    """
    env = gym.make("Blackjack-v1", natural=True, sab=True)
    writer = SummaryWriter(log_dir=str(LOGS_DIR / f"tensorboard/{agent_key}"))
    csv_writer = MetricsCSVWriter(
        RESULTS_DIR / f"{agent_key}_training.csv",
        ["step", "avg_reward", "avg_loss", "epsilon", "strategy_agreement"],
    )

    logger.info(
        "Starting %s training: timesteps=%d, device=%s",
        agent.name, config.total_timesteps, agent.device,
    )

    reward_window: list[float] = []
    loss_window: list[float] = []
    start_time = time.time()

    state, _ = env.reset()
    episode_reward = 0.0
    episode_count = 0

    for step in tqdm(range(config.total_timesteps), desc=f"{agent.name} Training", unit="step"):
        # Select action
        action = agent.get_action(state)

        # Environment step
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        # Store transition
        agent.store_transition(state, action, reward, next_state, done)

        # Train on mini-batch
        loss = agent.train_step()
        if loss is not None:
            loss_window.append(loss)

        # Update target network: soft (Polyak) every step, or hard every N steps
        agent.total_steps = step + 1
        if config.soft_update_tau > 0:
            agent.soft_update_target_network(config.soft_update_tau)
        elif (step + 1) % config.target_update_freq == 0:
            agent.update_target_network()

        # Episode bookkeeping
        if done:
            reward_window.append(episode_reward)
            agent.training_rewards.append(episode_reward)
            episode_count += 1
            episode_reward = 0.0
            state, _ = env.reset()
        else:
            state = next_state

        # Periodic logging
        if (step + 1) % config.log_interval == 0:
            avg_reward = np.mean(reward_window[-1000:]) if reward_window else 0.0
            avg_loss = np.mean(loss_window[-1000:]) if loss_window else 0.0
            epsilon = config.get_epsilon(step)
            agreement = calculate_strategy_agreement(agent.get_policy())

            writer.add_scalar("train/avg_reward", avg_reward, step + 1)
            writer.add_scalar("train/avg_loss", avg_loss, step + 1)
            writer.add_scalar("train/epsilon", epsilon, step + 1)
            writer.add_scalar("train/strategy_agreement", agreement["total_agreement"], step + 1)
            writer.add_scalar("train/episodes", episode_count, step + 1)

            logger.info(
                "Step %d/%d | Episodes: %d | Avg Reward: %.4f | Avg Loss: %.4f | "
                "Epsilon: %.4f | Strategy: %.1f%%",
                step + 1, config.total_timesteps, episode_count,
                avg_reward, avg_loss, epsilon, agreement["total_agreement"],
            )

            csv_writer.write_row({
                "step": step + 1,
                "avg_reward": f"{avg_reward:.4f}",
                "avg_loss": f"{avg_loss:.4f}",
                "epsilon": f"{epsilon:.4f}",
                "strategy_agreement": f"{agreement['total_agreement']:.1f}",
            })

        # Periodic checkpointing
        if (step + 1) % config.checkpoint_interval == 0:
            agent.save(MODELS_DIR / f"{agent_key}_step{step + 1}.pt")

    env.close()
    writer.close()

    elapsed = time.time() - start_time
    logger.info(
        "%s training complete: %d steps in %.1f seconds (%.0f steps/s), %d episodes",
        agent.name, config.total_timesteps, elapsed,
        config.total_timesteps / elapsed, episode_count,
    )


def main() -> None:
    """Entry point for deep RL agent training."""
    parser = argparse.ArgumentParser(description="Train deep RL agents for Blackjack")
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=list(AGENT_MAP.keys()),
        help="Agent type to train",
    )
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--eval-episodes", type=int, default=100_000, help="Post-training eval episodes")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device")
    parser.add_argument("--no-eval", action="store_true", help="Skip post-training evaluation")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Blackjack AI — Deep RL Training")
    logger.info("=" * 60)
    logger.info("Agent: %s | Timesteps: %d | LR: %.1e | Device: %s",
                args.agent, args.timesteps, args.lr, args.device)

    config = DQNConfig(
        total_timesteps=args.timesteps,
        eval_episodes=args.eval_episodes,
        learning_rate=args.lr,
        device=args.device,
    )

    agent_class = AGENT_MAP[args.agent]
    agent = agent_class(config=config)

    train_dqn_agent(agent, config, args.agent)

    # Save final model
    final_path = MODELS_DIR / f"{args.agent}_final.pt"
    agent.save(final_path)

    # Post-training evaluation
    if not args.no_eval:
        logger.info("Running post-training evaluation with %d hands...", config.eval_episodes)
        result = evaluate_agent(agent, num_hands=config.eval_episodes)
        logger.info(result.summary())

        agreement = calculate_strategy_agreement(agent.get_policy())
        logger.info(
            "Strategy Agreement — Hard: %.1f%%, Soft: %.1f%%, Total: %.1f%%",
            agreement["hard_agreement"],
            agreement["soft_agreement"],
            agreement["total_agreement"],
        )

    logger.info("Deep RL training pipeline complete.")


if __name__ == "__main__":
    main()
