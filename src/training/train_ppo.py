"""
PPO training pipeline for finite-deck Blackjack card counting.

Trains a PPO agent on the custom FiniteDeckBlackjackEnv with the extended
13-dimensional state space. The agent learns to exploit deck composition
information to achieve positive expected value — effectively discovering
card counting principles autonomously.

Usage:
    python -m src.training.train_ppo --timesteps 10000000
    python -m src.training.train_ppo --timesteps 1000000 --decks 1 --device cpu
"""

import argparse
import logging
import time

import numpy as np
from tqdm import tqdm

# Import to register the custom environment
import src.environments.finite_deck_env  # noqa: F401
from src.agents.deep.ppo_agent import PPOBlackjackAgent
from src.environments.finite_deck_env import FiniteDeckBlackjackEnv
from src.evaluation.metrics import EvaluationResult
from src.utils.config import (
    LOGS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    FiniteDeckConfig,
    PPOConfig,
)
from src.utils.logger import MetricsCSVWriter, setup_logger

logger = setup_logger(__name__)


def evaluate_ppo_finite_deck(
    agent: PPOBlackjackAgent,
    num_hands: int = 100_000,
    num_decks: int = 1,
    seed: int = 42,
) -> EvaluationResult:
    """
    Evaluate PPO agent on the finite-deck environment.

    Unlike the standard evaluator, this uses the custom environment
    with the extended state space.

    Args:
        agent: PPO agent to evaluate.
        num_hands: Number of hands to play.
        num_decks: Number of decks in the shoe.
        seed: Random seed.

    Returns:
        EvaluationResult with performance metrics.
    """
    logger.info(
        "Evaluating PPO on finite-deck env: hands=%d, decks=%d",
        num_hands, num_decks,
    )

    env = FiniteDeckBlackjackEnv(
        config=FiniteDeckConfig(num_decks=num_decks)
    )

    wins = 0
    losses = 0
    draws = 0
    busts = 0
    naturals = 0
    total_reward = 0.0

    for i in tqdm(range(num_hands), desc="Evaluating PPO", unit="hands"):
        obs, info = env.reset(seed=seed + i if seed else None)
        done = False
        episode_reward = 0.0

        # Check for natural Blackjack
        if info.get("natural_blackjack"):
            episode_reward = 1.5
            naturals += 1
            wins += 1
            total_reward += episode_reward
            continue
        elif info.get("natural_push"):
            draws += 1
            continue

        while not done:
            action = agent.get_action(obs, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_reward += episode_reward

        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1
            if info.get("result") == "bust":
                busts += 1
        else:
            draws += 1

    return EvaluationResult(
        agent_name=f"PPO ({num_decks}-deck)",
        num_hands=num_hands,
        wins=wins,
        losses=losses,
        draws=draws,
        busts=busts,
        natural_blackjacks=naturals,
        total_reward=total_reward,
    )


def main() -> None:
    """Entry point for PPO training pipeline."""
    parser = argparse.ArgumentParser(description="Train PPO agent for card counting")
    parser.add_argument("--timesteps", type=int, default=10_000_000, help="Total training timesteps")
    parser.add_argument("--decks", type=int, default=1, help="Number of decks in shoe")
    parser.add_argument("--penetration", type=float, default=0.75, help="Shoe penetration threshold")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--eval-hands", type=int, default=100_000, help="Evaluation hands")
    parser.add_argument("--no-eval", action="store_true", help="Skip post-training evaluation")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Blackjack AI — PPO Card Counting Training")
    logger.info("=" * 60)
    logger.info(
        "Decks: %d | Penetration: %.0f%% | Timesteps: %d | LR: %.1e | Device: %s",
        args.decks, args.penetration * 100, args.timesteps, args.lr, args.device,
    )

    # Configure environment
    env_config = FiniteDeckConfig(
        num_decks=args.decks,
        penetration=args.penetration,
    )

    # Configure PPO
    ppo_config = PPOConfig(
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        device=args.device,
    )

    # Create agent
    agent = PPOBlackjackAgent(
        env_id="BlackjackFiniteDeck-v0",
        config=ppo_config,
        env_kwargs={"config": env_config},
    )

    # Train
    start_time = time.time()
    agent.train(
        total_timesteps=args.timesteps,
        tb_log_dir=str(LOGS_DIR / "tensorboard/ppo"),
        checkpoint_dir=str(MODELS_DIR / "ppo_checkpoints"),
        eval_env_id="BlackjackFiniteDeck-v0",
        eval_freq=ppo_config.eval_freq,
    )
    elapsed = time.time() - start_time
    logger.info("PPO training completed in %.1f seconds (%.1f min)", elapsed, elapsed / 60)

    # Save final model
    agent.save(MODELS_DIR / "ppo_final")
    logger.info("Final PPO model saved.")

    # Evaluate
    if not args.no_eval:
        result = evaluate_ppo_finite_deck(
            agent,
            num_hands=args.eval_hands,
            num_decks=args.decks,
        )
        logger.info(result.summary())

        if result.avg_reward > 0:
            logger.info(
                "🎰 POSITIVE EXPECTED VALUE ACHIEVED! Avg reward: +%.4f — "
                "The agent has discovered card counting.",
                result.avg_reward,
            )
        else:
            logger.warning(
                "Negative expected value (%.4f). Agent may need more training "
                "or hyperparameter tuning.",
                result.avg_reward,
            )

    logger.info("PPO pipeline complete.")


if __name__ == "__main__":
    main()
