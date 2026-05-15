"""
Curriculum Learning orchestrator for Blackjack.

Implements phased action-space masking to accelerate PPO convergence:
  Phase 1 (0–2M steps):   Hit/Stand only → learn basic hand evaluation
  Phase 2 (2M–5M steps):  + Double Down  → learn high-leverage situations
  Phase 3 (5M–10M steps): + Split/Surrender → fine-tune edge cases

This approach elevated DQN win rate from 43.97% to 47.41% and reduced
bust rate from 32.9% to 28.0% in academic benchmarking.

Reference: arXiv 2604.00076v2 (2026), "Learning to Play Blackjack: A Curriculum Perspective"

Usage:
    python -m src.training.curriculum --timesteps 10000000
"""

import argparse
import logging
import time

from src.agents.deep.ppo_agent import PPOBlackjackAgent
from src.training.train_ppo import evaluate_ppo_finite_deck
from src.utils.config import (
    LOGS_DIR,
    MODELS_DIR,
    CurriculumConfig,
    FiniteDeckConfig,
    PPOConfig,
)
from src.utils.logger import setup_logger

# Register environments
import src.environments.finite_deck_env  # noqa: F401
import src.environments.full_casino_env  # noqa: F401

logger = setup_logger(__name__)


PHASE_ACTIONS = {
    1: [0, 1],        # Stand, Hit
    2: [0, 1, 2],     # + Double
    3: [0, 1, 2, 3, 4],  # + Split, Surrender
}


def train_with_curriculum(
    config: CurriculumConfig,
    num_decks: int = 1,
    device: str = "auto",
) -> PPOBlackjackAgent:
    """
    Execute the three-phase curriculum training pipeline.

    Args:
        config: Curriculum phase boundaries and PPO config.
        num_decks: Number of decks in the shoe.
        device: Training device.

    Returns:
        Trained PPO agent (final phase).
    """
    env_config = FiniteDeckConfig(num_decks=num_decks)
    ppo_config = config.ppo_config

    total_start = time.time()
    agent = None

    phases = [
        (1, config.phase1_end, "Hit/Stand only"),
        (2, config.phase2_end - config.phase1_end, "+ Double Down"),
        (3, config.phase3_end - config.phase2_end, "+ Split/Surrender"),
    ]

    for phase_num, phase_steps, description in phases:
        logger.info("=" * 60)
        logger.info("CURRICULUM PHASE %d: %s (%d steps)", phase_num, description, phase_steps)
        logger.info("=" * 60)
        logger.info("Allowed actions: %s", PHASE_ACTIONS[phase_num])

        phase_start = time.time()

        # Create environment with action mask for this phase
        env_kwargs = {
            "config": env_config,
            "allowed_actions": PHASE_ACTIONS[phase_num],
        }

        if agent is None:
            # First phase: create fresh agent
            agent = PPOBlackjackAgent(
                env_id="BlackjackFullCasino-v0",
                config=PPOConfig(
                    total_timesteps=phase_steps,
                    learning_rate=ppo_config.learning_rate,
                    n_steps=ppo_config.n_steps,
                    batch_size=ppo_config.batch_size,
                    n_epochs=ppo_config.n_epochs,
                    gamma=ppo_config.gamma,
                    clip_range=ppo_config.clip_range,
                    ent_coef=ppo_config.ent_coef,
                    net_arch=ppo_config.net_arch,
                    device=device,
                ),
                env_kwargs=env_kwargs,
            )
        else:
            # Subsequent phases: transfer weights to new env with expanded actions
            previous_model_path = MODELS_DIR / f"curriculum_phase{phase_num - 1}"
            agent.save(previous_model_path)

            agent = PPOBlackjackAgent(
                env_id="BlackjackFullCasino-v0",
                config=PPOConfig(
                    total_timesteps=phase_steps,
                    learning_rate=ppo_config.learning_rate,
                    n_steps=ppo_config.n_steps,
                    batch_size=ppo_config.batch_size,
                    n_epochs=ppo_config.n_epochs,
                    gamma=ppo_config.gamma,
                    clip_range=ppo_config.clip_range,
                    ent_coef=ppo_config.ent_coef,
                    net_arch=ppo_config.net_arch,
                    device=device,
                ),
                env_kwargs=env_kwargs,
            )
            # Load previous weights for transfer
            try:
                agent.load(previous_model_path)
                logger.info("Loaded Phase %d weights for transfer learning", phase_num - 1)
            except Exception as e:
                logger.warning("Could not load previous weights: %s. Starting fresh.", e)

        # Train this phase
        agent.train(
            total_timesteps=phase_steps,
            tb_log_dir=str(LOGS_DIR / f"tensorboard/curriculum_phase{phase_num}"),
            checkpoint_dir=str(MODELS_DIR / f"curriculum_phase{phase_num}"),
        )

        phase_elapsed = time.time() - phase_start
        logger.info(
            "Phase %d complete in %.1f seconds (%.1f min)",
            phase_num, phase_elapsed, phase_elapsed / 60,
        )

        # Save phase checkpoint
        agent.save(MODELS_DIR / f"curriculum_phase{phase_num}")

    total_elapsed = time.time() - total_start
    logger.info(
        "Curriculum training complete: total %.1f seconds (%.1f min)",
        total_elapsed, total_elapsed / 60,
    )

    return agent


def main() -> None:
    """Entry point for curriculum learning training."""
    parser = argparse.ArgumentParser(description="Train PPO with Curriculum Learning")
    parser.add_argument("--timesteps", type=int, default=10_000_000, help="Total timesteps")
    parser.add_argument("--decks", type=int, default=1, help="Number of decks")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--eval-hands", type=int, default=100_000, help="Eval hands")
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Blackjack AI — Curriculum Learning Pipeline")
    logger.info("=" * 60)

    # Configure phases proportionally
    total = args.timesteps
    config = CurriculumConfig(
        phase1_end=int(total * 0.2),      # 20% on basics
        phase2_end=int(total * 0.5),      # 30% adding doubles
        phase3_end=total,                  # 50% full action space
    )

    agent = train_with_curriculum(config, num_decks=args.decks, device=args.device)

    # Save final model
    agent.save(MODELS_DIR / "curriculum_final")

    # Evaluate
    if not args.no_eval:
        result = evaluate_ppo_finite_deck(
            agent, num_hands=args.eval_hands, num_decks=args.decks,
        )
        logger.info(result.summary())

    logger.info("Curriculum pipeline complete.")


if __name__ == "__main__":
    main()
