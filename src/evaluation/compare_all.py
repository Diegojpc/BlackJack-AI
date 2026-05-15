"""
Head-to-head comparison of ALL trained agents.

Loads every final model, evaluates them on 100K hands each,
ranks them, and generates a comparison report.

Usage:
    python -m src.evaluation.compare_all
"""

import logging
import sys

import gymnasium as gym
import numpy as np

# Register custom envs
import src.environments.finite_deck_env  # noqa: F401

from src.agents.tabular.monte_carlo import MonteCarloAgent
from src.agents.tabular.sarsa import SarsaAgent
from src.agents.tabular.q_learning import QLearningAgent
from src.agents.deep.dqn import DQNAgent
from src.agents.deep.double_dqn import DoubleDQNAgent
from src.agents.deep.dueling_dqn import DuelingDQNAgent
from src.agents.deep.ppo_agent import PPOBlackjackAgent
from src.evaluation.basic_strategy import calculate_strategy_agreement
from src.evaluation.evaluator import BasicStrategyAgent, evaluate_agent
from src.evaluation.metrics import EvaluationResult
from src.environments.finite_deck_env import FiniteDeckBlackjackEnv
from src.utils.config import MODELS_DIR, DQNConfig, FiniteDeckConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

EVAL_HANDS = 100_000


def load_and_eval_tabular(agent_class, name: str, path: str) -> tuple:
    """Load a tabular agent and evaluate it."""
    logger.info("Loading %s from %s...", name, path)
    agent = agent_class()
    agent.load(path)

    result = evaluate_agent(agent, num_hands=EVAL_HANDS, show_progress=True)
    agreement = calculate_strategy_agreement(agent.get_policy())
    return result, agreement


def load_and_eval_dqn(agent_class, name: str, path: str) -> tuple:
    """Load a DQN-family agent and evaluate it."""
    logger.info("Loading %s from %s...", name, path)
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", DQNConfig())
    config.device = "cpu"
    agent = agent_class(config=config)
    agent.load(path)

    result = evaluate_agent(agent, num_hands=EVAL_HANDS, show_progress=True)
    agreement = calculate_strategy_agreement(agent.get_policy())
    return result, agreement


def eval_ppo(path: str) -> tuple:
    """Load and evaluate PPO on the finite-deck environment."""
    logger.info("Loading PPO from %s...", path)
    agent = PPOBlackjackAgent(
        env_id="BlackjackFiniteDeck-v0",
        env_kwargs={"config": FiniteDeckConfig(num_decks=1)},
    )
    agent.load(path)

    # Evaluate on finite-deck env
    env = FiniteDeckBlackjackEnv(config=FiniteDeckConfig(num_decks=1))
    wins, losses, draws, busts, naturals = 0, 0, 0, 0, 0
    total_reward = 0.0

    for i in range(EVAL_HANDS):
        obs, info = env.reset(seed=42 + i)
        done = False
        ep_reward = 0.0

        if info.get("natural_blackjack"):
            ep_reward = 1.5
            naturals += 1
            wins += 1
            total_reward += ep_reward
            continue
        elif info.get("natural_push"):
            draws += 1
            continue

        while not done:
            action = agent.get_action(obs, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        total_reward += ep_reward
        if ep_reward > 0:
            wins += 1
        elif ep_reward < 0:
            losses += 1
            if info.get("result") == "bust":
                busts += 1
        else:
            draws += 1

    result = EvaluationResult(
        agent_name="PPO (1-deck Card Counting)",
        num_hands=EVAL_HANDS,
        wins=wins, losses=losses, draws=draws,
        busts=busts, natural_blackjacks=naturals,
        total_reward=total_reward,
    )
    agreement = calculate_strategy_agreement(agent.get_policy())
    return result, agreement


def main() -> None:
    """Run head-to-head comparison of all trained agents."""
    logger.info("=" * 70)
    logger.info("  BLACKJACK AI — FULL MODEL COMPARISON")
    logger.info("  Evaluating all agents on %d hands each", EVAL_HANDS)
    logger.info("=" * 70)

    results: list[tuple[EvaluationResult, dict]] = []

    # 1. Basic Strategy baseline
    logger.info("\n--- Basic Strategy (Reference) ---")
    bs_agent = BasicStrategyAgent()
    bs_result = evaluate_agent(bs_agent, num_hands=EVAL_HANDS, show_progress=True)
    bs_agreement = calculate_strategy_agreement(bs_agent.get_policy())
    results.append((bs_result, bs_agreement))

    # 2. Tabular agents
    tabular_agents = [
        (MonteCarloAgent, "Monte Carlo", str(MODELS_DIR / "monte_carlo_final.pkl")),
        (SarsaAgent, "SARSA", str(MODELS_DIR / "sarsa_final.pkl")),
        (QLearningAgent, "Q-Learning", str(MODELS_DIR / "q_learning_final.pkl")),
    ]
    for cls, name, path in tabular_agents:
        logger.info("\n--- %s ---", name)
        try:
            r, a = load_and_eval_tabular(cls, name, path)
            results.append((r, a))
        except Exception as e:
            logger.error("Failed to evaluate %s: %s", name, e)

    # 3. Deep RL agents
    dqn_agents = [
        (DQNAgent, "DQN", str(MODELS_DIR / "dqn_final.pt")),
        (DoubleDQNAgent, "Double DQN", str(MODELS_DIR / "double_dqn_final.pt")),
        (DuelingDQNAgent, "Dueling DQN", str(MODELS_DIR / "dueling_dqn_final.pt")),
    ]
    for cls, name, path in dqn_agents:
        logger.info("\n--- %s ---", name)
        try:
            r, a = load_and_eval_dqn(cls, name, path)
            results.append((r, a))
        except Exception as e:
            logger.error("Failed to evaluate %s: %s", name, e)

    # 4. PPO
    ppo_path = str(MODELS_DIR / "ppo_final")
    logger.info("\n--- PPO (Card Counting) ---")
    try:
        r, a = eval_ppo(ppo_path)
        results.append((r, a))
    except Exception as e:
        logger.error("Failed to evaluate PPO: %s", e)

    # ── FINAL RANKING ──────────────────────────────────────────────────
    print("\n")
    print("=" * 90)
    print("  🏆 FINAL RANKING — ALL AGENTS (sorted by Avg Reward)")
    print("=" * 90)
    print(f"  {'Rank':<5} {'Agent':<30} {'Win%':>8} {'Loss%':>8} {'Draw%':>8} "
          f"{'Bust%':>8} {'AvgReward':>10} {'Strategy%':>10}")
    print("-" * 90)

    # Sort by avg_reward (higher is better)
    ranked = sorted(results, key=lambda x: x[0].avg_reward, reverse=True)

    for rank, (result, agreement) in enumerate(ranked, 1):
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        positive = "✅" if result.avg_reward > 0 else ""
        print(
            f"  {emoji}{rank:<3} {result.agent_name:<30} {result.win_rate:>7.2f}% "
            f"{result.loss_rate:>7.2f}% {result.draw_rate:>7.2f}% "
            f"{result.bust_rate:>7.2f}% {result.avg_reward:>+9.4f} "
            f"{agreement['total_agreement']:>9.1f}% {positive}"
        )

    print("-" * 90)
    print()

    # Highlight winner
    winner_result, winner_agreement = ranked[0]
    print(f"  🏆 BEST AGENT: {winner_result.agent_name}")
    print(f"     Win Rate:            {winner_result.win_rate:.2f}%")
    print(f"     Avg Reward per Hand: {winner_result.avg_reward:+.4f}")
    print(f"     Strategy Agreement:  {winner_agreement['total_agreement']:.1f}%")
    print(f"     Bust Rate:           {winner_result.bust_rate:.2f}%")
    print()

    if winner_result.avg_reward > 0:
        print("  🎰 POSITIVE EXPECTED VALUE! The agent beats the house.")
    else:
        house_edge = abs(winner_result.avg_reward) * 100
        print(f"  📊 House edge: {house_edge:.2f}% — the casino still wins long-term.")
        print("     (This is expected. Even perfect Basic Strategy has ~0.5% house edge.)")
    print()


if __name__ == "__main__":
    main()
