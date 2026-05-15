"""
Interactive play mode — play Blackjack against any trained agent.

Watch the AI make decisions in real-time and compare against
Basic Strategy. See what the agent would do in every situation.

Usage:
    python -m src.evaluation.play --agent q_learning
    python -m src.evaluation.play --agent dueling_dqn
    python -m src.evaluation.play --agent basic_strategy
"""

import argparse
import logging

import gymnasium as gym
import numpy as np

from src.agents.tabular.monte_carlo import MonteCarloAgent
from src.agents.tabular.sarsa import SarsaAgent
from src.agents.tabular.q_learning import QLearningAgent
from src.agents.deep.dqn import DQNAgent
from src.agents.deep.double_dqn import DoubleDQNAgent
from src.agents.deep.dueling_dqn import DuelingDQNAgent
from src.evaluation.basic_strategy import get_basic_strategy_action
from src.evaluation.evaluator import BasicStrategyAgent
from src.utils.config import MODELS_DIR, DQNConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

CARD_NAMES = {1: "A", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "10"}
ACTION_NAMES = {0: "STAND", 1: "HIT"}

AGENT_CONFIGS = {
    "basic_strategy": {"type": "basic", "path": None},
    "monte_carlo": {"type": "tabular", "cls": MonteCarloAgent, "path": "monte_carlo_final.pkl"},
    "sarsa": {"type": "tabular", "cls": SarsaAgent, "path": "sarsa_final.pkl"},
    "q_learning": {"type": "tabular", "cls": QLearningAgent, "path": "q_learning_final.pkl"},
    "dqn": {"type": "dqn", "cls": DQNAgent, "path": "dqn_final.pt"},
    "double_dqn": {"type": "dqn", "cls": DoubleDQNAgent, "path": "double_dqn_final.pt"},
    "dueling_dqn": {"type": "dqn", "cls": DuelingDQNAgent, "path": "dueling_dqn_final.pt"},
}


def load_agent(agent_key: str):
    """Load a trained agent by key."""
    cfg = AGENT_CONFIGS[agent_key]

    if cfg["type"] == "basic":
        return BasicStrategyAgent()

    if cfg["type"] == "tabular":
        agent = cfg["cls"]()
        agent.load(MODELS_DIR / cfg["path"])
        return agent

    if cfg["type"] == "dqn":
        import torch
        ckpt = torch.load(MODELS_DIR / cfg["path"], map_location="cpu", weights_only=False)
        config = ckpt.get("config", DQNConfig())
        config.device = "cpu"
        agent = cfg["cls"](config=config)
        agent.load(MODELS_DIR / cfg["path"])
        return agent

    raise ValueError(f"Unknown agent type: {cfg['type']}")


def play_hands(agent, agent_name: str, num_hands: int = 20) -> None:
    """Play N hands with detailed output showing AI decisions."""
    env = gym.make("Blackjack-v1", natural=True, sab=True)

    wins, losses, draws = 0, 0, 0
    total_reward = 0.0

    print()
    print("=" * 60)
    print(f"  🃏 BLACKJACK — Playing with {agent_name}")
    print(f"  {num_hands} hands | Infinite deck | Natural BJ pays 1.5x")
    print("=" * 60)

    for hand_num in range(1, num_hands + 1):
        state, _ = env.reset()
        player_sum, dealer_card, usable_ace = state

        print(f"\n  ── Hand {hand_num} ──")
        print(f"  Your hand: {player_sum}" + (" (soft)" if usable_ace else " (hard)"))
        print(f"  Dealer shows: {CARD_NAMES.get(dealer_card, str(dealer_card))}")

        done = False
        step = 0
        while not done:
            # Get agent's decision
            action = agent.get_action(state, greedy=True)
            agent_action_name = ACTION_NAMES[action]

            # What would Basic Strategy do?
            ps, dc, ua = state
            bs_action = get_basic_strategy_action(ps, dc, ua)
            bs_action_name = ACTION_NAMES[bs_action]

            agree = "✓" if action == bs_action else "✗ DISAGREE"

            print(f"    → AI: {agent_action_name}  |  Basic Strategy: {bs_action_name}  [{agree}]")

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if not done:
                ns_sum, ns_dc, ns_ua = next_state
                print(f"    → New hand value: {ns_sum}" + (" (soft)" if ns_ua else ""))
                state = next_state
            step += 1

        total_reward += reward

        if reward > 0:
            result = "🟢 WIN" if reward == 1.0 else f"🟢 NATURAL BJ (+{reward})"
            wins += 1
        elif reward < 0:
            result = "🔴 LOSE"
            losses += 1
        else:
            result = "⚪ PUSH"
            draws += 1

        print(f"    → Result: {result}")

    env.close()

    # Summary
    print()
    print("=" * 60)
    print(f"  📊 SESSION SUMMARY — {agent_name}")
    print("=" * 60)
    print(f"  Hands played: {num_hands}")
    print(f"  Wins: {wins} ({wins/num_hands*100:.0f}%)")
    print(f"  Losses: {losses} ({losses/num_hands*100:.0f}%)")
    print(f"  Draws: {draws} ({draws/num_hands*100:.0f}%)")
    print(f"  Net reward: {total_reward:+.1f}")
    print(f"  Avg reward/hand: {total_reward/num_hands:+.4f}")
    print("=" * 60)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Blackjack with a trained AI agent")
    parser.add_argument(
        "--agent",
        type=str,
        default="dueling_dqn",
        choices=list(AGENT_CONFIGS.keys()),
        help="Agent to play with (default: dueling_dqn — the best performer)",
    )
    parser.add_argument("--hands", type=int, default=20, help="Number of hands to play")
    args = parser.parse_args()

    agent = load_agent(args.agent)
    play_hands(agent, args.agent.replace("_", " ").title(), args.hands)


if __name__ == "__main__":
    main()
