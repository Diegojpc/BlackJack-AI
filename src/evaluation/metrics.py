"""
Evaluation metrics and statistical analysis for Blackjack agents.

Provides functions to compute win rates, bust rates, average rewards,
and statistical confidence intervals from evaluation runs.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for standardized evaluation metrics."""

    agent_name: str
    num_hands: int
    wins: int
    losses: int
    draws: int
    busts: int
    natural_blackjacks: int
    total_reward: float

    @property
    def win_rate(self) -> float:
        """Win percentage (wins / total hands)."""
        return self.wins / self.num_hands * 100 if self.num_hands > 0 else 0.0

    @property
    def loss_rate(self) -> float:
        """Loss percentage."""
        return self.losses / self.num_hands * 100 if self.num_hands > 0 else 0.0

    @property
    def draw_rate(self) -> float:
        """Draw percentage."""
        return self.draws / self.num_hands * 100 if self.num_hands > 0 else 0.0

    @property
    def bust_rate(self) -> float:
        """Bust percentage (times agent busted / total hands)."""
        return self.busts / self.num_hands * 100 if self.num_hands > 0 else 0.0

    @property
    def avg_reward(self) -> float:
        """Average reward per hand."""
        return self.total_reward / self.num_hands if self.num_hands > 0 else 0.0

    @property
    def natural_rate(self) -> float:
        """Natural Blackjack percentage."""
        return self.natural_blackjacks / self.num_hands * 100 if self.num_hands > 0 else 0.0

    def summary(self) -> str:
        """Return a formatted summary string of all metrics."""
        return (
            f"\n{'='*60}\n"
            f"  Evaluation Results: {self.agent_name}\n"
            f"{'='*60}\n"
            f"  Hands Played:       {self.num_hands:>10,}\n"
            f"  Win Rate:           {self.win_rate:>10.2f}%\n"
            f"  Loss Rate:          {self.loss_rate:>10.2f}%\n"
            f"  Draw Rate:          {self.draw_rate:>10.2f}%\n"
            f"  Bust Rate:          {self.bust_rate:>10.2f}%\n"
            f"  Natural BJ Rate:    {self.natural_rate:>10.2f}%\n"
            f"  Avg Reward/Hand:    {self.avg_reward:>10.4f}\n"
            f"  Total Reward:       {self.total_reward:>10.2f}\n"
            f"{'='*60}\n"
        )


def compute_confidence_interval(
    rewards: np.ndarray, confidence: float = 0.95
) -> tuple[float, float]:
    """
    Compute the confidence interval for mean reward.

    Args:
        rewards: Array of per-episode rewards.
        confidence: Confidence level (default 95%).

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    n = len(rewards)
    if n < 2:
        logger.warning("Cannot compute CI with fewer than 2 samples. Returning (0, 0).")
        return (0.0, 0.0)

    mean = np.mean(rewards)
    se = np.std(rewards, ddof=1) / np.sqrt(n)

    # Use z-score approximation for large N
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)

    ci_lower = mean - z * se
    ci_upper = mean + z * se

    logger.debug(
        "CI computation: n=%d, mean=%.4f, se=%.6f, z=%.3f, CI=(%.4f, %.4f)",
        n, mean, se, z, ci_lower, ci_upper,
    )

    return (float(ci_lower), float(ci_upper))
