"""
Training convergence and performance plots.

Generates publication-quality charts for analyzing agent training dynamics:
convergence curves, reward distributions, Q-value surfaces, and comparative
bar charts across all algorithms.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.evaluation.metrics import EvaluationResult

logger = logging.getLogger(__name__)

# Style configuration
sns.set_theme(style="darkgrid", palette="deep")
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#e0e0e0",
    "grid.color": "#2a2a4a",
    "font.size": 12,
})

AGENT_COLORS = {
    "Monte Carlo": "#e74c3c",
    "SARSA": "#3498db",
    "Q-Learning": "#2ecc71",
    "DQN": "#f39c12",
    "Double DQN": "#9b59b6",
    "Dueling DQN": "#e67e22",
    "PPO": "#1abc9c",
    "PPO + Curriculum": "#f1c40f",
    "Basic Strategy": "#95a5a6",
}


def plot_convergence_curves(
    training_csvs: dict[str, str | Path],
    save_path: str | Path | None = None,
    show: bool = True,
    window_size: int = 5000,
) -> plt.Figure:
    """
    Plot training convergence curves (avg reward vs. episode) for multiple agents.

    Args:
        training_csvs: Dict mapping agent_name → CSV file path.
        save_path: If provided, save figure to this path.
        show: If True, display the figure.
        window_size: Smoothing window for rolling average.

    Returns:
        Matplotlib Figure object.
    """
    logger.info("Generating convergence curves from %d CSV files", len(training_csvs))

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_title("Training Convergence — Average Reward per Episode", fontsize=15, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=13)
    ax.set_ylabel("Average Reward (Rolling Window)", fontsize=13)

    for agent_name, csv_path in training_csvs.items():
        try:
            df = pd.read_csv(csv_path)
            color = AGENT_COLORS.get(agent_name, "#ffffff")
            ax.plot(
                df["episode"],
                df["avg_reward"].astype(float),
                label=agent_name,
                color=color,
                linewidth=2,
                alpha=0.85,
            )
        except Exception as e:
            logger.error("Failed to load CSV for %s: %s", agent_name, e)

    ax.axhline(y=0, color="#ffffff", linestyle="--", alpha=0.3, label="Break Even")
    ax.legend(fontsize=11, loc="lower right", framealpha=0.8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info("Convergence plot saved to: %s", save_path)

    if show:
        plt.show()

    return fig


def plot_evaluation_comparison(
    results: list[EvaluationResult],
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Bar chart comparing evaluation metrics across multiple agents.

    Args:
        results: List of EvaluationResult objects from different agents.
        save_path: If provided, save figure to this path.
        show: If True, display the figure.

    Returns:
        Matplotlib Figure object.
    """
    logger.info("Generating evaluation comparison for %d agents", len(results))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Agent Performance Comparison", fontsize=16, fontweight="bold")

    agent_names = [r.agent_name for r in results]
    colors = [AGENT_COLORS.get(n.split("(")[0].strip(), "#ffffff") for n in agent_names]

    # Win Rate
    win_rates = [r.win_rate for r in results]
    axes[0].barh(agent_names, win_rates, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_title("Win Rate (%)", fontsize=13)
    axes[0].set_xlim(35, 50)
    for i, v in enumerate(win_rates):
        axes[0].text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=10)

    # Bust Rate
    bust_rates = [r.bust_rate for r in results]
    axes[1].barh(agent_names, bust_rates, color=colors, edgecolor="white", linewidth=0.5)
    axes[1].set_title("Bust Rate (%)", fontsize=13)
    for i, v in enumerate(bust_rates):
        axes[1].text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=10)

    # Average Reward
    avg_rewards = [r.avg_reward for r in results]
    bar_colors = ["#2ecc71" if r > 0 else "#e74c3c" for r in avg_rewards]
    axes[2].barh(agent_names, avg_rewards, color=bar_colors, edgecolor="white", linewidth=0.5)
    axes[2].set_title("Avg Reward per Hand", fontsize=13)
    axes[2].axvline(x=0, color="white", linestyle="--", alpha=0.5)
    for i, v in enumerate(avg_rewards):
        axes[2].text(v + 0.002 if v >= 0 else v - 0.015, i, f"{v:.4f}", va="center", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info("Evaluation comparison saved to: %s", save_path)

    if show:
        plt.show()

    return fig


def plot_strategy_agreement_evolution(
    training_csv: str | Path,
    agent_name: str = "Agent",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot how strategy agreement with Basic Strategy evolves during training.

    Args:
        training_csv: Path to training CSV with 'episode' and 'strategy_agreement' columns.
        agent_name: Name for the plot title.
        save_path: If provided, save figure to this path.
        show: If True, display the figure.

    Returns:
        Matplotlib Figure object.
    """
    logger.info("Generating strategy agreement evolution for: %s", agent_name)

    df = pd.read_csv(training_csv)
    fig, ax = plt.subplots(figsize=(12, 6))

    color = AGENT_COLORS.get(agent_name, "#3498db")
    ax.plot(
        df["episode"],
        df["strategy_agreement"].astype(float),
        color=color,
        linewidth=2,
    )
    ax.fill_between(
        df["episode"],
        df["strategy_agreement"].astype(float),
        alpha=0.15,
        color=color,
    )

    ax.axhline(y=100, color="#2ecc71", linestyle="--", alpha=0.5, label="Perfect Agreement")
    ax.set_title(f"{agent_name} — Strategy Agreement During Training", fontsize=14, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Agreement with Basic Strategy (%)", fontsize=12)
    ax.set_ylim(40, 105)
    ax.legend(fontsize=11)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info("Strategy agreement plot saved to: %s", save_path)

    if show:
        plt.show()

    return fig
