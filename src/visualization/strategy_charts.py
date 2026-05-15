"""
Strategy chart visualization for comparing learned policies against Basic Strategy.

Generates publication-quality heatmaps showing agent decisions (Hit/Stand)
across the full state space, with color-coded agreement/disagreement highlighting.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.evaluation.basic_strategy import get_basic_strategy_action

logger = logging.getLogger(__name__)

# Color scheme for actions
ACTION_COLORS = {
    0: "#2ecc71",  # Stand — green
    1: "#e74c3c",  # Hit — red
}
ACTION_LABELS = {0: "S", 1: "H"}


def plot_strategy_heatmap(
    policy: dict[tuple[int, int, bool], int],
    agent_name: str = "Agent",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Generate side-by-side strategy heatmaps: Hard Hands and Soft Hands.

    Each cell shows the agent's decision (H=Hit, S=Stand), color-coded
    to match or diverge from optimal Basic Strategy.

    Args:
        policy: Dictionary mapping (player_sum, dealer_card, usable_ace) → action.
        agent_name: Name for the plot title.
        save_path: If provided, save figure to this path.
        show: If True, display the figure.

    Returns:
        Matplotlib Figure object.
    """
    logger.info("Generating strategy heatmap for: %s", agent_name)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f"{agent_name} — Learned Strategy vs. Basic Strategy", fontsize=16, fontweight="bold")

    dealer_cards = list(range(1, 11))
    dealer_labels = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

    for idx, (usable_ace, title) in enumerate([(False, "Hard Hands"), (True, "Soft Hands")]):
        ax = axes[idx]

        if usable_ace:
            player_sums = list(range(12, 22))  # Soft hands: 12-21
        else:
            player_sums = list(range(4, 22))   # Hard hands: 4-21

        # Build matrices for agent decisions and agreement with optimal
        agent_matrix = np.zeros((len(player_sums), len(dealer_cards)))
        agreement_matrix = np.zeros((len(player_sums), len(dealer_cards)))
        annotations = []

        for i, ps in enumerate(player_sums):
            row_annotations = []
            for j, dc in enumerate(dealer_cards):
                state = (ps, dc, usable_ace)
                agent_action = policy.get(state, 0)
                optimal_action = get_basic_strategy_action(ps, dc, usable_ace)

                agent_matrix[i, j] = agent_action
                agreement_matrix[i, j] = 1 if agent_action == optimal_action else -1

                label = ACTION_LABELS[agent_action]
                if agent_action != optimal_action:
                    label += "✗"  # Mark disagreements
                row_annotations.append(label)
            annotations.append(row_annotations)

        # Create heatmap: green=Stand, red=Hit, with disagreements highlighted
        cmap = sns.color_palette(["#2ecc71", "#e74c3c"], as_cmap=True)
        sns.heatmap(
            agent_matrix,
            ax=ax,
            cmap=cmap,
            xticklabels=dealer_labels,
            yticklabels=player_sums,
            annot=np.array(annotations, dtype=object),
            fmt="",
            linewidths=0.5,
            linecolor="white",
            cbar=False,
            vmin=0,
            vmax=1,
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Dealer Showing", fontsize=12)
        ax.set_ylabel("Player Sum", fontsize=12)
        ax.invert_yaxis()

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Stand (S)"),
        Patch(facecolor="#e74c3c", label="Hit (H)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=12)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Strategy heatmap saved to: %s", save_path)

    if show:
        plt.show()

    return fig


def plot_strategy_comparison(
    policies: dict[str, dict[tuple[int, int, bool], int]],
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Compare multiple agent strategies in a single figure.

    Shows hard-hand strategy for each agent in a grid layout.

    Args:
        policies: Dictionary mapping agent_name → policy dict.
        save_path: If provided, save figure to this path.
        show: If True, display the figure.

    Returns:
        Matplotlib Figure object.
    """
    n_agents = len(policies)
    fig, axes = plt.subplots(1, n_agents, figsize=(8 * n_agents, 10))
    if n_agents == 1:
        axes = [axes]

    fig.suptitle("Strategy Comparison — Hard Hands", fontsize=16, fontweight="bold")

    dealer_cards = list(range(1, 11))
    dealer_labels = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    player_sums = list(range(4, 22))

    for ax, (agent_name, policy) in zip(axes, policies.items()):
        agent_matrix = np.zeros((len(player_sums), len(dealer_cards)))
        annotations = []

        for i, ps in enumerate(player_sums):
            row_annotations = []
            for j, dc in enumerate(dealer_cards):
                state = (ps, dc, False)
                action = policy.get(state, 0)
                agent_matrix[i, j] = action
                row_annotations.append(ACTION_LABELS[action])
            annotations.append(row_annotations)

        cmap = sns.color_palette(["#2ecc71", "#e74c3c"], as_cmap=True)
        sns.heatmap(
            agent_matrix,
            ax=ax,
            cmap=cmap,
            xticklabels=dealer_labels,
            yticklabels=player_sums,
            annot=np.array(annotations, dtype=object),
            fmt="",
            linewidths=0.5,
            linecolor="white",
            cbar=False,
            vmin=0,
            vmax=1,
        )
        ax.set_title(agent_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Dealer Showing", fontsize=11)
        ax.set_ylabel("Player Sum", fontsize=11)
        ax.invert_yaxis()

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Strategy comparison saved to: %s", save_path)

    if show:
        plt.show()

    return fig
