"""
Mathematically optimal Basic Strategy reference chart for Blackjack.

This module encodes the well-known, computationally verified Basic Strategy for
an infinite-deck Blackjack game (dealer stands on soft 17). It serves as the
ground truth for evaluating how closely RL agents approximate optimal play.

The simplified chart covers only Hit/Stand decisions for the standard
Gymnasium Blackjack-v1 state space: (player_sum, dealer_card, usable_ace).

References:
    - Edward O. Thorp, "Beat the Dealer" (1966)
    - Gymnasium Blackjack-v1 tutorial (Farama Foundation)
    - Stanford AA228 Blackjack reports (2020)
"""

import logging

logger = logging.getLogger(__name__)

# Actions
HIT = 1
STAND = 0

# ──────────────────────────────────────────────────────────────────────────────
# Hard Hand Strategy (no usable ace)
# Key: (player_sum, dealer_card) → action
# Dealer card: 1=Ace, 2-10=face value (10 includes J/Q/K)
# ──────────────────────────────────────────────────────────────────────────────
HARD_STRATEGY: dict[tuple[int, int], int] = {}

# Player sum 4-8: Always hit regardless of dealer card
for player_sum in range(4, 9):
    for dealer_card in range(1, 11):
        HARD_STRATEGY[(player_sum, dealer_card)] = HIT

# Player sum 9: Hit against strong dealer (1, 7-10), otherwise hit
# (In basic Hit/Stand only, 9 is always hit since we can't double)
for dealer_card in range(1, 11):
    HARD_STRATEGY[(9, dealer_card)] = HIT

# Player sum 10: Hit (can't double in Hit/Stand only)
for dealer_card in range(1, 11):
    HARD_STRATEGY[(10, dealer_card)] = HIT

# Player sum 11: Hit (would double if available, but Hit/Stand only)
for dealer_card in range(1, 11):
    HARD_STRATEGY[(11, dealer_card)] = HIT

# Player sum 12: Stand against dealer 4-6, Hit otherwise
for dealer_card in range(1, 11):
    if dealer_card in (4, 5, 6):
        HARD_STRATEGY[(12, dealer_card)] = STAND
    else:
        HARD_STRATEGY[(12, dealer_card)] = HIT

# Player sum 13-16: Stand against dealer 2-6, Hit otherwise
for player_sum in range(13, 17):
    for dealer_card in range(1, 11):
        if dealer_card in (2, 3, 4, 5, 6):
            HARD_STRATEGY[(player_sum, dealer_card)] = STAND
        else:
            HARD_STRATEGY[(player_sum, dealer_card)] = HIT

# Player sum 17-21: Always stand
for player_sum in range(17, 22):
    for dealer_card in range(1, 11):
        HARD_STRATEGY[(player_sum, dealer_card)] = STAND


# ──────────────────────────────────────────────────────────────────────────────
# Soft Hand Strategy (usable ace)
# Key: (player_sum, dealer_card) → action
# ──────────────────────────────────────────────────────────────────────────────
SOFT_STRATEGY: dict[tuple[int, int], int] = {}

# Soft 12 (A+A) to Soft 17 (A+6): Always hit
for player_sum in range(12, 18):
    for dealer_card in range(1, 11):
        SOFT_STRATEGY[(player_sum, dealer_card)] = HIT

# Soft 18 (A+7): Stand against dealer 2, 7, 8; Hit against 9, 10, A
for dealer_card in range(1, 11):
    if dealer_card in (9, 10, 1):
        SOFT_STRATEGY[(18, dealer_card)] = HIT
    else:
        SOFT_STRATEGY[(18, dealer_card)] = STAND

# Soft 19-21: Always stand
for player_sum in range(19, 22):
    for dealer_card in range(1, 11):
        SOFT_STRATEGY[(player_sum, dealer_card)] = STAND


def get_basic_strategy_action(
    player_sum: int, dealer_card: int, usable_ace: bool
) -> int:
    """
    Look up the optimal Basic Strategy action for a given state.

    Args:
        player_sum: Total value of the player's hand (4-21).
        dealer_card: Dealer's visible card value (1-10, where 1=Ace).
        usable_ace: Whether the player has a usable ace.

    Returns:
        Optimal action: 0 (Stand) or 1 (Hit).
    """
    key = (player_sum, dealer_card)

    if usable_ace and key in SOFT_STRATEGY:
        return SOFT_STRATEGY[key]
    elif key in HARD_STRATEGY:
        return HARD_STRATEGY[key]
    else:
        # Edge case: player_sum > 21 shouldn't happen, but stand if it does
        logger.warning(
            "State not in strategy chart: player_sum=%d, dealer_card=%d, usable_ace=%s. Defaulting to STAND.",
            player_sum, dealer_card, usable_ace,
        )
        return STAND


def calculate_strategy_agreement(
    learned_policy: dict[tuple[int, int, bool], int],
) -> dict[str, float]:
    """
    Calculate the percentage agreement between a learned policy and Basic Strategy.

    Args:
        learned_policy: Dictionary mapping (player_sum, dealer_card, usable_ace) → action.

    Returns:
        Dictionary with 'hard_agreement', 'soft_agreement', and 'total_agreement' percentages.
    """
    hard_total = 0
    hard_match = 0
    soft_total = 0
    soft_match = 0

    for (player_sum, dealer_card), optimal_action in HARD_STRATEGY.items():
        state = (player_sum, dealer_card, False)
        if state in learned_policy:
            hard_total += 1
            if learned_policy[state] == optimal_action:
                hard_match += 1

    for (player_sum, dealer_card), optimal_action in SOFT_STRATEGY.items():
        state = (player_sum, dealer_card, True)
        if state in learned_policy:
            soft_total += 1
            if learned_policy[state] == optimal_action:
                soft_match += 1

    hard_pct = (hard_match / hard_total * 100) if hard_total > 0 else 0.0
    soft_pct = (soft_match / soft_total * 100) if soft_total > 0 else 0.0
    total_pct = (
        ((hard_match + soft_match) / (hard_total + soft_total) * 100)
        if (hard_total + soft_total) > 0
        else 0.0
    )

    logger.info(
        "Strategy agreement: hard=%.1f%% (%d/%d), soft=%.1f%% (%d/%d), total=%.1f%%",
        hard_pct, hard_match, hard_total,
        soft_pct, soft_match, soft_total,
        total_pct,
    )

    return {
        "hard_agreement": hard_pct,
        "soft_agreement": soft_pct,
        "total_agreement": total_pct,
    }
