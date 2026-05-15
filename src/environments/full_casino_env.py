"""
Full casino Blackjack environment with 5 actions: Hit, Stand, Double, Split, Surrender.

Extends the finite-deck environment with realistic casino rules including
action masking to prevent illegal moves (e.g., splitting non-pairs,
doubling after 3+ cards). Designed for use with SB3's MaskablePPO
for curriculum learning.

Action Space:
    0: Stand  — End the hand, keep current cards
    1: Hit    — Draw one card
    2: Double — Double bet, draw exactly one card, then stand
    3: Split  — Split a pair into two hands (requires matching values)
    4: Surrender — Forfeit half the bet, end the hand
"""

import logging
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.environments.finite_deck_env import FiniteDeckBlackjackEnv
from src.utils.config import FiniteDeckConfig

logger = logging.getLogger(__name__)

# Action constants
STAND = 0
HIT = 1
DOUBLE = 2
SPLIT = 3
SURRENDER = 4

ACTION_NAMES = {0: "Stand", 1: "Hit", 2: "Double", 3: "Split", 4: "Surrender"}


class FullCasinoBlackjackEnv(FiniteDeckBlackjackEnv):
    """
    Full casino Blackjack with 5 actions and action masking.

    Extends FiniteDeckBlackjackEnv with:
    - Double Down: double bet, one card, auto-stand
    - Split: split matching-value pairs into two hands
    - Surrender: forfeit half the bet
    - Action masking: prevents illegal actions

    Observation: same 13-dim vector + 2 extra dims for hand context
        [13]: num_cards_in_hand / 10.0  (for double/surrender eligibility)
        [14]: is_pair (0.0 or 1.0, for split eligibility)

    Total observation: 15 dimensions
    """

    def __init__(
        self,
        config: FiniteDeckConfig | None = None,
        render_mode: str | None = None,
        allowed_actions: list[int] | None = None,
    ) -> None:
        """
        Initialize full casino environment.

        Args:
            config: Environment configuration.
            render_mode: Gymnasium render mode.
            allowed_actions: If provided, mask all other actions. Used by
                           curriculum learning to phase in actions gradually.
        """
        super().__init__(config=config, render_mode=render_mode)

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(15,),
            dtype=np.float32,
        )

        # Curriculum: if specified, mask actions not in this list
        self.allowed_actions = set(allowed_actions) if allowed_actions else {0, 1, 2, 3, 4}

        # Split tracking
        self.split_hands: list[list[int]] = []
        self.current_hand_index: int = 0
        self.is_doubled: bool = False
        self.initial_deal: bool = True

        logger.info(
            "FullCasinoBlackjackEnv initialized: actions=%s, obs_dim=15",
            [ACTION_NAMES[a] for a in sorted(self.allowed_actions)],
        )

    def _get_observation(self) -> np.ndarray:
        """Construct the 15-dimensional observation."""
        # Get base 13-dim observation
        base_obs = super()._get_observation()

        # Extra context for action eligibility
        num_cards = len(self.player_hand) / 10.0
        is_pair = float(
            len(self.player_hand) == 2
            and self.player_hand[0] == self.player_hand[1]
        )

        return np.concatenate([base_obs, [num_cards, is_pair]])

    def action_masks(self) -> np.ndarray:
        """
        Return a boolean mask of valid actions for the current state.

        Used by MaskablePPO to prevent the agent from selecting illegal
        or curriculum-restricted actions.

        Returns:
            Boolean array of shape (5,) — True = action is available.
        """
        masks = np.zeros(5, dtype=bool)

        # Stand and Hit are always available
        masks[STAND] = True
        masks[HIT] = True

        # Double Down: only on initial 2-card hand
        masks[DOUBLE] = (len(self.player_hand) == 2 and self.initial_deal)

        # Split: only with matching pair and 2-card hand
        masks[SPLIT] = (
            len(self.player_hand) == 2
            and self.player_hand[0] == self.player_hand[1]
            and self.initial_deal
            and len(self.split_hands) == 0  # No re-splitting
        )

        # Surrender: only on initial 2-card hand
        masks[SURRENDER] = (len(self.player_hand) == 2 and self.initial_deal)

        # Apply curriculum mask
        for action in range(5):
            if action not in self.allowed_actions:
                masks[action] = False

        return masks

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment for a new hand."""
        obs, info = super().reset(seed=seed, options=options)

        self.split_hands = []
        self.current_hand_index = 0
        self.is_doubled = False
        self.initial_deal = True

        # Extend observation to 15 dims
        obs = self._get_observation()
        info["action_mask"] = self.action_masks()

        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute an action in the full casino environment.

        Args:
            action: 0=Stand, 1=Hit, 2=Double, 3=Split, 4=Surrender.
        """
        masks = self.action_masks()
        if not masks[action]:
            logger.warning(
                "Invalid action %s (%s) attempted. Defaulting to Stand.",
                action, ACTION_NAMES.get(action, "?"),
            )
            action = STAND

        info: dict[str, Any] = {"action": ACTION_NAMES[action]}

        if action == SURRENDER:
            # Forfeit half the bet
            self.initial_deal = False
            reward = self.config.surrender_penalty
            info["result"] = "surrender"
            obs = self._get_observation()
            return obs, reward, True, False, info

        elif action == DOUBLE:
            # Double bet, one card, auto-stand
            self.initial_deal = False
            self.is_doubled = True
            self.player_hand.append(self._draw_card())
            player_value, _ = self._hand_value(self.player_hand)

            if player_value > 21:
                reward = -2.0  # Double the loss
                info["result"] = "bust_doubled"
                obs = self._get_observation()
                return obs, reward, True, False, info

            # Auto-stand: resolve dealer
            obs, reward, terminated, truncated, resolve_info = self._resolve_dealer()
            reward *= 2.0  # Double the reward/loss
            resolve_info["doubled"] = True
            return self._get_observation(), reward, True, truncated, resolve_info

        elif action == SPLIT:
            # Split pair into two hands
            self.initial_deal = False
            card = self.player_hand.pop()
            self.split_hands = [
                [self.player_hand[0], self._draw_card()],
                [card, self._draw_card()],
            ]
            self.player_hand = self.split_hands[0]
            self.current_hand_index = 0

            obs = self._get_observation()
            info["result"] = "split"
            info["action_mask"] = self.action_masks()
            return obs, 0.0, False, False, info

        elif action == HIT:
            self.initial_deal = False
            return super().step(1)  # Delegate to parent Hit logic

        else:  # STAND
            self.initial_deal = False

            # If we have split hands, move to next hand
            if self.split_hands and self.current_hand_index < len(self.split_hands) - 1:
                self.current_hand_index += 1
                self.player_hand = self.split_hands[self.current_hand_index]
                self.initial_deal = True  # Can double/surrender on split hand
                obs = self._get_observation()
                info["action_mask"] = self.action_masks()
                return obs, 0.0, False, False, info

            # Resolve all hands
            if self.split_hands:
                total_reward = 0.0
                for hand in self.split_hands:
                    self.player_hand = hand
                    _, reward, _, _, _ = self._resolve_dealer()
                    total_reward += reward
                    # Reset dealer for next hand evaluation
                    if hand != self.split_hands[-1]:
                        # Re-deal dealer (simplified — in real casino, same dealer hand)
                        pass
                info["result"] = "split_resolved"
                info["total_reward"] = total_reward
                return self._get_observation(), total_reward, True, False, info

            return super().step(0)  # Delegate to parent Stand logic


# Register full casino environment
gym.register(
    id="BlackjackFullCasino-v0",
    entry_point="src.environments.full_casino_env:FullCasinoBlackjackEnv",
)
