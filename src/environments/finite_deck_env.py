"""
Custom finite-deck Blackjack environment for card counting via Deep RL.

Unlike Gymnasium's default infinite-deck implementation, this environment
draws cards WITHOUT replacement from a finite shoe (1-8 decks). The
extended state space tracks the exact count of each card value dealt,
enabling the agent to discover card counting principles autonomously.

This inflates the state space from ~360 discrete states to approximately
1 BILLION permutations for single-deck, making tabular methods completely
intractable and requiring deep function approximators (PPO).

Reference: Stanford CS230 (2021), "Learning Explainable Policy for Blackjack"
"""

import logging
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.utils.config import FiniteDeckConfig

logger = logging.getLogger(__name__)


class FiniteDeckBlackjackEnv(gym.Env):
    """
    Gymnasium-compatible Blackjack environment with finite deck shoe.

    Key differences from Blackjack-v1:
    1. Cards drawn WITHOUT replacement from a physical shoe
    2. Extended observation includes counts of all card values dealt
    3. Configurable number of decks (1-8) and shoe penetration
    4. Reshuffle triggered when penetration threshold is reached
    5. Natural Blackjack pays 1.5x

    Observation Space (13-dimensional Box):
        [0]: player_sum / 31.0           (normalized hand value)
        [1]: dealer_card / 10.0          (normalized dealer up-card)
        [2]: usable_ace                  (0.0 or 1.0)
        [3-12]: count_of_value_dealt / max_count  (normalized card counts for values 1-10)

    Action Space (Discrete(2)):
        0: Stand
        1: Hit
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: FiniteDeckConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        """
        Initialize finite-deck environment.

        Args:
            config: Environment configuration.
            render_mode: Gymnasium render mode.
        """
        super().__init__()

        self.config = config or FiniteDeckConfig()
        self.render_mode = render_mode

        # Deck composition: values 1 (Ace) through 10
        # Each value appears 4 times per deck, except 10-value (16 times: 10, J, Q, K)
        self.num_decks = self.config.num_decks
        self.cards_per_value = 4 * self.num_decks  # Ace through 9
        self.ten_value_cards = 16 * self.num_decks  # 10, J, Q, K
        self.total_cards_in_shoe = 52 * self.num_decks

        # Shoe state: counts remaining for each value (1=Ace, 2-9, 10)
        self.shoe: np.ndarray = np.zeros(10, dtype=np.int32)
        self.cards_dealt_counts: np.ndarray = np.zeros(10, dtype=np.int32)
        self.total_cards_dealt: int = 0

        # Max possible counts for normalization
        self.max_counts = np.array(
            [self.cards_per_value] * 9 + [self.ten_value_cards],
            dtype=np.float32,
        )

        # Player and dealer hands
        self.player_hand: list[int] = []
        self.dealer_hand: list[int] = []

        # Spaces
        self.action_space = spaces.Discrete(2)  # 0=Stand, 1=Hit
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(13,),
            dtype=np.float32,
        )

        self._shuffle_shoe()

        logger.info(
            "FiniteDeckBlackjackEnv initialized: decks=%d, penetration=%.0f%%, "
            "total_cards=%d, observation_dim=13",
            self.num_decks, self.config.penetration * 100,
            self.total_cards_in_shoe,
        )

    def _shuffle_shoe(self) -> None:
        """Reset the shoe to a full, undealt state."""
        self.shoe = np.array(
            [self.cards_per_value] * 9 + [self.ten_value_cards],
            dtype=np.int32,
        )
        self.cards_dealt_counts = np.zeros(10, dtype=np.int32)
        self.total_cards_dealt = 0
        logger.debug("Shoe reshuffled: %d cards", self.total_cards_in_shoe)

    def _draw_card(self) -> int:
        """
        Draw a single card from the shoe without replacement.

        Returns:
            Card value (1-10), where 1=Ace.

        Raises:
            RuntimeError: If shoe is exhausted (should never happen with penetration).
        """
        remaining = self.shoe.sum()
        if remaining == 0:
            logger.warning("Shoe exhausted during hand. Reshuffling mid-hand.")
            self._shuffle_shoe()

        # Weighted random draw based on remaining card counts
        probabilities = self.shoe.astype(np.float64) / self.shoe.sum()
        card_value = int(np.random.choice(np.arange(1, 11), p=probabilities))

        # Remove card from shoe
        self.shoe[card_value - 1] -= 1
        self.cards_dealt_counts[card_value - 1] += 1
        self.total_cards_dealt += 1

        return card_value

    @staticmethod
    def _hand_value(hand: list[int]) -> tuple[int, bool]:
        """
        Calculate optimal hand value and usable ace status.

        Args:
            hand: List of card values (1=Ace, 2-10).

        Returns:
            Tuple of (total_value, has_usable_ace).
        """
        total = sum(hand)
        usable_ace = False

        # Check if an Ace can be used as 11 without busting
        if 1 in hand and total + 10 <= 21:
            total += 10
            usable_ace = True

        return total, usable_ace

    def _get_observation(self) -> np.ndarray:
        """
        Construct the 13-dimensional extended observation.

        Returns:
            Normalized observation vector.
        """
        player_value, usable_ace = self._hand_value(self.player_hand)
        dealer_showing = self.dealer_hand[0]  # Only the up-card

        # Base state (3 dims)
        base = np.array([
            player_value / 31.0,
            dealer_showing / 10.0,
            float(usable_ace),
        ], dtype=np.float32)

        # Extended state: normalized counts of each card value dealt (10 dims)
        # This is the information that enables implicit card counting
        safe_max = np.where(self.max_counts > 0, self.max_counts, 1.0)
        card_counts = self.cards_dealt_counts.astype(np.float32) / safe_max

        return np.concatenate([base, card_counts])

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset environment for a new hand.

        Checks penetration threshold and reshuffles if needed.

        Returns:
            Tuple of (observation, info_dict).
        """
        super().reset(seed=seed)

        # Check penetration: reshuffle if we've dealt past the threshold
        penetration_reached = (
            self.total_cards_dealt / self.total_cards_in_shoe
        ) >= self.config.penetration

        if penetration_reached:
            self._shuffle_shoe()
            logger.debug(
                "Penetration threshold reached (%.0f%%). Reshuffling.",
                self.config.penetration * 100,
            )

        # Deal initial hands
        self.player_hand = [self._draw_card(), self._draw_card()]
        self.dealer_hand = [self._draw_card(), self._draw_card()]

        # Check for natural Blackjack
        player_value, _ = self._hand_value(self.player_hand)
        dealer_value, _ = self._hand_value(self.dealer_hand)

        info = {
            "player_hand": list(self.player_hand),
            "dealer_hand": list(self.dealer_hand),
            "cards_dealt": self.total_cards_dealt,
            "shoe_remaining": int(self.shoe.sum()),
        }

        # Handle natural Blackjack immediately
        if player_value == 21:
            # Natural BJ for player
            if dealer_value == 21:
                info["natural_push"] = True
            else:
                info["natural_blackjack"] = True

        return self._get_observation(), info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one action in the environment.

        Args:
            action: 0 (Stand) or 1 (Hit).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        player_value, _ = self._hand_value(self.player_hand)
        info: dict[str, Any] = {}

        # Check if this is a natural Blackjack hand (should be auto-resolved)
        if len(self.player_hand) == 2 and player_value == 21:
            dealer_value, _ = self._hand_value(self.dealer_hand)
            if dealer_value == 21:
                return self._get_observation(), 0.0, True, False, {"result": "push"}
            else:
                return (
                    self._get_observation(),
                    self.config.natural_bonus,
                    True,
                    False,
                    {"result": "natural_blackjack"},
                )

        if action == 1:  # Hit
            self.player_hand.append(self._draw_card())
            player_value, _ = self._hand_value(self.player_hand)

            if player_value > 21:
                # Player busted
                return (
                    self._get_observation(),
                    -1.0,
                    True,
                    False,
                    {"result": "bust", "player_value": player_value},
                )

            if player_value == 21:
                # Got 21, auto-stand
                return self._resolve_dealer()

            # Hand continues
            return self._get_observation(), 0.0, False, False, info

        else:  # Stand
            return self._resolve_dealer()

    def _resolve_dealer(self) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Play out the dealer's hand according to standard rules (hit on soft 17).

        Returns:
            Final (observation, reward, terminated, truncated, info) tuple.
        """
        # Dealer draws until reaching 17+
        while True:
            dealer_value, dealer_usable_ace = self._hand_value(self.dealer_hand)
            if dealer_value >= 17:
                break
            self.dealer_hand.append(self._draw_card())

        dealer_value, _ = self._hand_value(self.dealer_hand)
        player_value, _ = self._hand_value(self.player_hand)

        info = {
            "player_value": player_value,
            "dealer_value": dealer_value,
            "player_hand": list(self.player_hand),
            "dealer_hand": list(self.dealer_hand),
        }

        # Determine outcome
        if dealer_value > 21:
            reward = 1.0
            info["result"] = "dealer_bust"
        elif player_value > dealer_value:
            reward = 1.0
            info["result"] = "win"
        elif player_value < dealer_value:
            reward = -1.0
            info["result"] = "loss"
        else:
            reward = 0.0
            info["result"] = "push"

        return self._get_observation(), reward, True, False, info

    def render(self) -> None:
        """Render current game state to console."""
        if self.render_mode == "human":
            player_value, usable_ace = self._hand_value(self.player_hand)
            dealer_value, _ = self._hand_value(self.dealer_hand)
            print(f"Player: {self.player_hand} = {player_value} (usable_ace={usable_ace})")
            print(f"Dealer: [{self.dealer_hand[0]}, ?] showing {self.dealer_hand[0]}")
            print(f"Shoe remaining: {self.shoe.sum()}/{self.total_cards_in_shoe}")


# Register environment with Gymnasium
gym.register(
    id="BlackjackFiniteDeck-v0",
    entry_point="src.environments.finite_deck_env:FiniteDeckBlackjackEnv",
)
