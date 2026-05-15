"""
Deep Q-Network (DQN) agent for Blackjack.

Implements the foundational DQN algorithm with two critical stabilization
mechanisms: Experience Replay (breaks temporal correlation) and a Target
Network (provides stable learning targets).

The network approximates Q(s,a) for all actions simultaneously, replacing
the massive Q-table required for extended state spaces.

Reference: Mnih et al. (2015), "Human-level control through deep RL"
"""

import logging
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.config import DQNConfig

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Experience Replay buffer for DQN.

    Stores transition tuples (state, action, reward, next_state, done) and
    provides uniform random sampling. Breaking temporal correlations via
    random sampling is critical for stable neural network training.
    """

    def __init__(self, capacity: int) -> None:
        self.buffer: deque = deque(maxlen=capacity)
        logger.debug("ReplayBuffer initialized: capacity=%d", capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        """
        Sample a random batch of transitions.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as numpy arrays.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Feedforward neural network for Q-value approximation.

    Architecture: Input → [Linear → ReLU] × N → Linear → Output
    Input dimension: state size (3 for basic, 13 for extended)
    Output dimension: number of actions (2 for Hit/Stand)
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

        # Count parameters for logging
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "QNetwork created: input=%d, hidden=%s, output=%d, params=%d",
            input_dim, hidden_dims, output_dim, total_params,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: state → Q-values for all actions."""
        return self.network(x)


class DQNAgent:
    """
    Deep Q-Network agent with Experience Replay and Target Network.

    Experience Replay breaks temporal correlations by training on random
    mini-batches from a large memory buffer. The Target Network provides
    a stable learning signal by using a frozen copy of the network weights
    for computing TD targets, updated periodically via hard copy.
    """

    def __init__(self, config: DQNConfig | None = None) -> None:
        """
        Initialize DQN agent.

        Args:
            config: DQN hyperparameters. Uses defaults if None.
        """
        self.config = config or DQNConfig()
        self._name = "DQN"

        # Resolve device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        # Networks
        self.online_net = QNetwork(
            self.config.input_dim,
            self.config.output_dim,
            self.config.hidden_dims,
        ).to(self.device)

        self.target_net = QNetwork(
            self.config.input_dim,
            self.config.output_dim,
            self.config.hidden_dims,
        ).to(self.device)

        # Initialize target network with same weights
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # Target net is never trained directly

        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.config.learning_rate)
        self.loss_fn = nn.MSELoss()

        # Experience Replay
        self.replay_buffer = ReplayBuffer(self.config.replay_buffer_size)

        # Training state
        self.total_steps: int = 0
        self.training_losses: list[float] = []
        self.training_rewards: list[float] = []

        logger.info(
            "DQNAgent initialized: device=%s, hidden=%s, buffer_size=%d, "
            "target_update_freq=%d, lr=%.1e",
            self.device, self.config.hidden_dims, self.config.replay_buffer_size,
            self.config.target_update_freq, self.config.learning_rate,
        )

    @property
    def name(self) -> str:
        return self._name

    def _state_to_tensor(self, state: tuple | np.ndarray) -> torch.Tensor:
        """Convert state to normalized tensor for network input."""
        if isinstance(state, tuple):
            player_sum, dealer_card, usable_ace = state
            arr = np.array(
                [player_sum / 31.0, dealer_card / 10.0, float(usable_ace)],
                dtype=np.float32,
            )
        else:
            arr = state.astype(np.float32)
        return torch.FloatTensor(arr).unsqueeze(0).to(self.device)

    def _state_to_array(self, state: tuple | np.ndarray) -> np.ndarray:
        """Convert state to normalized numpy array for replay buffer."""
        if isinstance(state, tuple):
            player_sum, dealer_card, usable_ace = state
            return np.array(
                [player_sum / 31.0, dealer_card / 10.0, float(usable_ace)],
                dtype=np.float32,
            )
        return state.astype(np.float32)

    def get_action(self, state: tuple | np.ndarray, greedy: bool = False) -> int:
        """
        Select action using ε-greedy policy.

        Args:
            state: Current game state.
            greedy: If True, always select best action (no exploration).

        Returns:
            Action: 0 (Stand) or 1 (Hit).
        """
        if not greedy:
            epsilon = self.config.get_epsilon(self.total_steps)
            if random.random() < epsilon:
                return random.randint(0, self.config.output_dim - 1)

        with torch.no_grad():
            state_tensor = self._state_to_tensor(state)
            q_values = self.online_net(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def store_transition(
        self,
        state: tuple | np.ndarray,
        action: int,
        reward: float,
        next_state: tuple | np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        s = self._state_to_array(state)
        ns = self._state_to_array(next_state)
        self.replay_buffer.push(s, action, reward, ns, done)

    def train_step(self) -> float | None:
        """
        Perform one gradient descent step on a mini-batch from replay buffer.

        Returns:
            Loss value, or None if buffer is too small.
        """
        if len(self.replay_buffer) < self.config.min_replay_size:
            return None

        # Sample mini-batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values: Q(s, a) from online network
        current_q = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            target_q = rewards_t + (1 - dones_t) * self.config.discount_factor * next_q

        # Compute loss and backprop
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        loss_val = loss.item()
        self.training_losses.append(loss_val)

        return loss_val

    def update_target_network(self) -> None:
        """Hard copy online network weights to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())
        logger.debug("Target network updated at step %d", self.total_steps)

    def get_policy(self) -> dict[tuple[int, int, bool], int]:
        """Extract greedy policy for all standard Blackjack states."""
        policy: dict[tuple[int, int, bool], int] = {}
        self.online_net.eval()
        with torch.no_grad():
            for player_sum in range(4, 22):
                for dealer_card in range(1, 11):
                    for usable_ace in [False, True]:
                        state = (player_sum, dealer_card, usable_ace)
                        action = self.get_action(state, greedy=True)
                        policy[state] = action
        self.online_net.train()
        return policy

    def save(self, filepath: str | Path) -> None:
        """Save model weights and training state."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_net_state_dict": self.online_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "training_losses": self.training_losses[-10000:],  # Keep last 10K
                "training_rewards": self.training_rewards[-10000:],
                "config": self.config,
            },
            filepath,
        )
        logger.info("DQNAgent saved to %s (steps=%d)", filepath, self.total_steps)

    def load(self, filepath: str | Path) -> None:
        """Load model weights and training state."""
        filepath = Path(filepath)
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(checkpoint["online_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint["total_steps"]
        self.training_losses = checkpoint.get("training_losses", [])
        self.training_rewards = checkpoint.get("training_rewards", [])
        logger.info("DQNAgent loaded from %s (steps=%d)", filepath, self.total_steps)
