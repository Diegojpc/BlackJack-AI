"""
Dueling DQN agent for Blackjack.

Splits the network into two parallel streams:
  - Value stream V(s): estimates the intrinsic value of being in a state
  - Advantage stream A(s,a): estimates the relative advantage of each action

Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))

This architecture is uniquely powerful for Blackjack because many states
have inherently high or low value regardless of the action taken. If a
player has 20, the state value is high whether they stand (correct) or
hit (terrible). Dueling DQN learns this state value rapidly without
exhaustively evaluating every action, accelerating convergence.

Reference: Wang et al. (2016), "Dueling Network Architectures for Deep RL"
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.deep.dqn import DQNAgent, ReplayBuffer
from src.utils.config import DQNConfig

logger = logging.getLogger(__name__)


class DuelingQNetwork(nn.Module):
    """
    Dueling architecture: shared feature extractor splits into
    Value and Advantage streams.

    Architecture:
        Input → [Shared Linear → ReLU] →  ┌─ Value:     Linear → V(s)
                                            └─ Advantage: Linear → A(s,a)
                                            → Q(s,a) = V(s) + (A(s,a) - mean(A))
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
    ) -> None:
        super().__init__()

        # Shared feature extractor
        shared_layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims[:-1]:  # All but last hidden layer
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.shared = nn.Sequential(*shared_layers) if shared_layers else nn.Identity()

        # Value stream: V(s) → scalar
        stream_input = hidden_dims[-2] if len(hidden_dims) > 1 else input_dim
        value_hidden = hidden_dims[-1] // 2  # Half the neurons for each stream
        self.value_stream = nn.Sequential(
            nn.Linear(stream_input, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
        )

        # Advantage stream: A(s,a) → one value per action
        self.advantage_stream = nn.Sequential(
            nn.Linear(stream_input, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, output_dim),
        )

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "DuelingQNetwork created: input=%d, shared=%s, stream_hidden=%d, "
            "output=%d, params=%d",
            input_dim, hidden_dims[:-1], value_hidden, output_dim, total_params,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with value-advantage decomposition.

        Returns Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        The mean subtraction ensures identifiability — without it, V and A
        are not uniquely defined.
        """
        features = self.shared(x)
        value = self.value_stream(features)          # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, num_actions)

        # Combine: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class DuelingDQNAgent(DQNAgent):
    """
    Dueling DQN agent with Value-Advantage decomposition.

    Replaces the standard QNetwork with DuelingQNetwork while inheriting
    all other DQN machinery (replay buffer, target network, training loop).
    """

    def __init__(self, config: DQNConfig | None = None) -> None:
        """
        Initialize Dueling DQN.

        We override __init__ to replace the network architecture but keep
        everything else from DQNAgent.
        """
        # Initialize config and device first (without calling super().__init__)
        self.config = config or DQNConfig()
        self._name = "Dueling DQN"

        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        # Dueling networks instead of standard QNetwork
        self.online_net = DuelingQNetwork(
            self.config.input_dim,
            self.config.output_dim,
            self.config.hidden_dims,
        ).to(self.device)

        self.target_net = DuelingQNetwork(
            self.config.input_dim,
            self.config.output_dim,
            self.config.hidden_dims,
        ).to(self.device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.config.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(self.config.replay_buffer_size)

        self.total_steps: int = 0
        self.training_losses: list[float] = []
        self.training_rewards: list[float] = []

        logger.info(
            "DuelingDQNAgent initialized: device=%s, hidden=%s, "
            "value/advantage streams active",
            self.device, self.config.hidden_dims,
        )
