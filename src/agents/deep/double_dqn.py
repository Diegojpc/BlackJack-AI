"""
Double DQN agent for Blackjack.

Addresses the overestimation bias of vanilla DQN by decoupling action
selection from action evaluation. The online network selects the best
action, but the target network evaluates its value. This prevents the
max operator from propagating positively-biased noise.

Critical for Blackjack where the stochastic variance of the deck can
inflate Q-values for objectively bad actions (e.g., hitting on 16 vs 7
after a lucky streak of draws).

Reference: Van Hasselt et al. (2016), "Deep RL with Double Q-Learning"
"""

import logging

import torch

from src.agents.deep.dqn import DQNAgent
from src.utils.config import DQNConfig

logger = logging.getLogger(__name__)


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN: uses online network for action selection,
    target network for value evaluation.

    The ONLY difference from vanilla DQN is in the target computation:
    - DQN:        target = r + γ * max_a' Q_target(s', a')
    - Double DQN: target = r + γ * Q_target(s', argmax_a' Q_online(s', a'))

    This single change dramatically reduces overestimation in stochastic
    environments like Blackjack.
    """

    def __init__(self, config: DQNConfig | None = None) -> None:
        super().__init__(config)
        self._name = "Double DQN"
        logger.info("DoubleDQNAgent initialized (inherits DQN with decoupled evaluation)")

    def train_step(self) -> float | None:
        """
        Perform Double DQN gradient step.

        Key difference: action selection via online net, evaluation via target net.
        """
        if len(self.replay_buffer) < self.config.min_replay_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # DOUBLE DQN: Online net SELECTS action, Target net EVALUATES value
            # Step 1: Online network selects the best action for each next state
            best_actions = self.online_net(next_states_t).argmax(dim=1, keepdim=True)

            # Step 2: Target network evaluates Q-value of that selected action
            next_q = self.target_net(next_states_t).gather(1, best_actions).squeeze(1)

            target_q = rewards_t + (1 - dones_t) * self.config.discount_factor * next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        loss_val = loss.item()
        self.training_losses.append(loss_val)
        return loss_val
