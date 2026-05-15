# 🃏 Blackjack AI — Reinforcement Learning from Scratch to Card Counting

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A progressive, research-driven reinforcement learning project that trains AI agents to master Blackjack — from basic strategy through **implicit card counting** — culminating in a PPO agent that achieves **positive expected value against the house**.

## 🎯 Project Overview

| Phase | Algorithms | Goal | Status |
|-------|-----------|------|--------|
| **Phase 1** | Monte Carlo, SARSA, Q-Learning | Match Basic Strategy (~42.5% win rate) | 🔨 In Progress |
| **Phase 2** | DQN, Double DQN, Dueling DQN | Scale to neural network function approximation | ⏳ Planned |
| **Phase 3** | PPO (Stable-Baselines3) | **Beat the house** with implicit card counting | ⏳ Planned |
| **Phase 4** | PPO + Curriculum Learning | Full casino rules (Split/Double/Surrender) | ⏳ Planned |

## 📐 Architecture

```
BlackJack-AI/
├── src/
│   ├── agents/           # RL agent implementations
│   │   ├── tabular/      # Monte Carlo, SARSA, Q-Learning
│   │   └── deep/         # DQN variants, PPO
│   ├── environments/     # Gymnasium wrappers & custom envs
│   ├── training/         # Training pipelines
│   ├── evaluation/       # Evaluation harness & Basic Strategy reference
│   ├── visualization/    # Strategy heatmaps, convergence plots
│   └── utils/            # Config, logging
├── notebooks/            # Exploratory Jupyter notebooks
├── models/               # Saved model checkpoints
├── results/              # Training metrics, plots
└── tests/                # Unit tests
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- NVIDIA GPU (optional, GTX 1050 Ti or better)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/BlackJack-AI.git
cd BlackJack-AI

# Install dependencies
uv sync
```

### Training

```bash
# Train all tabular agents (Phase 1)
make train-tabular

# Train individual agents
make train-mc        # Monte Carlo (500K episodes)
make train-sarsa     # SARSA (500K episodes)
make train-ql        # Q-Learning (500K episodes)

# Quick smoke test (1K episodes, fast)
make smoke
```

### Evaluation

```bash
# Evaluate the Basic Strategy reference agent
make eval-basic

# Run unit tests
make test
```

## 📊 Phase 1 Results

*Results will be updated after training completes.*

| Agent | Win Rate | Bust Rate | Avg Reward | Strategy Agreement |
|-------|----------|-----------|------------|-------------------|
| Basic Strategy (Reference) | ~42.5% | ~15% | -0.058 | 100% |
| Monte Carlo | — | — | — | — |
| SARSA | — | — | — | — |
| Q-Learning | — | — | — | — |

## 🧠 How It Works

### Tabular RL (Phase 1)
The standard Gymnasium `Blackjack-v1` environment provides a 3-tuple state: `(player_sum, dealer_card, usable_ace)`. With ~360 unique states, tabular methods can store and update Q-values for every state-action pair directly.

- **Monte Carlo**: Learns from complete episodes; highest variance but unbiased
- **SARSA**: On-policy TD learning; conservative due to ε-greedy awareness
- **Q-Learning**: Off-policy; fastest convergence via max-Q updates

### Why Card Counting Requires Deep RL (Phase 3)
The standard environment uses an infinite deck — card counting is mathematically impossible. Phase 3 introduces a **custom finite-deck environment** with an extended state space tracking dealt cards, inflating the state space to **~1 billion permutations**. Only deep neural networks (PPO) can handle this.

## 📚 Research Foundation

This project is built on a comprehensive research survey covering 48+ academic papers and implementations. See [`docs/Blackjack AI Research Plan.pdf`](docs/Blackjack%20AI%20Research%20Plan.pdf) for the full analysis.

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with 🎰 and reinforcement learning.*
