.PHONY: install test train-tabular train-all eval-basic lint clean

# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────
install:
	uv sync
	uv pip install -e ".[dev]"

# ──────────────────────────────────────────────────────────────────────────────
# Testing
# ──────────────────────────────────────────────────────────────────────────────
test:
	uv run pytest tests/ -v --tb=short

test-cov:
	uv run pytest tests/ -v --cov=src --cov-report=term-missing

# ──────────────────────────────────────────────────────────────────────────────
# Training — Phase 1: Tabular RL
# ──────────────────────────────────────────────────────────────────────────────
train-mc:
	uv run python -m src.training.train_tabular --agent monte_carlo --episodes 500000

train-sarsa:
	uv run python -m src.training.train_tabular --agent sarsa --episodes 500000

train-ql:
	uv run python -m src.training.train_tabular --agent q_learning --episodes 500000

train-tabular: train-mc train-sarsa train-ql

# ──────────────────────────────────────────────────────────────────────────────
# Training — Phase 2: Deep RL
# ──────────────────────────────────────────────────────────────────────────────
train-dqn:
	uv run python -m src.training.train_deep --agent dqn --timesteps 1000000

train-ddqn:
	uv run python -m src.training.train_deep --agent double_dqn --timesteps 1000000

train-dueling:
	uv run python -m src.training.train_deep --agent dueling_dqn --timesteps 1000000

train-deep: train-dqn train-ddqn train-dueling

# ──────────────────────────────────────────────────────────────────────────────
# Training — Phase 3: PPO Card Counting
# ──────────────────────────────────────────────────────────────────────────────
train-ppo:
	uv run python -m src.training.train_ppo --timesteps 10000000

# ──────────────────────────────────────────────────────────────────────────────
# Training — All Phases
# ──────────────────────────────────────────────────────────────────────────────
train-all: train-tabular train-deep train-ppo

# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────
eval-basic:
	uv run python -c "from src.evaluation.evaluator import evaluate_basic_strategy; r = evaluate_basic_strategy(100_000); print(r.summary())"

# ──────────────────────────────────────────────────────────────────────────────
# Quick Smoke Test (fast training for CI/CD)
# ──────────────────────────────────────────────────────────────────────────────
smoke:
	uv run python -m src.training.train_tabular --agent monte_carlo --episodes 1000 --no-eval
	uv run python -m src.training.train_tabular --agent sarsa --episodes 1000 --no-eval
	uv run python -m src.training.train_tabular --agent q_learning --episodes 1000 --no-eval

# ──────────────────────────────────────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────────────────────────────────────
clean:
	rm -rf models/*.pkl results/*.csv results/logs/ __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
