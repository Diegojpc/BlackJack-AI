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
# Training — Phase 2: Deep RL (improved: Huber loss, soft updates, wider nets)
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
	uv run python -m src.training.train_ppo --timesteps 20000000 --decks 1

# Extended run — 30M steps to push card-counting strategy deeper
train-ppo-long:
	uv run python -m src.training.train_ppo --timesteps 30000000 --decks 1

# ──────────────────────────────────────────────────────────────────────────────
# Training — Phase 4: Curriculum (Double Down + Split + Surrender)
# Best expected value — recommended for maximum performance
# ──────────────────────────────────────────────────────────────────────────────
train-curriculum:
	uv run python -m src.training.curriculum --timesteps 20000000 --decks 1

train-curriculum-long:
	uv run python -m src.training.curriculum --timesteps 50000000 --decks 1

# ──────────────────────────────────────────────────────────────────────────────
# Training — All Phases
# ──────────────────────────────────────────────────────────────────────────────
train-all: train-tabular train-deep train-ppo

# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────
eval-basic:
	uv run python -c "from src.evaluation.evaluator import evaluate_basic_strategy; r = evaluate_basic_strategy(100_000); print(r.summary())"

compare:
	uv run python -m src.evaluation.compare_all

play:
	uv run python -m src.evaluation.play --agent dueling_dqn --hands 20

# ──────────────────────────────────────────────────────────────────────────────
# Interactive Advisor App (for real games with friends)
# ──────────────────────────────────────────────────────────────────────────────
app:
	uv run streamlit run app.py

# Export a self-contained model for sharing — run once after training
# Output: inference/model.pt (no project dependencies, safe to share)
export-model:
	uv run python -c "\
import torch, pathlib; \
p = pathlib.Path('models'); \
candidates = [p/'dueling_dqn_final.pt', *sorted(p.glob('dueling_dqn_step*.pt'), reverse=True)]; \
src = next(f for f in candidates if f.exists()); \
ckpt = torch.load(src, map_location='cpu', weights_only=False); \
cfg = ckpt.get('config'); \
torch.save({'weights': ckpt['online_net_state_dict'], 'hidden_dims': list(cfg.hidden_dims), 'input_dim': cfg.input_dim, 'output_dim': cfg.output_dim}, 'inference/model.pt'); \
print(f'Exported {src.name} -> inference/model.pt') \
"

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
