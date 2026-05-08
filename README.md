# Multi-Agent Reinforcement Learning for Cooperative Inventory Management

## Overview

This project benchmarks multi-agent reinforcement learning (MARL) methods against classical inventory policies on a 3-node serial supply chain. Three agents (Supplier, Warehouse, Retailer) cooperate to minimize shared holding and stockout costs under Poisson demand. We evaluate IDQN, CDQN, and VDN against Base-Stock and (s,S) baselines under both stationary and non-stationary demand regimes, measuring episode cost, bullwhip ratio, and stockout frequency across 5 random seeds.

## Setup

```bash
pip install -e .
```

## Quickstart

```bash
# ~20 min sanity check (quick mode: fewer timesteps + fewer seeds)
python scripts/run_all_experiments.py --quick

# Full sweep (~8h, 5 seeds x 3 RL methods x 2 settings + baselines)
python scripts/run_all_experiments.py

# Generate figures and tables after experiments complete
python scripts/make_figures.py && python scripts/make_tables.py
```

## Repository Layout

```
configs/                   # YAML configs for each agent and environment
  idqn.yaml                # IDQN hyperparameters + env config
  cdqn.yaml                # CDQN hyperparameters + env config
  vdn.yaml                 # VDN hyperparameters + env config
  env_default.yaml         # Stationary environment config
  env_nonstationary.yaml   # Non-stationary environment config
  baselines.yaml           # Classical baseline config

scripts/
  run_all_experiments.py   # Full experiment runner (train + eval all methods)
  make_figures.py          # Generate 6 report figures
  make_tables.py           # Generate LaTeX + CSV tables

src/
  agents/
    base.py                # Abstract BaseAgent interface
    base_stock.py          # Base-stock policy
    ss_policy.py           # (s,S) reorder-point policy
    idqn.py                # Independent DQN
    cdqn.py                # Centralized DQN
    vdn.py                 # Value Decomposition Networks
  common/
    networks.py            # MLP neural network module
    replay_buffer.py       # Experience replay buffer
    schedules.py           # Epsilon-greedy schedule
    seeding.py             # Global random seed utilities
  envs/
    supply_chain.py        # 3-node serial supply chain (Gymnasium env)
  evaluation/
    metrics.py             # total_cost, bullwhip_ratio, stockout_frequency
    evaluator.py           # Greedy rollout evaluator
  training/
    train_dqn.py           # Training loop for RL agents
    tune_baselines.py      # Grid-search tuning for classical policies
  utils/
    config.py              # YAML config loading and merging
    logger.py              # CSV training logger

results/
  checkpoints/             # Saved model checkpoints (.pt files)
  logs/                    # Training curves and eval CSVs
  figures/                 # Generated PNG + PDF figures
  tables/                  # Generated CSV + LaTeX tables

tests/                     # Pytest test suite (21 tests)
```

## Methods

- **Base-Stock**: Classical order-up-to policy. Each node orders `max(0, S_i - inventory_position_i)`. Level `S_i` is grid-searched per setting.
- **(s,S) Policy**: Reorder-point policy. Order up to `S_i` when `inventory_position_i < s_i`. Parameters `(s_i, S_i)` are grid-searched per setting.
- **IDQN**: Independent DQN. Each agent trains its own Q-network using the shared global reward. Decentralized execution.
- **CDQN**: Centralized DQN. A single Q-network over the joint observation and joint action space. Used as an upper-bound cooperative benchmark; not scalable beyond small action spaces.
- **VDN**: Value Decomposition Networks. Per-agent Q-networks trained jointly via `Q_total = sum_i Q_i`. Cooperative training with decentralized execution.

## Adding a New Method

1. Subclass `BaseAgent` in `src/agents/base.py` and implement `act()`, `store_transition()`, `update()`, `save()`, `load()`.
2. Register the new class in `AGENT_CLASSES` in `src/training/train_dqn.py`.
3. Add a YAML config under `configs/<method>.yaml` with `agent_type: <method>` and all hyperparameters.
4. Add the method name to `RL_METHODS` in `scripts/run_all_experiments.py` and the color/label dicts in `scripts/make_figures.py`.

## Hyperparameters

| Parameter         | IDQN / VDN | CDQN    |
|-------------------|------------|---------|
| gamma             | 0.99       | 0.99    |
| lr                | 5e-4       | 5e-4    |
| batch_size        | 64         | 64      |
| buffer_size       | 100,000    | 100,000 |
| total_timesteps   | 200,000    | 200,000 |
| eps_start         | 1.0        | 1.0     |
| eps_end           | 0.05       | 0.05    |
| eps_decay_steps   | 20,000     | 20,000  |
| hidden_sizes      | [128, 128] | [256, 256] |
| tau (target sync) | 0.005      | 0.005   |

## Reproducibility

All runs use explicit integer seeds passed to NumPy, PyTorch, and the environment's RNG. Training seeds are `[0, 1, 2, 3, 4]`. Eval seeds are offset from training seeds to avoid overlap. Given the same seed, results in `results/logs/` are deterministic on the same hardware and PyTorch version.
