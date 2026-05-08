from __future__ import annotations

"""Training script for IDQN, CDQN, and VDN agents."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

from src.envs.supply_chain import SupplyChainEnv
from src.agents.idqn import IDQNAgent
from src.agents.cdqn import CDQNAgent
from src.agents.vdn import VDNAgent
from src.common.seeding import set_global_seeds
from src.utils.config import load_config
from src.utils.logger import CSVLogger


AGENT_CLASSES = {
    "idqn": IDQNAgent,
    "cdqn": CDQNAgent,
    "vdn": VDNAgent,
}


def build_agent(agent_type: str, config: dict, env: SupplyChainEnv):
    """Instantiate the correct agent class."""
    if agent_type == "cdqn":
        return CDQNAgent(config, env.global_obs_size, env.action_size, env.num_agents)
    else:
        return AGENT_CLASSES[agent_type](config, env.obs_sizes, env.action_size, env.num_agents)


def evaluate_agent(agent, env_config: dict, n_episodes: int = 20, seed_offset: int = 9999) -> float:
    """Run greedy rollouts, return mean episode cost (positive)."""
    eval_env = SupplyChainEnv(env_config)
    total_costs = []
    for ep in range(n_episodes):
        obs, _ = eval_env.reset(seed=seed_offset + ep)
        episode_cost = 0.0
        done = False
        while not done:
            actions = agent.act(obs, explore=False)
            obs, reward, terminated, truncated, _ = eval_env.step(actions)
            episode_cost += -reward  # reward is negative cost
            done = terminated or truncated
        total_costs.append(episode_cost)
    return float(np.mean(total_costs))


def train(config_path: str, seed: int, quick: bool = False) -> None:
    """Main training entry point."""
    config = load_config(config_path)

    if quick:
        config["total_timesteps"] = config.get("quick_timesteps", 5000)
        config["learning_starts"] = min(500, config.get("learning_starts", 1000))
        config["eval_freq"] = 1000

    set_global_seeds(seed)

    env_config = config.get("env", {})
    env_config["seed"] = seed
    agent_type = config["agent_type"]
    demand_mode = env_config.get("demand_mode", "stationary")

    env = SupplyChainEnv(env_config)
    agent = build_agent(agent_type, config, env)

    log_path = Path(f"results/logs/{agent_type}_{demand_mode}_seed{seed}.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(log_path, ["step", "episode_return", "mean_eval_cost", "loss", "epsilon"])

    checkpoint_dir = Path("results/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    total_timesteps = config.get("total_timesteps", 200_000)
    eval_freq = config.get("eval_freq", 10_000)
    checkpoint_freq = config.get("checkpoint_freq", 50_000)

    obs, _ = env.reset(seed=seed)
    episode_return = 0.0
    episode_count = 0
    last_eval_cost = float("nan")
    last_loss = float("nan")

    print(f"[{agent_type.upper()} seed={seed}] Training for {total_timesteps} steps...")

    for step in range(total_timesteps):
        actions = agent.act(obs, explore=True)
        next_obs, reward, terminated, truncated, _ = env.step(actions)
        done = terminated or truncated

        agent.store_transition(obs, actions, reward, next_obs, done)
        episode_return += reward
        obs = next_obs

        metrics = agent.update()
        if metrics is not None:
            last_loss = metrics.get("loss", float("nan"))

        if done:
            episode_count += 1
            obs, _ = env.reset(seed=seed + episode_count)
            episode_return = 0.0

        if (step + 1) % eval_freq == 0:
            last_eval_cost = evaluate_agent(agent, env_config, n_episodes=20, seed_offset=seed * 10000)
            eps = agent.epsilon_schedule.value(agent.total_steps)
            logger.log({
                "step": step + 1,
                "episode_return": episode_return,
                "mean_eval_cost": last_eval_cost,
                "loss": last_loss,
                "epsilon": eps,
            })
            print(f"  step={step+1:>7} | eval_cost={last_eval_cost:.1f} | loss={last_loss:.4f} | eps={eps:.3f}")

        if (step + 1) % checkpoint_freq == 0:
            ckpt_path = checkpoint_dir / f"{agent_type}_{demand_mode}_seed{seed}_step{step+1}.pt"
            agent.save(str(ckpt_path))

    # Save final model
    final_path = checkpoint_dir / f"{agent_type}_{demand_mode}_seed{seed}_final.pt"
    agent.save(str(final_path))
    logger.close()
    print(f"[{agent_type.upper()} seed={seed}] Done. Final eval cost: {last_eval_cost:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to agent config YAML")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="Short run for smoke testing")
    args = parser.parse_args()
    train(args.config, args.seed, args.quick)
