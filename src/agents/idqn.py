from __future__ import annotations

"""Independent DQN: one Q-network per agent, trained independently."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from src.common.networks import MLP
from src.common.replay_buffer import ReplayBuffer
from src.common.schedules import LinearEpsilonSchedule
from src.agents.base import BaseAgent


class IDQNAgent(BaseAgent):
    """
    Independent DQN: each agent trains its own Q-network using the shared global reward.

    Each agent has:
    - Q-network: obs_i -> Q-values over actions
    - Target Q-network (soft updated)
    - Own replay buffer
    - Own optimizer
    """

    def __init__(self, config: dict, obs_sizes: list, action_size: int, num_agents: int = 3):
        """
        Args:
            config: training hyperparams dict (see configs/idqn.yaml)
            obs_sizes: observation size per agent
            action_size: number of discrete actions per agent
            num_agents: number of agents
        """
        self.config = config
        self.num_agents = num_agents
        self.action_size = action_size
        self.device = config.get("device", "cpu")

        hidden = config.get("hidden_sizes", [128, 128])
        self.q_nets = [MLP(obs_sizes[i], action_size, hidden).to(self.device) for i in range(num_agents)]
        self.target_nets = [MLP(obs_sizes[i], action_size, hidden).to(self.device) for i in range(num_agents)]
        for i in range(num_agents):
            self.target_nets[i].load_state_dict(self.q_nets[i].state_dict())
            self.target_nets[i].eval()

        lr = config.get("lr", 5e-4)
        self.optimizers = [optim.Adam(self.q_nets[i].parameters(), lr=lr) for i in range(num_agents)]

        buf_size = config.get("buffer_size", 100_000)
        self.buffers = [ReplayBuffer(buf_size, obs_sizes[i]) for i in range(num_agents)]

        self.epsilon_schedule = LinearEpsilonSchedule(
            eps_start=config.get("eps_start", 1.0),
            eps_end=config.get("eps_end", 0.05),
            decay_steps=config.get("eps_decay_steps", 20_000),
        )

        self.gamma = config.get("gamma", 0.99)
        self.batch_size = config.get("batch_size", 64)
        self.learning_starts = config.get("learning_starts", 1000)
        self.tau = config.get("tau", 0.005)
        self.grad_clip = config.get("grad_clip", 10.0)
        self.total_steps = 0

    def act(self, obs_dict: dict, explore: bool = True) -> list:
        """Epsilon-greedy action selection."""
        eps = self.epsilon_schedule.value(self.total_steps) if explore else 0.0
        actions = []
        for i in range(self.num_agents):
            if explore and np.random.random() < eps:
                actions.append(np.random.randint(self.action_size))
            else:
                obs = torch.tensor(obs_dict["agents"][i], dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    q_vals = self.q_nets[i](obs)
                actions.append(q_vals.argmax(dim=1).item())
        return actions

    def store_transition(self, obs_dict, actions, reward, next_obs_dict, done):
        """Store transition in each agent's buffer."""
        for i in range(self.num_agents):
            self.buffers[i].push(
                obs_dict["agents"][i],
                actions[i],
                reward,  # shared global reward
                next_obs_dict["agents"][i],
                done,
            )
        self.total_steps += 1

    def update(self) -> dict | None:
        """Update all agents' Q-networks. Returns dict of metrics."""
        if len(self.buffers[0]) < self.learning_starts:
            return None

        total_loss = 0.0
        for i in range(self.num_agents):
            loss = self._update_agent(i)
            total_loss += loss
            self._soft_update_target(i)

        return {"loss": total_loss / self.num_agents, "epsilon": self.epsilon_schedule.value(self.total_steps)}

    def _update_agent(self, i: int) -> float:
        """One gradient step for agent i. Returns scalar loss."""
        batch = self.buffers[i].sample(self.batch_size, self.device)
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        # Current Q values
        q_vals = self.q_nets[i](obs)
        q_current = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values (double DQN-style: greedy action from online net, value from target)
        with torch.no_grad():
            next_q_online = self.q_nets[i](next_obs)
            best_actions = next_q_online.argmax(dim=1)
            next_q_target = self.target_nets[i](next_obs)
            next_q = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            td_target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = nn.functional.mse_loss(q_current, td_target)
        self.optimizers[i].zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_nets[i].parameters(), self.grad_clip)
        self.optimizers[i].step()
        return loss.item()

    def _soft_update_target(self, i: int) -> None:
        """Soft update target network: theta_target = tau*theta + (1-tau)*theta_target."""
        tau = self.tau
        for param, target_param in zip(self.q_nets[i].parameters(), self.target_nets[i].parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, path: str) -> None:
        """Save all Q-network weights."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {f"q_net_{i}": self.q_nets[i].state_dict() for i in range(self.num_agents)}
        state.update({f"target_net_{i}": self.target_nets[i].state_dict() for i in range(self.num_agents)})
        torch.save(state, path)

    def load(self, path: str) -> None:
        """Load Q-network weights."""
        state = torch.load(path, map_location=self.device)
        for i in range(self.num_agents):
            self.q_nets[i].load_state_dict(state[f"q_net_{i}"])
            self.target_nets[i].load_state_dict(state[f"target_net_{i}"])
