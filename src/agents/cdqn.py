from __future__ import annotations

"""Centralized DQN: single Q-network over joint state and joint action space."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from src.common.networks import MLP
from src.common.replay_buffer import ReplayBuffer
from src.common.schedules import LinearEpsilonSchedule
from src.agents.base import BaseAgent


class CDQNAgent(BaseAgent):
    """
    Centralized DQN: one Q-network.
    Input: global observation (concatenation of all per-agent obs).
    Output: Q-values over joint action space (max_order+1)^num_agents.

    Joint action index decoded as base-(max_order+1) number:
        joint_idx = a_0 + a_1*(max_order+1) + a_2*(max_order+1)^2
    """

    def __init__(self, config: dict, global_obs_size: int, action_size: int, num_agents: int = 3):
        """
        Args:
            config: training config dict
            global_obs_size: size of global observation vector
            action_size: per-agent action space size (max_order + 1)
            num_agents: number of agents
        """
        self.config = config
        self.num_agents = num_agents
        self.action_size = action_size
        self.joint_action_size = action_size ** num_agents
        self.device = config.get("device", "cpu")

        hidden = config.get("hidden_sizes", [256, 256])
        self.q_net = MLP(global_obs_size, self.joint_action_size, hidden).to(self.device)
        self.target_net = MLP(global_obs_size, self.joint_action_size, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        lr = config.get("lr", 5e-4)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        buf_size = config.get("buffer_size", 100_000)
        self.buffer = ReplayBuffer(buf_size, global_obs_size)

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

    def _encode_joint_action(self, per_agent_actions: list) -> int:
        """Encode per-agent actions to joint action index."""
        idx = 0
        for i, a in enumerate(per_agent_actions):
            idx += a * (self.action_size ** i)
        return idx

    def _decode_joint_action(self, joint_idx: int) -> list:
        """Decode joint action index to per-agent actions."""
        actions = []
        for _ in range(self.num_agents):
            actions.append(joint_idx % self.action_size)
            joint_idx //= self.action_size
        return actions

    def act(self, obs_dict: dict, explore: bool = True) -> list:
        """Epsilon-greedy over joint action space."""
        eps = self.epsilon_schedule.value(self.total_steps) if explore else 0.0
        if explore and np.random.random() < eps:
            return [np.random.randint(self.action_size) for _ in range(self.num_agents)]

        global_obs = torch.tensor(obs_dict["global"], dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.q_net(global_obs)
        joint_idx = q_vals.argmax(dim=1).item()
        return self._decode_joint_action(joint_idx)

    def store_transition(self, obs_dict, actions, reward, next_obs_dict, done):
        """Store joint transition."""
        joint_action = self._encode_joint_action(actions)
        self.buffer.push(
            obs_dict["global"],
            joint_action,
            reward,
            next_obs_dict["global"],
            done,
        )
        self.total_steps += 1

    def update(self) -> dict | None:
        """One gradient step. Returns metrics dict or None."""
        if len(self.buffer) < self.learning_starts:
            return None

        batch = self.buffer.sample(self.batch_size, self.device)
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        q_vals = self.q_net(obs)
        q_current = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_online = self.q_net(next_obs)
            best_actions = next_q_online.argmax(dim=1)
            next_q_target = self.target_net(next_obs)
            next_q = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            td_target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = nn.functional.mse_loss(q_current, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        # Soft update target
        tau = self.tau
        for param, target_param in zip(self.q_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return {
            "loss": loss.item(),
            "epsilon": self.epsilon_schedule.value(self.total_steps),
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(state["q_net"])
        self.target_net.load_state_dict(state["target_net"])
