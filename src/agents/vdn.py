from __future__ import annotations

"""Value Decomposition Networks: Q_total = sum of per-agent Q-values."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from src.common.networks import MLP
from src.common.schedules import LinearEpsilonSchedule
from src.agents.base import BaseAgent


class JointReplayBuffer:
    """Replay buffer storing joint transitions (all agents together)."""

    def __init__(self, capacity: int, obs_sizes: list, num_agents: int):
        self._capacity = capacity
        self._num_agents = num_agents
        self._obs_sizes = obs_sizes
        # Store per-agent obs and actions + shared reward + done
        self._obs = [np.zeros((capacity, obs_sizes[i]), dtype=np.float32) for i in range(num_agents)]
        self._next_obs = [np.zeros((capacity, obs_sizes[i]), dtype=np.float32) for i in range(num_agents)]
        self._actions = np.zeros((capacity, num_agents), dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def push(self, obs_list, actions, reward, next_obs_list, done):
        """Store one joint transition."""
        for i in range(self._num_agents):
            self._obs[i][self._ptr] = obs_list[i]
            self._next_obs[i][self._ptr] = next_obs_list[i]
        self._actions[self._ptr] = actions
        self._rewards[self._ptr] = reward
        self._dones[self._ptr] = float(done)
        self._ptr = (self._ptr + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int, device: str = "cpu") -> dict:
        """Sample batch, return tensors on device."""
        idx = np.random.randint(0, self._size, size=batch_size)
        return {
            "obs": [torch.tensor(self._obs[i][idx], dtype=torch.float32, device=device) for i in range(self._num_agents)],
            "next_obs": [torch.tensor(self._next_obs[i][idx], dtype=torch.float32, device=device) for i in range(self._num_agents)],
            "actions": torch.tensor(self._actions[idx], dtype=torch.int64, device=device),
            "rewards": torch.tensor(self._rewards[idx], dtype=torch.float32, device=device),
            "dones": torch.tensor(self._dones[idx], dtype=torch.float32, device=device),
        }

    def __len__(self) -> int:
        return self._size


class VDNAgent(BaseAgent):
    """
    Value Decomposition Networks.

    Per-agent Q-networks take local observations.
    Q_total = sum_i Q_i(o_i, a_i).
    Trained on shared global reward via joint TD loss.
    Execution is fully decentralized (each agent uses only its Q_i).
    """

    def __init__(self, config: dict, obs_sizes: list, action_size: int, num_agents: int = 3):
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

        # Single optimizer over all parameters
        all_params = []
        for net in self.q_nets:
            all_params += list(net.parameters())
        lr = config.get("lr", 5e-4)
        self.optimizer = optim.Adam(all_params, lr=lr)

        buf_size = config.get("buffer_size", 100_000)
        self.buffer = JointReplayBuffer(buf_size, obs_sizes, num_agents)

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
        """Decentralized epsilon-greedy: each agent uses only its Q_i."""
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
        """Store joint transition."""
        self.buffer.push(
            obs_dict["agents"],
            actions,
            reward,
            next_obs_dict["agents"],
            done,
        )
        self.total_steps += 1

    def update(self) -> dict | None:
        """VDN joint update: minimize (r + gamma * max_a Q_total_target - Q_total)^2."""
        if len(self.buffer) < self.learning_starts:
            return None

        batch = self.buffer.sample(self.batch_size, self.device)
        obs_list = batch["obs"]
        next_obs_list = batch["next_obs"]
        actions = batch["actions"]   # (batch, num_agents)
        rewards = batch["rewards"]   # (batch,)
        dones = batch["dones"]       # (batch,)

        # Current Q_total = sum_i Q_i(o_i, a_i)
        q_total = torch.zeros(self.batch_size, device=self.device)
        for i in range(self.num_agents):
            q_vals_i = self.q_nets[i](obs_list[i])
            q_i = q_vals_i.gather(1, actions[:, i].unsqueeze(1)).squeeze(1)
            q_total = q_total + q_i

        # Target Q_total = sum_i max_a Q_target_i(o'_i, a)
        with torch.no_grad():
            q_total_target = torch.zeros(self.batch_size, device=self.device)
            for i in range(self.num_agents):
                # Double DQN: online net selects action, target net evaluates
                next_q_online_i = self.q_nets[i](next_obs_list[i])
                best_a_i = next_q_online_i.argmax(dim=1)
                next_q_target_i = self.target_nets[i](next_obs_list[i])
                next_q_i = next_q_target_i.gather(1, best_a_i.unsqueeze(1)).squeeze(1)
                q_total_target = q_total_target + next_q_i
            td_target = rewards + self.gamma * q_total_target * (1.0 - dones)

        loss = nn.functional.mse_loss(q_total, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        for net in self.q_nets:
            nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip)
        self.optimizer.step()

        # Soft update all target networks
        tau = self.tau
        for i in range(self.num_agents):
            for param, target_param in zip(self.q_nets[i].parameters(), self.target_nets[i].parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return {
            "loss": loss.item(),
            "epsilon": self.epsilon_schedule.value(self.total_steps),
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {f"q_net_{i}": self.q_nets[i].state_dict() for i in range(self.num_agents)}
        state.update({f"target_net_{i}": self.target_nets[i].state_dict() for i in range(self.num_agents)})
        torch.save(state, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        for i in range(self.num_agents):
            self.q_nets[i].load_state_dict(state[f"q_net_{i}"])
            self.target_nets[i].load_state_dict(state[f"target_net_{i}"])
