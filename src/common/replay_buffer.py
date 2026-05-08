"""Replay buffer for DQN-family agents."""
import numpy as np
import torch
from typing import NamedTuple


class Transition(NamedTuple):
    obs: np.ndarray        # shape: (obs_size,) for per-agent or (global_obs_size,) for CDQN
    action: int            # scalar action index
    reward: float
    next_obs: np.ndarray   # same shape as obs
    done: bool


class ReplayBuffer:
    """Circular replay buffer storing transitions as numpy arrays."""

    def __init__(self, capacity: int, obs_size: int):
        """
        Args:
            capacity: max number of transitions
            obs_size: dimension of observation vector
        """
        self._capacity = capacity
        self._obs_size = obs_size
        self._obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def push(self, obs, action, reward, next_obs, done):
        """Add one transition."""
        self._obs[self._ptr] = obs
        self._actions[self._ptr] = action
        self._rewards[self._ptr] = reward
        self._next_obs[self._ptr] = next_obs
        self._dones[self._ptr] = float(done)
        self._ptr = (self._ptr + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int, device: str = "cpu") -> dict:
        """
        Sample batch_size transitions.
        Returns dict with keys: obs, actions, rewards, next_obs, dones
        All values are torch.Tensor on device.
        """
        idx = np.random.randint(0, self._size, size=batch_size)
        return {
            "obs": torch.tensor(self._obs[idx], dtype=torch.float32, device=device),
            "actions": torch.tensor(self._actions[idx], dtype=torch.int64, device=device),
            "rewards": torch.tensor(self._rewards[idx], dtype=torch.float32, device=device),
            "next_obs": torch.tensor(self._next_obs[idx], dtype=torch.float32, device=device),
            "dones": torch.tensor(self._dones[idx], dtype=torch.float32, device=device),
        }

    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity
