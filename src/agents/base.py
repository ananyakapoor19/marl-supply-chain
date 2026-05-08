from __future__ import annotations

"""Abstract base class for all agents."""
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Interface all RL agents must implement."""

    @abstractmethod
    def act(self, obs_dict: dict, explore: bool = True) -> list:
        """
        Choose actions for all agents.

        Args:
            obs_dict: dict with keys "agents" (list of per-agent obs) and "global"
            explore: if True, use epsilon-greedy; if False, greedy
        Returns:
            list of integer actions, one per node
        """

    @abstractmethod
    def update(self) -> dict | None:
        """
        Sample from replay buffer and do one gradient update.
        Returns dict of training metrics (loss, etc.) or None if not enough data.
        """

    @abstractmethod
    def store_transition(self, obs_dict, actions, reward, next_obs_dict, done):
        """Store one environment transition."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model weights."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights."""
