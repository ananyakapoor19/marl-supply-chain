"""Tests for ReplayBuffer."""
import numpy as np
import pytest
import torch

from src.common.replay_buffer import ReplayBuffer


def make_transition(obs_size: int, obs_val: float = 1.0):
    """Helper to create a dummy transition."""
    obs = np.ones(obs_size, dtype=np.float32) * obs_val
    action = 3
    reward = -2.5
    next_obs = np.ones(obs_size, dtype=np.float32) * (obs_val + 1.0)
    done = False
    return obs, action, reward, next_obs, done


def test_replay_buffer_push_sample():
    """Buffer stores and retrieves transitions correctly."""
    obs_size = 8
    buf = ReplayBuffer(capacity=1000, obs_size=obs_size)

    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    next_obs = obs + 10.0
    action = 5
    reward = -3.0
    done = True

    buf.push(obs, action, reward, next_obs, done)
    assert len(buf) == 1

    batch = buf.sample(1, device="cpu")
    np.testing.assert_array_almost_equal(batch["obs"].numpy()[0], obs)
    np.testing.assert_array_almost_equal(batch["next_obs"].numpy()[0], next_obs)
    assert batch["actions"].item() == action
    assert abs(batch["rewards"].item() - reward) < 1e-5
    assert batch["dones"].item() == 1.0  # done=True -> 1.0


def test_replay_buffer_circular():
    """Buffer wraps around when full."""
    obs_size = 4
    capacity = 10
    buf = ReplayBuffer(capacity=capacity, obs_size=obs_size)

    # Push 15 transitions (5 over capacity)
    for i in range(15):
        obs = np.full(obs_size, float(i), dtype=np.float32)
        next_obs = np.full(obs_size, float(i + 1), dtype=np.float32)
        buf.push(obs, i % 21, float(-i), next_obs, False)

    # Size should be capped at capacity
    assert len(buf) == capacity

    # The pointer should have wrapped: ptr = 15 % 10 = 5
    # Oldest stored value should be index 5 (transition 5..14 stored)
    # Check that all stored obs values are in range [5, 14]
    for i in range(capacity):
        stored_val = buf._obs[i][0]
        assert 5 <= stored_val <= 14, f"Unexpected stored value {stored_val} at slot {i}"


def test_replay_buffer_sample_shapes():
    """Sample returns correct tensor shapes and dtypes."""
    obs_size = 12
    capacity = 500
    batch_size = 32
    buf = ReplayBuffer(capacity=capacity, obs_size=obs_size)

    # Fill buffer with enough transitions
    for i in range(200):
        obs, action, reward, next_obs, done = make_transition(obs_size, float(i))
        buf.push(obs, action, reward, next_obs, done)

    batch = buf.sample(batch_size, device="cpu")

    # Check keys
    assert set(batch.keys()) == {"obs", "actions", "rewards", "next_obs", "dones"}

    # Check shapes
    assert batch["obs"].shape == (batch_size, obs_size)
    assert batch["next_obs"].shape == (batch_size, obs_size)
    assert batch["actions"].shape == (batch_size,)
    assert batch["rewards"].shape == (batch_size,)
    assert batch["dones"].shape == (batch_size,)

    # Check dtypes
    assert batch["obs"].dtype == torch.float32
    assert batch["next_obs"].dtype == torch.float32
    assert batch["actions"].dtype == torch.int64
    assert batch["rewards"].dtype == torch.float32
    assert batch["dones"].dtype == torch.float32


def test_replay_buffer_len_and_capacity():
    """len() grows up to capacity, capacity property works."""
    buf = ReplayBuffer(capacity=50, obs_size=4)
    assert len(buf) == 0
    assert buf.capacity == 50

    for i in range(30):
        obs = np.zeros(4, dtype=np.float32)
        buf.push(obs, 0, 0.0, obs, False)
    assert len(buf) == 30

    for i in range(30):
        obs = np.zeros(4, dtype=np.float32)
        buf.push(obs, 0, 0.0, obs, False)
    assert len(buf) == 50  # capped at capacity
