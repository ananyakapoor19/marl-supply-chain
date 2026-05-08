"""(s, S) reorder-point / order-up-to policy for supply chain inventory management."""

from __future__ import annotations

import numpy as np


class SSPolicyAgent:
    """(s, S) policy: if inventory_position < s_i, order S_i - inventory_position."""

    def __init__(
        self,
        s_levels: list[float],
        S_levels: list[float],
        max_order: int = 20,
    ) -> None:
        """
        Initialise with per-node reorder and order-up-to levels.

        Args:
            s_levels: Reorder-point s_i per node.
            S_levels: Order-up-to level S_i per node.
            max_order: Maximum order quantity to clip actions.
        """
        if len(s_levels) != len(S_levels):
            raise ValueError("s_levels and S_levels must have the same length.")
        self.s_levels = list(s_levels)
        self.S_levels = list(S_levels)
        self.max_order = max_order

    def act(self, obs_dict: dict, node_idx: int) -> int:
        """
        Compute order for a single node.

        If inventory_position < s_i: order = clip(S_i - inventory_position, 0, max_order)
        Otherwise: order = 0
        """
        obs = obs_dict["agents"][node_idx]
        inventory = float(obs[0])
        pipeline = obs[2:]
        inventory_position = inventory + float(np.sum(pipeline))
        s = self.s_levels[node_idx]
        S = self.S_levels[node_idx]
        if inventory_position < s:
            order = max(0.0, S - inventory_position)
            return int(min(order, self.max_order))
        return 0

    def act_all(self, obs_dict: dict) -> list[int]:
        """Return actions for all nodes."""
        return [self.act(obs_dict, i) for i in range(len(self.s_levels))]
