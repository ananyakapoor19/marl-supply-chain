"""Base-stock policy for supply chain inventory management."""

from __future__ import annotations

import numpy as np


class BaseStockAgent:
    """Base-stock policy: order max(0, S_i - inventory_position_i)."""

    def __init__(self, base_stock_levels: list[float], max_order: int = 20) -> None:
        """
        Initialise with per-node base-stock levels.

        Args:
            base_stock_levels: S_i target inventory position per node.
            max_order: Maximum order quantity to clip actions.
        """
        self.base_stock_levels = list(base_stock_levels)
        self.max_order = max_order

    def act(self, obs_dict: dict, node_idx: int) -> int:
        """
        Compute order for a single node.

        obs_dict["agents"][node_idx] = [inventory, incoming_demand, *pipeline]
        inventory_position = inventory + sum(pipeline)
        order = clip(max(0, S_i - inventory_position), 0, max_order)
        """
        obs = obs_dict["agents"][node_idx]
        inventory = float(obs[0])
        pipeline = obs[2:]  # elements beyond inventory and incoming_demand
        inventory_position = inventory + float(np.sum(pipeline))
        target = self.base_stock_levels[node_idx]
        order = max(0.0, target - inventory_position)
        return int(min(order, self.max_order))

    def act_all(self, obs_dict: dict) -> list[int]:
        """Return actions for all nodes."""
        return [self.act(obs_dict, i) for i in range(len(self.base_stock_levels))]
