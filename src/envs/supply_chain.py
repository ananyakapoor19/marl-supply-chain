from __future__ import annotations

"""
3-node serial supply chain environment for multi-agent reinforcement learning.
"""



import numpy as np
import gymnasium
from gymnasium import spaces
from typing import Any


class SupplyChainEnv(gymnasium.Env):
    """
    3-node serial supply chain environment.

    Nodes: 0=supplier, 1=warehouse, 2=retailer
    Customer demand arrives at retailer (node 2).
    Each node orders from the node upstream.
    Supplier has infinite upstream supply.

    Observation per agent i:
        [inventory_i, incoming_demand_i, *pipeline_i]  (float32)

    Global observation: concatenation of all per-agent obs.

    Action per agent: discrete order quantity in {0, 1, ..., max_order}

    Reward: -total_cost (shared across all agents, cooperative)
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict | None = None) -> None:
        """Initialise environment from config dict."""
        cfg = config or {}
        self.num_agents: int = int(cfg.get("num_agents", 3))
        self.lead_times: list[int] = [int(x) for x in cfg.get("lead_times", [2, 2, 2])]
        self.max_order: int = int(cfg.get("max_order", 20))
        self.holding_cost: float = float(cfg.get("holding_cost", 1.0))
        self.stockout_cost: float = float(cfg.get("stockout_cost", 2.0))
        self.initial_inventory: float = float(cfg.get("initial_inventory", 10.0))
        self.episode_length: int = int(cfg.get("episode_length", 100))
        self.demand_mode: str = str(cfg.get("demand_mode", "stationary"))
        self.demand_lambda: float = float(cfg.get("demand_lambda", 5.0))
        raw_schedule = cfg.get(
            "nonstationary_schedule",
            [{"step": 0, "lambda": 5.0}, {"step": 50, "lambda": 12.0}],
        )
        self.nonstationary_schedule: list[dict] = [
            {"step": int(e["step"]), "lambda": float(e["lambda"])} for e in raw_schedule
        ]
        seed_val = cfg.get("seed", None)

        # Derived sizes
        self.obs_sizes: list[int] = [2 + self.lead_times[i] for i in range(self.num_agents)]
        self.global_obs_size: int = sum(self.obs_sizes)
        self.action_size: int = self.max_order + 1
        self.joint_action_size: int = self.action_size ** self.num_agents

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.global_obs_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(
            [self.action_size] * self.num_agents, dtype=np.int64
        )

        # Internal state (initialised properly in reset)
        self.inventory: np.ndarray = np.zeros(self.num_agents, dtype=np.float32)
        self.pipelines: list[np.ndarray] = [
            np.zeros(lt, dtype=np.float32) for lt in self.lead_times
        ]
        self.downstream_orders: np.ndarray = np.zeros(self.num_agents, dtype=np.float32)
        self.current_step: int = 0
        self.last_actions: list[int] = [0] * self.num_agents
        self.current_lambda: float = self.demand_lambda

        # RNG
        self.np_rng = np.random.default_rng(seed_val)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_agents(self) -> int:
        """Number of agents (alias)."""
        return self.num_agents

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_current_lambda(self, step: int) -> float:
        """Return demand lambda for *step* given the schedule."""
        if self.demand_mode == "nonstationary":
            current = self.demand_lambda
            for entry in self.nonstationary_schedule:
                if entry["step"] <= step:
                    current = entry["lambda"]
            return current
        return self.demand_lambda

    def _build_obs(self, customer_demand: float) -> dict:
        """Build obs_dict from current state."""
        agent_obs: list[np.ndarray] = []
        for i in range(self.num_agents):
            if i < self.num_agents - 1:
                # incoming demand = what the downstream agent ordered last step
                incoming_demand = float(self.downstream_orders[i + 1])
            else:
                # retailer's incoming demand = customer demand (last realised)
                incoming_demand = float(customer_demand)
            obs = np.concatenate(
                [
                    np.array([self.inventory[i], incoming_demand], dtype=np.float32),
                    self.pipelines[i].astype(np.float32),
                ]
            )
            agent_obs.append(obs)
        global_obs = np.concatenate(agent_obs).astype(np.float32)
        return {"agents": agent_obs, "global": global_obs}

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        """Reset environment; return (obs_dict, info)."""
        super().reset(seed=seed)
        if seed is not None:
            self.np_rng = np.random.default_rng(seed)

        self.inventory = np.full(self.num_agents, self.initial_inventory, dtype=np.float32)
        self.pipelines = [np.zeros(lt, dtype=np.float32) for lt in self.lead_times]
        self.downstream_orders = np.zeros(self.num_agents, dtype=np.float32)
        self.current_step = 0
        self.last_actions = [0] * self.num_agents
        self.current_lambda = self._get_current_lambda(0)

        obs_dict = self._build_obs(customer_demand=0.0)
        info = {"step": 0, "current_lambda": self.current_lambda}
        return obs_dict, info

    def step(
        self, actions: list[int] | np.ndarray
    ) -> tuple[dict, float, bool, bool, dict]:
        """
        Advance environment by one step.

        actions: list/array of num_agents integers in [0, max_order].
        Returns (obs_dict, reward, terminated, truncated, info).
        """
        actions = [int(a) for a in actions]

        # ---- 1. Determine current lambda and sample customer demand ----
        self.current_lambda = self._get_current_lambda(self.current_step)
        customer_demand = float(self.np_rng.poisson(self.current_lambda))

        # ---- 2. Receive incoming shipments (pipeline front arrives) ----
        incoming_shipments = np.zeros(self.num_agents, dtype=np.float32)
        for i in range(self.num_agents):
            if len(self.pipelines[i]) > 0:
                incoming_shipments[i] = self.pipelines[i][0]
                self.inventory[i] += self.pipelines[i][0]
                # Shift pipeline left (drop front, append zero at back)
                self.pipelines[i] = np.concatenate(
                    [self.pipelines[i][1:], np.array([0.0], dtype=np.float32)]
                )

        # ---- 3. Fulfill demand (with backlog — inventory can go negative) ----
        # Node demands:
        # retailer (node 2): customer_demand
        # warehouse (node 1): last order placed by retailer
        # supplier (node 0): last order placed by warehouse
        node_demand = np.zeros(self.num_agents, dtype=np.float32)
        node_demand[self.num_agents - 1] = customer_demand  # retailer
        for i in range(self.num_agents - 1):
            node_demand[i] = self.downstream_orders[i + 1]  # warehouse gets retailer's last order, etc.

        for i in range(self.num_agents):
            self.inventory[i] -= node_demand[i]  # backlog allowed (inventory can go negative)

        # ---- 4. Compute costs ----
        costs = np.array(
            [
                self.holding_cost * max(float(self.inventory[i]), 0.0)
                + self.stockout_cost * max(-float(self.inventory[i]), 0.0)
                for i in range(self.num_agents)
            ],
            dtype=np.float32,
        )
        total_cost = float(np.sum(costs))
        reward = -total_cost

        # ---- 5. Place new orders (actions) into pipelines ----
        for i in range(self.num_agents):
            order = float(actions[i])
            if len(self.pipelines[i]) > 0:
                self.pipelines[i][-1] += order
            # record so downstream knows demand next step
            self.downstream_orders[i] = order

        self.last_actions = actions
        self.current_step += 1

        # ---- 6. Build next observation ----
        obs_dict = self._build_obs(customer_demand=customer_demand)

        # ---- 7. Termination ----
        terminated = False
        truncated = self.current_step >= self.episode_length

        info = {
            "per_node_costs": costs.tolist(),
            "per_node_orders": actions,
            "per_node_inventory": self.inventory.tolist(),
            "customer_demand": customer_demand,
            "step": self.current_step,
        }
        return obs_dict, reward, terminated, truncated, info

    def render(self) -> None:
        """Render is not implemented (no-op)."""
        pass
