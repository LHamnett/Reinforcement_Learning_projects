"""Gymnasium-compatible battery optimisation environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Observation normaliser
# ---------------------------------------------------------------------------


_CYCLICAL_COLS = [
    "hour_sin", "hour_cos",
    "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos",
]


@dataclass
class ObsNormaliser:
    """Z-score statistics for the five continuous data features.

    Cyclical time features (sin/cos encodings) are already in [-1, 1] by
    construction and are passed through unchanged.  Battery SoC is normalised
    separately as ``soc / capacity`` → [0, 1].

    Attributes:
        mean: Per-feature mean, shape (5,).
        std:  Per-feature standard deviation, shape (5,), clipped to ≥1e-8.
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_episodes(cls, episodes: dict[int, pd.DataFrame]) -> ObsNormaliser:
        """Compute normalisation statistics from training episodes only."""
        all_rows = pd.concat(episodes.values(), ignore_index=True)
        obs = np.column_stack([
            all_rows["total_consumption"].values,
            all_rows["gg_kwh"].values,
            (all_rows["gg_kwh"] - all_rows["total_consumption"]).clip(lower=0).values,
            (all_rows["total_consumption"] - all_rows["gg_kwh"]).clip(lower=0).values,
            all_rows["import_tariff"].values,
        ]).astype(np.float32)
        return cls(
            mean=obs.mean(axis=0),
            std=obs.std(axis=0).clip(min=1e-8),
        )

    def normalise(
        self,
        data_features: np.ndarray,
        cyclical_features: np.ndarray,
        battery_soc: float,
        battery_capacity: float,
    ) -> np.ndarray:
        """Return the full 12-element normalised observation vector.

        Writes into a single pre-allocated array to avoid multiple
        intermediate allocations from ``np.concatenate``.
        """
        out = np.empty(12, dtype=np.float32)
        out[:5]   = (data_features - self.mean) / self.std
        out[5:11] = cyclical_features
        out[11]   = battery_soc / battery_capacity
        return out


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class BatteryEnv(gym.Env):
    """Gymnasium environment for home battery charge/discharge optimisation.

    One episode = one full year of hourly data for one customer (~8 760 steps).

    **Observation space** — 12 features, ``Box(-inf, inf, (12,), float32)``:

    +---------+------------------------------+------------------+
    | Index   | Feature                      | Normalisation    |
    +=========+==============================+==================+
    | 0       | Total consumption (kWh)      | z-score (train)  |
    | 1       | Solar generation (kWh)       | z-score (train)  |
    | 2       | Solar surplus (kWh)          | z-score (train)  |
    | 3       | Unmet demand (kWh)           | z-score (train)  |
    | 4       | Import tariff (A$/kWh)       | z-score (train)  |
    | 5       | sin(2π · hour / 24)          | pass-through     |
    | 6       | cos(2π · hour / 24)          | pass-through     |
    | 7       | sin(2π · day_of_week / 7)    | pass-through     |
    | 8       | cos(2π · day_of_week / 7)    | pass-through     |
    | 9       | sin(2π · month / 12)         | pass-through     |
    | 10      | cos(2π · month / 12)         | pass-through     |
    | 11      | Battery SoC                  | soc / capacity   |
    +---------+------------------------------+------------------+

    **Action space** — ``Discrete(3)``:

    +--------+----------------+-----------------+
    | Action | Meaning        | Internal action |
    +========+================+=================+
    | 0      | Discharge      | −1              |
    | 1      | Idle           | 0               |
    | 2      | Charge         | +1              |
    +--------+----------------+-----------------+

    **Reward**: ``-cost`` per step (minimise electricity cost).

    Performance note: call :meth:`set_skip_obs` before running a tabular agent
    (e.g. Q-learning) that ignores the observation.  This skips computing the
    normalised observation on every step, giving a significant speedup.

    Args:
        episodes: Mapping of customer_id → episode DataFrame (train or val),
            as returned by :func:`~battery_opt.data.create_episodes.make_episodes`.
        normaliser: Fitted :class:`ObsNormaliser` (computed from training data only).
        battery_capacity_kwh: Maximum battery energy (kWh).
        charge_efficiency: Round-trip charge efficiency (0–1).
        discharge_efficiency: Discharge delivery efficiency (0–1).
        max_charge_rate_kw: Maximum charge power per hour (kW).
        max_discharge_rate_kw: Maximum discharge power per hour (kW).
    """

    metadata = {"render_modes": []}

    _ENV_ACTIONS = [-1, 0, 1]  # Discrete index 0/1/2 → battery action -1/0/+1
    N_OBS = 12  # 5 z-scored + 6 cyclical (pass-through) + 1 soc/capacity

    def __init__(
        self,
        episodes: dict[int, pd.DataFrame],
        normaliser: ObsNormaliser,
        battery_capacity_kwh: float = 10.0,
        charge_efficiency: float = 0.90,
        discharge_efficiency: float = 0.90,
        max_charge_rate_kw: float = 5.0,
        max_discharge_rate_kw: float = 5.0,
    ) -> None:
        super().__init__()

        self.episodes = episodes
        self.normaliser = normaliser
        self.battery_capacity_kwh = battery_capacity_kwh
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.max_charge_rate_kw = max_charge_rate_kw
        self.max_discharge_rate_kw = max_discharge_rate_kw
        self._skip_obs: bool = False

        # Pre-extract episode data as numpy arrays to avoid per-step dict
        # lookups and repeated to_dict("records") calls at each reset().
        self._ep_consumption: dict[int, np.ndarray] = {}
        self._ep_solar:       dict[int, np.ndarray] = {}
        self._ep_tariff:      dict[int, np.ndarray] = {}
        self._ep_cycl:        dict[int, np.ndarray] = {}  # shape (n, 6)
        self._ep_n_steps:     dict[int, int]         = {}
        for cid, df in episodes.items():
            self._ep_consumption[cid] = df["total_consumption"].values.astype(np.float64)
            self._ep_solar[cid]       = df["gg_kwh"].values.astype(np.float64)
            self._ep_tariff[cid]      = df["import_tariff"].values.astype(np.float64)
            self._ep_cycl[cid]        = df[_CYCLICAL_COLS].values.astype(np.float32)
            self._ep_n_steps[cid]     = len(df)

        low = np.array(
            [-np.inf] * 5 +  # z-scored data features: unbounded
            [-1.0]   * 6 +   # cyclical sin/cos: always in [-1, 1]
            [0.0],            # battery SoC / capacity: always in [0, 1]
            dtype=np.float32,
        )
        high = np.array(
            [np.inf] * 5 +
            [1.0]   * 6 +
            [1.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        # episode state — populated by reset()
        self._step_idx:    int       = 0
        self._battery_soc: float     = 0.0
        self._customer_id: int | None = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Start a new episode."""
        super().reset(seed=seed)

        if options and "customer_id" in options:
            self._customer_id = int(options["customer_id"])
        else:
            self._customer_id = int(
                self.np_random.choice(list(self.episodes.keys()))
            )

        self._step_idx    = 0
        self._battery_soc = 0.0

        obs = np.zeros(self.N_OBS, dtype=np.float32) if self._skip_obs else self._get_obs()
        return obs, self._state_info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply ``action`` for one hour and return the transition."""
        assert self._customer_id is not None, "Call reset() before step()."

        cid = self._customer_id
        idx = self._step_idx

        consumption = self._ep_consumption[cid][idx]
        solar       = self._ep_solar[cid][idx]
        rate        = self._ep_tariff[cid][idx]
        surplus     = max(solar - consumption, 0.0)
        demand      = max(consumption - solar, 0.0)

        env_action = self._ENV_ACTIONS[action]

        if env_action == 1:    # charge
            charge_rate = min(
                surplus,
                self.max_charge_rate_kw,
                self.battery_capacity_kwh - self._battery_soc,
            )
            self._battery_soc += charge_rate * self.charge_efficiency
            grid_demand = demand

        elif env_action == -1:  # discharge
            discharge_rate = min(demand, self.max_discharge_rate_kw, self._battery_soc)
            self._battery_soc -= discharge_rate
            grid_demand = max(demand - discharge_rate * self.discharge_efficiency, 0.0)

        else:                   # idle
            grid_demand = demand

        cost   = grid_demand * rate
        reward = -cost

        self._step_idx += 1
        truncated  = self._step_idx >= self._ep_n_steps[cid]
        terminated = False

        if truncated or self._skip_obs:
            obs = np.zeros(self.N_OBS, dtype=np.float32)
        else:
            obs = self._get_obs()

        return obs, float(reward), terminated, truncated, {
            "cost":            cost,
            "grid_demand_kwh": grid_demand,
            **self._state_info(),
        }

    def set_skip_obs(self, skip: bool) -> None:
        """Toggle observation computation.

        Set ``skip=True`` before training a tabular agent that never reads the
        observation — avoids computing the normalised obs on every step.
        Reset to ``skip=False`` before evaluation or use with a neural agent.
        """
        self._skip_obs = skip

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _state_info(self) -> dict:
        """Raw state context at the current step, for Q-table discretisation."""
        cid  = self._customer_id
        base = {"customer_id": cid, "battery_soc_kwh": self._battery_soc}

        if self._step_idx >= self._ep_n_steps[cid]:
            return {**base, "surplus_kwh": 0.0, "import_tariff": 0.0}

        idx         = self._step_idx
        consumption = self._ep_consumption[cid][idx]
        solar       = self._ep_solar[cid][idx]
        return {
            **base,
            "surplus_kwh":   max(solar - consumption, 0.0),
            "import_tariff": float(self._ep_tariff[cid][idx]),
        }

    def _get_obs(self) -> np.ndarray:
        cid  = self._customer_id
        idx  = self._step_idx
        cons = self._ep_consumption[cid][idx]
        sol  = self._ep_solar[cid][idx]

        data = np.empty(5, dtype=np.float32)
        data[0] = cons
        data[1] = sol
        data[2] = max(sol - cons, 0.0)
        data[3] = max(cons - sol, 0.0)
        data[4] = self._ep_tariff[cid][idx]

        return self.normaliser.normalise(
            data, self._ep_cycl[cid][idx], self._battery_soc, self.battery_capacity_kwh
        )
