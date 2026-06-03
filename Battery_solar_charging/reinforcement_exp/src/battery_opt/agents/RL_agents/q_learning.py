"""Tabular Q-learning agent for BatteryEnv."""

from __future__ import annotations

import random

import numpy as np

# AU tariff thresholds (A$/kWh): off-peak 0.20 | shoulder 0.28 | peak 0.45
_RATE_THRESHOLDS = (0.22, 0.35)


def discretise(info: dict, battery_capacity_kwh: float) -> tuple:
    """Map BatteryEnv state info to a discrete Q-table key.

    Uses three features:
    - SoC split into 5 equal bins over [0, capacity]
    - Import tariff split into 3 bins (off-peak / shoulder / peak)
    - Solar surplus binary (above/below 0.05 kWh threshold)

    The tariff bin already encodes time-of-day implicitly (off-peak at night,
    peak in the evening), so no explicit hour feature is needed.

    Args:
        info: Info dict from :meth:`BatteryEnv.reset` or :meth:`BatteryEnv.step`.
        battery_capacity_kwh: Battery capacity used for SoC binning.

    Returns:
        Tuple ``(soc_bin, rate_bin, solar_bin)``.
    """
    soc_bin = min(int(info["battery_soc_kwh"] / battery_capacity_kwh * 5), 4)

    rate = info["import_tariff"]
    if rate <= _RATE_THRESHOLDS[0]:
        rate_bin = 0   # off-peak
    elif rate <= _RATE_THRESHOLDS[1]:
        rate_bin = 1   # shoulder
    else:
        rate_bin = 2   # peak

    solar_bin = 1 if info["surplus_kwh"] > 0.05 else 0

    return (soc_bin, rate_bin, solar_bin)


class QLearningAgent:
    """Tabular Q-learning agent compatible with BatteryEnv.

    Actions match BatteryEnv's ``Discrete(3)`` space:
    ``0`` = discharge, ``1`` = idle, ``2`` = charge.

    Args:
        battery_capacity_kwh: Battery capacity, forwarded to :func:`discretise`.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon_start: Initial ε-greedy exploration probability.
        epsilon_end: Minimum exploration probability.
        epsilon_decay_episodes: Episodes over which ε decays linearly to its
            minimum. Decay is applied once per episode via :meth:`decay_epsilon`.
    """

    N_ACTIONS = 3

    def __init__(
        self,
        battery_capacity_kwh: float,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 500,
    ) -> None:
        self.battery_capacity_kwh = battery_capacity_kwh
        self.alpha           = alpha
        self.gamma           = gamma
        self.epsilon         = epsilon_start
        self.epsilon_end     = epsilon_end
        self._epsilon_decay  = (epsilon_start - epsilon_end) / epsilon_decay_episodes
        self.q_table: dict[tuple, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def state_from_info(self, info: dict) -> tuple:
        """Discretise a BatteryEnv info dict into a Q-table key."""
        return discretise(info, self.battery_capacity_kwh)

    def select_action(self, state: tuple, training: bool = True) -> int:
        """ε-greedy action selection.

        Args:
            state: Discrete state tuple from :func:`discretise`.
            training: If ``False``, always act greedily.

        Returns:
            Action integer in ``{0, 1, 2}``.
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.N_ACTIONS)
        return int(np.argmax(self._q(state)))

    def update(
        self, state: tuple, action: int, reward: float, next_state: tuple
    ) -> None:
        """Standard Q-learning update with bootstrapped next-state value."""
        td_target = reward + self.gamma * float(self._q(next_state).max())
        self._q(state)[action] += self.alpha * (td_target - self._q(state)[action])

    def update_terminal(self, state: tuple, action: int, reward: float) -> None:
        """Q-update for the final step of an episode (no bootstrap)."""
        self._q(state)[action] += self.alpha * (reward - self._q(state)[action])

    def decay_epsilon(self) -> None:
        """Reduce exploration rate by one step. Call once per episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon - self._epsilon_decay)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _q(self, state: tuple) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.N_ACTIONS)
        return self.q_table[state]
