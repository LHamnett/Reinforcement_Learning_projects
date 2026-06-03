"""Rule-based battery simulation environment (pre-Gymnasium version).

This module contains the original step-through battery simulator and the
basic self-consumption policy baseline. It will be superseded by the
Gymnasium-compatible ``BatteryEnv`` in Phase 2, but is kept here as a
reference implementation and to preserve the baseline policy results.
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Type alias for the policy callable
# ---------------------------------------------------------------------------

StateDict = dict[str, float]
PolicyFn = Callable[[StateDict], int]


# ---------------------------------------------------------------------------
# Baseline policies
# ---------------------------------------------------------------------------


def basic_self_consumption_policy(state: StateDict) -> int:
    """Charge when solar surplus is available; discharge to meet unmet demand.

    Args:
        state: Current environment state (see :class:`BatteryEnvV0`).

    Returns:
        ``1`` to charge, ``-1`` to discharge, ``0`` to idle.
    """
    if state["surplus_from_solar"] > 0:
        return 1
    if state["demand_remaining"] > 0:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Simulation environment
# ---------------------------------------------------------------------------


class BatteryEnvV0:
    """Step-through battery simulator for evaluating rule-based policies.

    Iterates over a pre-loaded hourly DataFrame, applies a policy at each
    timestep, and records battery state and costs.

    Args:
        battery_capacity_kwh: Maximum energy the battery can store (kWh).
        charge_efficiency: Fraction of surplus energy stored (0–1).
        discharge_efficiency: Fraction of stored energy delivered (0–1).
        max_charge_rate_kw: Maximum charge power (kW per hour).
        max_discharge_rate_kw: Maximum discharge power (kW per hour).
        cols_of_interest: If provided, only these columns are kept in the
            simulation DataFrame.
        data_df: Full combined cost+solar DataFrame (output of the data
            pipeline).
    """

    def __init__(
        self,
        battery_capacity_kwh: float = 10.0,
        charge_efficiency: float = 0.90,
        discharge_efficiency: float = 0.90,
        max_charge_rate_kw: float = 5.0,
        max_discharge_rate_kw: float = 5.0,
        cols_of_interest: list[str] | None = None,
        data_df: pd.DataFrame | None = None,
    ) -> None:
        self.battery_capacity_kwh = battery_capacity_kwh
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.max_charge_rate_kw = max_charge_rate_kw
        self.max_discharge_rate_kw = max_discharge_rate_kw

        if data_df is None:
            raise ValueError("data_df must be provided.")
        self.data_df = data_df
        self.cols_of_interest = cols_of_interest
        self.simulation_df = self._prepare_simulation_df()

    def _prepare_simulation_df(self) -> pd.DataFrame:
        df = self.data_df.copy()
        if self.cols_of_interest is not None:
            df = df[self.cols_of_interest]
        df["simulated_battery_charge"] = 0.0
        df["simulated_cost_per_hour"] = 0.0
        return df

    def simulate_policy(
        self,
        consumption_kwh: list[float],
        solar_kwh: list[float],
        tariff_import: list[float],
        policy: PolicyFn,
    ) -> None:
        """Run a full episode under ``policy`` and store results in-place.

        Args:
            consumption_kwh: Hourly household consumption (kWh).
            solar_kwh: Hourly solar generation (kWh).
            tariff_import: Hourly import tariff rate (£/kWh).
            policy: Callable mapping a state dict to an action (1/0/−1).
        """
        assert len(consumption_kwh) == len(solar_kwh), (
            "consumption_kwh and solar_kwh must have equal length."
        )

        battery_charge = 0.0
        actions: list[int] = []
        battery_levels: list[float] = []
        costs: list[float] = []

        for consumption, solar, rate in zip(
            consumption_kwh, solar_kwh, tariff_import, strict=True
        ):
            surplus = max(solar - consumption, 0.0)
            demand = max(consumption - solar, 0.0)

            state: StateDict = {
                "consumption_per_hour": consumption,
                "solar_generation_per_hour": solar,
                "surplus_from_solar": surplus,
                "demand_remaining": demand,
                "current_tarriff": rate,
                "battery_charge": battery_charge,
            }

            action = policy(state)
            actions.append(action)

            if action == 1:  # charge from solar surplus
                charge_rate = min(
                    surplus, self.max_charge_rate_kw, self.battery_capacity_kwh - battery_charge
                )
                battery_charge += charge_rate * self.charge_efficiency
                grid_demand = demand

            elif action == -1:  # discharge to meet grid demand
                discharge_rate = min(demand, self.max_discharge_rate_kw, battery_charge)
                delivered = discharge_rate * self.discharge_efficiency
                battery_charge -= discharge_rate
                grid_demand = max(demand - delivered, 0.0)

            else:  # idle
                grid_demand = demand

            battery_levels.append(battery_charge)
            costs.append(grid_demand * rate)

        self.simulation_df["simulated_battery_charge"] = battery_levels
        self.simulation_df["simulated_cost_per_hour"] = costs
        self.simulation_df["simulated_actions_taken"] = actions

    def visualise_simulation(self) -> None:
        """Plot consumption vs solar, battery charge, and actions over the episode."""
        df = self.simulation_df.copy()
        df["timestamp_min_artificial"] = pd.to_datetime(df["timestamp_min_artificial"])
        x = df["timestamp_min_artificial"]

        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 1, figure=fig)

        ax1 = fig.add_subplot(gs[0])
        ax1.plot(x, df["Total_consumption"], label="Consumption (kWh)", color="blue")
        ax1.plot(
            x, df["manchester_solar_generation_kw"], label="Solar Generation (kWh)", color="orange"
        )
        ax1.set_title("Consumption and Solar Generation")
        ax1.set_ylabel("kWh")
        ax1.legend()
        ax1.minorticks_on()
        ax1.grid(visible=True, which="both")

        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        solar_col = df["manchester_solar_generation_kw"]
        solar_surplus = np.clip(solar_col - df["Total_consumption"].values, 0, None)
        ax2.plot(x, solar_surplus, label="Solar Surplus (kWh)", color="green")
        ax2.plot(x, df["simulated_battery_charge"], label="Battery Charge (kWh)", color="purple")
        ax2.axhline(y=self.battery_capacity_kwh, color="red", linestyle="--", label="Max Capacity")
        ax2.set_title("Solar Surplus and Battery Charge")
        ax2.set_ylabel("kWh")
        ax2.legend()
        ax2.minorticks_on()
        ax2.grid(visible=True, which="both")

        ax3 = fig.add_subplot(gs[2])
        ax3.step(x, df["simulated_actions_taken"], color="green", label="Action")
        ax3.set_title("Actions Taken")
        ax3.set_ylabel("Action")
        ax3.set_yticks([-1, 0, 1])
        ax3.set_yticklabels(["Discharge", "Idle", "Charge"])
        ax3.legend()
        ax3.minorticks_on()
        ax3.grid(visible=True, which="both")

        plt.tight_layout()
        plt.show()
