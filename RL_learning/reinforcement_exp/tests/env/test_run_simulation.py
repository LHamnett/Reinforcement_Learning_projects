"""Tests for battery_opt.env.run_simulation (train/evaluate loops)."""

import numpy as np
import pandas as pd
import pytest

from battery_opt.agents.RL_agents.q_learning import QLearningAgent
from battery_opt.env.battery_env import BatteryEnv, ObsNormaliser
from battery_opt.env.run_simulation import evaluate_agent, train_agent


# ---------------------------------------------------------------------------
# Fixtures (shared with test_battery_env via local helpers)
# ---------------------------------------------------------------------------

N_HOURS = 24


def _make_episode(n_hours: int = N_HOURS, seed: int = 0) -> pd.DataFrame:
    ts  = pd.date_range("2010-07-01", periods=n_hours, freq="1h")
    rng = np.random.default_rng(seed)
    cons = rng.uniform(0.5, 1.5, n_hours)
    sol  = rng.uniform(0.0, 0.8, n_hours)
    tariff = np.where(ts.hour < 6, 0.20, np.where(ts.hour < 16, 0.28, 0.45))
    return pd.DataFrame({
        "timestamp":    ts,
        "total_consumption": cons,
        "gg_kwh":       sol,
        "import_tariff": tariff,
        "import_cost_solar_used_immediate": np.maximum(cons - sol, 0) * tariff,
        "hour_sin":     np.sin(2 * np.pi * ts.hour / 24),
        "hour_cos":     np.cos(2 * np.pi * ts.hour / 24),
        "day_of_week_sin": np.sin(2 * np.pi * ts.dayofweek / 7),
        "day_of_week_cos": np.cos(2 * np.pi * ts.dayofweek / 7),
        "month_sin":    np.sin(2 * np.pi * ts.month / 12),
        "month_cos":    np.cos(2 * np.pi * ts.month / 12),
    }).reset_index(drop=True)


@pytest.fixture()
def train_episodes() -> dict[int, pd.DataFrame]:
    return {cid: _make_episode(seed=cid) for cid in [1, 2, 3]}


@pytest.fixture()
def val_episodes() -> dict[int, pd.DataFrame]:
    return {cid: _make_episode(seed=cid + 10) for cid in [4, 5]}


@pytest.fixture()
def normaliser(train_episodes: dict) -> ObsNormaliser:
    return ObsNormaliser.from_episodes(train_episodes)


@pytest.fixture()
def train_env(train_episodes: dict, normaliser: ObsNormaliser) -> BatteryEnv:
    return BatteryEnv(train_episodes, normaliser, battery_capacity_kwh=10.0)


@pytest.fixture()
def val_env(val_episodes: dict, normaliser: ObsNormaliser) -> BatteryEnv:
    return BatteryEnv(val_episodes, normaliser, battery_capacity_kwh=10.0)


@pytest.fixture()
def agent() -> QLearningAgent:
    return QLearningAgent(
        battery_capacity_kwh=10.0,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_episodes=5,
    )


# ---------------------------------------------------------------------------
# train_agent
# ---------------------------------------------------------------------------


class TestTrainQAgent:
    def test_returns_list_of_epoch_costs(
        self, train_env: BatteryEnv, agent: QLearningAgent
    ) -> None:
        costs = train_agent(train_env, agent, n_epochs=2, max_episodes=1)
        assert isinstance(costs, list) and len(costs) == 2

    def test_epoch_costs_are_positive(
        self, train_env: BatteryEnv, agent: QLearningAgent
    ) -> None:
        costs = train_agent(train_env, agent, n_epochs=2, max_episodes=1)
        assert all(c > 0 for c in costs)

    def test_q_table_populated_after_training(
        self, train_env: BatteryEnv, agent: QLearningAgent
    ) -> None:
        train_agent(train_env, agent, n_epochs=1, max_episodes=1)
        assert len(agent.q_table) > 0

    def test_epsilon_decreases_after_training(
        self, train_env: BatteryEnv, agent: QLearningAgent
    ) -> None:
        before = agent.epsilon
        train_agent(train_env, agent, n_epochs=2, max_episodes=1)
        assert agent.epsilon < before

    def test_epsilon_clamps_at_minimum(
        self, train_env: BatteryEnv, agent: QLearningAgent
    ) -> None:
        train_agent(train_env, agent, n_epochs=20, max_episodes=1)
        assert agent.epsilon >= agent.epsilon_end


# ---------------------------------------------------------------------------
# evaluate_agent
# ---------------------------------------------------------------------------


class TestEvaluateQAgent:
    def test_returns_all_customer_ids(
        self, val_env: BatteryEnv, agent: QLearningAgent
    ) -> None:
        results = evaluate_agent(val_env, agent, max_episodes=1)
        assert set(results.keys()).issubset(set(val_env.episodes.keys()))

    def test_result_has_required_keys(
        self, val_env: BatteryEnv, agent: QLearningAgent
    ) -> None:
        results = evaluate_agent(val_env, agent, max_episodes=1)
        for r in results.values():
            for key in ("total_cost", "baseline_cost", "saving", "saving_pct"):
                assert key in r

    def test_total_cost_positive(
        self, val_env: BatteryEnv, agent: QLearningAgent
    ) -> None:
        results = evaluate_agent(val_env, agent, max_episodes=1)
        assert all(r["total_cost"] > 0 for r in results.values())

    def test_saving_equals_baseline_minus_cost(
        self, val_env: BatteryEnv, agent: QLearningAgent
    ) -> None:
        results = evaluate_agent(val_env, agent, max_episodes=1)
        for r in results.values():
            assert r["saving"] == pytest.approx(r["baseline_cost"] - r["total_cost"])

    def test_saving_pct_consistent_with_saving(
        self, val_env: BatteryEnv, agent: QLearningAgent
    ) -> None:
        results = evaluate_agent(val_env, agent, max_episodes=1)
        for r in results.values():
            expected_pct = 100 * r["saving"] / r["baseline_cost"]
            assert r["saving_pct"] == pytest.approx(expected_pct)

    def test_greedy_eval_uses_no_exploration(
        self, train_env: BatteryEnv, val_env: BatteryEnv, agent: QLearningAgent
    ) -> None:
        """Two identical evaluations should return identical costs."""
        train_agent(train_env, agent, n_epochs=1, max_episodes=1)
        r1 = evaluate_agent(val_env, agent, max_episodes=1)
        r2 = evaluate_agent(val_env, agent, max_episodes=1)
        for cid in r1:
            assert r1[cid]["total_cost"] == pytest.approx(r2[cid]["total_cost"])
