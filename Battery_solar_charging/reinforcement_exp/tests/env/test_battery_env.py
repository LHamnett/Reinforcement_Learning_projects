"""Tests for battery_opt.env.battery_env (ObsNormaliser and BatteryEnv)."""

import numpy as np
import pandas as pd
import pytest

from battery_opt.env.battery_env import BatteryEnv, ObsNormaliser, _CYCLICAL_COLS


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

N_HOURS = 24


def _make_episode(
    n_hours: int = N_HOURS,
    consumption: float = 1.0,
    solar: float = 0.5,
    tariff: float = 0.28,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic episode DataFrame with all required columns."""
    ts  = pd.date_range("2010-07-01", periods=n_hours, freq="1h")
    rng = np.random.default_rng(seed)
    cons = rng.uniform(0.5, 1.5, n_hours) if consumption is None else np.full(n_hours, consumption)
    sol  = rng.uniform(0.0, 1.0, n_hours) if solar is None else np.full(n_hours, solar)
    hrs  = ts.hour
    dow  = ts.dayofweek
    mon  = ts.month
    df = pd.DataFrame({
        "timestamp":   ts,
        "total_consumption": cons,
        "gg_kwh":      sol,
        "import_tariff": tariff,
        "import_cost_solar_used_immediate": np.maximum(cons - sol, 0) * tariff,
        "hour_sin":    np.sin(2 * np.pi * hrs / 24),
        "hour_cos":    np.cos(2 * np.pi * hrs / 24),
        "day_of_week_sin": np.sin(2 * np.pi * dow / 7),
        "day_of_week_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin":   np.sin(2 * np.pi * mon / 12),
        "month_cos":   np.cos(2 * np.pi * mon / 12),
    })
    return df.reset_index(drop=True)


@pytest.fixture()
def sample_episodes() -> dict[int, pd.DataFrame]:
    return {1: _make_episode(seed=1), 2: _make_episode(seed=2), 3: _make_episode(seed=3)}


@pytest.fixture()
def normaliser(sample_episodes: dict) -> ObsNormaliser:
    return ObsNormaliser.from_episodes(sample_episodes)


@pytest.fixture()
def env(sample_episodes: dict, normaliser: ObsNormaliser) -> BatteryEnv:
    return BatteryEnv(episodes=sample_episodes, normaliser=normaliser,
                      battery_capacity_kwh=10.0)


# ---------------------------------------------------------------------------
# ObsNormaliser
# ---------------------------------------------------------------------------


class TestObsNormaliser:
    def test_mean_shape(self, normaliser: ObsNormaliser) -> None:
        assert normaliser.mean.shape == (5,)

    def test_std_shape(self, normaliser: ObsNormaliser) -> None:
        assert normaliser.std.shape == (5,)

    def test_std_positive(self, normaliser: ObsNormaliser) -> None:
        assert (normaliser.std >= 1e-8).all()

    def test_normalise_output_shape(self, normaliser: ObsNormaliser) -> None:
        data = np.ones(5, dtype=np.float32)
        cycl = np.zeros(6, dtype=np.float32)
        out = normaliser.normalise(data, cycl, battery_soc=0.0, battery_capacity=10.0)
        assert out.shape == (12,)

    def test_normalise_dtype_float32(self, normaliser: ObsNormaliser) -> None:
        out = normaliser.normalise(
            np.ones(5, np.float32), np.zeros(6, np.float32), 0.0, 10.0
        )
        assert out.dtype == np.float32

    def test_cyclical_features_pass_through_unchanged(self, normaliser: ObsNormaliser) -> None:
        cycl = np.array([0.5, -0.3, 0.1, 0.9, -0.7, 0.2], dtype=np.float32)
        out = normaliser.normalise(np.zeros(5, np.float32), cycl, 0.0, 10.0)
        np.testing.assert_array_almost_equal(out[5:11], cycl)

    def test_soc_zero_gives_last_element_zero(self, normaliser: ObsNormaliser) -> None:
        out = normaliser.normalise(np.zeros(5, np.float32), np.zeros(6, np.float32),
                                   battery_soc=0.0, battery_capacity=10.0)
        assert out[11] == pytest.approx(0.0)

    def test_soc_full_gives_last_element_one(self, normaliser: ObsNormaliser) -> None:
        out = normaliser.normalise(np.zeros(5, np.float32), np.zeros(6, np.float32),
                                   battery_soc=10.0, battery_capacity=10.0)
        assert out[11] == pytest.approx(1.0)

    def test_fitted_on_training_data_only(self) -> None:
        """Normaliser fitted on high-consumption data has different stats."""
        high = {1: _make_episode(consumption=5.0), 2: _make_episode(consumption=5.0)}
        low  = {1: _make_episode(consumption=0.5), 2: _make_episode(consumption=0.5)}
        norm_high = ObsNormaliser.from_episodes(high)
        norm_low  = ObsNormaliser.from_episodes(low)
        assert not np.allclose(norm_high.mean, norm_low.mean)


# ---------------------------------------------------------------------------
# BatteryEnv
# ---------------------------------------------------------------------------


class TestBatteryEnvReset:
    def test_obs_shape(self, env: BatteryEnv) -> None:
        obs, _ = env.reset()
        assert obs.shape == (12,)

    def test_obs_dtype(self, env: BatteryEnv) -> None:
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_info_contains_required_keys(self, env: BatteryEnv) -> None:
        _, info = env.reset()
        for key in ("customer_id", "battery_soc_kwh", "surplus_kwh", "import_tariff"):
            assert key in info, f"missing key: {key}"

    def test_battery_soc_zero_at_reset(self, env: BatteryEnv) -> None:
        _, info = env.reset()
        assert info["battery_soc_kwh"] == pytest.approx(0.0)

    def test_customer_id_respected(self, env: BatteryEnv) -> None:
        _, info = env.reset(options={"customer_id": 2})
        assert info["customer_id"] == 2

    def test_random_customer_within_episodes(self, env: BatteryEnv) -> None:
        for _ in range(10):
            _, info = env.reset()
            assert info["customer_id"] in env.episodes


class TestBatteryEnvStep:
    def test_idle_leaves_soc_unchanged(self, env: BatteryEnv) -> None:
        env.reset(options={"customer_id": 1})
        _, _, _, _, info = env.step(1)  # action 1 = idle
        assert info["battery_soc_kwh"] == pytest.approx(0.0)

    def test_charge_increases_soc(self) -> None:
        """Episode with guaranteed surplus so charging actually stores energy."""
        eps = {1: _make_episode(consumption=0.1, solar=2.0)}
        norm = ObsNormaliser.from_episodes(eps)
        e = BatteryEnv(eps, norm, battery_capacity_kwh=10.0)
        e.reset(options={"customer_id": 1})
        _, _, _, _, info = e.step(2)  # action 2 = charge
        assert info["battery_soc_kwh"] > 0.0

    def test_discharge_reduces_soc(self) -> None:
        """Step 1: high surplus → charge. Step 2: high demand, no solar → discharge."""
        ts = pd.date_range("2010-07-01", periods=2, freq="1h")
        zeros6 = np.zeros(2)
        df = pd.DataFrame({
            "timestamp":   ts,
            "total_consumption": [0.1, 2.0],   # step 2 has high demand
            "gg_kwh":            [5.0, 0.0],   # step 1 has surplus, step 2 has none
            "import_tariff":     [0.28, 0.28],
            "import_cost_solar_used_immediate": [0.0, 2.0 * 0.28],
            "hour_sin": zeros6, "hour_cos": np.ones(2),
            "day_of_week_sin": zeros6, "day_of_week_cos": np.ones(2),
            "month_sin": zeros6, "month_cos": np.ones(2),
        })
        eps = {1: df}
        norm = ObsNormaliser.from_episodes(eps)
        e = BatteryEnv(eps, norm, battery_capacity_kwh=10.0)
        e.reset(options={"customer_id": 1})
        e.step(2)                        # charge on surplus
        soc_after_charge = e._battery_soc
        e.step(0)                        # discharge into demand
        assert e._battery_soc < soc_after_charge

    def test_cannot_charge_beyond_capacity(self) -> None:
        cap = 2.0
        eps = {1: _make_episode(consumption=0.0, solar=100.0, n_hours=48)}
        norm = ObsNormaliser.from_episodes(eps)
        e = BatteryEnv(eps, norm, battery_capacity_kwh=cap)
        e.reset(options={"customer_id": 1})
        for _ in range(20):
            e.step(2)
        assert e._battery_soc <= cap + 1e-6

    def test_cannot_discharge_below_zero(self) -> None:
        eps = {1: _make_episode(consumption=1.0, solar=0.0, n_hours=48)}
        norm = ObsNormaliser.from_episodes(eps)
        e = BatteryEnv(eps, norm, battery_capacity_kwh=10.0)
        e.reset(options={"customer_id": 1})
        for _ in range(20):
            e.step(0)  # keep discharging
        assert e._battery_soc >= -1e-6

    def test_step_returns_five_values(self, env: BatteryEnv) -> None:
        env.reset(options={"customer_id": 1})
        result = env.step(1)
        assert len(result) == 5

    def test_terminated_always_false(self, env: BatteryEnv) -> None:
        env.reset(options={"customer_id": 1})
        for _ in range(N_HOURS):
            _, _, terminated, _, _ = env.step(1)
            assert not terminated

    def test_truncated_after_all_steps(self, env: BatteryEnv) -> None:
        env.reset(options={"customer_id": 1})
        truncated = False
        for _ in range(N_HOURS):
            _, _, _, truncated, _ = env.step(1)
        assert truncated

    def test_obs_is_zeros_when_truncated(self, env: BatteryEnv) -> None:
        env.reset(options={"customer_id": 1})
        for _ in range(N_HOURS):
            obs, _, _, truncated, _ = env.step(1)
        assert truncated
        np.testing.assert_array_equal(obs, np.zeros(12, dtype=np.float32))

    def test_reward_is_negative_cost(self, env: BatteryEnv) -> None:
        env.reset(options={"customer_id": 1})
        _, reward, _, _, info = env.step(1)
        assert reward == pytest.approx(-info["cost"])

    def test_step_count_increments(self, env: BatteryEnv) -> None:
        env.reset(options={"customer_id": 1})
        env.step(1)
        assert env._step_idx == 1


class TestBatteryEnvObsSpaceBounds:
    def test_cyclical_obs_within_bounds(self, env: BatteryEnv) -> None:
        obs, _ = env.reset(options={"customer_id": 1})
        # indices 5-10 are cyclical — must be in [-1, 1]
        assert (obs[5:11] >= -1.0 - 1e-6).all()
        assert (obs[5:11] <=  1.0 + 1e-6).all()

    def test_soc_obs_within_bounds_at_reset(self, env: BatteryEnv) -> None:
        obs, _ = env.reset(options={"customer_id": 1})
        # index 11 is SoC/capacity — starts at 0
        assert obs[11] == pytest.approx(0.0)

    def test_observation_space_shape(self, env: BatteryEnv) -> None:
        assert env.observation_space.shape == (12,)

    def test_action_space_size(self, env: BatteryEnv) -> None:
        assert env.action_space.n == 3
