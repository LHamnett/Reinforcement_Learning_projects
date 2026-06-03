"""Tests for battery_opt.data.load_transform."""

import numpy as np
import pandas as pd
import pytest

from battery_opt.data.load_transform import (
    aggregate_consumption,
    calculate_tilted_solar_power,
    convert_tariff_hourly,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_tariff_df() -> pd.DataFrame:
    """Minimal two-interval tariff table spanning one day."""
    return pd.DataFrame(
        {
            "start_time": ["2025-07-01 00:00:00", "2025-07-01 12:00:00"],
            "end_time": ["2025-07-01 12:00:00", "2025-07-02 00:00:00"],
            "rate": [0.243, 0.498],
        }
    )


@pytest.fixture()
def sample_consumption_df() -> pd.DataFrame:
    """30-min resolution consumption for 2 hours (4 rows)."""
    return pd.DataFrame(
        {
            "timestamp_min_artificial": [
                "2025-07-01 00:00:00",
                "2025-07-01 00:30:00",
                "2025-07-01 01:00:00",
                "2025-07-01 01:30:00",
            ],
            "appliance_a": [0.1, 0.2, 0.3, 0.4],
            "Total_consumption": [0.1, 0.2, 0.3, 0.4],
        }
    )


# ---------------------------------------------------------------------------
# convert_tariff_hourly
# ---------------------------------------------------------------------------


class TestConvertTariffHourly:
    def test_output_has_time_and_rate_columns(self, sample_tariff_df: pd.DataFrame) -> None:
        result = convert_tariff_hourly(sample_tariff_df)
        assert "time" in result.columns
        assert "rate" in result.columns

    def test_one_row_per_hour(self, sample_tariff_df: pd.DataFrame) -> None:
        result = convert_tariff_hourly(sample_tariff_df)
        assert len(result) == 24  # 24 hours in one day

    def test_correct_rate_assigned(self, sample_tariff_df: pd.DataFrame) -> None:
        result = convert_tariff_hourly(sample_tariff_df)
        # Hours 0–11 → off-peak rate 0.243
        assert (result.iloc[:12]["rate"] == 0.243).all()
        # Hours 12–23 → peak rate 0.498
        assert (result.iloc[12:]["rate"] == 0.498).all()

    def test_original_df_not_mutated(self, sample_tariff_df: pd.DataFrame) -> None:
        original_dtypes = sample_tariff_df.dtypes.to_dict()
        convert_tariff_hourly(sample_tariff_df)
        assert sample_tariff_df.dtypes.to_dict() == original_dtypes


# ---------------------------------------------------------------------------
# aggregate_consumption
# ---------------------------------------------------------------------------


class TestAggregateConsumption:
    def test_output_length(self, sample_consumption_df: pd.DataFrame) -> None:
        result = aggregate_consumption(
            sample_consumption_df,
            time_col="timestamp_min_artificial",
            cols_to_aggregate=["appliance_a", "Total_consumption"],
            freq="1h",
        )
        assert len(result) == 2  # 4 half-hourly rows → 2 hourly buckets

    def test_values_are_summed(self, sample_consumption_df: pd.DataFrame) -> None:
        result = aggregate_consumption(
            sample_consumption_df,
            time_col="timestamp_min_artificial",
            cols_to_aggregate=["appliance_a"],
            freq="1h",
        )
        # First hour: 0.1 + 0.2 = 0.3
        assert result["appliance_a"].iloc[0] == pytest.approx(0.3)
        # Second hour: 0.3 + 0.4 = 0.7
        assert result["appliance_a"].iloc[1] == pytest.approx(0.7)

    def test_index_is_datetime(self, sample_consumption_df: pd.DataFrame) -> None:
        result = aggregate_consumption(
            sample_consumption_df,
            time_col="timestamp_min_artificial",
            cols_to_aggregate=["Total_consumption"],
            freq="1h",
        )
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_original_df_not_mutated(self, sample_consumption_df: pd.DataFrame) -> None:
        original = sample_consumption_df.copy()
        aggregate_consumption(
            sample_consumption_df,
            time_col="timestamp_min_artificial",
            cols_to_aggregate=["Total_consumption"],
            freq="1h",
        )
        pd.testing.assert_frame_equal(sample_consumption_df, original)


# ---------------------------------------------------------------------------
# calculate_tilted_solar_power
# ---------------------------------------------------------------------------


class TestCalculateTiltedSolarPower:
    def test_zero_irradiance_gives_zero_output(self) -> None:
        result = calculate_tilted_solar_power(
            ghi=np.zeros(5),
            dhi=np.zeros(5),
            sza=np.full(5, 80.0),
            latitude=53.0,
        )
        assert (result == 0.0).all()

    def test_output_capped_at_system_max(self) -> None:
        system_size = 3.0
        perf_ratio = 0.7
        result = calculate_tilted_solar_power(
            ghi=np.full(10, 2000.0),  # unrealistically high
            dhi=np.full(10, 500.0),
            sza=np.zeros(10),
            latitude=53.0,
            system_size_kwp=system_size,
            performance_ratio=perf_ratio,
        )
        assert (result <= system_size * perf_ratio + 1e-9).all()

    def test_output_non_negative(self) -> None:
        rng = np.random.default_rng(0)
        ghi = rng.uniform(0, 800, 100)
        dhi = rng.uniform(0, 200, 100)
        sza = rng.uniform(0, 90, 100)
        result = calculate_tilted_solar_power(ghi=ghi, dhi=dhi, sza=sza, latitude=53.0)
        assert (result >= 0).all()

    def test_higher_irradiance_gives_higher_output(self) -> None:
        low = calculate_tilted_solar_power(
            ghi=np.array([200.0]), dhi=np.array([50.0]), sza=np.array([45.0]), latitude=53.0
        )
        high = calculate_tilted_solar_power(
            ghi=np.array([600.0]), dhi=np.array([150.0]), sza=np.array([45.0]), latitude=53.0
        )
        assert high[0] > low[0]