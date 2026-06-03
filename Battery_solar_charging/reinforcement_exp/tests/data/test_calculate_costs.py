"""Tests for battery_opt.data.calculate_costs."""

import pandas as pd
import pytest

from battery_opt.data.calculate_costs import build_cost_dataframe, merge_solar_generation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tariff_csv(tmp_path: pytest.TempPathFactory) -> str:
    """Write a minimal hourly tariff CSV and return its path."""
    df = pd.DataFrame(
        {
            "time": pd.date_range("2025-07-01", periods=24, freq="1h"),
            "rate": [0.243] * 12 + [0.498] * 12,
        }
    )
    path = tmp_path / "tariff_hourly.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture()
def house_agg_csv(tmp_path: pytest.TempPathFactory) -> str:
    """Write a minimal 24-row hourly consumption CSV and return its path."""
    df = pd.DataFrame(
        {
            "timestamp_min_artificial": pd.date_range("2025-07-01", periods=24, freq="1h"),
            "Total_consumption": [0.5] * 24,
        }
    )
    path = tmp_path / "house_a_agg_1h.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture()
def solar_csv(tmp_path: pytest.TempPathFactory) -> str:
    """Write a solar power CSV for 'manchester' and return its path."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-07-01", periods=24, freq="1h"),
            "power_kw": [0.2] * 24,
        }
    )
    path = tmp_path / "manchester_solar_power.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture()
def cost_df(tariff_csv: str, house_agg_csv: str) -> pd.DataFrame:
    return build_cost_dataframe(tariff_csv, house_agg_csv)


# ---------------------------------------------------------------------------
# build_cost_dataframe
# ---------------------------------------------------------------------------


class TestBuildCostDataframe:
    def test_has_required_columns(self, cost_df: pd.DataFrame) -> None:
        assert "rate" in cost_df.columns
        assert "cost_no_solar" in cost_df.columns
        assert "Total_consumption" in cost_df.columns

    def test_cost_equals_consumption_times_rate(self, cost_df: pd.DataFrame) -> None:
        expected = cost_df["Total_consumption"] * cost_df["rate"]
        pd.testing.assert_series_equal(cost_df["cost_no_solar"], expected, check_names=False)

    def test_row_count_matches_consumption(
        self, cost_df: pd.DataFrame, house_agg_csv: str
    ) -> None:
        original = pd.read_csv(house_agg_csv)
        assert len(cost_df) == len(original)

    def test_correct_rate_applied_by_hour(self, cost_df: pd.DataFrame) -> None:
        # First 12 hours → off-peak 0.243
        assert (cost_df["rate"].iloc[:12] == 0.243).all()
        # Last 12 hours → peak 0.498
        assert (cost_df["rate"].iloc[12:] == 0.498).all()


# ---------------------------------------------------------------------------
# merge_solar_generation
# ---------------------------------------------------------------------------


class TestMergeSolarGeneration:
    def test_adds_solar_generation_column(
        self, cost_df: pd.DataFrame, solar_csv: str
    ) -> None:
        result = merge_solar_generation(cost_df, [solar_csv])
        assert "manchester_solar_generation_kw" in result.columns

    def test_adds_grid_demand_column(
        self, cost_df: pd.DataFrame, solar_csv: str
    ) -> None:
        result = merge_solar_generation(cost_df, [solar_csv])
        assert "consumption_diff_no_battery_manchester" in result.columns

    def test_adds_cost_no_battery_column(
        self, cost_df: pd.DataFrame, solar_csv: str
    ) -> None:
        result = merge_solar_generation(cost_df, [solar_csv])
        assert "cost_no_battery_manchester" in result.columns

    def test_grid_demand_non_negative(
        self, cost_df: pd.DataFrame, solar_csv: str
    ) -> None:
        result = merge_solar_generation(cost_df, [solar_csv])
        assert (result["consumption_diff_no_battery_manchester"] >= 0).all()

    def test_solar_reduces_grid_demand(
        self, cost_df: pd.DataFrame, solar_csv: str
    ) -> None:
        result = merge_solar_generation(cost_df, [solar_csv])
        # Solar 0.2 kW, consumption 0.5 kW → grid demand should be 0.3 kW
        import numpy as np
        np.testing.assert_allclose(
            result["consumption_diff_no_battery_manchester"].values, 0.3
        )

    def test_no_duplicate_timestamp_columns(
        self, cost_df: pd.DataFrame, solar_csv: str
    ) -> None:
        result = merge_solar_generation(cost_df, [solar_csv])
        unexpected = [c for c in result.columns if c.startswith("timestamp") and c != "timestamp_min_artificial"]
        assert unexpected == [], f"Unexpected timestamp columns: {unexpected}"

    def test_original_cost_df_not_mutated(
        self, cost_df: pd.DataFrame, solar_csv: str
    ) -> None:
        original_cols = set(cost_df.columns)
        merge_solar_generation(cost_df, [solar_csv])
        assert set(cost_df.columns) == original_cols