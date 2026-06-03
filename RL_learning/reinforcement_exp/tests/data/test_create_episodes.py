"""Tests for battery_opt.data.create_episodes.make_episodes."""

import numpy as np
import pandas as pd
import pytest

from battery_opt.data.create_episodes import make_episodes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def costs_df() -> pd.DataFrame:
    """Minimal costs DataFrame with 3 customers × 48 hours each (single annual chunk)."""
    rows = []
    for cid in [1, 2, 3]:
        ts = pd.date_range("2010-07-01", periods=48, freq="1h")
        rows.append(pd.DataFrame({
            "timestamp":        ts,
            "customer_id":      cid,
            "total_consumption": np.random.default_rng(cid).uniform(0.3, 1.5, 48),
            "gg_kwh":           np.random.default_rng(cid + 10).uniform(0.0, 1.0, 48),
            "import_tariff":    0.28,
            "import_cost_solar_used_immediate": 0.1,
        }))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture()
def multi_year_costs_df() -> pd.DataFrame:
    """3 customers × 3 annual chunks (Jul-Jun years) to test annual splitting."""
    rows = []
    for cid in [1, 2, 3]:
        # Span 3 full July-to-June years (simplified as 3 × 24h here)
        for year_offset in range(3):
            ts = pd.date_range(f"{2010 + year_offset}-07-01", periods=24, freq="1h")
            rows.append(pd.DataFrame({
                "timestamp":        ts,
                "customer_id":      cid,
                "total_consumption": 1.0,
                "gg_kwh":           0.3,
                "import_tariff":    0.28,
                "import_cost_solar_used_immediate": 0.1,
            }))
    return pd.concat(rows, ignore_index=True)


# Helper: recover customer_id from episode key (key = cid * 10 + year_idx)
def _cid(key: int) -> int:
    return key // 10


# ---------------------------------------------------------------------------
# make_episodes
# ---------------------------------------------------------------------------


class TestMakeEpisodes:
    def test_returns_three_dicts(self, costs_df: pd.DataFrame) -> None:
        result = make_episodes(costs_df, train_ids=[1, 2], val_ids=[3])
        assert len(result) == 3

    def test_train_customer_ids_covered(self, costs_df: pd.DataFrame) -> None:
        train, val, _ = make_episodes(costs_df, [1, 2], [3])
        assert {_cid(k) for k in train.keys()} == {1, 2}
        assert {_cid(k) for k in val.keys()} == {3}

    def test_test_ids_none_returns_empty(self, costs_df: pd.DataFrame) -> None:
        _, _, test = make_episodes(costs_df, [1], [2], test_ids=None)
        assert test == {}

    def test_test_ids_populated(self, costs_df: pd.DataFrame) -> None:
        _, _, test = make_episodes(costs_df, [1], [2], test_ids=[3])
        assert {_cid(k) for k in test.keys()} == {3}

    def test_each_episode_contains_only_its_customer(self, costs_df: pd.DataFrame) -> None:
        train, _, _ = make_episodes(costs_df, [1, 2], [3])
        for key, df in train.items():
            assert (df["customer_id"] == _cid(key)).all()

    def test_episode_row_count_matches_source(self, costs_df: pd.DataFrame) -> None:
        train, _, _ = make_episodes(costs_df, [1], [2])
        # Single annual chunk: total rows for customer 1 = 48
        total = sum(len(df) for key, df in train.items() if _cid(key) == 1)
        assert total == len(costs_df[costs_df["customer_id"] == 1])

    def test_index_is_reset(self, costs_df: pd.DataFrame) -> None:
        train, _, _ = make_episodes(costs_df, [1], [2])
        for df in train.values():
            assert df.index.tolist() == list(range(len(df)))

    def test_annual_split_creates_multiple_episodes(
        self, multi_year_costs_df: pd.DataFrame
    ) -> None:
        train, _, _ = make_episodes(multi_year_costs_df, [1, 2], [3])
        # Each of customers 1 and 2 has 3 annual chunks → 6 episodes total
        assert len(train) == 6
        assert {_cid(k) for k in train.keys()} == {1, 2}

    def test_annual_split_keys_are_unique(self, multi_year_costs_df: pd.DataFrame) -> None:
        train, val, _ = make_episodes(multi_year_costs_df, [1, 2], [3])
        all_keys = list(train.keys()) + list(val.keys())
        assert len(all_keys) == len(set(all_keys))

    def test_cyclical_hour_columns_present(self, costs_df: pd.DataFrame) -> None:
        train, _, _ = make_episodes(costs_df, [1], [2])
        ep = next(iter(train.values()))
        for col in ["hour_sin", "hour_cos"]:
            assert col in ep.columns

    def test_cyclical_columns_present(self, costs_df: pd.DataFrame) -> None:
        train, _, _ = make_episodes(costs_df, [1], [2])
        ep = next(iter(train.values()))
        expected = {
            "hour_sin", "hour_cos",
            "day_of_week_sin", "day_of_week_cos",
            "month_sin", "month_cos",
        }
        assert expected.issubset(ep.columns)

    def test_cyclical_values_in_range(self, costs_df: pd.DataFrame) -> None:
        train, _, _ = make_episodes(costs_df, [1], [2])
        ep = next(iter(train.values()))
        for col in ["hour_sin", "hour_cos", "day_of_week_sin",
                    "day_of_week_cos", "month_sin", "month_cos"]:
            assert ep[col].between(-1.0, 1.0).all(), f"{col} out of [-1, 1]"
