from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq

yaml = YAML()
yaml.default_flow_style = False  # block style everywhere by default
yaml.best_width = 4096           # prevent wrapping inside flow-style lists

# Resolve the project root (reinforcement_exp/) from this file's location.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _flow_seq(lst: list) -> CommentedSeq:
    """Wrap a list in a ruamel CommentedSeq rendered as a single inline line."""
    seq = CommentedSeq(lst)
    seq.fa.set_flow_style()
    return seq


def load_config(config_path: Path | str | None = None):
    """Load the YAML config, defaulting to configs/default.yaml."""
    path = Path(config_path) if config_path else _PROJECT_ROOT / "configs" / "default.yaml"
    with path.open() as f:
        return yaml.load(f)


def make_episode_cust_id_lists(
    costs_df: pd.DataFrame,
    train_val_test_pct: list[float] | None = None,
    train_val_test_nums: list[int] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Split unique customer IDs into train / validation / test sets.

    Provide either ``train_val_test_pct`` (fractions summing to ≤1) or
    ``train_val_test_nums`` (absolute counts), not both.

    Args:
        costs_df: DataFrame containing a ``customer_id`` column.
        train_val_test_pct: Fractional split, e.g. ``[0.7, 0.15, 0.15]``.
        train_val_test_nums: Absolute split, e.g. ``[210, 45, 44]``.

    Returns:
        Three lists of customer IDs: (train_ids, val_ids, test_ids).
    """
    assert not (train_val_test_pct and train_val_test_nums), (
        "Provide either percentages or absolute numbers, not both."
    )

    households = np.array(costs_df["customer_id"].unique())
    rng = np.random.default_rng(42)
    rng.shuffle(households)
    n = len(households)

    if train_val_test_pct:
        train_end = int(train_val_test_pct[0] * n)
        val_end = train_end + int(train_val_test_pct[1] * n)
    else:
        train_end = train_val_test_nums[0]
        val_end = train_end + train_val_test_nums[1]

    return (
        households[:train_end].tolist(),
        households[train_end:val_end].tolist(),
        households[val_end:].tolist(),
    )


def make_episodes(
    costs_df: pd.DataFrame,
    train_ids: list[int],
    val_ids: list[int],
    test_ids: list[int] | None = None,
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    """Split the augrid costs DataFrame into per-customer episode DataFrames.

    Precomputes an integer ``hour`` column on each episode so the env's
    observation builder avoids repeated timestamp parsing in the step loop.

    Args:
        costs_df: Full augrid costs DataFrame with a ``customer_id`` column.
        train_ids: Customer IDs for the training split.
        val_ids: Customer IDs for the validation split.
        test_ids: Customer IDs for the test split (default: empty).

    Returns:
        Three dicts mapping ``customer_id → episode DataFrame``
        (train_episodes, val_episodes, test_episodes).
    """
    costs_df["timestamp"] = pd.to_datetime(costs_df["timestamp"])

    costs_df['hour'] = costs_df['timestamp'].dt.hour
    costs_df['day_of_week'] = costs_df['timestamp'].dt.dayofweek
    costs_df['month'] = costs_df['timestamp'].dt.month

    costs_df['hour_sin'] = np.sin(2 * np.pi * costs_df['hour'] / 24)
    costs_df['hour_cos'] = np.cos(2 * np.pi * costs_df['hour'] / 24)
    costs_df['day_of_week_sin'] = np.sin(2 * np.pi * costs_df['day_of_week'] / 7)
    costs_df['day_of_week_cos'] = np.cos(2 * np.pi * costs_df['day_of_week'] / 7)
    costs_df['month_sin'] = np.sin(2 * np.pi * costs_df['month'] / 12)
    costs_df['month_cos'] = np.cos(2 * np.pi * costs_df['month'] / 12)

    costs_df.drop(columns=['hour', 'day_of_week', 'month'], inplace=True)
    

    def _build(ids: list[int]) -> dict[int, pd.DataFrame]:
        """Split each customer into yearly episodes keyed by customer_id * 10 + year_idx."""
        episodes = {}
        for cid in ids:
            cid_df = costs_df[costs_df["customer_id"] == cid].copy()
            # Group into July-to-June annual chunks (matches the data's start date)
            year_label = (
                cid_df["timestamp"].dt.year
                - (cid_df["timestamp"].dt.month < 7).astype(int)
            )
            for year_idx, (_, year_df) in enumerate(cid_df.groupby(year_label)):
                ep_key = cid * 10 + year_idx
                episodes[ep_key] = year_df.reset_index(drop=True)
        return episodes

    return _build(train_ids), _build(val_ids), _build(test_ids or [])


if __name__ == "__main__":
    config = load_config()

    costs_csv = _PROJECT_ROOT / config["data"]["combined_dir"] / "augrid_costs.csv"
    costs_df = pd.read_csv(costs_csv)

    train_ids, val_ids, test_ids = make_episode_cust_id_lists(
        costs_df, train_val_test_pct=[0.85, 0.12, 0.03]
    )

    print(f"Train IDs:      {len(train_ids)}")
    print(f"Validation IDs: {len(val_ids)}")
    print(f"Test IDs:       {len(test_ids)}")

    config['data_splits'] = {
        'train_ids': _flow_seq(train_ids),
        'val_ids':   _flow_seq(val_ids),
        'test_ids':  _flow_seq(test_ids),
    }

    output_config_path = _PROJECT_ROOT / "configs" / "default.yaml"
    with output_config_path.open("w") as f:
        yaml.dump(config, f)

    print(f"Splits generated, Updated config saved to {output_config_path}")



