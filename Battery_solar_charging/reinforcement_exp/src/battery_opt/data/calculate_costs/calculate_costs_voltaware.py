"""Cost calculation utilities: merge consumption, tariff, and solar data.

Builds the combined dataset used by the battery simulation environment:
  hourly consumption + tariff rate → baseline cost
  + solar generation → cost with solar but no battery
  → saved to data/combined/ for use by the Gymnasium env
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_cost_dataframe(
    tariff_csv: Path | str,
    house_agg_csv: Path | str,
) -> pd.DataFrame:
    """Merge hourly consumption with tariff rates and compute baseline cost.

    Args:
        tariff_csv: Path to the hourly tariff CSV produced by
            :func:`~battery_opt.data.load_transform.process_and_save_aggregated`.
        house_agg_csv: Path to the hourly aggregated consumption CSV.

    Returns:
        DataFrame containing all consumption columns, the matched tariff
        ``rate``, and ``cost_no_solar`` (cost without any solar or battery).
    """
    tariff_df = pd.read_csv(tariff_csv)
    tariff_df["time"] = pd.to_datetime(tariff_df["time"])

    house_df = pd.read_csv(house_agg_csv)
    house_df["timestamp_min_artificial"] = pd.to_datetime(
        house_df["timestamp_min_artificial"]
    )

    merged = pd.merge(
        house_df,
        tariff_df,
        left_on="timestamp_min_artificial",
        right_on="time",
        how="left",
    )
    merged["cost_no_solar"] = merged["Total_consumption"] * merged["rate"]

    keep_cols = house_df.columns.tolist() + ["rate", "cost_no_solar"]
    return merged[keep_cols]


def merge_solar_generation(
    cost_df: pd.DataFrame,
    solar_csvs: list[Path | str],
) -> pd.DataFrame:
    """Append solar generation columns and compute cost with solar (no battery).

    For each solar file, derives a location name from the filename prefix
    (e.g. ``manchester_solar_power.csv`` → ``manchester``) and adds:
    - ``{location}_solar_generation_kw``
    - ``consumption_diff_no_battery_{location}``: grid demand after solar offset
    - ``cost_no_battery_{location}``: cost of remaining grid demand

    Args:
        cost_df: Output of :func:`build_cost_dataframe`.
        solar_csvs: List of paths to per-location solar power CSVs.

    Returns:
        Extended DataFrame with additional columns for each solar location.
    """
    df = cost_df.copy()

    for solar_path in solar_csvs:
        location = Path(solar_path).name.split("_")[0]
        solar_df = pd.read_csv(solar_path)
        solar_df["timestamp"] = pd.to_datetime(solar_df["timestamp"])

        solar_col = f"{location}_solar_generation_kw"
        solar_df = solar_df[["timestamp", "power_kw"]].rename(
            columns={"power_kw": solar_col}
        )

        df = pd.merge(
            df,
            solar_df,
            left_on="timestamp_min_artificial",
            right_on="timestamp",
            how="outer",
        )

        grid_demand_col = f"consumption_diff_no_battery_{location}"
        df[grid_demand_col] = (df["Total_consumption"] - df[solar_col]).clip(lower=0)
        df[f"cost_no_battery_{location}"] = df[grid_demand_col] * df["rate"]

    # Drop the solar 'timestamp' key column left over from each outer merge
    df = df.drop(columns=["timestamp"], errors="ignore")
    return df


def process_and_save_combined(
    aggregated_dir: Path | str,
    solar_dir: Path | str,
    output_dir: Path | str,
    solar_locations: list[str] | None = None,
) -> None:
    """Build and save the combined cost+solar datasets for both households.

    Args:
        aggregated_dir: Directory containing ``tariff_hourly.csv`` and
            ``house_{a,b}_agg_1h.csv``.
        solar_dir: Directory containing ``{location}_solar_power.csv`` files.
        output_dir: Destination directory for ``house_{a,b}_costs_with_solar.csv``.
        solar_locations: List of location name prefixes to include. Defaults to
            ``["manchester", "barcelona"]``.
    """
    if solar_locations is None:
        solar_locations = ["manchester", "barcelona"]

    aggregated_dir = Path(aggregated_dir)
    solar_dir = Path(solar_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tariff_csv = aggregated_dir / "tariff_hourly.csv"
    solar_csvs = [solar_dir / f"{loc}_solar_power.csv" for loc in solar_locations]

    for house in ("a", "b"):
        house_agg_csv = aggregated_dir / f"house_{house}_agg_1h.csv"
        cost_df = build_cost_dataframe(tariff_csv, house_agg_csv)
        combined_df = merge_solar_generation(cost_df, solar_csvs)
        out_path = output_dir / f"house_{house}_costs_with_solar.csv"
        combined_df.to_csv(out_path, index=False)

    print(f"Combined datasets saved to {output_dir}")


def plot_consumption_vs_solar(house_data: dict[str, pd.DataFrame]) -> None:
    """Plot total consumption against solar generation for each house and location.

    Args:
        house_data: Mapping of house name (e.g. ``"House A"``) to its combined
            DataFrame (output of :func:`merge_solar_generation`).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle("Total Consumption vs Solar Power Generated")

    for ax_row, (house_name, df) in zip(axes, house_data.items(), strict=False):
        for ax, location in zip(ax_row, ["manchester", "barcelona"], strict=False):
            ax.plot(
                df["timestamp_min_artificial"],
                df["Total_consumption"],
                label="Total Consumption",
                color="blue",
            )
            ax.plot(
                df["timestamp_min_artificial"],
                df[f"{location}_solar_generation_kw"],
                label=f"{location.capitalize()} Solar",
                color="orange",
            )
            ax.set_title(f"{house_name} – {location.capitalize()} Solar")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("kW")
            ax.legend()
            ax.tick_params(axis="x", rotation=45)
            ax.xaxis.set_major_locator(plt.MaxNLocator(6))

    plt.tight_layout()
    plt.show()
