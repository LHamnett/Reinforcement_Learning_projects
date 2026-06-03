"""Data loading and transformation utilities for the UK household pipeline.

Handles three data sources:
- Household consumption CSVs (5-min resolution, per-appliance kWh)
- Tariff table (variable-rate intervals → expanded to hourly)
- NASA POWER irradiance CSVs → estimated solar panel output
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_csv(path: Path | str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path)


def convert_tariff_hourly(tariff_df: pd.DataFrame) -> pd.DataFrame:
    """Expand an interval-based tariff table to one row per hour.

    Args:
        tariff_df: DataFrame with columns [start_time, end_time, rate].
            start_time and end_time define non-overlapping intervals covering
            the full date range.

    Returns:
        DataFrame with columns [time, rate], one row per hour from the
        earliest start_time (inclusive) to the latest end_time (exclusive).
    """
    df = tariff_df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])

    time_grid = pd.date_range(
        start=df["start_time"].min(),
        end=df["end_time"].max(),
        freq="1h",
        inclusive="left",
    )

    resampled = pd.DataFrame({"time": time_grid})
    resampled["rate"] = resampled["time"].apply(_get_tariff_rate, args=(df,))
    return resampled


def _get_tariff_rate(t: pd.Timestamp, tariff_df: pd.DataFrame) -> float | None:
    """Return the tariff rate that covers timestamp t."""
    row = tariff_df[(tariff_df.start_time <= t) & (tariff_df.end_time > t)]
    return float(row["rate"].iloc[0]) if not row.empty else None


def aggregate_consumption(
    df: pd.DataFrame,
    time_col: str,
    cols_to_aggregate: list[str],
    freq: str,
) -> pd.DataFrame:
    """Resample per-appliance consumption data to a coarser time resolution.

    Args:
        df: Input DataFrame containing consumption data.
        time_col: Name of the datetime column to use as the index.
        cols_to_aggregate: Columns to sum during resampling.
        freq: Pandas offset alias for the target resolution (e.g. ``"1h"``).

    Returns:
        DataFrame with a DatetimeIndex and one column per entry in
        ``cols_to_aggregate``, values summed over each interval.
    """
    df_copy = df.copy()
    df_copy[time_col] = pd.to_datetime(df_copy[time_col])
    df_copy = df_copy.set_index(time_col)
    return df_copy[cols_to_aggregate].resample(freq).sum()


def read_nasa_csv(path: Path | str) -> pd.DataFrame:
    """Parse a NASA POWER point-hourly CSV, skipping the metadata header.

    NASA POWER CSVs contain a variable-length header block terminated by the
    line ``-END HEADER-``. This function locates that line and reads the
    tabular data that follows.

    Args:
        path: Path to the NASA POWER CSV file.

    Returns:
        DataFrame with the raw irradiance columns as provided by NASA POWER
        (YEAR, MO, DY, HR, ALLSKY_SFC_SW_DWN, ALLSKY_SFC_SW_DIFF, SZA, …).
    """
    with open(path) as f:
        lines = f.readlines()

    start_row = next(
        i + 1 for i, line in enumerate(lines) if "-END HEADER-" in line
    )
    return pd.read_csv(path, skiprows=start_row)


def calculate_tilted_solar_power(
    ghi: pd.Series | np.ndarray,
    dhi: pd.Series | np.ndarray,
    sza: pd.Series | np.ndarray,
    latitude: float,
    system_size_kwp: float = 3.0,
    performance_ratio: float = 0.7,
) -> np.ndarray:
    """Estimate household solar output from irradiance and geometry.

    Uses an isotropic sky model with panel tilt fixed to the site latitude
    (close to the optimal fixed-tilt for a south-facing roof in the UK).

    Args:
        ghi: Global horizontal irradiance (W/m²).
        dhi: Diffuse horizontal irradiance (W/m²).
        sza: Solar zenith angle (degrees).
        latitude: Site latitude (degrees). Determines panel tilt.
        system_size_kwp: Installed capacity of the PV system (kWp).
        performance_ratio: System efficiency factor accounting for inverter
            losses, wiring, soiling, etc. Typical range 0.65–0.80.

    Returns:
        Array of estimated AC power output (kW), one value per input row.
        Capped at ``system_size_kwp * performance_ratio``.
    """
    tilt_rad = np.radians(abs(latitude))
    sza_rad = np.radians(np.asarray(sza, dtype=float))

    cos_sza = np.cos(sza_rad)
    cos_sza_safe = np.maximum(cos_sza, 0.05)  # avoid divide-by-zero at night

    dni = np.maximum((np.asarray(ghi) - np.asarray(dhi)) / cos_sza_safe, 0)
    cos_incidence = np.maximum(np.cos(sza_rad - tilt_rad), 0)

    diffuse_tilted = np.asarray(dhi) * (1 + np.cos(tilt_rad)) / 2
    poa = dni * cos_incidence + diffuse_tilted

    power_kw = (poa / 1000) * system_size_kwp * performance_ratio
    return np.minimum(power_kw, system_size_kwp * performance_ratio)


def make_power_df_from_irradiance(
    irrad_df: pd.DataFrame,
    latitude: float,
    system_size_kwp: float,
    performance_ratio: float,
) -> pd.DataFrame:
    """Add a ``power_kw`` column and a ``timestamp`` column to a NASA POWER DataFrame.

    Args:
        irrad_df: DataFrame from :func:`read_nasa_csv`.
        latitude: Site latitude used for panel tilt calculation.
        system_size_kwp: Installed PV capacity (kWp).
        performance_ratio: System efficiency factor.

    Returns:
        Copy of ``irrad_df`` with additional columns:
        - ``power_kw``: estimated AC output (kW)
        - ``timestamp``: datetime assembled from YEAR/MO/DY/HR columns
    """
    df = irrad_df.copy()
    df["power_kw"] = calculate_tilted_solar_power(
        ghi=df["ALLSKY_SFC_SW_DWN"],
        dhi=df["ALLSKY_SFC_SW_DIFF"],
        sza=df["SZA"],
        latitude=latitude,
        system_size_kwp=system_size_kwp,
        performance_ratio=performance_ratio,
    )
    df["timestamp"] = pd.to_datetime(
        df[["YEAR", "MO", "DY", "HR"]].rename(
            columns={"YEAR": "year", "MO": "month", "DY": "day", "HR": "hour"}
        )
    )
    return df


def process_and_save_solar(
    input_csv: Path | str,
    output_csv: Path | str,
    latitude: float,
    system_size_kwp: float,
    performance_ratio: float,
) -> None:
    """Read a NASA POWER CSV, compute solar output, and save to ``output_csv``.

    Args:
        input_csv: Path to the raw NASA POWER irradiance CSV.
        output_csv: Destination path for the processed output.
        latitude: Site latitude (degrees).
        system_size_kwp: Installed PV capacity (kWp).
        performance_ratio: System efficiency factor.
    """
    irrad_df = read_nasa_csv(input_csv)
    power_df = make_power_df_from_irradiance(
        irrad_df,
        latitude=latitude,
        system_size_kwp=system_size_kwp,
        performance_ratio=performance_ratio,
    )
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    power_df.to_csv(output_path, index=False)


def process_and_save_aggregated(
    raw_dir: Path | str,
    output_dir: Path | str,
) -> None:
    """Aggregate raw 5-min consumption and tariff data to hourly and save.

    Reads from ``raw_dir``:
    - ``household_a_consumption.csv``
    - ``household_b_consumption.csv``
    - ``tariff_table.csv``

    Writes to ``output_dir``:
    - ``house_a_agg_1h.csv``
    - ``house_b_agg_1h.csv``
    - ``tariff_hourly.csv``

    Args:
        raw_dir: Directory containing the raw source CSVs.
        output_dir: Directory to write processed outputs (created if absent).
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tariff_df = load_csv(raw_dir / "tariff_table.csv")
    tariff_hourly = convert_tariff_hourly(tariff_df)
    tariff_hourly.to_csv(output_dir / "tariff_hourly.csv", index=False)

    for house in ("a", "b"):
        cons_df = load_csv(raw_dir / f"household_{house}_consumption.csv")
        agg_cols = [c for c in cons_df.columns if c != "timestamp_min_artificial"]
        agg_df = aggregate_consumption(
            cons_df,
            time_col="timestamp_min_artificial",
            cols_to_aggregate=agg_cols,
            freq="1h",
        )
        agg_df.to_csv(output_dir / f"house_{house}_agg_1h.csv")

    print(f"Aggregated data saved to {output_dir}")
