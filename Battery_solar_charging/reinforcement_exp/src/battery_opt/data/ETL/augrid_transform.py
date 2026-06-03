"""Data loading and transformation utilities for the Ausgrid solar home dataset.

Handles the three annual Ausgrid CSVs (2010–2013): half-hourly GC/CL/GG readings
for 300 solar customers, aggregated to hourly and pivoted to wide format.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# Ausgrid half-hour period labels in their on-disk order.
# Each label is the *end* time of that period: "0:30" = 00:00–00:30, …,
# "0:00" (last column) = 23:30–00:00.
_AUGRID_PERIOD_COLS: list[str] = [
    "0:30", "1:00", "1:30", "2:00", "2:30", "3:00", "3:30", "4:00",
    "4:30", "5:00", "5:30", "6:00", "6:30", "7:00", "7:30", "8:00",
    "8:30", "9:00", "9:30", "10:00", "10:30", "11:00", "11:30", "12:00",
    "12:30", "13:00", "13:30", "14:00", "14:30", "15:00", "15:30", "16:00",
    "16:30", "17:00", "17:30", "18:00", "18:30", "19:00", "19:30", "20:00",
    "20:30", "21:00", "21:30", "22:00", "22:30", "23:00", "23:30", "0:00",
]

# Period-start offset (minutes from midnight) for each period-end label.
_AUGRID_PERIOD_START_MINUTES: dict[str, int] = {
    col: i * 30 for i, col in enumerate(_AUGRID_PERIOD_COLS)
}


def load_augrid_csv(path: Path | str) -> pd.DataFrame:
    """Parse an Ausgrid solar home CSV, skipping the first metadata row.

    Each file has a plain-text description on row 1 before the column headers.

    Args:
        path: Path to one of the three Ausgrid annual CSV files.

    Returns:
        DataFrame with columns: Customer, Generator Capacity, Postcode,
        Consumption Category, date, and 48 half-hour period columns.
        An optional Row Quality column (column 54) is retained if present.
    """
    return pd.read_csv(path, skiprows=1, low_memory=False)


def melt_augrid_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convert an Ausgrid wide DataFrame to long format with period-start timestamps.

    Period-end labels are mapped to period-start timestamps: "0:30" → 00:00,
    "1:00" → 00:30, …, "0:00" (last column) → 23:30.

    Args:
        df: Raw DataFrame from :func:`load_augrid_csv`.

    Returns:
        Long-format DataFrame with columns: timestamp, customer_id,
        generator_capacity_kwp, postcode, category, kwh.
        One row per (customer, category, half-hour interval).
    """
    id_cols = ["Customer", "Generator Capacity", "Postcode", "Consumption Category", "date"]
    period_cols = [c for c in _AUGRID_PERIOD_COLS if c in df.columns]

    long = df[id_cols + period_cols].melt(
        id_vars=id_cols,
        value_vars=period_cols,
        var_name="period_end",
        value_name="kwh",
    )

    # Files use inconsistent date formats across years ("%d-%b-%y" in 2010-11,
    # "%d/%m/%Y" in 2011-12 and 2012-13).
    dates = pd.to_datetime(long["date"], format="mixed", dayfirst=True)
    offsets = long["period_end"].map(_AUGRID_PERIOD_START_MINUTES)
    long["timestamp"] = dates + pd.to_timedelta(offsets, unit="min")

    return (
        long
        .rename(columns={
            "Customer": "customer_id",
            "Generator Capacity": "generator_capacity_kwp",
            "Postcode": "postcode",
            "Consumption Category": "category",
        })
        [["timestamp", "customer_id", "generator_capacity_kwp", "postcode", "category", "kwh"]]
    )


def aggregate_augrid_hourly(long_df: pd.DataFrame) -> pd.DataFrame:
    """Resample Ausgrid half-hourly long data to hourly and pivot categories to columns.

    Args:
        long_df: Long-format DataFrame from :func:`melt_augrid_to_long`.

    Returns:
        DataFrame with columns: timestamp, customer_id, generator_capacity_kwp,
        postcode, gc_kwh, cl_kwh, gg_kwh.  One row per customer per hour.
    """
    df = long_df.copy()
    df["timestamp"] = df["timestamp"].dt.floor("1h")

    hourly = (
        df.groupby(
            ["timestamp", "customer_id", "generator_capacity_kwp", "postcode", "category"]
        )["kwh"]
        .sum()
        .unstack("category")
        .fillna(0.0)  # customers without a CL meter have no controlled-load rows
        .rename(columns={"GC": "gc_kwh", "CL": "cl_kwh", "GG": "gg_kwh"})
        .reset_index()
    )
    hourly.columns.name = None
    return hourly


def process_and_save_augrid(
    augrid_dir: Path | str,
    output_dir: Path | str,
) -> None:
    """Read all Ausgrid solar home CSVs, aggregate to hourly, and save.

    Reads every ``Solar home *.csv`` file in ``augrid_dir``, concatenates
    across years, converts to hourly resolution, drops any customers with
    incomplete time series, and writes ``augrid_solar_hourly.csv``.

    Args:
        augrid_dir: Directory containing the three Ausgrid annual CSVs.
        output_dir: Destination directory (created if absent).
    """
    augrid_dir = Path(augrid_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    year_files = sorted(augrid_dir.glob("Solar home *.csv"))
    frames = [melt_augrid_to_long(load_augrid_csv(p)) for p in year_files]

    all_long = pd.concat(frames, ignore_index=True)
    hourly = aggregate_augrid_hourly(all_long)

    counts = hourly.groupby("customer_id").size()
    complete = counts[counts == counts.max()].index
    dropped = sorted(counts[counts < counts.max()].index.tolist())
    hourly = hourly[hourly["customer_id"].isin(complete)]

    if dropped:
        print(f"Dropped {len(dropped)} customer(s) with incomplete data: {dropped}")

    out_path = output_dir / "augrid_solar_hourly.csv"
    hourly.to_csv(out_path, index=False)
    print(f"Ausgrid data saved to {out_path}")
