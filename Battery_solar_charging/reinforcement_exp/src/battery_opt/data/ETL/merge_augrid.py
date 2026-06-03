from __future__ import annotations

from pathlib import Path

import pandas as pd


def merge_augrid_with_tariff(
    augrid_csv: Path | str,
    tariff_csv: Path | str,
    output_csv: Path | str,
) -> None:
    """Merge Ausgrid hourly data with hour-of-day tariff rates and save.

    Args:
        augrid_csv: Path to the processed Ausgrid hourly CSV.
        tariff_csv: Path to the simulated Australian tariff CSV
            (columns: hour, import_tariff, export_tariff).
        output_csv: Destination path for the merged output.
    """
    augrid_df = pd.read_csv(augrid_csv, parse_dates=["timestamp"])
    tariff_df = pd.read_csv(tariff_csv)

    augrid_df["hour"] = augrid_df["timestamp"].dt.hour
    merged = augrid_df.merge(tariff_df, on="hour", how="left")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Merged Ausgrid data saved to {output_path}")
