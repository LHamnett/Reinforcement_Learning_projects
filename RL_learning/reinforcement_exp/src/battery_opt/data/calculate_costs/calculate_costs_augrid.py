"""Cost calculation utilities for the Ausgrid solar home dataset.

Computes per-hour electricity costs from the merged Ausgrid dataset.

Ausgrid customers use gross metering: all solar generation (GG) is exported
to the grid and billed separately from household consumption.  There is no
on-site solar offset — import and export are measured independently.

Cost model:
    gc_cost     = gc_kwh  × import_tariff   (general consumption)
    cl_cost     = cl_kwh  × import_tariff   (controlled load, runs off-peak)
    gg_revenue  = gg_kwh  × export_tariff   (solar feed-in)
    import_cost = gc_cost + cl_cost
    net_cost    = import_cost − gg_revenue
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def calculate_augrid_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Add cost columns to a merged Ausgrid DataFrame.

    Args:
        df: DataFrame with columns: gc_kwh, cl_kwh, gg_kwh,
            import_tariff, export_tariff.

    Returns:
        Copy of ``df`` with additional columns:
        - ``gc_cost``: cost of general consumption (AUD)
        - ``cl_cost``: cost of controlled-load consumption (AUD)
        - ``import_cost``: total grid import cost (AUD)
    """
    out = df.copy()
    out["total_consumption"] = out["gc_kwh"] + out["cl_kwh"]
    out["cons_min_generation"] = out["total_consumption"] - out["gg_kwh"]

    out["import_cost_no_solar"] = out["total_consumption"] * out["import_tariff"]
    out["import_cost_solar_used_immediate"] = (
        (out["cons_min_generation"]).clip(lower=0) * out["import_tariff"]
    )
    
    # out["export_profit"] = out.apply(lambda x: abs(x["cons_min_generation"]) * x["export_tariff"] if x["cons_min_generation"] < 0 else 0, axis=1)

    # out["total_cost"] = out["import_cost"] - out["export_profit"]
    

    return out


def process_and_save_augrid_costs(
    combined_csv: Path | str,
    output_csv: Path | str,
) -> None:
    """Compute costs for the merged Ausgrid dataset and save.

    Args:
        combined_csv: Path to ``augrid_combined.csv`` (output of
            :func:`~battery_opt.data.merge_augrid.merge_augrid_with_tariff`).
        output_csv: Destination path for the cost-annotated output.
    """
    df = pd.read_csv(combined_csv, parse_dates=["timestamp"])
    df = calculate_augrid_costs(df)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Augrid costs saved to {output_path}")

if __name__ == '__main__':
    combined_csv = Path("data/combined/augrid_combined.csv")
    output_csv = Path("data/combined/augrid_costs.csv")
    process_and_save_augrid_costs(combined_csv, output_csv)
    print('Done calculating Ausgrid costs.')
