"""Clean the campus charging data, engineer features, and write EDA summaries.

Run from any directory with:
    python data_pipeline.py

By default this reads ev_campus_dataset.csv beside this file and writes
cleaned_ev_dataset.csv plus docs/eda_summary_by_charger_peak.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "ev_campus_dataset.csv"
DEFAULT_OUTPUT = ROOT / "cleaned_ev_dataset.csv"
DEFAULT_SUMMARY = ROOT / "docs" / "eda_summary_by_charger_peak.csv"

REQUIRED_COLUMNS = {
    "Session_ID",
    "Charger_Type",
    "Energy_kWh",
    "Duration_Mins",
    "Grid_Status_Label",
}
NUMERIC_CANDIDATES = [
    "Start_Hour",
    "Duration_Mins",
    "Energy_kWh",
    "Overstay_Mins",
    "Battery_Temp_C",
    "Ambient_Temp_C",
    "Peak_Hour_Flag",
]
TARGET_ALIASES = {"Grid Heavy": "Grid_Surge"}


def _load_and_clean(input_path: Path) -> tuple[pd.DataFrame, dict[str, int], str]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset was not found: {input_path}")

    data = pd.read_csv(input_path)
    data.columns = [str(column).strip() for column in data.columns]
    if data.columns.duplicated().any():
        duplicates = data.columns[data.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names after trimming whitespace: {duplicates}")

    missing_columns = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    temperature_column = next(
        (name for name in ("Ambient_Temp_C", "Battery_Temp_C") if name in data.columns),
        None,
    )
    if temperature_column is None:
        raise ValueError(
            "Dataset needs Ambient_Temp_C or Battery_Temp_C to produce the temperature summary."
        )

    # Keep a before-cleaning snapshot for the run report.
    missing_before = int(data.isna().sum().sum())
    rows_before = len(data)

    # A missing identifier or target cannot be safely inferred; discard those rows.
    no_id = int(data["Session_ID"].isna().sum())
    if no_id:
        data = data.dropna(subset=["Session_ID"])
    data["Session_ID"] = data["Session_ID"].astype(str).str.strip()
    blank_id = data["Session_ID"].eq("")
    blank_id_count = int(blank_id.sum())
    if blank_id_count:
        data = data.loc[~blank_id].copy()

    data["Grid_Status_Label"] = data["Grid_Status_Label"].astype("string").str.strip()
    data["Grid_Status_Label"] = data["Grid_Status_Label"].replace(TARGET_ALIASES)
    no_target = int(data["Grid_Status_Label"].isna().sum())
    if no_target:
        data = data.dropna(subset=["Grid_Status_Label"])

    duplicate_sessions = int(data["Session_ID"].duplicated(keep="first").sum())
    if duplicate_sessions:
        data = data.drop_duplicates(subset=["Session_ID"], keep="first")
    data = data.reset_index(drop=True)

    # Parse known numeric fields even if a malformed value was stored as text.
    for column in NUMERIC_CANDIDATES:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    # If the peak flag is missing from the source, create it from the documented
    # 12:00–16:59 peak window. The supplied raw file has Start_Hour but no flag.
    flag_was_supplied = "Peak_Hour_Flag" in data.columns
    if not flag_was_supplied and "Start_Hour" not in data.columns:
        raise ValueError("Peak_Hour_Flag is absent and Start_Hour is unavailable to derive it.")

    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    imputed_counts: dict[str, int] = {}
    for column in numeric_columns:
        null_count = int(data[column].isna().sum())
        if not null_count:
            continue
        median = data[column].median()
        if pd.isna(median):
            raise ValueError(f"Cannot median-impute {column}: every value is missing.")
        data[column] = data[column].fillna(median)
        imputed_counts[column] = null_count

    # Fill missing categorical predictors with their mode. Rows without a target
    # were dropped above because imputing the answer would corrupt model labels.
    categorical_columns = data.select_dtypes(include=["object", "string", "category"]).columns
    for column in categorical_columns:
        if column in {"Session_ID", "Grid_Status_Label"}:
            continue
        if data[column].isna().any():
            modes = data[column].mode(dropna=True)
            if modes.empty:
                raise ValueError(f"Cannot fill {column}: every value is missing.")
            count = int(data[column].isna().sum())
            data[column] = data[column].fillna(modes.iloc[0])
            imputed_counts[column] = count

    # The project owner clarified that the source's Battery_Temp_C values
    # represent ambient temperature at the charging location. Keep the raw file
    # untouched, and use the assignment's Ambient_Temp_C name in the clean output.
    renamed_source_battery_temp = temperature_column == "Battery_Temp_C"
    if renamed_source_battery_temp:
        data = data.rename(columns={"Battery_Temp_C": "Ambient_Temp_C"})
        temperature_column = "Ambient_Temp_C"

    # Derive the flag from Start_Hour when possible so it follows one consistent
    # definition. If Start_Hour is absent, preserve and validate a supplied flag.
    if "Start_Hour" in data.columns:
        invalid_hours = ~data["Start_Hour"].between(0, 23)
        if invalid_hours.any():
            raise ValueError("Start_Hour must be between 0 and 23 after imputation.")
        data["Peak_Hour_Flag"] = data["Start_Hour"].between(12, 16).astype("int64")
    else:
        data["Peak_Hour_Flag"] = data["Peak_Hour_Flag"].round().astype("int64")
        if not data["Peak_Hour_Flag"].isin([0, 1]).all():
            raise ValueError("Peak_Hour_Flag must contain only 0 or 1.")

    if (data["Duration_Mins"] <= 0).any():
        raise ValueError("Duration_Mins must be positive to calculate Power_kW.")
    if (data["Energy_kWh"] < 0).any():
        raise ValueError("Energy_kWh cannot be negative.")

    # Charging power in kW and rate bands. The bands describe power rate, not
    # measured battery or charger energy-conversion efficiency.
    data["Power_kW"] = (data["Energy_kWh"] / (data["Duration_Mins"] / 60)).round(3)
    data["Charging_Efficiency_Group"] = pd.cut(
        data["Power_kW"],
        bins=[-np.inf, 7.4, 22.0, np.inf],
        labels=["Low_Rate", "Medium_Rate", "High_Rate"],
        include_lowest=True,
    ).astype("string")

    if data.empty:
        raise ValueError("No usable rows remain after removing rows without IDs or target labels.")
    if data.isna().any().any():
        remaining = data.columns[data.isna().any()].tolist()
        raise ValueError(f"Missing values remain after cleaning: {remaining}")

    run_stats = {
        "input_rows": rows_before,
        "output_rows": len(data),
        "missing_cells_at_input": missing_before,
        "removed_missing_id_rows": no_id + blank_id_count,
        "removed_missing_target_rows": no_target,
        "removed_duplicate_sessions": duplicate_sessions,
        "renamed_source_Battery_Temp_C_to_Ambient_Temp_C": int(renamed_source_battery_temp),
    }
    run_stats.update({f"imputed_{key}": value for key, value in imputed_counts.items()})
    return data, run_stats, temperature_column


def build_summary(data: pd.DataFrame, temperature_column: str) -> pd.DataFrame:
    """Average energy and recorded temperature by charger and peak period."""
    return (
        data.groupby(["Charger_Type", "Peak_Hour_Flag"], as_index=False, observed=True)
        .agg(
            Session_Count=("Session_ID", "count"),
            Average_Energy_kWh=("Energy_kWh", "mean"),
            **{f"Average_{temperature_column}": (temperature_column, "mean")},
        )
        .sort_values(["Charger_Type", "Peak_Hour_Flag"])
        .reset_index(drop=True)
    )


def run_pipeline(input_path: Path, output_path: Path, summary_path: Path) -> None:
    cleaned, stats, temperature_column = _load_and_clean(input_path)
    summary = build_summary(cleaned, temperature_column)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Data pipeline completed.")
    print(f"Input: {input_path}")
    print(f"Cleaned output: {output_path}")
    print(f"Group summary: {summary_path}")
    print(f"Temperature summarized from: {temperature_column}")
    for name, value in stats.items():
        print(f"{name}: {value}")
    print("\nAverage energy and temperature by charger type and peak flag:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))

    if stats["renamed_source_Battery_Temp_C_to_Ambient_Temp_C"]:
        print(
            "\nNote: source Battery_Temp_C readings are exported as Ambient_Temp_C "
            "per the project owner's clarification that they measure charging-location temperature."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Raw input CSV path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Cleaned output CSV path")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="Group summary CSV path")
    args = parser.parse_args()
    run_pipeline(args.input.resolve(), args.output.resolve(), args.summary.resolve())


if __name__ == "__main__":
    main()
