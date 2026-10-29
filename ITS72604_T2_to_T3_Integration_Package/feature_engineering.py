#JIAO JIE 0395413
"""Shared, deterministic feature engineering for the supplied *team-cleaned* CSV.

This does not modify the group's original file. A power-rate category is not a
measured thermodynamic charging efficiency; it is an explicitly labeled proxy.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# 1. Define required dataset columns and target classes
REQUIRED_COLUMNS = {
    "Session_ID", "Charger_Type", "User_Type", "Start_Hour",
    "Duration_Mins", "Energy_kWh", "Overstay_Mins",
    "Grid_Status_Label", "Peak_Hour_Flag"
}
TEMPERATURE_COLUMNS = {"Ambient_Temp_C", "Battery_Temp_C"}
CATEGORIES = {"Normal", "Grid_Surge", "Overheated"}

# 2. Validate the cleaned dataset before model training
def validate_group_data(df: pd.DataFrame) -> None:
    """Check required columns, data quality, and valid grid status labels."""

    # Check missing columns and empty records
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if not TEMPERATURE_COLUMNS.intersection(df.columns):
        missing.append("Ambient_Temp_C or Battery_Temp_C")
    if missing:
        raise ValueError(f"Team-cleaned CSV is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Dataset is empty")

    # Check missing values and duplicate sessions
    checked_columns = list(REQUIRED_COLUMNS | (TEMPERATURE_COLUMNS & set(df.columns)))
    if df[checked_columns].isna().any().any():
        raise ValueError("Team-cleaned dataset contains missing values")
    if df["Session_ID"].duplicated().any():
        raise ValueError("Duplicate Session_ID found")

    # Validate numerical values and charging start time
    if (df["Duration_Mins"] <= 0).any():
        raise ValueError("Duration_Mins must be positive")
    if (df["Energy_kWh"] < 0).any():
        raise ValueError("Energy_kWh cannot be negative")
    if not df["Start_Hour"].between(0, 23).all():
        raise ValueError("Start_Hour must be in 0..23")

    # Validate target classification labels
    wrong = set(df["Grid_Status_Label"]) - CATEGORIES
    if wrong:
        raise ValueError(f"Unexpected grid status labels: {sorted(wrong)}")

    # Verify peak-hour flags against the defined time range
    expected_peak = df["Start_Hour"].between(12, 16).astype(int)
    if not df["Peak_Hour_Flag"].astype(int).eq(expected_peak).all():
        raise ValueError("Peak_Hour_Flag conflicts with the 12:00–17:00 definition")

# 3. Generate engineered features for machine learning
def add_engineered_features(df: pd.DataFrame, *, validate: bool = True) -> pd.DataFrame:
    """Return an enriched copy; never overwrite source CSV or change existing records."""
    """Create charging power and rate categories without modifying source data."""
    if validate:
        validate_group_data(df)

     # Work on a copy to preserve the original dataset
    enriched = df.copy()

    # The task 1 clean file uses Ambient_Temp_C; the already-trained sklearn
    # pipeline was fit with the historical feature name Battery_Temp_C.
    if "Battery_Temp_C" not in enriched.columns and "Ambient_Temp_C" in enriched.columns:
        enriched["Battery_Temp_C"] = enriched["Ambient_Temp_C"]

    # Calculate average charging power in kilowatts
    power = (enriched["Energy_kWh"] * 60 / enriched["Duration_Mins"]).round(3)
    enriched["Power_kW"] = power

    # Categorize charging sessions by power rate
    # This is a power-rate proxy, not measured energy efficiency
    enriched["Charging_Efficiency_Group"] = pd.cut(
        power,
        bins=[-np.inf, 7.4, 22.0, np.inf],
        labels=["Low_Rate", "Medium_Rate", "High_Rate"],
        include_lowest=True,
    ).astype(str)
    return enriched

# 4. Prepare input features for real-time model prediction
def single_prediction_features(*, charger_type: str, user_type: str,
                               start_hour: int, duration_mins: float,
                               energy_kwh: float, ambient_temp_c: float,
                               overstay_mins: int) -> pd.DataFrame:
    """Construct an ambient-temperature row using the saved model's feature contract."""
    hour = int(start_hour)

    # Prevent invalid charging duration
    if duration_mins <= 0:
        raise ValueError("Duration must be positive")

     # Construct a single input record from Streamlit user inputs
    row = pd.DataFrame([{
        "Session_ID": "LIVE-PREDICTION",
        "Charger_Type": charger_type,
        "User_Type": user_type,
        "Start_Hour": hour,
        "Duration_Mins": duration_mins,
        "Energy_kWh": energy_kwh,
        "Overstay_Mins": overstay_mins,
        # Keep the saved model's input feature name; the app presents this as
        # charging-location ambient temperature per the project data definition.
        "Battery_Temp_C": ambient_temp_c,
        "Peak_Hour_Flag": int(12 <= hour < 17),
    }])

    # Apply the same feature engineering used during model training
    return add_engineered_features(row, validate=False)
