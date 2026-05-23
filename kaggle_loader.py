"""
kaggle_loader.py
----------------
Downloads the UCI Individual Household Electric Power Consumption dataset
from Kaggle, resamples it to hourly kWh, fetches real historical temperatures
from Open-Meteo, and saves it in the project's standard CSV format.

Dataset: https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set
Location used for temperature: Paris, France (where the UCI data was collected)

Requirements:
    pip install kaggle
    Place kaggle.json in ~/.kaggle/  (from https://www.kaggle.com/settings → API)
"""

import os
import zipfile
import requests
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paris coordinates (UCI dataset location) ─────────────────────────────────
LAT, LON = 48.8566, 2.3522

KAGGLE_DATASET = "uciml/electric-power-consumption-data-set"
KAGGLE_FILE    = "household_power_consumption.txt"


def _download_kaggle(dest_dir: Path) -> Path:
    """Download dataset zip via Kaggle API and extract."""
    try:
        from kaggle import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except ImportError:
        raise ImportError(
            "kaggle package not installed. Run: pip install kaggle\n"
            "Then place kaggle.json in ~/.kaggle/"
        )

    txt_path = dest_dir / KAGGLE_FILE
    if not txt_path.exists():
        print("Downloading UCI dataset from Kaggle ...")
        api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(dest_dir),
            unzip=True,
            quiet=False,
        )
        print(f"Extracted -> {txt_path}")

    return txt_path


def _load_uci(txt_path: Path) -> pd.DataFrame:
    """Parse UCI file → hourly kWh DataFrame."""
    print("Parsing UCI power consumption file ...")
    df = pd.read_csv(
        txt_path,
        sep=";",
        parse_dates={"datetime": ["Date", "Time"]},
        dayfirst=True,
        na_values=["?"],
        low_memory=False,
    )
    df = df.dropna(subset=["Global_active_power"])
    df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")
    df = df.dropna(subset=["Global_active_power"])

    # Convert watts → kWh per minute, then resample to hourly sum
    df = df.set_index("datetime")
    df["kwh_per_min"] = df["Global_active_power"] / 60.0  # kW * (1/60 h) = kWh
    hourly = df["kwh_per_min"].resample("h").sum().reset_index()
    hourly.columns = ["datetime", "consumption"]
    hourly = hourly.dropna()
    hourly = hourly[hourly["consumption"] > 0]
    print(f"   -> {len(hourly):,} hourly rows after resampling")
    return hourly


def _fetch_temperature(hourly: pd.DataFrame) -> pd.DataFrame:
    """Fetch historical hourly temperature from Open-Meteo for each date range."""
    start = hourly["datetime"].min().date()
    end   = hourly["datetime"].max().date()
    print(f"Fetching historical temperature {start} -> {end} from Open-Meteo ...")

    # Open-Meteo archive API (free, no key)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":        LAT,
        "longitude":       LON,
        "start_date":      str(start),
        "end_date":        str(end),
        "hourly":          "temperature_2m",
        "timezone":        "Europe/Paris",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    temp_df = pd.DataFrame({
        "datetime":    pd.to_datetime(data["hourly"]["time"]),
        "temperature": data["hourly"]["temperature_2m"],
    })
    temp_df["temperature"] = pd.to_numeric(temp_df["temperature"], errors="coerce")

    # Merge on nearest hour
    hourly["datetime"] = pd.to_datetime(hourly["datetime"]).dt.floor("h")
    merged = hourly.merge(temp_df, on="datetime", how="left")
    merged["temperature"] = merged["temperature"].ffill().bfill().fillna(15.0)

    # Validate no NaN temperatures
    nan_temp = merged["temperature"].isna().sum()
    if nan_temp > 0:
        merged["temperature"] = merged["temperature"].fillna(15.0)

    print(f"   -> Temperature merged, {nan_temp} gaps filled with 15 deg C")
    return merged


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time features and prev_consumption to match project schema."""
    df = df.sort_values("datetime").reset_index(drop=True)
    df["hour"]             = df["datetime"].dt.hour
    df["day_of_week"]      = df["datetime"].dt.dayofweek
    df["month"]            = df["datetime"].dt.month
    df["day_of_year"]      = df["datetime"].dt.dayofyear
    df["prev_consumption"] = df["consumption"].shift(1).bfill()
    return df[[
        "datetime", "hour", "day_of_week", "month", "day_of_year",
        "temperature", "prev_consumption", "consumption",
    ]]


def load_kaggle_dataset(save_path: str) -> pd.DataFrame:
    """
    Full pipeline: download → parse → temperature → features → save CSV.
    Returns the final DataFrame.
    """
    dest_dir = Path(save_path).parent / "_kaggle_cache"
    dest_dir.mkdir(exist_ok=True)

    txt_path = _download_kaggle(dest_dir)
    hourly   = _load_uci(txt_path)
    merged   = _fetch_temperature(hourly)
    final    = _build_features(merged)

    final.to_csv(save_path, index=False)
    print(f"Real dataset saved -> {save_path}  ({len(final):,} rows)")
    return final


if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).parent / "electricity_data.csv"
    load_kaggle_dataset(str(out))
