"""
generate_dataset.py
-------------------
Generates a synthetic electricity consumption dataset for training and testing.
The dataset simulates realistic patterns: daily cycles, seasonal trends,
temperature effects, and weekend dips.
"""

import pandas as pd
import numpy as np

def generate_electricity_dataset(n_days=730, save_path="electricity_data.csv"):
    """
    Generate a realistic electricity consumption dataset.
    
    Parameters:
        n_days  : Number of days of data to generate (default: 2 years)
        save_path: File path to save the CSV
    
    Returns:
        df: pandas DataFrame with the generated dataset
    """
    np.random.seed(42)  # For reproducibility

    # ── Date range ──────────────────────────────────────────────────────────
    dates = pd.date_range(start="2022-01-01", periods=n_days * 24, freq="h")

    # ── Time features ────────────────────────────────────────────────────────
    hour        = dates.hour
    day_of_week = dates.dayofweek   # 0=Monday … 6=Sunday
    month       = dates.month
    day_of_year = dates.dayofyear

    # ── Synthetic temperature (°C) ──────────────────────────────────────────
    # Seasonal sinusoidal curve + daily swing + noise
    seasonal_temp = 20 + 15 * np.sin(2 * np.pi * (day_of_year / 365) - np.pi / 2)
    daily_temp    = 5  * np.sin(2 * np.pi * (hour / 24) - np.pi / 3)
    temperature   = seasonal_temp + daily_temp + np.random.normal(0, 2, len(dates))

    # ── Base consumption (kWh) ───────────────────────────────────────────────
    # Hourly pattern: peak morning + evening, low night
    hour_pattern = (
        50
        + 30 * np.sin(2 * np.pi * (hour - 6) / 24)
        + 20 * np.sin(2 * np.pi * (hour - 18) / 12)
    )

    # Weekend reduction (~20 % less)
    weekend_factor = np.where(day_of_week >= 5, 0.80, 1.0)

    # Seasonal effect: higher consumption in summer & winter
    seasonal_effect = 1.0 + 0.25 * np.abs(np.sin(2 * np.pi * (day_of_year / 365)))

    # Temperature-driven HVAC load
    temp_effect = 0.5 * np.abs(temperature - 18)   # comfort setpoint ~18 °C

    # Combine all effects
    consumption = (
        hour_pattern * weekend_factor * seasonal_effect
        + temp_effect
        + np.random.normal(0, 5, len(dates))          # random noise
    )
    consumption = np.clip(consumption, 10, None)       # never below 10 kWh

    # ── Previous-hour consumption feature ───────────────────────────────────
    prev_consumption = np.roll(consumption, 1)
    prev_consumption[0] = consumption[0]

    # ── Assemble DataFrame ───────────────────────────────────────────────────
    df = pd.DataFrame({
        "datetime":          dates,
        "hour":              hour,
        "day_of_week":       day_of_week,
        "month":             month,
        "day_of_year":       day_of_year,
        "temperature":       np.round(temperature, 2),
        "prev_consumption":  np.round(prev_consumption, 2),
        "consumption":       np.round(consumption, 2),
    })

    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"✅  Dataset saved → {save_path}  ({len(df):,} rows)")
    return df


# Run standalone
if __name__ == "__main__":
    generate_electricity_dataset()
