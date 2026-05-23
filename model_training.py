"""
model_training.py
-----------------
Handles all ML tasks:
  • Data loading / preprocessing
  • Feature engineering
  • Model training  (Linear Regression & Random Forest)
  • Model evaluation (MAE, MSE, R²)
  • Automatic best-model selection
  • Persisting models with joblib
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from generate_dataset import generate_electricity_dataset
try:
    from kaggle_loader import load_kaggle_dataset
    _KAGGLE_AVAILABLE = True
except ImportError:
    _KAGGLE_AVAILABLE = False

# ── Paths ────────────────────────────────────────────────────────────────────
from pathlib import Path
_BASE        = Path(__file__).parent
DATA_PATH    = str(_BASE / "electricity_data.csv")
MODEL_DIR    = str(_BASE / "models")
LR_PATH      = str(_BASE / "models" / "linear_regression.pkl")
RF_PATH      = str(_BASE / "models" / "random_forest.pkl")
SCALER_PATH  = str(_BASE / "models" / "scaler.pkl")
BEST_PATH    = str(_BASE / "models" / "best_model.pkl")
METRICS_PATH = str(_BASE / "models" / "metrics.pkl")

# ── Feature columns used for training ────────────────────────────────────────
FEATURE_COLS = [
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "temperature",
    "prev_consumption",
]
TARGET_COL = "consumption"


# ────────────────────────────────────────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ────────────────────────────────────────────────────────────────────────────
def load_and_preprocess(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the CSV, parse datetime, fill missing values, and return clean df.
    """
    # Resolve and validate path stays within project base directory
    resolved = Path(path).resolve()
    if not str(resolved).startswith(str(_BASE.resolve())):
        raise ValueError(f"Invalid data path: {path}")

    # Generate fresh data if file is missing
    if not os.path.exists(resolved):
        if _KAGGLE_AVAILABLE:
            try:
                print("📊  Dataset not found – loading real Kaggle dataset …")
                load_kaggle_dataset(save_path=str(resolved))
            except Exception as e:
                print(f"⚠️  Kaggle load failed ({e}), falling back to synthetic data …")
                generate_electricity_dataset(save_path=str(resolved))
        else:
            print("📊  Dataset not found – generating synthetic data …")
            generate_electricity_dataset(save_path=str(resolved))

    # Detect delimiter automatically for CSV/TSV data.
    with open(resolved, "r", encoding="utf-8") as f:
        first_line = f.readline()
    sep = "\t" if "\t" in first_line else ","

    df = pd.read_csv(resolved, sep=sep, parse_dates=["datetime"], encoding="utf-8")

    # Ensure datetime column is real datetime dtype, with day-first parsing for your dataset.
    df["datetime"] = pd.to_datetime(
        df["datetime"],
        dayfirst=True,
        errors="coerce",
    )
    if df["datetime"].isna().any():
        raise ValueError(
            f"Unable to parse some datetime values in {resolved}. "
            f"Check the format and ensure consistent day-first dates."
        )

    # ── Missing-value imputation (forward-fill then median) ──────────────────
    df = df.ffill()
    for col in FEATURE_COLS + [TARGET_COL]:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # ── Derived features ─────────────────────────────────────────────────────
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"]     = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]     = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"]    = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]    = np.cos(2 * np.pi * df["month"] / 12)

    return df


# ────────────────────────────────────────────────────────────────────────────
# 2. PREPARE FEATURES / TARGET
# ────────────────────────────────────────────────────────────────────────────
EXTENDED_FEATURES = FEATURE_COLS + [
    "is_weekend", "hour_sin", "hour_cos", "month_sin", "month_cos"
]

def prepare_xy(df: pd.DataFrame):
    # Ensure derived features exist when training from raw or already-loaded data.
    if "is_weekend" not in df.columns:
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    if "hour_sin" not in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    if "hour_cos" not in df.columns:
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    if "month_sin" not in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    if "month_cos" not in df.columns:
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    X = df[EXTENDED_FEATURES].values
    y = df[TARGET_COL].values
    return X, y


# ────────────────────────────────────────────────────────────────────────────
# 3. TRAIN MODELS
# ────────────────────────────────────────────────────────────────────────────
def train_models(df: pd.DataFrame):
    """
    Train Linear Regression and Random Forest, evaluate both,
    persist all artefacts, and return a metrics dictionary.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y = prepare_xy(df)

    # ── Train / test split (80 / 20, chronological) ──────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # ── Feature scaling (important for Linear Regression) ────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)

    # ── 3a. Linear Regression ────────────────────────────────────────────────
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    y_pred_lr = lr.predict(X_test_s)
    joblib.dump(lr, LR_PATH)

    lr_metrics = {
        "MAE":  round(mean_absolute_error(y_test, y_pred_lr), 4),
        "MSE":  round(mean_squared_error(y_test, y_pred_lr), 4),
        "R2":   round(r2_score(y_test, y_pred_lr), 4),
        "predictions": y_pred_lr,
        "actuals":     y_test,
    }

    # ── 3b. Random Forest Regressor ──────────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)           # RF doesn't need scaling
    y_pred_rf = rf.predict(X_test)
    joblib.dump(rf, RF_PATH)

    rf_metrics = {
        "MAE":  round(mean_absolute_error(y_test, y_pred_rf), 4),
        "MSE":  round(mean_squared_error(y_test, y_pred_rf), 4),
        "R2":   round(r2_score(y_test, y_pred_rf), 4),
        "predictions": y_pred_rf,
        "actuals":     y_test,
    }

    # ── 3c. Automatic best-model selection (highest R²) ──────────────────────
    if rf_metrics["R2"] >= lr_metrics["R2"]:
        best_name  = "Random Forest"
        best_model = rf
        best_needs_scaling = False
    else:
        best_name  = "Linear Regression"
        best_model = lr
        best_needs_scaling = True

    joblib.dump(
        {"model": best_model, "name": best_name, "needs_scaling": best_needs_scaling},
        BEST_PATH,
    )

    metrics = {
        "Linear Regression": lr_metrics,
        "Random Forest":     rf_metrics,
        "best_model_name":   best_name,
        "test_dates":        df["datetime"].iloc[-len(y_test):].values,
    }
    joblib.dump(metrics, METRICS_PATH)
    invalidate_model_cache()

    print(f"\n{'='*50}")
    print(f"  Linear Regression -> MAE={lr_metrics['MAE']}  R2={lr_metrics['R2']}")
    print(f"  Random Forest     -> MAE={rf_metrics['MAE']}  R2={rf_metrics['R2']}")
    print(f"  Best model: {best_name}")
    print(f"{'='*50}\n")

    return metrics


# ────────────────────────────────────────────────────────────────────────────
# 4. PREDICT (single sample)
# ────────────────────────────────────────────────────────────────────────────
# ── Model cache container (avoids bare globals) ──────────────────────────────
class _ModelCache:
    bundle: dict | None = None
    scaler = None

_cache = _ModelCache()

def _load_bundle():
    if _cache.bundle is None:
        _cache.bundle = joblib.load(BEST_PATH)
        if _cache.bundle["needs_scaling"]:
            _cache.scaler = joblib.load(SCALER_PATH)

def invalidate_model_cache():
    _cache.bundle = None
    _cache.scaler = None


def predict_consumption(
    hour: int,
    day_of_week: int,
    month: int,
    day_of_year: int,
    temperature: float,
    prev_consumption: float,
) -> float:
    """
    Run inference with the persisted best model.
    Returns predicted electricity consumption in kWh.
    """
    _load_bundle()

    is_weekend = int(day_of_week >= 5)
    hour_sin   = np.sin(2 * np.pi * hour / 24)
    hour_cos   = np.cos(2 * np.pi * hour / 24)
    month_sin  = np.sin(2 * np.pi * month / 12)
    month_cos  = np.cos(2 * np.pi * month / 12)

    sample = np.array([[
        hour, day_of_week, month, day_of_year,
        temperature, prev_consumption,
        is_weekend, hour_sin, hour_cos, month_sin, month_cos,
    ]])

    if _cache.bundle["needs_scaling"]:
        sample = _cache.scaler.transform(sample)

    return float(_cache.bundle["model"].predict(sample)[0])


# ────────────────────────────────────────────────────────────────────────────
# 5. LOAD SAVED METRICS
# ────────────────────────────────────────────────────────────────────────────
def load_metrics():
    if os.path.exists(METRICS_PATH):
        return joblib.load(METRICS_PATH)
    return None


# ────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_and_preprocess()
    train_models(df)
